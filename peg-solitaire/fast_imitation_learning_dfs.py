"""
fast_imitation_learning_dtfs.py — DAgger for SquareBoard peg solitaire using fast_dfs as teacher.

Differences from fast_imitation_learning.py
--------------------------------------------
1. fast_dfs replaces fast_mcts as the labeling oracle:
     - Depth-limited DFS with transposition table and move ordering
     - JAX JIT-compiled move-validity check (warmed up inside fast_dfs)
     - Single-threaded per call

2. Parallelism lives at the labeling level, not inside the oracle. fast_dfs
   is single-threaded; instead the states of a trajectory are labeled
   concurrently with a thread pool (n_workers threads). Each fast_dfs call
   works on its own arrays/transposition table, and the JIT'd move check
   releases the GIL, so calls overlap.

3. mcts_time_limit → dfs_max_depth (and optional dfs_q for quiescence).

4. reward_mode is not applicable (DFS always minimises pegs remaining).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import keras

from board import SquareBoard, CrossBoard
from fast_dfs import fast_dfs
from policy_network_square import select_action


# ── JSONL progress logging ────────────────────────────────────────────────────

def _record(log_path: str | None, run_start: float, **fields) -> None:
    if log_path is None:
        return
    fields["t"] = round(time.perf_counter() - run_start, 3)
    os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
    with open(log_path, "a") as f:
        f.write(json.dumps(fields) + "\n")


# ── Training ──────────────────────────────────────────────────────────────────

def _ensure_jit_compiled(
    model: keras.Model,
    optimizer: keras.optimizers.Optimizer,
) -> None:
    if not getattr(model, '_fast_il_jit_compiled', False):
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            jit_compile=True,
        )
        model._fast_il_jit_compiled = True  # type: ignore[attr-defined]


def learn(
    D: list[tuple[np.ndarray, int]],
    policy_model: keras.Model,
    optimizer: keras.optimizers.Optimizer,
    epochs: int,
    batch_size: int,
    record_fn=None,
    iteration: int | None = None,
) -> keras.Model:
    states  = np.array([s for s, _ in D], dtype=np.float32)  # (N, n, n, 2)
    actions = np.array([a for _, a in D], dtype=np.int32)                      # (N,)

    _ensure_jit_compiled(policy_model, optimizer)

    t0 = time.perf_counter()
    history = policy_model.fit(
        states, actions,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )
    train_time = time.perf_counter() - t0

    losses = history.history["loss"]
    for ep, loss in enumerate(losses, 1):
        print(f"    epoch {ep:>{len(str(epochs))}}/{epochs}  loss={loss:.4f}")
        if record_fn is not None:
            record_fn(type="epoch_loss", iteration=iteration, epoch=ep, loss=round(float(loss), 6))
    print(f"  training: {train_time:.1f}s  "
          f"loss {losses[0]:.4f} → {losses[-1]:.4f}  "
          f"(Δ={losses[-1] - losses[0]:+.4f})")

    return policy_model


# ── Trajectory generation ─────────────────────────────────────────────────────

def _gen_trajectory(
    policy: keras.Model, board: SquareBoard
) -> tuple[list[SquareBoard], SquareBoard]:
    """Roll out policy from board; return (states visited, terminal board)."""
    states: list[SquareBoard] = []
    b = board.copy()
    while b.available_moves():
        states.append(b.copy())
        move = select_action(policy, b, greedy=False)
        b.move(move[0], move[2])
    return states, b


# ── Initial-board set construction ─────────────────────────────────────────────

def _random_descendant(parent: SquareBoard, n_moves: int, rng: random.Random):
    """Play n_moves random legal moves from parent.

    Returns the resulting board, or None if it dead-ends (no legal moves) before
    n_moves have been played.
    """
    b = parent.copy()
    for _ in range(n_moves):
        moves = b.available_moves()
        if not moves:
            return None
        mv = rng.choice(moves)
        b.move(mv[0], mv[2])
    return b


def build_initial_boards(
    base_board: SquareBoard,
    *,
    branching: int = 5,
    moves_per_step: int = 5,
    depth: int = 2,
    seed: int | None = None,
    max_tries_per_child: int = 200,
) -> list[SquareBoard]:
    """Build a set of starting boards by repeated random descent from base_board.

    Level 0 is base_board itself. Every board at level d is expanded into
    `branching` children, each child reached by playing `moves_per_step` random
    legal moves from that parent. The construction runs for `depth` levels.

    All boards produced are de-duplicated by peg configuration, so no two starts
    in the returned set are identical (this is what guarantees "no repeats" both
    among siblings and across all boards generated at a given level). Because
    every board at a level has the same peg count (base − level·moves_per_step),
    a single global `seen` set is enough — boards from different levels can never
    collide.

    Returns base_board plus every descendant: 1 + Σ_{d=1..depth} branching^d boards.
    With the defaults (branching=5, depth=2) that is 1 + 5 + 25 = 31 boards.
    """
    rng = random.Random(seed)
    seen: set[frozenset] = {frozenset(base_board.pegs)}
    all_boards: list[SquareBoard] = [base_board.copy()]
    frontier: list[SquareBoard] = [base_board]

    for level in range(depth):
        next_frontier: list[SquareBoard] = []
        for parent in frontier:
            produced = 0
            tries = 0
            while produced < branching:
                if tries >= max_tries_per_child * branching:
                    raise RuntimeError(
                        f"build_initial_boards: only generated {produced}/{branching} "
                        f"unique descendants at level {level + 1} after {tries} tries. "
                        f"Reduce branching/moves_per_step/depth or use a larger board."
                    )
                tries += 1
                child = _random_descendant(parent, moves_per_step, rng)
                if child is None:
                    continue  # dead-ended before moves_per_step moves
                key = frozenset(child.pegs)
                if key in seen:
                    continue  # duplicate position — try again
                seen.add(key)
                next_frontier.append(child)
                all_boards.append(child)
                produced += 1
        frontier = next_frontier

    return all_boards


# ── DAgger ────────────────────────────────────────────────────────────────────

def _collect_trajectories(
    pi: keras.Model,
    initial_boards: list[SquareBoard],
    n_trajectories: int,
    dfs_max_depth: int,
    dfs_q: int,
    dfs_max_breadth: int | None,
    n_workers: int,
    rng: random.Random,
    record_fn=None,
    iteration: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Generate n_trajectories rollouts, label every state with fast_dfs, return samples.

    Each rollout starts from a board sampled uniformly at random from
    initial_boards (the set built by build_initial_boards). fast_dfs is
    single-threaded; the states within a trajectory are labeled concurrently
    using a pool of n_workers threads.
    """
    samples: list[tuple[np.ndarray, int]] = []

    for t in range(n_trajectories):
        start_board = rng.choice(initial_boards)
        start_pegs = len(start_board.pegs)
        traj_start = time.perf_counter()
        trajectory, final_board = _gen_trajectory(pi, start_board)
        traj_time = time.perf_counter() - traj_start
        pegs_left = int(final_board.encode()[..., 0].sum())

        traj_label = f"trajectory {t + 1}/{n_trajectories}" if n_trajectories > 1 else "trajectory"
        print(f"\n  [{traj_label}]  start {start_pegs} pegs, {len(trajectory)} steps, "
              f"{pegs_left} peg(s) remaining  ({traj_time:.2f}s)")
        if record_fn is not None:
            record_fn(
                type="trajectory",
                iteration=iteration,
                trajectory_idx=t,
                start_pegs=start_pegs,
                steps=len(trajectory),
                pegs_remaining=pegs_left,
                traj_time_s=round(traj_time, 3),
            )

        n_states = len(trajectory)
        labeled = 0
        skipped = 0
        label_start = time.perf_counter()

        def _label(state):
            return state, fast_dfs(
                state, max_depth=dfs_max_depth, q=dfs_q, max_breadth=dfs_max_breadth
            )

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_label, state) for state in trajectory]
            for done, future in enumerate(as_completed(futures), start=1):
                state, move = future.result()
                if move is None:
                    skipped += 1
                else:
                    samples.append((state.encode(), state.encode_move(move)))
                    labeled += 1

                elapsed_label = time.perf_counter() - label_start
                rate = done / elapsed_label if elapsed_label > 0 else 0
                sys.stdout.write(
                    f"\r  labeling: {done:>{len(str(n_states))}}/{n_states} "
                    f"({done / n_states * 100:5.1f}%)  {rate:.2f} states/s  "
                    f"threads={n_workers}     "
                )
                sys.stdout.flush()

        label_time = time.perf_counter() - label_start
        sys.stdout.write("\n")

        print(f"  labeled {labeled}/{n_states} states  "
              f"(skipped {skipped})  in {label_time:.1f}s  "
              f"avg {label_time / n_states:.2f}s/state")
        if record_fn is not None:
            record_fn(
                type="labeling",
                iteration=iteration,
                trajectory_idx=t,
                labeled=labeled,
                skipped=skipped,
                label_time_s=round(label_time, 3),
                rate_states_per_s=round(labeled / label_time if label_time > 0 else 0, 2),
            )

    return samples


def dagger(
    pi0: keras.Model,
    initial_boards: list[SquareBoard],
    optimizer: keras.optimizers.Optimizer,
    n_iterations: int,
    epochs: int,
    batch_size: int,
    dfs_max_depth: int = 8,
    dfs_q: int = 1,
    dfs_max_breadth: int | None = None,
    n_trajectories: int = 1,
    n_initial_trajectories: int = 0,
    max_dataset_size: int | None = None,
    save_path: str | None = "policy_model.keras",
    n_workers: int | None = None,
    log_path: str | None = None,
    sample_seed: int | None = None,
) -> keras.Model:
    """DAgger using fast_dfs as the teacher, for SquareBoard.

    pi0                      — initial policy from build_square_policy_network
    initial_boards           — set of starting boards; each rollout samples one
                               uniformly at random (see build_initial_boards)
    optimizer                — e.g. keras.optimizers.Adam(1e-3)
    n_iterations             — DAgger iterations
    epochs                   — learn() epochs per iteration
    batch_size               — learn() batch size
    dfs_max_depth            — look-ahead depth for fast_dfs
    dfs_q                    — quiescence threshold (extend search when moves <= q)
    dfs_max_breadth          — max children expanded per node; None → expand all
    n_trajectories           — trajectories rolled out per iteration (default 1)
    n_initial_trajectories   — trajectories collected before iteration 1 to seed the dataset
    max_dataset_size         — cap on dataset length; oldest samples evicted first; None → unlimited
    save_path                — save model after each iteration; None disables saving
    n_workers                — number of threads for concurrent state labeling
    log_path                 — JSONL file for progress logging; None disables logging
    sample_seed              — seed for per-trajectory start-board sampling; None → nondeterministic

    Returns the final updated policy.
    """
    n_workers = n_workers or (os.cpu_count() or 1)
    sample_rng = random.Random(sample_seed)

    _ensure_jit_compiled(pi0, optimizer)

    D: list[tuple[np.ndarray, int]] = []
    pi = pi0

    def _append_samples(dataset: list, new: list) -> None:
        dataset.extend(new)
        if max_dataset_size is not None and len(dataset) > max_dataset_size:
            del dataset[:len(dataset) - max_dataset_size]

    run_start = time.perf_counter()

    def record(**fields):
        _record(log_path, run_start, **fields)

    record(
        type="run_start",
        n_iterations=n_iterations,
        n_trajectories=n_trajectories,
        n_initial_trajectories=n_initial_trajectories,
        epochs=epochs,
        batch_size=batch_size,
        dfs_max_depth=dfs_max_depth,
        dfs_q=dfs_q,
        dfs_max_breadth=dfs_max_breadth,
        n_workers=n_workers,
        board_n=initial_boards[0].n,
        n_initial_boards=len(initial_boards),
        dataset_size=0,
    )

    print(f"Using fast_dfs teacher  (max_depth={dfs_max_depth}, q={dfs_q}, "
          f"max_breadth={dfs_max_breadth}, label_threads={n_workers})")

    if n_initial_trajectories > 0:
        print(f"\n{'='*60}")
        print(f"Pre-DAgger data collection: {n_initial_trajectories} initial trajectory/trajectories")
        print(f"{'='*60}")
        initial_samples = _collect_trajectories(
            pi, initial_boards, n_initial_trajectories,
            dfs_max_depth, dfs_q, dfs_max_breadth, n_workers, sample_rng,
            record_fn=record, iteration=0,
        )
        _append_samples(D, initial_samples)
        print(f"\n  collected {len(initial_samples)} initial samples  (dataset now {len(D)})")
        record(type="initial_collection_end", dataset_size=len(D))

    for i in range(n_iterations):
        iter_start = time.perf_counter()
        elapsed_total = iter_start - run_start
        print(f"\n{'='*60}")
        print(f"DAgger iteration {i + 1}/{n_iterations}  "
              f"(elapsed {elapsed_total:.0f}s, dataset {len(D)} samples)")
        print(f"{'='*60}")
        record(type="iteration_start", iteration=i + 1, dataset_size=len(D))

        new_samples_list = _collect_trajectories(
            pi, initial_boards, n_trajectories,
            dfs_max_depth, dfs_q, dfs_max_breadth, n_workers, sample_rng,
            record_fn=record, iteration=i + 1,
        )
        _append_samples(D, new_samples_list)

        print(f"\n  dataset: {len(D)} total  (+{len(new_samples_list)} this iteration)")
        print(f"  --- training ---")
        train_start = time.perf_counter()
        pi = learn(D, pi, optimizer, epochs, batch_size, record_fn=record, iteration=i + 1)
        train_time = time.perf_counter() - train_start

        if save_path:
            pi.save(save_path)
            print(f"  model saved → {save_path}")

        iter_time = time.perf_counter() - iter_start
        print(f"\n  iteration {i+1} complete in {iter_time:.1f}s")
        record(
            type="iteration_end",
            iteration=i + 1,
            dataset_size=len(D),
            new_samples=len(new_samples_list),
            train_time_s=round(train_time, 3),
            iter_time_s=round(iter_time, 3),
        )

    total_time = time.perf_counter() - run_start
    print(f"\n{'='*60}")
    print(f"DAgger complete: {n_iterations} iterations in {total_time:.1f}s  "
          f"({total_time/n_iterations:.1f}s/iter avg)  "
          f"dataset={len(D)}")
    print(f"{'='*60}")
    record(type="run_end", total_time_s=round(total_time, 3), dataset_size=len(D))
    return pi


# ── Entry point ───────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str = "config_dfs.yaml") -> None:
    import argparse
    from policy_network_square import build_square_policy_network

    parser = argparse.ArgumentParser(description="DAgger with fast_dfs teacher for peg solitaire")
    parser.add_argument("--config", default=config_path, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # Board
    bc          = cfg["board"]
    board_type  = bc.get("type", "square").lower()
    empty_start = tuple(bc["empty_start"]) if bc.get("empty_start") else None
    if board_type == "cross":
        # CrossBoard is the classic English 7×7 cross; n is fixed at 7 and
        # only the empty starting hole is configurable (defaults to centre).
        board = CrossBoard(**({"empty_start": empty_start} if empty_start else {}))
    elif board_type == "square":
        board = SquareBoard(bc["n"], empty_start=empty_start)
    else:
        raise ValueError(f"Unknown board type {board_type!r}; expected 'square' or 'cross'")
    n = board.n

    # Initial-board set: the standard board plus random-descent descendants.
    isc = bc.get("init_set") or {}
    initial_boards = build_initial_boards(
        board,
        branching=isc.get("branching", 5),
        moves_per_step=isc.get("moves_per_step", 5),
        depth=isc.get("depth", 2),
        seed=isc.get("seed"),
    )
    print(f"Initial-board set: {len(initial_boards)} boards "
          f"(branching={isc.get('branching', 5)}, "
          f"moves_per_step={isc.get('moves_per_step', 5)}, "
          f"depth={isc.get('depth', 2)})")

    # Network
    nc  = cfg["network"]
    if cfg["dagger"].get("load_path"):
        print(f"Loading model from {cfg['dagger']['load_path']}")
        pi0 = keras.models.load_model(cfg["dagger"]["load_path"])
    else:
        pi0 = build_square_policy_network(
            n,
            res_blocks=nc["res_blocks"],
            filters=nc["filters"],
        )
        pi0.summary()

    # Optimizer
    optimizer = keras.optimizers.Adam(
        learning_rate=cfg["optimizer"]["learning_rate"]
    )

    # DAgger
    dc = cfg["dagger"]

    log_path = None
    log_dir = dc.get("log_dir")
    if log_dir is not None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{timestamp}_n{n}_dfs.jsonl")
        print(f"Progress log → {log_path}")

    dagger(
        pi0=pi0,
        initial_boards=initial_boards,
        optimizer=optimizer,
        n_iterations=dc["n_iterations"],
        epochs=dc["epochs"],
        batch_size=dc["batch_size"],
        dfs_max_depth=dc.get("dfs_max_depth", 8),
        dfs_q=dc.get("dfs_q", 1),
        dfs_max_breadth=dc.get("dfs_max_breadth"),
        n_trajectories=dc.get("n_trajectories", 1),
        n_initial_trajectories=dc.get("n_initial_trajectories", 0),
        max_dataset_size=dc.get("max_dataset_size"),
        save_path=dc.get("save_path"),
        n_workers=dc.get("n_workers"),
        log_path=log_path,
        sample_seed=isc.get("sample_seed"),
    )


if __name__ == "__main__":
    main()

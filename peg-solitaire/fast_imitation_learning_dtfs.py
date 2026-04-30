"""
fast_imitation_learning_dtfs.py — DAgger for SquareBoard peg solitaire using fast_dfs as teacher.

Differences from fast_imitation_learning.py
--------------------------------------------
1. fast_dfs replaces fast_mcts as the labeling oracle:
     - Depth-limited DFS with transposition table and move ordering
     - JAX JIT-compiled move-validity check (warmed up inside fast_dfs)
     - Root moves evaluated in parallel via multiprocessing (n_workers)

2. No outer labeling pool — fast_dfs handles its own internal parallelism,
   so nesting an extra Pool would cause spawn conflicts. States in a trajectory
   are labeled sequentially; fast_dfs parallelises over root moves per call.

3. mcts_time_limit → dfs_max_depth (and optional dfs_q for quiescence).

4. reward_mode is not applicable (DFS always minimises pegs remaining).
"""
from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
import keras

from board import SquareBoard
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
    states  = np.array([s for s, _ in D], dtype=np.float32)[..., np.newaxis]  # (N, n, n, 1)
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


# ── DAgger ────────────────────────────────────────────────────────────────────

def _collect_trajectories(
    pi: keras.Model,
    initial_board: SquareBoard,
    n_trajectories: int,
    dfs_max_depth: int,
    dfs_q: int,
    n_workers: int,
    record_fn=None,
    iteration: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Generate n_trajectories rollouts, label every state with fast_dfs, return samples.

    States are labeled sequentially because fast_dfs already parallelises
    internally over root moves using n_workers processes.
    """
    samples: list[tuple[np.ndarray, int]] = []

    for t in range(n_trajectories):
        traj_start = time.perf_counter()
        trajectory, final_board = _gen_trajectory(pi, initial_board)
        traj_time = time.perf_counter() - traj_start
        pegs_left = int(final_board.encode().sum())

        traj_label = f"trajectory {t + 1}/{n_trajectories}" if n_trajectories > 1 else "trajectory"
        print(f"\n  [{traj_label}]  {len(trajectory)} steps, "
              f"{pegs_left} peg(s) remaining  ({traj_time:.2f}s)")
        if record_fn is not None:
            record_fn(
                type="trajectory",
                iteration=iteration,
                trajectory_idx=t,
                steps=len(trajectory),
                pegs_remaining=pegs_left,
                traj_time_s=round(traj_time, 3),
            )

        n_states = len(trajectory)
        labeled = 0
        skipped = 0
        label_start = time.perf_counter()

        for done, state in enumerate(trajectory, start=1):
            move = fast_dfs(state, max_depth=dfs_max_depth, q=dfs_q, n_workers=n_workers)
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
                f"workers={n_workers}     "
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
    initial_board: SquareBoard,
    optimizer: keras.optimizers.Optimizer,
    n_iterations: int,
    epochs: int,
    batch_size: int,
    dfs_max_depth: int = 8,
    dfs_q: int = 1,
    n_trajectories: int = 1,
    n_initial_trajectories: int = 0,
    max_dataset_size: int | None = None,
    save_path: str | None = "policy_model.keras",
    n_workers: int | None = None,
    log_path: str | None = None,
) -> keras.Model:
    """DAgger using fast_dfs as the teacher, for SquareBoard.

    pi0                      — initial policy from build_square_policy_network
    initial_board            — starting board for every trajectory
    optimizer                — e.g. keras.optimizers.Adam(1e-3)
    n_iterations             — DAgger iterations
    epochs                   — learn() epochs per iteration
    batch_size               — learn() batch size
    dfs_max_depth            — look-ahead depth for fast_dfs
    dfs_q                    — quiescence threshold (extend search when moves <= q)
    n_trajectories           — trajectories rolled out per iteration (default 1)
    n_initial_trajectories   — trajectories collected before iteration 1 to seed the dataset
    max_dataset_size         — cap on dataset length; oldest samples evicted first; None → unlimited
    save_path                — save model after each iteration; None disables saving
    n_workers                — worker processes passed to fast_dfs for root-move parallelism
    log_path                 — JSONL file for progress logging; None disables logging

    Returns the final updated policy.
    """
    n_workers = n_workers or (os.cpu_count() or 1)

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
        n_workers=n_workers,
        board_n=initial_board.n,
        dataset_size=0,
    )

    print(f"Using fast_dfs teacher  (max_depth={dfs_max_depth}, q={dfs_q}, workers={n_workers})")

    if n_initial_trajectories > 0:
        print(f"\n{'='*60}")
        print(f"Pre-DAgger data collection: {n_initial_trajectories} initial trajectory/trajectories")
        print(f"{'='*60}")
        initial_samples = _collect_trajectories(
            pi, initial_board, n_initial_trajectories,
            dfs_max_depth, dfs_q, n_workers,
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
            pi, initial_board, n_trajectories,
            dfs_max_depth, dfs_q, n_workers,
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
    n           = bc["n"]
    empty_start = tuple(bc["empty_start"]) if bc.get("empty_start") else None
    board       = SquareBoard(n, empty_start=empty_start)

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
        initial_board=board,
        optimizer=optimizer,
        n_iterations=dc["n_iterations"],
        epochs=dc["epochs"],
        batch_size=dc["batch_size"],
        dfs_max_depth=dc.get("dfs_max_depth", 8),
        dfs_q=dc.get("dfs_q", 1),
        n_trajectories=dc.get("n_trajectories", 1),
        n_initial_trajectories=dc.get("n_initial_trajectories", 0),
        max_dataset_size=dc.get("max_dataset_size"),
        save_path=dc.get("save_path"),
        n_workers=dc.get("n_workers"),
        log_path=log_path,
    )


if __name__ == "__main__":
    main()

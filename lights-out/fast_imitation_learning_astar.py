"""
fast_imitation_learning_astar.py — DAgger for SquareLightsBoard (Lights Out)
using A* as the labeling teacher.

This mirrors fast_imitation_learning_dfs.py from the peg-solitaire project, with
the pieces swapped for Lights Out:

1. astar replaces fast_dfs as the labeling oracle. astar returns a full
   solution path (a *list* of presses) from a given state; the imitation label
   for a state is the *first* press on that path. States the oracle cannot
   improve (an empty path — already solved/best) are skipped.

2. Parallelism lives at the labeling level, not inside the oracle. astar is
   single-threaded; instead the states of a trajectory are labeled
   concurrently with a process pool (n_workers). Each astar call works on its
   own copy of the board, so the calls are independent.

3. mcts_time_limit / dfs_max_depth → astar_max_expansions.

4. The initial-board set is built with SquareLightsBoard.scramble: a collection
   of random, guaranteed-solvable boards reached by applying k random presses
   to the all-off state (see build_initial_boards).

5. reward_mode is not applicable (A* always minimises lights remaining).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from joblib import Parallel, delayed
import numpy as np
import keras

from board import SquareLightsBoard
from astar import astar
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
    actions = np.array([a for _, a in D], dtype=np.int32)     # (N,)

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
    policy: keras.Model, board: SquareLightsBoard
) -> tuple[list[SquareLightsBoard], SquareLightsBoard]:
    """Roll out policy from board; return (states visited, terminal board).

    Presses are sampled from the policy until the board is over — either solved
    (no lights remain) or every cell has been pressed (no legal moves left).
    The trajectory is bounded by n² steps.
    """
    states: list[SquareLightsBoard] = []
    b = board.copy()
    while not b.is_over():
        states.append(b.copy())
        move = select_action(policy, b, greedy=False)
        b.apply(move)
    return states, b


# ── Initial-board set construction ─────────────────────────────────────────────

def build_initial_boards(
    n: int,
    *,
    num_boards: int = 30,
    scramble_k: int = 10,
    seed: int | None = None,
    max_tries_per_board: int = 200,
) -> list[SquareLightsBoard]:
    """Build a set of random, guaranteed-solvable starting boards.

    Each board is produced by SquareLightsBoard.scramble(n, scramble_k): from
    the all-off (solved) position, scramble_k random presses are applied. Because
    presses are reversible, every scrambled board is solvable.

    Boards are de-duplicated by light configuration so the returned set contains
    no two identical starts. If `seed` is given, board i uses seed + i so the set
    is reproducible; duplicates are retried with fresh offset seeds.

    Returns a list of `num_boards` distinct boards.
    """
    boards: list[SquareLightsBoard] = []
    seen: set[frozenset] = set()
    next_offset = 0

    for _ in range(num_boards):
        tries = 0
        while True:
            if tries >= max_tries_per_board:
                raise RuntimeError(
                    f"build_initial_boards: could not find a new unique board after "
                    f"{tries} tries (have {len(boards)}/{num_boards}). "
                    f"Increase scramble_k or board size, or reduce num_boards."
                )
            tries += 1
            board_seed = None if seed is None else seed + next_offset
            next_offset += 1
            board = SquareLightsBoard.scramble(n, scramble_k, seed=board_seed)
            key = frozenset(board.lights)
            if key in seen:
                continue  # duplicate configuration — try another scramble
            seen.add(key)
            boards.append(board)
            break

    return boards


# ── DAgger ────────────────────────────────────────────────────────────────────

def _label_state(state, astar_max_expansions):
    """Label a single state with the oracle's first press (top-level → picklable).

    astar returns the full press path to the best state it finds. The imitation
    label is the *first* press on that path. An empty path means the oracle could
    not improve on the state (e.g. already solved); such states return None and
    are skipped during collection.
    """
    path = astar(state, max_expansions=astar_max_expansions)
    move = path[0] if path else None
    return state, move


def _collect_trajectories(
    pi: keras.Model,
    initial_boards: list[SquareLightsBoard],
    n_trajectories: int,
    astar_max_expansions: int | None,
    n_workers: int,
    record_fn=None,
    iteration: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Roll out n_trajectories from every initial board, label states, return samples.

    Every board in initial_boards (the set built by build_initial_boards) is used
    as a start: n_trajectories rollouts are generated from each one, for a total of
    len(initial_boards) * n_trajectories trajectories. astar is single-threaded;
    the states within a trajectory are labeled concurrently using a pool of
    n_workers processes.
    """
    samples: list[tuple[np.ndarray, int]] = []

    n_boards = len(initial_boards)
    total_trajectories = n_boards * n_trajectories
    t = 0  # global trajectory counter across all initial boards

    for b_idx, start_board in enumerate(initial_boards):
        for traj_in_board in range(n_trajectories):
            start_lights = start_board.score()
            traj_start = time.perf_counter()
            trajectory, final_board = _gen_trajectory(pi, start_board)
            traj_time = time.perf_counter() - traj_start
            lights_left = final_board.score()

            print(f"\n  [trajectory {t + 1}/{total_trajectories}  "
                  f"board {b_idx + 1}/{n_boards}, rollout {traj_in_board + 1}/{n_trajectories}]  "
                  f"start {start_lights} lights, {len(trajectory)} steps, "
                  f"{lights_left} light(s) remaining  ({traj_time:.2f}s)")
            if record_fn is not None:
                record_fn(
                    type="trajectory",
                    iteration=iteration,
                    trajectory_idx=t,
                    board_idx=b_idx,
                    rollout_idx=traj_in_board,
                    start_lights=start_lights,
                    steps=len(trajectory),
                    lights_remaining=lights_left,
                    traj_time_s=round(traj_time, 3),
                )

            n_states = len(trajectory)
            labeled = 0
            skipped = 0
            label_start = time.perf_counter()

            results = Parallel(
                n_jobs=n_workers, backend="loky", return_as="generator_unordered"
            )(
                delayed(_label_state)(state, astar_max_expansions)
                for state in trajectory
            )
            for done, (state, move) in enumerate(results, start=1):
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
                    f"procs={n_workers}     "
                )
                sys.stdout.flush()

            label_time = time.perf_counter() - label_start
            sys.stdout.write("\n")

            print(f"  labeled {labeled}/{n_states} states  "
                  f"(skipped {skipped})  in {label_time:.1f}s  "
                  f"avg {label_time / max(n_states, 1):.2f}s/state")
            if record_fn is not None:
                record_fn(
                    type="labeling",
                    iteration=iteration,
                    trajectory_idx=t,
                    board_idx=b_idx,
                    rollout_idx=traj_in_board,
                    labeled=labeled,
                    skipped=skipped,
                    label_time_s=round(label_time, 3),
                    rate_states_per_s=round(labeled / label_time if label_time > 0 else 0, 2),
                )

            t += 1

    return samples


def dagger(
    pi0: keras.Model,
    initial_boards: list[SquareLightsBoard],
    optimizer: keras.optimizers.Optimizer,
    n_iterations: int,
    epochs: int,
    batch_size: int,
    astar_max_expansions: int | None = 10000,
    n_trajectories: int = 1,
    max_dataset_size: int | None = None,
    save_path: str | None = "policy_model.keras",
    n_workers: int | None = None,
    log_path: str | None = None,
) -> keras.Model:
    """DAgger using astar as the teacher, for SquareLightsBoard (Lights Out).

    pi0                      — initial policy from build_square_policy_network
    initial_boards           — set of starting boards (see build_initial_boards);
                               every board is rolled out from each iteration
    optimizer                — e.g. keras.optimizers.Adam(1e-3)
    n_iterations             — DAgger iterations
    epochs                   — learn() epochs per iteration
    batch_size               — learn() batch size
    astar_max_expansions     — node-expansion budget for the astar teacher;
                               None → search until solved or exhausted
    n_trajectories           — rollouts per initial board per iteration; total
                               trajectories = len(initial_boards) * n_trajectories
    max_dataset_size         — cap on dataset length; oldest samples evicted first; None → unlimited
    save_path                — save model after each iteration; None disables saving
    n_workers                — number of processes for concurrent state labeling
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
        epochs=epochs,
        batch_size=batch_size,
        astar_max_expansions=astar_max_expansions,
        n_workers=n_workers,
        board_n=initial_boards[0].n,
        n_initial_boards=len(initial_boards),
        dataset_size=0,
    )

    print(f"Using astar teacher  (max_expansions={astar_max_expansions}, "
          f"label_procs={n_workers})")

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
            astar_max_expansions, n_workers,
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


def main(config_path: str = "config_astar.yaml") -> None:
    import argparse
    from policy_network_square import build_square_policy_network

    parser = argparse.ArgumentParser(description="DAgger with astar teacher for Lights Out")
    parser.add_argument("--config", default=config_path, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # Board
    bc = cfg["board"]
    n = bc["n"]

    # Initial-board set: random solvable boards built with scramble.
    isc = bc.get("init_set") or {}
    initial_boards = build_initial_boards(
        n,
        num_boards=isc.get("num_boards", 30),
        scramble_k=isc.get("scramble_k", 10),
        seed=isc.get("seed"),
    )
    print(f"Initial-board set: {len(initial_boards)} boards "
          f"(num_boards={isc.get('num_boards', 30)}, "
          f"scramble_k={isc.get('scramble_k', 10)})")

    # Network
    nc = cfg["network"]
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
        log_path = os.path.join(log_dir, f"{timestamp}_n{n}_astar.jsonl")
        print(f"Progress log → {log_path}")

    dagger(
        pi0=pi0,
        initial_boards=initial_boards,
        optimizer=optimizer,
        n_iterations=dc["n_iterations"],
        epochs=dc["epochs"],
        batch_size=dc["batch_size"],
        astar_max_expansions=dc.get("astar_max_expansions", 10000),
        n_trajectories=dc.get("n_trajectories", 1),
        max_dataset_size=dc.get("max_dataset_size"),
        save_path=dc.get("save_path"),
        n_workers=dc.get("n_workers"),
        log_path=log_path,
    )


if __name__ == "__main__":
    main()

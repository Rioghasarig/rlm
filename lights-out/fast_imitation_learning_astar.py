"""
fast_imitation_learning_astar.py — DAgger for SquareLightsBoard (Lights Out)
using A* as the labeling teacher.

This mirrors fast_imitation_learning_dfs.py from the peg-solitaire project, with
the pieces swapped for Lights Out:

1. astar replaces fast_dfs as the labeling oracle. astar returns a full
   solution path (a *list* of presses) from a given state; the imitation label
   for a state is the *first* press on that path. States the oracle cannot
   improve (an empty path — already solved/best) are skipped.

2. Parallelism lives at the board level, not inside the oracle. astar is
   single-threaded; instead all the work for each initial board — generating
   that board's policy rollouts *and* labeling every visited state with A* — is
   dispatched as one task to a separate process in a pool of n_workers. The
   policy model is saved once per collection round and lazily loaded (and
   cached) inside each worker, so rollouts and labeling for a board happen
   together in the same process. Each board's work is fully independent.

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
import shutil
import sys
import tempfile
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


# Per-worker policy-model cache. Each loky worker loads the model from disk on
# its first board and reuses it for every subsequent board, so the (expensive)
# load happens at most once per process rather than once per board.
_WORKER_MODEL_CACHE: dict[str, keras.Model] = {}


def _get_worker_model(model_path: str) -> keras.Model:
    model = _WORKER_MODEL_CACHE.get(model_path)
    if model is None:
        model = keras.models.load_model(model_path)
        _WORKER_MODEL_CACHE[model_path] = model
    return model


def _process_board(b_idx, start_board, n_trajectories, model_path, astar_max_expansions):
    """Roll out and label every state for one initial board (top-level → picklable).

    Runs entirely inside a worker process: it loads the policy model (cached per
    process), generates n_trajectories rollouts from start_board, and labels
    every visited state with the A* oracle's first press.

    Returns (b_idx, results, traj_meta) where:
      - results   is a list of (state, move) pairs — one per visited state —
                  with move possibly None for states the oracle could not improve;
      - traj_meta is a list of per-rollout dicts (start_lights, steps,
                  lights_remaining, traj_time_s) for logging.
    Returning b_idx lets the caller attribute results to the board even when
    board work completes out of order.
    """
    model = _get_worker_model(model_path)
    results: list[tuple[SquareLightsBoard, object]] = []
    traj_meta: list[dict] = []

    for _ in range(n_trajectories):
        start_lights = start_board.score()
        traj_start = time.perf_counter()
        trajectory, final_board = _gen_trajectory(model, start_board)
        traj_time = time.perf_counter() - traj_start

        for state in trajectory:
            results.append(_label_state(state, astar_max_expansions))

        traj_meta.append({
            "start_lights": start_lights,
            "steps": len(trajectory),
            "lights_remaining": final_board.score(),
            "traj_time_s": round(traj_time, 3),
        })

    return b_idx, results, traj_meta


def _collect_trajectories(
    pi: keras.Model,
    initial_boards: list[SquareLightsBoard],
    n_trajectories: int,
    astar_max_expansions: int | None,
    n_workers: int,
    record_fn=None,
    iteration: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Roll out and label n_trajectories from every initial board; return samples.

    Every board in initial_boards (the set built by build_initial_boards) is used
    as a start: n_trajectories rollouts are generated from each one, for a total of
    len(initial_boards) * n_trajectories trajectories. Rollout generation and A*
    labeling are fused into a single per-board task and parallelized across
    n_workers processes — each process handles one board's rollouts and labels
    all the states it visits. The policy model is saved to a temp file once and
    lazily loaded inside each worker (see _process_board / _get_worker_model).
    """
    samples: list[tuple[np.ndarray, int]] = []

    n_boards = len(initial_boards)
    total_trajectories = n_boards * n_trajectories

    # Persist the current policy so workers can load it (the keras model itself
    # is awkward to pickle per-task; a shared on-disk copy is loaded once per
    # worker and cached). Removed once collection finishes.
    model_dir = tempfile.mkdtemp(prefix="fast_il_astar_model_")
    model_path = os.path.join(model_dir, "policy.keras")
    pi.save(model_path)

    boards_done = 0
    states_done = 0
    trajectories_done = 0
    work_start = time.perf_counter()

    try:
        results = Parallel(
            n_jobs=n_workers, backend="loky", return_as="generator_unordered"
        )(
            delayed(_process_board)(
                b_idx, start_board, n_trajectories, model_path, astar_max_expansions
            )
            for b_idx, start_board in enumerate(initial_boards)
        )
        for b_idx, board_results, traj_meta in results:
            labeled = 0
            skipped = 0
            for state, move in board_results:
                if move is None:
                    skipped += 1
                else:
                    samples.append((state.encode(), state.encode_move(move)))
                    labeled += 1

            boards_done += 1
            states_done += len(board_results)
            trajectories_done += len(traj_meta)

            for rollout_idx, meta in enumerate(traj_meta):
                if record_fn is not None:
                    record_fn(
                        type="trajectory",
                        iteration=iteration,
                        board_idx=b_idx,
                        rollout_idx=rollout_idx,
                        **meta,
                    )

            elapsed = time.perf_counter() - work_start
            rate = states_done / elapsed if elapsed > 0 else 0
            print(f"  [board {b_idx + 1}/{n_boards}]  {len(traj_meta)} rollout(s), "
                  f"labeled {labeled}/{len(board_results)} states (skipped {skipped})  |  "
                  f"{boards_done}/{n_boards} boards, {trajectories_done}/{total_trajectories} "
                  f"trajectories, {states_done} states  {rate:.2f} states/s  procs={n_workers}")
            if record_fn is not None:
                record_fn(
                    type="labeling",
                    iteration=iteration,
                    board_idx=b_idx,
                    labeled=labeled,
                    skipped=skipped,
                    states=len(board_results),
                )
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)

    total_states = states_done
    work_time = time.perf_counter() - work_start
    print(f"  rolled out & labeled {total_states} state(s) across {n_boards} board(s) in "
          f"{work_time:.1f}s  avg {work_time / max(total_states, 1):.2f}s/state")
    if record_fn is not None:
        record_fn(
            type="labeling_summary",
            iteration=iteration,
            total_states=total_states,
            label_time_s=round(work_time, 3),
            rate_states_per_s=round(total_states / work_time if work_time > 0 else 0, 2),
        )

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
    n_workers                — number of processes for concurrent per-board
                               rollout generation and state labeling
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

"""
fast_imitation_learning_astar.py — DAgger for SquareLightsBoard (Lights Out)
using A* as the labeling teacher.

This mirrors fast_imitation_learning_dfs.py from the peg-solitaire project, with
the pieces swapped for Lights Out:

1. astar replaces fast_dfs as the labeling oracle. astar returns the single best
   next move from a given state — a cell press when one improves the board, or
   STOP when nothing does (the right action being to end the game). That move is
   the imitation label directly; it is always actionable, so no state is skipped.

2. Collection runs in two phases. First every initial board's policy rollouts
   are generated sequentially in the main process (using the in-memory policy),
   producing a flat list of every visited state. Then all of those states are
   labeled with the A* oracle in a single parallel pass across n_workers
   processes — one labeling task per state. astar is single-threaded, so the
   parallelism is purely across states; because A* cost varies a lot from state
   to state, per-state dispatch load-balances better than per-board. Labeling
   needs no policy model, so nothing is serialized to disk.

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
from collections import defaultdict
from joblib import Parallel, delayed
import numpy as np
import keras
from tqdm import tqdm

from board import SquareLightsBoard
from astar import astar


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

def _gen_trajectories_batched(
    policy: keras.Model,
    lane_specs: list[tuple[int, int, SquareLightsBoard]],
    gen_start: float,
    max_trajectory_length: int | None = None,
) -> list[dict]:
    """Roll out many trajectories concurrently with one batched forward pass/step.

    Each entry in lane_specs is (board_idx, rollout_idx, start_board) describing
    one independent rollout ("lane"). All lanes advance in lockstep: at every
    step the still-active lanes are stacked into a single (A, n, n, 2) batch and
    scored with one policy call, then each lane samples a legal press (the same
    masked-softmax sampling as the original per-board rollout) and applies it.
    Lanes drop out as they finish; because each press is a distinct cell, every
    lane is over within n² steps.

    If max_trajectory_length is given, a lane is also cut off once it has
    collected that many states, even if the board is not yet solved (None → no
    limit).

    Returns one dict per lane: board_idx, rollout_idx, start_lights, states
    (the visited pre-press boards, in order), the terminal board, and the
    wall-clock time (relative to gen_start) at which that lane finished.
    """
    lanes: list[dict] = [
        {
            "board_idx": b_idx,
            "rollout_idx": rollout_idx,
            "board": start_board.copy(),
            "start_lights": start_board.score(),
            "states": [],
            "traj_time_s": 0.0,
        }
        for b_idx, rollout_idx, start_board in lane_specs
    ]

    active = [ln for ln in lanes if not ln["board"].is_over()]
    while active:
        batch = np.stack([ln["board"].encode() for ln in active])  # (A, n, n, 2)
        logits = policy(batch, training=False).numpy()             # (A, n*n)

        still_active = []
        for ln, lane_logits in zip(active, logits):
            b = ln["board"]
            ln["states"].append(b.copy())

            legal_moves = b.available_moves()
            legal_codes = [b.encode_move(m) for m in legal_moves]
            legal_logits = lane_logits[legal_codes]
            legal_logits = legal_logits - legal_logits.max()  # numerical stability
            probs = np.exp(legal_logits)
            probs /= probs.sum()
            idx = int(np.random.choice(len(legal_moves), p=probs))
            b.apply(legal_moves[idx])
            reached_limit = (
                max_trajectory_length is not None
                and len(ln["states"]) >= max_trajectory_length
            )
            if b.is_over() or reached_limit:
                ln["traj_time_s"] = time.perf_counter() - gen_start
            else:
                still_active.append(ln)
        active = still_active

    return lanes


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
    """Label a single state with the oracle's best next move (top-level → picklable).

    astar returns the single move to play next from *state* — a cell press when
    some press improves it, or STOP when nothing does (including an already-solved
    board). The label is that move directly; it is always actionable, so no state
    is skipped during collection.
    """
    move = astar(state, max_expansions=astar_max_expansions)
    return state, move


def _collect_trajectories(
    pi: keras.Model,
    initial_boards: list[SquareLightsBoard],
    n_trajectories: int,
    astar_max_expansions: int | None,
    n_workers: int,
    record_fn=None,
    iteration: int | None = None,
    max_trajectory_length: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Roll out every initial board, then label every visited state; return samples.

    Runs in two phases:

      1. Generation — every board in initial_boards (the set built by
         build_initial_boards) is rolled out n_trajectories times in the main
         process using the in-memory policy, for a total of
         len(initial_boards) * n_trajectories trajectories. Every visited state
         is collected into a single flat list.

      2. Labeling — all collected states are labeled with the A* oracle in one
         parallel pass across n_workers processes (one task per state). Labeling
         needs no policy model, so nothing is serialized to disk.
    """
    n_boards = len(initial_boards)
    total_trajectories = n_boards * n_trajectories

    # ── Phase 1: generate every trajectory (batched rollouts, main process) ──
    # All boards × rollouts advance in lockstep so each step is a single batched
    # policy forward pass rather than one batch-of-1 call per board per step.
    gen_start = time.perf_counter()
    states: list[SquareLightsBoard] = []   # flat list of every visited state
    state_board: list[int] = []            # board index per state (for logging)

    lane_specs = [
        (b_idx, rollout_idx, start_board)
        for b_idx, start_board in enumerate(initial_boards)
        for rollout_idx in range(n_trajectories)
    ]
    lanes = _gen_trajectories_batched(pi, lane_specs, gen_start, max_trajectory_length)

    # lane_specs is board-major, so flattening lanes preserves board order.
    for ln in lanes:
        states.extend(ln["states"])
        state_board.extend(ln["board_idx"] for _ in ln["states"])

        if record_fn is not None:
            record_fn(
                type="trajectory",
                iteration=iteration,
                board_idx=ln["board_idx"],
                rollout_idx=ln["rollout_idx"],
                start_lights=ln["start_lights"],
                steps=len(ln["states"]),
                lights_remaining=ln["board"].score(),
                traj_time_s=round(ln["traj_time_s"], 3),
            )

    gen_time = time.perf_counter() - gen_start
    print(f"  generated {total_trajectories} trajectory(ies) across {n_boards} board(s) "
          f"→ {len(states)} state(s) in {gen_time:.1f}s")

    # ── Phase 2: label every state in parallel (one task per state) ──
    label_start = time.perf_counter()
    labeled_results = Parallel(
        n_jobs=n_workers, backend="loky", return_as="generator"
    )(
        delayed(_label_state)(state, astar_max_expansions) for state in states
    )

    # return_as="generator" preserves input order, so results line up with
    # state_board and can be attributed back to each board.
    samples: list[tuple[np.ndarray, int]] = []
    per_board: dict[int, dict[str, int]] = defaultdict(
        lambda: {"labeled": 0, "skipped": 0, "states": 0}
    )
    for b_idx, (state, move) in tqdm(
        zip(state_board, labeled_results),
        total=len(states),
        desc="Labeling states",
        unit="state",
    ):
        pb = per_board[b_idx]
        pb["states"] += 1
        if move is None:
            pb["skipped"] += 1
        else:
            samples.append((state.encode(), state.encode_move(move)))
            pb["labeled"] += 1

    states_done = 0
    label_time = time.perf_counter() - label_start
    for b_idx in range(n_boards):
        pb = per_board[b_idx]
        states_done += pb["states"]
        rate = states_done / label_time if label_time > 0 else 0
        print(f"  [board {b_idx + 1}/{n_boards}]  "
              f"labeled {pb['labeled']}/{pb['states']} states (skipped {pb['skipped']})  |  "
              f"{states_done}/{len(states)} states labeled  {rate:.2f} states/s  procs={n_workers}")
        if record_fn is not None:
            record_fn(
                type="labeling",
                iteration=iteration,
                board_idx=b_idx,
                labeled=pb["labeled"],
                skipped=pb["skipped"],
                states=pb["states"],
            )

    total_states = len(states)
    print(f"  rolled out & labeled {total_states} state(s) across {n_boards} board(s) in "
          f"{gen_time + label_time:.1f}s "
          f"(gen {gen_time:.1f}s, label {label_time:.1f}s)  "
          f"avg {label_time / max(total_states, 1):.2f}s/state")
    if record_fn is not None:
        record_fn(
            type="labeling_summary",
            iteration=iteration,
            total_states=total_states,
            gen_time_s=round(gen_time, 3),
            label_time_s=round(label_time, 3),
            rate_states_per_s=round(total_states / label_time if label_time > 0 else 0, 2),
        )

    return samples


def dagger(
    pi0: keras.Model,
    optimizer: keras.optimizers.Optimizer,
    n_iterations: int,
    epochs: int,
    batch_size: int,
    board_n: int,
    num_boards: int = 30,
    scramble_k: int = 10,
    board_seed: int | None = None,
    astar_max_expansions: int | None = 10000,
    n_trajectories: int = 1,
    max_trajectory_length: int | None = None,
    max_dataset_size: int | None = None,
    checkpoint_dir: str | None = "checkpoints",
    model_name: str = "policy_model",
    n_workers: int | None = None,
    log_path: str | None = None,
) -> keras.Model:
    """DAgger using astar as the teacher, for SquareLightsBoard (Lights Out).

    A fresh set of boards is generated at the start of each iteration.

    pi0                      — initial policy from build_square_policy_network
    optimizer                — e.g. keras.optimizers.Adam(1e-3)
    n_iterations             — DAgger iterations
    epochs                   — learn() epochs per iteration
    batch_size               — learn() batch size
    board_n                  — board size (n×n grid)
    num_boards               — number of boards to generate per iteration
    scramble_k               — number of random presses used to scramble boards
    board_seed               — base seed for board generation; None → random
    astar_max_expansions     — node-expansion budget for the astar teacher;
                               None → search until solved or exhausted
    n_trajectories           — rollouts per initial board per iteration; total
                               trajectories = num_boards * n_trajectories
    max_trajectory_length    — cap on the number of states collected per rollout;
                               a lane is cut off once it reaches this many states
                               even if unsolved; None → no limit
    max_dataset_size         — cap on dataset length; oldest samples evicted first; None → unlimited
    checkpoint_dir           — directory in which a distinct per-iteration model
                               checkpoint is saved after each iteration; None
                               disables per-iteration checkpointing
    model_name               — base name for the per-iteration checkpoint files
                               (saved as <model_name>_iter<NNNN>.keras)
    n_workers                — number of processes for concurrent per-board
                               rollout generation and state labeling
    log_path                 — JSONL file for progress logging; None disables logging

    Returns the final updated policy.
    """
    n_workers = n_workers or (os.cpu_count() or 1)

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

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
        max_trajectory_length=max_trajectory_length,
        epochs=epochs,
        batch_size=batch_size,
        astar_max_expansions=astar_max_expansions,
        n_workers=n_workers,
        board_n=board_n,
        num_boards=num_boards,
        scramble_k=scramble_k,
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

        iter_seed = None if board_seed is None else board_seed + i * num_boards
        initial_boards = build_initial_boards(
            board_n,
            num_boards=num_boards,
            scramble_k=scramble_k,
            seed=iter_seed,
        )
        print(f"  generated {len(initial_boards)} boards (scramble_k={scramble_k})")

        new_samples_list = _collect_trajectories(
            pi, initial_boards, n_trajectories,
            astar_max_expansions, n_workers,
            record_fn=record, iteration=i + 1,
            max_trajectory_length=max_trajectory_length,
        )
        _append_samples(D, new_samples_list)

        print(f"\n  dataset: {len(D)} total  (+{len(new_samples_list)} this iteration)")
        print(f"  --- training ---")
        train_start = time.perf_counter()
        pi = learn(D, pi, optimizer, epochs, batch_size, record_fn=record, iteration=i + 1)
        train_time = time.perf_counter() - train_start

        if checkpoint_dir:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"{model_name}_iter{i + 1:04d}.keras"
            )
            pi.save(checkpoint_path)
            print(f"  checkpoint saved → {checkpoint_path}")

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

    isc = bc.get("init_set") or {}

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
        optimizer=optimizer,
        n_iterations=dc["n_iterations"],
        epochs=dc["epochs"],
        batch_size=dc["batch_size"],
        board_n=n,
        num_boards=isc.get("num_boards", 30),
        scramble_k=isc.get("scramble_k", 10),
        board_seed=isc.get("seed"),
        astar_max_expansions=dc.get("astar_max_expansions", 10000),
        n_trajectories=dc.get("n_trajectories", 1),
        max_trajectory_length=dc.get("max_trajectory_length"),
        max_dataset_size=dc.get("max_dataset_size"),
        checkpoint_dir=dc.get("checkpoint_dir", "checkpoints"),
        model_name=dc.get("model_name", "policy_model"),
        n_workers=dc.get("n_workers"),
        log_path=log_path,
    )


if __name__ == "__main__":
    main()

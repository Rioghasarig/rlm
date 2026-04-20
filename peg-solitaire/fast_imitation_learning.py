"""
fast_imitation_learning.py — JAX-accelerated DAgger for SquareBoard peg solitaire.

Speedups over imitation_learning.py
------------------------------------
1. fast_mcts (fast_mcts_square.py) replaces mcts:
     - JAX lax.scan rollout compiled to a single XLA kernel (no Python loop per step)
     - Flat NumPy tree with vectorised UCB1 / scatter-add backpropagation
     - Precomputed move table — validity checks are pure array indexing

2. learn() uses model.compile(jit_compile=True) + model.fit():
     - XLA-compiles the entire training step (forward + backward + update) once
     - Eliminates Python GradientTape overhead on every batch
     - Keras handles epoch shuffling and batching inside the compiled graph

3. JAX rollout kernel is warmed up once before the DAgger loop so the first
   real MCTS search is not penalised by XLA compilation latency (~1-2 s).

4. dagger() pre-compiles the training step before iteration 1 so that the first
   learn() call does not pay compilation cost mid-loop.

5. Multiple trajectories per DAgger iteration (n_trajectories) diversify the
   dataset with no additional MCTS warm-up cost.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
import numpy as np
import keras
import jax
import jax.numpy as jnp

from board import SquareBoard
from fast_mcts_square import fast_mcts, _move_table, _board_to_flat, _run_rollout
from policy_network_square import select_action


# ── JSONL progress logging ────────────────────────────────────────────────────

def _record(log_path: str | None, run_start: float, **fields) -> None:
    if log_path is None:
        return
    fields["t"] = round(time.perf_counter() - run_start, 3)
    os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
    with open(log_path, "a") as f:
        f.write(json.dumps(fields) + "\n")


# ── Parallel MCTS workers ─────────────────────────────────────────────────────
# Must be module-level for pickling under 'spawn'.  Each worker warms up the
# JAX XLA kernel once in its initializer so timed MCTS calls pay no compile cost.

_worker_time_limit: float = 1.0


def _mcts_worker_init(n: int, time_limit: float) -> None:
    global _worker_time_limit
    _worker_time_limit = time_limit
    warmup_jax_kernel(SquareBoard(n))


def _mcts_label_one(args: tuple) -> tuple[int, tuple | None]:
    idx, board = args
    return idx, fast_mcts(board, time_limit=_worker_time_limit)


# ── JAX warm-up ───────────────────────────────────────────────────────────────

def warmup_jax_kernel(board: SquareBoard) -> None:
    """Trigger XLA compilation of the JAX rollout kernel for board.n.

    Call once before any timed MCTS search; subsequent calls to fast_mcts()
    will reuse the already-compiled kernel.
    """
    n  = board.n
    mt = jnp.array(_move_table(n))
    bf = jnp.array(_board_to_flat(board))
    _run_rollout(bf, jax.random.PRNGKey(0), mt, n * n).block_until_ready()


# ── Training ──────────────────────────────────────────────────────────────────

def _ensure_jit_compiled(
    model: keras.Model,
    optimizer: keras.optimizers.Optimizer,
) -> None:
    """Compile model with jit_compile=True exactly once per model instance.

    Subsequent calls with the same model are no-ops, so optimizer momentum
    and other stateful variables are preserved across DAgger iterations.
    """
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
    """Train policy_model on dataset D using an XLA-compiled training step.

    Replaces the manual GradientTape loop with model.compile(jit_compile=True)
    + model.fit().  The model is compiled on the first call and reused on all
    subsequent calls with the same instance, preserving optimizer state.

    D               — list of (encoded_board, action_code) pairs
    policy_model    — built with build_square_policy_network; expects (B,n,n,1)
    optimizer       — e.g. keras.optimizers.Adam(1e-3)
    epochs          — number of full passes over D
    batch_size      — mini-batch size

    Returns the updated policy model.
    """
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
        move = select_action(policy, b, greedy=True)
        b.move(move[0], move[2])
    return states, b


# ── DAgger ────────────────────────────────────────────────────────────────────

def dagger(
    pi0: keras.Model,
    initial_board: SquareBoard,
    optimizer: keras.optimizers.Optimizer,
    n_iterations: int,
    epochs: int,
    batch_size: int,
    mcts_time_limit: float = 1.0,
    n_trajectories: int = 1,
    save_path: str | None = "policy_model.keras",
    n_workers: int | None = None,
    log_path: str | None = None,
) -> keras.Model:
    """DAgger using fast_mcts as the teacher, for SquareBoard.

    Differences from imitation_learning.dagger():
      - fast_mcts (JAX lax.scan rollouts, flat-array tree) labels each state.
      - MCTS labeling is parallelized across n_workers processes (default:
        os.cpu_count()).  Each worker warms up the JAX XLA kernel once in its
        initializer; the pool is reused across all DAgger iterations.
      - Training uses jit_compile=True compiled once before iteration 1 and
        reused on every subsequent learn() call without retracing.
      - n_trajectories trajectories are rolled out per iteration, diversifying
        the dataset with the current policy before MCTS labelling.

    pi0                      — initial policy from build_square_policy_network
    initial_board            — starting board for every trajectory
    optimizer                — e.g. keras.optimizers.Adam(1e-3)
    n_iterations             — DAgger iterations
    epochs                   — learn() epochs per iteration
    batch_size               — learn() batch size
    mcts_time_limit          — wall-clock seconds given to fast_mcts per state
    n_trajectories           — trajectories rolled out per iteration (default 1)
    save_path                — save model after each iteration; None disables saving
    n_workers                — worker processes for parallel MCTS labeling (default: all CPUs)

    Returns the final updated policy.
    """
    n_workers = n_workers or (os.cpu_count() or 1)

    # Pre-compile the training step before iteration 1 so the first learn()
    # call does not pay XLA compilation cost mid-loop.
    _ensure_jit_compiled(pi0, optimizer)

    D: list[tuple[np.ndarray, int]] = []
    pi = pi0
    run_start = time.perf_counter()

    def record(**fields):
        _record(log_path, run_start, **fields)

    record(
        type="run_start",
        n_iterations=n_iterations,
        n_trajectories=n_trajectories,
        epochs=epochs,
        batch_size=batch_size,
        mcts_time_limit=mcts_time_limit,
        n_workers=n_workers,
        board_n=initial_board.n,
        dataset_size=0,
    )

    # 'spawn' is required: JAX initializes device handles that don't survive fork.
    # Workers warm up the XLA kernel once in their initializer.
    ctx = mp.get_context("spawn")
    print(f"Spawning {n_workers} MCTS worker process(es) and warming up JAX...")
    with ctx.Pool(
        processes=n_workers,
        initializer=_mcts_worker_init,
        initargs=(initial_board.n, mcts_time_limit),
    ) as pool:
        print("Workers ready.")

        for i in range(n_iterations):
            iter_start = time.perf_counter()
            elapsed_total = iter_start - run_start
            print(f"\n{'='*60}")
            print(f"DAgger iteration {i + 1}/{n_iterations}  "
                  f"(elapsed {elapsed_total:.0f}s, dataset {len(D)} samples)")
            print(f"{'='*60}")
            record(type="iteration_start", iteration=i + 1, dataset_size=len(D))

            new_samples = 0

            for t in range(n_trajectories):
                traj_start = time.perf_counter()
                trajectory, final_board = _gen_trajectory(pi, initial_board)
                traj_time = time.perf_counter() - traj_start
                pegs_left = int(final_board.encode().sum())

                traj_label = f"trajectory {t + 1}/{n_trajectories}" if n_trajectories > 1 else "trajectory"
                print(f"\n  [{traj_label}]  {len(trajectory)} steps, "
                      f"{pegs_left} peg(s) remaining  ({traj_time:.2f}s)")
                record(
                    type="trajectory",
                    iteration=i + 1,
                    trajectory_idx=t,
                    steps=len(trajectory),
                    pegs_remaining=pegs_left,
                    traj_time_s=round(traj_time, 3),
                )

                # Parallel MCTS labeling — each state is independent.
                n_states    = len(trajectory)
                label_start = time.perf_counter()
                raw_results: list[tuple[int, tuple | None]] = []

                for done, (idx, move) in enumerate(
                    pool.imap_unordered(_mcts_label_one, enumerate(trajectory)),
                    start=1,
                ):
                    raw_results.append((idx, move))
                    pct  = done / n_states * 100
                    elapsed_label = time.perf_counter() - label_start
                    rate = done / elapsed_label if elapsed_label > 0 else 0
                    sys.stdout.write(
                        f"\r  labeling: {done:>{len(str(n_states))}}/{n_states} "
                        f"({pct:5.1f}%)  {rate:.1f} states/s  "
                        f"workers={n_workers}     "
                    )
                    sys.stdout.flush()

                label_time = time.perf_counter() - label_start
                sys.stdout.write("\n")

                labeled = 0
                skipped = 0
                for idx, move in raw_results:
                    if move is None:
                        skipped += 1
                        continue
                    state = trajectory[idx]
                    D.append((state.encode(), state.encode_move(move)))
                    labeled += 1
                    new_samples += 1

                print(f"  labeled {labeled}/{n_states} states  "
                      f"(skipped {skipped})  in {label_time:.1f}s  "
                      f"avg {label_time/n_states:.2f}s/state")
                record(
                    type="labeling",
                    iteration=i + 1,
                    trajectory_idx=t,
                    labeled=labeled,
                    skipped=skipped,
                    label_time_s=round(label_time, 3),
                    rate_states_per_s=round(labeled / label_time if label_time > 0 else 0, 2),
                )

            print(f"\n  dataset: {len(D)} total  (+{new_samples} this iteration)")
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
                new_samples=new_samples,
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


def main(config_path: str = "config.yaml") -> None:
    import argparse
    from policy_network_square import build_square_policy_network

    parser = argparse.ArgumentParser(description="Fast DAgger for peg solitaire")
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
        log_path = os.path.join(log_dir, f"{timestamp}_n{n}.jsonl")
        print(f"Progress log → {log_path}")

    dagger(
        pi0=pi0,
        initial_board=board,
        optimizer=optimizer,
        n_iterations=dc["n_iterations"],
        epochs=dc["epochs"],
        batch_size=dc["batch_size"],
        mcts_time_limit=dc["mcts_time_limit"],
        n_trajectories=dc.get("n_trajectories", 1),
        save_path=dc.get("save_path"),
        n_workers=dc.get("n_workers"),
        log_path=log_path,
    )


if __name__ == "__main__":
    main()

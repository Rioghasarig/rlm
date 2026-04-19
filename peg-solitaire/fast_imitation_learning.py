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

import numpy as np
import keras
import jax
import jax.numpy as jnp

from board import SquareBoard
from fast_mcts_square import fast_mcts, _move_table, _board_to_flat, _run_rollout
from policy_network_square import select_action


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

    history = policy_model.fit(
        states, actions,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )
    for ep, loss in enumerate(history.history["loss"], 1):
        print(f"  epoch {ep}/{epochs}  loss={loss:.4f}")

    return policy_model


# ── Trajectory generation ─────────────────────────────────────────────────────

def _gen_trajectory(policy: keras.Model, board: SquareBoard) -> list[SquareBoard]:
    """Roll out policy from board; return each state visited."""
    states: list[SquareBoard] = []
    b = board.copy()
    while b.available_moves():
        states.append(b.copy())
        move = select_action(policy, b)
        b.move(move[0], move[2])
    return states


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
) -> keras.Model:
    """DAgger using fast_mcts as the teacher, for SquareBoard.

    Differences from imitation_learning.dagger():
      - fast_mcts (JAX lax.scan rollouts, flat-array tree) labels each state.
      - JAX rollout kernel is warmed up once before the iteration loop so no
        compilation cost is paid inside the timed MCTS budget.
      - Training uses jit_compile=True compiled once before iteration 1 and
        reused on every subsequent learn() call without retracing.
      - n_trajectories trajectories are rolled out per iteration, diversifying
        the dataset with the current policy before MCTS labelling.

    pi0             — initial policy from build_square_policy_network
    initial_board   — starting board for every trajectory
    optimizer       — e.g. keras.optimizers.Adam(1e-3)
    n_iterations    — DAgger iterations
    epochs          — learn() epochs per iteration
    batch_size      — learn() batch size
    mcts_time_limit — wall-clock seconds given to fast_mcts per state
    n_trajectories  — trajectories rolled out per iteration (default 1)
    save_path       — save model after each iteration; None disables saving

    Returns the final updated policy.
    """
    print("Warming up JAX rollout kernel...")
    warmup_jax_kernel(initial_board)
    print("JAX rollout kernel ready.")

    # Pre-compile the training step before iteration 1 so the first learn()
    # call does not pay XLA compilation cost mid-loop.
    _ensure_jit_compiled(pi0, optimizer)

    D: list[tuple[np.ndarray, int]] = []
    pi = pi0

    for i in range(n_iterations):
        print(f"\n--- DAgger iteration {i + 1}/{n_iterations} ---")

        for t in range(n_trajectories):
            trajectory = _gen_trajectory(pi, initial_board)
            if n_trajectories > 1:
                print(f"  trajectory {t + 1}/{n_trajectories}: {len(trajectory)} states")

            for state in trajectory:
                move = fast_mcts(state, time_limit=mcts_time_limit)
                if move is None:
                    continue
                D.append((state.encode(), state.encode_move(move)))

        print(f"  dataset size: {len(D)}")
        pi = learn(D, pi, optimizer, epochs, batch_size)

        if save_path:
            pi.save(save_path)
            print(f"  model saved → {save_path}")

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
    )


if __name__ == "__main__":
    main()

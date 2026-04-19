import numpy as np
import keras

from board import SquareBoard
from mcts import mcts
from policy_network_square import select_action


def learn(
    D: list[tuple[np.ndarray, int]],
    policy_model: keras.Model,
    optimizer: keras.optimizers.Optimizer,
    epochs: int,
    batch_size: int,
) -> keras.Model:
    """Train policy_model on a dataset of (board_encoding, action_code) pairs.

    D               — list of (encoded_board, action_code) where encoded_board
                      has shape (n, n) and action_code is from board.encode_move()
    policy_model    — built with build_square_policy_network; expects (B, n, n, 1) input
    optimizer       — e.g. keras.optimizers.Adam(1e-3)
    epochs          — number of full passes over D
    batch_size      — mini-batch size

    Returns the updated policy model.
    """
    states = np.array([s for s, _ in D], dtype=np.float32)[..., np.newaxis]  # (N, n, n, 1)
    actions = np.array([a for _, a in D], dtype=np.int32)                     # (N,)

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    n_samples = len(states)
    indices = np.arange(n_samples)

    for epoch in range(epochs):
        np.random.shuffle(indices)
        batch_losses = []

        for start in range(0, n_samples, batch_size):
            idx = indices[start : start + batch_size]
            s_batch = states[idx]
            a_batch = actions[idx]

            with keras.GradientTape() as tape:
                logits = policy_model(s_batch, training=True)
                loss = loss_fn(a_batch, logits)

            grads = tape.gradient(loss, policy_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))
            batch_losses.append(float(loss))

        print(f"epoch {epoch + 1}/{epochs}  loss={np.mean(batch_losses):.4f}")

    return policy_model


def _gen_trajectory(policy: keras.Model, board: SquareBoard) -> list[SquareBoard]:
    """Roll out the policy from the given board; return each state visited."""
    states = []
    b = board.copy()
    while b.available_moves():
        states.append(b.copy())
        move = select_action(policy, b)
        b.move(move[0], move[2])
    return states


def dagger(
    pi0: keras.Model,
    initial_board: SquareBoard,
    optimizer: keras.optimizers.Optimizer,
    n_iterations: int,
    epochs: int,
    batch_size: int,
    mcts_time_limit: float = 1.0,
    save_path: str | None = "policy_model.keras",
) -> keras.Model:
    """DAgger using MCTS as the teacher, for SquareBoard.

    pi0             — initial policy built with build_square_policy_network
    initial_board   — starting board position for every trajectory
    optimizer       — e.g. keras.optimizers.Adam(1e-3)
    n_iterations    — number of DAgger iterations
    epochs          — learn() epochs per iteration
    batch_size      — learn() batch size
    mcts_time_limit — seconds given to MCTS per state label
    save_path       — path to save the model after each iteration; None disables saving

    Returns the final updated policy.
    """
    D: list[tuple[np.ndarray, int]] = []
    pi = pi0

    for i in range(n_iterations):
        print(f"\n--- DAgger iteration {i + 1}/{n_iterations} ---")

        trajectory = _gen_trajectory(pi, initial_board)

        for state in trajectory:
            move = mcts(state, time_limit=mcts_time_limit)
            if move is None:
                continue
            D.append((state.encode(), state.encode_move(move)))

        print(f"  dataset size: {len(D)}")
        pi = learn(D, pi, optimizer, epochs, batch_size)

        if save_path:
            pi.save(save_path)
            print(f"  model saved → {save_path}")

    return pi

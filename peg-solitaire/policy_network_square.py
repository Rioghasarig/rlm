import numpy as np
import keras
from keras import layers

from board import SquareBoard


def _residual_block(x, filters):
    skip = x
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([skip, x])
    return layers.Activation('relu')(x)


def build_square_policy_network(n: int, res_blocks: int = 4, filters: int = 64) -> keras.Model:
    """Policy network for SquareBoard of size n.

    Input shape:  (batch, n, n, 1)  — board.encode()[..., np.newaxis]
    Output shape: (batch, n*n*4)    — raw logits over encode_move action codes
    """
    action_size = n * n * len(SquareBoard._DIRECTIONS)

    inp = keras.Input(shape=(n, n, 1), name='board')

    x = layers.Conv2D(filters, 3, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    for _ in range(res_blocks):
        x = _residual_block(x, filters)

    x = layers.Conv2D(2, 1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    policy = layers.Dense(action_size, name='policy')(x)

    return keras.Model(inp, policy, name=f'square_policy_{n}x{n}')


def encode_board(board: SquareBoard) -> np.ndarray:
    """Return (1, n, n, 1) float32 array ready for model input."""
    return board.encode()[np.newaxis, ..., np.newaxis]


def select_action(model: keras.Model, board: SquareBoard, greedy: bool = False) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Select a legal move using the policy model.

    greedy=True  → pick the move with highest probability (argmax).
    greedy=False → sample from the softmax distribution over legal moves.
    """
    legal_moves = board.available_moves()
    if not legal_moves:
        raise ValueError("No legal moves available")

    logits = model(encode_board(board), training=False)[0].numpy()

    legal_codes = [board.encode_move(m) for m in legal_moves]
    legal_logits = logits[legal_codes]
    legal_logits -= legal_logits.max()  # numerical stability

    if greedy:
        idx = int(np.argmax(legal_logits))
    else:
        probs = np.exp(legal_logits)
        probs /= probs.sum()
        idx = np.random.choice(len(legal_moves), p=probs)

    return legal_moves[idx]

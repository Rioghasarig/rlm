"""
fast_mcts_square.py — JAX-accelerated MCTS for SquareBoard peg solitaire.

Speedups over mcts.py:
  - Precomputed move table: validity checks are pure array indexing (no Python loops)
  - Flat NumPy arrays for the tree: UCB1 and backprop are vectorized NumPy ops
  - JAX lax.scan: rollout compiled to a single XLA kernel, no Python interpreter per step
  - Path recording during selection: backprop is a single scatter-add, no pointer chase
"""
from __future__ import annotations

import random
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from board import SquareBoard


# ── Move table ────────────────────────────────────────────────────────────────
# All (from_idx, over_idx, to_idx) flat-index triples for an n×n board,
# computed once per board size.

def _move_table(n: int) -> np.ndarray:
    rows = []
    for r in range(n):
        for c in range(n):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                tr, tc = r + 2*dr, c + 2*dc
                if 0 <= tr < n and 0 <= tc < n:
                    rows.append((r*n + c, (r+dr)*n + (c+dc), tr*n + tc))
    return np.array(rows, dtype=np.int32)


def _board_to_flat(board: SquareBoard) -> np.ndarray:
    arr = np.zeros(board.n ** 2, dtype=bool)
    for r, c in board.pegs:
        arr[r * board.n + c] = True
    return arr


def _flat_to_pos(idx: int, n: int) -> tuple[int, int]:
    return (idx // n, idx % n)


# ── JAX rollout ───────────────────────────────────────────────────────────────
# One JIT-compiled XLA kernel plays a single random game via lax.scan.
# max_steps is static so XLA knows the loop bound at compile time.

@partial(jax.jit, static_argnums=(3,))
def _run_rollout(
    board_flat: jnp.ndarray,  # (n*n,) bool  — starting board
    key: jnp.ndarray,         # (2,)   uint32 — PRNG key
    move_table: jnp.ndarray,  # (M, 3) int32  — precomputed move triples
    max_steps: int,           # static upper bound on game length
) -> jnp.ndarray:             # ()     int32  — pegs remaining
    M = move_table.shape[0]

    def step(carry, _):
        board, k, done = carry
        k, sk = jax.random.split(k)

        mask = (board[move_table[:, 0]] &
                board[move_table[:, 1]] &
                ~board[move_table[:, 2]])
        n_valid = mask.sum()

        # Give invalid moves a tiny weight so probs always sums to 1.
        # The chosen move is discarded whenever done | no_moves.
        raw   = jnp.where(mask, 1.0, 1e-9)
        probs = raw / raw.sum()
        idx   = jax.random.choice(sk, M, p=probs)

        new_board = (board
                     .at[move_table[idx, 0]].set(False)
                     .at[move_table[idx, 1]].set(False)
                     .at[move_table[idx, 2]].set(True))

        no_moves  = n_valid == 0
        board_out = jnp.where(done | no_moves, board, new_board)
        return (board_out, k, done | no_moves), None

    (final, _, _), _ = jax.lax.scan(
        step,
        (board_flat, key, jnp.bool_(False)),
        None,
        length=max_steps,
    )
    return final.sum().astype(jnp.int32)


# ── Flat tree ─────────────────────────────────────────────────────────────────
# The entire MCTS tree lives in parallel NumPy arrays indexed by node ID.
# UCB1 selection over a node's children is a single np.argmax call.
# Backpropagation is a scatter-add on a recorded integer path.

_MAX_NODES    = 50_000
_MAX_CHILDREN = 160  # upper bound: 4 directions × n² cells, generous for any n


class _FlatTree:
    __slots__ = (
        'visits', 'reward', 'parent', 'children', 'n_children',
        'boards', 'untried', 'move_fr', 'move_ov', 'move_to', 'size',
    )

    def __init__(self, board_flat: np.ndarray, root_untried: list[int]) -> None:
        N, B = _MAX_NODES, len(board_flat)
        self.visits     = np.zeros(N, np.int32)
        self.reward     = np.zeros(N, np.float32)
        self.parent     = np.full(N, -1, np.int32)
        self.n_children = np.zeros(N, np.int32)
        self.children   = np.full((N, _MAX_CHILDREN), -1, np.int32)
        self.move_fr    = np.zeros(N, np.int32)
        self.move_ov    = np.zeros(N, np.int32)
        self.move_to    = np.zeros(N, np.int32)
        self.boards     = np.zeros((N, B), bool)
        self.untried: list[list[int] | None] = [None] * N
        self.size       = 1

        self.boards[0]  = board_flat
        self.untried[0] = root_untried

    def is_terminal(self, node: int) -> bool:
        return self.n_children[node] == 0 and not self.untried[node]

    def is_fully_expanded(self, node: int) -> bool:
        return not self.untried[node]

    def best_child(self, node: int, c: float) -> int:
        nc  = self.n_children[node]
        ids = self.children[node, :nc]
        q   = self.reward[ids] / self.visits[ids]
        u   = c * np.sqrt(np.log(self.visits[node]) / self.visits[ids])
        return int(ids[np.argmax(q + u)])

    def expand(self, node: int, move_table: np.ndarray) -> int:
        untried  = self.untried[node]
        move_idx = untried.pop(random.randrange(len(untried)))
        fr = int(move_table[move_idx, 0])
        ov = int(move_table[move_idx, 1])
        to = int(move_table[move_idx, 2])

        child = self.size
        self.size += 1

        # Copy parent board and apply the move in-place (no Board.copy() overhead)
        self.boards[child]     = self.boards[node]
        self.boards[child, fr] = False
        self.boards[child, ov] = False
        self.boards[child, to] = True

        self.parent[child]  = node
        self.move_fr[child] = fr
        self.move_ov[child] = ov
        self.move_to[child] = to

        b    = self.boards[child]
        mask = b[move_table[:, 0]] & b[move_table[:, 1]] & ~b[move_table[:, 2]]
        child_untried = list(np.where(mask)[0])
        random.shuffle(child_untried)
        self.untried[child] = child_untried

        nc = self.n_children[node]
        self.children[node, nc] = child
        self.n_children[node]  += 1
        return child

    def backpropagate(self, path: list[int], reward: float) -> None:
        arr = np.array(path, np.int32)
        self.visits[arr] += 1
        self.reward[arr] += reward


# ── Public entry point ────────────────────────────────────────────────────────

def fast_mcts(
    board: SquareBoard,
    time_limit: float = 2.0,
    exploration: float = 1.41,
) -> tuple | None:
    """
    JAX-accelerated MCTS from *board* for up to *time_limit* seconds.

    Args:
        board:       SquareBoard to search (not mutated).
        time_limit:  Wall-clock seconds to search.
        exploration: UCB1 constant (default √2 ≈ 1.41).

    Returns:
        Best (from_pos, over_pos, to_pos) triple, or None if no moves exist.
    """
    n          = board.n
    move_table = _move_table(n)
    max_steps  = n * n
    board_flat = _board_to_flat(board)

    initial_count = int(board_flat.sum())
    if initial_count <= 1:
        return None

    move_table_jax = jnp.array(move_table)

    mask = (board_flat[move_table[:, 0]] &
            board_flat[move_table[:, 1]] &
            ~board_flat[move_table[:, 2]])
    root_untried = list(np.where(mask)[0])
    if not root_untried:
        return None

    # Warm up JIT before the clock starts — first call triggers XLA compilation
    _run_rollout(jnp.array(board_flat), jax.random.PRNGKey(0), move_table_jax, max_steps).block_until_ready()

    random.shuffle(root_untried)
    tree     = _FlatTree(board_flat, root_untried)
    key      = jax.random.PRNGKey(random.randint(0, 2**31 - 1))
    deadline = time.monotonic() + time_limit

    while time.monotonic() < deadline and tree.size < _MAX_NODES - 1:

        # 1. Selection — descend via UCB1, recording the path as a flat int list
        node = 0
        path = [0]
        while tree.is_fully_expanded(node) and not tree.is_terminal(node):
            node = tree.best_child(node, exploration)
            path.append(node)

        if tree.is_terminal(node):
            continue

        # 2. Expansion — one new child, board copied in NumPy (no Board objects)
        node = tree.expand(node, move_table)
        path.append(node)

        # 3. Simulation — one compiled JAX rollout
        key, sub  = jax.random.split(key)
        remaining = int(_run_rollout(jnp.asarray(tree.boards[node]), sub, move_table_jax, max_steps))
        reward    = (initial_count - remaining) / max(initial_count - 1, 1)

        # 4. Backpropagation — two NumPy scatter-adds along the recorded path
        tree.backpropagate(path, float(reward))

    if tree.n_children[0] == 0:
        return None

    nc   = tree.n_children[0]
    ids  = tree.children[0, :nc]
    best = int(ids[np.argmax(tree.visits[ids])])

    return (
        _flat_to_pos(tree.move_fr[best], n),
        _flat_to_pos(tree.move_ov[best], n),
        _flat_to_pos(tree.move_to[best], n),
    )

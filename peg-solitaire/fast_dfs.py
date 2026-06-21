"""
Optimized depth-first search for peg solitaire.

Optimizations over dfs.py:
  1. JAX JIT-compiled move generation  — eliminates Python-loop overhead on
                                         the hot inner loop
  2. In-place numpy mutation (apply/undo) — avoids array copies
  3. Move ordering                     — most-open moves first so the win
                                         cutoff fires sooner
  4. Quiescence search                 — extends past the depth limit in
                                         narrow positions (moves <= q)

This module is single-threaded. Parallelism, when wanted, is the caller's
responsibility (e.g. labeling many states concurrently with a thread pool).

Public API
----------
fast_dfs(board, max_depth, q) -> (from_pos, over_pos, to_pos) | None
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from board import Board


# ── board conversion ──────────────────────────────────────────────────────────

def _to_array(board: Board) -> np.ndarray:
    arr = np.zeros((board.n, board.n), dtype=np.int8)
    for r, c in board.pegs:
        arr[r, c] = 1
    return arr


# ── candidate move tables ─────────────────────────────────────────────────────

def _build_candidates(
    board: Board,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute all geometrically possible (fr, ov, to) triples."""
    frs, ovs, tos = [], [], []
    for r in range(board.n):
        for c in range(board.n):
            if not board._in_bounds(r, c):
                continue
            for dr, dc in board._DIRECTIONS:
                ovr, ovc = r + dr, c + dc
                tor, toc = r + 2 * dr, c + 2 * dc
                if board._in_bounds(ovr, ovc) and board._in_bounds(tor, toc):
                    frs.append((r, c))
                    ovs.append((ovr, ovc))
                    tos.append((tor, toc))
    return (
        np.array(frs, dtype=np.int32),
        np.array(ovs, dtype=np.int32),
        np.array(tos, dtype=np.int32),
    )


# ── JAX JIT-compiled move generation ─────────────────────────────────────────

@jax.jit
def _valid_mask(
    board_arr: jnp.ndarray,
    frs: jnp.ndarray,
    ovs: jnp.ndarray,
    tos: jnp.ndarray,
) -> jnp.ndarray:
    """Boolean mask over candidate moves that are currently legal."""
    return (
        (board_arr[frs[:, 0], frs[:, 1]] == 1)
        & (board_arr[ovs[:, 0], ovs[:, 1]] == 1)
        & (board_arr[tos[:, 0], tos[:, 1]] == 0)
    )


def _get_moves(arr: np.ndarray, frs, ovs, tos) -> list:
    mask = np.asarray(_valid_mask(arr, frs, ovs, tos))
    idx = np.where(mask)[0]
    return [(tuple(frs[i]), tuple(ovs[i]), tuple(tos[i])) for i in idx]


# ── in-place apply / undo ─────────────────────────────────────────────────────

def _apply(arr, fr, ov, to) -> None:
    arr[fr] = 0
    arr[ov] = 0
    arr[to] = 1


def _undo(arr, fr, ov, to) -> None:
    arr[fr] = 1
    arr[ov] = 1
    arr[to] = 0


# ── move ordering ─────────────────────────────────────────────────────────────

def _order_moves(arr, frs, ovs, tos, moves) -> list:
    """Sort moves by successor count descending (most open first)."""
    scored = []
    for fr, ov, to in moves:
        _apply(arr, fr, ov, to)
        n_succ = int(np.sum(_valid_mask(arr, frs, ovs, tos)))
        _undo(arr, fr, ov, to)
        scored.append((n_succ, fr, ov, to))
    scored.sort(reverse=True)
    return [(fr, ov, to) for _, fr, ov, to in scored]


def _limit_breadth(moves, max_breadth) -> list:
    """Keep at most *max_breadth* moves, discarding the rest.

    Assumes *moves* is already sorted most-open first, so this keeps the moves
    with the most successors.
    """
    if max_breadth is None or len(moves) <= max_breadth:
        return moves
    return moves[:max_breadth]


# ── recursive DFS ─────────────────────────────────────────────────────────────

def _dfs(arr, frs, ovs, tos, depth: int, q: int, max_breadth) -> int:
    moves = _get_moves(arr, frs, ovs, tos)
    if not moves:
        return int(arr.sum())
    if depth == 0 and len(moves) > q:
        return int(arr.sum())

    next_depth = depth - 1 if depth > 0 else 0
    best = int(arr.sum())

    moves = _order_moves(arr, frs, ovs, tos, moves)
    moves = _limit_breadth(moves, max_breadth)
    for fr, ov, to in moves:
        _apply(arr, fr, ov, to)
        result = _dfs(arr, frs, ovs, tos, next_depth, q, max_breadth)
        _undo(arr, fr, ov, to)
        if result < best:
            best = result
            if best == 1:
                break

    return best


# ── public entry point ────────────────────────────────────────────────────────

def fast_dfs(
    board: Board,
    max_depth: int,
    q: int = 1,
    max_breadth: int | None = None,
) -> tuple | None:
    """
    Search *board* to *max_depth* plies and return the move that minimises
    the number of pegs remaining.

    Single-threaded: root moves are evaluated sequentially. The win cutoff
    (1 peg) short-circuits the search.

    Args:
        board:       Board to search from (not mutated).
        max_depth:   Maximum number of moves to look ahead.
        q:           Quiescence threshold — at the depth limit, keep searching
                     if available moves <= q (default 1).
        max_breadth: Maximum number of children to expand per node. When a
                     node has more than *max_breadth* legal moves, the
                     *max_breadth* moves with the most successors are kept and
                     the rest are discarded. None (default) expands every child.

    Returns:
        Best (from_pos, over_pos, to_pos) triple, or None if no moves exist.
    """
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if max_breadth is not None and max_breadth < 1:
        raise ValueError("max_breadth must be at least 1")

    arr = _to_array(board)
    frs, ovs, tos = _build_candidates(board)

    # Warm up JIT.
    _ = _valid_mask(arr, frs, ovs, tos)

    moves = _get_moves(arr, frs, ovs, tos)
    if not moves:
        return None

    moves = _order_moves(arr, frs, ovs, tos, moves)
    moves = _limit_breadth(moves, max_breadth)

    best_score = int(arr.sum()) + 1
    best_move = None
    for fr, ov, to in moves:
        _apply(arr, fr, ov, to)
        score = _dfs(arr, frs, ovs, tos, max_depth - 1, q, max_breadth)
        _undo(arr, fr, ov, to)
        if score < best_score:
            best_score = score
            best_move = (fr, ov, to)
            if best_score == 1:  # optimal — no need to look further
                break

    return best_move

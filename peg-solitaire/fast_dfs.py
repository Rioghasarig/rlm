"""
Optimized depth-first search for peg solitaire.

Optimizations over dfs.py:
  1. JAX JIT-compiled move generation  — eliminates Python-loop overhead on
                                         the hot inner loop
  2. Multiprocessing                   — root moves evaluated in parallel
                                         across CPU cores
  3. Transposition table per worker    — skips re-evaluating repeated states
  4. In-place numpy mutation (apply/undo) — avoids array copies
  5. Move ordering                     — most-open moves first so the win
                                         cutoff fires sooner
  6. Quiescence search                 — extends past the depth limit in
                                         narrow positions (moves <= q)

Public API
----------
fast_dfs(board, max_depth, q, n_workers) -> (from_pos, over_pos, to_pos) | None
"""
from __future__ import annotations

import multiprocessing

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


# ── recursive DFS ─────────────────────────────────────────────────────────────

def _dfs(arr, frs, ovs, tos, depth: int, q: int, table: dict) -> int:
    moves = _get_moves(arr, frs, ovs, tos)
    if not moves:
        return int(arr.sum())
    if depth == 0 and len(moves) > q:
        return int(arr.sum())

    key = (arr.tobytes(), depth)
    cached = table.get(key)
    if cached is not None:
        return cached

    next_depth = depth - 1 if depth > 0 else 0
    best = int(arr.sum())

    for fr, ov, to in _order_moves(arr, frs, ovs, tos, moves):
        _apply(arr, fr, ov, to)
        result = _dfs(arr, frs, ovs, tos, next_depth, q, table)
        _undo(arr, fr, ov, to)
        if result < best:
            best = result
            if best == 1:
                break

    table[key] = best
    return best


# ── multiprocessing worker ────────────────────────────────────────────────────

def _worker(args: tuple) -> tuple[int, tuple]:
    arr, frs, ovs, tos, fr, ov, to, depth, q = args
    arr = arr.copy()
    _apply(arr, fr, ov, to)
    score = _dfs(arr, frs, ovs, tos, depth, q, {})
    return score, (fr, ov, to)


# ── public entry point ────────────────────────────────────────────────────────

def fast_dfs(
    board: Board,
    max_depth: int,
    q: int = 1,
    n_workers: int | None = None,
) -> tuple | None:
    """
    Search *board* to *max_depth* plies and return the move that minimises
    the number of pegs remaining.

    Args:
        board:     Board to search from (not mutated).
        max_depth: Maximum number of moves to look ahead.
        q:         Quiescence threshold — at the depth limit, keep searching
                   if available moves <= q (default 1).
        n_workers: Worker processes for root-move parallelism
                   (default: os.cpu_count()).

    Returns:
        Best (from_pos, over_pos, to_pos) triple, or None if no moves exist.
    """
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")

    arr = _to_array(board)
    frs, ovs, tos = _build_candidates(board)

    # Warm up JIT before spawning so workers inherit compiled XLA kernels.
    _ = _valid_mask(arr, frs, ovs, tos)

    moves = _get_moves(arr, frs, ovs, tos)
    if not moves:
        return None

    moves = _order_moves(arr, frs, ovs, tos, moves)
    worker_args = [
        (arr, frs, ovs, tos, fr, ov, to, max_depth - 1, q)
        for fr, ov, to in moves
    ]

    if n_workers == 1 or len(worker_args) == 1:
        results = [_worker(a) for a in worker_args]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_worker, worker_args)

    best_score = int(arr.sum()) + 1
    best_move = None
    for score, move in results:
        if score < best_score:
            best_score = score
            best_move = move

    return best_move

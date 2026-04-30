"""
Depth-first search for peg solitaire.

Searches up to max_depth moves ahead and returns the first move that leads
to the minimum number of pegs reachable within that horizon.

Quiescence search: when the depth limit is reached, the search continues at
depth 0 as long as the number of available moves is <= q (a narrow position).
Wide positions (moves > q) are cut off as usual.

Public API
----------
dfs(board, max_depth, q) -> (from_pos, over_pos, to_pos) | None
"""
from __future__ import annotations

from board import Board


def _dfs(board: Board, depth: int, q: int) -> int:
    """Return the minimum peg count reachable from *board* within *depth* moves."""
    moves = board.available_moves()
    if not moves:
        return len(board.pegs)
    if depth == 0 and len(moves) > q:
        return len(board.pegs)

    next_depth = depth - 1 if depth > 0 else 0
    best = len(board.pegs)
    for fr, _ov, to in moves:
        child = board.copy()
        child.move(fr, to)
        result = _dfs(child, next_depth, q)
        if result < best:
            best = result
    return best


def dfs(board: Board, max_depth: int, q: int = 1) -> tuple | None:
    """
    Search *board* to *max_depth* plies and return the move that minimises
    the number of pegs remaining.

    Args:
        board:     Board to search from (not mutated).
        max_depth: Maximum number of moves to look ahead.
        q:         Quiescence threshold — at the depth limit, keep searching
                   if available moves <= q (default 1).

    Returns:
        Best (from_pos, over_pos, to_pos) triple, or None if no moves exist.
    """
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")

    moves = board.available_moves()
    if not moves:
        return None

    best_move = None
    best_score = len(board.pegs) + 1  # worse than any reachable outcome

    for fr, ov, to in moves:
        child = board.copy()
        child.move(fr, to)
        score = _dfs(child, max_depth - 1, q)
        if score < best_score:
            best_score = score
            best_move = (fr, ov, to)

    return best_move

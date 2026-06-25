"""
A* search for Lights Out.

By default the heuristic is simply the number of lights still on, and every
move has zero cost (we don't care about the length of the solution, only about
turning lights off). With g = 0 the priority f = g + h reduces to h, so the
open set is a priority queue ordered by lights-remaining — the search always
expands the most promising (fewest-lights) state next.

Alternatively, a trained policy network can drive the search instead. When a
`policy` model is supplied, a child's priority is the policy's negative
log-likelihood of the press that produced it rather than its lights-remaining
score, so the open set is ordered by how strongly the policy favours each move
and the most promising presses are expanded first.

Because presses commute and are self-inverse, the *set* of pressed cells fully
determines the resulting position, so states are deduplicated by that set: all
orderings of the same presses collapse to a single node.

The search still explores full paths internally, but only the *first* move of
the best path is returned — the single move to play next from `board`. This
matches how the result is consumed (one move per state, re-queried after each
step). The result is always an actionable move: a cell press when some press
reduces the lights, otherwise the "stop" sentinel (``board.STOP``) — including
when the board is already solved, where stopping locks in the win.

Public API
----------
astar(board, max_expansions, policy) -> move   # next move (a press or STOP)
"""
from __future__ import annotations

import heapq
import itertools

import numpy as np

from board import Board


def _move_log_probs(policy, node: Board) -> dict:
    """Return {move: log P(move)} over *node*'s legal presses under *policy*.

    A single forward pass scores every cell; the logits for the legal presses
    are softmaxed among themselves so the probabilities form a distribution
    over the moves actually available at this node.
    """
    legal_moves = node.available_moves()
    if not legal_moves:
        return {}

    logits = policy(node.encode()[np.newaxis, ...], training=False)[0]
    logits = np.asarray(logits, dtype=np.float64)

    legal_codes = [node.encode_move(m) for m in legal_moves]
    legal_logits = logits[legal_codes]
    legal_logits -= legal_logits.max()  # numerical stability
    log_norm = np.log(np.exp(legal_logits).sum())
    log_probs = legal_logits - log_norm
    return {move: float(lp) for move, lp in zip(legal_moves, log_probs)}


def astar(
    board: Board,
    max_expansions: int | None = None,
    policy=None,
):
    """Search *board* for the best next move toward turning off every light.

    Internally the search explores full paths to find the state with the fewest
    lights remaining; only the *first* move of that best path is returned — the
    single move to play next from *board*.

    Args:
        board:          Starting board (not mutated).
        max_expansions: Stop after this many nodes have been expanded and use
                        the best solution found so far. None (default) searches
                        until the open set is exhausted or a full solution (0
                        lights) is found.
        policy:         Optional trained policy model (e.g. from
                        ``build_square_policy_network``). When given, children
                        are prioritised by the policy's negative log-likelihood
                        of the press that produced them instead of by their
                        lights-remaining score. When None (default) the search
                        is plain lights-remaining A*.

    Returns:
        The best next move — always actionable, never "do nothing":

        * a cell press ``(row, col)`` when some press improves on the board;
        * ``board.STOP`` otherwise — when no press reduces the lights (stop
          rather than waste presses), or the board is already solved (stop to
          lock in the win).
    """
    start = board.copy()

    # Best (fewest-lights) state found so far, and the path that reaches it.
    # When nothing improves the board, the right move is to stop.
    best_score = start.score()
    best_path: list = []
    if start.is_won():
        return start.STOP

    # A unique sequence number keeps heap entries from ever comparing boards
    # when their f-values tie.
    counter = itertools.count()
    open_heap = [(start.score(), next(counter), start, [])]
    visited: set[frozenset] = set()
    expansions = 0

    while open_heap:
        _, _, node, path = heapq.heappop(open_heap)

        key = frozenset(node.pressed)
        if key in visited:
            continue
        visited.add(key)

        # h = 0: every light is off — this is the best possible outcome, so the
        # first press on the path that reaches it is the best next move.
        if node.is_won():
            return path[0] if path else node.STOP

        if max_expansions is not None and expansions >= max_expansions:
            break
        expansions += 1

        # With a policy, score each child by the move's negative log-likelihood
        # (one forward pass per node); otherwise by its lights-remaining count.
        log_probs = _move_log_probs(policy, node) if policy is not None else None

        for move in node.available_moves():
            child = node.copy()
            child.apply(move)
            if frozenset(child.pressed) in visited:
                continue

            child_score = child.score()
            child_path = path + [move]
            if child_score < best_score:
                best_score = child_score
                best_path = child_path

            priority = -log_probs[move] if log_probs is not None else child_score
            heapq.heappush(open_heap, (priority, next(counter), child, child_path))

    # A press improves on the start: play the first one toward the best state.
    # Otherwise nothing helps — stop here rather than waste a press.
    if best_path:
        return best_path[0]
    return start.STOP

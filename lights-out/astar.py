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

Public API
----------
astar(board, max_expansions, policy) -> list[move]   # presses to the best state
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
) -> list:
    """Search *board* for a sequence of presses that turns off as many lights
    as possible (ideally all of them).

    Args:
        board:          Starting board (not mutated).
        max_expansions: Stop after this many nodes have been expanded and
                        return the best solution found so far. None (default)
                        searches until the open set is exhausted or a full
                        solution (0 lights) is found.
        policy:         Optional trained policy model (e.g. from
                        ``build_square_policy_network``). When given, children
                        are prioritised by the policy's negative log-likelihood
                        of the press that produced them instead of by their
                        lights-remaining score. When None (default) the search
                        is plain lights-remaining A*.

    Returns:
        The list of moves (cell presses) leading to the state with the fewest
        lights remaining that was found. Empty if the start is already the best
        state seen (e.g. already solved).
    """
    start = board.copy()

    # Best (fewest-lights) state found so far, and the path that reaches it.
    best_score = start.score()
    best_path: list = []
    if start.is_won():
        return best_path

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

        # h = 0: every light is off — this is the best possible outcome.
        if node.is_won():
            return path

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

    return best_path

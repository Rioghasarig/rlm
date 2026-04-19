"""
Monte Carlo Tree Search for peg solitaire.

Single-player MCTS (no adversary).  Each rollout plays random moves to a
terminal state and scores the result by how many pegs were removed relative
to what was possible from the search root.  UCB1 drives exploration.

Public API
----------
mcts(board, time_limit, exploration) -> (from_pos, over_pos, to_pos) | None
"""
from __future__ import annotations

import math
import random
import time

from board import Board


# ── helpers ───────────────────────────────────────────────────────────────────

def _rollout(board: Board) -> int:
    """Play random moves to a terminal state; return pegs remaining."""
    b = board.copy()
    while True:
        moves = b.available_moves()
        if not moves:
            return len(b.pegs)
        fr, _ov, to = random.choice(moves)
        b.move(fr, to)


# ── MCTS node ─────────────────────────────────────────────────────────────────

class _Node:
    __slots__ = ('board', 'move', 'parent', 'children',
                 'visits', 'total_reward', 'untried')

    def __init__(
        self,
        board: Board,
        move: tuple | None = None,
        parent: '_Node | None' = None,
    ) -> None:
        self.board = board
        self.move = move          # (from, over, to) that led to this node
        self.parent = parent
        self.children: list[_Node] = []
        self.visits = 0
        self.total_reward = 0.0
        self.untried = board.available_moves()
        random.shuffle(self.untried)

    # ---- tree predicates ----

    @property
    def is_terminal(self) -> bool:
        return not self.untried and not self.children

    @property
    def is_fully_expanded(self) -> bool:
        return not self.untried

    # ---- tree operations ----

    def ucb1(self, c: float) -> float:
        return (
            self.total_reward / self.visits
            + c * math.sqrt(math.log(self.parent.visits) / self.visits)
        )

    def best_child(self, c: float) -> '_Node':
        return max(self.children, key=lambda ch: ch.ucb1(c))

    def expand(self) -> '_Node':
        index = random.randint(0, len(self.untried)-1)
        fr, ov, to = self.untried.pop(index)
        child_board = self.board.copy()
        child_board.move(fr, to)
        child = _Node(child_board, move=(fr, ov, to), parent=self)
        self.children.append(child)
        return child

    def backpropagate(self, reward: float) -> None:
        node: _Node | None = self
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent


# ── public entry point ────────────────────────────────────────────────────────

def mcts(
    board: Board,
    time_limit: float = 2.0,
    exploration: float = 1.41,
) -> tuple | None:
    """
    Run MCTS from *board* for up to *time_limit* wall-clock seconds.

    Args:
        board:       Board to search from (not mutated).
        time_limit:  seconds to search (default 2).
        exploration: UCB1 constant (default √2 ≈ 1.41).

    Returns:
        Best (from_pos, over_pos, to_pos) triple, or None if no moves exist.
    """
    if len(board.pegs) <= 1:
        return None

    root = _Node(board.copy())
    if not root.untried:
        return None

    initial_count = len(board.pegs)
    deadline = time.monotonic() + time_limit

    while time.monotonic() < deadline:
        # 1. Selection — descend via UCB1 until a non-fully-expanded node
        node = root
        while node.is_fully_expanded and not node.is_terminal:
            node = node.best_child(exploration)

        # 2. Expansion — add one new child
        if not node.is_terminal:
            node = node.expand()

        # 3. Simulation — random playout
        remaining = _rollout(node.board)
        reward = (initial_count - remaining) / max(initial_count - 1, 1)

        # 4. Backpropagation
        node.backpropagate(reward)

    if not root.children:
        return None

    # Most-visited child is the most robust estimate
    best = max(root.children, key=lambda c: c.visits)
    return best.move

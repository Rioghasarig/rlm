from abc import ABC, abstractmethod

import numpy as np


class Board(ABC):
    """Abstract base class for peg solitaire boards."""

    pegs: set[tuple[int, int]]
    n: int

    @abstractmethod
    def _in_bounds(self, row: int, col: int) -> bool: ...

    @abstractmethod
    def available_moves(self) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]: ...

    @abstractmethod
    def move(self, from_pos: tuple[int, int], to_pos: tuple[int, int]) -> None: ...

    @abstractmethod
    def is_won(self) -> bool: ...

    @abstractmethod
    def copy(self) -> 'Board': ...

    @abstractmethod
    def encode(self) -> np.ndarray: ...

    @abstractmethod
    def encode_move(self, move: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> int: ...

    @abstractmethod
    def decode_move(self, code: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]: ...


class TriangularBoard(Board):
    """
    Peg solitaire on a triangular grid of side length n.

    Holes are addressed as (row, col) where:
      - row in [0, n-1]
      - col in [0, row]

    The six axial move directions on a triangular grid:
      horizontal:  (0, -1)  (0, +1)
      upper:       (-1, -1) (-1,  0)
      lower:       (+1,  0) (+1, +1)
    """

    _DIRECTIONS = [(0, -1), (0, 1), (-1, -1), (-1, 0), (1, 0), (1, 1)]

    def __init__(self, n: int, empty_start: tuple[int, int] = (0, 0)):
        if n < 2:
            raise ValueError("Board size must be at least 2")
        self.n = n
        self.pegs: set[tuple[int, int]] = {
            (row, col)
            for row in range(n)
            for col in range(row + 1)
        }
        if not self._in_bounds(*empty_start):
            raise ValueError(f"empty_start {empty_start} is not on the board")
        self.pegs.discard(empty_start)

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.n and 0 <= col <= row

    def available_moves(self) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
        moves = []
        for (r, c) in self.pegs:
            for dr, dc in self._DIRECTIONS:
                over = (r + dr, c + dc)
                to   = (r + 2 * dr, c + 2 * dc)
                if (
                    over in self.pegs
                    and self._in_bounds(*to)
                    and to not in self.pegs
                ):
                    moves.append(((r, c), over, to))
        return moves

    def move(self, from_pos: tuple[int, int], to_pos: tuple[int, int]) -> None:
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]

        valid_jumps = {(2 * d[0], 2 * d[1]) for d in self._DIRECTIONS}
        if (dr, dc) not in valid_jumps:
            raise ValueError(f"({from_pos} -> {to_pos}) is not a valid jump direction")

        over = (from_pos[0] + dr // 2, from_pos[1] + dc // 2)

        if from_pos not in self.pegs:
            raise ValueError(f"No peg at {from_pos}")
        if over not in self.pegs:
            raise ValueError(f"No peg to jump over at {over}")
        if to_pos in self.pegs:
            raise ValueError(f"Destination {to_pos} is already occupied")
        if not self._in_bounds(*to_pos):
            raise ValueError(f"Destination {to_pos} is out of bounds")

        self.pegs.remove(from_pos)
        self.pegs.remove(over)
        self.pegs.add(to_pos)

    def is_won(self) -> bool:
        return len(self.pegs) == 1

    def copy(self) -> 'TriangularBoard':
        c = TriangularBoard.__new__(TriangularBoard)
        c.n = self.n
        c.pegs = self.pegs.copy()
        return c

    def encode(self) -> np.ndarray:
        t = np.zeros((2, self.n, self.n), dtype=np.float32)
        for row in range(self.n):
            for col in range(row + 1):
                t[1, row, col] = 1.0
                t[0, row, col] = 1.0 if (row, col) in self.pegs else 0.0
        return t

    def encode_move(self, move: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> int:
        (r, c), _, _ = move
        dr = move[2][0] - r
        dc = move[2][1] - c
        dir_idx = self._DIRECTIONS.index((dr // 2, dc // 2))
        return (r * self.n + c) * len(self._DIRECTIONS) + dir_idx

    def decode_move(self, code: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        nd = len(self._DIRECTIONS)
        flat, dir_idx = divmod(code, nd)
        r, c = divmod(flat, self.n)
        dr, dc = self._DIRECTIONS[dir_idx]
        return (r, c), (r + dr, c + dc), (r + 2 * dr, c + 2 * dc)

    def __repr__(self) -> str:
        rows = []
        for row in range(self.n):
            padding = " " * (self.n - row - 1)
            cells = " ".join("o" if (row, col) in self.pegs else "." for col in range(row + 1))
            rows.append(padding + cells)
        return "\n".join(rows)


class SquareBoard(Board):
    """
    Peg solitaire on an n×n square grid.

    Holes are addressed as (row, col) where row and col are in [0, n-1].
    Pegs jump orthogonally (up, down, left, right) exactly two steps,
    removing the peg in between.
    """

    _DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, n: int, empty_start: tuple[int, int] | None = None):
        if n < 2:
            raise ValueError("Board size must be at least 2")
        self.n = n
        self.pegs: set[tuple[int, int]] = {
            (row, col)
            for row in range(n)
            for col in range(n)
        }
        if empty_start is None:
            empty_start = (n // 2, n // 2)
        if not self._in_bounds(*empty_start):
            raise ValueError(f"empty_start {empty_start} is not on the board")
        self.pegs.discard(empty_start)

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.n and 0 <= col < self.n

    def available_moves(self) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
        moves = []
        for (r, c) in self.pegs:
            for dr, dc in self._DIRECTIONS:
                over = (r + dr, c + dc)
                to   = (r + 2 * dr, c + 2 * dc)
                if (
                    over in self.pegs
                    and self._in_bounds(*to)
                    and to not in self.pegs
                ):
                    moves.append(((r, c), over, to))
        return moves

    def move(self, from_pos: tuple[int, int], to_pos: tuple[int, int]) -> None:
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]

        valid_jumps = {(2 * d[0], 2 * d[1]) for d in self._DIRECTIONS}
        if (dr, dc) not in valid_jumps:
            raise ValueError(f"({from_pos} -> {to_pos}) is not a valid jump direction")

        over = (from_pos[0] + dr // 2, from_pos[1] + dc // 2)

        if from_pos not in self.pegs:
            raise ValueError(f"No peg at {from_pos}")
        if over not in self.pegs:
            raise ValueError(f"No peg to jump over at {over}")
        if to_pos in self.pegs:
            raise ValueError(f"Destination {to_pos} is already occupied")
        if not self._in_bounds(*to_pos):
            raise ValueError(f"Destination {to_pos} is out of bounds")

        self.pegs.remove(from_pos)
        self.pegs.remove(over)
        self.pegs.add(to_pos)

    def is_won(self) -> bool:
        return len(self.pegs) == 1

    def copy(self) -> 'SquareBoard':
        c = SquareBoard.__new__(SquareBoard)
        c.n = self.n
        c.pegs = self.pegs.copy()
        return c

    def encode(self) -> np.ndarray:
        """Return an (n, n, 2) float32 tensor describing the position.

        Channel 0 — pegs: 1.0 where a peg sits, else 0.0.
        Channel 1 — in-bounds mask: 1.0 for playable holes, 0.0 for off-board
                    cells (e.g. the CrossBoard corners). For a plain SquareBoard
                    every cell is in bounds, so channel 1 is all ones.
        """
        t = np.zeros((self.n, self.n, 2), dtype=np.float32)
        for row in range(self.n):
            for col in range(self.n):
                if self._in_bounds(row, col):
                    t[row, col, 1] = 1.0
                    if (row, col) in self.pegs:
                        t[row, col, 0] = 1.0
        return t

    def encode_move(self, move: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> int:
        (r, c), _, _ = move
        dr = move[2][0] - r
        dc = move[2][1] - c
        dir_idx = self._DIRECTIONS.index((dr // 2, dc // 2))
        return (r * self.n + c) * len(self._DIRECTIONS) + dir_idx

    def decode_move(self, code: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        nd = len(self._DIRECTIONS)
        flat, dir_idx = divmod(code, nd)
        r, c = divmod(flat, self.n)
        dr, dc = self._DIRECTIONS[dir_idx]
        return (r, c), (r + dr, c + dc), (r + 2 * dr, c + 2 * dc)

    def __repr__(self) -> str:
        rows = []
        for row in range(self.n):
            cells = " ".join("o" if (row, col) in self.pegs else "." for col in range(self.n))
            rows.append(cells)
        return "\n".join(rows)


class CrossBoard(SquareBoard):
    """
    The classic English peg solitaire board: a plus/cross shape carved out of
    a 7×7 grid by removing the four 2×2 corner blocks, leaving 33 holes.

    A cell (row, col) is part of the board when it lies in the central three
    rows or the central three columns; the corners (where both row and col fall
    in {0, 1, 5, 6}) are off the board.

    Pegs jump orthogonally exactly two steps, removing the peg in between.
    The standard starting position has every hole filled except the centre.
    """

    n = 7

    def __init__(self, empty_start: tuple[int, int] = (3, 3)):
        self.pegs: set[tuple[int, int]] = {
            (row, col)
            for row in range(self.n)
            for col in range(self.n)
            if self._in_bounds(row, col)
        }
        if not self._in_bounds(*empty_start):
            raise ValueError(f"empty_start {empty_start} is not on the board")
        self.pegs.discard(empty_start)

    def _in_bounds(self, row: int, col: int) -> bool:
        if not (0 <= row < self.n and 0 <= col < self.n):
            return False
        arm = self.n // 3  # = 2 for the standard 7×7 board
        return (arm <= row < self.n - arm) or (arm <= col < self.n - arm)

    def copy(self) -> 'CrossBoard':
        c = CrossBoard.__new__(CrossBoard)
        c.pegs = self.pegs.copy()
        return c

    def __repr__(self) -> str:
        rows = []
        for row in range(self.n):
            cells = " ".join(
                ("o" if (row, col) in self.pegs else ".") if self._in_bounds(row, col) else " "
                for col in range(self.n)
            )
            rows.append(cells)
        return "\n".join(rows)

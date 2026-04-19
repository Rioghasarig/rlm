from abc import ABC, abstractmethod


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

    def is_won(self) -> bool:
        return len(self.pegs) == 1

    def copy(self) -> 'Board':
        c = self.__class__.__new__(self.__class__)
        c.n = self.n
        c.pegs = self.pegs.copy()
        return c


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

    def __repr__(self) -> str:
        rows = []
        for row in range(self.n):
            cells = " ".join("o" if (row, col) in self.pegs else "." for col in range(self.n))
            rows.append(cells)
        return "\n".join(rows)

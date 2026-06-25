from abc import ABC, abstractmethod

import numpy as np


class Board(ABC):
    """Abstract base class for Lights Out boards.

    A Lights Out board is a grid of lights, each on or off. A *move* is a
    single cell press, which toggles that cell together with its orthogonal
    neighbours (the "plus" stencil). The puzzle is won when every light is
    off; the game ends with that many lights still on as the score.

    Unlike peg solitaire, presses are self-inverse (pressing the same cell
    twice is a no-op) and commute (the order of presses does not matter), so a
    solution is a *set* of cells to press rather than an ordered sequence.
    Because re-pressing a cell can only undo earlier work, each cell may be
    pressed at most once: a cell already pressed is no longer an available
    move. The game therefore terminates on its own — once every cell has been
    pressed there are no legal moves left, exactly as peg solitaire ends when
    no jumps remain.
    """

    lights: set[tuple[int, int]]
    pressed: set[tuple[int, int]]
    n: int

    @abstractmethod
    def _in_bounds(self, row: int, col: int) -> bool: ...

    @abstractmethod
    def available_moves(self) -> list[tuple[int, int]]: ...

    @abstractmethod
    def apply(self, move: tuple[int, int]) -> None: ...

    @abstractmethod
    def is_won(self) -> bool: ...

    @abstractmethod
    def is_over(self) -> bool: ...

    @abstractmethod
    def score(self) -> int: ...

    @abstractmethod
    def copy(self) -> 'Board': ...

    @abstractmethod
    def encode(self) -> np.ndarray: ...

    @abstractmethod
    def encode_move(self, move: tuple[int, int]) -> int: ...

    @abstractmethod
    def decode_move(self, code: int) -> tuple[int, int]: ...


class SquareLightsBoard(Board):
    """
    Lights Out on an n×n square grid.

    Cells are addressed as (row, col) with row and col in [0, n-1]. Pressing a
    cell toggles that cell and each of its in-bounds orthogonal neighbours
    (up, down, left, right). Each cell may be pressed at most once — an
    already-pressed cell is removed from the available moves. The board is won
    when no lights remain on, and the game ends once no unpressed cells remain.

    The starting position has every light on by default; pass `lights` to
    specify an arbitrary configuration, or use `SquareLightsBoard.scramble` to
    generate a random *solvable* board.

    A special "stop" move is always available while ordinary presses remain.
    It does not change the board; instead it ends the game immediately,
    leaving any remaining lights on as the final score. This lets a player (or
    policy) bail out when no further press would help.

    Action codes:
        0 … n²-1  press cell (r, c), code = r·n + c
        n²        stop (end the game, board unchanged)
    """

    # The press stencil: the cell itself plus its four orthogonal neighbours.
    _STENCIL = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

    # Sentinel "stop" move: ends the game without touching the board.
    STOP = (-1, -1)

    def __init__(self, n: int, lights: set[tuple[int, int]] | None = None):
        if n < 2:
            raise ValueError("Board size must be at least 2")
        self.n = n
        self.pressed = set()
        self.stopped = False
        if lights is None:
            # Classic starting position: every light on.
            self.lights = {
                (row, col)
                for row in range(n)
                for col in range(n)
            }
        else:
            for pos in lights:
                if not self._in_bounds(*pos):
                    raise ValueError(f"light {pos} is not on the board")
            self.lights = set(lights)

    @classmethod
    def scramble(cls, n: int, k: int, seed: int | None = None) -> 'SquareLightsBoard':
        """Build a guaranteed-solvable board by pressing k random cells.

        Starting from the all-off (already-solved) state, apply k random
        presses. Because presses are reversible, the resulting position is
        always solvable (press the same cells again to undo it).
        """
        rng = np.random.default_rng(seed)
        board = cls(n, lights=set())  # all off
        # Toggle directly rather than via apply(): scramble presses are used
        # to build the starting position, so they must be allowed to repeat
        # and must not count against the in-game one-press-per-cell history.
        for _ in range(k):
            r = int(rng.integers(n))
            c = int(rng.integers(n))
            board._toggle((r, c))
        return board

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.n and 0 <= col < self.n

    def _toggle(self, pos: tuple[int, int]) -> None:
        """Flip the press stencil centred on pos. Does not record history."""
        r, c = pos
        for dr, dc in self._STENCIL:
            cell = (r + dr, c + dc)
            if not self._in_bounds(*cell):
                continue
            if cell in self.lights:
                self.lights.remove(cell)
            else:
                self.lights.add(cell)

    def available_moves(self) -> list[tuple[int, int]]:
        # Once stopped, the game is over and nothing is legal.
        if self.stopped:
            return []
        # Every in-bounds, not-yet-pressed cell is a legal press.
        presses = [
            (row, col)
            for row in range(self.n)
            for col in range(self.n)
            if self._in_bounds(row, col) and (row, col) not in self.pressed
        ]
        # Offer "stop" only while there is still something to press, so a board
        # with no presses left still terminates on its own.
        if presses:
            presses.append(self.STOP)
        return presses

    def apply(self, move: tuple[int, int]) -> None:
        """Press a cell, or end the game with the "stop" move.

        The "stop" move (``SquareLightsBoard.STOP``) leaves the board unchanged
        and ends the game immediately; any remaining lights stay on.
        """
        if move == self.STOP:
            if self.stopped:
                raise ValueError("Game has already been stopped")
            self.stopped = True
            return
        r, c = move
        if not self._in_bounds(r, c):
            raise ValueError(f"Press {move} is out of bounds")
        if (r, c) in self.pressed:
            raise ValueError(f"Cell {move} has already been pressed")
        self._toggle((r, c))
        self.pressed.add((r, c))

    def is_won(self) -> bool:
        return len(self.lights) == 0

    def is_over(self) -> bool:
        return self.is_won() or not self.available_moves()

    def score(self) -> int:
        """Number of lights still on (0 = solved). Lower is better."""
        return len(self.lights)

    def copy(self) -> 'SquareLightsBoard':
        b = SquareLightsBoard.__new__(SquareLightsBoard)
        b.n = self.n
        b.lights = self.lights.copy()
        b.pressed = self.pressed.copy()
        b.stopped = self.stopped
        return b

    def encode(self) -> np.ndarray:
        """Return an (n, n, 2) float32 tensor describing the position.

        Channel 0 — lights: 1.0 where a light is on, else 0.0.
        Channel 1 — in-bounds mask: 1.0 for playable cells, 0.0 for off-board
                    cells. For a plain SquareLightsBoard every cell is in
                    bounds, so channel 1 is all ones.
        """
        t = np.zeros((self.n, self.n, 2), dtype=np.float32)
        for row in range(self.n):
            for col in range(self.n):
                if self._in_bounds(row, col):
                    t[row, col, 1] = 1.0
                    if (row, col) in self.lights:
                        t[row, col, 0] = 1.0
        return t

    def encode_move(self, move: tuple[int, int]) -> int:
        if move == self.STOP:
            return self.n * self.n
        r, c = move
        return r * self.n + c

    def decode_move(self, code: int) -> tuple[int, int]:
        if code == self.n * self.n:
            return self.STOP
        return divmod(code, self.n)

    def __repr__(self) -> str:
        rows = []
        for row in range(self.n):
            cells = " ".join("#" if (row, col) in self.lights else "." for col in range(self.n))
            rows.append(cells)
        return "\n".join(rows)

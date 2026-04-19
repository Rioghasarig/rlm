import tkinter as tk
from tkinter import messagebox
import math
import threading
from board import Board, TriangularBoard, SquareBoard
from mcts import mcts as _mcts


class PegSolitaireGUI:
    """
    Peg solitaire GUI supporting triangular and square boards.

    Click a peg to select it (turns gold).  Valid landing squares turn green.
    Click a green square to jump.  Click the selected peg again to deselect.
    Clicking a different peg re-selects.
    """

    # ── colours ──────────────────────────────────────────────────────────────
    BG          = "#1e1e2e"
    LINE        = "#45475a"
    PEG_FILL    = "#89b4fa"   # blue pegs
    PEG_OUTLINE = "#1e66f5"
    SEL_FILL    = "#f9e2af"   # gold when selected
    SEL_OUTLINE = "#df8e1d"
    TGT_FILL    = "#a6e3a1"   # green valid targets
    TGT_OUTLINE = "#40a02b"
    HOLE_FILL   = "#313244"
    HOLE_OUTLINE= "#585b70"
    SUGG_FILL    = "#fab387"  # peach — MCTS suggested peg
    SUGG_OUTLINE = "#fe640b"
    SUGG_TGT_FILL    = "#89dceb"  # sky-blue — MCTS suggested landing
    SUGG_TGT_OUTLINE = "#04a5e5"

    MCTS_TIME = 2.0

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Peg Solitaire")
        self.root.configure(bg=self.BG)

        self.board: Board | None = None
        self.selected: tuple[int, int] | None = None
        self.valid_targets: dict[tuple, tuple] = {}
        self._history: list[set] = []
        self._suggestion: tuple | None = None
        self._mcts_running = False

        self._build_ui()
        self._new_game()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        ctrl = tk.Frame(self.root, bg=self.BG)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(10, 4))

        lbl_style = dict(bg=self.BG, fg="#cdd6f4", font=("Helvetica", 11))
        btn_style = dict(
            bg="#313244", fg="#cdd6f4", activebackground="#45475a",
            activeforeground="#cdd6f4", relief=tk.FLAT,
            font=("Helvetica", 11), padx=10, pady=4, cursor="hand2",
        )
        opt_style = dict(
            bg="#313244", fg="#cdd6f4", activebackground="#45475a",
            activeforeground="#cdd6f4", relief=tk.FLAT,
            font=("Helvetica", 11), highlightthickness=0,
        )

        tk.Label(ctrl, text="Board:", **lbl_style).pack(side=tk.LEFT)
        self.board_type_var = tk.StringVar(value="Triangular")
        board_menu = tk.OptionMenu(ctrl, self.board_type_var, "Triangular", "Square")
        board_menu.config(**opt_style)
        board_menu["menu"].config(bg="#313244", fg="#cdd6f4", activebackground="#45475a")
        board_menu.pack(side=tk.LEFT, padx=(4, 12))

        tk.Label(ctrl, text="Size:", **lbl_style).pack(side=tk.LEFT)
        self.size_var = tk.IntVar(value=5)
        size_spin = tk.Spinbox(
            ctrl, from_=2, to=10, textvariable=self.size_var, width=3,
            font=("Helvetica", 11), bg="#313244", fg="#cdd6f4",
            buttonbackground="#45475a", relief=tk.FLAT, justify=tk.CENTER,
        )
        size_spin.pack(side=tk.LEFT, padx=(4, 12))

        tk.Button(ctrl, text="New Game", command=self._new_game, **btn_style).pack(side=tk.LEFT)
        tk.Button(ctrl, text="Undo",     command=self._undo,     **btn_style).pack(side=tk.LEFT, padx=6)

        self._suggest_btn = tk.Button(
            ctrl, text="Suggest Move", command=self._suggest_move, **btn_style
        )
        self._suggest_btn.pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar()
        tk.Label(ctrl, textvariable=self.status_var, **lbl_style).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(self.root, bg=self.BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 12))
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", lambda _e: self._draw())

        self.root.minsize(480, 420)

    # ── game logic ────────────────────────────────────────────────────────────

    def _new_game(self) -> None:
        n = self.size_var.get()
        if self.board_type_var.get() == "Triangular":
            self.board = TriangularBoard(n)
        else:
            self.board = SquareBoard(n)
        self.selected = None
        self.valid_targets = {}
        self._history = []
        self._suggestion = None
        self._draw()
        self._update_status()

    def _undo(self) -> None:
        if not self._history:
            return
        self.board.pegs = self._history.pop()
        self.selected = None
        self.valid_targets = {}
        self._suggestion = None
        self._draw()
        self._update_status()

    # ── geometry helpers ─────────────────────────────────────────────────────

    def _spacing(self) -> float:
        w = self.canvas.winfo_width()  or 480
        h = self.canvas.winfo_height() or 420
        n = self.board.n
        usable = min(w * 0.88, h * 0.88)
        return usable / n

    def _hole_center(self, row: int, col: int) -> tuple[float, float]:
        sp = self._spacing()
        w  = self.canvas.winfo_width()  or 480
        h  = self.canvas.winfo_height() or 420
        n  = self.board.n

        if isinstance(self.board, TriangularBoard):
            tri_w = (n - 1) * sp
            tri_h = (n - 1) * sp * math.sqrt(3) / 2
            ox = (w - tri_w) / 2
            oy = (h - tri_h) / 2
            x = ox + col * sp + (n - 1 - row) * sp / 2
            y = oy + row * sp * math.sqrt(3) / 2
        else:  # SquareBoard
            grid_w = (n - 1) * sp
            grid_h = (n - 1) * sp
            ox = (w - grid_w) / 2
            oy = (h - grid_h) / 2
            x = ox + col * sp
            y = oy + row * sp

        return x, y

    def _radius(self) -> float:
        return self._spacing() * 0.34

    def _all_positions(self) -> list[tuple[int, int]]:
        n = self.board.n
        if isinstance(self.board, TriangularBoard):
            return [(row, col) for row in range(n) for col in range(row + 1)]
        else:
            return [(row, col) for row in range(n) for col in range(n)]

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        if self.board is None:
            return
        self.canvas.delete("all")

        n  = self.board.n
        r  = self._radius()
        sp = self._spacing()
        target_set = set(self.valid_targets)

        # Grid lines
        self._draw_grid(n, sp)

        sugg_from = self._suggestion[0] if self._suggestion else None
        sugg_to   = self._suggestion[2] if self._suggestion else None

        for pos in self._all_positions():
            row, col = pos
            x, y = self._hole_center(row, col)

            if pos in self.board.pegs:
                if pos == self.selected:
                    fill, outline, lw = self.SEL_FILL, self.SEL_OUTLINE, 3
                elif pos == sugg_from:
                    fill, outline, lw = self.SUGG_FILL, self.SUGG_OUTLINE, 3
                else:
                    fill, outline, lw = self.PEG_FILL, self.PEG_OUTLINE, 2
                self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                        fill=fill, outline=outline, width=lw)
            elif pos in target_set:
                self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                        fill=self.TGT_FILL, outline=self.TGT_OUTLINE, width=2)
            elif pos == sugg_to:
                self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                        fill=self.SUGG_TGT_FILL,
                                        outline=self.SUGG_TGT_OUTLINE, width=2)
            else:
                hr = r * 0.38
                self.canvas.create_oval(x - hr, y - hr, x + hr, y + hr,
                                        fill=self.HOLE_FILL, outline=self.HOLE_OUTLINE, width=1)

    def _draw_grid(self, n: int, sp: float) -> None:
        lw = max(1, sp * 0.05)
        if isinstance(self.board, TriangularBoard):
            dirs = [(0, 1), (1, 0), (1, 1)]
            for row in range(n):
                for col in range(row + 1):
                    x1, y1 = self._hole_center(row, col)
                    for dr, dc in dirs:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < n and 0 <= nc <= nr:
                            x2, y2 = self._hole_center(nr, nc)
                            self.canvas.create_line(x1, y1, x2, y2,
                                                    fill=self.LINE, width=lw)
        else:  # SquareBoard
            dirs = [(0, 1), (1, 0)]
            for row in range(n):
                for col in range(n):
                    x1, y1 = self._hole_center(row, col)
                    for dr, dc in dirs:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < n and 0 <= nc < n:
                            x2, y2 = self._hole_center(nr, nc)
                            self.canvas.create_line(x1, y1, x2, y2,
                                                    fill=self.LINE, width=lw)

    # ── interaction ───────────────────────────────────────────────────────────

    def _find_clicked(self, px: float, py: float) -> tuple[int, int] | None:
        r = self._radius()
        best, best_d = None, float("inf")
        for pos in self._all_positions():
            x, y = self._hole_center(*pos)
            d = math.hypot(px - x, py - y)
            if d < best_d:
                best_d, best = d, pos
        return best if best_d <= r * 1.4 else None

    def _on_click(self, event: tk.Event) -> None:
        if self.board is None:
            return
        pos = self._find_clicked(event.x, event.y)
        if pos is None:
            return

        if self.selected is None:
            if pos in self.board.pegs:
                self._select(pos)
        else:
            if pos == self.selected:
                self._deselect()
            elif pos in self.valid_targets:
                self._execute_move(pos)
            elif pos in self.board.pegs:
                self._select(pos)
            else:
                self._deselect()

    def _select(self, pos: tuple[int, int]) -> None:
        self.selected = pos
        moves = self.board.available_moves()
        self.valid_targets = {to: (fr, ov, to) for fr, ov, to in moves if fr == pos}
        self._draw()

    def _deselect(self) -> None:
        self.selected = None
        self.valid_targets = {}
        self._draw()

    def _execute_move(self, to_pos: tuple[int, int]) -> None:
        fr, _ov, to = self.valid_targets[to_pos]
        self._history.append(frozenset(self.board.pegs))
        self.board.move(fr, to)
        self.selected = None
        self.valid_targets = {}
        self._suggestion = None
        self._draw()
        self._update_status()
        self._check_game_over()

    # ── MCTS suggestion ──────────────────────────────────────────────────────

    def _suggest_move(self) -> None:
        if self.board is None or self._mcts_running:
            return
        if not self.board.available_moves():
            return

        self._mcts_running = True
        self._suggestion = None
        self._suggest_btn.config(state=tk.DISABLED, text="Thinking…")
        self.status_var.set("MCTS running…")
        self._draw()

        board_snap = self.board.copy()
        time_limit = self.MCTS_TIME

        def _run() -> None:
            move = _mcts(board_snap, time_limit=time_limit)
            self.root.after(0, lambda: self._on_suggestion(move))

        threading.Thread(target=_run, daemon=True).start()

    def _on_suggestion(self, move: tuple | None) -> None:
        self._mcts_running = False
        self._suggest_btn.config(state=tk.NORMAL, text="Suggest Move")
        self._suggestion = move
        self._update_status()
        self._draw()

    # ── status / end ─────────────────────────────────────────────────────────

    def _update_status(self) -> None:
        n = len(self.board.pegs)
        self.status_var.set(f"Pegs remaining: {n}")

    def _check_game_over(self) -> None:
        if self.board.is_won():
            messagebox.showinfo("You Win!",
                                "Brilliant! Only one peg remains.\nPress 'New Game' to play again.")
        elif not self.board.available_moves():
            n = len(self.board.pegs)
            messagebox.showinfo("Game Over",
                                f"No moves left — {n} peg{'s' if n != 1 else ''} remain.\n"
                                "Press 'Undo' to keep trying or 'New Game' to restart.")


def main() -> None:
    root = tk.Tk()
    root.geometry("600x560")
    PegSolitaireGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

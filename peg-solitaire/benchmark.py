"""
benchmark.py — Benchmark strategies on a square, triangular, or cross
peg-solitaire board and report per-board and aggregate results.

Instead of replaying a single standard starting position, every strategy is
evaluated on the *same* set of starting boards produced by
`build_initial_boards` (random descents from the standard start). This lets two
or more strategies be compared head-to-head on identical positions.

Strategies benchmarked:
  - MCTS  : fast_mcts with a configurable time limit per move (square only)
  - Random: uniformly random legal move selection
  - NN    : greedy policy network (loads a .keras checkpoint) (square/cross)
  - DFS   : fast_dfs depth-first search with a configurable depth limit

Usage:
    python benchmark.py [--n N] [--strategies dfs random ...]
    python benchmark.py --board triangular --strategies dfs random
    python benchmark.py --board cross --strategies dfs random
    python benchmark.py --strategies nn dfs --model policy_model.keras
    python benchmark.py --strategies dfs --max_depth 6 --max_breadth 4
    python benchmark.py --strategies dfs dfs_nn --dfs_model policy_model.keras
    python benchmark.py --strategies dfs random --csv results.csv
"""

import argparse
import csv
import io
import multiprocessing as mp
import random
import statistics
import time

from board import CrossBoard, SquareBoard, TriangularBoard
from fast_dfs import fast_dfs
from fast_imitation_learning_dfs import build_initial_boards
from fast_mcts_square import fast_mcts


def make_board(board_type: str, n: int):
    """Construct a fresh starting board of the requested type."""
    if board_type == "square":
        return SquareBoard(n)
    if board_type == "triangular":
        return TriangularBoard(n)
    if board_type == "cross":
        # The cross (English) board is a fixed 7×7 shape; n is ignored.
        return CrossBoard()
    raise ValueError(f"Unknown board type: {board_type}")


def play_game_mcts(board, time_limit: float) -> int:
    # fast_mcts is SquareBoard-specific (hardcoded n×n move table).
    board = board.copy()
    while True:
        move = fast_mcts(board, time_limit=time_limit)
        if move is None:
            break
        fr, ov, to = move
        board.move(fr, to)
    return len(board.pegs)


def play_game_nn(board, model_path: str) -> int:
    # The policy network targets SquareBoard and any subclass (e.g. CrossBoard),
    # which share encode()/encode_move(); supply a model sized to the board.
    import keras
    from policy_network_square import select_action
    model = keras.saving.load_model(model_path)
    board = board.copy()
    while True:
        moves = board.available_moves()
        if not moves:
            break
        move = select_action(model, board)
        fr, ov, to = move
        board.move(fr, to)
    return len(board.pegs)


def play_game_dfs(board, max_depth: int, q: int, max_breadth, policy=None) -> int:
    board = board.copy()
    while True:
        move = fast_dfs(board, max_depth=max_depth, q=q, max_breadth=max_breadth,
                        policy=policy)
        if move is None:
            break
        fr, ov, to = move
        board.move(fr, to)
    return len(board.pegs)


def play_game_random(board) -> int:
    board = board.copy()
    while True:
        moves = board.available_moves()
        if not moves:
            break
        fr, ov, to = random.choice(moves)
        board.move(fr, to)
    return len(board.pegs)


def _run_mcts_trial(args: tuple) -> tuple[int, int, float]:
    i, board, time_limit = args
    t0 = time.monotonic()
    remaining = play_game_mcts(board, time_limit)
    return i, remaining, time.monotonic() - t0


def _run_nn_trial(args: tuple) -> tuple[int, int, float]:
    i, board, model_path = args
    t0 = time.monotonic()
    remaining = play_game_nn(board, model_path)
    return i, remaining, time.monotonic() - t0


def _run_dfs_trial(args: tuple) -> tuple[int, int, float]:
    i, board, max_depth, q, max_breadth, policy = args
    t0 = time.monotonic()
    remaining = play_game_dfs(board, max_depth, q, max_breadth, policy)
    return i, remaining, time.monotonic() - t0


def _run_random_trial(args: tuple) -> tuple[int, int, float]:
    i, board = args
    t0 = time.monotonic()
    remaining = play_game_random(board)
    return i, remaining, time.monotonic() - t0


def run_trials_parallel(fn, trial_args: list, workers: int):
    """Yield (i, remaining, elapsed) results as each board finishes.

    Uses imap_unordered so completed boards stream out live (in completion
    order, not board order); the caller indexes results by i.
    """
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        yield from pool.imap_unordered(fn, trial_args)


def run_trials_sequential(fn, trial_args: list):
    """Yield (i, remaining, elapsed) results one board at a time, in order."""
    for a in trial_args:
        yield fn(a)


def report(label: str, results: list[int]) -> None:
    n = len(results)
    print(f"\n{label} — results over {n} board(s):")
    print(f"  min   : {min(results)}")
    print(f"  max   : {max(results)}")
    print(f"  mean  : {statistics.mean(results):.2f}")
    if n > 1:
        print(f"  stdev : {statistics.stdev(results):.2f}")
    print(f"  solved: {sum(r == 1 for r in results)}/{n}  (1 peg = solved)")


def write_csv(handle, initial_boards, strategy_results: dict) -> None:
    """Write one row per starting board with each strategy's pegs remaining."""
    strategies = list(strategy_results.keys())
    writer = csv.writer(handle)
    writer.writerow(["board", "initial_pegs"] + strategies)
    for idx, board in enumerate(initial_boards):
        row = [idx, len(board.pegs)]
        row += [strategy_results[s][idx] for s in strategies]
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board",      type=str,   default="square",
                        choices=["square", "triangular", "cross"],
                        help="Board geometry (default: square). The cross board "
                             "is the fixed 7×7 English board (n is ignored). "
                             "Note: mcts supports square only; nn/dfs_nn support "
                             "square and cross (provide a matching model).")
    parser.add_argument("--n",          type=int,   default=5,   help="Board side length")
    parser.add_argument("--time_limit", type=float, default=1.0, help="MCTS time limit per move (s)")
    parser.add_argument("--strategies", nargs="+",  default=["dfs", "random"],
                        choices=["mcts", "random", "nn", "dfs", "dfs_nn"],
                        help="One or more strategies to benchmark and compare "
                             "(default: dfs random). 'dfs' orders moves by "
                             "successor count; 'dfs_nn' orders them with the "
                             "--dfs_model policy network (square/cross boards). "
                             "List both to compare them head-to-head.")
    parser.add_argument("--model", type=str, default="policy_model.keras",
                        help="Path to .keras policy model for the nn strategy")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(),
                        help="Parallel worker processes (default: CPU count)")
    parser.add_argument("--max_depth", type=int, default=5,
                        help="DFS maximum search depth (default: 5)")
    parser.add_argument("--q", type=int, default=1,
                        help="DFS quiescence threshold — keep searching at depth limit if moves <= q (default: 1)")
    parser.add_argument("--max_breadth", type=int, default=None,
                        help="DFS maximum children expanded per node; the top-ranked subset is kept when a "
                             "node has more legal moves (default: None = expand all)")
    parser.add_argument("--dfs_model", type=str, default=None,
                        help="Path to the .keras policy model used by the 'dfs_nn' strategy, which orders "
                             "fast_dfs moves by NN likelihood instead of successor count (square/cross boards)")
    # build_initial_boards controls.
    parser.add_argument("--branching", type=int, default=5,
                        help="build_initial_boards: children per board (default: 5)")
    parser.add_argument("--moves_per_step", type=int, default=5,
                        help="build_initial_boards: random moves between levels (default: 5)")
    parser.add_argument("--depth", type=int, default=2,
                        help="build_initial_boards: number of descent levels (default: 2)")
    parser.add_argument("--seed", type=int, default=None,
                        help="build_initial_boards: RNG seed (default: None)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to write the per-board results CSV "
                             "(the CSV table is always printed to stdout)")
    args = parser.parse_args()

    board_type = args.board
    n, tl, workers = args.n, args.time_limit, args.workers
    model_path = args.model
    max_depth, q, max_breadth = args.max_depth, args.q, args.max_breadth

    # MCTS uses a hardcoded n×n move table, so it is SquareBoard-only. The
    # policy networks (nn, dfs_nn) work on any SquareBoard subclass — including
    # CrossBoard, which inherits encode()/encode_move() — so they also accept
    # the cross board (provide a matching --model / --dfs_model checkpoint).
    square_only = {"mcts"} & set(args.strategies)
    if board_type == "triangular":
        square_only |= {"nn", "dfs_nn"} & set(args.strategies)
    if board_type != "square" and square_only:
        parser.error(
            f"strategies {sorted(square_only)} support the square board only; "
            f"use --board square or drop them from --strategies"
        )
    if "dfs_nn" in args.strategies and not args.dfs_model:
        parser.error("strategy 'dfs_nn' requires --dfs_model (a .keras policy network)")

    # Preserve the order given on the command line, de-duplicated.
    strategies = list(dict.fromkeys(args.strategies))

    # The DFS policy network (if any) is loaded once and reused across boards;
    # the DFS strategy runs sequentially in this process, so no pickling needed.
    dfs_policy = None
    if "dfs_nn" in strategies:
        import keras
        dfs_policy = keras.saving.load_model(args.dfs_model)

    base_board = make_board(board_type, n)
    initial_boards = build_initial_boards(
        base_board,
        branching=args.branching,
        moves_per_step=args.moves_per_step,
        depth=args.depth,
        seed=args.seed,
    )
    n_boards = len(initial_boards)
    w = len(str(n_boards))

    if board_type == "square":
        shape = f"{n}×{n}"
    elif board_type == "cross":
        shape = "7×7 English"
    else:
        shape = f"side {n}"
    print(f"Board: {board_type} ({shape})  |  boards: {n_boards}  |  "
          f"strategies: {' '.join(strategies)}  |  workers: {workers}\n")

    strategy_results: dict[str, list[int]] = {}

    for s in strategies:
        if strategy_results:
            print()

        if s == "mcts":
            print(f"[MCTS]  time/move: {tl}s  (running {n_boards} board(s) in parallel…)")
            trial_args = [(i, b, tl) for i, b in enumerate(initial_boards)]
            results = run_trials_parallel(_run_mcts_trial, trial_args, workers)
        elif s == "nn":
            print(f"[NN]  model: {model_path}  (running {n_boards} board(s) in parallel…)")
            trial_args = [(i, b, model_path) for i, b in enumerate(initial_boards)]
            results = run_trials_parallel(_run_nn_trial, trial_args, workers)
        elif s in ("dfs", "dfs_nn"):
            policy = dfs_policy if s == "dfs_nn" else None
            ordering = f"nn ({args.dfs_model})" if policy is not None else "successor-count"
            print(f"[{s.upper()}]  max_depth: {max_depth}  q: {q}  max_breadth: {max_breadth}  "
                  f"ordering: {ordering}  (running {n_boards} board(s) sequentially…)")
            trial_args = [(i, b, max_depth, q, max_breadth, policy)
                          for i, b in enumerate(initial_boards)]
            results = run_trials_sequential(_run_dfs_trial, trial_args)
        elif s == "random":
            print(f"[Random]  (uniform random legal move — running {n_boards} board(s) in parallel…)")
            trial_args = [(i, b) for i, b in enumerate(initial_boards)]
            results = run_trials_parallel(_run_random_trial, trial_args, workers)
        else:
            raise ValueError(f"Unknown strategy: {s}")

        per_board = [0] * n_boards
        for i, remaining, elapsed in results:
            per_board[i] = remaining
            print(f"  board {i:>{w}}: {remaining} pegs remaining  ({elapsed:.1f}s)",
                  flush=True)
        strategy_results[s] = per_board
        report(s.upper(), per_board)

    # Per-board results as a CSV table.
    print("\nPer-board results (CSV):")
    buf = io.StringIO()
    write_csv(buf, initial_boards, strategy_results)
    print(buf.getvalue(), end="")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            write_csv(f, initial_boards, strategy_results)
        print(f"\nWrote per-board CSV to {args.csv}")


if __name__ == "__main__":
    main()

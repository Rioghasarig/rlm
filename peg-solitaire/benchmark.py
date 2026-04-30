"""
benchmark.py — Benchmark strategies on an n×n SquareBoard and report
the average number of pegs remaining across multiple trials.

Strategies benchmarked:
  - MCTS  : fast_mcts with a configurable time limit per move
  - Random: uniformly random legal move selection
  - NN    : greedy policy network (loads a .keras checkpoint)
  - DFS   : fast_dfs depth-first search with a configurable depth limit

Usage:
    python benchmark.py [--n N] [--trials T] [--time_limit S]
    python benchmark.py --strategies nn --model policy_model.keras
    python benchmark.py --strategies dfs --max_depth 6 --q 1
"""

import argparse
import multiprocessing as mp
import random
import statistics
import time

from board import SquareBoard
from fast_dfs import fast_dfs
from fast_mcts_square import fast_mcts


def play_game_mcts(n: int, time_limit: float) -> int:
    board = SquareBoard(n)
    while True:
        move = fast_mcts(board, time_limit=time_limit)
        if move is None:
            break
        fr, ov, to = move
        board.move(fr, to)
    return len(board.pegs)


def play_game_nn(n: int, model_path: str) -> int:
    import keras
    from policy_network_square import select_action
    model = keras.saving.load_model(model_path)
    board = SquareBoard(n)
    while True:
        moves = board.available_moves()
        if not moves:
            break
        move = select_action(model, board)
        fr, ov, to = move
        board.move(fr, to)
    return len(board.pegs)


def play_game_dfs(n: int, max_depth: int, q: int, n_workers: int) -> int:
    board = SquareBoard(n)
    while True:
        move = fast_dfs(board, max_depth=max_depth, q=q, n_workers=n_workers)
        if move is None:
            break
        fr, ov, to = move
        board.move(fr, to)
    return len(board.pegs)


def play_game_random(n: int) -> int:
    board = SquareBoard(n)
    while True:
        moves = board.available_moves()
        if not moves:
            break
        fr, ov, to = random.choice(moves)
        board.move(fr, to)
    return len(board.pegs)


def _run_mcts_trial(args: tuple) -> tuple[int, int, float]:
    i, n, time_limit = args
    t0 = time.monotonic()
    remaining = play_game_mcts(n, time_limit)
    return i, remaining, time.monotonic() - t0


def _run_nn_trial(args: tuple) -> tuple[int, int, float]:
    i, n, model_path = args
    t0 = time.monotonic()
    remaining = play_game_nn(n, model_path)
    return i, remaining, time.monotonic() - t0


def _run_dfs_trial(args: tuple) -> tuple[int, int, float]:
    i, n, max_depth, q, n_workers = args
    t0 = time.monotonic()
    remaining = play_game_dfs(n, max_depth, q, n_workers)
    return i, remaining, time.monotonic() - t0


def _run_random_trial(args: tuple) -> tuple[int, int, float]:
    i, n = args
    t0 = time.monotonic()
    remaining = play_game_random(n)
    return i, remaining, time.monotonic() - t0


def run_trials_parallel(fn, trial_args: list, workers: int) -> list[tuple[int, int, float]]:
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(fn, trial_args)
    results.sort(key=lambda r: r[0])
    return results


def report(label: str, results: list[int], trials: int) -> None:
    print(f"\n{label} — results over {trials} trial(s):")
    print(f"  min   : {min(results)}")
    print(f"  max   : {max(results)}")
    print(f"  mean  : {statistics.mean(results):.2f}")
    if trials > 1:
        print(f"  stdev : {statistics.stdev(results):.2f}")
    print(f"  solved: {sum(r == 1 for r in results)}/{trials}  (1 peg = solved)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int,   default=5,   help="Board side length")
    parser.add_argument("--trials",     type=int,   default=5,   help="Number of games per strategy")
    parser.add_argument("--time_limit", type=float, default=1.0, help="MCTS time limit per move (s)")
    parser.add_argument("--strategies", nargs="+",  default=["mcts", "random"],
                        choices=["mcts", "random", "nn", "dfs"],
                        help="Strategies to benchmark (default: mcts random)")
    parser.add_argument("--model", type=str, default="policy_model.keras",
                        help="Path to .keras policy model for the nn strategy")
    parser.add_argument("--workers", type=int, default=mp.cpu_count(),
                        help="Parallel worker processes (default: CPU count)")
    parser.add_argument("--max_depth", type=int, default=5,
                        help="DFS maximum search depth (default: 5)")
    parser.add_argument("--q", type=int, default=1,
                        help="DFS quiescence threshold — keep searching at depth limit if moves <= q (default: 1)")
    args = parser.parse_args()

    n, trials, tl, workers = args.n, args.trials, args.time_limit, args.workers
    model_path = args.model
    max_depth, q = args.max_depth, args.q
    initial_pegs = n * n - 1
    w = len(str(trials))

    print(f"Board: {n}×{n}  |  initial pegs: {initial_pegs}  |  trials: {trials}  |  workers: {workers}\n")

    first = True
    if "mcts" in args.strategies:
        first = False
        print(f"[MCTS]  time/move: {tl}s  (running {trials} trial(s) in parallel…)")
        trial_args = [(i, n, tl) for i in range(1, trials + 1)]
        results = run_trials_parallel(_run_mcts_trial, trial_args, workers)
        mcts_results = []
        for i, remaining, elapsed in results:
            mcts_results.append(remaining)
            print(f"  game {i:>{w}}: {remaining} pegs remaining  ({elapsed:.1f}s)")
        report("MCTS", mcts_results, trials)

    if "nn" in args.strategies:
        if not first:
            print()
        first = False
        print(f"[NN]  model: {model_path}  (running {trials} trial(s) in parallel…)")
        trial_args = [(i, n, model_path) for i in range(1, trials + 1)]
        results = run_trials_parallel(_run_nn_trial, trial_args, workers)
        nn_results = []
        for i, remaining, elapsed in results:
            nn_results.append(remaining)
            print(f"  game {i:>{w}}: {remaining} pegs remaining  ({elapsed:.1f}s)")
        report("NN", nn_results, trials)

    if "dfs" in args.strategies:
        if not first:
            print()
        first = False
        print(f"[DFS]  max_depth: {max_depth}  q: {q}  (running {trials} trial(s) sequentially, {workers} internal workers…)")
        dfs_results = []
        for i in range(1, trials + 1):
            trial_i, remaining, elapsed = _run_dfs_trial((i, n, max_depth, q, workers))
            dfs_results.append(remaining)
            print(f"  game {i:>{w}}: {remaining} pegs remaining  ({elapsed:.1f}s)")
        report("DFS", dfs_results, trials)

    if "random" in args.strategies:
        if not first:
            print()
        print(f"[Random]  (uniform random legal move — running {trials} trial(s) in parallel…)")
        trial_args = [(i, n) for i in range(1, trials + 1)]
        results = run_trials_parallel(_run_random_trial, trial_args, workers)
        rand_results = []
        for i, remaining, elapsed in results:
            rand_results.append(remaining)
            print(f"  game {i:>{w}}: {remaining} pegs remaining  ({elapsed:.1f}s)")
        report("Random", rand_results, trials)


if __name__ == "__main__":
    main()

"""
Benchmark A* on random Lights Out boards at two `max_expansions` settings.

Generates a collection of random *solvable* boards via
`SquareLightsBoard.scramble` and runs `astar` on each board twice — once at a
"low" expansion budget and once at a "high" one — side by side on the *same*
board, so the two settings are compared on identical inputs. Per-board results
are written to a CSV file and summary statistics are printed to stdout.

Usage:
    python benchmark_astar.py [options]

    --n N                 board size (default 5)
    --scramble K          number of random presses per board (default 10)
    --num-boards M        number of boards to generate (default 100)
    --low-expansions L    "low" max_expansions budget (default 100)
    --high-expansions H   "high" max_expansions budget (default 10000)
    --seed S              base RNG seed (default 0)
    --csv PATH            output CSV path (default benchmark_astar_results.csv)
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time

from board import SquareLightsBoard
from astar import astar


def evaluate(board: SquareLightsBoard, max_expansions: int) -> dict:
    """Greedily follow A* on a copy of *board* and report how it did.

    astar returns a single best next move, so we apply it and re-query until it
    says STOP (or the board is solved). Returns a dict with the starting score,
    the score reached, the number of lights cleared, whether the board was fully
    solved, the solution length (number of presses made), and the wall-clock
    time taken for the whole rollout.
    """
    start_score = board.score()

    final = board.copy()
    t0 = time.perf_counter()
    n_presses = 0
    while True:
        move = astar(final, max_expansions=max_expansions)
        if move == SquareLightsBoard.STOP:
            break
        final.apply(move)
        n_presses += 1
    elapsed = time.perf_counter() - t0

    final_score = final.score()

    return {
        "start_score": start_score,
        "final_score": final_score,
        "cleared": start_score - final_score,
        "solved": final.is_won(),
        "solution_length": n_presses,
        "time_s": elapsed,
    }


def run_benchmark(
    n: int,
    scramble_k: int,
    num_boards: int,
    low_expansions: int,
    high_expansions: int,
    seed: int,
) -> list[dict]:
    """Generate `num_boards` random boards and evaluate both settings on each."""
    results: list[dict] = []
    for i in range(num_boards):
        # Distinct, reproducible seed per board.
        board = SquareLightsBoard.scramble(n, scramble_k, seed=seed + i)

        low = evaluate(board, low_expansions)
        high = evaluate(board, high_expansions)

        print(
            f"[{i + 1:>{len(str(num_boards))}}/{num_boards}] "
            f"start={low['start_score']:>3} | "
            f"low: cleared={low['cleared']:>3} solved={int(low['solved'])} "
            f"t={low['time_s']:.4f}s | "
            f"high: cleared={high['cleared']:>3} solved={int(high['solved'])} "
            f"t={high['time_s']:.4f}s",
            flush=True,
        )

        row = {
            "board_index": i,
            "n": n,
            "scramble_k": scramble_k,
            "start_score": low["start_score"],
        }
        for label, res in (("low", low), ("high", high)):
            row[f"{label}_final_score"] = res["final_score"]
            row[f"{label}_cleared"] = res["cleared"]
            row[f"{label}_solved"] = res["solved"]
            row[f"{label}_solution_length"] = res["solution_length"]
            row[f"{label}_time_s"] = res["time_s"]
        results.append(row)
    return results


def _summarize(results: list[dict], label: str) -> dict:
    """Aggregate stats for one setting ("low" or "high") across all boards."""
    final_scores = [r[f"{label}_final_score"] for r in results]
    cleared = [r[f"{label}_cleared"] for r in results]
    times = [r[f"{label}_time_s"] for r in results]
    solved = [r[f"{label}_solved"] for r in results]
    return {
        "solved_count": sum(solved),
        "solved_rate": sum(solved) / len(results),
        "mean_final_score": statistics.mean(final_scores),
        "mean_cleared": statistics.mean(cleared),
        "total_time_s": sum(times),
        "mean_time_s": statistics.mean(times),
    }


def print_summary(
    results: list[dict], low_expansions: int, high_expansions: int
) -> None:
    n_boards = len(results)
    mean_start = statistics.mean(r["start_score"] for r in results)
    print(f"\nBenchmark over {n_boards} boards "
          f"(mean starting lights: {mean_start:.2f})")
    print("=" * 72)
    header = f"{'metric':<22}{'low (' + str(low_expansions) + ')':>24}" \
             f"{'high (' + str(high_expansions) + ')':>24}"
    print(header)
    print("-" * 72)

    low = _summarize(results, "low")
    high = _summarize(results, "high")
    rows = [
        ("solved", f"{low['solved_count']}/{n_boards} "
                   f"({low['solved_rate']:.1%})",
                   f"{high['solved_count']}/{n_boards} "
                   f"({high['solved_rate']:.1%})"),
        ("mean final lights", f"{low['mean_final_score']:.3f}",
                              f"{high['mean_final_score']:.3f}"),
        ("mean lights cleared", f"{low['mean_cleared']:.3f}",
                                f"{high['mean_cleared']:.3f}"),
        ("mean time (s)", f"{low['mean_time_s']:.5f}",
                          f"{high['mean_time_s']:.5f}"),
        ("total time (s)", f"{low['total_time_s']:.3f}",
                           f"{high['total_time_s']:.3f}"),
    ]
    for name, lo, hi in rows:
        print(f"{name:<22}{lo:>24}{hi:>24}")
    print("=" * 72)


def save_csv(results: list[dict], path: str) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--scramble", type=int, default=10, dest="scramble_k")
    parser.add_argument("--num-boards", type=int, default=100)
    parser.add_argument("--low-expansions", type=int, default=100)
    parser.add_argument("--high-expansions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", default="benchmark_astar_results.csv")
    args = parser.parse_args()

    results = run_benchmark(
        n=args.n,
        scramble_k=args.scramble_k,
        num_boards=args.num_boards,
        low_expansions=args.low_expansions,
        high_expansions=args.high_expansions,
        seed=args.seed,
    )
    print_summary(results, args.low_expansions, args.high_expansions)
    save_csv(results, args.csv)


if __name__ == "__main__":
    main()

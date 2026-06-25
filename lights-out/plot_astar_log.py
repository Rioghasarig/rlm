#!/usr/bin/env python3
"""Plot training progress logged by fast_imitation_learning_astar.py.

The trainer writes a JSONL file (one record per line) to its ``log_dir``.
This script reads such a file and renders a multi-panel summary figure.

Usage:
    python plot_astar_log.py                      # newest *.jsonl under runs/
    python plot_astar_log.py runs/<file>.jsonl    # a specific log
    python plot_astar_log.py <file> -o out.png    # save instead of show
    python plot_astar_log.py <file> --show        # also open a window when saving

Record types produced by the trainer (see _record / record_fn calls):
    run_start, iteration_start, trajectory, labeling, labeling_summary,
    epoch_loss, iteration_end, run_end
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def by_type(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r.get("type", "?")].append(r)
    return groups


def find_latest_log(log_dir: str = "runs") -> str | None:
    files = glob.glob(os.path.join(log_dir, "*.jsonl"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _mean_per_iter(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    """Average ``key`` over rows sharing the same iteration."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if key in r and r[key] is not None:
            buckets[r["iteration"]].append(r[key])
    its = sorted(buckets)
    return its, [float(np.mean(buckets[i])) for i in its]


def plot(records: list[dict], title: str):
    g = by_type(records)
    cfg = g["run_start"][0] if g["run_start"] else {}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=13)

    # 1. Training loss across all epochs (flattened over iterations).
    ax = axes[0, 0]
    losses = g["epoch_loss"]
    if losses:
        ys = [r["loss"] for r in losses]
        ax.plot(range(1, len(ys) + 1), ys, lw=1, color="tab:red")
        # Mark iteration boundaries.
        epochs_per_iter = cfg.get("epochs")
        if epochs_per_iter:
            for x in range(epochs_per_iter, len(ys), epochs_per_iter):
                ax.axvline(x + 0.5, color="0.85", lw=0.6, zorder=0)
    ax.set_title("Training loss")
    ax.set_xlabel("epoch (cumulative over iterations)")
    ax.set_ylabel("loss")

    # 2. Lights remaining at end of rollouts (mean per iteration) — the
    #    headline "are we solving boards?" signal. Lower is better; 0 = solved.
    ax = axes[0, 1]
    traj = g["trajectory"]
    if traj:
        its, mean_rem = _mean_per_iter(traj, "lights_remaining")
        ax.plot(its, mean_rem, "-o", ms=3, color="tab:blue", label="mean")
        # Fraction of rollouts fully solved (lights_remaining == 0).
        solved: dict[int, list[float]] = defaultdict(list)
        for r in traj:
            solved[r["iteration"]].append(1.0 if r["lights_remaining"] == 0 else 0.0)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Rollout lights remaining (mean / iter)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("lights remaining")

    # 3. Fraction of rollouts solved per iteration.
    ax = axes[0, 2]
    if traj:
        solved = defaultdict(list)
        for r in traj:
            solved[r["iteration"]].append(1.0 if r["lights_remaining"] == 0 else 0.0)
        its = sorted(solved)
        frac = [float(np.mean(solved[i])) for i in its]
        ax.plot(its, frac, "-o", ms=3, color="tab:green")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Fraction of rollouts solved")
    ax.set_xlabel("iteration")
    ax.set_ylabel("solved fraction")

    # 4. Dataset size growth (from iteration_end records).
    ax = axes[1, 0]
    ie = g["iteration_end"]
    if ie:
        its = [r["iteration"] for r in ie]
        ax.plot(its, [r["dataset_size"] for r in ie], "-o", ms=3,
                color="tab:purple", label="dataset size")
        ax.plot(its, [r["new_samples"] for r in ie], "-o", ms=3,
                color="tab:orange", label="new samples/iter")
        ax.legend(loc="best", fontsize=8)
    ax.set_title("Dataset size")
    ax.set_xlabel("iteration")
    ax.set_ylabel("samples")

    # 5. Labeling: states generated and labeled-vs-skipped per iteration.
    ax = axes[1, 1]
    lab = g["labeling"]
    if lab:
        labeled, skipped = defaultdict(int), defaultdict(int)
        for r in lab:
            labeled[r["iteration"]] += r["labeled"]
            skipped[r["iteration"]] += r["skipped"]
        its = sorted(set(labeled) | set(skipped))
        lvals = [labeled[i] for i in its]
        svals = [skipped[i] for i in its]
        ax.bar(its, lvals, color="tab:green", label="labeled")
        ax.bar(its, svals, bottom=lvals, color="tab:gray", label="skipped (no soln)")
        ax.legend(loc="best", fontsize=8)
    ax.set_title("States labeled vs skipped")
    ax.set_xlabel("iteration")
    ax.set_ylabel("states")

    # 6. Time breakdown per iteration: generation, labeling, training.
    ax = axes[1, 2]
    ls = {r["iteration"]: r for r in g["labeling_summary"]}
    iemap = {r["iteration"]: r for r in ie}
    its = sorted(set(ls) | set(iemap))
    if its:
        gen = np.array([ls.get(i, {}).get("gen_time_s", 0.0) for i in its])
        label = np.array([ls.get(i, {}).get("label_time_s", 0.0) for i in its])
        train = np.array([iemap.get(i, {}).get("train_time_s", 0.0) for i in its])
        ax.bar(its, gen, color="tab:blue", label="gen")
        ax.bar(its, label, bottom=gen, color="tab:orange", label="label")
        ax.bar(its, train, bottom=gen + label, color="tab:red", label="train")
        ax.legend(loc="best", fontsize=8)
    ax.set_title("Time per iteration (s)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("seconds")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _summary_line(records: list[dict]) -> str:
    g = by_type(records)
    cfg = g["run_start"][0] if g["run_start"] else {}
    parts = []
    if cfg:
        parts.append(f"n={cfg.get('board_n')}")
        parts.append(f"iters={cfg.get('n_iterations')}")
        parts.append(f"boards={cfg.get('n_initial_boards')}")
    if g["run_end"]:
        parts.append(f"total={g['run_end'][0].get('total_time_s')}s")
    return "  ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="Path to JSONL log (default: newest in runs/)")
    ap.add_argument("-o", "--output", help="Save figure to this path instead of showing")
    ap.add_argument("--show", action="store_true", help="Show window even when saving")
    ap.add_argument("--log-dir", default="runs", help="Directory to search when no log given")
    args = ap.parse_args()

    path = args.log or find_latest_log(args.log_dir)
    if not path:
        ap.error(f"no log given and no *.jsonl found in {args.log_dir}/")
    if not os.path.exists(path):
        ap.error(f"file not found: {path}")

    records = load_records(path)
    print(f"Loaded {len(records)} records from {path}")
    print(f"  {_summary_line(records)}")

    title = f"{os.path.basename(path)}    {_summary_line(records)}"
    fig = plot(records, title)

    if args.output:
        fig.savefig(args.output, dpi=120)
        print(f"Saved → {args.output}")
        if args.show:
            plt.show()
    else:
        plt.show()


if __name__ == "__main__":
    main()

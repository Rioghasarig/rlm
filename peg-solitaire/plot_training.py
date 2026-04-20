import argparse
import pandas as pd
import matplotlib.pyplot as plt

JSONL = "runs/20260419_081743_n10.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--jsonl", default=JSONL)
parser.add_argument("--plot", choices=["loss", "pegs", "both"], default="loss",
                    help="Which metric to plot (default: loss)")
args = parser.parse_args()

df = pd.read_json(args.jsonl, lines=True)


def plot_loss(ax):
    losses = df[df.type == "epoch_loss"][["iteration", "epoch", "loss"]].reset_index(drop=True)

    global_epoch = []
    offset = 0
    for iteration, group in losses.groupby("iteration", sort=True):
        n = len(group)
        global_epoch.extend(range(offset + 1, offset + n + 1))
        offset += n
    losses = losses.copy()
    losses["global_epoch"] = global_epoch

    ax.plot(losses["global_epoch"], losses["loss"], linewidth=1.5)

    boundaries = losses.groupby("iteration")["global_epoch"].min()
    for it, ep in boundaries.items():
        ax.axvline(ep, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
        ax.text(ep + 0.1, ax.get_ylim()[1], f"iter {it}", fontsize=6, color="gray", va="top")

    ax.set_xlabel("Global epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Epoch loss")
    ax.grid(True, alpha=0.3)


def plot_pegs(ax):
    trajs = df[df.type == "trajectory"][["iteration", "trajectory_idx", "pegs_remaining"]].reset_index(drop=True)

    # Scatter individual trajectories
    ax.scatter(trajs["iteration"], trajs["pegs_remaining"], alpha=0.5, s=20, label="trajectory")

    # Mean per iteration
    mean_pegs = trajs.groupby("iteration")["pegs_remaining"].mean()
    ax.plot(mean_pegs.index, mean_pegs.values, color="tab:red", linewidth=1.5, marker="o", markersize=4, label="mean")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Pegs remaining")
    ax.set_title("Policy quality (pegs remaining — lower is better)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


if args.plot == "both":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    plot_loss(ax1)
    plot_pegs(ax2)
    fig.suptitle(args.jsonl)
else:
    fig, ax = plt.subplots(figsize=(10, 5))
    if args.plot == "loss":
        plot_loss(ax)
    else:
        plot_pegs(ax)
    ax.set_title(f"{ax.get_title()} — {args.jsonl}")

plt.tight_layout()
out = args.jsonl.replace(".jsonl", f"_{args.plot}.png")
plt.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()

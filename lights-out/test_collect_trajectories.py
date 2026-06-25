"""
Functional test for `_collect_trajectories` (the rollout + A*-labeling pass).

Builds an initial-board set and a policy network, runs one collection pass, and
validates the samples it produces:

  * every sample is an (encoded-state, action-code) pair with the right shape,
    dtype, and an action code in the legal range [0, n*n] (n*n == STOP);
  * the recorded progress events line up with the rollout/labeling phases;
  * the STOP action actually shows up as a label (the whole point of the new
    STOP handling — the teacher tells the policy when to bail).

Parameters default to config_astar.yaml but every relevant one can be overridden
on the command line, e.g.

    python test_collect_trajectories.py --n 4 --num-boards 5 --n-trajectories 1 \
        --astar-max-expansions 2000 --n-workers 4
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np

from board import SquareLightsBoard
from fast_imitation_learning_astar import (
    build_initial_boards,
    _collect_trajectories,
    _load_config,
)


def _resolve(cli_value, cfg_value):
    """CLI override wins when provided (not None); otherwise fall back to config."""
    return cli_value if cli_value is not None else cfg_value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="config_astar.yaml",
                   help="YAML config supplying defaults for everything below")
    # Board / initial-board set
    p.add_argument("--n", type=int, help="board side length (board.n)")
    p.add_argument("--num-boards", type=int,
                   help="number of distinct starting boards (board.init_set.num_boards)")
    p.add_argument("--scramble-k", type=int,
                   help="random presses per scramble (board.init_set.scramble_k)")
    p.add_argument("--seed", type=int,
                   help="RNG seed for board-set construction (board.init_set.seed)")
    # Network
    p.add_argument("--res-blocks", type=int, help="policy tower residual blocks")
    p.add_argument("--filters", type=int, help="conv filters per block")
    p.add_argument("--load-path", help="load policy from this .keras instead of building fresh")
    # Collection
    p.add_argument("--n-trajectories", type=int,
                   help="rollouts per initial board (dagger.n_trajectories)")
    p.add_argument("--astar-max-expansions", type=int,
                   help="A* teacher node budget (dagger.astar_max_expansions)")
    p.add_argument("--n-workers", type=int,
                   help="labeling worker processes (dagger.n_workers)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    bc, nc, dc = cfg["board"], cfg["network"], cfg["dagger"]
    isc = bc.get("init_set") or {}

    # Resolve every parameter: command line overrides config.
    n               = _resolve(args.n, bc["n"])
    num_boards      = _resolve(args.num_boards, isc.get("num_boards", 30))
    scramble_k      = _resolve(args.scramble_k, isc.get("scramble_k", 10))
    seed            = _resolve(args.seed, isc.get("seed"))
    res_blocks      = _resolve(args.res_blocks, nc["res_blocks"])
    filters         = _resolve(args.filters, nc["filters"])
    load_path       = _resolve(args.load_path, dc.get("load_path"))
    n_trajectories  = _resolve(args.n_trajectories, dc.get("n_trajectories", 1))
    max_expansions  = _resolve(args.astar_max_expansions, dc.get("astar_max_expansions"))
    n_workers       = _resolve(args.n_workers, dc.get("n_workers")) or (os.cpu_count() or 1)

    stop_code = n * n  # encode_move(STOP)

    print("Parameters (config + overrides):")
    for k, v in [
        ("n", n), ("num_boards", num_boards), ("scramble_k", scramble_k), ("seed", seed),
        ("res_blocks", res_blocks), ("filters", filters), ("load_path", load_path),
        ("n_trajectories", n_trajectories), ("astar_max_expansions", max_expansions),
        ("n_workers", n_workers),
    ]:
        print(f"  {k:<22}= {v}")
    print()

    # ── Build inputs ──────────────────────────────────────────────────────────
    initial_boards = build_initial_boards(
        n, num_boards=num_boards, scramble_k=scramble_k, seed=seed,
    )
    print(f"Built {len(initial_boards)} initial boards "
          f"(mean starting lights: {np.mean([b.score() for b in initial_boards]):.2f})")

    import keras  # imported late so --help stays fast
    if load_path:
        print(f"Loading policy from {load_path}")
        pi = keras.models.load_model(load_path)
    else:
        from policy_network_square import build_square_policy_network
        pi = build_square_policy_network(n, res_blocks=res_blocks, filters=filters)
    assert pi.output_shape[-1] == n * n + 1, (
        f"policy output {pi.output_shape[-1]} != n*n+1={n * n + 1}; "
        f"network must include the STOP logit"
    )

    # Capture progress events instead of writing a log file.
    events: list[dict] = []
    record_fn = lambda **fields: events.append(fields)

    # ── Run the function under test ───────────────────────────────────────────
    print("\nRunning _collect_trajectories ...\n")
    samples = _collect_trajectories(
        pi, initial_boards, n_trajectories,
        max_expansions, n_workers,
        record_fn=record_fn, iteration=1,
    )

    # ── Validate the samples ──────────────────────────────────────────────────
    print("\nValidating samples ...")
    assert isinstance(samples, list), f"samples should be a list, got {type(samples)}"
    assert samples, "no samples produced — collection returned nothing"

    action_counts: Counter = Counter()
    for i, sample in enumerate(samples):
        assert isinstance(sample, tuple) and len(sample) == 2, \
            f"sample {i} is not a (state, action) pair: {sample!r}"
        state, action = sample
        assert isinstance(state, np.ndarray), f"sample {i} state is {type(state)}"
        assert state.shape == (n, n, 2), f"sample {i} state shape {state.shape} != {(n, n, 2)}"
        assert state.dtype == np.float32, f"sample {i} state dtype {state.dtype}"
        assert isinstance(action, (int, np.integer)), \
            f"sample {i} action is {type(action)}"
        assert 0 <= action <= stop_code, \
            f"sample {i} action {action} out of range [0, {stop_code}]"
        action_counts[int(action)] += 1

    n_stop = action_counts.get(stop_code, 0)
    n_press = len(samples) - n_stop

    print(f"  ✓ {len(samples)} samples, all well-formed "
          f"(state {(n, n, 2)} float32, action in [0, {stop_code}])")
    print(f"  press labels: {n_press}   STOP labels: {n_stop}   "
          f"distinct action codes: {len(action_counts)}")

    # ── Validate the recorded progress events ─────────────────────────────────
    by_type = Counter(e.get("type") for e in events)
    print(f"\nProgress events: {dict(by_type)}")
    assert by_type["trajectory"] == len(initial_boards) * n_trajectories, \
        "one 'trajectory' event expected per rollout"
    assert by_type["labeling"] == len(initial_boards), \
        "one 'labeling' event expected per board"
    assert by_type["labeling_summary"] == 1, "expected a single labeling_summary event"
    total_labeled = sum(e["labeled"] for e in events if e.get("type") == "labeling")
    assert total_labeled == len(samples), \
        f"labeling events report {total_labeled} labeled but got {len(samples)} samples"
    print("  ✓ event counts consistent with boards × rollouts and sample total")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

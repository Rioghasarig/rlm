"""
compare_nn_mcts.py — Play one greedy NN rollout and compare NN vs MCTS move
rankings at every step.

For each state on the rollout a side-by-side table shows:
  Left  : top-k moves ranked by NN softmax probability (* = chosen move)
  Right : top-k moves ranked by MCTS UCB1 score (with mean Q-value)

Usage:
    python compare_nn_mcts.py [--n N] [--model PATH] [--time_limit S]
                              [--top_k K] [--seed S]
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import jax
import jax.numpy as jnp
import keras

from board import SquareBoard
from policy_network_square import encode_board
from fast_mcts_square import (
    fast_mcts,
    _board_to_flat,
    _move_table,
    _run_rollout,
)


# ── NN helpers ────────────────────────────────────────────────────────────────

def nn_ranked_moves(
    model: keras.Model,
    board: SquareBoard,
    top_k: int = 5,
) -> list[tuple[tuple, float]]:
    """Return up to *top_k* legal moves sorted by descending softmax probability."""
    legal_moves = board.available_moves()
    if not legal_moves:
        return []

    logits = model(encode_board(board), training=False)[0].numpy()
    codes = [board.encode_move(m) for m in legal_moves]
    lg = logits[codes]
    lg -= lg.max()
    probs = np.exp(lg)
    probs /= probs.sum()

    ranked = sorted(zip(legal_moves, probs.tolist()), key=lambda x: -x[1])
    return ranked[:top_k]


# ── MCTS helpers ──────────────────────────────────────────────────────────────

def mcts_ranked_moves(
    board: SquareBoard,
    time_limit: float,
    exploration: float = 1.41,
    top_k: int = 5,
) -> list[tuple[tuple, float, float]]:
    """
    Run MCTS for *time_limit* seconds and return up to *top_k* root children
    sorted by descending win ratio (mean Q-value).  Each entry is (move, win_ratio, visits).
    """
    result = fast_mcts(
        board,
        time_limit=time_limit,
        exploration=exploration,
        top_k=top_k,
        metric="win_ratio",
    )
    return result if result is not None else []


# ── Display ───────────────────────────────────────────────────────────────────

def _fmt_move(move: tuple) -> str:
    (r1, c1), _, (r2, c2) = move
    return f"({r1},{c1})->({r2},{c2})"


def print_step(
    step: int,
    board: SquareBoard,
    chosen_move: tuple,
    nn_top: list,
    mcts_top: list,
) -> None:
    chosen_str = _fmt_move(chosen_move)

    print(f"\n{'─'*68}")
    print(f" Step {step:>3}  |  Pegs remaining: {len(board.pegs)}")
    print(f"{'─'*68}")
    print(board)

    # Column headers
    nn_hdr   = f"{'NN Move':<17}  {'Prob':>7}"
    mcts_hdr = f"{'MCTS Move':<17}  {'WinRatio':>8}  {'Visits':>6}"
    sep      = "  |  "
    print(f"\n {'#':<3}  {nn_hdr}{sep}{mcts_hdr}")
    print(f" {'─'*3}  {'─'*25}{sep}{'─'*35}")

    n_rows = max(len(nn_top), len(mcts_top))
    for i in range(n_rows):
        rank = str(i + 1)

        if i < len(nn_top):
            m, p = nn_top[i]
            ms   = _fmt_move(m)
            flag = "*" if _fmt_move(m) == chosen_str else " "
            nn_col = f"{ms:<15}{flag}  {p*100:>6.2f}%"
        else:
            nn_col = " " * 25

        if i < len(mcts_top):
            m, win_ratio, visits = mcts_top[i]
            mcts_col = f"{_fmt_move(m):<17}  {win_ratio:>8.4f}  {visits:>6}"
        else:
            mcts_col = ""

        print(f" {rank:<3}  {nn_col}{sep}{mcts_col}")

    # Top-1 agreement
    agree_str = ""
    if nn_top and mcts_top:
        nn_best   = _fmt_move(nn_top[0][0])
        mcts_best = _fmt_move(mcts_top[0][0])
        agree     = nn_best == mcts_best
        agree_str = f"  Top-1 agree: {'YES ✓' if agree else 'NO  ✗'}  (NN: {nn_best}, MCTS: {mcts_best})"
    print(agree_str)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare NN and MCTS move rankings along one greedy NN rollout."
    )
    parser.add_argument("--n",          type=int,   default=5,
                        help="Board side length (default 5)")
    parser.add_argument("--model",      type=str,   default="policy_model.keras",
                        help="Path to .keras policy model")
    parser.add_argument("--time_limit", type=float, default=1.0,
                        help="MCTS time budget per move in seconds (default 1.0)")
    parser.add_argument("--top_k",      type=int,   default=5,
                        help="Number of top moves to show (default 5)")
    parser.add_argument("--seed",       type=int,   default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from '{args.model}' …")
    model = keras.saving.load_model(args.model)

    board = SquareBoard(args.n)
    print(f"Board: {args.n}×{args.n}  |  initial pegs: {len(board.pegs)}")
    print(f"MCTS time/move: {args.time_limit}s  |  top-k: {args.top_k}")

    # ── Pre-compute shared MCTS artefacts ─────────────────────────────────────
    move_table     = _move_table(args.n)
    move_table_jax = jnp.array(move_table)
    max_steps      = args.n * args.n
    board_flat_tmp = _board_to_flat(board)

    print("Warming up JAX JIT …", end=" ", flush=True)
    _run_rollout(
        jnp.array(board_flat_tmp),
        jax.random.PRNGKey(0),
        move_table_jax,
        max_steps,
    ).block_until_ready()
    print("done.")

    # ── Rollout ───────────────────────────────────────────────────────────────
    step = 0
    while True:
        moves = board.available_moves()
        if not moves:
            break

        step += 1

        nn_top     = nn_ranked_moves(model, board, top_k=args.top_k)
        chosen     = nn_top[0][0]   # greedy = highest probability move
        mcts_top   = mcts_ranked_moves(
            board,
            time_limit=args.time_limit,
            top_k=args.top_k,
        )

        print_step(step, board, chosen, nn_top, mcts_top)

        fr, _ov, to = chosen
        board.move(fr, to)

    print(f"\n{'═'*68}")
    print(f" Game over.  Pegs remaining: {len(board.pegs)}")
    print(f"{'═'*68}")
    print(board)


if __name__ == "__main__":
    main()

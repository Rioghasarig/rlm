#pragma once

#include <optional>
#include <vector>

#include "board.hpp"

// Precomputed table of every geometrically-possible (fr, ov, to) triple for a
// given board geometry. Stored as three parallel arrays of flat indices so the
// legality test in the hot loop is a few array lookups with no bounds checks.
struct Candidates {
    std::vector<int> fr;
    std::vector<int> ov;
    std::vector<int> to;
};

Candidates build_candidates(const Board& board);

// Search *board* to *max_depth* plies and return the move that minimises the
// number of pegs remaining, or std::nullopt if no moves exist.
//
//   max_depth   : maximum number of moves to look ahead (>= 1).
//   q           : quiescence threshold -- at the depth limit, keep searching if
//                 the number of available moves <= q.
//   max_breadth : maximum children expanded per node; when a node has more
//                 legal moves, the max_breadth with the most successors are
//                 kept and the rest discarded. -1 = expand all.
std::optional<Move> fast_dfs(const Board& board, int max_depth, int q,
                             int max_breadth);

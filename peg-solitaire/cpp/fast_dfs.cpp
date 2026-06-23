// Optimized depth-first search for square-board peg solitaire.
//
// This is a C++ port of fast_dfs.py. The Python version leans on JAX to
// vectorise move generation; in C++ the equivalent tight loops over a small
// precomputed candidate table are already fast, so we keep the same algorithm:
//
//   1. Precomputed candidate table  -- all geometrically possible (fr, ov, to)
//                                       triples, tested for legality in place.
//   2. In-place apply / undo         -- one working array, no copies.
//   3. Move ordering                 -- most-open moves first so the win cutoff
//                                       (1 peg) fires sooner.
//   4. Quiescence search             -- extends past the depth limit in narrow
//                                       positions (moves <= q).
//
// Single-threaded: root moves are evaluated sequentially.
#include "fast_dfs.hpp"

#include <algorithm>
#include <stdexcept>

Candidates build_candidates(const Board& board) {
    Candidates c;
    for (int r = 0; r < board.n; ++r) {
        for (int col = 0; col < board.n; ++col) {
            if (!board.in_bounds(r, col)) {
                continue;
            }
            for (const auto& d : SquareBoard::DIRECTIONS) {
                int ovr = r + d[0], ovc = col + d[1];
                int tor = r + 2 * d[0], toc = col + 2 * d[1];
                if (board.in_bounds(ovr, ovc) && board.in_bounds(tor, toc)) {
                    c.fr.push_back(board.idx(r, col));
                    c.ov.push_back(board.idx(ovr, ovc));
                    c.to.push_back(board.idx(tor, toc));
                }
            }
        }
    }
    return c;
}

namespace {

using Array = std::vector<uint8_t>;

// Count pegs remaining. Cells are 0/1/2 (HOLE/PEG/INVALID), so we count cells
// equal to PEG rather than summing -- INVALID corners must not contribute.
int count_pegs(const Array& arr) {
    int s = 0;
    for (uint8_t v : arr) {
        if (v == Board::PEG) {
            ++s;
        }
    }
    return s;
}

std::vector<Move> get_moves(const Array& arr, const Candidates& c) {
    std::vector<Move> moves;
    for (size_t i = 0; i < c.fr.size(); ++i) {
        if (arr[c.fr[i]] == 1 && arr[c.ov[i]] == 1 && arr[c.to[i]] == 0) {
            moves.push_back({c.fr[i], c.ov[i], c.to[i]});
        }
    }
    return moves;
}

int count_valid(const Array& arr, const Candidates& c) {
    int s = 0;
    for (size_t i = 0; i < c.fr.size(); ++i) {
        if (arr[c.fr[i]] == 1 && arr[c.ov[i]] == 1 && arr[c.to[i]] == 0) {
            ++s;
        }
    }
    return s;
}

inline void apply(Array& arr, const Move& m) {
    arr[m.fr] = 0;
    arr[m.ov] = 0;
    arr[m.to] = 1;
}

inline void undo(Array& arr, const Move& m) {
    arr[m.fr] = 1;
    arr[m.ov] = 1;
    arr[m.to] = 0;
}

// Sort moves by successor count descending (most open first).
void order_moves(Array& arr, const Candidates& c, std::vector<Move>& moves) {
    std::vector<std::pair<int, Move>> scored;
    scored.reserve(moves.size());
    for (const Move& m : moves) {
        apply(arr, m);
        scored.push_back({count_valid(arr, c), m});
        undo(arr, m);
    }
    std::stable_sort(scored.begin(), scored.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    moves.clear();
    for (auto& p : scored) {
        moves.push_back(p.second);
    }
}

// Keep at most max_breadth moves (max_breadth < 0 means keep all). Assumes
// moves is already sorted most-open first, so this keeps the moves with the
// most successors and discards the rest.
void limit_breadth(std::vector<Move>& moves, int max_breadth) {
    if (max_breadth < 0 || static_cast<int>(moves.size()) <= max_breadth) {
        return;
    }
    moves.resize(static_cast<size_t>(max_breadth));
}

int dfs(Array& arr, const Candidates& c, int depth, int q, int max_breadth) {
    std::vector<Move> moves = get_moves(arr, c);
    if (moves.empty()) {
        return count_pegs(arr);
    }
    if (depth == 0 && static_cast<int>(moves.size()) > q) {
        return count_pegs(arr);
    }

    int next_depth = depth > 0 ? depth - 1 : 0;
    int best = count_pegs(arr);

    order_moves(arr, c, moves);
    limit_breadth(moves, max_breadth);
    for (const Move& m : moves) {
        apply(arr, m);
        int result = dfs(arr, c, next_depth, q, max_breadth);
        undo(arr, m);
        if (result < best) {
            best = result;
            if (best == 1) {
                break;
            }
        }
    }
    return best;
}

}  // namespace

std::optional<Move> fast_dfs(const Board& board, int max_depth, int q,
                             int max_breadth) {
    if (max_depth < 1) {
        throw std::invalid_argument("max_depth must be at least 1");
    }
    if (max_breadth >= 0 && max_breadth < 1) {
        throw std::invalid_argument("max_breadth must be at least 1");
    }

    Array arr = board.cells;
    Candidates c = build_candidates(board);

    std::vector<Move> moves = get_moves(arr, c);
    if (moves.empty()) {
        return std::nullopt;
    }

    order_moves(arr, c, moves);
    limit_breadth(moves, max_breadth);

    int best_score = count_pegs(arr) + 1;
    std::optional<Move> best_move;
    for (const Move& m : moves) {
        apply(arr, m);
        int score = dfs(arr, c, max_depth - 1, q, max_breadth);
        undo(arr, m);
        if (score < best_score) {
            best_score = score;
            best_move = m;
            if (best_score == 1) {  // optimal -- no need to look further
                break;
            }
        }
    }
    return best_move;
}

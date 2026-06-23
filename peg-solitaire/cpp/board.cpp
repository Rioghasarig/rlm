#include "board.hpp"

#include <stdexcept>

const std::array<std::array<int, 2>, 4> Board::DIRECTIONS = {{
    {{-1, 0}}, {{1, 0}}, {{0, -1}}, {{0, 1}},
}};

Board::Board(int n_) : n(n_), cells(static_cast<size_t>(n_) * n_, PEG) {
    if (n < 2) {
        throw std::invalid_argument("Board size must be at least 2");
    }
}

void Board::init(int empty_r, int empty_c) {
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            cells[idx(r, c)] = in_bounds(r, c) ? PEG : INVALID;
        }
    }
    if (empty_r < 0 || empty_c < 0) {
        empty_r = n / 2;
        empty_c = n / 2;
    }
    if (!in_bounds(empty_r, empty_c)) {
        throw std::invalid_argument("empty_start is not on the board");
    }
    cells[idx(empty_r, empty_c)] = HOLE;
}

std::vector<Move> Board::available_moves() const {
    std::vector<Move> moves;
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            if (cells[idx(r, c)] != PEG) {
                continue;
            }
            for (const auto& d : DIRECTIONS) {
                int ovr = r + d[0], ovc = c + d[1];
                int tor = r + 2 * d[0], toc = c + 2 * d[1];
                // Off-board cells are INVALID (never PEG or HOLE), so the cell
                // checks below also enforce the board shape; we only need the
                // grid-bounds check to index safely.
                if (0 <= tor && tor < n && 0 <= toc && toc < n &&
                    cells[idx(ovr, ovc)] == PEG && cells[idx(tor, toc)] == HOLE) {
                    moves.push_back({idx(r, c), idx(ovr, ovc), idx(tor, toc)});
                }
            }
        }
    }
    return moves;
}

void Board::apply_move(const Move& m) {
    cells[m.fr] = HOLE;
    cells[m.ov] = HOLE;
    cells[m.to] = PEG;
}

int Board::peg_count() const {
    int s = 0;
    for (uint8_t v : cells) {
        if (v == PEG) {
            ++s;
        }
    }
    return s;
}

bool Board::is_won() const {
    return peg_count() == 1;
}

std::string Board::to_string() const {
    std::string out;
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            if (c) {
                out += ' ';
            }
            uint8_t v = cells[idx(r, c)];
            out += v == PEG ? 'o' : (v == HOLE ? '.' : ' ');
        }
        if (r + 1 < n) {
            out += '\n';
        }
    }
    return out;
}

SquareBoard::SquareBoard(int n_, int empty_r, int empty_c) : Board(n_) {
    init(empty_r, empty_c);
}

bool SquareBoard::in_bounds(int r, int c) const {
    return 0 <= r && r < n && 0 <= c && c < n;
}

CrossBoard::CrossBoard(int empty_r, int empty_c) : Board(7) {
    init(empty_r, empty_c);
}

bool CrossBoard::in_bounds(int r, int c) const {
    if (!(0 <= r && r < n && 0 <= c && c < n)) {
        return false;
    }
    int arm = n / 3;  // = 2 for the standard 7x7 board
    return (arm <= r && r < n - arm) || (arm <= c && c < n - arm);
}

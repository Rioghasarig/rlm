#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// A single jump move, expressed as flat board indices (row * n + col):
//   fr -> peg that jumps,  ov -> peg jumped over (removed),  to -> empty landing cell.
struct Move {
    int fr;
    int ov;
    int to;
};

// Abstract base class for peg solitaire on a square bounding grid.
//
// Cells are stored as a flat row-major array over the n x n bounding grid:
//   PEG (1)     -- a peg
//   HOLE (0)    -- an empty cell that is part of the board
//   INVALID (2) -- a cell that is not part of the board (e.g. a cross corner)
//
// Pegs jump orthogonally exactly two steps, landing on an empty cell and
// removing the peg jumped over. Subclasses define the board geometry -- which
// cells are part of the board, via in_bounds() -- in their constructors; all
// move logic is shared and simply respects the INVALID markers.
class Board {
public:
    static constexpr uint8_t HOLE = 0;
    static constexpr uint8_t PEG = 1;
    static constexpr uint8_t INVALID = 2;

    int n;                       // bounding-grid dimension
    std::vector<uint8_t> cells;  // size n*n

    virtual ~Board() = default;

    // The four orthogonal jump directions.
    static const std::array<std::array<int, 2>, 4> DIRECTIONS;

    // Whether (r, c) is a cell of this board's shape (also checks grid bounds).
    virtual bool in_bounds(int r, int c) const = 0;

    inline int idx(int r, int c) const { return r * n + c; }

    std::vector<Move> available_moves() const;
    void apply_move(const Move& m);
    int peg_count() const;
    bool is_won() const;
    std::string to_string() const;

protected:
    explicit Board(int n);
    // Mark every in-bounds cell PEG and every off-board cell INVALID, then empty
    // the single start cell. empty_r/empty_c < 0 selects the centre (n/2, n/2).
    // Subclasses must call this from their constructor (so in_bounds() dispatches
    // to the derived geometry).
    void init(int empty_r, int empty_c);
};

// Peg solitaire on an n x n square grid; every cell is part of the board.
class SquareBoard : public Board {
public:
    // empty_r/empty_c < 0 selects the centre cell (n/2, n/2).
    explicit SquareBoard(int n, int empty_r = -1, int empty_c = -1);
    bool in_bounds(int r, int c) const override;
};

// The classic English peg solitaire board: a plus/cross shape carved out of a
// 7x7 grid by removing the four 2x2 corner blocks, leaving 33 cells. A cell is
// part of the board when it lies in the central three rows or central three
// columns; the corners (both row and col in {0, 1, 5, 6}) are off the board.
class CrossBoard : public Board {
public:
    // empty_r/empty_c < 0 selects the centre cell (3, 3).
    explicit CrossBoard(int empty_r = -1, int empty_c = -1);
    bool in_bounds(int r, int c) const override;
};

// benchmark.cpp -- Benchmark strategies on a square peg-solitaire board and
// report the average number of pegs remaining across multiple trials.
//
// This is a C++ port of benchmark.py, restricted to the square and cross boards
// and the strategies that do not need the Python ML stack:
//
//   dfs    : fast_dfs depth-first search with a configurable depth limit.
//   random : uniformly random legal move selection.
//
// Usage:
//   ./benchmark [--board square|cross] [--n N] [--trials T]
//               [--strategies dfs random]
//               [--max_depth D] [--q Q] [--max_breadth B] [--seed S]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "board.hpp"
#include "fast_dfs.hpp"

namespace {

std::unique_ptr<Board> make_board(const std::string& type, int n) {
    if (type == "cross") {
        return std::make_unique<CrossBoard>();
    }
    return std::make_unique<SquareBoard>(n);
}

int play_game_dfs(const std::string& board_type, int n, int max_depth, int q,
                  int max_breadth) {
    std::unique_ptr<Board> board = make_board(board_type, n);
    while (true) {
        std::optional<Move> move = fast_dfs(*board, max_depth, q, max_breadth);
        if (!move) {
            break;
        }
        board->apply_move(*move);
    }
    return board->peg_count();
}

int play_game_random(const std::string& board_type, int n, std::mt19937& rng) {
    std::unique_ptr<Board> board = make_board(board_type, n);
    while (true) {
        std::vector<Move> moves = board->available_moves();
        if (moves.empty()) {
            break;
        }
        std::uniform_int_distribution<size_t> pick(0, moves.size() - 1);
        board->apply_move(moves[pick(rng)]);
    }
    return board->peg_count();
}

void report(const std::string& label, const std::vector<int>& results) {
    int trials = static_cast<int>(results.size());
    int mn = results[0], mx = results[0], sum = 0, solved = 0;
    for (int r : results) {
        mn = std::min(mn, r);
        mx = std::max(mx, r);
        sum += r;
        solved += (r == 1);
    }
    double mean = static_cast<double>(sum) / trials;

    std::cout << "\n" << label << " -- results over " << trials << " trial(s):\n";
    std::cout << "  min   : " << mn << "\n";
    std::cout << "  max   : " << mx << "\n";
    std::cout.setf(std::ios::fixed);
    std::cout.precision(2);
    std::cout << "  mean  : " << mean << "\n";
    if (trials > 1) {
        double var = 0.0;
        for (int r : results) {
            var += (r - mean) * (r - mean);
        }
        var /= (trials - 1);  // sample standard deviation, matching Python's statistics.stdev
        std::cout << "  stdev : " << std::sqrt(var) << "\n";
    }
    std::cout << "  solved: " << solved << "/" << trials << "  (1 peg = solved)\n";
    std::cout.unsetf(std::ios::fixed);
}

[[noreturn]] void usage_error(const std::string& msg) {
    std::cerr << "error: " << msg << "\n"
              << "usage: benchmark [--board square|cross] [--n N] [--trials T]\n"
              << "                 [--strategies dfs random]\n"
              << "                 [--max_depth D] [--q Q] [--max_breadth B] [--seed S]\n";
    std::exit(2);
}

int parse_int(const char* s, const std::string& flag) {
    try {
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (s[pos] != '\0') {
            throw std::invalid_argument("trailing characters");
        }
        return v;
    } catch (const std::exception&) {
        usage_error("invalid integer for " + flag + ": '" + s + "'");
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string board_type = "square";
    int n = 5;
    int trials = 5;
    int max_depth = 5;
    int q = 1;
    int max_breadth = -1;  // -1 = expand all
    unsigned seed = std::random_device{}();
    bool seed_set = false;
    std::vector<std::string> strategies;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const std::string& flag) -> const char* {
            if (i + 1 >= argc) {
                usage_error("missing value for " + flag);
            }
            return argv[++i];
        };
        if (arg == "--board") {
            board_type = next(arg);
            if (board_type != "square" && board_type != "cross") {
                usage_error("unknown board: '" + board_type + "' (choices: square, cross)");
            }
        } else if (arg == "--n") {
            n = parse_int(next(arg), arg);
        } else if (arg == "--trials") {
            trials = parse_int(next(arg), arg);
        } else if (arg == "--max_depth") {
            max_depth = parse_int(next(arg), arg);
        } else if (arg == "--q") {
            q = parse_int(next(arg), arg);
        } else if (arg == "--max_breadth") {
            max_breadth = parse_int(next(arg), arg);
        } else if (arg == "--seed") {
            seed = static_cast<unsigned>(parse_int(next(arg), arg));
            seed_set = true;
        } else if (arg == "--strategies") {
            // Consume following tokens until the next --flag.
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                std::string s = argv[++i];
                if (s != "dfs" && s != "random") {
                    usage_error("unknown strategy: '" + s + "' (choices: dfs, random)");
                }
                strategies.push_back(s);
            }
        } else if (arg == "-h" || arg == "--help") {
            usage_error("help");
        } else {
            usage_error("unknown argument: '" + arg + "'");
        }
    }

    if (strategies.empty()) {
        strategies = {"dfs", "random"};
    }
    if (n < 2) {
        usage_error("--n must be at least 2");
    }
    if (trials < 1) {
        usage_error("--trials must be at least 1");
    }

    std::mt19937 rng(seed);
    int initial_pegs = make_board(board_type, n)->peg_count();

    if (board_type == "cross") {
        std::cout << "Board: cross (English 7x7)";
    } else {
        std::cout << "Board: square (" << n << "x" << n << ")";
    }
    std::cout << "  |  initial pegs: " << initial_pegs << "  |  trials: " << trials;
    if (seed_set) {
        std::cout << "  |  seed: " << seed;
    }
    std::cout << "\n";

    int width = static_cast<int>(std::to_string(trials).size());

    for (const std::string& strat : strategies) {
        std::vector<int> results;
        results.reserve(trials);
        if (strat == "dfs") {
            std::cout << "\n[DFS]  max_depth: " << max_depth << "  q: " << q
                      << "  max_breadth: " << (max_breadth < 0 ? "all" : std::to_string(max_breadth))
                      << "  (running " << trials << " trial(s)...)\n";
        } else {
            std::cout << "\n[Random]  (uniform random legal move -- running " << trials
                      << " trial(s)...)\n";
        }

        for (int i = 1; i <= trials; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            int remaining = (strat == "dfs")
                                ? play_game_dfs(board_type, n, max_depth, q, max_breadth)
                                : play_game_random(board_type, n, rng);
            double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            results.push_back(remaining);
            std::cout << "  game " << std::string(width - std::to_string(i).size(), ' ') << i
                      << ": " << remaining << " pegs remaining  (" << std::fixed
                      << std::setprecision(1) << elapsed << "s)\n";
            std::cout.unsetf(std::ios::fixed);
        }
        report(strat == "dfs" ? "DFS" : "Random", results);
    }

    return 0;
}

# fast_imitation_learning.py

JAX-accelerated DAgger (Dataset Aggregation) for training a policy network to play SquareBoard peg solitaire.

## Algorithm overview

DAgger is an imitation learning algorithm that iteratively builds a training dataset by rolling out the *current* policy, then labelling every visited state with the *teacher's* best action. This avoids the distributional shift problem of pure behavioural cloning: the learner sees the states it will actually visit, not only those from expert demonstrations.

Each iteration:
1. Roll out the current policy from the starting board to collect `n_trajectories` trajectories.
2. Label every state in each trajectory using MCTS as the teacher.
3. Add the labelled (state, action) pairs to the aggregate dataset `D`.
4. Train the policy network on all of `D` for `epochs` passes.
5. Save the updated model.

## Key speedups over `imitation_learning.py`

| Component | Technique |
|-----------|-----------|
| MCTS rollouts | JAX `lax.scan` compiled to a single XLA kernel; flat NumPy tree with vectorised UCB1 and scatter-add backprop; precomputed move table |
| Training step | `model.compile(jit_compile=True)` compiles the forward + backward + update pass once; reused across all DAgger iterations without retracing |
| MCTS labelling | Parallelised across `n_workers` processes via `multiprocessing.Pool`; each worker warms its XLA kernel once at startup |
| Compilation cost | JAX rollout kernel and Keras training step are both warmed before the DAgger loop starts |

## Configuration (`config.yaml`)

```yaml
board:
  n: 5                         # board side length (n×n grid)
  empty_start: [2, 2]          # starting hole; null → centre

network:
  res_blocks: 4                # residual blocks in the policy tower
  filters: 64                  # convolutional filters per block

optimizer:
  learning_rate: 0.001

dagger:
  n_iterations: 20             # DAgger outer iterations
  epochs: 5                    # supervised-learning epochs per iteration
  batch_size: 64               # mini-batch size
  large_dataset_threshold: null  # switch to larger epochs/batch above this dataset size; null → never
  large_dataset_epochs: 10
  large_dataset_batch_size: 128
  mcts_time_limit: 1.0         # wall-clock seconds given to fast_mcts per state
  n_trajectories: 2            # rollouts collected per iteration
  save_path: policy_model.keras  # checkpoint path; null to disable
  load_path: null              # resume from this .keras file; null → fresh model
  n_workers: null              # parallel MCTS workers; null → os.cpu_count()
```

## Usage

```bash
python fast_imitation_learning.py                  # uses config.yaml
python fast_imitation_learning.py --config my.yaml # custom config
```

To resume training from a saved checkpoint, set `load_path` in the config to the `.keras` file written by a previous run.

## Training progress log format

Each run can write a **JSONL** file (one JSON object per line) recording every notable event. This is the intended format for later graphing and analysis — it is append-only, human-readable, and trivially consumed by pandas.

### File naming

```
runs/<YYYYMMDD_HHMMSS>[_<tag>].jsonl
```

Example: `runs/20260419_143022_n5.jsonl`

### Record types

Every record carries a `"type"` field and `"t"` (seconds since run start).

#### `run_start`
Written once at the beginning.

```json
{"type": "run_start", "t": 0.0, "n_iterations": 20, "n_trajectories": 2,
 "epochs": 5, "batch_size": 64, "mcts_time_limit": 1.0, "n_workers": 8,
 "board_n": 5, "dataset_size": 0}
```

#### `iteration_start`
Written at the beginning of each DAgger iteration.

```json
{"type": "iteration_start", "t": 12.4, "iteration": 1, "dataset_size": 0}
```

#### `trajectory`
Written after each policy rollout.

```json
{"type": "trajectory", "t": 15.1, "iteration": 1, "trajectory_idx": 0,
 "steps": 47, "pegs_remaining": 3, "traj_time_s": 2.7}
```

- `steps` — number of moves made before the board was stuck
- `pegs_remaining` — pegs left on the final board (lower is better; 1 is a perfect solve)

#### `labeling`
Written after MCTS has labelled all states in a trajectory.

```json
{"type": "labeling", "t": 38.2, "iteration": 1, "trajectory_idx": 0,
 "labeled": 44, "skipped": 3, "label_time_s": 23.1, "rate_states_per_s": 1.9}
```

- `skipped` — states where MCTS returned no move (terminal or all moves pruned)

#### `epoch_loss`
Written once per training epoch inside `learn()`.

```json
{"type": "epoch_loss", "t": 42.1, "iteration": 1, "epoch": 1, "loss": 0.8431}
```

#### `iteration_end`
Written at the end of each DAgger iteration.

```json
{"type": "iteration_end", "t": 55.0, "iteration": 1,
 "dataset_size": 132, "new_samples": 132,
 "train_time_s": 12.1, "iter_time_s": 42.6}
```

#### `run_end`
Written once when the DAgger loop finishes.

```json
{"type": "run_end", "t": 980.0, "total_time_s": 980.0, "dataset_size": 2640}
```

### Reading logs for graphing

```python
import pandas as pd

df = pd.read_json("runs/20260419_143022_n5.jsonl", lines=True)

# Loss curve across all iterations and epochs
losses = df[df.type == "epoch_loss"][["iteration", "epoch", "loss"]]

# Per-iteration dataset growth and timing
iters = df[df.type == "iteration_end"][["iteration", "dataset_size", "new_samples", "iter_time_s"]]

# Policy quality over time (pegs remaining per trajectory)
trajs = df[df.type == "trajectory"][["iteration", "trajectory_idx", "pegs_remaining", "steps"]]

# Overlay multiple runs
runs = {
    "baseline": pd.read_json("runs/20260419_143022_n5.jsonl", lines=True),
    "longer":   pd.read_json("runs/20260420_090000_n5_long.jsonl", lines=True),
}
```

Useful plots:
- `epoch_loss.loss` vs global epoch number — training loss curve
- `iteration_end.dataset_size` vs `iteration` — dataset growth
- `trajectory.pegs_remaining` vs `iteration` — policy improvement over time
- `iteration_end.iter_time_s` vs `iteration` — wall-clock cost per iteration

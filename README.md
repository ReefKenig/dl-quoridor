# Quoridor AI — AlphaZero-Inspired Agent

Deep Reinforcement Learning final project (Group 501, Colman College).

An autonomous AI agent for the board game Quoridor, using a dual-headed neural network + MCTS trained via self-play. Supports 2 and 4 players: the 2-player path uses negamax MCTS with a scalar value head; the N-player path uses max^n MCTS with a vector value head.

The project runs at two scales: a **5×5 proof-of-concept** that validated the architecture and training dynamics, and **full-size 9×9 Quoridor** (10 walls/player at N=2, 5 at N=4), which took eleven run generations to produce an agent that both races and walls.

## Results

Measured against a fixed shortest-path opponent ("greedy": always take the move that most shortens your own path, never place a wall) — not against random, which saturates at 100% and hides the failure mode below.

| Board | Players | Model | vs. greedy racer | vs. random |
|---|---|---|---|---|
| 5×5 | 2 | `runs/n2_5x5_v1/ship.pt` | 40/40 | 97.9% ± 1.9% |
| 5×5 | 4 | `runs/n4_5x5_v3/ship.pt` | not measured | 84.2% ± 4.6% |
| 9×9 | 2 | `runs/n2_9x9_v9/greedy_peak.pt` | **93.8%** (seat 0 39/40, seat 1 36/40) | 100% |
| 9×9 | 4 | `runs/n4_9x9_v10/greedy_peak.pt` | **seat 0 10/20**; seats 1–3 structurally unwinnable by racing | 100% |

Fair share vs. random is 50% at N=2 and 25% at N=4.

Two findings worth knowing before reading any number in this repo:

- **A saturated baseline is worse than no baseline.** Through most of the 9×9 work the accept gate reported steady improvement and vs-random read 96–100% while the models scored **0% against the greedy racer** — relative strength with zero absolute competence.
- **`best.pt` is not the best model.** It is written by the accept gate, which at 9×9 either never fired (leaving the untrained initialization) or fired on the iteration where strength collapsed. The shipped 9×9 models are `greedy_peak.pt`, ratcheted on score against the racer. Always resolve checkpoints through `runs/MODELS.json`, never by hardcoding a filename.

## Project Structure

```
dl-quoridor/
├── src/
│   ├── env/                    # Game engine & environment interface
│   │   ├── env_interface.py    #   ABC contract for MCTS ↔ Engine
│   │   ├── quoridor_env.py     #   2-player game logic (per-player walls)
│   │   ├── quoridor_env_mp.py  #   N-player engine (2..4, shared walls)
│   │   ├── tensor_spec.py      #   2-player tensor (10 channels)
│   │   └── tensor_spec_mp.py   #   N-player tensor (3N+3 channels)
│   ├── mcts/                   # Monte Carlo Tree Search
│   │   ├── mcts.py             #   Negamax MCTS (2-player)
│   │   ├── mcts_maxn.py        #   Max^n MCTS (N-player)
│   │   ├── evaluator.py        #   2-player evaluation & matchups
│   │   ├── evaluator_mp.py     #   N-player evaluation (seat rotation)
│   │   ├── self_play.py        #   2-player self-play & training loop
│   │   ├── self_play_mp.py     #   N-player self-play (vector targets)
│   │   └── training_mp.py      #   N-player training loop
│   ├── model/                  # Neural network architecture
│   │   ├── network.py          #   Dual-headed ResNet (scalar value)
│   │   └── network_mp.py       #   Dual-headed ResNet (vector value)
│   ├── server/                 # Inference server
│   │   └── app.py              #   Flask API for remote NN evaluation
│   ├── ui/                     # PyGame client
│   │   └── game_ui.py          #   Interactive board with AI opponent
│   └── utils/                  # Shared utilities
│       ├── checkpoint.py       #   Model checkpointing & resume
│       ├── config.py           #   JSON config loader
│       ├── model_registry.py   #   Which checkpoint backs which board/players
│       └── logger.py           #   Training metrics & W&B logging
├── tests/                      # Unit & integration tests
├── notebooks/                  # Colab training notebooks
├── configs/                    # Hyperparameter configs (5×5 POC, 9×9)
├── scripts/                    # Dev helper & validation scripts
├── docs/                       # Design docs (action space, etc.)
├── runs/                       # Training runs, versioned (see runs/README.md)
│   ├── README.md               #   Layout + versioning convention
│   ├── MODELS.json             #   Checkpoint registry — which model backs which
│   │                           #     board/player combo, + the spec it was trained under
│   └── <arch>_<board>_<vN>/    #   One self-contained dir per run:
│       ├── config.json         #     frozen hyperparams (tracked)
│       ├── meta.json           #     per-iteration progress history (tracked)
│       ├── train.log           #     run log (tracked)
│       ├── figures/            #     plots from THIS run's metrics (tracked)
│       ├── greedy_peak.pt      #     best score vs the racer — what 9×9
│       │                       #       ships (git-ignored)
│       ├── best.pt / latest.pt #     gate champion / most recent (git-ignored)
│       └── peaks/              #     new-high snapshots (git-ignored)
├── outputs/                    # Cross-run artifacts only
│   ├── model_comparison.png    #   2p vs 4p summary
│   ├── held_out_eval.json      #   held-out scoring vs greedy + depth-2 minimax
│   └── results/                #   model-vs-model eval dumps
├── requirements.txt
└── README.md
```

> **Checkpoints & versioning:** each training run lives in its own
> `runs/<id>/` directory. Git tracks the lightweight progress record
> (`config.json`, `meta.json`, logs, figures) and ignores model weights
> (`*.pt`). See [`runs/README.md`](runs/README.md) for the full convention.

## Setup

```bash
git clone https://github.com/ReefKenig/dl-quoridor.git
cd dl-quoridor
python -m venv venv
source venv/bin/activate        # Linux/Mac
pip install -r requirements.txt
```

## Training

```bash
# 5×5 POC (default)
python -m src --config configs/config_5x5.json

# Resume from checkpoint
python -m src --config configs/config_5x5.json --checkpoint-dir checkpoints

# Start fresh (ignore existing checkpoints)
python -m src --config configs/config_5x5.json --no-resume
```

Training runs an AlphaZero-style loop: self-play → collect data (with data augmentation) → train network → evaluate vs best model → checkpoint.

### N-player training (2 and 4 players)

The N-player path uses `QuoridorEnvMP`, `MCTSMaxN`, and `QuoridorModelMP`:

```python
from src.mcts.training_mp import training_loop_mp, TrainingConfigMP
```

See `scripts/run_train_eval.py` for a smoke-test example. Wall counts are deliberately reduced for the 5×5 POC (N=4 → 2 walls/seat). To demonstrate leader-blocking or coalition-emergence at N=4, override with `max_walls_per_player=4` or higher.

### 9×9 training (GPU required)

Full-size runs are driven by the training notebooks, executed headlessly so they survive a dropped connection. All hyperparameters come from `configs/config_9x9.json`.

```bash
scripts/run_notebook.sh n2      # N=2, 10 walls/player
scripts/run_notebook.sh n4      # N=4, 5 walls/player

tail -f runs/<run_dir>/notebook.log        # monitor
kill $(cat runs/<run_dir>/notebook.pid)    # stop
```

There is no default variant — `n2` or `n4` must be passed explicitly. Runs resume from `latest.pt` + `meta.json`.

**Do not run 9×9 training on a laptop.** The inference batcher probes only for CUDA and otherwise falls back to CPU, which measured ~66× slower than the GPU server (810 s/game vs 12.3) — a 60-iteration run would take about 90 days. Check the `resources:` line in `games.log` before letting a run proceed.

Two settings are load-bearing and easy to get wrong:

- `wall_candidates=16` restricts which wall placements MCTS expands. Unrestricted, search spreads across 128 wall actions at 4.6 visits each and the resulting policy walls instead of racing; restricting it raises resolution to 31.6 visits/action. This is not an optimization — it is the difference between a model that scores 0% and one that scores 85%+.
- A warm start (`scripts/pretrain_greedy.py`) imitates the racer before self-play begins. Six cold runs never passed 2/80 at N=4; thirty minutes of imitation reaches 20/20 in seat 0 with no search at all.

### Validation scripts

| Script | What it proves |
|---|---|
| `scripts/run_reduction.py` | max^n(N=2) produces bit-identical visit distributions to negamax — the equivalence proof |
| `scripts/run_mp_validate.py` | N=2 lockstep parity, jump rules, random termination (N=2/4), max^n search drives to terminal |
| `scripts/run_train_eval.py` | Evaluator harness + tiny N=4 training loop + checkpoint reload |
| `scripts/run_compare.py` | Same-weights negamax vs max^n H2H (structural 50/50, not a correctness proof) |
| `scripts/probe_greedy.py` | Per-seat scoring vs the greedy racer; `--trace` prints a game move by move |
| `scripts/eval_all_checkpoints.py` | Held-out table: every registered checkpoint vs greedy and depth-2 minimax |
| `scripts/pretrain_greedy.py` | Supervised warm start — imitate the racer before self-play |

## Play vs AI

```bash
python -m src.ui.game_ui                          # 5×5, 4 players (default)
python -m src.ui.game_ui --board 9 --players 2    # full-size, the 93.8% model
python -m src.ui.game_ui --board 9 --players 4    # full-size, 4 players
```

`--board {5,9}`, `--players {2,4}`, `--difficulty {easy,medium,hard}`. Which
checkpoint each combination loads — along with the architecture, tensor spec and
wall count it was trained under — comes from `runs/MODELS.json`; the UI prints
the file it resolved and why on startup.

## Run Tests

```bash
pytest tests/
```

## Team

| Member | Responsibility |
|--------|---------------|
| Reef Kenig | Game Engine, Gymnasium Wrapper, UI |
| Rom Gotshal | DL Architecture, SB3 Research |
| Iris Yedidia | MCTS, Self-play Training Loop |

Supervisor: Moshe Butman

# Quoridor AI — AlphaZero-Inspired Agent

Deep Reinforcement Learning final project (Group 501, Colman College).

An autonomous AI agent for the board game Quoridor, using a dual-headed neural network + MCTS trained via self-play. Supports 2–4 players: the 2-player path uses negamax MCTS with a scalar value head; the N-player path uses max^n MCTS with a vector value head.

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
│       └── logger.py           #   Training metrics & W&B logging
├── tests/                      # Unit & integration tests
├── notebooks/                  # Colab training notebooks
├── configs/                    # Hyperparameter configs (5×5 POC, 9×9)
├── scripts/                    # Dev helper & validation scripts
├── docs/                       # Design docs (action space, etc.)
├── checkpoints/                # Saved model weights
├── requirements.txt
└── README.md
```

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

### N-player training (2–4 players)

The N-player path uses `QuoridorEnvMP`, `MCTSMaxN`, and `QuoridorModelMP`:

```python
from src.mcts.training_mp import training_loop_mp, TrainingConfigMP
```

See `scripts/run_train_eval.py` for a smoke-test example. Wall counts are deliberately reduced for the 5×5 POC (N≥3 → 2 walls/seat). To demonstrate leader-blocking or coalition-emergence at N=4, override with `max_walls_per_player=4` or higher.

### Validation scripts

| Script | What it proves |
|---|---|
| `scripts/run_reduction.py` | max^n(N=2) produces bit-identical visit distributions to negamax — the equivalence proof |
| `scripts/run_mp_validate.py` | N=2 lockstep parity, jump rules, random termination (N=2/3/4), max^n search drives to terminal |
| `scripts/run_train_eval.py` | Evaluator harness + tiny N=4 training loop + checkpoint reload |
| `scripts/run_compare.py` | Same-weights negamax vs max^n H2H (structural 50/50, not a correctness proof) |

## Play vs AI

```bash
python -m src.ui.game_ui
```

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

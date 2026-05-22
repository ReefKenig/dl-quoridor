# Quoridor AI — AlphaZero-Inspired Agent

Deep Reinforcement Learning final project (Group 501, Colman College).

An autonomous AI agent for the board game Quoridor, using a dual-headed neural network + MCTS trained via self-play.

## Project Structure

```
dl-quoridor/
├── src/
│   ├── env/                    # Game engine & environment interface
│   │   ├── env_interface.py    #   ABC contract for MCTS ↔ Engine
│   │   ├── quoridor_env.py     #   Gymnasium wrapper & game logic
│   │   └── tensor_spec.py      #   Observation/action space specs
│   ├── mcts/                   # Monte Carlo Tree Search
│   │   ├── mcts.py             #   Core MCTS engine
│   │   ├── evaluator.py        #   Model evaluation & agent matchups
│   │   └── self_play.py        #   Self-play data generation & training loop
│   ├── model/                  # Neural network architecture
│   │   └── network.py          #   Dual-headed ResNet (Policy + Value)
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
├── scripts/                    # Dev helper scripts
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

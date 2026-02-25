# Quoridor AI — AlphaZero-Inspired Agent

Deep Reinforcement Learning final project (Group 501, Colman College).

An autonomous AI agent for the board game Quoridor, using a dual-headed neural network + MCTS trained via self-play.

## Project Structure

```
dl-quoridor/
├── src/
│   ├── env/            # Game engine & environment interface
│   │   ├── env_interface.py    # ABC contract for MCTS ↔ Engine
│   │   └── quoridor_env.py     # Gymnasium/PettingZoo wrapper (Reef)
│   ├── mcts/           # Monte Carlo Tree Search
│   │   ├── mcts.py             # Core MCTS engine
│   │   └── self_play.py        # Self-play data generation & training loop
│   ├── model/          # Neural network architecture (Rom)
│   │   └── network.py          # Dual-headed CNN (Policy + Value)
│   └── ui/             # PyGame client (Reef) — placeholder
├── tests/              # Validation & unit tests
├── notebooks/          # Colab training notebooks
├── configs/            # Hyperparameter configs
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

## Run MCTS Validation

> **Note:** MCTS logic lives in `feature/mcts-engine`. Stubs only on `dev`.

```bash
python -m tests.test_mcts
```

## Team

| Member | Responsibility |
|--------|---------------|
| Reef Kenig | Game Engine, Gymnasium Wrapper, UI |
| Rom Gotshal | DL Architecture, SB3 Research |
| Iris Yedidia | MCTS, Self-play Training Loop |

Supervisor: Moshe Butman

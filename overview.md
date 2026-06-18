## Project Overview

This repository is a small AlphaZero-style Quoridor AI project with two main goals:

- implement the Quoridor game environment and action semantics (2–4 players),
- train a dual-headed neural network with MCTS self-play.

The 2-player path uses negamax MCTS with a scalar value head. The N-player
path (2–4) uses max^n MCTS with a vector value head (length N). Both share
the same action space (44 on 5×5, 140 on 9×9) and the same `QuoridorEnvInterface`
contract.

It is organized into:

- src — main implementation
- configs — JSON hyperparameter files
- checkpoints — model and training checkpoints
- tests — validation and unit tests
- notebooks — experiment / training notebooks

---

## Core Components

### 1. env — Game environment

- env_interface.py
  - Defines `QuoridorEnvInterface`
  - This is the contract MCTS and training rely on:
    - `get_valid_actions`
    - `step`
    - `clone_state`
    - `reset`
    - `state_to_tensor`
    - `get_current_player`
- quoridor_env.py
  - Implements the Quoridor game logic.
  - Defines `QuoridorState` dataclass containing:
    - pawn positions
    - sets of horizontal/vertical walls for both players
    - remaining wall counts
    - current player, turn count, game-over/winner flags
  - Action encoding:
    - first 12 ints = pawn moves
    - then horizontal wall placements
    - then vertical wall placements
  - Provides:
    - `step(state, action)` to apply moves
    - `get_valid_actions(state)` to compute legal pawn moves and valid walls, including path existence checks
    - `state_to_tensor(state)` to convert board state into a neural-network input tensor
- tensor_spec.py
  - Defines the 10-channel tensor format used by the 2-player network
  - Channels:
    - 0-1: player pawn positions
    - 2-5: wall maps per player / orientation
    - 6-7: remaining walls for each player (broadcast plane)
    - 8-9: BFS distance maps to each player's goal row
  - This makes the network aware of spatial board state, wall ownership, resource counts, and connectivity.
- quoridor_env_mp.py
  - N-player engine (2..4). Shared wall sets, per-seat remaining counts.
  - Seat layout via `seat_specs`: 0=bottom→top, 1=top→bottom, 2=left→right, 3=right→left.
  - Jump rule: official Quoridor (straight over one pawn if landing is in-board/unblocked/empty, else diagonals beside it; no double-jump).
  - Wall counts are deliberately scaled down for the 5×5 POC (N≥3 → 2 walls/seat). Override `max_walls_per_player` for coalition-emergence experiments.
- tensor_spec_mp.py
  - N-player tensor (3N+3 channels): N pawn planes, 2 shared wall maps, N walls-remaining planes, N BFS distance maps, 1 turn plane.
  - Parity with `tensor_spec.py` at N=2 is verified by `tests/test_env_mp.py::TestTensorParity`.

---

### 2. model — Neural network

- network.py
  - Implements `QuoridorNetwork` and `QuoridorModel` (2-player, scalar value)
  - Architecture:
    - input conv layer
    - residual tower (`ResidualBlock`)
    - dual heads:
      - policy head → action probabilities
      - value head → win probability in [-1, 1] (scalar)
  - `QuoridorModel` wrapper provides:
    - `predict(state_tensor)` for inference
    - `train_step(states, policies, values)` for training
    - `save(path)` and `load(path)` for checkpointing
  - Uses PyTorch
- network_mp.py
  - `QuoridorNetworkMP` / `QuoridorModelMP` — same architecture but value head emits length-N vector (tanh per component)
  - No sign-flipping; the vector targets carry absolute-perspective values
  - Same interface as `QuoridorModel` except `predict` returns `(policy, value_vec)` instead of `(policy, scalar)`

---

### 3. mcts — Search and training

- mcts.py
  - Implements AlphaZero-style Monte Carlo Tree Search
  - Key classes:
    - `MCTSConfig` — hyperparameters (`num_simulations`, `c_puct`, `temperature`, `dirichlet_alpha`, etc.)
    - `Node` — tree node storing priors, visit counts, values, children
    - `MCTS` — search logic
  - Search flow:
    - expand root and add Dirichlet noise
    - repeated simulation:
      - select child by UCB
      - step state in env
      - expand leaf and evaluate
      - backpropagate value
    - return visit-count distribution as policy target
  - Supports two modes:
    - random rollout when `evaluate_fn` is `None`
    - neural network evaluation when `evaluate_fn` is provided
- self_play.py
  - Implements 2-player self-play training pipeline
  - Main pieces:
    - `TrainingSample` dataclass
    - `ReplayBuffer`
    - `play_one_game(...)`
      - generates trajectories with MCTS action distributions
      - stores state tensors, MCTS policies, and game result values
    - `training_loop(...)`
      - self-play generation
      - replay-buffer sampling
      - network training
      - evaluation vs best model and vs random
      - checkpointing and logging
  - Uses `CheckpointManager` and `TrainingLogger`
- mcts_maxn.py
  - Max^n MCTS for N players. Node stores length-N value vector; selection maximises Q[parent_mover]; backprop adds same vector to all ancestors (no sign flip).
  - Reduces to negamax at N=2 (proven by `scripts/run_reduction.py`).
- self_play_mp.py
  - Vector value-target assignment: `vec[j] = +disc if j==winner, -disc otherwise, 0 for draw`.
  - No per-position mover-perspective sign flip (the vector design eliminates it).
- evaluator_mp.py
  - N-player evaluation harness. Candidate agent rotates through all N seats; remaining seats filled by opponent. Fair share = 1/N.
- training_mp.py
  - Full N-player AlphaZero loop: max^n self-play → vector-target training → accept/reject gate (candidate rotated through seats) → eval vs random → checkpoint.

---

### 4. server — Inference API

- app.py
  - Flask app exposing `/predict`
  - Loads a `QuoridorModel`
  - Accepts JSON with `state` tensor
  - Returns:
    - `policy` probabilities
    - `value`
  - Designed to support remote inference for the UI or other clients

---

### 5. ui — Human interface

- game_ui.py
  - Pygame-based GUI for human vs AI play
  - Draws the board, pawns, walls, and highlights valid moves
  - Supports:
    - local inference using the PyTorch model
    - remote inference via the Flask server
  - Uses `MCTS` to choose AI actions

---

### 6. utils — Configuration, checkpointing, logging

- config.py
  - Loads JSON configs like config_5x5.json
  - Creates typed config wrappers:
    - `AppConfig`
    - `mcts_config()`
    - `training_config()`
  - exposes `learning_rate`, `weight_decay`, etc.
- checkpoint.py
  - Manages training checkpoints
  - Saves:
    - model weights
    - replay buffer
    - iteration metadata
    - “best model” copy
  - Loads latest checkpoint and allows resume
- logger.py
  - Logs training metrics
  - Supports:
    - local CSV logging
    - optional `wandb` remote logging
  - Records metrics like losses, win rates, game length, buffer size

---

### 7. notebooks — Training experiments

- train_5x5_poc.ipynb
  - Original Colab notebook for training the Quoridor AI on a 5×5 board (POC mode)
  - Includes:
    - GPU verification and anti-disconnect script for Colab
    - Google Drive mounting for persistent checkpoints and logs
    - Repo cloning and dependency installation
    - Validation tests (smoke test of full pipeline)
    - Training loop execution with resume capability
  - Designed for Google Colab with T4 GPU runtime
- train_5x5_poc_v2.ipynb
  - Updated version (v2) with experimental improvements:
    - Discounted value targets (γ=0.97) for stronger endgame signal
    - Data augmentation via left-right mirroring to double training data
    - Reduced walls to 3 per player (37.5% saturation vs 62.5% in v1)
    - Increased training epochs to 100 per iteration for better buffer utilization
    - Exploration bonus: temperature 1.0 for first 20 moves, 0.3 thereafter
  - Same Colab setup as v1 but with enhanced training techniques

---

## Execution flow

- **main**.py
  - CLI entrypoint for training
  - Loads config and environment
  - Builds `QuoridorModel`
  - Calls `training_loop(...)`
- The config selects 5x5 POC or full 9x9 mode via config_5x5.json or config_9x9.json

---

## Supporting files

- requirements.txt
  - Python dependencies
- config_5x5.json, config_9x9.json
  - training, MCTS, and network hyperparameters
- tests
  - unit tests for environment, MCTS, network, tensor spec, checkpoints, and performance
- manual_server_check.py
  - utility for manually checking server availability

---

## Presentation-friendly summary

1. Problem: Quoridor board game AI
2. Game engine: quoridor_env.py
3. Neural net: network.py
4. Search: mcts.py
5. Training loop: self_play.py
6. Persistence: checkpoint.py
7. UI / API: game_ui.py + app.py
8. Configs/tests: configs, tests

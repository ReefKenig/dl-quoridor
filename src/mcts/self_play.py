"""
Self-Play Training Pipeline
============================
Owner: Iris

Drives the AlphaZero training cycle:
    1. Self-play → generate (state, mcts_policy, outcome) tuples
    2. Train network on collected data
    3. Evaluate new model vs previous best
    4. Repeat

Wire in once Rom's network and Reef's env are ready.
"""

from src.mcts.mcts import MCTS, MCTSConfig
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """Single training example for the neural network."""
    state: np.ndarray           # observation tensor (e.g., 5x5x10)
    policy_target: np.ndarray   # MCTS visit count distribution
    value_target: float         # +1 or -1 from this player's perspective


class ReplayBuffer:
    """
    Fixed-size FIFO buffer of training samples.
    For 5x5 POC, 10k-50k samples is sufficient.
    """

    def __init__(self, max_size: int = 50_000):
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, samples: List[TrainingSample]):
        self.buffer.extend(samples)

    def sample_batch(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        actual_size = min(batch_size, len(self.buffer))
        if actual_size < batch_size:
            logger.warning(
                "Requested batch_size=%d but buffer only has %d samples",
                batch_size, len(self.buffer),
            )
        indices = np.random.choice(
            len(self.buffer), size=actual_size, replace=False
        )
        batch = [self.buffer[i] for i in indices]

        states = np.array([s.state for s in batch], dtype=np.float32)
        policies = np.array([s.policy_target for s in batch], dtype=np.float32)
        values = np.array([s.value_target for s in batch], dtype=np.float32)

        return states, policies, values

    def __len__(self) -> int:
        return len(self.buffer)


def play_one_game(
    env,
    mcts: MCTS,
    temperature_schedule: Optional[dict] = None,
) -> Tuple[List[TrainingSample], Optional[int]]:
    """
    Play a single self-play game. Both sides use the same MCTS + model.

    Args:
        env: QuoridorEnvInterface implementation
        mcts: MCTS instance (with or without NN evaluate_fn)
        temperature_schedule: {move_threshold: temperature}
            e.g., {15: 1.0, 999: 0.1} → explore first 15 moves, then exploit

    Returns:
        samples: list of TrainingSample
        winner: 0 or 1 or None (draw)
    """
    if temperature_schedule is None:
        temperature_schedule = {15: 1.0, 999: 0.1}

    MAX_MOVES = 500  # safety cap — prevent infinite games

    state = env.reset()
    trajectory = []  # (state_tensor, mcts_policy, player_who_moved)
    move_count = 0

    while True:
        if move_count >= MAX_MOVES:
            winner = None
            break

        # Temperature schedule
        temp = 0.1
        for threshold, t in sorted(temperature_schedule.items()):
            if move_count < threshold:
                temp = t
                break

        # MCTS search (pass temperature explicitly, don't mutate config)
        action_probs = mcts.search(env, state, temperature=temp)

        # Record training data
        current_player = env.get_current_player(state)
        state_tensor = env.state_to_tensor(state)
        trajectory.append((state_tensor, action_probs, current_player))

        # Sample action
        action = np.random.choice(len(action_probs), p=action_probs)
        state, reward, done, info = env.step(state, action)
        move_count += 1

        if done:
            winner = info.get("winner")
            break

    # Assign value targets based on game outcome
    samples = []
    for state_tensor, policy, player in trajectory:
        if winner is None:
            value = 0.0
        elif player == winner:
            value = 1.0
        else:
            value = -1.0

        samples.append(TrainingSample(
            state=state_tensor,
            policy_target=policy,
            value_target=value,
        ))

    return samples, winner


@dataclass
class TrainingConfig:
    """Full AlphaZero training loop configuration."""
    num_iterations: int = 50
    games_per_iteration: int = 100
    batch_size: int = 64
    training_epochs: int = 10
    eval_games: int = 40
    win_threshold: float = 0.55
    mcts_simulations: int = 400
    replay_buffer_size: int = 50_000


def training_loop(env, model, config: TrainingConfig = None):
    """
    Main AlphaZero training loop.

    TODO (wire in when Rom's model is ready):
        - model.predict(tensor) -> (policy, value)
        - model.train_step(states, policies, values) -> (loss_p, loss_v)
        - model.save(path) / model.load(path)
    """
    if config is None:
        config = TrainingConfig()

    buffer = ReplayBuffer(max_size=config.replay_buffer_size)

    def nn_evaluate(state):
        tensor = env.state_to_tensor(state)
        # TODO: uncomment when model is ready
        # policy, value = model.predict(tensor)
        # return policy, value
        raise NotImplementedError("Wire in Rom's model here")

    mcts = MCTS(
        config=MCTSConfig(num_simulations=config.mcts_simulations),
        evaluate_fn=nn_evaluate,
    )

    for iteration in range(config.num_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}/{config.num_iterations}")

        # --- 1. Self-play ---
        print(f"  Generating {config.games_per_iteration} self-play games...")
        iteration_samples = []
        wins = {0: 0, 1: 0}

        for game_idx in range(config.games_per_iteration):
            samples, winner = play_one_game(env, mcts)
            iteration_samples.extend(samples)
            if winner is not None:
                wins[winner] += 1

            if (game_idx + 1) % 20 == 0:
                print(
                    f"    Games: {game_idx + 1}/{config.games_per_iteration}")

        buffer.add(iteration_samples)
        print(
            f"  Samples collected: {len(iteration_samples)} | Buffer: {len(buffer)}")

        # --- 2. Train ---
        if len(buffer) < config.batch_size:
            print("  Buffer too small, skipping training.")
            continue

        print(f"  Training for {config.training_epochs} epochs...")
        for epoch in range(config.training_epochs):
            states, policies, values = buffer.sample_batch(config.batch_size)
            # TODO: loss_p, loss_v = model.train_step(states, policies, values)
            pass

        # --- 3. Evaluate (TODO) ---
        # Pit new model vs previous checkpoint.
        # Accept if win rate > config.win_threshold.

        print(f"  Iteration {iteration + 1} complete.")

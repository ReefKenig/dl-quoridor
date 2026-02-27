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
from src.mcts.evaluator import evaluate, mcts_agent, random_agent
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import TrainingLogger, Timer
import logging
import numpy as np
from dataclasses import dataclass, asdict
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
    max_moves: int = 500,
) -> Tuple[List[TrainingSample], Optional[int]]:
    """
    Play a single self-play game. Both sides use the same MCTS + model.

    Args:
        env: QuoridorEnvInterface implementation
        mcts: MCTS instance (with or without NN evaluate_fn)
        temperature_schedule: mapping of {move_threshold: temperature}.
            Keys are checked in ascending order; the first threshold
            satisfying ``move_count < threshold`` sets the temperature.
            Example: ``{15: 1.0, 999: 0.1}`` → explore for the first
            15 moves (temp=1.0), then exploit (temp=0.1) for the rest.
        max_moves: safety cap to prevent infinite games

    Returns:
        samples: list of TrainingSample
        winner: 0 or 1 or None (draw)
    """
    if temperature_schedule is None:
        temperature_schedule = {15: 1.0, 999: 0.1}

    state = env.reset()
    trajectory = []  # (state_tensor, mcts_policy, player_who_moved)
    move_count = 0

    while True:
        if move_count >= max_moves:
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
    max_game_moves: int = 500


def training_loop(
    env,
    model,
    config: TrainingConfig = None,
    checkpoint_dir: str = "checkpoints",
    resume: bool = True,
    log_dir: str = "logs",
    use_wandb: bool = True,
):
    """
    Main AlphaZero training loop.

    Cycle per iteration:
        1. Self-play → collect (state, mcts_policy, outcome) tuples
        2. Train network on replay buffer
        3. Evaluate model vs random baseline
        4. Checkpoint (saves best model when win-rate improves)

    Args:
        env: QuoridorEnvInterface implementation
        model: object with predict / train_step / save / load interface
        config: training hyperparameters
        checkpoint_dir: directory for training checkpoints
        resume: if True, attempt to resume from the latest checkpoint
        log_dir: directory for CSV / JSON metrics logs
        use_wandb: enable wandb remote logging (falls back to CSV if unavailable)
    """
    if config is None:
        config = TrainingConfig()

    buffer = ReplayBuffer(max_size=config.replay_buffer_size)
    ckpt = CheckpointManager(base_dir=checkpoint_dir, keep_last_n=3)
    train_logger = TrainingLogger(
        project="quoridor-ai",
        run_name=checkpoint_dir.replace("/", "_").replace("\\", "_"),
        log_dir=log_dir,
        use_wandb=use_wandb,
        config=asdict(config),
    )

    # --- Resume from checkpoint if available ---
    start_iteration = 0
    best_win_rate = 0.0

    if resume:
        state = ckpt.load_latest(
            replay_buffer_max_size=config.replay_buffer_size,
        )
        if state is not None:
            start_iteration = state["iteration"]
            model.load(state["model_path"])
            buffer = state["replay_buffer"]
            best_win_rate = state.get("metrics", {}).get(
                "win_rate_vs_random", 0.0,
            )
            logger.info(
                "Resumed from iteration %d (buffer=%d, best_wr=%.1f%%)",
                start_iteration, len(buffer), best_win_rate * 100,
            )

    # --- NN evaluation closure for MCTS ---
    def nn_evaluate(game_state):
        tensor = env.state_to_tensor(game_state)
        policy, value = model.predict(tensor)
        return policy, value

    mcts = MCTS(
        config=MCTSConfig(num_simulations=config.mcts_simulations),
        evaluate_fn=nn_evaluate,
    )

    for iteration in range(start_iteration, config.num_iterations):
        logger.info("=" * 50)
        logger.info("Iteration %d/%d", iteration + 1, config.num_iterations)

        # --- 1. Self-play ---
        logger.info(
            "  Generating %d self-play games...", config.games_per_iteration,
        )
        iteration_samples = []
        wins = {0: 0, 1: 0}

        with Timer() as sp_timer:
            for game_idx in range(config.games_per_iteration):
                samples, winner = play_one_game(
                    env, mcts, max_moves=config.max_game_moves,
                )
                iteration_samples.extend(samples)
                if winner is not None:
                    wins[winner] += 1

                if (game_idx + 1) % 20 == 0:
                    logger.info(
                        "    Games: %d/%d",
                        game_idx + 1, config.games_per_iteration,
                    )

        buffer.add(iteration_samples)
        logger.info(
            "  Samples collected: %d | Buffer: %d",
            len(iteration_samples), len(buffer),
        )

        # --- 2. Train ---
        if len(buffer) < config.batch_size:
            logger.info("  Buffer too small, skipping training.")
            continue

        logger.info("  Training for %d epochs...", config.training_epochs)
        total_loss_p, total_loss_v = 0.0, 0.0
        with Timer() as train_timer:
            for epoch in range(config.training_epochs):
                states, policies, values = buffer.sample_batch(
                    config.batch_size)
                loss_p, loss_v = model.train_step(states, policies, values)
                total_loss_p += loss_p
                total_loss_v += loss_v

        avg_lp = total_loss_p / max(config.training_epochs, 1)
        avg_lv = total_loss_v / max(config.training_epochs, 1)
        logger.info(
            "  Avg losses — policy: %.4f, value: %.4f", avg_lp, avg_lv,
        )

        # --- 3. Evaluate ---
        # TODO: upgrade to evaluate vs previous best model once a second
        #       model instance can be provided by the caller.
        candidate = mcts_agent(mcts, temperature=0.1)
        with Timer() as eval_timer:
            result = evaluate(
                env, agent_a=candidate, agent_b=random_agent(),
                num_games=config.eval_games, verbose=True,
            )
        logger.info("  Eval: %s", result.summary())

        win_rate = result.agent_a_win_rate
        is_best = win_rate > best_win_rate
        if is_best:
            best_win_rate = win_rate
            logger.info("  New best win rate: %.1f%%", win_rate * 100)

        # --- 4. Checkpoint ---
        ckpt.save(
            iteration=iteration + 1,
            model=model,
            replay_buffer=buffer,
            metrics={
                "loss_policy": avg_lp,
                "loss_value": avg_lv,
                "win_rate_vs_random": win_rate,
            },
            is_best=is_best,
        )

        # --- 5. Log metrics ---
        train_logger.log_iteration(
            iteration=iteration + 1,
            loss_policy=avg_lp,
            loss_value=avg_lv,
            win_rate_vs_random=win_rate,
            avg_game_length=result.avg_game_length,
            total_games_played=config.games_per_iteration,
            self_play_duration_s=sp_timer.elapsed_s,
            training_duration_s=train_timer.elapsed_s,
            eval_duration_s=eval_timer.elapsed_s,
            buffer_size=len(buffer),
            samples_generated=len(iteration_samples),
            model_accepted=is_best,
        )

        logger.info("  Iteration %d complete. best=%s", iteration + 1, is_best)

    train_logger.finish()

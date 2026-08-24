"""
Self-Play Training Pipeline
============================
Drives the AlphaZero training cycle:
    1. Self-play → generate (state, mcts_policy, outcome) tuples
    2. Train network on collected data
    3. Evaluate new model vs current best (accept/reject gate)
    4. Evaluate vs random (absolute strength tracking)
    5. Checkpoint & log
    6. Repeat
"""

import copy
import logging
import os
from collections import deque
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.mcts.evaluator import evaluate, mcts_agent, random_agent
from src.mcts.mcts import MCTS, MCTSConfig
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import Timer, TrainingLogger

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """Single training example for the neural network."""

    state: np.ndarray  # observation tensor (e.g., 5x5x10)
    policy_target: np.ndarray  # MCTS visit count distribution
    value_target: float  # discounted value from this player's perspective


def _augment_sample(
    sample: TrainingSample,
    board_size: int,
    action_space_size: int,
) -> TrainingSample:
    """Mirror a training sample along the vertical axis (left-right flip).

    The 5×5 (and 9×9) board is symmetric about the vertical centre line.
    Flipping doubles the effective training data for free.
    """
    # Flip the state tensor: reverse columns (axis 1)
    flipped_state = sample.state[:, ::-1, :].copy()

    # Flip the policy: mirror pawn move actions and wall columns
    flipped_policy = np.zeros_like(sample.policy_target)
    W = board_size - 1
    h_offset = 12
    v_offset = 12 + W**2

    # Pawn move mirror map (left <-> right, diagonals swap)
    mirror_map = {
        0: 0,
        1: 1,
        2: 3,
        3: 2,  # UP, DOWN, LEFT<->RIGHT
        4: 4,
        5: 5,
        6: 7,
        7: 6,  # JUMP_UP, JUMP_DOWN, JUMP_LEFT<->JUMP_RIGHT
        8: 9,
        9: 8,
        10: 11,
        11: 10,  # diagonals flip
    }

    for action in range(action_space_size):
        if action < 12:
            flipped_policy[mirror_map[action]] = sample.policy_target[action]
        elif action < v_offset:
            # Horizontal wall: flip column index
            w = action - h_offset
            r, c = w // W, w % W
            flipped_c = W - 1 - c
            flipped_policy[h_offset + r * W + flipped_c] = sample.policy_target[action]
        else:
            # Vertical wall: flip column index
            w = action - v_offset
            r, c = w // W, w % W
            flipped_c = W - 1 - c
            flipped_policy[v_offset + r * W + flipped_c] = sample.policy_target[action]

    return TrainingSample(
        state=flipped_state,
        policy_target=flipped_policy,
        value_target=sample.value_target,
    )


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
                batch_size,
                len(self.buffer),
            )
        indices = np.random.choice(len(self.buffer), size=actual_size, replace=False)
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
        temperature_schedule = {20: 1.0, 999: 0.3}

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

    # Assign value targets with temporal discount
    # Positions near the end of the game get stronger signal (closer to ±1),
    # while early positions get weaker signal.  This helps the value head
    # learn a gradient of position quality instead of flat binary labels.
    samples = []
    num_moves = len(trajectory)
    discount = 0.97

    for idx, (state_tensor, policy, player) in enumerate(trajectory):
        if winner is None:
            value = 0.0
        else:
            moves_from_end = num_moves - idx
            discounted = discount**moves_from_end
            if player == winner:
                value = discounted
            else:
                value = -discounted

        samples.append(
            TrainingSample(
                state=state_tensor,
                policy_target=policy,
                value_target=value,
            )
        )

    # Augment with left-right mirror (doubles effective training data)
    augmented = [
        _augment_sample(s, env.board_size, env.action_space_size) for s in samples
    ]
    samples.extend(augmented)

    return samples, winner


@dataclass
class TrainingConfig:
    """Full AlphaZero training loop configuration."""

    num_iterations: int = 50
    games_per_iteration: int = 100
    batch_size: int = 64
    training_epochs: int = 10
    eval_games: int = 40
    eval_random_games: int = 20  # quick eval vs random for absolute strength
    win_threshold: float = 0.55
    mcts_simulations: int = 400
    replay_buffer_size: int = 50_000
    max_game_moves: int = 500
    # save progress every N games during self-play (0 = disabled)
    self_play_checkpoint_freq: int = 10


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
        3. Evaluate new model vs current best (accept/reject gate)
        4. Quick eval vs random (absolute strength tracking)
        5. Checkpoint (saves best model when accepted)
        6. Log metrics

    Args:
        env: QuoridorEnvInterface implementation
        model: object with predict / train_step / save / load / copy_weights_from
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
    resume_self_play_game = 0
    loaded_state = None

    if resume:
        loaded_state = ckpt.load_latest(
            replay_buffer_max_size=config.replay_buffer_size,
        )
        if loaded_state is not None:
            start_iteration = loaded_state["iteration"]
            model.load(loaded_state["model_path"])
            buffer = loaded_state["replay_buffer"]
            # Mid-iteration resume
            mid_game = loaded_state.get("metrics", {}).get("self_play_game")
            if mid_game is not None:
                resume_self_play_game = mid_game
                logger.warning(
                    "MID-ITERATION RESUME: iteration %d, "
                    "resuming from game %d/%d (skipping %d already-played games)",
                    start_iteration,
                    resume_self_play_game,
                    config.games_per_iteration,
                    resume_self_play_game,
                )
            else:
                logger.info(
                    "Resuming from completed iteration %d",
                    start_iteration,
                )
            logger.info(
                "Checkpoint loaded: iteration=%d, buffer=%d samples",
                start_iteration,
                len(buffer),
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

    # --- Best model for AlphaZero accept/reject gate ---
    best_model = copy.deepcopy(model)

    if resume and loaded_state is not None:
        best_model_path = loaded_state.get("best_model_path", "")
        if best_model_path and os.path.exists(best_model_path):
            best_model.load(best_model_path)
            logger.info("Loaded best model from %s", best_model_path)

    def best_nn_evaluate(game_state):
        tensor = env.state_to_tensor(game_state)
        return best_model.predict(tensor)

    best_mcts = MCTS(
        config=MCTSConfig(num_simulations=config.mcts_simulations),
        evaluate_fn=best_nn_evaluate,
    )

    for iteration in range(start_iteration, config.num_iterations):
        logger.info("=" * 50)
        logger.info("Iteration %d/%d", iteration + 1, config.num_iterations)

        # --- 1. Self-play ---
        start_game = resume_self_play_game if iteration == start_iteration else 0
        remaining_games = config.games_per_iteration - start_game
        if start_game > 0:
            logger.info(
                "  Resuming self-play from game %d/%d (%d remaining)...",
                start_game,
                config.games_per_iteration,
                remaining_games,
            )
        else:
            logger.info(
                "  Generating %d self-play games...",
                config.games_per_iteration,
            )
        iteration_samples = []
        iteration_sample_count = 0
        wins = {0: 0, 1: 0}

        with Timer() as sp_timer:
            for game_idx in range(start_game, config.games_per_iteration):
                samples, winner = play_one_game(
                    env,
                    mcts,
                    max_moves=config.max_game_moves,
                )
                iteration_samples.extend(samples)
                if winner is not None:
                    wins[winner] += 1

                if (game_idx + 1) % 20 == 0:
                    logger.info(
                        "    Games: %d/%d",
                        game_idx + 1,
                        config.games_per_iteration,
                    )

                # Mid-iteration checkpoint
                freq = config.self_play_checkpoint_freq
                if (
                    freq > 0
                    and (game_idx + 1) % freq == 0
                    and (game_idx + 1) < config.games_per_iteration
                ):
                    buffer.add(iteration_samples)
                    iteration_sample_count += len(iteration_samples)
                    iteration_samples = []
                    ckpt.save(
                        iteration=iteration,
                        model=model,
                        replay_buffer=buffer,
                        metrics={"self_play_game": game_idx + 1},
                        is_best=False,
                    )
                    logger.info(
                        "    Mid-iteration checkpoint saved (%d/%d games)",
                        game_idx + 1,
                        config.games_per_iteration,
                    )

        buffer.add(iteration_samples)
        iteration_sample_count += len(iteration_samples)
        logger.info(
            "  Samples collected: %d | Buffer: %d",
            iteration_sample_count,
            len(buffer),
        )

        # --- 2. Train ---
        if len(buffer) < config.batch_size:
            logger.info(
                "  Buffer too small (%d < %d), skipping training.",
                len(buffer),
                config.batch_size,
            )
            train_logger.log_iteration(
                iteration=iteration + 1,
                buffer_size=len(buffer),
                samples_generated=iteration_sample_count,
                self_play_duration_s=sp_timer.elapsed_s,
                total_games_played=config.games_per_iteration,
            )
            continue

        logger.info("  Training for %d epochs...", config.training_epochs)
        total_loss_p, total_loss_v = 0.0, 0.0

        # Calculate how many batchesmake up onefull pass of the current buffer
        steps_per_epoch = max(1, len(buffer) // config.batch_size)
        total_steps = steps_per_epoch * config.training_epochs

        with Timer() as train_timer:
            for step in range(total_steps):
                states, policies, values = buffer.sample_batch(config.batch_size)
                loss_p, loss_v = model.train_step(states, policies, values)
                total_loss_p += loss_p
                total_loss_v += loss_v

        # Average the loss over the total number of gradient steps
        avg_lp = total_loss_p / max(total_steps, 1)
        avg_lv = total_loss_v / max(total_steps, 1)
        logger.info(
            "  Avg losses - policy: %.4f, value: %.4f",
            avg_lp,
            avg_lv,
        )

        # --- 3. Evaluate new model vs current best (accept/reject) ---
        candidate = mcts_agent(mcts, temperature=0.1)
        champion = mcts_agent(best_mcts, temperature=0.1)
        with Timer() as eval_best_timer:
            result_vs_best = evaluate(
                env,
                agent_a=candidate,
                agent_b=champion,
                num_games=config.eval_games,
                verbose=True,
            )
        logger.info("  Eval vs best: %s", result_vs_best.summary())

        win_rate_vs_best = result_vs_best.agent_a_win_rate
        is_best = result_vs_best.should_accept(threshold=config.win_threshold)
        if is_best:
            best_model.copy_weights_from(model)
            logger.info(
                "  Model ACCEPTED (%.1f%% > %.0f%% threshold)",
                win_rate_vs_best * 100,
                config.win_threshold * 100,
            )
        else:
            logger.info(
                "  Model REJECTED (%.1f%% <= %.0f%% threshold)",
                win_rate_vs_best * 100,
                config.win_threshold * 100,
            )

        # --- 4. Quick eval vs random (absolute strength) ---
        win_rate_vs_random = 0.0
        avg_game_length = result_vs_best.avg_game_length
        eval_random_duration = 0.0

        if config.eval_random_games > 0:
            with Timer() as eval_random_timer:
                result_vs_random = evaluate(
                    env,
                    agent_a=candidate,
                    agent_b=random_agent(),
                    num_games=config.eval_random_games,
                    verbose=False,
                )
            win_rate_vs_random = result_vs_random.agent_a_win_rate
            avg_game_length = result_vs_random.avg_game_length
            eval_random_duration = eval_random_timer.elapsed_s
            logger.info(
                "  Eval vs random: %s",
                result_vs_random.summary(),
            )

        # --- 5. Checkpoint ---
        ckpt.save(
            iteration=iteration + 1,
            model=model,
            replay_buffer=buffer,
            metrics={
                "loss_policy": avg_lp,
                "loss_value": avg_lv,
                "win_rate_vs_best": win_rate_vs_best,
                "win_rate_vs_random": win_rate_vs_random,
            },
            is_best=is_best,
        )

        # --- 6. Log metrics ---
        train_logger.log_iteration(
            iteration=iteration + 1,
            loss_policy=avg_lp,
            loss_value=avg_lv,
            win_rate_vs_best=win_rate_vs_best,
            win_rate_vs_random=win_rate_vs_random,
            avg_game_length=avg_game_length,
            total_games_played=config.games_per_iteration,
            self_play_duration_s=sp_timer.elapsed_s,
            training_duration_s=train_timer.elapsed_s,
            eval_duration_s=eval_best_timer.elapsed_s + eval_random_duration,
            buffer_size=len(buffer),
            samples_generated=iteration_sample_count,
            model_accepted=is_best,
        )

        logger.info("  Iteration %d complete. accepted=%s", iteration + 1, is_best)

    train_logger.finish()

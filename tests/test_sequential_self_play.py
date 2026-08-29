"""Sequential self-play must survive a full iteration.

Only the parallel and vectorized branches assigned sp_samples; the sequential
branch fed the buffer per game and left the name unbound, so the unconditional
sample_diagnostics call after training raised UnboundLocalError. Found by the
factored-head validation smoke; the crash is mode-specific, not head-specific.
"""
import os

from src.env.quoridor_env_mp import QuoridorEnvMP, compute_action_space_size
from src.mcts.training_mp import TrainingConfigMP, training_loop_mp
from src.model.network_mp import QuoridorModelMP


def test_sequential_mode_completes_an_iteration(tmp_path):
    board, n = 5, 2
    env = QuoridorEnvMP(board_size=board, num_players=n, max_turns=40,
                        max_walls_per_player=3)

    def make_model():
        return QuoridorModelMP(
            board_size=board,
            action_space_size=compute_action_space_size(board),
            num_channels=8, num_res_blocks=1, num_players=n, device="cpu")

    cfg = TrainingConfigMP(
        num_players=n, num_iterations=1, games_per_iteration=1,
        mcts_simulations=2, eval_simulations=2, batch_size=8,
        train_steps_per_iter=1, warmup_min_samples=1,
        eval_games=0, eval_random_games=0, eval_greedy_games=0,
        eval_minimax_games=0, max_game_moves=40, explore_moves=2,
        self_play_mode="sequential", parallel_self_play=False,
        parallel_eval=False, eval_every=100)

    training_loop_mp(env, make_model(), make_model, cfg,
                     checkpoint_dir=str(tmp_path))

    assert os.path.exists(os.path.join(str(tmp_path), "meta.json"))

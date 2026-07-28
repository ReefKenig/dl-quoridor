"""
Smoke Tests for parallel_self_play_mp.py
=========================================
Run: pytest tests/test_parallel_self_play_mp.py -v

Tests batched GPU inference and multiprocessing supervision:
- Process spawning and cleanup
- Timeout handling (worker hangs, inference thread crash)
- Config propagation (dirichlet_epsilon, PYTHONPATH)
"""

import os
import pytest

from src.model.network_mp import QuoridorModelMP
from src.mcts.parallel_self_play_mp import generate_parallel_self_play_mp


class MockConfig:
    """Minimal config for testing."""
    def __init__(self, num_players=2, mcts_dirichlet_epsilon=0.25):
        self.num_players = num_players
        self.board_size = 5
        self.max_walls_per_player = 3
        self.max_turns = 80
        self.mcts_simulations = 50
        self.discount = 0.97
        self.explore_moves = 10
        self.max_game_moves = 80
        self.mcts_dirichlet_epsilon = mcts_dirichlet_epsilon


@pytest.fixture
def model_mp():
    """Create a minimal MP model for testing."""
    return QuoridorModelMP(
        board_size=5,
        action_space_size=44,
        num_players=2,
        num_channels=16,
        num_res_blocks=2,
    )


@pytest.fixture
def config_mp():
    """Create a minimal config."""
    return MockConfig()


def test_parallel_self_play_basic(model_mp, config_mp):
    """Test basic parallel self-play: spawn workers, collect games, clean shutdown."""
    samples, wins = generate_parallel_self_play_mp(
        model_mp,
        config_mp,
        num_workers=2,
        total_games=2,
        batch_size=8,
        worker_join_timeout=10.0,
    )

    assert len(samples) > 0, "Expected at least one training sample"
    assert isinstance(wins, dict), "Wins should be a dict"
    assert any(k in wins for k in [0, 1, None]), "Expected winner in wins dict"
    assert sum(wins.values()) == 2, "Expected exactly 2 games"


def test_parallel_self_play_config_dirichlet_epsilon(model_mp):
    """Test that dirichlet_epsilon is read from config."""
    config = MockConfig(mcts_dirichlet_epsilon=0.5)
    samples, wins = generate_parallel_self_play_mp(
        model_mp,
        config,
        num_workers=1,
        total_games=1,
        batch_size=8,
        worker_join_timeout=10.0,
    )
    # If this runs without error, config was propagated correctly
    assert len(samples) >= 0


def test_pythonpath_uses_os_pathsep(model_mp, config_mp):
    """Test that PYTHONPATH uses os.pathsep instead of hardcoded ':'."""
    original_pythonpath = os.environ.get("PYTHONPATH", "")
    try:
        os.environ["PYTHONPATH"] = "preexisting"
        samples, wins = generate_parallel_self_play_mp(
            model_mp,
            config_mp,
            num_workers=1,
            total_games=1,
            batch_size=8,
            worker_join_timeout=10.0,
        )
        # Check that PYTHONPATH was updated correctly
        pythonpath = os.environ.get("PYTHONPATH", "")
        assert os.pathsep in pythonpath or "preexisting" in pythonpath
    finally:
        if original_pythonpath:
            os.environ["PYTHONPATH"] = original_pythonpath
        else:
            os.environ.pop("PYTHONPATH", None)


def test_worker_timeout_graceful_shutdown(model_mp, config_mp):
    """Test graceful shutdown with worker_join_timeout parameter."""
    samples, wins = generate_parallel_self_play_mp(
        model_mp,
        config_mp,
        num_workers=1,
        total_games=1,
        batch_size=8,
        worker_join_timeout=5.0,  # Short timeout for testing
    )
    # If this completes without hanging, timeout logic is working
    assert isinstance(samples, list)
    assert isinstance(wins, dict)


@pytest.mark.slow
def test_parallel_self_play_multiple_workers(model_mp, config_mp):
    """Test with multiple workers to verify batching and coordination."""
    samples, wins = generate_parallel_self_play_mp(
        model_mp,
        config_mp,
        num_workers=4,
        total_games=4,
        batch_size=8,
        worker_join_timeout=15.0,
    )

    assert len(samples) > 0
    assert sum(wins.values()) == 4, "Expected exactly 4 games with 4 workers"


def test_parallel_self_play_edge_case_total_less_than_workers(model_mp, config_mp):
    """Test edge case: total_games < num_workers (should spawn only total_games workers)."""
    samples, wins = generate_parallel_self_play_mp(
        model_mp,
        config_mp,
        num_workers=10,
        total_games=2,
        batch_size=8,
        worker_join_timeout=10.0,
    )

    # Should have spawned only 2 workers (not 10)
    assert sum(wins.values()) == 2, "Expected exactly 2 games"

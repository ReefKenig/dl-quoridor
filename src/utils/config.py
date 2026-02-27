"""
Configuration Loader
=====================
Loads JSON config files and creates typed config dataclass instances.

Usage:
    from src.utils.config import load_config

    cfg = load_config("configs/config_5x5.json")
    mcts_cfg = cfg.mcts_config()
    train_cfg = cfg.training_config()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from src.mcts.mcts import MCTSConfig
    from src.mcts.self_play import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Typed wrapper around the raw JSON config."""
    is_poc: bool
    board_size: int
    raw: Dict[str, Any]

    def mcts_config(self) -> MCTSConfig:
        """Create MCTSConfig from the 'mcts' section."""
        from src.mcts.mcts import MCTSConfig

        m = self.raw.get("mcts", {})
        return MCTSConfig(
            num_simulations=m.get("num_simulations", 400),
            c_puct=m.get("c_puct", 1.41),
            temperature=m.get("temperature", 1.0),
            dirichlet_alpha=m.get("dirichlet_alpha", 0.3),
            dirichlet_epsilon=m.get("dirichlet_epsilon", 0.25),
            max_rollout_depth=m.get("max_rollout_depth", 100),
        )

    def training_config(self) -> TrainingConfig:
        """Create TrainingConfig from the 'training' section."""
        from src.mcts.self_play import TrainingConfig

        t = self.raw.get("training", {})
        m = self.raw.get("mcts", {})
        return TrainingConfig(
            num_iterations=t.get("num_iterations", 50),
            games_per_iteration=t.get("games_per_iteration", 100),
            batch_size=t.get("batch_size", 64),
            training_epochs=t.get("training_epochs", 10),
            eval_games=t.get("eval_games", 40),
            win_threshold=t.get("win_threshold", 0.55),
            mcts_simulations=t.get(
                "mcts_simulations", m.get("num_simulations", 400),
            ),
            replay_buffer_size=t.get("replay_buffer_size", 50_000),
        )

    def network_config(self) -> Dict[str, Any]:
        """Return the 'network' section as a dict."""
        return self.raw.get("network", {})

    @property
    def learning_rate(self) -> float:
        return self.raw.get("training", {}).get("learning_rate", 0.001)

    @property
    def weight_decay(self) -> float:
        return self.raw.get("training", {}).get("weight_decay", 0.0001)


def load_config(path: str = "configs/config_5x5.json") -> AppConfig:
    """
    Load a JSON config file and return a typed AppConfig.

    Args:
        path: path to the JSON config file (relative or absolute)

    Returns:
        AppConfig instance

    Raises:
        FileNotFoundError: if config file doesn't exist
        json.JSONDecodeError: if config file is invalid JSON
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path.resolve()}")

    with open(config_path, "r") as f:
        raw = json.load(f)

    is_poc = raw.get("is_poc", True)
    board_size = raw.get("board_size", 5 if is_poc else 9)
    logger.info(
        "Loaded config: is_poc=%s, board_size=%d from %s",
        is_poc, board_size, config_path,
    )

    return AppConfig(is_poc=is_poc, board_size=board_size, raw=raw)

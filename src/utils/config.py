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

from src.env.pathing import SPEC_V1_DIST_SQ
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from src.mcts.mcts import MCTSConfig
    from src.mcts.self_play import TrainingConfig

logger = logging.getLogger(__name__)

# Root Dirichlet noise is a SELF-PLAY exploration device and is off at every
# difficulty: at 9x9 it lifts opening wall prior mass from 0.00024 to 0.245, so
# a quarter of the demo's moves became near-random walls MCTS could not reject
# (configs/config_9x9.json, _wall_mask_note). Difficulty is simulations and
# temperature. "hard" is the configuration every reported number was measured
# under - eps 0, temperature ~0, c_puct 1.41 (scripts/probe_greedy.py, and
# mcts_c_puct in every run config); it is the only preset that reproduces them.
DIFFICULTY_SETTINGS = {
    "easy": {"num_simulations": 50, "temperature": 1.5, "c_puct": 1.41, "dirichlet_epsilon": 0.0},
    "medium": {"num_simulations": 200, "temperature": 0.4, "c_puct": 1.41, "dirichlet_epsilon": 0.0},
    "hard": {"num_simulations": 1200, "temperature": 0.0, "c_puct": 1.41, "dirichlet_epsilon": 0.0},
}
DEFAULT_DIFFICULTY = "medium"


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
                "mcts_simulations",
                m.get("num_simulations", 400),
            ),
            replay_buffer_size=t.get("replay_buffer_size", 50_000),
            max_game_moves=t.get("max_game_moves", 500),
            eval_random_games=t.get("eval_random_games", 20),
            self_play_checkpoint_freq=t.get("self_play_checkpoint_freq", 10),
        )

    def network_config(self) -> Dict[str, Any]:
        """Return the 'network' section as a dict."""
        return self.raw.get("network", {})

    @property
    def reward_decay(self) -> float:
        return self.raw.get("training", {}).get("reward_decay", 0.97)

    @property
    def max_walls_per_player(self) -> Optional[int]:
        return self.raw.get("max_walls_per_player", None)

    @property
    def learning_rate(self) -> float:
        return self.raw.get("training", {}).get("learning_rate", 0.001)

    @property
    def weight_decay(self) -> float:
        return self.raw.get("training", {}).get("weight_decay", 0.0001)


def resolve_variant(raw: Dict[str, Any], variant: str) -> Dict[str, Any]:
    """The `training` block with `variants[variant]` merged over it.

    Some settings cannot be shared across player counts. `max_game_moves` counts
    individual plies, not rounds, so a single shared value hands N=2 players
    twice the per-player budget of N=4 players on the same board: 160 plies is 80
    turns each at N=2 but only 40 each at N=4. That is what drove the N=4 9x9
    timeout rate to 84-90% and collapsed its value head.
    """
    merged = dict(raw.get("training", {}))
    merged.update(raw.get("variants", {}).get(variant, {}))
    return merged


RUN_SECTIONS = ("mcts", "network", "training", "parallel")



def resolve_run_config(raw: Dict[str, Any], variant: str) -> Dict[str, Any]:
    """Every setting for one run, flattened into a single dict.

    The sections organise the JSON; they are not meaningful to a caller, which
    wants `num_simulations` and `num_iterations` from the same place. Reading
    some settings per-section and others through the variant merge is what let
    the 9x9 notebooks silently ignore half the config. Variant overrides are
    applied last, so per-player-count settings win.

    Keys starting with "_" are notes and are dropped. A key defined in two
    sections raises: it would otherwise resolve by section order, and quietly
    depend on it.
    """
    out = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    origin = {k: "<top level>" for k in out}
    for section in RUN_SECTIONS:
        for key, value in raw.get(section, {}).items():
            if key.startswith("_"):
                continue
            if key in origin:
                raise ValueError(
                    f"{key!r} is defined in both {origin[key]} and {section!r}; "
                    "a flattened run config cannot resolve it unambiguously.")
            origin[key] = section
            out[key] = value
    out.update({k: v for k, v in raw.get("variants", {}).get(variant, {}).items()
                if not k.startswith("_")})
    return out


def ply_budget_per_player(raw: Dict[str, Any], variant: str) -> float:
    """Plies each player gets before a game is cut short at `max_game_moves`.

    The number that should be comparable across variants - not the raw cap.
    """
    merged = resolve_variant(raw, variant)
    num_players = merged.get("num_players")
    if not num_players:
        raise KeyError(f"variant {variant!r} does not declare num_players")
    return merged["max_game_moves"] / num_players


def env_max_turns(raw: Dict[str, Any], variant: str) -> int:
    """The env's no-winner cutoff for a run, which may never bind first.

    The env stops at `max_turns`, the drivers stop rolling out at
    `max_game_moves`, and the smaller wins. `max_rollout_depth` is shared while
    `max_game_moves` is per-variant, so deriving the cutoff is what keeps the
    pair from diverging.
    """
    merged = resolve_run_config(raw, variant)
    return max(int(merged.get("max_rollout_depth", 0)),
               int(merged["max_game_moves"]))


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
        is_poc,
        board_size,
        config_path,
    )

    return AppConfig(is_poc=is_poc, board_size=board_size, raw=raw)


def read_frozen_config(run_dir):
    """A run dir's config.json, or None when absent or unreadable.

    Normalizes spec_version here rather than at each call site: dirs written
    before the tensor spec was versioned have no key, and those are all v1.
    """
    try:
        with open(Path(run_dir) / "config.json") as f:
            frozen = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    frozen.setdefault("spec_version", SPEC_V1_DIST_SQ)
    return frozen

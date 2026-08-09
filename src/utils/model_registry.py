"""
Model Registry
==============
Which checkpoint backs each (board_size, num_players) demo, and how to build the
network and environment that checkpoint expects.

One source of truth for the pygame UI and the web server. A checkpoint loaded
under the wrong architecture fails loudly on a shape mismatch, but one loaded
under the wrong tensor spec or the wrong wall count fails *silently* — the model
just plays badly on planes it never saw during training.

Usage:
    from src.utils.model_registry import load_variant

    env, model, spec = load_variant(board_size=9, num_players=2)
    print(spec.label)   # which file, and why that one
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.checkpoint import resolve_ship_checkpoint

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "runs" / "MODELS.json"

# The 5x5 POC entries predate these being configurable.
DEFAULT_NUM_CHANNELS = 64
DEFAULT_NUM_RES_BLOCKS = 4
DEFAULT_MAX_TURNS = 300


@dataclass
class VariantSpec:
    """Everything needed to reconstruct one demo-ready model."""

    board_size: int
    num_players: int
    model_key: str
    checkpoint: Optional[str]
    label: str
    in_channels: int
    num_channels: int
    num_res_blocks: int
    tensor_spec: int
    max_walls: int
    max_turns: int
    notes: str = ""

    @property
    def is_loadable(self) -> bool:
        return self.checkpoint is not None and os.path.exists(self.checkpoint)


def load_registry(path=None) -> Dict[str, Any]:
    with open(path or REGISTRY_PATH) as f:
        return json.load(f)


def variant_key(board_size: int, num_players: int) -> str:
    return f"{board_size}x{board_size}_{num_players}p"


def available_variants(registry=None) -> Dict[str, Any]:
    """Servable (board, players) combos. Keys opening with "_" are prose."""
    variants = (registry or load_registry()).get("variants", {})
    return {k: v for k, v in variants.items() if not k.startswith("_")}


def variant_spec(board_size: int, num_players: int, registry=None,
                 root=None) -> VariantSpec:
    """Resolve a (board, players) combo to a fully specified VariantSpec.

    Entries carry either `path` (a specific file) or `run_dir` (a run whose
    shipping checkpoint is chosen by resolve_ship_checkpoint), so a run's demo
    model follows the resolver instead of being pinned by hand.
    """
    registry = registry or load_registry()
    root = Path(root) if root else REGISTRY_PATH.parent.parent
    key = variant_key(board_size, num_players)

    variants = available_variants(registry)
    if key not in variants:
        raise KeyError(
            f"No registry variant for {key}. Known: {sorted(variants)}")
    variant = variants[key]
    model_key = variant["model"]
    entry = registry["models"][model_key]

    if "run_dir" in entry:
        checkpoint, label = resolve_ship_checkpoint(root / entry["run_dir"])
        label = f"{entry['run_dir']}/{label}"
    else:
        checkpoint = str(root / entry["path"])
        label = entry["path"]
        if not os.path.exists(checkpoint):
            checkpoint, label = None, f"{entry['path']} (missing)"

    return VariantSpec(
        board_size=board_size,
        num_players=num_players,
        model_key=model_key,
        checkpoint=checkpoint,
        label=label,
        in_channels=entry.get("in_channels", 3 * num_players + 3),
        num_channels=entry.get("num_channels", DEFAULT_NUM_CHANNELS),
        num_res_blocks=entry.get("num_res_blocks", DEFAULT_NUM_RES_BLOCKS),
        tensor_spec=entry["tensor_spec"],
        max_walls=variant["max_walls"],
        max_turns=variant.get("max_turns", DEFAULT_MAX_TURNS),
        notes=entry.get("notes", ""),
    )


def build_env(spec: VariantSpec):
    """The environment the checkpoint was trained against."""
    from src.env.quoridor_env_mp import QuoridorEnvMP

    return QuoridorEnvMP(board_size=spec.board_size,
                         num_players=spec.num_players,
                         max_turns=spec.max_turns,
                         max_walls_per_player=spec.max_walls,
                         spec_version=spec.tensor_spec)


def build_model(spec: VariantSpec, action_space_size: int):
    from src.model.network_mp import QuoridorModelMP

    return QuoridorModelMP(board_size=spec.board_size,
                           action_space_size=action_space_size,
                           in_channels=spec.in_channels,
                           num_channels=spec.num_channels,
                           num_res_blocks=spec.num_res_blocks,
                           num_players=spec.num_players)


def load_variant(board_size: int, num_players: int, registry=None, root=None,
                 log=print):
    """(env, model, spec) with weights loaded when the checkpoint is on disk.

    Returns an untrained model rather than raising when it is not, so the demo
    still starts — but says so, because an untrained net is indistinguishable
    from a trained one at the UI unless it announces itself.
    """
    spec = variant_spec(board_size, num_players, registry=registry, root=root)
    env = build_env(spec)
    model = build_model(spec, env.action_space_size)

    if spec.is_loadable:
        model.load(spec.checkpoint)
        log(f"[{variant_key(board_size, num_players)}] loaded {spec.label}")
    else:
        log(f"[{variant_key(board_size, num_players)}] WARNING: no checkpoint "
            f"({spec.label}) — the AI is an untrained network and will play "
            f"close to randomly")
    return env, model, spec

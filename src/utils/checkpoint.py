"""
Checkpoint Manager
===================
Saves and restores full training state so you can resume after
Colab disconnects, crashes, or session timeouts.

Saves:
    - Current iteration number
    - Model weights (delegates to model.save/load)
    - Best model weights
    - Replay buffer contents
    - Training metrics history

Usage:
    from src.utils.checkpoint import CheckpointManager

    ckpt = CheckpointManager(base_dir="checkpoints")

    # Save during training
    ckpt.save(
        iteration=5,
        model=model,
        replay_buffer=buffer,
        metrics=metrics_dict,
        is_best=True,
    )

    # Resume after crash
    state = ckpt.load_latest()
    if state is not None:
        iteration = state["iteration"]
        model.load(state["model_path"])
        buffer = state["replay_buffer"]
        ...
"""

import json
import pickle
import shutil
import logging
from typing import Optional, Any, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_ship_checkpoint(run_dir):
    """(path, label) of the checkpoint to ship for `run_dir`, or (None, reason).

    `best.pt` is the gating champion and is written up front from the *untrained*
    starting model, so a run where no iteration was ever accepted has a randomly
    initialised best.pt. Shipping it would deploy a network that never trained —
    which is exactly what the n4_9x9_v3 run would have produced. Fall back to
    latest.pt in that case, and always say which one was chosen.
    """
    run = Path(run_dir)
    best, latest = run / "best.pt", run / "latest.pt"

    history = []
    try:
        with open(run / "meta.json") as f:
            history = json.load(f).get("history", [])
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    accepts = [row["iter"] for row in history if row.get("accepted")]
    last_iter = history[-1]["iter"] if history else None

    if accepts and best.exists():
        return str(best), f"best.pt (accepted at iter {accepts[-1]}, {len(accepts)} accepts)"
    if latest.exists():
        why = ("no iteration was ever accepted, so best.pt is the untrained "
               "initialization" if history else "no history in meta.json")
        return str(latest), f"latest.pt (iter {last_iter}) — {why}"
    if best.exists():
        return str(best), "best.pt (no latest.pt found)"
    return None, f"no best.pt or latest.pt in {run_dir}"


class CheckpointManager:
    """
    Manages training checkpoints on disk.

    Directory structure:
        base_dir/
        ├── latest.json              # pointer to latest checkpoint
        ├── iter_005/
        │   ├── meta.json            # iteration, timestamp, metrics
        │   ├── model.pt             # current model weights
        │   ├── best_model.pt        # best model weights
        │   └── replay_buffer.pkl    # serialized buffer
        ├── iter_010/
        │   └── ...
        └── best/
            └── model.pt             # always points to the best model
    """

    def __init__(self, base_dir: str = "checkpoints", keep_last_n: int = 3):
        """
        Args:
            base_dir: root directory for checkpoints
            keep_last_n: number of recent checkpoints to keep (older ones deleted)
        """
        self.base_dir = Path(base_dir)
        self.keep_last_n = keep_last_n
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "best").mkdir(exist_ok=True)

    def save(
        self,
        iteration: int,
        model: Any,
        replay_buffer: Any,
        metrics: Optional[Dict] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save a training checkpoint.

        Args:
            iteration: current training iteration
            model: object with .save(path) method
            replay_buffer: ReplayBuffer instance
            metrics: dict of training metrics for this iteration
            is_best: if True, also copies model to best/

        Returns:
            path to the checkpoint directory
        """
        ckpt_dir = self.base_dir / f"iter_{iteration:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_path = str(ckpt_dir / "model.pt")
        model.save(model_path)

        # Save replay buffer
        buffer_path = ckpt_dir / "replay_buffer.pkl"
        self._save_replay_buffer(replay_buffer, buffer_path)

        # Save metadata
        meta = {
            "iteration": iteration,
            "buffer_size": len(replay_buffer),
            "metrics": metrics or {},
            "is_best": is_best,
            "self_play_game": (metrics or {}).get("self_play_game"),
        }
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Update latest pointer
        with open(self.base_dir / "latest.json", "w") as f:
            json.dump({"iteration": iteration, "path": str(ckpt_dir)}, f)

        # Copy to best/ if this is the best model
        if is_best:
            best_path = self.base_dir / "best" / "model.pt"
            shutil.copy2(model_path, str(best_path))
            logger.info("New best model saved at iteration %d", iteration)

        # Prune old checkpoints
        self._prune_old_checkpoints()

        logger.info(
            "Checkpoint saved: iter=%d, buffer=%d, path=%s",
            iteration,
            len(replay_buffer),
            ckpt_dir,
        )
        return str(ckpt_dir)

    def load_latest(
        self,
        replay_buffer_max_size: int = 50_000,
    ) -> Optional[Dict]:
        """
        Load the most recent checkpoint.

        Args:
            replay_buffer_max_size: max capacity for the loaded replay buffer.
                Pass the current config value so a resumed buffer respects
                any size changes between runs.

        Returns:
            dict with keys: iteration, model_path, best_model_path,
                            replay_buffer, metrics
            or None if no checkpoint exists
        """
        latest_file = self.base_dir / "latest.json"
        if not latest_file.exists():
            logger.info("No checkpoint found in %s", self.base_dir)
            return None

        with open(latest_file, "r") as f:
            latest = json.load(f)

        return self.load_iteration(
            latest["iteration"],
            replay_buffer_max_size=replay_buffer_max_size,
        )

    def load_iteration(
        self,
        iteration: int,
        replay_buffer_max_size: int = 50_000,
    ) -> Optional[Dict]:
        """Load a specific iteration checkpoint."""
        ckpt_dir = self.base_dir / f"iter_{iteration:04d}"
        if not ckpt_dir.exists():
            logger.warning("Checkpoint iter_%04d not found", iteration)
            return None

        # Load metadata
        with open(ckpt_dir / "meta.json", "r") as f:
            meta = json.load(f)

        # Load replay buffer
        buffer_path = ckpt_dir / "replay_buffer.pkl"
        replay_buffer = self._load_replay_buffer(
            buffer_path,
            max_size=replay_buffer_max_size,
        )

        # Model paths (caller loads weights via model.load())
        model_path = str(ckpt_dir / "model.pt")
        best_model_path = str(self.base_dir / "best" / "model.pt")

        state = {
            "iteration": meta["iteration"],
            "model_path": model_path,
            "best_model_path": best_model_path,
            "replay_buffer": replay_buffer,
            "metrics": meta.get("metrics", {}),
            "self_play_game": meta.get("self_play_game"),
        }

        logger.info(
            "Loaded checkpoint: iter=%d, buffer=%d",
            meta["iteration"],
            meta["buffer_size"],
        )
        return state

    def get_best_model_path(self) -> Optional[str]:
        """Return path to the best model, or None if not yet saved."""
        path = self.base_dir / "best" / "model.pt"
        if path.exists():
            return str(path)
        return None

    def list_checkpoints(self) -> List[int]:
        """Return sorted list of saved iteration numbers."""
        iterations = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.startswith("iter_"):
                try:
                    iterations.append(int(item.name.split("_")[1]))
                except (ValueError, IndexError):
                    continue
        return sorted(iterations)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_replay_buffer(self, buffer, path: Path):
        """Serialize replay buffer. Uses pickle — buffer contains numpy arrays."""
        with open(path, "wb") as f:
            pickle.dump(list(buffer.buffer), f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_replay_buffer(self, path: Path, max_size: int = 50_000):
        """Deserialize replay buffer."""
        from src.mcts.self_play import ReplayBuffer  # noqa: F811

        if not path.exists():
            logger.warning("Replay buffer file not found: %s", path)
            return ReplayBuffer(max_size=max_size)

        with open(path, "rb") as f:
            items = pickle.load(f)

        buffer = ReplayBuffer(max_size=max_size)
        buffer.buffer.extend(items)  # deque(maxlen) drops oldest if oversized
        return buffer

    def _prune_old_checkpoints(self):
        """Keep only the N most recent checkpoints + best/."""
        iterations = self.list_checkpoints()
        if len(iterations) <= self.keep_last_n:
            return

        to_remove = iterations[: len(iterations) - self.keep_last_n]
        for it in to_remove:
            ckpt_dir = self.base_dir / f"iter_{it:04d}"
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            logger.debug("Pruned old checkpoint: iter_%04d", it)

"""
Training Logger
================
Owner: Iris

Centralized metrics logging for the training loop.
Wraps wandb for remote tracking + local CSV fallback.

Tracks:
    - Win rate vs random per iteration
    - Policy loss, value loss
    - Average game length
    - MCTS search time
    - Replay buffer size
    - Model acceptance rate (new vs best)

Usage:
    from src.utils.logger import TrainingLogger

    logger = TrainingLogger(project="quoridor-5x5", run_name="run_01")

    # During training
    logger.log_iteration(
        iteration=1,
        loss_policy=0.85,
        loss_value=0.42,
        win_rate_vs_random=0.65,
        avg_game_length=28.3,
        avg_search_time_ms=12.5,
        buffer_size=5000,
        model_accepted=True,
    )

    # At end
    logger.finish()
"""

import csv
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path

module_logger = logging.getLogger(__name__)


@dataclass
class IterationMetrics:
    """All metrics recorded for a single training iteration."""
    iteration: int
    timestamp: float = 0.0

    # Losses
    loss_policy: float = 0.0
    loss_value: float = 0.0
    loss_total: float = 0.0

    # Performance
    win_rate_vs_random: float = 0.0
    win_rate_vs_best: float = 0.0
    model_accepted: bool = False

    # Game stats
    avg_game_length: float = 0.0
    total_games_played: int = 0

    # Timing
    avg_search_time_ms: float = 0.0
    self_play_duration_s: float = 0.0
    training_duration_s: float = 0.0
    eval_duration_s: float = 0.0

    # Buffer
    buffer_size: int = 0
    samples_generated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrainingLogger:
    """
    Dual-mode logger: wandb (remote) + CSV (local fallback).

    wandb is optional — if not installed or not configured, falls back
    to CSV-only mode silently. This way the training loop never crashes
    because of a logging issue.
    """

    def __init__(
        self,
        project: str = "quoridor-ai",
        run_name: Optional[str] = None,
        log_dir: str = "logs",
        use_wandb: bool = True,
        config: Optional[Dict] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[IterationMetrics] = []
        self.wandb_run = None

        # Try to init wandb
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=project,
                    name=run_name,
                    config=config or {},
                    resume="allow",
                    id=run_name,
                )
                module_logger.info(
                    "wandb initialized: %s/%s", project, run_name)
            except Exception as e:
                module_logger.warning(
                    "wandb init failed (%s), using CSV only", e)
                self.wandb_run = None

        # CSV file
        self.csv_path = self.log_dir / "metrics.csv"
        self._csv_initialized = False

    def log_iteration(self, **kwargs) -> IterationMetrics:
        """
        Log metrics for one training iteration.

        Pass any IterationMetrics field as a keyword argument.
        At minimum, pass `iteration`.
        """
        metrics = IterationMetrics(
            timestamp=time.time(),
            **kwargs,
        )

        # Compute total loss if components provided
        if metrics.loss_policy > 0 or metrics.loss_value > 0:
            metrics.loss_total = metrics.loss_policy + metrics.loss_value

        self.history.append(metrics)

        # Log to wandb
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log(metrics.to_dict(), step=metrics.iteration)
            except Exception as e:
                module_logger.warning("wandb log failed: %s", e)

        # Log to CSV
        self._write_csv(metrics)

        # Log to console
        module_logger.info(
            "Iter %d | loss_p=%.4f loss_v=%.4f | vs_random=%.1f%% | "
            "games=%d avg_len=%.1f | buffer=%d",
            metrics.iteration,
            metrics.loss_policy,
            metrics.loss_value,
            metrics.win_rate_vs_random * 100,
            metrics.total_games_played,
            metrics.avg_game_length,
            metrics.buffer_size,
        )

        return metrics

    def log_custom(self, data: Dict[str, Any], step: Optional[int] = None):
        """Log arbitrary key-value pairs (e.g., learning rate changes)."""
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log(data, step=step)
            except Exception:
                pass

    def finish(self):
        """Flush and close all logging backends."""
        # Save full history as JSON
        json_path = self.log_dir / "metrics_full.json"
        with open(json_path, "w") as f:
            json.dump(
                [m.to_dict() for m in self.history],
                f, indent=2,
            )

        if self.wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        module_logger.info(
            "Training log saved: %d iterations, csv=%s, json=%s",
            len(self.history), self.csv_path, json_path,
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """Return full metrics history as list of dicts."""
        return [m.to_dict() for m in self.history]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_csv(self, metrics: IterationMetrics):
        """Append one row to the CSV file."""
        data = metrics.to_dict()
        mode = "a" if self._csv_initialized else "w"

        with open(self.csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not self._csv_initialized:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(data)


class Timer:
    """Context manager for timing blocks of code."""

    def __init__(self):
        self.elapsed_s: float = 0.0
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_s = time.perf_counter() - self._start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000

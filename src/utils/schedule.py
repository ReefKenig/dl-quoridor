"""Learning-rate schedules for the training loop.

The optimizer was constructed once with a constant lr=1e-3 and no scheduler
anywhere. That is fine for a 15-iteration 5x5 proof-of-concept and a real gap
over a 100-iteration 9x9 run, where a constant step size keeps the network
chasing the most recent self-play data instead of settling.

Schedules here are pure functions of the iteration index rather than stateful
torch schedulers, which matters for this project specifically: runs are resumed
from checkpoints, and a stateful scheduler would need its own serialisation to
survive that. `lr_at(cfg, it)` gives the same answer on a fresh run and on a
run resumed at iteration `it`, with nothing extra to persist.
"""
import math

SCHEDULES = ("constant", "cosine")


def cosine_lr(base_lr: float, final_lr: float, step: int, total_steps: int) -> float:
    """Cosine anneal from `base_lr` at step 0 to `final_lr` at `total_steps`."""
    if total_steps <= 0:
        return base_lr
    t = min(max(step, 0), total_steps) / total_steps
    return final_lr + 0.5 * (base_lr - final_lr) * (1.0 + math.cos(math.pi * t))


def lr_at(schedule: str, base_lr: float, step: int, total_steps: int,
          final_frac: float = 0.1) -> float:
    """Learning rate for iteration `step` under the named schedule.

    final_frac: the fraction of base_lr to end at (0.1 -> decay 1e-3 to 1e-4).
    """
    if schedule not in SCHEDULES:
        raise ValueError(
            f"lr_schedule={schedule!r} is not recognised — expected one of {SCHEDULES}.")
    if schedule == "constant":
        return base_lr
    return cosine_lr(base_lr, base_lr * final_frac, step, total_steps)

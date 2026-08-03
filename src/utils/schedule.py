"""Learning-rate and wall-curriculum schedules for the training loop.

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


def wall_budget_at(it, mask_iters, ramp_hold, max_walls):
    """Walls each player starts self-play with at iteration `it` (0-based).

    0 while masked, then one more wall every `ramp_hold` iterations.
    Note a budget of 1 already exposes the FULL 128-wall action space — the env
    gates walls on `walls_remaining > 0` — so ramping above 1 changes nothing the
    policy can see. Kept for reproducing the earlier probes; prefer
    `wall_mask_fraction`. 0 = full allowance at once.
    """
    if it < mask_iters:
        return 0
    if ramp_hold <= 0:
        return max_walls
    return min(max_walls, 1 + (it - mask_iters) // ramp_hold)


def takes_share(index, fraction):
    """Whether item `index` falls in `fraction` of the stream.

    Spread evenly (Bresenham), so any prefix is already near the target share
    and a short or interrupted iteration stays balanced.
    """
    if fraction <= 0:
        return False
    if fraction >= 1:
        return True
    return int((index + 1) * fraction) > int(index * fraction)


def game_is_masked(game_index, fraction):
    """Whether game `game_index` of an iteration plays wall-free."""
    return takes_share(game_index, fraction)


def opponent_for_game(game_index, greedy_share):
    """Which opponent game `game_index` trains against: 'greedy' or 'self'.

    Anchored games put the model in one seat against a scripted racer, which is
    the distribution the greedy baseline scores and the one pure self-play
    drifts away from — at N=4 it converges on jump-camping, which wins among
    four identical agents and loses to three racers.
    """
    return "greedy" if takes_share(game_index, greedy_share) else "self"

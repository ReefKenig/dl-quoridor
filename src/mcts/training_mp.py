"""
N-player AlphaZero training loop (vector value + maxⁿ).

Self-contained: self-play with the current model's maxⁿ → train on vector targets
→ accept/reject vs the best model (candidate rotated through all seats) → eval vs
random → checkpoint best/latest. Reduces to the standard duel at N=2.
"""
import json
import math
import os
import time
import logging
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from src.env.pathing import CURRENT_SPEC
from src.env.quoridor_env_mp import NUM_MOVE_ACTIONS
from src.mcts.mcts_maxn import MCTSMaxN, mcts_config_for
from src.mcts.self_play_mp import play_one_game
from src.utils.schedule import (ANCHORED_OPPONENTS, iteration_plans, lr_at,
                                wall_budget_at)
from src.mcts.batched_inference_mp import DEFAULT_BATCH_WAIT_MS
from src.mcts.parallel_self_play_mp import generate_parallel_self_play_mp
from src.mcts.vectorized_self_play_mp import generate_vectorized_self_play_mp
from src.mcts.evaluator_mp import (DEFAULT_EVAL_OPENING_PLIES, evaluate_mp,
                                   evaluate_against_random_mp, greedy_agent,
                                   mcts_agent_mp, minimax_agent)
from src.mcts.parallel_eval_mp import (evaluate_parallel_mp,
                                       evaluate_against_greedy_parallel_mp,
                                       evaluate_against_minimax_parallel_mp,
                                       evaluate_against_random_parallel_mp)
from src.mcts.pretrain_data import pretrain_report_path
from src.utils.checkpoint import atomic_model_save
from src.utils.config import read_frozen_config
from src.utils.logger import make_progress_logger

logger = logging.getLogger(__name__)


def _make_progress_logger(log_path):
    """Return a log(*parts) fn that prints to console AND appends to log_path on
    disk. The disk copy keeps recording even if the Jupyter UI disconnects, so
    progress can be tailed from a terminal.

    Accepts one or more strings; multi-line messages (embedded "\\n" or extra
    positional args) are timestamped per line so the on-disk log stays aligned.
    """
    def _log(*parts):
        msg = "\n".join(str(p) for p in parts)
        print(msg, flush=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            for line in msg.splitlines() or [""]:
                f.write(f"{ts} {line}\n")
    return _log


@dataclass
class TrainingConfigMP:
    num_players: int = 4
    num_iterations: int = 20
    games_per_iteration: int = 40
    batch_size: int = 64
    train_steps_per_iter: int = 200
    # Skip training until the buffer holds at least this many samples (0 = off).
    # Guards against overfitting train_steps to a tiny early, mostly-draw buffer.
    warmup_min_samples: int = 0
    mcts_simulations: int = 100
    # 0 => use mcts_simulations; else lower for faster eval
    eval_simulations: int = 0
    replay_buffer_size: int = 50_000
    max_game_moves: int = 300
    eval_games: int = 80
    eval_random_games: int = 24
    # Absolute yardstick that does not saturate the way vs-random does (9x9 N=2
    # hit 100% vs random at iter 5 and stayed there). 0 disables it.
    eval_greedy_games: int = 0
    accept_margin: float = 0.05          # accept if win_rate > fair_share + margin
    # Hold the gate until the learner clears an absolute bar, then seed the
    # champion from it. An untrained 9x9 net is a wall-spammer that never
    # reaches its goal, so gate games cannot finish against it.
    gate_arm_on_greedy: bool = False
    gate_arm_greedy_min: float = 0.0     # arm once win_vs_greedy exceeds this
    # Move cap for the vs-best eval only. Two trained wall-capable models stall
    # each other far past the self-play cap (v8: 32-38 of 40 gate games timed
    # out, so the decided floor was unreachable at any strength). 0 = max_game_moves.
    gate_max_game_moves: int = 0
    # Adjudicate gate games that still hit the cap by shortest-path distance
    # (unique leader wins, tie stays a draw). v9 showed no cap is enough:
    # 3-6 of 40 decided at 320 plies. Gate only; the yardstick evals keep
    # their play-to-the-goal semantics.
    gate_adjudicate: bool = False
    # "constant" keeps existing scripts unchanged; 9x9 opts into cosine.
    lr_schedule: str = "constant"
    lr_final_frac: float = 0.1           # cosine end point, as a fraction of base_lr
    # Batcher accumulation window; 0 restores the old no-wait draining.
    batch_wait_ms: float = DEFAULT_BATCH_WAIT_MS
    # Sampled opening plies in eval; 0 makes same-seat games identical replays.
    eval_opening_plies: int = DEFAULT_EVAL_OPENING_PLIES
    # run eval every N iterations (1 = every iter)
    eval_every: int = 1
    discount: float = 0.97
    # "round" = decay per the mover's own turns; "ply" = per move by
    # anybody. Per-variant: measured better at N=2, worse at N=4.
    discount_unit: str = "round"
    explore_moves: int = 15
    mcts_dirichlet_epsilon: float = 0.25
    # Defaults match MCTSConfig, which is what every run through v7 actually
    # used: neither reached the search before, so the config values were inert.
    mcts_dirichlet_alpha: float = 0.3
    mcts_c_puct: float = 1.41
    # --- self-play engine selector ---
    # "auto" (default) => derive from parallel_self_play (back-compat: parallel if
    # True else sequential). Explicit values override:
    #   "sequential" — one game at a time, no batching.
    #   "parallel"   — K worker processes + shared GPU batcher (leaf-parallel).
    #   "vectorized" — in-process, G games share one predict_batch (Option B;
    #                  exact sequential MCTS per game, no straggler tail).
    self_play_mode: str = "auto"
    # Games run concurrently by the vectorized engine (GPU batch width).
    # 0 => driver default (min(games_per_iteration, 64)).
    vec_games: int = 0
    # --- parallel self-play ---
    parallel_self_play: bool = False
    num_workers: int = 8
    inference_batch_size: int = 64
    # Leaf-parallel MCTS in the spawned self-play/eval workers: leaves collected per
    # GPU forward (>1 breaks the batch<=num_workers ceiling) + virtual loss to
    # diversify the concurrent tree walks. leaf_batch=1 keeps the one-leaf path.
    leaf_batch: int = 1
    virtual_loss: float = 1.0
    # Expand only pawn moves + this many path-cutting walls. 0 = every legal
    # action (4.6 visits/action at the 9x9 opening, i.e. search cannot move away
    # from the prior). Applies to self-play, eval and the UI alike.
    mcts_wall_candidates: int = 0
    # GPU-batched parallel evaluation (candidate/champion served by one batcher).
    # Reuses num_workers / inference_batch_size. Opt-in; sequential eval otherwise.
    parallel_eval: bool = False
    # --- env geometry (needed by parallel workers to rebuild env) ---
    board_size: int = 5
    max_walls_per_player: int = 3
    max_turns: int = 300
    # Wall curriculum for SELF-PLAY only; eval always plays the full game so the
    # gate and the greedy baseline stay comparable across the whole run.
    # wall_mask_iters opening iterations at 0 walls (a pure race), then
    # one more wall every wall_ramp_hold iterations. 0/0 = off.
    # Handing back all the walls at once relapsed the policy in one iteration at
    # both player counts; see academic_experiences.md 8.6.
    wall_mask_iters: int = 0
    wall_ramp_hold: int = 0
    # Fraction of EVERY iteration's games played wall-free, mixed in alongside
    # full-wall games. A masked phase that ends leaves the value head with no
    # coverage of walled states, and root noise drives self-play straight into
    # them; mixing keeps both distributions in the buffer for the whole run.
    # 0 = off (use the phase/ramp above instead).
    wall_mask_fraction: float = 0.0
    # Set per iteration onto cfg.wall_budget, which the workers read.
    wall_budget: int = None
    # Input-plane spec the run trains under; frozen into the run dir's config.json
    # so the resulting checkpoint can be replayed on the planes it actually saw.
    spec_version: int = CURRENT_SPEC
    # Early stop on racing decay. Greedy always has one seat a pure racer wins
    # outright (seat 1 at N=2, seat 0 at N=4 — whoever the head-on pawn jump
    # favours), so the BEST per-seat greedy rate is a hard-ceiling probe: it
    # cannot drift up, and a sustained fall means the policy stopped racing.
    # The gate cannot see this — in local_9x9_v6 it kept accepting (69%, 60%)
    # while greedy went 60% -> 10%. Stop after this many consecutive evals at
    # least greedy_stop_drop below the best rate seen so far. 0 = disabled.
    greedy_stop_patience: int = 0
    greedy_stop_drop: float = 0.20
    # Sigma the drop must also clear, so noise on a 20-game seat cannot strike.
    greedy_stop_z: float = 2.0
    # Companion watch: stop if the best per-seat rate never REACHES this once the
    # masked curriculum is over. Decay cannot catch a run that never climbed.
    # 0 = off; shares greedy_stop_patience.
    greedy_min_seat: float = 0.0
    # Iterations before that floor arms. Defaults to the masked phase, but a
    # mixed curriculum has no phase to end, so it needs its own grace window.
    greedy_min_seat_after: int = 0
    # Stop when this many consecutive greedy evals pass without a new pooled
    # peak. greedy_peak.pt already holds the deliverable, so running on is
    # only justified while a new peak is plausible — v9's N=2 spent ~44
    # iterations (~20 h) after its last improvement. 0 = off.
    peak_stall_evals: int = 0
    # Held-out baseline. greedy becomes a training opponent once the pool
    # anchors on it, so it stops measuring generalisation; minimax never trains
    # and it places walls. 0 games = off.
    eval_minimax_games: int = 0
    minimax_depth: int = 2
    minimax_wall_candidates: int = 16
    # Self-play opponent pool. Shares of each iteration's games; the remainder
    # after past/greedy is played against the current model, i.e. plain
    # self-play. A single fixed opponent overfits and pure self-play at N=4
    # converges on a jump-camping equilibrium the evaluation never rewards.
    opponent_past_share: float = 0.0
    opponent_greedy_share: float = 0.0
    # Target share of each TRAINING BATCH drawn from anchored games. The shares
    # above govern how many games are produced; this governs how much gradient
    # they get, and the two diverge by ~9x at N=2 and ~20x at N=4. 0 = uniform,
    # which let 35% of games become ~10% of the gradient — enough to acquire
    # racing at iteration 8 and not enough to still have it at iteration 12.
    anchored_sample_share: float = 0.0
    # Fraction of anchored games pinned to seat 0 (rest rotate over 1..N-1).
    # At N=4 seat 0 is the only seat a racer can win and its games are the
    # shortest, so uniform rotation starves the one seat that matters.
    anchored_seat0_share: float = 0.0
    # Warm-start weights for a FRESH run (ignored on resume). Lets supervised
    # pretraining set the opening prior instead of hoping self-play finds it.
    init_checkpoint: str = ""
    # Cross-entropy pull toward the init_checkpoint policy on every training
    # batch. n4_9x9_v9 measured why the data-mix lever is not enough: seat-0
    # anchored games held their configured 40 games/~580 samples every
    # iteration and the racer still eroded to 0/80 by iteration 28
    # (first_wall_ply 3.4 -> 0.7) — 65%-walled self-play gradient outweighs
    # any realistic sample share, so the prior must be defended in the loss.
    # 0 = off.
    anchor_weight: float = 0.0
    # Champion snapshots kept for the past-opponent share.
    champion_pool_size: int = 5
    # Weight on the SEAT-0 value target of clone-vs-clone samples; 0.0 excludes
    # them. Clone games teach the head that seat-0 racing loses, which is the
    # erosion v10 measured surviving the policy anchor. Anchored games keep
    # every column. 1.0 = the plain step.
    clone_seat0_value_weight: float = 1.0

    def __post_init__(self):
        # The smaller of the two limits ends the game, so an env cutoff below
        # the driver's cap shortens every game silently.
        if self.max_turns < self.max_game_moves:
            logger.warning(
                "max_turns=%d is below max_game_moves=%d, which would end every "
                "game %d plies early; raising it to match.",
                self.max_turns, self.max_game_moves,
                self.max_game_moves - self.max_turns)
            self.max_turns = self.max_game_moves
        if self.anchor_weight and not self.init_checkpoint:
            raise ValueError(
                "anchor_weight requires init_checkpoint — the warm-start "
                "policy is the anchor.")
        if self.opponent_past_share and not self.parallel_self_play:
            raise NotImplementedError(
                "opponent_past_share needs the parallel engine — the champion "
                "is served as model_id 1 on the shared inference batcher.")
        if not 0.0 <= self.clone_seat0_value_weight <= 1.0:
            raise ValueError(
                f"clone_seat0_value_weight is a weight on an existing target "
                f"({self.clone_seat0_value_weight}); 0.0 drops it, 1.0 keeps it.")
        total = self.opponent_past_share + self.opponent_greedy_share
        if total > 1.0:
            raise ValueError(
                f"opponent shares sum to {total:.2f}; they are fractions of "
                f"each iteration's games and cannot exceed 1.0.")


BUFFER_FILE = "replay_buffer.npz"


SELF_SOURCE = "self"

# Fixed width so the persisted array and the in-memory cache agree; source names
# are short labels ("self", "greedy", "past"), not free text.
SOURCE_DTYPE = "U16"


def clone_seat0_value_weights(sources, num_players, weight):
    """Per-sample, per-seat weights for the value loss; None when it is uniform.

    Only the seat-0 column of clone-vs-clone samples moves. `_seat_perm` keeps
    seat 0 fixed under the mirror, so column 0 is seat 0 for augmented samples too.
    """
    if weight == 1.0:
        return None
    w = np.ones((len(sources), num_players), dtype=np.float32)
    w[np.asarray(sources, dtype=SOURCE_DTYPE) == SELF_SOURCE, 0] = weight
    return w


class ReplayBufferMP:
    """Replay buffer that can draw a batch at a TARGET source mix.

    Uniform sampling makes the gradient share a consequence of how many samples
    each opponent happened to produce, which is not what the config asks for: an
    anchored game yields only the model's own plies and ends sooner, so 35% of
    games came out as ~10% of the gradient and the anchored signal was too weak
    to hold. Sampling to a target share decouples production from consumption.

    NOT thread-safe, and deliberately so: `add`, `sample_batch`, `save` and
    `load` all assume a single caller. The buffer lives in the main training
    process only — self-play workers are separate processes that ship samples
    back over a queue, and `training_loop_mp` adds and samples sequentially. The
    two deques and the source cache are mutated without a lock on that basis, so
    concurrent access would need locking added here first.
    """

    def __init__(self, max_size=50_000):
        self.buffer = deque(maxlen=max_size)
        # Parallel deque: same maxlen, appended and evicted in lockstep, so a
        # source always describes the sample at the same index.
        self.sources = deque(maxlen=max_size)
        # Vectorized view of `sources`, rebuilt lazily. Sampling runs
        # train_steps_per_iter times against a buffer that only changes once per
        # iteration, so the O(N) build is paid once rather than per batch.
        self._sources_cache = None

    def add(self, samples, sources=None):
        samples = list(samples)
        self.buffer.extend(samples)
        if sources is None:
            self.sources.extend([SELF_SOURCE] * len(samples))
        else:
            sources = list(sources)
            if len(sources) != len(samples):
                raise ValueError(
                    f"{len(sources)} sources for {len(samples)} samples — the "
                    f"two deques must stay index-aligned.")
            self.sources.extend(sources)
        self._sources_cache = None      # eviction can shift every index

    def sources_array(self):
        """`sources` as a numpy array, cached until the next `add`."""
        if self._sources_cache is None:
            self._sources_cache = np.array(list(self.sources), dtype=SOURCE_DTYPE)
        return self._sources_cache

    def indices_by_source(self, source):
        return np.flatnonzero(self.sources_array() == source)

    def sample_batch(self, batch_size, source=None, source_share=0.0,
                     with_sources=False):
        """A batch, optionally drawing `source_share` of it from `source`.

        Falls back to whatever is available rather than failing: early
        iterations may hold fewer anchored samples than the target asks for.

        with_sources also returns the drawn samples' source labels, so a caller
        can weight the loss by where a sample came from.
        """
        n = min(batch_size, len(self.buffer))
        if source and source_share > 0:
            wanted = int(round(n * min(source_share, 1.0)))
            is_source = self.sources_array() == source
            pool = np.flatnonzero(is_source)
            rest = np.flatnonzero(~is_source)
            take = min(wanted, len(pool))
            chosen = (np.random.choice(pool, take, replace=False) if take
                      else np.empty(0, dtype=int))
            need = n - len(chosen)
            if need > 0:
                if len(rest) >= need:
                    extra = np.random.choice(rest, need, replace=False)
                else:                       # not enough of the other source yet
                    spare_mask = np.ones(len(self.buffer), dtype=bool)
                    spare_mask[chosen] = False
                    spare = np.flatnonzero(spare_mask)
                    extra = np.random.choice(
                        spare, min(need, len(spare)), replace=False)
                chosen = np.concatenate([chosen, extra])
            idx = chosen.astype(int)
        else:
            idx = np.random.choice(len(self.buffer), n, replace=False)
        # The source-mixed path appends its two draws in blocks, so without this
        # every batch would be ordered source-first.
        np.random.shuffle(idx)
        b = [self.buffer[i] for i in idx]
        S = np.array([x[0] for x in b], np.float32)
        P = np.array([x[1] for x in b], np.float32)
        V = np.array([x[2] for x in b], np.float32)
        if with_sources:
            return S, P, V, self.sources_array()[idx]
        return S, P, V

    def __len__(self):
        return len(self.buffer)

    def save(self, checkpoint_dir):
        """Persist the buffer so a resume keeps its samples. tmp+rename so a
        kill mid-write leaves the previous buffer rather than a truncated one."""
        path = os.path.join(checkpoint_dir, BUFFER_FILE)
        tmp = path + ".tmp"
        S = np.array([x[0] for x in self.buffer], np.float32)
        P = np.array([x[1] for x in self.buffer], np.float32)
        V = np.array([x[2] for x in self.buffer], np.float32)
        src = self.sources_array()
        with open(tmp, "wb") as f:
            np.savez(f, S=S, P=P, V=V, src=src)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path

    def load(self, checkpoint_dir, log=print):
        """Samples restored from disk. Missing or unreadable leaves it empty,
        which is the old resume behaviour rather than a failed run."""
        path = os.path.join(checkpoint_dir, BUFFER_FILE)
        if not os.path.exists(path):
            return 0
        try:
            with np.load(path) as data:
                S, P, V = data["S"], data["P"], data["V"]
                # Buffers written before source tagging have no "src"; treating
                # them as self-play is what they were.
                src = data["src"] if "src" in data else None
        except (OSError, ValueError, KeyError) as exc:
            log(f"WARNING: could not read {path} ({exc}) — starting with an "
                f"empty buffer, as if it had not been persisted.")
            return 0
        # A truncated src would silently shift every label onto the wrong sample
        # and quietly mis-weight the source mix for the rest of the run.
        if src is not None and len(src) != len(S):
            log(f"WARNING: {path} holds {len(src)} source labels for {len(S)} "
                f"samples — dropping the labels and treating the buffer as "
                f"self-play rather than mislabelling it.")
            src = None
        # .copy() because iterating a stacked array yields views: one surviving
        # view keeps the entire ~500 MB base alive. maxlen trims the oldest if
        # the run resumes at a smaller buffer size.
        self.add([(s.copy(), p.copy(), v.copy()) for s, p, v in zip(S, P, V)],
                 sources=(list(src) if src is not None else None))
        return len(self.buffer)


def _mcts(model, env, cfg, sims=None, dirichlet_epsilon=None):
    # dirichlet_epsilon override: pass 0.0 for eval (deterministic best-play, no
    # exploration noise); leave None for self-play to use the cfg default.
    eps = (dirichlet_epsilon if dirichlet_epsilon is not None
           else getattr(cfg, 'mcts_dirichlet_epsilon', 0.25))
    return MCTSMaxN(
        config=mcts_config_for(cfg, num_simulations=sims, dirichlet_epsilon=eps,
                               max_rollout_depth=cfg.max_game_moves),
        evaluate_fn=lambda st: model.predict(env.state_to_tensor(st)),
        num_players=cfg.num_players,
    )


def drop_is_significant(peak, now, n_per_seat, z_min):
    """Two-proportion z: is the fall bigger than n-game sampling noise?

    The peak is a max over seats AND over evals, so it overshoots the true rate;
    without this the threshold lands near the mean and a stable model strikes.
    """
    if not n_per_seat:
        return True
    pooled = (peak + now) / 2
    se = math.sqrt(2 * pooled * (1 - pooled) / n_per_seat)
    return se == 0 or (peak - now) >= z_min * se


def sample_diagnostics(samples, num_players, model=None, max_states=512):
    """What the iteration actually trained on, and how well the value head knows it.

    Channels N and N+1 are the wall planes (see build_tensor_mp), so a position
    is "walled" iff either is non-zero. Splitting value error that way measures
    the coverage gap directly: a masked phase leaves the head with no walled
    states, and exploration then drives self-play straight into them.
    """
    if not samples:
        return {}
    wall_ch = slice(num_players, num_players + 2)
    walled, wall_counts, policy_wall = [], [], []
    for tensor, policy, _v in samples:
        planes = tensor[:, :, wall_ch]
        n_walls = int(np.count_nonzero(planes))
        wall_counts.append(n_walls)
        walled.append(n_walls > 0)
        policy_wall.append(float(policy[NUM_MOVE_ACTIONS:].sum()))
    out = {
        # The 0.00024 -> 0.245 quantity from the curriculum analysis, but on the
        # POLICY TARGET rather than the prior: this is what training consumes.
        "policy_wall_mass": float(np.mean(policy_wall)),
        "walled_state_share": float(np.mean(walled)),
        # Non-zero wall-plane CELLS, not walls: a wall spans ~2 cells, so halve
        # it to read walls. Sample-weighted, so long timeout games dominate.
        "walls_on_board_mean": float(np.mean(wall_counts)),
    }
    if model is None:
        return out
    idx = np.random.choice(len(samples), min(max_states, len(samples)),
                           replace=False)
    err_walled, err_free = [], []
    for i in idx:
        tensor, _p, target = samples[int(i)]
        _policy, value = model.predict(tensor)
        err = float(np.mean(np.abs(np.asarray(value) - np.asarray(target))))
        (err_walled if walled[int(i)] else err_free).append(err)
    if err_walled:
        out["value_mae_walled"] = float(np.mean(err_walled))
    if err_free:
        out["value_mae_wallfree"] = float(np.mean(err_free))
    return out


def evals_since_last_peak(history):
    """Greedy evals since the last new pooled peak, from the writer's
    greedy_peak_saved stamps — so a resume keeps the staleness clock."""
    since = 0
    for r in history:
        if r.get("win_vs_greedy") is None:
            continue
        since = 0 if r.get("greedy_peak_saved") else since + 1
    return since


def stalled_below_floor(best_seat, floor, below):
    """Advance the never-acquired watch. Returns consecutive evals under floor.

    racing_decay_strike can only fire on a fall from a peak, so a run that never
    climbs is invisible to it: probe_n4_ramp peaked at 0.10 against a 0.20 drop,
    making a strike arithmetically impossible, and it ran 10 hours past the point
    the answer was known.
    """
    return below + 1 if best_seat < floor else 0


def racing_decay_strike(best_seat, peak, below, drop, n_per_seat=0, z_min=0.0):
    """Advance the racing-decay watch by one greedy eval.

    Compares against the best rate ever seen, not the previous eval, so a slow
    slide cannot hide. A strike needs the drop to clear `drop` AND, when
    n_per_seat/z_min are given, to be significant at that many sigma.
    Returns (peak, consecutive_strikes).
    """
    if peak is None or best_seat > peak:
        return best_seat, 0
    if best_seat > peak - drop:
        return peak, 0
    if not drop_is_significant(peak, best_seat, n_per_seat, z_min):
        return peak, 0
    return peak, below + 1


SELF_PLAY_MODES = ("sequential", "parallel", "vectorized")


def resolve_self_play_mode(cfg):
    """Resolve cfg.self_play_mode to a concrete engine name, raising on typos
    rather than silently falling through to the sequential path."""
    mode = getattr(cfg, "self_play_mode", "auto")
    if mode == "auto":
        return "parallel" if cfg.parallel_self_play else "sequential"
    if mode not in SELF_PLAY_MODES:
        raise ValueError(
            f"self_play_mode={mode!r} is not recognised — expected 'auto' or one "
            f"of {SELF_PLAY_MODES}.")
    return mode


def zero_sample_reason(it, engine, wins, games_per_iteration, checkpoint_dir):
    """Diagnose a zero-sample iteration: all-timeout vs a stalled batcher."""
    if wins.get(None, 0) >= games_per_iteration:
        return (
            f"[iter {it}] all {games_per_iteration} games timed out at "
            f"max_game_moves, so self-play produced no training samples. This is "
            f"a game-length problem, not a crash: raise max_game_moves for this "
            f"variant. It counts plies rather than rounds, so N=4 needs roughly "
            f"twice the N=2 value to give each player the same budget.")
    return (
        f"[iter {it}] {engine} self-play produced 0 samples — aborting before "
        f"empty training. Check {os.path.join(checkpoint_dir, 'games.log')} for "
        f"'[GPU INFERENCE] CRASHED' or '[WORKER … ] CRASHED'.")


def _rss_gb():
    """Resident memory of this process, in GB (0.0 if psutil is unavailable)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return 0.0


def _host_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return 0.0


def _read_first(*paths):
    for path in paths:
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            continue
    return None


def _cgroup_mem_limit_gb():
    """Container memory ceiling in GB, or None when unlimited. psutil reports the
    host's RAM, which on a shared or MIG-partitioned box is not what we may use."""
    raw = _read_first("/sys/fs/cgroup/memory.max",                  # cgroup v2
                      "/sys/fs/cgroup/memory/memory.limit_in_bytes")  # cgroup v1
    if raw is None or raw == "max":
        return None
    try:
        limit = int(raw)
    except ValueError:
        return None
    # v1 reports a sentinel near 2**63 when unlimited.
    return None if limit >= 2**62 else limit / 1e9


def _cpu_budget():
    """(usable_cpus, source): cgroup quota -> affinity -> host cores."""
    raw = _read_first("/sys/fs/cgroup/cpu.max")                     # cgroup v2
    if raw and not raw.startswith("max"):
        try:
            quota, period = raw.split()
            return int(quota) / int(period), "cgroup quota"
        except (ValueError, ZeroDivisionError):
            pass
    quota = _read_first("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")      # cgroup v1
    period = _read_first("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and int(quota) > 0:
        return int(quota) / int(period), "cgroup quota"
    try:
        return float(len(os.sched_getaffinity(0))), "affinity"
    except AttributeError:
        return float(os.cpu_count() or 0), "host cores"


def _gpu_desc():
    """Device name and memory as this process sees it — a MIG slice, not the card."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "no CUDA"
        props = torch.cuda.get_device_properties(0)
        return f"{props.name}, {props.total_memory / 1e9:.0f} GB"
    except Exception:
        return "unknown"


def resource_banner(cfg):
    """One line describing what this process may actually use, plus a warning
    when num_workers oversubscribes the CPU budget."""
    cpus, source = _cpu_budget()
    mem_limit = _cgroup_mem_limit_gb()
    mem = (f"{mem_limit:.0f} GB limit (host {_host_ram_gb():.0f} GB)"
           if mem_limit else f"{_host_ram_gb():.0f} GB, no cgroup limit")
    lines = [f"resources: {cpus:.0f} usable cpus ({source}), {mem}, "
             f"gpu: {_gpu_desc()}, rss={_rss_gb():.2f} GB at launch"]
    if cpus and cfg.num_workers > cpus:
        lines.append(
            f"WARNING: num_workers={cfg.num_workers} exceeds the {cpus:.0f} usable "
            f"cpus — expect contention, and slowdowns that end in a killed kernel.")
    return lines


@contextmanager
def _wall_budget(cfg, env, budget):
    """Apply the curriculum's wall budget to self-play only.

    Restores on the way out even if self-play raises — eval, gating and the
    greedy baseline must always play the full game, and the caller may keep
    using this env after a failed iteration.
    """
    cfg.wall_budget = env.wall_budget = budget
    try:
        yield
    finally:
        cfg.wall_budget = env.wall_budget = None


def assert_resume_spec_matches(cfg, checkpoint_dir):
    """Refuse to resume a run onto input planes it was not trained on. Run dirs
    predating the versioning have no spec_version key — those are all v1."""
    frozen = read_frozen_config(checkpoint_dir)
    if frozen is None:
        return
    frozen_spec = frozen["spec_version"]
    if frozen_spec != cfg.spec_version:
        raise ValueError(
            f"{checkpoint_dir} was trained under tensor spec v{frozen_spec} but "
            f"this launch is configured for v{cfg.spec_version}. Resuming would "
            f"feed the checkpoint planes on a scale it never saw. Start a fresh "
            f"run dir, or set spec_version={frozen_spec} to continue the old one.")


def freeze_config(cfg, checkpoint_dir, log=print):
    """Write the resolved config next to the checkpoints, so a run dir records
    what it actually ran rather than relying on the shared config file.

    First launch writes it; later launches only compare, so the record stays
    the run's rather than the last relaunch's."""
    path = os.path.join(checkpoint_dir, "config.json")
    resolved = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    frozen = read_frozen_config(checkpoint_dir)
    if frozen is not None:
        changed = {k: (frozen[k], v) for k, v in resolved.items()
                   if k in frozen and frozen[k] != v}
        added = sorted(set(resolved) - set(frozen))
        if changed or added:
            log(f"WARNING: config differs from {path}, which records what the "
                f"earlier iterations ran. Keeping the frozen copy; this launch "
                f"uses:")
            for k, (was, now) in sorted(changed.items()):
                log(f"  {k}: frozen={was!r} -> now={now!r}")
            if added:
                log(f"  new keys: {', '.join(added)}")
        else:
            log(f"Config matches the frozen copy -> {path}")
        return path
    with open(path, "w") as f:
        json.dump(resolved, f, indent=2, default=str, sort_keys=True)
    log(f"Config frozen -> {path}")
    return path


CHAMPION_DIR = "champions"


def snapshot_champion(best, checkpoint_dir, iteration, pool_size):
    """Save this champion and drop the oldest beyond `pool_size`. Returns its path."""
    pool_dir = os.path.join(checkpoint_dir, CHAMPION_DIR)
    os.makedirs(pool_dir, exist_ok=True)
    path = os.path.join(pool_dir, f"champion_iter{iteration:04d}.pt")
    best.save(path)
    kept = sorted(f for f in os.listdir(pool_dir) if f.endswith(".pt"))
    for stale in kept[:max(0, len(kept) - pool_size)]:
        os.remove(os.path.join(pool_dir, stale))
    return path


def champion_pool_paths(checkpoint_dir):
    """Snapshot paths currently on disk, oldest first. Survives a resume."""
    pool_dir = os.path.join(checkpoint_dir, CHAMPION_DIR)
    if not os.path.isdir(pool_dir):
        return []
    return [os.path.join(pool_dir, f)
            for f in sorted(os.listdir(pool_dir)) if f.endswith(".pt")]


def load_past_champion(past_model, path, env):
    """Load a pooled champion and prove it can serve a forward pass.

    The past opponent rides the shared inference batcher as a second model. A
    checkpoint that loads but cannot predict (wrong player count, stale tensor
    spec) would otherwise surface an hour later, mid-iteration.
    """
    try:
        past_model.load(path)
        past_model.predict(env.state_to_tensor(env.reset()))
    except Exception as exc:
        raise RuntimeError(
            f"past-opponent champion {path} failed to load or serve a warmup "
            f"forward pass — it cannot be used on the inference batcher") from exc
    return past_model


def _iteration_plans(cfg):
    """The (opponent, seat, wall-mask) plan the self-play workers will follow."""
    return iteration_plans(
        cfg.games_per_iteration, cfg.num_players,
        getattr(cfg, "opponent_greedy_share", 0.0) or 0.0,
        getattr(cfg, "opponent_past_share", 0.0) or 0.0,
        getattr(cfg, "wall_mask_fraction", 0.0) or 0.0,
        getattr(cfg, "anchored_seat0_share", 0.0) or 0.0)


def anchored_walled_share_by_seat(cfg):
    """Fraction of each seat's anchored games that are played WITH walls legal.

    In meta.json because the aliasing it replaces was invisible in every metric
    the run recorded. A seat pinned at 0.0/1.0, or missing, is the signature.
    """
    walled, total = {}, {}
    for plan in _iteration_plans(cfg):
        if plan.opponent not in ANCHORED_OPPONENTS:
            continue
        seat = plan.model_seat
        total[seat] = total.get(seat, 0) + 1
        if not plan.walls_masked:
            walled[seat] = walled.get(seat, 0) + 1
    return {str(seat): round(walled.get(seat, 0) / n, 4)
            for seat, n in sorted(total.items())}


def _load_warm_start(target, path):
    """Load init_checkpoint weights with errors that name the usual causes.
    Shared by the learner (fresh start) and the policy anchor (every start)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"init_checkpoint={path} does not exist. Build it with "
            f"scripts/pretrain_greedy.py (the notebooks have a cell that does "
            f"this when it is missing).")
    try:
        target.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"init_checkpoint {path} failed to load — usually a channels/"
            f"blocks/players mismatch with this run's network config.") from exc


def init_champion(best, model, checkpoint_dir, log=print):
    """Establish the gating champion and make it durable from iteration 0.

    Written up front so a run that never accepts still has a real opponent.
    """
    best.copy_weights_from(model)
    best_path = os.path.join(checkpoint_dir, "best.pt")
    if not os.path.exists(best_path):
        best.save(best_path)
        log(f"Champion initialised from the starting model -> {best_path}")
    return best_path


def load_champion(best, model, checkpoint_dir, log=print):
    """Restore the gating champion on resume. Returns True if loaded from disk.

    A missing best.pt warns loudly: falling back to the learner makes the gate
    compare the model against itself and measure nothing.
    """
    best_path = os.path.join(checkpoint_dir, "best.pt")
    if os.path.exists(best_path):
        best.load(best_path)
        return True
    best.copy_weights_from(model)
    best.save(best_path)
    log(f"WARNING: {best_path} is missing on resume — the gating champion has "
        f"been re-seeded from the current model and saved. Until a candidate is "
        f"accepted, eval-vs-best compares the model against an identical copy of "
        f"itself and cannot exceed the accept threshold.")
    return False


def training_loop_mp(env, model, make_model, cfg: TrainingConfigMP,
                     checkpoint_dir="checkpoints_mp"):
    """
    env   : QuoridorEnvMP (num_players == cfg.num_players)
    model : QuoridorModelMP (the learner)
    make_model : zero-arg callable returning a fresh QuoridorModelMP (for `best`)
    """
    assert env.num_players == cfg.num_players
    # The workers rebuild the env from cfg, so a disagreement here would have the
    # parent and the workers training one model on two different plane scales.
    if env.spec_version != cfg.spec_version:
        raise ValueError(
            f"env tensor spec v{env.spec_version} != cfg spec v{cfg.spec_version}; "
            "resuming a run started under an older spec needs the env built with "
            "that run's config.json spec_version")
    os.makedirs(checkpoint_dir, exist_ok=True)
    buffer = ReplayBufferMP(cfg.replay_buffer_size)
    best = make_model()
    fair = 1.0 / cfg.num_players
    threshold = fair + cfg.accept_margin
    history = []

    # Disk log: keeps recording progress even if the Jupyter UI disconnects.
    _log = make_progress_logger(os.path.join(checkpoint_dir, "games.log"))

    # Resolved once, up front: validates the config before any work is done, and
    # the banner then reports exactly what the loop will run.
    sp_mode = resolve_self_play_mode(cfg)
    _log(
        "=" * 70,
        f"training_loop_mp launched | N={cfg.num_players} board={cfg.board_size}x{cfg.board_size} "
        f"| sims={cfg.mcts_simulations} games/iter={cfg.games_per_iteration} "
        f"| self_play={sp_mode} workers={cfg.num_workers} vec_games={cfg.vec_games}",
        f"checkpoint_dir={checkpoint_dir} | eval={cfg.eval_games}+{cfg.eval_random_games}"
        f"{'+' + str(cfg.eval_greedy_games) + ' greedy' if cfg.eval_greedy_games else ' (no greedy)'} "
        f"| accept_margin={cfg.accept_margin} | buffer={cfg.replay_buffer_size}",
        f"train_steps={cfg.train_steps_per_iter} max_moves={cfg.max_game_moves} "
        f"explore_moves={cfg.explore_moves} warmup={cfg.warmup_min_samples} "
        f"discount={cfg.discount}/{cfg.discount_unit} "
        f"leaf_batch={cfg.leaf_batch} vloss={cfg.virtual_loss}"
        + (f" | clone seat-0 value targets weighted "
           f"{cfg.clone_seat0_value_weight}"
           if cfg.clone_seat0_value_weight != 1.0 else ""),
        *resource_banner(cfg),
        "=" * 70,
    )
    freeze_config(cfg, checkpoint_dir, log=_log)

    # --- Resume from checkpoint if available ---
    start_iter = 0
    meta_path = os.path.join(checkpoint_dir, "meta.json")
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(meta_path) and os.path.exists(latest_path):
        assert_resume_spec_matches(cfg, checkpoint_dir)
        with open(meta_path) as f:
            meta = json.load(f)
        start_iter = meta.get("completed_iterations", 0)
        model.load(latest_path)
        load_champion(best, model, checkpoint_dir, log=_log)
        history = meta.get("history", [])
        n_buffered = buffer.load(checkpoint_dir, log=_log)
        _log(
            f"Resumed from iteration {start_iter} (checkpoint: {checkpoint_dir})",
            f"Replay buffer restored: {n_buffered} samples" if n_buffered else
            "Replay buffer not on disk — refilling from scratch")
    else:
        if cfg.init_checkpoint:
            # Before init_champion, so the iteration-0 gate opponent is the
            # pretrained racer rather than a wall-spamming random init.
            _load_warm_start(model, cfg.init_checkpoint)
            _log(f"Learner warm-started from {cfg.init_checkpoint}")
            report_path = pretrain_report_path(cfg.init_checkpoint)
            if os.path.exists(report_path):
                with open(report_path) as f:
                    rep = json.load(f)
                _log(f"Warm-start report: agreement={rep.get('agreement')} "
                     f"opening_wall_mass={rep.get('opening_wall_mass')} "
                     f"({rep.get('games')} games, {rep.get('samples')} samples)")
        init_champion(best, model, checkpoint_dir, log=_log)
        _log(f"Starting N={cfg.num_players} training: {cfg.num_iterations} iterations, "
             f"{cfg.games_per_iteration} games/iter, {cfg.mcts_simulations} sims")

    # Champion snapshots for the past-opponent share; repopulated from disk so a
    # resumed run keeps the pool it built rather than starting from one self.
    champion_pool = champion_pool_paths(checkpoint_dir) if cfg.opponent_past_share else []
    past_model = make_model() if cfg.opponent_past_share else None

    # Frozen policy anchor (see cfg.anchor_weight). Built OUTSIDE the fresh-start
    # branch: a resumed run must keep pulling toward the same prior.
    anchor = None
    if cfg.anchor_weight:
        anchor = make_model()
        _load_warm_start(anchor, cfg.init_checkpoint)
        _log(f"Policy anchored to {cfg.init_checkpoint} "
             f"(anchor_weight={cfg.anchor_weight})")

    # Racing-decay watch (see cfg.greedy_stop_patience). The watermark is
    # recovered from history like every other resumed state: v9's N=2 restart
    # at iteration ~40 silently reset an in-memory peak of 100% to 97.5%, and
    # the weakened reference let the run coast past its true stop.
    seat_rates = [w / n for r in history
                  for w, n in (r.get("greedy_by_seat") or {}).values() if n]
    best_seat_peak = max(seat_rates, default=None)
    greedy_below_peak = 0
    greedy_below_floor = 0
    stop_reason = None

    greedy_rates = [(r.get("win_vs_greedy") or 0.0) for r in history]
    # Peak staleness (see cfg.peak_stall_evals), resumed like everything else.
    evals_since_peak = evals_since_last_peak(history)
    # Strongest POOLED greedy eval so far, ratcheted to greedy_peak.pt — distinct
    # from best_seat_peak above, the per-seat watermark the decay watch tracks.
    # The gate can go a whole run without accepting while latest.pt declines
    # past the peak, so neither checkpoint holds the strongest model without it.
    greedy_peak_rate = max(greedy_rates, default=0.0)

    # Gate liveness (see cfg.gate_arm_on_greedy). Recovered from history rather
    # than persisted, so a resume cannot silently re-disarm an armed gate.
    gate_armed = not cfg.gate_arm_on_greedy or any(
        rate > cfg.gate_arm_greedy_min for rate in greedy_rates)
    thin_evals = 0
    warned_unreachable = False
    if cfg.gate_arm_on_greedy and not gate_armed:
        _log(f"Gate held: eval-vs-best is skipped until win_vs_greedy > "
             f"{cfg.gate_arm_greedy_min:.2f}, then the champion is seeded from "
             f"the learner. An untrained net cannot reach its goal, so gate "
             f"games against it time out rather than decide.")

    for it in range(start_iter, cfg.num_iterations):
        t0 = time.time()
        # --- 1. self-play ---
        budget = wall_budget_at(it, cfg.wall_mask_iters, cfg.wall_ramp_hold,
                                cfg.max_walls_per_player)
        with _wall_budget(cfg, env, budget):
            frac = getattr(cfg, "wall_mask_fraction", 0.0) or 0.0
            curriculum = ("" if budget == cfg.max_walls_per_player else
                          f", WALLS MASKED" if budget == 0 else
                          f", {budget}/{cfg.max_walls_per_player} WALLS")
            if frac:
                # The same plan the workers follow, so the log cannot describe a
                # different mix than the one that is played.
                n_masked = sum(p.walls_masked for p in _iteration_plans(cfg))
                curriculum = (f", MIXED {n_masked}/{cfg.games_per_iteration} "
                              f"race-only")
            _log(f"[iter {it+1}/{cfg.num_iterations}] self-play starting "
                 f"({cfg.games_per_iteration} games, {cfg.mcts_simulations} sims"
                 f"{curriculum})...")

            def _on_progress(done, total, w):
                if done % 5 == 0 or done == total:
                    _log(
                        f"[iter {it+1}/{cfg.num_iterations}] self-play: {done}/{total} games...")

            # Only the parallel engine samples an opponent pool; the others
            # report an empty mix rather than a wrong one.
            sp_stats = {}
            # One champion per iteration rather than per game: a uniform draw
            # over the pool still spreads across earlier selves over a run, and
            # keeps a single extra model on the batcher.
            past_for_iter = pick = None
            if cfg.opponent_past_share and champion_pool:
                pick = champion_pool[np.random.randint(len(champion_pool))]
                past_for_iter = load_past_champion(past_model, pick, env)
                _log(f"[iter {it+1}/{cfg.num_iterations}] past opponent: "
                     f"{os.path.basename(pick)} (pool of {len(champion_pool)})")
            if sp_mode == "vectorized":
                # In-process vectorized self-play (Option B): G games share one
                # predict_batch; exact sequential MCTS per game.
                sp_samples, wins = generate_vectorized_self_play_mp(
                    model, cfg,
                    total_games=cfg.games_per_iteration,
                    vec_games=(cfg.vec_games or None),
                    batch_size=cfg.inference_batch_size,
                    on_games_complete=_on_progress,
                    base_seed=it * cfg.games_per_iteration,
                    log=_log,
                )
                if not sp_samples:
                    raise RuntimeError(zero_sample_reason(
                        it + 1, "vectorized", wins, cfg.games_per_iteration,
                        checkpoint_dir))
                buffer.add(sp_samples)
                n_new_samples = len(sp_samples)
            elif sp_mode == "parallel":
                # GPU-batched parallel self-play
                sp_samples, wins, sp_stats = generate_parallel_self_play_mp(
                    model, cfg,
                    num_workers=cfg.num_workers,
                    total_games=cfg.games_per_iteration,
                    batch_size=cfg.inference_batch_size,
                    on_games_complete=_on_progress,
                    # Seeds are per-game (base_seed + game_index, index in
                    # [0, games_per_iteration)); stride by games_per_iteration so
                    # iterations never reuse each other's seeds.
                    base_seed=it * cfg.games_per_iteration,
                    log=_log,
                    past_model=past_for_iter,
                )
                if not sp_samples:
                    # Either self-play stalled (usually: the GPU inference thread died
                    # and workers hung until the queue timeout) or every game timed
                    # out. Abort loudly instead of "training" on an empty buffer and
                    # silently advancing completed_iterations — the run can then be
                    # resumed from the last good checkpoint.
                    raise RuntimeError(zero_sample_reason(
                        it + 1, "parallel", wins, cfg.games_per_iteration,
                        checkpoint_dir))
                buffer.add(sp_samples, sources=sp_stats.get("sources"))
                n_new_samples = len(sp_samples)
            else:
                # Sequential self-play (original path)
                sp_mcts = _mcts(model, env, cfg)
                wins = {}
                n_new_samples = 0
                for g in range(cfg.games_per_iteration):
                    samples, w = play_one_game(env, sp_mcts, cfg.num_players,
                                               max_moves=cfg.max_game_moves,
                                               discount=cfg.discount,
                                               explore_moves=cfg.explore_moves)
                    buffer.add(samples)
                    n_new_samples += len(samples)
                    wins[w] = wins.get(w, 0) + 1
                    # Log every 5 games so long iterations don't look stuck.
                    if (g + 1) % 5 == 0 or (g + 1) == cfg.games_per_iteration:
                        elapsed = time.time() - t0
                        rate = (g + 1) / elapsed if elapsed > 0 else 0
                        _log(f"[iter {it+1}/{cfg.num_iterations}] self-play: "
                             f"{g+1}/{cfg.games_per_iteration} games "
                             f"({elapsed:.0f}s, {rate*60:.1f} games/min)")

            sp_secs = time.time() - t0
        # Win distribution across seats (None = draw/timeout). A healthy self-play
        # iteration is roughly balanced; a lopsided split or all-draws is an early
        # warning of seat bias or a degenerate policy.
        win_dist = ", ".join(
            f"{'draw' if w is None else f'P{w}'}={wins[w]}"
            for w in sorted(wins, key=lambda k: (k is None, k)))
        # Quoridor has no true draws — a None winner is a timeout at max_game_moves,
        # which yields an all-zero value target (weak learning signal). Track the rate
        # as a watchdog. avg_len divides by 2 because augment_mp doubles each game's
        # samples (original + mirror).
        draws = wins.get(None, 0)
        draw_rate = draws / cfg.games_per_iteration if cfg.games_per_iteration else 0.0
        avg_len = (n_new_samples / 2.0 / cfg.games_per_iteration
                   if cfg.games_per_iteration else 0.0)
        _log(f"[iter {it+1}/{cfg.num_iterations}] self-play done: "
             f"{cfg.games_per_iteration} games ({sp_secs:.0f}s) | wins: {win_dist} "
             f"| draw_rate={100*draw_rate:.0f}% avg_len~{avg_len:.0f}")
        # One timer: sp_secs is the column the row reports, so the rate cannot
        # disagree with its own denominator.
        learner_sims = sp_stats.get("learner_sims")
        learner_sims_per_second = (round(learner_sims / sp_secs, 2)
                                   if learner_sims and sp_secs > 0 else None)
        if sp_stats.get("mean_expanded_actions"):
            _log(f"[iter {it+1}/{cfg.num_iterations}] search: "
                 f"expanded={sp_stats['mean_expanded_actions']:.1f} actions/node "
                 f"visits/action={sp_stats['visits_per_action']:.1f} "
                 f"walls/game={sp_stats['walls_placed_per_game']:.2f} "
                 f"first_wall_ply={sp_stats['first_wall_ply']} "
                 f"learner_sims/s={learner_sims_per_second}")
        if draw_rate > 0.20:
            _log(f"[iter {it+1}/{cfg.num_iterations}] WARNING: draw_rate "
                 f"{100*draw_rate:.0f}% > 20% — games timing out at "
                 f"max_game_moves={cfg.max_game_moves} (weak value signal).")

        # --- 2. train ---
        t_train = time.time()
        # Pure function of the iteration - nothing to persist across resume.
        cur_lr = lr_at(cfg.lr_schedule, model.base_lr, it, cfg.num_iterations,
                       cfg.lr_final_frac)
        model.set_lr(cur_lr)
        _log(
            f"[iter {it+1}/{cfg.num_iterations}] training ({cfg.train_steps_per_iter} steps, "
            f"lr={cur_lr:.2e})...")
        lp = lv = 0.0
        warmup = max(cfg.batch_size, cfg.warmup_min_samples)
        if len(buffer) >= warmup:
            steps = cfg.train_steps_per_iter
        else:
            steps = 0
            _log(f"[iter {it+1}/{cfg.num_iterations}] training skipped — "
                 f"buffer {len(buffer)} < warmup {warmup} (filling)")
        for _ in range(steps):
            S, P, V, src = buffer.sample_batch(
                cfg.batch_size, source="greedy",
                source_share=cfg.anchored_sample_share, with_sources=True)
            a, b = model.train_step(
                S, P, V, anchor_model=anchor, anchor_weight=cfg.anchor_weight,
                value_weights=clone_seat0_value_weights(
                    src, cfg.num_players, cfg.clone_seat0_value_weight))
            lp += a
            lv += b
        lp /= max(steps, 1)
        lv /= max(steps, 1)
        train_secs = time.time() - t_train
        _log(f"[iter {it+1}/{cfg.num_iterations}] training done: "
             f"loss_p={lp:.3f} loss_v={lv:.3f} ({train_secs:.0f}s)")
        # After training, so the value error describes the head the next
        # iteration will actually self-play with.
        diagnostics = sample_diagnostics(sp_samples, cfg.num_players, model)
        if diagnostics:
            _log(f"[iter {it+1}/{cfg.num_iterations}] data: "
                 f"policy_wall_mass={diagnostics['policy_wall_mass']:.3f} "
                 f"walled_states={100*diagnostics['walled_state_share']:.0f}% "
                 f"walls_on_board={diagnostics['walls_on_board_mean']:.1f} "
                 f"value_mae walled={diagnostics.get('value_mae_walled', float('nan')):.3f} "
                 f"free={diagnostics.get('value_mae_wallfree', float('nan')):.3f}")

        # --- 2b. Checkpoint immediately after training, BEFORE eval. ---
        # Eval is a long (~hours), sequential, best-effort phase. If the process
        # dies during it we must NOT lose the self-play + training work, so we
        # persist the trained weights and advance completed_iterations here. The
        # model is not modified during eval, so latest.pt saved now == saved after
        # eval. Eval only affects best.pt (the gating champion) and this row's eval
        # columns, both updated in place below as eval progresses. An eval-phase
        # interruption therefore costs only that iteration's eval — the ~1h of
        # self-play + training is already durable and resume skips to the next iter.
        run_eval = (it + 1) % cfg.eval_every == 0 or (it +
                                                      1) == cfg.num_iterations
        eval_sims = cfg.eval_simulations or cfg.mcts_simulations
        accepted = False
        eval_best_secs = 0.0
        eval_rand_secs = 0.0
        # None, not 0.0 - a skipped eval measured nothing, and zeros here are
        # indistinguishable from a real 0%.
        ev_wr = None
        evr_wr = None

        model.save(os.path.join(checkpoint_dir, "latest.pt"))
        # Before _write_meta, so meta.json is never newer than the buffer.
        t_buf = time.time()
        buffer.save(checkpoint_dir)
        buf_secs = time.time() - t_buf
        row = dict(iter=it + 1, loss_p=lp, loss_v=lv,
                   win_vs_best=ev_wr, accepted=accepted,
                   win_vs_random=evr_wr, fair=fair, draw_rate=draw_rate,
                   # Racing evidence: under the wall mask this should fall
                   # toward a pure race. Was only ever in games.log.
                   avg_len=avg_len,
                   # First iteration after a relaunch, so a reader of the loss
                   # curve can tell a resume dip from learning dynamics.
                   resumed=(it == start_iter and start_iter > 0),
                   # Walls self-play ran with, so the curriculum is legible in
                   # the record rather than only in games.log.
                   wall_budget=budget,
                   wall_mask_fraction=frac or None,
                   # Realised, not configured — the two diverge if the sampler
                   # or a worker misbehaves, and only the realised one explains
                   # the data the iteration actually trained on.
                   opponent_mix=sp_stats.get("opponent_mix") or None,
                   samples_by_source=sp_stats.get("samples_by_source") or None,
                   # Per seat, the share of its anchored games with walls legal.
                   anchored_walled_share_by_seat=(
                       anchored_walled_share_by_seat(cfg)
                       if (getattr(cfg, "opponent_greedy_share", 0.0) or
                           getattr(cfg, "opponent_past_share", 0.0)) else None),
                   # The same cross-tab COUNTED FROM SELF-PLAY, plus samples:
                   # a run can match the intended game counts and still put
                   # almost no gradient on a seat whose games end in 6 plies.
                   anchored_realized_by_seat=sp_stats.get(
                       "anchored_realized_by_seat"),
                   # Search resolution: how wide expansion actually was and how
                   # many visits per action the sim budget bought.
                   mean_expanded_actions=sp_stats.get("mean_expanded_actions"),
                   visits_per_action=sp_stats.get("visits_per_action"),
                   walls_placed_per_game=sp_stats.get("walls_placed_per_game"),
                   first_wall_ply=sp_stats.get("first_wall_ply"),
                   # Learner searches only; see search_wall_metrics.
                   learner_sims_per_second=learner_sims_per_second,
                   # Which snapshot the past share played, so a result can be replayed.
                   champion_pool_size=len(champion_pool) or None,
                   past_opponent=(os.path.basename(pick) if past_for_iter else None),
                   **diagnostics,
                   # Denominators, so a rate in meta.json is readable on its own.
                   decided_games=None, eval_timeouts=None,
                   rand_decided_games=None, greedy_decided_games=None,
                   win_vs_greedy=None, greedy_by_seat=None,
                   secs=time.time() - t0, buffer=len(buffer),
                   sp_secs=sp_secs, train_secs=train_secs, buf_secs=buf_secs,
                   eval_best_secs=eval_best_secs, eval_rand_secs=eval_rand_secs,
                   eval_ran=run_eval)
        history.append(row)

        def _write_meta():
            with open(os.path.join(checkpoint_dir, "meta.json"), "w") as f:
                json.dump({"completed_iterations": it +
                          1, "history": history}, f)

        _write_meta()  # durable resume point: self-play + training now survive a crash

        # --- 3. accept/reject vs best (candidate rotates seats) — best-effort ---
        # Eval uses fewer sims than self-play when eval_simulations is set:
        # eval only measures relative strength, so it doesn't need full search
        # depth. This does NOT weaken the trained model (self-play keeps full sims).
        # eval_every: skip eval on most iterations to save time (eval is expensive).
        if run_eval:
            # Geometry + sims for the parallel eval workers (spawned processes).
            # Built for every eval below, not just the gate.
            eval_config_dict = {
                "num_players": cfg.num_players,
                "board_size": getattr(cfg, "board_size", None) or model.board_size,
                "max_walls_per_player": getattr(cfg, "max_walls_per_player", 3),
                "max_turns": getattr(cfg, "max_turns", cfg.max_game_moves),
                # Without this a resumed v1 run would self-play on v1 planes and
                # gate on v2 ones, comparing both models on a scale neither saw.
                "spec_version": getattr(cfg, "spec_version", CURRENT_SPEC),
                "eval_simulations": eval_sims,
                "max_game_moves": cfg.max_game_moves,
                "leaf_batch": cfg.leaf_batch,
                "virtual_loss": cfg.virtual_loss,
                "eval_opening_plies": cfg.eval_opening_plies,
                "batch_wait_ms": cfg.batch_wait_ms,
                # Every search setting, not just the wall cap: the gate compares
                # a candidate to a champion, so eval must search the way
                # self-play does or it measures a different pair of agents.
                "mcts_wall_candidates": cfg.mcts_wall_candidates,
                "mcts_c_puct": cfg.mcts_c_puct,
                "mcts_dirichlet_alpha": cfg.mcts_dirichlet_alpha,
                "minimax_depth": cfg.minimax_depth,
                "minimax_wall_candidates": cfg.minimax_wall_candidates,
                "adjudicate": False,
            }
            # How the GATE deviates from a normal eval, defined once for both
            # engines: its own move cap (max_turns raised alongside — the env
            # ends games there on its own) and timeout adjudication.
            gate_moves = cfg.gate_max_game_moves or cfg.max_game_moves
            gate_overrides = {
                "max_game_moves": gate_moves,
                "max_turns": max(eval_config_dict["max_turns"], gate_moves),
                "adjudicate": cfg.gate_adjudicate,
            }
            if not cfg.parallel_eval:
                # The learner's eval agent, shared by every sequential eval
                # below — vs-random/greedy/minimax still run while the gate is
                # held, so it cannot live inside the vs-best branch.
                # dirichlet_epsilon=0 → best-play eval (matches the parallel
                # path); game diversity comes from the sampled opening.
                cand = mcts_agent_mp(
                    _mcts(model, env, cfg, sims=eval_sims, dirichlet_epsilon=0.0),
                    temperature=0.1, opening_plies=cfg.eval_opening_plies)

            if not gate_armed:
                # Skipped outright rather than run and discarded: this eval was
                # 41% of v7's wall time and could not fire.
                _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs best SKIPPED "
                     f"(gate held until win_vs_greedy > {cfg.gate_arm_greedy_min:.2f})")
                row.update(gate_armed=False, win_vs_best=None, accepted=False)
                _write_meta()

            if gate_armed:
                t_eval_best = time.time()
                _log(
                    f"[iter {it+1}/{cfg.num_iterations}] eval vs best ({cfg.eval_games} games, {eval_sims} sims)...")

                def _eval_progress(done, total, r):
                    elapsed = time.time() - t_eval_best
                    _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs best: "
                         f"{done}/{total} games ({elapsed:.0f}s, cand {r.candidate_win_rate:.0%})")

                if cfg.parallel_eval:
                    ev = evaluate_parallel_mp(
                        model, best, {**eval_config_dict, **gate_overrides},
                        num_games=cfg.eval_games,
                        num_workers=cfg.num_workers, batch_size=cfg.inference_batch_size,
                        on_progress=_eval_progress, base_seed=it * 100_003, log=_log)
                else:
                    champ = mcts_agent_mp(
                        _mcts(best, env, cfg, sims=eval_sims, dirichlet_epsilon=0.0),
                        temperature=0.1, opening_plies=cfg.eval_opening_plies)
                    ev = evaluate_mp(env, cand, champ, num_games=cfg.eval_games,
                                     max_moves=gate_overrides["max_game_moves"],
                                     on_progress=_eval_progress,
                                     base_seed=it * 100_003,
                                     adjudicate=gate_overrides["adjudicate"])
                accepted = ev.should_accept(threshold)
                eval_best_secs = time.time() - t_eval_best
                ev_wr = ev.candidate_win_rate
                if accepted:
                    best.copy_weights_from(model)
                    best.save(os.path.join(checkpoint_dir, "best.pt"))
                    # Snapshot every champion, so the past-opponent share draws from
                    # a spread of earlier selves rather than only the newest one —
                    # sampling old versions is what stabilised Bansal et al.
                    if cfg.opponent_past_share:
                        champion_pool.append(snapshot_champion(
                            best, checkpoint_dir, it + 1, cfg.champion_pool_size))
                # Distinguish losing the gate from having too little evidence.
                min_decided = ev.num_games * ev.MIN_DECIDED_FRACTION
                if not accepted and ev.decided_games < min_decided:
                    verdict = (f"reject (only {ev.decided_games}/{ev.num_games} games "
                               f"decided — too few to gate on)")
                    thin_evals += 1
                    # Nothing aggregated the per-eval line, so 13 consecutive
                    # unreachable evals went by unnoticed in v7.
                    if thin_evals >= 3 and not warned_unreachable:
                        warned_unreachable = True
                        _log(f"WARNING: {thin_evals} consecutive evals decided "
                             f"fewer than the {min_decided:.0f} games the accept "
                             f"gate requires ({ev.MIN_DECIDED_FRACTION:.0%} of "
                             f"{ev.num_games}). The gate CANNOT accept at this "
                             f"decided rate regardless of strength — lower "
                             f"eval_games, raise max_game_moves, or check that "
                             f"the champion can reach its goal at all.")
                else:
                    thin_evals = 0
                    verdict = "ACCEPT" if accepted else "reject"
                _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs best done: "
                     f"{100*ev_wr:.1f}% of {ev.decided_games} decided{ev.adj_note} "
                     f"{verdict} ({eval_best_secs:.0f}s)")
                # Persist the accept/reject before eval-vs-random, so best.pt and the
                # row's `accepted` stay consistent even if the next phase is interrupted.
                row.update(win_vs_best=ev_wr, accepted=accepted, gate_armed=True,
                           decided_games=ev.decided_games,
                           eval_adjudicated=ev.adjudicated,
                           eval_timeouts=ev.num_games - ev.decided_games,
                           eval_best_secs=eval_best_secs, secs=time.time() - t0)
                _write_meta()

            # --- 4. eval vs random ---
            t_eval_rand = time.time()
            _log(
                f"[iter {it+1}/{cfg.num_iterations}] eval vs random ({cfg.eval_random_games} games, {eval_sims} sims)...")

            def _eval_rand_progress(done, total, r):
                elapsed = time.time() - t_eval_rand
                _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs random: "
                     f"{done}/{total} games ({elapsed:.0f}s, cand {r.candidate_win_rate:.0%})")

            if cfg.parallel_eval:
                evr = evaluate_against_random_parallel_mp(
                    model, eval_config_dict, num_games=cfg.eval_random_games,
                    num_workers=cfg.num_workers, batch_size=cfg.inference_batch_size,
                    on_progress=_eval_rand_progress, base_seed=it * 100_003 + 50_000,
                    log=_log)
            else:
                evr = evaluate_against_random_mp(env, cand, num_games=cfg.eval_random_games,
                                                 max_moves=cfg.max_game_moves,
                                                 on_progress=_eval_rand_progress,
                                                 base_seed=it * 100_003 + 50_000)
            eval_rand_secs = time.time() - t_eval_rand
            evr_wr = evr.candidate_win_rate
            _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs random done: "
                 f"{100*evr_wr:.1f}% ({eval_rand_secs:.0f}s)")
            row.update(win_vs_random=evr_wr, eval_rand_secs=eval_rand_secs,
                       rand_decided_games=evr.decided_games,
                       secs=time.time() - t0, eval_ran=True)
            _write_meta()

            # --- 5. eval vs greedy (absolute yardstick; opt-in) ---
            if cfg.eval_greedy_games:
                t_eval_greedy = time.time()
                _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs greedy "
                     f"({cfg.eval_greedy_games} games, {eval_sims} sims)...")

                def _eval_greedy_progress(done, total, r):
                    elapsed = time.time() - t_eval_greedy
                    _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs greedy: "
                         f"{done}/{total} games ({elapsed:.0f}s, cand {r.candidate_win_rate:.0%})")

                if cfg.parallel_eval:
                    evg = evaluate_against_greedy_parallel_mp(
                        model, eval_config_dict, num_games=cfg.eval_greedy_games,
                        num_workers=cfg.num_workers, batch_size=cfg.inference_batch_size,
                        on_progress=_eval_greedy_progress,
                        base_seed=it * 100_003 + 70_000, log=_log)
                else:
                    greedy = greedy_agent()
                    evg = evaluate_mp(env, cand, greedy, num_games=cfg.eval_greedy_games,
                                      max_moves=cfg.max_game_moves,
                                      on_progress=_eval_greedy_progress,
                                      base_seed=it * 100_003 + 70_000)
                eval_greedy_secs = time.time() - t_eval_greedy
                evg_wr = evg.candidate_win_rate
                # Per seat, because the seats are not symmetric: two racers meeting
                # head-on let the SECOND one jump the first, so seat 0 must spend a
                # wall to win while seat 1 wins by racing. A pooled number hides
                # which of the two the model has failed to learn.
                # str keys: json.dump stringifies int keys anyway, so building
                # them that way keeps the in-memory row and the reloaded
                # meta.json the same shape.
                greedy_by_seat = {str(s): [evg.seat_wins.get(s, 0), n]
                                  for s, n in sorted(evg.games_per_seat.items())}
                seat_str = " ".join(f"seat{s}:{w}/{n}"
                                    for s, (w, n) in greedy_by_seat.items())
                _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs greedy done: "
                     f"{100*evg_wr:.1f}% of {evg.decided_games} decided "
                     f"[{seat_str}] ({eval_greedy_secs:.0f}s)")
                row.update(win_vs_greedy=evg_wr, eval_greedy_secs=eval_greedy_secs,
                           greedy_decided_games=evg.decided_games,
                           greedy_by_seat=greedy_by_seat,
                           # Whether this number still measures generalisation.
                           greedy_in_training=bool(cfg.opponent_greedy_share),
                           secs=time.time() - t0)

                if evg_wr > greedy_peak_rate:
                    greedy_peak_rate = evg_wr
                    evals_since_peak = 0
                    atomic_model_save(
                        model, os.path.join(checkpoint_dir, "greedy_peak.pt"))
                    row["greedy_peak_saved"] = True
                    _log(f"[iter {it+1}/{cfg.num_iterations}] new greedy peak "
                         f"{evg_wr:.1%} — saved greedy_peak.pt")
                else:
                    evals_since_peak += 1
                    if (cfg.peak_stall_evals
                            and evals_since_peak >= cfg.peak_stall_evals):
                        stop_reason = (
                            f"peak stall: {evals_since_peak} consecutive greedy "
                            f"evals without a new pooled peak (best "
                            f"{greedy_peak_rate:.1%}). greedy_peak.pt holds it; "
                            f"resume by raising peak_stall_evals to search "
                            f"longer.")

                # Arm the gate the moment the learner clears the absolute bar,
                # and seed the champion from it — a racer that reaches its goal,
                # so gate games can decide.
                if not gate_armed and evg_wr > cfg.gate_arm_greedy_min:
                    gate_armed = True
                    best.copy_weights_from(model)
                    best.save(os.path.join(checkpoint_dir, "best.pt"))
                    row.update(gate_armed=True, gate_armed_at=it + 1)
                    _log(f"[iter {it+1}/{cfg.num_iterations}] GATE ARMED: "
                         f"win_vs_greedy={evg_wr:.1%} > {cfg.gate_arm_greedy_min:.2f}. "
                         f"Champion seeded from this iteration; gating resumes "
                         f"at the next eval.")

            if cfg.eval_minimax_games:
                t_eval_mm = time.time()
                _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs minimax "
                     f"(depth {cfg.minimax_depth}, {cfg.eval_minimax_games} games)...")

                def _eval_mm_progress(done, total, r):
                    _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs minimax: "
                         f"{done}/{total} games ({r.candidate_win_rate:.0%})")

                if cfg.parallel_eval:
                    evm = evaluate_against_minimax_parallel_mp(
                        model, eval_config_dict, num_games=cfg.eval_minimax_games,
                        num_workers=cfg.num_workers, batch_size=cfg.inference_batch_size,
                        on_progress=_eval_mm_progress,
                        base_seed=it * 100_003 + 90_001, log=_log)
                else:
                    evm = evaluate_mp(
                        env, cand,
                        minimax_agent(cfg.minimax_depth, cfg.minimax_wall_candidates),
                        num_games=cfg.eval_minimax_games,
                        max_moves=cfg.max_game_moves,
                        on_progress=_eval_mm_progress,
                        base_seed=it * 100_003 + 90_001)
                eval_minimax_secs = time.time() - t_eval_mm
                minimax_by_seat = {str(s): [evm.seat_wins.get(s, 0), n]
                                   for s, n in sorted(evm.games_per_seat.items())}
                mm_seat_str = " ".join(f"seat{s}:{w}/{n}"
                                       for s, (w, n) in minimax_by_seat.items())
                _log(f"[iter {it+1}/{cfg.num_iterations}] eval vs minimax done: "
                     f"{100*evm.candidate_win_rate:.1f}% of {evm.decided_games} "
                     f"decided [{mm_seat_str}] ({eval_minimax_secs:.0f}s)")
                row.update(win_vs_minimax=evm.candidate_win_rate,
                           eval_minimax_secs=eval_minimax_secs,
                           minimax_decided_games=evm.decided_games,
                           minimax_by_seat=minimax_by_seat,
                           minimax_depth=cfg.minimax_depth,
                           secs=time.time() - t0)

            if cfg.eval_greedy_games:
                # Racing decay: best per-seat rate against its own running peak.
                if cfg.greedy_stop_patience:
                    seated = [(w / n, n) for w, n in greedy_by_seat.values() if n]
                    best_seat, best_seat_n = max(seated) if seated else (0.0, 0)
                    row["greedy_best_seat"] = best_seat
                    prev_below = greedy_below_peak
                    best_seat_peak, greedy_below_peak = racing_decay_strike(
                        best_seat, best_seat_peak, greedy_below_peak,
                        cfg.greedy_stop_drop, best_seat_n, cfg.greedy_stop_z)
                    floor_after = max(cfg.wall_mask_iters,
                                      getattr(cfg, "greedy_min_seat_after", 0))
                    if cfg.greedy_min_seat and it >= floor_after:
                        # Armed only once the curriculum has had its chance —
                        # every run is below the floor while still untrained.
                        greedy_below_floor = stalled_below_floor(
                            best_seat, cfg.greedy_min_seat, greedy_below_floor)
                        if greedy_below_floor >= cfg.greedy_stop_patience:
                            stop_reason = (
                                f"never acquired racing: best per-seat greedy rate "
                                f"{100*best_seat:.0f}% stayed under the "
                                f"{100*cfg.greedy_min_seat:.0f}% floor for "
                                f"{greedy_below_floor} consecutive evals after "
                                f"iteration {floor_after}. The curriculum is not "
                                f"delivering at this setting; more iterations will "
                                f"not fix it.")
                    if greedy_below_peak > prev_below:
                        _log(f"[iter {it+1}/{cfg.num_iterations}] WARNING: racing decay — "
                             f"best seat {100*best_seat:.0f}% is {100*(best_seat_peak-best_seat):.0f} "
                             f"pts below the peak {100*best_seat_peak:.0f}% "
                             f"({greedy_below_peak}/{cfg.greedy_stop_patience} evals)")
                        if greedy_below_peak >= cfg.greedy_stop_patience:
                            stop_reason = (
                                f"racing decay: best per-seat greedy rate {100*best_seat:.0f}% "
                                f"stayed >={100*cfg.greedy_stop_drop:.0f} pts below its peak "
                                f"{100*best_seat_peak:.0f}% for {greedy_below_peak} consecutive "
                                f"evals. greedy_peak.pt holds the strongest greedy eval "
                                f"({greedy_peak_rate:.1%} pooled); resume by raising "
                                f"greedy_stop_patience if this was noise.")
                _write_meta()
        else:
            _log(
                f"[iter {it+1}/{cfg.num_iterations}] eval skipped (eval_every={cfg.eval_every})")

        vs_best_txt = (f"{100*ev_wr:.1f}% {'ACCEPT' if accepted else 'reject'}"
                       if ev_wr is not None else "n/a (not evaluated)")
        vs_rand_txt = f"{100*evr_wr:.1f}%" if evr_wr is not None else "n/a"
        # Measured at iteration end, so it needs its own write.
        row["rss_gb"] = _rss_gb()
        _write_meta()
        _log(f">>> iter {it+1} | loss_p={lp:.3f} loss_v={lv:.3f} | "
             f"vs_best={vs_best_txt} | "
             f"vs_rand={vs_rand_txt} | draw={100*draw_rate:.0f}% | buf={len(buffer)} | "
             f"rss={row['rss_gb']:.2f}GB | "
             f"sp={sp_secs:.0f}s train={train_secs:.0f}s "
             f"eval_best={eval_best_secs:.0f}s eval_rand={eval_rand_secs:.0f}s | "
             f"total={row['secs']:.0f}s")

        if stop_reason:
            # meta.json and both checkpoints are already written for this
            # iteration, so the run resumes from here untouched if wanted.
            _log(f"STOPPING EARLY at iteration {it+1}/{cfg.num_iterations} — {stop_reason}")
            break
    return history

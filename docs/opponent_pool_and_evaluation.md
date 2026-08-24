# Opponent pool and held-out evaluation

Design note for `feat/seat-anchored-opponents`. Covers why the greedy baseline
cannot stay the headline metric once we train against it, what the literature
does instead, the concrete design for this repo, and the instrumentation needed
to attribute any change to a cause.

---

## 1. The problem

`win_vs_greedy` is currently doing two incompatible jobs.

As a **diagnostic** it has been the most valuable metric in the project: it is
the only one that did not saturate, and it exposed that both 9x9 models score 0%
against a walk-forward opponent while `vs_random` said 96-100% and the gate said
"improving".

As soon as greedy enters the **training** distribution it stops measuring
generalisation and starts measuring fit. That is the standard train/test
argument, and it is not hypothetical here - the whole reason to add greedy to
training is that it presents a distribution the model is failing on, which is
exactly the distribution we report.

We currently have no held-out opponent at all:

| metric | what it measures | usable as headline? |
|---|---|---|
| `vs_random` | saturates at 100% by iteration 5 | no - no discriminative power |
| `vs_best` (gate) | strength relative to a moving champion | no - relative only |
| `vs_greedy` | absolute competence vs a fixed racer | **only while it stays out of training** |

## 2. What the literature does

**Training against scripted opponents is normal; training against *only* one is
the failure mode.** Agents trained against a single fixed expert overfit and fail
to generalise, and fixed-policy opponents visit a narrow slice of the state
space. The established answer is a population:

- **Bansal et al., ICLR 2018** - training against only the most recent opponent
  destabilises training (one side runs away with it); sampling **random past
  versions** yields stable training and more robust policies.
- **AlphaStar (Vinyals et al., Nature 2019)** - a league of main agents plus
  main/league *exploiters*, with **PFSP** sampling opponents in proportion to
  their win rate against the main agent. Stated principle: "playing to win is
  insufficient" - some agents exist to expose flaws.
- **PSRO / Double Oracle (Lanctot et al., 2017)** - best-respond to a *mixture*
  over a population; best-responding to any single strategy overfits to it.

**The evaluation discipline is the decisive precedent.** AlphaZero trained purely
by self-play and used Stockfish / elmo / AlphaGo Zero **only** as benchmarks,
including as periodic in-training progress checks - measurement without gradient.
AlphaStar league-trained, then reported against **humans on Battle.net under
blind conditions**. The invariant in both: *the headline opponent is never in the
training pool.*

Conclusion for us: **train against greedy, but stop reporting it as the headline.**

## 3. Design

### 3.1 Training opponent pool

Per self-play game, sample the opponent set. Fixed shares to start - PFSP-style
adaptive weighting is a later refinement and would confound the first ablation.

| opponent | share | purpose | precedent |
|---|---|---|---|
| current self | ~60% | keeps this a self-play method | AlphaZero |
| past champions (`best.pt` snapshots) | ~25% | prevents cycling and forgetting | Bansal et al. |
| greedy racer, model in the evaluated seat | ~15% | supplies the seat the self-play equilibrium under-trains | AlphaStar exploiters |

Training samples are collected **only for the model's own seats**; a scripted
opponent's moves are not policy targets.

**The share is over games, but the gradient is over samples, and the two differ
by an order of magnitude.** An anchored game contributes only the model's own
plies - 1/2 of them at N=2, **1/4 at N=4** - and a scripted racer also *ends the
game sooner*, since it heads straight for its goal instead of wandering.
Measured end to end with an untrained net:

| board | samples/anchored game | samples/self-play game | ratio | 25% game share becomes |
|---|---|---|---|---|
| 5x5 N=2 | 6.0 | 50.3 | 8x | 3.8% of samples |
| **9x9 N=2** | **14.0** | **127.0** | **9x** | **3.5% of samples** |

Solving for the share needed at a 9x ratio: a **20% sample** share wants roughly
a **70% game** share. That is an early-training figure - as the model learns to
race, self-play games shorten toward the anchored length and the ratio falls
toward the seat count (2x at N=2, 4x at N=4) - but early training is exactly
when the anchoring has to bite.

So the doc's original 15% is far too low. Start nearer **0.35 at N=2** and
higher at N=4, and treat `samples_by_source` as the feedback signal: check it on
the first evaluated iteration and raise `opponent_greedy_share` until the
anchored sample share is where you want it. `opponent_mix` counts games and will
always look reassuringly larger.

### 3.2 Why this is expected to help each variant

Different diseases, same medicine:

- **N=4 - objective mismatch.** Greedy-vs-greedy at N=4 9x9 is 200/200 for
  seat 0, with the winner finishing in **6.2 moves on an 8-move board**, i.e.
  through jumps. Among four identical agents, being *late* is rewarded (masked
  self-play: P0 wins ~0-6 of 40, P3 wins 15-27), so self-play converges away from
  the seat-0 racing the evaluation measures. Anchoring in seat 0 against racers
  makes the evaluated problem stationary and present in training.
- **N=2 - a missing sub-problem.** A pure racer caps at **50% pooled** (greedy
  seat 1 wins 200/200). The model is at 45% - 18/20 in seat 1, **0/20 in seat 0**.
  Seat 0 is won only by spending a tempo on a wall that costs the opponent more;
  the 5x5 model does exactly this at 1 wall/game. Anchoring is the only route
  past 50%, because self-play presents that problem only incidentally and against
  a drifting opponent.

Symmetry worth stating in the writeup: **at both player counts the seat self-play
under-trains is exactly the seat the greedy baseline makes decisive.**

### 3.3 The held-out baseline: depth-limited minimax

A minimax / alpha-beta agent on the standard Quoridor path-difference heuristic:

```
eval(s) = MovesToFinish(opponent) - MovesToFinish(self)
```

Chosen for four reasons, not arbitrarily:

1. It is **the** classic Quoridor baseline (the widely-cited Quoridor AI design
   reports use exactly this term; the Respall et al. MCTS-for-Quoridor study
   benchmarks against a minimax bot).
2. **It places walls.** `greedy_agent` is documented pawn-rush - "never a wall".
   So minimax tests precisely the skill our models lack, and is unlikely to
   saturate the way `vs_random` did.
3. The BFS it needs already exists in `src/env/pathing.py`.
4. It is deterministic given a tie-break seed, so run-to-run variance is only
   from our own agent.

**Compute caveat.** Naive depth 3 over 131 actions is ~2M nodes with a BFS at
each. Mitigation is the standard engine trick: restrict wall candidates to those
adjacent to either player's current shortest path (typically 10-20, not 128).
Depth 2 with that restriction is realistic for 20-40 evaluation games; depth 3 is
a stretch goal and must be benchmarked before it goes in an eval loop.

### 3.4 Reporting split

| metric | role after anchoring |
|---|---|
| **vs minimax-d2, per seat** | **headline - held out, never trained against** |
| vs greedy, per seat | training-time diagnostic; **must be labelled contaminated** |
| vs random | sanity floor (known to saturate) |
| vs champion (gate) | relative progress only |

Report per seat and normalise by the pure-racer ceiling, so the player counts are
comparable: N=2's 45/50 is **90% of achievable**; N=4's 2/25 is **8%**.

## 4. Instrumentation

The project has twice been unable to attribute an improvement to a cause
(`n4_9x9_v5`: "three things changed together"; the v4/v5 runs turned out to
contain none of their intended fixes). Anchoring changes the data distribution,
so without new instrumentation any result will be similarly unattributable.

### 4.1 Per-iteration history rows (`meta.json`)

Existing rows keep everything they have. New keys:

| key | why |
|---|---|
| `opponent_mix` | realised counts `{self, past, greedy}`, not the configured shares - confirms the sampler did what the config said |
| `samples_by_source` | training samples contributed per opponent type; the shares are over *games*, and anchored games differ in length |
| `champion_pool_size`, `champion_pool_iters` | which snapshots were live, so a result can be replayed |
| `wall_mask_fraction`, `wall_budget` | already added; keeps the curriculum legible |
| `seat_win_rate_selfplay` | per-seat self-play wins - the jump-camping signal at N=4 |
| `walls_placed_per_game` | mean wall ACTIONS played per game, over every game including timeouts - distinguishes "learned wall economy" from "stopped walling". Not `walls_on_board_mean`, which counts wall *cells*, sample-weighted. (Shipped unsplit; the masked/full split was never built.) |
| `first_wall_ply` | when the first wall lands, averaged over the games that placed one (`None` if none did); the 5x5 model's competence signature is an *early, single* wall |
| `mean_expanded_actions`, `visits_per_action` | search width averaged over EVERY expanded node - not the root - and the resolution the sim budget buys at that width. The 9x9 opening figures (131 → 19 expanded, 4.6 → 31.6 visits/action at 600 sims) are ROOT measurements from `scripts/bench_search_actions.py`; these keys sit below them because deep nodes have fewer legal walls. Compare the K=0 vs K=16 ratio, not the absolute number against 131 |
| `learner_sims` | learner simulations this iteration; `training_mp` divides by the
  `sp_secs` it reports to get `learner_sims_per_second`, so the rate and its
  denominator always agree. Excludes the frozen champion's own searches and
  scripted-greedy plies, so it is a relative trend, not absolute throughput |
| `anchored_realized_by_seat` | the anchored cross-tab COUNTED FROM SELF-PLAY - per seat `{games, samples, walled_share}`. `anchored_walled_share_by_seat` reports what the schedule *intends*, so it cannot see a run diverging from it; the samples column catches a seat whose games end in a handful of plies and therefore carries almost no gradient despite a matching game count |
| `value_mae_by_state_type` | value error on walled vs wall-free states - measures the coverage loss in §3b of PR #41 directly |
| `policy_wall_mass` | mean policy mass on wall actions at the root, and after Dirichlet noise - the 0.00024 → 0.245 quantity, tracked over training |

### 4.2 Evaluation rows

Per seat for every baseline, plus decided-game denominators (already present for
greedy - extend to minimax):

- `minimax_by_seat`, `minimax_decided_games`, `minimax_depth`, `minimax_nodes`
- `greedy_by_seat` (existing) - now flagged `greedy_in_training: true/false`
- `ceiling_fraction`: score ÷ pure-racer ceiling, per baseline

### 4.3 Figures

`src/utils/plots.py` is already the single figure implementation, so these are
new panels rather than a new plotting path:

1. **Baselines on one axis** - minimax (emphasis), greedy (dashed once
   contaminated), random (gray), with the pure-racer ceiling as a horizontal rule.
2. **Per-seat split** - one line per seat against each baseline. Pooled numbers
   fuse two different tests and hide which one moved.
3. **Opponent mix over iterations** - stacked area of realised shares; catches a
   sampler that silently stopped anchoring.
4. **Wall economy** - walls placed per game and first-wall ply, masked vs full.
5. **Value error by state type** - walled vs wall-free, the coverage-loss panel.
6. **Policy wall mass** - pre- and post-noise, with the 0→1 budget transition
   marked.

### 4.4 Logs

Extend the existing `>>> iter N | ...` line with the realised mix and the
held-out score, so a tail of the log answers "what is it training against and is
it working" without opening `meta.json`.

## 5. Ablation discipline

Anchoring, the mixed curriculum, and the new run sizing must not land in one run,
or the result is unattributable in exactly the way `n4_9x9_v5` was.

| arm | curriculum | opponents | purpose |
|---|---|---|---|
| A (control) | mixed | self only | the queued N=2 run - also the N=2 control |
| B | mixed | self + past + greedy | the anchoring effect |

At N=4 the existing three probes (0/1200 at 150 iters, 1/80, 2/80) stand in as
the no-anchoring evidence. They are not a matched control - different curriculum,
different sizing - and the writeup must say so. If time allows, a matched N=4 arm
A is worth 14 h; if not, the effect size needed to claim anything at N=4 is large
enough that the unmatched comparison is still informative.

## 6. Risks

- **The minimax baseline may be much stronger than greedy**, since it walls.
  Early scores near zero are expected and are not a regression; the value of the
  metric is that it has headroom, unlike `vs_random`.
- **Anchoring can bias toward beating greedy specifically.** The past-champion
  share and the majority self-play share exist to limit this, and the held-out
  baseline is what detects it if it happens.
- **Compute.** Anchored games run a scripted opponent for some seats, which is
  cheaper than MCTS; anchored iterations should be slightly *faster* than pure
  self-play. Verify rather than assume.
- **`max_walls_per_player` is fixed by the official rules** (10 at N=2, 5 at N=4)
  and is not a tuning knob.

## References

- Bansal et al., *Emergent Complexity via Multi-Agent Competition*, ICLR 2018 -
  https://arxiv.org/pdf/1710.03748
- Vinyals et al., *Grandmaster level in StarCraft II* (AlphaStar), Nature 2019 -
  https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/
- Lanctot et al., *A Unified Game-Theoretic Approach to Multiagent RL* (PSRO),
  2017 - https://arxiv.org/pdf/1711.00832
- Silver et al., *AlphaZero* - https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/
- Respall, Brown, Aslam, *Quoridor agent using Monte Carlo Tree Search*, 2018
- Quoridor AI design report (path-difference minimax) -
  https://github.com/ltiao/quoridor/blob/master/report.mdown

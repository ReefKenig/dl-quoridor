# N=4: the trailing-seat frontier

Written 2026-08-26, after v13. State of the four-player arm, why seat 0 is
finished as an axis of progress, and the ranked plan for seats 1-3.

## Where the N=4 arm stands

Held-out protocol throughout (evaluate_mp, K=16, 200 sims, opening plies from
each run's frozen config, base_seed 13). Seat 0 x/N is games won of games
played at seat 0; seats 1-3 are 0 for every entry unless stated.

Rows were measured at different games/seat, so the column is stated: pooled
is seat-0 wins over all four seats' games, and the seat-0 fraction alone is
not comparable across rows without it.

| checkpoint | recipe | games/seat | seat 0 | pooled | opening wall mass |
|---|---|---|---|---|---|
| pretrain (baseline, not shippable) | supervised greedy distillation, 0 RL iters | 40 | 34/40 (85%) | 21.2% | 0.000 |
| **v13 greedy_peak** | v12 + factored policy head (PR #69) | **40** | **33/40 (82.5%)** | **20.6%** | 0.078 |
| v13 greedy_peak (round-robin cell) | as above, 20/seat | 20 | 17/20 (85%) | 21.2% | 0.078 |
| v12 greedy_peak | v11 + clone seat-0 value defense | 85 | 55/85 (64.7%) | 16.2% | 0.236 |
| v10 greedy_peak (old pin) | v9 + policy anchor | 85 | 37/85 (43.5%) | 10.9% | 0.541 |
| v11 greedy_peak | v10 recipe + game-length fix | 20 | 6/20 (30%) | 7.5% | 0.306 |

The arc is one mechanism per step, and opening wall mass tracks all of it:
flat RL erodes the racing prior (v9-v11); excluding clone-game seat-0 value
targets converts erosion into retention with recovery (v12); factoring the
policy into a move-vs-wall gate removes the structural wall bias and restores
the full prior (v13). In-run at 600 sims, v13's iteration 4 was a PERFECT
eval: seat 0 = 20/20, pooled 25.0% - the racer ceiling exactly, the first
time any N=4 run reached it.

## Why "more iterations" cannot help seat 0

Two ceilings are touching:

1. **The racer ceiling.** Versus three greedy racers only the first mover can
   win a pure race, so a racing policy pools at most 20/80 = 25%. v13 reached
   exactly that in-run.
2. **The protocol ceiling.** At the held-out 200 sims, v13's 17/20 equals the
   supervised prior's 17/20 - the same model family cannot exceed what the
   prior itself scores under this search budget.

Post-peak iterations only reproduce the known decay (v13: 20/20 -> 15/20 ->
9/20 before the tripwire; note the tripwire is hair-triggered from a 100%
peak, since any drop is z-significant - v12 survived an identical iter-12
level only because its 75% peak kept the z-tests under threshold, then dipped
to 15% and RECOVERED). A resume with greedy_stop_patience raised can test
consolidation, but it is a stability claim, not a higher number.

**Everything that remains at N=4 is seats 1-3**: winning from behind requires
walls that delay the leader, a skill orthogonal to racing.

## Step 0 - is the target achievable at all? (measure before engineering)

Nothing in the project has ever measured whether ANY wall-using player wins
from seats 1-3 against three greedy racers. The existence proof is minimax
(depth 2, K=16) as the CANDIDATE rotated through all four seats - the exact
mirror of the greedy_vs_minimax control we already run. Scripted agents only,
no network, minutes of CPU.

- If minimax (or the v13 net at a large sim budget) scores >0 from seats 1-3:
  the seats are winnable, the per-seat rates calibrate the target, and
  minimax becomes a teacher.
- If even minimax scores 0 there: with 5 walls against three racers ahead in
  turn order the seats are effectively unwinnable UNDER THIS EVAL, the "25%
  ceiling" is a property of the evaluation rather than the models, and effort
  moves to eval design (symmetric model-vs-model tables, Elo) instead of
  training. That is a publishable finding, not a failure.

### Step 0 VERDICT (measured 2026-08-26, `outputs/trailing_seat_existence/`)

Minimax as candidate vs three greedies, all seats, scripted agents only:
depth 2 (20/seat): seat 0 2/20, seat 3 1/20, seats 1-2 0/20; depth 3
(40/seat): seat 0 4/40, **seats 1-3 = 0/120**. One trailing-seat win in 180
games across both depths. **The vs-racer evaluation is effectively unwinnable
from seats 1-3 with 5 walls - the 25% pooled ceiling is a property of the
evaluation, not a model deficiency.** Consequences:

- v13 (82.5% seat 0, 20.6% pooled) has saturated what this instrument can
  measure; the interventions below lose their premise AND their teacher
  (minimax cannot win there either) and are NOT scheduled.
- The vs-racer table remains the right instrument for racing skill and its
  erosion - the project's central finding - but the headline metric for
  overall N=4 play moves to the symmetric model-vs-model table (all four
  seats the same model, fair share 25%) and Elo over the checkpoint pool.
- The ranked list below is kept for the record, as what we WOULD have run had
  the seats been winnable.

## Update 2026-08-29: the round robin puts a number on the saturation

`outputs/round_robin/factored_pool/round_robin_n4.json` (20 cells, complete;
20 games/seat, K=16, 200 sims) measures the null this document only argued
for. Three findings change how the table above should be read.

**1. Against the greedy field, v13 ties the null.** Greedy as candidate
versus three greedies scores 20/80 = 25.0% - seat 0 wins every game, the
other three seats win none, by construction. v13 scores 17/80 = 21.2%
(Fisher p=0.71) and the factored pretrain 16/80 = 20.0%. The vs-racer pooled
number therefore does not distinguish v13 from a scripted racer: it is a
measurement of the instrument, which is the conclusion Step 0 reached from
the other direction. Quote seat-0 rates for racing skill, not the pooled
number.

**2. Pretrain does not outrank v13.** 16/80 vs 17/80 is p=1.00. The earlier
"the warm start beats every RL checkpoint" result does not survive at the
factored head; v13 matches the prior it was warm-started from rather than
losing to it, which was the point of the factored head.

**3. The self-table shows RL causing the stall.** With all four seats the
same model, the pretrain resolves 52/80 games inside the move cap while v13
resolves 29/80 and v12 28/80 (pretrain vs v13, p=5e-4). Self-play RL is not
weakening the racer here so much as producing mutual stalemate: two trained
copies both refuse to commit and the game times out. That is the same
trained-vs-trained timeout that kept the accept gate dead in v8.

The Bradley-Terry ladder over this pool (greedy anchored at 1000) reads
minimax_d2 1077, greedy 1000, pretrain 750, v13 721, v12 641 - but the model
entries rest on 312-339 decided games out of far more played, so the gaps
among the three models are within the stalling noise and should not be
quoted as an ordering.

### Open before any of this is published

- **Seat-2 anomaly in the factored pretrain self-table**: seats read
  [1, 0, 10, 2] of 20: seat 2 wins ten times while its neighbours win once or
  not at all. No mechanism predicts a seat-2 advantage in a symmetric table.
  Check the seat/mask aliasing (the v7 class of bug) before treating any
  self-table per-seat number as real.
- A second, wider sweep (v9-v13 plus the flat pretrain, 64 cells) is in
  progress in `outputs/round_robin/round_robin_n4.json`. Its scripted cells
  reproduce this one exactly; its model cells do not (v13 vs greedy is 71/80
  there against 70/80 here), so model cells carry search nondeterminism on
  top of sampling noise. Do not treat a single cell's seat split as exact.

## Interventions, ranked (moot per step 0; kept for the record)

1. **Seat-conditional anchor (v14 candidate - small, one variable).** The
   policy anchor currently pulls EVERY training state toward the racing
   prior, including trailing-seat states where racing is precisely what
   cannot win - our own defense suppresses the skill seats 1-3 need. Mask the
   anchor to seat-0-to-move states: hold racing where racing wins, release
   the prior where it cannot. The factored head can already express
   seat-conditional wall probability (the turn channel is an input).

2. **Trailing-seat teacher distillation (the real bet).** The one lesson that
   has held all project: supervised prior first, RL to polish - the warm
   start is the only thing that ever produced N=4 strength. Apply it to the
   missing skill: generate minimax-blocking games, build a mixed prior
   (greedy racing at seat 0, minimax blocking at seats 1-3), then RL with
   both defenses on. v9's playbook aimed at the other three seats.

3. **Curriculum from winnable positions.** Start trailing-seat games from
   states where a win is reachable (leader already delayed, deficit reduced),
   anneal toward the true start. No teacher required; needs custom initial
   states in the env.

4. **Canonization (egocentric frame).** Rotating the observation so every
   seat sees itself as seat 0 shares wall skill across seats (blocking from
   seat 1 and seat 2 are the same problem up to rotation) - but it cannot
   create the skill, and it does not touch turn order, so it multiplies
   whichever of 1-3 works rather than replacing them. It is a tensor-spec-3
   migration (board/wall rotation, action permutation, relative value order):
   future work.

## Honest caveat for the writeup

Even perfect play from seats 1-3 against three racers may cap at low per-seat
rates - the trailing seats must spend walls the leader never has to. The fair
headline for "is N=4 solved" may need to become the symmetric model-vs-model
table (all four seats the same model, fair share 25% each) rather than the
vs-racer pool. The vs-racer table remains the right instrument for measuring
the racing skill and its erosion; it is not the right instrument for grading
trailing-seat play.

## Operational notes

- Factored checkpoints (v13 onward) load only with PR #69's head-aware
  loaders. DONE 2026-08-29: #69 and #70 are both merged to dev, and 9x9_4p is
  pinned to `runs/n4_9x9_v13/greedy_peak.pt`.
- v13 ran in the dedicated clone /tf/dl-quoridor-v13 on its branch; the
  shared clone stayed frozen on dev for the live seed2 run. Keep that
  discipline: never git pull the shared clone while a run is live.
- The existence proof, all rescores, and the Elo/cross-table work run fine on
  the MacBook (~6 s/game at 200 sims; scripted-agent matches need no GPU at
  all). Pod time is for training runs only.

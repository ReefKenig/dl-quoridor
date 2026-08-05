# Restricted wall search (`mcts_wall_candidates`)

## 1. The problem

At 9x9 the opening offers 131 legal actions and 128 of them are walls. A
600-simulation search therefore buys **4.6 visits per action**, which is not
enough for the visit histogram to move away from the prior that seeded it: the
search returns a slightly sharpened copy of the policy it started from, and the
policy target it produces teaches the network what it already believed.

5x5, where this project's models play their board optimally, gets 17.1
visits/action at the same budget. That gap — not the network, the curriculum or
the wall mask — is what separated the two boards.

## 2. What the restriction does

`env.get_search_actions(state, max_walls)` returns pawn moves plus the
`max_walls` legal walls that cut some player's current shortest path, ranked by
how many players they cut. `MCTSConfig.wall_candidates` turns it on;
`MCTSMaxN._expand_valids` is the single place any engine lists actions, so the
sequential, root-batched, leaf-parallel and vectorized paths all inherit it.

Measured with `scripts/bench_search_actions.py`:

```
board 9x9  N=2  sims=600  K=16
                              actions  visits/action
opening, unrestricted             131            4.6
opening, K=16                      19           31.6
mean over 40 plies               48.3           12.4
  same plies, K=16               10.6           56.6

listing 40 states: unrestricted 21.0 ms, K=16 23.1 ms  (+10%)
```

At 1200 simulations the opening reaches 63.2 visits/action. N=4 behaves
identically at the opening (131 -> 19) and costs +8%.

## 3. Why dropping the other walls is exact, not a heuristic

A wall that misses every player's shortest path cannot change any player's
distance to goal on the move it is placed. The slots that *do* cut a path are
already computed — `_path_blockers` produces them for wall-legality checking and
throws them away — so the filter reuses that work, which is why it costs ~10%
rather than another BFS sweep.

`tests/test_search_actions.py::test_a_dropped_non_cutting_wall_cannot_change_any_distance`
checks this exhaustively rather than arguing it: every legal non-cutting wall is
placed and every player's distance must be unchanged, at N=2 and N=4 across
three game depths.

**What the restriction does give up.** Keeping only `K` of the *cutting* walls
is a real narrowing, and a wall that is worthless this move but valuable two
moves later can fall outside the set. That is the knob's actual trade-off; the
exactness argument above covers only the walls dropped for cutting nothing.
Pawn moves are never dropped.

## 4. Comparability — read this before putting numbers in one table

**An evaluation run with `wall_candidates != 0` is not comparable to a number
produced without it.** The restriction changes what search explores, so it
changes what a checkpoint scores, and the size of that change depends on whether
the checkpoint *trained* under it:

| checkpoint | K | vs greedy | seat 0 | seat 1 |
|---|---|---|---|---|
| `probe_n2_ramp/best.pt` (pure racer, trained K=0) | 0 | 35.0% | 0/20 | 14/20 |
| `probe_n2_ramp/best.pt` | 16 | 37.5% | 0/20 | 15/20 |
| `n2_9x9_v7/latest.pt` (trained K=16) | 0 | 47.5% | 18/20 | 1/20 |
| `n2_9x9_v7/latest.pt` | 16 | **80.0%** | 20/20 | 12/20 |

40 games, `eval_opening_plies=4`, each checkpoint on its own tensor spec.

Two conclusions, and they are different:

- **On a checkpoint that never trained under it, the restriction does nothing.**
  35.0% -> 37.5% is noise, and seat 0 stays 0/20. It is not free strength for an
  arbitrary checkpoint.
- **On a model that trained under it, it is mandatory at inference.** v7 falls
  80% -> 47.5% under unrestricted search, almost entirely in seat 1
  (12/20 -> 1/20). Its policy and value were shaped by a 19-action search and
  degrade under the 131-action one.

So the value of this branch is that it makes walled play **learnable during
training**, plus a train/test consistency requirement at deploy time. Both
halves have to be stated; either alone is misleading.

**Practical rules.**

- Do not put pre-restriction and post-restriction results in the same table
  without a K column. `runs/n2_9x9_v7` ran at K=16; v4/v5/v6 did not.
- Any script that scores a checkpoint must pass the K the checkpoint trained
  under, and must run it on its own tensor spec.
- The UI defaults to `wall_candidates=16` (`src/server/app.py`), which is
  correct for v7 and harmless for older checkpoints — `probe_n2_ramp` is flat
  under it, per the table above.

## 5. Configuration

| where | key | default |
|---|---|---|
| `configs/config_9x9.json` | `mcts_wall_candidates` | 16 |
| `TrainingConfigMP` | `mcts_wall_candidates` | 0 (off) |
| `MCTSConfig` | `wall_candidates` | 0 (off) |
| `src/server/app.py` (UI) | `wall_candidates` setting | 16 |
| minimax baseline | `minimax_wall_candidates` | 16 |

`0` disables the filter and reproduces the unrestricted search exactly —
`test_the_knob_off_is_identical_to_the_pre_restriction_search` pins the visit
distribution against an env with no `get_search_actions` at all. Negative values
behave as `0`.

The default stays off in `TrainingConfigMP` so existing 5x5 scripts and their
recorded results are untouched; 9x9 opts in through its config.

## 6. Reuse and concurrency

`_player_blockers` and `_path_blockers` are pure functions of the state passed
in — no instance cache, no memo, fresh sets per call — so there is nothing
shared for concurrent callers to race over. The parallel engines are separate
*processes* with their own env objects in any case.
`test_the_blocker_computation_holds_no_state_between_calls` and
`test_concurrent_callers_on_one_env_agree` pin both properties, so a future
cache cannot be added without a failing test.

## 7. Where this is not applied

`MCTSMaxN._random_rollout` still lists every legal action. A rollout simulates
the game rather than expanding the tree; narrowing it would change the playout's
rules, not just what search looks at. Pinned by
`test_rollouts_are_deliberately_not_restricted`.

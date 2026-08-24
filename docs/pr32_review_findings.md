# PR #32 review - findings

Review of `feat/self-play-speedup` (work-stealing tail fix + in-process
vectorized self-play, "Option B"), July 28 2026. Base `dev`, 10 commits,
+1100/−29 at review time.

**Outcome:** 7 defects found and fixed, 14 pre-existing test failures repaired,
one production startup crash found, and the benchmark that decides the branch's
own headline feature produced a negative result. 16 commits added, suite
60 passed / 0 failed.

Everything below was verified by execution, not by reading. Where a claim rests
on inference rather than measurement, it says so.

---

## Part 1 - Defects in the PR

### 1. Per-game seeding in the vectorized driver was a no-op (high)

`vectorized_self_play_mp.py` called `np.random.seed(base_seed + gi)` when a slot
started a game, commented "per-game reproducibility". It was neither.

- All G slots then drew from that one **global** stream, interleaved. A game's
  Dirichlet noise and action sampling therefore depended on how many other games
  were in flight and where each had got to.
- Every refill re-seeded the global stream mid-iteration, resetting it underneath
  the G−1 games still running.
- Samples were appended in **completion order**, which also varies with G.

**Evidence.** Same `base_seed`, production settings (eps=0.25, temp 1.0/0.3):

| `vec_games` | samples produced |
|---|---|
| 1 | 190 |
| 3 | 356 |

Entirely different data, not a different tail.

**Why the tests missed it.** All four parity tests ran at `eps=0.0` and
`temperature=0.0` - no RNG is consulted at all in that regime. The one test using
`eps=0.25` asserted only a game count.

> A determinism test that disables all randomness cannot detect a randomness bug.

**Fix.** Each slot owns a `np.random.Generator` seeded from
`game_seed(base_seed, index)`, threaded through both Dirichlet noise (new optional
`rng` argument on `_add_dirichlet_noise` / `VectorizedSearch`, defaulting to the
global stream so single-search callers are unchanged) and action sampling.
Samples are keyed by game index and flushed in order.

**Verified by** `test_vectorized_concurrency_invariant_with_noise` and
`test_vectorized_refill_does_not_disturb_inflight_games` - both fail without the
fix.

**Impact on the PR's claims.** "Bit-identical to the sequential path" held only in
the eps=0/temp=0 regime, i.e. never in a real run. What is true after the fix:
the *search* is exact sequential MCTS, and a game's data depends on its index
alone. Strength-neutrality survives - it rests on the absence of virtual loss,
not on RNG equality.

### 2. Consecutive integer seeds (medium)

The work-stealing commit seeded game *g* with `base_seed + g`, so neighbouring
games started MT19937 from adjacent integers. Exploration noise across nearby
games risks correlation, which narrows sample diversity - the exact axis the
branch promises not to move.

**Fix.** `self_play_mp.game_seed()` runs `(base_seed, index)` through a
`SeedSequence`. Deterministic per pair, so reproducibility under dynamic
game-to-worker assignment is unchanged. Applied to both engines.

### 3. All-zero action distribution → NaN → crash (medium)

`probs / probs.sum()` yields NaN when the search root has no children, surfacing
later as an opaque `ValueError` from the action sampler.

`play_one_game` has carried this line for a long time, where a crash kills one
worker and work-stealing reclaims its games. The vectorized driver runs in the
**main process**, so the same NaN takes down the whole training iteration - the
blast radius changed, which is what made it worth closing.

**Fix.** `normalize_action_probs()` in `self_play_mp`, shared by both engines:
renormalize as before, fall back to uniform over the position's valid actions when
counts are all zero, raise with a named condition when there are none.

### 4. No CPU/GPU overlap in the driver loop (low–medium)

Each round was strictly serial: descend all G trees in Python on one core → one
forward → scatter. The GPU idled through the descents and the CPU through the
forward - and at 9×9/800 sims the descents dominate.

**Fix.** Split the eval into `_submit` (queues the forward, no readback) and
`_apply` (the sync point), then process slots in `pipeline_groups` sub-batches so
one forward is always in flight. With *k* groups, *k−1* of the *k* forwards
overlap CPU work. Defaults to 2 on CUDA, 1 on CPU (where the forward is
synchronous and splitting would only shrink the batch). `k=1` is byte-for-byte
the old schedule.

**Verified by** `test_vectorized_pipeline_groups_do_not_change_results`.

**Caveat found later:** this helps, but does not address the real limit - see
Part 3.

### 5. Benchmark methodology was biased (low)

`bench_self_play.py` ran parallel first, in-process, with no warmup and a single
timed run - so parallel absorbed CUDA context init and cuDNN autotune. This is the
script that gates a multi-day training decision.

**Fix.** `--warmup-games` (auto-sized, see Part 3), `--repeat` (best-of-N), and
`--order` to swap which engine runs first.

### 6. Two docstrings overclaimed (low)

- Module docstring said the GPU batch "stays full because finished games are
  refilled" - true until the quota runs out, after which slots retire and the
  batch narrows over the final G games. There *is* a tail, just bounded by G.
- `VectorizedSearch.collect()` documented a "skip and revisit" protocol; in the
  code `collect() is None` always implies `done()`, so it is never revisited.

Both corrected to match the implementation.

### 7. `self_play_mode` was unvalidated (nit)

A typo (`"vectorised"`, `"Vectorized"`) matched neither branch and fell through to
the sequential path - a working-looking run at a fraction of the intended
throughput. The resolution logic was also duplicated between the launch banner and
the loop.

**Fix.** `resolve_self_play_mode()` raises on anything outside
`auto|sequential|parallel|vectorized`, and runs once before the loop so the config
is validated before any work starts.

---

## Part 2 - Test findings

### 2.1 The PR's own test claim was wrong

The PR body stated "9 pre-existing failures in the legacy (non-`_mp`) stack are
unrelated". Actual state on the branch: **14 failed, 1 collection error, 45
passed**. Confirmed identical before and after the PR's changes, so genuinely
pre-existing - but not 9, and not harmless.

### 2.2 Root cause A - a removed constructor kwarg (10 failures)

`QuoridorEnv` had dropped `is_poc` in favour of an explicit `board_size`
(`is_poc=True` ⇔ 5×5/3 walls, `False` ⇔ 9×9/10 walls - confirmed from the commit
that removed it). No caller was updated, so every one raised `TypeError`.

**This was not only stale tests.** `src/__main__.py:62` - the legacy training
entry point - still passed `is_poc=cfg.is_poc`, so **`python -m src` crashed on
startup**. A test suite nobody could run was hiding a broken production entry
point.

Fixed by passing `cfg.board_size`, which already derives from `is_poc` when the
config omits it and honours an explicit value when it doesn't.

### 2.3 Root cause B - the observation width had drifted (4 failures)

`state_to_tensor` returns **11** channels: `build_tensor`'s 10 base planes plus a
side-to-move plane. That number was hand-written in four places and they no longer
agreed:

| location | said |
|---|---|
| `QuoridorNetwork.in_channels` default | 10 |
| `QuoridorModel.in_channels` default | 11 |
| `env_interface` docstring | 10 |
| `test_network` synthetic states | 10 |

The conv rejected the 10-channel test inputs.

Fixed by defining `BASE_CHANNELS` / `OBS_CHANNELS` in `tensor_spec.py` - the file
that already owns the layout - and deriving from it everywhere. `build_tensor`
still emits 10; it produces the *base* planes, and `test_tensor_spec`'s contract
is unchanged.

### 2.4 Root cause C - SB3 test was structurally impossible (1 failure)

With the kwarg and channel count fixed, `test_sb3_ppo_cnn` still failed on its own
terms. SB3's `CnnPolicy` defaults to `NatureCNN`, built for Atari frames: its first
layer is an 8×8 kernel at stride 4, which **cannot consume a 5×5 board at any
channel count**, and it asserts the observation is a uint8 image while ours is
float32 in [0, 1].

Fixed with a small extractor (two padded 3×3 convs, no downsampling) plus
`normalize_images=False`. Also dropped `n_steps` to 64 so a 100-timestep smoke test
stops collecting SB3's default 2048-step rollout first - the test went from
timing out to 0.9 s.

### 2.5 Coverage gaps found

| gap | consequence | now covered by |
|---|---|---|
| Parity tests all ran at `eps=0`, `temp=0` | RNG defects invisible (Part 1 §1) | `..._with_noise`, `..._refill_does_not_disturb_inflight_games` |
| No test on output ordering | completion-order output was a function of `vec_games` | same two tests |
| No test that scheduling changes are data-neutral | pipelining could silently alter samples | `..._pipeline_groups_do_not_change_results` |
| `scripts/` is not imported by any test | a syntax error in `bench_self_play.py` passed 60 tests | see §2.6 |

### 2.6 A verification failure of my own, worth recording

While trimming comments I rewrote an `argparse` help string and deleted the
closing paren with it, leaving `bench_self_play.py` unparseable. I "verified" the
change with `pytest`, which never imports `scripts/` - **60 tests passed on a file
that could not be parsed**, and the break only surfaced in the notebook.

The same shape had occurred earlier: the bench notebook's cell sources were
generated without trailing newlines, so Jupyter joined each cell onto one line and
commented out the imports; my validation re-added the newlines before compiling,
testing something the file never contained.

Both were cases of validating something *adjacent* to the artifact rather than the
artifact itself. A syntax sweep over every changed `.py` is now part of the
pre-commit check for this repo.

### 2.7 Final state

| | before | after |
|---|---|---|
| passed | 45 | **60** |
| failed | 14 | **0** |
| errors | 1 | **0** |

3 tests added for the concurrency-invariance and pipelining properties; no test
removed.

---

## Part 3 - Benchmark findings

The branch's headline feature is opt-in pending a benchmark. Running it produced
two results, one expected and one not.

### 3.1 The benchmark itself had two measurement defects

**Warmup starved the pool.** Both engines cap concurrency at the game count
(`n_workers = min(num_workers, games)`, `G = min(vec_games, games)`). A 2-game
warmup of a 32-worker run therefore spawned **2 workers** - warming a
configuration the benchmark never measures, at full sims and full game length. At
9×9/N=4 that cost ~18 minutes per engine. Warmup now auto-sizes to the engine's
full pool at reduced sims (batch shapes, which cuDNN autotunes against, are set by
worker/leaf count, not sims).

**The same trap applies to the measurement.** Benchmarking `games=12` with
`vec_games=64` measures a 12-wide vectorized engine - removing the batch width
that is its entire advantage. `run_bench` now warns when `games` is below an
engine's configured width.

> Governing rule: **keep width, scale depth.** Width distinguishes the engines;
> `sims` and `max_moves` are paid identically by both, so reducing them preserves
> the ratio.

### 3.2 Vectorized self-play is CPU-bound - do not enable it

On the RTX PRO 6000 box at 9×9/N=4, the vectorized phase ran at **~72% of a
single core with the GPU at 0%**.

The cause is structural. The driver is single-process, so all G games' tree-walks -
`env.step` and `get_valid_actions`, the latter running a BFS per player to test
wall legality - execute on one core. At N=4 that move generation dominates, and no
amount of GPU batching addresses it.

| | tree-walk | GPU batch width |
|---|---|---|
| parallel (today) | 32 cores | 9 of 256 |
| parallel + batcher fix | 32 cores | ~240 |
| vectorized (Option B) | **1 core** | 64 |

The decisive point is asymmetric headroom: vectorized never touches the GPU
batcher (it calls `predict_batch` directly), so the batcher fix cannot help it,
while that same fix is a ~20× occupancy improvement on the engine already ahead.

**Honest limitation.** No clean production-width head-to-head was obtained. Two
completed smoke comparisons favoured parallel (0.82×, 0.58×) but were 5×5/N=2 and
too small to be conclusive. The conclusion rests on the mechanism and the resource
measurements, not on a decisive number.

### 3.3 Incidental finding - the batcher is the real bottleneck

```
[GPU] 400,008 evals (42805 batches, avg 9/batch, 904 evals/s, 120 msg/s, 7.6 leaves/msg, 442s)
```

- 42,805 batches / 442 s = 96.8 forwards/s
- 120 msg/s ÷ 96.8 forwards/s = **1.24 worker messages per forward**
- 32 workers ÷ 120 msg/s = each worker waits **~267 ms** per reply

The batcher fires on roughly one message at a time: **9 leaves per forward against
a 256 cap, ~3.5% occupancy**. Workers are not GPU-bound, they are queued.

Feasibility for N=4 (≈ 50 games × ~300 plies × 800 sims ≈ 12M evals/iteration):

| throughput | per iteration | 50-iteration run |
|---|---|---|
| 904 evals/s (measured) | 3.7 h | **~184 h - infeasible** |
| 5,000 evals/s | 40 min | ~33 h |
| 10,000 evals/s | 20 min | ~17 h |

The batch-accumulation fix on `fix/9x9-gating-and-timeouts` is therefore not an
optimization but a precondition for the N=4 run existing at all.

---

## Part 4 - Recommendations

1. **Merge PR #32.** Roughly 60% of it is valuable regardless of the vectorized
   decision: the work-stealing tail fix, the `base_seed` reuse bug, decorrelated
   seeding, the NaN guard, mode validation, the benchmark tooling, and the 14 test
   repairs including the `python -m src` crash. PR #33 is stacked on it.
2. **Do not set `self_play_mode: "vectorized"`.** Keep it in-tree, off by default,
   as a documented negative result (`academic_experiences.md` §7).
3. **Rebase PR #33** - it branches from `7086842`, before the last bench commits.
4. **Re-measure after the batcher fix**, reading `avg N/batch` and `evals/s` off
   the `[GPU]` line. That number, not the engine comparison, decides whether to
   relaunch.
5. **If throughput still disappoints, try fewer workers before writing code.**
   §6.4 of `academic_experiences.md` measured 16 workers beating 32 (995 vs 571
   evals/s) at N=2. The current config uses 32.
6. **Defer "c9"** (multi-process vectorized). It converges on what fixed-parallel
   already provides - multi-core tree-walking with wide batches - so it is not
   worth building until the batcher fix is measured and found wanting.
7. **Correct the PR body.** It currently reads "here's a speedup, benchmark then
   switch". It should read: work-stealing and correctness fixes land; Option B is
   implemented, measured, and **not** recommended on this hardware, with the
   reason.
8. **Add CI.** `mergeStateStatus: CLEAN` only means no checks are configured.
   Nothing but a local run has validated these ~1500 lines, and §2.6 shows how a
   passing suite can coexist with an unparseable file.

# `runs/` — training runs, versioned

Every training run gets its own self-contained directory. Everything about a run
— its config, its progress record, its logs, its checkpoints, and its figures —
lives together so a run is reproducible and identifiable at a glance.

## Naming convention

```
runs/<arch>_<board>_<version>/
```

e.g. `n4_5x5_v3` = 4 players, 5×5 board, run version 3.
Bump the `vN` suffix for each new run (set `RUN_DIR` in the `scripts/run_train_*.py`
driver). Never overwrite a finished run — start a new version.

## Layout of a single run

```
runs/n4_5x5_v3/
├── config.json        # frozen hyperparameters + env (TRACKED — reproducibility)
├── meta.json          # completed_iterations + full per-iter history (TRACKED — progress)
├── train.log          # training stdout for this run (TRACKED)
├── figures/           # plots generated FROM this run's metrics (TRACKED)
│   ├── n4_training_curves.png
│   └── n4_full_dashboard.png
├── latest.pt          # weights — ignored by git (large, regenerable)
├── best.pt            # weights — ignored
├── ship.pt            # the chosen release checkpoint — ignored
└── peaks/             # auto-captured new-high snapshots (watch_peak.py) — ignored
    └── peak_iter69_100.pt ...
```

## What git tracks vs. ignores

**Metadata-only versioning.** Git tracks the things that record *progress and
identity* and are small: `config.json`, `meta.json`, `*.log`, and `figures/*.png`.
Git ignores all model weights (`*.pt`/`*.pth`/`*.onnx`) and replay buffers
(`*.pkl`) — they're large and reproducible from the config + code. See `.gitignore`.

So in git, each run reads as: which config produced what curve, iteration by
iteration — without dragging multi-MB binaries into history.

## Current runs

| Run                 | Players | Status                  | Iters | Notes                          |
|---------------------|---------|-------------------------|-------|--------------------------------|
| `legacy_2p`         | 2       | legacy (old trainer)    | —     | older `metrics_full.json` format; `model_export.pt` is a downloaded snapshot |
| `n4_5x5_v1`         | 4       | superseded              | 20    | early run                      |
| `n4_5x5_v2_killed`  | 4       | killed mid-run          | —     | abandoned                      |
| `n4_5x5_v3`         | 4       | **current / finished**  | 70    | `ship.pt` is the release model |

> `config.json` is only present for runs created after this convention (v3+).
> Legacy runs keep their original `meta.json` / `metrics_full.json` as the progress record.

## Cross-run artifacts

Anything that compares *multiple* runs (e.g. 2p vs 4p) is not owned by one run and
lives in the shared top-level `outputs/` instead:

```
outputs/
├── model_comparison.png   # 2p vs 4p summary
└── results/               # model-vs-model eval dumps (negamax vs maxⁿ, etc.)
```

## Regenerating figures

```
PYTHONPATH=. python scripts/plot_all_figures.py
```

Writes each run's figures into its own `runs/<id>/figures/` and the cross-run
comparison into `outputs/`.

# XGB-v1 isolated study

From the repository root:

```bash
python3 studies/xgb_v1/run.py
```

The runner builds `airline-xgb-v1:3.0.5` with the study Dockerfile and hash-locked
CPU wheels, starts only its own two-worker Compose project, checks the full
selection population, runs the smoke checks, and starts at most one full fit
only if the resource gate passes. It saves/reloads, evaluates all validation
months, stops study services, and generates `RESULTS.md` and artifact `metrics.json`.
Current outcome: **BLOCKED / INCOMPLETE**, with zero full fits.

All new generated data and logs are under `artifacts/experiments/xgb-v1/`, already
ignored by the repository. `execution/` is exclusive: the runner refuses a second
invocation against an occupied execution directory. Do not remove fit markers or
rename full-fit outputs to bypass this protection. No automatic retries exist.
The already completed 256-row smoke fit is reused in this continuation; package,
adapter-source, model-parameter and serialization checks bind it to this runtime.

The existing production base image must match
`sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf`.
Offline wheel cache: `studies/xgb_feasibility_v1/runtime/wheels/`; the runner copies
it into the ignored build context. The build uses `--network=none` and pip's
`--require-hashes --no-deps`; production dependencies and images are not replaced.
On a fresh machine, obtain the six CPython 3.10 Linux x86-64 wheels from the study
lock into that cache first (or `artifacts/experiments/xgb-v1/build/wheels/`).
This pins the tested architecture; other platforms have not been validated.

Required inputs are the existing accepted bundle at
`artifacts/flighttime-v3/models/`, selection-only features at
`data/warehouse/dws/ml_flighttime_v3_selection/`, and LR evidence at
`studies/lr_v1/evidence/diagnostics/results.json`. Do not rebuild the production
pipeline merely for this study. All source/data mounts are read-only except the
study output directory. Deep source paths inside the container accommodate the
existing LR utility's path assumptions without editing that utility.

To regenerate the report from recorded evidence, without fitting or scoring:

```bash
python3 studies/xgb_v1/report.py
```

The full path uses a 7,200-second total wall-clock budget, including data
preparation, fitting, serialization and validation evaluation. It preserves
natural-prevalence, unweighted metrics and the frozen threshold/budget protocols.
The current dense-input lower bound exceeds physical memory plus total swap;
there is no allowed resource-preserving workaround. Full evaluation remains
unexecuted. No qualification or improvement is claimed.

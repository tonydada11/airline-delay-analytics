# Approved LR candidate results

The two approved one-factor attempts ran sequentially. Production remains the accepted LR at threshold **0.39**; no model was promoted. All comparisons use full-precision baseline monthly values from the frozen diagnostic JSON, never rounded report figures.

Baseline equal-month PR-AUC: **0.428098323**. Paired mean gain must be >=0.005, with mean recall change >=-0.01 at both monthly alert budgets. These are retrospective development-data criteria, not statistical significance or evidence of future improvement.

## Completion and compute

| Candidate | Status | Full attempt runtime | Fit runtime | Mean PR-AUC | Paired mean gain | Qualifies |
|---|---|---:|---:|---:|---:|---|
| lr-positive-weight-1 | complete | 372.447s | 129.344s | 0.427724936 | -0.000373386 | False |
| lr-reg-0p3 | complete | 357.377s | 118.628s | 0.423713216 | -0.004385106 | False |

Total active Spark study time charged: **1223.018s**, including the earlier 493.194s diagnostics. Remaining: **13176.982s**. Classifier fits started: **2**. No automatic retries or additional candidates were run.

Each attempt was bounded at 60 minutes including preparation, fitting, save/reload, evaluation, and shutdown; the in-container timeout reserved termination time within that cap. Incomplete attempts are excluded from qualification and ranking regardless of how many monthly files exist.

## Baseline monthly comparison values

| Month | PR-AUC | Precision 10% | Recall 10% | Precision 20% | Recall 20% |
|---|---:|---:|---:|---:|---:|
| 2024-05 | 0.432454907 | 51.625% | 19.545% | 46.563% | 35.257% |
| 2024-06 | 0.403536552 | 47.566% | 19.074% | 43.456% | 34.851% |
| 2024-07 | 0.470587237 | 55.951% | 18.888% | 51.171% | 34.548% |
| 2024-08 | 0.405814595 | 49.393% | 21.048% | 43.807% | 37.335% |
| Equal-month mean | 0.428098323 | 51.134% | 19.638% | 46.249% | 35.498% |

## lr-positive-weight-1

Complete evaluation. Settings: positive-class weight **1.0**, regularization **0.1**. Full training rows: 8,946,370; optimizer iterations: 20.

| Month | PR-AUC | Baseline PR-AUC | Difference |
|---|---:|---:|---:|
| 2024-05 | 0.432416610 | 0.432454907 | -0.000038297 |
| 2024-06 | 0.403231887 | 0.403536552 | -0.000304665 |
| 2024-07 | 0.470319993 | 0.470587237 | -0.000267244 |
| 2024-08 | 0.404931256 | 0.405814595 | -0.000883339 |
| Equal-month mean | 0.427724936 | 0.428098323 | -0.000373386 |

### Top 10% within each calendar month

| Month | Precision | Precision difference | Recall | Recall difference |
|---|---:|---:|---:|---:|
| 2024-05 | 51.659% | +0.033 pp | 19.558% | +0.013 pp |
| 2024-06 | 47.614% | +0.048 pp | 19.093% | +0.019 pp |
| 2024-07 | 56.006% | +0.055 pp | 18.906% | +0.019 pp |
| 2024-08 | 49.313% | -0.079 pp | 21.014% | -0.034 pp |
| Equal-month mean | 51.148% | +0.014 pp | 19.643% | +0.004 pp |

### Top 20% within each calendar month

| Month | Precision | Precision difference | Recall | Recall difference |
|---|---:|---:|---:|---:|
| 2024-05 | 46.536% | -0.028 pp | 35.237% | -0.021 pp |
| 2024-06 | 43.480% | +0.025 pp | 34.871% | +0.020 pp |
| 2024-07 | 51.161% | -0.010 pp | 34.542% | -0.007 pp |
| 2024-08 | 43.736% | -0.071 pp | 37.274% | -0.061 pp |
| Equal-month mean | 46.228% | -0.021 pp | 35.481% | -0.017 pp |

Worst monthly PR-AUC change: **-0.000883339** in **2024-08**. Mean recall constraints do not guarantee that every month improves; no extra monthly elimination rule was added.
Smallest monthly recall change at 10%: **-0.034 pp** in **2024-08**.
Smallest monthly recall change at 20%: **-0.061 pp** in **2024-08**.

## lr-reg-0p3

Complete evaluation. Settings: positive-class weight **2.0**, regularization **0.3**. Full training rows: 8,946,370; optimizer iterations: 15.

| Month | PR-AUC | Baseline PR-AUC | Difference |
|---|---:|---:|---:|
| 2024-05 | 0.429807710 | 0.432454907 | -0.002647197 |
| 2024-06 | 0.399154115 | 0.403536552 | -0.004382437 |
| 2024-07 | 0.466013005 | 0.470587237 | -0.004574232 |
| 2024-08 | 0.399878035 | 0.405814595 | -0.005936560 |
| Equal-month mean | 0.423713216 | 0.428098323 | -0.004385106 |

### Top 10% within each calendar month

| Month | Precision | Precision difference | Recall | Recall difference |
|---|---:|---:|---:|---:|
| 2024-05 | 51.455% | -0.170 pp | 19.481% | -0.064 pp |
| 2024-06 | 47.159% | -0.407 pp | 18.910% | -0.163 pp |
| 2024-07 | 55.329% | -0.622 pp | 18.678% | -0.210 pp |
| 2024-08 | 48.704% | -0.688 pp | 20.754% | -0.293 pp |
| Equal-month mean | 50.662% | -0.472 pp | 19.456% | -0.183 pp |

### Top 20% within each calendar month

| Month | Precision | Precision difference | Recall | Recall difference |
|---|---:|---:|---:|---:|
| 2024-05 | 46.257% | -0.306 pp | 35.026% | -0.232 pp |
| 2024-06 | 43.011% | -0.444 pp | 34.495% | -0.356 pp |
| 2024-07 | 50.590% | -0.581 pp | 34.156% | -0.392 pp |
| 2024-08 | 43.252% | -0.555 pp | 36.861% | -0.473 pp |
| Equal-month mean | 45.778% | -0.472 pp | 35.134% | -0.363 pp |

Worst monthly PR-AUC change: **-0.005936560** in **2024-08**. Mean recall constraints do not guarantee that every month improves; no extra monthly elimination rule was added.
Smallest monthly recall change at 10%: **-0.293 pp** in **2024-08**.
Smallest monthly recall change at 20%: **-0.473 pp** in **2024-08**.

## Decision and verification

**Study recommendation: retain baseline.** Production remains the accepted LR at 0.39 even if an experimental candidate qualifies. No September-December 2024 evaluation was used for selection, and no later untouched evaluation period is available.

- The saved accepted preprocessing, 50 input features / 784 prepared dimensions, full January 2023-April 2024 training population, May-August 2024 validation population, remaining estimator settings, and two-core/768 MiB driver/640 MiB executor runtime were retained. No window experiment, combined settings, additional candidate, preprocessing fit, or production publication was executed.
- Each complete candidate passed saved-model parameter/dimension checks, full-validation score multiset equality across serialization/reload, and saved prediction population/threshold validation. Metrics are unweighted, natural-prevalence PR-AUC using the preregistered evaluator and per-month top-10%/20% capacities. Score/ID boundary ties retain multiplicity and receive equal fractional mass.
- Direct comparison of saved classifier metadata confirmed identical default parameters for every complete candidate. Candidate 1 has no classifier parameter-map changes (the approved class weight is applied to training rows); candidate 2 changes only regParam from 0.1 to 0.3.
- Fifteen synthetic metric/approval/output-boundary/budget tests passed before fitting. The candidate runner was committed before execution; exact commits, commands, exit codes, timings, and limits appear in the [budget ledger](evidence/budget.json).
- [Before](evidence/candidate-preservation-before-candidates.json) and [after](evidence/candidate-preservation-after-candidates.json) checks confirmed all 700 protected files unchanged, including raw data, baseline/current artifacts, feature datasets, and B1. Source/runtime acceptance identities, existing tags, and the main cleanup commit remain unchanged.
- Services were stopped afterward. Generated models and prediction Parquet remain separately under the ignored study artifact directory; the original study completion report predates this GitHub review export. Study source and compact results are now included here; binary artifacts remain local.
- [Full-precision compact results](candidate-results-summary.json) and [raw execution results](evidence/candidate-results.json) retain every monthly value and comparison. Display rounding in this report does not enter qualification calculations. The original [diagnostic report](REPORT.md) describes the pre-candidate stage.

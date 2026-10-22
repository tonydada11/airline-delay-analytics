# Controlled GBT study — Steps 3 and 4

**COMPLETE — DOES NOT QUALIFY.** The single approved `gbt-feature-subset-all`
attempt completed all required checks and four validation months. Its equal-month
PR-AUC was **0.4182170318313452**, below accepted LR's **0.4280983226531411**;
the mean paired gain was **-0.009881290821795952**, failing the required +0.005.
Both mean budget-recall guardrails passed. This is a completed performance result,
not an incomplete attempt.

Production remains **accepted LR at 0.39**, bundle
`d0098a9572834c53bf76b04d732a94a6`, selection
`production-5936316dd7764ed79b049ae10ee1ca18`.
**Step 5 awaits separate review.** No promotion, next-experiment decision, or main
merge is authorized by this report.

## Execution and preserved contract

Execution used the clean, committed `study/gbt-improvement-v1` tree at reviewed
commit `9e73cc43201a1b0369e2943cc97a96cf9a8378c3`. HEAD matched the reviewed
commit, including the elapsed-duration timeout correction; there were no newer
runner changes or unrelated worktree changes. The intended remote is `github`
(`git@github.com:tonydada11/airline-spark.git`); `origin` is Wisconsin GitLab.
The exclusive `attempt-001/` path was absent before launch.

Exactly this command was executed once from the repository root:

```sh
python3 -B studies/gbt_v1/launch.py --execute
```

**Launcher exit 0; worker exit 0; classifier fits started: 1.** The supervisor's
[completion.json](evidence/completion.json) is authoritative and byte-identical
to the final [attempt.json](evidence/attempt.json). The original worker
[results.json](evidence/results.json) intentionally still says
`pending_termination`; no generated record was rewritten.

| Timing | Recorded value |
| --- | --- |
| Supervisor start (UTC) | 2026-10-22T02:10:34.565366+00:00 |
| Supervisor finish (UTC) | 2026-10-22T02:45:21.812753+00:00 |
| Local execution date | 2026-10-21, America/Chicago |
| Classifier fit seconds | 1633.7367135810055 |
| Complete attempt seconds, including preparation/cleanup | 2087.4594736050058 (34 min 47.459 s) |
| Charged launcher/source preparation seconds | 0.21062753700243775 |
| Hard ceiling / final termination allowance | 14,400 / 60 seconds; unchanged; no timeout or retry |

The sole estimator change was `featureSubsetStrategy: "sqrt" -> "all"`.
The saved classifier parameter map equals a copy of source `GBT_SETTINGS` with
only that change. Effective defaults also match the frozen existing GBT after
this substitution. Preserved settings include 10 trees, depth 4, maxBins 32,
stepSize 0.1, maxMemoryInMB 128, seed 42, intrinsic subsamplingRate 0.5,
positive fitting weight 2.0 and negative weight 1.0. The saved model is Spark
GBTClassificationModel with 784 features; the runner verified two classes and
10 trees before and after reload. Complete settings/defaults and input
fingerprints are in [inputs.json](evidence/inputs.json).

Accepted preprocessing `PipelineModel_196025fd5a4b` was loaded without fitting.
The runner validated and retained all **8,946,370 training rows** from
2023-01-01–2024-04-30, then evaluated all **2,418,739 validation rows** from
2024-05-01–2024-08-31. Dataset schema 3, feature version `m7-flighttime-v2`,
50 ordered raw predictors, duplicate multiplicity, partition policy (36 prepared
training partitions), and `DISK_ONLY` persistence were preserved. Input manifests
initially supplied these sizes as claims; the completed runner's population
checks validated agreement before fitting. No train-plus-validation fit,
preprocessing fit, baseline refit/scoring, or test-period scoring/analysis occurred.

The existing pinned image was
`sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf`,
Spark 3.5.9 / Python 3.10.12 / x86_64, Docker context `default`.
Runtime master came from the existing session helper. Resource checks passed:
two total cores, one core per executor, 640m executor / 768m driver memory,
shuffle partitions 4, file partition limit 67,108,864 bytes, cache batch size
1,000. No builds, pulls, installs, Docker-environment changes, or resource changes
were made. Spark application: `app-20261022021038-0000`.

## Qualification from saved full-precision values

The unchanged `studies.lr_v1.metrics.monthly_comparison` helper was called on
saved May–August objects. It exactly reproduced the supervisor comparison.
The frozen LR diagnostics SHA-256 is
`c1313197e55e9d9d90793ec3401c3c0d3d61597ca35da94eae5c1f7b8fdee828`.
Its four saved PR-AUC values recompute to **0.4280983226531411**; adding 0.005
gives **0.4330983226531411**. Decisions use paired monthly differences, not
rounded table cells or pooled PR-AUC. Full precision is retained in
[comparison.json](evidence/comparison.json) and the original monthly evidence.

| Criterion | Actual | Result |
| --- | --- | --- |
| Mean paired monthly PR-AUC gain >= 0.005 | -0.009881290821795952 | FAIL |
| Mean recall change at monthly top 10% >= -0.01 | -0.004475281336162512 | PASS |
| Mean recall change at monthly top 20% >= -0.01 | -0.007217435689373511 | PASS |
| All four months, required checks, successful termination | 4/4; all checks true; exit 0 | PASS |

Metrics use unweighted natural-prevalence evaluation and the existing Spark
`areaUnderPR`/`areaUnderROC` evaluators with `numBins=0`. The pooled overall PR-AUC
below is descriptive only. It is not substituted for the qualification measure.
Tables use nine decimal places for readability; JSON is authoritative.

### Ranking by month

| Month | LR PR-AUC | Candidate PR-AUC | PR-AUC change | Candidate ROC-AUC | ROC-AUC change |
| --- | --- | --- | --- | --- | --- |
| 2024-05 | 0.432454907 | 0.421048095 | -0.011406812 | 0.680875487 | -0.005549344 |
| 2024-06 | 0.403536552 | 0.397617478 | -0.005919074 | 0.681984243 | -0.001163474 |
| 2024-07 | 0.470587237 | 0.463796062 | -0.006791174 | 0.690187645 | -0.000171321 |
| 2024-08 | 0.405814595 | 0.390406492 | -0.015408103 | 0.686759234 | -0.009866887 |
| Equal-month mean | 0.428098323 | 0.418217032 | -0.009881291 | 0.684951652 | -0.004187756 |

### Candidate operating threshold

The complete 101-step inclusive threshold grid selected **35/100 = 0.35** on the
entire validation population, independently of LR's 0.39. Standard-library
`Fraction` arithmetic confirmed step 35 is the unique maximum exact-F1 choice;
the existing lower-step tie rule was preserved. There are no per-month thresholds.
[thresholds.json](evidence/thresholds.json) contains all counts and the selected row.

| Overall validation metric at 0.35 | Full-precision result |
| --- | --- |
| Rows | 2418739 |
| precision | 0.36529815183579445 |
| recall | 0.712703632548291 |
| f1 | 0.483022057195013 |
| roc auc | 0.6857396506389911 |
| pr auc | 0.417298965477465 |
| prevalence | 0.26125100723972283 |
| predicted positive rate | 0.5097056772144494 |

| Population at 0.35 | TP | FP | FN | TN |
| --- | --- | --- | --- | --- |
| 2024-05 | 107048 | 178556 | 51228 | 262394 |
| 2024-06 | 108957 | 204663 | 40988 | 246664 |
| 2024-07 | 138384 | 211549 | 43481 | 220524 |
| 2024-08 | 95967 | 187721 | 45845 | 274770 |
| Overall | 450356 | 782489 | 181542 | 1004352 |

| Month | Rows | Precision | Recall | F1 | Prevalence | Alert rate |
| --- | --- | --- | --- | --- | --- | --- |
| 2024-05 | 599226 | 0.374812678 | 0.676337537 | 0.482328557 | 0.264134066 | 0.476621508 |
| 2024-06 | 601272 | 0.347417257 | 0.726646437 | 0.470082944 | 0.249379648 | 0.521594220 |
| 2024-07 | 613938 | 0.395458559 | 0.760916064 | 0.520438212 | 0.296226981 | 0.569981008 |
| 2024-08 | 604303 | 0.338283607 | 0.676719883 | 0.451078731 | 0.234670356 | 0.469446619 |
| Equal-month mean | — | 0.363993025 | 0.710154980 | 0.480982111 | 0.261102763 | 0.509410839 |

Changes below compare candidate 0.35 with the frozen LR 0.39 operating point.

| Month | Precision change | Recall change | F1 change | Alert-rate change |
| --- | --- | --- | --- | --- |
| 2024-05 | -0.040057394 | +0.132749122 | +0.011742574 | +0.130536726 |
| 2024-06 | +0.004694646 | -0.007789523 | +0.002727782 | -0.012812837 |
| 2024-07 | +0.007595915 | -0.014120364 | +0.003441190 | -0.021946842 |
| 2024-08 | -0.021339444 | +0.041914648 | -0.008060654 | +0.055207404 |
| Equal-month mean | -0.012276569 | +0.038188471 | +0.002462723 | +0.037746113 |

All four monthly prevalence changes are exactly zero.

### Monthly capacity budgets

The existing `budget_metrics`/`capacity` definitions were retained: capacity is
`rows * percent // 100`, score descending then flight ID ascending; labels never
break ties. Physical duplicates retain multiplicity and indistinguishable boundary
groups receive equal fractional mass. Every saved month reports zero identical
score/ID groups, zero duplicate excess rows, and zero fractional boundary groups.
These capacity metrics are independent of the selected operating threshold.

**Top 10%**

| Month | Capacity | Precision | Precision change | Recall | Recall change |
| --- | --- | --- | --- | --- | --- |
| 2024-05 | 59922 | 0.506107940 | -0.010146524 | 0.191608330 | -0.003841391 |
| 2024-06 | 60127 | 0.466712126 | -0.008947727 | 0.187148621 | -0.003587982 |
| 2024-07 | 61393 | 0.548417572 | -0.011092470 | 0.185131829 | -0.003744536 |
| 2024-08 | 60430 | 0.478139997 | -0.015786861 | 0.203748625 | -0.006727216 |
| Equal-month mean | — | 0.499844409 | -0.011493395 | 0.191909351 | -0.004475281 |

**Top 20%**

| Month | Capacity | Precision | Precision change | Recall | Recall change |
| --- | --- | --- | --- | --- | --- |
| 2024-05 | 119845 | 0.453352247 | -0.012282532 | 0.343273775 | -0.009300210 |
| 2024-06 | 120254 | 0.426081461 | -0.008473731 | 0.341711961 | -0.006795825 |
| 2024-07 | 122787 | 0.509801526 | -0.001905739 | 0.344194870 | -0.001286669 |
| 2024-08 | 120860 | 0.424590435 | -0.013478405 | 0.361859363 | -0.011487039 |
| Equal-month mean | — | 0.453456417 | -0.009035102 | 0.347759992 | -0.007217436 |

### Worst monthly changes

| Metric | Worst monthly change | Month |
| --- | --- | --- |
| pr auc | -0.015408103178153498 | 2024-08 |
| roc auc | -0.00986688663081714 | 2024-08 |
| precision | -0.04005739362373134 | 2024-05 |
| recall | -0.014120364006268371 | 2024-07 |
| f1 | -0.008060653699581644 | 2024-08 |
| prevalence | 0.0 | 2024-05, 2024-06, 2024-07, 2024-08 |
| predicted positive rate | -0.021946841537744954 | 2024-07 |
| top 10% precision | -0.01578686083071318 | 2024-08 |
| top 10% recall | -0.006727216314557305 | 2024-08 |
| top 20% precision | -0.013478404765844776 | 2024-08 |
| top 20% recall | -0.011487039178630865 | 2024-08 |

The top-20% recall change in August is below -0.01, but the preregistered
recall gates apply to the equal-month means. No additional worst-month exclusion
rule was applied. PR-AUC was below LR in all four months.

## Artifact verification, preservation, and cleanup

All five required worker checks are true: parameter/default/dimension agreement;
full-validation score multiset equality before/after GBT save/reload; saved
prediction schema, lineage/population/label multiset and threshold decisions;
all four months; and unchanged inputs. These are checks performed by the completed
runner. Step 4 verified saved records and bytes without repeating Spark scoring.

Independent read-only verification passed **68 assertions**: completion/attempt
identity, worker/launcher exits, one fit, reviewed source/digests, all monthly
records equal their aggregate, exact-F1 threshold/count agreement, capacity
arithmetic, paired deltas and means/worst changes, runtime/settings, artifact
hashes, input preservation, and process/service cleanup. Saved classifier inventory:
6 files / 25,338 bytes; saved prediction inventory: 12 files / 181,188,824 bytes
(inventory definitions omit dot sidecars). Every recorded artifact digest matches
the local file bytes and prediction `_SUCCESS` is present. Worker results SHA-256:
`89a66fae1b905395a8a5f2c88d0613c741fc8a46ff134133a9130e1fb996a007`.

Before report edits, all **92 protected small-file SHA-256 values** matched
preflight, including accepted preprocessing/classifier, manifests, selection
records, frozen monthly baseline, runtime/configuration and reviewed study source.
All **47 selection Parquet parts** retained their exact preflight byte sizes and
mtime metadata. The runner independently hashed the full selection inventory
before and after evaluation and matched the frozen accepted inventory. No raw
warehouse scan or additional baseline evaluation was added for reporting.
An additional targeted check reconciled all **160 executable model/bundle/selection
artifact files** with the existing LR-study preservation inventory (184,286 declared
bytes); every SHA-256 matched. The reviewed PLAN digest in the generated attempt
describes the pre-execution plan; this delivery adds only its execution status/link.

All four existing project services were stopped initially; no other containers
were running. Only these were started with the existing images:

```sh
docker compose start --wait spark-master spark-worker-1 spark-worker-2 spark-notebook
```

After worker exit and successful Spark shutdown, PID/start-tick checks found the
supervisor and worker gone. The Java child was a terminated zombie, with zero
live owned processes; no study executors remained in the four containers.
Container-staged source was removed before completion. The four inspected
container IDs were then stopped with `docker stop --timeout 10` (exit 0), restoring
all services to stopped; Docker context/image IDs stayed unchanged and no
unrelated application was stopped. Full evidence is in
[verification.json](evidence/verification.json).

Verification commands included the exact launch command above, targeted JSON /
SHA-256 / filesystem calculations and the existing comparison helper, plus:

```sh
PYTHONPATH=. python3 -B /tmp/gbt-step4-verify.py
git diff --check
```

The temporary verification calculation exited 0 with 68 assertions; it is not a
new repository runner or reporting framework. Initial read-only calculations
stopped on AST extraction of a `dict(...)` expression and on an exact ROC-AUC
summary comparison. Keyword extraction resolved the first; reproducing Python
3.10 left-fold summation resolved the second. Host Python 3.12 sums the saved
monthly ROC-AUC values to 0.6849516522011272 versus the pinned runtime's
0.6849516522011271 (difference 1.1102230246251565e-16). The qualification helper's
full-precision result matches exactly; all original records remain unchanged.
No full pipeline/test suite, extra model fit, or additional Spark evaluation was
run for Step 4. Delivery checks and evidence-copy reconciliation passed.

## Evidence and original local locations

All paths below are relative to the repository root. Ten original JSON records
were copied byte-for-byte into [evidence/](evidence/), together with derived
[comparison.json](evidence/comparison.json),
[verification.json](evidence/verification.json), and
[provenance.json](evidence/provenance.json) recording source paths, sizes, and
SHA-256 digests.

- Attempt/supervisor records and large local log:
  `data/studies/gbt-improvement-v1/attempt-001/{attempt.json,completion.json,worker.log}`.
- Original candidate records (`status.json`, `inputs.json`, `results.json`,
  `thresholds.json`, `validation-2024-05.json` through `validation-2024-08.json`):
  `data/studies/gbt-improvement-v1/attempt-001/gbt-feature-subset-all/`.
  Published `worker-status.json` is the unmodified original `status.json`.
- Candidate classifier and validation Parquet stay ignored and local beneath
  that candidate directory: `classifier/`, `validation-predictions/`.
- Configured selection input remains
  `data/warehouse/dws/ml_flighttime_v3_selection/`; accepted fitted preprocessing
  and bundle remain `artifacts/flighttime-v3/models/`.
- Frozen comparison source remains
  `studies/lr_v1/evidence/diagnostics/results.json` and its four monthly JSON files;
  original local LR diagnostics remain `data/studies/lr-improvement-v1/diagnostics/`.

Only this report, compact JSON evidence, and the necessary plan status/link are
published. Models, predictions, warehouse files, large logs, LR-study evidence,
and production artifacts remain local/preserved. Steps 3 and 4 are complete;
**Step 5 awaits separate review**.

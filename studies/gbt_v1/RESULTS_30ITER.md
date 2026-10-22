# GBT all-features, 30 iterations — results

**COMPLETE — DOES NOT QUALIFY.** `gbt-all-30iter` completed one full-data
classifier fit, all four validation months, all required artifact/input checks,
and successful shutdown. Mean monthly PR-AUC was **0.4235378972669571**:
**-0.004560425386183925 versus accepted LR**, failing the required +0.005.
Both mean budget-recall guardrails against LR passed.

Against the completed ten-iteration GBT, mean paired PR-AUC improved by
**+0.005320865435612027**, with positive PR-AUC changes in every month.
That secondary comparison is descriptive and cannot replace qualification against
LR. This is a complete evaluated result, not an incomplete losing attempt.

Production remains **accepted LR at 0.39**, bundle
`d0098a9572834c53bf76b04d732a94a6`, selection
`production-5936316dd7764ed79b049ae10ee1ca18`. The new results await separate
review. No promotion, main merge, next-experiment selection, or further tuning
was performed.

## Implementation, execution, and references

Implementation/plan commit and executed source:
**`27c832f3b093b901fa20a9210258d50922d95a13`**, a child of the previously reviewed
results commit `8bb83faa9b3507e9c57ad6c1094c0cd410faec9e`, on
`study/gbt-improvement-v1`. Initial status was clean; no conflicting newer commits
were present. The implementation was tested and committed before execution;
launch used a clean committed tree. No reset or history rewrite occurred.

[PLAN_30ITER.md](PLAN_30ITER.md) is frozen at the executed source commit.
The shared [runner](run_candidate.py) and [launcher](launch.py) gained only explicit
support for two fixed selections; [focused tests](test_candidate.py) reuse existing
fixtures and protections. The original command still selects
`gbt-feature-subset-all` / `attempt-001`, refusing that occupied path.
The new selection maps exclusively to `attempt-002` and records the second plan's
digest. Container staging includes that plan and four pinned compact/local
records from the completed reference.

Exactly this new full-data command was executed once from the repository root:

```sh
python3 -B studies/gbt_v1/launch.py --execute --candidate gbt-all-30iter
```

**Launcher exit 0; worker exit 0; full-data classifier fits started: 1.**
`attempt-002/` was absent before launch and was created only by the supervisor.
No overwrite, resume, replacement attempt, retry, direct worker launch, reduced
evaluation, or budget extension occurred. Authoritative
[completion.json](evidence_30iter/completion.json) matches final
[attempt.json](evidence_30iter/attempt.json) byte-for-byte. Original worker
[results.json](evidence_30iter/results.json) intentionally remains
`pending_termination`; no generated record was rewritten.

The new classifier was fitted **from scratch**, changing only **maxIter 10 -> 30
relative to the completed `featureSubsetStrategy="all"` reference**.
Relative to production `GBT_SETTINGS`, **two** fields differ:
`featureSubsetStrategy="all"` and `maxIter=30`. No saved model was extended.
Explicit/effective settings differ from the ten-iteration reference only in
maxIter. Preserve depth 4, stepSize 0.1, maxBins 32, intrinsic subsamplingRate 0.5,
maxMemoryInMB 128, seed 42, positive/negative fitting weights 2.0/1.0 and every
other default. Production settings were not mutated, and no early stopping or
validation indicator was introduced. Full settings are in
[inputs.json](evidence_30iter/inputs.json).

Both references were read from saved full-precision validation records:

- Primary LR: [frozen diagnostics](../lr_v1/evidence/diagnostics/results.json),
  threshold 0.39, SHA-256
  `c1313197e55e9d9d90793ec3401c3c0d3d61597ca35da94eae5c1f7b8fdee828`.
- Secondary GBT: [ten-iteration results](evidence/results.json) and
  [completion](evidence/completion.json), threshold 0.35, results SHA-256
  `89a66fae1b905395a8a5f2c88d0613c741fc8a46ff134133a9130e1fb996a007`,
  original execution revision `9e73cc43201a1b0369e2943cc97a96cf9a8378c3`.

No reference was retrained or rescored. Accepted preprocessing
`PipelineModel_196025fd5a4b` was loaded without refitting. All **8,946,370**
training rows (2023-01-01–2024-04-30), **2,418,739** validation rows
(2024-05-01–2024-08-31), feature version `m7-flighttime-v2`, schema 3, 50 ordered
predictors, 784 prepared dimensions, duplicate multiplicity, DISK_ONLY persistence,
and the 36-partition prepared-training policy were preserved and checked by the
runner. No extra sampling, train-plus-validation fit, feature rebuild, baseline
refit/scoring, or test-period scoring/analysis occurred.

The existing pinned native image was
`sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf`,
Spark 3.5.9 / Python 3.10.12 / x86_64, Docker context `default`.
Master came from the existing session helper. Resources stayed at two total
cores, one core per executor, 640m executor / 768m driver memory, shuffle
partitions 4, file partition limit 67,108,864 bytes, cached batch size 1,000.
No images were built/pulled, dependencies installed, environments switched,
resources changed, or production validators modified.

| Timing / allowance | Ten-iteration reference | New 30-iteration candidate |
| --- | --- | --- |
| Fit seconds | 1633.7367135810055 | 4790.826843599993 |
| Complete attempt seconds | 2087.4594736050058 | 5280.869890715025 |
| New supervisor start, UTC | — | 2026-10-22T04:38:16.927623+00:00 |
| New supervisor finish, UTC | — | 2026-10-22T06:06:17.551728+00:00 |
| Hard cap / final termination allowance | 14,400 / 60 seconds | 14,400 / 60 seconds |

New fit duration: **79 min 50.827 s**; complete attempt: **88 min 0.870 s**,
including charged preparation, source staging/cleanup and shutdown. Launcher
preparation was 0.2444834610214457 s. Only durations crossed clock boundaries;
container process-group supervision and the final 60-second allowance were kept.
Runtime values are observations, not qualification rules.

## Qualification and monthly ranking

The existing `monthly_comparison` helper was rerun on saved JSON in pinned Python
3.10.12 without Spark. It reproduced the supervisor comparison exactly.
[comparison.json](evidence_30iter/comparison.json) retains full-precision inputs,
paired changes, equal-month means and worst changes against both references.
LR's four saved PR-AUC values recompute to **0.4280983226531411**; the +0.005
target is **0.4330983226531411**. Decisions use paired monthly changes, never
rounded Markdown cells, pooled PR-AUC or another library's average precision.

| Qualification criterion against LR | Actual | Result |
| --- | --- | --- |
| Mean paired monthly PR-AUC gain >= 0.005 | -0.004560425386183925 | FAIL |
| Mean top-10% recall change >= -0.01 | -0.0018174130069710526 | PASS |
| Mean top-20% recall change >= -0.01 | -0.0015157072582849263 | PASS |
| Complete execution, one fit, four months and required checks | All passed; successful termination | PASS |

PR-AUC was lower than LR in every month and higher than the ten-iteration GBT in
every month. No secondary qualification flag or extra worst-month rule was applied.

| Month | LR PR-AUC | GBT10 PR-AUC | GBT30 PR-AUC | Change vs LR | Change vs GBT10 | GBT30 ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 2024-05 | 0.432454907 | 0.421048095 | 0.424094552 | -0.008360354 | +0.003046458 | 0.684562852 |
| 2024-06 | 0.403536552 | 0.397617478 | 0.399533583 | -0.004002969 | +0.001916105 | 0.685457729 |
| 2024-07 | 0.470587237 | 0.463796062 | 0.470524123 | -0.000063114 | +0.006728061 | 0.695559247 |
| 2024-08 | 0.405814595 | 0.390406492 | 0.399999331 | -0.005815265 | +0.009592839 | 0.694347801 |
| Equal-month mean | 0.428098323 | 0.418217032 | 0.423537897 | -0.004560425 | +0.005320865 | 0.689981907 |

All evaluation is unweighted at natural prevalence. The existing Spark
areaUnderPR/areaUnderROC evaluators with `numBins=0` are unchanged. Tables below
round to nine decimal places for readability; JSON values determine decisions.

## Selected threshold and operating metrics

The full-validation 101-step inclusive grid independently selected **36/100 =
0.36**. Standard-library exact rational F1 confirms step 36 is the unique winner
(`446419/920109`); the existing lower-step tie rule is preserved. Neither LR's
0.39 nor GBT10's 0.35 was inherited. One threshold is used across all months.
[thresholds.json](evidence_30iter/thresholds.json) contains every step's counts.

| Overall May–August metric at 0.36 | Full-precision result |
| --- | --- |
| rows | 2418739 |
| precision | 0.3694542836334746 |
| recall | 0.7064731966235057 |
| f1 | 0.4851805601292891 |
| roc auc | 0.6902277934065765 |
| pr auc | 0.42240714654825096 |
| prevalence | 0.26125100723972283 |
| predicted positive rate | 0.49956609621790526 |

Overall confusion counts: TP **446419**, FP **761901**,
FN **185479**, TN **1024940**. Pooled overall PR-AUC
is descriptive and does not replace monthly qualification.

| Month at 0.36 | Rows | Precision | Recall | F1 | Prevalence | Alert rate |
| --- | --- | --- | --- | --- | --- | --- |
| 2024-05 | 599226 | 0.384777307 | 0.647899871 | 0.482817230 | 0.264134066 | 0.444757070 |
| 2024-06 | 601272 | 0.344710141 | 0.741125079 | 0.470556353 | 0.249379648 | 0.536164997 |
| 2024-07 | 613938 | 0.398941591 | 0.757726885 | 0.522688696 | 0.296226981 | 0.562636618 |
| 2024-08 | 604303 | 0.346490028 | 0.669477900 | 0.456643187 | 0.234670356 | 0.453423200 |
| Equal-month mean | — | 0.368729766 | 0.704057434 | 0.483176367 | 0.261102763 | 0.499245471 |

## Fixed monthly capacity budgets

The existing `budget_metrics`/`capacity` definitions remain: `rows * percent // 100`,
score descending then flight ID ascending, with no label-based tie breaking.
Physical duplicates retain multiplicity; indistinguishable boundary groups receive
equal fractional selection mass. All saved months report zero identical score/ID
groups, zero duplicate excess rows and zero fractional boundary groups. Budget
metrics are independent of the operating threshold.

**Top 10%**

| Month | Capacity | Precision | P change vs LR | P change vs GBT10 | Recall | R change vs LR | R change vs GBT10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-05 | 59922 | 0.509095157 | -0.007159307 | +0.002987217 | 0.192739266 | -0.002710455 | +0.001130936 |
| 2024-06 | 60127 | 0.474645334 | -0.001014519 | +0.007933208 | 0.190329788 | -0.000406816 | +0.003181166 |
| 2024-07 | 61393 | 0.557946346 | -0.001563696 | +0.009528774 | 0.188348500 | -0.000527864 | +0.003216672 |
| 2024-08 | 60430 | 0.485421148 | -0.008505709 | +0.007281152 | 0.206851324 | -0.003624517 | +0.003102699 |
| Equal-month mean | — | 0.506776996 | -0.004560808 | +0.006932588 | 0.194567219 | -0.001817413 | +0.002657868 |

**Top 20%**

| Month | Capacity | Precision | P change vs LR | P change vs GBT10 | Recall | R change vs LR | R change vs GBT10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-05 | 119845 | 0.459627018 | -0.006007760 | +0.006274772 | 0.348024969 | -0.004549016 | +0.004751194 |
| 2024-06 | 120254 | 0.433598882 | -0.000956309 | +0.007517421 | 0.347740838 | -0.000766948 | +0.006028877 |
| 2024-07 | 122787 | 0.517066139 | +0.005358873 | +0.007264613 | 0.349099607 | +0.003618068 | +0.004904737 |
| 2024-08 | 120860 | 0.432947212 | -0.005121628 | +0.008356776 | 0.368981468 | -0.004364934 | +0.007122105 |
| Equal-month mean | — | 0.460809813 | -0.001681706 | +0.007353396 | 0.353461721 | -0.001515707 | +0.005701728 |

## Worst monthly changes

| Metric | Worst vs LR | Month | Worst vs GBT10 | Month |
| --- | --- | --- | --- | --- |
| pr auc | -0.00836035427194487 | 2024-05 | 0.0019161047943949128 | 2024-06 |
| roc auc | -0.0022783190815460763 | 2024-08 | 0.003473485932058984 | 2024-06 |
| precision | -0.030092764649653825 | 2024-05 | -0.002707115446185071 | 2024-06 |
| recall | -0.017309542792730892 | 2024-07 | -0.02843766584952867 | 2024-05 |
| f1 | -0.002496197817904555 | 2024-08 | 0.0004734088879171372 | 2024-06 |
| prevalence | 0.0 | 2024-05, 2024-06, 2024-07, 2024-08 | 0.0 | 2024-05, 2024-06, 2024-07, 2024-08 |
| predicted positive rate | -0.029291231362124526 | 2024-07 | -0.03186443845894538 | 2024-05 |
| top 10% precision | -0.008505709084891566 | 2024-08 | 0.002987216715062968 | 2024-05 |
| top 10% recall | -0.003624516966124175 | 2024-08 | 0.0011309358336071273 | 2024-05 |
| top 20% precision | -0.0060077600233635065 | 2024-05 | 0.006274771579957428 | 2024-05 |
| top 20% recall | -0.004549015643559362 | 2024-05 | 0.004751194116606461 | 2024-05 |

Worst changes are descriptive. Recall qualification uses only the two equal-month
mean changes against LR. Full monthly operating-metric changes and equal-month
changes against both references are also retained in the comparison JSON.

## Verification, preservation, and cleanup

All five required runner checks passed: parameters/defaults/dimensions; complete
validation score multiset equality before/after GBT serialization/reload; saved
prediction schema, lineage/population/label multiset and threshold decisions;
all months; and unchanged inputs. The saved GBT has **30 trees, two classes and
784 features**. The runner performed these distributed checks; Phase C verified
saved records/bytes without additional Spark evaluations or baseline scoring.

Independent read-only verification passed **76 assertions**, including
completion/worker/exit status, executed source and correct plan digests, one fit,
parameter/default agreement, both reference identities, all monthly records,
exact-F1 counts and selection, population sums, budget arithmetic, monthly/mean/
worst changes, and artifact/preservation/shutdown checks. All recorded artifact
digests match: classifier inventory **6 files / 58,865 bytes**; predictions
**12 files / 193,949,228 bytes**, including `_SUCCESS` (dot sidecars excluded by
the existing inventory definition). Worker results SHA-256:
`6c02d060a8b974e89d59ac210410bb048c74720f5212af76918191162675718e`.
See [verification.json](evidence_30iter/verification.json).

All **70 protected original study/attempt/reference files** retain their preflight
SHA-256 values, and attempt-001's file set is unchanged. This includes original
PLAN.md, RESULTS.md, evidence/, generated attempt-001 records, classifier,
predictions, logs and source provenance. All **160 accepted executable model /
bundle / selection files** match both preflight and the existing historical
preservation inventory. All **47 selection Parquet parts** retain their byte sizes
and mtimes; the runner's full before/after selection fingerprints match the frozen
inventory and ten-iteration reference. No additional whole-warehouse scan or
baseline evaluation was added. Evidence: [preservation.json](evidence_30iter/preservation.json).

All four project containers were initially stopped, with no other containers
running. Only these existing services were started for Phase A and the study:

```sh
docker compose start --wait spark-master spark-worker-1 spark-worker-2 spark-notebook
```

After successful shutdown, Spark master reported application
`app-20261022043820-0001` **FINISHED**, no active applications or study executors.
PID/start-tick checks found the supervisor and worker gone, Java terminated as a
zombie, and zero live owned processes. The supervisor removed its staged source
before completion. Task-owned fixture/calculation staging was removed afterward;
all four inspected container IDs were restored to stopped with
`docker stop --timeout 10` (exit 0). No unrelated application was stopped.

## Tests and delivery checks

Phase A used only generated fixtures and mocks in the pinned environment:

```sh
docker compose exec -T -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/tmp/gbt30-fixtures/source/project:/workspace/src:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip \
  -w /workspace spark-notebook python3 -B -m pytest -q -p no:cacheprovider \
  /tmp/gbt30-fixtures/source/project/studies/gbt_v1/test_candidate.py \
  /tmp/gbt30-fixtures/source/project/studies/lr_v1/test_metrics.py \
  /workspace/tests/unit/test_experiment_selection.py \
  /workspace/tests/unit/test_model_contract.py
```

**70 passed, 1 pre-existing PySpark distutils deprecation warning, 94.36 seconds,
exit 0.** Two tiny generated-data GBT fits (10 and 30 trees) tested API / save-reload
wiring; they did not evaluate the study hypothesis or read real warehouse inputs.
Full-data attempt-002 remained absent through testing. Existing original-candidate,
threshold, metric, capacity/tie, population/multiplicity, incomplete/timeout,
process-group and deliberately incompatible clock-origin protections passed for
the affected paths. [tests.json](evidence_30iter/tests.json) records the exact
command/result. No full pipeline or broad test suite was run.

After execution, the temporary JSON-only comparison calculation ran with
`docker compose exec -T` in the same pinned Python, importing the unchanged
`studies.lr_v1.metrics.monthly_comparison`; exit 0. The host command
`python3 -B /tmp/gbt30-verify.py` performed the 76 read-only checks; exit 0.
Temporary calculations write only derived evidence, not study originals.
`git diff --check`, staged scope/content inspection, JSON/provenance digest and
original-copy checks passed. Source implementation/tests and the frozen second
plan are unchanged after the execution commit. Results/evidence are committed
separately from implementation.

## Original local artifacts and review evidence

Project-relative original root:
`data/studies/gbt-improvement-v1/attempt-002/`; authoritative `completion.json`,
final `attempt.json` and large `worker.log` are at that root. Candidate originals
are under `gbt-all-30iter/`: `status.json`, `inputs.json`, `results.json`,
`thresholds.json`, and `validation-2024-05.json` through `validation-2024-08.json`.
Classifier and prediction Parquet remain local/ignored in that candidate's
`classifier/` and `validation-predictions/` directories.

[evidence_30iter/](evidence_30iter/) contains ten unmodified original JSON copies,
plus derived comparisons, verification, preservation, test results and
[provenance.json](evidence_30iter/provenance.json) with source paths, byte sizes,
digests, executed revision and reference mappings. Published `worker-status.json`
is the original `status.json`. Temporary container paths in original records
refer to staged source that was cleaned up; permanent source is the execution
commit, and compact reference originals remain in their existing study paths.
The original report/evidence and attempt-001 are preserved unchanged.

Only study source/plan/tests and this separate report/compact evidence are
published. Production remains **LR at 0.39**. The new results await separate review;
no further experiment or tuning was selected.

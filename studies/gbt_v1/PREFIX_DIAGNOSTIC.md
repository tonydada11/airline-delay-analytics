# Saved GBT30 prefix diagnostic

**COMPLETE. Validation PR-AUC continues improving through tree 30 in every
month.** The equal-month gain from trees 25→30 is **+0.0012471695554793216**.
That five-tree increment is smaller than the 20→25 gain
(**+0.0014511095078957037**). Monthly behavior varies: the last increment grows
in May and August, and shrinks in June and July. No month reverses over 25→30;
the observed mean gain slows without flattening to zero. This saved ensemble
provides no measurement or promised improvement beyond tree 30.

Production remains **accepted LR at threshold 0.39**, bundle
`d0098a9572834c53bf76b04d732a94a6`. The next experiment awaits separate review.
No fitting, preprocessing refit, feature change, threshold tuning, test-period
access, promotion, next-experiment selection or merge occurred.

## Monthly PR-AUC

All **2,418,739** May–August 2024 validation rows, including **631,898** positives,
were evaluated at natural prevalence. Tables round to nine decimals; the
[original results JSON](evidence_prefix/results.json) retains full precision.

| Month | Frozen LR | GBT30 prefix 10 | Prefix 20 | Prefix 25 | Prefix 30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024-05 | 0.432454907 | 0.417791577 | 0.421852321 | 0.422916220 | 0.424094552 |
| 2024-06 | 0.403536552 | 0.393898766 | 0.397170151 | 0.398949022 | 0.399533583 |
| 2024-07 | 0.470587237 | 0.464435461 | 0.468348784 | 0.470098745 | 0.470524123 |
| 2024-08 | 0.405814595 | 0.388942662 | 0.395987216 | 0.397198924 | 0.399999331 |
| Equal-month mean | 0.428098323 | 0.416267116 | 0.420839618 | 0.422290728 | 0.423537897 |

The equal-month prefix means are **0.4162671164569616**, **0.42083961820358085**,
**0.4222907277114765**, and **0.42353789726695584**, respectively. Frozen LR's
mean is **0.4280983226531411**. The complete 30-tree model remains below LR in
all four months; this diagnostic does not change its previous qualification
outcome.

The **10-tree prefix belongs to the saved GBT30 fit**. The separately trained
GBT10 experiment has a different saved result (mean **0.4182170318313452**);
it was not rescored or used as this diagnostic's ten-tree prefix.

## Paired changes

| Month | 10→20 (10 trees) | 20→25 (5 trees) | 25→30 (5 trees) |
| --- | ---: | ---: | ---: |
| 2024-05 | +0.004060745 | +0.001063898 | +0.001178332 |
| 2024-06 | +0.003271385 | +0.001778871 | +0.000584561 |
| 2024-07 | +0.003913323 | +0.001749961 | +0.000425378 |
| 2024-08 | +0.007044554 | +0.001211708 | +0.002800406 |
| Equal-month mean | +0.004572502 | +0.001451110 | +0.001247170 |
| Worst monthly gain | +0.003271385 | +0.001063898 | +0.000425378 |

Average gain per additional tree is approximately 0.000457250 for 10→20,
0.000290222 for 20→25, and 0.000249434 for 25→30. The first interval contains
ten trees; the last two each contain five. August supplies the largest late
increment (+0.002800406480514772); July supplies the smallest
(+0.0004253780152612685). These differences describe this fixed validation
period and do not establish an underfitting diagnosis or future gains.

## Method and verification gates

The [study-local script](prefix_diagnostic.py) loads accepted fitted
`PipelineModel_196025fd5a4b` and the saved GBT30 classifier (784 prepared features,
two classes, 30 fitted trees). Each prefix uses the **first N original fitted
regression-tree objects in order and their exact original weights**: 1.0 for
the first tree and 0.1 for each subsequent tree. Weights are neither normalized
nor recomputed. NumPy binary64 weights are transferred as Python floats without
changing their values.

An in-memory native Spark `GBTClassificationModel` is constructed with those
arrays and original dimensions, then receives the original scoring parameters.
The diagnostic checks tree UIDs, weight arrays and dimensions. It leaves
`maxIter` metadata unchanged; changing that parameter on a fitted copy does
not remove fitted trees. No model is saved, modified or extended.

Native semantics were verified against the installed **Spark 3.5.9 JAR** with
`javap -c -p`: `GBTClassificationModel.margin` uses the weighted dot product,
`predictRaw` returns `[-margin, margin]`, and `LogLoss` /
`raw2probabilityInPlace` produce `p1 = 1 - 1 / (1 + exp(2 * margin))`.
The diagnostic uses native transform through `continuous_score`, rather than a
Python reimplementation on the full data. Generated-vector tests independently
check the weighted formula, preserve a duplicate, and forbid `Estimator.fit`.

PR-AUC uses the same **unweighted `BinaryClassificationEvaluator`** settings as
`airline.ml.metrics.calculate_metrics`: `areaUnderPR`, scalar positive-class
probability, and **`numBins=0`**. No average precision, pooled replacement,
threshold search or new qualification rule was introduced.

The numerical tolerances were fixed in the tested, committed code before launch:

- **Score absolute tolerance: 1e-12.** Maximum observed 30-prefix/native
  difference was **0.0** across all **2,418,739** rows; invalid scores: **0**.
- **Monthly PR-AUC absolute tolerance: 1e-10.** Reconciliation against the
  original saved GBT30 monthly values passed in every month. Maximum observed
  difference was **1.8318679906315083e-15**. The tiny integration differences
  occur despite identical scores under the distributed evaluator.
- Before interpreting increments, both checks must pass. Bidirectional
  `exceptAll` checks on full lineage and labels preserve row/label multiplicity
  after preprocessing and after all scoring transforms. No joins, deduplication,
  sampling or collected feature/prediction tables are used in the diagnostic.
- Existing feature validation checks schema, flight IDs, labels, histories and
  split/date agreement. Four monthly row/positive summaries agree with frozen
  LR: May 599226/158276; June 601272/149945; July 613938/181865;
  August 604303/141812. Only compact summaries are collected.

[Inputs](evidence_prefix/inputs.json) reconcile configured selection features,
completion/dataset manifests, source/feature contracts, accepted artifacts,
frozen LR records and original GBT30 completion/results/model inventory. All ten
validation Parquet files match their frozen SHA-256 values. The dataset manifest's
**11,365,109** selection rows are a manifest claim; this diagnostic independently
validates only the **2,418,739** validation rows. Training rows are not materialized
or scored; the test dataset is never opened.

## Execution, tests and provenance

Base: `d8314485d1e92fc85953c624675c7813a16ac5a0`, clean branch
`study/gbt-improvement-v1`. Tested implementation committed before execution:
**`b5077bfbdaffba4260ae385d275f4801435a3612`**. The old runner, both previous
attempts, plans, reports and evidence remain unchanged. New files are
[prefix_diagnostic.py](prefix_diagnostic.py), [launch_prefix.py](launch_prefix.py)
and [test_prefix_diagnostic.py](test_prefix_diagnostic.py).

The exact final focused test command was:

```sh
docker compose exec -T -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/tmp/gbt-prefix-fixtures/source/project:/workspace/src:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip \
  -w /workspace spark-notebook python3 -B -m pytest -q -p no:cacheprovider \
  /tmp/gbt-prefix-fixtures/source/project/studies/gbt_v1/test_prefix_diagnostic.py \
  /tmp/gbt-prefix-fixtures/source/project/studies/gbt_v1/test_candidate.py \
  -k 'prefix or timeout_interrupts or clock_origins or container_bootstrap'
```

**Exit 0: 19 passed, 39 deselected, 1 existing PySpark deprecation warning,
35.05 seconds.** The tests use generated vectors, read the saved classifier,
and perform **zero fits**. They never read warehouse rows. They also cover
exclusive paths, missing-month/tolerance gates, completion/failure checks,
import/help side effects, a short sleeping-process timeout and reused
cross-clock-origin protections. [tests.json](evidence_prefix/tests.json) records
the development runs, including the corrected pre-commit NumPy/Py4J binding
failure. No full pipeline or training suite was run.

Exactly one diagnostic was launched from the clean repository root:

```sh
python3 -B studies/gbt_v1/launch_prefix.py --execute
```

**Launcher exit 0; worker exit 0; fits 0; retries 0.** Exclusive output
`data/studies/gbt-improvement-v1/prefix-diagnostic-001/` was absent and created
only by the supervisor. Preparation, container source staging, distributed
scoring/evaluation, source cleanup and successful shutdown took
**211.2919645429938 seconds (3 min 31.292 s)**. The unchanged diagnostic ceiling
was **3,600 seconds**, including the final **60-second** termination allowance.
Only elapsed durations cross host/container clocks, and the actual Spark worker
process group is bounded inside the container.

Supervisor start/finish: **2026-10-22 12:52:05.787–12:55:36.875 America/Chicago**
(17:52:05.787–17:55:36.875 UTC). Host/container preparation charge:
**0.20236284099519253 seconds**.

Pinned image:
`sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf`,
Spark 3.5.9 / Python 3.10.12, Docker context `default`. Master was inherited from
the existing session helper. Resources remained two total cores, one core per
executor, 640m executor / 768m driver memory, shuffle partitions 4, file partition
limit 67,108,864 bytes, batch size 1,000. Reused frames used **DISK_ONLY**, and
validation retained the existing partition policy without repartitioning.
No build/pull, dependency installation, resource change or environment switch
occurred.

## Saved evidence and preservation

Original local inputs:

- Classifier: `data/studies/gbt-improvement-v1/attempt-002/gbt-all-30iter/classifier/`.
- Preprocessing: `artifacts/flighttime-v3/models/preprocessing/`.
- Features: `data/warehouse/dws/ml_flighttime_v3_selection/`, validation partition.
- Frozen LR: `studies/lr_v1/evidence/diagnostics/results.json`.
- Saved GBT30 results: `studies/gbt_v1/evidence_30iter/results.json`, SHA-256
  `6c02d060a8b974e89d59ac210410bb048c74720f5212af76918191162675718e`.

[completion.json](evidence_prefix/completion.json) is authoritative and matches
final [attempt.json](evidence_prefix/attempt.json) byte-for-byte. Worker
[status](evidence_prefix/worker-status.json) reports evaluated, successful
shutdown and zero fits. Original [results](evidence_prefix/results.json)
intentionally retains `pending_termination`; its recorded SHA-256 is
`a7b920368e41528e3eaeb494914c8809b5c8ead635d74c2158c1135993d4dcf9`.
All six published generated JSON files are verbatim copies. The local worker log
and all original large artifacts remain ignored; no model or predictions were
added to Git.

Saved-JSON verification in pinned Python, without Spark, passed **23 assertions**
covering authoritative completion, digests, all prefix means/paired changes,
weights/UID prefix alignment, zero fits, shutdown and tolerances. Additional
checks reconcile actual monthly populations and frozen LR/GBT30 values.
See [verification.json](evidence_prefix/verification.json) and
[provenance.json](evidence_prefix/provenance.json).

Targeted before/after SHA-256 checks found **all 246 protected files unchanged**,
including accepted artifacts, attempt-001, attempt-002, prior reports/evidence and
configuration. Both attempt file sets are unchanged. All **97 selection file
metadata records** are unchanged; the worker also rechecked validation Parquet
hashes and all accepted/reference identities after evaluation. No raw or
whole-warehouse hash scan was performed. [preservation.json](evidence_prefix/preservation.json)
contains the initial inventory and reconciliation.

The owned application `app-20261022175209-0003` reached **FINISHED**; the master
reported no active applications. No owned diagnostic worker/driver processes or
staged source directories remain. All four project services were initially
stopped; only those existing services were started and they were restored to
stopped afterward. Unrelated applications were left alone.

`git diff --check` and staged-diff scope checks passed. Delivery uses a separate
results commit and normal push to `github` → `tonydada11/airline-spark`, branch
`study/gbt-improvement-v1`, without rewriting history or merging to main.

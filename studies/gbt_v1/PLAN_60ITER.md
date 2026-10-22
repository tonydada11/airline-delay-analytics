# Controlled GBT60 experiment

User-authorized: implement, test and fit **one** `gbt-all-60iter` from scratch,
then compare with accepted LR (primary) and completed GBT30 (descriptive secondary).
Base `1b6d6e5174f87fcef1e82a0660edded2c6147d34`, initially clean on
`study/gbt-improvement-v1`; delivery remote `github` resolves to
`tonydada11/airline-spark`. Preserve previous attempts, reports and evidence.
No baseline refit/scoring, preprocessing fit, promotion or test evaluation.

## Fixed change and basis

Reference: `gbt-all-30iter`, `featureSubsetStrategy="all"`, `maxIter=30`.
Candidate: **`gbt-all-60iter`**, `featureSubsetStrategy="all"`, **`maxIter=60`**.
**Only maxIter changes relative to GBT30.** Relative to production `GBT_SETTINGS`,
two fields differ: featureSubsetStrategy and maxIter. Copy imported settings;
never mutate them. Do not extend the saved model or introduce early stopping.

```json
{
  "maxIter": 60, "maxDepth": 4, "maxBins": 32,
  "stepSize": 0.1, "subsamplingRate": 0.5,
  "featureSubsetStrategy": "all", "maxMemoryInMB": 128, "seed": 42,
  "featuresCol": "features", "labelCol": "is_arrival_delayed_15m",
  "weightCol": "class_weight"
}
```

Positive/negative fitting weights stay 2.0/1.0. Preserve every other effective
default, including logistic loss, variance impurity, minimum-node settings,
validationTol, cache/checkpoint settings and output columns. Reload checks require
**60 trees, two classes and 784 prepared features**.

[The saved GBT30 prefix diagnostic](PREFIX_DIAGNOSTIC.md) found a positive
25→30 PR-AUC increment in every validation month, mean +0.0012471695554793216,
with differing monthly gain sizes. This supports measuring a bounded capacity
increase; it does not demonstrate that 60 is optimal, diagnose underfitting, or
predict qualification. More boosting rounds may eventually worsen validation
performance. [Official Spark 3.5.9 GBTClassifier documentation](https://spark.apache.org/docs/3.5.9/api/python/reference/api/pyspark.ml.classification.GBTClassifier.html)
was verified: maxIter controls iteration count; library settings do not prove
improvement on this dataset. The existing research limits in [PLAN.md](PLAN.md)
remain applicable. No additional direction is selected.

## Inputs and references

Preserve schema 3, feature version `m7-flighttime-v2`, source
`bts-2023-2024-v1`, 50 ordered predictors, duplicates and original populations.
Selection-only input: `data/warehouse/dws/ml_flighttime_v3_selection/`, with
`_SUCCESS` and `_dataset_manifest.json`. Declared populations: train **8,946,370**
rows, **2023-01-01–2024-04-30**; validation **2,418,739** rows,
**2024-05-01–2024-08-31**. The existing runner checks actual population/schema,
split/date agreement, all source file digests and multiplicity before fitting.
No extra sampling, reduced evaluation or test inputs.

Load accepted fitted preprocessing **`PipelineModel_196025fd5a4b`**, without
refitting, from `artifacts/flighttime-v3/models/preprocessing/`. Accepted production
remains **LR at 0.39**, bundle `d0098a9572834c53bf76b04d732a94a6`, selection
`production-5936316dd7764ed79b049ae10ee1ca18`.

Frozen LR monthly source: `studies/lr_v1/evidence/diagnostics/results.json` and
four May–August JSON records; aggregate SHA-256
`c1313197e55e9d9d90793ec3401c3c0d3d61597ca35da94eae5c1f7b8fdee828`.
Keep the existing accepted bundle/selection/dataset, preprocessing, runtime-lock,
timezone and production-source checks unchanged; their identities are recorded
in [PLAN_30ITER.md](PLAN_30ITER.md) and [evidence_30iter/inputs.json](evidence_30iter/inputs.json).

Secondary reference: [RESULTS_30ITER.md](RESULTS_30ITER.md), saved equal-month
PR-AUC **0.4235378972669571**, selected threshold 0.36, original source
`27c832f3b093b901fa20a9210258d50922d95a13`. Reuse saved full-precision values;
do not rescore or refit. Four pinned compact records in `evidence_30iter/` must
match their original `attempt-002/` records before and after execution:

| Record | SHA-256 |
| --- | --- |
| completion.json | `7ec0129ed00dd93887d2d9ded749496404e276a45622160b7a73e249fe57d12c` |
| results.json | `6c02d060a8b974e89d59ac210410bb048c74720f5212af76918191162675718e` |
| inputs.json | `9cf30477fe48bb72a342409a7f5f3e4e5d8cd8bc125f07e71ba118b8bed884f2` |
| worker-status.json | `69743c0e2cd9a52d87da4e9ba1a16f097d6b446508e1d7a8b9a03aef804b3e34` |

Prior model/prediction files and all previous evidence are protected with targeted
before/after inventories. File presence alone is not readiness; missing or
incompatible inputs fail explicitly. No rebuilding, regeneration or substitutes.

## Unchanged evaluation and budget

Use existing natural-prevalence unweighted `areaUnderPR`/`areaUnderROC`,
`numBins=0`; no average-precision or pooled-PR replacement. Choose **one new
threshold** using complete-validation `threshold_counts`/`best_threshold`,
inclusive steps 0..100 / 100, exact rational F1 and lower step on ties. Inherit
neither LR's 0.39 nor GBT30's 0.36. Monthly capacity remains rows*percent//100,
score descending/ID ascending, fractional boundary ties and duplicate multiplicity.

Qualification remains **against LR only**, via `monthly_comparison`:

- Mean four paired monthly PR-AUC gains >= **0.005**.
- Mean recall change >= **-0.01** at **both** monthly top-10% and top-20% budgets.
- All four months, one classifier fit, all artifact/input checks, successful
  shutdown and authoritative supervisor completion.

LR mean **0.4280983226531411**; +0.005 comparison target **0.4330983226531411**,
recomputed from full-precision saved values, never rounded Markdown. Report
each month, means, worst changes, candidate threshold, overall operating metrics,
fit/total time, and paired changes against GBT30. Improvement over GBT30 alone
does not qualify; an incomplete attempt has no valid model-quality conclusion.

**One new 14,400-second wall-clock ceiling**, including preparation, source
staging, full-data fit, complete validation, save/reload, preservation checks,
cleanup and shutdown; reserve the final **60 seconds** for termination. No
automatic retry, replacement attempt, extension, subsampling or reduced checks.
GBT30 measured fit **4790.826843599993 s**, total **5280.869890715025 s**; these
are context, not a GBT60 completion-time prediction or permission to extend.

Keep pinned Spark 3.5.9 / Python 3.10.12 / x86_64 image
`sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf`,
Docker context `default`, inherited master, two total cores, one per executor,
640m executor / 768m driver memory, shuffle partitions 4, max file partition
67,108,864 bytes, batch size 1,000. Preserve DISK_ONLY, the 36-partition prepared
training policy and finally cleanup. Reuse corrected duration-only clock handoff
and container-owned process-group timeout; completion follows termination.

Exclusive output: **`data/studies/gbt-improvement-v1/attempt-003/`**, candidate
files under **`gbt-all-60iter/`**. If occupied, preserve and stop; never overwrite,
resume, retry or auto-select a replacement. Commit tested implementation first.
Only the explicit selection below authorizes the new fixed candidate:

```sh
python3 -B studies/gbt_v1/launch.py --execute --candidate gbt-all-60iter
```

The old default and explicit GBT30 commands retain their identities and occupied
paths. Minimal extensions live in `run_candidate.py`, `launch.py` and
`test_candidate.py`. Container staging uses this plan and the pinned GBT30
records. Tests reuse generated fixtures plus existing threshold/metric/population,
save/reload, completion, timeout and clock-origin cases. Test commands/results
are recorded below before the implementation commit. Results will be saved
separately in `RESULTS_60ITER.md` and `evidence_60iter/`.

Pre-execution verification in the unchanged pinned environment:

```sh
docker compose exec -T -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/tmp/gbt60-fixtures/source/project:/workspace/src:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip \
  -w /workspace spark-notebook python3 -B -m pytest -q -p no:cacheprovider \
  /tmp/gbt60-fixtures/source/project/studies/gbt_v1/test_candidate.py \
  /tmp/gbt60-fixtures/source/project/studies/lr_v1/test_metrics.py \
  /workspace/tests/unit/test_experiment_selection.py \
  /workspace/tests/unit/test_model_contract.py
```

**Exit 0: 86 passed, 1 existing PySpark deprecation warning, 117.39 seconds.**
Three tiny generated-data classifier fits (10/30/60 trees) exercise training-only
orchestration, save/reload, thresholds and full validation checks; these are
fixture tests, not study measurements. Other cases cover reference damage,
immutable settings, protected paths, exact F1, capacity/ties/multiplicity,
completion/failures and both incompatible clock origins for every selection.
Read-only readiness checks matched accepted LR/preprocessing, all four GBT30
reference records and the exact LR target. Syntax and `git diff --check` passed.
Attempt-003 remained absent. The four existing project services were initially
stopped; only they were started for tests and the authorized full-data attempt.

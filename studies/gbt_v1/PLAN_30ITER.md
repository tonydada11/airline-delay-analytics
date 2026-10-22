# Controlled GBT iteration study

Authorized scope: implement, test, and execute **one** `gbt-all-30iter` full-data
attempt, then report against saved LR and ten-iteration GBT results. No promotion,
next-experiment selection, baseline fit/scoring, or test-period evaluation.
Inspected base: `8bb83faa9b3507e9c57ad6c1094c0cd410faec9e` on
`study/gbt-improvement-v1`, initially clean; no conflicting newer commits.
Delivery remote: `github`, `git@github.com:tonydada11/airline-spark.git`.

## Fixed change and references

Reference: completed `gbt-feature-subset-all`, `featureSubsetStrategy="all"`,
`maxIter=10`, [RESULTS.md](RESULTS.md), authoritative
[evidence/completion.json](evidence/completion.json), execution source
`9e73cc43201a1b0369e2943cc97a96cf9a8378c3`.
New candidate: **`gbt-all-30iter`**, fitted **from scratch once**, changing only
**`maxIter: 10 -> 30` relative to that reference**. Relative to production
`GBT_SETTINGS`, **two** fields differ: `featureSubsetStrategy="all"`, `maxIter=30`.
Do not extend the saved ten-tree model or introduce early stopping.

```json
{
  "maxIter": 30, "maxDepth": 4, "maxBins": 32,
  "stepSize": 0.1, "subsamplingRate": 0.5,
  "featureSubsetStrategy": "all", "maxMemoryInMB": 128, "seed": 42,
  "featuresCol": "features", "labelCol": "is_arrival_delayed_15m",
  "weightCol": "class_weight"
}
```

Copy imported settings; never mutate production. Positive/negative fitting
weights remain 2.0/1.0. Preserve all effective defaults from saved production GBT,
including logistic loss, variance impurity, minimum-node settings, cache and
checkpoint settings, validationTol, and output columns; no validation indicator
or custom class thresholds. Save/reload checks expect 30 trees, two classes and
784 prepared features. The hypothesis is that further boosting iterations may
improve ranking; improvement over either reference is unconfirmed.

The primary comparator remains accepted **LR at 0.39**, bundle
`d0098a9572834c53bf76b04d732a94a6`, selection
`production-5936316dd7764ed79b049ae10ee1ca18`. The ten-iteration GBT comparison is
**descriptive only**. Its equal-month PR-AUC is 0.4182170318313452, threshold
0.35, fit 1633.7367135810055 s, complete attempt 2087.4594736050058 s. Neither
reference will be rescored or refitted. The research basis and feature/runtime
limitations in [PLAN.md](PLAN.md) remain applicable; no new model family or
feature direction is approved.

## Inputs and preserved contract

Reuse [PLAN.md](PLAN.md)'s accepted population, 50 ordered predictors, schema 3,
feature version `m7-flighttime-v2`, snapshot `bts-2023-2024-v1`, lineage/duplicates,
and fitted preprocessing `PipelineModel_196025fd5a4b`, without refitting.
Selection-only input: `data/warehouse/dws/ml_flighttime_v3_selection/`, completed
by `_SUCCESS` and `_dataset_manifest.json`. All original training rows from
2023-01-01–2024-04-30 (8,946,370 declared); validation 2024-05-01–2024-08-31
(2,418,739 declared). The runner validates sizes, schema, splits/date agreement
and multiplicity before fitting. No extra sampling or test-input reads.

Accepted executable bundle/preprocessing: `artifacts/flighttime-v3/models/`.
Frozen LR monthly source: `studies/lr_v1/evidence/diagnostics/results.json` and
`validation-2024-05.json` through `validation-2024-08.json`. Existing source and
input integrity checks remain unchanged. Key SHA-256 identities:

| Record | SHA-256 |
| --- | --- |
| Accepted bundle manifest | `7e1d0a1a9336dd383da9802db02586e23fcc411a4cd79294ee52c554c6d49ee7` |
| Selection manifest | `55f0022c50bf987df48d0506b15dc4bdf8acd57933838deae0708215ea35109e` |
| Selection dataset manifest | `093b15fd2085ba50d005b0fd70a804e81597b8df4c3e2f89a56fef18a799a265` |
| Frozen LR monthly aggregate | `c1313197e55e9d9d90793ec3401c3c0d3d61597ca35da94eae5c1f7b8fdee828` |
| Ten-iteration completion | `a07dfb207edd9c79c01ce22563ccd298d5aa8f2690f5886289d292a0d2a71b56` |
| Ten-iteration results | `89a66fae1b905395a8a5f2c88d0613c741fc8a46ff134133a9130e1fb996a007` |
| Ten-iteration inputs | `28cb48ed058d19661cd627fb1923ce3591285e57f2c9ad0f049202631ddebb16` |
| Ten-iteration worker status | `37b34c11b7e3b90b7947761d3ebf5e36fc12fcf109bc04a49e16b60dcee50342` |

Before implementation, all required inputs were present; the ten-iteration
completion and saved model/prediction digests matched. All 160 accepted
model/bundle/selection files matched the historical preservation inventory.
Protected original study files and attempt-001 were fingerprinted; selection's
47 Parquet part sizes/mtimes were recorded. Full-data readiness still requires
the runner's checks. Missing/incompatible inputs fail; no rebuilding/substitution.
The second candidate additionally pins the four compact ten-iteration records
above against their original local attempt-001 files, before and after evaluation.
Container staging includes those records and this plan; attempt provenance hashes
this plan. Original PLAN.md, RESULTS.md, evidence/ and attempt-001 stay unchanged.

## Evaluation and budget

Keep the original exact-F1 101-step complete-validation threshold selection:
inclusive probability >= step/100, steps 0..100, exact rational F1, lower step
on ties. Select one new threshold; inherit neither 0.39 nor 0.35. Reuse existing
unweighted natural-prevalence metrics, Spark areaUnderPR/ROC with numBins=0, and
monthly capacity `rows * percent // 100`; preserve score/ID ordering, fractional
boundary ties, and duplicate multiplicity. No pooled PR-AUC or average precision
substitution. Reuse `monthly_comparison` on saved full-precision results.

Qualification against **LR only** requires mean paired monthly PR-AUC gain
**>= 0.005**, mean recall change **>= -0.01 at both monthly 10% and 20% budgets**,
all four months, all artifact checks, one fit, successful shutdown and supervisor
completion. LR mean 0.4280983226531411; +0.005 target 0.4330983226531411.
Report all monthly metrics/deltas, equal-month means, worst changes, threshold,
overall metrics and fit/total time. Separately report paired monthly changes
against the ten-iteration reference without applying a secondary qualification rule.

**New authorized wall-clock cap: 14,400 seconds**, including launcher/container
preparation, one full-data classifier fit, complete validation, save/reload,
input/artifact checks, source cleanup and shutdown; reserve the final **60 seconds**
inside the cap for termination. The prior measured runtime above is context,
not a 30-iteration forecast. No extension, retry, sampling or reduced evaluation.
Keep pinned image `sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf`,
Spark 3.5.9/Python 3.10.12/x86_64, existing master, resources, partition policy
(36 prepared train partitions), DISK_ONLY and finally cleanup. Only elapsed
durations cross host/container clocks; the container owns the worker process
group and its timeout. Completion follows successful termination and cleanup.

Exclusive output: `data/studies/gbt-improvement-v1/attempt-002/`, candidate files
under `gbt-all-30iter/`. Never create it manually. Refuse any existing attempt,
including incomplete attempts: no overwrite, resume, replacement or retry.
Only the following explicit command selects the new candidate (after Phase A
passes and source is committed/clean):

```sh
python3 -B studies/gbt_v1/launch.py --execute --candidate gbt-all-30iter
```

The old `python3 -B studies/gbt_v1/launch.py --execute` still selects the original
candidate/attempt-001 and therefore refuses that already occupied path.

## Implementation and verification

Minimal fixed-candidate support is in [run_candidate.py](run_candidate.py) and
[launch.py](launch.py); shared safety/evaluation logic is retained.
[test_candidate.py](test_candidate.py) extends existing fixtures for both fixed
selections, 30-tree reload checks, output/plan mapping, reference identities,
timeout/clock-origin protections and unchanged production/original settings.
Phase A command in the existing pinned containers (source/compact JSON staged
only under `/tmp/gbt30-fixtures/source/project`; generated fixture data under
`data/.test-tmp/gbt-step2-*`, removed by fixture cleanup):

```sh
docker compose exec -T -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/tmp/gbt30-fixtures/source/project:/workspace/src:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip \
  -w /workspace spark-notebook python3 -B -m pytest -q -p no:cacheprovider \
  /tmp/gbt30-fixtures/source/project/studies/gbt_v1/test_candidate.py \
  /tmp/gbt30-fixtures/source/project/studies/lr_v1/test_metrics.py \
  /workspace/tests/unit/test_experiment_selection.py \
  /workspace/tests/unit/test_model_contract.py
```

Result: **70 passed, 1 existing PySpark distutils deprecation warning in 94.36s**,
exit 0. Two tiny generated-data classifier fits (10 and 30 trees) exercised the
shared orchestration/reload checks; these do not evaluate the study hypothesis.
Import/help, exclusive paths, immutable settings, reference fingerprints,
threshold/metric/capacity/multiplicity semantics, incomplete completion rejection,
timeout interruption and both clock-origin regressions passed. Python syntax and
`git diff --check` passed. Full-data attempt-002 remained absent through Phase A.
The four pre-existing project containers were initially stopped and started with
existing images/resources for tests and the subsequently authorized full-data run.
Implementation and this plan are committed before execution. Full-data results
will be reported separately in `RESULTS_30ITER.md` and `evidence_30iter/`;
this execution plan is frozen at that source commit.

# Controlled GBT study — Step 1

2026-10-21 (America/Chicago). Step 1 covered documentation and local metadata
inspection only. Step 2 implementation/fixture verification is recorded below;
Step 2 did not execute the real-data experiment or authorize promotion.

**Steps 3 and 4 status (2026-10-21, America/Chicago): COMPLETE — DOES NOT QUALIFY.**
The separately authorized single attempt completed with all checks and four months;
mean paired PR-AUC gain was -0.009881290821795952. See [RESULTS.md](RESULTS.md)
and its compact evidence. Production remains accepted LR at 0.39. Step 5 awaits
separate review. The candidate, budget, and qualification criteria below are unchanged.

## Baseline and preserved contract

Inspected/base commit: `ab6bd50e4f59d535836f61879f37c84a4546f9ae` (`main`).
The initial worktree was clean. Local branches, worktrees, and remotes were
inspected before creating `study/gbt-improvement-v1` from that commit. Remote
`github` is `git@github.com:tonydada11/airline-spark.git`; its `main` matched the
base and the proposed study branch did not exist remotely. `origin` points to
Wisconsin GitLab and is not the delivery destination. The existing
`study/lr-improvement-v1` branch/worktree at `9b99c88d2d922ae3cfdfc0ceb93bea6071889178`
was clean and is preserved, along with its outputs and findings B, C, and D.

Verified from the local executable bundle, selection records, and their
`results/production/` copies: accepted classifier **LR**, threshold step **39/100
= 0.39**, bundle `d0098a9572834c53bf76b04d732a94a6`, selection run
`production-5936316dd7764ed79b049ae10ee1ca18`. A standard-library `Fraction`
recalculation over both saved 101-step threshold grids confirmed LR step 39
wins; existing GBT selects step 33. The completed LR weight-1 and regParam-0.3
attempts both failed qualification and retained this baseline; do not repeat them
([results](../lr_v1/CANDIDATE_RESULTS.md),
[full-precision comparisons](../lr_v1/evidence/candidate-results.json)).

Preserve the current explicit decisions in [pipeline rules](https://github.com/tonydada11/airline-delay-analytics/blob/main/docs/reference/pipeline-rules.md)
and [selection rules](https://github.com/tonydada11/airline-delay-analytics/blob/main/docs/reference/model-selection.md), the inspected source contracts,
and `conf/project.yaml`:

- Snapshot `bts-2023-2024-v1`, feature version `m7-flighttime-v2`, dataset/model
  schema 3. Eligible non-cancelled, non-diverted flights with binary target
  `is_arrival_delayed_15m`; preserve physical rows and duplicate multiplicity.
  `flight_id`/lineage fields are metadata, never predictors.
- All 50 ordered predictors: five schedule categories, nine schedule numeric
  fields, and 12 each of causal history rates, counts, and missing indicators
  for carrier/origin/destination/route over 7/30/90 days. Preserve scheduled
  departure UTC prediction time, strict prior-arrival availability, timezone/DST
  handling, and the existing zero-reporting-latency assumption. Operational
  publication/ingestion availability is not established.
- Inclusive chronological boundaries: train **2023-01-01–2024-04-30**;
  validation **2024-05-01–2024-08-31**; test **2024-09-01–2024-12-31**.
  Selection features contain only train/validation. No train-plus-validation fit.
- Reuse the accepted train-fitted `PipelineModel_196025fd5a4b` without refitting:
  five alphabetical `StringIndexerModel`s (`handleInvalid=keep`), one-hot
  encoding (`handleInvalid=keep`, existing `dropLast=true`), mean imputation of
  the 21 numeric inputs, then assembly with unchanged counts/missing indicators.
  Preserve category dictionaries, imputation values, order, and **784 prepared
  dimensions** (frozen diagnostics and saved GBT metadata).
- Keep configured native Spark **3.5.9**, pinned image/dependency lock, and current
  Compose base plus local `docker-compose.override.yml`: two one-core workers
  (768m worker allocation), `spark.cores.max=2`, executor cores 1, executor memory
  640m, driver memory 768m, shuffle partitions 4, file partition limit 67108864
  bytes, cached batch size 1000. Preserve `DISK_ONLY`, finally cleanup, and
  prepared partitions `max(2, ceil(train_rows/250000))` (36 for the declared size).
  Inherit the configured Spark master through `create_spark_session`; do not
  copy external hardcoded masters or random splits. No runtime was started to
  verify its present health. The accepted manifest records x86_64/Python 3.10.12.

Discrepancies/precedence: older fixed-LR wording does not replace the current
validation selection contract; 0.39 is the verified accepted outcome, not a
universal threshold. The handoff's tested commit
`1a4756e86568f6e464fcc2f8dea34be0a5521128` and the LR study's historical main
commit differ from today's Git base. Accepted manifests have null Git commit/
dirty fields, but their production-source SHA-256 matches today's source
(`55dd33b3cd4ca61d71432b827cc8f63774928ca49b3a41281d1ee1849f090017`).
Do not rewrite that provenance. The local Compose override is untracked and
byte-identical to `evidence/consolidation/local-runtime-override.yaml`. Current
healthy-Docker/native-architecture rules supersede older Colima-only guidance.

## Exactly one future candidate

Name: **`gbt-feature-subset-all`**. The only estimator change is
`featureSubsetStrategy: "sqrt" -> "all"` relative to the existing GBT; accepted
LR is the primary performance comparator. Existing GBT source configuration
(`src/airline/ml/model_contract.py::GBT_SETTINGS`), also verified against the
frozen selection and saved classifier metadata:

```json
{
  "maxIter": 10, "maxDepth": 4, "maxBins": 32,
  "stepSize": 0.1, "subsamplingRate": 0.5,
  "featureSubsetStrategy": "sqrt", "maxMemoryInMB": 128, "seed": 42,
  "featuresCol": "features", "labelCol": "is_arrival_delayed_15m",
  "weightCol": "class_weight"
}
```

Positive-class fitting weight stays **2.0**; negative weight stays **1.0**.
All training rows enter preparation and classifier fitting; keep the estimator's
intrinsic `subsamplingRate=0.5` and seed, with no additional sampling. Preserve
all other pinned effective defaults, verified in saved GBT metadata:
`lossType=logistic`, `impurity=variance`, `minInstancesPerNode=1`,
`minInfoGain=0.0`, `minWeightFractionPerNode=0.0`, `cacheNodeIds=false`,
`checkpointInterval=10`, `validationTol=0.01`, `leafCol=""`, output columns
`prediction`/`probability`/`rawPrediction`; no validation indicator or custom
class thresholds. Do not refit preprocessing or refit either baseline classifier.

Hypothesis: allowing each split to consider all prepared features may improve
ranking relative to the constrained existing GBT. This is not a confirmed
underfitting diagnosis or a promised gain against either GBT or accepted LR.

## Research basis and deferred review

Primary references/URLs verified on 2026-10-21; these motivate an experiment,
not a claim of improvement:

- **Apache Spark 3.5.9:** [GBT API](https://spark.apache.org/docs/3.5.9/api/python/reference/api/pyspark.ml.classification.GBTClassifier.html)
  defines per-node feature consideration: `sqrt` considers the square root of
  prepared feature count (28 of 784); `all` considers every feature. The
  [pinned GBT constructor and `_GBTClassifierParams` source](https://spark.apache.org/docs/3.5.9/api/python/_modules/pyspark/ml/classification.html#GBTClassifier)
  set the GBT default to `all`. The inherited parameter description mentions
  generic forest default `auto`; that is not the GBT constructor default.
  Repository `sqrt` is explicit. Defaults establish semantics, not superiority.
- **Grinsztajn, Oyallon, Varoquaux (2022),**
  [“Why do tree-based models still outperform deep learning on typical tabular data?”](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html).
  Its tabular benchmarks motivate examining tree inductive biases and feature
  access. Its medium-sized datasets, tuning budgets, and comparisons with neural
  networks do not establish Spark GBT performance on this chronological,
  multimillion-row airline population or select this parameter value.
- **Chen and Guestrin (2016),**
  [“XGBoost: A Scalable Tree Boosting System”](https://arxiv.org/abs/1603.02754),
  [dmlc/xgboost](https://github.com/dmlc/xgboost), and
  [official PySpark integration](https://xgboost.readthedocs.io/en/stable/tutorials/spark_estimator.html).
  Sparse-data/scalable boosting methods and `SparkXGBClassifier` make XGBoost
  on the existing validated features a plausible **Step 5 review option**.
  A complete controlled GBT result showing persistent ranking limitations, plus
  a verified dependency/resource/persistence compatibility plan, would justify
  considering it. Different algorithms and published results do not transfer
  measured gains or runtime; compatibility with this pinned deployment is
  **UNVERIFIED**. Deferred because it adds a library/integration change beyond
  the single-factor experiment; no installation or implementation is approved.
- **Rebollo and Balakrishnan (2014),**
  [“Characterization and prediction of air traffic delays”](https://web.mit.edu/hamsa/www/pubs/RebolloBalakrishnanTRC2014.pdf),
  DOI [10.1016/j.trc.2014.04.007](https://doi.org/10.1016/j.trc.2014.04.007).
  Temporal/network delay states motivate **short-horizon causal operating-history
  features as a Step 5 review option**. The study concerns aggregate departure
  delays on selected links, different years/horizons/targets; its results do not
  prove gains for individual 15-minute arrival-delay prediction. Its centered
  delay-state windows cannot be copied as causal features. Residual errors
  associated with recent disruption, sufficient historical coverage, and explicit
  availability/leakage evidence would justify review. Deferred because new
  windows require a separate feature contract/build and availability audit;
  actual publication timestamps remain unavailable. No new features are approved.

No additional model families enter this study. Step 5 options require a separate
decision after the controlled result; none is a fallback triggered automatically.

## Local input readiness (filesystem/metadata only)

All paths are project-relative to the repository root. **PRESENT** means files
were located; **MISSING** means an expected path is absent; **INCONSISTENT** means
inspected identities disagree; **UNVERIFIED** means inspection did not establish
the property. GitHub review JSON alone is insufficient for execution.

| Input | Finding |
|---|---|
| `data/warehouse/dws/ml_flighttime_v3_selection/` | **PRESENT**: `_SUCCESS`, `_dataset_manifest.json`, `split=train/` (37 nonempty Parquet parts), `split=validation/` (10). Every recorded part name/byte size matches the LR preservation inventory; no missing recorded part or `_temporary` entry. Parquet bytes were not hashed/read. |
| `artifacts/flighttime-v3/models/preprocessing/` and `classifier/` | **PRESENT**: executable Spark model metadata and data files, not just review exports. All 64 preprocessing files and 8 classifier files (31,819 bytes combined) match saved SHA-256 fingerprints in `studies/lr_v1/evidence/baseline-identities-before.json`. |
| `artifacts/flighttime-v3/models/manifest.json` | **PRESENT**: accepted LR/bundle/threshold identity above; embeds the exact selection record. |
| `artifacts/flighttime-v3/experiments/production-5936316dd7764ed79b049ae10ee1ca18/` | **PRESENT**: `selection.json`, `baseline.json`, `validation.json`, saved GBT; its 12 small files match the same preservation inventory. Source GBT settings equal saved settings. |
| `studies/lr_v1/evidence/diagnostics/{results,status,validation-2024-05,validation-2024-06,validation-2024-07,validation-2024-08}.json` | **PRESENT**: complete frozen LR monthly baseline. Original files under `data/studies/lr-improvement-v1/diagnostics/` match byte-for-byte; each monthly object equals the aggregate record. Compact candidate comparison baseline PR-AUC/recall values also match exactly. |
| Missing/inconsistent inputs | **MISSING: none** among the required paths inspected. **INCONSISTENT: none** among the reconciled identities. This is not certification of full-data correctness. |
| Execution properties | **UNVERIFIED**: Spark readability, current row counts/schema/value correctness, full Parquet fingerprints, live runtime health, and future save/reload/evaluation checks. No Spark job or raw/warehouse-wide scan was run. |

Small-file SHA-256 reconciliation (local files equal the corresponding
`results/production/` review copies; selection's declared dataset, baseline,
validation, and raw-manifest fingerprints match):

| Record | SHA-256 |
|---|---|
| Bundle manifest | `7e1d0a1a9336dd383da9802db02586e23fcc411a4cd79294ee52c554c6d49ee7` |
| Selection manifest | `55f0022c50bf987df48d0506b15dc4bdf8acd57933838deae0708215ea35109e` |
| Selection dataset completion manifest | `093b15fd2085ba50d005b0fd70a804e81597b8df4c3e2f89a56fef18a799a265` |
| Selection `baseline.json` / `validation.json` | `06be1ed2a1dd191e7798a9e021612e7ae1546239efc563ebb85eae245f0ba523` / `1d6a4acb84070edd0c68dea080c8581514d5d1b0b66a917f8cf027b5c41e918e` |
| Frozen LR diagnostics `results.json` | `c1313197e55e9d9d90793ec3401c3c0d3d61597ca35da94eae5c1f7b8fdee828` |
| `conf/raw_data_manifest.csv` | `fef0d967bbc36d2beb2898701428ca55aa5b0f0a7c9bb0f5ede2ed8ac7852efb` |
| `requirements.lock` | `909ee6549a004c7378493f5c2b1cb16cd366cce6a700e149a5c2d0b228a06b8a` |
| `conf/airport_timezones.csv` | `60fd42f7a8ba6a1f13f37c42a28e920a2dce2291c3fd5c815047228d2d6fc1a8` |

Dataset/model/selection records agree on schema, feature version, snapshot,
raw-manifest fingerprint, and time policy. Selection-feature bundle/run IDs are
correctly null: these features predate selection and retain immutable provenance.
**Declared claims, not independently recounted rows:** dataset 11,365,109;
training 8,946,370; validation 2,418,739; dataset duplicate groups/excess rows 0/0.
Manifest presence and these targeted checks do not prove Spark readability or
full-data correctness. Inputs are present for future verification; execution
still requires the later step's authorization and checks. If any listed input
is absent or fails reconciliation then, record its exact path and block execution;
do not download, rebuild, regenerate, or substitute it in this step.

## Frozen evaluation and qualification

Use `airline.ml.metrics.calculate_metrics`: unweighted natural-prevalence
`BinaryClassificationEvaluator(areaUnderPR, numBins=0)` on positive-class
probabilities. No pooled PR-AUC substitution and no other library's average
precision. Reuse `studies/lr_v1/diagnose.py::budget_metrics` and
`studies/lr_v1/metrics.py`: calendar-month capacities `rows * percent // 100`,
descending score then ascending `flight_id`. Preserve duplicate multiplicity;
indistinguishable score/ID boundary groups receive equal fractional selection
mass, reporting expected precision/recall. Labels do not break selection ties.

Frozen source: `studies/lr_v1/evidence/diagnostics/validation-<month>.json`, also
embedded in the fingerprinted diagnostics `results.json`:

| Month | Declared rows | LR PR-AUC | LR recall at 10% | LR recall at 20% |
|---|---:|---:|---:|---:|
| 2024-05 | 599226 | 0.43245490658602814 | 0.1954497207409841 | 0.3525739846849807 |
| 2024-06 | 601272 | 0.40353655194596116 | 0.19073660342125445 | 0.34850778618826905 |
| 2024-07 | 613938 | 0.4705872368424865 | 0.18887636433618343 | 0.3454815385038353 |
| 2024-08 | 604303 | 0.4058145952380886 | 0.2104758412546188 | 0.37334640227907373 |
| Equal-month mean | — | **0.4280983226531411** | **0.1963846324382602** | **0.3549774279140397** |

Recomputed directly from saved JSON using only standard-library `json`, `sum`,
`math.fsum`, and `decimal.Decimal`: `sum(monthly_pr_auc) / 4` and `fsum(...) / 4`
both give **0.4280983226531411**; adding **0.005** gives the comparison target
**0.4330983226531411**. Decimal arithmetic over the saved numeric strings agrees
(`0.42809832265314110`, `0.43309832265314110`). Rounded Markdown was not an input.
The executable decision remains the four paired differences in
`monthly_comparison`, avoiding a rounded or pooled target comparison.

Qualification requires **all** of: mean of the four paired monthly PR-AUC
gains versus accepted LR **>= 0.005**; equal-month mean recall change
**>= -0.01 absolute at BOTH top-10% and top-20%**; all four months and required
artifact checks complete. Report each month's PR-AUC, budget precision/recall,
paired changes, equal-month means, worst monthly PR-AUC/recall changes, fit and
complete-attempt runtime, and validation-selected precision/recall/F1. There is
no additional worst-month elimination rule. These are retrospective practical
criteria, not statistical significance or prospective validation.

Select the candidate's own operating threshold on the complete May–August
validation population using `threshold_counts`/`best_threshold`: inclusive
`probability >= step/100`, steps 0..100; maximize exact rational
`2TP/(2TP+FP+FN)` by cross multiplication (zero denominator = 0/1), then lower
step on a tie. Do not inherit LR's 0.39 or select separate monthly thresholds.
Compute monthly and overall confusion metrics at that frozen candidate step.
Required checks include parameter/default/dimension agreement, unchanged accepted
preprocessing, full-validation score multiset equality across classifier save/
reload, and saved prediction schema, population/label multiset, and threshold
agreement. Write completion evidence only after all checks and months succeed.

Do not read September–December scores, labels, or metric artifacts for this study,
or invoke test evaluation. The required historical handoff/verification documents
were read as context; their existing test reporting is not study evidence. This
historically inspected test period is not a pristine holdout. Qualification does
not authorize promotion, production publication, or changes to the authoritative
LR/GBT validation-F1 production selection contract.

## Execution boundary, proposed budget, and Step 2 reuse

Only this plan is created in Step 1. Proposed future run root
`data/studies/gbt-improvement-v1/attempt-001/`, with candidate artifacts under
`gbt-feature-subset-all/` (classifier, validation predictions, monthly results,
completion/status) and attempt logs/timings alongside. The entire study root was
absent at inspection; no output directories were created. Preserve root accepted
artifacts, B1/checkpoints, `data/studies/lr-improvement-v1/`, and all historical
study/baseline outputs. Recheck exclusive path availability before a future run;
never overwrite an incomplete attempt.

**Proposed hard wall-clock cap: 14,400 seconds (4 hours), ONE full-data classifier
fit maximum, sequential, no automatic retries.** It includes process/session
preparation, input checks, transformation/materialization, fit, save/reload
checks, full validation/threshold/monthly evaluation, artifact completion, cleanup,
and shutdown. Reserve the final 60 seconds inside the cap for termination and
shutdown; do not start dependent work without time to finish. The fit counts
once started even if it fails. Record timeout/failure/unfinished checks as an
incomplete attempt, exclude it from qualification, and stop. No additional
candidate, reduced evaluation, silent training subsampling, resource increase,
or automatic extension is permitted. This budget is a proposal for review,
not permission to execute or a completion-time forecast.

Evidence versus estimate: the accepted pipeline log
`evidence/acceptance/consolidation-20261020/pipeline.log` records the full
preprocessing + LR/GBT selection/publication phase at 13:12:10–13:51:34 -0500,
**2,364 seconds**; it does not isolate GBT fit duration. The saved LR study's
`studies/lr_v1/evidence/budget.json` records complete attempts of **372.44712954999704** and
**357.3769675789954** seconds (fits **129.34389798200573** and
**118.62844850600231** seconds); baseline diagnostics took
**493.19425812299596** seconds. These are historical measured observations on
the preserved runtime, not GBT-all measurements. Four hours proposes roughly
six times the historical whole selection phase for the larger feature search
and complete validation. Considering 784 rather than 28 dimensions per split
can be substantially slower; runtime need not scale linearly and completion
within this cap is **UNVERIFIED**. This is a new proposed allowance, not reuse
of the LR study's remaining budget.

Suitable existing helpers for Step 2 review: configured paths/session creation;
`load_bundle` for accepted inputs; `validate_dataset_manifest` and
`validate_feature_frame`; `continuous_score`, `threshold_counts`, `best_threshold`;
`calculate_metrics`; `budget_metrics`, `capacity`, `monthly_comparison`;
`validate_predictions` and existing save/reload multiset checks. The old LR fit
runner hardcodes 0.39 and must not be invoked unchanged. Production `select`
refits preprocessing and both classifiers. `airline.ml.experiments.evaluate`
is test-only; `airline.ml.evaluate.evaluate` supports validation and test but
evaluates the frozen production bundle. Neither is this candidate's entrypoint.
Production `validate_models` requires the original GBT `sqrt` contract; preserve
it and use study-specific candidate checks. Step 1 implemented no runner.

## Step 2 implementation and verification

[run_candidate.py](run_candidate.py) implements the fixed candidate, pinned input
identities, configured selection-only reads, distributed population checks,
one classifier fit, GBT parameter/default/dimension and full-validation reload
checks, validation-selected exact-F1 threshold, saved lineage/multiplicity checks,
and the unchanged monthly metrics/comparison helpers. Accepted preprocessing is
loaded, never fitted. Future execution also compares selection Parquet fingerprints
to the frozen preservation inventory and checks inputs again after evaluation.
Reused frames use `DISK_ONLY` and finally cleanup. No production/LR-study files
or validators were changed.

[launch.py](launch.py) is a thin host launcher because the existing notebook
mounts omit `studies/`. It requires a clean committed source tree and an already
running configured runtime; it stages only required source/compact evidence in
container temporary storage and invokes the supervisor **inside that container**.
It never starts/stops services, builds/pulls images, or changes resources/master.
After container readiness, it sends a measured host preparation **duration**,
including Docker/VM startup. No raw clock timestamp crosses the boundary.
The container uses only its own monotonic clock to charge source staging,
supervision, and source cleanup against the remaining 14,400-second allowance;
readiness overlap is conservatively charged. Source cleanup precedes completion.
The supervisor sends TERM after the first 14,340 seconds, reserves the final 60
seconds for cleanup, and kills remaining owned worker-group processes before
the ceiling. Failure/timeout exits nonzero with `attempt.json`, worker status/log,
and any partial artifacts retained. No retry, resume, or replacement attempt is
selected. Successful Spark shutdown and worker exit zero precede the sole final
`attempt-001/completion.json`; candidate `results.json` remains explicitly pending
termination and has no qualification flag. Completion alone publishes qualification.

[test_candidate.py](test_candidate.py) covers isolation, import/help, input/output
rejection, failed/incomplete/timeout termination, exact candidate construction,
and generated-fixture fitting/reload, population/schema/threshold checks, and
distributed capacity/tie semantics. Timeout tests use a one-second supervisor
budget and a sleeping subprocess, including a TERM-ignoring worker. Temporary
source snapshots used `/tmp/gbt-step2-source/source/project` inside the notebook;
only generated fixture Parquet used `data/.test-tmp/gbt-step2-*` and was removed.
An initial shallow-source import failure in the historical helper was corrected
in source staging and covered by a bootstrap regression; LR code was preserved.

Exact targeted command (pinned image
`sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf`):

```sh
docker compose exec -T -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/tmp/gbt-step2-source/source/project:/workspace/src:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip \
  -w /workspace spark-notebook python3 -B -m pytest -q -p no:cacheprovider \
  /tmp/gbt-step2-source/source/project/studies/gbt_v1/test_candidate.py \
  /tmp/gbt-step2-source/source/project/studies/lr_v1/test_metrics.py \
  /workspace/tests/unit/test_experiment_selection.py \
  /workspace/tests/unit/test_model_contract.py
```

Result: **49 passed in 66.19s**, exit 0; one existing PySpark `distutils`
deprecation warning. This includes a tiny generated-data GBT fit and real Spark
serialization/metric checks; it does not establish full-data execution readiness
or performance. Existing input-completion regressions were also run:

```sh
docker compose exec -T -e PYTHONDONTWRITEBYTECODE=1 -w /workspace \
  spark-notebook python3 -B -m pytest -q -p no:cacheprovider \
  tests/integration/test_dataset_completion.py
```

Result: **7 passed in 23.92s**, exit 0, using only generated Parquet. The four
existing services were initially stopped, started with `docker compose start
--wait spark-master spark-worker-1 spark-worker-2 spark-notebook` without image
build/pull or resource changes, then stopped by their inspected container IDs.
Python syntax, relative documentation links, and `git diff --check` passed;
reviewed changes are confined to this plan and the three implementation/test
files above. Full-data input validation, classifier execution, and qualification
remain **NOT RUN**. No full pipeline, benchmark, or broad test suite was run.

**Step 3 only—not executed:** from the repository root, after separate execution
authorization and with the unchanged configured services already healthy:

```sh
python3 -B studies/gbt_v1/launch.py --execute
```

No real-data study fit, accepted/baseline refit, baseline scoring, test-period
evaluation, or promotion occurred in Step 2. The real `attempt-001/` directory
was not created. Stop after Step 2; the candidate definition, comparison target,
and proposed four-hour budget above remain unchanged.

### Step 2 clock-origin correction

Replaced the cross-host monotonic timestamp with elapsed-duration accounting;
[Python 3.10 documents an unspecified clock reference point](https://docs.python.org/3.10/library/time.html#time.monotonic).
The regression exercises the launcher, bootstrap, and supervisor with host
origins 1,000 and 1,000,000,000 versus container origin 100,000. It checks the
worker's remaining allowance and charges simulated staging and cleanup time.
Cleanup failure/exhaustion cannot publish completion; the existing one-second
blocking-worker timeout check also passes. The four-hour ceiling, final
60-second allowance, process-group termination, and candidate contract remain.

Focused correction check in the same pinned image, with the repository mounted
read-only at `/review`, networking disabled, and no project services started:

```sh
python3 -B -m pytest -q -p no:cacheprovider studies/gbt_v1/test_candidate.py \
  -k 'not distributed and not failed_fit and not population_checks'
```

Result: **23 passed, 6 deselected in 1.87s**, exit 0, with the existing PySpark
import deprecation warning. No Spark session, fixture model fit, real-data
execution, or promotion occurred during this correction. Syntax and
`git diff --check` passed; the real study attempt directory remains absent.

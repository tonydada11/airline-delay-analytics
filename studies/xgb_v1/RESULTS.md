# XGB-v1 results

**Execution status: BLOCKED. Recommendation: INCOMPLETE.**

**No valid full-data performance comparison exists.** Full-data fits started: 0. Complete candidate metrics and selected threshold are unavailable.

[Full-precision metrics and execution summary](../../artifacts/experiments/xgb-v1/metrics.json). [Reproduction instructions](README.md).

**Environment and model.** Spark 3.5.9, Python 3.10.12; study image built offline from the unchanged production image with hash-locked wheels. Driver and both executor hosts use identical versions:

| Dependency | Version |
|---|---|
| joblib | 1.4.2 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| pyarrow | 19.0.1 |
| scikit-learn | 1.6.1 |
| scipy | 1.15.3 |
| threadpoolctl | 3.6.0 |
| xgboost-cpu | 3.0.5 |

Two one-core workers, 768 MiB worker advertisements, **640 MiB executor heaps**, 768 MiB driver heap, one CPU per task, four shuffle partitions. Isolated container caps: workers/driver 2 GiB each; master 512 MiB. No resources increased. Exact runtime and image identity are recorded in the artifact directory.

Official `SparkXGBClassifier`: 100 estimators, depth 4, learning rate 0.05, CPU `hist`, seed 42, two workers; columns `features`, `is_arrival_delayed_15m`, `class_weight`. Binary labels select `binary:logistic` automatically because the Spark API does not accept an explicit objective override. Weights: positive 2.0, negative 1.0. All other estimator defaults retained, including `missing=NaN` and disabled sparse optimization. No early stopping or tuning.

Compatibility was tested empirically with the exact installed packages. PyArrow 19.0.1 is outside the older `<13` range documented for pandas API/Spark Connect in the [Spark 3.5.9 installation table](https://spark.apache.org/docs/3.5.9/api/python/getting_started/install.html); this report claims only the exercised Spark XGBoost paths, not broad official compatibility.

**Dataset and preprocessing.** Accepted bundle `d0098a9572834c53bf76b04d732a94a6`; preprocessing `PipelineModel_196025fd5a4b` reused without fitting, **50 input features / 784 prepared dimensions**. Full feature validity, manifest, chronological bounds, and monthly positive counts were checked.

| Split | Dates | Rows | Positive rows |
|---|---|---:|---:|
| train | 2023-01-01–2024-04-30 | 8,946,370 | 1,827,125 |
| validation | 2024-05-01–2024-08-31 | 2,418,739 | 631,898 |

**Validation comparison.** LR values are frozen evidence. N/A is unavailable, never zero. Display rounding is not used in qualification.

| Month | LR ROC | XGB ROC | LR PR | XGB PR | Paired PR difference |
|---|---:|---:|---:|---:|---:|
| 2024-05 | 0.686424830652 | N/A | 0.432454906586 | N/A | N/A |
| 2024-06 | 0.683147717472 | N/A | 0.403536551946 | N/A | N/A |
| 2024-07 | 0.690358965680 | N/A | 0.470587236842 | N/A | N/A |
| 2024-08 | 0.696626120513 | N/A | 0.405814595238 | N/A | N/A |
| Pooled | 0.687063985607 | N/A | 0.427191548086 | N/A | N/A |
| Equal-month mean | — | — | **0.4280983226531411** | N/A | N/A |

LR threshold **0.39**; XGB threshold **N/A**. Existing inclusive 0.00–1.00 grid maximizes exact rational pooled F1, then prefers the lower threshold.

| Month | LR precision / recall / F1 | XGB precision / recall / F1 |
|---|---|---|
| 2024-05 | 0.414870071317 / 0.543588415173 / 0.470585983115 | N/A / N/A / N/A |
| 2024-06 | 0.342722610200 / 0.734435959852 / 0.467355162338 | N/A / N/A / N/A |
| 2024-07 | 0.387862644363 / 0.775036428120 / 0.516997021670 | N/A / N/A / N/A |
| 2024-08 | 0.359623051541 / 0.634805235100 / 0.459139384604 | N/A / N/A / N/A |
| Pooled | 0.373881341690 / 0.675958778157 / 0.481460691255 | N/A / N/A / N/A |

| Month / budget | LR precision | LR recall | XGB precision | XGB recall | Δ precision | Δ recall |
|---|---:|---:|---:|---:|---:|---:|
| 2024-05 / 10% | 0.516254464137 | 0.195449720741 | N/A | N/A | N/A | N/A |
| 2024-05 / 20% | 0.465634778255 | 0.352573984685 | N/A | N/A | N/A | N/A |
| 2024-06 / 10% | 0.475659853310 | 0.190736603421 | N/A | N/A | N/A | N/A |
| 2024-06 / 20% | 0.434555191511 | 0.348507786188 | N/A | N/A | N/A | N/A |
| 2024-07 / 10% | 0.559510041861 | 0.188876364336 | N/A | N/A | N/A | N/A |
| 2024-07 / 20% | 0.511707265427 | 0.345481538504 | N/A | N/A | N/A | N/A |
| 2024-08 / 10% | 0.493926857521 | 0.210475841255 | N/A | N/A | N/A | N/A |
| 2024-08 / 20% | 0.438068839980 | 0.373346402279 | N/A | N/A | N/A | N/A |

Metrics reuse unweighted Spark AUC (`numBins=0`) and the LR study budget utility: `floor(budget × monthly rows)`, score descending then flight ID ascending, fractional selection within indistinguishable boundary duplicates. Threshold choice does not imply improved ranking.

**Compute and blocker.**
The installed adapter materializes and retains dense float32 batches before QuantileDMatrix: **8,946,370 × 784 × 4 = 28,055,816,320 bytes (26.129 GiB)**, or 13.065 GiB per worker at an even split. Current host RAM is 11.681 GiB; total swap is 4.000 GiB. Input alone exceeds both combined, excluding JVM, Arrow, labels, weights and matrix overhead. Sparse optimization would reinterpret valid zeros as missing, changing the frozen contract. This is a verified lower bound, not an observed full-fit OOM.

Minimum prerequisite for any future full attempt: an explicitly authorized change in available memory beyond this input lower bound, plus separately assessed overhead. A sufficient total cannot be inferred from the smoke fit. That would depart from the current fixed-resource instruction, so no such change was made.

Current supervisor elapsed: **129.792 s**; Spark preflight: **109.290 s**. Original smoke training: **6.945 s**, 256 synthetic rows. The existing single smoke fit was reused; current checks reloaded it, serialized it again, and verified all probabilities/labels and dense/sparse inference. Its weighted constant probability is 0.4000000059604645 (analytic value 0.4). This is not flight-model performance.

Full-data fit time: N/A; evaluation time: N/A. The complete full attempt has a 7,200-second cap, including preparation, fitting, serialization and evaluation, using host supervision and `timeout --signal=TERM --kill-after=10s 7170`. No automatic fit retries. The resource gate prevented entering that attempt.

| Container | Lifetime peak MiB | OOM kills |
|---|---:|---:|
| airline-xgb-v1-driver-1 | 1306.9 | 0 |
| airline-xgb-v1-master-1 | 353.1 | 0 |
| airline-xgb-v1-worker1-1 | 1479.6 | 0 |
| airline-xgb-v1-worker2-1 | 1585.7 | 0 |

Peaks include cache and are not necessarily simultaneous.

**Commands and verification.** `python3 studies/xgb_v1/run.py` built the image, ran targeted tests and the Spark preflight, preserved production, stopped study containers and generated this report. Exact argv and exit codes:

- `build`: exit **0**; [command and log record](../../artifacts/experiments/xgb-v1/execution/build.json).
- `tests`: exit **0**; [command and log record](../../artifacts/experiments/xgb-v1/execution/tests.json).
- `preflight-execution`: exit **0**; [command and log record](../../artifacts/experiments/xgb-v1/execution/preflight-execution.json).

**13 passed in 0.08s.** Test log (historical original-repository file `../../artifacts/experiments/xgb-v1/execution/tests.log`; omitted from this compact reconstruction); [pip check](../../artifacts/experiments/xgb-v1/execution/pip-check.json); [runtime](../../artifacts/experiments/xgb-v1/environment.json); [effective fitted parameters](../../artifacts/experiments/xgb-v1/smoke-booster-config.json). Tests cover metric/tie calculations, baseline evidence, the memory bound, and distributed execution evidence. Runtime checks cover all feature/population contracts, label/weight alignment, missing-value interpretation and model serialization.

Preservation: 311 files checked by SHA-256 (unchanged: True); 1332 warehouse/raw/model/metric/acceptance files checked by size and mtime (unchanged: True). Metadata checks do not hash content. Test-period contents were not opened for these checks. Production containers unchanged: True. Only selection data and accepted models were mounted read-only; no test data was scored. No production pipeline, promotion or push.

Files changed: study `run.py`, `experiment.py`, `compose.yml`, `RESULTS.md`; added `Dockerfile`, `report.py`, `README.md`. Existing targeted tests and study lock retained. Generated outputs/logs now reside solely in `artifacts/experiments/xgb-v1/` for this execution: `execution/`, `build/`, `smoke-model/`, `smoke-reload-check/`, `metrics.json`, and environment/population records. Prior evidence and reports were preserved byte-for-byte in `prior/`; original ignored data artifacts remain untouched. Any partial candidate artifacts are excluded from comparison.

Repository identity: worked in the supplied checkout; README names `tonydada11/airline-spark`, while the configured origin is a Wisconsin GitLab URL. No remote was changed. Historical setup failures and the prior supervisor logging correction remain in `prior/`; the current reuse/fit counts are recorded in the summary.

**Decision: INCOMPLETE.** Qualification requires mean monthly PR-AUC gain ≥0.005 and mean recall change ≥−0.01 at both budgets. **Criteria cannot be evaluated. No full-data performance conclusion is available.** The accepted LR remains at threshold 0.39.

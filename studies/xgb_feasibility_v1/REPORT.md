# Isolated Spark XGBoost feasibility

**BLOCKED for full-data training on the existing two-worker resources.** The
pinned CPU implementation passes the generated-fixture compatibility tests, but
the path preserving the accepted feature semantics requires at least **26.13 GiB**
of dense training arrays. This exceeds the approximately 11.7 GiB host before
JVM/native training overhead. Small-fixture success is not a full-data memory
qualification. No production runtime, feature, model or warehouse was changed.

## Contract and version

Inspected schema 3 / `m7-flighttime-v2`, the fitted preprocessing and saved LR/GBT
evaluations. Accepted bundle `d0098a9572834c53bf76b04d732a94a6` remains LR at 0.39.
Its training manifest records **8,946,370** rows, January 2023–April 2024.
The frozen pipeline `PipelineModel_196025fd5a4b` has five alphabetical indexers,
one-hot encoding, mean imputation and assembly: **739 categorical + 45 numeric,
count and missingness dimensions = 784**, from 50 predictors. It was loaded
read-only and never refitted. Positive/negative weights remain 2.0/1.0.
The production bundle loader recognizes LR/GBT only: XGBoost is not a drop-in
published bundle. Future integration would also need to apply the existing
validation-selected inclusive threshold to `probability[1]`, rather than reuse
XGBoost's native classification decision or inherit LR's 0.39 without selection.

Saved equal-month validation PR-AUC: accepted LR **0.428098323**; GBT all-feature
10/30/60-tree studies **0.418217032 / 0.423537897 / 0.427431474**. Neither approved
LR variant qualified either. These are existing records, not new evaluations;
no test-period data or predictions were read. [Input hashes](evidence/contract-review.json)
identify the contracts and saved records inspected.

**Runtime compatibility passes:** `xgboost-cpu==3.0.5` supplies
`xgboost.spark.SparkXGBClassifier` and works with the actual **Spark 3.5.9,
Python 3.10.12, Linux x86_64** base image. The study adds six
[hash-locked wheels](requirements.lock) in a separate image under `/opt/xgb-study`;
NumPy 2.2.6 and pandas 2.3.3 are inherited unchanged. PyArrow 19.0.1, SciPy 1.15.3
and scikit-learn 1.6.1 are explicit. This verifies this combination, not every
Python 3.10 platform or a future XGBoost release. The
[upstream API](https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html#xgboost.spark.SparkXGBClassifier)
supports Spark vectors, `weight_col`, two worker tasks and Spark-controlled CPU
threads. `pip check` passed in the isolated image.

## Exact generated tests

**18 runtime assertions passed across eight generated-data fits**, maximum
32,768 rows. Every fit recorded ranks 0/1, world size 2, on distinct worker hosts.
The replica retained one core per executor, 640m executor heaps, 768m driver heap
and `spark.task.cpus=1`; it used a separate network, no published ports and no
flight/warehouse mounts. [Results](evidence/results.json), execution log (historical original-repository file `evidence/fixtures.log`; omitted from this compact reconstruction)
and [test source](fixtures.py) contain the exact checks.

| Check | Observed result |
|---|---|
| Fitted preprocessing on 514 generated rows, including unknown categories/null numeric inputs | All finite, sparse, 784 dimensions; mean 36.74 nonzeros |
| Sparse versus dense training, same generated rows | Maximum probability difference **0.0** |
| Implicit sparse zero, explicit stored zero and dense zero | Identical predictions with `missing=NaN` |
| Zero versus NaN | Positive probabilities **0.200209 / 0.806121**; correctly distinct |
| Sparse optimization with `missing=NaN` | Rejected by the estimator |
| Sparse optimization with `missing=0` | Zero and NaN both score **0.5**; changes the accepted meaning |
| Class weights | One-step weighted/unweighted probabilities **0.401312 / 0.268941**, matching analytic Newton updates with 2:1/1:1 weights |
| Binary labels, probabilities, prediction columns, duplicate multiplicity | Passed; three identical occurrences of ID 7 remain three |
| Save/reload | All 514 prediction rows exactly equal; 784 dimensions and label/weight column parameters retained |

The sparse optimization cannot be used as a transparent memory fix. One-hot
off-values, zero history counts and zero numeric/missingness indicators are real
zeros in the accepted contract. The installed adapter explicitly requires
`missing=0` for its CSR optimization. Preserve `missing=NaN` and
`enable_sparse_data_optim=False`; do not double-weight with `scale_pos_weight`.
Model callbacks serialize through cloudpickle, so reload was verified only in
the pinned environment; portability across dependency versions is unproven.

## Resource finding

The pinned [adapter source](evidence/upstream-core.py.txt) expands Spark vectors
to float32 arrays. Its [data path](evidence/upstream-data.py.txt) retains all
training batches per XGBoost worker before constructing QuantileDMatrix.
An Arrow batch limit of 1,000 does **not** bound total retained training data.
Repartitioning to `num_workers=2` consolidates the data into two worker tasks;
more upstream Spark partitions and training subsampling do not remove this cost.

| Measurement / bound | Result |
|---|---:|
| Entire fixture application, including startup and probes before cleanup | 102.69 s |
| 4,096 / 32,768 generated-row fit | 5.75 / 9.61 s |
| Python worker RSS at final iteration, largest fixture | 299.8–300.3 MiB |
| Worker-container lifetime peaks | 1.26 / 1.31 GiB |
| Driver / master container peaks | 1,020 / 350 MiB |
| Full training dense arrays, `8,946,370 × 784 × 4` | **26.13 GiB total; 13.06 GiB/worker** |

Container peaks include daemon/executor/native memory and cache; they are not
simultaneous totals. Python high-water marks can include earlier fixtures.
Study container limits were 2 GiB per worker/driver and 512 MiB for the master;
zero cgroup OOM events occurred. Existing production containers have **no Docker
memory limits**: their 768m worker advertisements and 640m executor heaps do not
cap Python/native allocations. The physical host, not just JVM heap, is the
constraint. Full-data requirements exceed the dense lower bound; no precise
full-job runtime or sufficient enlarged-cluster size is claimed. See
[resource calculations and measurements](evidence/resources.json).

## Proposed bounded candidate and stop

[One proposal](candidate.json), **not executed or authorized**:
`xgb-cpu-hist-32-depth4-bounded`, at most 100,000 deterministically selected
training rows, accepted preprocessing, 32 trees, depth 4, learning rate 0.1,
max_bin 32, subsample 0.5, seed 42, weights 2:1, zero-preserving input. One fit,
30-minute cap including cleanup, the study memory caps, no retries/tuning.
Its dense input lower bound is 149.54 MiB per worker; the proposed workload itself
has not been tested. Any validation scoring requires separate review and excludes
the test period. A sampled-training result would not establish a fair full-data
LR comparison or qualify production promotion.

Fixture process exit was **0**. The initial supervisor exited **1** after cleanup
because its container comparison treated Docker's mount-list order as meaningful.
Repeated read-only inspections reproduced order-only differences. The checker
was corrected and **two offline regression tests pass**. The initial failed
[isolation record](evidence/isolation.json) is retained; its original full
before/after objects were not saved, so that specific difference cannot be
retrospectively proven. [Investigation](evidence/metadata-investigation.json)
records this limitation and lifecycle evidence; there was no fixture rerun.

[Preservation checks](evidence/verification.json) found all 1,093 pre-existing
tracked files and 64 fitted-preprocessing files unchanged by SHA-256, plus
unchanged size/mtime for 1,966 local artifact files (metadata, not full content
hashes). Study containers/network were removed; production containers remained
stopped and the production image ID is unchanged. No full-data fit, tuning,
weather ingestion, feature rebuild, test evaluation or production promotion
occurred. **Stop before further training; separate review is required.**

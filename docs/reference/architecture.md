# Architecture

**The September–December 2024 test period was previously inspected. It is historical evaluation, not a pristine external holdout.**

## Production dependencies

```text
Pinned BTS CSV -> preflight -> ODS -> DWD
                                     |-> DIM
                                     |-> analytical DWS -> ADS
                                     |-> ML features -> train / validate -> frozen model
                                                                         -> historical test
```

ODS retains the explicit 109-column BTS schema and lineage. DWD standardizes fields, schedule values, and the target. Dimensions, analytical daily DWS, and the canonical ML feature builder derive from DWD. ADS uses analytical DWS and DWD. Shared warehouse logic remains independent of ML.

`dws/ml_feature_contract.py` contains definitions only and imports neither Spark nor ML. The feature builder has separate selection and final-test phases. Selection reads only train/validation source rows; test materialization requires a completed production selection. Active ML modules are `model_contract`, `preprocessing`, `train`, `evaluate`, and `metrics`. Production and the experiment entrypoint call the same authoritative M7 selection implementation. M9 remains separate.

## Spark Standalone topology

Execution uses native ARM64 or x86-64 Linux containers with identical pinned dependency versions; Compose selects the host architecture. Historical full-data acceptance applies only to its recorded runtime. A healthy active Docker daemon with Compose v2 is accepted; Colima is the fallback (fresh bootstrap: 2 CPUs, 6 GiB, 40 GiB disk). This supersedes the old Colima-only requirement. Docker Compose defines one master, two workers, and one notebook/driver service on the `airline-spark` project network. The master listens at `spark://spark-master:7077` and exposes its UI on host port 8080. Each worker has one core and 768 MiB. The notebook service mounts `conf` and `src` read-only, and mounts `data`, `artifacts`, and `notebooks` for project output. It exposes JupyterLab on 8888 and the active application UI on 4040. Local overrides can change project identity and published ports; use `docker compose port` to inspect the active mapping.

The notebook driver is configured for 768 MiB, two maximum Spark cores, one core per executor, 640 MiB executor memory, 64 MiB file partitions, and four SQL shuffle partitions.

## Snapshot and quality boundaries

`conf/raw_data_manifest.csv` pins `bts-2023-2024-v1`. Before ODS, preflight checks two live registered workers, a successful Spark action, the exact monthly file inventory and filenames, manifest validity, and streaming SHA-256 hashes. Runtime code cannot refresh the expected hashes.

The fail-fast orchestration order is preflight; ODS + quality; DWD + quality; DIM + quality; analytical DWS + quality; selection features + quality; ADS + quality; train/validation selection and publication + quality; validation evaluation; test features + quality; test evaluation; final artifact quality. Cluster startup is separate.

Feature validation checks required source fields, eligibility/status domains, schema, chronology, history count/rate/missing domains, and eligible row reconciliation before writing. Ordinary full-data checks use exact snapshot counts and independently sampled histories without rebuilding features. Explicit comprehensive acceptance rebuilds into a separate location and compares all output columns with bidirectional `EXCEPT ALL`.

## Flight-time histories

Eligibility and row grain are unchanged. Convert scheduled departure from origin-local time to UTC using the pinned airport-ID reference and locked pytz rules. Reconstruct source arrival as scheduled departure UTC plus signed departure delay plus actual elapsed time. This assumes immediate arrival-outcome availability; actual publication/ingestion latency is unobserved.

For each entity and N=7,30,90, source departure must be in `[P−N*24h, P)` and source arrival must be strictly before target scheduled departure P. Implement this as a timestamp event sweep: a source contributes over `max(departure, arrival) < P <= departure+N*24h`. Integer-microsecond additions/removals encode strict comparisons. Aggregate simultaneous events before cumulative sums, then join unique entity/time lookup rows back with null-safe keys, preserving duplicate flights.

Invalid or ambiguous scheduled times produce empty target histories; invalid outcome times exclude sources. Missing airport mappings fail rather than guessing. Timestamp metadata and target actual outcomes never enter model vectors. Train/validation/test date boundaries remain locked. Selection histories exclude all test sources; final evaluation permits causal rolling test history after selection is frozen.

Feature layout uses `max(4, ceil(total_rows/250000))` with `repartitionByRange(num_partitions, "split", "flight_date")`, then `partitionBy("split")`. Three directory values do not imply three execution partitions or three files. DWD retains year/month partitioning.

## Training, evaluation, and publishing

Training reads only train, fits alphabetical StringIndexer/unknown handling, OneHotEncoder, train-mean imputation, and ordered VectorAssembler. Prepared vectors use `max(2, ceil(training_rows/250000))` partitions with `DISK_ONLY` persistence, released in `finally`. Fixed LR and GBT settings are listed in [M7 selection](model-selection.md). Exact rational validation F1 selects the threshold step and classifier; ties choose lower steps then LR. Baseline and ranking metrics do not influence selection. Publish the exact winning fit without train-plus-validation refitting.

A unique temporary sibling directory receives preprocessing and the selected LR or GBT classifier. Both are reloaded and checked for compatibility before `manifest.json` is written last. Replacement renames the old bundle to a backup, publishes the complete new directory, and removes the backup. A failed publication restores the previous bundle. Missing or incompatible manifests fail closed. This is sequential local-filesystem publication.

The manifest includes bundle identity, features/order, split boundaries, settings/threshold, training profile, snapshot ID/raw-manifest digest, Spark version, Git commit/dirty state when available, and a digest of production Python sources. Git metadata may be unavailable in a container without a checkout mount; the source digest is still recorded.

Evaluation explicitly selects validation or test, loads and validates the bundle, transforms, scores, and applies probability >= the selected integer step / 100, ignoring Spark’s generated prediction column. It contains no fitting, tuning, candidate comparison, or retraining. Predictions preserve date, split, target, scalar probability, and binary prediction. Metrics reconcile with natural-prevalence predictions and include continuous-score ROC/PR areas; single-class areas are undefined. Both reports identify the bundle and historical-test status.

The historical experiment reports remain evidence of earlier work. Newly generated metrics live only under the configured artifact root; no saved experiment artifact is needed to reproduce production.

## Revision contracts

Dataset/model schema 3 and feature definition m7-flighttime-v2 retain business-key metadata and last-written completion manifests. They validate the configured snapshot, phase, timezone policy and selected result, and reject schema 1/2 artifacts with rebuild/reselect guidance. `common/lineage.py` defines canonical JSON/hash; `dws/dataset_manifest.py` governs dataset completion. `ml/experiments.py` supplies selection shared by `ml/train.py`; the latter publishes the selected production bundle. Selection and test feature manifests are separate so test publication cannot change frozen train/validation provenance. See [pipeline rules](pipeline-rules.md).

The shared Spark session limits columnar cache batches to 1,000 rows for the 640 MiB executors. DISK_ONLY caches also deserialize batches on the heap; this bounds transient memory during staged history materialization without changing feature values or output partitions.

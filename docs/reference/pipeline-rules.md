# Pipeline rules

## Population, grain, and lineage

Eligibility is unchanged: non-cancelled, non-diverted flights with a binary arrival-delay target. Preserve one row per eligible source flight, including duplicates. `flight_id` identifies a business-key value, not necessarily a unique physical row. Report duplicate groups and excess rows; the pinned snapshot retains its zero-duplicate-key acceptance gate.

Canonical compact UTF-8 JSON has these ordered fields: `flight_date` (ISO yyyy-MM-dd), `reporting_airline` (unaltered string), then integer `flight_number_reporting_airline`, `origin_airport_id`, `dest_airport_id`, `crs_dep_time`. Include explicit nulls and standard JSON escaping; hash with SHA-256 to lowercase hexadecimal. Identifiers added for lineage never enter predictor vectors.

## Flight-time availability

Feature definition `m7-flighttime-v2` retains 50 predictors and uses dataset/model schema 3. Prediction time P is scheduled departure converted from origin-local time to UTC. Use the pinned airport-ID timezone reference and locked pytz database. `2400` means midnight on the following date. Ambiguous/nonexistent DST times and invalid schedule values have unknown prediction time; retain those target rows with count 0, null rate, and missing indicator 1. Unknown airport IDs fail the build instead of guessing offsets.

For a historical eligible flight h, reconstruct outcome availability A as scheduled departure UTC plus signed departure delay plus actual elapsed time. Missing/non-finite delay or elapsed time, negative elapsed time, or uncertain scheduled time excludes that source. This is an **actual-arrival event assumption with zero reporting latency**. Source publication/ingestion timestamps are unavailable; real operational availability is not proved.

For each carrier, origin, destination, and route history over N=7,30,90 days, require both:

- `P − N*24h <= scheduled_departure(h) < P`
- `A(h) < P`

The lower departure boundary is inclusive; both upper comparisons are strict. Same-day completed flights can contribute; same-day in-flight, overnight not-yet-arrived, simultaneous, and future flights cannot. Actual outcomes are usable as historical sources at scheduled-departure prediction time only under the zero-reporting-latency assumption, never as target predictors. Reconstructed arrival is an event time, not evidence of publication or ingestion. Proving operational availability requires additional timestamp data. Duplicate sources contribute with their original multiplicity.

Train remains 2023-01-01–2024-04-30; validation 2024-05-01–2024-08-31; test 2024-09-01–2024-12-31. Build train/validation features without test sources. After selection is frozen, build test features with causal rolling history, including earlier test outcomes only when A < P. The test was previously inspected and is not a pristine external holdout.

## Authoritative M7 selection

Production and experiment selection share one implementation. Fit preprocessing and exactly one weighted LR and one weighted GBT on all train rows; never refit on train plus validation. Fixed estimator parameters are documented in [M7 selection](model-selection.md). Positive-class weight 2.0 affects fitting only.

Compare thresholds s=0..100 using direct probability >= s/100 and integer confusion counts on validation. Compare F1 exactly as 2TP/(2TP+FP+FN), with zero denominator represented as 0/1, using cross multiplication. Across all candidate model/threshold pairs, choose highest exact rational F1, then lowest integer threshold step, then LR only when both F1 and step are equal. The constant training-prevalence baseline uses threshold 0.5, predicting positive on a tie; it never participates in selection. Continuous-score ROC/PR areas are reporting metrics only.

Publish the exact selected fitted classifier and preprocessing model. The integer threshold step is authoritative; no fixed production model or operating threshold remains. Evaluation loads the frozen bundle and cannot fit, tune, or select. Report natural-prevalence confusion and ranking metrics.

## Completion and migration

Write `_dataset_manifest.json` last after Parquet reload validation; invalidate an old marker before overwrite. Record phase, split inventory, ordered types, snapshot, duplicates, time policy/reference version, and unknown-time counts. Loading also reconciles the declared positive integer row count with the loaded data; missing input parts require a feature rebuild rather than partial evaluation. Test features identify the selection bundle and run. Selection feature provenance remains immutable when test features are added.

Run IDs are exclusive; retry incomplete selection under a new ID. Selection saves both models and verifies reload agreement before its completion manifest. Publish production through a staged save/reload and atomic local-directory replacement with rollback. Evaluation fingerprints the input Parquet, dataset manifest, saved preprocessing/classifier contents and model/selection manifests. Cache reuse also requires completed, unchanged prediction files, the expected schema and threshold decisions, and bidirectional lineage/label multiset equality with the evaluated split. Reports contain versioned completion evidence and a report digest. Legacy or invalid evaluation output is regenerated with the frozen model and threshold, without fitting or tuning. Repair revokes the old report before scoring; predictions are staged, reloaded and validated before directory publication, and the report is atomically replaced last. Interrupted writes cannot constitute completion. This is a sequential local-filesystem contract.

Versions 1 and 2 are incompatible. Preserve old evidence, rebuild features, reselect, and regenerate predictions; never wrap old outputs in new manifests. Default schema-3 paths are separate from prior outputs.

Prepared training partitions remain max(2, ceil(rows/250000)); feature writes remain max(4, ceil(rows/250000)). Reused feature/training frames retain DISK_ONLY persistence and finally cleanup. Evaluation writes its prediction frame once, then validates and computes metrics from the reloaded Parquet without persisting the one-use frame. Small ADS outputs and analytical DWS are unchanged.

## Runtime and evidence

Native runtime targets are Linux ARM64 (aarch64) and x86-64 (x86_64), with the same pinned Python, Spark and dependency versions. Compose selects the host architecture; emulation is unnecessary. Historical full-data acceptance applies only to its recorded runtime. Accept a healthy active Docker daemon with Compose v2; use Colima only as fallback. This project decision supersedes the older Colima-only requirement. Fresh Colima bootstrap uses 2 CPUs, 6 GiB memory, and a 40 GiB disk; existing healthy runtimes are not resized or switched.

Historical results do not validate a new execution. Record commands, exit codes,
source/runtime identities, and limitations for each acceptance run. Keep execution
records bound to the source and runtime that produced them.

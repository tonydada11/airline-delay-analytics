# LR-3h production promotion readiness — BLOCKED

Candidate `ca23fe0`; reviewed without executing Spark, fitting, scoring, rebuilding features, evaluating test data or changing production. **The validation experiment passes; production promotion is blocked by missing versioned integration and unproved operational availability.** This report proposes changes only.

## Evidence and current incompatibilities

The candidate's four-month mean PR-AUC is **0.46373486139457926**, versus frozen LR **0.4280983226531411**; paired gain **+0.035636538741438184**. Mean recall changes are **+0.019339208632215853 / +0.02416794834284132** at 10%/20%. Its one fit, 790-coordinate prefix checks, complete validation score reload, shutdown and cleanup passed. These are saved experimental results, not new evaluations in this review. See [candidate report](../lr_3h_v2/REPORT.md).

Read-only byte verification in this review matched **all 50 candidate inventory files (390,636,662 bytes), all 80 qualified feature Parquet files, and all 64 accepted preprocessing files**. Existing evidence establishes exact 8,946,370 train / 2,418,739 validation populations and original-column/label multiplicity; this review did not recompute populations or predictions. Accepted production remains bundle **`d0098a9572834c53bf76b04d732a94a6`**, schema **3**, feature version **`m7-flighttime-v2`**, LR at **0.39**.

Specific blockers found in current code:

- [`ml_feature_contract.py`](../../src/airline/dws/ml_feature_contract.py), [`dataset_manifest.py`](../../src/airline/dws/dataset_manifest.py) and `validate_feature_frame` require the old exact output columns. They reject the six appended columns; the study files have no production dataset completion manifest.
- [`build_ml_flight_features.py`](../../src/airline/dws/build_ml_flight_features.py) uses the same eligible, labeled flights for history sources and queries. Its eligibility requires observed cancellation/diversion/label values, and chronological split assignment stops in 2024. It cannot directly construct features for unlabeled future schedules. This affects legacy long histories as well as the new 3h histories.
- [`preprocessing.py`](../../src/airline/ml/preprocessing.py) produces the frozen 784 coordinates only. Expanding its global numeric list/refitting would not reproduce the candidate. The two new means exist in `append-contract.json`; no production append transformer loads them.
- [`model_contract.py`](../../src/airline/ml/model_contract.py) validates schema 3, legacy stages/width, and a selection manifest containing **both LR and GBT** evidence. The candidate is one fixed LR experiment, not that selection protocol. `load_bundle` cannot load it, and `publish_bundle` would reject the 790d classifier paired with the old preprocessing. The study output has **no standalone bundle manifest** or colocated frozen preprocessing.
- [`evaluate.py`](../../src/airline/ml/evaluate.py) scores `preprocessing.transform(features)` directly, missing the six-coordinate append; its output requires target/split columns. [`evaluation_artifacts.py`](../../src/airline/ml/evaluation_artifacts.py) binds only preprocessing/classifier plus existing manifests unless callers explicitly include new artifacts. Means must participate in identity/cache invalidation.
- **The saved LR model's internal default threshold is 0.5.** The experiment applied `probability >= 0.39` externally. Using its raw `prediction` column would silently change the decision rule.

## Temporal decision required before live release

For scheduled-departure UTC cutoff **t**, the qualified policy admits a source with scheduled departure **d** and reconstructed arrival **a** only when **t−3h ≤ d < t AND a < t**. `a = d + signed departure delay + actual elapsed time`. Windows are departure-anchored, not arrival-anchored. Source origin matches target origin; source destination matches target destination. Preserve integer-microsecond events at `max(d,a)+1` and `d+3h+1`, tie collapse, independent duplicate contributions and null-safe query joins from [the qualified sweep](../track_a_3h_v1/features.py). Empty histories are `(null, 0, 1)`; uncertain target times retain empty histories; unusable source times exclude contributions. Keep the pinned timezone map, `2400` rollover, DST behavior, finite signed delay, nonnegative elapsed time and overflow guards from [`flight_time.py`](../../src/airline/dws/flight_time.py).

**Zero reporting latency is an accepted research assumption, not measured receipt availability.** Existing DWD has no demonstrated publication/ingestion timestamps or historical revision log. Arrival before t does not prove that the arrival-delay label and timing fields were available before t. The 3h window is especially exposed to delayed reporting. This limitation also applies to the original histories; preserving the old bundle does not resolve it.

For strict live correctness, obtain versioned source outcomes and schedule snapshots as received. Admit only versions actually available before the cutoff, conservatively requiring `max(a, received_at) < t` as well as `d < t`; exclude unknown receipt times. Later corrections must not rewrite earlier predictions. Apply availability rules to **all** historical coordinates, not just the six additions. Verify feed coverage, clock semantics and missing/late behavior; do not invent a reporting-lag constant or treat a waiver as timing evidence. A receipt-aware policy is a **different feature policy/version** and needs separately authorized replay/qualification; the current +0.0356 gain cannot be transferred to it without evidence.

The candidate cutoff is scheduled departure t. If a prediction is issued at an earlier actual request time p, it may use only information available before p; it cannot use later information merely because it precedes t. An earlier-cutoff service needs an explicit versioned window/cutoff contract and qualification. Freeze schedules as known at cutoff, not retrospective schedule corrections. Live target rows must not require their eventual outcomes or cancellation/diversion status.

## Minimum versioned file changes (proposed; none implemented)

Use an additive **schema 4 / `m7-flighttime-3h-v1`** path for the candidate's existing assumption. Keep all schema-3 globals, default paths and accepted bundle bytes unchanged. A receipt-aware change needs a distinct policy identity. Proposed minimum files:

| Exact file to add | Required responsibility |
|---|---|
| `src/airline/dws/ml_feature_contract_v4.py` | Define old named predictors plus exactly six ordered additions, 64-column labeled layout matching saved study Parquet, separate unlabeled inference layout, types/domains, UTC cutoff and availability policy IDs. Training/evaluation boundaries remain frozen; future targets are not forced into old train/validation/test dates. |
| `src/airline/dws/build_ml_flight_features_v4.py` | Separate target schedules from eligible observed history. Reuse existing schedule/UTC functions; generalize interval queries for the existing four-entity 7/30/90d histories and origin/destination 3h only. Preserve legacy values on identical replay inputs, row multiplicity and null behavior. Record cutoff/as-of source versions and history completeness; do not interpret a missing history feed as a genuinely empty airport history. Retain enough source history for the legacy 90d windows. |
| `src/airline/dws/dataset_manifest_v4.py` | Validate versioned labeled/inference schemas and write completion last. Bind source snapshot/inventory, feature/time policy, timezone hash, query cutoff, history coverage, row/duplicate/time audits and phase. Do not reuse schema-3 manifests or imply that current monthly study folders already satisfy that contract. |
| `src/airline/ml/preprocessing_v4.py` | Load the exact frozen 784d PipelineModel, preserve its order, and append six coordinates using a strict saved means document. No `.fit`, refitted indexers, reordered numeric lists, or serving-time statistics. Validate 784-prefix/790 total and invalid input behavior. |
| `src/airline/ml/model_contract_v4.py` | Validate/load an LR-only frozen-study selection contract, not a fabricated LR/GBT competition. Require two classes, 790d classifier, accepted settings/weights, parent bundle identity, exact preprocessing/coordinate order, finite means, threshold step 39/denominator 100, dataset/source/runtime/code fingerprints and all artifact hashes. Reject partial, mixed-version and altered bundles. Preserve the old loader unchanged. |
| `src/airline/ml/evaluate_v4.py` | Provide a label-free scoring API and a separate labeled evaluation path, both using the loaded append transform and continuous score followed by explicit `>= 0.39`. Bind means and complete transform identity via `evaluation_identity`'s existing directory/manifest maps. Reuse exact PR-AUC (`numBins=0`) and threshold evidence; bring the pure study budget helper into this versioned module without study-path/import side effects. No model fit or new threshold selection. |
| `src/airline/ml/release_lr_3h_v4.py` | Package the **existing** fitted candidate into a new immutable bundle ID/directory: byte-copy classifier and frozen preprocessing; persist means, inventories, threshold/selection evidence and completion manifest last. Verify staged artifacts/reload before sealing. Never invoke `train.py`/`experiments.select` (they refit preprocessing and LR/GBT), or replace `artifacts/flighttime-v3/models`. |
| `conf/project_lr_3h_v4.yaml` | An explicit opt-in path profile with the existing config keys, separate v4 dataset/artifact roots and unchanged resource settings. Existing `ProjectConfig`/`ProjectPaths` can resolve these keys; no change to `conf/project.yaml` or default routing. Inference target/history/output arguments belong to the new CLI. |
| `tests/integration/test_ml_flight_features_v4.py` | Boundary/as-of/multiplicity, unlabeled future targets, legacy-feature parity, source correction/receipt, missing-feed and cutoff tests. |
| `tests/unit/test_model_contract_v4.py` | Strict schema/order/mean/threshold/hash validation, negative mixed-version/tamper cases and independent production imports. |
| `tests/integration/test_ml_inference_v4.py` | Frozen transform/prefix, null means, unseen categories, saved candidate reload, unlabeled scoring, labeled population/metrics and evaluation-cache identity. No test fixture may start a replacement candidate fit. |
| `docs/PIPELINE_RULES.md`, `docs/RUNBOOK.md`, `README.md` | Document both versions, availability limitations, explicit path routing, acceptance record, immutable artifact transfer, rollback and historical-test status. Preserve historical records. |

No change is needed to existing LR settings, weights, metric definitions or default production loader/builder. The new release command owns v4 acceptance/publication. Existing `scripts/package_evidence.py` remains schema-3-only; do not weaken it to certify v4.

Persist the exact trained values in a hashed `append_means.json` together with training dates/count, parent preprocessing hash and source `append-contract.json` hash:

| Appended coordinate order | Serving handling |
|---|---|
| `origin_delay_rate_3h` | NULL → **0.08079999358628864** |
| `origin_flight_count_3h` | Observed bigint, zero retained |
| `origin_history_missing_3h` | Observed binary double |
| `destination_delay_rate_3h` | NULL → **0.0796816299126744** |
| `destination_flight_count_3h` | Observed bigint, zero retained |
| `destination_history_missing_3h` | Observed binary double |

Round-trip means without numeric rounding; reject missing/extra keys, NaN/infinity, invalid rate/count/flag combinations and wrong order. The candidate coalesces NULL rates only; do not silently introduce a new NaN-imputation policy. The release bundle must be self-contained with the exact copied parent preprocessing; an implicit machine-local study path is insufficient.

## Acceptance gates and release sequence

1. **Resolve operational availability first.** Freeze issue time, schedule/outcome receipt/version semantics and feed coverage. Unknown availability cannot be inferred from reconstructed arrival. A receipt-aware or earlier-cutoff policy requires distinct versioning and separately authorized qualification; current gains do not automatically transfer.
2. **Implement/review the additive path and commit tested changes.** Acceptance: independent inequality-oracle parity at exact microsecond boundaries; future/current/in-flight and late-receipt exclusion; signed/invalid/DST times; null entities/times; duplicates; ordering; cross-month/year history; corrections and future-label/schedule perturbation invariance. Unlabeled future schedules must survive; unavailable feeds must not masquerade as empty histories.
3. **Replay the saved candidate without fitting**, under its unchanged assumed-arrival policy. Require exact 8,946,370 / 2,418,739 original-column/label multisets, 784-prefix and six-coordinate equality, unchanged means, binary 790d model/settings and full validation probability/prediction multiset equality at 0.39. Reconcile four monthly PR-AUC/budget results and the saved 101-step grid, including threshold ties and fractional duplicate-boundary mass. No baseline refit/rescore or new threshold selection.
4. **Seal a separate immutable bundle after artifact acceptance.** Verify byte hashes after transfer on the pinned runtime; fresh-process loading independent of study mounts; tamper/missing/partial/mixed-version rejection; means-sensitive evaluation-cache identity; v3 compatibility fixtures in separate output paths; successful shutdown and resource measurements. Do not treat worker peaks near 2 GiB as spare headroom or a live latency/throughput guarantee.
5. **Obtain independent release evidence and explicit approval.** May–August was reused for candidate/threshold selection across studies. September–December 2024 was previously inspected: this candidate's non-access does not restore pristine holdout status. A later authorized historical test is descriptive, frozen-model evidence, with no tuning. Operational confirmation needs an untouched later/prospective, as-of-valid period and a protocol frozen before labels are examined. None is evaluated here.
6. **Only after all gates pass**, separately authorize opt-in routing to the new profile/bundle at fixed 0.39. Keep v3 bytes/default profile for rollback, bind acceptance to source/runtime/artifacts and roll back by routing to the old bundle. No automatic retraining, threshold tuning or resource fallback.

## Artifact reproducibility and review boundary

Saved artifacts are locally recoverable now; Git contains their inventories/evidence, **not** the model/feature binaries. A checkout alone cannot reproduce scoring. The candidate's classifier default threshold 0.5, external append code/means, parent preprocessing dependency, missing production manifests and local paths must all be resolved by the versioned package. Artifact replay is reproducible; bit-identical retraining across runtimes/partitions is not established or needed to package this fixed model.

Byte anchors checked in this review (SHA-256):

- [Candidate inventory](../lr_3h_v2/evidence/artifact-hashes.json): `1dc0c88d7936623c49bed77645cd4ec78b511ca70d718b7b8015ec23b81454f9`.
- [Append contract](../lr_3h_v2/evidence/append-contract.json): `1ded5468f8a89a570c8e3a0dbba4a9378d0f9eb437f42e5103a51ede7d6da952`.
- [80-file feature inventory](../track_a_3h_v1/evidence/output-files.json): `40bb577c427188ba2d28d78947137bd950b7979410f6050623ab63209d8ddeb6`.
- [64-file preprocessing inventory](../track_a_3h_v1/evidence/preprocessing-sha256.json): `33ac4324ed06f13d4af3113579f136ce0a59723511fbcfc22579b82c5d4030cc`.
- Existing production `artifacts/flighttime-v3/models/manifest.json`: `7e1d0a1a9336dd383da9802db02586e23fcc411a4cd79294ee52c554c6d49ee7`.
- Saved candidate classifier coefficient Parquet: `762942a278e13231cf94753fdbe8bffdcb4ea42e4a44528c02e008aa8d0b9c32` (path and size in candidate inventory).

Review method: source/report/JSON/metadata inspection; Python standard-library SHA-256 and size comparison for each recorded file. No new Spark session, model loading/scoring, fit, feature build or test-period data read. No acceptance tests above were executed in this review; they are release requirements. Only this report is committed. **Stop: production remains the accepted 784-coordinate LR at 0.39; no promotion or main-branch merge.**

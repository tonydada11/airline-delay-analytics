# Offline LR-3h finalization — PASS

Existing model `offline-lr3h-ca23fe0-v1` is preserved in the stable versioned local directory:

```text
/home/server/workspace/industry/Airline/artifacts/offline/offline-lr3h-ca23fe0-v1/
```

All **85 files**, including hidden Spark CRC files and the completion manifest, match the immutable bundle from `93f0c58` byte-for-byte. All files/directories remain read-only. No bundle contents, loader, preprocessing, means, feature order or classifier were changed. The original study bundle and scored outputs are retained unchanged. Binary artifacts remain local and ignored by Git; a clone requires separate transfer of the entire verified bundle. [Artifact provenance and full SHA-256 inventory](evidence/artifact.json).

Bundle manifest SHA-256: `9157123a82926767a0ac69b6c1f9ab04899d0ad86ec32c95b42ffe46809a564f`. Previous full-validation output completion SHA-256, unchanged: `ad4ee1c6445008158d0dea0d179803d81b0c2e29d599dacf2f717fa655fc11f7`. Smoke Parquet SHA-256: `774e430b6bbb3bba8566f45263d78d98c5137a35e13b95944a37ce5025d69129`. [Smoke output hashes](evidence/output-inventory.json).

## Usage and smoke verification

The [offline usage guide](../OFFLINE_LR3H.md) contains the complete command, environment setup and cleanup. In its explicitly opt-in pinned cluster, run the **existing immutable scorer**:

```bash
docker compose -f conf/docker-compose.offline-lr3h.yml exec -T driver \
  timeout --signal=TERM --kill-after=15 14100 \
  /opt/spark/bin/spark-submit --driver-memory 768m /bundle/offline.py \
  --bundle /bundle --input /input --output /output/scores
```

`OFFLINE_FEATURES` selects an absolute directory of already-computed compatible Parquet rows; `OFFLINE_OUTPUT` selects a new run directory. The bundle/input are mounted read-only. Original predictors, the six approved 3h columns and business keys are required; labels are optional. Existing output paths are rejected. Scoring retains the frozen 784-prefix and saved means, appends the fixed six coordinates, and explicitly applies probability >= 0.39. No production route or default is consulted or changed.

**Exactly one small smoke invocation passed:** 16 original rows from one pinned May 2024 feature Parquet file; labels/split omitted before scoring. It loads the stable bundle and reuses its unchanged `load`, `prepare` and `scores` helpers. Final width 790, model ID, feature policy, exact saved means, threshold 0.39 and all saved probability/prediction values were verified. Exact multiset equality includes flight ID and all business keys. Results were written/read at `artifacts/offline-runs/lr3h-finalize-smoke-v1/scores/`. The saved reference was filtered to the 16 IDs; no full model scoring or full validation rerun occurred. [Smoke evidence](evidence/smoke.json).

Exact smoke commands from the execution checkout:

```bash
export OFFLINE_FEATURES="$PWD/studies/track_a_3h_v1/runtime/output/features/validation-2024-05"
export OFFLINE_OUTPUT="$PWD/artifacts/offline-runs/lr3h-finalize-smoke-v1"
mkdir -p "$(dirname "$OFFLINE_OUTPUT")"
mkdir "$OFFLINE_OUTPUT"
docker compose -f conf/docker-compose.offline-lr3h.yml -f studies/lr_3h_offline_finalize_v1/compose.smoke.yml config --quiet
docker compose -f conf/docker-compose.offline-lr3h.yml -f studies/lr_3h_offline_finalize_v1/compose.smoke.yml up -d
docker compose -f conf/docker-compose.offline-lr3h.yml -f studies/lr_3h_offline_finalize_v1/compose.smoke.yml exec -T driver timeout --signal=TERM --kill-after=15 600 /opt/spark/bin/spark-submit --driver-memory 768m /smoke.py
docker compose -f conf/docker-compose.offline-lr3h.yml -f studies/lr_3h_offline_finalize_v1/compose.smoke.yml down --timeout 15
```

The smoke-only override requires the original saved output mounted read-only. It is not needed for ordinary batch scoring. Static checks: the existing two no-fit/fixed-contract unittest checks passed in the reference checkout; compile and Compose validation passed. [Checks](evidence/checks.json).

Runtime: **96.823 seconds including shutdown**; one Spark application with two one-core executors, unchanged Spark 3.5.9 / Python 3.10.12 image and existing resource limits. All 54 tasks succeeded, no spill or OOM events. Lifetime cgroup peaks MiB: driver 1030.2, master 405.6, worker1 803.2, worker2 879.2. All study containers/network removed; prior stopped containers unchanged. [Execution](evidence/execution.json), [resources](evidence/resources.json), [preservation](evidence/preservation.json).

## Review scope and stop

Main-based review branch: **`study/lr-3h-offline-review-v1`**, base `ab6bd50`. Runtime configuration and smoke test were committed before execution as `9193a71` in the preserved reference-based execution checkout. The minimal review implementation is `e825c94`; its three executable/configuration files have exactly the same bytes as those tested. Only documentation source links were adapted to frozen GitHub references. [Code identity](evidence/review-code.json).

The review branch contains only the opt-in Compose file, offline usage guide, two ignore entries, focused smoke test/override and this finalization evidence. It excludes unrelated historical study ancestry. Production schema-3 code, configuration, defaults and fitted artifact hashes remain unchanged; accepted bundle `d0098a9572834c53bf76b04d732a94a6` remains the accepted model at threshold 0.39. No fits, tuning, preprocessing refits, feature builds, new dependencies, full validation/test evaluation or external acquisition.

This remains offline research under the original zero-reporting-latency assumption. It does not prove live receipt/schedule-version correctness or independent predictive performance. No production routing change or main merge occurred. **STOP for review.**

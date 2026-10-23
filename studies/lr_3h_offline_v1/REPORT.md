# Offline LR-3h packaging and batch scoring — PASS

Scope: Steps 1–3 only, candidate `ca23fe0`, qualified input `88b632f`, promotion review `c196317`, availability limitation `8b670c2`. Tested implementation committed before execution as `c4a13f3`; standalone Compose correction committed and config-validated as `10509eb` before continuation. The final-verifier sidecar correction was committed as `b52b130`, and container-owned file permission publication correction as `81ea65d`; both were committed before their respective continuations. **Zero classifier fits, zero preprocessing refits, no history rebuild, threshold reselection, independent evaluation, external acquisition, test-period evaluation or production changes.**

Offline model ID: **`offline-lr3h-ca23fe0-v1`**. Contract `airline-offline-lr3h-v1`; feature policy `research-scheduled-utc-zero-reporting-latency-v1`. New self-contained bundle: `studies/lr_3h_offline_v1/runtime/output/bundle/`. Scored results: `studies/lr_3h_offline_v1/runtime/output/scored-offline-v1/`.

**Artifacts:** the classifier and accepted preprocessing were byte-copied, not resaved in the release bundle. All 50 candidate inventory files and 64 parent preprocessing hashes were verified during packaging; all 80 qualified feature files were checked before execution. The bundle contains its own stdlib/PySpark loader/scorer, complete file inventory, copied fitted models, saved means and frozen result/threshold/input evidence. Runtime loading has no original-study-path dependencies. SHA-256 checks reject altered files/means; fixed schema/order/790 width/settings and explicit 0.39 are validated separately from schema 3.

**Feature order:** preserve the frozen 784 coordinates, then append:

| Coordinate | Type / handling |
|---|---|
| `origin_delay_rate_3h` | Nullable double; NULL → 0.08079999358628864 |
| `origin_flight_count_3h` | Observed nonnegative integer count; zero retained |
| `origin_history_missing_3h` | Observed binary double |
| `destination_delay_rate_3h` | Nullable double; NULL → 0.0796816299126744 |
| `destination_flight_count_3h` | Observed nonnegative integer count; zero retained |
| `destination_history_missing_3h` | Observed binary double |

Only saved nullable-rate means are applied. No means are recomputed, rounded or estimated from validation. Inputs require the 50 original predictors, six additions and six business keys. Flight_id, split and target are optional and preserved when supplied. Output adds positive-class probability, explicit `prediction = probability >= 0.39`, model_id and feature_policy. The classifier internal 0.5 decision is discarded.

**Verification:** focused pinned-runtime tests passed for 16 rows, exact prefix/order/null handling, label-free scoring, 17 duplicate occurrences, 0.39 boundary inclusivity, classifier save/reload, score parity and altered mean/order rejection. A read-only nested helper mount failed during initial test setup before Spark; the mount was corrected and its record retained. Local static checks confirm zero `.fit` call sites.

Full source split counts: `{'train': 8946370, 'validation': 2418739}`. Every validation vector verified its exact 784-prefix, 790 width and six appended values against an independent array-concatenation expression. Saved original continuous scores and predictions matched in both multiset directions, including business keys/labels/duplicates. A separate fresh Spark process, with **only the bundle, precomputed validation inputs and output volume mounted**, scored all 2,418,739 rows; final bidirectional prediction parity passed. The old production schema-3 bundle loaded successfully without scoring and remained unchanged.

**Saved-result reconciliation (not independent performance evaluation):**

- Candidate equal-month PR-AUC: **0.46373486139457926**; baseline **0.4280983226531411**.
- Paired PR-AUC gain: **0.035636538741438184**.
- Mean recall changes, top-10%/20%: **{'10': 0.019339208632215853, '20': 0.02416794834284132}**.

| Saved month | Rows | PR-AUC | Paired PR-AUC gain | Recall at 10% | Recall at 20% |
|---|---:|---:|---:|---:|---:|
| 2024-05 | 599226 | 0.4754740513825120 | +0.0430191447964838 | 0.2190730117010791 | 0.3819277717405039 |
| 2024-06 | 601272 | 0.4294257326030069 | +0.0258891806570458 | 0.2065223915435660 | 0.3656407349361432 |
| 2024-07 | 613938 | 0.5115979635514667 | +0.0410107267089802 | 0.2097874797239711 | 0.3721001842025679 |
| 2024-08 | 604303 | 0.4384416980413315 | +0.0326271028032429 | 0.2275124813132880 | 0.3969128141483090 |

All four monthly results and the complete saved 101-step threshold record are bound inside the bundle; the existing comparison helper reconciles the frozen evidence. No new monthly model evaluation or threshold selection was run. Exact prediction parity establishes reproduction of the saved budget/tie/duplicate results.

**Execution/resources:** full verification controller wall 1341.240s including corrections and all sequential cluster cleanup (focused tests separately 143.379s). Pinned Spark 3.5.9/Python 3.10.12 image `sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf`; two one-core workers, 640 MiB executor heaps, 768 MiB driver, four shuffle partitions, unchanged 2 GiB worker/driver and 512 MiB master limits. No overlapping heavy jobs. Failed tasks: 0; completed application-end events: 4.

Task totals: shuffle_read_bytes: 4.914 GiB, shuffle_write_bytes: 4.914 GiB, Disk Bytes Spilled: 0.000 GiB, Memory Bytes Spilled: 0.000 GiB. Memory-spill volume is not peak RAM.

| Phase | Seconds including teardown | Master / worker1 / worker2 / driver lifetime cgroup peaks MiB |
|---|---:|---|
| stage (exit 0) | 349.3 | 258.6 / 2048.6 / 2048.0 / 977.0 |
| fresh (exit 0) | 266.5 | 266.6 / 2048.2 / 2049.5 / 843.8 |
| final (exit 1) | 93.1 | 268.0 / 1312.8 / 1303.2 / 683.8 |
| final (exit 0) | 114.5 | 257.0 / 1660.1 / 1523.0 / 774.4 |

Workers reached their unchanged 2 GiB cgroup limit with memory.max events but zero OOM/kill events; this run does not establish memory headroom. Cgroup peaks include file cache; sampled phase memory is a lower bound, not process RSS or spare-capacity proof. Per-task/phase details: [resources](evidence/resources.json), [memory samples](evidence/memory.json), [execution](evidence/execution.json), [preservation](evidence/preservation.json).

## Exact commands and artifact identities

Tested before execution:
```bash
python3 -m unittest studies.lr_3h_offline_v1.test_no_fits -v
python3 -m py_compile studies/lr_3h_offline_v1/*.py
docker compose -f studies/lr_3h_offline_v1/compose.focused.yml up -d
docker compose -f studies/lr_3h_offline_v1/compose.focused.yml exec -T driver timeout --kill-after=15 1200 /opt/spark/bin/spark-submit --driver-memory 768m /study/verify.py --mode focused
docker compose -f studies/lr_3h_offline_v1/compose.focused.yml down --timeout 15
```

After committing implementation:
```bash
python3 studies/lr_3h_offline_v1/run.py
# 10509eb: YAML fix; standalone scoring completed, verifier sidecar check failed
python3 studies/lr_3h_offline_v1/run.py --finish
# b52b130: reuse scores; final verifier passed, host chmod failed
python3 studies/lr_3h_offline_v1/run.py --finish
# 81ea65d: validate existing results and publish read-only; no Spark launched
python3 studies/lr_3h_offline_v1/run.py --finish
```
The controller ran these payloads, each wrapped by timeout and followed by `docker compose ... down --timeout 15`:
```bash
# compose.yml: staged artifact, full vector/score parity and legacy loader checks
/opt/spark/bin/spark-submit --driver-memory 768m /study/verify.py --mode stage
# compose.fresh.yml: independent bundle-only scoring process
/opt/spark/bin/spark-submit --driver-memory 768m /bundle/offline.py --bundle /bundle --input /input/2024-05 /input/2024-06 /input/2024-07 /input/2024-08 --output /output/scored-offline-v1
# compose.yml: compare fresh output; no fit or threshold selection
/opt/spark/bin/spark-submit --driver-memory 768m /study/verify.py --mode final
# Offline resource/hash aggregation only
python3 studies/lr_3h_offline_v1/report.py
```

The initial staged full-data check passed, then the fresh Compose file was rejected before launch because of an unquoted trailing-colon PYTHONPATH. Evidence is retained in initial-execution.json and standalone-config-failure.json. The syntax was corrected, all Compose files validated, and the already-verified staged bundle was continued without rerunning that stage, refitting or rebuilding data. The first final verifier then completed the Spark multiset comparisons but exited because it looked for `completion.json` instead of the actual `_completion.json`. Its corrected rerun passed without rescoring. Host chmod then failed on container-owned copied files; the same pinned image applied read-only modes with no byte changes and no further Spark execution. All failed-process evidence is retained, including the exit-1 verifier phase. Four Spark applications ended cleanly; no failed Spark tasks were observed. Reported wall time includes these correction intervals.

For later opt-in scoring, mount the published `bundle/` at `/bundle` instead of its staging name, precomputed input directories read-only, and a new writable output directory in the same pinned cluster configuration; run the bundle-only command above with a **new** output path. See [README](README.md). Incomplete bundles lack a completion manifest; staging completion follows load/full-score verification, and final immutable bundle publication occurs only after fresh-process parity. Existing outputs are never overwritten.

Bundle manifest SHA-256: `9157123a82926767a0ac69b6c1f9ab04899d0ad86ec32c95b42ffe46809a564f`. Output completion SHA-256: `ad4ee1c6445008158d0dea0d179803d81b0c2e29d599dacf2f717fa655fc11f7`. [Bundle hashes](evidence/bundle-inventory.json), [output hashes](evidence/scored-inventory.json), [final integrity recheck](evidence/final-integrity.json). Model/preprocessing files and final bundle are read-only; binary files remain local and ignored by Git. Verify the recorded manifest hash when transferring the whole self-contained directory.

**Remaining limitation:** this is an offline research bundle using precomputed histories. At scheduled departure UTC cutoff t, the new windows require t−3h ≤ source scheduled departure < t and reconstructed source arrival < t. Existing 7/30/90-day predictors retain their frozen research construction. Reconstructed arrival is assumed to make the outcome available immediately; it is not measured receipt time. It proves neither actual outcome receipt, schedule-as-known timing, correction availability, live coverage nor independent future performance. September–December 2024 is previously inspected and not pristine. Nothing here clears the operational BLOCKED finding or authorizes routing changes.

**STOP:** Steps 1–3 end here. Accepted schema-3 LR bundle `d0098a9572834c53bf76b04d732a94a6`, 784 dimensions, default threshold 0.39 and production paths remain intact. No promotion, automatic switch, main merge or further experiment.

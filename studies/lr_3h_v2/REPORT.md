# Controlled six-feature LR experiment — PASS

Reference `88b632f`; implementation committed before execution as `c6921d4`. Branch `study/lr-3h-v2` preserves the prior blocked-run commits. Production code/artifacts and resource limits were preserved; the accepted baseline was never refit or rescored, and no test-period rows were read.

**Execution:** one exclusive full-data attempt, 2790.449s including cleanup; cap 14,400s. Exit 0. Study containers/network removed: True. Original files/containers preserved: True.

**Preflight:** existing Airline host Docker daemon and pinned image verified; all 80 saved feature file hashes, 64 preprocessing hashes and frozen baseline-evidence hash matched. Focused tests passed on actual Spark 3.5.9/Python 3.10.12: imports, 16-row frozen 784d preprocessing, null-rate means, exact six-coordinate order/790d width, preprocessing save/reload, distributed duplicate-budget fixture. Three local protocol tests also passed. The first fixture had an unstable unmaterialized limit; its failure evidence is retained, the fixture was cached/fixed and tested before commit. Preflight started zero classifier fits.

**Population:** 20/20 months passed exact bidirectional exceptAll over every original column, including business keys and labels, plus feature schema/domain checks. Required final counts: 8,946,370 train and 2,418,739 validation. Recorded final counts: {'validation': 2418739, 'train': 8946370}. Full prepared rows verify exact 784-value original prefixes and total width 790 before fitting.

**Features:** frozen accepted PipelineModel reused without fit; only the origin/destination nullable rate means use training rows. Appended order: `origin_delay_rate_3h`, `origin_flight_count_3h`, `origin_history_missing_3h`, `destination_delay_rate_3h`, `destination_flight_count_3h`, `destination_history_missing_3h`. Original types are nullable double rates, bigint counts and double 0/1 missing indicators. Qualified historical values were reused verbatim under the accepted zero-latency availability assumption; no histories were rebuilt.

Training-only rate means: `{'origin_delay_rate_3h': 0.08079999358628864, 'destination_delay_rate_3h': 0.0796816299126744}`. LR settings: `{'maxIter': 100, 'regParam': 0.1, 'elasticNetParam': 0.0, 'tol': 1e-06, 'family': 'binomial', 'featuresCol': 'features', 'labelCol': 'is_arrival_delayed_15m', 'weightCol': 'class_weight'}`; class weights 2:1. All defined estimator parameters were compared with the accepted LR contract. No random step was introduced; Spark LR has no seed parameter (frozen reproducibility seed 42).

**Qualification: PASS.** Mean monthly PR-AUC 0.4637348613945793, frozen baseline 0.4280983226531411; paired gain **+0.0356365387414382**, required ≥0.005. Mean recall changes: top-10% **+0.0193392086322159**, top-20% **+0.0241679483428413**; both required ≥−0.01.

Fit elapsed: 284.902s. Classifier fits started: 1. Candidate threshold: **0.39**, selected once on pooled May–August validation using the existing 101-step grid, exact F1 comparison and lower-step tie break. Production threshold remains **0.39**.

| Month | Rows | PR-AUC | Baseline PR-AUC | Paired gain | Recall @10% | Δ recall @10% | Recall @20% | Δ recall @20% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024-05 | 599,226 | 0.475474051383 | 0.432454906586 | +0.043019144796 | 0.219073011701 | +0.023623290960 | 0.381927771741 | +0.029353787056 |
| 2024-06 | 601,272 | 0.429425732603 | 0.403536551946 | +0.025889180657 | 0.206522391544 | +0.015785788122 | 0.365640734936 | +0.017132948748 |
| 2024-07 | 613,938 | 0.511597963551 | 0.470587236842 | +0.041010726709 | 0.209787479724 | +0.020911115388 | 0.372100184203 | +0.026618645699 |
| 2024-08 | 604,303 | 0.438441698041 | 0.405814595238 | +0.032627102803 | 0.227512481313 | +0.017036640059 | 0.396912814148 | +0.023566411869 |

Monthly JSON evidence contains exact confusion matrices, PR/ROC AUC, precision/recall and budget capacities. Ranking is descending score then ascending flight_id; identical score/id boundary rows retain equal fractional selection mass. No deduplication. Candidate model parameters/790d width were verified after reload; all 2,418,739 validation score rows matched exactly before/after reload, and saved predictions matched full business-key/label multiplicity.

**Resources:** unchanged pinned image `sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf`; two one-core executors, 640 MiB heaps, 768 MiB driver, four shuffle partitions; 2 GiB worker/driver and 512 MiB master cgroup limits. Task/stage retries disabled.

Task totals: shuffle_read_bytes: 15.803 GiB, shuffle_write_bytes: 15.687 GiB, Disk Bytes Spilled: 0.000 GiB, Memory Bytes Spilled: 0.000 GiB. Failed tasks: 0. Memory-spill bytes are deserialized spill volume, not RAM usage.

| Container | Lifetime peak MiB | OOM / kills |
|---|---:|---:|
| airline-lr-3h-v2-master-1 | 274.89 | 0 / 0 |
| airline-lr-3h-v2-worker1-1 | 2051.36 | 0 / 0 |
| airline-lr-3h-v2-worker2-1 | 2059.39 | 0 / 0 |
| airline-lr-3h-v2-driver-1 | 1394.18 | 0 / 0 |

Phase-tagged memory samples and shuffle/spill/task measurements: [resources.json](evidence/resources.json). Cgroup peaks include cache; per-phase sampled maxima are lower bounds. No spare-memory guarantee is inferred.

## Commands and provenance

```bash
python3 -m unittest studies.lr_3h_v2.test_protocol -v
python3 -m py_compile studies/lr_3h_v2/*.py
docker compose -f studies/lr_3h_v2/compose.preflight.yml up -d
docker compose -f studies/lr_3h_v2/compose.preflight.yml exec -T driver timeout --kill-after=15 1200 /opt/spark/bin/spark-submit --driver-memory 768m /study/preflight.py
docker compose -f studies/lr_3h_v2/compose.preflight.yml down --timeout 15
python3 studies/lr_3h_v2/run.py
# Supervisor invokes:
docker compose -f studies/lr_3h_v2/compose.yml exec -T driver timeout --signal=TERM --kill-after=15 14100 /opt/spark/bin/spark-submit --driver-memory 768m /study/experiment.py
# Always clean up:
docker compose -f studies/lr_3h_v2/compose.yml down --timeout 15
# Offline summaries only:
python3 studies/lr_3h_v2/summarize.py
python3 studies/lr_3h_v2/report.py
```

Input identities: [input verification](evidence/input-verification.json). Saved model, score and prediction file identities: [artifact hashes](evidence/artifact-hashes.json). Exact model/validation outputs remain in `studies/lr_3h_v2/runtime/output/`, with hashes committed and binary files ignored. Complete optimizer objective history, selected threshold grid and all four monthly results are retained. Full Spark event-log hashes appear in resource evidence. [Execution/cleanup](evidence/execution.json); [preservation](evidence/preservation.json); [runtime preflight](evidence/preflight.json); [population](evidence/population.json).

**Stop:** no additional fit, baseline scoring, test-period evaluation, model promotion or main-branch merge. Production remains LR at 0.39.

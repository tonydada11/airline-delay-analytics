# Track A three-hour feature qualification — PASS

Reference `16e841e`; implementation `3a237eb`. One isolated Spark attempt; no retries, preprocessing fit, classifier fit, threshold tuning or promotion.

**Population:** {'train': 8946370, 'validation': 2418739}. 20/20 months verified. Every completed month passed both directions of `exceptAll` over every original column, preserving business-key/label multiplicity and original feature values. All six new columns passed domain/missingness checks.

**Temporal verification:** `python3 -m unittest studies.ml_three_track_v1.test_track_a -v` passed four tests. Spark `checks.generated` passed exact microsecond bounds, duplicate sources/targets, null entities/times, future/current/in-flight exclusion, forbidden-outcome perturbation and reversed ordering (three comparisons × five target occurrences). `checks.sample` matched all six values for all 109,423 targets against `track_a.enrich`, using 331,355 sources; prior sample source SHA-256s matched.

Sources use the existing eligibility and UTC converter. Contributions are active exactly when `t−3h ≤ departure < t` and `availability < t`. Availability remains scheduled departure + signed delay + elapsed time, with the accepted zero reporting latency assumption; it is not observed publication time. Only 2023 and January–August 2024 DWD partitions were read. Monthly source bounds use exact target UTC extrema, retaining cross-month history. Eight source rows had uncertain prediction/availability times and were excluded from history contributions; null-time targets retain empty histories.

**Schema:** nullable `double` rates; non-null observed `bigint` counts and binary `double` missing indicators (Spark schema nullability may be conservative). Empty history is `(null, 0, 1)`.

| Column | Type |
|---|---|
| `origin_delay_rate_3h` | double (nullable) |
| `origin_flight_count_3h` | bigint |
| `origin_history_missing_3h` | double (0/1) |
| `destination_delay_rate_3h` | double (nullable) |
| `destination_flight_count_3h` | bigint |
| `destination_history_missing_3h` | double (0/1) |

All original inputs and the frozen preprocessing artifact are preserved; original width is 784, proposed appended width is 790. No 790-dimensional vector or new imputation means were fit.

**Resources:** 2789.22 seconds including cleanup, within the 14,400-second cap. Pinned Spark 3.5.9 image; two one-core workers, 640 MiB executor heaps, 768 MiB driver heap, four shuffle partitions. No resource changes.

Whole-application task totals: shuffle read **19.52 GiB**, shuffle write **19.52 GiB**, disk spill **6.97 GiB**, memory spill **15.26 GiB**. Memory-spill bytes describe deserialized spill volume, not peak RAM. Failed task events: 0.

| Container | Exact lifetime cgroup peak, MiB | OOM events |
|---|---:|---|
| master-1 | 433.2 | 0 / 0 kills |
| worker1-1 | 2,053.5 | 0 / 0 kills |
| worker2-1 | 2,054.9 | 0 / 0 kills |
| driver-1 | 1,702.4 | 0 / 0 kills |

Worker memory limits remained 2,048 MiB; recorded cgroup peaks were slightly above those limits, with 46,082 / 46,058 memory-limit events and no OOM events. This run does not establish spare memory headroom.

| Split/month | Rows | Seconds | Origin missing | Destination missing | Shuffle write MiB | Disk spill MiB |
|---|---:|---:|---:|---:|---:|---:|
| train-2023-01 | 527,197 | 168.6 | 15.50% | 17.68% | 821.6 | 442.7 |
| train-2023-02 | 492,747 | 125.3 | 14.96% | 16.88% | 851.9 | 147.7 |
| train-2023-03 | 571,533 | 137.1 | 14.95% | 16.96% | 1,017.4 | 472.6 |
| train-2023-04 | 550,249 | 112.9 | 14.47% | 16.02% | 971.0 | 307.7 |
| train-2023-05 | 575,429 | 131.5 | 13.80% | 15.13% | 1,025.0 | 478.0 |
| train-2023-06 | 562,804 | 107.2 | 14.09% | 16.23% | 995.1 | 315.4 |
| train-2023-07 | 585,058 | 120.2 | 14.24% | 16.36% | 1,036.5 | 329.1 |
| train-2023-08 | 592,142 | 140.8 | 13.42% | 15.60% | 1,048.4 | 332.8 |
| train-2023-09 | 560,887 | 141.1 | 13.47% | 15.35% | 999.8 | 458.3 |
| train-2023-10 | 596,003 | 123.9 | 13.28% | 14.75% | 1,048.9 | 334.4 |
| train-2023-11 | 562,413 | 124.9 | 13.97% | 14.95% | 987.0 | 314.3 |
| train-2023-12 | 566,941 | 125.5 | 14.59% | 16.11% | 996.9 | 317.1 |
| train-2024-01 | 525,370 | 114.9 | 16.02% | 17.94% | 926.4 | 293.8 |
| train-2024-02 | 515,269 | 95.0 | 14.64% | 16.51% | 906.7 | 287.0 |
| train-2024-03 | 585,413 | 109.4 | 14.39% | 16.27% | 1,029.5 | 328.2 |
| train-2024-04 | 576,915 | 114.9 | 13.39% | 15.30% | 1,026.1 | 474.0 |
| validation-2024-05 | 599,226 | 95.2 | 13.91% | 16.04% | 1,055.0 | 336.9 |
| validation-2024-06 | 601,272 | 102.3 | 13.88% | 16.09% | 1,058.6 | 338.1 |
| validation-2024-07 | 613,938 | 104.5 | 14.71% | 16.80% | 1,092.3 | 488.3 |
| validation-2024-08 | 604,303 | 109.9 | 13.72% | 15.97% | 1,065.6 | 339.9 |

**Split totals:** train: 1993.2s monthly build/audit time, origin missing 14.30%, destination missing 16.10%; validation: 411.9s monthly build/audit time, origin missing 14.06%, destination missing 16.23%. Shared source preparation, fixtures, startup and cleanup are additional.


Per-month shuffle reads, memory spill, CPU/GC/task time and sampled cgroup memory are in [resources.json](evidence/resources.json); exact monthly population, UTC bounds and missingness are in [months.json](evidence/months.json). Per-month cgroup samples are lower bounds at approximately 10-second intervals; lifetime peaks include cache and are not simultaneous totals. Source preparation and fixture costs are separately tagged, not attributed to a split/month.

**Preservation and cleanup:** 1183 original tracked files verified by SHA-256; 3120 existing data/artifact/study files verified by size/mtime. Unchanged: True. Study containers/network removed: True. Original mounts were read-only. Frozen preprocessing file hashes are separately recorded. These metadata checks are not full content hashing of all datasets.

Output: `studies/track_a_3h_v1/runtime/output/features/`, exclusive month directories; 80 Parquet files (2,013,633,097 bytes) are SHA-256 recorded in [output-files.json](evidence/output-files.json); binary data and full event logs remain local and ignored. [Execution](evidence/execution.json), [sample parity](evidence/sample.json), [generated checks](evidence/generated.json), [schema](evidence/schema.json), and event-log hashes in [resource evidence](evidence/resources.json) provide provenance.

**Stop:** feature qualification only. No predictive improvement is claimed; no model fitting or promotion occurred.

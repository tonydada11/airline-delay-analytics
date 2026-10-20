# Bounded Spark performance experiment

These measurements were **newly executed during repository reconstruction** using
Spark 3.5.9, the pinned dependencies, two one-core workers and the unchanged M9
implementation. The actual completion timestamp is in [complete.json](complete.json); both the
benchmark command and the optional full M9 quality command exited 0.

The workload reads January–April 2023 DWD flights and applies the existing
route-daily transformation. It runs a warm-up plus two measured repetitions per
variant, verifies invariant output, records physical plans and live Spark runtime
counters, and restores configuration and clears persistence afterward.

| Experiment | Variant | Measured runs | Median seconds | Median shuffle read bytes | Median tasks |
|---|---|---:|---:|---:|---:|
| cache_behavior | cache | 2 | 1.596995 | 11993 | 9 |
| cache_behavior | no_cache | 2 | 0.364618 | 11993 | 7 |
| join_strategy | broadcast | 2 | 0.749378 | 1371 | 4 |
| join_strategy | non_broadcast | 2 | 1.746594 | 4366262 | 6 |
| partition_count | 1 | 2 | 2.784116 | 23478822 | 3 |
| partition_count | 2 | 2 | 3.117732 | 58580728 | 6 |
| partition_count | 4 | 2 | 3.803832 | 66063844 | 8 |
| partition_count | 8 | 2 | 3.500057 | 71390940 | 12 |
| repartition_vs_coalesce | coalesce_2 | 2 | 2.311141 | 17437440 | 4 |
| repartition_vs_coalesce | repartition_2 | 2 | 5.748649 | 58579750 | 6 |
| shuffle_partitions | 2 | 2 | 1.706928 | 16866399 | 4 |
| shuffle_partitions | 4 | 2 | 1.406191 | 17437440 | 4 |
| shuffle_partitions | 8 | 2 | 1.483344 | 18038257 | 5 |
| storage_level | memory_and_disk | 2 | 0.916680 | 11993 | 9 |
| storage_level | memory_only | 2 | 1.180400 | 11993 | 9 |

The baseline reconciled **2,183,349** flights
into **570,181** route-day rows, with
**0** duplicate grains.
All variants passed the implementation's logical-output invariants. Configuration
restoration and persistence cleanup passed. The optional full M9 quality suite
passed against these regenerated manifest, completion, run and plan records.

Partition changes trade shuffle work against output parallelism; broadcasting
changes the dimension join; caching adds materialization cost before reuse.
The measured differences describe this bounded workload on one Docker host.
Two repetitions do not support general throughput or scalability claims, and
these measurements do not change production Spark settings or model selection.

- [Summary JSON](summary.json): logical profiles and median runtime counters.
- [Summary CSV](summary.csv): per-run timing and execution counters.
- [Manifest](manifest.json), [completion](complete.json), [run records](runs/)
  and [physical plans](plans/): evidence used by the optional quality checks.
- [Historical interpretation](../../../docs/SPARK_CONCEPTS.md): explicitly labeled
  findings from the original run; its numeric timings are not this new run.

Reproduce from the repository root with the production cluster running and no
other Spark workload active:

```bash
docker compose exec -T spark-notebook python3 -m airline.performance.run_experiments
docker compose exec -T -e AIRLINE_RUN_FULL_M9_QUALITY=1 spark-notebook \
  python3 -m pytest -q tests/data_quality/test_m9_performance_quality.py
```

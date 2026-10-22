# Bounded Spark performance experiment

These measurements were regenerated on October 21, 2026 using Spark 3.5.9, the
pinned dependencies, two one-core workers and the unchanged M9 implementation.

The workload reads January–April 2023 DWD flights and applies the existing
route-daily transformation. It runs a warm-up plus two measured repetitions per
variant, verifies invariant output, records physical plans and live Spark runtime
counters, and restores configuration and clears persistence afterward.

| Experiment | Variant | Measured runs | Median seconds | Median shuffle read bytes | Median tasks |
|---|---|---:|---:|---:|---:|
| cache_behavior | cache | 2 | 1.946596 | 11982 | 9 |
| cache_behavior | no_cache | 2 | 0.402372 | 11982 | 7 |
| join_strategy | broadcast | 2 | 0.917137 | 1371 | 4 |
| join_strategy | non_broadcast | 2 | 1.685407 | 4366262 | 6 |
| partition_count | 1 | 2 | 3.214544 | 23478822 | 3 |
| partition_count | 2 | 2 | 3.670923 | 58579750 | 6 |
| partition_count | 4 | 2 | 4.116217 | 66067505 | 8 |
| partition_count | 8 | 2 | 4.380926 | 71394430 | 12 |
| repartition_vs_coalesce | coalesce_2 | 2 | 2.122819 | 17437440 | 4 |
| repartition_vs_coalesce | repartition_2 | 2 | 4.920776 | 58578773 | 6 |
| shuffle_partitions | 2 | 2 | 1.936778 | 16866399 | 4 |
| shuffle_partitions | 4 | 2 | 1.700388 | 17437440 | 4 |
| shuffle_partitions | 8 | 2 | 1.904672 | 18038257 | 5 |
| storage_level | memory_and_disk | 2 | 1.248509 | 11982 | 9 |
| storage_level | memory_only | 2 | 1.349956 | 11982 | 9 |

The baseline read **2,183,349** flights and built
**570,181** route-day rows, with
**0** duplicate grains.
Status: **PASS**. All variants passed the implementation's logical-output
invariants. Configuration restoration and persistence cleanup passed. The
production change was not retained. The optional full M9 quality suite passed
against the generated run and plan records (3 tests passed).

Partition changes trade shuffle work against output parallelism; broadcasting
changes the dimension join; caching adds materialization cost before reuse.
The measured differences describe this bounded workload on one Docker host.
Two repetitions do not support general throughput or scalability claims, and
these measurements do not change production Spark settings or model selection.

- [Summary JSON](summary.json): logical profiles and median runtime counters.
- [Summary CSV](summary.csv): per-run timing and execution counters.
- Detailed run records, manifests, completion markers, configuration-restoration
  records and physical plans are generated locally by the benchmark command. They
  are not tracked in Git; the optional quality suite reads them from the local
  output directory after regeneration.

Reproduce from the repository root with the production cluster running and no
other Spark workload active:

```bash
docker compose exec -T spark-notebook python3 -m airline.performance.run_experiments
docker compose exec -T -e AIRLINE_RUN_FULL_M9_QUALITY=1 spark-notebook \
  python3 -m pytest -q tests/data_quality/test_m9_performance_quality.py
```

# M9 Spark performance experiment report

Status: **PASS**

Scope: real DWD/DIM/DWS data from 2023-01-01 through 2023-04-30, one warm-up,
two measured runs per variant, two-worker/two-core Spark Standalone cluster.

| Comparison | Variant | Runs (seconds) | Median (seconds) |
|---|---|---:|---:|
| Repartition/coalesce | `coalesce(2)` | 3.509, 2.774 | 3.141 |
| Repartition/coalesce | `repartition(2)` | 8.467, 9.241 | 8.854 |
| Partition count | 1 | 4.159, 4.309 | 4.234 |
| Partition count | 2 | 5.506, 5.082 | 5.294 |
| Partition count | 4 | 6.602, 5.784 | 6.193 |
| Partition count | 8 | 7.711, 7.196 | 7.454 |
| Join strategy | broadcast | 2.121, 1.476 | 1.798 |
| Join strategy | non-broadcast | 4.104, 1.911 | 3.008 |
| Cache behavior | no persistence | 0.837, 0.822 | 0.829 |
| Cache behavior | `cache()` | 3.476, 1.715 | 2.595 |
| Storage level | `MEMORY_ONLY` | 1.824, 1.772 | 1.798 |
| Storage level | `MEMORY_AND_DISK` | 1.364, 2.135 | 1.750 |
| Shuffle partitions | 2 | 3.093, 3.311 | 3.202 |
| Shuffle partitions | 4 | 2.255, 3.031 | 2.643 |
| Shuffle partitions | 8 | 3.221, 2.214 | 2.717 |

All route variants produced 570,181 unique route-day grains, zero duplicates,
2,183,349 scheduled flights, and identical metrics. All join and reuse variants
also reconciled exactly. No spill was recorded. Normal Spark configuration was
restored, persisted RDD count returned to zero, and no production change was
retained.

See `summary.json`, `summary.csv`, `runs/`, and `plans/` for machine-readable
runtime and physical-plan evidence. Evidence-backed interpretation is in
`docs/SPARK_CONCEPTS.md`.

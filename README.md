# Airline Delay Analytics

An Apache Spark project that turns two years of U.S. flight records into a
warehouse, operational reports, and an arrival-delay model.

## How it works

```text
BTS CSV → ODS → DWD
                 ├→ dimensions
                 ├→ daily analytics → scorecards
                 └→ flight-time features → model selection → frozen evaluation
```

The pipeline uses Spark 3.5.9, PySpark, Spark MLlib, Python 3.10.12, Docker
Compose, and a two-worker Spark Standalone cluster. Its pinned snapshot contains
24 monthly files from 2023–2024 and 13,926,960 flights.

ODS preserves source rows and lineage. DWD standardizes flight facts.
Dimensions describe dates, carriers, airports, and directed routes. Daily DWS
tables support ADS reports, while a separate flight-level feature branch feeds
machine learning. Raw data, warehouse tables, and fitted models remain in local
runtime storage.

## Model

The model predicts whether an eligible flight arrives at least 15 minutes late.
Its 50 inputs describe schedules, routes, calendar information, and previous
flight delays. Training, validation, and testing use consecutive time periods.

The selected production model is logistic regression with a 0.39 decision
threshold.

| Split | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Validation | 0.427192 | 0.687064 | 0.481461 | 0.373881 | 0.675959 |
| Historical test | 0.264542 | 0.653149 | 0.253188 | 0.318536 | 0.210088 |

The September–December 2024 test period was previously inspected, so this is
historical evaluation, not a fresh external holdout. Historical features assume
arrival outcomes become available immediately; actual publication times are
not recorded.

## Spark experiments

In the recorded benchmark, broadcasting the small carrier table took 0.92
seconds versus 1.69 seconds without broadcast. Coalescing to two partitions took
2.12 seconds versus 4.92 seconds for repartitioning. These measurements describe
a bounded workload on one Docker host.

The recovery experiment stopped a worker during an active task. Spark retried
the task on the surviving worker and produced the expected output. It does not
test recovery from losing the driver or shared host.

## Reproduce

Use Docker Compose with a healthy Docker daemon and at least 6 GiB available
to the runtime. The downloader also requires curl and unzip. Run from the
repository root:

```bash
bash scripts/scripts_download_bts.sh
bash scripts/start_cluster.sh
bash scripts/run_pipeline.sh
AIRLINE_RUN_REBUILD_ACCEPTANCE=1 bash scripts/run_tests.sh
```

The pipeline verifies the pinned source files and stops when a required quality
check fails. The final command includes the comprehensive feature-rebuild
checks.

For notebooks, select the `spark` kernel. Obtain the Jupyter login URL with
`docker compose exec spark-notebook jupyter server list`.

## Report, walkthrough, and course learning

- [Project report](docs/guides/project-guide.html): pipeline stages, recorded
  results, and worker recovery; download and open locally in a browser.
- [Video walkthrough notebook](notebooks/04_video_walkthrough.ipynb).
- [What I learned and applied from CS544](docs/learning/cs544-concepts.md).

## Chat history and AI-generated tests

The development conversations cover:

- [Dataset selection and architecture](docs/conversations/01-dataset-and-architecture.md).
- [Implementation planning](docs/conversations/02-implementation-planning.md).
- [Milestone review and test development](docs/conversations/03-code-review-and-testing.md).

The [shared prompt](docs/conversations/00-system-prompt.md) records the working
instructions used for these discussions.

Codex generated the automated test code under `tests/`. These files are marked
with AI-generation comments.

## Research

- [`feature/spark-fault-tolerance`](https://github.com/tonydada11/airline-delay-analytics/tree/feature/spark-fault-tolerance):
  worker-loss recovery, merged after task retry and output checks.
- [`feature/flight-time-history`](https://github.com/tonydada11/airline-delay-analytics/tree/feature/flight-time-history):
  causal historical features and the accepted model.
- [`experiment/model-comparison`](https://github.com/tonydada11/airline-delay-analytics/tree/experiment/model-comparison):
  LR, GBT, and XGBoost studies; no alternate candidate was promoted.
- [`experiment/recent-delay-features`](https://github.com/tonydada11/airline-delay-analytics/tree/experiment/recent-delay-features):
  improved offline validation results, but operational input availability
  prevented promotion.

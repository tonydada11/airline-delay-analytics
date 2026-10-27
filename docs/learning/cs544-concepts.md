## CS544 — Course Concepts Applied

I applied CS544 concepts to a Spark-based airline analytics project using 13.9 million flight records.

- **Docker and Distributed Computing:** Built a Spark Standalone cluster using Docker Compose with one master, two workers, and a notebook/driver service. Applied container networking, resource allocation, and distributed task execution.

- **Spark DataFrames and SQL:** Used PySpark transformations, joins, and aggregations to process 24 monthly CSV datasets. Developed an ODS → DWD → DIM/DWS → ADS data warehouse using explicit schemas and Parquet storage.

- **Spark Execution Internals:** Explored RDDs versus DataFrames, lazy evaluation, transformations versus actions, DAGs, jobs, stages, and tasks. Inspected Spark UI metrics and physical execution plans to understand shuffle operations and execution behavior.

- **Performance Optimization:** Experimentally compared partition counts, `repartition()` versus `coalesce()`, broadcast versus non-broadcast joins, and caching strategies. Measured execution time, shuffle traffic, and task counts while verifying equivalent outputs.

- **Fault Tolerance:** Simulated worker failure during an active Spark job and verified that Spark retried the failed task on another worker, producing the expected result.

- **Spark MLlib:** Built a distributed classification pipeline using feature encoding, imputation, Logistic Regression, and Gradient-Boosted Trees. Extended the course concepts with 50 predictors, leakage-aware historical features, and chronological train/validation/test evaluation.

- **Data Integrity and Reproducibility:** Implemented data-quality checks, automated tests, deterministic pipeline execution, and reproducible Docker environments to ensure reliable processing and consistent results.

**Overall**, this project extends CS544's classroom examples into a complete distributed data-engineering and machine-learning application, emphasizing Spark execution, performance tradeoffs, fault tolerance, and correctness.

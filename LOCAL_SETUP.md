# Local runtime for the consolidated root project

Run commands from this repository root. There is one implementation and one
active artifact root, `artifacts/flighttime-v3`. Historical source versions are
available through Git checkpoints, not a second executable checkout directory.

The ignored `docker-compose.override.yml` preserves this machine's accepted native
runtime: project `airline-revision`, image `airline-revision-spark:3.5.9`, notebook
and worker user alignment, and the root evidence bind mount. Local addresses are:

| Service | Address |
|---|---|
| JupyterLab | http://localhost:8889 |
| Spark master UI | http://localhost:8082 |
| Active Spark application UI | http://localhost:4041 |

A clean clone without the local override uses ports 8888, 8080, and 4040. All
containers use the locked Python 3.10.12 / Spark 3.5.9 native runtime. Driver and
executor heaps, partition settings, and estimator settings remain unchanged.

```bash
bash scripts/start_cluster.sh
docker compose exec -T spark-notebook python3 scripts/verify_dependencies.py
bash scripts/run_pipeline.sh
bash scripts/run_tests.sh
AIRLINE_RUN_REBUILD_ACCEPTANCE=1 bash scripts/run_tests.sh
bash scripts/stop_cluster.sh
```

Pinned raw CSVs and ZIP downloads remain under `data/raw/`. Do not edit them or
refresh expected hashes. Historical setup logs/configuration are indexed in
`evidence/archive/pre-consolidation/INDEX.md`; the old two-project setup and
temporary copy-recovery paths are historical and are not current instructions.

Consolidation acceptance requires at least 70 GiB free before starting, including
a conservative 40 GiB allowance for temporary Spark work. Run Spark drivers
sequentially. Current status is in `docs/FLIGHTTIME_VERIFICATION.md`.

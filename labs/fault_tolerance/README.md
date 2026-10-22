# Worker recovery lab

This small Spark experiment kills a worker while one of its tasks is running.
The recorded run shows Spark retrying that task on another worker and returning
the expected result. The [compact event summary](evidence/verification.json)
contains the failed task, retry, executor removal, and output checks.

Run it from the repository root with Docker Compose available:

```bash
python3 labs/fault_tolerance/run_experiment.py
```

Each run writes its detailed logs to a new local folder under
`artifacts/fault_tolerance/`. The lab uses a small synthetic workload and does
not test driver recovery or external writes.

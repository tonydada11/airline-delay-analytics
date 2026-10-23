# Reproduce the isolated XGBoost fixture study

Read [REPORT.md](REPORT.md) for the BLOCKED full-data decision and successful
bounded compatibility checks. This directory is the complete change scope.

## Environment

Base: local `airline-revision-spark:3.5.9`, verified immutable image ID
`sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf`.
The derivative image ID and package versions are in [execution](evidence/execution.json)
and [results](evidence/results.json). The base supplies Spark 3.5.9, Python 3.10.12,
NumPy 2.2.6 and pandas 2.3.3. Production requirements and Docker files are untouched.
`requirements.in` pins six added CPU packages; `requirements.lock` hashes the exact
Linux x86_64/Python 3.10 wheels. This is not a cross-platform lock.

For a fresh, separately authorized replay, use the same base image. Download only
the locked wheels into ignored `runtime/wheels`, inside a disposable base container:

```sh
mkdir -p studies/xgb_feasibility_v1/runtime/wheels studies/xgb_feasibility_v1/runtime/output
docker run --rm --user 0:0 --entrypoint python3 \
  --mount type=bind,src="$PWD/studies/xgb_feasibility_v1",dst=/study \
  sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf \
  -m pip download --no-deps --only-binary=:all: --require-hashes \
  -r /study/requirements.lock -d /study/runtime/wheels
docker build --network none -f studies/xgb_feasibility_v1/Dockerfile \
  -t airline-xgb-feasibility:3.0.5 studies/xgb_feasibility_v1
python3 studies/xgb_feasibility_v1/run.py
```

Verify the base image ID **before** building; Dockerfile's local tag alone is not
an immutable reference. `run.py` also checks it before execution. The original
acquisition used `requirements.in`; wheel hashes were frozen before installation.
No production image layer or package installation was changed.

The supervisor refuses occupied output and active production applications, caps
the generated fixture application at 1,200 seconds, samples Docker memory, and
removes only the study Compose project in `finally`. No published ports or
production data mounts exist. Source and accepted preprocessing are read-only;
only ignored study output is writable. The derivative image/wheels may remain
cached for audit; no study service remains running.

The executed application was:

```sh
docker compose -f studies/xgb_feasibility_v1/compose.yml exec -T driver \
  timeout 1200 /opt/spark/bin/spark-submit --driver-memory 768m /study/fixtures.py
```

Eight fits are fixed tests, not tuning: accepted sparse/dense inputs (514 rows
each), zero/NaN default/unsafe CSR semantics (192 each), weighted/unweighted
analytic fixture (256 each), and two scaling fixtures (4,096 / 32,768 rows).
All trees/depths are small fixed fixture settings, distinct from the unexecuted
candidate. The classifier output probabilities are checked, not evaluated as
airline predictions. No `BinaryClassificationEvaluator` or flight split is read.

## Offline verification

```sh
python3 -m unittest studies.xgb_feasibility_v1.test_supervisor -v
python3 studies/xgb_feasibility_v1/verify.py
```

The second command needs the original local preprocessing/artifacts and checks
them against `preservation_before.json`. It verifies 18 saved runtime assertions,
eight fits, source hashes and the supervisor's two offline regression tests.
It does not run Spark, install packages or train anything. Its timing text can
vary between runs.

The initial supervisor's raw equality check failed after successful fixture
completion; Docker returns mount arrays in varying order. `run-executed.py.txt`
preserves that executed source. The corrected `run.py` sorts mounts and persists
both comparison inputs for future replays. Original execution/isolation evidence
is not rewritten, and the successful fixture suite was not rerun. See
[metadata investigation](evidence/metadata-investigation.json).

## Provenance and measurements

- `evidence/results.json`: verbatim fixture result; runtime versions/config,
  18 assertions, fit counts/durations and rank/host/RSS observations.
- `evidence/execution.json`: original fixture exit code, image IDs, Docker samples
  and cgroup peaks/OOM counters before study shutdown.
- `evidence/wheels.json`, `source-provenance.json`: exact wheel hashes/metadata,
  executed source hashes, adapter members extracted from the pinned wheel and
  upstream Apache license. No upstream code was modified.
- `evidence/contract-review.json`: hashes of accepted contracts and saved LR/GBT
  evaluations; these files are references only.
- `evidence/resources.json`: measurements and explicit dense float32 lower-bound
  calculation. Python RSS is measured at the final boosting iteration, whereas
  container cgroup peaks include all process memory/cache over the whole suite.
- `candidate.json`: exactly one bounded proposal requiring separate review. It
  is not a runnable full-data training plan.

The Spark model and event log remain in ignored `runtime/output`; their size is
small and they contain only generated fixtures. They are not production artifacts.
Changing `missing` to zero, substituting an unverified XGBoost release, installing
into production or expanding the dataset is outside this study's authorization.

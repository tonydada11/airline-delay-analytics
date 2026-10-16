# Isolated worker-loss experiment

**PASS — newly executed 24 October 2026.** Spark 3.5.9 recovered the exact active
shuffle-map task whose worker was killed. Baseline and recovery outputs matched
all ten independently calculated groups: 100 rows per group, 1,000 total rows,
and sum 499,500. This is a worker/executor-loss experiment, separate from the
original project's synthetic Python-exception demonstration.

## Reproduce

Build the pinned image with `./scripts/start_cluster.sh` if it is not present,
then shut down the idle production cluster before running this lab. From the root:

```bash
python3 labs/fault_tolerance/run_experiment.py
python3 labs/fault_tolerance/run_experiment.py --verify labs/fault_tolerance/evidence
```

The host needs Python 3 and Docker Compose. Spark and its dependencies come from
`airline-spark:3.5.9`, built by the project's pinned Dockerfile. The host controller
uses only Python's standard library. Every new run receives a unique Compose
project and exclusive output directory; existing evidence cannot be overwritten.

The recorded execution used a staging location before these exact source bytes
were copied into the reconstructed repository. [execution.json](evidence/execution.json)
records the exact absolute commands, actual UTC finish time, project name and
SHA-256 of both source/configuration files. Those hashes match the delivered files.
The public command above resolves the same configuration from its new location.

## Topology, bounds and isolation

A private Compose network contains master, worker1, worker2 and driver. No ports
are published. Only experiment-owned evidence and the workload script are mounted;
there are no production data, model, source or warehouse mounts. Workers have one
core each and executors use 640 MiB. Docker caps each non-driver container at one
CPU and 1.5 GiB, and the driver at two CPUs and 2 GiB. Each container has a 256 MiB
temporary filesystem; log rotation caps each container log at 5 MiB. Executable
temporary storage is required for Spark's native compression library. The fixed
1,000-row workload bounds output size. The controller enforces a 240-second
startup/workload deadline and bounded startup/shutdown subprocess timeouts.

Before sending a signal the controller checks Docker's project and service labels
against its freshly allocated project and worker1. It always executes `down
--volumes --remove-orphans` in `finally`, then verifies no project containers remain.
It retains evidence files for review. An interrupted worker's active-task marker
is scratch evidence, not a partial result dataset.

## Workload and failure trigger

Eight input partitions sleep briefly, then map integers to `(value % 10,
(1, value))`; four reduce partitions sum count and value. The driver compares all
groups against `count=100, sum=49500+100*key`. No executor performs business writes.
The second job is identical, with an additional diagnostic marker written before
each task's bounded sleep. The controller reads worker1's marker and kills that
worker while the identified task is active. The marker is instrumentation, not
an exactly-once output or a recovery mechanism.

Actual injection (project identity from the observed run):

```bash
docker compose -p airline-ft-825445010659 -f /tmp/airline-reconstruction/lab/compose.yml kill -s SIGKILL worker1
```

The exact authoritative project/command is in [injection.json](evidence/injection.json).

## Observed result

| Check | Observation |
|---|---|
| Baseline | Exact ten-group reference matched |
| Worker failure | Label-verified worker1 received SIGKILL |
| Active task | Stage 2, partition 1, task 13, attempt 0, executor 1 |
| Failure | `ExecutorLostFailure`, followed by executor removal |
| Retry | Stage 2, partition 1, task 14, attempt 1, executor 0 on worker2 |
| Application | Two successful jobs; four completed stages; 25 task attempts |
| Recovery output | Exact baseline/reference equality; no missing or duplicate groups |
| Shutdown | Compose down exit 0; no experiment containers remained |

[verification.json](evidence/verification.json) includes the actual failed and
successful TaskEnd records. [Raw events](evidence/events/) preserve independent
verification of task IDs, stages, executor removal and job completion.
[Baseline](evidence/baseline.json) and [recovery](evidence/recovery.json) contain
all output groups. The original production containers remained stopped with the
same IDs; the experiment never mounted original datasets or artifacts.

## Failed attempts and verification limits

[Prior attempts](evidence/prior-attempts.json) record two startup/runtime failures
and a third completed workload that failed strict verification. The third run's
buffered event log exposed a stale active-task observation: Spark lost and retried
work, but not the task selected by the controller. The current executor-written
marker fixed that observation problem without relaxing the loss/retry assertion.
Full development logs remain in the external reconstruction audit.

The passing run demonstrates recovery of deterministic task work after one worker
and its executor disappear. It does not test driver/master failure, replicated
storage, simultaneous faults, host loss, arbitrary workloads, or exactly-once
external side effects. All services share one physical host. The synthetic
workload is deliberately small and is not a throughput benchmark.

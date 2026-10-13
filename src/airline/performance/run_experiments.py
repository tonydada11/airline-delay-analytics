"""Run bounded M9 Spark experiments against real airline warehouse data.

The runner reads only warehouse inputs and writes only beneath a named
``artifacts/performance`` directory.  It never imports or invokes ML scoring.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import csv
from datetime import date, datetime, timezone
import io
import json
from pathlib import Path, PureWindowsPath
import shutil
import statistics
import time
from typing import Callable, Iterator, Mapping, Sequence
from urllib.request import urlopen

from pyspark import StorageLevel
from pyspark.sql import DataFrame, Row, SparkSession, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.spark_session import create_spark_session
from airline.dws.build_route_daily import transform_route_daily


DEFAULT_START_DATE = date(2023, 1, 1)
DEFAULT_END_DATE = date(2023, 4, 30)
DEFAULT_REPETITIONS = 2
CONFIG_KEYS = (
    "spark.sql.shuffle.partitions",
    "spark.sql.autoBroadcastJoinThreshold",
    "spark.sql.adaptive.enabled",
    "spark.sql.adaptive.coalescePartitions.enabled",
)
STAGE_COUNTERS = (
    "numTasks",
    "executorRunTime",
    "inputBytes",
    "inputRecords",
    "outputBytes",
    "outputRecords",
    "shuffleReadBytes",
    "shuffleReadRecords",
    "shuffleWriteBytes",
    "shuffleWriteRecords",
    "memoryBytesSpilled",
    "diskBytesSpilled",
)
ROUTE_GRAIN = ("flight_date", "origin", "destination")


def write_json(document: object, path: Path) -> None:
    """Write deterministic JSON, creating only the requested parent tree."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def validate_output_root(output: Path, project_root: Path) -> Path:
    """Require a named, project-relative directory below artifacts/performance."""

    if output.is_absolute() or PureWindowsPath(str(output)).is_absolute():
        raise ValueError("M9 output must be project-relative.")
    if ".." in output.parts:
        raise ValueError("M9 output must not traverse above the project root.")
    resolved = (project_root / output).resolve()
    parent = (project_root / "artifacts/performance").resolve()
    if resolved == parent or parent not in resolved.parents:
        raise ValueError(
            "M9 output must be a named directory below artifacts/performance."
        )
    return resolved


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("dates must use YYYY-MM-DD") from error


def median_runtime(runs: Sequence[Mapping[str, object]]) -> float:
    """Return the median elapsed time from a non-empty run collection."""

    if not runs:
        raise ValueError("At least one run is required.")
    return float(statistics.median(float(run["elapsed_seconds"]) for run in runs))


def alternating_order(variants: Sequence[str], repetition: int) -> list[str]:
    """Reverse every second measured pass to reduce fixed ordering bias."""

    if repetition < 1:
        raise ValueError("Repetition numbers start at one.")
    ordered = list(variants)
    return ordered if repetition % 2 else list(reversed(ordered))


def _configuration(spark: SparkSession) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for key in CONFIG_KEYS:
        try:
            result[key] = spark.conf.get(key)
        except Exception:
            result[key] = None
    return result


@contextmanager
def temporary_spark_conf(
    spark: SparkSession, overrides: Mapping[str, object]
) -> Iterator[None]:
    """Apply session-only SQL settings and restore every prior value."""

    original = _configuration(spark)
    original = {key: original.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
            spark.conf.set(key, rendered)
        yield
    finally:
        for key, value in original.items():
            if value is None:
                spark.conf.unset(key)
            else:
                spark.conf.set(key, value)


def _api_json(url: str) -> object:
    with urlopen(url, timeout=5) as response:
        return json.load(response)


def spark_snapshot(spark: SparkSession) -> dict[str, object]:
    """Read completed jobs and stages from the active application's UI API."""

    base = spark.sparkContext.uiWebUrl
    if not base:
        return {"available": False, "reason": "Spark UI is disabled"}
    app_id = spark.sparkContext.applicationId
    try:
        jobs = _api_json(f"{base}/api/v1/applications/{app_id}/jobs")
        stages = _api_json(f"{base}/api/v1/applications/{app_id}/stages")
        completed_jobs = {
            str(job["jobId"]): job
            for job in jobs
            if job.get("status") == "SUCCEEDED"
        }
        completed_stages = {
            f"{stage['stageId']}:{stage['attemptId']}": stage
            for stage in stages
            if stage.get("status") == "COMPLETE"
        }
        return {
            "available": True,
            "base": base,
            "application_id": app_id,
            "jobs": completed_jobs,
            "stages": completed_stages,
        }
    except Exception as error:  # REST diagnostics must be reported, not masked.
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


def _task_summary(base: str, app_id: str, stage: Mapping[str, object]) -> object:
    stage_id = stage["stageId"]
    attempt = stage["attemptId"]
    url = (
        f"{base}/api/v1/applications/{app_id}/stages/"
        f"{stage_id}/{attempt}/taskSummary?quantiles=0.5,0.95,1.0"
    )
    try:
        return _api_json(url)
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}


def runtime_delta(
    spark: SparkSession,
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    """Summarize newly completed jobs, stages, tasks, shuffle, and spill."""

    if not before.get("available") or not after.get("available"):
        return {
            "available": False,
            "before_reason": before.get("reason"),
            "after_reason": after.get("reason"),
        }
    before_jobs = before["jobs"]
    after_jobs = after["jobs"]
    before_stages = before["stages"]
    after_stages = after["stages"]
    new_jobs = [value for key, value in after_jobs.items() if key not in before_jobs]
    new_stages = [
        value for key, value in after_stages.items() if key not in before_stages
    ]
    totals = {
        key: sum(int(stage.get(key, 0) or 0) for stage in new_stages)
        for key in STAGE_COUNTERS
    }
    stage_details = []
    for stage in sorted(new_stages, key=lambda item: int(item["stageId"])):
        detail = {
            "stage_id": stage["stageId"],
            "attempt_id": stage["attemptId"],
            "name": stage.get("name"),
            **{key: int(stage.get(key, 0) or 0) for key in STAGE_COUNTERS},
            "task_summary": _task_summary(
                str(after["base"]), str(after["application_id"]), stage
            ),
        }
        stage_details.append(detail)
    return {
        "available": True,
        "job_count": len(new_jobs),
        "stage_count": len(new_stages),
        **totals,
        "jobs": [
            {
                "job_id": job["jobId"],
                "name": job.get("name"),
                "stage_ids": job.get("stageIds", []),
                "num_tasks": int(job.get("numTasks", 0) or 0),
            }
            for job in sorted(new_jobs, key=lambda item: int(item["jobId"]))
        ],
        "stages": stage_details,
    }


def _wait_for_snapshot(spark: SparkSession) -> dict[str, object]:
    snapshot = spark_snapshot(spark)
    for _ in range(8):
        if snapshot.get("available"):
            return snapshot
        time.sleep(0.25)
        snapshot = spark_snapshot(spark)
    return snapshot


def _write_plan(dataframe: DataFrame, path: Path) -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        dataframe.explain(mode="formatted")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _executed_plan(dataframe: DataFrame) -> str:
    return dataframe._jdf.queryExecution().executedPlan().toString()


def _output_partition_count(path: Path) -> int:
    return len(list(path.glob("part-*.parquet")))


def _persistent_rdd_count(spark: SparkSession) -> int:
    return int(spark.sparkContext._jsc.sc().getPersistentRDDs().size())


def storage_snapshot(spark: SparkSession) -> list[dict[str, object]]:
    """Return current executor storage information without collecting data rows."""

    result = []
    for info in spark.sparkContext._jsc.sc().getRDDStorageInfo():
        result.append(
            {
                "rdd_id": int(info.id()),
                "name": str(info.name()),
                "cached_partitions": int(info.numCachedPartitions()),
                "partitions": int(info.numPartitions()),
                "memory_bytes": int(info.memSize()),
                "disk_bytes": int(info.diskSize()),
                "storage_level": str(info.storageLevel().description()),
            }
        )
    return result


def _scalar_row(row: Row) -> dict[str, object]:
    return {name: row[name] for name in row.__fields__}


class PerformanceRunner:
    """Execute controlled variants and persist compact evidence."""

    def __init__(
        self,
        spark: SparkSession,
        project_root: Path,
        dwd_path: Path,
        carrier_dim_path: Path,
        route_daily_path: Path,
        output: Path,
        start_date: date,
        end_date: date,
        repetitions: int,
    ) -> None:
        if start_date > end_date:
            raise ValueError("start-date must be on or before end-date")
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        self.spark = spark
        self.project_root = project_root
        self.dwd_path = dwd_path
        self.carrier_dim_path = carrier_dim_path
        self.route_daily_path = route_daily_path
        self.output = output
        self.start_date = start_date
        self.end_date = end_date
        self.repetitions = repetitions
        self.work = output / "work"
        self.runs: list[dict[str, object]] = []
        self.baseline_path = self.work / "baseline"
        self.original_configuration = _configuration(spark)

    def prepare(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        for name in ("runs", "plans", "work"):
            target = self.output / name
            if target.exists():
                shutil.rmtree(target)
        self.work.mkdir(parents=True)
        write_json(
            {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "spark_version": self.spark.version,
                "spark_application_id": self.spark.sparkContext.applicationId,
                "scope": {
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "repetitions": self.repetitions,
                    "warmups_per_workload": 1,
                },
                "inputs": {
                    "dwd": self.dwd_path.relative_to(self.project_root),
                    "carrier_dimension": self.carrier_dim_path.relative_to(
                        self.project_root
                    ),
                    "route_daily": self.route_daily_path.relative_to(
                        self.project_root
                    ),
                },
                "output": self.output.relative_to(self.project_root),
                "initial_configuration": self.original_configuration,
                "protected_test_evaluation_executed": False,
            },
            self.output / "manifest.json",
        )

    def bounded_dwd(self) -> DataFrame:
        start, end = self.start_date, self.end_date
        partition_filter = (
            (F.col("year") > start.year)
            | (
                (F.col("year") == start.year)
                & (F.col("month") >= start.month)
            )
        ) & (
            (F.col("year") < end.year)
            | ((F.col("year") == end.year) & (F.col("month") <= end.month))
        )
        return self.spark.read.parquet(str(self.dwd_path)).filter(
            partition_filter
            & F.col("flight_date").between(F.lit(start), F.lit(end))
        )

    def bounded_route_daily(self) -> DataFrame:
        return self.spark.read.parquet(str(self.route_daily_path)).filter(
            F.col("flight_date").between(
                F.lit(self.start_date), F.lit(self.end_date)
            )
        )

    def _profile_route(self, path: Path) -> dict[str, object]:
        frame = self.spark.read.parquet(str(path))
        profile = frame.agg(
            F.count("*").alias("rows"),
            F.countDistinct(F.struct(*[F.col(name) for name in ROUTE_GRAIN])).alias(
                "unique_grains"
            ),
            F.sum("scheduled_flights").alias("scheduled_flights"),
            F.sum("eligible_delay_flights").alias("eligible_delay_flights"),
            F.sum("delayed_flights").alias("delayed_flights"),
        ).first()
        result = _scalar_row(profile)
        result["duplicate_grains"] = result["rows"] - result["unique_grains"]
        result["partitions"] = _output_partition_count(path)
        return result

    def _route_differences(self, path: Path) -> dict[str, int]:
        baseline = self.spark.read.parquet(str(self.baseline_path))
        candidate = self.spark.read.parquet(str(path))
        columns = sorted(baseline.columns)
        schema_differs = baseline.select(*columns).schema != candidate.select(
            *columns
        ).schema
        if columns != sorted(candidate.columns) or schema_differs:
            raise AssertionError("Route experiment schema differs from baseline.")
        return {
            "candidate_minus_baseline": candidate.select(*columns)
            .exceptAll(baseline.select(*columns))
            .limit(1)
            .count(),
            "baseline_minus_candidate": baseline.select(*columns)
            .exceptAll(candidate.select(*columns))
            .limit(1)
            .count(),
        }

    def warm_route_workload(self) -> None:
        transform_route_daily(self.bounded_dwd()).agg(
            F.sum("scheduled_flights")
        ).first()

    def run_route_variant(
        self,
        experiment: str,
        variant: str,
        repetition: int,
        transform_source: Callable[[DataFrame], DataFrame] | None = None,
        conf: Mapping[str, object] | None = None,
        baseline: bool = False,
    ) -> dict[str, object]:
        target = self.baseline_path if baseline else (
            self.work / experiment / variant / f"run-{repetition}"
        )
        if target.exists():
            shutil.rmtree(target)
        with temporary_spark_conf(self.spark, conf or {}):
            source = self.bounded_dwd().select(
                "flight_date",
                "origin",
                "dest",
                "is_cancelled",
                "is_diverted",
                "is_arrival_delayed_15m",
                "arr_delay",
            )
            if transform_source is not None:
                source = transform_source(source)
            result = transform_route_daily(source)
            plan_path = self.output / "plans" / experiment / f"{variant}.formatted.txt"
            if not plan_path.exists():
                _write_plan(result, plan_path)
            before = _wait_for_snapshot(self.spark)
            started = time.monotonic()
            result.write.mode("overwrite").parquet(str(target))
            elapsed = time.monotonic() - started
            after = _wait_for_snapshot(self.spark)
            runtime = runtime_delta(self.spark, before, after)
            profile = self._profile_route(target)
            if profile["duplicate_grains"] != 0:
                raise AssertionError("Route experiment introduced duplicate grains.")
            differences = (
                {"candidate_minus_baseline": 0, "baseline_minus_candidate": 0}
                if baseline
                else self._route_differences(target)
            )
            if any(differences.values()):
                raise AssertionError(
                    f"{experiment}/{variant} differs logically from baseline: "
                    f"{differences}"
                )
            document = {
                "experiment": experiment,
                "variant": variant,
                "repetition": repetition,
                "elapsed_seconds": elapsed,
                "spark_configuration": _configuration(self.spark),
                "logical_profile": profile,
                "logical_differences": differences,
                "runtime": runtime,
                "executed_plan": _executed_plan(result),
                "status": "PASS",
            }
            write_json(
                document,
                self.output
                / "runs"
                / experiment
                / variant
                / f"run-{repetition}.json",
            )
            self.runs.append(document)
        if not baseline:
            shutil.rmtree(target)
        return document

    def join_result(self, variant: str) -> DataFrame:
        fact = self.bounded_dwd().select(
            "reporting_airline", "is_arrival_delayed_15m"
        )
        dimension = self.spark.read.parquet(str(self.carrier_dim_path)).select(
            "carrier_key",
            "reporting_airline",
            "dot_id_reporting_airline",
        )
        if variant == "broadcast":
            dimension = F.broadcast(dimension)
        elif variant == "non_broadcast":
            dimension = dimension.hint("merge")
        else:
            raise ValueError(f"Unknown join variant: {variant}")
        return fact.join(dimension, "reporting_airline", "inner").groupBy(
            "carrier_key", "dot_id_reporting_airline"
        ).agg(
            F.count("*").cast("long").alias("scheduled_flights"),
            F.count("is_arrival_delayed_15m")
            .cast("long")
            .alias("eligible_delay_flights"),
            F.sum("is_arrival_delayed_15m").cast("long").alias("delayed_flights"),
        )

    def warm_join_workload(self) -> None:
        with temporary_spark_conf(
            self.spark, {"spark.sql.autoBroadcastJoinThreshold": -1}
        ):
            self.join_result("broadcast").agg(F.sum("scheduled_flights")).first()

    def run_join_variant(
        self, variant: str, repetition: int, baseline_path: Path
    ) -> dict[str, object]:
        experiment = "join_strategy"
        target = self.work / experiment / variant / f"run-{repetition}"
        if target.exists():
            shutil.rmtree(target)
        with temporary_spark_conf(
            self.spark, {"spark.sql.autoBroadcastJoinThreshold": -1}
        ):
            result = self.join_result(variant)
            plan_path = self.output / "plans" / experiment / f"{variant}.formatted.txt"
            if not plan_path.exists():
                _write_plan(result, plan_path)
            before = _wait_for_snapshot(self.spark)
            started = time.monotonic()
            result.write.mode("overwrite").parquet(str(target))
            elapsed = time.monotonic() - started
            after = _wait_for_snapshot(self.spark)
            runtime = runtime_delta(self.spark, before, after)
            frame = self.spark.read.parquet(str(target))
            profile_row = frame.agg(
                F.count("*").alias("rows"),
                F.countDistinct("carrier_key").alias("unique_grains"),
                F.sum("scheduled_flights").alias("scheduled_flights"),
                F.sum("eligible_delay_flights").alias("eligible_delay_flights"),
                F.sum("delayed_flights").alias("delayed_flights"),
            ).first()
            profile = _scalar_row(profile_row)
            profile["duplicate_grains"] = profile["rows"] - profile["unique_grains"]
            profile["partitions"] = _output_partition_count(target)
            if repetition == 1 and variant == "broadcast":
                if baseline_path.exists():
                    shutil.rmtree(baseline_path)
                shutil.copytree(target, baseline_path)
            reference = self.spark.read.parquet(str(baseline_path))
            columns = sorted(frame.columns)
            differences = {
                "candidate_minus_baseline": frame.select(*columns)
                .exceptAll(reference.select(*columns))
                .limit(1)
                .count(),
                "baseline_minus_candidate": reference.select(*columns)
                .exceptAll(frame.select(*columns))
                .limit(1)
                .count(),
            }
            if profile["duplicate_grains"] or any(differences.values()):
                raise AssertionError(
                    f"Join experiment correctness failure: {profile}, {differences}"
                )
            document = {
                "experiment": experiment,
                "variant": variant,
                "repetition": repetition,
                "elapsed_seconds": elapsed,
                "spark_configuration": _configuration(self.spark),
                "logical_profile": profile,
                "logical_differences": differences,
                "runtime": runtime,
                "executed_plan": _executed_plan(result),
                "status": "PASS",
            }
            write_json(
                document,
                self.output
                / "runs"
                / experiment
                / variant
                / f"run-{repetition}.json",
            )
            self.runs.append(document)
        shutil.rmtree(target)
        return document

    def reuse_actions(
        self, frame: DataFrame
    ) -> tuple[dict[str, object], dict[str, object]]:
        first = frame.agg(
            F.count("*").alias("route_daily_rows"),
            F.sum("scheduled_flights").alias("scheduled_flights"),
            F.sum("delayed_flights").alias("delayed_flights"),
        ).first()
        second = (
            frame.groupBy("origin")
            .agg(
                F.sum("scheduled_flights").alias("scheduled_flights"),
                F.sum("delayed_flights").alias("delayed_flights"),
            )
            .agg(
                F.count("*").alias("origin_rows"),
                F.sum("scheduled_flights").alias("scheduled_flights"),
                F.sum("delayed_flights").alias("delayed_flights"),
            )
            .first()
        )
        return _scalar_row(first), _scalar_row(second)

    def warm_reuse_workload(self) -> None:
        self.reuse_actions(self.bounded_route_daily())

    def run_reuse_variant(
        self,
        experiment: str,
        variant: str,
        repetition: int,
        storage_level: StorageLevel | None,
        expected: tuple[dict[str, object], dict[str, object]] | None,
    ) -> tuple[dict[str, object], tuple[dict[str, object], dict[str, object]]]:
        frame = self.bounded_route_daily()
        if variant == "cache":
            frame = frame.cache()
        elif storage_level is not None:
            frame = frame.persist(storage_level)
        aggregation = frame.agg(
            F.count("*").alias("route_daily_rows"),
            F.sum("scheduled_flights").alias("scheduled_flights"),
            F.sum("delayed_flights").alias("delayed_flights"),
        )
        plan_path = self.output / "plans" / experiment / f"{variant}.formatted.txt"
        if not plan_path.exists():
            _write_plan(aggregation, plan_path)
        before = _wait_for_snapshot(self.spark)
        action_started = time.monotonic()
        first = aggregation.first()
        first_elapsed = time.monotonic() - action_started
        after_first = _wait_for_snapshot(self.spark)
        stored_after_first = storage_snapshot(self.spark)
        action_started = time.monotonic()
        second = (
            frame.groupBy("origin")
            .agg(
                F.sum("scheduled_flights").alias("scheduled_flights"),
                F.sum("delayed_flights").alias("delayed_flights"),
            )
            .agg(
                F.count("*").alias("origin_rows"),
                F.sum("scheduled_flights").alias("scheduled_flights"),
                F.sum("delayed_flights").alias("delayed_flights"),
            )
            .first()
        )
        second_elapsed = time.monotonic() - action_started
        after_second = _wait_for_snapshot(self.spark)
        values = (_scalar_row(first), _scalar_row(second))
        if expected is not None and values != expected:
            raise AssertionError(
                f"{experiment}/{variant} reuse results differ from baseline."
            )
        runtime = runtime_delta(self.spark, before, after_second)
        document = {
            "experiment": experiment,
            "variant": variant,
            "repetition": repetition,
            "elapsed_seconds": first_elapsed + second_elapsed,
            "action_seconds": {
                "first": first_elapsed,
                "second": second_elapsed,
            },
            "spark_configuration": _configuration(self.spark),
            "logical_results": {"first": values[0], "second": values[1]},
            "logical_differences": {
                "candidate_minus_baseline": 0,
                "baseline_minus_candidate": 0,
            },
            "runtime": runtime,
            "first_action_runtime": runtime_delta(self.spark, before, after_first),
            "storage_after_first_action": stored_after_first,
            "storage_level": str(frame.storageLevel),
            "executed_plan": _executed_plan(aggregation),
            "status": "PASS",
        }
        write_json(
            document,
            self.output
            / "runs"
            / experiment
            / variant
            / f"run-{repetition}.json",
        )
        self.runs.append(document)
        if variant == "cache" or storage_level is not None:
            frame.unpersist(blocking=True)
        self.spark.catalog.clearCache()
        if _persistent_rdd_count(self.spark):
            raise AssertionError(
                "Persisted RDDs remain after reuse experiment cleanup."
            )
        return document, values

    def run(self) -> dict[str, object]:
        self.prepare()
        print("[M9] Warm route workload", flush=True)
        self.warm_route_workload()
        print("[M9] Baseline", flush=True)
        baseline = self.run_route_variant(
            "baseline", "normal", 1, baseline=True
        )
        print(
            f"[M9] Baseline PASS in {baseline['elapsed_seconds']:.3f}s",
            flush=True,
        )

        route_experiments: list[
            tuple[
                str,
                list[str],
                Callable[[str], Callable[[DataFrame], DataFrame]],
                Mapping[str, object],
            ]
        ] = [
            (
                "repartition_vs_coalesce",
                ["repartition_2", "coalesce_2"],
                lambda variant: (
                    (lambda frame: frame.repartition(2))
                    if variant == "repartition_2"
                    else (lambda frame: frame.coalesce(2))
                ),
                {},
            ),
            (
                "partition_count",
                ["1", "2", "4", "8"],
                lambda variant: lambda frame: frame.repartition(int(variant)),
                {},
            ),
            (
                "shuffle_partitions",
                ["2", "4", "8"],
                lambda _variant: lambda frame: frame,
                {},
            ),
        ]
        for experiment, variants, transform_factory, shared_conf in route_experiments:
            print(f"[M9] {experiment}", flush=True)
            for repetition in range(1, self.repetitions + 1):
                for variant in alternating_order(variants, repetition):
                    conf = dict(shared_conf)
                    if experiment == "shuffle_partitions":
                        conf["spark.sql.shuffle.partitions"] = int(variant)
                    self.run_route_variant(
                        experiment,
                        variant,
                        repetition,
                        transform_source=transform_factory(variant),
                        conf=conf,
                    )

        print("[M9] Warm join workload", flush=True)
        self.warm_join_workload()
        join_baseline = self.work / "join-baseline"
        join_variants = ["broadcast", "non_broadcast"]
        for repetition in range(1, self.repetitions + 1):
            for variant in alternating_order(join_variants, repetition):
                self.run_join_variant(variant, repetition, join_baseline)

        print("[M9] Warm reuse workload", flush=True)
        self.warm_reuse_workload()
        expected_reuse: tuple[dict[str, object], dict[str, object]] | None = None
        cache_variants = ["no_cache", "cache"]
        for repetition in range(1, self.repetitions + 1):
            for variant in alternating_order(cache_variants, repetition):
                _document, values = self.run_reuse_variant(
                    "cache_behavior",
                    variant,
                    repetition,
                    None,
                    expected_reuse,
                )
                if expected_reuse is None:
                    expected_reuse = values

        storage_variants = {
            "memory_only": StorageLevel.MEMORY_ONLY,
            "memory_and_disk": StorageLevel.MEMORY_AND_DISK,
        }
        for repetition in range(1, self.repetitions + 1):
            for variant in alternating_order(list(storage_variants), repetition):
                self.run_reuse_variant(
                    "storage_level",
                    variant,
                    repetition,
                    storage_variants[variant],
                    expected_reuse,
                )

        final_configuration = _configuration(self.spark)
        configuration_restored = final_configuration == self.original_configuration
        persistence_cleared = _persistent_rdd_count(self.spark) == 0
        restore = {
            "initial": self.original_configuration,
            "final": final_configuration,
            "configuration_restored": configuration_restored,
            "persistent_rdd_count": _persistent_rdd_count(self.spark),
            "persistence_cleared": persistence_cleared,
        }
        write_json(restore, self.output / "configuration_restore.json")
        if not configuration_restored or not persistence_cleared:
            raise AssertionError(f"Spark state was not restored: {restore}")

        summary = self.write_summary(baseline, restore)
        if self.work.exists():
            shutil.rmtree(self.work)
        write_json(
            {
                "status": "PASS",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            self.output / "complete.json",
        )
        print("[M9] All experiments PASS", flush=True)
        return summary

    def write_summary(
        self, baseline: Mapping[str, object], restore: Mapping[str, object]
    ) -> dict[str, object]:
        grouped: dict[str, dict[str, list[dict[str, object]]]] = {}
        for run in self.runs:
            if run["experiment"] == "baseline":
                continue
            grouped.setdefault(str(run["experiment"]), {}).setdefault(
                str(run["variant"]), []
            ).append(run)
        experiments: dict[str, object] = {}
        for experiment, variants in grouped.items():
            experiments[experiment] = {
                "variants": {
                    variant: {
                        "median_seconds": median_runtime(runs),
                        "runs_seconds": [run["elapsed_seconds"] for run in runs],
                        "run_count": len(runs),
                        "logical_profile": runs[0].get(
                            "logical_profile", runs[0].get("logical_results")
                        ),
                        "runtime_medians": {
                            key: statistics.median(
                                int(run["runtime"].get(key, 0)) for run in runs
                            )
                            for key in (
                                "job_count",
                                "stage_count",
                                "numTasks",
                                "shuffleReadBytes",
                                "shuffleWriteBytes",
                                "memoryBytesSpilled",
                                "diskBytesSpilled",
                            )
                        },
                    }
                    for variant, runs in variants.items()
                }
            }
        summary = {
            "status": "PASS",
            "scope": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "repetitions": self.repetitions,
            },
            "baseline": {
                "elapsed_seconds": baseline["elapsed_seconds"],
                "logical_profile": baseline["logical_profile"],
                "runtime": baseline["runtime"],
            },
            "experiments": experiments,
            "configuration_restore": restore,
            "protected_test_evaluation_executed": False,
            "production_change_retained": False,
        }
        write_json(summary, self.output / "summary.json")
        csv_path = self.output / "summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "experiment",
                    "variant",
                    "repetition",
                    "elapsed_seconds",
                    "jobs",
                    "stages",
                    "tasks",
                    "shuffle_read_bytes",
                    "shuffle_write_bytes",
                    "memory_spill_bytes",
                    "disk_spill_bytes",
                ),
            )
            writer.writeheader()
            for run in self.runs:
                runtime = run["runtime"]
                writer.writerow(
                    {
                        "experiment": run["experiment"],
                        "variant": run["variant"],
                        "repetition": run["repetition"],
                        "elapsed_seconds": run["elapsed_seconds"],
                        "jobs": runtime.get("job_count"),
                        "stages": runtime.get("stage_count"),
                        "tasks": runtime.get("numTasks"),
                        "shuffle_read_bytes": runtime.get("shuffleReadBytes"),
                        "shuffle_write_bytes": runtime.get("shuffleWriteBytes"),
                        "memory_spill_bytes": runtime.get("memoryBytesSpilled"),
                        "disk_spill_bytes": runtime.get("diskBytesSpilled"),
                    }
                )
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/performance/m9")
    )
    parser.add_argument("--start-date", type=_parse_date, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=_parse_date, default=DEFAULT_END_DATE)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    args = parser.parse_args()

    paths = get_project_paths(load_config())
    output = validate_output_root(args.output, paths.project_root)
    spark = create_spark_session("airline-m9-performance-experiments")
    try:
        runner = PerformanceRunner(
            spark=spark,
            project_root=paths.project_root,
            dwd_path=paths.resolve(paths.dwd_flights),
            carrier_dim_path=paths.resolve(paths.dim_carrier),
            route_daily_path=paths.resolve(paths.dws_route_daily),
            output=output,
            start_date=args.start_date,
            end_date=args.end_date,
            repetitions=args.repetitions,
        )
        runner.run()
    finally:
        spark.catalog.clearCache()
        spark.stop()


if __name__ == "__main__":
    main()

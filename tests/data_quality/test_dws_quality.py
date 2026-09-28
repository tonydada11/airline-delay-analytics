# AI-generated test code: generated with OpenAI Codex.
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterator

import pytest
from pyspark.sql import DataFrame, SparkSession, functions as F

from airline.common.config import load_config
from airline.common.paths import ProjectPaths, get_project_paths
from airline.common.quality import assert_required_columns
from airline.common.spark_session import create_spark_session
from airline.dws import METRIC_COLUMNS
from airline.dws.build_carrier_daily import build_carrier_daily
from airline.dws.build_origin_airport_daily import build_origin_airport_daily
from airline.dws.build_route_daily import build_route_daily


EXPECTED_ROW_COUNTS = {
    "carrier": 10_965,
    "origin": 239_591,
    "route": 3_554_199,
}
GRAINS = {
    "carrier": ["flight_date", "reporting_airline"],
    "origin": ["flight_date", "origin"],
    "route": ["flight_date", "origin", "destination"],
}

pytestmark = pytest.mark.skipif(
    os.environ.get("AIRLINE_RUN_FULL_DATA_QUALITY") != "1",
    reason="set AIRLINE_RUN_FULL_DATA_QUALITY=1 for mandatory M4 acceptance checks",
)


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    session = create_spark_session("airline-m4-full-data-quality")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def snapshot_directory() -> Iterator[Path]:
    shared_root = Path(os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir()))
    shared_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="airline-m4-", dir=shared_root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _assert_same_rows(first: DataFrame, second: DataFrame) -> None:
    columns = sorted(first.columns)
    assert columns == sorted(second.columns)
    first_aligned = first.select(*columns)
    second_aligned = second.select(*columns)
    assert first_aligned.schema == second_aligned.schema
    assert first_aligned.count() == second_aligned.count()
    assert first_aligned.exceptAll(second_aligned).limit(1).count() == 0


def _load_dws(spark: SparkSession, paths: ProjectPaths) -> Dict[str, DataFrame]:
    return {
        "carrier": spark.read.parquet(str(paths.resolve(paths.dws_carrier_daily))),
        "origin": spark.read.parquet(
            str(paths.resolve(paths.dws_origin_airport_daily))
        ),
        "route": spark.read.parquet(str(paths.resolve(paths.dws_route_daily))),
    }


def _output_paths(paths: ProjectPaths) -> Dict[str, Path]:
    return {
        "carrier": paths.resolve(paths.dws_carrier_daily),
        "origin": paths.resolve(paths.dws_origin_airport_daily),
        "route": paths.resolve(paths.dws_route_daily),
    }


def _source_at_output_grain(dwd: DataFrame, name: str) -> DataFrame:
    if name == "route":
        return dwd.withColumn("destination", F.col("dest")).drop("dest")
    return dwd


def _dwd_count_totals(dwd: DataFrame) -> Dict[str, int]:
    completed = (F.col("is_cancelled") == 0) & (F.col("is_diverted") == 0)
    totals = dwd.agg(
        F.count("*").alias("scheduled_flights"),
        F.sum(completed.cast("long")).alias("completed_flights"),
        F.sum("is_cancelled").alias("cancelled_flights"),
        F.sum("is_diverted").alias("diverted_flights"),
        F.count("is_arrival_delayed_15m").alias("eligible_delay_flights"),
        F.sum("is_arrival_delayed_15m").alias("delayed_flights"),
    ).first()
    return totals.asDict()


def _dws_count_totals(dataframe: DataFrame) -> Dict[str, int]:
    totals = dataframe.agg(
        *[F.sum(column).alias(column) for column in METRIC_COLUMNS[:6]]
    ).first()
    return totals.asDict()


def _quality_violations(dataframe: DataFrame, grain: list[str]) -> Dict[str, int]:
    duplicate_grains = (
        dataframe.groupBy(*grain).count().filter(F.col("count") > 1).count()
    )
    null_grain = F.lit(False)
    for column_name in grain:
        null_grain = null_grain | F.col(column_name).isNull()
    metrics = dataframe.agg(
        F.sum(F.when(null_grain, 1).otherwise(0)).alias("null_grain"),
        F.sum(
            F.when(
                F.col("scheduled_flights")
                != (
                    F.col("completed_flights")
                    + F.col("cancelled_flights")
                    + F.col("diverted_flights")
                ),
                1,
            ).otherwise(0)
        ).alias("component_count_mismatch"),
        F.sum(
            F.when(
                (F.col("delayed_flights") > F.col("eligible_delay_flights"))
                | (F.col("eligible_delay_flights") > F.col("completed_flights")),
                1,
            ).otherwise(0)
        ).alias("count_bound_violation"),
        F.sum(
            F.when(
                (
                    F.col("delay_rate").isNotNull()
                    & ~F.col("delay_rate").between(0.0, 1.0)
                )
                | ~F.col("cancellation_rate").between(0.0, 1.0),
                1,
            ).otherwise(0)
        ).alias("rate_bound_violation"),
        F.sum(
            F.when(
                (
                    (F.col("eligible_delay_flights") == 0)
                    & F.col("delay_rate").isNotNull()
                )
                | (
                    (F.col("eligible_delay_flights") > 0)
                    & F.col("delay_rate").isNull()
                ),
                1,
            ).otherwise(0)
        ).alias("delay_rate_null_mismatch"),
        F.sum(
            F.when(
                (F.col("eligible_delay_flights") > 0)
                & (
                    F.abs(
                        F.col("delay_rate")
                        - F.col("delayed_flights")
                        / F.col("eligible_delay_flights")
                    )
                    > 1e-12
                ),
                1,
            ).otherwise(0)
        ).alias("delay_rate_formula_mismatch"),
        F.sum(
            F.when(
                F.abs(
                    F.col("cancellation_rate")
                    - F.col("cancelled_flights") / F.col("scheduled_flights")
                )
                > 1e-12,
                1,
            ).otherwise(0)
        ).alias("cancellation_rate_formula_mismatch"),
    ).first()
    return {"duplicate_grains": duplicate_grains, **metrics.asDict()}


def _weighted_average_reconciliation(
    dwd: DataFrame, dataframe: DataFrame, name: str
) -> Dict[str, float | int]:
    completed = (F.col("is_cancelled") == 0) & (F.col("is_diverted") == 0)
    source = _source_at_output_grain(dwd, name)
    grain = GRAINS[name]
    weights = source.groupBy(*grain).agg(
        F.sum(
            F.when(completed & F.col("arr_delay").isNotNull(), 1).otherwise(0)
        ).alias("known_delay_count")
    )
    joined = dataframe.join(weights, grain, "inner")
    result = joined.agg(
        (
            F.sum(F.col("avg_arrival_delay") * F.col("known_delay_count"))
            / F.sum("known_delay_count")
        ).alias("weighted_average"),
        F.sum(
            F.when(
                ((F.col("known_delay_count") == 0) & F.col("avg_arrival_delay").isNotNull())
                | ((F.col("known_delay_count") > 0) & F.col("avg_arrival_delay").isNull()),
                1,
            ).otherwise(0)
        ).alias("average_null_mismatch"),
    ).first()
    return result.asDict()


def test_full_dws_quality_reconciliation_and_second_build_idempotency(
    spark: SparkSession,
    snapshot_directory: Path,
) -> None:
    config = load_config()
    paths = get_project_paths(config)
    dwd_path = paths.resolve(paths.dwd_flights)
    outputs = _output_paths(paths)
    dwd = spark.read.parquet(str(dwd_path))
    dws = _load_dws(spark, paths)

    dwd_totals = _dwd_count_totals(dwd)
    completed = (F.col("is_cancelled") == 0) & (F.col("is_diverted") == 0)
    expected_average = dwd.agg(
        F.avg(F.when(completed, F.col("arr_delay"))).alias("avg_arrival_delay")
    ).first()["avg_arrival_delay"]

    row_counts: Dict[str, int] = {}
    reconciled_totals: Dict[str, Dict[str, int]] = {}
    quality_violations: Dict[str, Dict[str, int]] = {}
    weighted_averages: Dict[str, Dict[str, float | int]] = {}
    for name, dataframe in dws.items():
        assert_required_columns(dataframe, [*GRAINS[name], *METRIC_COLUMNS])
        assert "known_arrival_delay_flights" not in dataframe.columns
        if name == "route":
            assert "destination" in dataframe.columns
            assert "dest" not in dataframe.columns

        row_counts[name] = dataframe.count()
        assert row_counts[name] == EXPECTED_ROW_COUNTS[name]
        assert row_counts[name] == _source_at_output_grain(dwd, name).select(
            *GRAINS[name]
        ).distinct().count()

        reconciled_totals[name] = _dws_count_totals(dataframe)
        assert reconciled_totals[name] == dwd_totals

        quality_violations[name] = _quality_violations(dataframe, GRAINS[name])
        assert all(count == 0 for count in quality_violations[name].values())

        weighted_averages[name] = _weighted_average_reconciliation(
            dwd, dataframe, name
        )
        assert weighted_averages[name]["average_null_mismatch"] == 0
        assert weighted_averages[name]["weighted_average"] == pytest.approx(
            expected_average, abs=1e-9
        )
        assert not any(path.is_dir() for path in outputs[name].iterdir())

    snapshot_paths = {name: snapshot_directory / name for name in dws}
    for name, dataframe in dws.items():
        dataframe.write.mode("overwrite").parquet(str(snapshot_paths[name]))

    build_carrier_daily(spark, dwd_path, outputs["carrier"])
    build_origin_airport_daily(spark, dwd_path, outputs["origin"])
    build_route_daily(spark, dwd_path, outputs["route"])
    spark.catalog.clearCache()

    second_dws = _load_dws(spark, paths)
    for name, second_dataframe in second_dws.items():
        first_dataframe = spark.read.parquet(str(snapshot_paths[name]))
        _assert_same_rows(first_dataframe, second_dataframe)

    print(f"dws_row_counts={row_counts}")
    print(f"dwd_count_totals={dwd_totals}")
    print(f"dws_reconciled_totals={reconciled_totals}")
    print(f"quality_violations={quality_violations}")
    print(f"dwd_avg_arrival_delay={expected_average}")
    print(f"weighted_dws_averages={weighted_averages}")
    print("dws_second_build_exact_match=True")

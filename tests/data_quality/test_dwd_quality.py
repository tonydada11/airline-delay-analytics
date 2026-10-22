# AI-generated test code: generated with OpenAI Codex.
"""Tests dwd quality behavior."""

import os
from pathlib import Path
from typing import Iterator, Set, Tuple

import pytest
from pyspark.sql import SparkSession, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.quality import assert_required_columns
from airline.common.spark_session import create_spark_session


EXPECTED_ROW_COUNT = 13_926_960
EXPECTED_DELAYED_COUNT = 2_836_665
EXPECTED_NOT_DELAYED_COUNT = 10_871_985
EXPECTED_NULL_TARGET_COUNT = 218_310
EXPECTED_CANCELLED_COUNT = 184_258
EXPECTED_DIVERTED_COUNT = 34_051
EXPECTED_MONTHS = {
    (year, month) for year in (2023, 2024) for month in range(1, 13)
}
CANDIDATE_FLIGHT_KEY = [
    "flight_date",
    "reporting_airline",
    "flight_number_reporting_airline",
    "origin_airport_id",
    "dest_airport_id",
    "crs_dep_time",
]

pytestmark = pytest.mark.skipif(
    os.environ.get("AIRLINE_RUN_FULL_DATA_QUALITY") != "1",
    reason="set AIRLINE_RUN_FULL_DATA_QUALITY=1 for mandatory M2 acceptance checks",
)


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    """Create the test Spark session and stop it after use."""
    session = create_spark_session("airline-m2-full-data-quality")
    yield session
    session.stop()


def _partition_directories(output_directory: Path) -> Set[Tuple[int, int]]:
    """Process a partition directories."""
    return {
        (
            int(year_directory.name.removeprefix("year=")),
            int(month_directory.name.removeprefix("month=")),
        )
        for year_directory in output_directory.glob("year=*")
        for month_directory in year_directory.glob("month=*")
    }


def test_full_dwd_flights_quality(spark: SparkSession) -> None:
    """Exercise the full dwd flights quality case."""
    config = load_config()
    paths = get_project_paths(config)
    ods = spark.read.parquet(str(paths.resolve(paths.ods_flights)))
    output_directory = paths.resolve(paths.dwd_flights)
    dwd = spark.read.parquet(str(output_directory))

    assert_required_columns(
        dwd,
        [
            "flight_date",
            "year",
            "month",
            "is_cancelled",
            "is_diverted",
            "is_arrival_delayed_15m",
            "scheduled_departure_hour",
            "scheduled_departure_minute",
            "scheduled_arrival_hour",
            "scheduled_arrival_minute",
            "_source_file",
            "_ingested_at",
        ],
    )

    expected_target = F.when(
        (F.col("is_cancelled") == 0)
        & (F.col("is_diverted") == 0)
        & F.col("arr_delay_minutes").isNotNull(),
        (F.col("arr_delay_minutes") >= 15).cast("int"),
    ).otherwise(F.lit(None).cast("int"))

    metrics = dwd.agg(
        F.count("*").alias("row_count"),
        F.min("flight_date").alias("min_date"),
        F.max("flight_date").alias("max_date"),
        F.sum(F.when(F.col("flight_date").isNull(), 1).otherwise(0)).alias(
            "null_flight_date"
        ),
        F.sum(
            F.when(
                (F.year("flight_date") != F.col("year"))
                | (F.month("flight_date") != F.col("month")),
                1,
            ).otherwise(0)
        ).alias("date_partition_mismatch"),
        F.sum(F.when(F.col("is_cancelled") == 1, 1).otherwise(0)).alias(
            "cancelled_count"
        ),
        F.sum(F.when(F.col("is_diverted") == 1, 1).otherwise(0)).alias(
            "diverted_count"
        ),
        F.sum(F.when(F.col("is_arrival_delayed_15m") == 1, 1).otherwise(0)).alias(
            "delayed_count"
        ),
        F.sum(F.when(F.col("is_arrival_delayed_15m") == 0, 1).otherwise(0)).alias(
            "not_delayed_count"
        ),
        F.sum(
            F.when(F.col("is_arrival_delayed_15m").isNull(), 1).otherwise(0)
        ).alias("null_target_count"),
        F.sum(
            F.when(
                ~F.col("is_arrival_delayed_15m").eqNullSafe(expected_target), 1
            ).otherwise(0)
        ).alias("target_rule_mismatch"),
    ).first()

    ods_count = ods.count()
    months = {
        (row["year"], row["month"])
        for row in dwd.select("year", "month").distinct().collect()
    }
    duplicate_groups = (
        dwd.groupBy(*CANDIDATE_FLIGHT_KEY)
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    partitions = _partition_directories(output_directory)

    print(f"ods_count={ods_count}")
    print(f"dwd_metrics={metrics}")
    print(f"months={sorted(months)}")
    print(f"partition_directories={sorted(partitions)}")
    print(f"candidate_key_duplicate_groups={duplicate_groups}")

    assert ods_count == EXPECTED_ROW_COUNT
    assert metrics["row_count"] == EXPECTED_ROW_COUNT
    assert metrics["min_date"].isoformat() == "2023-01-01"
    assert metrics["max_date"].isoformat() == "2024-12-31"
    assert metrics["null_flight_date"] == 0
    assert metrics["date_partition_mismatch"] == 0
    assert metrics["cancelled_count"] == EXPECTED_CANCELLED_COUNT
    assert metrics["diverted_count"] == EXPECTED_DIVERTED_COUNT
    assert metrics["delayed_count"] == EXPECTED_DELAYED_COUNT
    assert metrics["not_delayed_count"] == EXPECTED_NOT_DELAYED_COUNT
    assert metrics["null_target_count"] == EXPECTED_NULL_TARGET_COUNT
    assert metrics["target_rule_mismatch"] == 0
    assert months == EXPECTED_MONTHS
    assert partitions == EXPECTED_MONTHS
    assert duplicate_groups == 0

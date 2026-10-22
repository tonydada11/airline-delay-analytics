# AI-generated test code: generated with OpenAI Codex.
"""Tests ads analytics behavior."""

import os
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pytest
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from airline.ads import ADS_METRIC_COLUMNS, COUNT_METRIC_COLUMNS
from airline.ads.build_airport_scorecard import (
    build_airport_scorecard,
    transform_airport_scorecard,
)
from airline.ads.build_carrier_scorecard import (
    build_carrier_scorecard,
    transform_carrier_scorecard,
)
from airline.ads.build_monthly_operations import (
    build_monthly_operations,
    transform_monthly_operations,
)
from airline.ads.build_route_ranking import (
    build_route_ranking,
    transform_route_ranking,
)
from airline.common.spark_session import create_spark_session
from airline.dws.build_carrier_daily import (
    build_carrier_daily,
    transform_carrier_daily,
)
from airline.dws.build_origin_airport_daily import (
    build_origin_airport_daily,
    transform_origin_airport_daily,
)
from airline.dws.build_route_daily import build_route_daily, transform_route_daily


DWD_ADS_SCHEMA = StructType(
    [
        StructField("flight_date", DateType(), True),
        StructField("year", IntegerType(), True),
        StructField("month", IntegerType(), True),
        StructField("reporting_airline", StringType(), True),
        StructField("origin", StringType(), True),
        StructField("dest", StringType(), True),
        StructField("is_cancelled", IntegerType(), True),
        StructField("is_diverted", IntegerType(), True),
        StructField("is_arrival_delayed_15m", IntegerType(), True),
        StructField("arr_delay", DoubleType(), True),
        StructField("arr_delay_minutes", DoubleType(), True),
    ]
)
GRAINS = {
    "monthly": ["year", "month"],
    "carrier": ["reporting_airline"],
    "airport": ["origin"],
    "route": ["origin", "destination"],
}


def _row(
    flight_date: date,
    target: Optional[int],
    arr_delay: Optional[float],
    *,
    carrier: str = "AA",
    origin: str = "JFK",
    destination: str = "LAX",
    cancelled: int = 0,
    diverted: int = 0,
) -> Dict[str, object]:
    """Build one representative flight row."""
    return {
        "flight_date": flight_date,
        "year": flight_date.year,
        "month": flight_date.month,
        "reporting_airline": carrier,
        "origin": origin,
        "dest": destination,
        "is_cancelled": cancelled,
        "is_diverted": diverted,
        "is_arrival_delayed_15m": target,
        "arr_delay": arr_delay,
        "arr_delay_minutes": (
            max(arr_delay, 0.0) if arr_delay is not None else None
        ),
    }


@pytest.fixture(scope="module")
def dwd_rows() -> List[Dict[str, object]]:
    """Create the sample trusted flight records."""
    return [
        _row(date(2024, 1, 1), 1, 30.0),
        _row(date(2024, 1, 1), None, None, cancelled=1),
        _row(
            date(2024, 1, 1),
            0,
            -5.0,
            origin="LAX",
            destination="JFK",
        ),
        _row(date(2024, 1, 2), 0, 0.0),
        _row(date(2024, 1, 2), 0, 10.0),
        _row(date(2024, 1, 2), 0, 0.0),
        _row(
            date(2024, 1, 2),
            1,
            20.0,
            carrier="DL",
            origin="SEA",
            destination="JFK",
        ),
        _row(
            date(2024, 1, 2),
            None,
            None,
            carrier="DL",
            origin="SEA",
            destination="JFK",
        ),
        _row(
            date(2024, 1, 2),
            None,
            None,
            carrier="DL",
            origin="SEA",
            destination="JFK",
            diverted=1,
        ),
        _row(
            date(2024, 2, 1),
            0,
            -10.0,
            carrier="DL",
            origin="SEA",
            destination="JFK",
        ),
    ]


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    """Create the test Spark session and stop it after use."""
    session = create_spark_session("airline-m6-integration-tests")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def shared_tmp_path() -> Iterator[Path]:
    """Return the shared temporary directory for test outputs."""
    shared_root = Path(os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir()))
    shared_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="airline-m6-", dir=shared_root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _dataframe(spark: SparkSession, rows: List[Dict[str, object]]) -> DataFrame:
    """Build a small Spark DataFrame for the test."""
    return spark.createDataFrame(
        [[row.get(field.name) for field in DWD_ADS_SCHEMA.fields] for row in rows],
        DWD_ADS_SCHEMA,
    )


@pytest.fixture(scope="module")
def built_ads(
    spark: SparkSession,
    shared_tmp_path: Path,
    dwd_rows: List[Dict[str, object]],
) -> Dict[str, Path]:
    """Build and return the ADS fixture tables."""
    paths = {
        "dwd": shared_tmp_path / "dwd" / "flights",
        "dws_carrier": shared_tmp_path / "dws" / "carrier_daily",
        "dws_airport": shared_tmp_path / "dws" / "origin_airport_daily",
        "dws_route": shared_tmp_path / "dws" / "route_daily",
        "monthly": shared_tmp_path / "ads" / "monthly_operations",
        "carrier": shared_tmp_path / "ads" / "carrier_scorecard",
        "airport": shared_tmp_path / "ads" / "airport_scorecard",
        "route": shared_tmp_path / "ads" / "route_ranking",
    }
    _dataframe(spark, dwd_rows).write.mode("overwrite").parquet(str(paths["dwd"]))
    build_carrier_daily(spark, paths["dwd"], paths["dws_carrier"])
    build_origin_airport_daily(spark, paths["dwd"], paths["dws_airport"])
    build_route_daily(spark, paths["dwd"], paths["dws_route"])
    _build_all_ads(spark, paths)
    return paths


def _build_all_ads(spark: SparkSession, paths: Dict[str, Path]) -> None:
    """Build all ads."""
    build_monthly_operations(
        spark, paths["dwd"], paths["dws_carrier"], paths["monthly"]
    )
    build_carrier_scorecard(
        spark, paths["dwd"], paths["dws_carrier"], paths["carrier"]
    )
    build_airport_scorecard(
        spark, paths["dwd"], paths["dws_airport"], paths["airport"]
    )
    build_route_ranking(
        spark, paths["dwd"], paths["dws_route"], paths["route"]
    )


def _assert_same_rows(first: DataFrame, second: DataFrame) -> None:
    """Check same rows."""
    assert first.columns == second.columns
    assert [field.dataType for field in first.schema.fields] == [
        field.dataType for field in second.schema.fields
    ]
    assert first.exceptAll(second).count() == 0
    assert second.exceptAll(first).count() == 0


def test_transforms_have_exact_ordered_schemas_and_types(
    spark: SparkSession, dwd_rows: List[Dict[str, object]]
) -> None:
    """Exercise the transforms have exact ordered schemas and types case."""
    dwd = _dataframe(spark, dwd_rows)
    carrier_daily = transform_carrier_daily(dwd)
    airport_daily = transform_origin_airport_daily(dwd)
    route_daily = transform_route_daily(dwd)
    outputs = {
        "monthly": transform_monthly_operations(carrier_daily, dwd),
        "carrier": transform_carrier_scorecard(carrier_daily, dwd),
        "airport": transform_airport_scorecard(airport_daily, dwd),
        "route": transform_route_ranking(route_daily, dwd),
    }

    for name, output in outputs.items():
        assert isinstance(output, DataFrame)
        assert output.columns == [*GRAINS[name], *ADS_METRIC_COLUMNS]
        for column_name in COUNT_METRIC_COLUMNS:
            assert output.schema[column_name].dataType.typeName() == "long"
        for column_name in ADS_METRIC_COLUMNS[6:]:
            assert output.schema[column_name].dataType.typeName() == "double"


def test_weighted_rates_and_arrival_averages_are_not_daily_averages(
    spark: SparkSession, built_ads: Dict[str, Path]
) -> None:
    """Exercise the weighted rates and arrival averages are not daily averages case."""
    january = spark.read.parquet(str(built_ads["monthly"])).filter(
        (F.col("year") == 2024) & (F.col("month") == 1)
    ).first()
    aa = spark.read.parquet(str(built_ads["carrier"])).filter(
        F.col("reporting_airline") == "AA"
    ).first()
    route = spark.read.parquet(str(built_ads["route"])).filter(
        (F.col("origin") == "JFK") & (F.col("destination") == "LAX")
    ).first()

    assert january["scheduled_flights"] == 9
    assert january["completed_flights"] == 7
    assert january["cancelled_flights"] == 1
    assert january["diverted_flights"] == 1
    assert january["eligible_delay_flights"] == 6
    assert january["delayed_flights"] == 2
    assert january["delay_rate"] == pytest.approx(2 / 6)
    assert january["avg_arrival_delay"] == pytest.approx(55 / 6)
    assert january["cancellation_rate"] == pytest.approx(1 / 9)

    assert aa["delay_rate"] == pytest.approx(1 / 5)
    assert aa["avg_arrival_delay"] == pytest.approx(7.0)
    assert route["delay_rate"] == pytest.approx(1 / 4)
    assert route["avg_arrival_delay"] == pytest.approx(10.0)


def test_monthly_counts_reconcile_to_carrier_daily_and_independent_dwd(
    spark: SparkSession, built_ads: Dict[str, Path]
) -> None:
    """Exercise the monthly counts reconcile to carrier daily and independent dwd case."""
    monthly = spark.read.parquet(str(built_ads["monthly"]))
    carrier_daily = spark.read.parquet(str(built_ads["dws_carrier"])).select(
        F.year("flight_date").cast("int").alias("year"),
        F.month("flight_date").cast("int").alias("month"),
        *COUNT_METRIC_COLUMNS,
    )
    dwd = spark.read.parquet(str(built_ads["dwd"]))
    completed = (F.col("is_cancelled") == 0) & (F.col("is_diverted") == 0)
    expected_dwd = dwd.groupBy("year", "month").agg(
        F.count("*").cast("long").alias("scheduled_flights"),
        F.sum(completed.cast("long")).cast("long").alias("completed_flights"),
        F.sum("is_cancelled").cast("long").alias("cancelled_flights"),
        F.sum("is_diverted").cast("long").alias("diverted_flights"),
        F.count("is_arrival_delayed_15m")
        .cast("long")
        .alias("eligible_delay_flights"),
        F.coalesce(F.sum("is_arrival_delayed_15m"), F.lit(0))
        .cast("long")
        .alias("delayed_flights"),
    )
    expected_dws = carrier_daily.groupBy("year", "month").agg(
        *[F.sum(column).cast("long").alias(column) for column in COUNT_METRIC_COLUMNS]
    )
    actual = monthly.select("year", "month", *COUNT_METRIC_COLUMNS)

    _assert_same_rows(actual, expected_dws)
    _assert_same_rows(actual, expected_dwd)


def test_grains_rates_and_directional_routes(
    spark: SparkSession, built_ads: Dict[str, Path]
) -> None:
    """Exercise the grains rates and directional routes case."""
    for name, grain in GRAINS.items():
        dataframe = spark.read.parquet(str(built_ads[name]))
        assert dataframe.count() == dataframe.select(*grain).distinct().count()
        assert dataframe.filter(
            (F.col("delay_rate").isNotNull() & ~F.col("delay_rate").between(0, 1))
            | ~F.col("cancellation_rate").between(0, 1)
        ).count() == 0

    routes = spark.read.parquet(str(built_ads["route"]))
    pairs = {(row["origin"], row["destination"]) for row in routes.collect()}
    forward = routes.filter(
        (F.col("origin") == "JFK") & (F.col("destination") == "LAX")
    ).first()
    reverse = routes.filter(
        (F.col("origin") == "LAX") & (F.col("destination") == "JFK")
    ).first()
    assert {("JFK", "LAX"), ("LAX", "JFK")} <= pairs
    assert forward["scheduled_flights"] == 5
    assert reverse["scheduled_flights"] == 1
    assert reverse["avg_arrival_delay"] == pytest.approx(-5.0)


def test_parquet_round_trip_is_unpartitioned_and_second_build_is_identical(
    spark: SparkSession, built_ads: Dict[str, Path]
) -> None:
    """Exercise the parquet round trip is unpartitioned and second build is identical case."""
    snapshots = {
        name: spark.read.parquet(str(built_ads[name])).cache()
        for name in GRAINS
    }
    try:
        for name, dataframe in snapshots.items():
            dataframe.count()
            assert dataframe.columns == [*GRAINS[name], *ADS_METRIC_COLUMNS]
            assert not any(path.is_dir() for path in built_ads[name].iterdir())

        _build_all_ads(spark, built_ads)

        for name, first in snapshots.items():
            second = spark.read.parquet(str(built_ads[name]))
            _assert_same_rows(first, second)
    finally:
        for dataframe in snapshots.values():
            dataframe.unpersist()

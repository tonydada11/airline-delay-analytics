# AI-generated test code: generated with OpenAI Codex.
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, Iterator, List

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

from airline.common.spark_session import create_spark_session
from airline.dws.daily_metrics import METRIC_COLUMNS, validate_daily_source
from airline.dws.build_carrier_daily import (
    build_carrier_daily,
    transform_carrier_daily,
)
from airline.dws.build_origin_airport_daily import (
    build_origin_airport_daily,
    transform_origin_airport_daily,
)
from airline.dws.build_route_daily import build_route_daily, transform_route_daily


DWD_ANALYTICS_SCHEMA = StructType(
    [
        StructField("flight_date", DateType(), True),
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


def _row(**overrides: object) -> Dict[str, object]:
    row: Dict[str, object] = {
        "flight_date": date(2024, 9, 13),
        "reporting_airline": "AA",
        "origin": "JFK",
        "dest": "LAX",
        "is_cancelled": 0,
        "is_diverted": 0,
        "is_arrival_delayed_15m": 1,
        "arr_delay": 20.0,
        "arr_delay_minutes": 20.0,
    }
    row.update(overrides)
    return row


@pytest.fixture(scope="module")
def dwd_rows() -> List[Dict[str, object]]:
    return [
        _row(),
        _row(
            is_arrival_delayed_15m=0,
            arr_delay=-5.0,
            arr_delay_minutes=0.0,
        ),
        _row(
            origin="LAX",
            dest="JFK",
            is_cancelled=1,
            is_arrival_delayed_15m=None,
            arr_delay=None,
            arr_delay_minutes=None,
        ),
        _row(
            origin="LAX",
            dest="JFK",
            is_diverted=1,
            is_arrival_delayed_15m=None,
            arr_delay=None,
            arr_delay_minutes=None,
        ),
        _row(
            dest="SEA",
            is_arrival_delayed_15m=None,
            arr_delay=None,
            arr_delay_minutes=None,
        ),
        _row(
            flight_date=date(2024, 9, 14),
            reporting_airline="DL",
            origin="SEA",
            dest="JFK",
            is_arrival_delayed_15m=0,
            arr_delay=-10.0,
            arr_delay_minutes=0.0,
        ),
    ]


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    session = create_spark_session("airline-m4-integration-tests")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def shared_tmp_path() -> Iterator[Path]:
    shared_root = Path(os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir()))
    shared_root.mkdir(parents=True, exist_ok=True)
    test_directory = Path(tempfile.mkdtemp(prefix="airline-m4-", dir=shared_root))
    try:
        yield test_directory
    finally:
        shutil.rmtree(test_directory)


def _dataframe(spark: SparkSession, rows: List[Dict[str, object]]) -> DataFrame:
    return spark.createDataFrame(
        [[row.get(field.name) for field in DWD_ANALYTICS_SCHEMA.fields] for row in rows],
        DWD_ANALYTICS_SCHEMA,
    )


@pytest.fixture(scope="module")
def built_dws(
    spark: SparkSession, shared_tmp_path: Path, dwd_rows: List[Dict[str, object]]
) -> Dict[str, Path]:
    paths = {
        "dwd": shared_tmp_path / "dwd" / "flights",
        "carrier": shared_tmp_path / "dws" / "carrier_daily",
        "origin": shared_tmp_path / "dws" / "origin_airport_daily",
        "route": shared_tmp_path / "dws" / "route_daily",
    }
    _dataframe(spark, dwd_rows).write.mode("overwrite").parquet(str(paths["dwd"]))
    build_carrier_daily(spark, paths["dwd"], paths["carrier"])
    build_origin_airport_daily(spark, paths["dwd"], paths["origin"])
    build_route_daily(spark, paths["dwd"], paths["route"])
    return paths


def _stable_rows(dataframe: DataFrame) -> List[str]:
    columns = sorted(dataframe.columns)
    return sorted(repr(row.asDict()) for row in dataframe.select(*columns).collect())


def test_transforms_have_exact_grains_metrics_and_types(
    spark: SparkSession, dwd_rows: List[Dict[str, object]]
) -> None:
    dwd = _dataframe(spark, dwd_rows)
    outputs = {
        "carrier": transform_carrier_daily(dwd),
        "origin": transform_origin_airport_daily(dwd),
        "route": transform_route_daily(dwd),
    }

    assert outputs["carrier"].columns == [
        "flight_date",
        "reporting_airline",
        *METRIC_COLUMNS,
    ]
    assert outputs["origin"].columns == ["flight_date", "origin", *METRIC_COLUMNS]
    assert outputs["route"].columns == [
        "flight_date",
        "origin",
        "destination",
        *METRIC_COLUMNS,
    ]
    assert "dest" not in outputs["route"].columns

    for output in outputs.values():
        for column_name in METRIC_COLUMNS[:6]:
            assert output.schema[column_name].dataType.typeName() == "long"
        for column_name in METRIC_COLUMNS[6:]:
            assert output.schema[column_name].dataType.typeName() == "double"


def test_carrier_metrics_use_arr_delay_and_preserve_early_arrivals(
    spark: SparkSession, built_dws: Dict[str, Path]
) -> None:
    carrier = spark.read.parquet(str(built_dws["carrier"]))
    aa = carrier.filter(
        (F.col("flight_date") == date(2024, 9, 13))
        & (F.col("reporting_airline") == "AA")
    ).first()

    assert aa["scheduled_flights"] == 5
    assert aa["completed_flights"] == 3
    assert aa["cancelled_flights"] == 1
    assert aa["diverted_flights"] == 1
    assert aa["eligible_delay_flights"] == 2
    assert aa["delayed_flights"] == 1
    assert aa["delay_rate"] == pytest.approx(0.5)
    assert aa["avg_arrival_delay"] == pytest.approx(7.5)
    assert aa["avg_arrival_delay"] != pytest.approx(10.0)
    assert aa["cancellation_rate"] == pytest.approx(0.2)


def test_zero_denominator_nulls_and_reverse_routes_are_distinct(
    spark: SparkSession, built_dws: Dict[str, Path]
) -> None:
    routes = spark.read.parquet(str(built_dws["route"]))
    pairs = {
        (row["origin"], row["destination"])
        for row in routes.filter(F.col("flight_date") == date(2024, 9, 13)).collect()
    }
    reverse = routes.filter(
        (F.col("flight_date") == date(2024, 9, 13))
        & (F.col("origin") == "LAX")
        & (F.col("destination") == "JFK")
    ).first()

    assert ("JFK", "LAX") in pairs
    assert ("LAX", "JFK") in pairs
    assert reverse["eligible_delay_flights"] == 0
    assert reverse["delayed_flights"] == 0
    assert reverse["delay_rate"] is None
    assert reverse["avg_arrival_delay"] is None
    assert reverse["cancellation_rate"] == pytest.approx(0.5)


def test_grains_rates_and_components_are_internally_consistent(
    spark: SparkSession, built_dws: Dict[str, Path]
) -> None:
    grains = {
        "carrier": ["flight_date", "reporting_airline"],
        "origin": ["flight_date", "origin"],
        "route": ["flight_date", "origin", "destination"],
    }
    for name, grain in grains.items():
        dataframe = spark.read.parquet(str(built_dws[name]))
        assert dataframe.count() == dataframe.select(*grain).distinct().count()
        assert dataframe.filter(
            F.col("scheduled_flights")
            != (
                F.col("completed_flights")
                + F.col("cancelled_flights")
                + F.col("diverted_flights")
            )
        ).count() == 0
        assert dataframe.filter(
            (F.col("delayed_flights") > F.col("eligible_delay_flights"))
            | (F.col("eligible_delay_flights") > F.col("completed_flights"))
        ).count() == 0
        assert dataframe.filter(
            (F.col("delay_rate").isNotNull() & ~F.col("delay_rate").between(0, 1))
            | ~F.col("cancellation_rate").between(0, 1)
        ).count() == 0
        assert dataframe.filter(
            ((F.col("eligible_delay_flights") == 0) & F.col("delay_rate").isNotNull())
            | ((F.col("eligible_delay_flights") > 0) & F.col("delay_rate").isNull())
        ).count() == 0


def test_aggregate_count_totals_reconcile_to_dwd(
    spark: SparkSession, built_dws: Dict[str, Path]
) -> None:
    dwd = spark.read.parquet(str(built_dws["dwd"]))
    completed = (F.col("is_cancelled") == 0) & (F.col("is_diverted") == 0)
    expected = dwd.agg(
        F.count("*").alias("scheduled_flights"),
        F.sum(completed.cast("long")).alias("completed_flights"),
        F.sum("is_cancelled").alias("cancelled_flights"),
        F.sum("is_diverted").alias("diverted_flights"),
        F.count("is_arrival_delayed_15m").alias("eligible_delay_flights"),
        F.sum("is_arrival_delayed_15m").alias("delayed_flights"),
    ).first()

    for name in ("carrier", "origin", "route"):
        actual = spark.read.parquet(str(built_dws[name])).agg(
            *[F.sum(column).alias(column) for column in METRIC_COLUMNS[:6]]
        ).first()
        assert actual.asDict() == expected.asDict()


def test_validation_rejects_overlapping_statuses_and_ineligible_targets(
    spark: SparkSession,
) -> None:
    overlapping = _row(
        is_cancelled=1,
        is_diverted=1,
        is_arrival_delayed_15m=None,
        arr_delay=None,
    )
    with pytest.raises(AssertionError, match="both cancelled and diverted"):
        validate_daily_source(
            _dataframe(spark, [overlapping]),
            ["flight_date", "reporting_airline"],
        )

    ineligible_target = _row(is_cancelled=1, is_arrival_delayed_15m=1)
    with pytest.raises(AssertionError, match="delay targets on ineligible rows"):
        validate_daily_source(
            _dataframe(spark, [ineligible_target]),
            ["flight_date", "reporting_airline"],
        )


def test_parquet_round_trip_is_unpartitioned_and_content_idempotent(
    spark: SparkSession, built_dws: Dict[str, Path]
) -> None:
    first_rows = {
        name: _stable_rows(spark.read.parquet(str(built_dws[name])))
        for name in ("carrier", "origin", "route")
    }
    for name in first_rows:
        assert not any(path.is_dir() for path in built_dws[name].iterdir())

    build_carrier_daily(spark, built_dws["dwd"], built_dws["carrier"])
    build_origin_airport_daily(spark, built_dws["dwd"], built_dws["origin"])
    build_route_daily(spark, built_dws["dwd"], built_dws["route"])

    for name, expected_rows in first_rows.items():
        actual = spark.read.parquet(str(built_dws[name]))
        assert _stable_rows(actual) == expected_rows

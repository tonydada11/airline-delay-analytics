# AI-generated test code: generated with OpenAI Codex.
import os
import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

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
from airline.dws.build_ml_flight_features import (
    HISTORICAL_FEATURE_COLUMNS,
    LEAKAGE_BLACKLIST,
    OUTPUT_COLUMNS,
    PREDICTOR_COLUMNS,
    build_historical_rate,
    build_ml_flight_features,
    select_eligible_flights,
    transform_ml_flight_features,
    validate_ml_source,
)


DWD_ML_SCHEMA = StructType(
    [
        StructField("flight_date", DateType(), True),
        StructField("reporting_airline", StringType(), True),
        StructField("origin", StringType(), True),
        StructField("dest", StringType(), True),
        StructField("distance", DoubleType(), True),
        StructField("month", IntegerType(), True),
        StructField("day_of_week", IntegerType(), True),
        StructField("scheduled_departure_hour", IntegerType(), True),
        StructField("is_cancelled", IntegerType(), True),
        StructField("is_diverted", IntegerType(), True),
        StructField("is_arrival_delayed_15m", IntegerType(), True),
        StructField("dep_time", IntegerType(), True),
        StructField("dep_delay", DoubleType(), True),
        StructField("arr_delay", DoubleType(), True),
    ]
)


def _row(
    flight_date: date,
    target: Optional[int],
    *,
    carrier: str = "AA",
    origin: str = "JFK",
    destination: str = "LAX",
    distance: float = 2_475.0,
    departure_hour: int = 8,
    cancelled: int = 0,
    diverted: int = 0,
) -> Dict[str, object]:
    return {
        "flight_date": flight_date,
        "reporting_airline": carrier,
        "origin": origin,
        "dest": destination,
        "distance": distance,
        "month": flight_date.month,
        "day_of_week": flight_date.isoweekday(),
        "scheduled_departure_hour": departure_hour,
        "is_cancelled": cancelled,
        "is_diverted": diverted,
        "is_arrival_delayed_15m": target,
        "dep_time": 900,
        "dep_delay": 30.0,
        "arr_delay": 25.0,
    }


@pytest.fixture(scope="module")
def dwd_rows() -> List[Dict[str, object]]:
    return [
        # For 2023-02-01, these are D-31, D-30, and D-1 respectively.
        _row(date(2023, 1, 1), 1),
        _row(date(2023, 1, 2), 1),
        _row(date(2023, 1, 31), 0),
        _row(date(2023, 1, 31), 0, origin="LAX", destination="JFK"),
        # Same-day outcomes must not affect any 2023-02-01 history.
        _row(date(2023, 2, 1), 1, departure_hour=8),
        _row(date(2023, 2, 1), 0, departure_hour=9),
        _row(date(2023, 2, 1), 1, origin="LAX", destination="JFK"),
        _row(
            date(2023, 2, 1),
            0,
            carrier="DL",
            origin="SEA",
            destination="BOS",
        ),
        # All three rows below are ineligible, including non-null outcomes.
        _row(date(2023, 2, 1), 1, cancelled=1, departure_hour=10),
        _row(date(2023, 2, 1), 0, diverted=1, departure_hour=11),
        _row(date(2023, 2, 1), None, departure_hour=12),
        # Exact chronological split boundaries.
        _row(date(2024, 4, 30), 0, carrier="UA"),
        _row(date(2024, 5, 1), 0, carrier="UA"),
        _row(date(2024, 8, 31), 0, carrier="UA"),
        _row(date(2024, 9, 1), 0, carrier="UA"),
        _row(date(2024, 12, 31), 0, carrier="UA"),
    ]


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    session = create_spark_session("airline-m5-integration-tests")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def shared_tmp_path() -> Iterator[Path]:
    shared_root = Path(os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir()))
    shared_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="airline-m5-", dir=shared_root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _dataframe(spark: SparkSession, rows: List[Dict[str, object]]) -> DataFrame:
    return spark.createDataFrame(
        [[row.get(field.name) for field in DWD_ML_SCHEMA.fields] for row in rows],
        DWD_ML_SCHEMA,
    )


@pytest.fixture(scope="module")
def built_features(
    spark: SparkSession,
    shared_tmp_path: Path,
    dwd_rows: List[Dict[str, object]],
) -> Dict[str, Path]:
    paths = {
        "dwd": shared_tmp_path / "dwd" / "flights",
        "features": shared_tmp_path / "dws" / "ml_flight_features",
    }
    _dataframe(spark, dwd_rows).write.mode("overwrite").parquet(str(paths["dwd"]))
    build_ml_flight_features(spark, paths["dwd"], paths["features"])
    return paths


def _eligible(row: Dict[str, object]) -> bool:
    return (
        row["is_cancelled"] == 0
        and row["is_diverted"] == 0
        and row["is_arrival_delayed_15m"] is not None
    )


def _independent_rate(
    rows: Sequence[Dict[str, object]],
    target_date: date,
    entity_pairs: Sequence[tuple[str, object]],
) -> Optional[float]:
    start_date = target_date - timedelta(days=30)
    history = [
        row
        for row in rows
        if _eligible(row)
        and start_date <= row["flight_date"] < target_date
        and all(row[column] == value for column, value in entity_pairs)
    ]
    if not history:
        return None
    return sum(int(row["is_arrival_delayed_15m"]) for row in history) / len(
        history
    )


def test_transform_has_exact_allowlists_and_is_lazy(
    spark: SparkSession, dwd_rows: List[Dict[str, object]]
) -> None:
    features = transform_ml_flight_features(_dataframe(spark, dwd_rows))

    assert isinstance(features, DataFrame)
    assert tuple(features.columns) == OUTPUT_COLUMNS
    assert PREDICTOR_COLUMNS == (
        "reporting_airline",
        "origin",
        "destination",
        "distance",
        "month",
        "day_of_week",
        "scheduled_departure_hour",
        "carrier_delay_rate_30d",
        "origin_delay_rate_30d",
        "destination_delay_rate_30d",
        "route_delay_rate_30d",
    )
    assert LEAKAGE_BLACKLIST.isdisjoint(PREDICTOR_COLUMNS)
    assert LEAKAGE_BLACKLIST.isdisjoint(features.columns)
    assert "flight_date" not in PREDICTOR_COLUMNS
    assert "is_arrival_delayed_15m" not in PREDICTOR_COLUMNS
    assert "split" not in PREDICTOR_COLUMNS
    assert "flight_key" not in features.columns
    assert "destination" in features.columns
    assert "dest" not in features.columns
    for column_name in HISTORICAL_FEATURE_COLUMNS:
        assert features.schema[column_name].dataType.typeName() == "double"


def test_eligibility_target_domain_and_flight_count_reconcile(
    spark: SparkSession,
    built_features: Dict[str, Path],
) -> None:
    dwd = spark.read.parquet(str(built_features["dwd"]))
    features = spark.read.parquet(str(built_features["features"]))
    eligible_count = dwd.filter(
        (F.col("is_cancelled") == 0)
        & (F.col("is_diverted") == 0)
        & F.col("is_arrival_delayed_15m").isNotNull()
    ).count()

    assert features.count() == eligible_count
    assert features.filter(
        F.col("is_arrival_delayed_15m").isNull()
        | ~F.col("is_arrival_delayed_15m").isin(0, 1)
    ).count() == 0


def test_historical_lookup_grains_are_unique_and_joins_do_not_multiply(
    spark: SparkSession, dwd_rows: List[Dict[str, object]]
) -> None:
    eligible = select_eligible_flights(_dataframe(spark, dwd_rows))
    specs = (
        (("reporting_airline",), "carrier_delay_rate_30d"),
        (("origin",), "origin_delay_rate_30d"),
        (("destination",), "destination_delay_rate_30d"),
        (("origin", "destination"), "route_delay_rate_30d"),
    )
    for entity_columns, output_column in specs:
        history = build_historical_rate(eligible, entity_columns, output_column)
        duplicate_keys = (
            history.groupBy("flight_date", *entity_columns)
            .count()
            .filter(F.col("count") != 1)
            .count()
        )
        assert duplicate_keys == 0

    assert transform_ml_flight_features(_dataframe(spark, dwd_rows)).count() == (
        eligible.count()
    )


def test_history_is_exactly_d_minus_30_through_d_minus_1_and_same_for_day(
    spark: SparkSession,
    built_features: Dict[str, Path],
    dwd_rows: List[Dict[str, object]],
) -> None:
    target_date = date(2023, 2, 1)
    features = spark.read.parquet(str(built_features["features"]))
    same_entity_date = features.filter(
        (F.col("flight_date") == target_date)
        & (F.col("reporting_airline") == "AA")
        & (F.col("origin") == "JFK")
        & (F.col("destination") == "LAX")
    ).collect()
    expected = {
        "carrier_delay_rate_30d": _independent_rate(
            dwd_rows, target_date, (("reporting_airline", "AA"),)
        ),
        "origin_delay_rate_30d": _independent_rate(
            dwd_rows, target_date, (("origin", "JFK"),)
        ),
        "destination_delay_rate_30d": _independent_rate(
            dwd_rows, target_date, (("dest", "LAX"),)
        ),
        "route_delay_rate_30d": _independent_rate(
            dwd_rows,
            target_date,
            (("origin", "JFK"), ("dest", "LAX")),
        ),
    }

    assert len(same_entity_date) == 2
    assert expected == {
        "carrier_delay_rate_30d": pytest.approx(1 / 3),
        "origin_delay_rate_30d": pytest.approx(1 / 2),
        "destination_delay_rate_30d": pytest.approx(1 / 2),
        "route_delay_rate_30d": pytest.approx(1 / 2),
    }
    for row in same_entity_date:
        for column_name, expected_rate in expected.items():
            assert row[column_name] == pytest.approx(expected_rate)


def test_no_history_is_null_partial_history_is_valid_and_routes_are_directional(
    spark: SparkSession, built_features: Dict[str, Path]
) -> None:
    features = spark.read.parquet(str(built_features["features"]))
    target_date = date(2023, 2, 1)
    no_history = features.filter(
        (F.col("flight_date") == target_date)
        & (F.col("reporting_airline") == "DL")
    ).first()
    reverse = features.filter(
        (F.col("flight_date") == target_date)
        & (F.col("origin") == "LAX")
        & (F.col("destination") == "JFK")
    ).first()

    assert all(no_history[column] is None for column in HISTORICAL_FEATURE_COLUMNS)
    assert reverse["route_delay_rate_30d"] == pytest.approx(0.0)
    assert reverse["route_delay_rate_30d"] != pytest.approx(0.5)


def test_split_boundaries_are_exact_exhaustive_and_non_overlapping(
    spark: SparkSession, built_features: Dict[str, Path]
) -> None:
    features = spark.read.parquet(str(built_features["features"]))
    expected = {
        date(2023, 1, 1): "train",
        date(2024, 4, 30): "train",
        date(2024, 5, 1): "validation",
        date(2024, 8, 31): "validation",
        date(2024, 9, 1): "test",
        date(2024, 12, 31): "test",
    }
    actual = {
        row["flight_date"]: row["split"]
        for row in features.filter(F.col("flight_date").isin(*expected))
        .select("flight_date", "split")
        .distinct()
        .collect()
    }

    assert actual == expected
    assert features.filter(F.col("split").isNull()).count() == 0
    assert {
        row["split"] for row in features.select("split").distinct().collect()
    } == {"train", "validation", "test"}


def test_validation_rejects_invalid_domains_and_unassigned_dates(
    spark: SparkSession,
) -> None:
    with pytest.raises(AssertionError, match="invalid is_cancelled"):
        validate_ml_source(_dataframe(spark, [_row(date(2023, 1, 1), 0, cancelled=2)]))
    with pytest.raises(AssertionError, match="invalid target"):
        validate_ml_source(_dataframe(spark, [_row(date(2023, 1, 1), 2)]))
    with pytest.raises(AssertionError, match="outside split range"):
        validate_ml_source(_dataframe(spark, [_row(date(2025, 1, 1), 0)]))


def test_parquet_round_trip_is_unpartitioned_and_build_is_rerunnable(
    spark: SparkSession, built_features: Dict[str, Path]
) -> None:
    first = spark.read.parquet(str(built_features["features"]))
    first_count = first.count()
    assert tuple(first.columns) == OUTPUT_COLUMNS
    assert not any(path.is_dir() for path in built_features["features"].iterdir())

    build_ml_flight_features(
        spark, built_features["dwd"], built_features["features"]
    )
    second = spark.read.parquet(str(built_features["features"]))

    assert tuple(second.columns) == OUTPUT_COLUMNS
    assert second.count() == first_count

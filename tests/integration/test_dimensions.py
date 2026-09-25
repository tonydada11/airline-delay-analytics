# AI-generated test code: generated with OpenAI Codex.
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, Iterator, List

import pytest
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import DateType, IntegerType, StringType, StructField, StructType

from airline.common.spark_session import create_spark_session
from airline.dim.build_airport import (
    build_airport_dimension,
    transform_airport_dimension,
    validate_airport_source,
)
from airline.dim.build_carrier import (
    build_carrier_dimension,
    transform_carrier_dimension,
    validate_carrier_source,
)
from airline.dim.build_date import (
    build_date_dimension,
    transform_date_dimension,
    validate_date_source,
)
from airline.dim.build_route import (
    build_route_dimension,
    transform_route_dimension,
    validate_route_source,
)


AIRPORT_ATTRIBUTES = (
    "airport_id",
    "airport_seq_id",
    "city_market_id",
    "city_name",
    "state",
    "state_fips",
    "state_name",
    "wac",
)
DWD_DIMENSION_SCHEMA = StructType(
    [
        StructField("flight_date", DateType(), True),
        StructField("reporting_airline", StringType(), True),
        StructField("dot_id_reporting_airline", IntegerType(), True),
        StructField("iata_code_reporting_airline", StringType(), True),
        StructField("origin", StringType(), True),
        StructField("origin_airport_id", IntegerType(), True),
        StructField("origin_airport_seq_id", IntegerType(), True),
        StructField("origin_city_market_id", IntegerType(), True),
        StructField("origin_city_name", StringType(), True),
        StructField("origin_state", StringType(), True),
        StructField("origin_state_fips", StringType(), True),
        StructField("origin_state_name", StringType(), True),
        StructField("origin_wac", IntegerType(), True),
        StructField("dest", StringType(), True),
        StructField("dest_airport_id", IntegerType(), True),
        StructField("dest_airport_seq_id", IntegerType(), True),
        StructField("dest_city_market_id", IntegerType(), True),
        StructField("dest_city_name", StringType(), True),
        StructField("dest_state", StringType(), True),
        StructField("dest_state_fips", StringType(), True),
        StructField("dest_state_name", StringType(), True),
        StructField("dest_wac", IntegerType(), True),
    ]
)


def _airport(
    code: str,
    airport_id: int,
    airport_seq_id: int,
    city_market_id: int,
    city_name: str,
    state: str,
    state_fips: str,
    state_name: str,
    wac: int,
) -> Dict[str, object]:
    return {
        "code": code,
        "airport_id": airport_id,
        "airport_seq_id": airport_seq_id,
        "city_market_id": city_market_id,
        "city_name": city_name,
        "state": state,
        "state_fips": state_fips,
        "state_name": state_name,
        "wac": wac,
    }


JFK_OLD = _airport("JFK", 12478, 1247805, 31703, "New York, NY", "NY", "36", "New York", 22)
JFK_TIE_LOW = _airport("JFK", 12478, 1247806, 31703, "New York, NY", "NY", "36", "New York", 22)
JFK_TIE_HIGH = _airport("JFK", 12478, 1247807, 31703, "New York, NY", "NY", "36", "New York", 22)
LAX = _airport("LAX", 12892, 1289208, 32575, "Los Angeles, CA", "CA", "06", "California", 91)
SEA = _airport("SEA", 14747, 1474703, 30559, "Seattle, WA", "WA", "53", "Washington", 93)
BOI = _airport("BOI", 10713, 1071302, 30713, "Boise, ID", "ID", "16", "Idaho", 83)


def _row(
    flight_date: date,
    carrier: str,
    dot_id: int,
    origin: Dict[str, object],
    destination: Dict[str, object],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "flight_date": flight_date,
        "reporting_airline": carrier,
        "dot_id_reporting_airline": dot_id,
        "iata_code_reporting_airline": carrier,
    }
    for prefix, airport in (("origin", origin), ("dest", destination)):
        row[prefix] = airport["code"]
        for attribute in AIRPORT_ATTRIBUTES:
            row[f"{prefix}_{attribute}"] = airport[attribute]
    return row


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    session = create_spark_session("airline-m3-integration-tests")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def shared_tmp_path() -> Iterator[Path]:
    shared_root = Path(os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir()))
    shared_root.mkdir(parents=True, exist_ok=True)
    test_directory = Path(tempfile.mkdtemp(prefix="airline-m3-", dir=shared_root))
    try:
        yield test_directory
    finally:
        shutil.rmtree(test_directory)


@pytest.fixture(scope="module")
def dwd_rows() -> List[Dict[str, object]]:
    first = _row(date(2024, 9, 13), "AA", 1, JFK_OLD, LAX)
    return [
        first,
        dict(first),
        _row(date(2024, 9, 14), "AA", 2, LAX, JFK_TIE_LOW),
        _row(date(2024, 9, 14), "AA", 3, JFK_TIE_HIGH, SEA),
        _row(date(2024, 9, 15), "DL", 4, BOI, LAX),
    ]


def _dataframe(spark: SparkSession, rows: List[Dict[str, object]]) -> DataFrame:
    return spark.createDataFrame(
        [[row.get(field.name) for field in DWD_DIMENSION_SCHEMA.fields] for row in rows],
        DWD_DIMENSION_SCHEMA,
    )


@pytest.fixture(scope="module")
def built_dimensions(
    spark: SparkSession, shared_tmp_path: Path, dwd_rows: List[Dict[str, object]]
) -> Dict[str, Path]:
    paths = {
        "dwd": shared_tmp_path / "dwd" / "flights",
        "date": shared_tmp_path / "dim" / "date",
        "carrier": shared_tmp_path / "dim" / "carrier",
        "airport": shared_tmp_path / "dim" / "airport",
        "route": shared_tmp_path / "dim" / "route",
    }
    _dataframe(spark, dwd_rows).write.mode("overwrite").parquet(str(paths["dwd"]))
    build_date_dimension(spark, paths["dwd"], paths["date"])
    build_carrier_dimension(spark, paths["dwd"], paths["carrier"])
    build_airport_dimension(spark, paths["dwd"], paths["airport"])
    build_route_dimension(spark, paths["dwd"], paths["route"])
    return paths


def _stable_rows(dataframe: DataFrame) -> List[str]:
    columns = sorted(dataframe.columns)
    return sorted(repr(row.asDict()) for row in dataframe.select(*columns).collect())


def test_transforms_return_expected_schemas(
    spark: SparkSession, dwd_rows: List[Dict[str, object]]
) -> None:
    dwd = _dataframe(spark, dwd_rows)

    assert transform_date_dimension(dwd).columns == [
        "date_key",
        "flight_date",
        "year",
        "month",
        "day_of_month",
        "day_of_week",
        "day_name",
    ]
    assert transform_carrier_dimension(dwd).columns == [
        "carrier_key",
        "reporting_airline",
        "dot_id_reporting_airline",
        "iata_code_reporting_airline",
    ]
    assert transform_airport_dimension(dwd).columns == [
        "airport_key",
        "airport_code",
        *AIRPORT_ATTRIBUTES,
    ]
    assert transform_route_dimension(dwd).columns == [
        "route_key",
        "origin",
        "destination",
        "origin_airport_key",
        "destination_airport_key",
    ]


def test_date_values_are_readable_and_unambiguous(
    spark: SparkSession, built_dimensions: Dict[str, Path]
) -> None:
    rows = {
        row["flight_date"]: row
        for row in spark.read.parquet(str(built_dimensions["date"])).collect()
    }

    assert len(rows) == 3
    assert rows[date(2024, 9, 13)]["date_key"] == 20240913
    assert rows[date(2024, 9, 13)]["day_of_month"] == 13
    assert rows[date(2024, 9, 13)]["day_of_week"] == 5
    assert rows[date(2024, 9, 13)]["day_name"] == "Friday"
    assert "day" not in rows[date(2024, 9, 13)].asDict()


def test_surrogate_keys_use_readable_natural_key_values(
    spark: SparkSession, dwd_rows: List[Dict[str, object]]
) -> None:
    dwd = _dataframe(spark, dwd_rows)
    carrier = transform_carrier_dimension(dwd).filter(
        F.col("reporting_airline") == "AA"
    ).first()
    airport = transform_airport_dimension(dwd).filter(
        F.col("airport_code") == "JFK"
    ).first()
    route = transform_route_dimension(dwd).filter(
        (F.col("origin") == "JFK") & (F.col("destination") == "LAX")
    ).first()

    assert carrier["carrier_key"] == "AA"
    assert airport["airport_key"] == "JFK"
    assert route["route_key"] == "JFK->LAX"
    assert route["origin_airport_key"] == "JFK"
    assert route["destination_airport_key"] == "LAX"


def test_latest_attributes_use_a_fully_deterministic_tie_break(
    spark: SparkSession, built_dimensions: Dict[str, Path]
) -> None:
    carriers = spark.read.parquet(str(built_dimensions["carrier"]))
    airports = spark.read.parquet(str(built_dimensions["airport"]))

    aa = carriers.filter(F.col("reporting_airline") == "AA").first()
    jfk = airports.filter(F.col("airport_code") == "JFK").first()
    assert aa["dot_id_reporting_airline"] == 3
    assert jfk["airport_seq_id"] == JFK_TIE_HIGH["airport_seq_id"]


def test_grains_keys_directionality_and_airport_union(
    spark: SparkSession, built_dimensions: Dict[str, Path]
) -> None:
    dimensions = {
        name: spark.read.parquet(str(built_dimensions[name]))
        for name in ("date", "carrier", "airport", "route")
    }
    grains = {
        "date": ("date_key", ["flight_date"]),
        "carrier": ("carrier_key", ["reporting_airline"]),
        "airport": ("airport_key", ["airport_code"]),
        "route": ("route_key", ["origin", "destination"]),
    }
    for name, (surrogate_key, natural_key) in grains.items():
        dataframe = dimensions[name]
        assert dataframe.count() == dataframe.select(surrogate_key).distinct().count()
        assert dataframe.count() == dataframe.select(*natural_key).distinct().count()
        assert dataframe.filter(F.col(surrogate_key).isNull()).count() == 0

    routes = {
        (row["origin"], row["destination"])
        for row in dimensions["route"].collect()
    }
    airports = {row["airport_code"] for row in dimensions["airport"].collect()}
    assert ("JFK", "LAX") in routes
    assert ("LAX", "JFK") in routes
    assert airports == {"JFK", "LAX", "SEA", "BOI"}
    assert "BOI" in airports  # origin-only fixture airport
    assert "SEA" in airports  # destination-only fixture airport


def test_all_dwd_natural_keys_resolve_with_left_anti_joins(
    spark: SparkSession, built_dimensions: Dict[str, Path]
) -> None:
    dwd = spark.read.parquet(str(built_dimensions["dwd"]))
    dates = spark.read.parquet(str(built_dimensions["date"]))
    carriers = spark.read.parquet(str(built_dimensions["carrier"]))
    airports = spark.read.parquet(str(built_dimensions["airport"]))
    routes = spark.read.parquet(str(built_dimensions["route"]))

    dwd_airports = dwd.select(F.col("origin").alias("airport_code")).union(
        dwd.select(F.col("dest").alias("airport_code"))
    ).distinct()
    dwd_routes = dwd.select("origin", F.col("dest").alias("destination")).distinct()
    assert dwd.select("flight_date").distinct().join(
        dates.select("flight_date"), "flight_date", "left_anti"
    ).count() == 0
    assert dwd.select("reporting_airline").distinct().join(
        carriers.select("reporting_airline"), "reporting_airline", "left_anti"
    ).count() == 0
    assert dwd_airports.join(
        airports.select("airport_code"), "airport_code", "left_anti"
    ).count() == 0
    assert dwd_routes.join(
        routes.select("origin", "destination"),
        ["origin", "destination"],
        "left_anti",
    ).count() == 0
    assert routes.select(F.col("origin_airport_key").alias("airport_key")).join(
        airports.select("airport_key"), "airport_key", "left_anti"
    ).count() == 0
    assert routes.select(F.col("destination_airport_key").alias("airport_key")).join(
        airports.select("airport_key"), "airport_key", "left_anti"
    ).count() == 0


def test_outputs_are_unpartitioned_and_second_build_is_content_idempotent(
    spark: SparkSession, built_dimensions: Dict[str, Path]
) -> None:
    first_rows = {
        name: _stable_rows(spark.read.parquet(str(built_dimensions[name])))
        for name in ("date", "carrier", "airport", "route")
    }
    for name in first_rows:
        assert not any(path.is_dir() for path in built_dimensions[name].iterdir())

    build_date_dimension(spark, built_dimensions["dwd"], built_dimensions["date"])
    build_carrier_dimension(spark, built_dimensions["dwd"], built_dimensions["carrier"])
    build_airport_dimension(spark, built_dimensions["dwd"], built_dimensions["airport"])
    build_route_dimension(spark, built_dimensions["dwd"], built_dimensions["route"])

    for name, expected_rows in first_rows.items():
        actual = spark.read.parquet(str(built_dimensions[name]))
        assert _stable_rows(actual) == expected_rows


@pytest.mark.parametrize(
    "validator,column_name",
    [
        (validate_date_source, "flight_date"),
        (validate_carrier_source, "reporting_airline"),
        (validate_airport_source, "origin"),
        (validate_route_source, "dest"),
    ],
)
def test_validation_rejects_null_natural_keys(
    spark: SparkSession,
    dwd_rows: List[Dict[str, object]],
    validator: object,
    column_name: str,
) -> None:
    invalid = dict(dwd_rows[0])
    invalid[column_name] = None
    with pytest.raises(AssertionError):
        validator(_dataframe(spark, [invalid]))

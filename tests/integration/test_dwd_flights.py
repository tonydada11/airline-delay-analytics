# AI-generated test code: generated with OpenAI Codex.
"""Tests dwd flights behavior."""

import os
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterator, List

import pytest
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import StringType, StructField, StructType, TimestampType

from airline.common.schemas import BTS_FLIGHTS_SCHEMA
from airline.common.spark_session import create_spark_session
from airline.dwd.build_flights import (
    build_dwd_flights,
    read_ods_flights,
    to_snake_case,
    transform_flights,
    validate_ods_flights,
)
from airline.ods.build_flights import INGESTED_AT_COLUMN, SOURCE_FILE_COLUMN


ODS_SCHEMA = StructType(
    [
        *BTS_FLIGHTS_SCHEMA.fields,
        StructField(SOURCE_FILE_COLUMN, StringType(), True),
        StructField(INGESTED_AT_COLUMN, TimestampType(), True),
    ]
)


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    """Create the test Spark session and stop it after use."""
    session = create_spark_session("airline-m2-integration-tests")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def shared_tmp_path() -> Iterator[Path]:
    """Return the shared temporary directory for test outputs."""
    shared_root = Path(
        os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir())
    )
    shared_root.mkdir(parents=True, exist_ok=True)
    test_directory = Path(tempfile.mkdtemp(prefix="airline-m2-", dir=shared_root))
    try:
        yield test_directory
    finally:
        shutil.rmtree(test_directory)


def _ods_row(**overrides: object) -> Dict[str, object]:
    """Build one source ODS flight row."""
    row = {field.name: None for field in BTS_FLIGHTS_SCHEMA.fields}
    row.update(
        {
            "Year": 2023,
            "Quarter": 1,
            "Month": 1,
            "DayofMonth": 2,
            "DayOfWeek": 1,
            "FlightDate": date(2023, 1, 2),
            "Reporting_Airline": "AA",
            "Flight_Number_Reporting_Airline": 1,
            "OriginAirportID": 10,
            "Origin": "ORD",
            "DestAirportID": 20,
            "Dest": "DFW",
            "CRSDepTime": 2400,
            "CRSArrTime": 5,
            "ArrDelayMinutes": 15.0,
            "Cancelled": 0.0,
            "Diverted": 0.0,
            SOURCE_FILE_COLUMN: "fixture_2023_1.csv",
            INGESTED_AT_COLUMN: datetime(2025, 1, 1, 12, 0, 0),
        }
    )
    row.update(overrides)
    return row


def _dataframe(
    spark: SparkSession,
    rows: List[Dict[str, object]],
    schema: StructType = ODS_SCHEMA,
) -> DataFrame:
    """Build a small Spark DataFrame for the test."""
    return spark.createDataFrame(
        [[row.get(field.name) for field in schema.fields] for row in rows], schema
    )


@pytest.fixture(scope="module")
def ods_rows() -> List[Dict[str, object]]:
    """Create sample source flight rows."""
    delayed = _ods_row()
    duplicate_delayed = dict(delayed)
    return [
        delayed,
        duplicate_delayed,
        _ods_row(
            Year=2024,
            Month=2,
            FlightDate=date(2024, 2, 3),
            DayofMonth=3,
            Flight_Number_Reporting_Airline=2,
            CRSDepTime=1345,
            CRSArrTime=2359,
            ArrDelayMinutes=14.0,
            _source_file="fixture_2024_2.csv",
        ),
        _ods_row(
            Flight_Number_Reporting_Airline=3,
            ArrDelayMinutes=90.0,
            Cancelled=1.0,
        ),
        _ods_row(
            Flight_Number_Reporting_Airline=4,
            ArrDelayMinutes=90.0,
            Diverted=1.0,
        ),
        _ods_row(
            Flight_Number_Reporting_Airline=5,
            CRSDepTime=None,
            CRSArrTime=None,
            ArrDelayMinutes=None,
        ),
    ]


@pytest.fixture(scope="module")
def built_dwd(
    spark: SparkSession,
    shared_tmp_path: Path,
    ods_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    """Build and return trusted flight fixture rows."""
    ods_directory = shared_tmp_path / "ods" / "flights"
    dwd_directory = shared_tmp_path / "dwd" / "flights"
    _dataframe(spark, ods_rows).write.mode("overwrite").parquet(str(ods_directory))
    build_dwd_flights(spark, ods_directory, dwd_directory)
    return {
        "ods": ods_directory,
        "dwd": dwd_directory,
        "row_count": len(ods_rows),
    }


def _stable_rows(dataframe: DataFrame) -> List[str]:
    """Return fixture rows in deterministic order."""
    columns = sorted(dataframe.columns)
    return sorted(repr(row.asDict()) for row in dataframe.select(*columns).collect())


def test_snake_case_conversion_is_deterministic() -> None:
    """Exercise the snake case conversion is deterministic case."""
    assert to_snake_case("DayofMonth") == "day_of_month"
    assert to_snake_case("CRSDepTime") == "crs_dep_time"
    assert to_snake_case("DOT_ID_Reporting_Airline") == "dot_id_reporting_airline"
    assert to_snake_case("Div5TailNum") == "div5_tail_num"
    assert to_snake_case("_source_file") == "_source_file"
    normalized = [to_snake_case(name) for name in ODS_SCHEMA.fieldNames()]
    assert len(normalized) == len(set(normalized))


@pytest.mark.parametrize("column_name", ["Cancelled", "Diverted"])
@pytest.mark.parametrize("invalid_value", [None, -1.0, 2.0])
def test_validation_rejects_invalid_flag_domains(
    spark: SparkSession, column_name: str, invalid_value: object
) -> None:
    """Exercise the validation rejects invalid flag domains case."""
    dataframe = _dataframe(spark, [_ods_row(**{column_name: invalid_value})])

    with pytest.raises(AssertionError, match=column_name):
        validate_ods_flights(dataframe)


@pytest.mark.parametrize("column_name", ["CRSDepTime", "CRSArrTime"])
@pytest.mark.parametrize("invalid_value", [-1, 1260, 2500])
def test_validation_rejects_impossible_hhmm_values(
    spark: SparkSession, column_name: str, invalid_value: int
) -> None:
    """Exercise the validation rejects impossible hhmm values case."""
    dataframe = _dataframe(spark, [_ods_row(**{column_name: invalid_value})])

    with pytest.raises(AssertionError, match=column_name):
        validate_ods_flights(dataframe)


def test_validation_rejects_malformed_hhmm_schema(spark: SparkSession) -> None:
    """Exercise the validation rejects malformed hhmm schema case."""
    malformed_schema = StructType(
        [
            StructField(field.name, StringType(), field.nullable)
            if field.name == "CRSDepTime"
            else field
            for field in ODS_SCHEMA.fields
        ]
    )
    dataframe = _dataframe(spark, [_ods_row(CRSDepTime="1200")], malformed_schema)

    with pytest.raises(AssertionError, match="CRSDepTime.*integer"):
        validate_ods_flights(dataframe)


def test_transform_is_lazy_and_has_trusted_schema(
    spark: SparkSession, ods_rows: List[Dict[str, object]]
) -> None:
    """Exercise the transform is lazy and has trusted schema case."""
    transformed = transform_flights(_dataframe(spark, ods_rows))

    assert isinstance(transformed, DataFrame)
    assert "FlightDate" not in transformed.columns
    assert "Cancelled" not in transformed.columns
    assert "Diverted" not in transformed.columns
    assert "flight_date" in transformed.columns
    assert "is_cancelled" in transformed.columns
    assert "is_diverted" in transformed.columns
    assert "is_arrival_delayed_15m" in transformed.columns
    assert transformed.schema["is_cancelled"].dataType.typeName() == "integer"
    assert transformed.schema["is_diverted"].dataType.typeName() == "integer"
    assert transformed.schema["is_arrival_delayed_15m"].dataType.typeName() == "integer"


def test_target_schedule_and_duplicate_logic(
    spark: SparkSession, built_dwd: Dict[str, object]
) -> None:
    """Exercise the target schedule and duplicate logic case."""
    dwd = spark.read.parquet(str(built_dwd["dwd"]))
    rows = {
        row["flight_number_reporting_airline"]: row
        for row in dwd.orderBy("flight_number_reporting_airline").collect()
    }

    assert dwd.count() == built_dwd["row_count"]
    assert dwd.filter(F.col("flight_number_reporting_airline") == 1).count() == 2
    assert rows[1]["flight_date"] == date(2023, 1, 2)
    assert (rows[1]["year"], rows[1]["month"]) == (2023, 1)
    assert rows[2]["flight_date"] == date(2024, 2, 3)
    assert (rows[2]["year"], rows[2]["month"]) == (2024, 2)
    assert rows[1]["is_arrival_delayed_15m"] == 1
    assert rows[2]["is_arrival_delayed_15m"] == 0
    assert rows[3]["is_arrival_delayed_15m"] is None
    assert rows[4]["is_arrival_delayed_15m"] is None
    assert rows[5]["is_arrival_delayed_15m"] is None
    assert rows[3]["is_cancelled"] == 1
    assert rows[4]["is_diverted"] == 1
    assert (rows[1]["scheduled_departure_hour"], rows[1]["scheduled_departure_minute"]) == (0, 0)
    assert (rows[1]["scheduled_arrival_hour"], rows[1]["scheduled_arrival_minute"]) == (0, 5)
    assert (rows[2]["scheduled_departure_hour"], rows[2]["scheduled_departure_minute"]) == (13, 45)
    assert (rows[2]["scheduled_arrival_hour"], rows[2]["scheduled_arrival_minute"]) == (23, 59)
    assert rows[5]["scheduled_departure_hour"] is None
    assert rows[5]["scheduled_departure_minute"] is None


def test_partition_layout_and_pruning(
    spark: SparkSession, built_dwd: Dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise the partition layout and pruning case."""
    dwd_directory = Path(built_dwd["dwd"])
    partitions = {
        (year_directory.name, month_directory.name)
        for year_directory in dwd_directory.glob("year=*")
        for month_directory in year_directory.glob("month=*")
    }
    assert partitions == {("year=2023", "month=1"), ("year=2024", "month=2")}

    filtered = read_ods_flights(spark, dwd_directory).filter(
        (F.col("year") == 2024) & (F.col("month") == 2)
    )
    filtered.explain(mode="formatted")
    plan = capsys.readouterr().out
    partition_evidence = plan[plan.index("PartitionFilters") :]
    assert "year" in partition_evidence
    assert "month" in partition_evidence


def test_second_build_is_content_idempotent(
    spark: SparkSession, built_dwd: Dict[str, object]
) -> None:
    """Exercise the second build is content idempotent case."""
    first_snapshot = spark.read.parquet(str(built_dwd["dwd"]))
    first_rows = _stable_rows(first_snapshot)

    build_dwd_flights(spark, built_dwd["ods"], built_dwd["dwd"])
    second_snapshot = spark.read.parquet(str(built_dwd["dwd"]))

    assert second_snapshot.count() == built_dwd["row_count"]
    assert _stable_rows(second_snapshot) == first_rows

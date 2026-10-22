# AI-generated test code: generated with OpenAI Codex.
"""Tests ods flights behavior."""

import csv
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import DateType, DoubleType, IntegerType

from airline.common.schemas import BTS_FLIGHTS_SCHEMA
from airline.common.spark_session import create_spark_session
from airline.ods.build_flights import (
    INGESTED_AT_COLUMN,
    SOURCE_FILE_COLUMN,
    add_lineage_columns,
    build_ods_flights,
    read_raw_flights,
)


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    """Create the test Spark session and stop it after use."""
    session = create_spark_session("airline-m1-integration-tests")
    yield session
    session.stop()


@pytest.fixture()
def shared_tmp_path() -> Iterator[Path]:
    """Return the shared temporary directory for test outputs."""
    shared_root = Path(
        os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir())
    )
    shared_root.mkdir(parents=True, exist_ok=True)
    test_directory = Path(tempfile.mkdtemp(prefix="airline-m1-", dir=shared_root))
    try:
        yield test_directory
    finally:
        shutil.rmtree(test_directory)


def _source_row(seed: int) -> Tuple[List[str], Dict[str, object]]:
    """Build one row from the raw source fixture."""
    csv_values: List[str] = []
    expected: Dict[str, object] = {}

    for index, field in enumerate(BTS_FLIGHTS_SCHEMA.fields, start=1):
        if isinstance(field.dataType, IntegerType):
            value = seed * 1_000 + index
            csv_value = str(value)
        elif isinstance(field.dataType, DoubleType):
            value = seed * 1_000 + index + 0.25
            csv_value = str(value)
        elif isinstance(field.dataType, DateType):
            value = date(2024, seed, 2)
            csv_value = value.isoformat()
        else:
            value = f"value_{seed}_{index}"
            csv_value = value

        csv_values.append(csv_value)
        expected[field.name] = value

    return csv_values, expected


def _write_source_file(path: Path, seed: int) -> Dict[str, object]:
    """Write source file."""
    values, expected = _source_row(seed)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*BTS_FLIGHTS_SCHEMA.fieldNames(), ""])
        writer.writerow([*values, ""])
    return expected


@pytest.fixture()
def raw_fixture(shared_tmp_path: Path) -> Tuple[Path, Dict[str, Dict[str, object]]]:
    """Create the pinned raw snapshot fixture."""
    raw_directory = shared_tmp_path / "raw"
    raw_directory.mkdir()
    expected = {
        "month_1.csv": _write_source_file(raw_directory / "month_1.csv", 1),
        "month_2.csv": _write_source_file(raw_directory / "month_2.csv", 2),
    }
    return raw_directory, expected


def test_read_raw_flights_applies_all_109_fields_without_shift(
    spark: SparkSession,
    raw_fixture: Tuple[Path, Dict[str, Dict[str, object]]],
) -> None:
    """Exercise the read raw flights applies all 109 fields without shift case."""
    raw_directory, expected_by_file = raw_fixture

    for source_path in sorted(raw_directory.glob("*.csv")):
        with source_path.open(encoding="utf-8", newline="") as handle:
            physical_rows = list(csv.reader(handle))
        assert len(physical_rows[0]) == 110
        assert len(physical_rows[1]) == 110
        assert physical_rows[0][-1] == ""
        assert physical_rows[1][-1] == ""

    flights = read_raw_flights(spark, raw_directory)

    assert flights.schema == BTS_FLIGHTS_SCHEMA
    assert len(flights.columns) == 109
    actual_rows = add_lineage_columns(flights).collect()
    actual_by_file = {
        row[SOURCE_FILE_COLUMN]: {
            name: row[name] for name in BTS_FLIGHTS_SCHEMA.fieldNames()
        }
        for row in actual_rows
    }
    assert actual_by_file == expected_by_file
    assert {
        row["Div5TailNum"] for row in actual_rows
    } == {"value_1_109", "value_2_109"}


def test_build_writes_parquet_with_lineage_and_is_idempotent(
    spark: SparkSession,
    raw_fixture: Tuple[Path, Dict[str, Dict[str, object]]],
    shared_tmp_path: Path,
) -> None:
    """Exercise the build writes parquet with lineage and is idempotent case."""
    raw_directory, expected_by_file = raw_fixture
    output_directory = shared_tmp_path / "ods" / "flights"
    stable_columns = [*BTS_FLIGHTS_SCHEMA.fieldNames(), SOURCE_FILE_COLUMN]

    build_ods_flights(spark, raw_directory, output_directory)
    first_snapshot = spark.read.parquet(str(output_directory))
    first_rows = [row.asDict() for row in first_snapshot.select(*stable_columns).collect()]

    assert first_snapshot.columns == [
        *BTS_FLIGHTS_SCHEMA.fieldNames(),
        SOURCE_FILE_COLUMN,
        INGESTED_AT_COLUMN,
    ]
    assert first_snapshot.filter(f"{INGESTED_AT_COLUMN} IS NULL").count() == 0
    assert {row[SOURCE_FILE_COLUMN] for row in first_snapshot.collect()} == set(
        expected_by_file
    )

    build_ods_flights(spark, raw_directory, output_directory)
    second_snapshot = spark.read.parquet(str(output_directory))
    second_rows = [
        row.asDict() for row in second_snapshot.select(*stable_columns).collect()
    ]

    assert second_snapshot.count() == len(expected_by_file)
    assert sorted(first_rows, key=lambda row: row["Year"]) == sorted(
        second_rows, key=lambda row: row["Year"]
    )

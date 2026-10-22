# AI-generated test code: generated with OpenAI Codex.
"""Tests ods quality behavior."""

import os
from pathlib import Path
from typing import Iterator, Sequence, Set, Tuple

import pytest
from pyspark.sql import DataFrame, SparkSession, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.quality import assert_required_columns
from airline.common.schemas import BTS_FLIGHTS_SCHEMA
from airline.common.spark_session import create_spark_session
from airline.ods.build_flights import (
    INGESTED_AT_COLUMN,
    SOURCE_FILE_COLUMN,
    add_lineage_columns,
    read_raw_flights,
)


EXPECTED_ROW_COUNT = 13_926_960
EXPECTED_SOURCE_FILES = {
    f"On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{month}.csv"
    for year in (2023, 2024)
    for month in range(1, 13)
}
EXPECTED_MONTHS = {
    (year, month) for year in (2023, 2024) for month in range(1, 13)
}

pytestmark = pytest.mark.skipif(
    os.environ.get("AIRLINE_RUN_FULL_DATA_QUALITY") != "1",
    reason="set AIRLINE_RUN_FULL_DATA_QUALITY=1 for mandatory M1 acceptance checks",
)


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    """Create the test Spark session and stop it after use."""
    session = create_spark_session("airline-m1-full-data-quality")
    yield session
    session.stop()


def _assert_same_rows(
    first: DataFrame, second: DataFrame, columns: Sequence[str]
) -> None:
    """Check same rows."""
    first_aligned = first.select(*columns)
    second_aligned = second.select(*columns)
    assert [
        (field.name, field.dataType) for field in first_aligned.schema.fields
    ] == [
        (field.name, field.dataType) for field in second_aligned.schema.fields
    ]
    assert first_aligned.exceptAll(second_aligned).limit(1).count() == 0


def _distinct_pairs(dataframe: DataFrame, first: str, second: str) -> Set[Tuple[int, int]]:
    """Return distinct key pairs from the fixture."""
    return {
        (row[first], row[second])
        for row in dataframe.select(first, second).distinct().collect()
    }


def test_full_ods_flights_quality(spark: SparkSession) -> None:
    """Exercise the full ods flights quality case."""
    config = load_config()
    paths = get_project_paths(config)
    raw_directory = paths.resolve(paths.raw_csv)
    output_directory = paths.resolve(paths.ods_flights)

    raw_with_lineage = add_lineage_columns(read_raw_flights(spark, raw_directory))
    ods = spark.read.parquet(str(output_directory))

    assert_required_columns(
        ods, [*BTS_FLIGHTS_SCHEMA.fieldNames(), SOURCE_FILE_COLUMN, INGESTED_AT_COLUMN]
    )
    assert ods.columns == [
        *BTS_FLIGHTS_SCHEMA.fieldNames(),
        SOURCE_FILE_COLUMN,
        INGESTED_AT_COLUMN,
    ]

    raw_count = raw_with_lineage.count()
    ods_count = ods.count()
    assert raw_count == EXPECTED_ROW_COUNT
    assert ods_count == EXPECTED_ROW_COUNT

    stable_columns = [*BTS_FLIGHTS_SCHEMA.fieldNames(), SOURCE_FILE_COLUMN]
    _assert_same_rows(raw_with_lineage, ods, stable_columns)
    print(f"raw_count={raw_count}")
    print(f"ods_count={ods_count}")

    source_files = {
        row[SOURCE_FILE_COLUMN]
        for row in ods.select(SOURCE_FILE_COLUMN).distinct().collect()
    }
    months = _distinct_pairs(ods, "Year", "Month")
    airline_count = ods.select("Reporting_Airline").distinct().count()
    missing_lineage = ods.filter(
        F.col(SOURCE_FILE_COLUMN).isNull() | F.col(INGESTED_AT_COLUMN).isNull()
    ).count()

    print(f"source_files={sorted(source_files)}")
    print(f"months={sorted(months)}")
    print(f"airline_count={airline_count}")
    print(f"missing_lineage={missing_lineage}")

    assert source_files == EXPECTED_SOURCE_FILES
    assert months == EXPECTED_MONTHS
    assert airline_count > 1
    assert missing_lineage == 0

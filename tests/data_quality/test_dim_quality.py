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
from airline.dim.build_airport import build_airport_dimension
from airline.dim.build_carrier import build_carrier_dimension
from airline.dim.build_date import build_date_dimension
from airline.dim.build_route import build_route_dimension


EXPECTED_COUNTS = {
    "date": 731,
    "carrier": 15,
    "airport": 355,
    "route": 7_093,
}
DIMENSION_GRAINS = {
    "date": ("date_key", ["flight_date"]),
    "carrier": ("carrier_key", ["reporting_airline"]),
    "airport": ("airport_key", ["airport_code"]),
    "route": ("route_key", ["origin", "destination"]),
}

pytestmark = pytest.mark.skipif(
    os.environ.get("AIRLINE_RUN_FULL_DATA_QUALITY") != "1",
    reason="set AIRLINE_RUN_FULL_DATA_QUALITY=1 for mandatory M3 acceptance checks",
)


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    session = create_spark_session("airline-m3-full-data-quality")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def snapshot_directory() -> Iterator[Path]:
    shared_root = Path(os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir()))
    shared_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="airline-m3-", dir=shared_root))
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


def _load_dimensions(
    spark: SparkSession, paths: ProjectPaths
) -> Dict[str, DataFrame]:
    return {
        "date": spark.read.parquet(str(paths.resolve(paths.dim_date))),
        "carrier": spark.read.parquet(str(paths.resolve(paths.dim_carrier))),
        "airport": spark.read.parquet(str(paths.resolve(paths.dim_airport))),
        "route": spark.read.parquet(str(paths.resolve(paths.dim_route))),
    }


def _anti_join_counts(
    dwd: DataFrame, dimensions: Dict[str, DataFrame]
) -> Dict[str, int]:
    dwd_airports = (
        dwd.select(F.col("origin").alias("airport_code"))
        .union(dwd.select(F.col("dest").alias("airport_code")))
        .distinct()
    )
    dwd_routes = dwd.select(
        "origin", F.col("dest").alias("destination")
    ).distinct()
    return {
        "date": dwd.select("flight_date").distinct().join(
            dimensions["date"].select("flight_date"),
            "flight_date",
            "left_anti",
        ).count(),
        "carrier": dwd.select("reporting_airline").distinct().join(
            dimensions["carrier"].select("reporting_airline"),
            "reporting_airline",
            "left_anti",
        ).count(),
        "airport": dwd_airports.join(
            dimensions["airport"].select("airport_code"),
            "airport_code",
            "left_anti",
        ).count(),
        "route": dwd_routes.join(
            dimensions["route"].select("origin", "destination"),
            ["origin", "destination"],
            "left_anti",
        ).count(),
        "route_origin_airport_key": dimensions["route"].select(
            F.col("origin_airport_key").alias("airport_key")
        ).join(
            dimensions["airport"].select("airport_key"),
            "airport_key",
            "left_anti",
        ).count(),
        "route_destination_airport_key": dimensions["route"].select(
            F.col("destination_airport_key").alias("airport_key")
        ).join(
            dimensions["airport"].select("airport_key"),
            "airport_key",
            "left_anti",
        ).count(),
    }


def test_full_dimensions_quality_and_second_build_idempotency(
    spark: SparkSession,
    snapshot_directory: Path,
) -> None:
    config = load_config()
    paths = get_project_paths(config)
    dwd_path = paths.resolve(paths.dwd_flights)
    output_paths = {
        "date": paths.resolve(paths.dim_date),
        "carrier": paths.resolve(paths.dim_carrier),
        "airport": paths.resolve(paths.dim_airport),
        "route": paths.resolve(paths.dim_route),
    }
    dwd = spark.read.parquet(str(dwd_path))
    dimensions = _load_dimensions(spark, paths)

    assert_required_columns(
        dimensions["date"],
        [
            "date_key",
            "flight_date",
            "year",
            "month",
            "day_of_month",
            "day_of_week",
            "day_name",
        ],
    )
    assert_required_columns(
        dimensions["carrier"],
        [
            "carrier_key",
            "reporting_airline",
            "dot_id_reporting_airline",
            "iata_code_reporting_airline",
        ],
    )
    assert_required_columns(
        dimensions["airport"],
        [
            "airport_key",
            "airport_code",
            "airport_id",
            "airport_seq_id",
            "city_market_id",
            "city_name",
            "state",
            "state_fips",
            "state_name",
            "wac",
        ],
    )
    assert_required_columns(
        dimensions["route"],
        [
            "route_key",
            "origin",
            "destination",
            "origin_airport_key",
            "destination_airport_key",
        ],
    )

    counts: Dict[str, int] = {}
    uniqueness: Dict[str, Dict[str, int]] = {}
    for name, (surrogate_key, natural_key) in DIMENSION_GRAINS.items():
        dataframe = dimensions[name]
        row_count = dataframe.count()
        counts[name] = row_count
        uniqueness[name] = {
            "surrogate_keys": dataframe.select(surrogate_key).distinct().count(),
            "natural_keys": dataframe.select(*natural_key).distinct().count(),
            "null_surrogate_keys": dataframe.filter(
                F.col(surrogate_key).isNull()
            ).count(),
            "null_natural_keys": dataframe.filter(
                F.expr(" OR ".join(f"`{column}` IS NULL" for column in natural_key))
            ).count(),
        }
        assert row_count == EXPECTED_COUNTS[name]
        assert uniqueness[name] == {
            "surrogate_keys": row_count,
            "natural_keys": row_count,
            "null_surrogate_keys": 0,
            "null_natural_keys": 0,
        }
        assert not any(path.is_dir() for path in output_paths[name].iterdir())

    date_mismatches = dimensions["date"].filter(
        (F.col("date_key") != F.date_format("flight_date", "yyyyMMdd").cast("int"))
        | (F.col("year") != F.year("flight_date"))
        | (F.col("month") != F.month("flight_date"))
        | (F.col("day_of_month") != F.dayofmonth("flight_date"))
        | (F.col("day_of_week") != (F.weekday("flight_date") + F.lit(1)))
        | (F.col("day_name") != F.date_format("flight_date", "EEEE"))
    ).count()
    date_range = dimensions["date"].agg(
        F.min("flight_date").alias("min_date"),
        F.max("flight_date").alias("max_date"),
        F.countDistinct("year", "month").alias("month_count"),
    ).first()
    anti_join_counts = _anti_join_counts(dwd, dimensions)
    natural_key_mismatches = {
        "carrier": dimensions["carrier"].filter(
            F.col("carrier_key") != F.col("reporting_airline")
        ).count(),
        "airport": dimensions["airport"].filter(
            F.col("airport_key") != F.col("airport_code")
        ).count(),
        "route": dimensions["route"].filter(
            F.col("route_key")
            != F.concat(F.col("origin"), F.lit("->"), F.col("destination"))
        ).count(),
        "route_origin_airport": dimensions["route"].filter(
            F.col("origin_airport_key") != F.col("origin")
        ).count(),
        "route_destination_airport": dimensions["route"].filter(
            F.col("destination_airport_key") != F.col("destination")
        ).count(),
    }
    snapshot_paths = {
        name: snapshot_directory / name for name in dimensions
    }
    for name, dataframe in dimensions.items():
        dataframe.write.mode("overwrite").parquet(str(snapshot_paths[name]))

    assert date_mismatches == 0
    assert date_range["min_date"].isoformat() == "2023-01-01"
    assert date_range["max_date"].isoformat() == "2024-12-31"
    assert date_range["month_count"] == 24
    assert all(count == 0 for count in anti_join_counts.values())
    assert all(count == 0 for count in natural_key_mismatches.values())

    build_date_dimension(spark, dwd_path, output_paths["date"])
    build_carrier_dimension(spark, dwd_path, output_paths["carrier"])
    build_airport_dimension(spark, dwd_path, output_paths["airport"])
    build_route_dimension(spark, dwd_path, output_paths["route"])
    spark.catalog.clearCache()

    second_dimensions = _load_dimensions(spark, paths)
    for name, second_dataframe in second_dimensions.items():
        first_dataframe = spark.read.parquet(str(snapshot_paths[name]))
        _assert_same_rows(first_dataframe, second_dataframe)

    print(f"dimension_counts={counts}")
    print(f"uniqueness={uniqueness}")
    print(f"referential_integrity_unmatched={anti_join_counts}")
    print(f"natural_key_mismatches={natural_key_mismatches}")
    print("dimension_second_build_exact_match=True")

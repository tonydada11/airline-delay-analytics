"""Build the airport dimension from trusted DWD flights."""

from pathlib import Path
from typing import List, Union

from pyspark.sql import Column, DataFrame, SparkSession, Window, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.quality import assert_not_empty, assert_required_columns
from airline.common.spark_session import create_spark_session


PathLike = Union[str, Path]
NATURAL_KEY = "airport_code"
ATTRIBUTES = (
    "airport_id",
    "airport_seq_id",
    "city_market_id",
    "city_name",
    "state",
    "state_fips",
    "state_name",
    "wac",
)


def _airport_key(column: Column) -> Column:
    return column.cast("string")


def _source_columns(prefix: str, code_column: str) -> List[Column]:
    return [
        F.col("flight_date"),
        F.col(code_column).alias(NATURAL_KEY),
        *[F.col(f"{prefix}_{name}").alias(name) for name in ATTRIBUTES],
    ]


def read_dwd_flights(spark: SparkSession, dwd_directory: PathLike) -> DataFrame:
    """Read trusted flights from DWD Parquet."""

    return spark.read.parquet(str(dwd_directory))


def validate_airport_source(dataframe: DataFrame) -> None:
    """Validate the origin and destination fields required by the dimension."""

    required_columns = ["flight_date", "origin", "dest"]
    required_columns.extend(f"origin_{name}" for name in ATTRIBUTES)
    required_columns.extend(f"dest_{name}" for name in ATTRIBUTES)
    assert_not_empty(dataframe)
    assert_required_columns(dataframe, required_columns)
    if dataframe.filter(F.col("origin").isNull() | F.col("dest").isNull()).take(1):
        raise AssertionError("DWD airport natural keys must not contain null values.")


def transform_airport_dimension(dataframe: DataFrame) -> DataFrame:
    """Union both flight ends and return one deterministic latest row per code."""

    origins = dataframe.select(*_source_columns("origin", "origin"))
    destinations = dataframe.select(*_source_columns("dest", "dest"))
    candidates = origins.unionByName(destinations).distinct()
    latest_first = Window.partitionBy(NATURAL_KEY).orderBy(
        F.col("flight_date").desc_nulls_last(),
        *[F.col(name).desc_nulls_last() for name in ATTRIBUTES],
    )
    canonical = (
        candidates.withColumn("_row_number", F.row_number().over(latest_first))
        .filter(F.col("_row_number") == 1)
        .drop("_row_number", "flight_date")
    )
    return canonical.select(
        _airport_key(F.col(NATURAL_KEY)).alias("airport_key"),
        NATURAL_KEY,
        *ATTRIBUTES,
    )


def write_airport_dimension(dataframe: DataFrame, output_directory: PathLike) -> None:
    """Replace the unpartitioned airport dimension with Parquet output."""

    dataframe.write.mode("overwrite").parquet(str(output_directory))


def build_airport_dimension(
    spark: SparkSession, dwd_directory: PathLike, output_directory: PathLike
) -> None:
    """Validate, transform, and replace the configured airport dimension."""

    flights = read_dwd_flights(spark, dwd_directory)
    validate_airport_source(flights)
    write_airport_dimension(transform_airport_dimension(flights), output_directory)


def main() -> None:
    """Run the configured M3 airport-dimension build."""

    config = load_config()
    paths = get_project_paths(config)
    spark = create_spark_session("airline-m3-build-dim-airport")
    print(f"DWD input: {paths.dwd_flights.as_posix()}/")
    print(f"Airport dimension output: {paths.dim_airport.as_posix()}/")
    try:
        build_airport_dimension(
            spark,
            paths.resolve(paths.dwd_flights),
            paths.resolve(paths.dim_airport),
        )
        print("Airport dimension written successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

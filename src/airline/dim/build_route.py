"""Build the directional route dimension from trusted DWD flights."""

from pathlib import Path
from typing import Union

from pyspark.sql import Column, DataFrame, SparkSession, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.quality import assert_not_empty, assert_required_columns
from airline.common.spark_session import create_spark_session


PathLike = Union[str, Path]
NATURAL_KEY = ("origin", "destination")


def _airport_key(column: Column) -> Column:
    return column.cast("string")


def _route_key(origin: Column, destination: Column) -> Column:
    return F.concat(
        origin.cast("string"),
        F.lit("->"),
        destination.cast("string"),
    )


def read_dwd_flights(spark: SparkSession, dwd_directory: PathLike) -> DataFrame:
    """Read trusted flights from DWD Parquet."""

    return spark.read.parquet(str(dwd_directory))


def validate_route_source(dataframe: DataFrame) -> None:
    """Validate the DWD fields required by the route dimension."""

    assert_not_empty(dataframe)
    assert_required_columns(dataframe, ["origin", "dest"])
    if dataframe.filter(F.col("origin").isNull() | F.col("dest").isNull()).take(1):
        raise AssertionError("DWD route natural keys must not contain null values.")


def transform_route_dimension(dataframe: DataFrame) -> DataFrame:
    """Return one row per directional origin/destination pair."""

    routes = dataframe.select("origin", F.col("dest").alias("destination")).distinct()
    return routes.select(
        _route_key(F.col("origin"), F.col("destination")).alias("route_key"),
        "origin",
        "destination",
        _airport_key(F.col("origin")).alias("origin_airport_key"),
        _airport_key(F.col("destination")).alias("destination_airport_key"),
    )


def write_route_dimension(dataframe: DataFrame, output_directory: PathLike) -> None:
    """Replace the unpartitioned route dimension with Parquet output."""

    dataframe.write.mode("overwrite").parquet(str(output_directory))


def build_route_dimension(
    spark: SparkSession, dwd_directory: PathLike, output_directory: PathLike
) -> None:
    """Validate, transform, and replace the configured route dimension."""

    flights = read_dwd_flights(spark, dwd_directory)
    validate_route_source(flights)
    write_route_dimension(transform_route_dimension(flights), output_directory)


def main() -> None:
    """Run the configured M3 route-dimension build."""

    config = load_config()
    paths = get_project_paths(config)
    spark = create_spark_session("airline-m3-build-dim-route")
    print(f"DWD input: {paths.dwd_flights.as_posix()}/")
    print(f"Route dimension output: {paths.dim_route.as_posix()}/")
    try:
        build_route_dimension(
            spark,
            paths.resolve(paths.dwd_flights),
            paths.resolve(paths.dim_route),
        )
        print("Route dimension written successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

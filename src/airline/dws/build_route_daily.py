"""Build daily directional-route analytics from trusted DWD flights."""

from pathlib import Path
from typing import Union

from pyspark.sql import DataFrame, SparkSession, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.spark_session import create_spark_session
from airline.dws import aggregate_daily_flights, validate_daily_source


PathLike = Union[str, Path]
SOURCE_GRAIN_COLUMNS = ("flight_date", "origin", "dest")
OUTPUT_GRAIN_COLUMNS = ("flight_date", "origin", "destination")


def read_dwd_flights(spark: SparkSession, dwd_directory: PathLike) -> DataFrame:
    """Read trusted flights from DWD Parquet."""

    return spark.read.parquet(str(dwd_directory))


def transform_route_daily(dataframe: DataFrame) -> DataFrame:
    """Return daily directional-route metrics with an intentional rename."""

    with_destination = dataframe.withColumn("destination", F.col("dest")).drop("dest")
    return aggregate_daily_flights(with_destination, OUTPUT_GRAIN_COLUMNS)


def write_route_daily(dataframe: DataFrame, output_directory: PathLike) -> None:
    """Replace the unpartitioned route-daily Parquet table."""

    dataframe.write.mode("overwrite").parquet(str(output_directory))


def build_route_daily(
    spark: SparkSession, dwd_directory: PathLike, output_directory: PathLike
) -> None:
    """Validate, aggregate, and replace route-daily analytics."""

    flights = read_dwd_flights(spark, dwd_directory)
    validate_daily_source(flights, SOURCE_GRAIN_COLUMNS)
    write_route_daily(transform_route_daily(flights), output_directory)


def main() -> None:
    """Run the configured M4 route-daily build."""

    config = load_config()
    paths = get_project_paths(config)
    spark = create_spark_session("airline-m4-build-route-daily")
    print(f"DWD input: {paths.dwd_flights.as_posix()}/")
    print(f"Route-daily output: {paths.dws_route_daily.as_posix()}/")
    try:
        build_route_daily(
            spark,
            paths.resolve(paths.dwd_flights),
            paths.resolve(paths.dws_route_daily),
        )
        print("Route-daily analytics written successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

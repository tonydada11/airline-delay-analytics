"""Build daily origin-airport analytics from trusted DWD flights."""

from pathlib import Path
from typing import Union

from pyspark.sql import DataFrame, SparkSession

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.spark_session import create_spark_session
from airline.dws.daily_metrics import aggregate_daily_flights, validate_daily_source


PathLike = Union[str, Path]
GRAIN_COLUMNS = ("flight_date", "origin")


def read_dwd_flights(spark: SparkSession, dwd_directory: PathLike) -> DataFrame:
    """Read trusted flights from DWD Parquet."""

    return spark.read.parquet(str(dwd_directory))


def transform_origin_airport_daily(dataframe: DataFrame) -> DataFrame:
    """Return daily origin-airport metrics without triggering an action."""

    return aggregate_daily_flights(dataframe, GRAIN_COLUMNS)


def write_origin_airport_daily(
    dataframe: DataFrame, output_directory: PathLike
) -> None:
    """Replace the unpartitioned origin-airport-daily Parquet table."""

    dataframe.write.mode("overwrite").parquet(str(output_directory))


def build_origin_airport_daily(
    spark: SparkSession, dwd_directory: PathLike, output_directory: PathLike
) -> None:
    """Validate, aggregate, and replace origin-airport-daily analytics."""

    flights = read_dwd_flights(spark, dwd_directory)
    validate_daily_source(flights, GRAIN_COLUMNS)
    write_origin_airport_daily(
        transform_origin_airport_daily(flights), output_directory
    )


def main() -> None:
    """Run the configured M4 origin-airport-daily build."""

    config = load_config()
    paths = get_project_paths(config)
    spark = create_spark_session("airline-m4-build-origin-airport-daily")
    print(f"DWD input: {paths.dwd_flights.as_posix()}/")
    print(
        "Origin-airport-daily output: "
        f"{paths.dws_origin_airport_daily.as_posix()}/"
    )
    try:
        build_origin_airport_daily(
            spark,
            paths.resolve(paths.dwd_flights),
            paths.resolve(paths.dws_origin_airport_daily),
        )
        print("Origin-airport-daily analytics written successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

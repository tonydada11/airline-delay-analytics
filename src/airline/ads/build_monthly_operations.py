"""Build monthly airline operations analytics from trusted DWS and DWD."""

from pathlib import Path
from typing import Union

from pyspark.sql import DataFrame, SparkSession, functions as F

from airline.ads import (
    aggregate_ads_metrics,
    read_parquet,
    validate_daily_ads_source,
    validate_dwd_ads_source,
    write_ads_table,
)
from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.spark_session import create_spark_session


PathLike = Union[str, Path]
DWS_SOURCE_GRAIN = ("flight_date", "reporting_airline")
OUTPUT_GRAIN = ("year", "month")


def transform_monthly_operations(
    carrier_daily: DataFrame, dwd_flights: DataFrame
) -> DataFrame:
    """Return monthly system-wide metrics without triggering an action."""

    monthly_daily = carrier_daily.withColumn(
        "year", F.year("flight_date").cast("int")
    ).withColumn("month", F.month("flight_date").cast("int"))
    return aggregate_ads_metrics(monthly_daily, dwd_flights, OUTPUT_GRAIN)


def build_monthly_operations(
    spark: SparkSession,
    dwd_directory: PathLike,
    carrier_daily_directory: PathLike,
    output_directory: PathLike,
) -> None:
    """Validate sources and replace monthly operations analytics."""

    carrier_daily = read_parquet(spark, carrier_daily_directory)
    dwd_flights = read_parquet(spark, dwd_directory)
    validate_daily_ads_source(carrier_daily, DWS_SOURCE_GRAIN)
    validate_dwd_ads_source(dwd_flights, OUTPUT_GRAIN)
    write_ads_table(
        transform_monthly_operations(carrier_daily, dwd_flights),
        output_directory,
    )


def main() -> None:
    """Run the configured M6 monthly operations build."""

    paths = get_project_paths(load_config())
    spark = create_spark_session("airline-m6-build-monthly-operations")
    print(f"DWD input: {paths.dwd_flights.as_posix()}/")
    print(f"Carrier-daily input: {paths.dws_carrier_daily.as_posix()}/")
    print(f"Monthly operations output: {paths.ads_monthly_operations.as_posix()}/")
    try:
        build_monthly_operations(
            spark,
            paths.resolve(paths.dwd_flights),
            paths.resolve(paths.dws_carrier_daily),
            paths.resolve(paths.ads_monthly_operations),
        )
        print("Monthly operations analytics written successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

"""Build full-period reporting-carrier scorecards from DWS and DWD."""

from pathlib import Path
from typing import Union

from pyspark.sql import DataFrame, SparkSession

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
OUTPUT_GRAIN = ("reporting_airline",)


def transform_carrier_scorecard(
    carrier_daily: DataFrame, dwd_flights: DataFrame
) -> DataFrame:
    """Return full-period carrier metrics without triggering an action."""

    return aggregate_ads_metrics(carrier_daily, dwd_flights, OUTPUT_GRAIN)


def build_carrier_scorecard(
    spark: SparkSession,
    dwd_directory: PathLike,
    carrier_daily_directory: PathLike,
    output_directory: PathLike,
) -> None:
    """Validate sources and replace carrier scorecard analytics."""

    carrier_daily = read_parquet(spark, carrier_daily_directory)
    dwd_flights = read_parquet(spark, dwd_directory)
    validate_daily_ads_source(carrier_daily, DWS_SOURCE_GRAIN)
    validate_dwd_ads_source(dwd_flights, OUTPUT_GRAIN)
    write_ads_table(
        transform_carrier_scorecard(carrier_daily, dwd_flights),
        output_directory,
    )


def main() -> None:
    """Run the configured M6 carrier scorecard build."""

    paths = get_project_paths(load_config())
    spark = create_spark_session("airline-m6-build-carrier-scorecard")
    print(f"DWD input: {paths.dwd_flights.as_posix()}/")
    print(f"Carrier-daily input: {paths.dws_carrier_daily.as_posix()}/")
    print(f"Carrier scorecard output: {paths.ads_carrier_scorecard.as_posix()}/")
    try:
        build_carrier_scorecard(
            spark,
            paths.resolve(paths.dwd_flights),
            paths.resolve(paths.dws_carrier_daily),
            paths.resolve(paths.ads_carrier_scorecard),
        )
        print("Carrier scorecard analytics written successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

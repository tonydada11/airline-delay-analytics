"""Build full-period directional-route analytics from DWS and DWD."""

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
DWS_SOURCE_GRAIN = ("flight_date", "origin", "destination")
DWD_SOURCE_GRAIN = ("origin", "dest")
OUTPUT_GRAIN = ("origin", "destination")


def transform_route_ranking(
    route_daily: DataFrame, dwd_flights: DataFrame
) -> DataFrame:
    """Return full-period directional-route metrics lazily."""

    dwd_with_destination = dwd_flights.withColumn(
        "destination", F.col("dest")
    ).drop("dest")
    return aggregate_ads_metrics(route_daily, dwd_with_destination, OUTPUT_GRAIN)


def build_route_ranking(
    spark: SparkSession,
    dwd_directory: PathLike,
    route_daily_directory: PathLike,
    output_directory: PathLike,
) -> None:
    """Validate sources and replace directional-route analytics."""

    route_daily = read_parquet(spark, route_daily_directory)
    dwd_flights = read_parquet(spark, dwd_directory)
    validate_daily_ads_source(route_daily, DWS_SOURCE_GRAIN)
    validate_dwd_ads_source(dwd_flights, DWD_SOURCE_GRAIN)
    write_ads_table(
        transform_route_ranking(route_daily, dwd_flights), output_directory
    )


def main() -> None:
    """Run the configured M6 route ranking build."""

    paths = get_project_paths(load_config())
    spark = create_spark_session("airline-m6-build-route-ranking")
    print(f"DWD input: {paths.dwd_flights.as_posix()}/")
    print(f"Route-daily input: {paths.dws_route_daily.as_posix()}/")
    print(f"Route ranking output: {paths.ads_route_ranking.as_posix()}/")
    try:
        build_route_ranking(
            spark,
            paths.resolve(paths.dwd_flights),
            paths.resolve(paths.dws_route_daily),
            paths.resolve(paths.ads_route_ranking),
        )
        print("Route ranking analytics written successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

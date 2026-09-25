"""Build the flight-date dimension from trusted DWD flights."""

from pathlib import Path
from typing import Union

from pyspark.sql import DataFrame, SparkSession, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.quality import assert_not_empty, assert_required_columns
from airline.common.spark_session import create_spark_session


PathLike = Union[str, Path]
REQUIRED_COLUMNS = ("flight_date",)


def read_dwd_flights(spark: SparkSession, dwd_directory: PathLike) -> DataFrame:
    """Read trusted flights from DWD Parquet."""

    return spark.read.parquet(str(dwd_directory))


def validate_date_source(dataframe: DataFrame) -> None:
    """Validate the DWD fields required by the date dimension."""

    assert_not_empty(dataframe)
    assert_required_columns(dataframe, REQUIRED_COLUMNS)
    if dataframe.filter(F.col("flight_date").isNull()).take(1):
        raise AssertionError("DWD column 'flight_date' must not contain null values.")


def transform_date_dimension(dataframe: DataFrame) -> DataFrame:
    """Return one deterministic row per flight date without triggering actions."""

    dates = dataframe.select("flight_date").distinct()
    return dates.select(
        F.date_format("flight_date", "yyyyMMdd").cast("int").alias("date_key"),
        "flight_date",
        F.year("flight_date").cast("int").alias("year"),
        F.month("flight_date").cast("int").alias("month"),
        F.dayofmonth("flight_date").cast("int").alias("day_of_month"),
        (F.weekday("flight_date") + F.lit(1)).cast("int").alias("day_of_week"),
        F.date_format("flight_date", "EEEE").alias("day_name"),
    )


def write_date_dimension(dataframe: DataFrame, output_directory: PathLike) -> None:
    """Replace the unpartitioned date dimension with Parquet output."""

    dataframe.write.mode("overwrite").parquet(str(output_directory))


def build_date_dimension(
    spark: SparkSession, dwd_directory: PathLike, output_directory: PathLike
) -> None:
    """Validate, transform, and replace the configured date dimension."""

    flights = read_dwd_flights(spark, dwd_directory)
    validate_date_source(flights)
    write_date_dimension(transform_date_dimension(flights), output_directory)


def main() -> None:
    """Run the configured M3 date-dimension build."""

    config = load_config()
    paths = get_project_paths(config)
    spark = create_spark_session("airline-m3-build-dim-date")
    print(f"DWD input: {paths.dwd_flights.as_posix()}/")
    print(f"Date dimension output: {paths.dim_date.as_posix()}/")
    try:
        build_date_dimension(
            spark,
            paths.resolve(paths.dwd_flights),
            paths.resolve(paths.dim_date),
        )
        print("Date dimension written successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

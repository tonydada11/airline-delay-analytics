"""Build the reporting-carrier dimension from trusted DWD flights."""

from pathlib import Path
from typing import Union

from pyspark.sql import Column, DataFrame, SparkSession, Window, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.quality import assert_not_empty, assert_required_columns
from airline.common.spark_session import create_spark_session


PathLike = Union[str, Path]
NATURAL_KEY = "reporting_airline"
ATTRIBUTES = ("dot_id_reporting_airline", "iata_code_reporting_airline")
REQUIRED_COLUMNS = ("flight_date", NATURAL_KEY, *ATTRIBUTES)


def _carrier_key(column: Column) -> Column:
    return column.cast("string")


def read_dwd_flights(spark: SparkSession, dwd_directory: PathLike) -> DataFrame:
    """Read trusted flights from DWD Parquet."""

    return spark.read.parquet(str(dwd_directory))


def validate_carrier_source(dataframe: DataFrame) -> None:
    """Validate the DWD fields required by the carrier dimension."""

    assert_not_empty(dataframe)
    assert_required_columns(dataframe, REQUIRED_COLUMNS)
    if dataframe.filter(F.col(NATURAL_KEY).isNull()).take(1):
        raise AssertionError(f"DWD column '{NATURAL_KEY}' must not contain null values.")


def transform_carrier_dimension(dataframe: DataFrame) -> DataFrame:
    """Return one deterministic latest row per reporting carrier."""

    candidates = dataframe.select("flight_date", NATURAL_KEY, *ATTRIBUTES).distinct()
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
        _carrier_key(F.col(NATURAL_KEY)).alias("carrier_key"),
        NATURAL_KEY,
        *ATTRIBUTES,
    )


def write_carrier_dimension(dataframe: DataFrame, output_directory: PathLike) -> None:
    """Replace the unpartitioned carrier dimension with Parquet output."""

    dataframe.write.mode("overwrite").parquet(str(output_directory))


def build_carrier_dimension(
    spark: SparkSession, dwd_directory: PathLike, output_directory: PathLike
) -> None:
    """Validate, transform, and replace the configured carrier dimension."""

    flights = read_dwd_flights(spark, dwd_directory)
    validate_carrier_source(flights)
    write_carrier_dimension(transform_carrier_dimension(flights), output_directory)


def main() -> None:
    """Run the configured M3 carrier-dimension build."""

    config = load_config()
    paths = get_project_paths(config)
    spark = create_spark_session("airline-m3-build-dim-carrier")
    print(f"DWD input: {paths.dwd_flights.as_posix()}/")
    print(f"Carrier dimension output: {paths.dim_carrier.as_posix()}/")
    try:
        build_carrier_dimension(
            spark,
            paths.resolve(paths.dwd_flights),
            paths.resolve(paths.dim_carrier),
        )
        print("Carrier dimension written successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

"""Build the trusted DWD flight fact from the ODS flight snapshot."""

import re
from pathlib import Path
from typing import Dict, Union

from pyspark.sql import Column, DataFrame, SparkSession, functions as F
from pyspark.sql.types import DateType, DoubleType, IntegerType

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.quality import assert_required_columns
from airline.common.schemas import BTS_FLIGHTS_SCHEMA
from airline.common.spark_session import create_spark_session
from airline.ods.build_flights import INGESTED_AT_COLUMN, SOURCE_FILE_COLUMN


PathLike = Union[str, Path]
IRREGULAR_COLUMN_NAMES: Dict[str, str] = {
    "DayofMonth": "day_of_month",
}
SOURCE_FLAG_COLUMNS = ("Cancelled", "Diverted")
SCHEDULE_COLUMNS = ("CRSDepTime", "CRSArrTime")


def to_snake_case(name: str) -> str:
    """Convert a BTS column name to deterministic snake_case."""

    if name in IRREGULAR_COLUMN_NAMES:
        return IRREGULAR_COLUMN_NAMES[name]
    if name.startswith("_"):
        return name.lower()

    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    second_pass = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass)
    return re.sub(r"_+", "_", second_pass).lower()


def read_ods_flights(spark: SparkSession, ods_directory: PathLike) -> DataFrame:
    """Read the ODS flight snapshot from Parquet."""

    return spark.read.parquet(str(ods_directory))


def _invalid_hhmm(column_name: str) -> Column:
    value = F.col(column_name)
    return value.isNotNull() & (
        (value < 0)
        | (value > 2400)
        | ((value != 2400) & ((value % 100) >= 60))
    )


def validate_ods_flights(dataframe: DataFrame) -> None:
    """Validate source domains required by the DWD transformation."""

    required_columns = [
        *BTS_FLIGHTS_SCHEMA.fieldNames(),
        SOURCE_FILE_COLUMN,
        INGESTED_AT_COLUMN,
    ]
    assert_required_columns(dataframe, required_columns)

    types = {field.name: field.dataType for field in dataframe.schema.fields}
    for flag_column in SOURCE_FLAG_COLUMNS:
        if not isinstance(types[flag_column], DoubleType):
            raise AssertionError(f"ODS column '{flag_column}' must have type double.")
    for schedule_column in SCHEDULE_COLUMNS:
        if not isinstance(types[schedule_column], IntegerType):
            raise AssertionError(f"ODS column '{schedule_column}' must have type integer.")
    if not isinstance(types["FlightDate"], DateType):
        raise AssertionError("ODS column 'FlightDate' must have type date.")

    invalid_counts = dataframe.agg(
        *[
            F.sum(
                F.when(
                    F.col(column_name).isNull()
                    | ~F.col(column_name).isin(0.0, 1.0),
                    1,
                ).otherwise(0)
            ).alias(f"invalid_{column_name}")
            for column_name in SOURCE_FLAG_COLUMNS
        ],
        *[
            F.sum(F.when(_invalid_hhmm(column_name), 1).otherwise(0)).alias(
                f"invalid_{column_name}"
            )
            for column_name in SCHEDULE_COLUMNS
        ],
    ).first()

    for column_name in (*SOURCE_FLAG_COLUMNS, *SCHEDULE_COLUMNS):
        invalid_count = invalid_counts[f"invalid_{column_name}"]
        if invalid_count:
            raise AssertionError(
                f"ODS column '{column_name}' contains {invalid_count} invalid values."
            )


def _scheduled_hour(column_name: str) -> Column:
    value = F.col(column_name)
    return F.when(value.isNull(), F.lit(None).cast("int")).when(
        value == 2400, F.lit(0)
    ).otherwise(F.floor(value / 100).cast("int"))


def _scheduled_minute(column_name: str) -> Column:
    value = F.col(column_name)
    return F.when(value.isNull(), F.lit(None).cast("int")).when(
        value == 2400, F.lit(0)
    ).otherwise((value % 100).cast("int"))


def transform_flights(dataframe: DataFrame) -> DataFrame:
    """Return a lazy trusted-flight transform that retains every input row."""

    renamed_columns = [to_snake_case(name) for name in dataframe.columns]
    if len(renamed_columns) != len(set(renamed_columns)):
        raise ValueError("Snake-case conversion produced duplicate DWD column names.")

    flights = dataframe.select(
        *[
            F.col(source_name).alias(target_name)
            for source_name, target_name in zip(dataframe.columns, renamed_columns)
        ]
    )
    flights = (
        flights.withColumn("flight_date", F.to_date("flight_date"))
        .withColumn("year", F.year("flight_date").cast("int"))
        .withColumn("month", F.month("flight_date").cast("int"))
        .withColumn("is_cancelled", F.col("cancelled").cast("int"))
        .withColumn("is_diverted", F.col("diverted").cast("int"))
        .withColumn(
            "scheduled_departure_hour", _scheduled_hour("crs_dep_time")
        )
        .withColumn(
            "scheduled_departure_minute", _scheduled_minute("crs_dep_time")
        )
        .withColumn("scheduled_arrival_hour", _scheduled_hour("crs_arr_time"))
        .withColumn("scheduled_arrival_minute", _scheduled_minute("crs_arr_time"))
    )

    eligible_arrival = (
        (F.col("is_cancelled") == 0)
        & (F.col("is_diverted") == 0)
        & F.col("arr_delay_minutes").isNotNull()
    )
    flights = flights.withColumn(
        "is_arrival_delayed_15m",
        F.when(
            eligible_arrival,
            (F.col("arr_delay_minutes") >= 15).cast("int"),
        ).otherwise(F.lit(None).cast("int")),
    )
    return flights.drop("cancelled", "diverted")


def write_dwd_flights(dataframe: DataFrame, output_directory: PathLike) -> None:
    """Replace DWD flights with year/month-partitioned Parquet."""

    dataframe.write.mode("overwrite").partitionBy("year", "month").parquet(
        str(output_directory)
    )


def build_dwd_flights(
    spark: SparkSession, ods_directory: PathLike, output_directory: PathLike
) -> None:
    """Validate, transform, and write the configured DWD flight snapshot."""

    ods_flights = read_ods_flights(spark, ods_directory)
    validate_ods_flights(ods_flights)
    trusted_flights = transform_flights(ods_flights)
    write_dwd_flights(trusted_flights, output_directory)


def main() -> None:
    """Run the configured M2 DWD flight build."""

    config = load_config()
    paths = get_project_paths(config)
    spark = create_spark_session("airline-m2-build-dwd-flights")

    print(f"ODS input: {paths.ods_flights.as_posix()}/")
    print(f"DWD output: {paths.dwd_flights.as_posix()}/")

    try:
        build_dwd_flights(
            spark,
            paths.resolve(paths.ods_flights),
            paths.resolve(paths.dwd_flights),
        )
        print("DWD flight snapshot written successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

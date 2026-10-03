"""Build leakage-safe flight-level ML features from trusted DWD flights."""

from datetime import date
from pathlib import Path
from typing import Sequence, Union

from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, SparkSession, Window, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.quality import assert_not_empty, assert_required_columns
from airline.common.spark_session import create_spark_session


PathLike = Union[str, Path]
TARGET_COLUMN = "is_arrival_delayed_15m"
SPLIT_COLUMN = "split"
BASE_FEATURE_COLUMNS = (
    "reporting_airline",
    "origin",
    "destination",
    "distance",
    "month",
    "day_of_week",
    "scheduled_departure_hour",
)
HISTORICAL_FEATURE_COLUMNS = (
    "carrier_delay_rate_30d",
    "origin_delay_rate_30d",
    "destination_delay_rate_30d",
    "route_delay_rate_30d",
)
PREDICTOR_COLUMNS = (*BASE_FEATURE_COLUMNS, *HISTORICAL_FEATURE_COLUMNS)
OUTPUT_COLUMNS = (
    "flight_date",
    *PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    SPLIT_COLUMN,
)
LEAKAGE_BLACKLIST = frozenset(
    {
        "dep_time",
        "dep_delay",
        "dep_delay_minutes",
        "taxi_out",
        "wheels_off",
        "wheels_on",
        "arr_time",
        "arr_delay",
        "arr_delay_minutes",
        "actual_elapsed_time",
        "air_time",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay",
    }
)
TRAIN_START = date(2023, 1, 1)
TRAIN_END = date(2024, 4, 30)
VALIDATION_START = date(2024, 5, 1)
VALIDATION_END = date(2024, 8, 31)
TEST_START = date(2024, 9, 1)
TEST_END = date(2024, 12, 31)
HISTORY_DAYS = 30


def eligibility_condition() -> Column:
    """Return the locked ML eligibility predicate."""

    return (
        (F.col("is_cancelled") == 0)
        & (F.col("is_diverted") == 0)
        & F.col(TARGET_COLUMN).isNotNull()
    )


def read_dwd_flights(spark: SparkSession, dwd_directory: PathLike) -> DataFrame:
    """Read trusted flights from DWD Parquet."""

    return spark.read.parquet(str(dwd_directory))


def validate_ml_source(dataframe: DataFrame) -> None:
    """Validate DWD fields required by the ML feature build."""

    required_columns = {
        "flight_date",
        "reporting_airline",
        "origin",
        "dest",
        "distance",
        "month",
        "day_of_week",
        "scheduled_departure_hour",
        "is_cancelled",
        "is_diverted",
        TARGET_COLUMN,
    }
    assert_required_columns(dataframe, required_columns)
    assert_not_empty(dataframe)

    eligible = eligibility_condition()
    metrics = dataframe.agg(
        F.sum(
            F.when(
                F.col("is_cancelled").isNull()
                | ~F.col("is_cancelled").isin(0, 1),
                1,
            ).otherwise(0)
        ).alias("invalid_is_cancelled"),
        F.sum(
            F.when(
                F.col("is_diverted").isNull()
                | ~F.col("is_diverted").isin(0, 1),
                1,
            ).otherwise(0)
        ).alias("invalid_is_diverted"),
        F.sum(
            F.when(
                F.col(TARGET_COLUMN).isNotNull()
                & ~F.col(TARGET_COLUMN).isin(0, 1),
                1,
            ).otherwise(0)
        ).alias("invalid_target"),
        F.sum(
            F.when(
                eligible
                & (
                    F.col("flight_date").isNull()
                    | (F.col("flight_date") < F.lit(TRAIN_START))
                    | (F.col("flight_date") > F.lit(TEST_END))
                ),
                1,
            ).otherwise(0)
        ).alias("eligible_date_outside_split_range"),
    ).first()

    failures = {
        "invalid is_cancelled values": metrics["invalid_is_cancelled"],
        "invalid is_diverted values": metrics["invalid_is_diverted"],
        "invalid target values": metrics["invalid_target"],
        "eligible dates outside split range": metrics[
            "eligible_date_outside_split_range"
        ],
    }
    present = [f"{label}: {count}" for label, count in failures.items() if count]
    if present:
        raise AssertionError("Invalid trusted DWD input: " + "; ".join(present))


def select_eligible_flights(dataframe: DataFrame) -> DataFrame:
    """Return the lazy, exact eligible-flight projection."""

    return dataframe.filter(eligibility_condition()).select(
        "flight_date",
        "reporting_airline",
        "origin",
        F.col("dest").alias("destination"),
        "distance",
        "month",
        "day_of_week",
        "scheduled_departure_hour",
        TARGET_COLUMN,
    )


def build_historical_rate(
    eligible_flights: DataFrame,
    entity_columns: Sequence[str],
    output_column: str,
) -> DataFrame:
    """Return a prior-30-calendar-day rate at entity/date grain."""

    daily = eligible_flights.groupBy("flight_date", *entity_columns).agg(
        F.count(F.lit(1)).cast("long").alias("_eligible_count"),
        F.sum(TARGET_COLUMN).cast("long").alias("_delayed_count"),
    )
    dated = daily.withColumn(
        "_date_ordinal",
        F.datediff(F.col("flight_date"), F.lit(date(1970, 1, 1))),
    )
    history_window = (
        Window.partitionBy(*entity_columns)
        .orderBy(F.col("_date_ordinal"))
        .rangeBetween(-HISTORY_DAYS, -1)
    )
    historical = dated.withColumn(
        "_historical_eligible_count",
        F.sum("_eligible_count").over(history_window),
    ).withColumn(
        "_historical_delayed_count",
        F.sum("_delayed_count").over(history_window),
    )
    return historical.select(
        "flight_date",
        *entity_columns,
        F.when(
            F.col("_historical_eligible_count") > 0,
            F.col("_historical_delayed_count").cast("double")
            / F.col("_historical_eligible_count").cast("double"),
        )
        .otherwise(F.lit(None).cast("double"))
        .alias(output_column),
    )


def add_historical_features(eligible_flights: DataFrame) -> DataFrame:
    """Attach the four entity/date histories without changing flight grain."""

    history_specs = (
        (("reporting_airline",), "carrier_delay_rate_30d"),
        (("origin",), "origin_delay_rate_30d"),
        (("destination",), "destination_delay_rate_30d"),
        (("origin", "destination"), "route_delay_rate_30d"),
    )
    featured = eligible_flights
    for entity_columns, output_column in history_specs:
        history = build_historical_rate(
            eligible_flights, entity_columns, output_column
        )
        featured = featured.join(
            history,
            ["flight_date", *entity_columns],
            "left",
        )
    return featured


def assign_chronological_split(dataframe: DataFrame) -> DataFrame:
    """Assign the locked, non-overlapping calendar-date splits."""

    split = (
        F.when(
            F.col("flight_date").between(F.lit(TRAIN_START), F.lit(TRAIN_END)),
            F.lit("train"),
        )
        .when(
            F.col("flight_date").between(
                F.lit(VALIDATION_START), F.lit(VALIDATION_END)
            ),
            F.lit("validation"),
        )
        .when(
            F.col("flight_date").between(F.lit(TEST_START), F.lit(TEST_END)),
            F.lit("test"),
        )
        .otherwise(F.lit(None).cast("string"))
    )
    return dataframe.withColumn(SPLIT_COLUMN, split)


def assemble_ml_flight_features(eligible_flights: DataFrame) -> DataFrame:
    """Return the lazy feature and split projection for eligible flights."""

    return assign_chronological_split(
        add_historical_features(eligible_flights)
    ).select(*OUTPUT_COLUMNS)


def transform_ml_flight_features(dataframe: DataFrame) -> DataFrame:
    """Return the complete lazy DWD-to-M5 transformation."""

    return assemble_ml_flight_features(select_eligible_flights(dataframe))


def write_ml_flight_features(
    dataframe: DataFrame, output_directory: PathLike
) -> None:
    """Overwrite an unpartitioned Parquet dataset at the full output path."""

    dataframe.write.mode("overwrite").parquet(str(output_directory))


def build_ml_flight_features(
    spark: SparkSession, dwd_directory: PathLike, output_directory: PathLike
) -> None:
    """Validate DWD, build M5, and overwrite the configured output."""

    dwd_flights = read_dwd_flights(spark, dwd_directory)
    validate_ml_source(dwd_flights)
    eligible_flights = select_eligible_flights(dwd_flights).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    try:
        assert_not_empty(eligible_flights)
        write_ml_flight_features(
            assemble_ml_flight_features(eligible_flights), output_directory
        )
    finally:
        eligible_flights.unpersist()


def main() -> None:
    """Run the configured M5 feature build."""

    config = load_config()
    paths = get_project_paths(config)
    spark = create_spark_session("airline-m5-build-ml-flight-features")
    print(f"DWD input: {paths.dwd_flights.as_posix()}/")
    print(f"ML feature output: {paths.dws_ml_flight_features.as_posix()}/")
    try:
        build_ml_flight_features(
            spark,
            paths.resolve(paths.dwd_flights),
            paths.resolve(paths.dws_ml_flight_features),
        )
        print("ML flight feature dataset written successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

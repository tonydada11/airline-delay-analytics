"""Build production features from DWD; histories include only days D-N through D-1."""
import math
from pathlib import Path
from typing import Union
from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, SparkSession, Window, functions as F
from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.quality import assert_not_empty, assert_required_columns
from airline.common.spark_session import create_spark_session
from airline.dws.ml_feature_contract import (
    TARGET_COLUMN, SPLIT_COLUMN, TRAIN_START, TRAIN_END, VALIDATION_START,
    VALIDATION_END, TEST_START, TEST_END, ENTITIES, WINDOW_DAYS, OUTPUT_COLUMNS,
    CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, HISTORY_COUNTS, HISTORY_MISSING,
    LEAKAGE_BLACKLIST, SPLIT_BOUNDARIES,
)
PathLike = Union[str, Path]

def eligibility_condition() -> Column:
    """Return the locked ML eligibility predicate."""

    return (
        (F.col("is_cancelled") == 0)
        & (F.col("is_diverted") == 0)
        & F.col(TARGET_COLUMN).isin(0, 1)
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
        "scheduled_departure_minute", "scheduled_arrival_hour",
        "scheduled_arrival_minute", "crs_elapsed_time",
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
    return dataframe.filter(eligibility_condition()).select(
        "flight_date", "reporting_airline", "origin", F.col("dest").alias("destination"),
        "distance", "month", "day_of_week", "scheduled_departure_hour",
        "scheduled_departure_minute", "scheduled_arrival_hour", "scheduled_arrival_minute",
        "crs_elapsed_time", TARGET_COLUMN,
    )


def add_schedule_features(eligible: DataFrame) -> DataFrame:
    """Calendar categoricals and local schedule cycles, never actual times."""
    result = eligible.withColumn("calendar_month", F.col("month").cast("string")).withColumn(
        "calendar_weekday", F.col("day_of_week").cast("string")
    )
    for name, prefix in (("departure", "scheduled_departure"), ("arrival", "scheduled_arrival")):
        minutes = F.col(f"{prefix}_hour") * 60 + F.col(f"{prefix}_minute")
        angle = minutes * (2 * math.pi / 1440)
        result = (result.withColumn(f"{name}_sin", F.sin(angle))
                  .withColumn(f"{name}_cos", F.cos(angle))
                  .withColumn(f"{name}_missing", minutes.isNull().cast("double")))
    return result.withColumn("duration_missing", F.col("crs_elapsed_time").isNull().cast("double"))


def entity_history(eligible: DataFrame, name: str, columns: tuple[str, ...]) -> DataFrame:
    """One daily aggregation and one ordered partition for all three windows."""
    daily = eligible.groupBy("flight_date", *columns).agg(
        F.count("*").alias("_count"), F.sum(TARGET_COLUMN).alias("_delayed")
    ).withColumn("history_day", F.datediff("flight_date", F.lit("1970-01-01")))
    result = daily
    outputs = []
    for days in WINDOW_DAYS:
        window = Window.partitionBy(*columns).orderBy("history_day").rangeBetween(-days, -1)
        count_name = f"{name}_flight_count_{days}d"
        rate_name = f"{name}_delay_rate_{days}d"
        missing_name = f"{name}_history_missing_{days}d"
        result = result.withColumn(count_name, F.coalesce(F.sum("_count").over(window), F.lit(0)))
        result = result.withColumn(rate_name, F.when(
            F.col(count_name) > 0, F.sum("_delayed").over(window).cast("double") / F.col(count_name)
        )).withColumn(missing_name, (F.col(count_name) == 0).cast("double"))
        outputs.extend((count_name, rate_name, missing_name))
    return result.select("flight_date", *columns, *outputs)


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


def assemble_ml_flight_features(eligible: DataFrame) -> DataFrame:
    featured = add_schedule_features(eligible)
    for name, columns in ENTITIES:
        history = entity_history(eligible, name, columns)
        # Null-safe keys also preserve histories for an unknown entity.
        left, right = featured.alias("f"), history.alias("h")
        keys = ["flight_date", *columns]
        condition = F.lit(True)
        for key in keys:
            condition = condition & F.col(f"f.{key}").eqNullSafe(F.col(f"h.{key}"))
        featured = left.join(right, condition, "left").select(
            "f.*", *[F.col(f"h.{c}").alias(c) for c in history.columns if c not in keys]
        )
    return assign_chronological_split(featured).select(*OUTPUT_COLUMNS)


def transform_ml_flight_features(dataframe: DataFrame) -> DataFrame:
    return assemble_ml_flight_features(select_eligible_flights(dataframe))


def validate_feature_frame(dataframe: DataFrame, expected_rows=None) -> int:
    if tuple(dataframe.columns) != OUTPUT_COLUMNS:
        raise AssertionError("Feature column order/schema differs from contract")
    types = dict(dataframe.dtypes)
    if types["flight_date"] != "date" or types["split"] != "string":
        raise AssertionError("Invalid date/split schema")
    if any(types[c] != "string" for c in CATEGORICAL_COLUMNS):
        raise AssertionError("Invalid categorical schema")
    if any(types[c] not in ("double", "float", "int", "bigint")
           for c in (*NUMERIC_COLUMNS, *HISTORY_COUNTS, *HISTORY_MISSING, TARGET_COLUMN)):
        raise AssertionError("Invalid numeric schema")
    invalid = F.col("flight_date").isNull() | F.col("split").isNull()
    invalid = invalid | F.col(TARGET_COLUMN).isNull() | ~F.col(TARGET_COLUMN).isin(0, 1)
    valid_split = F.lit(False)
    for name, bounds in SPLIT_BOUNDARIES.items():
        valid_split = valid_split | ((F.col("split") == name) & F.col("flight_date").between(bounds["start"], bounds["end"]))
    invalid = invalid | ~valid_split
    for name, _ in ENTITIES:
        for days in WINDOW_DAYS:
            count = F.col(f"{name}_flight_count_{days}d")
            rate = F.col(f"{name}_delay_rate_{days}d")
            missing = F.col(f"{name}_history_missing_{days}d")
            invalid = invalid | count.isNull() | (count < 0) | (count != F.floor(count))
            invalid = invalid | missing.isNull() | (missing != (count == 0).cast("double"))
            invalid = invalid | ((count == 0) & rate.isNotNull())
            invalid = invalid | ((count > 0) & (rate.isNull() | F.isnan(rate) | ~rate.between(0, 1)))
    for c in (*NUMERIC_COLUMNS, *HISTORY_COUNTS, *HISTORY_MISSING):
        invalid = invalid | F.isnan(c) | (F.abs(F.col(c)) == float("inf"))
    for c in ("departure_missing", "arrival_missing", "duration_missing"):
        invalid = invalid | F.col(c).isNull() | ~F.col(c).isin(0, 1)
    stats = dataframe.agg(F.count("*").alias("rows"), F.sum(F.when(invalid, 1).otherwise(0)).alias("invalid")).first()
    if not stats.rows or stats.invalid:
        raise AssertionError(f"Empty or invalid features: {stats}")
    if expected_rows is not None and stats.rows != expected_rows:
        raise AssertionError("Eligible-flight multiplicity changed")
    return stats.rows


def write_ml_flight_features(dataframe: DataFrame, output_directory: PathLike, rows=None) -> None:
    rows = dataframe.count() if rows is None else rows
    partitions = max(4, math.ceil(rows / 250000))
    dataframe.repartitionByRange(partitions, "split", "flight_date").write.mode("overwrite").partitionBy("split").parquet(str(output_directory))


def build_ml_flight_features(spark: SparkSession, dwd_directory: PathLike, output_directory: PathLike) -> None:
    dwd = read_dwd_flights(spark, dwd_directory)
    validate_ml_source(dwd)
    eligible = select_eligible_flights(dwd).persist(StorageLevel.DISK_ONLY)
    featured = None
    try:
        rows = eligible.count()
        featured = assemble_ml_flight_features(eligible).persist(StorageLevel.DISK_ONLY)
        validate_feature_frame(featured, rows)
        write_ml_flight_features(featured, output_directory, rows)
    finally:
        if featured is not None:
            featured.unpersist()
        eligible.unpersist()


def main() -> None:
    paths = get_project_paths(load_config())
    spark = create_spark_session("airline-build-ml-features")
    try:
        build_ml_flight_features(spark, paths.resolve(paths.dwd_flights), paths.resolve(paths.dws_ml_flight_features))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

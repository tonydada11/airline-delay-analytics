"""Shared analytical expressions for daily DWS tables."""

from typing import Sequence

from pyspark.sql import DataFrame, functions as F

from airline.common.quality import assert_not_empty, assert_required_columns


METRIC_SOURCE_COLUMNS = (
    "is_cancelled",
    "is_diverted",
    "is_arrival_delayed_15m",
    "arr_delay",
)
METRIC_COLUMNS = (
    "scheduled_flights",
    "completed_flights",
    "cancelled_flights",
    "diverted_flights",
    "eligible_delay_flights",
    "delayed_flights",
    "delay_rate",
    "avg_arrival_delay",
    "cancellation_rate",
)


def validate_daily_source(dataframe: DataFrame, grain_columns: Sequence[str]) -> None:
    """Validate trusted fields shared by all daily analytical builds."""

    required_columns = [*grain_columns, *METRIC_SOURCE_COLUMNS]
    assert_not_empty(dataframe)
    assert_required_columns(dataframe, required_columns)

    null_grain = F.lit(False)
    for column_name in grain_columns:
        null_grain = null_grain | F.col(column_name).isNull()
    if dataframe.filter(null_grain).take(1):
        raise AssertionError("DWS grain columns must not contain null values.")

    target = F.col("is_arrival_delayed_15m")
    invalid_counts = dataframe.agg(
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
            F.when(target.isNotNull() & ~target.isin(0, 1), 1).otherwise(0)
        ).alias("invalid_delay_target"),
        F.sum(
            F.when(
                (F.col("is_cancelled") == 1) & (F.col("is_diverted") == 1),
                1,
            ).otherwise(0)
        ).alias("overlapping_status"),
        F.sum(
            F.when(
                target.isNotNull()
                & (
                    (F.col("is_cancelled") != 0)
                    | (F.col("is_diverted") != 0)
                ),
                1,
            ).otherwise(0)
        ).alias("ineligible_delay_target"),
    ).first()

    failures = {
        "invalid is_cancelled values": invalid_counts["invalid_is_cancelled"],
        "invalid is_diverted values": invalid_counts["invalid_is_diverted"],
        "invalid delay-target values": invalid_counts["invalid_delay_target"],
        "rows both cancelled and diverted": invalid_counts["overlapping_status"],
        "delay targets on ineligible rows": invalid_counts["ineligible_delay_target"],
    }
    present_failures = [
        f"{label}: {count}" for label, count in failures.items() if count
    ]
    if present_failures:
        raise AssertionError("Invalid trusted DWD input: " + "; ".join(present_failures))


def aggregate_daily_flights(
    dataframe: DataFrame, grain_columns: Sequence[str]
) -> DataFrame:
    """Return lazy daily metrics at the requested grain."""

    completed = (F.col("is_cancelled") == 0) & (F.col("is_diverted") == 0)
    aggregated = dataframe.groupBy(*grain_columns).agg(
        F.count(F.lit(1)).cast("long").alias("scheduled_flights"),
        F.sum(F.when(completed, 1).otherwise(0))
        .cast("long")
        .alias("completed_flights"),
        F.sum("is_cancelled").cast("long").alias("cancelled_flights"),
        F.sum("is_diverted").cast("long").alias("diverted_flights"),
        F.count("is_arrival_delayed_15m")
        .cast("long")
        .alias("eligible_delay_flights"),
        F.coalesce(F.sum("is_arrival_delayed_15m"), F.lit(0))
        .cast("long")
        .alias("delayed_flights"),
        F.avg(F.when(completed, F.col("arr_delay"))).alias("avg_arrival_delay"),
    )
    with_rates = aggregated.withColumn(
        "delay_rate",
        F.when(
            F.col("eligible_delay_flights") > 0,
            F.col("delayed_flights").cast("double")
            / F.col("eligible_delay_flights").cast("double"),
        ).otherwise(F.lit(None).cast("double")),
    ).withColumn(
        "cancellation_rate",
        F.when(
            F.col("scheduled_flights") > 0,
            F.col("cancelled_flights").cast("double")
            / F.col("scheduled_flights").cast("double"),
        ).otherwise(F.lit(None).cast("double")),
    )
    return with_rates.select(*grain_columns, *METRIC_COLUMNS)

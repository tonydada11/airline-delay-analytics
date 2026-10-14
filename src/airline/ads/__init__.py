"""Shared aggregation rules for presentation-ready ADS tables."""

from pathlib import Path
from typing import Sequence, Union

from pyspark.sql import DataFrame, SparkSession, functions as F

from airline.common.quality import assert_not_empty, assert_required_columns
from airline.dws.daily_metrics import METRIC_COLUMNS


PathLike = Union[str, Path]
COUNT_METRIC_COLUMNS = METRIC_COLUMNS[:6]
ADS_METRIC_COLUMNS = METRIC_COLUMNS
DWD_AVERAGE_SOURCE_COLUMNS = (
    "is_cancelled",
    "is_diverted",
    "arr_delay",
)


def read_parquet(spark: SparkSession, directory: PathLike) -> DataFrame:
    """Read a trusted warehouse Parquet dataset."""

    return spark.read.parquet(str(directory))


def validate_daily_ads_source(
    dataframe: DataFrame, grain_columns: Sequence[str]
) -> None:
    """Validate the daily DWS contract consumed by an ADS build."""

    assert_not_empty(dataframe)
    assert_required_columns(dataframe, [*grain_columns, *ADS_METRIC_COLUMNS])

    null_grain = F.lit(False)
    for column_name in grain_columns:
        null_grain = null_grain | F.col(column_name).isNull()
    if dataframe.filter(null_grain).take(1):
        raise AssertionError("ADS daily-source grain columns must not be null.")

    duplicate = (
        dataframe.groupBy(*grain_columns)
        .count()
        .filter(F.col("count") > 1)
        .take(1)
    )
    if duplicate:
        raise AssertionError("ADS daily-source grain must be unique.")


def validate_dwd_ads_source(
    dataframe: DataFrame, grain_columns: Sequence[str]
) -> None:
    """Validate DWD fields required for the M4-equivalent weighted average."""

    assert_not_empty(dataframe)
    assert_required_columns(
        dataframe, [*grain_columns, *DWD_AVERAGE_SOURCE_COLUMNS]
    )

    null_grain = F.lit(False)
    for column_name in grain_columns:
        null_grain = null_grain | F.col(column_name).isNull()
    if dataframe.filter(null_grain).take(1):
        raise AssertionError("ADS DWD-source grain columns must not be null.")


def aggregate_ads_metrics(
    daily_metrics: DataFrame,
    dwd_flights: DataFrame,
    grain_columns: Sequence[str],
) -> DataFrame:
    """Roll daily counts up and attach the exact M4 arrival-delay average."""

    counts = daily_metrics.groupBy(*grain_columns).agg(
        *[
            F.sum(column_name).cast("long").alias(column_name)
            for column_name in COUNT_METRIC_COLUMNS
        ]
    )
    completed = (F.col("is_cancelled") == 0) & (F.col("is_diverted") == 0)
    averages = dwd_flights.groupBy(*grain_columns).agg(
        F.avg(F.when(completed, F.col("arr_delay"))).alias(
            "avg_arrival_delay"
        )
    )
    aggregated = counts.join(averages, list(grain_columns), "left")
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
    return with_rates.select(*grain_columns, *ADS_METRIC_COLUMNS)


def write_ads_table(dataframe: DataFrame, output_directory: PathLike) -> None:
    """Replace a full unpartitioned ADS Parquet path."""

    dataframe.write.mode("overwrite").parquet(str(output_directory))

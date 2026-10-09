"""Versioned development features, separate from the locked M5 contract."""

import math

from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, Window, functions as F

from airline.dws.build_ml_flight_features import (
    HISTORICAL_FEATURE_COLUMNS, TARGET_COLUMN, add_historical_features,
    assign_chronological_split, eligibility_condition, select_eligible_flights,
)
from airline.ml.train_models import build_preprocessing_pipeline


BUNDLES = ("baseline", "schedule", "history")
ENTITIES = (
    ("carrier", ("reporting_airline",)),
    ("origin", ("origin",)),
    ("destination", ("destination",)),
    ("route", ("origin", "destination")),
)
WINDOW_DAYS = (7, 30, 90)
SCHEDULE_NUMERIC = ("distance", "crs_elapsed_time", "departure_sin", "departure_cos",
                    "arrival_sin", "arrival_cos", "departure_missing", "arrival_missing",
                    "duration_missing")
SCHEDULE_CATEGORICAL = ("reporting_airline", "origin", "destination", "calendar_month", "calendar_weekday")
HISTORY_RATES = tuple(f"{name}_delay_rate_{days}d" for name, _ in ENTITIES for days in WINDOW_DAYS)
HISTORY_COUNTS = tuple(f"{name}_flight_count_{days}d" for name, _ in ENTITIES for days in WINDOW_DAYS)
HISTORY_MISSING = tuple(f"{name}_history_missing_{days}d" for name, _ in ENTITIES for days in WINDOW_DAYS)


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
    ).withColumn("_ordinal", F.datediff("flight_date", F.lit("1970-01-01")))
    result = daily
    outputs = []
    for days in WINDOW_DAYS:
        window = Window.partitionBy(*columns).orderBy("_ordinal").rangeBetween(-days, -1)
        count_name = f"{name}_flight_count_{days}d"
        rate_name = f"{name}_delay_rate_{days}d"
        missing_name = f"{name}_history_missing_{days}d"
        result = result.withColumn(count_name, F.coalesce(F.sum("_count").over(window), F.lit(0)))
        result = result.withColumn(rate_name, F.when(
            F.col(count_name) > 0, F.sum("_delayed").over(window).cast("double") / F.col(count_name)
        )).withColumn(missing_name, (F.col(count_name) == 0).cast("double"))
        outputs.extend((count_name, rate_name, missing_name))
    return result.select("flight_date", *columns, *outputs)


def build_features(dwd: DataFrame, bundle: str) -> DataFrame:
    if bundle not in BUNDLES:
        raise ValueError(f"Unknown feature bundle: {bundle}")
    if bundle == "baseline":
        # Delegate unchanged feature definitions to the original implementation.
        return assign_chronological_split(add_historical_features(select_eligible_flights(dwd)))
    eligible = dwd.filter(eligibility_condition()).select(
        "flight_date", "reporting_airline", "origin", F.col("dest").alias("destination"),
        "distance", "month", "day_of_week", "scheduled_departure_hour",
        "scheduled_departure_minute", "scheduled_arrival_hour", "scheduled_arrival_minute",
        "crs_elapsed_time", TARGET_COLUMN,
    )
    featured = add_schedule_features(eligible)
    if bundle == "history":
        for name, columns in ENTITIES:
            featured = featured.join(entity_history(eligible, name, columns), ["flight_date", *columns], "left")
    else:
        featured = add_historical_features(featured)
    rates = HISTORY_RATES if bundle == "history" else HISTORICAL_FEATURE_COLUMNS
    extra = (*HISTORY_COUNTS, *HISTORY_MISSING) if bundle == "history" else ()
    return assign_chronological_split(featured).select(
        "flight_date", *SCHEDULE_CATEGORICAL, *SCHEDULE_NUMERIC, *rates, *extra, TARGET_COLUMN, "split"
    )


def preprocessing_pipeline(bundle: str) -> Pipeline:
    if bundle == "baseline":
        return build_preprocessing_pipeline()
    if bundle not in BUNDLES:
        raise ValueError(f"Unknown feature bundle: {bundle}")
    rates = HISTORY_RATES if bundle == "history" else HISTORICAL_FEATURE_COLUMNS
    numeric = (*SCHEDULE_NUMERIC, *rates)
    extra = (*HISTORY_COUNTS, *HISTORY_MISSING) if bundle == "history" else ()
    indexers = [StringIndexer(inputCol=c, outputCol=f"_{c}_index", handleInvalid="keep",
                              stringOrderType="alphabetAsc") for c in SCHEDULE_CATEGORICAL]
    encoder = OneHotEncoder(inputCols=[f"_{c}_index" for c in SCHEDULE_CATEGORICAL],
                            outputCols=[f"_{c}_ohe" for c in SCHEDULE_CATEGORICAL], handleInvalid="keep")
    imputer = Imputer(inputCols=list(numeric), outputCols=[f"_{c}_imputed" for c in numeric], strategy="mean")
    assembler = VectorAssembler(inputCols=[*[f"_{c}_ohe" for c in SCHEDULE_CATEGORICAL],
                                           *[f"_{c}_imputed" for c in numeric], *extra], outputCol="features")
    return Pipeline(stages=[*indexers, encoder, imputer, assembler])

"""Flight-time causal histories; selection and final-test datasets are separate."""
import argparse
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
from airline.common.lineage import BUSINESS_KEY, INTEGER_KEYS, with_lineage, key_json_column
from airline.common.raw_snapshot import verify_snapshot
from airline.dws.dataset_manifest import MANIFEST_NAME, publish_dataset_manifest
from airline.dws.flight_time import with_event_times, DAY_US, time_provenance
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
        "is_cancelled", "dep_delay", "actual_elapsed_time",
        "is_diverted",
        TARGET_COLUMN,
    }
    assert_required_columns(dataframe, required_columns | set(BUSINESS_KEY))
    key_json_column(dataframe)
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
    return with_event_times(with_lineage(dataframe.filter(eligibility_condition()))).select(
        "flight_id", *INTEGER_KEYS,
        "flight_date", "reporting_airline", "origin", F.col("dest").alias("destination"),
        "distance", "month", "day_of_week", "scheduled_departure_hour",
        "scheduled_departure_minute", "scheduled_arrival_hour", "scheduled_arrival_minute",
        "crs_elapsed_time", TARGET_COLUMN, "_prediction_us", "_available_us",
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
    """Sweep contribution intervals; integer microseconds encode strict boundaries.

    A source is active for max(departure, arrival) < target <= departure+N days.
    Collapse simultaneous events before cumulative sums; duplicates contribute independently.
    """
    query = eligible.select(*columns, F.col('_prediction_us').alias('_time')).distinct()
    source = eligible.filter(F.col('_prediction_us').isNotNull() & F.col('_available_us').isNotNull())
    start = F.greatest('_prediction_us', '_available_us') + 1
    deltas = [f'_{kind}{days}' for days in WINDOW_DAYS for kind in ('count', 'delayed')]
    events = []
    for days in WINDOW_DAYS:
        end = F.col('_prediction_us') + days * DAY_US + 1
        valid = start < end
        for point, sign in ((start, 1), (end, -1)):
            values = []
            for window in WINDOW_DAYS:
                count = F.lit(sign if days == window else 0).cast('long')
                values.extend([count.alias(f'_count{window}'),
                               (count * F.col(TARGET_COLUMN)).alias(f'_delayed{window}')])
            events.append(source.filter(valid).select(*columns, point.alias('_time'),
                          F.lit(0).alias('_query'), *values))
    timeline = query.select('*', F.lit(1).alias('_query'),
                           *[F.lit(0).cast('long').alias(c) for c in deltas])
    for event in events:
        timeline = timeline.unionByName(event)
    timeline = timeline.groupBy(*columns, '_time').agg(F.max('_query').alias('_query'),
                                                     *[F.sum(c).alias(c) for c in deltas])
    window = Window.partitionBy(*columns).orderBy('_time').rowsBetween(Window.unboundedPreceding, 0)
    for c in deltas:
        timeline = timeline.withColumn(c, F.sum(c).over(window))
    outputs = []
    for days in WINDOW_DAYS:
        count = F.col(f'_count{days}')
        outputs.extend([count.alias(f'{name}_flight_count_{days}d'),
                        F.when(count > 0, F.col(f'_delayed{days}').cast('double') / count).alias(f'{name}_delay_rate_{days}d'),
                        (count == 0).cast('double').alias(f'{name}_history_missing_{days}d')])
    return timeline.filter('_query = 1').select(F.col('_time').alias('_prediction_us'), *columns, *outputs)


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


def assemble_ml_flight_features(eligible: DataFrame, staged=None) -> DataFrame:
    featured = add_schedule_features(eligible)
    for name, columns in ENTITIES:
        history = entity_history(eligible, name, columns)
        # Null-safe keys also preserve histories for an unknown entity.
        left, right = featured.alias("f"), history.alias("h")
        keys = ["_prediction_us", *columns]
        condition = F.lit(True)
        for key in keys:
            condition = condition & F.col(f"f.{key}").eqNullSafe(F.col(f"h.{key}"))
        featured = left.join(right, condition, "left").select(
            "f.*", *[F.col(f"h.{c}").alias(c) for c in history.columns if c not in keys]
        )
        if staged is not None:
            # Materialize one entity at a time on the bounded runtime, so the four
            # event sweeps do not compete for executor memory. Track before action
            # so even an interrupted count is released by the caller's finally.
            featured = featured.persist(StorageLevel.DISK_ONLY)
            previous = list(staged)
            staged.append(featured)
            featured.count()
            for frame in previous:
                frame.unpersist()
            staged[:] = [featured]
            print(f'Completed {name} timestamp histories', flush=True)
    return assign_chronological_split(featured).select(*OUTPUT_COLUMNS)


def transform_ml_flight_features(dataframe: DataFrame, phase='selection') -> DataFrame:
    if phase not in ('selection', 'test'):
        raise ValueError('Feature phase must be selection|test')
    source = dataframe.filter(F.col('flight_date') <= F.lit(VALIDATION_END)) if phase == 'selection' else dataframe
    result = assemble_ml_flight_features(select_eligible_flights(source))
    return result.filter(F.col('split') == 'test') if phase == 'test' else result


def validate_feature_frame(dataframe: DataFrame, expected_rows=None) -> int:
    if tuple(dataframe.columns) != OUTPUT_COLUMNS:
        raise AssertionError("Feature column order/schema differs from contract")
    types = dict(dataframe.dtypes)
    if types["flight_id"] != "string":
        raise AssertionError("Invalid flight_id schema")
    if types["flight_date"] != "date" or types["split"] != "string":
        raise AssertionError("Invalid date/split schema")
    if any(types[c] != "string" for c in CATEGORICAL_COLUMNS):
        raise AssertionError("Invalid categorical schema")
    if any(types[c] not in ("double", "float", "int", "bigint")
           for c in (*NUMERIC_COLUMNS, *HISTORY_COUNTS, *HISTORY_MISSING, TARGET_COLUMN)):
        raise AssertionError("Invalid numeric schema")
    invalid = F.col("flight_id").isNull() | ~F.col("flight_id").eqNullSafe(F.sha2(key_json_column(dataframe), 256))
    invalid = invalid | F.col("flight_date").isNull() | F.col("split").isNull()
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
    (Path(output_directory) / MANIFEST_NAME).unlink(missing_ok=True)
    rows = dataframe.count() if rows is None else rows
    partitions = max(4, math.ceil(rows / 250000))
    dataframe.repartitionByRange(partitions, "split", "flight_date").write.mode("overwrite").partitionBy("split").parquet(str(output_directory))


def build_ml_flight_features(spark: SparkSession, dwd_directory: PathLike, output_directory: PathLike, snapshot=None, phase='selection', selection_manifest=None) -> None:
    if snapshot is None:
        paths = get_project_paths(load_config())
        snapshot = verify_snapshot(paths.resolve(paths.raw_csv), paths.resolve(paths.raw_manifest))
    if phase not in ('selection', 'test'):
        raise ValueError('Feature phase must be selection|test')
    if phase == 'test':
        from airline.ml.model_contract import validate_manifest
        validate_manifest(selection_manifest, snapshot['raw_manifest_sha256'])
    dwd = read_dwd_flights(spark, dwd_directory)
    if phase == 'selection':
        dwd = dwd.filter(F.col('flight_date') <= F.lit(VALIDATION_END))
    validate_ml_source(dwd)
    eligible = select_eligible_flights(dwd).persist(StorageLevel.DISK_ONLY)
    featured = None
    staged = []
    try:
        eligible.count()
        audit = eligible.agg(F.sum(F.col('_prediction_us').isNull().cast('long')).alias('uncertain_prediction_rows'),
                             F.sum(F.col('_available_us').isNull().cast('long')).alias('unavailable_source_rows')).first().asDict()
        target = eligible.filter(F.col('flight_date') >= F.lit(TEST_START)) if phase == 'test' else eligible
        rows = target.count()
        featured = assemble_ml_flight_features(eligible, staged)
        if phase == 'test':
            featured = featured.filter(F.col('split') == 'test')
        featured = featured.persist(StorageLevel.DISK_ONLY)
        validate_feature_frame(featured, rows)
        write_ml_flight_features(featured, output_directory, rows)
        reloaded = spark.read.parquet(str(output_directory))
        publish_dataset_manifest(output_directory, reloaded, snapshot, phase=phase,
                                 time_audit=audit, selection_manifest=selection_manifest, expected_rows=rows)
    finally:
        if featured is not None:
            featured.unpersist()
        for frame in staged:
            frame.unpersist()
        eligible.unpersist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('selection', 'test'), default='selection')
    args = parser.parse_args()
    paths = get_project_paths(load_config())
    manifest = None
    spark = create_spark_session("airline-build-ml-features")
    try:
        if args.phase == 'test':
            from airline.ml.model_contract import load_bundle
            _, _, manifest = load_bundle(paths.resolve(paths.artifact_root) / 'models')
        output = paths.dws_ml_test_features if args.phase == 'test' else paths.dws_ml_flight_features
        build_ml_flight_features(spark, paths.resolve(paths.dwd_flights), paths.resolve(output),
                                phase=args.phase, selection_manifest=manifest)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

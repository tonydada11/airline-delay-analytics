# AI-generated test code: generated with OpenAI Codex.
import os
import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

import pytest
from pyspark.sql import Column, DataFrame, SparkSession, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.quality import assert_required_columns
from airline.common.spark_session import create_spark_session
from airline.dws.build_ml_flight_features import (
    BASE_FEATURE_COLUMNS,
    HISTORICAL_FEATURE_COLUMNS,
    LEAKAGE_BLACKLIST,
    OUTPUT_COLUMNS,
    PREDICTOR_COLUMNS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    build_ml_flight_features,
)


EXPECTED_FEATURE_COUNT = 13_708_650
EXPECTED_SPLIT_COUNTS = {
    "train": 8_946_370,
    "validation": 2_418_739,
    "test": 2_343_541,
}

pytestmark = pytest.mark.skipif(
    os.environ.get("AIRLINE_RUN_FULL_DATA_QUALITY") != "1",
    reason="set AIRLINE_RUN_FULL_DATA_QUALITY=1 for mandatory M5 acceptance checks",
)


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    session = create_spark_session("airline-m5-full-data-quality")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def snapshot_directory() -> Iterator[Path]:
    shared_root = Path(os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir()))
    shared_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="airline-m5-", dir=shared_root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _eligibility_condition() -> Column:
    return (
        (F.col("is_cancelled") == 0)
        & (F.col("is_diverted") == 0)
        & F.col("is_arrival_delayed_15m").isNotNull()
    )


def _split_target_distribution(dataframe: DataFrame) -> Dict[tuple[str, int], int]:
    return {
        (row["split"], row["is_arrival_delayed_15m"]): row["count"]
        for row in dataframe.groupBy("split", "is_arrival_delayed_15m")
        .count()
        .collect()
    }


def _split_metrics(dataframe: DataFrame) -> Dict[str, Dict[str, float | int]]:
    rows = dataframe.groupBy("split").agg(
        F.count("*").alias("rows"),
        F.sum("is_arrival_delayed_15m").alias("delayed"),
    ).collect()
    return {
        row["split"]: {
            "rows": row["rows"],
            "delayed": row["delayed"],
            "prevalence": row["delayed"] / row["rows"],
        }
        for row in rows
    }


def _same_entity_date_rate_mismatches(dataframe: DataFrame) -> Dict[str, int]:
    specs: Sequence[tuple[str, Sequence[str]]] = (
        ("carrier_delay_rate_30d", ("reporting_airline",)),
        ("origin_delay_rate_30d", ("origin",)),
        ("destination_delay_rate_30d", ("destination",)),
        ("route_delay_rate_30d", ("origin", "destination")),
    )
    mismatches: Dict[str, int] = {}
    for feature_name, entity_columns in specs:
        inconsistent = (
            dataframe.groupBy("flight_date", *entity_columns)
            .agg(
                F.min(feature_name).alias("minimum_rate"),
                F.max(feature_name).alias("maximum_rate"),
            )
            .filter(~F.col("minimum_rate").eqNullSafe(F.col("maximum_rate")))
            .count()
        )
        mismatches[feature_name] = inconsistent
    return mismatches


def _independent_sample_history_mismatches(
    dwd: DataFrame, features: DataFrame
) -> int:
    samples = (
        features.filter(F.col("flight_date") >= F.lit(TEST_START))
        .select(
            "flight_date",
            "reporting_airline",
            "origin",
            "destination",
            *HISTORICAL_FEATURE_COLUMNS,
        )
        .distinct()
        .orderBy("flight_date", "reporting_airline", "origin", "destination")
        .limit(8)
        .collect()
    )
    specs: Sequence[tuple[str, Sequence[tuple[str, str]]]] = (
        ("carrier_delay_rate_30d", (("reporting_airline", "reporting_airline"),)),
        ("origin_delay_rate_30d", (("origin", "origin"),)),
        ("destination_delay_rate_30d", (("dest", "destination"),)),
        (
            "route_delay_rate_30d",
            (("origin", "origin"), ("dest", "destination")),
        ),
    )
    expressions = []
    for sample_index, sample in enumerate(samples):
        for feature_name, entity_pairs in specs:
            start_date = sample["flight_date"] - timedelta(days=30)
            condition = (
                _eligibility_condition()
                & (F.col("flight_date") >= F.lit(start_date))
                & (F.col("flight_date") < F.lit(sample["flight_date"]))
            )
            for source_column, sample_column in entity_pairs:
                condition = condition & (
                    F.col(source_column) == F.lit(sample[sample_column])
                )
            expressions.extend(
                [
                    F.sum(
                        F.when(
                            condition, F.col("is_arrival_delayed_15m")
                        ).otherwise(0)
                    ).alias(f"numerator_{sample_index}_{feature_name}"),
                    F.sum(F.when(condition, 1).otherwise(0)).alias(
                        f"denominator_{sample_index}_{feature_name}"
                    ),
                ]
            )

    aggregates = dwd.agg(*expressions).first()
    mismatches = 0
    for sample_index, sample in enumerate(samples):
        for feature_name, _ in specs:
            numerator = aggregates[f"numerator_{sample_index}_{feature_name}"]
            denominator = aggregates[f"denominator_{sample_index}_{feature_name}"]
            expected: Optional[float] = (
                numerator / denominator if denominator else None
            )
            actual = sample[feature_name]
            if expected is None:
                mismatches += int(actual is not None)
            else:
                mismatches += int(actual is None or abs(actual - expected) > 1e-12)
    return mismatches


def _bidirectional_multiset_differences(
    first: DataFrame, second: DataFrame
) -> tuple[int, int]:
    first_aligned = first.select(*OUTPUT_COLUMNS)
    second_aligned = second.select(*OUTPUT_COLUMNS)
    assert first_aligned.schema == second_aligned.schema
    first_minus_second = first_aligned.exceptAll(second_aligned).limit(1).count()
    second_minus_first = second_aligned.exceptAll(first_aligned).limit(1).count()
    return first_minus_second, second_minus_first


def test_full_ml_feature_quality_leakage_and_second_build_idempotency(
    spark: SparkSession,
    snapshot_directory: Path,
) -> None:
    paths = get_project_paths(load_config())
    dwd_path = paths.resolve(paths.dwd_flights)
    output_path = paths.resolve(paths.dws_ml_flight_features)
    first_snapshot_path = snapshot_directory / "first_build"

    dwd = spark.read.parquet(str(dwd_path))
    features = spark.read.parquet(str(output_path))
    assert_required_columns(features, OUTPUT_COLUMNS)
    assert tuple(features.columns) == OUTPUT_COLUMNS
    assert LEAKAGE_BLACKLIST.isdisjoint(PREDICTOR_COLUMNS)
    assert LEAKAGE_BLACKLIST.isdisjoint(features.columns)
    assert "flight_date" not in PREDICTOR_COLUMNS
    assert "is_arrival_delayed_15m" not in PREDICTOR_COLUMNS
    assert "split" not in PREDICTOR_COLUMNS
    assert "flight_key" not in features.columns
    assert "destination" in features.columns
    assert "dest" not in features.columns

    eligible_dwd = dwd.filter(_eligibility_condition())
    eligible_count = eligible_dwd.count()
    feature_count = features.count()
    invalid_targets = features.filter(
        F.col("is_arrival_delayed_15m").isNull()
        | ~F.col("is_arrival_delayed_15m").isin(0, 1)
    ).count()
    dwd_target_counts = {
        row["is_arrival_delayed_15m"]: row["count"]
        for row in eligible_dwd.groupBy("is_arrival_delayed_15m").count().collect()
    }
    feature_target_counts = {
        row["is_arrival_delayed_15m"]: row["count"]
        for row in features.groupBy("is_arrival_delayed_15m").count().collect()
    }

    expected_split = (
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
    split_mismatches = features.filter(
        ~F.col("split").eqNullSafe(expected_split)
    ).count()
    split_metrics = _split_metrics(features)

    base_feature_nulls = features.agg(
        *[
            F.sum(F.col(column).isNull().cast("long")).alias(column)
            for column in BASE_FEATURE_COLUMNS
        ]
    ).first().asDict()
    historical_quality = features.agg(
        *[
            F.sum(
                F.when(
                    F.col(column).isNotNull()
                    & ~F.col(column).between(0.0, 1.0),
                    1,
                ).otherwise(0)
            ).alias(f"out_of_bounds_{column}")
            for column in HISTORICAL_FEATURE_COLUMNS
        ]
    ).first().asDict()
    same_day_rate_mismatches = _same_entity_date_rate_mismatches(features)
    sample_history_mismatches = _independent_sample_history_mismatches(
        dwd, features
    )

    assert eligible_count == EXPECTED_FEATURE_COUNT
    assert feature_count == EXPECTED_FEATURE_COUNT
    assert invalid_targets == 0
    assert dwd_target_counts == feature_target_counts
    assert split_mismatches == 0
    assert {name: metrics["rows"] for name, metrics in split_metrics.items()} == (
        EXPECTED_SPLIT_COUNTS
    )
    assert all(count == 0 for count in base_feature_nulls.values())
    assert all(count == 0 for count in historical_quality.values())
    assert all(count == 0 for count in same_day_rate_mismatches.values())
    assert sample_history_mismatches == 0
    assert not any(path.is_dir() for path in output_path.iterdir())

    first_distribution = _split_target_distribution(features)
    features.select(*OUTPUT_COLUMNS).write.mode("overwrite").parquet(
        str(first_snapshot_path)
    )

    build_ml_flight_features(spark, dwd_path, output_path)
    spark.catalog.clearCache()
    first_build = spark.read.parquet(str(first_snapshot_path))
    second_build = spark.read.parquet(str(output_path))
    second_count = second_build.count()
    second_distribution = _split_target_distribution(second_build)
    first_minus_second, second_minus_first = _bidirectional_multiset_differences(
        first_build, second_build
    )

    overall_delayed = feature_target_counts[1]
    overall_prevalence = overall_delayed / feature_count
    print(f"eligible_dwd_count={eligible_count}")
    print(f"feature_count={feature_count}")
    print(f"feature_target_counts={feature_target_counts}")
    print(f"overall_target_prevalence={overall_prevalence}")
    print(f"split_metrics={split_metrics}")
    print(f"base_feature_nulls={base_feature_nulls}")
    print(f"historical_feature_quality={historical_quality}")
    print(f"same_entity_date_rate_mismatches={same_day_rate_mismatches}")
    print(f"independent_sample_history_mismatches={sample_history_mismatches}")
    print(f"first_build_count={feature_count}")
    print(f"second_build_count={second_count}")
    print(f"first_build_split_target_distribution={first_distribution}")
    print(f"second_build_split_target_distribution={second_distribution}")
    print(f"first_minus_second_rows={first_minus_second}")
    print(f"second_minus_first_rows={second_minus_first}")

    assert second_count == feature_count
    assert second_distribution == first_distribution
    assert first_minus_second == 0
    assert second_minus_first == 0

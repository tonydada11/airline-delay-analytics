# AI-generated test code: generated with OpenAI Codex.
"""Tests ads quality behavior."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterator, Sequence

import pytest
from pyspark.sql import Column, DataFrame, SparkSession, functions as F

from airline.ads import ADS_METRIC_COLUMNS, COUNT_METRIC_COLUMNS
from airline.ads.build_airport_scorecard import build_airport_scorecard
from airline.ads.build_carrier_scorecard import build_carrier_scorecard
from airline.ads.build_monthly_operations import build_monthly_operations
from airline.ads.build_route_ranking import build_route_ranking
from airline.common.config import load_config
from airline.common.paths import ProjectPaths, get_project_paths
from airline.common.quality import assert_required_columns
from airline.common.spark_session import create_spark_session


GRAINS = {
    "monthly": ["year", "month"],
    "carrier": ["reporting_airline"],
    "airport": ["origin"],
    "route": ["origin", "destination"],
}

pytestmark = pytest.mark.skipif(
    os.environ.get("AIRLINE_RUN_FULL_DATA_QUALITY") != "1",
    reason="set AIRLINE_RUN_FULL_DATA_QUALITY=1 for mandatory M6 acceptance checks",
)


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    """Create the test Spark session and stop it after use."""
    session = create_spark_session("airline-m6-full-data-quality")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def snapshot_directory() -> Iterator[Path]:
    """Create a valid dataset snapshot in the test workspace."""
    shared_root = Path(os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir()))
    shared_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="airline-m6-", dir=shared_root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _output_paths(paths: ProjectPaths) -> Dict[str, Path]:
    """Return the expected output paths for the test."""
    return {
        "monthly": paths.resolve(paths.ads_monthly_operations),
        "carrier": paths.resolve(paths.ads_carrier_scorecard),
        "airport": paths.resolve(paths.ads_airport_scorecard),
        "route": paths.resolve(paths.ads_route_ranking),
    }


def _load_ads(
    spark: SparkSession, output_paths: Dict[str, Path]
) -> Dict[str, DataFrame]:
    """Load ads."""
    return {
        name: spark.read.parquet(str(output_path))
        for name, output_path in output_paths.items()
    }


def _build_all_ads(
    spark: SparkSession, paths: ProjectPaths, outputs: Dict[str, Path]
) -> None:
    """Build all ads."""
    dwd_path = paths.resolve(paths.dwd_flights)
    build_monthly_operations(
        spark,
        dwd_path,
        paths.resolve(paths.dws_carrier_daily),
        outputs["monthly"],
    )
    build_carrier_scorecard(
        spark,
        dwd_path,
        paths.resolve(paths.dws_carrier_daily),
        outputs["carrier"],
    )
    build_airport_scorecard(
        spark,
        dwd_path,
        paths.resolve(paths.dws_origin_airport_daily),
        outputs["airport"],
    )
    build_route_ranking(
        spark,
        dwd_path,
        paths.resolve(paths.dws_route_daily),
        outputs["route"],
    )


def _source_views(
    dwd: DataFrame,
    carrier_daily: DataFrame,
    airport_daily: DataFrame,
    route_daily: DataFrame,
) -> tuple[Dict[str, DataFrame], Dict[str, DataFrame]]:
    """Create temporary views for the source tables."""
    dws_sources = {
        "monthly": carrier_daily.withColumn(
            "year", F.year("flight_date").cast("int")
        ).withColumn("month", F.month("flight_date").cast("int")),
        "carrier": carrier_daily,
        "airport": airport_daily,
        "route": route_daily,
    }
    dwd_sources = {
        "monthly": dwd,
        "carrier": dwd,
        "airport": dwd,
        "route": dwd.withColumn("destination", F.col("dest")).drop("dest"),
    }
    return dws_sources, dwd_sources


def _aggregate_dws_counts(dataframe: DataFrame, grain: Sequence[str]) -> DataFrame:
    """Aggregate the expected DWS row counts."""
    return dataframe.groupBy(*grain).agg(
        *[
            F.sum(column_name).cast("long").alias(column_name)
            for column_name in COUNT_METRIC_COLUMNS
        ]
    )


def _aggregate_dwd_metrics(dataframe: DataFrame, grain: Sequence[str]) -> DataFrame:
    """Aggregate source metrics for reconciliation."""
    completed = (F.col("is_cancelled") == 0) & (F.col("is_diverted") == 0)
    counts_and_average = dataframe.groupBy(*grain).agg(
        F.count("*").cast("long").alias("scheduled_flights"),
        F.sum(completed.cast("long")).cast("long").alias("completed_flights"),
        F.sum("is_cancelled").cast("long").alias("cancelled_flights"),
        F.sum("is_diverted").cast("long").alias("diverted_flights"),
        F.count("is_arrival_delayed_15m")
        .cast("long")
        .alias("eligible_delay_flights"),
        F.coalesce(F.sum("is_arrival_delayed_15m"), F.lit(0))
        .cast("long")
        .alias("delayed_flights"),
        F.avg(F.when(completed, F.col("arr_delay"))).alias(
            "avg_arrival_delay"
        ),
    )
    return counts_and_average.withColumn(
        "delay_rate",
        F.when(
            F.col("eligible_delay_flights") > 0,
            F.col("delayed_flights") / F.col("eligible_delay_flights"),
        ).otherwise(F.lit(None).cast("double")),
    ).withColumn(
        "cancellation_rate",
        F.col("cancelled_flights") / F.col("scheduled_flights"),
    ).select(*grain, *ADS_METRIC_COLUMNS)


def _quality_violations(dataframe: DataFrame, grain: Sequence[str]) -> Dict[str, int]:
    """Collect the quality checks that fail."""
    null_grain: Column = F.lit(False)
    for column_name in grain:
        null_grain = null_grain | F.col(column_name).isNull()
    duplicate_grains = (
        dataframe.groupBy(*grain).count().filter(F.col("count") > 1).count()
    )
    values = dataframe.agg(
        F.sum(F.when(null_grain, 1).otherwise(0)).alias("null_grain"),
        F.sum(
            F.when(
                F.col("scheduled_flights")
                != (
                    F.col("completed_flights")
                    + F.col("cancelled_flights")
                    + F.col("diverted_flights")
                ),
                1,
            ).otherwise(0)
        ).alias("component_count_mismatch"),
        F.sum(
            F.when(
                (F.col("delayed_flights") > F.col("eligible_delay_flights"))
                | (F.col("eligible_delay_flights") > F.col("completed_flights")),
                1,
            ).otherwise(0)
        ).alias("count_bound_violation"),
        F.sum(
            F.when(
                (
                    F.col("delay_rate").isNotNull()
                    & ~F.col("delay_rate").between(0.0, 1.0)
                )
                | ~F.col("cancellation_rate").between(0.0, 1.0),
                1,
            ).otherwise(0)
        ).alias("rate_bound_violation"),
        F.sum(
            F.when(
                (
                    (F.col("eligible_delay_flights") == 0)
                    & F.col("delay_rate").isNotNull()
                )
                | (
                    (F.col("eligible_delay_flights") > 0)
                    & F.col("delay_rate").isNull()
                ),
                1,
            ).otherwise(0)
        ).alias("delay_rate_null_mismatch"),
        F.sum(
            F.when(
                (F.col("eligible_delay_flights") > 0)
                & (
                    F.abs(
                        F.col("delay_rate")
                        - F.col("delayed_flights")
                        / F.col("eligible_delay_flights")
                    )
                    > 1e-12
                ),
                1,
            ).otherwise(0)
        ).alias("delay_rate_formula_mismatch"),
        F.sum(
            F.when(
                F.abs(
                    F.col("cancellation_rate")
                    - F.col("cancelled_flights") / F.col("scheduled_flights")
                )
                > 1e-12,
                1,
            ).otherwise(0)
        ).alias("cancellation_rate_formula_mismatch"),
    ).first()
    return {"duplicate_grains": duplicate_grains, **values.asDict()}


def _reconciliation_mismatches(
    actual: DataFrame,
    expected_dws_counts: DataFrame,
    expected_dwd: DataFrame,
    grain: Sequence[str],
) -> Dict[str, int]:
    """Find row-count and metric differences."""
    actual_counts = actual.select(*grain, *COUNT_METRIC_COLUMNS)
    dws_count_mismatch = (
        actual_counts.exceptAll(expected_dws_counts).limit(1).count()
        + expected_dws_counts.exceptAll(actual_counts).limit(1).count()
    )
    joined = actual.alias("actual").join(
        expected_dwd.alias("expected"), list(grain), "full"
    )
    missing_or_count_mismatch: Column = F.col("actual.scheduled_flights").isNull()
    missing_or_count_mismatch = (
        missing_or_count_mismatch | F.col("expected.scheduled_flights").isNull()
    )
    for column_name in COUNT_METRIC_COLUMNS:
        missing_or_count_mismatch = missing_or_count_mismatch | ~F.col(
            f"actual.{column_name}"
        ).eqNullSafe(F.col(f"expected.{column_name}"))
    dwd_count_mismatch = joined.filter(missing_or_count_mismatch).count()
    average_mismatch = joined.filter(
        ~F.col("actual.avg_arrival_delay").eqNullSafe(
            F.col("expected.avg_arrival_delay")
        )
        & (
            F.col("actual.avg_arrival_delay").isNull()
            | F.col("expected.avg_arrival_delay").isNull()
            | (
                F.abs(
                    F.col("actual.avg_arrival_delay")
                    - F.col("expected.avg_arrival_delay")
                )
                > 1e-12
            )
        )
    ).count()
    return {
        "dws_count_mismatch": dws_count_mismatch,
        "dwd_count_mismatch": dwd_count_mismatch,
        "dwd_average_mismatch": average_mismatch,
    }


def _multiset_differences(
    first: DataFrame, second: DataFrame, columns: Sequence[str]
) -> tuple[int, int]:
    """Compare the expected and actual row multisets."""
    first_aligned = first.select(*columns)
    second_aligned = second.select(*columns)
    assert first_aligned.schema == second_aligned.schema
    return (
        first_aligned.exceptAll(second_aligned).limit(1).count(),
        second_aligned.exceptAll(first_aligned).limit(1).count(),
    )


def test_full_ads_reconciliation_quality_and_second_build_idempotency(
    spark: SparkSession, snapshot_directory: Path
) -> None:
    """Exercise the full ads reconciliation quality and second build idempotency case."""
    paths = get_project_paths(load_config())
    outputs = _output_paths(paths)
    dwd = spark.read.parquet(str(paths.resolve(paths.dwd_flights)))
    carrier_daily = spark.read.parquet(
        str(paths.resolve(paths.dws_carrier_daily))
    )
    airport_daily = spark.read.parquet(
        str(paths.resolve(paths.dws_origin_airport_daily))
    )
    route_daily = spark.read.parquet(str(paths.resolve(paths.dws_route_daily)))
    ads = _load_ads(spark, outputs)
    dws_sources, dwd_sources = _source_views(
        dwd, carrier_daily, airport_daily, route_daily
    )

    row_counts: Dict[str, int] = {}
    quality: Dict[str, Dict[str, int]] = {}
    reconciliations: Dict[str, Dict[str, int]] = {}
    for name, dataframe in ads.items():
        grain = GRAINS[name]
        expected_columns = [*grain, *ADS_METRIC_COLUMNS]
        assert_required_columns(dataframe, expected_columns)
        assert dataframe.columns == expected_columns
        row_counts[name] = dataframe.count()
        if row_counts[name]:
            assert len(list(outputs[name].glob("part-*.parquet"))) == 1
        assert row_counts[name] == dws_sources[name].select(*grain).distinct().count()

        expected_dws_counts = _aggregate_dws_counts(dws_sources[name], grain)
        expected_dwd = _aggregate_dwd_metrics(dwd_sources[name], grain)
        quality[name] = _quality_violations(dataframe, grain)
        reconciliations[name] = _reconciliation_mismatches(
            dataframe, expected_dws_counts, expected_dwd, grain
        )
        assert all(count == 0 for count in quality[name].values())
        assert all(count == 0 for count in reconciliations[name].values())
        assert not any(path.is_dir() for path in outputs[name].iterdir())

    reverse_route_pairs = (
        ads["route"]
        .alias("forward")
        .join(
            ads["route"].alias("reverse"),
            (F.col("forward.origin") == F.col("reverse.destination"))
            & (F.col("forward.destination") == F.col("reverse.origin"))
            & (F.col("forward.origin") < F.col("forward.destination")),
            "inner",
        )
        .count()
    )
    assert reverse_route_pairs > 0

    snapshot_paths = {
        name: snapshot_directory / name for name in ads
    }
    for name, dataframe in ads.items():
        dataframe.select(*GRAINS[name], *ADS_METRIC_COLUMNS).write.mode(
            "overwrite"
        ).parquet(str(snapshot_paths[name]))

    _build_all_ads(spark, paths, outputs)
    spark.catalog.clearCache()
    second_ads = _load_ads(spark, outputs)
    idempotency: Dict[str, tuple[int, int]] = {}
    for name, second_dataframe in second_ads.items():
        first_dataframe = spark.read.parquet(str(snapshot_paths[name]))
        columns = [*GRAINS[name], *ADS_METRIC_COLUMNS]
        idempotency[name] = _multiset_differences(
            first_dataframe, second_dataframe, columns
        )
        assert idempotency[name] == (0, 0)
        if row_counts[name]:
            assert len(list(outputs[name].glob("part-*.parquet"))) == 1

    print(f"ads_row_counts={row_counts}")
    print(f"ads_quality_violations={quality}")
    print(f"ads_reconciliation_mismatches={reconciliations}")
    print(f"reverse_route_pairs={reverse_route_pairs}")
    print(f"ads_idempotency_differences={idempotency}")

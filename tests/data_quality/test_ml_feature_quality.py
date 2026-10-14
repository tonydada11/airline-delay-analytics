# AI-generated test code: generated with OpenAI Codex.
"""Snapshot acceptance: ordinary validation never rebuilds the feature dataset."""
from datetime import date, timedelta
import os
from pathlib import Path
import shutil
import tempfile
import pytest
from pyspark.sql import functions as F
from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.spark_session import create_spark_session
from airline.dws.ml_feature_contract import *
from airline.dws.build_ml_flight_features import (
    eligibility_condition, select_eligible_flights, validate_feature_frame,
    build_ml_flight_features,
)

EXPECTED_ROWS = {"train": 8_946_370, "validation": 2_418_739, "test": 2_343_541}
pytestmark = pytest.mark.skipif(os.environ.get("AIRLINE_RUN_FULL_ML_QUALITY") != "1", reason="full snapshot acceptance is opt-in")


@pytest.fixture(scope="module")
def spark():
    session = create_spark_session("airline-ml-feature-quality")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def datasets(spark):
    paths = get_project_paths(load_config())
    dwd = spark.read.parquet(str(paths.resolve(paths.dwd_flights)))
    features = spark.read.parquet(str(paths.resolve(paths.dws_ml_flight_features)))
    return paths, dwd, features


def test_feature_contract_counts_and_reconciliation(datasets):
    _, dwd, features = datasets
    assert validate_feature_frame(features) == sum(EXPECTED_ROWS.values())
    assert {r.split: r['count'] for r in features.groupBy("split").count().collect()} == EXPECTED_ROWS
    eligible = select_eligible_flights(dwd)
    assert eligible.count() == features.count()
    # Per-date/target/entity reconciliation catches multiplicity changes, not just totals.
    keys = ["flight_date", "reporting_airline", "origin", "destination", TARGET_COLUMN]
    a, b = eligible.groupBy(*keys).count(), features.groupBy(*keys).count()
    assert a.exceptAll(b).limit(1).count() == 0
    assert b.exceptAll(a).limit(1).count() == 0


def test_sampled_independent_histories(datasets):
    _, dwd, features = datasets
    eligible = select_eligible_flights(dwd)
    # Deterministic samples include both split transitions and the documented October example.
    for day in (date(2023,1,1), VALIDATION_START, TEST_START, date(2024,10,20)):
        row = features.filter(F.col("flight_date") == day).orderBy("reporting_airline", "origin", "destination").first()
        assert row is not None
        prior = eligible.filter(F.col("flight_date").between(day-timedelta(days=90), day-timedelta(days=1)))
        expressions = []
        for entity, keys in ENTITIES:
            condition = F.lit(True)
            for key in keys:
                condition = condition & F.col(key).eqNullSafe(F.lit(row[key]))
            for days in WINDOW_DAYS:
                selected = condition & (F.col("flight_date") >= F.lit(day-timedelta(days=days)))
                expressions.extend([F.sum(F.when(selected, 1).otherwise(0)).alias(f"{entity}_{days}_count"),
                                    F.sum(F.when(selected, F.col(TARGET_COLUMN)).otherwise(0)).alias(f"{entity}_{days}_delayed")])
        independent = prior.agg(*expressions).first()
        for entity, _ in ENTITIES:
            for days in WINDOW_DAYS:
                count = independent[f"{entity}_{days}_count"] or 0
                delayed = independent[f"{entity}_{days}_delayed"] or 0
                assert row[f"{entity}_flight_count_{days}d"] == count
                assert row[f"{entity}_history_missing_{days}d"] == int(count == 0)
                rate = row[f"{entity}_delay_rate_{days}d"]
                assert rate == pytest.approx(delayed/count) if count else rate is None


@pytest.mark.skipif(os.environ.get("AIRLINE_RUN_REBUILD_ACCEPTANCE") != "1", reason="explicit comprehensive rebuild only")
def test_comprehensive_rebuild_multiset_equality(spark, datasets):
    paths, _, original = datasets
    root = paths.project_root / "data/.test-tmp"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="rebuild-", dir=root))
    try:
        build_ml_flight_features(spark, paths.resolve(paths.dwd_flights), directory / "features")
        rebuilt = spark.read.parquet(str(directory / "features")).select(*OUTPUT_COLUMNS)
        assert original.exceptAll(rebuilt).limit(1).count() == 0
        assert rebuilt.exceptAll(original).limit(1).count() == 0
    finally:
        shutil.rmtree(directory)

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
from airline.dws.dataset_manifest import validate_dataset_manifest
from airline.common.raw_snapshot import sha256_file
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
    manifest = validate_dataset_manifest(paths.resolve(paths.dws_ml_flight_features), features, sha256_file(paths.resolve(paths.raw_manifest)))
    assert manifest["duplicate_key_count"] == 0
    assert manifest["rows"] == features.count()
    if os.environ.get('AIRLINE_ML_FEATURE_PHASE') != 'selection':
        test = spark.read.parquet(str(paths.resolve(paths.dws_ml_test_features)))
        test_manifest = validate_dataset_manifest(paths.resolve(paths.dws_ml_test_features), test, sha256_file(paths.resolve(paths.raw_manifest)), phase='test')
        assert test_manifest['duplicate_key_count'] == 0
        assert test_manifest['rows'] == test.count()
        features = features.unionByName(test)
    else:
        dwd = dwd.filter(F.col('flight_date') <= F.lit(VALIDATION_END))
    return paths, dwd, features


def test_feature_contract_counts_and_reconciliation(datasets):
    _, dwd, features = datasets
    expected = {k:v for k,v in EXPECTED_ROWS.items() if k != 'test' or os.environ.get('AIRLINE_ML_FEATURE_PHASE') != 'selection'}
    assert validate_feature_frame(features) == sum(expected.values())
    assert {r.split: r['count'] for r in features.groupBy('split').count().collect()} == expected
    eligible = select_eligible_flights(dwd)
    assert eligible.count() == features.count()
    # Per-date/target/entity reconciliation catches multiplicity changes, not just totals.
    keys = ["flight_id", "flight_number_reporting_airline", "origin_airport_id", "dest_airport_id", "crs_dep_time", "flight_date", "reporting_airline", "origin", "destination", TARGET_COLUMN]
    a, b = eligible.groupBy(*keys).count(), features.groupBy(*keys).count()
    assert a.exceptAll(b).limit(1).count() == 0
    assert b.exceptAll(a).limit(1).count() == 0


def test_sampled_independent_histories(datasets):
    _, dwd, features = datasets
    from airline.dws.flight_time import DAY_US
    eligible = select_eligible_flights(dwd)
    # Deterministic samples include both split transitions and the documented October example.
    for day in (date(2023,1,1), VALIDATION_START, TEST_START, date(2024,10,20)):
        row = features.filter(F.col("flight_date") == day).orderBy("reporting_airline", "origin", "destination").first()
        if row is None:
            assert os.environ.get('AIRLINE_ML_FEATURE_PHASE') == 'selection' and day >= TEST_START
            continue
        clock = eligible.filter((F.col('flight_date') == day) & (F.col('flight_id') == row.flight_id)).first()
        target = clock['_prediction_us']
        # Broad local-date envelope is only a scan optimization; exact UTC predicates
        # below remain authoritative, including timezone/2400 rollover boundaries.
        candidates = dwd.filter(F.col('flight_date').between(day-timedelta(days=92), day+timedelta(days=2)))
        prior = select_eligible_flights(candidates).filter((F.col('_available_us') < target) & (F.col('_prediction_us') < target))
        if day <= VALIDATION_END:
            prior = prior.filter(F.col('flight_date') <= F.lit(VALIDATION_END))
        expressions = []
        for entity, keys in ENTITIES:
            condition = F.lit(True)
            for key in keys:
                condition = condition & F.col(key).eqNullSafe(F.lit(row[key]))
            for days in WINDOW_DAYS:
                selected = condition & (F.col("_prediction_us") >= F.lit(target-days*DAY_US if target is not None else None))
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
        selection = original.filter(~F.col('split').eqNullSafe(F.lit('test')))
        assert selection.exceptAll(rebuilt).limit(1).count() == 0
        assert rebuilt.exceptAll(selection).limit(1).count() == 0
        print('selection rebuild: zero multiset differences in both directions')
        if os.environ.get('AIRLINE_ML_FEATURE_PHASE') != 'selection':
            from airline.ml.model_contract import load_bundle
            _, _, manifest = load_bundle(paths.resolve(paths.artifact_root) / 'models')
            build_ml_flight_features(spark, paths.resolve(paths.dwd_flights), directory / 'test', phase='test', selection_manifest=manifest)
            rebuilt_test = spark.read.parquet(str(directory / 'test')).select(*OUTPUT_COLUMNS)
            original_test = original.filter(F.col('split') == 'test')
            assert original_test.exceptAll(rebuilt_test).limit(1).count() == 0
            assert rebuilt_test.exceptAll(original_test).limit(1).count() == 0
            print('test rebuild: zero multiset differences in both directions')
    finally:
        shutil.rmtree(directory)

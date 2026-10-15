# AI-generated test code: generated with OpenAI Codex.
"""Compare accepted features and frozen-model scoring without altering production outputs."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import shutil
from unittest.mock import patch

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.sql import functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.raw_snapshot import sha256_file
from airline.common.spark_session import create_spark_session
from airline.dws.ml_feature_contract import OUTPUT_COLUMNS
from airline.ml.evaluate import evaluate
from airline.ml.model_contract import PREDICTION_COLUMNS


def assert_same_metrics(expected, actual):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_same_metrics(expected[key], actual[key])
    elif isinstance(expected, float):
        assert math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-10), (expected, actual)
    else:
        assert expected == actual, (expected, actual)


def check_features(spark, paths, baseline):
    result = {}
    for phase, configured in (('selection', paths.dws_ml_flight_features), ('test', paths.dws_ml_test_features)):
        old = spark.read.parquet(str(baseline / phase)).select(*OUTPUT_COLUMNS)
        new = spark.read.parquet(str(paths.resolve(configured))).select(*OUTPUT_COLUMNS)
        assert old.dtypes == new.dtypes
        assert old.exceptAll(new).limit(1).count() == 0, f'{phase}: baseline rows missing or changed'
        assert new.exceptAll(old).limit(1).count() == 0, f'{phase}: new or changed rows'
        result[phase] = {'rows': old.count(), 'baseline_minus_current': 0, 'current_minus_baseline': 0,
                         'columns': list(OUTPUT_COLUMNS)}
        print(f'{phase}: all-column baseline/current multisets are equal', flush=True)
    return result


def check_frozen_rescore(spark, paths, baseline, verification):
    original = baseline / 'artifacts/flighttime-v3'
    manifest_path = original / 'models/manifest.json'
    manifest_sha256 = sha256_file(manifest_path)
    verification.mkdir(parents=True, exist_ok=False)
    shutil.copytree(original / 'models', verification / 'models')
    fingerprint = sha256_file(paths.resolve(paths.raw_manifest))
    results = {}
    with ExitStack() as stack:
        for estimator in (Pipeline, LogisticRegression, GBTClassifier):
            stack.enter_context(patch.object(estimator, 'fit', side_effect=AssertionError('Frozen-model verification attempted fitting')))
        for split, phase in (('validation', 'selection'), ('test', 'test')):
            actual = evaluate(spark, baseline / phase, verification, split, fingerprint)
            expected = json.loads((original / 'metrics' / f'{split}.json').read_text())
            assert_same_metrics({k: v for k, v in expected.items() if k != 'completion'},
                                {k: v for k, v in actual.items() if k != 'completion'})
            old = spark.read.parquet(str(original / 'predictions' / split))
            new = spark.read.parquet(str(verification / 'predictions' / split))
            decisions = [c for c in PREDICTION_COLUMNS if c != 'probability']
            assert old.select(*decisions).exceptAll(new.select(*decisions)).limit(1).count() == 0
            assert new.select(*decisions).exceptAll(old.select(*decisions)).limit(1).count() == 0
            compared = [F.round('probability', 10).alias('probability') if c == 'probability' else c for c in PREDICTION_COLUMNS]
            assert old.select(*compared).exceptAll(new.select(*compared)).limit(1).count() == 0
            assert new.select(*compared).exceptAll(old.select(*compared)).limit(1).count() == 0
            results[split] = {'rows': actual['rows'], 'decision_multiset_differences': 0,
                              'score_multiset_differences_at_10_decimal_places': 0,
                              'metric_absolute_tolerance': 1e-10, 'metric_relative_tolerance': 1e-6}
            print(f'{split}: original frozen-model predictions and metrics are equivalent', flush=True)
    assert sha256_file(manifest_path) == manifest_sha256
    return {'baseline_model_manifest_sha256': manifest_sha256,
            'baseline_code_provenance': json.loads(manifest_path.read_text())['code_provenance'],
            'verification_output': str(verification.relative_to(paths.project_root)), 'splits': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('check', choices=('features', 'frozen-rescore'))
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verification-root', type=Path)
    args = parser.parse_args()
    paths = get_project_paths(load_config())
    baseline = (paths.project_root / args.baseline).resolve()
    spark = create_spark_session('airline-consolidation-' + args.check)
    try:
        if args.check == 'features':
            result = check_features(spark, paths, baseline)
        else:
            if args.verification_root is None:
                parser.error('--verification-root is required for frozen-rescore')
            result = check_frozen_rescore(spark, paths, baseline, paths.project_root / args.verification_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump({'status': 'passed', 'check': args.check, 'result': result}, stream, indent=2)
            stream.write('\n')
    finally:
        spark.stop()

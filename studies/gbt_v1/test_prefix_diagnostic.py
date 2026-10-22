"""No fitting: generated vectors plus read-only loading of the saved GBT30."""
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from studies.gbt_v1 import prefix_diagnostic as diagnostic
from studies.lr_v1.metrics import MONTHS


def test_ordered_prefix_retains_exact_nonuniform_weights():
    trees, weights = diagnostic.prefix_members(['a', 'b', 'c'], [1.0, 0.1, -0.25], 2)
    assert trees == ['a', 'b'] and weights == [1.0, 0.1]
    assert sum(w * p for w, p in zip(weights, [2.0, 3.0])) == 2.3
    for count in (0, 4, True):
        with pytest.raises(ValueError, match='Invalid prefix'):
            diagnostic.prefix_members(['a', 'b', 'c'], [1.0, 0.1, -0.25], count)


def test_reconciliation_is_required_and_missing_months_fail():
    saved = {m: {'pr_auc': 0.4} for m in MONTHS}
    months = {m: {'pr_auc': {'30': 0.4 + diagnostic.AUC_ATOL / 2}} for m in MONTHS}
    assert len(diagnostic.reconcile(months, saved)) == 4
    months[MONTHS[0]]['pr_auc']['30'] = 0.41
    with pytest.raises(ValueError, match='differs from saved'):
        diagnostic.reconcile(months, saved)
    del months[MONTHS[0]]
    with pytest.raises(ValueError, match='Missing validation month'):
        diagnostic.reconcile(months, saved)


def test_paired_changes_and_equal_month_means():
    months = {m: {'pr_auc': {'10': .3, '20': .4, '25': .41, '30': .409 if i == 0 else .412}}
              for i, m in enumerate(MONTHS)}
    lr = {m: {'pr_auc': .43} for m in MONTHS}
    summary = diagnostic.summarize(months, lr)
    assert summary['equal_month_means']['10'] == .3
    assert summary['changes']['25->30']['mean'] == pytest.approx(.00125)
    assert summary['changes']['25->30']['worst'] == pytest.approx(-.001)


def test_exclusive_path_and_symlink_protection(tmp_path):
    path = diagnostic.exclusive_output(tmp_path)
    assert path == tmp_path / diagnostic.OUTPUT
    path.mkdir(parents=True)
    (path / 'keep').write_text('partial')
    with pytest.raises(FileExistsError):
        diagnostic.exclusive_output(tmp_path)
    assert (path / 'keep').read_text() == 'partial'
    (path / 'keep').unlink()
    path.rmdir()
    path.symlink_to(tmp_path / 'absent')
    with pytest.raises(ValueError, match='Symlink'):
        diagnostic.exclusive_output(tmp_path)


def test_timeout_interrupts_blocking_work_and_never_completes(tmp_path):
    exit_code = diagnostic.supervise(tmp_path, [sys.executable, '-B', '-c', 'import time; time.sleep(30)'],
                                    revision='fixture', limit=.8, grace=.2)
    assert exit_code == 1
    output = tmp_path / diagnostic.OUTPUT
    record = diagnostic.study.read_json(output / 'attempt.json')
    assert record['status'] == 'incomplete' and record['exit_code'] != 0
    assert record['total_seconds'] < 2 and record['classifier_fits_started'] == 0
    assert not (output / 'completion.json').exists()
    with pytest.raises(FileExistsError):
        diagnostic.supervise(tmp_path, ['must-not-start'], revision='fixture')


def test_import_and_help_do_not_start_spark_or_create_outputs(tmp_path):
    code = '''
import subprocess, sys
def forbidden(*a, **kw): raise AssertionError("unexpected subprocess")
subprocess.Popen = forbidden
from studies.gbt_v1 import prefix_diagnostic, launch_prefix
for module in (prefix_diagnostic, launch_prefix):
    try: module.main(["--help"])
    except SystemExit as exc: assert exc.code == 0
assert not any(n.startswith("pyspark") for n in sys.modules)
'''
    subprocess.run([sys.executable, '-B', '-c', code], cwd=tmp_path, check=True,
                   env=dict(os.environ, PYTHONPATH=str(diagnostic.ROOT)))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('damage', [None, 'check', 'fit', 'shutdown', 'exit'])
def test_completion_requires_all_checks_and_successful_worker_termination(tmp_path, damage):
    saved = diagnostic.study.read_json(diagnostic.ROOT / diagnostic.REFERENCE / 'results.json')['months']
    result = {'status': 'pending_termination', 'months': {m: {'pr_auc': {'30': saved[m]['pr_auc']}} for m in MONTHS},
              'checks': {key: True for key in diagnostic.CHECKS}}
    state = {'stage': 'evaluated', 'shutdown_successful': damage != 'shutdown',
             'classifier_fits_started': int(damage == 'fit')}
    if damage == 'check':
        result['checks']['population'] = False
    output = tmp_path / diagnostic.OUTPUT
    code = f'''
import json, pathlib, sys
output = pathlib.Path({str(output)!r})
(output / 'results.json').write_text(json.dumps({result!r}))
(output / 'worker-status.json').write_text(json.dumps({state!r}))
sys.exit({int(damage == 'exit')})
'''
    status = diagnostic.supervise(tmp_path, [sys.executable, '-B', '-c', code], revision='fixture')
    assert status == int(damage is not None)
    assert (output / 'completion.json').exists() == (damage is None)


def test_native_prefix_scoring_matches_weighted_tree_oracle_without_fitting(monkeypatch):
    from pyspark.ml.base import Estimator
    from pyspark.ml.classification import GBTClassificationModel
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import functions as F
    from airline.common.spark_session import create_spark_session
    from airline.ml.experiments import continuous_score
    def forbidden(*args, **kwargs):
        raise AssertionError('No fixture or real-data fitting permitted')
    monkeypatch.setattr(Estimator, 'fit', forbidden)
    spark = create_spark_session('gbt-prefix-generated-vectors-test')
    try:
        root = Path(os.environ.get('AIRLINE_PREFIX_INPUT_ROOT', '/workspace'))
        model = GBTClassificationModel.load(str(root / diagnostic.MODEL))
        original_weights = list(model.treeWeights)
        original_parameters = {p.name: model.getOrDefault(p) for p in model.params if model.isDefined(p)}
        assert model.getNumTrees == 30 and len(set(original_weights)) > 1
        vectors = [Vectors.sparse(model.numFeatures, {}),
                   Vectors.dense([1.0] * model.numFeatures),
                   Vectors.sparse(model.numFeatures, {0: 1.0, 783: 100.0})]
        # Duplicate fixture records deliberately retain multiplicity.
        fixture = spark.createDataFrame([(i, v) for i, v in enumerate(vectors)] + [(0, vectors[0])], ['id', 'features'])
        for count in diagnostic.PREFIXES:
            prefix = diagnostic.prefix_model(model, count, spark)
            expected = []
            for i, vector in enumerate(vectors):
                margin = sum(t.predict(vector) * w for t, w in zip(model.trees[:count], original_weights[:count]))
                # Pinned LogLoss + GBT raw2probability: p1 = 1 - logistic(-margin).
                probability = 1.0 - 1.0 / (1.0 + math.exp(2.0 * margin))
                expected.append((i, probability))
            expected_frame = spark.createDataFrame(expected, ['id', 'expected'])
            scores = continuous_score(prefix, fixture).select('id', 'probability')
            check = scores.join(expected_frame, 'id').agg(F.count('*').alias('rows'),
                F.max(F.abs(F.col('probability') - F.col('expected'))).alias('error')).first()
            assert check.rows == 4 and check.error <= diagnostic.SCORE_ATOL
            if count == 30:
                diagnostic.study.same_multiset(scores, continuous_score(model, fixture).select('id', 'probability'))
        assert model.getNumTrees == 30 and list(model.treeWeights) == original_weights
        assert {p.name: model.getOrDefault(p) for p in model.params if model.isDefined(p)} == original_parameters
        # A parameter change on a COPY demonstrably does not remove fitted trees.
        assert model.copy({model.maxIter: 10}).getNumTrees == 30
        # Existing multiset helper detects dropped duplicates and changed labels.
        original = spark.createDataFrame([(1, 0), (1, 0), (2, 1)], ['id', 'label'])
        with pytest.raises(ValueError, match='multiset differs'):
            diagnostic.study.same_multiset(original, original.distinct())
        with pytest.raises(ValueError, match='multiset differs'):
            diagnostic.study.same_multiset(original, original.withColumn('label', F.lit(1)))
    finally:
        spark.stop()

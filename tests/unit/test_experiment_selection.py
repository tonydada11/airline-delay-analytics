# AI-generated test code: generated with OpenAI Codex.
from airline.ml.experiments import f1_fraction, compare_f1, best_threshold, select_classifier, run_directory, GBT_SETTINGS
import pytest


def c(tp=0, fp=0, fn=0, step=0, **extra):
    return dict(tp=tp, fp=fp, fn=fn, tn=10, step=step, **extra)


def test_exact_rational_ties_and_zero():
    assert f1_fraction(c()) == (0, 1)
    assert compare_f1(c(1,1,1), c(2,2,2)) == 0
    assert compare_f1(c(10**20,1,0), c(10**20,2,0)) == 1
    assert best_threshold([c(1,1,1,99), c(2,2,2,3)])['step'] == 3
    assert best_threshold([c(step=100), c(step=0)])['step'] == 0


def test_model_tie_baseline_and_rankings_excluded():
    results = {'lr':c(1,1,1, pr_auc=0), 'gbt':c(2,2,2, pr_auc=1), 'baseline':c(10)}
    assert select_classifier(results) == 'lr'
    results['gbt'] = c(3,1,1, pr_auc=0)
    assert select_classifier(results) == 'gbt'
    assert GBT_SETTINGS['seed'] == 42 and GBT_SETTINGS['maxIter'] == 10


@pytest.mark.parametrize('lr,gbt,winner', [
    (c(1,1,1,80), c(2,2,2,20), 'gbt'),
    (c(1,1,1,20), c(2,2,2,80), 'lr'),
    (c(1,1,1,20), c(2,2,2,20), 'lr'),
    (c(step=80), c(step=20), 'gbt'),
    (c(step=20), c(step=20), 'lr'),
    (c(3,1,1,80), c(1,1,1,20), 'lr'),
    (c(1,1,1,20), c(3,1,1,80), 'gbt'),
])
def test_classifier_order_is_f1_then_step_then_lr(lr, gbt, winner):
    assert select_classifier({'lr': lr, 'gbt': gbt, 'baseline': c(100)}) == winner


@pytest.mark.parametrize('lr_step,gbt_step,winner', [(80,20,'gbt'), (20,20,'lr'), (20,80,'lr')])
def test_selection_manifest_enforces_cross_model_step_ties(lr_step, gbt_step, winner):
    from airline.ml.model_contract import validate_selection, ESTIMATOR_SETTINGS
    from airline.dws.flight_time import time_provenance
    from airline.dws.ml_feature_contract import FEATURE_CONTRACT_VERSION
    from airline.common.raw_snapshot import SNAPSHOT_ID
    results = {}
    for name, step in [('lr', lr_step), ('gbt', gbt_step)]:
        rows = [c(tp=int(s >= step), fn=int(s < step), step=s) for s in range(101)]
        results[name] = {'thresholds': rows, 'selected': best_threshold(rows)}
    manifest = dict(schema_version=3, source_snapshot_id=SNAPSHOT_ID,
        feature_version=FEATURE_CONTRACT_VERSION, estimator_settings=ESTIMATOR_SETTINGS,
        time_policy=time_provenance(), threshold_denominator=100, positive_class_weight=2.0,
        validation_results=results, selected_classifier=winner,
        threshold_step=results[winner]['selected']['step'])
    validate_selection(manifest)
    other = 'lr' if winner == 'gbt' else 'gbt'
    manifest.update(selected_classifier=other, threshold_step=results[other]['selected']['step'])
    with pytest.raises(ValueError, match='validation evidence'):
        validate_selection(manifest)


def test_run_paths_fail_closed(tmp_path):
    for value in ('../models', '/tmp/x', '', 'a/b', '.'):
        with pytest.raises(ValueError): run_directory(tmp_path, value)
    (tmp_path / 'experiments').symlink_to(tmp_path / 'models')
    with pytest.raises(ValueError): run_directory(tmp_path, 'run')

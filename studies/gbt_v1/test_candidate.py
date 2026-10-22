"""Focused checks; Spark reads only generated fixtures, never the warehouse."""
import copy
from datetime import date
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tarfile
import textwrap
import time
from types import SimpleNamespace

import pytest

from studies.gbt_v1 import run_candidate as runner
from studies.lr_v1.metrics import MONTHS, monthly_comparison, expected_budget


def baseline_months(rows=10):
    return {m: {'rows': rows, 'pr_auc': 0.4, 'roc_auc': 0.5, 'precision': 0.5,
                'recall': 1.0, 'f1': 2 / 3, 'prevalence': 0.5, 'predicted_positive_rate': 1.0,
                'confusion_matrix': {'true_positive': rows // 2, 'false_negative': 0,
                                     'false_positive': rows // 2, 'true_negative': 0},
                'budgets': {str(p): {'recall': 0.2, 'precision': 0.5} for p in (10, 20)}} for m in MONTHS}


def test_candidate_is_the_single_change_without_mutation():
    from airline.ml.model_contract import GBT_SETTINGS, POSITIVE_CLASS_WEIGHT
    original = copy.deepcopy(GBT_SETTINGS)
    candidate = runner.candidate_settings()
    assert candidate is not GBT_SETTINGS and GBT_SETTINGS == original
    assert {k for k in candidate if candidate[k] != original[k]} == {'featureSubsetStrategy'}
    assert candidate['featureSubsetStrategy'] == 'all'
    assert candidate['subsamplingRate'] == 0.5 and candidate['seed'] == 42
    assert POSITIVE_CLASS_WEIGHT == 2.0


def test_30iter_changes_only_iterations_relative_to_reference():
    from airline.ml.model_contract import GBT_SETTINGS
    original = copy.deepcopy(GBT_SETTINGS)
    reference = runner.candidate_settings()
    candidate = runner.candidate_settings('gbt-all-30iter')
    assert {k for k in candidate if candidate[k] != reference[k]} == {'maxIter'}
    assert candidate['maxIter'] == 30 and reference['maxIter'] == 10
    assert {k for k in candidate if candidate[k] != original[k]} == {'maxIter', 'featureSubsetStrategy'}
    assert GBT_SETTINGS == original and runner.candidate_settings() == reference


def test_60iter_changes_only_iterations_relative_to_30iter():
    from airline.ml.model_contract import GBT_SETTINGS
    original = copy.deepcopy(GBT_SETTINGS)
    reference = runner.candidate_settings('gbt-all-30iter')
    candidate = runner.candidate_settings('gbt-all-60iter')
    assert {k for k in candidate if candidate[k] != reference[k]} == {'maxIter'}
    assert candidate['maxIter'] == 60 and reference['maxIter'] == 30
    assert {k for k in candidate if candidate[k] != original[k]} == {'maxIter', 'featureSubsetStrategy'}
    assert GBT_SETTINGS == original and runner.candidate_settings()['maxIter'] == 10


def test_fixed_candidate_output_and_plan_selection(tmp_path):
    assert runner.candidate_contract() == (runner.ATTEMPT, 'PLAN.md', 10)
    attempt, plan, iterations = runner.candidate_contract('gbt-all-30iter')
    assert attempt == Path('data/studies/gbt-improvement-v1/attempt-002')
    assert plan == 'PLAN_30ITER.md' and iterations == 30
    assert runner.attempt_path(tmp_path, 'gbt-all-30iter') == tmp_path / attempt
    attempt, plan, iterations = runner.candidate_contract('gbt-all-60iter')
    assert attempt == Path('data/studies/gbt-improvement-v1/attempt-003')
    assert plan == 'PLAN_60ITER.md' and iterations == 60
    assert runner.attempt_path(tmp_path, 'gbt-all-60iter') == tmp_path / attempt
    with pytest.raises(ValueError, match='Unknown fixed candidate'):
        runner.candidate_contract('unapproved')


@pytest.mark.parametrize('component', ['data', 'data/studies', 'data/studies/gbt-improvement-v1'])
def test_protected_paths_and_symlinks_cannot_be_followed(tmp_path, component):
    protected = tmp_path / 'artifacts/flighttime-v3/models'
    protected.mkdir(parents=True)
    (protected / 'untouched').write_text('accepted')
    link = tmp_path / component
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(protected, target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        runner.supervise(tmp_path, ['must-not-start'])
    assert (protected / 'untouched').read_text() == 'accepted'
    assert not (protected / 'attempt-001').exists()


@pytest.mark.parametrize('candidate', runner.CANDIDATES)
def test_existing_or_broken_link_attempt_is_preserved(tmp_path, candidate):
    attempt = tmp_path / runner.candidate_contract(candidate)[0]
    attempt.mkdir(parents=True)
    (attempt / 'partial').write_text('keep')
    with pytest.raises(FileExistsError):
        runner.supervise(tmp_path, ['must-not-start'], candidate=candidate)
    assert (attempt / 'partial').read_text() == 'keep'
    shutil.rmtree(attempt)
    attempt.symlink_to(tmp_path / 'missing')
    with pytest.raises(ValueError):
        runner.attempt_path(tmp_path, candidate)


def test_import_help_and_missing_execution_flag_have_no_side_effects(tmp_path):
    code = '''
import sys, subprocess
def forbidden(*a, **k): raise AssertionError("unexpected subprocess/Spark startup")
subprocess.Popen = forbidden
from studies.gbt_v1 import run_candidate, launch
assert not any(n.startswith("pyspark") for n in sys.modules)
for module in (run_candidate, launch):
    for args in (["--help"], ["--candidate", "gbt-all-30iter", "--help"], ["--candidate", "gbt-all-60iter", "--help"]):
        try: module.main(args)
        except SystemExit as exc: assert exc.code == 0
    try: module.main([])
    except SystemExit as exc: assert exc.code == 2
assert not any(n.startswith("pyspark") for n in sys.modules)
'''
    subprocess.run([sys.executable, '-B', '-c', code], cwd=tmp_path, check=True,
                   env=dict(os.environ, PYTHONPATH=str(runner.SOURCE_ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')))
    assert list(tmp_path.iterdir()) == []


def test_worker_cannot_bypass_supervisor(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.worker(tmp_path, 'not-authorized')
    assert not (tmp_path / runner.ATTEMPT).exists()


def test_container_bootstrap_imports_reused_helper_without_running_it():
    from studies.gbt_v1.launch import BOOTSTRAP
    payload = io.BytesIO()
    files = {'studies/gbt_v1/run_candidate.py': b'from studies.lr_v1.diagnose import budget_metrics\ndef main(argv, *, cleanup):\n    cleanup()\n    print("fixture bootstrap passed")\n    return 0\n'}
    for name in ('diagnose.py', 'metrics.py'):
        files['studies/lr_v1/' + name] = (runner.SOURCE_ROOT / 'studies/lr_v1' / name).read_bytes()
    with tarfile.open(fileobj=payload, mode='w') as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    completed = subprocess.run([sys.executable, '-B', '-c', BOOTSTRAP, 'fixture-revision', runner.NAME],
                               input=b'0\n' + payload.getvalue(), capture_output=True, check=True)
    assert completed.stdout.splitlines() == [b'GBT_STUDY_READY', b'fixture bootstrap passed']


def test_worker_input_failure_records_no_fit_and_no_completion(tmp_path, monkeypatch):
    attempt = tmp_path / runner.ATTEMPT
    attempt.mkdir(parents=True)
    runner.write_json(attempt / 'attempt.json', {'token': 'fixture'})
    monkeypatch.setenv('AIRLINE_GBT_WORKER_TOKEN', 'fixture')
    monkeypatch.setattr(runner, 'inspect_inputs', lambda root, **kw: (_ for _ in ()).throw(ValueError('fixture missing input')))
    with pytest.raises(ValueError, match='fixture missing input'):
        runner.worker(tmp_path, 'fixture')
    state = runner.read_json(attempt / runner.NAME / 'status.json')
    assert state['stage'] == 'incomplete' and not state['fit_started']
    assert state['shutdown_successful'] and 'fixture missing input' in state['error']
    assert not (attempt / 'completion.json').exists()


def test_missing_metadata_fails_explicitly_without_fallback(tmp_path):
    # The actual config loader must fail, not fall back to repository defaults.
    with pytest.raises(FileNotFoundError, match='project.yaml'):
        runner.inspect_inputs(tmp_path)
    assert not (tmp_path / runner.ATTEMPT).exists()


@pytest.mark.parametrize('damage', ['configured_test_path', 'changed_bundle'])
def test_incompatible_input_contracts_fail_before_spark(tmp_path, damage):
    from airline.common.config import ProjectConfig
    import yaml
    paths = {k: 'unused/' + k for k in ProjectConfig.__dataclass_fields__ if k != 'project_root'}
    paths.update(dws_ml_flight_features=runner.FEATURES, artifact_root=runner.ARTIFACTS)
    if damage == 'configured_test_path':
        paths['dws_ml_flight_features'] = 'data/warehouse/dws/ml_flighttime_v3_test'
    conf = tmp_path / 'conf/project.yaml'
    conf.parent.mkdir()
    conf.write_text(yaml.safe_dump({'paths': paths}))
    manifest = tmp_path / runner.ARTIFACTS / 'models/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='Configured selection|Missing/changed input'):
        runner.inspect_inputs(tmp_path)
    assert not (tmp_path / runner.ATTEMPT).exists()


@pytest.mark.parametrize('damage', [None, 'local', 'published', 'missing'])
@pytest.mark.parametrize('candidate', ['gbt-all-30iter', 'gbt-all-60iter'])
def test_iteration_study_requires_unchanged_complete_reference_records(tmp_path, damage, candidate):
    source = tmp_path / 'source'
    directory, hashes, reference = runner.reference_records(candidate)
    for name in hashes:
        content = (runner.SOURCE_ROOT / directory / name).read_bytes()
        published = source / directory / name
        local = tmp_path / runner.candidate_contract(reference)[0] / ('completion.json' if name == 'completion.json'
                else str(Path(reference) / ('status.json' if name == 'worker-status.json' else name)))
        for target in (published, local):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    if damage == 'missing':
        local.unlink()
    elif damage == 'local':
        local.write_text('{}')
    elif damage == 'published':
        published.write_text('{}')
    if damage:
        with pytest.raises(ValueError, match='Missing/changed .* reference'):
            runner.inspect_reference(tmp_path, source, candidate=candidate)
    else:
        identities = runner.inspect_reference(tmp_path, source, candidate=candidate)
        assert {k: v['sha256'] for k, v in identities.items()} == hashes
    assert not (tmp_path / runner.candidate_contract(candidate)[0]).exists()


def dummy_result(tmp_path, monkeypatch, candidate=runner.NAME):
    source = tmp_path / 'source'
    path = source / runner.DIAGNOSTICS / 'results.json'
    path.parent.mkdir(parents=True)
    runner.write_json(path, {'validation_months': baseline_months()})
    monkeypatch.setattr(runner, 'SOURCE_ROOT', source)
    monkeypatch.setitem(runner.HASHES, 'baseline_results', runner.digest(path))
    result = {'name': candidate, 'status': 'pending_termination', 'months': baseline_months(),
              'checks': {k: True for k in ('parameters_defaults_dimensions', 'full_validation_score_reload',
                'saved_prediction_schema_population_threshold', 'all_months', 'inputs_unchanged')}}
    state = {'stage': 'evaluated', 'shutdown_successful': True, 'classifier_fits_started': 1,
             'fit_started': True, 'fit_seconds': 0.01}
    return result, state


def dummy_command(tmp_path, result, state, tail='', candidate=runner.NAME):
    attempt = tmp_path / runner.candidate_contract(candidate)[0]
    code = f'''
import atexit, json, pathlib, signal, time
attempt = pathlib.Path({str(attempt)!r})
output = attempt / {candidate!r}
output.mkdir()
(output / 'results.json').write_text({json.dumps(result)!r})
(output / 'status.json').write_text({json.dumps(state)!r})
atexit.register(lambda: (attempt / 'worker-exited').write_text('yes'))
{tail}
'''
    return [sys.executable, '-B', '-c', code]


@pytest.mark.parametrize('candidate', runner.CANDIDATES)
def test_only_supervisor_completes_after_worker_exit(tmp_path, monkeypatch, candidate):
    result, state = dummy_result(tmp_path, monkeypatch, candidate)
    assert runner.supervise(tmp_path, dummy_command(tmp_path, result, state, candidate=candidate),
                            limit=5, grace=1, candidate=candidate) == 0
    attempt = tmp_path / runner.candidate_contract(candidate)[0]
    assert (attempt / 'worker-exited').exists()
    completed = runner.read_json(attempt / 'completion.json')
    assert completed['status'] == 'complete' and completed['exit_code'] == 0
    assert completed['comparison'] == monthly_comparison(baseline_months(), result['months'])
    assert not completed['qualifies']
    assert completed['name'] == candidate
    assert completed['plan_sha256'] == runner.digest(Path(runner.__file__).with_name(runner.candidate_contract(candidate)[1]))


@pytest.mark.parametrize('damage', ['exit', 'shutdown', 'months', 'checks', 'second_fit'])
@pytest.mark.parametrize('candidate', runner.CANDIDATES)
def test_failed_or_incomplete_attempts_never_complete(tmp_path, monkeypatch, damage, candidate):
    result, state = dummy_result(tmp_path, monkeypatch, candidate)
    tail = ''
    if damage == 'exit':
        tail = 'raise SystemExit(7)'
    elif damage == 'shutdown':
        state['shutdown_successful'] = False
    elif damage == 'months':
        del result['months'][MONTHS[-1]]
    elif damage == 'checks':
        result['checks']['full_validation_score_reload'] = False
    else:
        state['classifier_fits_started'] = 2
    assert runner.supervise(tmp_path, dummy_command(tmp_path, result, state, tail, candidate),
                            limit=5, grace=1, candidate=candidate) == 1
    attempt = tmp_path / runner.candidate_contract(candidate)[0]
    assert not (attempt / 'completion.json').exists()
    assert runner.read_json(attempt / 'attempt.json')['qualifies'] is False
    with pytest.raises(FileExistsError):
        runner.supervise(tmp_path, ['must-not-retry'], candidate=candidate)


@pytest.mark.parametrize('candidate', runner.CANDIDATES)
def test_timeout_interrupts_blocking_process_and_records_failure(tmp_path, monkeypatch, candidate):
    result, state = dummy_result(tmp_path, monkeypatch, candidate)
    command = dummy_command(tmp_path, result, state, 'signal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(30)', candidate)
    start = time.monotonic()
    assert runner.supervise(tmp_path, command, limit=1.0, grace=0.4, candidate=candidate) == 1
    assert time.monotonic() - start < 2.5
    attempt = tmp_path / runner.candidate_contract(candidate)[0]
    record = runner.read_json(attempt / 'attempt.json')
    assert record['status'] == 'incomplete' and record['exit_code'] < 0
    assert not record['qualifies'] and not (attempt / 'completion.json').exists()
    assert not (attempt / 'worker-exited').exists()  # SIGKILL, not a clean exit.


@pytest.mark.parametrize('host_origin', [1_000.0, 1_000_000_000.0])
@pytest.mark.parametrize('candidate', runner.CANDIDATES)
def test_launcher_bootstrap_and_supervisor_use_durations_across_clock_origins(tmp_path, monkeypatch, host_origin, candidate):
    from studies.gbt_v1 import launch
    result, state = dummy_result(tmp_path, monkeypatch, candidate)
    command = dummy_command(tmp_path, result, state, candidate=candidate)
    observed = tmp_path / 'observed.json'
    cleaned = tmp_path / 'cleaned.txt'
    # The host clock charges seven seconds including Docker startup/readiness.
    host_times = iter((0.0, 2.0, 5.0, 7.0))
    monkeypatch.setattr(launch, 'time', SimpleNamespace(monotonic=lambda: host_origin + next(host_times)))
    real_popen = subprocess.Popen
    # Run the actual bootstrap in a separate process with an incompatible clock
    # origin, charging four seconds for staging and two for source cleanup.
    clock = f'''
import pathlib, tarfile, tempfile, time
real_monotonic = time.monotonic
clock_start = real_monotonic()
extra_seconds = [0.0]
time.monotonic = lambda: 100_000.0 + real_monotonic() - clock_start + extra_seconds[0]
original_extract = tarfile.TarFile.extract
staged = [False]
def charged_extract(self, *args, **kwargs):
    if not staged[0]:
        extra_seconds[0] += 4.0
        staged[0] = True
    return original_extract(self, *args, **kwargs)
tarfile.TarFile.extract = charged_extract
original_cleanup = tempfile.TemporaryDirectory.cleanup
def charged_cleanup(self):
    if not getattr(self, '_charged', False):
        self._charged = True
        extra_seconds[0] += 2.0
        pathlib.Path({str(cleaned)!r}).write_text(self.name)
    return original_cleanup(self)
tempfile.TemporaryDirectory.cleanup = charged_cleanup
'''
    # Keep the production budget path intact; replace only the Spark worker with
    # the existing small dummy process and redirect its attempt to a fixture root.
    injection = f'''
from studies.gbt_v1 import run_candidate as candidate
original_supervise = candidate.supervise
original_popen = candidate.subprocess.Popen
class ObservedPopen(original_popen):
    def wait(self, timeout=None):
        path = candidate.Path({str(observed)!r})
        if timeout is not None and not path.exists():
            path.write_text(candidate.json.dumps({{'worker_timeout': timeout}}))
        return super().wait(timeout=timeout)
def fixture_supervise(root, **kwargs):
    assert kwargs['candidate'] == {candidate!r}
    source = candidate.Path(candidate.__file__).resolve().parents[2]
    assert (source / 'studies/gbt_v1' / candidate.candidate_contract({candidate!r})[1]).is_file()
    if {candidate!r} != candidate.NAME:
        directory, hashes, _ = candidate.reference_records({candidate!r})
        for name, expected in hashes.items():
            assert candidate.digest(source / directory / name) == expected
    candidate.subprocess.Popen = ObservedPopen
    return original_supervise(candidate.Path({str(tmp_path)!r}), command={command!r}, **kwargs)
candidate.supervise = fixture_supervise
'''
    bootstrap = launch.BOOTSTRAP.replace(
        '    from studies.gbt_v1.run_candidate import main\n',
        '    from studies.gbt_v1.run_candidate import main\n' + textwrap.indent(injection, '    '))

    def container_process(argv, **kwargs):
        assert '--started-monotonic' not in ' '.join(argv)
        assert str(host_origin) not in argv
        # Use real pipes/readiness/payload transport, without invoking Docker or Spark.
        return real_popen([sys.executable, '-B', '-c', clock + bootstrap, argv[-2], argv[-1]], **kwargs)

    monkeypatch.setattr(launch, 'subprocess', SimpleNamespace(
        check_output=lambda argv, **kwargs: 'fixture-revision\n' if 'rev-parse' in argv else '',
        Popen=container_process, PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired))
    args = ['--execute'] if candidate == runner.NAME else ['--execute', '--candidate', candidate]
    assert launch.main(args) == 0
    completion = runner.read_json(tmp_path / runner.candidate_contract(candidate)[0] / 'completion.json')
    assert 11.0 <= completion['preparation_seconds'] < 20.0
    assert 13.0 <= completion['total_seconds'] < 30.0
    assert completion['limit_seconds'] == 14400 and completion['termination_allowance_seconds'] == 60
    assert 14320.0 < runner.read_json(observed)['worker_timeout'] <= 14329.0
    assert not Path(cleaned.read_text()).exists()  # Source cleanup precedes completion.


@pytest.mark.parametrize('damage', ['exhausted', 'failure'])
def test_source_cleanup_failure_or_exhausted_budget_never_completes(tmp_path, monkeypatch, damage):
    result, state = dummy_result(tmp_path, monkeypatch)
    extra = [0.0]
    real_monotonic = time.monotonic
    monkeypatch.setattr(runner, 'time', SimpleNamespace(monotonic=lambda: real_monotonic() + extra[0]))

    def cleanup():
        if damage == 'failure':
            raise OSError('injected source cleanup failure')
        extra[0] += 10.0

    assert runner.supervise(tmp_path, dummy_command(tmp_path, result, state), limit=5, grace=1,
                            preparation_seconds=0.5, cleanup=cleanup) == 1
    attempt = tmp_path / runner.ATTEMPT
    assert not (attempt / 'completion.json').exists()
    assert runner.read_json(attempt / 'attempt.json')['qualifies'] is False


@pytest.fixture(scope='module')
def spark_fixture():
    from airline.common.spark_session import create_spark_session
    spark = create_spark_session('gbt-step2-generated-fixtures-only')
    parent = Path(os.environ.get('AIRLINE_TEST_TMP_ROOT', tempfile.gettempdir()))
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='gbt-step2-', dir=parent))
    try:
        yield spark, directory
    finally:
        spark.stop()
        shutil.rmtree(directory)


@pytest.fixture(scope='module')
def generated(spark_fixture):
    from airline.common.lineage import INTEGER_KEYS, with_lineage
    from airline.dws.ml_feature_contract import (CATEGORICAL_COLUMNS, SCHEDULE_NUMERIC, HISTORY_RATES,
                                                HISTORY_COUNTS, HISTORY_MISSING, OUTPUT_COLUMNS, TARGET_COLUMN)
    from airline.ml.preprocessing import build_preprocessing_pipeline
    from pyspark.ml.classification import GBTClassifier
    spark, directory = spark_fixture
    rows = []
    for split, month, count in [('train', 1, 40), *[('validation', m, 10) for m in range(5, 9)]]:
        for i in range(count):
            row = {'flight_date': date(2023 if split == 'train' else 2024, month, 1),
                   'split': split, TARGET_COLUMN: i % 2}
            row.update({c: str(i % 2) for c in CATEGORICAL_COLUMNS})
            row.update({c: float(i % 2 + 1) for c in SCHEDULE_NUMERIC})
            row.update({c: 0.0 for c in ('departure_missing', 'arrival_missing', 'duration_missing')})
            row.update({c: 0.3 for c in HISTORY_RATES})
            row.update({c: 10 for c in HISTORY_COUNTS})
            row.update({c: 0.0 for c in HISTORY_MISSING})
            row.update({k: i // 4 for k in INTEGER_KEYS})
            rows.append(row)
    frame = with_lineage(spark.createDataFrame(rows)).select(*OUTPUT_COLUMNS)
    # Only synthetic preprocessing is fitted, before runner orchestration begins.
    pre = build_preprocessing_pipeline().fit(frame.filter("split = 'train'"))
    params = GBTClassifier(**runner.candidate_settings())
    defaults = {p.name: params.getOrDefault(p) for p in params.params if params.isDefined(p)}
    encoder = pre.stages[5]
    width = sum(s for s in encoder.categorySizes) + 45  # keep + dropLast cancel.
    return spark, directory, frame, pre, defaults, width


@pytest.mark.parametrize('candidate', runner.CANDIDATES)
def test_distributed_runner_fit_reload_threshold_metrics_and_multiplicity(generated, monkeypatch, candidate):
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import GBTClassifier, LogisticRegression
    from airline.ml import model_contract, experiments, evaluation_artifacts, evaluate
    from airline.ml.experiments import best_threshold
    spark, directory, frame, pre, defaults, width = generated
    estimator = GBTClassifier(**runner.candidate_settings(candidate))
    defaults = {p.name: estimator.getOrDefault(p) for p in estimator.params if estimator.isDefined(p)}
    calls, splits, checks = [], [], []
    original_fit = GBTClassifier.fit

    def forbidden(*args, **kwargs):
        raise AssertionError('Preprocessing/baseline fit or production entrypoint called')

    class FrozenPreprocessing:
        def transform(self, dataset):
            splits.append([r.split for r in dataset.select('split').distinct().collect()])
            return pre.transform(dataset)

    def counted_fit(self, prepared):
        calls.append('fit')
        assert prepared.count() == 40 and prepared.rdd.getNumPartitions() == 2
        assert {(r.class_weight, r['count']) for r in prepared.groupBy('class_weight').count().collect()} == {(1.0, 20), (2.0, 20)}
        return original_fit(self, prepared)

    original_validation = evaluation_artifacts.validate_predictions
    def checked(*args):
        checks.append('population')
        return original_validation(*args)

    monkeypatch.setattr(GBTClassifier, 'fit', counted_fit)
    monkeypatch.setattr(Pipeline, 'fit', forbidden)
    monkeypatch.setattr(LogisticRegression, 'fit', forbidden)
    monkeypatch.setattr(model_contract, 'publish_bundle', forbidden)
    monkeypatch.setattr(experiments, 'select', forbidden)
    monkeypatch.setattr(experiments, 'evaluate', forbidden)
    monkeypatch.setattr(evaluate, 'evaluate', forbidden)
    monkeypatch.setattr(evaluation_artifacts, 'validate_predictions', checked)
    original_multiset = runner.same_multiset
    def multiset(*args):
        checks.append('reload')
        return original_multiset(*args)
    monkeypatch.setattr(runner, 'same_multiset', multiset)
    output = directory / candidate
    output.mkdir()
    state = {'fit_started': False}
    result = runner.fit_and_evaluate(spark, frame, FrozenPreprocessing(),
        {'rows': 40, 'min_date': '2023-01-01', 'max_date': '2023-01-01'}, baseline_months(),
        defaults, width, output, state, lambda: None, candidate=candidate)
    assert calls == ['fit'] and splits == [['train'], ['validation']]
    assert checks == ['reload', 'population', 'reload']
    grid = runner.read_json(output / 'thresholds.json')
    assert result['threshold_step'] == best_threshold(grid['rows'])['step'] != 39
    assert result['overall']['rows'] == 40 and set(result['months']) == set(MONTHS)
    assert result['status'] == 'pending_termination' and 'qualifies' not in result['comparison']
    assert not (output / 'completion.json').exists()
    assert result['comparison'] == {k: v for k, v in monthly_comparison(baseline_months(), result['months']).items() if k != 'qualifies'}
    saved = spark.read.parquet(str(output / 'validation-predictions'))
    from pyspark.ml.classification import GBTClassificationModel
    reloaded = GBTClassificationModel.load(str(output / 'classifier'))
    assert result['name'] == candidate and reloaded.getNumTrees == runner.candidate_contract(candidate)[2]
    assert reloaded.getMaxIter() == defaults['maxIter']
    with pytest.raises(ValueError, match='dimensions'):
        runner.validate_candidate(reloaded, {**defaults, 'maxIter': defaults['maxIter'] + 1}, width)
    with pytest.raises(ValueError, match='dimensions'):
        runner.validate_candidate(reloaded, defaults, width + 1)
    with pytest.raises(ValueError, match='parameters/defaults'):
        runner.validate_candidate(reloaded, {**defaults, 'minInfoGain': 0.1}, width)
    validation = frame.filter("split = 'validation'")
    assert original_validation(saved, validation, result['threshold']) == 40
    for changed in (saved.drop('flight_id'), saved.withColumn('prediction', 1 - saved.prediction), saved.unionByName(saved.limit(1))):
        with pytest.raises(ValueError):
            original_validation(changed, validation, result['threshold'])
    with pytest.raises(ValueError, match='score multiset'):
        original_multiset(saved, saved.unionByName(saved.limit(1)))


def test_distributed_capacity_matches_existing_fractional_reference(spark_fixture):
    from studies.lr_v1.diagnose import budget_metrics
    spark, _ = spark_fixture
    rows = [(0.9, 'same', 1), (0.9, 'same', 0), *[(0.1, str(i), 0) for i in range(8)]]
    frame = spark.createDataFrame(rows, ['probability', 'flight_id', 'is_arrival_delayed_15m'])
    actual, _ = budget_metrics(frame, 10, 1)
    for percent in (10, 20):
        reference = expected_budget(rows, percent)
        for key in reference:
            assert actual[str(percent)][key] == reference[key]


def test_failed_fit_is_not_retried_and_reused_frames_are_released(generated, monkeypatch):
    from pyspark.ml.classification import GBTClassifier
    from pyspark.sql import DataFrame
    spark, directory, frame, pre, defaults, width = generated
    calls, released = [], []
    def fail(*args, **kwargs):
        calls.append('fit')
        raise RuntimeError('injected fitting failure')
    original = DataFrame.unpersist
    def unpersist(self, *args, **kwargs):
        released.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(GBTClassifier, 'fit', fail)
    monkeypatch.setattr(DataFrame, 'unpersist', unpersist)
    output = directory / 'failed-fit'
    output.mkdir()
    state = {'fit_started': False}
    with pytest.raises(RuntimeError, match='injected fitting failure'):
        runner.fit_and_evaluate(spark, frame, pre,
            {'rows': 40, 'min_date': '2023-01-01', 'max_date': '2023-01-01'}, baseline_months(),
            defaults, width, output, state, lambda: None)
    assert calls == ['fit'] and state['classifier_fits_started'] == 1
    assert len(released) == 2 and not list(output.iterdir())


@pytest.mark.parametrize('damage', ['test_split', 'date', 'train_rows'])
def test_population_checks_reject_split_date_or_multiplicity_changes_before_fit(generated, damage):
    from pyspark.sql import functions as F
    _, _, frame, _, _, _ = generated
    profile = {'rows': 40, 'min_date': '2023-01-01', 'max_date': '2023-01-01'}
    if damage == 'test_split':
        frame = frame.withColumn('split', F.lit('test'))
    elif damage == 'date':
        frame = frame.withColumn('flight_date', F.lit('2025-01-01').cast('date'))
    else:
        profile['rows'] = 41
    with pytest.raises((ValueError, AssertionError)):
        runner.validate_populations(frame, profile, baseline_months())

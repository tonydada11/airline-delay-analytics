"""One isolated GBT attempt. --execute is Step 3 only; imports never start Spark."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

SOURCE_ROOT = Path(__file__).resolve().parents[2]
NAME = 'gbt-feature-subset-all'
ATTEMPT = Path('data/studies/gbt-improvement-v1/attempt-001')
CANDIDATES = (NAME, 'gbt-all-30iter', 'gbt-all-60iter')
REFERENCE = Path('studies/gbt_v1/evidence')
REFERENCE_HASHES = {
    'completion.json': 'a07dfb207edd9c79c01ce22563ccd298d5aa8f2690f5886289d292a0d2a71b56',
    'results.json': '89a66fae1b905395a8a5f2c88d0613c741fc8a46ff134133a9130e1fb996a007',
    'inputs.json': '28cb48ed058d19661cd627fb1923ce3591285e57f2c9ad0f049202631ddebb16',
    'worker-status.json': '37b34c11b7e3b90b7947761d3ebf5e36fc12fcf109bc04a49e16b60dcee50342',
}
REFERENCE_30 = Path('studies/gbt_v1/evidence_30iter')
REFERENCE_30_HASHES = {
    'completion.json': '7ec0129ed00dd93887d2d9ded749496404e276a45622160b7a73e249fe57d12c',
    'results.json': '6c02d060a8b974e89d59ac210410bb048c74720f5212af76918191162675718e',
    'inputs.json': '9cf30477fe48bb72a342409a7f5f3e4e5d8cd8bc125f07e71ba118b8bed884f2',
    'worker-status.json': '69743c0e2cd9a52d87da4e9ba1a16f097d6b446508e1d7a8b9a03aef804b3e34',
}
LIMIT = 14400
GRACE = 60
BUNDLE = 'd0098a9572834c53bf76b04d732a94a6'
RUN = 'production-5936316dd7764ed79b049ae10ee1ca18'
FEATURES = 'data/warehouse/dws/ml_flighttime_v3_selection'
ARTIFACTS = 'artifacts/flighttime-v3'
DIAGNOSTICS = Path('studies/lr_v1/evidence/diagnostics')
HASHES = {
    'bundle': '7e1d0a1a9336dd383da9802db02586e23fcc411a4cd79294ee52c554c6d49ee7',
    'selection': '55f0022c50bf987df48d0506b15dc4bdf8acd57933838deae0708215ea35109e',
    'dataset': '093b15fd2085ba50d005b0fd70a804e81597b8df4c3e2f89a56fef18a799a265',
    'baseline_results': 'c1313197e55e9d9d90793ec3401c3c0d3d61597ca35da94eae5c1f7b8fdee828',
    'raw_manifest': 'fef0d967bbc36d2beb2898701428ca55aa5b0f0a7c9bb0f5ede2ed8ac7852efb',
    'lock': '909ee6549a004c7378493f5c2b1cb16cd366cce6a700e149a5c2d0b228a06b8a',
    'timezone': '60fd42f7a8ba6a1f13f37c42a28e920a2dce2291c3fd5c815047228d2d6fc1a8',
}
PRODUCTION_SHA = '55dd33b3cd4ca61d71432b827cc8f63774928ca49b3a41281d1ee1849f090017'
RUNTIME = {
    'spark.cores.max': '2', 'spark.executor.cores': '1',
    'spark.executor.memory': '640m', 'spark.driver.memory': '768m',
    'spark.sql.shuffle.partitions': '4', 'spark.sql.files.maxPartitionBytes': '67108864',
    'spark.sql.inMemoryColumnarStorage.batchSize': '1000',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest(stream)


def _digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """Atomic updates, only inside the newly reserved attempt."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def candidate_contract(candidate=NAME):
    """Only the three explicitly approved, exclusive attempts are supported."""
    if candidate == NAME:
        return ATTEMPT, 'PLAN.md', 10
    if candidate == 'gbt-all-30iter':
        return Path('data/studies/gbt-improvement-v1/attempt-002'), 'PLAN_30ITER.md', 30
    require(candidate == 'gbt-all-60iter', 'Unknown fixed candidate')
    return Path('data/studies/gbt-improvement-v1/attempt-003'), 'PLAN_60ITER.md', 60


def reference_records(candidate):
    """Fixed saved reference for each iteration study; never choose dynamically."""
    if candidate == 'gbt-all-30iter':
        return REFERENCE, REFERENCE_HASHES, NAME
    require(candidate == 'gbt-all-60iter', 'Unknown reference candidate')
    return REFERENCE_30, REFERENCE_30_HASHES, 'gbt-all-30iter'


def candidate_settings(candidate=NAME):
    from airline.ml.model_contract import GBT_SETTINGS, POSITIVE_CLASS_WEIGHT
    _, _, iterations = candidate_contract(candidate)
    require(GBT_SETTINGS['featureSubsetStrategy'] == 'sqrt' and GBT_SETTINGS['maxIter'] == 10
            and POSITIVE_CLASS_WEIGHT == 2.0,
            'Production GBT/weight contract changed')
    settings = dict(GBT_SETTINGS)
    settings['featureSubsetStrategy'] = 'all'
    settings['maxIter'] = iterations
    return settings


def attempt_path(root, candidate=NAME):
    """Fixed path; reject symlinks in every component, including broken links."""
    root = Path(root).absolute()
    for component in [root, *root.parents]:
        require(not component.is_symlink(), 'Symlink project root is forbidden')
    path = root
    for part in candidate_contract(candidate)[0].parts:
        path = path / part
        require(not path.is_symlink(), 'Symlink output path is forbidden')
    if path.exists():
        raise FileExistsError('Preserve existing attempt; no overwrite, resume, or retry')
    return path


def inspect_reference(root, source_root=SOURCE_ROOT, *, candidate='gbt-all-30iter'):
    """Pin saved reference records, never load/score/refit its classifier."""
    from studies.lr_v1.metrics import MONTHS
    directory, hashes, name = reference_records(candidate)
    label = 'ten-iteration' if name == NAME else '30-iteration'
    identities = {}
    for record, expected in hashes.items():
        published = Path(source_root) / directory / record
        local = Path(root) / candidate_contract(name)[0] / ('completion.json' if record == 'completion.json'
                else str(Path(name) / ('status.json' if record == 'worker-status.json' else record)))
        require(published.is_file() and local.is_file()
                and digest(published) == digest(local) == expected,
                f'Missing/changed {label} reference: {published} / {local}')
        identities[record] = {'published_path': str(published), 'local_path': str(local), 'sha256': expected}
    completion = read_json(Path(source_root) / directory / 'completion.json')
    result = read_json(Path(source_root) / directory / 'results.json')
    inputs = read_json(Path(source_root) / directory / 'inputs.json')
    state = read_json(Path(source_root) / directory / 'worker-status.json')
    require(completion['status'] == 'complete' and completion['exit_code'] == 0
            and completion['classifier_fits_started'] == state['classifier_fits_started'] == 1
            and completion['shutdown_successful'] and state['shutdown_successful']
            and completion['results_sha256'] == hashes['results.json']
            and result['name'] == name and set(result['months']) == set(MONTHS)
            and all(result['checks'].values()) and inputs['settings'] == candidate_settings(name)
            and inputs['baseline_bundle_id'] == BUNDLE,
            f'{label} reference is incomplete/incompatible')
    return identities


def inspect_inputs(root, source_root=SOURCE_ROOT, *, candidate=NAME):
    """No data reconstruction, alternative inputs, or test-period access."""
    from airline.common.config import load_config
    from airline.common.paths import get_project_paths
    from airline.ml.model_contract import code_provenance
    from studies.lr_v1.metrics import MONTHS
    root, source_root = Path(root), Path(source_root)
    paths = get_project_paths(load_config(root / 'conf/project.yaml'))
    require(str(paths.dws_ml_flight_features) == FEATURES and str(paths.artifact_root) == ARTIFACTS,
            'Configured selection input/artifact paths differ from PLAN.md')
    files = {
        'bundle': paths.resolve(paths.artifact_root) / 'models/manifest.json',
        'selection': paths.resolve(paths.artifact_root) / 'experiments' / RUN / 'selection.json',
        'dataset': paths.resolve(paths.dws_ml_flight_features) / '_dataset_manifest.json',
        'baseline_results': source_root / DIAGNOSTICS / 'results.json',
        'raw_manifest': paths.resolve(paths.raw_manifest),
        'lock': root / 'requirements.lock', 'timezone': root / 'conf/airport_timezones.csv',
    }
    for name, path in files.items():
        require(path.is_file() and digest(path) == HASHES[name], f'Missing/changed input: {path}')
    manifest, selection = read_json(files['bundle']), read_json(files['selection'])
    require(manifest['bundle_id'] == BUNDLE and manifest['selection'] == selection
            and manifest['selected_classifier'] == 'lr' and manifest['threshold_step'] == 39,
            'Accepted baseline identity differs from PLAN.md')
    require(code_provenance(root)['production_python_sha256'] == PRODUCTION_SHA,
            'Production source differs from the accepted contract')
    for name in ('baseline', 'validation'):
        path = files['selection'].parent / (name + '.json')
        require(digest(path) == selection[name + '_sha256'], f'Changed selection record: {path}')
    preserved = read_json(source_root / 'studies/lr_v1/evidence/baseline-identities-before.json')['files']
    model_files = {}
    for name in ('preprocessing', 'classifier'):
        directory = files['bundle'].parent / name
        expected = {p: v['sha256'] for p, v in preserved.items()
                    if p.startswith(f'{ARTIFACTS}/models/{name}/')}
        actual = {p.relative_to(root).as_posix(): digest(p)
                  for p in sorted(directory.rglob('*')) if p.is_file()}
        require(bool(expected) and actual == expected, f'Accepted {name} files differ/missing')
        model_files[name] = actual
    gbt_metadata = read_json(files['selection'].parent / 'gbt/metadata/part-00000')
    from airline.ml.model_contract import GBT_SETTINGS
    require(gbt_metadata['paramMap'] == GBT_SETTINGS == selection['estimator_settings']['gbt'],
            'Existing GBT settings differ from source')
    defaults = {**gbt_metadata['defaultParamMap'], **candidate_settings(candidate)}
    baseline = read_json(files['baseline_results'])
    require(baseline['status'] == 'complete' and baseline['baseline_bundle_id'] == BUNDLE
            and baseline['threshold'] == 0.39 and set(baseline['validation_months']) == set(MONTHS),
            'Frozen monthly baseline is incompatible/incomplete')
    for month in MONTHS:
        require(read_json(source_root / DIAGNOSTICS / f'validation-{month}.json') == baseline['validation_months'][month],
                f'Frozen month disagrees with aggregate: {month}')
    require((paths.resolve(paths.dws_ml_flight_features) / '_SUCCESS').is_file(),
            'Selection Parquet completion marker missing')
    expected_parquet = {p[len(FEATURES) + 1:]: v['sha256'] for p, v in preserved.items()
                        if p.startswith(FEATURES + '/') and (p.endswith('.parquet') or p.endswith('/_SUCCESS'))}
    require(bool(expected_parquet), 'Frozen selection file inventory missing')
    identities = {
        'small_files': {name: {'path': str(path), 'sha256': digest(path)} for name, path in files.items()},
        'accepted_model_files': model_files,
        'expected_selection_parquet': expected_parquet,
    }
    if candidate == 'gbt-all-30iter':
        identities['ten_iteration_reference'] = inspect_reference(root, source_root)
    elif candidate == 'gbt-all-60iter':
        identities['thirty_iteration_reference'] = inspect_reference(root, source_root, candidate=candidate)
    return paths, manifest, baseline['validation_months'], defaults, identities


def validate_candidate(model, expected_parameters, width):
    from pyspark.ml.classification import GBTClassificationModel
    require(isinstance(model, GBTClassificationModel), 'Candidate is not a GBT model')
    require(model.numClasses == 2 and model.numFeatures == width
            and model.getNumTrees == expected_parameters['maxIter'], 'Candidate class/feature/tree dimensions differ')
    actual = {p.name: model.getOrDefault(p) for p in model.params if model.isDefined(p)}
    require(actual == expected_parameters, 'Candidate effective parameters/defaults differ')


def same_multiset(left, right):
    require(not left.exceptAll(right).limit(1).count()
            and not right.exceptAll(left).limit(1).count(), 'Save/reload score multiset differs')


def validate_populations(dataset, train_profile, baseline):
    from pyspark.sql import functions as F
    from airline.dws.build_ml_flight_features import validate_feature_frame
    from studies.lr_v1.metrics import MONTHS
    require(not dataset.filter(~F.col('split').isin('train', 'validation')).limit(1).count(),
            'Only train/validation selection rows are permitted')
    training, validation = dataset.filter("split = 'train'"), dataset.filter("split = 'validation'")
    rows = validate_feature_frame(training, train_profile['rows'])
    validate_feature_frame(validation, sum(baseline[m]['rows'] for m in MONTHS))
    profile = training.agg(F.min('flight_date').alias('start'), F.max('flight_date').alias('end')).first()
    require(str(profile.start) == train_profile['min_date'] and str(profile.end) == train_profile['max_date'],
            'Training date extent differs')
    # Small summaries only. No joins, unique-key assumptions, or collected flight tables.
    months = validation.groupBy(F.date_format('flight_date', 'yyyy-MM').alias('month')).agg(
        F.count('*').alias('rows'), F.sum('is_arrival_delayed_15m').alias('positives')).collect()
    require({r.month for r in months} == set(MONTHS), 'All four validation months are required')
    for row in months:
        expected = baseline[row.month]
        cm = expected['confusion_matrix']
        require(row.rows == expected['rows'] and row.positives == cm['true_positive'] + cm['false_negative'],
                f'Validation population differs in {row.month}')
    return training, validation, rows


def fit_and_evaluate(spark, dataset, preprocessing, train_profile, baseline, expected_parameters,
                     width, output, state, save_state, *, candidate=NAME):
    """One distributed classifier fit; callable on generated fixtures in tests."""
    from pyspark import StorageLevel
    from pyspark.ml.classification import GBTClassifier, GBTClassificationModel
    from pyspark.sql import functions as F
    from airline.common.lineage import BUSINESS_KEY
    from airline.dws.ml_feature_contract import TARGET_COLUMN
    from airline.ml.model_contract import PREDICTION_COLUMNS, POSITIVE_CLASS_WEIGHT
    from airline.ml.experiments import continuous_score, threshold_counts, best_threshold
    from airline.ml.metrics import calculate_metrics
    from airline.ml.evaluation_artifacts import validate_predictions, inventory
    from studies.lr_v1.diagnose import budget_metrics
    from studies.lr_v1.metrics import MONTHS, monthly_comparison
    training, validation, rows = validate_populations(dataset, train_profile, baseline)
    prepared = validation_prepared = scored = None
    try:
        prepared = (preprocessing.transform(training)
                    .withColumn('class_weight', F.when(F.col(TARGET_COLUMN) == 1, POSITIVE_CLASS_WEIGHT).otherwise(1.0))
                    .select('features', TARGET_COLUMN, 'class_weight')
                    .repartition(max(2, math.ceil(rows / 250000))).persist(StorageLevel.DISK_ONLY))
        validation_prepared = preprocessing.transform(validation).select(
            'flight_id', *BUSINESS_KEY, 'split', TARGET_COLUMN, 'features').persist(StorageLevel.DISK_ONLY)
        require(prepared.count() == rows and validation_prepared.count() == sum(b['rows'] for b in baseline.values()),
                'Prepared population changed')
        estimator = GBTClassifier(**candidate_settings(candidate))
        require({p.name: estimator.getOrDefault(p) for p in estimator.params if estimator.isDefined(p)} == expected_parameters,
                'Estimator effective defaults differ from frozen GBT')
        require(not state['fit_started'], 'A second fit is forbidden')
        state.update(stage='fitting', fit_started=True, classifier_fits_started=1, fit_started_at=utc())
        save_state()
        started = time.monotonic()
        classifier = estimator.fit(prepared)
        state['fit_seconds'] = time.monotonic() - started
        validate_candidate(classifier, expected_parameters, width)
        classifier.write().save(str(output / 'classifier'))
        reloaded = GBTClassificationModel.load(str(output / 'classifier'))
        validate_candidate(reloaded, expected_parameters, width)
        state.update(stage='validating', fit_finished_at=utc())
        save_state()
        prepared.unpersist()
        prepared = None
        before = continuous_score(classifier, validation_prepared).select('flight_id', TARGET_COLUMN, 'probability')
        scored = continuous_score(reloaded, validation_prepared).select(
            'flight_id', *BUSINESS_KEY, 'split', TARGET_COLUMN, 'probability').persist(StorageLevel.DISK_ONLY)
        after = scored.select('flight_id', TARGET_COLUMN, 'probability')
        same_multiset(before, after)
        # One complete validation threshold grid, independent of the accepted LR threshold.
        grid = threshold_counts(after)
        require(sorted(r['step'] for r in grid) == list(range(101)), 'Incomplete threshold grid')
        selected = best_threshold(grid)
        threshold = selected['step'] / 100
        write_json(output / 'thresholds.json', {'threshold_denominator': 100, 'rows': grid, 'selected': selected})
        predictions = (scored.withColumn('prediction', (F.col('probability') >= threshold).cast('int'))
                       .select(*PREDICTION_COLUMNS))
        predictions.write.parquet(str(output / 'validation-predictions'))
        saved = spark.read.parquet(str(output / 'validation-predictions'))
        validate_predictions(saved, validation, threshold)
        same_multiset(after, saved.select('flight_id', TARGET_COLUMN, 'probability'))
        scored.unpersist()
        scored = None
        validation_prepared.unpersist()
        validation_prepared = None
        overall = calculate_metrics(saved, threshold)
        months = {}
        for month in MONTHS:
            frame = saved.filter(F.date_format('flight_date', 'yyyy-MM') == month).persist(StorageLevel.DISK_ONLY)
            try:
                metrics = calculate_metrics(frame, threshold)
                cm = metrics['confusion_matrix']
                budgets, duplicates = budget_metrics(frame, metrics['rows'], cm['true_positive'] + cm['false_negative'])
                changes = {k: metrics[k] - baseline[month][k] for k in
                           ('pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'prevalence', 'predicted_positive_rate')}
                months[month] = {**metrics, 'budgets': budgets, 'duplicate_score_id_check': duplicates,
                                'changes': {**changes,
                                            'budgets': {p: {k: budgets[p][k] - baseline[month]['budgets'][p][k]
                                                               for k in ('precision', 'recall')} for p in ('10', '20')}}}
                write_json(output / f'validation-{month}.json', months[month])
            finally:
                frame.unpersist()
        comparison = monthly_comparison(baseline, months)
        # Qualification is published only by the supervisor after successful termination.
        comparison.pop('qualifies')
        return {
            'name': candidate, 'status': 'pending_termination', 'overall': overall,
            'threshold_step': selected['step'], 'threshold': threshold,
            'months': months, 'comparison': comparison,
            'equal_month_means': {k: sum(months[m][k] for m in MONTHS) / 4
                                 for k in ('pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'prevalence', 'predicted_positive_rate')},
            'worst_monthly_changes': {k: min(months[m]['changes'][k] for m in MONTHS)
                                      for k in ('pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'prevalence', 'predicted_positive_rate')},
            'equal_month_budget_means': {p: {k: sum(months[m]['budgets'][p][k] for m in MONTHS) / 4
                                             for k in ('precision', 'recall')} for p in ('10', '20')},
            'worst_monthly_budget_changes': {p: {k: min(months[m]['changes']['budgets'][p][k] for m in MONTHS)
                                                 for k in ('precision', 'recall')} for p in ('10', '20')},
            'checks': {'parameters_defaults_dimensions': True, 'full_validation_score_reload': True,
                       'saved_prediction_schema_population_threshold': True, 'all_months': True},
            'artifacts': {'classifier': inventory(output / 'classifier'),
                          'predictions': inventory(output / 'validation-predictions', parquet_only=True)},
        }
    finally:
        if prepared is not None:
            prepared.unpersist()
        if validation_prepared is not None:
            validation_prepared.unpersist()
        if scored is not None:
            scored.unpersist()


def utc():
    return datetime.now(timezone.utc).isoformat()


def worker(root, token, *, candidate=NAME):
    attempt = Path(root) / candidate_contract(candidate)[0]
    require(os.environ.get('AIRLINE_GBT_WORKER_TOKEN') == token == read_json(attempt / 'attempt.json')['token'],
            'Worker requires the bounded supervisor')
    state = {'stage': 'preparing', 'fit_started': False, 'classifier_fits_started': 0,
             'started_at': utc(), 'shutdown_successful': False}
    output = attempt / candidate
    output.mkdir()  # Parent is owned by the supervisor, candidate directory is exclusive too.
    save_state = lambda: write_json(output / 'status.json', state)
    spark = dataset = None
    started = time.monotonic()

    def interrupted(signum, frame):
        raise InterruptedError(f'Worker received signal {signum}')

    handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    save_state()
    try:
        paths, manifest, baseline, defaults, identities = inspect_inputs(root, candidate=candidate)
        from airline.ml.model_contract import load_bundle, runtime_provenance
        from airline.common.spark_session import create_spark_session
        from airline.dws.dataset_manifest import validate_dataset_manifest
        from airline.ml.evaluation_artifacts import inventory
        from pyspark import SparkContext
        from pyspark import StorageLevel
        require(SparkContext._active_spark_context is None, 'Refuse to reuse an unrelated Spark application')
        spark = create_spark_session('airline-study-' + candidate)
        spark.sparkContext.setLogLevel('WARN')
        runtime = {k: (spark.conf.get(k) if k.startswith('spark.sql.') else spark.sparkContext.getConf().get(k))
                   for k in RUNTIME}
        state.update(runtime=runtime, spark_version=spark.version, spark_application_id=spark.sparkContext.applicationId,
                     settings=candidate_settings(candidate), positive_class_weight=2.0, negative_class_weight=1.0)
        save_state()
        require(runtime == RUNTIME and spark.version == '3.5.9', 'Runtime resources/version differ from PLAN.md')
        require(runtime_provenance(root) == manifest['runtime_provenance'], 'Pinned Python/architecture/lock changed')
        preprocessing, accepted, loaded = load_bundle(paths.resolve(paths.artifact_root) / 'models', HASHES['raw_manifest'])
        require(loaded == manifest and preprocessing.uid == 'PipelineModel_196025fd5a4b'
                and accepted.numFeatures == 784, 'Accepted preprocessing/dimensions changed')
        source = paths.resolve(paths.dws_ml_flight_features)
        identities['selection_parquet'] = inventory(source, parquet_only=True)
        require(identities['selection_parquet'] == identities['expected_selection_parquet'],
                'Selection Parquet differs from the frozen accepted file inventory')
        dataset = spark.read.parquet(str(source)).persist(StorageLevel.DISK_ONLY)
        validate_dataset_manifest(source, dataset, HASHES['raw_manifest'], phase='selection')
        write_json(output / 'inputs.json', {'identities': identities, 'runtime': runtime,
                   'runtime_provenance': runtime_provenance(root), 'spark_application_id': spark.sparkContext.applicationId,
                   'baseline_bundle_id': BUNDLE, 'settings': candidate_settings(candidate),
                   'effective_parameters': defaults, 'positive_class_weight': 2.0, 'negative_class_weight': 1.0})
        result = fit_and_evaluate(spark, dataset, preprocessing, manifest['training'], baseline,
                                  defaults, 784, output, state, save_state, candidate=candidate)
        require(inspect_inputs(root, candidate=candidate)[-1] == {k: v for k, v in identities.items() if k != 'selection_parquet'}
                and inventory(source, parquet_only=True) == identities['selection_parquet'], 'Inputs changed during attempt')
        result['checks']['inputs_unchanged'] = True
        write_json(output / 'results.json', result)
        state['stage'] = 'evaluated'
    except BaseException as exc:
        state.update(stage='incomplete', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        try:
            try:
                if dataset is not None:
                    dataset.unpersist()
            finally:
                if spark is not None:
                    try:
                        spark.sparkContext.cancelAllJobs()  # Only this fresh process's application.
                    finally:
                        spark.stop()
            state['shutdown_successful'] = True
        except BaseException as exc:
            state.update(stage='incomplete', cleanup_error=repr(exc), cleanup_traceback=traceback.format_exc())
            raise
        finally:
            state.update(elapsed_seconds=time.monotonic() - started, finished_at=utc())
            save_state()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def supervise(root, command=None, *, limit=LIMIT, grace=GRACE, source_revision=None,
              preparation_seconds=0.0, cleanup=None, candidate=NAME):
    """Bound the actual worker process group, including blocking JVM/Py4J calls."""
    started = time.monotonic()
    require(0 < grace < limit, 'Invalid timeout allowance')
    require(math.isfinite(preparation_seconds) and 0 <= preparation_seconds < limit - grace,
            'Preparation exhausted the execution allowance')
    remaining = limit - preparation_seconds
    attempt = attempt_path(root, candidate)
    attempt.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex
    command = command or [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', token,
                          '--project-root', str(root), '--candidate', candidate]
    record = {'name': candidate, 'token': token, 'status': 'running', 'qualifies': False,
              'started_at': utc(), 'limit_seconds': limit, 'termination_allowance_seconds': grace,
              'preparation_seconds': preparation_seconds,
              'command': command, 'source_revision': source_revision,
              'runner_sha256': digest(__file__),
              'plan_sha256': digest(Path(__file__).with_name(candidate_contract(candidate)[1])),
              'automatic_retries': False, 'maximum_classifier_fits': 1}
    write_json(attempt / 'attempt.json', record)
    process = None
    handlers = {}

    def interrupted(signum, frame):
        raise InterruptedError(f'Supervisor received signal {signum}')

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, interrupted)
        environment = dict(os.environ, AIRLINE_GBT_WORKER_TOKEN=token, PYTHONDONTWRITEBYTECODE='1')
        with (attempt / 'worker.log').open('x') as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            process.wait(timeout=max(0, remaining - grace - (time.monotonic() - started)))
        require(process.returncode == 0, f'Worker exited {process.returncode}')
        require(time.monotonic() - started < remaining - grace, 'Worker exhausted execution allowance')
        state = read_json(attempt / candidate / 'status.json')
        result = read_json(attempt / candidate / 'results.json')
        require(state['stage'] == 'evaluated' and state['shutdown_successful']
                and state['classifier_fits_started'] == 1 and state['fit_started']
                and result['status'] == 'pending_termination' and result['name'] == candidate
                and set(result['checks']) == {'parameters_defaults_dimensions', 'full_validation_score_reload',
                    'saved_prediction_schema_population_threshold', 'all_months', 'inputs_unchanged'}
                and all(result['checks'].values()), 'Worker incomplete or artifact checks failed')
        # Re-read the frozen baseline, never test reports, after the worker has terminated.
        from studies.lr_v1.metrics import monthly_comparison
        baseline_path = SOURCE_ROOT / DIAGNOSTICS / 'results.json'
        require(digest(baseline_path) == HASHES['baseline_results'], 'Baseline monthly record changed')
        comparison = monthly_comparison(read_json(baseline_path)['validation_months'], result['months'])
        record.update(status='complete', comparison=comparison, qualifies=comparison['qualifies'],
                      results_sha256=digest(attempt / candidate / 'results.json'), fit_seconds=state['fit_seconds'])
    except BaseException as exc:
        record.update(status='incomplete', qualifies=False, error=repr(exc), traceback=traceback.format_exc())
    finally:
        # Signals during cleanup must not skip the hard-kill phase or publish completion.
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=max(0, remaining - (time.monotonic() - started) - 1))
            except subprocess.TimeoutExpired:
                record.update(status='incomplete', qualifies=False, error='Worker termination timed out')
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            record['exit_code'] = process.returncode
        state_path = attempt / candidate / 'status.json'
        if state_path.is_file():
            try:
                state = read_json(state_path)
                record.update(fit_started=state.get('fit_started'), classifier_fits_started=state.get('classifier_fits_started'),
                              fit_seconds=state.get('fit_seconds'), shutdown_successful=state.get('shutdown_successful'))
            except (OSError, ValueError):
                record['worker_status_unreadable'] = True
        if cleanup is not None:
            try:
                cleanup()  # Container source cleanup is charged before completion.
            except BaseException as exc:
                record.update(status='incomplete', qualifies=False, cleanup_error=repr(exc))
        record.update(total_seconds=preparation_seconds + (time.monotonic() - started), finished_at=utc())
        if record['total_seconds'] >= limit:
            record.update(status='incomplete', qualifies=False, error='Wall-clock ceiling exceeded')
        write_json(attempt / 'attempt.json', record)
        if record['status'] == 'complete':
            write_json(attempt / 'completion.json', record)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return 0 if record['status'] == 'complete' else 1


def main(argv=None, *, cleanup=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--execute', action='store_true', help='Step 3 only: reserve and run the single bounded attempt')
    action.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--candidate', choices=CANDIDATES, default=NAME,
                        help='Explicit fixed candidate; default preserves attempt-001')
    parser.add_argument('--project-root', type=Path, default=SOURCE_ROOT)
    parser.add_argument('--source-revision', default=None)
    parser.add_argument('--preparation-seconds', type=float, default=0.0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.project_root, args.worker, candidate=args.candidate)
        return 0
    revision = args.source_revision or subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=args.project_root, text=True).strip()
    return supervise(args.project_root, source_revision=revision, cleanup=cleanup, candidate=args.candidate,
                     preparation_seconds=args.preparation_seconds + (time.monotonic() - started))


if __name__ == '__main__':
    sys.exit(main())

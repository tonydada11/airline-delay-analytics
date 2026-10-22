"""Fit one approved experimental LR once; never publish a production bundle."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import traceback

from studies.lr_v1.candidate_spec import validate_approval, exclusive_candidate_path
from studies.lr_v1.diagnose import budget_metrics
from studies.lr_v1.metrics import MONTHS, monthly_comparison
from studies.lr_v1.preserve import digest

STUDY = Path(__file__).resolve().parents[3]
ROOT = STUDY.parents[2]
TARGET = 'is_arrival_delayed_15m'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def progress(message):
    print(f'[{datetime.now(timezone.utc).isoformat()}] {message}', flush=True)


def main(name):
    from pyspark import StorageLevel
    from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
    from pyspark.sql import functions as F
    from airline.common.spark_session import create_spark_session
    from airline.common.lineage import BUSINESS_KEY
    from airline.dws.dataset_manifest import validate_dataset_manifest
    from airline.ml.model_contract import load_bundle, PREDICTION_COLUMNS
    from airline.ml.experiments import continuous_score
    from airline.ml.evaluation_artifacts import validate_predictions
    from airline.ml.metrics import calculate_metrics

    approval_path = Path(__file__).with_name('approved-candidates.json')
    approval = json.loads(approval_path.read_text())
    candidates = validate_approval(approval)
    spec = next(c for c in candidates if c['name'] == name)
    baseline_file = STUDY / 'diagnostics/results.json'
    assert digest(baseline_file) == approval['baseline_results_sha256']
    assert digest(Path(__file__).with_name('protocol.json')) == approval['protocol_sha256']
    baseline = json.loads(baseline_file.read_text())['validation_months']
    output = exclusive_candidate_path(STUDY, name)
    output.mkdir(parents=True)
    started = time.monotonic()
    status = {'name': name, 'status': 'preparing', 'fit_started': False,
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'settings': spec['settings']}
    write_json(output / 'status.json', status)
    spark, prepared, validation_prepared = None, None, None
    try:
        spark = create_spark_session('airline-study-' + name)
        spark.sparkContext.setLogLevel('WARN')
        runtime = {key: spark.sparkContext.getConf().get(key) for key in
                   ('spark.cores.max', 'spark.executor.cores', 'spark.executor.memory', 'spark.driver.memory')}
        runtime.update({key: spark.conf.get(key) for key in
                        ('spark.sql.shuffle.partitions', 'spark.sql.inMemoryColumnarStorage.batchSize')})
        assert runtime == {'spark.cores.max': '2', 'spark.executor.cores': '1',
                           'spark.executor.memory': '640m', 'spark.driver.memory': '768m',
                           'spark.sql.shuffle.partitions': '4', 'spark.sql.inMemoryColumnarStorage.batchSize': '1000'}, runtime
        artifact = ROOT / 'artifacts/flighttime-v3'
        fingerprint = digest(ROOT / 'conf/raw_data_manifest.csv')
        preprocessing, accepted_classifier, manifest = load_bundle(artifact / 'models', fingerprint)
        assert manifest['bundle_id'] == approval['baseline_bundle_id'] and manifest['threshold_step'] == 39
        source = ROOT / 'data/warehouse/dws/ml_flighttime_v3_selection'
        features = spark.read.parquet(str(source))
        validate_dataset_manifest(source, features, fingerprint, phase='selection')
        assert digest(source / '_dataset_manifest.json') == manifest['selection']['dataset_manifest_sha256']
        assert not features.filter(~F.col('split').isin('train', 'validation')).limit(1).count()
        training = features.filter("split = 'train'")
        validation = features.filter("split = 'validation'")
        profile = training.agg(F.count('*').alias('rows'), F.sum(TARGET).alias('positives'),
                               F.min('flight_date').alias('start'), F.max('flight_date').alias('end')).first()
        assert profile.rows == 8946370 and str(profile.start) == '2023-01-01' and str(profile.end) == '2024-04-30'
        assert not validation.filter((F.col('flight_date') < '2024-05-01') | (F.col('flight_date') > '2024-08-31')).limit(1).count()
        progress(name + ': materializing all training rows with accepted frozen preprocessing')
        settings = spec['settings']
        prepared = (preprocessing.transform(training)
            .withColumn('class_weight', F.when(F.col(TARGET) == 1, settings['positive_class_weight']).otherwise(1.0))
            .select('features', TARGET, 'class_weight')
            .repartition(max(2, math.ceil(profile.rows / 250000))).persist(StorageLevel.DISK_ONLY))
        assert prepared.count() == profile.rows
        validation_prepared = preprocessing.transform(validation).select(
            'flight_id', *BUSINESS_KEY, 'split', TARGET, 'features').persist(StorageLevel.DISK_ONLY)
        assert validation_prepared.count() == sum(baseline[m]['rows'] for m in MONTHS)
        estimator = LogisticRegression(**settings['classifier_settings'])
        for key, value in settings['classifier_settings'].items():
            assert estimator.getOrDefault(key) == value, key
        status.update(status='fitting', fit_started=True, fit_started_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / 'status.json', status)
        progress(name + ': starting the single approved classifier fit')
        fit_start = time.monotonic()
        classifier = estimator.fit(prepared)
        fit_seconds = time.monotonic() - fit_start
        status.update(status='evaluating', fit_elapsed_seconds=fit_seconds)
        write_json(output / 'status.json', status)
        progress(name + f': fit finished in {fit_seconds:.1f}s; saving and reloading experimental classifier')
        classifier.write().save(str(output / 'classifier'))
        reloaded = LogisticRegressionModel.load(str(output / 'classifier'))
        assert reloaded.numClasses == 2 and reloaded.numFeatures == accepted_classifier.numFeatures == 784
        for key, value in settings['classifier_settings'].items():
            assert reloaded.getOrDefault(key) == value, key
        training_info = {'rows': profile.rows, 'positive_rows': profile.positives,
                         'start': str(profile.start), 'end': str(profile.end),
                         'optimizer_iterations': classifier.summary.totalIterations,
                         'objective_history': classifier.summary.objectiveHistory}
        prepared.unpersist()
        prepared = None

        def predictions(model):
            return (continuous_score(model, validation_prepared)
                .withColumn('prediction', (F.col('probability') >= 0.39).cast('int'))
                .select(*PREDICTION_COLUMNS))

        predictions(reloaded).write.parquet(str(output / 'validation-predictions'))
        saved = spark.read.parquet(str(output / 'validation-predictions'))
        assert validate_predictions(saved, validation, 0.39) == 2418739
        original = predictions(classifier)
        assert not original.exceptAll(saved).limit(1).count()
        assert not saved.exceptAll(original).limit(1).count()
        validation_prepared.unpersist()
        validation_prepared = None
        months = {}
        for month in MONTHS:
            progress(name + ': full monthly evaluation ' + month)
            frame = saved.filter(F.date_format('flight_date', 'yyyy-MM') == month).persist(StorageLevel.DISK_ONLY)
            try:
                metrics = calculate_metrics(frame, 0.39)
                positives = metrics['confusion_matrix']['true_positive'] + metrics['confusion_matrix']['false_negative']
                expected_positives = baseline[month]['confusion_matrix']['true_positive'] + baseline[month]['confusion_matrix']['false_negative']
                assert metrics['rows'] == baseline[month]['rows'] and positives == expected_positives
                assert metrics['pr_auc'] is not None
                budgets, duplicates = budget_metrics(frame, metrics['rows'], positives)
                changes = {'pr_auc': metrics['pr_auc'] - baseline[month]['pr_auc'],
                           'budgets': {p: {k: budgets[p][k] - baseline[month]['budgets'][p][k]
                                           for k in ('precision', 'recall')} for p in ('10', '20')}}
                months[month] = {**metrics, 'budgets': budgets, 'duplicate_score_id_check': duplicates,
                                 'differences_from_baseline': changes}
                write_json(output / ('validation-' + month + '.json'), months[month])
            finally:
                frame.unpersist()
        comparison = monthly_comparison(baseline, months)
        assert digest(baseline_file) == approval['baseline_results_sha256']
        result = {'name': name, 'order': spec['order'], 'status': 'complete', 'settings': settings,
                  'changed_factor': spec['changed_factor'], 'months': months,
                  'mean_monthly_PR_AUC': sum(months[m]['pr_auc'] for m in MONTHS) / len(MONTHS),
                  'mean_monthly_budget_metrics': {p: {k: sum(months[m]['budgets'][p][k] for m in MONTHS) / len(MONTHS)
                                                      for k in ('precision', 'recall')} for p in ('10', '20')},
                  'comparison': comparison, 'training': training_info, 'runtime_configuration': runtime,
                  'fit_elapsed_seconds': fit_seconds, 'baseline_bundle_id': manifest['bundle_id'],
                  'baseline_results_sha256': approval['baseline_results_sha256'],
                  'dataset_manifest_sha256': manifest['selection']['dataset_manifest_sha256'],
                  'accepted_preprocessing_reused_without_fitting': True,
                  'reload_parameters_dimensions_and_full_validation_score_multiset': 'passed',
                  'validation_prediction_population_and_threshold': 'passed',
                  'classifier_fits_started': 1, 'historical_test_accessed_for_analysis': False,
                  'production_publication': False}
        write_json(output / 'results.json', result)
        status['status'] = 'complete'
        progress(name + ': all four months complete; qualification=' + str(comparison['qualifies']))
    except BaseException as exc:
        status.update(status='incomplete', error=repr(exc))
        traceback.print_exc()
        raise
    finally:
        if prepared is not None:
            prepared.unpersist()
        if validation_prepared is not None:
            validation_prepared.unpersist()
        if spark is not None:
            spark.stop()
        status['elapsed_seconds'] = time.monotonic() - started
        status['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'status.json', status)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Supply exactly one approved candidate name')
    main(sys.argv[1])

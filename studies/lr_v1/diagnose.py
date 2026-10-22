"""Read-only accepted-model diagnostics. This entry point never fits a model."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

from studies.lr_v1.metrics import MONTHS, capacity, expected_budget

TARGET = 'is_arrival_delayed_15m'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def progress(message):
    print(f'[{datetime.now(timezone.utc).isoformat()}] {message}', flush=True)


def budget_metrics(frame, rows, positives):
    from pyspark.sql import functions as F, Window
    groups = frame.groupBy('probability', 'flight_id').agg(
        F.count('*').alias('n'), F.sum(TARGET).alias('positive_rows'))
    ordering = Window.orderBy(F.desc('probability'), F.asc('flight_id')).rowsBetween(Window.unboundedPreceding, -1)
    groups = groups.withColumn('before', F.coalesce(F.sum('n').over(ordering), F.lit(0)))
    expressions = [F.sum(F.when(F.col('n') > 1, 1).otherwise(0)).alias('duplicate_groups'),
                   F.sum(F.col('n') - 1).alias('duplicate_excess_rows')]
    for percent in (10, 20):
        take = F.greatest(F.lit(0), F.least(F.col('n'), F.lit(capacity(rows, percent)) - F.col('before')))
        expressions.extend([
            F.sum(take).alias(f'selected_{percent}'),
            F.sum(take / F.col('n') * F.col('positive_rows')).alias(f'tp_{percent}'),
            F.sum(F.when((take > 0) & (take < F.col('n')), 1).otherwise(0)).alias(f'fractional_{percent}'),
            F.min(F.when(take > 0, F.col('probability'))).alias(f'cutoff_{percent}')])
    row = groups.agg(*expressions).first().asDict()
    result = {}
    for percent in (10, 20):
        selected, tp = int(row[f'selected_{percent}'] or 0), float(row[f'tp_{percent}'] or 0)
        assert selected == capacity(rows, percent)
        result[str(percent)] = {'capacity': selected, 'selected_mass': selected,
            'true_positive_mass': tp, 'precision': tp / selected if selected else 0.0,
            'recall': tp / positives if positives else 0.0,
            'minimum_selected_score': row[f'cutoff_{percent}'],
            'fractional_boundary_groups': int(row[f'fractional_{percent}'] or 0)}
    return result, {'identical_score_id_groups': int(row['duplicate_groups'] or 0),
                    'duplicate_excess_rows': int(row['duplicate_excess_rows'] or 0)}


def distribution(frame):
    from pyspark.sql import functions as F
    quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    stats = frame.groupBy(TARGET).agg(F.count('*').alias('rows'),
        F.mean('probability').alias('mean'), F.stddev_pop('probability').alias('stddev'),
        F.min('probability').alias('min'), F.max('probability').alias('max'),
        F.percentile_approx('probability', quantiles, 10000).alias('quantiles')).collect()
    bins = (frame.withColumn('bin', F.least(F.lit(19), F.floor(F.col('probability') * 20)))
        .groupBy(TARGET, 'bin').count().orderBy(TARGET, 'bin').collect())
    return {'quantile_probabilities': quantiles, 'quantile_accuracy': 10000,
            'by_label': [r.asDict() for r in stats],
            'histogram_bin_width': 0.05, 'histogram': [r.asDict() for r in bins]}


def threshold_report(frame):
    from airline.ml.experiments import threshold_counts
    from airline.ml.metrics import confusion_metrics
    return [dict(step=r['step'], threshold=r['step'] / 100,
                 **confusion_metrics(r['tp'], r['fp'], r['fn'], r['tn']))
            for r in threshold_counts(frame)]


def feature_profiles(features, preprocessing):
    from pyspark.sql import functions as F
    from airline.dws.ml_feature_contract import NUMERIC_COLUMNS, HISTORY_COUNTS, HISTORY_MISSING, CATEGORICAL_COLUMNS
    numeric = (*NUMERIC_COLUMNS, *HISTORY_COUNTS, *HISTORY_MISSING)
    expressions = [F.count('*').alias('rows'), F.sum(TARGET).alias('positive_rows')]
    for index, name in enumerate(numeric):
        expressions.extend([F.mean(name).alias(f'mean_{index}'), F.stddev_pop(name).alias(f'stddev_{index}'),
                            F.min(name).alias(f'min_{index}'), F.max(name).alias(f'max_{index}'),
                            F.sum(F.when(F.col(name).isNull(), 1).otherwise(0)).alias(f'null_{index}')])
    monthly = []
    for row in features.groupBy('split', 'month').agg(*expressions).orderBy('month').collect():
        data = row.asDict()
        monthly.append({'month': data['month'], 'split': data['split'], 'rows': data['rows'],
                        'positive_rows': data['positive_rows'], 'prevalence': data['positive_rows'] / data['rows'],
                        'numeric': {name: {'mean': data[f'mean_{i}'], 'stddev': data[f'stddev_{i}'],
                                  'min': data[f'min_{i}'], 'max': data[f'max_{i}'],
                                  'null_fraction': data[f'null_{i}'] / data['rows']} for i, name in enumerate(numeric)}})
    categorical = {}
    for index, name in enumerate(CATEGORICAL_COLUMNS):
        known = set(preprocessing.stages[index].labels)
        grouped = {}
        for row in features.groupBy('split', 'month', name).count().collect():
            grouped.setdefault((row['split'], row['month']), []).append({'value': row[name], 'rows': row['count']})
        categorical[name] = []
        for (split, month), values in sorted(grouped.items()):
            total = sum(v['rows'] for v in values)
            categorical[name].append({'split': split, 'month': month, 'distinct_values': len(values),
                'unseen_in_accepted_training_rows': sum(v['rows'] for v in values if v['value'] not in known),
                'top_values': [dict(v, fraction=v['rows'] / total) for v in sorted(values, key=lambda v: (-v['rows'], str(v['value'])))[:10]]})
    return {'monthly': monthly, 'categorical': categorical}


def main():
    STUDY_ROOT = Path(__file__).resolve().parents[3]
    BASELINE_ROOT = STUDY_ROOT.parents[2]
    OUTPUT = STUDY_ROOT / 'diagnostics'
    from pyspark import StorageLevel
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import LogisticRegressionModel
    from pyspark.sql import functions as F
    from airline.common.spark_session import create_spark_session
    from airline.common.raw_snapshot import sha256_file
    from airline.dws.dataset_manifest import validate_dataset_manifest
    from airline.ml.model_contract import load_bundle
    from airline.ml.evaluate import score
    from airline.ml.evaluation_artifacts import validate_predictions
    from airline.ml.metrics import calculate_metrics

    protocol = json.loads(Path(__file__).with_name('protocol.json').read_text())
    assert protocol['baseline_threshold'] == 0.39
    assert protocol['validation_months'] == list(MONTHS)
    assert not OUTPUT.exists(), 'Diagnostic outputs are exclusive; preserve an interrupted attempt.'
    OUTPUT.mkdir()
    start = time.monotonic()
    status = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(), 'classifier_fits_started': 0}
    write_json(OUTPUT / 'status.json', status)
    spark = None
    try:
        spark = create_spark_session('airline-lr-baseline-diagnostics')
        spark.sparkContext.setLogLevel('WARN')
        # A synthetic fixture verifies the distributed duplicate-boundary logic.
        fixture = [(0.9, 'same', 1), (0.9, 'same', 0), *[(0.1, str(i), 0) for i in range(8)]]
        test_frame = spark.createDataFrame(fixture, ['probability', 'flight_id', TARGET])
        tested, duplicate_test = budget_metrics(test_frame, len(fixture), 1)
        for percent in (10, 20):
            reference = expected_budget(fixture, percent)
            assert all(abs(tested[str(percent)][k] - reference[k]) < 1e-12 for k in ('capacity', 'selected_mass', 'true_positive_mass', 'precision', 'recall', 'fractional_boundary_groups'))
        progress('Synthetic duplicate-boundary check passed; loading accepted models and selection features')
        artifact = BASELINE_ROOT / 'artifacts/flighttime-v3'
        fingerprint = sha256_file(BASELINE_ROOT / 'conf/raw_data_manifest.csv')
        preprocessing, classifier, manifest = load_bundle(artifact / 'models', fingerprint)
        assert manifest['bundle_id'] == protocol['baseline_bundle_id'] and manifest['threshold_step'] == 39
        source = BASELINE_ROOT / 'data/warehouse/dws/ml_flighttime_v3_selection'
        features = spark.read.parquet(str(source))
        validate_dataset_manifest(source, features, fingerprint, phase='selection')
        assert sha256_file(source / '_dataset_manifest.json') == manifest['selection']['dataset_manifest_sha256']
        assert not features.filter(~F.col('split').isin('train', 'validation')).limit(1).count()
        assert not features.filter((F.col('flight_date') < '2023-01-01') | (F.col('flight_date') > '2024-08-31')).limit(1).count()
        validation_features = features.filter("split = 'validation'")
        predictions = spark.read.parquet(str(artifact / 'predictions/validation'))
        validate_predictions(predictions, validation_features, 0.39)
        predictions = predictions.withColumn('month', F.date_format('flight_date', 'yyyy-MM')).persist(StorageLevel.DISK_ONLY)
        assert sorted(r.month for r in predictions.select('month').distinct().collect()) == list(MONTHS)
        months = {}
        for month in MONTHS:
            progress('Evaluating full validation month ' + month)
            frame = predictions.filter(F.col('month') == month).persist(StorageLevel.DISK_ONLY)
            metrics = calculate_metrics(frame, 0.39)
            assert metrics['pr_auc'] is not None
            positives = metrics['confusion_matrix']['true_positive'] + metrics['confusion_matrix']['false_negative']
            budgets, duplicates = budget_metrics(frame, metrics['rows'], positives)
            months[month] = {**metrics, 'budgets': budgets, 'duplicate_score_id_check': duplicates,
                             'score_distribution': distribution(frame), 'thresholds': threshold_report(frame)}
            write_json(OUTPUT / ('validation-' + month + '.json'), months[month])
            frame.unpersist()
        predictions.unpersist()
        progress('Profiling full training/validation feature distributions')
        profile = feature_profiles(features.withColumn('month', F.date_format('flight_date', 'yyyy-MM')), preprocessing)
        write_json(OUTPUT / 'feature-profiles.json', profile)
        progress('Scoring deterministic approximately 1% training sample with the accepted frozen LR; in-sample only')
        training = features.filter("split = 'train'")
        sample = training.filter(F.pmod(F.xxhash64('flight_id'), F.lit(100)) == 0)
        sample_scores = score(preprocessing, classifier, sample, 0.39).withColumn('month', F.date_format('flight_date', 'yyyy-MM')).persist(StorageLevel.DISK_ONLY)
        sample_scores.count()
        train_months = {}
        for month in sorted(r.month for r in sample_scores.select('month').distinct().collect()):
            frame = sample_scores.filter(F.col('month') == month)
            train_months[month] = {**calculate_metrics(frame, 0.39), 'score_distribution': distribution(frame), 'interpretation': 'in-sample diagnostic on deterministic approximately 1% sample; not a validation estimate'}
        train_overall = calculate_metrics(sample_scores, 0.39)
        write_json(OUTPUT / 'training-in-sample.json', {'months': train_months, 'overall': train_overall,
                   'sampling': protocol['training_score_diagnostic']})
        sample_scores.unpersist()
        progress('Checking frozen model serialization/reload on a small training-only fixture')
        reload_root = OUTPUT / 'reload-check'
        preprocessing.write().save(str(reload_root / 'preprocessing'))
        classifier.write().save(str(reload_root / 'classifier'))
        reloaded_pre = PipelineModel.load(str(reload_root / 'preprocessing'))
        reloaded_classifier = LogisticRegressionModel.load(str(reload_root / 'classifier'))
        check = sample.limit(1000).persist(StorageLevel.DISK_ONLY)
        check.count()
        before = score(preprocessing, classifier, check, 0.39)
        after = score(reloaded_pre, reloaded_classifier, check, 0.39)
        assert not before.exceptAll(after).limit(1).count() and not after.exceptAll(before).limit(1).count()
        check.unpersist()
        coefficients = classifier.coefficients.toArray()
        result = {'status': 'complete', 'baseline_bundle_id': manifest['bundle_id'], 'threshold': 0.39,
            'protocol_sha256': sha256_file(Path(__file__).with_name('protocol.json')),
            'validation_months': months,
            'mean_monthly_PR_AUC': sum(months[m]['pr_auc'] for m in MONTHS) / len(MONTHS),
            'mean_monthly_budget_metrics': {str(p): {name: sum(months[m]['budgets'][str(p)][name] for m in MONTHS) / len(MONTHS) for name in ('precision', 'recall')} for p in (10, 20)},
            'model_summary': {'features': classifier.numFeatures, 'intercept': classifier.intercept,
                              'coefficient_L2_norm': float((coefficients ** 2).sum() ** 0.5),
                              'regularization': classifier.getRegParam(), 'positive_class_weight': manifest['positive_class_weight']},
            'training_in_sample_overall': train_overall,
            'checks': {'distributed_duplicate_boundary_fixture': 'passed', 'existing_prediction_population_and_threshold': 'passed',
                       'frozen_model_serialization_reload_scores': 'passed', 'candidate_specific_checks': 'not executed; candidate training not approved'},
            'historical_test_accessed_for_analysis': False, 'classifier_fits_started': 0}
        write_json(OUTPUT / 'results.json', result)
        status['status'] = 'complete'
        progress('Diagnostics complete; no model fitting or candidate selection was executed')
    except BaseException as exc:
        status.update(status='incomplete', error=repr(exc))
        traceback.print_exc()
        raise
    finally:
        if spark is not None:
            spark.stop()
        status['elapsed_seconds'] = time.monotonic() - start
        status['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(OUTPUT / 'status.json', status)


if __name__ == '__main__':
    main()

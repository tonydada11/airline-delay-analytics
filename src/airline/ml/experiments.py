"""Authoritative bounded M7 selection, shared with production publication."""
import argparse
import json
import math
from pathlib import Path
import re
from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.classification import GBTClassifier, GBTClassificationModel, LogisticRegressionModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.raw_snapshot import sha256_file, SNAPSHOT_ID
from airline.common.spark_session import create_spark_session
from airline.dws.dataset_manifest import validate_dataset_manifest, MANIFEST_NAME
from airline.dws.build_ml_flight_features import validate_feature_frame
from airline.dws.ml_feature_contract import TARGET_COLUMN, HISTORICAL_TEST_STATUS, FEATURE_CONTRACT_VERSION
from airline.ml.preprocessing import build_preprocessing_pipeline
from airline.ml.model_contract import (build_logistic_regression, MODEL_SETTINGS, POSITIVE_CLASS_WEIGHT,
                                       PREDICTION_COLUMNS, validate_models, write_json, code_provenance, runtime_provenance)
from airline.ml.metrics import calculate_metrics, confusion_metrics
from airline.ml.model_contract import GBT_SETTINGS, validate_selection
from airline.dws.flight_time import time_provenance
from airline.ml.evaluation_artifacts import complete_evaluation, evaluation_identity



def f1_fraction(counts):
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    denominator = 2 * tp + fp + fn
    return (2 * tp, denominator) if denominator else (0, 1)


def compare_f1(left, right):
    a, b = f1_fraction(left)
    c, d = f1_fraction(right)
    return (a * d > c * b) - (a * d < c * b)


def best_threshold(rows):
    best = None
    for row in sorted(rows, key=lambda r: r['step']):
        if best is None or compare_f1(row, best) > 0:
            best = row
    if best is None:
        raise ValueError('No candidate thresholds')
    return best


def select_classifier(results):
    # Baseline and ranking metrics are deliberately absent from this comparison.
    comparison = compare_f1(results['gbt'], results['lr'])
    if comparison:
        return 'gbt' if comparison > 0 else 'lr'
    return 'gbt' if results['gbt']['step'] < results['lr']['step'] else 'lr'


def threshold_counts(scored):
    label = F.col(TARGET_COLUMN) == 1
    expressions = []
    for step in range(101):
        positive = F.col('probability') >= F.lit(step / 100)
        for name, condition in [('tp', positive & label), ('fp', positive & ~label),
                                ('fn', ~positive & label), ('tn', ~positive & ~label)]:
            expressions.append(F.sum(F.when(condition, 1).otherwise(0)).alias(f'{name}_{step}'))
    counts = scored.agg(*expressions).first()
    return [dict(step=step, **{k: int(counts[f'{k}_{step}'] or 0) for k in ('tp','fp','fn','tn')})
            for step in range(101)]


def run_directory(artifact_root, run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Run ID must be 1-80 letters, digits, underscores or hyphens')
    root = Path(artifact_root).resolve()
    parent = root / 'experiments'
    destination = parent / run_id
    if parent.is_symlink() or destination.is_symlink() or destination.resolve().parent != parent:
        raise ValueError('Experiment paths may not escape the experiment root')
    return destination


def continuous_score(classifier, prepared):
    return classifier.transform(prepared).withColumn('probability', vector_to_array('probability')[1])


def select(spark, input_directory, artifact_root, run_id, snapshot, project_root):
    directory = run_directory(artifact_root, run_id)
    directory.mkdir(parents=True, exist_ok=False)  # Incomplete retries also require a new ID.
    dataset = spark.read.parquet(str(input_directory))
    validate_dataset_manifest(input_directory, dataset, snapshot['raw_manifest_sha256'], phase='selection')
    training = dataset.filter(F.col('split') == 'train')
    validation = dataset.filter(F.col('split') == 'validation')
    rows = validate_feature_frame(training)
    validate_feature_frame(validation)
    profile = training.agg(F.sum(TARGET_COLUMN).alias('positives'), F.min('flight_date').alias('min_date'), F.max('flight_date').alias('max_date')).first()
    prevalence = profile.positives / rows
    baseline = {'training_rows': rows, 'training_positive_rows': profile.positives,
                'probability': prevalence, 'threshold': 0.5,
                'validation': calculate_metrics(validation.withColumn('probability', F.lit(prevalence)), 0.5)}
    preprocessing = build_preprocessing_pipeline().fit(training)
    prepared = (preprocessing.transform(training)
                .withColumn('class_weight', F.when(F.col(TARGET_COLUMN) == 1, POSITIVE_CLASS_WEIGHT).otherwise(1.0))
                .select('features', TARGET_COLUMN, 'class_weight')
                .repartition(max(2, math.ceil(rows / 250000))).persist(StorageLevel.DISK_ONLY))
    validation_prepared = preprocessing.transform(validation).select(
        'flight_id', TARGET_COLUMN, 'features').persist(StorageLevel.DISK_ONLY)
    results, selected = {}, {}
    try:
        prepared.count()
        validation_prepared.count()
        preprocessing.write().save(str(directory / 'preprocessing'))
        reloaded_pre = PipelineModel.load(str(directory / 'preprocessing'))
        before_vectors = validation_prepared.select('flight_id', vector_to_array('features').alias('vector'))
        after_vectors = reloaded_pre.transform(validation).select('flight_id', vector_to_array('features').alias('vector'))
        if before_vectors.exceptAll(after_vectors).limit(1).count() or after_vectors.exceptAll(before_vectors).limit(1).count():
            raise ValueError('Preprocessing save/reload vectors differ')
        for name, estimator, loader in [('lr', build_logistic_regression(), LogisticRegressionModel),
                                        ('gbt', GBTClassifier(**GBT_SETTINGS), GBTClassificationModel)]:
            classifier = estimator.fit(prepared)
            classifier.write().save(str(directory / name))
            reloaded = loader.load(str(directory / name))
            if name == 'lr':
                validate_models(reloaded_pre, reloaded)
            else:
                for key, value in GBT_SETTINGS.items():
                    if reloaded.getOrDefault(key) != value:
                        raise ValueError(f'GBT reload setting mismatch: {key}')
                if reloaded.numClasses != 2 or reloaded.numFeatures != classifier.numFeatures:
                    raise ValueError('GBT reload dimensions mismatch')
            scored = continuous_score(reloaded, validation_prepared).select(
                'flight_id', TARGET_COLUMN, 'probability').persist(StorageLevel.DISK_ONLY)
            try:
                # Compare full validation score multisets after serialization, not a sample.
                before = continuous_score(classifier, validation_prepared).select('flight_id', TARGET_COLUMN, 'probability')
                after = scored.select('flight_id', TARGET_COLUMN, 'probability')
                if before.exceptAll(after).limit(1).count() or after.exceptAll(before).limit(1).count():
                    raise ValueError('Classifier save/reload scores differ')
                candidates = threshold_counts(scored)
                best = best_threshold(candidates)
                selected[name] = best
                results[name] = {'thresholds': candidates, 'selected': best,
                                 'validation': calculate_metrics(scored, best['step'] / 100)}
            finally:
                scored.unpersist()
        write_json(directory / 'baseline.json', baseline)
        write_json(directory / 'validation.json', results)
        winner = select_classifier(selected)
        manifest = {'schema_version': 3, 'run_id': run_id, 'selected_classifier': winner,
                    'runtime_provenance': runtime_provenance(project_root),
                    'threshold_step': selected[winner]['step'], 'threshold_denominator': 100,
                    'selection_rule': 'exact rational F1; lower step then LR on ties; baseline excluded',
                    'training': {'rows': rows, 'min_date': str(profile.min_date), 'max_date': str(profile.max_date)},
                    'time_policy': time_provenance(), 'validation_results': results,
                    'estimator_settings': {'lr': MODEL_SETTINGS, 'gbt': GBT_SETTINGS},
                    'positive_class_weight': POSITIVE_CLASS_WEIGHT, 'feature_version': FEATURE_CONTRACT_VERSION,
                    'historical_test_status': HISTORICAL_TEST_STATUS, **snapshot,
                    'dataset_manifest_sha256': sha256_file(Path(input_directory) / MANIFEST_NAME),
                    'code_provenance': code_provenance(project_root), 'spark_version': spark.version,
                    'baseline_sha256': sha256_file(directory / 'baseline.json'),
                    'validation_sha256': sha256_file(directory / 'validation.json')}
        validate_selection(manifest, snapshot['raw_manifest_sha256'])
        write_json(directory / 'selection.json', manifest)  # Completion marker, last.
        return manifest
    finally:
        prepared.unpersist()
        validation_prepared.unpersist()


def evaluate(spark, input_directory, artifact_root, run_id, split, expected_fingerprint):
    if split != 'test':
        raise ValueError('Experiment evaluation requires --split test')
    directory = run_directory(artifact_root, run_id)
    try:
        manifest = json.loads((directory / 'selection.json').read_text())
    except (OSError, ValueError) as exc:
        raise ValueError('Experiment selection is incomplete; use a new run ID to retry selection') from exc
    validate_selection(manifest, expected_fingerprint)
    if manifest.get('run_id') != run_id:
        raise ValueError('Selection run ID differs')
    for name in ('baseline', 'validation'):
        if sha256_file(directory / f'{name}.json') != manifest[f'{name}_sha256']:
            raise ValueError(f'Changed frozen {name} report')
    step = manifest['threshold_step']
    if type(step) is not int or not 0 <= step <= 100 or manifest['selected_classifier'] not in ('lr','gbt'):
        raise ValueError('Invalid frozen selection')
    validation = json.loads((directory / 'validation.json').read_text())
    selections = {name: best_threshold(validation[name]['thresholds']) for name in ('lr', 'gbt')}
    winner = select_classifier(selections)
    if (manifest.get('threshold_denominator') != 100 or winner != manifest['selected_classifier']
            or selections[winner]['step'] != step):
        raise ValueError('Frozen selection differs from validation evidence')
    report_path = directory / 'test.json'
    dataset = spark.read.parquet(str(input_directory))
    test_manifest = validate_dataset_manifest(input_directory, dataset, expected_fingerprint, phase='test')
    if test_manifest.get('selection_run_id') != run_id:
        raise ValueError('Test features belong to a different frozen selection')
    features = dataset.filter(F.col('split') == 'test')
    validate_feature_frame(features)
    preprocessing = PipelineModel.load(str(directory / 'preprocessing'))
    name = manifest['selected_classifier']
    classifier = (LogisticRegressionModel if name == 'lr' else GBTClassificationModel).load(str(directory / name))
    baseline = json.loads((directory / 'baseline.json').read_text())
    identity = evaluation_identity(input_directory,
        {'preprocessing': directory / 'preprocessing', 'classifier': directory / name},
        {'selection': directory / 'selection.json', 'dataset': Path(input_directory) / MANIFEST_NAME,
         'baseline': directory / 'baseline.json', 'validation': directory / 'validation.json'}, split, step)

    def predict():
        return continuous_score(classifier, preprocessing.transform(features)).withColumn(
            'prediction', (F.col('probability') >= step / 100).cast('int')).select(*PREDICTION_COLUMNS)

    def make_report(predictions):
        return {'run_id': run_id, 'split': split, 'selected_classifier': name, 'threshold_step': step,
                'historical_test_status': HISTORICAL_TEST_STATUS,
                'selection_sha256': identity['manifests']['selection'],
                'dataset_manifest_sha256': identity['manifests']['dataset'],
                'selected': calculate_metrics(predictions, step / 100),
                'baseline': calculate_metrics(features.withColumn('probability', F.lit(baseline['probability'])), 0.5)}

    return complete_evaluation(spark, features, step / 100, report_path,
                               directory / 'test_predictions', identity, predict, make_report)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('select').add_argument('--run-id', required=True)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--run-id', required=True)
    evaluation.add_argument('--split', required=True, choices=['test'])
    args = parser.parse_args()
    paths = get_project_paths(load_config())
    fingerprint = sha256_file(paths.resolve(paths.raw_manifest))
    spark = create_spark_session(f'airline-experiment-{args.command}')
    try:
        input_path = paths.dws_ml_flight_features if args.command == 'select' else paths.dws_ml_test_features
        common = (spark, paths.resolve(input_path), paths.resolve(paths.artifact_root), args.run_id)
        if args.command == 'select':
            result = select(*common, {'source_snapshot_id': SNAPSHOT_ID, 'raw_manifest_sha256': fingerprint}, paths.project_root)
        else:
            result = evaluate(*common, args.split, fingerprint)
        print(json.dumps(result, sort_keys=True))
    finally:
        spark.stop()


if __name__ == '__main__':
    main()

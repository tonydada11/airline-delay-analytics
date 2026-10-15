"""Completion and recovery for sequential local-filesystem evaluations."""
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from pyspark.sql import functions as F

from airline.common.lineage import BUSINESS_KEY
from airline.common.raw_snapshot import sha256_file
from airline.dws.ml_feature_contract import TARGET_COLUMN
from airline.ml.model_contract import PREDICTION_COLUMNS, write_json


def inventory(directory, parquet_only=False):
    """Hash bytes, not mtimes or manifests alone; retain portable relative names."""
    root = Path(directory)
    if not root.is_dir() or (parquet_only and not (root / '_SUCCESS').is_file()):
        raise ValueError('Missing completed artifact directory')
    if any(p.name == '_temporary' for p in root.rglob('*')):
        raise ValueError('Unfinished artifact write')
    files = [p for p in sorted(root.rglob('*')) if p.is_file()
             and not p.name.startswith('.')
             and (not parquet_only or p.suffix == '.parquet' or p.name == '_SUCCESS')]
    if not files or (parquet_only and not any(p.suffix == '.parquet' for p in files)):
        raise ValueError('Missing artifact data')
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


def evaluation_identity(input_directory, model_directories, manifests, split, step):
    return {'version': 1, 'split': split, 'threshold_step': step,
            'input_files': inventory(input_directory, parquet_only=True),
            'model_files': {name: inventory(path) for name, path in model_directories.items()},
            'manifests': {name: sha256_file(path) for name, path in manifests.items()}}


def report_digest(report):
    payload = {k: v for k, v in report.items() if k != 'completion'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_predictions(predictions, features, threshold):
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError('Prediction columns differ from contract')
    expected = dict(features.dtypes)
    expected.update(probability='double', prediction='int')
    if any(expected[c] != t for c, t in predictions.dtypes):
        raise ValueError('Prediction types differ from contract')
    invalid = (F.col('probability').isNull() | F.isnan('probability')
               | ~F.col('probability').between(0, 1) | F.col('prediction').isNull()
               | (F.col('prediction') != (F.col('probability') >= threshold).cast('int')))
    stats = predictions.agg(F.count('*').alias('rows'),
                           F.sum(F.when(invalid, 1).otherwise(0)).alias('invalid')).first()
    if not stats.rows or stats.invalid:
        raise ValueError('Empty or invalid predictions')
    keys = ['flight_id', *BUSINESS_KEY, 'split', TARGET_COLUMN]
    left, right = features.select(*keys), predictions.select(*keys)
    if left.exceptAll(right).limit(1).count() or right.exceptAll(left).limit(1).count():
        raise ValueError('Prediction population differs from evaluated features')
    return stats.rows


def cached_report(spark, features, threshold, report_path, prediction_path, identity):
    try:
        report = json.loads(Path(report_path).read_text())
        completion = report['completion']
        expected_rows = completion['rows']
        if (type(expected_rows) is not int or expected_rows <= 0
                or completion['identity'] != identity
                or completion['report_sha256'] != report_digest(report)
                or completion['prediction_files'] != inventory(prediction_path, parquet_only=True)):
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    # File fingerprints also catch corruption before asking Spark to read it.
    predictions = spark.read.parquet(str(prediction_path))
    try:
        rows = validate_predictions(predictions, features, threshold)
        if rows != expected_rows:
            return None
    except (ValueError, AssertionError):
        return None
    return report


def complete_evaluation(spark, features, threshold, report_path, prediction_path,
                        identity, predict, make_report):
    report_path, prediction_path = Path(report_path), Path(prediction_path)
    recorded = cached_report(spark, features, threshold, report_path, prediction_path, identity)
    if recorded is not None:
        return recorded
    # Revoke completion before any repair, including scoring and interrupted writes.
    report_path.unlink(missing_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staged = prediction_path.with_name(f'.{prediction_path.name}.tmp-{token}')
    temporary_report = report_path.with_name(f'.{report_path.name}.tmp-{token}')
    try:
        predictions = predict()
        predictions.write.parquet(str(staged))
        saved = spark.read.parquet(str(staged))
        rows = validate_predictions(saved, features, threshold)
        report = make_report(saved)
        report['completion'] = {'identity': identity, 'rows': rows,
                                'prediction_files': inventory(staged, parquet_only=True),
                                'report_sha256': report_digest(report)}
        write_json(temporary_report, report)
        if prediction_path.exists():
            shutil.rmtree(prediction_path)
        staged.rename(prediction_path)
        temporary_report.replace(report_path)  # The only completion marker, last.
        return report
    finally:
        if staged.exists():
            shutil.rmtree(staged)
        temporary_report.unlink(missing_ok=True)

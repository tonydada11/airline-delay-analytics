"""Dataset completion and compatibility contract, published after Parquet validation."""
import json
from pathlib import Path
from airline.common.lineage import LINEAGE_DEFINITION
from airline.common.raw_snapshot import SNAPSHOT_ID
from airline.dws.flight_time import time_provenance
from airline.dws.ml_feature_contract import FEATURE_CONTRACT_VERSION, ML_DATASET_SCHEMA_VERSION, OUTPUT_COLUMNS

MANIFEST_NAME = '_dataset_manifest.json'


def schema_columns(frame):
    return [{'name': c, 'type': t} for c, t in frame.dtypes]


def publish_dataset_manifest(directory, frame, snapshot, phase='selection', time_audit=None, selection_manifest=None, expected_rows=None):
    from pyspark.sql import functions as F
    from airline.dws.build_ml_flight_features import validate_feature_frame
    rows = validate_feature_frame(frame, expected_rows)
    splits = sorted(r.split for r in frame.select('split').distinct().collect())
    allowed = {'train', 'validation'} if phase == 'selection' else {'test'}
    if not set(splits) <= allowed:
        raise ValueError('Dataset contains splits outside its phase')
    duplicates = frame.groupBy('flight_id').count().filter('count > 1').agg(
        F.count('*').alias('keys'), F.sum(F.col('count') - 1).alias('extra')).first()
    document = {'dataset_schema_version': ML_DATASET_SCHEMA_VERSION,
                'feature_version': FEATURE_CONTRACT_VERSION, 'lineage': LINEAGE_DEFINITION,
                'time_policy': time_provenance(), 'phase': phase, 'time_audit': time_audit or {},
                'selection_bundle_id': selection_manifest['bundle_id'] if selection_manifest else None,
                'selection_run_id': selection_manifest['selection']['run_id'] if selection_manifest else None,
                'columns': schema_columns(frame), 'splits': splits, 'rows': rows,
                'duplicate_key_count': duplicates['keys'], 'duplicate_extra_rows': duplicates['extra'] or 0,
                **snapshot}
    path = Path(directory) / MANIFEST_NAME
    path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return document


def validate_dataset_manifest(directory, frame, expected_fingerprint=None, phase=None):
    try:
        document = json.loads((Path(directory) / MANIFEST_NAME).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError('Missing or invalid ML dataset completion manifest. Rebuild ML features.') from exc
    expected = {'dataset_schema_version': ML_DATASET_SCHEMA_VERSION,
                'feature_version': FEATURE_CONTRACT_VERSION, 'lineage': LINEAGE_DEFINITION,
                'time_policy': time_provenance(),
                'source_snapshot_id': SNAPSHOT_ID, 'columns': schema_columns(frame)}
    if phase is not None:
        expected['phase'] = phase
    if document.get('phase') not in ('selection', 'test'):
        raise ValueError('Invalid dataset phase. Rebuild ML features.')
    allowed = {'train', 'validation'} if document['phase'] == 'selection' else {'test'}
    if not document.get('splits') or not set(document['splits']) <= allowed:
        raise ValueError('Invalid dataset split contract')
    if tuple(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError('Incompatible ML dataset columns. Rebuild ML features.')
    for key, value in expected.items():
        if document.get(key) != value:
            raise ValueError(f'Incompatible ML dataset manifest: {key}. Rebuild ML features.')
    import re
    if not re.fullmatch('[0-9a-f]{64}', document.get('raw_manifest_sha256', '')):
        raise ValueError('Invalid ML dataset snapshot fingerprint')
    if expected_fingerprint is not None and document['raw_manifest_sha256'] != expected_fingerprint:
        raise ValueError('ML dataset snapshot fingerprint differs from configured manifest')
    rows = document.get('rows')
    if type(rows) is not int or rows <= 0 or frame.count() != rows:
        raise ValueError('ML dataset row count differs from completion manifest. Rebuild ML features.')
    return document

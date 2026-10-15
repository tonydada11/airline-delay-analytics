# AI-generated test code: generated with OpenAI Codex.
"""A surviving manifest/_SUCCESS cannot hide a missing input Parquet part."""
from datetime import date
import json
import os
from pathlib import Path
import shutil
import tempfile

import pytest

from airline.common.lineage import INTEGER_KEYS, with_lineage
from airline.common.raw_snapshot import SNAPSHOT_ID
from airline.common.spark_session import create_spark_session
from airline.dws.dataset_manifest import MANIFEST_NAME, publish_dataset_manifest, validate_dataset_manifest
from airline.dws.ml_feature_contract import (
    OUTPUT_COLUMNS, TARGET_COLUMN, CATEGORICAL_COLUMNS, SCHEDULE_NUMERIC,
    HISTORY_RATES, HISTORY_COUNTS, HISTORY_MISSING,
)


@pytest.fixture(scope='module')
def completed_dataset():
    spark = create_spark_session('airline-dataset-completion-regression')
    parent = Path(os.environ.get('AIRLINE_TEST_TMP_ROOT', tempfile.gettempdir()))
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='dataset-completion-', dir=parent))
    try:
        row = {'flight_date': date(2023, 1, 1), 'split': 'train', TARGET_COLUMN: 0}
        row.update({c: 'AA' for c in CATEGORICAL_COLUMNS})
        row.update({c: 1.0 for c in SCHEDULE_NUMERIC})
        row.update({c: 0.0 for c in ('departure_missing', 'arrival_missing', 'duration_missing')})
        row.update({c: 0.3 for c in HISTORY_RATES})
        row.update({c: 10 for c in HISTORY_COUNTS})
        row.update({c: 0.0 for c in HISTORY_MISSING})
        rows = [{**row, **{k: i for k in INTEGER_KEYS}} for i in range(5)]
        frame = with_lineage(spark.createDataFrame(rows)).select(*OUTPUT_COLUMNS)
        directory = root / 'complete'
        frame.repartition(2).write.parquet(str(directory))
        publish_dataset_manifest(directory, spark.read.parquet(str(directory)),
            {'source_snapshot_id': SNAPSHOT_ID, 'raw_manifest_sha256': 'a'*64})
        yield spark, root, directory
    finally:
        spark.stop()
        shutil.rmtree(root)


def test_complete_input_matches_declared_rows(completed_dataset):
    spark, _, directory = completed_dataset
    assert validate_dataset_manifest(directory, spark.read.parquet(str(directory)))['rows'] == 5


def test_publisher_rejects_reload_multiplicity_change_before_completion(completed_dataset):
    spark, root, directory = completed_dataset
    destination = root / 'wrong-population'
    destination.mkdir()
    with pytest.raises(AssertionError, match='Eligible-flight multiplicity changed'):
        publish_dataset_manifest(destination, spark.read.parquet(str(directory)),
            {'source_snapshot_id': SNAPSHOT_ID, 'raw_manifest_sha256': 'a' * 64}, expected_rows=6)
    assert not (destination / MANIFEST_NAME).exists()


def test_missing_input_part_is_not_a_complete_changed_dataset(completed_dataset):
    spark, root, directory = completed_dataset
    partial = root / 'partial'
    shutil.copytree(directory, partial)
    parts = list(partial.glob('*.parquet'))
    assert len(parts) == 2
    parts[0].unlink()
    assert (partial / '_SUCCESS').exists()
    assert (partial / MANIFEST_NAME).exists()
    with pytest.raises(ValueError, match='row count differs from completion manifest'):
        validate_dataset_manifest(partial, spark.read.parquet(str(partial)))


@pytest.mark.parametrize('declared', [None, 0, 4, 5.0])
def test_invalid_declared_row_count_is_rejected(completed_dataset, declared):
    spark, _, directory = completed_dataset
    path = directory / MANIFEST_NAME
    original = path.read_text()
    try:
        document = json.loads(original)
        document['rows'] = declared
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match='row count differs from completion manifest'):
            validate_dataset_manifest(directory, spark.read.parquet(str(directory)))
    finally:
        path.write_text(original)

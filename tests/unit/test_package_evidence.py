# AI-generated test code: generated with OpenAI Codex.
"""Packaging must bind explicit acceptance records to current bytes and reports."""
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location('package_evidence', Path(__file__).resolve().parents[2] / 'scripts/package_evidence.py')
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


@pytest.fixture
def accepted_project(tmp_path, monkeypatch):
    # Model validation itself is covered by model-contract tests. This fixture
    # isolates packaging's record/content checks without fitting Spark models.
    monkeypatch.setattr(packaging, 'validate_manifest', lambda *args: None)
    root = tmp_path / 'project'
    for name in ('src/example.py', 'tests/example.py', 'scripts/example.sh',
                 *packaging.RUNTIME_NAMES, 'README.md', 'LOCAL_SETUP.md', '.gitattributes', '.gitignore',
                 'docs/PIPELINE_RULES.md', 'docs/RUNBOOK.md'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture\n')
    (root / 'scripts/example.sh').chmod(0o755)
    config = Path(__file__).resolve().parents[2] / 'conf/project.yaml'
    (root / 'conf').mkdir()
    (root / 'conf/project.yaml').write_bytes(config.read_bytes())
    (root / 'conf/raw_data_manifest.csv').write_text('pinned manifest fixture\n')
    write(root / 'notebooks/example.ipynb', {'cells': [{'cell_type': 'code', 'execution_count': 1, 'outputs': [{'output_type': 'stream', 'text': 'old output'}], 'source': ['1+1']}], 'nbformat': 4, 'nbformat_minor': 5, 'metadata': {}})
    paths = packaging.get_project_paths(packaging.load_config(root / 'conf/project.yaml'))
    artifact = paths.resolve(paths.artifact_root)
    run_id = 'production-packaging-test'
    experiment = packaging.run_directory(artifact, run_id)
    for split, configured in (('validation', paths.dws_ml_flight_features), ('test', paths.dws_ml_test_features)):
        directory = paths.resolve(configured)
        directory.mkdir(parents=True)
        (directory / 'part-00000.parquet').write_bytes(b'PAR1-fixture')
        (directory / '_SUCCESS').touch()
        write(directory / '_dataset_manifest.json', {'dataset_schema_version': 3,
            'phase': 'selection' if split == 'validation' else 'test', 'selection_bundle_id': 'bundle-test'})
    write(experiment / 'baseline.json', {'probability': 0.2})
    write(experiment / 'validation.json', {'fixture': 'frozen validation'})
    selection = {'run_id': run_id,
                 'dataset_manifest_sha256': packaging.digest(paths.resolve(paths.dws_ml_flight_features) / '_dataset_manifest.json'),
                 'baseline_sha256': packaging.digest(experiment / 'baseline.json'),
                 'validation_sha256': packaging.digest(experiment / 'validation.json')}
    write(experiment / 'selection.json', selection)
    model = {'selection': selection, 'bundle_id': 'bundle-test', 'threshold_step': 39}
    write(artifact / 'models/manifest.json', model)
    for name in ('preprocessing', 'classifier'):
        path = artifact / 'models' / name / 'metadata/part-00000'
        path.parent.mkdir(parents=True)
        path.write_text('frozen model bytes\n')
    for split, configured in (('validation', paths.dws_ml_flight_features), ('test', paths.dws_ml_test_features)):
        prediction = artifact / 'predictions' / split
        prediction.mkdir(parents=True)
        (prediction / 'part-00000.parquet').write_bytes(b'PAR1-predictions')
        (prediction / '_SUCCESS').touch()
        identity = packaging.evaluation_identity(paths.resolve(configured),
            {'preprocessing': artifact / 'models/preprocessing', 'classifier': artifact / 'models/classifier'},
            {'model': artifact / 'models/manifest.json', 'dataset': paths.resolve(configured) / '_dataset_manifest.json'}, split, 39)
        report = {'rows': 2, 'bundle_id': 'bundle-test', 'split': split}
        report['completion'] = {'identity': identity, 'rows': 2,
            'prediction_files': packaging.inventory(prediction, parquet_only=True),
            'report_sha256': packaging.report_digest(report)}
        write(artifact / 'metrics' / f'{split}.json', report)
    record = {'schema_version': 3, 'status': 'passed', 'tested_commit': 'a' * 40,
              'run_id': run_id, 'source_files': packaging.file_identities(root, packaging.SOURCE_FOLDERS),
              'runtime_files': packaging.runtime_identities(root),
              'artifact_files': packaging.artifact_identities(root, paths, run_id),
              'commands': {}, 'comparisons': {}}
    record_path = root / 'evidence/acceptance/test/run-summary.json'
    for group, names in (('commands', ('pipeline', 'ordinary-tests', 'rebuild-tests')),
                         ('comparisons', ('features', 'frozen-rescore'))):
        for name in names:
            log = record_path.parent / f'{name}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text('passed\n')
            record[group][name] = {'exit_code': 0, 'log': log.relative_to(root).as_posix(),
                                   'log_sha256': packaging.digest(log)}
    write(record_path, record)
    return root, paths, artifact, run_id, record_path, record


def test_package_uses_schema3_configured_reports_and_verifies_extraction(accepted_project, tmp_path):
    root, _, _, run_id, record_path, _ = accepted_project
    output = tmp_path / 'delivery.zip'
    result = packaging.package(root, output, run_id, record_path)
    assert result['sha256'] == packaging.digest(output)
    assert output.with_suffix('.zip.sha256').read_text().startswith(result['sha256'])
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert {'airline/evidence/selection_dataset_manifest.json', 'airline/evidence/test_dataset_manifest.json',
                'airline/evidence/model_manifest.json', 'airline/evidence/m7_selection.json'} <= names
        assert 'airline/evidence/m7_test.json' not in names
        assert not any('/artifacts/' in n or '/data/' in n for n in names)
        notebook = json.loads(archive.read('airline/notebooks/example.ipynb'))
        assert notebook['cells'][0]['outputs'] == []
        assert notebook['cells'][0]['execution_count'] is None
        assert (archive.getinfo('airline/scripts/example.sh').external_attr >> 16) & 0o111
    with pytest.raises(FileExistsError):
        packaging.package(root, output, run_id, record_path)


@pytest.mark.parametrize('damage', ['missing_record', 'historical_format', 'failed_pipeline', 'missing_comparison', 'changed_log', 'wrong_run'])
def test_package_rejects_incomplete_or_unrelated_acceptance(accepted_project, damage):
    root, _, _, run_id, record_path, record = accepted_project
    if damage == 'missing_record':
        record_path.unlink()
    elif damage == 'historical_format':
        write(record_path, {'production': 'passed', 'ordinary_acceptance': 'passed'})
    elif damage == 'failed_pipeline':
        record['commands']['pipeline']['exit_code'] = 1
        write(record_path, record)
    elif damage == 'missing_comparison':
        del record['comparisons']['features']
        write(record_path, record)
    elif damage == 'changed_log':
        (root / record['commands']['pipeline']['log']).write_text('changed\n')
    else:
        run_id = 'another-run'
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        packaging.validate_acceptance(root, run_id, record_path)


@pytest.mark.parametrize('changed', ['source', 'runtime', 'model', 'features', 'predictions', 'report'])
def test_package_rejects_changed_accepted_bytes(accepted_project, changed):
    root, paths, artifact, run_id, record_path, _ = accepted_project
    path = {'source': root / 'src/example.py', 'runtime': root / 'requirements.lock',
            'model': artifact / 'models/classifier/metadata/part-00000',
            'features': paths.resolve(paths.dws_ml_flight_features) / 'part-00000.parquet',
            'predictions': artifact / 'predictions/test/part-00000.parquet',
            'report': artifact / 'metrics/test.json'}[changed]
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='identity differs'):
        packaging.validate_acceptance(root, run_id, record_path)

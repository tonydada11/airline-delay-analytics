#!/usr/bin/env python3
"""Package source and compact reports from one explicit schema-3 acceptance record."""
import argparse
import json
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.raw_snapshot import sha256_file
from airline.ml.evaluation_artifacts import evaluation_identity, inventory, report_digest
from airline.ml.experiments import run_directory
from airline.ml.model_contract import validate_manifest

SOURCE_FOLDERS = ('src', 'tests', 'scripts', 'conf')
RUNTIME_NAMES = ('.dockerignore', 'docker-compose.yml', 'docker/Dockerfile',
                 'pyproject.toml', 'requirements.in', 'requirements.lock')


def file_identities(root, folders):
    return {p.relative_to(root).as_posix(): sha256_file(p)
            for folder in folders for p in sorted((root / folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def runtime_identities(root):
    names = (*RUNTIME_NAMES, 'docker-compose.override.yml') if (root / 'docker-compose.override.yml').exists() else RUNTIME_NAMES
    return {name: sha256_file(root / name) for name in names}


def artifact_identities(root, paths, run_id):
    artifact = paths.resolve(paths.artifact_root)
    directories = [artifact / 'models', artifact / 'predictions/validation',
                   artifact / 'predictions/test', run_directory(artifact, run_id),
                   paths.resolve(paths.dws_ml_flight_features), paths.resolve(paths.dws_ml_test_features)]
    result = {}
    for directory in directories:
        for name, digest in inventory(directory).items():
            result[(directory / name).relative_to(root).as_posix()] = digest
    for split in ('validation', 'test'):
        path = artifact / 'metrics' / f'{split}.json'
        result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def validate_acceptance(root, run_id, record_path):
    record = json.loads(record_path.read_text())
    if (record['schema_version'] != 3 or record['status'] != 'passed'
            or len(record['tested_commit']) != 40
            or any(c not in '0123456789abcdef' for c in record['tested_commit'])):
        raise ValueError('A passed current schema-3 acceptance record is required')
    if record['source_files'] != file_identities(root, SOURCE_FOLDERS):
        raise ValueError('Acceptance source identity differs from current files')
    if record['runtime_files'] != runtime_identities(root):
        raise ValueError('Acceptance runtime identity differs from current files')
    for group, names in (('commands', ('pipeline', 'ordinary-tests', 'rebuild-tests')),
                         ('comparisons', ('features', 'frozen-rescore'))):
        for name in names:
            command = record[group][name]
            if command['exit_code'] != 0 or sha256_file(root / command['log']) != command['log_sha256']:
                raise ValueError(f'Acceptance command failed or log changed: {name}')
    paths = get_project_paths(load_config(root / 'conf/project.yaml'))
    artifact = paths.resolve(paths.artifact_root)
    model = json.loads((artifact / 'models/manifest.json').read_text())
    validate_manifest(model, sha256_file(paths.resolve(paths.raw_manifest)))
    if run_id != model['selection']['run_id'] or run_id != record['run_id']:
        raise ValueError('Run ID differs from the published selection or acceptance record')
    if record['artifact_files'] != artifact_identities(root, paths, run_id):
        raise ValueError('Acceptance artifact identity differs from current files')
    experiment = run_directory(artifact, run_id)
    selection = json.loads((experiment / 'selection.json').read_text())
    if selection != model['selection']:
        raise ValueError('Published selection differs from saved selection evidence')
    for name in ('baseline', 'validation'):
        if sha256_file(experiment / f'{name}.json') != selection[f'{name}_sha256']:
            raise ValueError(f'Frozen selection report changed: {name}')
    for split, configured in (('validation', paths.dws_ml_flight_features), ('test', paths.dws_ml_test_features)):
        dataset = paths.resolve(configured)
        document = json.loads((dataset / '_dataset_manifest.json').read_text())
        if document['dataset_schema_version'] != 3 or document['phase'] != ('selection' if split == 'validation' else 'test'):
            raise ValueError('Dataset is not the required schema-3 phase')
        if split == 'test' and document['selection_bundle_id'] != model['bundle_id']:
            raise ValueError('Test features belong to a different frozen bundle')
        if split == 'validation' and sha256_file(dataset / '_dataset_manifest.json') != selection['dataset_manifest_sha256']:
            raise ValueError('Selection dataset manifest changed')
        report = json.loads((artifact / 'metrics' / f'{split}.json').read_text())
        identity = evaluation_identity(dataset,
            {'preprocessing': artifact / 'models/preprocessing', 'classifier': artifact / 'models/classifier'},
            {'model': artifact / 'models/manifest.json', 'dataset': dataset / '_dataset_manifest.json'},
            split, model['threshold_step'])
        completion = report['completion']
        if (completion['identity'] != identity or completion['report_sha256'] != report_digest(report)
                or completion['prediction_files'] != inventory(artifact / 'predictions' / split, parquet_only=True)
                or type(completion['rows']) is not int or completion['rows'] <= 0
                or completion['rows'] != report['rows'] or report['bundle_id'] != model['bundle_id']):
            raise ValueError('Evaluation completion identity differs from current artifacts')
    return paths


def digest(path):
    return sha256_file(path)


def package(root, output, run_id, acceptance_record):
    root, output, acceptance_record = Path(root).resolve(), Path(output).resolve(), Path(acceptance_record).resolve()
    paths = validate_acceptance(root, run_id, acceptance_record)
    artifact = paths.resolve(paths.artifact_root)
    if output.exists() or output.with_suffix(output.suffix + '.sha256').exists():
        raise FileExistsError('Archive or sidecar already exists')
    with tempfile.TemporaryDirectory(prefix='airline-share-') as temporary:
        staging = Path(temporary) / 'airline'
        staging.mkdir()
        for directory in ('src', 'tests', 'scripts', 'docs', 'conf', 'docker', 'notebooks'):
            shutil.copytree(root / directory, staging / directory,
                ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '.ipynb_checkpoints', '.DS_Store', '*.pyc'))
        # Include immutable records, never the package's later external delivery record.
        for directory in ('archive', 'consolidation'):
            if (root / 'evidence' / directory).exists():
                shutil.copytree(root / 'evidence' / directory, staging / 'evidence' / directory,
                    ignore=shutil.ignore_patterns('data', 'warehouse', 'models', 'predictions', '__pycache__', '.pytest_cache'))
        accepted = staging / acceptance_record.parent.relative_to(root)
        accepted.mkdir(parents=True, exist_ok=True)
        for source in acceptance_record.parent.iterdir():
            if source.is_file() and source.name not in {'delivery.json'}:
                shutil.copy2(source, accepted / source.name)
        for name in ('README.md', 'LOCAL_SETUP.md', 'pyproject.toml', 'requirements.in', 'requirements.lock',
                     'docker-compose.yml', '.dockerignore', '.gitattributes', '.gitignore'):
            shutil.copy2(root / name, staging / name)
        reports = {'selection_dataset_manifest.json': paths.resolve(paths.dws_ml_flight_features) / '_dataset_manifest.json',
                   'test_dataset_manifest.json': paths.resolve(paths.dws_ml_test_features) / '_dataset_manifest.json',
                   'model_manifest.json': artifact / 'models/manifest.json'}
        for split in ('validation', 'test'):
            reports[f'production_{split}.json'] = artifact / 'metrics' / f'{split}.json'
        for name in ('baseline', 'validation', 'selection'):
            reports[f'm7_{name}.json'] = run_directory(artifact, run_id) / f'{name}.json'
        for name, source in reports.items():
            shutil.copy2(source, staging / 'evidence' / name)
        for path in (staging / 'notebooks').glob('*.ipynb'):
            notebook = json.loads(path.read_text())
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    cell['outputs'] = []
                    cell['execution_count'] = None
            path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
        files = sorted((p for p in staging.rglob('*') if p.is_file()), key=lambda p: p.relative_to(staging).as_posix())
        manifest = [{'path': p.relative_to(staging).as_posix(), 'sha256': digest(p),
                     'executable': bool(p.stat().st_mode & stat.S_IXUSR)} for p in files]
        (staging / 'evidence_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(p for p in staging.rglob('*') if p.is_file()):
                archive.write(path, Path('airline') / path.relative_to(staging))
        extracted = Path(temporary) / 'verify'
        with zipfile.ZipFile(output) as archive:
            assert archive.testzip() is None
            archive.extractall(extracted)
            for entry in archive.infolist():
                (extracted / entry.filename).chmod((entry.external_attr >> 16) & 0o777)
        verify = extracted / 'airline'
        actual = sorted(p.relative_to(verify).as_posix() for p in verify.rglob('*') if p.is_file() and p.name != 'evidence_manifest.json')
        assert actual == [item['path'] for item in manifest]
        for item in manifest:
            path = verify / item['path']
            assert digest(path) == item['sha256']
            assert bool(path.stat().st_mode & stat.S_IXUSR) == item['executable']
            assert not set(path.relative_to(verify).parts) & {'data', 'warehouse', 'models', 'predictions', '__pycache__', '.git'}
        for path in (verify / 'notebooks').glob('*.ipynb'):
            for cell in json.loads(path.read_text())['cells']:
                if cell['cell_type'] == 'code':
                    assert cell['execution_count'] is None and cell['outputs'] == []
        for relative in ('docs/PIPELINE_RULES.md', 'docs/RUNBOOK.md', 'requirements.lock',
                         'evidence/selection_dataset_manifest.json', 'evidence/test_dataset_manifest.json',
                         'evidence/m7_selection.json', 'evidence/production_test.json'):
            assert (verify / relative).is_file()
    checksum = digest(output)
    with output.with_suffix(output.suffix + '.sha256').open('x') as stream:
        stream.write(f'{checksum}  {output.name}\n')
    result = {'archive': str(output), 'sha256': checksum, 'verified_files': len(manifest),
              'acceptance_record': acceptance_record.relative_to(root).as_posix(),
              'verification': 'extracted; hashes, permissions, notebooks, contents and exclusions passed'}
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--acceptance-record', required=True, type=Path)
    args = parser.parse_args()
    package(Path(__file__).resolve().parents[1], args.output, args.run_id, args.acceptance_record)

"""Verify the existing protected-file inventory; never alter protected paths."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import subprocess


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(study, phase):
    study = Path(study).resolve()
    root = study.parents[2]
    before_path = study / 'baseline-identities-before.json'
    before = json.loads(before_path.read_text())
    for name, expected in before['files'].items():
        path = root / name
        s = path.stat()
        actual = dict(sha256=digest(path), bytes=s.st_size, mode=oct(stat.S_IMODE(s.st_mode)),
                      device=s.st_dev, inode=s.st_ino, links=s.st_nlink, mtime_ns=s.st_mtime_ns)
        if actual != expected:
            raise ValueError('Protected file changed: ' + name)
    for name, mode in before['directory_modes'].items():
        if oct(stat.S_IMODE((root / name).stat().st_mode)) != mode:
            raise ValueError('Protected directory permissions changed: ' + name)
    folders = ('artifacts/flighttime-v3', 'data/history/baseline/checkpoint',
               'data/warehouse/dws/ml_flighttime_v3_selection',
               'data/warehouse/dws/ml_flighttime_v3_test', 'data/raw')
    present = {p.relative_to(root).as_posix() for folder in folders for p in (root / folder).rglob('*') if p.is_file()}
    assert present == set(before['files']), 'Protected file set changed'
    record = json.loads((root / 'evidence/acceptance/consolidation-20261020/run-summary.json').read_text())
    for group in ('source_files', 'runtime_files'):
        assert all(digest(root / p) == h for p, h in record[group].items()), group
    assert subprocess.check_output(['git', 'show-ref', '--tags'], cwd=root, text=True).splitlines() == before['tags']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == before['main_commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root)
    result = {'status': 'passed', 'phase': phase, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'protected_files': len(present), 'protected_bytes': sum(v['bytes'] for v in before['files'].values()),
              'inventory_sha256': digest(before_path), 'hashes_modes_mtime_inodes_links_file_set_unchanged': True,
              'source_runtime_acceptance_identities_unchanged': True, 'tags_and_main_unchanged': True}
    (study / f'candidate-preservation-{phase}.json').write_text(json.dumps(result, indent=2) + '\n')
    return result

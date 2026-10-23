"""Offline verification of generated evidence and preservation of baseline files."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = HERE/'evidence'
    before = json.loads((evidence/'preservation_before.json').read_text())
    tracked_changes = [name for name,wanted in before['tracked_sha256'].items()
                       if digest(ROOT/name)!=wanted]
    model_changes = [name for name,wanted in before['preprocessing_sha256'].items()
                     if digest(ROOT/name)!=wanted]
    artifact_changes = []
    for name,wanted in before['protected_size_mtime'].items():
        stat = (ROOT/name).stat()
        if [stat.st_size,stat.st_mtime_ns]!=wanted:
            artifact_changes.append(name)
    assert not tracked_changes and not model_changes and not artifact_changes
    results = json.loads((evidence/'results.json').read_text())
    assert results['status']=='PASS_FIXTURES_ONLY'
    assert len(results['checks'])==18 and all(v['passed'] for v in results['checks'].values())
    assert len(results['fits'])==8 and max(x['rows'] for x in results['fits'])==32768
    assert all(x['features']==784 for x in results['fits'])
    assert json.loads((evidence/'execution.json').read_text())['exit_code']==0
    provenance = json.loads((evidence/'source-provenance.json').read_text())
    for item in provenance['members']:
        assert digest(evidence/item['file'])==item['sha256']
    for name,wanted in provenance['executed_files_sha256'].items():
        assert digest(HERE/name)==wanted
    tests = subprocess.run(['python3','-m','unittest','studies.xgb_feasibility_v1.test_supervisor','-v'],
                           cwd=ROOT,capture_output=True,text=True,check=True)
    report = {'tracked_files_sha256_checked':len(before['tracked_sha256']),
        'accepted_preprocessing_files_sha256_checked':len(before['preprocessing_sha256']),
        'local_artifact_files_size_mtime_checked':len(before['protected_size_mtime']),
        'tracked_changes':tracked_changes,'preprocessing_changes':model_changes,'artifact_metadata_changes':artifact_changes,
        'artifact_check_limitation':'Non-preprocessing local artifacts checked by size/mtime, not content hash.',
        'fixture_checks':18,'fixture_fits':8,'supervisor_regression_tests':tests.stderr,
        'note':'Original post-run raw container comparison failed; see metadata-investigation.json. It is not overwritten or counted as a pass.'}
    (evidence/'verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='supervisor_regression_tests'},indent=2))


if __name__=='__main__':
    main()

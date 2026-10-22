"""One read-only, 60-minute GBT30 prefix diagnostic in the existing pinned runtime."""
import argparse
import io
from pathlib import Path
import select
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    sys.path.insert(0, str(ROOT))
    from studies.gbt_v1 import launch, prefix_diagnostic as diagnostic
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.parse_args(argv)
    started = time.monotonic()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True):
        raise ValueError('Commit tested diagnostic code before execution')
    diagnostic.exclusive_output(ROOT)
    files = ['studies/gbt_v1/prefix_diagnostic.py', 'studies/gbt_v1/run_candidate.py',
             'studies/lr_v1/diagnose.py', 'studies/lr_v1/metrics.py',
             'studies/lr_v1/evidence/baseline-identities-before.json',
             'studies/lr_v1/evidence/diagnostics/results.json']
    files += [f'studies/lr_v1/evidence/diagnostics/validation-2024-{m:02}.json' for m in range(5, 9)]
    files += ['studies/gbt_v1/evidence/' + name for name in diagnostic.study.REFERENCE_HASHES]
    files += ['studies/gbt_v1/evidence_30iter/' + name for name in
              ('completion.json', 'results.json', 'inputs.json', 'worker-status.json')]
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w') as archive:
        for name in files:
            archive.add(ROOT / name, arcname=name, recursive=False)
    # Reuse the corrected duration-only budget handoff and safe temporary source staging.
    bootstrap = launch.BOOTSTRAP.replace('from studies.gbt_v1.run_candidate import main',
                                        'from studies.gbt_v1.prefix_diagnostic import main').replace(
                                            '"--candidate", sys.argv[2],', '')
    command = ['docker', 'compose', 'exec', '-T', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-w', '/workspace',
               'spark-notebook', 'python3', '-B', '-c', bootstrap, revision]
    with subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        try:
            remaining = diagnostic.LIMIT - (time.monotonic() - started)
            if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                raise subprocess.TimeoutExpired(command, max(0, remaining))
            if process.stdout.readline() != b'GBT_STUDY_READY\n':
                raise ValueError('Container did not acknowledge duration handoff')
            preparation = time.monotonic() - started
            if preparation >= diagnostic.LIMIT - diagnostic.GRACE:
                raise ValueError('Host preparation exhausted diagnostic allowance')
            output, _ = process.communicate(input=(str(preparation) + '\n').encode() + payload.getvalue(),
                                            timeout=diagnostic.LIMIT - preparation)
            print(output.decode('utf-8', errors='replace'), end='')
            return process.returncode
        except BaseException:
            process.kill()
            process.communicate()
            raise


if __name__ == '__main__':
    sys.exit(main())

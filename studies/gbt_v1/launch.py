"""Stage read-only study source in the existing notebook; Step 3 only."""
import argparse
import io
from pathlib import Path
import select
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]

# This supervisor runs INSIDE the container. Killing a Docker client alone would
# not bound the Spark driver. No service startup, image build/pull, or Compose stop.
BOOTSTRAP = '''
import os, pathlib, sys, tarfile, tempfile, time
started = time.monotonic()
print("GBT_STUDY_READY", flush=True)
host_preparation_seconds = float(sys.stdin.buffer.readline())
staging = tempfile.TemporaryDirectory(prefix="gbt-study-code-")
with staging as directory:
    directory = directory + "/source/project"
    os.makedirs(directory)
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as archive:
        for member in archive:
            path = pathlib.PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise ValueError("Invalid study source member")
            archive.extract(member, directory)
    os.environ["PYTHONPATH"] = directory + ":/workspace/src:" + os.environ.get("PYTHONPATH", "")
    sys.path[:0] = [directory, "/workspace/src"]
    from studies.gbt_v1.run_candidate import main
    preparation_seconds = host_preparation_seconds + (time.monotonic() - started)
    sys.exit(main(["--execute", "--project-root", "/workspace", "--source-revision", sys.argv[1],
                   "--candidate", sys.argv[2],
                   "--preparation-seconds", str(preparation_seconds)], cleanup=staging.cleanup))
'''


def main(argv=None):
    sys.path.insert(0, str(ROOT))  # Also support direct script execution from the repository root.
    from studies.gbt_v1.run_candidate import NAME, CANDIDATES, candidate_contract, reference_records
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True,
                        help='Step 3 only: run the single 14,400-second attempt in the existing pinned runtime')
    parser.add_argument('--candidate', choices=CANDIDATES, default=NAME,
                        help='Explicit fixed candidate; default preserves attempt-001')
    args = parser.parse_args(argv)
    started = time.monotonic()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require_clean = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    if require_clean:
        raise ValueError('Commit the reviewed runner before the future execution')
    files = [ROOT / 'studies/gbt_v1/run_candidate.py', ROOT / 'studies/gbt_v1' / candidate_contract(args.candidate)[1],
             ROOT / 'studies/lr_v1/diagnose.py', ROOT / 'studies/lr_v1/metrics.py',
             ROOT / 'studies/lr_v1/evidence/baseline-identities-before.json',
             ROOT / 'studies/lr_v1/evidence/diagnostics/results.json',
             *[ROOT / f'studies/lr_v1/evidence/diagnostics/validation-2024-{m:02}.json' for m in range(5, 9)]]
    if args.candidate != NAME:
        directory, hashes, _ = reference_records(args.candidate)
        files.extend(ROOT / directory / name for name in hashes)
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w') as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(ROOT), recursive=False)
    command = ['docker', 'compose', 'exec', '-T', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '-w', '/workspace', 'spark-notebook', 'python3', '-B', '-c', BOOTSTRAP,
               revision, args.candidate]
    remaining = 14400 - (time.monotonic() - started)
    if remaining <= 60:
        raise ValueError('Host preparation exhausted the execution allowance')
    with subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        try:
            # Budget handoff follows readiness, accounting for Docker/VM startup.
            # Only durations cross this boundary; clock reference points do not.
            remaining = 14400 - (time.monotonic() - started)
            if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                raise subprocess.TimeoutExpired(command, max(0, remaining))
            if process.stdout.readline() != b'GBT_STUDY_READY\n':
                raise ValueError('Container did not acknowledge budget handoff')
            preparation_seconds = time.monotonic() - started
            remaining = 14400 - preparation_seconds
            if remaining <= 60:
                raise ValueError('Host preparation exhausted the execution allowance')
            message = (str(preparation_seconds) + '\n').encode() + payload.getvalue()
            output, _ = process.communicate(input=message, timeout=remaining)
            print(output.decode('utf-8', errors='replace'), end='')
            return process.returncode
        except BaseException:
            process.kill()
            process.communicate()
            raise


if __name__ == '__main__':
    sys.exit(main())

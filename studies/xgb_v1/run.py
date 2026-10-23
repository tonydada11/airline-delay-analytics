"""Run preflight, then at most one 2-hour full attempt; remove only study services."""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'artifacts/experiments/xgb-v1'
EVIDENCE = OUT / 'execution'
COMPOSE = ['docker', 'compose', '-f', str(HERE / 'compose.yml')]
NAMES = ['airline-xgb-v1-' + service + '-1' for service in ('master', 'worker1', 'worker2', 'driver')]
BASE_IMAGE = 'sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf'
IMAGE = 'airline-xgb-v1:3.0.5'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def output(command, timeout=30):
    return subprocess.check_output(command, text=True, timeout=timeout).strip()


def build_image():
    assert output(['docker', 'image', 'inspect', 'airline-revision-spark:3.5.9', '--format', '{{.Id}}']) == BASE_IMAGE
    context = OUT / 'build'
    wheels = context / 'wheels'
    wheels.mkdir(parents=True, exist_ok=True)
    for path in (ROOT / 'studies/xgb_feasibility_v1/runtime/wheels').glob('*.whl'):
        shutil.copy2(path, wheels / path.name)
    for name in ('Dockerfile', 'requirements.lock'):
        shutil.copy2(HERE / name, context / name)
    command = ['docker', 'build', '--network=none', '-t', IMAGE, str(context)]
    with (EVIDENCE / 'build.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    write(EVIDENCE / 'build.json', dict(command=command, exit_code=result.returncode, base_image=BASE_IMAGE))
    if result.returncode:
        raise RuntimeError('Study image build failed; see build.log')


def artifact_metadata():
    # No test labels or metric contents are opened: preservation uses stat only.
    return {str(p.relative_to(ROOT)): [p.stat().st_size, p.stat().st_mtime_ns]
            for directory in ('data/warehouse', 'data/raw', 'data/history',
                              'artifacts/flighttime-v3', 'evidence/acceptance', 'results/production')
            for p in (ROOT / directory).rglob('*') if p.is_file()}


def protected():
    paths = [ROOT / name for name in ('requirements.lock', 'requirements.in', 'docker-compose.yml',
                                    'docker-compose.override.yml', 'docker/Dockerfile')]
    for directory in ('src', 'tests', 'conf', 'artifacts/flighttime-v3/models',
                      'data/warehouse/dws/ml_flighttime_v3_selection', 'studies/lr_v1'):
        paths.extend(p for p in (ROOT / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    result = {}
    for p in sorted(paths):
        digest = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
        result[str(p.relative_to(ROOT))] = digest.hexdigest()
    return result


def production():
    names = ['airline-revision-spark-' + n + '-1' for n in ('master', 'worker-1', 'worker-2', 'notebook')]
    records = json.loads(output(['docker', 'inspect', *names]))
    return [{k: sorted(r[k], key=lambda m: m['Destination']) if k == 'Mounts' else r[k]
             for k in ('Id', 'Image', 'State', 'Mounts')} for r in records]


def verify_distributed_fit(log, environment, metadata):
    # Worker rank messages are executor stderr, not necessarily driver stdout.
    # With one core/task per executor, the two barrier training tasks require
    # both hosts. Bind the completed fit log to the saved model's worker count.
    if ('Running xgboost-3.0.5 on 2 workers' not in log
            or 'Finished xgboost training!' not in log
            or len({w['host'] for w in environment['workers']}) != 2
            or metadata['paramMap']['num_workers'] != 2):
        raise ValueError('Incomplete two-worker fit evidence')


def run_phase(phase, cap):
    # Reserve 30 seconds inside the cap for termination and container removal.
    command = COMPOSE + ['exec', '-T', 'driver', 'timeout', '--signal=TERM', '--kill-after=10s',
                        str(cap - 30), '/opt/spark/bin/spark-submit', '--driver-memory', '768m',
                        '/workspace/data/studies/xgb-v1/project/studies/xgb_v1/experiment.py', phase]
    started = time.monotonic()
    samples = []
    with (EVIDENCE / (phase + '.log')).open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= cap - 20:
                    raise TimeoutError(phase + ' supervisor deadline')
                samples.append(dict(seconds=time.monotonic() - started,
                    stats=[json.loads(s) for s in output(['docker', 'stats', '--no-stream',
                           '--format', '{{json .}}', *NAMES], timeout=5).splitlines()]))
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                subprocess.run(COMPOSE + ['down', '--timeout', '5'], check=True, timeout=15)
                process.kill()
                process.wait(timeout=5)
    result = dict(command=command, exit_code=process.returncode,
                  elapsed_seconds=time.monotonic() - started, cap_seconds=cap, samples=samples)
    write(EVIDENCE / (phase + '-execution.json'), result)
    if process.returncode:
        raise RuntimeError(f'{phase} exited {process.returncode}; no automatic retry')
    return result


def main():
    # One exclusive attempt directory also covers failures before the fit.
    OUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(exist_ok=False)
    started = time.monotonic()
    before = protected()
    metadata_before = artifact_metadata()
    write(EVIDENCE / 'artifact-metadata-before.json', metadata_before)
    write(EVIDENCE / 'preservation-before.json', before)
    prod = production()
    assert all(not p['State']['Running'] for p in prod), 'Production workload must not compete'
    write(EVIDENCE / 'runtime-before.json', dict(production=prod, study_image=IMAGE,
          docker_resources=output(['docker', 'info', '--format', '{{json .MemTotal}} {{json .NCPU}}']),
          host_meminfo=Path('/proc/meminfo').read_text()))
    status = dict(status='failed', full_classifier_fits_started=0, full_attempt_cap_seconds=7200,
                  automatic_retries=False, production_publication=False)
    try:
        build_image()
        write(EVIDENCE / 'image.json', dict(tag=IMAGE, image_id=output(
            ['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}'])))
        subprocess.run(COMPOSE + ['up', '-d'], check=True, timeout=120)
        tests = COMPOSE + ['exec', '-T', 'driver', 'python3', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                          '/workspace/data/studies/xgb-v1/project/studies/lr_v1/test_metrics.py', '/workspace/data/studies/xgb-v1/project/studies/xgb_v1/test_experiment.py']
        with (EVIDENCE / 'tests.log').open('w') as log:
            result = subprocess.run(tests, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        write(EVIDENCE / 'tests.json', dict(command=tests, exit_code=result.returncode))
        assert result.returncode == 0
        pip = COMPOSE + ['exec', '-T', 'driver', 'python3', '-m', 'pip', 'check']
        write(EVIDENCE / 'pip-check.json', dict(command=pip, output=output(pip)))
        run_phase('preflight', 1800)
        preflight = json.loads((OUT / 'preflight.json').read_text())
        log = (EVIDENCE / 'preflight.log').read_text()
        if preflight['smoke']['reused_existing_fit']:
            log = (OUT / 'prior/evidence/preflight.log').read_text()
        verify_distributed_fit(log, json.loads((OUT / 'environment.json').read_text()),
                               json.loads((OUT / 'smoke-model/metadata/part-00000').read_text()))
        if preflight['status'] == 'blocked':
            status.update(status='blocked', reason=preflight['memory'])
        else:
            full_start = time.monotonic()
            run_phase('full', 7200)
            status.update(status='complete', full_attempt_elapsed_before_cleanup=time.monotonic() - full_start)
    except BaseException as exc:
        status['error'] = repr(exc)
        raise
    finally:
        if (OUT / 'full/fit-started').exists():
            status['full_classifier_fits_started'] = 1
        cgroups = {}
        for name in NAMES:
            try:
                cgroups[name] = json.loads(output(['docker', 'exec', name, 'python3', '-c',
                    'from pathlib import Path; import json; print(json.dumps({n:(Path("/sys/fs/cgroup")/n).read_text() for n in ["memory.peak","memory.max","memory.events"]}))']))
            except (subprocess.SubprocessError, ValueError) as exc:
                cgroups[name] = dict(error=str(exc))
        write(EVIDENCE / 'cgroups.json', cgroups)
        subprocess.run(COMPOSE + ['down', '--timeout', '5'], check=True, timeout=15)
        after = protected()
        preservation = dict(protected_files=len(before), sha256_unchanged=before == after,
                            production_containers_unchanged=prod == production(),
                            artifact_metadata_files=len(metadata_before),
                            artifact_metadata_unchanged=metadata_before == artifact_metadata(),
                            changed=[p for p in set(before) | set(after) if before.get(p) != after.get(p)])
        write(EVIDENCE / 'preservation.json', preservation)
        status['elapsed_seconds'] = time.monotonic() - started
        write(EVIDENCE / 'status.json', status)
        assert all(preservation[k] for k in ('sha256_unchanged', 'production_containers_unchanged', 'artifact_metadata_unchanged'))
        from report import generate
        generate(ROOT)


if __name__ == '__main__':
    main()

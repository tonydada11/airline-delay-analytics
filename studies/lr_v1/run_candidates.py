"""Host launcher: exactly two sequential attempts, shared budget, no retries."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from studies.lr_v1.candidate_spec import validate_approval, exclusive_candidate_path, candidate_time_limit
from studies.lr_v1.metrics import monthly_comparison, rank_complete_candidates
from studies.lr_v1.preserve import digest, verify

STUDY = Path(__file__).resolve().parents[3]
ROOT = STUDY.parents[2]
IMAGE = 'sha256:662942de790c1cac915e5688953d2ae5669c6f3d5f8d7841639323cd4ecd6abf'


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    approval = json.loads(Path(__file__).with_name('approved-candidates.json').read_text())
    candidates = validate_approval(approval)
    baseline_file = STUDY / 'diagnostics/results.json'
    assert digest(baseline_file) == approval['baseline_results_sha256']
    baseline = json.loads(baseline_file.read_text())['validation_months']
    budget_file = STUDY / 'budget.json'
    budget = json.loads(budget_file.read_text())
    assert budget['classifier_fits_started'] == 0
    assert not any(r.get('stage') == 'candidate' for r in budget['runs']), 'Existing attempts cannot be rerun'
    for candidate in candidates:
        exclusive_candidate_path(STUDY, candidate['name'])
    assert subprocess.check_output(['docker', 'image', 'inspect', 'airline-revision-spark:3.5.9', '--format', '{{.Id}}'], cwd=ROOT, text=True).strip() == IMAGE
    print('Checking all protected baseline files before candidate execution', flush=True)
    verify(STUDY, 'before-candidates')
    logs = STUDY / 'candidate-logs'
    logs.mkdir(exist_ok=False)
    runs = []
    services_started = False
    try:
        services_started = True
        subprocess.run(['docker', 'compose', 'up', '-d', '--no-build', '--pull', 'never'], cwd=ROOT, check=True)
        for spec in candidates:
            name = spec['name']
            hard_limit, execution_limit = candidate_time_limit(budget['active_Spark_seconds_remaining'])
            command = ['docker', 'compose', 'exec', '-T', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                '-e', 'PYTHONPATH=/workspace/data/studies/lr-improvement-v1/project:/workspace/data/studies/lr-improvement-v1/project/src:/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip',
                '-w', '/workspace/data/studies/lr-improvement-v1/project', 'spark-notebook',
                'timeout', '--signal=TERM', '--kill-after=5s', f'{execution_limit}s',
                'python3', '-u', '-m', 'studies.lr_v1.fit_candidate', name]
            record = {'stage': 'candidate', 'name': name, 'order': spec['order'], 'status': 'running',
                'started_at_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
                'maximum_seconds_including_evaluation': hard_limit, 'automatic_retries': False,
                'study_code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=STUDY / 'project', text=True).strip()}
            budget['runs'].append(record)
            write_json(budget_file, budget)
            print(f'Starting {name}; at most {hard_limit}s including preparation and evaluation', flush=True)
            start = time.monotonic()
            try:
                with (logs / (name + '.log')).open('x') as log:
                    completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                exit_code = completed.returncode
            finally:
                elapsed = time.monotonic() - start
                budget['active_Spark_seconds_used'] += elapsed
                budget['active_Spark_seconds_remaining'] = max(0, budget['limit_seconds'] - budget['active_Spark_seconds_used'])
                record.update(elapsed_seconds=elapsed, finished_at_utc=datetime.now(timezone.utc).isoformat())
                write_json(budget_file, budget)
            output = STUDY / 'candidates' / name
            status_path = output / 'status.json'
            state = json.loads(status_path.read_text()) if status_path.exists() else {}
            started_fit = bool(state.get('fit_started'))
            budget['classifier_fits_started'] += int(started_fit)
            record.update(exit_code=exit_code, classifier_fits_started=int(started_fit))
            result_path = output / 'results.json'
            if exit_code == 0 and elapsed <= hard_limit and state.get('status') == 'complete' and result_path.exists():
                result = json.loads(result_path.read_text())
                assert result['comparison'] == monthly_comparison(baseline, result['months'])
                result.update(runtime_seconds_including_evaluation=elapsed, exit_code=exit_code)
                record['status'] = 'complete'
            else:
                result = {'name': name, 'order': spec['order'], 'status': 'incomplete', 'months': {},
                          'runtime_seconds_including_evaluation': elapsed, 'exit_code': exit_code,
                          'fit_started': started_fit, 'reason': state.get('error', 'Timeout, failed command, incomplete evaluation, or time limit exceeded'),
                          'available_months': sorted(p.stem.removeprefix('validation-') for p in output.glob('validation-*.json'))}
                record['status'] = 'incomplete'
            runs.append(result)
            write_json(logs / (name + '-execution.json'), record)
            write_json(budget_file, budget)
            print(json.dumps({'candidate': name, 'status': result['status'], 'seconds': elapsed,
                              'fits_started': budget['classifier_fits_started'],
                              'remaining_seconds': budget['active_Spark_seconds_remaining']}), flush=True)
            assert budget['classifier_fits_started'] <= 2
        assert digest(baseline_file) == approval['baseline_results_sha256']
        ranked = rank_complete_candidates(baseline, runs)
        summary = {'status': 'finished; no retries or additional candidates', 'candidates': runs,
                   'baseline_results_sha256': approval['baseline_results_sha256'],
                   'qualified_ranking': ranked,
                   'study_recommendation': 'retain baseline' if not ranked else 'retrospective qualifier: ' + ranked[0],
                   'production_model': 'accepted LR at 0.39 retained; no promotion',
                   'budget': budget, 'historical_test_used_for_selection': False}
        write_json(STUDY / 'candidate-results.json', summary)
    finally:
        if services_started:
            subprocess.run(['docker', 'compose', 'stop'], cwd=ROOT, check=True)
    print('Checking all protected baseline files after candidate execution', flush=True)
    verify(STUDY, 'after-candidates')
    print('Candidate execution and preservation verification finished', flush=True)


if __name__ == '__main__':
    main()

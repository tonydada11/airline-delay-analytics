"""Export full-precision candidate comparisons and a compact retrospective report."""
import json
from pathlib import Path

from studies.lr_v1.metrics import MONTHS, monthly_comparison, rank_complete_candidates
from studies.lr_v1.preserve import digest

CODE = Path(__file__).resolve().parent
STUDY = CODE.parents[2]


def mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def pp(value):
    return f'{value * 100:+.3f} pp'


def main():
    approval = json.loads((CODE / 'approved-candidates.json').read_text())
    assert digest(STUDY / 'diagnostics/results.json') == approval['baseline_results_sha256']
    baseline = json.loads((STUDY / 'diagnostics/results.json').read_text())['validation_months']
    raw = json.loads((STUDY / 'candidate-results.json').read_text())
    preservation = json.loads((STUDY / 'candidate-preservation-after-candidates.json').read_text())
    assert preservation['status'] == 'passed'
    assert tuple(c['name'] for c in raw['candidates']) == ('lr-positive-weight-1', 'lr-reg-0p3')
    assert raw['qualified_ranking'] == rank_complete_candidates(baseline, raw['candidates'])
    accepted_metadata = json.loads((STUDY.parents[2] / 'artifacts/flighttime-v3/models/classifier/metadata/part-00000').read_text())
    parameter_audit = {}
    for candidate in raw['candidates']:
        if candidate['status'] != 'complete':
            continue
        saved = json.loads((STUDY / 'candidates' / candidate['name'] / 'classifier/metadata/part-00000').read_text())
        differences = {}
        for group in ('paramMap', 'defaultParamMap'):
            differences[group] = {k: [accepted_metadata[group].get(k), saved[group].get(k)]
                                  for k in set(accepted_metadata[group]) | set(saved[group])
                                  if accepted_metadata[group].get(k) != saved[group].get(k)}
        assert differences['defaultParamMap'] == {}
        assert differences['paramMap'] == ({} if candidate['name'] == 'lr-positive-weight-1' else {'regParam': [0.1, 0.3]})
        parameter_audit[candidate['name']] = differences
    summary = {'baseline_months': {m: {k: baseline[m][k] for k in
                                     ('rows', 'prevalence', 'pr_auc', 'budgets', 'duplicate_score_id_check')} for m in MONTHS},
               'baseline_mean_monthly_PR_AUC': mean(list(baseline.values()), 'pr_auc'),
               'baseline_results_sha256': approval['baseline_results_sha256'],
               'candidates': raw['candidates'], 'qualified_ranking': raw['qualified_ranking'],
               'study_recommendation': raw['study_recommendation'], 'production_model': raw['production_model'],
               'budget': raw['budget'], 'preservation': preservation,
               'saved_estimator_parameter_audit': parameter_audit,
               'synthetic_tests': {'passed': 15, 'log': '../../../candidate-unit-tests.log'},
               'historical_test_used_for_selection': False, 'automatic_retries': False}
    for candidate in summary['candidates']:
        if candidate['status'] == 'complete':
            candidate['mean_monthly_budget_changes'] = {
                p: {k: sum(candidate['months'][m]['budgets'][p][k] - baseline[m]['budgets'][p][k] for m in MONTHS)/4
                    for k in ('precision', 'recall')} for p in ('10', '20')}
    lines = ['# Approved LR candidate results', '',
        'The two approved one-factor attempts ran sequentially. Production remains the accepted LR at threshold **0.39**; no model was promoted. All comparisons use full-precision baseline monthly values from the frozen diagnostic JSON, never rounded report figures.', '',
        f'Baseline equal-month PR-AUC: **{summary["baseline_mean_monthly_PR_AUC"]:.9f}**. Paired mean gain must be >=0.005, with mean recall change >=-0.01 at both monthly alert budgets. These are retrospective development-data criteria, not statistical significance or evidence of future improvement.', '',
        '## Completion and compute', '',
        '| Candidate | Status | Full attempt runtime | Fit runtime | Mean PR-AUC | Paired mean gain | Qualifies |',
        '|---|---|---:|---:|---:|---:|---|']
    for candidate in raw['candidates']:
        complete = candidate['status'] == 'complete'
        comparison = monthly_comparison(baseline, candidate['months']) if complete else None
        if complete:
            assert comparison == candidate['comparison']
        lines.append('| {} | {} | {:.3f}s | {} | {} | {} | {} |'.format(
            candidate['name'], candidate['status'], candidate['runtime_seconds_including_evaluation'],
            f'{candidate["fit_elapsed_seconds"]:.3f}s' if complete else 'see attempt log',
            f'{candidate["mean_monthly_PR_AUC"]:.9f}' if complete else 'excluded',
            f'{comparison["mean_monthly_PR_AUC_gain"]:+.9f}' if complete else 'excluded',
            str(comparison['qualifies']) if complete else 'not ranked'))
    budget = raw['budget']
    lines += ['', f'Total active Spark study time charged: **{budget["active_Spark_seconds_used"]:.3f}s**, including the earlier 493.194s diagnostics. Remaining: **{budget["active_Spark_seconds_remaining"]:.3f}s**. Classifier fits started: **{budget["classifier_fits_started"]}**. No automatic retries or additional candidates were run.', '',
        'Each attempt was bounded at 60 minutes including preparation, fitting, save/reload, evaluation, and shutdown; the in-container timeout reserved termination time within that cap. Incomplete attempts are excluded from qualification and ranking regardless of how many monthly files exist.', '',
        '## Baseline monthly comparison values', '',
        '| Month | PR-AUC | Precision 10% | Recall 10% | Precision 20% | Recall 20% |',
        '|---|---:|---:|---:|---:|---:|']
    for month in MONTHS:
        d = baseline[month]
        lines.append(f'| {month} | {d["pr_auc"]:.9f} | {d["budgets"]["10"]["precision"]:.3%} | {d["budgets"]["10"]["recall"]:.3%} | {d["budgets"]["20"]["precision"]:.3%} | {d["budgets"]["20"]["recall"]:.3%} |')
    baseline_means = {p: {k: sum(baseline[m]['budgets'][p][k] for m in MONTHS)/4
                          for k in ('precision', 'recall')} for p in ('10', '20')}
    lines.append(f'| Equal-month mean | {summary["baseline_mean_monthly_PR_AUC"]:.9f} | {baseline_means["10"]["precision"]:.3%} | {baseline_means["10"]["recall"]:.3%} | {baseline_means["20"]["precision"]:.3%} | {baseline_means["20"]["recall"]:.3%} |')
    for candidate in raw['candidates']:
        complete = candidate['status'] == 'complete'
        lines += ['', '## ' + candidate['name'], '']
        if complete:
            months = candidate['months']
            lines += [f'Complete evaluation. Settings: positive-class weight **{candidate["settings"]["positive_class_weight"]}**, regularization **{candidate["settings"]["classifier_settings"]["regParam"]}**. Full training rows: {candidate["training"]["rows"]:,}; optimizer iterations: {candidate["training"]["optimizer_iterations"]}.', '']
        else:
            months = {m: json.loads((STUDY / 'candidates' / candidate['name'] / f'validation-{m}.json').read_text()) for m in candidate['available_months']}
            lines += [f'Incomplete and excluded from ranking. Reason: {candidate["reason"]}. Exit: {candidate["exit_code"]}. Any monthly values below are partial observations; no candidate mean or qualification is calculated.', '']
        lines += ['| Month | PR-AUC | Baseline PR-AUC | Difference |', '|---|---:|---:|---:|']
        for m, d in months.items():
            lines.append(f'| {m} | {d["pr_auc"]:.9f} | {baseline[m]["pr_auc"]:.9f} | {d["pr_auc"]-baseline[m]["pr_auc"]:+.9f} |')
        if complete:
            lines.append(f'| Equal-month mean | {candidate["mean_monthly_PR_AUC"]:.9f} | {summary["baseline_mean_monthly_PR_AUC"]:.9f} | {candidate["comparison"]["mean_monthly_PR_AUC_gain"]:+.9f} |')
        for percent in ('10', '20'):
            lines += ['', f'### Top {percent}% within each calendar month', '',
                '| Month | Precision | Precision difference | Recall | Recall difference |',
                '|---|---:|---:|---:|---:|']
            for m, d in months.items():
                value, base = d['budgets'][percent], baseline[m]['budgets'][percent]
                lines.append(f'| {m} | {value["precision"]:.3%} | {pp(value["precision"]-base["precision"])} | {value["recall"]:.3%} | {pp(value["recall"]-base["recall"])} |')
            if complete:
                values = candidate['mean_monthly_budget_metrics'][percent]
                differences = {k: sum(months[m]['budgets'][percent][k]-baseline[m]['budgets'][percent][k] for m in MONTHS)/4 for k in ('precision', 'recall')}
                lines.append(f'| Equal-month mean | {values["precision"]:.3%} | {pp(differences["precision"])} | {values["recall"]:.3%} | {pp(differences["recall"])} |')
        if complete:
            changes = candidate['comparison']['monthly_PR_AUC_changes']
            worst = min(changes, key=changes.get)
            lines += ['', f'Worst monthly PR-AUC change: **{changes[worst]:+.9f}** in **{worst}**. Mean recall constraints do not guarantee that every month improves; no extra monthly elimination rule was added.']
            for percent in ('10', '20'):
                recall_changes = {m: months[m]['budgets'][percent]['recall'] - baseline[m]['budgets'][percent]['recall'] for m in MONTHS}
                worst_recall = min(recall_changes, key=recall_changes.get)
                lines.append(f'Smallest monthly recall change at {percent}%: **{pp(recall_changes[worst_recall])}** in **{worst_recall}**.')
    lines += ['', '## Decision and verification', '',
        '**Study recommendation: ' + raw['study_recommendation'] + '.** Production remains the accepted LR at 0.39 even if an experimental candidate qualifies. No September-December 2024 evaluation was used for selection, and no later untouched evaluation period is available.', '',
        '- The saved accepted preprocessing, 50 input features / 784 prepared dimensions, full January 2023-April 2024 training population, May-August 2024 validation population, remaining estimator settings, and two-core/768 MiB driver/640 MiB executor runtime were retained. No window experiment, combined settings, additional candidate, preprocessing fit, or production publication was executed.',
        '- Each complete candidate passed saved-model parameter/dimension checks, full-validation score multiset equality across serialization/reload, and saved prediction population/threshold validation. Metrics are unweighted, natural-prevalence PR-AUC using the preregistered evaluator and per-month top-10%/20% capacities. Score/ID boundary ties retain multiplicity and receive equal fractional mass.',
        '- Direct comparison of saved classifier metadata confirmed identical default parameters for every complete candidate. Candidate 1 has no classifier parameter-map changes (the approved class weight is applied to training rows); candidate 2 changes only regParam from 0.1 to 0.3.',
        '- Fifteen synthetic metric/approval/output-boundary/budget tests passed before fitting. The candidate runner was committed before execution; exact commits, commands, exit codes, timings, and limits appear in the [budget ledger](../../../budget.json).',
        '- [Before](../../../candidate-preservation-before-candidates.json) and [after](../../../candidate-preservation-after-candidates.json) checks confirmed all 700 protected files unchanged, including raw data, baseline/current artifacts, feature datasets, and B1. Source/runtime acceptance identities, existing tags, and the main cleanup commit remain unchanged.',
        '- Services were stopped afterward. Generated models and prediction Parquet remain separately under the ignored study artifact directory; code, compact results, and this report are committed on the study branch. Nothing was pushed.',
        '- [Full-precision compact results](candidate-results-summary.json) and [raw execution results](../../../candidate-results.json) retain every monthly value and comparison. Display rounding in this report does not enter qualification calculations. The original [diagnostic report](REPORT.md) describes the pre-candidate stage.', '']
    (CODE / 'candidate-results-summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    (CODE / 'CANDIDATE_RESULTS.md').write_text('\n'.join(lines))
    print(raw['study_recommendation'])


if __name__ == '__main__':
    main()

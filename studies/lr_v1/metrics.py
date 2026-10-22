"""Small, label-blind capacity calculations and preregistered study comparisons."""
from collections import defaultdict
from fractions import Fraction
import math

MONTHS = ('2024-05', '2024-06', '2024-07', '2024-08')


def capacity(rows, percent):
    return rows * percent // 100


def expected_budget(rows, percent):
    """Reference calculation; tuples are (score, flight_id, binary label).

    Labels contribute to metrics only. Indistinguishable boundary rows receive
    equal selection mass, so duplicates are retained without inventing an order.
    """
    groups = defaultdict(lambda: [0, 0])
    for score, flight_id, label in rows:
        groups[(score, flight_id)][0] += 1
        groups[(score, flight_id)][1] += label
    k = capacity(len(rows), percent)
    seen, tp, selected, fractional = 0, Fraction(0), Fraction(0), 0
    for key in sorted(groups, key=lambda key: (-key[0], key[1])):
        n, positives = groups[key]
        take = max(0, min(n, k - seen))
        weight = Fraction(take, n)
        tp += weight * positives
        selected += weight * n
        fractional += int(0 < weight < 1)
        seen += n
    positives = sum(label for _, _, label in rows)
    return {'capacity': k, 'selected_mass': float(selected), 'true_positive_mass': float(tp),
            'precision': float(tp / selected) if selected else 0.0,
            'recall': float(tp / positives) if positives else 0.0,
            'fractional_boundary_groups': fractional}


def monthly_comparison(baseline, candidate):
    """All four months are required; pooled PR-AUC is never an input."""
    if set(baseline) != set(MONTHS) or set(candidate) != set(MONTHS):
        raise ValueError('A complete four-month evaluation is required')
    for results in (baseline, candidate):
        if any(not math.isfinite(results[m]['pr_auc']) for m in MONTHS):
            raise ValueError('Every monthly PR-AUC must be defined')
    changes = {m: candidate[m]['pr_auc'] - baseline[m]['pr_auc'] for m in MONTHS}
    delta = sum(changes.values()) / len(MONTHS)
    recall_changes = {str(p): sum(candidate[m]['budgets'][str(p)]['recall'] -
                                    baseline[m]['budgets'][str(p)]['recall'] for m in MONTHS) / len(MONTHS)
                      for p in (10, 20)}
    return {'mean_monthly_PR_AUC_gain': delta, 'monthly_PR_AUC_changes': changes,
            'mean_monthly_recall_changes': recall_changes,
            'worst_monthly_PR_AUC_change': min(changes.values()),
            'qualifies': delta >= 0.005 and all(v >= -0.01 for v in recall_changes.values())}


def rank_complete_candidates(baseline, candidates):
    eligible = []
    for candidate in candidates:
        if candidate['status'] != 'complete':
            continue
        comparison = monthly_comparison(baseline, candidate['months'])
        if comparison['qualifies']:
            months = candidate['months']
            key = (-sum(months[m]['pr_auc'] for m in MONTHS) / len(MONTHS),
                   -sum(months[m]['budgets']['10']['recall'] for m in MONTHS) / len(MONTHS),
                   -sum(months[m]['budgets']['20']['recall'] for m in MONTHS) / len(MONTHS),
                   candidate['order'])
            eligible.append((key, candidate['name']))
    return [name for _, name in sorted(eligible)]

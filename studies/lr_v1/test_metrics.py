import pytest

from studies.lr_v1.metrics import (MONTHS, capacity, expected_budget,
                                   monthly_comparison, rank_complete_candidates)


def months(pr_auc=0.4, recall=0.3):
    return {m: {'pr_auc': pr_auc, 'budgets': {'10': {'recall': recall},
                                            '20': {'recall': recall + 0.1}}} for m in MONTHS}


def test_budget_floor_and_empty_capacity():
    assert capacity(19, 10) == 1
    result = expected_budget([(0.5, 'a', 1)], 10)
    assert result['selected_mass'] == 0 and result['precision'] == 0


def test_identical_score_id_boundary_preserves_multiplicity():
    rows = [(0.9, 'same', 1), (0.9, 'same', 0), *[(0.1, str(i), 0) for i in range(8)]]
    result = expected_budget(rows, 10)
    assert result['capacity'] == 1 and result['selected_mass'] == 1
    assert result['true_positive_mass'] == 0.5
    assert result['precision'] == result['recall'] == 0.5
    assert result['fractional_boundary_groups'] == 1


def test_flight_id_breaks_score_ties_without_using_labels():
    rows = [(0.9, 'b', 1), (0.9, 'a', 0), *[(0.1, str(i), 0) for i in range(8)]]
    result = expected_budget(rows, 10)
    assert result['true_positive_mass'] == 0
    assert result['fractional_boundary_groups'] == 0


def test_months_have_equal_weight_and_comparison_is_paired():
    baseline = months()
    baseline[MONTHS[0]]['pr_auc'] = 0.1
    candidate = months(0.42)
    result = monthly_comparison(baseline, candidate)
    assert result['mean_monthly_PR_AUC_gain'] == pytest.approx((0.32 + 0.02 * 3) / 4)
    assert result['qualifies']


def test_incomplete_or_undefined_months_cannot_be_compared():
    baseline, candidate = months(), months(0.42)
    del candidate[MONTHS[-1]]
    with pytest.raises(ValueError):
        monthly_comparison(baseline, candidate)
    candidate = months(float('nan'))
    with pytest.raises(ValueError):
        monthly_comparison(baseline, candidate)


def test_gain_and_recall_constraints_without_new_monthly_rule():
    baseline = months()
    assert not monthly_comparison(baseline, months(0.404))['qualifies']
    assert not monthly_comparison(baseline, months(0.42, 0.28))['qualifies']
    candidate = months(0.42)
    candidate[MONTHS[0]]['pr_auc'] = 0.38
    assert monthly_comparison(baseline, candidate)['qualifies']


def test_only_complete_qualifying_candidates_are_ranked():
    baseline = months()
    candidates = [{'name': 'interrupted', 'status': 'incomplete', 'months': {}, 'order': 0},
                  {'name': 'small', 'status': 'complete', 'months': months(0.402), 'order': 1},
                  {'name': 'later', 'status': 'complete', 'months': months(0.42), 'order': 3},
                  {'name': 'earlier', 'status': 'complete', 'months': months(0.42), 'order': 2}]
    assert rank_complete_candidates(baseline, candidates) == ['earlier', 'later']

# AI-generated test code: generated with OpenAI Codex.
from dataclasses import replace

import pytest

from airline.ml.experiment_policy import (
    BACKTESTS, Candidate, baseline_candidates, choose_f1_candidate,
    choose_ranking_candidate, feature_gate, tuning_candidates, training_partition_count,
)


def panel(a, b):
    return {"logistic_regression": a, "gbt": b}


def test_gate_requires_both_folds_to_improve_or_hold():
    before = panel([0.3, 0.4], [0.2, 0.3])
    assert feature_gate(before, panel([0.31, 0.4], [0.21, 0.3]))["accepted"]
    assert not feature_gate(before, panel([0.4, 0.39], [0.3, 0.29]))["accepted"]
    assert not feature_gate(before, before)["accepted"]
    with pytest.raises(ValueError):
        feature_gate(before, {"gbt": [0.4, 0.5]})


def test_ranking_and_formal_f1_selection_are_separate():
    lr, gbt = baseline_candidates("schedule")
    assert choose_ranking_candidate([(lr, [0.3, 0.4]), (replace(lr, iterations=100), [0.3, 0.4])]) == lr
    assert choose_f1_candidate([(lr, {"f1": 0.3}), (gbt, {"f1": 0.4})]) == gbt
    assert choose_f1_candidate([(lr, {"f1": 0.4}), (gbt, {"f1": 0.4})]) == lr
    assert choose_f1_candidate([(replace(lr, positive_weight=2), {"f1": 0.4}), (lr, {"f1": 0.4})]) == lr


def test_fit_budget_and_nonoverlapping_backtests():
    # Baseline/schedule/history: 12. Tuning: 6. Weighting: 4. Final refit: 1.
    fits = 3 * len(baseline_candidates("baseline")) * len(BACKTESTS)
    fits += sum(len(tuning_candidates("history", family)) for family in ("logistic_regression", "gbt")) * len(BACKTESTS)
    fits += 2 * len(BACKTESTS) + 1
    assert fits == 23
    for fold in BACKTESTS:
        assert fold.train_start <= fold.train_end < fold.evaluation_start <= fold.evaluation_end
    assert BACKTESTS[0].evaluation_end < BACKTESTS[1].evaluation_start


def test_candidate_identifiers_include_feature_parameters_and_weights():
    base = Candidate("baseline", "logistic_regression", 20)
    assert len({base.key, replace(base, bundle="schedule").key,
                replace(base, positive_weight=2).key, replace(base, regularization=0.01).key}) == 4


def test_training_partitions_bound_average_vector_rows_without_reducing_data():
    assert training_partition_count(48) == 2
    assert training_partition_count(500_000) == 2
    assert training_partition_count(500_001) == 3
    assert training_partition_count(8_946_370) == 36

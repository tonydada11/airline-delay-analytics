# AI-generated test code: generated with OpenAI Codex.
import inspect

from airline.ml import train_models
from airline.ml.m7_contract import (
    CLASSIFICATION_THRESHOLD,
    FEATURE_COLUMNS,
    FEATURE_CONTRACT_VERSION,
    MAX_ITER,
    MODEL_FAMILY,
    POSITIVE_CLASS_WEIGHT,
    REG_PARAM,
    SPLIT_BOUNDARIES,
    assert_frozen_contract,
)


def test_frozen_model_contract_is_exact():
    assert_frozen_contract()
    assert FEATURE_CONTRACT_VERSION == "m7-history-v1"
    assert MODEL_FAMILY == "logistic_regression"
    assert MAX_ITER == 100
    assert REG_PARAM == 0.1
    assert POSITIVE_CLASS_WEIGHT == 2.0
    assert CLASSIFICATION_THRESHOLD == 0.34
    assert SPLIT_BOUNDARIES == {
        "train": {"start": "2023-01-01", "end": "2024-04-30"},
        "official_validation": {"start": "2024-05-01", "end": "2024-08-31"},
        "protected_test": {"start": "2024-09-01", "end": "2024-12-31"},
    }
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_production_entrypoint_cannot_automatically_evaluate_test():
    source = inspect.getsource(train_models.run_training)
    assert "evaluate_final_test(" not in source
    assert "finalize(spark, project_root)" in source

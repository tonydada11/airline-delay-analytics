"""Frozen M7 feature, chronology, and model contract.

This module is deliberately free of Spark actions so the contract can be
validated by unit tests without starting a cluster.
"""

from airline.dws.build_ml_flight_features import (
    LEAKAGE_BLACKLIST,
    TARGET_COLUMN,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)
from airline.ml.experiment_features import (
    ENTITIES,
    HISTORY_COUNTS,
    HISTORY_MISSING,
    HISTORY_RATES,
    SCHEDULE_CATEGORICAL,
    SCHEDULE_NUMERIC,
    WINDOW_DAYS,
)


FEATURE_CONTRACT_VERSION = "m7-history-v1"
MODEL_FAMILY = "logistic_regression"
MAX_ITER = 100
REG_PARAM = 0.1
ELASTIC_NET_PARAM = 0.0
POSITIVE_CLASS_WEIGHT = 2.0
CLASSIFICATION_THRESHOLD = 0.34
SEED = 42

CATEGORICAL_COLUMNS = SCHEDULE_CATEGORICAL
NUMERIC_COLUMNS = (*SCHEDULE_NUMERIC, *HISTORY_RATES)
COUNT_COLUMNS = HISTORY_COUNTS
MISSING_INDICATOR_COLUMNS = HISTORY_MISSING
FEATURE_COLUMNS = (
    *CATEGORICAL_COLUMNS,
    *NUMERIC_COLUMNS,
    *COUNT_COLUMNS,
    *MISSING_INDICATOR_COLUMNS,
)
OUTPUT_COLUMNS = ("flight_date", *FEATURE_COLUMNS, TARGET_COLUMN, "split")

HISTORICAL_DEFINITIONS = {
    "entities": {name: list(columns) for name, columns in ENTITIES},
    "windows_days": list(WINDOW_DAYS),
    "lower_bound": "D-N calendar days, inclusive",
    "upper_bound": "D-1 calendar day, inclusive",
    "same_day_allowed": False,
    "future_allowed": False,
    "rates": list(HISTORY_RATES),
    "counts": list(HISTORY_COUNTS),
    "missing_indicators": list(HISTORY_MISSING),
}

SPLIT_BOUNDARIES = {
    "train": {"start": TRAIN_START.isoformat(), "end": TRAIN_END.isoformat()},
    "official_validation": {
        "start": VALIDATION_START.isoformat(),
        "end": VALIDATION_END.isoformat(),
    },
    "protected_test": {"start": TEST_START.isoformat(), "end": TEST_END.isoformat()},
}


def assert_frozen_contract() -> None:
    """Fail fast if a future edit silently changes the frozen selection."""

    if FEATURE_CONTRACT_VERSION != "m7-history-v1":
        raise AssertionError("Unexpected frozen feature-contract version.")
    if MODEL_FAMILY != "logistic_regression":
        raise AssertionError("Frozen model must be Logistic Regression.")
    if (MAX_ITER, REG_PARAM, POSITIVE_CLASS_WEIGHT, CLASSIFICATION_THRESHOLD) != (
        100,
        0.1,
        2.0,
        0.34,
    ):
        raise AssertionError("Frozen M7 model settings changed.")
    if set(FEATURE_COLUMNS) & LEAKAGE_BLACKLIST:
        raise AssertionError("Frozen feature contract contains leakage columns.")

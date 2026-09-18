"""Small reusable DataFrame quality assertions."""

from typing import Iterable

from pyspark.sql import DataFrame


def assert_not_empty(dataframe: DataFrame) -> None:
    """Assert that a DataFrame contains at least one row."""

    if not dataframe.take(1):
        raise AssertionError("Expected DataFrame to contain at least one row.")


def assert_required_columns(dataframe: DataFrame, required_columns: Iterable[str]) -> None:
    """Assert that all required column names are present."""

    missing = sorted(set(required_columns) - set(dataframe.columns))
    if missing:
        raise AssertionError(f"DataFrame is missing required columns: {missing}")


def assert_row_count(dataframe: DataFrame, expected_count: int) -> None:
    """Assert that a DataFrame has exactly ``expected_count`` rows."""

    actual_count = dataframe.count()
    if actual_count != expected_count:
        raise AssertionError(
            f"Expected DataFrame row count {expected_count}, found {actual_count}."
        )

# AI-generated test code: generated with OpenAI Codex.
"""Fast contract tests for the M9 performance runner."""

from pathlib import Path

import pytest

from airline.performance.run_experiments import (
    alternating_order,
    median_runtime,
    validate_output_root,
)


def test_output_root_must_be_named_performance_directory(tmp_path: Path) -> None:
    expected = (tmp_path / "artifacts/performance/m9").resolve()
    assert validate_output_root(Path("artifacts/performance/m9"), tmp_path) == expected

    invalid = (
        Path("artifacts/performance"),
        Path("artifacts/experiments/m9"),
        Path("../performance/m9"),
        expected,
    )
    for candidate in invalid:
        with pytest.raises(ValueError):
            validate_output_root(candidate, tmp_path)


def test_alternating_order_reverses_even_repetitions() -> None:
    variants = ["one", "two", "three"]
    assert alternating_order(variants, 1) == variants
    assert alternating_order(variants, 2) == list(reversed(variants))
    assert alternating_order(variants, 3) == variants
    with pytest.raises(ValueError):
        alternating_order(variants, 0)


def test_median_runtime_uses_all_measured_runs() -> None:
    assert median_runtime([{"elapsed_seconds": 9}, {"elapsed_seconds": 3}]) == 6.0
    assert median_runtime(
        [
            {"elapsed_seconds": 7.0},
            {"elapsed_seconds": 1.0},
            {"elapsed_seconds": 4.0},
        ]
    ) == 4.0
    with pytest.raises(ValueError):
        median_runtime([])

# AI-generated test code: generated with OpenAI Codex.
"""Fast contract tests for the M9 performance runner."""

from datetime import date
import json
from pathlib import Path
import shutil
from unittest.mock import Mock, patch

import pytest

from airline.performance.run_experiments import (
    PerformanceRunner,
    alternating_order,
    median_runtime,
    validate_output_root,
)


@pytest.fixture
def runner(tmp_path: Path) -> PerformanceRunner:
    """Load the M9 runner for unit tests."""
    spark = Mock()
    spark.version = "3.5.9"
    spark.sparkContext.applicationId = "m9-unit-test"
    spark.conf.get.return_value = None
    return PerformanceRunner(
        spark=spark,
        project_root=tmp_path,
        dwd_path=tmp_path / "warehouse/dwd",
        carrier_dim_path=tmp_path / "warehouse/dim_carrier",
        route_daily_path=tmp_path / "warehouse/route_daily",
        output=tmp_path / "artifacts/performance/m9",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 4, 30),
        repetitions=1,
    )


def test_prepare_removes_completion_marker_before_replacing_evidence(
    runner: PerformanceRunner,
) -> None:
    """Exercise the prepare removes completion marker before replacing evidence case."""
    runner.output.mkdir(parents=True)
    marker = runner.output / "complete.json"
    marker.write_text('{"status": "PASS"}', encoding="utf-8")
    for name in ("runs", "plans", "work"):
        target = runner.output / name
        target.mkdir()
        (target / "old-evidence.txt").write_text("old evidence", encoding="utf-8")
    manifest = runner.output / "manifest.json"
    manifest.write_text('{"old": true}', encoding="utf-8")
    rmtree = shutil.rmtree

    def remove_evidence(target: Path) -> None:
        """Remove evidence."""
        assert not marker.exists()
        rmtree(target)

    with patch(
        "airline.performance.run_experiments.shutil.rmtree",
        side_effect=remove_evidence,
    ) as remove:
        runner.prepare()

    assert [call.args[0].name for call in remove.call_args_list] == [
        "runs", "plans", "work"
    ]
    assert not marker.exists()
    assert not (runner.output / "runs").exists()
    assert not (runner.output / "plans").exists()
    assert list(runner.work.iterdir()) == []
    assert json.loads(manifest.read_text(encoding="utf-8"))["schema_version"] == 1


@pytest.mark.parametrize("failure_stage", ["cleanup", "warmup"])
def test_interrupted_rerun_leaves_no_completion_marker(
    runner: PerformanceRunner, failure_stage: str
) -> None:
    """Exercise the interrupted rerun leaves no completion marker case."""
    runner.output.mkdir(parents=True)
    marker = runner.output / "complete.json"
    marker.write_text('{"status": "PASS"}', encoding="utf-8")
    (runner.output / "runs").mkdir()
    failure = RuntimeError("simulated rerun failure")
    target = (
        "airline.performance.run_experiments.shutil.rmtree"
        if failure_stage == "cleanup"
        else "airline.performance.run_experiments.PerformanceRunner.warm_route_workload"
    )
    with patch(target, side_effect=failure) as fail:
        with pytest.raises(RuntimeError, match="simulated rerun failure"):
            runner.run()

    fail.assert_called_once()
    assert not marker.exists()
    if failure_stage == "warmup":
        assert (runner.output / "manifest.json").is_file()


def test_prepare_without_previous_completion_marker(runner: PerformanceRunner) -> None:
    """Exercise the prepare without previous completion marker case."""
    runner.prepare()

    assert runner.work.is_dir()
    assert (runner.output / "manifest.json").is_file()
    assert not (runner.output / "complete.json").exists()


def test_output_root_must_be_named_performance_directory(tmp_path: Path) -> None:
    """Exercise the output root must be named performance directory case."""
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
    """Exercise the alternating order reverses even repetitions case."""
    variants = ["one", "two", "three"]
    assert alternating_order(variants, 1) == variants
    assert alternating_order(variants, 2) == list(reversed(variants))
    assert alternating_order(variants, 3) == variants
    with pytest.raises(ValueError):
        alternating_order(variants, 0)


def test_median_runtime_uses_all_measured_runs() -> None:
    """Exercise the median runtime uses all measured runs case."""
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

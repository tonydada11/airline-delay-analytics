# AI-generated test code: generated with OpenAI Codex.
"""Acceptance checks for generated M9 Spark performance evidence."""

import json
import os
from pathlib import Path

import pytest

from airline.common.config import load_config
from airline.common.paths import get_project_paths


pytestmark = pytest.mark.skipif(
    os.environ.get("AIRLINE_RUN_FULL_M9_QUALITY") != "1",
    reason="set AIRLINE_RUN_FULL_M9_QUALITY=1 for mandatory M9 gates",
)

EXPECTED_VARIANTS = {
    "repartition_vs_coalesce": {"repartition_2", "coalesce_2"},
    "partition_count": {"1", "2", "4", "8"},
    "join_strategy": {"broadcast", "non_broadcast"},
    "cache_behavior": {"no_cache", "cache"},
    "storage_level": {"memory_only", "memory_and_disk"},
    "shuffle_partitions": {"2", "4", "8"},
}


def _read_json(path: Path) -> dict:
    """Read json."""
    return json.loads(path.read_text(encoding="utf-8"))


def test_m9_manifest_summary_and_restoration_are_complete() -> None:
    """Exercise the m9 manifest summary and restoration are complete case."""
    root = get_project_paths(load_config()).project_root / "artifacts/performance/m9"
    manifest = _read_json(root / "manifest.json")
    summary = _read_json(root / "summary.json")
    completion = _read_json(root / "complete.json")
    restore = _read_json(root / "configuration_restore.json")

    assert completion["status"] == "PASS"
    assert summary["status"] == "PASS"
    assert manifest["scope"] == {
        "start_date": "2023-01-01",
        "end_date": "2023-04-30",
        "repetitions": 2,
        "warmups_per_workload": 1,
    }
    assert set(summary["experiments"]) == set(EXPECTED_VARIANTS)
    assert summary["baseline"]["logical_profile"]["rows"] > 0
    assert summary["baseline"]["logical_profile"]["duplicate_grains"] == 0
    assert summary["baseline"]["runtime"]["available"] is True
    assert summary["baseline"]["runtime"]["numTasks"] > 0
    assert restore["configuration_restored"] is True
    assert restore["persistence_cleared"] is True
    assert restore["persistent_rdd_count"] == 0
    assert summary["production_change_retained"] is False
    assert summary["protected_test_evaluation_executed"] is False
    assert manifest["protected_test_evaluation_executed"] is False
    assert not (root / "work").exists()
    assert (root / "summary.csv").is_file()


def test_every_variant_has_two_correct_runs_with_runtime_and_plan_evidence() -> None:
    """Exercise the every variant has two correct runs with runtime and plan evidence case."""
    root = get_project_paths(load_config()).project_root / "artifacts/performance/m9"
    summary = _read_json(root / "summary.json")
    for experiment, expected_variants in EXPECTED_VARIANTS.items():
        variants = summary["experiments"][experiment]["variants"]
        assert set(variants) == expected_variants
        for variant in expected_variants:
            aggregate = variants[variant]
            assert aggregate["run_count"] == 2
            assert len(aggregate["runs_seconds"]) == 2
            assert aggregate["median_seconds"] > 0
            plan = root / "plans" / experiment / f"{variant}.formatted.txt"
            assert plan.is_file()
            assert "Physical Plan" in plan.read_text(encoding="utf-8")
            for repetition in (1, 2):
                run = _read_json(
                    root
                    / "runs"
                    / experiment
                    / variant
                    / f"run-{repetition}.json"
                )
                assert run["status"] == "PASS"
                assert run["elapsed_seconds"] > 0
                assert run["runtime"]["available"] is True
                assert run["runtime"]["job_count"] > 0
                assert run["runtime"]["stage_count"] > 0
                assert run["runtime"]["numTasks"] > 0
                assert all(
                    "executorRunTime" in stage["task_summary"]
                    for stage in run["runtime"]["stages"]
                )
                assert run["logical_differences"] == {
                    "candidate_minus_baseline": 0,
                    "baseline_minus_candidate": 0,
                }
                assert run["executed_plan"]


def test_plan_operators_and_protected_test_exclusion() -> None:
    """Exercise the plan operators and protected test exclusion case."""
    paths = get_project_paths(load_config())
    root = paths.project_root / "artifacts/performance/m9"
    plans = root / "plans"
    repartition = (
        plans / "repartition_vs_coalesce/repartition_2.formatted.txt"
    ).read_text()
    coalesce = (plans / "repartition_vs_coalesce/coalesce_2.formatted.txt").read_text()
    broadcast = (plans / "join_strategy/broadcast.formatted.txt").read_text()
    non_broadcast = (plans / "join_strategy/non_broadcast.formatted.txt").read_text()
    runner_source = (
        paths.project_root / "src/airline/performance/run_experiments.py"
    ).read_text(encoding="utf-8")

    assert "Exchange" in repartition
    assert "Coalesce" in coalesce
    assert "BroadcastHashJoin" in broadcast
    assert "SortMergeJoin" in non_broadcast
    for forbidden in (
        "airline.ml",
        "train_models",
        "evaluate_final_test",
        "artifacts/predictions/final_test",
        "artifacts/metrics/final_test.json",
    ):
        assert forbidden not in runner_source

    report = root / "report.md"
    assert report.is_file()
    content = report.read_text(encoding="utf-8")
    assert "Status: **PASS**" in content
    assert "production change was" in content
    assert "retained" in content

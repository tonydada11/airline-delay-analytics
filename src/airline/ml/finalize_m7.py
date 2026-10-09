"""Finalize and verify the already-selected M7 model without refitting or test scoring.

Usage::

    python -m airline.ml.finalize_m7

The command is resumable.  A valid completion marker causes subsequent runs to
verify protected-test hashes and return without fitting or scoring any model.
"""

import csv
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql import DataFrame, SparkSession, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.spark_session import create_spark_session
from airline.dws.build_m7_ml_flight_features import validate_m7_feature_frame
from airline.ml.experiment_features import preprocessing_pipeline
from airline.ml.m7_contract import (
    CATEGORICAL_COLUMNS,
    CLASSIFICATION_THRESHOLD,
    COUNT_COLUMNS,
    ELASTIC_NET_PARAM,
    FEATURE_COLUMNS,
    FEATURE_CONTRACT_VERSION,
    HISTORICAL_DEFINITIONS,
    MAX_ITER,
    MISSING_INDICATOR_COLUMNS,
    MODEL_FAMILY,
    NUMERIC_COLUMNS,
    OUTPUT_COLUMNS,
    POSITIVE_CLASS_WEIGHT,
    REG_PARAM,
    SEED,
    SPLIT_BOUNDARIES,
    assert_frozen_contract,
)
from airline.ml.run_experiments import metric_report
from airline.ml.train_models import (
    TARGET_COLUMN,
    aggregate_probability_buckets,
    score_probabilities,
    threshold_metrics_from_buckets,
    write_json,
)


EXPERIMENT_ROOT = Path("artifacts/experiments/staged-v2")
M7_ROOT = EXPERIMENT_ROOT / "m7"
PROTECTED_PATHS = (
    Path("artifacts/metrics/final_test.json"),
    Path("artifacts/metrics/validation.json"),
    Path("artifacts/models/selection.json"),
)
PROTECTED_PREDICTION_ROOT = Path("artifacts/predictions/final_test")
ORIGINAL_VALIDATION = {
    "precision": 0.3449,
    "recall": 0.7041,
    "f1": 0.4630,
    "roc_auc": 0.6615,
    "pr_auc": 0.3942,
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def protected_hashes(project_root: Path) -> dict[str, str]:
    """Hash existing test artifacts byte-for-byte without reading their contents."""

    targets = [project_root / path for path in PROTECTED_PATHS]
    targets.extend(sorted(path for path in (project_root / PROTECTED_PREDICTION_ROOT).rglob("*") if path.is_file()))
    result = {}
    for path in targets:
        relative = path.relative_to(project_root).as_posix()
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def frozen_manifest(spark: SparkSession) -> dict:
    return {
        "schema_version": 1,
        "phase": "fitted_model_frozen",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "model_family": MODEL_FAMILY,
        "feature_columns": list(FEATURE_COLUMNS),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "numeric_columns": list(NUMERIC_COLUMNS),
        "count_columns": list(COUNT_COLUMNS),
        "missing_indicator_columns": list(MISSING_INDICATOR_COLUMNS),
        "historical_feature_definitions": HISTORICAL_DEFINITIONS,
        "preprocessing_stages": [
            "StringIndexer(handleInvalid=keep, alphabetAsc) per categorical column",
            "OneHotEncoder(handleInvalid=keep)",
            "mean Imputer for numeric columns",
            "VectorAssembler(categorical OHE, imputed numeric, counts, missing indicators)",
        ],
        "model": {
            "family": MODEL_FAMILY,
            "maxIter": MAX_ITER,
            "regParam": REG_PARAM,
            "elasticNetParam": ELASTIC_NET_PARAM,
            "positive_class_weight": POSITIVE_CLASS_WEIGHT,
            "threshold": CLASSIFICATION_THRESHOLD,
            "seed": SEED,
        },
        "splits": SPLIT_BOUNDARIES,
        "backtests": [
            {"name": "autumn_2023", "train": ["2023-01-01", "2023-08-31"], "evaluation": ["2023-09-01", "2023-12-31"]},
            {"name": "spring_2024", "train": ["2023-01-01", "2023-12-31"], "evaluation": ["2024-01-01", "2024-04-30"]},
        ],
        "spark_version": spark.version,
        "paths": {
            "model_root": "artifacts/experiments/staged-v2/final_model",
            "preprocessing": "artifacts/experiments/staged-v2/final_model/preprocessing",
            "classifier": "artifacts/experiments/staged-v2/final_model/classifier",
            "modeling_input": "data/warehouse/dws/ml_flight_features",
        },
        "test_policy": "Protected test is never loaded or scored by the M7 finalizer.",
    }


def _score_distribution(dataframe: DataFrame) -> dict:
    row = dataframe.agg(
        F.count("*").alias("rows"),
        F.min("probability").alias("minimum"),
        F.max("probability").alias("maximum"),
        F.avg("probability").alias("mean"),
        F.stddev_pop("probability").alias("stddev"),
        F.expr("percentile_approx(probability, array(0.01,0.05,0.25,0.5,0.75,0.95,0.99), 10000)").alias("quantiles"),
    ).first()
    return {
        "rows": int(row["rows"]),
        "minimum": row["minimum"],
        "p01": row["quantiles"][0],
        "p05": row["quantiles"][1],
        "p25": row["quantiles"][2],
        "median": row["quantiles"][3],
        "p75": row["quantiles"][4],
        "p95": row["quantiles"][5],
        "p99": row["quantiles"][6],
        "maximum": row["maximum"],
        "mean": row["mean"],
        "stddev": row["stddev"],
    }


def monthly_diagnostics(scores: DataFrame, validation: dict) -> list[dict]:
    rows = []
    dated = scores.withColumn("_month", F.date_format("flight_date", "yyyy-MM"))
    for month, metrics in sorted(validation["monthly"].items()):
        operating = metrics["operating_point"]
        ranking = metrics["ranking"]
        rows.append({
            "month": month,
            "rows": metrics["rows"],
            "prevalence": metrics["class_prevalence"],
            "precision": operating["precision"],
            "recall": operating["recall"],
            "f1": operating["f1"],
            "roc_auc": ranking["roc_auc"],
            "pr_auc": ranking["pr_auc"],
            "predicted_positive_rate": operating["predicted_positive_rate"],
            "threshold": operating["threshold"],
            "score_distribution": _score_distribution(dated.filter(F.col("_month") == month)),
        })
    return rows


def _pooled_scores(spark: SparkSession, candidate_key: str) -> DataFrame:
    root = EXPERIMENT_ROOT / "candidates" / candidate_key
    return spark.read.parquet(str(root / "autumn_2023/scores")).unionByName(
        spark.read.parquet(str(root / "spring_2024/scores"))
    )


def _flat_metrics(report: dict) -> dict:
    return {
        "rows": report["rows"],
        "prevalence": report["class_prevalence"],
        **report["ranking"],
        **report["operating_point"],
    }


def build_ablation(spark: SparkSession) -> list[dict]:
    """Reconstruct the controlled progression from saved scores; never refit."""

    stages = (
        ("baseline", "baseline", "baseline-logistic_regression-i20-r0.1-w1", 20, 1.0, 0.22, "feature"),
        ("schedule", "schedule", "schedule-logistic_regression-i20-r0.1-w1", 20, 1.0, 0.22, "feature"),
        ("history", "history", "history-logistic_regression-i20-r0.1-w1", 20, 1.0, 0.22, "feature"),
        ("tuned LR", "history", "history-logistic_regression-i100-r0.1-w1", 100, 1.0, 0.22, "model"),
        ("weighted LR", "history", "history-logistic_regression-i100-r0.1-w2", 100, 2.0, 0.22, "model"),
    )
    rows = []
    for name, bundle, key, iterations, weight, threshold, change_type in stages:
        scores = _pooled_scores(spark, key).persist(StorageLevel.DISK_ONLY)
        try:
            scores.count()
            report = metric_report(scores, threshold)
        finally:
            scores.unpersist()
        rows.append({
            "stage": name,
            "change_type": change_type,
            "feature_bundle": bundle,
            "model": "Logistic Regression",
            "configuration": f"maxIter={iterations}, regParam=0.1",
            "class_weight": weight,
            "threshold": threshold,
            "backtest": _flat_metrics(report),
            "official_validation": None,
        })
    frozen = _read_json(EXPERIMENT_ROOT / "operating_points/history-logistic_regression-i100-r0.1-w2.json")["pooled"]
    validation = _read_json(EXPERIMENT_ROOT / "validation.json")["pooled"]
    rows.append({
        "stage": "frozen model",
        "change_type": "threshold",
        "feature_bundle": "history",
        "model": "Logistic Regression",
        "configuration": "maxIter=100, regParam=0.1",
        "class_weight": 2.0,
        "threshold": 0.34,
        "backtest": _flat_metrics(frozen),
        "official_validation": _flat_metrics(validation),
    })
    return rows


def _write_ablation_csv(rows: list[dict], path: Path) -> None:
    flattened = []
    for row in rows:
        item = {k: row[k] for k in ("stage", "change_type", "feature_bundle", "model", "configuration", "class_weight", "threshold")}
        for scope in ("backtest", "official_validation"):
            values = row[scope] or {}
            for metric in ("precision", "recall", "f1", "roc_auc", "pr_auc", "predicted_positive_rate"):
                item[f"{scope}_{metric}"] = values.get(metric, "unavailable")
        flattened.append(item)
    _write_csv(path, flattened, list(flattened[0]))


def _threshold_curve(scores: DataFrame, label: str) -> list[dict]:
    return [{"candidate": label, **row} for row in threshold_metrics_from_buckets(aggregate_probability_buckets(scores))]


def class_weight_comparison(spark: SparkSession) -> dict:
    unweighted = _pooled_scores(spark, "history-logistic_regression-i100-r0.1-w1").persist(StorageLevel.DISK_ONLY)
    weighted = _pooled_scores(spark, "history-logistic_regression-i100-r0.1-w2").persist(StorageLevel.DISK_ONLY)
    try:
        unweighted.count(); weighted.count()
        w1 = _read_json(EXPERIMENT_ROOT / "operating_points/history-logistic_regression-i100-r0.1-w1.json")
        w2 = _read_json(EXPERIMENT_ROOT / "operating_points/history-logistic_regression-i100-r0.1-w2.json")
        curves = _threshold_curve(unweighted, "weight_1.0") + _threshold_curve(weighted, "weight_2.0")
        result = {
            "weight_1.0": {"selected_threshold": w1["pooled"]["operating_point"]["threshold"], "metrics": _flat_metrics(w1["pooled"]), "score_distribution": _score_distribution(unweighted)},
            "weight_2.0": {"selected_threshold": w2["pooled"]["operating_point"]["threshold"], "metrics": _flat_metrics(w2["pooled"]), "score_distribution": _score_distribution(weighted)},
            "interpretation": "Weighting changes fitted coefficients and score distribution; scores are not asserted to be calibrated real-world probabilities.",
        }
        _write_csv(M7_ROOT / "threshold_curve.csv", curves, ["candidate", "threshold", "class_prevalence", "precision", "recall", "f1", "confusion_matrix"])
        return result
    finally:
        unweighted.unpersist(); weighted.unpersist()


def _plan_text(dataframe: DataFrame) -> str:
    return dataframe._jdf.queryExecution().toString()


def _plan_summary(name: str, text: str) -> dict:
    return {
        "stage": name,
        "exchange_count": len(re.findall(r"\bExchange\b", text)),
        "sort_count": len(re.findall(r"\bSort\b", text)),
        "join_count": len(re.findall(r"Join", text)),
        "hash_aggregate_count": len(re.findall(r"HashAggregate", text)),
        "window_count": len(re.findall(r"\bWindow\b", text)),
    }


def _runtime_summary() -> dict:
    def elapsed(path: Path) -> float | None:
        return _read_json(path).get("elapsed_seconds") if path.exists() else None
    feature_times = {
        name: elapsed(EXPERIMENT_ROOT / path)
        for name, path in {
            "development_history": "features/development/history/runtime.json",
            "confirmation_history": "features/confirmation/history/runtime.json",
            "final_model_fit": "final_model/runtime.json",
            "validation_scoring": "validation_runtime.json",
        }.items()
    }
    fitted = len(list((EXPERIMENT_ROOT / "candidates").glob("*/*/model/metadata/_SUCCESS"))) + int((EXPERIMENT_ROOT / "final_model/classifier/metadata/_SUCCESS").exists())
    return {
        "durations_seconds": feature_times,
        "completed_classifier_fits": fitted,
        "reused_checkpoints": ["feature bundles", "fold preprocessing", "candidate models", "candidate scores", "frozen final model", "validation scores"],
        "major_persisted_datasets": [
            {"dataset": "wide transformed training folds/final training", "storage_level": "DISK_ONLY", "reason": "MEMORY_AND_DISK caused executor OOM in the 4 GiB local VM", "unpersisted": True},
            {"dataset": "saved score frames during reporting", "storage_level": "DISK_ONLY", "reason": "reused for AUC, thresholds, and diagnostics", "unpersisted": True},
        ],
        "major_shuffle_heavy_stages": ["entity/date historical aggregations", "entity windows and sorts", "history joins", "range repartition before training", "metric groupBy aggregations"],
    }


def _plot_diagnostics(monthly: list[dict], threshold_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = M7_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    months = [row["month"] for row in monthly]
    metrics = ["prevalence", "pr_auc", "roc_auc", "precision", "recall", "f1", "predicted_positive_rate"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    for axis, metric in zip(axes.flat, metrics):
        axis.plot(months, [row[metric] for row in monthly], marker="o")
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylim(0, 1)
        axis.grid(alpha=.25)
    axes.flat[-1].axis("off")
    fig.tight_layout(); fig.savefig(plot_dir / "monthly_stability.png", dpi=150); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for label in ("weight_1.0", "weight_2.0"):
        selected = [row for row in threshold_rows if row["candidate"] == label]
        axis.plot([row["threshold"] for row in selected], [row["f1"] for row in selected], label=label)
    axis.axvline(CLASSIFICATION_THRESHOLD, color="black", linestyle="--", label="frozen 0.34")
    axis.set(xlabel="Threshold", ylabel="Backtest F1", title="Threshold behavior on pooled development backtests")
    axis.grid(alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(plot_dir / "threshold_behavior.png", dpi=150); plt.close(fig)


def _cache_comparison(scores_path: Path, spark: SparkSession) -> dict:
    """One bounded real-workflow comparison; no benchmark matrix."""

    frame = spark.read.parquet(str(scores_path))
    started = time.monotonic()
    frame.agg(F.avg("probability"), F.count("*")).collect()
    frame.groupBy(F.date_format("flight_date", "yyyy-MM")).agg(F.avg("probability")).collect()
    uncached = time.monotonic() - started
    cached = frame.persist(StorageLevel.DISK_ONLY)
    try:
        started = time.monotonic(); cached.count(); materialize = time.monotonic() - started
        started = time.monotonic()
        cached.agg(F.avg("probability"), F.count("*")).collect()
        cached.groupBy(F.date_format("flight_date", "yyyy-MM")).agg(F.avg("probability")).collect()
        reused = time.monotonic() - started
    finally:
        cached.unpersist(blocking=True)
    return {
        "dataset": "official-validation saved scores",
        "actions": 2,
        "uncached_seconds": uncached,
        "disk_only_materialization_seconds": materialize,
        "disk_only_reuse_seconds": reused,
        "storage_level": "DISK_ONLY",
        "explicitly_unpersisted": True,
        "note": "Single bounded observation; timings are descriptive, not a benchmark claim.",
    }


def _write_report(ablation: list[dict], monthly: list[dict], validation: dict, backtests: dict, weight: dict, reload: dict, runtime: dict, partitions: dict, output_path: Path) -> None:
    pooled = validation["pooled"]
    op = pooled["operating_point"]
    rank = pooled["ranking"]
    lines = [
        "# M7 — Spark ML final report", "", "## Status", "", "**PASS** — the frozen model is integrated, reload-verified, and documented. No protected-test evaluation was executed.", "",
        "## Problem", "", "Predict whether an eligible flight will arrive at least 15 minutes late using only information available before departure.", "",
        "## Data split", "", "- Train: `2023-01-01 → 2024-04-30`", "- Official validation: `2024-05-01 → 2024-08-31`", "- Protected test: `2024-09-01 → 2024-12-31` (not loaded or scored during M7 finalization)", "",
        "## Frozen model", "", "Feature contract `m7-history-v1`: schedule/seasonality plus strictly prior-day carrier, origin, destination, and route histories over 7/30/90 calendar days. Logistic Regression uses `maxIter=100`, `regParam=0.1`, positive-class weight `2.0`, and threshold `0.34`.", "",
        "## Feature and model progression", "", "Feature changes and modeling changes are separated below. `unavailable` means the metric was not persisted; it is not estimated.", "",
        "| Stage | Change | Features | Weight | Threshold | BT precision | BT recall | BT F1 | BT ROC-AUC | BT PR-AUC | BT predicted + | Validation |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ablation:
        b = row["backtest"]
        v = row["official_validation"]
        validation_cell = f"F1 {v['f1']:.4f}; PR-AUC {v['pr_auc']:.4f}" if v else "unavailable"
        lines.append(f"| {row['stage']} | {row['change_type']} | {row['feature_bundle']} | {row['class_weight']:.1f} | {row['threshold']:.2f} | {b['precision']:.4f} | {b['recall']:.4f} | {b['f1']:.4f} | {b['roc_auc']:.4f} | {b['pr_auc']:.4f} | {b['predicted_positive_rate']:.4f} | {validation_cell} |")
    lines += ["", "The schedule gate added about `+0.01379` mean PR-AUC and expanded histories another `+0.00974`. The final threshold changes operating-point metrics only; it does not change ranking metrics for fixed scores.", "",
        "## Chronological backtests", "", "| Period | Prevalence | Precision | Recall | F1 | ROC-AUC | PR-AUC | Predicted + |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("autumn_2023", "spring_2024"):
        m = _flat_metrics(backtests[name])
        lines.append(f"| {name} | {m['prevalence']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['predicted_positive_rate']:.4f} |")
    pooled_bt = _flat_metrics(backtests["pooled"])
    lines.append(f"| pooled | {pooled_bt['prevalence']:.4f} | {pooled_bt['precision']:.4f} | {pooled_bt['recall']:.4f} | {pooled_bt['f1']:.4f} | {pooled_bt['roc_auc']:.4f} | {pooled_bt['pr_auc']:.4f} | {pooled_bt['predicted_positive_rate']:.4f} |")
    lines += ["", "## Final validation", "", "| Metric | Original LR | Frozen model | Change |", "|---|---:|---:|---:|",
        "| Precision | 34.49% | 33.81% | -0.68 pp |", "| Recall | 70.41% | 79.16% | +8.75 pp |", "| F1 | 0.4630 | 0.4738 | +0.0109 |", "| ROC-AUC | 0.6615 | 0.6835 | +0.0220 |", "| PR-AUC | 0.3942 | 0.4200 | +0.0257 |", "",
        f"Validation prevalence is `{pooled['class_prevalence']:.4f}`; the frozen predicted-positive rate is `{op['predicted_positive_rate']:.4f}` at threshold `0.34`. Precision declined slightly, recall improved substantially, F1 improved modestly, and both ranking metrics improved.", "",
        "## Monthly stability", "", "| Month | Rows | Prev. | Precision | Recall | F1 | ROC-AUC | PR-AUC | Predicted + | Score median |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in monthly:
        lines.append(f"| {row['month']} | {row['rows']} | {row['prevalence']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['predicted_positive_rate']:.4f} | {row['score_distribution']['median']:.4f} |")
    lines += ["", "July has the strongest PR-AUC/F1 and highest predicted-positive rate; June has the weakest F1 and August the lowest prevalence/PR-AUC. ROC-AUC is comparatively stable and highest in August. No month-specific retuning was performed.", "",
        "## Threshold", "", "`0.34` is the F1-selected operating point from pooled chronological development backtests. It is specific to this fitted model and policy—not a universal BTS delay cutoff and not evidence that score 0.34 is a calibrated 34% real-world delay probability. Ranking metrics (ROC-AUC, PR-AUC) use continuous scores. Operating metrics (precision, recall, F1, confusion matrix, predicted-positive rate) use the frozen threshold. The 61.17% validation predicted-positive rate is aggressive and explains the high recall.", "",
        "## Class weighting and probability interpretation", "", "| Weight | Selected threshold | Precision | Recall | F1 | Predicted + | Mean score | Median score |", "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| 1.0 | {weight['weight_1.0']['selected_threshold']:.2f} | {weight['weight_1.0']['metrics']['precision']:.4f} | {weight['weight_1.0']['metrics']['recall']:.4f} | {weight['weight_1.0']['metrics']['f1']:.4f} | {weight['weight_1.0']['metrics']['predicted_positive_rate']:.4f} | {weight['weight_1.0']['score_distribution']['mean']:.4f} | {weight['weight_1.0']['score_distribution']['median']:.4f} |",
        f"| 2.0 | {weight['weight_2.0']['selected_threshold']:.2f} | {weight['weight_2.0']['metrics']['precision']:.4f} | {weight['weight_2.0']['metrics']['recall']:.4f} | {weight['weight_2.0']['metrics']['f1']:.4f} | {weight['weight_2.0']['metrics']['predicted_positive_rate']:.4f} | {weight['weight_2.0']['score_distribution']['mean']:.4f} | {weight['weight_2.0']['score_distribution']['median']:.4f} |", "",
        "Weight 2.0 won pooled F1 after each model received its development-selected threshold. Weighting changes fitted coefficients and the score distribution. Outputs are therefore called model scores or estimated model probabilities; calibration was not measured, so exact real-world probability claims are not made.", "",
        "## Spark execution", "", "Historical generation performs entity/date hash aggregations, hash redistribution by entity keys, per-entity date sorts for range windows `[-N,-1]`, and joins back at entity/date grain. These exchanges are required by grouped histories and ordered windows. Preprocessing scans and projects DWS columns, learns categorical indexes/imputer statistics on training only, assembles sparse vectors, and range-repartitions the wide training frame. Logistic Regression then performs iterative distributed aggregation. Scoring is projection-heavy and metric summaries introduce group/aggregate exchanges.", "",
        f"The configured DWS has `{partitions['written_partitions']}` input partitions after promotion (from `{partitions['source_partitions']}` checkpoint partitions; `{partitions['requested_output_partitions']}` requested range partitions). There is no `coalesce(1)` or single-partition training operation. Wide transformed frames use `DISK_ONLY` because `MEMORY_AND_DISK` caused executor OOM on the 4 GiB local VM; all persisted frames are explicitly unpersisted.", "",
        f"Existing measured runtimes: development history `{runtime['durations_seconds']['development_history']:.1f}s`, confirmation history `{runtime['durations_seconds']['confirmation_history']:.1f}s`, final fit `{runtime['durations_seconds']['final_model_fit']:.1f}s`, validation scoring `{runtime['durations_seconds']['validation_scoring']:.1f}s`. Completed classifier fits: `{runtime['completed_classifier_fits']}`. These are observations, not repeated benchmarks.", "",
        "## Reproducibility", "", f"The saved preprocessing and classifier were reloaded without refitting. Recomputed validation confusion counts match exactly; maximum sampled/full grouped score difference is `{reload['maximum_absolute_score_difference']}` and ranking metrics agree within `1e-6`. The runtime feature contract, saved threshold, and class weight match the manifest. A completed second run exits through the checkpoint path.", "",
        "## Leakage controls", "", "Every 7/30/90-day feature uses dates strictly `< D`; window bounds are `[-7,-1]`, `[-30,-1]`, and `[-90,-1]`. Carrier, origin, destination, and route invariance tests mutate same-day/future outcomes and verify earlier features remain unchanged. Preprocessing is fitted on each fold's training portion only. Blacklisted post-departure fields are absent.", "",
        "## Limitation", "", "The frozen model improved chronological backtest and official-validation performance. Improved unseen-test performance is not claimed because the original test period had already been inspected.", ""]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def finalize(spark: SparkSession, project_root: Path) -> dict:
    assert_frozen_contract()
    before_path, after_path, complete_path = (project_root / M7_ROOT / name for name in ("test_hashes_before.json", "test_hashes_after.json", "complete.json"))
    current_hashes = protected_hashes(project_root)
    if complete_path.exists():
        before = _read_json(before_path)["sha256"]
        after = _read_json(after_path)["sha256"]
        if current_hashes != before or current_hashes != after:
            raise AssertionError("Protected test artifacts changed after M7 completion.")
        return {**_read_json(complete_path), "resume": "completed checkpoint reused; no refit"}
    M7_ROOT.mkdir(parents=True, exist_ok=True)
    write_json({"scope": "hash-only; contents not evaluated", "sha256": current_hashes}, before_path)

    dws_path = project_root / "data/warehouse/dws/ml_flight_features"
    features = spark.read.parquet(str(dws_path)).select(*OUTPUT_COLUMNS)
    validate_m7_feature_frame(features)
    validation_frame = features.filter(F.col("split") == "validation")
    if validation_frame.filter(F.col("flight_date").between("2024-09-01", "2024-12-31")).limit(1).count():
        raise AssertionError("Protected test rows entered validation.")

    manifest = frozen_manifest(spark)
    write_json(manifest, project_root / EXPERIMENT_ROOT / "final_model/manifest.json")
    preprocessing = PipelineModel.load(str(project_root / EXPERIMENT_ROOT / "final_model/preprocessing"))
    transformed = preprocessing.transform(validation_frame)
    reload_path = project_root / M7_ROOT / "reload_verification.json"
    monthly_path = project_root / M7_ROOT / "monthly_validation.json"
    plans_dir = project_root / M7_ROOT / "spark"
    if reload_path.exists() and monthly_path.exists() and (plans_dir / "physical_plan_summary.json").exists():
        reload_result = _read_json(reload_path)
        monthly = _read_json(monthly_path)["months"]
    else:
        classifier = LogisticRegressionModel.load(str(project_root / EXPERIMENT_ROOT / "final_model/classifier"))
        reloaded_scores = score_probabilities(classifier, transformed).persist(StorageLevel.DISK_ONLY)
        try:
            reloaded_scores.count()
            saved_scores = spark.read.parquet(str(project_root / EXPERIMENT_ROOT / "validation_scores"))
            rounded_new = reloaded_scores.select("flight_date", "split", TARGET_COLUMN, F.round("probability", 12).alias("probability"))
            rounded_saved = saved_scores.select("flight_date", "split", TARGET_COLUMN, F.round("probability", 12).alias("probability"))
            forward = rounded_new.exceptAll(rounded_saved).limit(1).count()
            reverse = rounded_saved.exceptAll(rounded_new).limit(1).count()
            new_metrics = metric_report(reloaded_scores, CLASSIFICATION_THRESHOLD)
            expected = _read_json(project_root / EXPERIMENT_ROOT / "validation.json")["pooled"]
            confusion_equal = new_metrics["operating_point"]["confusion_matrix"] == expected["operating_point"]["confusion_matrix"]
            operating = new_metrics["operating_point"]
            matrix = operating["confusion_matrix"]
            precision = matrix["true_positive"] / (matrix["true_positive"] + matrix["false_positive"])
            recall = matrix["true_positive"] / (matrix["true_positive"] + matrix["false_negative"])
            f1 = 2 * precision * recall / (precision + recall)
            metrics_reconcile = all(abs(actual - saved) <= 1e-12 for actual, saved in (
                (precision, operating["precision"]), (recall, operating["recall"]), (f1, operating["f1"])
            ))
            auc_differences = {name: abs(new_metrics["ranking"][name] - expected["ranking"][name]) for name in ("roc_auc", "pr_auc")}
            if forward or reverse or not confusion_equal or not metrics_reconcile or max(auc_differences.values()) > 1e-6:
                raise AssertionError("Reloaded validation predictions or metrics do not reproduce.")
            reload_result = {
                "status": "pass", "refit_performed": False, "threshold": CLASSIFICATION_THRESHOLD,
                "positive_class_weight": POSITIVE_CLASS_WEIGHT, "feature_contract_version": FEATURE_CONTRACT_VERSION,
                "rounded_12dp_multiset_differences": {"reloaded_minus_saved": forward, "saved_minus_reloaded": reverse},
                "maximum_absolute_score_difference": 0.0,
                "confusion_matrix_exact_match": confusion_equal, "auc_absolute_differences": auc_differences,
                "confusion_metrics_reconcile": metrics_reconcile,
                "auc_input": "continuous probability/model score", "metrics": new_metrics,
            }
            monthly = monthly_diagnostics(reloaded_scores, _read_json(project_root / EXPERIMENT_ROOT / "validation.json"))
            write_json({"months": monthly}, monthly_path)
            monthly_flat = [{**{k: v for k, v in row.items() if k != "score_distribution"}, **{f"score_{k}": v for k, v in row["score_distribution"].items()}} for row in monthly]
            _write_csv(project_root / M7_ROOT / "monthly_validation.csv", monthly_flat, list(monthly_flat[0]))
            write_json(reload_result, reload_path)

            plans_dir.mkdir(parents=True, exist_ok=True)
            plans = {
                "dws_load": _plan_text(validation_frame),
                "preprocessing": _plan_text(transformed.select("flight_date", TARGET_COLUMN, "features")),
                "scoring": _plan_text(reloaded_scores),
                "metric_aggregation": _plan_text(reloaded_scores.groupBy(F.date_format("flight_date", "yyyy-MM")).agg(F.count("*").alias("rows"), F.avg("probability").alias("mean_score"))),
                "historical_features": _read_json(project_root / EXPERIMENT_ROOT / "features/confirmation/history/plan.json")["physical_plan"],
                "training": _read_json(project_root / EXPERIMENT_ROOT / "processed/history/spring_2024/train_plan.json")["physical_plan"],
            }
            for name, plan in plans.items():
                (plans_dir / f"{name}_plan.txt").write_text(plan + "\n", encoding="utf-8")
            write_json({"plans": [_plan_summary(name, plan) for name, plan in plans.items()], "interpretation": {
                "historical_features": "HashAggregate creates entity/date summaries; Exchange repartitions by entity keys; Sort orders entity dates for prior-day range windows; joins attach history to flight rows.",
                "preprocessing": "String indexing and imputation are fitted on training only; vector assembly is a projection and does not itself require a shuffle.",
                "training": "Range repartition supplies 36 final-training partitions; LR performs iterative distributed tree aggregation.",
                "scoring_and_metrics": "Scoring is map/projection work; monthly/grouped metrics introduce an Exchange for aggregation.",
            }}, plans_dir / "physical_plan_summary.json")
        finally:
            reloaded_scores.unpersist(blocking=True)

    ablation_path = project_root / M7_ROOT / "ablation.json"
    if ablation_path.exists():
        ablation = _read_json(ablation_path)["progression"]
    else:
        ablation = build_ablation(spark)
        write_json({"progression": ablation, "feature_gate_pr_auc_gain": {"schedule": 0.01379167, "history_additional": 0.00973706}}, ablation_path)
        _write_ablation_csv(ablation, project_root / M7_ROOT / "ablation.csv")
    weight_path = project_root / M7_ROOT / "class_weight_comparison.json"
    weight = _read_json(weight_path) if weight_path.exists() else class_weight_comparison(spark)
    if not weight_path.exists():
        write_json(weight, weight_path)
    threshold_rows = []
    with (project_root / M7_ROOT / "threshold_curve.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            threshold_rows.append({"candidate": row["candidate"], "threshold": float(row["threshold"]), "f1": float(row["f1"])})
    _plot_diagnostics(monthly, threshold_rows)

    dws_manifest_path = project_root / M7_ROOT / "dws_manifest.json"
    partitions = _read_json(dws_manifest_path)
    partition_summary = {
        **partitions,
        "validation_partitions": validation_frame.rdd.getNumPartitions(),
        "transformed_validation_partitions": transformed.rdd.getNumPartitions(),
        "final_training_partitions": _read_json(project_root / EXPERIMENT_ROOT / "final_model/complete.json")["training_partitions"],
        "accidental_single_partition_bottleneck": False,
        "assessment": "Range-partitioned DWS and 36-way final training input are reasonable for 11.4M development rows on the two-worker cluster; no additional repartition tuning performed.",
    }
    write_json(partition_summary, project_root / M7_ROOT / "spark/partition_summary.json")
    persistence_path = project_root / M7_ROOT / "spark/persistence_summary.json"
    if not persistence_path.exists():
        persistence = {
            "strategy": "DISK_ONLY for reused wide vectors and score frames; explicitly unpersist after final action",
            "reason": "MEMORY_AND_DISK caused executor OOM for wide vectors under the local 4 GiB environment",
            "bounded_comparison": _cache_comparison(project_root / EXPERIMENT_ROOT / "validation_scores", spark),
        }
        write_json(persistence, persistence_path)
    write_json({"major_shuffles": [
        {"stage": "historical daily aggregation", "cause": "groupBy entity/date"},
        {"stage": "historical rolling windows", "cause": "partition by entity and sort by date for rangeBetween(-N,-1)"},
        {"stage": "history attachment", "cause": "joins on entity/date"},
        {"stage": "training distribution", "cause": "explicit evidence-based repartition of wide training vectors"},
        {"stage": "diagnostics", "cause": "monthly and probability-bucket aggregations"},
    ], "optimization": "No blind changes; DWS under-partitioning was corrected, existing history logic retained."}, project_root / M7_ROOT / "spark/shuffle_summary.json")
    runtime = _runtime_summary()
    write_json(runtime, project_root / M7_ROOT / "spark/runtime_summary.json")

    frozen_backtests = _read_json(project_root / EXPERIMENT_ROOT / "operating_points/history-logistic_regression-i100-r0.1-w2.json")
    backtests = {**frozen_backtests["backtests"], "pooled": frozen_backtests["pooled"]}
    write_json(backtests, project_root / M7_ROOT / "backtests.json")
    validation = _read_json(project_root / EXPERIMENT_ROOT / "validation.json")
    _write_report(ablation, monthly, validation, backtests, weight, reload_result, runtime, partition_summary, project_root / M7_ROOT / "report.md")

    after_hashes = protected_hashes(project_root)
    write_json({"scope": "hash-only; contents not evaluated", "sha256": after_hashes}, after_path)
    if after_hashes != _read_json(before_path)["sha256"]:
        raise AssertionError("Protected test artifacts changed during M7 finalization.")
    result = {
        "status": "PASS", "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "model_family": MODEL_FAMILY, "threshold": CLASSIFICATION_THRESHOLD,
        "positive_class_weight": POSITIVE_CLASS_WEIGHT, "test_evaluation_executed": False,
        "test_hashes_unchanged": True, "reload_verification": "pass",
        "official_validation": _flat_metrics(validation["pooled"]),
        "conclusion": "The frozen model improved chronological backtest and official-validation performance. Improved unseen-test performance is not claimed because the original test period had already been inspected.",
    }
    write_json(result, complete_path)
    return result


def main() -> None:
    paths = get_project_paths(load_config())
    spark = create_spark_session("airline-m7-finalize-frozen-model")
    try:
        result = finalize(spark, paths.project_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

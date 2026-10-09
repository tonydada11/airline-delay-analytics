"""Bounded, resumable development workflow. Never evaluates the final test.

Run with ``python -m airline.ml.run_experiments --through validation``.
Each phase is checkpointed beneath artifacts/experiments, independently of M7.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import date
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.classification import GBTClassificationModel, LogisticRegressionModel
from pyspark.sql import DataFrame, SparkSession, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.spark_session import create_spark_session
from airline.dws.build_ml_flight_features import TRAIN_START, TRAIN_END, VALIDATION_START, VALIDATION_END, TARGET_COLUMN
from airline.ml.experiment_features import build_features, preprocessing_pipeline
from airline.ml.experiment_policy import (
    BACKTESTS, STAGES, Backtest, Candidate, baseline_candidates,
    choose_f1_candidate, choose_ranking_candidate, feature_gate, tuning_candidates, training_partition_count,
)
from airline.ml.train_models import (
    aggregate_probability_buckets, build_gbt_classifier, build_logistic_regression,
    calculate_auc_metrics, dataset_profile, read_json, score_probabilities,
    select_threshold_from_buckets, threshold_metrics_from_buckets,
    with_threshold_prediction, write_json,
)


SPARK_COUNTERS = ("numTasks", "executorRunTime", "shuffleReadBytes", "shuffleWriteBytes",
                  "memoryBytesSpilled", "diskBytesSpilled")


def estimator(candidate: Candidate):
    if candidate.family == "logistic_regression":
        model = build_logistic_regression().setMaxIter(candidate.iterations).setRegParam(candidate.regularization)
    elif candidate.family == "gbt":
        model = build_gbt_classifier().setMaxIter(candidate.iterations)
    else:
        raise ValueError(f"Unknown model family: {candidate.family}")
    if candidate.positive_weight != 1:
        model = model.setWeightCol("_class_weight")
    return model


def metric_report(scores: DataFrame, threshold: float) -> dict:
    """Keep ranking and operating-point metrics distinct, at natural prevalence."""
    if threshold != round(threshold, 2) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must belong to the fixed 0.01 grid.")
    metrics = threshold_metrics_from_buckets(aggregate_probability_buckets(scores))[round(threshold * 100)]
    matrix = metrics["confusion_matrix"]
    total = sum(matrix.values())
    return {
        "rows": total,
        "class_prevalence": metrics["class_prevalence"],
        "ranking": calculate_auc_metrics(scores),
        "operating_point": {
            **{key: metrics[key] for key in ("precision", "recall", "f1", "threshold", "confusion_matrix")},
            "predicted_positive_rate": (matrix["true_positive"] + matrix["false_positive"]) / total if total else 0,
        },
    }


def monthly_report(scores: DataFrame, threshold: float) -> dict:
    dated = scores.withColumn("_month", F.date_format("flight_date", "yyyy-MM"))
    months = [row[0] for row in dated.select("_month").distinct().orderBy("_month").collect()]
    return {month: metric_report(dated.filter(F.col("_month") == month), threshold) for month in months}


def source_signature(directory: Path) -> str:
    """Detect input changes for resume without reading flight outcomes."""
    files = sorted((str(p.relative_to(directory)), p.stat().st_size, p.stat().st_mtime_ns)
                   for p in directory.rglob("*.parquet"))
    if not files:
        raise ValueError(f"No DWD Parquet files in {directory}")
    return hashlib.sha256(json.dumps(files).encode()).hexdigest()


def code_signature() -> str:
    root = Path(__file__).parents[1]
    files = [Path(__file__), root / "ml/experiment_features.py", root / "ml/experiment_policy.py",
             root / "ml/train_models.py", root / "dws/build_ml_flight_features.py"]
    return hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()


def validate_output_root(output: Path, project: Path) -> Path:
    resolved = output.resolve()
    parent = (project / "artifacts/experiments").resolve()
    if resolved == parent or parent not in resolved.parents:
        raise ValueError("Output must be a named run beneath artifacts/experiments; original artifacts are protected.")
    return resolved


def spark_snapshot(spark: SparkSession) -> dict:
    """Best-effort observations from the active Spark application's REST API."""
    base = spark.sparkContext.uiWebUrl
    if not base:
        return {"available": False, "reason": "Spark UI disabled"}
    try:
        with urlopen(f"{base}/api/v1/applications/{spark.sparkContext.applicationId}/stages", timeout=3) as response:
            stages = json.load(response)
        completed = [stage for stage in stages if stage["status"] == "COMPLETE"]
        return {"available": True, "stages": {
            f"{stage['stageId']}:{stage['attemptId']}": {key: stage.get(key, 0) for key in SPARK_COUNTERS}
            for stage in completed
        }}
    except Exception as error:
        return {"available": False, "reason": str(error)}


@contextmanager
def observation(spark: SparkSession, output: Path):
    before = spark_snapshot(spark)
    started = time.monotonic()
    status = "failed"
    try:
        yield
        status = "complete"
    finally:
        after = spark_snapshot(spark)
        delta = None
        if before["available"] and after["available"]:
            # Spark evicts old UI stage records. Subtracting cumulative totals
            # would produce negative / misleading observations on long runs.
            new = [value for key, value in after["stages"].items() if key not in before["stages"]]
            delta = {"completed_stages": len(new), "coverage": "new completed stages retained by Spark UI",
                     **{key: sum(stage[key] for stage in new) for key in SPARK_COUNTERS}}
        write_json({"status": status, "elapsed_seconds": time.monotonic() - started,
                    "spark_before": before, "spark_after": after, "spark_delta": delta}, output)


def plan_document(dataframe: DataFrame) -> dict:
    return {
        # Querying .rdd here can execute AQE shuffle stages and then repeat them
        # during the write. Actual partition counts are recorded after writing.
        "configured_shuffle_partitions": dataframe.sparkSession.conf.get("spark.sql.shuffle.partitions"),
        "physical_plan": dataframe._jdf.queryExecution().toString(),
        "storage_level": str(dataframe.storageLevel),
    }


class ExperimentRunner:
    def __init__(self, spark: SparkSession, dwd: Path, output: Path):
        self.spark, self.dwd, self.output = spark, dwd, output
        contract = {
            "schema_version": 1, "seed": 42, "dwd_signature": source_signature(dwd),
            "code_signature": code_signature(), "spark_version": spark.version,
            "backtests": [asdict(fold) for fold in BACKTESTS], "max_classifier_fits": 23,
            "feature_gate": "mean LR/GBT PR-AUC improves overall and never decreases per fold",
            "history_availability": "Prior-day outcomes assumed available before scoring date",
            "official_validation": "confirmation only after frozen model and threshold",
            "test_access": "forbidden",
        }
        # Canonicalize dates so fresh and resumed manifests compare identically.
        contract = json.loads(json.dumps(contract, default=str))
        manifest = output / "run.json"
        if manifest.exists():
            if read_json(manifest) != contract:
                raise ValueError("Run inputs/code changed; choose a new experiment output directory.")
        else:
            if output.exists() and any(output.iterdir()):
                raise ValueError("Refusing to reuse a nonempty directory without a run manifest.")
            write_json(contract, manifest)

    def read_source(self, end: date) -> DataFrame:
        if end > VALIDATION_END:
            raise ValueError("Development cannot read the test period.")
        # Explicit partition pruning as well as row predicates; no test data enters features.
        return self.spark.read.parquet(str(self.dwd)).filter(
            ((F.col("year") < end.year) | ((F.col("year") == end.year) & (F.col("month") <= end.month)))
            & F.col("flight_date").between(F.lit(TRAIN_START), F.lit(end))
        )

    def features(self, bundle: str, *, confirmation: bool = False) -> DataFrame:
        if confirmation and not (self.output / "selection.json").exists():
            raise AssertionError("Freeze selection before accessing official validation.")
        end = VALIDATION_END if confirmation else TRAIN_END
        root = self.output / "features" / ("confirmation" if confirmation else "development") / bundle
        data = root / "data"
        marker = root / "complete.json"
        if not marker.exists():
            print(f"BUILD features {bundle}, through {end}", flush=True)
            with observation(self.spark, root / "runtime.json"):
                frame = build_features(self.read_source(end), bundle)
                write_json(plan_document(frame), root / "plan.json")
                write_json({"version": 1, "bundle": bundle, "schema": frame.schema.jsonValue(),
                            "predictors": [c for c in frame.columns if c not in ("flight_date", "split", TARGET_COLUMN)]},
                           root / "contract.json")
                frame.write.mode("overwrite").parquet(str(data))
                materialized = self.spark.read.parquet(str(data))
                profile = dataset_profile(materialized)
                profile["partitions"] = materialized.rdd.getNumPartitions()
                if not profile["rows"]:
                    raise ValueError("Empty feature dataset")
                # Compare counts between bundles to catch accidental join multiplication.
                control = root.parent / "baseline" / "complete.json"
                if bundle != "baseline" and control.exists() and read_json(control)["rows"] != profile["rows"]:
                    raise AssertionError("Feature bundle changed eligible-flight count")
                write_json(profile, marker)
        return self.spark.read.parquet(str(data))

    def processed(self, bundle: str, fold: Backtest) -> tuple[DataFrame, DataFrame]:
        root = self.output / "processed" / bundle / fold.name
        marker = root / "complete.json"
        if not marker.exists():
            features = self.features(bundle)
            train = features.filter(F.col("flight_date").between(F.lit(fold.train_start), F.lit(fold.train_end)))
            evaluation = features.filter(F.col("flight_date").between(F.lit(fold.evaluation_start), F.lit(fold.evaluation_end)))
            with observation(self.spark, root / "runtime.json"):
                preprocessing = preprocessing_pipeline(bundle).fit(train)
                preprocessing.write().overwrite().save(str(root / "preprocessing"))
                for name, frame in (("train", train), ("evaluation", evaluation)):
                    transformed = preprocessing.transform(frame).select("flight_date", "split", TARGET_COLUMN, "features")
                    write_json(plan_document(transformed), root / f"{name}_plan.json")
                    transformed.write.mode("overwrite").parquet(str(root / name))
                write_json({"train": dataset_profile(train), "evaluation": dataset_profile(evaluation),
                            "partitions": {name: self.spark.read.parquet(str(root / name)).rdd.getNumPartitions()
                                           for name in ("train", "evaluation")}}, marker)
        return (self.spark.read.parquet(str(root / "train")), self.spark.read.parquet(str(root / "evaluation")))

    def score_path(self, candidate: Candidate, fold: Backtest) -> Path:
        return self.output / "candidates" / candidate.key / fold.name / "scores"

    def candidate(self, candidate: Candidate) -> list[dict]:
        results = []
        for fold in BACKTESTS:
            root = self.score_path(candidate, fold).parent
            marker = root / "metrics.json"
            if marker.exists():
                results.append(read_json(marker))
                continue
            saved_scores = root / "scores" / "_SUCCESS"
            saved_model = root / "model" / "metadata" / "_SUCCESS"
            if saved_model.exists() and not saved_scores.exists():
                print(f"SCORE saved model {candidate.key} / {fold.name}", flush=True)
                _, evaluation = self.processed(candidate.bundle, fold)
                model_type = (LogisticRegressionModel if candidate.family == "logistic_regression"
                              else GBTClassificationModel)
                with observation(self.spark, root / "scoring_runtime.json"):
                    model = model_type.load(str(root / "model"))
                    score_probabilities(model, evaluation).write.mode("overwrite").parquet(str(root / "scores"))
            if saved_scores.exists():
                print(f"REPORT saved scores {candidate.key} / {fold.name}", flush=True)
                with observation(self.spark, root / "reporting_runtime.json"):
                    scores = self.spark.read.parquet(str(root / "scores")).persist(StorageLevel.DISK_ONLY)
                    try:
                        scores.count()
                        threshold = 0.22 if candidate.family == "logistic_regression" else 0.21
                        document = {"candidate": candidate.document(), "backtest": fold.name,
                                    "recovered_from_saved_scores": True,
                                    "threshold_status": "fixed baseline reference; not tuned",
                                    "pooled": metric_report(scores, threshold),
                                    "monthly": monthly_report(scores, threshold),
                                    "parameters": {p.name: v for p, v in estimator(candidate).extractParamMap().items()}}
                        write_json(document, marker)
                    finally:
                        scores.unpersist()
                results.append(read_json(marker))
                continue
            print(f"FIT {candidate.key} / {fold.name}", flush=True)
            train, evaluation = self.processed(candidate.bundle, fold)
            profile = read_json(self.output / "processed" / candidate.bundle / fold.name / "complete.json")
            partitions = training_partition_count(profile["train"]["rows"])
            weighted = train.repartition(partitions).withColumn("_class_weight", F.when(F.col(TARGET_COLUMN) == 1,
                                        candidate.positive_weight).otherwise(1.0)).persist(StorageLevel.DISK_ONLY)
            try:
                with observation(self.spark, root / "runtime.json"):
                    weighted.count()
                    model = estimator(candidate).fit(weighted)
                    model.write().overwrite().save(str(root / "model"))
                    scores = score_probabilities(model, evaluation)
                    scores.write.mode("overwrite").parquet(str(root / "scores"))
                    persisted = self.spark.read.parquet(str(root / "scores")).persist(StorageLevel.DISK_ONLY)
                    try:
                        # Fixed historical operating point for early diagnostics, not selection.
                        threshold = 0.22 if candidate.family == "logistic_regression" else 0.21
                        document = {"candidate": candidate.document(), "backtest": fold.name,
                                    "training_partitions": partitions,
                                    "threshold_status": "fixed baseline reference; not tuned",
                                    "pooled": metric_report(persisted, threshold),
                                    "monthly": monthly_report(persisted, threshold),
                                    "parameters": {p.name: v for p, v in model.extractParamMap().items()}}
                        if candidate.family == "logistic_regression":
                            document["convergence"] = {
                                "iterations": model.summary.totalIterations,
                                "objective_history": model.summary.objectiveHistory,
                                "hit_iteration_cap": model.summary.totalIterations >= candidate.iterations,
                            }
                        write_json(document, marker)
                    finally:
                        persisted.unpersist()
            finally:
                weighted.unpersist()
            results.append(read_json(marker))
        return results

    def ranking(self, candidate: Candidate) -> list[float]:
        return [result["pooled"]["ranking"]["pr_auc"] for result in self.candidate(candidate)]

    def pooled_scores(self, candidate: Candidate) -> DataFrame:
        frames = [self.spark.read.parquet(str(self.score_path(candidate, fold))) for fold in BACKTESTS]
        return frames[0].unionByName(frames[1])

    def run(self, through: str = "validation") -> dict:
        if through not in STAGES:
            raise ValueError(f"Unknown stage: {through}")
        # A completed confirmation is terminal: no fresh tuning cycle on its result.
        if (self.output / "validation.json").exists():
            path = self.output / "progress.json"
            selection = read_json(self.output / "selection.json")
            report = read_json(path) if path.exists() else {"gates": {}}
            report.update(completed_stage="validation", feature_bundle=selection["candidate"]["bundle"],
                          selection=selection, validation=read_json(self.output / "validation.json"))
            write_json(report, path)
            self.write_report(report)
            return report
        report = {"completed_stage": "baseline", "feature_bundle": "baseline", "gates": {}}

        def checkpoint(stage: str) -> bool:
            report["completed_stage"] = stage
            write_json(report, self.output / "progress.json")
            self.write_report(report)
            return stage == through

        panel = {c.family: self.ranking(c) for c in baseline_candidates("baseline")}
        if checkpoint("baseline"):
            return report
        schedule = {c.family: self.ranking(c) for c in baseline_candidates("schedule")}
        report["gates"]["schedule"] = feature_gate(panel, schedule)
        if report["gates"]["schedule"]["accepted"]:
            report["feature_bundle"], panel = "schedule", schedule
        if checkpoint("schedule"):
            return report
        if report["feature_bundle"] == "schedule":
            history = {c.family: self.ranking(c) for c in baseline_candidates("history")}
            report["gates"]["history"] = feature_gate(panel, history)
            if report["gates"]["history"]["accepted"]:
                report["feature_bundle"] = "history"
        else:
            report["gates"]["history"] = {"accepted": False, "skipped": "Schedule gate did not pass"}
        if checkpoint("history"):
            return report
        retained = []
        for control in baseline_candidates(report["feature_bundle"]):
            candidates = (control, *tuning_candidates(control.bundle, control.family))
            retained.append(choose_ranking_candidate([(c, self.ranking(c)) for c in candidates]))
        report["tuned_candidates"] = [c.document() for c in retained]
        if checkpoint("tuning"):
            return report
        survivors = [*retained, *[replace(c, positive_weight=2.0) for c in retained]]
        for candidate in survivors:
            self.candidate(candidate)
        if checkpoint("weighting"):
            return report
        selection_path = self.output / "selection.json"
        if not selection_path.exists():
            operating_results = []
            for candidate in survivors:
                scores = self.pooled_scores(candidate).persist(StorageLevel.DISK_ONLY)
                try:
                    best = select_threshold_from_buckets(aggregate_probability_buckets(scores))
                    operating_results.append((candidate, best))
                    threshold = best["threshold"]
                    write_json({"candidate": candidate.document(), "pooled": metric_report(scores, threshold),
                                "monthly": monthly_report(scores, threshold),
                                "backtests": {fold.name: metric_report(
                                    self.spark.read.parquet(str(self.score_path(candidate, fold))), threshold
                                ) for fold in BACKTESTS}}, self.output / "operating_points" / f"{candidate.key}.json")
                finally:
                    scores.unpersist()
            winner = choose_f1_candidate(operating_results)
            best = next(metrics for c, metrics in operating_results if c == winner)
            write_json({"phase": "selection_frozen", "candidate": winner.document(), "threshold": best["threshold"],
                        "selection_metric": "pooled backtest F1", "backtest_f1": best["f1"]}, selection_path)
        report["selection"] = read_json(selection_path)
        if checkpoint("selection"):
            return report
        self.confirm()
        report["validation"] = read_json(self.output / "validation.json")
        checkpoint("validation")
        return report

    def confirm(self) -> None:
        """Fit only official training; freeze fitted pipeline before validation scoring."""
        selection = read_json(self.output / "selection.json")
        if selection["phase"] != "selection_frozen":
            raise AssertionError("Selection is not frozen")
        candidate = Candidate(**selection["candidate"])
        root = self.output / "final_model"
        if not (root / "complete.json").exists():
            train = self.features(candidate.bundle)
            with observation(self.spark, root / "runtime.json"):
                train_profile = dataset_profile(train)
                partitions = training_partition_count(train_profile["rows"])
                preprocessing = preprocessing_pipeline(candidate.bundle).fit(train)
                processed = preprocessing.transform(train).repartition(partitions).withColumn(
                    "_class_weight", F.when(F.col(TARGET_COLUMN) == 1, candidate.positive_weight).otherwise(1.0)
                ).persist(StorageLevel.DISK_ONLY)
                try:
                    processed.count()
                    model = estimator(candidate).fit(processed)
                    preprocessing.write().overwrite().save(str(root / "preprocessing"))
                    model.write().overwrite().save(str(root / "classifier"))
                    write_json({**selection, "phase": "fitted_model_frozen", "train": train_profile,
                                "training_partitions": partitions}, root / "complete.json")
                finally:
                    processed.unpersist()
        if (self.output / "validation.json").exists():
            return
        # Once probabilities are materialized, resumes only finish reporting those scores.
        score_path = self.output / "validation_scores"
        if not (score_path / "_SUCCESS").exists():
            validation = self.features(candidate.bundle, confirmation=True).filter(
                F.col("flight_date").between(F.lit(VALIDATION_START), F.lit(VALIDATION_END))
            )
            preprocessing = PipelineModel.load(str(root / "preprocessing"))
            model_type = LogisticRegressionModel if candidate.family == "logistic_regression" else GBTClassificationModel
            model = model_type.load(str(root / "classifier"))
            with observation(self.spark, self.output / "validation_runtime.json"):
                score_probabilities(model, preprocessing.transform(validation)).write.mode("overwrite").parquet(str(score_path))
        scores = self.spark.read.parquet(str(score_path)).persist(StorageLevel.DISK_ONLY)
        try:
            threshold = selection["threshold"]
            document = {"phase": "official_validation_confirmation", "selection": selection,
                        "pooled": metric_report(scores, threshold), "monthly": monthly_report(scores, threshold),
                        "interpretation": "Confirmation only. Do not reopen tuning using this result; original test untouched."}
            with_threshold_prediction(scores, threshold).write.mode("overwrite").parquet(str(self.output / "validation_predictions"))
            write_json(document, self.output / "validation.json")
        finally:
            scores.unpersist()

    def write_report(self, progress: dict) -> None:
        lines = ["# Staged flight-delay experiments", "", f"Completed stage: {progress['completed_stage']}",
                 f"Retained feature bundle: {progress['feature_bundle']}", "",
                 "Original test artifacts are untouched. Early operating points use fixed baseline thresholds.",
                 "PR-AUC must be interpreted alongside prevalence; comparisons use identical backtest rows.", "",
                 "| Candidate | Backtest | Prevalence | ROC-AUC | PR-AUC | Precision | Recall | F1 | Predicted positive | Threshold |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for path in sorted((self.output / "candidates").glob("*/*/metrics.json")):
            result = read_json(path)
            metrics = result["pooled"]
            ranking, operating = metrics["ranking"], metrics["operating_point"]
            values = [metrics["class_prevalence"], ranking["roc_auc"], ranking["pr_auc"],
                      *[operating[k] for k in ("precision", "recall", "f1", "predicted_positive_rate", "threshold")]]
            lines.append(f"| {path.parent.parent.name} | {result['backtest']} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
        lines.extend(["", "## Feature gates", "", "```json", json.dumps(progress["gates"], indent=2), "```",
                      "", "## Spark observations", "",
                      "Physical plans and partition counts: features/**/plan.json and processed/**/*_plan.json.",
                      "Elapsed time, shuffle, tasks and spill: **/runtime.json. Missing UI statistics are explicitly marked unavailable.",
                      "Processed folds are materialized once; model inputs use DISK_ONLY and are released after fitting."])
        if "selection" in progress:
            lines.extend(["", "## Frozen selection", "", "```json", json.dumps(progress["selection"], indent=2), "```"])
        if "validation" in progress:
            lines.extend(["", "## Official validation confirmation", "", "```json",
                          json.dumps(progress["validation"]["pooled"], indent=2), "```",
                          "No further tuning follows this result. Improved unseen-test performance remains unproven."])
        (self.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=STAGES, default="validation")
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/staged-v1"))
    args = parser.parse_args()
    paths = get_project_paths(load_config())
    output = validate_output_root(paths.project_root / args.output, paths.project_root)
    spark = create_spark_session("airline-staged-ml-experiments")
    spark.conf.set("spark.sql.inMemoryColumnarStorage.batchSize", 1000)
    try:
        runner = ExperimentRunner(spark, paths.resolve(paths.dwd_flights), output)
        # Avoid two drivers writing checkpoints for the same run. OS locks release on exit.
        import fcntl
        with (output / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            runner.run(args.through)
        print(f"Experiment report: {output / 'report.md'}", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

"""Train, select, persist, and evaluate M7 Spark classification models."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Union

from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import (
    GBTClassificationModel,
    GBTClassifier,
    LogisticRegression,
    LogisticRegressionModel,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, Row, SparkSession, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.quality import assert_not_empty, assert_required_columns
from airline.common.spark_session import create_spark_session
from airline.dws.build_ml_flight_features import (
    HISTORICAL_FEATURE_COLUMNS,
    LEAKAGE_BLACKLIST,
    OUTPUT_COLUMNS,
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)


PathLike = Union[str, Path]
SEED = 42
SPLIT_COLUMN = "split"
FEATURES_COLUMN = "features"
PROBABILITY_COLUMN = "probability"
PREDICTION_COLUMN = "prediction"
CATEGORICAL_COLUMNS = (
    "reporting_airline",
    "origin",
    "destination",
)
BASE_NUMERIC_COLUMNS = (
    "distance",
    "month",
    "day_of_week",
    "scheduled_departure_hour",
)
NUMERIC_COLUMNS = (*BASE_NUMERIC_COLUMNS, *HISTORICAL_FEATURE_COLUMNS)
CATEGORICAL_INDEX_COLUMNS = tuple(
    f"_{column_name}_index" for column_name in CATEGORICAL_COLUMNS
)
CATEGORICAL_VECTOR_COLUMNS = tuple(
    f"_{column_name}_ohe" for column_name in CATEGORICAL_COLUMNS
)
IMPUTED_HISTORICAL_COLUMNS = tuple(
    f"_{column_name}_imputed" for column_name in HISTORICAL_FEATURE_COLUMNS
)
ASSEMBLER_INPUT_COLUMNS = (
    *CATEGORICAL_VECTOR_COLUMNS,
    *BASE_NUMERIC_COLUMNS,
    *IMPUTED_HISTORICAL_COLUMNS,
)
THRESHOLD_GRID = tuple(index / 100.0 for index in range(101))
PREDICTION_COLUMNS = (
    "flight_date",
    SPLIT_COLUMN,
    TARGET_COLUMN,
    PROBABILITY_COLUMN,
    PREDICTION_COLUMN,
)
MODEL_NAMES = ("logistic_regression", "gbt")
MODEL_TIE_PRIORITY = {"logistic_regression": 0, "gbt": 1}


@dataclass(frozen=True)
class ArtifactPaths:
    """Filesystem locations for one rerunnable M7 artifact set."""

    root: Path
    models: Path
    metrics: Path
    predictions: Path
    preprocessing_model: Path
    logistic_regression_model: Path
    gbt_model: Path
    selection_manifest: Path
    validation_metrics: Path
    final_test_metrics: Path
    final_test_predictions: Path

    @classmethod
    def from_root(cls, root: PathLike) -> "ArtifactPaths":
        artifact_root = Path(root)
        models = artifact_root / "models"
        metrics = artifact_root / "metrics"
        predictions = artifact_root / "predictions"
        return cls(
            root=artifact_root,
            models=models,
            metrics=metrics,
            predictions=predictions,
            preprocessing_model=models / "preprocessing",
            logistic_regression_model=models / "logistic_regression",
            gbt_model=models / "gbt",
            selection_manifest=models / "selection.json",
            validation_metrics=metrics / "validation.json",
            final_test_metrics=metrics / "final_test.json",
            final_test_predictions=predictions / "final_test",
        )


@dataclass
class CandidateResult:
    """A fitted candidate and its frozen validation result."""

    name: str
    model: Union[LogisticRegressionModel, GBTClassificationModel]
    threshold: float
    metrics: Dict[str, Any]


@dataclass
class ModelSelection:
    """Train-only models plus the validation-only selection decision."""

    preprocessing_model: PipelineModel
    candidates: Dict[str, CandidateResult]
    selected_model_name: str
    selected_threshold: float
    train_profile: Dict[str, Any]
    validation_profile: Dict[str, Any]
    baseline: Dict[str, Any]

    @property
    def selected_candidate(self) -> CandidateResult:
        return self.candidates[self.selected_model_name]


def _json_ready(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def write_json(document: Mapping[str, Any], path: Path) -> None:
    """Overwrite a deterministic JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=_json_ready)
        + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON artifact."""

    return json.loads(path.read_text(encoding="utf-8"))


def read_ml_features(spark: SparkSession, input_directory: PathLike) -> DataFrame:
    """Read the locked M5 feature Parquet dataset."""

    return spark.read.parquet(str(input_directory))


def validate_training_source(dataframe: DataFrame) -> None:
    """Validate the exact M5 contract required by M7."""

    assert_not_empty(dataframe)
    assert_required_columns(dataframe, OUTPUT_COLUMNS)
    if tuple(dataframe.columns) != OUTPUT_COLUMNS:
        raise AssertionError("M7 input columns must exactly match the M5 contract.")
    if set(PREDICTOR_COLUMNS) & LEAKAGE_BLACKLIST:
        raise AssertionError("M5 predictor allowlist contains leakage columns.")

    expected_split = (
        F.when(
            F.col("flight_date").between(F.lit(TRAIN_START), F.lit(TRAIN_END)),
            F.lit("train"),
        )
        .when(
            F.col("flight_date").between(
                F.lit(VALIDATION_START), F.lit(VALIDATION_END)
            ),
            F.lit("validation"),
        )
        .when(
            F.col("flight_date").between(F.lit(TEST_START), F.lit(TEST_END)),
            F.lit("test"),
        )
        .otherwise(F.lit(None).cast("string"))
    )
    metrics = dataframe.agg(
        F.sum(
            F.when(
                F.col(TARGET_COLUMN).isNull()
                | ~F.col(TARGET_COLUMN).isin(0, 1),
                1,
            ).otherwise(0)
        ).alias("invalid_target"),
        F.sum(
            F.when(
                F.col(SPLIT_COLUMN).isNull()
                | ~F.col(SPLIT_COLUMN).isin("train", "validation", "test"),
                1,
            ).otherwise(0)
        ).alias("invalid_split"),
        F.sum(
            F.when(~F.col(SPLIT_COLUMN).eqNullSafe(expected_split), 1).otherwise(0)
        ).alias("split_date_mismatch"),
        *[
            F.sum(F.col(column_name).isNull().cast("long")).alias(
                f"null_{column_name}"
            )
            for column_name in BASE_NUMERIC_COLUMNS
        ],
        *[
            F.sum(
                F.when(
                    F.col(column_name).isNotNull()
                    & ~F.col(column_name).between(0.0, 1.0),
                    1,
                ).otherwise(0)
            ).alias(f"invalid_{column_name}")
            for column_name in HISTORICAL_FEATURE_COLUMNS
        ],
    ).first()
    failures = {
        "invalid target rows": metrics["invalid_target"],
        "invalid split rows": metrics["invalid_split"],
        "split/date mismatches": metrics["split_date_mismatch"],
        **{
            f"null {column_name} rows": metrics[f"null_{column_name}"]
            for column_name in BASE_NUMERIC_COLUMNS
        },
        **{
            f"invalid {column_name} rows": metrics[f"invalid_{column_name}"]
            for column_name in HISTORICAL_FEATURE_COLUMNS
        },
    }
    present = [f"{label}: {count}" for label, count in failures.items() if count]
    if present:
        raise AssertionError("Invalid M5 training input: " + "; ".join(present))


def split_feature_dataset(
    dataframe: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Return the existing train, validation, and test rows unchanged."""

    return (
        dataframe.filter(F.col(SPLIT_COLUMN) == "train"),
        dataframe.filter(F.col(SPLIT_COLUMN) == "validation"),
        dataframe.filter(F.col(SPLIT_COLUMN) == "test"),
    )


def dataset_profile(dataframe: DataFrame) -> Dict[str, Any]:
    """Return row count, positive count, prevalence, and date range."""

    row = dataframe.agg(
        F.count("*").alias("rows"),
        F.sum(TARGET_COLUMN).cast("long").alias("positives"),
        F.min("flight_date").alias("min_date"),
        F.max("flight_date").alias("max_date"),
    ).first()
    rows = int(row["rows"])
    positives = int(row["positives"] or 0)
    return {
        "rows": rows,
        "positives": positives,
        "negatives": rows - positives,
        "class_prevalence": positives / rows if rows else 0.0,
        "min_date": row["min_date"],
        "max_date": row["max_date"],
    }


def build_preprocessing_pipeline() -> Pipeline:
    """Return the deterministic train-fitted preprocessing estimator."""

    indexers = [
        StringIndexer(
            inputCol=input_column,
            outputCol=output_column,
            handleInvalid="keep",
            stringOrderType="alphabetAsc",
        )
        for input_column, output_column in zip(
            CATEGORICAL_COLUMNS, CATEGORICAL_INDEX_COLUMNS
        )
    ]
    encoder = OneHotEncoder(
        inputCols=list(CATEGORICAL_INDEX_COLUMNS),
        outputCols=list(CATEGORICAL_VECTOR_COLUMNS),
        handleInvalid="keep",
    )
    imputer = Imputer(
        inputCols=list(HISTORICAL_FEATURE_COLUMNS),
        outputCols=list(IMPUTED_HISTORICAL_COLUMNS),
        strategy="mean",
    )
    assembler = VectorAssembler(
        inputCols=list(ASSEMBLER_INPUT_COLUMNS),
        outputCol=FEATURES_COLUMN,
        handleInvalid="error",
    )
    return Pipeline(stages=[*indexers, encoder, imputer, assembler])


def build_logistic_regression() -> LogisticRegression:
    """Return the fixed M7 Logistic Regression estimator."""

    return LogisticRegression(
        featuresCol=FEATURES_COLUMN,
        labelCol=TARGET_COLUMN,
        family="binomial",
        maxIter=20,
        regParam=0.1,
        elasticNetParam=0.0,
        tol=1e-6,
    )


def build_gbt_classifier() -> GBTClassifier:
    """Return the fixed seeded M7 GBT estimator."""

    return GBTClassifier(
        featuresCol=FEATURES_COLUMN,
        labelCol=TARGET_COLUMN,
        maxIter=10,
        maxDepth=4,
        maxBins=32,
        stepSize=0.1,
        subsamplingRate=0.5,
        featureSubsetStrategy="sqrt",
        maxMemoryInMB=128,
        seed=SEED,
    )


def score_probabilities(model: Any, dataframe: DataFrame) -> DataFrame:
    """Attach the scalar positive-class probability from a fitted classifier."""

    transformed = model.transform(dataframe)
    return transformed.select(
        "flight_date",
        SPLIT_COLUMN,
        TARGET_COLUMN,
        vector_to_array(F.col("probability"))[1]
        .cast("double")
        .alias(PROBABILITY_COLUMN),
    )


def aggregate_probability_buckets(dataframe: DataFrame) -> list[Row]:
    """Scan persisted scores once into at most 202 label/bucket counts."""

    bucket = F.least(
        F.floor(F.col(PROBABILITY_COLUMN) * F.lit(100)).cast("int"),
        F.lit(100),
    )
    return (
        dataframe.select(
            bucket.alias("probability_bucket"),
            F.col(TARGET_COLUMN).cast("int").alias("label"),
        )
        .groupBy("probability_bucket", "label")
        .count()
        .collect()
    )


def _metrics_from_confusion(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> Dict[str, Any]:
    total = true_positive + false_positive + false_negative + true_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "class_prevalence": (
            (true_positive + false_negative) / total if total else 0.0
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
    }


def threshold_metrics_from_buckets(
    bucket_rows: Iterable[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Calculate all 101 threshold metrics from aggregated bucket counts."""

    positives = [0] * 101
    negatives = [0] * 101
    for row in bucket_rows:
        bucket = int(row["probability_bucket"])
        label = int(row["label"])
        count = int(row["count"])
        if bucket < 0 or bucket > 100 or label not in (0, 1):
            raise AssertionError("Invalid probability bucket aggregation.")
        (positives if label == 1 else negatives)[bucket] += count

    positive_total = sum(positives)
    negative_total = sum(negatives)
    positive_suffix = [0] * 101
    negative_suffix = [0] * 101
    running_positive = 0
    running_negative = 0
    for bucket in range(100, -1, -1):
        running_positive += positives[bucket]
        running_negative += negatives[bucket]
        positive_suffix[bucket] = running_positive
        negative_suffix[bucket] = running_negative

    results = []
    for index, threshold in enumerate(THRESHOLD_GRID):
        true_positive = positive_suffix[index]
        false_positive = negative_suffix[index]
        result = _metrics_from_confusion(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=positive_total - true_positive,
            true_negative=negative_total - false_positive,
        )
        results.append({"threshold": threshold, **result})
    return results


def select_threshold_from_buckets(
    bucket_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Select maximum F1 and retain the lower threshold on ties."""

    threshold_metrics = threshold_metrics_from_buckets(bucket_rows)
    return max(
        threshold_metrics,
        key=lambda result: (result["f1"], -result["threshold"]),
    )


def calculate_auc_metrics(dataframe: DataFrame) -> Dict[str, float]:
    """Calculate Spark ROC-AUC and PR-AUC from scalar probabilities."""

    evaluator = BinaryClassificationEvaluator(
        labelCol=TARGET_COLUMN,
        rawPredictionCol=PROBABILITY_COLUMN,
    )
    return {
        "roc_auc": evaluator.setMetricName("areaUnderROC").evaluate(dataframe),
        "pr_auc": evaluator.setMetricName("areaUnderPR").evaluate(dataframe),
    }


def evaluate_validation_scores(dataframe: DataFrame) -> Dict[str, Any]:
    """Evaluate one persisted candidate and select its validation threshold."""

    selected = select_threshold_from_buckets(
        aggregate_probability_buckets(dataframe)
    )
    return {**selected, **calculate_auc_metrics(dataframe)}


def evaluate_majority_baseline(
    train_profile: Mapping[str, Any], validation: DataFrame
) -> Dict[str, Any]:
    """Evaluate the train-derived majority classifier on validation only."""

    majority_class = int(
        int(train_profile["positives"]) * 2 > int(train_profile["rows"])
    )
    train_prevalence = float(train_profile["class_prevalence"])
    scored = validation.select(TARGET_COLUMN).withColumn(
        PROBABILITY_COLUMN, F.lit(train_prevalence).cast("double")
    )
    counts = validation.agg(
        F.sum(F.col(TARGET_COLUMN)).cast("long").alias("positives"),
        F.count("*").cast("long").alias("rows"),
    ).first()
    positives = int(counts["positives"] or 0)
    rows = int(counts["rows"])
    if majority_class == 1:
        confusion = _metrics_from_confusion(positives, rows - positives, 0, 0)
    else:
        confusion = _metrics_from_confusion(0, 0, positives, rows - positives)
    return {
        "majority_class": majority_class,
        "constant_probability": train_prevalence,
        **confusion,
        **calculate_auc_metrics(scored),
    }


def train_and_select_models(
    train: DataFrame, validation: DataFrame
) -> ModelSelection:
    """Fit on train and select using validation without accepting test data."""

    train_profile = dataset_profile(train)
    validation_profile = dataset_profile(validation)
    baseline = evaluate_majority_baseline(train_profile, validation)
    preprocessing_model = build_preprocessing_pipeline().fit(train)
    processed_train = preprocessing_model.transform(train)
    processed_validation = preprocessing_model.transform(validation)
    fitted_models = {
        "logistic_regression": build_logistic_regression().fit(processed_train),
        "gbt": build_gbt_classifier().fit(processed_train),
    }
    candidates: Dict[str, CandidateResult] = {}
    for name in MODEL_NAMES:
        scored = score_probabilities(
            fitted_models[name], processed_validation
        ).persist(StorageLevel.MEMORY_AND_DISK)
        try:
            scored.count()
            metrics = evaluate_validation_scores(scored)
        finally:
            scored.unpersist()
        candidates[name] = CandidateResult(
            name=name,
            model=fitted_models[name],
            threshold=float(metrics["threshold"]),
            metrics=metrics,
        )

    selected_name = min(
        MODEL_NAMES,
        key=lambda name: (
            -float(candidates[name].metrics["f1"]),
            MODEL_TIE_PRIORITY[name],
        ),
    )
    return ModelSelection(
        preprocessing_model=preprocessing_model,
        candidates=candidates,
        selected_model_name=selected_name,
        selected_threshold=candidates[selected_name].threshold,
        train_profile=train_profile,
        validation_profile=validation_profile,
        baseline=baseline,
    )


def _model_parameters() -> Dict[str, Dict[str, Any]]:
    return {
        "logistic_regression": {
            "family": "binomial",
            "max_iter": 20,
            "reg_param": 0.1,
            "elastic_net_param": 0.0,
            "tolerance": 1e-6,
        },
        "gbt": {
            "max_iter": 10,
            "max_depth": 4,
            "max_bins": 32,
            "step_size": 0.1,
            "subsampling_rate": 0.5,
            "feature_subset_strategy": "sqrt",
            "max_memory_mb": 128,
            "seed": SEED,
        },
    }


def selection_manifest(selection: ModelSelection) -> Dict[str, Any]:
    """Return the frozen model-selection contract."""

    return {
        "schema_version": 1,
        "phase": "selection_frozen",
        "seed": SEED,
        "predictor_columns": list(PREDICTOR_COLUMNS),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "numeric_columns": list(NUMERIC_COLUMNS),
        "imputed_columns": list(HISTORICAL_FEATURE_COLUMNS),
        "threshold_grid": {
            "minimum": 0.0,
            "maximum": 1.0,
            "step": 0.01,
            "tie_break": "lower_threshold",
        },
        "model_tie_break": "logistic_regression",
        "model_parameters": _model_parameters(),
        "selected_model": selection.selected_model_name,
        "selected_threshold": selection.selected_threshold,
        "model_paths": {
            "preprocessing": "preprocessing",
            "logistic_regression": "logistic_regression",
            "gbt": "gbt",
        },
    }


def validation_metrics_document(selection: ModelSelection) -> Dict[str, Any]:
    """Return validation-only metrics with no test information."""

    return {
        "schema_version": 1,
        "phase": "validation_selection",
        "splits": {
            "train": selection.train_profile,
            "validation": selection.validation_profile,
        },
        "baseline": selection.baseline,
        "models": {
            name: candidate.metrics
            for name, candidate in selection.candidates.items()
        },
        "selection": {
            "model": selection.selected_model_name,
            "threshold": selection.selected_threshold,
            "metric": "f1",
        },
    }


def save_selection(selection: ModelSelection, artifacts: ArtifactPaths) -> None:
    """Persist both candidates and freeze validation selection before test."""

    artifacts.models.mkdir(parents=True, exist_ok=True)
    artifacts.metrics.mkdir(parents=True, exist_ok=True)
    selection.preprocessing_model.write().overwrite().save(
        str(artifacts.preprocessing_model)
    )
    selection.candidates["logistic_regression"].model.write().overwrite().save(
        str(artifacts.logistic_regression_model)
    )
    selection.candidates["gbt"].model.write().overwrite().save(
        str(artifacts.gbt_model)
    )
    write_json(selection_manifest(selection), artifacts.selection_manifest)
    write_json(
        validation_metrics_document(selection), artifacts.validation_metrics
    )


def with_threshold_prediction(
    scored: DataFrame, threshold: float
) -> DataFrame:
    """Apply a frozen probability threshold and select audit columns."""

    return scored.withColumn(
        PREDICTION_COLUMN,
        (F.col(PROBABILITY_COLUMN) >= F.lit(threshold)).cast("int"),
    ).select(*PREDICTION_COLUMNS)


def evaluate_final_test(
    selection: ModelSelection,
    test: DataFrame,
    artifacts: ArtifactPaths,
) -> Dict[str, Any]:
    """Evaluate the already frozen winner on test exactly once."""

    if not artifacts.selection_manifest.is_file():
        raise AssertionError("Selection must be persisted before test evaluation.")
    manifest = read_json(artifacts.selection_manifest)
    if manifest.get("phase") != "selection_frozen":
        raise AssertionError("Selection manifest is not frozen.")
    if manifest.get("selected_model") != selection.selected_model_name:
        raise AssertionError("Persisted selection does not match in-memory selection.")

    test_profile = dataset_profile(test)
    processed_test = selection.preprocessing_model.transform(test)
    scored = score_probabilities(
        selection.selected_candidate.model, processed_test
    ).persist(StorageLevel.MEMORY_AND_DISK)
    try:
        scored.count()
        bucket_metrics = threshold_metrics_from_buckets(
            aggregate_probability_buckets(scored)
        )
        threshold_index = int(round(selection.selected_threshold * 100))
        metrics = {
            **bucket_metrics[threshold_index],
            **calculate_auc_metrics(scored),
        }
        predictions = with_threshold_prediction(
            scored, selection.selected_threshold
        )
        artifacts.predictions.mkdir(parents=True, exist_ok=True)
        predictions.write.mode("overwrite").parquet(
            str(artifacts.final_test_predictions)
        )
    finally:
        scored.unpersist()

    document = {
        "schema_version": 1,
        "phase": "final_test",
        "selection": {
            "model": selection.selected_model_name,
            "threshold": selection.selected_threshold,
        },
        "test": test_profile,
        "metrics": metrics,
    }
    write_json(document, artifacts.final_test_metrics)
    return document


def load_saved_bundle(
    models_directory: PathLike,
) -> tuple[
    PipelineModel,
    Union[LogisticRegressionModel, GBTClassificationModel],
    float,
    str,
]:
    """Load the train-fitted preprocessor and frozen selected classifier."""

    models = Path(models_directory)
    manifest = read_json(models / "selection.json")
    preprocessing = PipelineModel.load(str(models / "preprocessing"))
    selected_name = str(manifest["selected_model"])
    if selected_name == "logistic_regression":
        classifier = LogisticRegressionModel.load(
            str(models / "logistic_regression")
        )
    elif selected_name == "gbt":
        classifier = GBTClassificationModel.load(str(models / "gbt"))
    else:
        raise AssertionError(f"Unsupported selected model: {selected_name}")
    return (
        preprocessing,
        classifier,
        float(manifest["selected_threshold"]),
        selected_name,
    )


def score_saved_model(dataframe: DataFrame, models_directory: PathLike) -> DataFrame:
    """Reload the frozen bundle and score raw M5-shaped rows."""

    preprocessing, classifier, threshold, _ = load_saved_bundle(
        models_directory
    )
    scored = score_probabilities(classifier, preprocessing.transform(dataframe))
    return with_threshold_prediction(scored, threshold)


def run_training(
    spark: SparkSession,
    input_directory: PathLike,
    artifact_root: PathLike,
) -> Dict[str, Any]:
    """Verify the frozen M7 bundle; never refit or automatically score test.

    ``evaluate_final_test`` remains an explicit legacy helper for reproducing
    the already-inspected M5 result in isolated tests.  It is intentionally not
    reachable from the production entry point.
    """

    from airline.ml.finalize_m7 import finalize

    project_root = Path(artifact_root).resolve().parent
    expected_input = project_root / "data/warehouse/dws/ml_flight_features"
    if Path(input_directory).resolve() != expected_input.resolve():
        raise ValueError("Frozen M7 must use data/warehouse/dws/ml_flight_features.")
    return finalize(spark, project_root)


def main() -> None:
    """Run configured frozen-model validation and reproducibility checks."""

    paths = get_project_paths(load_config())
    artifact_root = paths.project_root / "artifacts"
    spark = create_spark_session("airline-m7-frozen-model-workflow")
    print(f"M7 feature input: {paths.dws_ml_flight_features.as_posix()}/")
    print("Frozen M7 artifact root: artifacts/experiments/staged-v2/")
    try:
        result = run_training(
            spark,
            paths.resolve(paths.dws_ml_flight_features),
            artifact_root,
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=_json_ready))
        print("M7 frozen-model verification completed; protected test not evaluated.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

# AI-generated test code: generated with OpenAI Codex.
import inspect
import json
import os
import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pytest
from pyspark.ml.feature import ImputerModel, StringIndexerModel, VectorAssembler
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from airline.common.spark_session import create_spark_session
from airline.dws.build_ml_flight_features import (
    HISTORICAL_FEATURE_COLUMNS,
    LEAKAGE_BLACKLIST,
    OUTPUT_COLUMNS,
    PREDICTOR_COLUMNS,
)
from airline.ml.train_models import (
    ASSEMBLER_INPUT_COLUMNS,
    BASE_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    PREDICTION_COLUMNS,
    ArtifactPaths,
    aggregate_probability_buckets,
    calculate_auc_metrics,
    evaluate_final_test,
    evaluate_validation_scores,
    save_selection,
    score_probabilities,
    score_saved_model,
    select_threshold_from_buckets,
    split_feature_dataset,
    threshold_metrics_from_buckets,
    train_and_select_models,
    validate_training_source,
)


M5_SCHEMA = StructType(
    [
        StructField("flight_date", DateType(), False),
        StructField("reporting_airline", StringType(), False),
        StructField("origin", StringType(), False),
        StructField("destination", StringType(), False),
        StructField("distance", DoubleType(), True),
        StructField("month", IntegerType(), False),
        StructField("day_of_week", IntegerType(), False),
        StructField("scheduled_departure_hour", IntegerType(), False),
        StructField("carrier_delay_rate_30d", DoubleType(), True),
        StructField("origin_delay_rate_30d", DoubleType(), True),
        StructField("destination_delay_rate_30d", DoubleType(), True),
        StructField("route_delay_rate_30d", DoubleType(), True),
        StructField("is_arrival_delayed_15m", IntegerType(), False),
        StructField("split", StringType(), False),
    ]
)


def _row(
    flight_date: date,
    split: str,
    index: int,
    *,
    airline: str,
    origin: str,
    destination: str,
    historical_null: bool = False,
) -> Dict[str, object]:
    target = int(index % 4 == 0 or airline == "DL" and index % 5 == 0)
    history: Optional[float] = None if historical_null else (index % 10) / 10.0
    return {
        "flight_date": flight_date,
        "reporting_airline": airline,
        "origin": origin,
        "destination": destination,
        "distance": 500.0 + index,
        "month": flight_date.month,
        "day_of_week": flight_date.isoweekday(),
        "scheduled_departure_hour": index % 24,
        "carrier_delay_rate_30d": history,
        "origin_delay_rate_30d": history,
        "destination_delay_rate_30d": history,
        "route_delay_rate_30d": history,
        "is_arrival_delayed_15m": target,
        "split": split,
    }


@pytest.fixture(scope="module")
def feature_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for index in range(60):
        rows.append(
            _row(
                date(2023, 1, 1) + timedelta(days=index % 30),
                "train",
                index,
                airline="AA" if index % 2 == 0 else "DL",
                origin="JFK" if index % 3 else "LAX",
                destination="SFO" if index % 2 == 0 else "SEA",
                historical_null=index % 7 == 0,
            )
        )
    for index in range(20):
        rows.append(
            _row(
                date(2024, 5, 1) + timedelta(days=index),
                "validation",
                100 + index,
                airline="VALIDATION_ONLY" if index == 0 else "AA",
                origin="BOS" if index == 0 else "JFK",
                destination="MIA" if index == 0 else "SFO",
                historical_null=index % 6 == 0,
            )
        )
    for index in range(20):
        rows.append(
            _row(
                date(2024, 9, 1) + timedelta(days=index),
                "test",
                200 + index,
                airline="TEST_ONLY" if index == 0 else "DL",
                origin="ORD" if index == 0 else "LAX",
                destination="DFW" if index == 0 else "SEA",
                historical_null=index % 5 == 0,
            )
        )
    return rows


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    session = create_spark_session("airline-m7-integration-tests")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def shared_tmp_path() -> Iterator[Path]:
    shared_root = Path(os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir()))
    shared_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="airline-m7-", dir=shared_root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _dataframe(
    spark: SparkSession, rows: List[Dict[str, object]]
) -> DataFrame:
    return spark.createDataFrame(
        [[row[field.name] for field in M5_SCHEMA.fields] for row in rows],
        M5_SCHEMA,
    ).select(*OUTPUT_COLUMNS)


@pytest.fixture(scope="module")
def features(
    spark: SparkSession, feature_rows: List[Dict[str, object]]
) -> DataFrame:
    return _dataframe(spark, feature_rows).cache()


@pytest.fixture(scope="module")
def trained_selection(features: DataFrame):
    validate_training_source(features)
    train, validation, _ = split_feature_dataset(features)
    return train_and_select_models(train, validation)


def _prediction_values(
    selection, validation: DataFrame, model_name: str
) -> List[tuple[float, float]]:
    processed = selection.preprocessing_model.transform(validation)
    scored = score_probabilities(
        selection.candidates[model_name].model, processed
    )
    threshold = selection.candidates[model_name].threshold
    return [
        (row["probability"], float(row["probability"] >= threshold))
        for row in scored.orderBy("flight_date", "probability")
        .select("probability")
        .collect()
    ]


def test_exact_predictor_contract_and_null_validation(
    spark: SparkSession,
    feature_rows: List[Dict[str, object]],
    features: DataFrame,
) -> None:
    assert tuple(features.columns) == OUTPUT_COLUMNS
    assert set(PREDICTOR_COLUMNS).isdisjoint(LEAKAGE_BLACKLIST)
    assert tuple(inspect.signature(train_and_select_models).parameters) == (
        "train",
        "validation",
    )
    validate_training_source(features)

    invalid_rows = [dict(row) for row in feature_rows]
    invalid_rows[0]["distance"] = None
    with pytest.raises(AssertionError, match="null distance rows"):
        validate_training_source(_dataframe(spark, invalid_rows))

    historical_null_rows = [dict(row) for row in feature_rows]
    for column_name in HISTORICAL_FEATURE_COLUMNS:
        historical_null_rows[0][column_name] = None
    validate_training_source(_dataframe(spark, historical_null_rows))


def test_preprocessing_is_train_fitted_and_uses_only_allowed_predictors(
    features: DataFrame, trained_selection
) -> None:
    train, validation, test = split_feature_dataset(features)
    stages = trained_selection.preprocessing_model.stages
    indexers = [stage for stage in stages if isinstance(stage, StringIndexerModel)]
    imputer = next(stage for stage in stages if isinstance(stage, ImputerModel))
    assembler = next(stage for stage in stages if isinstance(stage, VectorAssembler))

    assert [stage.getInputCol() for stage in indexers] == list(
        CATEGORICAL_COLUMNS
    )
    assert "VALIDATION_ONLY" not in indexers[0].labels
    assert "TEST_ONLY" not in indexers[0].labels
    assert assembler.getInputCols() == list(ASSEMBLER_INPUT_COLUMNS)
    assert set(assembler.getInputCols()).isdisjoint(LEAKAGE_BLACKLIST)
    assert "flight_date" not in assembler.getInputCols()
    assert "split" not in assembler.getInputCols()
    assert "is_arrival_delayed_15m" not in assembler.getInputCols()

    train_means = train.agg(
        *[F.avg(column).alias(column) for column in HISTORICAL_FEATURE_COLUMNS]
    ).first()
    surrogates = imputer.surrogateDF.first()
    for column_name in HISTORICAL_FEATURE_COLUMNS:
        assert surrogates[column_name] == pytest.approx(train_means[column_name])

    assert trained_selection.preprocessing_model.transform(validation).count() == 20
    assert trained_selection.preprocessing_model.transform(test).count() == 20
    assert train.filter(F.col("split") != "train").count() == 0
    assert validation.filter(F.col("split") != "validation").count() == 0
    assert test.filter(F.col("split") != "test").count() == 0


def test_bucketed_threshold_search_and_metrics_are_correct(
    spark: SparkSession,
) -> None:
    scored = spark.createDataFrame(
        [(0, 0.10), (0, 0.20), (1, 0.80), (1, 0.90)],
        ["is_arrival_delayed_15m", "probability"],
    ).cache()
    try:
        bucket_rows = aggregate_probability_buckets(scored)
        all_metrics = threshold_metrics_from_buckets(bucket_rows)
        selected = select_threshold_from_buckets(bucket_rows)
        evaluated = evaluate_validation_scores(scored)

        assert len(all_metrics) == 101
        assert selected["threshold"] == pytest.approx(0.21)
        assert selected["precision"] == pytest.approx(1.0)
        assert selected["recall"] == pytest.approx(1.0)
        assert selected["f1"] == pytest.approx(1.0)
        assert selected["confusion_matrix"] == {
            "true_positive": 2,
            "false_positive": 0,
            "false_negative": 0,
            "true_negative": 2,
        }
        assert evaluated["roc_auc"] == pytest.approx(1.0)
        assert evaluated["pr_auc"] == pytest.approx(1.0)
        assert calculate_auc_metrics(scored) == pytest.approx(
            {"roc_auc": 1.0, "pr_auc": 1.0}
        )
    finally:
        scored.unpersist()


def test_fixed_seed_retraining_is_numerically_reproducible(
    features: DataFrame, trained_selection
) -> None:
    train, validation, _ = split_feature_dataset(features)
    repeated = train_and_select_models(train, validation)

    assert repeated.selected_model_name == trained_selection.selected_model_name
    assert repeated.selected_threshold == pytest.approx(
        trained_selection.selected_threshold
    )
    for model_name in ("logistic_regression", "gbt"):
        first_metrics = trained_selection.candidates[model_name].metrics
        second_metrics = repeated.candidates[model_name].metrics
        assert first_metrics["threshold"] == pytest.approx(
            second_metrics["threshold"]
        )
        for metric_name in (
            "class_prevalence",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ):
            assert first_metrics[metric_name] == pytest.approx(
                second_metrics[metric_name], abs=1e-10
            )
        first_values = _prediction_values(
            trained_selection, validation, model_name
        )
        second_values = _prediction_values(repeated, validation, model_name)
        assert [prediction for _, prediction in first_values] == [
            prediction for _, prediction in second_values
        ]
        assert [probability for probability, _ in first_values] == pytest.approx(
            [probability for probability, _ in second_values], abs=1e-10
        )


def test_artifacts_reload_and_final_test_is_evaluated_after_selection(
    spark: SparkSession,
    features: DataFrame,
    trained_selection,
    shared_tmp_path: Path,
) -> None:
    _, _, test = split_feature_dataset(features)
    artifacts = ArtifactPaths.from_root(shared_tmp_path / "artifacts")
    save_selection(trained_selection, artifacts)

    validation_document = json.loads(
        artifacts.validation_metrics.read_text(encoding="utf-8")
    )
    selection_document = json.loads(
        artifacts.selection_manifest.read_text(encoding="utf-8")
    )
    assert set(validation_document["splits"]) == {"train", "validation"}
    assert "test" not in validation_document
    assert set(validation_document["models"]) == {"logistic_regression", "gbt"}
    assert selection_document["phase"] == "selection_frozen"

    final_document = evaluate_final_test(trained_selection, test, artifacts)
    saved = spark.read.parquet(str(artifacts.final_test_predictions))
    reloaded = score_saved_model(test, artifacts.models)

    assert final_document["phase"] == "final_test"
    assert final_document["selection"]["model"] == selection_document[
        "selected_model"
    ]
    assert final_document["selection"]["threshold"] == pytest.approx(
        selection_document["selected_threshold"]
    )
    assert saved.columns == list(PREDICTION_COLUMNS)
    assert saved.count() == test.count()
    assert saved.filter(~F.col("probability").between(0.0, 1.0)).count() == 0
    assert saved.filter(~F.col("prediction").isin(0, 1)).count() == 0
    assert saved.filter(F.col("split") != "test").count() == 0
    assert saved.exceptAll(reloaded).count() == 0
    assert reloaded.exceptAll(saved).count() == 0

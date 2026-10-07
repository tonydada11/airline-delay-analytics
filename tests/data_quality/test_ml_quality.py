# AI-generated test code: generated with OpenAI Codex.
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
from pyspark.ml.feature import ImputerModel, StringIndexerModel, VectorAssembler
from pyspark.sql import DataFrame, SparkSession, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.spark_session import create_spark_session
from airline.dws.build_ml_flight_features import (
    HISTORICAL_FEATURE_COLUMNS,
    LEAKAGE_BLACKLIST,
    PREDICTOR_COLUMNS,
)
from airline.ml.train_models import (
    ASSEMBLER_INPUT_COLUMNS,
    BASE_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    PREDICTION_COLUMNS,
    SEED,
    ArtifactPaths,
    aggregate_probability_buckets,
    calculate_auc_metrics,
    load_saved_bundle,
    score_saved_model,
    threshold_metrics_from_buckets,
)


EXPECTED_SPLITS = {
    "train": {"rows": 8_946_370, "positives": 1_827_125},
    "validation": {"rows": 2_418_739, "positives": 631_898},
    "test": {"rows": 2_343_541, "positives": 377_642},
}

pytestmark = pytest.mark.skipif(
    os.environ.get("AIRLINE_RUN_FULL_ML_QUALITY") != "1",
    reason="set AIRLINE_RUN_FULL_ML_QUALITY=1 for mandatory M7 acceptance checks",
)


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    session = create_spark_session("airline-m7-full-data-quality")
    yield session
    session.stop()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_profiles(features: DataFrame) -> Dict[str, Dict[str, float | int]]:
    rows = features.groupBy("split").agg(
        F.count("*").cast("long").alias("rows"),
        F.sum("is_arrival_delayed_15m").cast("long").alias("positives"),
    ).collect()
    return {
        row["split"]: {
            "rows": row["rows"],
            "positives": row["positives"],
            "negatives": row["rows"] - row["positives"],
            "class_prevalence": row["positives"] / row["rows"],
        }
        for row in rows
    }


def _assert_metric_documents_match(
    actual: Dict[str, Any], expected: Dict[str, Any]
) -> None:
    assert actual["confusion_matrix"] == expected["confusion_matrix"]
    assert actual["threshold"] == pytest.approx(expected["threshold"])
    for metric_name in (
        "class_prevalence",
        "precision",
        "recall",
        "f1",
    ):
        assert actual[metric_name] == pytest.approx(
            expected[metric_name], abs=1e-12
        )
    for metric_name in ("roc_auc", "pr_auc"):
        assert actual[metric_name] == pytest.approx(
            expected[metric_name], abs=1e-6
        )


def test_full_ml_artifacts_quality_metrics_and_reload_scoring(
    spark: SparkSession,
) -> None:
    paths = get_project_paths(load_config())
    artifacts = ArtifactPaths.from_root(paths.project_root / "artifacts")
    features = spark.read.parquet(
        str(paths.resolve(paths.dws_ml_flight_features))
    )
    validation_document = _read_json(artifacts.validation_metrics)
    selection_document = _read_json(artifacts.selection_manifest)
    final_document = _read_json(artifacts.final_test_metrics)
    predictions = spark.read.parquet(str(artifacts.final_test_predictions))

    split_profiles = _split_profiles(features)
    for split_name, expected in EXPECTED_SPLITS.items():
        assert split_profiles[split_name]["rows"] == expected["rows"]
        assert split_profiles[split_name]["positives"] == expected["positives"]
    assert set(split_profiles) == set(EXPECTED_SPLITS)

    assert selection_document["phase"] == "selection_frozen"
    assert selection_document["seed"] == SEED
    assert selection_document["predictor_columns"] == list(PREDICTOR_COLUMNS)
    assert set(selection_document["predictor_columns"]).isdisjoint(
        LEAKAGE_BLACKLIST
    )
    assert selection_document["threshold_grid"] == {
        "minimum": 0.0,
        "maximum": 1.0,
        "step": 0.01,
        "tie_break": "lower_threshold",
    }
    assert selection_document["model_tie_break"] == "logistic_regression"
    assert selection_document["selected_model"] in {
        "logistic_regression",
        "gbt",
    }
    assert 0.0 <= selection_document["selected_threshold"] <= 1.0

    assert validation_document["phase"] == "validation_selection"
    assert set(validation_document["splits"]) == {"train", "validation"}
    assert "test" not in validation_document
    assert set(validation_document["models"]) == {
        "logistic_regression",
        "gbt",
    }
    assert validation_document["selection"] == {
        "model": selection_document["selected_model"],
        "threshold": selection_document["selected_threshold"],
        "metric": "f1",
    }
    for split_name in ("train", "validation"):
        saved_profile = validation_document["splits"][split_name]
        assert saved_profile["rows"] == split_profiles[split_name]["rows"]
        assert saved_profile["positives"] == split_profiles[split_name][
            "positives"
        ]
        assert saved_profile["class_prevalence"] == pytest.approx(
            split_profiles[split_name]["class_prevalence"]
        )
    for result in (
        validation_document["baseline"],
        *validation_document["models"].values(),
    ):
        assert 0.0 <= result["class_prevalence"] <= 1.0
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0
        assert 0.0 <= result["roc_auc"] <= 1.0
        assert 0.0 <= result["pr_auc"] <= 1.0

    preprocessing, _, loaded_threshold, loaded_name = load_saved_bundle(
        artifacts.models
    )
    assert loaded_name == selection_document["selected_model"]
    assert loaded_threshold == pytest.approx(
        selection_document["selected_threshold"]
    )
    stages = preprocessing.stages
    indexers = [stage for stage in stages if isinstance(stage, StringIndexerModel)]
    imputer = next(stage for stage in stages if isinstance(stage, ImputerModel))
    assembler = next(stage for stage in stages if isinstance(stage, VectorAssembler))
    train = features.filter(F.col("split") == "train")
    for column_name, indexer in zip(CATEGORICAL_COLUMNS, indexers):
        train_labels = [
            row[column_name]
            for row in train.select(column_name)
            .distinct()
            .orderBy(column_name)
            .collect()
        ]
        assert indexer.labels == train_labels
    assert assembler.getInputCols() == list(ASSEMBLER_INPUT_COLUMNS)
    assert set(assembler.getInputCols()).isdisjoint(LEAKAGE_BLACKLIST)
    assert set(BASE_NUMERIC_COLUMNS).issubset(assembler.getInputCols())
    train_means = train.agg(
        *[F.avg(column).alias(column) for column in HISTORICAL_FEATURE_COLUMNS]
    ).first()
    surrogates = imputer.surrogateDF.first()
    for column_name in HISTORICAL_FEATURE_COLUMNS:
        assert surrogates[column_name] == pytest.approx(
            train_means[column_name], abs=1e-12
        )

    test = features.filter(F.col("split") == "test")
    assert final_document["phase"] == "final_test"
    assert final_document["selection"] == {
        "model": selection_document["selected_model"],
        "threshold": selection_document["selected_threshold"],
    }
    assert final_document["test"]["rows"] == EXPECTED_SPLITS["test"]["rows"]
    assert final_document["test"]["positives"] == EXPECTED_SPLITS["test"][
        "positives"
    ]
    assert predictions.columns == list(PREDICTION_COLUMNS)
    assert predictions.count() == EXPECTED_SPLITS["test"]["rows"]
    assert predictions.filter(F.col("split") != "test").count() == 0
    assert predictions.filter(
        F.col("probability").isNull()
        | F.isnan("probability")
        | ~F.col("probability").between(0.0, 1.0)
    ).count() == 0
    assert predictions.filter(~F.col("prediction").isin(0, 1)).count() == 0
    prediction_targets = {
        row["is_arrival_delayed_15m"]: row["count"]
        for row in predictions.groupBy("is_arrival_delayed_15m").count().collect()
    }
    assert prediction_targets == {
        0: EXPECTED_SPLITS["test"]["rows"]
        - EXPECTED_SPLITS["test"]["positives"],
        1: EXPECTED_SPLITS["test"]["positives"],
    }

    bucket_metrics = threshold_metrics_from_buckets(
        aggregate_probability_buckets(predictions)
    )
    threshold_index = int(round(loaded_threshold * 100))
    independent_metrics = {
        **bucket_metrics[threshold_index],
        **calculate_auc_metrics(predictions),
    }
    _assert_metric_documents_match(
        final_document["metrics"], independent_metrics
    )

    reloaded_predictions = score_saved_model(test, artifacts.models)
    first_minus_reload = predictions.exceptAll(reloaded_predictions).limit(1).count()
    reload_minus_first = reloaded_predictions.exceptAll(predictions).limit(1).count()
    assert first_minus_reload == 0
    assert reload_minus_first == 0
    assert not any(
        path.is_dir() for path in artifacts.final_test_predictions.iterdir()
    )

    print(f"ml_split_profiles={split_profiles}")
    print(f"validation_baseline={validation_document['baseline']}")
    print(f"validation_models={validation_document['models']}")
    print(f"selected_model={loaded_name}")
    print(f"selected_threshold={loaded_threshold}")
    print(f"final_test_metrics={final_document['metrics']}")
    print(f"prediction_rows={predictions.count()}")
    print(f"prediction_target_counts={prediction_targets}")
    print(f"first_minus_reload_rows={first_minus_reload}")
    print(f"reload_minus_first_rows={reload_minus_first}")

# AI-generated test code: generated with OpenAI Codex.
"""Full-data M7 acceptance checks. Protected test data is never loaded."""

import json
import os
from pathlib import Path

import pytest
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql import functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.spark_session import create_spark_session
from airline.dws.build_m7_ml_flight_features import validate_m7_feature_frame
from airline.ml.finalize_m7 import protected_hashes
from airline.ml.m7_contract import (
    CLASSIFICATION_THRESHOLD,
    FEATURE_CONTRACT_VERSION,
    MAX_ITER,
    MODEL_FAMILY,
    OUTPUT_COLUMNS,
    POSITIVE_CLASS_WEIGHT,
    REG_PARAM,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("AIRLINE_RUN_FULL_ML_QUALITY") != "1",
    reason="set AIRLINE_RUN_FULL_ML_QUALITY=1 for mandatory M7 acceptance checks",
)


@pytest.fixture(scope="module")
def spark():
    session = create_spark_session("airline-m7-full-data-quality")
    yield session
    session.stop()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_dws_contract_and_chronology(spark):
    paths = get_project_paths(load_config())
    features = spark.read.parquet(str(paths.resolve(paths.dws_ml_flight_features))).select(*OUTPUT_COLUMNS)
    validate_m7_feature_frame(features)
    assert tuple(features.columns) == OUTPUT_COLUMNS
    profiles = {row["split"]: row.asDict() for row in features.groupBy("split").agg(
        F.count("*").alias("rows"), F.min("flight_date").alias("minimum"), F.max("flight_date").alias("maximum")
    ).collect()}
    assert set(profiles) == {"train", "validation"}
    assert str(profiles["train"]["minimum"]) == "2023-01-01"
    assert str(profiles["train"]["maximum"]) == "2024-04-30"
    assert str(profiles["validation"]["minimum"]) == "2024-05-01"
    assert str(profiles["validation"]["maximum"]) == "2024-08-31"
    assert profiles["train"]["rows"] == 8_946_370
    assert profiles["validation"]["rows"] == 2_418_739


def test_frozen_manifest_models_and_reload_evidence(spark):
    root = Path("artifacts/experiments/staged-v2")
    manifest = _json(root / "final_model/manifest.json")
    assert manifest["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    assert manifest["model"] == {
        "family": MODEL_FAMILY,
        "maxIter": MAX_ITER,
        "regParam": REG_PARAM,
        "elasticNetParam": 0.0,
        "positive_class_weight": POSITIVE_CLASS_WEIGHT,
        "threshold": CLASSIFICATION_THRESHOLD,
        "seed": 42,
    }
    PipelineModel.load(str(root / "final_model/preprocessing"))
    LogisticRegressionModel.load(str(root / "final_model/classifier"))
    reload = _json(root / "m7/reload_verification.json")
    assert reload["status"] == "pass"
    assert reload["refit_performed"] is False
    assert reload["rounded_12dp_multiset_differences"] == {
        "reloaded_minus_saved": 0,
        "saved_minus_reloaded": 0,
    }
    assert reload["confusion_matrix_exact_match"] is True
    assert reload["confusion_metrics_reconcile"] is True
    assert max(reload["auc_absolute_differences"].values()) < 1e-6
    assert reload["auc_input"] == "continuous probability/model score"


def test_execution_artifacts_idempotency_and_test_protection():
    project = Path.cwd()
    root = project / "artifacts/experiments/staged-v2/m7"
    complete = _json(root / "complete.json")
    assert complete["status"] == "PASS"
    assert complete["test_evaluation_executed"] is False
    before = _json(root / "test_hashes_before.json")["sha256"]
    after = _json(root / "test_hashes_after.json")["sha256"]
    assert before == after == protected_hashes(project)
    for relative in (
        "ablation.json",
        "monthly_validation.json",
        "spark/physical_plan_summary.json",
        "spark/partition_summary.json",
        "spark/persistence_summary.json",
        "spark/shuffle_summary.json",
        "spark/runtime_summary.json",
    ):
        assert (root / relative).is_file()
    assert _json(root / "spark/partition_summary.json")["accidental_single_partition_bottleneck"] is False

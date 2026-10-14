# AI-generated test code: generated with OpenAI Codex.
from datetime import date, timedelta
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import pytest
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.feature import ImputerModel, StringIndexerModel
from pyspark.sql import functions as F
from airline.common.spark_session import create_spark_session
from airline.common.raw_snapshot import SNAPSHOT_ID
from airline.dws.ml_feature_contract import *
from airline.ml import model_contract as contract
from airline.ml.train import train
from airline.ml.evaluate import evaluate, score, with_threshold_prediction
from airline.ml.metrics import calculate_metrics


@pytest.fixture(scope="module")
def spark():
    session = create_spark_session("airline-production-ml-tests")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def workspace():
    root = Path(os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir()))
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="airline-ml-", dir=root))
    yield path
    shutil.rmtree(path)


@pytest.fixture(scope="module")
def trained(spark, workspace):
    rows = []
    for i in range(50):
        split = "train" if i < 40 else "validation" if i < 45 else "test"
        day = date(2023, 1, 1) + timedelta(days=i) if split == "train" else date(2024, 5 if split == "validation" else 9, i % 5 + 1)
        row = {"flight_date": day, "split": split, TARGET_COLUMN: i % 2}
        row.update({c: ("ZZ_UNSEEN" if split != "train" else "BB" if i % 2 else "AA") for c in CATEGORICAL_COLUMNS})
        row.update({c: float(i % 3 + 1) if split == "train" else 9999.0 for c in SCHEDULE_NUMERIC})
        row.update({c: 0.0 for c in ("departure_missing", "arrival_missing", "duration_missing")})
        row.update({c: 0.3 for c in HISTORY_RATES})
        row.update({c: 10 for c in HISTORY_COUNTS})
        row.update({c: 0.0 for c in HISTORY_MISSING})
        rows.append(row)
    features = spark.createDataFrame(rows).select(*OUTPUT_COLUMNS)
    features.write.partitionBy("split").parquet(str(workspace / "features"))
    snapshot = {"source_snapshot_id": SNAPSHOT_ID, "raw_manifest_sha256": "a" * 64}
    manifest = train(spark, workspace / "features", workspace / "artifacts/models", snapshot, Path(__file__).resolve().parents[2])
    return features, manifest


def test_train_only_preprocessing_and_save_reload(spark, workspace, trained):
    features, manifest = trained
    preprocessing, classifier, loaded = contract.load_bundle(workspace / "artifacts/models", "a" * 64)
    assert loaded == manifest
    class WrongDimensions:
        numClasses = 2
        numFeatures = classifier.numFeatures + 1
        def getOrDefault(self, key): return classifier.getOrDefault(key)
    with pytest.raises(ValueError, match="dimensions"):
        contract.validate_models(preprocessing, WrongDimensions())
    assert manifest["training"]["rows"] == 40
    assert manifest["training"]["max_date"] == "2023-02-09"
    assert manifest["reproducibility_seed"] == 42
    for stage in preprocessing.stages:
        if isinstance(stage, StringIndexerModel): assert stage.labels == ["AA", "BB"]
        if isinstance(stage, ImputerModel):
            means = features.filter("split = 'train'").agg(*[F.avg(c).alias(c) for c in NUMERIC_COLUMNS]).first()
            for column in NUMERIC_COLUMNS:
                assert stage.surrogateDF.first()[column] == pytest.approx(means[column])
    unseen = features.filter("split = 'validation'").withColumn("distance", F.lit(None).cast("double"))
    before = score(preprocessing, classifier, unseen).orderBy("flight_date").collect()
    preprocessing.write().save(str(workspace / "reload-pre"))
    classifier.write().save(str(workspace / "reload-lr"))
    after = score(PipelineModel.load(str(workspace / "reload-pre")), LogisticRegressionModel.load(str(workspace / "reload-lr")), unseen).orderBy("flight_date").collect()
    assert [r.probability for r in before] == pytest.approx([r.probability for r in after], abs=1e-12)


def test_evaluation_never_fits_and_reports_bundle(spark, workspace, trained, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("Evaluation invoked fit")
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    for split in ("validation", "test"):
        report = evaluate(spark, workspace / "features", workspace / "artifacts", split, "a" * 64)
        assert report["rows"] == 5
        assert report["bundle_id"] == trained[1]["bundle_id"]
        assert "previously inspected" in report["historical_test_status"]
        predictions = spark.read.parquet(str(workspace / "artifacts/predictions" / split))
        assert tuple(predictions.columns) == contract.PREDICTION_COLUMNS
        assert predictions.count() == report["rows"]
        assert {r.split for r in predictions.select("split").distinct().collect()} == {split}


def test_threshold_and_continuous_ranking(spark):
    scores = spark.createDataFrame([(date(2024,5,1), "validation", y, p) for y,p in
                                   [(0,0.33999), (1,0.34), (0,0.341), (1,0.9)]],
                                  ["flight_date", "split", TARGET_COLUMN, "probability"])
    predictions = with_threshold_prediction(scores)
    assert [r.prediction for r in predictions.orderBy("probability").collect()] == [0,1,1,1]
    report = calculate_metrics(predictions)
    assert report["confusion_matrix"] == dict(true_positive=2, false_positive=1, false_negative=0, true_negative=1)
    assert report["roc_auc"] == pytest.approx(0.75)
    assert report["pr_auc"] == pytest.approx(19 / 24)
    # These two scores would tie if bucketed to hundredths.
    close = spark.createDataFrame([(0,0.341),(1,0.342)], [TARGET_COLUMN,"probability"])
    assert calculate_metrics(close)["roc_auc"] == 1
    assert calculate_metrics(close.withColumn(TARGET_COLUMN,F.lit(1)))["roc_auc"] is None
    with pytest.raises(ValueError, match="empty"): calculate_metrics(scores.limit(0))


def test_invalid_and_absent_manifests_fail(workspace, trained):
    manifest = copy.deepcopy(trained[1])
    for key, value in (("feature_order", []), ("threshold", 0.5), ("model_settings", {}), ("training", {})):
        changed = {**manifest, key: value}
        with pytest.raises(ValueError): contract.validate_manifest(changed)
    with pytest.raises(ValueError, match="fingerprint"):
        contract.load_bundle(workspace / "artifacts/models", "b" * 64)
    with pytest.raises(ValueError, match="manifest"):
        contract.load_bundle(workspace / "absent")


def test_training_partitioning_and_persistence_released_on_failure(spark, workspace, trained, monkeypatch):
    import importlib
    workflow = importlib.import_module("airline.ml.train")
    preprocessing, _, _ = contract.load_bundle(workspace / "artifacts/models")
    class Preprocessing:
        def fit(self, frame):
            assert {r.split for r in frame.select("split").distinct().collect()} == {"train"}
            return preprocessing
    class FailingEstimator:
        def fit(self, frame):
            assert frame.rdd.getNumPartitions() == 2
            assert frame.storageLevel.useDisk and not frame.storageLevel.useMemory
            assert {r.class_weight for r in frame.filter(F.col(TARGET_COLUMN) == 1).select("class_weight").distinct().collect()} == {2.0}
            raise RuntimeError("injected fitting failure")
    monkeypatch.setattr(workflow, "build_preprocessing_pipeline", Preprocessing)
    monkeypatch.setattr(workflow, "build_logistic_regression", FailingEstimator)
    before = spark.sparkContext._jsc.sc().getPersistentRDDs().size()
    with pytest.raises(RuntimeError, match="injected"):
        workflow.train(spark, workspace / "features", workspace / "failed-model", {}, workspace)
    assert spark.sparkContext._jsc.sc().getPersistentRDDs().size() == before
    assert not (workspace / "failed-model").exists()

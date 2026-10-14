# AI-generated test code: generated with OpenAI Codex.
"""Production bundle and natural-prevalence prediction reconciliation."""
import json
import os
import pytest
from pyspark.sql import functions as F
from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.raw_snapshot import sha256_file
from airline.common.spark_session import create_spark_session
from airline.dws.ml_feature_contract import OUTPUT_COLUMNS, TARGET_COLUMN, HISTORICAL_TEST_STATUS
from airline.ml.model_contract import load_bundle, PREDICTION_COLUMNS, CLASSIFICATION_THRESHOLD
from airline.ml.metrics import calculate_metrics
from airline.ml.evaluate import score

pytestmark = pytest.mark.skipif(os.environ.get("AIRLINE_RUN_FULL_ML_QUALITY") != "1", reason="full ML quality is opt-in")


@pytest.fixture(scope="module")
def context():
    paths = get_project_paths(load_config())
    spark = create_spark_session("airline-artifact-quality")
    root = paths.resolve(paths.artifact_root)
    bundle = load_bundle(root / "models", sha256_file(paths.resolve(paths.raw_manifest)))
    yield spark, paths, root, bundle
    spark.stop()


def test_training_bundle(context):
    spark, paths, _, (_, _, manifest) = context
    features = spark.read.parquet(str(paths.resolve(paths.dws_ml_flight_features))).filter("split = 'train'")
    stats = features.agg(F.count("*"), F.min("flight_date"), F.max("flight_date")).first()
    assert manifest["training"] == {"rows": stats[0], "min_date": str(stats[1]), "max_date": str(stats[2])}


@pytest.mark.parametrize("split", ["validation", "test"])
def test_predictions_metrics_and_reload(context, split):
    spark, paths, root, (preprocessing, classifier, manifest) = context
    predictions = spark.read.parquet(str(root / "predictions" / split))
    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    features = spark.read.parquet(str(paths.resolve(paths.dws_ml_flight_features))).filter(F.col("split") == split).select(*OUTPUT_COLUMNS)
    assert predictions.count() == features.count()
    invalid = (F.col("probability").isNull() | F.isnan("probability") | ~F.col("probability").between(0,1)
               | F.col("prediction").isNull() | (F.col("prediction") != (F.col("probability") >= CLASSIFICATION_THRESHOLD).cast("int")))
    assert predictions.filter(invalid).limit(1).count() == 0
    keys = ["flight_date", "split", TARGET_COLUMN]
    a, b = features.groupBy(*keys).count(), predictions.groupBy(*keys).count()
    assert a.exceptAll(b).limit(1).count() == 0
    assert b.exceptAll(a).limit(1).count() == 0
    report = json.loads((root / "metrics" / f"{split}.json").read_text())
    assert report["bundle_id"] == manifest["bundle_id"]
    assert report["model_manifest_sha256"] == sha256_file(root / "models/manifest.json")
    assert report["historical_test_status"] == HISTORICAL_TEST_STATUS
    for key, value in calculate_metrics(predictions).items():
        assert report[key] == pytest.approx(value, abs=1e-10) if isinstance(value, float) else report[key] == value
    # Rescore a complete day so duplicate flights remain reconcilable as a multiset.
    day = features.agg(F.min("flight_date")).first()[0]
    saved = predictions.filter(F.col("flight_date") == day)
    reloaded = score(preprocessing, classifier, features.filter(F.col("flight_date") == day))
    columns = ["flight_date", "split", TARGET_COLUMN, F.round("probability", 10).alias("probability"), "prediction"]
    assert saved.select(*columns).exceptAll(reloaded.select(*columns)).limit(1).count() == 0
    assert reloaded.select(*columns).exceptAll(saved.select(*columns)).limit(1).count() == 0

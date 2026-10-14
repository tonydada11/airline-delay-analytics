"""Explicit validation/test scoring; no fitting or model selection."""
import argparse
from pyspark import StorageLevel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.raw_snapshot import sha256_file
from airline.common.spark_session import create_spark_session
from airline.dws.build_ml_flight_features import validate_feature_frame
from airline.dws.ml_feature_contract import OUTPUT_COLUMNS, TARGET_COLUMN, HISTORICAL_TEST_STATUS
from airline.ml.model_contract import CLASSIFICATION_THRESHOLD, PREDICTION_COLUMNS, load_bundle, write_json
from airline.ml.metrics import calculate_metrics


def with_threshold_prediction(scored):
    return scored.withColumn("prediction", (F.col("probability") >= CLASSIFICATION_THRESHOLD).cast("int")).select(*PREDICTION_COLUMNS)


def score(preprocessing, classifier, features):
    return with_threshold_prediction(classifier.transform(preprocessing.transform(features)).select(
        "flight_date", "split", TARGET_COLUMN, vector_to_array("probability")[1].alias("probability")))


def evaluate(spark, input_directory, artifact_root, split, expected_fingerprint=None):
    if split not in ("validation", "test"):
        raise ValueError("Evaluation requires --split validation|test")
    preprocessing, classifier, manifest = load_bundle(artifact_root / "models", expected_fingerprint)
    features = spark.read.parquet(str(input_directory)).filter(F.col("split") == split)
    validate_feature_frame(features)
    predictions = score(preprocessing, classifier, features).persist(StorageLevel.DISK_ONLY)
    try:
        metrics = calculate_metrics(predictions)
        predictions.write.mode("overwrite").parquet(str(artifact_root / "predictions" / split))
        report = {"split": split, "bundle_id": manifest["bundle_id"],
                  "model_manifest_sha256": sha256_file(artifact_root / "models/manifest.json"),
                  "threshold": CLASSIFICATION_THRESHOLD, "historical_test_status": HISTORICAL_TEST_STATUS,
                  **metrics}
        write_json(artifact_root / "metrics" / f"{split}.json", report)
        return report
    finally:
        predictions.unpersist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    args = parser.parse_args()
    paths = get_project_paths(load_config())
    spark = create_spark_session(f"airline-evaluate-{args.split}")
    try:
        print(evaluate(spark, paths.resolve(paths.dws_ml_flight_features), paths.resolve(paths.artifact_root), args.split,
                       sha256_file(paths.resolve(paths.raw_manifest))))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

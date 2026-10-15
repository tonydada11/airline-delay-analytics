"""Explicit validation/test scoring; no fitting or model selection."""
import argparse
from pathlib import Path
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.raw_snapshot import sha256_file
from airline.common.spark_session import create_spark_session
from airline.dws.build_ml_flight_features import validate_feature_frame
from airline.dws.ml_feature_contract import TARGET_COLUMN, HISTORICAL_TEST_STATUS
from airline.ml.model_contract import PREDICTION_COLUMNS, load_bundle
from airline.common.lineage import BUSINESS_KEY
from airline.dws.dataset_manifest import validate_dataset_manifest
from airline.ml.metrics import calculate_metrics
from airline.ml.evaluation_artifacts import complete_evaluation, evaluation_identity


def with_threshold_prediction(scored, threshold):
    return scored.withColumn("prediction", (F.col("probability") >= threshold).cast("int")).select(*PREDICTION_COLUMNS)


def score(preprocessing, classifier, features, threshold):
    return with_threshold_prediction(classifier.transform(preprocessing.transform(features)).select(
        "flight_id", *BUSINESS_KEY, "split", TARGET_COLUMN, vector_to_array("probability")[1].alias("probability")), threshold)


def evaluate(spark, input_directory, artifact_root, split, expected_fingerprint=None):
    input_directory, artifact_root = Path(input_directory), Path(artifact_root)
    if split not in ("validation", "test"):
        raise ValueError("Evaluation requires --split validation|test")
    preprocessing, classifier, manifest = load_bundle(artifact_root / "models", expected_fingerprint)
    dataset = spark.read.parquet(str(input_directory))
    dataset_manifest = validate_dataset_manifest(input_directory, dataset, expected_fingerprint or manifest["raw_manifest_sha256"],
                                                 phase='test' if split == 'test' else 'selection')
    if split == 'test' and dataset_manifest.get('selection_bundle_id') != manifest['bundle_id']:
        raise ValueError('Test features belong to a different frozen selection bundle')
    if split == 'validation' and sha256_file(input_directory / '_dataset_manifest.json') != manifest['selection']['dataset_manifest_sha256']:
        raise ValueError('Selection feature dataset changed')
    threshold = manifest['threshold_step'] / 100
    report_path = artifact_root / 'metrics' / f'{split}.json'
    features = dataset.filter(F.col("split") == split)
    validate_feature_frame(features)
    identity = evaluation_identity(input_directory,
        {'preprocessing': artifact_root / 'models/preprocessing',
         'classifier': artifact_root / 'models/classifier'},
        {'model': artifact_root / 'models/manifest.json',
         'dataset': input_directory / '_dataset_manifest.json'}, split, manifest['threshold_step'])

    def make_report(predictions):
        return {"split": split, "bundle_id": manifest["bundle_id"],
                "model_manifest_sha256": identity['manifests']['model'],
                "selected_classifier": manifest["selected_classifier"], "threshold_step": manifest["threshold_step"],
                "dataset_manifest_sha256": identity['manifests']['dataset'],
                "threshold": threshold, "historical_test_status": HISTORICAL_TEST_STATUS,
                **calculate_metrics(predictions, threshold)}

    return complete_evaluation(spark, features, threshold, report_path,
        artifact_root / 'predictions' / split, identity,
        lambda: score(preprocessing, classifier, features, threshold), make_report)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    args = parser.parse_args()
    paths = get_project_paths(load_config())
    spark = create_spark_session(f"airline-evaluate-{args.split}")
    try:
        input_path = paths.dws_ml_test_features if args.split == 'test' else paths.dws_ml_flight_features
        print(evaluate(spark, paths.resolve(input_path), paths.resolve(paths.artifact_root), args.split,
                       sha256_file(paths.resolve(paths.raw_manifest))))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

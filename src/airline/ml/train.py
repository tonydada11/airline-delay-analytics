"""Fit one frozen model on train, publish its bundle, and stop."""
import math
from pyspark import StorageLevel
from pyspark.sql import functions as F
from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.raw_snapshot import verify_snapshot
from airline.common.spark_session import create_spark_session
from airline.dws.build_ml_flight_features import validate_feature_frame
from airline.dws.ml_feature_contract import OUTPUT_COLUMNS, TARGET_COLUMN
from airline.ml.preprocessing import build_preprocessing_pipeline
from airline.ml.model_contract import POSITIVE_CLASS_WEIGHT, build_logistic_regression, make_manifest, publish_bundle


def train(spark, input_directory, bundle_directory, snapshot, project_root):
    training = spark.read.parquet(str(input_directory)).filter(F.col("split") == "train")
    rows = validate_feature_frame(training)
    dates = training.agg(F.min("flight_date"), F.max("flight_date")).first()
    profile = {"rows": rows, "min_date": str(dates[0]), "max_date": str(dates[1])}
    preprocessing = build_preprocessing_pipeline().fit(training)
    prepared = (preprocessing.transform(training)
                .withColumn("class_weight", F.when(F.col(TARGET_COLUMN) == 1, POSITIVE_CLASS_WEIGHT).otherwise(1.0))
                .select("features", TARGET_COLUMN, "class_weight")
                .repartition(max(2, math.ceil(rows / 250000))).persist(StorageLevel.DISK_ONLY))
    try:
        prepared.count()
        classifier = build_logistic_regression().fit(prepared)
        manifest = make_manifest(spark, profile, snapshot, project_root)
        publish_bundle(bundle_directory, preprocessing, classifier, manifest)
        return manifest
    finally:
        prepared.unpersist()


def main():
    paths = get_project_paths(load_config())
    snapshot = verify_snapshot(paths.resolve(paths.raw_csv), paths.resolve(paths.raw_manifest))
    spark = create_spark_session("airline-train")
    try:
        result = train(spark, paths.resolve(paths.dws_ml_flight_features), paths.resolve(paths.artifact_root) / "models", snapshot, paths.project_root)
        print(f"Published bundle {result['bundle_id']}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

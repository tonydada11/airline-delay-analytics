"""Select on train/validation, then publish the exact selected fitted model."""
import argparse
import uuid
from pathlib import Path
from pyspark.ml import PipelineModel
from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.raw_snapshot import verify_snapshot
from airline.common.spark_session import create_spark_session
from airline.ml.experiments import select, run_directory
from airline.ml.model_contract import CLASSIFIERS, make_manifest, publish_bundle


def train(spark, input_directory, bundle_directory, snapshot, project_root, run_id=None):
    bundle_directory = Path(bundle_directory)
    run_id = run_id or 'production-' + uuid.uuid4().hex
    artifact_root = bundle_directory.parent
    selection = select(spark, input_directory, artifact_root, run_id, snapshot, project_root)
    directory = run_directory(artifact_root, run_id)
    preprocessing = PipelineModel.load(str(directory / 'preprocessing'))
    classifier = CLASSIFIERS[selection['selected_classifier']].load(str(directory / selection['selected_classifier']))
    manifest = make_manifest(spark, selection['training'], snapshot, project_root, selection)
    publish_bundle(bundle_directory, preprocessing, classifier, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    paths = get_project_paths(load_config())
    snapshot = verify_snapshot(paths.resolve(paths.raw_csv), paths.resolve(paths.raw_manifest))
    spark = create_spark_session('airline-select-and-publish')
    try:
        result = train(spark, paths.resolve(paths.dws_ml_flight_features), paths.resolve(paths.artifact_root) / 'models', snapshot, paths.project_root, args.run_id)
        print(f"Published bundle {result['bundle_id']}: {result['selected_classifier']} at step {result['threshold_step']}")
    finally:
        spark.stop()


if __name__ == '__main__':
    main()

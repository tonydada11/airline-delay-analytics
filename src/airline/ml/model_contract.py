"""Frozen model settings and validated, sequential local-filesystem publication."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import uuid

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from airline.common.raw_snapshot import SNAPSHOT_ID, sha256_file
from airline.dws.ml_feature_contract import (
    FEATURE_CONTRACT_VERSION, FEATURE_COLUMNS, SPLIT_BOUNDARIES, TARGET_COLUMN,
    HISTORICAL_TEST_STATUS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, HISTORY_COUNTS, HISTORY_MISSING,
)
from airline.ml.preprocessing import build_preprocessing_pipeline

CLASSIFICATION_THRESHOLD = 0.34
POSITIVE_CLASS_WEIGHT = 2.0
REPRODUCIBILITY_SEED = 42
MODEL_SETTINGS = dict(maxIter=100, regParam=0.1, elasticNetParam=0.0, tol=1e-6,
                      family="binomial", featuresCol="features", labelCol=TARGET_COLUMN,
                      weightCol="class_weight", threshold=CLASSIFICATION_THRESHOLD)
PREDICTION_COLUMNS = ("flight_date", "split", TARGET_COLUMN, "probability", "prediction")


def build_logistic_regression():
    return LogisticRegression(**MODEL_SETTINGS)


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def code_provenance(project_root):
    root = Path(project_root)
    digest = hashlib.sha256()
    for path in sorted((root / "src/airline").rglob("*.py")):
        if "performance" in path.relative_to(root / "src/airline").parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    provenance = {"production_python_sha256": digest.hexdigest(), "git_commit": None, "git_dirty": None}
    try:
        provenance["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        provenance["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return provenance


def make_manifest(spark, profile, snapshot, project_root):
    return {
        "schema_version": 1, "bundle_id": uuid.uuid4().hex,
        "feature_version": FEATURE_CONTRACT_VERSION, "feature_order": list(FEATURE_COLUMNS),
        "split_boundaries": SPLIT_BOUNDARIES, "model_settings": MODEL_SETTINGS,
        "positive_class_weight": POSITIVE_CLASS_WEIGHT, "threshold": CLASSIFICATION_THRESHOLD,
        "reproducibility_seed": REPRODUCIBILITY_SEED, "training": profile,
        **snapshot, "spark_version": spark.version, "code_provenance": code_provenance(project_root),
        "historical_test_status": HISTORICAL_TEST_STATUS,
    }


def validate_manifest(manifest, expected_fingerprint=None):
    if not isinstance(manifest, dict):
        raise ValueError("Model manifest must be an object")
    expected = {"schema_version": 1, "feature_version": FEATURE_CONTRACT_VERSION,
                "feature_order": list(FEATURE_COLUMNS), "split_boundaries": SPLIT_BOUNDARIES,
                "model_settings": MODEL_SETTINGS, "positive_class_weight": POSITIVE_CLASS_WEIGHT,
                "threshold": CLASSIFICATION_THRESHOLD, "reproducibility_seed": REPRODUCIBILITY_SEED,
                "source_snapshot_id": SNAPSHOT_ID, "historical_test_status": HISTORICAL_TEST_STATUS}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Incompatible model manifest: {key}")
    for key in ("bundle_id", "spark_version", "code_provenance", "raw_manifest_sha256", "training"):
        if not manifest.get(key):
            raise ValueError(f"Missing model manifest field: {key}")
    import re
    for value in (manifest["raw_manifest_sha256"], manifest["code_provenance"].get("production_python_sha256", "")):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Invalid provenance fingerprint")
    profile = manifest["training"]
    bounds = SPLIT_BOUNDARIES["train"]
    if (not isinstance(profile.get("rows"), int) or profile["rows"] <= 0
            or not bounds["start"] <= profile.get("min_date", "") <= profile.get("max_date", "") <= bounds["end"]):
        raise ValueError("Invalid training profile")
    if expected_fingerprint is not None and manifest["raw_manifest_sha256"] != expected_fingerprint:
        raise ValueError("Model raw snapshot fingerprint differs from configured manifest")


def validate_models(preprocessing, classifier):
    expected = build_preprocessing_pipeline().getStages()
    if len(preprocessing.stages) != len(expected):
        raise ValueError("Incompatible preprocessing stages")
    for fitted, estimator in zip(preprocessing.stages, expected):
        if type(fitted).__name__ != type(estimator).__name__ + ("Model" if type(estimator).__name__ != "VectorAssembler" else ""):
            raise ValueError("Incompatible preprocessing stage type")
        for param in estimator.params:
            if estimator.isDefined(param) and fitted.hasParam(param.name):
                # Multi-column estimators retain an unused UID-derived outputCol default.
                if param.name == "outputCol" and not estimator.isSet(param):
                    continue
                actual, wanted = fitted.getOrDefault(param.name), estimator.getOrDefault(param)
                if isinstance(wanted, float) and math.isnan(wanted) and isinstance(actual, float) and math.isnan(actual):
                    continue
                if actual != wanted:
                    raise ValueError(f"Incompatible preprocessing setting: {param.name}")
        if hasattr(fitted, "labels") and list(fitted.labels) != sorted(fitted.labels):
            raise ValueError("Indexer labels are not alphabetically ordered")
    encoder = preprocessing.stages[len(CATEGORICAL_COLUMNS)]
    categorical_width = sum(size + int(encoder.getHandleInvalid() == "keep") - int(encoder.getDropLast())
                            for size in encoder.categorySizes)
    expected_width = categorical_width + len(NUMERIC_COLUMNS) + len(HISTORY_COUNTS) + len(HISTORY_MISSING)
    if classifier.numClasses != 2 or classifier.numFeatures != expected_width:
        raise ValueError("Classifier dimensions differ from preprocessing contract")
    for key, value in MODEL_SETTINGS.items():
        if classifier.getOrDefault(key) != value:
            raise ValueError(f"Incompatible Logistic Regression setting: {key}")


def load_bundle(directory, expected_fingerprint=None):
    root = Path(directory)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Model bundle has no valid manifest") from exc
    validate_manifest(manifest, expected_fingerprint)
    preprocessing = PipelineModel.load(str(root / "preprocessing"))
    classifier = LogisticRegressionModel.load(str(root / "logistic_regression"))
    validate_models(preprocessing, classifier)
    return preprocessing, classifier, manifest


def publish_bundle(directory, preprocessing, classifier, manifest):
    """A manifest is a completion marker. Failed replacement restores the old bundle."""
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = destination.with_name(f".{destination.name}.tmp-{token}")
    backup = destination.with_name(f".{destination.name}.backup-{token}")
    moved_old = False
    published = False
    try:
        validate_manifest(manifest)
        temporary.mkdir()
        preprocessing.write().save(str(temporary / "preprocessing"))
        classifier.write().save(str(temporary / "logistic_regression"))
        reloaded_preprocessing = PipelineModel.load(str(temporary / "preprocessing"))
        reloaded_classifier = LogisticRegressionModel.load(str(temporary / "logistic_regression"))
        validate_models(reloaded_preprocessing, reloaded_classifier)
        write_json(temporary / "manifest.json", manifest)
        if destination.exists():
            destination.rename(backup)
            moved_old = True
        try:
            temporary.rename(destination)
            published = True
        except BaseException:
            if moved_old:
                backup.rename(destination)
            raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    if published and moved_old:
        shutil.rmtree(backup)

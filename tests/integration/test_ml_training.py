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
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel, GBTClassifier
from pyspark.ml.feature import ImputerModel, StringIndexerModel
from pyspark.sql import functions as F
from airline.common.spark_session import create_spark_session
from airline.common.raw_snapshot import SNAPSHOT_ID
from airline.dws.ml_feature_contract import *
from airline.ml import model_contract as contract
from airline.ml.train import train
from airline.ml.evaluate import evaluate, score, with_threshold_prediction
from airline.ml.metrics import calculate_metrics
from airline.common.lineage import INTEGER_KEYS, with_lineage
from airline.dws.dataset_manifest import publish_dataset_manifest


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
        row.update({k: i for k in INTEGER_KEYS})
        rows.append(row)
    features = with_lineage(spark.createDataFrame(rows)).select(*OUTPUT_COLUMNS)
    features.filter("split != 'test'").write.partitionBy("split").parquet(str(workspace / "features"))
    snapshot = {"source_snapshot_id": SNAPSHOT_ID, "raw_manifest_sha256": "a" * 64}
    publish_dataset_manifest(workspace / "features", spark.read.parquet(str(workspace / "features")), snapshot)
    manifest = train(spark, workspace / "features", str(workspace / "artifacts/models"), snapshot, Path(__file__).resolve().parents[2])
    features.filter("split = 'test'").write.partitionBy('split').parquet(str(workspace / 'test-features'))
    publish_dataset_manifest(workspace / 'test-features', spark.read.parquet(str(workspace / 'test-features')), snapshot,
                             phase='test', selection_manifest=manifest)
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
        contract.validate_models(preprocessing, WrongDimensions(), manifest["selected_classifier"])
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
    before = score(preprocessing, classifier, unseen, manifest["threshold"]).orderBy("flight_date").collect()
    preprocessing.write().save(str(workspace / "reload-pre"))
    classifier.write().save(str(workspace / "reload-lr"))
    after = score(PipelineModel.load(str(workspace / "reload-pre")), contract.CLASSIFIERS[manifest["selected_classifier"]].load(str(workspace / "reload-lr")), unseen, manifest["threshold"]).orderBy("flight_date").collect()
    assert [r.probability for r in before] == pytest.approx([r.probability for r in after], abs=1e-12)


def test_evaluation_never_fits_and_reports_bundle(spark, workspace, trained, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("Evaluation invoked fit")
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(GBTClassifier, "fit", forbidden)
    for split in ("validation", "test"):
        report = evaluate(spark, str(workspace / ("test-features" if split == "test" else "features")), str(workspace / "artifacts"), split, "a" * 64)
        assert report["rows"] == 5
        assert report["selected_classifier"] == trained[1]["selected_classifier"]
        assert report["threshold_step"] == trained[1]["selection"]["threshold_step"]
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
    for k in INTEGER_KEYS:
        scores = scores.withColumn(k, F.lit(1))
    scores = with_lineage(scores.withColumn("reporting_airline", F.lit("AA")))
    predictions = with_threshold_prediction(scores, 0.34)
    assert [r.prediction for r in predictions.orderBy("probability").collect()] == [0,1,1,1]
    report = calculate_metrics(predictions, 0.34)
    assert report["confusion_matrix"] == dict(true_positive=2, false_positive=1, false_negative=0, true_negative=1)
    assert report["roc_auc"] == pytest.approx(0.75)
    assert report["pr_auc"] == pytest.approx(19 / 24)
    # These two scores would tie if bucketed to hundredths.
    close = spark.createDataFrame([(0,0.341),(1,0.342)], [TARGET_COLUMN,"probability"])
    assert calculate_metrics(close, 0.34)["roc_auc"] == 1
    assert calculate_metrics(close.withColumn(TARGET_COLUMN,F.lit(1)), 0.34)["roc_auc"] is None
    with pytest.raises(ValueError, match="empty"): calculate_metrics(scores.limit(0), 0.34)


def test_invalid_and_absent_manifests_fail(workspace, trained):
    manifest = copy.deepcopy(trained[1])
    for key, value in (("feature_order", []), ("threshold", -1), ("model_settings", {}), ("training", {})):
        changed = {**manifest, key: value}
        with pytest.raises(ValueError): contract.validate_manifest(changed)
    with pytest.raises(ValueError, match="fingerprint"):
        contract.load_bundle(workspace / "artifacts/models", "b" * 64)
    with pytest.raises(ValueError, match="manifest"):
        contract.load_bundle(workspace / "absent")


def test_training_partitioning_and_persistence_released_on_failure(spark, workspace, trained, monkeypatch):
    import importlib
    workflow = importlib.import_module("airline.ml.experiments")
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
        workflow.select(spark, workspace / "features", workspace / "failed-model", "incomplete", {"raw_manifest_sha256": "a" * 64}, workspace)
    assert spark.sparkContext._jsc.sc().getPersistentRDDs().size() == before
    assert not (workspace / "failed-model/experiments/incomplete/selection.json").exists()


def test_dataset_manifest_fail_closed(spark, workspace, trained):
    from airline.dws.dataset_manifest import validate_dataset_manifest, MANIFEST_NAME
    path = workspace / 'features' / MANIFEST_NAME
    original = path.read_text()
    dataset = spark.read.parquet(str(workspace / 'features'))
    try:
        path.unlink()
        with pytest.raises(ValueError, match='Rebuild ML features'):
            validate_dataset_manifest(workspace / 'features', dataset)
        document = json.loads(original)
        document['dataset_schema_version'] = 1
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match='Incompatible'):
            validate_dataset_manifest(workspace / 'features', dataset)
        path.write_text(original)
        with pytest.raises(ValueError, match='fingerprint differs'):
            validate_dataset_manifest(workspace / 'features', dataset, 'b' * 64)
        with pytest.raises(ValueError, match='Rebuild ML features and retrain'):
            contract.validate_manifest({'schema_version': 1})
    finally:
        path.write_text(original)


def test_experiments_train_only_no_test_selection_and_no_fit_evaluation(spark, workspace, trained, monkeypatch):
    from airline.ml import experiments as exp
    from pyspark.ml.classification import GBTClassifier
    # Wide scored caches exhausted full-data executor heaps even with DISK_ONLY.
    from pyspark.sql import DataFrame
    original_persist = DataFrame.persist
    scored_cache_columns = []
    def bounded_cache(self, *args, **kwargs):
        if 'probability' in self.columns:
            assert 'features' not in self.columns and 'rawPrediction' not in self.columns
            scored_cache_columns.append(tuple(self.columns))
        return original_persist(self, *args, **kwargs)
    monkeypatch.setattr(DataFrame, 'persist', bounded_cache)
    original_fit = Pipeline.fit
    def train_only(self, frame, *args, **kwargs):
        assert {r.split for r in frame.select('split').distinct().collect()} == {'train'}
        return original_fit(self, frame, *args, **kwargs)
    monkeypatch.setattr(Pipeline, 'fit', train_only)
    from pyspark.ml.functions import vector_to_array
    fitted_estimators = []
    def train_classifier_only(original):
        def fit(self, frame, *args, **kwargs):
            assert frame.count() == 40
            assert frame.select(F.max(F.array_max(vector_to_array('features')))).first()[0] <= 10
            fitted_estimators.append(type(self).__name__)
            return original(self, frame, *args, **kwargs)
        return fit
    for cls in (LogisticRegression, GBTClassifier):
        monkeypatch.setattr(cls, 'fit', train_classifier_only(cls.fit))
    original_validate = exp.validate_feature_frame
    seen = []
    def record(frame):
        splits = {r.split for r in frame.select('split').distinct().collect()}
        assert 'test' not in splits
        seen.extend(splits)
        return original_validate(frame)
    monkeypatch.setattr(exp, 'validate_feature_frame', record)
    # Change every test label and a predictor after production selection. The second
    # selection must be identical and is forbidden from opening that dataset.
    from pyspark.sql.readwriter import DataFrameReader
    changed_test = trained[0].filter("split = 'test'").withColumn(TARGET_COLUMN, 1-F.col(TARGET_COLUMN)).withColumn('distance', F.lit(-12345.0))
    changed_test.write.mode('overwrite').partitionBy('split').parquet(str(workspace / 'test-features'))
    original_read = DataFrameReader.parquet
    def no_test_read(self, *paths, **kwargs):
        assert not any('test-features' in str(path) for path in paths)
        return original_read(self, *paths, **kwargs)
    monkeypatch.setattr(DataFrameReader, 'parquet', no_test_read)
    snapshot = {'source_snapshot_id': SNAPSHOT_ID, 'raw_manifest_sha256': 'a' * 64}
    manifest = exp.select(spark, workspace / 'features', workspace / 'artifacts', 'fixture', snapshot,
                          Path(__file__).resolve().parents[2])
    monkeypatch.setattr(DataFrameReader, 'parquet', original_read)
    # Restore the completion marker invalidated by the deliberate adversarial overwrite.
    publish_dataset_manifest(workspace / 'test-features', spark.read.parquet(str(workspace / 'test-features')),
                             snapshot, phase='test', selection_manifest=trained[1])
    assert manifest['selected_classifier'] == trained[1]['selected_classifier']
    assert manifest['threshold_step'] == trained[1]['threshold_step']
    for name in ('lr', 'gbt'):
        assert manifest['validation_results'][name]['thresholds'] == trained[1]['selection']['validation_results'][name]['thresholds']
    assert seen == ['train', 'validation']
    assert fitted_estimators == ['LogisticRegression', 'GBTClassifier']
    assert type(manifest['threshold_step']) is int
    with pytest.raises(FileExistsError):
        exp.select(spark, workspace / 'features', workspace / 'artifacts', 'fixture', snapshot, workspace)
    monkeypatch.setattr(exp, 'validate_feature_frame', original_validate)
    def forbidden(*args, **kwargs): raise AssertionError('Evaluation invoked fit')
    for cls in (Pipeline, LogisticRegression, GBTClassifier):
        monkeypatch.setattr(cls, 'fit', forbidden)
    # Test features are bound to the frozen selection, not to mutable selection data.
    document_path = workspace / 'test-features/_dataset_manifest.json'
    document = json.loads(document_path.read_text())
    document['selection_run_id'] = 'fixture'
    document_path.write_text(json.dumps(document))
    result = exp.evaluate(spark, workspace / 'test-features', workspace / 'artifacts', 'fixture', 'test', 'a' * 64)
    assert result['selected']['rows'] == result['baseline']['rows'] == 5
    assert result['baseline']['predicted_positive_rate'] == 1.0  # Training prevalence tie predicts positive.
    assert result == exp.evaluate(spark, workspace / 'test-features', workspace / 'artifacts', 'fixture', 'test', 'a' * 64)
    assert len(scored_cache_columns) == 2  # Reused LR/GBT validation scores; one-write test predictions are not cached.


def test_threshold_counts_use_direct_comparisons(spark):
    from airline.ml.experiments import threshold_counts
    scores = spark.createDataFrame([(1, 0.34), (0, 0.339999999999), (1, 0.340000000001), (0, 1.0), (1, 0.0)],
                                  [TARGET_COLUMN, 'probability'])
    counts = threshold_counts(scores)
    assert counts[34] == dict(step=34, tp=2, fp=1, fn=1, tn=1)
    assert counts[0] == dict(step=0, tp=3, fp=2, fn=0, tn=0)
    assert counts[100] == dict(step=100, tp=0, fp=1, fn=3, tn=1)


def test_added_identifiers_never_change_predictor_vectors(workspace, trained):
    from pyspark.ml.functions import vector_to_array
    features, _ = trained
    pre, _, _ = contract.load_bundle(workspace / 'artifacts/models')
    original = pre.transform(features).select('flight_date', vector_to_array('features').alias('vector'))
    changed = features
    for key in INTEGER_KEYS:
        changed = changed.withColumn(key, F.col(key) + 10000)
    changed = with_lineage(changed)
    after = pre.transform(changed).select('flight_date', vector_to_array('features').alias('vector'))
    assert original.exceptAll(after).count() == after.exceptAll(original).count() == 0


def test_overwrite_invalidates_completion_before_writing(spark, workspace, trained, monkeypatch):
    from airline.dws.build_ml_flight_features import write_ml_flight_features
    from airline.dws.dataset_manifest import MANIFEST_NAME
    from pyspark.sql.readwriter import DataFrameWriter
    root = workspace / 'interrupted'
    root.mkdir()
    marker = root / MANIFEST_NAME
    marker.write_text('{}')
    def fail(*args, **kwargs):
        assert not marker.exists()
        raise RuntimeError('interrupted write')
    monkeypatch.setattr(DataFrameWriter, 'parquet', fail)
    with pytest.raises(RuntimeError, match='interrupted'):
        write_ml_flight_features(trained[0], root, 50)
    assert not marker.exists()


def test_production_selection_rejects_test_dataset_before_fitting(spark, workspace, trained, monkeypatch):
    from airline.ml import experiments
    def forbidden(*args, **kwargs):
        raise AssertionError('Selection attempted fitting a test dataset')
    monkeypatch.setattr(Pipeline, 'fit', forbidden)
    with pytest.raises(ValueError, match='phase'):
        experiments.select(spark, workspace / 'test-features', workspace / 'rejected', 'test-data',
                           {'raw_manifest_sha256': 'a'*64}, workspace)


def test_production_manifest_reconciles_selection_and_rejects_old_schema(trained):
    manifest = trained[1]
    assert manifest['threshold'] == manifest['selection']['threshold_step'] / 100
    assert manifest['selected_classifier'] == manifest['selection']['selected_classifier']
    for version in (1, 2):
        with pytest.raises(ValueError, match='Rebuild ML features and retrain'):
            contract.validate_manifest({'schema_version': version})
    altered = copy.deepcopy(manifest)
    altered['selection']['threshold_step'] = (altered['threshold_step'] + 1) % 101
    with pytest.raises(ValueError, match='validation evidence'):
        contract.validate_manifest(altered)


def test_selected_gbt_publication_and_reload(spark, workspace, trained):
    # Exercise the GBT publication branch even when this fixture's real winner is LR.
    selection = copy.deepcopy(trained[1]['selection'])
    from airline.ml.experiments import run_directory, best_threshold
    original = run_directory(workspace / 'artifacts', selection['run_id'])
    pre = PipelineModel.load(str(original / 'preprocessing'))
    gbt = contract.CLASSIFIERS['gbt'].load(str(original / 'gbt'))
    for name in ('lr', 'gbt'):
        rows = [dict(step=s, tp=int(name == 'gbt' and s <= 70), fp=0,
                     fn=int(name != 'gbt' or s > 70), tn=4) for s in range(101)]
        selection['validation_results'][name]['thresholds'] = rows
        selection['validation_results'][name]['selected'] = best_threshold(rows)
    selection['selected_classifier'] = 'gbt'
    selection['threshold_step'] = 0
    manifest = contract.make_manifest(spark, selection['training'],
        {'source_snapshot_id': SNAPSHOT_ID, 'raw_manifest_sha256': 'a'*64}, Path(__file__).resolve().parents[2], selection)
    contract.publish_bundle(workspace / 'gbt-bundle', pre, gbt, manifest)
    loaded_pre, loaded_gbt, loaded_manifest = contract.load_bundle(workspace / 'gbt-bundle')
    assert loaded_manifest['selected_classifier'] == 'gbt'
    before = score(pre, gbt, trained[0], 0)
    after = score(loaded_pre, loaded_gbt, trained[0], loaded_manifest['threshold'])
    assert before.exceptAll(after).count() == after.exceptAll(before).count() == 0


@pytest.fixture
def recovery_case(spark, workspace, trained, monkeypatch, request):
    """Reuse frozen fixture models; every repair is forbidden from fitting."""
    from airline.ml import evaluate as production, experiments
    root = Path(tempfile.mkdtemp(prefix='recovery-', dir=workspace))
    shutil.copytree(workspace / 'artifacts', root / 'artifacts')
    shutil.copytree(workspace / 'test-features', root / 'features')
    run_id = trained[1]['selection']['run_id']
    marker = root / 'features/_dataset_manifest.json'
    document = json.loads(marker.read_text())
    document['selection_run_id'] = run_id
    marker.write_text(json.dumps(document))
    def forbidden(*args, **kwargs):
        raise AssertionError('Recovery attempted fitting')
    for cls in (Pipeline, LogisticRegression, GBTClassifier):
        monkeypatch.setattr(cls, 'fit', forbidden)
    calls = []
    if request.param == 'production':
        original = production.score
        def counted(*args, **kwargs):
            calls.append('score')
            return original(*args, **kwargs)
        monkeypatch.setattr(production, 'score', counted)
        def run():
            return production.evaluate(spark, root / 'features', root / 'artifacts', 'test', 'a'*64)
        report = root / 'artifacts/metrics/test.json'
        predictions = root / 'artifacts/predictions/test'
        model = root / 'artifacts/models/classifier'
        model_manifest = root / 'artifacts/models/manifest.json'
    else:
        original = experiments.continuous_score
        def counted(*args, **kwargs):
            calls.append('score')
            return original(*args, **kwargs)
        monkeypatch.setattr(experiments, 'continuous_score', counted)
        def run():
            return experiments.evaluate(spark, root / 'features', root / 'artifacts', run_id, 'test', 'a'*64)
        directory = experiments.run_directory(root / 'artifacts', run_id)
        report, predictions = directory / 'test.json', directory / 'test_predictions'
        model = directory / trained[1]['selected_classifier']
        model_manifest = directory / 'selection.json'
    report.unlink(missing_ok=True)  # Start cold regardless of test execution order.
    first = run()
    assert len(calls) == 1
    yield dict(root=root, run=run, report=report, predictions=predictions, model=model,
               model_manifest=model_manifest, first=first, calls=calls)
    shutil.rmtree(root)


@pytest.mark.parametrize('recovery_case', ['production', 'experiment'], indirect=True)
def test_evaluation_complete_cache_reuses_without_scoring(recovery_case):
    case = recovery_case
    assert case['run']() == case['first']
    assert len(case['calls']) == 1


@pytest.mark.parametrize('recovery_case', ['production', 'experiment'], indirect=True)
@pytest.mark.parametrize('damage', ['missing', 'part', 'truncated', 'success', 'temporary',
                                    'report', 'legacy', 'completion', 'schema', 'population', 'threshold'])
def test_evaluation_recovers_incomplete_outputs(spark, recovery_case, damage):
    from airline.ml.evaluation_artifacts import inventory
    case = recovery_case
    predictions = case['predictions']
    if damage == 'missing':
        shutil.rmtree(predictions)
    elif damage == 'part':
        next(predictions.glob('*.parquet')).unlink()
    elif damage == 'truncated':
        next(predictions.glob('*.parquet')).write_bytes(b'PAR1broken')
    elif damage == 'success':
        (predictions / '_SUCCESS').unlink()
    elif damage == 'temporary':
        (predictions / '_temporary').mkdir()
    elif damage == 'report':
        case['report'].write_text('{broken')
    elif damage == 'legacy':
        document = json.loads(case['report'].read_text())
        document.pop('completion')
        case['report'].write_text(json.dumps(document))
    elif damage == 'completion':
        document = json.loads(case['report'].read_text())
        document['completion'].pop('rows')
        case['report'].write_text(json.dumps(document))
    else:
        # Keep the file inventory consistent to exercise semantic validation too.
        saved = spark.read.parquet(str(predictions))
        if damage == 'schema':
            changed = saved.drop('probability')
        elif damage == 'population':
            changed = saved.withColumn('flight_id', F.lit('different-key'))
        else:
            changed = saved.withColumn('prediction', 1-F.col('prediction'))
        detached = spark.createDataFrame(changed.collect(), changed.schema)
        detached.write.mode('overwrite').parquet(str(predictions))
        document = json.loads(case['report'].read_text())
        document['completion']['prediction_files'] = inventory(predictions, parquet_only=True)
        case['report'].write_text(json.dumps(document))
    recovered = case['run']()
    assert len(case['calls']) == 2
    assert {k:v for k,v in recovered.items() if k != 'completion'} == {k:v for k,v in case['first'].items() if k != 'completion'}
    assert case['run']() == recovered
    assert len(case['calls']) == 2


@pytest.mark.parametrize('recovery_case', ['production', 'experiment'], indirect=True)
@pytest.mark.parametrize('changed', ['model_bytes', 'model_manifest', 'dataset_bytes', 'dataset_manifest'])
def test_evaluation_rejects_changed_identity(spark, recovery_case, changed):
    case = recovery_case
    if changed == 'model_bytes':
        marker = case['model'] / 'metadata/part-00000'
        document = json.loads(marker.read_text())
        document['recovery_probe'] = 'same settings; changed saved-model identity'
        marker.write_text(json.dumps(document)+'\n')
        # Hadoop checksum sidecars must match deliberate fixture edits.
        for checksum in marker.parent.glob('.*.crc'):
            checksum.unlink()
    elif changed == 'model_manifest':
        document = json.loads(case['model_manifest'].read_text())
        document['recovery_probe'] = True
        case['model_manifest'].write_text(json.dumps(document))
    elif changed == 'dataset_bytes':
        path = case['root'] / 'features'
        frame = spark.read.parquet(str(path)).withColumn('distance', F.lit(12345.0))
        detached = spark.createDataFrame(frame.collect(), frame.schema)
        # Keep the existing manifest to prove the data bytes themselves are bound.
        manifest = (path / '_dataset_manifest.json').read_text()
        detached.write.mode('overwrite').partitionBy('split').parquet(str(path))
        (path / '_dataset_manifest.json').write_text(manifest)
    else:
        marker = case['root'] / 'features/_dataset_manifest.json'
        document = json.loads(marker.read_text())
        document['recovery_probe'] = True
        marker.write_text(json.dumps(document))
    recovered = case['run']()
    assert len(case['calls']) == 2
    assert recovered['completion']['identity'] != case['first']['completion']['identity']
    assert case['run']() == recovered
    assert len(case['calls']) == 2


@pytest.mark.parametrize('recovery_case', ['production', 'experiment'], indirect=True)
@pytest.mark.parametrize('failure', ['parquet', 'report'])
def test_evaluation_interruption_revokes_completion(recovery_case, monkeypatch, failure):
    from pyspark.sql.readwriter import DataFrameWriter
    case = recovery_case
    (case['predictions'] / '_SUCCESS').unlink()  # Force repair.
    if failure == 'parquet':
        original = DataFrameWriter.parquet
        def interrupt(self, *args, **kwargs):
            assert not case['report'].exists()
            raise RuntimeError('injected interrupted predictions')
        monkeypatch.setattr(DataFrameWriter, 'parquet', interrupt)
    else:
        original = Path.replace
        def interrupt(self, target):
            if target == case['report']:
                assert not case['report'].exists()
                raise RuntimeError('injected interrupted report')
            return original(self, target)
        monkeypatch.setattr(Path, 'replace', interrupt)
    with pytest.raises(RuntimeError, match='injected interrupted'):
        case['run']()
    assert not case['report'].exists()
    if failure == 'parquet':
        monkeypatch.setattr(DataFrameWriter, 'parquet', original)
    else:
        monkeypatch.setattr(Path, 'replace', original)
    recovered = case['run']()
    assert {k:v for k,v in recovered.items() if k != 'completion'} == {k:v for k,v in case['first'].items() if k != 'completion'}
    assert len(case['calls']) == 3

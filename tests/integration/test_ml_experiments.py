# AI-generated test code: generated with OpenAI Codex.
from datetime import date, timedelta
import json
import os
from pathlib import Path
import tempfile

import pytest
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import ImputerModel, StringIndexerModel, VectorAssembler
from pyspark.sql import functions as F

from airline.common.spark_session import create_spark_session
from airline.dws.build_ml_flight_features import (
    HISTORICAL_FEATURE_COLUMNS, OUTPUT_COLUMNS, TARGET_COLUMN, transform_ml_flight_features,
)
from airline.ml.experiment_features import (
    ENTITIES, HISTORY_RATES, add_schedule_features, build_features, entity_history, preprocessing_pipeline,
)
from airline.ml.experiment_policy import BACKTESTS, Candidate
from airline.ml.run_experiments import ExperimentRunner, estimator, metric_report, validate_output_root
from airline.ml.train_models import build_logistic_regression, build_preprocessing_pipeline, score_probabilities


@pytest.fixture(scope="module")
def spark():
    session = create_spark_session("airline-staged-experiment-tests")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def shared_path():
    root = Path(os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir()))
    root.mkdir(parents=True, exist_ok=True)
    # Keep failed-run artifacts inspectable; use a unique, narrow fixture directory.
    return Path(tempfile.mkdtemp(prefix="airline-experiments-", dir=root))


@pytest.fixture(scope="module")
def dwd(spark):
    rows = []
    for i in range(48):
        day = date(2023, 1, 1) + timedelta(days=i * 3)
        rows.append((day, "AA" if i % 2 else "DL", "ORD", "LAX", 1000.0 + i,
                     day.year, day.month, day.isoweekday(), i % 24, i % 60,
                     (i + 4) % 24, i % 60, 240.0, 0, 0, int(i % 3 == 0)))
    return spark.createDataFrame(rows, "flight_date date, reporting_airline string, origin string, dest string, "
        "distance double, year int, month int, day_of_week int, scheduled_departure_hour int, "
        "scheduled_departure_minute int, scheduled_arrival_hour int, scheduled_arrival_minute int, "
        "crs_elapsed_time double, is_cancelled int, is_diverted int, is_arrival_delayed_15m int")


def test_calendar_window_boundaries_duplicates_gaps_and_future_invariance(spark):
    target = date(2024, 3, 10)
    offsets = [-91, -90, -31, -30, -8, -7, -1, -1, 0, 1]
    rows = [(target + timedelta(days=offset), "ORD", int(i % 2 == 0)) for i, offset in enumerate(offsets)]
    rows += [(target, "NEW", 1)]
    frame = spark.createDataFrame(rows, "flight_date date, origin string, is_arrival_delayed_15m int")
    history = entity_history(frame, "origin", ("origin",))
    actual = history.filter((F.col("flight_date") == target) & (F.col("origin") == "ORD")).first()
    for days in (7, 30, 90):
        expected = [r for r in rows if target - timedelta(days=days) <= r[0] < target and r[1] == "ORD"]
        assert actual[f"origin_flight_count_{days}d"] == len(expected)
        assert actual[f"origin_delay_rate_{days}d"] == pytest.approx(sum(r[2] for r in expected) / len(expected))
    changed = frame.withColumn(TARGET_COLUMN, F.when(F.col("flight_date") >= target, 1 - F.col(TARGET_COLUMN))
                               .otherwise(F.col(TARGET_COLUMN)))
    after = entity_history(changed, "origin", ("origin",)).filter(
        (F.col("flight_date") == target) & (F.col("origin") == "ORD")).first()
    assert actual.asDict() == after.asDict()
    sparse = history.filter(F.col("origin") == "NEW").first()
    for days in (7, 30, 90):
        assert sparse[f"origin_flight_count_{days}d"] == 0
        assert sparse[f"origin_delay_rate_{days}d"] is None
        assert sparse[f"origin_history_missing_{days}d"] == 1


@pytest.mark.parametrize("entity_name,entity_columns", ENTITIES)
def test_future_mutation_invariance_for_every_history_entity_and_window(
    spark, entity_name, entity_columns
):
    target = date(2024, 3, 10)
    rows = []
    for offset in (-100, -90, -30, -7, -1, 0, 1, 30):
        rows.append((target + timedelta(days=offset), "AA", "ORD", "LAX", int(offset % 2 == 0)))
    frame = spark.createDataFrame(
        rows,
        "flight_date date, reporting_airline string, origin string, destination string, is_arrival_delayed_15m int",
    )
    before = entity_history(frame, entity_name, entity_columns).filter(F.col("flight_date") == target).first()
    mutated = frame.withColumn(
        TARGET_COLUMN,
        F.when(F.col("flight_date") >= F.lit(target), 1 - F.col(TARGET_COLUMN)).otherwise(F.col(TARGET_COLUMN)),
    )
    after = entity_history(mutated, entity_name, entity_columns).filter(F.col("flight_date") == target).first()
    assert before.asDict() == after.asDict()
    for days in (7, 30, 90):
        assert before[f"{entity_name}_flight_count_{days}d"] > 0


def test_schedule_midnight_missing_and_no_actual_time_dependency(dwd):
    row = dwd.limit(1).withColumn("scheduled_departure_hour", F.lit(0)).withColumn("scheduled_departure_minute", F.lit(0))
    row = row.withColumn("scheduled_arrival_minute", F.lit(None).cast("int"))
    row = row.withColumn("dep_delay", F.lit(999)).withColumn("arr_delay", F.lit(999))
    value = add_schedule_features(row).first()
    assert value["departure_sin"] == pytest.approx(0)
    assert value["departure_cos"] == pytest.approx(1)
    assert value["arrival_missing"] == 1
    assert value["arrival_sin"] is None


def test_bundle_eligibility_grain_and_baseline_equivalence(dwd):
    extra = dwd.limit(1).withColumn("is_cancelled", F.lit(1))
    source = dwd.unionByName(extra).unionByName(dwd.limit(1).withColumn("is_diverted", F.lit(1)))
    expected = transform_ml_flight_features(source).select(*OUTPUT_COLUMNS)
    baseline = build_features(source, "baseline").select(*OUTPUT_COLUMNS)
    assert baseline.exceptAll(expected).count() == 0
    assert expected.exceptAll(baseline).count() == 0
    history = build_features(source, "history").cache()
    try:
        assert history.count() == dwd.count()
        assert len(HISTORY_RATES) == 12
        expected_rates = baseline.select("flight_date", "reporting_airline", *HISTORICAL_FEATURE_COLUMNS)
        assert history.select(*expected_rates.columns).exceptAll(expected_rates).count() == 0
        assert not {"dep_delay", "arr_delay", "is_cancelled", "is_diverted"} & set(history.columns)
    finally:
        history.unpersist()


def test_baseline_estimator_equivalence_and_reload(dwd, shared_path):
    features = build_features(dwd, "baseline").cache()
    try:
        original = build_preprocessing_pipeline().fit(features)
        replacement = preprocessing_pipeline("baseline").fit(features)
        train_a, train_b = original.transform(features), replacement.transform(features)
        old_model = build_logistic_regression().fit(train_a)
        new_model = estimator(Candidate("baseline", "logistic_regression", 20)).fit(train_b)
        old = score_probabilities(old_model, train_a).orderBy("flight_date").collect()
        new = score_probabilities(new_model, train_b).orderBy("flight_date").collect()
        assert [r.probability for r in old] == pytest.approx([r.probability for r in new], abs=1e-10)
        replacement.write().save(str(shared_path / "preprocessor"))
        new_model.write().save(str(shared_path / "classifier"))
        reloaded = score_probabilities(LogisticRegressionModel.load(str(shared_path / "classifier")),
            PipelineModel.load(str(shared_path / "preprocessor")).transform(features)).orderBy("flight_date").collect()
        assert [r.probability for r in new] == pytest.approx([r.probability for r in reloaded], abs=1e-12)
    finally:
        features.unpersist()


def test_schedule_preprocessor_fits_only_training_and_handles_unseen(dwd):
    features = build_features(dwd, "schedule").cache()
    try:
        pipeline = preprocessing_pipeline("schedule").fit(features)
        unseen = features.limit(1).withColumn("reporting_airline", F.lit("UNSEEN"))
        unseen = unseen.withColumn("crs_elapsed_time", F.lit(None).cast("double"))
        assert pipeline.transform(unseen).select("features").first()[0].size > 0
        for stage in pipeline.stages:
            if isinstance(stage, StringIndexerModel):
                assert "UNSEEN" not in stage.labels
            if isinstance(stage, VectorAssembler):
                assert not {TARGET_COLUMN, "flight_date", "split"} & set(stage.getInputCols())
            if isinstance(stage, ImputerModel):
                assert stage.surrogateDF.first()["crs_elapsed_time"] == pytest.approx(240)
    finally:
        features.unpersist()


def test_metric_layers_use_one_frozen_threshold(spark):
    scores = spark.createDataFrame([(0, 0.1), (1, 0.2), (0, 0.3), (1, 0.9)], [TARGET_COLUMN, "probability"])
    report = metric_report(scores, 0.3)
    assert report["operating_point"]["predicted_positive_rate"] == 0.5
    assert report["operating_point"]["precision"] == 0.5
    assert report["operating_point"]["recall"] == 0.5
    assert report["class_prevalence"] == 0.5
    assert set(report["ranking"]) == {"pr_auc", "roc_auc"}
    with pytest.raises(ValueError):
        metric_report(scores, 0.305)


def test_guards_block_test_and_original_artifact_paths(tmp_path):
    runner = object.__new__(ExperimentRunner)
    runner.output = tmp_path
    with pytest.raises(ValueError, match="test period"):
        runner.read_source(date(2024, 9, 1))
    with pytest.raises(AssertionError, match="Freeze"):
        runner.features("baseline", confirmation=True)
    with pytest.raises(ValueError, match="protected"):
        validate_output_root(tmp_path / "artifacts", tmp_path)
    assert validate_output_root(tmp_path / "artifacts/experiments/run", tmp_path).name == "run"


def test_failed_schedule_gate_skips_history_and_reuses_baseline_for_tuning(tmp_path, monkeypatch):
    runner = object.__new__(ExperimentRunner)
    runner.output = tmp_path
    seen = []
    def ranking(candidate):
        seen.append(candidate)
        return [0.4, 0.4] if candidate.bundle == "baseline" else [0.3, 0.4]
    monkeypatch.setattr(runner, "ranking", ranking)
    monkeypatch.setattr(runner, "write_report", lambda _: None)
    result = runner.run("tuning")
    assert not result["gates"]["schedule"]["accepted"]
    assert "skipped" in result["gates"]["history"]
    assert not any(c.bundle == "history" for c in seen)
    assert all(c["bundle"] == "baseline" for c in result["tuned_candidates"])
    assert not (tmp_path / "selection.json").exists()


def test_runner_checkpoints_fits_and_confirms_frozen_model_once(spark, dwd, shared_path, monkeypatch):
    from airline.ml.train_models import write_json
    import airline.ml.run_experiments as workflow

    # Include held-out test rows in storage: the runner must filter them out.
    source = dwd
    for offset in (240, 365, 487):
        shifted = dwd.withColumn("flight_date", F.date_add("flight_date", offset))
        shifted = (shifted.withColumn("year", F.year("flight_date"))
                   .withColumn("month", F.month("flight_date"))
                   .withColumn("day_of_week", F.dayofweek("flight_date")))
        source = source.unionByName(shifted)
    source_path = shared_path / "dwd"
    source.write.partitionBy("year", "month").parquet(str(source_path))
    output = shared_path / "run"
    runner = ExperimentRunner(spark, source_path, output)
    candidate = Candidate("baseline", "logistic_regression", 20)
    results = runner.candidate(candidate)
    assert len(results) == 2
    assert all(r["pooled"]["rows"] > 0 for r in results)
    assert all(r["training_partitions"] == 2 for r in results)
    assert all((runner.score_path(candidate, fold).parent / "runtime.json").is_file() for fold in BACKTESTS)
    selection = {"phase": "selection_frozen", "candidate": candidate.document(), "threshold": 0.22}
    write_json(selection, output / "selection.json")
    runner.confirm()
    confirmation = json.loads((output / "validation.json").read_text())
    assert confirmation["phase"] == "official_validation_confirmation"
    assert set(confirmation["monthly"]) <= {"2024-05", "2024-06", "2024-07", "2024-08"}
    assert json.loads((output / "final_model/complete.json").read_text())["train"]["max_date"] <= "2024-04-30"
    before = (output / "validation.json").read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("A completed run must not fit or access source data again")

    monkeypatch.setattr(workflow, "estimator", forbidden)
    monkeypatch.setattr(runner, "read_source", forbidden)
    assert runner.candidate(candidate) == results
    runner.confirm()
    assert (output / "validation.json").read_bytes() == before
    assert runner.run()["completed_stage"] == "validation"
    # Re-instantiation accepts matching code/data and rejects modified input metadata.
    ExperimentRunner(spark, source_path, output)
    monkeypatch.setattr(workflow, "source_signature", lambda _: "changed")
    with pytest.raises(ValueError, match="changed"):
        ExperimentRunner(spark, source_path, output)


def test_missing_metrics_recover_from_saved_scores_without_refitting(spark, dwd, shared_path, monkeypatch):
    import airline.ml.run_experiments as workflow

    source_path = shared_path / "recovery-dwd"
    dwd.write.partitionBy("year", "month").parquet(str(source_path))
    output = shared_path / "recovery-run"
    runner = ExperimentRunner(spark, source_path, output)
    candidate = Candidate("baseline", "logistic_regression", 20)
    for fold in BACKTESTS:
        spark.createDataFrame(
            [(fold.evaluation_start, "development", 0, 0.1),
             (fold.evaluation_start, "development", 1, 0.9)],
            "flight_date date, split string, is_arrival_delayed_15m int, probability double",
        ).write.parquet(str(runner.score_path(candidate, fold)))
    class MetadataOnlyEstimator:
        def extractParamMap(self):
            return {}

    monkeypatch.setattr(workflow, "estimator", lambda _: MetadataOnlyEstimator())
    recovered = runner.candidate(candidate)
    assert all(result["recovered_from_saved_scores"] is True for result in recovered)
    assert all((runner.score_path(candidate, fold).parent / "metrics.json").is_file() for fold in BACKTESTS)

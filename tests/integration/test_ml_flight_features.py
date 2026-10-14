# AI-generated test code: generated with OpenAI Codex.
from datetime import date, timedelta
import os
from pathlib import Path
import tempfile
import pytest
from pyspark.sql import functions as F
from airline.common.spark_session import create_spark_session
from airline.dws.ml_feature_contract import *
from airline.dws.build_ml_flight_features import (
    add_schedule_features, entity_history, transform_ml_flight_features,
    validate_ml_source, validate_feature_frame, build_ml_flight_features,
)

@pytest.fixture(scope="module")
def spark():
    session = create_spark_session("airline-feature-tests")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def shared_path():
    root = Path(os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir()))
    root.mkdir(parents=True, exist_ok=True)
    # Keep failed-run artifacts inspectable; use a unique, narrow fixture directory.
    return Path(tempfile.mkdtemp(prefix="airline-features-", dir=root))


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
    row = dwd.orderBy("flight_date").limit(1).withColumn("scheduled_departure_hour", F.lit(0)).withColumn("scheduled_departure_minute", F.lit(0))
    row = row.withColumn("scheduled_arrival_minute", F.lit(None).cast("int"))
    row = row.withColumn("dep_delay", F.lit(999)).withColumn("arr_delay", F.lit(999))
    value = add_schedule_features(row).first()
    assert value["departure_sin"] == pytest.approx(0)
    assert value["departure_cos"] == pytest.approx(1)
    assert value["arrival_missing"] == 1
    assert value["arrival_sin"] is None



def test_eligibility_multiplicity_and_split_boundaries(spark, dwd, shared_path):
    source = dwd.unionByName(dwd.orderBy("flight_date").limit(1))
    source = source.unionByName(dwd.orderBy("flight_date").limit(1).withColumn("is_cancelled", F.lit(1)))
    source = source.unionByName(dwd.orderBy("flight_date").limit(1).withColumn("is_diverted", F.lit(1)))
    source = source.unionByName(dwd.orderBy("flight_date").limit(1).withColumn(TARGET_COLUMN, F.lit(None).cast("int")))
    dates = [TRAIN_START, TRAIN_END, VALIDATION_START, VALIDATION_END, TEST_START, TEST_END]
    expected = ["train", "train", "validation", "validation", "test", "test"]
    endpoints = dwd.orderBy("flight_date").limit(1).withColumn("flight_date", F.lit(dates[0]))
    for day in dates[1:]:
        endpoints = endpoints.unionByName(dwd.orderBy("flight_date").limit(1).withColumn("flight_date", F.lit(day)))
    source = source.unionByName(endpoints)
    source.write.parquet(str(shared_path / "dwd"))
    build_ml_flight_features(spark, shared_path / "dwd", shared_path / "features")
    features = spark.read.parquet(str(shared_path / "features")).select(*OUTPUT_COLUMNS).cache()
    try:
        assert validate_feature_frame(features, 55) == 55
        assert set(p.name for p in (shared_path / "features").glob("split=*")) == {"split=train", "split=validation", "split=test"}
        for day, split in zip(dates, expected):
            assert features.filter(F.col("flight_date") == day).first().split == split
        assert set(FEATURE_COLUMNS).isdisjoint(LEAKAGE_BLACKLIST)
        duplicate = features.filter(F.col("flight_date") == TRAIN_START)
        assert duplicate.count() == 3
    finally:
        features.unpersist()


@pytest.mark.parametrize("boundary", [VALIDATION_START, TEST_START])
@pytest.mark.parametrize("entity,keys", ENTITIES)
def test_history_crosses_both_split_boundaries(spark, boundary, entity, keys):
    source = spark.createDataFrame([(boundary-timedelta(days=1), "AA", "ORD", "LAX", 1),
                                    (boundary, "AA", "ORD", "LAX", 0)],
                                   "flight_date date, reporting_airline string, origin string, destination string, is_arrival_delayed_15m int")
    row = entity_history(source, entity, keys).filter(F.col("flight_date") == boundary).first()
    for days in WINDOW_DAYS:
        assert row[f"{entity}_flight_count_{days}d"] == 1
        assert row[f"{entity}_delay_rate_{days}d"] == 1.0
        assert row[f"{entity}_history_missing_{days}d"] == 0


@pytest.mark.parametrize("column,value", [("is_cancelled", 2), (TARGET_COLUMN, 2), ("flight_date", date(2025,1,1))])
def test_invalid_source_fails(dwd, column, value):
    with pytest.raises(AssertionError):
        validate_ml_source(dwd.withColumn(column, F.lit(value)))


def test_null_entities_preserve_multiplicity_and_history(dwd):
    source = dwd.orderBy("flight_date").limit(3).withColumn("origin", F.lit(None).cast("string"))
    featured = transform_ml_flight_features(source).orderBy("flight_date").collect()
    assert len(featured) == 3
    assert featured[0].origin_flight_count_7d == 0
    assert featured[1].origin_flight_count_7d == 1
    assert featured[2].origin_flight_count_7d == 2


def test_history_domain_and_schema_fail_closed(dwd):
    features = transform_ml_flight_features(dwd)
    with pytest.raises(AssertionError, match="schema"):
        validate_feature_frame(features.drop("distance"))
    with pytest.raises(AssertionError, match="schema"):
        validate_feature_frame(features.withColumn("arr_delay", F.lit(20.0)))
    with pytest.raises(AssertionError, match="invalid"):
        validate_feature_frame(features.withColumn("carrier_flight_count_7d", F.lit(-1)))
    with pytest.raises(AssertionError, match="invalid"):
        validate_feature_frame(features.withColumn("split", F.lit("test")))
    with pytest.raises(AssertionError, match="invalid"):
        validate_feature_frame(features.withColumn("carrier_flight_count_7d", F.lit(float("nan")))
                               .withColumn("carrier_delay_rate_7d", F.lit(0.3))
                               .withColumn("carrier_history_missing_7d", F.lit(0.0)))

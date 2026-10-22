# AI-generated test code: generated with OpenAI Codex.
"""Tests ml flight features behavior."""

from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
import tempfile
import pytest
from pyspark.sql import functions as F
from airline.common.spark_session import create_spark_session
from airline.dws.ml_feature_contract import *
from airline.common.raw_snapshot import SNAPSHOT_ID
from airline.dws.build_ml_flight_features import (
    add_schedule_features, entity_history, assign_chronological_split, transform_ml_flight_features,
    validate_ml_source, validate_feature_frame, build_ml_flight_features,
)

@pytest.fixture(scope="module")
def spark():
    """Create the test Spark session and stop it after use."""
    session = create_spark_session("airline-feature-tests")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def shared_path():
    """Return the shared test data path."""
    root = Path(os.environ.get("AIRLINE_TEST_TMP_ROOT", tempfile.gettempdir()))
    root.mkdir(parents=True, exist_ok=True)
    # Keep failed-run artifacts inspectable; use a unique, narrow fixture directory.
    return Path(tempfile.mkdtemp(prefix="airline-features-", dir=root))


@pytest.fixture(scope="module")
def dwd(spark):
    """Build a small trusted-flight DataFrame."""
    rows = []
    for i in range(48):
        day = date(2023, 1, 1) + timedelta(days=i * 3)
        rows.append((day, "AA" if i % 2 else "DL", "ORD", "LAX", 1000.0 + i,
                     day.year, day.month, day.isoweekday(), i % 24, i % 60,
                     (i + 4) % 24, i % 60, 240.0, 0, 0, int(i % 3 == 0)))
    return spark.createDataFrame(rows, "flight_date date, reporting_airline string, origin string, dest string, "
        "distance double, year int, month int, day_of_week int, scheduled_departure_hour int, "
        "scheduled_departure_minute int, scheduled_arrival_hour int, scheduled_arrival_minute int, "
        "crs_elapsed_time double, is_cancelled int, is_diverted int, is_arrival_delayed_15m int").withColumn("flight_number_reporting_airline", F.lit(12)).withColumn("origin_airport_id", F.lit(13930)).withColumn("dest_airport_id", F.lit(12892)).withColumn("crs_dep_time", F.lit(900)).withColumn("dep_delay", F.lit(0.0)).withColumn("actual_elapsed_time", F.lit(240.0))


@pytest.mark.parametrize("entity,keys", ENTITIES)
def test_timestamp_boundaries_overnight_same_day_future_duplicates(spark, entity, keys):
    """Exercise the timestamp boundaries overnight same day future duplicates case."""
    from airline.dws.flight_time import DAY_US
    target = 200 * DAY_US
    hour = DAY_US // 24
    # Departure offsets, outcome offsets, labels. Includes simultaneous, in-flight,
    # same-day completed, overnight, exact arrival, and all lookback boundaries.
    offsets = [(-91*DAY_US, -90*DAY_US, 1), (-90*DAY_US, -89*DAY_US, 1),
               (-30*DAY_US, -29*DAY_US, 0), (-7*DAY_US, -6*DAY_US, 1),
               (-7*DAY_US-1, -6*DAY_US, 0), (-3*hour, -hour, 1),
               (-3*hour, -hour, 1), (-2*hour, 0, 1), (-hour, hour, 1),
               (-25*hour, hour, 1), (-25*hour, -hour, 0),
               (-hour, -1, 0), (0, -1, 1), (hour, 2*hour, 1),
               (-hour, None, 1)]
    rows = [(target+d, None if a is None else target+a, 'AA', 'ORD', 'LAX', y) for d,a,y in offsets]
    source = spark.createDataFrame(rows, '_prediction_us long, _available_us long, reporting_airline string, origin string, destination string, is_arrival_delayed_15m int')
    result = entity_history(source, entity, keys).filter(F.col('_prediction_us') == target).first()
    for days in WINDOW_DAYS:
        oracle = [y for d,a,y in offsets if -days*DAY_US <= d < 0 and a is not None and a < 0]
        assert result[f'{entity}_flight_count_{days}d'] == len(oracle)
        assert result[f'{entity}_delay_rate_{days}d'] == pytest.approx(sum(oracle)/len(oracle))
    changed = source.withColumn(TARGET_COLUMN, F.when(
        (F.col('_prediction_us') >= target) | (F.col('_available_us') >= target),
        1-F.col(TARGET_COLUMN)).otherwise(F.col(TARGET_COLUMN)))
    after = entity_history(changed, entity, keys).filter(F.col('_prediction_us') == target).first()
    assert result == after


def test_schedule_midnight_missing_and_no_actual_time_dependency(dwd):
    """Exercise the schedule midnight missing and no actual time dependency case."""
    row = dwd.orderBy("flight_date").limit(1).withColumn("scheduled_departure_hour", F.lit(0)).withColumn("scheduled_departure_minute", F.lit(0))
    row = row.withColumn("scheduled_arrival_minute", F.lit(None).cast("int"))
    row = row.withColumn("dep_delay", F.lit(999)).withColumn("arr_delay", F.lit(999))
    value = add_schedule_features(row).first()
    assert value["departure_sin"] == pytest.approx(0)
    assert value["departure_cos"] == pytest.approx(1)
    assert value["arrival_missing"] == 1
    assert value["arrival_sin"] is None



def test_eligibility_multiplicity_and_split_boundaries(spark, dwd, shared_path):
    """Exercise the eligibility multiplicity and split boundaries case."""
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
    build_ml_flight_features(spark, shared_path / "dwd", shared_path / "features", {"source_snapshot_id": SNAPSHOT_ID, "raw_manifest_sha256": "a" * 64})
    features = spark.read.parquet(str(shared_path / "features")).select(*OUTPUT_COLUMNS).cache()
    try:
        assert validate_feature_frame(features, 53) == 53
        assert set(p.name for p in (shared_path / "features").glob("split=*")) == {"split=train", "split=validation"}
        for day, split in zip(dates, expected):
            assigned = assign_chronological_split(source.filter(F.col("flight_date") == day)).first()
            assert assigned.split == split
        assert set(FEATURE_COLUMNS).isdisjoint(LEAKAGE_BLACKLIST)
        duplicate = features.filter(F.col("flight_date") == TRAIN_START)
        assert duplicate.count() == 3
    finally:
        features.unpersist()


@pytest.mark.parametrize("boundary", [VALIDATION_START, TEST_START])
@pytest.mark.parametrize("entity,keys", ENTITIES)
def test_history_crosses_both_split_boundaries(spark, boundary, entity, keys):
    """Exercise the history crosses both split boundaries case."""
    from airline.dws.flight_time import DAY_US
    target = int(datetime.combine(boundary, datetime.min.time(), tzinfo=timezone.utc).timestamp())*1_000_000
    source = spark.createDataFrame([(target-DAY_US, target-1, 'AA', 'ORD', 'LAX', 1),
                                    (target, target+1, 'AA', 'ORD', 'LAX', 0)],
        '_prediction_us long, _available_us long, reporting_airline string, origin string, destination string, is_arrival_delayed_15m int')
    row = entity_history(source, entity, keys).filter(F.col('_prediction_us') == target).first()
    for days in WINDOW_DAYS:
        assert row[f'{entity}_flight_count_{days}d'] == 1
        assert row[f'{entity}_delay_rate_{days}d'] == 1.0


@pytest.mark.parametrize("column,value", [("is_cancelled", 2), (TARGET_COLUMN, 2), ("flight_date", date(2025,1,1))])
def test_invalid_source_fails(dwd, column, value):
    """Exercise the invalid source fails case."""
    with pytest.raises(AssertionError):
        validate_ml_source(dwd.withColumn(column, F.lit(value)))


def test_null_entities_preserve_multiplicity_and_history(dwd):
    """Exercise the null entities preserve multiplicity and history case."""
    source = dwd.orderBy("flight_date").limit(3).withColumn("origin", F.lit(None).cast("string"))
    featured = transform_ml_flight_features(source).orderBy("flight_date").collect()
    assert len(featured) == 3
    assert featured[0].origin_flight_count_7d == 0
    assert featured[1].origin_flight_count_7d == 1
    assert featured[2].origin_flight_count_7d == 2


def test_history_domain_and_schema_fail_closed(dwd):
    """Exercise the history domain and schema fail closed case."""
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


def test_selection_features_ignore_test_changes_and_target_outcomes(dwd):
    """Exercise the selection features ignore test changes and target outcomes case."""
    from airline.dws.build_ml_flight_features import select_eligible_flights
    source = dwd.orderBy('flight_date').limit(2)
    test = source.withColumn('flight_date', F.lit(TEST_START))
    before = transform_ml_flight_features(source.unionByName(test))
    after = transform_ml_flight_features(source.unionByName(test.withColumn(TARGET_COLUMN, 1-F.col(TARGET_COLUMN)).withColumn('dep_delay', F.lit(-9000.0))))
    assert before.exceptAll(after).count() == after.exceptAll(before).count() == 0
    one = source.orderBy('flight_date').limit(1)
    a = transform_ml_flight_features(one).drop(TARGET_COLUMN)
    b = transform_ml_flight_features(one.withColumn('dep_delay', F.lit(50000.0)).withColumn('actual_elapsed_time', F.lit(None).cast('double')).withColumn(TARGET_COLUMN, 1-F.col(TARGET_COLUMN))).drop(TARGET_COLUMN)
    assert a.exceptAll(b).count() == b.exceptAll(a).count() == 0


def test_utc_conversion_stable_after_repartition_reload(spark, dwd, shared_path):
    """Exercise the utc conversion stable after repartition reload case."""
    from airline.dws.build_ml_flight_features import select_eligible_flights
    original = select_eligible_flights(dwd).select('flight_id', '_prediction_us', '_available_us')
    dwd.repartition(3).write.parquet(str(shared_path / 'time-reload'))
    old = spark.conf.get('spark.sql.session.timeZone')
    try:
        spark.conf.set('spark.sql.session.timeZone', 'Pacific/Honolulu')
        after = select_eligible_flights(spark.read.parquet(str(shared_path / 'time-reload'))).select(*original.columns)
        assert original.exceptAll(after).count() == after.exceptAll(original).count() == 0
    finally:
        spark.conf.set('spark.sql.session.timeZone', old)


def test_test_materialization_requires_completed_selection(spark, shared_path):
    """Exercise the test materialization requires completed selection case."""
    with pytest.raises(ValueError, match='manifest'):
        build_ml_flight_features(spark, shared_path / 'absent-dwd', shared_path / 'blocked-test',
                                {'raw_manifest_sha256': 'a'*64}, phase='test')
    assert not (shared_path / 'blocked-test').exists()


def test_actual_arrival_uses_signed_delay_elapsed_and_fails_closed(dwd):
    """Exercise the actual arrival uses signed delay elapsed and fails closed case."""
    from airline.dws.build_ml_flight_features import select_eligible_flights
    one = dwd.orderBy('flight_date').limit(1)
    early = select_eligible_flights(one.withColumn('dep_delay', F.lit(-20.0)).withColumn('actual_elapsed_time', F.lit(60.0))).first()
    assert early['_available_us'] == early['_prediction_us'] + 40*60_000_000
    for value in (-1.0, float('nan'), float('inf'), 1e308, None):
        row = select_eligible_flights(one.withColumn('actual_elapsed_time', F.lit(value).cast('double'))).first()
        assert row['_available_us'] is None
        assert row['_prediction_us'] is not None


@pytest.mark.parametrize('entity,keys', ENTITIES)
def test_unknown_target_time_and_empty_entity_have_empty_histories(spark, entity, keys):
    """Exercise the unknown target time and empty entity have empty histories case."""
    frame = spark.createDataFrame([(None, None, 'AA', 'ORD', 'LAX', 1),
                                   (10, 20, 'AA', 'ORD', 'LAX', 1),
                                   (30, 40, 'NEW', 'NEW', 'NEW', 0)],
        '_prediction_us long, _available_us long, reporting_airline string, origin string, destination string, is_arrival_delayed_15m int')
    histories = entity_history(frame, entity, keys).collect()
    for row in histories:
        for days in WINDOW_DAYS:
            assert row[f'{entity}_flight_count_{days}d'] == 0
            assert row[f'{entity}_delay_rate_{days}d'] is None
            assert row[f'{entity}_history_missing_{days}d'] == 1

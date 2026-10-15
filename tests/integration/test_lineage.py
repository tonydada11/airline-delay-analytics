# AI-generated test code: generated with OpenAI Codex.
from datetime import date
import json
import os
from pathlib import Path
import tempfile
import pytest
from pyspark.sql import functions as F
from airline.common.spark_session import create_spark_session
from airline.common.lineage import BUSINESS_KEY, canonical_json, flight_id, with_lineage, key_json_column
from airline.ads import write_ads_table


@pytest.fixture(scope='module')
def spark():
    s = create_spark_session('lineage-tests')
    yield s
    s.stop()


def test_golden_json_hash_reload_and_duplicate_preservation(spark):
    values = dict(zip(BUSINESS_KEY, [date(2024, 2, 29), ' A"\\\n雪 ', 7, None, 12, 0]))
    literal = '{"flight_date":"2024-02-29","reporting_airline":" A\\"\\\\\\n雪 ","flight_number_reporting_airline":7,"origin_airport_id":null,"dest_airport_id":12,"crs_dep_time":0}'
    assert canonical_json(values) == literal
    # Literal digest pinned independently when the fixture was authored.
    assert flight_id(values) == '28c17df484189012eefa7fbbdba846771f83b32069a3df55278114a6bf5ced76'
    frame = spark.createDataFrame([tuple(values.values())] * 2,
        'flight_date date, reporting_airline string, flight_number_reporting_airline int, origin_airport_id int, dest_airport_id int, crs_dep_time int')
    assert frame.select(key_json_column(frame)).first()[0] == literal
    hashed = with_lineage(frame).repartition(2)
    root = Path(tempfile.mkdtemp(dir=os.environ.get('AIRLINE_TEST_TMP_ROOT', tempfile.gettempdir())))
    try:
        hashed.write.parquet(str(root / 'rows'))
        reloaded = spark.read.parquet(str(root / 'rows'))
        assert reloaded.count() == 2
        assert [r.flight_id for r in reloaded.collect()] == [flight_id(values)] * 2
        with pytest.raises(ValueError, match='integer'):
            with_lineage(frame.withColumn('crs_dep_time', F.lit(1.2)))
        nulls = frame.select(*[F.lit(None).cast(t).alias(k) for k,t in frame.dtypes])
        expected = '{' + ','.join('"'+k+'":null' for k in BUSINESS_KEY) + '}'
        assert nulls.select(key_json_column(nulls)).first()[0] == expected
    finally:
        import shutil
        shutil.rmtree(root)


def test_ads_one_file_logical_equality(spark):
    root = Path(tempfile.mkdtemp(dir=os.environ.get('AIRLINE_TEST_TMP_ROOT', tempfile.gettempdir())))
    try:
        original = spark.range(12).repartition(4)
        write_ads_table(original, root / 'ads')
        result = spark.read.parquet(str(root / 'ads'))
        assert result.dtypes == original.dtypes
        assert original.exceptAll(result).count() == result.exceptAll(original).count() == 0
        assert len(list((root / 'ads').glob('part-*.parquet'))) == 1
    finally:
        import shutil
        shutil.rmtree(root)

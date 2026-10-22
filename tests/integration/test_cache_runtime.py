# AI-generated test code: generated with OpenAI Codex.
"""Wide feature caches must round-trip under the supported small executor heap."""
import pytest
from pyspark import StorageLevel
from pyspark.sql import functions as F
from airline.common.spark_session import create_spark_session


def test_wide_disk_cache_roundtrip():
    """Exercise the wide disk cache roundtrip case."""
    spark = create_spark_session('airline-wide-cache-regression')
    first = second = None
    try:
        assert spark.conf.get('spark.sql.inMemoryColumnarStorage.batchSize') == '1000'
        # Multiple cache batches, with wide nonconstant values as in the staged
        # history build. Read one disk cache while materializing another.
        source = spark.range(25000, numPartitions=2).select(
            'id', *[(F.xxhash64('id', F.lit(i)).cast('double') / 1e18).alias(f'x{i}')
                    for i in range(60)])
        first = source.persist(StorageLevel.DISK_ONLY)
        assert first.count() == 25000
        second = first.select('id', *[(F.col(f'x{i}') + 1).alias(f'x{i}')
                                     for i in range(60)]).persist(StorageLevel.DISK_ONLY)
        assert second.count() == 25000
        actual = second.agg(*[F.sum(f'x{i}').alias(f'x{i}') for i in range(60)]).first()
        expected = first.agg(*[F.sum(F.col(f'x{i}') + 1).alias(f'x{i}') for i in range(60)]).first()
        assert list(actual) == pytest.approx(list(expected))
    finally:
        if second is not None:
            second.unpersist()
        if first is not None:
            first.unpersist()
        spark.stop()

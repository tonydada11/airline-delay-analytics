# AI-generated test code: generated with OpenAI Codex.
import pytest
from pyspark.sql import SparkSession

from airline.common.quality import (
    assert_not_empty,
    assert_required_columns,
    assert_row_count,
)
from airline.common.spark_session import create_spark_session


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    session = create_spark_session("airline-m0-quality-tests")
    yield session
    session.stop()


def test_assert_not_empty_accepts_rows(spark: SparkSession) -> None:
    assert_not_empty(spark.createDataFrame([(1,)], ["id"]))


def test_assert_not_empty_rejects_empty_dataframe(spark: SparkSession) -> None:
    with pytest.raises(AssertionError, match="at least one row"):
        assert_not_empty(spark.createDataFrame([], "id integer"))


def test_assert_required_columns_accepts_present_columns(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(1, "ORD")], ["id", "origin"])

    assert_required_columns(dataframe, ["id", "origin"])


def test_assert_required_columns_reports_missing_columns(spark: SparkSession) -> None:
    dataframe = spark.createDataFrame([(1,)], ["id"])

    with pytest.raises(AssertionError, match="origin"):
        assert_required_columns(dataframe, ["id", "origin"])


def test_assert_row_count_accepts_exact_count(spark: SparkSession) -> None:
    assert_row_count(spark.range(3), 3)


def test_assert_row_count_reports_actual_count(spark: SparkSession) -> None:
    with pytest.raises(AssertionError, match="Expected DataFrame row count 2, found 1"):
        assert_row_count(spark.range(1), 2)

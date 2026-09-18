"""SparkSession construction shared by pipeline jobs."""

from pyspark.sql import SparkSession


def create_spark_session(app_name: str) -> SparkSession:
    """Create or reuse a session using the runtime's Spark master settings."""

    if not isinstance(app_name, str) or not app_name.strip():
        raise ValueError("Spark application name must be a non-empty string.")
    return SparkSession.builder.appName(app_name.strip()).getOrCreate()

"""Build the ODS flight snapshot from raw BTS CSV files."""

from pathlib import Path
from typing import Union

from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import StringType, StructField, StructType

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.schemas import BTS_FLIGHTS_SCHEMA
from airline.common.spark_session import create_spark_session


PathLike = Union[str, Path]
SOURCE_FILE_COLUMN = "_source_file"
INGESTED_AT_COLUMN = "_ingested_at"
_TRAILING_ARTIFACT_COLUMN = "_trailing_csv_artifact"
_BTS_CSV_READ_SCHEMA = StructType(
    [
        *BTS_FLIGHTS_SCHEMA.fields,
        StructField(_TRAILING_ARTIFACT_COLUMN, StringType(), True),
    ]
)


def read_raw_flights(spark: SparkSession, raw_directory: PathLike) -> DataFrame:
    """Read all raw flight CSVs using the explicit BTS source schema."""

    input_pattern = str(Path(raw_directory) / "*.csv")
    physical_csv = (
        spark.read.option("header", True)
        .option("mode", "FAILFAST")
        .schema(_BTS_CSV_READ_SCHEMA)
        .csv(input_pattern)
    )
    return physical_csv.select(*BTS_FLIGHTS_SCHEMA.fieldNames())


def add_lineage_columns(dataframe: DataFrame) -> DataFrame:
    """Add portable source-file and ingestion-time lineage."""

    source_basename = F.regexp_extract(F.input_file_name(), r"([^/]+)$", 1)
    return dataframe.withColumn(SOURCE_FILE_COLUMN, source_basename).withColumn(
        INGESTED_AT_COLUMN, F.current_timestamp()
    )


def write_ods_flights(dataframe: DataFrame, output_directory: PathLike) -> None:
    """Replace the ODS flight snapshot with a Parquet dataset."""

    dataframe.write.mode("overwrite").parquet(str(output_directory))


def build_ods_flights(
    spark: SparkSession, raw_directory: PathLike, output_directory: PathLike
) -> None:
    """Orchestrate the raw-to-ODS flight snapshot build."""

    raw_flights = read_raw_flights(spark, raw_directory)
    flights_with_lineage = add_lineage_columns(raw_flights)
    write_ods_flights(flights_with_lineage, output_directory)


def main() -> None:
    """Run the configured M1 ODS flight build."""

    config = load_config()
    paths = get_project_paths(config)
    spark = create_spark_session("airline-m1-build-ods-flights")

    print(f"Raw input: {paths.raw_csv.as_posix()}/")
    print(f"ODS output: {paths.ods_flights.as_posix()}/")

    try:
        build_ods_flights(
            spark,
            paths.resolve(paths.raw_csv),
            paths.resolve(paths.ods_flights),
        )
        print("ODS flight snapshot written successfully.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

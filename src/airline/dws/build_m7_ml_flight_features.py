"""Materialize the frozen M7 feature contract at the configured ML DWS path.

The staged experiment already produced the accepted history feature bundle.
This promotion job copies that immutable checkpoint into the production-facing
DWS without reading or scoring the protected September--December test period.
"""

import json
from pathlib import Path
import time
from typing import Union

from pyspark.sql import DataFrame, SparkSession, functions as F

from airline.common.config import load_config
from airline.common.paths import get_project_paths
from airline.common.quality import assert_not_empty, assert_required_columns
from airline.common.spark_session import create_spark_session
from airline.dws.build_ml_flight_features import TARGET_COLUMN, VALIDATION_END
from airline.ml.m7_contract import FEATURE_CONTRACT_VERSION, OUTPUT_COLUMNS


PathLike = Union[str, Path]
DEFAULT_SOURCE = Path("artifacts/experiments/staged-v2/features/confirmation/history/data")
TARGET_ROWS_PER_PARTITION = 250_000


def validate_m7_feature_frame(dataframe: DataFrame) -> None:
    """Validate schema, target, chronology, and split domain without test access."""

    assert_not_empty(dataframe)
    assert_required_columns(dataframe, OUTPUT_COLUMNS)
    if tuple(dataframe.columns) != OUTPUT_COLUMNS:
        raise AssertionError("Configured ML DWS does not match m7-history-v1.")
    invalid = dataframe.agg(
        F.sum((F.col(TARGET_COLUMN).isNull() | ~F.col(TARGET_COLUMN).isin(0, 1)).cast("long")).alias("target"),
        F.sum((~F.col("split").isin("train", "validation") | F.col("split").isNull()).cast("long")).alias("split"),
        F.sum((F.col("flight_date") > F.lit(VALIDATION_END)).cast("long")).alias("test_or_future"),
    ).first()
    if any(int(invalid[name] or 0) for name in ("target", "split", "test_or_future")):
        raise AssertionError(f"Invalid M7 DWS rows: {invalid.asDict()}")


def materialize_frozen_m7_features(
    spark: SparkSession,
    source_directory: PathLike,
    output_directory: PathLike,
) -> dict:
    """Promote the accepted feature checkpoint with evidence-based partitioning."""

    source = spark.read.parquet(str(source_directory)).select(*OUTPUT_COLUMNS)
    validate_m7_feature_frame(source)
    rows = source.count()
    partitions = max(4, (rows + TARGET_ROWS_PER_PARTITION - 1) // TARGET_ROWS_PER_PARTITION)
    # Range distribution on date prevents a four-file/11M-row bottleneck while
    # partitionBy(split) preserves split pruning at read time.
    distributed = source.repartitionByRange(partitions, "flight_date")
    distributed.write.mode("overwrite").partitionBy("split").parquet(str(output_directory))
    written = spark.read.parquet(str(output_directory))
    validate_m7_feature_frame(written.select(*OUTPUT_COLUMNS))
    return {
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "rows": rows,
        "source_partitions": source.rdd.getNumPartitions(),
        "requested_output_partitions": partitions,
        "written_partitions": written.rdd.getNumPartitions(),
        "physical_data_files": len(list(Path(output_directory).rglob("part-*.parquet"))),
        "partition_columns": ["split"],
        "distribution": "repartitionByRange(flight_date)",
        "target_rows_per_partition": TARGET_ROWS_PER_PARTITION,
        "maximum_date": str(written.agg(F.max("flight_date")).first()[0]),
        "protected_test_rows_materialized": 0,
    }


def main() -> None:
    paths = get_project_paths(load_config())
    spark = create_spark_session("airline-m7-promote-frozen-features")
    try:
        started = time.monotonic()
        summary = materialize_frozen_m7_features(
            spark,
            paths.project_root / DEFAULT_SOURCE,
            paths.resolve(paths.dws_ml_flight_features),
        )
        summary["elapsed_seconds"] = time.monotonic() - started
        manifest = paths.project_root / "artifacts/experiments/staged-v2/m7/dws_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(summary)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

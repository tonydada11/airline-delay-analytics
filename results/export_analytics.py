"""Export existing aggregate ADS tables for review; never rebuild the pipeline."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


GRAINS = {
    "monthly_operations": ("year", "month"),
    "carrier_scorecard": ("reporting_airline",),
    "airport_scorecard": ("origin",),
    "route_ranking": ("origin", "destination"),
}


def digest(path):
    """Calculate a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Export the analytics tables to the results folder."""
    from pyspark.sql import SparkSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "results/analytics"
    output.mkdir(parents=True, exist_ok=True)
    spark = (SparkSession.builder.master("local[1]")
             .appName("airline-review-existing-ads")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "1").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    records = {}
    try:
        for name, grain in GRAINS.items():
            source = root / "data/warehouse/ads" / name
            frame = spark.read.parquet(str(source)).orderBy(*grain)
            rows = frame.limit(100001).collect()
            if len(rows) > 100000:
                raise ValueError("Aggregate table exceeds review export limit: " + name)
            destination = output / (name + ".csv")
            with destination.open("w", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(frame.columns)
                writer.writerows(tuple(row) for row in rows)
            records[name] = {
                "source": source.relative_to(root).as_posix(),
                "input_sha256": {p.relative_to(root).as_posix(): digest(p)
                                 for p in sorted(source.glob("*.parquet"))},
                "rows": len(rows), "columns": frame.columns, "ordered_by": grain,
                "csv": destination.name, "csv_sha256": digest(destination),
            }
    finally:
        spark.stop()
    (output / "manifest.json").write_text(json.dumps({
        "scope": "Existing aggregate tables only; no fitting, rescoring, or pipeline rebuild",
        "tables": records,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: value["rows"] for name, value in records.items()}, sort_keys=True))


if __name__ == "__main__":
    main()

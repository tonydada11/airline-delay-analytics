"""Pinned raw snapshot verification. Expected hashes are never updated at runtime."""
import csv
import hashlib
from pathlib import Path
import re
from airline.common.config import load_config

SNAPSHOT_ID = "bts-2023-2024-v1"
MONTHS = {(year, month) for year in (2023, 2024) for month in range(1, 13)}
FILENAME = "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{month}.csv"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["year", "month", "filename", "sha256"]:
            raise ValueError("Malformed raw manifest header")
        rows = list(reader)
    seen = set()
    for row in rows:
        try:
            month = (int(row["year"]), int(row["month"]))
        except (ValueError, TypeError) as exc:
            raise ValueError("Malformed manifest month") from exc
        if set(row) != {"year", "month", "filename", "sha256"} or month not in MONTHS or month in seen:
            raise ValueError("Unexpected or duplicate manifest month")
        seen.add(month)
        if row["filename"] != FILENAME.format(year=month[0], month=month[1]):
            raise ValueError("Wrong raw filename in manifest")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"] or ""):
            raise ValueError("Malformed SHA-256 in manifest")
    if seen != MONTHS:
        raise ValueError("Manifest must cover January 2023 through December 2024")
    return rows


def verify_snapshot(raw_directory, manifest_path):
    rows = read_manifest(manifest_path)
    root = Path(raw_directory)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"}
    expected = {r["filename"] for r in rows}
    if actual != expected:
        raise ValueError(f"Raw CSV inventory mismatch: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}")
    for row in rows:
        if sha256_file(root / row["filename"]) != row["sha256"]:
            raise ValueError(f"Raw SHA-256 mismatch: {row['filename']}")
    return {"source_snapshot_id": SNAPSHOT_ID, "raw_manifest_sha256": sha256_file(manifest_path)}


def main():
    config = load_config()
    print(verify_snapshot(config.project_root / config.raw_csv, config.project_root / config.raw_manifest))


if __name__ == "__main__":
    main()

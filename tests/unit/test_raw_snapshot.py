# AI-generated test code: generated with OpenAI Codex.
"""Tests raw snapshot behavior."""

import csv
import hashlib
import pytest
from airline.common.raw_snapshot import FILENAME, MONTHS, read_manifest, verify_snapshot


@pytest.fixture
def snapshot(tmp_path):
    """Create a valid raw snapshot fixture."""
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for year, month in sorted(MONTHS):
        filename = FILENAME.format(year=year, month=month)
        payload = f"{year},{month}\n".encode()
        (raw / filename).write_bytes(payload)
        rows.append([year, month, filename, hashlib.sha256(payload).hexdigest()])
    manifest = tmp_path / "manifest.csv"
    def write(rows, header=("year", "month", "filename", "sha256")):
        """Write the supplied value to the target path."""
        with manifest.open("w", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)
    write(rows)
    return raw, manifest, rows, write


def test_valid_snapshot(snapshot):
    """Exercise the valid snapshot case."""
    raw, manifest, _, _ = snapshot
    assert verify_snapshot(raw, manifest)["source_snapshot_id"] == "bts-2023-2024-v1"


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "wrong_name", "header", "hash", "coverage", "malformed_hash", "malformed_month", "extra_field"])
def test_rejects_snapshot_mismatch(snapshot, mutation):
    """Exercise the rejects snapshot mismatch case."""
    raw, manifest, rows, write = snapshot
    if mutation == "missing": (raw / rows[0][2]).unlink()
    elif mutation == "extra": (raw / "extra.CSV").write_text("extra")
    elif mutation == "duplicate": write(rows + [rows[0]])
    elif mutation == "wrong_name":
        rows[0][2] = "wrong.csv"; write(rows)
    elif mutation == "header": write(rows, ("bad", "month", "filename", "sha256"))
    elif mutation == "hash": (raw / rows[0][2]).write_text("changed")
    elif mutation == "coverage": write(rows[:-1])
    elif mutation == "malformed_hash":
        rows[0][3] = "no"; write(rows)
    elif mutation == "malformed_month":
        rows[0][1] = "no"; write(rows)
    elif mutation == "extra_field":
        rows[0].append("extra"); write(rows)
    with pytest.raises(ValueError): verify_snapshot(raw, manifest)

from pathlib import Path
import csv
from collections import Counter

DATA_DIR = Path("data/raw/csv")

required_columns = {
    "Year",
    "Month",
    "FlightDate",
    "Reporting_Airline",
    "Origin",
    "Dest",
}

files = sorted(DATA_DIR.glob("*.csv"))

print(f"CSV files found: {len(files)}\n")

for path in files:
    airlines = Counter()
    rows = 0
    bad_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            print(f"INVALID: {path.name} — no header")
            continue

        missing = required_columns - set(reader.fieldnames)

        if missing:
            print(f"INVALID: {path.name}")
            print(f"  Missing columns: {sorted(missing)}")
            continue

        expected_columns = len(reader.fieldnames)

        for row in reader:
            rows += 1

            if len(row) != expected_columns:
                bad_rows += 1

            airline = row.get("Reporting_Airline", "").strip()
            if airline:
                airlines[airline] += 1

    print(path.name)
    print(f"  rows: {rows:,}")
    print(f"  airlines: {len(airlines)}")
    print(f"  airline codes: {sorted(airlines)}")
    print(f"  top airlines: {airlines.most_common(10)}")
    print(f"  malformed rows: {bad_rows}")

    if len(airlines) == 1:
        print("  WARNING: only one airline found!")

    print()
"""Pure definitions for the rolling as-of-date production feature contract."""
from datetime import date

FEATURE_CONTRACT_VERSION = "m7-history-v1"
TARGET_COLUMN = "is_arrival_delayed_15m"
SPLIT_COLUMN = "split"
ENTITIES = (
    ("carrier", ("reporting_airline",)),
    ("origin", ("origin",)),
    ("destination", ("destination",)),
    ("route", ("origin", "destination")),
)
WINDOW_DAYS = (7, 30, 90)
SCHEDULE_NUMERIC = ("distance", "crs_elapsed_time", "departure_sin", "departure_cos",
                    "arrival_sin", "arrival_cos", "departure_missing", "arrival_missing",
                    "duration_missing")
SCHEDULE_CATEGORICAL = ("reporting_airline", "origin", "destination", "calendar_month", "calendar_weekday")
HISTORY_RATES = tuple(f"{name}_delay_rate_{days}d" for name, _ in ENTITIES for days in WINDOW_DAYS)
HISTORY_COUNTS = tuple(f"{name}_flight_count_{days}d" for name, _ in ENTITIES for days in WINDOW_DAYS)
HISTORY_MISSING = tuple(f"{name}_history_missing_{days}d" for name, _ in ENTITIES for days in WINDOW_DAYS)

LEAKAGE_BLACKLIST = frozenset(
    {
        "dep_time",
        "dep_delay",
        "dep_delay_minutes",
        "taxi_out",
        "wheels_off",
        "wheels_on",
        "arr_time",
        "arr_delay",
        "arr_delay_minutes",
        "actual_elapsed_time",
        "air_time",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay",
    }
)
TRAIN_START = date(2023, 1, 1)
TRAIN_END = date(2024, 4, 30)
VALIDATION_START = date(2024, 5, 1)
VALIDATION_END = date(2024, 8, 31)
TEST_START = date(2024, 9, 1)
TEST_END = date(2024, 12, 31)

CATEGORICAL_COLUMNS = SCHEDULE_CATEGORICAL
NUMERIC_COLUMNS = (*SCHEDULE_NUMERIC, *HISTORY_RATES)
FEATURE_COLUMNS = (*CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS, *HISTORY_COUNTS, *HISTORY_MISSING)
OUTPUT_COLUMNS = ("flight_date", *FEATURE_COLUMNS, TARGET_COLUMN, SPLIT_COLUMN)
SPLIT_BOUNDARIES = {
    "train": {"start": TRAIN_START.isoformat(), "end": TRAIN_END.isoformat()},
    "validation": {"start": VALIDATION_START.isoformat(), "end": VALIDATION_END.isoformat()},
    "test": {"start": TEST_START.isoformat(), "end": TEST_END.isoformat()},
}
HISTORICAL_TEST_STATUS = "previously inspected historical test period; not a pristine external holdout"

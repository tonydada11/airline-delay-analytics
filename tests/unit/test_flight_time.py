# AI-generated test code: generated with OpenAI Codex.
"""Tests flight time behavior."""

from datetime import date, datetime, timezone
import pytest
from airline.dws.flight_time import scheduled_converter, airport_zones, time_provenance


def epoch(day, hour=0, minute=0):
    """Convert a timestamp to epoch microseconds."""
    return int(datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc).timestamp())*1_000_000


def test_clock_midnight_timezone_and_dst_fail_closed():
    """Exercise the clock midnight timezone and dst fail closed case."""
    convert = scheduled_converter({1: 'America/Chicago', 2: 'Pacific/Honolulu', 3: 'America/New_York'})
    assert convert(date(2024, 1, 1), 2400, 1) == epoch(date(2024, 1, 2), 6)
    assert convert(date(2024, 1, 1), 2330, 2) == epoch(date(2024, 1, 2), 9, 30)
    assert convert(date(2024, 3, 10), 230, 3) is None
    assert convert(date(2024, 11, 3), 130, 3) is None
    assert convert(date(2024, 3, 10), 330, 3) == epoch(date(2024, 3, 10), 7, 30)
    for invalid in (-1, 1260, 2401, 2500, None):
        assert convert(date(2024, 1, 1), invalid, 1) is None
    with pytest.raises(ValueError, match='Unmapped'):
        convert(date(2024, 1, 1), 900, 99)


def test_reference_and_explicit_availability_assumption():
    """Exercise the reference and explicit availability assumption case."""
    zones = airport_zones()
    assert len(zones) == 355
    assert zones[13930] == 'America/Chicago'
    assert zones[12892] == 'America/Los_Angeles'
    assert time_provenance()['reporting_latency_seconds'] == 0
    assert 'not observed' in time_provenance()['limitation']

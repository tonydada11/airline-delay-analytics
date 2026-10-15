"""UTC event times, with an explicit zero-latency arrival availability assumption."""
import csv
from pathlib import Path
import pytz
from pyspark.sql import functions as F
from airline.common.raw_snapshot import sha256_file

REFERENCE = Path(__file__).resolve().parents[3] / 'conf/airport_timezones.csv'
DAY_US = 86_400_000_000
TIME_POLICY = {
    'prediction': 'scheduled departure UTC',
    'availability': 'scheduled departure UTC + signed departure delay + actual elapsed time',
    'reporting_latency_seconds': 0,
    'limitation': 'Arrival-event assumption; actual publication/ingestion availability is not observed.',
    'window': 'target-N*24h <= source scheduled departure < target; arrival < target',
    'uncertain_time': 'exclude source; retain target with empty histories',
    'test_history': 'causal rolling outcomes, after frozen selection only',
}


def time_provenance():
    return {**TIME_POLICY, 'timezone_reference_sha256': sha256_file(REFERENCE),
            'pytz_version': pytz.__version__}


def airport_zones():
    with REFERENCE.open() as stream:
        rows = list(csv.DictReader(stream))
    mapping = {int(r['airport_id']): r['timezone'] for r in rows}
    if len(mapping) != len(rows):
        raise ValueError('Duplicate airport timezone reference keys')
    for zone in set(mapping.values()):
        pytz.timezone(zone)
    return mapping


def scheduled_converter(zones):
    # Self-contained closure: workers need only the locked pytz distribution, not source mounts.
    def convert(day, hhmm, airport_id):
        from datetime import datetime, timedelta
        import pytz
        if day is None or hhmm is None:
            return None
        if airport_id not in zones:
            raise ValueError('Unmapped airport ID; update the pinned timezone reference')
        if hhmm != int(hhmm) or hhmm < 0 or hhmm > 2400 or hhmm % 100 >= 60:
            return None
        local = datetime.combine(day, datetime.min.time()) + timedelta(
            hours=hhmm // 100, minutes=hhmm % 100)
        try:
            aware = pytz.timezone(zones[airport_id]).localize(local, is_dst=None)
        except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError):
            return None
        delta = aware.astimezone(pytz.UTC) - datetime(1970, 1, 1, tzinfo=pytz.UTC)
        return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
    return convert


def with_event_times(frame):
    zones = airport_zones()
    unknown = frame.select('origin_airport_id').distinct().filter(
        F.col('origin_airport_id').isNull() | ~F.col('origin_airport_id').isin(list(zones)))
    if unknown.limit(1).count():
        raise ValueError('Unmapped origin airport ID; update the pinned timezone reference')
    convert = F.udf(scheduled_converter(zones), 'long')
    result = frame.withColumn('_prediction_us', convert('flight_date', 'crs_dep_time', 'origin_airport_id'))
    valid = F.col('_prediction_us').isNotNull()
    for column in ('dep_delay', 'actual_elapsed_time'):
        valid = valid & F.col(column).isNotNull() & ~F.isnan(column) & (F.abs(F.col(column)) != float('inf'))
    offset = F.round((F.col('dep_delay') + F.col('actual_elapsed_time')) * 60_000_000)
    # Leave ample signed-long headroom for the epoch before casting/adding.
    # Non-finite sums and overflowing durations must never wrap into the past.
    valid = valid & (F.col('actual_elapsed_time') >= 0) & offset.between(-8e18, 8e18)
    return result.withColumn('_available_us', F.when(valid,
        F.col('_prediction_us') + offset.cast('long')))

"""Canonical UTF-8 business-key JSON; IDs describe key values, not source rows."""
import hashlib
import json
from datetime import date

BUSINESS_KEY = ('flight_date', 'reporting_airline', 'flight_number_reporting_airline',
                'origin_airport_id', 'dest_airport_id', 'crs_dep_time')
INTEGER_KEYS = BUSINESS_KEY[2:]
LINEAGE_DEFINITION = {
    'fields': [{'name': k, 'type': 'integer' if k in INTEGER_KEYS else 'string'} for k in BUSINESS_KEY],
    'date_format': 'yyyy-MM-dd', 'encoding': 'UTF-8', 'json': 'compact, explicit nulls, standard escaping, preserved strings',
    'hash': 'lowercase SHA-256 hex', 'grain': 'business-key value; duplicates preserved',
}


def canonical_json(values):
    result = {k: values[k] for k in BUSINESS_KEY}
    if result['reporting_airline'] is not None and not isinstance(result['reporting_airline'], str):
        raise ValueError('reporting_airline must be a string or null')
    if type(result['flight_date']) is date:
        result['flight_date'] = result['flight_date'].isoformat()
    elif result['flight_date'] is not None:
        value = result['flight_date']
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError('flight_date must be an ISO yyyy-MM-dd date or null')
    for k in INTEGER_KEYS:
        if result[k] is not None and type(result[k]) is not int:
            raise ValueError(f'{k} must be an integer or null')
    return json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def flight_id(values):
    return hashlib.sha256(canonical_json(values).encode('utf-8')).hexdigest()


def key_json_column(frame):
    from pyspark.sql import functions as F
    types = dict(frame.dtypes)
    for k in INTEGER_KEYS:
        if types.get(k) not in ('tinyint', 'smallint', 'int', 'bigint'):
            raise ValueError(f'{k} must have an integer input type')
    if types.get('flight_date') != 'date' or types.get('reporting_airline') != 'string':
        raise ValueError('Lineage requires date flight_date and string reporting_airline')
    fields = [F.date_format('flight_date', 'yyyy-MM-dd').alias('flight_date'),
              *[F.col(k) for k in BUSINESS_KEY[1:]]]
    return F.to_json(F.struct(*fields), options={'ignoreNullFields': 'false'})


def with_lineage(frame):
    from pyspark.sql import functions as F
    return frame.withColumn('flight_id', F.sha2(key_json_column(frame), 256))

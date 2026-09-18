# AI-generated test code: generated with OpenAI Codex.
from pyspark.sql.types import DateType, DoubleType, IntegerType, StringType

from airline.common.schemas import BTS_FLIGHTS_SCHEMA


def test_bts_schema_has_all_named_source_fields_in_order() -> None:
    names = BTS_FLIGHTS_SCHEMA.fieldNames()

    assert len(names) == 109
    assert len(names) == len(set(names))
    assert names[:6] == [
        "Year",
        "Quarter",
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "FlightDate",
    ]
    assert names[-1] == "Div5TailNum"
    assert "" not in names
    assert "_c109" not in names


def test_bts_schema_uses_explicit_semantic_types() -> None:
    assert isinstance(BTS_FLIGHTS_SCHEMA["Year"].dataType, IntegerType)
    assert isinstance(BTS_FLIGHTS_SCHEMA["FlightDate"].dataType, DateType)
    assert isinstance(BTS_FLIGHTS_SCHEMA["OriginStateFips"].dataType, StringType)
    assert isinstance(BTS_FLIGHTS_SCHEMA["DestStateFips"].dataType, StringType)
    assert isinstance(BTS_FLIGHTS_SCHEMA["DepDelay"].dataType, DoubleType)
    assert isinstance(BTS_FLIGHTS_SCHEMA["Div5AirportID"].dataType, IntegerType)


def test_bts_source_fields_are_nullable() -> None:
    assert all(field.nullable for field in BTS_FLIGHTS_SCHEMA.fields)

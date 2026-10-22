# AI-generated test code: generated with OpenAI Codex.
"""Tests schemas behavior."""

from pyspark.sql.types import DateType, DoubleType, IntegerType, StringType

from airline.common.schemas import BTS_FLIGHTS_SCHEMA


def test_bts_schema_has_all_named_source_fields_in_order() -> None:
    """Exercise the bts schema has all named source fields in order case."""
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
    """Exercise the bts schema uses explicit semantic types case."""
    assert isinstance(BTS_FLIGHTS_SCHEMA["Year"].dataType, IntegerType)
    assert isinstance(BTS_FLIGHTS_SCHEMA["FlightDate"].dataType, DateType)
    assert isinstance(BTS_FLIGHTS_SCHEMA["OriginStateFips"].dataType, StringType)
    assert isinstance(BTS_FLIGHTS_SCHEMA["DestStateFips"].dataType, StringType)
    assert isinstance(BTS_FLIGHTS_SCHEMA["DepDelay"].dataType, DoubleType)
    assert isinstance(BTS_FLIGHTS_SCHEMA["Div5AirportID"].dataType, IntegerType)


def test_bts_source_fields_are_nullable() -> None:
    """Exercise the bts source fields are nullable case."""
    assert all(field.nullable for field in BTS_FLIGHTS_SCHEMA.fields)

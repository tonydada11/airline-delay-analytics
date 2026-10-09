# AI-generated test code: generated with OpenAI Codex.
from pathlib import Path

import pytest

from airline.common.config import ProjectConfig, load_config
from airline.common.paths import get_project_paths


def test_project_paths_remain_portable() -> None:
    paths = get_project_paths(load_config())

    assert paths.raw_csv == Path("data/raw/csv")
    assert paths.ods_flights == Path("data/warehouse/ods/flights")
    assert paths.dwd_flights == Path("data/warehouse/dwd/flights")
    assert paths.dim_date == Path("data/warehouse/dim/date")
    assert paths.dim_carrier == Path("data/warehouse/dim/carrier")
    assert paths.dim_airport == Path("data/warehouse/dim/airport")
    assert paths.dim_route == Path("data/warehouse/dim/route")
    assert paths.dws_carrier_daily == Path("data/warehouse/dws/carrier_daily")
    assert paths.dws_origin_airport_daily == Path(
        "data/warehouse/dws/origin_airport_daily"
    )
    assert paths.dws_route_daily == Path("data/warehouse/dws/route_daily")
    assert paths.dws_ml_flight_features == Path(
        "data/warehouse/dws/ml_flight_features"
    )
    assert paths.ads_monthly_operations == Path(
        "data/warehouse/ads/monthly_operations"
    )
    assert paths.ads_carrier_scorecard == Path(
        "data/warehouse/ads/carrier_scorecard"
    )
    assert paths.ads_airport_scorecard == Path(
        "data/warehouse/ads/airport_scorecard"
    )
    assert paths.ads_route_ranking == Path("data/warehouse/ads/route_ranking")
    assert not paths.raw_csv.is_absolute()
    assert not paths.ods_flights.is_absolute()
    assert not paths.dwd_flights.is_absolute()
    assert not paths.dim_date.is_absolute()
    assert not paths.dim_carrier.is_absolute()
    assert not paths.dim_airport.is_absolute()
    assert not paths.dim_route.is_absolute()
    assert not paths.dws_carrier_daily.is_absolute()
    assert not paths.dws_origin_airport_daily.is_absolute()
    assert not paths.dws_route_daily.is_absolute()
    assert not paths.dws_ml_flight_features.is_absolute()
    assert not paths.ads_monthly_operations.is_absolute()
    assert not paths.ads_carrier_scorecard.is_absolute()
    assert not paths.ads_airport_scorecard.is_absolute()
    assert not paths.ads_route_ranking.is_absolute()


def test_resolve_uses_configured_project_root(tmp_path: Path) -> None:
    config = ProjectConfig(
        project_root=tmp_path,
        raw_csv=Path("input/flights"),
        ods_flights=Path("warehouse/ods/flights"),
        dwd_flights=Path("warehouse/dwd/flights"),
        dim_date=Path("warehouse/dim/date"),
        dim_carrier=Path("warehouse/dim/carrier"),
        dim_airport=Path("warehouse/dim/airport"),
        dim_route=Path("warehouse/dim/route"),
        dws_carrier_daily=Path("warehouse/dws/carrier_daily"),
        dws_origin_airport_daily=Path("warehouse/dws/origin_airport_daily"),
        dws_route_daily=Path("warehouse/dws/route_daily"),
        dws_ml_flight_features=Path("warehouse/dws/ml_flight_features"),
        ads_monthly_operations=Path("warehouse/ads/monthly_operations"),
        ads_carrier_scorecard=Path("warehouse/ads/carrier_scorecard"),
        ads_airport_scorecard=Path("warehouse/ads/airport_scorecard"),
        ads_route_ranking=Path("warehouse/ads/route_ranking"),
    )
    paths = get_project_paths(config)

    assert paths.resolve(paths.raw_csv) == tmp_path / "input/flights"
    assert paths.resolve(paths.ods_flights) == tmp_path / "warehouse/ods/flights"
    assert paths.resolve(paths.dwd_flights) == tmp_path / "warehouse/dwd/flights"
    assert paths.resolve(paths.dim_date) == tmp_path / "warehouse/dim/date"
    assert paths.resolve(paths.dim_carrier) == tmp_path / "warehouse/dim/carrier"
    assert paths.resolve(paths.dim_airport) == tmp_path / "warehouse/dim/airport"
    assert paths.resolve(paths.dim_route) == tmp_path / "warehouse/dim/route"
    assert paths.resolve(paths.dws_carrier_daily) == (
        tmp_path / "warehouse/dws/carrier_daily"
    )
    assert paths.resolve(paths.dws_origin_airport_daily) == (
        tmp_path / "warehouse/dws/origin_airport_daily"
    )
    assert paths.resolve(paths.dws_route_daily) == (
        tmp_path / "warehouse/dws/route_daily"
    )
    assert paths.resolve(paths.dws_ml_flight_features) == (
        tmp_path / "warehouse/dws/ml_flight_features"
    )
    assert paths.resolve(paths.ads_monthly_operations) == (
        tmp_path / "warehouse/ads/monthly_operations"
    )
    assert paths.resolve(paths.ads_carrier_scorecard) == (
        tmp_path / "warehouse/ads/carrier_scorecard"
    )
    assert paths.resolve(paths.ads_airport_scorecard) == (
        tmp_path / "warehouse/ads/airport_scorecard"
    )
    assert paths.resolve(paths.ads_route_ranking) == (
        tmp_path / "warehouse/ads/route_ranking"
    )


@pytest.mark.parametrize(
    "unsafe_path", [Path("/tmp/flights"), Path("../flights"), Path("C:\\data\\flights")]
)
def test_resolve_rejects_unsafe_paths(unsafe_path: Path) -> None:
    paths = get_project_paths(load_config())

    with pytest.raises(ValueError, match="project-relative"):
        paths.resolve(unsafe_path)

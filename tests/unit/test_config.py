# AI-generated test code: generated with OpenAI Codex.
"""Tests config behavior."""

from pathlib import Path

import pytest

from airline.common.config import load_config


DIMENSION_PATHS = (
    "  dim_date: warehouse/dim/date\n"
    "  dim_carrier: warehouse/dim/carrier\n"
    "  dim_airport: warehouse/dim/airport\n"
    "  dim_route: warehouse/dim/route\n"
)
DWS_PATHS = (
    "  dws_carrier_daily: warehouse/dws/carrier_daily\n"
    "  dws_origin_airport_daily: warehouse/dws/origin_airport_daily\n"
    "  dws_route_daily: warehouse/dws/route_daily\n"
    "  dws_ml_flight_features: warehouse/dws/ml_flight_features\n"
    "  dws_ml_test_features: warehouse/dws/ml_test_features\n"
)
ADS_PATHS = (
    "  ads_monthly_operations: warehouse/ads/monthly_operations\n"
    "  ads_carrier_scorecard: warehouse/ads/carrier_scorecard\n"
    "  ads_airport_scorecard: warehouse/ads/airport_scorecard\n"
    "  ads_route_ranking: warehouse/ads/route_ranking\n"
)


def _write_config(tmp_path: Path, body: str) -> Path:
    """Write config."""
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    config_path = config_dir / "project.yaml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def test_load_default_config() -> None:
    """Exercise the load default config case."""
    config = load_config()

    assert config.raw_csv == Path("data/raw/csv")
    assert config.ods_flights == Path("data/warehouse/ods/flights")
    assert config.dwd_flights == Path("data/warehouse/dwd/flights")
    assert config.dim_date == Path("data/warehouse/dim/date")
    assert config.dim_carrier == Path("data/warehouse/dim/carrier")
    assert config.dim_airport == Path("data/warehouse/dim/airport")
    assert config.dim_route == Path("data/warehouse/dim/route")
    assert config.dws_carrier_daily == Path("data/warehouse/dws/carrier_daily")
    assert config.dws_origin_airport_daily == Path(
        "data/warehouse/dws/origin_airport_daily"
    )
    assert config.dws_route_daily == Path("data/warehouse/dws/route_daily")
    assert config.dws_ml_flight_features == Path(
        "data/warehouse/dws/ml_flighttime_v3_selection"
    )
    assert config.ads_monthly_operations == Path(
        "data/warehouse/ads/monthly_operations"
    )
    assert config.ads_carrier_scorecard == Path(
        "data/warehouse/ads/carrier_scorecard"
    )
    assert config.ads_airport_scorecard == Path(
        "data/warehouse/ads/airport_scorecard"
    )
    assert config.ads_route_ranking == Path("data/warehouse/ads/route_ranking")
    assert (config.project_root / "conf/project.yaml").is_file()


def test_load_custom_config_sets_project_root(tmp_path: Path) -> None:
    """Exercise the load custom config sets project root case."""
    config_path = _write_config(
        tmp_path,
        "paths:\n  raw_manifest: conf/raw_data_manifest.csv\n  artifact_root: artifacts\n  raw_csv: inputs/flights\n  ods_flights: warehouse/ods\n"
        "  dwd_flights: warehouse/dwd\n"
        + DIMENSION_PATHS
        + DWS_PATHS
        + ADS_PATHS,
    )

    config = load_config(config_path)

    assert config.project_root == tmp_path
    assert config.raw_csv == Path("inputs/flights")
    assert config.dwd_flights == Path("warehouse/dwd")
    assert config.dim_route == Path("warehouse/dim/route")
    assert config.dws_route_daily == Path("warehouse/dws/route_daily")
    assert config.dws_ml_flight_features == Path(
        "warehouse/dws/ml_flight_features"
    )
    assert config.ads_monthly_operations == Path(
        "warehouse/ads/monthly_operations"
    )
    assert config.ads_route_ranking == Path("warehouse/ads/route_ranking")


@pytest.mark.parametrize(
    "body, message",
    [
        ("[]\n", "YAML mapping"),
        ("paths: []\n", "'paths' mapping"),
        ("paths:\n  raw_manifest: conf/raw_data_manifest.csv\n  artifact_root: artifacts\n  raw_csv: data/raw/csv\n", "missing required path keys"),
        (
            "paths:\n  raw_manifest: conf/raw_data_manifest.csv\n  artifact_root: artifacts\n  raw_csv: /var/data/flights\n  ods_flights: warehouse/ods\n"
            "  dwd_flights: warehouse/dwd\n"
            + DIMENSION_PATHS
            + DWS_PATHS
            + ADS_PATHS,
            "project-relative",
        ),
        (
            "paths:\n  raw_manifest: conf/raw_data_manifest.csv\n  artifact_root: artifacts\n  raw_csv: 'C:\\data\\flights'\n  ods_flights: warehouse/ods\n"
            "  dwd_flights: warehouse/dwd\n"
            + DIMENSION_PATHS
            + DWS_PATHS
            + ADS_PATHS,
            "project-relative",
        ),
        (
            "paths:\n  raw_manifest: conf/raw_data_manifest.csv\n  artifact_root: artifacts\n  raw_csv: ../flights\n  ods_flights: warehouse/ods\n"
            "  dwd_flights: warehouse/dwd\n"
            + DIMENSION_PATHS
            + DWS_PATHS
            + ADS_PATHS,
            "must not traverse",
        ),
        (
            "paths:\n  raw_manifest: conf/raw_data_manifest.csv\n  artifact_root: artifacts\n  raw_csv: ''\n  ods_flights: warehouse/ods\n"
            "  dwd_flights: warehouse/dwd\n"
            + DIMENSION_PATHS
            + DWS_PATHS
            + ADS_PATHS,
            "non-empty string",
        ),
        (
            "paths:\n  raw_manifest: conf/raw_data_manifest.csv\n  artifact_root: artifacts\n  raw_csv: data/raw/csv\n  ods_flights: warehouse/ods\n"
            "  dwd_flights: warehouse/dwd\n"
            + DIMENSION_PATHS
            + "  dws_carrier_daily: warehouse/dws/carrier_daily\n"
            "  dws_origin_airport_daily: warehouse/dws/origin_airport_daily\n"
            "  dws_route_daily: warehouse/dws/route_daily\n"
            "  dws_ml_flight_features: warehouse/dws/ml_flight_features\n"
    "  dws_ml_test_features: warehouse/dws/ml_test_features\n"
            "  ads_monthly_operations: warehouse/ads/monthly_operations\n"
            "  ads_carrier_scorecard: warehouse/ads/carrier_scorecard\n"
            "  ads_airport_scorecard: warehouse/ads/airport_scorecard\n"
            "  ads_route_ranking: /var/data/route_ranking\n",
            "project-relative",
        ),
    ],
)
def test_rejects_invalid_config(tmp_path: Path, body: str, message: str) -> None:
    """Exercise the rejects invalid config case."""
    with pytest.raises(ValueError, match=message):
        load_config(_write_config(tmp_path, body))


def test_rejects_invalid_yaml(tmp_path: Path) -> None:
    """Exercise the rejects invalid yaml case."""
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(_write_config(tmp_path, "paths: [\n"))


def test_missing_config_file_raises_file_not_found(tmp_path: Path) -> None:
    """Exercise the missing config file raises file not found case."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")

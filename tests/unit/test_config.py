# AI-generated test code: generated with OpenAI Codex.
from pathlib import Path

import pytest

from airline.common.config import load_config


DIMENSION_PATHS = (
    "  dim_date: warehouse/dim/date\n"
    "  dim_carrier: warehouse/dim/carrier\n"
    "  dim_airport: warehouse/dim/airport\n"
    "  dim_route: warehouse/dim/route\n"
)


def _write_config(tmp_path: Path, body: str) -> Path:
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    config_path = config_dir / "project.yaml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def test_load_default_config() -> None:
    config = load_config()

    assert config.raw_csv == Path("data/raw/csv")
    assert config.ods_flights == Path("data/warehouse/ods/flights")
    assert config.dwd_flights == Path("data/warehouse/dwd/flights")
    assert config.dim_date == Path("data/warehouse/dim/date")
    assert config.dim_carrier == Path("data/warehouse/dim/carrier")
    assert config.dim_airport == Path("data/warehouse/dim/airport")
    assert config.dim_route == Path("data/warehouse/dim/route")
    assert (config.project_root / "conf/project.yaml").is_file()


def test_load_custom_config_sets_project_root(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        "paths:\n  raw_csv: inputs/flights\n  ods_flights: warehouse/ods\n"
        "  dwd_flights: warehouse/dwd\n"
        + DIMENSION_PATHS,
    )

    config = load_config(config_path)

    assert config.project_root == tmp_path
    assert config.raw_csv == Path("inputs/flights")
    assert config.dwd_flights == Path("warehouse/dwd")
    assert config.dim_route == Path("warehouse/dim/route")


@pytest.mark.parametrize(
    "body, message",
    [
        ("[]\n", "YAML mapping"),
        ("paths: []\n", "'paths' mapping"),
        ("paths:\n  raw_csv: data/raw/csv\n", "missing required path keys"),
        (
            "paths:\n  raw_csv: /var/data/flights\n  ods_flights: warehouse/ods\n"
            "  dwd_flights: warehouse/dwd\n" + DIMENSION_PATHS,
            "project-relative",
        ),
        (
            "paths:\n  raw_csv: 'C:\\data\\flights'\n  ods_flights: warehouse/ods\n"
            "  dwd_flights: warehouse/dwd\n" + DIMENSION_PATHS,
            "project-relative",
        ),
        (
            "paths:\n  raw_csv: ../flights\n  ods_flights: warehouse/ods\n"
            "  dwd_flights: warehouse/dwd\n" + DIMENSION_PATHS,
            "must not traverse",
        ),
        (
            "paths:\n  raw_csv: ''\n  ods_flights: warehouse/ods\n"
            "  dwd_flights: warehouse/dwd\n" + DIMENSION_PATHS,
            "non-empty string",
        ),
    ],
)
def test_rejects_invalid_config(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(_write_config(tmp_path, body))


def test_rejects_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid YAML"):
        load_config(_write_config(tmp_path, "paths: [\n"))


def test_missing_config_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")

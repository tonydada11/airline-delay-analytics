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
    assert not paths.raw_csv.is_absolute()
    assert not paths.ods_flights.is_absolute()
    assert not paths.dwd_flights.is_absolute()


def test_resolve_uses_configured_project_root(tmp_path: Path) -> None:
    config = ProjectConfig(
        project_root=tmp_path,
        raw_csv=Path("input/flights"),
        ods_flights=Path("warehouse/ods/flights"),
        dwd_flights=Path("warehouse/dwd/flights"),
    )
    paths = get_project_paths(config)

    assert paths.resolve(paths.raw_csv) == tmp_path / "input/flights"
    assert paths.resolve(paths.ods_flights) == tmp_path / "warehouse/ods/flights"
    assert paths.resolve(paths.dwd_flights) == tmp_path / "warehouse/dwd/flights"


@pytest.mark.parametrize(
    "unsafe_path", [Path("/tmp/flights"), Path("../flights"), Path("C:\\data\\flights")]
)
def test_resolve_rejects_unsafe_paths(unsafe_path: Path) -> None:
    paths = get_project_paths(load_config())

    with pytest.raises(ValueError, match="project-relative"):
        paths.resolve(unsafe_path)

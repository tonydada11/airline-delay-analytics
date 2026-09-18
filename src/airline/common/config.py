"""Project configuration loading and validation."""

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Union

import yaml


PathLike = Union[str, Path]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "project.yaml"


@dataclass(frozen=True)
class ProjectConfig:
    """Validated values loaded from ``conf/project.yaml``."""

    project_root: Path
    raw_csv: Path
    ods_flights: Path


def _relative_path(value: object, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration value 'paths.{key}' must be a non-empty string.")

    path = Path(value.strip())
    if path.is_absolute() or PureWindowsPath(value.strip()).is_absolute():
        raise ValueError(f"Configuration value 'paths.{key}' must be project-relative.")
    if ".." in path.parts:
        raise ValueError(f"Configuration value 'paths.{key}' must not traverse above the project root.")

    return path


def load_config(config_path: PathLike = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Load and validate a project YAML configuration file."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Project configuration file not found: {path}")

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in project configuration: {path}") from exc

    if not isinstance(document, dict):
        raise ValueError("Project configuration must be a YAML mapping.")

    paths = document.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Project configuration must contain a 'paths' mapping.")

    missing = sorted({"raw_csv", "ods_flights"} - paths.keys())
    if missing:
        raise ValueError(f"Project configuration is missing required path keys: {missing}")

    project_root = path.parent.parent
    return ProjectConfig(
        project_root=project_root,
        raw_csv=_relative_path(paths["raw_csv"], "raw_csv"),
        ods_flights=_relative_path(paths["ods_flights"], "ods_flights"),
    )

"""Central project dataset paths."""

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Union

from airline.common.config import ProjectConfig


PathLike = Union[str, Path]


@dataclass(frozen=True)
class ProjectPaths:
    """Portable dataset paths and their project root."""

    project_root: Path
    raw_csv: Path
    ods_flights: Path
    dwd_flights: Path
    dim_date: Path
    dim_carrier: Path
    dim_airport: Path
    dim_route: Path
    dws_carrier_daily: Path
    dws_origin_airport_daily: Path
    dws_route_daily: Path
    dws_ml_flight_features: Path

    def resolve(self, path: PathLike) -> Path:
        """Resolve a configured relative path at a filesystem boundary."""

        relative_path = Path(path)
        if (
            relative_path.is_absolute()
            or PureWindowsPath(str(path)).is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError("Only project-relative paths can be resolved.")
        return self.project_root / relative_path


def get_project_paths(config: ProjectConfig) -> ProjectPaths:
    """Build the centralized path view from validated configuration."""

    return ProjectPaths(
        project_root=config.project_root,
        raw_csv=config.raw_csv,
        ods_flights=config.ods_flights,
        dwd_flights=config.dwd_flights,
        dim_date=config.dim_date,
        dim_carrier=config.dim_carrier,
        dim_airport=config.dim_airport,
        dim_route=config.dim_route,
        dws_carrier_daily=config.dws_carrier_daily,
        dws_origin_airport_daily=config.dws_origin_airport_daily,
        dws_route_daily=config.dws_route_daily,
        dws_ml_flight_features=config.dws_ml_flight_features,
    )

"""Project configuration loading.

All periods, grid sizes, thresholds, seeds and paths live in ``config/project.yaml``;
AOIs live in ``config/aois.geojson``. Code reads them through this module instead of
hard-coding values (see CLAUDE.md, "Code conventions").
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ROOT_ENV_VAR = "DESERT_PROJECT_ROOT"
CONFIG_RELPATH = Path("config") / "project.yaml"
REQUIRED_SECTIONS = ("project", "periods", "grid", "sentinel2", "quota", "labels", "paths")
REQUIRED_PERIODS = ("study", "early", "gap", "late")


class ConfigError(ValueError):
    """Raised when the project configuration is missing or inconsistent."""


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root (the directory containing ``config/project.yaml``).

    Resolution order: the ``DESERT_PROJECT_ROOT`` environment variable, then the
    current working directory and its parents, then this file's location (works for
    an editable install of the ``src/`` layout).

    Parameters
    ----------
    start : Path, optional
        Directory to start the upward search from. Defaults to the current working
        directory.

    Returns
    -------
    Path
        Absolute path to the repository root.
    """
    env_root = os.environ.get(ROOT_ENV_VAR)
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not (root / CONFIG_RELPATH).is_file():
            raise ConfigError(f"{ROOT_ENV_VAR}={root} does not contain {CONFIG_RELPATH}")
        return root

    candidates = [Path(start or Path.cwd()).resolve()]
    candidates += list(candidates[0].parents)
    candidates.append(Path(__file__).resolve().parents[2])
    for candidate in candidates:
        if (candidate / CONFIG_RELPATH).is_file():
            return candidate
    raise ConfigError(
        f"Could not find {CONFIG_RELPATH}. Run from inside the repository or set {ROOT_ENV_VAR}."
    )


def _parse_date(value: Any, where: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ConfigError(f"{where}: {value!r} is not an ISO date (YYYY-MM-DD)") from exc


def validate_config(cfg: dict[str, Any]) -> None:
    """Check that the configuration has the required sections and is internally consistent.

    Parameters
    ----------
    cfg : dict
        Parsed ``project.yaml`` contents.

    Raises
    ------
    ConfigError
        If a section is missing, a period is malformed or out of order, or the grid is
        inconsistent.
    """
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"project.yaml is missing sections: {missing}")

    periods = cfg["periods"]
    bounds: dict[str, tuple[date, date]] = {}
    for name in REQUIRED_PERIODS:
        if name not in periods:
            raise ConfigError(f"periods.{name} is missing")
        start = _parse_date(periods[name].get("start"), f"periods.{name}.start")
        end = _parse_date(periods[name].get("end"), f"periods.{name}.end")
        if start > end:
            raise ConfigError(f"periods.{name}: start {start} is after end {end}")
        bounds[name] = (start, end)

    # Feature window must end before the gap, and the gap before the label window
    # (no overlap -> no label leakage, ROADMAP §4.2/§4.5).
    if not bounds["early"][1] < bounds["gap"][0] <= bounds["gap"][1] < bounds["late"][0]:
        raise ConfigError("periods must be ordered early < gap < late without overlap")
    for name in ("early", "gap", "late"):
        if not (bounds["study"][0] <= bounds[name][0] and bounds[name][1] <= bounds["study"][1]):
            raise ConfigError(f"periods.{name} must lie inside periods.study")

    grid = cfg["grid"]
    res, cell = grid.get("pixel_resolution_m"), grid.get("cell_size_m")
    numbers = all(isinstance(v, int | float) and v > 0 for v in (res, cell))
    if not numbers:
        raise ConfigError("grid.pixel_resolution_m and grid.cell_size_m must be positive numbers")
    if cell % res != 0:
        raise ConfigError(
            f"grid.cell_size_m ({cell}) must be a whole multiple of pixel_resolution_m ({res})"
        )

    caps = cfg["quota"].get("pu_caps", {})
    if sum(caps.values()) > cfg["quota"].get("pu_total", 0):
        raise ConfigError("quota.pu_caps add up to more than quota.pu_total")


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate ``config/project.yaml``.

    Parameters
    ----------
    path : Path or str, optional
        Explicit path to a YAML config. Defaults to ``<project root>/config/project.yaml``.

    Returns
    -------
    dict
        The parsed configuration, plus ``cfg["_root"]``: the absolute project root
        (``Path``) used to resolve relative paths.
    """
    if path is None:
        root = find_project_root()
        path = root / CONFIG_RELPATH
    else:
        path = Path(path).resolve()
        root = path.parent.parent
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    validate_config(cfg)
    cfg["_root"] = root
    logger.debug("Loaded config from %s", path)
    return cfg


def get_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a ``paths.<key>`` entry to an absolute path under the project root.

    Parameters
    ----------
    cfg : dict
        Configuration returned by :func:`load_config`.
    key : str
        Key in the ``paths`` section, e.g. ``"data_raw"``.

    Returns
    -------
    Path
        Absolute path (not created).
    """
    try:
        rel = cfg["paths"][key]
    except KeyError as exc:
        raise ConfigError(f"paths.{key} is not defined in project.yaml") from exc
    return (Path(cfg["_root"]) / rel).resolve()


def load_aois(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the AOI GeoJSON FeatureCollection named by ``paths.aois``.

    Parameters
    ----------
    cfg : dict
        Configuration returned by :func:`load_config`.

    Returns
    -------
    dict
        GeoJSON FeatureCollection (WGS84). Empty until task M1-T02 fills it.
    """
    with open(get_path(cfg, "aois"), encoding="utf-8") as fh:
        aois = json.load(fh)
    if aois.get("type") != "FeatureCollection":
        raise ConfigError("aois.geojson must be a GeoJSON FeatureCollection")
    return aois


def load_env(cfg: dict[str, Any] | None = None) -> bool:
    """Load variables from ``<project root>/.env`` into the process environment.

    Existing environment variables are not overridden. Secret values are never
    returned or logged.

    Returns
    -------
    bool
        True if a ``.env`` file was found and loaded.
    """
    root = Path(cfg["_root"]) if cfg else find_project_root()
    env_file = root / ".env"
    if not env_file.is_file():
        logger.info("No .env file at %s; relying on the existing environment", root)
        return False
    return load_dotenv(env_file, override=False)

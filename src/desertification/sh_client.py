"""Sentinel Hub client setup for the Copernicus Data Space Ecosystem (CDSE).

Credentials are read only from the ``SH_CLIENT_ID`` / ``SH_CLIENT_SECRET`` environment
variables (loaded from the gitignored ``.env`` by :func:`desertification.config.load_env`).
Endpoints come from the ``sentinel_hub`` section of ``config/project.yaml``. Nothing in this
module prints or logs secret values.

Checked against sentinelhub-py 3.12.1: ``SHConfig`` fields ``sh_client_id``,
``sh_client_secret``, ``sh_base_url``, ``sh_token_url``; ``DataCollection.define_from``;
``SentinelHubCatalog.search`` (Catalog requests go to ``{sh_base_url}/api/v1/catalog/1.0.0``).
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from desertification.cache import load_cached_json, save_json_with_sidecar

if TYPE_CHECKING:
    from sentinelhub import BBox, DataCollection, SHConfig

logger = logging.getLogger(__name__)

CLIENT_ID_VAR = "SH_CLIENT_ID"
CLIENT_SECRET_VAR = "SH_CLIENT_SECRET"
CATALOG_PAGE_LIMIT = 100

# Sentinel-2 product ids contain the MGRS tile as "_T<2 digits><3 letters>_",
# e.g. S2A_MSIL2A_20230105T112451_N0509_R037_T28PEB_20230105T150510.SAFE -> 28PEB.
_TILE_RE = re.compile(r"_T(\d{2}[A-Z]{3})_")


class CredentialsError(RuntimeError):
    """Raised when Sentinel Hub credentials are missing. Never contains secret values."""


def missing_credentials(env: dict[str, str] | None = None) -> list[str]:
    """Return the names of credential variables that are unset or empty.

    Parameters
    ----------
    env : dict, optional
        Mapping to check; defaults to ``os.environ``.

    Returns
    -------
    list of str
        Missing variable names (empty list when both are set).
    """
    env = os.environ if env is None else env
    return [name for name in (CLIENT_ID_VAR, CLIENT_SECRET_VAR) if not env.get(name)]


def make_sh_config(cfg: dict[str, Any], env: dict[str, str] | None = None) -> SHConfig:
    """Build a ``SHConfig`` pointed at CDSE using credentials from the environment.

    Parameters
    ----------
    cfg : dict
        Project configuration from :func:`desertification.config.load_config`.
    env : dict, optional
        Mapping holding the credentials; defaults to ``os.environ``.

    Returns
    -------
    sentinelhub.SHConfig
        Config with CDSE base and token URLs. Its ``repr`` masks the credentials.

    Raises
    ------
    CredentialsError
        If ``SH_CLIENT_ID`` or ``SH_CLIENT_SECRET`` is missing.
    """
    from sentinelhub import SHConfig

    env = os.environ if env is None else env
    missing = missing_credentials(env)
    if missing:
        raise CredentialsError(
            f"Missing {', '.join(missing)}. Copy .env.example to .env, fill in your CDSE "
            "OAuth client, and call desertification.config.load_env()."
        )
    sh = cfg["sentinel_hub"]
    config = SHConfig()
    config.sh_client_id = env[CLIENT_ID_VAR]
    config.sh_client_secret = env[CLIENT_SECRET_VAR]
    config.sh_base_url = sh["base_url"]
    config.sh_token_url = sh["token_url"]
    return config


@cache
def _define_collection(name: str, service_url: str) -> DataCollection:
    from sentinelhub import DataCollection

    return DataCollection.SENTINEL2_L2A.define_from(name, service_url=service_url)


def get_s2l2a_collection(cfg: dict[str, Any]) -> DataCollection:
    """Return the Sentinel-2 L2A collection bound to the CDSE service URL.

    Defined once per process (``define_from`` registers the name globally).

    Parameters
    ----------
    cfg : dict
        Project configuration.

    Returns
    -------
    sentinelhub.DataCollection
    """
    sh = cfg["sentinel_hub"]
    return _define_collection(sh["collection_name"], sh["base_url"])


def validate_bbox_wgs84(bbox: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    """Check a ``[lon_min, lat_min, lon_max, lat_max]`` box in degrees (WGS84).

    Parameters
    ----------
    bbox : sequence of 4 floats
        Longitudes in [-180, 180], latitudes in [-90, 90], min < max.

    Returns
    -------
    tuple of 4 floats

    Raises
    ------
    ValueError
        If the box is malformed.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(bbox)}")
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
    if not (-180 <= lon_min < lon_max <= 180):
        raise ValueError(f"bad longitudes: {lon_min}, {lon_max}")
    if not (-90 <= lat_min < lat_max <= 90):
        raise ValueError(f"bad latitudes: {lat_min}, {lat_max}")
    return lon_min, lat_min, lon_max, lat_max


def make_bbox(bbox_wgs84: list[float] | tuple[float, ...]) -> BBox:
    """Build a sentinelhub ``BBox`` in WGS84 from ``[lon_min, lat_min, lon_max, lat_max]``."""
    from sentinelhub import CRS, BBox

    return BBox(validate_bbox_wgs84(bbox_wgs84), crs=CRS.WGS84)


def parse_tile_id(product_id: str) -> str | None:
    """Extract the MGRS tile (e.g. ``"28PEB"``) from a Sentinel-2 product id, or None."""
    match = _TILE_RE.search(product_id)
    return match.group(1) if match else None


def catalog_request_count(n_features: int, limit: int = CATALOG_PAGE_LIMIT) -> int:
    """Number of Catalog API page requests needed to return ``n_features`` results.

    A search always makes at least one request, even when it finds nothing.
    """
    return max(1, math.ceil(n_features / limit))


def summarize_scenes(features: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn Catalog API features into a tidy table of scenes.

    Parameters
    ----------
    features : list of dict
        STAC items as returned by ``SentinelHubCatalog.search``.

    Returns
    -------
    pandas.DataFrame
        One row per distinct product id, sorted by time, with columns ``id``
        (str), ``datetime`` (UTC timestamp), ``date`` (date), ``tile_id`` (str),
        ``platform`` (str) and ``cloud_cover_pct`` (float, 0–100).
    """
    columns = ["id", "datetime", "date", "tile_id", "platform", "cloud_cover_pct"]
    rows = []
    for feature in features:
        props = feature.get("properties", {})
        product_id = feature.get("id", "")
        rows.append(
            {
                "id": product_id,
                "datetime": props.get("datetime"),
                "tile_id": parse_tile_id(product_id),
                "platform": props.get("platform") or product_id[:3] or None,
                "cloud_cover_pct": props.get("eo:cloud_cover"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows).drop_duplicates(subset="id")
    # Catalog timestamps mix "…:56Z" and "…:56.123Z"; ISO8601 parses both (inference fails).
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")
    df["date"] = df["datetime"].dt.date
    df["cloud_cover_pct"] = pd.to_numeric(df["cloud_cover_pct"], errors="coerce")
    return df[columns].sort_values("datetime").reset_index(drop=True)


def search_catalog(
    sh_config: SHConfig,
    collection: DataCollection,
    bbox_wgs84: list[float] | tuple[float, ...],
    start: str,
    end: str,
) -> list[dict[str, Any]]:
    """Run a Catalog API search (makes network requests; costs quota).

    Parameters
    ----------
    sh_config : sentinelhub.SHConfig
        From :func:`make_sh_config`.
    collection : sentinelhub.DataCollection
        From :func:`get_s2l2a_collection`.
    bbox_wgs84 : sequence of 4 floats
        ``[lon_min, lat_min, lon_max, lat_max]`` in degrees.
    start, end : str
        ISO dates, inclusive.

    Returns
    -------
    list of dict
        STAC features.
    """
    from sentinelhub import SentinelHubCatalog

    catalog = SentinelHubCatalog(config=sh_config)
    results = catalog.search(
        collection,
        bbox=make_bbox(bbox_wgs84),
        time=(start, end),
        limit=CATALOG_PAGE_LIMIT,
    )
    return list(results)


def cached_catalog_search(
    cfg: dict[str, Any],
    name: str,
    bbox_wgs84: list[float] | tuple[float, ...],
    start: str,
    end: str,
    allow_api: bool = False,
    search_fn: Callable[[list[float] | tuple[float, ...], str, str], list[dict[str, Any]]]
    | None = None,
) -> tuple[list[dict[str, Any]] | None, Path, bool]:
    """Return Catalog search results from the cache, or from the API if allowed.

    Parameters
    ----------
    cfg : dict
        Project configuration (gives the cache folder and endpoints).
    name : str
        Short label used in the cache file name.
    bbox_wgs84 : sequence of 4 floats
        ``[lon_min, lat_min, lon_max, lat_max]`` in degrees.
    start, end : str
        ISO dates, inclusive.
    allow_api : bool, default False
        If False, never contact the API: a cache miss returns ``None``.
    search_fn : callable, optional
        ``search_fn(bbox, start, end) -> features``. Defaults to a real CDSE search built
        from the environment's credentials (injectable for tests).

    Returns
    -------
    features : list of dict or None
        STAC features, or None when not cached and ``allow_api`` is False.
    path : Path
        Cache file location.
    called_api : bool
        True if this call contacted the API.
    """
    from desertification.config import get_path

    bbox = list(validate_bbox_wgs84(bbox_wgs84))
    params = {
        "api": "catalog",
        "base_url": cfg["sentinel_hub"]["base_url"],
        "collection": "sentinel-2-l2a",
        "bbox_wgs84": bbox,
        "time": [start, end],
        "limit": CATALOG_PAGE_LIMIT,
    }
    path = get_path(cfg, "catalog_cache") / f"{name}_{start}_{end}.json"
    cached = load_cached_json(path, params)
    if cached is not None:
        logger.info("Catalog results loaded from cache %s", path)
        return cached, path, False
    if not allow_api:
        logger.info("No cached catalog results at %s and API calls are not allowed", path)
        return None, path, False

    if search_fn is None:

        def search_fn(b, s, e):  # noqa: ANN001 - simple default closure
            return search_catalog(make_sh_config(cfg), get_s2l2a_collection(cfg), b, s, e)

    features = search_fn(bbox, start, end)
    save_json_with_sidecar(
        path,
        features,
        params,
        extra={
            "n_features": len(features),
            "catalog_requests": catalog_request_count(len(features)),
        },
    )
    logger.info("Catalog search returned %d features; cached at %s", len(features), path)
    return features, path, True

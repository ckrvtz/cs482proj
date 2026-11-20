"""Auxiliary (non-Sentinel) data: monthly rainfall and reference evapotranspiration.

Sources (open, no account, read straight from the Climate Hazards Center server):

* CHIRPS v3 monthly rainfall, Cloud-Optimized GeoTIFFs, 0.05°, mm/month.
* Hobbins reference evapotranspiration (ETo), monthly GeoTIFFs, 0.05°, mm/month.

Only the pixels around each AOI are read (HTTP range requests through GDAL ``/vsicurl/``),
so nothing large is downloaded. Results are cached as JSON with a provenance sidecar.
URLs come from ``config/project.yaml`` (``climate_data``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from desertification.cache import load_cached_json, save_json_with_sidecar

logger = logging.getLogger(__name__)

# GDAL settings for efficient remote reads of single files.
GDAL_REMOTE_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".cog,.tif,.tiff",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


def window_for_bounds(
    transform: Any, bounds: Iterable[float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Pixel window covering every pixel that intersects ``bounds``.

    Parameters
    ----------
    transform : affine.Affine
        Raster geotransform (north-up, no rotation).
    bounds : (xmin, ymin, xmax, ymax)
        In the raster's CRS.
    width, height : int
        Raster size in pixels, used to clip the window.

    Returns
    -------
    (row_start, row_stop, col_start, col_stop) or None
        Half-open pixel ranges, or None if the bounds miss the raster.
    """
    xmin, ymin, xmax, ymax = bounds
    inv = ~transform
    c0, r0 = inv @ (xmin, ymax)
    c1, r1 = inv @ (xmax, ymin)
    col_start, col_stop = max(0, math.floor(c0)), min(width, math.ceil(c1))
    row_start, row_stop = max(0, math.floor(r0)), min(height, math.ceil(r1))
    if col_start >= col_stop or row_start >= row_stop:
        return None
    return row_start, row_stop, col_start, col_stop


def masked_mean(values: np.ndarray, nodata_below: float) -> float:
    """Mean of values >= ``nodata_below`` (NaN if none are valid)."""
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr) & (arr >= nodata_below)]
    return float(valid.mean()) if valid.size else float("nan")


def read_window_means(
    url: str, bboxes: dict[str, list[float]], nodata_below: float
) -> dict[str, float]:
    """Open one remote raster and return the mean value inside each WGS84 bbox.

    Parameters
    ----------
    url : str
        HTTP(S) URL of a GeoTIFF/COG in EPSG:4326.
    bboxes : dict
        ``aoi_id -> [lon_min, lat_min, lon_max, lat_max]``.
    nodata_below : float
        Values below this are treated as missing (e.g. -9999 over the ocean).

    Returns
    -------
    dict
        ``aoi_id -> mean`` (NaN where nothing valid).
    """
    import rasterio
    from rasterio.windows import Window

    out = {}
    with rasterio.Env(**GDAL_REMOTE_ENV), rasterio.open(f"/vsicurl/{url}") as ds:
        for aoi_id, bbox in bboxes.items():
            win = window_for_bounds(ds.transform, bbox, ds.width, ds.height)
            if win is None:
                out[aoi_id] = float("nan")
                continue
            r0, r1, c0, c1 = win
            data = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
            out[aoi_id] = masked_mean(data, nodata_below)
    return out


def month_list(first_year: int, last_year: int) -> list[tuple[int, int]]:
    """All ``(year, month)`` pairs from January of ``first_year`` to December of ``last_year``."""
    if last_year < first_year:
        raise ValueError("last_year must be >= first_year")
    return [(y, m) for y in range(first_year, last_year + 1) for m in range(1, 13)]


def monthly_series(
    cfg: dict[str, Any],
    source: str,
    bboxes: dict[str, list[float]],
    first_year: int,
    last_year: int,
    allow_download: bool = False,
    max_workers: int = 8,
) -> pd.DataFrame | None:
    """Monthly AOI-mean values of a climate source, from cache or the CHC server.

    Parameters
    ----------
    cfg : dict
        Project configuration.
    source : {"chirps", "eto"}
        CHIRPS v3 rainfall or Hobbins reference ET (both mm/month).
    bboxes : dict
        ``aoi_id -> [lon_min, lat_min, lon_max, lat_max]`` (WGS84).
    first_year, last_year : int
        Inclusive year range.
    allow_download : bool, default False
        If False, return None on a cache miss instead of reading remote files.
    max_workers : int, default 8
        Parallel file reads.

    Returns
    -------
    pandas.DataFrame or None
        Long table: ``aoi_id``, ``year``, ``month``, ``value`` (mm/month).
    """
    from desertification.config import get_path

    templates = {
        "chirps": cfg["climate_data"]["chirps_monthly_url"],
        "eto": cfg["climate_data"]["eto_monthly_url"],
    }
    if source not in templates:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(templates)}")
    nodata_below = cfg["climate_data"]["nodata_below"]
    params = {
        "source": source,
        "url_template": templates[source],
        "bboxes": {k: list(map(float, v)) for k, v in sorted(bboxes.items())},
        "years": [first_year, last_year],
        "nodata_below": nodata_below,
    }
    path = get_path(cfg, "aux_cache") / source / f"monthly_{first_year}_{last_year}.json"
    cached = load_cached_json(path, params)
    if cached is not None:
        return pd.DataFrame(cached)
    if not allow_download:
        logger.info("No cached %s series for these AOIs; downloads not allowed", source)
        return None

    months = month_list(first_year, last_year)
    template = templates[source]

    def fetch(ym: tuple[int, int]) -> list[dict[str, Any]]:
        year, month = ym
        url = template.format(year=year, month=month)
        try:
            means = read_window_means(url, params["bboxes"], nodata_below)
        except Exception as exc:  # noqa: BLE001 - one bad month must not stop the rest
            logger.warning("Could not read %s: %s", url, exc)
            means = dict.fromkeys(params["bboxes"], float("nan"))
        return [{"aoi_id": k, "year": year, "month": month, "value": v} for k, v in means.items()]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rows = [row for batch in pool.map(fetch, months) for row in batch]
    df = pd.DataFrame(rows).sort_values(["aoi_id", "year", "month"]).reset_index(drop=True)
    n_missing = int(df["value"].isna().sum())
    save_json_with_sidecar(
        path,
        df.to_dict(orient="records"),
        params,
        extra={"n_files": len(months), "n_missing_values": n_missing},
    )
    logger.info("Cached %d %s values (%d missing) at %s", len(df), source, n_missing, path)
    return df


def rainfall_overview(
    cfg: dict[str, Any],
    bounds: list[float],
    first_year: int,
    last_year: int,
    decimation: int = 8,
    allow_download: bool = False,
) -> tuple[np.ndarray, list[float]] | None:
    """Mean annual CHIRPS rainfall over a region at reduced resolution (for maps).

    Reads COG overviews, so each month costs a small download.

    Parameters
    ----------
    cfg : dict
        Project configuration.
    bounds : [lon_min, lat_min, lon_max, lat_max]
        Region in degrees.
    first_year, last_year : int
        Inclusive years to average.
    decimation : int, default 8
        Output pixel = ``decimation`` × 0.05°.
    allow_download : bool, default False
        If False, return None on a cache miss.

    Returns
    -------
    (numpy.ndarray, list) or None
        Mean annual rainfall in mm/year (NaN over ocean), shape (rows, cols), and the
        ``[lon_min, lon_max, lat_min, lat_max]`` extent for ``matplotlib.imshow``.
    """
    import rasterio
    from rasterio.windows import Window

    from desertification.cache import cached_file_valid, write_sidecar
    from desertification.config import get_path

    template = cfg["climate_data"]["chirps_monthly_url"]
    nodata_below = cfg["climate_data"]["nodata_below"]
    params = {
        "source": "chirps_overview",
        "url_template": template,
        "bounds": list(map(float, bounds)),
        "years": [first_year, last_year],
        "decimation": decimation,
    }
    path = get_path(cfg, "aux_cache") / "chirps" / f"overview_{first_year}_{last_year}.npz"
    if cached_file_valid(path, params):
        with np.load(path) as npz:
            return npz["rain"], npz["extent"].tolist()
    if not allow_download:
        return None

    def read(ym: tuple[int, int]) -> tuple[np.ndarray, list[float]]:
        url = template.format(year=ym[0], month=ym[1])
        with rasterio.Env(**GDAL_REMOTE_ENV), rasterio.open(f"/vsicurl/{url}") as ds:
            r0, r1, c0, c1 = window_for_bounds(ds.transform, bounds, ds.width, ds.height)
            shape = (max(1, (r1 - r0) // decimation), max(1, (c1 - c0) // decimation))
            data = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), out_shape=shape)
            left, top = ds.transform @ (c0, r0)
            right, bottom = ds.transform @ (c1, r1)
        data = np.where(data >= nodata_below, data, np.nan).astype("float32")
        return data, [left, right, bottom, top]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(read, month_list(first_year, last_year)))
    n_years = last_year - first_year + 1
    rain = np.sum([r[0] for r in results], axis=0) / n_years
    extent = results[0][1]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rain=rain, extent=np.array(extent))
    write_sidecar(path, params)
    return rain, extent

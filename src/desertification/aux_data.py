"""Auxiliary (non-Sentinel) data: monthly rainfall and reference evapotranspiration.

Sources (open, no account, read straight from the Climate Hazards Center server):

* CHIRPS v3 monthly rainfall, Cloud-Optimized GeoTIFFs, 0.05°, mm/month.
* Hobbins reference evapotranspiration (ETo), monthly GeoTIFFs, 0.05°, mm/month.

Only the pixels around each AOI are read (HTTP range requests through GDAL ``/vsicurl/``),
so nothing large is downloaded. Results are cached as JSON with a provenance sidecar.
URLs come from ``config/project.yaml`` (``climate_data``).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from desertification.cache import load_cached_json, save_json_with_sidecar, sidecar_path

logger = logging.getLogger(__name__)

# GDAL settings for efficient remote reads of single files.
GDAL_REMOTE_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".cog,.tif,.tiff",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


class ClimateDownloadError(RuntimeError):
    """Some remote climate files could not be read; nothing was cached as complete."""


def fetch_with_retries(
    items: list[Any],
    fetch: Callable[[Any], Any],
    max_workers: int,
    retry_rounds: int,
    backoff_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[Any, Any], list[Any]]:
    """Run ``fetch`` over ``items`` in parallel, then retry failures with backoff.

    ``fetch`` returns None on failure. Retries run one item at a time; before retry round
    ``r`` (0-based) it waits ``backoff_s * 2**r`` seconds, which gives a rate-limiting
    server (HTTP 429) time to recover.

    Parameters
    ----------
    items : list
        Hashable work items (e.g. ``(year, month)``).
    fetch : callable
        ``fetch(item) -> result or None``.
    max_workers : int
        Threads for the first pass.
    retry_rounds : int
        Extra passes over failed items.
    backoff_s : float
        Base pause in seconds.
    sleep : callable, default time.sleep
        Injectable for tests.

    Returns
    -------
    (dict, list)
        ``item -> result`` for successes, and the items that still failed.
    """
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        results = dict(zip(items, pool.map(fetch, items), strict=True))
    for round_no in range(retry_rounds):
        failed = [item for item, res in results.items() if res is None]
        if not failed:
            break
        wait = backoff_s * 2**round_no
        logger.warning(
            "%d file(s) failed; retry round %d in %.0f s", len(failed), round_no + 1, wait
        )
        sleep(wait)
        for item in failed:
            results[item] = fetch(item)
    done = {k: v for k, v in results.items() if v is not None}
    return done, [k for k, v in results.items() if v is None]


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


def clean_overview(data: np.ndarray, fill_below: float = -1.0) -> np.ndarray:
    """Clean a rainfall array read from COG overviews.

    Overview resampling leaves small negative values (about -0.04 mm) in dry areas and
    spreads the -9999 ocean fill into large negatives along coasts. Values below
    ``fill_below`` become NaN; the remaining negatives are set to 0.

    Parameters
    ----------
    data : numpy.ndarray
        Rainfall in mm read from overviews.
    fill_below : float, default -1.0
        Threshold (mm) under which a value is treated as fill.

    Returns
    -------
    numpy.ndarray
        float32 array, NaN where fill.
    """
    arr = np.asarray(data, dtype="float32")
    return np.where(arr >= fill_below, np.clip(arr, 0, None), np.nan).astype("float32")


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
    max_workers: int | None = None,
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
    max_workers : int, optional
        Parallel file reads; defaults to ``climate_data.max_workers``.

    Returns
    -------
    pandas.DataFrame or None
        Long table: ``aoi_id``, ``year``, ``month``, ``value`` (mm/month).

    Raises
    ------
    ClimateDownloadError
        If some files still fail after the retry rounds. Months already read are kept in
        ``<cache>.partial.json`` and only the missing ones are fetched on the next call.

    Notes
    -----
    A cache whose sidecar records missing values (written by an older version of this
    function) is ignored and rebuilt.
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
        meta = json.loads(sidecar_path(path).read_text(encoding="utf-8"))
        if meta.get("n_missing_values", 0) == 0:
            return pd.DataFrame(cached)
        logger.warning("Cached %s series has missing values; rebuilding it", source)
    if not allow_download:
        logger.info("No complete cached %s series for these AOIs; downloads not allowed", source)
        return None

    climate = cfg["climate_data"]
    months = month_list(first_year, last_year)
    template = templates[source]

    # Resume from earlier partial runs (same parameters only).
    partial_path = path.with_name(path.stem + ".partial.json")
    partial = load_cached_json(partial_path, params) or {}
    done = {tuple(map(int, key.split("-"))): means for key, means in partial.items()}
    todo = [ym for ym in months if ym not in done]

    def fetch(ym: tuple[int, int]) -> dict[str, float] | None:
        url = template.format(year=ym[0], month=ym[1])
        try:
            return read_window_means(url, params["bboxes"], nodata_below)
        except Exception as exc:  # noqa: BLE001 - one bad month must not stop the rest
            logger.warning("Could not read %s: %s", url, exc)
            return None

    fetched, failed = fetch_with_retries(
        todo,
        fetch,
        max_workers=max_workers or climate.get("max_workers", 2),
        retry_rounds=climate.get("retry_rounds", 3),
        backoff_s=climate.get("retry_backoff_s", 20),
    )
    done.update(fetched)
    if failed:
        save_json_with_sidecar(
            partial_path, {f"{y}-{m:02d}": v for (y, m), v in sorted(done.items())}, params
        )
        examples = ", ".join(f"{y}-{m:02d}" for y, m in failed[:5])
        raise ClimateDownloadError(
            f"{len(failed)} of {len(months)} {source} files could not be read (e.g. {examples}); "
            f"the server may be rate-limiting (HTTP 429). {len(done)} months are saved; run "
            "again later and only the missing months will be fetched."
        )

    rows = []
    for (year, month), means in sorted(done.items()):
        rows += [{"aoi_id": k, "year": year, "month": month, "value": v} for k, v in means.items()]
    df = pd.DataFrame(rows).sort_values(["aoi_id", "year", "month"]).reset_index(drop=True)
    n_missing = int(df["value"].isna().sum())  # NaN here = no valid pixel (e.g. ocean)
    save_json_with_sidecar(
        path,
        df.to_dict(orient="records"),
        params,
        extra={"n_files": len(months), "n_missing_values": n_missing},
    )
    # The partial file was only a resume aid for this cache; the full cache supersedes it.
    for leftover in (partial_path, sidecar_path(partial_path)):
        leftover.unlink(missing_ok=True)
    logger.info("Cached %d %s values at %s", len(df), source, path)
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
    params = {
        "source": "chirps_overview",
        "url_template": template,
        "bounds": list(map(float, bounds)),
        "years": [first_year, last_year],
        "decimation": decimation,
        "cleaning": "clean_overview(fill_below=-1)",
    }
    path = get_path(cfg, "aux_cache") / "chirps" / f"overview_{first_year}_{last_year}.npz"
    if cached_file_valid(path, params):
        with np.load(path) as npz:
            return npz["rain"], npz["extent"].tolist()
    if not allow_download:
        return None

    def read(ym: tuple[int, int]) -> tuple[np.ndarray, list[float]] | None:
        url = template.format(year=ym[0], month=ym[1])
        try:
            with rasterio.Env(**GDAL_REMOTE_ENV), rasterio.open(f"/vsicurl/{url}") as ds:
                r0, r1, c0, c1 = window_for_bounds(ds.transform, bounds, ds.width, ds.height)
                shape = (max(1, (r1 - r0) // decimation), max(1, (c1 - c0) // decimation))
                data = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), out_shape=shape)
                left, top = ds.transform @ (c0, r0)
                right, bottom = ds.transform @ (c1, r1)
        except Exception as exc:  # noqa: BLE001 - retried below
            logger.warning("Could not read %s: %s", url, exc)
            return None
        return clean_overview(data), [left, right, bottom, top]

    climate = cfg["climate_data"]
    months = month_list(first_year, last_year)
    results, failed = fetch_with_retries(
        months,
        read,
        max_workers=climate.get("max_workers", 2),
        retry_rounds=climate.get("retry_rounds", 3),
        backoff_s=climate.get("retry_backoff_s", 20),
    )
    if failed:
        raise ClimateDownloadError(
            f"{len(failed)} of {len(months)} CHIRPS overview files could not be read; the "
            "server may be rate-limiting (HTTP 429). Run again later."
        )
    n_years = last_year - first_year + 1
    rain = np.sum([results[ym][0] for ym in months], axis=0) / n_years
    extent = results[months[0]][1]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rain=rain, extent=np.array(extent))
    write_sidecar(path, params)
    return rain, extent

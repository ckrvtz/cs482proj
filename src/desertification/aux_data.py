"""Auxiliary (non-Sentinel) data: rainfall, evapotranspiration, DEM and land cover.

Sources (open, no account, read straight from the Climate Hazards Center server):

* CHIRPS v3 monthly rainfall, Cloud-Optimized GeoTIFFs, 0.05°, mm/month.
* Hobbins reference evapotranspiration (ETo), monthly GeoTIFFs, 0.05°, mm/month.

Only the pixels around each AOI are read (HTTP range requests through GDAL ``/vsicurl/``),
so nothing large is downloaded. Results are cached as JSON with a provenance sidecar.
URLs come from ``config/project.yaml`` (``climate_data``).

M1-T07 adds per-AOI layers on the Sentinel-2 extraction grid: Copernicus DEM GLO-30
(elevation, slope, aspect) and ESA WorldCover 2020/2021 from Microsoft Planetary Computer,
and CHIRPS monthly grids kept at native 0.05° and sampled per 100 m cell.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from desertification.cache import (
    cached_file_valid,
    load_cached_json,
    request_hash,
    save_json_with_sidecar,
    sidecar_path,
    write_sidecar,
)

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
    Each AOI is cached in its own file (name includes a hash of its box), so moving or
    adding an AOI only downloads that AOI. Rows from the older single-file cache are reused
    for AOIs whose box is unchanged. A failed download keeps finished months in a
    ``.partial-<hash>.json`` file and resumes from it.
    """
    from desertification.config import get_path

    templates = {
        "chirps": cfg["climate_data"]["chirps_monthly_url"],
        "eto": cfg["climate_data"]["eto_monthly_url"],
    }
    if source not in templates:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(templates)}")
    climate = cfg["climate_data"]
    nodata_below = climate["nodata_below"]
    template = templates[source]
    base = {
        "source": source,
        "url_template": template,
        "years": [first_year, last_year],
        "nodata_below": nodata_below,
    }
    cache_dir = get_path(cfg, "aux_cache") / source
    boxes = {k: [float(v) for v in b] for k, b in sorted(bboxes.items())}

    # 1. Per-AOI caches (a moved or new AOI does not invalidate the others).
    frames: dict[str, pd.DataFrame] = {}
    for aoi_id, bbox in boxes.items():
        path, params = aoi_cache_entry(cache_dir, base, aoi_id, bbox, first_year, last_year)
        cached = load_cached_json(path, params)
        if cached is not None:
            frames[aoi_id] = pd.DataFrame(cached)
    # 2. Rows from the older all-AOI cache file, where the AOI's box is unchanged.
    missing = {k: b for k, b in boxes.items() if k not in frames}
    for aoi_id, df in legacy_rows(cache_dir, base, missing, first_year, last_year).items():
        path, params = aoi_cache_entry(
            cache_dir, base, aoi_id, missing[aoi_id], first_year, last_year
        )
        save_json_with_sidecar(path, df.to_dict(orient="records"), params, extra={"migrated": True})
        frames[aoi_id] = df
    missing = {k: b for k, b in boxes.items() if k not in frames}
    if not missing:
        return _concat(frames)
    if not allow_download:
        logger.info(
            "No complete cached %s series for %s; downloads not allowed", source, sorted(missing)
        )
        return None

    # 3. Download only the missing AOIs, resuming from an earlier partial run.
    group = dict(base, bboxes=missing)
    partial_path = (
        cache_dir / f"monthly_{first_year}_{last_year}.partial-{request_hash(group)[:10]}.json"
    )
    partial = load_cached_json(partial_path, group) or {}
    done = {tuple(map(int, key.split("-"))): means for key, means in partial.items()}
    months = month_list(first_year, last_year)
    todo = [ym for ym in months if ym not in done]

    def fetch(ym: tuple[int, int]) -> dict[str, float] | None:
        url = template.format(year=ym[0], month=ym[1])
        try:
            return read_window_means(url, missing, nodata_below)
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
            partial_path, {f"{y}-{m:02d}": v for (y, m), v in sorted(done.items())}, group
        )
        examples = ", ".join(f"{y}-{m:02d}" for y, m in failed[:5])
        raise ClimateDownloadError(
            f"{len(failed)} of {len(months)} {source} files could not be read (e.g. {examples}); "
            f"the server may be rate-limiting (HTTP 429). {len(done)} months are saved; run "
            "again later and only the missing months will be fetched."
        )

    for aoi_id, bbox in missing.items():
        rows = [
            {"aoi_id": aoi_id, "year": y, "month": m, "value": means[aoi_id]}
            for (y, m), means in sorted(done.items())
        ]
        df = pd.DataFrame(rows)
        path, params = aoi_cache_entry(cache_dir, base, aoi_id, bbox, first_year, last_year)
        save_json_with_sidecar(path, rows, params, extra={"n_files": len(months)})
        frames[aoi_id] = df
    # The partial file was only a resume aid; the per-AOI caches supersede it.
    for leftover in (partial_path, sidecar_path(partial_path)):
        leftover.unlink(missing_ok=True)
    logger.info("Cached %s series for %s", source, sorted(missing))
    return _concat(frames)


def aoi_cache_entry(
    cache_dir: Path,
    base: dict[str, Any],
    aoi_id: str,
    bbox: list[float],
    first_year: int,
    last_year: int,
) -> tuple[Path, dict[str, Any]]:
    """Cache path and request parameters for one AOI's monthly series.

    The file name carries a short hash of the parameters, so a moved AOI gets a new file
    instead of a name clash with the old one (also when caches are copied between machines).
    """
    params = dict(base, aoi_id=aoi_id, bbox=[float(v) for v in bbox])
    name = f"monthly_{first_year}_{last_year}_{aoi_id}_{request_hash(params)[:10]}.json"
    return cache_dir / name, params


def legacy_rows(
    cache_dir: Path,
    base: dict[str, Any],
    boxes: dict[str, list[float]],
    first_year: int,
    last_year: int,
) -> dict[str, pd.DataFrame]:
    """Reuse rows from the older single-file cache (``monthly_<y0>_<y1>.json``).

    Rows are taken only for AOIs whose box is unchanged, and only if that file was
    complete and made with the same source, URL template, years and nodata threshold.
    """
    path = cache_dir / f"monthly_{first_year}_{last_year}.json"
    meta_path = sidecar_path(path)
    if not boxes or not (path.is_file() and meta_path.is_file()):
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    request = meta.get("request", {})
    if meta.get("n_missing_values", 1) != 0 or any(request.get(k) != v for k, v in base.items()):
        return {}
    old_boxes = request.get("bboxes", {})
    reusable = [k for k, b in boxes.items() if old_boxes.get(k) == [float(v) for v in b]]
    if not reusable:
        return {}
    df = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return {k: df[df["aoi_id"] == k].reset_index(drop=True) for k in reusable}


def _concat(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return (
        pd.concat([frames[k] for k in sorted(frames)], ignore_index=True)
        .sort_values(["aoi_id", "year", "month"])
        .reset_index(drop=True)
    )


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

    # Resume from an earlier partial run (same parameters only).
    partial_path = path.with_name(path.stem + ".partial.npz")
    results = load_partial_arrays(partial_path) if cached_file_valid(partial_path, params) else {}
    todo = [ym for ym in months if ym not in results]
    fetched, failed = fetch_with_retries(
        todo,
        read,
        max_workers=climate.get("max_workers", 2),
        retry_rounds=climate.get("retry_rounds", 3),
        backoff_s=climate.get("retry_backoff_s", 20),
    )
    results.update(fetched)
    if failed:
        save_partial_arrays(partial_path, results)
        write_sidecar(partial_path, params, extra={"n_months_done": len(results)})
        raise ClimateDownloadError(
            f"{len(failed)} of {len(months)} CHIRPS overview files could not be read; the "
            f"server may be rate-limiting (HTTP 429). {len(results)} months are saved; run "
            "again later and only the missing months will be fetched."
        )
    n_years = last_year - first_year + 1
    rain = np.sum([results[ym][0] for ym in months], axis=0) / n_years
    extent = results[months[0]][1]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rain=rain, extent=np.array(extent))
    write_sidecar(path, params)
    # The partial file was only a resume aid for this cache; the full cache supersedes it.
    for leftover in (partial_path, sidecar_path(partial_path)):
        leftover.unlink(missing_ok=True)
    return rain, extent


def save_partial_arrays(
    path: Path, results: dict[tuple[int, int], tuple[np.ndarray, list[float]]]
) -> None:
    """Save per-month ``(array, extent)`` results to one ``.npz`` (keys ``m_YYYY_MM``)."""
    payload = {}
    for (year, month), (arr, extent) in results.items():
        payload[f"m_{year}_{month:02d}"] = arr
        payload[f"e_{year}_{month:02d}"] = np.asarray(extent, dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_partial_arrays(path: Path) -> dict[tuple[int, int], tuple[np.ndarray, list[float]]]:
    """Inverse of :func:`save_partial_arrays`; returns ``{(year, month): (array, extent)}``."""
    out = {}
    with np.load(path) as npz:
        for key in npz.files:
            if key.startswith("m_"):
                _, year, month = key.split("_")
                out[(int(year), int(month))] = (npz[key], npz[f"e_{year}_{month}"].tolist())
    return out


# =====================================================================================
# Auxiliary layers on the Sentinel-2 extraction grid (M1-T07)
# =====================================================================================

#: ESA WorldCover class codes (Planetary Computer `esa-worldcover` collection metadata).
WORLDCOVER_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}


class AuxDownloadError(RuntimeError):
    """Some auxiliary layers could not be downloaded; the others were cached."""


def grid_transform(window_utm: Sequence[float], resolution_m: float) -> tuple[Any, int, int]:
    """North-up geotransform, width and height of a grid covering a UTM window.

    Parameters
    ----------
    window_utm : [xmin, ymin, xmax, ymax]
        Window in metres; sides must be whole multiples of ``resolution_m``.
    resolution_m : float
        Pixel size in metres.

    Returns
    -------
    (affine.Affine, int, int)
        Transform, width, height. Matches the grid of a Sentinel Hub request for the same
        bbox and resolution.
    """
    from affine import Affine

    xmin, ymin, xmax, ymax = window_utm
    width = (xmax - xmin) / resolution_m
    height = (ymax - ymin) / resolution_m
    if width != int(width) or height != int(height):
        raise ValueError("window is not a whole number of pixels")
    return Affine(resolution_m, 0, xmin, 0, -resolution_m, ymax), int(width), int(height)


def grids_nested(
    fine: Any, fine_shape: tuple[int, int], coarse: Any, coarse_shape: tuple[int, int], factor: int
) -> bool:
    """True if a fine grid subdivides a coarse grid exactly (``factor`` x ``factor`` per pixel).

    ``factor = 1`` tests identical grids. Shapes are ``(height, width)``.
    """
    return (
        abs(fine.c - coarse.c) < 1e-6
        and abs(fine.f - coarse.f) < 1e-6
        and abs(fine.a * factor - coarse.a) < 1e-9
        and abs(fine.e * factor - coarse.e) < 1e-9
        and fine.b == coarse.b == 0
        and fine.d == coarse.d == 0
        and fine_shape == (coarse_shape[0] * factor, coarse_shape[1] * factor)
    )


def slope_aspect(dem: np.ndarray, resolution_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope and aspect from a north-up DEM (Horn's 3 x 3 method, edges replicated).

    Parameters
    ----------
    dem : numpy.ndarray, shape (H, W)
        Elevation in metres; rows run north to south.
    resolution_m : float
        Pixel size in metres.

    Returns
    -------
    slope_deg : numpy.ndarray
        Slope in degrees (0 = flat).
    aspect_deg : numpy.ndarray
        Downslope direction in degrees clockwise from north (0 = N, 90 = E); NaN where flat.
    """
    z = np.pad(np.asarray(dem, dtype="float64"), 1, mode="edge")
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dz_east = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * resolution_m)
    dz_south = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * resolution_m)
    dz_north = -dz_south
    slope = np.degrees(np.arctan(np.hypot(dz_east, dz_north)))
    aspect = np.degrees(np.arctan2(-dz_east, -dz_north)) % 360
    aspect = np.where((dz_east == 0) & (dz_north == 0), np.nan, aspect)
    return slope.astype("float32"), aspect.astype("float32")


def block_mode(arr: np.ndarray, factor: int, nodata: int = 0) -> np.ndarray:
    """Most frequent value in each ``factor`` x ``factor`` block (ties: smallest code).

    ``nodata`` is ignored unless a block has nothing else.
    """
    h, w = arr.shape
    if h % factor or w % factor:
        raise ValueError("array shape must be divisible by factor")
    blocks = arr.reshape(h // factor, factor, w // factor, factor).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(h // factor, w // factor, factor * factor)
    out = np.full((h // factor, w // factor), nodata, dtype=arr.dtype)
    best = np.zeros(out.shape, dtype=int)
    for value in np.unique(arr):
        if value == nodata:
            continue
        count = (blocks == value).sum(axis=2)
        better = count > best
        out[better] = value
        best[better] = count[better]
    return out


def block_class_fractions(
    arr: np.ndarray, factor: int, classes: Sequence[int]
) -> dict[int, np.ndarray]:
    """Fraction of each class in ``factor`` x ``factor`` blocks (e.g. 10 m pixels per 100 m cell).

    Returns ``{class: (H/f, W/f) float32 array}``; fractions are of all pixels in the block.
    """
    h, w = arr.shape
    if h % factor or w % factor:
        raise ValueError("array shape must be divisible by factor")
    blocks = arr.reshape(h // factor, factor, w // factor, factor)
    n = factor * factor
    return {c: (blocks == c).sum(axis=(1, 3)).astype("float32") / n for c in classes}


def cell_centres(window_utm: Sequence[float], cell_m: float) -> tuple[np.ndarray, np.ndarray]:
    """UTM x and y of every cell centre, as (rows, cols) arrays (row 0 = north)."""
    xmin, ymin, xmax, ymax = window_utm
    xs = np.arange(xmin + cell_m / 2, xmax, cell_m)
    ys = np.arange(ymax - cell_m / 2, ymin, -cell_m)
    return np.meshgrid(xs, ys)


def sample_nearest(arr: np.ndarray, transform: Any, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Value of the pixel containing each point (NaN outside the raster).

    Parameters
    ----------
    arr : numpy.ndarray, shape (..., H, W)
        Raster; leading dimensions (e.g. months) are kept.
    transform : affine.Affine
        Raster geotransform, same CRS as the points.
    xs, ys : numpy.ndarray
        Point coordinates.
    """
    inv = ~transform
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    cols = inv.a * x + inv.b * y + inv.c
    rows = inv.d * x + inv.e * y + inv.f
    rows = np.floor(rows).astype(int)
    cols = np.floor(cols).astype(int)
    h, w = arr.shape[-2:]
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out = np.full(arr.shape[:-2] + rows.shape, np.nan, dtype="float32")
    out[..., inside] = arr[..., rows[inside], cols[inside]]
    return out


def window_bounds_wgs84(
    window_utm: Sequence[float], epsg: int, pad_deg: float = 0.0
) -> list[float]:
    """WGS84 bounding box of a UTM window (all four corners), padded by ``pad_deg``."""
    from pyproj import Transformer

    xmin, ymin, xmax, ymax = window_utm
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)
    lons, lats = tr.transform([xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax])
    return [min(lons) - pad_deg, min(lats) - pad_deg, max(lons) + pad_deg, max(lats) + pad_deg]


# --- downloads: Planetary Computer (DEM, WorldCover) -------------------------------------

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def pc_items(collection: str, bbox_wgs84: Sequence[float], query: dict | None = None) -> list[Any]:
    """Signed STAC items of a Planetary Computer collection intersecting a bbox."""
    import planetary_computer
    import pystac_client

    catalog = pystac_client.Client.open(PC_STAC_URL, modifier=planetary_computer.sign_inplace)
    return list(
        catalog.search(collections=[collection], bbox=list(bbox_wgs84), query=query).items()
    )


def warp_to_grid(
    hrefs: Sequence[str],
    bounds_wgs84: Sequence[float],
    dst_transform: Any,
    dst_shape: tuple[int, int],
    dst_epsg: int,
    resampling: str,
    dst_dtype: str,
    src_nodata: float | None = None,
) -> np.ndarray:
    """Mosaic the needed part of remote rasters and warp it onto a target UTM grid.

    Only the windows covering ``bounds_wgs84`` are read (COGs over HTTP).
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    from rasterio.warp import reproject

    with rasterio.Env(**GDAL_REMOTE_ENV):
        sources = [rasterio.open(h) for h in hrefs]
        try:
            mosaic, src_transform = merge(sources, bounds=tuple(bounds_wgs84), nodata=src_nodata)
            src_crs = sources[0].crs
        finally:
            for s in sources:
                s.close()
    out = np.zeros(dst_shape, dtype=dst_dtype)
    reproject(
        source=mosaic[0],
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=f"EPSG:{dst_epsg}",
        resampling=getattr(Resampling, resampling),
    )
    return out


def _write_tif(
    path: Path,
    bands: list[np.ndarray],
    names: list[str],
    transform: Any,
    epsg: int,
    nodata: float | None = None,
) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = bands[0].shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=len(bands),
        dtype=bands[0].dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
        compress="deflate",
        nodata=nodata,
    ) as dst:
        for i, (band, name) in enumerate(zip(bands, names, strict=True), start=1):
            dst.write(band, i)
            dst.set_band_description(i, name)


def _aux_entry(
    cfg: dict[str, Any], dataset: str, aoi_id: str, stem: str, params: dict[str, Any]
) -> Path:
    from desertification.config import get_path

    return get_path(cfg, "aux_cache") / dataset / aoi_id / f"{stem}_{request_hash(params)[:10]}.tif"


def dem_layers(
    cfg: dict[str, Any],
    aoi_id: str,
    window_utm: Sequence[float],
    epsg: int,
    resolution_m: float,
    allow_download: bool = False,
    fetch_fn: Callable[..., np.ndarray] | None = None,
) -> Path | None:
    """Copernicus DEM GLO-30 on the extraction grid: elevation (m), slope and aspect (degrees).

    Elevation is bilinearly resampled from the ~30 m DEM onto the exact Sentinel-2 grid of the
    window; slope and aspect are derived at that resolution (:func:`slope_aspect`). The DEM is
    read one pixel wider on every side and cropped after the 3 x 3 derivatives, so border
    pixels get true slopes instead of edge-replicated ones.

    ``fetch_fn()`` (injectable for tests) must return elevation in metres on that padded
    grid, shape ``(H + 2, W + 2)``.

    Returns
    -------
    Path or None
        Cached 3-band float32 GeoTIFF (``elevation_m``, ``slope_deg``, ``aspect_deg``), or
        None if not cached and downloads are not allowed.
    """
    transform, width, height = grid_transform(window_utm, resolution_m)
    params = {
        "dataset": "cop-dem-glo-30",
        "source": "planetary-computer",
        "window_utm": list(map(float, window_utm)),
        "epsg": int(epsg),
        "resolution_m": float(resolution_m),
        "resampling": "bilinear",
        "derived": "slope/aspect Horn 3x3, 1-pixel padded read",
    }
    path = _aux_entry(cfg, "dem", aoi_id, f"dem_{int(resolution_m)}m", params)
    if cached_file_valid(path, params):
        return path
    if not allow_download:
        return None
    xmin, ymin, xmax, ymax = window_utm
    r = resolution_m
    padded = [xmin - r, ymin - r, xmax + r, ymax + r]
    pad_transform, _, _ = grid_transform(padded, r)
    bounds = window_bounds_wgs84(padded, epsg, pad_deg=0.01)
    if fetch_fn is None:

        def fetch_fn() -> np.ndarray:
            items = pc_items("cop-dem-glo-30", bounds)
            if not items:
                raise AuxDownloadError(f"no DEM tiles found for {aoi_id}")
            hrefs = [it.assets["data"].href for it in items]
            return warp_to_grid(
                hrefs, bounds, pad_transform, (height + 2, width + 2), epsg, "bilinear", "float32"
            )

    padded_dem = np.asarray(fetch_fn(), dtype="float32")
    if padded_dem.shape != (height + 2, width + 2):
        raise ValueError(f"DEM fetch returned {padded_dem.shape}, expected padded grid")
    slope, aspect = (a[1:-1, 1:-1] for a in slope_aspect(padded_dem, r))
    elevation = padded_dem[1:-1, 1:-1]
    _write_tif(
        path,
        [elevation, slope, aspect],
        ["elevation_m", "slope_deg", "aspect_deg"],
        transform,
        epsg,
    )
    write_sidecar(
        path,
        params,
        extra={
            "band_names": ["elevation_m", "slope_deg", "aspect_deg"],
            "attribution": "produced using Copernicus WorldDEM-30",
        },
    )
    return path


def worldcover_layer(
    cfg: dict[str, Any],
    aoi_id: str,
    window_utm: Sequence[float],
    epsg: int,
    year: int,
    resolution_m: float = 10.0,
    allow_download: bool = False,
    fetch_fn: Callable[..., np.ndarray] | None = None,
) -> Path | None:
    """ESA WorldCover map for one year on a UTM grid nested in the extraction grid.

    Nearest-neighbour onto a 10 m grid with the same origin as the Sentinel-2 20 m grid, so
    each 20 m pixel holds exactly 2 x 2 WorldCover pixels and each 100 m cell 10 x 10
    (class fractions per cell are computed later with :func:`block_class_fractions`).

    Returns
    -------
    Path or None
        Cached uint8 GeoTIFF (class codes, 0 = no data), or None if not cached and downloads
        are not allowed.
    """
    transform, width, height = grid_transform(window_utm, resolution_m)
    params = {
        "dataset": "esa-worldcover",
        "source": "planetary-computer",
        "year": int(year),
        "window_utm": list(map(float, window_utm)),
        "epsg": int(epsg),
        "resolution_m": float(resolution_m),
        "resampling": "nearest",
    }
    path = _aux_entry(cfg, "worldcover", aoi_id, f"worldcover_{year}_{int(resolution_m)}m", params)
    if cached_file_valid(path, params):
        return path
    if not allow_download:
        return None
    bounds = window_bounds_wgs84(window_utm, epsg, pad_deg=0.005)
    if fetch_fn is None:

        def fetch_fn() -> np.ndarray:
            items = [
                it
                for it in pc_items("esa-worldcover", bounds)
                if str(it.properties.get("start_datetime", it.properties.get("datetime", "")))[:4]
                == str(year)
            ]
            if not items:
                raise AuxDownloadError(f"no WorldCover {year} tiles found for {aoi_id}")
            hrefs = [it.assets["map"].href for it in items]
            return warp_to_grid(
                hrefs, bounds, transform, (height, width), epsg, "nearest", "uint8", src_nodata=0
            )

    classes = np.asarray(fetch_fn(), dtype="uint8")
    _write_tif(path, [classes], [f"worldcover_{year}"], transform, epsg, nodata=0)
    write_sidecar(
        path,
        params,
        extra={"classes": WORLDCOVER_CLASSES, "version": "v100" if year == 2020 else "v200"},
    )
    return path


def fetch_aux_per_aoi(
    aoi_ids: Sequence[str],
    fetch_one: Callable[[str], Path | None],
    max_workers: int,
    retry_rounds: int,
    backoff_s: float,
) -> dict[str, Path]:
    """Run one auxiliary download per AOI with retries; raise if any AOI still fails.

    Successful AOIs are cached by ``fetch_one`` and are not downloaded again next time.
    """

    def fetch(aoi_id: str) -> Path | None:
        try:
            return fetch_one(aoi_id)
        except Exception as exc:  # noqa: BLE001 - retried below, then reported
            logger.warning("Auxiliary download failed for %s: %s", aoi_id, exc)
            return None

    done, failed = fetch_with_retries(list(aoi_ids), fetch, max_workers, retry_rounds, backoff_s)
    if failed:
        raise AuxDownloadError(
            f"{len(failed)} AOI(s) failed: {', '.join(failed)}; the other {len(done)} are cached. "
            "Run again later to fetch only the missing ones."
        )
    return done


# --- downloads: CHIRPS monthly grids ---------------------------------------------------


def read_window_array(
    url: str, bounds: Sequence[float], nodata_below: float
) -> tuple[np.ndarray, Any]:
    """Pixels of a remote 4326 raster covering ``bounds`` (all touched) and their transform."""
    import rasterio
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform

    with rasterio.Env(**GDAL_REMOTE_ENV), rasterio.open(f"/vsicurl/{url}") as ds:
        win = window_for_bounds(ds.transform, bounds, ds.width, ds.height)
        if win is None:
            raise ValueError("bounds outside the raster")
        r0, r1, c0, c1 = win
        w = Window(c0, r0, c1 - c0, r1 - r0)
        data = ds.read(1, window=w).astype("float32")
        tr = window_transform(w, ds.transform)
    return np.where(data >= nodata_below, data, np.nan).astype("float32"), tr


def climate_monthly_grids(
    cfg: dict[str, Any],
    source: str,
    bounds_by_aoi: dict[str, list[float]],
    first_year: int,
    last_year: int,
    allow_download: bool = False,
    reader: Callable[[str, Sequence[float], float], tuple[np.ndarray, Any]] | None = None,
) -> dict[str, Path] | None:
    """Monthly climate grids (mm/month, native 0.05°) around each AOI window.

    ``source`` is ``"chirps"`` (CHIRPS v3 rainfall) or ``"eto"`` (Hobbins reference ET); the
    URL template is ``climate_data.<source>_monthly_url``. One multi-band GeoTIFF per AOI
    in EPSG:4326 (band ``YYYY-MM``), to be sampled per cell with :func:`sample_nearest`
    at :func:`cell_centres_lonlat`. Cached per AOI; downloads of missing AOIs resume from a
    partial file and are cached only when every month succeeded.

    Returns
    -------
    dict or None
        ``aoi_id -> Path``, or None if something is missing and downloads are not allowed.
    """
    from desertification.config import get_path

    climate = cfg["climate_data"]
    template, nodata_below = climate[f"{source}_monthly_url"], climate["nodata_below"]
    months = month_list(first_year, last_year)
    names = [f"{y}-{m:02d}" for y, m in months]
    entries = {}
    for aoi_id, bounds in sorted(bounds_by_aoi.items()):
        params = {
            "dataset": f"{source}-monthly",
            "url_template": template,
            "bounds_wgs84": list(map(float, bounds)),
            "years": [first_year, last_year],
            "nodata_below": nodata_below,
        }
        path = (
            get_path(cfg, "aux_cache")
            / source
            / aoi_id
            / (f"monthly_grid_{first_year}_{last_year}_{request_hash(params)[:10]}.tif")
        )
        entries[aoi_id] = (path, params)
    done = {k: p for k, (p, prm) in entries.items() if cached_file_valid(p, prm)}
    missing = {k: bounds_by_aoi[k] for k in entries if k not in done}
    if not missing:
        return done
    if not allow_download:
        return None

    reader = reader or read_window_array
    group = {
        "dataset": f"{source}-monthly",
        "url_template": template,
        "years": [first_year, last_year],
        "bounds": {k: list(map(float, v)) for k, v in sorted(missing.items())},
    }
    partial_path = (
        get_path(cfg, "aux_cache") / source / f"grids.partial-{request_hash(group)[:10]}.json"
    )
    partial = load_cached_json(partial_path, group) or {}
    got = {tuple(map(int, k.split("-"))): v for k, v in partial.items()}

    def fetch(ym: tuple[int, int]) -> dict[str, Any] | None:
        url = template.format(year=ym[0], month=ym[1])
        try:
            out = {}
            for aoi_id, bounds in missing.items():
                arr, tr = reader(url, bounds, nodata_below)
                out[aoi_id] = {
                    "data": np.where(np.isfinite(arr), arr, -1).tolist(),
                    "transform": list(tr)[:6],
                }
            return out
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            logger.warning("Could not read %s: %s", url, exc)
            return None

    todo = [ym for ym in months if ym not in got]
    fetched, failed = fetch_with_retries(
        todo,
        fetch,
        climate.get("max_workers", 2),
        climate.get("retry_rounds", 3),
        climate.get("retry_backoff_s", 20),
    )
    got.update(fetched)
    if failed:
        save_json_with_sidecar(
            partial_path, {f"{y}-{m:02d}": v for (y, m), v in sorted(got.items())}, group
        )
        raise ClimateDownloadError(
            f"{len(failed)} of {len(months)} {source} files could not be read; "
            f"{len(got)} months are saved."
            " Run again later and only the missing months will be fetched."
        )
    from affine import Affine

    for aoi_id in missing:
        path, params = entries[aoi_id]
        bands = []
        for ym in months:
            arr = np.asarray(got[ym][aoi_id]["data"], dtype="float32")
            bands.append(np.where(arr < 0, np.nan, arr).astype("float32"))
        transform = Affine(*got[months[0]][aoi_id]["transform"])
        _write_tif(path, bands, names, transform, 4326)
        write_sidecar(path, params, extra={"units": "mm/month", "band_names": names})
        done[aoi_id] = path
    for leftover in (partial_path, sidecar_path(partial_path)):
        leftover.unlink(missing_ok=True)
    return done


# --- per-cell sampling helpers ---------------------------------------------------------


def read_tif(path: Path) -> tuple[np.ndarray, Any, int, list[str]]:
    """All bands of a GeoTIFF as ``(bands, H, W)`` plus transform, EPSG and band names."""
    import rasterio

    with rasterio.open(path) as ds:
        return ds.read(), ds.transform, ds.crs.to_epsg(), list(ds.descriptions)


def cell_centres_lonlat(
    window_utm: Sequence[float], epsg: int, cell_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre longitudes and latitudes (degrees), shape (rows, cols), row 0 = north."""
    from pyproj import Transformer

    xs, ys = cell_centres(window_utm, cell_m)
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(xs, ys)
    return np.asarray(lon), np.asarray(lat)


def mean_annual_total(monthly: np.ndarray) -> np.ndarray:
    """Mean annual total (mm/year) of a ``(12 * n_years, ...)`` monthly stack in mm/month.

    Pixels with any missing month are NaN (no gap filling).
    """
    monthly = np.asarray(monthly, dtype="float64")
    if monthly.shape[0] % 12:
        raise ValueError("stack must hold whole years of months")
    total = monthly.sum(axis=0)  # NaN propagates
    return (total / (monthly.shape[0] / 12)).astype("float32")


def aridity_grid(precip_monthly: np.ndarray, eto_monthly: np.ndarray) -> np.ndarray:
    """Aridity index P / ETo per pixel from aligned monthly stacks (both mm/month).

    NaN where either input has a missing month or ETo is not positive.
    """
    p = mean_annual_total(precip_monthly)
    e = mean_annual_total(eto_monthly)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(e > 0, p / e, np.nan).astype("float32")


# --- registration-check helpers (M1-T07 overlay) -----------------------------------------


def hillshade(
    dem: np.ndarray, resolution_m: float, azimuth_deg: float = 315.0, altitude_deg: float = 45.0
) -> np.ndarray:
    """Analytical hillshade (0-1) of a north-up DEM in metres, lit from ``azimuth_deg``."""
    slope, aspect = slope_aspect(dem, resolution_m)
    s = np.radians(slope)
    a = np.radians(np.nan_to_num(aspect, nan=0.0))
    az, alt = np.radians(azimuth_deg), np.radians(altitude_deg)
    shade = np.sin(alt) * np.cos(s) + np.cos(alt) * np.sin(s) * np.cos(az - a)
    return np.clip(shade, 0, 1).astype("float32")


def class_edges(arr: np.ndarray) -> np.ndarray:
    """True on pixels whose right or lower neighbour has a different class (boundary mask)."""
    edges = np.zeros(arr.shape, dtype=bool)
    edges[:, :-1] |= arr[:, :-1] != arr[:, 1:]
    edges[:-1, :] |= arr[:-1, :] != arr[1:, :]
    return edges


def find_composite(
    data_raw: Path, aoi_id: str, window_utm: Sequence[float], label_prefix: str = "s10b"
) -> Path | None:
    """Latest cached seasonal composite of an AOI whose sidecar bbox equals ``window_utm``.

    Local lookup only (no network). Returns None if no matching composite is cached.
    """
    want = [float(v) for v in window_utm]
    found = []
    for tif in sorted((Path(data_raw) / aoi_id).glob(f"*_{label_prefix}*.tif")):
        meta_path = sidecar_path(tif)
        if not meta_path.exists():
            continue
        req = json.loads(meta_path.read_text(encoding="utf-8")).get("request", {})
        if [float(v) for v in req.get("bbox_utm", [])] == want:
            found.append(tif)
    return found[-1] if found else None  # names start with the season start date


def grid_lines_projected(
    transform: Any, shape: tuple[int, int], src_epsg: int, dst_epsg: int, n: int = 20
) -> list[np.ndarray]:
    """Pixel boundary lines of a raster grid, densified and reprojected (for overlay plots).

    Returns a list of ``(n, 2)`` arrays of ``(x, y)`` in ``dst_epsg``.
    """
    from pyproj import Transformer

    tr = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    h, w = shape
    t = np.linspace(0, 1, n)
    lines = []
    a, b, c, d, e, f = (
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )

    def project(cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
        return np.column_stack(tr.transform(a * cols + b * rows + c, d * cols + e * rows + f))

    for col in range(w + 1):
        lines.append(project(np.full(n, float(col)), t * h))
    for row in range(h + 1):
        lines.append(project(t * w, np.full(n, float(row))))
    return lines


def busiest_block(mask: np.ndarray, block_px: int) -> tuple[int, int]:
    """Top-left (row, col) of the ``block_px`` square window with the most True pixels.

    Used to choose where to zoom for a registration check (many class boundaries).
    """
    h, w = mask.shape
    if block_px > min(h, w):
        raise ValueError("block larger than the array")
    integral = np.pad(np.cumsum(np.cumsum(mask.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    b = block_px
    sums = integral[b:, b:] - integral[:-b, b:] - integral[b:, :-b] + integral[:-b, :-b]
    r, c = np.unravel_index(np.argmax(sums), sums.shape)
    return int(r), int(c)

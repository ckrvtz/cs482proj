"""Sentinel Hub Process API downloads with caching and provenance (CDSE).

Every download is written under ``data/raw/`` with a ``.meta.json`` sidecar (request
parameters, evalscript SHA-256, timestamp, sentinelhub version) and is skipped when a
matching file already exists. API calls need an explicit ``allow_api=True``.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from desertification.cache import cached_file_valid, write_sidecar

logger = logging.getLogger(__name__)

EVALSCRIPT_DIR = Path(__file__).parent / "evalscripts"


def load_evalscript(name: str) -> tuple[str, str]:
    """Return the text of ``evalscripts/<name>.js`` and its SHA-256."""
    text = (EVALSCRIPT_DIR / f"{name}.js").read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def output_size(bbox_utm: list[float], resolution_m: float) -> tuple[int, int]:
    """Output (width, height) in pixels for a UTM bbox at a given resolution (m)."""
    xmin, ymin, xmax, ymax = bbox_utm
    return round((xmax - xmin) / resolution_m), round((ymax - ymin) / resolution_m)


def write_geotiff(path: Path, array: np.ndarray, bbox_utm: list[float], epsg: int) -> None:
    """Write an (H, W, bands) or (H, W) array as a GeoTIFF in the given UTM CRS."""
    import rasterio
    from rasterio.transform import from_bounds

    data = array if array.ndim == 3 else array[..., np.newaxis]
    height, width, count = data.shape
    transform = from_bounds(*bbox_utm, width, height)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=data.dtype,
        crs=f"EPSG:{epsg}",
        transform=transform,
        compress="deflate",
    ) as dst:
        for band in range(count):
            dst.write(data[..., band], band + 1)


def read_geotiff(path: Path) -> np.ndarray:
    """Read a GeoTIFF as an (H, W, bands) array."""
    import rasterio

    with rasterio.open(path) as src:
        return np.moveaxis(src.read(), 0, -1)


def fetch_true_color(
    cfg: dict[str, Any],
    aoi_id: str,
    bbox_utm: list[float],
    epsg: int,
    start: str,
    end: str,
    allow_api: bool = False,
    request_fn: Callable[..., np.ndarray] | None = None,
) -> tuple[np.ndarray | None, Path, bool]:
    """Least-cloudy true-colour image of an AOI for a time window (visual check only).

    Parameters
    ----------
    cfg : dict
        Project configuration (resolution from ``aoi_selection.true_color_resolution_m``).
    aoi_id : str
        Used in the cache file name.
    bbox_utm : [xmin, ymin, xmax, ymax]
        AOI bounds in metres, in ``epsg``.
    epsg : int
        UTM EPSG code.
    start, end : str
        ISO dates of the window searched for the least-cloudy acquisition.
    allow_api : bool, default False
        If False, never call the API (cache miss returns None).
    request_fn : callable, optional
        Injectable replacement for the Process API call (tests).

    Returns
    -------
    image : numpy.ndarray or None
        (H, W, 4) uint8: R, G, B, alpha (dataMask × 255).
    path : Path
        Cache file.
    called_api : bool
        True if this call made a Process API request.
    """
    from desertification.config import get_path

    resolution = cfg["aoi_selection"]["true_color_resolution_m"]
    evalscript, evalscript_sha = load_evalscript("true_color")
    params = {
        "api": "process",
        "base_url": cfg["sentinel_hub"]["base_url"],
        "collection": "sentinel-2-l2a",
        "evalscript_sha256": evalscript_sha,
        "bbox_utm": list(map(float, bbox_utm)),
        "epsg": int(epsg),
        "time": [start, end],
        "resolution_m": resolution,
        "mosaicking_order": "leastCC",
        "format": "image/tiff",
    }
    path = get_path(cfg, "true_color_cache") / f"{aoi_id}_{start}_{end}_{resolution}m.tif"
    if cached_file_valid(path, params):
        return read_geotiff(path), path, False
    if not allow_api:
        return None, path, False

    if request_fn is None:
        from sentinelhub import CRS, BBox, MimeType, MosaickingOrder, SentinelHubRequest

        from desertification.sh_client import get_s2l2a_collection, make_sh_config

        def request_fn() -> np.ndarray:
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=get_s2l2a_collection(cfg),
                        time_interval=(start, end),
                        mosaicking_order=MosaickingOrder.LEAST_CC,
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=BBox(bbox_utm, crs=CRS(epsg)),
                resolution=(resolution, resolution),
                config=make_sh_config(cfg),
            )
            return request.get_data()[0]

    image = np.asarray(request_fn())
    write_geotiff(path, image, bbox_utm, epsg)
    width, height = output_size(bbox_utm, resolution)
    write_sidecar(path, params, extra={"evalscript": "true_color.js", "size_px": [width, height]})
    logger.info("Saved true-colour image %s", path)
    return image, path, True


TEMPLATE_MARK = "// TEMPLATE"


def render_evalscript(name: str, **values: Any) -> tuple[str, str]:
    """Fill the ``// TEMPLATE`` constants of an evalscript and return (text, SHA-256).

    Each keyword replaces the line ``const <NAME> = ...; // TEMPLATE`` whose upper-case
    name matches, e.g. ``bands=["B04"]`` sets ``const BANDS = ["B04"];``.

    Raises
    ------
    KeyError
        If a keyword has no matching template line.
    """
    import json as _json

    text = (EVALSCRIPT_DIR / f"{name}.js").read_text(encoding="utf-8")
    lines = text.splitlines()
    for key, value in values.items():
        prefix = f"const {key.upper()} = "
        idx = [i for i, ln in enumerate(lines) if ln.startswith(prefix) and TEMPLATE_MARK in ln]
        if len(idx) != 1:
            raise KeyError(f"{name}.js has no single template line for {key.upper()}")
        lines[idx[0]] = f"{prefix}{_json.dumps(value)}; {TEMPLATE_MARK}"
    rendered = "\n".join(lines) + "\n"
    return rendered, hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def processing_units_from_headers(headers: dict[str, str] | None) -> float | None:
    """PU reported in response headers, if the service sends such a header.

    Looks for any header whose name contains "processingunits" (case-insensitive). Returns
    None when there is none, so callers fall back to dashboard readings.
    """
    for key, value in (headers or {}).items():
        if "processingunits" in key.lower().replace("-", "").replace("_", ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def composite_band_names(bands: list[str]) -> list[str]:
    """Band names of a composite GeoTIFF: the reflectance bands then ``valid_count``."""
    return [*bands, "valid_count"]


def download_composite(
    cfg: dict[str, Any],
    aoi_id: str,
    bbox_utm: list[float],
    epsg: int,
    start: str,
    end: str,
    resolution: float,
    bands: list[str] | None = None,
    label: str = "",
    allow_api: bool = False,
    request_fn: Callable[..., Any] | None = None,
    max_cloud_cover: float | None = None,
) -> tuple[Path | None, bool, float | None]:
    """Cloud-masked median composite (UINT16, reflectance x 10000) for one AOI and window.

    Parameters
    ----------
    cfg : dict
        Project configuration (bands and SCL mask default to ``sentinel2`` settings).
    aoi_id : str
        AOI id; files go to ``data/raw/<aoi_id>/``.
    bbox_utm : [xmin, ymin, xmax, ymax]
        Request bounds in metres, in ``epsg``.
    epsg : int
        UTM EPSG code.
    start, end : str
        ISO dates of the compositing window (inclusive).
    resolution : float
        Pixel size in metres.
    bands : list of str, optional
        Bands to composite (default: ``sentinel2.bands``).
    label : str, optional
        Extra file-name tag (e.g. ``"pilot_b10_w5km"``) so variants of the same AOI and
        window can coexist; the request hash in the sidecar is what guards the cache.
    allow_api : bool, default False
        If False, never call the API (a cache miss returns ``None``).
    request_fn : callable, optional
        Replacement for the Process API call (tests). Must return an object with
        ``.decode()`` (giving an (H, W, bands+1) array) and ``.headers``.
    max_cloud_cover : float, optional
        Scene filter, 0–1: only acquisitions whose tile cloud cover is at most this value
        are passed to the evalscript (sentinelhub ``maxcc``). None means no filter. Only
        added to the cache key when set, so unfiltered caches stay valid.

    Returns
    -------
    path : Path or None
        GeoTIFF ``<start>_<end>_<res>m[_<label>].tif``, or None if not cached and not allowed.
    called_api : bool
        True if a Process API request was made.
    reported_pu : float or None
        PU reported by the service in a response header, if any (else None).
    """
    from desertification.config import get_path

    bands = list(bands or cfg["sentinel2"]["bands"])
    evalscript, sha = render_evalscript(
        "median_composite", bands=bands, invalid_scl=cfg["sentinel2"]["invalid_scl"]
    )
    params = {
        "api": "process",
        "base_url": cfg["sentinel_hub"]["base_url"],
        "collection": "sentinel-2-l2a",
        "evalscript_sha256": sha,
        "bands": bands,
        "invalid_scl": list(cfg["sentinel2"]["invalid_scl"]),
        "bbox_utm": [float(v) for v in bbox_utm],
        "epsg": int(epsg),
        "time": [start, end],
        "resolution_m": float(resolution),
        "mosaicking": "ORBIT",
        "sample_type": "UINT16",
        "harmonize_values": "default (true)",
        "format": "image/tiff",
    }
    if max_cloud_cover is not None:
        if not 0 <= max_cloud_cover <= 1:
            raise ValueError("max_cloud_cover must be between 0 and 1")
        params["max_cloud_cover"] = float(max_cloud_cover)
    suffix = f"_{label}" if label else ""
    name = f"{start}_{end}_{int(resolution)}m{suffix}.tif"
    path = get_path(cfg, "data_raw") / aoi_id / name
    if cached_file_valid(path, params):
        return path, False, None
    if not allow_api:
        return None, False, None

    if request_fn is None:
        from sentinelhub import CRS, BBox, MimeType, SentinelHubRequest

        from desertification.sh_client import get_s2l2a_collection, make_sh_config

        def request_fn() -> Any:
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=get_s2l2a_collection(cfg),
                        time_interval=(start, end),
                        maxcc=max_cloud_cover,
                    )
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=BBox(bbox_utm, crs=CRS(epsg)),
                resolution=(resolution, resolution),
                config=make_sh_config(cfg),
            )
            return request.get_data(decode_data=False)[0]

    response = request_fn()
    image = np.asarray(response.decode())
    headers = dict(getattr(response, "headers", {}) or {})
    reported = processing_units_from_headers(headers)
    write_geotiff(path, image.astype(np.uint16), bbox_utm, epsg)
    _set_band_descriptions(path, composite_band_names(bands))
    width, height = output_size(bbox_utm, resolution)
    write_sidecar(
        path,
        params,
        extra={
            "evalscript": "median_composite.js",
            "size_px": [width, height],
            "band_names": composite_band_names(bands),
            "scale": "reflectance x 10000 (valid_count unscaled)",
            "reported_processing_units": reported,
            "response_headers": {k: v for k, v in headers.items() if "auth" not in k.lower()},
        },
    )
    logger.info("Saved composite %s (reported PU: %s)", path, reported)
    return path, True, reported


def _set_band_descriptions(path: Path, names: list[str]) -> None:
    import rasterio

    with rasterio.open(path, "r+") as dst:
        for i, name in enumerate(names, start=1):
            dst.set_band_description(i, name)


def fetch_reflectance_stats(
    cfg: dict[str, Any],
    name: str,
    bbox_utm: list[float],
    epsg: int,
    start: str,
    end: str,
    interval: str,
    resolution: float,
    bands: list[str],
    allow_api: bool = False,
    request_fn: Callable[..., Any] | None = None,
) -> tuple[Any | None, Path, bool]:
    """Statistical API time series of cloud-masked reflectance over a small area (cached).

    Parameters
    ----------
    cfg : dict
        Project configuration.
    name : str
        Cache file label (``data/raw/stats/<name>_<start>_<end>.json``).
    bbox_utm, epsg : list, int
        Area in metres and its UTM EPSG code.
    start, end : str
        ISO dates.
    interval : str
        ISO-8601 aggregation interval, e.g. ``"P1M"``.
    resolution : float
        Pixel size in metres.
    bands : list of str
        Bands to summarise (reflectance 0-1).
    allow_api : bool, default False
        If False, a cache miss returns None.
    request_fn : callable, optional
        Replacement for the API call (tests); returns the response JSON.

    Returns
    -------
    (dict or None, Path, bool)
        Raw response JSON, cache path, whether the API was called.
    """
    from desertification.cache import load_cached_json, save_json_with_sidecar
    from desertification.config import get_path

    evalscript, sha = render_evalscript(
        "stats_reflectance", bands=bands, invalid_scl=cfg["sentinel2"]["invalid_scl"]
    )
    params = {
        "api": "statistical",
        "base_url": cfg["sentinel_hub"]["base_url"],
        "collection": "sentinel-2-l2a",
        "evalscript_sha256": sha,
        "bands": list(bands),
        "bbox_utm": [float(v) for v in bbox_utm],
        "epsg": int(epsg),
        "time": [start, end],
        "interval": interval,
        "resolution_m": float(resolution),
        "mosaicking_order": "leastCC",
    }
    path = get_path(cfg, "data_raw") / "stats" / f"{name}_{start}_{end}.json"
    cached = load_cached_json(path, params)
    if cached is not None:
        return cached, path, False
    if not allow_api:
        return None, path, False

    if request_fn is None:
        from sentinelhub import CRS, BBox, MosaickingOrder, SentinelHubStatistical

        from desertification.sh_client import get_s2l2a_collection, make_sh_config

        def request_fn() -> Any:
            request = SentinelHubStatistical(
                aggregation=SentinelHubStatistical.aggregation(
                    evalscript=evalscript,
                    time_interval=(start, end),
                    aggregation_interval=interval,
                    resolution=(resolution, resolution),
                ),
                input_data=[
                    SentinelHubStatistical.input_data(
                        get_s2l2a_collection(cfg), mosaicking_order=MosaickingOrder.LEAST_CC
                    )
                ],
                bbox=BBox(bbox_utm, crs=CRS(epsg)),
                config=make_sh_config(cfg),
            )
            return request.get_data()[0]

    data = request_fn()
    save_json_with_sidecar(path, data, params, extra={"evalscript": "stats_reflectance.js"})
    return data, path, True

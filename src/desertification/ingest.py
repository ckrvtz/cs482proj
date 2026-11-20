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

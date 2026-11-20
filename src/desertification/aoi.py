"""Area-of-interest (AOI) geometry and selection evidence (M1-T02).

Pure functions: build square AOIs in UTM, derive wet/dry seasons from a rainfall
climatology, compute the aridity index, and summarise Sentinel-2 availability per year.
I/O (downloads, files) lives in ``aux_data``, ``ingest`` and the notebooks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

MONTHS = tuple(range(1, 13))


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing a point.

    Standard 6° zones (no Norway/Svalbard exceptions; none of our AOIs are there).

    Parameters
    ----------
    lon, lat : float
        Degrees (WGS84).

    Returns
    -------
    int
        326xx for the northern hemisphere, 327xx for the southern.
    """
    if not (-180 <= lon <= 180 and -80 <= lat <= 84):
        raise ValueError(f"point outside the UTM domain: lon={lon}, lat={lat}")
    zone = min(int((lon + 180) // 6) + 1, 60)
    return (32600 if lat >= 0 else 32700) + zone


def snapped_square(
    x: float, y: float, size_m: float, snap_m: float
) -> tuple[float, float, float, float]:
    """Square of side ``size_m`` centred near ``(x, y)`` with bounds on a ``snap_m`` grid.

    Parameters
    ----------
    x, y : float
        Centre in projected metres.
    size_m : float
        Side length in metres; must be a whole multiple of ``snap_m``.
    snap_m : float
        Grid spacing in metres (e.g. the 100 m cell size).

    Returns
    -------
    tuple
        ``(xmin, ymin, xmax, ymax)`` in metres.
    """
    if size_m <= 0 or snap_m <= 0 or not math.isclose(size_m / snap_m, round(size_m / snap_m)):
        raise ValueError("size_m must be a positive whole multiple of snap_m")
    xmin = round((x - size_m / 2) / snap_m) * snap_m
    ymin = round((y - size_m / 2) / snap_m) * snap_m
    return (float(xmin), float(ymin), float(xmin + size_m), float(ymin + size_m))


def square_aoi(lon: float, lat: float, size_m: float, snap_m: float) -> dict[str, Any]:
    """Build a square AOI in the UTM zone of its centre.

    Parameters
    ----------
    lon, lat : float
        Centre in degrees (WGS84).
    size_m, snap_m : float
        Side length and snapping grid in metres.

    Returns
    -------
    dict
        ``utm_epsg`` (int), ``bbox_utm`` ([xmin, ymin, xmax, ymax], m), ``polygon_wgs84``
        (closed ring of 5 [lon, lat] pairs: the UTM square's corners) and ``bbox_wgs84``
        ([lon_min, lat_min, lon_max, lat_max] envelope of that ring).
    """
    from pyproj import Transformer

    epsg = utm_epsg(lon, lat)
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    to_wgs = Transformer.from_crs(epsg, 4326, always_xy=True)
    x, y = to_utm.transform(lon, lat)
    xmin, ymin, xmax, ymax = snapped_square(x, y, size_m, snap_m)
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
    ring = [list(map(lambda v: round(v, 6), to_wgs.transform(cx, cy))) for cx, cy in corners]
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return {
        "utm_epsg": epsg,
        "bbox_utm": [xmin, ymin, xmax, ymax],
        "polygon_wgs84": ring,
        "bbox_wgs84": [min(lons), min(lats), max(lons), max(lats)],
    }


def circular_window_totals(monthly: Sequence[float], length: int) -> np.ndarray:
    """Totals of every ``length``-month window, wrapping around the year.

    Parameters
    ----------
    monthly : sequence of 12 floats
        Values for January..December.
    length : int
        Window length in months (1–12).

    Returns
    -------
    numpy.ndarray, shape (12,)
        Element ``i`` is the total of months ``i+1 .. i+length`` (1-based, wrapping).
    """
    values = np.asarray(monthly, dtype=float)
    if values.shape != (12,):
        raise ValueError("monthly must have 12 values")
    if not 1 <= length <= 12:
        raise ValueError("length must be between 1 and 12")
    doubled = np.concatenate([values, values])
    return np.array([doubled[i : i + length].sum() for i in range(12)])


def seasons_from_climatology(
    monthly_mm: Sequence[float], length: int
) -> tuple[list[int], list[int]]:
    """Wet and dry seasons as the wettest and driest ``length`` consecutive months.

    The two windows may wrap around the year end (e.g. Nov–Feb). Ties go to the earliest
    starting month. For strongly seasonal climates the windows do not overlap; in nearly
    aseasonal climates they can, which signals that seasons matter little there.

    Parameters
    ----------
    monthly_mm : sequence of 12 floats
        Mean monthly rainfall (mm), January..December.
    length : int
        Season length in months.

    Returns
    -------
    (list of int, list of int)
        Wet-season and dry-season month numbers (1–12), in calendar order of the window.
    """
    totals = circular_window_totals(monthly_mm, length)
    wet_start = int(np.argmax(totals))
    dry_start = int(np.argmin(totals))
    wet = [(wet_start + k) % 12 + 1 for k in range(length)]
    dry = [(dry_start + k) % 12 + 1 for k in range(length)]
    return wet, dry


def season_share(monthly_mm: Sequence[float], months: Sequence[int]) -> float:
    """Fraction of annual rainfall falling in ``months`` (0–1; NaN if the year is dry)."""
    values = np.asarray(monthly_mm, dtype=float)
    total = values.sum()
    if total <= 0:
        return float("nan")
    return float(values[[m - 1 for m in months]].sum() / total)


def aridity_index(annual_precip_mm: float, annual_pet_mm: float) -> float:
    """Aridity index = mean annual precipitation / mean annual potential evapotranspiration.

    Both in mm/year. Returns NaN when PET is missing or not positive.
    """
    if annual_pet_mm is None or not np.isfinite(annual_pet_mm) or annual_pet_mm <= 0:
        return float("nan")
    return float(annual_precip_mm) / float(annual_pet_mm)


def aridity_class(ai: float, dryland_range: Sequence[float]) -> str:
    """Classify an aridity index with the UNCCD dryland range.

    Parameters
    ----------
    ai : float
        Aridity index (P/PET).
    dryland_range : (float, float)
        Inclusive range defining drylands; UNCCD Art. 1 uses 0.05–0.65.

    Returns
    -------
    str
        ``"hyper-arid"`` (below the range), ``"dryland"`` (inside) or ``"humid"`` (above);
        ``"unknown"`` for NaN.
    """
    low, high = dryland_range
    if ai is None or not np.isfinite(ai):
        return "unknown"
    if ai < low:
        return "hyper-arid"
    if ai <= high:
        return "dryland"
    return "humid"


def monthly_climatology(series: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Mean of each calendar month across years.

    Parameters
    ----------
    series : pandas.DataFrame
        Long table with columns ``aoi_id``, ``year``, ``month`` and ``value_col``.

    Returns
    -------
    pandas.DataFrame
        Index ``aoi_id``, columns 1..12, mean value per month (NaNs ignored).
    """
    clim = series.groupby(["aoi_id", "month"])[value_col].mean().unstack("month")
    return clim.reindex(columns=list(MONTHS))


def availability_by_year(scenes: pd.DataFrame, usable_cloud_pct: float) -> pd.DataFrame:
    """Summarise Sentinel-2 L2A catalog results per year for one AOI.

    A *date* counts once even if several tiles or products cover the AOI that day; it is
    *usable* if at least one of them has tile cloud cover <= ``usable_cloud_pct``.

    Parameters
    ----------
    scenes : pandas.DataFrame
        Output of :func:`desertification.sh_client.summarize_scenes` (columns
        ``datetime``, ``date``, ``cloud_cover_pct``).
    usable_cloud_pct : float
        Cloud-cover threshold in percent (0–100).

    Returns
    -------
    pandas.DataFrame
        One row per year with ``n_products``, ``n_dates``, ``n_usable_dates`` and
        ``median_cloud_pct`` (median over dates of the least-cloudy product that day).
    """
    columns = ["year", "n_products", "n_dates", "n_usable_dates", "median_cloud_pct"]
    if scenes.empty:
        return pd.DataFrame(columns=columns)
    df = scenes.copy()
    df["year"] = pd.to_datetime(df["datetime"], utc=True).dt.year
    per_date = df.groupby(["year", "date"])["cloud_cover_pct"].min().reset_index()
    out = per_date.groupby("year").agg(
        n_dates=("date", "size"),
        n_usable_dates=("cloud_cover_pct", lambda c: int((c <= usable_cloud_pct).sum())),
        median_cloud_pct=("cloud_cover_pct", "median"),
    )
    out["n_products"] = df.groupby("year").size()
    return out.reset_index()[columns]


def aoi_feature(
    candidate: dict[str, Any], geometry: dict[str, Any], **properties: Any
) -> dict[str, Any]:
    """Build a GeoJSON Feature for ``config/aois.geojson``.

    Parameters
    ----------
    candidate : dict
        Entry from ``config/aoi_candidates.yaml``.
    geometry : dict
        Output of :func:`square_aoi`.
    **properties
        Extra properties (aridity index, seasons, availability summary, ...).

    Returns
    -------
    dict
        GeoJSON Feature with a WGS84 Polygon and the ROADMAP §4.1 properties.
    """
    props = {
        "aoi_id": candidate["aoi_id"],
        "name": candidate["name"],
        "role": candidate["role"],
        "expected_dynamic": candidate.get("expected_dynamic"),
        "centre_wgs84": [candidate["centre_lon"], candidate["centre_lat"]],
        "utm_epsg": geometry["utm_epsg"],
        "bbox_utm": geometry["bbox_utm"],
        "bbox_wgs84": geometry["bbox_wgs84"],
        "notes": candidate.get("notes"),
    }
    props.update(properties)
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "Polygon", "coordinates": [geometry["polygon_wgs84"]]},
    }


def season_window(months: Sequence[int], year: int) -> tuple[str, str]:
    """ISO start and end dates of a season that ends in ``year``.

    Parameters
    ----------
    months : sequence of int
        Consecutive months (1–12) in season order, possibly wrapping the year end
        (e.g. ``[12, 1, 2, 3]``).
    year : int
        Year in which the season's last month falls.

    Returns
    -------
    (str, str)
        First day of the first month and last day of the last month, e.g.
        ``("2022-12-01", "2023-03-31")`` for ``[12, 1, 2, 3]`` and 2023.
    """
    import calendar

    months = list(months)
    if not months or any(not 1 <= m <= 12 for m in months):
        raise ValueError("months must be non-empty values in 1..12")
    first, last = months[0], months[-1]
    start_year = year - 1 if first > last else year
    last_day = calendar.monthrange(year, last)[1]
    return f"{start_year}-{first:02d}-01", f"{year}-{last:02d}-{last_day:02d}"

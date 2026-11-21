"""Quality checks for downloaded data (M1-T05 acceptance checks). Pure functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DN_SCALE = 10_000  # harmonised Sentinel-2 L2A DN = reflectance x 10 000


def valid_count_summary(count: np.ndarray) -> dict[str, float]:
    """Summarise a per-pixel valid-observation count band.

    Returns the share of pixels with no valid observation (holes), the minimum, median and
    maximum count, and the share of pixels with fewer than 3 valid observations.
    """
    c = np.asarray(count, dtype=float).ravel()
    return {
        "pixels": int(c.size),
        "share_zero": float((c == 0).mean()),
        "share_below_3": float((c < 3).mean()),
        "min": float(c.min()),
        "median": float(np.median(c)),
        "max": float(c.max()),
    }


def reflectance_summary(
    dn: np.ndarray, valid: np.ndarray, scale: float = DN_SCALE
) -> dict[str, float]:
    """Range check of reflectance (DN / scale) over valid pixels.

    Parameters
    ----------
    dn : numpy.ndarray
        One band of harmonised DN (reflectance x scale).
    valid : numpy.ndarray of bool
        Pixels to include (e.g. valid_count > 0).

    Returns
    -------
    dict
        Percentiles (1, 50, 99) and the share of valid pixels above 1.0 reflectance.
    """
    r = np.asarray(dn, dtype=float)[np.asarray(valid, dtype=bool)] / scale
    if r.size == 0:
        return {
            "p01": float("nan"),
            "p50": float("nan"),
            "p99": float("nan"),
            "share_above_1": float("nan"),
        }
    p01, p50, p99 = np.percentile(r, [1, 50, 99])
    return {
        "p01": float(p01),
        "p50": float(p50),
        "p99": float(p99),
        "share_above_1": float((r > 1).mean()),
    }


def parse_stats_response(data: dict[str, Any], output_id: str = "reflectance") -> pd.DataFrame:
    """Flatten a Statistical API response into one row per interval and band.

    Expects ``data["data"][i]["interval"]["from"]`` and
    ``data["data"][i]["outputs"][output_id]["bands"][band]["stats"]`` (mean, sampleCount,
    noDataCount). Intervals without that output (no data) are skipped.

    Returns
    -------
    pandas.DataFrame
        Columns ``date`` (interval start), ``band``, ``mean``, ``sample_count``,
        ``nodata_count``.
    """
    rows = []
    for entry in data.get("data", []):
        output = entry.get("outputs", {}).get(output_id)
        if not output:
            continue
        start = pd.Timestamp(entry["interval"]["from"])
        for band, content in output.get("bands", {}).items():
            stats = content.get("stats", {})
            mean = stats.get("mean")
            rows.append(
                {
                    "date": start,
                    "band": band,
                    "mean": float(mean) if mean not in (None, "NaN") else float("nan"),
                    "sample_count": stats.get("sampleCount"),
                    "nodata_count": stats.get("noDataCount"),
                }
            )
    return pd.DataFrame(rows, columns=["date", "band", "mean", "sample_count", "nodata_count"])


def year_over_year_step(series: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Median same-month change across a date, per band (harmonisation check).

    For every month on or after ``cutoff`` that also has the same month one year earlier
    (before the cutoff), take ``mean(after) - mean(before)``; report the median per band.
    Seasonal effects cancel because the same months are compared. An unharmonised baseline
    change would show as a step of about -0.1 reflectance (the -1000 DN offset).

    Parameters
    ----------
    series : pandas.DataFrame
        Output of :func:`parse_stats_response` (``date``, ``band``, ``mean``).
    cutoff : str
        ISO date of the change (2022-01-25 for processing baseline 04.00).

    Returns
    -------
    pandas.DataFrame
        ``band``, ``n_pairs``, ``median_step`` (reflectance units).
    """
    cut = pd.Timestamp(cutoff, tz="UTC")
    df = series.dropna(subset=["mean"]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["ym"] = df["date"].dt.year * 12 + df["date"].dt.month
    out = []
    for band, g in df.groupby("band"):
        by_ym = g.groupby("ym")["mean"].mean()
        dates = g.groupby("ym")["date"].min()
        steps = [
            by_ym[ym] - by_ym[ym - 12]
            for ym in by_ym.index
            if dates[ym] >= cut and (ym - 12) in by_ym.index and dates[ym - 12] < cut
        ]
        out.append(
            {
                "band": band,
                "n_pairs": len(steps),
                "median_step": float(np.median(steps)) if steps else float("nan"),
            }
        )
    return pd.DataFrame(out, columns=["band", "n_pairs", "median_step"])


def harmonisation_ok(steps: pd.DataFrame, tolerance: float, min_pairs: int = 3) -> bool:
    """True if every band has enough pairs and |median step| <= tolerance (reflectance)."""
    return bool(
        len(steps)
        and (steps["n_pairs"] >= min_pairs).all()
        and (steps["median_step"].abs() <= tolerance).all()
    )

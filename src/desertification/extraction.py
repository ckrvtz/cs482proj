"""Scaled extraction plan and runner (M1-T06).

The plan (pure) lists every composite request with its time window, extraction window and
projected PU; the runner (I/O) downloads them through :func:`ingest.download_composite`,
which caches each file with a provenance sidecar. Runs are resumable: cached requests are
skipped, a per-run request limit and a PU budget stop the run early, and every outcome is
appended to ``data/raw/download_log.csv``.
"""

from __future__ import annotations

import calendar
import csv
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from desertification.aoi import season_window, snapped_square
from desertification.feasibility import complete_seasons, composite_pu, grid_shape, season_fraction

logger = logging.getLogger(__name__)

LOG_COLUMNS = [
    "timestamp_utc",
    "request_id",
    "aoi_id",
    "product",
    "start",
    "end",
    "status",  # downloaded | cached | failed | not_run
    "reported_pu",
    "file",
    "error",
]


# --- windows ---------------------------------------------------------------------------


def extraction_window(
    bbox_utm: Sequence[float],
    size_m: float,
    snap_m: float,
    offset_m: Sequence[float] = (0.0, 0.0),
) -> list[float]:
    """Square window of ``size_m`` inside an AOI bbox: centred, optionally shifted.

    Parameters
    ----------
    bbox_utm : [xmin, ymin, xmax, ymax]
        Validated AOI square in metres.
    size_m, snap_m : float
        Window side and snapping grid in metres.
    offset_m : (dx, dy), default (0, 0)
        Shift of the window centre in metres (east, north).

    Returns
    -------
    list of float
        Window bounds in metres.

    Raises
    ------
    ValueError
        If the window does not fit inside the AOI bbox.
    """
    xmin, ymin, xmax, ymax = bbox_utm
    cx = (xmin + xmax) / 2 + offset_m[0]
    cy = (ymin + ymax) / 2 + offset_m[1]
    window = list(snapped_square(cx, cy, size_m, snap_m))
    if window[0] < xmin or window[1] < ymin or window[2] > xmax or window[3] > ymax:
        raise ValueError(f"window {window} does not fit inside the AOI {list(bbox_utm)}")
    return window


# --- time windows ----------------------------------------------------------------------


def season_windows(
    months: Sequence[int], period_start: str, period_end: str
) -> list[tuple[int, str, str]]:
    """Complete seasons in the period as ``(label_year, start, end)``.

    ``label_year`` is the season's start year (ROADMAP §7.2).
    """
    out = []
    for start_year in complete_seasons(months, period_start, period_end):
        end_year = start_year + 1 if months[0] > months[-1] else start_year
        start, end = season_window(months, end_year)
        out.append((start_year, start, end))
    return out


def month_windows(period_start: str, period_end: str) -> list[tuple[int, int, str, str]]:
    """Every calendar month in the period as ``(year, month, first_day, last_day)``."""
    y0, y1 = int(period_start[:4]), int(period_end[:4])
    out = []
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            last = calendar.monthrange(y, m)[1]
            first_day, last_day = f"{y}-{m:02d}-01", f"{y}-{m:02d}-{last:02d}"
            if first_day >= period_start and last_day <= period_end:
                out.append((y, m, first_day, last_day))
    return out


# --- plan ------------------------------------------------------------------------------


def low_cloud_factor(usable_share: float, adjust: float) -> float:
    """Share of the unfiltered cost charged with the low-cloud filter (capped at 1)."""
    return min(1.0, usable_share * adjust)


def build_plan(
    aois: Sequence[dict[str, Any]],
    extraction: dict[str, Any],
    period: dict[str, str],
    snap_m: float,
    dates_per_year: dict[str, float],
    usable_share: dict[str, float],
    calib: float = 1.0,
    lc_adjust: float = 1.0,
    offsets: dict[str, Sequence[float]] | None = None,
) -> pd.DataFrame:
    """All composite requests of the scaled extraction, with projected PU.

    Parameters
    ----------
    aois : list of dict
        AOI properties (``aoi_id``, ``bbox_utm``, ``utm_epsg``, season months).
    extraction : dict
        ``extraction`` section of the config (window, resolution, bands, cloud filter).
    period : dict
        ``{"start": ..., "end": ...}`` study period.
    snap_m : float
        Grid snapping in metres.
    dates_per_year, usable_share : dict
        Per AOI: mean acquisition dates per year, and the share that are low-cloud.
    calib, lc_adjust : float
        Measured calibration factor and low-cloud adjustment (from the pilot).
    offsets : dict, optional
        Per-AOI window offsets in metres.

    Returns
    -------
    pandas.DataFrame
        One row per request: ``request_id``, ``aoi_id``, ``product`` (seasonal/monthly),
        ``season``, ``year``, ``month``, ``start``, ``end``, ``bands``, ``label``,
        ``window_utm``, ``epsg``, ``projected_pu``.
    """
    res = extraction["resolution_m"]
    maxcc = extraction.get("max_cloud_cover")
    lc_tag = f"_lc{int(round(maxcc * 100))}" if maxcc is not None else ""
    km = int(extraction["window_m"] // 1000)
    rows = []
    for p in aois:
        aoi_id = p["aoi_id"]
        window = extraction_window(
            p["bbox_utm"], extraction["window_m"], snap_m, (offsets or {}).get(aoi_id, (0.0, 0.0))
        )
        w, h = grid_shape(window, res)
        factor = low_cloud_factor(usable_share[aoi_id], lc_adjust) if maxcc is not None else 1.0
        per_year = dates_per_year[aoi_id]
        for season in ("wet", "dry"):
            months = p[f"{season}_season_months"]
            bands = list(extraction["seasonal_bands"])
            pu = composite_pu(
                w, h, len(bands) + 1, per_year * season_fraction(months), float32=False
            )
            for year, start, end in season_windows(months, period["start"], period["end"]):
                rows.append(
                    {
                        "request_id": f"{aoi_id}_{season}_{year}",
                        "aoi_id": aoi_id,
                        "product": "seasonal",
                        "season": season,
                        "year": year,
                        "month": None,
                        "start": start,
                        "end": end,
                        "bands": bands,
                        "label": f"s{len(bands)}b{lc_tag}_w{km}km",
                        "window_utm": window,
                        "epsg": p["utm_epsg"],
                        "projected_pu": pu * calib * factor,
                    }
                )
        bands = list(extraction["monthly_bands"])
        pu = composite_pu(w, h, len(bands) + 1, per_year / 12, float32=False)
        for year, month, start, end in month_windows(period["start"], period["end"]):
            rows.append(
                {
                    "request_id": f"{aoi_id}_m_{year}_{month:02d}",
                    "aoi_id": aoi_id,
                    "product": "monthly",
                    "season": None,
                    "year": year,
                    "month": month,
                    "start": start,
                    "end": end,
                    "bands": bands,
                    "label": f"m{len(bands)}b{lc_tag}_w{km}km",
                    "window_utm": window,
                    "epsg": p["utm_epsg"],
                    "projected_pu": pu * calib * factor,
                }
            )
    plan = pd.DataFrame(rows)
    if plan["request_id"].duplicated().any():
        raise ValueError("duplicate request ids in plan")
    return plan


def expected_filename(row: dict[str, Any], resolution_m: float) -> str:
    """File name ``download_composite`` will use for a plan row."""
    return f"{row['start']}_{row['end']}_{int(resolution_m)}m_{row['label']}.tif"


# --- runner ----------------------------------------------------------------------------


@dataclass
class RunReport:
    """Outcome of one extraction run."""

    downloaded: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    not_run: list[str] = field(default_factory=list)
    reported_pu: float = 0.0
    stopped_reason: str | None = None

    @property
    def api_requests(self) -> int:
        """Requests sent in this run (downloads + failed attempts that reached the API)."""
        return len(self.downloaded)


def run_plan(
    plan: pd.DataFrame,
    download: Callable[[dict[str, Any], bool], tuple[Path | None, bool, float | None]],
    allow_api: bool,
    max_requests: int | None,
    pu_budget: float | None,
    attempts: int = 3,
    backoff_s: float = 15.0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[dict[str, Any]], None] | None = None,
) -> RunReport:
    """Download every plan row that is not cached, within limits, with retries.

    Parameters
    ----------
    plan : pandas.DataFrame
        From :func:`build_plan`.
    download : callable
        ``download(row, allow_api) -> (path, called_api, reported_pu)``; wraps
        :func:`ingest.download_composite`.
    allow_api : bool
        If False, only cached rows are found; everything else is ``not_run``.
    max_requests : int or None
        Stop after this many new downloads in this run (None: no limit).
    pu_budget : float or None
        Stop before a request whose projected PU would push this run's reported PU past
        the budget (None: no limit).
    attempts : int, default 3
        Tries per request; waits ``backoff_s * 2**k`` seconds before retry ``k``.
    sleep : callable
        Injectable for tests.
    log : callable, optional
        Receives one dict per row outcome (``LOG_COLUMNS``).

    Returns
    -------
    RunReport
    """
    report = RunReport()

    def record(
        row: dict[str, Any], status: str, pu: float | None, path: Path | None, error: str = ""
    ) -> None:
        if log is not None:
            log(
                {
                    "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                    "request_id": row["request_id"],
                    "aoi_id": row["aoi_id"],
                    "product": row["product"],
                    "start": row["start"],
                    "end": row["end"],
                    "status": status,
                    "reported_pu": "" if pu is None else pu,
                    "file": "" if path is None else str(path),
                    "error": error,
                }
            )

    for row in plan.to_dict(orient="records"):
        # Cache check first (no API call).
        path, _, pu = download(row, False)
        if path is not None:
            report.cached.append(row["request_id"])
            record(row, "cached", pu, path)
            continue
        if not allow_api:
            report.not_run.append(row["request_id"])
            continue
        if max_requests is not None and len(report.downloaded) >= max_requests:
            report.not_run.append(row["request_id"])
            report.stopped_reason = (
                report.stopped_reason or f"request limit ({max_requests}) reached"
            )
            continue
        if pu_budget is not None and report.reported_pu + row["projected_pu"] > pu_budget:
            report.not_run.append(row["request_id"])
            report.stopped_reason = report.stopped_reason or f"PU budget ({pu_budget:.0f}) reached"
            continue
        error = ""
        for k in range(attempts):
            try:
                path, called, pu = download(row, True)
                break
            except Exception as exc:  # noqa: BLE001 - logged, retried, then reported
                error = f"{type(exc).__name__}: {exc}"[:300]
                logger.warning(
                    "Request %s failed (attempt %d): %s", row["request_id"], k + 1, error
                )
                if k + 1 < attempts:
                    sleep(backoff_s * 2**k)
        else:
            report.failed[row["request_id"]] = error
            record(row, "failed", None, None, error)
            continue
        report.downloaded.append(row["request_id"])
        report.reported_pu += pu or row["projected_pu"]
        record(row, "downloaded", pu, path)
    return report


def append_log(path: Path, entry: dict[str, Any]) -> None:
    """Append one outcome to the download log CSV (header written on first use)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(entry)


def plan_status(plan: pd.DataFrame, present: set[str], failed: dict[str, str]) -> pd.DataFrame:
    """Per-request status for the acceptance check: present, failed (with reason) or missing."""
    status = plan[["request_id", "aoi_id", "product"]].copy()
    status["status"] = [
        "present" if r in present else ("failed" if r in failed else "missing")
        for r in plan["request_id"]
    ]
    status["reason"] = [failed.get(r, "") for r in plan["request_id"]]
    return status

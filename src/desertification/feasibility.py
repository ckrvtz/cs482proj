"""Dataset size and processing-unit projections (M1-T03).

Pure functions: count cells and records from the AOI squares, count complete seasons in
the study period, and project Process API costs from the official CDSE PU rules (via
:func:`desertification.usage.estimate_process_pu`). Projections are planning numbers; the
real cost per request is measured in the pilot (M1-T05).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from desertification.aoi import season_window
from desertification.usage import estimate_process_pu


def grid_shape(bbox_utm: Sequence[float], size_m: float) -> tuple[int, int]:
    """Number of (columns, rows) of ``size_m`` squares in a UTM bbox.

    Raises
    ------
    ValueError
        If the bbox sides are not whole multiples of ``size_m``.
    """
    xmin, ymin, xmax, ymax = bbox_utm
    nx, ny = (xmax - xmin) / size_m, (ymax - ymin) / size_m
    if nx <= 0 or ny <= 0 or nx != int(nx) or ny != int(ny):
        raise ValueError(f"bbox {list(bbox_utm)} is not a whole number of {size_m} m cells")
    return int(nx), int(ny)


def complete_seasons(months: Sequence[int], period_start: str, period_end: str) -> list[int]:
    """Years whose season window lies entirely inside the period.

    Returns the *start* year of each season (ROADMAP §7.2 labels seasons that cross the
    year end by their start year). A Dec–Mar season in a 2019–2025 period gives 2019–2024:
    Dec 2018–Mar 2019 starts before the period, and Dec 2025–Mar 2026 ends after it.

    Parameters
    ----------
    months : sequence of int
        Season months in order (may wrap the year end).
    period_start, period_end : str
        ISO dates (inclusive).

    Returns
    -------
    list of int
    """
    p0, p1 = date.fromisoformat(period_start), date.fromisoformat(period_end)
    years = []
    for end_year in range(p0.year, p1.year + 2):
        start, end = (date.fromisoformat(d) for d in season_window(months, end_year))
        if start >= p0 and end <= p1:
            years.append(start.year)
    return years


def season_fraction(months: Sequence[int]) -> float:
    """Fraction of the year covered by a season (number of months / 12)."""
    return len(months) / 12


def composite_pu(
    width_px: int,
    height_px: int,
    n_input_bands: int,
    acquisitions: float,
    float32: bool,
) -> float:
    """Projected PU of one multi-date composite request (all acquisitions as samples).

    With ``mosaicking: "ORBIT"`` every acquisition in the time window is a data sample per
    pixel, so cost scales with the number of acquisitions (CDSE PU rules).
    """
    return estimate_process_pu(
        width_px, height_px, n_input_bands, n_samples=max(1, round(acquisitions)), float32=float32
    )

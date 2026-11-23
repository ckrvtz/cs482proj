"""Report figures with one consistent style.

Colours follow the project's validated palette: categorical slot 1 blue ``#2a78d6`` and
slot 2 orange ``#eb6834`` (checked with the dataviz validator: CVD ΔE 24.7, all checks
pass on the light surface), a single-hue blue ramp for magnitudes, ink colours for text.
Identity is never carried by colour alone (legends and direct labels).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
WET_TINT = "#cde2fb"
DRY_TINT = "#efeeea"
MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def apply_style() -> None:
    """Set matplotlib defaults: light surface, ink text, recessive grid, thin marks."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2,
            "font.size": 9,
        }
    )


def _save(fig: Any, out_path: Path | None, dpi: int = 200) -> None:
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


def plot_aoi_map(
    rain: np.ndarray,
    extent: Sequence[float],
    aois: Sequence[dict[str, Any]],
    years_label: str,
    out_path: Path | None = None,
) -> Any:
    """Map of AOI locations over mean annual rainfall.

    Parameters
    ----------
    rain : numpy.ndarray
        Mean annual rainfall (mm/year), NaN over the ocean.
    extent : [lon_min, lon_max, lat_min, lat_max]
        Image extent in degrees.
    aois : list of dict
        Each with ``aoi_id``, ``name``, ``lon``, ``lat``; optional ``label_dx``/``label_dy``
        (points) and ``label_ha`` move a label away from neighbours or the plot edge.
    years_label : str
        E.g. ``"2019–2025"``, used in the title and source note.
    out_path : Path, optional
        Where to save the PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bounds = [0, 50, 100, 200, 400, 800, 1600, 4000]
    cmap = ListedColormap(BLUE_RAMP)
    cmap.set_bad(SURFACE)
    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(
        rain, extent=extent, origin="upper", cmap=cmap, norm=norm, interpolation="nearest"
    )
    ax.grid(False)
    for aoi in aois:
        ax.plot(
            aoi["lon"],
            aoi["lat"],
            marker="s",
            markersize=8,
            color=INK,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            linestyle="none",
        )
        ax.annotate(
            aoi["name"],
            (aoi["lon"], aoi["lat"]),
            xytext=(aoi.get("label_dx", 6), aoi.get("label_dy", 6)),
            textcoords="offset points",
            fontsize=8,
            color=INK,
            ha=aoi.get("label_ha", "left"),
            bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none", "alpha": 0.85},
        )
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"Candidate AOIs over mean annual rainfall, {years_label}")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cbar.set_label("Mean annual rainfall (mm/year)")
    cbar.outline.set_visible(False)
    fig.text(
        0.01,
        -0.02,
        f"Rainfall: CHIRPS v3 (Climate Hazards Center), {years_label}. "
        "Squares mark AOI centres (AOIs are ~20 km wide). White = ocean / no data.",
        fontsize=7,
        color=INK_2,
    )
    _save(fig, out_path)
    return fig


def plot_climatology_panels(
    rain_clim: Any,
    eto_clim: Any,
    seasons: dict[str, dict[str, list[int]]],
    titles: dict[str, str],
    years_label: str,
    out_path: Path | None = None,
) -> Any:
    """Small multiples: monthly rainfall (bars) and reference ET (line) per AOI.

    Both series are mm/month on one shared y-axis per panel. Wet and dry seasons are
    shaded and labelled.

    Parameters
    ----------
    rain_clim, eto_clim : pandas.DataFrame
        Index ``aoi_id``, columns 1..12, mm/month.
    seasons : dict
        ``aoi_id -> {"wet": [months], "dry": [months]}``.
    titles : dict
        ``aoi_id -> panel title``.
    years_label : str
        Climatology years, e.g. ``"1991–2020"``.
    out_path : Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    apply_style()
    ids = list(rain_clim.index)
    ncols = 3
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.0 * nrows), squeeze=False)
    months = np.arange(1, 13)
    for ax, aoi_id in zip(axes.flat, ids, strict=False):
        for key, tint, label in (("wet", WET_TINT, "wet"), ("dry", DRY_TINT, "dry")):
            for m in seasons[aoi_id][key]:
                ax.axvspan(m - 0.5, m + 0.5, color=tint, zorder=0, lw=0)
            mid = seasons[aoi_id][key][len(seasons[aoi_id][key]) // 2]
            ax.text(
                mid,
                1.0,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK_2,
            )
        ax.bar(
            months,
            rain_clim.loc[aoi_id].values,
            width=0.7,
            color=SERIES_1,
            label="Rainfall (CHIRPS v3)",
            zorder=2,
        )
        ax.plot(
            months,
            eto_clim.loc[aoi_id].values,
            color=SERIES_2,
            marker="o",
            markersize=4,
            label="Reference ET (Hobbins ETo)",
            zorder=3,
        )
        ax.set_title(titles.get(aoi_id, aoi_id), pad=12)
        ax.set_xticks(months, MONTH_LABELS)
        ax.set_xlim(0.4, 12.6)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("mm/month")
    for ax in list(axes.flat)[len(ids) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"Monthly climatology {years_label}: rainfall, reference ET and seasons", y=1.07, color=INK
    )
    fig.text(
        0.01,
        -0.02,
        "Shading: wet season = wettest 4 consecutive months; dry season = driest "
        "4 consecutive months. Sources: CHIRPS v3; Hobbins reference ET (CHC).",
        fontsize=7,
        color=INK_2,
    )
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_availability(
    availability: Any,
    threshold: int,
    titles: dict[str, str],
    out_path: Path | None = None,
) -> Any:
    """Small multiples: usable Sentinel-2 acquisition dates per year for each AOI.

    Parameters
    ----------
    availability : pandas.DataFrame
        Columns ``aoi_id``, ``year``, ``n_dates``, ``n_usable_dates``.
    threshold : int
        Minimum usable dates per year (drawn as a labelled line).
    titles : dict
        ``aoi_id -> panel title``.
    out_path : Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    apply_style()
    ids = list(dict.fromkeys(availability["aoi_id"]))
    ncols = 3
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.8 * nrows), squeeze=False, sharey=True)
    ymax = max(availability["n_dates"].max(), threshold) * 1.15
    for ax, aoi_id in zip(axes.flat, ids, strict=False):
        sub = availability[availability["aoi_id"] == aoi_id]
        ax.bar(
            sub["year"],
            sub["n_dates"],
            width=0.7,
            color=BLUE_RAMP[1],
            label="All acquisition dates",
            zorder=2,
        )
        ax.bar(
            sub["year"],
            sub["n_usable_dates"],
            width=0.7,
            color=SERIES_1,
            label="Usable dates",
            zorder=3,
        )
        ax.axhline(threshold, color=INK, linewidth=1, linestyle="--", zorder=4)
        ax.text(
            sub["year"].min() - 0.4,
            threshold,
            f"min {threshold}",
            va="bottom",
            fontsize=7,
            color=INK,
        )
        ax.set_title(titles.get(aoi_id, aoi_id))
        ax.set_xticks(sub["year"])
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Dates per year")
    for ax in list(axes.flat)[len(ids) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Sentinel-2 L2A acquisition dates per year (Catalog API)", y=1.08, color=INK)
    fig.text(
        0.01,
        -0.02,
        "Usable = at least one product that day with tile cloud cover at or "
        "below the threshold in config/project.yaml. Contains modified Copernicus "
        "Sentinel data.",
        fontsize=7,
        color=INK_2,
    )
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_image_grid(
    images: dict[str, dict[str, np.ndarray]],
    titles: dict[str, str],
    years_label: str,
    out_path: Path | None = None,
) -> Any:
    """Grid of true-colour thumbnails: one row per AOI, one column per period.

    Parameters
    ----------
    images : dict
        ``aoi_id -> {column label -> (H, W, 3 or 4) uint8 array}``.
    titles : dict
        ``aoi_id -> row label``.
    years_label : str
        Years shown, for the attribution line.
    out_path : Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import textwrap

    import matplotlib.pyplot as plt

    apply_style()
    ids = list(images)
    periods = list(dict.fromkeys(c for d in images.values() for c in d))
    # Landscape: one column per AOI, one row per period (fits a report page).
    fig, axes = plt.subplots(
        len(periods), len(ids), figsize=(2.2 * len(ids), 2.45 * len(periods) + 0.6), squeeze=False
    )
    for c, aoi_id in enumerate(ids):
        for r, period in enumerate(periods):
            ax = axes[r][c]
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            img = images[aoi_id].get(period)
            if img is not None:
                ax.imshow(img[..., :3])
            else:
                ax.text(0.5, 0.5, "not downloaded", ha="center", va="center", color=INK_2)
            if r == 0:
                ax.set_title(textwrap.fill(titles.get(aoi_id, aoi_id), 22), fontsize=8)
            if c == 0:
                ax.set_ylabel(period, fontsize=8, color=INK)
    fig.suptitle("True-colour views of each AOI (least-cloudy image in the dry season)", color=INK)
    fig.text(
        0.01,
        0.005,
        f"Contains modified Copernicus Sentinel data {years_label}, processed "
        "by Sentinel Hub. Each image is ~20 km across, 60 m pixels.",
        fontsize=7,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    _save(fig, out_path, dpi=150)
    return fig


def plot_pilot_panels(
    rgb: np.ndarray,
    ndvi_img: np.ndarray,
    count: np.ndarray,
    title: str,
    attribution: str,
    out_path: Path | None = None,
) -> Any:
    """Three panels for a composite: true colour, NDVI and valid-observation count.

    Parameters
    ----------
    rgb : numpy.ndarray
        (H, W, 3) reflectance 0–1 (already gain-adjusted for display).
    ndvi_img : numpy.ndarray
        (H, W) NDVI, NaN where no data.
    count : numpy.ndarray
        (H, W) valid observations per pixel.
    title, attribution : str
        Figure title and source line.
    out_path : Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    apply_style()
    ramp = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    axes[0].imshow(np.clip(rgb, 0, 1))
    axes[0].set_title("True colour (B04, B03, B02)")
    im1 = axes[1].imshow(
        ndvi_img, cmap=ramp, vmin=0, vmax=max(0.3, float(np.nanpercentile(ndvi_img, 99)))
    )
    axes[1].set_title("NDVI")
    c1 = fig.colorbar(im1, ax=axes[1], shrink=0.8)
    c1.set_label("NDVI (unitless)")
    ramp_holes = LinearSegmentedColormap.from_list("blue_ramp_holes", BLUE_RAMP)
    ramp_holes.set_under(INK_2)  # pixels with no valid observation (cloud-mask holes)
    im2 = axes[2].imshow(count, cmap=ramp_holes, vmin=0.5, vmax=max(1.0, float(np.max(count))))
    axes[2].set_title("Valid observations per pixel\n(grey = none: cloud-mask hole)")
    c2 = fig.colorbar(im2, ax=axes[2], shrink=0.8)
    c2.set_label("Count of valid acquisitions")
    for ax in axes:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, color=INK)
    fig.text(0.01, 0.01, attribution, fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _save(fig, out_path, dpi=150)
    return fig


def plot_reflectance_series(
    series: Any, cutoff: str, title: str, attribution: str, out_path: Path | None = None
) -> Any:
    """Monthly mean reflectance per band with a vertical line at ``cutoff``.

    Parameters
    ----------
    series : pandas.DataFrame
        Columns ``date``, ``band``, ``mean`` (reflectance 0–1).
    cutoff : str
        ISO date marked on the plot (e.g. the 2022-01-25 processing-baseline change).
    title, attribution : str
    out_path : Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    apply_style()
    palette = [SERIES_1, SERIES_2, "#1baf7a", "#eda100"]  # validated categorical slots 1-4
    fig, ax = plt.subplots(figsize=(9, 4))
    for color, (band, g) in zip(palette, series.groupby("band", sort=False), strict=False):
        g = g.sort_values("date")
        ax.plot(
            pd.to_datetime(g["date"]), g["mean"], marker="o", markersize=4, color=color, label=band
        )
        last = g.dropna(subset=["mean"]).iloc[-1]
        ax.annotate(
            band,
            (pd.to_datetime(last["date"]), last["mean"]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.axvline(pd.Timestamp(cutoff), color=INK, linewidth=1, linestyle="--")
    ax.text(
        pd.Timestamp(cutoff),
        1.0,
        f" processing baseline 04.00 ({cutoff})",
        transform=ax.get_xaxis_transform(),
        va="bottom",
        fontsize=7,
        color=INK,
    )
    ax.set_ylabel("Surface reflectance (0–1)")
    ax.set_xlabel("Month")
    ax.set_title(title, pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(palette))
    fig.text(0.01, -0.02, attribution, fontsize=7, color=INK_2)
    fig.tight_layout()
    _save(fig, out_path, dpi=150)
    return fig


def plot_window_preview(
    images: dict[str, np.ndarray | None],
    aoi_bboxes: dict[str, Sequence[float]],
    windows: dict[str, Sequence[float]],
    titles: dict[str, str],
    caption: str,
    out_path: Path | None = None,
) -> Any:
    """True-colour AOI images with the extraction window outlined.

    Parameters
    ----------
    images : dict
        ``aoi_id -> (H, W, 3 or 4) uint8`` image covering the AOI bbox, or None.
    aoi_bboxes, windows : dict
        ``aoi_id -> [xmin, ymin, xmax, ymax]`` in the AOI's UTM metres.
    titles : dict
        ``aoi_id -> panel title``.
    caption : str
        Source line.
    out_path : Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import textwrap

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    apply_style()
    ids = list(aoi_bboxes)
    ncols = 3
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.8 * nrows), squeeze=False)
    for ax, aoi_id in zip(axes.flat, ids, strict=False):
        xmin, ymin, xmax, ymax = aoi_bboxes[aoi_id]
        img = images.get(aoi_id)
        if img is not None:
            ax.imshow(img[..., :3], extent=(xmin, xmax, ymin, ymax))
        else:
            ax.text(
                0.5,
                0.5,
                "no cached image",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=INK_2,
            )
        wx0, wy0, wx1, wy1 = windows[aoi_id]
        ax.add_patch(
            Rectangle((wx0, wy0), wx1 - wx0, wy1 - wy0, fill=False, edgecolor=SURFACE, linewidth=3)
        )
        ax.add_patch(
            Rectangle(
                (wx0, wy0),
                wx1 - wx0,
                wy1 - wy0,
                fill=False,
                edgecolor=SERIES_2,
                linewidth=1.6,
                linestyle="--",
            )
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(textwrap.fill(titles.get(aoi_id, aoi_id), 30), fontsize=9)
    for ax in list(axes.flat)[len(ids) :]:
        ax.set_visible(False)
    fig.suptitle("Extraction windows (dashed) inside the validated 20 km AOI squares", color=INK)
    fig.text(0.01, 0.005, caption, fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, out_path, dpi=150)
    return fig


def plot_aux_overlay(
    panels: list[dict[str, Any]],
    suptitle: str,
    caption: str,
    out_path: Path | None = None,
    ncols: int = 3,
) -> Any:
    """Registration check: auxiliary layers drawn over a Sentinel-2 (or hillshade) background.

    Each panel is a dict with

    * ``title`` (str), ``view`` ([xmin, ymin, xmax, ymax] UTM metres shown);
    * ``background`` ((H, W, 3) float 0-1 true colour, or (H, W) hillshade 0-1) and
      ``background_extent`` (UTM bounds of that array);
    * ``wc_edges`` (bool (h, w) WorldCover class-boundary mask) and ``wc_extent``;
    * ``dem`` ((H, W) elevation in m) and ``dem_extent`` for contour lines;
    * ``chirps_lines`` (list of (n, 2) UTM polylines: CHIRPS 0.05° pixel borders);
    * optional ``pixel_grid`` (float, metres): draw the S2 pixel grid at this spacing.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import textwrap

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    apply_style()
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.9 * nrows), squeeze=False)
    for ax, p in zip(axes.flat, panels, strict=False):
        xmin, ymin, xmax, ymax = p["view"]
        bg = p.get("background")
        if bg is not None:
            x0, y0, x1, y1 = p["background_extent"]
            kw = {"cmap": "gray", "vmin": 0, "vmax": 1} if bg.ndim == 2 else {}
            ax.imshow(bg, extent=(x0, x1, y0, y1), interpolation="nearest", **kw)
        edges = p.get("wc_edges")
        if edges is not None:
            x0, y0, x1, y1 = p["wc_extent"]
            rgba = np.zeros((*edges.shape, 4))
            rgba[edges] = (*_hex_rgb(SERIES_2), 1.0)
            ax.imshow(rgba, extent=(x0, x1, y0, y1), interpolation="nearest")
        dem = p.get("dem")
        if dem is not None:
            x0, y0, x1, y1 = p["dem_extent"]
            h, w = dem.shape
            xs = np.linspace(x0, x1, w, endpoint=False) + (x1 - x0) / w / 2
            ys = np.linspace(y1, y0, h, endpoint=False) - (y1 - y0) / h / 2
            ax.contour(xs, ys, dem, levels=8, colors=SERIES_1, linewidths=0.7)
        for line in p.get("chirps_lines", []):
            ax.plot(line[:, 0], line[:, 1], color=SURFACE, linewidth=1.8)
            ax.plot(line[:, 0], line[:, 1], color=INK, linewidth=0.8, linestyle="--")
        step = p.get("pixel_grid")
        if step:
            for x in np.arange(xmin, xmax + step / 2, step):
                ax.axvline(x, color=SURFACE, linewidth=0.3, alpha=0.6)
            for y in np.arange(ymin, ymax + step / 2, step):
                ax.axhline(y, color=SURFACE, linewidth=0.3, alpha=0.6)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.set_xlabel("UTM easting (m)", fontsize=7)
        ax.set_ylabel("UTM northing (m)", fontsize=7)
        ax.set_title(textwrap.fill(p["title"], 36), fontsize=9)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    handles = [
        Patch(color=SERIES_2, label="WorldCover class boundary (10 m)"),
        Line2D([], [], color=SERIES_1, linewidth=0.9, label="DEM contours (20 m grid)"),
        Line2D([], [], color=INK, linewidth=0.9, linestyle="--", label="CHIRPS 0.05° pixels"),
    ]
    fig.legend(handles=handles, loc="lower right", ncol=3, fontsize=7)
    fig.suptitle(suptitle, color=INK)
    fig.text(0.01, 0.005, textwrap.fill(caption, 120), fontsize=7, color=INK_2, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    _save(fig, out_path, dpi=150)
    return fig


def _hex_rgb(colour: str) -> tuple[float, float, float]:
    colour = colour.lstrip("#")
    return tuple(int(colour[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def stretch_true_colour(rgb: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Percentile stretch of an (H, W, 3) reflectance image to 0-1 (NaN -> 0)."""
    out = np.zeros(rgb.shape, dtype="float32")
    finite = np.isfinite(rgb)
    if not finite.any():
        return out
    lo, hi = np.nanpercentile(rgb, [low, high])
    out[finite] = np.clip((rgb[finite] - lo) / max(hi - lo, 1e-6), 0, 1)
    return out

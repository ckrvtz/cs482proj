"""Column definitions for the processed tables (ROADMAP §7).

One source of truth for column names, dtypes and units, used to size the dataset (M1-T03)
and later to validate the parquet files (M1-T08). Keep in sync with ROADMAP §7; a test
checks that every column named in §7.2 is defined here.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Spectral indices in ``cell_season`` (ROADMAP §4.4), computed from cell-mean reflectances.
INDEX_NAMES = ("NDVI", "EVI", "SAVI", "MSAVI2", "NDMI", "NDWI", "BSI", "TGSI", "albedo")

#: Key columns (identify a record; not counted as attributes).
CELL_SEASON_KEYS = ("cell_id", "aoi_id")


def cell_season_columns(
    bands: Sequence[str], include_band_std: bool = True
) -> list[tuple[str, str, str]]:
    """Columns of ``cell_season.parquet`` (ROADMAP §7.2), in order.

    Parameters
    ----------
    bands : sequence of str
        Sentinel-2 bands from ``config/project.yaml`` (``sentinel2.bands``).
    include_band_std : bool, default True
        Include the within-cell standard deviation of each band (optional in §7.2).

    Returns
    -------
    list of (name, dtype, description)
    """
    cols = [
        ("cell_id", "string", "Cell identifier <aoi_id>_<row>_<col>"),
        ("aoi_id", "category", "AOI identifier"),
        ("year", "int16", "Year (start year for a season that crosses the year end)"),
        ("season", "category", "wet / dry (AOI-specific months)"),
    ]
    cols += [
        (f"{b}_mean", "float32", f"{b} reflectance (0-1), mean over cell pixels") for b in bands
    ]
    if include_band_std:
        cols += [(f"{b}_std", "float32", f"{b} reflectance (0-1), within-cell std") for b in bands]
    cols += [(name, "float32", "Index from cell-mean reflectances") for name in INDEX_NAMES]
    cols += [
        ("valid_obs_median", "float32", "Median per-pixel count of valid observations"),
        ("valid_pixel_frac", "float32", "Fraction of cell pixels with >= 1 valid observation"),
        ("precip_season_mm", "float32", "CHIRPS v3 seasonal rainfall total (mm)"),
    ]
    return cols


def attribute_names(columns: Sequence[tuple[str, str, str]], keys: Sequence[str]) -> list[str]:
    """Column names that count as attributes (everything except the key columns)."""
    return [name for name, _, _ in columns if name not in keys]

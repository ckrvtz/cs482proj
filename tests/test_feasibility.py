"""Tests for the §7 schema and the M1-T03 sizing helpers."""

from __future__ import annotations

import re

import pytest

from desertification.config import find_project_root, load_config
from desertification.feasibility import complete_seasons, composite_pu, grid_shape, season_fraction
from desertification.schema import (
    CELL_SEASON_KEYS,
    INDEX_NAMES,
    attribute_names,
    cell_season_columns,
)

BANDS = load_config()["sentinel2"]["bands"]


def roadmap_section_7_2_names() -> set[str]:
    """Column names written in backticks in ROADMAP §7.2, with 'B02_x … B12_x' expanded."""
    text = (find_project_root() / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    start = text.index("### 7.2 `cell_season.parquet`")
    end = text.index("### 7.3", start)
    section = text[start:end]
    names: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        first_cell = line.split("|")[1]
        tokens = re.findall(r"`([^`]+)`", first_cell)
        if "…" in first_cell and len(tokens) == 2:  # e.g. `B02_mean` … `B12_mean`
            suffix = tokens[0].split("_", 1)[1]
            names.update(f"{b}_{suffix}" for b in BANDS)
        else:
            names.update(tokens)
    return names


def test_schema_matches_roadmap_section_7_2():
    defined = {name for name, _, _ in cell_season_columns(BANDS, include_band_std=True)}
    assert roadmap_section_7_2_names() == defined


def test_cell_season_columns_counts():
    with_std = cell_season_columns(BANDS, include_band_std=True)
    without_std = cell_season_columns(BANDS, include_band_std=False)
    assert len(with_std) == 4 + 2 * len(BANDS) + len(INDEX_NAMES) + 3 == 36
    assert len(without_std) == 36 - len(BANDS)
    assert len({n for n, _, _ in with_std}) == len(with_std)  # unique names
    assert len(attribute_names(with_std, CELL_SEASON_KEYS)) == 34
    assert len(attribute_names(without_std, CELL_SEASON_KEYS)) == 24


def test_grid_shape():
    assert grid_shape([543600, 1703700, 563600, 1723700], 100) == (200, 200)
    assert grid_shape([0, 0, 20000, 20000], 20) == (1000, 1000)
    with pytest.raises(ValueError):
        grid_shape([0, 0, 20050, 20000], 100)


@pytest.mark.parametrize(
    ("months", "expected"),
    [
        ([6, 7, 8, 9], list(range(2019, 2026))),  # within calendar years: 7 seasons
        ([12, 1, 2, 3], list(range(2019, 2025))),  # crosses year end: 6 complete seasons
        ([11, 12, 1, 2], list(range(2019, 2025))),
    ],
)
def test_complete_seasons(months, expected):
    assert complete_seasons(months, "2019-01-01", "2025-12-31") == expected


def test_season_fraction():
    assert season_fraction([6, 7, 8, 9]) == pytest.approx(1 / 3)


def test_composite_pu():
    # 1000 x 1000 px, 3 bands, 1 sample, 16-bit: 1000²/512² PU
    assert composite_pu(1000, 1000, 3, 1, float32=False) == pytest.approx(1e6 / 512**2)
    # scales with samples and doubles for float32
    assert composite_pu(1000, 1000, 3, 10, float32=True) == pytest.approx(20 * 1e6 / 512**2)
    assert composite_pu(1000, 1000, 3, 0.2, float32=False) == pytest.approx(1e6 / 512**2)

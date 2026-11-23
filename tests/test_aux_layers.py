"""Tests for auxiliary layers on the Sentinel-2 grid (M1-T07). No network access."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from desertification.aux_data import (
    WORLDCOVER_CLASSES,
    AuxDownloadError,
    ClimateDownloadError,
    aridity_grid,
    block_class_fractions,
    block_mode,
    busiest_block,
    cell_centres,
    cell_centres_lonlat,
    class_edges,
    climate_monthly_grids,
    dem_layers,
    fetch_aux_per_aoi,
    find_composite,
    grid_lines_projected,
    grid_transform,
    grids_nested,
    hillshade,
    mean_annual_total,
    read_tif,
    sample_nearest,
    slope_aspect,
    window_bounds_wgs84,
    worldcover_layer,
)
from desertification.config import load_config

WINDOW = [500000.0, 1700000.0, 501000.0, 1701000.0]  # 1 km
EPSG = 32628


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = dict(cfg["paths"], aux_cache=str(tmp_path / "aux"))
    cfg["climate_data"] = dict(cfg["climate_data"], retry_rounds=1, retry_backoff_s=0)
    return cfg


# --- grids -----------------------------------------------------------------------------


def test_grid_transform_matches_window():
    tr, w, h = grid_transform(WINDOW, 20)
    assert (w, h) == (50, 50)
    assert tr == Affine(20, 0, 500000, 0, -20, 1701000)
    assert tr @ (w, h) == (501000, 1700000)
    with pytest.raises(ValueError):
        grid_transform([0, 0, 1010, 1000], 20)


def test_grids_nested():
    t20, w20, h20 = grid_transform(WINDOW, 20)
    t10, w10, h10 = grid_transform(WINDOW, 10)
    assert grids_nested(t10, (h10, w10), t20, (h20, w20), 2)
    assert grids_nested(t20, (h20, w20), t20, (h20, w20), 1)
    shifted = Affine(10, 0, 500005, 0, -10, 1701000)
    assert not grids_nested(shifted, (h10, w10), t20, (h20, w20), 2)
    assert not grids_nested(t10, (h10, w10 - 1), t20, (h20, w20), 2)


def test_cell_centres_north_first():
    xs, ys = cell_centres(WINDOW, 100)
    assert xs.shape == ys.shape == (10, 10)
    assert (xs[0, 0], ys[0, 0]) == (500050, 1700950)
    assert (xs[-1, -1], ys[-1, -1]) == (500950, 1700050)


def test_cell_centres_lonlat_and_bounds():
    lon, lat = cell_centres_lonlat(WINDOW, EPSG, 100)
    w, s, e, n = window_bounds_wgs84(WINDOW, EPSG)
    assert lon.shape == (10, 10)
    assert (w < lon).all() and (lon < e).all() and (s < lat).all() and (lat < n).all()
    assert lat[0, 0] > lat[-1, 0]  # row 0 is north
    assert -15.1 < w < -14.9  # UTM 28N central meridian is 15 W
    padded = window_bounds_wgs84(WINDOW, EPSG, pad_deg=0.1)
    assert padded[0] == pytest.approx(w - 0.1) and padded[3] == pytest.approx(n + 0.1)


# --- terrain ---------------------------------------------------------------------------


def plane(dx, dy, n=6, res=20):
    """Elevation rising dx m per m eastward and dy m per m northward (row 0 = north)."""
    cols, rows = np.meshgrid(np.arange(n), np.arange(n))
    return dx * cols * res + dy * (n - 1 - rows) * res


def test_slope_aspect_planes():
    slope, aspect = slope_aspect(plane(0.1, 0), 20)
    assert np.allclose(slope[1:-1, 1:-1], np.degrees(np.arctan(0.1)), atol=1e-4)
    assert np.allclose(aspect[1:-1, 1:-1], 270)  # rises to the east, faces west
    _, aspect = slope_aspect(plane(0, 0.1), 20)
    assert np.allclose(aspect[1:-1, 1:-1], 180)  # rises to the north, faces south
    _, aspect = slope_aspect(plane(-0.1, -0.1), 20)
    assert np.allclose(aspect[1:-1, 1:-1], 45)  # falls to the north-east
    slope, aspect = slope_aspect(np.full((4, 4), 5.0), 20)
    assert (slope == 0).all() and np.isnan(aspect).all()


def test_slope_45_degrees():
    slope, _ = slope_aspect(plane(1.0, 0), 20)
    assert np.allclose(slope[1:-1, 1:-1], 45, atol=1e-4)


# --- land cover ------------------------------------------------------------------------


def test_block_mode_and_ties():
    arr = np.array([[60, 60, 30, 0], [60, 30, 0, 0], [80, 80, 0, 0], [30, 30, 0, 0]], "uint8")
    out = block_mode(arr, 2)
    assert out.tolist() == [[60, 30], [30, 0]]  # tie 80/30 -> smaller code; all-nodata -> 0
    with pytest.raises(ValueError):
        block_mode(arr[:3], 2)


def test_block_class_fractions_sum_to_valid_share():
    rng = np.random.default_rng(482)
    arr = rng.choice([0, 30, 60], size=(20, 20)).astype("uint8")
    fr = block_class_fractions(arr, 10, list(WORLDCOVER_CLASSES))
    assert fr[60].shape == (2, 2)
    total = sum(fr.values())
    valid = (arr != 0).reshape(2, 10, 2, 10).mean(axis=(1, 3))
    assert np.allclose(total, valid)


# --- sampling and aridity ----------------------------------------------------------------


def test_sample_nearest_with_leading_dims():
    tr = Affine(0.05, 0, -15.0, 0, -0.05, 15.4)
    arr = np.arange(2 * 3 * 4, dtype="float32").reshape(2, 3, 4)
    xs = np.array([-14.99, -14.81, -15.5])
    ys = np.array([15.39, 15.26, 15.3])
    out = sample_nearest(arr, tr, xs, ys)
    assert out.shape == (2, 3)
    assert out[0, :2].tolist() == [0, 2 * 4 + 3] and np.isnan(out[:, 2]).all()
    assert out[1, 0] == 12


def test_mean_annual_total_and_aridity():
    p = np.full((24, 2, 2), 10.0)
    e = np.full((24, 2, 2), 100.0)
    e[5, 1, 1] = np.nan
    e[:, 0, 1] = 0
    assert mean_annual_total(p)[0, 0] == pytest.approx(120)
    ai = aridity_grid(p, e)
    assert ai[0, 0] == pytest.approx(0.1)
    assert np.isnan(ai[1, 1]) and np.isnan(ai[0, 1])
    with pytest.raises(ValueError):
        mean_annual_total(p[:13])


# --- downloads (fakes) -----------------------------------------------------------------


def test_dem_layers_cache_and_registration(cfg):
    calls = []

    def fake():
        calls.append(1)
        return plane(0.05, 0, n=52).astype("float32")

    assert dem_layers(cfg, "x", WINDOW, EPSG, 20) is None  # no download allowed
    path = dem_layers(cfg, "x", WINDOW, EPSG, 20, allow_download=True, fetch_fn=fake)
    again = dem_layers(cfg, "x", WINDOW, EPSG, 20, allow_download=True, fetch_fn=fake)
    assert path == again and len(calls) == 1  # cache hit makes no call
    bands, tr, epsg, names = read_tif(path)
    assert names == ["elevation_m", "slope_deg", "aspect_deg"] and epsg == EPSG
    t20, w, h = grid_transform(WINDOW, 20)
    assert grids_nested(tr, bands.shape[1:], t20, (h, w), 1)
    assert np.allclose(bands[1], np.degrees(np.arctan(0.05)), atol=1e-3)  # edges too
    assert bands[0, 0, 0] == pytest.approx(0.05 * 20)  # first pixel after the padding column


def test_worldcover_layer_nested_grid(cfg):
    def fake():
        return np.full((100, 100), 60, "uint8")

    path = worldcover_layer(cfg, "x", WINDOW, EPSG, 2021, allow_download=True, fetch_fn=fake)
    bands, tr, _, names = read_tif(path)
    t20, w, h = grid_transform(WINDOW, 20)
    assert names == ["worldcover_2021"] and bands.dtype == np.uint8
    assert grids_nested(tr, bands.shape[1:], t20, (h, w), 2)
    other = worldcover_layer(cfg, "x", WINDOW, EPSG, 2020, allow_download=True, fetch_fn=fake)
    assert other != path  # years cached separately


def test_fetch_aux_per_aoi_partial_then_resume(tmp_path):
    attempts = {"a": 0, "b": 0}
    down = {"b"}

    def fetch_one(aoi):
        attempts[aoi] += 1
        if aoi in down:
            raise OSError("HTTP 503")
        return tmp_path / f"{aoi}.tif"

    with pytest.raises(AuxDownloadError, match="1 AOI"):
        fetch_aux_per_aoi(["a", "b"], fetch_one, 2, 1, 0)
    down.clear()
    done = fetch_aux_per_aoi(["a", "b"], fetch_one, 2, 1, 0)
    assert set(done) == {"a", "b"}


class FakeClimateReader:
    """Returns a 2 x 3 grid per (url, bounds); urls in ``fail`` raise."""

    def __init__(self, fail=()):
        self.fail = set(fail)
        self.calls = []

    def __call__(self, url, bounds, nodata_below):
        self.calls.append(url)
        if url in self.fail:
            raise OSError("HTTP 429")
        month = int(url.split(".")[-2]) if url.count(".") > 2 else 1
        arr = np.full((2, 3), float(month), "float32")
        arr[0, 0] = np.nan
        return arr, Affine(0.05, 0, bounds[0], 0, -0.05, bounds[3])


def test_climate_grids_cache_partial_resume(cfg):
    cfg["climate_data"] = dict(
        cfg["climate_data"], chirps_monthly_url="mem://p.{year}.{month:02d}.cog"
    )
    bounds = {"x": [-15.0, 15.3, -14.85, 15.4], "y": [1.0, 2.0, 1.15, 2.1]}
    assert climate_monthly_grids(cfg, "chirps", bounds, 2020, 2020) is None
    bad = "mem://p.2020.03.cog"
    reader = FakeClimateReader(fail={bad})
    with pytest.raises(ClimateDownloadError, match="11 months are saved"):
        climate_monthly_grids(cfg, "chirps", bounds, 2020, 2020, allow_download=True, reader=reader)
    assert reader.calls.count(bad) == 2  # first pass + one retry round
    partials = list((cfg["_root"] / cfg["paths"]["aux_cache"] / "chirps").glob("*.partial-*"))
    assert partials  # successes saved, nothing cached as complete
    resume = FakeClimateReader()
    done = climate_monthly_grids(
        cfg, "chirps", bounds, 2020, 2020, allow_download=True, reader=resume
    )
    assert set(resume.calls) == {bad}  # only the missing month
    assert not list(partials[0].parent.glob("*.partial-*"))
    bands, tr, epsg, names = read_tif(done["x"])
    assert epsg == 4326 and names[0] == "2020-01" and bands.shape == (12, 2, 3)
    assert bands[2, 1, 1] == 3 and np.isnan(bands[0, 0, 0])
    hit = FakeClimateReader()
    assert climate_monthly_grids(cfg, "chirps", bounds, 2020, 2020, reader=hit) == done
    assert hit.calls == []


def test_hillshade_and_class_edges():
    flat = hillshade(np.full((5, 5), 10.0), 20)
    assert np.allclose(flat, np.sin(np.radians(45)))
    lit = hillshade(plane(0.5, -0.5), 20)  # rises to the SE, faces NW towards the sun
    dark = hillshade(plane(-0.5, 0.5), 20)  # faces SE, away from it
    assert lit[2, 2] > flat[2, 2] > dark[2, 2]
    arr = np.array([[1, 1, 2], [1, 1, 2], [3, 3, 3]])
    assert class_edges(arr).tolist() == [
        [False, True, False],
        [True, True, True],
        [False, False, False],
    ]


def test_find_composite_matches_window(tmp_path):
    import json

    def put(name, bbox):
        f = tmp_path / "x" / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
        (f.parent / (f.name + ".meta.json")).write_text(json.dumps({"request": {"bbox_utm": bbox}}))

    put("2019-06-01_2019-09-30_20m_s10b_lc40_w10km.tif", WINDOW)
    put("2023-06-01_2023-09-30_20m_s10b_lc40_w10km.tif", WINDOW)
    put("2024-06-01_2024-09-30_20m_s10b_lc40_w10km.tif", [0, 0, 1, 1])
    assert find_composite(tmp_path, "x", WINDOW).name.startswith("2023")
    assert find_composite(tmp_path, "y", WINDOW) is None


def test_grid_lines_projected_identity():
    tr = Affine(20, 0, 500000, 0, -20, 1701000)
    lines = grid_lines_projected(tr, (2, 3), EPSG, EPSG, n=5)
    assert len(lines) == 4 + 3
    assert np.allclose(lines[0][:, 0], 500000) and lines[0][-1, 1] == pytest.approx(1700960)
    assert np.allclose(lines[-1][:, 1], 1700960)


def test_busiest_block():
    mask = np.zeros((10, 10), bool)
    mask[6:9, 2:5] = True
    assert busiest_block(mask, 3) == (6, 2)
    with pytest.raises(ValueError):
        busiest_block(mask, 11)


def test_contour_levels():
    from desertification.viz import contour_levels

    assert contour_levels(28.0, 31.0).size == 0  # Aral: 3 m relief is DEM noise
    lv = contour_levels(1224.7, 1429.6)
    assert lv[0] == 1250 and np.allclose(np.diff(lv), 50)
    assert contour_levels(5, 5).size == 0

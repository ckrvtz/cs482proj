"""Tests for AOI geometry, seasons, aridity, availability and cached fetches (M1-T02).

No network access: remote reads and API calls are replaced with fakes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from affine import Affine

from desertification import aux_data
from desertification.aoi import (
    aoi_feature,
    aridity_class,
    aridity_index,
    availability_by_year,
    circular_window_totals,
    monthly_climatology,
    season_share,
    seasons_from_climatology,
    snapped_square,
    square_aoi,
    utm_epsg,
)
from desertification.aux_data import masked_mean, month_list, monthly_series, window_for_bounds
from desertification.cache import cached_file_valid
from desertification.config import load_config
from desertification.ingest import fetch_true_color, load_evalscript, output_size
from desertification.usage import estimate_process_pu

# Sahel-like rainfall (mm/month): rains Jun–Oct, peak Aug.
SAHEL = [0, 0, 0, 1, 10, 50, 120, 180, 90, 20, 1, 0]


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = dict(
        cfg["paths"],
        aux_cache=str(tmp_path / "aux"),
        true_color_cache=str(tmp_path / "tc"),
    )
    cfg["climate_data"] = dict(cfg["climate_data"], retry_backoff_s=0)  # no real waiting
    return cfg


# --- geometry ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lon", "lat", "epsg"),
    [
        (-14.5, 15.5, 32628),
        (2.6, 13.5, 32631),
        (60.0, 45.0, 32641),
        (108.5, 40.2, 32649),
        (45.3, 37.7, 32638),
        (0.6, 30.6, 32631),
        (18.4, -33.9, 32734),
        (-180, 0, 32601),
    ],
)
def test_utm_epsg(lon, lat, epsg):
    assert utm_epsg(lon, lat) == epsg


def test_utm_epsg_rejects_polar():
    with pytest.raises(ValueError):
        utm_epsg(0, 85)


def test_snapped_square():
    assert snapped_square(1049.0, 2051.0, 2000, 100) == (0.0, 1100.0, 2000.0, 3100.0)
    with pytest.raises(ValueError):
        snapped_square(0, 0, 250, 100)


def test_square_aoi_is_20km_and_aligned():
    geom = square_aoi(-14.5, 15.5, 20000, 100)
    xmin, ymin, xmax, ymax = geom["bbox_utm"]
    assert geom["utm_epsg"] == 32628
    assert (xmax - xmin, ymax - ymin) == (20000, 20000)
    assert all(v % 100 == 0 for v in geom["bbox_utm"])
    ring = geom["polygon_wgs84"]
    assert len(ring) == 5 and ring[0] == ring[-1]
    lon_min, lat_min, lon_max, lat_max = geom["bbox_wgs84"]
    assert lon_min < -14.5 < lon_max and lat_min < 15.5 < lat_max
    assert 0.15 < lat_max - lat_min < 0.2  # 20 km ≈ 0.18° of latitude


# --- seasons and aridity ----------------------------------------------------------------


def test_circular_window_totals_wraps():
    totals = circular_window_totals(list(range(1, 13)), 3)
    assert totals[0] == 1 + 2 + 3
    assert totals[11] == 12 + 1 + 2
    with pytest.raises(ValueError):
        circular_window_totals([1] * 11, 3)


def test_sahel_seasons():
    wet, dry = seasons_from_climatology(SAHEL, 4)
    assert wet == [6, 7, 8, 9]
    assert set(dry).isdisjoint(wet)
    assert dry == [12, 1, 2, 3]
    assert season_share(SAHEL, wet) == pytest.approx(440 / 472)


def test_winter_rain_season_wraps_year_end():
    mediterranean = [60, 50, 40, 20, 10, 2, 0, 0, 5, 20, 40, 60]
    wet, dry = seasons_from_climatology(mediterranean, 4)
    assert wet == [11, 12, 1, 2]
    assert dry == [6, 7, 8, 9]


def test_season_share_dry_year_is_nan():
    assert np.isnan(season_share([0] * 12, [1, 2]))


def test_aridity_index_and_class():
    assert aridity_index(400, 2000) == pytest.approx(0.2)
    assert np.isnan(aridity_index(400, 0))
    rng = (0.05, 0.65)
    assert aridity_class(0.01, rng) == "hyper-arid"
    assert aridity_class(0.05, rng) == "dryland"
    assert aridity_class(0.65, rng) == "dryland"
    assert aridity_class(0.9, rng) == "humid"
    assert aridity_class(float("nan"), rng) == "unknown"


def test_monthly_climatology():
    df = pd.DataFrame(
        {
            "aoi_id": ["a"] * 24,
            "year": [2000] * 12 + [2001] * 12,
            "month": list(range(1, 13)) * 2,
            "value": [1.0] * 12 + [3.0] * 12,
        }
    )
    clim = monthly_climatology(df)
    assert list(clim.columns) == list(range(1, 13))
    assert clim.loc["a", 5] == 2.0


# --- availability -----------------------------------------------------------------------


def test_availability_counts_dates_not_products():
    scenes = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2020-01-01T10:00Z",
                    "2020-01-01T10:00Z",
                    "2020-01-06T10:00Z",
                    "2021-03-01T10:00Z",
                ],
                utc=True,
            ),
            "cloud_cover_pct": [80.0, 10.0, 55.0, 5.0],
        }
    )
    scenes["date"] = scenes["datetime"].dt.date
    out = availability_by_year(scenes, usable_cloud_pct=40).set_index("year")
    assert out.loc[2020, "n_products"] == 3
    assert out.loc[2020, "n_dates"] == 2
    assert out.loc[2020, "n_usable_dates"] == 1  # Jan 1 usable via the 10 % tile
    assert out.loc[2020, "median_cloud_pct"] == pytest.approx((10 + 55) / 2)
    assert out.loc[2021, "n_usable_dates"] == 1


def test_availability_empty():
    assert availability_by_year(pd.DataFrame(), 40).empty


def test_aoi_feature_properties():
    cand = {"aoi_id": "x", "name": "X", "role": "mixed", "centre_lon": 1.0, "centre_lat": 2.0}
    geom = square_aoi(1.0, 2.0, 20000, 100)
    feat = aoi_feature(cand, geom, aridity_index=0.2, wet_season_months=[6, 7, 8, 9])
    assert feat["type"] == "Feature" and feat["geometry"]["type"] == "Polygon"
    for key in (
        "aoi_id",
        "utm_epsg",
        "bbox_utm",
        "bbox_wgs84",
        "aridity_index",
        "wet_season_months",
        "role",
    ):
        assert key in feat["properties"]


# --- raster window helpers --------------------------------------------------------------

GRID_005 = Affine(0.05, 0, -180, 0, -0.05, 60)  # CHIRPS-like global grid


def test_window_for_bounds_covers_all_touched_pixels():
    win = window_for_bounds(GRID_005, (-14.62, 15.41, -14.38, 15.59), 7200, 2400)
    r0, r1, c0, c1 = win
    assert (c0, c1) == (3307, 3313)  # columns -14.65 .. -14.35
    assert (r0, r1) == (888, 892)  # rows 15.60 .. 15.40
    assert window_for_bounds(GRID_005, (200, 0, 201, 1), 7200, 2400) is None


def test_masked_mean_ignores_ocean_fill():
    assert masked_mean(np.array([-9999.0, 10.0, 20.0]), 0) == 15.0
    assert np.isnan(masked_mean(np.array([-9999.0]), 0))


def test_month_list():
    months = month_list(2019, 2020)
    assert len(months) == 24 and months[0] == (2019, 1) and months[-1] == (2020, 12)


def test_monthly_series_caches_and_respects_permission(cfg, monkeypatch):
    calls = []

    def fake_read(url, bboxes, nodata_below):
        calls.append(url)
        return dict.fromkeys(bboxes, 5.0)

    monkeypatch.setattr(aux_data, "read_window_means", fake_read)
    boxes = {"a": [0.0, 0.0, 0.2, 0.2], "b": [1.0, 1.0, 1.2, 1.2]}
    assert monthly_series(cfg, "chirps", boxes, 2020, 2020, allow_download=False) is None
    df = monthly_series(cfg, "chirps", boxes, 2020, 2020, allow_download=True, max_workers=2)
    assert len(df) == 24 and len(calls) == 12
    assert "chirps-v3.0.2020.01.cog" in calls[0] or any("2020.01" in c for c in calls)
    again = monthly_series(cfg, "chirps", boxes, 2020, 2020, allow_download=False)
    assert len(calls) == 12 and again["value"].eq(5.0).all()
    with pytest.raises(ValueError):
        monthly_series(cfg, "nope", boxes, 2020, 2020)


# --- Process API helpers ----------------------------------------------------------------


def test_estimate_process_pu_follows_cdse_rules():
    assert estimate_process_pu(512, 512, 3) == pytest.approx(1.0)
    assert estimate_process_pu(333, 333, 3) == pytest.approx(333 * 333 / 512**2)
    assert estimate_process_pu(1000, 1000, 10, n_samples=20, float32=True) == pytest.approx(
        (1000 * 1000 / 512**2) * (10 / 3) * 20 * 2
    )
    assert estimate_process_pu(10, 10, 3) == pytest.approx(0.01)  # area factor floor 0.01
    assert estimate_process_pu(10, 10, 1) == 0.005  # 0.0033 is below the 0.005 request minimum
    assert estimate_process_pu(1, 1, 1) >= 0.005 - 1e-12
    assert estimate_process_pu(1, 1, 1, statistical=True) == 0.01
    with pytest.raises(ValueError):
        estimate_process_pu(0, 10, 3)


def test_output_size():
    assert output_size([0, 0, 20000, 20000], 60) == (333, 333)
    assert output_size([0, 0, 20000, 20000], 20) == (1000, 1000)


def test_true_color_evalscript_loads():
    text, sha = load_evalscript("true_color")
    assert text.startswith("//VERSION=3") and len(sha) == 64
    assert '"B04", "B03", "B02"' in text


def test_fetch_true_color_cache_and_permission(cfg):
    bbox = [500000.0, 1700000.0, 520000.0, 1720000.0]
    calls = []

    def fake_request():
        calls.append(1)
        return np.full((333, 333, 4), 128, dtype=np.uint8)

    args = (cfg, "t", bbox, 32628, "2023-01-01", "2023-03-31")
    img, path, called = fetch_true_color(*args, allow_api=False, request_fn=fake_request)
    assert img is None and called is False and not calls
    img, path, called = fetch_true_color(*args, allow_api=True, request_fn=fake_request)
    assert called is True and img.shape == (333, 333, 4) and len(calls) == 1
    assert cached_file_valid(path, {"x": 1}) is False
    img2, _, called2 = fetch_true_color(*args, allow_api=True, request_fn=fake_request)
    assert called2 is False and len(calls) == 1
    assert img2.shape == (333, 333, 4) and img2.dtype == np.uint8

    import rasterio

    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 32628
        assert tuple(src.bounds) == tuple(bbox)


def test_season_window():
    from desertification.aoi import season_window

    assert season_window([6, 7, 8, 9], 2023) == ("2023-06-01", "2023-09-30")
    assert season_window([12, 1, 2, 3], 2023) == ("2022-12-01", "2023-03-31")
    assert season_window([1, 2], 2024) == ("2024-01-01", "2024-02-29")
    with pytest.raises(ValueError):
        season_window([], 2023)


def test_monthly_series_retries_a_failed_file_once(cfg, monkeypatch):
    calls = []

    def flaky_read(url, bboxes, nodata_below):
        calls.append(url)
        if url.endswith("2020.07.cog") and calls.count(url) == 1:
            raise OSError("transient network error")
        return dict.fromkeys(bboxes, 7.0)

    monkeypatch.setattr(aux_data, "read_window_means", flaky_read)
    df = monthly_series(
        cfg, "chirps", {"a": [0.0, 0.0, 0.2, 0.2]}, 2020, 2020, allow_download=True, max_workers=2
    )
    assert not df["value"].isna().any()
    assert len(calls) == 13  # 12 months + 1 retry


def test_monthly_series_failure_is_not_cached_and_resumes(cfg, monkeypatch):
    from desertification.aux_data import ClimateDownloadError

    calls = []

    def broken_read(url, bboxes, nodata_below):
        calls.append(url)
        if url.endswith("2020.03.cog"):
            raise OSError("HTTP 429")
        return dict.fromkeys(bboxes, 1.0)

    monkeypatch.setattr(aux_data, "read_window_means", broken_read)
    box = {"a": [0.0, 0.0, 0.2, 0.2]}
    with pytest.raises(ClimateDownloadError, match="1 of 12"):
        monthly_series(cfg, "chirps", box, 2020, 2020, allow_download=True, max_workers=2)
    assert len(calls) == 12 + 3  # first pass + 3 retry rounds for March
    assert monthly_series(cfg, "chirps", box, 2020, 2020, allow_download=False) is None

    # Next run: only the missing month is fetched, then the full cache is written.
    calls.clear()
    monkeypatch.setattr(
        aux_data,
        "read_window_means",
        lambda url, b, n: (calls.append(url), dict.fromkeys(b, 2.0))[1],
    )
    df = monthly_series(cfg, "chirps", box, 2020, 2020, allow_download=True)
    assert calls == [cfg["climate_data"]["chirps_monthly_url"].format(year=2020, month=3)]
    assert len(df) == 12 and not df["value"].isna().any()
    assert df.loc[df["month"] == 3, "value"].item() == 2.0
    assert monthly_series(cfg, "chirps", box, 2020, 2020, allow_download=False) is not None


def test_cache_with_missing_values_is_rebuilt(cfg, monkeypatch):
    import json as _json

    from desertification.cache import sidecar_path

    monkeypatch.setattr(aux_data, "read_window_means", lambda url, b, n: dict.fromkeys(b, 4.0))
    box = {"a": [0.0, 0.0, 0.2, 0.2]}
    df = monthly_series(cfg, "chirps", box, 2020, 2020, allow_download=True)
    path = next((pathlib_path for pathlib_path in _cache_files(cfg)), None)
    meta = _json.loads(sidecar_path(path).read_text())
    meta["n_missing_values"] = 3  # simulate a gap-filled cache from the old code
    sidecar_path(path).write_text(_json.dumps(meta))
    assert monthly_series(cfg, "chirps", box, 2020, 2020, allow_download=False) is None
    assert len(df) == 12


def _cache_files(cfg):
    from desertification.config import get_path

    return sorted((get_path(cfg, "aux_cache") / "chirps").glob("monthly_*_*.json"))


def test_fetch_with_retries_backoff():
    from desertification.aux_data import fetch_with_retries

    attempts, waits = {}, []

    def flaky(item):
        attempts[item] = attempts.get(item, 0) + 1
        return None if item == "b" and attempts[item] < 3 else item.upper()

    done, failed = fetch_with_retries(
        ["a", "b"], flaky, 2, retry_rounds=3, backoff_s=5, sleep=waits.append
    )
    assert done == {"a": "A", "b": "B"} and failed == []
    assert waits == [5, 10]  # two retry rounds needed: 5 s then 10 s

    done, failed = fetch_with_retries(
        ["x"], lambda i: None, 1, retry_rounds=2, backoff_s=1, sleep=waits.append
    )
    assert done == {} and failed == ["x"]


def test_clean_overview():
    from desertification.aux_data import clean_overview

    out = clean_overview(np.array([-9999.0, -4000.0, -0.04, 0.0, 12.5]))
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == 0.0 and out[3] == 0.0 and out[4] == pytest.approx(12.5)
    assert out.dtype == np.float32


def test_rainfall_overview_resumes_from_partial(cfg, monkeypatch):
    """A failed month is retried on the next run; months already read are not re-fetched."""
    from desertification.aux_data import ClimateDownloadError, rainfall_overview
    from desertification.config import get_path

    calls = []
    failing = {"2020-07"}

    class FakeDataset:
        """Minimal stand-in for a rasterio dataset on the CHIRPS 0.05° grid."""

        transform = GRID_005
        width, height = 7200, 2400

        def __init__(self, url):
            self.url = url

        def __enter__(self):
            key = self.url.split("chirps-v3.0.")[1][:7].replace(".", "-")
            calls.append(key)
            if key in failing:
                raise OSError("HTTP 429")
            return self

        def __exit__(self, *exc):
            return False

        def read(self, band, window, out_shape):
            return np.full(out_shape, 10.0, dtype="float32")  # 10 mm every month

    import rasterio

    monkeypatch.setattr(rasterio, "open", lambda url: FakeDataset(url))
    bounds = [0.0, 10.0, 4.0, 14.0]
    with pytest.raises(ClimateDownloadError, match="1 of 12"):
        rainfall_overview(cfg, bounds, 2020, 2020, decimation=8, allow_download=True)
    partial = get_path(cfg, "aux_cache") / "chirps" / "overview_2020_2020.partial.npz"
    assert partial.is_file()
    assert calls.count("2020-01") == 1 and calls.count("2020-07") == 4  # 1 + 3 retry rounds

    calls.clear()
    failing.clear()
    rain, extent = rainfall_overview(cfg, bounds, 2020, 2020, decimation=8, allow_download=True)
    assert calls == ["2020-07"]  # only the missing month
    assert rain.shape == (10, 10) and np.allclose(rain, 120.0)  # 12 x 10 mm
    assert extent == pytest.approx([0.0, 4.0, 10.0, 14.0])
    assert not partial.exists()
    assert rainfall_overview(cfg, bounds, 2020, 2020, decimation=8) is not None  # cached


def test_partial_arrays_round_trip(tmp_path):
    from desertification.aux_data import load_partial_arrays, save_partial_arrays

    data = {
        (2020, 1): (np.ones((2, 3), "float32"), [0.0, 1.0, 2.0, 3.0]),
        (2021, 12): (np.zeros((2, 3), "float32"), [0.0, 1.0, 2.0, 3.0]),
    }
    save_partial_arrays(tmp_path / "p.npz", data)
    back = load_partial_arrays(tmp_path / "p.npz")
    assert set(back) == set(data)
    assert np.array_equal(back[(2020, 1)][0], data[(2020, 1)][0]) and back[(2021, 12)][1] == [
        0.0,
        1.0,
        2.0,
        3.0,
    ]

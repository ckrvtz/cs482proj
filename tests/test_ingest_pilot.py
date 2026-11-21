"""Tests for the pilot extraction pieces (M1-T05). No network access."""

from __future__ import annotations

import json

import numpy as np
import pytest

from desertification.aoi import sub_window
from desertification.cache import sidecar_path
from desertification.config import load_config
from desertification.features import ndvi, normalized_difference
from desertification.ingest import (
    composite_band_names,
    download_composite,
    fetch_reflectance_stats,
    processing_units_from_headers,
    render_evalscript,
)
from desertification.qa import (
    harmonisation_ok,
    parse_stats_response,
    reflectance_summary,
    valid_count_summary,
    year_over_year_step,
)

BBOX = [500000.0, 1700000.0, 505000.0, 1705000.0]  # 5 km


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = dict(cfg["paths"], data_raw=str(tmp_path / "raw"))
    return cfg


class FakeResponse:
    def __init__(self, image, headers=None):
        self._image = image
        self.headers = headers or {}

    def decode(self):
        return self._image


# --- evalscripts ----------------------------------------------------------------------


def test_render_evalscript_fills_template_and_hash_changes():
    text_a, sha_a = render_evalscript("median_composite", bands=["B04", "B08"], invalid_scl=[0, 1])
    text_b, sha_b = render_evalscript("median_composite", bands=["B04"], invalid_scl=[0, 1])
    assert 'const BANDS = ["B04", "B08"]; // TEMPLATE' in text_a
    assert "const INVALID_SCL = [0, 1]; // TEMPLATE" in text_a
    assert 'sampleType: "UINT16"' in text_a and 'mosaicking: "ORBIT"' in text_a
    assert sha_a != sha_b and len(sha_a) == 64
    with pytest.raises(KeyError):
        render_evalscript("median_composite", nonexistent=[1])


def test_stats_evalscript_has_datamask_output():
    text, _ = render_evalscript("stats_reflectance", bands=["B04"], invalid_scl=[3])
    assert 'id: "dataMask"' in text and 'const BANDS = ["B04"];' in text


# --- headers --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"x-processingunits-spent": "12.5"}, 12.5),
        ({"X-ProcessingUnits-Spent": "0.3"}, 0.3),
        ({"content-type": "image/tiff"}, None),
        ({"x-processingunits-spent": "n/a"}, None),
        (None, None),
    ],
)
def test_processing_units_from_headers(headers, expected):
    assert processing_units_from_headers(headers) == expected


# --- composite download ---------------------------------------------------------------


def test_download_composite_cache_permission_and_metadata(cfg):
    bands = ["B04", "B08"]
    calls = []
    image = np.dstack(
        [np.full((250, 250), 1200), np.full((250, 250), 3000), np.full((250, 250), 9)]
    )

    def fake():
        calls.append(1)
        return FakeResponse(image, {"x-processingunits-spent": "1.7", "authorization": "secret"})

    args = (cfg, "aral", BBOX, 32641, "2023-06-01", "2023-09-30", 20)
    path, called, pu = download_composite(*args, bands=bands, label="t", request_fn=fake)
    assert path is None and called is False and not calls  # allow_api defaults to False
    path, called, pu = download_composite(
        *args, bands=bands, label="t", allow_api=True, request_fn=fake
    )
    assert called and pu == 1.7 and path.name == "2023-06-01_2023-09-30_20m_t.tif"
    again, called2, _ = download_composite(
        *args, bands=bands, label="t", allow_api=True, request_fn=fake
    )
    assert again == path and called2 is False and len(calls) == 1

    meta = json.loads(sidecar_path(path).read_text())
    assert meta["request"]["sample_type"] == "UINT16" and meta["request"]["bands"] == bands
    assert meta["band_names"] == ["B04", "B08", "valid_count"]
    assert "authorization" not in meta["response_headers"]

    import rasterio

    with rasterio.open(path) as src:
        assert src.dtypes[0] == "uint16" and src.count == 3
        assert src.descriptions == ("B04", "B08", "valid_count")
        assert src.crs.to_epsg() == 32641 and tuple(src.bounds) == tuple(BBOX)

    # A different band set is a different request: new download, own file name.
    path_b, called_b, _ = download_composite(
        *args,
        bands=["B04"],
        label="t1",
        allow_api=True,
        request_fn=lambda: FakeResponse(image[..., [0, 2]]),
    )
    assert called_b and path_b != path


def test_composite_band_names():
    assert composite_band_names(["B04"]) == ["B04", "valid_count"]


# --- statistical API ------------------------------------------------------------------


def stats_json(months, values):
    def band(v):
        return {"B04": {"stats": {"mean": v, "sampleCount": 2500, "noDataCount": 0}}}

    return {
        "data": [
            {
                "interval": {"from": f"{m}-01T00:00:00Z", "to": f"{m}-28T00:00:00Z"},
                "outputs": {"reflectance": {"bands": band(v)}},
            }
            for m, v in zip(months, values, strict=True)
        ]
    }


def test_fetch_reflectance_stats_cache(cfg):
    calls = []
    data = stats_json(["2021-01"], [0.4])

    def fake():
        calls.append(1)
        return data

    args = (cfg, "erg", BBOX, 32631, "2021-01-01", "2022-12-31", "P1M", 20, ["B04"])
    assert fetch_reflectance_stats(*args)[0] is None
    got, _, called = fetch_reflectance_stats(*args, allow_api=True, request_fn=fake)
    assert got == data and called
    got2, _, called2 = fetch_reflectance_stats(*args, allow_api=True, request_fn=fake)
    assert got2 == data and not called2 and len(calls) == 1


def test_parse_stats_response_skips_empty_intervals():
    data = stats_json(["2021-01", "2021-02"], [0.41, "NaN"])
    data["data"].append({"interval": {"from": "2021-03-01T00:00:00Z", "to": "x"}, "outputs": {}})
    df = parse_stats_response(data)
    assert len(df) == 2 and df["band"].tolist() == ["B04", "B04"]
    assert df["mean"].iloc[0] == pytest.approx(0.41) and np.isnan(df["mean"].iloc[1])


def _series(offset_after):
    months = [f"{y}-{m:02d}" for y in (2021, 2022) for m in range(1, 13)]
    seasonal = [0.40 + 0.02 * np.sin(m / 12 * 2 * np.pi) for m in range(24)]
    values = [v + (offset_after if months[i] >= "2022-02" else 0) for i, v in enumerate(seasonal)]
    return parse_stats_response(stats_json(months, values))


def test_year_over_year_step_detects_offset_and_passes_harmonised():
    steps_ok = year_over_year_step(_series(0.0), "2022-01-25")
    assert steps_ok.loc[0, "n_pairs"] == 11  # Feb-Dec 2022 vs 2021
    assert abs(steps_ok.loc[0, "median_step"]) < 1e-9
    assert harmonisation_ok(steps_ok, tolerance=0.02)
    steps_bad = year_over_year_step(_series(-0.1), "2022-01-25")
    assert steps_bad.loc[0, "median_step"] == pytest.approx(-0.1)
    assert not harmonisation_ok(steps_bad, tolerance=0.02)
    assert not harmonisation_ok(steps_ok, tolerance=0.02, min_pairs=12)


# --- QA and features ------------------------------------------------------------------


def test_valid_count_summary():
    s = valid_count_summary(np.array([[0, 1], [5, 10]]))
    assert s["share_zero"] == 0.25 and s["share_below_3"] == 0.5 and s["median"] == 3.0


def test_reflectance_summary():
    dn = np.array([1000, 2000, 12000, 0])
    s = reflectance_summary(dn, dn > 0)
    assert s["share_above_1"] == pytest.approx(1 / 3) and s["p50"] == pytest.approx(0.2)
    assert np.isnan(reflectance_summary(dn, np.zeros(4, bool))["p50"])


def test_ndvi():
    assert ndvi(np.array([0.5]), np.array([0.1]))[0] == pytest.approx(0.4 / 0.6)
    assert np.isnan(normalized_difference(np.array([0.0]), np.array([0.0]))[0])


def test_sub_window():
    assert sub_window([0, 0, 20000, 20000], 5000, 100) == [7500.0, 7500.0, 12500.0, 12500.0]
    with pytest.raises(ValueError):
        sub_window([0, 0, 1000, 1000], 5000, 100)

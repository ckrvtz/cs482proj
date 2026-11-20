"""Tests for the CDSE client helpers and catalog cache (P0-T03). No network access."""

from __future__ import annotations

import json

import pytest

from desertification.cache import (
    load_cached_json,
    request_hash,
    save_json_with_sidecar,
    sidecar_path,
)
from desertification.config import load_config
from desertification.sh_client import (
    CredentialsError,
    cached_catalog_search,
    catalog_request_count,
    make_sh_config,
    missing_credentials,
    parse_tile_id,
    summarize_scenes,
    validate_bbox_wgs84,
)

FAKE_ID = "fake-client-id-0123456789"
FAKE_SECRET = "fake-client-secret-abcdefghij"


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["paths"] = dict(cfg["paths"], catalog_cache=str(tmp_path / "catalog"))
    return cfg


def feature(product_id, when, cloud):
    return {"id": product_id, "properties": {"datetime": when, "eo:cloud_cover": cloud}}


FEATURES = [
    feature(
        "S2B_MSIL2A_20230110T112339_N0509_R037_T28PEB_20230110T140203.SAFE",
        "2023-01-10T11:33:00Z",
        12.5,
    ),
    feature(
        "S2A_MSIL2A_20230105T112451_N0509_R037_T28PEB_20230105T150510.SAFE",
        "2023-01-05T11:33:00Z",
        0.4,
    ),
    # exact duplicate, as can happen across overlapping pages
    feature(
        "S2A_MSIL2A_20230105T112451_N0509_R037_T28PEB_20230105T150510.SAFE",
        "2023-01-05T11:33:00Z",
        0.4,
    ),
]


# --- credentials and config -------------------------------------------------------------


def test_missing_credentials_lists_names_only():
    assert missing_credentials({}) == ["SH_CLIENT_ID", "SH_CLIENT_SECRET"]
    assert missing_credentials({"SH_CLIENT_ID": "x", "SH_CLIENT_SECRET": ""}) == [
        "SH_CLIENT_SECRET"
    ]
    assert missing_credentials({"SH_CLIENT_ID": "x", "SH_CLIENT_SECRET": "y"}) == []


def test_make_sh_config_raises_without_credentials(cfg):
    with pytest.raises(CredentialsError) as err:
        make_sh_config(cfg, env={"SH_CLIENT_ID": FAKE_ID})
    assert "SH_CLIENT_SECRET" in str(err.value)
    assert FAKE_ID not in str(err.value)


def test_make_sh_config_points_at_cdse_and_masks_secret(cfg):
    config = make_sh_config(cfg, env={"SH_CLIENT_ID": FAKE_ID, "SH_CLIENT_SECRET": FAKE_SECRET})
    assert config.sh_base_url == "https://sh.dataspace.copernicus.eu"
    assert config.sh_token_url.startswith("https://identity.dataspace.copernicus.eu/")
    assert config.sh_client_id == FAKE_ID
    assert FAKE_SECRET not in repr(config)
    assert "services.sentinel-hub.com" not in config.sh_base_url


# --- pure helpers -----------------------------------------------------------------------


def test_validate_bbox():
    assert validate_bbox_wgs84([-14.55, 15.45, -14.45, 15.55]) == (-14.55, 15.45, -14.45, 15.55)
    for bad in ([0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0], [-200, 0, 0, 1], [0, -95, 1, 0]):
        with pytest.raises(ValueError):
            validate_bbox_wgs84(bad)


def test_connectivity_bbox_in_config_is_valid(cfg):
    validate_bbox_wgs84(cfg["connectivity_check"]["bbox_wgs84"])


def test_parse_tile_id():
    assert parse_tile_id(FEATURES[0]["id"]) == "28PEB"
    assert parse_tile_id("not-a-product") is None


def test_catalog_request_count():
    assert catalog_request_count(0) == 1
    assert catalog_request_count(100) == 1
    assert catalog_request_count(101) == 2


def test_summarize_scenes_dedupes_and_sorts():
    df = summarize_scenes(FEATURES)
    assert len(df) == 2
    assert list(df["cloud_cover_pct"]) == [0.4, 12.5]
    assert list(df["tile_id"]) == ["28PEB", "28PEB"]
    assert str(df["date"].iloc[0]) == "2023-01-05"
    assert list(df["platform"]) == ["S2A", "S2B"]


def test_summarize_scenes_empty():
    df = summarize_scenes([])
    assert df.empty
    assert "cloud_cover_pct" in df.columns


# --- cache ------------------------------------------------------------------------------


def test_request_hash_ignores_key_order():
    assert request_hash({"a": 1, "b": [1, 2]}) == request_hash({"b": [1, 2], "a": 1})
    assert request_hash({"a": 1}) != request_hash({"a": 2})


def test_cache_roundtrip_and_param_mismatch(tmp_path):
    path = tmp_path / "x.json"
    params = {"bbox": [0, 0, 1, 1], "time": ["2023-01-01", "2023-01-31"]}
    assert load_cached_json(path, params) is None
    meta = save_json_with_sidecar(path, [{"id": 1}], params, extra={"n_features": 1})
    assert meta == sidecar_path(path)
    sidecar = json.loads(meta.read_text())
    assert sidecar["request"] == params and sidecar["n_features"] == 1
    assert load_cached_json(path, params) == [{"id": 1}]
    assert load_cached_json(path, dict(params, time=["2023-02-01", "2023-02-28"])) is None


def test_cached_search_without_permission_never_calls_api(cfg):
    def boom(*_args):
        raise AssertionError("API must not be called")

    features, path, called = cached_catalog_search(
        cfg, "t", [0, 0, 1, 1], "2023-01-01", "2023-01-31", allow_api=False, search_fn=boom
    )
    assert features is None and called is False
    assert not path.exists()


def test_cached_search_calls_once_then_uses_cache(cfg):
    calls = []

    def fake_search(bbox, start, end):
        calls.append((bbox, start, end))
        return FEATURES

    args = (cfg, "t", [0, 0, 1, 1], "2023-01-01", "2023-01-31")
    first, path, called_first = cached_catalog_search(*args, allow_api=True, search_fn=fake_search)
    second, _, called_second = cached_catalog_search(*args, allow_api=True, search_fn=fake_search)
    assert called_first is True and called_second is False
    assert len(calls) == 1
    assert first == second == FEATURES
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["catalog_requests"] == 1 and meta["n_features"] == 3
    assert FAKE_SECRET not in path.read_text() + sidecar_path(path).read_text()


def test_search_catalog_wiring(cfg, monkeypatch):
    """search_catalog passes CDSE config, collection, WGS84 bbox and dates to sentinelhub."""
    import sentinelhub

    from desertification.sh_client import get_s2l2a_collection, search_catalog

    captured = {}

    class FakeCatalog:
        def __init__(self, config):
            captured["config"] = config

        def search(self, collection, **kwargs):
            captured["collection"] = collection
            captured.update(kwargs)
            return iter(FEATURES[:1])

    monkeypatch.setattr(sentinelhub, "SentinelHubCatalog", FakeCatalog)
    config = make_sh_config(cfg, env={"SH_CLIENT_ID": FAKE_ID, "SH_CLIENT_SECRET": FAKE_SECRET})
    collection = get_s2l2a_collection(cfg)
    bbox = [-14.55, 15.45, -14.45, 15.55]
    result = search_catalog(config, collection, bbox, "2023-01-01", "2023-01-31")

    assert result == FEATURES[:1]
    assert captured["config"].sh_base_url == "https://sh.dataspace.copernicus.eu"
    assert captured["collection"].service_url == "https://sh.dataspace.copernicus.eu"
    assert captured["collection"].catalog_id == "sentinel-2-l2a"
    assert captured["time"] == ("2023-01-01", "2023-01-31")
    assert captured["bbox"].crs == sentinelhub.CRS.WGS84
    assert tuple(captured["bbox"]) == tuple(bbox)

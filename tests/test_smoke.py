"""Smoke tests for the package skeleton and configuration (P0-T01)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

import desertification
from desertification.config import (
    ConfigError,
    find_project_root,
    get_path,
    load_aois,
    load_config,
    validate_config,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_package_imports():
    assert desertification.__version__


def test_project_root_found():
    root = find_project_root()
    assert (root / "config" / "project.yaml").is_file()
    assert (root / "src" / "desertification").is_dir()


def test_config_loads_with_required_sections(cfg):
    for section in ("project", "periods", "grid", "sentinel2", "quota", "labels", "paths"):
        assert section in cfg
    assert isinstance(cfg["project"]["seed"], int)


def test_all_paths_resolve_inside_repo(cfg):
    root = Path(cfg["_root"]).resolve()
    for key in cfg["paths"]:
        assert get_path(cfg, key).is_relative_to(root), key


def test_unknown_path_key_raises(cfg):
    with pytest.raises(ConfigError):
        get_path(cfg, "does_not_exist")


def test_aois_is_feature_collection(cfg):
    aois = load_aois(cfg)
    assert aois["type"] == "FeatureCollection"
    assert isinstance(aois["features"], list)


def _raw(cfg):
    raw = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
    return raw


def test_overlapping_feature_and_label_periods_rejected(cfg):
    bad = _raw(cfg)
    bad["periods"]["early"]["end"] = "2023-06-30"
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_cell_size_must_be_multiple_of_resolution(cfg):
    bad = _raw(cfg)
    bad["grid"]["cell_size_m"] = 110
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_quota_caps_cannot_exceed_total(cfg):
    bad = _raw(cfg)
    bad["quota"]["pu_caps"]["reserve"] = 100_000
    with pytest.raises(ConfigError):
        validate_config(bad)

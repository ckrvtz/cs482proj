"""Tests for packing/unpacking the local data cache."""

from __future__ import annotations

import json
import zipfile

import pytest

from desertification.datacache import (
    MANIFEST_NAME,
    collect_files,
    is_safe_member,
    pack,
    plan_action,
    unpack,
)

DIRS = ["data/raw", "data/interim"]


def make_repo(root, files):
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


@pytest.fixture
def src(tmp_path):
    return make_repo(
        tmp_path / "src",
        {
            "data/raw/.gitkeep": b"",
            "data/raw/catalog/aral_2019.json": b'{"a": 1}',
            "data/raw/catalog/aral_2019.json.meta.json": b'{"request_sha256": "x"}',
            "data/raw/true_color/aral.tif": bytes(range(256)) * 40,
            "data/interim/grid.npz": b"npz-bytes",
            "data/processed/table.parquet": b"not part of the cache",
            ".env": b"SH_CLIENT_SECRET=never-packed",
        },
    )


def test_collect_files_only_cache_dirs_and_no_placeholders(src):
    files = collect_files(src, DIRS)
    assert files == [
        "data/interim/grid.npz",
        "data/raw/catalog/aral_2019.json",
        "data/raw/catalog/aral_2019.json.meta.json",
        "data/raw/true_color/aral.tif",
    ]


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("data/raw/catalog/x.json", True),
        ("data/interim/a/b.npz", True),
        ("data/raw", False),
        ("data/processed/x.parquet", False),
        ("../data/raw/x", False),
        ("data/raw/../../.env", False),
        ("/data/raw/x", False),
        ("C:/data/raw/x", False),
        ("data\\raw\\x", False),
        (".env", False),
        ("", False),
    ],
)
def test_is_safe_member(name, ok):
    assert is_safe_member(name, DIRS) is ok


def test_plan_action():
    assert plan_action(None, "a", False) == "write"
    assert plan_action("a", "a", False) == "skip"
    assert plan_action("a", "b", False) == "conflict"
    assert plan_action("a", "b", True) == "overwrite"


def test_pack_writes_manifest_and_refuses_to_clobber(src, tmp_path):
    archive = tmp_path / "data-cache.zip"
    manifest = pack(src, archive, DIRS, git_commit="abc123")
    assert manifest["n_files"] == 4 and manifest["git_commit"] == "abc123"
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert MANIFEST_NAME in names and ".env" not in names
        assert "data/processed/table.parquet" not in names
        assert json.loads(zf.read(MANIFEST_NAME))["n_files"] == 4
    assert not (tmp_path / "data-cache.zip.tmp").exists()
    with pytest.raises(FileExistsError):
        pack(src, archive, DIRS)
    pack(src, archive, DIRS, overwrite=True)


def test_round_trip_into_empty_repo(src, tmp_path):
    archive = tmp_path / "data-cache.zip"
    pack(src, archive, DIRS)
    dst = tmp_path / "dst"
    dry = unpack(archive, dst, DIRS, dry_run=True)
    assert len(dry.written) == 4 and not (dst / "data").exists()
    report = unpack(archive, dst, DIRS)
    assert len(report.written) == 4 and not report.conflicts and not report.rejected
    for rel in collect_files(src, DIRS):
        assert (dst / rel).read_bytes() == (src / rel).read_bytes()
    assert not list(dst.rglob("*.unpack-tmp"))
    again = unpack(archive, dst, DIRS)
    assert len(again.skipped_identical) == 4 and not again.written


def test_differing_local_file_is_kept_unless_overwrite(src, tmp_path):
    archive = tmp_path / "data-cache.zip"
    pack(src, archive, DIRS)
    dst = make_repo(tmp_path / "dst", {"data/raw/catalog/aral_2019.json": b"local edit"})
    report = unpack(archive, dst, DIRS)
    assert report.conflicts == ["data/raw/catalog/aral_2019.json"]
    assert (dst / "data/raw/catalog/aral_2019.json").read_bytes() == b"local edit"
    report = unpack(archive, dst, DIRS, overwrite=True)
    assert report.overwritten == ["data/raw/catalog/aral_2019.json"]
    assert (dst / "data/raw/catalog/aral_2019.json").read_bytes() == b'{"a": 1}'


def test_unsafe_and_unlisted_members_are_rejected(tmp_path):
    archive = tmp_path / "evil.zip"
    manifest = {"files": {"../outside.txt": {"size": 1, "sha256": "0"}}}
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../outside.txt", "x")
        zf.writestr("data/raw/unlisted.json", "{}")
        zf.writestr(MANIFEST_NAME, json.dumps(manifest))
    dst = tmp_path / "dst"
    report = unpack(archive, dst, DIRS)
    assert sorted(report.rejected) == ["../outside.txt", "data/raw/unlisted.json"]
    assert not (tmp_path / "outside.txt").exists() and not (dst / "data").exists()


def test_corrupt_archive_raises_and_writes_nothing_final(src, tmp_path):
    archive = tmp_path / "data-cache.zip"
    pack(src, archive, DIRS)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive) as zin, zipfile.ZipFile(bad, "w") as zout:
        for name in zin.namelist():
            data = zin.read(name)
            zout.writestr(name, b"tampered" if name.endswith("grid.npz") else data)
    with pytest.raises(ValueError, match="checksum"):
        unpack(bad, tmp_path / "dst", DIRS)
    assert not (tmp_path / "dst/data/interim/grid.npz").exists()


def test_archive_without_manifest_is_refused(tmp_path):
    archive = tmp_path / "plain.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("data/raw/x.json", "{}")
    with pytest.raises(ValueError, match="MANIFEST"):
        unpack(archive, tmp_path / "dst", DIRS)

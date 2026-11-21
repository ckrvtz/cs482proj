"""Pack and unpack the local data cache (``data/raw``, ``data/interim``) as one zip file.

The cache is gitignored, so it is moved between machines by hand (USB drive, file share).
``scripts/pack_data.py`` writes ``data-cache.zip`` at the repository root;
``scripts/unpack_data.py`` restores it. The archive holds a ``MANIFEST.json`` with each
file's size and SHA-256 so the unpacker can verify it. Unpacking never overwrites a file
whose content differs unless explicitly asked (CLAUDE.md: never overwrite ``data/raw``
without asking).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_NAME = "MANIFEST.json"
SKIP_NAMES = {".gitkeep", ".DS_Store", "Thumbs.db"}
MANIFEST_VERSION = 1


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path, cache_dirs: list[str]) -> list[str]:
    """Repo-relative POSIX paths of every file to archive, sorted.

    Parameters
    ----------
    root : Path
        Repository root.
    cache_dirs : list of str
        Repo-relative directories, e.g. ``["data/raw", "data/interim"]``.

    Returns
    -------
    list of str
        Files under those directories, excluding placeholders such as ``.gitkeep``.
    """
    files = []
    for rel_dir in cache_dirs:
        base = root / rel_dir
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.name not in SKIP_NAMES:
                files.append(path.relative_to(root).as_posix())
    return sorted(files)


def is_safe_member(name: str, cache_dirs: list[str]) -> bool:
    """True if a zip member name is a plain relative path inside one of ``cache_dirs``.

    Rejects absolute paths, drive letters, backslashes, ``..`` components and anything
    outside the cache directories (guards against "zip slip").
    """
    if not name or "\\" in name or name.startswith("/") or ":" in name.split("/")[0]:
        return False
    parts = PurePosixPath(name).parts
    if any(part in ("..", "") for part in parts):
        return False
    return any(
        parts[: len(PurePosixPath(d).parts)] == PurePosixPath(d).parts
        and len(parts) > len(PurePosixPath(d).parts)
        for d in cache_dirs
    )


def build_manifest(root: Path, files: list[str], git_commit: str | None = None) -> dict[str, Any]:
    """Manifest dict: creation time, git commit, and size + SHA-256 per file."""
    entries = {}
    for rel in files:
        path = root / rel
        entries[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "n_files": len(entries),
        "total_bytes": sum(e["size"] for e in entries.values()),
        "files": entries,
    }


def pack(
    root: Path,
    archive: Path,
    cache_dirs: list[str],
    git_commit: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write ``archive`` with every cache file plus ``MANIFEST.json``.

    The zip is written to a temporary name first and renamed at the end, so an
    interrupted run never leaves a half-written archive under the final name.

    Parameters
    ----------
    root : Path
        Repository root.
    archive : Path
        Destination zip file.
    cache_dirs : list of str
        Repo-relative directories to include.
    git_commit : str, optional
        Recorded in the manifest.
    overwrite : bool, default False
        Replace an existing archive.

    Returns
    -------
    dict
        The manifest.

    Raises
    ------
    FileExistsError
        If ``archive`` exists and ``overwrite`` is False.
    """
    if archive.exists() and not overwrite:
        raise FileExistsError(f"{archive} exists; pass overwrite=True to replace it")
    files = collect_files(root, cache_dirs)
    manifest = build_manifest(root, files, git_commit)
    tmp = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in files:
            zf.write(root / rel, arcname=rel)
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
    tmp.replace(archive)
    return manifest


@dataclass
class UnpackReport:
    """What :func:`unpack` did (or would do, in a dry run), by repo-relative path."""

    written: list[str] = field(default_factory=list)
    skipped_identical: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)  # existing file differs; left alone
    overwritten: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)  # unsafe or unlisted members


def plan_action(existing_sha: str | None, archive_sha: str, overwrite: bool) -> str:
    """Decide what to do with one file.

    Returns ``"write"`` (not present), ``"skip"`` (identical), ``"overwrite"`` (differs and
    overwriting allowed) or ``"conflict"`` (differs; leave the local file alone).
    """
    if existing_sha is None:
        return "write"
    if existing_sha == archive_sha:
        return "skip"
    return "overwrite" if overwrite else "conflict"


def unpack(
    archive: Path,
    root: Path,
    cache_dirs: list[str],
    overwrite: bool = False,
    dry_run: bool = False,
) -> UnpackReport:
    """Restore cache files from ``archive`` into the repository.

    Every member must be listed in the manifest and lie inside ``cache_dirs``; extracted
    bytes are checked against the manifest's SHA-256 before the file is moved into place.

    Parameters
    ----------
    archive : Path
        Zip written by :func:`pack`.
    root : Path
        Repository root.
    cache_dirs : list of str
        Allowed repo-relative directories.
    overwrite : bool, default False
        Replace local files whose content differs from the archive.
    dry_run : bool, default False
        Report what would happen without writing anything.

    Returns
    -------
    UnpackReport

    Raises
    ------
    ValueError
        If the archive has no manifest, or a file fails its checksum.
    """
    report = UnpackReport()
    with zipfile.ZipFile(archive) as zf:
        if MANIFEST_NAME not in zf.namelist():
            raise ValueError(f"{archive} has no {MANIFEST_NAME}; was it made by pack_data.py?")
        manifest = json.loads(zf.read(MANIFEST_NAME))
        listed = manifest.get("files", {})
        for name in zf.namelist():
            if name == MANIFEST_NAME or name.endswith("/"):
                continue
            if name not in listed or not is_safe_member(name, cache_dirs):
                report.rejected.append(name)
                continue
            target = root / name
            existing = sha256_file(target) if target.is_file() else None
            action = plan_action(existing, listed[name]["sha256"], overwrite)
            if action == "skip":
                report.skipped_identical.append(name)
                continue
            if action == "conflict":
                report.conflicts.append(name)
                continue
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".unpack-tmp")
                with zf.open(name) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                if sha256_file(tmp) != listed[name]["sha256"]:
                    tmp.unlink()
                    raise ValueError(f"checksum mismatch for {name}; archive is corrupt")
                tmp.replace(target)
            (report.overwritten if action == "overwrite" else report.written).append(name)
    return report

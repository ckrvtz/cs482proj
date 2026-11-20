"""Download cache with JSON provenance sidecars.

Every API result is written next to a ``<file>.meta.json`` sidecar that records the request
parameters, a hash of them, the time it was made and the sentinelhub version (CLAUDE.md,
"Sentinel Hub / data rules"). A cached file is reused only when its sidecar's request hash
matches the new request, so changing any parameter forces a fresh download instead of
silently returning stale data.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".meta.json"


def request_hash(params: dict[str, Any]) -> str:
    """Stable SHA-256 of request parameters (key order does not matter).

    Parameters
    ----------
    params : dict
        JSON-serialisable request parameters.

    Returns
    -------
    str
        Hex digest.
    """
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sidecar_path(path: Path) -> Path:
    """Return the sidecar path for a cached file (``x.json`` -> ``x.json.meta.json``)."""
    return path.with_name(path.name + SIDECAR_SUFFIX)


def build_sidecar(params: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble provenance metadata for a request.

    Parameters
    ----------
    params : dict
        Request parameters (must not contain credentials).
    extra : dict, optional
        Additional facts to record (e.g. result counts).

    Returns
    -------
    dict
        Sidecar content with ``request``, ``request_sha256``, ``created_utc`` and
        ``sentinelhub_version``.
    """
    try:
        import sentinelhub

        sh_version = sentinelhub.__version__
    except ImportError:  # pragma: no cover - sentinelhub is a pinned dependency
        sh_version = None
    sidecar = {
        "request": params,
        "request_sha256": request_hash(params),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "sentinelhub_version": sh_version,
    }
    if extra:
        sidecar.update(extra)
    return sidecar


def load_cached_json(path: Path, params: dict[str, Any]) -> Any | None:
    """Return cached JSON content if ``path`` and a matching sidecar exist, else None.

    Parameters
    ----------
    path : Path
        Cached data file.
    params : dict
        Parameters of the request about to be made.

    Returns
    -------
    object or None
        Parsed JSON, or None on a cache miss (missing file, missing sidecar, or a sidecar
        whose request hash differs).
    """
    meta_path = sidecar_path(path)
    if not (path.is_file() and meta_path.is_file()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("request_sha256") != request_hash(params):
        logger.info("Cache at %s was made with different parameters; ignoring it", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_with_sidecar(
    path: Path, data: Any, params: dict[str, Any], extra: dict[str, Any] | None = None
) -> Path:
    """Write ``data`` as JSON to ``path`` and its provenance sidecar next to it.

    Parameters
    ----------
    path : Path
        Destination file; parent folders are created.
    data : object
        JSON-serialisable result.
    params : dict
        Request parameters used to obtain ``data`` (no credentials).
    extra : dict, optional
        Additional provenance facts.

    Returns
    -------
    Path
        The sidecar path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    meta_path = sidecar_path(path)
    meta_path.write_text(json.dumps(build_sidecar(params, extra), indent=2), encoding="utf-8")
    return meta_path

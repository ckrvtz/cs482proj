"""Hardware helpers: device selection and CUDA architecture compatibility checks.

The project targets a local NVIDIA GTX 1080 Ti (Pascal, compute capability 6.1) and must
still run on CPU (CLAUDE.md, "Compute and storage"). ``torch`` is imported lazily so the
rest of the package works without it.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_ARCH_RE = re.compile(r"^(sm|compute)_(\d+)([a-z]?)$")


def parse_arch(arch: str) -> tuple[str, int, int]:
    """Split a CUDA arch tag such as ``"sm_61"`` or ``"compute_90"``.

    Parameters
    ----------
    arch : str
        Tag as returned by ``torch.cuda.get_arch_list()``. Suffixes such as the ``a`` in
        ``sm_90a`` are accepted and ignored.

    Returns
    -------
    tuple of (str, int, int)
        Kind (``"sm"`` = compiled binary, ``"compute"`` = PTX), major and minor version.
    """
    match = _ARCH_RE.match(arch.strip())
    if not match:
        raise ValueError(f"Unrecognised CUDA arch tag: {arch!r}")
    kind, digits, _suffix = match.groups()
    return kind, int(digits[:-1]), int(digits[-1])


def arch_list_supports(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """Return True if a build compiled for ``arch_list`` can run on a GPU of ``capability``.

    CUDA rules: a binary (``sm_XY``) runs on devices with the same major version X and a
    minor version >= Y; PTX (``compute_XY``) can be JIT-compiled for any device with
    capability >= X.Y.

    Parameters
    ----------
    capability : tuple of (int, int)
        Device compute capability, e.g. ``(6, 1)`` for a GTX 1080 Ti.
    arch_list : list of str
        Arch tags the build was compiled for, e.g. ``["sm_60", "sm_70"]``.

    Returns
    -------
    bool
    """
    major, minor = capability
    for arch in arch_list:
        kind, a_major, a_minor = parse_arch(arch)
        if kind == "sm" and a_major == major and a_minor <= minor:
            return True
        if kind == "compute" and (a_major, a_minor) <= (major, minor):
            return True
    return False


def select_device(prefer_gpu: bool = True) -> str:
    """Pick the torch device string: ``"cuda"`` if usable and preferred, else ``"cpu"``.

    Parameters
    ----------
    prefer_gpu : bool, default True
        Set False to force CPU (e.g. for debugging or exact CPU reproducibility).

    Returns
    -------
    str
        ``"cuda"`` or ``"cpu"``.
    """
    import torch

    if prefer_gpu and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        if arch_list_supports(capability, torch.cuda.get_arch_list()):
            return "cuda"
        logger.warning(
            "GPU %s (capability %s) is not supported by this torch build (%s); using CPU",
            torch.cuda.get_device_name(0),
            capability,
            torch.cuda.get_arch_list(),
        )
    return "cpu"

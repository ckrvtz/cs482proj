"""Spectral indices and features (ROADMAP §4.4). Inputs are surface reflectance in 0–1.

More indices are added in M1-T08; each formula has a unit test.
"""

from __future__ import annotations

import numpy as np


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN where a + b == 0."""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom != 0, (a - b) / denom, np.nan)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (B08 - B04) / (B08 + B04); unitless, -1..1."""
    return normalized_difference(nir, red)

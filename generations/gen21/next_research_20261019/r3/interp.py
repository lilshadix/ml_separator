"""Interpolation baselines on the read metals of one cell (no training at all).

PWL: piecewise-linear interpolation of the read log D values along a coordinate x (atomic number Z,
or the Shannon CN8 radius used by the project's physics basis -- declared before any run), constant
continuation beyond the extreme reads.  Read metals keep their exact readings (observable target).
PCHIP: the pre-declared shape-preserving control (Fritsch-Carlson monotone cubic between reads, same
constant continuation).  Every function returns a level vector (14,) with NaN at metals the cell
does not have; log SF(A/B) = v[A] - v[B].
"""
from __future__ import annotations

import numpy as np

from . import paths  # noqa: F401
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402
from gen20k.data import N_LN, Z_VEC  # noqa: E402

X_Z = Z_VEC.astype(float)
X_R = -np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES])   # minus radius: increasing along the series


def _coords(kind: str) -> np.ndarray:
    if kind == "Z":
        return X_Z
    if kind == "R":
        return X_R
    raise ValueError(kind)


def pwl_levels(y_masked: np.ndarray, metals, kind: str = "R") -> np.ndarray:
    x = _coords(kind)
    read = [m for m in range(N_LN) if np.isfinite(y_masked[m])]
    out = np.full(N_LN, np.nan)
    if not read:
        return out
    xr = x[read]
    order = np.argsort(xr)
    xr, yr = xr[order], y_masked[read][order]
    for m in metals:
        out[m] = float(np.interp(x[m], xr, yr))          # np.interp holds the end values outside the range
    for m in read:
        out[m] = y_masked[m]
    return out


def pchip_levels(y_masked: np.ndarray, metals, kind: str = "R") -> np.ndarray:
    from scipy.interpolate import PchipInterpolator
    x = _coords(kind)
    read = [m for m in range(N_LN) if np.isfinite(y_masked[m])]
    out = np.full(N_LN, np.nan)
    if len(read) < 3:
        return pwl_levels(y_masked, metals, kind)
    xr = x[read]
    order = np.argsort(xr)
    xr, yr = xr[order], y_masked[read][order]
    f = PchipInterpolator(xr, yr, extrapolate=False)
    for m in metals:
        v = f(x[m])
        out[m] = float(v) if np.isfinite(v) else float(yr[0] if x[m] < xr[0] else yr[-1])
    for m in read:
        out[m] = y_masked[m]
    return out

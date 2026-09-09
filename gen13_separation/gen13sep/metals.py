"""Lanthanide-side constants used as *fixed*, target-free basis inputs.

Sources (see gen13_separation/DATA_AUDIT.md §metal priors):
* Shannon crystal radii for CN 8 and CN 9 (Å), Jordan, Inorg. Chem. 2023 SI.
* Hydration free energies (kJ/mol), Kepp, J. Phys. Chem. A 2019, Table 1
  (Marcus-1991 column re-scaled; used only as a shape prior, never absolute).
* 4f electron count q for Ln3+ and Jørgensen's spin-pairing shape functions
  (Kawabe 2001 form): E1(q) and E3(q) are zero at q = 0, 7, 14 and encode the
  tetrad structure; they enter only as candidate basis functions.

Promethium is absent from the bundle and from every table here.
"""
from __future__ import annotations

import numpy as np

LANTHANIDES: tuple[str, ...] = ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy",
                                "Ho", "Er", "Tm", "Yb", "Lu")
ATOMIC_NUMBER: dict[str, int] = {"La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Sm": 62, "Eu": 63,
                                 "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69,
                                 "Yb": 70, "Lu": 71}
F_COUNT: dict[str, int] = {m: ATOMIC_NUMBER[m] - 57 for m in LANTHANIDES}      # 4f^q of Ln3+
SHANNON_RADIUS_CN8: dict[str, float] = {
    "La": 1.160, "Ce": 1.143, "Pr": 1.126, "Nd": 1.109, "Sm": 1.079, "Eu": 1.066, "Gd": 1.053,
    "Tb": 1.040, "Dy": 1.027, "Ho": 1.015, "Er": 1.004, "Tm": 0.994, "Yb": 0.985, "Lu": 0.977}
SHANNON_RADIUS_CN9: dict[str, float] = {
    "La": 1.216, "Ce": 1.196, "Pr": 1.179, "Nd": 1.163, "Sm": 1.132, "Eu": 1.120, "Gd": 1.107,
    "Tb": 1.095, "Dy": 1.083, "Ho": 1.072, "Er": 1.062, "Tm": 1.052, "Yb": 1.042, "Lu": 1.032}
HYDRATION_FREE_ENERGY_KJ: dict[str, float] = {   # Kepp 2019 scale
    "La": -3277, "Ce": -3332, "Pr": -3377, "Nd": -3412, "Sm": -3456, "Eu": -3476, "Gd": -3500,
    "Tb": -3549, "Dy": -3557, "Ho": -3602, "Er": -3627, "Tm": -3647, "Yb": -3688, "Lu": -3698}


def _standardise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std()


def jorgensen_e1(q: np.ndarray) -> np.ndarray:
    """Spin-pairing shape function (Jørgensen / Kawabe), zero at q = 0, 7, 14."""
    q = np.asarray(q, dtype=float)
    # Piecewise form of the E1 refined spin-pairing energy contribution (up to scale).
    out = np.zeros_like(q)
    first = q <= 7
    out[first] = -q[first] * (7 - q[first]) / 7.0
    out[~first] = -(q[~first] - 7) * (14 - q[~first]) / 7.0
    return out


def jorgensen_e3(q: np.ndarray) -> np.ndarray:
    """Orbital-orbital term, zero at q = 0, 3.5, 7, 10.5, 14 (quarter-shell tetrads)."""
    q = np.asarray(q, dtype=float)
    return np.sin(2 * np.pi * q / 7.0) * (7 - np.abs(q - 7)) / 7.0


def physics_basis(names: tuple[str, ...] = LANTHANIDES) -> dict[str, np.ndarray]:
    """Candidate centred basis functions over the 14 lanthanides (each mean zero)."""
    r8 = np.array([SHANNON_RADIUS_CN8[m] for m in names])
    r9 = np.array([SHANNON_RADIUS_CN9[m] for m in names])
    q = np.array([F_COUNT[m] for m in names], dtype=float)
    ghyd = np.array([HYDRATION_FREE_ENERGY_KJ[m] for m in names], dtype=float)
    z = _standardise(r8)
    basis = {
        "radius": z,
        "radius_sq": _standardise(z ** 2),
        "inv_radius": _standardise(1.0 / r8),
        "radius_cn9": _standardise(r9),
        "hydration": _standardise(ghyd),
        "gd_break": _standardise((q >= 7).astype(float)),
        "tetrad_e1": _standardise(jorgensen_e1(q)),
        "tetrad_e3": _standardise(jorgensen_e3(q)),
    }
    return {k: v - v.mean() for k, v in basis.items()}

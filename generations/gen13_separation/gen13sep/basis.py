"""Lanthanide-axis bases.  A cell's centred curve ``c(Ln) = log D(Ln) - mean_Ln log D``
is written as ``c = coef @ basis`` with a small number of basis curves over the 14
lanthanides; separation factors are differences of two entries of ``c``.

Two families:

* ``data``    — rank-K basis fitted on the *training* fold's centred matrix by weighted
                alternating least squares on the observed entries only (missing metals are
                never imputed and cannot drift), orthonormalised, with a fixed sign rule.
* ``physics`` — fixed, target-free curves from ``metals.physics_basis``: standardised
                Shannon radius, its square, the Gd break and the Jørgensen tetrad
                functions.  Chosen by name, frozen in the pre-registration.

Every basis row is scaled to the same norm (``sqrt(14)``, the norm of a standardised
curve) so that the per-cell ridge in ``fit_coefficients`` means the same shrinkage for
both families.  With that scale a ridge of 0.5 shrinks a fully observed cell's
coefficient by 0.5 / (14 + 0.5) = 3.4 %; a two-metal cell is shrunk much more, which
is intended — its curve is under-determined.
"""
from __future__ import annotations

import numpy as np

from .metals import LANTHANIDES, physics_basis

N_LN = len(LANTHANIDES)
ROW_NORM = float(np.sqrt(N_LN))
DEFAULT_RIDGE = 0.5


def centre_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix - np.nanmean(matrix, axis=1, keepdims=True)


def _normalise_rows(basis: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(basis, axis=1, keepdims=True)
    return basis / np.maximum(norms, 1e-12) * ROW_NORM


def _fix_signs(basis: np.ndarray) -> np.ndarray:
    """Sign rule: component 1 correlates positively with the radius curve (increasing La->Lu);
    every later component has its largest-magnitude entry positive."""
    table = physics_basis()
    ref = table["radius"]
    out = basis.copy()
    for k in range(out.shape[0]):
        if k == 0:
            if float(out[k] @ ref) < 0:
                out[k] *= -1
        else:
            j = int(np.argmax(np.abs(out[k])))
            if out[k, j] < 0:
                out[k] *= -1
    return out


def fit_data_basis(centred: np.ndarray, rank: int, *, iters: int = 200, ridge: float = 1e-3,
                   tol: float = 1e-6, return_info: bool = False):
    """Rank-``rank`` basis (rank x 14) from weighted ALS on observed entries only.

    Minimises ``sum_observed (C_ij - U_i . V_j)^2 + ridge (|U|^2 + |V|^2)`` by alternating
    ridge solves, starting from the SVD of the zero-filled matrix.  The returned basis is
    the orthonormalised row space of ``V``, scaled to row norm ``sqrt(14)``.
    """
    if rank < 1 or rank > N_LN:
        raise ValueError("rank out of range")
    mask = ~np.isnan(centred)
    C = np.where(mask, centred, 0.0)
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    Uf = U[:, :rank] * np.sqrt(s[:rank])
    V = (Vt[:rank].T * np.sqrt(s[:rank]))          # 14 x rank
    eye = np.eye(rank)
    prev = np.inf
    info = {"iterations": 0, "converged": False, "final_change": np.nan}
    for it in range(iters):
        # rows
        for i in range(C.shape[0]):
            m = mask[i]
            Vm = V[m]
            Uf[i] = np.linalg.solve(Vm.T @ Vm + ridge * eye, Vm.T @ C[i, m])
        # columns
        for j in range(N_LN):
            m = mask[:, j]
            Um = Uf[m]
            V[j] = np.linalg.solve(Um.T @ Um + ridge * eye, Um.T @ C[m, j])
        R = Uf @ V.T
        loss = float(((C - R)[mask] ** 2).sum())
        info["iterations"] = it + 1
        info["final_change"] = float(abs(prev - loss))
        if abs(prev - loss) < tol * max(loss, 1e-12):
            info["converged"] = True
            break
        prev = loss
    Q, _ = np.linalg.qr(V)                           # orthonormal column basis of span(V)
    basis = _fix_signs(_normalise_rows(Q.T))
    return (basis, info) if return_info else basis


def physics_basis_matrix(names: tuple[str, ...]) -> np.ndarray:
    table = physics_basis()
    unknown = [n for n in names if n not in table]
    if unknown:
        raise KeyError(f"unknown physics basis {unknown}; have {sorted(table)}")
    return _normalise_rows(np.vstack([table[n] for n in names]))


def fit_coefficients(centred_row: np.ndarray, basis: np.ndarray, *, ridge: float = DEFAULT_RIDGE) -> np.ndarray:
    """Ridge coefficients on the observed metals of one cell (rank,)."""
    m = ~np.isnan(centred_row)
    B = basis[:, m].T
    y = centred_row[m]
    k = basis.shape[0]
    return np.linalg.solve(B.T @ B + ridge * np.eye(k), B.T @ y)


def fit_all_coefficients(centred: np.ndarray, basis: np.ndarray, *, ridge: float = DEFAULT_RIDGE) -> np.ndarray:
    return np.vstack([fit_coefficients(centred[i], basis, ridge=ridge) for i in range(len(centred))])


def curves(coefficients: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.atleast_2d(coefficients) @ basis


def explained_variance(centred: np.ndarray, basis: np.ndarray, *, ridge: float = DEFAULT_RIDGE) -> float:
    """In-sample fraction of centred variance the basis reproduces on observed metals."""
    coef = fit_all_coefficients(centred, basis, ridge=ridge)
    rec = curves(coef, basis)
    mask = ~np.isnan(centred)
    sse = float(((centred - rec)[mask] ** 2).sum())
    sst = float((centred[mask] ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")

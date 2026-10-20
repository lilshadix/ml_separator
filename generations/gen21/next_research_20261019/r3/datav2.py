"""Data version V2 (pinned 2026-10-19 from the data-quality inventory, before any V2 run; DATA_QUALITY_REPORT.md).

V2 = V1 with source-verified quarantines and the resolution-floor censoring:
  V2-1  ORNL re-import duplicating a curated cell: the 5 DMDPhPDA "copy B" cells (bare mantissas without the
        column exponent; exact +4/+3/+2/+1 decade shifts against copy A) and 0ff595d576155f6c (stereo-free
        Me2-TODGA re-import averaging the cis and trans curves)                       -> whole cell removed
  V2-2  the same Table 1 entered twice, split only by a NaN contact time: the NaN member is removed
  V2-3  hidden-condition merges (archive complexant name / concentration or solvent mixture differ inside the
        cell): quarantined (whole cell removed; splitting would create cells absent from the fold plans)
  V2-5  resolution floor: readings equal to one unit of the finest decimal used in their cell, at the cell
        minimum and tied by >= 2 readings (D_raw strings)                              -> reading removed
Not in V2 (flag-only sensitivities, never used to select rows): REPCAP (replicate spread, label-derived),
the suspected low-D floor (log D <= -3.5), smoothness-only jumps, the suspected cross-acid ORNL copy.
Every rule is evidence-based on sources or fixed data-internal patterns; none uses a model, a residual or
curve smoothness.  ``verify(bench)`` re-checks the rules that can be re-derived from the bench itself.
"""
from __future__ import annotations

import numpy as np

from . import paths  # noqa: F401
from gen13sep.metals import LANTHANIDES  # noqa: E402

V2_1_CELLS = ("9754896d384a387f", "cd097fce666bd34a", "196797dc24f71fda", "8a78c3b1777b0625", "7054e94bc4c6833f",
              "0ff595d576155f6c")
DMD_COPY_A = {"9754896d384a387f": "f457e9a3d6532cc8", "cd097fce666bd34a": "e48388064c30fbb4",
              "196797dc24f71fda": "fbd013078d5396c3", "8a78c3b1777b0625": "575484a7b0e30ce4",
              "7054e94bc4c6833f": "ceb21275e22d7bc5"}
V2_2_DROP = {"c61ecb085d8df371": "5a6a2f3bf13a7d55", "f8ed87d3cf7f4281": "5df28757679cae26",
             "62e7cdd6fa003391": "247f57741141188a"}
V2_3_CELLS = ("b4fd2a49bcbf3d25", "a1d1bb9061317e70", "bd699b6a4b2c634b", "d5e6b9dc02e2364e", "372778033811af53",
              "71f1d37fa44d6ef0", "a3bc2e7b55a95970", "d294b06ac73b17a1", "48baeebc441c0964", "c41d53a5c2625ad2",
              "a9cb3ebbf14bf055", "754e9dee83f16dd1", "818f1d78902d2a6d", "b3438d0b792d8a58", "180c0a7827d406e0",
              "2cf8480c768c5553", "0ecd4ac4bf29bfb9")
V2_5_FLOOR = {"029b985df55ed873": ("Sm", "Eu"), "d294b06ac73b17a1": ("Sm", "Eu", "Gd", "Tb"),
              "4ab68efc604014ca": ("Ce", "Tb", "Ho", "Yb", "Lu"), "b33dc03c40457417": ("Eu", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"),
              "3bb58e5ef9b428b7": ("La", "Nd", "Eu", "Tb"), "786daf3e0a82df88": ("Ce", "Gd")}

_CACHE: dict = {}


def matrix(bench) -> np.ndarray:
    if "Y" in _CACHE:
        return _CACHE["Y"]
    ids = bench.frame["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    Y = bench.Y.copy()
    for c in V2_1_CELLS + tuple(V2_2_DROP) + V2_3_CELLS:
        Y[pos[c]] = np.nan
    for c, mets in V2_5_FLOOR.items():
        for m in mets:
            Y[pos[c], LANTHANIDES.index(m)] = np.nan
    _CACHE["Y"] = Y
    return Y


def verify(bench) -> dict:
    """Re-derive what the bench alone can show: DMDPhPDA copy B decade shifts, the NaN-split duplicates being
    bit-identical, the floor readings being tied cell minima with a one-unit decimal value."""
    f = bench.frame
    ids = f["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    out = {"dmd_shift_max_frac": 0.0, "dmd_shifts": {}, "dup_identical": {}, "floor_ok": {}}
    for b, a in DMD_COPY_A.items():
        d = bench.Y[pos[b]] - bench.Y[pos[a]]
        d = d[np.isfinite(d)]
        out["dmd_shift_max_frac"] = max(out["dmd_shift_max_frac"], float(np.max(np.abs(d - np.round(d)))))
        out["dmd_shifts"][b] = sorted(set(np.round(d).astype(int).tolist()))
    for drop, keep in V2_2_DROP.items():
        y1, y2 = bench.Y[pos[drop]], bench.Y[pos[keep]]
        sh = np.isfinite(y1) & np.isfinite(y2)
        out["dup_identical"][drop] = bool(sh.sum() >= 3 and np.array_equal(y1[sh], y2[sh]))
    for c, mets in V2_5_FLOOR.items():
        y = bench.Y[pos[c]]
        v = [y[LANTHANIDES.index(m)] for m in mets]
        out["floor_ok"][c] = bool(np.allclose(v, v[0]) and np.isclose(v[0], np.nanmin(y)) and len(v) >= 2)
    return out



def proxy(bench, Y):
    """Bench view with the V2 matrix (everything round 3 fits reads ``Y`` from it)."""
    from types import SimpleNamespace
    return SimpleNamespace(Y=Y, frame=bench.frame, frames=bench.frames, groups=bench.groups, n_obs=np.isfinite(Y).sum(1),
                           basis=bench.basis, coef=bench.coef, matrix=bench.matrix, columns=bench.columns)


def _cell_v2(cell, Y, cov, nv):
    from g21 import evaluate as E
    from gen20k import design as DS
    from gen20k.data import CellView
    mets = tuple(int(m) for m in cell["metals"] if np.isfinite(Y[cell["ci"]][m]))
    if not mets:
        return None
    cv = CellView(index=cell["ci"], cell_id=cell["cell_id"], extractant=cell["extractant"], chemotype=cell["chemotype"],
                  publication=cell["publication"], metals=mets)
    dz = DS.make_design(cv, "adjacent", "metal")
    if not dz.query or dz.k_max < 1:
        return None
    kmax = min(dz.k_max, 6)
    return dict(cell, metals=mets, query=tuple(dz.query), k_max=int(dz.k_max),
                seqs={"DOPT_Q": DS.support_sequence(dz, "DOPT_Q", kmax, cov=cov, noise_var=nv),
                      "UNIFORM": DS.support_sequence(dz, "UNIFORM", kmax)})


def v2_record(rec, Y):
    """The cached fold with every cell's metals / queries / k_max / read sequences recomputed on V2 (same cached
    training covariances and noise; cells left without queries dropped)."""
    out = dict(rec)
    o = dict(rec["outer"])
    o["cells"] = [c2 for c in rec["outer"]["cells"] if (c2 := _cell_v2(c, Y, o["cov"], 2 * o["s2"])) is not None]
    out["outer"] = o
    inner = []
    for blk in rec["inner"]:
        b = dict(blk)
        b["cells"] = [c2 for c in blk["cells"] if (c2 := _cell_v2(c, Y, blk["cov"], 2 * blk["s2"])) is not None]
        inner.append(b)
    out["inner"] = inner
    return out

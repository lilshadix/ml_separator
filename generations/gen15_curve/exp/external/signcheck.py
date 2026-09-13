"""Prove the two datasets' slopes live on the same axis with the same sign, three ways.

1. *Basis identity*: the extraction coefficient ``a`` is the coefficient of ``bench.basis[0]``;
   the logK slope is the coefficient of ``rz = (r - mean r) / sd r`` on the same
   ``SHANNON_RADIUS_CN8`` table.  If ``bench.basis[0] == rz`` elementwise, the axes are identical.
2. *Direction of the radius axis*: r decreases La -> Lu, so a negative coefficient means the
   quantity RISES toward the heavy end.  Checked by reconstructing a curve and reading it off.
3. *Chemical anchors*: aminopolycarboxylates (EDTA family) are known to bind heavy lanthanides more
   strongly in water; HDEHP/D2EHPA is the textbook heavy-selective extractant.  Both must come out
   with a negative coefficient on their own side.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
for p in (ROOT / "generations" / "gen15_curve", ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction"):
    sys.path.insert(0, str(p))

from gen15 import valuebench as V                                   # noqa: E402
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8         # noqa: E402

out: dict = {}
bench = V.load()
r = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES])
rz = (r - r.mean()) / r.std()

out["1_basis"] = {
    "max_abs_diff_basis0_vs_rz": float(np.abs(bench.basis[0] - rz).max()),
    "basis0_norm": float(np.linalg.norm(bench.basis[0])),
    "rz_norm": float(np.linalg.norm(rz)),
    "rz_La": float(rz[0]), "rz_Lu": float(rz[-1]),
    "metal_order": list(LANTHANIDES),
}

# a curve with a = -1 must rise from La to Lu
curve = (-1.0) * bench.basis[0]
out["2_direction"] = {
    "curve_with_a_minus1_La": float(curve[0]),
    "curve_with_a_minus1_Lu": float(curve[-1]),
    "negative_coefficient_means": "value rises toward the HEAVY end (Lu)",
}

# chemical anchors -------------------------------------------------------------------------
K = pd.read_parquet(HERE / "data" / "side_K_logk.parquet")
E = pd.read_parquet(HERE / "data" / "side_E_extraction.parquet")

anchors_k = {
    "EDTA": "O=C(O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O",
    "DTPA": "O=C(O)CN(CC(=O)O)CCN(CC(=O)O)CCN(CC(=O)O)CC(=O)O",
    "NTA": "O=C(O)CN(CC(=O)O)CC(=O)O",
    "HEDTA": "O=C(O)CN(CCO)CCN(CC(=O)O)CC(=O)O",
    "CDTA": "O=C(O)CN(CC(=O)O)C1CCCCC1N(CC(=O)O)CC(=O)O",
    "acetate": "CC(=O)O",
    "oxalate(ish) glycolate": "OCC(=O)O",
}
ak = {}
for name, smi in anchors_k.items():
    m = K[K.smiles == smi]
    ak[name] = None if m.empty else {"slope": float(m.slope.iat[0]), "n": int(m.n.iat[0]),
                                     "r2": float(m.r2.iat[0])}
out["3a_logk_anchors"] = ak

# HDEHP / D2EHPA and other classic heavy-selective acidic organophosphorus extractants
ae = {}
for name, sub in {"HDEHP/D2EHPA (P(=O)(O)OCC(CC)CCCC x2)": "OP(=O)(OCC(CC)CCCC)OCC(CC)CCCC",
                  "any phosphoric/phosphonic acid": None}.items():
    if sub is None:
        hit = E[E.smiles.str.contains(r"P\(=O\)\(O\)", regex=True)]
        ae[name] = {"n": int(len(hit)), "frac_heavy": float(hit.y_heavy.mean()) if len(hit) else None,
                    "mean_amp": float(hit.amp.mean()) if len(hit) else None}
    else:
        m = E[E.smiles == sub]
        ae[name] = None if m.empty else {"amp": float(m.amp.iat[0])}
out["3b_extraction_anchors"] = ae

# the five structures present on BOTH sides: their two coefficients side by side
shared = sorted(set(E.smiles) & set(K.smiles))
rows = []
for s in shared:
    e = E[E.smiles == s].iloc[0]
    k = K[K.smiles == s].iloc[0]
    rows.append({"smiles": s, "extraction_amp": float(e.amp), "logk_slope": float(k.slope),
                 "logk_n": int(k.n), "logk_r2": round(float(k.r2), 3),
                 "same_sign": bool(np.sign(e.amp) == np.sign(k.slope))})
out["3c_shared_structures"] = rows
out["3c_shared_agreement"] = float(np.mean([r["same_sign"] for r in rows])) if rows else None

(HERE / "signcheck.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=1))

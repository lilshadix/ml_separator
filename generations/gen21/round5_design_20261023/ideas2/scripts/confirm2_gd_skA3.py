#!/usr/bin/env python
"""SKEPTIC A for confirm-2, splice check (EXPLORATORY, seen cells).
Two HS cells (C5-BTBP) have a Gd..Lu segment that is a constant-shift copy (sd < 0.02) of a DGA series from another
publication, while their La..Eu segment is not -- i.e. a possible splice exactly between Eu and Gd, where the Gd
chord is measured.  This script searches the whole V1 bench for segment copies:
  heavy segment Gd..Lu (>= 5 shared metals) and light segment La..Eu (>= 4 shared), shift sd < 0.03,
  between cells of different extractants or publications,
flags C3 chord cells whose heavy segment copies another series but whose Eu..Tb triad is not entirely inside the
copied part, and recomputes the HS stratum and its power without them.
Writes only ideas2/scripts/confirm2_skA_out/.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parent / "confirm2_skA_out"
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import donor as DN  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k import data as D  # noqa: E402

LN = list(D.LANTHANIDES)
iEu, iGd, iTb = LN.index("Eu"), LN.index("Gd"), LN.index("Tb")
XR = np.asarray(interp.X_R, float)
TG = (XR[iGd] - XR[iEu]) / (XR[iTb] - XR[iEu])
BETA = -0.0923
CLIP = (-0.40, 0.20)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 40)


def seg_pairs(Y, idx, min_shared, thr=0.03):
    out = []
    n = len(Y)
    S = Y[:, idx]
    for i in range(n):
        for j in range(i + 1, n):
            m = np.isfinite(S[i]) & np.isfinite(S[j])
            if m.sum() < min_shared:
                continue
            dd = S[i, m] - S[j, m]
            sd = dd.std(ddof=1)
            if sd < thr:
                out.append((i, j, int(m.sum()), float(sd), float(dd.mean()), float(np.ptp(S[i, m]))))
    return out


def main() -> int:
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = f.cell_id.isin(set(ref.cell_id)).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    pub = f.publication_id.astype(str).to_numpy()
    name = f.extractant_name.astype(str).to_numpy()
    heavy = list(range(iGd, len(LN)))
    light = list(range(0, iEu + 1))
    hp = seg_pairs(Y, heavy, 5)
    lp = seg_pairs(Y, light, 4)
    lset = {(i, j) for i, j, *_ in lp}
    rows = []
    for i, j, ns, sd, mu, rng_ in hp:
        if ext[i] == ext[j] and pub[i] == pub[j]:
            continue
        # a flat segment matches anything flat: require some structure (range of the segment > 0.3)
        rows.append({"a": f.cell_id[i], "b": f.cell_id[j], "pub_a": pub[i], "pub_b": pub[j],
                     "name_a": name[i][:30], "name_b": name[j][:30], "dga_a": DN.is_dga(ext[i]), "dga_b": DN.is_dga(ext[j]),
                     "c3_a": bool(c3[i]), "c3_b": bool(c3[j]), "shared_heavy": ns, "sd": sd, "shift": mu, "range": rng_,
                     "light_also_copy": (i, j) in lset,
                     "a_has_EuGdTb": bool(np.isfinite(Y[i, [iEu, iGd, iTb]]).all()),
                     "b_has_EuGdTb": bool(np.isfinite(Y[j, [iEu, iGd, iTb]]).all()),
                     "a_has_Eu": bool(np.isfinite(Y[i, iEu])), "b_has_Eu": bool(np.isfinite(Y[j, iEu]))})
    hc = pd.DataFrame(rows)
    hc = hc[hc.range > 0.3]
    print(f"heavy-segment (Gd..Lu) constant-shift copies across extractant/publication with range > 0.3: {len(hc)}")
    hc.to_csv(OUT / "heavy_segment_copies.csv", index=False)
    # cross-extractant only, and a splice signature: one member has Eu, the other does not (series starts at Gd)
    sp = hc[(hc.name_a != hc.name_b) & (hc.a_has_Eu != hc.b_has_Eu)]
    print(f"   of which cross-extractant with one member lacking Eu (copied part starts at Gd): {len(sp)}")
    print(sp.round(3).to_string())
    # the chord cells involved (the member that HAS Eu, Gd, Tb -> its chord spans the splice)
    sp_cells = set(sp.loc[sp.a_has_EuGdTb, "a"]) | set(sp.loc[sp.b_has_EuGdTb, "b"])
    # any cross-extractant heavy copy, whichever member
    xc = hc[hc.name_a != hc.name_b]
    xc_cells = set(xc.loc[xc.a_has_EuGdTb, "a"]) | set(xc.loc[xc.b_has_EuGdTb, "b"])
    print(f"   chord cells spanning a splice: {len(sp_cells)}; chord cells in any cross-extractant heavy copy: {len(xc_cells)}")

    u = pd.read_csv(OUT / "all_units.csv", index_col=0)
    c = Y[:, iGd] - ((1 - TG) * Y[:, iEu] + TG * Y[:, iTb])
    d = pd.DataFrame({"cell_id": f.cell_id, "unit": f.publication_id.astype(str) + "|" + f.chemotype.astype(str),
                      "pub": pub, "c": c, "name": name})
    d = d[c3 & np.isfinite(c)].copy()
    d["cls"] = d.unit.map(u.cls)
    print("\nchord cells spanning a splice / in a cross-extractant heavy copy, by stratum:")
    d["splice"] = d.cell_id.isin(sp_cells)
    d["xcopy"] = d.cell_id.isin(xc_cells)
    print(d[d.xcopy | d.splice][["cell_id", "unit", "cls", "c", "splice", "xcopy", "name"]].round(3).to_string())
    print(d.groupby("cls")[["splice", "xcopy"]].sum().to_string())
    for lab, q in (("all chord cells", d), ("drop splice cells", d[~d.splice]), ("drop any cross-extractant heavy copy", d[~d.xcopy])):
        uu = q.groupby("unit").agg(c=("c", "mean"), cls=("cls", "first"), n=("c", "size"))
        uu["cw"] = uu.c.clip(*CLIP)
        hs = uu[uu.cls == "HS"].cw.to_numpy()
        msg = f"{lab}: HS units {len(hs)}, mean {hs.mean():+.3f}, sd {hs.std(ddof=1):.3f}"
        for G in (6, 8, 10):
            r_ = np.random.default_rng(20261023 + zlib.crc32(f"skA3|{lab}|{G}".encode()))
            X = hs[r_.integers(len(hs), size=(4000, G))]
            t = (X.mean(1) - BETA) / (X.std(1, ddof=1) / np.sqrt(G))
            msg += f"; boot power G={G} {float((t < SS.t.ppf(0.025, G - 1)).mean()):.2f}"
        print("   " + msg)
        # C5-BTBP unit only: chord of non-spliced cells
    c5 = d[d.pub == "pub_6a0ef8528c"]
    print(f"\nC5-BTBP pub_6a0ef8528c chord cells {len(c5)}: mean {c5.c.mean():+.3f}; spliced {int(c5.splice.sum())} "
          f"(mean {c5[c5.splice].c.mean():+.3f}); not spliced mean {c5[~c5.splice].c.mean():+.3f}")
    # which publications donate the heavy segments (all cross-extractant copies)
    print("\ncross-extractant heavy copies by publication pair:")
    print(xc.groupby(["pub_a", "pub_b", "name_a", "name_b"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

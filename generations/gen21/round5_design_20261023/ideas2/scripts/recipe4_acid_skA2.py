#!/usr/bin/env python
"""Skeptic A of recipe-4, second pass (EXPLORATORY, already-seen cells).  Reads only recipe4_skA_out/ rows + bench.

  G  RMSE per arm (log units) against the 0.06-0.12 per-measurement noise range; relative SSE reductions.
  H  same-cell comparison: the 30 interior cells of the de-duplicated variant, scored with and without the copy link.
  I  C5-BTBP sensitivity: rebuild the acid-only population including pub_6a0ef8528c.
  J  3-level design per publication, and the (n-1)-level leave-one-out frame restricted to the same targets.
  K  exterior distance: how far outside the measured range the exterior targets sit (log10 units).

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe4_acid_skA2.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES, Z_VEC, N_LN  # noqa: E402
import recipe4_acid_skA as S  # noqa: E402

OUT = S.OUT


def r2(g, col):
    y = g.y.to_numpy()
    return 1 - ((g[col] - g.y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main() -> int:
    T = pd.read_parquet(OUT / "recipe4_skA_rows.parquet")
    print("=== G. RMSE (log10 SF units) per arm; relative SSE vs MEAN")
    for var in ("all", "nosuspect"):
        for lab, g in (("interior", T[(T.variant == var) & T.inside]), ("exterior", T[(T.variant == var) & ~T.inside])):
            s = []
            for arm in ("MEAN", "NEAR_TIE", "LIN", "LINHALF", "LINCLIP", "LINHALF_CLIP"):
                rm = math.sqrt(((g[arm] - g.y) ** 2).mean())
                rel = ((g[arm] - g.y) ** 2).sum() / ((g.MEAN - g.y) ** 2).sum()
                s.append(f"{arm} {rm:.3f} ({rel:.2f})")
            print(f"  {var:9s} {lab:8s} sd(y) {g.y.std():.3f}: " + "; ".join(s))
    print("\n=== H. same 30 interior cells (de-duplicated population), copy link in vs out")
    ids30 = set(T[(T.variant == "nosuspect") & T.inside].cell_id)
    for var in ("all", "nosuspect"):
        g = T[(T.variant == var) & T.cell_id.isin(ids30)]
        g = g[g.inside] if var == "nosuspect" else g
        print(f"  {var:9s} cells {g.cell_id.nunique()} rows {len(g)}: MEAN {r2(g, 'MEAN'):.3f}, LINHALF_CLIP "
              f"{r2(g, 'LINHALF_CLIP'):.3f} (d {r2(g, 'LINHALF_CLIP') - r2(g, 'MEAN'):+.3f}), LIN {r2(g, 'LIN'):.3f}")
    moved = sorted(set(T[(T.variant == "all") & T.inside].cell_id) - ids30)
    print("  cells moved interior->exterior when the copy link is dropped:", moved)

    print("\n=== K. exterior distance outside the sibling range (log10 acid)")
    g = T[(T.variant == "all") & ~T.inside].groupby("cell_id").agg(dist=("dist_out", "first"), pub=("publication", "first"))
    print(g.dist.describe().round(3).to_dict())
    for q in (0.15, 0.3):
        sub = T[(T.variant == "all") & ~T.inside & (T.dist_out <= q)]
        if len(sub):
            print(f"  exterior with dist <= {q}: cells {sub.cell_id.nunique()}, MEAN {r2(sub, 'MEAN'):.3f}, LIN "
                  f"{r2(sub, 'LIN'):.3f}, LINHALF {r2(sub, 'LINHALF'):.3f}, LINHALF_CLIP {r2(sub, 'LINHALF_CLIP'):.3f}, "
                  f"NEAR {r2(sub, 'NEAR_TIE'):.3f}")

    # ---------------------------------------------------------------- I. C5 sensitivity
    print("\n=== I. C5-BTBP: does pub_6a0ef8528c have C3 targets with >= 2 acid-only siblings?")
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    acid = f[S.ACID].to_numpy(float)
    others = [c for c in f.columns if c.startswith("cond__") and c != S.ACID and c not in S.TRIVIAL]
    F = f[others].to_numpy(float)
    c3 = set(pd.read_parquet(paths.PRED / "REF_BPG.parquet", columns=["cell_id"]).cell_id.unique())
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    rows = []
    for i in np.where(pub == S.C5)[0]:
        t = ids[i]
        if t not in c3 or t in v2drop or not (np.isfinite(acid[i]) and acid[i] > 0):
            continue
        Mi = [m for m in range(N_LN) if np.isfinite(Y[i, m])]
        cover = np.all(np.isfinite(Y[:, Mi]), axis=1)
        sib = [j for j in range(len(ids)) if j != i and pub[j] == pub[i] and ext[j] == ext[i] and cover[j]
               and ids[j] not in v2drop and np.isfinite(acid[j]) and acid[j] > 0 and not np.isclose(acid[i], acid[j])
               and np.all((np.isnan(F[i]) & np.isnan(F[j])) | np.isclose(F[i], F[j]))]
        if len(sib) < 2:
            continue
        la_s = np.log10(acid[sib])
        lt = math.log10(acid[i])
        inside = la_s.min() - 1e-9 <= lt <= la_s.max() + 1e-9
        for p in range(len(Mi) - 1):
            a, b = Mi[p], Mi[p + 1]
            if Z_VEC[b] - Z_VEC[a] != 1:
                continue
            v = Y[sib, a] - Y[sib, b]
            rec = {"cell_id": t, "inside": inside, "y": Y[i, a] - Y[i, b]}
            rec.update(S.arm_values(la_s, v, lt, S.LAM))
            rows.append(rec)
    C = pd.DataFrame(rows)
    if len(C):
        print(f"  C5 rows {len(C)}, cells {C.cell_id.nunique()}, interior cells {C[C.inside].cell_id.nunique()}")
        TA = T[T.variant == "all"]
        for lab, msk_t, msk_c in (("interior", TA.inside, C.inside), ("exterior", ~TA.inside, ~C.inside)):
            gg = pd.concat([TA[msk_t][["y", "MEAN", "LIN", "LINHALF", "LINHALF_CLIP", "NEAR_TIE"]],
                            C[msk_c][["y", "MEAN", "LIN", "LINHALF", "LINHALF_CLIP", "NEAR_TIE"]]])
            cc = C[msk_c]
            if len(cc):
                print(f"  C5 alone {lab}: MEAN {r2(cc, 'MEAN'):.3f}, LINHALF_CLIP {r2(cc, 'LINHALF_CLIP'):.3f}, "
                      f"LIN {r2(cc, 'LIN'):.3f}, NEAR {r2(cc, 'NEAR_TIE'):.3f}")
            print(f"  with C5 {lab}: MEAN {r2(gg, 'MEAN'):.3f}, LINHALF_CLIP {r2(gg, 'LINHALF_CLIP'):.3f} "
                  f"(d {r2(gg, 'LINHALF_CLIP') - r2(gg, 'MEAN'):+.3f}), LIN {r2(gg, 'LIN'):.3f}, NEAR {r2(gg, 'NEAR_TIE'):.3f}")
    else:
        print("  none")

    # ---------------------------------------------------------------- J. 3-level design per publication
    print("\n=== J. 3-level design, per publication (re-built as in skA section E)")
    keep = np.array([(c not in v2drop) and (p != S.C5) for c, p in zip(ids, pub)])
    key = [f"{pub[i]}#{ext[i]}#" + "|".join("nan" if np.isnan(x) else f"{x:.6g}" for x in F[i]) for i in range(len(ids))]
    groups = {}
    for i in np.where(keep & np.isfinite(acid) & (acid > 0))[0]:
        groups.setdefault(key[i], []).append(i)
    bad = {c for s in S.SUSPECT for c in s}
    er = []
    for g, ii in groups.items():
        ii = [q for q in ii if ids[q] not in bad]
        la_all = np.round(np.log10(acid[ii]), 6)
        ul = np.unique(la_all)
        if len(ul) < 5:
            continue
        mid = ul[1:-1][int(np.argmin(np.abs(ul[1:-1] - np.median(ul))))]
        meas = sorted({ul[0], mid, ul[-1]})
        for a in range(N_LN - 1):
            if Z_VEC[a + 1] - Z_VEC[a] != 1:
                continue
            sf = Y[ii, a] - Y[ii, a + 1]
            ok = np.isfinite(sf)
            lv = {u: sf[ok & (la_all == u)] for u in ul}
            if any(len(lv[u]) == 0 for u in meas):
                continue
            xs = np.array(meas)
            ys = np.array([lv[u].mean() for u in xs])
            b = np.polyfit(xs, ys, 1)
            allm = np.array([lv[u].mean() for u in ul if len(lv[u])])
            for u in ul:
                if u in meas or len(lv[u]) == 0:
                    continue
                for yv in lv[u]:
                    er.append({"pub": g.split("#")[0], "group": g, "pair": f"{LANTHANIDES[a]}/{LANTHANIDES[a + 1]}",
                               "n_levels": len(ul), "acid_span": float(ul[-1] - ul[0]), "y": yv, "MEAN3": ys.mean(),
                               "LINHALF3": ys.mean() + 0.5 * (np.polyval(b, u) - ys.mean()),
                               "PWL3": float(np.interp(u, xs, ys)), "ALLMEAN_oracle": allm.mean()})
    E = pd.DataFrame(er)
    for pb, g in E.groupby("pub"):
        print(f"  {pb}: groups {g.group.nunique()}, levels {sorted(g.n_levels.unique())}, span {g.acid_span.max():.2f}, "
              f"rows {len(g)}, sd(y) {g.y.std():.3f}; RMSE MEAN3 {math.sqrt(((g.MEAN3 - g.y) ** 2).mean()):.3f}, "
              f"LINHALF3 {math.sqrt(((g.LINHALF3 - g.y) ** 2).mean()):.3f}, PWL3 {math.sqrt(((g.PWL3 - g.y) ** 2).mean()):.3f}, "
              f"oracle all-level mean {math.sqrt(((g.ALLMEAN_oracle - g.y) ** 2).mean()):.3f}; "
              f"pooled R2 MEAN3 {r2(g, 'MEAN3'):.3f}, LINHALF3 {r2(g, 'LINHALF3'):.3f}, PWL3 {r2(g, 'PWL3'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

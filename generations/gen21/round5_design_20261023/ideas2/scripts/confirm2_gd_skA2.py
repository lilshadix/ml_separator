#!/usr/bin/env python
"""SKEPTIC A for confirm-2, follow-up (EXPLORATORY, seen cells).  Reads confirm2_skA_out/all_units.csv written by
confirm2_gd_skA.py and the bench.
  F2  transfer of a stratum amplitude restricted to non-DGA cells, hold-out by publication and by cluster_cpg,
      cell-weighted and unit-weighted, with and without C5-BTBP / DMDPhPDA
  D2  steepness measured OUTSIDE the chord (slope over metals other than Eu, Gd, Tb) vs direction
  M   metadata-defined stratum (what the test would actually use) vs the slope-defined corpus stratum
  R   reads-only (DOPT_Q k=3) unit classification -> stratum mean
  N   the two cross-unit near-copy pairs touching HS units: full series side by side
Writes only ideas2/scripts/confirm2_skA_out/.
"""
from __future__ import annotations

import sys
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
from r4 import metrics as M4  # noqa: E402
from r4 import donor as DN  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k import data as D  # noqa: E402

LN = list(D.LANTHANIDES)
iEu, iGd, iTb = LN.index("Eu"), LN.index("Gd"), LN.index("Tb")
XR = np.asarray(interp.X_R, float)
TG = (XR[iGd] - XR[iEu]) / (XR[iTb] - XR[iEu])
BETA = -0.0923
CLIP = (-0.40, 0.20)
C5, DMD = "pub_6a0ef8528c", "pub_8415538291"
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 50)


def main() -> int:
    b = D.load_bench()
    f = b.frame.reset_index(drop=True).copy()
    Y = np.asarray(b.Y, float)
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    cl = M4.add_clusters(ref[["cell_id", "publication", "chemotype"]].drop_duplicates("cell_id"), None)
    f = f.merge(cl[["cell_id", "cluster_cpg"]], on="cell_id", how="left")
    c = Y[:, iGd] - ((1 - TG) * Y[:, iEu] + TG * Y[:, iTb])
    ext = np.full(len(Y), np.nan)
    full = np.full(len(Y), np.nan)
    for k in range(len(Y)):
        m = np.isfinite(Y[k])
        if m.sum() >= 5:
            full[k] = np.polyfit(XR[m], Y[k, m], 1)[0]
        m2 = m.copy()
        m2[[iEu, iGd, iTb]] = False
        if m2.sum() >= 3:
            ext[k] = np.polyfit(XR[m2], Y[k, m2], 1)[0]
    d = pd.DataFrame({"cell_id": f.cell_id, "pub": f.publication_id.astype(str), "chem": f.chemotype.astype(str),
                      "name": f.extractant_name.astype(str), "c": c, "slope": full, "ext": ext,
                      "dga": f.extractant.astype(str).map(DN.is_dga), "c3": f.cell_id.isin(set(ref.cell_id)),
                      "cl": f.cluster_cpg})
    d = d[d.c3 & np.isfinite(d.c)].copy()
    d["unit"] = d.pub + "|" + d.chem
    u = pd.read_csv(OUT / "all_units.csv", index_col=0)
    d["cls"] = d.unit.map(u.cls)
    nd = d[d.cls != "DGA"].copy()

    # ---------------- F2 transfer within non-DGA -----------------------------------------------------------
    print("[F2] non-DGA cells: predicting the held-out Gd chord (= Gd error of an amplitude a)")
    rows = []
    for scope, q in (("nonDGA", nd), ("nonDGA noC5", nd[nd.pub != C5]), ("nonDGA noC5 noDMD", nd[~nd.pub.isin([C5, DMD])])):
        for grp in ("pub", "cl"):
            recs = []
            for g, te in q.groupby(grp):
                tr = q[q[grp] != g]
                ut = tr.groupby("unit").agg(c=("c", "mean"), cls=("cls", "first"))
                ut["cw"] = ut.c.clip(*CLIP)
                a_nd = ut.cw.mean()
                a_hs = ut[ut.cls == "HS"].cw.mean()
                a_ls = ut[ut.cls == "LS"].cw.mean()
                for unit, tu in te.groupby("unit"):
                    k = tu.cls.iat[0]
                    a3 = a_hs if k == "HS" else a_ls
                    cc = tu.c.to_numpy()
                    recs.append({"g": g, "unit": unit, "cls": k, "n": len(cc),
                                 "e_beta": ((cc - BETA) ** 2).sum(), "e_nd": ((cc - a_nd) ** 2).sum(),
                                 "e_3": ((cc - a3) ** 2).sum(),
                                 "u_beta": ((cc - BETA) ** 2).mean(), "u_nd": ((cc - a_nd) ** 2).mean(),
                                 "u_3": ((cc - a3) ** 2).mean()})
            r = pd.DataFrame(recs)
            for k in ("all", "HS", "LS"):
                rr = r if k == "all" else r[r.cls == k]
                gg = rr.groupby("g")[["e_beta", "e_nd", "e_3"]].sum()
                rows.append({"scope": scope, "holdout": grp, "cls": k, "units": len(rr), "groups": len(gg),
                             "cell_rel_3_vs_beta": 1 - rr.e_3.sum() / rr.e_beta.sum(),
                             "cell_rel_3_vs_nd": 1 - rr.e_3.sum() / rr.e_nd.sum(),
                             "unit_rel_3_vs_beta": 1 - rr.u_3.sum() / rr.u_beta.sum(),
                             "unit_rel_3_vs_nd": 1 - rr.u_3.sum() / rr.u_nd.sum(),
                             "groups_3_better_beta": int((gg.e_3 < gg.e_beta).sum()),
                             "groups_3_better_nd": int((gg.e_3 < gg.e_nd).sum())})
    F2 = pd.DataFrame(rows)
    print(F2.round(3).to_string())
    F2.to_csv(OUT / "transfer_nonDGA.csv", index=False)

    # ---------------- D2 steepness outside the chord --------------------------------------------------------
    un = nd.groupby("unit").agg(c=("c", "mean"), ext=("ext", "mean"), slope=("slope", "mean"), cls=("cls", "first"))
    un["cw"] = un.c.clip(*CLIP)
    un["hs"] = (un.cls == "HS").astype(float)
    ok = np.isfinite(un.ext)
    print(f"\n[D2] non-DGA units with an external slope: {ok.sum()} of {len(un)}")
    for xc in ("ext", "slope"):
        rho, p = SS.spearmanr(un.loc[ok, xc], un.loc[ok, "cw"])
        print(f"   Spearman(cw, {xc}) {rho:+.3f} (p {p:.3f})")
    print(f"   sign(ext) == sign(full slope): {float((np.sign(un.ext[ok]) == np.sign(un.slope[ok])).mean()):.3f}")
    for xc in ("ext",):
        for form in ("signed", "abs"):
            x = un.loc[ok, xc].to_numpy()
            x = np.abs(x) if form == "abs" else x
            X = np.column_stack([np.ones(ok.sum()), un.loc[ok, "hs"], x])
            yv = un.loc[ok, "cw"].to_numpy()
            bb, *_ = np.linalg.lstsq(X, yv, rcond=None)
            e = yv - X @ bb
            s2 = e @ e / (len(yv) - 3)
            se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
            print(f"   cw ~ HS + {form} {xc}: HS {bb[1]:+.3f} (se {se[1]:.3f}); {xc} {bb[2]:+.4f} (se {se[2]:.4f})")
    # HS vs LS within matched |ext| (steep only)
    med = un.ext.abs().median()
    for lab, q in (("|ext| <= median", un[un.ext.abs() <= med]), ("|ext| > median", un[un.ext.abs() > med])):
        print(f"   {lab}: " + "; ".join(f"{k} n={len(g)} mean {g.cw.mean():+.3f}" for k, g in q.groupby("cls")))

    # ---------------- M metadata stratum -------------------------------------------------------------------
    def meta(name: str) -> str:
        n = name.lower()
        if "hydrogen phosphate" in n or "phosphoric" in n or "phosphinic" in n or "phosphonic" in n:
            return "acidic_OP"
        if "triazin" in n or "triazatricyclo" in n or "benzotriazin" in n:
            return "soft_N_BTP_BTBP"
        if "dicarboxamide" in n or "carboxamide" in n:
            return "NO_amide_PDA_phen"
        if "propanediamide" in n:
            return "malonamide"
        if "phosphoryl" in n:
            return "neutral_P_O"
        return "other"
    u2 = u[u.cls != "DGA"].copy()
    u2["meta"] = u2.name.astype(str).map(meta)
    print("\n[M] non-DGA units by metadata class x slope class (winsorised unit chords)")
    print(u2.groupby(["meta", "cls"]).cw.agg(["size", "mean"]).round(3).to_string())
    for lab, sel in (("metadata: acidic OP + BTP/BTBP", u2.meta.isin(["acidic_OP", "soft_N_BTP_BTBP"])),
                     ("metadata: acidic OP + BTP/BTBP + PDA/phen amides", u2.meta.isin(["acidic_OP", "soft_N_BTP_BTBP", "NO_amide_PDA_phen"])),
                     ("PDA/phen amides only (any direction)", u2.meta == "NO_amide_PDA_phen"),
                     ("slope rule HS (corpus stratum)", u2.cls == "HS"),
                     ("slope rule HS minus BTP/BTBP", (u2.cls == "HS") & (u2.meta != "soft_N_BTP_BTBP"))):
        v = u2[sel].cw
        print(f"   {lab}: units {len(v)}, mean {v.mean():+.3f}, sd {v.std(ddof=1):.3f}")
    u2.to_csv(OUT / "nonDGA_units_meta.csv")

    # ---------------- R reads-only classification ----------------------------------------------------------
    rr = ref[(ref.policy == "DOPT_Q") & (ref.k == 3)][["cell_id", "split_seed", "read_set"]].drop_duplicates()
    pos = {cid: i for i, cid in enumerate(f.cell_id)}
    unit_of = dict(zip(d.cell_id, d.unit))
    sg = []
    for cid, seed, rs in rr.itertuples(index=False):
        if cid not in unit_of:
            continue
        mets = np.array([LN.index(m) for m in str(rs).split(",")])
        y = Y[pos[cid], mets]
        okm = np.isfinite(y)
        sg.append({"unit": unit_of[cid], "seed": seed, "rs": np.polyfit(XR[mets[okm]], y[okm], 1)[0] > 0})
    sg = pd.DataFrame(sg)
    per_seed = sg.groupby(["seed", "unit"]).rs.mean().reset_index()
    out = []
    for seed, q in per_seed.groupby("seed"):
        q = q.set_index("unit").join(u[["cw", "cls"]])
        q = q[q.cls != "DGA"]
        hs = q[q.rs > 0.5]
        out.append({"seed": seed, "HS_units": len(hs), "HS_mean": hs.cw.mean(),
                    "LS_units": int((q.rs <= 0.5).sum()), "LS_mean": q[q.rs <= 0.5].cw.mean()})
    out = pd.DataFrame(out)
    print("\n[R] reads-only (DOPT_Q 4 reads) unit direction, per seed\n", out.round(3).to_string())

    # ---------------- N near-copy series ---------------------------------------------------------------------
    print("\n[N] near-copy pairs touching HS units (log D series)")
    pairs = (("354fe634113a078e", "c6571910bb94ca86"), ("326289072f176f80", "1b71b86063cd1faf"))
    for a, bb in pairs:
        ya, yb = Y[pos[a]], Y[pos[bb]]
        t = pd.DataFrame({a: ya, bb: yb, "diff": ya - yb}, index=LN).T
        print(f"   {a} ({f.extractant_name.iat[pos[a]]}, {f.publication_id.iat[pos[a]]}) vs {bb} "
              f"({f.extractant_name.iat[pos[bb]]}, {f.publication_id.iat[pos[bb]]})")
        print(t.round(3).to_string())
        print("   conditions a:", {k: v for k, v in f.iloc[pos[a]].items() if str(k).startswith("cond__") and pd.notna(v) and v not in (0, 0.0, False)})
        print("   conditions b:", {k: v for k, v in f.iloc[pos[bb]].items() if str(k).startswith("cond__") and pd.notna(v) and v not in (0, 0.0, False)})
    # all cells of the HS unit pub_723ceb6c2c|sc061 and pub_1845fd6492|sc015 and pub_40c355ae86|sc058
    for unit in ("pub_723ceb6c2c|sc061", "pub_1845fd6492|sc015", "pub_40c355ae86|sc058", "pub_401a06b4b7|sc012"):
        ids = d[d.unit == unit].cell_id
        for cid in ids:
            print(f"   {unit} {cid}: " + " ".join(f"{m}:{v:.2f}" for m, v in zip(LN, Y[pos[cid]]) if np.isfinite(v)))
            print("      cond:", {k: v for k, v in f.iloc[pos[cid]].items() if str(k).startswith("cond__") and pd.notna(v) and v not in (0, 0.0, False)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

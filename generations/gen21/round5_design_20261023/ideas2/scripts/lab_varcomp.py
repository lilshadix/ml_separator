#!/usr/bin/env python
"""E-LAB check C: is the within-publication shared per-metal pattern an analytical (laboratory-wide, common-mode)
fingerprint or chemistry?  EXPLORATORY and descriptive (labels of all bench cells are used; nothing is predicted).

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab_varcomp.py

Per-metal deviation d[i, m] = log D minus the cell's own least-squares quadratic in -r (CN8), cells with >= 6 metals,
outside the round-3 V2-1/2/3 lists (all 521 bench cells, not only C3).  The corpus-mean deviation of each metal is
removed first, so products estimate covariances around the universal pattern (Gd dip etc.).

Cell-pair classes: SS same publication + same extractant (other condition); SH same publication, other extractant,
same chemotype (homologue); SO same publication, other chemotype; XE other publication, same extractant; XC other
publication, same chemotype, other extractant; XX other publication, other chemotype.

Under a laboratory-wide analytical common-mode (same per-metal bias for every extractant measured in the lab),
cov(SO) ~ cov(SH) > cov(XX) = 0.  Under chemistry, cov(SH) > cov(SO) ~ cov(XC) and level does not matter by itself.
Under a level-dependent artefact (floor, additive background, interference), sharing concentrates where both cells sit
at similar, extreme levels, regardless of chemotype.

Also: the Ce deviation (Ce(IV) / analytical) and the per-publication pair breakdown of the E-LAB arm (lab_rows).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\lab_out")
pd.set_option("display.width", 250)
X = np.asarray(interp._coords("R"), float)
CLS = ("SS", "SH", "SO", "XE", "XC", "XX")


def main() -> int:
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    ok = np.array([c not in v2drop for c in ids]) & (np.isfinite(Y).sum(1) >= 6)
    idx = np.where(ok)[0]
    Dv = np.full_like(Y, np.nan)
    for i in idx:
        m = np.isfinite(Y[i])
        co = np.polyfit(X[m], Y[i, m], 2)
        Dv[i, m] = Y[i, m] - np.polyval(co, X[m])
    Dv = Dv[idx]
    Yl = Y[idx]
    mu = np.nanmean(Dv, 0)
    Dc = Dv - mu
    pub = f.publication_id.astype(str).to_numpy()[idx]
    ext = f.extractant.astype(str).to_numpy()[idx]
    chem = f.chemotype.astype(str).to_numpy()[idx]
    acid_hcl = f["cond__acid__hcl"].fillna(0).to_numpy()[idx] > 0
    n = len(idx)
    print(f"cells used: {n} (publications {len(set(pub))}); corpus-mean deviation removed: "
          + ", ".join(f"{LANTHANIDES[m]} {mu[m]:+.3f}" for m in range(14)))
    I, J = np.triu_indices(n, 1)
    sp, se, sc = pub[I] == pub[J], ext[I] == ext[J], chem[I] == chem[J]
    cls = np.full(len(I), "XX", dtype=object)
    cls[sp & se] = "SS"
    cls[sp & ~se & sc] = "SH"
    cls[sp & ~sc] = "SO"
    cls[~sp & se] = "XE"
    cls[~sp & ~se & sc] = "XC"
    pubpair = np.where(sp, pub[I], "")
    rows = []
    for m in range(14):
        a, b = Dc[I, m], Dc[J, m]
        fin = np.isfinite(a) & np.isfinite(b)
        var = np.nanmean(Dc[:, m] ** 2)
        la, lb = Yl[I, m], Yl[J, m]
        inw = (np.abs(la) <= 2) & (np.abs(lb) <= 2)
        near = np.abs(la - lb) <= 1
        for c in CLS:
            s = fin & (cls == c)
            if s.sum() == 0:
                continue
            prod = a[s] * b[s]
            r = {"metal": LANTHANIDES[m], "cls": c, "pairs": int(s.sum()), "rho_pairw": prod.mean() / var}
            if c in ("SS", "SH", "SO"):
                pp = pd.Series(prod).groupby(pubpair[s]).mean()
                r["rho_pubw"] = pp.mean() / var
                r["pubs"] = len(pp)
            for nm, msk in (("in", inw), ("out", ~inw), ("near", near), ("far", ~near)):
                ss = s & msk
                r[f"rho_{nm}"] = (a[ss] * b[ss]).mean() / var if ss.sum() >= 20 else np.nan
                r[f"n_{nm}"] = int(ss.sum())
            rows.append(r)
    R = pd.DataFrame(rows)
    print("\nshared fraction rho = mean product / variance, per metal and class (pair-weighted; pub-weighted for S*):")
    print(R.pivot(index="metal", columns="cls", values="rho_pairw").reindex(LANTHANIDES)[list(CLS)].round(3).to_string())
    print("\npub-weighted within-publication classes:")
    print(R[R.cls.isin(["SS", "SH", "SO"])].pivot(index="metal", columns="cls", values="rho_pubw").reindex(LANTHANIDES).round(3).to_string())
    print("\npairs per class (median over metals):", R.groupby("cls").pairs.median().to_dict(),
          "; pubs contributing (median):", R.groupby("cls").pubs.median().dropna().to_dict())
    # pooled over metals (variance-weighted: sum of products / sum of variances)
    pooled = []
    for c in CLS:
        num = den = 0.0
        out = {"cls": c}
        for nm in ("all", "in", "out", "near", "far"):
            num = den = 0.0
            for m in range(14):
                a, b = Dc[I, m], Dc[J, m]
                la, lb = Yl[I, m], Yl[J, m]
                s = np.isfinite(a) & np.isfinite(b) & (cls == c)
                if nm == "in":
                    s &= (np.abs(la) <= 2) & (np.abs(lb) <= 2)
                elif nm == "out":
                    s &= ~((np.abs(la) <= 2) & (np.abs(lb) <= 2))
                elif nm == "near":
                    s &= np.abs(la - lb) <= 1
                elif nm == "far":
                    s &= np.abs(la - lb) > 1
                if s.sum() == 0:
                    continue
                num += (a[s] * b[s]).sum()
                den += 0.5 * (a[s] ** 2 + b[s] ** 2).sum()
            out[f"r_{nm}"] = num / den if den > 0 else np.nan
        pooled.append(out)
    print("\npooled over metals: r = sum(d_i d_j) / sum((d_i^2 + d_j^2)/2) (a covariance-to-variance ratio on the "
          "pairs themselves):")
    print(pd.DataFrame(pooled).round(3).to_string(index=False))
    # metal groups: ICP-MS oxide victims vs non-victims
    victims = ["Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
    for c in ("SH", "SO", "SS"):
        g = R[R.cls == c].set_index("metal")
        print(f"{c}: mean rho at oxide-victim metals (Gd-Lu) {g.loc[victims, 'rho_pairw'].mean():+.3f} vs La-Eu "
              f"{g.drop(victims).rho_pairw.mean():+.3f}")
    # ---- Ce ----
    ce = Dv[:, 1]
    fin = np.isfinite(ce)
    print(f"\nCe deviation from own quadratic (raw, not centred): n {fin.sum()}, mean {np.nanmean(ce):+.3f}, median "
          f"{np.nanmedian(ce):+.3f}; > +0.3: {(ce > 0.3).sum()}, > +0.5: {(ce > 0.5).sum()}, < -0.3: {(ce < -0.3).sum()}")
    cl = Yl[:, 1]
    for nm, msk in (("Ce level < -2", fin & (cl < -2)), ("Ce level in [-2, 2]", fin & (np.abs(cl) <= 2)),
                    ("Ce level > 2", fin & (cl > 2)), ("HCl media", fin & acid_hcl), ("non-HCl", fin & ~acid_hcl)):
        print(f"   {nm}: n {msk.sum()}, mean {np.nanmean(ce[msk]):+.3f}, sd {np.nanstd(ce[msk]):.3f}, > +0.3: {(ce[msk] > 0.3).sum()}")
    top = pd.DataFrame({"pub": pub[fin], "ce": ce[fin], "ext": ext[fin]}).groupby("pub").agg(
        n=("ce", "size"), n_ext=("ext", "nunique"), mean=("ce", "mean"), sd=("ce", "std"))
    print("   publications with the largest |mean Ce deviation| (n >= 3):")
    print(top[top.n >= 3].reindex(top[top.n >= 3]["mean"].abs().sort_values(ascending=False).index).head(8).round(3).to_string())
    # ---- per-publication pair breakdown of the E-LAB arm ----
    T = pd.read_parquet(OUT / "lab_rows.parquet")
    X0 = T[(T.design == "BPG") & (T.n_MX > 0) & ~T.t_v2drop]
    X0 = X0.assign(g=(X0.B1 - X0.y) ** 2 - (X0.MX - X0.y) ** 2)
    tab = X0.pivot_table(index="publication", columns="pair", values="g", aggfunc="sum") / 5
    tab = tab.reindex(columns=[p for p in ["La/Ce", "Ce/Pr", "Pr/Nd", "Sm/Eu", "Eu/Gd", "Gd/Tb", "Tb/Dy", "Dy/Ho", "Ho/Er",
                                            "Er/Tm", "Tm/Yb", "Yb/Lu"] if p in tab.columns])
    tab["total"] = tab.sum(1)
    print("\nMX - B1 SSE gain per seed, publication x pair (POPX, BPG):")
    print(tab.sort_values("total", ascending=False).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

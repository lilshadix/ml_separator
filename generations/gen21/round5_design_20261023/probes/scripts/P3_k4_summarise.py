#!/usr/bin/env python
"""PROBE P3 -- summarise step (reads P3_out/rows.parquet, cells.parquet, fits.json written by P3_k4_compute.py).

EXPLORATORY: every number is computed on the 229 already-seen C3 cells and the 5 dev seeds.

R2 = round 4's primary: pooled strict-adjacent R2 inside each split seed (1 - SSE / sum (y - mean y)^2 over that
seed's rows), then the seed mean.  The query rows are identical at every rung and under every policy (C3, 1,802 per
seed): the k = 4 population row replaces the k = 3 prediction only for cells that admit a legal 5th read; the 51 cells
with k_max = 3 keep their k = 3 prediction (M18 "C3 min(K, kmax)" row).  C4 = the fixed cohort of cells with k_max >= 4
(identical rows at k = 3 and k = 4).

Clusters: round 4's cluster_cpg (r4.metrics.add_clusters); bootstrap = one shared draw matrix over clusters for all
seeds (gen20k.inference._draw_counts, seed 8675309, 2,000 draws); LOCO = leave one cluster out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

R4 = Path(r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, str(R4))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r4 import metrics as M4  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k import inference as INF  # noqa: E402

OUT = Path(__file__).resolve().parent / "P3_out"
KEY = ["design", "split_seed", "fold", "cell_id", "A", "B"]
CELLKEY = ["design", "split_seed", "cell_id"]
ARMS = ["B0", "B1", "HIST_D7", "B1_k3beta"]
C5 = "pub_6a0ef8528c"
PAIR_ORDER = ["La/Ce", "Ce/Pr", "Pr/Nd", "Sm/Eu", "Eu/Gd", "Gd/Tb", "Tb/Dy", "Dy/Ho", "Ho/Er", "Er/Tm", "Tm/Yb", "Yb/Lu"]
DET = ["DOPTQ", "MAXMIN_R", "BISECT_R", "MAXMIN_Z", "BISECT_Z", "BISECT_RMID", "HULL_BISECT"]
MAIN = ["DOPTQ", "MAXMIN_R", "BISECT_R", "RANDOM"]
REPS = 2000
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)
J: dict = {}


def f4(v):
    return float(np.round(v, 4))


# ----------------------------------------------------------------------------------------------
# load + identity checks
# ----------------------------------------------------------------------------------------------
rows = pd.read_parquet(OUT / "rows.parquet")
cells = pd.read_parquet(OUT / "cells.parquet")
fits = pd.DataFrame(json.loads((OUT / "fits.json").read_text()))
rows["pair"] = rows.A + "/" + rows.B
for a in ARMS:
    rows["se_" + a] = (rows[a] - rows.y) ** 2
k3 = rows[rows.k == 3].copy()
k4 = rows[rows.k == 4].copy()

print("=== identity checks ===")
fits["dbeta"] = (fits.beta_k3 - fits.beta_k3_stored).abs()
lam_eq = (fits.lambda_k3 == fits.lambda_k3_stored.replace({"inf": "inf"})).mean()
print(f"refit beta_k3 vs stored REF fitted json: max |diff| {fits.dbeta.max():.3e}; lambda equal in {lam_eq:.3f} of folds")
J["identity"] = {"beta_k3_max_abs_diff": float(fits.dbeta.max()), "lambda_k3_equal_share": float(lam_eq)}
for d in ("BPG", "BP"):
    ref = pd.read_parquet(R4 / "predictions" / f"REF_{d}.parquet")
    m = k3[k3.design == d].merge(ref[KEY + ["y", "B0", "B1", "HIST_D7", "read_set"]], on=KEY, suffixes=("", "_ref"))
    rs = cells[cells.design == d].set_index(["split_seed", "cell_id"]).read3
    rs_ok = (m.set_index(["split_seed", "cell_id"]).index.map(rs) == m.read_set).mean()
    diffs = {c: float((m[c] - m[c + "_ref"]).abs().max()) for c in ("y", "B0", "B1", "HIST_D7")}
    print(f"{d} k=3 vs REF_{d}.parquet: {len(m)} / {len(ref)} / {int((k3.design == d).sum())} rows matched; "
          f"read sets equal {rs_ok:.3f}; max |diff| {diffs}")
    r3 = pd.read_parquet(R4 / "predictions" / f"R3-B0_{d}.parquet")
    r3 = r3[(r3.k == 4) & (r3.k_max >= 3)]
    pk = cells[(cells.design == d) & (cells.pick_DOPTQ != "")][["split_seed", "cell_id", "pick_DOPTQ"]]
    sel = k4[k4.design == d].merge(pk, on=["split_seed", "cell_id"])
    sel = sel[sel.cand == sel.pick_DOPTQ]
    mm = sel.merge(r3[KEY + ["PWL_R", "read_set"]], on=KEY)
    order = {n: i for i, n in enumerate(D.LANTHANIDES)}
    c3map = cells[cells.design == d].set_index(["split_seed", "cell_id"]).read3
    mm["read4"] = [",".join(sorted(c3map[(s, c)].split(",") + [p], key=order.get)) for s, c, p in
                   zip(mm.split_seed, mm.cell_id, mm.cand)]
    print(f"{d} k=4 DOPT_Q-nested B0 vs R3-B0_{d}.parquet PWL_R (k=4): {len(mm)} / {len(r3)} rows matched; "
          f"max |diff| {float((mm.B0 - mm.PWL_R).abs().max()):.3e}; read sets equal {(mm.read4 == mm.read_set).mean():.3f}")
    J["identity"][d] = {"k3_rows_matched": len(m), "k3_max_abs_diff": diffs, "k3_readsets_equal": float(rs_ok),
                        "k4_dopt_rows_matched": len(mm), "k4_dopt_rows_in_R3B0": len(r3),
                        "k4_dopt_B0_max_abs_diff": float((mm.B0 - mm.PWL_R).abs().max()),
                        "k4_dopt_readsets_equal": float((mm.read4 == mm.read_set).mean())}

# beta table
print("\n=== B1 beta refit per outer fold (r4/refs.py fit_reference; k = 3 and k = 4) ===")
bt = fits.groupby("design").agg(beta_k3_mean=("beta_k3", "mean"), beta_k3_sd=("beta_k3", "std"),
                                beta_k4_mean=("beta_k4", "mean"), beta_k4_sd=("beta_k4", "std"),
                                beta_k4_min=("beta_k4", "min"), beta_k4_max=("beta_k4", "max"))
print(bt.round(4).to_string())
for d in ("BPG", "BP"):
    print(d, "lambda k3:", fits[fits.design == d].lambda_k3.value_counts().to_dict(),
          "lambda k4:", fits[fits.design == d].lambda_k4.value_counts().to_dict(),
          "train rows k3/k4 (mean):", int(fits[fits.design == d].n_rows_k3.mean()), int(fits[fits.design == d].n_rows_k4.mean()))
J["beta"] = bt.round(5).reset_index().to_dict("records")
J["lambda_k4"] = {d: fits[fits.design == d].lambda_k4.value_counts().to_dict() for d in ("BPG", "BP")}

# legal-set facts
c1 = cells[(cells.design == "BPG") & (cells.split_seed == cells.split_seed.min())]
print("\n=== legality ===")
print(f"C3 cells per seed {len(c1)}; admit a legal 5th read {int((c1.n_legal > 0).sum())}; "
      f"n_legal mean {c1[c1.n_legal > 0].n_legal.mean():.2f} (range {c1[c1.n_legal > 0].n_legal.min()}-{c1.n_legal.max()}); "
      f"loose rule mean {c1[c1.n_legal > 0].n_legal_loose.mean():.2f}; cells where loose > legal "
      f"{int((c1.n_legal_loose > c1.n_legal).sum())}")
q1 = k3[(k3.design == "BPG") & (k3.split_seed == k3.split_seed.min())]
print(f"rows per seed {len(q1)}; rows in cells with a 5th read {int((q1.k_max >= 4).sum())}")
J["legality"] = {"cells": len(c1), "cells_5th": int((c1.n_legal > 0).sum()), "rows": len(q1), "rows_5th": int((q1.k_max >= 4).sum()),
                 "n_legal_mean": float(c1[c1.n_legal > 0].n_legal.mean()), "n_loose_mean": float(c1[c1.n_legal > 0].n_legal_loose.mean()),
                 "cells_loose_gt_legal": int((c1.n_legal_loose > c1.n_legal).sum())}

# ----------------------------------------------------------------------------------------------
# clusters
# ----------------------------------------------------------------------------------------------
bench = D.load_bench()
cl = M4.add_clusters(k3[["chemotype", "publication"]].drop_duplicates().assign(_=0), bench.frame)
chem2cl = dict(zip(cl.chemotype, cl.cluster_cpg))
rows["cluster"] = rows.chemotype.map(chem2cl)
k3["cluster"] = k3.chemotype.map(chem2cl)
k4["cluster"] = k4.chemotype.map(chem2cl)
CLUSTERS = sorted(k3.cluster.unique())
W = INF._draw_counts(len(CLUSTERS), REPS, INF.SEED)
print(f"\nclusters (cluster_cpg): {len(CLUSTERS)}")

# ----------------------------------------------------------------------------------------------
# policy frames on identical rows
# ----------------------------------------------------------------------------------------------
COLS = ARMS + ["se_" + a for a in ARMS] + ["n_read", "hull_out", "a_read", "b_read"]


def population(sel: pd.DataFrame) -> pd.DataFrame:
    """k = 3 rows with the k = 4 rows of ``sel`` (one per key, cells with a 5th read) substituted."""
    b = k3.set_index(KEY).copy()
    b["rung"] = "k3"
    s = sel.set_index(KEY)
    assert s.index.is_unique and s.index.isin(b.index).all()
    for c in COLS:
        if c in s.columns:
            b[c] = b[c].astype(float)
            b.loc[s.index, c] = s[c].astype(float)
    b.loc[s.index, "rung"] = "k4"
    return b.reset_index()


def det_frame(pol: str) -> pd.DataFrame:
    pk = cells[cells["pick_" + pol] != ""][CELLKEY + ["pick_" + pol]].rename(columns={"pick_" + pol: "cand"})
    sel = k4.merge(pk, on=CELLKEY + ["cand"])
    if pol != "MAXMIN_R_LOOSE":
        assert sel.strict_legal.astype(bool).all()
    return population(sel)


def random_frame(loose: bool) -> pd.DataFrame:
    s = k4 if loose else k4[k4.strict_legal.astype(bool)]
    g = s.groupby(KEY)[["se_" + a for a in ARMS] + ["hull_out", "n_read", "a_read", "b_read"]].mean().reset_index()
    return population(g)


def oracle_frame() -> pd.DataFrame:
    s = k4[k4.strict_legal.astype(bool)]
    cs = s.groupby(CELLKEY + ["cand"]).se_B1.sum().reset_index()
    best = cs.loc[cs.groupby(CELLKEY).se_B1.idxmin(), CELLKEY + ["cand"]]
    return population(s.merge(best, on=CELLKEY + ["cand"]))


F = {"K3": population(k4.iloc[:0])}
for p in DET + ["MAXMIN_R_LOOSE"]:
    F[p] = det_frame(p)
F["RANDOM"] = random_frame(False)
F["RANDOM_LOOSE"] = random_frame(True)
F["ORACLE_B1"] = oracle_frame()
for n, fr in F.items():
    fr["cluster"] = fr.chemotype.map(chem2cl)
    fr["pair"] = fr.A + "/" + fr.B
    fr["density"] = fr.n_read / fr.n_metals
    assert len(fr) == len(k3), n


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------
def r2_seeds(fr: pd.DataFrame, arm: str) -> pd.Series:
    out = {}
    for s, g in fr.groupby("split_seed"):
        sst = ((g.y - g.y.mean()) ** 2).sum()
        out[s] = 1.0 - g["se_" + arm].sum() / sst if sst > 0 and len(g) >= 2 else np.nan
    return pd.Series(out)


def mse_seeds(fr, arm):
    return fr.groupby("split_seed")["se_" + arm].mean()


def sums(fr: pd.DataFrame, arm: str):
    """per seed: arrays over CLUSTERS of n, Sy, Syy, SSE."""
    out = {}
    for s, g in fr.groupby("split_seed"):
        a = g.groupby("cluster").agg(n=("y", "size"), sy=("y", "sum"), syy=("y", lambda v: (v ** 2).sum()),
                                     sse=("se_" + arm, "sum")).reindex(CLUSTERS).fillna(0.0)
        out[s] = a
    return out


def boot_r2(fr, arm):
    S = sums(fr, arm)
    draws, loco = [], []
    for s, a in S.items():
        n, sy, syy, sse = (a[c].to_numpy(float) for c in ("n", "sy", "syy", "sse"))
        wn = W @ n
        sst = W @ syy - (W @ sy) ** 2 / np.maximum(wn, 1e-12)
        draws.append(1 - (W @ sse) / sst)
        lo = []
        for gi in range(len(CLUSTERS)):
            k = np.ones(len(CLUSTERS), bool)
            k[gi] = False
            nn = n[k].sum()
            ss = syy[k].sum() - sy[k].sum() ** 2 / nn if nn > 1 else np.nan
            lo.append(1 - sse[k].sum() / ss if nn > 1 and ss > 0 else np.nan)
        loco.append(lo)
    return np.mean(np.vstack(draws), 0), np.nanmean(np.array(loco, float), 0)


def summary_row(fr, arm, label, design, cohort):
    r = r2_seeds(fr, arm)
    bd, lo = boot_r2(fr, arm)
    return {"design": design, "cohort": cohort, "label": label, "arm": arm, "R2": r.mean(), "R2_min": r.min(), "R2_max": r.max(),
            "seeds_ge_050": int((r >= 0.5).sum()), "MSE": mse_seeds(fr, arm).mean(), "P_boot_ge_050": float((bd >= 0.5).mean()),
            "boot_lo": float(np.quantile(bd, 0.025)), "boot_hi": float(np.quantile(bd, 0.975)),
            "loco_min": float(np.nanmin(lo)), "loco_max": float(np.nanmax(lo)),
            "per_seed": ";".join(f"{v:.4f}" for v in r.to_numpy()), "rows_per_seed": len(fr) / fr.split_seed.nunique()}


def contrast(f1, a1, f2, a2):
    """R2(f2, a2) - R2(f1, a1) on identical rows; seed mean, cluster bootstrap percentile interval, LOCO range."""
    r1, r2 = r2_seeds(f1, a1), r2_seeds(f2, a2)
    b1, l1 = boot_r2(f1, a1)
    b2, l2 = boot_r2(f2, a2)
    d = r2 - r1
    db = b2 - b1
    dl = l2 - l1
    return {"delta": float(d.mean()), "seeds_pos": int((d > 0).sum()), "lo": float(np.quantile(db, 0.025)),
            "hi": float(np.quantile(db, 0.975)), "loco_min": float(np.nanmin(dl)), "loco_max": float(np.nanmax(dl))}


LADDER = ["K3"] + MAIN + ["MAXMIN_Z", "BISECT_Z", "BISECT_RMID", "HULL_BISECT", "MAXMIN_R_LOOSE", "RANDOM_LOOSE", "ORACLE_B1"]
summ = []
for d in ("BPG", "BP"):
    for excl in (False, True):
        for coh in ("C3", "C4"):
            for p in LADDER:
                fr = F[p][F[p].design == d]
                if excl:
                    fr = fr[fr.publication != C5]
                if coh == "C4":
                    fr = fr[fr.k_max >= 4]
                for arm in ARMS:
                    if p == "K3" and arm == "B1_k3beta":
                        continue
                    summ.append({**summary_row(fr, arm, p, d, coh + ("-noC5" if excl else "")), "policy": p})
S = pd.DataFrame(summ)
S.to_csv(OUT / "ladder_summary.csv", index=False)


def show(design, cohort, arms=("B1", "B0", "HIST_D7"), labels=LADDER):
    t = S[(S.design == design) & (S.cohort == cohort) & S.arm.isin(arms) & S.label.isin(labels)]
    print(t[["label", "arm", "R2", "R2_min", "R2_max", "seeds_ge_050", "P_boot_ge_050", "boot_lo", "boot_hi", "loco_min",
             "loco_max", "MSE", "per_seed"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


for d in ("BPG", "BP"):
    for coh in ("C3", "C4", "C3-noC5", "C4-noC5"):
        print(f"\n=== ladder {d} {coh} (rows/seed {S[(S.design == d) & (S.cohort == coh)].rows_per_seed.iloc[0]:.0f}) ===")
        show(d, coh)
print("\n=== B1 with the k = 3 beta at k = 4 (sensitivity), BPG / BP C3 ===")
for d in ("BPG", "BP"):
    show(d, "C3", arms=("B1_k3beta",), labels=MAIN)

# ---- random policy: Monte Carlo distribution of R2 over joint draws (B1) ----
print("\n=== RANDOM legal 5th read: exact expectation and Monte Carlo over joint draws (B1; 2,000 draws) ===")
rng = np.random.default_rng(20261023)
J["random_mc"] = {}
for d in ("BPG", "BP"):
    for excl in (False, True):
        s4 = k4[(k4.design == d) & k4.strict_legal.astype(bool)]
        b3 = k3[k3.design == d]
        if excl:
            s4, b3 = s4[s4.publication != C5], b3[b3.publication != C5]
        draws, ge = [], []
        for s in sorted(b3.split_seed.unique()):
            g3 = b3[b3.split_seed == s]
            sst = ((g3.y - g3.y.mean()) ** 2).sum()
            fixed = g3[g3.k_max < 4].se_B1.sum()
            cs = s4[s4.split_seed == s].groupby(["cell_id", "cand"]).se_B1.sum()
            tot = np.full(REPS, fixed)
            for c, v in cs.groupby(level=0):
                vv = v.to_numpy()
                tot += vv[rng.integers(len(vv), size=REPS)]
            r = 1 - tot / sst
            draws.append(r)
            ge.append(float((r >= 0.5).mean()))
        dm = np.mean(np.vstack(draws), 0)
        key = f"{d}{'-noC5' if excl else ''}"
        J["random_mc"][key] = {"mean": float(dm.mean()), "sd_seedmean": float(dm.std()), "q05": float(np.quantile(dm, 0.05)),
                               "q95": float(np.quantile(dm, 0.95)), "P_seedmean_ge_050": float((dm >= 0.5).mean()),
                               "P_seed_ge_050": [round(x, 3) for x in ge], "E_n_seeds_ge_050": float(sum(ge))}
        print(key, json.dumps(J["random_mc"][key]))

# ---- paired contrasts (B1 unless stated) ----
print("\n=== paired contrasts on identical rows (seed mean; cluster bootstrap 95 % percentile; LOCO range) ===")
CT = []
specs = [("k4 DOPTQ - k3", "K3", "B1", "DOPTQ", "B1"), ("k4 MAXMIN_R - k3", "K3", "B1", "MAXMIN_R", "B1"),
         ("k4 BISECT_R - k3", "K3", "B1", "BISECT_R", "B1"), ("k4 RANDOM - k3", "K3", "B1", "RANDOM", "B1"),
         ("MAXMIN_R - DOPTQ (k4)", "DOPTQ", "B1", "MAXMIN_R", "B1"), ("BISECT_R - DOPTQ (k4)", "DOPTQ", "B1", "BISECT_R", "B1"),
         ("RANDOM - DOPTQ (k4)", "DOPTQ", "B1", "RANDOM", "B1"), ("HULL_BISECT - DOPTQ (k4)", "DOPTQ", "B1", "HULL_BISECT", "B1"),
         ("B1 - B0 (k4 DOPTQ)", "DOPTQ", "B0", "DOPTQ", "B1"), ("B1 - HIST_D7 (k4 DOPTQ)", "DOPTQ", "HIST_D7", "DOPTQ", "B1"),
         ("B1 refit k4 - B1 k3 beta (k4 DOPTQ)", "DOPTQ", "B1_k3beta", "DOPTQ", "B1"),
         ("B1 - B0 (k3)", "K3", "B0", "K3", "B1")]
for d in ("BPG", "BP"):
    for excl in (False, True):
        for coh in ("C3", "C4"):
            for name, p1, a1, p2, a2 in specs:
                if coh == "C4" and not name.startswith("k4"):
                    continue
                f1, f2 = F[p1][F[p1].design == d], F[p2][F[p2].design == d]
                if excl:
                    f1, f2 = f1[f1.publication != C5], f2[f2.publication != C5]
                if coh == "C4":
                    f1, f2 = f1[f1.k_max >= 4], f2[f2.k_max >= 4]
                CT.append({"design": d, "cohort": coh + ("-noC5" if excl else ""), "contrast": name, **contrast(f1, a1, f2, a2)})
CT = pd.DataFrame(CT)
CT.to_csv(OUT / "contrasts.csv", index=False)
print(CT.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

# ---- per pair type ----
print("\n=== per pair type R2 (B1; seed mean of per-seed pooled R2 within the pair type), C3 rows ===")
PP = []
for d in ("BPG", "BP"):
    for excl in (False, True):
        for p in ["K3"] + MAIN + ["HULL_BISECT", "ORACLE_B1"]:
            fr = F[p][F[p].design == d]
            if excl:
                fr = fr[fr.publication != C5]
            for pt in PAIR_ORDER:
                g = fr[fr.pair == pt]
                for arm in ("B1", "B0"):
                    r = r2_seeds(g, arm)
                    PP.append({"design": d, "noC5": excl, "policy": p, "pair": pt, "arm": arm, "R2": r.mean(),
                               "rows_per_seed": len(g) / g.split_seed.nunique(), "hull_out_share": g.hull_out.mean(),
                               "SSE_share": g["se_" + arm].sum() / fr["se_" + arm].sum()})
PP = pd.DataFrame(PP)
PP.to_csv(OUT / "per_pair.csv", index=False)
for d in ("BPG", "BP"):
    for excl in (False, True):
        t = PP[(PP.design == d) & (PP.noC5 == excl) & (PP.arm == "B1")].pivot(index="pair", columns="policy", values="R2")
        t = t.loc[PAIR_ORDER, ["K3"] + MAIN + ["HULL_BISECT", "ORACLE_B1"]]
        rp = PP[(PP.design == d) & (PP.noC5 == excl) & (PP.arm == "B1") & (PP.policy == "K3")].set_index("pair").rows_per_seed
        t.insert(0, "rows/seed", rp.loc[PAIR_ORDER])
        print(f"-- {d}{' without C5-BTBP' if excl else ''} --")
        print(t.to_string(float_format=lambda v: f"{v:.3f}"))

# ---- per cluster ----
print("\n=== per cluster (cluster_cpg), B1, C3 rows: within-cluster pooled R2 (seed mean) ===")
PC = []
for d in ("BPG", "BP"):
    for c in CLUSTERS:
        base = F["K3"][(F["K3"].design == d) & (F["K3"].cluster == c)]
        rec = {"design": d, "cluster": c, "cells": base.cell_id.nunique(), "rows_per_seed": len(base) / 5,
               "cells_5th": base[base.k_max >= 4].cell_id.nunique(),
               "pubs": base.publication.nunique(), "chemotypes": ",".join(sorted(base.chemotype.unique()))[:40],
               "has_C5": bool((base.publication == C5).any())}
        for p in ["K3"] + MAIN:
            g = F[p][(F[p].design == d) & (F[p].cluster == c)]
            rec[f"R2_{p}"] = r2_seeds(g, "B1").mean()
            rec[f"MSE_{p}"] = mse_seeds(g, "B1").mean()
        # LOCO effect on pooled R2
        for p in ("K3", "DOPTQ"):
            g = F[p][(F[p].design == d) & (F[p].cluster != c)]
            rec[f"pooled_without_{p}"] = r2_seeds(g, "B1").mean()
        PC.append(rec)
PC = pd.DataFrame(PC).sort_values(["design", "rows_per_seed"], ascending=[True, False])
PC.to_csv(OUT / "per_cluster.csv", index=False)
print(PC.drop(columns=[c for c in PC.columns if c.startswith("MSE_")]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---- density-indexed view ----
print("\n=== density-indexed view (B1): pooled R2 of rows grouped by their cell's reads per measured metal (seed mean) ===")
DBINS = [0, 0.29, 0.32, 0.37, 0.45, 0.55, 0.70, 1.01]
DLAB = ["0.286 (4/14)", "0.29-0.32 (4/13)", "0.32-0.37 (5/14,4/12,4/11)", "0.37-0.45 (5/13,4/10,4/9,5/12)",
        "0.45-0.55 (5/11,4/8,5/10)", "0.55-0.70 (5/9,4/7,5/8,4/6)", ">0.70 (5/7,4/5,5/6)"]
DN = []
for d in ("BPG", "BP"):
    for excl in (False, True):
        srcs = {"4 reads (K3 rows)": F["K3"]}
        for p in MAIN:
            srcs[f"5 reads ({p} k4 rows)"] = F[p][F[p].rung == "k4"]
        srcs["pooled 4+5 (K3 + DOPTQ k4 rows)"] = pd.concat([F["K3"], F["DOPTQ"][F["DOPTQ"].rung == "k4"]])
        for name, fr in srcs.items():
            fr = fr[fr.design == d]
            if excl:
                fr = fr[fr.publication != C5]
            fr = fr.assign(dbin=pd.cut(fr.density, DBINS, labels=DLAB, right=False))
            for b_, g in fr.groupby("dbin", observed=True):
                DN.append({"design": d, "noC5": excl, "source": name, "density_bin": b_, "R2": r2_seeds(g, "B1").mean(),
                           "R2_B0": r2_seeds(g, "B0").mean(), "rows_per_seed": len(g) / 5,
                           "clusters": g.cluster.nunique(), "mean_density": g.density.mean()})
DN = pd.DataFrame(DN)
DN.to_csv(OUT / "density.csv", index=False)
for d in ("BPG", "BP"):
    for excl in (False, True):
        t = DN[(DN.design == d) & (DN.noC5 == excl)]
        print(f"-- {d}{' without C5-BTBP' if excl else ''}: R2 (rows/seed, clusters) --")
        t = t.assign(cell=[f"{r:.3f} ({n:.0f}, {c})" for r, n, c in zip(t.R2, t.rows_per_seed, t.clusters)])
        print(t.pivot(index="density_bin", columns="source", values="cell").reindex(DLAB).fillna("").to_string())
print("\n=== population view: mean reads per measured metal over C3 rows vs pooled R2 (B1) ===")
PV = []
for d in ("BPG", "BP"):
    for excl in (False, True):
        for coh in ("C3", "C4"):
            for p in ["K3"] + MAIN + ["HULL_BISECT", "ORACLE_B1"]:
                fr = F[p][F[p].design == d]
                if excl:
                    fr = fr[fr.publication != C5]
                if coh == "C4":
                    fr = fr[fr.k_max >= 4]
                PV.append({"design": d, "noC5": excl, "cohort": coh, "policy": p, "mean_density_rows": fr.density.mean(),
                           "mean_reads_per_cell": fr.groupby(["split_seed", "cell_id"]).n_read.first().mean(),
                           "hull_out_share": fr.hull_out.mean(), "R2_B1": r2_seeds(fr, "B1").mean()})
PV = pd.DataFrame(PV)
PV.to_csv(OUT / "population_density.csv", index=False)
print(PV.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\n=== saturation (reads = k_max + 1, every other metal read): B1 R2 by k_max stratum ===")
SA = []
for d in ("BPG", "BP"):
    for excl in (False, True):
        parts = {"k_max 3 @ k=3": F["K3"][F["K3"].k_max == 3]}
        for p in MAIN:
            parts[f"k_max 4 @ k=4 {p}"] = F[p][(F[p].k_max == 4)]
            parts[f"k_max 5 @ k=4 {p} (not saturated)"] = F[p][(F[p].k_max == 5)]
            parts[f"k_max 6 @ k=4 {p} (not saturated)"] = F[p][(F[p].k_max == 6)]
        for name, g in parts.items():
            g = g[g.design == d]
            if excl:
                g = g[g.publication != C5]
            if g.empty:
                continue
            SA.append({"design": d, "noC5": excl, "stratum": name, "R2_B1": r2_seeds(g, "B1").mean(), "R2_B0": r2_seeds(g, "B0").mean(),
                       "rows_per_seed": len(g) / 5, "cells_per_seed": g.groupby("split_seed").cell_id.nunique().mean(),
                       "clusters": g.cluster.nunique(), "largest_cluster_share": g.cluster.value_counts(normalize=True).iloc[0],
                       "mean_density": g.density.mean()})
SA = pd.DataFrame(SA)
SA.to_csv(OUT / "saturation.csv", index=False)
print(SA.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\n=== strata: n_metals class x rung, and k_max (saturation = reads = k_max + 1), B1, C3 rows ===")
ST = []


def nm_class(n):
    return "14" if n == 14 else "13" if n == 13 else "8-12" if n >= 8 else "<=7"


for d in ("BPG", "BP"):
    for excl in (False, True):
        for p in ["K3"] + MAIN:
            fr = F[p][F[p].design == d]
            if excl:
                fr = fr[fr.publication != C5]
            fr = fr.assign(nmc=fr.n_metals.map(nm_class))
            for col in ("nmc", "k_max"):
                for v, g in fr.groupby(col):
                    ST.append({"design": d, "noC5": excl, "policy": p, "by": col, "stratum": str(v), "R2": r2_seeds(g, "B1").mean(),
                               "R2_B0": r2_seeds(g, "B0").mean(), "rows_per_seed": len(g) / 5,
                               "clusters": g.cluster.nunique(), "mean_density": g.density.mean(),
                               "hull_out": g.hull_out.mean()})
ST = pd.DataFrame(ST)
ST.to_csv(OUT / "strata.csv", index=False)
for d in ("BPG", "BP"):
    t = ST[(ST.design == d) & (~ST.noC5)]
    print(f"-- {d} --")
    print(t.pivot_table(index=["by", "stratum"], columns="policy", values="R2").round(3).to_string())
    print(t[t.policy == "K3"].set_index(["by", "stratum"])[["rows_per_seed", "clusters"]].to_string())
    print(t.pivot_table(index=["by", "stratum"], columns="policy", values="mean_density").round(3).to_string())

# ---- 5th-read placement and hull ----
print("\n=== 5th-read metal frequency (share of cell instances with a 5th read, BPG, 5 seeds) ===")
cc = cells[(cells.design == "BPG") & (cells.n_legal > 0)]
fq = pd.DataFrame({p: cc["pick_" + p].value_counts(normalize=True) for p in DET + ["MAXMIN_R_LOOSE"]}).fillna(0)
fq = fq.loc[[m for m in D.LANTHANIDES if m in fq.index]]
print(fq.round(3).to_string())
c14 = cc[cc.n_metals == 14]
lu = {p: float(((c14["pick_" + p] == "Lu") | c14.read3.str.contains("Lu")).mean()) for p in DET}
print("14-metal cells: Lu in the 5-metal read set:", {k: round(v, 3) for k, v in lu.items()},
      "| Lu in the k=3 read set:", round(float(c14.read3.str.contains("Lu").mean()), 3))
J["lu_in_set_14"] = lu
print("\n=== hull: share of C3 rows outside the read hull, and B1 R2 on in-hull rows (BPG) ===")
HL = []
for p in ["K3"] + MAIN + ["HULL_BISECT"]:
    fr = F[p][F[p].design == "BPG"]
    HL.append({"policy": p, "hull_out_share": fr.hull_out.mean(), "R2_in_hull": r2_seeds(fr[fr.hull_out < 0.5], "B1").mean()
               if p != "RANDOM" else np.nan, "SSE_share_out": fr[fr.hull_out > 0].se_B1.sum() / fr.se_B1.sum()})
print(pd.DataFrame(HL).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---- key-number json ----
key = S[(S.arm.isin(["B1", "B0", "HIST_D7", "B1_k3beta"]))][["design", "cohort", "label", "arm", "R2", "R2_min", "R2_max",
                                                              "seeds_ge_050", "P_boot_ge_050", "boot_lo", "boot_hi",
                                                              "loco_min", "loco_max", "per_seed"]]
J["ladder"] = key.round(5).to_dict("records")
(OUT / "summary.json").write_text(json.dumps(J, indent=1, default=str), encoding="utf-8")
print("\nEXIT 0")

"""REFUTER lens B, part 2 -- confounds, generalisation, units of inference."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa
from gen16 import l1_cycle as L  # noqa

OUT = L.RESULTS.parent / "refutation" / "L1NULL" / "B"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return np.nan, np.nan, int(ok.sum())
    r = stats.spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue), int(ok.sum())


def kish(counts):
    c = np.asarray(counts, float)
    return float(c.sum() ** 2 / (c ** 2).sum())


df = pd.read_pickle(OUT / "_df.pkl")

from gen15 import valuebench as V  # noqa
bench = V.load()
bf = bench.frame
pub = bf.groupby("extractant")["publication_id"].agg(list).to_dict()
modal = bf.groupby(["extractant", "publication_id"]).size().reset_index(name="n")
modal = modal.sort_values(["extractant", "n"], ascending=[True, False]).groupby("extractant").head(1)
df["modal_pub"] = df["extractant"].map(modal.set_index("extractant")["publication_id"].to_dict())

MODELS = ("NAIVE", "ELEM", "SPECIES", "SPECIES_CONST", "SPECIES_NFILLCOL")


def sub(model, set_):
    return df[(df.model == model) & df[set_]]


log("=== 1. diglycolamide (sc009) removal ===")
dig_rows = []
for m in MODELS:
    for s in ("S8", "S14", "S3"):
        d = sub(m, s)
        r0, p0, n0 = rho(d.slope.to_numpy(float), d.a.to_numpy(float))
        d1 = d[d.chemotype != "sc009"]
        r1, p1, n1 = rho(d1.slope.to_numpy(float), d1.a.to_numpy(float))
        d2 = d[d.chemotype == "sc009"]
        r2, p2, n2 = rho(d2.slope.to_numpy(float), d2.a.to_numpy(float))
        dig_rows.append(dict(model=m, set=s, n_all=n0, rho_all=r0, p_all=p0,
                             n_no_sc009=n1, rho_no_sc009=r1, p_no_sc009=p1,
                             n_sc009_only=n2, rho_sc009_only=r2, p_sc009_only=p2))
dig = pd.DataFrame(dig_rows)
dig.to_csv(OUT / "check1_diglycolamide.csv", index=False)
print(dig.round(4).to_string(index=False))

log("=== 2. n_metals / n_computed_metals confound ===")
c2 = []
for m in MODELS:
    for s in ("S8", "S14", "S3"):
        d = sub(m, s)
        x = d.slope.to_numpy(float)
        y = d.a.to_numpy(float)
        nm = d.n_metals.to_numpy(float)
        nc = d.n_computed_metals.to_numpy(float)
        ncell = d.n_cells.to_numpy(float)
        c2.append(dict(model=m, set=s, n=len(d), rho=rho(x, y)[0],
                       partial_n_metals=L._partial_rho(x, y, nm),
                       partial_n_computed=L._partial_rho(x, y, nc),
                       partial_n_cells=L._partial_rho(x, y, ncell),
                       rho_slope_vs_nmetals=rho(x, nm)[0],
                       rho_slope_vs_ncomputed=rho(x, nc)[0],
                       rho_a_vs_nmetals=rho(y, nm)[0],
                       rho_absa_vs_nmetals=rho(np.abs(y), nm)[0]))
c2 = pd.DataFrame(c2)
c2.to_csv(OUT / "check2_nmetals.csv", index=False)
print(c2.round(4).to_string(index=False))

log("--- stratified S8 ---")
strat = []
for m in MODELS:
    d = sub(m, "S8")
    med = float(np.median(d.n_metals))
    bands = [("n_metals<=%g" % med, d[d.n_metals <= med]),
             ("n_metals>%g" % med, d[d.n_metals > med]),
             ("n_computed 8-11", d[d.n_computed_metals < 12]),
             ("n_computed 12-14", d[d.n_computed_metals >= 12]),
             ("absa low half", d[d.abs_a <= d.abs_a.median()]),
             ("absa high half", d[d.abs_a > d.abs_a.median()]),
             ("comp constant", d[d.composition_varies == 0]),
             ("comp varies", d[d.composition_varies == 1]),
             ("nligs constant", d[d.n_ligs_varies == 0]),
             ("nligs varies", d[d.n_ligs_varies == 1])]
    for lab, dd in bands:
        r, p, n = rho(dd.slope.to_numpy(float), dd.a.to_numpy(float))
        strat.append(dict(model=m, band=lab, n=n, rho=r, p=p))
strat = pd.DataFrame(strat)
strat.to_csv(OUT / "check2b_strata.csv", index=False)
print(strat.round(4).to_string(index=False))

log("=== 3-4. units, LOCO chemotype, LOPO publication ===")
u = []
for s in ("S8", "S14", "S3"):
    d = sub("SPECIES", s)
    u.append(dict(set=s, n_extractants=len(d), n_chemotypes=d.chemotype.nunique(),
                  kish_chemotype=kish(d.chemotype.value_counts().to_numpy()),
                  n_modal_pubs=d.modal_pub.nunique(),
                  kish_modal_pub=kish(d.modal_pub.value_counts().to_numpy()),
                  frac_sc009=float((d.chemotype == "sc009").mean())))
u = pd.DataFrame(u)
u.to_csv(OUT / "check3_units.csv", index=False)
print(u.round(4).to_string(index=False))

loco = []
for m in MODELS:
    for s in ("S8", "S14", "S3"):
        d = sub(m, s)
        base = rho(d.slope.to_numpy(float), d.a.to_numpy(float))[0]
        vc = []
        for ch in sorted(d.chemotype.unique()):
            dd = d[d.chemotype != ch]
            vc.append(rho(dd.slope.to_numpy(float), dd.a.to_numpy(float))[0])
        vp = []
        allpubs = sorted(set(x for e in d.extractant for x in set(pub.get(e, []))))
        for p_ in allpubs:
            keep = d[~d.extractant.map(lambda e: p_ in set(pub.get(e, [])))]
            if len(keep) >= 10:
                vp.append(rho(keep.slope.to_numpy(float), keep.a.to_numpy(float))[0])
        vm = []
        for p_ in sorted(d.modal_pub.dropna().unique()):
            dd = d[d.modal_pub != p_]
            if len(dd) >= 10:
                vm.append(rho(dd.slope.to_numpy(float), dd.a.to_numpy(float))[0])
        vc, vp, vm = np.array(vc), np.array(vp), np.array(vm)
        loco.append(dict(model=m, set=s, rho=base,
                         n_contributing_chemotypes=int(d.chemotype.nunique()),
                         loco_chem_min=float(np.nanmin(vc)), loco_chem_max=float(np.nanmax(vc)),
                         loco_chem_sign_stable=bool(np.all(np.sign(vc) == np.sign(base))),
                         n_pubs=len(allpubs), n_lopo_any=len(vp),
                         lopo_any_min=float(np.nanmin(vp)) if len(vp) else np.nan,
                         lopo_any_max=float(np.nanmax(vp)) if len(vp) else np.nan,
                         lopo_any_sign_stable=bool(np.all(np.sign(vp) == np.sign(base))) if len(vp) else None,
                         n_modal_pubs=len(vm),
                         lopo_modal_min=float(np.nanmin(vm)) if len(vm) else np.nan,
                         lopo_modal_max=float(np.nanmax(vm)) if len(vm) else np.nan,
                         lopo_modal_sign_stable=bool(np.all(np.sign(vm) == np.sign(base))) if len(vm) else None))
loco = pd.DataFrame(loco)
loco.to_csv(OUT / "check4_loco_lopo.csv", index=False)
print(loco.round(4).to_string(index=False))

log("--- between/within chemotype ---")
bw = []
for m in MODELS:
    for s in ("S8", "S14", "S3"):
        d = sub(m, s)
        dd = pd.DataFrame({"x": d.slope.to_numpy(float), "y": d.a.to_numpy(float),
                           "c": d.chemotype.to_numpy()}).dropna()
        cm = dd.groupby("c").mean()
        bet = rho(cm.x.to_numpy(), cm.y.to_numpy())
        dd["xw"] = dd.x - dd.groupby("c").x.transform("mean")
        dd["yw"] = dd.y - dd.groupby("c").y.transform("mean")
        multi = dd[dd.groupby("c").x.transform("size") > 1]
        wit = rho(multi.xw.to_numpy(), multi.yw.to_numpy()) if len(multi) >= 10 else (np.nan, np.nan, len(multi))
        icc = 1 - dd.xw.var() / dd.x.var() if dd.x.var() > 0 else np.nan
        bw.append(dict(model=m, set=s, rho_extractant=rho(dd.x.to_numpy(), dd.y.to_numpy())[0],
                       rho_between_chemotype=bet[0], n_chemo=bet[2], p_between=bet[1],
                       rho_within_chemotype=wit[0], n_within=wit[2], icc_chemotype=float(icc)))
bw = pd.DataFrame(bw)
bw.to_csv(OUT / "check4b_between_within.csv", index=False)
print(bw.round(4).to_string(index=False))
log("part2 done")

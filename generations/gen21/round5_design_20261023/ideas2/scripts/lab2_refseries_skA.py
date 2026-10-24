#!/usr/bin/env python
"""Skeptic A for idea lab-2 (in-run TODGA reference series).  EXPLORATORY: every number comes from the 229 already-seen
C3 cells and the 5 dev seeds.  Prints only; writes nothing except stdout.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab2_refseries_skA.py

Independent re-implementation (does not import lab_calib*): PWL interpolation on the radius axis of (y - v) through the
DOPT_Q[:3] reads, plus v; B1 = v with the fold's fitted Gd amplitude.  Donor deviation = residual from the donor's own
least-squares quadratic in the radius coordinate (>= 6 measured metals); shape c_m = mean over donors measured at m;
w_m = n_m / (n_m + K), K = 6.1.  Targets: non-TODGA C3 cells (not on the round-3 V2 lists) of publications holding >= 1
usable TODGA cell.

Checks added beyond the idea's own quick check:
  * near-copy screen between every target and every TODGA / same-publication donor (raw values and deviation shapes);
  * OWN with near-copy donors removed, and on targets with no near-copy anywhere;
  * same-laboratory-other-publication TODGA (author rule >= 2 shared surnames, as lab_samelab) at own w;
  * same-medium other-publication TODGA at own w (removes the n / w confound of lab_extra's medium arm);
  * own-campaign TODGA at the same vs at a different condition (acid conc / diluent) inside the campaign;
  * is TODGA special?  count-matched random own-campaign non-TODGA donors (expectation over 30 draws);
  * K sensitivity; consensus with nan-mean instead of zero-fill; equal-weight publication summaries; LOPO.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import metrics as M4  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
G21 = r"D:\ml_separator_gh\generations\gen21"
XR = np.asarray(interp._coords("R"), float)
K = 6.1
REF = "TODGA"
NPB = 30


def devs(Y):
    out = np.full_like(Y, np.nan)
    for i in range(len(Y)):
        m = np.isfinite(Y[i])
        if m.sum() >= 6:
            A = np.vstack([XR[m] ** 2, XR[m], np.ones(m.sum())]).T
            co, *_ = np.linalg.lstsq(A, Y[i, m], rcond=None)
            out[i, m] = Y[i, m] - A @ co
    return out


def shp(Dv, idx, k=K, nanmean_cons=None):
    idx = np.asarray(idx, int)
    if len(idx) == 0:
        return np.zeros(N_LN), np.zeros(N_LN)
    A = Dv[idx]
    n = np.isfinite(A).sum(0).astype(float)
    c = np.where(n > 0, np.nansum(A, 0) / np.maximum(n, 1), 0.0)
    return c, n / (n + k)


def pwl(y, reads, metals, v):
    r = np.asarray(sorted(reads, key=lambda m: XR[m]), int)
    out = np.full(N_LN, np.nan)
    mm = np.asarray(metals, int)
    out[mm] = np.interp(XR[mm], XR[r], y[r] - v[r]) + v[mm]
    out[r] = y[r]
    return out


def surnames(s):
    out = set()
    for a in re.split(r"[;,&]", str(s)):
        toks = [t for t in re.findall(r"[A-Za-zÀ-ÿ\-']+", a) if t.lower().strip(".") not in {"dr", "prof", "mr", "ms", "mrs", "phd"}]
        if toks:
            out.add(toks[-1].lower())
    return out


def sq(d, a):
    return (d[a].to_numpy(float) - d.y.to_numpy(float)) ** 2 if not a.startswith("E_") else d[a].to_numpy(float)


def r2(d, a):
    o = []
    for _, g in d.groupby("split_seed"):
        y = g.y.to_numpy(float)
        o.append(1 - sq(g, a).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(o))


def rel(d, a, b="B1"):
    return float(1 - sq(d, a).sum() / sq(d, b).sum())


def unit(d, a, b="B1"):
    g = d.assign(_a=sq(d, a), _b=sq(d, b)).groupby("publication")[["_a", "_b"]].sum()
    per = 1 - g._a / g._b
    pooled = 1 - g._a.sum() / g._b.sum()
    jk = [1 - g._a.drop(p).sum() / g._b.drop(p).sum() for p in g.index]
    return {"pooled": pooled, "eqw_mean": per.mean(), "median": per.median(), "up": int((per > 1e-9).sum()),
            "down": int((per < -1e-9).sum()), "G": len(per), "lopo_min": min(jk), "lopo_max": max(jk),
            "per": {p[4:10]: round(v, 3) for p, v in per.items()}}


def main():
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    nm = f.extractant_name.astype(str).to_numpy()
    acidv = f[[c for c in f.columns if c.startswith("cond__acid__")]].fillna(0).to_numpy() > 0
    dilv = f[[c for c in f.columns if c.startswith("cond__diluent__")]].fillna(0).to_numpy() > 0
    lac = np.log10(np.clip(f["cond__acid_concentration_M"].to_numpy(float), 1e-4, None))
    v2 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    ok = np.array([c not in v2 for c in ids])
    Dv = devs(Y)
    has = np.isfinite(Dv).any(1) & ok
    ar = np.arange(len(f))
    tod = has & (nm == REF)
    tod_pubs = sorted(set(pub[tod]))
    # same-laboratory map (author rule as lab_samelab)
    src = pd.read_csv(fr"{G21}\round4_20261022\results\provenance\sources_by_doi.csv")
    reg = pd.read_csv(fr"{G21}\round5_design_20261023\probes\P4_corpus_registry.csv")
    au = dict(zip(reg.key.astype(str), reg.authors.astype(str)))
    p2s = {}
    for r in src.itertuples(index=False):
        for p in str(r.publication_ids).split(";"):
            p = p.strip()
            if p and str(r.doi) in au:
                p2s[p] = p2s.get(p, set()) | surnames(au[str(r.doi)])
    def same_lab(p, q):
        return p != q and p in p2s and q in p2s and len(p2s[p] & p2s[q]) >= 2
    print("usable TODGA cells (>= 6 metals, not V2) per publication:",
          {p[4:10]: int((tod & (pub == p)).sum()) for p in tod_pubs})
    # same-medium identity of conditions
    def same_med(i, j, tol=0.15):
        return bool((acidv[i] == acidv[j]).all() and (dilv[i] == dilv[j]).all() and abs(lac[i] - lac[j]) <= tol)

    par = {d: pd.read_parquet(paths.PRED / f"REF_{d}.parquet") for d in ("BPG", "BP")}
    c3 = sorted(set(par["BPG"].cell_id))
    targets = [t for t in c3 if t not in v2 and nm[pos[t]] != REF and pub[pos[t]] in tod_pubs
               and (tod & (pub == pub[pos[t]]) & (ar != pos[t])).any()]
    print(f"targets: {len(targets)} cells in {len({pub[pos[t]] for t in targets})} publications")

    # ---------------- near-copy screen ----------------
    print("\n=== near-copy screen: target vs every usable bench cell (raw |dY| <= 0.03 on >= 4 shared metals, or "
          "deviation-shape rms diff <= 0.02 on >= 6 shared) ===")
    nc = {}
    for t in targets:
        i = pos[t]
        hits = []
        for j in np.where(np.isfinite(Y).sum(1) >= 4)[0]:
            if j == i:
                continue
            sh = np.isfinite(Y[i]) & np.isfinite(Y[j])
            if sh.sum() < 4:
                continue
            raw = np.max(np.abs(Y[i, sh] - Y[j, sh]))
            shd = np.isfinite(Dv[i]) & np.isfinite(Dv[j])
            dr = np.sqrt(np.mean((Dv[i, shd] - Dv[j, shd]) ** 2)) if shd.sum() >= 6 else np.inf
            if raw <= 0.03 or dr <= 0.02:
                hits.append((ids[j][:16], pub[j][4:10], nm[j], int(sh.sum()), round(float(raw), 3), round(float(dr), 4),
                             bool(ok[j])))
                nc.setdefault(t, set()).add(j)
        if hits:
            print(f"  target {t[:16]} {pub[i][4:10]} {nm[i]:10s}: {hits}")
    nc_t = {t for t in targets if t in nc}

    rows = []
    for design in ("BPG", "BP"):
        P = par[design]
        fitted = json.loads((paths.RES / "REF" / f"fitted_{design}.json").read_text())
        for seed in paths.DEV_SEEDS:
            for fold in range(5):
                rec = C.load(design, seed, fold)
                u = np.zeros(N_LN)
                u[CO.GD] = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
                prow = P[(P.split_seed == seed) & (P.fold == fold)]
                pcg = {c: g for c, g in prow.groupby("cell_id")}
                for cell in rec["outer"]["cells"]:
                    t = cell["cell_id"]
                    if t not in pcg or t not in targets:
                        continue
                    i = pos[t]
                    yi = Y[i]
                    metals = list(cell["metals"])
                    reads = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                    L = {"B1": pwl(yi, reads, metals, u)}

                    def lev(c, w):
                        return pwl(yi, reads, metals, (1 - w) * u + w * c)
                    own = np.where(tod & (pub == pub[i]) & (ar != i))[0]
                    co, wo = shp(Dv, own)
                    L["OWN"] = lev(co, wo)
                    for kk, nmk in ((1.78, "OWN_K178"), (20.0, "OWN_K20")):
                        L[nmk] = lev(*shp(Dv, own, kk))
                    own_nc = np.array([j for j in own if j not in nc.get(t, set())], int)
                    L["OWN_noNC"] = lev(*shp(Dv, own_nc))
                    # same-condition vs different-condition own TODGA
                    osame = np.array([j for j in own if same_med(i, j)], int)
                    odiff = np.array([j for j in own if not same_med(i, j)], int)
                    L["OWN_samecond"] = lev(*shp(Dv, osame)) if len(osame) else L["B1"]
                    L["OWN_diffcond"] = lev(*shp(Dv, odiff)) if len(odiff) else L["B1"]
                    # consensus (zero-fill, as lab_calib2; and nan-mean)
                    shapes = {p: shp(Dv, np.where(tod & (pub == p))[0]) for p in tod_pubs if p != pub[i]}
                    cons0 = np.mean([s[0] for s in shapes.values()], 0)
                    Cm = np.array([np.where(s[1] > 0, s[0], np.nan) for s in shapes.values()])
                    consn = np.where(np.isfinite(Cm).any(0), np.nanmean(np.where(np.isfinite(Cm), Cm, np.nan), 0), 0.0)
                    L["CONS"] = lev(cons0, wo)
                    L["FP"] = pwl(yi, reads, metals, u + wo * (co - cons0))
                    L["FP_nan"] = pwl(yi, reads, metals, u + wo * (co - consn))
                    # same-lab other publication TODGA (author rule), at own w
                    sl = np.where(tod & np.array([same_lab(pub[i], q) for q in pub]))[0]
                    L["SAMELAB_ownw"] = lev(shp(Dv, sl)[0], wo) if len(sl) else L["B1"]
                    L["SAMELAB_ownn"] = lev(*shp(Dv, sl)) if len(sl) else L["B1"]
                    # same-medium other-publication TODGA at own w
                    md = np.array([j for j in np.where(tod & (pub != pub[i]))[0] if same_med(i, j)], int)
                    L["MED_ownw"] = lev(shp(Dv, md)[0], wo) if len(md) else L["B1"]
                    # placebo: count-matched random own-campaign non-TODGA other-extractant donors
                    pool = np.where(has & (pub == pub[i]) & (ar != i) & (nm != REF) & (ext != ext[i]))[0]
                    PB = []
                    if len(pool):
                        rng = np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16))
                        for _ in range(NPB):
                            PB.append(lev(*shp(Dv, rng.choice(pool, min(len(own), len(pool)), replace=False))))
                    L["MXall"] = lev(*shp(Dv, pool)) if len(pool) else L["B1"]
                    for r in pcg[t].itertuples(index=False):
                        ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                        o = {"design": design, "split_seed": seed, "cell_id": t, "publication": pub[i], "name": nm[i],
                             "chemotype": f.chemotype.iloc[i], "pair": f"{r.A}/{r.B}", "y": r.y, "B1_par": r.B1,
                             "n_own": len(own), "n_same": len(osame), "n_diff": len(odiff), "n_sl": len(sl), "n_md": len(md),
                             "n_pool": len(pool), "nc_t": t in nc_t, "n_nc_own": len(own) - len(own_nc)}
                        for k2, lv in L.items():
                            o[k2] = lv[ia] - lv[ib]
                        o["E_PB"] = float(np.mean([(lv[ia] - lv[ib] - r.y) ** 2 for lv in PB])) if PB else (o["B1"] - r.y) ** 2
                        rows.append(o)
    T = pd.DataFrame(rows)
    print(f"\nmax |B1 recomputed - B1 parquet| = {float((T.B1 - T.B1_par).abs().max()):.2e}")
    T = M4.add_clusters(T.rename(columns={}), f)
    print("source clusters of the targets:", T.drop_duplicates("cell_id").cluster_cpg.value_counts().to_dict())
    print("copy groups:", T.drop_duplicates("cell_id").groupby("publication").copy_group.first().to_dict())

    arms = ["OWN", "OWN_K178", "OWN_K20", "OWN_noNC", "CONS", "FP", "FP_nan", "SAMELAB_ownw", "SAMELAB_ownn", "MED_ownw",
            "OWN_samecond", "OWN_diffcond", "E_PB", "MXall"]
    for design in ("BPG", "BP"):
        d = T[T.design == design]
        print(f"\n==================== {design}: {d.cell_id.nunique()} cells, {d.publication.nunique()} pubs, "
              f"{len(d) / 5:.0f} rows/seed; B1 R2 {r2(d, 'B1'):.4f} ====================")
        out = []
        for a in arms:
            u_ = unit(d, a)
            out.append({"arm": a, "R2": round(r2(d, a), 4), **{k: (round(v, 4) if isinstance(v, float) else v)
                                                              for k, v in u_.items()}})
        print(pd.DataFrame(out).to_string(index=False))
        if design == "BP":
            continue
        print("\nper-publication target composition (cells; own TODGA n; same-cond n; same-lab n; same-medium n):")
        print(d.drop_duplicates("cell_id").groupby("publication").agg(cells=("cell_id", "nunique"), n_own=("n_own", "max"),
              n_same=("n_same", "max"), n_sl=("n_sl", "max"), n_md=("n_md", "max"), nc_targets=("nc_t", "sum"),
              nc_own_donors=("n_nc_own", "max")).to_string())
        # per-target contribution of OWN
        g = d.assign(dS=(sq(d, "B1") - sq(d, "OWN")) / 5, S=sq(d, "B1") / 5).groupby(["publication", "cell_id", "name"])[["dS", "S"]].sum()
        g["share_of_total_gain"] = g.dS / g.dS.sum()
        print(f"\nOWN: total SSE gain per seed {g.dS.sum():.4f} of B1 SSE {g.S.sum():.4f}; top targets:")
        print(g.sort_values("dS", ascending=False).head(10).round(4).to_string())
        print(g.sort_values("dS").head(5).round(4).to_string())
        # subsets
        for title, m in (("targets with no near-copy anywhere in the bench", ~d.nc_t),
                         ("without pub_5a68dc5665", d.publication != "pub_5a68dc5665"),
                         ("without pub_4cba27f5f3 and pub_5a68dc5665", ~d.publication.isin(["pub_4cba27f5f3", "pub_5a68dc5665"])),
                         ("targets with a same-condition own TODGA", d.n_same > 0),
                         ("targets with a same-lab other-pub TODGA", d.n_sl > 0),
                         ("targets with a same-medium other-pub TODGA", d.n_md > 0)):
            s = d[m]
            if s.empty:
                continue
            print(f"\n--- {title}: {s.cell_id.nunique()} cells, {s.publication.nunique()} pubs ---")
            out = []
            for a in ("OWN", "OWN_noNC", "FP", "CONS", "SAMELAB_ownw", "MED_ownw", "OWN_samecond", "OWN_diffcond", "E_PB", "MXall"):
                u_ = unit(s, a)
                out.append({"arm": a, "pooled": round(u_["pooled"], 4), "eqw_mean": round(u_["eqw_mean"], 4),
                            "up/down": f"{u_['up']}/{u_['down']}", "per": u_["per"]})
            print(pd.DataFrame(out).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

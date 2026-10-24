#!/usr/bin/env python
"""Skeptic B check of idea lab-2 (in-run TODGA reference series).  EXPLORATORY: 229 already-seen C3 cells, 5 dev seeds.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/skepB_lab2_refcheck.py

Questions (read-only on the bench, round-4 REF parquets and the fold cache; reuses lab_calib / lab_analyze helpers):
 1. In-run analogue.  Split the target's own-publication TODGA donors into SAME-MEDIUM (same acid one-hot,
    |dlog10 acid| <= 0.15, same diluent one-hot -- the protocol's "same tube condition") and OTHER-MEDIUM donors.
    If the gain lives in other-medium TODGA, the corpus supports "campaign", not "same condition / same run".
 2. The kill-(b) comparator on the corpus: own TODGA vs medium-matched literature TODGA, paired, publication units.
 3. Size of the fingerprint f = own TODGA shape - consensus TODGA shape (per metal), raw and as applied (w * f), to set
    against plausible duplicate repeatability (kill (a), and the 50 %-norm attribution rule (c)).
 4. Operating characteristics of kill (b)'s "laboratory-unit median gain <= 0" at G = 8, using the corpus per-publication
    spread of MREF_fp / MREF_own gains.
Prints only; writes ideas2/skepB_lab2_out/refcheck_rows.parquet.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402
from lab_calib import detrend, shape, levels, K6  # noqa: E402
from lab_analyze import r2, rel, infer  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\skepB_lab2_out")
pd.set_option("display.width", 250)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okd = np.array([c not in v2drop for c in ids])
    D2 = detrend(Y, 2, 6)
    has = np.isfinite(D2).any(1) & okd
    pub = f.publication_id.astype(str).to_numpy()
    name = f.extractant_name.astype(str).to_numpy()
    acid = f[[c for c in f.columns if c.startswith("cond__acid__")]].fillna(0).to_numpy() > 0
    dil = f[[c for c in f.columns if c.startswith("cond__diluent__")]].fillna(0).to_numpy() > 0
    lac = np.log10(np.clip(f["cond__acid_concentration_M"].to_numpy(float), 1e-4, None))
    ar = np.arange(len(f))
    ref_pubs = sorted({pub[i] for i in range(len(f)) if name[i] == "TODGA"})
    tod = np.where(has & (name == "TODGA"))[0]
    tod_shape = {p: shape(D2, tod[pub[tod] == p], K6) for p in ref_pubs}

    def same_med(i, cand):
        cand = np.asarray(cand, int)
        if len(cand) == 0:
            return cand
        return cand[(acid[cand] == acid[i]).all(1) & (dil[cand] == dil[i]).all(1) & (np.abs(lac[cand] - lac[i]) <= 0.15)]

    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    fitted = json.loads((paths.RES / "REF" / "fitted_BPG.json").read_text())
    rows, fmag = [], []
    seen_f = set()
    for seed in paths.DEV_SEEDS:
        for fold in range(5):
            rec = C.load("BPG", seed, fold)
            u = np.zeros(N_LN)
            u[CO.GD] = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
            prow = par[(par.split_seed == seed) & (par.fold == fold)]
            pcg = {c: g for c, g in prow.groupby("cell_id")}
            for cell in rec["outer"]["cells"]:
                t = cell["cell_id"]
                if t not in pcg or t in v2drop:
                    continue
                i = pos[t]
                if pub[i] not in ref_pubs or name[i] == "TODGA":
                    continue
                own = np.where(has & (pub == pub[i]) & (name == "TODGA") & (ar != i))[0]
                if len(own) == 0:
                    continue
                own_s = same_med(i, own)
                own_d = np.setdiff1d(own, own_s)
                med = same_med(i, tod[pub[tod] != pub[i]])
                oth = [tod_shape[p][0] for p in ref_pubs if p != pub[i] and tod_shape[p][1].max() > 0]
                cons = np.mean(oth, 0)
                yi = Y[i]
                reads = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                metals = list(cell["metals"])
                L = {"B1": levels(yi, reads, metals, u)}
                co, wo = shape(D2, own, K6)
                L["own_all"] = levels(yi, reads, metals, (1 - wo) * u + wo * co)
                L["fp_all"] = levels(yi, reads, metals, u + wo * (co - cons))
                L["cons"] = levels(yi, reads, metals, (1 - wo) * u + wo * cons)
                if len(own_s):
                    cs, ws = shape(D2, own_s, K6)
                    L["own_same"] = levels(yi, reads, metals, (1 - ws) * u + ws * cs)
                    L["fp_same"] = levels(yi, reads, metals, u + ws * (cs - cons))
                if len(own_d):
                    cd, wd = shape(D2, own_d, K6)
                    L["own_diff"] = levels(yi, reads, metals, (1 - wd) * u + wd * cd)
                if len(med):
                    cm, wm = shape(D2, med, K6)
                    L["med_lit"] = levels(yi, reads, metals, (1 - wm) * u + wm * cm)
                if t not in seen_f:
                    seen_f.add(t)
                    mm = np.asarray(metals, int)
                    fr = (co - cons)[mm]
                    fmag.append({"cell_id": t, "publication": pub[i], "n_own": len(own), "n_same": len(own_s),
                                 "med_abs_f": float(np.median(np.abs(fr))), "rms_f": float(np.sqrt(np.mean(fr ** 2))),
                                 "med_abs_wf": float(np.median(np.abs((wo * (co - cons))[mm]))),
                                 "w_mean": float(wo[mm].mean()),
                                 "tgt_minus_tod_level": float(np.nanmean(Y[i, mm]) - np.nanmean(Y[own][:, mm]))})
                for r in pcg[t].itertuples(index=False):
                    ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                    o = {"split_seed": seed, "fold": fold, "cell_id": t, "publication": pub[i], "pair": f"{r.A}/{r.B}",
                         "y": r.y, "n_own": len(own), "n_same": len(own_s), "n_diff": len(own_d), "n_med": len(med)}
                    for k in ("B1", "own_all", "fp_all", "cons", "own_same", "fp_same", "own_diff", "med_lit"):
                        o[k] = (L[k][ia] - L[k][ib]) if k in L else np.nan
                    rows.append(o)
    T = pd.DataFrame(rows)
    T.to_parquet(OUT / "refcheck_rows.parquet", index=False)
    print(f"rows {len(T)}, cells {T.cell_id.nunique()}, pubs {T.publication.nunique()}  ({time.time() - t0:.0f}s)")
    cells = T.drop_duplicates("cell_id")
    print("\nper publication: cells, with same-medium own TODGA, with other-medium own TODGA, with medium-matched literature TODGA")
    print(cells.groupby("publication").agg(cells=("cell_id", "size"), same=("n_same", lambda s: int((s > 0).sum())),
                                           diff=("n_diff", lambda s: int((s > 0).sum())),
                                           lit=("n_med", lambda s: int((s > 0).sum()))).to_string())

    def tab(title, d, arms):
        print(f"\n--- {title}: cells {d.cell_id.nunique()}, pubs {d.publication.nunique()}, rows/seed {len(d) / 5:.0f} ---")
        print(pd.DataFrame([{"arm": a, "R2": r2(d, a), "rel_vs_B1": rel(d, a)} for a in arms
                            if not d[a].isna().any()]).round(4).to_string(index=False))

    def inf(title, d, pairs):
        out = []
        for a, b in pairs:
            if d[a].isna().any() or d[b].isna().any():
                continue
            r = infer(d, a, b)
            out.append({"contrast": f"{a} - {b}", **{k: r[k] for k in ("rel", "lo", "hi", "G", "up", "down", "p_sf")}})
        print(f"[{title}]")
        print(pd.DataFrame(out).round(4).to_string(index=False))

    tab("all reference targets", T, ["B1", "own_all", "fp_all", "cons"])
    inf("all reference targets", T, [("own_all", "B1"), ("fp_all", "B1"), ("own_all", "fp_all")])
    S = T[T.n_same > 0]
    tab("targets WITH a same-medium own TODGA (in-run analogue)", S, ["B1", "own_all", "own_same", "fp_all", "fp_same"])
    inf("same-medium subset", S, [("own_same", "B1"), ("fp_same", "B1"), ("own_same", "own_all")])
    SD = T[(T.n_same > 0) & (T.n_diff > 0)]
    if len(SD):
        tab("targets with BOTH same- and other-medium own TODGA", SD, ["B1", "own_same", "own_diff", "own_all"])
        inf("same vs other medium (both exist)", SD, [("own_same", "own_diff"), ("own_same", "B1"), ("own_diff", "B1")])
    ND = T[(T.n_same == 0)]
    if len(ND):
        tab("targets with ONLY other-medium own TODGA", ND, ["B1", "own_all", "fp_all"])
    M = T[T.n_med > 0]
    tab("targets with a medium-matched literature TODGA", M, ["B1", "own_all", "med_lit"] + (["own_same"] if not M.own_same.isna().any() else []))
    inf("kill-(b) comparator: in-house vs literature at matched medium", M, [("own_all", "med_lit"), ("med_lit", "B1"), ("own_all", "B1")])
    MS = M[M.n_same > 0]
    if len(MS):
        tab("matched literature AND same-medium own TODGA", MS, ["B1", "own_same", "med_lit"])
        inf("same-medium own vs matched literature", MS, [("own_same", "med_lit")])

    F = pd.DataFrame(fmag)
    print("\nfingerprint size f = own TODGA shape - consensus TODGA shape (log D units, over the target's metals):")
    print(F[["med_abs_f", "rms_f", "med_abs_wf", "w_mean", "tgt_minus_tod_level"]].describe().round(3).to_string())
    print(F.groupby("publication")[["med_abs_f", "med_abs_wf", "w_mean", "tgt_minus_tod_level"]].median().round(3).to_string())

    # kill-(b) operating characteristics at G = 8 laboratories
    rng = np.random.default_rng(20261023)
    for arm in ("fp_all", "own_all"):
        per = np.array([rel(g, arm) for _, g in T.groupby("publication")])
        mu, sd = per.mean(), per.std(ddof=1)
        for G in (8, 20):
            sims = rng.normal(mu, sd, size=(100000, G))
            p_kill_true = float((np.median(sims, 1) <= 0).mean())
            sims0 = rng.normal(0.0, sd, size=(100000, G))
            p_kill_null = float((np.median(sims0, 1) <= 0).mean())
            sims3 = rng.normal(0.03, sd, size=(100000, G))
            p_kill_3 = float((np.median(sims3, 1) <= 0).mean())
            print(f"kill (b) median<=0, arm {arm}: per-pub rel {np.round(per, 3)} -> mean {mu:+.3f} sd {sd:.3f}; G={G}: "
                  f"P(kill | corpus mean) {p_kill_true:.2f}, P(kill | +3 %) {p_kill_3:.2f}, P(kill | 0) {p_kill_null:.2f}")
    per_d = np.array([rel(g, "own_all", "med_lit") for _, g in M.groupby("publication")])
    sd_d = per_d.std(ddof=1)
    print(f"own - literature per pub (rel SSE vs literature): {np.round(per_d, 3)}; sd {sd_d:.3f}; "
          f"SE of an 8-lab mean {sd_d / np.sqrt(8):.3f} vs the 1 % equivalence margin")
    print(f"done {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

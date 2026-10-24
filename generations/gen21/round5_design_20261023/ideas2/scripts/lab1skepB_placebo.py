#!/usr/bin/env python
"""Skeptic-B check on idea lab-1 (E-LAB, level-matched campaign shape MX_LM10).

EXPLORATORY: the 229 already-seen C3 cells, 5 dev seeds, BPG only (BP agreed to 0.003 in lab_calib2).
Read-only on every existing file; reuses lab_calib / lab_calib2 / lab_analyze functions without modifying them.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab1skepB_placebo.py

Question 1 (is the 'campaign-specific' control fair?).  lab-1's kill uses MG_LM10_all, which pools every other
publication, so its w_m -> 1 while MX_LM10's w_m is small.  Two count-matched placebos with the SAME level matching:
  E_PCM    per metal, draw exactly n_m (= MX_LM10's matched donor count at m) level-matched donors from OTHER
           publications; expected squared error over 30 draws (same w_m as MX_LM10).
  E_PCMC   the same, but the pool is restricted to other-publication donors of the same broad ligand class
           (DGA / malonamide / other by SMARTS; 'chemistry, not campaign' control).
  E_RPUB   one random other publication (with >= 2 donor series), all its cells level-matched (a 'wrong campaign').
Question 2 (is it campaign or same-medium chemistry?).  Own-campaign donors split by medium:
  MX_LM10_SM  donors in the same medium (same acid type, |dlog acid| <= 0.15, same diluent)
  MX_LM10_DM  donors in a different medium
Question 3: publication-level median of rel SSE (lab-1's kill statistic) and equal-weight mean.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

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
from lab_calib import detrend, levels, K6  # noqa: E402
from lab_calib2 import lm_shape  # noqa: E402
from lab_analyze import r2, rel, infer, sqe  # noqa: E402

GD = CO.GD
NPB = 30
pd.set_option("display.width", 250)


def broad_class(smiles):
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return "other"
        if m.HasSubstructMatch(Chem.MolFromSmarts("[NX3]C(=O)[CH2]O[CH2]C(=O)")):
            return "DGA"
        if m.HasSubstructMatch(Chem.MolFromSmarts("[NX3]C(=O)[CX4]C(=O)[NX3]")):
            return "malonamide"
        if m.HasSubstructMatch(Chem.MolFromSmarts("[NX3]C(=O)c1ncccc1")):
            return "pyridine-amide"
        if m.HasSubstructMatch(Chem.MolFromSmarts("P(=O)")):
            return "organoP"
        if m.HasSubstructMatch(Chem.MolFromSmarts("N(CC(=O)[OX2H1,OX1-])CC(=O)[OX2H1,OX1-]")):
            return "aminocarboxylate"
    except Exception:
        pass
    return "other"


def main() -> int:
    t0 = time.time()
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f["publication_id"].astype(str).to_numpy()
    ext = f["extractant"].astype(str).to_numpy()
    cls = np.array([broad_class(s) for s in ext])
    acid_cols = [c for c in f.columns if c.startswith("cond__acid__")]
    dil_cols = [c for c in f.columns if c.startswith("cond__diluent__")]
    acid_id = f[acid_cols].fillna(0).to_numpy().argmax(1) + 1000 * (f[acid_cols].fillna(0).to_numpy().max(1) > 0)
    dil_id = f[dil_cols].fillna(0).to_numpy().argmax(1) + 1000 * (f[dil_cols].fillna(0).to_numpy().max(1) > 0)
    lacid = np.log10(np.clip(f["cond__acid_concentration_M"].astype(float).to_numpy(), 1e-6, None))
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okd = np.array([c not in v2drop for c in ids])
    D2 = detrend(Y, 2, 6)
    has = np.isfinite(D2).any(1) & okd
    ar = np.arange(len(f))
    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    fitted = json.loads((paths.RES / "REF" / "fitted_BPG.json").read_text())
    pubs_with_donors = {}
    for p in np.unique(pub):
        pubs_with_donors[p] = np.where(has & (pub == p))[0]
    elig_pubs = [p for p, v in pubs_with_donors.items() if len(v) >= 2]
    rows = []
    for seed in paths.DEV_SEEDS:
        for fold in range(5):
            rec = C.load("BPG", seed, fold)
            beta = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
            u = np.zeros(N_LN)
            u[GD] = beta
            prow = par[(par.split_seed == seed) & (par.fold == fold)]
            pcg = {c: g for c, g in prow.groupby("cell_id")}
            for cell in rec["outer"]["cells"]:
                t = cell["cell_id"]
                if t not in pcg:
                    continue
                i = pos[t]
                yi = Y[i]
                metals = list(cell["metals"])
                reads = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                base = has & (ar != i)
                sp = base & (pub == pub[i])
                dX = np.where(sp & (ext != ext[i]))[0]
                if len(dX) == 0 or t in v2drop:
                    continue
                same_med = (acid_id == acid_id[i]) & (dil_id == dil_id[i]) & (np.abs(lacid - lacid[i]) <= 0.15)
                dSM = np.where(sp & (ext != ext[i]) & same_med)[0]
                dDM = np.where(sp & (ext != ext[i]) & ~same_med)[0]
                L = {"B1": levels(yi, reads, metals, u)}
                lv = L["B1"]

                def lev(cw):
                    c, w = cw
                    return levels(yi, reads, metals, (1 - w) * u + w * c)
                L["MX_LM10"] = lev(lm_shape(D2, Y, dX, lv, 1.0, K6))
                L["MX_LM10_SM"] = lev(lm_shape(D2, Y, dSM, lv, 1.0, K6))
                L["MX_LM10_DM"] = lev(lm_shape(D2, Y, dDM, lv, 1.0, K6))
                # matched donor counts per metal for MX_LM10
                A = D2[dX]
                msk = np.isfinite(A) & (np.abs(Y[dX] - lv[None, :]) <= 1.0)
                nm = msk.sum(0)
                pool = np.where(base & (pub != pub[i]))[0]
                poolc = pool[cls[pool] == cls[i]]
                rng = np.random.default_rng(int(hashlib.sha256((t + str(seed)).encode()).hexdigest()[:8], 16))
                PCM, PCMC, RP = [], [], []
                for _ in range(NPB):
                    for P, store in ((pool, PCM), (poolc, PCMC)):
                        c = np.zeros(N_LN)
                        w = np.zeros(N_LN)
                        for m in range(N_LN):
                            if nm[m] == 0:
                                continue
                            cand = P[np.isfinite(D2[P, m]) & (np.abs(Y[P, m] - lv[m]) <= 1.0)]
                            if len(cand) == 0:
                                continue
                            k = min(int(nm[m]), len(cand))
                            dd = rng.choice(cand, k, replace=False)
                            c[m] = D2[dd, m].mean()
                            w[m] = k / (k + K6)
                        store.append(lev((c, w)))
                    op = [p for p in elig_pubs if p != pub[i]]
                    pp = op[rng.integers(len(op))]
                    RP.append(lev(lm_shape(D2, Y, pubs_with_donors[pp], lv, 1.0, K6)))
                for r in pcg[t].itertuples(index=False):
                    ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                    o = {"split_seed": seed, "fold": fold, "cell_id": t, "publication": pub[i], "cls": cls[i],
                         "pair": f"{r.A}/{r.B}", "y": r.y, "B1_par": r.B1, "n_SM": len(dSM), "n_DM": len(dDM),
                         "nm_hidden": int(nm[ia] + nm[ib])}
                    for k2, l2 in L.items():
                        o[k2] = l2[ia] - l2[ib]
                    for nmn, PB in (("E_PCM", PCM), ("E_PCMC", PCMC), ("E_RPUB", RP)):
                        o[nmn] = float(np.mean([(l2[ia] - l2[ib] - r.y) ** 2 for l2 in PB]))
                    rows.append(o)
        print(f"seed {seed} done {time.time() - t0:.0f}s", flush=True)
    T = pd.DataFrame(rows)
    print("max |B1 - B1_par|:", float((T.B1 - T.B1_par).abs().max()))
    ref = pd.read_parquet(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\lab_out\lab_rows2.parquet")
    ref = ref[(ref.design == "BPG")][["split_seed", "cell_id", "pair", "MX_LM10"]].rename(columns={"MX_LM10": "ref"})
    chk = T.merge(ref, on=["split_seed", "cell_id", "pair"])
    print("rows matched to lab_rows2:", len(chk), "of", len(T), "max |MX_LM10 - lab_rows2|:",
          float((chk.MX_LM10 - chk.ref).abs().max()))

    def table(title, d, arms):
        print(f"\n--- {title}: cells {d.cell_id.nunique()}, pubs {d.publication.nunique()}, rows/seed {len(d) / 5:.0f} ---")
        print(pd.DataFrame([{"arm": a, "R2": r2(d, a), "rel_vs_B1": rel(d, a)} for a in arms]).round(4).to_string(index=False))

    def inf(title, d, pairs):
        out = []
        for a, b in pairs:
            r = infer(d, a, b)
            out.append({"contrast": f"{a} - {b}", **{k: r[k] for k in
                        ("rel", "lo", "hi", "G", "up", "down", "p_sf", "top3", "jk_min", "jk_min_pub")}})
        print(f"[{title}]")
        print(pd.DataFrame(out).round(4).to_string(index=False))

    def pubstats(d, a, b="B1"):
        g = d.assign(_a=sqe(d, a), _b=sqe(d, b)).groupby("publication")[["_a", "_b"]].sum()
        relp = 1 - g._a / g._b
        return float(relp.median()), float(relp.mean()), int((relp > 0).sum()), len(relp)

    arms = ["B1", "MX_LM10", "E_PCM", "E_PCMC", "E_RPUB"]
    X = T
    table("POPX (BPG)", X, arms)
    inf("POPX", X, [("MX_LM10", "B1"), ("MX_LM10", "E_PCM"), ("MX_LM10", "E_PCMC"), ("MX_LM10", "E_RPUB"),
                    ("E_PCM", "B1"), ("E_PCMC", "B1")])
    for a, b in (("MX_LM10", "B1"), ("MX_LM10", "E_PCM"), ("E_PCMC", "B1"), ("E_PCM", "B1")):
        md, mn, npos, G = pubstats(X, a, b)
        print(f"   per-publication rel SSE {a} vs {b}: median {md:+.3f}, equal-weight mean {mn:+.3f}, positive {npos}/{G}")
    XL = X[~X.publication.isin(["pub_126bbe7976", "pub_1bd76c48d9"])]
    table("POPX without pub_126bbe7976, pub_1bd76c48d9", XL, arms)
    inf("POPX without the two lead publications", XL, [("MX_LM10", "B1"), ("MX_LM10", "E_PCM"), ("MX_LM10", "E_PCMC")])
    print("\nbroad class of POPX targets (cells):", X.groupby("cls").cell_id.nunique().to_dict())
    for c, g in X.groupby("cls"):
        if g.cell_id.nunique() >= 3:
            print(f"   class {c}: cells {g.cell_id.nunique()} pubs {g.publication.nunique()}  "
                  f"MX_LM10 {rel(g, 'MX_LM10'):+.3f}  E_PCM {rel(g, 'E_PCM'):+.3f}  E_PCMC {rel(g, 'E_PCMC'):+.3f}")
    both = X[(X.n_SM > 0) & (X.n_DM > 0)]
    table("rows with both same-medium and different-medium own-campaign donors", both,
          ["B1", "MX_LM10", "MX_LM10_SM", "MX_LM10_DM", "E_PCM"])
    if both.publication.nunique() >= 3:
        inf("medium split", both, [("MX_LM10_SM", "B1"), ("MX_LM10_DM", "B1"), ("MX_LM10_SM", "MX_LM10_DM")])
    onlyDM = X[(X.n_SM == 0)]
    table("rows with no same-medium own-campaign donor", onlyDM, ["B1", "MX_LM10", "E_PCM"])
    onlySM = X[(X.n_SM > 0)]
    table("rows with >= 1 same-medium own-campaign donor", onlySM, ["B1", "MX_LM10", "MX_LM10_SM", "E_PCM"])
    print(f"done {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Skeptic-B check 2 on idea lab-1 (E-LAB, MX_LM10): near-copy donors and a medium-matched other-campaign placebo.

EXPLORATORY: 229 already-seen C3 cells, 5 dev seeds, BPG only.  Read-only on existing files.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab1skepB_nearcopy.py

Arms (all level-matched at 1.0 log, K = 6.1, as MX_LM10):
  MX_LM10_NC  own-campaign other-extractant donors, minus label-free near-copies: a donor is a near-copy when it is in
              the same medium AND |log D_donor - y_target| <= 0.15 at every one of the target's 4 read metals
              (only the reads are used, so the rule is legal in the E-LAB scenario).
  E_PCMM      count-matched placebo (n_m per metal as MX_LM10) drawn from OTHER publications' donors of the same broad
              ligand class AND the same medium (same acid type, |dlog acid| <= 0.15, same diluent); 30 draws; rows
              where no such donor exists at a needed metal fall back to fewer donors (smaller w).
Also reports which publications have near-copy donors and how much of the gain they carry.
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
from lab1skepB_placebo import broad_class  # noqa: E402

GD = CO.GD
NPB = 30
pd.set_option("display.width", 250)


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
    Aa = f[acid_cols].fillna(0).to_numpy()
    Dd = f[dil_cols].fillna(0).to_numpy()
    acid_id = Aa.argmax(1) + 1000 * (Aa.max(1) > 0)
    dil_id = Dd.argmax(1) + 1000 * (Dd.max(1) > 0)
    lacid = np.log10(np.clip(f["cond__acid_concentration_M"].astype(float).to_numpy(), 1e-6, None))
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okd = np.array([c not in v2drop for c in ids])
    D2 = detrend(Y, 2, 6)
    has = np.isfinite(D2).any(1) & okd
    ar = np.arange(len(f))
    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    fitted = json.loads((paths.RES / "REF" / "fitted_BPG.json").read_text())
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
                with np.errstate(invalid="ignore"):
                    close = np.all(np.abs(Y[dX][:, reads] - yi[reads][None, :]) <= 0.15, axis=1)
                nc = same_med[dX] & close
                dNC = dX[~nc]
                L = {"B1": levels(yi, reads, metals, u)}
                lv = L["B1"]

                def lev(cw):
                    c, w = cw
                    return levels(yi, reads, metals, (1 - w) * u + w * c)
                L["MX_LM10"] = lev(lm_shape(D2, Y, dX, lv, 1.0, K6))
                L["MX_LM10_NC"] = lev(lm_shape(D2, Y, dNC, lv, 1.0, K6))
                A = D2[dX]
                nm = (np.isfinite(A) & (np.abs(Y[dX] - lv[None, :]) <= 1.0)).sum(0)
                pool = np.where(base & (pub != pub[i]) & (cls == cls[i]) & same_med)[0]
                rng = np.random.default_rng(int(hashlib.sha256((t + str(seed) + "mm").encode()).hexdigest()[:8], 16))
                PB = []
                cover = 0
                for _ in range(NPB):
                    c = np.zeros(N_LN)
                    w = np.zeros(N_LN)
                    for m in range(N_LN):
                        if nm[m] == 0:
                            continue
                        cand = pool[np.isfinite(D2[pool, m]) & (np.abs(Y[pool, m] - lv[m]) <= 1.0)]
                        if len(cand) == 0:
                            continue
                        k = min(int(nm[m]), len(cand))
                        dd = rng.choice(cand, k, replace=False)
                        c[m] = D2[dd, m].mean()
                        w[m] = k / (k + K6)
                    cover = max(cover, int((w > 0).sum()))
                    PB.append(lev((c, w)))
                for r in pcg[t].itertuples(index=False):
                    ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                    o = {"split_seed": seed, "fold": fold, "cell_id": t, "publication": pub[i], "cls": cls[i],
                         "pair": f"{r.A}/{r.B}", "y": r.y, "n_nc": int(nc.sum()), "n_dX": len(dX),
                         "pool_mm": len(pool), "cover_mm": cover}
                    for k2, l2 in L.items():
                        o[k2] = l2[ia] - l2[ib]
                    o["E_PCMM"] = float(np.mean([(l2[ia] - l2[ib] - r.y) ** 2 for l2 in PB]))
                    rows.append(o)
        print(f"seed {seed} done {time.time() - t0:.0f}s", flush=True)
    T = pd.DataFrame(rows)

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

    print("\ncells with >= 1 near-copy own-campaign donor:", T[T.n_nc > 0].cell_id.nunique(), "of", T.cell_id.nunique(),
          "; by publication:", T[T.n_nc > 0].groupby("publication").cell_id.nunique().to_dict())
    table("POPX (BPG)", T, ["B1", "MX_LM10", "MX_LM10_NC"])
    inf("POPX near-copy", T, [("MX_LM10", "B1"), ("MX_LM10_NC", "B1"), ("MX_LM10", "MX_LM10_NC")])
    nc = T[T.n_nc > 0]
    if len(nc):
        table("cells with a near-copy donor", nc, ["B1", "MX_LM10", "MX_LM10_NC"])
    mm = T[T.pool_mm > 0]
    table("POPX cells with >= 1 same-class same-medium other-publication donor", mm, ["B1", "MX_LM10", "E_PCMM"])
    if mm.publication.nunique() >= 3:
        inf("medium-matched placebo", mm, [("MX_LM10", "E_PCMM"), ("E_PCMM", "B1")])
    print("   cover (metals with a placebo donor) median:", float(mm.groupby("cell_id").cover_mm.first().median()) if len(mm) else None)
    print(f"done {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

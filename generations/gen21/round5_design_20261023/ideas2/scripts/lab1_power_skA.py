#!/usr/bin/env python
"""Skeptic A for idea lab-1, step 4.  EXPLORATORY (seen rows).

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab1_power_skA.py

1. Kill-rule operating characteristics.  Treat the in-corpus laboratory units of POPX / POPX0 (BPG, MX_LM10 vs B1;
   per-unit SSE totals) as the truth (optimistic: the arm was selected on them) and resample n units with replacement.
   Kill rule of the idea: kill if the unit median of per-unit relative SSE <= 0 OR the one-sided 95 % lower bound of
   the mean is <= 0.  Two readings of 'mean': (a) equal-weight mean of per-unit rel SSE, t bound; (b) pooled rel SSE,
   LOPO-jackknife t bound.  Reports P(kill) for n = 6, 8, 11, 16, 25.
2. Same experiment series vs other series: MX_LM10 with donors restricted to cells sharing an experiment_series_id with
   the target (same table/figure) vs not sharing one.
3. Corpus-wide scan of C3 cells for bit-identical or offset-identical series (>= 6 shared metals) that the round-3
   V2 lists do not flag.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402
from lab1_recompute_skA import qdev, interp_levels, lm_cw, r2, rel, infer, lab_map  # noqa: E402


def kill_prob(sa, sb, n, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    G = len(sa)
    ka = kb = 0
    for _ in range(B):
        k = rng.integers(0, G, n)
        a, b = sa[k], sb[k]
        ru = (b - a) / b
        med = np.median(ru)
        lb_a = ru.mean() - SS.t.ppf(0.95, n - 1) * ru.std(ddof=1) / np.sqrt(n)
        pt = 1 - a.sum() / b.sum()
        jk = np.array([1 - (a.sum() - a[j]) / (b.sum() - b[j]) for j in range(n)])
        se = np.sqrt((n - 1) / n * ((jk - jk.mean()) ** 2).sum())
        lb_b = pt - SS.t.ppf(0.95, n - 1) * se
        ka += (med <= 0) or (lb_a <= 0)
        kb += (med <= 0) or (lb_b <= 0)
    return ka / B, kb / B


def main() -> int:
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okd = np.array([c not in v2drop for c in ids])
    Dv = np.array([qdev(Y[i]) for i in range(len(Y))])
    has = np.isfinite(Dv).any(1) & okd
    ar = np.arange(len(f))
    es = [set(str(x).split(";")) for x in f.experiment_series_ids]
    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    fitted = json.loads((paths.RES / "REF" / "fitted_BPG.json").read_text())
    l1 = pd.read_parquet(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\lab_out\lab_rows.parquet")
    nms = l1[l1.design == "BPG"].drop_duplicates("cell_id").set_index("cell_id").n_MS.to_dict()
    rows = []
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
                dX = np.where(has & (ar != i) & (pub == pub[i]) & (ext != ext[i]))[0]
                if not len(dX):
                    continue
                yi = Y[i]
                metals = list(cell["metals"])
                reads = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                lv = interp_levels(yi, reads, metals, u)
                sameS = np.array([len(es[i] & es[j]) > 0 for j in dX])
                L = {"B1": lv}
                for nm, dd in (("LM", dX), ("LM_sameS", dX[sameS]), ("LM_diffS", dX[~sameS])):
                    c, w = lm_cw(Dv, Y, dd, lv)
                    L[nm] = interp_levels(yi, reads, metals, (1 - w) * u + w * c)
                for r in pcg[t].itertuples(index=False):
                    ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                    o = {"split_seed": seed, "cell_id": t, "publication": pub[i], "y": r.y, "n_same": int(sameS.sum()),
                         "n_diff": int((~sameS).sum()), "n_MS": nms.get(t, 0)}
                    for k2, l2 in L.items():
                        o[k2] = l2[ia] - l2[ib]
                    rows.append(o)
    T = pd.DataFrame(rows)
    lm = lab_map(sorted(set(T.publication)))
    T["lab"] = T.publication.map(lm)
    print(f"POPX rows/seed {len(T) / 5:.0f}: LM rel {rel(T, 'LM'):+.4f} (check vs idea +0.0896)")
    # ---- 1. kill-rule operating characteristics ----
    for nm, s in (("POPX", T), ("POPX0", T[T.n_MS == 0])):
        g = s.assign(_a=(s.LM - s.y) ** 2, _b=(s.B1 - s.y) ** 2).groupby("lab")[["_a", "_b"]].sum()
        sa, sb = g._a.to_numpy(), g._b.to_numpy()
        ru = (sb - sa) / sb
        print(f"\n{nm}: {len(g)} lab units; per-unit rel SSE: {np.round(np.sort(ru), 3).tolist()}")
        for n in (6, 8, 11, 16, 25):
            pa, pb = kill_prob(sa, sb, n)
            print(f"   n = {n:2d}: P(kill | in-corpus effect is true)  (a) equal-weight mean rule {pa:.2f}   (b) pooled + jackknife rule {pb:.2f}")
        # null: sign-symmetrised effect (true effect zero) -> P(not killed) = false promote
        rng = np.random.default_rng(1)
        fp = 0
        B = 20000
        for _ in range(B):
            k = rng.integers(0, len(g), 8)
            ru8 = ru[k] * rng.choice((-1, 1), 8)
            lb = ru8.mean() - SS.t.ppf(0.95, 7) * ru8.std(ddof=1) / np.sqrt(8)
            fp += (np.median(ru8) > 0) and (lb > 0)
        print(f"   n = 8, sign-symmetrised null: P(survives the kill rule, rule a) {fp / B:.3f}")
    # ---- 2. same experiment series vs not ----
    S = T[(T.n_same > 0) & (T.n_diff > 0)]
    print(f"\nrows with donors both in the same experiment series and outside it: cells {S.cell_id.nunique()}, pubs "
          f"{S.publication.nunique()}, rows/seed {len(S) / 5:.0f}")
    if len(S):
        for a in ("LM", "LM_sameS", "LM_diffS"):
            print(f"   {a}: R2 {r2(S, a):.4f} rel vs B1 {rel(S, a):+.4f}")
        r = infer(S, "LM_sameS", "LM_diffS")
        print(f"   LM_sameS - LM_diffS: rel {r['rel']:+.4f} [{r['lo95']:+.3f}, {r['hi95']:+.3f}] up/down {r['up']}/{r['down']} p {r['p_sf']:.3f}")
    print(f"targets whose donors all share a series: {T[T.n_diff == 0].cell_id.nunique()}; none share: {T[T.n_same == 0].cell_id.nunique()}")
    # ---- 3. corpus-wide duplicate scan among C3 cells ----
    c3 = sorted(set(par.cell_id))
    dup = []
    for a in range(len(c3)):
        ia = pos[c3[a]]
        for b in range(a + 1, len(c3)):
            ib = pos[c3[b]]
            sh = np.isfinite(Y[ia]) & np.isfinite(Y[ib])
            if sh.sum() < 6:
                continue
            dy = Y[ia, sh] - Y[ib, sh]
            sd = float(np.std(dy - dy.mean()))
            if sd < 0.02:
                dup.append({"a": c3[a], "b": c3[b], "pub_a": pub[ia], "pub_b": pub[ib], "ext_a": f.extractant_name[ia],
                            "ext_b": f.extractant_name[ib], "n_sh": int(sh.sum()), "mean_dy": round(float(dy.mean()), 3),
                            "offset_sd": round(sd, 4), "v2_a": c3[a] in v2drop, "v2_b": c3[b] in v2drop})
    Dd = pd.DataFrame(dup)
    print(f"\nC3 cell pairs with offset-removed residual sd < 0.02 on >= 6 shared metals: {len(Dd)}")
    if len(Dd):
        pd.set_option("display.width", 250)
        print(Dd.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

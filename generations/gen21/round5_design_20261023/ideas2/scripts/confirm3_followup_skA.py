#!/usr/bin/env python
"""SKEPTIC A for confirm-3, follow-up (EXPLORATORY, 229 already-seen C3 cells).
(a) algebraic identity of the chord link with the correct sign (pair y = level(A) - level(B); Gd potential = -d);
(b) rule accuracy split by whether DOPT_Q reads Gd;  (c) one-unit-per-lab null worlds centred on the lab-level mean;
(d) A3 on chord units plus non-chord units at the corpus proportion (the test design says 'all units for A3');
(e) the pub_e9799b75f3 near-identical chords.  Writes only ideas2/confirm3_skA_out/followup_*.
"""
from __future__ import annotations

import itertools
import json
import sys
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "confirm3_skA_out"
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402
from gen20k import data as D  # noqa: E402

LN = list(D.LANTHANIDES)
BETA = -0.0923
CLIP = (-0.40, 0.20)
R = 4000


def main() -> int:
    res = {}
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    iE, iG, iT = (LN.index(m) for m in ("Eu", "Gd", "Tb"))
    c = Y[:, iG] - 0.5 * (Y[:, iE] + Y[:, iT])
    cm = pd.Series(c, index=f.cell_id.astype(str))
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    t = ref[(ref.k_max >= 3) & (ref.k == 3)].copy()
    t["gd_read"] = t.read_set.str.split(",").map(lambda s: "Gd" in s)
    t["c"] = t.cell_id.map(cm)
    t["g"] = (t.B0 - t.y) ** 2 - (t.B1 - t.y) ** 2
    K = ["split_seed", "cell_id", "read_set", "policy", "draw"]
    # (a) identity on Gd-hidden rows
    h = t[~t.gd_read & t.c.notna()]
    eg = h[(h.A == "Eu") & (h.B == "Gd")].set_index(K)
    gt = h[(h.A == "Gd") & (h.B == "Tb")].set_index(K)
    jj = eg[["B0", "B1", "g", "c"]].join(gt[["g"]], rsuffix="_t", how="inner")
    beta_g = -(jj.B1 - jj.B0)                       # Gd potential of the fold
    jj["g_alg"] = 2 * (jj.c ** 2 - (jj.c - beta_g) ** 2)
    res["identity_gd_hidden"] = {"n": int(len(jj)), "beta_fold_min": float(beta_g.min()), "beta_fold_max": float(beta_g.max()),
                                 "max_abs_gain_minus_algebra": float((jj.g + jj.g_t - jj.g_alg).abs().max())}
    # (b) rule accuracy by Gd-read status, cell level (seed-summed gain)
    ns = t.split_seed.nunique()
    acc = {}
    for nm, q in (("gd_hidden", t[~t.gd_read]), ("gd_read", t[t.gd_read])):
        cg = q.groupby("cell_id").g.sum() / ns
        d = pd.DataFrame({"gain": cg, "c": cm.reindex(cg.index)}).dropna()
        d = d[d.gain.abs() > 1e-12]
        acc[nm] = {"cells": int(len(d)), "rule_acc": float(((d.c < BETA / 2) == (d.gain > 0)).mean()),
                   "spearman": float(SS.spearmanr(d.c, d.gain)[0]), "gain_sum": float(d.gain.sum())}
    res["rule_by_gd_read"] = acc
    print(json.dumps(res, indent=1), flush=True)

    # unit table
    from r4 import donor as DN
    t["pubchem"] = t.publication.astype(str) + "|" + t.chemotype.astype(str)
    g = t.assign(er=(t.B0 - t.y) ** 2, ec=(t.B1 - t.y) ** 2).groupby(["pubchem", "split_seed"])[["er", "ec"]].sum()
    us = g.groupby(level=0).mean()
    us["dga"] = t.assign(dga=t.extractant.astype(str).map(DN.is_dga)).groupby("pubchem").dga.mean().reindex(us.index) > 0.5
    us["pub"] = t.groupby("pubchem").publication.first().reindex(us.index)
    cc = t.dropna(subset=["c"]).drop_duplicates("cell_id").groupby("pubchem").c.mean()
    us["c"] = cc.reindex(us.index)
    nd = us[~us.dga]
    ndc, ndn = nd[nd.c.notna()], nd[nd.c.isna()]

    # (c) one-unit-per-lab null worlds, centred on the lab-level winsorised mean
    pubs = sorted(ndc.pub.unique())
    byp = {p: np.clip(ndc[ndc.pub == p].c.to_numpy(), *CLIP) for p in pubs}
    mu_lab = float(np.mean([v.mean() for v in byp.values()]))
    rows = []
    for G in (6, 8, 13):
        for world, cen in (("NULL_A1_0", 0.0), ("NULL_A2_halfbeta", BETA / 2), ("WORLD_beta", BETA), ("EMP", None)):
            rng = np.random.default_rng(20261023 + zlib.crc32(f"fu|{G}|{world}".encode()))
            pi = rng.integers(len(pubs), size=(R, G))
            X = np.array([[rng.choice(byp[pubs[k]]) for k in row] for row in pi])
            if cen is not None:
                X = X - mu_lab + cen
            m, se = X.mean(1), X.std(1, ddof=1) / np.sqrt(G)
            tc = SS.t.ppf(0.975, G - 1)
            rows.append({"G_labs": G, "world": world, "A1_rej": float(((m / se) < -tc).mean()),
                         "A2_rej": float((((m - BETA / 2) / se) < -tc).mean()),
                         "A1_kill_nonrejection": float(((m / se) >= -tc).mean())})
    lab = pd.DataFrame(rows)
    print(f"lab-level winsorised mean {mu_lab:.4f}", flush=True)
    print(lab.round(3).to_string(), flush=True)
    lab.to_csv(OUT / "followup_lab_null.csv", index=False)

    # (d) chain with A3 on chord units + non-chord units (corpus proportion 8/25)
    E = {G: np.array(list(itertools.product((1.0, -1.0), repeat=G))) for G in (10, 11, 17)}
    crow = []
    for G in (8, 13):
        extra = int(round(G * len(ndn) / len(ndc)))
        for tau in (0.0, 0.065):
            rng = np.random.default_rng(20261023 + zlib.crc32(f"fu3|{G}|{tau}".encode()))
            ui = rng.integers(len(ndc), size=(R, G))
            vi = rng.integers(len(ndn), size=(R, extra))
            X = np.clip(ndc.c.to_numpy(), *CLIP)[ui]
            sr = np.hstack([ndc.er.to_numpy()[ui], ndn.er.to_numpy()[vi]])
            sc = np.hstack([ndc.ec.to_numpy()[ui], ndn.ec.to_numpy()[vi]])
            if tau > 0:
                s = tau / np.sqrt(2)
                sr = sr * (1 - rng.normal(0, s, sr.shape)) ** 2
                sc = sc * (1 - rng.normal(0, s, sc.shape)) ** 2
            L = np.log(sr / sc)
            Gt = G + extra
            Ee = E[Gt] if Gt in E else rng.choice((1.0, -1.0), size=(4999, Gt))
            pL = ((L @ Ee.T) >= L.sum(1)[:, None] - 1e-12).sum(1) / Ee.shape[0]
            a3 = (pL <= 0.025 + 1e-12) & (sr.sum(1) > sc.sum(1))
            m, se = X.mean(1), X.std(1, ddof=1) / np.sqrt(G)
            tc = SS.t.ppf(0.975, G - 1)
            a1, a2 = (m / se) < -tc, ((m - BETA / 2) / se) < -tc
            crow.append({"G_chord": G, "extra_nonchord": extra, "tau": tau, "A3": float(a3.mean()),
                         "chain": float((a1 & a2 & a3).mean())})
    ch = pd.DataFrame(crow)
    print(ch.round(3).to_string(), flush=True)
    ch.to_csv(OUT / "followup_chain_A3all.csv", index=False)
    print("non-chord non-DGA units:", ndn[["er", "ec"]].assign(L=np.log(ndn.er / ndn.ec)).round(4).to_string(), flush=True)

    # (e) pub_e9799b75f3
    q = f[f.publication_id.astype(str) == "pub_e9799b75f3"]
    print(pd.DataFrame(Y[q.index], index=q.cell_id.astype(str) + "|" + q.chemotype.astype(str), columns=LN).round(4).to_string())
    print(q[["cell_id", "chemotype"]].assign(c=c[q.index]).to_string())
    (OUT / "followup_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Skeptic B check for idea confirm-3 (C-B1 chain A1 -> A2 -> A3).  EXPLORATORY: 229 already-seen C3 cells, 5 dev seeds.

Questions:
  Q1  Is the rule "chord < beta/2 <=> B1 beats B0" exact only when Gd is hidden?  Split cell x seed rows by read
      configuration (Gd read / Gd hidden; Eu and Tb both read).  Where does B1 change predictions when Gd is read?
  Q2  Kill-criterion realism.  On the same drawn units as confirm_chain.py (pubchem units with a chord):
        A1-kill  = "point estimate >= 0 or 97.5 % upper bound > 0"  (= A1 not rejected)  -> withdraw the Gd term
        A2-fut   = 97.5 % lower bound of the mean chord > beta/2                         -> deploy B0
      How often does each fire in worlds where B1 is in truth better on average (pool mean chord < beta/2 and pool
      equal-weight log SSE ratio > 0)?  Compare with the plug-in rule "B1 iff point estimate < beta/2".
  Q3  Does the A1/A2 gate cost A3 power (A3 alone vs chain)?
  Q4  Expected-supply world: heavy-selective non-DGA units (the stratum P4 says the supply fills).  Is the chain
      outcome predictable from chemistry (power ~1)?
  Q5  Decision value: expected excess Gd-pair loss for a FUTURE unit drawn from the same pool, mean (c - a)^2, for
      a = 0 (B0), a = beta (B1 frozen), a = chain decision (B1 on pass / B0 on kill or futility / B1 status quo
      otherwise), a = re-estimated amplitude (mean winsorised chord of the G new units).
Writes only ideas2/confirm3_skB_out/.
"""
from __future__ import annotations

import itertools
import json
import sys
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
ROOT5 = HERE.parents[2]
OUT = HERE.parents[1] / "confirm3_skB_out"
P2DIR = ROOT5 / "probes" / "scripts"
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
sys.path.insert(0, str(P2DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

import P2_power_replay as P2  # noqa: E402
from r4 import donor as DN  # noqa: E402
from r4 import paths  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k import data as D  # noqa: E402

BETA = -0.0923
CLIP = (-0.40, 0.20)
R = 4000
GS = (6, 8, 10, 13)


def flips(G):
    return np.array(list(itertools.product((1.0, -1.0), repeat=G)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    # ---------------- cell chords and selectivity direction (bench, full series) ----------------
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    LN = list(D.LANTHANIDES)
    XR = interp.X_R
    iEu, iGd, iTb = LN.index("Eu"), LN.index("Gd"), LN.index("Tb")
    tt = (XR[iGd] - XR[iEu]) / (XR[iTb] - XR[iEu])
    c = Y[:, iGd] - ((1 - tt) * Y[:, iEu] + tt * Y[:, iTb])
    sl = np.full(len(Y), np.nan)
    for k in range(len(Y)):
        m = np.isfinite(Y[k])
        if m.sum() >= 5:
            sl[k] = np.polyfit(XR[m], Y[k, m], 1)[0]
    cell = pd.DataFrame({"cell_id": f.cell_id.to_numpy(), "c": c, "hs": sl > 0,
                         "pubchem": (f.publication_id.astype(str) + "|" + f.chemotype.astype(str)).to_numpy()})
    cell = cell.set_index("cell_id")

    # ---------------- Q1: rule exactness by read configuration ----------------
    t = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    t = t[(t.k_max >= 3) & (t.k == 3)].copy()
    t["g"] = (t.B0 - t.y) ** 2 - (t.B1 - t.y) ** 2
    t["changed"] = (t.B1 - t.B0).abs() > 1e-12
    t["gdpair"] = t.A.isin(["Eu", "Gd"]) & t.B.isin(["Gd", "Tb"])
    rs = t.read_set.astype(str).str.split(",")
    t["gd_read"] = rs.map(lambda s: "Gd" in s)
    t["eutb_read"] = rs.map(lambda s: ("Eu" in s) and ("Tb" in s))
    cs = t.groupby(["cell_id", "split_seed"]).agg(g=("g", "sum"), gd_read=("gd_read", "first"),
                                                    eutb_read=("eutb_read", "first"),
                                                    n_changed=("changed", "sum"),
                                                    n_changed_gd=("changed", lambda s: 0))
    cs["n_changed_gd"] = t[t.gdpair].groupby(["cell_id", "split_seed"]).changed.sum().reindex(cs.index).fillna(0)
    cs["g_gd"] = t[t.gdpair].groupby(["cell_id", "split_seed"]).g.sum().reindex(cs.index).fillna(0)
    cs = cs.reset_index()
    cs["c"] = cell.c.reindex(cs.cell_id).to_numpy()
    cs = cs.dropna(subset=["c"])
    q1 = []
    for lab, q in (("all", cs), ("Gd_read", cs[cs.gd_read]), ("Gd_hidden", cs[~cs.gd_read]),
                   ("Gd_hidden_EuTb_read", cs[~cs.gd_read & cs.eutb_read])):
        if len(q) == 0:
            continue
        win = q.g > 1e-15
        tie = q.g.abs() <= 1e-15
        pred = q.c < BETA / 2
        q1.append({"subset": lab, "cell_seed_rows": len(q), "share_rows": len(q) / len(cs),
                   "share_of_total_gain": float(q.g.sum() / cs.g.sum()),
                   "share_gain_on_gd_pairs_within": float(q.g_gd.sum() / q.g.sum()) if q.g.sum() != 0 else np.nan,
                   "mean_pairs_changed": float(q.n_changed.mean()),
                   "mean_gd_pairs_changed": float(q.n_changed_gd.mean()),
                   "ties": int(tie.sum()),
                   "rule_accuracy_nontied": float((pred[~tie] == win[~tie]).mean()),
                   "spearman_c_gain": float(SS.spearmanr(q.c, q.g)[0])})
    q1 = pd.DataFrame(q1)
    q1.to_csv(OUT / "q1_rule_by_readconfig.csv", index=False)
    print(q1.round(3).to_string(), flush=True)

    # ---------------- unit pools (as confirm_chain.py) ----------------
    tb = DN.add_donor(P2.load("BPG"))
    us = P2.unit_stats(tb, "pubchem", "B1", "B0")
    dga = tb.groupby("pubchem").dga_donor.mean().loc[us.index] > 0.5
    uc = cell.dropna(subset=["c"]).groupby("pubchem").agg(c=("c", "mean"), hs=("hs", "mean"))
    us = us.assign(c=uc.c.reindex(us.index), hsf=uc.hs.reindex(us.index), dga=dga)
    us["hs"] = us.hsf > 0.5
    pools = {"all_chord": us.c.notna(), "nonDGA_chord": us.c.notna() & ~us.dga,
             "DGA_chord": us.c.notna() & us.dga, "nonDGA_HS": us.c.notna() & ~us.dga & us.hs,
             "nonDGA_LS": us.c.notna() & ~us.dga & ~us.hs}
    rows, truth = [], []
    for pn, keep in pools.items():
        P = us[keep]
        craw = P.c.to_numpy()
        cw = np.clip(craw, *CLIP)
        Sr, Sc = P.er.to_numpy(), P.ec.to_numpy()
        L = np.log(Sr / Sc)
        tr = {"pool": pn, "units": len(P), "mean_cw": float(cw.mean()), "sd_cw": float(cw.std(ddof=1)),
              "mean_L": float(L.mean()), "pooled_dSSE": float((Sr - Sc).sum()),
              "B1_better_by_chord": bool(cw.mean() < BETA / 2), "B1_better_by_L": bool(L.mean() > 0),
              # future-unit excess Gd-pair loss (raw chords) for fixed amplitudes
              "loss_B0": float(np.mean(craw ** 2)), "loss_B1": float(np.mean((craw - BETA) ** 2)),
              "loss_oracle_mean": float(np.mean((craw - craw.mean()) ** 2))}
        truth.append(tr)
        for tau in (0.0, 0.065):
            for G in GS:
                rng = np.random.default_rng(20261023 + zlib.crc32(f"c3skB|{pn}|{tau}|{G}".encode()))
                ui = rng.integers(len(P), size=(R, G))
                X = cw[ui]
                sr, sc = Sr[ui].copy(), Sc[ui].copy()
                if tau > 0:
                    s = tau / np.sqrt(2)
                    sr *= (1 - rng.normal(0, s, size=sr.shape)) ** 2
                    sc *= (1 - rng.normal(0, s, size=sc.shape)) ** 2
                m = X.mean(1)
                se = X.std(1, ddof=1) / np.sqrt(G)
                tc = SS.t.ppf(0.975, G - 1)
                ub, lb = m + tc * se, m - tc * se
                Lx = np.log(sr / sc)
                E = flips(G)
                pL = ((Lx @ E.T) >= Lx.sum(1)[:, None] - 1e-12).sum(1) / E.shape[0]
                guard = sr.sum(1) > sc.sum(1)
                a1 = ub < 0
                a2 = (m - BETA / 2) / se < -tc
                a3 = (pL <= 0.025 + 1e-12) & guard
                chain = a1 & a2 & a3
                a1kill = ~a1                      # "point estimate >= 0 or 97.5 % UB > 0"
                a2fut = lb > BETA / 2
                dec_b0 = (a1kill | a2fut) & ~chain
                dec_b1 = chain
                incon = ~dec_b0 & ~dec_b1
                plug_b1 = m < BETA / 2
                # future-unit excess loss: E_new[(c - a)^2] = var_pool + (mu_pool - a)^2 (raw chords)
                mu, v = craw.mean(), craw.var()
                a_chain = np.where(dec_b0, 0.0, BETA)          # inconclusive -> status quo B1
                a_chain_b0 = np.where(dec_b1, BETA, 0.0)        # inconclusive -> B0
                a_hat = m                                       # re-estimated amplitude from the G new units
                a_plug = np.where(plug_b1, BETA, 0.0)
                lossf = lambda a: float(np.mean(v + (mu - a) ** 2))  # noqa: E731
                rows.append({"pool": pn, "tau": tau, "G": G,
                             "A1": a1.mean(), "A2": a2.mean(), "A3_alone": a3.mean(), "chain": chain.mean(),
                             "A1kill_fires": a1kill.mean(), "A1kill_pointest_ge0": (m >= 0).mean(),
                             "A2fut_fires": a2fut.mean(), "decide_B0": dec_b0.mean(), "decide_B1": dec_b1.mean(),
                             "inconclusive": incon.mean(), "A1kill_and_A3pass": (a1kill & a3).mean(),
                             "plugin_B1": plug_b1.mean(),
                             "loss_B0": lossf(0.0), "loss_B1": lossf(BETA), "loss_chain_sqB1": lossf(a_chain),
                             "loss_chain_sqB0": lossf(a_chain_b0), "loss_plugin": lossf(a_plug),
                             "loss_reest": lossf(a_hat), "loss_oracle": float(v)})
    tr = pd.DataFrame(truth)
    rr = pd.DataFrame(rows)
    tr.to_csv(OUT / "q2_pool_truth.csv", index=False)
    rr.to_csv(OUT / "q2_q5_replay.csv", index=False)
    pd.set_option("display.width", 300)
    print(tr.round(4).to_string(), flush=True)
    cols = ["pool", "tau", "G", "A1", "A2", "A3_alone", "chain", "A1kill_fires", "A1kill_pointest_ge0", "A2fut_fires",
            "decide_B0", "decide_B1", "inconclusive", "A1kill_and_A3pass"]
    print(rr[cols].round(3).to_string(), flush=True)
    cols2 = ["pool", "tau", "G", "loss_B0", "loss_B1", "loss_chain_sqB1", "loss_chain_sqB0", "loss_plugin", "loss_reest",
             "loss_oracle"]
    print(rr[rr.tau == 0][cols2].round(4).to_string(), flush=True)
    res["q1"] = q1.to_dict(orient="records")
    res["truth"] = tr.to_dict(orient="records")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

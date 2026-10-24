#!/usr/bin/env python
"""confirm theme, check 4 (EXPLORATORY): joint power of the fixed-sequence B1-transport chain of the frozen package
  A1  winsorised unit-mean Gd chord < 0             (t, df G-1, one-sided 0.025)
  A2  winsorised unit-mean Gd chord < beta/2        (same test; = "B1's Gd term lowers the average unit's error")
  A3  equal-weight B1 - B0 log SSE ratio > 0         (exact sign-flip) with pooled guard sum dSSE > 0
on the SAME drawn units (publication x chemotype units of C3 that have an Eu-Gd-Tb chord), so the link between a
unit's chord and its B1 - B0 contrast is the real one.  Also the chain at Bonferroni alpha 0.025/3.
EXPLORATORY: 229 already-seen C3 cells.  Writes only ideas2/scripts/confirm_out/chain_*.
"""
from __future__ import annotations

import itertools
import json
import sys
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parent / "confirm_out"
P2DIR = HERE.parents[2] / "probes" / "scripts"
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
sys.path.insert(0, str(P2DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

import P2_power_replay as P2  # noqa: E402
from r4 import donor as DN  # noqa: E402

BETA = -0.0923
CLIP = (-0.40, 0.20)
C5, DMD = "pub_6a0ef8528c", "pub_8415538291"
GS = (5, 6, 8, 10, 13, 16)
R = 4000


def flips(G, rng):
    if G <= 13:
        return np.array(list(itertools.product((1.0, -1.0), repeat=G)))
    return rng.choice((1.0, -1.0), size=(4999, G))


def main() -> int:
    t = DN.add_donor(P2.load("BPG"))
    us = P2.unit_stats(t, "pubchem", "B1", "B0")
    dga = t.groupby("pubchem").dga_donor.mean().loc[us.index] > 0.5
    pub = t.groupby("pubchem").publication.first().loc[us.index]
    ch = pd.read_csv(OUT / "gd_units_C3_pubchem.csv").set_index("pubchem").c
    us = us.assign(c=ch.reindex(us.index), dga=dga, pub=pub)
    us["L"] = np.log(us.er / us.ec)
    out, dev = [], []
    pools = {"all_chord": us.c.notna(), "nonDGA_chord": us.c.notna() & ~us.dga,
             "nonDGA_chord_noC5DMD": us.c.notna() & ~us.dga & ~us.pub.isin([C5, DMD]),
             "DGA_chord": us.c.notna() & us.dga}
    for pn, keep in pools.items():
        P = us[keep]
        cw = np.clip(P.c.to_numpy(), *CLIP)
        Sr, Sc = P.er.to_numpy(), P.ec.to_numpy()
        L = np.log(Sr / Sc)
        dev.append({"pool": pn, "units": len(P), "chord_mean_w": float(cw.mean()), "chord_sd_w": float(cw.std(ddof=1)),
                    "B1B0_wins": int((Sr > Sc).sum()), "B1B0_losses": int((Sr < Sc).sum()), "meanL": float(L.mean()),
                    "spearman_chord_L": float(SS.spearmanr(P.c, L)[0]),
                    "agree_chord_lt_halfbeta_vs_win": float(((P.c < BETA / 2) == (Sr > Sc)).mean())})
        for tau in (0.0, 0.065):
            for G in GS:
                rng = np.random.default_rng(20261023 + zlib.crc32(f"chain|{pn}|{tau}|{G}".encode()))
                ui = rng.integers(len(P), size=(R, G))
                X = cw[ui]
                sr, sc = Sr[ui].copy(), Sc[ui].copy()
                if tau > 0:
                    s = tau / np.sqrt(2)
                    sr *= (1 - rng.normal(0, s, size=sr.shape)) ** 2
                    sc *= (1 - rng.normal(0, s, size=sc.shape)) ** 2
                m = X.mean(1)
                se = X.std(1, ddof=1) / np.sqrt(G)
                Lx = np.log(sr / sc)
                E = flips(G, rng)
                pL = ((Lx @ E.T) >= Lx.sum(1)[:, None] - 1e-12).sum(1) / E.shape[0]
                guard = sr.sum(1) > sc.sum(1)
                rec = {"pool": pn, "tau": tau, "G": G}
                for lab, a in (("a025", 0.025), ("bonf", 0.025 / 3)):
                    tc = SS.t.ppf(1 - a, G - 1)
                    a1 = (m / se) < -tc
                    a2 = ((m - BETA / 2) / se) < -tc
                    a3 = (pL <= a + 1e-12) & guard
                    rec[f"A1_{lab}"] = float(a1.mean())
                    rec[f"A2_{lab}"] = float(a2.mean())
                    rec[f"A3_{lab}"] = float(a3.mean())
                    rec[f"chain_A1A2_{lab}"] = float((a1 & a2).mean())
                    rec[f"chain_A1A2A3_{lab}"] = float((a1 & a2 & a3).mean())
                    rec[f"A2_or_A3_disagree_{lab}"] = float((a2 != a3).mean())
                out.append(rec)
    d = pd.DataFrame(dev)
    o = pd.DataFrame(out)
    d.to_csv(OUT / "chain_dev.csv", index=False)
    o.to_csv(OUT / "chain_replay.csv", index=False)
    pd.set_option("display.width", 250)
    print(d.round(3).to_string())
    print(o.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

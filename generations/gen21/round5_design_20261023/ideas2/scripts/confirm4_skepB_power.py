#!/usr/bin/env python
"""Skeptic B for confirm-4 (C-ECT), check 2 (EXPLORATORY; planning replay on the 14 already-seen PRIMARY units).
Operating characteristics of the confirm-4 decision rules under alternative worlds:
  S0 corpus resample (reproduces confirm's 0.63-0.67 / 0.89-0.93); S1 effect halved (L - 0.33); S2 win share ~0.71
  (L - 0.30); S3 null (sign-symmetrised corpus, SWAP-like).  tau = 0 and 0.065 as in confirm_bonf.py.
Rules: exact one-sided sign-flip at 0.025 on the primary alone; Holm over {SH4, SH3} at 0.025 (as written);
fixed sequence SH4 -> SH3; kill (mean <= 0 or < 4/7 share improving, scaled as share < 4/7); inconclusive = neither.
Lab clustering: families arrive in labs of size 2 with a shared lab component (ICC 0.5); family-level sign-flip
size under the null and lab-level sign-flip power.
Writes only ideas2/confirm4_skepB_out/power_*.csv.
"""
from __future__ import annotations
import itertools
import sys
import zlib
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parents[1] / "confirm4_skepB_out"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = 4000


def signflip_p(X):
    G = X.shape[1]
    E = np.array(list(itertools.product((1.0, -1.0), repeat=G)))
    return ((X @ E.T) >= X.sum(1)[:, None] - 1e-12).sum(1) / E.shape[0]


def noise(X, tau, rng):
    if tau <= 0:
        return X
    s = tau / np.sqrt(2)
    return X + 2 * np.log((1 - rng.normal(0, s, size=X.shape)) / (1 - rng.normal(0, s, size=X.shape)))


def main() -> int:
    U = pd.read_csv(OUT / "units_primary.csv")
    L4 = U.L_SH4.to_numpy(); L3 = U.L_SH3.to_numpy()
    print("pool: SH4 mean %.3f W %d/14; SH3 mean %.3f W %d/14" % (L4.mean(), (L4 > 0).sum(), L3.mean(), (L3 > 0).sum()))
    worlds = {"S0_corpus": 0.0, "S1_halved": L4.mean() / 2, "S2_win0.71": 0.30, "S3_null": None}
    rows = []
    for wname, shift in worlds.items():
        for tau in (0.0, 0.065):
            for G in (5, 6, 7, 8, 9, 10):
                rng = np.random.default_rng(20261024 + zlib.crc32(f"c4b|{wname}|{tau}|{G}".encode()))
                idx = rng.integers(len(L4), size=(R, G))
                A4, A3 = L4[idx].copy(), L3[idx].copy()
                if shift is None:
                    sg = rng.choice((-1.0, 1.0), size=(R, G))
                    A4 *= sg; A3 *= sg
                else:
                    A4 -= shift; A3 -= shift
                A4 = noise(A4, tau, rng); A3 = noise(A3, tau, rng)
                p4, p3 = signflip_p(A4), signflip_p(A3)
                prim = p4 <= 0.025 + 1e-12
                holm1 = np.minimum(p4, p3) <= 0.0125 + 1e-12
                holm_p4 = holm1 & ((p4 <= 0.0125 + 1e-12) | (p4 <= 0.025 + 1e-12))
                holm_p3 = holm1 & ((p3 <= 0.0125 + 1e-12) | (p3 <= 0.025 + 1e-12))
                seq3 = prim & (p3 <= 0.025 + 1e-12)
                share = (A4 > 0).mean(1)
                kill = (A4.mean(1) <= 0) | (share < 4 / 7)
                rows.append({"world": wname, "tau": tau, "G": G, "primary_alone": prim.mean(), "holm_primary": holm_p4.mean(),
                             "holm_secondary": holm_p3.mean(), "fixedseq_secondary": seq3.mean(), "kill": kill.mean(),
                             "inconclusive": (~prim & ~kill).mean(), "min_p": 1 / 2 ** G})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "power_rules.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.round(3).to_string())
    # ---- lab clustering: families come in labs of 2 (G_fam = 2 * G_lab); shared lab component, ICC 0.5
    rows = []
    sd = L4.std(ddof=1)
    for icc in (0.0, 0.3, 0.5):
        for G_lab in (3, 4, 5):
            for world in ("null", "corpus_mean", "half_mean"):
                mu = {"null": 0.0, "corpus_mean": L4.mean(), "half_mean": L4.mean() / 2}[world]
                rng = np.random.default_rng(777 + zlib.crc32(f"lab|{icc}|{G_lab}|{world}".encode()))
                lab = rng.normal(0, sd * np.sqrt(icc), size=(R, G_lab))
                fam = mu + np.repeat(lab, 2, axis=1) + rng.normal(0, sd * np.sqrt(1 - icc), size=(R, 2 * G_lab))
                pf = signflip_p(fam)
                pl = signflip_p(fam.reshape(R, G_lab, 2).mean(2))
                rows.append({"icc": icc, "G_lab": G_lab, "G_fam": 2 * G_lab, "world": world,
                             "family_level_reject": (pf <= 0.025 + 1e-12).mean(), "lab_level_reject": (pl <= 0.025 + 1e-12).mean(),
                             "lab_min_p": 1 / 2 ** G_lab})
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "power_labcluster.csv", index=False)
    print(c.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

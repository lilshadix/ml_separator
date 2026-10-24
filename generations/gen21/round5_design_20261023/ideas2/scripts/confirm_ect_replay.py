#!/usr/bin/env python
"""confirm theme, check 3 (EXPLORATORY): power of a frozen E-CT (condition transfer) confirmation leg on the few new
laboratories that will have sibling conditions (expected 4-9 families), via the P2 resampling machinery.

Input: P1_rows.parquet rebuilt by probes/scripts/P1_ct_build.py into ideas2/scripts/confirm_out/p1 (unchanged code).
EXPLORATORY: 229 already-seen C3 cells, 5 dev seeds; the strata and K were fixed after earlier looks at these rows.
Units: publication x chemotype (P1's unit) and publication x extractant (family).  Per-unit seed-mean SSE of each arm.
Contrasts (candidate vs reference, positive = candidate better):
  SH4-B1   SH at 4 reads vs B1 at 4 reads (primary E-CT)
  SH3-B1   SH at 3 reads vs B1 at 4 reads (one read saved)
  SH2-B1   SH at 2 reads vs B1 at 4 reads (two reads saved)
  COPY-B1  sibling mean template, 0 reads, vs B1 at 4 reads
  SH4-PBPUB SH vs the same-publication other-extractant blend (campaign-fingerprint control; rows where it exists)
Strata: PRIMARY (CLEAN, n_sib >= 2, no C5-BTBP), NSIB1 (n_sib = 1), by condition-change class.
Writes only ideas2/scripts/confirm_out/ect_*.
"""
from __future__ import annotations

import json
import sys
import time
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

import P2_power_replay as P2  # noqa: E402

GS = (3, 4, 5, 6, 7, 8, 9, 12)
TESTS = ("SIGN", "SF_LOG", "T_LOG", "SF_POOL", "WCR_POOL")
CONTRASTS = {"SH4-B1": ("CL_SH_3", "B1_3"), "SH3-B1": ("CL_SH_2", "B1_3"), "SH2-B1": ("CL_SH_1", "B1_3"),
             "COPY-B1": ("CL_COPY", "B1_3"), "SH4-PBPUB": ("CL_SH_3", "CL_PB_PUB"), "PBPUB-B1": ("CL_PB_PUB", "B1_3"),
             "SH4-OFFXEXT?": None}


def unit_stats(t, unit, cand, ref):
    g = t.assign(er=(t[ref] - t.y) ** 2, ec=(t[cand] - t.y) ** 2, y2=t.y ** 2, one=1.0)
    agg = g.groupby([unit, "split_seed"])[["one", "y", "y2", "er", "ec"]].sum()
    return agg.groupby(level=0).mean()


def main() -> int:
    t0 = time.time()
    t = pd.read_parquet(OUT / "p1" / "P1_rows.parquet")
    t = t[t.design == "BPG"].copy()
    t["pubchem"] = t.publication + "|" + t.chemotype
    t["pubext"] = t.publication + "|" + t.extractant
    clean = ~t.t_v2drop & (t.n_clean >= 1)
    strata = {"PRIMARY": clean & (t.n_clean >= 2) & ~t.c5,
              "NSIB1": clean & (t.n_clean == 1) & ~t.c5,
              "ANY_noC5": clean & ~t.c5,
              "PRIMARY_acid_conc_only": clean & (t.n_clean >= 2) & ~t.c5 & (t.cls == "acid_conc_only"),
              "PRIMARY_not_acidtype_pub": clean & (t.n_clean >= 2) & ~t.c5 & (t.publication != "pub_b0b3296ec5"),
              "PRIMARY_trivialish": clean & (t.n_clean >= 2) & ~t.c5 & t.cls.isin(["diluent_only", "extr_conc_only",
                                                                                   "temperature_only"])}
    dev, grid = [], []
    for sname, mask in strata.items():
        tt = t[mask]
        for unit in ("pubchem", "pubext"):
            for cname, spec in CONTRASTS.items():
                if spec is None:
                    continue
                cand, ref = spec
                ts = tt.dropna(subset=[cand, ref])
                if ts.empty:
                    continue
                us = unit_stats(ts, unit, cand, ref)
                p = P2.make_pool(us)
                dS = p["Sr"] - p["Sc"]
                L = np.log(p["Sr"] / p["Sc"])
                sst = (ts.groupby("split_seed").y.var(ddof=0) * ts.groupby("split_seed").y.size()).mean()
                dev.append({"stratum": sname, "unit": unit, "contrast": cname, "units": len(dS),
                            "pubs": ts.publication.nunique(), "cells": ts.cell_id.nunique(),
                            "rows_per_seed": len(ts) / ts.split_seed.nunique(),
                            "wins": int((dS > 0).sum()), "losses": int((dS < 0).sum()),
                            "meanL": float(L.mean()), "medianL": float(np.median(L)), "sdL": float(L.std(ddof=1)) if len(L) > 1 else np.nan,
                            "pooled_rel": float(1 - p["Sc"].sum() / p["Sr"].sum()),
                            "dR2": float(dS.sum() / sst),
                            "top1_share": float(np.abs(dS).max() / np.abs(dS).sum())})
                if len(dS) < 3 or sname not in ("PRIMARY", "NSIB1", "PRIMARY_acid_conc_only", "PRIMARY_not_acidtype_pub"):
                    continue
                if cname not in ("SH4-B1", "SH3-B1", "SH2-B1", "COPY-B1", "SH4-PBPUB"):
                    continue
                for world in ("EMP", "SWAP"):
                    for tau in (0.0, 0.065):
                        for G in GS:
                            R = 2000
                            rng = np.random.default_rng(20261023 + zlib.crc32(f"ect|{sname}|{unit}|{cname}|{world}|{tau}|{G}".encode()))
                            n, Sy, Sy2, Sr, Sc = P2.draw(p, G, R, rng, world, tau)
                            P, rej, extra = P2.run_tests(n, Sy, Sy2, Sr, Sc, rng, do_sealed=False)
                            rec = {"stratum": sname, "unit": unit, "contrast": cname, "world": world, "tau": tau, "G": G,
                                   "pool_units": len(dS)}
                            for k in TESTS:
                                rec[k + "_025"] = float(rej[k][0.025].mean())
                                rec[k + "_05"] = float(rej[k][0.05].mean())
                            grid.append(rec)
                print(f"{time.time() - t0:6.1f}s {sname} {unit} {cname}", flush=True)
    d = pd.DataFrame(dev)
    g = pd.DataFrame(grid)
    d.to_csv(OUT / "ect_dev.csv", index=False)
    g.to_csv(OUT / "ect_replay.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(d.round(3).to_string())
    x = g[(g.world == "EMP")]
    print(x[["stratum", "unit", "contrast", "tau", "G", "SIGN_025", "SF_LOG_025", "T_LOG_025", "SF_POOL_025",
             "SIGN_05", "SF_LOG_05"]].round(3).to_string())
    s = g[g.world == "SWAP"].groupby(["stratum", "unit", "contrast"])[[k + "_025" for k in TESTS]].max()
    print(s.round(3).to_string())
    print("seconds", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

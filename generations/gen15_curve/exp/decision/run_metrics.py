"""Compute every decision metric, for every arm, under every design, from the cached pair tables.

Writes one CSV per question into ``results/``.  Nothing here refits a model; the pair tables in
``tables/`` are the frozen out-of-fold predictions of ``run_tables.py``.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "generations" / "gen15_curve"))
sys.path.insert(0, str(HERE))
warnings.filterwarnings("ignore")

from gen15 import valuebench as V           # noqa: E402
from gen13sep.metals import LANTHANIDES     # noqa: E402
from gen13sep.metrics import heavier_always_sign_accuracy  # noqa: E402
import decmetrics as M                      # noqa: E402

DESIGNS = ("B", "BR", "BQ", "A", "BP")
ARMS = ["FLAT", "MEAN_CURVE", "G14", "G13_FULL", "O_BOTH"]
REF = ["HEAVIER_ALWAYS"]
OUT = HERE / "results"
TAB = HERE / "tables"


def load_design(bench, design: str) -> pd.DataFrame:
    tab = M.prepare(pd.read_parquet(TAB / f"pairs_{design}.parquet"), bench.basis, LANTHANIDES)
    conf = pd.read_csv(TAB / f"g14_conf_{design}.csv")
    tab = tab.merge(conf, on=["split_seed", "fold", "cell_id"], how="left", validate="many_to_one")
    # fold-artifact control: gen14's direction bit with ONE global magnitude and curvature, so
    # every same-direction system is exactly tied and cross-fold jitter cannot rank extractants
    mag, cur = conf["train_mean_magnitude"].mean(), conf["train_mean_curvature"].mean()
    tab["G14_TIED"] = tab["sign"] * mag * tab["dr"] + cur * tab["dr2"]
    return tab


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bench = V.load()
    arms = ARMS + REF + ["G14_TIED"]
    q1, q2, q2cells, q3, q3ind, q3cell, q4s, q4b, cov, chem, nmet = [], [], [], [], [], [], [], [], [], [], []
    for d in DESIGNS:
        t0 = time.time()
        tab = load_design(bench, d)

        # ---- Q1 direction --------------------------------------------------------------
        units = M.sign_units(tab, arms)
        sb = M.sign_from_units(units)
        sb.insert(0, "design", d)
        q1.append(sb)

        # ---- Q2 within-cell ------------------------------------------------------------
        wc, cells = M.within_cell(tab, arms)
        wc.insert(0, "design", d)
        q2.append(wc)
        cells.insert(0, "design", d)
        q2cells.append(cells)

        # ---- Q3 cross-extractant -------------------------------------------------------
        ce, per_pair = M.cross_extractant(tab, arms, unit="extractant")
        ce.insert(0, "design", d)
        q3.append(ce)
        it = M.industrial_table(per_pair, arms)
        it.insert(0, "design", d)
        q3ind.append(it)
        cc, _ = M.cross_extractant(tab, arms, unit="cell")
        cc.insert(0, "design", d)
        q3cell.append(cc)

        # ---- confound: the number of metals a cell measured (Spearman +0.49 with |a|) ---
        band = pd.cut(cells["n_metals"], [3, 5, 8, 13, 14],
                      labels=["4-5", "6-8", "9-13", "14"])
        for (bnd, arm), blk in cells.assign(_b=band).groupby(["_b", "arm"], observed=True):
            m_s, _ = M._macro(blk, "spearman")
            m_t, _ = M._macro(blk, "top1")
            nmet.append({"design": d, "n_metals_band": str(bnd), "arm": arm,
                         "n_cells": int(blk["cell_id"].nunique()),
                         "cell_spearman": m_s, "cell_top1": m_t})

        # ---- Q4 calibration and coverage -----------------------------------------------
        for arm in arms:
            sl, ic, sd, r2 = M.calibration_slope(tab, arm)
            q4s.append({"design": d, "arm": arm, "slope": sl, "intercept": ic,
                        "slope_sd_seed": sd, "r2": r2})
            b = M.calibration_bins(tab, arm)
            b.insert(0, "arm", arm)
            b.insert(0, "design", d)
            q4b.append(b)
            c = M.coverage(tab, arm, tab[arm].abs().to_numpy())
            c.insert(0, "confidence", "|prediction|")
            c.insert(0, "design", d)
            cov.append(c)
        c = M.coverage(tab, "G14", tab["dir_conf"].to_numpy())
        c.insert(0, "confidence", "gen14 |p-0.5|")
        c.insert(0, "design", d)
        cov.append(c)

        # ---- confound: leave-one-chemotype-out stability of the headline numbers --------
        # Q1 and Q2 re-aggregate the cached per-unit tables exactly; Q3 must be recomputed
        # because dropping a chemotype changes the pool of candidate extractants to rank.
        sizes = tab.groupby("chemotype")["extractant"].nunique().sort_values(ascending=False)
        for ct in sizes.index:
            keep_u = units[units["chemotype"] != ct]
            keep_c = cells[cells["chemotype"] != ct]
            sub = tab[tab["chemotype"] != ct]
            s = M.sign_from_units(keep_u)
            w = M.within_from_cells(keep_c)
            x, _ = M.cross_extractant(sub, arms)
            for arm in arms:
                chem.append({"design": d, "dropped_chemotype": ct, "arm": arm,
                             "n_extractants_dropped": int(sizes[ct]),
                             "n_extractants_left": int(sub["extractant"].nunique()),
                             "sign_acc": float(s[(s.band == "all") & (s.arm == arm)].sign_acc.iat[0]),
                             "cell_spearman": float(w[w.arm == arm].spearman.iat[0]),
                             "cell_top1": float(w[w.arm == arm].top1.iat[0]),
                             "cross_spearman": float(x[x.arm == arm].spearman.iat[0]),
                             "cross_regret": float(x[x.arm == arm].regret.iat[0])})
        print(f"[{d}] metrics in {time.time() - t0:.0f}s "
              f"(heavier-always repo check {heavier_always_sign_accuracy(tab):.4f})", flush=True)

    pd.concat(q1, ignore_index=True).to_csv(OUT / "q1_sign.csv", index=False)
    pd.concat(q2, ignore_index=True).to_csv(OUT / "q2_within_cell.csv", index=False)
    pd.concat(q2cells, ignore_index=True).to_csv(OUT / "q2_cells_long.csv", index=False)
    pd.concat(q3, ignore_index=True).to_csv(OUT / "q3_cross_extractant.csv", index=False)
    pd.concat(q3ind, ignore_index=True).to_csv(OUT / "q3_industrial.csv", index=False)
    pd.concat(q3cell, ignore_index=True).to_csv(OUT / "q3_cross_cell_unit.csv", index=False)
    pd.DataFrame(q4s).to_csv(OUT / "q4_calibration_slope.csv", index=False)
    pd.concat(q4b, ignore_index=True).to_csv(OUT / "q4_calibration_bins.csv", index=False)
    pd.concat(cov, ignore_index=True).to_csv(OUT / "q4_coverage.csv", index=False)
    pd.DataFrame(chem).to_csv(OUT / "confound_loco.csv", index=False)
    pd.DataFrame(nmet).to_csv(OUT / "confound_n_metals.csv", index=False)
    print("written to", OUT)


if __name__ == "__main__":
    main()

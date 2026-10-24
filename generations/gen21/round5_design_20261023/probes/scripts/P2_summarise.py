#!/usr/bin/env python
"""P2 (M34): summarise the replay grids written by P2_power_replay.py into the tables of P2_power_replay.md.
Reads probes/P2_out/P2_grid_<design>.csv; prints tables; writes probes/P2_out/P2_summary.json.
    .venv/Scripts/python.exe -B generations/gen21/round5_design_20261023/probes/scripts/P2_summarise.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "P2_out"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 500)
MAIN_TESTS = ["SIGN", "SF_LOG", "WCR_LOG", "T_LOG", "SF_POOL", "WCR_POOL", "JK_KISH", "JK_G1", "IM_REL", "SEALED"]
UNITS = ["cluster_cpg", "publication", "chemotype", "pubchem"]


def g_needed(sub: pd.DataFrame, col: str, target=0.8):
    """Smallest grid G with power >= target, and a log-G interpolation between the bracketing grid points."""
    s = sub.sort_values("G")
    Gs, pw = s.G.to_numpy(float), s[col].to_numpy(float)
    ok = np.where(pw >= target)[0]
    if not len(ok):
        return None, None
    i = ok[0]
    if i == 0:
        return int(Gs[0]), float(Gs[0])
    g0, g1, p0, p1 = Gs[i - 1], Gs[i], pw[i - 1], pw[i]
    lg = np.log(g0) + (target - p0) / (p1 - p0) * (np.log(g1) - np.log(g0)) if p1 > p0 else np.log(g1)
    return int(Gs[i]), float(np.exp(lg))


def main() -> int:
    designs = [d for d in ("BPG", "BP") if (OUT / f"P2_grid_{d}.csv").exists()]
    g = pd.concat([pd.read_csv(OUT / f"P2_grid_{d}.csv") for d in designs], ignore_index=True)
    summ = {"designs": designs}

    # ---------------- size under the gating nulls
    nul = g[g.world.isin(["SWAP", "SYM"]) & ~g.pool.str.contains(r"\|")]
    nul = nul.assign(flag025=nul.cp_lo_025 > 0.035, flag05=nul.cp_lo_05 > 0.07)
    print("\n=== SIZE (gating nulls SWAP/SYM): max rejection over cells, per test x unit (both designs, both tau, all G)")
    t = nul.groupby(["test", "unit_type"]).agg(cells=("rej_025", "size"), max025=("rej_025", "max"), mean025=("rej_025", "mean"),
                                               max05=("rej_05", "max"), n_flag025=("flag025", "sum"),
                                               n_flag05=("flag05", "sum")).unstack("unit_type")
    print(t.round(4).to_string())
    flagged = nul[nul.flag025 | nul.flag05].sort_values(["test", "unit_type", "pool", "G"])
    print("\n=== flagged null cells (CP lower > 0.035 at 0.025 or > 0.07 at 0.05)")
    print(flagged[["design", "test", "unit_type", "pool", "world", "tau", "G", "reps", "rej_025", "cp_lo_025", "rej_05",
                   "cp_lo_05", "kish_median"]].to_string())
    summ["size_flags"] = flagged[["design", "test", "unit_type", "pool", "world", "tau", "G", "rej_025", "cp_lo_025",
                                  "rej_05", "cp_lo_05"]].to_dict("records")
    # size by G for the main pools
    for pool in ("B1-B0", "HIST_D7-B1", "B1-B1", "S02-B1", "S06-B1"):
        s = nul[(nul.pool == pool) & (nul.design == "BPG")]
        if s.empty:
            continue
        print(f"\n=== SIZE at 0.025, pool {pool} (BPG), rows test, cols unit|tau|G")
        print(s.pivot_table(index="test", columns=["unit_type", "tau", "G"], values="rej_025").reindex(MAIN_TESTS).round(3)
              .to_string())

    # ---------------- power, B1-B0 and scalings
    for pool in ("B1-B0", "B1s50-B0", "B1s2-B0"):
        emp = g[(g.world == "EMP") & (g.pool == pool)]
        for a in ("025", "05"):
            print(f"\n=== POWER {pool} EMP at one-sided {a}: rows test, cols design|unit|tau|G")
            for unit in UNITS:
                s = emp[emp.unit_type == unit]
                print(f"--- {unit}")
                print(s.pivot_table(index="test", columns=["design", "tau", "G"], values=f"rej_{a}").reindex(MAIN_TESTS)
                      .round(3).to_string())
    # G needed for 80 %
    rows = []
    for (d, pool, unit, tau, test), s in g[(g.world == "EMP") & g.pool.isin(["B1-B0", "B1s50-B0", "B1s2-B0", "B1-B0|perseed",
                                                                                "B1-B0|DGA", "B1-B0|nonDGA"])].groupby(
            ["design", "pool", "unit_type", "tau", "test"]):
        for a in ("025", "05"):
            gi, gl = g_needed(s, f"rej_{a}")
            rows.append({"design": d, "pool": pool, "unit_type": unit, "tau": tau, "test": test, "alpha": a,
                         "G80_grid": gi, "G80_interp": gl, "power_G8": float(s[s.G == 8][f"rej_{a}"].iloc[0]),
                         "power_G20": float(s[s.G == 20][f"rej_{a}"].iloc[0]),
                         "power_G40": float(s[s.G == 40][f"rej_{a}"].iloc[0]),
                         "power_G80": float(s[s.G == 80][f"rej_{a}"].iloc[0])})
    need = pd.DataFrame(rows)
    need.to_csv(OUT / "P2_G80.csv", index=False)
    for a in ("025", "05"):
        s = need[(need.pool == "B1-B0") & (need.alpha == a)]
        print(f"\n=== G for 80 % power, B1-B0, alpha {a} (interp; '>80' = not reached)")
        s = s.assign(v=s.G80_interp.map(lambda x: ">80" if pd.isna(x) else f"{x:.0f}"))
        print(s.pivot_table(index="test", columns=["design", "unit_type", "tau"], values="v", aggfunc="first")
              .reindex(MAIN_TESTS).to_string())
    for pool in ("B1s50-B0", "B1s2-B0"):
        s = need[(need.pool == pool) & (need.alpha == "025")]
        s = s.assign(v=s.G80_interp.map(lambda x: ">80" if pd.isna(x) else f"{x:.0f}"))
        print(f"\n=== G for 80 % power, {pool}, alpha 0.025")
        print(s.pivot_table(index="test", columns=["design", "unit_type", "tau"], values="v", aggfunc="first")
              .reindex(MAIN_TESTS).to_string())
    kill = need[(need.pool == "B1-B0") & (need.unit_type == "publication") & (need.alpha == "025")]
    print("\n=== KILL check: power at G = 8, publication units, one-sided 0.025, B1-B0")
    print(kill[["design", "tau", "test", "power_G8"]].pivot_table(index="test", columns=["design", "tau"], values="power_G8")
          .reindex(MAIN_TESTS).round(3).to_string())
    summ["kill_power_G8_pub_025"] = kill[["design", "tau", "test", "power_G8"]].to_dict("records")

    # ---------------- negative controls / slots / mismatch, EMP world
    print("\n=== EMP rejection at 0.025 for the negative controls, slots and HBK@ML-B0 (G = 8 / 20 / 40), BPG")
    emp = g[(g.world == "EMP") & (g.design == "BPG") & g.G.isin([8, 20, 40]) &
            g.pool.isin(["HIST_D7-B1", "HBK@ML-B0", "S01-B1", "S02-B1", "S05-B1", "S06-B1", "S11-B1"])]
    for unit in UNITS:
        print(f"--- {unit}")
        print(emp[emp.unit_type == unit].pivot_table(index="test", columns=["pool", "tau", "G"], values="rej_025")
              .reindex(MAIN_TESTS).round(3).to_string())
    print("\n=== same, BP")
    emp = g[(g.world == "EMP") & (g.design == "BP") & g.G.isin([8, 20, 40]) &
            g.pool.isin(["HIST_D7-B1", "HBK@ML-B0", "S01-B1", "S02-B1", "S05-B1", "S06-B1", "S11-B1"])]
    for unit in UNITS:
        print(f"--- {unit}")
        print(emp[emp.unit_type == unit].pivot_table(index="test", columns=["pool", "tau", "G"], values="rej_025")
              .reindex(MAIN_TESTS).round(3).to_string())

    # ---------------- skewed heterogeneous nulls (not gating)
    print("\n=== C-* worlds (B1-B0 pool, recentred; NOT gating) rejection at 0.025, BPG")
    c = g[g.world.str.startswith("C-") & (g.design == "BPG")]
    for unit in UNITS:
        print(f"--- {unit}")
        print(c[c.unit_type == unit].pivot_table(index="test", columns=["world", "tau", "G"], values="rej_025")
              .reindex(MAIN_TESTS).round(3).to_string())

    # ---------------- extensions
    ext = g[g.pool.str.contains(r"\|")]
    if len(ext):
        print("\n=== extensions (BPG): per-seed pools and DGA strata, rejection at 0.025")
        for pool in sorted(ext.pool.unique()):
            s = ext[ext.pool == pool]
            print(f"--- {pool}")
            print(s.pivot_table(index="test", columns=["unit_type", "world", "tau", "G"], values="rej_025")
                  .reindex([x for x in MAIN_TESTS if x != "SEALED"]).round(3).to_string())
    # medians of the draws (Kish etc.) for B1-B0 EMP
    print("\n=== draw medians, B1-B0 EMP (test SIGN rows), BPG")
    s = g[(g.pool == "B1-B0") & (g.world == "EMP") & (g.test == "SIGN") & (g.design == "BPG")]
    print(s[["unit_type", "tau", "G", "kish_median", "pooled_rel_median", "meanL_median", "win_share_mean"]].round(3).to_string())
    (OUT / "P2_summary.json").write_text(json.dumps(summ, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

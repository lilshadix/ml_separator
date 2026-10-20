#!/usr/bin/env python
"""One table of every round-3 configuration (from the stored result files only): primary column, data
version, policy, cohort / budget, cells / pairs per seed, R2 BP / BPG, MAE / RMSE BP, PWL_R on the same
rows, paired delta vs PWL_R and vs HBK@ML (BP, 95 % chemotype bootstrap) with seed counts, intervention.

    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_table.py
writes results/ROUND3_TABLE.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3 import paths  # noqa: E402

# primary column, comparator subset, budget k, cohort K (from the registry; R24 / R20 / B0 are budget ladders)
ROWS = [
    ("R01", "PCHIP_R"), ("R02", "COR_G"), ("R03", "COR_GP"), ("R04", None), ("R05", None), ("R06", None),
    ("R07", "COR_GC"), ("R08", "COR_GDC"), ("R12", "COR[GD]"), ("R13", "STACK"), ("R14", "COR[Dnc]"),
    ("R15", None), ("R16", None), ("R17", "PWL_R"), ("R18", "COR[Dnc]"), ("R19", None), ("R21", None),
    ("R20", "COR[GD]@k"), ("R22", None), ("R23", "COR[Dnc]"), ("R24", "COR[Dnc]@k"), ("CTRL23", "COR[Dnc]_SHUFFLEDALL"),
    ("CTRL14W", "COR[Dnc]_SHUFFLED_W"), ("CTRL23W", "COR[Dnc]_SHUFFLED_W"), ("CTRL02W", "COR_G_SHUFFLED_W"),
]
POLICY = {"R09": "POL_ERR", "R10": "POL_ERR", "R11": "POL_ERR", "R12": "POL_ERR", "R15": "POL_ERR", "R16": "POL_ERR",
          "R18": "POL_ERR (V2)", "R23": "POL_ERR", "R24": "POL_ERR(k)", "CTRL23": "POL_ERR", "CTRL23W": "POL_ERR"}
CROSS = {"R09": "PWL_R", "R10": "PWL_Z", "R11": "HBK@ML"}     # results/policy_cross: predictor @POL_ERR vs @DOPT_Q


def _cross_rows() -> list:
    out = []
    for cid, pred in CROSS.items():
        rec = {"id": cid, "column": f"{pred}@POL_ERR"}
        for design in ("BP", "BPG"):
            s = pd.read_csv(paths.RES / "policy_cross" / f"cross_{design}_R09.csv")
            s = s[(s.subset == "all") & (s.k == 3)].set_index("arm")
            a = f"{pred}@POL_ERR"
            rec.update({"k": 3, "cells": s.loc[a, "n_cells_per_seed"], "pairs": s.loc[a, "n_rows_per_seed"],
                        f"R2_{design}": s.loc[a, "R2"], f"R2_min_seed_{design}": s.loc[a, "R2_min_seed"],
                        f"PWL_R_{design}": s.loc["PWL_R@POL_ERR", "R2"], f"HBK_{design}": s.loc["HBK@ML@POL_ERR", "R2"]})
            if design == "BP":
                rec.update({"MAE_BP": s.loc[a, "MAE"], "RMSE_BP": s.loc[a, "RMSE"], "PWL_R_MAE_BP": s.loc["PWL_R@POL_ERR", "MAE"]})
            ct = pd.read_csv(paths.RES / "policy_cross" / f"contrasts_{design}_R09.csv")
            r = ct[(ct.name == f"{a} - {pred}@DOPT_Q") & (ct.statistic == "R2")]
            if len(r):
                r = r.iloc[0]
                m = ct[(ct.name == f"{a} - {pred}@DOPT_Q") & (ct.statistic == "MAE")].iloc[0]
                rec[f"dR2_same_predictor_DOPT_Q_{design}"] = r.point
                rec[f"dR2_same_predictor_DOPT_Q_{design}_ci95"] = f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]"
                rec[f"seeds_R2_MAE_same_predictor_DOPT_Q_{design}"] = f"{int(r.seeds_R2_better)}/5, {int(m.seeds_MAE_better)}/5"
        out.append(rec)
    return out


def _primary(cid: str):
    s = pd.read_csv(paths.RES / cid / "summary_BP.csv")
    arms = [a for a in s.arm.unique() if a not in ("PWL_R", "PWL_Z", "LINE", "RIDGE2", "HBK@ML", "G14@ZERO", "FLAT")]
    return arms


def main() -> int:
    out = []
    for cid, col in ROWS:
        d = paths.RES / cid
        if not (d / "summary_BP.csv").exists():
            continue
        cols = [col] if col else _primary(cid)
        for c in cols:
            rec = {"id": cid, "column": c}
            for design in ("BP", "BPG"):
                s = pd.read_csv(d / f"summary_{design}.csv")
                s = s[s.subset == "all"]
                kk = 3 if 3 in set(s.k) else int(s.k.max())
                s = s[s.k == kk].set_index("arm")
                if c not in s.index:
                    continue
                rec.update({"k": kk, "cells": s.loc[c, "n_cells_per_seed"], "pairs": s.loc[c, "n_rows_per_seed"],
                            f"R2_{design}": s.loc[c, "R2"], f"R2_min_seed_{design}": s.loc[c, "R2_min_seed"],
                            f"PWL_R_{design}": s.loc["PWL_R", "R2"], f"HBK_{design}": s.loc["HBK@ML", "R2"]})
                if design == "BP":
                    rec.update({"MAE_BP": s.loc[c, "MAE"], "RMSE_BP": s.loc[c, "RMSE"], "PWL_R_MAE_BP": s.loc["PWL_R", "MAE"]})
                cf = d / f"contrasts_{design}.csv"
                if cf.exists():
                    ct = pd.read_csv(cf)
                    for ref in ("PWL_R", "HBK@ML"):
                        r = ct[(ct.name == f"{c} - {ref}") & (ct.statistic == "R2")]
                        if len(r):
                            r = r.iloc[0]
                            rec[f"dR2_{ref}_{design}"] = r.point
                            rec[f"dR2_{ref}_{design}_ci95"] = f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]"
                            m = ct[(ct.name == f"{c} - {ref}") & (ct.statistic == "MAE")].iloc[0]
                            rec[f"seeds_R2_MAE_{ref}_{design}"] = f"{int(r.seeds_R2_better)}/5, {int(m.seeds_MAE_better)}/5"
                iv = d / f"intervention_{design}.json"
                if iv.exists():
                    j = json.loads(iv.read_text())
                    rec[f"intervention_{design}"] = f"{'PASS' if j.get('passed') else 'FAIL'} ({j.get('n_predictions')})"
            out.append(rec)
    out += _cross_rows()
    t = pd.DataFrame(out)
    t.insert(2, "policy", [POLICY.get(i, "DOPT_Q") for i in t.id])
    t.insert(3, "cohort", ["C6" if i in ("R20", "R24") else ("C3 V2" if i in ("R17", "R18") else "C3") for i in t.id])
    t.to_csv(paths.RES / "ROUND3_TABLE.csv", index=False)
    pd.set_option("display.width", 250)
    print(t[["id", "column", "policy", "cells", "pairs", "R2_BP", "R2_BPG", "MAE_BP", "PWL_R_BP", "dR2_PWL_R_BP", "dR2_PWL_R_BP_ci95",
             "seeds_R2_MAE_PWL_R_BP"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

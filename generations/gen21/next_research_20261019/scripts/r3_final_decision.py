#!/usr/bin/env python
"""Apply the frozen decision rule (manifests/freeze_manifest.json) to the single final run in results/final/.
Reads only the final outputs and the freeze manifest; computes nothing new except cross-configuration contrasts on
identical final rows (descriptive).  Writes results/final/DECISION.json and prints the tables.

    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_final_decision.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3 import metrics as MT, paths  # noqa: E402

FIN = paths.RES / "final"
KEY = ["split_seed", "fold", "cell_id", "A", "B", "k"]


def _summary(cid, design):
    s = pd.read_csv(FIN / f"summary_{cid}_{design}.csv")
    return s[s.k == 3]


def _row(s, subset, arm):
    r = s[(s.subset == subset) & (s.arm == arm)]
    return r.iloc[0] if len(r) else None


def _contrast(cid, design, name, stat):
    c = pd.read_csv(FIN / f"contrasts_{cid}_{design}.csv")
    r = c[(c.name == name) & (c.statistic == stat)].iloc[0]
    return {"point": float(r.point), "ci95": [float(r.ci95_low), float(r.ci95_high)],
            "bca95": [float(r.bca95_low), float(r.bca95_high)], "seeds_R2_better": int(r.seeds_R2_better),
            "seeds_MAE_better": int(r.seeds_MAE_better)}


def main() -> int:
    freeze = json.loads((paths.MAN / "freeze_manifest.json").read_text(encoding="utf-8"))
    cand, col = freeze["config_id"], freeze["candidate_column"]
    cfgs = [(cand, col)] + [(c, d["columns"][0]) for c, d in freeze["secondary_detail"].items()]
    out = {"freeze_seal": (paths.MAN / "freeze_seal.txt").read_text().split()[0],
           "run_once": json.loads((FIN / "RUN_ONCE.json").read_text(encoding="utf-8")), "configs": {}}
    for cid, c in cfgs:
        res = {}
        for design in ("BP", "BPG"):
            s = _summary(cid, design)
            iv = json.loads((FIN / f"intervention_{cid}_{design}.json").read_text(encoding="utf-8"))
            d = {"R2": float(_row(s, "all", c).R2), "MAE": float(_row(s, "all", c).MAE), "RMSE": float(_row(s, "all", c).RMSE),
                 "R2_per_seed": _row(s, "all", c).R2_per_seed, "R2_min_seed": float(_row(s, "all", c).R2_min_seed),
                 "cells_per_seed": float(_row(s, "all", c).n_cells_per_seed), "pairs_per_seed": float(_row(s, "all", c).n_rows_per_seed),
                 "intervention": iv}
            for arm in ("PWL_R", "PWL_Z", "HBK@ML", "RIDGE2"):
                d[f"{arm}_R2"], d[f"{arm}_MAE"] = float(_row(s, "all", arm).R2), float(_row(s, "all", arm).MAE)
            for sub in ("DGA_sc009", "non_DGA", "one_end_read", "both_hidden", "Pr/Nd"):
                r, rp = _row(s, sub, c), _row(s, sub, "PWL_R")
                if r is not None:
                    d[f"{sub}_R2"], d[f"{sub}_PWL_R_R2"] = float(r.R2), float(rp.R2)
            lv = pd.read_csv(FIN / f"levels_{cid}_{design}.csv")
            lr = lv[lv.arm.str.split("|").str[0] == c].iloc[0]
            d["R2_level_ci95"] = [float(lr.R2_ci_low), float(lr.R2_ci_high)]
            d["contrasts"] = {f"vs {r}": {st: _contrast(cid, design, f"{c} - {r}", st) for st in ("R2", "MAE")}
                              for r in ("PWL_R", "PWL_Z", "HBK@ML")}
            res[design] = d
        prim = all(res[x]["R2"] >= 0.50 for x in ("BP", "BPG"))
        mae = all(res[x]["MAE"] < res[x]["PWL_R_MAE"] and res[x]["MAE"] < res[x]["PWL_Z_MAE"] for x in ("BP", "BPG"))
        ivok = all(res[x]["intervention"]["passed"] for x in ("BP", "BPG"))
        gain = all(res[x]["contrasts"]["vs PWL_R"]["R2"]["point"] > 0 and res[x]["contrasts"]["vs PWL_R"]["MAE"]["point"] > 0
                   and res[x]["contrasts"]["vs PWL_R"]["R2"]["ci95"][0] > 0 and res[x]["contrasts"]["vs PWL_R"]["MAE"]["ci95"][0] > 0
                   for x in ("BP", "BPG"))
        res["decision"] = {"role": "candidate" if cid == cand else "secondary (descriptive)",
                           "R2_ge_0.50_both_designs": bool(prim), "MAE_below_PWL_R_and_PWL_Z_both": bool(mae),
                           "interventions_pass": bool(ivok), "primary_target_reached": bool(prim and mae and ivok),
                           "gain_over_PWL_R_confirmed": bool(gain)}
        out["configs"][cid] = res
    # descriptive cross-configuration contrasts on identical final rows
    cross = {}
    for design in ("BP", "BPG"):
        t02 = pd.read_parquet(FIN / f"final_{cand}_{design}.parquet")
        merged = t02.rename(columns={"PWL_R": "PWL_R@DOPT_Q"})
        for cid, c in cfgs[1:]:
            t = pd.read_parquet(FIN / f"final_{cid}_{design}.parquet")
            add = t[KEY + ["y", c]].rename(columns={c: f"{c}@{cid}", "y": f"y_{cid}"})
            merged = merged.merge(add, on=KEY, validate="one_to_one")
            assert len(merged) == len(t02) and np.allclose(merged.y, merged[f"y_{cid}"])
        pairs = [(f"{c}@{cid}", "PWL_R@DOPT_Q") for cid, c in cfgs[1:]]
        if len(cfgs) == 3:
            pairs.append((f"{cfgs[2][1]}@{cfgs[2][0]}", f"{cfgs[1][1]}@{cfgs[1][0]}"))
        ct = MT.contrasts(merged, pairs, K=3, k=3)
        ct.insert(0, "design", design)
        ct.to_csv(FIN / f"contrasts_cross_{design}.csv", index=False)
        cross[design] = {r.name: {"point": float(r.point), "ci95": [float(r.ci95_low), float(r.ci95_high)],
                                  "seeds_R2_better": int(r.seeds_R2_better)} for r in ct[ct.statistic == "R2"].itertuples()}
    out["cross_contrasts_R2"] = cross
    (FIN / "DECISION.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    for cid, res in out["configs"].items():
        print("==", cid, res["decision"])
        for design in ("BP", "BPG"):
            d = res[design]
            print(f"  {design}: R2 {d['R2']:.4f} (min seed {d['R2_min_seed']:.4f}; level CI {d['R2_level_ci95'][0]:.3f}-{d['R2_level_ci95'][1]:.3f}) "
                  f"MAE {d['MAE']:.4f} | PWL_R {d['PWL_R_R2']:.4f} PWL_Z {d['PWL_Z_R2']:.4f} HBK {d['HBK@ML_R2']:.4f} | "
                  f"dR2 vs PWL_R {d['contrasts']['vs PWL_R']['R2']['point']:+.4f} {np.round(d['contrasts']['vs PWL_R']['R2']['ci95'], 4)} "
                  f"seeds {d['contrasts']['vs PWL_R']['R2']['seeds_R2_better']}/{d['contrasts']['vs PWL_R']['MAE']['seeds_MAE_better']} | "
                  f"iv {d['intervention']['passed']} ({d['intervention']['n_predictions']})")
    print("cross", json.dumps(cross, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

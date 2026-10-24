#!/usr/bin/env python
"""P5SK (skeptic of P5): is P5's training-side result ("B1 refit without the 69 true no-location cells loses to B0 on
the 160 located cells, -0.0045 R2 BPG") specific to those 69 cells, or what any removal of ~69 C3 cells does?

EXPLORATORY: 229 C3 cells / 5 dev seeds already seen.  Uses r4.refs.fit_reference unchanged, exactly as
P5_refit.py does (outer training index minus the excluded cells; inner covariances, read sets kept).

Modes (argv):  DESIGN MODE [N] [SEED] [LAM]
  DESIGN  BPG | BP
  MODE    t7       : identity (T0) + the true no-location 69 (T7), per-seed numbers printed
          uniform  : N draws of 69 C3 cells uniformly at random (numpy default_rng(SEED))
          cluster  : N draws of whole C3 publications in random order until >= 69 cells
          ornl     : T3 ORNL-54 only (per seed)
  LAM     'cv' (default, the round-4 grid incl. infinity) or a number: fixed ridge strength (grid of one value)

  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore \
      generations/gen21/round5_design_20261023/probes/scripts/P5SK_refit_null.py BPG t7
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
R4 = ROOT / "generations" / "gen21" / "round4_20261022"
sys.path.insert(0, str(R4))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import refs as RF  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402

C5 = "pub_6a0ef8528c"


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    design, mode = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    lam = sys.argv[5] if len(sys.argv) > 5 else "cv"
    grid = RF.LAMBDA_GRID if lam == "cv" else (float(lam),)
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    ci_of = dict(zip(f.cell_id, f.index))
    cells = pd.read_csv(R4 / "results" / "provenance" / "cells.csv")
    c3 = cells[cells.c3].reset_index(drop=True)
    noloc = c3.data_location.isna()
    true69 = set(c3.cell_id[noloc & (c3.ornl_import_all | (c3.publication_id == C5))])
    ornl54 = set(c3.cell_id[noloc & c3.ornl_import_all])
    assert len(true69) == 69 and len(ornl54) == 54
    scen = {"T0": set()}
    if mode == "t7":
        scen["T7"] = true69
    elif mode == "ornl":
        scen["T3"] = ornl54
    elif mode == "uniform":
        rng = np.random.default_rng(seed)
        ids = sorted(c3.cell_id)
        for i in range(n):
            scen[f"U{i:02d}"] = set(ids[j] for j in rng.choice(len(ids), 69, replace=False))
    elif mode == "cluster":
        rng = np.random.default_rng(seed)
        pubs = sorted(c3.publication_id.unique())
        size = c3.groupby("publication_id").size()
        for i in range(n):
            chosen, tot = [], 0
            for j in rng.permutation(len(pubs)):
                if tot >= 69:
                    break
                chosen.append(pubs[j])
                tot += int(size[pubs[j]])
            scen[f"P{i:02d}"] = set(c3.cell_id[c3.publication_id.isin(chosen)])
    else:
        raise SystemExit("unknown mode")
    print(f"EXPLORATORY P5SK refit null: design {design} mode {mode} n {n} seed {seed} lam {lam}; sizes",
          {k: len(v) for k, v in scen.items()}, " overlap with true69:", {k: len(v & true69) for k, v in scen.items()},
          flush=True)

    t = pd.read_parquet(R4 / "predictions" / f"REF_{design}.parquet")
    t = t[(t.k == 3) & (t.k_max >= 3)].reset_index(drop=True)
    dphi = ((t.HIST_D7 - t.B0) / RF.HIST_D7_BETA).to_numpy()
    fitted = json.loads((R4 / "results" / "REF" / f"fitted_{design}.json").read_text())
    pred = {k: t.B0.to_numpy().copy() for k in scen}
    betas = {k: [] for k in scen}
    lams = {k: [] for k in scen}
    t0 = time.time()
    for fo in folds_for(bench.frame, design, seeds=list(paths.DEV_SEEDS)):
        rec = C.load(fo.design, fo.seed, fo.fold)
        stored = fitted[f"{fo.seed}_{fo.fold}"]["ref:B1"]["coef"][0]
        rows = ((t.split_seed == fo.seed) & (t.fold == fo.fold)).to_numpy()
        assert np.allclose(t.B1[rows], t.B0[rows] + stored * dphi[rows], atol=1e-9)
        tr = np.asarray(rec["outer"]["train_index"])
        for k, excl in scen.items():
            if k == "T0" and lam != "cv":
                # build the orphan memo from the full index, as P5 does, but with the full grid for the identity check
                fit0 = RF.fit_reference(bench, rec, bench.Y, "B1")
                assert abs(float(fit0["coef"][0]) - stored) < 1e-9
            drop = np.array([ci_of[c] for c in excl], dtype=int)
            rec2 = dict(rec)
            rec2["outer"] = dict(rec["outer"], train_index=tr[~np.isin(tr, drop)])
            fit = RF.fit_reference(bench, rec2, bench.Y, "B1", lam_grid=grid)
            b = float(fit["coef"][0])
            if k == "T0" and lam == "cv":
                assert abs(b - stored) < 1e-9, (fo.seed, fo.fold, b, stored)
            betas[k].append(b)
            lams[k].append(fit["lambda"])
            pred[k][rows] = t.B0[rows].to_numpy() + b * dphi[rows]
        print(f"  {fo.seed}/{fo.fold} train {len(tr)}: " + " ".join(f"{k}={betas[k][-1]:+.4f}" for k in list(scen)[:6])
              + f" ({time.time() - t0:.0f}s)", flush=True)

    res = []
    for k, excl in scen.items():
        out = ~t.cell_id.isin(excl).to_numpy()
        per = []
        for s, g in t[out].groupby("split_seed"):
            ix = g.index.to_numpy()
            per.append((s, r2(g.y, pred[k][ix]), r2(g.y, g.B1), r2(g.y, g.B0)))
        per = np.array(per)
        adv = per[:, 1] - per[:, 3]
        res.append({"scen": k, "n_excl": len(excl), "ov69": len(excl & true69),
                    "rows_out/seed": int(out.sum() / 5), "beta_mean": np.mean(betas[k]), "beta_min": np.min(betas[k]),
                    "beta_max": np.max(betas[k]),
                    "inf_folds": sum(1 for x in lams[k] if isinstance(x, float) and math.isinf(x)),
                    "R2_B1ref_out": per[:, 1].mean(), "R2_B1v1_out": per[:, 2].mean(), "R2_B0_out": per[:, 3].mean(),
                    "train_loss": (per[:, 1] - per[:, 2]).mean(), "adv_vs_B0": adv.mean(), "adv_min": adv.min(),
                    "adv_max": adv.max(), "adv_per_seed": ";".join(f"{v:+.4f}" for v in adv),
                    "v1adv_out": (per[:, 2] - per[:, 3]).mean()})
    d = pd.DataFrame(res)
    pd.set_option("display.width", 300)
    print(d.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(d) > 2:
        dd = d[d.scen != "T0"]
        print("\nnull summary (excluding T0): n", len(dd))
        for col in ("train_loss", "adv_vs_B0", "v1adv_out", "beta_mean"):
            print(f"  {col}: mean {dd[col].mean():+.4f} sd {dd[col].std():.4f} min {dd[col].min():+.4f} "
                  f"max {dd[col].max():+.4f}  frac<=0 {np.mean(dd[col] <= 0):.2f}")
    print("EXIT 0")


if __name__ == "__main__":
    main()

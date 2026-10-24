#!/usr/bin/env python
"""DRAFT (moved to scripts/ with amendment A2).  Discriminating power of block B's geometry controls on synthetic
tasks (brief section 8 item 8).

Real structure of two BPG folds (seed 130363, folds 2 and 3) with SYNTHETIC log D (no real label read):
    y_c(m) = a_c + b_c x + q_c x^2 - 0.09 e_Gd + nuisance_chemotype(c) + signal + noise (sd 0.11)
  NULL   signal = 0
  GEO    signal = 0.08 * z_c(m), z = the cell's own complex's mean Ln-donor distance at metal m minus the
         cohort per-metal mean, standardised over all cell-metals (label-free geometry; 0 where missing)
  MISSING  signal = 0.10 where the cell has NO geometry at metal m, else 0 (run 2): a gain S06 can reach through its
         coverage flag; S09 and S22 keep S06's missingness, so the geometry claim must NOT fire
Each replicate fits B1, S05, S06, S09 (5 draws) and S22 on the fold's training cells and scores its C3 test pairs.
Geometry claim (upper bound of the sealed rule; vs B1 only, one fold): crit1(S06) and S06 - S05 >= 0.005 and
S06 > every S09 draw and mean(S09 - S05) < 0.5 (S06 - S05) and S06 - S22 >= 0.005.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import refs as RF  # noqa: E402
from r4 import registry as RG  # noqa: E402
from r4 import slots_a as SA  # noqa: E402
from r4 import slots_b as SB  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import metrics as MT  # noqa: E402
from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN  # noqa: E402

FOLDS = (("BPG", 130363, 2), ("BPG", 130363, 3))
WORLDS = ("NULL", "GEO", "MISSING")
NOISE_SD, NUISANCE_SD, GEO_AMP = 0.11, 0.03, 0.08
NUISANCE_SHAPES = ("e_Gd", "e_Nd", "gd_break", "tetrad_e1", "tetrad_e3", "P2", "P3", "P4")


def geo_z(bench) -> dict:
    tab = SB.geometry_table()
    vals = {}
    for (cid, m), v in tab.items():
        if np.isfinite(v[0]):
            vals.setdefault(m, []).append(v[0])
    mu = {m: np.mean(v) for m, v in vals.items()}
    allz = [v[0] - mu[m] for (cid, m), v in tab.items() if np.isfinite(v[0])]
    sd = np.std(allz)
    return {(cid, m): (v[0] - mu[m]) / sd for (cid, m), v in tab.items() if np.isfinite(v[0])}


def synthetic_Y(bench, world, seed, z):
    rng = np.random.default_rng(seed)
    geo_keys = set(SB.geometry_table())
    x = interp.X_R
    xs = (x - x.mean()) / x.std()
    sh = SA.shapes()
    chems = sorted(bench.frame.chemotype.astype(str).unique())
    nuis = {c: sum(rng.normal(0, NUISANCE_SD) * sh[s] for s in NUISANCE_SHAPES) for c in chems}
    Ys = np.full_like(bench.Y, np.nan)
    for ci in range(bench.Y.shape[0]):
        a, b, q = rng.normal(0, [1.0, 1.0, 0.3])
        cid = bench.frame.cell_id.iat[ci]
        if world == "GEO":
            sig = np.array([z.get((cid, m), 0.0) for m in range(N_LN)]) * GEO_AMP
        elif world == "MISSING":
            sig = np.array([0.0 if (cid, m) in geo_keys else 0.10 for m in range(N_LN)])
        else:
            sig = 0.0
        curve = (a + b * xs + q * xs ** 2 - 0.09 * sh["e_Gd"] + nuis[str(bench.frame.chemotype.iat[ci])] + sig
                 + rng.normal(0, NOISE_SD, N_LN))
        Ys[ci] = np.where(np.isfinite(bench.Y[ci]), curve, np.nan)
    return Ys


def score(t, cand, ref="B1"):
    y = t.y.to_numpy()
    g = t.assign(ec=(t[cand] - t.y).abs(), er=(t[ref] - t.y).abs())
    ch = g.groupby(["chemotype", "extractant"])[["ec", "er"]].mean().groupby(level=0).mean()
    return float(MT._r2(y, t[cand]) - MT._r2(y, t[ref])), float((ch.er - ch.ec).mean())


def one(bench, rec, world, seed, z, fk) -> dict:
    Ys = synthetic_Y(bench, world, seed, z)
    pb = copy.copy(bench)
    pb.Y = Ys
    b1 = RF.fit_reference(pb, rec, Ys, "B1")
    parts = ["b:S05", "b:S06", "b:S22"] + [f"b:S09:{i}" for i in range(SB.N_DRAWS)]
    fits = {p: SB.fit_slot(pb, rec, Ys, RG._b_mode(p, bench), b1) for p in parts}
    rows = []
    for cell in rec["outer"]["cells"]:
        if cell["k_max"] < 3:
            continue
        obs = E.observe(Ys[cell["ci"]], cell, cell["seqs"]["DOPT_Q"], 3)
        lv = {"B1": RF.predict_levels(b1, obs, cell["metals"])}
        lv.update({p: SB.predict_levels(f, obs, cell) for p, f in fits.items()})
        for a, b in cell["query"]:
            r = {"chemotype": cell["chemotype"], "extractant": cell["extractant"], "y": Ys[cell["ci"]][a] - Ys[cell["ci"]][b]}
            r.update({k: v[a] - v[b] for k, v in lv.items()})
            rows.append(r)
    t = pd.DataFrame(rows)
    res = {p: score(t, p) for p in parts}
    g5, g6, g22 = res["b:S05"][0], res["b:S06"][0], res["b:S22"][0]
    s09 = [res[f"b:S09:{i}"][0] for i in range(SB.N_DRAWS)]
    crit1 = res["b:S06"][0] >= 0.01 and res["b:S06"][1] >= 0.005
    rule = (g6 - g5 >= 0.005 and g6 > max(s09) and np.mean([v - g5 for v in s09]) < 0.5 * (g6 - g5) and g6 - g22 >= 0.005)
    return {"seed": seed, "fold": list(fk), "gain": {p: v[0] for p, v in res.items()}, "gain_mae": {p: v[1] for p, v in res.items()},
            "n_trees": {p: f["n_trees"] for p, f in fits.items()}, "crit1_S06": bool(crit1), "geometry_rule": bool(rule),
            "geometry_claim": bool(crit1 and rule), "crit1_S05": bool(res["b:S05"][0] >= 0.01 and res["b:S05"][1] >= 0.005)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, choices=WORLDS)
    ap.add_argument("--reps", type=int, default=6)
    a = ap.parse_args()
    bench = D.load_bench()
    z = geo_z(bench)
    recs = {f: C.load(*f) for f in FOLDS}
    reps = []
    for i in range(a.reps):
        fk = FOLDS[i % len(FOLDS)]
        reps.append(one(bench, recs[fk], a.world, 11000 + 100 * WORLDS.index(a.world) + i, z, fk))
        print(a.world, i, {k: reps[-1][k] for k in ("crit1_S05", "crit1_S06", "geometry_rule", "geometry_claim")},
              {k: round(v, 4) for k, v in reps[-1]["gain"].items() if "S09" not in k}, flush=True)
    out = paths.RES / "synthetic_b"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.world}.json").write_text(json.dumps({"world": a.world, "folds": FOLDS, "replicates": reps}, indent=1),
                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

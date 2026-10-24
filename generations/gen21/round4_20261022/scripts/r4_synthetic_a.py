#!/usr/bin/env python
"""Discriminating power of block A's slots and controls on synthetic tasks (brief section 8 item 8; PROTOCOL A1).

Run 2 (after the block-A pre-seal review; run 1 is kept in results/synthetic_a/run1/).  The real structure (cells,
metals present, read sets, masks, inner blocks, weights, donor classes) of two BPG folds (seed 130363, folds 2 and
3; replicates alternate) with SYNTHETIC log D for every cell (no real label is read):
    y_c(m) = a_c + b_c x(m) + q_c x(m)^2 - 0.09 e_Gd(m) + nuisance_chemotype(c)(m) + signal_class(c)(m) + noise
x = standardised -radius CN8; a, b, q per cell ~ N(0, 1), N(0, 1), N(0, 0.3); noise sd 0.11 per metal (B1's synthetic
pair RMSE is reported per replicate; the real one is about 0.16); nuisance = class-agnostic per-CHEMOTYPE amplitudes
~ N(0, 0.03) on every slot shape (e_Gd, e_Nd, gd_break, tetrad_e1, tetrad_e3, P2-P4): structure a model can overfit,
unrelated to donor class.  Worlds (signal on top of the shared -0.09 Gd dip):
  NULL       no class signal (structured null)
  CHEM_D7    gen13 D7's direction: DGA Gd -0.025 / Nd -0.109, other classes Gd -0.127 / Nd -0.007
  GDCLASS    class-specific Gd dip only: DGA +0.12, SOFT_N -0.12, OTHER 0 (must NOT give an electronic claim)
  ELEC       class-specific tetrad_e3: DGA +0.08, SOFT_N -0.08, OTHER 0 (should give the electronic claim)
  PROJ       class-specific NON-electronic shape that S02 can partly fit (gd_break + tetrad_e3 projected onto P2-P4,
             rms 1): DGA +0.15, SOFT_N -0.15 (must NOT give an electronic claim)
  UNIVERSAL  class-free 0.08 tetrad_e3 for every class (must NOT give a donor claim)
Each replicate fits B1, S01, S02, S03, S04 (5 draws) and S21 on routes S01 and S02 (5 draws each) on the fold's
training cells and scores the fold's C3 test cells (DOPT_Q, k = 3).  Rules per replicate (an UPPER BOUND of the sealed
rules: vs B1 only, no harm rule, no B2, one fold, no seed mean):
  crit1(X)        ΔR² vs B1 >= 0.01 and Δbalanced MAE >= 0.005
  elec_specific   S02 - S03 >= 0.005 R² and S02 > every S04 draw            (claim = crit1(S02) and elec_specific)
  chem_specific   route > every S21 draw and mean S21 < 50 % of the route's   (claim = crit1(route) and chem_specific)
Output: counts k/n with exact Clopper-Pearson 95 % intervals (report/r4_synthetic_summary.py).

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_synthetic_a.py --world NULL --reps 4
writes results/synthetic_a/<world>.json
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
from r3 import interp  # noqa: E402
from r3 import metrics as MT  # noqa: E402
from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN  # noqa: E402

FOLDS = (("BPG", 130363, 2), ("BPG", 130363, 3))
WORLDS = ("NULL", "CHEM_D7", "GDCLASS", "ELEC", "PROJ", "UNIVERSAL")
NOISE_SD, NUISANCE_SD = 0.11, 0.03
NUISANCE_SHAPES = ("e_Gd", "e_Nd", "gd_break", "tetrad_e1", "tetrad_e3", "P2", "P3", "P4")


def _proj_shape() -> np.ndarray:
    """A non-electronic shape S02 can partly fit: gd_break + tetrad_e3 projected onto P2-P4, rms 1."""
    sh = SA.shapes()
    v = sh["gd_break"] + sh["tetrad_e3"]
    P = np.vstack([sh[f"P{d}"] for d in (2, 3, 4)])
    out = (P @ v / N_LN) @ P
    return out / np.sqrt((out ** 2).mean())


def signal(world: str, cls: str) -> np.ndarray:
    sh = SA.shapes()
    s = np.zeros(N_LN)
    if world == "CHEM_D7":
        gd, nd = {"DGA": (-0.025, -0.109)}.get(cls, (-0.127, -0.007))
        s = gd * sh["e_Gd"] + nd * sh["e_Nd"]
    elif world == "GDCLASS":
        s = {"DGA": 0.12, "SOFT_N": -0.12}.get(cls, 0.0) * sh["e_Gd"]
    elif world == "ELEC":
        s = {"DGA": 0.08, "SOFT_N": -0.08}.get(cls, 0.0) * sh["tetrad_e3"]
    elif world == "PROJ":
        s = {"DGA": 0.15, "SOFT_N": -0.15}.get(cls, 0.0) * _proj_shape()
    elif world == "UNIVERSAL":
        s = 0.08 * sh["tetrad_e3"]
    return s


def synthetic_Y(bench, world, seed):
    rng = np.random.default_rng(seed)
    x = interp.X_R
    xs = (x - x.mean()) / x.std()
    sh = SA.shapes()
    chems = sorted(bench.frame.chemotype.astype(str).unique())
    nuis = {c: sum(rng.normal(0, NUISANCE_SD) * sh[s] for s in NUISANCE_SHAPES) for c in chems}
    Ys = np.full_like(bench.Y, np.nan)
    for ci in range(bench.Y.shape[0]):
        a, b, q = rng.normal(0, [1.0, 1.0, 0.3])
        cls = SA.donor_class(str(bench.frame.extractant.iat[ci]))
        curve = (a + b * xs + q * xs ** 2 - 0.09 * sh["e_Gd"] + nuis[str(bench.frame.chemotype.iat[ci])]
                 + signal(world, cls) + rng.normal(0, NOISE_SD, N_LN))
        Ys[ci] = np.where(np.isfinite(bench.Y[ci]), curve, np.nan)
    return Ys


def score(rows: pd.DataFrame, cand: str, ref: str = "B1") -> tuple[float, float]:
    y = rows.y.to_numpy()
    r2 = MT._r2(y, rows[cand]) - MT._r2(y, rows[ref])
    g = rows.assign(ec=(rows[cand] - rows.y).abs(), er=(rows[ref] - rows.y).abs())
    ch = g.groupby(["chemotype", "extractant"])[["ec", "er"]].mean().groupby(level=0).mean()
    return float(r2), float((ch.er - ch.ec).mean())


def one(bench, rec, world, seed, rec_key) -> dict:
    Ys = synthetic_Y(bench, world, seed)
    pb = copy.copy(bench)
    pb.Y = Ys
    b1 = RF.fit_reference(pb, rec, Ys, "B1")
    fits = {}
    parts = ["slot:S01", "slot:S02", "slot:S03"] + [f"slot:S04:{i}" for i in range(SA.N_DRAWS)] + \
            [f"slot:S21:{r}:{i}" for r in ("S01", "S02") for i in range(SA.N_DRAWS)]
    for p in parts:
        spec, cmap = RG._slot_spec(p, bench)
        f = SA.fit_slot(pb, rec, Ys, spec, b1, lambda cell, m=cmap: RG._cls({"class_map": m}, cell))
        f["class_map"] = cmap
        fits[p] = f
    rows = []
    for cell in rec["outer"]["cells"]:
        if cell["k_max"] < 3:
            continue
        obs = E.observe(Ys[cell["ci"]], cell, cell["seqs"]["DOPT_Q"], 3)
        lv = {"B1": RF.predict_levels(b1, obs, cell["metals"])}
        for p, f in fits.items():
            lv[p] = SA.predict_levels(f, obs, cell["metals"], RG._cls(f, cell))
        for a, b in cell["query"]:
            r = {"chemotype": cell["chemotype"], "extractant": cell["extractant"], "y": Ys[cell["ci"]][a] - Ys[cell["ci"]][b]}
            r.update({k: v[a] - v[b] for k, v in lv.items()})
            rows.append(r)
    t = pd.DataFrame(rows)
    res = {p: score(t, p) for p in parts}
    out = {"seed": seed, "fold": list(rec_key), "lambda": {p: str(f["lambda"]) for p, f in fits.items()},
           "gain": {p: v[0] for p, v in res.items()}, "gain_mae": {p: v[1] for p, v in res.items()},
           "b1_rmse": float(np.sqrt(((t.B1 - t.y) ** 2).mean())), "b1_r2": float(MT._r2(t.y.to_numpy(), t.B1)),
           "n_rows": len(t)}
    crit1 = {s: res[f"slot:{s}"][0] >= 0.01 and res[f"slot:{s}"][1] >= 0.005 for s in ("S01", "S02", "S03")}
    s04 = [res[f"slot:S04:{i}"][0] for i in range(SA.N_DRAWS)]
    out["crit1"] = crit1
    out["elec_specific"] = bool(res["slot:S02"][0] - res["slot:S03"][0] >= 0.005 and res["slot:S02"][0] > max(s04))
    for r in ("S01", "S02"):
        sh = [res[f"slot:S21:{r}:{i}"][0] for i in range(SA.N_DRAWS)]
        g = res[f"slot:{r}"][0]
        out[f"chem_specific_{r}"] = bool(g > max(sh) and np.mean(sh) < 0.5 * g) if g > 0 else False
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True, choices=WORLDS)
    ap.add_argument("--reps", type=int, default=4)
    a = ap.parse_args()
    bench = D.load_bench()
    recs = {f: C.load(*f) for f in FOLDS}
    reps = []
    for i in range(a.reps):
        fk = FOLDS[i % len(FOLDS)]
        reps.append(one(bench, recs[fk], a.world, 9000 + 100 * WORLDS.index(a.world) + i, fk))
        print(a.world, i, {k: reps[-1][k] for k in ("crit1", "elec_specific", "chem_specific_S01", "chem_specific_S02")},
              {k.split(":", 1)[1]: round(v, 4) for k, v in reps[-1]["gain"].items() if "S04" not in k and "S21" not in k}, flush=True)
    n = len(reps)

    def rate(fn):
        p = float(np.mean([fn(r) for r in reps]))
        return {"rate": p, "mcse": float(np.sqrt(p * (1 - p) / n)), "n": n}
    summary = {"world": a.world, "folds": FOLDS, "reps": n,
               "crit1": {s: rate(lambda r, s=s: r["crit1"][s]) for s in ("S01", "S02", "S03")},
               "elec_specific": rate(lambda r: r["elec_specific"]),
               "chem_specific_S01": rate(lambda r: r["chem_specific_S01"]),
               "chem_specific_S02": rate(lambda r: r["chem_specific_S02"]),
               "mean_gain_R2": {p: float(np.mean([r["gain"][p] for r in reps])) for p in reps[0]["gain"]},
               "replicates": reps}
    out = paths.RES / "synthetic_a"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.world}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("replicates", "mean_gain_R2")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

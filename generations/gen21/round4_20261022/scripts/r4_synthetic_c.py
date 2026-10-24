#!/usr/bin/env python
"""Discriminating power of block C's transfer rule and controls on synthetic tasks (brief section 8 item 8; A4), run 3.

Real structure of two BPG folds (seed 130363, folds 2 and 3; replicates alternate) and the REAL An(III) record set
(extractants, publications, metals, union exclusions), with SYNTHETIC targets for both domains (no real label read).
Latent extractant properties from the standardised RDKit descriptors over cohort and An extractants (principal
components PC1..PC3, each scaled to unit sd):
  u_e = tanh(1.5 * PC1) * sign(PC2)          (the property that carries the Ln signal)
  s_e = tanh(1.5 * PC3)                       (smooth in descriptors, unrelated to the Ln signal)
Ln (identical in every world of a replicate -- the worlds are PAIRED):
  y_c(m) = a_c + b_c x + q_c x^2 - 0.09 e_Gd + nuisance_chemotype + SIGNAL_AMP * u_e * tetrad_e3 / sd(tetrad_e3) + noise
  (a, b, q ~ N(0, [1, 1, 0.3]); nuisance = sum of N(0, 0.03) * block A's shapes per chemotype; noise sd 0.11)
An log D = 1.5 * level_e + metal offset (Am 0, Cm 0.1, Pu -0.3, Cf 0.2) + noise (sd 0.3), with level_e =
  AN_SIGNAL:   u_e
  SMOOTH_NULL: s_e                            (a smooth auxiliary target the Ln task does not need)
  NULL:        v_e ~ N(0, 1) independently per extractant
Each replicate fits B1, B2 and S11 once (S11's inner stage is shared by the worlds), S15 once (its rows do not depend on
An targets), and S13, S14 and S16 (5 draws) per world, then scores the fold's C3 test pairs.
Transfer claim (upper bound of the sealed rule: vs B1 only, one fold, no harm rule, no B2):
  crit1(S13) and S13 - S11 >= 0.005 and S13 - S15 >= 0.005 and (S15 - S11) < 0.5 (S13 - S11) and S13 > max(S16)
  and mean(S16 - S11) < 0.5 (S13 - S11)
Selectivity sub-label: S13 - S14 > 0.  Gate-equivalent: S13's best positive-w_X inner score below S11's and B2's.

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_synthetic_c.py --rep 0
writes results/synthetic_c/rep_<i>.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import an_data as AD  # noqa: E402
from r4 import paths  # noqa: E402
from r4 import refs as RF  # noqa: E402
from r4 import slots_a as SA  # noqa: E402
from r4 import slots_b as SB  # noqa: E402
from r4 import slots_c as SC  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import metrics as MT  # noqa: E402
from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN  # noqa: E402

FOLDS = (("BPG", 130363, 2), ("BPG", 130363, 3))
WORLDS = ("NULL", "SMOOTH_NULL", "AN_SIGNAL")
NUISANCE_SHAPES = ("e_Gd", "e_Nd", "gd_break", "tetrad_e1", "tetrad_e3", "P2", "P3", "P4")
OFFSET = {"Am": 0.0, "Cm": 0.1, "Pu": -0.3, "Cf": 0.2}
SIGNAL_AMP, NOISE_SD, AN_NOISE_SD = 0.15, 0.11, 0.3


def latents(bench) -> tuple[dict, dict]:
    exts = sorted(set(map(str, bench.frame.extractant)) | set(AD.an3_records().extractant))
    R = np.array([SB.rdkit_desc(e) for e in exts])
    R = (R - R.mean(0)) / (R.std(0) + 1e-9)
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    pc = U[:, :3] * S[:3]
    pc = pc / pc.std(0)
    u = np.tanh(1.5 * pc[:, 0]) * np.sign(pc[:, 1])
    s = np.tanh(1.5 * pc[:, 2])
    out_u, out_s = {}, {}
    for e, a, b in zip(exts, u, s):
        for k in {e, AD._canon(e)}:
            out_u[k], out_s[k] = float(a), float(b)
    return out_u, out_s


def ln_targets(bench, seed, u):
    rng = np.random.default_rng(seed)
    x = interp.X_R
    xs = (x - x.mean()) / x.std()
    sh = SA.shapes()
    e3 = sh["tetrad_e3"] / np.std(sh["tetrad_e3"])
    chems = sorted(bench.frame.chemotype.astype(str).unique())
    nuis = {c: sum(rng.normal(0, 0.03) * sh[s] for s in NUISANCE_SHAPES) for c in chems}
    Ys = np.full_like(bench.Y, np.nan)
    for ci in range(bench.Y.shape[0]):
        a, b, q = rng.normal(0, [1.0, 1.0, 0.3])
        e = str(bench.frame.extractant.iat[ci])
        curve = (a + b * xs + q * xs ** 2 - 0.09 * sh["e_Gd"] + nuis[str(bench.frame.chemotype.iat[ci])]
                 + SIGNAL_AMP * u[e] * e3 + rng.normal(0, NOISE_SD, N_LN))
        Ys[ci] = np.where(np.isfinite(bench.Y[ci]), curve, np.nan)
    return Ys


def an_targets(world, seed, u, s, real):
    rng = np.random.default_rng(seed + 7)
    an = real.copy()
    v = {e: rng.normal() for e in sorted(an.extractant.unique())}
    lev = an.extractant.map({"AN_SIGNAL": u, "SMOOTH_NULL": s, "NULL": v}[world]).astype(float)
    an["log_D"] = 1.5 * lev + an.metal_symbol.map(OFFSET) + rng.normal(0, AN_NOISE_SD, len(an))
    return an


def score(t, cand, ref="B1"):
    y = t.y.to_numpy()
    g = t.assign(ec=(t[cand] - t.y).abs(), er=(t[ref] - t.y).abs())
    ch = g.groupby(["chemotype", "extractant"])[["ec", "er"]].mean().groupby(level=0).mean()
    return float(MT._r2(y, t[cand]) - MT._r2(y, t[ref])), float((ch.er - ch.ec).mean())


def table(rec, Ys, b1, fits) -> pd.DataFrame:
    rows = []
    for cell in rec["outer"]["cells"]:
        if cell["k_max"] < 3:
            continue
        obs = E.observe(Ys[cell["ci"]], cell, cell["seqs"]["DOPT_Q"], 3)
        lv = {"B1": RF.predict_levels(b1, obs, cell["metals"])}
        lv.update({k: SC.predict_levels(f, obs, cell) for k, f in fits.items()})
        for a, b in cell["query"]:
            r = {"chemotype": cell["chemotype"], "extractant": cell["extractant"], "y": Ys[cell["ci"]][a] - Ys[cell["ci"]][b]}
            r.update({k: v[a] - v[b] for k, v in lv.items()})
            rows.append(r)
    return pd.DataFrame(rows)


def one(bench, rec, seed, u, s, fk) -> dict:
    Ys = ln_targets(bench, seed, u)
    pb = copy.copy(bench)
    pb.Y = Ys
    real = AD.an3_records
    an_real = real()
    SC._S11_CACHE.clear()
    b1 = RF.fit_reference(pb, rec, Ys, "B1")
    b2 = RF.fit_reference(pb, rec, Ys, "B2")
    b2_best = min(b2["inner_scores"].values())
    out = {"seed": seed, "fold": list(fk), "worlds": {}}
    shared = {}
    for world in WORLDS:
        an_w = an_targets(world, seed, u, s, an_real)
        AD.an3_records = lambda a=an_w: a                    # synthetic An targets (same records, rows, exclusions)
        try:
            fits = {}
            if not shared:                                     # S11 and S15 do not depend on An targets
                shared["S11:0"] = SC.fit_slot(pb, rec, Ys, "S11", b1)
                shared["S15:0"] = SC.fit_slot(pb, rec, Ys, "S15", b1)
            fits.update(shared)
            arms = [("S13", 0), ("S14", 0)] + [("S16", i) for i in range(len(SC.PERM_SEEDS))]
            fits.update({f"{a}:{d}": SC.fit_slot(pb, rec, Ys, a, b1, draw=d) for a, d in arms})
        finally:
            AD.an3_records = real
        t = table(rec, Ys, b1, fits)
        res = {k: score(t, k) for k in fits}
        g = {k: v[0] for k, v in res.items()}
        s16 = [g[f"S16:{i}"] for i in range(len(SC.PERM_SEEDS))]
        crit1 = res["S13:0"][0] >= 0.01 and res["S13:0"][1] >= 0.005
        up = g["S13:0"] - g["S11:0"]
        rule = (up >= 0.005 and g["S13:0"] - g["S15:0"] >= 0.005 and (g["S15:0"] - g["S11:0"]) < 0.5 * up
                and g["S13:0"] > max(s16) and np.mean([v - g["S11:0"] for v in s16]) < 0.5 * up)
        sx = fits["S13:0"]["inner_scores"]["S13"]
        best_pos = min(v for k, v in sx.items() if k not in ("zero", "0.0"))
        out["worlds"][world] = {
            "gain": g, "gain_mae": {k: v[1] for k, v in res.items()},
            "choices": {k: f["choice"] for k, f in fits.items()},
            "inner": {k: f["inner_scores"] for k, f in fits.items() if not k.startswith("S16")},
            "crit1_S11": bool(res["S11:0"][0] >= 0.01 and res["S11:0"][1] >= 0.005), "crit1_S13": bool(crit1),
            "transfer_rule": bool(rule), "transfer_claim": bool(crit1 and rule),
            "selectivity": bool(g["S13:0"] - g["S14:0"] > 0),
            "gate_win": bool(best_pos < min(fits["S13:0"]["inner_scores"]["S11"].values()) and best_pos < b2_best)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, required=True)
    a = ap.parse_args()
    t0 = time.time()
    bench = D.load_bench()
    u, s = latents(bench)
    fk = FOLDS[a.rep % len(FOLDS)]
    res = one(bench, C.load(*fk), 13300 + a.rep, u, s, fk)
    out = paths.RES / "synthetic_c"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"rep_{a.rep}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8", newline="\n")
    for w, r in res["worlds"].items():
        print(a.rep, w, {k: r[k] for k in ("crit1_S13", "transfer_rule", "transfer_claim", "selectivity", "gate_win")},
              {k: round(v, 4) for k, v in r["gain"].items()}, flush=True)
    print(f"rep {a.rep} done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""D4 follow-up - is a gain calibration worth anything if you may measure ONE pair?

The global-gain sweep in d4_amplitude_shape.py is an oracle and buys almost nothing.
The oracle per-extractant gain buys ~0.09 MAE, so the amplitude error is extractant-specific
rather than a corpus-wide scale factor. This script asks the achievable version of that:
fit g from a single held-out pair ("anchor") of the new extractant / cell, then score the
REMAINING pairs, always comparing against g = 1 on the identical evaluation set.

Run from the repo root:
    .venv/Scripts/python.exe generations/gen13_separation/analysis/stage2/d4_amplitude_shape/d4_anchor_calibration.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
GEN13 = ROOT / "generations" / "gen13_separation"
PRED = GEN13 / "predictions" / "B_primary"
OUT = GEN13 / "analysis" / "stage2" / "d4_amplitude_shape"
SLUG = "d4_amplitude_shape"

ARMS = ["C_DIRECT_ROW", "M_SELECTED", "M_PHYSICS_radius+radius_sq", "M_LOWRANK_K2",
        "X_ENS_DIRECT+LOWRANK_K2"]
LN = ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu")
IDX = {m: i for i, m in enumerate(LN)}
MIN_ANCHOR_PRED = 0.05   # do not calibrate off a near-zero predicted contrast
N_REPS = 20
# the single-pair gain is noisy, so the clip range matters: report the sensitivity
CLIPS = [(0.8, 1.25), (0.7, 1.5), (0.5, 2.0), (0.25, 4.0)]
DEFAULT_CLIP = (0.5, 2.0)


def macro_mae(seed: np.ndarray, ext: np.ndarray, ae: np.ndarray) -> float:
    t = pd.DataFrame({"seed": seed, "ext": ext, "ae": ae})
    return float(t.groupby(["seed", "ext"])["ae"].mean().groupby(level=0).mean().mean())


def run() -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        df = pd.read_parquet(PRED / f"{arm}.parquet").reset_index(drop=True)
        df["gap"] = (df["A"].map(IDX) - df["B"].map(IDX)).abs()
        y = df["y"].to_numpy()
        p = df["prediction"].to_numpy()
        seed = df["split_seed"].to_numpy()
        ext = df["extractant"].to_numpy()

        for mode, keys in [("per_extractant", ["split_seed", "extractant"]),
                           ("per_cell", ["split_seed", "cell_id"])]:
            gid = df.groupby(keys, sort=False).ngroup().to_numpy()
            ng = gid.max() + 1
            order = np.argsort(gid, kind="stable")
            starts = np.searchsorted(gid[order], np.arange(ng))
            sizes = np.bincount(gid, minlength=ng)

            def score(anchor_pos: np.ndarray, tag: str, rep: int,
                      clip: tuple[float, float]) -> None:
                is_anchor = np.zeros(len(df), bool)
                is_anchor[anchor_pos] = True
                pa, ya = p[anchor_pos], y[anchor_pos]
                raw = ya / np.where(pa == 0, np.nan, pa)
                g_grp = np.where(np.abs(pa) >= MIN_ANCHOR_PRED,
                                 np.clip(raw, clip[0], clip[1]), 1.0)
                g_grp = np.nan_to_num(g_grp, nan=1.0)
                g_row = g_grp[gid]
                ev = (~is_anchor) & (sizes[gid] >= 2)
                base = macro_mae(seed[ev], ext[ev], np.abs(p[ev] - y[ev]))
                cal = macro_mae(seed[ev], ext[ev], np.abs(g_row[ev] * p[ev] - y[ev]))
                rows.append(dict(arm=arm, mode=mode, anchor=tag, rep=rep,
                                 clip_lo=clip[0], clip_hi=clip[1],
                                 n_eval_pairs=int(ev.sum()), n_groups=int(ng),
                                 frac_groups_calibratable=float(
                                     (np.abs(pa) >= MIN_ANCHOR_PRED).mean()),
                                 frac_gains_clipped=float(
                                     np.mean((g_grp <= clip[0]) | (g_grp >= clip[1]))),
                                 median_fitted_gain=float(np.median(g_grp)),
                                 macro_mae_g1=base, macro_mae_calibrated=cal,
                                 macro_mae_gain=base - cal))

            # deterministic anchor: the widest-gap pair of the group (most informative)
            gap = df["gap"].to_numpy()
            best = np.full(ng, -1)
            bestgap = np.full(ng, -1)
            for i in range(len(df)):
                gg = gid[i]
                if gap[i] > bestgap[gg]:
                    bestgap[gg] = gap[i]
                    best[gg] = i
            for clip in CLIPS:
                score(best, "widest_gap", -1, clip)

            rng = np.random.default_rng(11)
            for rep in range(N_REPS):
                pick = order[starts + np.floor(rng.random(ng) * sizes).astype(int)]
                score(pick, "random", rep, DEFAULT_CLIP)
                if rep < 3:
                    for clip in CLIPS:
                        if clip != DEFAULT_CLIP:
                            score(pick, "random", rep, clip)

        print(f"[{arm}] done", flush=True)
        del df

    res = pd.DataFrame(rows)
    return res


def main() -> None:
    res = run()
    res.to_csv(OUT / f"{SLUG}_anchor_calibration_raw.csv", index=False)
    summ = (res.groupby(["arm", "mode", "anchor", "clip_lo", "clip_hi"], as_index=False)
            .agg(n_eval_pairs=("n_eval_pairs", "mean"),
                 n_groups=("n_groups", "first"),
                 frac_groups_calibratable=("frac_groups_calibratable", "mean"),
                 frac_gains_clipped=("frac_gains_clipped", "mean"),
                 median_fitted_gain=("median_fitted_gain", "mean"),
                 macro_mae_g1=("macro_mae_g1", "mean"),
                 macro_mae_calibrated=("macro_mae_calibrated", "mean"),
                 macro_mae_gain=("macro_mae_gain", "mean"),
                 macro_mae_gain_sd_over_reps=("macro_mae_gain", "std")))
    summ.to_csv(OUT / f"{SLUG}_anchor_calibration.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

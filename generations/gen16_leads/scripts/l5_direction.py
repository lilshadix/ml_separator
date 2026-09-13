"""L5 step 4b (registered): is the zero-shot direction *probability* calibrated?

Gen15 §8's verdict is that the only defensible zero-shot claim this programme has is a direction
call.  A calibrated probability attached to that call is itself a deliverable -- so this script
asks whether the number the deployed logistic emits can be read as a probability at all.

Three probabilities on the *same* out-of-fold cells, under all five designs:

* ``RAW``     ``gen14.models.dir_logistic()`` on ``TOPO39``, the deployed direction model, exactly
  as ``gen14.dirbench.run`` scores it.
* ``PLATT``   the same model, its output passed through a 1-D logistic in ``logit(p)`` fitted
  **inside the training fold** on the fold plan's own inner validation block
  (``Folds.inner_train_index`` / ``inner_validation_index``, restricted to well-determined cells
  and chemotype-balanced-weighted like every other fit here).  No held-out information is used.
* ``CONST``   ``gen14.models.dir_train_majority`` -- the chemotype-balanced training base rate,
  the constant the registered rule requires the recalibration to beat.

Endpoints: Brier score (unit = extractant, one vote each, averaged over the five discovery seeds;
resampling block = chemotype, gen13's frozen paired bootstrap) and ECE over 10 equal-count bins
(pooled over well-determined held-out cells, with a chemotype-blocked interval from
``gen14.dirbench.Blocked``'s frozen resampling indices).

Registered rule: the calibrated probability is a *deliverable* if PLATT's Brier beats CONST with a
CI excluding zero in all five designs and its ECE <= 0.10.

Usage:  .venv/Scripts/python.exe generations/gen16_leads/scripts/l5_direction.py [designs]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights  # noqa: E402
from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.splits import all_folds  # noqa: E402
from gen14 import models as M  # noqa: E402
from gen14.dirbench import (MIN_METALS, MIN_TRAIN, Blocked, feature_sets, load,  # noqa: E402
                            run, unit_hits)
from gen15 import valuebench as V  # noqa: E402

OUT = bootstrap.RESULTS / "L5"
OUT.mkdir(parents=True, exist_ok=True)
DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else list(V.DESIGNS)
N_BINS = 10
ECE_REPS = 2000
EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def platt_oof(bench, design: str) -> pd.DataFrame:
    """``dirbench.run``'s loop, plus an inner-fold Platt map fitted on the fold's own inner block.

    The raw column of the frame this returns is asserted identical to ``dirbench.run``'s, so the
    only thing added is the recalibration.
    """
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)[:, fs["TOPO39"]]
    amp = bench.coef[:, 0]
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    ext_all = bench.frame.extractant.to_numpy()
    direction = M.dir_logistic()
    rows = []
    for f in all_folds(bench.frame, design=design):
        tr = f.train_index[rich[f.train_index]]
        te = f.test_index
        if len(tr) < MIN_TRAIN or len(te) < 1 or len(set(amp[tr] < 0)) < 2:
            continue
        w = cell_weights(bench.groups[tr], bench.n_obs[tr])
        p_raw = np.asarray(direction(X[tr], amp[tr], w, bench.groups[tr], X[te], f.model_seed,
                                     ext_all[tr]), dtype=float).ravel()
        # ---- inner-fold Platt ----
        itr = f.inner_train_index[rich[f.inner_train_index]]
        iva = f.inner_validation_index[rich[f.inner_validation_index]]
        y_iva = (amp[iva] < 0).astype(int)
        fitted = False
        p_platt = p_raw.copy()
        if len(itr) >= MIN_TRAIN and len(iva) >= 10 and len(set(amp[itr] < 0)) == 2 \
                and len(set(y_iva)) == 2:
            wi = cell_weights(bench.groups[itr], bench.n_obs[itr])
            p_iva = np.asarray(direction(X[itr], amp[itr], wi, bench.groups[itr], X[iva],
                                         f.model_seed, ext_all[itr]), dtype=float).ravel()
            wv = cell_weights(bench.groups[iva], bench.n_obs[iva])
            cal = LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs")
            cal.fit(_logit(p_iva).reshape(-1, 1), y_iva, sample_weight=wv)
            p_platt = cal.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]
            fitted = True
        # EXPLORATORY, added after the registered arms were scored and never substituted for
        # them: the registered ``CONST`` is ``dir_train_majority``, the CHEMOTYPE-BALANCED
        # training base rate (0.37), which is a weak constant because most chemotypes are
        # light-selective while most cells are heavy-selective diglycolamides.  The unweighted
        # training cell base rate is the hardest constant in Brier score, so it is reported
        # beside it.  Adding a harder comparator cannot lower the bar.
        p_cellrate = float((amp[tr] < 0).mean())
        for j, ci in enumerate(te):
            rows.append({"design": design, "split_seed": f.seed, "fold": f.fold,
                         "cell_id": bench.frame.cell_id.iat[ci],
                         "extractant": ext_all[ci], "chemotype": bench.groups[ci],
                         "n_metals": int(bench.frame.n_metals.iat[ci]),
                         "y": int(amp[ci] < 0), "p_raw": float(p_raw[j]),
                         "p_platt": float(p_platt[j]), "p_cellrate": p_cellrate,
                         "platt_fitted": bool(fitted)})
    return pd.DataFrame(rows)


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error over ``n_bins`` EQUAL-COUNT bins."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(p)
    if n < n_bins:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    tot = 0.0
    for i in range(n_bins):
        idx = order[edges[i]:edges[i + 1]]
        if len(idx) == 0:
            continue
        tot += len(idx) / n * abs(p[idx].mean() - y[idx].mean())
    return float(tot)


def ece_blocked_ci(cells: pd.DataFrame, col: str, reps: int = ECE_REPS) -> tuple[float, float, float]:
    """Point ECE and a chemotype-blocked percentile interval, using ``Blocked``'s frozen picks."""
    boot = Blocked(cells.chemotype, reps=reps)
    members = [np.flatnonzero(cells.chemotype.to_numpy() == c) for c in boot.names]
    p = cells[col].to_numpy(dtype=float)
    y = cells["y"].to_numpy(dtype=float)
    draws = np.array([ece(p[t], y[t]) for t in
                      (np.concatenate([members[j] for j in row]) for row in boot.picks)])
    d = draws[np.isfinite(draws)]
    return ece(p, y), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def per_extractant_brier(cells: pd.DataFrame, arms: dict[str, str]) -> pd.DataFrame:
    """One row per (seed, extractant, arm): mean Brier over the extractant's well-determined cells."""
    rich = cells[cells.n_metals >= MIN_METALS]
    out = []
    for arm, col in arms.items():
        g = rich.assign(brier=(rich[col] - rich["y"]) ** 2, hit=((rich[col] >= 0.5).astype(int) == rich["y"]).astype(float))
        agg = (g.groupby(["split_seed", "extractant", "chemotype"], as_index=False)
                 .agg(brier=("brier", "mean"), hit=("hit", "mean"), n_cells=("brier", "size")))
        agg["arm"] = arm
        out.append(agg)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    t0 = time.time()
    bench = load()
    boards, cons, allcells = [], [], []
    comps = {"PLATT_vs_CONST": ("CONST", "PLATT"),
             "RAW_vs_CONST": ("CONST", "RAW"),
             "PLATT_vs_RAW": ("RAW", "PLATT"),
             "RAW_vs_CONSTCELL": ("CONSTCELL", "RAW"),
             "PLATT_vs_CONSTCELL": ("CONSTCELL", "PLATT")}
    fam = {"PLATT_vs_CONST": "registered", "RAW_vs_CONST": "exploratory",
           "PLATT_vs_RAW": "exploratory", "RAW_vs_CONSTCELL": "exploratory",
           "PLATT_vs_CONSTCELL": "exploratory"}
    for d in DESIGNS:
        td = time.time()
        cells = platt_oof(bench, d)
        # --- the raw column must be dirbench.run's, to the last bit ---
        oof_raw = run(bench, "RAW", M.candidate(M.dir_logistic()), features=feature_sets(bench)["TOPO39"],
                      design=d)
        key = ["split_seed", "fold", "cell_id"]
        a = oof_raw.cells.set_index(key).sort_index()
        b = cells.set_index(key).sort_index()
        assert a.index.equals(b.index), f"{d}: OOF cell sets differ from dirbench.run"
        dmax = float(np.abs(a["p"].to_numpy() - b["p_raw"].to_numpy()).max())
        assert dmax < 1e-12, f"{d}: raw p differs from dirbench.run by {dmax:.2e}"
        oof_const = run(bench, "CONST", M.candidate(M.dir_train_majority),
                        features=feature_sets(bench)["TOPO39"], design=d)
        c_const = oof_const.cells.set_index(key)["p"]
        cells["p_const"] = c_const.reindex(b.index).to_numpy()
        allcells.append(cells)

        pe = per_extractant_brier(cells, {"RAW": "p_raw", "PLATT": "p_platt", "CONST": "p_const",
                                          "CONSTCELL": "p_cellrate"})
        pe.insert(0, "design", d)
        pe.to_parquet(OUT / f"perext_direction_{d}.parquet")
        rich = cells[cells.n_metals >= MIN_METALS]
        for arm, col in (("RAW", "p_raw"), ("PLATT", "p_platt"), ("CONST", "p_const"),
                         ("CONSTCELL", "p_cellrate")):
            sub = pe[pe.arm == arm]
            macro = sub.groupby("split_seed")[["brier", "hit"]].mean().mean()
            e, lo, hi = ece_blocked_ci(rich, col)
            boards.append({"design": d, "arm": arm,
                           "macro_brier": float(macro["brier"]),
                           "macro_accuracy": float(macro["hit"]),
                           "pooled_brier": float(((rich[col] - rich["y"]) ** 2).mean()),
                           "ece10": e, "ece10_ci_low": lo, "ece10_ci_high": hi,
                           "mean_p": float(rich[col].mean()), "base_rate": float(rich["y"].mean()),
                           "n_cells": int(len(rich)), "n_units": int(sub.extractant.nunique()),
                           "n_chemotypes": int(sub.chemotype.nunique()),
                           "n_seeds": int(sub.split_seed.nunique()),
                           "platt_folds_fitted": float(cells.platt_fitted.mean())})
        c = paired_contrasts(pe, comps, value="brier", replicates=10_000)
        if len(c):
            c.insert(0, "design", d)
            c["family"] = c["comparison"].map(fam)
            c["lead"] = "L5"
            cons.append(c)
        print(f"  [{d}] {len(cells)} OOF cells, {len(rich)} well-determined, "
              f"{time.time() - td:.0f}s", flush=True)

    B = pd.DataFrame(boards)
    B.to_csv(OUT / "calibration_direction.csv", index=False)
    C = pd.concat(cons, ignore_index=True)
    C.to_csv(OUT / "calibration_direction_contrasts.csv", index=False)
    pd.concat(allcells, ignore_index=True).to_parquet(OUT / "direction_oof_cells.parquet")

    pd.set_option("display.width", 240)
    print("\n=== zero-shot direction probability: Brier and ECE ===")
    print(B.round(4).to_string(index=False))
    print("\n=== contrasts on Brier (positive favours the candidate) ===")
    show = ["design", "comparison", "family", "point", "ci95_low", "ci95_high", "bca_low",
            "bca_high", "p_two_sided", "n_units", "seeds_positive", "loco_sign_stable", "passes_P1"]
    print(C[show].round(4).to_string(index=False))
    print(f"\n[l5-direction] total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""Is there ANY out-of-fold signal in the embedding for log|a| and for b?

The MAE board answers "does this arm beat G14".  It cannot separate "the representation is empty"
from "the representation has something and the arm shape wastes it", because the composite arm has
to multiply a magnitude by a direction and both can be wrong.  This probe removes the arm entirely
and asks the direct question: fit the representation to the target inside the training fold, predict
the held-out cells, and correlate.

Three guards, because a raw correlation here is worthless:

* **n_metals** is Spearman ~+0.44 with |a| in this corpus.  Reported as the partial Spearman of the
  prediction with the target once n_metals is regressed out of both (rank-residual partial).
* **publication** identity is what BP masks; the BP row is therefore the only one that is about
  chemistry rather than about which laboratory reported the cell.
* **chemotype**, via the design B / BP hold-out itself.

A correlation that survives all three and is materially above zero would mean the arm is the problem.
A correlation at zero means the representation is.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT / "gen15_curve"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import features as FEAT                                            # noqa: E402
from arms_embed import _kernel_ridge_predict, _ridge_predict       # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights     # noqa: E402
from gen13sep.splits import all_folds                              # noqa: E402
from gen15 import valuebench as V                                  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
EPS = 0.05
MIN_METALS = 5


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman of x and y with z partialled out, on ranks (Spearman partial correlation)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 10:
        return float("nan")
    R = np.c_[rankdata(x[ok]), rankdata(y[ok]), rankdata(z[ok])].astype(float)
    R -= R.mean(0)
    A = np.c_[np.ones(len(R)), R[:, 2]]
    rx = R[:, 0] - A @ np.linalg.lstsq(A, R[:, 0], rcond=None)[0]
    ry = R[:, 1] - A @ np.linalg.lstsq(A, R[:, 1], rcond=None)[0]
    d = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / d) if d > 0 else float("nan")


def probe(bench, tag: str, target: str, design: str, how: str = "ridge",
          alpha: float = 100.0) -> pd.DataFrame:
    """Out-of-fold predictions of ``target`` from ``tag`` under ``design``."""
    ext = tuple(bench.frame.extractant.astype(str).tolist())
    X = FEAT.cell_block(tag, ext)
    amp, cur = bench.coef[:, 0], bench.coef[:, 1]
    y_all = np.log(np.abs(amp) + EPS) if target == "log_abs_a" else cur
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    rows = []
    for f in all_folds(bench.frame, design=design):
        tr = f.train_index[rich[f.train_index]]
        te = f.test_index[rich[f.test_index]]
        if len(tr) < 40 or len(te) < 2:
            continue
        w = cell_weights(bench.groups[tr], bench.n_obs[tr])
        if how == "ridge":
            pred = _ridge_predict(X[tr], y_all[tr], w, X[te], alpha, None, f.model_seed)
        else:
            pred = _kernel_ridge_predict(X[tr], y_all[tr], w, X[te], alpha, how, 1.0)
        for j, ci in enumerate(te):
            rows.append({"seed": f.seed, "fold": f.fold, "cell": int(ci),
                         "pred": float(pred[j]), "true": float(y_all[ci]),
                         "n_metals": float(bench.frame.n_metals.iat[ci]),
                         "chemotype": str(bench.frame.chemotype.iat[ci])})
    return pd.DataFrame(rows)


def summarise(d: pd.DataFrame, tag: str, target: str, design: str, how: str) -> dict:
    if d.empty:
        return {}
    # pool over seeds by averaging each cell's prediction, so a cell counts once
    g = d.groupby("cell").agg(pred=("pred", "mean"), true=("true", "mean"),
                              n_metals=("n_metals", "first"), chemotype=("chemotype", "first"))
    r = float(spearmanr(g["pred"], g["true"]).statistic)
    rp = partial_spearman(g["pred"].to_numpy(), g["true"].to_numpy(), g["n_metals"].to_numpy())
    # leave-one-chemotype-out range of the plain Spearman
    vals = []
    for ch in g.chemotype.unique():
        s = g[g.chemotype != ch]
        if len(s) > 20:
            vals.append(float(spearmanr(s["pred"], s["true"]).statistic))
    return {"block": tag, "target": target, "design": design, "estimator": how,
            "n_cells": int(len(g)), "spearman": r, "partial_spearman_given_n_metals": rp,
            "loco_spearman_min": float(np.min(vals)) if vals else np.nan,
            "loco_spearman_max": float(np.max(vals)) if vals else np.nan,
            "sd_pred": float(g["pred"].std()), "sd_true": float(g["true"].std())}


def main() -> None:
    tags = sys.argv[1].split(",") if len(sys.argv) > 1 else \
        ["chemberta_mtr__mean", "molformer__mean", "morgan2"]
    designs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["B", "BP"]
    bench = V.load()
    rows = []
    for tag in tags:
        for target in ("log_abs_a", "b"):
            for how in ("ridge", "cosine"):
                for design in designs:
                    d = probe(bench, tag, target, design, how=how)
                    s = summarise(d, tag, target, design, how)
                    if s:
                        rows.append(s)
                        print(f"{tag:22s} {target:10s} {design:3s} {how:7s} "
                              f"rho={s['spearman']:+.3f} partial={s['partial_spearman_given_n_metals']:+.3f} "
                              f"loco[{s['loco_spearman_min']:+.3f},{s['loco_spearman_max']:+.3f}]",
                              flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "signal_probe.csv", index=False)
    print("\n=== out-of-fold Spearman of prediction vs truth ===")
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

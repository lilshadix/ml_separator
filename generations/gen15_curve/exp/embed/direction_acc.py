"""Direction macro accuracy of a representation, on gen14's own metric and units.

``gen14.dirbench.run`` can only index columns of the LEAN block matrix, so it cannot be handed an
external representation; everything else -- the fold plan, the rich-cell training convention, the
per-extractant unit, the chemotype-blocked bootstrap -- is imported from it unchanged, and only the
feature matrix is swapped.  The number to beat is TOPO39's 0.821 macro accuracy under BP.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT / "gen15_curve"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import features as FEAT                                            # noqa: E402
from arms_embed import _logit_p                                    # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights     # noqa: E402
from gen13sep.splits import all_folds                              # noqa: E402
from gen14 import dirbench as D                                    # noqa: E402
from gen14.models import candidate, dir_logistic                   # noqa: E402
from gen15.valuebench import DESIGNS                               # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def run_matrix(bench, name: str, X: np.ndarray, design: str, *, C: float = 1.0,
               n_pca: int | None = None) -> D.OOF:
    """gen14.dirbench.run with an arbitrary cell-level feature matrix."""
    amp = bench.coef[:, 0]
    rich = bench.frame.n_metals.to_numpy() >= D.MIN_METALS
    rows, t0 = [], time.time()
    for f in all_folds(bench.frame, design=design):
        tr = f.train_index[rich[f.train_index]]
        te = f.test_index
        if len(tr) < D.MIN_TRAIN or len(te) < 1 or len(set(amp[tr] < 0)) < 2:
            continue
        w = cell_weights(bench.groups[tr], bench.n_obs[tr])
        p = _logit_p(X[tr], (amp[tr] < 0).astype(int), w, X[te], C, n_pca, f.model_seed)
        for j, ci in enumerate(te):
            rows.append({"design": design, "split_seed": f.seed, "fold": f.fold,
                         "cell_id": bench.frame.cell_id.iat[ci],
                         "extractant": bench.frame.extractant.iat[ci],
                         "chemotype": bench.frame.chemotype.iat[ci],
                         "n_metals": int(bench.frame.n_metals.iat[ci]),
                         "cell_index": int(ci), "amp": float(amp[ci]),
                         "y": int(amp[ci] < 0), "p": float(p[j]), "mag": 0.0})
    return D.OOF(cells=pd.DataFrame(rows), design=design, name=name, seconds=time.time() - t0)


def run_topo(bench, design: str) -> D.OOF:
    """Gen14's deployed direction model, re-measured here so the comparison is like for like."""
    fs = D.feature_sets(bench)
    return D.run(bench, "TOPO39", candidate(dir_logistic(C=1.0)), features=fs["TOPO39"],
                 design=design)


def run_majority(bench, design: str) -> D.OOF:
    """Always call the training fold's majority direction -- the floor a classifier must clear.

    Implemented as ``run_matrix`` on a single constant column, so the fold plan, the rich-cell
    training set, the chemotype weights and the 0.5 decision rule are byte-identical to every other
    row of the table; a constant feature leaves only the intercept, i.e. the weighted majority.
    """
    X = np.ones((len(bench.frame), 1), dtype=float)
    return run_matrix(bench, "MAJORITY", X, design, C=1.0)


def main() -> None:
    tags = sys.argv[1].split(",")
    designs = sys.argv[2].split(",") if len(sys.argv) > 2 else list(DESIGNS)
    Cs = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.01, 0.1, 1.0]
    bench = D.load()
    ext = tuple(bench.frame.extractant.astype(str).tolist())
    out = []
    for design in designs:
        oofs = [run_topo(bench, design)]
        for t in tags:
            X = FEAT.cell_block(t, ext)
            for C in Cs:
                oofs.append(run_matrix(bench, f"{t}_C{C:g}", X, design, C=C))
            for k in (8, 16, 20):
                oofs.append(run_matrix(bench, f"{t}_P{k}_C1", X, design, C=1.0, n_pca=k))
        b = D.direction_board(oofs, baseline="TOPO39")
        out.append(b)
        print(f"\n=== direction macro accuracy, design {design} ===")
        print(b.round(4).to_string(index=False))
    B = pd.concat(out, ignore_index=True)
    stem = "+".join(t.replace(":", "-")[:14] for t in tags)
    B.to_csv(OUT / f"direction_{stem}.csv", index=False)


if __name__ == "__main__":
    main()

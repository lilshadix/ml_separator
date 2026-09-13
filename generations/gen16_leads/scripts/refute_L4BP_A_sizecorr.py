"""Refuter lens A, step 1: is the AOPT order a size order?

Reads the lead's orders.parquet, recomputes per-fold chemotype cell counts from the frozen bench,
and reports (a) Spearman(order position, chemotype cell count) per order, (b) the fraction of
AOPT's first-k picks that are the k largest chemotypes, (c) the cumulative training-cell share.
No fits.  Writes to gen16_leads/results/refutation/L4BP/A/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction", ROOT / "generations" / "gen15_curve"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep.splits import all_folds  # noqa: E402
from gen14.dirbench import load  # noqa: E402
from gen15.valuebench import MIN_METALS  # noqa: E402

OUT = ROOT / "generations" / "gen16_leads" / "results" / "refutation" / "L4BP" / "A"
OUT.mkdir(parents=True, exist_ok=True)
DESIGNS = ("B", "BR", "BQ", "A", "BP")
BUDGETS = (6, 9, 12, 16, 20, 24)


def main() -> None:
    bench = load()
    groups = bench.groups.astype(str)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    n_metals = bench.frame.n_metals.to_numpy()

    orders = pd.read_parquet(ROOT / "generations" / "gen16_leads" / "results" / "L4" / "orders.parquet")
    rows, share_rows = [], []
    for design in DESIGNS:
        for f in all_folds(bench.frame, design=design):
            tr = f.train_index
            g = groups[tr]
            size_all = pd.Series(g).value_counts()
            size_rich = pd.Series(g[rich[tr]]).value_counts()
            metals = pd.Series(n_metals[tr]).groupby(pd.Series(g).to_numpy()).sum()
            sub = orders[(orders.design == design) & (orders.split_seed == f.seed)
                         & (orders.fold == f.fold)]
            for (order, draw), blk in sub.groupby(["order", "draw"]):
                blk = blk.sort_values("position")
                cs = blk.chemotype.to_numpy()
                pos = blk.position.to_numpy()
                sa = np.array([size_all.get(c, 0) for c in cs], dtype=float)
                sr = np.array([size_rich.get(c, 0) for c in cs], dtype=float)
                sm = np.array([metals.get(c, 0) for c in cs], dtype=float)
                rows.append({
                    "design": design, "seed": f.seed, "fold": f.fold, "order": order, "draw": draw,
                    "rho_pos_vs_cells": float(spearmanr(pos, sa).statistic),
                    "rho_pos_vs_richcells": float(spearmanr(pos, sr).statistic),
                    "rho_pos_vs_metals": float(spearmanr(pos, sm).statistic),
                    "n_chemotypes": len(cs), "n_train_cells": int(len(tr)),
                })
                if draw == 0 or order in ("AOPT", "UNCERT"):
                    big_all = list(size_all.sort_values(ascending=False).index)
                    big_rich = list(size_rich.sort_values(ascending=False).index)
                    for k in BUDGETS:
                        share_rows.append({
                            "design": design, "seed": f.seed, "fold": f.fold, "order": order,
                            "draw": draw, "k": k,
                            "cells_at_k": float(sa[:k].sum()),
                            "cell_share_at_k": float(sa[:k].sum() / len(tr)),
                            "overlap_top_k_by_cells": len(set(cs[:k]) & set(big_all[:k])) / k,
                            "overlap_top_k_by_richcells": len(set(cs[:k]) & set(big_rich[:k])) / k,
                        })
    R = pd.DataFrame(rows)
    S = pd.DataFrame(share_rows)
    R.to_csv(OUT / "order_size_correlation.csv", index=False)
    S.to_csv(OUT / "order_size_share.csv", index=False)

    pd.set_option("display.width", 250)
    print("=== Spearman(order position, chemotype size) -- mean over folds/draws ===")
    print(R.groupby(["design", "order"])[["rho_pos_vs_cells", "rho_pos_vs_richcells",
                                          "rho_pos_vs_metals"]].mean().round(3).to_string())
    print()
    print("=== fraction of the first k picks that are among the k largest chemotypes ===")
    piv = S[S.draw == 0].pivot_table(index=["design", "order"], columns="k",
                                     values="overlap_top_k_by_richcells")
    print(piv.round(3).to_string())
    print()
    print("=== share of the fold's training CELLS held at budget k ===")
    piv2 = S.pivot_table(index=["design", "order"], columns="k", values="cell_share_at_k")
    print(piv2.round(3).to_string())

    # Kish effective n of the 90 extractants over 45 chemotype blocks
    ext = bench.frame[["extractant", "chemotype"]].drop_duplicates("extractant")
    nb = ext.groupby("chemotype").size().to_numpy(dtype=float)
    print(f"\nextractant units {int(nb.sum())} in {len(nb)} chemotype blocks; "
          f"Kish n_eff over blocks = {nb.sum() ** 2 / (nb ** 2).sum():.2f}")


if __name__ == "__main__":
    main()

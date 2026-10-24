#!/usr/bin/env python
"""Theme "minimal-measurement recipes": is the static-set gain over HULL (no prior, D7) just "measure the worst pair
directly"?  Re-runs the LOPO-static rule of recipe_extra with sets that contain a strict-adjacent metal pair
masked out (legal-only), and reports the in-sample best legal sets.  EXPLORATORY.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe_static_legal.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from gen20k.data import N_LN, LANTHANIDES, Z_VEC  # noqa: E402
import recipe_analyze as RA  # noqa: E402
from recipe_extra import RA_lopo  # noqa: E402

OUT = RA.OUT


def main() -> int:
    cells = pd.read_parquet(OUT / "recipe_cells.parquet")
    pairs = pd.read_parquet(OUT / "recipe_pairs.parquet")
    rows = []
    for pk, sub in (("ALL229", cells), ("noC5_214", cells[~cells.c5])):
        P = RA.Pop(sub, pk)
        idx = sub.ci.to_numpy()
        for n in (2, 3, 4, 5):
            z = np.load(OUT / f"recipe_k{n}.npz")
            combos = list(itertools.combinations(range(N_LN), n))
            adj = np.array([any(Z_VEC[b] - Z_VEC[a] == 1 for a, b in itertools.combinations(c, 2)) for c in combos])
            hull = pairs[pairs.n_read == n].groupby("ci")["hull_D7"].sum().reindex(idx).fillna(0).to_numpy()
            full = {n: {"sse_D7": z["sse_D7"]}}
            legal = {n: {"sse_D7": np.where(adj[None, :], np.nan, z["sse_D7"])}}
            s_full, tops_f = RA_lopo(full, "D7", n, sub, hull)
            s_leg, tops_l = RA_lopo(legal, "D7", n, sub, hull)
            for nm, s, tops in (("LOPO_all_sets", s_full, tops_f), ("LOPO_legal_sets", s_leg, tops_l)):
                d = P.summ(s, hull)
                vc = pd.Series(tops).value_counts().head(3).to_dict()
                rows.append({"pop": pk, "n": n, "rule": nm, "R2": d["R2"], "lo": d["lo"], "hi": d["hi"],
                             "d_vs_HULL": d["d"], "d_lo": d["d_lo"], "d_hi": d["d_hi"], "top_choices": vc})
            d = P.summ(hull)
            rows.append({"pop": pk, "n": n, "rule": "HULL", "R2": d["R2"], "lo": d["lo"], "hi": d["hi"]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "recipe_static_legal.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print(t.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Replicate gen15's measured-mode board under BP with gen15's own code (Model 1 anchor), and
report the arithmetic-closure fraction on gen15's own scored rows.

    .venv/bin/python generations/gen20_kshot/scripts/g20_replicate_gen15.py

Writes results/gen15_replication/{board.csv, closure.csv, pairs.parquet}; touches nothing in gen15.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

G20 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G20))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen20k import bootstrap  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k import design as DS  # noqa: E402
from gen20k import manifest as MF  # noqa: E402
from gen15 import arms as A15  # noqa: E402
from gen15 import fewshot as FS  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402


def main() -> int:
    out = bootstrap.RESULTS / "gen15_replication"
    with MF.Run("g20_replicate_gen15", out, args={"label": "gen15_replication"}) as run:
        bench = D.load_bench()
        t0 = time.time()
        r = FS.evaluate(bench, {"G14": A15.g14, "FLAT": A15.flat}, "BP", ks=(0, 1, 2, 3), hows=("widest", "dopt", "random"),
                        cov_kind="empirical", mask_publication=True, verbose=False)
        pe = per_extractant(r.pairs, r.modes)
        bd = summarise(pe, r.pairs, r.modes)
        bd.insert(0, "design", "BP")
        bd.to_csv(out / "board.csv", index=False)
        r.pairs.to_parquet(out / "pairs.parquet", index=False)
        anchors = {m: float(bd.set_index("arm").loc[m, "macro_mae_extractant"]) for m in ("G14@doptk3", "NAIVE_LINE@doptk3", "G14@widestk1", "G14@doptk0", "FLAT@doptk0")}
        run.record("anchors", anchors)
        run.log(f"gen15 board replicated in {time.time() - t0:.0f}s: {anchors}")
        # closure on gen15's scored rows: which scored pairs are arithmetically determined by the
        # k=3 supports of each strategy (gen15 excluded the union of supports from scoring but had no closure rule)
        rows = []
        yidx = {str(c): i for i, c in enumerate(bench.frame.cell_id)}
        L = list(D.LANTHANIDES)
        for (seed, cid), g in r.pairs.groupby(["split_seed", "cell_id"]):
            ci = yidx[cid]
            cv = D.cell_view(bench, ci)
            metals = cv.metals
            for how in ("widest", "dopt", "random"):
                # reconstruct the k=3 support set exactly as gen15 did for this cell
                pairs = [(int(a), int(b)) for i, a in enumerate(metals) for b in metals[i + 1:]]
                if how == "widest":
                    sup = sorted(pairs, key=lambda p: (-(D.Z_VEC[p[1]] - D.Z_VEC[p[0]]), p[0]))[:3]
                elif how == "random":
                    rng = np.random.default_rng(abs(hash((int(ci), int(seed)))) % (2 ** 32))
                    idx = rng.choice(len(pairs), size=min(3, len(pairs)), replace=False)
                    sup = [pairs[i] for i in idx]
                else:
                    continue  # the D-optimal set depends on gen15's covariance and is not reconstructed here
                scored = [(L.index(a), L.index(b)) for a, b in zip(g.A, g.B)]
                closed = DS.closed_pairs(metals, sup, scored)
                rows.append({"split_seed": seed, "cell_id": cid, "how": how, "n_scored": len(scored), "n_closed": len(closed),
                             "n_adjacent_scored": int(sum(D.is_strict_adjacent(a, b) for a, b in scored)),
                             "n_adjacent_closed": int(sum(D.is_strict_adjacent(a, b) for a, b in closed))})
        cl = pd.DataFrame(rows)
        cl.to_csv(out / "closure.csv", index=False)
        summary = cl.groupby("how")[["n_scored", "n_closed", "n_adjacent_scored", "n_adjacent_closed"]].sum()
        summary["closed_fraction"] = summary.n_closed / summary.n_scored
        summary["adjacent_closed_fraction"] = summary.n_adjacent_closed / summary.n_adjacent_scored
        summary.to_csv(out / "closure_summary.csv")
        run.record("closure", summary.to_dict())
        run.log(f"closure on gen15's scored rows: {summary.to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

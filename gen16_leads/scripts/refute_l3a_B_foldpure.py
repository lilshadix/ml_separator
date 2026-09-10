"""Eighth stage of the lens-B refutation of CLAIM L3A: candidate sets that are simultaneously new.

The lead's task pools, for one split seed, the held-out predictions of ALL five folds.  So
candidate X was predicted by a model whose training set contained candidate Y, and vice versa.
A chemist facing "a set of candidate extractants" faces a set that is new *as a set*.  This
rebuilds L3a restricted to one fold at a time -- every candidate in the task was held out by the
same model, none of them informed any other's prediction -- and reports the saving, its
chemotype-blocked interval, the within-task permutation null and the exact sign-call ceiling.

Runs `valuebench.run_arms` once per design (discovery seeds, never `seeds=`) because the lead's
cached pair tables dropped the fold column.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen16 import l3_decision as L               # noqa: E402
from gen14.dirbench import load                  # noqa: E402
from gen15 import arms as A                      # noqa: E402
from gen15 import valuebench as V                # noqa: E402
from refute_l3a_B import OUT, block_boot         # noqa: E402
from refute_l3a_B_withinpub import grouped_tasks, evaluate   # noqa: E402
from refute_l3a_B_ceiling import best_sign_call  # noqa: E402


def main() -> int:
    t0 = time.time()
    bench = load()
    chem_map = bench.frame.drop_duplicates("extractant").set_index("extractant")["chemotype"].astype(str)
    ext = sorted(chem_map.index)
    chem = chem_map.reindex(ext).to_numpy()
    rows, notes = [], {}
    for d in L.DESIGNS:
        tab = V.run_arms(bench, {"G14": A.g14}, d, verbose=False)
        if "fold" not in tab.columns:
            notes["fold_column_missing"] = sorted(tab.columns)
            print("no fold column:", sorted(tab.columns))
            return 1
        tab = tab.merge(bench.frame[["cell_id", "extractant", "chemotype"]], on="cell_id", how="left") \
            if "extractant" not in tab.columns else tab
        # pooled reference, rebuilt from this table (must reproduce the lead exactly)
        T0 = L.build_tasks(tab, d, ["G14"])
        pooled = evaluate(T0, np.asarray(T0.chem_of))
        rows.append(dict(design=d, regime="pooled_rebuilt", **pooled))

        T, groups, n_before, n_drop = grouped_tasks(tab, d, "fold", ext, chem, L.MIN_EXT)
        res = evaluate(T, chem)
        ceil = float(np.atleast_1d(L.seed_macro(
            np.array([best_sign_call(o) for o in T.obs]), T.seeds))[0]) if T.n else np.nan
        rows.append(dict(design=d, regime="fold_pure", n_groups=int(len(set(groups))),
                         n_task_slots_before=n_before, n_dropped=n_drop, ceiling=ceil,
                         g14_over_ceiling=res["point"] / ceil if ceil else np.nan, **res))
        print(f"[{d}] t={time.time()-t0:.0f}s  pooled {pooled['point']:.4f}  "
              f"fold-pure {res['point']:.4f} on {res['n_tasks']} tasks", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "refute_l3a_B_foldpure.csv", index=False)
    (OUT / "refute_l3a_B_foldpure_notes.json").write_text(json.dumps(notes, indent=2, default=str),
                                                          encoding="utf-8")
    with pd.option_context("display.width", 280, "display.max_columns", 40):
        print(D.round(6).to_string(index=False))
    print(f"total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

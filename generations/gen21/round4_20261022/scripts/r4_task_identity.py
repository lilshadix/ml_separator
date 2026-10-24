#!/usr/bin/env python
"""Task construction check (CLAUDE_CODE_PROMPT section 3): recompute the primary task from the dev fold caches
and compare it with round 3's sealed inputs and stored B0 rows.

Per design and seed: C3 held-out cells (k_max >= 3), strict-adjacent queries (|dZ| = 1, lighter metal first),
the DOPT_Q read set at k = 3 (four metals), at least one hidden end per query, no query with both ends read,
every cell exactly once per seed.  Row identity (seed, fold, cell, A, B, read set) and the target are compared
with round 3's B0 predictions; input digests with round 3's manifest.  No prediction is compared with a target.

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_task_identity.py
writes results/task_identity.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4.leakage import isolation_rows  # noqa: E402
from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES, Z_VEC  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402

KEY = ["split_seed", "fold", "cell_id", "A", "B", "read_set"]


def rows_for(bench, design, iso: list):
    out = []
    for f in folds_for(bench.frame, design, seeds=list(paths.DEV_SEEDS)):
        rec = C.load(f.design, f.seed, f.fold)
        E.assert_test_isolation(bench, rec)
        iso += isolation_rows(bench, rec)
        for cell in rec["outer"]["cells"]:
            if cell["k_max"] < 3:
                continue
            seq = cell["seqs"]["DOPT_Q"]
            obs = E.observe(bench.Y[cell["ci"]], cell, seq, 3)
            read = set(obs.read)
            assert len(read) == 4, (cell["cell_id"], read)
            for a, b in cell["query"]:
                assert Z_VEC[b] - Z_VEC[a] == 1, (cell["cell_id"], a, b)
                assert not (a in read and b in read)
                assert np.isfinite(bench.Y[cell["ci"]][a]) and np.isfinite(bench.Y[cell["ci"]][b])
                out.append({"split_seed": f.seed, "fold": f.fold, "cell_id": cell["cell_id"], "A": LANTHANIDES[a],
                            "B": LANTHANIDES[b], "read_set": ",".join(LANTHANIDES[m] for m in sorted(read)),
                            "y": float(bench.Y[cell["ci"]][a] - bench.Y[cell["ci"]][b]),
                            "one_end_read": (a in read) or (b in read), "chemotype": cell["chemotype"],
                            "extractant": cell["extractant"], "publication": cell["publication"]})
    return pd.DataFrame(out)


def digest(t: pd.DataFrame) -> str:
    s = t.sort_values(KEY)[KEY + ["y"]].to_csv(index=False, lineterminator="\n", float_format="%.12g")
    return hashlib.sha256(s.encode()).hexdigest()


def main() -> int:
    import r4_seal
    bench = D.load_bench()
    res = {"expected": {"cells_per_seed": 229, "queries_per_seed": 1802}, "designs": {}}
    live_inputs = r4_seal.input_digests(bench)
    r3_inputs = json.loads((paths.R3DIR / "manifests" / "inputs_sha256.json").read_text())
    res["inputs_equal_round3"] = {k: live_inputs[k] == r3_inputs[k] for k in ("cohort_manifest_sha256", "logD_matrix_sha256",
                                                                                "n_cells", "folds", "cache_content")}
    ok = all(res["inputs_equal_round3"].values())
    for d in ("BP", "BPG"):
        iso = []
        t = rows_for(bench, d, iso)
        iso = pd.DataFrame(iso)
        (paths.RES / "leakage").mkdir(parents=True, exist_ok=True)
        iso.to_csv(paths.RES / "leakage" / f"isolation_{d}.csv", index=False)
        per = t.groupby("split_seed").agg(cells=("cell_id", "nunique"), queries=("cell_id", "size"),
                                          one_end_read=("one_end_read", "sum"))
        dup = int(t.groupby(["split_seed", "cell_id"]).fold.nunique().gt(1).sum())
        r3 = pd.read_parquet(paths.R3DIR / "predictions" / f"B0_{d}.parquet")
        r3 = r3[(r3.k == 3) & (r3.k_max >= 3)]
        same_rows = digest(t) == digest(r3)
        m = t.merge(r3[KEY + ["y"]], on=KEY, how="outer", suffixes=("", "_r3"), indicator=True)
        res["designs"][d] = {"per_seed": per.reset_index().to_dict(orient="records"),
                             "cells_in_more_than_one_fold_of_a_seed": dup, "row_identity_sha256": digest(t),
                             "equal_to_round3_B0_rows": bool(same_rows), "only_live": int((m._merge == "left_only").sum()),
                             "only_round3": int((m._merge == "right_only").sum()),
                             "max_abs_target_diff": float((m.y - m.y_r3).abs().max()),
                             "n_chemotypes": int(t.chemotype.nunique()), "n_extractants": int(t.extractant.nunique()),
                             "n_publications": int(t.publication.nunique()),
                             "isolation": {"rows": int(len(iso)), "all_required_zero": bool(iso.required_zero_ok.all()),
                                           "outer_folds_crossing_a_copy_group": int(((iso.level == "outer") & (iso.overlap_copy_group > 0)).sum()),
                                           "inner_blocks_crossing_a_copy_group": int(((iso.level == "inner") & (iso.overlap_copy_group > 0)).sum()),
                                           "file": f"results/leakage/isolation_{d}.csv"}}
        ok &= bool(iso.required_zero_ok.all())
        ok &= bool(same_rows) and dup == 0 and bool((per.cells == 229).all()) and bool((per.queries == 1802).all())
    res["passed"] = bool(ok)
    paths.RES.mkdir(parents=True, exist_ok=True)
    (paths.RES / "task_identity.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "designs"}, indent=1))
    for d, v in res["designs"].items():
        print(d, {k: v[k] for k in v if k != "per_seed"}, [(r["cells"], r["queries"]) for r in v["per_seed"]])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

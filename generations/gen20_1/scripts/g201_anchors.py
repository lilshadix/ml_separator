"""gen20.1 stage S7 (ANCHORS): reproduce gen20's k = 0 anchors on COH-SF-C3 under design BP before any gen20.1
outcome.

IMPLEMENTATION_PLAN.md §9 S7 and PRE_REGISTRATION.md §4.4 "Anchor gate".  Sealed gate first, guard installed, the
fold gates (a)-(e) re-asserted.  For every BP (seed, fold): the gen201 references (``baselines.gen20_ref_curves``:
``gen20k.cov.make_ctx(bench, fold, train_index, test cells)`` + ``gen20k.arms.standard_arms()``; FLAT = 0) predict
the COH-SF-C3 pairs of the fold's test cells (pair = c[a] - c[b]).  Per seed: pooled R2 (MET-06 form), pooled MAE and
the extractant-macro MAE, each seed's statistic computed on that seed's 1,802 rows, headline = mean over the five
seeds (gen20k inference.pooled_levels / report.levels).  Compared with the full-precision seed-mean columns R2, MAE
and MAE_macro_extractant of generations/gen20_kshot/results/discovery/levels_BP_metal_C3_DOPT_Q.csv rows 6, 18, 30,
62 (FLAT, G14, MEAN_CURVE, NN_TANIMOTO @ZERO|k0):

  |delta| <= 1e-9 "reproduced"; <= 1e-6 "reproduced (float32 storage or platform drift)"; beyond -> STOP.
  Per-seed values vs the csv's 5-decimal per-seed strings at 6e-6 (beyond -> STOP).  n_rows (1,802) and n_cells
  (229) per seed exact; Kish (chemotypes, pairs) 3.8511214634053776 within 1e-12.

gen20 cast predictions to float32 before scoring (runner.py:335-337); the float32-cast statistics are printed beside
the float64 ones (the gate uses float64).  ``--run-fold`` (default on) additionally runs ``gen20k.runner.run_fold`` in
memory (arms FLAT/G14/MEAN_CURVE/NN_TANIMOTO, metal model, k = 0, light policies; NNC never instantiated; imports
torch, allowed at S7 only) and asserts that the gen201 references equal its @ZERO columns on the DOPT_Q k = 0 query
rows (max |delta| <= 1e-12).  Output: results/anchors/anchors_g201.json and manifests/g201_anchors__<label>.json.

Usage (repo root):  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B generations/gen20_1/scripts/g201_anchors.py [--no-run-fold]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
G201 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G201))

from gen201 import bootstrap  # noqa: E402
from gen201 import guard  # noqa: E402
from gen201.seal import require_sealed  # noqa: E402

guard.install()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen201 import baselines as BL  # noqa: E402
from gen201 import data as D  # noqa: E402
from gen201 import fitting as FI  # noqa: E402
from gen201 import folds as FO  # noqa: E402
from gen201.run import Run  # noqa: E402

LEVELS_CSV = bootstrap.G20 / "results" / "discovery" / "levels_BP_metal_C3_DOPT_Q.csv"
CSV_ROWS = {"FLAT": 6, "G14": 18, "MEAN_CURVE": 30, "NN_TANIMOTO": 62}      # 1-based file lines (header = line 1)
COLUMNS = {"FLAT": "FLAT", "G14": "G14_ZERO", "MEAN_CURVE": "MEAN_CURVE_REF", "NN_TANIMOTO": "NN_TANIMOTO_REF"}
TOL_REPRODUCED = 1e-9
TOL_DRIFT = 1e-6
TOL_PER_SEED = 6e-6
TOL_KISH = 1e-12
EXPECTED_N_ROWS = 1802
EXPECTED_N_CELLS = 229
EXPECTED_KISH = 3.8511214634053776


# --------------------------------------------------------------------------------------
# statistics (pure; unit-tested on synthetic tables only)
# --------------------------------------------------------------------------------------
def seed_statistics(t: pd.DataFrame, col: str) -> dict:
    """gen20's per-seed statistics of one arm on one seed's rows: pooled R2 (SST about the rows' own mean), pooled
    MAE, extractant-macro MAE (mean over extractants of their pair-pooled MAE)."""
    y = t["y"].to_numpy(dtype=float)
    p = t[col].to_numpy(dtype=float)
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise AssertionError(f"{col}: non-finite rows (n_pred_finite != n_rows)")
    e = y - p
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - (e ** 2).sum() / sst) if (len(y) >= 2 and sst > 0) else float("nan")
    per_ext = pd.Series(np.abs(e)).groupby(t["extractant"].astype(str).to_numpy()).mean()
    return {"R2": r2, "MAE": float(np.abs(e).mean()), "MAE_macro_extractant": float(per_ext.mean()),
            "n_rows": int(len(t)), "n_cells": int(t["ci"].nunique())}


def parse_per_seed(s: str) -> dict[int, float]:
    return {int(k): float(v) for k, v in (part.split(":") for part in str(s).split(";"))}


def compare(name: str, got: dict, ref: pd.Series) -> dict:
    """Gate one anchor: seed means (R2, MAE, MAE_macro_extractant) and per-seed strings (R2, MAE)."""
    out = {"anchor": name, "checks": [], "status": "reproduced"}
    worst = 0.0
    for stat in ("R2", "MAE", "MAE_macro_extractant"):
        d = abs(got["seed_mean"][stat] - float(ref[stat]))
        worst = max(worst, d)
        out["checks"].append({"what": f"seed mean {stat}", "gen201": got["seed_mean"][stat], "gen20": float(ref[stat]),
                              "abs_delta": d})
    if worst > TOL_DRIFT:
        out["status"] = "STOP"
    elif worst > TOL_REPRODUCED:
        out["status"] = "reproduced (float32 storage or platform drift)"
    for stat in ("R2", "MAE"):
        ps = parse_per_seed(ref[f"{stat}_per_seed"])
        for s, v in ps.items():
            d = abs(got["per_seed"][s][stat] - v)
            ok = d <= TOL_PER_SEED
            out["checks"].append({"what": f"seed {s} {stat}", "gen201": got["per_seed"][s][stat], "gen20_5dp": v,
                                  "abs_delta": d, "pass": ok})
            if not ok:
                out["status"] = "STOP"
    for s, st in got["per_seed"].items():
        if st["n_rows"] != EXPECTED_N_ROWS or st["n_cells"] != EXPECTED_N_CELLS:
            out["status"] = "STOP"
            out["checks"].append({"what": f"seed {s} n_rows/n_cells", "gen201": [st["n_rows"], st["n_cells"]],
                                  "expected": [EXPECTED_N_ROWS, EXPECTED_N_CELLS], "pass": False})
    out["max_abs_delta_seed_mean"] = worst
    return out


def kish_pairs(t: pd.DataFrame) -> float:
    c = t.groupby("chemotype").size().to_numpy(dtype=float)
    return float(c.sum() ** 2 / (c ** 2).sum())


def anchor_table(bench, folds_bp, ctx: FI.UnitContext, man: pd.DataFrame) -> pd.DataFrame:
    """COH-SF-C3 rows of every BP unit with y and the four reference predictions (float64)."""
    Y = np.asarray(bench.Y, dtype=float)
    parts = []
    for f in folds_bp:
        rows = man[(man["design"] == "BP") & (man["split_seed"] == f.seed) & (man["fold"] == f.fold)]
        if rows.empty:
            continue
        rows = rows.reset_index(drop=True)
        cells = np.unique(rows["ci"].to_numpy(dtype=int))
        if not set(cells) <= set(np.asarray(f.test_index).tolist()):
            raise AssertionError("anchor rows outside the fold's test cells")
        refs = BL.gen20_ref_curves(bench, f, f.train_index, cells, names=("FLAT", "G14", "MEAN_CURVE", "NN_TANIMOTO"))
        loc = {int(c): i for i, c in enumerate(cells)}
        r = np.array([loc[int(c)] for c in rows["ci"]], dtype=int)
        ia, ib = rows["ia"].to_numpy(dtype=int), rows["ib"].to_numpy(dtype=int)
        t = rows[["split_seed", "fold", "ci", "cell_id", "extractant", "chemotype", "publication_id", "a", "b", "ia", "ib"]].copy()
        t["y"] = Y[t["ci"].to_numpy(dtype=int), ia] - Y[t["ci"].to_numpy(dtype=int), ib]
        for n, col in COLUMNS.items():
            t[col] = BL.pair_diff(refs[n], r, ia, ib)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def run_fold_crosscheck(bench, folds_bp, table: pd.DataFrame, log) -> dict:
    """gen20k.runner.run_fold in memory vs the gen201 references (predictions only)."""
    import gen20k.runner as R                              # imports torch (S7 only); the guard patches run/NNC
    guard.install()
    cfg = R.RunConfig(label="g201_anchor_crosscheck", designs=("BP",), models=("metal",), k_grid=(0,),
                      arms=("FLAT", "G14", "MEAN_CURVE", "NN_TANIMOTO"), full_policy_designs=(),
                      remaining_regime_designs=())
    worst, n = 0.0, 0
    for f in folds_bp:
        t, _ = R.run_fold(bench, f, cfg, None)
        q = t[(t["model"] == "metal") & (t["policy"] == "DOPT_Q") & (t["k"] == 0) & (t["regime"] == "Q") & (t["k_max"] >= 3)]
        mine = table[(table["split_seed"] == f.seed) & (table["fold"] == f.fold)]
        j = mine.merge(q, left_on=["cell_id", "a", "b"], right_on=["cell_id", "A", "B"], how="inner", validate="one_to_one")
        if len(j) != len(mine) or len(j) != len(q):
            raise AssertionError(f"run_fold rows differ from COH-SF-C3 at seed {f.seed} fold {f.fold}: "
                                 f"{len(mine)} vs {len(q)} ({len(j)} joined)")
        for n_, col in COLUMNS.items():
            d = float(np.max(np.abs(j[col].to_numpy(float) - j[f"{n_}@ZERO"].to_numpy(float)))) if len(j) else 0.0
            worst = max(worst, d)
        n += len(j)
        log(f"run_fold cross-check seed {f.seed} fold {f.fold}: {len(j)} rows, max |delta| so far {worst:.3g}")
    return {"rows_compared": int(n), "max_abs_delta": worst, "pass": bool(worst <= 1e-12)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", default="S7")
    ap.add_argument("--no-run-fold", action="store_true", help="skip the in-memory gen20k.runner.run_fold cross-check")
    ap.add_argument("--no-subprocess", action="store_true", help="skip the PYTHONHASHSEED subprocess fold gate")
    ap.add_argument("--out", default=None, help="output directory (default generations/gen20_1/results/anchors)")
    ap.add_argument("--manifest-dir", default=None, help="run-manifest directory (default generations/gen20_1/manifests)")
    args = ap.parse_args(argv)
    seal_state = require_sealed(bootstrap.PREREG)
    before = guard.snapshot()
    out_dir = bootstrap.assert_under_g201(args.out if args.out else bootstrap.RESULTS / "anchors")
    status = 1
    with Run("g201_anchors", label=args.label, out_dir=out_dir, args=vars(args), seeds=list(FO.SEEDS),
             manifest_dir=args.manifest_dir) as run:
        run.record("preregistration", seal_state)
        bench = D.load(check=True)
        run.record("cohort", D.LOAD_RECORD)
        F = FO.build(bench)
        run.record("fold_gates", FI.fold_gates(bench, F, subprocess_check=not args.no_subprocess))
        ctx = FI.make_context(bench)
        man = pd.read_parquet(bootstrap.MANIFESTS / "row_manifests" / "COH-SF-C3.parquet")
        table = anchor_table(bench, F["BP"], ctx, man)
        ref = pd.read_csv(LEVELS_CSV)
        result = {"levels_csv": str(LEVELS_CSV.relative_to(bootstrap.ROOT)).replace("\\", "/"), "anchors": {},
                  "float32_cast": {}}
        kis = {int(s): kish_pairs(g) for s, g in table.groupby("split_seed")}
        result["kish_pairs_per_seed"] = kis
        result["kish_pass"] = bool(all(abs(v - EXPECTED_KISH) <= TOL_KISH for v in kis.values()))
        stop = not result["kish_pass"]
        for name, line in CSV_ROWS.items():
            row = ref.iloc[line - 2]
            if not str(row["arm"]).startswith(f"{name}@ZERO"):
                raise AssertionError(f"levels csv line {line} is {row['arm']!r}, expected {name}@ZERO|k0")
            col = COLUMNS[name]
            per = {int(s): seed_statistics(g, col) for s, g in table.groupby("split_seed")}
            got = {"per_seed": per, "seed_mean": {k: float(np.mean([per[s][k] for s in per]))
                                                  for k in ("R2", "MAE", "MAE_macro_extractant")}}
            cmpd = compare(name, got, row)
            result["anchors"][name] = {**cmpd, "seed_mean": got["seed_mean"], "per_seed": per}
            t32 = table.assign(**{col: table[col].astype(np.float32).astype(np.float64)})
            per32 = {int(s): seed_statistics(g, col) for s, g in t32.groupby("split_seed")}
            result["float32_cast"][name] = {k: float(np.mean([per32[s][k] for s in per32]))
                                            for k in ("R2", "MAE", "MAE_macro_extractant")}
            stop |= cmpd["status"] == "STOP"
            run.log(f"{name}@ZERO: {cmpd['status']} (max |delta| seed mean {cmpd['max_abs_delta_seed_mean']:.3g})")
        if not args.no_run_fold:
            result["run_fold_crosscheck"] = run_fold_crosscheck(bench, F["BP"], table, run.log)
            stop |= not result["run_fold_crosscheck"]["pass"]
        result["status"] = "STOP" if stop else "PASS"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(bootstrap.assert_under_g201(out_dir / "anchors_g201.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(FI._jsonable(result), fh, indent=1, sort_keys=True)
        run.record("anchors", {k: v["status"] for k, v in result["anchors"].items()})
        run.record("status", result["status"])
        guard.assert_unchanged(before)
        run.record("frozen_trees_unchanged", True)
        status = 0 if not stop else 1
    return status


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Run one gen21 attempt on the dev seeds (BP and the copy-group-strict BPG) and write its tables.

    .venv/Scripts/python.exe generations/gen21/scripts/g21_attempt.py --attempt 1

Outputs: predictions/attempt<N>_<design>.parquet (row level), results/attempt<N>/ (levels with
chemotype-bootstrap intervals and per-seed values, per pair type, log D of unread metals, fitted
parameters per fold, leakage checks), results/attempt<N>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from g21 import attempts as AT  # noqa: E402
from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from g21 import paths  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k import report as RP  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402


def run_design(bench, spec, design, seeds, log):
    """Rows for every fold.  Also returns the fitted objects per (seed, fold) so the intervention
    test reuses exactly the fitted parameters that produced the rows."""
    rows, fits, logd, fitted_by_fold, dups = [], [], [], {}, []
    for f in folds_for(bench.frame, design, seeds=list(seeds)):
        rec = C.load(f.design, f.seed, f.fold)
        iso = E.assert_test_isolation(bench, rec)
        blk = dict(rec["outer"])
        fitted = spec.fit(bench, rec) if spec.fit else None
        fitted_by_fold[(f.seed, f.fold)] = fitted
        dups += [{"split_seed": f.seed, "cell_id": c} for c in E.duplicate_sibling_cells(bench, rec["outer"])]
        if fitted is not None:
            blk["fitted"] = fitted
            row = {"design": design, "split_seed": f.seed, "fold": f.fold}
            for k_, v in fitted.items():
                if isinstance(v, (int, float, str)):
                    row[k_] = v
                elif hasattr(v, "params"):
                    for pk, pv in v.params().items():
                        row[f"{k_}.{pk}"] = json.dumps(pv) if isinstance(pv, list) else pv
            fits.append(row)
        rows += E.block_rows(bench, rec, blk, spec.methods, policy=spec.policy, ks=E.KS,
                             logd_arms=spec.logd_arms, logd_sink=logd)
        log(f"  {design} {f.seed} fold {f.fold}: {len(rows)} rows; isolation {iso}")
    t = pd.DataFrame(rows)
    E.assert_no_known_pair(t)
    # evaluation-only data-quality flag (uses labels; never seen by a method): the cell has a sibling in its
    # series whose log D curve is an exact offset copy of its own on the shared metals
    d = pd.DataFrame(dups, columns=["split_seed", "cell_id"]).drop_duplicates()
    t["dup_sibling"] = pd.MultiIndex.from_frame(t[["split_seed", "cell_id"]]).isin(pd.MultiIndex.from_frame(d)) if len(d) else False
    return t, pd.DataFrame(fits), pd.DataFrame(logd), fitted_by_fold


def intervention(bench, spec, design, seeds, fitted_by_fold, log) -> dict:
    """For every dev seed, fold and budget k: the log D of every held-out metal not read at k (in
    every held-out cell, so sibling reads are covered too) is replaced by random numbers with reads
    and masks unchanged; every prediction must be bit-identical (NaN pattern included).  The fitted
    parameters are the ones that produced the reported rows; that fits never see held-out labels is
    checked separately by the poisoning tests in tests/test_g21.py."""
    rng = np.random.default_rng(20261018)
    worst, n = 0.0, 0
    for f in folds_for(bench.frame, design, seeds=list(seeds)):
        rec = C.load(f.design, f.seed, f.fold)
        blk = dict(rec["outer"])
        if spec.fit:
            blk["fitted"] = fitted_by_fold[(f.seed, f.fold)]
        for k in E.KS:
            Ys = E.scramble_block(bench.Y, blk, spec.policy, k, rng)
            r1 = pd.DataFrame(E.block_rows(bench, rec, blk, spec.methods, policy=spec.policy, ks=(k,)))
            r2 = pd.DataFrame(E.block_rows(bench, rec, blk, spec.methods, policy=spec.policy, ks=(k,), Y=Ys))
            if not len(r1):
                continue
            a1, a2 = r1[list(spec.methods)].to_numpy(float), r2[list(spec.methods)].to_numpy(float)
            if not np.array_equal(np.isnan(a1), np.isnan(a2)):
                return {"design": design, "n_predictions": n, "max_abs_change": float("inf"), "passed": False,
                        "reason": f"NaN pattern changed at seed {f.seed} fold {f.fold} k {k}"}
            ok = ~np.isnan(a1)
            if ok.any():
                worst = max(worst, float(np.max(np.abs(a1[ok] - a2[ok]))))
            n += int(ok.sum())
    log(f"  intervention {design} seeds {list(seeds)}: {n} predictions, max |change| {worst:.3g}")
    return {"design": design, "split_seeds": list(seeds), "n_predictions": n, "max_abs_change": worst, "passed": worst == 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", required=True)
    ap.add_argument("--designs", default="BP,BPG")
    ap.add_argument("--replicates", type=int, default=2000)
    a = ap.parse_args()
    bench = D.load_bench()
    spec = AT.get(a.attempt, bench)
    out = paths.RESULTS / f"attempt{a.attempt}"
    if a.designs != "BP,BPG":
        out = out / ("designs_" + a.designs.replace(",", "_"))
    out.mkdir(parents=True, exist_ok=True)
    paths.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(s, flush=True)  # noqa: E731
    summary = {"attempt": a.attempt, "hypothesis": spec.hypothesis, "contract": spec.contract, "policy": spec.policy
               if isinstance(spec.policy, str) else spec.policy_name, "methods": list(spec.methods), "designs": {}}
    summary["intervention"] = []
    for design in a.designs.split(","):
        t, fits, ldrows, fitted_by_fold = run_design(bench, spec, design, paths.DEV_SEEDS, log)
        t.to_parquet(paths.PREDICTIONS / f"attempt{a.attempt}_{design}.parquet", index=False, compression="zstd")
        if len(fits):
            fits.to_csv(out / f"fitted_{design}.csv", index=False)
        arms = list(spec.methods)
        pol = t["policy"].iloc[0]
        if "sib_pair_read" in t and spec.contrasts:
            sub = t[(t.k_max >= 3) & (t.k == 3)]
            strict = E.quick(sub[~sub.sib_pair_read], list(spec.methods), K=3, k=3)
            strict.insert(0, "subset", "no_sibling_reads_both_ends"); strict.insert(0, "design", design)
            ser = E.quick(sub[sub.n_siblings > 0], list(spec.methods), K=3, k=3)
            ser.insert(0, "subset", "cells_with_siblings"); ser.insert(0, "design", design)
            sgl = E.quick(sub[sub.n_siblings == 0], list(spec.methods), K=3, k=3)
            sgl.insert(0, "subset", "singletons"); sgl.insert(0, "design", design)
            nod = E.quick(sub[~sub.dup_sibling], list(spec.methods), K=3, k=3)
            nod.insert(0, "subset", "no_duplicate_sibling"); nod.insert(0, "design", design)
            dup = E.quick(sub[sub.dup_sibling], list(spec.methods), K=3, k=3) if sub.dup_sibling.any() else pd.DataFrame()
            if len(dup):
                dup.insert(0, "subset", "duplicate_sibling_only"); dup.insert(0, "design", design)
            pd.concat([strict, ser, sgl, nod, dup]).to_csv(out / f"subsets_{design}_C3_k3.csv", index=False)
        lv3, w3 = E.cohort_levels(t, arms, K=3, ks=[0, 1, 2, 3], replicates=a.replicates, policy=pol)
        lv3.insert(0, "cohort", "C3"); lv3.insert(0, "design", design)
        lv6, _ = E.cohort_levels(t, arms, K=6, ks=[0, 3, 5, 6], replicates=a.replicates, policy=pol)
        lv6.insert(0, "cohort", "C6"); lv6.insert(0, "design", design)
        lv = pd.concat([lv3, lv6])
        lv.to_csv(out / f"levels_{design}.csv", index=False)
        pp = []
        for arm in spec.per_pair_arms:
            x = RP.per_pair(w3, f"{arm}|k3"); x.insert(0, "arm", arm); pp.append(x)
        pd.concat(pp).to_csv(out / f"per_pair_{design}_C3_k3.csv", index=False)
        if spec.contrasts:
            from gen20k import inference as INF
            comps = {f"{c} - {r} |k{k}": (f"{r}|k{k}", f"{c}|k{k}") for c, r in spec.contrasts for k in (1, 2, 3)}
            ct = INF.pooled_contrasts(w3, comps, replicates=a.replicates)
            ct.insert(0, "design", design)
            ct.to_csv(out / f"contrasts_{design}_C3.csv", index=False)
            per_seed = []
            for c, r in spec.contrasts:
                for k in (1, 2, 3):
                    for s_, g in w3.groupby("split_seed"):
                        y = g["y"].to_numpy(float)
                        sst = ((y - y.mean()) ** 2).sum()
                        ec, er = y - g[f"{c}|k{k}"].to_numpy(float), y - g[f"{r}|k{k}"].to_numpy(float)
                        per_seed.append({"design": design, "candidate": c, "reference": r, "k": k, "split_seed": s_,
                                         "dR2": float(((er ** 2).sum() - (ec ** 2).sum()) / sst),
                                         "dMAE_ref_minus_cand": float(np.abs(er).mean() - np.abs(ec).mean())})
            pd.DataFrame(per_seed).to_csv(out / f"contrasts_per_seed_{design}_C3.csv", index=False)
        ld = AT.logd_table(ldrows) if len(ldrows) else pd.DataFrame()
        if len(ld):
            ld.to_csv(out / f"logd_unread_{design}.csv", index=False)
        keep = ["arm", "R2", "R2_ci_low", "R2_ci_high", "R2_per_seed", "MAE", "MAE_ci_low", "MAE_ci_high", "RMSE",
                "MAE_macro_extractant", "n_rows", "n_cells", "n_extractants", "n_chemotypes"]
        summary["designs"][design] = {"levels_C3": lv3[keep].to_dict("records"), "levels_C6": lv6[keep].to_dict("records"),
                                      "logd_unread": ld.to_dict("records") if len(ld) else None,
                                      "n_rows_total": int(len(t)),
                                      "dup_sibling_rows_C3_k3_per_seed": float(t[(t.k == 3) & (t.k_max >= 3) & t.dup_sibling].shape[0] / len(paths.DEV_SEEDS))}
        summary["intervention"].append(intervention(bench, spec, design, paths.DEV_SEEDS, fitted_by_fold, log))
    if not all(x["passed"] for x in summary["intervention"]):
        raise SystemExit("INTERVENTION TEST FAILED")
    summary["seconds"] = time.time() - t0
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    log(f"done in {summary['seconds']:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

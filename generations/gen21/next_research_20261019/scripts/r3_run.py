#!/usr/bin/env python
"""Run one round-3 registry configuration on one design (BP or BPG) over the five dev seeds.

    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_run.py --config R02 --design BP

Checkpoint/resume: every outer fold's rows, log D rows and fitted objects are pickled under
predictions/_parts/<ID>_<design>/; a rerun skips finished folds.  Outputs:
  predictions/<ID>_<design>.parquet       row-level OOF predictions (seed, fold, cell, pair, target, read flags,
                                          read set, policy, budget, data version, one column per method)
  predictions/<ID>_<design>_logd.parquet  log D of unread metals (diagnostic)
  results/<ID>/summary_<design>.csv       seed-mean metrics, all rows and declared subsets
  results/<ID>/per_pair_<design>.csv      12 pair types
  results/<ID>/contrasts_<design>.csv     paired contrasts vs PWL_R, PWL_Z, HBK@ML, RIDGE2 (identical rows/supports)
  results/<ID>/levels_<design>.csv        k ladder with chemotype-bootstrap intervals
  results/<ID>/fitted_<design>.json       fitted choices per fold (lambda, kernels, weights, ...)
  results/<ID>/intervention_<design>.json hidden-label replacement test on all seeds
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r3 import harness as H  # noqa: E402
from r3 import metrics as MT  # noqa: E402
from r3 import paths  # noqa: E402
from r3 import registry as RG  # noqa: E402
from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402


def fitted_info(fitted: dict) -> dict:
    out = {}
    for k, v in (fitted or {}).items():
        if k == "hbk":
            out["ml_kernel"] = v["ml_kernel"]
        elif k.startswith("cor:"):
            out[k] = {"lambda": v["lambda"], "inner_scores": v["inner_scores"], "inner_pwl": v["inner_pwl"],
                      "coef": [float(c) for c in v["coef"]], "n_rows": v["n_rows"]}
        elif k == "inner_lik":
            out[k] = v
        elif k.startswith("_task_"):
            out[k] = {"choice": v[0], "scores": v[1]}
        elif k == "_stack_w":
            out[k] = [float(x) for x in v]
        elif k in ("v2_note",):
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--design", required=True, choices=("BP", "BPG"))
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--skip-intervention", action="store_true", help="technical smoke only")
    ap.add_argument("--max-folds", type=int, default=None, help="technical smoke only (never a result)")
    a = ap.parse_args()
    bench = D.load_bench()
    cfg = RG.get(a.config, bench)
    Y = RG.data_matrix(cfg.version, bench)
    parts_dir = paths.PRED / "_parts" / f"{cfg.cid}_{a.design}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    out = paths.RES / cfg.cid
    out.mkdir(parents=True, exist_ok=True)
    methods = {**RG.base_methods(bench, cfg), **cfg.methods}
    t0 = time.time()
    rows, logd, fitted, info = [], [], {}, {}
    folds = folds_for(bench.frame, a.design, seeds=list(paths.DEV_SEEDS))
    if a.max_folds:
        folds = folds[: a.max_folds]
    for f in folds:
        pf = parts_dir / f"{f.seed}_{f.fold}.pkl"
        if pf.exists():
            with open(pf, "rb") as fh:
                part = pickle.load(fh)
        else:
            rec = C.load(f.design, f.seed, f.fold)
            E.assert_test_isolation(bench, rec)
            bfit = bench
            if cfg.version == "V2":
                from r3 import datav2
                rec, bfit = datav2.v2_record(rec, Y), datav2.proxy(bench, Y)
            blk = dict(rec["outer"], Y=Y, policy=cfg.policy, _rec=rec)
            blk["fitted"] = RG.fit(cfg, bfit, rec, Y)
            r_, l_ = [], []
            for cell in RG.cells_of(cfg, blk):
                r_ += H.cell_rows(rec, blk, cell, methods, policy=cfg.policy, ks=cfg.ks, Y=Y, version=cfg.version,
                                  logd_sink=l_, logd_arms=("PWL_R", "HBK@ML") + tuple(cfg.primary))
            part = {"rows": r_, "logd": l_, "fitted": blk["fitted"], "info": fitted_info(blk["fitted"])}
            with open(pf.with_suffix(".tmp"), "wb") as fh:
                pickle.dump(part, fh)
            pf.with_suffix(".tmp").replace(pf)
        rows += part["rows"]
        logd += part["logd"]
        fitted[(f.seed, f.fold)] = part["fitted"]
        info[f"{f.seed}_{f.fold}"] = part["info"]
        print(f"  {cfg.cid} {a.design} {f.seed} fold {f.fold}: {len(rows)} rows ({time.time() - t0:.0f}s)", flush=True)
    t = pd.DataFrame(rows)
    if t.empty:
        raise SystemExit("no rows")
    tag = "" if not a.max_folds else "_SMOKE"
    t.to_parquet(paths.PRED / f"{cfg.cid}_{a.design}{tag}.parquet", index=False, compression="zstd")
    pd.DataFrame(logd).to_parquet(paths.PRED / f"{cfg.cid}_{a.design}{tag}_logd.parquet", index=False, compression="zstd")
    (out / f"fitted_{a.design}{tag}.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    arms = list(methods)
    K, kmain = cfg.K, max(k for k in cfg.ks if k <= max(cfg.ks))
    kprim = 3 if 3 in cfg.ks else max(cfg.ks)
    MT.summary(t, arms, K=K, k=kprim).to_csv(out / f"summary_{a.design}{tag}.csv", index=False)
    MT.per_pair(t, arms, K=K, k=kprim).to_csv(out / f"per_pair_{a.design}{tag}.csv", index=False)
    pairs = [(c, r) for c in cfg.primary for r in ("PWL_R", "PWL_Z", "HBK@ML", "RIDGE2") if c != r] + list(cfg.contrasts)
    if pairs and not a.max_folds:
        MT.contrasts(t, pairs, K=K, k=kprim, replicates=a.replicates).to_csv(out / f"contrasts_{a.design}.csv", index=False)
        MT.levels_ci(t, arms, K=K, ks=tuple(k for k in cfg.ks if k <= K), replicates=a.replicates).to_csv(out / f"levels_{a.design}.csv", index=False)
    if not a.skip_intervention and not a.max_folds:
        tr = None
        if cfg.version == "V2":
            from r3 import datav2
            tr = lambda r: datav2.v2_record(r, Y)  # noqa: E731
        iv = H.intervention(bench, a.design, methods, fitted, policy=cfg.policy, ks=cfg.ks, Y=Y, rec_transform=tr)
        (out / f"intervention_{a.design}.json").write_text(json.dumps(iv, indent=1), encoding="utf-8")
        print("  intervention:", iv, flush=True)
        if not iv["passed"]:
            raise SystemExit("INTERVENTION FAILED")
    s = MT.summary(t, list(cfg.primary) + list(RG.SHOW_BASELINES), K=K, k=kprim, subsets=False)
    pd.set_option("display.width", 220)
    print(s[["arm", "R2", "MAE", "RMSE", "R2_min_seed", "n_rows_per_seed", "n_cells_per_seed"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"EXIT 0 ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

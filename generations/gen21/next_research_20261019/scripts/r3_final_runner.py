#!/usr/bin/env python
"""GUARDED final check of the frozen round-3 candidate on the withheld split seeds (PROTOCOL §2).

It does NOTHING unless every guard passes:
  1. --i-have-explicit-user-permission "<quote of the user's message granting the final check>" is given;
  2. manifests/freeze_manifest.json exists and its SHA-256 equals manifests/freeze_seal.txt;
  3. every code file listed in the freeze manifest still has the recorded SHA-256 (code identity), the data inputs
     (cohort manifest, log D matrix, dev fold plans) and the package versions equal the sealed ones;
  4. the seeds file's SHA-256 equals the sealed commitment (and ../manifests/final_seeds_commitment.txt), and the
     file holds 5 distinct integer seeds disjoint from the dev seeds -- all checked before RUN_ONCE is written;
  5. results/final/RUN_ONCE.json does not exist (a second run is refused); it is written, with the start
     time and the guards' digests, BEFORE any computation.
The final-seed fold caches are built in a separate directory (cache_final/), never mixed with the dev cache.

    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_final_runner.py --dry-check
    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_final_runner.py \\
        --seeds-file <withheld seeds file; path supplied by the user> --i-have-explicit-user-permission "..."
Writes results/final/, per configuration (the frozen candidate and the freeze's pre-declared secondary
configurations, which are reported descriptively) and design: final_<ID>_<design>.parquet (row-level, with read
sets), summary / per_pair / levels / contrasts CSVs, the hidden-label intervention on every final fold, and
isolation.json (fold isolation asserted on every fold).  Cross-configuration contrasts on identical rows can be
computed afterwards from the row-level files.  A crash leaves RUN_ONCE.json and writes CRASH.txt; a rerun then
needs a visible, documented decision.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3 import paths  # noqa: E402


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def live_inputs() -> dict:
    """The data digests the freeze recorded (cohort manifest, log D matrix, dev fold plans), recomputed now."""
    import numpy as np
    from gen20k import data as D
    from gen20k.splits import fold_record, folds_for
    bench = D.load_bench()
    out = {"cohort_manifest_sha256": paths.sha256_file(paths.ROOT / "generations/gen13_separation/manifests/cohort_exact.parquet"),
           "logD_matrix_sha256": hashlib.sha256(np.ascontiguousarray(np.nan_to_num(bench.Y, nan=-999.0)).tobytes()).hexdigest(),
           "n_cells": int(bench.Y.shape[0]), "folds": {}}
    for d in ("BP", "BPG"):
        fr = fold_record(bench.frame, folds_for(bench.frame, d, seeds=list(paths.DEV_SEEDS)))
        out["folds"][d] = {"seeds": list(paths.DEV_SEEDS), "n_rows": int(len(fr)),
                           "sha256": hashlib.sha256(fr.to_csv(index=False, lineterminator="\n").encode()).hexdigest()}
    return out


def live_environment() -> dict:
    import platform
    import numpy, pandas, scipy, sklearn  # noqa: E401
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def parse_seeds(text: str) -> list:
    """Accept {"final_seeds": [...]}, {"seeds": [...]} or a bare list; require 5 distinct ints, none a dev seed."""
    obj = json.loads(text)
    seeds = obj.get("final_seeds", obj.get("seeds")) if isinstance(obj, dict) else obj
    if not isinstance(seeds, list) or len(seeds) != 5 or not all(isinstance(x, int) for x in seeds):
        raise SystemExit("REFUSED: the seeds file does not hold a list of 5 integer seeds (RUN_ONCE not written)")
    if len(set(seeds)) != 5 or set(seeds) & set(paths.DEV_SEEDS):
        raise SystemExit("REFUSED: final seeds are not distinct or overlap the dev seeds (RUN_ONCE not written)")
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-file", default=None, help="required for the real run; never read by --dry-check")
    ap.add_argument("--i-have-explicit-user-permission", dest="permission", default=None)
    ap.add_argument("--dry-check", action="store_true",
                    help="check guards 2, 3, 5 and the sealed inputs / environment; never opens the seeds file")
    a = ap.parse_args()
    fm, seal = paths.MAN / "freeze_manifest.json", paths.MAN / "freeze_seal.txt"
    if not fm.exists() or not seal.exists():
        raise SystemExit("REFUSED: no sealed freeze manifest")
    if sha(fm) != seal.read_text().split()[0]:
        raise SystemExit("REFUSED: freeze manifest does not match its seal")
    freeze = json.loads(fm.read_text(encoding="utf-8"))
    for rel, digest in freeze["code_sha256"].items():
        if sha(paths.ROOT / rel) != digest:
            raise SystemExit(f"REFUSED: code changed since the freeze: {rel}")
    sealed_inputs = {k: v for k, v in freeze["inputs"].items() if k != "cache_content"}
    if live_inputs() != sealed_inputs:
        raise SystemExit("REFUSED: data inputs (cohort manifest, log D matrix or dev fold plans) differ from the freeze")
    if live_environment() != freeze["environment"]:
        raise SystemExit(f"REFUSED: package versions differ from the freeze: {live_environment()} vs {freeze['environment']}")
    once = paths.RES / "final" / "RUN_ONCE.json"
    if once.exists():
        raise SystemExit("REFUSED: the final check has already been run (results/final/RUN_ONCE.json)")
    if a.dry_check:
        print("guards 2, 3, 5, sealed inputs and environment pass (seeds file not opened)")
        return 0
    if not a.permission or len(a.permission.strip()) < 10:
        raise SystemExit("REFUSED: explicit user permission (quoted) is required")
    if not a.seeds_file:
        raise SystemExit("REFUSED: --seeds-file is required for the real run")
    digest = sha(a.seeds_file)
    sealed_commit = freeze["final_seeds_commitment"]
    if digest not in sealed_commit or digest not in (paths.G21 / "manifests" / "final_seeds_commitment.txt").read_text():
        raise SystemExit("REFUSED: seeds file does not match the committed SHA-256 (sealed and on disk)")
    seeds = parse_seeds(Path(a.seeds_file).read_text())          # validated BEFORE the single run is consumed
    once.parent.mkdir(parents=True, exist_ok=True)
    once.write_text(json.dumps({"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "permission": a.permission,
                                "freeze_seal": seal.read_text().split()[0], "seeds_sha256": digest}, indent=1),
                    encoding="utf-8")
    try:
        compute(freeze, seeds, paths.RES / "final", paths.R3 / "cache_final")
    except BaseException:                            # a crash is recorded; RUN_ONCE stays, a rerun needs a visible decision
        import traceback
        (paths.RES / "final" / "CRASH.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    print("final check written to", paths.RES / "final")
    return 0


def compute(freeze: dict, seeds, out: Path, cache_dir: Path | None, max_folds: int | None = None) -> dict:
    """The frozen candidate and the freeze's secondary configurations on the given split seeds, BP and BPG:
    row-level predictions, summaries, per-pair errors, levels, the hidden-label intervention on every fold, the
    candidate's paired contrasts against its comparators on the same reads, and every secondary configuration's
    own contrasts against PWL_R / PWL_Z / HBK@ML on its own reads (descriptive).  Fold isolation is asserted on
    every fold.  cache_dir=None reuses the dev fold cache (smoke tests on dev seeds only)."""
    import pandas as pd
    from g21 import cache as C
    from g21 import evaluate as E
    from g21 import paths as G21P
    from r3 import configs as CF, harness as H, metrics as MT, registry as RG
    from gen20k import data as D
    from gen20k.splits import folds_for
    bench = D.load_bench()
    if cache_dir is not None:
        G21P.CACHE = Path(cache_dir)                 # separate fold cache for the withheld seeds
        G21P.CACHE.mkdir(parents=True, exist_ok=True)
        CF.INNER_CACHE_DIR = Path(cache_dir) / "_inner"
    out.mkdir(parents=True, exist_ok=True)
    report = {"isolation": {}}
    for design in ("BP", "BPG"):
        folds = folds_for(bench.frame, design, seeds=list(seeds))[:max_folds]
        # fresh configuration objects per design: no policy memo is shared across designs
        cand = RG.get(freeze["config_id"], bench)
        cfgs = [cand] + [RG.get(c, bench) for c in freeze.get("secondary_configs", freeze.get("reference_configs", []))]
        for cfg in cfgs:
            assert cfg.version == "V1" and cfg.K == cand.K
            methods = {**RG.base_methods(bench, cfg), **cfg.methods}
            rows, fitted = [], {}
            for f in folds:
                cp = C.cache_path(f.design, f.seed, f.fold)
                if not cp.exists():                      # build first, then write atomically (a crash leaves no stub)
                    built = C.build_fold(bench, f)
                    with open(cp.with_suffix(".tmp"), "wb") as fh:
                        pickle.dump(built, fh)
                    cp.with_suffix(".tmp").replace(cp)
                rec = C.load(f.design, f.seed, f.fold)
                report["isolation"][f"{design}_{f.seed}_{f.fold}"] = E.assert_test_isolation(bench, rec)
                blk = dict(rec["outer"], Y=bench.Y, policy=cfg.policy, _rec=rec)
                blk["fitted"] = RG.fit(cfg, bench, rec, bench.Y)
                fitted[(f.seed, f.fold)] = blk["fitted"]
                for cell in RG.cells_of(cfg, blk):
                    rows += H.cell_rows(rec, blk, cell, methods, policy=cfg.policy, ks=cfg.ks, Y=bench.Y, version=cfg.version)
            t = pd.DataFrame(rows)
            tag = f"{cfg.cid}_{design}"
            t.to_parquet(out / f"final_{tag}.parquet", index=False)
            MT.summary(t, list(methods), K=cfg.K, k=3).to_csv(out / f"summary_{tag}.csv", index=False)
            MT.per_pair(t, list(methods), K=cfg.K, k=3).to_csv(out / f"per_pair_{tag}.csv", index=False)
            MT.levels_ci(t, list(methods), K=cfg.K, ks=(3,)).to_csv(out / f"levels_{tag}.csv", index=False)
            if cfg is cand:
                col, comps = freeze["candidate_column"], freeze["comparators"]
            else:
                col, comps = cfg.primary[0], ["PWL_R", "PWL_Z", "HBK@ML", "RIDGE2"]
            pairs = [(col, r) for r in comps if r != col]
            if pairs:
                MT.contrasts(t, pairs, K=cfg.K, k=3).to_csv(out / f"contrasts_{tag}.csv", index=False)
            iv = H.intervention(bench, design, methods, fitted, policy=cfg.policy, ks=cfg.ks, Y=bench.Y,
                                seeds=sorted({f.seed for f in folds}), fold_filter={(f.seed, f.fold) for f in folds},
                                methods_factory=lambda b, c=cfg: {**RG.base_methods(b, c), **c.methods})
            (out / f"intervention_{tag}.json").write_text(json.dumps(iv, indent=1), encoding="utf-8")
            report[tag] = iv
            print(tag, "intervention:", iv, flush=True)
    (out / "isolation.json").write_text(json.dumps(report["isolation"], indent=1, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    raise SystemExit(main())

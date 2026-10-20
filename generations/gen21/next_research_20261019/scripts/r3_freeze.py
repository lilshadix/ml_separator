#!/usr/bin/env python
"""Freeze one round-3 candidate for the final check: writes manifests/freeze_manifest.json (the whole chain:
data version, cohort/query rule, budget, policy, preprocessing, features, method and its fitted-parameter
rules, comparators, decision rule, evaluator and intervention code, code digests, dev-seed list, the
withheld-seed commitment, package versions) and manifests/freeze_seal.txt (its SHA-256).  The seeds file is NOT
opened.  Before hashing, the final runner's compute() is executed on one dev fold per design (dev fold cache, output to
a temporary directory) for the candidate and its secondary configurations; its rows must equal the stored dev
predictions bit for bit (results/leakage/runner_smoke.json), and every repository module that run imported (lazy
imports included) is hashed.

    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_freeze.py --config R02 \
        --column COR_G --secondary R14 --secondary R23 --reason "..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3 import paths  # noqa: E402


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


READS = {
    "DOPT_Q": "DOPT_Q: gen20's label-free D-optimal sequence from the outer-training covariance (the fixed reference policy)",
    "POL_ERR": ("POL_ERR: label-free; enumerates the legal (k+1)-sets of the cell (independent read set: no query pair "
                "with both ends read) and minimises a PWL error table fitted on outer-training cells (pair type, read "
                "status, Z distance to the nearest read capped at 3, extrapolation; pseudo-count 20)"),
}
PWL = "PWL_R: piecewise-linear interpolation of read log D along -radius (Shannon CN8), constant beyond the extreme reads"
RIDGE = ("fitted on masked outer-training cells (DOPT_Q nested / UNIFORM / 4 RANDOM masks, cell weight split over its "
         "masks); lambda from (0.003, 0.03, 0.3, 3, 30) x trace scale by inner task loss")
CORRECTION = {
    "COR[Dnc]": "ridge on one feature 1[b = Gd, unread] - w_Gd(b) (w_Gd = PWL weight of a read Gd), " + RIDGE,
    "COR_G": "ridge on 6 geometry features of the PWL_R residual (segment curvature, edge-slope extrapolation), " + RIDGE,
    "PWL_R": "none (interpolation only)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--column", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--secondary", action="append", default=[],
                    help="pre-declared secondary configuration(s) computed on the same final folds, reported descriptively")
    a = ap.parse_args()
    from r3 import registry as RG
    from gen20k import data as D
    bench = D.load_bench()
    cfg = RG.get(a.config, bench)
    sec = [RG.get(c, bench) for c in a.secondary]
    # run the final runner's compute() on one dev fold per design (this imports everything the final run imports,
    # lazy imports included), check its rows against the stored dev predictions, then hash the import closure
    import importlib
    import importlib.util
    import tempfile
    import numpy as np
    import pandas as pd
    spec = importlib.util.spec_from_file_location("r3_final_runner", paths.R3 / "scripts" / "r3_final_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    stub = {"config_id": cfg.cid, "candidate_column": a.column, "comparators": ["PWL_R", "PWL_Z", "HBK@ML", "RIDGE2"],
            "secondary_configs": [c.cid for c in sec]}
    smoke = {"seed": int(paths.DEV_SEEDS[0]), "fold": 0, "configs": {}}
    key = ["split_seed", "fold", "cell_id", "A", "B", "k"]
    with tempfile.TemporaryDirectory() as tmp:
        rep = runner.compute(stub, [paths.DEV_SEEDS[0]], Path(tmp), None, max_folds=1)
        for c in [cfg] + sec:
            for design in ("BP", "BPG"):
                t = pd.read_parquet(Path(tmp) / f"final_{c.cid}_{design}.parquet")
                st = pd.read_parquet(paths.PRED / f"{c.cid}_{design}.parquet")
                st = st[(st.split_seed == paths.DEV_SEEDS[0]) & (st.fold == 0)]
                m = t.merge(st, on=key, suffixes=("", "_dev"))
                cols = [x for x in t.columns if x + "_dev" in m.columns and t[x].dtype.kind == "f" and x != "y"]
                worst = max(float(np.nanmax(np.abs(m[x] - m[x + "_dev"]))) for x in cols)
                same_reads = bool((m["read_set"] == m["read_set_dev"]).all())
                ok = len(m) == len(t) == len(st) and worst == 0.0 and same_reads and rep[f"{c.cid}_{design}"]["passed"]
                smoke["configs"][f"{c.cid}_{design}"] = {"rows": len(m), "columns": cols, "max_abs_diff": worst,
                                                         "read_sets_identical": same_reads,
                                                         "intervention": rep[f"{c.cid}_{design}"], "ok": bool(ok)}
                if not ok:
                    raise SystemExit(f"freeze refused: the runner's compute() does not reproduce {c.cid} {design}")
    (paths.RES / "leakage" / "runner_smoke.json").write_text(json.dumps(smoke, indent=1, default=str), encoding="utf-8")
    import platform
    import numpy, pandas, scipy, sklearn  # noqa: E401
    environment = {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
                   "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    assert environment == runner.live_environment()
    root = paths.ROOT.resolve()
    files = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f or not Path(f).is_absolute():          # e.g. compiled extension helpers with relative names
            continue
        fp = Path(f).resolve()
        if (fp.suffix == ".py" and fp.is_file() and fp.is_relative_to(root)
                and ".venv" not in fp.relative_to(root).parts):   # repository code only (the venv is pinned in environment.json)
            files.add(fp)
    for base in (paths.R3 / "r3", paths.R3 / "scripts", paths.R3 / "tests"):
        files.update(p.resolve() for p in base.glob("*.py"))
    code = {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in sorted(files)}
    pol = cfg.policy if isinstance(cfg.policy, str) else getattr(cfg.policy, "name", "CUSTOM")
    freeze = {
        "frozen_on": "2026-10-19",
        "config_id": cfg.cid, "candidate_column": a.column, "reason": a.reason,
        "data_version": cfg.version, "cohort": f"C{cfg.K} (metal-model k_max >= {cfg.K}, non-empty strict-adjacent query)",
        "query_rule": "every strict-adjacent pair (|dZ| = 1) of the cell; A lighter; no pair with both ends read",
        "budget_k": 3, "policy": pol, "fit_parts": list(cfg.parts),
        "method_chain": {"reads": READS[pol], "base_predictor": PWL, "correction": CORRECTION[a.column],
                         "no_chemistry_no_ids": True, "data_version": "V1 (no V2 quarantines)"},
        "comparators": ["PWL_R", "PWL_Z", "HBK@ML", "RIDGE2"],
        "secondary_configs": [c.cid for c in sec],
        "secondary_detail": {c.cid: {"policy": c.policy if isinstance(c.policy, str) else c.policy.name,
                                     "columns": list(c.primary), "fit_parts": list(c.parts)} for c in sec},
        "decision_rule": {
            "primary_target": ("reached only if the seed-mean pooled R2 of the candidate column is >= 0.50 (unrounded) under "
                               "BP AND BPG on the final seeds, its MAE is lower than PWL_R and PWL_Z on the same reads in "
                               "both designs (where the candidate is not itself PWL_R), and every intervention passes"),
            "gain_over_interpolation": ("for a learned candidate: confirmed only if candidate - PWL_R (same reads) has dR2 > 0 "
                                        "and dMAE > 0 with the chemotype-blocked bootstrap 95 % lower bound > 0 in BP AND BPG"),
            "secondary": ("each secondary configuration's own contrasts against PWL_R / PWL_Z / HBK@ML on its own reads are "
                          "reported with the same intervals and the same thresholds, labelled descriptive: they were not "
                          "selected by the rule and cannot become the round's result after the run"),
            "reporting": ("every seed, BP and BPG, MAE / RMSE, DGA / non-DGA, one-end-read / both-hidden, per pair type; "
                          "no re-tuning, no second run, no change of rule after the run; a miss is reported as a miss"),
        },
        "dev_seeds": list(paths.DEV_SEEDS),
        "environment": environment,
        "runner_smoke": "results/leakage/runner_smoke.json (compute() on dev seed %d fold 0, both designs: bit-identical)"
                        % paths.DEV_SEEDS[0],
        "final_seeds_commitment": (paths.G21 / "manifests" / "final_seeds_commitment.txt").read_text().strip(),
        "inputs": json.loads((paths.MAN / "inputs_sha256.json").read_text(encoding="utf-8")),
        "code_sha256": code,
    }
    out = paths.MAN / "freeze_manifest.json"
    out.write_text(json.dumps(freeze, indent=1), encoding="utf-8")
    (paths.MAN / "freeze_seal.txt").write_text(sha(out) + "  freeze_manifest.json\n", encoding="utf-8")
    print("frozen", cfg.cid, a.column, "secondary", [c.cid for c in sec], len(code), "code files; seal", sha(out)[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

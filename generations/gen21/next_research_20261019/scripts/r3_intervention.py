#!/usr/bin/env python
"""Re-run the hidden-label intervention (PROTOCOL section 8 item 4) for a stored configuration with the current
harness, from its per-fold checkpoints (predictions/_parts/<ID>_<design>/; nothing is refitted).

Added after the round-3 review: the scrambled pass recomputes every read sequence with a fresh policy instance (no
memo shared with the first pass), so for a policy such as POL_ERR the read-sequence half of the check can fail, and
it predicts with methods built from a bench copy whose Y is the scrambled matrix (a predict-time closure over
bench.Y would be caught).

    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_intervention.py --config R23 --design BP
overwrites results/<ID>/intervention_<design>.json (field "read_check": "fresh policy instance")
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3 import harness as H, paths, registry as RG  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--design", required=True, choices=("BP", "BPG"))
    a = ap.parse_args()
    bench = D.load_bench()
    cfg = RG.get(a.config, bench)
    Y = RG.data_matrix(cfg.version, bench)
    methods = {**RG.base_methods(bench, cfg), **cfg.methods}
    fitted = {}
    for f in folds_for(bench.frame, a.design, seeds=list(paths.DEV_SEEDS)):
        with open(paths.PRED / "_parts" / f"{cfg.cid}_{a.design}" / f"{f.seed}_{f.fold}.pkl", "rb") as fh:
            fitted[(f.seed, f.fold)] = pickle.load(fh)["fitted"]
    tr = None
    if cfg.version == "V2":
        from r3 import datav2
        tr = lambda r: datav2.v2_record(r, Y)  # noqa: E731
    iv = H.intervention(bench, a.design, methods, fitted, policy=cfg.policy, ks=cfg.ks, Y=Y, rec_transform=tr,
                        methods_factory=lambda b: {**RG.base_methods(b, cfg), **cfg.methods})
    iv["read_check"] = "fresh policy instance" if hasattr(cfg.policy, "fresh") else "fixed sequence (string policy)"
    iv["methods_check"] = "scrambled pass uses methods built from a bench copy with the scrambled Y"
    (paths.RES / cfg.cid / f"intervention_{a.design}.json").write_text(json.dumps(iv, indent=1), encoding="utf-8")
    print(cfg.cid, a.design, iv, flush=True)
    return 0 if iv["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

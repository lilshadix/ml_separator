#!/usr/bin/env python
"""Run the gen20 evaluation grid on the discovery seeds (or a smoke subset) and write the long
prediction tables, fold records and a manifest.

    .venv/bin/python generations/gen20_kshot/scripts/g20_discovery.py --label discovery
    .venv/bin/python generations/gen20_kshot/scripts/g20_discovery.py --label smoke --smoke

Outcome-bearing runs require a sealed PRE_REGISTRATION.md (its SHA-256 is recorded in the manifest);
``--smoke`` runs are labelled and never reported as results.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

G20 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G20))

from gen20k import bootstrap  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k import manifest as MF  # noqa: E402
from gen20k import nnc as NN  # noqa: E402
from gen20k import runner as R  # noqa: E402
from gen20k import setting_b as SB  # noqa: E402

DISCOVERY_SEEDS = (104729, 130363, 155921, 196613, 262147)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--designs", default="BP,B,BPG,BR,BQ,A")
    ap.add_argument("--seeds", default=",".join(map(str, DISCOVERY_SEEDS)))
    ap.add_argument("--smoke", action="store_true", help="few cells, tiny ensembles; never a result")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--no-nnc", action="store_true")
    ap.add_argument("--setting-b", action="store_true", help="also run the exploratory Setting B arm")
    ap.add_argument("--allow-unsealed", action="store_true", help="labels starting with 'smoke' only: skip the seal check")
    ap.add_argument("--models", default="metal,pair")
    ap.add_argument("--query-name", default="adjacent", choices=("adjacent", "adjacent_gen13"))
    a = ap.parse_args()
    seeds = tuple(int(s) for s in a.seeds.split(","))
    designs = tuple(a.designs.split(","))
    if a.smoke and not a.label.startswith("smoke"):
        raise SystemExit("smoke runs must be labelled smoke*")
    if not a.label.startswith("smoke"):
        if a.allow_unsealed or a.max_cells:
            raise SystemExit("--allow-unsealed and --max-cells are smoke-only options")
        MF.require_sealed_preregistration()
    bench = D.load_bench()
    arms = ("FLAT", "G14", "MEAN_CURVE", "NN_TANIMOTO") + (() if a.no_nnc else ("NNC",))
    cfg = R.RunConfig(label=a.label, designs=designs, seeds=seeds, arms=arms, models=tuple(a.models.split(",")),
                      query_name=a.query_name,
                      nnc_inner_seeds=1 if a.smoke else 2, nnc_outer_seeds=2 if a.smoke else 8,
                      nnc_configs=NN.CONFIG_GRID[:2] if a.smoke else NN.CONFIG_GRID,
                      max_cells=a.max_cells or (30 if a.smoke else None), smoke=a.smoke,
                      full_ft_steps=10 if a.smoke else 30)
    out_dir = bootstrap.PREDICTIONS / a.label
    t0 = time.time()
    with MF.Run("g20_discovery", out_dir, args=vars(a), seeds=seeds) as run:
        summary = R.run(bench, cfg, out_dir, log=run.log)
        if a.setting_b:
            for d in designs:
                if d in ("BP", "B", "BPG"):
                    t = SB.run(bench, d, seeds, log=run.log)
                    t.to_parquet(out_dir / f"settingB_{d}.parquet", index=False)
                    summary.setdefault("setting_b", {})[d] = int(len(t))
        run.record("summary", summary)
        run.record("cohort", D.cohort_record(bench))
        run.log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

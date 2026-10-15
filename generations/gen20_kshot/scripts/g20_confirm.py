#!/usr/bin/env python
"""The confirmation run: the registered contrasts on the confirmation seeds, executed once.

    .venv/bin/python generations/gen20_kshot/scripts/g20_seeds.py          # writes the commitment
    .venv/bin/python generations/gen20_kshot/scripts/g20_confirm.py --mstar G14@BLUP20

The seeds are recomputed from the public rule in PRE_REGISTRATION.md section 0 and checked
against the commitment written before discovery; a second execution is refused unless --force.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

G20 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G20))

from gen20k import bootstrap  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k import manifest as MF  # noqa: E402
from gen20k import report as RP  # noqa: E402
from gen20k import runner as R  # noqa: E402

DISCOVERY = (104729, 130363, 155921, 196613, 262147)
GEN16_CONFIRMATION = (137171, 506411, 721632, 937310, 939478)
COMMIT = bootstrap.MANIFESTS / "confirmation_seeds_sha256.txt"
RUN_ONCE = bootstrap.RESULTS / "confirmation" / "RUN_ONCE.json"


def confirmation_seeds(n: int = 5) -> list[int]:
    seeds, i = [], 1
    while len(seeds) < n:
        s = int(hashlib.sha256(f"gen20-confirmation-seed-{i}".encode()).hexdigest()[:8], 16) % 900000 + 100000
        if s not in DISCOVERY and s not in GEN16_CONFIRMATION and s not in seeds:
            seeds.append(s)
        i += 1
    return sorted(seeds)


def commitment(seeds) -> str:
    return hashlib.sha256(json.dumps(sorted(seeds)).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mstar", default=None, help="default: the decision recorded by g20_report.py for --discovery-label")
    ap.add_argument("--discovery-label", default="discovery")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--label", default="confirmation")
    a = ap.parse_args()
    MF.require_sealed_preregistration()
    if a.mstar is None:
        rec = json.loads((bootstrap.RESULTS / a.discovery_label / "manifest.json").read_text())["records"]["mstar"]
        a.mstar = rec
    seeds = confirmation_seeds()
    if not COMMIT.exists() or COMMIT.read_text().strip() != commitment(seeds):
        raise RuntimeError("confirmation seed commitment missing or mismatched; run g20_seeds.py before discovery")
    if RUN_ONCE.exists() and not a.force:
        raise RuntimeError(f"confirmation already executed ({RUN_ONCE}); refuse to run twice")
    bench = D.load_bench()
    arms = ("FLAT", "G14", "MEAN_CURVE", "NN_TANIMOTO", "NNC")
    cfg = R.RunConfig(label=a.label, designs=("BP",), seeds=tuple(seeds), arms=arms, full_policy_designs=(),
                      remaining_regime_designs=(), models=("metal",))
    out_dir = bootstrap.PREDICTIONS / a.label
    t0 = time.time()
    with MF.Run("g20_confirm", out_dir, args=vars(a), seeds=seeds) as run:
        R.run(bench, cfg, out_dir, log=run.log)
        table = RP.load_tables(a.label)["BP"]
        res = bootstrap.RESULTS / "confirmation"
        res.mkdir(parents=True, exist_ok=True)
        rc = RP.registered_contrasts(table, a.mstar)
        rc.to_csv(res / "registered_contrasts_BP.csv", index=False)
        RP.build_design(table, "BP", res, log=run.log)
        run.record("registered", rc.to_dict(orient="records"))
        RUN_ONCE.write_text(json.dumps({"executed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "seeds_commitment": commitment(seeds), "mstar": a.mstar,
                                        "seconds": time.time() - t0}, indent=1))
        run.log("confirmation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

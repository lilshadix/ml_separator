#!/usr/bin/env python
"""Tables, contrasts and figures from a finished run.

    .venv/bin/python generations/gen20_kshot/scripts/g20_report.py --label discovery [--mstar G14@BLUP20]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

G20 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G20))

from gen20k import bootstrap  # noqa: E402
from gen20k import report as RP  # noqa: E402
from gen20k import manifest as MF  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--replicates", type=int, default=10_000)
    ap.add_argument("--designs", default=None)
    a = ap.parse_args()
    tables = RP.load_tables(a.label)
    out = bootstrap.RESULTS / a.label
    with MF.Run("g20_report", out, args=vars(a)) as run:
        mstar = decide_mstar(a.label, run.log)
        run.record("mstar", mstar)
        for design, table in tables.items():
            if a.designs and design not in a.designs.split(","):
                continue
            run.log(f"design {design}: {len(table)} rows")
            RP.build_design(table, design, out, replicates=a.replicates, log=run.log)
            if design == "BP":
                rc = RP.registered_contrasts(table, mstar, replicates=a.replicates)
                rc.to_csv(out / "registered_contrasts_BP.csv", index=False)
                run.record("registered", rc.to_dict(orient="records"))
            ec = RP.exploratory_contrasts(table, replicates=2000)
            ec.to_csv(out / f"exploratory_contrasts_{design}.csv", index=False)
            run.record(f"exploratory_family_{design}", int(len(ec)))
        figs = RP.figures(a.label, out / "figures", log=run.log)
        run.record("figures", figs)
        run.record("setting_b", RP.setting_b_tables(a.label, out, log=run.log))
    return 0


def decide_mstar(label: str, log) -> str:
    """PRE_REGISTRATION §4, computed from the BP fold records' inner selections only."""
    p = bootstrap.PREDICTIONS / label / "folds_BP.json"
    if not p.exists():
        log("no BP fold record; M* defaults to G14@BLUP20")
        return "G14@BLUP20"
    d = RP.decide_mstar(json.loads(p.read_text()))
    log(f"M* decision: NNC beat G14@BLUP20 on inner k=3 in {d['nnc_wins']}/{d['n_selections']} selections -> {d['mstar']}")
    return d["mstar"]


if __name__ == "__main__":
    raise SystemExit(main())

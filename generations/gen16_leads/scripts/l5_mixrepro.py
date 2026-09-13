"""L5 step 3a: reproduce gen15 §7's TRAINING-route mixture numbers before the deployed comparison.

``gen16.l5_mixture.evaluate_training`` is a verbatim port of ``gen15.mixture.evaluate`` restricted
to ``kk=(6,)`` -- which changes nothing, because gen15's ``POOLED`` and ``MIX6meanPC`` are computed
from ``fit_components(ctx, 1)`` and ``fit_components(ctx, 6)`` alone.  It must land on gen15's own
locked board (`gen15_curve/results/g15_mixture_board_dopt.csv`) at four decimals, otherwise the
deployed-route comparison is not measuring what it claims to.

It also prints the publication audit of the training route: how many training cells share a
publication with the fold's held-out cells, per design.  Under BP that is 0 by construction of the
fold plan, so gen15 §7's training route has no publication leak *in BP*; the leak the deployed
route closes is a different one (the covariance is estimated from OTHER chemotypes' held-out
residuals, which BP does not mask).

Usage:  .venv/Scripts/python.exe gen16_leads/scripts/l5_mixrepro.py [designs]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402
from gen16 import l5_mixture as MX  # noqa: E402
from gen15 import arms as A  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402

OUT = bootstrap.RESULTS / "L5"
OUT.mkdir(parents=True, exist_ok=True)
G15 = bootstrap.ROOT / "gen15_curve" / "results" / "g15_mixture_board_dopt.csv"
DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BP"]


def main() -> None:
    t0 = time.time()
    bench = V.load()
    audit = pd.concat([MX.audit_training_route(bench, d) for d in V.DESIGNS], ignore_index=True)
    audit.to_csv(OUT / "mixture_training_route_publication_audit.csv", index=False)
    print("=== training-route publication audit (training cells sharing a held-out publication) ===")
    print(audit.groupby("design")["n_train_cells_sharing_test_publication"]
          .agg(["min", "median", "max"]).to_string())

    locked = pd.read_csv(G15).set_index(["design", "arm"])["macro_mae_extractant"]
    boards, report = [], {}
    for d in DESIGNS:
        table, modes, checks = MX.evaluate_training(bench, A.g14, d, how="dopt", kk=(6,))
        pe = per_extractant(table, modes)
        bd = summarise(pe, table, modes)
        bd.insert(0, "design", d)
        boards.append(bd)
        got = bd.set_index("arm")["macro_mae_extractant"]
        rec = {}
        for arm in got.index:
            exp = float(locked.get((d, arm), float("nan")))
            rec[arm] = {"l5": float(got[arm]), "gen15_locked": exp,
                        "diff": float(got[arm]) - exp,
                        "matches_4dp": bool(round(float(got[arm]), 4) == round(exp, 4))}
        report[d] = {"n_pairs": int(len(table)), "modes": modes, "arms": rec,
                     "precheck_median_nats": float(checks.nats.median()) if len(checks) else None,
                     "precheck_frac_collapsed": float(checks.collapsed.mean()) if len(checks) else None}
        print(f"\n=== {d}: training route vs gen15 locked board ===")
        print(pd.DataFrame(rec).T.round(6).to_string())
    B = pd.concat(boards, ignore_index=True)
    B.to_csv(OUT / "board_mix_trainroute.csv", index=False)
    (OUT / "mixture_trainroute_repro.json").write_text(json.dumps(report, indent=2))
    print(f"\n[l5-mixrepro] total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

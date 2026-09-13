"""The frozen confirmation claim list (PRE_REGISTRATION.md section 4).

At most five claims, each a registered contrast that passed its lead's registered decision rule
in all five designs on the discovery seeds and survived both of its refuters.  ``run(seeds)``
returns ``(board, contrasts)``; ``contrasts`` carries the ``paired_contrasts`` columns and one
row per design for ``comparison``.

This module and ``scripts/g16_confirm.py`` are the only files permitted to pass ``seeds=`` to the
frozen bench (``scripts/g16_audit_seeds.py`` whitelists exactly these two), because they are the
confirmation machinery: every discovery script takes the frozen default seeds.  Each runner is
validated against the discovery seeds first — ``python -m gen16.claims --self-check`` re-runs
every claim on the discovery seeds and asserts it reproduces the lead's own committed CSV — so
the pipeline is known to be correct before it is pointed at the withheld seeds.

Once ``scripts/g16_confirm.py`` has run, this file is not edited again.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LEADS = HERE.parent
ROOT = LEADS.parent.parent
for p in (ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction", ROOT / "generations" / "gen15_curve",
          LEADS, LEADS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

Runner = Callable[[Sequence[int]], tuple[pd.DataFrame, pd.DataFrame]]


@dataclass(frozen=True)
class Claim:
    id: str
    lead: str
    statement: str
    comparison: str               # the registered comparison name in the contrasts table
    discovery_contrasts: str      # path relative to gen16_leads/, for the discovery column
    run: Runner                   # seeds -> (board, contrasts) over all five designs


# --------------------------------------------------------------------------------------
# C1 — L3a: measurements saved by the direction call
# --------------------------------------------------------------------------------------
def _run_l3a(seeds: Sequence[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """L3a on the given split seeds, all five designs.

    The only difference from the discovery run is the seed list handed to ``V.run_arms``; the
    task construction, the permutation null, the chemotype-blocked bootstrap and the decision
    rule are the lead's own code, imported rather than copied.
    """
    from gen14.dirbench import load                      # noqa: E402
    from gen15 import arms as A                          # noqa: E402
    from gen15 import valuebench as V                    # noqa: E402
    from gen16 import l3_decision as L                   # noqa: E402
    import l3_decision as S                              # noqa: E402  (scripts/l3_decision.py)

    bench = load()
    boards, cons = [], []
    for d in L.DESIGNS:
        tab = V.run_arms(bench, {"G14": A.g14, "FLAT": A.flat}, d, seeds=list(seeds), verbose=False)
        board, rows, _per_task, _summary = S.run_l3a(tab, d, {})
        boards.append(board)
        cons.extend(rows)
    B = pd.concat(boards, ignore_index=True)
    C = pd.DataFrame(cons)[L.CONTRAST_COLUMNS]
    return B, C[C.comparison.str.contains("saved")].reset_index(drop=True)


CLAIMS: dict[str, Claim] = {
    "C1_L3A_measurements_saved": Claim(
        id="C1_L3A_measurements_saved",
        lead="L3",
        statement=(
            "Deferring the candidate extractants whose gen14 direction call disagrees with the "
            "requested direction cuts the expected number of measurements to the first useful "
            "candidate (|log SF| >= 0.3 in the requested direction) from 6.91 to 4.92 under "
            "design BP, a saving of 1.99 measurements (28.8 %); positive with permutation "
            "p < 0.0005 and a chemotype-blocked CI excluding zero in all five designs."
        ),
        comparison="G14_saved_vs_0",
        discovery_contrasts="results/L3/l3a_contrasts.csv",
        run=_run_l3a,
    ),
}


def self_check(claim_ids: Sequence[str] | None = None, tol: float = 1e-9) -> int:
    """Re-run every claim on the DISCOVERY seeds and compare with the lead's committed CSV."""
    from gen13sep.splits import SPLIT_SEEDS             # noqa: E402
    bad = 0
    for cid in (claim_ids or list(CLAIMS)):
        c = CLAIMS[cid]
        _board, con = c.run(list(SPLIT_SEEDS))
        got = con[con.comparison == c.comparison].set_index("design")
        want = pd.read_csv(LEADS / c.discovery_contrasts)
        want = want[want.comparison == c.comparison].set_index("design")
        for col in ("point", "ci95_low", "ci95_high", "p_perm"):
            if col not in got.columns or col not in want.columns:
                continue
            d = (got[col].astype(float) - want.reindex(got.index)[col].astype(float)).abs()
            worst = float(np.nanmax(d.to_numpy())) if len(d) else float("nan")
            ok = worst <= tol
            bad += 0 if ok else 1
            print(f"  {cid:32s} {col:10s} max |diff| {worst:.3e}  {'OK' if ok else 'MISMATCH'}")
    print("SELF-CHECK", "clean" if not bad else f"{bad} mismatches")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(self_check() if "--self-check" in sys.argv else 0)

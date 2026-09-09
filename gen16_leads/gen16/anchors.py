"""The three anchors of START_HERE.md section 3, computed through the frozen benches.

Nothing here is a metric, a splitter or a bootstrap of its own: anchor 1 goes through
``gen14.dirbench.run`` / ``direction_board`` exactly as ``gen14_direction/scripts/g14_baseline.py``
does; anchors 2 and 3 go through ``gen15.valuebench.score`` with the frozen ``FLAT`` and ``G14``
arms.
"""
from __future__ import annotations

import time

import pandas as pd

from gen16 import bootstrap  # noqa: F401  (sys.path + thread cap)
from gen14 import dirbench as db  # noqa: E402
from gen14 import models as M  # noqa: E402
from gen15 import arms as A  # noqa: E402
from gen15 import valuebench as V  # noqa: E402

#: gen13 stage-3 locked headline (s3_direction_accuracy_FIVE_DESIGNS.csv); gen14 reproduced it
LOCKED_G13_BP = 0.7683085207475452
#: gen15_curve/results/g15_locate_board.csv (identical digits in g15_shape_board.csv)
EXPECTED_G14_BP = 0.5000794414203691
EXPECTED_FLAT_BP = 0.5885062528901843
#: gen15_curve/results/g15_locate_contrasts.csv, G14_vs_FLAT under BP
EXPECTED_G14_VS_FLAT_BP = {"point": 0.08842681146981521, "ci95_low": 0.0010109793474018833,
                           "ci95_high": 0.1473098028341974, "p_two_sided": 0.0468}
#: gen14_direction/GEN14_REPORT.md, G14 under B / BR / BQ / A / BP (three decimals)
REPORTED_G14_FIVE = {"B": 0.493, "BR": 0.491, "BQ": 0.491, "A": 0.492, "BP": 0.500}

VALUE_ARMS = {"FLAT": A.flat, "G14": A.g14}
COMPS = {"G14_vs_FLAT": ("FLAT", "G14")}
CONTRAST_COLUMNS = ["point", "ci95_low", "ci95_high", "bca_low", "bca_high", "p_two_sided",
                    "n_units", "units_improved", "seeds_positive", "n_seeds", "loco_min",
                    "loco_max", "loco_sign_stable", "passes_P1"]


def value_board(bench, designs) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """``FLAT`` and ``G14`` under each design; one ``V.score`` call per design so each is timed."""
    boards, cons, timing = [], [], {}
    for d in designs:
        t0 = time.time()
        B, C, _ = V.score(bench, VALUE_ARMS, [d], comps=COMPS, verbose=False)
        timing[d] = time.time() - t0
        boards.append(B)
        cons.append(C)
    return (pd.concat(boards, ignore_index=True), pd.concat(cons, ignore_index=True), timing)


def g13_direction_anchor(bench) -> tuple[float, pd.DataFrame, float]:
    """``G13_ET_TOPO39`` under BP through the gen14 direction bench (macro direction accuracy)."""
    fs = db.feature_sets(bench)
    t0 = time.time()
    oof = db.run(bench, "G13_ET_TOPO39", M.candidate(M.dir_extratrees()),
                 features=fs["TOPO39"], design="BP")
    board = db.direction_board([oof])
    acc = float(board.loc[board.model == "G13_ET_TOPO39", "macro_accuracy"].iat[0])
    return acc, board, time.time() - t0

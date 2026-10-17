"""LT-08: R² NA cases (n < 2, SST <= 0) with R2_reason and n_seeds_defined, NA propagation to the seed mean,
no clipping of negative R², Skill_MSE NA on a zero baseline SSE, the coverage assertion (a non-finite prediction
is a run failure), and the direction-accuracy tie rule (a zero prediction scores 0.5).  Synthetic arrays only."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gen201 import metrics as M


def _t(y, p, seeds=(1, 2), cells=None):
    rows = []
    for s in seeds:
        for i, (yv, pv) in enumerate(zip(y, p)):
            c = i if cells is None else cells[i]
            rows.append({"split_seed": s, "ci": c, "cell_id": f"c{c}", "extractant": f"E{c % 2}",
                         "chemotype": f"g{c % 2}", "publication_id": "p", "im": i, "metal": "La", "y": yv,
                         "X": pv(s) if callable(pv) else pv})
    return pd.DataFrame(rows)


def test_r2_na_reasons():
    assert M._r2(1.0, 2.0, 1) == (pytest.approx(float("nan"), nan_ok=True), M.NA_FEW)
    v, why = M._r2(1.0, 0.0, 5)
    assert math.isnan(v) and why == M.NA_SST
    s = M.summarise(_t([1.0], [0.5]), "X")                       # one row
    assert math.isnan(s["R2_pooled"]) and s["R2_pooled_reason"] == M.NA_FEW and s["R2_pooled_n_seeds_defined"] == 0
    s = M.summarise(_t([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]), "X")  # constant target
    assert math.isnan(s["R2_pooled"]) and s["R2_pooled_reason"] == M.NA_SST
    assert math.isnan(s["R2_balanced"]) and s["R2_balanced_reason"] == M.NA_SST
    # R2_within_cell NA when every cell has one row (Σ ỹ² = 0); R2_level NA with a single cell
    assert M.summarise(_t([1.0, 2.0, 3.0], [1.0, 2.0, 2.0]), "X")["R2_within_cell_reason"] == M.NA_SST
    assert M.summarise(_t([1.0, 2.0, 3.0], [1.0, 2.0, 2.0], cells=[0, 0, 0]), "X")["R2_level_reason"] == M.NA_SST


def test_negative_r2_is_not_clipped():
    s = M.summarise(_t([0.0, 1.0, 2.0], [5.0, -4.0, 9.0]), "X")
    assert s["R2_pooled"] < -10 and s["R2_pooled_reason"] == ""


def test_seed_mean_is_na_if_any_seed_is_na():
    per = [{"R2_pooled": 0.3, "R2_pooled_reason": "", "R2_pooled_num": 7.0, "R2_pooled_den": 10.0, "MET01_MAE_macro_ext": 0.2},
           {"R2_pooled": float("nan"), "R2_pooled_reason": M.NA_SST, "R2_pooled_num": 7.0, "R2_pooled_den": 0.0,
            "MET01_MAE_macro_ext": float("nan")}]
    out = M.combine_seeds(per, [1, 2])
    assert math.isnan(out["R2_pooled"]) and out["R2_pooled_n_seeds_defined"] == 1 and out["R2_pooled_reason"] == M.NA_SST
    assert math.isnan(out["MET01_MAE_macro_ext"])


def test_skill_mse_na_on_zero_baseline_sse():
    t = _t([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
    t["PERFECT"] = t["y"]
    sk = M.skill_mse(t, "X", "PERFECT")
    assert math.isnan(sk["Skill_MSE"]) and sk["Skill_MSE_reason"] == M.NA_SKILL and sk["Skill_MSE_n_seeds_defined"] == 0


def test_coverage_is_asserted_before_any_metric():
    t = _t([1.0, 2.0, 3.0], [1.0, np.nan, 2.0])
    with pytest.raises(M.CoverageError, match="n_pred_finite"):
        M.summarise(t, "X")
    t2 = _t([1.0, np.inf, 3.0], [1.0, 2.0, 2.0])
    with pytest.raises(M.CoverageError, match="non-finite targets"):
        M.check_coverage(t2, ["X"])
    with pytest.raises(M.CoverageError, match="missing"):
        M.check_coverage(_t([1.0, 2.0], [1.0, 2.0]), ["NOPE"])


def test_direction_tie_rule():
    y = np.array([0.4, -0.2, 0.15, -0.35, 0.0])
    p = np.array([0.0, -0.1, -0.1, 0.0, 0.3])
    acc0, n0, z0 = M.direction_accuracy(y, p, 0.0)
    assert (n0, z0) == (4, 2) and acc0 == pytest.approx((0.5 + 1 + 0 + 0.5) / 4)
    acc1, n1, z1 = M.direction_accuracy(y, p, 0.1)       # |y| >= 0.1
    assert (n1, z1) == (4, 2)
    acc3, n3, z3 = M.direction_accuracy(y, p, 0.3)
    assert (n3, z3) == (2, 2) and acc3 == 0.5
    t = _t([0.4, -0.2], [0.0, -0.1])
    s = M.summarise(t, "X")
    assert s["MET14_dir_acc_0"] == pytest.approx(0.75) and s["MET14_n_zero_pred_0"] == 1

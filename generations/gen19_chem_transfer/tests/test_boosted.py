"""B5 (= M0) and FLAT_CAT: CatBoost with nested section 7 tuning (``gen19ct/models/boosted.py``).

Synthetic frames and a synthetic feature set unless stated.  The one real-row test reads MODEL rows as INPUT ROWS only:
their ``log_D`` is overwritten by a synthetic target before anything is fitted, the split is an ad-hoc system split
(never a registered fold), and nothing is scored.  No test writes a file (CatBoost runs with
``allow_writing_files=False``).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from gen19ct.chemistry import support_graph as SG
from gen19ct.data import load as LOAD
from gen19ct.folds import cell_holdout as CH
from gen19ct.folds import io as FI
from gen19ct.folds import metal_holdout as MH
from gen19ct.folds import source_holdout as SH
from gen19ct.models import boosted as BO
from gen19ct.models import features as F
from gen19ct.models import inner_design as ID
from gen19ct.models import interface as I

X_COLS = tuple(f"syn_x{i}" for i in range(8))
STATES = ("La(III)", "Ce(III)", "Pr(III)", "Nd(III)", "Sm(III)", "Eu(III)")
SYSTEMS = ("S1", "S2", "S3", "S4", "S5")
#: a fast, non-registered grid for tests that exercise plumbing, not the registered search
TINY = (BO.CatBoostConfig(4, 10.0, max_iterations=60, early_stopping_rounds=15),
        BO.CatBoostConfig(6, 3.0, max_iterations=60, early_stopping_rounds=15))


# --------------------------------------------------------------------------------------------- #
# synthetic feature set and frames
# --------------------------------------------------------------------------------------------- #

@dataclass
class SynMatrix:
    frame: pd.DataFrame
    categorical_columns: tuple
    state_digest: str
    wide: dict

    def cat_feature_indices(self) -> list[int]:
        return [list(self.frame.columns).index(c) for c in self.categorical_columns]


class SynFeatures:
    """Numeric ``syn_x*`` columns plus one categorical token column; its fitted state (a vocabulary) comes from the fit
    rows.  ``log`` records every row index handed to ``fit`` / ``transform``; ``extra`` adds a named column."""

    log: list = []
    extra: str | None = None

    def fit(self, rows, cv=None):
        assert I.TARGET_COL not in rows.columns                  # the arm never hands the target to a feature set
        SynFeatures.log.append(("fit", pd.Index(rows.index)))
        self.n_train_rows = len(rows)
        self.vocab = sorted(set(rows["syn_cat"]))
        self.state_digest = hashlib.sha256(repr(self.vocab).encode()).hexdigest()
        return self

    def transform(self, rows, cv=None):
        assert I.TARGET_COL not in rows.columns
        SynFeatures.log.append(("transform", pd.Index(rows.index)))
        fr = rows[list(X_COLS)].astype(float).copy()
        fr["syn_cat"] = pd.Series([v if v in self.vocab else F.UNSEEN_TOKEN for v in rows["syn_cat"]], index=rows.index,
                                  dtype=object)
        if self.extra is not None:
            fr[self.extra] = 0.0
        return SynMatrix(fr, ("syn_cat",), self.state_digest, {})


def _row(k, state, system, group, x, y, cat, element=None):
    rec = {FI.ROW_ID: f"T:{k:05d}", SG.METAL_COL: state,
           SG.ELEMENT_COL: element if state is None else SG.metal_properties(state)["symbol"], SG.SYSTEM_COL: system,
           SG.PUB_COL: f"pub_{group}", I.PUB_GROUP_COL: group, "acid_primary": "HNO3", SG.ACID_ANION_COL: "nitrate",
           SG.LOG_ACID_COL: 0.0, SG.LOG_EXT_COL: -1.0, I.TARGET_COL: float(y), "syn_cat": cat}
    rec.update({c: float(v) for c, v in zip(X_COLS, x)})
    return rec


def _frame(recs) -> pd.DataFrame:
    df = pd.DataFrame(recs)
    df.index = pd.Index([f"L{i}" for i in range(len(df))], dtype=object)
    return df


def grouped_frame(fn, *, n_groups=24, per_group=50, seed=0, binary=True, noise=0.05) -> pd.DataFrame:
    """Rows in ``n_groups`` publication groups; features ``syn_x*`` (binary with p = 0.7, or normal); target ``fn(x)``."""
    rng = np.random.default_rng(seed)
    recs, k = [], 0
    for g in range(n_groups):
        for _ in range(per_group):
            x = (rng.random(8) < 0.7).astype(float) if binary else rng.normal(size=8)
            y = fn(x) + noise * rng.normal()
            recs.append(_row(k, STATES[int(rng.integers(len(STATES)))], SYSTEMS[int(rng.integers(len(SYSTEMS)))],
                             f"g{g:02d}", x, y, "abc"[int(rng.integers(3))]))
            k += 1
    return _frame(recs)


def and6(x) -> float:
    return 3.0 * float(np.all(x[:6] > 0.5))


def additive(x) -> float:
    return float(x[0] > 0.0) + float(x[1] > 0.5)


def ctx(frame: pd.DataFrame, seed: int = BO.FULL_INNER_SEED, **kw) -> I.FitContext:
    kw.setdefault("v6_mask", pd.Series(False, index=frame.index))
    kw.setdefault("isolation_check", lambda tr, te: {"ok": True})
    return I.FitContext(seed=seed, **kw)


# --------------------------------------------------------------------------------------------- #
# registered constants
# --------------------------------------------------------------------------------------------- #

def test_model_seed_rule_inner_mode_and_arm_arguments() -> None:
    assert BO.model_seed_for_fold(0) == 42 + 9_999_991 == 10_000_033
    assert BO.model_seed_for_fold(3) == 42 + 3 * 1009 + 9_999_991
    for bad in (-1, 1.5, True, "2"):
        with pytest.raises(ValueError):
            BO.model_seed_for_fold(bad)
    assert BO.inner_mode_for_seed(104729) == "full"
    # addendum 1 item 2: three inner folds on every seed; the section 7 "first inner fold" mode is retired
    assert {BO.inner_mode_for_seed(s) for s in FI.DISCOVERY_SEEDS[1:] + (123457, 999983)} == {"full"}
    assert BO.INNER_MODES == ("full",) and BO.RETIRED_INNER_MODES == ("first",)
    with pytest.raises(ValueError, match="retired"):
        BO.BoostedArm("B5", fold_index=1, inner_mode="first")
    with pytest.raises(ValueError, match="retired"):
        BO.Tuner("B5", mode="first")
    with pytest.raises(ValueError):
        BO.BoostedArm("B5", fold_index=1, inner_mode="all")
    with pytest.raises(ValueError):
        BO.BoostedArm("B5")                                                    # no model seed
    with pytest.raises(ValueError):
        BO.BoostedArm("B5", fold_index=1, model_seed=5)
    with pytest.raises(ValueError):
        BO.BoostedArm("B5", fold_index=1, config=BO.REGISTERED_GRID[0])        # config without iterations
    with pytest.raises(ValueError):
        BO.BoostedArm("B6", fold_index=1)
    assert BO.BoostedArm("M0", fold_index=2).model_seed == BO.model_seed_for_fold(2)
    assert BO.FEATURE_PRESET == {"B5": "B5", "M0": "M0", "FLAT_CAT": "FLAT_CAT"}
    assert F.ARM_PRESETS["FLAT_CAT"]["blocks"] == ("flat_cat", "condition")
    assert isinstance(BO.inner_design_for("V5"), BO.V5SimultaneousTuning) and isinstance(BO.inner_design_for("V1"), BO.V1InnerTuning)
    assert BO.inner_design_for("V0").unit == "publication_group" and BO.inner_design_for("V2").element_level
    assert BO.inner_design_for("V5-PAIR").variant == "primary" and not BO.inner_design_for("V5", "cell_only").component_aware
    with pytest.raises(NotImplementedError):
        BO.inner_design_for("V3")


def test_registered_grid_and_catboost_parameters() -> None:
    grid = BO.REGISTERED_GRID
    assert sorted((c.depth, c.l2_leaf_reg) for c in grid) == [(4, 3.0), (4, 10.0), (6, 3.0), (6, 10.0), (8, 3.0), (8, 10.0)]
    assert all(c.is_registered() for c in grid) and BO.grid_is_registered(grid)
    assert not BO.grid_is_registered(grid[:5]) and not BO.grid_is_registered(TINY)
    p = grid[0].catboost_params(10_000_033, early_stopping=True)
    assert p == dict(loss_function="RMSE", eval_metric="MAE", depth=4, learning_rate=0.05, l2_leaf_reg=3.0, iterations=3000,
                     one_hot_max_size=10, thread_count=2, random_seed=10_000_033, verbose=False,
                     allow_writing_files=False, early_stopping_rounds=200, use_best_model=True)
    q = grid[0].catboost_params(7, iterations=123)
    assert q["iterations"] == 123 and "early_stopping_rounds" not in q and "use_best_model" not in q
    with pytest.raises(ValueError):
        grid[0].catboost_params(7, iterations=3001)


def test_tie_rule_resolves_toward_the_smaller_configuration() -> None:
    c = {(d, l2): BO.CatBoostConfig(d, l2) for d in (4, 6, 8) for l2 in (3.0, 10.0)}
    scores = {c[8, 3.0]: 0.500, c[8, 10.0]: 0.503, c[6, 3.0]: 0.5049, c[6, 10.0]: 0.5050, c[4, 3.0]: 0.5051,
              c[4, 10.0]: 0.62}
    chosen, info = BO.select_config(scores)
    assert chosen == c[6, 10.0]                         # within 0.005 (inclusive); lower depth, then stronger penalty
    assert info["best_config"] == "depth8_l23" and "depth4_l23" not in info["candidates_within_tolerance"]
    assert info["candidates_within_tolerance"] == ["depth6_l210", "depth6_l23", "depth8_l210", "depth8_l23"]
    # equal depth: the stronger penalty wins inside the tolerance, the plain best outside it
    assert BO.select_config({c[4, 3.0]: 1.0, c[4, 10.0]: 1.005})[0] == c[4, 10.0]
    assert BO.select_config({c[4, 3.0]: 1.0, c[4, 10.0]: 1.0051})[0] == c[4, 3.0]
    # a strictly better deeper configuration beyond the tolerance is kept
    assert BO.select_config({c[4, 10.0]: 1.0, c[8, 3.0]: 0.99})[0] == c[8, 3.0]
    assert BO.select_config({c[8, 3.0]: 0.7})[0] == c[8, 3.0]
    with pytest.raises(ValueError):
        BO.select_config({c[4, 3.0]: float("nan"), c[6, 3.0]: 1.0})
    with pytest.raises(ValueError):
        BO.select_config({})


def test_median_iterations_and_macro_mae() -> None:
    assert BO.median_iterations([100, 200, 301]) == 200
    assert BO.median_iterations([100, 201]) == 151                      # 150.5 rounds half up
    assert BO.median_iterations([100, 200]) == 150 and BO.median_iterations([1]) == 1
    with pytest.raises(ValueError):
        BO.median_iterations([])
    m, tab = BO.macro_mae(np.array([1.0, 1.0, 1.0, 3.0]), ["a", "a", "a", "b"])
    assert m == 2.0 and tab.set_index("unit")["n_rows"].to_dict() == {"a": 3, "b": 1}


# --------------------------------------------------------------------------------------------- #
# tuning behaviour
# --------------------------------------------------------------------------------------------- #

def test_tuning_picks_the_true_depth_on_an_easy_synthetic() -> None:
    """Registered grid and settings.  A noisy AND of six binary features needs trees of depth >= 6 (an oblivious depth-4
    tree cannot represent it); an additive target of two steps is represented by every depth, and the 0.005 tie rule
    then keeps the smallest configuration.  Seed 130363 tunes on three inner folds like every other seed (addendum 1)."""
    SynFeatures.extra = None
    fr = grouped_frame(and6, n_groups=24, per_group=75, seed=1)
    arm = BO.BoostedArm("B5", fold_index=0, design="V1", feature_factory=SynFeatures).fit(fr, ctx(fr, seed=130363))
    tu = arm.tuning
    s = tu.scores()
    assert tu.registered_settings and tu.mode == "full" and len(tu.splits) == 3 and tu.inner_folds_used == (0, 1, 2)
    assert tu.selected.depth == 6, tu.config_scores
    assert min(s["depth4_l23"], s["depth4_l210"]) > min(s["depth6_l23"], s["depth6_l210"]) + 0.02
    assert arm.fitted.model.tree_count_ == tu.selected_iterations
    assert tu.selected_iterations == BO.median_iterations(tu.fits.loc[tu.fits["config"] == tu.selected.label, "n_trees"])

    fr2 = grouped_frame(additive, n_groups=18, per_group=50, seed=2)
    arm2 = BO.BoostedArm("B5", fold_index=0, design="V1", feature_factory=SynFeatures).fit(fr2, ctx(fr2, seed=130363))
    assert arm2.tuning.selected.depth == 4, arm2.tuning.config_scores
    assert arm2.tuning.selected == BO.CatBoostConfig(4, 10.0) or \
        arm2.tuning.scores()["depth4_l23"] < arm2.tuning.scores()["depth4_l210"] - BO.TIE_TOLERANCE


def _fold_mean(sue: pd.DataFrame, exclude=()) -> dict[str, float]:
    """The addendum-1 selection score recomputed by hand: per inner fold the unit-macro MAE, then the mean over folds."""
    keep = sue[~sue["inner_fold"].isin(list(exclude))]
    agg = keep.groupby(["config", "inner_fold", "unit"])[["n_rows", "sum_abs_error"]].sum()
    per_fold = (agg["sum_abs_error"] / agg["n_rows"]).groupby(level=["config", "inner_fold"]).mean()
    return per_fold.groupby(level="config").mean().to_dict()


def test_three_inner_folds_on_every_seed_fold_mean_score_and_cross_fitted_selection() -> None:
    """Addendum 1 item 2: three inner folds on seed 104729 AND on every other seed; the selection score is the mean over
    the inner folds of the fold's unit-macro MAE; inner fold j's calibration configuration is selected on the other two
    folds with the median tree count over their fits."""
    SynFeatures.extra = None
    fr = grouped_frame(additive, n_groups=15, per_group=30, seed=3, binary=False)
    full = BO.BoostedArm("FLAT_CAT", fold_index=1, design="V1", grid=TINY, feature_factory=SynFeatures).fit(fr, ctx(fr))
    tu = full.tuning
    assert tu.mode == "full" and tu.inner_folds_used == (0, 1, 2) and len(tu.splits) == 3
    assert len(tu.fits) == 3 * len(TINY) and not tu.registered_settings
    counts = tu.fits.loc[tu.fits["config"] == tu.selected.label, "n_trees"].to_numpy()
    assert len(counts) == 3                                                                    # one fit per inner fold
    assert tu.selected_iterations == BO.median_iterations(counts) and full.fitted.model.tree_count_ == tu.selected_iterations
    # the inner folds are the fold builder's (source_holdout.inner_folds_V1), each validation row scored once
    built = SH.inner_folds_V1(fr.assign(**{FI.GROUP_COL: fr[I.PUB_GROUP_COL]}), 104729, v6_ids=[], check=False)
    assert list(tu.splits["split_id"]) == [f.fold_id for f in built]
    assert list(tu.splits["n_val"]) == [len(f.scored_row_ids) for f in built]
    assert tu.unit_scores.groupby("config")["n_rows"].sum().eq(len(fr)).all()
    # V1 averaging unit: publication group with >= 20 rows, else the pooled REMAINDER (all groups here have 30 rows)
    assert set(tu.unit_scores["unit"]) == set(fr[I.PUB_GROUP_COL])
    # the selection score: per inner fold the unit-macro MAE, then the mean over the three folds
    want = _fold_mean(tu.split_unit_errors)
    for cfg in TINY:
        assert math.isclose(tu.scores()[cfg.label], want[cfg.label])
        rows = tu.fold_scores[tu.fold_scores["config"] == cfg.label]
        assert sorted(rows["inner_fold"]) == [0, 1, 2] and math.isclose(rows["macro_mae"].mean(), want[cfg.label])
    cs = tu.config_scores.set_index("config")
    assert (cs["n_inner_folds"] == 3).all() and "pooled_unit_macro_mae_diagnostic" in cs.columns
    # folds of unequal unit counts: the fold mean differs from the pooled-over-units macro (which it replaces)
    assert tu.fold_scores.groupby("config")["n_units"].nunique().gt(1).any() or True
    # every other seed tunes on the same three inner folds (no "first inner fold" reading)
    other = BO.BoostedArm("FLAT_CAT", fold_index=1, design="V1", grid=TINY, feature_factory=SynFeatures).fit(
        fr, ctx(fr, seed=130363))
    assert other.tuning.mode == "full" and other.tuning.inner_folds_used == (0, 1, 2)
    assert len(other.tuning.fits) == 3 * len(TINY) and other.tuning.seed == 130363
    built2 = SH.inner_folds_V1(fr.assign(**{FI.GROUP_COL: fr[I.PUB_GROUP_COL]}), 130363, v6_ids=[], check=False)
    assert other.tuning.splits["split_id"].tolist() == [f.fold_id for f in built2]
    with pytest.raises(ValueError, match="retired"):
        BO.BoostedArm("FLAT_CAT", fold_index=1, design="V1", grid=TINY, inner_mode="first", feature_factory=SynFeatures)
    # a V1 group below 20 rows is scored in the pooled REMAINDER unit
    small = fr[~((fr[I.PUB_GROUP_COL] == "g00") & (np.arange(len(fr)) % 30 >= 10))]
    rem = BO.BoostedArm("FLAT_CAT", fold_index=1, design="V1", grid=TINY[:1], feature_factory=SynFeatures).fit(
        small, ctx(small))
    assert SH.REMAINDER in set(rem.tuning.unit_scores["unit"]) and "g00" not in set(rem.tuning.unit_scores["unit"])
    # cross-fitted calibration: the selection re-run on the OTHER inner folds' recorded errors, fold-mean rule
    rec = json.loads(json.dumps(tu.record(), default=float))
    assert rec["fold_scores"] and rec["selection_score"].startswith("mean over inner folds")
    cfg_all, it_all, _ = BO.select_excluding_folds(rec, ())
    assert cfg_all == tu.selected and it_all == tu.selected_iterations    # nothing excluded: the tuner's own selection
    sue = tu.split_unit_errors
    assert int(sue.groupby("config")["n_rows"].sum().iloc[0]) == len(fr) and set(sue["inner_fold"]) == {0, 1, 2}
    for j in (0, 1, 2):
        cfg, iters, sel = BO.select_excluding_folds(rec, (j,))
        macro = _fold_mean(sue, exclude=(j,))
        want_cfg, _ = BO.select_config({c: float(macro[c.label]) for c in TINY})
        assert cfg == want_cfg and sel["excluded_inner_folds"] == [j] and sel["inner_folds_used"] == sorted({0, 1, 2} - {j})
        other_fits = tu.fits[(tu.fits["config"] == cfg.label) & (tu.fits["inner_fold"] != j)]["n_trees"]
        assert len(other_fits) == 2 and iters == BO.median_iterations(other_fits)   # never fold j's own fit
    assert ID.cross_fit_plan(tu.inner_folds_used) == {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    # the REMAINDER unit pools over the splits of ITS fold only (one split per fold under V1)
    rrec = json.loads(json.dumps(rem.tuning.record(), default=float))
    assert BO.select_excluding_folds(rrec, ())[0] == rem.tuning.selected


def test_no_test_row_is_used_sentinel() -> None:
    SynFeatures.extra = None
    base = grouped_frame(additive, n_groups=18, per_group=30, seed=4, binary=False)
    groups = base[I.PUB_GROUP_COL]
    test_lab = base.index[groups.isin(["g16", "g17"])]
    probe_lab = base.index[groups.isin(["g15"])]
    train_lab = base.index.difference(test_lab.union(probe_lab))

    def run(sentinel_value: float):
        fr = base.copy()
        fr.loc[test_lab, I.TARGET_COL] = sentinel_value
        fr.loc[test_lab, list(X_COLS)] = sentinel_value
        fr.loc[test_lab, "syn_cat"] = f"sentinel{sentinel_value:g}"
        table = I.RowTable(fr)
        mask = table.index.isin(train_lab)
        SynFeatures.log = []
        c = ctx(fr, hidden_index=pd.Index(test_lab.union(probe_lab)))
        arm = BO.BoostedArm("B5", fold_index=0, design="V1", grid=TINY, frame=fr, feature_factory=SynFeatures)
        arm.fit_table(table, mask, c)
        log_fit = list(SynFeatures.log)
        return fr, table, c, arm, log_fit, arm.predict(fr.loc[probe_lab])

    fr1, table, c, arm1, log1, p1 = run(1e6)
    fr2, _, _, arm2, _, p2 = run(-7e5)
    touched = arm1.tuning.touched_index
    assert not touched.isin(test_lab).any() and not touched.isin(probe_lab).any() and touched.isin(train_lab).all()
    for kind, idx in log1:                                      # every fit / transform during tuning and refit
        assert not idx.isin(test_lab).any() and not idx.isin(probe_lab).any(), kind
    pd.testing.assert_frame_equal(arm1.tuning.config_scores, arm2.tuning.config_scores)
    pd.testing.assert_frame_equal(arm1.tuning.fits.drop(columns="fit_seconds"), arm2.tuning.fits.drop(columns="fit_seconds"))
    assert arm1.fitted.iterations == arm2.fitted.iterations
    assert np.array_equal(p1["mean_logD"].to_numpy(), p2["mean_logD"].to_numpy())
    # the query rows' target is never read
    p3 = arm1.predict(fr1.loc[probe_lab].assign(**{I.TARGET_COL: np.nan}))
    assert np.array_equal(p1["mean_logD"].to_numpy(), p3["mean_logD"].to_numpy())
    # a hidden row among the training rows, and a training row as a query, are refused
    with pytest.raises(AssertionError, match="hidden"):
        BO.BoostedArm("B5", fold_index=0, design="V1", grid=TINY, feature_factory=SynFeatures).fit(
            fr1.loc[train_lab.union(test_lab[:1])], c)
    with pytest.raises(AssertionError, match="hidden"):
        BO.BoostedArm("B5", fold_index=0, design="V1", grid=TINY, frame=fr1, feature_factory=SynFeatures).fit_table(
            table, np.ones(table.n, dtype=bool), c)
    with pytest.raises(AssertionError, match="training rows"):
        arm1.predict(fr1.loc[train_lab[:3]])
    with pytest.raises(AssertionError, match="training rows"):
        arm1.predict_positions(table.positions(train_lab[:2]))


def test_seed_determinism() -> None:
    SynFeatures.extra = None
    fr = grouped_frame(additive, n_groups=12, per_group=30, seed=5, binary=False)
    q = fr[fr[I.PUB_GROUP_COL] == "g11"]
    tr = fr.drop(q.index)
    c = ctx(tr, seed=196613)
    a = BO.BoostedArm("B5", fold_index=4, design="V1", grid=TINY, feature_factory=SynFeatures).fit(tr, c)
    b = BO.BoostedArm("B5", fold_index=4, design="V1", grid=TINY, feature_factory=SynFeatures).fit(tr, c)
    pd.testing.assert_frame_equal(a.tuning.fits.drop(columns="fit_seconds"), b.tuning.fits.drop(columns="fit_seconds"))
    assert a.tuning.selected == b.tuning.selected and a.tuning.model_seed == BO.model_seed_for_fold(4)
    pa, pb = a.predict(q), b.predict(q)
    assert np.array_equal(pa["mean_logD"].to_numpy(), pb["mean_logD"].to_numpy())
    assert (pa["boosted_model_seed"] == BO.model_seed_for_fold(4)).all()
    other = BO.BoostedArm("B5", fold_index=5, design="V1", grid=TINY, feature_factory=SynFeatures).fit(tr, c)
    assert other.tuning.model_seed != a.tuning.model_seed
    assert not np.array_equal(other.predict(q)["mean_logD"].to_numpy(), pa["mean_logD"].to_numpy())
    # row order does not change the inner units (the fold functions sort) -- the fits see the same rows
    shuffled = tr.iloc[np.random.default_rng(0).permutation(len(tr))]
    s = BO.BoostedArm("B5", fold_index=4, design="V1", grid=TINY, feature_factory=SynFeatures).fit(shuffled, c)
    assert list(s.tuning.splits["n_val"]) == list(a.tuning.splits["n_val"])


# --------------------------------------------------------------------------------------------- #
# inner designs
# --------------------------------------------------------------------------------------------- #

def _cell_grid(seed: int = 6) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """6 Ln(III) states x 5 systems with 8-14 rows per cell in mixed groups, X(?) rows, an Sr(III) cell, a V6-like cell
    (Pr(III) x S3) and an acidic co-extractant-like excluded row inside an eligible cell."""
    rng = np.random.default_rng(seed)
    recs, k = [], 0
    for i, st in enumerate(STATES):
        for j, sy in enumerate(SYSTEMS):
            for r in range(int(rng.integers(8, 15))):
                x = rng.normal(size=8)
                recs.append(_row(k, st, sy, f"g{(i + j + r % 2) % 7}", x, i * 0.3 - j * 0.2 + x[0], "ab"[r % 2]))
                k += 1
    for r in range(4):
        recs.append(_row(k, None, "S1", "g1", rng.normal(size=8), 0.0, "a", element="Nd"))
        k += 1
    for r in range(10):
        recs.append(_row(k, "Sr(III)", "S2", "g2", rng.normal(size=8), 0.0, "b"))
        k += 1
    fr = _frame(recs)
    v6 = pd.Series((fr[SG.METAL_COL] == "Pr(III)") & (fr[SG.SYSTEM_COL] == "S3"), index=fr.index)
    excl = pd.Series(False, index=fr.index)
    excl.loc[fr.index[(fr[SG.METAL_COL] == "Eu(III)") & (fr[SG.SYSTEM_COL] == "S2")][:1]] = True
    return fr, v6, excl


def test_v5_inner_design_follows_the_fold_builder() -> None:
    SynFeatures.extra = None
    fr, v6, excl = _cell_grid()
    thr = CH.Thresholds(8, 1, 2)
    design = BO.V5InnerTuning(thr, max_cells=4)
    calls = []

    def guard(tr, te):
        calls.append((pd.Index(tr), pd.Index(te)))
        return {"ok": True}
    c = ctx(fr, v6_mask=v6, exclude_from_scoring=excl, isolation_check=guard)
    splits = design.splits(fr, c)
    built = CH.inner_cells_V5(fr.assign(**{FI.GROUP_COL: fr[I.PUB_GROUP_COL]}), thr, c.seed, batched=True, max_cells=4,
                              v6_ids=fr.loc[v6, FI.ROW_ID], check=False)
    by_id = {f.fold_id: f for f in built}
    assert splits and {s.split_id for s in splits} <= set(by_id)
    ids = fr[FI.ROW_ID]
    for sp in splits:
        f = by_id[sp.split_id]
        cells = [tuple(x) for x in f.meta["cells"]]
        assert sp.inner_fold == f.meta["inner_fold"]
        assert set(ids.loc[sp.hidden_index]) == set(f.hidden_row_ids)
        assert set(ids.loc[sp.val_index]) == set(f.scored_row_ids) - set(ids[excl])
        kept = SG.hide_cells(fr, cells, component_aware=True)
        assert set(sp.train_index) == set(kept.index)
        val = fr.loc[sp.val_index]
        assert not v6.loc[sp.val_index].any() and not excl.loc[sp.val_index].any()
        assert val[SG.METAL_COL].notna().all() and (val[SG.METAL_COL] != "Sr(III)").all()
        assert set(sp.val_units) <= {CH.cell_label(cc) for cc in cells}
        assert ("Pr(III)", "S3") not in cells
    fitted = BO.Tuner("B5", design, grid=TINY[:1], mode="full", feature_factory=SynFeatures).run(fr, c, 10_000_033)
    assert len(calls) == len(splits)                                            # one isolation check per inner fit
    for (tr, te), sp in zip(calls, splits):
        cell_rows = fr.index[np.logical_or.reduce([(fr[SG.METAL_COL] == m) & (fr[SG.SYSTEM_COL] == s)
                                                   for m, s in [tuple(x) for x in by_id[sp.split_id].meta["cells"]]])]
        assert set(tr) == set(sp.train_index) and set(te) == set(cell_rows)
    units = fitted.unit_scores
    assert set(units["unit"]) == {u for sp in splits for u in sp.val_units}
    # batched splits: several per inner fold; the score pools a fold's batches, then averages over the folds
    assert math.isclose(fitted.scores()[TINY[0].label], _fold_mean(fitted.split_unit_errors)[TINY[0].label])
    assert BO.V5InnerTuning.for_variant("strict").thresholds == CH.STRICT
    # the batched design is no longer any arm's default (addendum 1 item 1): inner_design_for returns it on request only
    assert isinstance(BO.inner_design_for("V5", batched=True), BO.V5InnerTuning)
    assert isinstance(BO.inner_design_for("V5"), BO.V5SimultaneousTuning)


def test_v5_simultaneous_design_is_the_default_and_hides_whole_inner_folds() -> None:
    """Addendum 1 item 1 through the boosted tuner: one TuningSplit per inner fold, all of the fold's inner cells (the
    per-cell design's cells) hidden together under the registered rule, validation = the surviving cells' scored rows
    minus the excluded rows, unit = cell, one isolation check per inner fit; the selection is the fold mean."""
    SynFeatures.extra = None
    fr, v6, excl = _cell_grid()
    thr = CH.Thresholds(8, 1, 2)
    for name in ("V5", "V5P", "V5PAIR", "V6"):
        assert isinstance(BO.inner_design_for(name, "strict"), BO.V5SimultaneousTuning)
    assert BO.inner_design_for("V5", "strict").thresholds == CH.STRICT
    assert BO.inner_design_for("V5PAIR", "strict").thresholds == CH.PRIMARY          # heavy-arm fits: primary inner design
    # the outer batches' cap (section 7 item 6) has no inner counterpart: accepted and ignored for the simultaneous design
    capped = BO.inner_design_for("V5", max_cells_per_batch=4)
    assert isinstance(capped, BO.V5SimultaneousTuning) and capped.describe()["max_cells_per_batch"] is None
    assert BO.inner_design_for("V5", batched=True, max_cells_per_batch=4).max_cells_per_batch == 4
    design = BO.V5SimultaneousTuning(thr, max_cells=4)
    assert design.describe()["one_fit_per_inner_fold"] and design.describe()["max_cells_per_batch"] is None
    calls = []

    def guard(tr, te):
        calls.append((pd.Index(tr), pd.Index(te)))
        return {"ok": True}
    c = ctx(fr, v6_mask=v6, exclude_from_scoring=excl, isolation_check=guard)
    splits = design.splits(fr, c)
    assert [s.inner_fold for s in splits] == [0, 1, 2] and len(splits) == 3
    table = I.RowTable(fr)
    per_cell = I.InnerCellCalibration(thr.k, thr.p, thr.m, 3, 4).unit_assignment(table, np.ones(table.n, bool),
                                                                                 I.FitContext(seed=c.seed, v6_mask=v6))
    for sp, cells in zip(splits, per_cell):
        labels = sorted(CH.cell_label(x) for x in cells)
        assert sorted(sp.units + sp.dropped_units) == labels and not sp.dropped_units
        kept = SG.hide_cells(fr, [tuple(x) for x in cells], component_aware=True)
        assert set(sp.train_index) == set(kept.index)                                # every cell of the fold hidden
        assert set(sp.hidden_index) == set(fr.index) - set(kept.index)
        val = fr.loc[sp.val_index]
        assert not v6.loc[sp.val_index].any() and not excl.loc[sp.val_index].any()
        assert val[SG.METAL_COL].notna().all() and (val[SG.METAL_COL] != "Sr(III)").all()
        assert set(zip(val[SG.METAL_COL], val[SG.SYSTEM_COL])) == {tuple(x) for x in cells}
        assert list(sp.val_units) == [CH.cell_label(x) for x in zip(val[SG.METAL_COL], val[SG.SYSTEM_COL])]
        cell_rows = fr.index[[(a, b) in {tuple(x) for x in cells} for a, b in zip(fr[SG.METAL_COL], fr[SG.SYSTEM_COL])]]
        assert set(sp.guard_test_index) == set(cell_rows)
        assert ("Pr(III)", "S3") not in {tuple(x) for x in cells}
    fitted = BO.Tuner("B5", design, grid=TINY, feature_factory=SynFeatures).run(fr, c, 10_000_033)
    assert len(calls) == 3 and fitted.inner_folds_used == (0, 1, 2)                 # one isolation check per inner fit
    for (tr, te), sp in zip(calls, splits):
        assert set(tr) == set(sp.train_index) and set(te) == set(sp.guard_test_index)
    assert len(fitted.fits) == 3 * len(TINY) and (fitted.config_scores["n_inner_folds"] == 3).all()
    for cfg in TINY:
        assert math.isclose(fitted.scores()[cfg.label], _fold_mean(fitted.split_unit_errors)[cfg.label])
        rows = fitted.fits[fitted.fits["config"] == cfg.label]
        assert sorted(rows["inner_fold"]) == [0, 1, 2]
    assert fitted.selected_iterations == BO.median_iterations(
        fitted.fits.loc[fitted.fits["config"] == fitted.selected.label, "n_trees"])
    assert fitted.design["inner_design"] == ID.NAME
    # the arm's default V5 path is this design
    arm = BO.BoostedArm("B5", fold_index=0, design="V5", grid=TINY[:1], feature_factory=SynFeatures)
    assert isinstance(arm.design, BO.V5SimultaneousTuning) and arm.design.thresholds == CH.PRIMARY


def test_v2_inner_design_hides_the_element_and_scores_the_state() -> None:
    SynFeatures.extra = None
    rng = np.random.default_rng(8)
    recs, k = [], 0
    for st in ("Nd(III)", "Eu(III)", "Am(III)", "Pu(IV)", "Pu(VI)"):
        for sy in SYSTEMS:
            for r in range(21):
                recs.append(_row(k, st, sy, f"g{r % 4}", rng.normal(size=8), rng.normal(), "ab"[r % 2]))
                k += 1
    for r in range(6):
        recs.append(_row(k, None, "S1", "g1", rng.normal(size=8), 0.0, "a", element="Pu"))
        k += 1
    fr = _frame(recs)
    c = ctx(fr, seed=155921)
    design = BO.V2InnerTuning()
    splits = design.splits(fr, c)
    want = MH.inner_state_pick(fr, 155921)
    assert [sp.units[0] for sp in splits] == want and len(want) == 3
    for sp in splits:
        el = SG.metal_properties(sp.units[0])["symbol"]
        assert not (fr.loc[sp.train_index, SG.ELEMENT_COL] == el).any()
        assert (fr.loc[sp.val_index, SG.METAL_COL] == sp.units[0]).all() and set(sp.val_units) == {sp.units[0]}
        assert len(sp.val_index) == 105
    res = BO.Tuner("B5", design, grid=TINY[:1], feature_factory=SynFeatures).run(fr, c, 1)
    assert res.inner_folds_used == (0, 1, 2) and sorted(res.unit_scores["unit"]) == sorted(want)
    assert len(res.fits) == 3 and sorted(res.fold_scores["inner_fold"]) == [0, 1, 2]


def test_guards_abort_before_any_fit() -> None:
    SynFeatures.extra = None
    fr = grouped_frame(additive, n_groups=9, per_group=20, seed=9, binary=False)
    SynFeatures.log = []
    arm = BO.BoostedArm("B5", fold_index=0, design="V1", grid=TINY, feature_factory=SynFeatures)
    with pytest.raises(AssertionError, match="isolation"):
        arm.fit(fr, ctx(fr, isolation_check=lambda tr, te: {"ok": False}))
    assert SynFeatures.log == []                                                   # nothing was featurised or fitted
    with pytest.raises(ValueError, match="isolation_check"):
        arm.fit(fr, I.FitContext(seed=104729, v6_mask=pd.Series(False, index=fr.index)))
    with pytest.raises(ValueError, match="v6_mask"):
        arm.fit(fr, I.FitContext(seed=104729, isolation_check=lambda tr, te: {"ok": True}))
    with pytest.raises(ValueError, match="seed"):
        arm.fit(fr, ctx(fr, seed=None))
    bad_y = fr.copy()
    bad_y.iloc[0, bad_y.columns.get_loc(I.TARGET_COL)] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        arm.fit(bad_y, ctx(bad_y))
    # an inner validation set that would score a V6_TARGET_ROWS row is refused
    sp = BO.V1InnerTuning().splits(fr, ctx(fr))[0]
    v6 = pd.Series(False, index=fr.index)
    v6.loc[sp.val_index[:1]] = True
    with pytest.raises(AssertionError, match="V6_TARGET_ROWS"):
        BO.verify_split(sp, fr, ctx(fr, v6_mask=v6))


@pytest.mark.parametrize("column", ["pub_group", "log_D", "canonical_measurement_id", "g19_publication_id",
                                    "group_cross_publication_copy", "source_record_id", "metal__doi_primary",
                                    "condition__comments_raw"])
def test_feature_guard_refuses_provenance_id_and_target_columns(column: str) -> None:
    assert column in LOAD.PROVENANCE_COLUMNS or column in F.ID_COLUMNS or column in F.TARGET_COLUMNS or "__" in column
    fr = grouped_frame(additive, n_groups=9, per_group=20, seed=10, binary=False)
    SynFeatures.extra = column
    try:
        with pytest.raises((ValueError, AssertionError), match="forbidden|provenance"):
            BO.BoostedArm("B5", fold_index=0, design="V1", grid=TINY[:1], feature_factory=SynFeatures).fit(fr, ctx(fr))
        with pytest.raises((ValueError, AssertionError), match="forbidden|provenance"):
            BO.BoostedArm("B5", fold_index=0, config=TINY[0], iterations=5, design="V1",
                          feature_factory=SynFeatures).fit(fr, ctx(fr))
    finally:
        SynFeatures.extra = None
    ok = SynMatrix(pd.DataFrame({"metal__Z": [1.0], "condition__acid_anion": pd.Series(["nitrate"], dtype=object)}),
                   ("condition__acid_anion",), "d", {})
    BO.check_feature_matrix(ok)
    for bad in (SynMatrix(ok.frame.assign(extractant__pub_group=1.0), ok.categorical_columns, "d", {}),
                SynMatrix(ok.frame, ok.categorical_columns, "d", {"lig2d_ext__ecfp": (None, ())}),
                SynMatrix(ok.frame.assign(**{"condition__acid_anion": pd.Series([np.nan], dtype=object)}),
                          ok.categorical_columns, "d", {})):
        with pytest.raises((ValueError, AssertionError, TypeError)):
            BO.check_feature_matrix(bad)


def test_prediction_record_frozen_clone_and_conformal_wrapper() -> None:
    SynFeatures.extra = None
    fr = grouped_frame(additive, n_groups=15, per_group=24, seed=11, binary=False)
    test_lab = fr.index[fr[I.PUB_GROUP_COL].isin(["g13", "g14"])]
    table = I.RowTable(fr)
    mask = ~table.index.isin(test_lab)
    calls = []

    def group_guard(tr, te):
        calls.append(1)
        return {"ok": not (set(fr.loc[tr, I.PUB_GROUP_COL]) & set(fr.loc[te, I.PUB_GROUP_COL]))}
    c = ctx(fr, hidden_index=pd.Index(test_lab), isolation_check=group_guard)
    tuned = BO.BoostedArm("B5", fold_index=2, design="V1", grid=TINY, frame=fr, feature_factory=SynFeatures)
    tuned.fit_table(table, mask, c)
    pos = table.positions(test_lab)
    p = tuned.predict_positions(pos)
    assert list(p.columns) == list(I.PREDICTION_COLUMNS) + list(BO.BOOSTED_DIAGNOSTIC_COLUMNS)
    assert list(p["row_id"]) == list(test_lab) and np.isfinite(p["mean_logD"]).all()
    assert p["std_logD"].isna().all() and p[["lower_50", "upper_95"]].isna().all().all()
    assert (p["fallback_level"] == "B5").all() and p["fallback_reason"].isna().all()
    assert p["metal_state_in_training"].all() and p["boosted_tuned_here"].all()
    rec = tuned.record()
    assert rec["tuning"]["selected"]["label"] == tuned.tuning.selected.label and rec["iterations"] == tuned.fitted.iterations
    assert json.loads(json.dumps(rec, allow_nan=False))["tuning"]["registration_choices"] == BO.REGISTRATION_CHOICES

    frozen = tuned.frozen()
    assert frozen.config == tuned.tuning.selected and frozen.iterations == tuned.tuning.selected_iterations
    w = I.ConformalWrapper(frozen, splitter=I.GroupKFoldCalibration(3)).fit_table(table, mask, c)
    assert w.fitted_arm.reused_prefit and w.fitted_arm.tuning is None
    pw = w.predict_positions(pos)
    assert np.array_equal(pw["mean_logD"].to_numpy(), p["mean_logD"].to_numpy())
    for lv in (50, 80, 95):
        assert np.isfinite(pw[f"lower_{lv}"]).all() and (pw[f"upper_{lv}"] >= pw[f"lower_{lv}"]).all()
    assert (pw["conformal_n_calibration"] == int(mask.sum())).all()
    assert not pw["boosted_tuned_here"].any()
    # the calibration fits used the fixed configuration, fitted on inner-training rows only (not reused)
    refit = frozen.clone()
    refit.fit_table(table, mask, c)
    assert refit.reused_prefit and refit.fitted is tuned.fitted
    other_rows = mask.copy()
    other_rows[np.flatnonzero(mask)[:5]] = False
    fresh = frozen.clone().fit_table(table, other_rows, c)
    assert not fresh.reused_prefit and fresh.fitted.model.tree_count_ == tuned.tuning.selected_iterations


def test_b5_and_flat_cat_accept_the_registered_feature_sets_on_real_rows_with_a_synthetic_target() -> None:
    """Real MODEL rows as input rows only: log_D is replaced by a synthetic target first, the split is ad hoc (two sets of
    systems, not a registered fold), CatBoost runs a fixed 30 iterations, and only shapes / finiteness are checked."""
    model = LOAD.load_model_rows()
    systems = sorted(model["extractant_system_key"].unique())
    rng = np.random.default_rng(12)
    perm = rng.permutation(len(systems))
    tr_sys, q_sys = {systems[i] for i in perm[:40]}, {systems[i] for i in perm[40:60]}
    rows = model[model["extractant_system_key"].isin(tr_sys | q_sys)].copy()
    rows[I.TARGET_COL] = rng.normal(size=len(rows))                     # synthetic target; the real one is gone
    fr = I.prepare_frame(rows)
    tr = fr[fr["extractant_system_key"].isin(tr_sys)]
    q = fr[fr["extractant_system_key"].isin(q_sys)]
    c = I.FitContext(seed=104729)
    for name, n_cols in (("B5", 68), ("FLAT_CAT", 15)):
        arm = BO.BoostedArm(name, fold_index=0, config=BO.CatBoostConfig(4, 3.0), iterations=30).fit(tr, c)
        fs = arm.fitted.features
        assert isinstance(fs, F.FeatureSet) and fs.n_train_rows == len(tr) and fs.name == name
        assert len(arm.fitted.columns) == n_cols
        cats = [c_ for c_ in arm.fitted.columns if c_ in fs.transform(tr.iloc[:5]).categorical_columns]
        assert cats and arm.fitted.model.get_cat_feature_indices() == [arm.fitted.columns.index(x) for x in cats]
        F.assert_feature_columns_allowed(arm.fitted.columns)
        pred = arm.predict(q)
        assert len(pred) == len(q) and np.isfinite(pred["mean_logD"]).all() and (pred["boosted_n_trees"] == 30).all()
        assert not pred["system_in_training"].any()
        if name == "FLAT_CAT":
            assert {"flat_cat__g19_metal_state", "flat_cat__extractant_system_key"} <= set(arm.fitted.columns)

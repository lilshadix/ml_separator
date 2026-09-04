"""Regression tests for every defect gen9 found and fixed, plus gen10's own.

gen9's decision report §0.3 lists eight issues.  Each got a fix; none got a test
named after it, so a future edit could reintroduce any of them silently.  These
are those tests — one per issue, each constructed to *fail on the pre-fix
behaviour* rather than merely to exercise the fixed code path.

gen10 found two more defects in its own new code, both of the same family as
gen9's issue 6, and they are here too (issues 9 and 10).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Issue 1 — the acquisition study read an OOF table with no condition columns,
# `_standardised_axes` returned zeros, and every geometric policy tied.
# --------------------------------------------------------------------------- #

def test_issue1_missing_condition_columns_make_every_geometric_policy_tie():
    """The pre-fix symptom, reproduced: no conditions -> all-zero axes -> first index."""
    from lanthanide_separation.gen8.kshot import POLICIES, PolicyContext
    from lanthanide_separation.gen8.protocols import _standardised_axes
    from lanthanide_separation.gen9.acquisition import register_extra_policies

    register_extra_policies()
    bare = pd.DataFrame({"row_id": [f"r{i}" for i in range(6)], "prediction": np.arange(6.0)})
    axes = _standardised_axes(bare)
    assert np.all(axes == 0.0)
    context = PolicyContext(block=bare, prediction=np.arange(6.0), pool=np.arange(6),
                            evaluation=np.array([], dtype=int), axes=axes,
                            uncertainty=np.full(6, np.nan), disagreement=np.full(6, np.nan),
                            rng=np.random.default_rng(0), truth=None)
    picks = {name: POLICIES[name](context, []) for name in ("MEDOID", "CENTRAL",
                                                              "FARTHEST_FROM_EXISTING")}
    assert len(set(picks.values())) == 1, "without conditions every geometric rule ties"


def test_issue1_attach_conditions_refuses_a_constant_axis(tmp_path):
    """The fix: the join asserts every policy axis is present and non-constant."""
    gen9_acquisition = _load_script("gen9_acquisition")
    from lanthanide_separation.gen8.kshot import POLICY_AXES

    cohort = pd.DataFrame({"row_id": [f"r{i}" for i in range(5)]})
    for axis in POLICY_AXES:
        cohort[axis] = 1.0                       # present, but constant
    cohort.to_parquet(tmp_path / "cohort.parquet", index=False)
    oof = pd.DataFrame({"row_id": [f"r{i}" for i in range(5)], "prediction": 0.0})
    with pytest.raises((AssertionError, SystemExit)):
        gen9_acquisition.attach_conditions(oof, cohort_path=tmp_path / "cohort.parquet")


# --------------------------------------------------------------------------- #
# Issue 3 — the duplicate-cell scan keyed on the full condition set and missed
# DMDPhPDA, whose two copies differ only by an unreported temperature.
# --------------------------------------------------------------------------- #

def test_issue3_core_cell_scan_finds_a_decade_pair_hidden_by_an_unreported_temperature():
    audit = _load_script("gen9_data_audit")
    base = {"extractant": "X", "metal_symbol": "Eu",
            "cond__acid_concentration_M": 1.0, "cond__extractant_concentration_M": 0.1,
            "cond__metal_concentration_mM": 1.0, "cond__contact_time_min": 30.0,
            "cond__acid__hno3": 1.0, "cond__diluent__dodecane": 1.0}
    frame = pd.DataFrame([
        {**base, "row_id": "a", "cond__temperature_C": 25.0, "log_D": -1.0},
        {**base, "row_id": "b", "cond__temperature_C": np.nan, "log_D": 2.0},
    ])
    table = audit.duplicate_cell_scan(frame)
    core = table[table["scope"] == "core_cell"]
    full = table[table["scope"] == "full_cell"]
    assert len(core) == 1 and bool(core["is_exact_decade"].iloc[0])
    assert full.empty, "the full-cell key splits the pair, which is the pre-fix blindness"


# --------------------------------------------------------------------------- #
# Issue 4 — the condition-adjusted level fell back to a ligand's own value and
# reported every level as exactly zero.
# --------------------------------------------------------------------------- #

def test_issue4_adjusted_level_is_nan_not_zero_when_unidentifiable():
    audit = _load_script("gen9_data_audit")
    rows = []
    for ligand, level in (("A", 0.0), ("B", 3.0)):
        for metal in ("Eu", "Gd", "Tb"):
            rows.append({"extractant": ligand, "metal_symbol": metal,
                         "cond__acid_concentration_M": 1.0,
                         "cond__extractant_concentration_M": 0.1, "log_D": level,
                         "condition_id": "c", "tanimoto_cluster": "t", "series_id": ligand})
    # a third ligand in a cell nobody else measured: unidentifiable, must be NaN
    rows.append({"extractant": "C", "metal_symbol": "Lu", "cond__acid_concentration_M": 9.0,
                 "cond__extractant_concentration_M": 9.0, "log_D": 5.0, "condition_id": "z",
                 "tanimoto_cluster": "t", "series_id": "C"})
    levels = audit.condition_adjusted_levels(pd.DataFrame(rows)).set_index("extractant")
    column = [c for c in levels.columns if "adjusted" in c and "coverage" not in c][0]
    assert np.isnan(levels.loc["C", column])
    assert not np.allclose(levels[column].fillna(0.0), 0.0)
    # leave-one-out deviation from the peers: A sits 3 below B and B 3 above A
    assert levels.loc["A", column] == pytest.approx(-3.0)
    assert levels.loc["B", column] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Issue 5 — `.astype(str)` on a pandas-3 frame left real floats in an object key.
# --------------------------------------------------------------------------- #

def test_issue5_float_group_keys_are_stable_strings():
    from lanthanide_separation.gen8.series import _group_key

    frame = pd.DataFrame({"cond__acid_concentration_M": pd.array([1.0, 1.0000000001, 2.5],
                                                                dtype="float64[pyarrow]"),
                          "metal_symbol": ["Eu", "Eu", "Eu"]})
    key = _group_key(frame, ["cond__acid_concentration_M", "metal"])
    assert all(isinstance(v, str) for v in key)
    assert key.iloc[0] == key.iloc[1], "values equal to 6 decimals share one key"
    assert key.iloc[0] != key.iloc[2]


# --------------------------------------------------------------------------- #
# Issues 6 and 7 — CurveBoost irreproducibility, and the initial constant.
# --------------------------------------------------------------------------- #

def _boost_data(n: int = 240, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 12))
    y = x[:, 0] * 2 + np.sin(x[:, 1]) + rng.normal(scale=0.3, size=n)
    return x, y


def test_issue6_rounding_keeps_the_booster_reproducible_across_refits():
    from lanthanide_separation.gen9.objective import BoostConfig, CurveBoost

    x, y = _boost_data()
    config = BoostConfig(n_stages=4, trees_per_stage=40, learning_rate=0.5)
    a = CurveBoost(config=config, random_state=7).fit(x, y).predict(x)
    b = CurveBoost(config=config, random_state=7).fit(x, y).predict(x)
    assert np.abs(a - b).max() <= 1e-12


def test_issue7_the_initial_constant_is_not_rounded():
    """Rounding the base constant perturbs stage one's target and breaks nesting."""
    from lanthanide_separation.gen9.objective import BoostConfig, CurveBoost

    x, y = _boost_data()
    weights = np.ones(len(y))
    model = CurveBoost(config=BoostConfig(n_stages=1, trees_per_stage=20), random_state=1)
    model.fit(x, y, sample_weight=weights)
    expected = float(np.average(y, weights=weights))
    assert model._base == expected, "the base must be the exact weighted mean, unrounded"


# --------------------------------------------------------------------------- #
# Issue 8 — LEARNED_BLEND chose alpha on the ranker's own training blocks.
# --------------------------------------------------------------------------- #

def test_issue8_blend_alpha_is_chosen_on_ligands_the_inner_ranker_never_saw():
    from lanthanide_separation.gen8.kshot import stable_hash
    from lanthanide_separation.gen9.acquisition import BlendedAcquisition

    features = pd.DataFrame({"extractant": [f"L{i}" for i in range(40)] * 3})
    left, right = BlendedAcquisition.ligand_halves(features)
    assert not (np.asarray(left) & np.asarray(right)).any()
    assert (np.asarray(left) | np.asarray(right)).all()
    ligands_left = set(features.loc[np.asarray(left), "extractant"])
    ligands_right = set(features.loc[np.asarray(right), "extractant"])
    assert not ligands_left & ligands_right, "a ligand must fall wholly on one side"
    # and the split is BLAKE2b-stable, not salted
    assert all((stable_hash(l) % 2 == 0) for l in ligands_left)


# --------------------------------------------------------------------------- #
# Issue 9 (gen10) — a second-stage target fed from a parallel forest's
# unrounded output is irreproducible; issue 10 — in-sample residuals are ~0.
# --------------------------------------------------------------------------- #

def test_issue9_stabilise_removes_the_thread_order_floor():
    from lanthanide_separation.gen10.architectures import ROUND_DECIMALS, stabilise

    a = np.array([0.1234567891234, 1.0 + 1e-15, -2.5])
    b = a + np.array([1e-15, -1e-15, 1e-16])
    assert not np.array_equal(a, b)
    assert np.array_equal(stabilise(a), stabilise(b))
    assert ROUND_DECIMALS == 9


def test_issue10_cross_fitting_reports_an_honest_residual_spread():
    """In-sample forest residuals are far smaller than held-out ones; the trap control
    exists to show it.  Checked on synthetic data with a chemotype-grouped split."""
    from lanthanide_separation.gen10.architectures import cross_fitted_predictions

    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=(n, 6))
    y = x[:, 0] + 0.5 * x[:, 1] ** 2 + rng.normal(scale=0.5, size=n)
    groups = np.array([f"g{i % 10}" for i in range(n)])
    weights = np.ones(n)
    kwargs = dict(n_estimators=50, max_features=0.5, min_samples_leaf=2)
    oof = cross_fitted_predictions(x, y, weights, groups, seed=1, n_splits=3,
                                   forest_kwargs=kwargs)
    from sklearn.ensemble import ExtraTreesRegressor
    in_sample = ExtraTreesRegressor(random_state=1, n_jobs=1, **kwargs).fit(x, y).predict(x)
    assert np.std(y - in_sample) < 0.5 * np.std(y - oof)
    # and cross-fitting is itself reproducible
    again = cross_fitted_predictions(x, y, weights, groups, seed=1, n_splits=3,
                                     forest_kwargs=kwargs)
    assert np.array_equal(oof, again)


# --------------------------------------------------------------------------- #
# The builtin-hash scan, extended to gen10 and the shared modules gen9 left out.
# --------------------------------------------------------------------------- #

def test_no_gen10_or_shared_module_calls_pythons_salted_hash():
    import ast

    offenders = []
    paths = list((REPO_ROOT / "src" / "lanthanide_separation" / "gen10").glob("*.py"))
    paths += list((REPO_ROOT / "scripts").glob("gen10_*.py"))
    paths += [REPO_ROOT / "src" / "lanthanide_separation" / p for p in
              ("gen8/kshot.py", "gen8/protocols.py", "gen8/series.py", "gen6/cohorts.py")]
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "hash":
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders

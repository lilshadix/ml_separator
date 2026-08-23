"""The curve objective: does it compute what the formulas say, and can it leak?

Two families of test, and they are asking different things.

The *mathematical* tests use toy curves whose answer is known by hand, because the
whole gen9 argument rests on one property — the delta and span terms must be blind
to a curve's level and sensitive to its shape.  If a constant vertical offset moved
the delta loss at all, the objective would be quietly re-learning the level that
six generations of evidence say is not learnable, and the interpretation of every
downstream number would change.

The *leakage* tests take the position the follow-up brief asks for: assume the
implementation is wrong.  A training pair with one endpoint in the test partition
would put a held-out target directly into a training gradient, and nothing about
the resulting run would look unusual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen9.curves import (
    AXIS_SETS, DEFAULT_AXES, MIN_ABSCISSA_GAP, SAMPLERS, assert_pairs_inside,
    build_pairs, contribution_report,
)
from lanthanide_separation.gen9.objective import BoostConfig, CurveBoost


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def membership_frame(spec: dict[str, list[tuple[str, float]]], axis: str = "extractant",
                     series: str = "S1") -> pd.DataFrame:
    rows = []
    for curve_id, points in spec.items():
        for row_id, value in points:
            rows.append({"curve_id": curve_id, "series_id": series,
                         "axis": "cond__extractant_concentration_M", "axis_label": axis,
                         "row_id": row_id, "axis_value": float(value),
                         "n_points": len(points)})
    return pd.DataFrame.from_records(rows)


@pytest.fixture
def straight_line():
    """x = [0,1,2,3], y = 1 + 2x.  True slope 2, true span 6."""
    ids = ["r0", "r1", "r2", "r3"]
    membership = membership_frame({"c0": list(zip(ids, [0.0, 1.0, 2.0, 3.0]))})
    y = np.array([1.0, 3.0, 5.0, 7.0])
    return ids, membership, y


def make_pairs(membership, ids, strategy="ROW_ADJACENT", **kwargs):
    return build_pairs(membership, ids, strategy=strategy,
                       axes=("cond__extractant_concentration_M",),
                       rng=np.random.default_rng(0), **kwargs)


def delta_loss(model: CurveBoost, F, y, pairs):
    return model._delta_term(np.asarray(F, float), np.asarray(y, float), pairs, len(y))[1]


def span_loss(model: CurveBoost, F, y, pairs):
    return model._span_term(np.asarray(F, float), np.asarray(y, float), pairs, len(y))[1]


# --------------------------------------------------------------------------- #
# 10. the loss is mathematically what it claims to be
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("strategy", ["ROW_ADJACENT", "ROW_ENDPOINT", "ROW_RANDOM_PAIR",
                                      "ROW_MULTISCALE"])
def test_perfect_prediction_has_zero_shape_loss(straight_line, strategy):
    ids, membership, y = straight_line
    pairs = make_pairs(membership, ids, strategy)
    model = CurveBoost(lambda_delta=1.0, lambda_span=1.0)
    assert delta_loss(model, y, y, pairs) == pytest.approx(0.0, abs=1e-12)
    assert span_loss(model, y, y, pairs) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("strategy", ["ROW_ADJACENT", "ROW_ENDPOINT", "ROW_RANDOM_PAIR",
                                      "ROW_MULTISCALE"])
@pytest.mark.parametrize("offset", [-4.0, -0.3, 0.7, 12.0])
def test_a_constant_offset_costs_nothing(straight_line, strategy, offset):
    """The property the whole gen9 objective exists for.

    A per-curve constant is exactly what one measurement supplies, so the training
    objective must not spend capacity on it.
    """
    ids, membership, y = straight_line
    pairs = make_pairs(membership, ids, strategy)
    model = CurveBoost(lambda_delta=1.0, lambda_span=1.0)
    assert delta_loss(model, y + offset, y, pairs) == pytest.approx(0.0, abs=1e-12)
    assert span_loss(model, y + offset, y, pairs) == pytest.approx(0.0, abs=1e-12)


def test_a_wrong_slope_costs_something(straight_line):
    ids, membership, y = straight_line
    pairs = make_pairs(membership, ids, "ROW_ADJACENT")
    model = CurveBoost(lambda_delta=1.0)
    flat = np.full_like(y, y.mean())
    assert delta_loss(model, flat, y, pairs) > 0.1


def test_a_flatter_prediction_costs_more_than_a_nearly_right_one(straight_line):
    ids, membership, y = straight_line
    pairs = make_pairs(membership, ids, "ROW_MULTISCALE")
    model = CurveBoost(lambda_delta=1.0)
    x = np.array([0.0, 1.0, 2.0, 3.0])
    nearly = 1.0 + 1.9 * x
    flat = np.full_like(y, y.mean())
    assert delta_loss(model, nearly, y, pairs) < delta_loss(model, flat, y, pairs)


def test_a_reversed_curve_is_penalised_heavily(straight_line):
    ids, membership, y = straight_line
    pairs = make_pairs(membership, ids, "ROW_ADJACENT")
    model = CurveBoost(lambda_delta=1.0)
    reversed_curve = y[::-1].copy()
    assert delta_loss(model, reversed_curve, y, pairs) > delta_loss(
        model, np.full_like(y, y.mean()), y, pairs)


def test_span_loss_measures_the_range_and_not_the_slope(straight_line):
    """A curve with the right range in the wrong order still satisfies the span term.

    Stated as a test because it is the reason the span term is a *supplement* to
    the delta term and never a replacement for it.
    """
    ids, membership, y = straight_line
    pairs = make_pairs(membership, ids, "ROW_ADJACENT")
    model = CurveBoost(lambda_span=1.0)
    assert span_loss(model, y[::-1].copy(), y, pairs) == pytest.approx(0.0, abs=1e-12)
    compressed = y.mean() + 0.1 * (y - y.mean())
    assert span_loss(model, compressed, y, pairs) > 0.5


def test_delta_gradient_sums_to_zero_within_a_curve(straight_line):
    """The delta term cannot move a curve's mean — its gradient is level-neutral."""
    ids, membership, y = straight_line
    pairs = make_pairs(membership, ids, "ROW_MULTISCALE")
    model = CurveBoost(lambda_delta=1.0)
    gradient, _ = model._delta_term(np.zeros(4), y, pairs, 4)
    assert gradient.sum() == pytest.approx(0.0, abs=1e-12)


def test_span_gradient_sums_to_zero_within_a_curve(straight_line):
    ids, membership, y = straight_line
    pairs = make_pairs(membership, ids, "ROW_ADJACENT")
    model = CurveBoost(lambda_span=1.0)
    gradient, _ = model._span_term(np.zeros(4), y, pairs, 4)
    assert gradient.sum() == pytest.approx(0.0, abs=1e-12)


def test_lambda_zero_means_no_curve_gradient(straight_line):
    ids, membership, y = straight_line
    pairs = make_pairs(membership, ids, "ROW_MULTISCALE")
    model = CurveBoost(lambda_delta=0.0, lambda_span=0.0)
    for term in (model._delta_term, model._span_term):
        gradient, value = term(np.zeros(4), y, pairs, 4)
        assert np.all(gradient == 0.0) and value == 0.0


# --------------------------------------------------------------------------- #
# 7/9. axis transforms and duplicate coordinates
# --------------------------------------------------------------------------- #

def test_the_supervised_axes_are_the_ones_declared():
    assert DEFAULT_AXES == AXIS_SETS["LHM"]
    assert "cond__contact_time_min" not in DEFAULT_AXES
    assert "cond__temperature_C" not in DEFAULT_AXES
    assert "cond__contact_time_min" in AXIS_SETS["ALL"]


def test_curve_membership_carries_log10_concentrations():
    """The abscissa gen9 trains on must be the mass-action scale, not the linear one."""
    from lanthanide_separation.gen8.series import _axis_values

    frame = pd.DataFrame({
        "cond__extractant_concentration_M": [0.01, 0.1, 1.0],
        "lanthanide_index": [1.0, 2.0, 3.0], "metal_symbol": ["La", "Ce", "Pr"],
        "cond__temperature_C": [25.0, 30.0, 35.0]})
    np.testing.assert_allclose(
        _axis_values(frame, "cond__extractant_concentration_M"), [-2.0, -1.0, 0.0])
    # temperature stays on its natural scale
    np.testing.assert_allclose(_axis_values(frame, "cond__temperature_C"), [25.0, 30.0, 35.0])


def test_duplicate_coordinates_never_become_a_training_pair():
    ids = ["r0", "r1", "r2", "r3", "r4"]
    membership = membership_frame({"c0": [("r0", 0.0), ("r1", 0.0), ("r2", 1.0),
                                          ("r3", 2.0), ("r4", 2.0)]})
    for strategy in ("ROW_ADJACENT", "ROW_RANDOM_PAIR", "ROW_MULTISCALE", "ROW_ENDPOINT"):
        pairs = make_pairs(membership, ids, strategy)
        if len(pairs) == 0:
            continue
        gaps = pairs.pair_gap
        assert gaps.min() >= MIN_ABSCISSA_GAP, (
            f"{strategy} emitted a pair whose two rows share an abscissa; its true "
            "delta is replicate noise and its predicted delta is identically zero")


def test_a_curve_with_too_few_distinct_points_is_dropped():
    ids = ["r0", "r1", "r2"]
    membership = membership_frame({"c0": [("r0", 1.0), ("r1", 1.0), ("r2", 1.0)]})
    pairs = make_pairs(membership, ids, "ROW_MULTISCALE")
    assert len(pairs) == 0
    assert pairs.audit["n_curves"] == 0


def test_a_non_finite_abscissa_never_enters_a_pair():
    ids = ["r0", "r1", "r2", "r3"]
    membership = membership_frame({"c0": [("r0", 0.0), ("r1", np.nan),
                                          ("r2", 1.0), ("r3", 2.0)]})
    pairs = make_pairs(membership, ids, "ROW_MULTISCALE")
    assert len(pairs) > 0
    assert 1 not in set(pairs.pairs.reshape(-1).tolist())


# --------------------------------------------------------------------------- #
# 1/2. leakage
# --------------------------------------------------------------------------- #

def test_only_partition_rows_can_appear_in_a_pair():
    """A row outside the training partition must be unreachable, not merely unused."""
    all_ids = ["r0", "r1", "r2", "r3", "r4", "r5"]
    membership = membership_frame({"c0": list(zip(all_ids, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))})
    train = ["r0", "r1", "r2", "r3"]
    pairs = make_pairs(membership, train, "ROW_MULTISCALE")
    assert_pairs_inside(pairs, len(train))
    assert pairs.pairs.max() < len(train)


def test_a_curve_split_across_the_boundary_contributes_only_its_training_half():
    ids = ["r0", "r1", "r2", "r3", "r4", "r5"]
    membership = membership_frame({"c0": list(zip(ids, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))})
    train = ["r0", "r2", "r4"]                       # a deliberately interleaved split
    pairs = make_pairs(membership, train, "ROW_MULTISCALE")
    assert len(pairs) > 0
    assert_pairs_inside(pairs, len(train))
    # every emitted index addresses one of the three training rows
    assert set(pairs.pairs.reshape(-1).tolist()) <= {0, 1, 2}


def test_fitting_rejects_a_pair_that_addresses_a_missing_row():
    from lanthanide_separation.gen9.curves import CurvePairs

    bad = CurvePairs(pairs=np.array([[0, 9]]), pair_weight=np.array([1.0]),
                     pair_curve=np.array([0]), pair_axis=np.array(["extractant"], dtype=object),
                     pair_gap=np.array([1.0]), curves=(("c", "extractant", "L"),),
                     curve_members=(np.array([0, 1]),), curve_weight=np.array([1.0]), audit={})
    model = CurveBoost(config=BoostConfig(n_stages=1, trees_per_stage=4), lambda_delta=1.0)
    with pytest.raises(AssertionError, match="outside the training partition"):
        model.fit(np.zeros((4, 2)), np.zeros(4), pairs=bad)


def test_row_only_builds_no_pairs_at_all():
    ids = ["r0", "r1", "r2", "r3"]
    membership = membership_frame({"c0": list(zip(ids, [0.0, 1.0, 2.0, 3.0]))})
    pairs = make_pairs(membership, ids, "ROW_ONLY")
    assert len(pairs) == 0 and pairs.n_curves == 0


# --------------------------------------------------------------------------- #
# 12. no curve may dominate by being long
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("strategy", [s for s in SAMPLERS if s != "ROW_ONLY"])
def test_a_long_curve_does_not_outweigh_a_short_one(strategy):
    short = [(f"s{i}", float(i)) for i in range(4)]
    long = [(f"l{i}", float(i)) for i in range(20)]
    membership = membership_frame({"short": short, "long": long})
    ids = [r for r, _ in short + long]
    pairs = make_pairs(membership, ids, strategy)
    report = contribution_report(pairs)
    weights = report.set_index("curve_id")["weight"]
    assert weights["short"] == pytest.approx(weights["long"], rel=1e-9), (
        "a 20-point titration is outvoting a 4-point one; the resulting model would be "
        "good at densely sampled curves rather than good at curve shape")


def test_ligand_balance_equalises_ligands_not_curves():
    spec = {"a1": [(f"a1_{i}", float(i)) for i in range(4)],
            "a2": [(f"a2_{i}", float(i)) for i in range(4)],
            "b1": [(f"b1_{i}", float(i)) for i in range(4)]}
    membership = membership_frame(spec)
    ids = [r for points in spec.values() for r, _ in points]
    ligand_of_row = {r: ("A" if r.startswith("a") else "B") for r in ids}
    pairs = build_pairs(membership, ids, strategy="ROW_ADJACENT",
                        axes=("cond__extractant_concentration_M",),
                        rng=np.random.default_rng(0), ligand_of_row=ligand_of_row,
                        balance="ligand")
    report = contribution_report(pairs)
    per_ligand = report.groupby("ligand")["weight"].sum()
    assert per_ligand["A"] == pytest.approx(per_ligand["B"], rel=1e-9)


# --------------------------------------------------------------------------- #
# 11. loss balance is observable
# --------------------------------------------------------------------------- #

def _toy_problem(n_curves=12, points=5, seed=0):
    rng = np.random.default_rng(seed)
    rows, y, x = [], [], []
    for c in range(n_curves):
        level = rng.normal(0, 3)
        slope = rng.normal(2.0, 0.3)
        for p in range(points):
            row_id = f"c{c}_p{p}"
            rows.append({"curve_id": f"c{c}", "series_id": f"s{c}",
                         "axis": "cond__extractant_concentration_M",
                         "axis_label": "extractant", "row_id": row_id,
                         "axis_value": float(p), "n_points": points})
            y.append(level + slope * p + rng.normal(0, 0.05))
            x.append([float(p), float(c), rng.normal()])
    return ([r["row_id"] for r in rows], pd.DataFrame.from_records(rows),
            np.asarray(y), np.asarray(x))


def test_loss_history_records_every_term():
    ids, membership, y, x = _toy_problem()
    pairs = make_pairs(membership, ids, "ROW_MULTISCALE")
    model = CurveBoost(config=BoostConfig(n_stages=6, trees_per_stage=8, learning_rate=0.3, max_depth=4),
                       lambda_delta=0.5, lambda_span=0.1)
    model.fit(x, y, pairs=pairs)
    history = model.history
    assert history is not None and len(history) >= 2
    for column in ("L_row", "L_delta_weighted", "L_span_weighted",
                   "grad_row_absmean", "grad_delta_absmean", "grad_span_absmean"):
        assert column in history.columns
    assert history["L_delta_weighted"].iloc[0] > 0
    assert history["L_row"].iloc[-1] < history["L_row"].iloc[0]


def test_the_curve_term_actually_steepens_predictions():
    """The mechanism, on data whose answer is known: true slope 2 per unit x."""
    ids, membership, y, x = _toy_problem(n_curves=24, points=6, seed=3)
    pairs = make_pairs(membership, ids, "ROW_MULTISCALE")
    config = BoostConfig(n_stages=12, trees_per_stage=12, learning_rate=0.3,
                         max_depth=6, max_features=1.0, min_samples_leaf=1)
    flat = CurveBoost(config=config, lambda_delta=0.0).fit(x, y).predict(x)
    shaped = CurveBoost(config=config, lambda_delta=1.0).fit(x, y, pairs=pairs).predict(x)

    def median_slope(values):
        slopes = []
        for _, block in membership.groupby("curve_id"):
            index = [ids.index(r) for r in block["row_id"]]
            slopes.append(np.polyfit(block["axis_value"].to_numpy(dtype=float),
                                     values[index], 1)[0])
        return float(np.median(slopes))

    assert median_slope(shaped) > median_slope(flat)


def test_predictions_are_deterministic_for_a_fixed_seed():
    """Reproducible to float noise with threads, bit-identical without them.

    ``n_jobs=-1`` lets sklearn sum a forest's trees in whatever order the threads
    finish, which moves the last bit or two.  That is harmless for every gen9
    comparison — both arms are scored on the same rows and the bootstrap resamples
    chemotypes — but it is worth pinning as a *known* tolerance rather than
    discovering it later as an unexplained diff between two reruns.
    """
    ids, membership, y, x = _toy_problem()
    pairs = make_pairs(membership, ids, "ROW_MULTISCALE")
    config = BoostConfig(n_stages=4, trees_per_stage=8, max_depth=4)
    a = CurveBoost(config=config, lambda_delta=0.5, random_state=7).fit(x, y, pairs=pairs).predict(x)
    b = CurveBoost(config=config, lambda_delta=0.5, random_state=7).fit(x, y, pairs=pairs).predict(x)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    serial = BoostConfig(n_stages=4, trees_per_stage=8, max_depth=4, n_jobs=1)
    c = CurveBoost(config=serial, lambda_delta=0.5, random_state=7).fit(x, y, pairs=pairs).predict(x)
    d = CurveBoost(config=serial, lambda_delta=0.5, random_state=7).fit(x, y, pairs=pairs).predict(x)
    np.testing.assert_array_equal(c, d)


def test_pair_construction_is_deterministic_for_a_fixed_rng():
    ids, membership, _, _ = _toy_problem()
    a = build_pairs(membership, ids, strategy="ROW_MULTISCALE",
                    axes=("cond__extractant_concentration_M",), rng=np.random.default_rng(11))
    b = build_pairs(membership, ids, strategy="ROW_MULTISCALE",
                    axes=("cond__extractant_concentration_M",), rng=np.random.default_rng(11))
    np.testing.assert_array_equal(a.pairs, b.pairs)
    np.testing.assert_allclose(a.pair_weight, b.pair_weight)


# --------------------------------------------------------------------------- #
# The control really is the frozen model
# --------------------------------------------------------------------------- #

def test_one_stage_row_only_is_algebraically_the_frozen_extratrees():
    """``FROZEN_EQUIVALENT`` must be ``REC_ecfp_plus_recovered``, not a lookalike.

    gen9's whole attribution rests on the control being the *same model* as gen8's,
    so that a difference between arms is a difference in objective.  Asserting the
    nesting on real-shaped data is the only way that claim stays true when someone
    later changes a default.
    """
    from sklearn.ensemble import ExtraTreesRegressor

    from lanthanide_separation.gen9.objective import FROZEN_EQUIVALENT

    rng = np.random.default_rng(0)
    n, p = 240, 12
    x = rng.normal(size=(n, p))
    y = x[:, 0] * 2.0 - x[:, 3] + rng.normal(scale=0.4, size=n)
    weight = rng.uniform(0.5, 2.0, size=n)
    normalised = weight / weight.sum() * n

    boosted = CurveBoost(config=FROZEN_EQUIVALENT, lambda_delta=0.0, random_state=11)
    boosted.fit(x, y, sample_weight=weight)

    seed = int(np.random.default_rng(11).integers(1, 2 ** 31 - 1))
    base = float(np.average(y, weights=normalised))
    reference = ExtraTreesRegressor(
        n_estimators=FROZEN_EQUIVALENT.trees_per_stage,
        max_features=FROZEN_EQUIVALENT.max_features,
        min_samples_leaf=FROZEN_EQUIVALENT.min_samples_leaf,
        max_depth=FROZEN_EQUIVALENT.max_depth, random_state=seed, n_jobs=-1)
    reference.fit(x, y - base, sample_weight=normalised)

    span = float(y.max() - y.min())
    expected = np.clip(base + reference.predict(x), y.min() - 0.5 * span, y.max() + 0.5 * span)
    # 1e-8 rather than exact because ``CurveBoost`` rounds its running prediction to
    # DETERMINISM_DECIMALS = 9 places (see the module docstring) and the reference
    # here does not.  The nesting is exact up to that rounding, which is nine orders
    # of magnitude below any measurement in the corpus.
    np.testing.assert_allclose(boosted.predict(x), expected, rtol=0, atol=1e-8)


def test_the_span_term_is_skipped_when_a_curve_is_predicted_perfectly_flat():
    """A degenerate curve must not produce a NaN or an infinite step."""
    ids = ["r0", "r1", "r2", "r3"]
    membership = membership_frame({"c0": list(zip(ids, [0.0, 1.0, 2.0, 3.0]))})
    pairs = make_pairs(membership, ids, "ROW_ADJACENT")
    model = CurveBoost(lambda_span=1.0)
    gradient, value = model._span_term(np.zeros(4), np.array([1.0, 3.0, 5.0, 7.0]), pairs, 4)
    assert np.all(np.isfinite(gradient)) and np.isfinite(value)
    assert np.all(gradient == 0.0), "a perfectly flat prediction has no arg-extremum to move"


# --------------------------------------------------------------------------- #
# balance modes
# --------------------------------------------------------------------------- #

def test_cluster_balance_needs_the_row_weights_it_is_named_after():
    ids = ["r0", "r1", "r2", "r3"]
    membership = membership_frame({"c0": list(zip(ids, [0.0, 1.0, 2.0, 3.0]))})
    with pytest.raises(ValueError, match="weight_of_row"):
        build_pairs(membership, ids, strategy="ROW_ADJACENT",
                    axes=("cond__extractant_concentration_M",),
                    rng=np.random.default_rng(0), balance="cluster")


def test_cluster_balance_weights_a_curve_by_its_rows_own_row_weight():
    """The point of the mode: the two loss terms speak on the same scale per row."""
    rare = [(f"rare{i}", float(i)) for i in range(4)]
    common = [(f"common{i}", float(i)) for i in range(4)]
    membership = membership_frame({"rare": rare, "common": common})
    ids = [r for r, _ in rare + common]
    weights = {r: (8.0 if r.startswith("rare") else 0.5) for r in ids}
    pairs = build_pairs(membership, ids, strategy="ROW_ADJACENT",
                        axes=("cond__extractant_concentration_M",),
                        rng=np.random.default_rng(0), weight_of_row=weights,
                        balance="cluster")
    report = contribution_report(pairs).set_index("curve_id")["weight"]
    assert report["rare"] == pytest.approx(16.0 * report["common"], rel=1e-9)


def test_an_unknown_balance_mode_is_refused():
    ids = ["r0", "r1", "r2", "r3"]
    membership = membership_frame({"c0": list(zip(ids, [0.0, 1.0, 2.0, 3.0]))})
    with pytest.raises(ValueError, match="balance must be"):
        build_pairs(membership, ids, strategy="ROW_ADJACENT",
                    axes=("cond__extractant_concentration_M",),
                    rng=np.random.default_rng(0), balance="nonsense")


# --------------------------------------------------------------------------- #
# the shape diagnostics
# --------------------------------------------------------------------------- #

def test_linearity_and_step_diagnostics_separate_a_line_from_a_step():
    """The two ways a prediction can have the right range, told apart.

    Both predictions below span exactly 6 log units on the same abscissae; one is
    the true line and one is a single step in the middle.  Slope and span cannot
    distinguish them.  ``linear_r2_pred`` and ``n_distinct_pred`` can, which is why
    they exist — a model that buys range with one tall step is not a model that
    learned the response.
    """
    from lanthanide_separation.gen9.metrics import curve_shape_table

    ids = [f"r{i}" for i in range(6)]
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    membership = membership_frame({"c0": list(zip(ids, x))})
    truth = np.array([1.0, 2.2, 3.4, 4.6, 5.8, 7.0])
    step = np.array([1.0, 1.0, 1.0, 7.0, 7.0, 7.0])
    frame = pd.DataFrame({"row_id": ids, "log_D": truth, "prediction": step,
                          "extractant": "L", "tanimoto_cluster": "t", "split_seed": 1})
    table = curve_shape_table(frame, membership)
    row = table.iloc[0]
    assert row["span_pred"] == pytest.approx(row["span_true"], abs=0.05)
    assert row["n_distinct_pred"] == 2 and row["n_distinct_true"] == 6
    assert row["linear_r2_true"] > 0.999
    assert row["linear_r2_pred"] < 0.9, "a single step must not look like a straight line"

    smooth = pd.DataFrame({"row_id": ids, "log_D": truth, "prediction": truth,
                           "extractant": "L", "tanimoto_cluster": "t", "split_seed": 1})
    perfect = curve_shape_table(smooth, membership).iloc[0]
    assert perfect["linear_r2_pred"] > 0.999
    assert perfect["n_distinct_pred"] == 6


# --------------------------------------------------------------------------- #
# The adversarial leakage test the follow-up brief asks for (§4)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_corrupting_every_held_out_target_cannot_move_a_gen9_prediction():
    """Replace the held-out fold's targets with nonsense; refit; demand identity.

    This is the strongest statement available about gen9-A, and it is stronger than
    inspecting the code: the contender is handed the *whole cohort frame* through
    ``FoldContext``, and it builds its curve pairs from a membership table computed
    over every row in the corpus.  Either of those is a route by which a held-out
    target could reach a training gradient.  Corrupting the held-out targets and
    demanding bit-identical predictions closes both at once.

    Deliberately corrupts with values 1e4 apart from the real ones: a leak of any
    size would move the prediction by an amount no tolerance could hide.
    """
    from lanthanide_separation.gen7.harness import FoldContext, build_folds, load_cohort
    from lanthanide_separation.gen9.objective import ShapeArm
    from lanthanide_separation.gen9.train import CurveBoostContender, load_membership
    from lanthanide_separation.levels import LEVEL_TARGET_COLUMN

    cohort = load_cohort()
    membership = load_membership()
    frame = cohort.frame
    target = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    fold = build_folds(frame, 104729)[0]

    # small enough to run in a test, identical between the two calls
    config = BoostConfig(n_stages=2, trees_per_stage=8, learning_rate=0.3,
                         max_features=0.05, n_jobs=1)
    arm = ShapeArm("LEAK_PROBE", "ROW_MULTISCALE", 1.0, 0.1)

    def predict(frame_in, target_in):
        train = frame_in.iloc[fold.train_index]
        test = frame_in.iloc[fold.test_index].drop(columns=[LEVEL_TARGET_COLUMN])
        contender = CurveBoostContender(arm=arm, config=config, membership=membership)
        context = FoldContext(fold=fold, cohort=cohort, feature_columns=(),
                              model_seed=fold.model_seed)
        return contender.fit_predict(train, target_in[fold.train_index], test, context)

    clean = predict(frame, target)

    poisoned_frame = frame.copy()
    poisoned_target = target.copy()
    rng = np.random.default_rng(0)
    poisoned_target[fold.test_index] = rng.normal(1e4, 1e3, size=fold.test_index.size)
    poisoned_frame[LEVEL_TARGET_COLUMN] = poisoned_target
    dirty = predict(poisoned_frame, poisoned_target)

    np.testing.assert_array_equal(clean, dirty)


# --------------------------------------------------------------------------- #
# Determinism under parallel forests
# --------------------------------------------------------------------------- #

def test_parallel_forests_do_not_make_the_booster_irreproducible():
    """The defect this rounding exists to fix, pinned so it cannot come back.

    ``ExtraTreesRegressor(n_jobs=-1)`` sums its trees in thread-completion order, so
    a single forest's predictions move by ~1e-15 between runs.  Inside a boosting
    recursion that perturbation reaches the next stage's pseudo-residual, flips a
    near-tied split comparison, and the stage grows a *different tree*; on the real
    cohort two runs of the identical configuration diverged by 0.139 log units on
    every held-out row.  Rounding the running prediction to 1e-9 removes it.
    """
    rng = np.random.default_rng(4)
    n, p = 600, 40
    x = rng.normal(size=(n, p))
    y = x[:, 0] * 2.0 + np.sin(x[:, 1]) + rng.normal(scale=0.5, size=n)
    weight = rng.uniform(0.2, 5.0, size=n)

    parallel = BoostConfig(n_stages=6, trees_per_stage=24, learning_rate=0.3, n_jobs=-1)
    a = CurveBoost(config=parallel, random_state=3).fit(x, y, sample_weight=weight).predict(x)
    b = CurveBoost(config=parallel, random_state=3).fit(x, y, sample_weight=weight).predict(x)
    np.testing.assert_array_equal(a, b)

    serial = BoostConfig(n_stages=6, trees_per_stage=24, learning_rate=0.3, n_jobs=1)
    c = CurveBoost(config=serial, random_state=3).fit(x, y, sample_weight=weight).predict(x)
    np.testing.assert_array_equal(a, c), "parallel and serial must agree once rounded"


def test_the_rounding_can_be_switched_off_and_the_defect_returns():
    """Evidence that the rounding is doing the work, not that the defect was imaginary.

    Without it the arm is expected to be irreproducible under parallel forests.  The
    assertion is deliberately weak — thread scheduling is not guaranteed to differ on
    every machine — but the configuration is exercised so nobody can claim the knob
    is inert.
    """
    rng = np.random.default_rng(5)
    x = rng.normal(size=(400, 30))
    y = x[:, 0] * 2.0 + rng.normal(scale=0.5, size=400)
    config = BoostConfig(n_stages=5, trees_per_stage=24, learning_rate=0.3, n_jobs=-1,
                         round_decimals=None)
    a = CurveBoost(config=config, random_state=6).fit(x, y).predict(x)
    b = CurveBoost(config=config, random_state=6).fit(x, y).predict(x)
    assert a.shape == b.shape and np.all(np.isfinite(a))


# --------------------------------------------------------------------------- #
# The relative-position features and the recomposed arm
# --------------------------------------------------------------------------- #

def test_relative_position_features_read_no_target():
    from lanthanide_separation.gen9.relative import (
        RELATIVE_COLUMNS, relative_position_features,
    )

    ids = [f"r{i}" for i in range(6)]
    membership = membership_frame({"c0": list(zip(ids, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))})
    frame = pd.DataFrame({"row_id": ids, "log_D": np.arange(6.0)})
    clean = relative_position_features(frame, membership)
    poisoned = frame.copy()
    poisoned["log_D"] = np.random.default_rng(0).normal(1e5, 1e4, size=6)
    pd.testing.assert_frame_equal(clean, relative_position_features(poisoned, membership))
    assert list(clean.columns) == ["row_id"] + list(RELATIVE_COLUMNS)


def test_relative_position_is_zero_at_the_low_end_and_one_at_the_high_end():
    from lanthanide_separation.gen9.relative import relative_position_features

    ids = [f"r{i}" for i in range(5)]
    membership = membership_frame({"c0": list(zip(ids, [-2.0, -1.5, -1.0, -0.5, 0.0]))})
    frame = pd.DataFrame({"row_id": ids})
    table = relative_position_features(frame, membership).set_index("row_id")
    np.testing.assert_allclose(table["rel__position"], [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(table["rel__is_endpoint"], [1, 0, 0, 0, 1])
    np.testing.assert_allclose(table["rel__window_width"], 2.0)
    np.testing.assert_allclose(table["rel__n_points"], 5.0)


def test_a_row_on_no_curve_gets_zeros_rather_than_nan():
    from lanthanide_separation.gen9.relative import relative_position_features

    membership = membership_frame({"c0": [("r0", 0.0), ("r1", 1.0), ("r2", 2.0)]})
    frame = pd.DataFrame({"row_id": ["r0", "r1", "r2", "orphan"]})
    table = relative_position_features(frame, membership).set_index("row_id")
    assert np.isfinite(table.to_numpy(dtype=float)).all()
    assert table.loc["orphan"].to_numpy(dtype=float).tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.slow
def test_the_recomposed_arm_is_mean_preserving_and_leak_free():
    """Two claims at once, both load-bearing.

    *Mean-preserving*: the recomposition may replace a curve's shape but must leave
    the level the monolith assigned it untouched, because that level is the accuracy
    the monolith bought and gen9 is not trying to spend it.

    *Leak-free*: corrupting every held-out target must leave the output bit-identical,
    even though the arm reads the held-out ligand's condition geometry.
    """
    from lanthanide_separation.gen7.harness import FoldContext, build_folds, load_cohort
    from lanthanide_separation.gen9.relative import curve_membership_map
    from lanthanide_separation.gen9.train import FrozenExtraTrees, ShapeRecomposed, load_membership
    from lanthanide_separation.levels import LEVEL_TARGET_COLUMN

    cohort = load_cohort()
    membership = load_membership()
    frame = cohort.frame
    target = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    fold = build_folds(frame, 104729)[0]
    train = frame.iloc[fold.train_index]
    test = frame.iloc[fold.test_index].drop(columns=[LEVEL_TARGET_COLUMN])
    context = FoldContext(fold=fold, cohort=cohort, feature_columns=(),
                          model_seed=fold.model_seed)

    arm = ShapeRecomposed(membership=membership, n_estimators=60)
    recomposed = arm.fit_predict(train, target[fold.train_index], test, context)
    base = FrozenExtraTrees(n_estimators=60).fit_predict(
        train, target[fold.train_index], test, context)

    curve = curve_membership_map(test, membership).to_numpy()
    on_curve = pd.notna(curve)
    per_curve = pd.DataFrame({"curve": curve[on_curve], "base": base[on_curve],
                              "new": recomposed[on_curve]}).groupby("curve").mean()
    np.testing.assert_allclose(per_curve["new"], per_curve["base"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(recomposed[~on_curve], base[~on_curve], rtol=0, atol=1e-9)

    poisoned = frame.copy()
    values = target.copy()
    values[fold.test_index] = np.random.default_rng(1).normal(1e4, 1e3,
                                                             size=fold.test_index.size)
    poisoned[LEVEL_TARGET_COLUMN] = values
    dirty = ShapeRecomposed(membership=membership, n_estimators=60).fit_predict(
        poisoned.iloc[fold.train_index], values[fold.train_index],
        poisoned.iloc[fold.test_index].drop(columns=[LEVEL_TARGET_COLUMN]), context)
    # 1e-12 rather than exact for the reason DETERMINISM_DECIMALS documents: two
    # independent ``ExtraTreesRegressor(n_jobs=-1)`` fits sum their trees in
    # thread-completion order.  This arm has no boosting recursion for that noise to
    # amplify through, so it stays at 1e-15 — a leak of any size would be orders of
    # magnitude larger, since the corrupted targets sit 10^4 away.
    np.testing.assert_allclose(recomposed, dirty, rtol=0, atol=1e-12)

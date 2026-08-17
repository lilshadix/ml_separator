"""Tests for the k-shot per-extractant calibration harness.

Several tests here exist specifically to kill mutants that the first version of
the suite let through (support/query leakage, lam↔k transposition, row-weighted
"macro", selecting on all seeds, scoring every arm with the first arm's column).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation import kshot_calibration as kc
from lanthanide_separation.kshot_calibration import (
    CALIBRATOR_FORMS,
    KShotConfig,
    adjacency_rank_distance,
    apply_calibrator,
    determined_by_support,
    draw_support_order,
    fit_calibrator,
    holm_adjust,
    paired_extractant_bootstrap_delta,
    prefix_fits,
    run_kshot_study,
    select_and_confirm,
    summarise_draw_deltas,
    summarise_grid,
    summarise_per_extractant,
    transitivity_violation_rate,
)
from lanthanide_separation.pairs import LANTHANIDE_Z, PAIR_TARGET_COLUMN

METALS = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]


def _synthetic_oof(
    seed: int = 0,
    n_extractants: int = 8,
    seeds=(104729, 130363, 155921),
    n_conditions: int = 2,
    sizes: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Wide OOF frame; truth is an extractant-specific affine distortion of the prediction.

    Targets are built from per-metal scores so that within-cell transitivity holds
    exactly, as it does in the real data.
    """
    rng = np.random.default_rng(seed)
    a_e = np.exp(rng.normal(0.0, 0.6, size=n_extractants))
    b_e = rng.normal(0.0, 0.3, size=n_extractants) / 5.0
    rows = []
    for split_seed in seeds:
        for e in range(n_extractants):
            for c in range(n_conditions):
                metals = METALS if sizes is None else METALS[: sizes.get(e, len(METALS))]
                score = {m: -0.08 * LANTHANIDE_Z[m] + 0.02 * c + rng.normal(0, 0.05) for m in metals}
                noise = {m: rng.normal(0, 0.1) for m in metals}
                for i in range(len(metals)):
                    for j in range(i + 1, len(metals)):
                        ma, mb = metals[i], metals[j]
                        za, zb = LANTHANIDE_Z[ma], LANTHANIDE_Z[mb]
                        p = score[ma] - score[mb]
                        y = a_e[e] * p + b_e[e] * (zb - za) + (noise[ma] - noise[mb])
                        rows.append({
                            "pair_id": f"s{split_seed}_e{e}_c{c}_{ma}{mb}",
                            "extractant": f"SMILES{e}", "condition_id": f"e{e}cond{c}",
                            "metal_A": ma, "metal_B": mb,
                            "pair__Z_A": float(za), "pair__Z_B": float(zb),
                            PAIR_TARGET_COLUMN: y, "split_seed": split_seed, "prediction_M": p,
                        })
    return pd.DataFrame(rows)


def _small_cfg(**over) -> KShotConfig:
    base = dict(ks=(1, 2, 3, 5), lams=(0.1, 1.0, 10.0),
                policies=("random", "adjacent", "foreign"), n_draws=4, min_query=10, min_pairs=20)
    base.update(over)
    return KShotConfig(**base)


# --------------------------------------------------------------------------- #
# Closed forms
# --------------------------------------------------------------------------- #

def test_fit_calibrator_matches_lstsq_without_shrinkage():
    rng = np.random.default_rng(1)
    p = rng.normal(size=25)
    d = rng.integers(1, 14, size=25).astype(float)
    y = 1.7 * p + 0.3 - 0.05 * d + rng.normal(0, 0.1, size=25)
    a, b, rej = fit_calibrator("affine", p, y, None, 0.0)
    assert not rej and np.allclose([a, b], np.linalg.lstsq(np.column_stack([p, np.ones_like(p)]), y, rcond=None)[0], atol=1e-6)
    a, b, rej = fit_calibrator("trend", p, y, d, 0.0)
    assert not rej and np.allclose([a, b], np.linalg.lstsq(np.column_stack([p, d]), y, rcond=None)[0], atol=1e-6)
    a, _, _ = fit_calibrator("scale", p, y, None, 0.0)
    assert np.isclose(a, np.dot(p, y) / np.dot(p, p), atol=1e-8)
    _, b, _ = fit_calibrator("offset", p, y, None, 0.0)
    assert np.isclose(b, np.mean(y - p), atol=1e-8)


def test_shrinkage_pulls_every_form_toward_the_identity():
    """Regression: `trend` used to push b AWAY from 0 as lam grew, because ΔZ is ~8x larger than p."""
    rng = np.random.default_rng(2)
    p = rng.normal(0, 0.4, size=10)
    d = rng.integers(1, 14, size=10).astype(float)
    y = 3.0 * p + 0.2 * d
    for form in CALIBRATOR_FORMS:
        a0, b0, _ = fit_calibrator(form, p, y, d, 0.0)
        prev_a, prev_b = abs(a0 - 1.0), abs(b0)
        for lam in (0.1, 1.0, 10.0, 1e4):
            a, b, _ = fit_calibrator(form, p, y, d, lam)
            assert abs(a - 1.0) <= prev_a + 1e-9, (form, lam)
            assert abs(b) <= prev_b + 1e-9, (form, lam)
            prev_a, prev_b = abs(a - 1.0), abs(b)
        a, b, _ = fit_calibrator(form, p, y, d, 1e12)
        assert abs(a - 1.0) < 1e-4 and abs(b) < 1e-4


def test_degenerate_fits_are_rejected_to_identity_not_clipped():
    """Regression: clipping `a` while keeping a jointly-fitted `b` gave MAE ~1e3."""
    p = np.array([0.100, 0.1005])
    y = np.array([0.30, 0.55])
    d = np.array([3.0, 3.0001])
    for form in ("affine", "trend"):
        a, b, rej = fit_calibrator(form, p, y, d, 0.0)
        assert rej and (a, b) == (1.0, 0.0)
        pred = apply_calibrator(form, a, b, np.array([0.12]), np.array([5.0]))
        assert abs(pred[0] - 0.12) < 1e-12  # identity, not an explosion
    a, _, rej = fit_calibrator("scale", np.array([1e-6]), np.array([5.0]), None, 0.0)
    assert rej and a == 1.0


def test_trend_with_constant_delta_z_is_rejected():
    """On the `adjacent` policy every support ΔZ is 1, so the trend basis is collinear with the intercept."""
    p = np.array([0.1, 0.2, 0.3])
    y = np.array([0.15, 0.31, 0.44])
    d = np.ones(3)
    a, b, rej = fit_calibrator("trend", p, y, d, 0.3)
    assert rej or b == 0.0


def test_prefix_fits_agree_with_scalar_fits():
    rng = np.random.default_rng(3)
    p = rng.normal(size=10)
    y = 1.5 * p + 0.2 + rng.normal(0, 0.1, size=10)
    d = rng.integers(1, 14, size=10).astype(float)
    ks, lams = (1, 2, 3, 5, 10), (0.1, 1.0)
    for form in CALIBRATOR_FORMS:
        A, B, R = prefix_fits(form, p, y, d, ks, lams)
        assert A.shape == (len(lams), len(ks))
        for i, lam in enumerate(lams):
            for j, k in enumerate(ks):
                a, b, rej = fit_calibrator(form, p[:k], y[:k], d[:k], lam)
                assert np.isclose(A[i, j], a, atol=1e-9) and np.isclose(B[i, j], b, atol=1e-9)
                assert R[i, j] == int(rej)


def test_calibrators_are_antisymmetric_under_pair_swap():
    rng = np.random.default_rng(4)
    p = rng.normal(size=6)
    d = rng.integers(1, 14, size=6).astype(float)
    y = 2.0 * p + 0.1 * d
    for form in ("scale", "trend"):
        a, b, _ = fit_calibrator(form, p, y, d, 0.5)
        a_s, b_s, _ = fit_calibrator(form, -p, -y, -d, 0.5)
        assert np.isclose(a, a_s) and np.isclose(b, b_s)
        assert np.allclose(apply_calibrator(form, a, b, p, d), -apply_calibrator(form, a_s, b_s, -p, -d))


def test_only_scale_and_trend_preserve_transitivity():
    z = {"La": 57.0, "Ce": 58.0, "Pr": 59.0}
    s = {"La": 0.3, "Ce": 0.1, "Pr": -0.4}
    pairs = [("La", "Ce"), ("Ce", "Pr"), ("La", "Pr")]
    p = np.array([s[a] - s[b] for a, b in pairs])
    d = np.array([z[b] - z[a] for a, b in pairs])
    assert np.isclose(*np.array([apply_calibrator("trend", 1.7, -0.05, p, d)[0]
                                 + apply_calibrator("trend", 1.7, -0.05, p, d)[1],
                                 apply_calibrator("trend", 1.7, -0.05, p, d)[2]]))
    out = apply_calibrator("offset", 1.0, 0.2, p, d)
    assert not np.isclose(out[0] + out[1], out[2])
    frame = pd.DataFrame({"extractant": ["E"] * 3, "condition_id": ["c"] * 3,
                          "metal_A": [a for a, _ in pairs], "metal_B": [b for _, b in pairs]})
    assert transitivity_violation_rate(frame, apply_calibrator("scale", 1.7, 0.0, p, d)) == 0.0
    assert transitivity_violation_rate(frame, apply_calibrator("affine", 1.3, 0.15, p, d)) == 1.0


# --------------------------------------------------------------------------- #
# Support drawing and determination
# --------------------------------------------------------------------------- #

def test_adjacency_uses_observed_metal_ordering_not_raw_delta_z():
    """Pm is absent, so Nd-Sm (ΔZ = 2) is chemically adjacent."""
    z = np.array([60.0, 57.0])   # Nd, La
    zb = np.array([62.0, 58.0])  # Sm, Ce
    assert list(adjacency_rank_distance(z, zb)) == [1, 1]


def test_draw_support_order_policies():
    rd = np.array([1, 5, 1, 13, 2, 1, 7], dtype=float)
    perm = draw_support_order(7, "random", np.random.default_rng(5))
    assert sorted(perm) == list(range(7))
    adj = draw_support_order(7, "adjacent", np.random.default_rng(5), rank_distance=rd)
    assert set(adj[:3]) == {0, 2, 5}
    wide = draw_support_order(7, "widest", np.random.default_rng(5), rank_distance=rd)
    assert list(rd[wide]) == sorted(rd, reverse=True)
    with pytest.raises(ValueError):
        draw_support_order(7, "adjacent", np.random.default_rng(5))


def test_determined_by_support_finds_transitive_consequences():
    # cell 0: pairs (0,1), (1,2), (0,2); support = first two -> (0,2) is determined
    cell = np.array([0, 0, 0, 1])
    ma = np.array([0, 1, 0, 0])
    mb = np.array([1, 2, 2, 1])
    det = determined_by_support(cell, ma, mb, np.array([0, 1]), np.array([2, 3]), 8)
    assert list(det) == [True, False]  # same cell determined; other cell never is


# --------------------------------------------------------------------------- #
# Study loop — leakage and labelling invariants
# --------------------------------------------------------------------------- #

def test_query_is_disjoint_from_support_and_every_cell_matches_a_direct_fit(monkeypatch):
    """End-to-end recomputation with a deterministic draw order.

    Kills: support/query leakage, lam<->k transposition, wrong-arm scoring,
    sign_correct computed against the prediction instead of the truth.
    """
    monkeypatch.setattr(kc, "draw_support_order", lambda n, policy, rng, **kw: np.arange(n))
    oof = _synthetic_oof(n_extractants=3, seeds=(104729,), n_conditions=1)
    cfg = _small_cfg(policies=("random",), n_draws=1, ks=(1, 3), lams=(0.1, 1.0), include_oracle=False)
    res = run_kshot_study(oof, arms=["M"], cfg=cfg)
    df = res.per_extractant_draw
    k_max = cfg.k_max
    checked = 0
    for ext, g in oof.groupby("extractant"):
        p = g["prediction_M"].to_numpy(float)
        y = g[PAIR_TARGET_COLUMN].to_numpy(float)
        d = g["pair__Z_B"].to_numpy(float) - g["pair__Z_A"].to_numpy(float)
        pq, yq, dq = p[k_max:], y[k_max:], d[k_max:]
        sub = df[df.extractant == ext]
        assert (sub.n_query == len(pq)).all()
        for r in sub.itertuples():
            if r.form == "none":
                pred = pq
            else:
                a, b, rej = fit_calibrator(r.form, p[:r.k], y[:r.k], d[:r.k], r.lam)
                assert (a, b, int(rej)) == pytest.approx((r.a, r.b, r.fit_rejected))
                pred = apply_calibrator(r.form, a, b, pq, dq)
            assert r.sae == pytest.approx(float(np.abs(pred - yq).sum()))
            assert r.sse == pytest.approx(float(((pred - yq) ** 2).sum()))
            assert r.sign_correct == int(np.sum(np.sign(pred) == np.sign(yq)))
            checked += 1
    assert checked > 50


def test_infinite_shrinkage_reproduces_the_raw_predictions_exactly():
    """Orthogonal check on (lam, k) labelling: lam -> inf must equal the k=0 row."""
    oof = _synthetic_oof(n_extractants=3, seeds=(104729,))
    cfg = _small_cfg(policies=("random",), n_draws=2, ks=(1, 3), lams=(1e12,), include_oracle=False)
    res = run_kshot_study(oof, arms=["M"], cfg=cfg)
    df = res.per_extractant_draw
    keys = ["seed", "policy", "extractant", "draw"]
    k0 = df[df.k == 0].set_index(keys)["sae"]
    for k in (1, 3):
        big = df[(df.k == k) & (df.form == "scale")].set_index(keys)["sae"]
        assert np.allclose(big.sort_index().to_numpy(), k0.sort_index().to_numpy(), atol=1e-6)


def test_arms_are_scored_independently_on_identical_query_rows():
    oof = _synthetic_oof(n_extractants=4, seeds=(104729,))
    oof["prediction_BAD"] = 0.25 * oof["prediction_M"] + 0.9
    res = run_kshot_study(oof, arms=["M", "BAD"], cfg=_small_cfg(policies=("random",), n_draws=2, include_oracle=False))
    df = res.per_extractant_draw
    sae_m = df[(df.arm == "M") & (df.k == 0)].sae.sum()
    sae_bad = df[(df.arm == "BAD") & (df.k == 0)].sae.sum()
    assert sae_bad > 1.2 * sae_m
    q = res.per_draw_query_stats
    cols = ["seed", "policy", "extractant", "draw", "n_query", "sum_y", "sum_y2"]
    a = q[q.arm == "M"][cols].sort_values(cols).reset_index(drop=True)
    b = q[q.arm == "BAD"][cols].sort_values(cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_macro_mae_is_equal_weighted_over_extractants_not_row_weighted():
    oof = _synthetic_oof(n_extractants=4, seeds=(104729,), sizes={0: 7, 1: 9, 2: 12, 3: 14})
    sizes = oof[oof.split_seed == 104729].groupby("extractant").size()
    assert sizes.nunique() > 1
    res = run_kshot_study(oof, arms=["M"], cfg=_small_cfg(policies=("random",), n_draws=1, ks=(2,),
                                                          lams=(1.0,), forms=("scale",), include_oracle=False))
    per_ext = summarise_per_extractant(res)
    grid = summarise_grid(res)
    for k in (0, 2):
        pe = per_ext[per_ext.k == k]
        g = grid[grid.k == k].iloc[0]
        assert np.isclose(pe.mae.mean(), g.macro_mae)
        weighted = np.average(pe.mae, weights=pe.n_query)
        assert not np.isclose(weighted, g.macro_mae)
        assert np.isclose(weighted, g.pooled_mae, atol=1e-9)


def test_random_and_foreign_share_query_rows():
    oof = _synthetic_oof()
    res = run_kshot_study(oof, arms=["M"], cfg=_small_cfg())
    q = res.per_draw_query_stats
    cols = ["n_query", "sum_y", "sum_y2"]
    r = q[q.policy == "random"].set_index(["seed", "extractant", "draw"])[cols]
    f = q[q.policy == "foreign"].set_index(["seed", "extractant", "draw"])[cols]
    pd.testing.assert_frame_equal(r.sort_index(), f.sort_index())


def test_cross_condition_support_never_shares_the_query_condition():
    oof = _synthetic_oof(n_extractants=4, seeds=(104729,), n_conditions=3)
    res = run_kshot_study(oof, arms=["M"], cfg=_small_cfg(policies=("random", "cross_condition"), n_draws=3,
                                                          include_oracle=False))
    q = res.per_draw_query_stats
    cc = q[q.policy == "cross_condition"]
    assert not cc.empty
    # No query row of a cross_condition draw can be transitively determined by the support.
    df = res.per_extractant_draw
    ccr = df[df.policy == "cross_condition"].merge(
        cc[["seed", "extractant", "draw", "n_free", "n_query"]], on=["seed", "extractant", "draw", "n_query"])
    assert (ccr.n_free == ccr.n_query).all()
    # ...whereas the random policy does determine some rows via within-cell transitivity.
    rq = q[q.policy == "random"]
    assert (rq.n_free < rq.n_query).any()


def test_small_extractants_are_excluded_by_a_k_independent_threshold():
    oof = _synthetic_oof(n_extractants=5)
    tiny = oof[oof.extractant == "SMILES0"].groupby("split_seed").head(12)
    oof = pd.concat([oof[oof.extractant != "SMILES0"], tiny], ignore_index=True)
    res_a = run_kshot_study(oof, arms=["M"], cfg=_small_cfg(ks=(1, 2), n_draws=1))
    res_b = run_kshot_study(oof, arms=["M"], cfg=_small_cfg(ks=(1, 2, 3, 5), n_draws=1))
    assert set(res_a.excluded_extractants.extractant) == {"SMILES0"}
    assert set(res_a.excluded_extractants.extractant) == set(res_b.excluded_extractants.extractant)


# --------------------------------------------------------------------------- #
# Science: does the harness measure what it claims?
# --------------------------------------------------------------------------- #

def test_calibration_recovers_synthetic_distortion_and_foreign_control_does_not():
    oof = _synthetic_oof(n_extractants=8, seeds=(104729, 130363, 155921, 196613))
    cfg = _small_cfg(ks=(1, 3, 5, 10), n_draws=5, policies=("random", "foreign"))
    res = run_kshot_study(oof, arms=["M"], cfg=cfg)
    grid = summarise_grid(res)
    per_ext = summarise_per_extractant(res)
    k0 = grid[(grid.policy == "random") & (grid.k == 0)]["macro_mae"].mean()
    k10 = grid[(grid.policy == "random") & (grid.k == 10) & (grid.form == "trend")
               & np.isclose(grid.lam, 0.1)]["macro_mae"].mean()
    assert k10 < 0.7 * k0
    curve = (grid[(grid.policy == "random") & (grid.form == "trend") & np.isclose(grid.lam, 0.1)]
             .groupby("k")["macro_mae"].mean())
    assert curve.loc[1] > curve.loc[3] > curve.loc[10]
    f0 = grid[(grid.policy == "foreign") & (grid.k == 0)]["macro_mae"].mean()
    f10 = grid[(grid.policy == "foreign") & (grid.k == 10) & (grid.form == "trend")
               & np.isclose(grid.lam, 0.1)]["macro_mae"].mean()
    assert np.isclose(f0, k0)
    assert (f0 - f10) < 0.5 * (k0 - k10) and f10 > k10
    sel, dec = select_and_confirm(per_ext, grid, selection_seed=104729,
                                 confirmation_seeds=[130363, 155921, 196613], replicates=300)
    row = dec[(dec.policy == "random") & (dec.k == 10)].iloc[0]
    assert row.positive_confirm_seeds == 3 and row.ci95_low > 0 and row.passes_rule
    frow = dec[(dec.policy == "foreign") & (dec.k == 10)].iloc[0]
    assert frow.is_control and not frow.passes_rule


def test_free_stratum_is_harder_than_the_determined_stratum():
    """Rows spanned by the support are arithmetic consequences; they must not carry the claim."""
    oof = _synthetic_oof(n_extractants=6, seeds=(104729,), n_conditions=1)
    cfg = _small_cfg(policies=("random",), n_draws=6, ks=(5,), forms=("trend",), lams=(0.1,), include_oracle=False)
    res = run_kshot_study(oof, arms=["M"], cfg=cfg)
    grid = summarise_grid(res)
    cal = grid[grid.k == 5].iloc[0]
    assert cal.macro_mae_free > cal.macro_mae
    assert cal.n_rows_free < cal.n_rows


def test_draw_delta_summary_exposes_the_downside():
    oof = _synthetic_oof(n_extractants=6, seeds=(104729,))
    res = run_kshot_study(oof, arms=["M"], cfg=_small_cfg(policies=("random",), n_draws=8, include_oracle=False))
    dd = summarise_draw_deltas(res)
    assert {"delta_mean", "delta_sd", "delta_q10", "frac_harmful"} <= set(dd.columns)
    assert (dd.delta_q10 <= dd.delta_mean + 1e-9).all()
    assert ((dd.frac_harmful >= 0) & (dd.frac_harmful <= 1)).all()


# --------------------------------------------------------------------------- #
# Decision statistics
# --------------------------------------------------------------------------- #

def test_bootstrap_p_is_bounded_away_from_zero_and_holm_can_bite():
    s = pd.Series(np.full(30, 0.2))
    out = paired_extractant_bootstrap_delta(s, replicates=500, seed=1)
    assert out["ci95_low"] > 0 and out["p_worse"] == pytest.approx(1 / 501)
    assert out["p_worse"] > 0
    holm = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.5})
    assert holm["a"] == pytest.approx(0.03) and holm["b"] == pytest.approx(0.08) and holm["c"] == 0.5
    s2 = pd.Series([0.1, -0.1, 0.05, -0.05])
    out2 = paired_extractant_bootstrap_delta(s2, replicates=500, seed=1)
    assert out2["ci95_low"] < 0 < out2["ci95_high"]


def test_harmful_configs_get_p_near_one_not_near_zero():
    """Regression: a two-sided p gave harmful configs p~0, letting them absorb Holm's budget."""
    out = paired_extractant_bootstrap_delta(pd.Series(np.full(20, -0.2)), replicates=500, seed=1)
    assert out["p_worse"] > 0.99


def test_selection_uses_only_the_selection_seed():
    oof = _synthetic_oof(n_extractants=6, seeds=(104729, 130363, 155921))
    cfg = _small_cfg(policies=("random",), n_draws=3, ks=(2,), include_oracle=False)
    res = run_kshot_study(oof, arms=["M"], cfg=cfg)
    per_ext, grid = summarise_per_extractant(res), summarise_grid(res)
    sel_a, _ = select_and_confirm(per_ext, grid, selection_seed=104729,
                                  confirmation_seeds=[130363, 155921], replicates=200)
    worst = (grid[(grid.k == 2) & (grid.seed == 104729)].sort_values("macro_mae").iloc[-1])
    planted = grid.copy()
    mask = ((planted.seed != 104729) & (planted.form == worst.form)
            & np.isclose(planted.lam, worst.lam) & (planted.k == 2))
    assert mask.any()
    planted.loc[mask, "macro_mae"] = -1.0
    sel_b, _ = select_and_confirm(per_ext, planted, selection_seed=104729,
                                  confirmation_seeds=[130363, 155921], replicates=200)
    assert sel_a.iloc[0].form == sel_b.iloc[0].form and sel_a.iloc[0].lam == sel_b.iloc[0].lam
    assert not (sel_b.iloc[0].form == worst.form and np.isclose(sel_b.iloc[0].lam, worst.lam))


def test_gate_reads_confirmation_seeds_only():
    oof = _synthetic_oof(n_extractants=6, seeds=(104729, 130363, 155921))
    cfg = _small_cfg(policies=("random",), n_draws=3, ks=(2,), include_oracle=False)
    res = run_kshot_study(oof, arms=["M"], cfg=cfg)
    per_ext, grid = summarise_per_extractant(res), summarise_grid(res)
    _, dec_a = select_and_confirm(per_ext, grid, selection_seed=104729,
                                  confirmation_seeds=[130363, 155921], replicates=200)
    bumped = grid.copy()
    sel_mask = bumped.seed == 104729
    bumped.loc[sel_mask, "macro_mae"] *= 0.5   # order-preserving: same argmin
    _, dec_b = select_and_confirm(per_ext, bumped, selection_seed=104729,
                                  confirmation_seeds=[130363, 155921], replicates=200)
    conf_cols = ["delta_mean_confirm", "positive_confirm_seeds", "ci95_low", "p_worse_one_sided", "passes_rule"]
    pd.testing.assert_frame_equal(dec_a[conf_cols], dec_b[conf_cols])
    assert not np.isclose(dec_a.delta_mean_incl_selection.iloc[0], dec_b.delta_mean_incl_selection.iloc[0])


def test_holm_family_spans_every_gated_cell_in_the_run():
    oof = _synthetic_oof(n_extractants=6, seeds=(104729, 130363, 155921))
    cfg = _small_cfg(policies=("random", "adjacent", "foreign"), n_draws=3, ks=(1, 2, 3), include_oracle=False)
    res = run_kshot_study(oof, arms=["M"], cfg=cfg)
    per_ext, grid = summarise_per_extractant(res), summarise_grid(res)
    _, dec = select_and_confirm(per_ext, grid, selection_seed=104729,
                                confirmation_seeds=[130363, 155921], replicates=200)
    gated = dec[~dec.is_control]
    assert gated.holm_family_size.nunique() == 1
    assert gated.holm_family_size.iloc[0] == len(gated)  # 2 policies x 3 k, controls excluded


def test_run_rejects_bad_inputs():
    oof = _synthetic_oof(n_extractants=3, seeds=(104729,))
    with pytest.raises(ValueError):
        run_kshot_study(oof, arms=["missing"], cfg=_small_cfg())
    with pytest.raises(ValueError):
        run_kshot_study(oof, arms=["M"], cfg=_small_cfg(ks=(0, 1)))
    with pytest.raises(ValueError):
        run_kshot_study(oof, arms=["M"], cfg=_small_cfg(policies=("adjacent",), include_oracle=True))
    per_ext = pd.DataFrame(); grid = pd.DataFrame()
    with pytest.raises(ValueError):
        select_and_confirm(per_ext, grid, selection_seed=1, confirmation_seeds=[])
    with pytest.raises(ValueError):
        select_and_confirm(per_ext, grid, selection_seed=1, confirmation_seeds=[1, 2])

"""Invariant tests for ``gen18proc.evalproto`` (DESIGN.md section 12.2, row ``test_evalproto``).

Synthetic records are generated from the solvating fixture equations with known ``n = 2.7``,
``p = 2.0`` and per-publication offsets (no dependence on WB1 files or WB2's ``testsystems``):

    log D = log K_metal + n log10[L] + p log10[acid] + delta_publication

Run:  .venv/Scripts/python.exe -m pytest generations/gen18_process/tests/test_evalproto.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

G18 = Path(__file__).resolve().parents[1]
if str(G18) not in sys.path:
    sys.path.insert(0, str(G18))

from gen18proc import evalproto as EP  # noqa: E402
from gen18proc.types import Flag, Mechanism, ProvStatus, SolvatingParams  # noqa: E402

LOG_K = {"Pr": 1.50, "Nd": 1.80}      # the solvating fixture of DESIGN.md section 12.1
N_TRUE = 2.7
P_TRUE = 2.0
LIGAND = "TODGA"
ACIDS = (0.5, 1.0, 2.0, 3.0)
LIGS = (0.05, 0.1, 0.2)


LAYOUTS = ("flat", "nested", "scalar")
"""The three accepted mapping-column forms (addenda/WB4a.md A1): ``ligand_M.<name>`` columns;
a nested JSON ``ligand_M`` column; WB1's scalar ``ligand_M`` + ``ligand_name`` (and scalar
``metal_initial_mM``)."""


def make_records(offsets: dict[str, float], *, metals=("Pr", "Nd"), acids=ACIDS, ligs=LIGS,
                 layout: str = "flat", replicates: int = 1, system_id: str = "sys_test",
                 noise_sd: float = 0.0, seed: int = 18,
                 per_pub_grid: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] | None = None,
                 ) -> pd.DataFrame:
    """Flat corpus_records.csv-like frame from the fixture equation.

    ``offsets`` maps publication_id -> delta; ``per_pub_grid`` optionally overrides the
    (acids, ligs) grid per publication.  ``replicates`` > 1 duplicates every condition row
    (bit-identical log D) so the aggregation and weights are exercised.  ``layout`` picks one of
    ``LAYOUTS``.
    """
    assert layout in LAYOUTS
    rng = np.random.default_rng(seed)
    rows = []
    k = 0
    for pub, delta in offsets.items():
        a_grid, l_grid = per_pub_grid.get(pub, (acids, ligs)) if per_pub_grid else (acids, ligs)
        for metal in metals:
            for a in a_grid:
                for lig in l_grid:
                    for r in range(replicates):
                        logd = (LOG_K[metal] + N_TRUE * np.log10(lig) + P_TRUE * np.log10(a)
                                + delta + (rng.normal(0.0, noise_sd) if noise_sd else 0.0))
                        row = {
                            "record_id": f"Ca_SAFE:{k:05d}", "system_id": system_id,
                            "metal": metal, "d": 10.0 ** logd, "log_d": logd,
                            "acid_nominal_M": a, "anion_M": a, "temperature_C": 25.0,
                            "publication_id": pub, "loading_series_id": None,
                            "is_tracer": True, "fit_eligible": True, "duplicate_flag": None,
                        }
                        if layout == "nested":
                            row["ligand_M"] = json.dumps({LIGAND: lig})
                            row["metals_initial_mM"] = json.dumps({metal: 0.1})
                        elif layout == "scalar":
                            row["ligand_name"] = LIGAND
                            row["ligand_M"] = lig
                            row["metal_initial_mM"] = 0.1
                            row["replicate_group_id"] = f"rg_{pub}_{metal}_{a}_{lig}"
                        else:
                            row[f"ligand_M.{LIGAND}"] = lig
                            row[f"metals_initial_mM.{metal}"] = 0.1
                        rows.append(row)
                        k += 1
    return pd.DataFrame(rows)


OFFSETS = {"pub_a": 0.30, "pub_b": -0.10, "pub_c": -0.20}   # sum to zero over multi-point pubs


# ---------------------------------------------------------------------------------------------
# M1 recovers the generating parameters
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", LAYOUTS)
def test_m1_recovers_slopes_intercepts_and_offsets_noise_free(layout):
    recs = make_records(OFFSETS, layout=layout)
    fit = EP.fit_mass_action(recs, mechanism=Mechanism.SOLVATING, ligand=LIGAND, band="20-30C")
    assert fit.status == "fitted"
    assert abs(fit.n - N_TRUE) < 1e-6
    assert abs(fit.p_eff - P_TRUE) < 1e-6
    for m, lk in LOG_K.items():
        assert abs(fit.intercepts[m] - lk) < 1e-6
    for pub, delta in OFFSETS.items():
        assert abs(fit.publication_effects[pub] - delta) < 1e-6
    assert abs(sum(fit.publication_effects.values())) < 1e-9
    assert fit.slope_status == {"n": "fitted", "p_eff": "fitted"}
    assert fit.n_points == 2 * len(ACIDS) * len(LIGS) * 3
    assert fit.n_publications == 3
    assert Flag.EQUILIBRIUM_ACID_ASSUMED_NOMINAL in fit.flags and Flag.OA_ASSUMED in fit.flags
    assert fit.residual_sd < 1e-9
    assert set(fit.param_names) == {"a.Pr", "a.Nd", "n", "p_eff", "pub.pub_a", "pub.pub_b"}


def test_m1_covariance_is_classical_and_hc1_shapes_agree():
    recs = make_records(OFFSETS, noise_sd=0.05, seed=3)
    fit = EP.fit_mass_action(recs, ligand=LIGAND, reliability=False)
    k = len(fit.param_names)
    assert fit.covariance.shape == (k, k) and fit.covariance_hc1.shape == (k, k)
    assert np.allclose(fit.covariance, fit.covariance.T)
    assert all(np.isfinite(fit.se[p]) and fit.se[p] > 0 for p in fit.param_names)
    assert "pub.pub_c" in fit.se           # the constrained (last) effect gets its SE too
    assert abs(fit.n - N_TRUE) < 0.2 and abs(fit.p_eff - P_TRUE) < 0.2


def test_replicates_are_aggregated_with_weight_n_rep():
    recs = make_records(OFFSETS, replicates=3)
    prepared = EP.prepare_records(recs, ligand=LIGAND, band="20-30C")
    pts = EP.aggregate_replicates(prepared)
    assert len(pts) == len(recs) // 3
    assert (pts["n_rep"] == 3).all()
    assert all(len(ids) == 3 for ids in pts["record_ids"])
    assert (pts["record_id"] == [min(ids) for ids in pts["record_ids"]]).all()
    fit = EP.fit_mass_action(recs, ligand=LIGAND, reliability=False)
    assert fit.n_points == len(pts) and fit.n_rows == len(recs)
    assert abs(fit.n - N_TRUE) < 1e-6


@pytest.mark.parametrize("layout", LAYOUTS)
def test_mapping_columns_resolve_under_every_layout(layout):
    recs = make_records(OFFSETS, layout=layout)
    lig = EP.ligand_column(recs, LIGAND)
    assert np.isfinite(lig).all() and set(np.round(lig, 12)) == set(LIGS)
    if layout == "flat":
        # no column at all for that ligand: malformed input, a ValueError by contract
        with pytest.raises(ValueError):
            EP.ligand_column(recs, "not_a_ligand")
    else:
        # the scalar / nested forms can say "absent": NaN on every row
        assert EP.ligand_column(recs, "not_a_ligand").isna().all()
    pr = EP.metals_initial_column(recs, "Pr")
    is_pr = (recs["metal"] == "Pr").to_numpy()
    assert (pr[is_pr] == 0.1).all() and pr[~is_pr].isna().all()
    # the replicate key is identical across layouts up to relabelling: same group sizes
    prepared = EP.prepare_records(recs, ligand=LIGAND, band="20-30C")
    pts = EP.aggregate_replicates(prepared)
    assert len(pts) == len(recs) and (pts["n_rep"] == 1).all()


def test_scalar_layout_rows_and_wb2_spelling():
    recs = make_records(OFFSETS, layout="scalar")
    # a row whose ligand_name is another ligand contributes NaN for LIGAND and is dropped
    recs.loc[recs.index[0], "ligand_name"] = "OTHER"
    prepared = EP.prepare_records(recs, ligand=LIGAND, band="20-30C")
    assert len(prepared) == len(recs) - 1
    # WB2's double-underscore spelling is accepted as a flattened column
    alt = make_records(OFFSETS).rename(columns={f"ligand_M.{LIGAND}": f"ligand_M__{LIGAND}"})
    fit = EP.fit_mass_action(alt, ligand=LIGAND, reliability=False)
    assert abs(fit.n - N_TRUE) < 1e-6
    # an object column mixing scalar cells and JSON cells is resolved cell by cell
    mixed = make_records(OFFSETS, layout="nested")
    mixed["ligand_M"] = mixed["ligand_M"].astype(object)
    mixed.loc[mixed.index[:5], "ligand_M"] = [0.05] * 5
    mixed["ligand_name"] = LIGAND
    lig = EP.ligand_column(mixed, LIGAND)
    assert np.isfinite(lig).all() and (lig.iloc[:5] == 0.05).all()


def test_explicit_replicate_group_id_is_the_aggregation_key():
    recs = make_records(OFFSETS, layout="scalar", replicates=2)
    prepared = EP.prepare_records(recs, ligand=LIGAND, band="20-30C")
    pts = EP.aggregate_replicates(prepared)
    assert len(pts) == len(recs) // 2 and (pts["n_rep"] == 2).all()
    # the explicit id wins over the condition tuple: two rows with different contact times but
    # one replicate_group_id form ONE group; identical conditions with two ids form TWO groups
    recs2 = make_records(OFFSETS, layout="scalar", replicates=2)
    recs2["contact_time_min"] = np.where(np.arange(len(recs2)) % 2 == 0, 10.0, 30.0)
    pts2 = EP.aggregate_replicates(EP.prepare_records(recs2, ligand=LIGAND, band="20-30C"))
    assert len(pts2) == len(recs2) // 2
    recs3 = make_records(OFFSETS, layout="scalar", replicates=2)
    recs3["replicate_group_id"] = recs3["record_id"]
    pts3 = EP.aggregate_replicates(EP.prepare_records(recs3, ligand=LIGAND, band="20-30C"))
    assert len(pts3) == len(recs3)
    # without any explicit column the condition tuple (including contact time) is the key
    recs4 = make_records(OFFSETS, replicates=2)
    recs4["contact_time_min"] = np.where(np.arange(len(recs4)) % 2 == 0, 10.0, 30.0)
    pts4 = EP.aggregate_replicates(EP.prepare_records(recs4, ligand=LIGAND, band="20-30C"))
    assert len(pts4) == len(recs4)


def test_domain_is_filled_from_wb2_builder_when_importable():
    pytest.importorskip("gen18proc.domain")
    from gen18proc.types import ApplicabilityDomain
    for layout in LAYOUTS:
        recs = make_records(OFFSETS, layout=layout)
        fit = EP.fit_mass_action(recs, ligand=LIGAND, reliability=False)
        assert isinstance(fit.domain, ApplicabilityDomain), layout
        assert fit.domain.n_records == fit.n_rows == len(recs)
        assert fit.domain.n_publications == 3
        lo, hi = fit.domain.log_ligand[LIGAND]
        assert abs(lo - np.log10(min(LIGS))) < 1e-12 and abs(hi - np.log10(max(LIGS))) < 1e-12
        lo, hi = fit.domain.log_acid
        assert abs(lo - np.log10(min(ACIDS))) < 1e-12 and abs(hi - np.log10(max(ACIDS))) < 1e-12


def test_real_corpus_records_layout_is_consumed_without_fitting():
    """Parse-only smoke test on WB1's file (skipped when the database is not built).  No
    parameter is estimated here: the pre-registration is sealed before any corpus fit."""
    path = G18 / "systems" / "corpus_records.csv"
    if not path.is_file():
        pytest.skip("systems/corpus_records.csv not built")
    df = pd.read_csv(path)
    sub = df.loc[df["system_id"] == "sys_5cb78e5000d40860"]
    if sub.empty:
        pytest.skip("TODGA/nitrate/aliphatic system absent from the built database")
    assert {"ligand_name", "ligand_M", "metal_initial_mM", "replicate_group_id"} <= set(df.columns)
    prepared = EP.prepare_records(sub, ligand="TODGA", band="20-30C")
    assert 0 < len(prepared) <= len(sub)
    assert np.isfinite(prepared["lE"]).all() and np.isfinite(prepared["lA"]).all()
    assert (prepared["_ligand_M"] == sub.set_index("record_id").loc[prepared["record_id"],
                                                                     "ligand_M"].to_numpy()).all()
    pts = EP.aggregate_replicates(prepared)
    assert pts["n_rep"].sum() == len(prepared)
    assert (pts["record_id"] == [min(ids) for ids in pts["record_ids"]]).all()
    # the explicit replicate_group_id is the aggregation unit
    assert len(pts) == prepared["replicate_group_id"].nunique()
    nd = EP.metals_initial_column(prepared, "Nd")
    assert nd.loc[prepared["metal"] != "Nd"].isna().all()


def test_slope_fixed_at_prior_when_fewer_than_three_levels():
    recs = make_records(OFFSETS, acids=(1.0, 3.0))      # two acid levels only
    fit = EP.fit_mass_action(recs, ligand=LIGAND, n_prior=3.0, p_prior=2.0, reliability=False)
    assert fit.slope_status == {"n": "fitted", "p_eff": "assumed"}
    assert fit.p_eff == 2.0 and fit.se["p_eff"] == 0.0
    assert abs(fit.n - N_TRUE) < 1e-6           # the other slope is unaffected
    assert fit.distinct_levels == {"lE": 3, "lA": 2}
    # with a wrong prior the intercepts absorb what they can but the fit is no longer exact
    fit2 = EP.fit_mass_action(recs, ligand=LIGAND, n_prior=3.0, p_prior=1.0, reliability=False)
    assert fit2.p_eff == 1.0 and fit2.residual_sd > 1e-3
    params = EP.as_solvating_params(fit2)
    assert params.p_anion.status == ProvStatus.ASSUMED
    assert params.p_anion.provenance.assumed_label == "ASSUMED_PLACEHOLDER"
    assert params.p_anion.range == (1.0, 1.0)


def test_band_filter_and_nan_temperature_assignment():
    recs = make_records(OFFSETS)
    recs.loc[recs.index[:10], "temperature_C"] = 45.0
    recs.loc[recs.index[10:20], "temperature_C"] = np.nan
    prepared = EP.prepare_records(recs, ligand=LIGAND, band="20-30C")
    assert len(prepared) == len(recs) - 10
    assert EP.band_contains("<20C", 5.0) and not EP.band_contains("<20C", 20.0)
    assert EP.band_contains(">=50C", 50.0) and EP.band_contains("20-30C", float("nan"))
    assert not EP.band_contains("30-40C", float("nan"))


def test_unit_slip_and_ineligible_rows_are_dropped():
    recs = make_records(OFFSETS)
    recs.loc[recs.index[0], "duplicate_flag"] = "UNIT_SLIP_DUPLICATE"
    recs.loc[recs.index[1], "fit_eligible"] = False
    recs.loc[recs.index[2], "duplicate_flag"] = "TIED_D"        # kept
    prepared = EP.prepare_records(recs, ligand=LIGAND, band="20-30C")
    assert len(prepared) == len(recs) - 2


def test_malformed_input_raises_value_error():
    recs = make_records(OFFSETS)
    with pytest.raises(ValueError):
        EP.fit_mass_action(recs.drop(columns=[f"ligand_M.{LIGAND}"]), ligand=LIGAND)
    two = pd.concat([recs, recs.assign(system_id="sys_other")])
    with pytest.raises(ValueError):
        EP.fit_mass_action(two, ligand=LIGAND)


def test_metal_specific_slopes_arm_x2():
    recs = make_records(OFFSETS)
    fit = EP.fit_mass_action(recs, ligand=LIGAND, pooled_slopes=False, reliability=False)
    assert "n.Pr" in fit.param_names and "p_eff.Nd" in fit.param_names
    assert abs(fit.n - N_TRUE) < 1e-6 and abs(fit.p_eff - P_TRUE) < 1e-6


# ---------------------------------------------------------------------------------------------
# LOPO, baselines
# ---------------------------------------------------------------------------------------------

def test_lopo_effect_zero_reproduces_injected_offsets_as_errors():
    recs = make_records(OFFSETS)
    table = EP.lopo_evaluate(recs, mechanism="solvating", ligand=LIGAND, band="20-30C")
    assert set(table["holdout"]) == {"lopo"}
    assert (table["status"] == "scored").all()
    assert len(table) == 3 * 2
    for pub, delta in OFFSETS.items():
        # the remaining publications' sum-to-zero effects put their mean offset into the
        # intercepts, so the held-out error is |delta_j - mean of the others' deltas|
        others = [d for p, d in OFFSETS.items() if p != pub]
        expected = abs(delta - float(np.mean(others)))
        sub = table.loc[table["publication_id"] == pub]
        assert np.allclose(sub["mae_M1"], expected, atol=1e-6)
        # one held-out point sets the effect -> the rest is exact
        assert np.allclose(sub["mae_M1_offset"], 0.0, atol=1e-9)
        assert (sub["n_points"] == len(ACIDS) * len(LIGS)).all()


def test_lopo_offsets_exact_when_others_sum_to_zero():
    # held-out publication with the others summing to zero: error == injected offset exactly
    offs = {"pub_a": 0.25, "pub_b": 0.10, "pub_c": -0.10}
    table = EP.lopo_evaluate(make_records(offs), ligand=LIGAND)
    sub = table.loc[table["publication_id"] == "pub_a"]
    assert np.allclose(sub["mae_M1"], 0.25, atol=1e-6)


def test_b0_and_b1_computed_as_specified():
    # training: two publications, one metal; held-out: one publication
    rows = [
        # record_id, pub, metal, acid, lig, log_d
        ("r_02", "pub_t", "Nd", 1.0, 0.1, 1.0),
        ("r_01", "pub_t", "Nd", 1.0, 0.2, 2.0),     # equidistant tie with r_02 for the test point
        ("r_03", "pub_t", "Nd", 3.0, 0.1, 3.0),
        ("r_04", "pub_u", "Nd", 0.5, 0.05, 5.0),
        ("r_05", "pub_u", "Nd", 2.0, 0.1, 6.0),
        ("r_06", "pub_u", "Nd", 3.0, 0.2, 7.0),
        ("r_10", "pub_h", "Nd", 1.0, 0.1414213562373095, 0.0),   # log lig midway 0.1 .. 0.2
    ]
    recs = pd.DataFrame([{
        "record_id": rid, "system_id": "s", "metal": m, "d": 10 ** ld, "log_d": ld,
        "acid_nominal_M": a, "anion_M": a, f"ligand_M.{LIGAND}": lig, "temperature_C": 25.0,
        "publication_id": pub, "fit_eligible": True, "duplicate_flag": None,
    } for rid, pub, m, a, lig, ld in rows])
    table = EP.lopo_evaluate(recs, ligand=LIGAND, band="20-30C")
    row = table.loc[table["publication_id"] == "pub_h"].iloc[0]
    # B0: mean of the six training log D values
    assert abs(row["mae_B0"] - abs(np.mean([1, 2, 3, 5, 6, 7]) - 0.0)) < 1e-12
    # B1: r_01 and r_02 are equidistant (same acid, +/- half a log-ligand step); |delta lA| ties
    # too; the smaller record_id r_01 (log D 2.0) wins
    assert abs(row["mae_B1"] - 2.0) < 1e-12
    assert row["n_points"] == 1
    assert np.isnan(row["mae_M1_offset"])          # one point: nothing left to score


def test_b1_tie_rule_prefers_smaller_delta_log_acid_before_record_id():
    train = pd.DataFrame({
        "record_id": ["r_01", "r_02"], "lA": [0.0, np.log10(2.0)],
        "lE": [np.log10(2.0), 0.0], "log_d": [1.0, 2.0]})
    # test point at (lA, lE) = (0, 0): both at distance log10(2); r_02 has the larger |dlA|
    assert EP._nearest_index(0.0, 0.0, train) == 0
    train2 = train.assign(lA=[np.log10(2.0), 0.0], lE=[0.0, np.log10(2.0)])
    assert EP._nearest_index(0.0, 0.0, train2) == 1


def test_metal_absent_from_training_is_counted_not_scored():
    recs = make_records(OFFSETS)
    # pub_c is the only publication with Pr -> holding it out leaves no Pr intercept
    recs = recs.loc[~((recs["metal"] == "Pr") & (recs["publication_id"] != "pub_c"))]
    table = EP.lopo_evaluate(recs, ligand=LIGAND)
    row = table.loc[(table["publication_id"] == "pub_c") & (table["metal"] == "Pr")].iloc[0]
    assert row["status"] == "metal_absent_from_training" and np.isnan(row["mae_M1"])
    macro = EP.macro_over_systems(table)
    assert macro.per_system.loc["sys_test", "n_pairs"] == len(table) - 1


def test_in_sample_rows_are_labelled_and_exact_noise_free():
    table = EP.in_sample_evaluate(make_records(OFFSETS), ligand=LIGAND)
    assert set(table["holdout"]) == {"in_sample"}
    assert np.allclose(table["mae_M1"], 0.0, atol=1e-9)


def test_macro_over_systems_averaging_order():
    t = pd.DataFrame({
        "system_id": ["A", "A", "A", "B"], "status": ["scored"] * 4,
        "publication_id": ["p", "p", "q", "r"], "metal": ["Pr", "Nd", "Pr", "Nd"],
        "mae_M1": [0.1, 0.3, 0.5, 1.0], "mae_B0": [0.2, 0.2, 0.2, 2.0],
        "mae_B1": [0.0, 0.0, 0.0, 0.0], "mae_M1_offset": [np.nan, 0.1, 0.1, 0.1],
        "mae_B1_crossmetal": [0.0] * 4, "n_points": [1, 1, 2, 4]})
    res = EP.macro_over_systems(t)
    assert res.n_systems == 2
    assert abs(res.per_system.loc["A", "mae_M1"] - 0.3) < 1e-12
    assert abs(res.macro["mae_M1"] - (0.3 + 1.0) / 2) < 1e-12
    assert abs(res.macro["mae_B0"] - (0.2 + 2.0) / 2) < 1e-12
    assert abs(res.point_weighted["mae_M1"] - (0.1 + 0.3 + 1.0 + 4.0) / 8) < 1e-12


def test_paired_bootstrap_is_deterministic_and_sensible():
    a = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    b = a - 0.2
    r1 = EP.paired_system_bootstrap(a, b, n_boot=500, seed=18)
    r2 = EP.paired_system_bootstrap(a, b, n_boot=500, seed=18)
    assert r1 == r2
    mean, lo, hi = r1
    assert abs(mean - 0.2) < 1e-12 and abs(lo - 0.2) < 1e-9 and abs(hi - 0.2) < 1e-9
    b2 = a - 0.2 + np.array([0.1, -0.1, 0.05, -0.05, 0.0, 0.2, -0.2])
    mean, lo, hi = EP.paired_system_bootstrap(a, b2, n_boot=2000, seed=18)
    assert lo <= mean <= hi and lo > 0.0
    assert EP.paired_system_bootstrap([1.0], [0.5])[1] != EP.paired_system_bootstrap([1.0],
                                                                                    [0.5])[1]
    with pytest.raises(ValueError):
        EP.paired_system_bootstrap([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------------------------

def test_jackknife_and_split_half_on_noise_free_records():
    offs = {"pub_a": 0.3, "pub_b": -0.1, "pub_c": -0.2, "pub_d": 0.05, "pub_e": -0.05}
    recs = make_records(offs)
    jk = EP.jackknife_by_publication(recs, ligand=LIGAND)
    assert abs(jk["n"][0] - N_TRUE) < 1e-6 and jk["n"][1] < 1e-6
    assert abs(jk["p_eff"][0] - P_TRUE) < 1e-6 and jk["p_eff"][1] < 1e-6
    sh = EP.split_half_by_publication(recs, repeats=20, seed=18, ligand=LIGAND)
    assert sh["repeats"] == 20 and sh["n_publications"] == 5
    assert sh["n_sign_agreement"] == 20 and sh["p_eff_sign_agreement"] == 20
    assert np.isnan(sh["intercept_r_median"])       # only two metals: r undefined (< 3)
    assert len(sh["per_repeat"]) == 20
    # the same seed gives the same halves
    sh2 = EP.split_half_by_publication(recs, repeats=20, seed=18, ligand=LIGAND)
    assert sh["per_repeat"]["pubs_a"].tolist() == sh2["per_repeat"]["pubs_a"].tolist()
    fit = EP.fit_mass_action(recs, ligand=LIGAND)
    assert fit.interpretable == {"n": True, "p_eff": True}
    assert fit.jackknife_se["n"][1] < 1e-6


def test_split_half_undefined_below_four_publications():
    recs = make_records(OFFSETS)
    assert EP.split_half_by_publication(recs, ligand=LIGAND) is None
    fit = EP.fit_mass_action(recs, ligand=LIGAND)
    assert fit.split_half is None and fit.jackknife_se is not None
    assert fit.interpretable == {"n": True, "p_eff": True}


def test_assumed_slope_is_never_interpretable():
    recs = make_records(OFFSETS, acids=(1.0, 3.0))
    fit = EP.fit_mass_action(recs, ligand=LIGAND)
    assert fit.slope_status["p_eff"] == "assumed" and fit.interpretable["p_eff"] is False


def test_intercept_r_with_three_metals():
    LOG_K["La"] = 0.9
    try:
        offs = {"pub_a": 0.3, "pub_b": -0.1, "pub_c": -0.2, "pub_d": 0.0}
        recs = make_records(offs, metals=("La", "Pr", "Nd"))
        sh = EP.split_half_by_publication(recs, repeats=5, seed=18, ligand=LIGAND)
        assert abs(sh["intercept_r_median"] - 1.0) < 1e-9
    finally:
        LOG_K.pop("La")


# ---------------------------------------------------------------------------------------------
# Parameter blocks, comparison counter, stub
# ---------------------------------------------------------------------------------------------

def test_as_solvating_params_block_shape():
    fit = EP.fit_mass_action(make_records(OFFSETS), ligand=LIGAND, fit_manifest_sha256="ab" * 32)
    p = EP.as_solvating_params(fit)
    assert isinstance(p, SolvatingParams)
    assert p.medium_anion == "nitrate" and p.temperature_band == "20-30C"
    assert set(p.log_k) == {"Pr", "Nd"}
    lk = p.log_k["Nd"]
    assert lk.status == ProvStatus.FITTED_FROM_CORPUS
    assert lk.provenance.model_id == "M1_pooled_sys_test_20-30C"
    assert lk.provenance.fit_manifest_sha256 == "ab" * 32
    assert lk.range is not None and lk.range[0] <= lk.value <= lk.range[1]
    assert len(lk.source.safe_exp_ids) == len(ACIDS) * len(LIGS) * 3
    assert abs(p.n_solvation.value - N_TRUE) < 1e-6
    assert p.n_solvation.status == ProvStatus.FITTED_FROM_CORPUS
    assert p.p_h.value == 0.0 and p.p_h.status == ProvStatus.ASSUMED
    assert p.k_acid_uptake.value is None and p.k_acid_uptake.status == ProvStatus.UNKNOWN
    json.dumps(p.log_k["Nd"].to_json())        # serialisable
    ce = EP.as_cation_exchange_params(fit)
    assert abs(ce.b_proton.value + P_TRUE) < 1e-6 and abs(ce.a_dimer.value - N_TRUE) < 1e-6


def test_comparison_counter_benjamini_hochberg():
    cc = EP.ComparisonCounter()
    cc.record("M1_vs_B0", "primary", 0.01)
    cc.record("M1_vs_B1", "primary", 0.20)
    cc.record("E2_active", "secondary", None)
    for name, p in zip("X1 X2 X3 X4".split(), [0.01, 0.02, 0.03, 0.04]):
        cc.record(name, "exploratory", p)
    t = cc.table()
    expl = t.loc[t["family"] == "exploratory"].set_index("name")
    assert np.allclose(expl["p_adjusted"], [0.04, 0.04, 0.04, 0.04])
    assert t.loc[t["family"] != "exploratory", "p_adjusted"].isna().all()
    assert cc.counts() == {"primary": 2, "secondary": 1, "exploratory": 4, "total": 7}
    with pytest.raises(ValueError):
        cc.record("X1", "exploratory", 0.5)
    with pytest.raises(ValueError):
        cc.record("bad", "tertiary", 0.5)
    adj = EP.ComparisonCounter.benjamini_hochberg([0.005, 0.5, np.nan, 0.03])
    assert np.allclose(adj[[0, 1, 3]], [0.015, 0.5, 0.045]) and np.isnan(adj[2])


LOAD_LOG_K, LOAD_N, LOAD_LT, LOAD_ACID = 3.0, 3.0, 0.1, 3.0
LOAD_MM = (1.0, 3.0, 5.0, 8.0, 12.0)


def _ideal_loading_log_d(metal_total_M: float) -> float:
    """``log10 D`` of the ideal depletion law, solved independently of ``gen18proc``.

    Single metal, one solvating ligand, O/A = 1, fixed acid (so the acid term is inside K):
    ``D = K (L_T - n y)^n`` with ``x = T / (1 + D)`` and ``y = T - x``.  Root-found here with
    ``brentq`` so that the test data do not come from the code path under test.
    """
    from scipy.optimize import brentq

    k = 10.0 ** LOAD_LOG_K

    def residual(d: float) -> float:
        x = metal_total_M / (1.0 + d)
        free = LOAD_LT - LOAD_N * (metal_total_M - x)
        return float("nan") if free <= 0 else k * free ** LOAD_N - d

    return float(np.log10(brentq(residual, 1e-12, k * LOAD_LT ** LOAD_N, xtol=1e-15, rtol=1e-15)))


def make_loading_records() -> pd.DataFrame:
    """Two series: one generated by the ideal depletion law, one flat (D independent of loading)."""
    rows = []
    for i, mm in enumerate(LOAD_MM):
        rows.append(dict(system_id="sys_test", record_id=f"load{i}", metal="Nd",
                         log_d=_ideal_loading_log_d(mm * 1e-3), acid_nominal_M=LOAD_ACID,
                         anion_M=LOAD_ACID, anion="nitrate", ligand_name=LIGAND,
                         ligand_M=LOAD_LT, metal_initial_mM=mm, publication_id="pub_load",
                         loading_series_id="ls_depletion", is_tracer=(i == 0),
                         fit_eligible=True, duplicate_flag=None, temperature_C=25.0))
    for i, mm in enumerate(LOAD_MM):
        rows.append(dict(system_id="sys_test", record_id=f"flat{i}", metal="Nd", log_d=0.0,
                         acid_nominal_M=1.0, anion_M=1.0, anion="nitrate", ligand_name=LIGAND,
                         ligand_M=LOAD_LT, metal_initial_mM=mm, publication_id="pub_flat",
                         loading_series_id="ls_flat", is_tracer=(i == 0),
                         fit_eligible=True, duplicate_flag=None, temperature_C=25.0))
    rows.append(dict(system_id="sys_test", record_id="lone0", metal="Nd", log_d=0.5,
                     acid_nominal_M=1.0, anion_M=1.0, anion="nitrate", ligand_name=LIGAND,
                     ligand_M=LOAD_LT, metal_initial_mM=2.0, publication_id="pub_lone",
                     loading_series_id="ls_single_point", is_tracer=True,
                     fit_eligible=True, duplicate_flag=None, temperature_C=25.0))
    for row in rows:
        row["d"] = 10.0 ** row["log_d"]
    return pd.DataFrame(rows)


def test_loading_series_evaluate_recovers_the_law_that_generated_the_series():
    """C1 (DESIGN 10.4) anchored at the tracer reproduces an independently solved series."""
    recs = make_loading_records()
    fits = {"sys_test": {"n": LOAD_N, "slope_status": {"n": "fitted"}}}
    out = EP.loading_series_evaluate(recs, fits, oa=1.0, phi_arm=False).set_index(
        "loading_series_id")

    dep = out.loc["ls_depletion"]
    assert dep["anchor_status"] == "anchored"
    # the anchor recovers the log K that generated the data, and every non-tracer point follows
    assert abs(dep["log_k_anchored"] - LOAD_LOG_K) < 1e-6
    assert dep["mae_ideal"] < 1e-9
    assert dep["c1_wins"] and dep["mae_constant"] > 0.05
    assert dep["n_predicted"] == len(LOAD_MM) - 1
    assert dep["n_used"] == LOAD_N

    # PRE_REGISTRATION section 7: every series counts as it falls -- a flat series is kept and
    # is a win for C0, not an exclusion.
    flat = out.loc["ls_flat"]
    assert flat["status"] == "scored"
    assert flat["mae_constant"] == pytest.approx(0.0, abs=1e-12)
    assert not flat["c1_wins"]

    # a series with no non-tracer point is reported with a status, never dropped; it is neither
    # a win nor a loss for C1 (NaN), so the R2 count must not treat it as either
    assert "ls_single_point" in out.index
    assert out.loc["ls_single_point", "status"] != "scored"
    assert np.isnan(out.loc["ls_single_point", "mae_ideal"])
    assert np.isnan(float(out.loc["ls_single_point", "c1_wins"]))


def test_loading_series_evaluate_falls_back_to_the_prior_exponent():
    """No fit for the system -> C1 uses the pre-registered prior n0 and says so."""
    recs = make_loading_records()
    out = EP.loading_series_evaluate(recs, {}, oa=1.0, n_fallback=LOAD_N,
                                     phi_arm=False).set_index("loading_series_id")
    assert out.loc["ls_depletion", "n_source"] == "prior_n0"
    assert out.loc["ls_depletion", "mae_ideal"] < 1e-9
    # a slope that was fixed at its prior is not a fitted slope either
    fixed = {"sys_test": {"n": LOAD_N, "slope_status": {"n": "prior"}}}
    out2 = EP.loading_series_evaluate(recs, fixed, oa=1.0, n_fallback=LOAD_N,
                                      phi_arm=False).set_index("loading_series_id")
    assert out2.loc["ls_depletion", "n_source"] == "prior_n0"


def test_loading_series_evaluate_rejects_a_malformed_fit():
    """Malformed input raises ValueError (DESIGN 1.5), not an opaque attribute error."""
    recs = make_loading_records()
    with pytest.raises(ValueError):
        EP.loading_series_evaluate(recs, {"sys_test": {"n": LOAD_N, "slope_status": "fitted"}})

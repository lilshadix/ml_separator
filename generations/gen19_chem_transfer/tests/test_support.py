"""Section 13 support score, domain status and per-fold thresholds (``gen19ct.evaluation.support``; brief section 27
"support-score correctness", verification finding VL-05).

Every expected value is worked by hand in the comments; the frames are synthetic.  Nothing reads ``log_D``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gen19ct.chemistry import support_graph as SG
from gen19ct.evaluation import support as ES

TODGA = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"
TDDGA = "CCCCCCCCCCN(CCCCCCCCCC)C(=O)COCC(=O)N(CCCCCCCCCC)CCCCCCCCCC"
TBP = "CCCCOP(=O)(OCCCC)OCCCC"


def _row(state, system, acid, ext=-1.0, temp=25.0, pub="p1", smiles=TODGA):
    return {SG.METAL_COL: state, SG.ELEMENT_COL: state.split("(")[0] if state else "Nd", SG.SYSTEM_COL: system,
            SG.PUB_COL: pub, SG.FAMILY_COL: "diglycolamide", SG.MECH_COL: "NEUTRAL_SOLVATING", SG.SMILES_COL: smiles,
            SG.ACID_ANION_COL: "nitrate", SG.DILUENT_COL: "aliphatic", SG.LOG_ACID_COL: acid, SG.LOG_EXT_COL: ext,
            SG.TEMP_COL: temp}


# --------------------------------------------------------------------------------------------- #
# thresholds
# --------------------------------------------------------------------------------------------- #

def test_tau_thresholds_leave_one_row_out_percentiles_by_hand() -> None:
    # pair A = (Nd(III), S): log10 acid 0, 1, 3; pair B = (Eu(III), S): one row at 2 (excluded: < 2 rows).
    # Standardisation over all 4 training rows: acid mean 1.5, SD sqrt(mean(2.25, .25, 2.25, .25)) = sqrt(1.25);
    # ext and T are constant -> SD 1 and z = 0.  Leave-one-row-out nearest distances inside A:
    #   row 0: |0 - 1| / sqrt(1.25) = 0.894427; row 1: min(1, 2) / sqrt(1.25) = 0.894427; row 2: |3 - 1| / sqrt(1.25) = 1.788854
    # numpy linear percentiles of [a, a, b]: p50 = a; p95 at position 1.9 -> a + 0.9 (b - a); p99.5 -> a + 0.99 (b - a)
    tr = pd.DataFrame([_row("Nd(III)", "S", 0.0), _row("Nd(III)", "S", 1.0), _row("Nd(III)", "S", 3.0),
                       _row("Eu(III)", "S", 2.0)])
    si = SG.SupportIndex(tr)
    t_in, t_ext, t_max, n = ES.tau_thresholds(si, tr)
    a, b = 1 / math.sqrt(1.25), 2 / math.sqrt(1.25)
    assert n == 3
    assert t_in == pytest.approx(a, abs=1e-12)
    assert t_ext == pytest.approx(a + 0.9 * (b - a), abs=1e-12)
    assert t_max == pytest.approx(a + 0.99 * (b - a), abs=1e-12)
    assert ES.fold_tau(tr) == {"tau_in": t_in, "tau_ext": t_ext, "tau_max": t_max, "n_tau_rows": 3}


def test_tau_thresholds_skip_missing_query_dimension_and_unknown_state() -> None:
    # The X(?) row and the singleton Eu pair never enter as queries.  Acid standardisation over the finite values
    # [1, 5, 2]: mean 8/3, SD sqrt(26/9).  Row 0 (acid 1) queries its partner row 1, whose missing acid is imputed at
    # the training mean (z = 0): distance |1 - 8/3| / sqrt(26/9) = 0.980581.  Row 1 (acid missing) skips the acid
    # dimension; ext and T are constant, so its distance to row 0 is 0.  Distances [0.980581, 0]:
    # p50 = 0.490290, p95 = 0.95 x 0.980581, p99.5 = 0.995 x 0.980581.
    tr = pd.DataFrame([_row("Nd(III)", "S", 1.0), _row("Nd(III)", "S", float("nan")), _row(None, "S", 5.0),
                       _row("Eu(III)", "S", 2.0)])
    si = SG.SupportIndex(tr)
    t_in, t_ext, t_max, n = ES.tau_thresholds(si, tr)
    d = (8 / 3 - 1) / math.sqrt(26 / 9)
    assert n == 2
    assert (t_in, t_ext, t_max) == pytest.approx((0.5 * d, 0.95 * d, 0.995 * d), abs=1e-12)
    with pytest.raises(ValueError):
        ES.tau_thresholds(si, tr.iloc[:2])
    empty = pd.DataFrame([_row("Nd(III)", "S", 1.0)])
    assert ES.tau_thresholds(SG.SupportIndex(empty), empty)[3] == 0


# --------------------------------------------------------------------------------------------- #
# components and score
# --------------------------------------------------------------------------------------------- #

def _features(**kw):
    f = {"exact_pair_rows": 0, "n_neighbour_metals_same_charge": 0, "nearest_radius_metal": None,
         "nearest_radius_distance_A": float("nan"), "n_same_family_rows_for_metal": 0, "n_publications_system": 0,
         "condition_distance_pair": float("nan"), "condition_distance_system": float("nan"), "series_bracketed": False,
         "n_series_neighbours_pm1": 0, "n_systems_for_metal": 0, "nearest_radius_same_charge": False}
    f.update(kw)
    return f


def test_support_components_s1_to_s8_by_hand() -> None:
    f = _features(exact_pair_rows=4, n_neighbour_metals_same_charge=3, nearest_radius_metal="Pr(III)",
                  nearest_radius_distance_A=0.017, n_same_family_rows_for_metal=196, n_publications_system=2,
                  condition_distance_pair=0.25, condition_distance_system=9.0, series_bracketed=False,
                  n_series_neighbours_pm1=1)
    c = ES.support_components(f, s4=0.6, tau_ext=0.5, metal_series="Ln")
    assert c["s1"] == pytest.approx(math.log(5) / math.log(21))          # log1p(4) / log1p(20) = 0.528634
    assert c["s2"] == pytest.approx(0.6)                                 # 3 / 5
    assert c["s3"] == pytest.approx(math.exp(-0.34))                     # exp(-0.017 / 0.05)
    assert c["s4"] == 0.6
    assert c["s5"] == 1.0                                                # log1p(196) / log1p(100) > 1 -> capped
    assert c["s6"] == pytest.approx(0.4)                                 # 2 / 5
    assert c["s7"] == pytest.approx(math.exp(-0.5))                      # the pair exists: CD = 0.25, / tau_ext 0.5
    assert c["s8"] == 0.5                                                # a +-1 neighbour, not bracketed
    assert ES.support_score(c) == pytest.approx(np.mean(list(c.values())))
    # no exact pair: s1 = 0 and s7 uses the system distance; no radius neighbour: s3 = 0; outside Ln/An: s8 omitted
    g = _features(n_neighbour_metals_same_charge=9, condition_distance_system=1.0, series_bracketed=True)
    d = ES.support_components(g, s4=0.0, tau_ext=2.0, metal_series=None)
    assert d["s1"] == 0.0 and d["s2"] == 1.0 and d["s3"] == 0.0 and d["s7"] == pytest.approx(math.exp(-0.5))
    assert math.isnan(d["s8"])
    assert ES.support_score(d) == pytest.approx((0 + 1 + 0 + 0 + 0 + 0 + math.exp(-0.5)) / 7)   # s8 not averaged
    assert ES.support_components(_features(), s4=0.0, tau_ext=1.0, metal_series="An")["s7"] == 0.5   # CD undefined
    assert ES.support_components(_features(series_bracketed=True), s4=0.0, tau_ext=1.0, metal_series="An")["s8"] == 1.0
    assert math.isnan(ES.support_score({"s1": float("nan")}))


def test_s4_component_reading_switches() -> None:
    fp = {k: SG.fingerprint(v) for k, v in (("TODGA", TODGA), ("TDDGA", TDDGA), ("TBP", TBP))}
    z0 = np.zeros(3)
    # the query's own system has Tanimoto 1 and d_desc 0 -> 1.0 when it counts
    cands = [("TODGA", fp["TODGA"], z0), ("TBP", fp["TBP"], np.array([3.0, 4.0, 0.0]))]
    assert ES.s4_component("TODGA", fp["TODGA"], z0, cands) == pytest.approx(1.0)
    from rdkit import DataStructs
    sim = DataStructs.TanimotoSimilarity(fp["TODGA"], fp["TBP"])
    # own system excluded: only TBP, d_desc = 5 -> sim x exp(-2.5)
    assert ES.s4_component("TODGA", fp["TODGA"], z0, cands, include_query_system=False) == pytest.approx(sim * math.exp(-2.5))
    nan_c = [("TDDGA", fp["TDDGA"], np.array([np.nan, 0.0, 0.0]))]
    assert ES.s4_component("TODGA", fp["TODGA"], z0, nan_c) == 0.0                              # skipped
    sim2 = DataStructs.TanimotoSimilarity(fp["TODGA"], fp["TDDGA"])
    assert ES.s4_component("TODGA", fp["TODGA"], z0, nan_c, missing_descriptor="zero") == pytest.approx(sim2)
    assert ES.s4_component("TODGA", None, z0, cands) == 0.0
    with pytest.raises(ValueError):
        ES.s4_component("TODGA", fp["TODGA"], z0, cands, missing_descriptor="mean")


def test_s4_registered_reading_resolved_2026_09_15() -> None:
    """Section 13 resolution: the query's own system counts when it has training rows of the query's metal state;
    a system with an undefined d_desc (or fingerprint) is skipped."""
    from rdkit import DataStructs

    fp = {k: SG.fingerprint(v) for k, v in (("TODGA", TODGA), ("TDDGA", TDDGA), ("TBP", TBP))}
    z0 = np.zeros(3)
    # candidate systems = training systems with >= 1 row of the query's state; X(?) rows and other states never count
    states = ["Nd(III)", "Nd(III)", "Nd(III)", "Nd(III)", "Eu(III)", None, "Nd(III)"]
    systems = ["TODGA", "TODGA", "TODGA", "TBP", "TDDGA", "TDDGA", None]
    assert ES.s4_candidate_counts(states, systems, "Nd(III)") == {"TBP": 1, "TODGA": 3}
    assert ES.s4_candidate_counts(states, systems, "Eu(III)") == {"TDDGA": 1}
    sim_tbp = DataStructs.TanimotoSimilarity(fp["TODGA"], fp["TBP"])
    tbp_z = np.array([3.0, 4.0, 0.0])                                     # d_desc = 5 -> exp(-2.5)
    # V1 / V0: the own system has rows of the state -> it counts (Tanimoto 1, d_desc 0) -> 1.0
    in_domain = [("TODGA", fp["TODGA"], z0, 3), ("TBP", fp["TBP"], tbp_z, 1), ("TDDGA", fp["TDDGA"], z0, 0)]
    assert ES.s4_registered("TODGA", fp["TODGA"], z0, in_domain) == pytest.approx(1.0)
    # V5: the hidden cell leaves the own system no row of the state -> it never counts; TDDGA (no Nd row) is no
    # candidate either, although its similarity would be 1 -> only TBP
    hidden = [("TODGA", fp["TODGA"], z0, 0), ("TBP", fp["TBP"], tbp_z, 1), ("TDDGA", fp["TDDGA"], z0, 0)]
    assert ES.s4_registered("TODGA", fp["TODGA"], z0, hidden) == pytest.approx(sim_tbp * math.exp(-2.5))
    # ... and there the two pre-seal readings of query-system inclusion coincide (why V5 support_score was computed)
    measured = [(k, f, z) for k, f, z, n in hidden if n > 0]
    assert ES.s4_component("TODGA", fp["TODGA"], z0, measured, include_query_system=True) == \
        ES.s4_component("TODGA", fp["TODGA"], z0, measured, include_query_system=False)
    # undefined d_desc on a candidate, on the query, or an undefined fingerprint: skipped (no support), never zero-filled
    nan_z = np.array([np.nan, 0.0, 0.0])
    assert ES.s4_registered("TODGA", fp["TODGA"], z0, [("TDDGA", fp["TDDGA"], nan_z, 5)]) == 0.0
    assert ES.s4_registered("TODGA", fp["TODGA"], nan_z, [("TBP", fp["TBP"], tbp_z, 1)]) == 0.0
    assert ES.s4_registered("TODGA", fp["TODGA"], z0, [("TBP", None, tbp_z, 1)]) == 0.0
    assert ES.s4_registered("TODGA", None, z0, in_domain) == 0.0
    assert ES.s4_registered("TODGA", fp["TODGA"], z0, []) == 0.0
    # the registered function equals the component with the registered defaults on the measured systems
    both = [(k, f, z) for k, f, z, n in in_domain if n > 0]
    assert ES.s4_registered("TODGA", fp["TODGA"], z0, in_domain) == ES.s4_component("TODGA", fp["TODGA"], z0, both)
    assert set(ES.S4_READING) == {"query_system", "missing_descriptor", "candidates"}
    with pytest.raises(ValueError):
        ES.s4_registered("TODGA", fp["TODGA"], z0, [("TBP", fp["TBP"], tbp_z, -1)])


@pytest.mark.slow
def test_preseal_support_score_update_outputs_consistent() -> None:
    """The outputs of ``scripts/g19_update_support_preseal.py`` (skipped when absent): support_score is computed for
    every support job, the recomputed s4 equals the stored pre-seal s4, and V5-primary / V2 support_score values equal
    the pre-seal candidate values."""
    import json

    from gen19ct import paths
    status_p = paths.G19_ROOT / "evaluation" / "preseal" / "support_status.json"
    if not status_p.exists():
        pytest.skip("pre-seal support status absent")
    status = json.loads(status_p.read_text(encoding="utf-8"))
    if "s4_reading" not in status:
        pytest.skip("g19_update_support_preseal.py has not been run")
    for job in ("V5__primary", "V2__element", "V1__copy", "V0__rows"):
        rec = status["jobs"][job]
        assert rec["support_score"].startswith("computed"), job
        assert rec["s4_recomputed_equals_preseal_s4"] is True, job
        f = paths.REPO_ROOT / rec["support_score_file"]                     # manifest paths are repository-relative
        sc = pd.read_parquet(f)
        assert len(sc) == rec["n_scored_row_entries"] and np.isfinite(sc["support_score"]).all(), job
        if job in ("V5__primary", "V2__element"):
            assert rec["support_score_equals_preseal_candidate"] is True, job


# --------------------------------------------------------------------------------------------- #
# domain status
# --------------------------------------------------------------------------------------------- #

TAU = (0.1, 1.0, 3.0)


def _status(f, **kw):
    base = dict(system_present=True, metal_present=True, family_rows=100, mechanism_rows=500,
                expert="E2_neutral_solvating", fam_cd=float("nan"), tau=TAU, anion_unseen=False)
    base.update(kw)
    return ES.domain_status(f=f, **base)


def test_domain_status_each_label_and_first_match_order() -> None:
    assert ES.DOMAIN_STATUS_ORDER[0] == "UNSUPPORTED" and ES.DOMAIN_STATUS_ORDER[-1] == "INTERPOLATION"
    # 1 UNSUPPORTED: mechanism UNKNOWN wins over everything below, even an in-domain exact pair
    in_dom = _features(exact_pair_rows=8, condition_distance_pair=0.05, condition_distance_system=0.05)
    assert _status(in_dom, expert="none_unknown")[0] == "UNSUPPORTED"
    assert _status(in_dom, mechanism_rows=0)[0] == "UNSUPPORTED"
    assert _status(in_dom, anion_unseen=True)[0] == "UNSUPPORTED"
    assert _status(_features(condition_distance_system=3.5), metal_present=True)[0] == "UNSUPPORTED"   # > tau_max
    assert _status(_features(), metal_present=False)[0] == "UNSUPPORTED"      # no same-charge metal, no +-1 neighbour
    # 2 FAMILY_EXTRAPOLATION: system absent, no family rows, metal present (mechanism present)
    assert _status(_features(), system_present=False, family_rows=0)[0] == "FAMILY_EXTRAPOLATION"
    # 3 CROSS_METAL_LIGAND_TRANSFER: the V5 cell, with condition_extrapolated from the system distance
    s = _status(_features(condition_distance_system=1.5))
    assert s == ("CROSS_METAL_LIGAND_TRANSFER", True, False, False)
    s = _status(_features(nearest_radius_same_charge=True), system_present=False, metal_present=False)
    assert s[0] == "CROSS_METAL_LIGAND_TRANSFER" and s[2] is True                                    # both nodes new
    # 4 CROSS_LIGAND_TRANSFER: system absent, family present, metal present; family distance > tau_max is unsupported
    assert _status(_features(), system_present=False, fam_cd=0.5)[0] == "CROSS_LIGAND_TRANSFER"
    assert _status(_features(), system_present=False, fam_cd=3.5)[0] == "UNSUPPORTED"
    # 5 CROSS_METAL_TRANSFER: system present, metal absent, a +-1 series neighbour
    assert _status(_features(n_series_neighbours_pm1=1), metal_present=False)[0] == "CROSS_METAL_TRANSFER"
    # 6 CONDITION_EXTRAPOLATION beats 7 IN_DOMAIN; 7 needs >= 5 rows; 8 is the rest
    assert _status(_features(exact_pair_rows=8, condition_distance_pair=1.2, condition_distance_system=0.0))[0] == \
        "CONDITION_EXTRAPOLATION"
    assert _status(in_dom)[0] == "IN_DOMAIN"
    assert _status(_features(exact_pair_rows=4, condition_distance_pair=0.05, condition_distance_system=0.05))[0] == \
        "INTERPOLATION"
    s = _status(_features(exact_pair_rows=2, condition_distance_system=0.2))
    assert s == ("INTERPOLATION", False, False, True)                                               # conditions unknown


@pytest.mark.slow
def test_fold_builder_tau_reproduces_on_registered_folds() -> None:
    from gen19ct import paths
    from gen19ct.data import load
    from gen19ct.folds import io as FI
    if not (paths.FOLDS_DIR / "INDEX.json").exists():
        pytest.skip("folds not built")
    model = load.load_model_rows()
    sup = SG.prepare_support_frame(model)[list(SG.REQUIRED_COLUMNS)]
    ids = model["canonical_measurement_id"].astype(str)
    for stem in ("V5__primary__exact", "V1__copy__exact", "V2__element__exact"):
        stored = FI.read_fold_fields(stem, "support_tau")
        f = FI.read_design(stem)[1]
        assert stored[f.fold_id] is not None, (stem, "support_tau missing: rebuild folds")
        hidden = ids.isin(set(f.hidden_row_ids)).to_numpy()
        mine = ES.fold_tau(sup[~hidden], SG.load_system_table())
        assert mine == stored[f.fold_id], (stem, f.fold_id)

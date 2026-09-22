"""The section 8 signal-injection power check and the "reliability before correlation" report, on synthetic data.

No model is fitted on a real fold and no injected value is written: the injection runs on a ~200-row synthetic frame and
the reliability estimators are checked against quantities of known reliability.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.evaluation import discovery as D
from gen19ct.evaluation import h3 as H3
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import power as PW
from gen19ct.evaluation import transfer as ET
from gen19ct.folds import io as FI
from gen19ct.models import interface as I

SCRIPTS = paths.G19_ROOT / "scripts"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RP = _load("g19_run_power")
RH = _load("g19_run_h3")
TH = importlib.import_module("test_h3") if "test_h3" in sys.modules else None
if TH is None:                                        # reuse the H3 mini-corpus without duplicating it
    spec = importlib.util.spec_from_file_location("test_h3", Path(__file__).resolve().parent / "test_h3.py")
    TH = importlib.util.module_from_spec(spec)
    sys.modules["test_h3"] = TH
    spec.loader.exec_module(TH)


# --------------------------------------------------------------------------------------------- #
# which contrasts need the check
# --------------------------------------------------------------------------------------------- #

def test_contrasts_needing_power_reads_the_failed_h1_h1b_h3_rows():
    rows = [
        {"family": "primary", "contrast": "M2 vs B3i", "design": "V5", "candidate": "M2", "comparator": "B3i",
         "margin": 0.1057, "r19_verdict_full": "UNDECIDED", "primary_cluster_unit": True, "key": "M2 vs B3i@V5"},
        {"family": "primary", "contrast": "M2 vs B3i", "design": "V5", "candidate": "M2", "comparator": "B3i",
         "margin": 0.1057, "r19_verdict_full": "UNDECIDED", "primary_cluster_unit": False, "key": "M2 vs B3i@V5"},
        {"family": "H1b", "contrast": "B6 vs B3i", "design": "V5", "candidate": "B6", "comparator": "B3i",
         "margin": 0.1057, "r19_verdict_full": "PASS", "primary_cluster_unit": True, "key": "B6 vs B3i@V5"},
        {"family": "H3", "contrast": "M2:WITH vs M2:WITHOUT", "design": "V5", "candidate": "M2:WITH",
         "comparator": "M2:WITHOUT", "margin": 0.1057, "r19_verdict_full": "FAIL", "primary_cluster_unit": True,
         "key": "M2:WITH vs M2:WITHOUT@V5"},
        {"family": "ladder", "contrast": "M2 vs M1", "design": "V5", "candidate": "M2", "comparator": "M1",
         "margin": 0.1, "r19_verdict_full": "FAIL", "primary_cluster_unit": True, "key": "M2 vs M1@V5#ladder"},
    ]
    need = PW.contrasts_needing_power(pd.DataFrame(rows))
    # only the registered power families, only the primary-cluster row
    assert list(need["family"]) == ["primary", "H1b", "H3"]
    assert list(need["needs_power_check"]) == [True, False, True]    # a PASS needs no check
    assert set(PW.POWER_FAMILIES) == set(D.POWER_CHECK_FAMILIES) >= {"primary", "H1b", "S1(b)", "H3"}
    assert PW.contrasts_needing_power(pd.DataFrame()).empty
    assert "ladder" not in set(need["family"])


def test_read_contrast_files_skips_absent_files(tmp_path):
    a = tmp_path / "contrasts_registered.csv"
    pd.DataFrame({"family": ["primary"], "contrast": ["M2 vs B3i"]}).to_csv(a, index=False)
    out = PW.read_contrast_files([a, tmp_path / "missing.csv"])
    assert len(out) == 1 and out["source_file"].iloc[0] == "contrasts_registered.csv"
    assert PW.read_contrast_files([tmp_path / "nope.csv"]).empty


# --------------------------------------------------------------------------------------------- #
# the signal (section 8), the H3 u-sharing and reproducibility
# --------------------------------------------------------------------------------------------- #

def test_h3_u_share_picks_the_nearest_cn8_radius_lanthanide():
    share = PW.h3_u_share(["Am(III)", "Cm(III)", "La(III)", "Nd(III)", "Eu(III)", "Lu(III)", "U(VI)", "Nd(?)", None])
    # Shannon CN8: Am(III) 1.09; Nd 1.109, Eu 1.066, La 1.160, Lu 0.977 -> Nd is nearest to Am
    assert share["Am(III)"] == "Nd(III)"
    assert "U(VI)" not in share and "Nd(?)" not in share       # not An(III) / no state
    assert "Cm(III)" not in share                              # Shannon tabulates no CN8 radius for Cm(III)
    assert set(share) <= {"Am(III)"}
    assert PW.h3_u_share(["Am(III)"]) == {}                     # no Ln(III) to share with
    # only an H3 contrast shares a u (section 8); H1 / H1b / S1(b) do not
    assert PW.u_share_for("H3", ["Am(III)", "Nd(III)"]) == {"Am(III)": "Nd(III)"}
    for fam in ("primary", "H1b", "S1(b)"):
        assert PW.u_share_for(fam, ["Am(III)", "Nd(III)"]) == {}


def _frame():
    df = TH._frame(n_per_cell=4)
    return df


def test_injection_is_reproducible_by_seed_and_shares_the_actinide_u():
    df = _frame()
    known = df[SG.METAL_COL].notna()
    scored = df.index[known & (df[SG.METAL_COL] == "Nd(III)")]
    share = PW.h3_u_share(df[SG.METAL_COL].dropna().unique())
    a = PW.injected_run(df, kappa=0.25, seed=104729, scored_index=scored, u_share=share)
    b = PW.injected_run(df, kappa=0.25, seed=104729, scored_index=scored, u_share=share)
    c = PW.injected_run(df, kappa=0.25, seed=130363, scored_index=scored, u_share=share)
    assert a.signal.equals(b.signal) and a.frame[I.TARGET_COL].equals(b.frame[I.TARGET_COL])
    assert not a.signal.equals(c.signal)
    # y' = y + kappa s on the kept rows, training and test alike
    kept = a.frame.index
    assert np.allclose(a.frame[I.TARGET_COL].to_numpy(dtype=float),
                       df.loc[kept, I.TARGET_COL].to_numpy(dtype=float) + 0.25 * a.signal.loc[kept].to_numpy(dtype=float))
    # a bigger kappa scales the same signal
    big = PW.injected_run(df, kappa=1.0, seed=104729, scored_index=scored, u_share=share)
    assert big.signal.equals(a.signal)
    delta_a = a.frame[I.TARGET_COL].to_numpy(dtype=float) - df.loc[kept, I.TARGET_COL].to_numpy(dtype=float)
    delta_b = big.frame[I.TARGET_COL].to_numpy(dtype=float) - df.loc[kept, I.TARGET_COL].to_numpy(dtype=float)
    assert np.allclose(delta_b, 4.0 * delta_a)
    # the shared u: an Am(III) row's signal equals the Nd(III) signal of the same system (u_Am = u_Nd, same v_s)
    for system in ("S1", "S2"):
        am = a.signal[(df[SG.METAL_COL] == "Am(III)") & (df[SG.SYSTEM_COL] == system)].dropna().unique()
        nd = a.signal[(df[SG.METAL_COL] == "Nd(III)") & (df[SG.SYSTEM_COL] == system)].dropna().unique()
        assert len(am) == 1 and len(nd) == 1 and am[0] == pytest.approx(nd[0])
    # without the sharing they differ
    plain = PW.injected_run(df, kappa=0.25, seed=104729, scored_index=scored)
    am = plain.signal[(df[SG.METAL_COL] == "Am(III)") & (df[SG.SYSTEM_COL] == "S1")].dropna().unique()
    nd = plain.signal[(df[SG.METAL_COL] == "Nd(III)") & (df[SG.SYSTEM_COL] == "S1")].dropna().unique()
    assert am[0] != pytest.approx(nd[0])
    assert PW.KAPPAS == ET.KAPPAS == (0.1, 0.25, 0.5, 1.0)


def test_unknown_state_rows_are_dropped_from_the_injected_refits(tmp_path):
    """POST-HOC addendum 1: X(?) rows are dropped from the injected refits of every arm, and the scored rows stay the
    un-injected run's."""
    df = _frame()
    unknown = df.index[df[SG.METAL_COL].isna()]
    assert len(unknown) == 3
    scored_ids = df.loc[(df[SG.METAL_COL] == "Nd(III)") & (df[SG.SYSTEM_COL] == "S1"), FI.ROW_ID].tolist()
    scored = df.index[df[FI.ROW_ID].isin(scored_ids)]
    run = PW.injected_run(df, kappa=0.5, seed=104729, scored_index=scored)
    assert set(run.dropped_unknown_state) == set(unknown)
    assert not run.frame.index.intersection(unknown).size          # absent from the refit frame
    assert run.frame[SG.METAL_COL].notna().all()
    # the fold's training rows: input minus hidden minus X(?) -- never a hidden row, never an X(?) row
    fold = TH._cell_fold(df, "Nd(III)", "S1")
    hidden = df.index[df[FI.ROW_ID].isin(fold.hidden_row_ids)]
    train, sc = PW.fold_inputs(run, fold, labels_of=lambda ids: df.index[df[FI.ROW_ID].isin(list(ids))],
                               scored_ids=scored_ids)
    assert not train.index.intersection(hidden).size
    assert not train.index.intersection(unknown).size
    assert set(sc) == set(scored) and not set(sc) & set(train.index)
    # an X(?) row among the scored rows is refused (X(?) rows are scored in no design, section 2)
    with pytest.raises(AssertionError, match="X\\(\\?\\)"):
        PW.injected_run(df, kappa=0.5, seed=1, scored_index=list(scored) + list(unknown[:1]))
    # the scored rows must stay identical to the un-injected run's
    ET.assert_same_scored_rows(pd.Index(scored), pd.Index(sc))
    with pytest.raises(AssertionError, match="injected scored rows differ"):
        ET.assert_same_scored_rows(pd.Index(scored), pd.Index(list(scored)[:-1]))


def test_finding_v03_is_checked_in_row_id_space_not_in_corpus_label_space():
    """The scored-row check of ``run_contrast`` compares ``discovery.scoring_frame``'s index, which is ``row_id``, with
    the row ids of the un-injected run -- NOT with ``_scored_index``'s corpus-frame labels.

    ``Corpus.labels_of`` maps a row id through ``idmap`` to ``frame.index``, so the two label spaces are disjoint
    whenever the corpus index is not the row id.  Comparing them directly raised
    "injected scored rows differ from the un-injected run: N added, N missing" for every contrast (equal counts, because
    the rows are the same and only the labels differ).
    """
    df = _frame()
    known = df[SG.METAL_COL].notna()
    scored_index = df.index[known & (df[SG.METAL_COL] == "Nd(III)")]

    class _Corpus:
        frame = df

    row_ids = RP._scored_row_ids(_Corpus(), scored_index)
    expected = df.loc[scored_index, FI.ROW_ID].astype(str)
    assert list(row_ids) == list(expected)
    assert not row_ids.has_duplicates and len(row_ids) == len(scored_index)

    # what a scoring frame of those rows is indexed by (discovery.scoring_frame sets the index to row_id)
    scoring_frame_index = pd.Index(expected.to_numpy())
    ET.assert_same_scored_rows(row_ids, scoring_frame_index)                 # the fixed comparison passes
    # the corpus-label space must be a DIFFERENT space, or this test would not guard anything
    assert set(map(str, scored_index)) != set(map(str, scoring_frame_index))
    with pytest.raises(AssertionError, match="injected scored rows differ"):
        ET.assert_same_scored_rows(pd.Index(scored_index), scoring_frame_index)


def test_kappa_min_logic_and_the_underpowered_bound():
    def res(kappa, passed):
        return PW.KappaResult(kappa=kappa, passed=passed, scope_verdict="PASS" if passed else "FAIL",
                              full_verdict="UNDECIDED", point=0.1, margin=0.05, n_units=10, n_rows=40)
    informative = PW.kappa_min([res(0.1, False), res(0.25, True), res(0.5, True), res(1.0, True)])
    assert informative["kappa_min"] == 0.25 and informative["verdict"] == "INFORMATIVE_NULL"
    assert "informative" in informative["reported"]
    under = PW.kappa_min([res(0.1, False), res(0.25, False), res(0.5, True), res(1.0, True)])
    assert under["kappa_min"] == 0.5 and under["verdict"] == "UNDECIDED_UNDERPOWERED"
    assert "UNDECIDED (underpowered)" in under["reported"]
    none = PW.kappa_min([res(k, False) for k in PW.KAPPAS])
    assert none["kappa_min"] is None and none["verdict"] == "UNDECIDED_UNDERPOWERED"
    best = PW.kappa_min([res(0.1, True), res(0.25, True), res(0.5, True), res(1.0, True)])
    assert best["kappa_min"] == 0.1 and best["verdict"] == "INFORMATIVE_NULL"
    assert best["kappa_min_informative"] == ET.KAPPA_MIN_INFORMATIVE == 0.25
    # every registered kappa is required: a missing one is an error, never a silent pass
    with pytest.raises(ValueError, match="missing"):
        PW.kappa_min([res(0.1, False), res(0.25, True)])


def test_power_record_carries_the_seed_the_sharing_and_no_injected_value():
    def res(kappa, passed):
        return PW.KappaResult(kappa=kappa, passed=passed, scope_verdict="PASS" if passed else "FAIL",
                              full_verdict="UNDECIDED", point=0.2, margin=0.05, n_units=9, n_rows=36)
    rec = PW.power_record("M2 vs B3i@V5", family="primary", design="V5", arms=["M2", "B3i"],
                          results=[res(k, k >= 0.5) for k in PW.KAPPAS], seed=104729,
                          u_share={"Am(III)": "Nd(III)"}, n_dropped_unknown_state=3,
                          uninjected={"point": 0.01, "r19_verdict_full": "UNDECIDED"})
    assert rec["kappa_min"] == 0.5 and rec["verdict"] == "UNDECIDED_UNDERPOWERED"
    assert rec["injection_seed"] == 104729 and rec["n_u_shared_states"] == 1
    assert rec["n_rows_dropped_unknown_state"] == 3 and rec["uninjected"]["point"] == 0.01
    assert set(rec["readings"]) >= {"refit", "scored_rows", "no_persisted_values"}
    rows = RP.kappa_rows([rec])
    assert len(rows) == len(PW.KAPPAS) and set(rows["kappa"]) == set(PW.KAPPAS)
    assert (rows["kappa_min"] == 0.5).all() and "log_D" not in rows.columns
    # an output frame carrying an injected value is refused
    PW.assert_no_injected_values(rows)
    for col in ("log_D", "log_D_injected", "injected_signal", "mean_logD"):
        with pytest.raises(AssertionError, match="never persisted"):
            PW.assert_no_injected_values(rows.assign(**{col: 0.0}))


# --------------------------------------------------------------------------------------------- #
# reliability before correlation (section 8)
# --------------------------------------------------------------------------------------------- #

def _groups(n: int = 8) -> list[str]:
    return [f"g{i}" for i in range(n)]


def _unit_rows(seed: int = 1, *, n_split: int = 20, n_jack: int = 6, n_single: int = 4, noise: float = 0.02):
    """Rows of units measured in 6 (split-half), 3 (jackknife) or 1 (no reliability) publication groups; the per-unit
    quantity is the mean of the rows' values and the truth is a per-unit ramp."""
    rng = np.random.default_rng(seed)
    recs = []
    for u in range(n_split + n_jack + n_single):
        truth = u / 10.0
        ng = 6 if u < n_split else (3 if u < n_split + n_jack else 1)
        for g in range(ng):
            for r in range(3):
                recs.append({"unit": f"u{u:02d}", "group": f"g{u % 7}_{g}", "v": truth + noise * rng.standard_normal()})
    df = pd.DataFrame(recs, index=[f"r{i}" for i in range(len(recs))])
    return df


def test_split_half_reliability_is_per_unit_and_recovers_a_known_high_and_zero_reliability():
    """Section 8 (task X finding V-02): split-half by publication group where THE UNIT has >= 4 groups -- each unit's
    own groups are halved, the halves are correlated across those units, Spearman-Brown, mean over 20 seeded halves."""
    df = _unit_rows()
    est = lambda idx: df.loc[idx].groupby("unit")["v"].mean()           # noqa: E731
    hi = PW.split_half_reliability(est, df["unit"], df["group"])
    assert hi["status"] == "computed" and hi["reliability"] > 0.9 and hi["gate"] == "RELIABLE"
    assert hi["n_halves"] == PW.N_SPLIT_HALVES == 20 and hi["n_units"] == 20 and hi["min_groups"] == 4
    assert set(hi["units"]) == {f"u{u:02d}" for u in range(20)}       # the 3- and 1-group units are NOT in this branch
    assert all(p["n_units"] == 20 for p in hi["per_half"])
    rng = np.random.default_rng(7)
    noise = lambda idx: pd.Series(rng.standard_normal(30), index=[f"u{u:02d}" for u in range(30)])   # noqa: E731
    lo = PW.split_half_reliability(noise, df["unit"], df["group"])
    assert lo["reliability"] < 0.3 and lo["gate"] == "UNDECIDED_UNRELIABLE"
    assert hi["floor"] == PW.RELIABILITY_FLOOR == 0.3
    # a middling quantity: Spearman-Brown steps the half-length correlation up
    def middling(idx, sd=0.25):
        m = df.loc[idx].groupby("unit")["v"].mean()
        return m + sd * rng.standard_normal(len(m))
    mid = PW.split_half_reliability(middling, df["unit"], df["group"])
    assert 0.3 < mid["reliability"] < 1.0 and mid["reliability"] > mid["mean_half_correlation"]
    assert ET.spearman_brown(0.5) == pytest.approx(2 / 3)
    # the halves are per unit: every unit's rows are split, the two halves are disjoint and together are all its rows
    seen = {}
    def probe(idx):
        for u, sub in df.loc[idx].groupby("unit"):
            seen.setdefault(u, []).append(set(sub["group"]))
        return df.loc[idx].groupby("unit")["v"].mean()
    PW.split_half_reliability(probe, df["unit"], df["group"], n_halves=1)
    for u, halves in seen.items():
        assert len(halves) == 2 and not halves[0] & halves[1] and len(halves[0] | halves[1]) == 6
        assert len(halves[0]) == 3 and len(halves[1]) == 3
    # no unit with >= 4 groups: not applicable (the jackknife branch takes over)
    few = df[df["unit"] >= "u20"]
    assert PW.split_half_reliability(est, few["unit"], few["group"])["status"] == "NOT_APPLICABLE"
    # the unit universe may be restricted to the units with a full-sample estimate
    sub = PW.split_half_reliability(est, df["unit"], df["group"], units=["u00", "u01", "u25"], n_halves=2)
    assert sub["n_units"] == 2 and sub["n_units_universe"] == 3
    # seeded
    a = PW.split_half_reliability(est, df["unit"], df["group"], seed=19, n_halves=3)
    b = PW.split_half_reliability(est, df["unit"], df["group"], seed=19, n_halves=3)
    assert a["reliability"] == b["reliability"]
    assert PW.split_halves(_groups(), seed=19) == PW.split_halves(_groups(), seed=19)


def test_jackknife_reliability_recovers_a_known_high_and_zero_reliability():
    units = [f"u{i}" for i in range(20)]
    full = pd.Series(np.linspace(0, 2, 20), index=units)
    groups = _groups(6)
    # tiny leave-one-out scatter against a wide between-unit spread: reliability ~ 1
    rng = np.random.default_rng(3)
    tight = {g: full + 0.001 * rng.standard_normal(20) for g in groups}
    hi = PW.jackknife_reliability(full, tight)
    assert hi["status"] == "computed" and hi["reliability"] > 0.9 and hi["gate"] == "RELIABLE"
    assert hi["n_publications_left_out"] == 6 and hi["n_units_with_se"] == 20
    # leave-one-out scatter far larger than the between-unit spread: reliability ~ 0
    wide = {g: full + 30.0 * rng.standard_normal(20) for g in groups}
    lo = PW.jackknife_reliability(full, wide)
    assert lo["reliability"] < 0.3 and lo["gate"] == "UNDECIDED_UNRELIABLE"
    assert lo["between_unit_variance"] < lo["mean_within_unit_se2"]
    # the formula is transfer.jackknife_reliability
    assert ET.jackknife_reliability(1.0, 3.0) == pytest.approx(0.25)
    assert PW.jackknife_reliability(pd.Series(dtype=float), {})["status"] == "NOT_RUN"
    assert PW.jackknife_reliability(full, {})["status"] == "NOT_RUN"
    # a unit's SE^2 uses only the estimates that left out one of ITS OWN groups (deleting a foreign group changes nothing)
    ug = {u: ["g0", "g1"] for u in units}
    own = PW.jackknife_reliability(full, wide, ug)
    assert all(v == 2 for v in own["per_unit_n_jackknife"].values())
    assert own["mean_within_unit_se2"] != lo["mean_within_unit_se2"]


def test_jackknife_by_unit_and_reliability_of_combine_the_per_unit_branches():
    """Task X finding V-02: units with 2-3 groups take the delete-one-publication jackknife over their own groups,
    single-group units are counted, and the quantity's reliability is the minimum over the computed branches."""
    df = _unit_rows()
    est = lambda idx: df.loc[idx].groupby("unit")["v"].mean()           # noqa: E731
    full = est(df.index)
    jk = PW.jackknife_by_unit(est, full, df["unit"], df["group"])
    assert jk["status"] == "computed" and jk["n_units"] == 6 and jk["n_units_single_group"] == 4
    assert set(jk["units"]) == {f"u{u:02d}" for u in range(20, 26)} and jk["n_publications_left_out"] == 18
    assert all(v == 3 for v in jk["per_unit_n_jackknife"].values()) and jk["reliability"] > 0.9
    rec = PW.reliability_of("b7_slopes", estimate_rows=est, full=full, unit_of_row=df["unit"], group_of_row=df["group"])
    assert rec["n_units"] == 30 and rec["n_units_split_half"] == 20 and rec["n_units_jackknife"] == 6
    assert rec["n_units_single_group"] == 4 and rec["n_units_without_rows"] == 0
    assert rec["reliability"] == pytest.approx(min(rec["reliability_split_half"], rec["reliability_jackknife"]))
    assert rec["reliability"] > 0.9 and rec["gate"] == "RELIABLE" and rec["status"] == "computed"
    assert rec["per_unit_method"]["u00"] == "split_half" and rec["per_unit_method"]["u22"] == "jackknife"
    assert rec["per_unit_method"]["u29"] == "single_group" and "V-02" in rec["reading"]
    # the unit universe is full.index: a unit without rows is counted, a row of an unknown unit is ignored
    rec2 = PW.reliability_of("b7_slopes", estimate_rows=est, full=pd.concat([full, pd.Series({"ghost": 1.0})]),
                             unit_of_row=df["unit"], group_of_row=df["group"], n_halves=3)
    assert rec2["n_units_without_rows"] == 1 and rec2["per_unit_method"]["ghost"] == "no_rows"
    # a noisy quantity fails the floor in both branches
    rng = np.random.default_rng(9)
    noise = lambda idx: pd.Series(rng.standard_normal(30), index=[f"u{u:02d}" for u in range(30)])   # noqa: E731
    bad = PW.reliability_of("logsf_amplitudes", estimate_rows=noise, full=full, unit_of_row=df["unit"],
                            group_of_row=df["group"], n_halves=5)
    assert bad["reliability"] < 0.3 and bad["gate"] == "UNDECIDED_UNRELIABLE"
    # only the jackknife branch applies when no unit has 4 groups; only split-half when none has 2-3
    few = df[df["unit"] >= "u20"]
    only_jk = PW.reliability_of("b7_slopes", estimate_rows=lambda idx: few.loc[idx].groupby("unit")["v"].mean(),
                                full=few.groupby("unit")["v"].mean(), unit_of_row=few["unit"], group_of_row=few["group"])
    assert only_jk["split_half"]["status"] == "NOT_APPLICABLE" and only_jk["status"] == "computed"
    assert only_jk["reliability"] == pytest.approx(only_jk["reliability_jackknife"])
    with pytest.raises(ValueError, match="section 8 reliability scope"):
        PW.reliability_of("something_else", estimate_rows=est, full=full, unit_of_row=df["unit"], group_of_row=df["group"])
    assert set(PW.RELIABILITY_QUANTITIES) == {"b7_slopes", "factor_loadings", "embeddings", "logsf_amplitudes",
                                              "support_score_components"}
    # the floor gates a correlation: below it the correlation may neither support nor close a claim
    badc = PW.gate_correlation(0.2, {"spearman": -0.4})
    assert badc["verdict"] == "UNDECIDED (unreliable)" and not badc["reportable"]
    good = PW.gate_correlation(0.55, {"spearman": -0.4})
    assert good["verdict"] == "readable" and good["reportable"]
    assert PW.gate_correlation(float("nan"), 0.1)["verdict"] == "UNDECIDED (unreliable)"
    tab = PW.reliability_table([rec, only_jk, badc | {"quantity": "embeddings", "method": "x", "status": "computed"}])
    assert list(tab.columns)[:5] == ["quantity", "unit", "method", "status", "reliability"]
    assert {"n_units_split_half", "n_units_jackknife", "n_units_single_group", "reliability_jackknife"} <= set(tab.columns)
    assert len(tab) == 3 and int(tab.loc[0, "n_units_split_half"]) == 20


def test_procrustes_alignment_and_embedding_bootstrap_stability():
    rng = np.random.default_rng(11)
    ref = pd.DataFrame(rng.standard_normal((12, 4)), index=[f"m{i}" for i in range(12)], columns=list("abcd"))
    q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    # a pure rotation (and a shift) is undone by the Procrustes alignment: stability 1
    rotated = [pd.DataFrame(ref.to_numpy() @ q + 5.0, index=ref.index, columns=ref.columns) for _ in range(4)]
    stab = PW.embedding_stability(ref, rotated)
    assert stab["status"] == "computed" and stab["stability"] == pytest.approx(1.0, abs=1e-6)
    assert stab["gate"] == "RELIABLE" and stab["n_replicates"] == 4 and stab["n_units"] == 12
    # real agreement sits far above its permuted-label null
    assert stab["stability_above_null"] > 0.3 and not stab["within_null_band"]
    # The alignment MAXIMISES agreement, so pure noise does not score 0: at 12 units x 4 dimensions it reaches ~0.42,
    # above the registered 0.3 floor. The permuted-label null is what separates signal from that bias, and a raw value
    # inside the null band is flagged instead of being read as agreement.
    noisy = [pd.DataFrame(rng.standard_normal((12, 4)), index=ref.index, columns=ref.columns) for _ in range(8)]
    ns = PW.embedding_stability(ref, noisy)
    assert ns["stability"] > PW.RELIABILITY_FLOOR and ns["gate"] == "RELIABLE"      # the bias alone clears the floor
    # ... but it is indistinguishable from its own null, which is what the report must show
    assert abs(ns["stability_above_null"]) < 0.1 and ns["null_stability"] > PW.RELIABILITY_FLOOR
    # the bias shrinks as the units outnumber the dimensions, and the null tracks it
    wide = pd.DataFrame(rng.standard_normal((120, 4)), index=[f"w{i}" for i in range(120)], columns=list("abcd"))
    wide_noise = [pd.DataFrame(rng.standard_normal((120, 4)), index=wide.index, columns=wide.columns) for _ in range(8)]
    wn = PW.embedding_stability(wide, wide_noise)
    assert wn["stability"] < ns["stability"] and abs(wn["stability_above_null"]) < 0.1
    assert wn["stability"] < PW.RELIABILITY_FLOOR and wn["gate"] == "UNDECIDED_UNRELIABLE"
    assert wn["n_dimensions"] == 4 and wn["n_units"] == 120
    # the null is seeded and reported
    assert PW.embedding_stability(ref, noisy)["null_stability"] == ns["null_stability"]
    assert ns["null_seed"] == PW.SPLIT_HALF_SEED and "upward-biased" in ns["note"]
    aligned = PW.procrustes_align(ref.to_numpy(), ref.to_numpy() @ q)
    assert np.allclose(aligned, ref.to_numpy() - ref.to_numpy().mean(axis=0), atol=1e-8)
    with pytest.raises(ValueError, match="shapes differ"):
        PW.procrustes_align(ref.to_numpy(), ref.to_numpy()[:, :2])
    assert PW.embedding_stability(ref, [])["status"] == "NOT_RUN"
    assert PW.embedding_stability(pd.DataFrame(), rotated)["status"] == "NOT_RUN"
    # a replicate covering too few shared units is skipped, not silently aligned
    assert PW.embedding_stability(ref, [rotated[0].iloc[:2]])["status"] == "NOT_RUN"


def test_quantity_series_helpers():
    slopes = pd.DataFrame({"system": ["s1", "s2", "s3"], "anion": ["nitrate"] * 3, "n": [3.0, 2.5, 9.9],
                          "n_status": ["fitted", "fitted", "assumed"], "p_eff": [2.0, 1.5, 2.0],
                          "p_eff_status": ["fitted", "assumed", "fitted"]})
    n = PW.b7_slope_series(slopes, "n")
    assert list(n.index) == ["s1 | nitrate", "s2 | nitrate"]      # an 'assumed' slope is the prior, not an estimate
    assert list(PW.b7_slope_series(slopes, "p_eff").index) == ["s1 | nitrate", "s3 | nitrate"]
    assert PW.b7_slope_series(pd.DataFrame(), "n").empty
    pairs = pd.DataFrame({EM.SYSTEM_COL: ["s1", "s1", "s2"], "logsf_obs": [0.4, -0.6, 2.0]})
    amp = PW.logsf_amplitudes(pairs)
    assert amp["s1"] == pytest.approx(0.5) and amp["s2"] == pytest.approx(2.0)
    assert PW.logsf_amplitudes(pd.DataFrame()).empty
    sup = pd.DataFrame({EM.METAL_STATE_COL: ["Nd(III)", "Nd(III)", "Eu(III)"], EM.SYSTEM_COL: ["s1", "s1", "s2"],
                        "support_s1": [0.2, 0.4, 0.9]})
    s1 = PW.support_component_series(sup, "s1")
    assert s1[f"Nd(III){EM.UNIT_KEY_SEP}s1"] == pytest.approx(0.3)
    assert PW.support_component_series(sup, "nope").empty


# --------------------------------------------------------------------------------------------- #
# the gates
# --------------------------------------------------------------------------------------------- #

def test_power_runner_refuses_without_the_seal_and_without_the_contrast_files(tmp_path, monkeypatch):
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="refused"):
        RP.main(["--dry-run", "--out-root", str(out)], check=lambda: 1)
    # the H3 gates run first (discovery complete, scorer decisions), then the contrast files
    monkeypatch.setattr(RP.h3_module(), "refuse_unless_ready",
                        lambda o, **k: {"prereg_gate": {"prereg_sha256": D.REGISTERED_PREREG_SHA256}})
    # ``main`` passes the POWER stage's own gate down (task X finding V-P05); without one this function takes it, and
    # the power stage's seal gate then refuses on its own account
    with pytest.raises(SystemExit, match="not the registered text"):
        RP.refuse_unless_ready(out, check=lambda: 0, digests=lambda: {})
    mine = {"stage": RP.STAGE, "prereg_sha256": D.REGISTERED_PREREG_SHA256}
    with pytest.raises(SystemExit, match="contrast files are missing"):
        RP.refuse_unless_ready(out, check=lambda: 0, digests=lambda: {}, prereg_gate=mine)
    (D.discovery_root(out)).mkdir(parents=True, exist_ok=True)
    (D.discovery_root(out) / "contrasts_registered.csv").write_text("family\n", encoding="utf-8")
    gate = RP.refuse_unless_ready(out, check=lambda: 0, digests=lambda: {}, prereg_gate=mine)
    assert gate["contrast_files"]["contrasts_registered.csv"]
    assert gate["prereg_gate"]["stage"] == "power" and gate["h3_prereg_gate"]["prereg_sha256"]


def test_power_paths_and_arm_specs():
    assert PW.power_root(paths.G19_ROOT) == paths.G19_ROOT / "evaluation" / "power"
    assert RP.parse_arm("M2") == ("M2", "WITH")
    assert RP.parse_arm("M2:WITHOUT") == ("M2", "WITHOUT")
    assert RP.parse_arm("M0") == ("B5", "WITH")                   # M0 is B5
    with pytest.raises(ValueError, match="not an H3 transform"):
        RP.parse_arm("M2:NOPE")
    row = {"key": "M2 vs B3i@V5", "contrast": "M2 vs B3i", "family": "primary", "design": "V5", "candidate": "M2",
           "comparator": "B3i", "margin": 0.1057, "point": 0.01, "r19_verdict_full": "UNDECIDED"}
    e = RP.contrast_entry(row)
    assert e["design"] == "V5" and e["margin"] == pytest.approx(0.1057) and e["seed"] == D.PRIMARY_SEED
    assert e["uninjected"]["r19_verdict_full"] == "UNDECIDED"
    # a V5-P row maps onto the V5 fold family; a missing margin falls back to the registered floor
    e2 = RP.contrast_entry({**row, "design": "V5-P", "margin": float("nan")})
    assert e2["design"] == "V5" and e2["design_label"] == "V5-P" and e2["margin"] == pytest.approx(ET.MARGIN_FLOOR)
    # task X finding V-03: the injected comparator is refitted on the exact leave-one-cell-out folds (section 3.1),
    # never on the heavy arm's batched folds; its V1 scoring scheme is its own (exact)
    passed = D.PlanState(v5_batched_check="passed", v1_tenfold_check="passed")
    assert H3.with_job("B3i", "V5", passed).scheme == "exact" and H3.with_job("B3i", "V5", passed).fold_seed is None
    assert H3.with_job("M2", "V5", passed).scheme == "batched"
    assert H3.with_job("B0", "V1", passed).design_dir == "V1__copy_exact"
    assert H3.v1_scheme_of("B3i", "V1", passed) == "exact" and H3.v1_scheme_of("M2", "V1", passed) == "grouped"
    assert "V-03" in H3.READINGS["comparator_folds"]


def test_a_contrast_that_is_not_a_null_is_not_labelled_an_informative_null():
    """Task X findings V-L3 / V-P04.  Section 8 scopes the check to a contrast "reported as a null"; addendum 2's
    ``needs_power`` widened the TRIGGER (family in primary / H1b / S1(b) / H3 with full verdict not PASS), not the
    verdict vocabulary.  Three of the four checked contrasts are the freezing candidates, whose un-injected
    freezing-screen verdict is PASS -- labelling them INFORMATIVE_NULL made the persisted artefacts call the frozen
    claim C1 a null."""
    def res(kappa, passed):
        return PW.KappaResult(kappa=kappa, passed=passed, scope_verdict="PASS" if passed else "FAIL",
                              full_verdict="UNDECIDED", point=0.26, margin=0.1057, n_units=105, n_rows=1200)
    every = [res(k, True) for k in PW.KAPPAS]
    # the un-injected contrast PASSES its reported scope: it is not a null, and says so -- kappa_min is still reported
    ok = PW.kappa_min(every, uninjected_verdict="PASS")
    assert ok["verdict"] == "POWERED_NOT_A_NULL" and ok["kappa_min"] == 0.1 and ok["informative"] is True
    assert ok["uninjected_is_a_null"] is False and "not a null" in ok["reported"]
    assert "the null is informative" not in ok["reported"]
    weak = PW.kappa_min([res(0.1, False), res(0.25, False), res(0.5, False), res(1.0, True)], uninjected_verdict="PASS")
    assert weak["verdict"] == "NOT_A_NULL_UNDERPOWERED" and weak["informative"] is False
    # a FAIL is a null and keeps the registered vocabulary
    null = PW.kappa_min(every, uninjected_verdict="FAIL")
    assert null["verdict"] == "INFORMATIVE_NULL" and null["uninjected_is_a_null"] is True
    assert "the null is informative" in null["reported"]
    under = PW.kappa_min([res(k, False) for k in PW.KAPPAS], uninjected_verdict="FAIL")
    assert under["verdict"] == "UNDECIDED_UNDERPOWERED" and under["kappa_min"] is None
    # unknown: the two null labels as before, and the record says the question was not answered
    unknown = PW.kappa_min(every)
    assert unknown["verdict"] == "INFORMATIVE_NULL" and unknown["uninjected_is_a_null"] is None
    assert set(ET.POWER_VERDICTS) == {"INFORMATIVE_NULL", "UNDECIDED_UNDERPOWERED", "POWERED_NOT_A_NULL",
                                      "NOT_A_NULL_UNDERPOWERED"}
    # power_record takes the un-injected verdict from the contrast row the scorer wrote
    rec = PW.power_record("M2 vs B3i@V5", family="primary", design="V5", arms=["M2", "B3i"], results=every, seed=104729,
                          u_share={}, n_dropped_unknown_state=0,
                          uninjected={"point": 0.262976, f"verdict_{H3.VERDICT_SCOPE}": "PASS",
                                      "r19_verdict_full": "UNDECIDED"})
    assert rec["verdict"] == "POWERED_NOT_A_NULL" and rec["uninjected_is_a_null"] is False
    fail = PW.power_record("B6 vs B3i@V5", family="H1b", design="V5", arms=["B6", "B3i"], results=every, seed=104729,
                           u_share={}, n_dropped_unknown_state=0,
                           uninjected={"point": -0.104441, f"verdict_{H3.VERDICT_SCOPE}": "FAIL",
                                       "r19_verdict_full": "FAIL"})
    assert fail["verdict"] == "INFORMATIVE_NULL" and fail["uninjected_is_a_null"] is True


def test_the_bootstrap_input_of_every_kappa_is_persisted_as_metrics():
    """Task X finding V-L2: the check persisted no injected prediction and no injected per-cell MAE, so every reported
    Delta and every R19 item behind kappa_min could only be re-derived by repeating the fits.  The per-unit MAE of both
    arms and the cluster labels the paired bootstrap resamples are metrics, not injected values, so they can be written
    (brief section 33 is unchanged and the same guard runs on the frame)."""
    idx = [f"cell{i}" for i in range(6)]
    pu = D.PairedUnits(design="V5", candidate="M2", comparator="B3i",
                       cand_mae=pd.Series([0.3, 0.4, 0.5, 0.2, 0.6, 0.35], index=idx),
                       comp_mae=pd.Series([0.6, 0.5, 0.9, 0.3, 0.8, 0.40], index=idx),
                       clusters={"system": pd.Series(list("aabbcc"), index=idx),
                                 "publication_group": pd.Series(list("pqpqpq"), index=idx)}, n_rows=120)
    fr = PW.per_unit_rows(pu, contrast="M2 vs B3i@V5", kappa=0.25, candidate="M2", comparator="B3i", design="V5")
    assert list(fr.columns)[:5] == ["contrast", "candidate", "comparator", "design", "kappa"]
    assert set(fr.columns) >= {"unit", "mae_candidate", "mae_comparator", "delta_unit", "cluster_system",
                               "cluster_publication_group"}
    assert len(fr) == 6 and (fr["kappa"] == 0.25).all() and fr["unit"].tolist() == idx
    # the paired difference is the bootstrap's own statistic: MAE(comparator) - MAE(candidate)
    assert np.allclose(fr["delta_unit"].to_numpy(), pu.comp_mae.to_numpy() - pu.cand_mae.to_numpy())
    # ... so the reported point estimate re-derives from the persisted rows without any refit
    assert math.isclose(float(fr["delta_unit"].mean()), float(pu.comp_mae.mean() - pu.cand_mae.mean()))
    # and it carries no injected target, signal or per-row prediction
    PW.assert_no_injected_values(fr, "test")
    with pytest.raises(AssertionError, match="injected values are never persisted"):
        PW.assert_no_injected_values(fr.assign(log_D_injected=0.0), "test")
    rec = PW.power_record("M2 vs B3i@V5", family="primary", design="V5", arms=["M2", "B3i"],
                          results=[PW.KappaResult(kappa=k, passed=True, scope_verdict="PASS", full_verdict="UNDECIDED",
                                                  point=0.2, margin=0.05, n_units=6, n_rows=120) for k in PW.KAPPAS],
                          seed=104729, u_share={}, n_dropped_unknown_state=0)
    assert rec["per_unit_metrics_persisted"] is True and rec["per_unit_metrics"] == PW.PER_UNIT_REL
    assert "re-derivable without repeating the refits" in rec["readings"]["per_unit_metrics"]


def test_the_power_run_records_its_own_stages_seal_gate_not_h3s(tmp_path, monkeypatch):
    """Task X finding V-P05: ``refuse_unless_ready`` returned the H3 stage's gate unchanged and that dict was written
    to both the manifest and ``power_checks.json``, so the audit trail named the wrong registry entry -- while the
    stage-correct ``registry.refuse_unless_sealed("power", ...)`` did run in ``main`` and had its return value
    discarded.  Addendum 2 item 5 requires a record to be verified against the registry entry of ITS stage."""
    monkeypatch.setattr(RP.h3_module(), "refuse_unless_ready",
                        lambda o, **k: {"prereg_gate": {"stage": "h3", "n_addenda": 3}, "discovery_complete": {}})
    root = tmp_path / "out"
    (root / "evaluation" / "discovery").mkdir(parents=True)
    for f in RP.CONTRAST_FILES:
        (root / "evaluation" / "discovery" / f).write_text("key\n", encoding="utf-8")
    mine = {"stage": "power", "n_addenda": 3, "seal_check_exit": 0}
    gate = RP.refuse_unless_ready(root, prereg_gate=mine)
    assert gate["prereg_gate"] == mine and gate["prereg_gate"]["stage"] == RP.STAGE == "power"
    assert gate["h3_prereg_gate"]["stage"] == "h3"               # H3's is kept, beside, under its own key

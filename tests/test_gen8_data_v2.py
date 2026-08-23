"""Tests for the gen8 data-v2 structural-absence flags and recovered bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen8.data_v2 import (
    ABSENT,
    FLAG_FAMILIES,
    PRESENT,
    RECOVERED_CELLS_PATH,
    SECOND_SPECIES_CLASSES,
    STRUCTURAL_FLAG_COLUMNS,
    STRUCTURAL_FLAG_NUMERIC_COLUMNS,
    RECOVERED_LABEL_COLUMNS,
    STRUCTURAL_LABEL_COLUMNS,
    UNRECORDED,
    apply_sentinel_policy,
    attach_recovered_v2,
    attach_structural_flags,
    flag_prevalence,
)

COHORT_PATH = (Path(__file__).resolve().parents[1]
               / "runs" / "gen7_architecture" / "cache" / "cohort.parquet")


# --------------------------------------------------------------------------- #
# Constructed input
# --------------------------------------------------------------------------- #

def _constructed() -> pd.DataFrame:
    """Four rows chosen so every family exercises the state we assert.

    row 0  fully specified nitrate experiment with a recorded additive
    row 1  no additive recorded, plain single-component diluent  -> structural
           absence of the phase modifier and of its concentration
    row 2  no additive recorded but a mixed diluent carrying 30 vol% octanol
           -> the modifier IS present; its concentration is unrecorded
    row 3  nothing recorded at all: NaN everywhere, acid block all zero
    """
    return pd.DataFrame({
        "row_id": ["r0", "r1", "r2", "r3"],
        "log_D": [0.5, -1.0, 2.0, 0.0],
        "cond__acid__hno3": [1, 1, 1, 0],
        "cond__acid__hcl": [0, 0, 0, 0],
        "cond__acid_concentration_M": [3.0, 1.0, 0.5, np.nan],
        "cond__additive__tbp": [1, 0, 0, 0],
        "cond__additive__octanol": [0, 0, 0, 0],
        "cond__diluent__n_dodecane": [1, 1, 0, 1],
        "cond__diluent__kerosene_with_30_vol_1_octanol": [0, 0, 1, 0],
        "cond__contact_time_min": [30.0, np.nan, 15.0, np.nan],
        "cond__temperature_C": [25.0, np.nan, 25.0, np.nan],
        "cond__metal_concentration_mM": [1.0, 0.5, np.nan, np.nan],
        "cond__extractant_concentration_M": [0.1, 0.1, 0.1, np.nan],
        "rec__has_phase_modifier": [1.0, 0.0, 0.0, 0.0],
        "rec__phase_modifier_concentration_M": [0.5, np.nan, np.nan, np.nan],
        "rec__shaking_time_min": [np.nan, 30.0, np.nan, np.nan],
        "rec__solvent_n_components": [1.0, 1.0, 2.0, 1.0],
        "rec__solvent_polar_fraction": [0.0, 0.0, 0.3, 0.0],
        "rec__name_mismatch": [0.0, 1.0, 1.0, np.nan],
        "rec__aqueous_complexant": [0.0, 1.0, 0.0, np.nan],
        "rec__n_names_for_structure": [1.0, 21.0, 2.0, np.nan],
    })


def _state(frame: pd.DataFrame, family: str) -> list[int]:
    codes = []
    for i in range(len(frame)):
        row = frame.iloc[i]
        hot = [row[f"sflag__{family}__present"], row[f"sflag__{family}__absent"],
               row[f"sflag__{family}__unrecorded"]]
        assert sum(int(v) for v in hot) == 1, (family, i, hot)
        codes.append([PRESENT, ABSENT, UNRECORDED][int(np.argmax(hot))])
    return codes


def test_emits_exactly_the_declared_columns():
    out, created = attach_structural_flags(_constructed())
    assert created == STRUCTURAL_FLAG_COLUMNS
    assert all(c in out.columns for c in STRUCTURAL_FLAG_COLUMNS)
    assert set(STRUCTURAL_FLAG_NUMERIC_COLUMNS) | set(STRUCTURAL_LABEL_COLUMNS) \
        == set(STRUCTURAL_FLAG_COLUMNS)


def test_states_are_mutually_exclusive_and_exhaustive():
    out, _ = attach_structural_flags(_constructed())
    for family in FLAG_FAMILIES:
        triple = out[[f"sflag__{family}__present", f"sflag__{family}__absent",
                      f"sflag__{family}__unrecorded"]].to_numpy(dtype=int)
        assert (triple.sum(axis=1) == 1).all(), family


def test_tri_state_correctness_on_constructed_input():
    out, _ = attach_structural_flags(_constructed())

    # the acid is recorded on rows 0-2; row 3 has an all-zero block and no
    # concentration, which is silence, not a neutral aqueous phase
    assert _state(out, "has_acid") == [PRESENT, PRESENT, PRESENT, UNRECORDED]
    assert list(out["sflag__acid_class"]) == ["nitrate", "nitrate", "nitrate", "unrecorded"]
    assert list(out["sflag__acid_class__nitrate"]) == [1, 1, 1, 0]
    assert _state(out, "has_acid_concentration") == [PRESENT, PRESENT, PRESENT, UNRECORDED]

    # the recorded vocabulary: blank means the curator recorded no modifier
    assert _state(out, "has_additive") == [PRESENT, ABSENT, ABSENT, ABSENT]

    # the chemistry: row 2's modifier is inside the diluent name, so the
    # bookkeeping ABSENT above must not survive into the physical flag
    assert _state(out, "has_phase_modifier") == [PRESENT, ABSENT, PRESENT, ABSENT]

    # ...and the concentration is structurally zero on 1 and 3, genuinely
    # unrecorded on 2 (a vol% is not a molarity)
    assert _state(out, "has_phase_modifier_concentration") == [
        PRESENT, ABSENT, UNRECORDED, ABSENT]

    # times / temperature / concentrations: a blank is silence, never absence
    assert _state(out, "has_contact_time") == [PRESENT, UNRECORDED, PRESENT, UNRECORDED]
    assert _state(out, "has_shaking_time") == [UNRECORDED, PRESENT, UNRECORDED, UNRECORDED]
    assert _state(out, "has_temperature") == [PRESENT, UNRECORDED, PRESENT, UNRECORDED]
    assert _state(out, "has_metal_concentration") == [PRESENT, PRESENT, UNRECORDED, UNRECORDED]
    assert _state(out, "has_extractant_concentration") == [
        PRESENT, PRESENT, PRESENT, UNRECORDED]

    # aqueous complexant: present by name evidence, absent when the structure
    # has a single name corpus-wide, unrecorded when it is ambiguous
    assert _state(out, "has_aqueous_complexant") == [ABSENT, PRESENT, UNRECORDED, UNRECORDED]

    # which timing descriptor the record used
    assert list(out["sflag__timing_descriptor__contact"]) == [1, 0, 1, 0]
    assert list(out["sflag__timing_descriptor__shaking"]) == [0, 1, 0, 0]
    assert list(out["sflag__timing_descriptor__none"]) == [0, 0, 0, 1]

    assert list(out["sflag__second_species_present"]) == [0, 1, 1, 0]


def test_no_evidence_means_unrecorded_never_absent():
    bare = pd.DataFrame({"row_id": ["a", "b"], "log_D": [1.0, 2.0]})
    out, created = attach_structural_flags(bare)
    assert created == STRUCTURAL_FLAG_COLUMNS
    for family in FLAG_FAMILIES:
        assert list(out[f"sflag__{family}__unrecorded"]) == [1, 1], family
        assert list(out[f"sflag__{family}__absent"]) == [0, 0], family
    assert list(out["sflag__second_species_present"]) == [0, 0]
    assert out["sflag__n_names_for_structure"].isna().all()


def test_acid_absent_when_concentration_recorded_as_zero():
    frame = pd.DataFrame({
        "cond__acid__hno3": [0, 0],
        "cond__acid_concentration_M": [0.0, 2.0],
    })
    out, _ = attach_structural_flags(frame)
    # zero acid -> structurally absent; positive acid with no identity -> present
    assert _state(out, "has_acid") == [ABSENT, PRESENT]
    assert list(out["sflag__acid_class"]) == ["none", "unrecorded"]
    assert _state(out, "has_acid_concentration") == [ABSENT, PRESENT]


def test_row_and_index_preserving_and_idempotent():
    frame = _constructed().set_index(pd.Index([10, 11, 12, 13], name="ix"))
    out, _ = attach_structural_flags(frame)
    assert len(out) == len(frame)
    assert out.index.equals(frame.index)
    twice, _ = attach_structural_flags(out)
    pd.testing.assert_frame_equal(out[list(STRUCTURAL_FLAG_COLUMNS)],
                                  twice[list(STRUCTURAL_FLAG_COLUMNS)])


def test_log_d_is_never_touched():
    frame = _constructed()
    before = frame["log_D"].copy()
    out, _ = attach_structural_flags(frame)
    pd.testing.assert_series_equal(out["log_D"], before)
    pd.testing.assert_series_equal(frame["log_D"], before)  # input not mutated


def test_sentinel_policy_zeroes_only_structural_absence():
    out, _ = attach_structural_flags(_constructed())
    out, created = apply_sentinel_policy(out)
    assert "sentinel__rec__phase_modifier_concentration_M" in created
    values = out["sentinel__rec__phase_modifier_concentration_M"]
    assert values.iloc[0] == 0.5          # present -> untouched
    assert values.iloc[1] == 0.0          # structurally absent -> 0 M
    assert np.isnan(values.iloc[2])       # unrecorded -> stays NaN
    assert values.iloc[3] == 0.0
    contact = out["sentinel__cond__contact_time_min"]
    assert np.isnan(contact.iloc[1]) and np.isnan(contact.iloc[3])


def test_sentinel_policy_requires_flags_first():
    with pytest.raises(KeyError):
        apply_sentinel_policy(_constructed())


# --------------------------------------------------------------------------- #
# Real cohort
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    if not COHORT_PATH.exists():
        pytest.skip(f"cohort not available at {COHORT_PATH}")
    return pd.read_parquet(COHORT_PATH)


@pytest.fixture(scope="module")
def recovered() -> pd.DataFrame:
    if not RECOVERED_CELLS_PATH.exists():
        pytest.skip(f"recovered cells not available at {RECOVERED_CELLS_PATH}")
    return pd.read_parquet(RECOVERED_CELLS_PATH)


def test_recovered_join_is_row_preserving(cohort):
    joined, blocks = attach_recovered_v2(cohort)
    assert len(joined) == len(cohort)
    assert joined.index.equals(cohort.index)
    assert list(joined["row_id"]) == list(cohort["row_id"])
    assert set(blocks) == {"RECOVERED_V2", "SECOND_SPECIES"}
    assert all(c in joined.columns for block in blocks.values() for c in block)
    # the nuisance keys stay out of the feature blocks and out of the frame
    assert not any(c.startswith("nuisance__") for c in joined.columns)
    assert not any(c.startswith("nuisance__") for block in blocks.values() for c in block)
    pd.testing.assert_series_equal(joined["log_D"], cohort["log_D"])


def test_recovered_join_can_carry_nuisance_when_asked(cohort):
    joined, blocks = attach_recovered_v2(cohort.head(200), include_nuisance=True)
    assert "nuisance__doi" in joined.columns
    assert not any(c.startswith("nuisance__") for block in blocks.values() for c in block)


def test_second_species_matches_the_recovered_table_exactly(cohort, recovered):
    joined, blocks = attach_recovered_v2(cohort)
    flagged = set(joined.loc[joined["second_species_present"] == 1, "row_id"])
    expected = set(recovered.loc[recovered["rec__name_mismatch"] > 0, "row_id"]) \
        & set(cohort["row_id"])
    assert flagged == expected
    assert len(flagged) > 0
    # and the complement is exactly zero, not merely mostly zero
    assert int((joined["second_species_present"] == 1).sum()) == len(expected)
    assert set(joined.loc[joined["second_species_present"] == 0, "row_id"]) \
        == set(cohort["row_id"]) - expected

    # class one-hots partition the rows
    one_hots = joined[[f"second_species_class__{c}" for c in SECOND_SPECIES_CLASSES]]
    assert (one_hots.to_numpy(dtype=int).sum(axis=1) == 1).all()
    aqueous = set(joined.loc[joined["second_species_class"] == "aqueous_complexant", "row_id"])
    assert aqueous == set(recovered.loc[recovered["rec__aqueous_complexant"] > 0, "row_id"]) \
        & set(cohort["row_id"])
    assert aqueous <= flagged


def test_flags_agree_with_recovered_evidence_on_the_cohort(cohort):
    joined, _ = attach_recovered_v2(cohort)
    flagged, _ = attach_structural_flags(joined)
    assert len(flagged) == len(cohort)
    # the sflag copy and the first-class column are the same quantity
    assert (flagged["sflag__second_species_present"].to_numpy()
            == flagged["second_species_present"].to_numpy()).all()
    # the recorded-additive block and the independently recovered upstream field
    # agree on every row: this is what licenses "blank means absent" there
    assert (flagged["sflag__has_additive__present"].to_numpy()
            == (flagged["rec__has_phase_modifier"] > 0).to_numpy()).all()
    # the physical modifier flag is strictly broader than the recorded one
    present_physical = flagged["sflag__has_phase_modifier__present"].to_numpy(dtype=bool)
    present_recorded = flagged["sflag__has_additive__present"].to_numpy(dtype=bool)
    assert present_physical[present_recorded].all()
    assert present_physical.sum() > present_recorded.sum()


# --------------------------------------------------------------------------- #
# Silent-failure guards
# --------------------------------------------------------------------------- #

def test_unmatched_row_ids_raise_instead_of_nan_filling(cohort):
    """A left join would fill these with NaN and every flag would degrade."""
    broken = cohort.head(10).copy()
    broken.loc[broken.index[:3], "row_id"] = ["NOT_A_ROW_%d" % i for i in range(3)]
    with pytest.raises(ValueError, match="not in"):
        attach_recovered_v2(broken)


def test_joining_twice_raises_instead_of_suffixing(cohort):
    """pandas would emit rec__*_x / rec__*_y and the block names would dangle."""
    joined, _ = attach_recovered_v2(cohort.head(50))
    with pytest.raises(ValueError, match="already carries"):
        attach_recovered_v2(joined)


def test_absent_modifier_requires_the_recovered_evidence(cohort):
    """Without the join the family must say UNRECORDED, never a wrong ABSENT.

    The additive block and the diluent one-hot names cannot see a modifier
    hidden inside ``cond__diluent__other``; 122 cohort rows are exactly that.
    """
    unjoined, _ = attach_structural_flags(cohort)
    assert int(unjoined["sflag__has_phase_modifier__absent"].sum()) == 0
    assert int(unjoined["sflag__has_phase_modifier_concentration__absent"].sum()) == 0

    joined, _ = attach_recovered_v2(cohort)
    full, _ = attach_structural_flags(joined)
    truth = full["sflag__has_phase_modifier__present"].to_numpy(dtype=bool)
    claimed_absent = unjoined["sflag__has_phase_modifier__absent"].to_numpy(dtype=bool)
    assert not (claimed_absent & truth).any()
    # PRESENT is still raised from the name evidence alone
    assert int(unjoined["sflag__has_phase_modifier__present"].sum()) == 1335


def test_idempotent_on_the_real_cohort(cohort):
    joined, _ = attach_recovered_v2(cohort)
    once, _ = attach_structural_flags(joined)
    twice, _ = attach_structural_flags(once)
    pd.testing.assert_frame_equal(once[list(STRUCTURAL_FLAG_COLUMNS)],
                                  twice[list(STRUCTURAL_FLAG_COLUMNS)])


# --------------------------------------------------------------------------- #
# Pinned measurements — these are the numbers the report quotes
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def flagged(cohort):
    joined, blocks = attach_recovered_v2(cohort)
    out, _ = attach_structural_flags(joined)
    return out, blocks


def test_column_budget_is_pinned(cohort, flagged):
    out, blocks = flagged
    assert cohort.shape == (5248, 2492)
    assert len(blocks["RECOVERED_V2"]) == 20
    assert len(blocks["SECOND_SPECIES"]) == 8
    assert out.shape[1] - cohort.shape[1] == 78          # 29 joined + 49 flags
    assert len(STRUCTURAL_FLAG_COLUMNS) == 49
    assert RECOVERED_LABEL_COLUMNS == ("second_species_class",)
    assert all(c in out.columns for c in RECOVERED_LABEL_COLUMNS)


def test_family_prevalence_is_pinned(flagged):
    out, _ = flagged
    expected = {                       # family: (PRESENT, ABSENT, UNRECORDED)
        "has_acid": (5248, 0, 0),
        "has_acid_concentration": (5245, 0, 3),
        "has_additive": (242, 5006, 0),
        "has_phase_modifier": (1457, 3791, 0),
        "has_phase_modifier_concentration": (242, 3791, 1215),
        "has_shaking_time": (1023, 0, 4225),
        "has_contact_time": (3131, 0, 2117),
        "has_temperature": (5169, 0, 79),
        "has_metal_concentration": (3907, 0, 1341),
        "has_extractant_concentration": (5247, 0, 1),
        "has_aqueous_complexant": (86, 2522, 2640),
    }
    assert set(expected) == set(FLAG_FAMILIES)
    for family, counts in expected.items():
        got = tuple(int(out[f"sflag__{family}__{s}"].sum())
                    for s in ("present", "absent", "unrecorded"))
        assert got == counts, (family, got, counts)
        assert sum(got) == len(out)


def test_acid_and_timing_labels_are_pinned(flagged):
    out, _ = flagged
    assert out["sflag__acid_class"].value_counts().to_dict() == {
        "nitrate": 4277, "chloride": 651, "carboxylate": 253, "sulfate": 66, "perchlorate": 1}
    assert int(out["sflag__acid_mixed"].sum()) == 2          # HNO3 + oxalic
    assert {t: int(out[f"sflag__timing_descriptor__{t}"].sum()) for t in ("contact", "shaking", "both", "none")} \
        == {"contact": 3131, "shaking": 1023, "both": 0, "none": 1094}


def test_second_species_cell_and_replicate_counts_are_pinned(flagged):
    out, _ = flagged
    sel = out["second_species_present"] == 1
    assert int(sel.sum()) == 353
    assert out.loc[sel, "extractant"].nunique() == 17
    # the brief's 414 are raw bundle rows; the cohort row is an averaged cell
    assert int(out.loc[sel, "n_replicates"].sum()) == 414
    # no cell mixes flagged and unflagged replicates, so the mean stays binary
    assert set(out["rec__name_mismatch"].unique()) == {0.0, 1.0}
    assert out["second_species_class"].value_counts().to_dict() == {
        "none": 4895, "name_mismatch_other": 267, "aqueous_complexant": 86}


def test_the_geom_modifier_class_side_finding_is_pinned(cohort, flagged):
    """gen7's modifier flag is wrong on 24.3 % of the rows it calls 'none'."""
    out, _ = flagged
    none_flag = cohort["geom_cond__modifier_class=none"].to_numpy(dtype=float) > 0
    assert int(none_flag.sum()) == 5006
    physical = out["sflag__has_phase_modifier__present"].to_numpy(dtype=bool)
    assert int((none_flag & physical).sum()) == 1215
    assert round(1215 / 5006, 3) == 0.243


def test_flag_prevalence_reports_which_columns_are_constant(flagged):
    """A fixed schema leaves empty states; the report must not hide them."""
    out, _ = flagged
    table = flag_prevalence(out)
    assert len(table) == len(STRUCTURAL_FLAG_NUMERIC_COLUMNS)
    constant = set(table.loc[table["constant"], "column"])
    assert constant == {
        "sflag__has_acid__present", "sflag__has_acid__absent", "sflag__has_acid__unrecorded",
        "sflag__has_acid_concentration__absent", "sflag__has_additive__unrecorded",
        "sflag__has_phase_modifier__unrecorded", "sflag__has_shaking_time__absent",
        "sflag__has_contact_time__absent", "sflag__has_temperature__absent",
        "sflag__has_metal_concentration__absent",
        "sflag__has_extractant_concentration__absent",
        "sflag__acid_class__other", "sflag__acid_class__none", "sflag__acid_class__unrecorded",
        "sflag__timing_descriptor__both",
    }
    assert len(constant) == 15
    # and nothing in the block is NaN except the name count
    nan_cols = [c for c in STRUCTURAL_FLAG_NUMERIC_COLUMNS if out[c].isna().any()]
    assert nan_cols == []


def test_sentinel_policy_on_the_cohort_destroys_no_recorded_value(flagged):
    out, _ = flagged
    sealed, created = apply_sentinel_policy(out)
    assert len(created) == 7
    for name in created:
        source = name[len("sentinel__"):]
        before = pd.to_numeric(out[source], errors="coerce")
        after = sealed[name]
        assert int((before.notna() & after.isna()).sum()) == 0, source
    # only the phase-modifier concentration has a structural absence to fill
    filled = {name[len("sentinel__"):]: int((pd.to_numeric(out[name[len("sentinel__"):]],
                                                           errors="coerce").isna()
                                             & sealed[name].notna()).sum())
              for name in created}
    assert filled["rec__phase_modifier_concentration_M"] == 3791
    assert int(sealed["sentinel__rec__phase_modifier_concentration_M"].isna().sum()) == 1215
    assert sum(v for k, v in filled.items() if k != "rec__phase_modifier_concentration_M") == 0


def test_second_species_414_reconciles_against_the_raw_bundle(flagged):
    """Re-derive the modal-name mismatch from the bundle, not from the cache."""
    bundle = COHORT_PATH.parents[3] / "dataset with 3D structures" / "dataset.parquet"
    if not bundle.exists():
        pytest.skip("raw bundle not available")
    raw = pd.read_parquet(bundle, columns=["extractant_name", "canonical_smiles", "log_D"])
    names = raw["extractant_name"].fillna("unknown").astype(str)
    structures = raw["canonical_smiles"].astype(str)
    modal = names.groupby(structures).agg(lambda s: s.value_counts().idxmax())
    mismatch = names.to_numpy() != structures.map(modal).to_numpy()
    assert len(raw) == 5992
    assert int(mismatch.sum()) == 414
    assert int((mismatch & (raw["log_D"] > -6).to_numpy()).sum()) == 414   # none lost
    out, _ = flagged
    assert int(out.loc[out["second_species_present"] == 1, "n_replicates"].sum()) == 414

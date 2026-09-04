"""Contract tests for ``scripts/gen8_recommend_experiment.py``.

The recommender is the only piece of gen8 a chemist would actually run, so the
things worth protecting are not its arithmetic but its *promises*:

* ``--demo`` runs end to end and produces a JSON a stranger can read;
* the features it builds for a ligand it has never modelled are the **same
  features the arm was trained on** — 2,143 of 2,146 columns reproduce the
  cohort's own row exactly, and the three that do not are the literature-
  provenance columns that cannot exist for a prospective experiment.  This is the
  test that would catch a silently-misencoded one-hot, which would otherwise show
  up only as a slightly worse prediction that nobody could attribute.  Keep the
  tolerated set minimal: widening it is how a real encoding bug gets reclassified
  as an unavoidable limitation;
* the numbers the output quotes are the ones the artefacts hold, the deployed
  policy reproduces the study it cites, and the reasons it gives for *not* doing
  something are the reasons the measurements actually support;
* the queried ligand's whole Tanimoto chemotype really is removed from training,
  so the demo is under the hold-out contract the quoted numbers were measured in;
* the acquisition rules are the ones :mod:`lanthanide_separation.gen8.kshot`
  measured — the first point is that module's own ``policy_medoid`` — rather than
  a lookalike written in the script;
* the adapter escalates from ``OFFSET_K1`` to ``OFFSET_K3`` at k = 2;
* no target reaches the recommendation path.

The model is fitted with a reduced forest so the suite stays fast; the script
records ``is_frozen_recipe: false`` when that happens, and one test asserts that
the *default* is the frozen 400 trees.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
COHORT_CACHE = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"


def _load_script(name: str) -> ModuleType:
    """Import a file under ``scripts/`` as a module without a package."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"gen8_script_{name}", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(
    not COHORT_CACHE.exists(),
    reason="the frozen gen7 cohort cache is not present in this checkout")

recommender = _load_script("gen8_recommend_experiment")

#: Small enough to keep the suite quick, large enough that the medoid and the
#: farthest point are different rows and the geography strata are non-degenerate.
DEMO_CANDIDATES = 16
FAST_TREES = 60


def _run(tmp_path: Path, *extra: str) -> dict:
    out = tmp_path / "recommended_condition.json"
    argv = ["--demo", "--quiet", "--out", str(out),
            "--demo-max-candidates", str(DEMO_CANDIDATES),
            "--n-estimators", str(FAST_TREES), *extra]
    assert recommender.main(argv) == 0
    return json.loads(out.read_text())


@pytest.fixture(scope="module")
def stage_zero(tmp_path_factory) -> dict:
    """``--demo`` with nothing measured: the first-experiment recommendation."""
    return _run(tmp_path_factory.mktemp("stage0"))


@pytest.fixture(scope="module")
def stage_two(tmp_path_factory) -> dict:
    """``--demo`` after two measurements: the third-experiment recommendation."""
    return _run(tmp_path_factory.mktemp("stage2"), "--demo-observe-truth", "2")


# --------------------------------------------------------------------------- #
# it runs, and the JSON is usable
# --------------------------------------------------------------------------- #

def test_demo_runs_end_to_end(stage_zero):
    assert stage_zero["schema_version"] == recommender.SCHEMA_VERSION
    for key in ("query", "model", "stage", "recommendation", "observations",
                "predictions", "evidence", "warnings", "field_documentation"):
        assert key in stage_zero, f"the output JSON has no {key!r} section"


def test_every_candidate_gets_a_prediction_with_an_uncertainty(stage_zero):
    predictions = stage_zero["predictions"]
    assert len(predictions) == stage_zero["query"]["n_candidates"] == DEMO_CANDIDATES
    ids = {p["candidate_id"] for p in predictions}
    assert len(ids) == len(predictions), "candidate ids are not unique in the output"
    for entry in predictions:
        assert isinstance(entry["predicted_log_D"], float)
        assert isinstance(entry["zero_shot_log_D"], float)
        assert entry["uncertainty_68"] is not None and entry["uncertainty_68"] > 0
        assert entry["uncertainty_90"] >= entry["uncertainty_68"], \
            "the 90 % interval must be at least as wide as the 68 % one"


def test_exactly_one_candidate_is_recommended(stage_zero):
    flagged = [p for p in stage_zero["predictions"] if p["is_recommended"]]
    assert len(flagged) == 1
    assert flagged[0]["candidate_id"] == stage_zero["recommendation"]["candidate_id"]


def test_field_documentation_covers_the_prediction_rows(stage_zero):
    documented = set(stage_zero["field_documentation"])
    emitted = set(stage_zero["predictions"][0])
    assert emitted <= documented, f"undocumented output fields: {sorted(emitted - documented)}"


# --------------------------------------------------------------------------- #
# the deployed features are the trained features
# --------------------------------------------------------------------------- #

#: The only columns a prospective experiment genuinely cannot supply: they describe
#: how the *paper* recorded the row, not the chemistry.  ``rec__shaking_time_min`` and
#: ``rec__has_shaking_time`` are deliberately NOT here — the first is an experimental
#: condition the user can state and the second is a deterministic function of it — and
#: an earlier version of this list included them, which hid a real encoding bug.
PROVENANCE_ONLY = {
    "rec__acid_concentration_organic_M", "rec__name_mismatch",
    "rec__n_names_for_structure", "rec__aqueous_complexant",
}


def test_candidate_features_reproduce_the_cohort_row(stage_zero):
    check = stage_zero["demo"]["reconstruction_check"]
    assert check["max_abs_difference_over_reproduced_columns"] == 0.0
    assert check["n_reproduced_exactly"] >= check["n_columns"] - len(PROVENANCE_ONLY)
    assert set(check["mismatched_columns"]) <= PROVENANCE_ONLY, (
        "a feature that CAN be reconstructed prospectively is being built differently "
        "from the way the cohort built it: "
        f"{sorted(set(check['mismatched_columns']) - PROVENANCE_ONLY)}")


def test_the_reconstruction_check_is_reported_with_its_discriminating_power(stage_zero):
    """A raw count of reproduced columns flatters itself; hold it to the useful subset.

    Most of the 2,146 arm columns are ECFP bits that are zero for every ligand in the
    cohort, so "2,143 of 2,146 reproduce" is mostly a statement about constants.  The
    columns that vary within the queried ligand's own rows are where a mis-encoded
    condition or a wrong mass-action product would actually show up, and every one of
    those must reproduce.
    """
    power = stage_zero["demo"]["reconstruction_check"]["discriminating_power"]
    assert power["n_columns_varying_within_this_ligand"] >= 8, \
        "the demo's candidate grid is too degenerate for this check to mean anything"
    assert power["n_within_ligand_varying_reproduced"] == \
        power["n_columns_varying_within_this_ligand"]
    assert power["n_columns_varying_across_the_cohort"] < stage_zero["demo"][
        "reconstruction_check"]["n_columns"], "expected constant columns to exist"


def test_the_shaking_time_indicator_is_built_not_left_unknown(tmp_path):
    """``rec__has_shaking_time`` is fully determined, so it must never be NaN.

    In the cohort it equals ``rec__shaking_time_min.notna()`` on all 5,248 rows and
    is never missing.  Leaving it NaN for a prospective row (as a first version did)
    put a value into the arm that the arm was never trained on; it happened to impute
    back to 0.0 only because 4,225 of 5,248 training rows are 0.
    """
    path = tmp_path / "shaking.csv"
    path.write_text("metal,acid_concentration_M,extractant_concentration_M,shaking_time_min\n"
                    "Nd,0.1,0.1,\nEu,1.0,0.1,30\nDy,3.0,0.1,\n")
    conditions = recommender.read_candidate_conditions(path)
    built = recommender.build_candidate_frame(
        recommender.load_cohort(), "CCCCCCCCP(=O)(CCCCCCCC)CCCCCCCC", conditions,
        cohort_ligand_row=None).frame
    assert list(built["rec__has_shaking_time"]) == [0.0, 1.0, 0.0]
    assert not built["rec__has_shaking_time"].isna().any()


# --------------------------------------------------------------------------- #
# the hold-out contract
# --------------------------------------------------------------------------- #

def test_the_query_chemotype_is_removed_from_training(stage_zero):
    model = stage_zero["model"]
    assert model["excluded_chemotype"], "the demo ligand is in the cohort and must be withheld"
    cohort = pd.read_parquet(COHORT_CACHE, columns=["extractant", "tanimoto_cluster"])
    withheld = cohort[cohort["tanimoto_cluster"].astype(str) == model["excluded_chemotype"]]
    assert stage_zero["query"]["ligand_smiles"] in set(withheld["extractant"].astype(str))
    assert model["n_train_rows"] == len(cohort) - len(withheld)
    assert model["n_train_ligands"] == cohort["extractant"].nunique() - \
        withheld["extractant"].nunique()


def test_the_arm_and_seed_are_the_frozen_ones(stage_zero):
    model = stage_zero["model"]
    assert model["arm"] == recommender.ARM_NAME == "REC_ecfp_plus_recovered"
    assert tuple(model["blocks"]) == recommender.ARM_BLOCKS
    # gen5's formula with the frozen base seed 42, which is independent of the split
    assert model["model_seed"] == 42 + recommender.FOLD_SEED_OFFSET
    # the fast forest used here is flagged as a departure from the frozen recipe
    assert model["is_frozen_recipe"] is False


def test_the_default_forest_is_the_frozen_one():
    defaults = recommender.build_parser().parse_args(["--demo"])
    assert defaults.n_estimators == 400


# --------------------------------------------------------------------------- #
# the policy is the measured one, not a lookalike
# --------------------------------------------------------------------------- #

def test_first_point_is_the_gen8_medoid_policy(stage_zero):
    from lanthanide_separation.gen8.kshot import PolicyContext, policy_medoid
    from lanthanide_separation.gen8.protocols import _standardised_axes

    assert stage_zero["recommendation"]["rule"] == "MEDOID"
    assert stage_zero["stage"]["k_observed"] == 0
    assert stage_zero["stage"]["adapter"] == "ZERO_SHOT"

    # rebuild the choice straight from the gen8 module and demand the same answer
    cohort = recommender.load_cohort()
    conditions = recommender.read_candidate_conditions(
        Path(stage_zero["demo"]["conditions_csv"]))
    smiles = stage_zero["query"]["ligand_smiles"]
    ligand_row = cohort.frame[cohort.frame["extractant"].astype(str) == smiles].iloc[0]
    block = recommender.build_candidate_frame(cohort, smiles, conditions,
                                              cohort_ligand_row=ligand_row).frame
    n = len(block)
    context = PolicyContext(block=block, prediction=np.zeros(n), pool=np.arange(n),
                            evaluation=np.arange(n), axes=_standardised_axes(block),
                            uncertainty=np.full(n, np.nan), disagreement=np.full(n, np.nan),
                            rng=np.random.default_rng(0), truth=None)
    expected = conditions["candidate_id"].iloc[policy_medoid(context, [])]
    assert stage_zero["recommendation"]["candidate_id"] == expected


def test_second_and_later_points_spread_and_escalate_the_adapter(stage_two):
    assert stage_two["stage"]["k_observed"] == 2
    assert stage_two["stage"]["adapter"] == "OFFSET_K3"
    rules = [o["rule"] for o in stage_two["observations"]]
    assert rules == ["MEDOID", "FARTHEST_FROM_EXISTING"]
    assert stage_two["recommendation"]["rule"] == "FARTHEST_FROM_EXISTING"


def test_measured_points_are_not_recommended_again(stage_two):
    observed = {o["candidate_id"] for o in stage_two["observations"]}
    assert stage_two["recommendation"]["candidate_id"] not in observed
    flagged = {p["candidate_id"] for p in stage_two["predictions"] if p["is_observed"]}
    assert flagged == observed


def test_calibration_moves_towards_the_measurements_it_was_given(stage_two):
    """An offset/response fit on two points must not move them further from their values."""
    by_id = {p["candidate_id"]: p for p in stage_two["predictions"]}
    for observation in stage_two["observations"]:
        entry = by_id[observation["candidate_id"]]
        before = abs(entry["zero_shot_log_D"] - observation["observed_log_D"])
        after = abs(entry["predicted_log_D"] - observation["observed_log_D"])
        assert after <= before + 1e-9, "calibration moved a measured row away from its value"


def test_the_justification_quotes_a_measured_improvement(stage_zero):
    improvement = stage_zero["recommendation"]["why"]["expected_improvement"]
    if not improvement.get("available"):
        pytest.skip("the gen8 acquisition run artefact is not present in this checkout")
    assert improvement["recommended_macro_mae"] < improvement["random_macro_mae"]
    assert improvement["oracle_macro_mae"] < improvement["recommended_macro_mae"]
    assert improvement["gain_over_random"] > 0
    assert improvement["gain_bca_low"] > 0, "the quoted gain must exclude zero"
    assert improvement["seeds_positive"] == improvement["n_seeds"]


def test_uncertainty_signals_are_reported_but_not_ranked_on(stage_zero):
    """gen8 measured uncertainty-driven acquisition to fail; the script must not use it."""
    predictions = stage_zero["predictions"]
    assert all(p["ensemble_sd"] is not None for p in predictions), \
        "the ensemble spread is a promised diagnostic"
    recommended = next(p for p in predictions if p["is_recommended"])
    spreads = sorted(p["ensemble_sd"] for p in predictions)
    # not a proof, but it would catch a max-uncertainty rule sneaking back in
    assert recommended["ensemble_sd"] < spreads[-1]


def test_the_refusal_to_use_uncertainty_matches_the_measured_table(stage_zero):
    """The stated reason must be the reason the artefact actually supports.

    Two of the four uncertainty policies have a *better* point estimate than RANDOM
    at k = 1, so "none of them beat random" is not what justifies ignoring them —
    the paired intervals are.  This test pins the real justification: no uncertainty
    policy separates from RANDOM, and the deployed rule separates from every one of
    them.  If a re-run of the study ever breaks either half, the refusal has to be
    revisited rather than restated.
    """
    paired = stage_zero["recommendation"]["why"].get("uncertainty_policies_at_k1_paired")
    if not paired:
        pytest.skip("the gen8 acquisition run artefact is not present in this checkout")
    for name in ("MAX_ENSEMBLE_SD", "MIN_ENSEMBLE_SD",
                 "MAX_MODEL_DISAGREEMENT", "MAX_PREDICTIVE_VARIANCE"):
        against_random = paired[f"{name}_vs_RANDOM"]
        assert against_random["bca_low"] <= 0 <= against_random["bca_high"], (
            f"{name} now separates from RANDOM; the script's stated reason for "
            f"ignoring uncertainty no longer holds")
        against_us = paired[f"RECOMMENDED_vs_{name}"]
        assert against_us["bca_low"] > 0, (
            f"the deployed rule no longer beats {name} by an interval excluding zero")


def test_the_policy_field_it_was_chosen_from_travels_with_the_recommendation(stage_zero):
    """The rule is one of 14 measured policies; the reader gets the whole board."""
    board = stage_zero["evidence"]["acquisition"].get("deployable_policy_leaderboard")
    if not board:
        pytest.skip("the gen8 acquisition run artefact is not present in this checkout")
    for k, rows in board.items():
        ranks = [row["rank"] for row in rows]
        assert ranks == sorted(ranks) == list(range(1, len(rows) + 1))
        maes = [row["macro_mae"] for row in rows]
        assert maes == sorted(maes), f"leaderboard at k={k} is not ordered by macro MAE"
        assert sum(row["is_recommended"] for row in rows) == 1


def test_the_oracle_is_flagged_when_it_is_not_a_lower_bound(stage_zero):
    """``ORACLE[OFFSET_K1]`` scored under ``OFFSET_K3`` is not a ceiling for it."""
    by_k = stage_zero["evidence"]["acquisition"].get("by_k")
    if not by_k:
        pytest.skip("the gen8 acquisition run artefact is not present in this checkout")
    for k, entry in by_k.items():
        valid = entry["oracle_macro_mae"] <= entry["recommended_macro_mae"]
        assert entry["oracle_is_a_valid_lower_bound"] == valid, (
            f"k={k}: the oracle bound flag disagrees with the numbers beside it")
        if not valid:
            assert "NOT a lower bound" in entry["oracle_caveat"]


def test_the_deployed_policy_reproduces_the_acquisition_study(stage_zero):
    """The drift check: same policy objects, same adapters, P2, exact agreement.

    This is what makes "the policy is the measured one" a fact rather than a claim —
    replaying medoid -> farthest with OFFSET_K1 -> OFFSET_K3 over the frozen
    out-of-fold predictions has to land on ``primary_detail.parquet``'s own macro MAEs.
    """
    replay = stage_zero["evidence"]["interval_calibration"].get("p2_replay", {})
    if not replay.get("available") or "acquisition_artefact_macro_mae" not in replay:
        pytest.skip("the gen8 acquisition run artefact is not present in this checkout")
    assert replay["reproduces_the_acquisition_artefact"], (
        f"the deployed policy no longer reproduces the study it quotes; "
        f"max |difference| = {replay['max_abs_difference']}")
    assert replay["max_abs_difference"] < 1e-9


# --------------------------------------------------------------------------- #
# leakage
# --------------------------------------------------------------------------- #

def test_the_candidate_frame_carries_no_target(stage_zero):
    cohort = recommender.load_cohort()
    conditions = recommender.read_candidate_conditions(
        Path(stage_zero["demo"]["conditions_csv"]))
    smiles = stage_zero["query"]["ligand_smiles"]
    built = recommender.build_candidate_frame(cohort, smiles, conditions,
                                              cohort_ligand_row=None)
    assert "log_D" not in built.frame.columns


def test_a_target_column_in_the_users_csv_is_dropped(tmp_path):
    table = pd.DataFrame({"metal": ["Nd", "Eu", "Dy"],
                          "acid_concentration_M": [0.1, 1.0, 3.0],
                          "extractant_concentration_M": [0.1, 0.1, 0.1],
                          "log_D": [9.9, 9.9, 9.9]})
    path = tmp_path / "with_target.csv"
    table.to_csv(path, index=False)
    conditions = recommender.read_candidate_conditions(path)
    assert "log_D" not in conditions.columns


def test_demo_truth_is_labelled_as_such(stage_two):
    assert all(o["source"] == "cohort_truth (demo only)" for o in stage_two["observations"])


# --------------------------------------------------------------------------- #
# the user-facing CSV contract
# --------------------------------------------------------------------------- #

def test_headers_are_matched_loosely_and_defaults_are_announced(tmp_path):
    path = tmp_path / "loose.csv"
    path.write_text("Experiment,Metal,Acid Conc (M),Extractant Conc. (M)\n"
                    "E1,Nd,0.1,0.1\nE2,Eu,1.0,0.1\nE3,Dy,3.0,0.1\n")
    conditions = recommender.read_candidate_conditions(path)
    assert list(conditions["candidate_id"]) == ["E1", "E2", "E3"]
    assert set(conditions.columns) >= {"metal", "acid_concentration_M",
                                       "extractant_concentration_M"}
    cohort = recommender.load_cohort()
    built = recommender.build_candidate_frame(
        cohort, "CCCCCCCCP(=O)(CCCCCCCC)CCCCCCCC", conditions, cohort_ligand_row=None)
    assumed = " ".join(built.warnings)
    assert "assumed hno3" in assumed and recommender.DEFAULT_DILUENT in assumed
    assert len(built.frame) == 3


def test_a_missing_required_column_is_a_clear_error(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("metal,acid_concentration_M\nNd,0.1\n")
    with pytest.raises(SystemExit) as excinfo:
        recommender.read_candidate_conditions(path)
    assert "extractant_concentration_M" in str(excinfo.value)


def test_an_unknown_metal_is_refused(tmp_path):
    path = tmp_path / "metal.csv"
    path.write_text("metal,acid_concentration_M,extractant_concentration_M\nFe,0.1,0.1\n")
    conditions = recommender.read_candidate_conditions(path)
    with pytest.raises(SystemExit) as excinfo:
        recommender.build_candidate_frame(recommender.load_cohort(), "CCOP(=O)(OCC)OCC",
                                          conditions, cohort_ligand_row=None)
    assert "Fe" in str(excinfo.value)


def test_the_dataset_fingerprint_is_reproduced_from_smiles():
    """The whole unseen-ligand path rests on this: ECFP built here == ECFP in the cohort."""
    cohort = pd.read_parquet(COHORT_CACHE,
                             columns=["extractant"] + [f"ecfp_{i}" for i in range(2048)])
    row = cohort.drop_duplicates("extractant").iloc[0]
    stored = row[[f"ecfp_{i}" for i in range(2048)]].to_numpy(dtype=float)
    assert np.array_equal(recommender.ecfp_bits(str(row["extractant"])), stored)


def test_rdkit_logging_is_not_disabled_globally(capfd):
    """A module-level ``RDLogger.DisableLog`` here would silence RDKit for the whole
    process — including :mod:`lanthanide_separation.gen8.mechanism`, whose own test
    asserts that RDKit still complains outside the featuriser.  Keep it scoped."""
    from rdkit import Chem

    with pytest.raises(SystemExit):
        recommender.ecfp_bits("C(((")
    capfd.readouterr()                      # quiet inside the recommender
    Chem.MolFromSmiles("C(((")
    assert "SMILES Parse Error" in capfd.readouterr().err


def test_the_json_documents_every_section_it_emits(stage_zero):
    """The output claims to be self-describing; hold it to that."""
    documented = set(stage_zero["section_documentation"])
    boilerplate = {"schema_version", "generated_at", "runtime_seconds",
                   "section_documentation", "field_documentation"}
    assert set(stage_zero) - boilerplate <= documented

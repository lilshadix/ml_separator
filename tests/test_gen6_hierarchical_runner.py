"""Tests for the Experiment C runner (``scripts/run_hierarchical_levels.py``).

Experiment C makes an *attribution* — "the level is the component that fails to
transfer" — and an attribution is only as good as the plumbing underneath it.
Four properties carry the whole conclusion, and each one would look perfectly
reasonable in the output if it silently stopped holding:

1. **MONO_ET is Experiment A's EXPANDED arm.**  If it is not, the "structure does
   not beat the monolith" comparison is against a different monolith than the one
   Phase 1 measured.  The assertion is fail-closed in a FULL run and must be
   *visibly* SKIPPED — never silently passed — in a pilot, where 120 trees cannot
   reproduce 400.
2. **The regime's own group column is the one held out**, in the outer folds and
   in the inner CV.  A chemotype hold-out whose inner CV grouped on something
   else would choose a ligand penalty that fits training beautifully and
   transfers nothing.
3. **The OOF layout scores all eight models on identical rows**, with the cell
   metadata (``cell_n_metals``) that hypothesis C1's endpoint is cut on.
4. **The verdict logic scores the pre-registered condition and not a paraphrase
   of it** — in particular C1 on BCa (not percentile), on the multi-metal
   endpoint, and C3 as a *negative* hypothesis whose FAIL is the surprise.

A fifth property is tested because the module gets it wrong:
``hierarchical.pair_consistency`` returns a hard-coded ``antisymmetry_max`` of
0.0, so the runner recomputes it and the recomputation must be sensitive to a
corrupted pair prediction.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen6.chemistry import ChemistryMap
from lanthanide_separation.gen6.hierarchical import MODEL_NAMES, derived_pairs, pair_consistency
from lanthanide_separation.gen6.manifest import RunManifest
from lanthanide_separation.levels import LevelData, LevelForestParameters

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_hierarchical_levels.py"
SPEC = importlib.util.spec_from_file_location("run_hierarchical_levels", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
# ``scripts/`` is not a package, so the runner is loaded by path.  It must be put
# in ``sys.modules`` *before* execution: the module defines dataclasses, and
# ``@dataclass`` resolves annotations through ``sys.modules[cls.__module__]``.
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


# --------------------------------------------------------------------------- #
# A synthetic cohort in the shape ``build_level_dataset`` produces
# --------------------------------------------------------------------------- #

def _synthetic(n_ligands: int = 12, n_conditions: int = 4, metals=(57, 60, 63, 66),
               seed: int = 0) -> LevelData:
    """A level frame whose target really is ``alpha_l + F_cond + F_metal + noise``.

    Two properties are deliberate and load-bearing:

    * the identity columns are **not** collinear — ECFP cluster, Tanimoto
      chemotype and series are three different partitions of the ligands, so the
      three regimes are three genuinely different hold-outs and a runner that
      used the wrong column for a regime would be caught;
    * the metal coverage is **ragged** — cells hold 1, 2, 3 or 4 metals, as on the
      real cohort where only 521 of 2,405 cells hold two or more.  A frame where
      every cell held every metal would make the ``multi_metal_cells`` endpoint
      vacuous *and* would make the pair-label-mean null exactly transitive by
      accident (every label would average over the same cells), which would in
      turn make the C4 non-vacuity check prove nothing.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_ligands):
        alpha = float(rng.normal(scale=1.5))
        d1, d2 = float(rng.normal()), float(rng.normal())
        for c in range(n_conditions):
            log_l = float(rng.uniform(-2, 0))
            log_h = float(rng.uniform(-1, 1))
            for z in metals[: 1 + ((i + c) % len(metals))]:
                radius = 1.2 - 0.01 * (z - 57)
                rows.append({
                    "row_id": f"L{i}-c{c}-{z}", "extractant": f"L{i}",
                    # three different partitions of the same ligands
                    "ecfp_cluster": f"e{i}", "tanimoto_cluster": f"t{i % 3}",
                    "series_id": f"s{i}-{c % 2}",
                    "condition_id": f"L{i}-c{c}", "metal_symbol": f"M{z}", "metal_Z": float(z),
                    "n_replicates": 1,
                    "log_D": alpha + 2.0 * log_l + 0.5 * log_h + 0.8 * radius * d1
                             + float(rng.normal(scale=0.1)),
                    "Atomic Number_metal": float(z), "lanthanide_index": float(z - 57),
                    "Ionic Radius_metal": radius,
                    "cond__acid_concentration_M": 10 ** log_h,
                    "cond__extractant_concentration_M": 10 ** log_l,
                    "cond__diluent__kerosene": 1.0,
                    "massact__log10_cond__acid_concentration_M": log_h,
                    "massact__log10_cond__extractant_concentration_M": log_l,
                    "massact__logL_x_DENTATE": 3 * log_l,
                    "MolWt": 300 + 20 * d1, "MolLogP": 4 + d2,
                    "donor__O(amide_carbonyl)": 2.0 + (i % 2), "donor__O(ether)": 1.0,
                    "DENTATE": 3.0, "coreCN": 9.0, "lig2d__a": d1, "lig2d__b": d2,
                })
    frame = pd.DataFrame(rows)
    blocks = {
        "METAL": ("Atomic Number_metal", "lanthanide_index", "Ionic Radius_metal"),
        "COND": ("cond__acid_concentration_M", "cond__extractant_concentration_M",
                 "cond__diluent__kerosene"),
        "PHYSCHEM": ("MolWt", "MolLogP"),
        "DONORS": ("donor__O(amide_carbonyl)", "donor__O(ether)", "DENTATE", "coreCN"),
        "MASSACTION": ("massact__log10_cond__acid_concentration_M",
                       "massact__log10_cond__extractant_concentration_M",
                       "massact__logL_x_DENTATE"),
        "LIG2D_EXT": ("lig2d__a", "lig2d__b"),
    }
    return LevelData(frame=frame, blocks=blocks, audit={"rows": len(frame)})


def _chemistry(frame: pd.DataFrame) -> ChemistryMap:
    """``similarity(i, j) = min(level_i, level_j)``, so the nearest neighbour of a
    test ligand to any reference cohort is controllable and the hard-chemistry
    bins are non-empty."""
    extractants = tuple(sorted(frame["extractant"].unique()))
    values = np.linspace(0.2, 0.9, len(extractants)).astype(np.float32)
    similarity = np.minimum.outer(values, values).astype(np.float32)
    np.fill_diagonal(similarity, 1.0)
    table = pd.DataFrame({
        "extractant": list(extractants),
        "chem__supercluster": [frame.loc[frame["extractant"] == e, "tanimoto_cluster"].iloc[0]
                               for e in extractants],
    })
    return ChemistryMap(table=table, similarity=similarity, extractants=extractants,
                        audit={"synthetic": True})


def _evaluate(data: LevelData, *, regime: str = "unseen_chemotype", seed: int = 104729,
              folds: int = 3) -> runner.RegimeSeedResult:
    return runner.evaluate_regime_seed(
        data, chemistry=_chemistry(data.frame), regime=regime, seed=seed, folds=folds,
        params=LevelForestParameters(n_estimators=10, random_state=42, n_jobs=1),
        base_min_cells=10, log=lambda message: None)


@pytest.fixture(scope="module")
def data() -> LevelData:
    return _synthetic()


@pytest.fixture(scope="module")
def chemotype_result(data: LevelData) -> runner.RegimeSeedResult:
    return _evaluate(data)


# --------------------------------------------------------------------------- #
# (a) the regime's own group column, in the outer folds
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("regime,column", [
    ("unseen_chemotype", "tanimoto_cluster"),
    ("unseen_ligand", "ecfp_cluster"),
    ("unseen_series", "series_id"),
])
def test_each_regime_holds_out_its_own_column(data: LevelData, regime: str, column: str):
    frame = data.frame
    splits = runner.regime_folds(frame, regime=regime, seed=104729, folds=3)
    assert splits, "no folds produced"
    covered: set[int] = set()
    for split in splits:
        values = frame[column].astype(str).to_numpy()
        assert not (set(values[split.train_index]) & set(values[split.test_index])), \
            f"{regime} leaked a {column} across the fold boundary"
        assert not (set(split.train_index.tolist()) & set(split.test_index.tolist()))
        covered |= set(split.test_index.tolist())
    assert covered == set(range(len(frame))), "the folds do not partition the cohort"
    assert runner.regime_integrity(frame, splits, regime=regime)["ok"] is True


def test_unseen_series_keeps_the_ligand_in_training_and_that_is_not_a_leak(data: LevelData):
    """The regime whose whole point is that C1's random intercept is live.

    A leakage checker that flagged ``extractant`` here would flag the design, so
    the runner's per-regime leakage columns must exclude it — and the test proves
    the situation actually arises rather than trusting the table.
    """
    frame = data.frame
    splits = runner.regime_folds(frame, regime="unseen_series", seed=104729, folds=3)
    ligands = frame["extractant"].astype(str).to_numpy()
    shared = [len(set(ligands[s.train_index]) & set(ligands[s.test_index])) for s in splits]
    assert min(shared) > 0, "no fold kept a test ligand in training; the regime is vacuous here"
    assert runner.regime_integrity(frame, splits, regime="unseen_series")["ok"] is True
    assert "extractant" not in runner.REGIME_LEAKAGE_COLUMNS["unseen_series"]


def test_chemotype_folds_are_experiment_as_folds(data: LevelData):
    """The chemotype regime must go through ``diversity_splits`` and take EXPANDED.

    That is what makes the MONO_ET reproduction check possible at all: a
    different partition would give different training rows and the identity could
    never hold, no matter how faithful the learner settings were.
    """
    from lanthanide_separation.gen6.cohorts import EXPANDED_ARM, diversity_splits
    frame = data.frame
    ours = runner.regime_folds(frame, regime="unseen_chemotype", seed=104729, folds=3)
    theirs = diversity_splits(frame, group_column="tanimoto_cluster", n_splits=3, seed=104729)
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs):
        np.testing.assert_array_equal(a.test_index, b.test_index)
        np.testing.assert_array_equal(a.train_index, b.train_index_by_arm[EXPANDED_ARM])


def test_cells_are_never_split_across_folds(data: LevelData):
    """Every Experiment C quantity is defined per (ligand, condition) cell."""
    frame = data.frame
    for regime in runner.REGIMES:
        splits = runner.regime_folds(frame, regime=regime, seed=104729, folds=3)
        audit = runner.cells_never_split_across_folds(frame, splits)
        assert audit["ok"] is True, f"{regime}: {audit}"
        assert audit["n_cells"] == frame.groupby(["extractant", "condition_id"]).ngroups


# --------------------------------------------------------------------------- #
# (b) the OOF layout
# --------------------------------------------------------------------------- #

def test_oof_scores_all_eight_models_on_identical_rows(chemotype_result, data: LevelData,
                                                       tmp_path: Path):
    oof = chemotype_result.oof
    path = tmp_path / "oof_predictions.parquet"
    oof.to_parquet(path, index=False)
    reloaded = pd.read_parquet(path)

    assert len(reloaded) == len(data.frame)
    assert set(reloaded["row_id"]) == set(data.frame["row_id"])
    assert list(MODEL_NAMES) == ["MONO_ET", "MONO_RIDGE", "C1_HIER_RIDGE", "C2_TWO_STAGE",
                                 "C2_TRUECENTRE", "C3_SHARED_RIDGE", "ORACLE_LEVEL",
                                 "ORACLE_METAL"], "the eight models moved; the report must too"
    scored = {m: frozenset(reloaded.loc[reloaded[f"prediction_{m}"].notna(), "row_id"])
              for m in MODEL_NAMES}
    assert len(set(scored.values())) == 1, "models were scored on different rows"
    assert scored["MONO_ET"] == frozenset(data.frame["row_id"])
    for model in MODEL_NAMES:
        assert np.isfinite(reloaded[f"prediction_{model}"]).all()
    # the identity and cell columns hypothesis C1's endpoint is cut on
    for column in ("regime", "split_seed", "outer_fold", "nn_train_tanimoto", "cell_n_metals",
                   "cell_true_mean", "stage_a", "stage_b", "stage_b_truecentre"):
        assert column in reloaded.columns
    assert (reloaded["cell_n_metals"] >= 1).all()
    assert reloaded["nn_train_tanimoto"].notna().all()


def test_the_models_are_genuinely_different_not_a_copied_column(chemotype_result):
    oof = chemotype_result.oof
    columns = [f"prediction_{m}" for m in MODEL_NAMES]
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            assert not np.allclose(oof[left], oof[right]), f"{left} == {right}"


def test_oracles_isolate_exactly_one_component(chemotype_result):
    """The oracle swaps must be the identities the attribution rests on."""
    oof = chemotype_result.oof
    y = oof["log_D"].to_numpy(dtype=float)
    np.testing.assert_allclose(np.abs(oof["prediction_ORACLE_METAL"] - y),
                               np.abs(oof["stage_a"] - oof["cell_true_mean"]), atol=1e-9)
    np.testing.assert_allclose(np.abs(oof["prediction_ORACLE_LEVEL"] - y),
                               np.abs(oof["stage_b_truecentre"] - (y - oof["cell_true_mean"])),
                               atol=1e-9)


def test_unseen_chemotype_gives_a_held_out_ligand_no_intercept(data: LevelData):
    """C1's mechanism: for a new ligand the prediction is the descriptor prior."""
    result = _evaluate(data, regime="unseen_chemotype")
    attribution = result.attribution
    intercepts = attribution[attribution["component"] == "ligand_intercept"]
    assert not intercepts.empty
    assert (intercepts["mean_abs_contribution"] == 0.0).all(), \
        "a held-out ligand received a non-zero random intercept"


def test_unseen_series_keeps_the_intercept_live(data: LevelData):
    """…and in the regime where the ligand IS in training, the intercept is used."""
    result = _evaluate(data, regime="unseen_series")
    intercepts = result.attribution[result.attribution["component"] == "ligand_intercept"]
    assert not intercepts.empty
    assert intercepts["mean_abs_contribution"].max() > 0.0, \
        "the ligand intercept was zero even for a ligand that is in training"


def test_penalties_are_recorded_per_regime_seed_fold(chemotype_result):
    penalties = chemotype_result.penalties
    assert set(penalties["model"]) == {"MONO_RIDGE", "C1_HIER_RIDGE", "C3_SHARED_RIDGE"}
    assert penalties[penalties["model"] == "MONO_RIDGE"]["lambda_ligand"].isna().all()
    # C3 reuses C1's chosen penalties by construction
    for fold, block in penalties.groupby("fold"):
        c1 = block[block["model"] == "C1_HIER_RIDGE"].iloc[0]
        c3 = block[block["model"] == "C3_SHARED_RIDGE"].iloc[0]
        assert c1["lambda_fixed"] == c3["lambda_fixed"]
        assert c1["lambda_ligand"] == c3["lambda_ligand"]
    table = runner.lambda_frequency_table(penalties)
    assert set(table["regime"]) == {"unseen_chemotype"}
    assert table["n_folds"].sum() == penalties["fold"].nunique()
    assert table["share_of_folds"].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# (c) the MONO_ET = Experiment A assertion
# --------------------------------------------------------------------------- #

def _reference(oof: pd.DataFrame, *, jitter: float = 0.0,
               feature_set: str = runner.REFERENCE_FEATURE_SET) -> pd.DataFrame:
    """An Experiment A style OOF frame that agrees with ours up to ``jitter``."""
    block = oof[oof["regime"] == "unseen_chemotype"]
    return pd.DataFrame({
        "feature_set": feature_set,
        "split_seed": block["split_seed"].to_numpy(),
        "row_id": block["row_id"].to_numpy(),
        f"prediction_{runner.REFERENCE_ARM}": block["prediction_MONO_ET"].to_numpy() + jitter,
    })


def test_reproduction_check_passes_on_an_identical_reference(chemotype_result, tmp_path: Path):
    path = tmp_path / "reference.parquet"
    _reference(chemotype_result.oof).to_parquet(path, index=False)
    report = runner.assert_mono_et_matches_experiment_a(
        chemotype_result.oof, reference_path=path, pilot=False)
    assert report["ok"] is True
    assert report["status"] == "CHECKED"
    assert report["max_abs_diff"] == 0.0
    assert report["n_rows_compared"] == len(chemotype_result.oof)


def test_reproduction_check_fails_on_a_tiny_difference(chemotype_result, tmp_path: Path):
    """1e-8 is invisible in any report table and must still stop the run."""
    path = tmp_path / "reference.parquet"
    _reference(chemotype_result.oof, jitter=1e-8).to_parquet(path, index=False)
    report = runner.assert_mono_et_matches_experiment_a(
        chemotype_result.oof, reference_path=path, pilot=False)
    assert report["ok"] is False
    assert report["status"] == "MISMATCH"
    assert report["max_abs_diff"] == pytest.approx(1e-8)


def test_reproduction_check_fails_closed_on_a_missing_reference(chemotype_result, tmp_path: Path):
    report = runner.assert_mono_et_matches_experiment_a(
        chemotype_result.oof, reference_path=tmp_path / "absent.parquet", pilot=False)
    assert report["ok"] is False
    assert report["status"] == "MISSING_REFERENCE"


def test_reproduction_check_fails_closed_on_the_wrong_feature_set(chemotype_result,
                                                                  tmp_path: Path):
    path = tmp_path / "reference.parquet"
    _reference(chemotype_result.oof, feature_set="MC_donors").to_parquet(path, index=False)
    report = runner.assert_mono_et_matches_experiment_a(
        chemotype_result.oof, reference_path=path, pilot=False)
    assert report["ok"] is False
    assert report["status"] == "MISSING_FEATURE_SET"


def test_reproduction_check_fails_closed_when_a_seed_is_absent(chemotype_result, tmp_path: Path):
    path = tmp_path / "reference.parquet"
    reference = _reference(chemotype_result.oof)
    reference["split_seed"] = 999
    reference.to_parquet(path, index=False)
    report = runner.assert_mono_et_matches_experiment_a(
        chemotype_result.oof, reference_path=path, pilot=False)
    assert report["ok"] is False


def test_reproduction_check_is_skipped_not_passed_in_a_pilot(chemotype_result, tmp_path: Path):
    """A pilot fits 120 trees; recording that as a pass would be false assurance."""
    path = tmp_path / "reference.parquet"
    _reference(chemotype_result.oof, jitter=5.0).to_parquet(path, index=False)
    report = runner.assert_mono_et_matches_experiment_a(
        chemotype_result.oof, reference_path=path, pilot=True)
    assert report["status"] == "SKIPPED"
    assert report["n_rows_compared"] == 0
    assert "SKIPPED" in report["reason"]
    # and the check lands under a key that SAYS it was skipped, so no reader of
    # validation.json can mistake it for a passed reproduction
    checks = runner.build_checks(
        results=[chemotype_result], oof=chemotype_result.oof, reproduction=report,
        pair_audit={"ok": True}, cohort_extractants=set(chemotype_result.oof["extractant"]),
        chemistry_extractants=set(chemotype_result.oof["extractant"]), pilot=True)
    assert "mono_et_reproduction_SKIPPED_in_pilot" in checks
    assert "mono_et_reproduces_experiment_a" not in checks

    # …and a real failure always lands under the plain name, where it fails the run
    failed = runner.build_checks(
        results=[chemotype_result], oof=chemotype_result.oof,
        reproduction={"ok": False, "status": "MISMATCH", "max_abs_diff": 1e-8},
        pair_audit={"ok": True}, cohort_extractants=set(chemotype_result.oof["extractant"]),
        chemistry_extractants=set(chemotype_result.oof["extractant"]), pilot=True)
    assert failed["mono_et_reproduces_experiment_a"]["ok"] is False


def test_reproduction_check_is_not_applicable_without_the_chemotype_regime(chemotype_result,
                                                                          tmp_path: Path):
    """A `--regimes unseen_series` run says NOT_APPLICABLE, in a pilot or not."""
    other = chemotype_result.oof.assign(regime="unseen_series")
    for pilot in (False, True):
        report = runner.assert_mono_et_matches_experiment_a(
            other, reference_path=tmp_path / "absent.parquet", pilot=pilot)
        assert report["status"] == "NOT_APPLICABLE"
        assert "no reproduction evidence" in report["reason"]
    checks = runner.build_checks(
        results=[chemotype_result], oof=other, reproduction=report, pair_audit={"ok": True},
        cohort_extractants=set(other["extractant"]),
        chemistry_extractants=set(other["extractant"]), pilot=True)
    assert "mono_et_reproduction_NOT_APPLICABLE_regime_not_run" in checks


# --------------------------------------------------------------------------- #
# (d) the pair identities — including the one the module does not measure
# --------------------------------------------------------------------------- #

def test_pair_audit_is_exact_and_not_vacuous(chemotype_result):
    table, audit = runner.pair_tables(chemotype_result.pairs, chemotype_result.oof)
    assert audit["ok"] is True
    assert audit["max_antisymmetry"] < 1e-9
    assert audit["max_transitivity"] < 1e-9
    assert audit["n_triples"] > 0, "an empty pair table satisfies the bound vacuously"
    assert set(table["model"]) == set(MODEL_NAMES) | {"NULL"}
    # the NULL is a pair-label mean, NOT a difference of two level predictions,
    # so it must NOT satisfy transitivity — otherwise the test above proves nothing
    null_rows = table[table["model"] == "NULL"]
    assert null_rows["transitivity_max"].max() > 1e-9


def test_module_pair_consistency_measures_antisymmetry_after_the_fix(chemotype_result):
    """The module defect this test used to pin (a hard-coded ``antisymmetry_max = 0.0``)
    was fixed: the residual is now rebuilt from the two row predictions when the
    pair table carries them, and is NaN — never a silent zero — when it cannot be.
    A pair column that is not a difference of row predictions must show a residual."""
    pairs = chemotype_result.pairs.copy()
    assert pair_consistency(pairs, "pair_prediction_MONO_ET")["antisymmetry_max"] == pytest.approx(0.0, abs=1e-12)
    pairs["pair_broken"] = np.arange(len(pairs), dtype=float)  # not antisymmetric in any sense
    residual = pair_consistency(pairs, "pair_broken")["antisymmetry_max"]
    assert np.isnan(residual)                                  # no row predictions for 'broken' -> NaN, not 0
    assert pair_consistency(pairs, "pair_broken")["transitivity_max"] > 1e-9

def test_runner_antisymmetry_measurement_is_sensitive_to_corruption(chemotype_result):
    """The runner's own antisymmetry number must move when the pairs are wrong."""
    oof = chemotype_result.oof
    fold = int(oof["outer_fold"].iloc[0])
    fold_rows = oof[oof["outer_fold"] == fold]
    fold_pairs = chemotype_result.pairs[chemotype_result.pairs["outer_fold"] == fold]
    assert not fold_pairs.empty
    clean = runner.antisymmetry_residual(fold_pairs, fold_rows=fold_rows,
                                         prediction_column="prediction_MONO_ET")
    assert clean == 0.0
    corrupted = fold_pairs.copy()
    corrupted.loc[corrupted.index[0], "pair_prediction_MONO_ET"] += 0.25
    assert runner.antisymmetry_residual(corrupted, fold_rows=fold_rows,
                                        prediction_column="prediction_MONO_ET") == \
        pytest.approx(0.25)


def test_antisymmetry_measurement_refuses_a_misaligned_pair_table(chemotype_result):
    """The positional join is load-bearing, so a misalignment must raise, not lie."""
    oof = chemotype_result.oof
    fold = int(oof["outer_fold"].iloc[0])
    fold_rows = oof[oof["outer_fold"] == fold]
    fold_pairs = chemotype_result.pairs[chemotype_result.pairs["outer_fold"] == fold].copy()
    fold_pairs["truth"] = fold_pairs["truth"] + 1.0
    with pytest.raises(RuntimeError, match="do not align"):
        runner.antisymmetry_residual(fold_pairs, fold_rows=fold_rows,
                                     prediction_column="prediction_MONO_ET")


def test_derived_pairs_cover_every_model(chemotype_result):
    pairs = chemotype_result.pairs
    for model in MODEL_NAMES:
        assert f"pair_prediction_{model}" in pairs.columns
    assert "pair_NULL" in pairs.columns
    assert (pairs["delta_z"] > 0).all(), "A must always be the lighter metal"


# --------------------------------------------------------------------------- #
# (e) the endpoint hypothesis C1 is scored on
# --------------------------------------------------------------------------- #

def test_multi_metal_endpoint_is_a_property_of_the_cell_not_the_model(chemotype_result):
    oof = chemotype_result.oof
    endpoints = dict(runner.endpoint_subsets(oof))
    assert list(endpoints) == ["all", "multi_metal_cells"]
    assert len(endpoints["all"]) == len(oof)
    assert set(endpoints["multi_metal_cells"]["row_id"]) == \
        set(oof.loc[oof["cell_n_metals"] >= 2, "row_id"])
    counts = (endpoints["multi_metal_cells"].groupby(["extractant", "condition_id"]).size())
    assert (counts >= 2).all()
    # non-vacuity: singleton cells must actually exist, or the restriction that
    # hypothesis C1 is pre-registered on would be a no-op here
    assert 0 < len(endpoints["multi_metal_cells"]) < len(oof)
    assert (oof["cell_n_metals"] == 1).any()


# --------------------------------------------------------------------------- #
# (f) verdict logic
# --------------------------------------------------------------------------- #

def _summary_row(regime: str, endpoint: str, comparison: str, statistic: str, *, delta: float,
                 bca: tuple[float, float], percentile: tuple[float, float] | None = None,
                 seeds_positive: int = 5, n_seeds: int = 5) -> dict:
    low, high = percentile if percentile is not None else bca
    return {"regime": regime, "endpoint": endpoint, "comparison": comparison,
            "statistic": statistic, "pooled_point_delta": delta,
            "pooled_ci95_low": low, "pooled_ci95_high": high,
            "pooled_bca_low": bca[0], "pooled_bca_high": bca[1],
            "pooled_units_total": 40, "pooled_blocks": 20, "mean_point_delta": delta,
            "seeds_positive": seeds_positive, "n_seeds": n_seeds}


def _summary(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


PAIRS_OK = {"ok": True, "max_antisymmetry": 0.0, "max_transitivity": 1e-17, "n_triples": 100,
            "n_pairs": 50}


def test_c1_passes_only_when_the_decisive_bca_low_is_above_zero():
    base = _summary_row("unseen_chemotype", "multi_metal_cells", "level_minus_metal_gain", "mae",
                        delta=0.5, bca=(0.1, 0.9))
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary([base]), pair_audit=PAIRS_OK, regimes=runner.REGIMES, n_seeds=5)}
    assert verdicts["C1"].verdict == "PASS"

    # same point estimate, BCa straddles zero -> INCONCLUSIVE, never a weak PASS
    straddle = dict(base, pooled_bca_low=-0.1, pooled_bca_high=0.9)
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary([straddle]), pair_audit=PAIRS_OK, regimes=runner.REGIMES, n_seeds=5)}
    assert verdicts["C1"].verdict == "INCONCLUSIVE"

    # the metal oracle removes MORE error -> evidence against, i.e. FAIL
    against = dict(base, pooled_point_delta=-0.5, pooled_bca_low=-0.9, pooled_bca_high=-0.1)
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary([against]), pair_audit=PAIRS_OK, regimes=runner.REGIMES, n_seeds=5)}
    assert verdicts["C1"].verdict == "FAIL"


def test_c1_reads_the_multi_metal_endpoint_and_not_all_rows():
    """A pass on all rows must not leak into the verdict; C1 is pre-registered on cells."""
    rows = [
        _summary_row("unseen_chemotype", "all", "level_minus_metal_gain", "mae",
                     delta=0.9, bca=(0.5, 1.2)),
        _summary_row("unseen_chemotype", "multi_metal_cells", "level_minus_metal_gain", "mae",
                     delta=0.02, bca=(-0.30, 0.40)),
    ]
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary(rows), pair_audit=PAIRS_OK, regimes=runner.REGIMES, n_seeds=5)}
    assert verdicts["C1"].verdict == "INCONCLUSIVE"
    assert "multi-metal test cells" in verdicts["C1"].evidence


def test_c1_uses_bca_not_the_percentile_interval():
    """The percentile interval's Type-I rate was measured at ~12.7 % on this cohort."""
    row = _summary_row("unseen_chemotype", "multi_metal_cells", "level_minus_metal_gain", "mae",
                       delta=0.4, bca=(-0.05, 0.9), percentile=(0.05, 0.8))
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary([row]), pair_audit=PAIRS_OK, regimes=runner.REGIMES, n_seeds=5)}
    assert verdicts["C1"].verdict == "INCONCLUSIVE", \
        "the percentile interval excluded zero but BCa did not; BCa decides"


def test_a_single_seed_pilot_cannot_deliver_a_statistical_pass():
    row = _summary_row("unseen_chemotype", "multi_metal_cells", "level_minus_metal_gain", "mae",
                       delta=0.5, bca=(0.1, 0.9), seeds_positive=1, n_seeds=1)
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary([row]), pair_audit=PAIRS_OK, regimes=["unseen_chemotype"], n_seeds=1)}
    assert verdicts["C1"].verdict == "INCONCLUSIVE"
    # …but C4 is an algebraic identity, so it is decided exactly even in a pilot
    assert verdicts["C4"].verdict == "PASS"


def test_c2_needs_both_clauses():
    helps = _summary_row("unseen_series", "all", "C1_HIER_RIDGE_vs_MONO_RIDGE", "mae",
                         delta=0.2, bca=(0.05, 0.4))
    quiet = _summary_row("unseen_chemotype", "all", "C1_HIER_RIDGE_vs_MONO_RIDGE", "mae",
                         delta=0.0, bca=(-0.1, 0.1))
    loud = _summary_row("unseen_chemotype", "all", "C1_HIER_RIDGE_vs_MONO_RIDGE", "mae",
                        delta=0.3, bca=(0.2, 0.4), percentile=(0.2, 0.4))
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary([helps, quiet]), pair_audit=PAIRS_OK, regimes=runner.REGIMES, n_seeds=5)}
    assert verdicts["C2"].verdict == "PASS"
    # helps on a NEW chemotype too -> the mixed-effects reading is wrong
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary([helps, loud]), pair_audit=PAIRS_OK, regimes=runner.REGIMES, n_seeds=5)}
    assert verdicts["C2"].verdict == "INCONCLUSIVE"
    # and with unseen_series missing it is simply not evaluable
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary([quiet]), pair_audit=PAIRS_OK, regimes=["unseen_chemotype"], n_seeds=5)}
    assert verdicts["C2"].verdict == "INCONCLUSIVE"
    assert "NOT EVALUABLE" in verdicts["C2"].evidence


def test_c3_is_a_negative_hypothesis_whose_fail_is_the_surprise():
    quiet = _summary_row("unseen_chemotype", "all", "C2_TWO_STAGE_vs_MONO_ET", "mae",
                         delta=0.005, bca=(-0.02, 0.03))
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary([quiet]), pair_audit=PAIRS_OK, regimes=runner.REGIMES, n_seeds=5)}
    assert verdicts["C3"].verdict == "PASS"

    # a real beat, bigger than the pre-registered 0.02 margin, with BCa low > 0
    beats = _summary_row("unseen_chemotype", "all", "C2_TWO_STAGE_vs_MONO_ET", "mae",
                         delta=0.08, bca=(0.03, 0.13))
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary([beats]), pair_audit=PAIRS_OK, regimes=runner.REGIMES, n_seeds=5)}
    assert verdicts["C3"].verdict == "FAIL"

    # a beat SMALLER than the margin is not a falsification
    small = _summary_row("unseen_chemotype", "all", "C2_TWO_STAGE_vs_MONO_ET", "mae",
                         delta=0.01, bca=(0.002, 0.02))
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary([small]), pair_audit=PAIRS_OK, regimes=runner.REGIMES, n_seeds=5)}
    assert verdicts["C3"].verdict == "PASS"


def test_c3_labels_the_truecentre_variant_exploratory_and_excludes_it_from_the_verdict():
    quiet = _summary_row("unseen_chemotype", "all", "C2_TWO_STAGE_vs_MONO_ET", "mae",
                         delta=0.0, bca=(-0.02, 0.02))
    truecentre_beats = _summary_row("unseen_chemotype", "all", "C2_TRUECENTRE_vs_MONO_ET", "mae",
                                    delta=0.30, bca=(0.20, 0.40))
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary([quiet, truecentre_beats]), pair_audit=PAIRS_OK, regimes=runner.REGIMES,
        n_seeds=5)}
    assert verdicts["C3"].verdict == "PASS", "the exploratory variant must not decide C3"
    assert "EXPLORATORY" in verdicts["C3"].evidence
    registry = {c.name: c for c in runner.CONTRASTS}
    assert registry["C2_TRUECENTRE_vs_MONO_ET"].preregistered is False


def test_c4_fails_on_a_broken_identity_and_on_an_empty_pair_table():
    broken = {"ok": False, "max_antisymmetry": 1e-6, "max_transitivity": 0.0, "n_triples": 10,
              "n_pairs": 5}
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary([]), pair_audit=broken, regimes=runner.REGIMES, n_seeds=5)}
    assert verdicts["C4"].verdict == "FAIL"
    vacuous = {"ok": False, "max_antisymmetry": 0.0, "max_transitivity": 0.0, "n_triples": 0,
               "n_pairs": 0}
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _summary([]), pair_audit=vacuous, regimes=runner.REGIMES, n_seeds=5)}
    assert verdicts["C4"].verdict == "FAIL", "zero triples must not satisfy the bound vacuously"


def test_every_hypothesis_carries_a_falsifier():
    verdicts = runner.score_hypotheses(_summary([]), pair_audit=PAIRS_OK, regimes=runner.REGIMES,
                                       n_seeds=5)
    assert [v.hypothesis for v in verdicts] == ["C1", "C2", "C3", "C4"]
    for v in verdicts:
        assert v.falsifier.strip() and v.pass_condition.strip()
        assert v.verdict in {"PASS", "FAIL", "INCONCLUSIVE"}


def test_the_decisive_contrast_is_the_difference_of_the_two_oracle_gains():
    """Arithmetic, not opinion: (C2 − ORACLE_LEVEL) − (C2 − ORACLE_METAL) = ORACLE_METAL − ORACLE_LEVEL."""
    registry = {c.name: c for c in runner.CONTRASTS}
    level = registry["level_oracle_gain"]
    metal = registry["metal_oracle_gain"]
    decisive = registry[runner.DECISIVE_CONTRAST]
    assert level.reference == metal.reference == "C2_TWO_STAGE"
    assert (decisive.reference, decisive.candidate) == (metal.candidate, level.candidate)


# --------------------------------------------------------------------------- #
# (g) fail-closed completion
# --------------------------------------------------------------------------- #

def _stub_manifest(output_dir: Path) -> RunManifest:
    manifest = RunManifest(layer="gen6_test", run_id="unit-test")
    manifest.record_many({
        "dataset_path": "synthetic", "dataset_file_sha256": "0" * 64,
        "source_table_sha256": "1" * 64, "feature_registry_sha256": "2" * 64,
        "code_sha256": {"synthetic.py": "3" * 64},
        "chemistry_cluster_definition": {"identity": "canonical_smiles"},
        "provenance_state": {"status": "not_audited_in_this_run"},
        "model_seed": 42, "split_seeds": [104729], "preprocessing": [{"step": "none"}],
    })
    manifest.record_split(definition={"group_column_by_regime": {"unseen_chemotype":
                                                                 "tanimoto_cluster"}}, folds=[{
        "fold": 0, "split_seed": 104729, "test_row_ids_sha256": "4" * 64,
        "test_extractants": ["L0"], "test_superclusters": ["t0"],
        "train_extractants_by_arm": {"ALL_MODELS": ["L1"]},
        "train_superclusters_by_arm": {"ALL_MODELS": ["t1"]},
        "n_test_rows": 12, "n_train_rows_by_arm": {"ALL_MODELS": 24},
    }])
    for name in runner.REQUIRED_ARTIFACTS:
        (output_dir / name).write_text("placeholder\n")
    return manifest


def test_no_success_marker_when_the_reproduction_check_fails(tmp_path: Path):
    manifest = _stub_manifest(tmp_path)
    validation, success = runner.finalise_run(
        tmp_path, manifest=manifest,
        checks={"mono_et_reproduces_experiment_a": {"ok": False, "status": "MISMATCH"},
                "regime_split_integrity": {"ok": True}})
    assert success is None
    assert not (tmp_path / "_SUCCESS.json").exists()
    assert (tmp_path / "_FAILED.json").exists()
    assert "mono_et_reproduces_experiment_a" in validation["failed_checks"]


def test_no_success_marker_when_the_pair_identities_break(tmp_path: Path):
    manifest = _stub_manifest(tmp_path)
    validation, success = runner.finalise_run(
        tmp_path, manifest=manifest,
        checks={"pair_identities_exact": {"ok": False, "max_antisymmetry": 1e-6},
                "regime_split_integrity": {"ok": True}})
    assert success is None
    assert "pair_identities_exact" in validation["failed_checks"]


def test_success_marker_when_every_check_passes(tmp_path: Path):
    manifest = _stub_manifest(tmp_path)
    validation, success = runner.finalise_run(
        tmp_path, manifest=manifest,
        checks={"regime_split_integrity": {"ok": True}, "pair_identities_exact": {"ok": True}})
    assert validation["ok"] is True
    assert success is not None and success.exists()
    payload = json.loads((tmp_path / "_SUCCESS.json").read_text())
    assert payload["validation_ok"] is True
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text())
    assert "validation.json" in hashes["files"], \
        "artifact hashes must cover the validation record written after the manifest"


def test_missing_artifact_also_blocks_success(tmp_path: Path):
    manifest = _stub_manifest(tmp_path)
    (tmp_path / "decision_report.md").unlink()
    validation, success = runner.finalise_run(tmp_path, manifest=manifest,
                                              checks={"regime_split_integrity": {"ok": True}})
    assert success is None
    assert "decision_report.md" in validation["missing_artifacts"]


def test_build_checks_passes_on_a_clean_run(chemotype_result):
    table, audit = runner.pair_tables(chemotype_result.pairs, chemotype_result.oof)
    checks = runner.build_checks(
        results=[chemotype_result], oof=chemotype_result.oof,
        reproduction={"ok": True, "status": "CHECKED"}, pair_audit=audit,
        cohort_extractants=set(chemotype_result.oof["extractant"]),
        chemistry_extractants=set(chemotype_result.oof["extractant"]), pilot=False)
    assert all(entry["ok"] for entry in checks.values()), \
        {k: v.get("ok") for k, v in checks.items()}
    assert "mono_et_reproduces_experiment_a" in checks
    assert checks["crossfit_provenance"]["ok"] is True


def test_build_checks_flags_a_missing_similarity_value(chemotype_result):
    damaged = chemotype_result.oof.copy()
    damaged.loc[damaged.index[0], "nn_train_tanimoto"] = np.nan
    checks = runner.build_checks(
        results=[chemotype_result], oof=damaged, reproduction={"ok": True},
        pair_audit={"ok": True},
        cohort_extractants=set(damaged["extractant"]),
        chemistry_extractants=set(damaged["extractant"]), pilot=False)
    assert checks["similarity_column_complete"]["ok"] is False
    assert checks["similarity_column_complete"]["n_missing"] == 1


# --------------------------------------------------------------------------- #
# (h) the CLI contract
# --------------------------------------------------------------------------- #

def test_defaults_are_the_pre_registered_design():
    args = runner.parse_args([])
    assert list(args.regimes) == ["unseen_chemotype", "unseen_ligand", "unseen_series"]
    assert list(args.split_seeds) == [104729, 130363, 155921, 196613, 262147]
    assert args.folds == 5 and args.n_estimators == 400 and args.model_seed == 42
    assert args.pilot is False
    assert Path(args.reference_oof).name == "oof_predictions.parquet"
    assert args.reference_feature_set == "MC_lig2d_ext_massaction"


def test_regimes_are_validated_by_argparse():
    with pytest.raises(SystemExit):
        runner.parse_args(["--regimes", "unseen_planet"])


def test_required_artifacts_cover_the_contract():
    for name in ("oof_predictions.parquet", "per_ligand_metrics.csv", "arm_metrics.csv",
                 "hard_chemistry_metrics.csv", "contrasts.csv", "contrast_summary.csv",
                 "component_attribution.csv", "penalties.csv", "pair_metrics.csv",
                 "pair_predictions.parquet", "decision_report.md", "summary.json"):
        assert name in runner.REQUIRED_ARTIFACTS

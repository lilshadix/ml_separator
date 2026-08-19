"""Tests for the Experiment A runner (``scripts/run_diversity_causal.py``).

Experiment A is a causal claim built on two literal properties, and if either
one quietly stops holding the whole comparison is void rather than merely noisy:

1. **The test rows do not move between arms.**  BASE and EXPANDED must be scored
   on byte-identical rows; only the training mask may differ.
2. **The hard-chemistry bins do not move between arms.**  "Hard chemistry" is cut
   on the nearest-neighbour similarity to the **BASE** training set of the fold,
   for every arm.  Cutting it per arm would compare the arms on different rows —
   and would look perfectly reasonable in the output.

Both are tested here on a synthetic cohort whose similarity structure is chosen
so the check cannot pass vacuously: the nearest-neighbour distance to the
EXPANDED training set is provably different from the distance to BASE, so a
runner that used the arm's own training set would produce different bins and
these tests would fail.

The other two tested properties are the information null (the shuffled arm's
targets must be a permutation of the honest ones, restricted to the added sparse
rows) and the fail-closed wiring (no ``_SUCCESS.json`` when a check fails).
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
from lanthanide_separation.gen6.cohorts import (
    BASE_ARM, DEFAULT_ARMS, EXPANDED_ARM, ROWMATCHED_ARM, SHUFFLED_ARM, arm_membership,
    diversity_splits,
)
from lanthanide_separation.gen6.manifest import RunManifest
from lanthanide_separation.gen6.metrics import gen6_metric_table, per_unit_statistics
from lanthanide_separation.levels import LevelData, LevelForestParameters, LevelRegressor

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_diversity_causal.py"
SPEC = importlib.util.spec_from_file_location("run_diversity_causal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
# ``scripts/`` is not a package, so the runner is loaded by path.  It must be put
# in ``sys.modules`` *before* execution: the module defines dataclasses, and
# ``@dataclass`` resolves annotations through ``sys.modules[cls.__module__]``.
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

FEATURE_COLUMNS = ("feat__ligand", "feat__metal", "feat__condition")
METALS = ("Nd", "Eu", "Yb")


# --------------------------------------------------------------------------- #
# Synthetic cohort and chemistry
# --------------------------------------------------------------------------- #

def _cohort(n_dense: int = 8, n_sparse: int = 6, dense_cells: int = 12, sparse_cells: int = 4,
            n_superclusters: int = 5, seed: int = 11) -> pd.DataFrame:
    """A shared cohort in the shape ``build_level_dataset`` produces.

    Dense extractants clear ``min_cells = 10``; sparse ones do not, so
    ``arm_membership`` splits them exactly as it does on the real cohort.  Each
    extractant owns one ECFP cluster and those nest inside super-clusters, which
    is what makes a super-cluster hold-out leakage-free.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_dense + n_sparse):
        dense = i < n_dense
        name = f"{'D' if dense else 'S'}{i:02d}"
        ligand_effect = float(rng.normal(scale=1.5))
        for j in range(dense_cells if dense else sparse_cells):
            metal = METALS[j % len(METALS)]
            metal_effect = float(METALS.index(metal))
            condition = float(j % 4)
            rows.append({
                "row_id": f"{name}-{j:03d}",
                "extractant": name,
                "ecfp_cluster": f"ec{i:02d}",
                "tanimoto_cluster": f"sc{i % n_superclusters:02d}",
                "condition_id": f"cond{j % 4:02d}",
                "series_id": f"ser{i:02d}",
                "metal_symbol": metal,
                "metal_Z": 60.0 + metal_effect,
                "n_replicates": 1,
                "log_D": 0.8 * ligand_effect + 0.4 * metal_effect + 0.2 * condition
                         + float(rng.normal(scale=0.2)),
                "feat__ligand": ligand_effect,
                "feat__metal": metal_effect,
                "feat__condition": condition,
            })
    return pd.DataFrame(rows)


def _chemistry(frame: pd.DataFrame) -> ChemistryMap:
    """A similarity matrix with a deliberately *controllable* nearest neighbour.

    ``similarity(i, j) = min(level_i, level_j)``, so the maximum similarity of a
    test ligand to any reference cohort is ``min(level_i, max level in the
    reference)``.  Dense extractants are given levels 0.15…0.50 and sparse ones
    0.55…0.90, which has three consequences the tests rely on:

    * a test ligand's distance to the **BASE** training set is capped at ~0.50,
      so it lands in a bin that depends only on its own level;
    * some test ligands fall below 0.4 and some do not, so both hard-chemistry
      endpoints are non-empty;
    * the distance to the **EXPANDED** training set is strictly larger for the
      high-level ligands, so a runner that cut the bins on the arm's own training
      set would produce different bins — the arm-invariance test cannot pass by
      accident.
    """
    extractants = tuple(sorted(frame["extractant"].unique()))
    dense = [e for e in extractants if e.startswith("D")]
    sparse = [e for e in extractants if e.startswith("S")]
    level = {}
    for name, value in zip(dense, np.linspace(0.15, 0.50, len(dense))):
        level[name] = float(value)
    for name, value in zip(sparse, np.linspace(0.55, 0.90, len(sparse))):
        level[name] = float(value)
    values = np.array([level[e] for e in extractants], dtype=np.float32)
    similarity = np.minimum.outer(values, values).astype(np.float32)
    np.fill_diagonal(similarity, 1.0)
    table = pd.DataFrame({
        "extractant": list(extractants),
        "chem__supercluster": [frame.loc[frame["extractant"] == e, "tanimoto_cluster"].iloc[0]
                               for e in extractants],
        "chem__level": [level[e] for e in extractants],
    })
    return ChemistryMap(table=table, similarity=similarity, extractants=extractants,
                        audit={"synthetic": True})


def _data(frame: pd.DataFrame) -> LevelData:
    return LevelData(frame=frame, blocks={"SYN": FEATURE_COLUMNS}, audit={"rows": len(frame)})


def _evaluate(frame: pd.DataFrame, *, seed: int = 3, folds: int = 3,
              arms=DEFAULT_ARMS, dispersion_arm: str | None = EXPANDED_ARM):
    params = LevelForestParameters(n_estimators=12, random_state=42, n_jobs=1)
    return runner.evaluate_seed(
        _data(frame), chemistry=_chemistry(frame), feature_set="SYN",
        feature_columns=FEATURE_COLUMNS, arms=list(arms), seed=seed, folds=folds,
        params=params, base_min_cells=10, weighting="cluster",
        dispersion_arm=dispersion_arm, log=lambda message: None)


# --------------------------------------------------------------------------- #
# (a) identical test rows
# --------------------------------------------------------------------------- #

def test_written_oof_scores_every_arm_on_identical_test_rows(tmp_path):
    frame = _cohort()
    result = _evaluate(frame)
    path = tmp_path / "oof_predictions.parquet"
    result.oof.to_parquet(path, index=False)
    oof = pd.read_parquet(path)

    assert len(oof) == len(frame)
    assert set(oof["row_id"]) == set(frame["row_id"])
    scored = {arm: frozenset(oof.loc[oof[f"prediction_{arm}"].notna(), "row_id"])
              for arm in DEFAULT_ARMS}
    assert len(set(scored.values())) == 1, "arms were scored on different rows"
    assert scored[BASE_ARM] == frozenset(frame["row_id"])

    # and per fold, not only in aggregate
    for fold, block in oof.groupby("outer_fold"):
        per_arm = {arm: frozenset(block.loc[block[f"prediction_{arm}"].notna(), "row_id"])
                   for arm in DEFAULT_ARMS}
        assert len(set(per_arm.values())) == 1, f"fold {fold} test rows moved with the arm"
    assert sorted(oof["outer_fold"].unique()) == [0, 1, 2]


def test_arms_are_genuinely_different_models_not_a_copied_column():
    """The identical-rows property must not be achieved by the arms being identical."""
    frame = _cohort()
    oof = _evaluate(frame).oof
    base = oof[f"prediction_{BASE_ARM}"].to_numpy()
    expanded = oof[f"prediction_{EXPANDED_ARM}"].to_numpy()
    assert not np.allclose(base, expanded)


def test_fold_records_partition_the_cohort_and_carry_the_manifest_keys():
    frame = _cohort()
    result = _evaluate(frame)
    seen: set[str] = set()
    for record in result.fold_records:
        for key in ("fold", "split_seed", "test_row_ids_sha256", "test_extractants",
                    "test_superclusters", "train_extractants_by_arm",
                    "train_superclusters_by_arm", "n_test_rows", "n_train_rows_by_arm"):
            assert key in record
        held_out = set(record["test_superclusters"])
        for arm, groups in record["train_superclusters_by_arm"].items():
            assert held_out.isdisjoint(groups), f"{arm} trained on a held-out super-cluster"
        seen |= set(record["test_extractants"])
    assert seen == set(frame["extractant"])
    assert result.integrity["ok"] is True
    assert result.integrity["all_rows_tested_once"] is True


# --------------------------------------------------------------------------- #
# (b) hard-chemistry bins fixed by the BASE reference
# --------------------------------------------------------------------------- #

def test_hard_chemistry_bins_do_not_move_with_the_arm():
    frame = _cohort()
    oof = _evaluate(frame).oof
    _, _, hard = gen6_metric_table(oof, list(DEFAULT_ARMS),
                                   similarity_column="nn_reference_tanimoto",
                                   thresholds=(0.4, 0.6))
    for endpoint, block in hard.groupby("endpoint"):
        assert block["n_rows"].nunique() == 1, f"{endpoint} row count depends on the arm"
        assert block["n_ligands"].nunique() == 1
        assert block["n_ecfp_clusters"].nunique() == 1
    # both hard endpoints must actually contain rows, or the test proves nothing
    counts = hard.drop_duplicates("endpoint").set_index("endpoint")["n_rows"]
    assert counts["nn<0.4"] > 0
    assert counts["nn<0.6"] > counts["nn<0.4"]


def test_the_bin_column_would_have_moved_had_the_arms_own_training_set_been_used():
    """Non-vacuity: BASE-referenced and EXPANDED-referenced bins genuinely differ."""
    frame = _cohort()
    oof = _evaluate(frame).oof
    reference = oof["nn_reference_tanimoto"].to_numpy()
    expanded = oof["nn_expanded_tanimoto"].to_numpy()
    assert np.all(expanded >= reference - 1e-9), "a superset reference cannot be less similar"
    assert np.any(expanded > reference + 1e-9)
    hard_reference = set(oof.loc[reference < 0.6, "row_id"])
    hard_expanded = set(oof.loc[expanded < 0.6, "row_id"])
    assert hard_reference != hard_expanded, (
        "the synthetic cohort does not distinguish the two references, so the "
        "arm-invariance test above would pass vacuously")


def test_bin_migration_counts_the_rows_the_fixed_reference_saves():
    frame = _cohort()
    oof = _evaluate(frame).oof
    table = runner.bin_migration_table(oof, (0.4, 0.6)).set_index("endpoint")
    assert list(table.index) == ["nn<0.4", "nn<0.6"]
    for endpoint in table.index:
        threshold = float(endpoint.split("<")[1])
        assert table.loc[endpoint, "n_rows_reference_bins"] == \
            int((oof["nn_reference_tanimoto"] < threshold).sum())
        # the EXPANDED reference is never less similar, so its bins can only shrink
        assert (table.loc[endpoint, "n_rows_if_each_arm_used_its_own_training_set"]
                <= table.loc[endpoint, "n_rows_reference_bins"])
    assert table["n_rows_that_would_change_bin"].sum() > 0, \
        "on this cohort the choice of reference must matter, or the test is vacuous"


def test_hard_chemistry_subsets_are_cut_on_the_reference_column_only():
    frame = _cohort()
    oof = _evaluate(frame).oof
    subsets = dict(runner.hard_chemistry_subsets(oof, (0.4, 0.6)))
    assert list(subsets) == ["all", "nn<0.4", "nn<0.6"]
    assert len(subsets["all"]) == len(oof)
    assert set(subsets["nn<0.4"]["row_id"]) == set(oof.loc[oof["nn_reference_tanimoto"] < 0.4,
                                                           "row_id"])
    assert set(subsets["nn<0.4"]["row_id"]) <= set(subsets["nn<0.6"]["row_id"])


# --------------------------------------------------------------------------- #
# (c) the information null
# --------------------------------------------------------------------------- #

def test_shuffled_arm_targets_are_a_permutation_confined_to_the_sparse_rows():
    frame = _cohort()
    target = frame["log_D"].to_numpy(dtype=float)
    dense = arm_membership(frame, min_cells=10)
    splits = diversity_splits(frame, n_splits=3, seed=5, base_min_cells=10)
    changed_somewhere = False
    for split in splits:
        index = split.train_index_by_arm[EXPANDED_ARM]
        honest = split.training_target(EXPANDED_ARM, target)
        shuffled = split.training_target(SHUFFLED_ARM, target)
        assert np.array_equal(split.train_index_by_arm[SHUFFLED_ARM], index), \
            "the null must train on the same ROWS, only the targets may move"
        assert np.allclose(np.sort(honest), np.sort(shuffled)), "not a permutation"
        sparse_positions = ~dense[index]
        assert np.array_equal(honest[~sparse_positions], shuffled[~sparse_positions]), \
            "dense targets were touched; the null would then be a different experiment"
        assert np.allclose(np.sort(honest[sparse_positions]), np.sort(shuffled[sparse_positions]))
        changed_somewhere |= not np.array_equal(honest[sparse_positions],
                                                shuffled[sparse_positions])
    assert changed_somewhere, "no fold actually permuted anything"


def test_honest_arms_train_on_untouched_targets():
    frame = _cohort()
    target = frame["log_D"].to_numpy(dtype=float)
    for split in diversity_splits(frame, n_splits=3, seed=5, base_min_cells=10):
        for arm in (BASE_ARM, EXPANDED_ARM, ROWMATCHED_ARM):
            index = split.train_index_by_arm[arm]
            assert np.array_equal(split.training_target(arm, target), target[index])


# --------------------------------------------------------------------------- #
# (d) fail-closed completion
# --------------------------------------------------------------------------- #

def _stub_manifest(output_dir: Path) -> RunManifest:
    """A manifest holding every required key, so only the *checks* can fail."""
    manifest = RunManifest(layer="gen6_test", run_id="unit-test")
    manifest.record_many({
        "dataset_path": "synthetic", "dataset_file_sha256": "0" * 64,
        "source_table_sha256": "1" * 64, "feature_registry_sha256": "2" * 64,
        "code_sha256": {"synthetic.py": "3" * 64},
        "chemistry_cluster_definition": {"identity": "canonical_smiles"},
        "provenance_state": {"status": "not_audited_in_this_run"},
        "model_seed": 42, "split_seeds": [104729], "preprocessing": [{"step": "none"}],
    })
    manifest.record_split(definition={"group_column": "tanimoto_cluster"}, folds=[{
        "fold": 0, "split_seed": 104729, "test_row_ids_sha256": "4" * 64,
        "test_extractants": ["D00"], "test_superclusters": ["sc00"],
        "train_extractants_by_arm": {BASE_ARM: ["D01"]},
        "train_superclusters_by_arm": {BASE_ARM: ["sc01"]},
        "n_test_rows": 12, "n_train_rows_by_arm": {BASE_ARM: 12},
    }])
    for name in runner.REQUIRED_ARTIFACTS:
        (output_dir / name).write_text("placeholder\n")
    return manifest


def test_no_success_marker_when_an_integrity_check_fails(tmp_path):
    manifest = _stub_manifest(tmp_path)
    validation, success = finalise = runner.finalise_run(
        tmp_path, manifest=manifest,
        checks={"split_integrity": {"ok": False, "detail": "extractant on both sides"},
                "identical_test_rows_across_arms": {"ok": True}})
    assert success is None
    assert not (tmp_path / "_SUCCESS.json").exists()
    assert (tmp_path / "_FAILED.json").exists()
    assert validation["ok"] is False
    assert "split_integrity" in validation["failed_checks"]
    assert finalise[0] is validation


def test_success_marker_when_every_check_passes(tmp_path):
    manifest = _stub_manifest(tmp_path)
    validation, success = runner.finalise_run(
        tmp_path, manifest=manifest,
        checks={"split_integrity": {"ok": True}, "identical_test_rows_across_arms": {"ok": True}})
    assert validation["ok"] is True
    assert success is not None and success.exists()
    assert not (tmp_path / "_FAILED.json").exists()
    payload = json.loads((tmp_path / "_SUCCESS.json").read_text())
    assert payload["validation_ok"] is True
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text())
    assert "validation.json" in hashes["files"], \
        "artifact hashes must cover the validation record written after the manifest"


def test_missing_artifact_also_blocks_success(tmp_path):
    manifest = _stub_manifest(tmp_path)
    (tmp_path / "decision_report.md").unlink()
    validation, success = runner.finalise_run(tmp_path, manifest=manifest,
                                              checks={"split_integrity": {"ok": True}})
    assert success is None
    assert "decision_report.md" in validation["missing_artifacts"]


# --------------------------------------------------------------------------- #
# The checks themselves
# --------------------------------------------------------------------------- #

def _seed_result(oof: pd.DataFrame, *, ok: bool = True) -> runner.SeedResult:
    return runner.SeedResult(feature_set="SYN", seed=3, oof=oof,
                             integrity={"ok": ok, "all_rows_tested_once": ok})


def test_build_checks_passes_on_a_clean_run():
    frame = _cohort()
    result = _evaluate(frame)
    _, _, hard = gen6_metric_table(result.oof, list(DEFAULT_ARMS),
                                   similarity_column="nn_reference_tanimoto")
    hard.insert(0, "split_seed", result.seed)
    hard.insert(0, "feature_set", result.feature_set)
    checks = runner.build_checks(
        results=[result], oof=result.oof, hard_metrics=hard, arms=list(DEFAULT_ARMS),
        cohort_extractants=set(frame["extractant"]), chemistry_extractants=set(frame["extractant"]))
    assert all(entry["ok"] for entry in checks.values())


def test_build_checks_flags_a_missing_similarity_value():
    """A NaN similarity drops the row from every hard-chemistry endpoint silently.

    ``metrics.hard_chemistry_endpoints`` builds its "all" row as
    ``similarity < inf``, and ``NaN < inf`` is False, so a missing similarity
    removes the row from the endpoint that is supposed to be the whole test set —
    without an error and without changing the arms' relative n. The runner
    therefore checks the column is complete instead of assuming it.
    """
    frame = _cohort()
    result = _evaluate(frame)
    damaged = result.oof.copy()
    damaged.loc[damaged.index[:3], "nn_reference_tanimoto"] = np.nan
    checks = runner.build_checks(
        results=[_seed_result(damaged)], oof=damaged, hard_metrics=pd.DataFrame(),
        arms=list(DEFAULT_ARMS), cohort_extractants=set(frame["extractant"]),
        chemistry_extractants=set(frame["extractant"]))
    assert checks["similarity_column_complete"]["ok"] is False
    assert checks["similarity_column_complete"]["n_missing"] == 3


def test_build_checks_flags_a_failed_split_integrity():
    frame = _cohort()
    result = _evaluate(frame)
    checks = runner.build_checks(
        results=[_seed_result(result.oof, ok=False)], oof=result.oof,
        hard_metrics=pd.DataFrame(), arms=list(DEFAULT_ARMS),
        cohort_extractants=set(frame["extractant"]), chemistry_extractants=set(frame["extractant"]))
    assert checks["split_integrity"]["ok"] is False


def test_build_checks_flags_hard_chemistry_bins_that_moved_with_the_arm():
    frame = _cohort()
    result = _evaluate(frame)
    hard = pd.DataFrame([
        {"feature_set": "SYN", "split_seed": 3, "endpoint": "nn<0.4", "arm": BASE_ARM,
         "n_rows": 100},
        {"feature_set": "SYN", "split_seed": 3, "endpoint": "nn<0.4", "arm": EXPANDED_ARM,
         "n_rows": 140},
    ])
    checks = runner.build_checks(
        results=[_seed_result(result.oof)], oof=result.oof, hard_metrics=hard,
        arms=list(DEFAULT_ARMS), cohort_extractants=set(frame["extractant"]),
        chemistry_extractants=set(frame["extractant"]))
    assert checks["hard_chemistry_bins_arm_invariant_by_construction"]["ok"] is False


def test_build_checks_flags_a_chemistry_map_that_does_not_cover_the_cohort():
    frame = _cohort()
    result = _evaluate(frame)
    checks = runner.build_checks(
        results=[_seed_result(result.oof)], oof=result.oof, hard_metrics=pd.DataFrame(),
        arms=list(DEFAULT_ARMS), cohort_extractants=set(frame["extractant"]) | {"UNKNOWN"},
        chemistry_extractants=set(frame["extractant"]))
    assert checks["chemistry_map_covers_cohort"]["ok"] is False


# --------------------------------------------------------------------------- #
# Contrasts, pooling and hypothesis scoring
# --------------------------------------------------------------------------- #

def test_preregistered_contrasts_match_the_protocol():
    by_name = {c.name: c for c in runner.CONTRASTS}
    assert by_name["EXPANDED_vs_BASE"].reference == BASE_ARM
    assert by_name["EXPANDED_vs_BASE"].candidate == EXPANDED_ARM
    assert by_name["EXPANDED_ROWMATCHED_vs_BASE"].candidate == ROWMATCHED_ARM
    assert by_name["EXPANDED_vs_EXPANDED_SHUFFLED"].reference == SHUFFLED_ARM
    assert by_name["EXPANDED_SHUFFLED_vs_BASE"].candidate == SHUFFLED_ARM
    preregistered = {c.name for c in runner.CONTRASTS if c.preregistered}
    assert preregistered == {"EXPANDED_vs_BASE", "EXPANDED_ROWMATCHED_vs_BASE",
                             "EXPANDED_vs_EXPANDED_SHUFFLED", "EXPANDED_SHUFFLED_vs_BASE"}
    assert by_name["EXPANDED_vs_EXPANDED_ROWMATCHED"].preregistered is False


def test_contrast_sign_is_positive_when_the_candidate_is_better():
    frame = _cohort()
    oof = _evaluate(frame).oof
    # make EXPANDED exactly right and BASE exactly wrong by a constant
    rigged = oof.copy()
    rigged[f"prediction_{EXPANDED_ARM}"] = rigged["log_D"]
    rigged[f"prediction_{BASE_ARM}"] = rigged["log_D"] + 1.0
    per_unit = per_unit_statistics(rigged, [BASE_ARM, EXPANDED_ARM])
    table = runner.contrast_frame(per_unit, block_of_unit=runner.unit_blocks(rigged),
                                  arms=[BASE_ARM, EXPANDED_ARM], replicates=200, seed=1)
    row = table[(table["comparison"] == "EXPANDED_vs_BASE") & (table["statistic"] == "mae")].iloc[0]
    assert row["point_delta"] == pytest.approx(1.0)
    assert row["ci95_low"] > 0


def test_unit_blocks_maps_the_scoring_unit_to_the_resampling_block():
    frame = _cohort()
    blocks = runner.unit_blocks(frame)
    assert blocks["ec00"] == "sc00"
    assert set(blocks) == set(frame["ecfp_cluster"])
    assert len(set(blocks.values())) < len(blocks), "the two units must not be identical here"


def test_pool_per_unit_over_seeds_averages_each_unit():
    a = pd.DataFrame({"unit": ["u1", "u1"], "arm": [BASE_ARM, EXPANDED_ARM], "mae": [1.0, 0.5]})
    b = pd.DataFrame({"unit": ["u1", "u1"], "arm": [BASE_ARM, EXPANDED_ARM], "mae": [3.0, 1.5]})
    pooled = runner.pool_per_unit_over_seeds([a, b])
    assert set(pooled["unit"]) == {"u1"}
    assert pooled.set_index("arm").loc[BASE_ARM, "mae"] == pytest.approx(2.0)
    assert pooled.set_index("arm").loc[EXPANDED_ARM, "mae"] == pytest.approx(1.0)
    assert set(pooled["n_seeds"]) == {2}


def _summary_row(**overrides) -> pd.DataFrame:
    base = {"feature_set": "SYN", "endpoint": "all", "comparison": "EXPANDED_vs_BASE",
            "statistic": "mae", "pooled_point_delta": 0.10, "pooled_ci95_low": 0.02,
            "pooled_ci95_high": 0.18, "mean_point_delta": 0.10, "seeds_positive": 5,
            "n_seeds": 5, "pooled_units_total": 100, "pooled_blocks": 40}
    base.update(overrides)
    return pd.DataFrame([base])


def _full_summary(**a1) -> pd.DataFrame:
    """A1 row plus the rows the other hypotheses read, all comfortably positive."""
    rows = [_summary_row(**a1)]
    rows.append(_summary_row(endpoint="nn<0.4", pooled_point_delta=0.30, pooled_ci95_low=0.10,
                             pooled_ci95_high=0.50, mean_point_delta=0.30,
                             pooled_units_total=30, pooled_blocks=25))
    rows.append(_summary_row(statistic="offset_mae", pooled_point_delta=0.09,
                             pooled_ci95_low=0.03, pooled_ci95_high=0.15,
                             mean_point_delta=0.09))
    rows.append(_summary_row(statistic="shape_mae", pooled_point_delta=0.01,
                             pooled_ci95_low=-0.01, pooled_ci95_high=0.03,
                             mean_point_delta=0.01))
    rows.append(_summary_row(comparison="EXPANDED_vs_EXPANDED_SHUFFLED",
                             pooled_point_delta=0.20, pooled_ci95_low=0.08,
                             pooled_ci95_high=0.33, mean_point_delta=0.20))
    rows.append(_summary_row(comparison="EXPANDED_vs_EXPANDED_ROWMATCHED",
                             pooled_point_delta=0.01, pooled_ci95_low=-0.02,
                             pooled_ci95_high=0.04, mean_point_delta=0.01))
    return pd.concat(rows, ignore_index=True)


def test_hypotheses_pass_when_every_pre_registered_condition_is_met():
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        _full_summary(), feature_set="SYN", thresholds=(0.4, 0.6), n_seeds=5)}
    assert [v.verdict for v in verdicts.values()] == ["PASS"] * 4


def test_a1_fails_only_on_evidence_against_it():
    against = runner.score_hypotheses(
        _full_summary(pooled_point_delta=-0.10, pooled_ci95_low=-0.20, pooled_ci95_high=-0.02,
                      seeds_positive=0),
        feature_set="SYN", thresholds=(0.4, 0.6), n_seeds=5)
    assert against[0].verdict == "FAIL"
    straddling = runner.score_hypotheses(
        _full_summary(pooled_ci95_low=-0.05, seeds_positive=3),
        feature_set="SYN", thresholds=(0.4, 0.6), n_seeds=5)
    assert straddling[0].verdict == "INCONCLUSIVE"


def test_a_single_seed_run_can_never_score_pass():
    # every row of a one-seed run carries n_seeds = 1, so the seed-agreement
    # clause of §5 cannot be evaluated for any hypothesis
    summary = _full_summary()
    summary[["seeds_positive", "n_seeds"]] = [1, 1]
    verdicts = runner.score_hypotheses(summary, feature_set="SYN", thresholds=(0.4, 0.6),
                                       n_seeds=1)
    assert {v.verdict for v in verdicts} == {"INCONCLUSIVE"}
    assert runner._seed_rule(1, 1) is None
    assert runner._seed_rule(4, 5) is True
    assert runner._seed_rule(3, 5) is False


def test_a3_needs_the_offset_gain_to_exceed_the_shape_gain():
    summary = _full_summary()
    summary.loc[summary["statistic"] == "shape_mae", "pooled_point_delta"] = 0.50
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        summary, feature_set="SYN", thresholds=(0.4, 0.6), n_seeds=5)}
    assert verdicts["A3"].verdict == "INCONCLUSIVE"


def test_a2_needs_the_hard_chemistry_gain_to_exceed_the_overall_gain():
    summary = _full_summary()
    summary.loc[summary["endpoint"] == "nn<0.4", "pooled_point_delta"] = 0.01
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        summary, feature_set="SYN", thresholds=(0.4, 0.6), n_seeds=5)}
    assert verdicts["A2"].verdict == "INCONCLUSIVE"


def test_a4_fails_when_the_information_null_is_as_good_as_the_honest_arm():
    summary = _full_summary()
    mask = summary["comparison"] == "EXPANDED_vs_EXPANDED_SHUFFLED"
    summary.loc[mask, ["pooled_point_delta", "pooled_ci95_low", "pooled_ci95_high"]] = \
        [-0.05, -0.12, -0.01]
    verdicts = {v.hypothesis: v for v in runner.score_hypotheses(
        summary, feature_set="SYN", thresholds=(0.4, 0.6), n_seeds=5)}
    assert verdicts["A4"].verdict == "FAIL"


# --------------------------------------------------------------------------- #
# Dispersion, feature sets and the CLI
# --------------------------------------------------------------------------- #

def test_tree_dispersion_is_zero_on_a_constant_target_and_positive_otherwise():
    frame = _cohort()
    params = LevelForestParameters(n_estimators=12, random_state=0, n_jobs=1)
    constant = frame.assign(log_D=1.0)
    flat = LevelRegressor(FEATURE_COLUMNS, params).fit(constant, constant["log_D"])
    spread = runner.tree_dispersion(flat, constant)
    assert spread.shape == (len(constant),)
    assert np.allclose(spread, 0.0)

    noisy = LevelRegressor(FEATURE_COLUMNS, params).fit(frame, frame["log_D"])
    assert (runner.tree_dispersion(noisy, frame) > 0).any()


def test_dispersion_column_is_recorded_only_for_the_named_arm():
    frame = _cohort()
    with_dispersion = _evaluate(frame, dispersion_arm=EXPANDED_ARM).oof
    assert with_dispersion["ood__prediction_sd"].notna().all()
    assert set(with_dispersion["ood__prediction_sd_arm"]) == {EXPANDED_ARM}
    without = _evaluate(frame, dispersion_arm=None).oof
    assert "ood__prediction_sd" not in without.columns


def test_resolve_feature_sets_skips_sets_whose_blocks_are_absent():
    frame = _cohort()
    data = LevelData(frame=frame, blocks={"METAL": ("feat__metal",), "COND": ("feat__condition",)},
                     audit={})
    messages: list[str] = []
    resolved = runner.resolve_feature_sets(data, ["MC", "MC_lig2d_ext_massaction"], messages.append)
    assert list(resolved) == ["MC"]
    assert resolved["MC"] == ("feat__metal", "feat__condition")
    assert any("MC_lig2d_ext_massaction" in m for m in messages)


def test_pilot_defaults_are_one_seed_one_feature_set_and_fewer_trees():
    args = runner.parse_args(["--pilot"])
    assert args.pilot is True
    assert args.n_estimators == 400 and len(args.split_seeds) == 5, \
        "parse_args must not silently apply the pilot budget; main() does that"
    default = runner.parse_args([])
    assert list(default.feature_sets) == list(runner.DEFAULT_FEATURE_SETS)
    assert list(default.split_seeds) == list(runner.DEFAULT_SEEDS)
    assert default.weighting == "cluster"
    assert default.tree_dispersion is True
    assert runner.parse_args(["--no-tree-dispersion"]).tree_dispersion is False
    assert runner.parse_args(["--weighting", "row"]).weighting == "row"


def test_fold_seed_formula_matches_gen5():
    """Reproduction depends on this constant, so it is asserted, not commented."""
    assert runner.FOLD_SEED_STRIDE == 1009
    assert runner.FOLD_SEED_OFFSET == 9_999_991


def test_row_weighting_changes_the_fit():
    frame = _cohort()
    weighted = _evaluate(frame).oof[f"prediction_{EXPANDED_ARM}"].to_numpy()
    params = LevelForestParameters(n_estimators=12, random_state=42, n_jobs=1)
    unweighted = runner.evaluate_seed(
        _data(frame), chemistry=_chemistry(frame), feature_set="SYN",
        feature_columns=FEATURE_COLUMNS, arms=list(DEFAULT_ARMS), seed=3, folds=3, params=params,
        base_min_cells=10, weighting="row", dispersion_arm=None,
        log=lambda message: None).oof[f"prediction_{EXPANDED_ARM}"].to_numpy()
    assert not np.allclose(weighted, unweighted), \
        "the disclosed weighting sensitivity must actually change something"


def test_default_feature_sets_resolve():
    """argparse `choices` validates what the user types, never the default.

    An invalid default crashes the script only when it is run with no arguments —
    exactly how gen6_phase0.py invokes it, which is how this was found.
    """
    from lanthanide_separation.levels import LEVEL_ARMS
    unknown = [n for n in runner.DEFAULT_FEATURE_SETS if n not in LEVEL_ARMS]
    assert unknown == [], f"default feature sets that do not exist: {unknown}"
    assert runner.parse_args([]).feature_sets == list(runner.DEFAULT_FEATURE_SETS)


def test_unknown_feature_set_fails_with_a_useful_message():
    data = LevelData(frame=pd.DataFrame({"log_D": [0.0]}), blocks={"METAL": ("m",)})
    with pytest.raises(SystemExit, match="unknown feature set"):
        runner.resolve_feature_sets(data, ["not_an_arm"], lambda message: None)

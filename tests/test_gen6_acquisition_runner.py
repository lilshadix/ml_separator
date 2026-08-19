"""Tests for the Experiment F runner (``scripts/run_ligand_acquisition_sim.py``).

Experiment F compares acquisition policies on a fixed test set, and there are
exactly three ways for that comparison to be void rather than merely noisy.  All
three are tested here on a synthetic world small enough to reason about:

1. **The hard-chemistry subset moves.**  "Hard chemistry" is a test row far from
   the *start* cohort.  If the reference were recomputed against the growing
   training set, the subset would shrink precisely for the policies that buy
   distant chemistry — the policies the experiment is meant to reward — and the
   effect would be manufactured rather than measured.  The evaluator freezes the
   masks at construction; ``test_hard_masks_are_fixed_across_steps_and_policies``
   proves they never move, and
   ``test_a_policy_reference_would_have_moved_the_mask`` proves the world is
   built so that they *could* have moved, i.e. the first test is not vacuous.
2. **An acquisition reaches into the test set.**  The closure receives the
   revealed training rows at every checkpoint and must raise on any overlap.
3. **A verdict is over-read.**  PASS requires the pre-registered condition *and*
   the pre-registered replication; a straddling interval is INCONCLUSIVE and only
   an interval excluding the predicted direction is a FAIL.

Also pinned: the vectorised similarity helper against the ``ChemistryMap`` method
it replaces, the F4 start-cap arithmetic, the checkpoint-frame layout the paired
bootstrap consumes, and the fail-closed completion wiring.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen6.acquisition import run_acquisition, start_cohort
from lanthanide_separation.gen6.chemistry import build_chemistry_map
from lanthanide_separation.gen6.manifest import RunManifest
from lanthanide_separation.levels import LevelData, LevelForestParameters

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ligand_acquisition_sim.py"
SPEC = importlib.util.spec_from_file_location("run_ligand_acquisition_sim", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
# ``scripts/`` is not a package, so the runner is loaded by path.  It must be put
# in ``sys.modules`` *before* execution: the module defines dataclasses, and
# ``@dataclass`` resolves annotations through ``sys.modules[cls.__module__]``.
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

FEATURE_COLUMNS = ("f_cond", "f_lig", "f_noise")


# --------------------------------------------------------------------------- #
# A small acquisition world
# --------------------------------------------------------------------------- #

def _world(n_ligands: int = 16, rows_per_ligand: int = 6, n_bits: int = 48, seed: int = 5):
    """Source table + cohort frame + chemistry map.

    Ligands 0–5 share one chemotype (near-identical fingerprints) and are the
    deep start material; 6–11 are scattered training chemistry; 12–15 are the
    held-out test chemotypes.  The scattered ligands are deliberately built so
    that *some* of them are close to the test ligands: that is what lets a
    policy's own training set become a different similarity reference from the
    start cohort, which is what makes the mask-invariance test non-vacuous.
    """
    rng = np.random.default_rng(seed)
    source_rows, frame_rows = [], []
    base = np.zeros(n_bits, dtype=int)
    base[rng.choice(n_bits, 12, replace=False)] = 1
    test_base = np.zeros(n_bits, dtype=int)
    test_base[rng.choice(n_bits, 12, replace=False)] = 1
    for i in range(n_ligands):
        if i < 6:                                   # the start chemotype
            bits = base.copy()
            flip = rng.choice(n_bits, 1)
            bits[flip] = 1 - bits[flip]
        elif i >= 12:                               # the test chemotypes
            bits = test_base.copy()
            flip = rng.choice(n_bits, 2, replace=False)
            bits[flip] = 1 - bits[flip]
        elif i in (10, 11):                         # pool ligands NEAR the test set
            bits = test_base.copy()
            flip = rng.choice(n_bits, 5, replace=False)
            bits[flip] = 1 - bits[flip]
        else:                                       # pool ligands far from everything
            bits = np.zeros(n_bits, dtype=int)
            bits[rng.choice(n_bits, 12, replace=False)] = 1
        level = float(rng.normal(scale=1.5))
        for j in range(rows_per_ligand):
            smiles = f"S{i:02d}"
            source_rows.append({
                "canonical_smiles": smiles, "extractant_name": f"L{i}", "metal_symbol": "Nd",
                "cond__acid_concentration_M": float(j), "log_D": level,
                **{f"ecfp_{b}": int(bits[b]) for b in range(n_bits)}})
            frame_rows.append({
                "row_id": f"{smiles}-{j}", "extractant": smiles, "ecfp_cluster": f"e{i:02d}",
                "tanimoto_cluster": ("start" if i < 6 else ("test" if i >= 12 else f"t{i:02d}")),
                "condition_id": f"c{j}", "series_id": f"ser{i:02d}", "metal_symbol": "Nd",
                "metal_Z": 60.0, "n_replicates": 1,
                "log_D": level + 0.3 * j + float(rng.normal(scale=0.05)),
                "f_cond": float(j), "f_lig": float(i), "f_noise": float(rng.normal())})
    frame = pd.DataFrame(frame_rows)
    data = LevelData(frame=frame, blocks={"X": FEATURE_COLUMNS}, audit={"rows": len(frame)})
    chemistry = build_chemistry_map(pd.DataFrame(source_rows))
    return frame, data, chemistry


def _split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Positional (train, test) indices: the ``test`` chemotype is held out."""
    is_test = (frame["tanimoto_cluster"] == "test").to_numpy()
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def _evaluator(frame, chemistry, test_index, start, thresholds=(0.4, 0.6)):
    return runner.FoldEvaluator(frame, chemistry=chemistry, test_index=test_index, start=start,
                                thresholds=thresholds)


# --------------------------------------------------------------------------- #
# (a) the similarity helper the masks are cut from
# --------------------------------------------------------------------------- #

def test_similarity_helper_matches_the_chemistry_map_method():
    """The vectorised helper must be numerically identical to the public method."""
    frame, _, chemistry = _world()
    query = sorted(frame.loc[frame["tanimoto_cluster"] == "test", "extractant"].unique())
    reference = sorted(frame.loc[frame["tanimoto_cluster"] == "start", "extractant"].unique())
    fast = runner.max_similarity_to_reference(chemistry, query, reference)
    slow = chemistry.nearest_neighbour(query, reference, exclude_self=False)["nn_tanimoto"]
    assert np.allclose(fast, slow.to_numpy(dtype=float))
    # an empty reference is 0.0, never NaN: the hardest ligands must not vanish
    assert np.array_equal(runner.max_similarity_to_reference(chemistry, query, []),
                          np.zeros(len(query)))


def test_similarity_helper_refuses_an_extractant_the_map_does_not_know():
    frame, _, chemistry = _world()
    with pytest.raises(KeyError, match="absent from the frozen chemistry map"):
        runner.max_similarity_to_reference(chemistry, ["not-a-smiles"], ["S00"])


# --------------------------------------------------------------------------- #
# (b) the hard masks are computed ONCE per fold and never move
# --------------------------------------------------------------------------- #

def test_hard_masks_are_fixed_across_steps_and_policies():
    frame, data, chemistry = _world()
    train_index, test_index = _split(frame)
    start = start_cohort(frame, train_index, n_ligands=4)
    evaluator = _evaluator(frame, chemistry, test_index, start)
    first = {t: mask.copy() for t, mask in evaluator.masks.items()}
    assert first[0.4].sum() > 0, "the synthetic world must have hard rows or this proves nothing"

    params = LevelForestParameters(n_estimators=8, random_state=0, n_jobs=1)
    for policy in ("random", "maxmin", "same_chemotype_first"):
        run_acquisition(frame, data, start=start, train_index=train_index, test_index=test_index,
                        policy=policy, feature_columns=FEATURE_COLUMNS, params=params,
                        chemistry=chemistry, budget_per_ligand=2, n_steps=4, checkpoints=(1, 2, 4),
                        seed=7, evaluate=evaluator)
    for threshold, mask in evaluator.masks.items():
        assert np.array_equal(mask, first[threshold]), f"the nn<{threshold} mask moved"
    audit = evaluator.mask_audit()
    assert audit["n_calls"] == 3 * 4                       # checkpoint 0 plus three checkpoints
    assert all(n == 1 for n in audit["distinct_mask_digests"].values()), \
        "more than one mask digest means the subset changed during the run"


def test_a_policy_reference_would_have_moved_the_mask():
    """Non-vacuity: the start-cohort reference and a policy's own training set differ.

    ``maxmin`` buys the pool ligands closest to nothing, but the world also holds
    pool ligands close to the test chemotype; once any of those is acquired, the
    nearest-neighbour distance of the test rows to the *training set* is strictly
    larger than to the start cohort, so a runner that recomputed the mask would be
    scoring a different, smaller subset.  If this test fails, the fixture no
    longer distinguishes the two references and the test above is meaningless.
    """
    frame, _, chemistry = _world()
    train_index, test_index = _split(frame)
    start = start_cohort(frame, train_index, n_ligands=4)
    evaluator = _evaluator(frame, chemistry, test_index, start)
    near_pool = ["S10", "S11"]
    with_pool = runner.max_similarity_to_reference(
        chemistry, evaluator.test_ligands, list(start.extractants) + near_pool)
    assert np.all(with_pool >= evaluator.nn_by_ligand - 1e-9), \
        "a superset reference cannot be less similar"
    assert np.any(with_pool > evaluator.nn_by_ligand + 1e-9)
    assert (with_pool < 0.4).sum() < (evaluator.nn_by_ligand < 0.4).sum(), \
        "a policy-specific reference must shrink the hard subset, or the fixture is vacuous"


def test_hard_metrics_are_computed_on_the_frozen_mask():
    frame, _, chemistry = _world()
    train_index, test_index = _split(frame)
    start = start_cohort(frame, train_index, n_ligands=4)
    evaluator = _evaluator(frame, chemistry, test_index, start)
    prediction = evaluator.y + 1.0                       # constant +1 error everywhere
    record = evaluator(prediction, train_index[:10], np.array(start.extractants))
    assert record["macro_mae"] == pytest.approx(1.0)
    assert record["hard_nn0.4__macro_mae"] == pytest.approx(1.0)
    assert record["hard_nn0.4__n_rows"] == int(evaluator.masks[0.4].sum())
    assert record["hard_nn0.6__n_rows"] >= record["hard_nn0.4__n_rows"]


# --------------------------------------------------------------------------- #
# (c) no revealed row is a test row
# --------------------------------------------------------------------------- #

def test_evaluator_raises_when_a_revealed_row_is_a_test_row():
    frame, _, chemistry = _world()
    train_index, test_index = _split(frame)
    start = start_cohort(frame, train_index, n_ligands=4)
    evaluator = _evaluator(frame, chemistry, test_index, start)
    leaky = np.concatenate([train_index[:5], test_index[:2]])
    with pytest.raises(RuntimeError, match="leaked into the fixed test set"):
        evaluator(evaluator.y, leaky, np.array(start.extractants))
    assert evaluator.mask_audit()["n_test_rows_revealed"] == 2


def test_a_real_run_never_reveals_a_test_row():
    frame, data, chemistry = _world()
    train_index, test_index = _split(frame)
    start = start_cohort(frame, train_index, n_ligands=4)
    evaluator = _evaluator(frame, chemistry, test_index, start)
    params = LevelForestParameters(n_estimators=8, random_state=0, n_jobs=1)
    trace = run_acquisition(frame, data, start=start, train_index=train_index,
                            test_index=test_index, policy="maxmin",
                            feature_columns=FEATURE_COLUMNS, params=params, chemistry=chemistry,
                            budget_per_ligand=2, n_steps=4, checkpoints=(1, 2, 4), seed=3,
                            evaluate=evaluator)
    assert evaluator.mask_audit()["n_test_rows_revealed"] == 0
    assert (trace.checkpoints["n_test_rows_revealed"] == 0).all()
    # and the acquired ligands are pool ligands, never test ligands
    assert set(trace.steps["extractant"]).isdisjoint(set(evaluator.test_ligands))


# --------------------------------------------------------------------------- #
# (d) coverage metrics
# --------------------------------------------------------------------------- #

def test_coverage_metrics_grow_with_the_training_set():
    frame, _, chemistry = _world()
    train_index, test_index = _split(frame)
    start = start_cohort(frame, train_index, n_ligands=4)
    evaluator = _evaluator(frame, chemistry, test_index, start)
    prediction = evaluator.y.copy()
    near_rows = np.flatnonzero(frame["extractant"].isin(["S10", "S11"]).to_numpy())
    only_start = evaluator(prediction, start.row_index, np.array(start.extractants))
    with_near = evaluator(prediction, np.concatenate([start.row_index, near_rows]),
                          np.array(start.extractants))
    assert only_start["n_train_chemotypes"] == 1
    assert with_near["n_train_chemotypes"] > only_start["n_train_chemotypes"]
    assert with_near["n_train_ligands"] == only_start["n_train_ligands"] + 2
    assert with_near["mean_nn_test_to_train"] > only_start["mean_nn_test_to_train"]
    # the START reference is untouched by what was acquired
    assert with_near["mean_nn_test_to_start"] == pytest.approx(only_start["mean_nn_test_to_start"])


# --------------------------------------------------------------------------- #
# (e) F4: the start cap and the depth arm
# --------------------------------------------------------------------------- #

def test_cap_start_cohort_keeps_n_rows_per_ligand_and_returns_the_rest():
    frame, _, _ = _world()
    train_index, _ = _split(frame)
    start = start_cohort(frame, train_index, n_ligands=4)
    capped, withheld = runner.cap_start_cohort(frame, start, cap=2,
                                               rng=np.random.default_rng(0))
    extractant = frame["extractant"].astype(str).to_numpy()
    for name in start.extractants:
        assert (extractant[capped.row_index] == name).sum() == 2
        assert len(withheld[name]) == 6 - 2
        assert set(withheld[name]).isdisjoint(set(capped.row_index.tolist()))
    assert capped.extractants == start.extractants
    assert capped.audit["n_start_rows"] == 4 * 2
    assert capped.audit["n_start_rows_withheld"] == 4 * 4
    assert set(capped.row_index.tolist()) <= set(start.row_index.tolist())


def test_depth_arm_buys_the_same_rows_from_the_start_cohort_only():
    frame, _, chemistry = _world()
    train_index, test_index = _split(frame)
    start = start_cohort(frame, train_index, n_ligands=4)
    capped, withheld = runner.cap_start_cohort(frame, start, cap=2,
                                               rng=np.random.default_rng(0))
    evaluator = _evaluator(frame, chemistry, test_index, capped)
    params = LevelForestParameters(n_estimators=8, random_state=0, n_jobs=1)
    trace = runner.run_depth_on_start(
        frame, start=capped, withheld=withheld, test_index=test_index,
        feature_columns=FEATURE_COLUMNS, params=params, budget_per_ligand=2, n_steps=4,
        checkpoints=(1, 2, 4), seed=11, evaluate=evaluator)
    assert trace.policy == runner.DEPTH_ARM
    assert set(trace.steps["extractant"]) <= set(capped.extractants), \
        "the depth arm may only buy rows of ligands it already holds"
    assert (trace.steps["n_rows_revealed"] == 2).all()
    assert list(trace.checkpoints["step"]) == [0, 1, 2, 4]
    # same row budget per step as a b = 2 acquisition policy
    assert trace.checkpoints["n_train_rows"].tolist() == [
        len(capped.row_index) + 2 * s for s in (0, 1, 2, 4)]
    # and it buys no new chemistry at all
    assert (trace.checkpoints["n_train_chemotypes"] == 1).all()


# --------------------------------------------------------------------------- #
# (f) the frame the paired bootstrap consumes
# --------------------------------------------------------------------------- #

def _fold_result(frame, chemistry, *, arms=("random", "maxmin"), checkpoints=(0, 1, 2),
                 replicates=1, seed=1, fold=0):
    """A FoldResult filled with deterministic predictions, no fitting involved."""
    train_index, test_index = _split(frame)
    start = start_cohort(frame, train_index, n_ligands=4)
    evaluator = _evaluator(frame, chemistry, test_index, start)
    # one real evaluation, so the mask digests exist exactly as a live run leaves them
    evaluator(evaluator.y, start.row_index, np.array(start.extractants))
    result = runner.FoldResult(seed=seed, fold=fold, start_audit=dict(start.audit),
                               start_extractants=tuple(start.extractants))
    result.test_frame = pd.DataFrame({
        "split_seed": seed, "fold": fold,
        "row_id": frame["row_id"].to_numpy()[test_index],
        "extractant": evaluator.ligand, "ecfp_cluster": evaluator.cluster,
        "tanimoto_cluster": evaluator.chemotype, "log_D": evaluator.y,
        "nn_to_start_tanimoto": evaluator.nn_to_start,
        **{f"hard_nn{t:g}": evaluator.masks[t] for t in evaluator.masks}})
    result.evaluator_audit = evaluator.mask_audit()
    for a, arm in enumerate(arms):
        for replicate in range(replicates):
            for checkpoint in checkpoints:
                # arm 0 is exact, arm 1 is off by a constant — a known ordering
                result.predictions[(arm, replicate, checkpoint)] = \
                    evaluator.y + a * (1.0 + replicate)
    return result


def test_checkpoint_frame_carries_one_replicate_mean_column_per_arm():
    frame, _, chemistry = _world()
    result = _fold_result(frame, chemistry, replicates=2)
    table, dropped = runner.checkpoint_frame([result], 1, ["random", "maxmin"])
    assert dropped == []
    assert len(table) == len(result.test_frame)
    assert np.allclose(table["prediction_random"], result.test_frame["log_D"])
    # replicate means: maxmin is off by 1.0 and by 2.0 in the two replicates
    assert np.allclose(table["prediction_maxmin"] - table["log_D"], 1.5)
    assert {"ecfp_cluster", "tanimoto_cluster", "hard_nn0.4"} <= set(table.columns)


def test_checkpoint_frame_drops_a_fold_whole_when_an_arm_stopped_early():
    frame, _, chemistry = _world()
    complete = _fold_result(frame, chemistry, fold=0)
    partial = _fold_result(frame, chemistry, fold=1)
    for key in [k for k in partial.predictions if k[0] == "maxmin" and k[2] == 2]:
        partial.predictions.pop(key)
    table, dropped = runner.checkpoint_frame([complete, partial], 2, ["random", "maxmin"])
    assert dropped == [(1, 1)]
    assert set(table["fold"]) == {0}


def test_contrast_table_signs_the_delta_so_positive_means_the_candidate_is_better():
    frame, _, chemistry = _world()
    results = [_fold_result(frame, chemistry, fold=f, seed=1) for f in (0, 1)]
    table = runner.contrast_table(
        results, arms=["random", "maxmin"], reference="random", checkpoints=[1],
        thresholds=(0.4,), replicates=200, seed=1, log=lambda message: None)
    assert not table.empty
    row = table[(table["statistic"] == "mae") & (table["endpoint"] == "all")].iloc[0]
    assert row["comparison"] == "maxmin_vs_random"
    # random is exact and maxmin is off by 1.0, so reference - candidate = -1.0
    assert row["point_delta"] == pytest.approx(-1.0)
    assert set(table["endpoint"]) == {"all", "hard_nn<0.4"}


# --------------------------------------------------------------------------- #
# (g) verdict logic
# --------------------------------------------------------------------------- #

def _contrasts(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _f1_row(checkpoint: int, low: float, high: float, comparison: str = "maxmin_vs_random",
            endpoint: str = "hard_nn<0.4", statistic: str = "mae") -> dict:
    return {"checkpoint": checkpoint, "endpoint": endpoint, "comparison": comparison,
            "statistic": statistic, "point_delta": (low + high) / 2, "ci95_low": low,
            "ci95_high": high, "bca_low": low, "bca_high": high}


def _score(rows: list[dict], **kwargs):
    verdicts = runner.score_hypotheses(
        _contrasts(rows), thresholds=(0.4, 0.6),
        f4_enabled=kwargs.pop("f4_enabled", False), f4_reason="skipped in this test",
        **kwargs)
    return {v.hypothesis: v for v in verdicts}


def test_f1_passes_only_when_half_the_checkpoints_from_five_onwards_exclude_zero():
    rows = [_f1_row(5, 0.10, 0.30), _f1_row(8, -0.05, 0.30), _f1_row(12, 0.02, 0.40)]
    assert _score(rows)["F1"].verdict == "PASS"
    weak = [_f1_row(5, -0.10, 0.30), _f1_row(8, -0.05, 0.30), _f1_row(12, 0.02, 0.40)]
    assert _score(weak)["F1"].verdict == "INCONCLUSIVE"


def test_f1_only_scores_checkpoints_from_five_onwards():
    """Early checkpoints must not be able to carry the verdict."""
    rows = [_f1_row(1, 0.5, 0.9), _f1_row(2, 0.5, 0.9), _f1_row(3, 0.5, 0.9),
            _f1_row(5, -0.5, 0.9), _f1_row(8, -0.5, 0.9)]
    verdict = _score(rows)["F1"]
    assert verdict.verdict == "INCONCLUSIVE"
    assert "0/2 checkpoints" in verdict.evidence


def test_f1_fails_when_the_interval_excludes_the_predicted_direction():
    rows = [_f1_row(5, -0.40, -0.10), _f1_row(8, -0.35, -0.05)]
    assert _score(rows)["F1"].verdict == "FAIL"


def test_a_short_run_cannot_pass_but_can_still_fail():
    """The pre-registered replication is part of the pass condition, as in Experiment A."""
    passing = [_f1_row(5, 0.10, 0.30), _f1_row(8, 0.10, 0.30)]
    verdict = _score(passing, replication_ok=False, replication_note="pilot")["F1"]
    assert verdict.verdict.startswith("INCONCLUSIVE")
    assert "pilot" in verdict.evidence
    failing = [_f1_row(5, -0.30, -0.10), _f1_row(8, -0.30, -0.10)]
    assert _score(failing, replication_ok=False, replication_note="pilot")["F1"].verdict == "FAIL"


def test_f3_is_scored_with_random_as_the_better_rule():
    """F3 passes when the interval is entirely NEGATIVE (random beats the analogue rule)."""
    rows = [_f1_row(k, -0.40, -0.10, comparison="same_chemotype_first_vs_random")
            for k in (1, 2, 3, 5)]
    verdicts = _score(rows)
    assert verdicts["F3"].verdict == "PASS"
    assert "negative** interval means random is better" in verdicts["F3"].evidence
    flipped = [_f1_row(k, 0.10, 0.40, comparison="same_chemotype_first_vs_random")
               for k in (1, 2, 3, 5)]
    assert _score(flipped)["F3"].verdict == "FAIL"


def test_f2_reads_the_offset_endpoint_and_reports_the_maxmin_comparison_beside_it():
    rows = [_f1_row(k, 0.10, 0.30, comparison="offset_uncertainty_vs_random", endpoint="all",
                    statistic="offset_mae") for k in (5, 8)]
    rows += [_f1_row(k, -0.20, -0.05, comparison="offset_uncertainty_vs_maxmin",
                     endpoint="vs_maxmin|all", statistic="offset_mae") for k in (5, 8)]
    verdict = _score(rows)["F2"]
    assert verdict.verdict == "PASS"
    assert "against **maxmin**" in verdict.evidence
    assert "k=5: -0.125" in verdict.evidence


def test_f4_is_skipped_with_a_reason_when_the_cap_is_absent():
    verdict = _score([])["F4"]
    assert verdict.verdict == "SKIPPED"
    assert verdict.evidence == "skipped in this test", "the reason must reach the report verbatim"
    # the reason a real run gives names the flag and the prior evidence, as pre-registered
    assert runner.parse_args([]).f4_cap_start_rows is None
    assert "--f4-cap-start-rows" in runner.F4_SKIP_REASON
    assert "Experiment B" in runner.F4_SKIP_REASON


def test_f4_is_scored_against_the_depth_arm_when_enabled():
    rows = [_f1_row(k, 0.10, 0.30, comparison=f"maxmin_vs_{runner.DEPTH_ARM}") for k in (1, 5, 8)]
    assert _score(rows, f4_enabled=True)["F4"].verdict == "PASS"


def test_a_missing_contrast_is_inconclusive_not_a_pass():
    verdicts = _score([])
    assert verdicts["F1"].verdict == "INCONCLUSIVE"
    assert verdicts["F2"].verdict == "INCONCLUSIVE"
    assert "no checkpoint produced this contrast" in verdicts["F1"].evidence


# --------------------------------------------------------------------------- #
# (h) checks and fail-closed completion
# --------------------------------------------------------------------------- #

def _specs(arms=("random", "maxmin")) -> list[runner.ArmSpec]:
    return [runner.ArmSpec(arm=a, policy=a, budget_per_ligand=3, replicates=1, family="primary")
            for a in arms]


def _curves(results, specs, checkpoints) -> pd.DataFrame:
    return pd.DataFrame([
        {"split_seed": r.seed, "fold": r.fold, "arm": s.arm, "policy": s.policy,
         "budget_per_ligand": "3", "family": "primary", "replicate": 0, "checkpoint": c}
        for r in results for s in specs for c in checkpoints])


def test_build_checks_passes_on_a_clean_run():
    frame, _, chemistry = _world()
    results = [_fold_result(frame, chemistry, fold=f) for f in (0, 1)]
    for result in results:                      # identical checkpoint-0 predictions
        for arm in ("random", "maxmin"):
            result.predictions[(arm, 0, 0)] = np.zeros(len(result.test_frame))
    specs = _specs()
    checks = runner.build_checks(
        results=results, integrity={"1": {"ok": True}},
        curves=_curves(results, specs, (0, 1, 2)), specs=specs,
        label_free={"ok": True}, checkpoints=(1, 2))
    assert all(entry["ok"] for entry in checks.values()), checks


def test_build_checks_flags_a_checkpoint_zero_that_moved_with_the_policy():
    frame, _, chemistry = _world()
    results = [_fold_result(frame, chemistry, fold=0)]
    results[0].predictions[("maxmin", 0, 0)] = results[0].predictions[("random", 0, 0)] + 0.01
    specs = _specs()
    checks = runner.build_checks(
        results=results, integrity={"1": {"ok": True}},
        curves=_curves(results, specs, (0, 1, 2)), specs=specs,
        label_free={"ok": True}, checkpoints=(1, 2))
    assert checks["checkpoint_zero_identical_across_arms"]["ok"] is False


def test_build_checks_flags_a_mask_that_moved():
    frame, _, chemistry = _world()
    results = [_fold_result(frame, chemistry, fold=0)]
    results[0].evaluator_audit["distinct_mask_digests"]["nn<0.4"] = 2
    specs = _specs()
    checks = runner.build_checks(
        results=results, integrity={"1": {"ok": True}},
        curves=_curves(results, specs, (0, 1, 2)), specs=specs,
        label_free={"ok": True}, checkpoints=(1, 2))
    assert checks["hard_masks_fixed_per_fold"]["ok"] is False


def test_build_checks_flags_a_revealed_test_row():
    frame, _, chemistry = _world()
    results = [_fold_result(frame, chemistry, fold=0)]
    results[0].evaluator_audit["n_test_rows_revealed"] = 4
    specs = _specs()
    checks = runner.build_checks(
        results=results, integrity={"1": {"ok": True}},
        curves=_curves(results, specs, (0, 1, 2)), specs=specs,
        label_free={"ok": True}, checkpoints=(1, 2))
    assert checks["no_test_row_revealed"]["ok"] is False
    assert checks["no_test_row_revealed"]["n_test_rows_revealed"] == 4


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
        "test_extractants": ["S12"], "test_superclusters": ["test"],
        "train_extractants_by_arm": {"POOL": ["S00"]},
        "train_superclusters_by_arm": {"POOL": ["start"]},
        "n_test_rows": 12, "n_train_rows_by_arm": {"POOL": 12},
    }])
    for name in runner.REQUIRED_ARTIFACTS:
        (output_dir / name).write_text("placeholder\n")
    return manifest


def test_no_success_marker_when_a_check_fails(tmp_path):
    manifest = _stub_manifest(tmp_path)
    validation, success = runner.finalise_run(
        tmp_path, manifest=manifest,
        checks={"no_test_row_revealed": {"ok": False, "n_test_rows_revealed": 3},
                "hard_masks_fixed_per_fold": {"ok": True}})
    assert success is None
    assert not (tmp_path / "_SUCCESS.json").exists()
    assert (tmp_path / "_FAILED.json").exists()
    assert "no_test_row_revealed" in validation["failed_checks"]


def test_success_marker_when_every_check_passes(tmp_path):
    manifest = _stub_manifest(tmp_path)
    validation, success = runner.finalise_run(
        tmp_path, manifest=manifest,
        checks={"no_test_row_revealed": {"ok": True}, "hard_masks_fixed_per_fold": {"ok": True}})
    assert validation["ok"] is True
    assert success is not None and success.exists()
    payload = json.loads((tmp_path / "_SUCCESS.json").read_text())
    assert payload["validation_ok"] is True
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text())
    assert "validation.json" in hashes["files"], \
        "artifact hashes must cover the validation record written after the manifest"


def test_a_skipped_label_free_check_cannot_validate(tmp_path):
    """Protocol §6 makes label-freeness a required control, so skipping it fails closed."""
    manifest = _stub_manifest(tmp_path)
    validation, success = runner.finalise_run(
        tmp_path, manifest=manifest,
        checks={"label_free_acquisition": {"ok": False, "status": "not run"}})
    assert success is None
    assert "label_free_acquisition" in validation["failed_checks"]


# --------------------------------------------------------------------------- #
# (i) run wiring: arms, seeds, and the pilot switch
# --------------------------------------------------------------------------- #

def test_pilot_arguments_match_the_contract():
    args = runner.parse_args(["--pilot"])
    assert args.pilot is True
    # the pilot values themselves are applied inside main(); the defaults must be
    # the pre-registered full design, or the "full run" command would be a pilot
    full = runner.parse_args([])
    assert list(full.split_seeds) == [104729, 130363, 155921]
    assert full.replicates == 2 and full.n_steps == 30 and full.budget_per_ligand == 3
    assert full.n_estimators == 200
    assert list(full.checkpoints) == [1, 2, 3, 5, 8, 12, 16, 20, 25, 30]
    assert list(full.policies) == list(runner.POLICIES)
    assert full.f4_cap_start_rows is None


def test_arm_specs_add_the_sensitivity_and_depth_arms_only_when_asked():
    args = runner.parse_args([])
    specs = runner.build_arm_specs(args)
    assert [s.arm for s in specs if s.family == "primary"] == list(runner.POLICIES)
    assert {s.arm for s in specs if s.family == "budget_all"} == {
        f"maxmin{runner.BUDGET_ALL_SUFFIX}", f"random{runner.BUDGET_ALL_SUFFIX}"}
    assert all(s.budget_per_ligand is None for s in specs if s.family == "budget_all")
    assert not [s for s in specs if s.family == "depth"]

    with_f4 = runner.build_arm_specs(runner.parse_args(["--f4-cap-start-rows", "3"]))
    assert [s.arm for s in with_f4 if s.family == "depth"] == [runner.DEPTH_ARM]

    without = runner.build_arm_specs(runner.parse_args(["--budget-all-policies"]))
    assert not [s for s in without if s.family == "budget_all"]


def test_fold_and_acquisition_seeds_are_deterministic_and_distinct():
    args = runner.parse_args([])
    a = runner.fold_parameters(args, fold=2, replicate=1)
    assert a.random_state == 42 + 2 * 1009 + 9_999_991 + 101
    assert runner.fold_parameters(args, fold=2, replicate=0).random_state != a.random_state
    seed = runner.acquisition_seed(104729, 2, "maxmin", 1)
    assert seed == runner.acquisition_seed(104729, 2, "maxmin", 1)
    assert seed != runner.acquisition_seed(104729, 2, "maxmin", 0)
    assert seed != runner.acquisition_seed(104729, 2, "random", 1)
    assert seed != runner.acquisition_seed(104729, 3, "maxmin", 1)

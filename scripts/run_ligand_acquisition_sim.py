"""gen6 Experiment F — which ligand should have been measured next?

Phase 1 answered *how much* breadth is worth (Experiment B: at equal row budget,
breadth beats depth).  Experiment F asks the operational question that follows:
starting from the narrow, deep cohort this project actually had for five
generations, **which** ligand should the next measurement have gone to, and does
the *choice* matter beyond the fact that a new ligand was bought at all?

The design replays history as a sequential decision.

* **Folds** are the Experiment A chemotype folds (``diversity_splits`` on
  ``tanimoto_cluster``).  The fold's held-out chemotypes are the test set; it is
  **fixed** and no acquisition ever touches it.  The closure that scores every
  checkpoint receives the revealed training rows and asserts, at run time, that
  not one of them is a test row — the run stops if it ever is.
* **Start cohort**: the ten most-measured training ligands of the largest
  training chemotype, all their rows.  Narrow by construction (~2,900 rows in
  four folds; the fold that holds out the diglycolamides degenerates to 2
  ligands / 244 rows — disclosed, not excluded).
* **Pool**: every other training ligand.  An acquisition reveals only ``b = 3``
  of the chosen ligand's rows (seeded) — "the first few measurements" — with
  ``b = all`` as a disclosed sensitivity for two policies.
* **Policies** (all label-free for the candidate; the pool's ``log_D`` is never
  read): ``random`` (the required null), ``maxmin``, ``uncertainty``,
  ``diversity_x_uncertainty``, ``offset_uncertainty``, ``same_chemotype_first``.

Four traps this runner is written around, each of which would silently void the
comparison rather than merely add noise:

1. **The hard-chemistry subset must not move.**  "Hard chemistry" is a test row
   whose nearest-neighbour Tanimoto to the **start cohort's** extractants is
   below 0.4 (and 0.6).  That reference is computed **once per fold**, before any
   acquisition, and never again — if it were recomputed against the current
   training set it would move with the policy and with the step, and two policies
   would be compared on different rows.  ``FoldEvaluator`` freezes the masks at
   construction and re-hashes them on every call, so a mutation would show up in
   ``validation.json``.
2. **Checkpoint 0 must be identical for every policy.**  Before the first
   acquisition every arm is the same model on the same start rows, so its test
   predictions must agree bit-for-bit.  They are compared, and a difference means
   the start cohort or the fold seed moved with the policy.
3. **A policy is a distribution over acquisition sequences, not one sequence.**
   Replicate predictions are averaged per checkpoint before the paired bootstrap,
   and the per-replicate curves are kept so the spread is visible.
4. **The bootstrap unit is not the row.**  Scoring unit = ECFP cluster;
   resampling unit = the Tanimoto-0.7 chemotype the folds actually held out.

Example::

    .venv/bin/python scripts/run_ligand_acquisition_sim.py --pilot
    .venv/bin/python scripts/run_ligand_acquisition_sim.py
    .venv/bin/python scripts/run_ligand_acquisition_sim.py --f4-cap-start-rows 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen6 import GEN6_LAYER  # noqa: E402
from lanthanide_separation.gen6.acquisition import (  # noqa: E402
    DEFAULT_CHECKPOINTS, POLICIES, AcquisitionTrace, StartCohort, run_acquisition, start_cohort,
)
from lanthanide_separation.gen6.chemistry import (  # noqa: E402
    ChemistryMap, build_chemistry_map, cluster_manifest,
)
from lanthanide_separation.gen6.cohorts import (  # noqa: E402
    BASE_ARM, BASE_MIN_CELLS, EXPANDED_ARM, EXPANDED_MIN_CELLS, assert_split_integrity,
    diversity_splits,
)
from lanthanide_separation.gen6.hierarchical import FOREST_BLOCKS  # noqa: E402
from lanthanide_separation.gen6.manifest import (  # noqa: E402
    RunManifest, sha256_frame, sha256_text, validate_run, write_success,
)
from lanthanide_separation.gen6.metrics import (  # noqa: E402
    HARD_CHEMISTRY_THRESHOLDS, UNIT_STATISTICS, decompose_level_shape, paired_unit_bootstrap,
    per_unit_statistics,
)
from lanthanide_separation.levels import (  # noqa: E402
    LEVEL_TARGET_COLUMN, LevelForestParameters, LevelRegressor, build_level_dataset,
)

DATASET_PATH = REPO_ROOT / "dataset with 3D structures" / "dataset.parquet"
DESCRIPTOR_PATH = REPO_ROOT / "dataset with 3D structures" / "ligand_2d_descriptors.parquet"

#: Three of the five gen5 split seeds (protocol §F: 3 seeds x 5 folds x 2 replicates).
DEFAULT_SEEDS: tuple[int, ...] = (104729, 130363, 155921)
#: gen5's fold seed formula, reused so the start-cohort model matches the rest of gen6.
FOLD_SEED_STRIDE = 1009
FOLD_SEED_OFFSET = 9_999_991
#: Stride between acquisition replicates inside one fold (protocol §F).
REPLICATE_SEED_STRIDE = 101
#: Salt for the acquisition RNG; keeps this experiment's draws disjoint from
#: Experiment A's row-matching stream and Experiment B's acquisition stream.
ACQUISITION_SEED_SALT = 808_080_809
#: The null every F hypothesis is measured against.
NULL_POLICY = "random"
#: The F4 arm: it spends the same rows on the start cohort's own ligands.  It is
#: not one of ``acquisition.POLICIES`` because it buys *depth*, not a new ligand,
#: so it cannot be expressed as a choice over the pool.
DEPTH_ARM = "depth_on_start"
#: Suffix for the ``budget = all rows of the acquired ligand`` sensitivity arms.
BUDGET_ALL_SUFFIX = "@ball"
#: Why F4 is off by default (protocol §F: it is a sanity check, not a new claim).
F4_SKIP_REASON = ("needs --f4-cap-start-rows; with the full start cohort there are no unrevealed "
                  "start rows to buy, and Experiment B already measured depth vs breadth")
#: Statistics carried through the paired bootstrap.  ``mae`` averaged over ECFP
#: clusters *is* the macro MAE, because ``per_unit_statistics`` defines every
#: statistic as a mean over the unit's members.
BOOTSTRAP_STATISTICS: tuple[str, ...] = tuple(UNIT_STATISTICS)

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "acquisition_curves.csv", "acquisition_steps.csv", "curve_summary.csv", "contrasts.csv",
    "start_cohorts.csv", "decision_report.md", "summary.json", "manifest.json",
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=DATASET_PATH)
    p.add_argument("--descriptors", type=Path, default=DESCRIPTOR_PATH,
                   help="gen4 extended-2D descriptor parquet (the LIG2D_EXT block); "
                        "pass '' to run without it (the champion feature set then fails closed)")
    p.add_argument("--chemistry-map", type=Path, default=None,
                   help="frozen chemistry_map.parquet from scripts/build_chemistry_map.py; "
                        "rebuilt from the source table when omitted (identical by construction)")
    p.add_argument("--eval-min-cells", type=int, default=EXPANDED_MIN_CELLS,
                   help="eligibility of the ONE shared cohort (3 = the gen6 EXPANDED rule)")
    p.add_argument("--base-min-cells", type=int, default=BASE_MIN_CELLS,
                   help="the gen5 rule; recorded per fold, not used to select anything here")
    p.add_argument("--split-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-indices", nargs="+", type=int, default=None,
                   help="run only these fold indices (default: all of them)")
    p.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    p.add_argument("--start-ligands", type=int, default=10,
                   help="size of the narrow start cohort (the most-measured ligands of the "
                        "largest training chemotype)")
    p.add_argument("--budget-per-ligand", type=int, default=3,
                   help="rows revealed per acquisition — the 'first few measurements' scenario")
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--checkpoints", nargs="+", type=int, default=list(DEFAULT_CHECKPOINTS))
    p.add_argument("--replicates", type=int, default=2,
                   help="acquisition replicates per (seed, fold, policy)")
    p.add_argument("--budget-all-policies", nargs="*", default=["maxmin", "random"],
                   help="disclosed sensitivity: these policies are re-run with budget = every row "
                        "of the acquired ligand; pass nothing to skip it")
    p.add_argument("--budget-all-replicates", type=int, default=1,
                   help="replicates for the budget=all sensitivity (fewer is fine; it is not a "
                        "protected claim)")
    p.add_argument("--f4-cap-start-rows", type=int, default=None,
                   help="cap every start ligand to N rows so that unrevealed start rows exist to "
                        "'buy'; enables the F4 depth_on_start arm. The cap applies to the whole "
                        "run, so every policy shares the same (capped) start cohort")
    p.add_argument("--model-seed", type=int, default=42)
    p.add_argument("--weighting", choices=["cluster", "row"], default="cluster",
                   help="cluster = gen5 equal-weight-per-ECFP-cluster rule (primary); row = no sample "
                        "weights. Under 'cluster' the ~90 acquired rows carry ~83 %% of the training "
                        "weight after 30 steps, so 'row' is the required sensitivity.")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-features", type=float, default=0.30)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--thresholds", nargs="+", type=float, default=list(HARD_CHEMISTRY_THRESHOLDS),
                   help="hard-chemistry nearest-neighbour Tanimoto cut-offs to the START cohort")
    p.add_argument("--replicate-policy", default="mean", choices=["mean", "unique", "all"])
    p.add_argument("--log-d-floor", type=float, default=-6.0)
    p.add_argument("--bootstrap-replicates", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=8675309)
    p.add_argument("--label-free-check", dest="label_free_check", action="store_true", default=True,
                   help="run-time proof on the real cohort that permuting the pool's log_D moves "
                        "no policy's first pick (default on)")
    p.add_argument("--no-label-free-check", dest="label_free_check", action="store_false",
                   help="skip that proof — the run then FAILS validation on purpose (protocol §6 "
                        "makes label-freeness a required control, so a run without it may not "
                        "carry a _SUCCESS marker)")
    p.add_argument("--provenance-state-json", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--pilot", action="store_true",
                   help="smoke test: one seed, folds 0 and 2, three policies, 8 steps, "
                        "checkpoints 1/2/3/5/8, one replicate, 80 trees")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Chemistry helper
# --------------------------------------------------------------------------- #

def max_similarity_to_reference(
    chemistry: ChemistryMap, query: Sequence[str], reference: Sequence[str],
) -> np.ndarray:
    """Max Tanimoto from each query extractant to any reference extractant.

    Numerically identical to ``ChemistryMap.nearest_neighbour(query,
    reference, exclude_self=False)["nn_tanimoto"]`` — asserted in
    ``tests/test_gen6_acquisition_runner.py`` so the two cannot drift — but
    vectorised, because the public method resolves every name through
    ``tuple.index`` inside a per-query Python loop.

    ``exclude_self`` is deliberately **not** offered: a test ligand that also sat
    in the start cohort would be a fold-integrity failure, and masking it here
    would hide it behind a plausible number instead of showing it as a similarity
    of 1.0.  An empty reference gives 0.0, never NaN — "nothing similar in the
    start cohort" is a legitimate state and the hardest ligands must not vanish
    into a NaN filter.
    """
    position = {name: i for i, name in enumerate(chemistry.extractants)}
    q_names = [str(x) for x in query]
    r_names = [str(x) for x in reference]
    if not q_names:
        return np.zeros(0, dtype=float)
    missing = [n for n in set(q_names) | set(r_names) if n not in position]
    if missing:
        raise KeyError(f"{len(missing)} extractants are absent from the frozen chemistry map, "
                       f"e.g. {missing[:2]}; map and cohort have drifted")
    if not r_names:
        return np.zeros(len(q_names), dtype=float)
    block = chemistry.similarity[np.ix_([position[n] for n in q_names],
                                        [position[n] for n in r_names])].astype(float)
    return block.max(axis=1)


def _macro_over(values: np.ndarray, keys: np.ndarray) -> float:
    """Mean over groups of the group means — one chemistry unit, one vote."""
    if len(values) == 0:
        return float("nan")
    return float(pd.Series(values).groupby(pd.Series(keys)).mean().mean())


# --------------------------------------------------------------------------- #
# The metric closure: fixed test rows, fixed hard masks
# --------------------------------------------------------------------------- #

class FoldEvaluator:
    """Scores one fold's **fixed** test rows at every checkpoint of every policy.

    Everything that could move with the policy or with the step is frozen in the
    constructor:

    * the test rows themselves (``test_index``);
    * ``nn_to_start`` — each test row's maximum Tanimoto to the **start cohort's**
      extractants — and the boolean hard-chemistry masks cut from it.

    The masks are re-hashed on every call and the digests are collected, so
    "the mask never moved" is a measurement written into ``validation.json``
    rather than a claim in a docstring.  Every call also intersects the revealed
    training rows with the test rows and raises on any overlap: an acquisition
    that reaches into the test set would otherwise look like a very good policy.
    """

    def __init__(
        self, frame: pd.DataFrame, *, chemistry: ChemistryMap, test_index: np.ndarray,
        start: StartCohort, thresholds: Sequence[float],
    ) -> None:
        self.test_index = np.asarray(test_index, dtype=int)
        self.thresholds = tuple(float(t) for t in thresholds)
        self.start_extractants = tuple(start.extractants)
        self.frame = frame
        self.chemistry = chemistry

        self.y = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)[self.test_index]
        self.ligand = frame["extractant"].astype(str).to_numpy()[self.test_index]
        self.cluster = frame["ecfp_cluster"].astype(str).to_numpy()[self.test_index]
        self.chemotype = frame["tanimoto_cluster"].astype(str).to_numpy()[self.test_index]
        self.test_ligands = tuple(pd.unique(self.ligand))
        self._extractant_of_row = frame["extractant"].astype(str).to_numpy()
        self._chemotype_of_row = frame["tanimoto_cluster"].astype(str).to_numpy()
        self._cluster_of_row = frame["ecfp_cluster"].astype(str).to_numpy()

        # --- the frozen reference: distance to the START cohort, once ---------
        self.nn_by_ligand = max_similarity_to_reference(
            chemistry, self.test_ligands, self.start_extractants)
        lookup = dict(zip(self.test_ligands, self.nn_by_ligand))
        self.nn_to_start = np.array([lookup[name] for name in self.ligand], dtype=float)
        self.masks: dict[float, np.ndarray] = {
            t: (self.nn_to_start < t) for t in self.thresholds}

        self._test_row_set = set(self.test_index.tolist())
        self.n_calls = 0
        self.n_test_rows_revealed = 0
        self.mask_digests: dict[float, set[str]] = {t: set() for t in self.thresholds}

    # -- run-time invariants ------------------------------------------------- #
    def _digest_masks(self) -> None:
        for threshold, mask in self.masks.items():
            self.mask_digests[threshold].add(
                sha256_text("".join("1" if bool(v) else "0" for v in mask)))

    def mask_audit(self) -> dict:
        return {
            "n_calls": int(self.n_calls),
            "n_test_rows": int(len(self.test_index)),
            "n_start_extractants": len(self.start_extractants),
            "distinct_mask_digests": {f"nn<{t:g}": len(d) for t, d in self.mask_digests.items()},
            "n_hard_rows": {f"nn<{t:g}": int(m.sum()) for t, m in self.masks.items()},
            "n_hard_ligands": {f"nn<{t:g}": int(pd.unique(self.ligand[m]).size)
                               for t, m in self.masks.items()},
            "n_test_rows_revealed": int(self.n_test_rows_revealed),
        }

    # -- the closure --------------------------------------------------------- #
    def __call__(self, prediction: np.ndarray, train_rows: np.ndarray,
                 acquired: np.ndarray) -> dict:
        self.n_calls += 1
        self._digest_masks()
        rows = np.asarray(train_rows, dtype=int)
        overlap = np.intersect1d(rows, self.test_index)
        if overlap.size:
            self.n_test_rows_revealed += int(overlap.size)
            raise RuntimeError(
                f"{overlap.size} revealed training rows are test rows of this fold "
                f"(first positions {overlap[:5].tolist()}); the acquisition leaked into the "
                f"fixed test set and the run must stop")

        prediction = np.asarray(prediction, dtype=float)
        error = np.abs(prediction - self.y)
        decomposition = decompose_level_shape(self.y, prediction, self.ligand)
        record: dict = {
            "macro_mae": _macro_over(error, self.cluster),
            "macro_mae_ligand": decomposition.summary["macro_mae_ligand"],
            "pooled_mae": decomposition.summary["pooled_mae"],
            "offset_mae": decomposition.summary["offset_mae"],
            "shape_mae": decomposition.summary["shape_mae"],
            "shape_r2": decomposition.summary["shape_r2"],
            "median_ligand_mae": decomposition.summary["median_ligand_mae"],
            "worst_quartile_ligand_mae": decomposition.summary["worst_quartile_ligand_mae"],
            "frac_within_1_log": decomposition.summary["frac_within_1_log"],
            "n_test_rows": int(len(self.y)),
            "n_test_ligands": int(len(self.test_ligands)),
            "n_test_rows_revealed": 0,
        }

        # --- hard chemistry, on the FIXED masks -------------------------------
        for threshold, mask in self.masks.items():
            tag = f"hard_nn{threshold:g}"
            n_rows = int(mask.sum())
            record[f"{tag}__n_rows"] = n_rows
            record[f"{tag}__n_ligands"] = int(pd.unique(self.ligand[mask]).size) if n_rows else 0
            if n_rows == 0:
                for key in ("macro_mae", "offset_mae", "shape_mae", "macro_mae_ligand"):
                    record[f"{tag}__{key}"] = np.nan
                continue
            hard = decompose_level_shape(self.y[mask], prediction[mask], self.ligand[mask])
            record[f"{tag}__macro_mae"] = _macro_over(error[mask], self.cluster[mask])
            record[f"{tag}__offset_mae"] = hard.summary["offset_mae"]
            record[f"{tag}__shape_mae"] = hard.summary["shape_mae"]
            record[f"{tag}__macro_mae_ligand"] = hard.summary["macro_mae_ligand"]

        # --- coverage: what the training set now holds -------------------------
        train_ligands = pd.unique(self._extractant_of_row[rows])
        nn_now = max_similarity_to_reference(self.chemistry, self.test_ligands, list(train_ligands))
        record.update({
            "n_train_rows": int(len(rows)),
            "n_train_ligands": int(len(train_ligands)),
            "n_train_chemotypes": int(pd.unique(self._chemotype_of_row[rows]).size),
            "n_train_ecfp_clusters": int(pd.unique(self._cluster_of_row[rows]).size),
            "n_acquired_ligands": int(len(np.asarray(acquired))),
            "mean_nn_test_to_train": float(np.mean(nn_now)),
            "median_nn_test_to_train": float(np.median(nn_now)),
            "frac_test_ligands_nn_below_0_4": float(np.mean(nn_now < 0.4)),
            "mean_nn_test_to_start": float(np.mean(self.nn_by_ligand)),
        })
        return record


# --------------------------------------------------------------------------- #
# Start cohort capping (F4) and the depth arm
# --------------------------------------------------------------------------- #

def cap_start_cohort(
    frame: pd.DataFrame, start: StartCohort, *, cap: int, rng: np.random.Generator,
) -> tuple[StartCohort, dict[str, np.ndarray]]:
    """Keep at most ``cap`` rows of each start ligand; return the withheld rows.

    F4 asks whether three measurements of a *distant* ligand beat three more of a
    *known* one.  With the full start cohort there is nothing left to buy on the
    known ligands, so the comparison cannot be run at all; capping the start makes
    the depth arm possible.  The cap then applies to **every** policy, so all arms
    still share one start cohort and the curves remain comparable — it changes the
    experiment (a smaller, shallower start), which is exactly why F4 is off by
    default.
    """
    extractant = frame["extractant"].astype(str).to_numpy()
    kept: list[np.ndarray] = []
    withheld: dict[str, np.ndarray] = {}
    for name in start.extractants:
        rows = np.sort(start.row_index[extractant[start.row_index] == str(name)])
        if len(rows) <= cap:
            kept.append(rows)
            withheld[str(name)] = np.empty(0, dtype=int)
            continue
        chosen = np.sort(rng.choice(rows, size=int(cap), replace=False))
        kept.append(chosen)
        withheld[str(name)] = np.sort(np.setdiff1d(rows, chosen))
    row_index = np.sort(np.concatenate(kept)) if kept else np.empty(0, dtype=int)
    audit = {
        **start.audit,
        "start_row_cap": int(cap),
        "n_start_rows_uncapped": int(len(start.row_index)),
        "n_start_rows": int(len(row_index)),
        "n_start_rows_withheld": int(sum(len(v) for v in withheld.values())),
    }
    capped = StartCohort(extractants=start.extractants, chemotype=start.chemotype,
                         row_index=row_index, pool_extractants=start.pool_extractants,
                         audit=audit)
    return capped, withheld


def run_depth_on_start(
    frame: pd.DataFrame,
    *,
    start: StartCohort,
    withheld: Mapping[str, np.ndarray],
    test_index: np.ndarray,
    feature_columns: Sequence[str],
    params: LevelForestParameters,
    budget_per_ligand: int | None,
    n_steps: int,
    checkpoints: Sequence[int],
    seed: int,
    evaluate: Callable[[np.ndarray, np.ndarray, np.ndarray], dict],
) -> AcquisitionTrace:
    """The F4 comparator: spend the same rows on the start cohort's own ligands.

    Round-robin over the start ligands in a seeded order, revealing
    ``budget_per_ligand`` of each one's withheld rows per step, so at every
    checkpoint this arm has bought **the same number of rows** as a ``b = 3``
    acquisition policy — only they are more measurements of ligands already held
    rather than the first measurements of new ones.  Refits at checkpoints only
    (no policy here needs the model to choose), exactly like the chemistry-only
    policies in :func:`acquisition.run_acquisition`.

    Returns an :class:`acquisition.AcquisitionTrace` so it flows through the same
    aggregation as every other arm.
    """
    rng = np.random.default_rng(seed)
    y = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    test = frame.iloc[test_index]
    remaining = {str(k): np.asarray(v, dtype=int).copy() for k, v in withheld.items()}
    order = [str(name) for name in start.extractants if len(remaining.get(str(name), ()))]
    order = [order[i] for i in rng.permutation(len(order))]
    revealed = np.array(start.row_index, dtype=int)
    checkpoint_set = {int(c) for c in checkpoints}
    step_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    predictions: dict[int, np.ndarray] = {}

    def fit_current() -> LevelRegressor:
        train = frame.iloc[revealed]
        return LevelRegressor(feature_columns, params).fit(
            train, y[revealed], groups=train["ecfp_cluster"])

    def record(step: int, model: LevelRegressor) -> None:
        prediction = model.predict(test)
        predictions[step] = prediction
        entry = {"step": step, "n_train_rows": int(len(revealed)), "n_acquired": step,
                 "n_train_ligands": int(pd.unique(
                     frame["extractant"].astype(str).to_numpy()[revealed]).size)}
        entry.update(evaluate(prediction, revealed, np.array(start.extractants)))
        checkpoint_rows.append(entry)

    model = fit_current()
    record(0, model)
    cursor = 0
    for step in range(1, int(n_steps) + 1):
        live = [name for name in order if len(remaining[name])]
        if not live:
            break
        name = live[cursor % len(live)]
        cursor += 1
        pool_rows = remaining[name]
        take = len(pool_rows) if budget_per_ligand is None else min(int(budget_per_ligand),
                                                                    len(pool_rows))
        chosen = np.sort(rng.choice(pool_rows, size=take, replace=False))
        remaining[name] = np.sort(np.setdiff1d(pool_rows, chosen))
        revealed = np.concatenate([revealed, chosen])
        step_rows.append({
            "step": step, "extractant": name, "n_rows_revealed": int(len(chosen)),
            "n_train_rows": int(len(revealed)),
            # the ligand is already in training, so its distance to the acquired
            # set is 1.0 by definition; recorded rather than left blank so the
            # column means the same thing in every arm
            "nn_to_acquired_before": 1.0,
            "chemotype": str(frame["tanimoto_cluster"].iloc[chosen[0]]) if len(chosen) else "",
        })
        if step in checkpoint_set:
            model = fit_current()
            record(step, model)

    return AcquisitionTrace(
        policy=DEPTH_ARM, steps=pd.DataFrame(step_rows), checkpoints=pd.DataFrame(checkpoint_rows),
        predictions=predictions,
        audit={"budget_per_ligand": budget_per_ligand, "n_steps_requested": int(n_steps),
               "n_steps_run": len(step_rows), "pool_exhausted": not any(
                   len(v) for v in remaining.values()), "seed": int(seed), **start.audit})


# --------------------------------------------------------------------------- #
# Run specification
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ArmSpec:
    """One arm of the simulation: a policy, a per-ligand budget and a replicate count."""

    arm: str
    policy: str
    budget_per_ligand: int | None
    replicates: int
    family: str          # "primary" | "budget_all" | "depth"

    @property
    def budget_label(self) -> str:
        return "all" if self.budget_per_ligand is None else str(self.budget_per_ligand)


def build_arm_specs(args: argparse.Namespace) -> list[ArmSpec]:
    """Primary arms, then the disclosed budget=all sensitivity, then F4's depth arm."""
    specs = [ArmSpec(arm=p, policy=p, budget_per_ligand=args.budget_per_ligand,
                     replicates=int(args.replicates), family="primary")
             for p in args.policies]
    for policy in (args.budget_all_policies or []):
        if policy not in args.policies:
            continue
        specs.append(ArmSpec(arm=f"{policy}{BUDGET_ALL_SUFFIX}", policy=policy,
                             budget_per_ligand=None,
                             replicates=max(1, int(args.budget_all_replicates)),
                             family="budget_all"))
    if args.f4_cap_start_rows is not None:
        specs.append(ArmSpec(arm=DEPTH_ARM, policy=DEPTH_ARM,
                             budget_per_ligand=args.budget_per_ligand,
                             replicates=int(args.replicates), family="depth"))
    return specs


def acquisition_seed(split_seed: int, fold: int, arm: str, replicate: int) -> int:
    """Deterministic acquisition RNG seed for one (seed, fold, arm, replicate).

    The arm is part of the key on purpose: the policies consume the generator at
    different rates (``score_random`` draws one uniform per candidate), so their
    streams diverge after the first call anyway, and a shared seed would only
    create the appearance of common random numbers.
    """
    digest = sha256_text(f"{int(split_seed)}|{int(fold)}|{arm}|{int(replicate)}|"
                         f"{ACQUISITION_SEED_SALT}")
    return int(digest[:15], 16)


def fold_parameters(args: argparse.Namespace, fold: int, replicate: int) -> LevelForestParameters:
    """gen5's fold seed plus the protocol's per-replicate stride."""
    return LevelForestParameters(
        n_estimators=int(args.n_estimators), max_features=float(args.max_features),
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=int(args.model_seed) + fold * FOLD_SEED_STRIDE + FOLD_SEED_OFFSET
        + REPLICATE_SEED_STRIDE * int(replicate),
        n_jobs=int(args.n_jobs))


# --------------------------------------------------------------------------- #
# One (seed, fold) job
# --------------------------------------------------------------------------- #

@dataclass
class FoldResult:
    """Everything one (split seed, fold) produced."""

    seed: int
    fold: int
    start_audit: dict
    start_extractants: tuple[str, ...] = ()
    curves: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    #: (arm, replicate, checkpoint) -> test predictions, kept per replicate so the
    #: checkpoint-0 identity check can compare like with like before averaging.
    predictions: dict[tuple[str, int, int], np.ndarray] = field(default_factory=dict)
    test_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    evaluator_audit: dict = field(default_factory=dict)
    trace_audit: list[dict] = field(default_factory=list)
    seconds: float = 0.0

    def mean_predictions(self, arm: str, checkpoint: int) -> np.ndarray | None:
        """Replicate-mean prediction vector — the unit the paired bootstrap uses."""
        parts = [v for (a, _r, c), v in self.predictions.items() if a == arm and c == checkpoint]
        if not parts:
            return None
        return np.mean(np.vstack(parts), axis=0)

    def checkpoints_for(self, arm: str) -> set[int]:
        return {c for (a, _r, c) in self.predictions if a == arm}


def run_fold(
    frame: pd.DataFrame,
    data,
    *,
    args: argparse.Namespace,
    chemistry: ChemistryMap,
    feature_columns: Sequence[str],
    specs: Sequence[ArmSpec],
    seed: int,
    split,
    checkpoints: Sequence[int],
    log: Callable[[str], None],
) -> FoldResult:
    """Simulate every arm on one chemotype fold, scoring the fixed test rows."""
    started = time.time()
    fold = int(split.fold)
    train_index = split.train_index_by_arm[EXPANDED_ARM]
    test_index = split.test_index
    start = start_cohort(frame, train_index, n_ligands=int(args.start_ligands))
    withheld: dict[str, np.ndarray] = {}
    if args.f4_cap_start_rows is not None:
        start, withheld = cap_start_cohort(
            frame, start, cap=int(args.f4_cap_start_rows),
            rng=np.random.default_rng(acquisition_seed(seed, fold, "cap", 0)))

    evaluator = FoldEvaluator(frame, chemistry=chemistry, test_index=test_index, start=start,
                              thresholds=args.thresholds)
    result = FoldResult(seed=int(seed), fold=fold, start_audit=dict(start.audit),
                        start_extractants=tuple(str(e) for e in start.extractants))
    log(f"  seed {seed} fold {fold}: test {len(test_index)} rows / "
        f"{len(evaluator.test_ligands)} ligands; start {start.audit['n_start_ligands']} ligands / "
        f"{start.audit['n_start_rows']} rows of chemotype {start.chemotype}; pool "
        f"{start.audit['n_pool_ligands']} ligands / {start.audit['n_pool_rows']} rows; "
        f"hard rows {evaluator.mask_audit()['n_hard_rows']}")

    for spec in specs:
        for replicate in range(spec.replicates):
            params = fold_parameters(args, fold, replicate)
            seed_here = acquisition_seed(seed, fold, spec.arm, replicate)
            if spec.family == "depth":
                trace = run_depth_on_start(
                    frame, start=start, withheld=withheld, test_index=test_index,
                    feature_columns=feature_columns, params=params,
                    budget_per_ligand=spec.budget_per_ligand, n_steps=int(args.n_steps),
                    checkpoints=checkpoints, seed=seed_here, evaluate=evaluator)
            else:
                trace = run_acquisition(
                    frame, data, start=start, train_index=train_index, test_index=test_index,
                    policy=spec.policy, feature_columns=feature_columns, params=params,
                    chemistry=chemistry, budget_per_ligand=spec.budget_per_ligand,
                    n_steps=int(args.n_steps), checkpoints=checkpoints, seed=seed_here,
                    evaluate=evaluator, group_weighting=(args.weighting == "cluster"))

            tags = {"split_seed": int(seed), "fold": fold, "arm": spec.arm,
                    "policy": spec.policy, "budget_per_ligand": spec.budget_label,
                    "family": spec.family, "replicate": int(replicate)}
            for record in trace.checkpoints.to_dict("records"):
                result.curves.append({**tags, "checkpoint": int(record["step"]), **record})
            for record in trace.steps.to_dict("records"):
                result.steps.append({**tags, **record})
            for checkpoint, prediction in trace.predictions.items():
                result.predictions[(spec.arm, int(replicate), int(checkpoint))] = np.asarray(
                    prediction, dtype=float)
            result.trace_audit.append({**tags, **trace.audit})

    result.test_frame = pd.DataFrame({
        "split_seed": int(seed),
        "fold": fold,
        "row_id": frame["row_id"].astype(str).to_numpy()[test_index],
        "extractant": evaluator.ligand,
        "ecfp_cluster": evaluator.cluster,
        "tanimoto_cluster": evaluator.chemotype,
        LEVEL_TARGET_COLUMN: evaluator.y,
        "nn_to_start_tanimoto": evaluator.nn_to_start,
        **{f"hard_nn{t:g}": evaluator.masks[t] for t in evaluator.masks},
    })
    result.evaluator_audit = evaluator.mask_audit()
    result.seconds = time.time() - started
    return result


# --------------------------------------------------------------------------- #
# Label-free acquisition, checked on the real cohort
# --------------------------------------------------------------------------- #

def label_free_first_pick(
    frame: pd.DataFrame, data, *, args: argparse.Namespace, chemistry: ChemistryMap,
    feature_columns: Sequence[str], policies: Sequence[str], seed: int, split,
    permutation_seed: int = 20260819,
) -> dict:
    """Permute the pool's ``log_D`` and prove no policy's first pick moves.

    The model that a model-based policy uses to choose step 1 is fitted on the
    **start** rows, which the permutation does not touch, so a policy that reads
    the candidate's label — and only such a policy — would move.  This is the
    run-time companion to the unit test: the property is asserted on the real
    cohort inside the run that publishes the numbers.

    Only the first pick is checked, on purpose: from step 2 the revealed rows
    include a pool ligand whose target *was* permuted, so a model-based policy is
    then legitimately allowed to differ.
    """
    train_index = split.train_index_by_arm[EXPANDED_ARM]
    test_index = split.test_index
    start = start_cohort(frame, train_index, n_ligands=int(args.start_ligands))
    pool_rows = train_index[~np.isin(frame["extractant"].astype(str).to_numpy()[train_index],
                                     np.array(start.extractants, dtype=object))]
    permuted = frame.copy()
    rng = np.random.default_rng(permutation_seed)
    # pandas 3 hands out read-only views under copy-on-write, so take a real copy
    # before permuting: mutating the view raises rather than silently aliasing.
    values = np.array(permuted[LEVEL_TARGET_COLUMN].to_numpy(dtype=float), copy=True)
    values[pool_rows] = values[pool_rows][rng.permutation(len(pool_rows))]
    permuted[LEVEL_TARGET_COLUMN] = values

    moved: list[str] = []
    picks: dict[str, str] = {}
    for policy in policies:
        kwargs = dict(start=start, train_index=train_index, test_index=test_index, policy=policy,
                      feature_columns=feature_columns,
                      params=fold_parameters(args, int(split.fold), 0), chemistry=chemistry,
                      budget_per_ligand=args.budget_per_ligand, n_steps=1, checkpoints=(),
                      seed=acquisition_seed(seed, int(split.fold), policy, 0), evaluate=None)
        honest = run_acquisition(frame, data, **kwargs)
        scrambled = run_acquisition(permuted, data, **kwargs)
        a = honest.steps["extractant"].tolist()
        b = scrambled.steps["extractant"].tolist()
        picks[policy] = a[0] if a else ""
        if a != b:
            moved.append(policy)
    return {"ok": not moved, "policies_checked": list(policies), "policies_that_moved": moved,
            "first_pick": picks, "split_seed": int(seed), "fold": int(split.fold),
            "n_pool_rows_permuted": int(len(pool_rows)),
            "note": "the pool's log_D is permuted among pool rows; the start rows (and hence the "
                    "model that chooses step 1) are untouched, so any movement is a label leak"}


# --------------------------------------------------------------------------- #
# Aggregation and the paired bootstrap
# --------------------------------------------------------------------------- #

def checkpoint_frame(
    results: Sequence[FoldResult], checkpoint: int, arms: Sequence[str],
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Test rows of every fold with one replicate-mean ``prediction_<arm>`` column.

    A (seed, fold) that lacks the checkpoint for *any* arm is dropped whole and
    reported, never partially filled: a bootstrap comparing arm A on five folds
    against arm B on four would be a different comparison wearing the same name.
    """
    parts: list[pd.DataFrame] = []
    dropped: list[tuple[int, int]] = []
    for result in results:
        if any(checkpoint not in result.checkpoints_for(arm) for arm in arms):
            dropped.append((result.seed, result.fold))
            continue
        block = result.test_frame.copy()
        for arm in arms:
            block[f"prediction_{arm}"] = result.mean_predictions(arm, checkpoint)
        parts.append(block)
    if not parts:
        return pd.DataFrame(), dropped
    return pd.concat(parts, ignore_index=True), dropped


def contrast_table(
    results: Sequence[FoldResult],
    *,
    arms: Sequence[str],
    reference: str,
    checkpoints: Sequence[int],
    thresholds: Sequence[float],
    replicates: int,
    seed: int,
    log: Callable[[str], None],
    endpoint_prefix: str = "",
) -> pd.DataFrame:
    """Paired bootstrap of ``reference − candidate`` at every checkpoint.

    Scoring unit = ECFP cluster; resampling unit = the Tanimoto-0.7 chemotype the
    folds actually held out.  The hard endpoints restrict the rows to the fold's
    **fixed** start-cohort mask *before* ``per_unit_statistics``, so the units
    themselves are the ones that survive the mask.
    """
    candidates = [a for a in arms if a != reference]
    if reference not in arms or not candidates:
        return pd.DataFrame()
    comparisons = {f"{candidate}_vs_{reference}": (reference, candidate) for candidate in candidates}
    rows: list[pd.DataFrame] = []
    for checkpoint in checkpoints:
        frame, dropped = checkpoint_frame(results, int(checkpoint), arms)
        if frame.empty:
            log(f"  checkpoint {checkpoint}: no fold has every arm — contrast skipped")
            continue
        if dropped:
            log(f"  checkpoint {checkpoint}: dropped (seed, fold) {dropped} — an arm stopped early")
        endpoints: list[tuple[str, pd.DataFrame]] = [("all", frame)]
        for threshold in thresholds:
            column = f"hard_nn{threshold:g}"
            endpoints.append((f"hard_nn<{threshold:g}", frame[frame[column].astype(bool)]))
        for endpoint, subset in endpoints:
            if subset["ecfp_cluster"].nunique() < 2:
                log(f"  checkpoint {checkpoint} endpoint {endpoint}: "
                    f"{subset['ecfp_cluster'].nunique()} scoring unit(s) — bootstrap skipped, "
                    f"not silently pooled")
                continue
            per_unit = per_unit_statistics(subset, arms, unit_column="ecfp_cluster")
            blocks = dict(zip(subset["ecfp_cluster"].astype(str),
                              subset["tanimoto_cluster"].astype(str)))
            table = paired_unit_bootstrap(
                per_unit, comparisons, statistics=BOOTSTRAP_STATISTICS, block_of_unit=blocks,
                replicates=replicates, seed=seed)
            if table.empty:
                continue
            table.insert(0, "n_rows", int(len(subset)))
            table.insert(0, "endpoint", f"{endpoint_prefix}{endpoint}")
            table.insert(0, "checkpoint", int(checkpoint))
            table.insert(0, "n_folds", int(subset.groupby(["split_seed", "fold"]).ngroups))
            rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def curve_summary_table(curves: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Policy x checkpoint mean and sd over (seed, fold, replicate)."""
    present = [m for m in metrics if m in curves.columns]
    grouped = curves.groupby(["arm", "policy", "budget_per_ligand", "family", "checkpoint"],
                             as_index=False, sort=True)
    mean = grouped[present].mean().rename(columns={m: f"{m}_mean" for m in present})
    sd = grouped[present].std().rename(columns={m: f"{m}_sd" for m in present})
    counts = grouped.size().rename(columns={"size": "n_runs"})
    keys = ["arm", "policy", "budget_per_ligand", "family", "checkpoint"]
    return mean.merge(sd, on=keys).merge(counts, on=keys)


# --------------------------------------------------------------------------- #
# Hypothesis scoring
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Verdict:
    hypothesis: str
    statement: str
    pass_condition: str
    verdict: str
    evidence: str
    falsifier: str


def _rows_for(contrasts: pd.DataFrame, *, endpoint: str, comparison: str, statistic: str,
              min_checkpoint: int) -> pd.DataFrame:
    if contrasts.empty:
        return contrasts
    sub = contrasts[(contrasts["endpoint"] == endpoint)
                    & (contrasts["comparison"] == comparison)
                    & (contrasts["statistic"] == statistic)
                    & (contrasts["checkpoint"] >= min_checkpoint)]
    return sub.sort_values("checkpoint")


def _majority(rows: pd.DataFrame, *, direction: str) -> dict:
    """Score the "at >= half the checkpoints" rule in one direction.

    ``direction = "+"`` — the candidate must beat the reference (``ci95_low > 0``);
    ``direction = "-"`` — the *reference* must beat the candidate
    (``ci95_high < 0``).  The opposite condition holding at >= half the
    checkpoints is scored as evidence *against*, i.e. FAIL; an interval that
    merely straddles zero is INCONCLUSIVE, never a FAIL.
    """
    n = int(len(rows))
    if n == 0:
        return {"n_checkpoints": 0, "n_meeting": 0, "n_against": 0, "n_bca_meeting": 0,
                "passes": False, "contradicted": False, "evaluable": False, "checkpoints": [],
                "meeting_checkpoints": []}
    low = rows["ci95_low"].to_numpy(dtype=float)
    high = rows["ci95_high"].to_numpy(dtype=float)
    bca_low = (rows["bca_low"].to_numpy(dtype=float) if "bca_low" in rows.columns
               else np.full(n, np.nan))
    bca_high = (rows["bca_high"].to_numpy(dtype=float) if "bca_high" in rows.columns
                else np.full(n, np.nan))
    if direction == "+":
        meets, against, bca_meets = low > 0, high < 0, bca_low > 0
    else:
        meets, against, bca_meets = high < 0, low > 0, bca_high < 0
    need = math.ceil(n / 2)
    return {
        "n_checkpoints": n,
        "n_meeting": int(meets.sum()),
        "n_against": int(against.sum()),
        "n_bca_meeting": int(np.nansum(bca_meets)),
        "need": int(need),
        "passes": bool(meets.sum() >= need),
        "contradicted": bool(against.sum() >= need),
        "evaluable": True,
        "checkpoints": [int(c) for c in rows["checkpoint"]],
        "meeting_checkpoints": [int(c) for c, m in zip(rows["checkpoint"], meets) if m],
    }


def _verdict(score: Mapping[str, object], *, replication_ok: bool = True) -> str:
    """PASS only on the pre-registered condition **and** the pre-registered replication.

    A confidence interval that straddles zero is not evidence that the effect is
    absent, so it scores INCONCLUSIVE rather than FAIL; FAIL is reserved for an
    interval that excludes the predicted direction at ≥ half the scored
    checkpoints.  A run that did not carry the pre-registered replication (3 split
    seeds x 2 acquisition replicates — a pilot does not) cannot PASS at all, for
    the same reason Experiment A refuses to PASS on a single seed: one partition
    of one cohort is not the experiment that was registered.  Evidence *against*
    still counts as FAIL, because a contradiction does not become weaker for
    having been found early.
    """
    if not score.get("evaluable"):
        return "INCONCLUSIVE"
    if score.get("contradicted"):
        return "FAIL"
    if score.get("passes"):
        return "PASS" if replication_ok else "INCONCLUSIVE (condition met, replication short)"
    return "INCONCLUSIVE"


def _series(rows: pd.DataFrame, *, digits: int = 3) -> str:
    if rows.empty:
        return "no checkpoint produced this contrast"
    parts = []
    for _, row in rows.iterrows():
        bca = ""
        if "bca_low" in rows.columns and np.isfinite(row.get("bca_low", np.nan)):
            bca = f", BCa [{row['bca_low']:+.{digits}f}, {row['bca_high']:+.{digits}f}]"
        parts.append(f"k={int(row['checkpoint'])}: {row['point_delta']:+.{digits}f} "
                     f"[{row['ci95_low']:+.{digits}f}, {row['ci95_high']:+.{digits}f}]{bca}")
    return "; ".join(parts)


def score_hypotheses(
    contrasts: pd.DataFrame, *, thresholds: Sequence[float], f4_enabled: bool, f4_reason: str,
    min_checkpoint_f1: int = 5, replication_ok: bool = True, replication_note: str = "",
) -> list[Verdict]:
    """Score F1–F4 against the pre-registered conditions of protocol §F."""
    hard = f"hard_nn<{min(thresholds):g}" if len(thresholds) else "all"
    short = f" {replication_note}" if replication_note and not replication_ok else ""
    verdicts: list[Verdict] = []

    # --- F1 ---------------------------------------------------------------- #
    rows = _rows_for(contrasts, endpoint=hard, comparison=f"maxmin_vs_{NULL_POLICY}",
                     statistic="mae", min_checkpoint=min_checkpoint_f1)
    score = _majority(rows, direction="+")
    verdicts.append(Verdict(
        "F1", "Max-min diversity beats random acquisition on hard chemistry.",
        f"hard macro MAE(random) − hard macro MAE(maxmin) with CI95 low > 0 at ≥ half the "
        f"checkpoints from {min_checkpoint_f1} onwards.",
        _verdict(score, replication_ok=replication_ok),
        f"endpoint {hard} (test rows whose nearest neighbour in the START cohort is below "
        f"{min(thresholds):g} Tanimoto — a fixed subset), statistic macro MAE over ECFP clusters; "
        f"{score['n_meeting']}/{score['n_checkpoints']} checkpoints meet the condition "
        f"(need {score.get('need', 0)}; BCa agrees at {score['n_bca_meeting']}); "
        f"Δ by checkpoint: {_series(rows)}.{short}",
        "If maxmin is indistinguishable from random, the *choice* of ligand does not matter, only "
        "the breadth — Experiment B already showed breadth matters, so F1 dying reduces the "
        "deployable advice to 'buy any new chemotype'. An interval whose upper end is below zero "
        "would be stronger still: random would then be the better rule.",
    ))

    # --- F2 ---------------------------------------------------------------- #
    rows = _rows_for(contrasts, endpoint="all", comparison=f"offset_uncertainty_vs_{NULL_POLICY}",
                     statistic="offset_mae", min_checkpoint=min_checkpoint_f1)
    score = _majority(rows, direction="+")
    against_maxmin = _rows_for(contrasts, endpoint="vs_maxmin|all",
                               comparison="offset_uncertainty_vs_maxmin", statistic="offset_mae",
                               min_checkpoint=min_checkpoint_f1)
    verdicts.append(Verdict(
        "F2", "Acquiring the most level-uncertain ligand beats random.",
        f"offset MAE(random) − offset MAE(offset_uncertainty) with CI95 low > 0 at ≥ half the "
        f"checkpoints from {min_checkpoint_f1} onwards.",
        _verdict(score, replication_ok=replication_ok),
        f"endpoint all test rows, statistic offset MAE (the per-ligand level error, the quantity "
        f"Phase 1 says is missing); {score['n_meeting']}/{score['n_checkpoints']} checkpoints meet "
        f"the condition (need {score.get('need', 0)}; BCa agrees at {score['n_bca_meeting']}); "
        f"Δ by checkpoint: {_series(rows)}.{short} Descriptive, for the protocol's own falsifier "
        f"(is the model's uncertainty worth anything beyond chemical distance?) — the same "
        f"statistic against **maxmin** rather than random: {_series(against_maxmin)}",
        "If the model's own level uncertainty is no better than chance — or no better than the "
        "chemistry-only maxmin rule — then the forest carries no acquisition information beyond "
        "chemical distance, and the cheap rule is the one to deploy.",
    ))

    # --- F3 ---------------------------------------------------------------- #
    rows = _rows_for(contrasts, endpoint=hard, comparison=f"same_chemotype_first_vs_{NULL_POLICY}",
                     statistic="mae", min_checkpoint=1)
    score = _majority(rows, direction="-")
    verdicts.append(Verdict(
        "F3", "Buying more of the same chemistry is worse than random.",
        "hard macro MAE(random) − hard macro MAE(same_chemotype_first) with CI95 **high < 0** "
        "(random is the better rule) at ≥ half the checkpoints. NOTE: the protocol table writes "
        "this as 'hard MAE(same_chemotype_first) − hard MAE(random), CI95 high < 0', which is the "
        "opposite of its own gloss 'i.e. random is better'; the gloss is scored here and the "
        "arithmetic is printed with every number so the reading cannot be hidden.",
        _verdict(score, replication_ok=replication_ok),
        f"endpoint {hard}, statistic macro MAE; delta is random minus same_chemotype_first, so a "
        f"**negative** interval means random is better; {score['n_meeting']}/"
        f"{score['n_checkpoints']} checkpoints meet the condition (need {score.get('need', 0)}; "
        f"BCa agrees at {score['n_bca_meeting']}); Δ by checkpoint: {_series(rows)}.{short}",
        "If buying another analogue of what you already have matches random acquisition on hard "
        "chemistry, then the field's actual behaviour cost nothing measurable at this scale, and "
        "the generation's advice weakens to 'do not bother choosing'.",
    ))

    # --- F4 ---------------------------------------------------------------- #
    if not f4_enabled:
        verdicts.append(Verdict(
            "F4", "Three measurements of a distant ligand beat three more of a known one.",
            "the b = 3 maxmin curve lies below the curve that spends the same rows on the start "
            "cohort's own ligands, at ≥ half the checkpoints.",
            "SKIPPED", f4_reason,
            "Not evaluated in this run. With --f4-cap-start-rows N it is evaluated against the "
            "depth_on_start arm; if depth matched breadth there, Experiment B's headline would be "
            "contradicted at the level of individual measurements.",
        ))
        return verdicts
    rows = _rows_for(contrasts, endpoint=hard, comparison=f"maxmin_vs_{DEPTH_ARM}",
                     statistic="mae", min_checkpoint=1)
    score = _majority(rows, direction="+")
    verdicts.append(Verdict(
        "F4", "Three measurements of a distant ligand beat three more of a known one.",
        "hard macro MAE(depth_on_start) − hard macro MAE(maxmin) with CI95 low > 0 at ≥ half the "
        "checkpoints.",
        _verdict(score, replication_ok=replication_ok),
        f"endpoint {hard}, statistic macro MAE, against the depth arm that spends the identical "
        f"row budget on the start cohort's own ligands; {score['n_meeting']}/"
        f"{score['n_checkpoints']} checkpoints meet the condition (need {score.get('need', 0)}); "
        f"Δ by checkpoint: {_series(rows)}.{short}",
        "This is Experiment B's depth-versus-breadth result re-measured at the level of single "
        "measurements; it is a sanity check, not a new claim. It dies if depth_on_start matches "
        "maxmin — and note that the capped start cohort makes this a different (shallower) "
        "experiment from the default run.",
    ))
    return verdicts


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def build_checks(
    *,
    results: Sequence[FoldResult],
    integrity: Mapping[str, dict],
    curves: pd.DataFrame,
    specs: Sequence[ArmSpec],
    label_free: Mapping[str, object],
    checkpoints: Sequence[int],
) -> dict:
    """The assertions this run must satisfy before it may claim success."""
    arms = [s.arm for s in specs]

    # (1) the hard masks never moved: one digest per (seed, fold, threshold)
    mask_detail: dict[str, dict] = {}
    masks_ok = True
    for result in results:
        audit = result.evaluator_audit
        key = f"seed{result.seed}_fold{result.fold}"
        mask_detail[key] = {"distinct_mask_digests": audit["distinct_mask_digests"],
                            "n_calls": audit["n_calls"], "n_hard_rows": audit["n_hard_rows"]}
        masks_ok = masks_ok and all(v == 1 for v in audit["distinct_mask_digests"].values())

    # (2) no revealed row was ever a test row (the closure raises, so this records
    #     the count it saw across every checkpoint of every arm)
    revealed = sum(int(r.evaluator_audit["n_test_rows_revealed"]) for r in results)
    n_calls = sum(int(r.evaluator_audit["n_calls"]) for r in results)

    # (3) checkpoint 0 must be identical for every arm inside a replicate: before
    #     the first acquisition every arm is the same model on the same start rows
    zero_detail: dict[str, float] = {}
    zero_ok = True
    for result in results:
        by_replicate: dict[int, list[np.ndarray]] = {}
        for (arm, replicate, checkpoint), prediction in result.predictions.items():
            if checkpoint == 0:
                by_replicate.setdefault(int(replicate), []).append(prediction)
        for replicate, block in by_replicate.items():
            if len(block) < 2:
                continue
            stacked = np.vstack(block)
            spread = float(np.max(np.abs(stacked - stacked[0])))
            zero_detail[f"seed{result.seed}_fold{result.fold}_r{replicate}"] = spread
            zero_ok = zero_ok and spread < 1e-12

    # (4) every arm reached every requested checkpoint on every fold, or it is named
    missing: list[str] = []
    for result in results:
        for arm in arms:
            for checkpoint in checkpoints:
                if int(checkpoint) not in result.checkpoints_for(arm):
                    missing.append(f"seed{result.seed}_fold{result.fold}_{arm}_k{checkpoint}")

    expected = sum(s.replicates for s in specs) * len(results)
    observed = int(curves.groupby(["split_seed", "fold", "arm", "replicate"]).ngroups) \
        if not curves.empty else 0
    return {
        "hard_masks_fixed_per_fold": {
            "ok": bool(masks_ok),
            "detail": mask_detail,
            "note": "the start-cohort hard-chemistry masks are hashed on every checkpoint call; "
                    "one distinct digest per fold and threshold means the subset never moved with "
                    "the policy or the step"},
        "no_test_row_revealed": {
            "ok": bool(revealed == 0),
            "n_checkpoint_evaluations": int(n_calls),
            "n_test_rows_revealed": int(revealed),
            "note": "the metric closure intersects the revealed training rows with the fold's test "
                    "rows at every checkpoint and raises on any overlap"},
        "checkpoint_zero_identical_across_arms": {
            "ok": bool(zero_ok),
            "max_abs_difference": zero_detail,
            "note": "before the first acquisition every arm is the same forest on the same start "
                    "rows; a non-zero spread would mean the start cohort or the fold seed moved "
                    "with the policy"},
        "all_arms_reached_all_checkpoints": {
            "ok": not missing, "missing": missing[:20], "n_missing": len(missing),
            "note": "an arm that exhausts the pool stops early; such (seed, fold) cells are "
                    "dropped whole from the paired bootstrap rather than partially filled"},
        "curve_shape": {
            "ok": bool(observed == expected), "observed": observed, "expected": expected,
            "note": "one curve per (split seed, fold, arm, replicate)"},
        "split_integrity": {
            "ok": all(bool(v.get("ok")) for v in integrity.values()),
            "per_seed": {k: bool(v.get("ok")) for k, v in integrity.items()}},
        "label_free_acquisition": dict(label_free),
    }


def finalise_run(
    output_dir: Path, *, manifest: RunManifest, checks: Mapping[str, object],
    required_artifacts: Sequence[str] = REQUIRED_ARTIFACTS,
) -> tuple[dict, Path | None]:
    """Write the manifest, validate the directory, mark success only if it validated."""
    payload = manifest.write(output_dir)
    validation = validate_run(output_dir, manifest=payload,
                              required_artifacts=list(required_artifacts), checks=checks)
    success = write_success(output_dir, manifest=payload, validation=validation)
    manifest.write(output_dir)
    return validation, success


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _table(frame: pd.DataFrame, columns: Sequence[str] | None = None, digits: int = 4) -> str:
    if frame is None or frame.empty:
        return "```\n(no rows)\n```"
    view = frame if columns is None else frame[[c for c in columns if c in frame.columns]]
    return "```\n" + view.round(digits).to_string(index=False) + "\n```"


def _pivot(curves: pd.DataFrame, metric: str, arms: Sequence[str]) -> pd.DataFrame:
    if metric not in curves.columns:
        return pd.DataFrame()
    table = curves.pivot_table(index="checkpoint", columns="arm", values=metric, aggfunc="mean")
    return table.reindex(columns=[a for a in arms if a in table.columns])


def render_report(
    *,
    stamp: str,
    args: argparse.Namespace,
    audit: dict,
    chemistry: ChemistryMap,
    specs: Sequence[ArmSpec],
    checkpoints: Sequence[int],
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    starts: pd.DataFrame,
    verdicts: Sequence[Verdict],
    checks: Mapping[str, object],
    seconds: float,
    time_estimate: str,
    f4_reason: str,
) -> str:
    """The decision report, written for a chemist who expects to be misled."""
    primary = [s.arm for s in specs if s.family == "primary"]
    sensitivity = [s.arm for s in specs if s.family == "budget_all"]
    depth = [s.arm for s in specs if s.family == "depth"]
    hard = f"hard_nn<{min(args.thresholds):g}" if len(args.thresholds) else "all"
    lines: list[str] = []
    A = lines.append

    A("# gen6 Experiment F — which ligand should have been measured next?")
    A("")
    A(f"**Run** `gen6_acquisition_sim_{stamp}`"
      + ("  ·  **PILOT** (one seed, two folds, three policies, 8 steps, 1 replicate, reduced "
         "trees — a smoke test, not evidence)" if args.pilot else ""))
    A("")
    A("## What was actually done")
    A("")
    A(f"* One shared cohort at `min_cells = {args.eval_min_cells}`: **{audit['rows']} rows, "
      f"{audit['extractants']} extractants, {audit['ecfp_clusters']} ECFP clusters, "
      f"{audit['tanimoto_clusters']} Tanimoto-0.7 chemotypes**; target sd "
      f"{audit['target_sd']:.3f} log units.")
    A(f"* Folds hold out whole chemotypes (Experiment A's `diversity_splits`, {args.folds} folds, "
      f"seeds {list(args.split_seeds)}"
      + (f", fold indices {list(args.fold_indices)}" if args.fold_indices else "")
      + "). **The fold's held-out chemotypes are the test set and are fixed**; acquisition draws "
        "only from the EXPANDED training index.")
    A(f"* Start cohort: the {args.start_ligands} most-measured training ligands of the largest "
      f"training chemotype, all their rows"
      + (f", capped at {args.f4_cap_start_rows} rows per ligand (F4 is enabled, so every arm "
         f"shares the capped start)" if args.f4_cap_start_rows is not None else "") + ".")
    A(f"* Acquisition: up to {args.n_steps} steps, {args.budget_per_ligand} rows revealed per "
      f"acquisition, checkpoints {list(checkpoints)}, "
      f"{args.replicates} replicate(s) per (seed, fold, policy).")
    A(f"* Learner: ExtraTrees, {args.n_estimators} trees, max_features {args.max_features}, "
      f"min_samples_leaf {args.min_samples_leaf}; feature columns from blocks "
      f"{list(FOREST_BLOCKS)}; fold seed `model_seed + fold*{FOLD_SEED_STRIDE} + "
      f"{FOLD_SEED_OFFSET} + {REPLICATE_SEED_STRIDE}*replicate`.")
    A(f"* Frozen chemistry map over {len(chemistry)} extractants "
      f"({chemistry.audit['n_ecfp_clusters']} ECFP clusters, "
      f"{chemistry.audit['n_superclusters']} chemotypes at threshold "
      f"{chemistry.audit['supercluster_threshold']}).")
    A(f"* Arms: primary {primary}"
      + (f"; budget=all sensitivity {sensitivity}" if sensitivity else "")
      + (f"; F4 depth arm {depth}" if depth else "") + ".")
    A(f"* Simulation took {seconds:.0f} s. {time_estimate}")
    A("")
    A("## How to read this")
    A("")
    A("* **macro MAE** = one ECFP cluster, one vote (the study's primary metric). Pooled MAE is in "
      "the CSVs and never used for selection: the largest single ligand holds "
      f"{100 * audit.get('largest_extractant_share', float('nan')):.0f} % of rows.")
    A(f"* **hard chemistry is a fixed subset.** `{hard}` means: test rows whose maximum Tanimoto "
      "to the **start cohort's** extractants is below the threshold. It is computed once per "
      "fold, before any acquisition, and never recomputed — so the same rows are scored for every "
      "policy at every step. Recomputing it against the current training set would shrink the "
      "subset exactly for the policies that buy distant chemistry, which would manufacture the "
      "result this experiment is trying to measure.")
    A("* **checkpoint 0 is the start cohort alone**, identical for every arm by construction "
      f"(measured max spread across arms: "
      f"{max(list(checks['checkpoint_zero_identical_across_arms']['max_abs_difference'].values()) or [0.0]):.2e}). "
      "It is excluded from the hypothesis scoring, where every delta there is structurally zero.")
    A("* **`point_delta = statistic(reference) − statistic(candidate)`**, so a **positive delta "
      "means the candidate policy is better** (lower error). Every contrast below uses `random` "
      "as the reference except F4's, which uses the depth arm.")
    A(f"* The paired bootstrap resamples **chemotype blocks** ({args.bootstrap_replicates} "
      "replicates) with the ECFP cluster as the scoring unit, and reports BCa and cluster-robust "
      "intervals beside the percentile one — the Phase 1 dominant-block caveat applies to the "
      "`all` endpoint and not to the hard subsets.")
    A("* A policy is a **distribution over acquisition sequences**. Predictions are averaged over "
      f"replicates before the bootstrap; the per-replicate spread is in `acquisition_curves.csv` "
      f"and in the sd columns of `curve_summary.csv`"
      + (" — with one replicate in this run, that spread is undefined." if args.replicates < 2
         else "."))
    A("")
    A("## The start cohorts (this is the experiment's premise, so read it first)")
    A("")
    A("A narrow, deep start is the situation this project was in for five generations. One fold "
      "degenerates: holding out the diglycolamide chemotype leaves a largest-training-chemotype "
      "with almost no ligands in it. That fold is disclosed, not excluded, and it drags every "
      "curve towards zero effect because there is barely a start cohort to improve on.")
    A("")
    A(_table(starts, ["split_seed", "fold", "start_chemotype", "n_start_ligands", "n_start_rows",
                      "largest_chemotype_ligands", "n_pool_ligands", "n_pool_rows",
                      "start_is_short", "n_test_rows", "n_hard_rows_0_4"], 1))
    A("")
    A("## Learning curves (mean over folds, seeds and replicates)")
    A("")
    for metric, title in (
        ("macro_mae", "macro MAE, all test rows (one ECFP cluster = one vote)"),
        (f"hard_nn{min(args.thresholds):g}__macro_mae",
         f"macro MAE on hard chemistry ({hard}; the FIXED start-cohort subset)"),
        ("offset_mae", "offset MAE (per-ligand level error — the quantity Phase 1 says is missing)"),
        ("shape_mae", "shape MAE (within-ligand response shape)"),
        ("worst_quartile_ligand_mae", "worst-quartile ligand MAE"),
    ):
        table = _pivot(curves, metric, [s.arm for s in specs])
        if table.empty or table.isna().all().all():
            continue
        A(f"**{title}**")
        A("")
        A("```")
        A(table.round(4).to_string())
        A("```")
        A("")
    A("### Coverage — what each policy actually bought")
    A("")
    A("The mechanism, not decoration: if two policies buy the same chemistry, no difference in "
      "accuracy is expected. `n_train_chemotypes` counts Tanimoto-0.7 chemotypes among the "
      "training ligands; `mean_nn_test_to_train` is the mean over **test ligands** of the maximum "
      "Tanimoto to the current training ligands (higher = the test set is better covered).")
    A("")
    for metric, title in (("n_train_chemotypes", "chemotypes in training"),
                          ("mean_nn_test_to_train",
                           "mean nearest-training-neighbour Tanimoto of the test ligands"),
                          ("n_train_rows", "training rows")):
        table = _pivot(curves, metric, [s.arm for s in specs])
        if table.empty:
            continue
        A(f"**{title}**")
        A("")
        A("```")
        A(table.round(3).to_string())
        A("```")
        A("")
    A("## Pre-registered hypotheses (protocol §F)")
    A("")
    A("Scoring rule, stated so a verdict cannot be over-read: **PASS** = the pre-registered "
      "condition is met **and** the run carried the pre-registered replication (≥ 2 split seeds, "
      "≥ 2 acquisition replicates, ≥ 2 fold-jobs); **FAIL** = the interval excludes the predicted "
      "direction at ≥ half the scored checkpoints (evidence *against*, which counts even in a "
      "short run); **INCONCLUSIVE** = the intervals straddle zero, the contrast could not be "
      "formed, or the condition was met on a run too short to register it. An INCONCLUSIVE is not "
      "a weak PASS. Only F1–F4 are protected; every other interval in this report is descriptive.")
    for verdict in verdicts:
        A("")
        A(f"**{verdict.hypothesis} — {verdict.verdict}**  ·  {verdict.statement}")
        A("")
        A(f"* pass condition: {verdict.pass_condition}")
        A(f"* measured: {verdict.evidence}")
        A(f"* what would falsify this: {verdict.falsifier}")
    A("")
    A("## Every contrast, every checkpoint")
    A("")
    A("Reference is `random` (F4's is the depth arm). Positive delta = the candidate policy is "
      "better. `n_folds` is the number of (seed, fold) cells behind the row; `units_total` the "
      "number of ECFP clusters; `bootstrap_blocks` the number of chemotypes resampled. Only the "
      "protected contrast set is printed here; the descriptive sets (`vs_maxmin|…`, "
      "`budget_all|…`) and the `shape_mae` statistic are in `contrasts.csv`.")
    A("")
    shown = contrasts[contrasts["statistic"].isin(("mae", "offset_mae"))
                      & ~contrasts["endpoint"].astype(str).str.contains(r"\|")] \
        if not contrasts.empty else contrasts
    A(_table(shown, ["endpoint", "checkpoint", "comparison", "statistic", "point_delta",
                     "ci95_low", "ci95_high", "bca_low", "bca_high", "cluster_robust_low",
                     "block_macro_delta", "units_improved", "units_total", "bootstrap_blocks",
                     "n_folds"]))
    A("")
    A("The descriptive contrast against the chemistry-only rule (`vs_maxmin|…`) answers the "
      "protocol's own falsifier for F2 — whether the model's uncertainty carries anything beyond "
      "chemical distance — and is quoted inside F2's evidence above.")
    A("")
    if sensitivity:
        A("## Disclosed sensitivity — buying every row of the acquired ligand")
        A("")
        A(f"`{'`, `'.join(sensitivity)}` repeat two policies with `budget = all rows of the "
          f"acquired ligand` instead of {args.budget_per_ligand}. It is a different experiment "
          "(each step buys ~30 rows rather than 3), reported so the b = 3 result cannot be read as "
          "a claim about all budgets. It is not a protected comparison.")
        A("")
        A(_table(summary[summary["family"] == "budget_all"],
                 ["arm", "checkpoint", "n_train_rows_mean", "macro_mae_mean",
                  f"hard_nn{min(args.thresholds):g}__macro_mae_mean", "offset_mae_mean", "n_runs"]))
        A("")
    A("## How to break this result")
    A("")
    A("* **One fold has almost no start cohort.** The fold that holds out the diglycolamides "
      "starts from a handful of ligands, so every policy improves enormously there and the "
      "between-policy contrast is dominated by folds where the start is genuinely narrow and deep. "
      "Re-run with `--fold-indices` excluding it and see whether the verdicts survive; the "
      "per-fold curves are in `acquisition_curves.csv`.")
    A("* **The hard subset is defined by the start cohort, so it differs per fold.** It is fixed "
      "*within* a fold (that is what makes the policies comparable), but a fold whose start "
      "chemotype is unusual has an unusual hard subset. The row and ligand counts per fold are in "
      "the start-cohort table above and in `validation.json`.")
    A("* **Replicates are not seeds and seeds are not experiments.** The split seeds re-partition "
      "the same ligands; the acquisition replicates re-draw the same policy. The load-bearing "
      "statistic is the bootstrap over chemotype blocks.")
    A("* **Every policy still trains on the same learner and the same features.** A policy that "
      "helps only this feature set would be an artefact of the representation.")
    A("* **The pool is not the world.** Acquisition can only buy ligands that exist in this "
      "dataset; a maxmin rule that is starved of genuinely distant candidates will look like "
      "random by construction. `nn_to_acquired_before` in `acquisition_steps.csv` shows how "
      "distant the purchases actually were.")
    f4_line = ("was run against the capped start cohort described above" if depth
               else f"is SKIPPED in this run: {f4_reason}")
    A(f"* **F4** {f4_line}.")
    A("* **Do not quote the percentile interval alone at the `all` endpoint.** One chemotype holds "
      "a fifth of the scoring units; BCa, cluster-robust and block-macro columns sit beside it in "
      "`contrasts.csv` for that reason.")
    A("")
    A("## Run-time checks")
    A("")
    for name, entry in checks.items():
        state = entry.get("ok") if isinstance(entry, Mapping) else entry
        A(f"* `{name}`: **{state}**"
          + (f" — {entry.get('note')}" if isinstance(entry, Mapping) and entry.get("note") else ""))
    A("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else \
        REPO_ROOT / "runs" / f"gen6_acquisition_sim_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    if args.pilot:
        args.split_seeds = list(args.split_seeds)[:1]
        args.fold_indices = [0, 2]
        args.policies = [p for p in ("random", "maxmin", "offset_uncertainty")
                         if p in args.policies]
        args.n_steps = 8
        args.checkpoints = [1, 2, 3, 5, 8]
        args.replicates = 1
        args.budget_all_replicates = 1
        args.n_estimators = min(int(args.n_estimators), 80)
        args.bootstrap_replicates = min(int(args.bootstrap_replicates), 2000)
        log("PILOT: one seed, folds 0 and 2, policies random/maxmin/offset_uncertainty, 8 steps, "
            "checkpoints 1/2/3/5/8, 1 replicate, 80 trees — a smoke test, not evidence")

    if NULL_POLICY not in args.policies:
        raise SystemExit(f"policy {NULL_POLICY!r} is the required null of every F hypothesis and "
                         f"must be in --policies")
    checkpoints = sorted({int(c) for c in args.checkpoints if 0 < int(c) <= int(args.n_steps)})
    dropped_checkpoints = sorted({int(c) for c in args.checkpoints} - set(checkpoints))
    if dropped_checkpoints:
        log(f"checkpoints {dropped_checkpoints} exceed --n-steps {args.n_steps}; not scored")
    specs = build_arm_specs(args)
    f4_reason = F4_SKIP_REASON

    # --- data ---------------------------------------------------------------
    source = pd.read_parquet(args.dataset)
    descriptor_path: Path | None = None
    if args.descriptors is not None and str(args.descriptors) not in ("", "."):
        descriptor_path = Path(args.descriptors)
    descriptors = None
    if descriptor_path is not None and descriptor_path.is_file():
        descriptors = pd.read_parquet(descriptor_path)
    else:
        log(f"WARNING: no descriptor parquet at {args.descriptors!s}; the LIG2D_EXT block will be "
            f"absent and the champion feature set will fail closed")

    data = build_level_dataset(
        source, min_rows_per_extractant=args.eval_min_cells,
        replicate_policy=args.replicate_policy, drop_below_log_d=args.log_d_floor,
        ligand_descriptors=descriptors)
    frame = data.frame.reset_index(drop=True)
    audit = data.audit
    missing_blocks = [b for b in FOREST_BLOCKS if b not in data.blocks]
    if missing_blocks:
        raise SystemExit(f"the champion feature set needs blocks {missing_blocks}, which this "
                         f"cohort does not have: {sorted(data.blocks)}")
    feature_columns = data.block_columns(FOREST_BLOCKS)
    log(f"shared cohort: {audit['rows']} rows, {audit['extractants']} extractants, "
        f"{audit['ecfp_clusters']} ECFP clusters, {audit['tanimoto_clusters']} chemotypes; "
        f"target sd {audit['target_sd']:.3f}")
    log(f"feature columns: {len(feature_columns)} from blocks {list(FOREST_BLOCKS)}")

    if args.chemistry_map and Path(args.chemistry_map).exists():
        chemistry = ChemistryMap.from_parquet(args.chemistry_map)
        chemistry_source = str(args.chemistry_map)
    else:
        chemistry = build_chemistry_map(source, ligand_descriptors=descriptors)
        chemistry_source = "rebuilt from the source table in this run"
    log(f"chemistry map: {len(chemistry)} extractants, "
        f"{chemistry.audit['n_ecfp_clusters']} ECFP clusters, "
        f"{chemistry.audit['n_superclusters']} chemotypes ({chemistry_source})")

    # --- folds ---------------------------------------------------------------
    jobs: list[tuple[int, object]] = []
    integrity: dict[str, dict] = {}
    for seed in args.split_seeds:
        splits = diversity_splits(frame, group_column="tanimoto_cluster", n_splits=int(args.folds),
                                  seed=int(seed), base_min_cells=int(args.base_min_cells),
                                  arms=(BASE_ARM, EXPANDED_ARM))
        report = assert_split_integrity(frame, splits)
        integrity[str(seed)] = report
        if not report["ok"]:
            log(f"WARNING: split integrity failed for seed {seed}; see validation.json")
        for split in splits:
            if args.fold_indices is None or int(split.fold) in set(args.fold_indices):
                jobs.append((int(seed), split))
    if not jobs:
        raise SystemExit(f"no fold matched --fold-indices {args.fold_indices}")
    log(f"{len(jobs)} (seed, fold) jobs; arms {[s.arm for s in specs]}; "
        f"{sum(s.replicates for s in specs)} simulations per fold")

    # --- label-free acquisition, on the real cohort --------------------------
    # Fail closed, not silent: protocol §6 lists label-free acquisition as a required
    # control, so a run that skipped the proof must not be able to claim success.
    label_free: dict = {
        "ok": False, "status": "not run (--no-label-free-check); a required control of protocol "
                               "§6 was skipped, so this run cannot validate"}
    if args.label_free_check:
        seed0, split0 = jobs[0]
        started = time.time()
        label_free = label_free_first_pick(
            frame, data, args=args, chemistry=chemistry, feature_columns=feature_columns,
            policies=args.policies, seed=seed0, split=split0)
        log(f"label-free acquisition check on seed {seed0} fold {split0.fold} "
            f"({time.time() - started:.0f}s): "
            f"{'PASS' if label_free['ok'] else 'FAIL ' + str(label_free['policies_that_moved'])}")

    # --- the main loop, with a time estimate from the first fold -------------
    started = time.time()
    first = run_fold(frame, data, args=args, chemistry=chemistry, feature_columns=feature_columns,
                     specs=specs, seed=jobs[0][0], split=jobs[0][1], checkpoints=checkpoints,
                     log=log)
    projected = first.seconds * len(jobs)
    time_estimate = (f"First fold took {first.seconds:.1f} s; {len(jobs)} fold-jobs → estimated "
                     f"{projected:.0f} s ({projected / 60:.1f} min) of simulation "
                     f"(plus bootstrap and reporting).")
    log(f"TIME ESTIMATE: {time_estimate}")
    results: list[FoldResult] = [first]
    for index, (seed, split) in enumerate(jobs[1:], start=2):
        results.append(run_fold(frame, data, args=args, chemistry=chemistry,
                                feature_columns=feature_columns, specs=specs, seed=seed,
                                split=split, checkpoints=checkpoints, log=log))
        log(f"  job {index}/{len(jobs)} done in {results[-1].seconds:.1f}s "
            f"({time.time() - started:.0f}s elapsed)")
    simulation_seconds = time.time() - started
    log(f"simulation done in {simulation_seconds:.1f}s")

    # --- tables ---------------------------------------------------------------
    curves = pd.DataFrame([row for r in results for row in r.curves])
    steps = pd.DataFrame([row for r in results for row in r.steps])
    starts = pd.DataFrame([{
        "split_seed": r.seed, "fold": r.fold, **r.start_audit,
        "n_test_rows": int(r.evaluator_audit["n_test_rows"]),
        **{f"n_hard_rows_{float(t):g}".replace(".", "_"):
           int(r.evaluator_audit["n_hard_rows"][f"nn<{float(t):g}"])
           for t in args.thresholds},
    } for r in results])
    curve_metrics = [c for c in curves.columns if c not in (
        "split_seed", "fold", "arm", "policy", "budget_per_ligand", "family", "replicate",
        "checkpoint", "step")]
    summary = curve_summary_table(curves, curve_metrics)

    curves.to_csv(output_dir / "acquisition_curves.csv", index=False)
    steps.to_csv(output_dir / "acquisition_steps.csv", index=False)
    summary.to_csv(output_dir / "curve_summary.csv", index=False)
    starts.to_csv(output_dir / "start_cohorts.csv", index=False)

    # --- paired bootstrap per checkpoint --------------------------------------
    log("paired bootstrap per checkpoint (unit ECFP cluster, block chemotype)")
    contrast_parts: list[pd.DataFrame] = []
    primary_arms = [s.arm for s in specs if s.family == "primary"]
    contrast_parts.append(contrast_table(
        results, arms=primary_arms, reference=NULL_POLICY, checkpoints=checkpoints,
        thresholds=args.thresholds, replicates=int(args.bootstrap_replicates),
        seed=int(args.bootstrap_seed), log=log))
    # Descriptive, not protected: the protocol's own falsifier for F2 asks whether an
    # uncertainty policy is better than the *chemistry-only* rule, not just than random.
    # ``random`` is left out of this set — its contrast against maxmin is F1 with the
    # sign flipped, and printing it twice invites double counting.
    versus_maxmin = ["maxmin"] + [a for a in primary_arms if a not in ("maxmin", NULL_POLICY)]
    if "maxmin" in primary_arms and len(versus_maxmin) > 1:
        contrast_parts.append(contrast_table(
            results, arms=versus_maxmin, reference="maxmin", checkpoints=checkpoints,
            thresholds=args.thresholds, replicates=int(args.bootstrap_replicates),
            seed=int(args.bootstrap_seed), log=log, endpoint_prefix="vs_maxmin|"))
    sensitivity_arms = [s.arm for s in specs if s.family == "budget_all"]
    if f"{NULL_POLICY}{BUDGET_ALL_SUFFIX}" in sensitivity_arms and len(sensitivity_arms) > 1:
        contrast_parts.append(contrast_table(
            results, arms=sensitivity_arms, reference=f"{NULL_POLICY}{BUDGET_ALL_SUFFIX}",
            checkpoints=checkpoints, thresholds=args.thresholds,
            replicates=int(args.bootstrap_replicates), seed=int(args.bootstrap_seed), log=log,
            endpoint_prefix="budget_all|"))
    if any(s.family == "depth" for s in specs) and "maxmin" in primary_arms:
        contrast_parts.append(contrast_table(
            results, arms=[DEPTH_ARM, "maxmin"], reference=DEPTH_ARM, checkpoints=checkpoints,
            thresholds=args.thresholds, replicates=int(args.bootstrap_replicates),
            seed=int(args.bootstrap_seed), log=log))
    contrasts = pd.concat([p for p in contrast_parts if not p.empty], ignore_index=True) \
        if any(not p.empty for p in contrast_parts) else pd.DataFrame()
    for column in ("endpoint", "checkpoint", "comparison", "statistic", "point_delta", "ci95_low",
                   "ci95_high"):
        if column not in contrasts.columns:
            contrasts[column] = pd.Series(dtype=float)
    contrasts.to_csv(output_dir / "contrasts.csv", index=False)

    # --- verdicts and checks ---------------------------------------------------
    # The pre-registered design is 3 split seeds x 5 folds x 2 acquisition
    # replicates.  A run that carried less than that may report its numbers but may
    # not PASS on them — the same rule Experiment A applies to a single-seed pilot.
    replication_ok = (len(args.split_seeds) >= 2 and int(args.replicates) >= 2
                      and len(results) >= 2)
    replication_note = (f"Replication short of the pre-registered design "
                        f"({len(args.split_seeds)} split seed(s), {args.replicates} acquisition "
                        f"replicate(s), {len(results)} fold-job(s) against 3 x 2 x 15), so no "
                        f"verdict here may be read as a PASS.")
    verdicts = score_hypotheses(
        contrasts, thresholds=args.thresholds,
        f4_enabled=any(s.family == "depth" for s in specs), f4_reason=f4_reason,
        replication_ok=replication_ok, replication_note=replication_note)
    checks = build_checks(results=results, integrity=integrity, curves=curves, specs=specs,
                          label_free=label_free, checkpoints=checkpoints)

    # --- manifest --------------------------------------------------------------
    fold_records = []
    for result, (seed, split) in zip(results, jobs):
        train_index = split.train_index_by_arm[EXPANDED_ARM]
        extractants = frame["extractant"].astype(str).to_numpy()
        chemotypes = frame["tanimoto_cluster"].astype(str).to_numpy()
        start_rows = int(result.start_audit["n_start_rows"])
        fold_records.append({
            "fold": int(split.fold), "split_seed": int(seed),
            "test_row_ids_sha256": sha256_text("|".join(
                sorted(frame["row_id"].astype(str).to_numpy()[split.test_index]))),
            "test_extractants": sorted(set(extractants[split.test_index])),
            "test_superclusters": list(split.held_out_groups),
            "train_extractants_by_arm": {
                "POOL": sorted(set(extractants[train_index])),
                "START": sorted(result.start_extractants)},
            "train_superclusters_by_arm": {
                "POOL": sorted(set(chemotypes[train_index])),
                "START": [str(result.start_audit["start_chemotype"])]},
            "n_test_rows": int(split.test_index.size),
            "n_train_rows_by_arm": {"POOL": int(train_index.size), "START": start_rows},
            "start_audit": result.start_audit,
            "hard_mask_audit": result.evaluator_audit,
        })
    split_definition = {
        "experiment": "F_ligand_acquisition",
        "shared_cohort_min_cells": int(args.eval_min_cells),
        "group_column": "tanimoto_cluster",
        "n_splits": int(args.folds),
        "fold_indices": list(args.fold_indices) if args.fold_indices else "all",
        "split_seeds": [int(s) for s in args.split_seeds],
        "fold_algorithm": "shuffle unique group labels with np.random.default_rng(seed), deal "
                          "round-robin (gen5 seeded_group_kfold, via diversity_splits)",
        "training_pool": f"{EXPANDED_ARM} (every training-fold row)",
        "test_set": "the fold's held-out chemotypes; fixed, never acquired from",
        "start_cohort": f"the {args.start_ligands} most-measured training ligands of the largest "
                        f"training chemotype",
        "start_row_cap": args.f4_cap_start_rows,
        "budget_per_ligand": args.budget_per_ligand,
        "n_steps": int(args.n_steps),
        "checkpoints": list(checkpoints),
        "hard_chemistry_reference": "the START cohort's extractants, frozen per fold",
        "model_fold_seed": f"model_seed + fold*{FOLD_SEED_STRIDE} + {FOLD_SEED_OFFSET} + "
                           f"{REPLICATE_SEED_STRIDE}*replicate",
        "acquisition_rng": f"sha256(split_seed|fold|arm|replicate|{ACQUISITION_SEED_SALT})",
        "arms": [{"arm": s.arm, "policy": s.policy, "budget_per_ligand": s.budget_label,
                  "replicates": s.replicates, "family": s.family} for s in specs],
    }

    provenance_state: dict = {"status": "not_audited_in_this_run"}
    if args.provenance_state_json and Path(args.provenance_state_json).exists():
        provenance_state = json.loads(Path(args.provenance_state_json).read_text())

    manifest = RunManifest(layer=GEN6_LAYER, run_id=f"gen6_acquisition_sim_{stamp}")
    manifest.record_dataset(dataset_path=args.dataset, source_frame=source,
                            descriptor_path=descriptor_path, descriptor_frame=descriptors)
    manifest.record_code([REPO_ROOT / "src" / "lanthanide_separation" / "gen6",
                          REPO_ROOT / "src" / "lanthanide_separation" / "levels.py",
                          Path(__file__).resolve()], repo_root=REPO_ROOT)
    manifest.record_features(feature_sets={"FOREST_BLOCKS": list(feature_columns)})
    manifest.record_split(definition=split_definition, folds=fold_records)
    manifest.record_chemistry(definition={
        **cluster_manifest(chemistry)["definition"],
        "source": chemistry_source,
        "chemistry_table_sha256": sha256_frame(chemistry.table),
        "n_extractants": len(chemistry), "audit": chemistry.audit})
    manifest.record_provenance(state=provenance_state)
    manifest.record_preprocessing([
        {"step": "build_level_dataset", "min_rows_per_extractant": int(args.eval_min_cells),
         "replicate_policy": args.replicate_policy, "drop_below_log_d": float(args.log_d_floor),
         "ligand_descriptors": bool(descriptors is not None), "cohort_audit": audit},
        {"step": "LevelRegressor", "learner": "extratrees",
         "n_estimators": int(args.n_estimators), "max_features": float(args.max_features),
         "min_samples_leaf": int(args.min_samples_leaf),
         "imputation": "median + missing indicator, fitted on the revealed rows only",
         "sample_weights": ("group_balanced_weights on the revealed rows' ecfp_cluster"
                            if args.weighting == "cluster" else "none (row weighting)"),
         "weighting": args.weighting},
        {"step": "hard chemistry masks", "reference": "the START cohort of the fold",
         "thresholds": [float(t) for t in args.thresholds],
         "detail": "computed once per fold before any acquisition and re-hashed at every "
                   "checkpoint; see checks.hard_masks_fixed_per_fold"},
    ])
    manifest.record_many({
        "model_seed": int(args.model_seed),
        "split_seeds": [int(s) for s in args.split_seeds],
        "cohort_sha256": sha256_frame(frame, sort_rows_by=["row_id"]),
        "pilot": bool(args.pilot),
        "bootstrap": {"replicates": int(args.bootstrap_replicates),
                      "seed": int(args.bootstrap_seed), "scoring_unit": "ecfp_cluster",
                      "resample_block": "tanimoto_cluster"},
        "hard_chemistry_thresholds": [float(t) for t in args.thresholds],
        "hypotheses": [v.__dict__ for v in verdicts],
        "simulation_seconds": float(simulation_seconds),
    })

    report = render_report(
        stamp=stamp, args=args, audit=audit, chemistry=chemistry, specs=specs,
        checkpoints=checkpoints, curves=curves, summary=summary, contrasts=contrasts,
        starts=starts, verdicts=verdicts, checks=checks, seconds=simulation_seconds,
        time_estimate=time_estimate, f4_reason=f4_reason)
    (output_dir / "decision_report.md").write_text(report)

    (output_dir / "summary.json").write_text(json.dumps({
        "run_id": f"gen6_acquisition_sim_{stamp}",
        "layer": GEN6_LAYER,
        "pilot": bool(args.pilot),
        "cohort_audit": audit,
        "chemistry_audit": chemistry.audit,
        "arms": [{"arm": s.arm, "policy": s.policy, "budget_per_ligand": s.budget_label,
                  "replicates": s.replicates, "family": s.family} for s in specs],
        "split_seeds": [int(s) for s in args.split_seeds],
        "fold_indices": list(args.fold_indices) if args.fold_indices else "all",
        "checkpoints": list(checkpoints),
        "n_estimators": int(args.n_estimators),
        "weighting": args.weighting,
        "simulation_seconds": float(simulation_seconds),
        "time_estimate": time_estimate,
        "replication": {"ok": bool(replication_ok), "note": replication_note,
                        "n_split_seeds": len(args.split_seeds),
                        "n_acquisition_replicates": int(args.replicates),
                        "n_fold_jobs": len(results)},
        "start_cohorts": starts.to_dict("records"),
        "curve_summary": summary.to_dict("records"),
        "contrasts": contrasts.to_dict("records") if not contrasts.empty else [],
        "hypotheses": [v.__dict__ for v in verdicts],
        "checks": checks,
    }, indent=2, default=str) + "\n")

    validation, success = finalise_run(output_dir, manifest=manifest, checks=checks)

    print(report)
    for verdict in verdicts:
        log(f"{verdict.hypothesis}: {verdict.verdict}")
    if success is None:
        log(f"VALIDATION FAILED: {validation['failed_checks']} "
            f"missing_keys={validation['missing_manifest_keys']} "
            f"missing_artifacts={validation['missing_artifacts']}; wrote _FAILED.json")
        return 1
    log(f"written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""``evaluation/discovery.py`` -- the discovery plan, the resumable per-fold records, the cost model and the scorer
core (``preregistration.md``, sealed 2026-09-15: sections 0 item 3, 3, 4, 5, 6, 7, 8, 9 S1, 11 H3, 12, 13, 16, 19).

Nothing here fits a model.  ``scripts/g19_run_discovery.py`` fits the arms and writes one prediction parquet plus one
JSON record per (arm, design, variant, seed, outer fold); ``scripts/g19_score_discovery.py`` aggregates them with the
functions below.

Plan (:func:`enumerate_plan`; section 7 compute plan items 3-8 as amended by **POST-HOC addendum 1**)
-----------------------------------------------------------------------------------------------------
The sealed section 7 plan was priced at 1,535.7 CPU-hours against a 60-hour budget, so the addendum below the sealed
footer (2026-09-15; no learned-model outcome existed) replaced items 1-2, the discovery seed plan, the learned-arm
refit set and the learned-arm V5-PAIR scope.  What this module implements (:data:`READINGS` ``addendum1_inner_design``,
``addendum1_seeds``, ``addendum1_sensitivities``, ``addendum1_v5pair``, ``plan_order``):

* **items 1-2** every V5 design and variant of every learned arm and of B6 tunes on three SIMULTANEOUS inner folds --
  one inner fit per configuration per fold, all of the fold's inner cells (<= 30) hidden at once
  (``gen19ct.models.inner_design.SimultaneousInnerCells``); V1 / V2 inner designs are unchanged; every job is
  :attr:`JobSpec.inner_mode` ``full`` and the "first inner fold" reading is gone;
* **item 3** learned arms (B5 = M0, FLAT_CAT, B6, B6r0, B8, M1-M7) run discovery seed **104729 only**
  (:data:`PLAN_SEEDS`); R19 item 4 is NOT_EVALUATED in discovery; the comparators' intervals are the seed-104729 ones;
* **item 4** a learned arm runs only the V5 strict and HNO3-only refits, and R19 item 6 is the reduced sensitivity set;
* **item 5** V5-PAIR carries M2 (with its M1 prerequisite) plus the scorer's B3x / B3i re-fit, and freezing candidates
  with a pair endpoint.

A job is ``(kind, arm, design, variant, scheme, seed = 104729, half = S)``.  Passes: the nested-certificate safeguard
-> **B6 / B6r0** (V5-primary exact and batched, the batched-vs-exact check; V1 exact and ten-fold, the ten-fold check;
V2; the strict / HNO3-only refits of H1b / H4) -> the seed-104729 main designs in the registered arm order (**B5 = M0,
FLAT_CAT, B8** -> **M1** -> **M2**, each on V5-primary batched, V1 ten-fold and V2) -> the stop rule (evaluated by the
scorer) -> the S1(c) V5-PAIR run -> the strict / HNO3-only refits of the H1 / H4 arms -> conditional
freezing-candidate runs -> ``M3+ not implemented``.  Everything the addendum drops is emitted as a ``marker`` job that
names it, so nothing is a silent absence.  Section 7 item 5 is unchanged: when the 60-hour budget is exhausted only
M3-M7 are demoted (:func:`demotable`); every job of this plan keeps running.

Guards (section 2)
------------------
Only the **selection half** is scored: :func:`selection_scored_ids` keeps the scored rows whose ``row_half`` is ``S``,
:func:`assert_selection_rows` re-derives every row's registered half from ``feasibility_halves.csv`` independently and
raises on a confirmation-half row; :func:`assert_v6_clean` passes every scoring index through
``registered.assert_not_scored``.  The scorer reads pre-seal comparator predictions with a parquet row filter
``half == "S"`` (confirmation rows are never materialised) and re-asserts both guards on every frame.

Records and resume (section 16)
-------------------------------
``evaluation/discovery/<arm>/<design>__<variant>_<scheme>[_sr_iii_dropped]/s<seed>/<fold>.parquet`` + ``.json``.
The JSON holds the job, the fold hash, the design hash, the selected configuration, inner scores, best iterations /
epochs, fit and calibration seconds, peak RSS, the sealed pre-registration digest and the digest :func:`fold_digest`
(code digest + job + fold id / hash + the runner's extras: guard mode, batching label, model fold number and seed,
design hash, registration digest, M1's record digest for M2); a fold is skipped when both files exist, the digest
matches and every requested step is recorded (:func:`resume_status`).  Every reader that decides something
(:func:`read_discovery_predictions`: the scorer, the B6 checks) requires each record's digest and fold hash to equal
the ones the current code, fold file and plan state produce, and the fold ids to be exactly the job's fittable folds.

Scorer core
-----------
:func:`paired_units` (identical scored rows, the section 4 unit of each design, every registered cluster unit),
:func:`evaluate_contrast` (``transfer.r19`` items 1-6 with item 4 over the discovery seeds and item 6 over the
registered sensitivities, scoped verdicts for the stop rule / ladder / freezing screen, TOST), :func:`stop_rule`,
:func:`ladder_step`, :func:`apply_bh`, :data:`REGISTERED_CONTRASTS` (section 19).

Readings where the registration is silent are listed in :data:`READINGS` (printed with every output).
"""
from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
import math
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.data import load as LOAD
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import transfer as ET
from gen19ct.folds import cell_holdout as CH
from gen19ct.folds import io as FI
from gen19ct.folds import registered as FR
from gen19ct.folds import source_holdout as SH
from gen19ct.models import interface as I

SCHEMA = "gen19.discovery.v1"

# --------------------------------------------------------------------------------------------- #
# registered constants (sections 3, 6, 7, 15)
# --------------------------------------------------------------------------------------------- #

DISCOVERY_SEEDS: tuple[int, ...] = tuple(FI.DISCOVERY_SEEDS)
#: section 7 compute plan items 1-2: the full inner design and every ladder decision
PRIMARY_SEED = DISCOVERY_SEEDS[0]
OTHER_SEEDS: tuple[int, ...] = DISCOVERY_SEEDS[1:]
#: POST-HOC addendum 1 item 3: in DISCOVERY every fitted arm runs seed 104729 only (the other four discovery seeds are
#: dropped; R19 item 4 is NOT_EVALUATED in discovery and unchanged at confirmation).  Every other module keeps
#: :data:`DISCOVERY_SEEDS` -- the registered seed set the model fold numbers are enumerated over (addendum reading 6(b))
PLAN_SEEDS: tuple[int, ...] = (PRIMARY_SEED,)
SELECTION, CONFIRMATION = "S", "C"
#: the sealed pre-registration (``preregistration.md`` footer and ``manifests/prereg_sha256.txt``); discovery refuses
#: any other text above the footer (task X finding V-LP-02)
REGISTERED_PREREG_SHA256 = "135842499a86eb3d478ece01a45718ac5b9673a4134bb4acda550f975bb45641"
#: section 7 compute plan item 5
BUDGET_HOURS = 60.0
#: POST-HOC addendum 2, "2. Ladder M3-M7" -> Budget: the ladder has its OWN 40 h wall-clock ledger, checked before each
#: step; :data:`BUDGET_HOURS` above stays as registered for the discovery stages.  ``scripts/g19_run_ladder.py`` takes
#: its ``LADDER_BUDGET_HOURS`` from here, so the figure has one source
LADDER_BUDGET_HOURS = 40.0
#: the ladder's own ledger of addendum 2 (written by the ladder runner; absent until the first step runs)
LADDER_WALL_CLOCK_REL = "evaluation/ladder/decisions/wall_clock.json"
DEMOTION_ORDER: tuple[str, ...] = ("M7", "M6", "M5", "M4", "M3")
NEVER_DEMOTED: tuple[str, ...] = ("H1", "H1b", "H4")
LADDER: tuple[str, ...] = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7")
IMPLEMENTED_LADDER: tuple[str, ...] = ("M0", "M1", "M2")
NOT_IMPLEMENTED = "M3+ not implemented"
#: section 6: M0 is B5
ARM_ALIASES: dict[str, str] = {"M0": "B5"}
HEAVY_ARMS: tuple[str, ...] = ("B5", "FLAT_CAT", "B8", "M1", "M2")
B6_ARMS: tuple[str, ...] = ("B6", "B6r0")
FITTED_ARMS: tuple[str, ...] = B6_ARMS + HEAVY_ARMS
#: addendum 1 items 1-4 "learned arms": B5 = M0, FLAT_CAT, B6, B6r0, B8 and M1-M7 -- the arms that are fitted, whose V5
#: inner tuning is the simultaneous design, that run seed 104729 only and whose refit sensitivities are the reduced set.
#: The closed-form arms (B0-B4 variants, B7, the pair yardsticks) keep the full registered sensitivity set
LEARNED_ARMS: tuple[str, ...] = B6_ARMS + HEAVY_ARMS + ("M3", "M4", "M5", "M6", "M7")
#: the number of POST-HOC addenda the DISCOVERY records were written under (addendum 1).  POST-HOC addendum 2 item 5:
#: "The gate constants are replaced by the registry" -- every gate reads ``registry.gate_expectations(stage)``
#: (manifests/digest_registry.json); this value is only the fallback when that file does not exist and the frozen value
#: the registry's ``discovery`` entry was read from the records with.  Never edit it: a record is verified against the
#: registry entry of its stage, never against the live text
N_ADDENDA_EXPECTED = 1
#: the SHA-256 of the LF-normalised text BELOW the sealed footer under which the DISCOVERY records were written (POST-HOC
#: addendum 1; task X finding V-F01).  Addendum 2 item 5: the registry entry of a stage governs its records and the seal
#: gate of its runner (``gen19ct.evaluation.registry``); this literal is the fallback without a registry and the value
#: the ``discovery`` entry carries (the report prints it from the registry, ``digest_registry_lines``)
REGISTERED_ADDENDA_SHA256 = "05f36fc0354adb75ae41655977115d827ffce1227b2bdbfeeba28a975f0e8e54"
ADDENDUM_LABEL = "reduced sensitivity set (addendum 1)"
#: addendum 1 item 1-2: the inner design of every V5 design and variant of every learned arm, on every seed -- three
#: inner folds, each ONE inner fit per configuration with all of the fold's inner cells (<= 30) hidden simultaneously
INNER_DESIGN_MODULE = "gen19ct.models.inner_design"
INNER_DESIGN_NAME = "V5_inner_simultaneous"
INNER_N_FOLDS = 3
INNER_MAX_CELLS_PER_FOLD = 30
#: section 15 resolution: the deterministic arms whose conformal inner folds are drawn per discovery seed, per design --
#: the section 5 comparators of the designs a learned arm is scored on with every discovery seed (V5 / V2: B3x, B3i;
#: V1: B3 = B4) plus B0 (every claim is quoted against B0).  V5-P (B4x) is not listed: every learned V5-P comparison is
#: on seed 104729 only, where the resolution keeps the seed-104729 (pre-seal) intervals
COMPARATOR_INTERVAL_JOBS: dict[str, tuple[str, ...]] = {"V5": ("B0", "B3x", "B3i"), "V1": ("B0", "B3"),
                                                        "V2": ("B0", "B3x", "B3i")}
COMPARATOR_INTERVAL_ARMS: tuple[str, ...] = tuple(dict.fromkeys(a for v in COMPARATOR_INTERVAL_JOBS.values() for a in v))
#: the closed-form arm run in both guard modes by the nested-certificate safeguard (the pre-seal cross-check's arm)
SAFEGUARD_PROBE_ARM = "B3x"
KINDS: tuple[str, ...] = ("safeguard", "fit", "comparator_intervals", "marker", "h3_spec")
STEPS: tuple[str, ...] = ("point", "intervals")
V5_REFIT_VARIANTS: tuple[str, ...] = ("loose", "strict", "hno3_only", "cell_only", "parent_structure")
#: addendum 1 item 4: the ONLY V5 refits a learned arm runs (seed 104729) -- V5 strict and V5 HNO3-only
LEARNED_REFIT_VARIANTS: tuple[str, ...] = ("strict", "hno3_only")
#: the V1 / V2 refit sensitivities (``transfer.REGISTERED_SENSITIVITIES``): setting -> (fold-file variant, drop Sr(III),
#: sensitivity name).  ``near_duplicate_key`` / ``compilation_doi`` have exact fold files only; a heavy arm on the ten
#: grouped folds has no fold file or inner design under those groupings (open issue; UNTESTABLE until built)
V1_REFIT_SETTINGS: dict[str, tuple[str, bool, str]] = {
    "sr_iii_dropped": ("copy", True, "sr_iii_dropped_training"),
    "near_duplicate_key": ("near_duplicate_key", False, "near_duplicate_key_groups_value_blind"),
    "compilation_doi": ("compilation_doi", False, "compilation_doi_groups")}
#: the V1 grouping sensitivities: a heavy arm needs outer folds AND an inner design under that grouping (not built)
V1_GROUPING_SETTINGS: tuple[str, ...] = ("near_duplicate_key", "compilation_doi")
V2_REFIT_SETTINGS: dict[str, tuple[str, bool, str]] = {
    "state": ("state", False, "state_level_hiding"),
    "sr_iii_dropped": ("element", True, "sr_iii_dropped_training")}
SR_DROPPED_SUFFIX = "_sr_iii_dropped"
#: fold-file design token -> the registered design label of metrics / transfer
DESIGN_LABEL: dict[str, str] = {"V5": "V5", "V5P": "V5-P", "V5PAIR": "V5-PAIR", "V1": "V1", "V2": "V2", "V0": "V0"}
#: which registered half table each design's rows take their half from (``feasibility_halves.csv``)
HALF_TABLE: dict[str, str] = {"V5": "V5_system", "V5P": "V5_system", "V5PAIR": "V5_system", "V1": "V1",
                              "V2": "V2_state"}
#: the V5 settings a contrast is evaluated on: setting -> the registered R19 item 6 sensitivity name
V5_SETTING_SENSITIVITY: dict[str, str] = {"loose": "loose_setting", "strict": "strict_setting",
                                          "hno3_only": "HNO3_only_cells", "cell_only": "V5-cell-only",
                                          "parent_structure": "parent_structure_hiding",
                                          "sr_iii_dropped": "sr_iii_dropped_training", "V5P": "V5-P"}
#: R19 item 6 sensitivities that re-score existing predictions (section 7 item 2: they apply to every contrast)
SCORING_FILTER_SENSITIVITIES: tuple[str, ...] = ("non_DGA_stratum", "acid_grid_rows_excluded",
                                                 "censoring_candidates_excluded_scoring", ET.WILDCARD_COPY_SENSITIVITY)
REFIT_NOT_RUN = ("not run: section 7 compute plan item 2 evaluates the refit sensitivities only for freezing "
                 "candidates and the registered H1, H1b and H4 contrasts")
#: addendum 1 item 4: for a LEARNED arm these registered refit sensitivities are not run at all; R19 item 6 is
#: evaluated over the refits that were run (V5 strict, V5 HNO3-only) plus every registered scoring-filter
#: sensitivity, and the report names the ones below.  Design -> (sensitivity name -> why)
LEARNED_REFITS_NOT_RUN: dict[str, dict[str, str]] = {
    "V5": {"loose_setting": "addendum 1 item 4: the loose V5 refit is not run for a learned arm",
           "V5-cell-only": "addendum 1 item 4: the V5-cell-only refit is not run for a learned arm",
           "parent_structure_hiding": "addendum 1 item 4: the parent-structure V5 refit is not run for a learned arm",
           "sr_iii_dropped_training": "addendum 1 item 4: the Sr(III)-dropped V5 refit is not run for a learned arm",
           "V5-P": "addendum 1 item 4: the heavy-arm V5-P runs are not run"},
    "V1": {"sr_iii_dropped_training": "addendum 1 item 4: the V1 refit sensitivities are not run for a learned arm",
           "near_duplicate_key_groups_value_blind": "addendum 1 item 4: the V1 refit sensitivities are not run for a "
                                                    "learned arm (and no fold file or inner design exists under that "
                                                    "grouping)",
           "compilation_doi_groups": "addendum 1 item 4: the V1 refit sensitivities are not run for a learned arm (and "
                                     "no fold file or inner design exists under that grouping)"},
    "V2": {"state_level_hiding": "addendum 1 item 4: the state-level V2 refit is not run for a learned arm",
           "sr_iii_dropped_training": "addendum 1 item 4: the V2 refit sensitivities are not run for a learned arm"}}
#: addendum 1 item 3: R19 item 4 in discovery
ITEM4_NOT_EVALUATED = "NOT_EVALUATED"
ITEM4_NOT_EVALUATED_DETAIL = ("addendum 1 item 3: learned arms run discovery seed 104729 only, so seed consistency is "
                              "NOT_EVALUATED in discovery; at confirmation R19 item 4 (5 of 5 withheld seeds) is "
                              "unchanged and required")
#: the arms of the never-demoted registered contrasts (section 7 item 2: their refit sensitivities always run)
H_CONTRAST_ARMS: tuple[str, ...] = ("M2", "B6", "B6r0", "B5", "FLAT_CAT")
#: the S1 families (section 9): reported UNDECIDED when the re-coloured batched-vs-exact check fails (section 7 item 6)
S1_FAMILIES: tuple[str, ...] = ("primary", "S1(b)", "S1(c)")
CHECK_FAILED_LABEL = "batched (check failed)"

READINGS: dict[str, str] = {
    "selection_half_only": "every fitted fold is restricted to its scored rows whose fold row_half is S; the registered "
                           "half of every such row is re-derived from feasibility_halves.csv (V5 family: the system, "
                           "V1: the row's outer-fold unit, V2: the metal state) and a confirmation-half row raises; "
                           "confirmation-half predictions are never written; V0 is not run (its rows span both halves; "
                           "section 10 F1 is evaluated at the report stage)",
    "b6_batched_check": "section 3.1 / 7 item 6 check computed per discovery seed on the selection half (B6 exact vs B6 "
                        "batched folds of that seed, identical scored cells); it passes only when every seed that "
                        "discovery runs passes (the conservative reading of 'same seeds'), which under addendum 1 item "
                        "3 is seed 104729 alone (:data:`PLAN_SEEDS`); both sides of the check tune on the addendum's "
                        "simultaneous inner design, so the check compares outer batching alone",
    "v1_tenfold_check": "section 3.2: B6 exact vs B6 ten-fold per discovery seed (addendum 1: seed 104729), identical "
                        "scored rows, V1 outer-fold unit; a failure moves the heavy arms to the exact design",
    "heavy_arm_intervals": "section 12, cross-fitted (task X finding V-LP-01; needs a POST-HOC addendum): no calibration "
                           "residual comes from a row that chose the configuration or the stopping point it is a "
                           "residual of. Seed 104729 (3 inner folds tuned): the residuals of inner fold j come from "
                           "refits (fixed CatBoost iterations / network epochs, no early stopping) of the configuration "
                           "and iteration / epoch count selected on the OTHER inner folds' recorded validation errors "
                           "(boosted / neural select_excluding_folds), on fold j's inner splits -- under addendum 1 "
                           "items 1-2 every seed that is run is tuned this way, on the 3 simultaneous inner folds, so "
                           "the first-inner-fold calibration path is unused in discovery. Fewer than two inner folds: "
                           "not calibrated (NaN intervals, status "
                           "recorded). B8 (no tuning) calibrates on the tuning splits of its inner mode. B6 / B6r0: "
                           "factorized.B6TunedConformal(calibration='cross_fit'). M2's retained M1 values are the outer "
                           "fold's (selected with every inner fold) -- a second-order reuse disclosed as open. The "
                           "interval centre is the outer refit of the point step",
    "addendum1_inner_design": "POST-HOC addendum 1 items 1-2 (the compute-driven reduction, below the sealed footer): "
                              "every V5 design and variant of every learned arm and of B6 tunes on the SIMULTANEOUS "
                              "inner design (gen19ct.models.inner_design.SimultaneousInnerCells): 3 inner folds, each "
                              "ONE inner fit per configuration with all of that fold's inner cells (<= 30) hidden at "
                              "once under the registered hiding; eligibility is re-checked with all of them hidden and "
                              "a cell that fails stays hidden but is dropped from the inner score. The inner batching "
                              "of sections 3.1 / 7 is not used for tuning. Selection = mean over the 3 inner folds of "
                              "the design's unit-macro MAE with the 0.005 tie rule; the outer refit uses the median "
                              "best iteration / epoch count. V1 and V2 inner designs are unchanged (3 inner folds, one "
                              "fit each). Every seed that is run uses this design, so JobSpec.inner_mode is always "
                              "'full' and the 'first inner fold' reading is gone (FirstInnerFold and "
                              "TuningSplitCalibration(mode='first') are deprecated and unused by the plan)",
    "addendum1_seeds": "POST-HOC addendum 1 item 3: in discovery every learned arm (B5 = M0, FLAT_CAT, B6, B6r0, B8, "
                       "M1-M7) runs discovery seed 104729 only (:data:`PLAN_SEEDS`); the pass over the other four "
                       "discovery seeds is dropped. R19 item 4 is NOT_EVALUATED in every discovery table, and the "
                       "deterministic comparators' conformal intervals are the seed-104729 (pre-seal) ones, since every "
                       "learned comparison is on that seed. At confirmation R19 item 4 is unchanged and required. The "
                       "registered batched-vs-exact and ten-fold checks are evaluated on the seeds discovery runs "
                       "(seed 104729), the conservative reading of 'same seeds' applied to the seeds that exist",
    "addendum1_sensitivities": "POST-HOC addendum 1 item 4: a learned arm runs only the V5 strict and V5 HNO3-only "
                               "refits (seed 104729) for H1 (M2), H1b (B6), H4 (M0, FLAT_CAT, M2, B6, B6r0) and any "
                               "freezing candidate. R19 item 6 of a learned-arm contrast is then evaluated over those "
                               "refits plus every registered scoring-filter sensitivity (non-DGA, censoring-candidate, "
                               "acid-grid, wildcard-copy) and labelled 'reduced sensitivity set (addendum 1)', with the "
                               "sensitivities that were not run named (:data:`LEARNED_REFITS_NOT_RUN`). Such a contrast "
                               "may still be frozen. The closed-form arms keep the full registered set",
    "addendum1_v5pair": "POST-HOC addendum 1 item 5: learned arms run the seed-104729 batched V5-PAIR folds only for M2 "
                        "(with its M1 prerequisite) and for any freezing candidate with a pair endpoint; B3x and B3i "
                        "are re-fitted on the same folds by the scorer (models.s1c_yardsticks, the section 9 S1(c) "
                        "counterweight), which needs no runner job",
    "fold_ordinal": "the section 15 model seed 42 + fold * 1009 + 9,999,991 takes fold = the position of the (discovery "
                    "seed, outer fold) pair when the design is enumerated once per discovery seed in the seed order "
                    "104729, 130363, 155921, 196613, 262147: a seeded fold file contributes that seed's folds in file "
                    "order, a seed-free file (V2, V5 exact, V5-P cell x group) all its folds. Seed 104729 keeps the file "
                    "position (s104729_S_b007 -> 7); the other seeds get distinct numbers, so the discovery seeds drive "
                    "model initialisation (section 15) on every design, V2 included. Every learned arm uses it: B5, "
                    "FLAT_CAT, B8, M1, M2 and the B6 / B6r0 factor initialisation (task X finding V-02; needs a "
                    "POST-HOC addendum)",
    "plan_order": "section 7 item 3's arm order is applied within each pass (addendum 1 reading 6(a)): pass one is "
                  "B6 / B6r0 on the primary exact / batched and V1 / V2 designs (their checks gate the heavy-arm "
                  "schemes), pass two the seed-104729 main designs B5, FLAT_CAT, B8 -> M1 -> M2, then the stop rule "
                  "and the S1(c) V5-PAIR run (M1 prerequisite, M2), pass three the seed-104729 strict / HNO3-only "
                  "refits in arm order -- B6 / B6r0 first, then B5, FLAT_CAT, M1, M2 (task X finding V-F04: the B6 "
                  "refits are refit sensitivities, no check reads them) -- then the conditional freezing-candidate "
                  "runs. Exhausting the section 7 item 5 budget demotes only M3-M7",
    "budget": "section 7 item 5: when the ledger reaches 60 h only M3-M7 are demoted (in the order M7 -> M3); no other "
              "step is named, so every job of this plan (H1, H1b, H4 and the rest) keeps running and the exhaustion is "
              "recorded; --max-hours is an operator pause (resumable), never a demotion",
    "b6_inner_units": "B6 / B6r0 select on the section 4 unit of the design (InnerSplit.row_units: V5 hidden cell, V1 "
                      "publication group or REMAINDER, V2 metal state), the units of the heavy arms, not one value per "
                      "inner split (task X finding V-03)",
    "s1_check_failed": "section 7 item 6: when the re-coloured batched-vs-exact check still fails, every heavy-arm V5 "
                       "contrast carries the label 'batched (check failed)' and every S1 component (primary, S1(b), "
                       "S1(c)) is reported UNDECIDED whatever its R19 verdict; freezing candidates carry both",
    "uncertainty_metrics": "no learned arm has a predictive SD (std_logD NaN; split-conformal intervals only): Gaussian "
                           "CRPS, Spearman(|error|, SD), the SD-binned reliability curve and the 'knows when it does "
                           "not know' test are NOT_RUN, and the section 12 methods other than split-conformal are not "
                           "built (task X finding V-10; needs a POST-HOC addendum or the heads)",
    "m_target_scaling": "M1 / M2 standardise the target with the mean / SD of each fit's own training rows (the outer "
                        "refit: outer-training rows; an inner tuning or calibration fit: its inner-training rows); the "
                        "section 6 resolution's 'outer-training mean and SD' read literally would put inner validation "
                        "targets into inner fits, against section 2 (task X finding V-08; needs a POST-HOC addendum)",
    "record_verification": "every decision reader requires each record's digest and fold hash to equal the ones the "
                           "current code, fold file and plan state produce and the fold ids to be exactly the job's "
                           "fittable folds (a stale or foreign record raises; a job with folds still missing is not "
                           "scored); M2 requires the verified digest of M1's record of the same fold",
    "m2_prerequisite": "M2 searches rank only, with the M1 values retained per outer fold (section 6 resolution), so M1 "
                       "is tuned on every outer fold M2 runs on (V5-PAIR batched, the refit sensitivities) and M2 "
                       "always builds on M1's per-fold values; the ladder compares M2 with the retained predecessor "
                       "(M1 when kept, else M0)",
    "inner_units": "M1 / M2 take their inner splits from the boosted inner design objects (the fold builders' "
                   "functions), so every heavy arm tunes on identical inner splits; the V1 inner averaging unit is the "
                   "V1 unit (publication group, or REMAINDER below 20 rows)",
    "margins": "R19 item 1 margin: delta5 (difficulty.json -> V5.delta5) for H1 primary, H1b, H4, the V5 ladder steps "
               "and every other V5 contrast; 0.05 for S1(b) (M2 vs B0, M2 vs B6r0) and for V1 / V2 contrasts (section "
               "9 names no V1 / V2 margin; the floor of the delta5 formula)",
    "r19_item4_availability": "R19 item 4 needs one Delta per discovery seed. For a learned arm it is NOT_EVALUATED in "
                              "discovery (addendum 1 item 3) and the full verdict is therefore UNDECIDED, never PASS; "
                              "the stop-rule, ladder and freezing scopes do not contain item 4, so decisions are "
                              "unaffected. For any other contrast a missing seed is still recorded NOT_RUN",
    "refit_sensitivities_not_run": "a registered refit sensitivity without predictions is recorded UNTESTABLE ('not "
                                   "run'), so the full R19 verdict is UNDECIDED unless another item fails. For a "
                                   "learned-arm contrast the sensitivities addendum 1 item 4 drops are named instead "
                                   "(:data:`LEARNED_REFITS_NOT_RUN`) and item 6 is decided on the reduced set",
    "decision_scopes": "stop rule: R19 items 1, 2, 3 and 5 (section 7 item 4); ladder and freezing screen: items 1, 2, 3, "
                       "5 and the scoring-filter sensitivities of item 6 on seed 104729 (item 2); V5-P heavy runs: items "
                       "1-5 on V5-primary (section 3.1 resolution); full: items 1-6",
    "wildcard_filter_pairing": "the wildcard-copy scoring filter drops a scored row when its own fold's training rows "
                               "hold a partner (folds/wildcard_copy_crossings.csv, per stem, fold and row); in a paired "
                               "contrast a row dropped for either arm is dropped for both",
    "bh_p": "Benjamini-Hochberg per family on the two-sided bootstrap p of the primary cluster unit on seed 104729 (V5: "
            "system; V1: publication group; V2: metal state); S1(c) enters with its five paired contrasts (three "
            "direction, two logSF MAE; system clusters). p_bh is over the contrasts evaluated; p_bh_full_family "
            "counts every registered discovery contrast of section 19 (m printed), a contrast not run entering as p = 1 "
            "(the conservative bound); neither decides",
    "bh_family_column": "task X finding V-S03: 'BH per family' is the FAMILY column (primary, H1b, S1(b), S1(c), H4, "
                        "ladder, secondary designs), so p_bh is computed within each family over its evaluated "
                        "primary-cluster rows and bh_m is that family's size; the earlier pooled adjustment over all "
                        "evaluated registered contrasts is kept beside it as p_bh_pooled_registered / "
                        "bh_m_pooled_registered, and p_bh_full_family stays the m = 60 section 19 family of reading "
                        "6(f). No verdict moves: R19 and every scope use the raw p (task X finding V-BH-06: 'M2 vs M0' "
                        "and 'M1 vs M0' hold a section 19 slot in family H4 AND in family ladder, so two slots share "
                        "one computed contrast -- flagged bh_shared_computed_contrast)",
    "tost_interval": "task X finding V-S07: the TOST bounds are the CONSERVATIVE ENVELOPE of the 90 % percentile and "
                     "90 % BCa intervals of the primary-cluster bootstrap (transfer.tost, interval='both': low = "
                     "min(percentile, BCa), high = max(percentile, BCa)), so a bound may come from either interval; the "
                     "cluster unit and cluster count are printed beside every TOST interval "
                     "(tost_cluster_unit / tost_n_clusters), because on 4 metal-state clusters the BCa bound rests on "
                     "2^4 distinct resamples",
    "cluster_degenerate": "task X finding V-S04: a registered cluster unit can have exactly one cluster per scoring "
                          "unit -- V1 (39 outer folds, 39 publication-group clusters: 38 folds hold one group each and "
                          "REMAINDER holds the 27 small groups), V2 (4 states, 4 state clusters). The 'clustered' "
                          "bootstrap is then the unclustered unit bootstrap of section 8, which the registration says "
                          "never decides, so such a row is labelled unclustered_equivalent "
                          "(cluster_equals_scoring_unit) and carries no clustering protection beyond the fold",
    "s1c_direction_gate": "section 4's direction gate |observed logSF| >= 0.3 is applied with the module's documented "
                          "float tolerance (evaluation.pairs: '>= threshold - 1e-9'; metrics.FLOAT_TOL), because a "
                          "difference of two stored log_D values that is exactly 0.3 is representable as "
                          "0.29999999999999993. Task X finding V-S05: the tolerance admits boundary pairs (12 Ln-Ln "
                          "pairs of the selection-half V5-PAIR set) and the S1(c) counterweight margin is only 0.0032, "
                          "so the convention and the pair count are recorded with the S1(c) block",
    "s1c_yardstick_records": "task X finding V-S06: the batched B3x / B3i refits of S1(c) ('same fitted folds') are "
                             "closed-form and were re-computed in memory by the scorer, so no record covered them. They "
                             "are now persisted as a record set under evaluation/discovery/_s1c_yardsticks/<stem>/s<seed> "
                             "with a digest over the scorer stage's code digest, the fold design hash, the arms and the "
                             "halves; a later pass recomputes them and refuses a record whose digest or values differ, "
                             "so the S1(c) comparators are inside the record_verification chain",
    "record_sets_not_planned": "task X finding V-REC-07: a record set with no records is 'missing' only when the "
                               "registered plan (enumerate_plan of the current plan state) contains a fit job that "
                               "writes it; otherwise it is 'not_planned' -- the scorer asked for a frame the plan never "
                               "scheduled (B8 has no registered strict / HNO3-only refit: addendum 1 item 4 registers "
                               "them for H1, H1b, H4 and freezing candidates only), which is not an incomplete run",
    "budget_ladder": "POST-HOC addendum 2, '2. Ladder M3-M7' -> Budget: the ladder has its own 40 h wall-clock ledger "
                     "(evaluation/ladder/decisions/wall_clock.json), checked before each step; the discovery ledger "
                     "stays at the registered 60 h for the discovery stages. Task X finding V-BUD-03: the discovery "
                     "ledger reached 76.6 h, so it recorded the section 7 item 5 exhaustion and demoted M3-M7 BY RULE "
                     "(wall_clock.json -> budget.demoted_now) while ledger.demoted stayed empty -- no step was removed "
                     "on evidence, and M3-M7 are also simply not implemented in this runner. Both reasons are recorded "
                     "in decisions.json -> budget, so an absence cannot be attributed to one of them alone",
    "s1d_s1e": "section 9 S1 is 'All of' (a)-(e), so the scorer records a verdict or an explicit NOT_EVALUATED for "
               "every component (task X finding V-S1-01). S1(d) is computed from this scorer's own V5-primary "
               "unit-macro coverage and its coverage by domain-status category (calibration.s1d_check; a category "
               "below 20 scored cells is outside the band's scope). S1(e) is NOT_EVALUATED: its Spearman(cell MAE, "
               "support_score) interval and the support-score components' reliability floor are the power stage "
               "(evaluation.power, 'reliability_not_fitted'), which is not implemented in this runner; the descriptive "
               "Spearman point is printed beside and decides nothing",
    "guard_counters": "task X finding V-MAN-09: the manifest's confirmation_half_read and v6_target_rows_scored are "
                      "DERIVED from what the guards saw in this process (discovery.guard_counts: frames checked, rows "
                      "whose registered half was re-derived, confirmation-half rows seen, V6_TARGET_ROWS rows in a "
                      "scoring index, pre-seal rows read through the parquet half filter), not written as literals",
    "full_family_60": "addendum 1 reading 6(f) counts 60 registered discovery contrasts; section 19 lists 57 discovery "
                      "contrasts and 3 confirmation-only S2 contrasts (60 in all), so the full family is m = 60 with "
                      "the S2 rows entered as p = 1 (the larger m is the conservative bound) and m_discovery = 57 is "
                      "printed beside it (task X finding V-F05; needs POST-HOC addendum 2 to state which count 6(f) "
                      "meant)",
    "calibration_refits": "addendum 1 item 2 says the cross-fitted residuals of inner fold j 'reuse the fits already "
                          "made (no extra fits)'. B6 / B6r0 do (B6TunedConformal.fit_from_tuning). For B5, FLAT_CAT, "
                          "M1, M2 (and B8) the inner fit of fold j chose its tree count / epoch by early stopping on "
                          "fold j's own validation rows, so its residuals on fold j are selection-touched; the runner "
                          "therefore REFITS the configuration and count selected without fold j on fold j's inner "
                          "training rows (CrossFitResidualConformal; 3 refits per outer fold, priced as "
                          "CALIBRATION_FITS_PER_SPLIT = 1, about 10 CPU-h of the addendum estimate). This is the "
                          "statistically conservative reading of the sentence, not its literal one (task X finding "
                          "V-F02; needs POST-HOC addendum 2)",
    "inner_recheck": "addendum 1 item 1 'eligibility is re-checked with all of them hidden': every condition of a cell, "
                     "its own row and publication counts (k, p) included, is counted with the OTHER cells of the inner "
                     "fold hidden and nothing restored -- the reading section 3.1 registered for the identical sentence "
                     "of the batch re-check ('the literal and stricter reading the fold builder applied'), so a cell "
                     "whose rows another cell's component-aware hiding removes stays hidden but is not scored "
                     "(inner_design.REGISTRATION_CHOICES['recheck']; task X finding V-F03; measured on real training "
                     "rows: 1 more cell of 90 dropped on V5 exact Eu(III)__a9ed8f70c7, 0 of 90 on batched "
                     "s104729_S_b000, seed 104729)",
    "inner_wildcard_copies": "section 2's 'copies the key cannot see' (folds/wildcard_copy_pairs.csv: the same "
                             "conditions and log_D under another structure key or state token) can sit in an inner "
                             "split's training rows while their partner is an inner calibration row; the registered "
                             "hiding does not remove them and the near-duplicate key includes the system, so the guard "
                             "passes them. The registration is silent on inner validation, so the inner selection score "
                             "is NOT changed: every V5 fold record counts them per inner fold "
                             "(inner_design.wildcard_copy_partners_in_inner_training) as a diagnostic (task X finding "
                             "VL-A1-02; needs POST-HOC addendum 2 to say whether the registered wildcard exclusion "
                             "applies to inner selection)",
    "addenda_digest_gate": "POST-HOC addendum 2 item 5: the seal gate pins the footer digest, the number of POST-HOC "
                           "addenda AND the SHA-256 of the below-footer text as REGISTERED FOR THE RUNNER'S STAGE in "
                           "manifests/digest_registry.json (gen19ct.evaluation.registry; the constants "
                           "N_ADDENDA_EXPECTED / REGISTERED_ADDENDA_SHA256 only when no registry exists); every fold "
                           "record's resume digest carries the discovery stage's addenda digest and the resolved inner "
                           "design, so a record made under another addendum text or inner design is stale, and a record "
                           "is verified against the registry entry of its stage, never against the live text or live "
                           "code (task X findings V-F01, VL-A1-01)",
    "safeguard_probe": "the section 2 safeguard runs ConformalWrapper(B3x) with nested_certificate and with every_split "
                       "on each drawn outer fold (the pre-seal cross-check's closed-form arm) and compares residuals, "
                       "quantiles and guard verdicts; a difference makes every_split mandatory for that fold file",
    "comparator_intervals": "section 15 resolution as amended by addendum 1 item 3: every learned comparison is on seed "
                            "104729, so the deterministic comparators' conformal intervals are the seed-104729 ones of "
                            "the pre-seal run on every design (the resolution's parenthesis), and no comparator-interval "
                            "job is run in discovery; the point predictions were never seed-dependent. At confirmation "
                            "they are drawn with each withheld seed and averaged, unchanged",
    "h3": "section 11 job specs only (WITH / WITHOUT / ACT_PERMUTED / ACT_METAL_SHUFFLED); not executed by this runner",
    "v2_summary": "V2 contrasts (the secondary-design contrast and the ladder's V2 non-inferiority) are evaluated on the "
                  "section 3.3 primary summary, the focus-7 lanthanides present in the selection half (Ce, Pr, Nd, Gd), "
                  "the summary the V2 comparator was chosen on; every selection-half state is printed beside as an "
                  "exploratory contrast (setting all_states)",
    "power_check": "section 8 signal injection: the plan (which failed H1 / H1b contrasts need it, kappa grid, X(?) rows "
                   "dropped, refits at the selected hyperparameters) is written to decisions.json; its execution is not "
                   "implemented in this runner",
}
#: section 3.3 primary V2 summary
V2_FOCUS7: tuple[str, ...] = ("La(III)", "Ce(III)", "Pr(III)", "Nd(III)", "Sm(III)", "Eu(III)", "Gd(III)")


# --------------------------------------------------------------------------------------------- #
# jobs and the plan
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class JobSpec:
    """One discovery job (module docstring).  ``fold_seed`` filters a multi-seed fold file (batched V5, V1 ten-fold) to
    the folds drawn with that seed; ``seed`` is the run seed (inner designs, initialisation, calibration)."""

    kind: str
    arm: str
    design: str = ""
    variant: str = ""
    scheme: str = ""
    seed: int | None = None
    fold_seed: int | None = None
    drop_sr: bool = False
    stage: str = ""
    group: str = ""
    purpose: str = ""
    registered: bool = True
    condition: str | None = None
    writes: tuple[str, ...] = ()
    stem_override: str = ""
    message: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"job kind {self.kind!r} not in {KINDS}")
        if self.kind in ("fit", "comparator_intervals") and (not self.design or self.seed is None):
            raise ValueError(f"{self.kind} job needs a design and a run seed")
        if not self.writes and self.kind in ("fit", "comparator_intervals"):
            object.__setattr__(self, "writes", (self.arm,))

    @property
    def stem(self) -> str:
        if self.stem_override:
            return self.stem_override
        return FI.design_stem(self.design, self.variant, self.scheme) if self.design else ""

    @property
    def variant_label(self) -> str:
        return f"{self.variant}_{self.scheme}" + (SR_DROPPED_SUFFIX if self.drop_sr else "")

    @property
    def design_dir(self) -> str:
        return f"{self.design}__{self.variant_label}"

    @property
    def key(self) -> str:
        if self.kind == "marker":
            return f"marker:{self.arm}"
        if self.kind == "safeguard":
            return f"safeguard:{self.stem}"
        return f"{self.kind}:{self.arm}:{self.design_dir}:s{self.seed}"

    @property
    def inner_mode(self) -> str:
        """Addendum 1 item 2: three inner folds on every seed that is run, with no 'first inner fold only' reading, so
        every job tunes in ``full`` mode (:data:`READINGS` ``addendum1_inner_design``)."""
        return "full"

    @property
    def design_label(self) -> str:
        return DESIGN_LABEL.get(self.design, self.design)

    def record(self) -> dict[str, Any]:
        rec = asdict(self)
        rec.update(key=self.key, stem=self.stem, variant_label=self.variant_label, inner_mode=self.inner_mode)
        return rec


@dataclass
class PlanState:
    """Decisions the plan depends on (``evaluation/discovery/decisions/plan_state.json``): the B6 checks, the safeguard
    guard modes, the freezing candidates of the scorer."""

    v5_batched_check: str = "pending"          # pending | passed | recolour | passed_after_recolour | failed
    v1_tenfold_check: str = "pending"          # pending | passed | failed
    guard_mode: dict[str, str] = field(default_factory=dict)          # fold stem -> nested_certificate | every_split
    freezing_candidates: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    V5_STATES = ("pending", "passed", "recolour", "passed_after_recolour", "failed")
    V1_STATES = ("pending", "passed", "failed")

    def __post_init__(self) -> None:
        if self.v5_batched_check not in self.V5_STATES:
            raise ValueError(f"v5_batched_check {self.v5_batched_check!r}")
        if self.v1_tenfold_check not in self.V1_STATES:
            raise ValueError(f"v1_tenfold_check {self.v1_tenfold_check!r}")

    @property
    def heavy_v5_scheme(self) -> str | None:
        """The V5 scheme of the heavy arms: ``batched`` after a passed check, ``batched_max4`` after the re-colouring
        (passed or failed), ``None`` while the check is pending or the re-coloured check has not run."""
        return {"passed": "batched", "passed_after_recolour": "batched_max4", "failed": "batched_max4"}.get(
            self.v5_batched_check)

    @property
    def heavy_v5_label(self) -> str:
        return CHECK_FAILED_LABEL if self.v5_batched_check == "failed" else "batched"

    @property
    def s1_forced_undecided(self) -> bool:
        """Section 7 item 6: the re-coloured batched-vs-exact check failed, so S1 is reported UNDECIDED, never passed."""
        return self.v5_batched_check == "failed"

    @property
    def heavy_v1_scheme(self) -> str | None:
        return {"passed": "grouped10", "failed": "exact"}.get(self.v1_tenfold_check)

    @classmethod
    def read(cls, path: Path) -> "PlanState":
        if not Path(path).exists():
            return cls()
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        keep = {k: body[k] for k in ("v5_batched_check", "v1_tenfold_check", "guard_mode", "freezing_candidates",
                                     "notes") if k in body}
        return cls(**keep)

    def record(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "v5_batched_check": self.v5_batched_check, "v1_tenfold_check": self.v1_tenfold_check,
                "guard_mode": dict(sorted(self.guard_mode.items())), "freezing_candidates": list(self.freezing_candidates),
                "notes": dict(self.notes), "heavy_v5_scheme": self.heavy_v5_scheme, "heavy_v5_label": self.heavy_v5_label,
                "heavy_v1_scheme": self.heavy_v1_scheme}


def _fit(arm: str, design: str, variant: str, scheme: str, seed: int, *, stage: str, group: str, purpose: str,
         multi_seed_file: bool = False, drop_sr: bool = False, writes: tuple[str, ...] = (), condition: str | None = None,
         registered: bool = True) -> JobSpec:
    return JobSpec(kind="fit", arm=arm, design=design, variant=variant, scheme=scheme, seed=int(seed),
                   fold_seed=int(seed) if multi_seed_file else None, drop_sr=drop_sr, stage=stage, group=group,
                   purpose=purpose, writes=writes or (arm,), condition=condition, registered=registered)


def _seed_order() -> tuple[int, ...]:
    return (PRIMARY_SEED,) + OTHER_SEEDS


def demotable(job: JobSpec) -> bool:
    """Section 7 item 5: only the ladder steps M3-M7 are demoted when the 60-hour budget is exhausted."""
    return ARM_ALIASES.get(job.arm, job.arm) in DEMOTION_ORDER


SAFEGUARD_STEMS_DEFAULT: tuple[str, ...] = (
    "V5__primary__exact", "V5__primary__batched", "V5__cell_only__batched", "V5__parent_structure__batched",
    "V5P__base__cell_x_group", "V5P__base__batched", "V5PAIR__primary__cell_pair", "V5PAIR__primary__batched",
    "V1__copy__grouped10", "V2__element__exact")


#: the passes, in the order :func:`enumerate_plan` emits them (addendum 1 reading 6(a)).  ``comparators`` carries only a
#: marker now (addendum 1 item 3), and the pass over the other discovery seeds is replaced by the S1(c) V5-PAIR pass
STAGES: dict[str, str] = {
    "safeguard": "00_safeguard", "b6": "01_B6", "comparators": "02_comparator_intervals",
    "p_b5": "03_B5_FLAT_CAT_B8", "p_m1": "04_M1", "p_m2": "05_M2", "stop": "06_stop_rule",
    "s1c": "07_s1c_v5pair", "refit": "08_refit_sensitivities", "candidates": "09_candidates",
    "not_implemented": "10_not_implemented"}


def enumerate_plan(state: PlanState | None = None, *, safeguard_stems: Sequence[str] = SAFEGUARD_STEMS_DEFAULT,
                   include_h3_specs: bool = False, folds_dir: Path | None = None) -> list[JobSpec]:
    """Every discovery job of POST-HOC addendum 1 in section 7 order (module docstring, :data:`READINGS`
    ``plan_order`` and ``addendum1_seeds``).

    Passes: the nested-certificate safeguard -> B6 / B6r0 on seed 104729 with the batched-vs-exact and ten-fold checks
    -> the seed-104729 main designs (B5 = M0, FLAT_CAT, B8 -> M1 -> M2) -> the stop rule -> the S1(c) V5-PAIR run (M1
    prerequisite, M2) -> the seed-104729 strict / HNO3-only refits (B6 / B6r0 first, then the H1 / H4 heavy arms) ->
    the conditional freezing-candidate runs -> ``M3+ not implemented``.  Every fitted arm runs discovery
    seed **104729 only** (addendum 1 item 3), tunes in ``full`` mode (items 1-2), and runs no sensitivity the addendum
    dropped; what is not run is a ``marker`` job naming it, never a silent absence.  Jobs that depend on a decision not
    yet taken are likewise replaced by a marker.  ``folds_dir`` (default the registered folds) is unused by the plan
    now that no V1 grouping-sensitivity job is scheduled; it is kept for callers and for the fold-file checks."""
    st = state or PlanState()
    fdir = paths.FOLDS_DIR if folds_dir is None else Path(folds_dir)
    jobs: list[JobSpec] = []
    for stem in safeguard_stems:
        jobs.append(JobSpec(kind="safeguard", arm=SAFEGUARD_PROBE_ARM, stem_override=stem, stage=STAGES["safeguard"],
                            group="nested-certificate safeguard", purpose="section 2 resolution: every_split vs "
                            "nested_certificate on the registered sample, once at the start of discovery"))
    # ---- B6 / B6r0 (seed 104729 only) and the batched-vs-exact / ten-fold checks
    s1 = STAGES["b6"]
    b6w = B6_ARMS
    tag = "seed 104729 decisions"
    jobs.append(_fit("B6", "V5", "primary", "exact", PRIMARY_SEED, stage=s1, group=f"B6 V5 exact ({tag})", writes=b6w,
                     purpose="H1b B6 vs B3i; H4 B6 - B6r0; S1(b) B6r0; stop rule (section 7 item 4)"))
    jobs.append(_fit("B6", "V5", "primary", "batched", PRIMARY_SEED, stage=s1, group=f"B6 V5 batched ({tag})",
                     writes=b6w, multi_seed_file=True,
                     purpose="section 3.1 registered batched-vs-exact check (section 7 item 6)"))
    jobs.append(_fit("B6", "V1", "copy", "exact", PRIMARY_SEED, stage=s1, group=f"B6 V1 exact ({tag})", writes=b6w,
                     purpose="section 3.2 registered ten-fold-vs-exact check"))
    jobs.append(_fit("B6", "V1", "copy", "grouped10", PRIMARY_SEED, stage=s1, group=f"B6 V1 ten-fold ({tag})",
                     writes=b6w, multi_seed_file=True, purpose="section 3.2 registered ten-fold-vs-exact check"))
    jobs.append(_fit("B6", "V2", "element", "exact", PRIMARY_SEED, stage=s1, group="B6 V2 (seed 104729)", writes=b6w,
                     registered=False, purpose="reference arm (section 11 H3 references; descriptive)"))
    if st.v5_batched_check in ("recolour", "passed_after_recolour", "failed"):
        jobs.append(_fit("B6", "V5", "primary", "batched_max4", PRIMARY_SEED, stage=s1, writes=b6w, multi_seed_file=True,
                         condition="b6_check_recolour", group="B6 V5 re-coloured batches (section 7 item 6)",
                         purpose="the repeated batched-vs-exact check with <= 4 cells per batch"))
    # ---- the deterministic comparators' conformal intervals (section 15 resolution as amended by addendum 1 item 3)
    jobs.append(JobSpec(kind="marker", arm="comparator_intervals", stage=STAGES["comparators"],
                        group="deterministic comparators", message=READINGS["comparator_intervals"]))
    # ---- heavy arms (seed 104729)
    v5s, v1s = st.heavy_v5_scheme, st.heavy_v1_scheme
    blocked: list[str] = []
    if v5s is None:
        blocked.append(f"heavy-arm V5 jobs wait for the B6 batched-vs-exact check (state {st.v5_batched_check!r})")
    if v1s is None:
        blocked.append(f"heavy-arm V1 jobs wait for the B6 ten-fold check (state {st.v1_tenfold_check!r})")

    def heavy_main(arm: str, stage: str, purpose: str) -> None:
        if v5s is not None:
            jobs.append(_fit(arm, "V5", "primary", v5s, PRIMARY_SEED, stage=stage, group=f"{arm} main designs (seed "
                             "104729)", multi_seed_file=True, purpose=purpose))
        if v1s is not None:
            jobs.append(_fit(arm, "V1", "copy", v1s, PRIMARY_SEED, stage=stage, group=f"{arm} main designs (seed "
                             "104729)", multi_seed_file=v1s != "exact", purpose=purpose))
        jobs.append(_fit(arm, "V2", "element", "exact", PRIMARY_SEED, stage=stage,
                         group=f"{arm} main designs (seed 104729)", purpose=purpose))

    def heavy_refit(arm: str, stage: str, purpose: str, group: str) -> None:
        """Addendum 1 item 4: the V5 strict and HNO3-only refits of a learned arm, seed 104729."""
        if v5s is None:
            return
        for v in LEARNED_REFIT_VARIANTS:
            jobs.append(_fit(arm, "V5", v, v5s, PRIMARY_SEED, stage=stage, group=group, multi_seed_file=True,
                             purpose=f"{purpose}: R19 item 6 {V5_SETTING_SENSITIVITY[v]} (addendum 1 item 4)"))

    def pair_run(arm: str, stage: str, purpose: str, group: str, condition: str | None = None) -> None:
        """Addendum 1 item 5: a learned arm on the seed-104729 batched V5-PAIR folds (M2 needs M1's per-fold values)."""
        for a in (("M1", arm) if arm == "M2" else (arm,)):
            jobs.append(_fit(a, "V5PAIR", "primary", "batched", PRIMARY_SEED, stage=stage, multi_seed_file=True,
                             condition=condition,
                             group=group if a == arm else "M1 prerequisites of M2 (per-fold M1 values)",
                             purpose=purpose if a == arm else "M2 on V5-PAIR needs M1's per-fold values"))

    purposes = {"B5": "M0 = B5: ladder base, H4 M2 - M0", "FLAT_CAT": "H4 M2 - FLAT_CAT",
                "B8": "section 5 previous-generation baseline", "M1": "ladder step M1 vs M0; M2's per-fold M1 values",
                "M2": "H1 primary; S1(a), S1(b); H4; ladder step M2"}
    for arm in ("B5", "FLAT_CAT", "B8"):
        heavy_main(arm, STAGES["p_b5"], purposes[arm])
    heavy_main("M1", STAGES["p_m1"], purposes["M1"])
    heavy_main("M2", STAGES["p_m2"], purposes["M2"])
    for msg in blocked:
        jobs.append(JobSpec(kind="marker", arm=f"blocked:{msg}", stage=STAGES["p_b5"], group="blocked", message=msg))
    jobs.append(JobSpec(kind="marker", arm="not_run:other_seeds", stage=STAGES["p_m2"], group="not run (addendum 1)",
                        message="addendum 1 item 3: the pass over the discovery seeds 130363, 155921, 196613 and "
                                "262147 is NOT run; every learned arm runs seed 104729 only and R19 item 4 is "
                                "NOT_EVALUATED in discovery (it is unchanged and required at confirmation)"))
    jobs.append(JobSpec(kind="marker", arm="stop_rule", stage=STAGES["stop"], group="stop rule",
                        message="section 7 item 4 stop rule: evaluated by scripts/g19_score_discovery.py "
                                "(decisions/stop_rule.json) on seed 104729, R19 items 1-3 and 5 of M2 and B6 vs B3i"))
    # ---- S1(c): the seed-104729 batched V5-PAIR folds (addendum 1 item 5)
    s1c = STAGES["s1c"]
    pair_run("M2", s1c, "S1(c) selection-half counterweight on the seed-104729 batched V5-PAIR folds (section 9)",
             "M2 V5-PAIR (seed 104729)")
    jobs.append(JobSpec(kind="marker", arm="s1c_yardsticks", stage=s1c, group="S1(c) yardsticks",
                        message="section 9 S1(c) 'same fitted folds': B3x and B3i are re-fitted on exactly these "
                                "batched V5-PAIR folds by scripts/g19_score_discovery.py "
                                "(models.s1c_yardsticks.refit_lookup_yardsticks; closed form, no runner job), and "
                                "HEAVIER needs no fit. Addendum 1 item 5: no other learned arm runs V5-PAIR unless it "
                                "is a freezing candidate with a pair endpoint"))
    # ---- the seed-104729 strict / HNO3-only refits (addendum 1 item 4), the third pass of reading 6(a): section 7
    # item 3's arm order within it puts B6 / B6r0 first (task X finding V-F04: no B6 check reads a refit)
    for v in LEARNED_REFIT_VARIANTS:
        jobs.append(_fit("B6", "V5", v, "exact", PRIMARY_SEED, stage=STAGES["refit"],
                         group="B6 refit sensitivities (seed 104729)", writes=b6w,
                         purpose=f"R19 item 6 {V5_SETTING_SENSITIVITY[v]} of H1b / H4 (addendum 1 item 4)"))
    jobs.append(JobSpec(kind="marker", arm="not_run:B6_refit_sensitivities", stage=STAGES["refit"],
                        group="not run (addendum 1)",
                        message="addendum 1 item 4: B6 / B6r0 run the V5 strict and HNO3-only refits only; the loose, "
                                "cell-only, parent-structure and Sr(III)-dropped V5 refits and the unbatched V5-P run "
                                "are NOT run, and R19 item 6 of a B6 contrast is the reduced sensitivity set"))
    heavy_refit("B5", STAGES["refit"], "H4 M2 - M0", "B5 refit sensitivities (seed 104729)")
    heavy_refit("FLAT_CAT", STAGES["refit"], "H4 M2 - FLAT_CAT", "FLAT_CAT refit sensitivities (seed 104729)")
    heavy_refit("M1", STAGES["refit"], "prerequisite of the M2 refit sensitivities",
                "M1 prerequisites of M2 (per-fold M1 values)")
    heavy_refit("M2", STAGES["refit"], "H1 / H4", "M2 refit sensitivities (seed 104729)")
    jobs.append(JobSpec(kind="marker", arm="not_run:learned_refit_sensitivities", stage=STAGES["refit"],
                        group="not run (addendum 1)",
                        message="addendum 1 item 4: for a learned arm the loose, cell-only, parent-structure and "
                                "Sr(III)-dropped V5 refits, the heavy-arm V5-P runs and the V1 / V2 refit "
                                "sensitivities (state-level V2, Sr(III)-dropped, near-duplicate-key and "
                                "compilation-DOI groupings) are NOT run; R19 item 6 is the reduced sensitivity set "
                                "(addendum 1) and the report names them"))
    # ---- conditional: freezing candidates (section 3.1 resolution, section 7 items 2 and 7; addendum 1 items 4-5)
    sc = STAGES["candidates"]
    for cand in st.freezing_candidates:
        contrast = str(cand.get("contrast"))
        design = cand.get("design") or _design_of_contrast_key(contrast)
        arms = [ARM_ALIASES.get(a, a) for a in cand.get("arms", ())]
        heavy = [a for a in arms if a in HEAVY_ARMS]
        need = set(heavy) | ({"M1"} if "M2" in heavy else set())
        if design == "V5":
            for arm in [a for a in HEAVY_ARMS if a in need and a not in H_CONTRAST_ARMS]:
                heavy_refit(arm, sc, f"freezing candidate {contrast}", "freezing-candidate refit sensitivities")
            jobs.append(JobSpec(kind="marker", arm=f"not_run:V5P:{contrast}", stage=sc, group="not run (addendum 1)",
                                message=f"{contrast}: addendum 1 item 4: the heavy-arm V5-P run of a freezing "
                                        "candidate is NOT run (V5-P is named among the sensitivities not run); "
                                        "section 3.1's trigger, R19 items 1-5 on V5-primary, cannot fire either, "
                                        "because item 4 is NOT_EVALUATED in discovery. Addendum 3 item 1 changes "
                                        "what 'passed R19 in discovery' means for FREEZING (section 15), not this "
                                        "trigger, so an eligible_for_freezing contrast still runs no V5-P job"))
        elif design in ("V5-PAIR", "V5PAIR"):
            for arm in [a for a in HEAVY_ARMS if a in need]:
                pair_run(arm, sc, f"V5-PAIR endpoint of the freezing candidate {contrast} (addendum 1 item 5)",
                         "freezing-candidate V5-PAIR runs", condition="freezing_candidate")
        elif design in ("V1", "V2"):
            jobs += _v1v2_refit_markers(design, sorted(need), stage=sc, contrast=contrast)
    jobs.append(JobSpec(kind="marker", arm="M3+", stage=STAGES["not_implemented"], group="ladder M3-M7",
                        message=NOT_IMPLEMENTED))
    if include_h3_specs:
        jobs += h3_job_specs()
    return dedupe_jobs(jobs)


def _design_of_contrast_key(key: str) -> str:
    """``"M2 vs B3@V1#ladder"`` -> ``"V1"``."""
    return str(key).split("@", 1)[1].split("#", 1)[0] if "@" in str(key) else ""


def _v1v2_refit_markers(design: str, arms: Sequence[str], *, stage: str, contrast: str) -> list[JobSpec]:
    """Addendum 1 item 4: a learned arm runs NO V1 / V2 refit sensitivity, not even as a freezing candidate.  One marker
    per (arm, sensitivity) so that the plan names every sensitivity that is not run instead of omitting it silently.
    The near-duplicate-key and compilation-DOI groupings have in addition no heavy-arm fold file and no inner design."""
    table = V1_REFIT_SETTINGS if design == "V1" else V2_REFIT_SETTINGS
    out: list[JobSpec] = []
    for arm in arms:
        for setting, (_, _, name) in table.items():
            extra = ("; the heavy-arm fold file and an inner design under that grouping are not built either"
                     if design == "V1" and setting in V1_GROUPING_SETTINGS else "")
            out.append(JobSpec(kind="marker", arm=f"not_run:{arm}:{design}:{setting}", stage=stage,
                               group="not run (addendum 1)",
                               message=f"{contrast}: R19 item 6 {name} of {arm} on {design} is NOT run -- "
                                       f"{LEARNED_REFITS_NOT_RUN[design][name]}{extra}"))
    return out


def dedupe_jobs(jobs: Sequence[JobSpec]) -> list[JobSpec]:
    """First occurrence of every job key (a conditional duplicate never runs twice)."""
    seen, out = set(), []
    for j in jobs:
        if j.key in seen:
            continue
        seen.add(j.key)
        out.append(j)
    return out


def filter_jobs(jobs: Sequence[JobSpec], only: str | None) -> list[JobSpec]:
    """``--only`` tokens, comma separated: an arm (``B6``, ``M2``, ``M0`` = ``B5``), a design token (``V5``, ``V1``,
    ``V2``, ``V5P``, ``V5PAIR``), ``arm:design``, or a kind (``safeguard``, ``comparator_intervals``).  Tokens of one
    kind are OR-ed; kinds are AND-ed.  Markers always pass."""
    if not only:
        return list(jobs)
    def arm_of(a: str) -> str:
        a = ARM_ALIASES.get(a, a)
        return "B6" if a == "B6r0" else a                     # B6r0 is written by the B6 job
    toks = [t.strip() for t in only.split(",") if t.strip()]
    arms, designs, pairs, kinds = set(), set(), set(), set()
    for t in toks:
        if ":" in t:
            a, d = t.split(":", 1)
            pairs.add((arm_of(a), d))
        elif t in KINDS:
            kinds.add(t)
        elif t in DESIGN_LABEL:
            designs.add(t)
        else:
            arms.add(arm_of(t))
    out = []
    for j in jobs:
        if j.kind == "marker":
            out.append(j)
            continue
        if kinds and j.kind not in kinds:
            continue
        if (arms or pairs) and not (j.arm in arms or (j.arm, j.design) in pairs):
            continue
        if designs and j.design not in designs:
            continue
        out.append(j)
    return out


# --------------------------------------------------------------------------------------------- #
# folds, halves and the scoring guards (section 2)
# --------------------------------------------------------------------------------------------- #

def fold_ordinals(folds: Sequence[FI.Fold], seed: int | None = None,
                  seed_set: Sequence[int] = DISCOVERY_SEEDS) -> dict[str, int]:
    """Fold id -> the section 15 model fold number (:data:`READINGS` ``fold_ordinal``): the position of the (seed, fold)
    pair in the design enumerated once per seed of ``seed_set`` (then any other seed of the file, sorted).  A seeded
    file contributes each seed's folds in file order; a seed-free file contributes all its folds for every seed, so the
    number depends on the run ``seed`` (a seed outside ``seed_set`` is enumerated after it).  ``seed=None``: a seeded
    file maps every fold (each under its own seed); a seed-free file takes the first seed of ``seed_set``."""
    folds = list(folds)
    order = list(seed_set)
    file_seeds = {f.seed for f in folds}
    if file_seeds - {None} and None in file_seeds:
        raise ValueError("a fold file mixes seeded and seed-free folds")
    if file_seeds - {None}:
        order += sorted(x for x in file_seeds if x is not None and x not in order)
        numbers: dict[tuple[int, str], int] = {}
        pos = 0
        for sd in order:
            for f in folds:
                if f.seed == sd:
                    numbers[(sd, f.fold_id)] = pos
                    pos += 1
        if seed is None:
            return {f.fold_id: numbers[(f.seed, f.fold_id)] for f in folds}
        return {f.fold_id: numbers[(int(seed), f.fold_id)] for f in folds if f.seed == int(seed)}
    if seed is not None and int(seed) not in order:
        order.append(int(seed))
    idx = 0 if seed is None else order.index(int(seed))
    return {f.fold_id: idx * len(folds) + k for k, f in enumerate(folds)}


def selection_scored_ids(fold: FI.Fold) -> tuple[str, ...]:
    """The fold's scored rows of the selection half (``row_half`` S; a fold without row halves uses its own half)."""
    return tuple(r for r in fold.scored_row_ids if str(fold.row_half.get(r, fold.half)) == SELECTION)


def job_folds(job: JobSpec, folds: Sequence[FI.Fold]) -> list[tuple[FI.Fold, int]]:
    """``(fold, ordinal)`` of every fold the job fits: the job's fold seed (multi-seed files), not a confirmation-half
    fold, and at least one selection-half scored row."""
    seeds = {f.seed for f in folds}
    if job.fold_seed is None and seeds != {None}:
        if len(seeds) != 1:
            raise ValueError(f"{job.key}: fold file {job.stem} holds seeds {sorted(map(str, seeds))}; give fold_seed")
    key_seed = job.fold_seed if job.fold_seed is not None else (job.seed if seeds == {None} else next(iter(seeds)))
    ords = fold_ordinals(folds, key_seed)
    out = []
    for f in folds:
        if job.fold_seed is not None and f.seed != job.fold_seed:
            continue
        if f.half == CONFIRMATION:
            continue
        if selection_scored_ids(f):
            out.append((f, ords[f.fold_id]))
    return out


def fittable_folds(job: JobSpec, folds: Sequence[FI.Fold], excluded_ids: Iterable[str]) -> list[tuple[FI.Fold, int]]:
    """:func:`job_folds` minus the folds whose selection-half scored rows are all ``excluded_ids`` (the acidic
    co-extractant rows): the folds the runner fits and every verified reader expects."""
    ex = {str(r) for r in excluded_ids}
    return [(f, k) for f, k in job_folds(job, folds) if any(str(r) not in ex for r in selection_scored_ids(f))]


def registered_row_halves(frame: pd.DataFrame, design: str, *, halves: Mapping[str, Mapping[str, str]] | None = None
                          ) -> pd.Series:
    """The registered half (``S`` / ``C`` / ``NA``) of every row of ``frame`` under ``design``'s half table (V5 family:
    the system's half; V1: the row's outer-fold unit; V2: the metal state), independent of any fold file."""
    table = HALF_TABLE.get(design)
    if table is None:
        raise ValueError(f"design {design!r} has no registered halves")
    h = dict((halves or {}).get(table) or FI.registered_halves(table))
    if table == "V5_system":
        return frame[SG.SYSTEM_COL].map(h).fillna("NA").astype(str)
    if table == "V2_state":
        return frame[SG.METAL_COL].map(h).fillna("NA").astype(str)
    return SH.row_halves(frame, h).astype(str)


#: what the section 2 guards SAW in this process (task X finding V-MAN-09): a manifest field derived from these is a
#: measurement, not a literal.  ``scoring_frame`` and :func:`read_preseal_selection` increment them; the guards
#: themselves still raise, so a non-zero ``confirmation_half_rows_seen`` / ``v6_target_rows_scored`` cannot survive a
#: completed pass -- the counter proves the guard was reached, the raise proves nothing slipped through
GUARD_COUNTS: dict[str, int] = {}


def reset_guard_counts() -> dict[str, int]:
    """Zero :data:`GUARD_COUNTS` (called once at the start of a scoring pass) and return it."""
    GUARD_COUNTS.clear()
    GUARD_COUNTS.update({"scoring_frames_checked": 0, "rows_half_rederived": 0, "confirmation_half_rows_seen": 0,
                         "v6_target_rows_scored": 0, "preseal_files_read": 0, "preseal_rows_read": 0,
                         "preseal_non_selection_rows": 0})
    return GUARD_COUNTS


def guard_counts() -> dict[str, int]:
    """A copy of :data:`GUARD_COUNTS` (empty until :func:`reset_guard_counts`)."""
    return dict(GUARD_COUNTS)


def _count_guard(key: str, n: int = 1) -> None:
    if GUARD_COUNTS:
        GUARD_COUNTS[key] = GUARD_COUNTS.get(key, 0) + int(n)


def assert_selection_rows(ids: Iterable[str], row_half: Mapping[str, str] | pd.Series, what: str) -> None:
    """Raise ``AssertionError`` unless every row id's registered half is the selection half (discovery never scores
    the confirmation half)."""
    ids = list(ids)
    get = row_half.get if isinstance(row_half, Mapping) else (lambda r, d=None: row_half.get(r, d))
    bad = [r for r in ids if get(r, "NA") != SELECTION]
    if bad:
        halves = sorted({str(get(r, "NA")) for r in bad})
        raise AssertionError(f"{what}: {len(bad)} scored row(s) outside the selection half ({halves}; first {bad[0]!r}); "
                             "discovery never scores the confirmation half")


def guard_mode_source(stem: str) -> str:
    """The safeguard sample (``folds/INDEX.json``) whose verdict governs a fold file's inner guard mode: the file itself
    when sampled; loose / strict / HNO3-only take the primary file of their scheme (they build the certificate as the
    primary does, ``folds.safeguard`` module docstring); cell-only and parent-structure exact files take their batched
    sample; a re-coloured (``batched_max4``) file takes its registered batched file; V1 and V2 files take their (vacuous)
    sample."""
    parts = str(stem).split("__")
    if len(parts) != 3:
        raise ValueError(f"not a fold stem: {stem!r}")
    design, variant, scheme = parts
    scheme = scheme.replace("_max4", "") if scheme.startswith("batched") else scheme
    if design in ("V5P", "V5PAIR"):
        return f"{design}__{variant}__{scheme}"
    if design == "V5":
        if variant in ("cell_only", "parent_structure"):
            return f"V5__{variant}__batched"
        return "V5__primary__exact" if scheme == "exact" else "V5__primary__batched"
    if design == "V1":
        return "V1__copy__grouped10"
    if design == "V2":
        return "V2__element__exact"
    return stem


def assert_v6_clean(labels: Iterable[Any], v6_mask: pd.Series, what: str) -> None:
    """``registered.assert_not_scored`` (no ``V6_TARGET_ROWS`` row in a scoring index)."""
    FR.assert_not_scored(labels, v6_mask, what)


def assert_no_provenance_features(columns: Iterable[str], what: str) -> None:
    """Section 2 / brief section 12: no provenance column, publication / study id, DOI or row id is a feature -- checked on
    the bare and block-prefixed names (``features.assert_feature_columns_allowed`` is applied by the arms as well)."""
    from gen19ct.models import features as F

    cols = [str(c) for c in columns]
    forbidden = (set(LOAD.PROVENANCE_COLUMNS) | set(F.ID_COLUMNS) | {I.PUB_GROUP_COL, FI.GROUP_COL, SG.PUB_COL, I.ID_COL,
                 "row_id", "fold_id"})
    bare = set()
    for c in cols:
        b = c.split("__", 1)[1] if "__" in c else c
        bare.add(b.split("=", 1)[0].removesuffix("__missing"))
    hit = sorted((set(cols) | bare) & forbidden)
    if hit:
        raise AssertionError(f"{what}: provenance / publication / row-id columns among the features: {hit}")
    F.assert_feature_columns_allowed(cols, what)


# --------------------------------------------------------------------------------------------- #
# records, digests and resume (section 16)
# --------------------------------------------------------------------------------------------- #

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.()\-]+$")


def safe_fold_name(fold_id: str) -> str:
    """A file-name-safe fold id (unchanged when already safe; otherwise sanitised plus a short hash)."""
    s = str(fold_id)
    if _SAFE_NAME.match(s) and len(s) <= 120:
        return s
    return re.sub(r"[^A-Za-z0-9_.()\-]", "_", s)[:80] + "__" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def discovery_root(out_root: Path) -> Path:
    return Path(out_root) / "evaluation" / "discovery"


def fold_paths(out_root: Path, job: JobSpec, arm: str, fold_id: str) -> tuple[Path, Path]:
    """``(parquet, json)`` of one arm's record of one outer fold."""
    d = discovery_root(out_root) / arm / job.design_dir / f"s{job.seed}"
    name = safe_fold_name(fold_id)
    return d / f"{name}.parquet", d / f"{name}.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_digest(files: Iterable[Path], objects: Iterable[Any] = ()) -> dict[str, Any]:
    """Digest of the prediction-affecting code: the LF-normalised SHA-256 of every file and the source of every object
    (``inspect.getsource``); ``combined`` is the SHA-256 of the sorted parts."""
    parts: dict[str, str] = {}
    for p in sorted({Path(f).resolve() for f in files}):
        try:
            key = paths.rel(p)
        except ValueError:
            key = p.name
        # a listed file that does not exist yet (a module still being written) is recorded as absent, so the digest
        # changes -- and every record becomes stale, as it must -- the moment it lands
        parts[key] = (hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest() if Path(p).exists()
                      else "absent")
    for obj in objects:
        name = f"{obj.__module__}.{getattr(obj, '__qualname__', getattr(obj, '__name__', repr(obj)))}"
        parts[name] = sha256_text(inspect.getsource(obj).replace("\r\n", "\n"))
    return {"parts": dict(sorted(parts.items())),
            "combined": sha256_text("\n".join(f"{k}={v}" for k, v in sorted(parts.items())))}


#: job fields that label a job without changing what it predicts (left out of the resume digest)
LABEL_FIELDS: tuple[str, ...] = ("stage", "group", "purpose", "registered", "condition", "message")


def fold_digest(job: JobSpec, fold: FI.Fold, code: str, extra: Mapping[str, Any] | None = None) -> str:
    """The resume key of one fold: code digest, the prediction-affecting job fields (:data:`LABEL_FIELDS` left out),
    fold id and hash, plus ``extra`` (the runner's guard mode, batching label, model fold number and seed, design hash,
    registration digest, M1's expected digest)."""
    body = {"code": code, "job": {k: v for k, v in job.record().items() if k not in LABEL_FIELDS},
            "fold_id": fold.fold_id, "fold_hash": fold.fold_hash, "extra": dict(extra or {})}
    return sha256_text(json.dumps(body, sort_keys=True, default=str))


def job_from_record(rec: Mapping[str, Any]) -> JobSpec:
    """The :class:`JobSpec` a record was written by (its ``job`` block)."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(JobSpec)}
    body = {k: v for k, v in dict(rec["job"]).items() if k in names}
    body["writes"] = tuple(body.get("writes") or ())
    return JobSpec(**body)


def read_record(path: Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def resume_status(job: JobSpec, fold: FI.Fold, out_root: Path, digest: str, steps: Sequence[str]) -> dict[str, Any]:
    """Whether every arm the job writes already holds a record of this fold with this digest and every step of
    ``steps``.  ``stale`` names arms with a record of another digest (recomputed, and reported)."""
    done, stale, missing = [], [], []
    for arm in job.writes:
        pq, js = fold_paths(out_root, job, arm, fold.fold_id)
        rec = read_record(js)
        if rec is None or not pq.exists():
            missing.append(arm)
        elif rec.get("digest") != digest:
            stale.append(arm)
        elif not all(s in (rec.get("steps") or {}) for s in steps):
            missing.append(arm)
        else:
            done.append(arm)
    return {"complete": len(done) == len(job.writes), "done": done, "stale": stale, "missing": missing,
            "steps_done": {a: sorted((read_record(fold_paths(out_root, job, a, fold.fold_id)[1]) or {}).get("steps", {}))
                           for a in job.writes}}


def peak_rss_bytes() -> int | None:
    """Peak resident set size of this process (Windows ``PeakWorkingSetSize``; ``ru_maxrss`` elsewhere)."""
    if sys.platform == "win32":
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        try:
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return int(pmc.PeakWorkingSetSize)
        except (AttributeError, OSError):
            return None
        return None
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r if sys.platform == "darwin" else r * 1024)
    except (ImportError, OSError):
        return None


#: columns of every discovery prediction parquet (one row per selection-half scored row of one fold)
PREDICTION_COLUMNS: tuple[str, ...] = (
    "row_id", "fold_id", "arm", "design", "variant", "scheme", "variant_label", "seed", "fold_seed", "half", "unit",
    "mean_logD", "std_logD", "lower_50", "upper_50", "lower_80", "upper_80", "lower_95", "upper_95", "conformal_q50",
    "conformal_q80", "conformal_q95", "conformal_n_calibration", "fallback_level", "fallback_reason", "selected_config",
    "fold_ordinal", "model_seed", "fit_seconds", "intervals_status", "batching_label")
_INTERVAL_COLUMNS = ("lower_50", "upper_50", "lower_80", "upper_80", "lower_95", "upper_95", "conformal_q50",
                     "conformal_q80", "conformal_q95")


def prediction_frame(pred: pd.DataFrame, *, job: JobSpec, arm: str, fold: FI.Fold, ordinal: int, row_ids: Sequence[str],
                     selected_config: str, model_seed: int | None, intervals_status: str,
                     batching_label: str = "", fit_seconds: float = float("nan")) -> pd.DataFrame:
    """The registered record layout (:data:`PREDICTION_COLUMNS`) of one arm's predictions of one fold's selection-half
    rows; ``pred`` is an arm prediction frame in ``row_ids`` order (``interface.PREDICTION_COLUMNS`` at least)."""
    if len(pred) != len(row_ids):
        raise AssertionError(f"{arm}/{fold.fold_id}: {len(pred)} predictions for {len(row_ids)} rows")
    mean = pd.to_numeric(pred["mean_logD"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(mean).all():
        raise AssertionError(f"{arm}/{fold.fold_id}: non-finite prediction")
    out = pd.DataFrame({"row_id": [str(r) for r in row_ids]})
    out["fold_id"] = fold.fold_id
    out["arm"] = arm
    out["design"], out["variant"], out["scheme"] = job.design, job.variant, job.scheme
    out["variant_label"] = job.variant_label
    out["seed"] = int(job.seed)
    out["fold_seed"] = -1 if fold.seed is None else int(fold.seed)
    out["half"] = [str(fold.row_half.get(r, fold.half)) for r in out["row_id"]]
    default_unit = fold.units[0] if fold.units else ""
    out["unit"] = [str(fold.row_unit.get(r, default_unit)) for r in out["row_id"]]
    out["mean_logD"] = mean
    for c in ("std_logD",) + _INTERVAL_COLUMNS:
        out[c] = pd.to_numeric(pred[c], errors="coerce").to_numpy(dtype=float) if c in pred.columns else np.nan
    out["conformal_n_calibration"] = (pd.to_numeric(pred["conformal_n_calibration"], errors="coerce").to_numpy(dtype=float)
                                      if "conformal_n_calibration" in pred.columns else np.nan)
    out["fallback_level"] = pred["fallback_level"].astype(object).to_numpy() if "fallback_level" in pred.columns else arm
    out["fallback_reason"] = (pred["fallback_reason"].astype(object).to_numpy() if "fallback_reason" in pred.columns
                              else None)
    out["selected_config"] = str(selected_config)
    out["fold_ordinal"] = int(ordinal)
    out["model_seed"] = -1 if model_seed is None else int(model_seed)
    out["fit_seconds"] = float(fit_seconds)
    out["intervals_status"] = intervals_status
    out["batching_label"] = batching_label
    if (out["half"] != SELECTION).any():
        raise AssertionError(f"{arm}/{fold.fold_id}: a prediction of a non-selection-half row")
    return out[list(PREDICTION_COLUMNS)]


def attach_intervals(frame: pd.DataFrame, quantiles: Mapping[float, float], n_calibration: int) -> pd.DataFrame:
    """``mean +- q_level`` on a stored prediction frame (the conformal interval of section 12)."""
    out = frame.copy()
    mean = out["mean_logD"].to_numpy(dtype=float)
    for lv in I.LEVELS:
        pct = int(round(lv * 100))
        q = float(quantiles[lv])
        out[f"lower_{pct}"] = mean - q
        out[f"upper_{pct}"] = mean + q
        out[f"conformal_q{pct}"] = q
    out["conformal_n_calibration"] = float(n_calibration)
    out["intervals_status"] = "split_conformal_inner"
    return out


# --------------------------------------------------------------------------------------------- #
# inner splits for calibration (section 12, compute plan item 1)
# --------------------------------------------------------------------------------------------- #

class FixedInnerSplits:
    """A splitter returning inner splits drawn once (the addendum 1 design is drawn once per outer fold and shared by
    tuning and calibration, so no arm redraws it)."""

    def __init__(self, splits: Sequence[I.InnerSplit], name: str = "fixed_inner_splits"):
        self._splits = list(splits)
        self.name = name

    def splits(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> list[I.InnerSplit]:
        return list(self._splits)


class InnerFoldSubset:
    """A splitter restricted to the inner folds ``folds`` (addendum 1 item 2: the cross-fitted calibration of inner
    fold ``j`` uses that fold's splits only).  Works for any splitter whose ``splits`` yields objects with ``fold``."""

    def __init__(self, splitter: Any, folds: Iterable[int]):
        self.splitter = splitter
        self.folds = tuple(sorted({int(f) for f in folds}))
        self.name = f"inner_folds({getattr(splitter, 'name', type(splitter).__name__)}, {list(self.folds)})"

    def splits(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> list[I.InnerSplit]:
        return [s for s in self.splitter.splits(table, mask, context) if int(s.fold) in self.folds]


class FirstInnerFold:
    """DEPRECATED (POST-HOC addendum 1 item 2: three inner folds on every seed, no 'first inner fold only' reading).
    Kept for the sealed-plan tests and for any caller that still needs it; the discovery plan never builds one.

    A splitter restricted to its lowest inner fold index holding a split (the sealed section 7 compute plan item 1)."""

    def __init__(self, splitter: Any):
        self.splitter = splitter
        self.name = f"first_inner_fold({getattr(splitter, 'name', type(splitter).__name__)})"

    def splits(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> list[I.InnerSplit]:
        sp = self.splitter.splits(table, mask, context)
        if not sp:
            return []
        first = min(int(s.fold) for s in sp)
        return [s for s in sp if int(s.fold) == first]


class TuningSplitCalibration:
    """The inner splits of a boosted inner design object (``boosted.V5InnerTuning`` / ``V1InnerTuning`` /
    ``V2InnerTuning``, built on the fold builders' functions) as ``interface.InnerSplit`` s, so a heavy arm's conformal
    calibration uses exactly the inner splits its tuning used.  ``frame`` is the arm frame over the table's rows; the
    rows handed to the design take the table's targets.  Each split carries the certificate ``(train, every hidden row
    of the split's cells / groups / state inside the training rows)`` -- the tuning guard's test side -- so
    ``ConformalWrapper(guard="nested_certificate")`` checks exactly that split.

    Addendum 1 items 1-2: the V5 designs no longer come from here but from
    ``gen19ct.models.inner_design.SimultaneousInnerCells``, which yields ``interface.InnerSplit`` directly.  This class
    remains the bridge for the V1 and V2 inner designs, whose ``mode`` is always ``full``; ``mode="first"`` is
    DEPRECATED and never built by the plan."""

    def __init__(self, design: Any, frame: pd.DataFrame, mode: str = "full", inner_folds: Iterable[int] | None = None):
        if mode not in ("full", "first"):
            raise ValueError(f"mode {mode!r}")
        self.design, self.frame, self.mode = design, frame, mode
        self.inner_folds = None if inner_folds is None else tuple(sorted({int(f) for f in inner_folds}))
        sel = mode if self.inner_folds is None else f"folds {list(self.inner_folds)}"
        self.name = f"tuning_splits({type(design).__name__}, {sel})"
        self._cache: tuple[tuple, list[Any]] | None = None

    def all_splits(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> list[Any]:
        """Every inner split of the design (one draw with ``context.seed``; cached for the same table, mask and seed)."""
        m = np.asarray(mask, dtype=bool)
        key = (id(table), hashlib.sha1(np.packbits(m).tobytes()).hexdigest(), int(m.sum()), context.seed)
        if self._cache is not None and self._cache[0] == key:
            return list(self._cache[1])
        labels = table.index[m]
        if not labels.isin(self.frame.index).all():
            raise KeyError("TuningSplitCalibration: frame does not cover the training rows")
        rows = self.frame.loc[labels].assign(**{I.TARGET_COL: table.y[m]})
        sp = list(self.design.splits(rows, context))
        self._cache = (key, sp)
        return list(sp)

    def tuning_splits(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> list[Any]:
        sp = self.all_splits(table, mask, context)
        if self.inner_folds is not None:
            return [s for s in sp if int(s.inner_fold) in self.inner_folds]
        if self.mode == "first" and sp:
            first = min(int(s.inner_fold) for s in sp)
            sp = [s for s in sp if int(s.inner_fold) == first]
        return sp

    def splits(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> list[I.InnerSplit]:
        mask = np.asarray(mask, dtype=bool)
        out = []
        for s in self.tuning_splits(table, mask, context):
            tr = table.mask_of(s.train_index) if len(s.train_index) else np.zeros(table.n, dtype=bool)
            cal = np.sort(table.positions(s.val_index))
            hid = np.sort(table.positions(s.hidden_index))
            if (tr & ~mask).any() or not mask[cal].all() or tr[cal].any() or tr[hid].any():
                raise AssertionError(f"inner split {s.split_id} is not a partition of the outer training rows")
            guard_te = np.sort(table.positions(s.guard_test_index)) if len(s.guard_test_index) else cal
            if not np.isin(cal, guard_te).all():
                raise AssertionError(f"inner split {s.split_id}: calibration rows outside its guard test side")
            vpos = table.positions(s.val_index)
            units = pd.Series(np.asarray(s.val_units, dtype=object), index=vpos).loc[cal].to_numpy(dtype=object)
            out.append(I.InnerSplit(unit=s.split_id, fold=int(s.inner_fold), train_mask=tr, cal_positions=cal,
                                    hidden_positions=hid, certificate=(tr.copy(), guard_te), row_units=units))
        return out


class V5ExactInnerCells(I.InnerCellCalibration):
    """DEPRECATED by POST-HOC addendum 1 item 1 (every V5 design and variant of every learned arm and of B6 now tunes
    on ``gen19ct.models.inner_design.SimultaneousInnerCells``, which hides a whole inner fold at once).  Kept because it
    is the sealed exact leave-one-inner-cell-out design and the medium restriction it implements is the one the
    simultaneous design inherits; the discovery plan builds none.

    ``interface.InnerCellCalibration`` (exact leave-one-inner-cell-out, the V5 inner design of the exact-fold arms)
    with the variant's medium: for the HNO3-only variant the inner cells are eligible on the HNO3 rows of the outer
    training rows and only those rows are calibration rows; ``medium="all"`` is the parent class unchanged."""

    def __init__(self, *args: Any, medium: str = "all", frame: pd.DataFrame | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if medium != "all" and frame is None:
            raise ValueError("a medium-restricted inner design needs the arm frame (acid_primary)")
        self.medium, self.frame = medium, frame

    def unit_assignment(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> list[list[tuple[str, str]]]:
        if self.medium == "all":
            return super().unit_assignment(table, mask, context)
        # the parent's unit cache is keyed without the medium: never share it with an all-media splitter
        seed = I._require_seed(context)
        return CH.inner_cell_assignment(self._majority(table, mask, context), seed, self.n_folds, self.max_cells)

    def _majority(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> dict[tuple[str, str], str]:
        if self.medium == "all":
            return super()._majority(table, mask, context)
        if context.v6_mask is None:
            raise ValueError("context.v6_mask is required: V6 cells are never inner validation cells")
        uf = I.unit_frame(table, mask)
        uf["acid_primary"] = self.frame.loc[uf.index, "acid_primary"].to_numpy(dtype=object)
        v6 = pd.Series(table.aligned_bool(context.v6_mask, "v6_mask")[np.flatnonzero(mask)], index=uf.index)
        return CH.inner_cell_majority(uf, CH.Thresholds(self.k, self.p, self.m), v6, self.medium)

    def splits(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> list[I.InnerSplit]:
        out = super().splits(table, mask, context)
        if self.medium == "all":
            return out
        med = CH.medium_mask(self.frame.loc[table.index], self.medium)
        keep = []
        for sp in out:
            sel = med[sp.cal_positions]
            cal = sp.cal_positions[sel]
            if len(cal):
                keep.append(replace(sp, cal_positions=cal,
                                    row_units=None if sp.row_units is None else sp.row_units[sel]))
        return keep


def calibration_folds(available: Iterable[int], inner_mode: str, *, tuned: bool = True) -> dict[str, Any]:
    """Which inner folds a heavy arm's conformal calibration uses (:data:`READINGS` ``heavy_arm_intervals``).

    ``tuned`` arms: ``full`` mode (every inner fold tuned) -> every inner fold, cross-fitted (``cross_fit`` True: fold
    ``j`` takes the configuration selected without it).  Fewer than two folds -> none (``status`` says why).  Untuned
    arms (B8): the folds of the inner mode itself.

    Addendum 1 item 2 leaves only ``full``: three inner folds on every seed that is run, so every job of the plan takes
    the cross-fitted branch.  ``first`` is the DEPRECATED sealed-plan reading (``boosted`` and ``factorized`` now refuse
    it); it is kept here for the sealed-plan tests and for an untuned arm's single-fold case."""
    avail = sorted({int(f) for f in available})
    if inner_mode not in ("full", "first"):
        raise ValueError(f"inner_mode {inner_mode!r}")
    if not tuned:
        use = avail if inner_mode == "full" else avail[:1]
        return {"tuning_folds": [], "calibration_folds": use, "cross_fit": False,
                "status": "calibrated" if use else "not_calibrated_no_inner_split"}
    if inner_mode == "full":
        ok = len(avail) >= 2
        return {"tuning_folds": avail, "calibration_folds": avail if ok else [], "cross_fit": True,
                "status": "calibrated" if ok else "not_calibrated_fewer_than_two_inner_folds"}
    if len(avail) < 2:
        return {"tuning_folds": avail[:1], "calibration_folds": [], "cross_fit": False,
                "status": "not_calibrated_no_inner_fold_after_the_tuning_fold"}
    return {"tuning_folds": avail[:1], "calibration_folds": [avail[1]], "cross_fit": False, "status": "calibrated"}


class CrossFitResidualConformal(I.ConformalWrapper):
    """Split-conformal calibration only (section 12): the absolute residuals of ``arms_by_fold[j]`` refitted on every
    inner split of inner fold ``j`` (the splitter must yield splits of those folds only), and the quantiles; the
    interval centre is the outer refit made by the point step (no second outer fit).  ``arms_by_fold`` carries, per
    calibration fold, the frozen arm whose configuration was selected without that fold's rows
    (:func:`calibration_folds`)."""

    def __init__(self, arms_by_fold: Mapping[int, Any], splitter: Any, guard: str = "nested_certificate",
                 selections: Mapping[int, Mapping[str, Any]] | None = None):
        if not arms_by_fold:
            raise ValueError("CrossFitResidualConformal needs at least one calibration fold")
        self.arms_by_fold = {int(k): v for k, v in arms_by_fold.items()}
        self.selections = {int(k): dict(v) for k, v in (selections or {}).items()}
        super().__init__(next(iter(self.arms_by_fold.values())), splitter=splitter, guard=guard)

    def _calibrate(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> tuple[np.ndarray, list[Any]]:
        splits = self.splitter.splits(table, mask, context)
        if not splits:
            raise ValueError("the inner design produced no calibration split")
        res, units = [], []
        for sp in splits:
            if int(sp.fold) not in self.arms_by_fold:
                raise AssertionError(f"inner split {sp.unit} belongs to fold {sp.fold}, not a calibration fold "
                                     f"{sorted(self.arms_by_fold)}")
            if (sp.train_mask & ~mask).any() or not mask[sp.cal_positions].all() or sp.train_mask[sp.cal_positions].any():
                raise AssertionError(f"inner split {sp.unit} is not inside the outer training rows")
            self._verify(table, sp, context)
            FR.assert_not_scored(table.index[sp.cal_positions], context.v6_mask, "conformal calibration set")
            ctx = context.for_training(None, hidden_index=table.index[sp.hidden_positions])
            arm = self.arms_by_fold[int(sp.fold)].clone().fit_table(table, sp.train_mask, ctx)
            mean = arm.predict_positions(sp.cal_positions)["mean_logD"].to_numpy(dtype=float)
            if not np.isfinite(mean).all():
                raise AssertionError(f"{self.name}: non-finite calibration prediction in {sp.unit}")
            res.append(np.abs(table.y[sp.cal_positions] - mean))
            units.append(sp.unit)
        return np.concatenate(res), units

    def fit_table(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> "CrossFitResidualConformal":
        if self.seeds is not None:
            raise ValueError("CrossFitResidualConformal is single-seed (the job's run seed)")
        self.fit_seed = context.seed
        self.residuals, self.calibration_units = self._calibrate(table, mask, context)
        self.quantiles = {lv: I.conformal_quantile(self.residuals, lv) for lv in I.LEVELS}
        return self

    def record(self) -> dict[str, Any]:
        if self.residuals is None:
            raise RuntimeError("fit first")
        return {"n_calibration": int(len(self.residuals)), "quantiles": {str(k): v for k, v in self.quantiles.items()},
                "n_inner_splits": len(self.calibration_units), "guard": self.guard, "seed": self.fit_seed,
                "calibration_folds": sorted(self.arms_by_fold), "selections": {str(k): v for k, v in
                                                                                sorted(self.selections.items())},
                "splitter": getattr(self.splitter, "name", type(self.splitter).__name__)}


# --------------------------------------------------------------------------------------------- #
# budget (section 7 item 5) and the cost model
# --------------------------------------------------------------------------------------------- #

def ledger_from_records(out_root: Path) -> pd.DataFrame:
    """One row per (record, step) of every discovery record under ``out_root``: job key, arm, stage, seconds."""
    recs = []
    root = discovery_root(out_root)
    if not root.exists():
        return pd.DataFrame(columns=["job", "arm", "stage", "fold_id", "step", "seconds"])
    for js in sorted(root.rglob("*.json")):
        body = read_record(js)
        if not body or "steps" not in body or "job" not in body:
            continue
        for step, info in (body.get("steps") or {}).items():
            if not isinstance(info, Mapping) or info.get("shared_with"):
                continue
            recs.append({"job": body["job"].get("key"), "arm": body.get("arm"), "stage": body["job"].get("stage"),
                         "fold_id": body.get("fold_id"), "step": step, "seconds": float(info.get("seconds") or 0.0)})
    return pd.DataFrame(recs, columns=["job", "arm", "stage", "fold_id", "step", "seconds"])


def budget_status(total_seconds: float, *, budget_hours: float = BUDGET_HOURS) -> dict[str, Any]:
    """Section 7 item 5: elapsed compute against the 60-hour budget and the demotion order."""
    used_h = float(total_seconds) / 3600.0
    exhausted = used_h >= budget_hours
    return {"budget_hours": budget_hours, "used_hours": round(used_h, 4), "remaining_hours": round(budget_hours - used_h, 4),
            "exhausted": exhausted, "demotion_order": list(DEMOTION_ORDER), "never_demoted": list(NEVER_DEMOTED),
            "demoted_now": list(DEMOTION_ORDER) if exhausted else [],
            "note": ("section 7 item 5: on exhaustion only M3-M7 are demoted (not implemented in this runner, so "
                     "demoted to exploratory in the registered order); every other job -- H1, H1b, H4 and the rest -- "
                     "keeps running (discovery.demotable)")}


#: grid sizes (sections 5 and 6): configurations fitted per inner split during tuning
CONFIGS_PER_SPLIT: dict[str, int] = {"B6": 12, "B5": 6, "FLAT_CAT": 6, "B8": 0, "M1": 9, "M2": 3}
#: calibration refits per inner split (section 12 cross-fitted reading, :data:`READINGS` ``calibration_refits``): B6
#: reuses its tuning predictions (0); every other learned arm refits the configuration selected without the fold on that
#: fold's inner training rows (1 per split, 3 per outer fold) -- the conservative reading of addendum 1 item 2's "no
#: extra fits", priced here so that the estimate matches the runner
CALIBRATION_FITS_PER_SPLIT: dict[str, int] = {"B6": 0, "B5": 1, "FLAT_CAT": 1, "B8": 1, "M1": 1, "M2": 1}
#: the readings of "the first inner fold (one fit per configuration)" on V5 designs the cost model can price (task X
#: finding V-04): every inner split of the first inner fold (implemented) or one split per configuration
FIRST_FOLD_READINGS: tuple[str, ...] = ("all_splits_of_first_inner_fold", "one_split")


@dataclass(frozen=True)
class UnitCost:
    """Measured (benchmark) or assumed cost of one arm on one design class.

    ``inner_fit_s``    one tuning configuration on one inner split (B6: all 12 configurations of one split, solved jointly)
    ``cal_fit_s``      one calibration refit on one inner split (fixed configuration)
    ``outer_refit_s``  one refit on all outer-training rows
    ``splits_full`` / ``splits_first``  inner splits per outer fold on seed 104729 / on the other seeds
    ``guard_s``        seconds per inner split for its isolation check

    Under POST-HOC addendum 1 (items 1-3) every job is priced in ``full`` mode and a V5 measurement has
    ``splits_full = 3`` -- one simultaneous split per inner fold, each hiding up to 30 cells -- instead of the ~21
    batched splits of the sealed plan; ``splits_first`` is then unused.  V1 and V2 measurements are unchanged (3)."""

    arm: str
    design_class: str
    inner_fit_s: float
    cal_fit_s: float
    outer_refit_s: float
    splits_full: float
    splits_first: float
    guard_s: float = 0.35
    source: str = "benchmark"

    def fold_seconds(self, mode: str, first_fold_reading: str = FIRST_FOLD_READINGS[0]) -> dict[str, float]:
        if first_fold_reading not in FIRST_FOLD_READINGS:
            raise ValueError(f"first_fold_reading {first_fold_reading!r}")
        n = self.splits_full if mode == "full" else (self.splits_first if first_fold_reading == FIRST_FOLD_READINGS[0]
                                                     else min(1.0, self.splits_first))
        per_split_cfg = 1 if self.arm == "B6" else CONFIGS_PER_SPLIT.get(self.arm, 0)
        tune = n * per_split_cfg * self.inner_fit_s
        if self.arm == "B6":
            cal = 0.0 if mode == "full" else n * self.inner_fit_s        # the selected configurations on the next fold
        else:
            cal = n * CALIBRATION_FITS_PER_SPLIT.get(self.arm, 0) * self.cal_fit_s
        guard = n * self.guard_s
        return {"tuning": tune, "calibration": cal, "outer_refit": self.outer_refit_s, "guard": guard,
                "total": tune + cal + self.outer_refit_s + guard}


def design_class(job: JobSpec) -> str:
    """The benchmark class a job's per-fold cost is taken from."""
    if job.design in ("V5", "V5P", "V5PAIR"):
        return "V5_exact" if job.scheme in ("exact", "cell_x_group", "cell_pair") else "V5_batched"
    return job.design


def estimate_cost(jobs: Sequence[JobSpec], fold_counts: Mapping[str, int], unit_costs: Mapping[tuple[str, str], UnitCost],
                  *, other_costs_s: Mapping[str, float] | None = None, workers: int = 2,
                  budget_hours: float = BUDGET_HOURS, fold_overhead_s: float = 0.0,
                  first_fold_reading: str = FIRST_FOLD_READINGS[0]) -> pd.DataFrame:
    """Per-job cost = folds x per-fold cost (:meth:`UnitCost.fold_seconds` of the job's arm, design class and inner mode),
    cumulative in plan order, with the 1-worker compute and the ``workers``-worker wall-clock estimate (compute /
    workers) against the budget.  ``fold_counts`` maps job keys to fitted folds; ``other_costs_s`` maps job keys of
    non-fit jobs (safeguard, comparator intervals) to seconds; ``fold_overhead_s`` is added per fitted fold (outer guard,
    section 13 support file, I/O)."""
    recs = []
    cum = 0.0
    for j in jobs:
        if j.kind == "marker" or j.kind == "h3_spec":
            continue
        n = int(fold_counts.get(j.key, 0))
        row = {"job": j.key, "kind": j.kind, "arm": j.arm, "design": j.design, "variant_label": j.variant_label,
               "seed": j.seed, "stage": j.stage, "group": j.group, "registered": j.registered, "n_folds": n,
               "inner_mode": j.inner_mode if j.kind == "fit" else "", "tuning_s": 0.0, "calibration_s": 0.0,
               "outer_refit_s": 0.0, "guard_s": 0.0, "cost_source": ""}
        if j.kind == "fit":
            arm = j.arm
            uc = unit_costs.get((arm, design_class(j)))
            if uc is None:
                raise KeyError(f"no unit cost for {arm} / {design_class(j)}")
            per = uc.fold_seconds(j.inner_mode, first_fold_reading)
            row.update(tuning_s=n * per["tuning"], calibration_s=n * per["calibration"],
                       outer_refit_s=n * per["outer_refit"], guard_s=n * (per["guard"] + fold_overhead_s),
                       cost_source=uc.source)
            total = n * (per["total"] + fold_overhead_s)
        else:
            total = float((other_costs_s or {}).get(j.key, 0.0))
            row["cost_source"] = "assumed"
        cum += total
        row.update(compute_s=total, cumulative_compute_h=cum / 3600.0,
                   cumulative_wall_h=cum / 3600.0 / max(1, int(workers)),
                   within_budget_wall=cum / 3600.0 / max(1, int(workers)) <= budget_hours)
        recs.append(row)
    return pd.DataFrame(recs)


def stage_checkpoints(table: pd.DataFrame, *, workers: int = 2, budget_hours: float = BUDGET_HOURS) -> list[dict[str, Any]]:
    """Cumulative wall clock at the end of every stage, in plan order (addendum 1: the checkpoints the operator watches
    against the 60-hour budget).  One row per stage: its own compute, the cumulative compute and wall clock, and
    whether the budget still holds there."""
    if table.empty:
        return []
    out = []
    for stage in [s for s in dict.fromkeys(table["stage"]) if s]:
        sub = table[table["stage"] == stage]
        cum_h = float(sub["cumulative_compute_h"].iloc[-1])
        out.append({"stage": str(stage), "n_jobs": int(len(sub)), "n_folds": int(sub["n_folds"].sum()),
                    "stage_compute_h": round(float(sub["compute_s"].sum()) / 3600.0, 3),
                    "cumulative_compute_h": round(cum_h, 3),
                    "cumulative_wall_h": round(cum_h / max(1, int(workers)), 3),
                    "within_budget_wall": bool(cum_h / max(1, int(workers)) <= budget_hours)})
    return out


def cost_summary(table: pd.DataFrame, *, workers: int = 2, budget_hours: float = BUDGET_HOURS) -> dict[str, Any]:
    """Totals per arm, per stage and overall, and where the cumulative wall clock crosses the budget."""
    if table.empty:
        return {"total_compute_h": 0.0, "total_wall_h": 0.0, "fits_budget": True}
    by_arm = (table.groupby("arm")["compute_s"].sum() / 3600.0).round(3).to_dict()
    by_stage = (table.groupby("stage")["compute_s"].sum() / 3600.0).round(3).to_dict()
    tot = float(table["compute_s"].sum()) / 3600.0
    over = table[~table["within_budget_wall"]]
    return {"workers": int(workers), "budget_hours": budget_hours, "total_compute_h": round(tot, 3),
            "total_wall_h": round(tot / max(1, workers), 3), "fits_budget": bool(tot / max(1, workers) <= budget_hours),
            "compute_h_by_arm": by_arm, "compute_h_by_stage": by_stage,
            "first_job_beyond_budget": None if over.empty else str(over["job"].iloc[0]),
            "n_jobs_beyond_budget": int(len(over))}


# --------------------------------------------------------------------------------------------- #
# H3 (section 11): job specs and training-row transforms (not executed by this runner)
# --------------------------------------------------------------------------------------------- #

H3_ARMS: tuple[str, ...] = ("WITH", "WITHOUT", "ACT_PERMUTED", "ACT_METAL_SHUFFLED")
H3_MODEL_ARMS: tuple[str, ...] = ("RETAINED_LADDER_CONFIGURATION", "B6", "B5")


def h3_job_specs(seeds: Sequence[int] = PLAN_SEEDS) -> list[JobSpec]:
    """Section 11 design: {retained ladder configuration, B6, B5} x {WITH, WITHOUT, ACT_PERMUTED, ACT_METAL_SHUFFLED} x
    {V5-primary Ln(III) selection cells, V2 selection Ln(III) states, V1 selection folds (Ln rows)} x the seeds a
    learned arm runs in discovery -- the sealed text's 5 discovery seeds reduced to seed 104729 by addendum 1 item 3
    (:data:`PLAN_SEEDS`; task X finding V-F06), same folds and batches as the main runs.  Specs only (kind
    ``h3_spec``); the sealed-plan enumeration is ``h3_job_specs(DISCOVERY_SEEDS)``."""
    out = []
    for model in H3_MODEL_ARMS:
        for h3 in H3_ARMS:
            for design, variant, scheme in (("V5", "primary", "batched"), ("V2", "element", "exact"),
                                            ("V1", "copy", "grouped10")):
                for seed in seeds:
                    out.append(JobSpec(kind="h3_spec", arm=f"{model}:{h3}", design=design, variant=variant,
                                       scheme="exact" if model == "B6" and design == "V5" else scheme, seed=seed,
                                       stage="11_h3_specs", group="section 11 actinide ablation (spec only)",
                                       purpose="H3: Ln test set = Ln(III) scored rows of the selection half; "
                                               f"training transform {h3}", registered=True, condition="spec_only"))
    return out


def actinide_rows(frame: pd.DataFrame) -> np.ndarray:
    """Rows whose metal category is actinide, unknown-state actinide rows included (section 11 WITHOUT)."""
    from gen19ct.chemistry import metals as MET

    el = frame[SG.ELEMENT_COL].astype(object).to_numpy()
    return np.array([isinstance(e, str) and e in MET.ACTINIDES for e in el], dtype=bool)


def h3_training_rows(train: pd.DataFrame, arm: str, *, seed: int, pub_group_col: str = I.PUB_GROUP_COL) -> pd.DataFrame:
    """The training rows of an H3 arm: WITH unchanged; WITHOUT drops every actinide row; ACT_PERMUTED permutes ``log_D``
    among actinide rows within (system, publication group); ACT_METAL_SHUFFLED shuffles the metal-state labels (state
    and element together) among actinide rows within a system.  Seeded (``default_rng(seed)``); target-free except the
    permutation, which only moves training targets."""
    if arm not in H3_ARMS:
        raise ValueError(f"H3 arm {arm!r}")
    if arm == "WITH":
        return train
    an = actinide_rows(train)
    if arm == "WITHOUT":
        return train[~an]
    out = train.copy()
    rng = np.random.default_rng(int(seed))
    sub = out[an]
    if arm == "ACT_PERMUTED":
        keys = [SG.SYSTEM_COL, pub_group_col]
        for _, g in sorted(sub.groupby(keys, sort=True).groups.items()):
            idx = list(g)
            out.loc[idx, I.TARGET_COL] = out.loc[idx, I.TARGET_COL].to_numpy()[rng.permutation(len(idx))]
        return out
    for _, g in sorted(sub.groupby(SG.SYSTEM_COL, sort=True).groups.items()):
        idx = list(g)
        perm = rng.permutation(len(idx))
        for col in (SG.METAL_COL, SG.ELEMENT_COL):
            out.loc[idx, col] = out.loc[idx, col].to_numpy(dtype=object)[perm]
    return out


# --------------------------------------------------------------------------------------------- #
# scoring: frames, paired units, R19 (sections 4, 8, 9)
# --------------------------------------------------------------------------------------------- #

def read_preseal_selection(path: Path, arms: Sequence[str] | None = None) -> pd.DataFrame:
    """Pre-seal predictions of the selection half only, via a parquet row filter (confirmation rows are never
    materialised), optionally restricted to ``arms``."""
    filters: list[tuple] = [("half", "==", SELECTION)]
    if arms is not None:
        filters.append(("arm", "in", list(arms)))
    df = pd.read_parquet(path, filters=filters)
    n_bad = int((df["half"].astype(str) != SELECTION).sum())
    _count_guard("preseal_files_read")
    _count_guard("preseal_rows_read", len(df))
    _count_guard("preseal_non_selection_rows", n_bad)
    if n_bad:
        raise AssertionError(f"{path}: the parquet filter returned a non-selection-half row")
    return df


class StaleRecordError(AssertionError):
    """A stored discovery record does not match what the current code, fold file and plan state produce."""


def read_discovery_record_set(out_root: Path, arm: str, design_dir: str, seed: int, *,
                              expected: Mapping[str, Mapping[str, str]], steps: Sequence[str] = ("point",)
                              ) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """The verified fold records of one (arm, design directory, seed) (:data:`READINGS` ``record_verification``).

    ``expected`` maps every fittable fold id of the job to ``{"digest": ..., "fold_hash": ...}`` as the runner would
    compute them now.  Raises :class:`StaleRecordError` on a record of a fold id the job does not fit, a digest or fold
    hash that differs, a prediction without a JSON record or without one of ``steps``, or a parquet whose fold / half
    columns disagree.  Returns ``(None, status)`` while folds are missing (an incomplete job is never scored) and
    ``(frame, status)`` when the record set is exactly the expected one."""
    if expected is None:
        raise TypeError("records are read only against the digests the current code produces (expected=...)")
    d = discovery_root(out_root) / arm / design_dir / f"s{seed}"
    exp = {str(k): dict(v) for k, v in expected.items()}
    status: dict[str, Any] = {"arm": arm, "design_dir": design_dir, "seed": int(seed), "n_expected": len(exp)}
    if not d.exists():
        return None, {**status, "status": "missing", "n_found": 0}
    frames, found = [], set()
    by_name = {safe_fold_name(k): k for k in exp}
    for pq in sorted(d.glob("*.parquet")):
        rec = read_record(pq.with_suffix(".json"))
        if rec is None:
            raise StaleRecordError(f"{pq}: prediction without a JSON record")
        fid = str(rec.get("fold_id"))
        if fid not in exp or by_name.get(pq.stem) != fid:
            raise StaleRecordError(f"{pq}: fold {fid!r} is not a fittable fold of the current job ({arm}/{design_dir}/"
                                   f"s{seed})")
        if rec.get("digest") != exp[fid]["digest"]:
            raise StaleRecordError(f"{pq}: record digest {str(rec.get('digest'))[:12]}... differs from the current "
                                   f"{str(exp[fid]['digest'])[:12]}... (code, fold file or plan state changed; rerun the job)")
        if "fold_hash" in exp[fid] and rec.get("fold_hash") != exp[fid]["fold_hash"]:
            raise StaleRecordError(f"{pq}: fold hash differs from the registered fold")
        missing_steps = [x for x in steps if x not in (rec.get("steps") or {})]
        if missing_steps:
            raise StaleRecordError(f"{pq}: record lacks step(s) {missing_steps}")
        fr = pd.read_parquet(pq)
        if len(fr) and (fr["fold_id"].astype(str) != fid).any():
            raise StaleRecordError(f"{pq}: parquet rows of another fold")
        frames.append(fr)
        found.add(fid)
    status.update(n_found=len(found))
    if found != set(exp):
        return None, {**status, "status": "incomplete", "missing_folds": sorted(set(exp) - found)[:10]}
    out = pd.concat(frames, ignore_index=True) if frames else None
    if out is not None and (out["half"].astype(str) != SELECTION).any():
        raise AssertionError(f"{d}: a stored prediction of a non-selection-half row")
    return out, {**status, "status": "complete"}


def read_discovery_predictions(out_root: Path, arm: str, design_dir: str, seed: int, *,
                               expected: Mapping[str, Mapping[str, str]], steps: Sequence[str] = ("point",)
                               ) -> pd.DataFrame | None:
    """:func:`read_discovery_record_set` without the status: the verified predictions, or ``None`` while incomplete."""
    return read_discovery_record_set(out_root, arm, design_dir, seed, expected=expected, steps=steps)[0]


def scoring_frame(pred: pd.DataFrame, attrs: pd.DataFrame, *, design: str, v6_mask: pd.Series, what: str,
                  half_col: str | None = None) -> pd.DataFrame:
    """One arm's predictions of one design and seed joined to the row attributes, indexed by ``row_id`` (a row scored
    in two folds is refused), with the selection-half and ``V6_TARGET_ROWS`` guards applied."""
    need = ["row_id", "fold_id", "mean_logD"]
    missing = [c for c in need if c not in pred.columns]
    if missing:
        raise KeyError(f"{what}: prediction columns missing {missing}")
    if "half" in pred.columns and (pred["half"].astype(str) != SELECTION).any():
        raise AssertionError(f"{what}: prediction frame holds a non-selection-half row")
    fr = pred.join(attrs.drop(columns=[c for c in ("row_id",) if c in attrs.columns]), on="row_id", rsuffix="_attr")
    fr = fr.set_index("row_id", drop=False)
    fr.index.name = None
    if fr.index.has_duplicates:
        raise AssertionError(f"{what}: a row is scored twice in one design and seed")
    hc = half_col or f"registered_half_{design}"
    if hc not in fr.columns:
        raise KeyError(f"{what}: attrs lack {hc}")
    # the guard counters (task X finding V-MAN-09) are taken BEFORE the guards raise, so a completed pass records what
    # was actually re-derived and seen rather than a constant
    halves = fr[hc].astype(str)
    _count_guard("scoring_frames_checked")
    _count_guard("rows_half_rederived", len(fr))
    _count_guard("confirmation_half_rows_seen", int((halves != SELECTION).sum()))
    _count_guard("v6_target_rows_scored", int(v6_mask.reindex(fr.index).fillna(False).astype(bool).sum()))
    assert_selection_rows(fr.index, halves, what)
    assert_v6_clean(fr.index, v6_mask, what)
    fr[EM.PRED_COL] = pd.to_numeric(fr["mean_logD"], errors="coerce").astype(float)
    return fr


def apply_scoring_filter(fr: pd.DataFrame, name: str, *, wildcard_col: str = "wildcard_copy_partner_in_training"
                         ) -> pd.DataFrame:
    """The rows a scoring-filter sensitivity keeps (section 8 R19 item 6; section 2 resolutions)."""
    if name == "none":
        return fr
    if name == "non_DGA_stratum":
        return fr[fr["dga_stratum"].astype(str) == "non_DGA"]
    if name == "acid_grid_rows_excluded":
        return fr[~fr["acid_grid_flag"].astype(bool)]
    if name == "censoring_candidates_excluded_scoring":
        return fr[~fr["censoring_candidate"].astype(bool)]
    if name in (ET.WILDCARD_COPY_SENSITIVITY, ET.WILDCARD_COPY_STRICT_SENSITIVITY):
        col = wildcard_col if name == ET.WILDCARD_COPY_SENSITIVITY else wildcard_col + "_strict"
        return fr[~fr[col].astype(bool)]
    raise ValueError(f"unknown scoring filter {name!r}")


@dataclass
class PairedUnits:
    """Per-unit MAE of a candidate and a comparator on identical scored rows, with every registered cluster unit."""

    design: str
    candidate: str
    comparator: str
    cand_mae: pd.Series
    comp_mae: pd.Series
    clusters: dict[str, pd.Series]
    n_rows: int

    @property
    def delta(self) -> float:
        """Delta = metric(comparator) - metric(candidate) of the unit macro (positive favours the candidate)."""
        return float(self.comp_mae.mean() - self.cand_mae.loc[self.comp_mae.index].mean())


def unit_mae(fr: pd.DataFrame, design: str, *, v6_mask: pd.Series, v1_scheme: str = "exact",
             remainder_groups: Iterable[str] | None = None) -> pd.Series:
    """The section 4 per-unit MAE (``metrics.design_per_unit_table``): V5 / V5-P cell, V1 outer-fold unit (grouped
    scheme: the row's exact-design unit), V2 metal state."""
    kw: dict[str, Any] = {}
    if design == "V1":
        kw = dict(v1_scheme=v1_scheme, v1_group_col=EM.PUB_GROUP_COL, remainder_groups=remainder_groups)
    return EM.design_per_unit_table(fr, design, v6_mask=v6_mask, **kw)["mae"]


def paired_units(cand: pd.DataFrame, comp: pd.DataFrame, design: str, *, candidate: str, comparator: str,
                 v6_mask: pd.Series, cand_v1_scheme: str = "exact", comp_v1_scheme: str = "exact",
                 remainder_groups: Iterable[str] | None = None, pub_group_col: str = EM.PUB_GROUP_COL) -> PairedUnits:
    """Identical scored rows are required (a paired contrast); per-unit MAE of both arms and the registered clusters."""
    if set(cand.index) != set(comp.index):
        raise ValueError(f"{candidate} vs {comparator} ({design}): the arms score different rows "
                         f"({len(set(cand.index) - set(comp.index))} only in the candidate, "
                         f"{len(set(comp.index) - set(cand.index))} only in the comparator)")
    rem = None if remainder_groups is None else list(remainder_groups)
    a = unit_mae(cand, design, v6_mask=v6_mask, v1_scheme=cand_v1_scheme, remainder_groups=rem)
    b = unit_mae(comp.loc[cand.index], design, v6_mask=v6_mask, v1_scheme=comp_v1_scheme, remainder_groups=rem)
    if set(a.index) != set(b.index):
        raise ValueError(f"{candidate} vs {comparator} ({design}): different averaging units")
    kw: dict[str, Any] = {"pub_group_col": pub_group_col}
    if design == "V1":
        kw.update(v1_scheme=cand_v1_scheme, v1_group_col=EM.PUB_GROUP_COL, remainder_groups=rem)
    clusters = EM.design_unit_clusters(cand, design, **kw)
    return PairedUnits(design=design, candidate=candidate, comparator=comparator, cand_mae=a, comp_mae=b.loc[a.index],
                       clusters={k: v.loc[a.index] for k, v in clusters.items()}, n_rows=int(len(cand)))


def filtered_pair(cand: pd.DataFrame, comp: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same scoring filter on both arms; a row dropped for either arm is dropped for both."""
    keep = apply_scoring_filter(cand, name).index.intersection(apply_scoring_filter(comp, name).index)
    return cand.loc[keep], comp.loc[keep]


def bootstraps(pu: PairedUnits, *, name: str, n_resamples: int = ET.N_RESAMPLES,
               seed: int = ET.BOOTSTRAP_SEED) -> dict[str, ET.BootstrapResult]:
    """A paired cluster bootstrap under every registered cluster unit of the design."""
    units = ET.REGISTERED_CLUSTER_UNITS[pu.design]
    out = {}
    for u in units:
        if u not in pu.clusters:
            raise KeyError(f"{name}: no {u} clusters for {pu.design}")
        out[u] = ET.paired_cluster_bootstrap(pu.comp_mae, pu.cand_mae, pu.clusters[u], n_resamples=n_resamples, seed=seed,
                                             contrast=name, cluster_unit=u)
    return out


SCOPES: dict[str, tuple[Any, ...]] = {"stop_rule": (1, 2, 3, 5), "ladder": (1, 2, 3, 5, "6s"),
                                      "freezing_screen": (1, 2, 3, 5, "6s"), "items_1_5": (1, 2, 3, 4, 5),
                                      "full": (1, 2, 3, 4, 5, 6)}

#: how ``transfer.tost`` built the 90 % bounds it reports (task X finding V-S07; ``both`` is the registered default)
TOST_ENVELOPE: dict[str, str] = {
    "both": "conservative envelope: low = min(percentile 90 %, BCa 90 %), high = max(percentile 90 %, BCa 90 %)",
    "percentile": "percentile 90 % interval", "bca": "BCa 90 % interval"}


#: item statuses that make a verdict UNDECIDED rather than PASS (never FAIL)
INCONCLUSIVE_STATUSES: tuple[str, ...] = ("UNTESTABLE", "NOT_RUN", ITEM4_NOT_EVALUATED)


def r19_verdict(items: Iterable[Mapping[str, Any]]) -> str:
    """``transfer.r19``'s verdict rule over (possibly rewritten) items: FAIL beats UNDECIDED beats PASS."""
    st = [str(i["status"]) for i in items]
    if "FAIL" in st:
        return "FAIL"
    return "UNDECIDED" if any(s in INCONCLUSIVE_STATUSES for s in st) else "PASS"


def reduced_item6(design: str, sensitivities: Mapping[str, Any], reduced: Sequence[str]) -> tuple[dict[str, Any],
                                                                                                  dict[str, str]]:
    """Addendum 1 item 4: R19 item 6 of a LEARNED-arm contrast, decided on ``reduced`` -- the refits that were run plus
    every registered scoring-filter sensitivity -- and labelled :data:`ADDENDUM_LABEL`.  Returns the item and the
    sensitivities that were not run, by name and reason (:data:`LEARNED_REFITS_NOT_RUN`), which the report prints."""
    fail, untestable = [], []
    for n in reduced:
        if n not in sensitivities:
            fail.append(f"{n}: missing")
            continue
        v = sensitivities[n]
        if isinstance(v, str):
            (untestable if v == ET.UNTESTABLE else fail).append(n if v == ET.UNTESTABLE else f"{n}: {v!r}")
            continue
        fv = float(v)
        if not (np.isfinite(fv) and fv > 0):
            fail.append(f"{n}: Delta={fv:.6g}")
    registered = set(ET.REGISTERED_SENSITIVITIES[design])
    not_run = {n: why for n, why in LEARNED_REFITS_NOT_RUN.get(design, {}).items() if n in registered}
    body = "; ".join(fail + [f"{n}: UNTESTABLE" for n in untestable]) or \
        f"positive in every sensitivity of the reduced set {list(reduced)}"
    status = "FAIL" if fail else ("UNTESTABLE" if untestable else "PASS")
    return ({"item": 6, "name": "positive_in_every_registered_sensitivity", "status": status,
             "detail": f"{ADDENDUM_LABEL}: {body}; not run (addendum 1 item 4): {sorted(not_run)}",
             "sensitivity_set": ADDENDUM_LABEL, "sensitivities_not_run": ", ".join(sorted(not_run))}, not_run)


def _scope_verdict(items: Mapping[int, str], scope: Sequence[Any], sens: Mapping[str, Any], design: str) -> dict[str, Any]:
    statuses = {}
    for it in scope:
        if it == "6s":
            names = [n for n in ET.REGISTERED_SENSITIVITIES.get(design, ()) if n in SCORING_FILTER_SENSITIVITIES]
            vals = [sens.get(n) for n in names]
            if any(isinstance(v, str) or v is None for v in vals):
                s = "FAIL" if any(v is None for v in vals) else "UNTESTABLE"
            else:
                s = "PASS" if all(np.isfinite(float(v)) and float(v) > 0 for v in vals) else "FAIL"
            statuses["6_scoring_filters"] = s
        else:
            statuses[str(it)] = items[int(it)]
    vals = list(statuses.values())
    verdict = ("FAIL" if "FAIL" in vals else
               "UNDECIDED" if any(v in INCONCLUSIVE_STATUSES for v in vals) else "PASS")
    return {"verdict": verdict, "items": statuses}


def evaluate_contrast(*, name: str, family: str, design: str, primary: PairedUnits, margin: float,
                      seed_deltas: Mapping[int, float | None], sensitivities: Mapping[str, float | str],
                      deterministic: bool = False, n_resamples: int = ET.N_RESAMPLES,
                      bootstrap_seed: int = ET.BOOTSTRAP_SEED, learned: bool = False,
                      reduced_sensitivities: Sequence[str] | None = None) -> dict[str, Any]:
    """R19 (section 8) of one registered or exploratory contrast on the selection half.

    ``primary`` is the seed-104729 paired units (items 1-3, 5 and the TOST come from them), ``seed_deltas`` the
    per-discovery-seed Delta (item 4; a missing seed makes item 4 NOT_RUN), ``sensitivities`` every item 6 sensitivity
    of the design (value, or ``transfer.UNTESTABLE`` for a refit sensitivity not run).

    Addendum 1: ``learned`` (a contrast with a learned arm) makes item 4 **NOT_EVALUATED** in discovery (item 3), and
    ``reduced_sensitivities`` decides item 6 on that reduced set, labelled :data:`ADDENDUM_LABEL`, naming the
    sensitivities that were not run (item 4 of the addendum).  The verdict is recomputed by :func:`r19_verdict`, so a
    learned-arm contrast is at best UNDECIDED in discovery -- the stop-rule, ladder and freezing scopes contain neither
    R19 item 4 nor the refit sensitivities, so no decision changes.

    Returns the R19 result, the bootstraps, the scoped verdicts (:data:`SCOPES`) and the TOST under the primary cluster
    unit."""
    boots = bootstraps(primary, name=name, n_resamples=n_resamples, seed=bootstrap_seed)
    sd = [seed_deltas.get(s) for s in DISCOVERY_SEEDS]
    complete = all(v is not None and np.isfinite(float(v)) for v in sd)
    sens = {k: (v if isinstance(v, str) else float(v)) for k, v in sensitivities.items()}
    reg = ET.REGISTERED_SENSITIVITIES[design]
    for n in reg:
        sens.setdefault(n, ET.UNTESTABLE)
    r = ET.r19(design=design, stage="discovery", point=primary.delta, margin=margin, bootstraps=boots,
               seed_deltas=[float(v) if v is not None else float("nan") for v in sd], deterministic=deterministic,
               sensitivity_deltas=sens, contrast=name)
    items, not_run = list(r.items), {}
    if not deterministic and learned:
        items = [dict(i, status=ITEM4_NOT_EVALUATED, detail=ITEM4_NOT_EVALUATED_DETAIL) if i["item"] == 4 else i
                 for i in items]
    elif not deterministic and not complete:
        n_have = sum(1 for v in sd if v is not None and np.isfinite(float(v)))
        items = [dict(i, status="NOT_RUN", detail=f"{n_have} of {ET.N_SEEDS} discovery seeds scored")
                 if i["item"] == 4 else i for i in items]
    if reduced_sensitivities is not None:
        item6, not_run = reduced_item6(design, sens, list(reduced_sensitivities))
        items = [item6 if i["item"] == 6 else i for i in items]
    if items != list(r.items):
        r = replace(r, items=tuple(items), verdict=r19_verdict(items))
    statuses = {int(i["item"]): i["status"] for i in r.items}
    scopes = {k: _scope_verdict(statuses, v, sens, design) for k, v in SCOPES.items()}
    primary_unit = ET.REGISTERED_CLUSTER_UNITS[design][0]
    tost = ET.tost(boots[primary_unit])
    return {"name": name, "family": family, "design": design, "candidate": primary.candidate,
            "comparator": primary.comparator, "margin": float(margin), "point": primary.delta, "r19": r,
            "bootstraps": boots, "scopes": scopes, "tost": tost, "primary_cluster_unit": primary_unit,
            "p_primary": boots[primary_unit].p_two_sided, "seed_deltas": {int(s): v for s, v in seed_deltas.items()},
            "sensitivities": sens, "n_units": int(len(primary.cand_mae)), "n_rows": primary.n_rows,
            "learned_arm": bool(learned), "seeds_evaluated": list(PLAN_SEEDS) if learned else list(DISCOVERY_SEEDS),
            "sensitivity_set": ADDENDUM_LABEL if reduced_sensitivities is not None else "registered (full)",
            "sensitivities_not_run": not_run,
            "label": "discovery, optimistically biased (selection half)"}


def contrast_rows(res: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Tidy rows (one per registered cluster unit) of an :func:`evaluate_contrast` result."""
    base = {"family": res["family"], "contrast": res["name"], "design": res["design"], "candidate": res["candidate"],
            "comparator": res["comparator"], "half": "selection", "seed_set": "discovery", "decision_seed": PRIMARY_SEED,
            "margin": res["margin"], "r19_verdict_full": res["r19"].verdict,
            **{f"verdict_{k}": v["verdict"] for k, v in res["scopes"].items()},
            "tost_verdict_eps0.05": res["tost"]["verdict"], "tost_low_90": res["tost"]["low_90"],
            "tost_high_90": res["tost"]["high_90"],
            # task X finding V-S07: the TOST bounds are the envelope of the 90 % percentile and 90 % BCa intervals, and
            # a bound on 4 clusters rests on 2^4 distinct resamples -- construction and cluster count travel with them
            "tost_interval_construction": TOST_ENVELOPE[res["tost"]["interval"]],
            "tost_cluster_unit": res["tost"]["cluster_unit"],
            "tost_n_clusters": res["bootstraps"][res["primary_cluster_unit"]].n_clusters,
            "label": res["label"],
            "batching_label": res.get("batching_label", ""),
            # task X finding V-S02 / V-H1B-05: the fold scheme of each arm, because a closed-form comparator is fitted on
            # the exact leave-one-cell-out folds of section 3.1 (prereg "comparator_folds") while a heavy arm is batched
            "candidate_fold_scheme": (res.get("fold_schemes") or {}).get(res["candidate"], ""),
            "comparator_fold_scheme": (res.get("fold_schemes") or {}).get(res["comparator"], ""),
            "r19_item4": res["r19"].item(4)["status"],
            "seeds_evaluated": ", ".join(str(s) for s in res.get("seeds_evaluated", DISCOVERY_SEEDS)),
            "sensitivity_set": res.get("sensitivity_set", "registered (full)"),
            "sensitivities_not_run": ", ".join(sorted(res.get("sensitivities_not_run") or {})),
            "reported_verdict": res.get("reported_verdict", res["r19"].verdict),
            "seed_deltas": json.dumps({str(k): v for k, v in sorted(res["seed_deltas"].items())}, sort_keys=True),
            "sensitivities": json.dumps(res["sensitivities"], sort_keys=True, default=str)}
    out = []
    for u, br in res["bootstraps"].items():
        rec = br.record(base)
        rec["primary_cluster_unit"] = u == res["primary_cluster_unit"]
        # task X finding V-S04: one cluster per scoring unit is the unclustered unit bootstrap, which never decides
        rec["cluster_equals_scoring_unit"] = bool(br.n_clusters == br.n_units)
        rec["clustering"] = "unclustered_equivalent" if br.n_clusters == br.n_units else "clustered"
        out.append(rec)
    return out


def r19_item_rows(res: Mapping[str, Any]) -> pd.DataFrame:
    return res["r19"].to_frame({"family": res["family"], "half": "selection", "seed_set": "discovery"})


#: the columns that identify ONE computed contrast: two section 19 slots can share it ('M2 vs M0' is a registered H4
#: contrast AND the ladder step M2 vs its retained predecessor), so the BH family counts it twice (task X finding V-BH-06)
COMPUTED_CONTRAST_KEY: tuple[str, ...] = ("contrast", "design", "cluster_unit", "point", "p_two_sided")


def apply_bh(table: pd.DataFrame, *, p_col: str = "p_two_sided", family_col: str = "family",
             registered_families: Iterable[str] | None = None, m_registered_full: int | None = None) -> pd.DataFrame:
    """Benjamini-Hochberg **per family** (section 8, :data:`READINGS` ``bh_p`` / ``bh_family_column``) on the rows
    flagged ``primary_cluster_unit``; ``p_bh`` stays NaN on the other rows.

    ``p_bh`` is adjusted within the contrast's own family -- the ``family_col`` value (primary, H1b, S1(b), S1(c), H4,
    ladder, secondary designs, exploratory) -- and ``bh_m`` is that family's evaluated size.  Beside it:
    ``p_bh_pooled_registered`` / ``bh_m_pooled_registered``, the adjustment pooled over every evaluated registered
    contrast (what this function returned before task X finding V-S03), and, with ``m_registered_full`` (the 60 of
    :func:`registered_family_accounting`), ``p_bh_full_family``: the pooled registered BH with every contrast not
    evaluated entering as p = 1.  ``bh_family`` stays the two-way split that decides which file a row is written to, and
    ``bh_shared_computed_contrast`` flags a computed contrast that holds a section 19 slot in two families.  Registered
    contrasts are judged by R19 on the raw p; every adjusted p is shown, never used."""
    reg = set(registered_families) if registered_families is not None else set(REGISTERED_FAMILIES)
    out = table.copy()
    out["bh_family"] = np.where(out[family_col].isin(reg), "registered", "exploratory")
    out["bh_family_name"] = out[family_col].astype(str)
    for c in ("p_bh", "p_bh_pooled_registered", "p_bh_full_family", "bh_m", "bh_m_pooled_registered"):
        out[c] = np.nan
    prim = out["primary_cluster_unit"].astype(bool) if "primary_cluster_unit" in out.columns else pd.Series(True, index=out.index)
    for fam in sorted(set(out.loc[prim, family_col].astype(str))):            # BH per family: the registered reading
        sel = prim & (out[family_col].astype(str) == fam)
        pv = out.loc[sel, p_col].to_numpy(dtype=float)
        out.loc[sel, "p_bh"] = ET.benjamini_hochberg(pv)
        out.loc[sel, "bh_m"] = float(len(pv))
    for split in ("registered", "exploratory"):                               # printed beside: the pooled adjustment
        sel = prim & (out["bh_family"] == split)
        if not sel.any():
            continue
        pv = out.loc[sel, p_col].to_numpy(dtype=float)
        out.loc[sel, "p_bh_pooled_registered"] = ET.benjamini_hochberg(pv)
        out.loc[sel, "bh_m_pooled_registered"] = float(len(pv))
        if split == "registered" and m_registered_full is not None:
            pad = max(0, int(m_registered_full) - len(pv))
            out.loc[sel, "p_bh_full_family"] = ET.benjamini_hochberg(np.concatenate([pv, np.ones(pad)]))[:len(pv)]
    keys = [c for c in COMPUTED_CONTRAST_KEY if c in out.columns]
    if keys:
        n_fam = out[prim].groupby(keys, dropna=False)[family_col].transform("nunique")
        out["bh_shared_computed_contrast"] = False
        out.loc[prim, "bh_shared_computed_contrast"] = (n_fam > 1).to_numpy()
    return out


# --------------------------------------------------------------------------------------------- #
# decisions: stop rule (section 7 item 4), ladder (section 6), S1(a) / (b) components (section 9)
# --------------------------------------------------------------------------------------------- #

def stop_rule(m2_vs_b3i: Mapping[str, Any] | None, b6_vs_b3i: Mapping[str, Any] | None) -> dict[str, Any]:
    """If neither M2 nor B6 passes R19 items 1-3 and 5 against B3i on the V5-primary selection half (seed 104729), M3-M6
    are not run as registered steps, M7 runs once for H6 and H7 is not run.  ``stop`` is None while an arm that could
    still pass is missing."""
    def passes(x: Mapping[str, Any] | None) -> bool | None:
        return None if x is None else x["scopes"]["stop_rule"]["verdict"] == "PASS"
    m2, b6 = passes(m2_vs_b3i), passes(b6_vs_b3i)
    if m2 or b6:
        stop = False
    elif m2 is None or b6 is None:
        stop = None
    else:
        stop = True
    return {"rule": "section 7 compute plan item 4: stop when neither M2 nor B6 passes R19 items 1-3 and 5 against B3i "
                    "on the V5-primary selection half (seed 104729)",
            "evaluated_on": {"design": "V5", "variant": "primary", "half": "selection", "seed": PRIMARY_SEED,
                             "items": [1, 2, 3, 5]},
            "M2_vs_B3i": None if m2_vs_b3i is None else m2_vs_b3i["scopes"]["stop_rule"],
            "B6_vs_B3i": None if b6_vs_b3i is None else b6_vs_b3i["scopes"]["stop_rule"],
            "stop": stop,
            "consequences": ([] if stop is not True else
                             ["M3-M6 not run as registered steps (exploratory only, labelled)",
                              "M7 run once on the retained configuration for H6", "H7 (process) not run: S1 cannot pass"]),
            "pending": [a for a, v in (("M2", m2), ("B6", b6)) if v is None]}


def ladder_step(step: str, predecessor: str, v5: Mapping[str, Any] | None, v1_tost: Mapping[str, Any] | None,
                v2_tost: Mapping[str, Any] | None) -> dict[str, Any]:
    """Section 6: a step is kept only if it passes R19 against its retained predecessor on the V5-primary selection half
    (seed 104729: items 1-3, 5 and the scoring-filter sensitivities, section 7 item 2) and is non-inferior on the V1 and
    V2 selection halves (TOST, epsilon 0.05).  ``kept`` is None while an input is missing."""
    v5_ok = None if v5 is None else v5["scopes"]["ladder"]["verdict"] == "PASS"
    ni = [None if t is None else bool(t.get("non_inferior")) for t in (v1_tost, v2_tost)]
    if v5_ok is False or False in ni:
        kept = False
    elif v5_ok is None or None in ni:
        kept = None
    else:
        kept = True
    return {"step": step, "predecessor": predecessor, "kept": kept,
            "v5_ladder_scope": None if v5 is None else v5["scopes"]["ladder"],
            "v1_tost": None if v1_tost is None else {k: v1_tost[k] for k in ("verdict", "non_inferior", "low_90", "high_90")},
            "v2_tost": None if v2_tost is None else {k: v2_tost[k] for k in ("verdict", "non_inferior", "low_90", "high_90")},
            "label": "selection-half ladder decision (seed 104729), optimistically biased"}


def retained_predecessor(ladder: Mapping[str, Mapping[str, Any]], step: str) -> str:
    """The last kept step before ``step`` (M0 is the base and always retained)."""
    i = LADDER.index(step)
    for prev in reversed(LADDER[:i]):
        if prev == "M0":
            return "M0"
        rec = ladder.get(prev)
        if rec is not None and rec.get("kept"):
            return prev
    return "M0"


@dataclass(frozen=True)
class ContrastSpec:
    """A registered (section 19) or exploratory contrast: candidate vs comparator on a design, its margin rule."""

    family: str
    candidate: str
    comparator: str
    design: str
    margin: str          # "delta5" | "0.05"
    note: str = ""

    @property
    def name(self) -> str:
        return f"{self.candidate} vs {self.comparator}"


REGISTERED_FAMILIES: tuple[str, ...] = ("primary", "H1b", "S1(b)", "S1(c)", "H4", "H5", "H3", "ladder",
                                        "secondary designs", "confirmation")
REGISTERED_CONTRASTS: tuple[ContrastSpec, ...] = (
    ContrastSpec("primary", "M2", "B3i", "V5", "delta5", "S1(a): M2 vs the V5 lookup comparator (B3i, section 9)"),
    ContrastSpec("H1b", "B6", "B3i", "V5", "delta5", "the same delta5 applies to H1b (section 9)"),
    ContrastSpec("S1(b)", "M2", "B0", "V5", "0.05", "constant baseline"),
    ContrastSpec("S1(b)", "M2", "B6r0", "V5", "0.05", "additive factorisation"),
    ContrastSpec("H4", "M2", "FLAT_CAT", "V5", "delta5", "architecture question (section 6)"),
    ContrastSpec("H4", "M2", "M0", "V5", "delta5", "architecture question (section 6)"),
    ContrastSpec("H4", "M3", "M2", "V5", "delta5", "not run: M3+ not implemented"),
    ContrastSpec("H4", "B6", "B6r0", "V5", "delta5", "the linear analogue (section 6)"),
    ContrastSpec("secondary designs", "M2", "B3", "V1", "0.05", "V1 comparator B3 (= B4)"),
    ContrastSpec("secondary designs", "M2", "B3i", "V2", "0.05", "V2 comparator fixed pre-seal: B3i (focus-7 summary)"),
)
#: exploratory contrasts printed beside (never deciding)
EXPLORATORY_CONTRASTS: tuple[ContrastSpec, ...] = tuple(
    ContrastSpec("exploratory", arm, comp, design, "delta5" if design == "V5" else "0.05")
    for arm in ("B6r0", "B5", "FLAT_CAT", "B8", "M1") for comp, design in (("B3i", "V5"), ("B0", "V5"))
) + tuple(ContrastSpec("exploratory", arm, "B3", "V1", "0.05") for arm in ("B5", "FLAT_CAT", "B8", "M1")) + tuple(
    ContrastSpec("exploratory", arm, "B3i", "V2", "0.05") for arm in ("B5", "FLAT_CAT", "B8", "M1"))


#: section 19 registered family, discovery stage: (family, contrast, design, how it is evaluated here)
REGISTERED_FAMILY_TABLE: tuple[dict[str, str], ...] = (
    {"family": "primary", "contrast": "M2 vs B3i", "design": "V5", "status": "evaluated"},
    {"family": "H1b", "contrast": "B6 vs B3i", "design": "V5", "status": "evaluated"},
    {"family": "S1(b)", "contrast": "M2 vs B0", "design": "V5", "status": "evaluated"},
    {"family": "S1(b)", "contrast": "M2 vs B6r0", "design": "V5", "status": "evaluated"},
    *({"family": "S1(c)", "contrast": f"direction M2 - {y}", "design": "V5-PAIR", "status": "evaluated"}
      for y in ("HEAVIER", "B3x", "B3i")),
    *({"family": "S1(c)", "contrast": f"logSF MAE {y} - M2", "design": "V5-PAIR", "status": "evaluated"}
      for y in ("FLAT", "B3i")),
    {"family": "H4", "contrast": "M2 vs FLAT_CAT", "design": "V5", "status": "evaluated"},
    {"family": "H4", "contrast": "M2 vs M0", "design": "V5", "status": "evaluated"},
    {"family": "H4", "contrast": "M3 vs M2", "design": "V5", "status": "not_run: M3+ not implemented"},
    {"family": "H4", "contrast": "B6 vs B6r0", "design": "V5", "status": "evaluated"},
    *({"family": "H5", "contrast": c, "design": d, "status": "not_run: M3-M6 not implemented"}
      for c in ("M3 vs M2", "M4 on vs off", "M6a on vs off", "M6b on vs off") for d in ("V1", "V2", "V5")),
    *({"family": "H3", "contrast": c, "design": d, "status": "not_run: section 11 job specs only"}
      for c in ("WITH - WITHOUT", "WITH - ACT_PERMUTED") for d in ("V5 Ln", "V2 Ln", "V1 Ln")),
    *({"family": "ladder", "contrast": f"{st} vs retained predecessor", "design": d,
       "status": "evaluated" if st in ("M1", "M2") else "not_run: M3+ not implemented"}
      for st in ("M1", "M2", "M3", "M4", "M5", "M6", "M7") for d in ("V5", "V1", "V2")),
    {"family": "secondary designs", "contrast": "M2 vs B3", "design": "V1", "status": "evaluated"},
    {"family": "secondary designs", "contrast": "M2 vs B3i", "design": "V2", "status": "evaluated"},
    *({"family": "secondary designs", "contrast": "M2 vs design comparator", "design": d,
       "status": "not_run: V3 / V4 / V7 folds and inner designs not built for learned arms"} for d in ("V3", "V4", "V7")),
)
#: section 19 confirmation family (never evaluated in discovery)
CONFIRMATION_ONLY_CONTRASTS: tuple[str, ...] = ("S2(a)", "S2(b)", "S2(c)")
#: addendum 1 reading 6(f): "BH-adjusted p is also printed over all 60 registered discovery contrasts of section 19,
#: with contrasts not run entered as p = 1".  Section 19 lists 57 discovery contrasts (ladder 21, H5 12, H3 6, S1(c) 5,
#: secondary designs 5, H4 4, S1(b) 2, primary 1, H1b 1) and the 3 confirmation-only S2 contrasts: the registered 60
#: counts the S2 rows, which are therefore entered as p = 1 rows of the full family (the conservative bound; task X
#: finding V-F05).  ``m_discovery`` (57) is reported beside it.  Needs POST-HOC addendum 2 to state which count 6(f) meant
REGISTERED_FULL_FAMILY_SIZE = 60
CONFIRMATION_ONLY_STATUS = ("confirmation_only: S2 on V6 is never evaluated in discovery; entered as p = 1 in the "
                            "60-contrast family of addendum 1 reading 6(f)")
if len(REGISTERED_FAMILY_TABLE) + len(CONFIRMATION_ONLY_CONTRASTS) != REGISTERED_FULL_FAMILY_SIZE:
    raise AssertionError("section 19 accounting: the family table plus the confirmation-only contrasts must number "
                         f"{REGISTERED_FULL_FAMILY_SIZE} (addendum 1 reading 6(f)), got "
                         f"{len(REGISTERED_FAMILY_TABLE)} + {len(CONFIRMATION_ONLY_CONTRASTS)}")


def registered_family_accounting(evaluated_keys: Iterable[str]) -> dict[str, Any]:
    """Section 19 accounting: every registered discovery contrast with whether it was evaluated in this scoring (a
    registered 'evaluated' contrast whose arms are not yet run is ``not_run: predictions missing``), ``m_full`` (the
    :data:`REGISTERED_FULL_FAMILY_SIZE` = 60 of addendum 1 reading 6(f), the confirmation-only S2 contrasts entered as
    p = 1 rows, which BH ``p_bh_full_family`` uses), ``m_discovery`` (the 57 discovery contrasts of the table) and the
    confirmation-only contrasts."""
    keys = {str(k) for k in evaluated_keys}
    rows = []
    for r in REGISTERED_FAMILY_TABLE:
        row = dict(r)
        if r["status"] == "evaluated":
            if r["family"] == "S1(c)":
                present = "S1(c)" in keys
            elif r["family"] == "ladder":
                present = any(k.startswith(f"{r['contrast'].split(' ')[0]} vs ") and k.endswith(f"@{r['design']}#ladder")
                              for k in keys)
            else:
                present = f"{r['contrast']}@{r['design']}" in keys
            row["status"] = "evaluated" if present else "not_run: predictions missing or incomplete"
        rows.append(row)
    m_discovery = len(rows)
    for c in CONFIRMATION_ONLY_CONTRASTS:
        rows.append({"family": "confirmation", "contrast": c, "design": "V6", "status": CONFIRMATION_ONLY_STATUS})
    n_eval = sum(r["status"] == "evaluated" for r in rows)
    if len(rows) != REGISTERED_FULL_FAMILY_SIZE:
        raise AssertionError(f"section 19 family: {len(rows)} rows, {REGISTERED_FULL_FAMILY_SIZE} registered")
    return {"m_full": len(rows), "m_discovery": m_discovery, "m_evaluated": n_eval, "contrasts": rows,
            "confirmation_only": list(CONFIRMATION_ONLY_CONTRASTS),
            "full_family_reading": READINGS["full_family_60"], "bh_note": READINGS["bh_p"]}


#: section 4 / 12 uncertainty quantities that need a predictive SD or a method not built (task X finding V-10)
UNCERTAINTY_NOT_RUN: dict[str, str] = {
    "gaussian_crps": "NOT_RUN: no learned arm has a predictive SD (std_logD NaN; split-conformal intervals only)",
    "spearman_abs_error_vs_sd": "NOT_RUN: no predictive SD",
    "sd_binned_reliability_curve": "NOT_RUN: no predictive SD",
    "knows_when_it_does_not_know": "NOT_RUN: its first condition needs Spearman(|error|, SD)",
    "section_12_methods_not_built": ("bootstrap ensembles over publication groups, deep ensembles (M7), heteroscedastic "
                                     "Gaussian head, CatBoost quantile loss, CV+ conformal, Mondrian conformal by "
                                     "domain status, conjugate Bayesian ridge head (B6), MC dropout"),
}


def margin_value(spec: ContrastSpec, delta5: float) -> float:
    if spec.margin == "delta5":
        return float(delta5)
    return float(spec.margin)


def registered_delta5(difficulty_json: Path | None = None) -> float:
    """``delta5`` of the pre-seal difficulty record (section 9), re-derived from L5 and asserted equal."""
    p = difficulty_json or (paths.G19_ROOT / "evaluation" / "preseal" / "difficulty.json")
    body = json.loads(Path(p).read_text(encoding="utf-8"))
    d5, l5 = float(body["V5"]["delta5"]), float(body["V5"]["L5"])
    if not math.isclose(d5, ET.delta5(l5), rel_tol=1e-12):
        raise AssertionError(f"difficulty.json delta5 {d5} != max(0.05, 0.20 x (L5 - N0)) = {ET.delta5(l5)}")
    if body["V5"]["lookup_comparator"]["choice"] != "B3i":
        raise AssertionError("the registered V5 lookup comparator is B3i (section 9)")
    return d5


NOT_EVALUATED = "NOT_EVALUATED"


def s1d_component(macro_coverage: Mapping[float, Any] | None, category_coverage_80: Mapping[str, Any] | None, *,
                  arm: str, source: str) -> dict[str, Any]:
    """Section 9 S1(d) as a decided component: the V5-primary unit-macro 50 / 80 / 95 % coverage inside its band and, in
    every domain-status category with >= 20 scored cells, 80 % coverage inside [0.65, 0.92] (``calibration.s1d_check``).

    Task X finding V-S1-01: S1 is "All of" (a)-(e), so this returns a verdict -- or :data:`NOT_EVALUATED` with the
    reason when a coverage level is missing -- never silence.  ``macro_coverage`` maps 0.50 / 0.80 / 0.95 to the
    coverage, ``category_coverage_80`` maps a domain-status category to ``(coverage_80, scored_cells)``."""
    from gen19ct.evaluation import calibration as EC

    levels = dict(macro_coverage or {})
    missing = [lvl for lvl in EC.S1D_BANDS if not np.isfinite(float(levels.get(lvl, float("nan"))))]
    base = {"component": "S1(d) calibration (section 9), V5-primary cells, unit macro", "arm": arm, "source": source,
            "bands": {f"coverage_{int(round(l * 100))}": list(b) for l, b in EC.S1D_BANDS.items()},
            "category_band_80": list(EC.S1D_CATEGORY_BAND_80), "category_min_cells": EC.S1D_CATEGORY_MIN_CELLS,
            "label": "discovery, optimistically biased (selection half); the confirmation half decides S1",
            "reading": READINGS["s1d_s1e"]}
    if missing:
        return {**base, "verdict": NOT_EVALUATED,
                "reason": f"no V5-primary unit-macro coverage for {arm} at {sorted(missing)} in {source}"}
    cats = {str(k): (float(v[0]), int(v[1])) for k, v in dict(category_coverage_80 or {}).items()
            if np.isfinite(float(v[0]))}
    chk = EC.s1d_check({float(l): float(levels[l]) for l in EC.S1D_BANDS}, cats)
    below = {str(k): int(v[1]) for k, v in sorted(dict(category_coverage_80 or {}).items())
             if int(v[1]) < EC.S1D_CATEGORY_MIN_CELLS}
    return {**base, "verdict": "PASS" if chk["pass"] else "FAIL", "items": chk["items"],
            "categories_below_scope": below,
            "categories_below_scope_note": f"a domain-status category with fewer than {EC.S1D_CATEGORY_MIN_CELLS} "
                                           "scored cells is outside the band's registered scope"}


def s1e_component(spearman_point: float | None = None, *, n_cells: int | None = None, arm: str = "M2",
                  source: str = "") -> dict[str, Any]:
    """Section 9 S1(e) as an explicit :data:`NOT_EVALUATED` component with its reason (task X finding V-S1-01).

    S1(e) needs Spearman(cell MAE, ``support_score``) <= -0.10 with the system-cluster bootstrap interval excluding 0
    AND the support-score components above the section 8 reliability floor (``transfer.s1e_check``).  The floor is
    estimated by the power stage (``evaluation.power``, reading ``reliability_not_fitted``: the support-score components
    are not fitted by default), which is not implemented in this runner, so the component is NOT_EVALUATED however the
    correlation comes out; the descriptive point is printed beside and decides nothing."""
    out = {"component": "S1(e) error grows with support distance (section 9), V5-primary cells", "arm": arm,
           "verdict": NOT_EVALUATED, "threshold": ET.S1E_MAX_SPEARMAN,
           "reason": ("the registered check needs the system-cluster bootstrap interval of Spearman(cell MAE, "
                      "support_score) AND the support-score components above the section 8 reliability floor; the "
                      "floor is the power stage (evaluation.power, 'reliability_not_fitted': support-score components "
                      "are not fitted by default), whose execution is not implemented in this runner"),
           "components_reliable": None, "interval": None, "reading": READINGS["s1d_s1e"],
           "label": "discovery, optimistically biased (selection half)"}
    if spearman_point is not None and np.isfinite(float(spearman_point)):
        out["descriptive_spearman"] = {"point": float(spearman_point), "n_cells": None if n_cells is None else int(n_cells),
                                       "source": source, "decides": False,
                                       "note": "descriptive only: no interval and no reliability floor, so it is not "
                                               "the registered S1(e) statistic"}
    return out


def budget_block(out_root: Path, *, run_manifest: Path | None = None) -> dict[str, Any]:
    """Section 7 item 5 and POST-HOC addendum 2's ladder budget as one disclosed block (task X finding V-BUD-03).

    Reads the discovery ledger (``evaluation/discovery/decisions/wall_clock.json``) and the run manifest's ledger, and
    names BOTH reasons M3-M7 are absent: the ledger demoted them by rule when the 60 h budget was exhausted, and they
    are not implemented in this runner -- while ``ledger.demoted`` is empty, i.e. no step was removed on evidence.  The
    ladder's own 40 h ledger of addendum 2 is reported with whether it exists yet."""
    wc = read_record(discovery_root(out_root) / "decisions" / "wall_clock.json") or {}
    man = read_record(paths.MANIFESTS_DIR / "g19_run_discovery.json" if run_manifest is None else Path(run_manifest)) or {}
    b = dict(wc.get("budget") or {})
    ledger = dict(man.get("ledger") or {})
    lb = dict(ledger.get("budget") or {})
    lwc = read_record(Path(out_root) / LADDER_WALL_CLOCK_REL) or {}
    demoted_now = list(b.get("demoted_now") or lb.get("demoted_now") or [])
    return {
        "rule": READINGS["budget"], "ladder_reading": READINGS["budget_ladder"],
        "discovery": {"budget_hours": b.get("budget_hours", lb.get("budget_hours", BUDGET_HOURS)),
                      "used_hours": b.get("used_hours", lb.get("used_hours")),
                      "total_hours": wc.get("total_hours"),
                      "exhausted": b.get("exhausted", lb.get("exhausted")),
                      "exhausted_before": lb.get("exhausted_before"),
                      "demotion_order": list(b.get("demotion_order") or DEMOTION_ORDER),
                      "demoted_now": demoted_now, "never_demoted": list(b.get("never_demoted") or NEVER_DEMOTED),
                      "n_invocations": len(wc.get("invocations") or []),
                      "ledger_demoted": list(ledger.get("demoted") or []), "ledger_stopped": ledger.get("stopped"),
                      "ledger": "evaluation/discovery/decisions/wall_clock.json"},
        "ladder": {"budget_hours": LADDER_BUDGET_HOURS, "ledger": LADDER_WALL_CLOCK_REL,
                   "ledger_exists": bool(lwc), "used_hours": lwc.get("total_hours"),
                   "note": "POST-HOC addendum 2, '2. Ladder M3-M7' -> Budget: the ladder's own 40 h wall clock, checked "
                           "before each step; the discovery ledger stays at the registered 60 h and the two are never "
                           "added"},
        "m3_m7_absence": {"not_implemented": NOT_IMPLEMENTED,
                          "demoted_by_discovery_ledger": demoted_now,
                          "removed_on_evidence": list(ledger.get("demoted") or []),
                          "disclosure": ("M3-M7 are absent for two independent recorded reasons: they are not "
                                         "implemented in this runner, AND the discovery ledger reached "
                                         f"{b.get('used_hours')} h of the {b.get('budget_hours', BUDGET_HOURS)} h "
                                         "budget and demoted them by the section 7 item 5 rule (order M7 -> M3). No "
                                         "step was removed on evidence (ledger.demoted is empty), and addendum 2 gives "
                                         f"the ladder its own {LADDER_BUDGET_HOURS:g} h budget, so the demotion does "
                                         "not bind the ladder runner")}}


def s1ab_components(results: Mapping[str, Mapping[str, Any]], state: PlanState | None = None, *,
                    s1d: Mapping[str, Any] | None = None, s1e: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Section 9 S1(a) and S1(b) components from the evaluated contrasts ``M2 vs B3i`` (delta5), ``M2 vs B0`` and ``M2 vs
    B6r0`` (0.05), labelled discovery / optimistically biased; the confirmation half decides S1.  When the re-coloured
    batched-vs-exact check failed (``state.s1_forced_undecided``, section 7 item 6) every component's
    ``reported_verdict`` is UNDECIDED and carries the label ``batched (check failed)``.

    ``s1d`` / ``s1e`` (:func:`s1d_component`, :func:`s1e_component`) are carried through so that every one of the five
    registered S1 components has a verdict or an explicit :data:`NOT_EVALUATED` (task X finding V-S1-01); omitting them
    records NOT_EVALUATED with the reason that the scorer did not compute them."""
    forced = bool(state is not None and state.s1_forced_undecided)
    label = state.heavy_v5_label if state is not None else ""

    def comp(key: str) -> dict[str, Any] | None:
        r = results.get(key)
        if r is None:
            return None
        return {"point": r["point"], "margin": r["margin"], "r19_full": r["r19"].verdict,
                "reported_verdict": "UNDECIDED" if forced else r["r19"].verdict, "batching_label": label,
                "scopes": {k: v["verdict"] for k, v in r["scopes"].items()},
                "items": [{"item": i["item"], "status": i["status"], "detail": i["detail"]} for i in r["r19"].items]}
    return {"label": "discovery, optimistically biased (selection half; S1 is decided on the confirmation half with the "
                     "withheld seeds, section 15)",
            "S1_forced_undecided": forced,
            "S1_forced_undecided_reason": ("section 7 item 6: the re-coloured batched-vs-exact check failed; every "
                                           "heavy-arm V5 result is labelled 'batched (check failed)' and S1 is reported "
                                           "UNDECIDED, never passed") if forced else None,
            "S1_components_registered": ["S1(a)", "S1(b)", "S1(c)", "S1(d)", "S1(e)"],
            "S1_rule": "section 9: S1 holds only if ALL of (a)-(e) hold; a component without a verdict is "
                       f"{NOT_EVALUATED} with its reason, never omitted (task X finding V-S1-01)",
            "S1a_M2_vs_B3i": comp("M2 vs B3i@V5"), "S1b_M2_vs_B0": comp("M2 vs B0@V5"),
            "S1b_M2_vs_B6r0": comp("M2 vs B6r0@V5"),
            "S1d_calibration": dict(s1d) if s1d is not None else
            {"component": "S1(d) calibration (section 9)", "verdict": NOT_EVALUATED,
             "reason": "the scorer did not compute the V5-primary coverage bands in this pass"},
            "S1e_support_distance": dict(s1e) if s1e is not None else s1e_component()}


POWER_CHECK_FAMILIES: tuple[str, ...] = ("primary", "H1b", "S1(b)", "H3")


def power_check_plan(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Section 8: before a failed H1, H1b or H3 contrast is reported as a null, the pipeline is re-run on injected
    targets ``y' = y + kappa s`` (kappa in {0.1, 0.25, 0.5, 1.0}; X(?) rows dropped from every injected refit; models
    refitted at their selected hyperparameters; the same scored rows).  Lists the contrasts that need it now."""
    need = []
    for key, r in sorted(results.items()):
        if r["family"] in POWER_CHECK_FAMILIES and r["scopes"]["full"]["verdict"] != "PASS":
            need.append({"contrast": key, "family": r["family"], "arms": [r["candidate"], r["comparator"]],
                         "full_verdict": r["scopes"]["full"]["verdict"]})
    return {"rule": "section 8 signal-injection power check (a null must be informative)", "kappas": list(ET.KAPPAS),
            "kappa_min_informative": ET.KAPPA_MIN_INFORMATIVE, "required_before_reporting_a_null": need,
            "status": "not implemented in this runner (open issue); a null without it is reported UNDECIDED",
            "reading": READINGS["power_check"]}


def heavy_v5_batching_label(arms: Iterable[str], design: str, state: PlanState | None) -> str:
    """The V5 fold-scheme label of a contrast: the section 7 item 6 heavy-arm label when a heavy arm is involved, else
    ``exact`` for a B6 / B6r0 contrast (section 3.1: B6 and B6r0 are scored on the exact leave-one-cell-out folds, and
    the registered ``comparator_folds`` reading fits the closed-form comparator on them too), else ``""``.

    Task X finding V-H1B-05: the B6 rows carried a BLANK label while the heavy-arm rows said ``batched``, which reads as
    if H1 and H1b sat on one regime; H1b is exact leave-one-cell-out and H1 is the batched scheme."""
    if state is None or design not in ("V5", "V5-P", "V5P", "V5-PAIR", "V5PAIR"):
        return ""
    aliased = [ARM_ALIASES.get(a, a) for a in arms]
    if any(a in HEAVY_ARMS for a in aliased):
        return state.heavy_v5_label
    return "exact" if any(a in B6_ARMS for a in aliased) else ""


#: POST-HOC addendum 3 item 1 (section 15): what "passed R19 in discovery" means when a claim is frozen.  The literal
#: reading is unsatisfiable under addendum 1 item 3 -- R19 item 4 is NOT_EVALUATED in discovery for EVERY learned arm, so
#: no contrast could be frozen and the confirmation run (with it the single V6 run of section 3.4, S2 and the applied
#: Pr/Nd question of brief section 34) could never happen
FREEZING_RULE = ("POST-HOC addendum 3 item 1 (section 15): a contrast is ELIGIBLE FOR FREEZING when every R19 item that "
                 "R19 EVALUATES in discovery is PASS and no item is FAIL. An item R19 does not evaluate in discovery -- "
                 "item 4, NOT_EVALUATED by addendum 1 item 3 (one discovery seed), and any UNTESTABLE / NOT_RUN item -- "
                 "does not disqualify it; item 4 is evaluated at confirmation exactly as registered (5 of 5 withheld "
                 "seeds). Any FAIL, on any item, makes the contrast ineligible. This is the reading addendum 2 already "
                 "gives F6 for deployment, and it would have been required whatever the scores were. Everything else "
                 "about section 15 is unchanged: at most five claims, named in decisions/CONFIRMATION_PLAN.md before the "
                 "run, one run on the withheld seeds and the confirmation half, V6 once.")

#: the literal section 3.1 trigger of the heavy-arm V5-P runs (R19 items 1-5 on V5-primary), kept as its OWN field: it is
#: NOT the freezing rule (addendum 3 item 1) and, under addendum 1 item 3, it cannot fire in discovery.  The V5-P runs are
#: not run in any case (addendum 1 item 4 names V5-P among the sensitivities not run)
V5P_TRIGGER_RULE = ("section 3.1: the heavy-arm V5-P run of a freezing candidate is triggered by R19 items 1-5 on "
                    "V5-primary. Item 4 is NOT_EVALUATED in discovery (addendum 1 item 3), so the trigger cannot fire; "
                    "addendum 1 item 4 does not run the heavy-arm V5-P runs either way. Addendum 3 item 1 changes what "
                    "'passed R19 in discovery' means for FREEZING (section 15), not this trigger, so the two are "
                    "reported separately.")


def freezing_eligibility(res: Mapping[str, Any]) -> dict[str, Any]:
    """POST-HOC addendum 3 item 1: is this contrast eligible for freezing, item by item (:data:`FREEZING_RULE`)?

    ``eligible`` is True when no R19 item is FAIL and every item R19 evaluated in discovery is PASS.  The items R19 did
    not evaluate (:data:`INCONCLUSIVE_STATUSES`: ``NOT_EVALUATED`` of item 4, ``UNTESTABLE``, ``NOT_RUN``) are listed
    with their status in ``items_not_evaluated_in_discovery`` and named in ``reason``, so the flag is never read without
    the reason it holds."""
    items = {int(i["item"]): str(i["status"]) for i in res["r19"].items}
    failed = [i for i, s in sorted(items.items()) if s == "FAIL"]
    not_evaluated = {i: s for i, s in sorted(items.items()) if s in INCONCLUSIVE_STATUSES}
    evaluated = {i: s for i, s in sorted(items.items()) if i not in not_evaluated}
    eligible = not failed and all(s == "PASS" for s in evaluated.values())
    passed = [i for i, s in sorted(evaluated.items()) if s == "PASS"]
    other = {i: s for i, s in sorted(evaluated.items()) if s != "PASS"}
    parts = [f"items {passed} PASS" if passed else "no item PASS"]
    if not_evaluated:
        parts.append("not evaluated in discovery: " + ", ".join(f"item {i} {s}" for i, s in not_evaluated.items()))
    parts.append("no item FAIL" if not other else "FAIL: " + ", ".join(f"item {i} {s}" for i, s in other.items()))
    return {"eligible": bool(eligible), "items": items, "evaluated_items": evaluated,
            "items_not_evaluated_in_discovery": not_evaluated, "failed_items": failed,
            "r19_verdict_full": res["r19"].verdict, "reason": "; ".join(parts), "rule": FREEZING_RULE}


def freezing_candidates(results: Mapping[str, Mapping[str, Any]], state: PlanState | None = None) -> list[dict[str, Any]]:
    """Registered contrasts that pass the freezing screen (R19 items 1-3, 5 and the scoring-filter sensitivities on seed
    104729), with their design, whether they are ELIGIBLE FOR FREEZING (POST-HOC addendum 3 item 1,
    :func:`freezing_eligibility`, with the item-by-item reason), the literal section 3.1 V5-P trigger
    (:data:`V5P_TRIGGER_RULE`), the section 7 item 6 batching label and whether S1 is forced UNDECIDED for an S1
    family."""
    out = []
    for key, r in sorted(results.items()):
        if r["family"] not in REGISTERED_FAMILIES or r["scopes"]["freezing_screen"]["verdict"] != "PASS":
            continue
        arms = [r["candidate"], r["comparator"]]
        elig = freezing_eligibility(r)
        out.append({"contrast": key, "family": r["family"], "arms": arms, "design": r["design"],
                    "eligible_for_freezing": elig["eligible"], "eligibility": elig,
                    "section_3_1_v5p_trigger": bool(r["design"] == "V5"
                                                    and r["scopes"]["items_1_5"]["verdict"] == "PASS"),
                    "section_3_1_v5p_trigger_rule": V5P_TRIGGER_RULE,
                    "batching_label": heavy_v5_batching_label(arms, r["design"], state),
                    "s1_forced_undecided": bool(state is not None and state.s1_forced_undecided
                                                and r["family"] in S1_FAMILIES)})
    return out


def b6_check(exact: Mapping[int, pd.Series], batched: Mapping[int, pd.Series], *,
             threshold: float = 0.01, seeds: Sequence[int] = PLAN_SEEDS) -> dict[str, Any]:
    """The per-seed registered check (sections 3.1 / 3.2): ``factorized.batched_exact_check`` logic on per-unit MAE of
    identical units; passes only when every seed of ``seeds`` is present and passes (:data:`READINGS`
    ``b6_batched_check``).  ``seeds`` defaults to the seeds discovery runs -- under addendum 1 item 3 seed 104729
    alone; pass :data:`DISCOVERY_SEEDS` for the five-seed reading of the sealed plan."""
    per = {}
    for s in seeds:
        if s not in exact or s not in batched:
            continue
        ex, ba = exact[s].astype(float), batched[s].astype(float)
        if set(ex.index) != set(ba.index):
            raise ValueError(f"seed {s}: the exact and batched designs score different units")
        e, b = float(ex.mean()), float(ba.loc[ex.index].mean())
        per[int(s)] = {"n_units": int(len(ex)), "exact_macro_mae": e, "batched_macro_mae": b, "delta": b - e,
                       "abs_delta": abs(b - e), "passed": bool(abs(b - e) < threshold)}
    want = [int(s) for s in seeds]
    complete = len(per) == len(want)
    passed = complete and all(v["passed"] for v in per.values())
    return {"threshold": threshold, "per_seed": per, "seeds": want, "complete": complete,
            "passed": passed if complete else None, "reading": READINGS["b6_batched_check"]}


def next_v5_check_state(current: str, first: Mapping[str, Any], second: Mapping[str, Any] | None) -> str:
    """Section 7 item 6 state machine: pending -> passed | recolour; recolour -> passed_after_recolour | failed."""
    if first.get("passed") is None:
        return "pending"
    if first["passed"]:
        return "passed"
    if second is None or second.get("passed") is None:
        return "recolour"
    return "passed_after_recolour" if second["passed"] else "failed"

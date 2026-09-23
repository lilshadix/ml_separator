"""``evaluation/h3.py`` -- the section 11 actinide ablation (H3; brief sections 15 and 30) and the section 10 F4 check.

Nothing here fits a model or reads a discovery record to decide anything: ``scripts/g19_run_h3.py`` fits the ablation
arms AFTER discovery is complete and calls the functions below; the readers verify records against the digests the
current code produces, as ``gen19ct.evaluation.discovery`` does.

Design (section 11, implemented as follows)
-------------------------------------------
* **Model arms** (:func:`model_arms`): the configuration deployed for lanthanide prediction
  (:func:`deployed_configuration`, read from the ladder runner's ``evaluation/ladder/decisions/ladder.json`` AND the
  scorer's ``decisions.json``: the LADDER'S RETAINED CONFIGURATION -- the highest kept ladder step
  M7 > M6 > M5 > M4 > M3 (ladder.json, a registered -- not stop-rule exploratory -- run) > M2 > M1 > **M0 (= B5)**
  (decisions.json; POST-HOC addendum 3 item 2: the ladder kept M0 and dropped M1 and M2, so M0 is what it retains), and
  only when the ladder has NO retained step addendum 2's remaining order: M2 when it passes the stop-rule scope against
  B3i, else B6 when it does, else the registered V5 lookup comparator B3i as the F6 "best-passing baseline") plus B6 and
  B5 as transparent references (``discovery.H3_MODEL_ARMS``; with M0 deployed the model arms are B5 and B6, since M0 IS
  B5).  The rule is undecidable (the runner refuses) until the ladder has run every step to a done / skipped status
  (:func:`ladder_complete`; task X finding V-01).
* **Transforms** (:data:`TRANSFORMS`, ``discovery.h3_training_rows``): WITH is the discovery record of the same arm,
  design, seed and fold (same architecture, same folds and batches, same seeds -- nothing is refitted for it); WITHOUT
  removes every actinide training row (unknown-state actinide rows included); ACT_PERMUTED permutes ``log_D`` among
  actinide training rows within (system, publication group).  **POST-HOC addendum 4 item 1**: ACT_METAL_SHUFFLED (which
  shuffles the actinide metal-state labels within a system) is NOT RUN -- no registered delta, verdict or failure
  condition reads it -- and the one fold already fitted is :data:`EXPLORATORY_NOT_SCORED`
  (:data:`EXPLORATORY_TRANSFORMS`, :func:`exploratory_records`); it is kept on disk and enters no contrast, table or
  verdict.  The transformed arms are refitted at the WITH run's SELECTED hyperparameters of the same fold
  (:class:`FrozenNeural`, :class:`FrozenBoosted`, :class:`FrozenB6`, :class:`FrozenLadder` for a deployed ladder step
  M3-M7 whose WITH record is its ``evaluation/ladder`` record; the closed-form comparators need none and are fitted on
  the exact leave-one-cell-out folds of section 3.1 as in discovery), with the cross-fitted split-conformal calibration
  of the discovery runner at the cross-fit selections of the WITH record (:data:`READINGS` ``tuning``).
* **Ln test set** (:func:`ln_rows`): the Ln(III) scored rows of the selection half -- V5-primary Ln(III) cells (V6
  carve-out and ``V6_TARGET_ROWS`` guard applied by ``discovery.scoring_frame``), every Ln(III) state of V2 in the
  selection half, the Ln(III) rows of the V1 selection folds.  Cells whose eligibility depends on actinide partners
  (:func:`actinide_dependent_cells`) are scored in both arms and reported as a stratum.
* **Designs, priority and the compute cap** (:data:`DESIGN_PRIORITY`, :data:`H3_BUDGET_HOURS`): section 11's Ln test set
  is run in full -- V5-primary Ln cells, V2 Ln folds, V1 Ln rows -- under the **POST-HOC addendum 4 item 2** cap of 20 h
  of wall clock on 2 workers, ledgered in ``evaluation/h3/decisions/wall_clock.json``
  (:func:`h3_wall_clock_path`, :func:`h3_budget_status`) and checked before each fold.  The priority order is
  V5 -> V2 -> V1 (:func:`plan_priority_key`); a design the cap does not reach is :data:`NOT_RUN` with its reason
  (:func:`not_run_design`) and **no verdict is taken from a partial design** (:func:`assert_no_partial_design`,
  asserted in the scorer).
* **Deltas**: WITH - WITHOUT and WITH - PERMUTED with the section 8 paired cluster bootstrap and
  R19 through ``discovery.evaluate_contrast`` (:func:`h3_contrast`); the reversed contrast (the transform as candidate)
  gives *hurts* and F4.  Delta macro MAE (R19), Delta rank accuracy (:func:`rank_accuracy_delta`), Delta calibration
  |cov80 - 0.80|, |cov95 - 0.95| and mean width (:func:`calibration_delta`; CRPS NOT_RUN without a predictive SD,
  ``discovery.UNCERTAINTY_NOT_RUN``), Delta logSF MAE on V5-PAIR Ln-Ln pairs when that design is run
  (:func:`logsf_delta`).
* **Verdicts** (:func:`h3_verdict`): helps / hurts / equivalent / UNDECIDED (with kappa_min of the section 8 power check
  when computed); **F4** (:func:`f4_check`).
* **Negative-transfer investigation** (:func:`system_actinide_stats`, :func:`negative_transfer_tables`): descriptive
  tables only.

Readings where section 11 is silent are listed in :data:`READINGS` (each needs a POST-HOC addendum before a result is
quoted as registered).
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gen19ct import paths
from gen19ct.chemistry import metals as MET
from gen19ct.chemistry import support_graph as SG
from gen19ct.evaluation import discovery as D
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import pairs as EP
from gen19ct.evaluation import registry as REG
from gen19ct.evaluation import transfer as ET
from gen19ct.folds import cell_holdout as CH
from gen19ct.folds import io as FI
from gen19ct.folds import registered as FR
from gen19ct.models import interface as I

SCHEMA = "gen19.h3.v1"
FAMILY = "H3"
#: the training transforms that are refitted AND SCORED (WITH is the discovery record; ``discovery.H3_ARMS`` lists all
#: four of the sealed text's).  POST-HOC addendum 4 item 1 drops ACT_METAL_SHUFFLED from the plan: section 11 registers
#: deltas only for WITH - WITHOUT and WITH - PERMUTED, so nothing reads it
TRANSFORMS: tuple[str, ...] = ("WITHOUT", "ACT_PERMUTED")
#: POST-HOC addendum 4 item 1: the transform that is NOT RUN.  The one fold already fitted before the addendum stays on
#: disk, is labelled :data:`EXPLORATORY_NOT_SCORED` in the H3 decisions and enters no contrast, table or verdict
EXPLORATORY_TRANSFORMS: tuple[str, ...] = ("ACT_METAL_SHUFFLED",)
EXPLORATORY_NOT_SCORED = "exploratory_not_scored"
ACT_METAL_SHUFFLED_NOT_RUN = (
    "POST-HOC addendum 4 item 1 (compute-driven): section 11 lists two controls but registers deltas only for "
    "WITH - WITHOUT and WITH - PERMUTED, and no registered delta, verdict or failure condition uses "
    "ACT_METAL_SHUFFLED; completing it would cost about a third of the H3 budget (the ablation refits the deployed "
    "CatBoost and its conformal calibration per fold, measured at about 620 s per fold) for a quantity nothing reads. "
    "ACT_METAL_SHUFFLED is NOT run; the single fold already fitted is labelled exploratory and is not scored. "
    "WITH - PERMUTED, which section 11 uses to separate actinide chemistry from row count, is unchanged and runs on "
    "every design")
ALL_ARMS: tuple[str, ...] = D.H3_ARMS
#: section 11 designs: token -> (design, variant); the scheme follows the WITH run (``with_design_dir``)
DESIGNS: dict[str, tuple[str, str]] = {"V5": ("V5", "primary"), "V2": ("V2", "element"), "V1": ("V1", "copy")}
PAIR_DESIGN = "V5PAIR"
#: POST-HOC addendum 4 item 2: "If the cap is reached, the priority order is V5 (done), then V2, then V1, and any design
#: not reached is reported NOT_RUN with its reason - no verdict is taken from a partial design"
DESIGN_PRIORITY: tuple[str, ...] = ("V5", "V2", "V1")
#: POST-HOC addendum 4 item 2: "H3 gets a cap of 20 h of wall clock on 2 workers, recorded in
#: evaluation/h3/decisions/wall_clock.json"
H3_BUDGET_HOURS = 20.0
H3_WORKERS = 2
NOT_RUN = "NOT_RUN"
#: POST-HOC addendum 4 item 2's ``NOT_RUN`` vocabulary belongs to the COMPUTE CAP ("If the cap is reached ... any design
#: not reached is reported NOT_RUN with its reason").  A record set the cap did not stop -- one a deterministic guard
#: failure left incomplete -- carries its OWN status, so the cause is never read as the cap (TASK F items 1 and 2)
INCOMPLETE_GUARD_FAILURE = "INCOMPLETE_GUARD_FAILURE"
#: the unit of completeness: one CONTRAST record set, ``<arm>:<transform>@<design>``
CONTRAST_UNIT_RULE = (
    "POST-HOC addendum 4 item 2, 'no verdict is taken from a partial design', applied per CONTRAST RECORD SET "
    "(<arm>:<transform>@<design>): every registered delta of section 11 is WITH - WITHOUT or WITH - PERMUTED, so the "
    "record set that a contrast is computed from is the arm x transform x design leg, and completeness is a property of "
    "THAT leg. A leg whose record set is complete is scored and carries its verdict; a leg whose record set is "
    "incomplete is reported with its status and reason and contributes no contrast row, no R19 item row, no delta, no "
    "per-unit row, no F4 entry and no non-inferiority input. A DESIGN is reported NOT_RUN only when no leg of it is "
    "complete. This narrowing of the NOT_RUN unit from the design to the record set is a READING of addendum 4 item 2 "
    "and a POST-HOC addendum stating it is REQUESTED (TASK F item 1)")
#: WHY the ACT_PERMUTED@V1 legs are incomplete, measured on fold ``pub_97510df3a0``
#: (``scripts/g19_h3_guard_diagnosis.py`` -> ``evaluation/h3/decisions/isolation_guard_diagnosis.json``).  TASK F item 2.
GUARD_FAILURE_DIAGNOSIS = (
    "MEASURED CAUSE (scripts/g19_h3_guard_diagnosis.py, fold pub_97510df3a0 of B5:ACT_PERMUTED@V1): the section 2 inner "
    "guard is fold_isolation_check(level='V1', near_dup_sig=6, near_dup_value_tol=0.005), and ACT_PERMUTED rebuilds the "
    "corpus over the PERMUTED frame, so the guard's VALUE comparison reads permuted log_D. Of the fold's three inner "
    "splits, all three pass on the registered values and two (inner_s104729_f1, inner_s104729_f2) fail on the permuted "
    "ones, each with exactly ONE near-duplicate key flagged. On f1 the flagged key (k_c75093d1f0ee) is a U(VI) row whose "
    "registered log_D 0.174458 was permuted to 0.376759, landing 0.00201 from a calibration row at 0.374748; on the "
    "registered values the nearest calibration value is 0.115576 away, an order of magnitude outside the 0.005 "
    "tolerance. The key itself cannot change: near_duplicate_key is built from the system, metal, state, acid, solvent, "
    "concentrations and temperature and never from log_D, and the publication group and archive duplicate group are "
    "stored columns, so the permutation can only move VALUES. The guard therefore flags a near-duplicate pair the "
    "recorded data never had. Running the guard VALUE-BLIND (near_dup_value_tol=None) is NOT the fix and is not what "
    "this diagnosis supports: value-blind counts every shared key whatever the values, and it fails all three inner "
    "splits on the REGISTERED frame too (2, 27 and 26 shared keys), so it would also block the WITH and WITHOUT arms "
    "that scored. The conservative behaviour taken until 2026-09-23: the two ACT_PERMUTED@V1 legs were left UNFITTED, no "
    "fold was forced, and they carried " + INCOMPLETE_GUARD_FAILURE + " with no verdict. RESOLVED by POST-HOC addendum 5 "
    "item 2, which registers the rule below; the legs are completed under it and the incompleteness is history")
#: POST-HOC addendum 5 item 2 (REGISTERED, quoted): for an arm whose training target is permuted by construction, the
#: section 2 guard's near-duplicate VALUE comparison reads the corpus's RECORDED log D.  Implemented in
#: ``scripts/g19_run_h3.py`` (``recorded_value_guards``): the outer guard and the inner isolation check of a
#: value-permuting transform are bound to the RECORDED corpus, whose frame differs from the permuted one in ``log_D``
#: ALONE -- so the publication group, the archive duplicate group and the 6-significant-figure near-duplicate key (built
#: from system, metal, state, acid, solvent, concentrations and temperature, never from ``log_D``) are bit-identical, and
#: only the value comparison changes.  The guard therefore stays exactly as strict as registered and becomes invariant
#: under the permutation.  It gates fitting only and changes no prediction, so the ACT_PERMUTED records written before
#: this rule (V5 and V2, which the permuted-value guard passed anyway) are unaffected and are not refitted.
GUARD_VALUE_SOURCE_RULE = (
    "REGISTERED (POST-HOC addendum 5 item 2): 'for an arm whose training target is permuted by construction, the value "
    "comparison reads the corpus's recorded log D; every other level of the guard - publication group, archive duplicate "
    "group, and the near-duplicate key at six significant figures - is value-independent and unchanged, and the "
    "value-blind mode stays what section 2 registers it as, a sensitivity'. Implementation: the outer guard and the "
    "inner isolation check of a value-permuting transform are bound to the RECORDED corpus (g19_run_h3."
    "recorded_value_guards), which differs from the permuted corpus in log_D alone. The guard gates FITTING and changes "
    "no prediction: the ACT_PERMUTED records written before the rule (V5, V2) are unaffected and are not refitted, and "
    "the 14 V1 folds the permuted-value comparison had blocked are completed under it")
#: the transforms whose TRAINING TARGET is permuted by construction, i.e. the arms addendum 5 item 2 names
VALUE_PERMUTING_TRANSFORMS: frozenset[str] = frozenset({"ACT_PERMUTED"})


def guard_value_source(transform: str) -> str:
    """Which ``log_D`` the section 2 guard's near-duplicate VALUE comparison reads for ``transform`` (addendum 5 item 2):
    the corpus's RECORDED values for a value-permuting control arm, the arm's own frame otherwise (they are the same
    frame for every transform that does not move ``log_D``)."""
    return "recorded_log_D" if str(transform) in VALUE_PERMUTING_TRANSFORMS else "arm_frame_log_D"
#: section 11 transparent references
REFERENCE_ARMS: tuple[str, ...] = ("B6", "B5")
#: the fallback "deployed" arm when no learned configuration passes (section 10 F6: the best-passing baseline; the
#: registered V5 lookup comparator, ``preseal difficulty.json -> V5.lookup_comparator``)
FALLBACK_DEPLOYED = "B3i"
DETERMINISTIC_ARMS: frozenset[str] = frozenset({"B0", "B1", "B2", "B3", "B3x", "B3i", "B3l", "B4", "B4x", "B4l", "B7"})
#: the ladder steps ``scripts/g19_run_ladder.py`` fits (their WITH records live under ``evaluation/ladder``)
LADDER_ARMS: tuple[str, ...] = ("M3", "M4", "M5", "M6", "M7")
#: the section 6 ladder's BASE step, ``decisions.json -> ladder.M0.kept`` (``discovery.retained_predecessor``: "M0 is the
#: base and always retained").  POST-HOC addendum 3 item 2: the deployed configuration is the ladder's RETAINED
#: configuration, and that includes M0 (= B5, ``discovery.ARM_ALIASES``) when neither M1 nor M2 is kept
LADDER_BASE = "M0"
#: the ladder-state statuses that end a step (``g19_run_ladder.STATUS_DONE`` + ``STATUS_SKIPPED`` without the H5 block's
#: ``complete``; ``tests/test_h3.py`` asserts the two agree): a step in any other status (running, incomplete,
#: undecided, absent) leaves the deployed configuration undecidable
LADDER_DONE_STATUSES: frozenset[str] = frozenset({"kept", "removed", "judged", "exploratory_not_run", "not_run", "demoted"})
LADDER_STATE_FILE = "evaluation/ladder/decisions/ladder.json"
#: the registered contrast key of each stop-rule contrast, for the F6 fallback's item-6 check (task X finding V-P09)
F6_CONTRAST_OF: dict[str, str] = {"M2_vs_B3i": "M2 vs B3i@V5", "B6_vs_B3i": "B6 vs B3i@V5"}
F6_FALLBACK_RULE = ("POST-HOC addendum 2 (F6): with no retained ladder step, M2 (and B6 for H1b) 'is deployed by the "
                    "fallback only when its contrast passes the stop-rule scope (items 1, 2, 3, 5) AND item 6 over the "
                    "reduced sensitivity set - no evaluated item may FAIL'. decisions.json -> stop_rule evaluates items "
                    "1, 2, 3 and 5 only, so the second half is read from decisions.json -> freezing_candidates, whose "
                    "screen is that scope plus item 6 with no evaluated item FAILing (discovery.freezing_candidates / "
                    "freezing_eligibility). A contrast the scorer did not list did not pass that screen; a decisions.json "
                    "with no freezing_candidates block leaves item 6 unchecked and the basis says so.")
#: section 10 F4 says "95 % interval excluding 0" in the singular and names no construction; section 8 registers the
#: percentile AND the BCa interval for every contrast and R19 item 2 requires both.  Which interval a FAILURE condition
#: reads is therefore an undisclosed reading, and an undisclosed reading may not decide a failure condition silently:
#: :func:`f4_check` keeps the percentile value as the decided one and reports the BCa reading beside it until a POST-HOC
#: addendum names the interval (task X finding protocol VH-02 / numbers VH-03)
F4_INTERVAL_READING = (
    "REGISTERED by POST-HOC addendum 5 item 3 (NOT REGISTERED before it, which is why the reading is spelled out here): "
    "section 10 F4 says '95 % interval excluding 0' without naming a construction, while section 8 "
    "registers the percentile and the BCa 95 % interval for every contrast and R19 item 2 requires BOTH to exclude 0. "
    "F4 is a FAILURE condition, so the CONSERVATIVE reading is the one that triggers more readily -- EITHER registered "
    "interval, on ANY registered design -- and that reading GOVERNS the headline (failure == failure_either_interval). "
    "The percentile-only and BCa-only readings are computed and reported beside it (failure_percentile, failure_bca, "
    "per_design.<design>.bca_beats), and a design where they disagree is named (designs_where_readings_disagree), so "
    "every F4 statement carries both. Addendum 5 item 3 states this reading in the same "
    "words ('section 8 registers the percentile and the BCa interval for every contrast, so F4 is read under both, and "
    "the conservative outcome governs') and records the outcome: on V2 the BCa interval of WITHOUT vs WITH excludes zero "
    "(+0.0264, [+0.0031, +0.0783]) while the percentile interval does not, V5 and V1 show nothing, so F4 HOLDS. The "
    "headline is the conservative one and is never softened to the percentile reading (TASK F item 3)")
#: POST-HOC addendum 5 item 3, second half: the two readings of F4 that must be printed SIDE BY SIDE, because section 11's
#: own consequence ("actinide rows enter the deployed Ln configuration only on helps") makes the deployed lanthanide
#: configuration the WITHOUT-actinide fit on any verdict that is not *helps*, and F4's first clause -- "the WITHOUT arm
#: beats the deployed configuration" -- is then false of the configuration actually deployed
F4_DEPLOYED_CONFIGURATION_READING = (
    "REGISTERED (POST-HOC addendum 5 item 3): F4 is reported under BOTH readings, side by side, and no verdict is "
    "softened. (i) Against a WITH deployment, F4 HOLDS: the reversed V2 contrast WITHOUT vs WITH is +0.0264 with a BCa "
    "95 % interval excluding zero. (ii) Against the REGISTERED deployment, F4 does not hold: section 11 says actinide "
    "rows enter the deployed Ln configuration only on a *helps* verdict, the verdict is UNDECIDED, so the deployed "
    "lanthanide configuration IS the WITHOUT-actinide fit and F4's first clause ('the WITHOUT arm beats the deployed "
    "configuration') is false of it. Both are printed, together with the V5 point-estimate cost of the registered "
    "choice: on the primary design V5 the WITH arm is the better one (+0.0669 against WITHOUT and +0.0782 against the "
    "permuted control, both intervals excluding zero, both below delta5), so the registered WITHOUT deployment gives up "
    "+0.0669 log D of selection-half V5 macro MAE")
#: POST-HOC addendum 4 item 3 says the 13 stale records are "refitted under the CURRENT registered h3 digest before any
#: H3 delta is scored".  Taken literally that is unsatisfiable in a stage whose code digest also covers scoring and
#: reporting: the refits were written under the entry that was current at 17:51-19:50 on 2026-09-22, and three
#: reporting-only edits to h3.py (the per_unit_delta_table TypeError fix and two D03 fixes) superseded that entry before
#: scoring ran, so NO h3 record verifies against the current entry.  The operative reading is addendum 2 change 5's
#: (task X findings numbers VH-04 / protocol VH-06)
OPERATIVE_DIGEST_READING = (
    "OPERATIVE READING (addendum 2 change 5): a record is valid under the registry entry that was CURRENT WHEN IT WAS "
    "WRITTEN, verified through registry.verify_record at the record's own timestamp -- not under whatever entry is "
    "current when a delta is scored. POST-HOC addendum 4 item 3's 'refitted under the current registered digest before "
    "any H3 delta is scored' is satisfied as 'refitted under the digest that was current at the moment of the refit, and "
    "never under the digest that was already superseded when the refit started'. Reading it as 'current at scoring time' "
    "would make every reporting-only edit to h3.py re-invalidate all 445 records, which addendum 2 change 5 explicitly "
    "forbids ('a later change can neither validate nor invalidate a record written earlier'). A POST-HOC addendum "
    "wording this is REQUESTED; splitting the fitting closure from the scoring / reporting closure in the h3 code digest "
    "would remove the question")
#: the intervals status recorded when a refit's inner folds differ from the WITH record's (no calibration is invented)
NOT_CALIBRATED_DIFFER = "not_calibrated_inner_folds_differ_from_the_with_record"
#: the R19 scope a discovery-stage H3 verdict is read on (R19 item 4 is NOT_EVALUATED in discovery, addendum 1 item 3)
VERDICT_SCOPE = "freezing_screen"
STAGE = "11_h3"
REFIT_NOT_RUN = "not run: section 11 names no refit sensitivity for the ablation; the scoring-filter sensitivities apply"
#: the registered V5 sensitivities addendum 1 item 4 DOES run for other families but not for H3, each with the reason
#: specific to H3.  ``discovery.LEARNED_REFITS_NOT_RUN`` covers only the refits the addendum drops for a learned arm in
#: discovery, so without these two names R19 item 6 of an H3 V5 contrast reported a PASS over the reduced set while 2 of
#: the design's 11 registered sensitivities appeared in neither the decided set nor the not-run list
#: (task X finding protocol VH-03)
H3_REFITS_NOT_RUN: dict[str, str] = {
    "strict_setting": "not run for H3: POST-HOC addendum 1 item 4 runs the V5 strict refit on seed 104729 for H1 (M2), "
                      "H1b (B6), H4 (M0, FLAT_CAT, M2, B6, B6r0) and any freezing candidate; an H3 contrast pairs WITH "
                      "against a section 11 TRANSFORM refit (WITHOUT / ACT_PERMUTED), and no strict-setting refit of a "
                      "transform arm is registered or fitted anywhere, so the sensitivity is UNTESTABLE for every H3 "
                      "contrast",
    "HNO3_only_cells": "not run for H3: POST-HOC addendum 1 item 4 runs the V5 HNO3-only refit on seed 104729 for H1 "
                       "(M2), H1b (B6), H4 (M0, FLAT_CAT, M2, B6, B6r0) and any freezing candidate; an H3 contrast pairs "
                       "WITH against a section 11 TRANSFORM refit (WITHOUT / ACT_PERMUTED), and no HNO3-only refit of a "
                       "transform arm is registered or fitted anywhere, so the sensitivity is UNTESTABLE for every H3 "
                       "contrast"}
NOT_COMPUTED = "not computed"
#: POST-HOC addendum 4 item 4: what an UNDECIDED H3 verdict with no registered section 8 power check is reported as --
#: never a null, never "no effect".  Duplicated from ``power.NO_POWER_CHECK_LABEL`` because ``power`` imports this
#: module; ``tests/test_h3.py`` asserts the two strings stay identical.
NO_POWER_CHECK_LABEL = "UNDECIDED (no registered power check)"
STEPS = D.STEPS
#: section 11's last bullet (addendum 2 item 4): the WITH arm re-run with a shared-only metal embedding; its records live
#: under ``evaluation/h3/records/<arm>/SHARED_ONLY`` and its rows are labelled with this transform token
SHARED_ONLY = "SHARED_ONLY"
#: the BH family of the re-run's contrasts: the negative-transfer investigation is descriptive (section 11), so its rows
#: are exploratory beside the registered H3 family
SHARED_ONLY_FAMILY = "H3 negative-transfer investigation (shared-only embedding)"
SHARED_ONLY_NOT_RUN = "not run (condition not met)"
#: the arms with a section 15 metal embedding the re-run applies to (M1 / M2 and the ladder steps built on them)
EMBEDDING_ARMS: tuple[str, ...] = ("M1", "M2") + LADDER_ARMS
#: section 11 states ONE condition for the whole negative-transfer investigation ("run if WITHOUT is better, point
#: estimate or passed") and names no design and no contrast.  The tables used to be written on the CLI flag alone, while
#: two artifacts stated two different triggers (D03 read it on V2, ``shared_only.json`` on V5), so the condition is now
#: evaluated in ONE place on ONE reading -- the same reading ``shared_only_condition`` already used -- and recorded beside
#: the tables (task X finding protocol VH-05)
NEGATIVE_TRANSFER_CONDITION = (
    "section 11, negative-transfer investigation: 'run if WITHOUT is better, point estimate or passed' -- read on the "
    "DEPLOYED arm's V5 Ln-cell contrast WITHOUT vs WITH, the primary design of section 11's Ln test set and the same "
    "contrast the shared-only re-run's condition reads (h3_verdicts.json): hurts == true (WITHOUT passes R19 on the "
    "freezing_screen scope against WITH; the 'passed' half) OR v5_without_vs_with_point > 0 (Delta macro MAE = MAE(WITH) "
    "- MAE(WITHOUT) under the primary cluster unit; the 'point estimate' half). Section 11 names neither the design nor "
    "the contrast, so this is a READING and a POST-HOC addendum fixing it is REQUESTED; the per-cell tables are V5 "
    "tables, which is why the V5 contrast is the one read. When the condition is NOT met the tables are still written "
    "(they cost nothing, the per-cell deltas already exist) but they are labelled EXPLORATORY, not 'the section 11 "
    "negative-transfer investigation', and no reading is taken from them")
NEGATIVE_TRANSFER_EXPLORATORY = "exploratory (section 11 condition not met on the registered reading)"
SHARED_ONLY_CONDITION = ("section 11, negative-transfer investigation: 'run if WITHOUT is better, point estimate or passed' "
                         "-- read on the deployed arm's V5 Ln-cell contrast WITHOUT vs WITH (h3_verdicts.json): "
                         "hurts == true (WITHOUT passes R19 on the freezing_screen scope against WITH; the 'passed' half) "
                         "OR v5_without_vs_with_point > 0 (the reversed contrast's Delta macro MAE = MAE(WITH) - "
                         "MAE(WITHOUT) under the primary cluster unit; the 'point estimate' half)")

READINGS: dict[str, str] = {
    "deployed_rule": "POST-HOC addendum 3 item 2: the configuration deployed for lanthanide prediction ('the retained "
                     "ladder configuration', section 11) is the ladder's RETAINED configuration -- the highest kept "
                     "ladder step: evaluation/ladder/decisions/ladder.json -> steps.<step>.kept == true for "
                     "M7 > M6 > M5 > M4 > M3 (only from a registered ladder run, i.e. ladder.json -> stop_rule == "
                     "false: under the stop rule M3-M6 are exploratory_not_run and M7 runs for H6 only, so no ladder "
                     "step deploys), then the scorer's decisions.json -> ladder.<step>.kept == true (M2 before M1), "
                     "then the ladder's BASE step M0 (= B5, the registered descriptor arm of section 5 and the M0 row "
                     "of the section 6 ladder) when decisions.json -> ladder.M0.kept == true and neither M1 nor M2 is "
                     "kept. Addendum 2's remaining order applies ONLY when the ladder has no retained step at all (no "
                     "ladder block / M0 not kept): M2 when stop_rule.M2_vs_B3i.verdict == PASS; else B6 when "
                     "stop_rule.B6_vs_B3i.verdict == PASS; else the registered V5 lookup comparator B3i (section 10 F6: "
                     "the best-passing baseline; the process chain's D source stays gen18 B1). The rule is undecidable "
                     "(the runners refuse) while ladder.json is absent or any step M3-M7 is not in a done / skipped "
                     "status, or while a discovery ladder decision is pending (kept == null) (task X finding V-01)",
    "ladder_arm_refit": "when the deployed configuration is a ladder step M3-M7, WITH is its evaluation/ladder record of "
                        "the same design, seed and fold (verified against the digests the current code and ladder "
                        "decisions produce) and WITHOUT / ACT_PERMUTED / ACT_METAL_SHUFFLED are refitted at that record's "
                        "selected LadderConfig, epoch count and model seed (M7: the 5 member seeds and the "
                        "publication-group bootstrap of the transformed training rows), with the ladder's cross-fitted "
                        "calibration at the WITH record's cross-fit selections (M7: normalised split conformal); the H3 "
                        "record digest carries the WITH record's digest, which carries the ladder code digest, so a "
                        "ladder code or decision change stales the H3 record transitively (INFERRED; needs a POST-HOC "
                        "addendum)",
    "comparator_folds": "a closed-form arm (B0-B4 variants, B7) in an H3 or injected re-run is fitted on the exact "
                        "leave-one-cell-out folds of section 3.1 (V5__primary_exact, V1__copy_exact, V2__element_exact) "
                        "as the discovery comparator is, not on the heavy arm's batched folds, and is paired with the "
                        "heavy arm on the scored rows / units; on V5-PAIR every arm uses the seed-104729 batched folds "
                        "(addendum 1 item 5) (task X finding V-03)",
    "calibration_guard": "a refit whose fold has different inner folds from the WITH record's inner_folds_used is "
                         "recorded with intervals_status " + NOT_CALIBRATED_DIFFER + " instead of a calibration at "
                         "foreign selections (as FrozenB6 already did; task X finding VL2-05)",
    "tuning": "the WITHOUT / ACT_PERMUTED / ACT_METAL_SHUFFLED arms are refitted at the hyperparameters the WITH run "
              "SELECTED on the same fold (its record's selected configuration, iteration / epoch count and model seed), "
              "not re-tuned; their cross-fitted conformal calibration uses the WITH record's cross-fit selections. "
              "Section 11 says 'same architecture, same folds, same seeds' and is silent on re-tuning; re-tuning would "
              "multiply the cost by the inner grid (INFERRED; needs a POST-HOC addendum)",
    "ln_test_set": "Ln(III) = a known metal state whose element is La-Lu and whose oxidation state is III; the Ln test "
                   "set of a design is the Ln(III) scored rows of its selection-half folds (V5: Ln(III) cells; V2: every "
                   "Ln(III) state in the selection half, the focus-7 subset printed beside; V1: Ln(III) rows of the "
                   "selection folds); V6_TARGET_ROWS and the acidic co-extractant rows are excluded by the discovery "
                   "scoring guards",
    "r19_scope": "a discovery-stage H3 contrast 'passes R19' on the freezing-screen scope (items 1-3, 5 and the "
                 "scoring-filter sensitivities on seed 104729): R19 item 4 is NOT_EVALUATED in discovery (addendum 1 "
                 "item 3) and section 11 names no refit sensitivity, so the full verdict is at best UNDECIDED; the full "
                 "verdict is printed beside and confirmation decides item 4",
    "margins": "R19 item 1 margin of an H3 contrast: delta5 on the V5 Ln cells, 0.05 on V1 / V2. This is REGISTERED, not "
               "a scorer reading: POST-HOC addendum 2, '3. H3' > 'margins' reads 'delta5 on V5 Ln cells, 0.05 on V1 / "
               "V2', and the same addendum's discovery reading says '`margins`: delta5 on V5; 0.05 for S1(b) and for V1 "
               "/ V2 (section 9 names none; the floor of the delta5 formula)'. R19 item 1 compares the POINT estimate "
               "with that margin ('the point estimate Delta >= the design's margin delta'), not an interval bound, so a "
               "V1 / V2 contrast needs no further addendum to be read against 0.05 (task X finding numbers VH-01)",
    "f4_designs": "F4 'the WITHOUT arm beats it on the Ln test set (95 % interval excluding 0)' is evaluated per design "
                  "(V5 Ln cells, V1 Ln rows, V2 Ln states) on the reversed contrast (WITHOUT candidate) under the "
                  "primary cluster unit, and F4 HOLDS when ANY registered design shows it under EITHER registered 95 % "
                  "interval (percentile or BCa) -- the conservative reading on both axes, since F4 is a FAILURE "
                  "condition and the conservative choice is the one that triggers more readily. The percentile-only and "
                  "BCa-only readings are reported beside it (failure_percentile, failure_bca) and never replace the "
                  "headline (TASK F item 3)",
    "actinide_dependent": "a cell's eligibility 'depends on actinide partners' when the eligible_cells rule of the "
                          "fold builder (folds.cell_holdout.EligibilitySpace.check) passes on all rows and fails once "
                          "every actinide row is removed",
    "equivalent": "TOST (90 % interval, epsilon 0.05, primary cluster) of WITH - WITHOUT on the V5 Ln cells",
    "helps_non_inferior": "non-inferiority on V1 / V2 is the TOST 'non_inferior' flag of WITH - WITHOUT and WITH - "
                          "ACT_PERMUTED on the V1 and V2 Ln sets",
    "shared_only_embedding": "POST-HOC addendum 2 item 4 ('it is implemented in a NEW module (an H3-side subclass of "
                             "neural.FactorisedNet with the e_series and e_ox offsets held at zero; no existing file "
                             "edited) before scripts/g19_run_h3.py runs, and executes only under section 11's condition "
                             "that WITHOUT beats WITH; otherwise D03 reports it as not run'): the WITH arm re-run with a "
                             "shared-only metal embedding (no e_series, e_ox) is models.shared_only (FrozenSharedOnly), "
                             "refitted at the WITH record's SELECTED hyperparameters, stopping count and model seed with "
                             "the cross-fitted calibration at the WITH record's selections, scored against the full "
                             "section 15 embedding (WITH) on the V5 Ln cells and the V1 / V2 Ln sets as an exploratory "
                             "negative-transfer contrast; it runs only when " + SHARED_ONLY_CONDITION + "; otherwise "
                             "D03 says '" + SHARED_ONLY_NOT_RUN + "'; a deployed arm without a metal embedding (B6, B5, "
                             "a closed-form comparator) makes it not applicable. Reading of the item's 'no existing file "
                             "edited': no file of the DISCOVERY import closure was edited (neural.FactorisedNet has no "
                             "switch; the zeroed offsets live in the new models/shared_only.py); the H3-side files "
                             "evaluation/h3.py and scripts/g19_run_h3.py, outside that closure and part of no discovery "
                             "record's digest, were extended to plan, fit, score and report the re-run",
    "crps": "CRPS is NOT_RUN: no arm has a predictive SD (discovery.UNCERTAINTY_NOT_RUN)",
    "logsf_delta": "section 11 scopes Delta logSF MAE to 'V5-PAIR Ln pairs; V6 at confirmation'. It is NOT_DEFINED for "
                   "either H3 arm in discovery: POST-HOC addendum 1 item 5 runs the seed-104729 batched V5-PAIR folds "
                   "for M2 and for the B3x / B3i counterweight only, plus any freezing candidate WITH a pair endpoint, "
                   "and neither H3 arm is one of those -- B5 (= M0, the deployed arm) and B6 have no V5-PAIR record, so "
                   "there is no Ln-Ln pair prediction of either arm to difference, under any transform. logsf_delta is "
                   "therefore never reached: plan_jobs plans the pair design only under --with-pairs, and with it the "
                   "pair jobs would find no WITH record and be reported NOT_RUN rather than fitted. Producing the "
                   "quantity would mean fitting V5-PAIR for B5 and B6, which addendum 1 item 5 does not run; it is "
                   "reported NOT_DEFINED here and is a registered V6 quantity at confirmation",
    "act_metal_shuffled": ACT_METAL_SHUFFLED_NOT_RUN,
    "h3_budget": "POST-HOC addendum 4 item 2: section 11 registers no compute budget, so H3 gets a cap of 20 h of wall "
                 "clock on 2 workers, recorded in evaluation/h3/decisions/wall_clock.json and CHECKED BEFORE EACH FOLD. "
                 "The budget number is WALL CLOCK, not worker-seconds: the ledger records every invocation's "
                 "[started_utc, ended_utc] interval and h3_wall_clock_total sums the UNION of those intervals, so two "
                 "workers running the same hour consume one hour of the cap; the sum over workers is recorded beside it "
                 "as worker_seconds and is never the number compared with the cap. More than 2 concurrent workers is "
                 "refused (H3_WORKERS). --max-hours stays an operator pause, never a NOT_RUN reason",
    "design_priority": "POST-HOC addendum 4 item 2: the H3 job plan is ordered V5 -> V2 -> V1 (DESIGN_PRIORITY) and the "
                       "cap is checked before each fold; a design the cap does not reach is written NOT_RUN with its "
                       "reason. " + CONTRAST_UNIT_RULE + " Mechanically: contrast_status classifies every leg, "
                       "drop_not_run_contrasts removes every row of an incomplete leg before any output is written, "
                       "assert_no_partial_design raises if an incomplete record set ever reaches a contrast, "
                       "designs_scored and designs_not_run are asserted DISJOINT, and h3_verdict names every incomplete "
                       "leg in inputs_missing so a helps verdict can never be read off the legs that did finish. The "
                       "cap's NOT_RUN and a guard failure's " + INCOMPLETE_GUARD_FAILURE + " are distinct statuses and "
                       "the entry says which applies (task X finding protocol VH-01 / numbers VH-02; TASK F items 1-2)",
    "contrast_unit": CONTRAST_UNIT_RULE,
    "guard_failure": GUARD_FAILURE_DIAGNOSIS,
    "guard_value_source": GUARD_VALUE_SOURCE_RULE,
    "f4_deployed_configuration": F4_DEPLOYED_CONFIGURATION_READING,
    "record_digest_basis": "addendum 2 item 5 and POST-HOC addendum 4 item 3: an EXISTING H3 fold record is verified "
                           "with the code digest AND the below-footer digest it was WRITTEN under -- its own fields, "
                           "accepted only when registry.verify_record matches them to an entry of stage 'h3' (current or "
                           "superseded, ordered by the record's own timestamp). A record written before a supersession "
                           "therefore keeps its entry and verifies as registered, while a record a runner holding stale "
                           "code wrote AFTER the supersession does not verify (stale_after_supersession) and is refitted "
                           "under the current registered digest. New records are always written under the LIVE code and "
                           "the currently registered below-footer digest",
}


# --------------------------------------------------------------------------------------------- #
# the deployed configuration and the arms
# --------------------------------------------------------------------------------------------- #

def read_ladder_state(out_root: Path) -> dict[str, Any] | None:
    """The ladder runner's resumable state ``evaluation/ladder/decisions/ladder.json`` (``None`` when absent)."""
    return D.read_record(Path(out_root) / LADDER_STATE_FILE)


def ladder_complete(ladder: Mapping[str, Any] | None) -> dict[str, Any]:
    """Whether the ladder has run to an end: ``ladder.json`` exists and every step M3-M7 is in a done / skipped status
    (:data:`LADDER_DONE_STATUSES`).  The deployed configuration (section 11 'the retained ladder configuration') is
    undecidable before that (task X finding V-01)."""
    definition = ("ladder.json exists and every step M3-M7 has a status in " + ", ".join(sorted(LADDER_DONE_STATUSES)))
    if ladder is None:
        return {"complete": False, "present": False, "steps": {}, "not_done": list(LADDER_ARMS), "definition": definition}
    steps = ladder.get("steps") or {}
    status = {s: (steps.get(s) or {}).get("status") for s in LADDER_ARMS}
    not_done = [s for s, st in status.items() if st not in LADDER_DONE_STATUSES]
    return {"complete": not not_done, "present": True, "steps": status, "not_done": not_done,
            "stop_rule": ladder.get("stop_rule"), "definition": definition}


def deployed_configuration(decisions: Mapping[str, Any], ladder: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """:func:`deployed_rule` plus ``arm_alias``: the arm name the RECORDS, PREDICTIONS and SUMMARY TABLES use for the
    deployed configuration, which is not always the name it is reported under.  POST-HOC addendum 3 item 2 names the
    deployed configuration **M0**, and every "deployed predictor" sentence says M0 -- but M0's discovery records,
    prediction frames and ``tables/discovery_summary.csv`` rows are all written under **B5**
    (``discovery.ARM_ALIASES``).  A reader that looks a table or record up by the deployed arm uses ``arm_alias``; a
    reader that prints the deployed predictor uses ``arm``.  ``None`` for both while the rule is pending."""
    out = deployed_rule(decisions, ladder)
    arm = out.get("arm")
    out["arm_alias"] = None if arm is None else D.ARM_ALIASES.get(arm, arm)
    out["arm_alias_note"] = ("the arm name records, prediction frames and tables/discovery_summary.csv use for the "
                             "deployed configuration; 'arm' is the name it is REPORTED under (addendum 3 item 2 names "
                             "M0, whose records are written under B5)")
    return out


def deployed_rule(decisions: Mapping[str, Any], ladder: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The configuration deployed for lanthanide prediction (:data:`READINGS` ``deployed_rule``) from the ladder runner's
    ``ladder.json`` (``steps.<step>.kept`` for M3-M7, registered run only) and the scorer's ``decisions.json``
    (``ladder.<step>.kept`` for M1 / M2, ``stop_rule.M2_vs_B3i.verdict``, ``stop_rule.B6_vs_B3i.verdict``).

    ``ladder`` is the body of ``evaluation/ladder/decisions/ladder.json``; ``None`` (absent) or an incomplete ladder
    makes the rule ``pending`` -- the H3 arm is 'the retained ladder configuration' and cannot be named before the
    ladder has run (task X finding V-01).

    POST-HOC addendum 3 item 2: the ladder's retained configuration INCLUDES its base step M0 (= B5) when neither M1 nor
    M2 is kept, and addendum 2's remaining order (M2 / B6 by the stop-rule scope, then B3i) applies only when the ladder
    has no retained step at all."""
    lad = decisions.get("ladder") or {}
    stop = decisions.get("stop_rule") or {}

    def kept(step: str) -> bool | None:
        rec = lad.get(step) or {}
        return rec.get("kept")

    def passes(key: str) -> bool | None:
        rec = stop.get(key)
        return None if rec is None else bool(rec.get("verdict") == "PASS")

    # addendum 2 (F6): the fallback deploys M2 (or B6) "only when its contrast passes the stop-rule scope (items 1, 2,
    # 3, 5) AND item 6 over the reduced sensitivity set - no evaluated item may FAIL".  decisions.json -> stop_rule
    # carries items 1, 2, 3 and 5 ONLY (its own evaluated_on.items), so item 6 is read from the scorer's
    # freezing_candidates block, whose screen (discovery.freezing_candidates) is exactly that scope plus item 6 with
    # discovery.freezing_eligibility (no evaluated item FAIL).  A contrast the scorer did not list did not pass the
    # screen; without the block at all the item is unknown and the basis says the check did not run (task X V-P09).
    cands = decisions.get("freezing_candidates")

    def no_evaluated_fail(contrast_key: str) -> bool | None:
        if cands is None:
            return None
        for c in cands:
            if str(c.get("contrast")) == contrast_key:
                return bool(c.get("eligible_for_freezing"))
        return False

    def f6_ok(stop_key: str) -> tuple[bool, bool | None]:
        """``(deploys, item6)``: the stop-rule scope PASSES and no evaluated item FAILs (unknown item 6 does not block,
        and the basis records that it was not checked)."""
        elig = no_evaluated_fail(F6_CONTRAST_OF[stop_key])
        return bool(stops[stop_key] is True and elig is not False), elig

    steps = {s: kept(s) for s in ("M2", "M1", LADDER_BASE)}
    stops = {"M2_vs_B3i": passes("M2_vs_B3i"), "B6_vs_B3i": passes("B6_vs_B3i")}
    f6 = {k: no_evaluated_fail(v) for k, v in F6_CONTRAST_OF.items()}
    lc = ladder_complete(ladder)
    lsteps = {s: (((ladder or {}).get("steps") or {}).get(s) or {}).get("kept") for s in LADDER_ARMS}
    keys = {"ladder_runner": {s: f"{LADDER_STATE_FILE} -> steps.{s}.kept" for s in LADDER_ARMS},
            "ladder": {s: f"ladder.{s}.kept" for s in steps}, "stop_rule": {k: f"stop_rule.{k}.verdict" for k in stops},
            "f6_no_evaluated_fail": {k: f"freezing_candidates[contrast={v}].eligible_for_freezing"
                                     for k, v in F6_CONTRAST_OF.items()}}
    base = {"rule": READINGS["deployed_rule"], "keys": keys, "ladder_kept": {**lsteps, **steps}, "stop_rule_pass": stops,
            "f6_no_evaluated_fail": f6, "f6_rule": F6_FALLBACK_RULE, "ladder_complete": lc}
    if not lc["complete"]:
        if not lc["present"]:
            why = "absent: scripts/g19_run_ladder.py has not run"
        else:
            why = "steps not in a done / skipped status: " + ", ".join(f"{s}={lc['steps'][s]!r}" for s in lc["not_done"])
        return {**base, "arm": None, "status": "pending",
                "basis": f"the ladder (M3-M7) is not complete -- {LADDER_STATE_FILE} {why}; section 11 names 'the retained "
                         "ladder configuration', which is undecidable before the ladder has run"}
    if ladder is not None and ladder.get("stop_rule") is True:
        base["ladder_note"] = ("ladder.json -> stop_rule == true: M3-M6 are exploratory_not_run and M7 ran for H6 only "
                               "(section 7 item 4), so no ladder step deploys")
    else:
        for s in reversed(LADDER_ARMS):                     # M7 > M6 > M5 > M4 > M3
            if lsteps[s] is True:
                return {**base, "arm": s, "status": "decided",
                        "basis": f"ladder step {s} kept (section 6; {LADDER_STATE_FILE} -> steps.{s}.kept)"}
    if steps["M2"] is True:
        return {**base, "arm": "M2", "status": "decided", "basis": "ladder step M2 kept (section 6)"}
    if steps["M1"] is True:
        return {**base, "arm": "M1", "status": "decided", "basis": "ladder step M1 kept, M2 not kept (section 6)"}
    if steps["M2"] is None or steps["M1"] is None:
        return {**base, "arm": None, "status": "pending",
                "basis": "a ladder decision is pending (kept == null); the deployed configuration is undecidable"}
    # POST-HOC addendum 3 item 2: the ladder has kept M0 and dropped M1 and M2, so ITS retained configuration is M0
    # (= B5) -- deploying M2 by addendum 2's stop-rule fallback would contradict the registered ladder decision.  The
    # rest of addendum 2's order below applies only when the ladder has no retained step at all.
    if steps[LADDER_BASE] is True:
        note = ("; the stop rule fired, and addendum 3 item 2 was resolved on a state where it did not -- the ladder "
                "still retains its base step, so the base step is what this rule names (addendum 2's F6 fallback is "
                "reached only without a retained step)" if (ladder or {}).get("stop_rule") is True else "")
        return {**base, "arm": LADDER_BASE, "status": "decided", "retained_ladder_step": LADDER_BASE,
                "basis": f"the ladder retains its base step {LADDER_BASE} (= {D.ARM_ALIASES[LADDER_BASE]}, the "
                         "registered descriptor arm of section 5), M1 and M2 are not kept, and no step M3-M7 is kept: "
                         f"the retained ladder configuration is {LADDER_BASE} (addendum 3 item 2; ladder."
                         f"{LADDER_BASE}.kept){note}"}
    m2_ok, m2_item6 = f6_ok("M2_vs_B3i")
    b6_ok, b6_item6 = f6_ok("B6_vs_B3i")
    if m2_ok:
        return {**base, "arm": "M2", "status": "decided",
                "basis": "the ladder has no retained step; M2 passes the stop-rule scope against B3i (section 7 item 4) "
                         + ("and no evaluated R19 item FAILs" if m2_item6 is True else
                            "-- item 6 not checked: decisions.json carries no freezing_candidates block")
                         + " (addendum 2's F6 order, addendum 3 item 2)"}
    if b6_ok:
        return {**base, "arm": "B6", "status": "decided",
                "basis": "the ladder has no retained step and M2 does not deploy by the F6 fallback; B6 passes the "
                         "stop-rule scope (H1b) "
                         + ("and no evaluated R19 item FAILs" if b6_item6 is True else
                            "-- item 6 not checked: decisions.json carries no freezing_candidates block")}
    if (stops["M2_vs_B3i"] is None and f6["M2_vs_B3i"] is not False) \
            or (stops["B6_vs_B3i"] is None and f6["B6_vs_B3i"] is not False):
        return {**base, "arm": None, "status": "pending", "basis": "a stop-rule contrast is missing"}
    return {**base, "arm": FALLBACK_DEPLOYED, "status": "decided", "fallback": True,
            "basis": "the ladder has no retained step and neither M2 nor B6 deploys by the F6 fallback (the stop-rule "
                     "scope against B3i, with no evaluated R19 item FAILing): the best-passing baseline is deployed "
                     "(section 10 F6); the process chain's default D source stays gen18 B1 (section 10)"}


def model_arms(deployed: str) -> tuple[str, ...]:
    """``(deployed, B6, B5)`` without repeats (M0 is B5, ``discovery.ARM_ALIASES``)."""
    arm = D.ARM_ALIASES.get(deployed, deployed)
    return tuple(dict.fromkeys((arm,) + REFERENCE_ARMS))


def is_deterministic(arm: str) -> bool:
    return D.ARM_ALIASES.get(arm, arm) in DETERMINISTIC_ARMS


def is_ladder_arm(arm: str) -> bool:
    return D.ARM_ALIASES.get(arm, arm) in LADDER_ARMS


def batching_arms(arm: str) -> list[str]:
    """The arm whose section 7 item 6 batching label a V5 contrast carries: a ladder step runs on discovery's M2 jobs
    (the heavy batched folds), so its label is M2's (``discovery.HEAVY_ARMS`` does not list M3-M7)."""
    a = D.ARM_ALIASES.get(arm, arm)
    return ["M2"] if a in LADDER_ARMS else [a]


def transforms_for(arm: str) -> tuple[str, ...]:
    """The transforms of ``arm`` that are FITTED AND SCORED: a learned arm's WITH is its discovery record; a closed-form
    arm has no discovery record, so WITH is fitted.  ACT_METAL_SHUFFLED is not among them (POST-HOC addendum 4 item 1,
    :data:`EXPLORATORY_TRANSFORMS`)."""
    return (("WITH",) if is_deterministic(arm) else ()) + TRANSFORMS


def is_scored_transform(transform: str) -> bool:
    """Whether ``transform`` may enter a contrast, delta, table or verdict (POST-HOC addendum 4 item 1)."""
    return str(transform) not in EXPLORATORY_TRANSFORMS


def plan_priority_key(entry: Mapping[str, Any]) -> tuple[int, str, str]:
    """Sort key of one job-plan entry under the POST-HOC addendum 4 item 2 priority order V5 -> V2 -> V1: the design
    first (a design outside :data:`DESIGN_PRIORITY`, e.g. V5PAIR, sorts after every priority design), then the model arm
    and the transform, so the cap is spent on the highest-priority design of EVERY arm before the next design starts."""
    design = str(entry.get("design"))
    idx = DESIGN_PRIORITY.index(design) if design in DESIGN_PRIORITY else len(DESIGN_PRIORITY)
    return (idx, str(entry.get("model_arm")), str(entry.get("transform")))


def order_by_priority(plan: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``plan`` in the registered priority order (:func:`plan_priority_key`); stable within a design."""
    return [dict(e) for e in sorted(plan, key=plan_priority_key)]


def with_design_dir(arm: str, design: str, state: D.PlanState) -> str:
    """The record directory of the WITH run (the scorer's ``Store.design_dir`` at the primary setting): a heavy arm and a
    ladder step (M2's jobs renamed) on the plan state's heavy scheme; B6 / B6r0 and every closed-form comparator on the
    exact leave-one-cell-out folds of section 3.1 (:data:`READINGS` ``comparator_folds``; task X finding V-03); every
    arm on the seed-104729 batched V5-PAIR folds (addendum 1 item 5)."""
    arm = D.ARM_ALIASES.get(arm, arm)
    exact = arm in D.B6_ARMS or arm in DETERMINISTIC_ARMS
    if design == "V5":
        return "V5__primary_exact" if exact else f"V5__primary_{state.heavy_v5_scheme or 'batched'}"
    if design == "V1":
        return "V1__copy_exact" if exact else f"V1__copy_{state.heavy_v1_scheme or 'grouped10'}"
    if design == "V2":
        return "V2__element_exact"
    if design == PAIR_DESIGN:
        return "V5PAIR__primary_batched"
    raise ValueError(f"no H3 design {design!r}")


def with_job(arm: str, design: str, state: D.PlanState, *, seed: int = D.PRIMARY_SEED) -> D.JobSpec:
    """The job whose records are the WITH arm (its fold seed for a multi-seed fold file): a discovery job, or for a
    ladder step the ladder runner's job (discovery's M2 job with the arm renamed; records under ``evaluation/ladder``)."""
    arm = D.ARM_ALIASES.get(arm, arm)
    dd = with_design_dir(arm, design, state)
    des, rest = dd.split("__", 1)
    variant, scheme = rest.split("_", 1)
    multi = scheme.startswith("batched") or scheme.startswith("grouped")
    writes = D.B6_ARMS if arm in D.B6_ARMS else (arm,)
    return D.JobSpec(kind="fit", arm="B6" if arm in D.B6_ARMS else arm, design=des, variant=variant, scheme=scheme,
                     seed=int(seed), fold_seed=int(seed) if multi else None, writes=writes, stage=STAGE,
                     group="section 11 actinide ablation", purpose=f"H3 WITH arm of {arm} on {design}")


def h3_job(base: D.JobSpec, model_arm: str, transform: str) -> D.JobSpec:
    """The H3 job of one (model arm, transform): the WITH job's design, scheme and seed under the H3 arm name."""
    if transform not in ALL_ARMS:
        raise ValueError(f"H3 transform {transform!r} not in {ALL_ARMS}")
    name = h3_arm_name(model_arm, transform)
    return replace(base, arm=name, writes=(name,), stage=STAGE, group="section 11 actinide ablation",
                   purpose=f"H3 {transform} arm of {model_arm}: training transform discovery.h3_training_rows")


def h3_arm_name(model_arm: str, transform: str) -> str:
    return f"{D.ARM_ALIASES.get(model_arm, model_arm)}:{transform}"


def shared_only_applicable(model_arm: str) -> bool:
    """The re-run applies to an arm with a section 15 metal embedding (M1 / M2, a ladder step); B6, B5 and the closed-form
    comparators have no ``e_series`` / ``e_ox`` to remove."""
    return D.ARM_ALIASES.get(model_arm, model_arm) in EMBEDDING_ARMS


def shared_only_job(base: D.JobSpec, model_arm: str) -> D.JobSpec:
    """The job of the shared-only re-run of ``model_arm`` (addendum 2 item 4): the WITH job's design, scheme and seed
    under the arm name ``<arm>:SHARED_ONLY``; the training rows are the WITH rows unchanged (no transform)."""
    if not shared_only_applicable(model_arm):
        raise ValueError(f"the shared-only embedding re-run applies to {EMBEDDING_ARMS}, not {model_arm!r}")
    name = h3_arm_name(model_arm, SHARED_ONLY)
    return replace(base, arm=name, writes=(name,), stage=STAGE, group="section 11 negative-transfer investigation",
                   purpose=f"H3 shared-only-embedding re-run of {model_arm} (section 11 last bullet; addendum 2 item 4)")


def shared_only_condition(verdict: Mapping[str, Any] | None, *, deployed_arm: str) -> dict[str, Any]:
    """Addendum 2 item 4: the re-run "executes only under section 11's condition that WITHOUT beats WITH; otherwise D03
    reports it as not run".  Section 11 states the condition as "run if WITHOUT is better, point estimate or passed", read
    on the deployed arm's :func:`h3_verdict`: ``hurts`` (WITHOUT passes R19 on :data:`VERDICT_SCOPE` against WITH on the
    V5 Ln cells) OR ``v5_without_vs_with_point > 0`` (the reversed V5 contrast's Delta macro MAE, MAE(WITH) -
    MAE(WITHOUT), primary cluster unit).  A missing verdict leaves the condition undecided (not run, reason named)."""
    arm = D.ARM_ALIASES.get(deployed_arm, deployed_arm)
    applicable = shared_only_applicable(arm)
    out: dict[str, Any] = {"condition": SHARED_ONLY_CONDITION, "deployed_arm": arm, "applicable": applicable,
                           "without_passed_r19_v5": None, "v5_without_vs_with_point": None,
                           "point_estimate_favours_without": None, "met": False, "status": SHARED_ONLY_NOT_RUN,
                           "reading": READINGS["shared_only_embedding"]}
    if not applicable:
        # the condition's two inputs are recorded even when the arm makes the re-run inapplicable: they are known, so
        # D03 must print them rather than "not computed" (the re-run stays not run either way)
        pt0 = None if verdict is None else verdict.get("v5_without_vs_with_point")
        out.update(status=f"not run (not applicable: {arm} has no section 15 metal embedding)",
                   reason=f"{arm} has no e_series / e_ox offsets to remove",
                   without_passed_r19_v5=None if verdict is None or verdict.get("hurts") is None
                   else bool(verdict.get("hurts")),
                   v5_without_vs_with_point=None if pt0 is None else float(pt0),
                   point_estimate_favours_without=None if pt0 is None else bool(float(pt0) > 0))
        return out
    if verdict is None:
        out.update(status="not run (condition undecided: the deployed arm's H3 verdict is absent)",
                   reason="h3_verdict of the deployed arm missing")
        return out
    passed = verdict.get("hurts")
    point = verdict.get("v5_without_vs_with_point")
    pt = None
    if point is not None:
        try:
            pt = float(point)
        except (TypeError, ValueError):
            pt = None
    favours = bool(pt is not None and np.isfinite(pt) and pt > 0)
    met = bool(passed) or favours
    out.update(without_passed_r19_v5=None if passed is None else bool(passed), v5_without_vs_with_point=pt,
               point_estimate_favours_without=favours if pt is not None else None, met=met,
               status="run" if met else SHARED_ONLY_NOT_RUN,
               reason=("WITHOUT passes R19 against WITH on the V5 Ln cells" if passed else
                       ("the V5 point estimate favours WITHOUT" if favours else
                        "WITHOUT neither passes R19 against WITH nor has the better V5 point estimate")))
    return out


def negative_transfer_condition(verdict: Mapping[str, Any] | None, *, deployed_arm: str,
                                design: str = "V5") -> dict[str, Any]:
    """Section 11's negative-transfer trigger, evaluated and recorded (:data:`NEGATIVE_TRANSFER_CONDITION`).

    Returns ``met``, both halves of the condition and the status the tables carry: ``"section 11 negative-transfer
    investigation"`` when met, :data:`NEGATIVE_TRANSFER_EXPLORATORY` when not.  The condition is read on the deployed
    arm's V5 Ln-cell reversed contrast only -- one design, one contrast -- so the trigger can never be reported as met on
    one design in one file and unmet on another in the next (task X finding protocol VH-05)."""
    arm = D.ARM_ALIASES.get(deployed_arm, deployed_arm)
    out: dict[str, Any] = {"condition": NEGATIVE_TRANSFER_CONDITION, "deployed_arm": arm, "design_read": str(design),
                           "contrast_read": f"{arm}:WITHOUT vs {arm}:WITH@{design}",
                           "without_passed_r19_v5": None, "v5_without_vs_with_point": None,
                           "point_estimate_favours_without": None, "met": False,
                           "status": NEGATIVE_TRANSFER_EXPLORATORY,
                           "tables_label": NEGATIVE_TRANSFER_EXPLORATORY,
                           "addendum_requested": True}
    if verdict is None:
        out["reason"] = (f"the deployed arm's H3 verdict is absent, so the condition is undecided; the tables are "
                        f"{NEGATIVE_TRANSFER_EXPLORATORY}")
        return out
    passed = verdict.get("hurts")
    pt = None
    try:
        pt = None if verdict.get("v5_without_vs_with_point") is None else float(verdict["v5_without_vs_with_point"])
    except (TypeError, ValueError):
        pt = None
    favours = bool(pt is not None and np.isfinite(pt) and pt > 0)
    met = bool(passed) or favours
    out.update(without_passed_r19_v5=None if passed is None else bool(passed), v5_without_vs_with_point=pt,
               point_estimate_favours_without=None if pt is None else favours, met=met,
               status="section 11 negative-transfer investigation (condition met)" if met
               else NEGATIVE_TRANSFER_EXPLORATORY,
               tables_label="section 11 negative-transfer investigation" if met else NEGATIVE_TRANSFER_EXPLORATORY,
               reason=("WITHOUT passes R19 against WITH on the V5 Ln cells" if passed else
                       ("the V5 point estimate favours WITHOUT" if favours else
                        "WITHOUT neither passes R19 against WITH nor has the better V5 point estimate on the deployed "
                        "arm's V5 Ln cells, so section 11's condition is NOT met")))
    return out


def stratum_excluded_headline(per_cell: pd.DataFrame, dependent: pd.DataFrame | None, *, point: float | None = None,
                             delta_col: str = "delta_mae") -> dict[str, Any]:
    """Section 11: "Cells whose eligibility depends on actinide partners are scored in both arms and reported as a
    separate stratum."

    The registered headline Delta is the macro over ALL scored V5 Ln cells, and it stays the registered number.  This
    reports the same macro with that stratum EXCLUDED beside it, with the stratum's share of the headline, the median and
    the share of cells with Delta > 0, so nobody reads a headline carried by two cells as a broad effect
    (task X finding protocol VH-08)."""
    out: dict[str, Any] = {"rule": "section 11: the actinide-partner-eligibility cells are a separate REPORTING unit; "
                                   "the registered headline is the all-cell macro and this is printed beside it, never "
                                   "instead of it",
                           "status": NOT_COMPUTED}
    if per_cell is None or per_cell.empty or delta_col not in per_cell.columns:
        return out
    d = pd.to_numeric(per_cell[delta_col], errors="coerce")
    keys = None
    if dependent is not None and not dependent.empty and "actinide_dependent" in dependent.columns:
        dep_keys = {(str(a), str(b)) for a, b, f in zip(dependent[EM.METAL_STATE_COL], dependent[EM.SYSTEM_COL],
                                                        dependent["actinide_dependent"]) if bool(f)}
        keys = pd.Series([(str(a), str(b)) in dep_keys
                          for a, b in zip(per_cell[EM.METAL_STATE_COL], per_cell[EM.SYSTEM_COL])],
                         index=per_cell.index)
    n_all = int(d.notna().sum())
    all_macro = float(d.mean())
    out.update({"status": "computed", "n_cells_all": n_all, "macro_delta_all_cells": all_macro,
                "median_delta_all_cells": float(d.median()),
                "share_delta_positive_all_cells": float((d > 0).mean()),
                "headline_point": None if point is None else float(point)})
    if keys is None:
        out["stratum"] = NOT_COMPUTED
        return out
    inc = d[~keys.to_numpy()]
    out.update({"n_cells_actinide_dependent": int(keys.sum()), "n_cells_excluding_stratum": int(inc.notna().sum()),
                "macro_delta_excluding_stratum": float(inc.mean()) if len(inc) else None,
                "median_delta_excluding_stratum": float(inc.median()) if len(inc) else None,
                "share_delta_positive_excluding_stratum": float((inc > 0).mean()) if len(inc) else None,
                "macro_delta_actinide_dependent": float(d[keys.to_numpy()].mean()) if bool(keys.any()) else None,
                "stratum_contribution_to_macro": (float(d[keys.to_numpy()].sum()) / n_all) if n_all else None})
    contrib, tot = out["stratum_contribution_to_macro"], out["macro_delta_all_cells"]
    out["stratum_share_of_macro"] = None if not contrib or not tot else float(contrib / tot)
    return out


def v1_scheme_of(arm: str, design: str, state: D.PlanState) -> str:
    """``metrics.v1_scoring_units`` scheme of the arm's V1 records (the scorer's ``Store.v1_scheme``)."""
    arm = D.ARM_ALIASES.get(arm, arm)
    if design != "V1" or arm in DETERMINISTIC_ARMS or arm in D.B6_ARMS:
        return "exact"
    return "grouped" if (state.heavy_v1_scheme or "grouped10") == "grouped10" else "exact"


def margin_of(design: str, delta5: float) -> float:
    """:data:`READINGS` ``margins``."""
    return float(delta5) if design in ("V5", "V5-P", "V5-PAIR", "V5PAIR") else float(ET.MARGIN_FLOOR)


# --------------------------------------------------------------------------------------------- #
# the Ln test set and the actinide-dependent stratum
# --------------------------------------------------------------------------------------------- #

def is_ln_iii(label: Any) -> bool:
    """``Nd(III)`` -> True; X(?) / non-lanthanide / other oxidation states -> False."""
    if not isinstance(label, str) or label.endswith("(?)"):
        return False
    try:
        el, ox = EP.parse_state(label)
    except (ValueError, KeyError):
        return False
    return el in MET.LANTHANIDES and ox == 3


def ln_iii_mask(states: Iterable[Any]) -> np.ndarray:
    return np.array([is_ln_iii(s) for s in states], dtype=bool)


def ln_rows(frame: pd.DataFrame, design: str, *, state_col: str = EM.METAL_STATE_COL) -> pd.DataFrame:
    """The Ln(III) scored rows of a scoring frame (:data:`READINGS` ``ln_test_set``); V6 rows are refused."""
    if state_col not in frame.columns:
        raise KeyError(state_col)
    out = frame[ln_iii_mask(frame[state_col].to_numpy(dtype=object))]
    if "v6_target_row" in out.columns and out["v6_target_row"].astype(bool).any():
        raise AssertionError(f"{design}: a V6_TARGET_ROWS row in the Ln test set")
    return out


def actinide_dependent_cells(frame: pd.DataFrame, cells: Iterable[tuple[str, str]],
                             thr: CH.Thresholds = CH.PRIMARY, *, medium: str = "all",
                             pub_col: str = CH.PUB_BASIS) -> pd.DataFrame:
    """Per cell: the fold builder's eligibility inputs on all rows and with every actinide row removed, and
    ``actinide_dependent`` (:data:`READINGS` ``actinide_dependent``).  Target-free."""
    space = CH.EligibilitySpace(frame, medium, pub_col=pub_col)
    keep = ~D.actinide_rows(frame)
    recs = []
    for m, s in cells:
        full = space.check((m, s), thr)
        wo = space.check((m, s), thr, keep=keep)
        recs.append({EM.METAL_STATE_COL: m, EM.SYSTEM_COL: s, "unit_key": EM.UNIT_KEY_SEP.join((m, s)),
                     **{f"{k}_with": v for k, v in full.items()}, **{f"{k}_without_actinides": v for k, v in wo.items()},
                     "actinide_dependent": bool(full["eligible"] and not wo["eligible"])})
    return pd.DataFrame(recs, columns=[EM.METAL_STATE_COL, EM.SYSTEM_COL, "unit_key"]
                        + [f"{k}_with" for k in ("n_rows", "n_publications", "other_systems_for_metal",
                                                 "other_metal_states_for_system", "connected_without_cell", "eligible")]
                        + [f"{k}_without_actinides" for k in ("n_rows", "n_publications", "other_systems_for_metal",
                                                              "other_metal_states_for_system", "connected_without_cell",
                                                              "eligible")] + ["actinide_dependent"])


# --------------------------------------------------------------------------------------------- #
# contrasts and deltas (sections 8, 11)
# --------------------------------------------------------------------------------------------- #

def _label(design: str) -> str:
    return D.DESIGN_LABEL.get(design, design)


def _paired(a: pd.DataFrame, b: pd.DataFrame, what: str) -> None:
    if set(a.index) != set(b.index):
        raise ValueError(f"{what}: the WITH and transformed arms score different Ln rows "
                         f"({len(set(a.index) ^ set(b.index))} differ); an incomplete transform is never scored")


def h3_contrast(with_fr: pd.DataFrame, other_fr: pd.DataFrame, *, design: str, model_arm: str, transform: str,
                margin: float, v6_mask: pd.Series, v1_scheme: str = "exact",
                remainder_groups: Iterable[str] | None = None, hurts: bool = False, seed: int = D.PRIMARY_SEED,
                n_resamples: int = ET.N_RESAMPLES, family: str = FAMILY) -> dict[str, Any]:
    """R19 of WITH vs the transform (``hurts=False``: Delta = MAE(transform) - MAE(WITH), positive favours WITH) or of
    the transform vs WITH (``hurts=True``) on the design's Ln test set, through ``discovery.evaluate_contrast``: the
    section 8 paired cluster bootstrap under every registered cluster unit, R19 items 1-6 (item 4 NOT_EVALUATED for a
    learned arm, addendum 1 item 3; item 6 on the scoring-filter sensitivities, the refits UNTESTABLE), TOST.

    ``n_resamples`` other than the registered ``transfer.N_RESAMPLES`` makes R19 items 2 and 3 FAIL by construction (they
    accept only the registered bootstrap), so such a result carries ``bootstrap_not_registered`` and is never a
    registered verdict; leave the default for anything that is reported."""
    label = _label(design)
    a, b = ln_rows(with_fr, design), ln_rows(other_fr, design)
    _paired(a, b, f"{model_arm}:{transform}@{design}")
    b = b.loc[a.index]
    cand, comp = (b, a) if hurts else (a, b)
    cand_name = h3_arm_name(model_arm, transform if hurts else "WITH")
    comp_name = h3_arm_name(model_arm, "WITH" if hurts else transform)
    rem = None if remainder_groups is None else list(remainder_groups)
    kw = dict(v6_mask=v6_mask, cand_v1_scheme=v1_scheme, comp_v1_scheme=v1_scheme, remainder_groups=rem)
    pu = D.paired_units(cand, comp, label, candidate=cand_name, comparator=comp_name, **kw)
    reg = ET.REGISTERED_SENSITIVITIES[label]
    sens: dict[str, Any] = {}
    reduced: list[str] = []
    for name in D.SCORING_FILTER_SENSITIVITIES:
        if name not in reg:
            continue
        c, k = D.filtered_pair(cand, comp, name)
        sens[name] = ET.UNTESTABLE if c.empty else D.paired_units(c, k, label, candidate=cand_name, comparator=comp_name,
                                                                  **kw).delta
        reduced.append(name)
    not_run = {n: H3_REFITS_NOT_RUN.get(n, REFIT_NOT_RUN) for n in reg if n not in reduced}
    for n in not_run:
        sens[n] = ET.UNTESTABLE
    det = is_deterministic(model_arm)
    res = D.evaluate_contrast(name=f"{cand_name} vs {comp_name}", family=family, design=label, primary=pu, margin=margin,
                              seed_deltas={int(seed): pu.delta}, sensitivities=sens, deterministic=det, learned=not det,
                              reduced_sensitivities=reduced if not det else None, n_resamples=n_resamples,
                              # every registered sensitivity of the design that H3 does not decide is DECLARED, including
                              # the two ``discovery.LEARNED_REFITS_NOT_RUN`` does not cover for V5 (strict_setting,
                              # HNO3_only_cells): task X finding protocol VH-03
                              sensitivities_not_run_extra=not_run)
    res.update({"model_arm": D.ARM_ALIASES.get(model_arm, model_arm), "transform": transform,
                "direction": "hurts (transform candidate)" if hurts else "helps (WITH candidate)",
                "h3_refits_not_run": not_run, "n_ln_rows": int(len(a)), "verdict_scope": VERDICT_SCOPE,
                "passes_discovery_scope": res["scopes"][VERDICT_SCOPE]["verdict"] == "PASS",
                "n_resamples": int(n_resamples),
                "key": f"{cand_name} vs {comp_name}@{label}", "readings": {k: READINGS[k] for k in ("r19_scope", "margins")}})
    if int(n_resamples) != ET.N_RESAMPLES:
        # R19 items 2 and 3 accept only the registered bootstrap (10,000 resamples, seed 19), so a reduced run FAILS
        # them by construction: flag it so no caller reads such a verdict as a registered result
        res["bootstrap_not_registered"] = (f"{n_resamples} resamples, not the registered {ET.N_RESAMPLES}: R19 items 2 "
                                          "and 3 FAIL by construction and this verdict is not a registered result")
    return res


def _unit_frames(with_fr: pd.DataFrame, other_fr: pd.DataFrame, design: str, *, v1_scheme: str,
                 remainder_groups: Iterable[str] | None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    label = _label(design)
    a, b = ln_rows(with_fr, design), ln_rows(other_fr, design)
    _paired(a, b, f"{design} units")
    rem = None if remainder_groups is None else list(remainder_groups)
    fa = EM.with_registered_units(a, label, v1_scheme=v1_scheme, v1_group_col=EM.PUB_GROUP_COL, remainder_groups=rem)
    fb = EM.with_registered_units(b.loc[a.index], label, v1_scheme=v1_scheme, v1_group_col=EM.PUB_GROUP_COL,
                                  remainder_groups=rem)
    return fa, fb, label


def _clusters(fr: pd.DataFrame, label: str, *, v1_scheme: str, remainder_groups: Iterable[str] | None
              ) -> dict[str, pd.Series]:
    rem = None if remainder_groups is None else list(remainder_groups)
    kw: dict[str, Any] = {"pub_group_col": EM.PUB_GROUP_COL}
    if label == "V1":
        kw.update(v1_scheme=v1_scheme, v1_group_col=EM.PUB_GROUP_COL, remainder_groups=rem)
    return EM.design_unit_clusters(fr, label, **kw)


def _regime(model_arm: str, transform: str, design: str, metric: str) -> dict[str, Any]:
    return {"family": FAMILY, "model_arm": D.ARM_ALIASES.get(model_arm, model_arm), "transform": transform,
            "design": design, "metric": metric, "half": "selection", "seed_set": "discovery",
            "decision_seed": D.PRIMARY_SEED, "label": "discovery, optimistically biased (selection half)"}


def rank_accuracy_delta(with_fr: pd.DataFrame, other_fr: pd.DataFrame, *, design: str, model_arm: str, transform: str,
                        v6_mask: pd.Series, v1_scheme: str = "exact", remainder_groups: Iterable[str] | None = None,
                        n_resamples: int = ET.N_RESAMPLES) -> list[dict[str, Any]]:
    """Delta rank accuracy = RA(WITH) - RA(transform) (positive favours WITH) over the units with an eligible row pair
    (section 4; pair eligibility never looks at a prediction, so both arms share the units), paired cluster bootstrap
    under every registered cluster unit.  One record per cluster unit."""
    fa, fb, label = _unit_frames(with_fr, other_fr, design, v1_scheme=v1_scheme, remainder_groups=remainder_groups)
    FR.assert_not_scored(fa.index, v6_mask, f"{design} rank accuracy")
    ucols = EM.registered_unit_cols(label)
    ra = EM.rank_accuracy_table(fa, unit_cols=ucols, pub_group_col=EM.PUB_GROUP_COL)
    rb = EM.rank_accuracy_table(fb, unit_cols=ucols, pub_group_col=EM.PUB_GROUP_COL)
    ra.index = pd.Index(EM.unit_keys(ra, ucols).to_numpy(), name="unit_key")
    rb.index = pd.Index(EM.unit_keys(rb, ucols).to_numpy(), name="unit_key")
    keep = ra.index[(ra["n_pairs"] > 0).to_numpy()]
    if not len(keep):
        return [{**_regime(model_arm, transform, label, "rank_accuracy"), "status": "NOT_RUN",
                 "detail": "no unit with an eligible row pair"}]
    cl = _clusters(fa, label, v1_scheme=v1_scheme, remainder_groups=remainder_groups)
    out = []
    for u in ET.REGISTERED_CLUSTER_UNITS[label]:
        br = ET.paired_cluster_bootstrap(rb.loc[keep, "rank_accuracy"], ra.loc[keep, "rank_accuracy"], cl[u].loc[keep],
                                         higher_is_better=True, n_resamples=n_resamples,
                                         contrast=f"{h3_arm_name(model_arm, 'WITH')} - {h3_arm_name(model_arm, transform)}",
                                         cluster_unit=u)
        rec = br.record(_regime(model_arm, transform, label, "rank_accuracy"))
        rec.update(with_value=float(ra.loc[keep, "rank_accuracy"].mean()),
                   transform_value=float(rb.loc[keep, "rank_accuracy"].mean()), n_pairs=int(ra.loc[keep, "n_pairs"].sum()),
                   primary_cluster_unit=u == ET.REGISTERED_CLUSTER_UNITS[label][0], status="computed")
        out.append(rec)
    return out


def _has_intervals(fr: pd.DataFrame, level: float) -> bool:
    lo, hi = EM.interval_columns(level)
    return lo in fr.columns and hi in fr.columns and bool(np.isfinite(fr[[lo, hi]].to_numpy(dtype=float)).all())


def calibration_delta(with_fr: pd.DataFrame, other_fr: pd.DataFrame, *, design: str, model_arm: str, transform: str,
                      v6_mask: pd.Series, v1_scheme: str = "exact", remainder_groups: Iterable[str] | None = None,
                      levels: Sequence[float] = (0.80, 0.95), n_resamples: int = ET.N_RESAMPLES) -> list[dict[str, Any]]:
    """Delta calibration (section 11): for each level ``|cov(transform) - target| - |cov(WITH) - target|`` (positive
    favours WITH) and ``width(transform) - width(WITH)`` of the unit macro, recomputed on every cluster resample
    (``transfer.cluster_bootstrap_statistic``); CRPS NOT_RUN (:data:`READINGS` ``crps``)."""
    fa, fb, label = _unit_frames(with_fr, other_fr, design, v1_scheme=v1_scheme, remainder_groups=remainder_groups)
    out: list[dict[str, Any]] = []
    cl = _clusters(fa, label, v1_scheme=v1_scheme, remainder_groups=remainder_groups)
    ucols = EM.registered_unit_cols(label)
    for lvl in levels:
        pct = int(round(lvl * 100))
        if not (_has_intervals(fa, lvl) and _has_intervals(fb, lvl)):
            out.append({**_regime(model_arm, transform, label, f"abs_coverage_gap_{pct}"), "status": "NOT_RUN",
                        "detail": f"no finite {pct} % interval on every Ln row of both arms"})
            continue
        pa = EM.per_unit_table(fa, unit_cols=ucols, levels=[lvl])
        pb = EM.per_unit_table(fb, unit_cols=ucols, levels=[lvl])
        pa.index = pd.Index(EM.unit_keys(pa, ucols).to_numpy(), name="unit_key")
        pb.index = pd.Index(EM.unit_keys(pb, ucols).to_numpy(), name="unit_key")
        pb = pb.loc[pa.index]
        base = pd.DataFrame({"cov_with": pa[f"coverage_{pct}"].to_numpy(dtype=float),
                             "cov_other": pb[f"coverage_{pct}"].to_numpy(dtype=float),
                             "w_with": pa[f"width_{pct}"].to_numpy(dtype=float),
                             "w_other": pb[f"width_{pct}"].to_numpy(dtype=float)}, index=pa.index)
        for u in ET.REGISTERED_CLUSTER_UNITS[label]:
            fr = base.assign(cluster=cl[u].loc[pa.index].astype(str).to_numpy())

            def gap(f: pd.DataFrame, target: float = float(lvl)) -> float:
                return abs(float(f["cov_other"].mean()) - target) - abs(float(f["cov_with"].mean()) - target)

            def width(f: pd.DataFrame) -> float:
                return float(f["w_other"].mean()) - float(f["w_with"].mean())
            for metric, stat in ((f"abs_coverage_gap_{pct}", gap), (f"width_{pct}", width)):
                br = ET.cluster_bootstrap_statistic(fr, "cluster", stat, n_resamples=n_resamples, contrast=f"{metric} "
                                                    f"{h3_arm_name(model_arm, transform)} - {h3_arm_name(model_arm, 'WITH')}",
                                                    cluster_unit=u)
                rec = br.record(_regime(model_arm, transform, label, metric))
                rec.update(with_value=float(base["cov_with"].mean()) if metric.startswith("abs") else float(base["w_with"].mean()),
                           transform_value=float(base["cov_other"].mean()) if metric.startswith("abs") else float(base["w_other"].mean()),
                           target=float(lvl) if metric.startswith("abs") else float("nan"),
                           primary_cluster_unit=u == ET.REGISTERED_CLUSTER_UNITS[label][0], status="computed")
                out.append(rec)
    out.append({**_regime(model_arm, transform, label, "crps"), "status": "NOT_RUN",
                "detail": D.UNCERTAINTY_NOT_RUN["gaussian_crps"]})
    return out


def per_unit_delta_table(with_fr: pd.DataFrame, other_fr: pd.DataFrame, *, design: str, model_arm: str, transform: str,
                         v6_mask: pd.Series, v1_scheme: str = "exact",
                         remainder_groups: Iterable[str] | None = None) -> pd.DataFrame:
    """Per Ln unit: MAE of both arms, ``delta_mae = MAE(transform) - MAE(WITH)`` (positive favours WITH), the unit's
    system, metal state and majority publication group (the negative-transfer inputs)."""
    fa, fb, label = _unit_frames(with_fr, other_fr, design, v1_scheme=v1_scheme, remainder_groups=remainder_groups)
    rem = None if remainder_groups is None else list(remainder_groups)
    kw: dict[str, Any] = {"v6_mask": v6_mask}
    if label == "V1":
        # discovery.unit_mae fixes the V1 group column itself (EM.PUB_GROUP_COL, the section 3.2 unit); passing it here
        # raised TypeError, so the V1 per-unit table could never be built (fixed 2026-09-23, not score-driven)
        kw.update(v1_scheme=v1_scheme, remainder_groups=rem)
    pa = D.unit_mae(fa, label, **kw)
    pb = D.unit_mae(fb, label, **kw).loc[pa.index]
    ucols = list(EM.registered_unit_cols(label))
    cl = _clusters(fa, label, v1_scheme=v1_scheme, remainder_groups=remainder_groups)
    units = EM.with_registered_units(fa, label, v1_scheme=v1_scheme, v1_group_col=EM.PUB_GROUP_COL, remainder_groups=rem)
    keys = EM.unit_keys(units, ucols)
    n_rows = keys.value_counts()
    out = pd.DataFrame({"unit_key": pa.index.to_numpy(), "mae_with": pa.to_numpy(dtype=float),
                        "mae_transform": pb.to_numpy(dtype=float)})
    out["delta_mae"] = out["mae_transform"] - out["mae_with"]
    out["n_rows"] = out["unit_key"].map(n_rows).astype(int)
    for name, ser in cl.items():
        out[name] = out["unit_key"].map(ser).astype(object)
    if label in ("V5", "V5-P"):
        parts = out["unit_key"].str.split(EM.UNIT_KEY_SEP, n=1, expand=True)
        out[EM.METAL_STATE_COL], out[EM.SYSTEM_COL] = parts[0], parts[1]
    out.insert(0, "transform", transform)
    out.insert(0, "model_arm", D.ARM_ALIASES.get(model_arm, model_arm))
    out.insert(0, "design", label)
    return out


def logsf_delta(pairs: pd.DataFrame, with_pred: pd.Series, other_pred: pd.Series, *, model_arm: str, transform: str,
                v6_mask: pd.Series | None, folds: pd.Series, n_resamples: int = ET.N_RESAMPLES) -> list[dict[str, Any]]:
    """Delta logSF MAE on the Ln-Ln pairs of a V5-PAIR pair set: ``MAE(transform) - MAE(WITH)`` per cell pair (positive
    favours WITH), paired system-cluster bootstrap (the V5-PAIR primary cluster) and the publication-group cluster."""
    ln = pairs[pairs["category_class"].astype(str) == "Ln-Ln"]
    if ln.empty:
        return [{**_regime(model_arm, transform, "V5-PAIR", "logsf_mae"), "status": "NOT_RUN", "detail": "no Ln-Ln pair"}]
    la = EP.derived_logsf(ln, with_pred)
    lb = EP.derived_logsf(ln, other_pred)
    sa = EP.score_pairs(ln, la, design="V5-PAIR")
    sb = EP.score_pairs(ln, lb, design="V5-PAIR")
    ua = EP.pair_unit_table(sa, design="V5-PAIR")
    ub = EP.pair_unit_table(sb, design="V5-PAIR")
    keys = ua[list(EP.CELL_PAIR_COLS)].astype(str).agg(" | ".join, axis=1)
    idx = pd.Index(keys.to_numpy(), name="unit_key")
    a = pd.Series(ua["logsf_mae"].to_numpy(dtype=float), index=idx)
    b = pd.Series(ub["logsf_mae"].to_numpy(dtype=float), index=idx)
    clusters = {"system": pd.Series(ua[EM.SYSTEM_COL].astype(str).to_numpy(), index=idx)}
    if EM.PUB_GROUP_COL in ln.columns:
        maj = ln.groupby(list(EP.CELL_PAIR_COLS))[EM.PUB_GROUP_COL].agg(lambda s: s.astype(str).value_counts().index[0])
        maj.index = pd.Index([" | ".join(map(str, k)) for k in maj.index])
        clusters["publication_group"] = maj.reindex(idx).astype(str)
    out = []
    for u in ET.REGISTERED_CLUSTER_UNITS["V5-PAIR"]:
        if u not in clusters:
            continue
        br = ET.paired_cluster_bootstrap(b, a, clusters[u], n_resamples=n_resamples, cluster_unit=u,
                                         contrast=f"logSF MAE {h3_arm_name(model_arm, transform)} - {h3_arm_name(model_arm, 'WITH')}")
        rec = br.record(_regime(model_arm, transform, "V5-PAIR", "logsf_mae"))
        rec.update(with_value=float(a.mean()), transform_value=float(b.mean()), n_pairs=int(len(ln)),
                   primary_cluster_unit=u == "system", status="computed")
        out.append(rec)
    return out


# --------------------------------------------------------------------------------------------- #
# verdicts (section 11) and F4 (section 10)
# --------------------------------------------------------------------------------------------- #

def _pass(res: Mapping[str, Any] | None) -> bool | None:
    return None if res is None else bool(res["scopes"][VERDICT_SCOPE]["verdict"] == "PASS")


def _ni(res: Mapping[str, Any] | None) -> bool | None:
    return None if res is None else bool(res["tost"].get("non_inferior"))


def h3_verdict(*, helps: Mapping[str, Mapping[str, Mapping[str, Any]]], hurts: Mapping[str, Mapping[str, Any]],
               kappa_min: float | None = None, kappa_status: str = NOT_COMPUTED,
               designs_not_run: Mapping[str, Any] | None = None,
               contrasts_not_run: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Section 11 verdicts of one model arm.

    ``helps[design][transform]`` are the WITH-candidate contrasts (:func:`h3_contrast`), ``hurts[transform]`` the V5
    transform-candidate contrasts.  *helps*: WITH passes R19 (:data:`VERDICT_SCOPE`) against WITHOUT and against
    ACT_PERMUTED on the V5 Ln cells and is non-inferior (TOST) to both on V1 and V2; *hurts*: WITHOUT passes against
    WITH on the V5 Ln cells; *equivalent*: the WITH - WITHOUT TOST on V5 is inside +-0.05; else UNDECIDED with
    kappa_min."""
    v5 = helps.get("V5", {})
    h_wo, h_pm = _pass(v5.get("WITHOUT")), _pass(v5.get("ACT_PERMUTED"))
    ni = {d: {t: _ni(helps.get(d, {}).get(t)) for t in ("WITHOUT", "ACT_PERMUTED")} for d in ("V1", "V2")}
    hurts_wo = _pass(hurts.get("WITHOUT"))
    tost = None if v5.get("WITHOUT") is None else v5["WITHOUT"]["tost"]
    equivalent = bool(tost and tost.get("equivalent"))
    helps_ok = bool(h_wo and h_pm and all(v for d in ni.values() for v in d.values()))
    if helps_ok:
        verdict = "helps"
    elif hurts_wo:
        verdict = "hurts"
    elif equivalent:
        verdict = "equivalent"
    else:
        verdict = "UNDECIDED"
    inputs_missing = [k for k, v in (("V5 WITH vs WITHOUT", h_wo), ("V5 WITH vs ACT_PERMUTED", h_pm),
                                     ("V5 WITHOUT vs WITH", hurts_wo)) if v is None]
    # POST-HOC addendum 4 item 2: a design the 20 h cap did not reach is NOT_RUN, and no verdict is taken from it; it is
    # named here so a "helps" verdict can never be read off V5 alone (helps_ok already requires V1 and V2 non-inferiority,
    # which a NOT_RUN design leaves None)
    nrun = {str(d): dict(v) if isinstance(v, Mapping) else v for d, v in dict(designs_not_run or {}).items()}
    inputs_missing += [f"{d}: {NOT_RUN}" for d in sorted(nrun)]
    # ... and every incomplete CONTRAST record set, by its own status (CONTRAST_UNIT_RULE): a leg that could not be
    # scored is named here even when another leg of the same design was, so no verdict can be read as complete
    crun = {str(k): dict(v) if isinstance(v, Mapping) else v for k, v in dict(contrasts_not_run or {}).items()}
    inputs_missing += [f"{k}: {(v or {}).get('status', INCOMPLETE_GUARD_FAILURE) if isinstance(v, Mapping) else v}"
                       for k, v in sorted(crun.items())]
    rev = hurts.get("WITHOUT")
    rev_point = None if rev is None or rev.get("point") is None else float(rev["point"])
    return {"verdict": verdict, "helps": helps_ok, "hurts": bool(hurts_wo), "equivalent": equivalent,
            "v5_with_beats_without": h_wo, "v5_with_beats_permuted": h_pm, "v1_v2_non_inferior": ni,
            # the reversed V5 contrast's point estimate (MAE(WITH) - MAE(WITHOUT), primary cluster unit): with ``hurts``
            # it is the section 11 condition of the negative-transfer investigation (addendum 2 item 4)
            "v5_without_vs_with_point": rev_point,
            "tost_v5_with_minus_without": None if tost is None else {k: tost[k] for k in ("verdict", "low_90", "high_90",
                                                                                          "equivalent", "non_inferior")},
            "kappa_min": kappa_min, "kappa_status": kappa_status if verdict == "UNDECIDED" else "not needed",
            "underpowered": (verdict == "UNDECIDED" and kappa_min is not None and kappa_min > ET.KAPPA_MIN_INFORMATIVE),
            "actinide_rows_enter_deployed_configuration": verdict == "helps",
            "inputs_missing": inputs_missing, "scope": VERDICT_SCOPE,
            "designs_not_run": nrun, "contrasts_not_run": crun, "contrast_unit_rule": CONTRAST_UNIT_RULE,
            "designs_scored": sorted(d for d in helps if d not in nrun),
            "transforms_scored": list(TRANSFORMS),
            "transforms_exploratory_not_scored": {t: EXPLORATORY_NOT_SCORED for t in EXPLORATORY_TRANSFORMS},
            "label": "discovery, optimistically biased (selection half; seed 104729); R19 item 4 decided at confirmation",
            "readings": {k: READINGS[k] for k in ("r19_scope", "equivalent", "helps_non_inferior", "act_metal_shuffled",
                                                  "design_priority")}}


def f4_check(hurts_by_design: Mapping[str, Mapping[str, Any]], *, deployed_arm: str,
             trained_with_actinides: bool = True) -> dict[str, Any]:
    """Section 10 F4: the deployed configuration was trained with actinide rows and the WITHOUT arm beats it on the Ln
    test set with the 95 % interval excluding 0 -- per design the reversed contrast's point > 0 and the primary-cluster
    lower bound > 0; F4 holds when any design shows it (:data:`READINGS` ``f4_designs``).

    **Which interval is a READING, both are reported, and the CONSERVATIVE one governs** (task X finding protocol VH-02 /
    numbers VH-03; TASK F item 3): section 10 F4 says "95 % interval" in the singular and names no construction, while
    section 8 registers the percentile AND the BCa interval for every contrast and R19 item 2 requires both to exclude 0.
    F4 is a FAILURE condition, so the conservative reading is the one that triggers more readily -- EITHER interval, on
    ANY registered design -- and that is what ``failure`` holds.  The percentile-only and BCa-only readings are computed
    and reported beside it (``failure_percentile``, ``failure_bca``, per design ``bca_beats``), and a design where they
    disagree is named, so no F4 statement is quoted without both."""
    per = {}
    for design, res in sorted(hurts_by_design.items()):
        if res is None:
            per[design] = {"status": NOT_COMPUTED,
                           "reason": "no reversed WITHOUT-vs-WITH contrast of this design was scored (its WITH or "
                                     f"WITHOUT record set is absent or incomplete, or the design is {NOT_RUN}): "
                                     "POST-HOC addendum 4 item 2 as read by CONTRAST_UNIT_RULE"}
            continue
        br = res["bootstraps"][res["primary_cluster_unit"]]
        plo, phi = br.percentile_interval()
        blo, bhi = br.bca_interval()
        pos = bool(np.isfinite(res["point"]) and res["point"] > 0)
        beats = bool(pos and np.isfinite(plo) and plo > 0)
        bca_beats = bool(pos and np.isfinite(blo) and blo > 0)
        per[design] = {"status": "computed", "point": res["point"], "percentile_95": [plo, phi], "bca_95": [blo, bhi],
                       "cluster_unit": res["primary_cluster_unit"], "without_beats_with_interval_excludes_0": beats,
                       "bca_also_excludes_0": bool(np.isfinite(blo) and blo > 0), "bca_beats": bca_beats,
                       "readings_disagree": bool(beats != bca_beats)}
    any_beats = any(p.get("without_beats_with_interval_excludes_0") for p in per.values())
    any_bca = any(p.get("bca_beats") for p in per.values())
    computed = [d for d, p in per.items() if p["status"] == "computed"]
    disagree = sorted(d for d, p in per.items() if p.get("readings_disagree"))
    return {"failure": bool(trained_with_actinides and (any_beats or any_bca)), "deployed_arm": deployed_arm,
            "deployed_trained_with_actinide_rows": bool(trained_with_actinides), "per_design": per,
            "designs_computed": computed, "designs_not_computed": [d for d in per if d not in computed],
            "interval_read": "either (percentile OR BCa) -- the conservative reading of a failure condition",
            "failure_percentile": bool(trained_with_actinides and any_beats),
            "failure_bca": bool(trained_with_actinides and any_bca),
            "failure_either_interval": bool(trained_with_actinides and (any_beats or any_bca)),
            "designs_where_readings_disagree": disagree,
            "interval_reading_not_registered": F4_INTERVAL_READING,
            "interval_reading_registered_by": "POST-HOC addendum 5 item 3",
            "deployed_configuration_reading": F4_DEPLOYED_CONFIGURATION_READING,
            # addendum 5 item 3: the two readings side by side.  (i) against a WITH deployment, the headline above;
            # (ii) against the deployment section 11 actually registers -- the WITHOUT-actinide fit on any verdict that is
            # not *helps* -- F4's first clause is false of the configuration deployed, whatever the intervals say
            "both_readings": {
                "against_a_with_deployment": bool(trained_with_actinides and (any_beats or any_bca)),
                "against_the_registered_without_deployment": False,
                "why": F4_DEPLOYED_CONFIGURATION_READING,
                "governs": "against_a_with_deployment (the conservative headline; no verdict is softened)"},
            "rule": "F4 -- negative actinide transfer (section 10)", "reading": READINGS["f4_designs"],
            "status": "computed" if computed else NOT_COMPUTED}


# --------------------------------------------------------------------------------------------- #
# negative-transfer investigation (descriptive)
# --------------------------------------------------------------------------------------------- #

def system_actinide_stats(frame: pd.DataFrame, *, family_col: str = "system_family", mechanism_col: str = "mechanism",
                          state_col: str = EM.METAL_STATE_COL, system_col: str = EM.SYSTEM_COL,
                          pub_group_col: str = EM.PUB_GROUP_COL, condition_key_col: str = EM.CONDITION_KEY_COL,
                          y_col: str = EM.Y_COL, monoamide_family: str = "monoamide") -> pd.DataFrame:
    """Per system (descriptive; target-derived amplitudes are printed, never claimed): actinide / lanthanide row
    counts, the comparable An-Ln pairs (``pairs.comparable_pairs`` on all rows as one fold) with their median
    |logSF| and the Am(III)/Eu(III) share, family, mechanism, whether the system is monoamide-only (monoamide family and
    no Ln(III) row) and whether it shares a component with a monoamide-only system."""
    el = frame[SG.ELEMENT_COL] if SG.ELEMENT_COL in frame.columns else frame[state_col].map(
        lambda s: EP.parse_state(s)[0] if is_ln_iii(s) or (isinstance(s, str) and "(" in s and not s.endswith("(?)")) else None)
    an = D.actinide_rows(frame.assign(**{SG.ELEMENT_COL: el}))
    ln = ln_iii_mask(frame[state_col].to_numpy(dtype=object))
    sys = frame[system_col].astype(str)
    out = pd.DataFrame({"n_rows": sys.value_counts()}).rename_axis(system_col).reset_index()
    out["n_actinide_rows"] = out[system_col].map(sys[an].value_counts()).fillna(0).astype(int)
    out["n_ln_iii_rows"] = out[system_col].map(sys[ln].value_counts()).fillna(0).astype(int)
    for col, name in ((family_col, "system_family"), (mechanism_col, "mechanism")):
        if col in frame.columns:
            first = frame.groupby(sys)[col].agg(lambda s: next((str(v) for v in s if isinstance(v, str)), None))
            out[name] = out[system_col].map(first)
        else:
            out[name] = None
    need = [pub_group_col, system_col, condition_key_col, state_col, y_col]
    if all(c in frame.columns for c in need):
        rows = frame[need].copy()
        rows["fold"] = "all"
        known = rows[rows[state_col].notna() & pd.to_numeric(rows[y_col], errors="coerce").notna()]
        pairs = EP.comparable_pairs(known, key_cols=(pub_group_col, system_col, condition_key_col), fold_col="fold",
                                    state_col=state_col, y_col=y_col)
        anln = pairs[pairs["category_class"].astype(str) == "An-Ln"]
        g = anln.groupby(system_col)
        out["n_an_ln_pairs"] = out[system_col].map(g.size()).fillna(0).astype(int)
        out["median_abs_an_ln_logsf"] = out[system_col].map(g["logsf_obs"].agg(lambda s: float(np.median(np.abs(s)))))
        ameu = anln[(anln["state_a"].astype(str).isin(("Am(III)", "Eu(III)"))) & (anln["state_b"].astype(str).isin(("Am(III)", "Eu(III)")))]
        share = ameu.groupby(system_col).size() / g.size()
        out["am_eu_pair_share"] = out[system_col].map(share)
    else:
        out["n_an_ln_pairs"], out["median_abs_an_ln_logsf"], out["am_eu_pair_share"] = 0, np.nan, np.nan
    mono_only = set(out.loc[(out["system_family"].astype(str) == monoamide_family) & (out["n_ln_iii_rows"] == 0),
                            system_col])
    out["monoamide_only_system"] = out[system_col].isin(mono_only)
    keys = list(out[system_col])
    share_mono = []
    for s in keys:
        others = SG.systems_sharing_component(keys, s) if mono_only else set()
        share_mono.append(bool(others & mono_only))
    out["shares_component_with_monoamide_only_system"] = share_mono
    return out.sort_values(system_col).reset_index(drop=True)


def negative_transfer_tables(per_cell: pd.DataFrame, systems: pd.DataFrame, *, dependent: pd.DataFrame | None = None,
                             system_col: str = EM.SYSTEM_COL, seed: int = ET.BOOTSTRAP_SEED,
                             n_resamples: int = ET.N_RESAMPLES) -> dict[str, pd.DataFrame]:
    """Section 11 negative-transfer investigation on the V5 per-cell deltas (descriptive): strata (Am/Eu-heavy
    diglycolamide cells; cells whose system shares a component with a monoamide-only system; the actinide-dependent
    stratum), per-family and per-mechanism deltas, and the per-cell delta against the system's actinide row count and
    its median |An/Ln logSF| (Spearman with a system-cluster interval; correlations are read only after the section 8
    reliability report)."""
    if per_cell.empty:
        return {}
    cells = per_cell.merge(systems, on=system_col, how="left", suffixes=("", "_sys"))
    if dependent is not None and not dependent.empty:
        cells = cells.merge(dependent[["unit_key", "actinide_dependent"]], on="unit_key", how="left")
        cells["actinide_dependent"] = cells["actinide_dependent"].fillna(False).astype(bool)
    else:
        cells["actinide_dependent"] = False
    cells["am_eu_heavy_dga"] = (cells["system_family"].astype(str) == "diglycolamide") & \
        (pd.to_numeric(cells["am_eu_pair_share"], errors="coerce").fillna(0.0) >= 0.5)
    cells["shares_component_with_monoamide_only_system"] = cells["shares_component_with_monoamide_only_system"].fillna(False).astype(bool)

    def summarise(g: pd.DataFrame) -> pd.Series:
        return pd.Series({"n_cells": int(len(g)), "n_systems": int(g[system_col].nunique()),
                          "mean_delta_mae": float(g["delta_mae"].mean()), "median_delta_mae": float(g["delta_mae"].median()),
                          "share_delta_positive": float((g["delta_mae"] > 0).mean()),
                          "mean_mae_with": float(g["mae_with"].mean()), "mean_mae_transform": float(g["mae_transform"].mean())})
    tables: dict[str, pd.DataFrame] = {}
    strata = []
    for name, col in (("am_eu_heavy_diglycolamide_cells", "am_eu_heavy_dga"),
                      ("cells_sharing_component_with_monoamide_only_system", "shares_component_with_monoamide_only_system"),
                      ("actinide_dependent_eligibility", "actinide_dependent")):
        for flag in (True, False):
            g = cells[cells[col].astype(bool) == flag]
            if len(g):
                strata.append({"stratum": name, "in_stratum": flag, **summarise(g).to_dict()})
    tables["strata"] = pd.DataFrame(strata)
    for col, name in (("system_family", "per_family"), ("mechanism", "per_mechanism")):
        g = cells.groupby(cells[col].astype(str), sort=True)
        tab = g.apply(summarise, include_groups=False).reset_index().rename(columns={col: name})
        tab.insert(0, "group", tab.pop(name) if name in tab.columns else tab.pop(col))
        tables[name] = tab
    corr = []
    for cov in ("n_actinide_rows", "median_abs_an_ln_logsf"):
        sub = cells[[cov, "delta_mae", system_col]].copy()
        sub[cov] = pd.to_numeric(sub[cov], errors="coerce")
        sub = sub[np.isfinite(sub[cov].to_numpy(dtype=float)) & np.isfinite(sub["delta_mae"].to_numpy(dtype=float))]
        if len(sub) < 3 or sub[system_col].nunique() < 2:
            corr.append({"covariate": cov, "n_cells": int(len(sub)), "spearman": float("nan"), "status": "NOT_RUN"})
            continue
        br = ET.cluster_bootstrap_statistic(sub.assign(cluster=sub[system_col].astype(str)), "cluster",
                                            lambda f, c=cov: EM.spearman_rho(f[c].to_numpy(dtype=float),
                                                                             f["delta_mae"].to_numpy(dtype=float)),
                                            n_resamples=n_resamples, seed=seed, contrast=f"spearman(delta_mae, {cov})",
                                            cluster_unit="system")
        plo, phi = br.percentile_interval()
        corr.append({"covariate": cov, "n_cells": int(len(sub)), "n_systems": int(sub[system_col].nunique()),
                     "spearman": br.point, "percentile_low": plo, "percentile_high": phi,
                     "status": "descriptive; read after the section 8 reliability report (UNDECIDED (unreliable) below "
                               "the 0.3 floor)"})
    tables["per_cell_covariates"] = pd.DataFrame(corr)
    tables["cells"] = cells
    return tables


# --------------------------------------------------------------------------------------------- #
# records (section 16) under evaluation/h3
# --------------------------------------------------------------------------------------------- #

def h3_root(out_root: Path) -> Path:
    return Path(out_root) / "evaluation" / "h3"


# --------------------------------------------------------------------------------------------- #
# the 20 h wall-clock cap of POST-HOC addendum 4 item 2 and the NOT_RUN designs
# --------------------------------------------------------------------------------------------- #

def h3_decisions_dir(out_root: Path) -> Path:
    return h3_root(out_root) / "decisions"


def h3_wall_clock_path(out_root: Path) -> Path:
    """``evaluation/h3/decisions/wall_clock.json`` -- the ledger POST-HOC addendum 4 item 2 names."""
    return h3_decisions_dir(out_root) / "wall_clock.json"


def h3_decisions_path(out_root: Path) -> Path:
    """``evaluation/h3/decisions/h3_decisions.json`` -- the exploratory transforms, the per-design statuses and the
    refits of POST-HOC addendum 4 items 1-3."""
    return h3_decisions_dir(out_root) / "h3_decisions.json"


def _rel(path: Path) -> str:
    """``paths.rel`` where the path is inside the repository, its own POSIX form otherwise (a ``tmp_path`` in tests)."""
    try:
        return paths.rel(path)
    except ValueError:
        return Path(path).resolve().as_posix()


def _iso(stamp: Any) -> float | None:
    try:
        return _dt.datetime.fromisoformat(str(stamp)).timestamp()
    except (TypeError, ValueError):
        return None


def union_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    """Seconds covered by the UNION of ``[start, end]`` intervals: the WALL CLOCK of several workers, so two workers
    running the same hour consume one hour (:data:`READINGS` ``h3_budget``)."""
    spans = sorted((float(a), float(b)) for a, b in intervals if b is not None and a is not None and float(b) >= float(a))
    total, cur_a, cur_b = 0.0, None, None
    for a, b in spans:
        if cur_a is None:
            cur_a, cur_b = a, b
        elif a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
    return total + (0.0 if cur_a is None else cur_b - cur_a)


def h3_wall_clock(out_root: Path) -> dict[str, Any]:
    """The ledger's wall clock: ``wall_seconds`` (the union of every recorded invocation's interval -- the number the cap
    reads), ``worker_seconds`` (their sum) and the invocation count.  An invocation without usable timestamps falls back
    to its own ``seconds`` so it can never be counted as free."""
    body = D.read_record(h3_wall_clock_path(out_root)) or {}
    invs = list(body.get("invocations") or [])
    spans, extra, worker = [], 0.0, 0.0
    for inv in invs:
        secs = float(inv.get("seconds") or 0.0)
        worker += secs
        a, b = _iso(inv.get("started_utc")), _iso(inv.get("ended_utc"))
        if a is None:
            extra += secs
            continue
        spans.append((a, b if b is not None else a + secs))
    return {"wall_seconds": union_seconds(spans) + extra, "worker_seconds": worker, "n_invocations": len(invs),
            "ledger": _rel(h3_wall_clock_path(out_root))}


def h3_budget_status(wall_seconds: float, *, worker_seconds: float | None = None,
                     budget_hours: float = H3_BUDGET_HOURS) -> dict[str, Any]:
    """POST-HOC addendum 4 item 2: wall clock against the 20 h cap, with the priority order and what a reached cap does.

    ``wall_seconds`` is the union of the workers' intervals (:func:`h3_wall_clock`); ``worker_seconds`` is recorded
    beside it and is NEVER the number compared with the cap."""
    used_h = float(wall_seconds) / 3600.0
    exhausted = used_h >= float(budget_hours)
    return {"budget_hours": float(budget_hours), "workers_max": H3_WORKERS, "used_hours": round(used_h, 4),
            "remaining_hours": round(float(budget_hours) - used_h, 4), "exhausted": exhausted,
            "worker_hours": None if worker_seconds is None else round(float(worker_seconds) / 3600.0, 4),
            "priority_order": list(DESIGN_PRIORITY), "ledger": "evaluation/h3/decisions/wall_clock.json",
            "on_exhaustion": ("the design being fitted and every later design in the priority order are reported "
                              f"{NOT_RUN} with the reason; no verdict is taken from a partial design"),
            "reading": READINGS["h3_budget"]}


def record_h3_wall_clock(out_root: Path, *, started: str, ended: str, seconds: float, designs_done: Sequence[str],
                         workers: int = 1, process_seconds: float | None = None, note: str | None = None,
                         path: Path | None = None) -> Path:
    """Append one invocation to ``evaluation/h3/decisions/wall_clock.json`` and rewrite the budget block.

    ``seconds`` is the fitting loop's own wall clock -- the number the cap reads; ``process_seconds`` is the whole
    invocation (gate, corpus load, scoring), recorded so an invocation that ran and skipped every fold is
    distinguishable from one that never ran."""
    from gen19ct.manifest import write_json as _write_json

    p = h3_wall_clock_path(out_root) if path is None else Path(path)
    body = D.read_record(p) or {"schema": SCHEMA, "invocations": []}
    body["invocations"] = list(body.get("invocations") or []) + [json_safe({
        "started_utc": str(started), "ended_utc": str(ended), "seconds": round(float(seconds), 3),
        "process_seconds": None if process_seconds is None else round(float(process_seconds), 3),
        "designs_done": list(designs_done), "workers": int(workers), "note": note})]
    paths.ensure_dir(p.parent)
    _write_json(p, body)                                    # so the clock below reads what was just appended
    wc = h3_wall_clock(out_root) if path is None else {
        "wall_seconds": union_seconds([(a, b) for a, b in
                                       ((_iso(i.get("started_utc")), _iso(i.get("ended_utc"))) for i in body["invocations"])
                                       if a is not None and b is not None]),
        "worker_seconds": sum(float(i.get("seconds") or 0.0) for i in body["invocations"]),
        "n_invocations": len(body["invocations"]), "ledger": str(p)}
    body.update({"budget_hours": H3_BUDGET_HOURS, "workers_max": H3_WORKERS,
                 "wall_hours": round(wc["wall_seconds"] / 3600.0, 4),
                 "worker_hours": round(wc["worker_seconds"] / 3600.0, 4),
                 "budget": h3_budget_status(wc["wall_seconds"], worker_seconds=wc["worker_seconds"]),
                 "invocation_reading": ("'seconds' is the fitting loop's own wall clock; the cap reads the UNION of the "
                                        "invocations' [started_utc, ended_utc] intervals (wall clock on up to "
                                        f"{H3_WORKERS} workers), never the sum"),
                 "reading": READINGS["h3_budget"]})
    return _write_json(p, body)


def open_h3_wall_clock(out_root: Path, *, prior_legs: Mapping[str, Any] | None = None, note: str | None = None,
                       path: Path | None = None) -> Path:
    """Create ``evaluation/h3/decisions/wall_clock.json`` with an EMPTY invocation list, the 20 h cap and its reading.

    POST-HOC addendum 4 item 2 grants the cap on 2026-09-23, after the V5 leg of the deployed arm had already been
    fitted; ``prior_legs`` records what that leg cost, measured from its own fold records, as a number the ledger
    REPORTS and does not count -- the cap governs the work the addendum scopes ("V5 (done), then V2, then V1").  An
    existing ledger is left untouched (its invocations are the cap's only input)."""
    from gen19ct.manifest import write_json as _write_json

    p = h3_wall_clock_path(out_root) if path is None else Path(path)
    if p.exists():
        return p
    body: dict[str, Any] = {
        "schema": SCHEMA, "invocations": [], "budget_hours": H3_BUDGET_HOURS, "workers_max": H3_WORKERS,
        "wall_hours": 0.0, "worker_hours": 0.0, "budget": h3_budget_status(0.0, worker_seconds=0.0),
        "opened_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "prior_legs_not_counted": dict(prior_legs or {}),
        "prior_legs_reading": ("POST-HOC addendum 4 item 2 grants the cap after the V5 leg of the deployed arm was "
                               "already fitted and names the priority order 'V5 (done), then V2, then V1', so the cap "
                               "governs the work from here; what the finished leg cost is measured from its own fold "
                               "records and REPORTED, never subtracted from the 20 h. If the orchestrator reads the cap "
                               "as covering H3 end to end, the prior hours below are the number to subtract"),
        "invocation_reading": ("'seconds' is the fitting loop's own wall clock; the cap reads the UNION of the "
                               "invocations' [started_utc, ended_utc] intervals (wall clock on up to "
                               f"{H3_WORKERS} workers), never the sum"),
        "note": note, "reading": READINGS["h3_budget"]}
    paths.ensure_dir(p.parent)
    return _write_json(p, body)


def measure_leg_seconds(out_root: Path, *, model_arm: str, transform: str, design_dir: str,
                        seed: int = D.PRIMARY_SEED) -> dict[str, Any]:
    """The wall clock one finished H3 leg cost, from its own fold records (``steps.<step>.seconds``): the per-fold mean
    and median and the serial total.  Used to price the remaining folds and to report what a finished leg cost."""
    d = h3_root(out_root) / "records" / D.ARM_ALIASES.get(model_arm, model_arm) / transform / design_dir / f"s{int(seed)}"
    secs = []
    for js in sorted(d.glob("*.json")) if d.exists() else []:
        rec = D.read_record(js)
        steps = (rec or {}).get("steps") or {}
        secs.append(sum(float(s.get("seconds") or 0.0) for s in steps.values() if isinstance(s, Mapping)))
    secs.sort()
    n = len(secs)
    return {"dir": _rel(d), "n_folds": n, "total_seconds": round(sum(secs), 1),
            "total_hours": round(sum(secs) / 3600.0, 4),
            "mean_seconds": None if not n else round(sum(secs) / n, 1),
            "median_seconds": None if not n else round(secs[n // 2] if n % 2 else (secs[n // 2 - 1] + secs[n // 2]) / 2, 1),
            "min_seconds": None if not n else round(secs[0], 1), "max_seconds": None if not n else round(secs[-1], 1)}


#: what a NOT_RUN design contributes, printed in its entry so the claim can be checked against the files
NOT_RUN_CONTRIBUTES = ("no contrast row, no R19 item row, no delta, no per-unit row, no F4 entry (NOT_COMPUTED) and no "
                       "V1 / V2 non-inferiority input (None)")


def not_run_design(design: str, reason: str, *, budget: Mapping[str, Any] | None = None,
                   n_folds_expected: int | None = None, n_folds_done: int | None = None,
                   blocked_by: str | None = None) -> dict[str, Any]:
    """The record of a design the plan did not reach or did not finish (POST-HOC addendum 4 item 2): ``NOT_RUN`` with its
    reason, never a partial verdict.

    ``blocked_by`` names the DEVIATION when the design was not stopped by the 20 h cap but by an error (addendum 4 item 2
    registers the arm set and says "That is registered and is run in full", so a design left incomplete by anything other
    than the cap is a deviation and is labelled one: task X finding protocol VH-07)."""
    cap = budget is not None and bool(dict(budget).get("exhausted"))
    return {"design": str(design), "status": NOT_RUN, "reason": str(reason),
            "n_folds_expected": None if n_folds_expected is None else int(n_folds_expected),
            "n_folds_done": None if n_folds_done is None else int(n_folds_done),
            "budget": None if budget is None else dict(budget), "verdict_taken": False,
            "contributes": NOT_RUN_CONTRIBUTES, "complete_legs": [],
            "stopped_by": "the 20 h wall-clock cap" if cap else (blocked_by or "not the cap"),
            "deviation": None if cap else (
                "POST-HOC addendum 4 item 2 registers the design scope and says 'That is registered and is run in "
                "full'; this design is incomplete and the 20 h cap was NOT reached, so its non-completion is a "
                "DEVIATION from the addendum, not an application of the cap"
                + (f" -- blocked by: {blocked_by}" if blocked_by else "")),
            "rule": "POST-HOC addendum 4 item 2: any design not reached is reported NOT_RUN with its reason -- no "
                    "verdict is taken from a partial design", "reading": READINGS["design_priority"]}


def blocking_errors(out_root: Path) -> dict[str, list[dict[str, Any]]]:
    """The fold errors the wall-clock ledger recorded, grouped by design (``<arm>:<transform>@<design>``).

    POST-HOC addendum 4 item 2 registers the H3 design scope and says it "is run in full"; the cap is the only registered
    reason a design may stay incomplete.  When a design is incomplete while the cap is unreached, the reason is an error,
    and the error must be named in the NOT_RUN entry and in D03 rather than hidden behind the cap's vocabulary
    (task X finding protocol VH-07).  The ledger's ``invocations[*].note`` is the only place the runner records it."""
    led = D.read_record(h3_wall_clock_path(out_root)) or {}
    out: dict[str, list[dict[str, Any]]] = {}
    for inv in led.get("invocations") or []:
        note = str((inv or {}).get("note") or "")
        if not note.startswith("error in "):
            continue
        # "error in B5:ACT_PERMUTED@V1: AssertionError: ..." -- the job key itself contains colons, so it is matched, not
        # split off
        m = re.search(r"([A-Za-z0-9_]+:[A-Za-z0-9_]+@([A-Za-z0-9_]+))", note)
        what, design = (m.group(1), m.group(2)) if m else (note[len("error in "):].strip(), "")
        out.setdefault(design, []).append({"what": what, "note": note, "started_utc": (inv or {}).get("started_utc"),
                                           "ended_utc": (inv or {}).get("ended_utc")})
    return out


def attach_not_run_context(not_run: Mapping[str, Any], *, scored_triples: Iterable[tuple[str, str, str]] = (),
                           blocking: Mapping[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Fill in, per ``NOT_RUN`` design, the legs that ARE complete and WHAT stopped it.

    The per-DESIGN view, which the 20 h cap's ledger uses (``run_plan``: the cap stops whole designs in the
    :data:`DESIGN_PRIORITY` order).  The scorer's unit is the contrast record set, so it uses
    :func:`attach_contrast_not_run_context` and :func:`drop_not_run_contrasts` instead (:data:`CONTRAST_UNIT_RULE`).

    Two things must not be lost when a design is suppressed: the record sets that are complete (real compute, kept on disk
    and scored as soon as the design completes) and the true cause, which addendum 4 item 2's vocabulary otherwise makes
    look like the 20 h cap.  When the cap is not exhausted the entry carries the blocking error from the wall-clock ledger
    and is labelled a DEVIATION (task X findings protocol VH-01 and VH-07)."""
    trip = list(scored_triples)
    out = dict(not_run)
    for design, st in out.items():
        if not isinstance(st, dict):
            continue
        st["complete_legs"] = sorted({f"{a}:{t}@{d}" for a, t, d in trip if str(d) == str(design)})
        st["complete_legs_reading"] = ("these record sets are COMPLETE and stay on disk; while the design is NOT_RUN "
                                      "nothing is read from them -- no contrast, delta, per-unit row, F4 entry or "
                                      "non-inferiority input (POST-HOC addendum 4 item 2)")
        errs = list((blocking or {}).get(str(design)) or [])
        cap = bool((st.get("budget") or {}).get("exhausted"))
        if errs:
            st["blocking_errors"] = errs
        if not cap:
            st["stopped_by"] = "; ".join(str(e.get("note")) for e in errs) if errs else "not the cap (cause not recorded)"
            st["deviation"] = (
                "POST-HOC addendum 4 item 2 registers the H3 design scope (V5 Ln cells, V2 Ln folds, V1 Ln rows; arms B5 "
                "and B6) and says 'That is registered and is run in full'. The 20 h cap is the only registered reason a "
                f"design may stay incomplete, and it was NOT reached, so {design} being incomplete is a DEVIATION from "
                "the addendum, not an application of the cap"
                + (f" -- cause: {st['stopped_by']}" if errs else ""))
    return out


def drop_not_run_designs(not_run: Mapping[str, Any], *, frames: Mapping[str, pd.DataFrame | None] = (),
                         rows: Mapping[str, list[dict[str, Any]]] = (),
                         nested: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """POST-HOC addendum 4 item 2, "no verdict is taken from a partial design": remove every row, frame row and nested
    ``[arm][design]`` entry of a ``NOT_RUN`` design, in place for the nested mappings and by returning filtered copies for
    the tabular ones.

    The per-DESIGN drop, for a design the 20 h cap never reached (no leg of it exists).  The scorer drops per CONTRAST
    RECORD SET (:func:`drop_not_run_contrasts`, :data:`CONTRAST_UNIT_RULE`), which removes a wholly-unreached design as a
    special case -- every leg of it is then incomplete -- while leaving a complete leg of a partly-run design scorable.

    The unit of NOT_RUN is the DESIGN (:data:`READINGS` ``design_priority``): a design one of whose arm x transform record
    sets is incomplete is NOT_RUN even when another leg is complete, so the complete leg's rows must not reach a table
    either -- otherwise ``designs_scored`` and ``designs_not_run`` name the same design and D03 prints both NOT_RUN and a
    FAIL verdict for it (task X finding protocol VH-01)."""
    bad = {str(d) for d in dict(not_run or {})}
    dropped: dict[str, Any] = {"designs": sorted(bad), "n_rows_dropped": {}, "n_frame_rows_dropped": {}}
    out_rows = {k: [r for r in v if str(r.get("design")) not in bad] for k, v in dict(rows or {}).items()}
    for k, v in dict(rows or {}).items():
        dropped["n_rows_dropped"][k] = len(v) - len(out_rows[k])
    out_frames: dict[str, pd.DataFrame | None] = {}
    for k, fr in dict(frames or {}).items():
        if fr is None or getattr(fr, "empty", True) or "design" not in getattr(fr, "columns", ()):
            out_frames[k] = fr
            dropped["n_frame_rows_dropped"][k] = 0
            continue
        keep = ~fr["design"].astype(str).isin(sorted(bad))
        out_frames[k] = fr[keep].reset_index(drop=True)
        dropped["n_frame_rows_dropped"][k] = int((~keep).sum())
    for holder in nested:
        for arm in list(holder):
            per = holder[arm]
            if isinstance(per, dict):
                for d in list(per):
                    if str(d) in bad:
                        per.pop(d)
    dropped["rows"], dropped["frames"] = out_rows, out_frames
    return dropped


def design_status(record_sets: Mapping[str, Any], design: str, transform: str, model_arm: str) -> dict[str, Any]:
    """The ``read_record_set`` status of one (transform, arm, design) as :class:`Frames` recorded it, as a design
    status: ``complete`` / ``NOT_RUN`` (missing) / ``NOT_RUN`` (partial: n of m folds)."""
    hit = [(k, v) for k, v in dict(record_sets).items()
           if isinstance(v, Mapping) and str(v.get("what") or "") == f"{model_arm}:{transform}@{design}"]
    if not hit:
        return not_run_design(design, f"no record set of {model_arm}:{transform}@{design} was read")
    key, st = hit[0]
    if str(st.get("status")) == "complete":
        return {"design": str(design), "status": "complete", "key": key, "n_folds": st.get("n_found"),
                "verdict_taken": True}
    return not_run_design(design, f"{model_arm}:{transform}@{design} record set is {st.get('status')} "
                                  f"({st.get('n_found')} of {st.get('n_expected')} folds; "
                                  f"missing {list(st.get('missing_folds') or [])[:5]})",
                          n_folds_expected=st.get("n_expected"), n_folds_done=st.get("n_found"))


#: what an incomplete CONTRAST record set contributes -- printed in its entry so the claim can be checked against the
#: files
CONTRAST_NOT_RUN_CONTRIBUTES = ("no contrast row, no R19 item row, no delta, no per-unit row, no F4 entry and no "
                                "V1 / V2 non-inferiority input (None) for THIS leg; the other legs of the same design "
                                "are unaffected")


def contrast_key(model_arm: str, transform: str, design: str) -> str:
    """The identity of one contrast record set: ``<arm>:<transform>@<design>``."""
    return f"{model_arm}:{transform}@{design}"


def not_run_contrast(model_arm: str, transform: str, design: str, reason: str, *,
                     status: str = INCOMPLETE_GUARD_FAILURE, budget: Mapping[str, Any] | None = None,
                     n_folds_expected: int | None = None, n_folds_done: int | None = None,
                     blocked_by: str | None = None) -> dict[str, Any]:
    """The record of ONE contrast record set that may not be scored (:data:`CONTRAST_UNIT_RULE`).

    ``status`` is :data:`NOT_RUN` only when the 20 h cap stopped it -- addendum 4 item 2's NOT_RUN vocabulary is the
    cap's.  Anything else (a deterministic guard failure, an absent record set) carries
    :data:`INCOMPLETE_GUARD_FAILURE` or the caller's own status, and the entry says so (TASK F items 1 and 2)."""
    cap = budget is not None and bool(dict(budget).get("exhausted"))
    st = NOT_RUN if cap else str(status)
    return {"key": contrast_key(model_arm, transform, design), "model_arm": str(model_arm),
            "transform": str(transform), "design": str(design), "status": st, "reason": str(reason),
            "n_folds_expected": None if n_folds_expected is None else int(n_folds_expected),
            "n_folds_done": None if n_folds_done is None else int(n_folds_done),
            "budget": None if budget is None else dict(budget), "verdict_taken": False,
            "contributes": CONTRAST_NOT_RUN_CONTRIBUTES,
            "stopped_by": "the 20 h wall-clock cap" if cap else (blocked_by or "not the cap"),
            "vocabulary": ("POST-HOC addendum 4 item 2's NOT_RUN is the CAP's vocabulary; this leg was not stopped by "
                           f"the cap, so it carries {INCOMPLETE_GUARD_FAILURE} instead and the cause is named"
                           if not cap else "POST-HOC addendum 4 item 2: the 20 h cap was reached"),
            "deviation": None if cap else (
                "POST-HOC addendum 4 item 2 registers the H3 design scope and says 'That is registered and is run in "
                "full'; this record set is incomplete and the 20 h cap was NOT reached, so its non-completion is a "
                "DEVIATION from the addendum, not an application of the cap"
                + (f" -- blocked by: {blocked_by}" if blocked_by else "")),
            "rule": CONTRAST_UNIT_RULE, "reading": READINGS["design_priority"]}


def contrast_status(record_sets: Mapping[str, Any], design: str, transform: str, model_arm: str) -> dict[str, Any]:
    """``complete`` / incomplete status of ONE contrast record set (:data:`CONTRAST_UNIT_RULE`)."""
    hit = [(k, v) for k, v in dict(record_sets).items()
           if isinstance(v, Mapping) and str(v.get("what") or "") == contrast_key(model_arm, transform, design)]
    if not hit:
        return not_run_contrast(model_arm, transform, design,
                                f"no record set of {contrast_key(model_arm, transform, design)} was read")
    key, st = hit[0]
    if str(st.get("status")) == "complete":
        return {"key": contrast_key(model_arm, transform, design), "model_arm": str(model_arm),
                "transform": str(transform), "design": str(design), "status": "complete", "record_set": key,
                "n_folds": st.get("n_found"), "verdict_taken": True}
    missing = list(st.get("missing_folds") or [])
    return not_run_contrast(model_arm, transform, design,
                            f"{contrast_key(model_arm, transform, design)} record set is {st.get('status')} "
                            f"({st.get('n_found')} of {st.get('n_expected')} folds; missing {missing[:5]}"
                            f"{' ...' if len(missing) > 5 else ''})",
                            n_folds_expected=st.get("n_expected"), n_folds_done=st.get("n_found"))


def attach_contrast_not_run_context(not_run: Mapping[str, Any], *,
                                    blocking: Mapping[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Fill in, per incomplete contrast record set, WHAT stopped it, from the wall-clock ledger's fold errors.

    ``blocking`` is :func:`blocking_errors` keyed by design; an entry is matched to the errors of its own
    ``<arm>:<transform>@<design>`` job key when there is one, and to the design's otherwise."""
    out = {k: dict(v) for k, v in dict(not_run or {}).items()}
    flat = [e for errs in (blocking or {}).values() for e in errs]
    for key, st in out.items():
        cap = bool((st.get("budget") or {}).get("exhausted"))
        mine = [e for e in flat if str(e.get("what")) == key] or \
            [e for e in (blocking or {}).get(str(st.get("design")), [])]
        if mine:
            st["blocking_errors"] = mine
        if not cap:
            st["stopped_by"] = "; ".join(str(e.get("note")) for e in mine) if mine else \
                "not the cap (cause not recorded)"
            st["deviation"] = (
                "POST-HOC addendum 4 item 2 registers the H3 design scope (V5 Ln cells, V2 Ln folds, V1 Ln rows; arms "
                "B5 and B6) and says 'That is registered and is run in full'. The 20 h cap is the only registered "
                f"reason a record set may stay incomplete, and it was NOT reached, so {key} being incomplete is a "
                "DEVIATION from the addendum, not an application of the cap"
                + (f" -- cause: {st['stopped_by']}" if mine else ""))
    return out


def drop_not_run_contrasts(not_run: Mapping[str, Any], *, frames: Mapping[str, pd.DataFrame | None] = (),
                           rows: Mapping[str, list[dict[str, Any]]] = (),
                           nested: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """:data:`CONTRAST_UNIT_RULE`: remove every row, frame row and nested ``[arm][design][transform]`` entry of an
    incomplete CONTRAST record set, leaving the complete legs of the same design untouched.

    The unit is the (model arm, transform, design) triple, which is what ``h3_contrast`` is computed from and what both
    directions (helps and hurts) of a leg share, so a single triple removes both rows of the pair."""
    bad = {(str(v.get("model_arm")), str(v.get("transform")), str(v.get("design")))
           for v in dict(not_run or {}).values() if isinstance(v, Mapping)}

    def is_bad(arm: Any, transform: Any, design: Any) -> bool:
        return (str(arm), str(transform), str(design)) in bad

    dropped: dict[str, Any] = {"contrast_record_sets": sorted(dict(not_run or {})), "n_rows_dropped": {},
                               "n_frame_rows_dropped": {}, "unit": "contrast record set (<arm>:<transform>@<design>)"}
    out_rows = {k: [r for r in v if not is_bad(r.get("model_arm"), r.get("transform"), r.get("design"))]
                for k, v in dict(rows or {}).items()}
    for k, v in dict(rows or {}).items():
        dropped["n_rows_dropped"][k] = len(v) - len(out_rows[k])
    out_frames: dict[str, pd.DataFrame | None] = {}
    for k, fr in dict(frames or {}).items():
        cols = set(getattr(fr, "columns", ()))
        if fr is None or getattr(fr, "empty", True) or not {"design", "model_arm", "transform"} <= cols:
            out_frames[k] = fr
            dropped["n_frame_rows_dropped"][k] = 0
            continue
        keep = ~pd.Series([is_bad(a, t, d) for a, t, d in
                           zip(fr["model_arm"], fr["transform"], fr["design"])], index=fr.index)
        out_frames[k] = fr[keep].reset_index(drop=True)
        dropped["n_frame_rows_dropped"][k] = int((~keep).sum())
    for holder in nested:
        for arm in list(holder):
            per = holder[arm]
            if not isinstance(per, dict):
                continue
            for d in list(per):
                inner = per[d]
                if not isinstance(inner, dict):
                    continue
                for t in list(inner):
                    if is_bad(arm, t, d):
                        inner.pop(t)
                if not inner:
                    per.pop(d)
    dropped["rows"], dropped["frames"] = out_rows, out_frames
    return dropped


def assert_no_partial_design(record_sets: Mapping[str, Any], scored: Iterable[tuple[str, str, str]]) -> dict[str, Any]:
    """POST-HOC addendum 4 item 2: "no verdict is taken from a partial design".

    ``scored`` lists the (model arm, transform, design) triples a contrast was actually computed for.  Every one of them
    must have a COMPLETE record set; a partial (``incomplete``) one raises :class:`AssertionError`, because
    :func:`h3_contrast` would then silently score a subset of the design's folds.  Returns the per-triple statuses."""
    out: dict[str, Any] = {}
    bad = []
    for arm, transform, design in scored:
        st = design_status(record_sets, design, transform, arm)
        out[f"{arm}:{transform}@{design}"] = st
        if st["status"] != "complete":
            bad.append(f"{arm}:{transform}@{design}: {st['reason']}")
    if bad:
        raise AssertionError("POST-HOC addendum 4 item 2: no verdict is taken from a partial design, yet a contrast was "
                             f"scored on an incomplete record set: {bad}")
    return out


def exploratory_records(out_root: Path, *, model_arms: Sequence[str] = (), state: D.PlanState | None = None,
                        designs: Sequence[str] = tuple(DESIGNS), seed: int = D.PRIMARY_SEED) -> dict[str, Any]:
    """POST-HOC addendum 4 item 1: every ACT_METAL_SHUFFLED record on disk, labelled
    :data:`EXPLORATORY_NOT_SCORED`.

    The records are neither deleted nor rewritten; this is the H3 decisions entry that says they exist and are not
    scored.  Directories are listed from disk (``records/<arm>/<transform>``), so the inventory does not depend on the
    plan."""
    root = h3_root(out_root) / "records"
    found: list[dict[str, Any]] = []
    for transform in EXPLORATORY_TRANSFORMS:
        for d in sorted(root.glob(f"*/{transform}/*/*")) if root.exists() else []:
            if not d.is_dir():
                continue
            js = sorted(d.glob("*.json"))
            found.append({"dir": _rel(d), "arm": d.parts[-4], "transform": transform, "design_dir": d.parts[-2],
                          "seed_dir": d.name, "n_records": len(js), "folds": [p.stem for p in js]})
    return {"transforms": list(EXPLORATORY_TRANSFORMS), "status": EXPLORATORY_NOT_SCORED,
            "n_records": sum(f["n_records"] for f in found), "record_sets": found,
            "scored": False, "deleted": False,
            "reason": ACT_METAL_SHUFFLED_NOT_RUN,
            "rule": "POST-HOC addendum 4 item 1: ACT_METAL_SHUFFLED is not run; the single fold already fitted is "
                    "labelled exploratory and is not scored (it is kept on disk, never rewritten)"}


# --------------------------------------------------------------------------------------------- #
# record layout and digests
# --------------------------------------------------------------------------------------------- #


def record_dir(out_root: Path, model_arm: str, transform: str, job: D.JobSpec) -> Path:
    return h3_root(out_root) / "records" / D.ARM_ALIASES.get(model_arm, model_arm) / transform / job.design_dir / f"s{job.seed}"


def fold_paths(out_root: Path, model_arm: str, transform: str, job: D.JobSpec, fold_id: str) -> tuple[Path, Path]:
    d = record_dir(out_root, model_arm, transform, job)
    name = D.safe_fold_name(fold_id)
    return d / f"{name}.parquet", d / f"{name}.json"


def fold_digest(job: D.JobSpec, fold: FI.Fold, code: str, *, transform: str, with_digest: str | None, guard_mode: str,
                design_hash: str, ordinal: int, model_seed: int | None, addenda: str | None = None) -> str:
    """The resume / verification digest of one H3 fold: ``discovery.fold_digest`` of the H3 job with the transform, the
    WITH record's digest (the hyperparameters it supplies), the guard mode, the fold file's design hash, the model fold
    number and seed and the registered pre-registration and addenda digests.  The addenda digest is the H3 stage's
    registry entry when ``manifests/digest_registry.json`` exists (addendum 2 item 5), else ``REGISTERED_ADDENDA_SHA256``.

    ``addenda`` overrides it with the below-footer digest an EXISTING record was WRITTEN under, exactly as ``code`` is
    overridden for the same reason (:func:`record_digest_basis`, :data:`READINGS` ``record_digest_basis``); a record
    being written now passes neither and carries the live pair."""
    return D.fold_digest(job, fold, code, {"schema": SCHEMA, "transform": transform, "with_record_digest": with_digest,
                                           "guard_mode": guard_mode, "design_hash": str(design_hash),
                                           "model_fold_number": int(ordinal), "model_seed": model_seed,
                                           "prereg_sha256": D.REGISTERED_PREREG_SHA256,
                                           "prereg_addenda_sha256": (REG.below_footer_sha256("h3") if addenda is None
                                                                     else str(addenda))})


def record_digest_basis(record: Mapping[str, Any] | None, live_code: str, *, stage: str = "h3",
                        path: Path | None = None) -> dict[str, Any]:
    """The (code digest, below-footer digest) ONE existing H3 fold record's digest is verified with.

    Addendum 2 item 5 ("a record is verified against the registry entry of its stage ... a later edit can neither
    validate nor invalidate a record written earlier") and POST-HOC addendum 4 item 3 ("records written before the
    supersession keep their entry and verify as registered"): the record's own ``code_digest`` and
    ``prereg_addenda_sha256`` are used when :func:`registry.verify_record` matches them to an entry of ``stage`` --
    current or superseded, ordered by the record's own timestamp.  Otherwise the LIVE pair is returned, so a stale
    record (``stale_after_supersession``) and a foreign record fail exactly where they did before and are refitted.

    Per RECORD, not per directory: after the addendum 4 item 3 refits a transform's directory holds records of two
    entries, and each must still be verified against its own."""
    live = {"code": str(live_code), "addenda": REG.below_footer_sha256(stage, path), "resolved": "live",
            "record_code": None, "record_addenda": None}
    if record is None:
        return {**live, "note": "no record on disk; the live code and registered below-footer digest govern"}
    have_code, have_add = record.get("code_digest"), record.get("prereg_addenda_sha256")
    live = {**live, "record_code": have_code, "record_addenda": have_add}
    if str(have_code) == live["code"] and str(have_add) == live["addenda"]:
        return {**live, "resolved": "current"}
    v = REG.verify_record(record, stage, path=path)
    if v.get("ok") is not True or not have_code or not have_add:
        return {**live, "verify": {k: v.get(k) for k in ("ok", "reason", "stale_after_supersession")},
                "note": "the record does not verify against any entry of this stage at its own timestamp: the live "
                        "digests govern and the record is refitted (POST-HOC addendum 4 item 3)"}
    return {"code": str(have_code), "addenda": str(have_add), "resolved": str(v.get("matched_entry") or "superseded"),
            "record_code": have_code, "record_addenda": have_add,
            "superseded_utc": (v.get("superseded_match") or {}).get("superseded_utc"),
            "written_utc": v.get("written_utc"),
            "note": f"the record was written under the {v.get('matched_entry')} entry of stage {stage!r} and verifies "
                    "against it (addendum 2 item 5; POST-HOC addendum 4 item 3)",
            "reading": READINGS["record_digest_basis"]}


def stale_records(out_root: Path, *, stage: str = "h3", path: Path | None = None) -> dict[str, Any]:
    """Every H3 fold record that does NOT verify against any entry of stage ``stage`` at its own timestamp.

    POST-HOC addendum 4 item 3: "Records written under a superseded code digest are refitted, not accepted" -- a record
    a runner holding stale code wrote AFTER its entry was superseded (``registry.verify_record`` ->
    ``stale_after_supersession``).  Returns the stale records, the verifying ones and the unverifiable ones grouped by
    record set.  Nothing is deleted here (:func:`delete_records`).

    A stale record of an EXPLORATORY transform is reported separately (``stale_exploratory``) and is NOT in ``stale``:
    POST-HOC addendum 4 item 1 keeps the one ACT_METAL_SHUFFLED fold on disk as :data:`EXPLORATORY_NOT_SCORED`, and item
    3's refits are the records a delta is scored from."""
    root = h3_root(out_root) / "records"
    stale: list[dict[str, Any]] = []
    stale_expl: list[dict[str, Any]] = []
    ok: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for js in sorted(root.glob("*/*/*/*/*.json")) if root.exists() else []:
        rec = D.read_record(js)
        if rec is None:
            other.append({"path": _rel(js), "reason": "unreadable record"})
            continue
        v = REG.verify_record(rec, stage, path=path)
        row = {"path": _rel(js), "arm": rec.get("model_arm"), "transform": rec.get("transform"),
               "design_dir": js.parts[-3], "seed_dir": js.parts[-2], "stem": rec.get("stem"),
               "fold_id": rec.get("fold_id"), "code_digest": rec.get("code_digest"),
               "prereg_addenda_sha256": rec.get("prereg_addenda_sha256"), "written_utc": REG.record_run_utc(rec),
               # the ordering that exempts a pre-supersession record from refitting rests on this stamp; a record
               # without written_utc is ordered by a DERIVED one, so the source and the check are recorded per record
               # (task X findings numbers VH-04 / VH-05)
               "written_utc_source": REG.record_run_utc_source(rec),
               "ordering_checked": bool(v.get("ordering_checked")),
               "superseded_utc_of_matched_entry": (v.get("superseded_match") or {}).get("superseded_utc"),
               "matched_entry": v.get("matched_entry")}
        if v.get("ok") is True:
            ok.append(row)
        elif v.get("stale_after_supersession"):
            hit = {**row, "stale_after_supersession": v["stale_after_supersession"], "reason": v.get("reason")}
            if is_scored_transform(str(row.get("transform"))):
                stale.append(hit)
            else:
                stale_expl.append({**hit, "status": EXPLORATORY_NOT_SCORED, "kept": True,
                                   "kept_reason": "POST-HOC addendum 4 item 1: the fold already fitted is labelled "
                                                  "exploratory and is not scored; it is not refitted and not deleted"})
        else:
            other.append({**row, "ok": v.get("ok"), "reason": v.get("reason")})
    by_entry: dict[str, int] = {}
    src: dict[str, int] = {}
    for r in (*ok, *stale, *stale_expl):
        by_entry[str(r.get("matched_entry"))] = by_entry.get(str(r.get("matched_entry")), 0) + 1
        src[str(r.get("written_utc_source"))] = src.get(str(r.get("written_utc_source")), 0) + 1
    n_current = by_entry.get("current", 0)
    return {"stage": stage, "n_records": len(stale) + len(stale_expl) + len(ok) + len(other), "stale": stale,
            "stale_exploratory": stale_expl, "verifying": ok, "not_verifying_other": other,
            "matched_entry_counts": by_entry, "written_utc_source_counts": src,
            "n_verifying_against_the_current_entry": n_current,
            "why_no_record_is_current": (
                None if n_current else
                f"NO record of stage '{stage}' verifies against the CURRENT registry entry, and none is refitted for "
                "that reason. Every record was written under the entry that was current at the moment of its fit; the "
                "entry was then re-registered by reporting-only edits to the scoring code (the per_unit_delta_table "
                "TypeError fix, the D03 reporting fixes and the fixes of this task), which addendum 2 change 5 says "
                "'can neither validate nor invalidate a record written earlier'. Refitting 444 folds because the REPORT "
                "changed would burn the H3 budget on a quantity nothing reads and would change no delta, so the records "
                "are kept and this is recorded instead (task X findings numbers VH-04; TASK F item 7). See "
                "operative_reading."),
            "rule": "POST-HOC addendum 4 item 3: a record written after its stage's entry was superseded, still carrying "
                    "the superseded digests, is DELETED and refitted under the current registered digest before any H3 "
                    "delta is scored; records written before the supersession keep their entry and verify as registered",
            # what "verifies" MEANS here, stated because no H3 record verifies against the CURRENT entry: the refits were
            # written under the entry that was current at the moment of the refit, and three later reporting-only edits
            # to h3.py superseded it before scoring (task X findings numbers VH-04 / protocol VH-06)
            "operative_reading": OPERATIVE_DIGEST_READING}


def delete_records(rows: Sequence[Mapping[str, Any]], *, out_root: Path | None = None, dry_run: bool = True
                   ) -> dict[str, Any]:
    """Delete exactly the listed fold records and their prediction parquets (POST-HOC addendum 4 item 3).

    ``dry_run`` (the default) only reports what would go.  Each row is one ``*.json`` path as :func:`stale_records`
    reports it; the sibling ``*.parquet`` goes with it, because a prediction without a record raises."""
    base = paths.REPO_ROOT if out_root is None else Path(out_root)
    planned, removed, missing = [], [], []
    for row in rows:
        js = Path(row["path"])
        js = js if js.is_absolute() else base / js
        pq = js.with_suffix(".parquet")
        for p in (js, pq):
            if not p.exists():
                missing.append(_rel(p))
                continue
            planned.append(_rel(p))
            if not dry_run:
                p.unlink()
                removed.append(_rel(p))
    return {"dry_run": bool(dry_run), "n_records": len(rows), "planned": planned, "removed": removed,
            "already_absent": missing,
            "rule": "POST-HOC addendum 4 item 3: exactly the records that do not verify at their own timestamp are "
                    "deleted; nothing else is touched and no record is rewritten"}


def record_set_code(directory: Path, live: str, *, stage: str = "h3", path: Path | None = None) -> dict[str, Any]:
    """The code digest the H3 record set in ``directory`` is VERIFIED with, as ``registry.record_dir_code`` resolves a
    discovery record set's (task X finding V-P02: fixing an H3 reading edits ``h3.py``, which is in this stage's code
    digest, and addendum 2 item 5 says such an edit "can neither validate nor invalidate a record written earlier").

    The set's own ``code_digest`` is used when every record in the directory agrees on it AND it is a code digest the
    registry holds for ``stage`` -- the current entry, or one it superseded (``registry.registered_code_digests``).
    Otherwise ``live`` is returned unchanged, so a genuinely foreign record still raises :class:`discovery.StaleRecordError`
    where it did before.  Records are FITTED under the live code as before (nothing here writes)."""
    d = Path(directory)
    out: dict[str, Any] = {"code": str(live), "resolved": "live", "record_code": None}
    if not d.exists():
        return out
    seen = {str(rec["code_digest"]) for rec in (D.read_record(js) for js in sorted(d.glob("*.json")))
            if rec is not None and rec.get("code_digest")}
    if len(seen) != 1:
        return {**out, "record_code": sorted(seen), "note": ("the directory holds no record" if not seen else
                                                             "the directory mixes code digests; the live digest governs")}
    have = seen.pop()
    out["record_code"] = have
    if have == str(live):
        return out
    for cand in REG.registered_code_digests(stage, path):
        if cand["code_digest"] == have:
            return {**out, "code": have, "resolved": cand["matched_entry"],
                    "superseded_utc": cand.get("superseded_utc"),
                    "note": (f"the record set was written under the {cand['matched_entry']} entry of stage {stage!r} "
                             f"({have[:12]}...); a later edit to this stage's code validates or invalidates no record "
                             "written earlier (addendum 2 item 5)")}
    return {**out, "note": (f"the record set's code digest {have[:12]}... is no entry of stage {stage!r}, current or "
                            "superseded: the live digest governs and the set does not verify")}


def resume_status(pq: Path, js: Path, digest: str, steps: Sequence[str]) -> dict[str, Any]:
    rec = D.read_record(js)
    if rec is None or not pq.exists():
        return {"complete": False, "stale": False, "steps_done": []}
    if rec.get("digest") != digest:
        return {"complete": False, "stale": True, "steps_done": sorted(rec.get("steps") or {})}
    done = sorted(rec.get("steps") or {})
    return {"complete": all(s in done for s in steps), "stale": False, "steps_done": done}


def read_record_set(root: Path, expected: Mapping[str, Mapping[str, str]], *, steps: Sequence[str] = ("point",),
                    what: str = "") -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """The verified fold records of one H3 (model arm, transform, design, seed) directory, exactly as
    ``discovery.read_discovery_record_set`` verifies discovery records: ``expected`` maps every fittable fold id to
    ``{"digest", "fold_hash"}`` as the current code computes them; a foreign fold, a differing digest or fold hash or a
    prediction without a record raises :class:`discovery.StaleRecordError`; missing folds give ``(None, status)``.

    A record that LACKS one of the requested ``steps`` (a fold whose ``intervals`` step never ran, e.g. after the job
    stopped on an error) is NOT counted as found: the set reads ``incomplete`` and the fold is named in
    ``folds_missing_steps``.  Callers ask for the steps whose columns they read -- :class:`Frames` asks for
    :data:`STEPS` -- so a point-only record can never make a set read complete and turn an interval-based delta silently
    NOT_RUN through :func:`_has_intervals` (task X finding numbers VH-06)."""
    exp = {str(k): dict(v) for k, v in expected.items()}
    status: dict[str, Any] = {"what": what, "n_expected": len(exp), "steps_required": list(steps)}
    if not Path(root).exists():
        return None, {**status, "status": "missing", "n_found": 0}
    by_name = {D.safe_fold_name(k): k for k in exp}
    frames, found, partial = [], set(), {}
    for pq in sorted(Path(root).glob("*.parquet")):
        rec = D.read_record(pq.with_suffix(".json"))
        if rec is None:
            raise D.StaleRecordError(f"{pq}: prediction without a JSON record")
        fid = str(rec.get("fold_id"))
        if fid not in exp or by_name.get(pq.stem) != fid:
            raise D.StaleRecordError(f"{pq}: fold {fid!r} is not a fittable fold of the current H3 job ({what})")
        if rec.get("digest") != exp[fid]["digest"]:
            raise D.StaleRecordError(f"{pq}: record digest differs from the current code, fold file, plan state or WITH "
                                     "record (rerun the H3 job)")
        if "fold_hash" in exp[fid] and rec.get("fold_hash") != exp[fid]["fold_hash"]:
            raise D.StaleRecordError(f"{pq}: fold hash differs from the registered fold")
        missing = [s for s in steps if s not in (rec.get("steps") or {})]
        if missing:
            # not found, not an error: the fold has to be re-run for the missing step, and the set must read
            # ``incomplete`` rather than count a point-only record as complete (task X finding numbers VH-06)
            partial[fid] = {"steps_done": sorted(rec.get("steps") or {}), "steps_missing": missing,
                            "intervals_status": rec.get("intervals_status")}
            continue
        fr = pd.read_parquet(pq)
        if len(fr) and (fr["fold_id"].astype(str) != fid).any():
            raise D.StaleRecordError(f"{pq}: parquet rows of another fold")
        frames.append(fr)
        found.add(fid)
    status["n_found"] = len(found)
    if partial:
        status["folds_missing_steps"] = {k: partial[k] for k in sorted(partial)}
    if found != set(exp):
        return None, {**status, "status": "incomplete", "missing_folds": sorted(set(exp) - found)[:10]}
    out = pd.concat(frames, ignore_index=True) if frames else None
    if out is not None and (out["half"].astype(str) != D.SELECTION).any():
        raise AssertionError(f"{root}: a stored prediction of a non-selection-half row")
    return out, {**status, "status": "complete"}


# --------------------------------------------------------------------------------------------- #
# frozen runners: the transformed arms at the WITH run's selected hyperparameters
# --------------------------------------------------------------------------------------------- #

def _runner_module():
    """``scripts/g19_run_discovery.py`` as a module (its runners, fold preparation and digest; never re-run)."""
    name = "g19_run_discovery"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, paths.G19_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ladder_module():
    """``scripts/g19_run_ladder.py`` as a module (its fold paths, verified readers and ``LadderRunner``; never re-run)."""
    name = "g19_run_ladder"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, paths.G19_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def selected_hyperparameters(arm: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """The hyperparameters a discovery record selected (one layout per arm family), for a refit without re-tuning."""
    arm = D.ARM_ALIASES.get(arm, arm)
    ar = record.get("arm_record") or {}
    if arm in ("M1", "M2"):
        sel = ar["selected"]
        return {"family": "neural", "emb_dim": int(sel["emb_dim"]), "weight_decay": float(sel["weight_decay"]),
                "rank": int(sel["rank"]), "n_epochs": int(ar["n_epochs"]), "model_seed": int(ar["model_seed"]),
                "label": str(record.get("selected_config"))}
    if arm in ("B5", "FLAT_CAT"):
        fz = ar["frozen"]
        cfg = {k: v for k, v in dict(fz["config"]).items() if k != "label"}
        return {"family": "boosted", "config": cfg, "iterations": int(fz["iterations"]),
                "model_seed": int(record["model_seed"]), "label": str(record.get("selected_config"))}
    if arm in D.B6_ARMS:
        sel = ar["selection"]
        return {"family": "b6", "rank": int(sel["selected_rank"]), "lambda": float(sel["selected_lambda"]),
                "init_seed": None if sel.get("init_seed") is None else int(sel["init_seed"]),
                "model_seed": None if record.get("model_seed") is None else int(record["model_seed"]),
                "inner_fold_macro_mae": dict(sel.get("inner_fold_macro_mae") or {}), "label": str(record.get("selected_config"))}
    if arm in LADDER_ARMS:
        sel = dict(ar["selected"])
        if str(sel.get("step")) != arm:
            raise ValueError(f"the {arm} WITH record's selected configuration is labelled {sel.get('step')!r}")
        return {"family": "ladder", "config": sel, "n_epochs": int(ar["n_epochs"]), "model_seed": int(ar["model_seed"]),
                "member_seeds": None if ar.get("member_seeds") is None else [int(x) for x in ar["member_seeds"]],
                "inner_folds_used": [int(f) for f in (ar.get("inner_folds_used") or [])],
                "label": str(record.get("selected_config"))}
    if arm == "B8":
        return {"family": "b8", "model_seed": int(record["model_seed"]), "label": str(record.get("selected_config"))}
    if arm in DETERMINISTIC_ARMS:
        return {"family": "closed_form", "label": "closed_form"}
    raise ValueError(f"no hyperparameter layout for arm {arm!r}")


def _training_rows(fc: Any) -> pd.DataFrame:
    c = fc.corpus
    return c.frame.loc[c.table.index[fc.mask]]


def _differing_inner_folds_plan(fc: Any, used: Sequence[int] | None, *, tuned: bool = True) -> dict[str, Any] | None:
    """``None`` when the fold's available inner folds are the WITH record's ``inner_folds_used``; otherwise the
    calibration plan with no calibration fold and status :data:`NOT_CALIBRATED_DIFFER` (:data:`READINGS`
    ``calibration_guard``).  The discovery runners assert this and would crash the intervals step of a WITHOUT fold
    whose actinide-free mask changed the inner cells (task X finding VL2-05)."""
    rd = _runner_module()
    _, avail = rd._calibration_splits(fc)
    plan = D.calibration_folds(avail, fc.job.inner_mode, tuned=tuned)
    have = sorted(int(f) for f in (used or []))
    if not plan["tuning_folds"] or have == list(plan["tuning_folds"]):
        return None
    return dict(plan, calibration_folds=[], status=NOT_CALIBRATED_DIFFER, with_record_inner_folds=have,
                available_inner_folds=list(plan["tuning_folds"]))


class FrozenNeural:
    """M1 / M2 refitted at the WITH record's selected configuration, epoch count and model seed; calibration through
    the discovery runner's cross-fitted plan on the WITH tuning record."""

    has_interval_step = True

    def __init__(self, step: str):
        if step not in ("M1", "M2"):
            raise ValueError(step)
        self.step, self.arms = step, (step,)

    def point(self, fc: Any, with_record: Mapping[str, Any]) -> dict[str, Any]:
        from gen19ct.models import neural as NN

        rd = _runner_module()
        hp = selected_hyperparameters(self.step, with_record)
        cfg = NN.NeuralConfig(hp["emb_dim"], hp["weight_decay"], hp["rank"])
        c = fc.corpus
        rows = _training_rows(fc)
        t0 = time.perf_counter()
        arm = NN.FactorisedArm(cfg, n_epochs=hp["n_epochs"], model_seed=hp["model_seed"], rows=c.frame,
                               condition_vectors=c.cv).fit(rows, fc.ctx)
        pred = arm.predict(c.frame.loc[fc.sc_labels])
        secs = time.perf_counter() - t0
        cols = tuple(col for cs in arm.encoder.block_columns.values() for col in cs)
        D.assert_no_provenance_features(cols, f"{self.step} H3 inputs")
        rec = {"selected_hyperparameters": hp, "fit_record": arm.fit_record(), "refit": READINGS["tuning"]}
        return {self.step: rd.ArmOutput(pred=pred, selected_config=cfg.label(), model_seed=hp["model_seed"], record=rec,
                                        feature_columns=cols, seconds=secs)}

    def calibration(self, fc: Any, with_arm_record: Mapping[str, Any]):
        bad = _differing_inner_folds_plan(fc, with_arm_record.get("inner_folds_used"))
        if bad is not None:
            return None, bad
        return _runner_module().NeuralRunner(self.step).calibration(fc, with_arm_record)


class FrozenBoosted:
    """B5 / FLAT_CAT refitted at the WITH record's frozen CatBoost configuration and tree count (section 15 model seed of
    the fold); cross-fitted calibration through the discovery runner on the WITH tuning record."""

    has_interval_step = True

    def __init__(self, name: str):
        self.name, self.arms = name, (name,)

    def point(self, fc: Any, with_record: Mapping[str, Any]) -> dict[str, Any]:
        from gen19ct.models import boosted as BO

        rd = _runner_module()
        hp = selected_hyperparameters(self.name, with_record)
        c = fc.corpus
        rows = _training_rows(fc)
        arm = BO.BoostedArm(self.name, fold_index=fc.ordinal, design=rd.inner_design_object(fc.job, c),
                            inner_mode=fc.job.inner_mode, config=BO.CatBoostConfig(**hp["config"]),
                            iterations=hp["iterations"], frame=c.frame, cv=c.cv,
                            max_cells_per_batch=rd.max_cells_per_batch(fc.job))
        t0 = time.perf_counter()
        arm.fit(rows, fc.ctx)
        pred = arm.predict(c.frame.loc[fc.sc_labels])
        secs = time.perf_counter() - t0
        D.assert_no_provenance_features(arm.fitted.columns, f"{self.name} H3 features")
        rec = {"selected_hyperparameters": hp, "frozen": {"config": arm.fitted.config.record(),
                                                          "iterations": arm.fitted.iterations}, "refit": READINGS["tuning"]}
        return {self.name: rd.ArmOutput(pred=pred, selected_config=arm.fitted.config.label, model_seed=arm.model_seed,
                                        record=rec, feature_columns=tuple(arm.fitted.columns), seconds=secs)}

    def calibration(self, fc: Any, with_arm_record: Mapping[str, Any]):
        used = ((with_arm_record.get("arm") or {}).get("tuning") or {}).get("inner_folds_used")
        bad = _differing_inner_folds_plan(fc, used)
        if bad is not None:
            return None, bad
        return _runner_module().BoostedRunner(self.name).calibration(fc, with_arm_record)


def b6_cross_fit_configs(inner_fold_macro_mae: Mapping[str, Mapping[str, float]], *, variant: str = "B6"
                         ) -> dict[int, tuple[int, float]]:
    """Per inner fold ``j``: the (rank, lambda) selected on the OTHER inner folds of the WITH record's per-fold inner
    macro MAE (``factorized.select_config`` with the registered tie margin) -- the cross-fitted calibration reading."""
    from gen19ct.models import factorized as FZ
    from gen19ct.models import inner_design as ID

    table: dict[tuple[int, float], dict[int, float]] = {}
    for label, per in inner_fold_macro_mae.items():
        k, lam = label[1:].split("_lam", 1)
        table[(int(k), float(lam))] = {int(f): float(v) for f, v in per.items()}
    # the grid the RECORD holds, restricted to the variant's registered ranks (B6r0 is rank 0 only): selecting over the
    # module defaults would ask select_config for configurations the record never scored
    ranks = tuple(sorted({k for k, _ in table} & set(FZ.VARIANT_RANKS[variant])))
    lambdas = tuple(sorted({lam for _, lam in table}))
    if not ranks or not lambdas:
        raise ValueError(f"b6_cross_fit_configs: the record holds no {variant} configuration "
                         f"(ranks {sorted({k for k, _ in table})}, lambdas {sorted(lambdas)})")
    missing = [(k, lam) for k in ranks for lam in lambdas if (k, lam) not in table]
    if missing:
        raise ValueError(f"b6_cross_fit_configs: the record's per-fold inner MAE lacks configuration(s) {missing}")
    folds = sorted({f for per in table.values() for f in per})
    out = {}
    for j, others in ID.cross_fit_plan(folds).items():
        maes = {cfg: float(np.mean([per[f] for f in others if f in per])) if others else float("nan")
                for cfg, per in table.items()}
        out[j] = FZ.select_config(maes, ranks, lambdas, FZ.TIE_MARGIN)
    return out


class FrozenB6:
    """B6 at the WITH record's selected (rank, lambda) and initialisation seed; cross-fitted calibration with the
    configuration selected on the other inner folds of the WITH tuning (:func:`b6_cross_fit_configs`)."""

    has_interval_step = True

    def __init__(self, variant: str = "B6"):
        if variant not in D.B6_ARMS:
            raise ValueError(variant)
        self.variant, self.arms = variant, (variant,)

    def point(self, fc: Any, with_record: Mapping[str, Any]) -> dict[str, Any]:
        from gen19ct.models import factorized as FZ

        rd = _runner_module()
        hp = selected_hyperparameters(self.variant, with_record)
        c = fc.corpus
        FZ.register_table(c.table, c.frame)
        seed = hp["init_seed"] if hp["init_seed"] is not None else hp["model_seed"]
        t0 = time.perf_counter()
        arm = FZ.B6Factorized(hp["rank"], hp["lambda"], name=self.variant, seed=seed).fit_table(c.table, fc.mask, fc.ctx)
        pred = arm.predict_positions(fc.positions)
        secs = time.perf_counter() - t0
        d = arm.design
        cols = tuple(d.x_names) + tuple(d.zm_names) + tuple(d.zs_names)
        D.assert_no_provenance_features(cols, f"{self.variant} H3 inputs")
        rec = {"selected_hyperparameters": hp, "fit_info": arm.info, "refit": READINGS["tuning"]}
        return {self.variant: rd.ArmOutput(pred=pred, selected_config=f"k{hp['rank']}_lam{hp['lambda']:g}", model_seed=seed,
                                           record=rec, feature_columns=cols, seconds=secs)}

    def calibration(self, fc: Any, with_arm_record: Mapping[str, Any]):
        from gen19ct.models import factorized as FZ

        rd = _runner_module()
        sel = with_arm_record["selection"]
        seed = sel.get("init_seed")
        if seed is None:
            seed = fc.model_seed
        configs = b6_cross_fit_configs(sel["inner_fold_macro_mae"], variant=self.variant)
        splitter = rd.B6Runner().splitter(fc)
        avail = sorted({int(s.fold) for s in splitter.splits(fc.corpus.table, fc.mask, fc.ctx)})
        plan = D.calibration_folds(avail, fc.job.inner_mode, tuned=True)
        if not plan["calibration_folds"] or any(j not in configs for j in plan["calibration_folds"]):
            plan = dict(plan, calibration_folds=[], status="not_calibrated_inner_folds_differ_from_the_with_record")
            return None, plan
        arms = {j: FZ.B6Factorized(configs[j][0], configs[j][1], name=self.variant, seed=int(seed))
                for j in plan["calibration_folds"]}
        sels = {j: {"config": f"k{configs[j][0]}_lam{configs[j][1]:g}", "excluded_inner_folds": [j]} for j in arms}
        return D.CrossFitResidualConformal(arms, D.InnerFoldSubset(splitter, plan["calibration_folds"]),
                                           guard=fc.guard_mode, selections=sels), plan


class FrozenComparator:
    """A closed-form arm (B3i, B0, ...): no hyperparameters; ``ComparatorIntervalsRunner`` gives point and intervals."""

    has_interval_step = False

    def __init__(self, name: str):
        self.name, self.arms = name, (name,)

    def point(self, fc: Any, with_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return _runner_module().ComparatorIntervalsRunner(self.name).point(fc)


class FrozenLadder:
    """A deployed ladder step M3-M7 refitted at its ``evaluation/ladder`` WITH record's selected ``LadderConfig``, epoch
    count and model seed (M7: the member seeds and the publication-group bootstrap of the transformed training rows);
    calibration through the ladder runner's cross-fitted scheme at the WITH record's cross-fit selections (M7:
    normalised split conformal) -- :data:`READINGS` ``ladder_arm_refit`` (task X finding V-01)."""

    has_interval_step = True

    def __init__(self, step: str, *, n_members: int | None = None, min_expert_rows: int | None = None):
        if step not in LADDER_ARMS:
            raise ValueError(f"{step!r} is not a ladder step {LADDER_ARMS}")
        self.step, self.arms = step, (step,)
        self.n_members, self.min_expert_rows = n_members, min_expert_rows

    def _kw(self) -> dict[str, Any]:
        from gen19ct.models import ladder as LAD

        return {"min_expert_rows": LAD.MIN_EXPERT_ROWS if self.min_expert_rows is None else int(self.min_expert_rows)}

    def point(self, fc: Any, with_record: Mapping[str, Any]) -> dict[str, Any]:
        from gen19ct.models import ladder as LAD

        rd = _runner_module()
        hp = selected_hyperparameters(self.step, with_record)
        cfg = LAD.LadderConfig.from_record(hp["config"])
        c = fc.corpus
        rows = _training_rows(fc)
        t0 = time.perf_counter()
        if self.step == "M7":
            n_members = LAD.N_MEMBERS if self.n_members is None else int(self.n_members)
            arm = LAD.EnsembleArm(cfg, n_epochs=hp["n_epochs"], fold_index=fc.ordinal, rows=c.frame, condition_vectors=c.cv,
                                  systems=c.systems, n_members=n_members, **self._kw()).fit(rows, fc.ctx)
            if hp["member_seeds"] is not None and arm.seeds != list(hp["member_seeds"])[:n_members]:
                raise AssertionError(f"{self.step}: member seeds {arm.seeds} differ from the WITH record's {hp['member_seeds']}")
            seed = int(arm.seeds[0])
            enc = arm.members[0].encoder
        else:
            arm = LAD.LadderArm(cfg, n_epochs=hp["n_epochs"], model_seed=hp["model_seed"], rows=c.frame, condition_vectors=c.cv,
                                systems=c.systems, **self._kw()).fit(rows, fc.ctx)
            seed = int(arm.model_seed)
            enc = arm.encoder
        pred = arm.predict(c.frame.loc[fc.sc_labels])
        secs = time.perf_counter() - t0
        cols = tuple(col for cs in enc.neural.block_columns.values() for col in cs)
        D.assert_no_provenance_features(cols, f"{self.step} H3 inputs")
        rec = {"selected_hyperparameters": hp, "fit_record": arm.fit_record(), "refit": READINGS["tuning"],
               "ladder_refit": READINGS["ladder_arm_refit"]}
        return {self.step: rd.ArmOutput(pred=pred, selected_config=cfg.label(), model_seed=seed, record=rec,
                                        feature_columns=cols, seconds=secs)}

    def calibration(self, fc: Any, with_arm_record: Mapping[str, Any]):
        RL = _ladder_module()
        tuned = self.step != "M7"
        bad = _differing_inner_folds_plan(fc, with_arm_record.get("inner_folds_used"), tuned=tuned)
        if bad is not None:
            return None, bad
        kw: dict[str, Any] = dict(self._kw())
        if self.n_members is not None:
            kw["n_members"] = int(self.n_members)
        # the runner's calibration reads only the record (selections, epochs, seeds): the ladder state and code digests
        # it carries are irrelevant to it, so a neutral state is passed
        runner = RL.LadderRunner(self.step, RL.LadderState(stop_rule=False, discovery_ladder={}),
                                 {"own": "", "discovery": "", "combined": ""}, **kw)
        return runner.calibration(fc, with_arm_record)


class FrozenSharedOnly:
    """Addendum 2 item 4: "an H3-side subclass of ``neural.FactorisedNet`` with the ``e_series`` and ``e_ox`` offsets
    held at zero" -- the WITH arm re-run with the shared-only metal embedding, on the :class:`FrozenNeural` pattern: the
    WITH record's SELECTED hyperparameters, stopping count and model seed, the WITH training rows unchanged, the
    cross-fitted calibration at the WITH record's selections.  M1 / M2 fit ``models.shared_only.SharedOnlyArm`` directly
    and calibrate with the discovery runner's cross-fitted plan whose inner arms are converted to shared-only twins; a
    ladder step M3-M7 refits through :class:`FrozenLadder` inside ``shared_only.shared_only_ladder_training`` (the
    ``LadderNet`` built by name is the shared-only subclass), which :meth:`training_context` also supplies for the
    calibration's fits.  Every fit asserts the offsets zero afterwards; a missing WITH record is refused."""

    has_interval_step = True
    transform = SHARED_ONLY

    def __init__(self, arm: str, **ladder_kw: Any):
        arm = D.ARM_ALIASES.get(arm, arm)
        if not shared_only_applicable(arm):
            raise ValueError(f"the shared-only embedding re-run applies to {EMBEDDING_ARMS}, not {arm!r} (no metal embedding)")
        self.step, self.arms = arm, (arm,)
        self.is_ladder = arm in LADDER_ARMS
        self.inner = FrozenLadder(arm, **ladder_kw) if self.is_ladder else FrozenNeural(arm)

    def training_context(self):
        """The context every fit of this arm runs in (the H3 runner wraps the calibration's ``fit_table`` with it)."""
        import contextlib

        from gen19ct.models import shared_only as SO

        return SO.shared_only_ladder_training() if self.is_ladder else contextlib.nullcontext()

    def point(self, fc: Any, with_record: Mapping[str, Any] | None) -> dict[str, Any]:
        from gen19ct.models import shared_only as SO

        if not with_record:
            raise ValueError(f"refused: the shared-only re-run of {self.step} needs the WITH record (its SELECTED "
                             "hyperparameters, stopping count and model seed); no record, no refit")
        if self.is_ladder:
            with self.training_context() as made:
                out = self.inner.point(fc, with_record)
            if not made:
                # the by-name patch saw no LadderNet built: the fit did not go through the shared-only subclass, so the
                # record would be labelled shared-only with nothing checked (n_networks 0) -- refuse, never label
                raise AssertionError(f"{self.step} shared-only re-run: the ladder fit built no network inside "
                                     "shared_only_ladder_training (models.ladder no longer builds LadderNet by name?); "
                                     "no offset was checked, so the record is not written")
            checks = [SO.assert_offsets_zero(n) for n in made]
            o = out[self.step]
            o.record.update({"shared_only": {"label": SO.LABEL, "reading": SO.READING, "n_networks": len(made),
                                             "offset_checks": checks}, "refit": READINGS["tuning"]})
            return {self.step: rd_replace(o, selected_config=f"{o.selected_config}; shared-only embedding")}
        from gen19ct.models import neural as NN

        rd = _runner_module()
        hp = selected_hyperparameters(self.step, with_record)
        cfg = NN.NeuralConfig(hp["emb_dim"], hp["weight_decay"], hp["rank"])
        c = fc.corpus
        rows = _training_rows(fc)
        t0 = time.perf_counter()
        arm = SO.SharedOnlyArm(cfg, n_epochs=hp["n_epochs"], model_seed=hp["model_seed"], rows=c.frame,
                               condition_vectors=c.cv).fit(rows, fc.ctx)
        check = SO.assert_offsets_zero(arm.result.net)
        pred = arm.predict(c.frame.loc[fc.sc_labels])
        secs = time.perf_counter() - t0
        cols = tuple(col for cs in arm.encoder.block_columns.values() for col in cs)
        D.assert_no_provenance_features(cols, f"{self.step} H3 shared-only inputs")
        rec = {"selected_hyperparameters": hp, "fit_record": arm.fit_record(), "refit": READINGS["tuning"],
               "shared_only": {"label": SO.LABEL, "reading": SO.READING, "offset_check": check}}
        return {self.step: rd.ArmOutput(pred=pred, selected_config=f"{cfg.label()}; shared-only embedding",
                                        model_seed=hp["model_seed"], record=rec, feature_columns=cols, seconds=secs)}

    def calibration(self, fc: Any, with_arm_record: Mapping[str, Any]):
        cal, plan = self.inner.calibration(fc, with_arm_record)
        if cal is None:
            return None, plan
        if not self.is_ladder:
            from gen19ct.models import shared_only as SO

            cal.arms_by_fold = {j: SO.SharedOnlyArm.from_arm(a) for j, a in cal.arms_by_fold.items()}
        return cal, dict(plan, shared_only=True)


def rd_replace(o: Any, **changes: Any) -> Any:
    return replace(o, **changes)


def shared_only_runner(arm: str, **ladder_kw: Any) -> FrozenSharedOnly:
    return FrozenSharedOnly(arm, **ladder_kw)


def frozen_runner(arm: str):
    arm = D.ARM_ALIASES.get(arm, arm)
    if arm in ("M1", "M2"):
        return FrozenNeural(arm)
    if arm in LADDER_ARMS:
        return FrozenLadder(arm)
    if arm in ("B5", "FLAT_CAT"):
        return FrozenBoosted(arm)
    if arm in D.B6_ARMS:
        return FrozenB6(arm)
    if arm in DETERMINISTIC_ARMS:
        return FrozenComparator(arm)
    raise ValueError(f"no frozen runner for {arm!r} (B8 and the H5 toggles are not H3 arms)")


# --------------------------------------------------------------------------------------------- #
# the transformed training set of one fold
# --------------------------------------------------------------------------------------------- #

def transformed_frame(frame: pd.DataFrame, train_index: pd.Index, transform: str, *, seed: int) -> pd.DataFrame:
    """The corpus frame with ``discovery.h3_training_rows`` applied to the fold's TRAINING rows only (a permutation or
    shuffle never touches a hidden row): ACT_PERMUTED moves ``log_D`` among actinide training rows, ACT_METAL_SHUFFLED
    the (state, element) labels.  WITH / WITHOUT / SHARED_ONLY return the frame unchanged (WITHOUT acts on the mask;
    the shared-only re-run changes the network, not a training row)."""
    if transform in ("WITH", "WITHOUT", SHARED_ONLY):
        return frame
    train = frame.loc[train_index]
    new = D.h3_training_rows(train, transform, seed=int(seed))
    if not new.index.equals(train.index):
        raise AssertionError("a training transform changed the training rows")
    out = frame.copy()
    cols = [I.TARGET_COL] if transform == "ACT_PERMUTED" else [SG.METAL_COL, SG.ELEMENT_COL]
    for col in cols:
        vals = out[col].to_numpy(dtype=object if col != I.TARGET_COL else float).copy()
        pos = out.index.get_indexer(new.index)
        vals[pos] = new[col].to_numpy(dtype=object if col != I.TARGET_COL else float)
        out[col] = vals
    return out


def without_mask(frame: pd.DataFrame, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """The WITHOUT training mask: the fold's mask minus every actinide row (unknown-state actinide rows included)."""
    an = D.actinide_rows(frame)
    out = np.asarray(mask, dtype=bool) & ~an
    return out, int((np.asarray(mask, dtype=bool) & an).sum())


# --------------------------------------------------------------------------------------------- #
# completion gates
# --------------------------------------------------------------------------------------------- #

def discovery_complete_cheap(out_root: Path) -> dict[str, Any]:
    """Item 1 of :func:`discovery_complete` alone (``wall_clock.json`` reached the final plan stage): a file read, so a
    runner can refuse BEFORE it loads the corpus or the feasibility frame (task X finding VL2-04)."""
    root = D.discovery_root(out_root)
    wc = D.read_record(root / "decisions" / "wall_clock.json") or {}
    inv = wc.get("invocations") or []
    stages_done: set[str] = set()
    for i in inv:
        stages_done |= set(i.get("stages_done") or [])
    reached_end = D.STAGES["not_implemented"] in stages_done
    return {"complete": bool(reached_end), "wall_clock_json_present": bool(inv), "reached_final_stage": reached_end,
            "final_stage": D.STAGES["not_implemented"], "stages_done": sorted(stages_done),
            "definition": "evaluation/discovery/decisions/wall_clock.json records an invocation whose stages_done reached "
                          f"{D.STAGES['not_implemented']} (the record verification follows only when this holds)"}


def refuse_unless_cheap_complete(out_root: Path) -> dict[str, Any]:
    """``SystemExit`` unless :func:`discovery_complete_cheap` holds -- the first gate every post-discovery runner runs,
    before any heavy load."""
    cheap = discovery_complete_cheap(out_root)
    if not cheap["complete"]:
        raise SystemExit("refused: the discovery run is not COMPLETE -- evaluation/discovery/decisions/wall_clock.json "
                         f"records no invocation reaching stage {cheap['final_stage']} (stages done: {cheap['stages_done']}); "
                         "let scripts/g19_run_discovery.py finish, then rerun (nothing heavy was loaded)")
    return cheap


def discovery_complete(out_root: Path, *, code: str, state: D.PlanState, excluded_ids: Iterable[str],
                       folds_dir: Path | None = None, runners: Mapping[str, Any] | None = None,
                       jobs: Sequence[D.JobSpec] | None = None) -> dict[str, Any]:
    """Whether the discovery run is COMPLETE, defined as all of:

    1. ``evaluation/discovery/decisions/wall_clock.json`` records an invocation whose ``stages_done`` reached the last
       stage of the plan (``discovery.STAGES['not_implemented']``, the ``M3+`` marker ``run_plan`` stops at) and every
       stage of the current plan is in the union of the recorded ``stages_done``;
    2. every ``fit`` job of ``discovery.enumerate_plan(state)`` (the plan under the final plan state, freezing-candidate
       runs included) has, for every arm it writes, a COMPLETE verified record set
       (``g19_run_discovery.verified_predictions``: digest, fold hash and fold set exactly what the code the record set's
       stage is registered under -- ``registry.record_dir_code``: the ``discovery`` or ``discovery_candidates`` entry,
       else ``code`` -- the fold files and the plan state produce; a stale record raises).

    ``jobs`` overrides the enumerated plan (tests)."""
    rd = _runner_module()
    root = D.discovery_root(out_root)
    wc = D.read_record(root / "decisions" / "wall_clock.json") or {}
    inv = wc.get("invocations") or []
    stages_done: set[str] = set()
    for i in inv:
        stages_done |= set(i.get("stages_done") or [])
    plan = list(jobs) if jobs is not None else D.enumerate_plan(state, folds_dir=folds_dir)
    stages = sorted({j.stage for j in plan if j.stage})
    missing_stages = sorted(set(stages) - stages_done)
    reached_end = D.STAGES["not_implemented"] in stages_done
    incomplete, complete = [], []
    cache: dict = {}
    for j in plan:
        if j.kind != "fit":
            continue
        for arm in j.writes:
            # the code digest of the record set's OWN stage (a freezing-candidate set is verified against the
            # discovery_candidates entry, never the discovery one); ``code`` is the no-registry fallback
            jcode = REG.record_dir_code(out_root, arm, j.design_dir, int(j.seed), default=code)
            _, st = rd.verified_predictions(out_root, arm, j.design_dir, int(j.seed), code=jcode, state=state,
                                            excluded_ids=list(excluded_ids), folds_dir=folds_dir, runners=runners,
                                            fold_cache=cache)
            (complete if st.get("status") == "complete" else incomplete).append(
                {"job": j.key, "arm": arm, **{k: v for k, v in st.items() if k in ("status", "n_expected", "n_found",
                                                                                  "missing_folds")}})
    ok = bool(reached_end and not missing_stages and not incomplete)
    return {"complete": ok, "wall_clock_json_present": bool(inv), "reached_final_stage": reached_end,
            "final_stage": D.STAGES["not_implemented"], "stages_done": sorted(stages_done), "missing_stages": missing_stages,
            "n_fit_record_sets": len(complete) + len(incomplete), "n_complete": len(complete),
            "incomplete_record_sets": incomplete[:20], "n_incomplete": len(incomplete),
            "definition": discovery_complete.__doc__.split("``jobs``")[0].strip()}


SCORER_FILES: tuple[str, ...] = ("decisions/decisions.json", "decisions/stop_rule.json", "contrasts_registered.csv",
                                 "r19_items.csv")


def scorer_decisions_present(out_root: Path) -> dict[str, Any]:
    """The scorer's decision and contrast files exist under ``evaluation/discovery`` and ``decisions.json`` is newer than
    every fold record (the scorer ran after the last fit)."""
    root = D.discovery_root(out_root)
    present = {f: (root / f).exists() for f in SCORER_FILES}
    dec = root / "decisions" / "decisions.json"
    newest = 0.0
    for js in root.rglob("*.json"):
        if js.parent.name.startswith("s") and js.parent.name[1:].isdigit():
            newest = max(newest, js.stat().st_mtime)
    scored_after = bool(dec.exists() and dec.stat().st_mtime >= newest)
    return {"present": present, "all_present": all(present.values()), "decisions_newer_than_every_record": scored_after,
            "ok": all(present.values()) and scored_after}


# --------------------------------------------------------------------------------------------- #
# the D03 decision file (brief section 29 format), generated from files
# --------------------------------------------------------------------------------------------- #

def _fmt(v: Any, fmt: str = "{:.3f}") -> str:
    if v is None:
        return NOT_COMPUTED
    if isinstance(v, float) and not np.isfinite(v):
        return "NaN"
    try:
        return fmt.format(v)
    except (TypeError, ValueError):
        return str(v)


def _addendum4_lines(summary: Mapping[str, Any]) -> list[str]:
    """The D03 paragraph of POST-HOC addendum 4: the dropped control, the 20 h cap and every ``NOT_RUN`` design."""
    expl = summary.get("exploratory") or {}
    bud = summary.get("budget") or {}
    nrun = summary.get("designs_not_run") or {}
    lines = ["POST-HOC addendum 4:", "",
             f"- item 1 -- ACT_METAL_SHUFFLED is **not run**; the {expl.get('n_records', 0)} fold(s) already fitted are "
             f"`{expl.get('status', EXPLORATORY_NOT_SCORED)}` and enter no contrast, table or verdict. {ACT_METAL_SHUFFLED_NOT_RUN}.",
             f"- item 2 -- compute cap: {_fmt(bud.get('used_hours'), '{:.2f}')} h of "
             f"{bud.get('budget_hours', H3_BUDGET_HOURS)} h wall clock used on at most {bud.get('workers_max', H3_WORKERS)} "
             f"workers (`evaluation/h3/decisions/wall_clock.json`); priority order "
             f"{' -> '.join(bud.get('priority_order') or DESIGN_PRIORITY)}; exhausted = {bud.get('exhausted')}."]
    if nrun:
        for d, rec in sorted(nrun.items()):
            r = rec if isinstance(rec, Mapping) else {}
            lines.append(f"- design **{d}: {NOT_RUN}** -- {r.get('reason', NOT_COMPUTED)}. It contributes "
                         f"{r.get('contributes', NOT_RUN_CONTRIBUTES)}; `designs_scored` does not name it.")
            if r.get("complete_legs"):
                lines.append(f"  - complete legs kept on disk but **not scored**: {', '.join(r['complete_legs'])} "
                             f"-- {r.get('complete_legs_reading', '')}")
            if r.get("deviation"):
                lines.append(f"  - **DEVIATION (not the cap):** {r['deviation']}")
            for e in r.get("blocking_errors") or []:
                lines.append(f"  - blocking error: `{e.get('note')}` (invocation {e.get('started_utc')} -> "
                             f"{e.get('ended_utc')}; `evaluation/h3/decisions/wall_clock.json -> invocations[*].note`)")
    else:
        lines.append(f"- every design in {', '.join(DESIGN_PRIORITY)} has at least one scored leg (no {NOT_RUN} design).")
    crun = summary.get("contrasts_not_run") or {}
    lines.append("")
    lines.append(f"- **the unit of completeness (TASK F item 1).** {summary.get('contrast_unit_rule', CONTRAST_UNIT_RULE)}")
    scored_sets = summary.get("contrast_record_sets_scored") or []
    lines.append(f"  - contrast record sets SCORED ({len(scored_sets)}): {', '.join(scored_sets) or 'none'}.")
    if crun:
        lines.append(f"  - contrast record sets NOT scored ({len(crun)}):")
        for k, rec in sorted(crun.items()):
            r = rec if isinstance(rec, Mapping) else {}
            lines.append(f"    - **{k}: {r.get('status', INCOMPLETE_GUARD_FAILURE)}** -- {r.get('reason', NOT_COMPUTED)}. "
                         f"It contributes {r.get('contributes', CONTRAST_NOT_RUN_CONTRIBUTES)}. "
                         f"{r.get('vocabulary', '')}")
            if r.get("deviation"):
                lines.append(f"      - **DEVIATION (not the cap):** {r['deviation']}")
            for e in r.get("blocking_errors") or []:
                lines.append(f"      - blocking error: `{e.get('note')}` (invocation {e.get('started_utc')} -> "
                             f"{e.get('ended_utc')}; `evaluation/h3/decisions/wall_clock.json -> invocations[*].note`)")
        lines.append(f"  - **why the guard fails, measured.** {GUARD_FAILURE_DIAGNOSIS}")
        lines.append(f"  - **the rule now registered.** {GUARD_VALUE_SOURCE_RULE}")
        lines.append("  - **the addendum sentence as proposed before it was written** (POST-HOC addendum 5 item 2 "
                     "registers it): \"For a control arm whose training `log_D` is permuted by construction "
                     "(`ACT_PERMUTED`, section 11 Controls), the near-duplicate VALUE comparison of the section 2 "
                     "fold-isolation guard reads the REGISTERED `log_D` of the corpus, not the permuted values: "
                     "`near_dup_value_tol` = 0.005 asks whether two rows record the same experiment at the same value, "
                     "which is a property of the recorded data and not of an arm's training target, and a permuted value "
                     "that happens to land within 0.005 of another row's value is a coincidence the data never had. "
                     "Every other level of the guard -- the publication group, the archive duplicate group and the "
                     "near-duplicate key at 6 significant figures -- is value-independent and is unchanged, so the "
                     "guard stays exactly as strict as registered and becomes invariant under the permutation.\"")
    else:
        lines.append("  - every planned contrast record set is complete and scored.")
    ref = summary.get("refits") or {}
    if ref:
        counts = ref.get("matched_entry_counts") or {}
        lines.append(f"- item 3 -- {len(ref.get('verifying') or [])} fold record(s) verify against the registry entry they "
                     f"were written under (`matched_entry`: "
                     + (", ".join(f"{k} = {v}" for k, v in sorted(counts.items())) if counts else NOT_COMPUTED)
                     + f"); {len(ref.get('stale') or [])} scored-transform record(s) still carry a "
                     "superseded `h3` digest written after the supersession (these are deleted and refitted before any "
                     f"delta is scored, and must be 0 here); {len(ref.get('stale_exploratory') or [])} exploratory "
                     f"record(s) are `{EXPLORATORY_NOT_SCORED}` and are neither refitted nor deleted.")
        lines.append(f"  - **what \"verifies\" means here.** {ref.get('operative_reading', OPERATIVE_DIGEST_READING)}")
        if ref.get("why_no_record_is_current"):
            lines.append(f"  - **none verifies as `current`, and none is refitted for that reason.** "
                         f"{ref['why_no_record_is_current']}")
        src = ref.get("written_utc_source_counts") or {}
        if src:
            lines.append("  - timestamp the ordering check used, per record: "
                         + ", ".join(f"`{k}` = {v}" for k, v in sorted(src.items()))
                         + " (a record without `written_utc` is ordered by the derived "
                           "`max(steps.*.date_utc)`; both are recorded per record in "
                           "`evaluation/h3/decisions/refits.json` and `h3_summary.json -> refits`).")
    lines.append("")
    return lines


#: POST-HOC addendum 2, reading ``needs_power``: "a contrast needs the check when its family is primary (H1), H1b, S1(b)
#: or H3 and its full R19 verdict is not PASS".  Every H3 contrast of this run satisfies it and none was checked, so the
#: debt is INVENTORIED rather than left as an absence (task X finding protocol VH-09)
NEEDS_POWER_RULE = ("POST-HOC addendum 2 (reading 'needs_power'): a contrast needs the section 8 signal-injection check "
                    "when its family is primary (H1), H1b, S1(b) or H3 and its full R19 verdict is not PASS. An H3 "
                    "contrast that owes the check and did not get one is reported '" + NO_POWER_CHECK_LABEL + "' "
                    "(POST-HOC addendum 4 item 4, third clause), never a null and never 'no effect'; the debt is listed "
                    "here so it is a RECORDED debt rather than an absence")


def h3_per_fold_seconds(out_root: Path) -> dict[str, float]:
    """Per-fold seconds for pricing a refit, from ``evaluation/h3/decisions/cost_estimate.json``: the MEASURED H3 mean
    per transform, and the discovery per-fold seconds per ``<arm>@<design_dir>`` as the fallback upper bound."""
    body = D.read_record(h3_decisions_dir(out_root) / "cost_estimate.json") or {}
    out: dict[str, float] = {}
    for t, m in (body.get("measured_per_fold_seconds") or {}).items():
        if isinstance(m, Mapping) and m.get("mean") is not None:
            out[str(t)] = float(m["mean"])
    for k, v in (body.get("discovery_per_fold_seconds") or {}).items():
        arm, _, ddir = str(k).partition("@")
        for design, (d, _v) in DESIGNS.items():
            if str(ddir).startswith(d):
                out.setdefault(f"{arm}@{design}", float(v))
    for r in body.get("rows") or []:
        if isinstance(r, Mapping) and r.get("key") and r.get("per_fold_seconds") is not None:
            out[str(r["key"])] = float(r["per_fold_seconds"])
    return out


def power_root_checks_path(out_root: Path) -> Path:
    """``evaluation/power/power_checks.json`` (named here because ``power`` imports this module, not the other way)."""
    return Path(out_root) / "evaluation" / "power" / "power_checks.json"


def power_checked_keys(out_root: Path) -> list[str]:
    """The contrast keys a section 8 power run has actually checked (``evaluation/power/power_checks.json ->
    checks[*].{key, contrast}``); empty when no power run has written the file."""
    body = D.read_record(power_root_checks_path(out_root)) or {}
    out: set[str] = set()
    for c in body.get("checks") or []:
        for k in ("key", "contrast"):
            if isinstance(c, Mapping) and c.get(k):
                out.add(str(c[k]))
    return sorted(out)


#: section 8's injection grid: "y' = y + kappa.s, kappa in {0.1, 0.25, 0.5, 1.0} log D"
POWER_KAPPA_GRID: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
POWER_COST_BASIS = (
    "section 8 re-runs the pipeline on injected targets at kappa in {0.1, 0.25, 0.5, 1.0}, refitting EVERY arm of the "
    "contrast at its selected hyperparameters without re-tuning, on the same scored rows. The cost of one H3 contrast's "
    "check is therefore 4 kappa x (folds of the WITH leg + folds of the transform leg) refits, priced at the per-fold "
    "seconds this design's H3 refits actually took (evaluation/h3/decisions/cost_estimate.json); it is a SERIAL "
    "estimate, halved only if the folds split evenly over the 2 registered workers. It is priced PER CONTRAST, so the "
    "total over contrasts double-counts a WITH leg two contrasts share (the same injected WITH refits serve WITH vs "
    "WITHOUT and WITH vs PERMUTED, and each direction of a pair is one contrast row): the total is an upper bound, and "
    "the per-arm figure is 4 kappa x folds x arms of the design")


def power_check_cost(model_arm: str, transform: str, design: str, *, record_sets: Mapping[str, Any] | None = None,
                     per_fold_seconds: Mapping[str, float] | None = None,
                     kappas: Sequence[float] = POWER_KAPPA_GRID) -> dict[str, Any]:
    """What the section 8 injection check of ONE H3 contrast would cost (:data:`POWER_COST_BASIS`)."""
    def n_folds(t: str) -> int | None:
        for v in dict(record_sets or {}).values():
            if isinstance(v, Mapping) and str(v.get("what") or "") == contrast_key(model_arm, t, design):
                return None if v.get("n_expected") is None else int(v["n_expected"])
        # the WITH legs are keyed without a ``what`` (they come from the discovery reader): <t>/<arm>/<design_dir>/s<seed>,
        # and the design_dir begins with the design token
        for k, v in dict(record_sets or {}).items():
            parts = str(k).split("/")
            if (isinstance(v, Mapping) and len(parts) >= 3 and parts[0] == str(t) and parts[1] == str(model_arm)
                    and parts[2].startswith(f"{design}__") and v.get("n_expected") is not None):
                return int(v["n_expected"])
        return None
    nw, nt = n_folds("WITH"), n_folds(transform)
    secs = None
    for key in (f"{model_arm}:{transform}@{design}", f"{model_arm}@{design}", str(transform), str(model_arm)):
        if per_fold_seconds and key in per_fold_seconds:
            secs = float(per_fold_seconds[key])
            break
    n_fits = None if nw is None or nt is None else int(len(list(kappas)) * (nw + nt))
    total = None if n_fits is None or secs is None else float(n_fits) * secs
    return {"n_kappa": len(list(kappas)), "kappas": list(kappas), "n_folds_with": nw, "n_folds_transform": nt,
            "n_refits": n_fits, "per_fold_seconds": secs,
            "serial_seconds": None if total is None else round(total, 1),
            "serial_hours": None if total is None else round(total / 3600.0, 3),
            "basis": POWER_COST_BASIS}


def needs_power_inventory(contrasts: pd.DataFrame | None, *, checked: Iterable[str] = (),
                          family: str = FAMILY, record_sets: Mapping[str, Any] | None = None,
                          per_fold_seconds: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Which H3 contrasts owe the section 8 power check, and which of them have one (:data:`NEEDS_POWER_RULE`).

    ``checked`` are the contrast keys a power run actually checked (``power_checks.json -> checks[*].contrast`` / ``key``).
    Every primary-cluster row of the family whose ``r19_verdict_full`` is not PASS is listed with its verdict, whether a
    check was run and why not."""
    done = {str(c) for c in checked}
    rows: list[dict[str, Any]] = []
    if contrasts is not None and not getattr(contrasts, "empty", True):
        fr = contrasts
        if "primary_cluster_unit" in fr.columns:
            fr = fr[fr["primary_cluster_unit"].astype(bool)]
        if "family" in fr.columns:
            fr = fr[fr["family"].astype(str) == str(family)]
        for _, r in fr.iterrows():
            full = str(r.get("r19_verdict_full") or NOT_COMPUTED)
            key = str(r.get("key") or r.get("contrast") or "")
            run = key in done or str(r.get("contrast") or "") in done
            cost = power_check_cost(str(r.get("model_arm") or ""), str(r.get("transform") or ""),
                                    str(r.get("design") or ""), record_sets=record_sets,
                                    per_fold_seconds=per_fold_seconds)
            rows.append({"key": key, "contrast": r.get("contrast"), "design": r.get("design"),
                         "family": r.get("family"), "r19_verdict_full": full,
                         "needs_power_check": full != "PASS", "check_run": bool(run),
                         "status": "CHECKED" if run else (NOT_RUN if full != "PASS" else "not owed"),
                         "cost": cost, "cost_hours_serial": cost.get("serial_hours"), "n_refits": cost.get("n_refits"),
                         "reason": ("checked" if run else
                                    "full R19 verdict is PASS, so no check is owed" if full == "PASS" else
                                    "OWED AND NOT RUN: no section 8 injection check of this contrast exists; the verdict "
                                    f"is reported '{NO_POWER_CHECK_LABEL}'")})
    owed = [r for r in rows if r["needs_power_check"]]
    unpaid = [r for r in owed if not r["check_run"]]
    hrs = [r["cost_hours_serial"] for r in unpaid if r.get("cost_hours_serial") is not None]
    return {"rule": NEEDS_POWER_RULE, "family": str(family), "n_contrasts": len(rows), "n_owed": len(owed),
            "n_owed_and_run": sum(1 for r in owed if r["check_run"]),
            "n_owed_and_not_run": len(unpaid),
            "cost_basis": POWER_COST_BASIS,
            "n_refits_owed_and_not_run": sum(r["n_refits"] for r in unpaid if r.get("n_refits") is not None) or None,
            "cost_hours_serial_owed_and_not_run": round(sum(hrs), 3) if hrs else None,
            "cost_priced_for": len(hrs), "cost_unpriced": len(unpaid) - len(hrs),
            "label_when_not_run": NO_POWER_CHECK_LABEL, "checked_keys": sorted(done), "contrasts": rows}


def item_status_map(r19_items: pd.DataFrame | None, item: int = 1) -> dict[str, str]:
    """``contrast key -> the status of R19 item ``item``` from the R19 item table, for statements that must agree with the
    item they talk about (task X finding protocol VH-04)."""
    if r19_items is None or getattr(r19_items, "empty", True) or "item" not in getattr(r19_items, "columns", ()):
        return {}
    sub = r19_items[pd.to_numeric(r19_items["item"], errors="coerce") == int(item)]
    key = "key" if "key" in sub.columns else "contrast"
    return {str(k): str(s) for k, s in zip(sub[key], sub["status"])}


def d03_markdown(summary: Mapping[str, Any], contrasts: pd.DataFrame | None, deltas: pd.DataFrame | None,
                 negative: Mapping[str, pd.DataFrame] | None = None,
                 r19_items: pd.DataFrame | None = None) -> str:
    """``decisions/D03_actinide_transfer.md`` (brief section 29: question, evidence, metrics, verdict, decision, next
    action) from the H3 outputs; every absent quantity prints ``not computed``.

    ``r19_items`` is the R19 item table: the power sentence beside each contrast is gated on that contrast's OWN item-1
    status, so it can never tell the reader that a contrast which PASSES item 1 "can never pass item 1"
    (task X finding protocol VH-04)."""
    dep = summary.get("deployed") or {}
    ver = summary.get("verdicts") or {}
    # task X finding V-P02: the verdicts are keyed by the arm the RECORDS use (M0's records are written under B5,
    # discovery.ARM_ALIASES), so a lookup takes ``arm_alias``; ``arm`` stays the name every printed sentence uses
    dep_key = dep.get("arm_alias") or dep.get("arm") or ""
    f4 = summary.get("f4") or {}
    lines = ["# D03 -- Does actinide extraction data improve hidden-lanthanide prediction? (H3, section 11)", "",
             f"*Generated by `scripts/g19_run_h3.py` at git HEAD `{summary.get('git_head') or NOT_COMPUTED}` from the files "
             "under `evaluation/h3/`. Every number is the selection half, discovery seed 104729, optimistically biased; "
             "R19 item 4 is NOT_EVALUATED in discovery (addendum 1 item 3), so verdicts are read on the "
             f"`{VERDICT_SCOPE}` scope and the full R19 verdict is printed beside. Readings: see `evaluation/h3/h3_summary.json` -> "
             "`readings`.*", "",
             "## Question", "",
             "Same architecture, same folds, same seeds, same lanthanide test set: does training WITH actinide rows beat "
             "training WITHOUT them (and the ACT_PERMUTED control) on hidden Ln(III) cells (V5), "
             "unseen Ln(III) states (V2) and unseen publications (V1)? Is the transfer negative (F4)?", "",
             "## Evidence", "",
             "| file | contents |", "|---|---|",
             "| `evaluation/h3/h3_contrasts.csv` | R19 rows (one per cluster unit) of every WITH vs transform and transform vs WITH contrast |",
             "| `evaluation/h3/h3_r19_items.csv` | R19 items 1-6 per contrast |",
             "| `evaluation/h3/h3_deltas.csv` | Delta rank accuracy and Delta calibration with intervals; Delta logSF MAE "
             "and CRPS appear only as their status rows (both absent here -- see below for why) |",
             "| `evaluation/h3/h3_per_unit_deltas.csv` | per Ln unit MAE of both arms |",
             "| `evaluation/h3/h3_actinide_dependent_cells.csv` | the eligibility of every scored V5 cell with and "
             "without actinide rows (the section 11 stratum) |",
             "| `evaluation/h3/negative_transfer/*.csv` | section 11 negative-transfer tables (descriptive) |",
             "| `evaluation/h3/h3_summary.json`, `h3_verdicts.json`, `h3_f4.json` | verdicts, F4, deployed rule, readings |",
             "| `evaluation/h3/records/<arm>/<transform>/...` | per-fold predictions and records of the refitted arms |", "",
             "## Metrics", "",
             f"Deployed configuration: **{dep.get('arm') or NOT_COMPUTED}** ({dep.get('basis') or NOT_COMPUTED}).",
             f"Model arms: {', '.join(summary.get('model_arms') or []) or NOT_COMPUTED}. Transforms scored: "
             f"{', '.join(summary.get('transforms') or []) or NOT_COMPUTED}.", ""]
    lines += _addendum4_lines(summary)
    if contrasts is not None and not contrasts.empty and "primary_cluster_unit" in contrasts.columns:
        prim = contrasts[contrasts["primary_cluster_unit"].astype(bool)]
        lines += ["| model arm | contrast | design | Delta MAE | margin | pct 95 % | BCa 95 % | p | scope verdict | full R19 | TOST |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in prim.sort_values(["model_arm", "design", "contrast"]).iterrows():
            lines.append(f"| {r.get('model_arm')} | {r['contrast']} | {r['design']} | {_fmt(r['point'])} | {_fmt(r.get('margin'))} | "
                         f"[{_fmt(r['percentile_low'])}, {_fmt(r['percentile_high'])}] | [{_fmt(r['bca_low'])}, {_fmt(r['bca_high'])}] | "
                         f"{_fmt(r['p_two_sided'], '{:.4f}')} | {r.get(f'verdict_{VERDICT_SCOPE}', NOT_COMPUTED)} | "
                         f"{r.get('r19_verdict_full', NOT_COMPUTED)} | {r.get('tost_verdict_eps0.05', NOT_COMPUTED)} |")
        item1 = item_status_map(r19_items, 1)
        lines += ["", "Power of each contrast at its own margin (`mde_80`, the 80 %-power detectable effect of the same "
                      "paired cluster bootstrap): a contrast that FAILS item 1 while its `mde_80` EXCEEDS its margin "
                      "could not have reached item 1 whatever the truth, so that FAIL is underpowered, not a "
                      "demonstration of no effect (task X finding V-L1). The sentence is read off each contrast's OWN "
                      "item-1 status (`h3_r19_items.csv`), never off `mde_80` alone: a contrast that PASSES item 1 has "
                      "reached it, and `mde_80` above the margin then says only that a FAIL would have been "
                      "uninformative (task X finding protocol VH-04).", "",
                  "| model arm | contrast | design | cluster unit | Delta MAE | margin | mde_80 | LOCO min | item 1 | reading |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in prim.sort_values(["model_arm", "design", "contrast"]).iterrows():
            m, mde = r.get("margin"), r.get("mde_80")
            ok = None if m is None or mde is None or not (np.isfinite(float(m)) and np.isfinite(float(mde))) \
                else float(mde) <= float(m)
            st = item1.get(str(r.get("key") or ""), item1.get(f"{r['contrast']}@{r['design']}", NOT_COMPUTED))
            read = (NOT_COMPUTED if ok is None else
                    "powered at this margin" if ok else
                    "**item 1 PASS; the margin is below `mde_80`, so a FAIL here would have been uninformative**"
                    if st == "PASS" else
                    "**UNDERPOWERED at this margin: item 1 FAILs and no kappa of the design could have made it PASS**"
                    if st == "FAIL" else
                    f"margin below `mde_80`; item 1 is {st}, so no underpowered-FAIL statement is made")
            lines.append(f"| {r.get('model_arm')} | {r['contrast']} | {r['design']} | {r.get('cluster_unit')} | "
                         f"{_fmt(r['point'])} | {_fmt(m)} | {_fmt(mde)} | {_fmt(r.get('loco_min'))} | {st} | {read} |")
        # R19 item 6's accounting: every registered sensitivity of the design is either decided or declared not run.
        # The two V5 refits addendum 1 item 4 runs for H1 / H1b / H4 / freezing candidates but not for an H3 transform
        # arm (strict_setting, HNO3_only_cells) are declared here by name (task X finding protocol VH-03).
        if "sensitivities_not_run" in prim.columns:
            it6 = item_status_map(r19_items, 6)
            lines += ["", "R19 item 6 accounting -- which registered sensitivities of the design DECIDED item 6 and "
                          "which were not run (an H3 contrast pairs WITH against a section 11 transform refit, and no "
                          "strict-setting or HNO3-only refit of a transform arm is registered or fitted anywhere, so "
                          "those two V5 sensitivities are UNTESTABLE for every H3 contrast). `strict_setting` and "
                          "`HNO3_only_cells` are members of POST-HOC addendum 1 item 4's OWN reduced set, so an H3 V5 "
                          "contrast decides item 6 on a set SMALLER than the registered reduced set and its item 6 is "
                          "reported **NOT_EVALUATED**, never PASS (TASK F item 4):", "",
                      "| contrast | design | item 6 | sensitivity set | not run (declared) |", "|---|---|---|---|---|"]
            seen: set[str] = set()
            for _, r in prim.sort_values(["model_arm", "design", "contrast"]).iterrows():
                k = f"{r['contrast']}@{r['design']}"
                if k in seen:
                    continue
                seen.add(k)
                lines.append(f"| {r['contrast']} | {r['design']} | "
                             f"{it6.get(str(r.get('key') or ''), it6.get(k, NOT_COMPUTED))} | "
                             f"{r.get('sensitivity_set', NOT_COMPUTED)} | {r.get('sensitivities_not_run') or 'none'} |")
            lines.append("")
    else:
        lines.append(f"Contrasts: {NOT_COMPUTED}.")
    lines.append("")
    if deltas is not None and not deltas.empty:
        lines += ["Other deltas (primary cluster unit; positive favours WITH):", "",
                  "| model arm | transform | design | metric | WITH | transform | Delta | pct 95 % | status |",
                  "|---|---|---|---|---|---|---|---|---|"]
        sub = deltas[deltas.get("primary_cluster_unit", pd.Series(True, index=deltas.index)).fillna(True).astype(bool)]
        for _, r in sub.sort_values(["model_arm", "design", "transform", "metric"]).iterrows():
            lines.append(f"| {r['model_arm']} | {r['transform']} | {r['design']} | {r['metric']} | {_fmt(r.get('with_value'))} | "
                         f"{_fmt(r.get('transform_value'))} | {_fmt(r.get('point'))} | [{_fmt(r.get('percentile_low'))}, "
                         f"{_fmt(r.get('percentile_high'))}] | {r.get('status', NOT_COMPUTED)} |")
    else:
        lines.append(f"Delta rank accuracy / calibration / logSF: {NOT_COMPUTED}.")
    # section 11's fourth delta: named with its reason whenever it is absent, never silently missing from the table
    nc = summary.get("not_computed") or {}
    lines += ["", f"Delta logSF MAE (`{PAIR_DESIGN}` Ln-Ln pairs): "
                  + ("computed above." if nc.get("logsf_mae_delta") in (None, "") else
                     f"**NOT_DEFINED.** {nc['logsf_mae_delta']}"),
              "", f"CRPS: **NOT_RUN.** {nc.get('crps', READINGS['crps'])}"]
    lines += ["", "## Verdict (null / supported / ambiguous)", ""]
    if ver:
        for arm, v in sorted(ver.items()):
            lines.append(f"- **{arm}: {v.get('verdict', NOT_COMPUTED)}** -- WITH beats WITHOUT on V5: {v.get('v5_with_beats_without')}; "
                         f"beats ACT_PERMUTED: {v.get('v5_with_beats_permuted')}; V1/V2 non-inferior: {v.get('v1_v2_non_inferior')}; "
                         f"TOST WITH-WITHOUT (V5): {(v.get('tost_v5_with_minus_without') or {}).get('verdict', NOT_COMPUTED)}; "
                         f"kappa_min: {v.get('kappa_min') if v.get('kappa_min') is not None else v.get('kappa_status', NOT_COMPUTED)}"
                         + (" -- UNDECIDED (underpowered)" if v.get("underpowered") else
                            # POST-HOC addendum 4 item 4: an UNDECIDED contrast with no registered power check is
                            # reported UNDECIDED (no registered power check), never a null and never "no effect"
                            f" -- **{NO_POWER_CHECK_LABEL}**" if v.get("verdict") == "UNDECIDED"
                            and v.get("kappa_min") is None else ""))
        mapping = {"helps": "supported (actinide rows help lanthanide prediction)", "hurts": "supported (negative transfer)",
                   "equivalent": "null (no difference inside +-0.05)", "UNDECIDED": "ambiguous"}
        dv = ver.get(dep_key, {}).get("verdict")
        lines += ["", f"Deployed arm reading ({dep.get('arm') or NOT_COMPUTED}"
                      + (f", records under {dep_key}" if dep_key and dep_key != dep.get("arm") else "")
                      + f"): **{mapping.get(dv, NOT_COMPUTED)}**."]
    else:
        lines.append(f"Verdicts: {NOT_COMPUTED}.")
    head = ("HOLDS -- gen19 is UNSUCCESSFUL on F4" if f4.get("failure") else
            ("does not hold" if f4.get("status") == "computed" else NOT_COMPUTED))
    lines += ["", f"**F4 (negative actinide transfer): {head}.** This is the headline and it is the CONSERVATIVE "
                  "reading -- F4 holds when ANY registered design shows the WITHOUT arm beating the deployed one under "
                  "EITHER registered 95 % interval (percentile or BCa). Deployed arm "
                  f"{f4.get('deployed_arm', NOT_COMPUTED)}; per design (either interval): "
              + ", ".join(f"{d}: {'beats' if (p.get('without_beats_with_interval_excludes_0') or p.get('bca_beats')) else ('no' if p.get('status') == 'computed' else NOT_COMPUTED)}"
                          for d, p in sorted((f4.get("per_design") or {}).items())) + ".",
              "", f"- percentile-only reading: **{'HOLDS' if f4.get('failure_percentile') else 'does not hold'}** ("
              + ", ".join(f"{d}: {'beats' if p.get('without_beats_with_interval_excludes_0') else ('no' if p.get('status') == 'computed' else NOT_COMPUTED)}"
                          for d, p in sorted((f4.get("per_design") or {}).items())) + ").",
              f"- BCa-only reading: **{'HOLDS' if f4.get('failure_bca') else 'does not hold'}** ("
              + ", ".join(f"{d}: {'beats' if p.get('bca_beats') else ('no' if p.get('status') == 'computed' else NOT_COMPUTED)}"
                          for d, p in sorted((f4.get("per_design") or {}).items())) + ")."
              + (f" The two readings DISAGREE on: {', '.join(f4.get('designs_where_readings_disagree') or [])}."
                 if f4.get("designs_where_readings_disagree") else " The two readings agree on every computed design."),
              "", f"*Which interval F4 reads is REGISTERED (POST-HOC addendum 5 item 3).* "
                  f"{f4.get('interval_reading_not_registered', F4_INTERVAL_READING)}",
              "", "**The two readings of F4, side by side (POST-HOC addendum 5 item 3).** "
                  f"Against a WITH deployment F4 **{'HOLDS' if (f4.get('both_readings') or {}).get('against_a_with_deployment') else 'does not hold'}**; "
                  "against the deployment section 11 actually registers -- the WITHOUT-actinide fit, because actinide "
                  "rows enter the deployed Ln configuration only on a *helps* verdict -- F4 **does not hold**, because "
                  "its first clause is false of the configuration deployed. "
                  f"{(f4.get('both_readings') or {}).get('why', F4_DEPLOYED_CONFIGURATION_READING)}",
              ""]
    per_f4 = f4.get("per_design") or {}
    if per_f4:
        lines += ["| design | reversed Delta (WITHOUT - WITH) | percentile 95 % | BCa 95 % | percentile reading | BCa reading |",
                  "|---|---|---|---|---|---|"]
        for d, p in sorted(per_f4.items()):
            if p.get("status") != "computed":
                lines.append(f"| {d} | {NOT_COMPUTED} | - | - | - | {p.get('reason', NOT_COMPUTED)} |")
                continue
            pc, bc = p.get("percentile_95") or [None, None], p.get("bca_95") or [None, None]
            lines.append(f"| {d} | {_fmt(p.get('point'))} | [{_fmt(pc[0])}, {_fmt(pc[1])}] | [{_fmt(bc[0])}, {_fmt(bc[1])}] | "
                         f"{'F4 holds' if p.get('without_beats_with_interval_excludes_0') else 'no'} | "
                         f"{'F4 holds' if p.get('bca_beats') else 'no'} |")
        lines.append("")
    strat = summary.get("actinide_dependent_stratum") or {}
    if strat.get("status") == "computed":
        lines += ["**Section 11's separate stratum beside the V5 headline.** The registered headline Delta is the macro "
                  "over every scored V5 Ln cell; section 11 makes the cells whose eligibility depends on actinide "
                  "partners a separate REPORTING unit, so the same macro without them is printed beside it (the "
                  "registered number is unchanged):", "",
                  "| quantity | all cells | excluding the actinide-partner-eligibility stratum |", "|---|---|---|",
                  f"| cells | {strat.get('n_cells_all')} | {strat.get('n_cells_excluding_stratum')} |",
                  f"| macro Delta MAE | {_fmt(strat.get('macro_delta_all_cells'), '{:.6f}')} | "
                  f"{_fmt(strat.get('macro_delta_excluding_stratum'), '{:.6f}')} |",
                  f"| median Delta MAE | {_fmt(strat.get('median_delta_all_cells'), '{:.6f}')} | "
                  f"{_fmt(strat.get('median_delta_excluding_stratum'), '{:.6f}')} |",
                  f"| share Delta > 0 | {_fmt(strat.get('share_delta_positive_all_cells'), '{:.3f}')} | "
                  f"{_fmt(strat.get('share_delta_positive_excluding_stratum'), '{:.3f}')} |", "",
                  f"The {strat.get('n_cells_actinide_dependent')} stratum cell(s) have macro Delta "
                  f"{_fmt(strat.get('macro_delta_actinide_dependent'), '{:.6f}')} and contribute "
                  f"{_fmt(strat.get('stratum_contribution_to_macro'), '{:.6f}')} of the headline "
                  f"{_fmt(strat.get('macro_delta_all_cells'), '{:.6f}')}, i.e. "
                  f"{_fmt(None if strat.get('stratum_share_of_macro') is None else 100.0 * strat['stratum_share_of_macro'], '{:.1f}')} %. "
                  "They are by construction the cells that stop being eligible once actinide rows are removed, so the "
                  "headline must not be read as a broad effect (task X finding protocol VH-08).", ""]
    if negative:
        # section 11's condition, evaluated in ONE place on ONE reading and recorded beside the tables
        # (h3.negative_transfer_condition; task X finding protocol VH-05).  The per-design point estimates are printed
        # beside it as context, never as a second trigger.
        cond = summary.get("negative_transfer_condition") or {}
        rev = pd.DataFrame() if contrasts is None or contrasts.empty else contrasts[
            (contrasts.get("model_arm") == dep_key) & (contrasts.get("transform") == "WITHOUT")
            & contrasts.get("primary_cluster_unit", pd.Series(False, index=contrasts.index)).astype(bool)
            & contrasts["contrast"].astype(str).str.startswith(f"{dep_key}:WITHOUT vs")]
        trig = {str(r["design"]): float(r["point"]) for _, r in rev.iterrows()} if not rev.empty else {}
        lines += ["Negative-transfer investigation (section 11, \"run if WITHOUT is better, point estimate or passed\"): "
                  + (f"**triggered** ({cond.get('reason', NOT_COMPUTED)})" if cond.get("met") else
                     f"**NOT triggered** -- {cond.get('reason', NOT_COMPUTED)}; the tables below are "
                     f"**{cond.get('tables_label', NEGATIVE_TRANSFER_EXPLORATORY)}** and no reading is taken from them")
                  + f". Condition read on **{cond.get('contrast_read', NOT_COMPUTED)}** "
                    f"(design {cond.get('design_read', NOT_COMPUTED)}): passed R19 = "
                    f"{cond.get('without_passed_r19_v5')}, point estimate MAE(WITH) - MAE(WITHOUT) = "
                    f"{_fmt(cond.get('v5_without_vs_with_point'), '{:.6f}')}, favours WITHOUT = "
                    f"{cond.get('point_estimate_favours_without')}.",
                  "", f"*The design and contrast the trigger is read on are a reading, not registered text.* "
                      f"{cond.get('condition', NEGATIVE_TRANSFER_CONDITION)}",
                  "", "Reversed point estimates of every SCORED design of the deployed arm, printed as context only "
                      "(a NOT_RUN design contributes none): "
                  + ("; ".join(f"{d}: {_fmt(p, '{:.6f}')}" for d, p in sorted(trig.items())) if trig else NOT_COMPUTED)
                  + ". R19-passing negative transfer is F4 above; the tables here are descriptive.", ""]
        strata = negative.get("strata")
        if strata is not None and not strata.empty:
            lines += ["Negative-transfer strata (V5 per-cell Delta MAE = MAE(WITHOUT) - MAE(WITH); descriptive):", "",
                      "| stratum | in stratum | cells | systems | mean Delta | median Delta | share Delta > 0 |", "|---|---|---|---|---|---|---|"]
            for _, r in strata.iterrows():
                lines.append(f"| {r['stratum']} | {r['in_stratum']} | {r['n_cells']} | {r['n_systems']} | {_fmt(r['mean_delta_mae'])} | "
                             f"{_fmt(r['median_delta_mae'])} | {_fmt(r['share_delta_positive'], '{:.2f}')} |")
            lines.append("")
        # section 11's remaining negative-transfer inputs: the per-family and per-mechanism deltas, and the per-cell
        # delta against the system's actinide row count and its median |An/Ln logSF|
        for key, title in (("per_family", "Per family"), ("per_mechanism", "Per mechanism")):
            tab = negative.get(key)
            if tab is not None and not tab.empty:
                lines += [f"{title} (same per-cell Delta MAE; descriptive):", "",
                          "| group | cells | systems | mean Delta | median Delta | share Delta > 0 | mean MAE WITH | mean MAE WITHOUT |",
                          "|---|---|---|---|---|---|---|---|"]
                for _, r in tab.iterrows():
                    lines.append(f"| {r['group']} | {r['n_cells']} | {r['n_systems']} | {_fmt(r['mean_delta_mae'])} | "
                                 f"{_fmt(r['median_delta_mae'])} | {_fmt(r['share_delta_positive'], '{:.2f}')} | "
                                 f"{_fmt(r['mean_mae_with'])} | {_fmt(r['mean_mae_transform'])} |")
                lines.append("")
        cov = negative.get("per_cell_covariates")
        if cov is not None and not cov.empty:
            lines += ["Per-cell Delta MAE against the section 11 covariates (Spearman, system-cluster percentile "
                      "interval; read only after the section 8 reliability report):", "",
                      "| covariate | cells | systems | Spearman | pct 95 % | status |", "|---|---|---|---|---|---|"]
            for _, r in cov.iterrows():
                lines.append(f"| {r['covariate']} | {r['n_cells']} | {r['n_systems']} | {_fmt(r['spearman'])} | "
                             f"[{_fmt(r['percentile_low'])}, {_fmt(r['percentile_high'])}] | {r.get('status', NOT_COMPUTED)} |")
            lines.append("")
    else:
        lines += [f"Negative-transfer investigation: {NOT_COMPUTED} (run when WITHOUT is better, section 11).", ""]
    lines += shared_only_lines(summary.get("shared_only"))
    debt = summary.get("power_debt") or {}
    if debt:
        lines += ["## Section 8 power check owed (a recorded debt, not an absence)", "",
                  f"{debt.get('rule', NEEDS_POWER_RULE)}", "",
                  f"- H3 contrasts (primary cluster): **{debt.get('n_contrasts')}**; owe the check: "
                  f"**{debt.get('n_owed')}**; checked: **{debt.get('n_owed_and_run')}**; "
                  f"**owed and NOT run: {debt.get('n_owed_and_not_run')}**.",
                  f"- every H3 verdict quoted anywhere therefore carries `{debt.get('label_when_not_run', NO_POWER_CHECK_LABEL)}` "
                  "(POST-HOC addendum 4 item 4).",
                  f"- what the unpaid debt COSTS: **{debt.get('n_refits_owed_and_not_run', NOT_COMPUTED)}** injected "
                  f"refits, **{_fmt(debt.get('cost_hours_serial_owed_and_not_run'), '{:.1f}')} h** of serial compute "
                  f"({debt.get('cost_priced_for', 0)} of {debt.get('n_owed_and_not_run', 0)} contrasts priced; "
                  f"{debt.get('cost_unpriced', 0)} unpriced). {debt.get('cost_basis', POWER_COST_BASIS)}", "",
                  "| contrast | design | full R19 | power check | refits | serial h | reason |",
                  "|---|---|---|---|---|---|---|"]
        for r in debt.get("contrasts") or []:
            lines.append(f"| {r.get('contrast')} | {r.get('design')} | {r.get('r19_verdict_full')} | "
                         f"**{r.get('status', NOT_RUN)}** | {r.get('n_refits', NOT_COMPUTED)} | "
                         f"{_fmt(r.get('cost_hours_serial'), '{:.2f}')} | {r.get('reason')} |")
        lines.append("")
    lines += ["## Decision", "",
              (f"Actinide rows enter the deployed Ln configuration: **{'yes' if ver.get(dep_key, {}).get('verdict') == 'helps' else 'no'}** "
               "(section 11: only on *helps*)."), "",
              "## Next action", "",
              "- confirmation: the frozen H3 claims (if any) on the withheld seeds and the V6 deltas (section 15);",
              "- a POST-HOC addendum for every reading in `h3_summary.json -> readings` before a result is quoted as registered. "
              "TWO of the five this run first recorded as REQUESTED are now REGISTERED by POST-HOC addendum 5: the interval "
              "F4 reads (item 3, with the side-by-side deployment readings) and the near-duplicate guard of a "
              "value-permuted control arm (item 2, under which the V1 WITH - PERMUTED legs are completed). THREE remain "
              "REQUESTED: the design and contrast section 11's negative-transfer trigger is read on "
              "(`negative_transfer_condition.condition`), the operative reading of addendum 4 item 3 "
              "(`refits.operative_reading`) and the narrowing of addendum 4 item 2's NOT_RUN unit from the DESIGN to the "
              "CONTRAST RECORD SET (`contrast_unit_rule`);",
              f"- the section 8 power check (`scripts/g19_run_power.py`) for the {debt.get('n_owed_and_not_run', 0)} H3 "
              f"contrast(s) that owe it: {debt.get('n_refits_owed_and_not_run', NOT_COMPUTED)} injected refits, "
              f"{_fmt(debt.get('cost_hours_serial_owed_and_not_run'), '{:.1f}')} h serial (table above);",
              "- the WITH - PERMUTED legs of V1: registered to run in full, blocked until 2026-09-23 by a deterministic "
              "isolation-check failure (not by the 20 h cap) and now UNBLOCKED by POST-HOC addendum 5 item 2, which "
              "registers that the guard's near-duplicate VALUE comparison reads the corpus's RECORDED log D for an arm "
              "whose training target is permuted by construction. The folds are completed under that rule and the legs "
              "are scored; no fold was ever forced and no other level of the guard changed. "
              + GUARD_VALUE_SOURCE_RULE, ""]
    return "\n".join(lines)


def shared_only_lines(so: Mapping[str, Any] | None) -> list[str]:
    """The D03 paragraph of section 11's last bullet (addendum 2 item 4): the condition, whether it was met, and the
    re-run's contrasts when it ran -- otherwise exactly "not run (condition not met)" (or the named reason)."""
    lines = ["Shared-only-embedding re-run (section 11 last bullet; addendum 2 item 4 -- the WITH arm with `e_series` and "
             "`e_ox` held at zero against the full section 15 embedding):", ""]
    if not so:
        lines += [f"- {NOT_COMPUTED} (the H3 summary carries no `shared_only` block)", ""]
        return lines
    cond = so.get("condition") or {}
    lines.append(f"- condition: {cond.get('condition', SHARED_ONLY_CONDITION)}")
    lines.append(f"- read on the deployed arm `{cond.get('deployed_arm', NOT_COMPUTED)}`: WITHOUT passes R19 against WITH on V5 = "
                 f"{cond.get('without_passed_r19_v5')}; V5 point estimate MAE(WITH) - MAE(WITHOUT) = "
                 f"{_fmt(cond.get('v5_without_vs_with_point'))}; condition met = {cond.get('met')}")
    status = so.get("status") or cond.get("status") or SHARED_ONLY_NOT_RUN
    lines.append(f"- status: **{status}**")
    rows = so.get("contrasts")
    if isinstance(rows, list) and rows:
        lines += ["", "| model arm | contrast | design | Delta MAE (positive favours the full embedding) | pct 95 % | BCa 95 % | p | scope verdict | TOST |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r.get('model_arm')} | {r.get('contrast')} | {r.get('design')} | {_fmt(r.get('point'))} | "
                         f"[{_fmt(r.get('percentile_low'))}, {_fmt(r.get('percentile_high'))}] | [{_fmt(r.get('bca_low'))}, "
                         f"{_fmt(r.get('bca_high'))}] | {_fmt(r.get('p_two_sided'), '{:.4f}')} | "
                         f"{r.get(f'verdict_{VERDICT_SCOPE}', NOT_COMPUTED)} | {r.get('tost_verdict_eps0.05', NOT_COMPUTED)} |")
    lines.append("")
    return lines


def json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else (o.as_posix() if isinstance(o, Path) else str(o))))

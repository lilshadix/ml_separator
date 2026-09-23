"""``evaluation/confirmation.py`` -- the ONE registered confirmation run (pre-registration section 15).

Section 15 registers exactly one run: *"One run on the withheld seeds scores those claims on the **confirmation half**
of each design and runs **V6 once**"*, *"A claim is **confirmed** only if it passes R19 with 5 of 5 confirmation
seeds"*, *"Whatever confirmation returns is the result. No sixth claim is added and nothing is re-run"*.

POST-HOC addendum 5 item 1 makes that run the **CORE** run, compute-driven:

* the frozen claims of ``decisions/CONFIRMATION_PLAN.md`` (at most five) on the confirmation half with the 5 withheld
  seeds, **R19 item 4 exactly as registered (5 of 5 seeds)**;
* the single V6 run of section 3.4 with **S2(a), S2(b), S2(c)**;
* **S1(c)** under its addendum-2 confirmation rule and **S1(d)** calibration;
* it does **NOT** run the section 11 V6 actinide deltas nor any section 8 power check: both are reported ``NOT_RUN``
  with their cost and their consequence (:data:`V6_ACTINIDE_DELTAS_NOT_RUN`, :data:`POWER_CHECK_NOT_RUN`; a contrast
  with no power check is UNDECIDED, never a null -- addendum 4 item 4);
* **R19 item 6 refits run on seed 104729 only**, as in discovery (addendum 1 item 4).

The withheld seeds
------------------
Five seeds committed as ``sha256(canonical JSON{n, salt, schema, seeds})`` =
``65e8ae8c...f82`` (``manifests/confirmation_seeds_sha256.txt``), stored OUTSIDE the repository.  They enter this code
**only** through the ``--seed-store PATH`` argument of ``scripts/g19_run_confirmation.py``, at run time, and only as a
:class:`SeedStore` whose ``repr`` is redacted.  Nothing written by this module names a seed: a fold, a record, a table
and a log line carry an **opaque seed index 1..5** (:meth:`SeedStore.index_of`) and the commitment digest, and
``decisions/CONFIRMATION.md`` reveals the seeds only as the ``--verify-seeds`` verdict (:data:`SEED_DISCLOSURE`).
:func:`scan_for_seed_leak` re-reads every file the run wrote and fails the run if a seed's decimal form appears.

What this module is, and is not
-------------------------------
It holds the run's **rules**: the gates, the plan it reads, the job and fold inventory, the cost model, the
confirmation-half scoring population, the statistics of R19 item 4 / S1 / S2, the ``NOT_RUN`` records and the writers.
The **fitting** is ``scripts/g19_run_confirmation.py`` (the stage's runner), which owns the confirmation-half
``prepare_fold`` -- discovery's refuses a confirmation-half fold by construction
(``g19_run_discovery.prepare_fold``: *"a confirmation-half fold is never fitted in discovery"*) -- and reuses the
discovery runners, tuning and record schema unchanged.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gen19ct import paths
from gen19ct.evaluation import discovery as D
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import registry as REG
from gen19ct.evaluation import transfer as ET
from gen19ct.folds import io as FI
from gen19ct.manifest import write_json

SCHEMA = "gen19.confirmation_record.v1"
DECISION_SCHEMA = "gen19.confirmation_decisions.v1"
#: the registry stage every record of this run carries (``registry.STAGES``)
STAGE = "confirmation"
CONF_REL = "evaluation/confirmation"
PLAN_REL = "decisions/CONFIRMATION_PLAN.md"
REPORT_REL = "decisions/CONFIRMATION.md"
SEED_COMMITMENT_REL = "manifests/confirmation_seeds_sha256.txt"
#: section 15: five withheld seeds, and R19 item 4 needs 5 of 5 of them
N_SEEDS = 5
N_SEEDS_REQUIRED = ET.MIN_SEEDS_POSITIVE_CONFIRMATION
#: section 15: "at most 5 claims are frozen in decisions/CONFIRMATION_PLAN.md"
MAX_CLAIMS = 5
#: the discovery seed R19 item 6's refits run on, at confirmation as in discovery (addendum 1 item 4, addendum 5 item 1)
ITEM6_REFIT_SEED = 104729
#: section 9 gamma5 / eta5 / section 8 epsilon, and the S2(a) sign count of section 9
GAMMA5, ETA5 = 0.05, 0.02
EPSILON = ET.EPSILON
S2A_MIN_SIGN_SYSTEMS, S2A_N_SYSTEMS = 11, 13
S2B_MARGIN = 0.02
#: section 9 S1(d) / S2(c) coverage bands
S1D_BANDS: dict[str, tuple[float, float]] = {"50": (0.40, 0.60), "80": (0.70, 0.90), "95": (0.88, 0.99)}
S1D_CATEGORY_80 = (0.65, 0.92)
S1D_CATEGORY_MIN_CELLS = 20
S2C_BANDS: dict[str, tuple[float, float]] = {"80": (0.70, 0.90)}
S2C_MIN_95 = 0.88
NOT_RUN = "NOT_RUN"
NOT_EVALUATED = "NOT_EVALUATED"
UNDECIDED = "UNDECIDED"
INCOMPLETE_GUARD_FAILURE = "INCOMPLETE_GUARD_FAILURE"
#: POST-HOC addendum 6 item 3(a): the completeness unit is the CONTRAST RECORD SET, never the design
COMPLETENESS_UNIT = "contrast_record_set"
#: the two V6 metal states of section 3.4, hidden TOGETHER per system
V6_METALS: tuple[str, str] = ("Pr(III)", "Nd(III)")
#: the V6 design of section 3.4 as a fold stem: ``V6__prnd__exact`` (one fold per system, seed-independent design)
V6_DESIGN, V6_VARIANT_NAME, V6_SCHEME = "V6", "prnd", "exact"
V6_STEM = "V6__prnd__exact"
#: the 7-system sensitivity of section 3.4, scored in the same single run
V6_SENSITIVITY: tuple[int, int] = (10, 5)

SEED_DISCLOSURE = (
    "decisions/CONFIRMATION.md reveals the withheld seeds ONLY as the verdict of "
    "`scripts/g19_seal_prereg.py --verify-seeds --seed-store <path>` against the commitment of section 15 "
    "(manifests/confirmation_seeds_sha256.txt), which prints no seed. The VALUES are never written to the repository, "
    "never logged and never put in a fold id, a record, a table or a figure: per fold and per record they appear as an "
    "opaque seed index 1..5 (confirmation.SeedStore.index_of, the store's own order). Section 15's sentence 'They are "
    "revealed in decisions/CONFIRMATION.md together with the --verify-seeds verdict' is satisfied by the verdict: the "
    "commitment is what makes the seeds checkable, and printing the values would let a later run reproduce a "
    "confirmation fit -- POST-HOC addendum 5 item 1 keeps this run single. SUPERSEDED by POST-HOC addendum 6 item 4, "
    "which honours section 15 as written: see SEED_REVELATION")

#: POST-HOC addendum 6 item 4, which SUPERSEDES the narrowing SEED_DISCLOSURE requested an addendum for.  Section 15 is
#: honoured as written -- the seeds ARE revealed in ``decisions/CONFIRMATION.md`` -- and the addendum fixes WHEN: after
#: the single run has completed, never before or during it.
SEED_REVELATION = (
    "POST-HOC addendum 6 item 4: 'The five seeds and the --verify-seeds verdict are written into "
    "decisions/CONFIRMATION.md AFTER the single run has completed, never before; no artefact written before or during "
    "the run contains a seed value, and the run logs only the commitment digest and an opaque per-seed index.' So: "
    "every fold file, record, prediction frame, table and log line of the run carries i1..i5 and the commitment digest "
    "alone (confirmation.scrub, checked by scan_for_seed_leak over everything the run wrote); decisions/CONFIRMATION.md "
    "is written LAST, and it is the ONE artefact that names the values, beside the --verify-seeds verdict. The leak "
    "scan therefore runs twice: before CONFIRMATION.md exists it must find no seed anywhere, and after it is written "
    "that file must contain all five and every other file still none")

READINGS: dict[str, str] = {
    "seed_entry": "the 5 withheld seeds enter this code ONLY through --seed-store PATH at run time, verified against "
                  "the section 15 commitment by scripts/g19_seal_prereg.py --verify-seeds (which never prints them); "
                  "no default path exists, no environment variable is read and nothing is cached: a second run needs "
                  "the store again",
    "seed_disclosure": SEED_DISCLOSURE,
    "seed_revelation": SEED_REVELATION,
    "completeness_unit": "POST-HOC addendum 6 item 3: the completeness unit is the CONTRAST RECORD SET, not the design "
                         "-- a design may carry a verdict on one complete contrast while another contrast of the same "
                         "design is incomplete, reported with its own status and carrying no verdict, no per-unit row "
                         "and no failure-condition input; an incompletion caused by a GUARD failure rather than a "
                         "compute cap is INCOMPLETE_GUARD_FAILURE, so the cap's vocabulary is never used for it (this "
                         "run has no compute cap: addendum 5 item 1 fixed its scope instead)",
    "v6_folds_in_run": "POST-HOC addendum 6 item 2: the V6 folds are built ONCE, INSIDE this run, by the registered "
                       "section 3.4 rule (the 13 systems; per system the V5 state-level hiding applied to Pr(III) x S "
                       "and Nd(III) x S TOGETHER, support_graph.hide_cells(component_aware=True), which removes the "
                       "Pr(III)/Nd(III) rows and the Pr(?)/Nd(?) rows in S and in every system sharing a component with "
                       "S) under each withheld seed, with their fold and design hashes recorded in the confirmation "
                       "manifest. Nothing before this run may touch them, which is why the builder lives here and not "
                       "in scripts/g19_build_folds*.py",
    "v6_half": "V6 is the single run of section 3.4 over all 13 systems, not a confirmation-HALF design: section 3.1 "
               "carves V6_TARGET_ROWS out of every design in BOTH halves, so no V6 row contributed to any ladder "
               "decision, claim or preferred-model choice and the half filter that protects the other designs has "
               "nothing to protect here. The V6 folds therefore carry half 'NA' and their scored rows are selected by "
               "assert_v6_scope (every scored row a V6_TARGET_ROW), never by assert_confirmation_rows",
    "half": "every row this run scores is in the CONFIRMATION half of its design (feasibility_halves.csv; "
            "assert_confirmation_rows), the half section 15 says contributed to no ladder decision, claim or "
            "preferred-model choice. Discovery's own prepare_fold refuses a confirmation-half fold, so the runner owns "
            "its own; nothing here ever scores a selection-half row and nothing in discovery ever scores a "
            "confirmation-half row",
    "v6_scope": "section 3.1 carves V6_TARGET_ROWS out of every design, and section 3.4 runs V6 ONCE, here. So this "
                "run is the only place a V6_TARGET_ROWS row is scored, and only by a V6 job: assert_v6_scope refuses a "
                "V6 row in any other design's scored set and refuses a V6 job that scores a row outside V6_TARGET_ROWS",
    "item4": "R19 item 4 as registered at confirmation: Delta > 0 in 5 of 5 withheld seeds (section 8 item 4; "
             "addendum 3 item 1 restates it; addendum 5 item 1 keeps it exactly). A claim with fewer than 5 scored "
             "seeds is NOT confirmed and is reported with the seeds it has -- never rounded up",
    "item6_seed": "R19 item 6's strict and HNO3-only REFITS run on seed 104729 only, as in discovery (addendum 1 item "
                  "4, kept by addendum 5 item 1); the four scoring-filter sensitivities are re-scorings of the primary "
                  "fits and are evaluated on all 5 withheld seeds. The report prints which sensitivity was decided on "
                  "how many seeds, and the plan's 'conservative reading' (all 5 seeds) is NOT taken: addendum 5 item 1 "
                  "chose the core run",
    "s1c_seed_combination": "S1(c) at confirmation (section 9 S1(c) as redefined, addendum 2): each Delta_Y is the MEAN "
                            "over the 5 withheld seeds of the per-seed cell-pair-macro Delta_Y; its interval is the "
                            "percentile interval of a system-cluster bootstrap (10,000 resamples, seed 19) of that seed "
                            "mean with THE SAME resampled systems applied to every seed; and Delta_Y > 0 in 5 of 5 "
                            "seeds. Pass: min_Y Delta_Y >= gamma5 = 0.05 and every Delta_Y's interval excludes 0, "
                            "together with the selection-half counterweight min_Y Delta_Y >= -0.02 already satisfied",
    "s2a": "S2(a): the sign of the per-system median PREDICTED logSF_Nd/Pr equals the sign of the observed median in "
           ">= 11 of the 13 V6 systems (section 9, fixed pre-seal), and pair-level direction accuracy (|observed| >= "
           "0.1) beats every yardstick Y in {HEAVIER, B3x-derived, B3i-derived, B8} by >= 0.05 under the paired rule of "
           "S1(c), pooled AND in the HNO3 pairs. A system whose observed or predicted median is 0 or undefined counts "
           "as NOT agreeing (the conservative side of a sign count)",
    "s2c_interval_reading": "REGISTERED by POST-HOC addendum 7 item 1 (the resolution the earlier wording REQUESTED): "
                            "'the logSF interval is a split-conformal interval fitted directly on the PAIR residuals -- "
                            "the absolute deviation of the predicted logSF from the observed logSF over the comparable "
                            "pairs of the fold's inner calibration set, with the same finite-sample quantile section 12 "
                            "registers for rows. A fold whose inner calibration set holds fewer than 20 comparable "
                            "pairs falls back to the convolution of the two row intervals under independence "
                            "(half-widths added in quadrature), and every such fold is flagged and counted in the "
                            "coverage table. Both constructions are printed; the conformal-on-pairs one is the "
                            "registered value.' Section 9 S2(c) fixes the bands ([0.70, 0.90] at 80 %, >= 0.88 at 95 %) "
                            "and the population (the V6 pairs); the addendum fixes the construction, before any V6 "
                            "number exists. So the verdict is 'pair_conformal' -- per outer fold, the pair residuals of "
                            "that fold's inner calibration comparable pairs (section 2's key: publication group, system, "
                            "condition key, within one inner split), the quantile ceil((n + 1) * level)-th smallest -- "
                            "and 'quadrature' is printed beside it with readings_disagree saying whether the choice "
                            "mattered. The earlier 'interval_arithmetic' reading ([lo_a - hi_b, hi_a - lo_b]) is "
                            "withdrawn: the addendum's reason is that a difference of two row intervals is not itself a "
                            "calibrated interval",
    "s2c_calibration_population": "RECORDED CONSEQUENCE of the population POST-HOC addendum 7 item 1 registers, not a "
                                  "change to it. The addendum calibrates on 'the comparable pairs of the fold's inner "
                                  "calibration set': section 2's pair key is (publication group, system, condition "
                                  "key), so ANY two distinct metal states of one condition group are a calibration pair "
                                  "-- Am/Eu counts exactly as much as Nd/Pr. The pairs S2(c) SCORES are Nd/Pr only, and "
                                  "|logSF_Nd/Pr| is small (adjacent lanthanides). A calibration population dominated by "
                                  "wider-separated pairs therefore yields a pair quantile that is too LARGE and an "
                                  "interval biased WIDE, which pushes coverage UP: the registered [0.70, 0.90] band at "
                                  "80 % can FAIL on the HIGH side as an artefact of the population rather than as "
                                  "miscalibration, and a >= 0.88 pass at 95 % can be earned by width. The registered "
                                  "value is unchanged and is the verdict. What is added is the evidence: the "
                                  "composition of every fold's calibration pairs by metal-state pair, the Pr/Nd share, "
                                  "the median residual overall and on the Pr/Nd subset, and a Pr/Nd-RESTRICTED quantile "
                                  "set printed as an EXPLORATORY diagnostic that decides nothing (it is not registered "
                                  "and no addendum authorises it as the verdict). Read a high-side 80 % FAIL together "
                                  "with prnd_share and the two medians before calling it miscalibration",
    "s2_averaging_unit": "section 4 / metrics.REGISTERED_UNIT_COLS: the registered averaging unit of V6 is the SYSTEM,"
                         "so every V6 macro number -- log D MAE and logSF MAE alike -- is a per-system mean then an "
                         "equal-weight mean over the 13 systems, never a pooled mean over pairs. A pooled pair mean is "
                         "printed beside it as a side aggregation (metrics: role 'side') and decides nothing. S2(a)'s "
                         "paired direction contrasts use the cell-pair unit of S1(c), which for V6 IS the system: the "
                         "cell pair is (system, Nd(III), Pr(III)) and there is exactly one per system",
    "s2_strata": "section 3.4: 'Every V6 number is reported pooled and per acid medium. The nitrate process case "
                 "(section 14) rests on the HNO3 pairs only.' So S2(a)'s margin is required in the pooled pairs AND in "
                 "the HNO3 pairs (142 of 209), every S2 number is reported per acid medium, the 7-system sensitivity "
                 "(>= 10 Pr and Nd rows, >= 5 other Ln(III); 168 pairs) is scored in the same single run, and TODGA is "
                 "reported on its own (83 pairs: HCl 54, HNO3 26, malonic 3). The sensitivity and the per-acid and "
                 "TODGA breakdowns are REPORTED, not deciding: section 9 S2 names only the pooled and HNO3 tests",
    "v6_deltas_not_run": "POST-HOC addendum 5 item 1: the section 11 V6 actinide deltas are NOT_RUN in this run, with "
                         "their cost and their consequence recorded (V6_ACTINIDE_DELTAS_NOT_RUN)",
    "power_not_run": "POST-HOC addendum 5 items 1 and 4: no section 8 power check runs here; the debt is inventoried, "
                     "not discharged, and every contrast owing one is UNDECIDED, never a null (POWER_CHECK_NOT_RUN)",
    "idempotence": "section 15's 'nothing is re-run' as a lock: the runner refuses to start when "
                   "evaluation/confirmation/decisions/confirmation.json exists unless --resume, and --resume may only "
                   "COMPLETE unfitted folds -- it re-scores no claim under a code digest different from the one the "
                   "lock records, and it never widens the claim list (lock_verdict). The lock is spent at the START, "
                   "not at the end: decisions/confirmation.json is written last, so the runner also writes "
                   "decisions/run_started.json once the gates pass and before any fold file or record exists "
                   "(started_path), and lock_verdict refuses on EITHER file. Without it a run that died after writing "
                   "records left the lock reading first_run, and a second invocation -- under changed code, after a "
                   "stage re-registration -- would have scored one claim from two record sets written under two code "
                   "digests (task X finding). Every record set a claim is scored from is also checked, record by "
                   "record, against the LIVE code digest (read_seed_predictions)",
}

#: POST-HOC addendum 5 item 1, with the cost that drove it and the consequence of not running it
V6_ACTINIDE_DELTAS_NOT_RUN = {
    "status": NOT_RUN,
    "what": "the section 11 V6 actinide deltas (H3 at confirmation): arms M0 (= B5, the deployed configuration of "
            "addendum 3 item 2) and B6, transforms WITH / WITHOUT / ACT_PERMUTED at the frozen per-fold "
            "hyperparameters, deltas WITH - WITHOUT and WITH - PERMUTED with R19 and TOST (epsilon = 0.05) on "
            "macro MAE log D, rank accuracy, logSF MAE and calibration",
    "registered_by": "section 11 ('V6, at confirmation only') and CONFIRMATION_PLAN section 3.4",
    "not_run_by": "POST-HOC addendum 5 item 1 (compute-driven, chosen by the user): 'It does NOT run the section 11 V6 "
                  "actinide deltas nor any section 8 power check; both are reported NOT_RUN with their cost and their "
                  "consequence'",
    "cost_hours_serial": 30.3,
    "cost_basis": "CONFIRMATION_PLAN section 6: M0 = B5 WITH (574.3 s) + WITHOUT (495.8) + PERMUTED (579.1) over 13 x 5 "
                  "= 65 folds = 29.8 h serial, plus the B6 reference's three transforms at about 30 s/fold = 0.5 h; "
                  "measured per-fold means from evaluation/h3/records/<arm>/<transform>/<design>/s104729/*.json steps",
    "consequence": "H3 keeps the discovery-side verdict: UNDECIDED for both arms, with no power check, so it is never "
                   "reported as a null (addendum 4 item 4). Delta logSF MAE for H3 stays NOT_DEFINED -- V6 is the only "
                   "design that could have supplied it (CONFIRMATION_PLAN section 3.4 item 2), so the quantity now "
                   "exists nowhere and no later run may create it: section 3.4 says V6 runs ONCE and addendum 5 spends "
                   "that run on S2. The section 11 consequence is unchanged: actinide rows enter the deployed Ln "
                   "configuration only on a *helps* verdict, and there is none",
}
POWER_CHECK_NOT_RUN = {
    "status": NOT_RUN,
    "what": "the section 8 signal-injection power check over the H3 contrasts (y' = y + kappa*s, kappa in "
            "{0.1, 0.25, 0.5, 1.0}, every endpoint of the contrast refitted at its selected hyperparameters)",
    "not_run_by": "POST-HOC addendum 5 items 1 and 4: 'the power-check debt is inventoried, not discharged'",
    "inventory": "20 of 20 H3 contrasts owe the check and 0 have one: 6,272 injected refits, about 436 h of serial "
                 "compute at the measured per-fold cost (evaluation/h3/h3_summary.json -> power_debt; the totals are an "
                 "upper bound because each contrast is priced separately and a shared WITH leg is counted twice, "
                 "h3.POWER_COST_BASIS). The de-duplicated figure for the deployed arm's two V5 contrasts is 336 refits, "
                 "about 50 h serial",
    "consequence": "every one of those contrasts is reported UNDECIDED (no registered power check) in the words "
                   "addendum 4 item 4 fixes, NEVER as a null and never as 'no effect'; the inventory and its cost are "
                   "part of this report so the omission is visible rather than implicit",
}

# --------------------------------------------------------------------------------------------- #
# measured unit costs (the --dry-run estimate)
# --------------------------------------------------------------------------------------------- #
#: measured per-fold seconds (point + intervals), CONFIRMATION_PLAN section 6, from the record ``steps`` of
#: ``evaluation/discovery/<arm>/<design>/s104729/*.json`` and the frozen H3 legs; the frozen-configuration factor 0.2785
#: is measured (``evaluation/h3/decisions/cost_estimate.json -> h3_over_discovery_ratio``)
UNIT_SECONDS: dict[str, float] = {
    "M1@V5__primary__batched_max4": 684.8, "M2@V5__primary__batched_max4": 303.6,
    "M1@V5__strict__batched_max4": 568.1, "M2@V5__strict__batched_max4": 279.8,
    "M1@V5__hno3_only__batched_max4": 680.2, "M2@V5__hno3_only__batched_max4": 283.6,
    "M1@V5PAIR__primary__batched": 686.4, "M2@V5PAIR__primary__batched": 286.9,
    "B6@V5__primary__exact": 9.6, "B6@V2__element__exact": 9.0, "B6:ACT_PERMUTED@V2__element__exact": 2.6,
    # the item 6 refits' comparator legs: the same arm on the same exact scheme, so the measured per-fold mean of
    # B6@V5__primary__exact is the measurement; B3i / B0 / B3x are closed form and keyed by arm below
    "B6@V5__strict__exact": 9.6, "B6@V5__hno3_only__exact": 9.6,
    "M1@V6__prnd__exact": 0.2785 * 684.8, "M2@V6__prnd__exact": 0.2785 * 303.6, "B8@V6__prnd__exact": 0.2785 * 699.6,
    # The closed-form comparators are NOT free.  They were priced 0.0 s here on the strength of
    # ``discovery.READINGS['comparator_intervals']`` -- no comparator-interval job ran in discovery, and
    # ``evaluation/discovery/`` holds no B0 / B3i / B3x record -- so nothing had ever MEASURED that zero, and it silently
    # priced 1,702 of the run's 3,418 folds at nothing (task X finding).  Each one pays ``prepare_fold`` plus a full
    # ``interface.ConformalWrapper`` fit over the design's inner calibration splits (90 of them on a V5 / V6 exact fold,
    # 3 on a V5-PAIR batched one), which is the whole cost of a closed-form arm.  MEASURED on this machine, mean over
    # the first 10 REAL selection-half folds of each stem (``prepare_fold`` + ``ComparatorIntervalsRunner.point``, the
    # two steps the run executes), writing no record and touching no confirmation-half or V6 row -- a cost benchmark of
    # the same kind as the pre-seal one in ``manifests/run_info/g19_run_preseal_jobs.json``:
    #   B3i @ V5__primary__exact 8.43 s, V5__strict__exact 8.65 s, V5__hno3_only__exact 8.51 s  (90 inner splits)
    #   B0  @ V5__primary__exact 1.40 s, V5__strict__exact 1.40 s                               (90 inner splits)
    #   B3x @ V5__primary__exact 1.54 s, V5PAIR__primary__batched 1.56 s                        (90 / 3 inner splits)
    # The first fold of a design pays the guard-cache fill (34 s for B3i) and the rest run at 11-13 s falling to ~8 s, so
    # these means are conservative for a 111-fold job.  The V6 legs take the same arm's V5-exact figure: addendum 7 item
    # 2 gives a V6 comparator-interval job the V5 ``InnerCellCalibration`` splitter, and its training set is the corpus
    # minus one system's Pr/Nd, so the fit is the V5-exact one -- the 0.2785 frozen-configuration factor is a TUNING
    # saving and a closed-form arm has no tuning to skip.
    "B3i": 8.6, "B0": 1.4, "B3x": 1.6, "FLAT": 0.0, "HEAVIER": 0.0,
}
#: the measured parallel efficiency of this machine at 2 workers (discovery: 149.91 h serial in 76.5955 h of wall clock)
PARALLEL_EFFICIENCY_2_WORKERS = 1.96
#: confirmation-half fold counts **per seed**, from ``folds/INDEX.json -> designs.*.by_half.C`` and ``placeholders``
#: (CONFIRMATION_PLAN sections 2.2, 2.3 and 3.1).  A batched file holds the colourings of every seed, so its ``by_half.C``
#: count is divided by the 5 discovery seeds that built it: V5-primary 135/5 = 27 (the plan's "expect 27 +- 1 per seed"),
#: strict 37/5 = 8 (rounded up), HNO3-only 135/5 = 27; V5-PAIR was built for seed 104729 alone, so its 38 IS per seed;
#: the exact V5 and V2 files are seed-INDEPENDENT (111 and 11 folds), and an arm is still refitted on them per withheld
#: seed because its conformal inner folds are drawn with that seed (section 15 resolution).  V6 has no fold design yet:
#: 13 systems, one fold each, built in this run by the section 3.4 component-aware hiding.
FOLD_COUNTS_CONFIRMATION: dict[str, int] = {
    "V5__primary__batched_max4": 27, "V5__strict__batched_max4": 8, "V5__hno3_only__batched_max4": 27,
    "V5PAIR__primary__batched": 38, "V5__primary__exact": 111, "V2__element__exact": 11, "V1__copy__exact": 52,
    "V6__prnd__exact": 13,
    # the item 6 refits' comparator legs, seed-independent exact files (folds/INDEX.json by_half.C, not divided)
    "V5__strict__exact": 25, "V5__hno3_only__exact": 111,
}


# --------------------------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------------------------- #

def conf_root(out_root: Path | str) -> Path:
    return Path(out_root) / CONF_REL


def decisions_path(out_root: Path | str) -> Path:
    return conf_root(out_root) / "decisions" / "confirmation.json"


def tables_dir(out_root: Path | str) -> Path:
    return Path(out_root) / "tables"


def report_path(out_root: Path | str) -> Path:
    return Path(out_root) / REPORT_REL


def plan_path(out_root: Path | str) -> Path:
    return Path(out_root) / PLAN_REL


def fold_paths(out_root: Path | str, arm: str, design_dir: str, seed_index: int, fold_id: str) -> tuple[Path, Path]:
    """Prediction parquet and record JSON of one confirmation fold.  The directory carries the OPAQUE seed index, never
    a seed: ``evaluation/confirmation/records/<arm>/<design_dir>/i<index>/<fold_id>.{parquet,json}``."""
    d = conf_root(out_root) / "records" / arm / design_dir / f"i{int(seed_index)}"
    return d / f"{fold_id}.parquet", d / f"{fold_id}.json"


def folds_dir(out_root: Path | str) -> Path:
    """Where the withheld-seed fold files live.  Separate from ``folds/`` so no registered discovery design is touched,
    and named by seed INDEX only."""
    return conf_root(out_root) / "folds"


def started_path(out_root: Path | str) -> Path:
    """The START marker of the single run: written once the gates have passed and BEFORE any fold file or record.

    ``decisions/confirmation.json`` is written LAST, so on its own it spends the once-only lock only when the run
    finishes.  A run that wrote 2,000 records and then died left the lock reading ``first_run``, and a second
    invocation without ``--resume`` -- under changed code, after a re-registration -- would have mixed two record sets
    into one score (task X finding).  This marker makes the lock spend at the START, which is what section 15's "one
    run" means, while ``--resume`` keeps its exact registered power: complete unfitted folds, under the SAME code
    digest, with no new claim."""
    return conf_root(out_root) / "decisions" / "run_started.json"


#: the fields of a confirmation record that identify WHICH fit it is.  A confirmation record carries no ``digest``
#: (discovery's :func:`g19_run_discovery.fold_digest` is a resume key of the discovery layout), so where discovery
#: names an M1 record by its fold digest this run names it by the digest of these fields.
RECORD_IDENTITY_FIELDS: tuple[str, ...] = ("schema", "registry_stage", "arm", "stem", "fold_id", "fold_hash",
                                           "seed_index", "code_digest", "prereg_sha256", "prereg_addenda_sha256",
                                           "selected_config", "model_seed")


def confirmation_record_digest(record: Mapping[str, Any]) -> str:
    """A stable content digest of one confirmation record's identity (:data:`RECORD_IDENTITY_FIELDS`).

    It is what M2's record stores as ``m1_record_digest`` in this run: discovery stores the M1 fold digest there and a
    confirmation record has none, so without this the field would be ``null`` and the link from an M2 fit to the exact
    M1 fit it took its hyperparameters from would not be recorded at all.  It is an implementation reading of section
    6's "M2 keeps each outer fold's retained M1 hyperparameters", not a registered quantity: nothing is scored from it.
    """
    body = {k: record.get(k) for k in RECORD_IDENTITY_FIELDS}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------- #
# the withheld seeds
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SeedStore:
    """The 5 withheld seeds, held in memory for the length of one run and never printed.

    ``digest`` is the commitment of section 15, which IS the public name of this object; ``values`` is private and its
    ``repr`` is redacted, so a traceback, a log line, a ``json.dumps(default=str)`` or an f-string cannot leak a seed.
    Code asks for :meth:`indices` and passes an index around; only the fitting call site asks for :meth:`seed`.
    """

    digest: str
    path: str
    values: tuple[int, ...] = field(repr=False)
    verified: bool = True

    def __post_init__(self) -> None:
        if len(self.values) != N_SEEDS:
            raise ValueError(f"the seed store holds {len(self.values)} seeds, section 15 registers {N_SEEDS}")
        if len(set(self.values)) != N_SEEDS:
            raise ValueError("the seed store holds a repeated seed")

    def __repr__(self) -> str:      # pragma: no cover - exercised by the leak test
        return f"SeedStore(digest={self.digest[:12]}..., n={len(self.values)}, values=<withheld>)"

    __str__ = __repr__

    def indices(self) -> tuple[int, ...]:
        """The opaque indices 1..5 every output is labelled with (the store's own order)."""
        return tuple(range(1, len(self.values) + 1))

    def seed(self, index: int) -> int:
        """The seed of ``index`` (1-based).  The ONLY accessor; call it at the fitting site, never in a log line."""
        if int(index) not in self.indices():
            raise ValueError(f"seed index {index!r} is not in {self.indices()}")
        return int(self.values[int(index) - 1])

    def index_of(self, seed: int) -> int:
        return int(self.values.index(int(seed))) + 1

    def public(self) -> dict[str, Any]:
        """What may be written to a file: the commitment, the count, the verdict -- no value."""
        return {"commitment_sha256": self.digest, "n_seeds": len(self.values), "verified_against_commitment":
                bool(self.verified), "seed_indices": list(self.indices()), "values": "withheld",
                # the OPERATIVE rule is addendum 6 item 4's: the seeds ARE revealed, in decisions/CONFIRMATION.md, after
                # the run.  SEED_DISCLOSURE is the narrowing it superseded and is carried only so the record shows what
                # was superseded and by what -- a reader of this block must not take it for the rule in force
                "disclosure": SEED_REVELATION, "superseded_disclosure": SEED_DISCLOSURE}


def committed_digest(root: Path | None = None) -> str:
    p = (paths.G19_ROOT if root is None else Path(root)) / SEED_COMMITMENT_REL
    if not p.exists():
        raise SystemExit(f"refused: no confirmation-seed commitment at {p} (section 15)")
    return p.read_text(encoding="utf-8").strip().split()[0]


def load_seed_store(store_path: Path | str, *, root: Path | None = None, seal=None, prereg_paths=None) -> SeedStore:
    """Verify ``store_path`` against the section 15 commitment and return a :class:`SeedStore`.

    ``seal`` is ``scripts/g19_seal_prereg.py`` as a module (the runner passes it; tests pass a stub exposing
    ``verify_seed_store`` and ``load_committed_seeds``).  A store that does not verify raises: the run never starts on
    unverified seeds, and no seed is printed on either path.
    """
    p = Path(store_path)
    if not p.exists():
        raise SystemExit(f"refused: --seed-store {p} does not exist; the 5 withheld seeds enter only through it")
    if seal is None:      # pragma: no cover - the runner always passes the module
        raise ValueError("load_seed_store needs the seal module (scripts/g19_seal_prereg.py)")
    pp = prereg_paths if prereg_paths is not None else (
        seal.default_paths() if root is None else seal.PreregPaths(root=Path(root), repo_root=paths.REPO_ROOT))
    ok, msg = seal.verify_seed_store(pp, p)
    if not ok:
        raise SystemExit(f"refused: the seed store does not verify against the section 15 commitment: {msg}")
    seeds = tuple(int(s) for s in seal.load_committed_seeds(pp, p))
    return SeedStore(digest=committed_digest(root), path=str(p), values=seeds, verified=True)


def seed_token(index: int) -> str:
    """The opaque label a written file carries where a seed would otherwise appear."""
    return f"i{int(index)}"


def scrub(obj: Any, store: SeedStore) -> Any:
    """Replace every withheld seed -- as an int, and as a substring of any string, including a fold id or a path -- by
    its opaque index token, recursively.

    Every record, table row, decision block and log line goes through this before it is written, so a seed cannot leak
    through a field nobody thought about (``job.seed``, a batched fold id ``b104729_3``, a directory name).
    :func:`scan_for_seed_leak` then re-reads the files and fails the run if one got through anyway: the scrub is the
    intent, the scan is the proof.
    """
    if isinstance(obj, Mapping):
        return {scrub(k, store): scrub(v, store) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        out = [scrub(v, store) for v in obj]
        return type(obj)(out) if not isinstance(obj, set) else set(out)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, np.integer)) and int(obj) in store.values:
        return seed_token(store.index_of(int(obj)))
    if isinstance(obj, str):
        out = obj
        for s in store.values:
            out = out.replace(str(s), seed_token(store.index_of(s)))
        return out
    return obj


def scan_for_seed_leak(store: SeedStore, targets: Iterable[Path | str], *, extra_text: Iterable[str] = (),
                       allow: Iterable[Path | str] = (), require: Iterable[Path | str] = ()) -> dict[str, Any]:
    """Re-read every file the run wrote (and any captured log text) and refuse if a seed's decimal form appears.

    The check is deliberately crude and total: a withheld seed is a 6-digit integer, so ``str(seed)`` is searched as a
    substring of the raw bytes of every listed file, whatever its format.  Its own report records only the COUNT of
    files scanned and the verdict -- never which seed, never where.

    ``allow`` and ``require`` implement POST-HOC addendum 6 item 4 (:data:`SEED_REVELATION`): the run scans everything it
    wrote BEFORE ``decisions/CONFIRMATION.md`` exists, and once that file is written it is scanned again with the report
    in ``allow`` (a seed there is the registered revelation, not a leak) and in ``require`` (it MUST name all five, or
    section 15's revelation did not happen).  A file in ``allow`` that is not in ``require`` is merely exempt.
    """
    hits: list[str] = []
    needles = [str(s).encode() for s in store.values]
    allowed = {str(Path(a).resolve()) for a in allow}
    required = {str(Path(r).resolve()): r for r in require}
    found_required: dict[str, int] = {}
    n = 0
    for t in targets:
        p = Path(t)
        if not p.exists():
            continue
        for f in ([p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())):
            n += 1
            b = f.read_bytes()
            key = str(f.resolve())
            n_seeds_named = sum(1 for x in needles if x in b)
            if key in required:
                found_required[str(required[key])] = n_seeds_named
            if n_seeds_named and key not in allowed:
                hits.append(str(f))
    for txt in extra_text:
        if any(x.decode() in str(txt) for x in needles):
            hits.append("<log text>")
    missing = sorted(k for k, v in found_required.items() if v != len(needles))
    for r in required.values():
        if str(r) not in found_required:
            missing.append(str(r))
    return {"files_scanned": n, "ok": not hits and not missing,
            "files_containing_a_withheld_seed": sorted(set(hits)),
            "revelation_files": sorted(str(Path(a)) for a in allow),
            "revelation_incomplete": sorted(set(missing)),
            "rule": "no withheld seed's decimal form may appear in any file this run wrote or in its log, EXCEPT "
                    "decisions/CONFIRMATION.md written last, which must name all five (addendum 6 item 4)"}


# --------------------------------------------------------------------------------------------- #
# the plan (decisions/CONFIRMATION_PLAN.md)
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Claim:
    """One frozen claim of the plan: the contrast R19 is applied to on the confirmation half."""

    claim_id: str
    label: str
    family: str
    candidate: str
    comparator: str
    design: str
    margin: float
    stem: str = ""
    comparator_stem: str = ""
    transform: str = ""
    comparator_transform: str = ""
    discovery_point: float | None = None

    @property
    def key(self) -> str:
        return f"{self.label}@{self.design}"


_CLAIM_ROW = re.compile(r"^\|\s*\*\*(C\d)\*\*\s*\|(.*)$")


def parse_claims(text: str) -> list[Claim]:
    """The frozen claims of ``decisions/CONFIRMATION_PLAN.md`` section 2, read from its table rows.

    A row is ``| **C<k>** | <claim> | <family> | <candidate> | <comparator> | <margin> | <designs> | <discovery Delta> |
    <verdict> |``.  Only the fields this run needs are parsed, and every one of them is checked: a claim whose margin,
    design or arms cannot be read is a refusal, never a default.
    """
    out: list[Claim] = []
    for line in text.splitlines():
        m = _CLAIM_ROW.match(line.strip())
        if not m:
            continue
        cid, rest = m.group(1), m.group(2)
        cells = [c.strip() for c in rest.split("|")]
        if len(cells) < 8:
            raise ValueError(f"{cid}: the plan's claim row has {len(cells)} fields, expected at least 8")
        label, family, cand, comp, margin_s, designs, point_s = cells[0], cells[1], cells[2], cells[3], cells[4], cells[5], cells[6]

        def plain(s: str) -> str:
            return re.sub(r"\*\*|`", "", s).strip()

        def clean(s: str) -> str:
            return plain(s).split("—")[0].split("(")[0].strip()

        cand_c, comp_c = clean(cand), clean(comp)
        # the margin cell may carry its NAME as well as its value ("δ5 = 0.10568948790529951", "0.05 (registered: ...)"),
        # and a name contains digits: take the longest numeric literal, never the first one
        nums = re.findall(r"[0-9]*\.?[0-9]+", plain(margin_s).replace(",", ""))
        mm = max(nums, key=len) if nums else None
        if mm is None:
            raise ValueError(f"{cid}: no margin in {margin_s!r}")
        design = "V5" if "V5-primary" in designs or "V5-PAIR" not in designs and "V5" in designs else ""
        for d in ("V5-PAIR", "V2", "V1", "V6"):
            if d in designs:
                design = d
        if not design:
            raise ValueError(f"{cid}: no design in {designs!r}")
        pt = None
        pm = re.search(r"([+-]?[0-9]*\.?[0-9]+)", re.sub(r"\*\*", "", point_s))
        if pm:
            pt = float(pm.group(1))
        cand_arm, cand_tf = (cand_c.split(":") + [""])[:2]
        comp_arm, comp_tf = (comp_c.split(":") + [""])[:2]
        out.append(Claim(claim_id=cid, label=clean(label), family=plain(family), candidate=cand_arm,
                         comparator=comp_arm, design=design.replace("-", ""), margin=float(mm),
                         transform=cand_tf, comparator_transform=comp_tf, discovery_point=pt))
    return out


def read_plan(path: Path | str) -> dict[str, Any]:
    """The plan as this run reads it: its claims, its digest and the ``<= 5`` check of section 15."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"refused: no frozen plan at {p}; section 15 freezes the claims BEFORE the run")
    text = p.read_text(encoding="utf-8")
    claims = parse_claims(text)
    if not claims:
        raise SystemExit(f"refused: {p} freezes no claim (section 15: the plan gives the arm, comparator, metric and "
                         "designs of each claim)")
    if len(claims) > MAX_CLAIMS:
        raise SystemExit(f"refused: {p} freezes {len(claims)} claims; section 15 allows at most {MAX_CLAIMS}")
    return {"path": str(p), "sha256": hashlib.sha256(text.encode("utf-8").replace(b"\r\n", b"\n")).hexdigest(),
            "n_claims": len(claims), "claims": claims,
            "claim_ids": [c.claim_id for c in claims], "claim_keys": [c.key for c in claims]}


def claim_of(plan: Mapping[str, Any], claim_id: str) -> Claim:
    for c in plan["claims"]:
        if c.claim_id == claim_id or c.key == claim_id or c.label == claim_id:
            return c
    raise SystemExit(f"refused: {claim_id!r} is not a frozen claim of the plan ({[c.claim_id for c in plan['claims']]}); "
                     "section 15: 'No sixth claim is added and nothing is re-run'")


# --------------------------------------------------------------------------------------------- #
# the job inventory and the cost estimate
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConfJob:
    """One (arm, design file, withheld-seed index) unit of work.  ``seed_index`` is opaque; no seed appears here."""

    purpose: str
    arm: str
    design: str
    stem: str
    seed_index: int
    n_folds: int
    transform: str = ""
    tuned: bool = True
    note: str = ""

    @property
    def key(self) -> str:
        t = f":{self.transform}" if self.transform else ""
        return f"{self.arm}{t}@{self.stem}/i{self.seed_index}"

    @property
    def unit_seconds(self) -> float:
        for k in (f"{self.arm}:{self.transform}@{self.stem}" if self.transform else "", f"{self.arm}@{self.stem}",
                  self.arm):
            if k and k in UNIT_SECONDS:
                return float(UNIT_SECONDS[k])
        return float("nan")


def enumerate_jobs(plan: Mapping[str, Any], *, n_seeds: int = N_SEEDS,
                   fold_counts: Mapping[str, int] | None = None) -> list[ConfJob]:
    """Every job of the single registered run, in the order of ``CONFIRMATION_PLAN`` section 5.

    Claims C1-C3 share one fit set (M1 then M2 on the confirmation V5-primary batched folds); C4 is B6 WITH tuned plus
    ACT_PERMUTED frozen on the V2 folds; R19 item 6's two refits run on ONE seed (addendum 1 item 4 / addendum 5 item
    1); S1(c) adds the V5-PAIR folds; V6 runs once with the frozen configurations.
    """
    fc = dict(FOLD_COUNTS_CONFIRMATION if fold_counts is None else fold_counts)
    ids = {c.claim_id for c in plan["claims"]}
    jobs: list[ConfJob] = []
    seeds = list(range(1, int(n_seeds) + 1))
    if ids & {"C1", "C2", "C3"}:
        for i in seeds:
            for arm in ("M1", "M2"):            # M2 keeps each outer fold's retained M1 hyperparameters (addendum 1 6(d))
                jobs.append(ConfJob("claims C1-C3 (core)", arm, "V5", "V5__primary__batched_max4", i,
                                    fc.get("V5__primary__batched_max4", 0),
                                    note="M2's record verifies against M1's record of the same fold"))
        if "C3" in ids:
            for i in seeds:
                jobs.append(ConfJob("claim C3 comparator B6r0", "B6", "V5", "V5__primary__exact", i,
                                    fc.get("V5__primary__exact", 0), note="B6r0 is read from the B6 fit (rank 0)"))
        for i in seeds:                          # the closed-form comparators and their multi-seed conformal intervals
            for arm in ("B3i", "B0"):
                jobs.append(ConfJob("claim comparators (closed form)", arm, "V5", "V5__primary__exact", i,
                                    fc.get("V5__primary__exact", 0), note="section 15: deterministic point predictions, "
                                                                          "conformal inner folds drawn with each seed"))
        for arm in ("M1", "M2"):                 # item 6 refits, seed 104729 only
            for stem in ("V5__strict__batched_max4", "V5__hno3_only__batched_max4"):
                jobs.append(ConfJob("R19 item 6 refits (discovery seed 104729 only)", arm, "V5", stem, 0,
                                    fc.get(stem, 0), note="addendum 1 item 4 scopes these refits to seed 104729; "
                                                          "addendum 5 item 1 keeps the core run"))
        # ... and the COMPARATOR leg of each refit, on the setting's own EXACT folds.  A sensitivity is a Delta: without
        # the comparator refitted on the same setting there is nothing to subtract, R19 item 6 reports the two members of
        # addendum 1 item 4's own reduced set NOT_EVALUATED, and every V5 claim is UNDECIDED whatever the data says
        for stem in ("V5__strict__exact", "V5__hno3_only__exact"):
            for arm in ("B3i", "B0"):
                jobs.append(ConfJob("R19 item 6 refit comparators (closed form, seed 104729)", arm, "V5", stem, 0,
                                    fc.get(stem, 0), note="section 5 comparator_folds: a closed-form comparator is "
                                                          "fitted on the setting's EXACT leave-one-cell-out folds"))
            if "C3" in ids:
                jobs.append(ConfJob("R19 item 6 refit comparators (closed form, seed 104729)", "B6", "V5", stem, 0,
                                    fc.get(stem, 0), note="B6r0 is read from the B6 fit (rank 0) of the same fold"))
    if "C4" in ids:
        for i in seeds:
            jobs.append(ConfJob("claim C4 (H3 control contrast)", "B6", "V2", "V2__element__exact", i,
                                fc.get("V2__element__exact", 0), transform="WITH"))
            jobs.append(ConfJob("claim C4 (H3 control contrast)", "B6", "V2", "V2__element__exact", i,
                                fc.get("V2__element__exact", 0), transform="ACT_PERMUTED", tuned=False,
                                note="refitted at the WITH run's selected hyperparameters of the same fold (section 11)"))
    for i in seeds:                              # S1(c): M1 then M2 on the batched V5-PAIR folds, yardsticks refitted there
        for arm in ("M1", "M2"):
            jobs.append(ConfJob("S1(c) direction and logSF (V5-PAIR)", arm, "V5PAIR", "V5PAIR__primary__batched", i,
                                fc.get("V5PAIR__primary__batched", 0)))
        jobs.append(ConfJob("S1(c) yardstick refits (closed form)", "B3x", "V5PAIR", "V5PAIR__primary__batched", i,
                            fc.get("V5PAIR__primary__batched", 0), note="B3x and B3i refitted on exactly M2's folds"))
    for i in seeds:                              # V6, once, at the frozen configurations
        for arm in ("M1", "M2", "B8"):
            jobs.append(ConfJob("the single V6 run (S2)", arm, "V6", "V6__prnd__exact", i, fc.get("V6__prnd__exact", 0),
                                tuned=False, note="frozen configurations, no re-tuning (section 3.4)"))
        # S2(a) requires Delta_Y against EVERY yardstick in {HEAVIER, B3x-derived, B3i-derived, B8}, "with B3x, B3i and
        # B8 fitted on the same V6 folds and withheld seeds as M2" (section 9, resolved 2026-09-15).  B8 is above;
        # HEAVIER needs no fit; B3x and B3i do, and B3i is also S2(b)'s lookup-derived comparator and the V5 lookup
        # comparator the hidden Pr/Nd log D MAE is measured against
        for arm in ("B3x", "B3i"):
            jobs.append(ConfJob("the single V6 run (S2) -- yardsticks", arm, "V6", "V6__prnd__exact", i,
                                fc.get("V6__prnd__exact", 0), tuned=False,
                                note="section 9 S2(a)/(b): the B3x- and B3i-derived yardsticks on M2's V6 folds"))
    return jobs


def cost_estimate(jobs: Sequence[ConfJob]) -> dict[str, Any]:
    """Serial hours and wall hours at 2 workers from the measured unit costs (:data:`UNIT_SECONDS`)."""
    rows = []
    for j in jobs:
        s = j.unit_seconds * j.n_folds
        rows.append({"key": j.key, "purpose": j.purpose, "arm": j.arm, "transform": j.transform, "stem": j.stem,
                     "seed_index": j.seed_index, "n_folds": j.n_folds, "unit_seconds": j.unit_seconds,
                     "serial_hours": s / 3600.0 if np.isfinite(s) else float("nan")})
    tot = float(np.nansum([r["serial_hours"] for r in rows]))
    by_purpose: dict[str, float] = {}
    for r in rows:
        if np.isfinite(r["serial_hours"]):
            by_purpose[r["purpose"]] = by_purpose.get(r["purpose"], 0.0) + r["serial_hours"]
    return {"jobs": rows, "n_jobs": len(rows), "n_folds": int(sum(j.n_folds for j in jobs)),
            "serial_hours": tot, "wall_hours_2_workers": tot / PARALLEL_EFFICIENCY_2_WORKERS,
            "by_purpose_serial_hours": dict(sorted(by_purpose.items())),
            "unknown_unit_cost": sorted({r["key"] for r in rows if not np.isfinite(r["unit_seconds"])}),
            "basis": "measured per-fold means (point + intervals) of the discovery and H3 records on this machine, and "
                     "for the closed-form comparators B0 / B3i / B3x -- which have no discovery record at all and had "
                     "been priced 0.0 s, 1,702 of these folds -- the measured mean of prepare_fold + "
                     "ComparatorIntervalsRunner.point over the first 10 real selection-half folds of each stem "
                     "(UNIT_SECONDS). The parallel efficiency 1.96x at 2 workers is measured (discovery: 149.91 h serial "
                     "in 76.5955 h wall) and the confirmation fit loop now actually uses a 2-process pool, so the wall "
                     "figure is reachable. Treat +-30 % as the honest band: discovery itself overran its 60 h budget by "
                     "28 %. The V6 learned-arm lines assume the section 3.4 FROZEN configurations (the 0.2785 factor is "
                     "a tuning saving): the tree as it stands re-tunes them, which is why the runner refuses at stage "
                     "'v6_frozen_configurations' -- implemented as registered, the V6 block is the 8.5 h costed here; "
                     "run as the code stands it would be about 30 h, and the plan about 132 h rather than 110 h",
            "not_in_this_estimate": {"v6_actinide_deltas": V6_ACTINIDE_DELTAS_NOT_RUN["cost_hours_serial"],
                                     "power_check": POWER_CHECK_NOT_RUN["inventory"]}}


# --------------------------------------------------------------------------------------------- #
# the confirmation-half scoring population
# --------------------------------------------------------------------------------------------- #

def confirmation_scored_ids(fold: FI.Fold) -> tuple[str, ...]:
    """The fold's scored rows of the CONFIRMATION half -- the mirror of ``discovery.selection_scored_ids``."""
    return tuple(r for r in fold.scored_row_ids if str(fold.row_half.get(r, fold.half)) == D.CONFIRMATION)


def fit_scored_ids(fold: FI.Fold) -> tuple[str, ...]:
    """The rows THIS run scores in one fold.

    A design with registered halves contributes its CONFIRMATION-half scored rows
    (:func:`confirmation_scored_ids`).  A **V6** fold contributes every scored row: V6 is the single run of section 3.4
    over all 13 systems and is not a confirmation-HALF design (:data:`READINGS` ``v6_half``), so its rows are selected
    by :func:`assert_v6_scope` -- every scored row a ``V6_TARGET_ROW`` -- and its folds carry half ``NA``.  Reading the
    half here instead would silently return nothing and make the whole V6 leg unreachable.
    """
    if is_v6(fold.design):
        return tuple(fold.scored_row_ids)
    return confirmation_scored_ids(fold)


def confirmation_job_folds(job: D.JobSpec, folds: Sequence[FI.Fold]) -> list[tuple[FI.Fold, int]]:
    """``(fold, ordinal)`` of every fold a CONFIRMATION job fits -- the mirror of ``discovery.job_folds``.

    Discovery's selector is unusable here by construction: it drops every fold whose ``half`` is the confirmation half
    and keeps only folds with a SELECTION-half scored row, so on the confirmation half it returns either nothing or
    exactly the folds this run may never fit.  This one keeps the job's fold seed (a multi-seed file), refuses a
    SELECTION-half fold, and requires at least one row this run scores (:func:`fit_scored_ids`).  The ordinals are
    discovery's (``discovery.fold_ordinals``): the section 15 model fold number is a property of the design file, not of
    the half.
    """
    seeds = {f.seed for f in folds}
    if job.fold_seed is None and seeds != {None} and len(seeds) != 1:
        raise ValueError(f"{job.key}: fold file {job.stem} holds seeds {sorted(map(str, seeds))}; give fold_seed")
    key_seed = job.fold_seed if job.fold_seed is not None else (job.seed if seeds == {None} else next(iter(seeds)))
    ords = D.fold_ordinals(folds, key_seed)
    out = []
    for f in folds:
        if job.fold_seed is not None and f.seed != job.fold_seed:
            continue
        if f.half == D.SELECTION:
            continue
        if fit_scored_ids(f):
            out.append((f, ords[f.fold_id]))
    return out


def confirmation_fittable_folds(job: D.JobSpec, folds: Sequence[FI.Fold],
                                excluded_ids: Iterable[str]) -> list[tuple[FI.Fold, int]]:
    """:func:`confirmation_job_folds` minus the folds whose scored rows are all acidic co-extractant rows (section 2)."""
    ex = {str(r) for r in excluded_ids}
    return [(f, k) for f, k in confirmation_job_folds(job, folds) if any(str(r) not in ex for r in fit_scored_ids(f))]


class RedactedRunError(RuntimeError):
    """An error of the single run whose message has passed through :func:`scrub`, raised ``from None``.

    A withheld seed must not reach a traceback: every message this run can print is built from the opaque index, and an
    exception raised deeper (``prepare_fold``'s ``ValueError``, a runner's ``KeyError``) is re-raised as this, with the
    original chain suppressed, so no ``__context__`` frame can carry the seed either (POST-HOC addendum 6 item 4).
    """


def assert_confirmation_rows(ids: Iterable[str], row_half: Mapping[str, str] | pd.Series, what: str) -> None:
    """Raise unless every row id's registered half is the CONFIRMATION half (this run scores no other row)."""
    ids = list(ids)
    get = row_half.get if isinstance(row_half, Mapping) else (lambda r, d=None: row_half.get(r, d))
    bad = [r for r in ids if get(r, "NA") != D.CONFIRMATION]
    if bad:
        halves = sorted({str(get(r, "NA")) for r in bad})
        raise AssertionError(f"{what}: {len(bad)} scored row(s) outside the confirmation half ({halves}; first "
                             f"{bad[0]!r}); the confirmation run scores the confirmation half only")


def is_v6(design: str) -> bool:
    return str(design).upper().replace("-", "").replace("_", "") == "V6"


def v6_cells(systems: Iterable[str]) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """The section 3.4 double cells: per system S the pair ``((Pr(III), S), (Nd(III), S))``, hidden TOGETHER."""
    return [((V6_METALS[0], s), (V6_METALS[1], s)) for s in sorted(set(systems))]


def confirmation_scoring_frame(pred: pd.DataFrame, attrs: pd.DataFrame, *, design: str, v6_mask: pd.Series, what: str,
                               half_col: str | None = None) -> pd.DataFrame:
    """The mirror of ``discovery.scoring_frame`` for this run: one arm's predictions of one design and ONE withheld-seed
    index joined to the row attributes, indexed by ``row_id``, with the CONFIRMATION-half guard where the design has
    halves and :func:`assert_v6_scope` in both directions everywhere.

    Discovery's own refuses a confirmation-half row by construction, so it cannot be reused; every assertion it makes is
    made here with the half switched, and the V6 design takes :data:`READINGS` ``v6_half`` instead of the half filter.
    """
    need = ["row_id", "fold_id", "mean_logD"]
    missing = [c for c in need if c not in pred.columns]
    if missing:
        raise KeyError(f"{what}: prediction columns missing {missing}")
    if "half" in pred.columns and not is_v6(design) and (pred["half"].astype(str) != D.CONFIRMATION).any():
        raise AssertionError(f"{what}: prediction frame holds a row outside the confirmation half")
    fr = pred.join(attrs.drop(columns=[c for c in ("row_id",) if c in attrs.columns]), on="row_id", rsuffix="_attr")
    fr = fr.set_index("row_id", drop=False)
    fr.index.name = None
    if fr.index.has_duplicates:
        raise AssertionError(f"{what}: a row is scored twice in one design and seed index")
    if not is_v6(design):
        hc = half_col or f"registered_half_{design}"
        if hc not in fr.columns:
            raise KeyError(f"{what}: attrs lack {hc}")
        assert_confirmation_rows(fr.index, fr[hc].astype(str), what)
    assert_v6_scope(fr.index, v6_mask, design, what)
    # the metric column every reader expects, as discovery's ``scoring_frame`` sets it: without it
    # ``metrics.design_per_unit_table`` (S1(d), S1(e), the V6 log D macro) and every pair reader raise KeyError('pred')
    fr[EM.PRED_COL] = pd.to_numeric(fr["mean_logD"], errors="coerce").astype(float)
    return fr


def assert_v6_scope(labels: pd.Index, v6_mask: pd.Series, design: str, what: str) -> None:
    """Section 3.4 read in both directions (:data:`READINGS` ``v6_scope``).

    A V6 job scores ONLY ``V6_TARGET_ROWS``; every other design scores NONE of them.  This is the one run where a V6
    row may be scored at all, so the assertion has to be two-sided -- an "is it carved out?" check alone would let the
    carve-out leak into V5 here.
    """
    m = v6_mask.reindex(labels)
    if m.isna().any():
        raise AssertionError(f"{what}: {int(m.isna().sum())} scored row(s) are not in the V6 mask")
    inside = int(m.astype(bool).sum())
    if str(design).upper().replace("-", "") == "V6":
        if inside != len(labels):
            raise AssertionError(f"{what}: a V6 job scored {len(labels) - inside} row(s) outside V6_TARGET_ROWS")
    elif inside:
        raise AssertionError(f"{what}: {inside} V6_TARGET_ROWS row(s) scored under design {design}; section 3.1 carves "
                             "them out of every design and section 3.4 runs V6 once, as its own job")


# --------------------------------------------------------------------------------------------- #
# R19 at confirmation
# --------------------------------------------------------------------------------------------- #

def item4(seed_deltas: Mapping[int, float] | Sequence[float], *, required: int = N_SEEDS_REQUIRED,
          n_seeds: int = N_SEEDS) -> dict[str, Any]:
    """R19 item 4 as registered at confirmation: ``Delta > 0`` in 5 of 5 withheld seeds.

    ``seed_deltas`` is keyed by the OPAQUE seed index (or a sequence in index order).  Fewer than ``n_seeds`` scored
    seeds is a FAIL with the count -- never a pass on the seeds that happen to exist.
    """
    if isinstance(seed_deltas, Mapping):
        items = {int(k): float(v) for k, v in seed_deltas.items()}
    else:
        items = {i + 1: float(v) for i, v in enumerate(seed_deltas)}
    pos = sorted(i for i, v in items.items() if np.isfinite(v) and v > 0)
    ok = len(items) == int(n_seeds) and len(pos) >= int(required)
    return {"item": 4, "name": "seed_sign_agreement", "status": "PASS" if ok else "FAIL",
            "n_seeds_scored": len(items), "n_seeds_expected": int(n_seeds), "n_positive": len(pos),
            "required": int(required), "seed_indices_positive": pos,
            "detail": f"{len(pos)} of {len(items)} scored seeds positive ({n_seeds} expected); confirmation needs "
                      f"{required} of {n_seeds}", "reading": READINGS["item4"]}


def score_claim(claim: Claim, *, point: float, bootstraps: Mapping[str, ET.BootstrapResult],
                seed_deltas: Mapping[int, float], sensitivity_deltas: Mapping[str, float | str],
                deterministic: bool = False, reduced_sensitivities: Sequence[str] | None = None,
                sensitivities_not_run_extra: Mapping[str, str] | None = None) -> dict[str, Any]:
    """R19 (section 8, ``stage='confirmation'``) plus TOST for one frozen claim.

    ``ET.r19`` evaluates items 1-6 with the confirmation reading of item 4 (5 of 5); :func:`item4` is computed beside it
    so the record carries the seed count explicitly, and the two must agree.

    ``reduced_sensitivities`` is POST-HOC addendum 1 item 4's REDUCED set (the refits a learned arm runs plus every
    registered scoring-filter sensitivity).  Item 6 is then rebuilt by ``discovery.reduced_item6`` -- the same builder
    discovery and H3 use -- and the verdict recomputed by ``discovery.r19_verdict``.  Without it ``ET.r19``'s own item-6
    loop reads every registered name and treats any string other than ``transfer.UNTESTABLE`` as a FAILURE, so a
    sensitivity addendum 1 item 4 does not run would FAIL the claim before any data was seen; with it the names that
    were not run are DECLARED (never silently dropped) and a member of the reduced set that this run could not evaluate
    makes item 6 ``NOT_EVALUATED``, not PASS (``discovery.ITEM6_NOT_EVALUATED``).
    """
    sens = dict(sensitivity_deltas)
    for n in ET.REGISTERED_SENSITIVITIES[claim.design]:
        sens.setdefault(n, ET.UNTESTABLE)
    res = ET.r19(design=claim.design, stage="confirmation", point=float(point), margin=float(claim.margin),
                 bootstraps=bootstraps, seed_deltas=[seed_deltas[i] for i in sorted(seed_deltas)],
                 deterministic=bool(deterministic), sensitivity_deltas=sens, contrast=claim.key)
    item6_not_run: dict[str, str] = {}
    if reduced_sensitivities is not None:
        item6, item6_not_run = D.reduced_item6(claim.design, sens, list(reduced_sensitivities),
                                               sensitivities_not_run_extra)
        items = [item6 if it["item"] == 6 else it for it in res.items]
        res = replace(res, items=tuple(items), verdict=D.r19_verdict(items))
    i4 = item4(seed_deltas)
    r19_i4 = next((it for it in res.items if it["item"] == 4), {})
    if not deterministic and r19_i4.get("status") != i4["status"]:
        raise AssertionError(f"{claim.key}: transfer.r19 item 4 {r19_i4.get('status')} disagrees with "
                             f"confirmation.item4 {i4['status']}")
    primary = ET.REGISTERED_CLUSTER_UNITS[claim.design][0]
    t = ET.tost(bootstraps[primary], epsilon=EPSILON) if primary in bootstraps else {"verdict": "NOT_COMPUTED"}
    confirmed = res.verdict == "PASS"
    return {"claim_id": claim.claim_id, "claim": claim.key, "family": claim.family, "candidate": claim.candidate,
            "comparator": claim.comparator, "design": claim.design, "margin": claim.margin, "point": float(point),
            "half": D.CONFIRMATION, "stage": "confirmation", "r19_verdict": res.verdict, "items": list(res.items),
            "item4": i4, "tost": t, "confirmed": bool(confirmed),
            "confirmed_rule": "section 15: a claim is confirmed only if it passes R19 with 5 of 5 confirmation seeds",
            "seed_deltas_by_index": {int(k): float(v) for k, v in sorted(seed_deltas.items())},
            "sensitivity_set": D.ADDENDUM_LABEL if reduced_sensitivities is not None else "registered (full)",
            "sensitivities_reduced_set": list(reduced_sensitivities or ()),
            "sensitivities_not_run": dict(sorted(item6_not_run.items())),
            "sensitivity_deltas": {k: (v if isinstance(v, str) else float(v)) for k, v in sorted(sens.items())},
            "discovery_point_selection_half": claim.discovery_point}


def bh_adjust(p_by_key: Mapping[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg within one family (section 8: printed beside raw p, deciding nothing)."""
    items = [(k, float(v)) for k, v in p_by_key.items() if np.isfinite(v)]
    m = len(items)
    out: dict[str, float] = {k: float("nan") for k in p_by_key}
    if not m:
        return out
    items.sort(key=lambda kv: kv[1])
    prev = 1.0
    for rank in range(m, 0, -1):
        k, p = items[rank - 1]
        prev = min(prev, p * m / rank)
        out[k] = min(1.0, prev)
    return out


# --------------------------------------------------------------------------------------------- #
# S1(c) at confirmation -- the seed mean and its system-cluster bootstrap
# --------------------------------------------------------------------------------------------- #

def seed_mean_system_bootstrap(per_seed_by_system: Mapping[int, Mapping[str, float]], *,
                              n_resamples: int = ET.N_RESAMPLES, seed: int = ET.BOOTSTRAP_SEED) -> dict[str, Any]:
    """The S1(c) / S2(a) confirmation statistic: the mean over the withheld seeds of each seed's system-macro Delta,
    with a system-cluster percentile bootstrap in which **the same resampled systems are applied to every seed**.

    ``per_seed_by_system`` maps the opaque seed index to {system: per-system Delta}.  Systems must be the same set in
    every seed (they are: the confirmation half's scored systems do not depend on a colouring).
    """
    idx = sorted(per_seed_by_system)
    if not idx:
        raise ValueError("no seed given")
    systems = sorted(per_seed_by_system[idx[0]])
    for i in idx:
        if sorted(per_seed_by_system[i]) != systems:
            raise ValueError(f"seed index {i} carries a different system set")
    mat = np.array([[float(per_seed_by_system[i][s]) for s in systems] for i in idx], dtype=float)   # seeds x systems
    per_seed = np.nanmean(mat, axis=1)
    point = float(np.nanmean(per_seed))
    rng = np.random.default_rng(int(seed))
    n = len(systems)
    draws = np.empty(int(n_resamples), dtype=float)
    for b in range(int(n_resamples)):
        take = rng.integers(0, n, size=n)
        draws[b] = float(np.nanmean(np.nanmean(mat[:, take], axis=1)))
    lo, hi = (float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5)))
    return {"point": point, "per_seed": {int(i): float(v) for i, v in zip(idx, per_seed)},
            "percentile_95": [lo, hi], "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
            "n_systems": n, "n_seeds": len(idx), "n_resamples": int(n_resamples), "bootstrap_seed": int(seed),
            "construction": "system-cluster percentile bootstrap of the SEED MEAN, the same resampled systems applied "
                            "to every seed (section 9 S1(c) as redefined; BCa is not registered here)",
            "reading": READINGS["s1c_seed_combination"]}


def s1c_confirmation(direction_by_yardstick: Mapping[str, Mapping[int, Mapping[str, float]]], *,
                     logsf_gain: Mapping[str, float], selection_min_delta: float | None,
                     gamma5: float = GAMMA5, eta5: float = ETA5) -> dict[str, Any]:
    """S1(c) at confirmation: ``min_Y Delta_Y >= gamma5``, every ``Delta_Y``'s interval excluding 0, ``Delta_Y > 0`` in
    5 of 5 seeds, the logSF MAE gains over FLAT and the B3i-derived value ``>= eta5``, and the selection-half
    counterweight ``min_Y Delta_Y >= -0.02`` (already satisfied in discovery)."""
    per: dict[str, Any] = {}
    for y, per_seed in sorted(direction_by_yardstick.items()):
        st = seed_mean_system_bootstrap(per_seed)
        st["item4"] = item4({i: v for i, v in st["per_seed"].items()})
        per[y] = st
    mins = {y: st["point"] for y, st in per.items()}
    min_y = min(mins.values()) if mins else float("nan")
    binding = min(mins, key=lambda k: mins[k]) if mins else None
    dir_ok = bool(mins and min_y >= gamma5 and all(st["excludes_zero"] for st in per.values())
                  and all(st["item4"]["status"] == "PASS" for st in per.values()))
    mag = {k: float(v) for k, v in logsf_gain.items()}
    mag_ok = bool(mag) and all(np.isfinite(v) and v >= eta5 for v in mag.values())
    cw_ok = None if selection_min_delta is None else bool(float(selection_min_delta) >= -0.02)
    verdict = "PASS" if (dir_ok and mag_ok and cw_ok) else ("UNDECIDED" if cw_ok is None else "FAIL")
    return {"per_yardstick": per, "min_delta": min_y, "binding_yardstick": binding, "gamma5": gamma5, "eta5": eta5,
            "direction_pass": dir_ok, "logsf_gain": mag, "logsf_pass": mag_ok,
            "counterweight_selection_half_min_delta": selection_min_delta, "counterweight_pass": cw_ok,
            "verdict": verdict, "reading": READINGS["s1c_seed_combination"],
            "rule": "section 9 S1(c) as redefined 2026-09-15 and its addendum-2 confirmation rule; S1(c) passes only if "
                    "both halves hold, and if V5-PAIR were dropped S1 would be UNDECIDED, never passed"}


# --------------------------------------------------------------------------------------------- #
# S2 -- the single V6 run
# --------------------------------------------------------------------------------------------- #

def s2a_sign_count(observed_median: Mapping[str, float], predicted_median: Mapping[str, float], *,
                   n_systems: int = S2A_N_SYSTEMS, required: int = S2A_MIN_SIGN_SYSTEMS) -> dict[str, Any]:
    """S2(a) first half: the sign of the per-system median predicted logSF_Nd/Pr equals the observed sign in >= 11 of
    the 13 systems.  A zero or non-finite median on either side does NOT agree (:data:`READINGS` ``s2a``)."""
    systems = sorted(set(observed_median) | set(predicted_median))
    rows, agree = [], []
    for s in systems:
        o, p = float(observed_median.get(s, float("nan"))), float(predicted_median.get(s, float("nan")))
        ok = bool(np.isfinite(o) and np.isfinite(p) and o != 0 and p != 0 and np.sign(o) == np.sign(p))
        rows.append({"system": s, "observed_median_logsf": o, "predicted_median_logsf": p, "sign_agrees": ok})
        if ok:
            agree.append(s)
    return {"n_systems_scored": len(systems), "n_systems_expected": int(n_systems), "n_sign_agree": len(agree),
            "required": int(required), "systems_agreeing": agree,
            "status": "PASS" if (len(systems) == int(n_systems) and len(agree) >= int(required)) else "FAIL",
            "per_system": rows, "reading": READINGS["s2a"]}


#: S2(a) second half (section 9): the paired direction margin and the four registered yardsticks.  B8 is defined only
#: in systems with TOPO39 columns, so its pair set is smaller -- which is exactly why the rule is PAIRED per yardstick
#: rather than a max over yardsticks scored on different pairs (the 2026-09-15 resolution).
S2A_DIRECTION_MARGIN = 0.05
S2A_YARDSTICKS: tuple[str, ...] = ("HEAVIER", "B3x_derived", "B3i_derived", "B8")
#: section 9 S2(a): min_Y Delta_Y >= 0.05 "in pooled pairs AND in the HNO3 pairs (142 of 209)"
S2A_STRATA: tuple[str, ...] = ("pooled", "HNO3")
V6_N_PAIRS, V6_N_PAIRS_HNO3 = 209, 142
#: section 3.4's sensitivity: the 7-system setting (>= 10 Pr and Nd rows, >= 5 other Ln(III); 168 pairs)
V6_SENSITIVITY_N_SYSTEMS, V6_SENSITIVITY_N_PAIRS = 7, 168
#: section 3.4: "TODGA is reported on its own as well"
V6_FOCUS_SYSTEM = "TODGA"


def s2a_direction(per_yardstick_per_seed_by_system: Mapping[str, Mapping[int, Mapping[str, float]]], *,
                  margin: float = S2A_DIRECTION_MARGIN, stratum: str = "pooled",
                  yardsticks: Sequence[str] = S2A_YARDSTICKS) -> dict[str, Any]:
    """S2(a) second half: pair-level direction accuracy (|observed| >= 0.1) beats EVERY yardstick by >= 0.05.

    Section 9 S2(a) as resolved 2026-09-15: for each ``Y`` in :data:`S2A_YARDSTICKS`, ``Delta_Y`` = accuracy(M2) -
    accuracy(Y) on the identical V6 test-test pairs where Y is defined, with B3x, B3i and B8 fitted on the same V6 folds
    and withheld seeds as M2 and **the same seed combination** as S1(c) (:func:`seed_mean_system_bootstrap`); the test is
    ``min_Y Delta_Y >= 0.05``, applied in the pooled pairs AND in the HNO3 pairs.  A yardstick that is not scored on any
    seed is UNTESTABLE and S2(a) is then UNDECIDED, never passed: the rule says *every* yardstick.
    """
    per: dict[str, Any] = {}
    missing: list[str] = []
    for y in yardsticks:
        block = per_yardstick_per_seed_by_system.get(y)
        if not block:
            missing.append(y)
            continue
        st = seed_mean_system_bootstrap(block)
        st["item4"] = item4({i: v for i, v in st["per_seed"].items()}, n_seeds=len(st["per_seed"]) or N_SEEDS)
        per[y] = st
    mins = {y: st["point"] for y, st in per.items()}
    min_y = min(mins.values()) if mins else float("nan")
    binding = min(mins, key=lambda k: mins[k]) if mins else None
    ok = bool(mins and not missing and np.isfinite(min_y) and min_y >= float(margin)
              and all(st["excludes_zero"] for st in per.values())
              and all(st["item4"]["status"] == "PASS" for st in per.values()))
    status = "PASS" if ok else (UNDECIDED if missing else "FAIL")
    return {"stratum": stratum, "per_yardstick": per, "min_delta": min_y, "binding_yardstick": binding,
            "margin": float(margin), "yardsticks_not_scored": missing, "status": status,
            "detail": (f"min_Y Delta_Y = {min_y:.6g} over {sorted(mins)} (binding {binding}); margin {margin:g}"
                       if mins else "no yardstick scored")
                      + (f"; UNTESTABLE yardstick(s) {missing}: S2(a) is UNDECIDED, never passed" if missing else ""),
            "reading": READINGS["s2a"]}


#: the two constructions of a predicted logSF interval that POST-HOC addendum 7 item 1 registers: ``pair_conformal``
#: (the REGISTERED value -- split conformal fitted on the PAIR residuals of the fold's inner calibration comparable
#: pairs, with the section 12 finite-sample quantile) and ``quadrature`` (the convolution of the two row intervals under
#: independence, half-widths added in root-sum-square), which is printed beside it and is also the per-fold FALLBACK
#: when the fold's inner calibration set holds fewer than :data:`S2C_MIN_CALIBRATION_PAIRS` comparable pairs.
S2C_INTERVAL_READINGS: tuple[str, ...] = ("pair_conformal", "quadrature")
S2C_PRIMARY_READING = "pair_conformal"
#: addendum 7 item 1: "a fold whose inner calibration set holds fewer than 20 comparable pairs falls back to the
#: convolution of the two row intervals under independence (half-widths added in quadrature), and every such fold is
#: flagged and counted in the coverage table"
S2C_MIN_CALIBRATION_PAIRS = 20
S2C_PAIR_CONFORMAL = "pair_conformal"
S2C_QUADRATURE_FALLBACK = "quadrature_fallback"


def pair_calibration_residuals(detail: Sequence[Mapping[str, Any]], rows: pd.DataFrame, *, what: str) -> dict[str, Any]:
    """The absolute PAIR residuals of one outer fold's inner calibration set (POST-HOC addendum 7 item 1).

    ``detail`` is ``ConformalWrapper.calibration_detail`` -- one entry per inner split, carrying that split's
    calibration rows and their SIGNED residuals ``prediction - observed`` (``interface.calibration_detail_of``).  Each
    entry's rows are read from ``row_ids`` when the caller has translated the table's index labels into row ids, else
    from ``labels``.  ``rows`` is an attribute frame indexed the same way, carrying the section 2 pair key columns
    (``pairs.PAIR_KEY_COLS``), the metal state and the observed log D.

    The comparable pairs are section 2's -- same publication group, system and condition key -- formed WITHIN one inner
    split, exactly as the outer pairs are formed within one fold: a pair whose members came from two different inner
    splits would combine two differently fitted models.  A pair's residual is
    ``(pred_a - pred_b) - (y_a - y_b) = signed_a - signed_b``, so the absolute pair residual needs no second prediction
    pass and is algebraically the deviation of the predicted logSF from the observed logSF.
    """
    from gen19ct.evaluation import pairs as EP

    need = list(EP.PAIR_KEY_COLS) + [EM.METAL_STATE_COL, EM.Y_COL]
    missing = [c for c in need if c not in rows.columns]
    if missing:
        raise KeyError(f"{what}: the calibration rows lack {missing}; a pair calibration cannot be fitted")
    frames, signed, per_split = [], {}, []
    for k, sp in enumerate(detail):
        ids = [str(r) for r in (sp.get("row_ids") if sp.get("row_ids") is not None else sp["labels"])]
        sig = np.asarray(sp["signed"], dtype=float)
        if len(ids) != len(sig):
            raise AssertionError(f"{what}: inner split {sp.get('unit')!r} has {len(ids)} rows and {len(sig)} residuals")
        miss = [r for r in ids if r not in rows.index]
        if miss:
            raise KeyError(f"{what}: {len(miss)} calibration row(s) are not in the attribute frame (first {miss[0]!r})")
        sub = rows.loc[ids, need].copy()
        labels = [f"s{k}{ET.FOLD_LABEL_SEP}{r}" for r in ids]
        sub.index = pd.Index(labels)
        sub["fold"] = f"s{k}"
        frames.append(sub)
        signed.update(dict(zip(labels, sig.tolist())))
        per_split.append({"inner_split": str(sp.get("unit")), "inner_fold": int(sp.get("fold", -1)),
                          "n_calibration_rows": len(ids)})
    if not frames:
        return {"abs_residuals": np.zeros(0, dtype=float), "n_pairs": 0, "n_calibration_rows": 0,
                "n_inner_splits": 0, "per_split": [],
                # the same keys the populated return carries, so no caller has to branch (v6_pair_conformal returns
                # before this on an empty detail, but a direct caller must not meet a KeyError)
                "population": _pair_population(pd.DataFrame(), pd.DataFrame(), np.zeros(0))[0],
                "prnd_only_abs_residuals": np.zeros(0, dtype=float),
                "population_reading": READINGS["s2c_calibration_population"]}
    df = pd.concat(frames)
    cp = EP.comparable_pairs(df, fold_col="fold")
    s = pd.Series(signed, dtype=float)
    res = np.abs(cp["idx_a"].map(s).to_numpy(dtype=float) - cp["idx_b"].map(s).to_numpy(dtype=float))
    if len(res) and not np.isfinite(res).all():
        raise AssertionError(f"{what}: a non-finite pair residual in the inner calibration set")
    counts = cp["fold"].astype(str).value_counts().to_dict() if len(cp) else {}
    for i, row in enumerate(per_split):
        row["n_pairs"] = int(counts.get(f"s{i}", 0))
    comp, prnd = _pair_population(cp, df, res)
    return {"abs_residuals": res, "n_pairs": int(len(res)), "n_calibration_rows": int(len(df)),
            "n_inner_splits": len(frames), "per_split": per_split,
            "population": comp, "prnd_only_abs_residuals": prnd,
            "population_reading": READINGS["s2c_calibration_population"]}


def _pair_population(cp: pd.DataFrame, df: pd.DataFrame, res: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """What the inner calibration comparable pairs ARE, by unordered metal-state pair, and the Pr/Nd subset of them.

    POST-HOC addendum 7 item 1 registers the calibration population as "the comparable pairs of the fold's inner
    calibration set", which is section 2's pair key (publication group, system, condition key) and therefore ANY two
    distinct metal states -- Am/Eu counts exactly as much as Nd/Pr.  The pairs S2(c) then SCORES are Nd/Pr only, whose
    |logSF| is small, so a calibration set dominated by wider-separated pairs gives a pair quantile that is too LARGE
    and an interval biased WIDE: coverage is pushed up, and the registered [0.70, 0.90] band at 80 % can FAIL on the
    HIGH side as an artefact of the population rather than as miscalibration.  The code is as registered and is not
    changed here; the composition and the Pr/Nd-restricted residuals are recorded so the consequence is a number rather
    than a caveat, and the Pr/Nd-only construction is an EXPLORATORY diagnostic that decides nothing.
    """
    if not len(cp):
        return {"n_pairs": 0, "n_prnd_pairs": 0, "by_state_pair": {}, "prnd_share": None,
                "median_abs_residual": None, "median_abs_residual_prnd": None}, np.zeros(0, dtype=float)
    sa = cp["idx_a"].map(df[EM.METAL_STATE_COL]).astype(str).to_numpy()
    sb = cp["idx_b"].map(df[EM.METAL_STATE_COL]).astype(str).to_numpy()
    keys = [" | ".join(sorted((a, b))) for a, b in zip(sa, sb)]
    want = " | ".join(sorted(V6_METALS))
    is_prnd = np.asarray([k == want for k in keys], dtype=bool)
    by: dict[str, int] = {}
    for k in keys:
        by[k] = by.get(k, 0) + 1
    return ({"n_pairs": int(len(keys)), "n_prnd_pairs": int(is_prnd.sum()),
             "prnd_state_pair": want, "prnd_share": float(is_prnd.mean()),
             "by_state_pair": dict(sorted(by.items(), key=lambda kv: (-kv[1], kv[0]))[:20]),
             "n_distinct_state_pairs": len(by),
             "median_abs_residual": float(np.median(res)) if len(res) else None,
             "median_abs_residual_prnd": float(np.median(res[is_prnd])) if is_prnd.any() else None},
            res[is_prnd])


def prnd_only_quantiles(abs_residuals: Sequence[float], *, min_pairs: int = S2C_MIN_CALIBRATION_PAIRS
                        ) -> dict[str, Any]:
    """EXPLORATORY: the same quantile rule on the Pr/Nd-only subset of the calibration pairs.

    Not registered, not a verdict, and not one of :data:`S2C_INTERVAL_READINGS`.  It exists so that the artefact
    ``READINGS['s2c_calibration_population']`` names is a number: if these quantiles are materially smaller than the
    registered ones, the registered interval is wide because of the calibration population and a high-side 80 % FAIL is
    that, not miscalibration.
    """
    from gen19ct.models import interface as I

    r = np.asarray(list(abs_residuals), dtype=float)
    n = int(len(r))
    out: dict[str, Any] = {"n_calibration_pairs": n, "min_calibration_pairs": int(min_pairs),
                           "label": "EXPLORATORY diagnostic, not registered and not the S2(c) verdict",
                           "reading": READINGS["s2c_calibration_population"]}
    out["quantiles"] = {} if n < int(min_pairs) else {str(int(round(lv * 100))): float(I.conformal_quantile(r, lv))
                                                      for lv in I.LEVELS}
    return out


def pair_conformal_quantiles(abs_residuals: Sequence[float], *, min_pairs: int = S2C_MIN_CALIBRATION_PAIRS
                             ) -> dict[str, Any]:
    """The section 12 finite-sample split-conformal quantile of the pair residuals at each registered level, or the
    declared fallback when there are fewer than ``min_pairs`` of them (POST-HOC addendum 7 item 1).

    ``method`` is :data:`S2C_PAIR_CONFORMAL` or :data:`S2C_QUADRATURE_FALLBACK`; ``fallback`` is the flag the coverage
    table counts.  A fallback fold carries no pair quantile at all -- its pair interval IS the quadrature one -- so the
    two constructions are never silently mixed inside one number without the count saying so.
    """
    from gen19ct.models import interface as I

    r = np.asarray(list(abs_residuals), dtype=float)
    n = int(len(r))
    fallback = n < int(min_pairs)
    out = {"n_calibration_pairs": n, "min_calibration_pairs": int(min_pairs), "fallback": bool(fallback),
           "method": S2C_QUADRATURE_FALLBACK if fallback else S2C_PAIR_CONFORMAL,
           "quantile_rule": "interface.conformal_quantile: the ceil((n + 1) * level)-th smallest absolute PAIR residual "
                            "(the finite-sample quantile section 12 registers for rows)",
           "reading": READINGS["s2c_interval_reading"]}
    if fallback:
        out["quantiles"] = {}
        return out
    out["quantiles"] = {str(int(round(lv * 100))): float(I.conformal_quantile(r, lv)) for lv in I.LEVELS}
    return out


def s2c_coverage(coverage_by_reading: Mapping[str, Mapping[str, float]], *,
                 primary: str = S2C_PRIMARY_READING, fallback_folds: Mapping[str, Any] | None = None,
                 population: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """S2(c): pooled logSF interval coverage in [0.70, 0.90] at 80 % and >= 0.88 at 95 % over the V6 pairs.

    ``coverage_by_reading`` maps each construction of :data:`S2C_INTERVAL_READINGS` to {"80": cov, "95": cov}.  The
    verdict is the registered ``primary`` reading's (``pair_conformal``, POST-HOC addendum 7 item 1); the other is
    printed, and ``readings_disagree`` says whether the choice mattered.

    ``fallback_folds`` is the addendum's flag and count: how many folds (and which, and how many of their pairs) fell
    back to the quadrature construction because their inner calibration set held fewer than
    :data:`S2C_MIN_CALIBRATION_PAIRS` comparable pairs.  It is carried into the verdict block so the coverage table
    cannot show a pair-conformal number without saying how much of it is the fallback.

    ``population`` is the recorded consequence of the registered calibration population
    (``READINGS['s2c_calibration_population']``): the calibration pairs are ANY two distinct metal states of a condition
    group while the SCORED pairs are Nd/Pr only, so the interval is biased wide and a high-side 80 % FAIL may be that
    artefact.  It is carried into the verdict block for the same reason the fallback count is -- so the number cannot be
    read without it -- and it changes no verdict.
    """
    per = {r: coverage_bands(c, S2C_BANDS, min_95=S2C_MIN_95) for r, c in sorted(coverage_by_reading.items())}
    flag = {"n_folds": 0, "n_pairs": 0, "min_calibration_pairs": S2C_MIN_CALIBRATION_PAIRS, **dict(fallback_folds or {})}
    pop = {"reading": READINGS["s2c_calibration_population"], **dict(population or {})}
    if primary not in per:
        return {"status": NOT_EVALUATED, "per_reading": per, "primary_reading": primary,
                "quadrature_fallback": flag, "calibration_population": pop,
                "detail": f"the primary reading {primary!r} was not computed; S2(c) carries no verdict",
                "reading": READINGS["s2c_interval_reading"]}
    statuses = {r: v["status"] for r, v in per.items()}
    nfb = int(flag.get("n_folds") or 0)
    share = pop.get("prnd_share")
    # is the 80 % band missed on the HIGH side?  That is the direction a wide-biased interval fails in, so it is the one
    # the population note must be read with (a low-side miss is NOT explained by the population)
    r80 = next((r for r in per[primary]["per_level"] if str(r["level"]) == "80" and r["band"][1] <= 1.0
                and r["band"][0] == S2C_BANDS["80"][0]), None)
    wide = None if r80 is None or bool(r80["in_band"]) else bool(float(r80["coverage"]) > float(r80["band"][1]))
    note = ""
    if share is not None:
        note = (f"; {float(share):.1%} of the inner calibration comparable pairs are the scored Pr/Nd state pair, the "
                f"rest are other metal-state pairs of the same condition groups, so the registered interval is biased "
                f"WIDE and a high-side 80 % FAIL may be that artefact (calibration_population)")
    return {"status": per[primary]["status"], "per_reading": per, "primary_reading": primary,
            "coverage": dict(coverage_by_reading.get(primary) or {}), "statuses_by_reading": statuses,
            "readings_disagree": len(set(statuses.values())) > 1, "quadrature_fallback": flag,
            "calibration_population": pop, "high_side_80": wide,
            "bands": {k: list(v) for k, v in S2C_BANDS.items()}, "min_95": S2C_MIN_95,
            "detail": per[primary]["detail"]
                      + (f"; {nfb} fold(s) with < {S2C_MIN_CALIBRATION_PAIRS} calibration pairs fell back to quadrature "
                         f"({flag.get('n_pairs')} of the scored pairs)" if nfb else "; no fold used the fallback")
                      + note,
            "reading": READINGS["s2c_interval_reading"]}


def s2b_magnitude(*, logsf_mae: float, flat_logsf_mae: float, lookup_logsf_mae: float,
                  gain_interval_excludes_zero: bool | None, logd_macro_mae: float, v5_lookup_comparator_mae: float,
                  margin: float = S2B_MARGIN) -> dict[str, Any]:
    """S2(b): logSF MAE <= FLAT - 0.02 with the 13-system percentile interval of the gain over FLAT excluding 0, also
    <= the lookup-derived logSF MAE, and the macro log D MAE of the hidden Pr / Nd rows <= the V5 lookup comparator."""
    gain = float(flat_logsf_mae) - float(logsf_mae)
    beats_flat = bool(np.isfinite(gain) and gain >= float(margin))
    beats_lookup = bool(np.isfinite(logsf_mae) and logsf_mae <= float(lookup_logsf_mae))
    logd_ok = bool(np.isfinite(logd_macro_mae) and logd_macro_mae <= float(v5_lookup_comparator_mae))
    parts = {"logsf_mae": float(logsf_mae), "flat_logsf_mae": float(flat_logsf_mae), "gain_over_flat": gain,
             "margin": float(margin), "beats_flat_by_margin": beats_flat,
             "gain_interval_excludes_zero": gain_interval_excludes_zero,
             "lookup_derived_logsf_mae": float(lookup_logsf_mae), "at_most_lookup": beats_lookup,
             "logd_macro_mae_hidden_pr_nd": float(logd_macro_mae),
             "v5_lookup_comparator_mae": float(v5_lookup_comparator_mae), "logd_at_most_lookup": logd_ok}
    ok = beats_flat and beats_lookup and logd_ok and bool(gain_interval_excludes_zero)
    parts["status"] = "PASS" if ok else ("UNDECIDED" if gain_interval_excludes_zero is None else "FAIL")
    return parts


def coverage_bands(coverage: Mapping[str, float], bands: Mapping[str, tuple[float, float]], *,
                   min_95: float | None = None) -> dict[str, Any]:
    """S1(d) / S2(c): every registered coverage band, with the value that decides each one."""
    rows, fails = [], []
    for lvl, (lo, hi) in sorted(bands.items()):
        v = float(coverage.get(lvl, float("nan")))
        ok = bool(np.isfinite(v) and lo <= v <= hi)
        rows.append({"level": lvl, "coverage": v, "band": [lo, hi], "in_band": ok})
        if not ok:
            fails.append(f"{lvl}: {v:.4g} outside [{lo}, {hi}]")
    if min_95 is not None:
        v = float(coverage.get("95", float("nan")))
        ok = bool(np.isfinite(v) and v >= float(min_95))
        rows.append({"level": "95", "coverage": v, "band": [float(min_95), 1.0], "in_band": ok})
        if not ok:
            fails.append(f"95: {v:.4g} below {min_95}")
    return {"per_level": rows, "status": "FAIL" if fails else "PASS", "detail": "; ".join(fails) or "every band met"}


def s1d_calibration(coverage: Mapping[str, float], by_category: Mapping[str, Mapping[str, float]] | None = None
                    ) -> dict[str, Any]:
    """S1(d): the three macro bands over the confirmation half's V5-primary cells, plus 80 % in [0.65, 0.92] in every
    domain-status category with >= 20 scored cells (a category with fewer is printed with its count and decides
    nothing)."""
    out = {"macro": coverage_bands(coverage, S1D_BANDS), "by_category": [], "min_cells": S1D_CATEGORY_MIN_CELLS}
    fails = [] if out["macro"]["status"] == "PASS" else ["macro bands"]
    for cat, c in sorted((by_category or {}).items()):
        n = int(c.get("n_cells", 0))
        v = float(c.get("80", float("nan")))
        decides = n >= S1D_CATEGORY_MIN_CELLS
        ok = bool(np.isfinite(v) and S1D_CATEGORY_80[0] <= v <= S1D_CATEGORY_80[1])
        out["by_category"].append({"category": cat, "n_cells": n, "coverage_80": v, "band": list(S1D_CATEGORY_80),
                                   "decides": decides, "in_band": ok})
        if decides and not ok:
            fails.append(f"{cat}: 80 % {v:.4g}")
    out["status"] = "FAIL" if fails else "PASS"
    out["detail"] = "; ".join(fails) or "every registered band met"
    return out


# --------------------------------------------------------------------------------------------- #
# the idempotence lock (section 15: "nothing is re-run")
# --------------------------------------------------------------------------------------------- #

def lock_verdict(out_root: Path | str, *, resume: bool, code_digest: str, claim_ids: Sequence[str]) -> dict[str, Any]:
    """Whether this invocation may proceed (:data:`READINGS` ``idempotence``).

    First run: neither ``decisions/confirmation.json`` nor the START marker (:func:`started_path`) exists -> proceed.
    Second run: refuse, unless ``--resume``, and then only to COMPLETE unfitted folds -- the recorded code digest must
    be the live one and the claim list must not grow, so no claim is ever re-scored under different code and no sixth
    claim appears.

    The marker is consulted because ``decisions/confirmation.json`` is written LAST: a run that died after writing
    records left the lock reading ``first_run``, so a second invocation could have mixed record sets written under two
    code digests into one score (task X finding).  Whichever of the two exists governs; when both do, the decisions
    file wins, because it is the completed run's own record of the code and claims.
    """
    p = decisions_path(out_root)
    started = started_path(out_root)
    if not p.exists() and not started.exists():
        return {"ok": True, "mode": "first_run", "decisions": str(p), "started_marker": str(started),
                "reading": READINGS["idempotence"]}
    if not p.exists():
        prev = json.loads(started.read_text(encoding="utf-8"))
        prev_code, prev_ids = str(prev.get("code_digest", "")), list(prev.get("claim_ids") or [])
        if not resume:
            return {"ok": False, "mode": "already_started", "decisions": str(p), "started_marker": str(started),
                    "started_utc": prev.get("started_utc"),
                    "reason": f"{started} exists: the single registered run of section 15 has already STARTED (and did "
                              "not finish -- decisions/confirmation.json was never written). It is registered to happen "
                              "ONCE. Pass --resume to COMPLETE its unfitted folds under the same code digest; nothing "
                              "else is permitted, and no record is deleted or re-scored.",
                    "reading": READINGS["idempotence"]}
        if prev_code and prev_code != str(code_digest):
            return {"ok": False, "mode": "resume_refused_code_changed", "decisions": str(p),
                    "started_marker": str(started), "recorded_code_digest": prev_code,
                    "live_code_digest": str(code_digest),
                    "reason": "--resume may only complete unfitted folds: the code digest of the STARTED run differs "
                              "from the live one, so the records already on disk and the ones a resume would write "
                              "would be two different code sets scored as one. Nothing is re-scored and nothing is "
                              "deleted.", "reading": READINGS["idempotence"]}
        extra = sorted(set(claim_ids) - set(prev_ids))
        if extra:
            return {"ok": False, "mode": "resume_refused_claims_grew", "decisions": str(p),
                    "started_marker": str(started), "new_claims": extra,
                    "reason": f"--resume may not widen the claim list; {extra} are not in the started run ({prev_ids}).",
                    "reading": READINGS["idempotence"]}
        return {"ok": True, "mode": "resume_unfinished", "decisions": str(p), "started_marker": str(started),
                "recorded_code_digest": prev_code, "claim_ids": prev_ids,
                "may_only": "complete unfitted folds of the started run", "reading": READINGS["idempotence"]}
    prev = json.loads(p.read_text(encoding="utf-8"))
    prev_code = str(prev.get("code_digest", ""))
    prev_ids = list(prev.get("claim_ids") or [])
    if not resume:
        return {"ok": False, "mode": "already_run", "decisions": str(p),
                "reason": f"{p} exists: the confirmation run is registered to happen ONCE (section 15: 'Whatever "
                          "confirmation returns is the result. No sixth claim is added and nothing is re-run'). Pass "
                          "--resume to COMPLETE unfitted folds; nothing else is permitted.",
                "reading": READINGS["idempotence"]}
    if prev_code and prev_code != str(code_digest):
        return {"ok": False, "mode": "resume_refused_code_changed", "decisions": str(p),
                "recorded_code_digest": prev_code, "live_code_digest": str(code_digest),
                "reason": "--resume may only complete unfitted folds: the code digest of the recorded run differs from "
                          "the live one, so scoring a claim now would score it under different code. Nothing is "
                          "re-scored and nothing is deleted.", "reading": READINGS["idempotence"]}
    extra = sorted(set(claim_ids) - set(prev_ids))
    if extra:
        return {"ok": False, "mode": "resume_refused_claims_grew", "decisions": str(p), "new_claims": extra,
                "reason": f"--resume may not widen the claim list; {extra} are not in the recorded run ({prev_ids}).",
                "reading": READINGS["idempotence"]}
    return {"ok": True, "mode": "resume", "decisions": str(p), "recorded_code_digest": prev_code,
            "claim_ids": prev_ids, "may_only": "complete unfitted folds", "reading": READINGS["idempotence"]}


# --------------------------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------------------------- #

def gate(out_root: Path | str, *, seed_store: SeedStore | None, code_digest: str, resume: bool = False,
         check: Any = None, digests: Any = None, expect_addenda: int | None = None,
         plan: Mapping[str, Any] | None = None, registry_path: Path | None = None) -> dict[str, Any]:
    """The five gates of the confirmation runner, in order, each refusing with what is missing:

    (a) the seal check and the registered below-footer digest of stage ``confirmation``
        (``registry.refuse_unless_sealed``);
    (b) the registry holds that stage and this code is the code it was registered with
        (``registry.refuse_unless_writable``);
    (c) ``decisions/CONFIRMATION_PLAN.md`` is present and freezes at most 5 claims;
    (d) a ``--seed-store`` that verifies against the committed digest;
    (e) the idempotence lock.
    """
    out_root = Path(out_root)
    prereg = REG.refuse_unless_sealed(STAGE, check, digests, expect_addenda=expect_addenda, path=registry_path)
    entry = REG.registered(STAGE, registry_path)
    if entry is None:
        raise SystemExit(f"refused: the registry holds no '{STAGE}' stage; register it before the run "
                         f"(python -m gen19ct.evaluation.registry register --stage {STAGE} --note ...)")
    writable = REG.refuse_unless_writable(STAGE, str(code_digest), registry_path)
    pl = dict(read_plan(plan_path(out_root))) if plan is None else dict(plan)
    if seed_store is None:
        raise SystemExit("refused: the confirmation run needs --seed-store PATH; the 5 withheld seeds enter only "
                         "through it (section 15)")
    if not seed_store.verified or seed_store.digest != committed_digest(out_root):
        raise SystemExit("refused: the seed store does not verify against manifests/confirmation_seeds_sha256.txt")
    lock = lock_verdict(out_root, resume=resume, code_digest=code_digest,
                        claim_ids=[c.claim_id for c in pl["claims"]])
    if not lock["ok"]:
        raise SystemExit("refused: " + str(lock.get("reason")))
    return {"prereg": prereg, "registry_entry": entry, "writable": writable, "lock": lock,
            "plan": {k: v for k, v in pl.items() if k != "claims"},
            "claims": [c.claim_id for c in pl["claims"]], "seed_store": seed_store.public(),
            "stage": STAGE, "code_digest": str(code_digest), "readings": dict(READINGS),
            "not_run": {"v6_actinide_deltas": V6_ACTINIDE_DELTAS_NOT_RUN, "power_check": POWER_CHECK_NOT_RUN}}


# --------------------------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------------------------- #

def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def write_decisions(out_root: Path | str, body: Mapping[str, Any]) -> Path:
    p = decisions_path(out_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json(p, {"schema": DECISION_SCHEMA, "written_utc": _now(), **dict(body)})
    return p


def confirmation_tables(decisions: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """``tables/confirmation_*.csv`` -- one row per claim, per R19 item and per NOT_RUN entry."""
    claims = decisions.get("claims") or []
    rows = []
    for c in claims:
        sd = c.get("seed_deltas_by_index") or {}
        rows.append({"claim_id": c.get("claim_id"), "claim": c.get("claim"), "family": c.get("family"),
                     "design": c.get("design"), "candidate": c.get("candidate"), "comparator": c.get("comparator"),
                     "half": c.get("half"), "margin": c.get("margin"), "point": c.get("point"),
                     "r19_verdict": c.get("r19_verdict"), "confirmed": c.get("confirmed"),
                     "n_seeds_positive": (c.get("item4") or {}).get("n_positive"),
                     "n_seeds_scored": (c.get("item4") or {}).get("n_seeds_scored"),
                     "tost_verdict": (c.get("tost") or {}).get("verdict"),
                     "seed_deltas_by_index": "; ".join(f"i{k}={v:.6g}" for k, v in sorted(sd.items())),
                     "discovery_point_selection_half": c.get("discovery_point_selection_half")})
    items = [{"claim_id": c.get("claim_id"), "claim": c.get("claim"), "item": it.get("item"), "name": it.get("name"),
              "status": it.get("status"), "detail": it.get("detail")}
             for c in claims for it in (c.get("items") or [])]
    nr = [{"what": k, **{kk: vv for kk, vv in v.items() if isinstance(vv, (str, int, float))}}
          for k, v in sorted((decisions.get("not_run") or {}).items())]
    return {"confirmation_claims": pd.DataFrame(rows), "confirmation_r19_items": pd.DataFrame(items),
            "confirmation_not_run": pd.DataFrame(nr)}


def confirmation_report(decisions: Mapping[str, Any], *, store: SeedStore | None = None,
                        verify_seeds: Mapping[str, Any] | None = None) -> str:
    """``decisions/CONFIRMATION.md`` -- the result, the REVEALED seeds with the ``--verify-seeds`` verdict, and what was
    not run.

    Section 15: *"They are revealed in decisions/CONFIRMATION.md together with the --verify-seeds verdict"*, and POST-HOC
    addendum 6 item 4 fixes WHEN: only after the single run has completed (:data:`SEED_REVELATION`).  ``store`` is passed
    ONLY by the runner, at the very end, after every other artefact has been written and scanned; without it the values
    are withheld and the file says so, which is what the pre-run leak test reads.
    """
    ss = decisions.get("seed_store") or {}
    vs = dict(verify_seeds or {})
    lines = ["# CONFIRMATION -- the single registered confirmation run (pre-registration section 15)", "",
             f"*Written {decisions.get('written_utc', _now())} by `scripts/g19_run_confirmation.py`. "
             "POST-HOC addendum 5 item 1 makes this the CORE run: the frozen claims on the confirmation half with the 5 "
             "withheld seeds, the single V6 run with S2(a)-(c), S1(c) and S1(d). Whatever it returns is the result "
             "(section 15).*", "",
             "## The withheld seeds", "",
             f"- commitment (section 15): `{ss.get('commitment_sha256', '?')}`",
             f"- `scripts/g19_seal_prereg.py --verify-seeds`: **"
             f"{'VERIFIED' if (vs.get('ok') if vs else ss.get('verified_against_commitment')) else 'NOT VERIFIED'}** "
             f"({ss.get('n_seeds', '?')} seeds)"
             + (f" -- {vs.get('message')}" if vs.get("message") else "") + ".",
             ""]
    if store is None:
        lines += [f"- the VALUES are not printed here (this file was written before the run completed). "
                  f"{ss.get('seed_revelation', SEED_REVELATION)}", ""]
    else:
        lines += ["| opaque index | withheld seed |", "|---|---|"]
        lines += [f"| i{i} | **{store.seed(i)}** |" for i in store.indices()]
        lines += ["", "*Revealed here and nowhere else: every fold file, record, table and log line of the run carries "
                  "the opaque index and the commitment digest alone. " + SEED_REVELATION + "*", ""]
    lines += ["## The frozen claims on the confirmation half", "",
             "| claim | design | Delta | margin | R19 | item 4 (seeds) | TOST | confirmed |", "|---|---|---|---|---|---|---|---|"]
    for c in decisions.get("claims") or []:
        i4 = c.get("item4") or {}
        lines.append(f"| {c.get('claim_id')} {c.get('claim')} | {c.get('design')} | "
                     f"{c.get('point', float('nan')):+.6g} | {c.get('margin')} | **{c.get('r19_verdict')}** | "
                     f"{i4.get('n_positive')} of {i4.get('n_seeds_scored')} | "
                     f"{(c.get('tost') or {}).get('verdict')} | **{'yes' if c.get('confirmed') else 'no'}** |")
    for key, title in (("s1", "## S1 -- the scientific claim (H1, V5)"), ("s2", "## S2 -- the applied claim (V6, once)")):
        block = decisions.get(key)
        if block:
            lines += ["", title, "", "```json", json.dumps(block, indent=1, default=str)[:4000], "```"]
    lines += ["", "## What this run did NOT do, with its cost and its consequence", ""]
    for k, v in sorted((decisions.get("not_run") or {}).items()):
        lines += [f"- **{k}: {v.get('status', NOT_RUN)}** -- {v.get('what')}.",
                  f"  - not run by: {v.get('not_run_by')}",
                  f"  - cost: {v.get('cost_hours_serial', v.get('inventory'))}",
                  f"  - consequence: {v.get('consequence')}"]
    leak = decisions.get("seed_leak_scan") or {}
    lines += ["", "## Seed hygiene", "",
              f"- files scanned for a withheld seed's decimal form: {leak.get('files_scanned', '?')}; "
              f"verdict **{'clean' if leak.get('ok') else 'LEAK'}**"
              + ("" if leak.get("ok") else f" -- {leak.get('files_containing_a_withheld_seed')}"), ""]
    return "\n".join(lines)

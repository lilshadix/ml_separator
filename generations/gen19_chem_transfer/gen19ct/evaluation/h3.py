"""``evaluation/h3.py`` -- the section 11 actinide ablation (H3; brief sections 15 and 30) and the section 10 F4 check.

Nothing here fits a model or reads a discovery record to decide anything: ``scripts/g19_run_h3.py`` fits the ablation
arms AFTER discovery is complete and calls the functions below; the readers verify records against the digests the
current code produces, as ``gen19ct.evaluation.discovery`` does.

Design (section 11, implemented as follows)
-------------------------------------------
* **Model arms** (:func:`model_arms`): the configuration deployed for lanthanide prediction
  (:func:`deployed_configuration`, read from the ladder runner's ``evaluation/ladder/decisions/ladder.json`` AND the
  scorer's ``decisions.json``: the highest kept ladder step M7 > M6 > M5 > M4 > M3 (ladder.json, a registered -- not
  stop-rule exploratory -- run) > M2 > M1 (decisions.json), else M2 when it passes the stop-rule scope against B3i, else
  B6 when it does, else the registered V5 lookup comparator B3i as the F6 "best-passing baseline") plus B6 and B5 as
  transparent references (``discovery.H3_MODEL_ARMS``).  The rule is undecidable (the runner refuses) until the ladder
  has run every step to a done / skipped status (:func:`ladder_complete`; task X finding V-01).
* **Transforms** (:data:`TRANSFORMS`, ``discovery.h3_training_rows``): WITH is the discovery record of the same arm,
  design, seed and fold (same architecture, same folds and batches, same seeds -- nothing is refitted for it); WITHOUT
  removes every actinide training row (unknown-state actinide rows included); ACT_PERMUTED permutes ``log_D`` among
  actinide training rows within (system, publication group); ACT_METAL_SHUFFLED shuffles the actinide metal-state labels
  within a system.  The transformed arms are refitted at the WITH run's SELECTED hyperparameters of the same fold
  (:class:`FrozenNeural`, :class:`FrozenBoosted`, :class:`FrozenB6`, :class:`FrozenLadder` for a deployed ladder step
  M3-M7 whose WITH record is its ``evaluation/ladder`` record; the closed-form comparators need none and are fitted on
  the exact leave-one-cell-out folds of section 3.1 as in discovery), with the cross-fitted split-conformal calibration
  of the discovery runner at the cross-fit selections of the WITH record (:data:`READINGS` ``tuning``).
* **Ln test set** (:func:`ln_rows`): the Ln(III) scored rows of the selection half -- V5-primary Ln(III) cells (V6
  carve-out and ``V6_TARGET_ROWS`` guard applied by ``discovery.scoring_frame``), every Ln(III) state of V2 in the
  selection half, the Ln(III) rows of the V1 selection folds.  Cells whose eligibility depends on actinide partners
  (:func:`actinide_dependent_cells`) are scored in both arms and reported as a stratum.
* **Deltas**: WITH - WITHOUT and WITH - PERMUTED (and WITH - SHUFFLED) with the section 8 paired cluster bootstrap and
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

import importlib.util
import json
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
from gen19ct.evaluation import transfer as ET
from gen19ct.folds import cell_holdout as CH
from gen19ct.folds import io as FI
from gen19ct.folds import registered as FR
from gen19ct.models import interface as I

SCHEMA = "gen19.h3.v1"
FAMILY = "H3"
#: the training transforms that are refitted (WITH is the discovery record; ``discovery.H3_ARMS`` lists all four)
TRANSFORMS: tuple[str, ...] = ("WITHOUT", "ACT_PERMUTED", "ACT_METAL_SHUFFLED")
ALL_ARMS: tuple[str, ...] = D.H3_ARMS
#: section 11 designs: token -> (design, variant); the scheme follows the WITH run (``with_design_dir``)
DESIGNS: dict[str, tuple[str, str]] = {"V5": ("V5", "primary"), "V2": ("V2", "element"), "V1": ("V1", "copy")}
PAIR_DESIGN = "V5PAIR"
#: section 11 transparent references
REFERENCE_ARMS: tuple[str, ...] = ("B6", "B5")
#: the fallback "deployed" arm when no learned configuration passes (section 10 F6: the best-passing baseline; the
#: registered V5 lookup comparator, ``preseal difficulty.json -> V5.lookup_comparator``)
FALLBACK_DEPLOYED = "B3i"
DETERMINISTIC_ARMS: frozenset[str] = frozenset({"B0", "B1", "B2", "B3", "B3x", "B3i", "B3l", "B4", "B4x", "B4l", "B7"})
#: the ladder steps ``scripts/g19_run_ladder.py`` fits (their WITH records live under ``evaluation/ladder``)
LADDER_ARMS: tuple[str, ...] = ("M3", "M4", "M5", "M6", "M7")
#: the ladder-state statuses that end a step (``g19_run_ladder.STATUS_DONE`` + ``STATUS_SKIPPED`` without the H5 block's
#: ``complete``; ``tests/test_h3.py`` asserts the two agree): a step in any other status (running, incomplete,
#: undecided, absent) leaves the deployed configuration undecidable
LADDER_DONE_STATUSES: frozenset[str] = frozenset({"kept", "removed", "judged", "exploratory_not_run", "not_run", "demoted"})
LADDER_STATE_FILE = "evaluation/ladder/decisions/ladder.json"
#: the intervals status recorded when a refit's inner folds differ from the WITH record's (no calibration is invented)
NOT_CALIBRATED_DIFFER = "not_calibrated_inner_folds_differ_from_the_with_record"
#: the R19 scope a discovery-stage H3 verdict is read on (R19 item 4 is NOT_EVALUATED in discovery, addendum 1 item 3)
VERDICT_SCOPE = "freezing_screen"
STAGE = "11_h3"
REFIT_NOT_RUN = "not run: section 11 names no refit sensitivity for the ablation; the scoring-filter sensitivities apply"
NOT_COMPUTED = "not computed"
STEPS = D.STEPS

READINGS: dict[str, str] = {
    "deployed_rule": "the configuration deployed for lanthanide prediction ('the retained ladder configuration', section "
                     "11) is the highest kept ladder step: evaluation/ladder/decisions/ladder.json -> steps.<step>.kept "
                     "== true for M7 > M6 > M5 > M4 > M3 (only from a registered ladder run, i.e. ladder.json -> "
                     "stop_rule == false: under the stop rule M3-M6 are exploratory_not_run and M7 runs for H6 only, so "
                     "no ladder step deploys), then the scorer's decisions.json -> ladder.<step>.kept == true (M2 "
                     "before M1); else M2 when stop_rule.M2_vs_B3i.verdict == PASS; else B6 when "
                     "stop_rule.B6_vs_B3i.verdict == PASS; else the registered V5 lookup comparator B3i (section 10 F6: "
                     "the best-passing baseline; the process chain's D source stays gen18 B1). The rule is undecidable "
                     "(the runners refuse) while ladder.json is absent or any step M3-M7 is not in a done / skipped "
                     "status, or while a discovery ladder decision is pending (kept == null) (needs a POST-HOC addendum; "
                     "task X finding V-01)",
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
    "margins": "R19 item 1 margin of an H3 contrast: delta5 on the V5 Ln cells, 0.05 on V1 / V2 (the scorer's reading of "
               "section 9 for the designs it names no margin for)",
    "f4_designs": "F4 'the WITHOUT arm beats it on the Ln test set (95 % interval excluding 0)' is evaluated per design "
                  "(V5 Ln cells, V1 Ln rows, V2 Ln states) with the primary-cluster percentile interval of the reversed "
                  "contrast (WITHOUT candidate), and F4 holds when ANY design shows it (the conservative reading)",
    "actinide_dependent": "a cell's eligibility 'depends on actinide partners' when the eligible_cells rule of the "
                          "fold builder (folds.cell_holdout.EligibilitySpace.check) passes on all rows and fails once "
                          "every actinide row is removed",
    "equivalent": "TOST (90 % interval, epsilon 0.05, primary cluster) of WITH - WITHOUT on the V5 Ln cells",
    "helps_non_inferior": "non-inferiority on V1 / V2 is the TOST 'non_inferior' flag of WITH - WITHOUT and WITH - "
                          "ACT_PERMUTED on the V1 and V2 Ln sets",
    "shared_only_embedding": "the WITH arm re-run with a shared-only metal embedding (no e_series, e_ox) is not "
                             "implemented: neural.FactorisedNet has no option to disable the offsets (deferred edit)",
    "crps": "CRPS is NOT_RUN: no arm has a predictive SD (discovery.UNCERTAINTY_NOT_RUN)",
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
    """The configuration deployed for lanthanide prediction (:data:`READINGS` ``deployed_rule``) from the ladder runner's
    ``ladder.json`` (``steps.<step>.kept`` for M3-M7, registered run only) and the scorer's ``decisions.json``
    (``ladder.<step>.kept`` for M1 / M2, ``stop_rule.M2_vs_B3i.verdict``, ``stop_rule.B6_vs_B3i.verdict``).

    ``ladder`` is the body of ``evaluation/ladder/decisions/ladder.json``; ``None`` (absent) or an incomplete ladder
    makes the rule ``pending`` -- the H3 arm is 'the retained ladder configuration' and cannot be named before the
    ladder has run (task X finding V-01)."""
    lad = decisions.get("ladder") or {}
    stop = decisions.get("stop_rule") or {}

    def kept(step: str) -> bool | None:
        rec = lad.get(step) or {}
        return rec.get("kept")

    def passes(key: str) -> bool | None:
        rec = stop.get(key)
        return None if rec is None else bool(rec.get("verdict") == "PASS")

    steps = {s: kept(s) for s in ("M2", "M1")}
    stops = {"M2_vs_B3i": passes("M2_vs_B3i"), "B6_vs_B3i": passes("B6_vs_B3i")}
    lc = ladder_complete(ladder)
    lsteps = {s: (((ladder or {}).get("steps") or {}).get(s) or {}).get("kept") for s in LADDER_ARMS}
    keys = {"ladder_runner": {s: f"{LADDER_STATE_FILE} -> steps.{s}.kept" for s in LADDER_ARMS},
            "ladder": {s: f"ladder.{s}.kept" for s in steps}, "stop_rule": {k: f"stop_rule.{k}.verdict" for k in stops}}
    base = {"rule": READINGS["deployed_rule"], "keys": keys, "ladder_kept": {**lsteps, **steps}, "stop_rule_pass": stops,
            "ladder_complete": lc}
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
    if stops["M2_vs_B3i"] is True:
        return {**base, "arm": "M2", "status": "decided",
                "basis": "no ladder step above M0 kept; M2 passes the stop-rule scope against B3i (section 7 item 4)"}
    if stops["B6_vs_B3i"] is True:
        return {**base, "arm": "B6", "status": "decided",
                "basis": "no ladder step kept and M2 does not pass against B3i; B6 passes the stop-rule scope (H1b)"}
    if stops["M2_vs_B3i"] is None or stops["B6_vs_B3i"] is None:
        return {**base, "arm": None, "status": "pending", "basis": "a stop-rule contrast is missing"}
    return {**base, "arm": FALLBACK_DEPLOYED, "status": "decided", "fallback": True,
            "basis": "neither M2 nor B6 passes against B3i: the best-passing baseline is deployed (section 10 F6); "
                     "the process chain's default D source stays gen18 B1 (section 10)"}


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
    """A learned arm's WITH is its discovery record; a closed-form arm has no discovery record, so WITH is fitted."""
    return (("WITH",) if is_deterministic(arm) else ()) + TRANSFORMS


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
                n_resamples: int = ET.N_RESAMPLES) -> dict[str, Any]:
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
    not_run = {n: REFIT_NOT_RUN for n in reg if n not in reduced}
    for n in not_run:
        sens[n] = ET.UNTESTABLE
    det = is_deterministic(model_arm)
    res = D.evaluate_contrast(name=f"{cand_name} vs {comp_name}", family=FAMILY, design=label, primary=pu, margin=margin,
                              seed_deltas={int(seed): pu.delta}, sensitivities=sens, deterministic=det, learned=not det,
                              reduced_sensitivities=reduced if not det else None, n_resamples=n_resamples)
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
        kw.update(v1_scheme=v1_scheme, v1_group_col=EM.PUB_GROUP_COL, remainder_groups=rem)
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
               kappa_min: float | None = None, kappa_status: str = NOT_COMPUTED) -> dict[str, Any]:
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
    return {"verdict": verdict, "helps": helps_ok, "hurts": bool(hurts_wo), "equivalent": equivalent,
            "v5_with_beats_without": h_wo, "v5_with_beats_permuted": h_pm, "v1_v2_non_inferior": ni,
            "tost_v5_with_minus_without": None if tost is None else {k: tost[k] for k in ("verdict", "low_90", "high_90",
                                                                                          "equivalent", "non_inferior")},
            "kappa_min": kappa_min, "kappa_status": kappa_status if verdict == "UNDECIDED" else "not needed",
            "underpowered": (verdict == "UNDECIDED" and kappa_min is not None and kappa_min > ET.KAPPA_MIN_INFORMATIVE),
            "actinide_rows_enter_deployed_configuration": verdict == "helps",
            "inputs_missing": inputs_missing, "scope": VERDICT_SCOPE,
            "label": "discovery, optimistically biased (selection half; seed 104729); R19 item 4 decided at confirmation",
            "readings": {k: READINGS[k] for k in ("r19_scope", "equivalent", "helps_non_inferior")}}


def f4_check(hurts_by_design: Mapping[str, Mapping[str, Any]], *, deployed_arm: str,
             trained_with_actinides: bool = True) -> dict[str, Any]:
    """Section 10 F4: the deployed configuration was trained with actinide rows and the WITHOUT arm beats it on the Ln
    test set with the 95 % interval excluding 0 -- per design the reversed contrast's point > 0 and the primary-cluster
    percentile (and BCa, printed) lower bound > 0; F4 holds when any design shows it (:data:`READINGS` ``f4_designs``)."""
    per = {}
    for design, res in sorted(hurts_by_design.items()):
        if res is None:
            per[design] = {"status": NOT_COMPUTED}
            continue
        br = res["bootstraps"][res["primary_cluster_unit"]]
        plo, phi = br.percentile_interval()
        blo, bhi = br.bca_interval()
        beats = bool(np.isfinite(res["point"]) and res["point"] > 0 and np.isfinite(plo) and plo > 0)
        per[design] = {"status": "computed", "point": res["point"], "percentile_95": [plo, phi], "bca_95": [blo, bhi],
                       "cluster_unit": res["primary_cluster_unit"], "without_beats_with_interval_excludes_0": beats,
                       "bca_also_excludes_0": bool(np.isfinite(blo) and blo > 0)}
    any_beats = any(p.get("without_beats_with_interval_excludes_0") for p in per.values())
    computed = [d for d, p in per.items() if p["status"] == "computed"]
    return {"failure": bool(trained_with_actinides and any_beats), "deployed_arm": deployed_arm,
            "deployed_trained_with_actinide_rows": bool(trained_with_actinides), "per_design": per,
            "designs_computed": computed, "designs_not_computed": [d for d in per if d not in computed],
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


def record_dir(out_root: Path, model_arm: str, transform: str, job: D.JobSpec) -> Path:
    return h3_root(out_root) / "records" / D.ARM_ALIASES.get(model_arm, model_arm) / transform / job.design_dir / f"s{job.seed}"


def fold_paths(out_root: Path, model_arm: str, transform: str, job: D.JobSpec, fold_id: str) -> tuple[Path, Path]:
    d = record_dir(out_root, model_arm, transform, job)
    name = D.safe_fold_name(fold_id)
    return d / f"{name}.parquet", d / f"{name}.json"


def fold_digest(job: D.JobSpec, fold: FI.Fold, code: str, *, transform: str, with_digest: str | None, guard_mode: str,
                design_hash: str, ordinal: int, model_seed: int | None) -> str:
    """The resume / verification digest of one H3 fold: ``discovery.fold_digest`` of the H3 job with the transform, the
    WITH record's digest (the hyperparameters it supplies), the guard mode, the fold file's design hash, the model fold
    number and seed and the registered pre-registration and addenda digests."""
    return D.fold_digest(job, fold, code, {"schema": SCHEMA, "transform": transform, "with_record_digest": with_digest,
                                           "guard_mode": guard_mode, "design_hash": str(design_hash),
                                           "model_fold_number": int(ordinal), "model_seed": model_seed,
                                           "prereg_sha256": D.REGISTERED_PREREG_SHA256,
                                           "prereg_addenda_sha256": D.REGISTERED_ADDENDA_SHA256})


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
    ``{"digest", "fold_hash"}`` as the current code computes them; a foreign fold, a differing digest or fold hash, a
    prediction without a record or a record lacking a step raises :class:`discovery.StaleRecordError`; missing folds
    give ``(None, status)``."""
    exp = {str(k): dict(v) for k, v in expected.items()}
    status: dict[str, Any] = {"what": what, "n_expected": len(exp)}
    if not Path(root).exists():
        return None, {**status, "status": "missing", "n_found": 0}
    by_name = {D.safe_fold_name(k): k for k in exp}
    frames, found = [], set()
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
            raise D.StaleRecordError(f"{pq}: record lacks step(s) {missing}")
        fr = pd.read_parquet(pq)
        if len(fr) and (fr["fold_id"].astype(str) != fid).any():
            raise D.StaleRecordError(f"{pq}: parquet rows of another fold")
        frames.append(fr)
        found.add(fid)
    status["n_found"] = len(found)
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
    the (state, element) labels.  WITH / WITHOUT return the frame unchanged (WITHOUT acts on the mask)."""
    if transform in ("WITH", "WITHOUT"):
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
       (``g19_run_discovery.verified_predictions``: digest, fold hash and fold set exactly what the current code, fold
       files and plan state produce; a stale record raises).

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
            _, st = rd.verified_predictions(out_root, arm, j.design_dir, int(j.seed), code=code, state=state,
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


def d03_markdown(summary: Mapping[str, Any], contrasts: pd.DataFrame | None, deltas: pd.DataFrame | None,
                 negative: Mapping[str, pd.DataFrame] | None = None) -> str:
    """``decisions/D03_actinide_transfer.md`` (brief section 29: question, evidence, metrics, verdict, decision, next
    action) from the H3 outputs; every absent quantity prints ``not computed``."""
    dep = summary.get("deployed") or {}
    ver = summary.get("verdicts") or {}
    f4 = summary.get("f4") or {}
    lines = ["# D03 -- Does actinide extraction data improve hidden-lanthanide prediction? (H3, section 11)", "",
             f"*Generated by `scripts/g19_run_h3.py` at git HEAD `{summary.get('git_head') or NOT_COMPUTED}` from the files "
             "under `evaluation/h3/`. Every number is the selection half, discovery seed 104729, optimistically biased; "
             "R19 item 4 is NOT_EVALUATED in discovery (addendum 1 item 3), so verdicts are read on the "
             f"`{VERDICT_SCOPE}` scope and the full R19 verdict is printed beside. Readings: see `evaluation/h3/h3_summary.json` -> "
             "`readings`.*", "",
             "## Question", "",
             "Same architecture, same folds, same seeds, same lanthanide test set: does training WITH actinide rows beat "
             "training WITHOUT them (and the ACT_PERMUTED / ACT_METAL_SHUFFLED controls) on hidden Ln(III) cells (V5), "
             "unseen Ln(III) states (V2) and unseen publications (V1)? Is the transfer negative (F4)?", "",
             "## Evidence", "",
             "| file | contents |", "|---|---|",
             "| `evaluation/h3/h3_contrasts.csv` | R19 rows (one per cluster unit) of every WITH vs transform and transform vs WITH contrast |",
             "| `evaluation/h3/h3_r19_items.csv` | R19 items 1-6 per contrast |",
             "| `evaluation/h3/h3_deltas.csv` | Delta rank accuracy, Delta calibration, Delta logSF MAE with intervals |",
             "| `evaluation/h3/h3_per_unit_deltas.csv` | per Ln unit MAE of both arms |",
             "| `evaluation/h3/negative_transfer/*.csv` | section 11 negative-transfer tables (descriptive) |",
             "| `evaluation/h3/h3_summary.json`, `h3_verdicts.json`, `h3_f4.json` | verdicts, F4, deployed rule, readings |",
             "| `evaluation/h3/records/<arm>/<transform>/...` | per-fold predictions and records of the refitted arms |", "",
             "## Metrics", "",
             f"Deployed configuration: **{dep.get('arm') or NOT_COMPUTED}** ({dep.get('basis') or NOT_COMPUTED}).",
             f"Model arms: {', '.join(summary.get('model_arms') or []) or NOT_COMPUTED}. Transforms: "
             f"{', '.join(summary.get('transforms') or []) or NOT_COMPUTED}.", ""]
    if contrasts is not None and not contrasts.empty and "primary_cluster_unit" in contrasts.columns:
        prim = contrasts[contrasts["primary_cluster_unit"].astype(bool)]
        lines += ["| model arm | contrast | design | Delta MAE | margin | pct 95 % | BCa 95 % | p | scope verdict | full R19 | TOST |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in prim.sort_values(["model_arm", "design", "contrast"]).iterrows():
            lines.append(f"| {r.get('model_arm')} | {r['contrast']} | {r['design']} | {_fmt(r['point'])} | {_fmt(r.get('margin'))} | "
                         f"[{_fmt(r['percentile_low'])}, {_fmt(r['percentile_high'])}] | [{_fmt(r['bca_low'])}, {_fmt(r['bca_high'])}] | "
                         f"{_fmt(r['p_two_sided'], '{:.4f}')} | {r.get(f'verdict_{VERDICT_SCOPE}', NOT_COMPUTED)} | "
                         f"{r.get('r19_verdict_full', NOT_COMPUTED)} | {r.get('tost_verdict_eps0.05', NOT_COMPUTED)} |")
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
    lines += ["", "## Verdict (null / supported / ambiguous)", ""]
    if ver:
        for arm, v in sorted(ver.items()):
            lines.append(f"- **{arm}: {v.get('verdict', NOT_COMPUTED)}** -- WITH beats WITHOUT on V5: {v.get('v5_with_beats_without')}; "
                         f"beats ACT_PERMUTED: {v.get('v5_with_beats_permuted')}; V1/V2 non-inferior: {v.get('v1_v2_non_inferior')}; "
                         f"TOST WITH-WITHOUT (V5): {(v.get('tost_v5_with_minus_without') or {}).get('verdict', NOT_COMPUTED)}; "
                         f"kappa_min: {v.get('kappa_min') if v.get('kappa_min') is not None else v.get('kappa_status', NOT_COMPUTED)}"
                         + (" -- UNDECIDED (underpowered)" if v.get("underpowered") else ""))
        mapping = {"helps": "supported (actinide rows help lanthanide prediction)", "hurts": "supported (negative transfer)",
                   "equivalent": "null (no difference inside +-0.05)", "UNDECIDED": "ambiguous"}
        dv = ver.get(dep.get("arm") or "", {}).get("verdict")
        lines += ["", f"Deployed arm reading: **{mapping.get(dv, NOT_COMPUTED)}**."]
    else:
        lines.append(f"Verdicts: {NOT_COMPUTED}.")
    lines += ["", f"**F4 (negative actinide transfer): {'HOLDS -- gen19 is unsuccessful on F4' if f4.get('failure') else ('does not hold' if f4.get('status') == 'computed' else NOT_COMPUTED)}.** "
              f"Deployed arm {f4.get('deployed_arm', NOT_COMPUTED)}; per design: "
              + ", ".join(f"{d}: {'beats' if p.get('without_beats_with_interval_excludes_0') else ('no' if p.get('status') == 'computed' else NOT_COMPUTED)}"
                          for d, p in sorted((f4.get("per_design") or {}).items())) + ".", ""]
    if negative:
        strata = negative.get("strata")
        if strata is not None and not strata.empty:
            lines += ["Negative-transfer strata (V5 per-cell Delta MAE = MAE(WITHOUT) - MAE(WITH); descriptive):", "",
                      "| stratum | in stratum | cells | systems | mean Delta | median Delta | share Delta > 0 |", "|---|---|---|---|---|---|---|"]
            for _, r in strata.iterrows():
                lines.append(f"| {r['stratum']} | {r['in_stratum']} | {r['n_cells']} | {r['n_systems']} | {_fmt(r['mean_delta_mae'])} | "
                             f"{_fmt(r['median_delta_mae'])} | {_fmt(r['share_delta_positive'], '{:.2f}')} |")
            lines.append("")
    else:
        lines += [f"Negative-transfer investigation: {NOT_COMPUTED} (run when WITHOUT is better, section 11).", ""]
    lines += ["## Decision", "",
              (f"Actinide rows enter the deployed Ln configuration: **{'yes' if ver.get(dep.get('arm') or '', {}).get('verdict') == 'helps' else 'no'}** "
               "(section 11: only on *helps*)."), "",
              "## Next action", "",
              "- confirmation: the frozen H3 claims (if any) on the withheld seeds and the V6 deltas (section 15);",
              "- a POST-HOC addendum for every reading in `h3_summary.json -> readings` before a result is quoted as registered;",
              "- the section 8 power check (`scripts/g19_run_power.py`) for every UNDECIDED contrast (kappa_min above).", ""]
    return "\n".join(lines)


def json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else (o.as_posix() if isinstance(o, Path) else str(o))))

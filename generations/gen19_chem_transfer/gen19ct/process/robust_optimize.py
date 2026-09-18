"""``robust_optimize.py`` -- the lexicographic robust objective, the S2(d) stability check and the F5 audit (brief
section 18; pre-registration sections 9 S2(d), 10 F5 and 14 "Objective" / "Checks").

**Objective** (section 14, brief section 18), a lexicographic ranking of the operating points of one spec cell
``(purity_min, recovery_min)``:

0. **support rank** -- recipes whose stages interpolated between any ``UNSUPPORTED`` prediction (or evaluated the table
   outside its grid, ``OUTSIDE_TABLE``: no prediction exists there) are **ineligible** and never ranked; recipes that
   used a ``CONDITION_EXTRAPOLATION`` or ``FAMILY_EXTRAPOLATION`` prediction rank below every recipe that used neither;
1. ``P(both targets)`` = ``P(purity >= P_min and recovery >= R_min)`` -- the sealed section 14 key (brief section 18's
   joint ``P(both and feasible)`` is computed and printed beside as ``obj_p_both_feasible`` but does not rank:
   :data:`OBJECTIVE_READING`; task X finding V-06);
2. ``P(feasible)``;
3. reagent consumption (median ``consumption_index``, ascending; unknown last);
4. stage count (ascending);
5. throughput (median kg oxide / h, descending);
6. the candidate index (a deterministic tie-break, never a preference).

:func:`assert_no_unsupported_winner` is the F5(i) code check: a reported recommendation that used an UNSUPPORTED D is a
code defect (``AssertionError``) and also a failure.

**S2(d)** (section 14 "Checks", section 9): the fraction of :data:`N_BOOTSTRAP_RERANKINGS` = 20 bootstrap re-rankings
of the 64 draws (draws resampled with replacement, seed :data:`BOOTSTRAP_SEED` = 19, section 15) that keep the top
recipe, where "keep" means identical stage counts +- 1 (``n_ext``, ``n_scr`` and ``n_str`` each) and O/A within 10 %
(:func:`same_recipe`); S2(d) passes at >= :data:`STABILITY_THRESHOLD` = 0.80.

**Usage tracking is mandatory** (task X finding VL2-03): :func:`rank_recipes` requires the ``statuses_used`` column and
refuses a candidate that converged in at least one draw but records no domain status -- such a recipe would otherwise
rank as clean and the F5(i) check could never fire on it.

**F5** (section 10): (i) any reported recommendation uses an UNSUPPORTED D; (ii) for the TODGA Pr/Nd case, no recipe
reaches ``P(both targets) >= 0.5`` when predictions labelled UNSUPPORTED or CONDITION_EXTRAPOLATION are barred while
one does when they are allowed (evaluated per spec cell; the case fails F5(ii) if any cell does -- a reading,
:data:`F5_READING`).  The section 14 audit re-ranks with the support gate lifted and reports whether the winner changes.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from gen19ct.process import monte_carlo as MC

__all__ = [
    "INELIGIBLE_STATUSES", "EXTRAPOLATION_STATUSES", "SUPPORT_RANK", "STABILITY_THRESHOLD", "N_BOOTSTRAP_RERANKINGS",
    "BOOTSTRAP_SEED", "F5_P_BOTH_THRESHOLD", "LEXICOGRAPHIC_KEYS", "OBJECTIVE_READING", "F5_READING",
    "support_rank", "rank_recipes", "winner", "assert_no_unsupported_winner", "same_recipe", "resample_draws",
    "stability", "f5_audit",
]

INELIGIBLE_STATUSES: frozenset[str] = frozenset({"UNSUPPORTED", "OUTSIDE_TABLE"})
EXTRAPOLATION_STATUSES: frozenset[str] = frozenset({"CONDITION_EXTRAPOLATION", "FAMILY_EXTRAPOLATION"})
SUPPORT_RANK: dict[str, int] = {"ineligible": 0, "extrapolation": 1, "clean": 2}
STABILITY_THRESHOLD = 0.80          # section 9 S2(d)
N_BOOTSTRAP_RERANKINGS = 20         # section 14 "Checks"
BOOTSTRAP_SEED = 19                 # section 15 "Bootstrap seed: 19"
F5_P_BOTH_THRESHOLD = 0.5           # section 10 F5(ii)
STAGE_TOLERANCE = 1                 # "identical stage counts +- 1"
OA_TOLERANCE = 0.10                 # "O/A within 10 %"

LEXICOGRAPHIC_KEYS: tuple[str, ...] = ("support_rank (desc; 0 = ineligible, removed)",
                                       "p_both_<cell> (desc)", "p_feasible (desc)",
                                       "consumption_index_median (asc; unknown last)", "n_stages_total (asc)",
                                       "throughput_median (desc; unknown last)", "candidate (asc; tie-break only)")
OBJECTIVE_READING = ("keys 1-2 rank by the sealed section 14 order: P(both targets) then P(feasible); brief section 18's "
                     "joint P(purity >= P_min and recovery >= R_min and feasible) is computed and printed beside "
                     "(obj_p_both_feasible) but does not rank (task X finding V-06: the sealed text governs; a POST-HOC "
                     "addendum would be needed to rank by the joint key)")
F5_READING = ("F5(ii) is evaluated per spec cell of the registered 3 x 3 grid; the case is reported as failing F5(ii) "
              "when any cell does (reading; the registered text names 'the TODGA Pr/Nd case' without a cell)")


# --------------------------------------------------------------------------------------------- #
# support rank and ranking
# --------------------------------------------------------------------------------------------- #

def support_rank(statuses_used: Iterable[str], *, barred: Iterable[str] = INELIGIBLE_STATUSES,
                 extrapolation: Iterable[str] = EXTRAPOLATION_STATUSES) -> int:
    """0 (ineligible) when any used status is barred, 1 when any is an extrapolation status, else 2."""
    used = {str(s) for s in statuses_used if str(s)}
    if used & set(barred):
        return SUPPORT_RANK["ineligible"]
    if used & set(extrapolation):
        return SUPPORT_RANK["extrapolation"]
    return SUPPORT_RANK["clean"]


def _statuses(cell: Any) -> frozenset[str]:
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return frozenset()
    return frozenset(s for s in str(cell).split("|") if s and s != "nan")


def rank_recipes(ops: pd.DataFrame, cell: tuple[float, float], *, barred: Iterable[str] = INELIGIBLE_STATUSES,
                 extrapolation: Iterable[str] = EXTRAPOLATION_STATUSES) -> pd.DataFrame:
    """The lexicographic ranking (module docstring) of ``ops`` (:func:`monte_carlo.aggregate_operating_points`
    output with ``statuses_used``) for one spec cell; ineligible recipes are removed.  Adds ``support_rank`` and
    ``rank`` (1 = top) and the objective columns ``obj_p_both_feasible`` / ``obj_p_both``."""
    key = MC.cell_key(*cell)
    col_joint, col_both = f"p_both_feasible_{key}", f"p_both_{key}"
    for c in (col_joint, col_both, "p_feasible", "consumption_index_median", "n_stages_total", "throughput_median",
              "statuses_used"):
        if c not in ops.columns:
            raise KeyError(f"operating-point table lacks {c!r}" + (" (the D-source usage record is mandatory: a recipe "
                                                                   "without it cannot be ranked; task X finding VL2-03)"
                                                                   if c == "statuses_used" else ""))
    df = ops.copy()
    sets = [_statuses(x) for x in df["statuses_used"]]
    conv = pd.to_numeric(df["n_converged"], errors="coerce").fillna(0).to_numpy() if "n_converged" in df.columns else \
        np.ones(len(df))
    untracked = [int(c) for c, st, n in zip(df["candidate"], sets, conv) if n > 0 and not st]
    if untracked:
        raise ValueError(f"rank_recipes: candidate(s) {untracked[:10]} converged in at least one draw but record no domain "
                         "status (statuses_used empty / NaN): the D-source usage was not tracked, and an untracked recipe "
                         "would rank as clean (task X finding VL2-03)")
    df["support_rank"] = [support_rank(st, barred=barred, extrapolation=extrapolation) for st in sets]
    df["obj_p_both_feasible"] = pd.to_numeric(df[col_joint], errors="coerce").fillna(0.0)
    df["obj_p_both"] = pd.to_numeric(df[col_both], errors="coerce").fillna(0.0)
    df["_cons"] = pd.to_numeric(df["consumption_index_median"], errors="coerce").fillna(np.inf)
    df["_thr"] = -pd.to_numeric(df["throughput_median"], errors="coerce").fillna(-np.inf)
    df["_pf"] = pd.to_numeric(df["p_feasible"], errors="coerce").fillna(0.0)
    elig = df.loc[df["support_rank"] > 0].copy()
    # sealed section 14 order: support rank, P(both targets), P(feasible), consumption, stages, throughput (V-06)
    elig = elig.sort_values(["support_rank", "obj_p_both", "_pf", "_cons", "n_stages_total", "_thr", "candidate"],
                            ascending=[False, False, False, True, True, True, True], kind="mergesort")
    elig = elig.drop(columns=["_cons", "_thr", "_pf"]).reset_index(drop=True)
    elig.insert(0, "rank", np.arange(1, len(elig) + 1))
    elig.attrs.update({"cell": (float(cell[0]), float(cell[1])), "n_ineligible": int((df["support_rank"] == 0).sum()),
                       "barred": sorted(barred), "keys": list(LEXICOGRAPHIC_KEYS)})
    return elig


def winner(ranked: pd.DataFrame) -> pd.Series | None:
    """The top row of :func:`rank_recipes` (``None`` when no recipe is eligible)."""
    return None if len(ranked) == 0 else ranked.iloc[0]


def assert_no_unsupported_winner(win: pd.Series | Mapping[str, Any] | None) -> None:
    """F5(i) code check: a reported recommendation whose ``statuses_used`` carries an ineligible status is a code
    defect (``AssertionError``)."""
    if win is None:
        return
    used = _statuses(win.get("statuses_used") if isinstance(win, Mapping) else win["statuses_used"])
    bad = sorted(used & INELIGIBLE_STATUSES)
    if bad or int(win["support_rank"]) == 0:
        raise AssertionError(f"F5(i): the recommended recipe (candidate {int(win['candidate'])}) uses {bad or 'an '
                             'ineligible'} prediction(s); this is a code defect and a registered failure")


# --------------------------------------------------------------------------------------------- #
# S2(d) stability under D-draw bootstrap re-rankings
# --------------------------------------------------------------------------------------------- #

def same_recipe(a: Mapping[str, Any] | pd.Series, b: Mapping[str, Any] | pd.Series, *, stage_tol: int = STAGE_TOLERANCE,
                oa_tol: float = OA_TOLERANCE) -> bool:
    """Section 14: identical stage counts within ``stage_tol`` (each of ``n_ext``, ``n_scr``, ``n_str``) and O/A within
    ``oa_tol`` (relative to ``a``'s O/A)."""
    for c in ("n_ext", "n_scr", "n_str"):
        if abs(int(round(float(a[c]))) - int(round(float(b[c])))) > stage_tol:
            return False
    oa_a, oa_b = float(a["oa_ext"]), float(b["oa_ext"])
    if oa_a <= 0:
        return oa_b <= 0
    return abs(oa_b / oa_a - 1.0) <= oa_tol


def resample_draws(process_table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The process table restricted to a bootstrap sample (with replacement) of its draw ids; a draw sampled twice
    appears twice (as ``draw`` = the original id, ``boot_copy`` = the copy index)."""
    ids = np.array(sorted(set(process_table["draw"].tolist())))
    sample = rng.choice(ids, size=ids.size, replace=True)
    parts = []
    for c, d in enumerate(sample):
        part = process_table.loc[process_table["draw"] == d].copy()
        part["boot_copy"] = c
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def stability(process_table: pd.DataFrame, usage: pd.DataFrame, cell: tuple[float, float], purity_grid: Sequence[float],
              recovery_grid: Sequence[float], *, n_boot: int = N_BOOTSTRAP_RERANKINGS, seed: int = BOOTSTRAP_SEED,
              barred: Iterable[str] = INELIGIBLE_STATUSES, variables: Iterable[str] = ()) -> dict[str, Any]:
    """S2(d): re-aggregate and re-rank ``n_boot`` bootstrap resamples of the draws; the fraction whose winner is
    :func:`same_recipe` as the full-data winner.  ``passes`` at >= :data:`STABILITY_THRESHOLD`; ``None`` verdict when
    no recipe is eligible on the full data."""
    ops = MC.aggregate_operating_points(process_table, purity_grid, recovery_grid, usage, variables=variables)
    ranked = rank_recipes(ops, cell, barred=barred)
    top = winner(ranked)
    if top is None:
        return {"cell": [float(cell[0]), float(cell[1])], "top_candidate": None, "fraction_kept": None, "passes": None,
                "n_boot": int(n_boot), "seed": int(seed), "threshold": STABILITY_THRESHOLD,
                "bootstrap_winners": [], "note": "no eligible recipe on the full draws"}
    rng = np.random.default_rng(int(seed))
    kept = 0
    boot_winners: list[dict[str, Any]] = []
    for b in range(int(n_boot)):
        sub = resample_draws(process_table, rng)
        ops_b = MC.aggregate_operating_points(sub, purity_grid, recovery_grid, usage, variables=variables)
        w = winner(rank_recipes(ops_b, cell, barred=barred))
        same = bool(w is not None and same_recipe(top, w))
        kept += int(same)
        boot_winners.append({"bootstrap": b, "candidate": None if w is None else int(w["candidate"]), "kept": same,
                             "n_ext": None if w is None else int(w["n_ext"]), "n_scr": None if w is None else int(w["n_scr"]),
                             "n_str": None if w is None else int(w["n_str"]),
                             "oa_ext": None if w is None else float(w["oa_ext"])})
    frac = kept / float(n_boot)
    return {"cell": [float(cell[0]), float(cell[1])], "top_candidate": int(top["candidate"]),
            "top_recipe": {"n_ext": int(top["n_ext"]), "n_scr": int(top["n_scr"]), "n_str": int(top["n_str"]),
                           "oa_ext": float(top["oa_ext"]), "p_both_feasible": float(top["obj_p_both_feasible"]),
                           "p_both": float(top["obj_p_both"])},
            "fraction_kept": frac, "passes": bool(frac >= STABILITY_THRESHOLD), "n_boot": int(n_boot), "seed": int(seed),
            "threshold": STABILITY_THRESHOLD, "rule": "identical stage counts +- 1 and O/A within 10 % (section 14)",
            "bootstrap_winners": boot_winners}


# --------------------------------------------------------------------------------------------- #
# F5 audit
# --------------------------------------------------------------------------------------------- #

def _max_p_both(ops: pd.DataFrame, cell: tuple[float, float], barred: Iterable[str]) -> dict[str, Any]:
    ranked = rank_recipes(ops, cell, barred=barred)
    if len(ranked) == 0:
        return {"n_eligible": 0, "max_p_both": None, "max_p_both_feasible": None, "reaches_threshold": False}
    pb = float(ranked["obj_p_both"].max())
    return {"n_eligible": int(len(ranked)), "max_p_both": pb, "max_p_both_feasible": float(ranked["obj_p_both_feasible"].max()),
            "reaches_threshold": bool(pb >= F5_P_BOTH_THRESHOLD)}


def f5_audit(ops: pd.DataFrame, cells: Sequence[tuple[float, float]], *, ops_allowed: pd.DataFrame | None = None,
             ) -> dict[str, Any]:
    """Section 10 F5 and the section 14 gate-lifted re-ranking, per spec cell.

    ``ops`` is the registered run (support gate as registered).  ``ops_allowed`` is the operating-point table of the
    run with ``allow_unsupported=True`` on the untrimmed table (when it differs from ``ops``); when ``None`` the
    "allowed" variant is ``ops`` with every bar lifted."""
    allowed = ops if ops_allowed is None else ops_allowed
    per_cell = []
    any_f5ii = False
    any_f5i = False
    for cell in cells:
        reg = rank_recipes(ops, cell)                                   # registered gate
        top = winner(reg)
        f5i = False
        try:
            assert_no_unsupported_winner(top)
        except AssertionError:
            f5i = True
        lifted = rank_recipes(allowed, cell, barred=(), extrapolation=())
        top_l = winner(lifted)
        barred_ii = _max_p_both(ops, cell, INELIGIBLE_STATUSES | {"CONDITION_EXTRAPOLATION"})
        allowed_ii = _max_p_both(allowed, cell, ())
        f5ii = bool((not barred_ii["reaches_threshold"]) and allowed_ii["reaches_threshold"])
        any_f5ii |= f5ii
        any_f5i |= f5i
        per_cell.append({"cell": [float(cell[0]), float(cell[1])], "key": MC.cell_key(*cell),
                         "winner_registered": None if top is None else int(top["candidate"]),
                         "winner_gate_lifted": None if top_l is None else int(top_l["candidate"]),
                         "winner_changes_when_gate_lifted": (None if top is None or top_l is None
                                                            else not same_recipe(top, top_l)),
                         "F5_i": f5i, "barred_unsupported_and_condition_extrapolation": barred_ii,
                         "allowed_all_statuses": allowed_ii, "F5_ii": f5ii})
    return {"threshold_p_both": F5_P_BOTH_THRESHOLD, "cells": per_cell, "F5_i_any_cell": any_f5i,
            "F5_ii_any_cell": any_f5ii, "F5": bool(any_f5i or any_f5ii), "reading": F5_READING,
            "allowed_variant": ("separate run with allow_unsupported=True on the untrimmed table" if ops_allowed is not None
                                else "the registered run with every support bar lifted (the table had no UNSUPPORTED cell)")}

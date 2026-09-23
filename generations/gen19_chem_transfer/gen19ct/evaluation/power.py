"""``evaluation/power.py`` -- the section 8 signal-injection power check and the section 8 "reliability before
correlation" report.

Nothing here fits a model on a registered outer fold of the REAL targets: the injection re-runs a contrast's arms on
``y' = y + kappa s`` (never data, brief section 33), and the reliability estimators take per-unit quantities the caller
computed.  ``scripts/g19_run_power.py`` drives both after discovery is complete.

Signal injection (section 8 "Signal-injection power check")
-----------------------------------------------------------
* Which contrasts need it (:func:`contrasts_needing_power`): a failed **H1 / H1b / H3** contrast of the scorer's
  contrast files -- ``discovery.POWER_CHECK_FAMILIES`` mapped onto the registered families ``primary`` (H1: M2 vs B3i),
  ``H1b`` (B6 vs B3i), ``S1(b)`` and ``H3`` -- whose full R19 verdict is not PASS.  Before such a null is reported the
  check runs; a null with ``kappa_min > 0.25`` is UNDECIDED (underpowered), ``transfer.power_verdict``.  **POST-HOC
  addendum 4 item 4** (:data:`ADDENDUM4_WORDING`, :data:`REPORTED_LABEL`) fixes what the check may CONCLUDE: a contrast
  that is not a null at all is **POWERED_NOT_A_NULL**, a genuine failure with no passing kappa is **UNDECIDED
  (underpowered)** -- never "no effect" -- and a failed contrast with no registered check is **UNDECIDED (no registered
  power check)**, never null.
* The signal (``transfer.injected_signal``, reused): ``s_row = u_m v_s``, ``u ~ N(0, 1)`` per metal state, ``v ~ N(0, 1)``
  per system, seeded and standardised.  For **H3** every An(III) state shares the ``u`` of its nearest-CN8-radius Ln(III)
  (:func:`h3_u_share`), so a cross-metal lookup cannot exploit it but actinide transfer can.
* X(?) rows are **dropped from the injected refits of every arm** (POST-HOC addendum 1 below the sealed footer) and the
  scored rows stay exactly the un-injected run's: both are enforced by ``transfer.prepare_injected_run`` /
  ``InjectedRun.fold_inputs``, which this module reuses rather than re-implements (:func:`injected_run`,
  :func:`fold_inputs`).
* Models are refitted at their **selected** hyperparameters, read from the discovery / ladder records
  (``h3.selected_hyperparameters``, ``h3.frozen_runner``), never re-tuned.
* Only metrics are written: :func:`assert_no_injected_values` refuses any output frame carrying an injected target, and
  the runner writes no prediction parquet of an injected fit.

Reliability before correlation (section 8)
------------------------------------------
Scope: B7 slopes ``n`` / ``p_eff``, factor loadings ``u_m`` / ``v_s``, learned embeddings, per-system logSF amplitudes and
support-score components (:data:`RELIABILITY_QUANTITIES`).  Each reports, before any correlation is read:

* :func:`split_half_reliability` -- split-half by publication group where the unit has >= 4 groups: 20 seeded random
  halves, the per-unit values of the two halves correlated across units and stepped up by ``transfer.spearman_brown``;
* :func:`jackknife_reliability` -- otherwise jackknife-by-publication: between-unit variance / (between-unit variance +
  mean within-unit SE^2), ``transfer.jackknife_reliability``;
* :func:`embedding_stability` -- Procrustes-aligned bootstrap stability of an embedding table (the figures' number).

``transfer.reliability_gate`` applies the **0.3 floor**: below it the quantity may neither support nor close a
correlation-based claim and its correlations print UNDECIDED (unreliable).

Readings where the registration is silent are in :data:`READINGS`; each needs a POST-HOC addendum before a number is
quoted as registered.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gen19ct.chemistry import support_graph as SG
from gen19ct.evaluation import discovery as D
from gen19ct.evaluation import h3 as H3
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import transfer as ET
from gen19ct.folds import io as FI
from gen19ct.models import interface as I

SCHEMA = "gen19.power.v1"
#: where the per-(contrast, kappa) bootstrap input is persisted as metrics (task X finding V-L2)
PER_UNIT_REL = "evaluation/power/per_unit_mae.csv"
STAGE = "08_power"
KAPPAS: tuple[float, ...] = ET.KAPPAS
#: the registered families of section 19 that carry H1 / H1b / H3 (``discovery.POWER_CHECK_FAMILIES``)
POWER_FAMILIES: tuple[str, ...] = D.POWER_CHECK_FAMILIES
#: section 8 reliability scope
RELIABILITY_QUANTITIES: tuple[str, ...] = ("b7_slopes", "factor_loadings", "embeddings", "logsf_amplitudes",
                                           "support_score_components")
#: section 8: split-half by publication group where THE UNIT has >= 4 groups; a unit with 2-3 groups takes the
#: delete-one-publication jackknife; a unit with one group has no reliability estimate (reported, never scored)
MIN_GROUPS_FOR_SPLIT_HALF = 4
MIN_GROUPS_FOR_JACKKNIFE = 2
N_SPLIT_HALVES = 20
SPLIT_HALF_SEED = ET.BOOTSTRAP_SEED
RELIABILITY_FLOOR = ET.RELIABILITY_FLOOR
NOT_COMPUTED = "not computed"

#: POST-HOC addendum 4 item 4, "What the power check may conclude" -- the registered rule made explicit
ADDENDUM4_WORDING = (
    "POST-HOC addendum 4 item 4: section 8 requires the injection check before a failed H1, H1b or H3 contrast is "
    "reported as a null. A contrast that is not a null at all -- its un-injected verdict PASSES its reported scope and "
    "its point estimate has a direction -- is reported POWERED_NOT_A_NULL, because the registered wording 'null' never "
    "applies to it. A genuine failure with no kappa at which R19 passes is UNDECIDED (underpowered), never 'no effect'. "
    "A contrast with no registered power check is reported UNDECIDED (no registered power check), never null. This is "
    "the registered rule made explicit, not a change to it")
#: the label each internal verdict is REPORTED under (POST-HOC addendum 4 item 4); the report and D03 / D02 print these
REPORTED_LABEL: dict[str, str] = {
    "INFORMATIVE_NULL": "null (informative)",
    "UNDECIDED_UNDERPOWERED": "UNDECIDED (underpowered)",
    "POWERED_NOT_A_NULL": "POWERED_NOT_A_NULL",
    "NOT_A_NULL_UNDERPOWERED": "POWERED_NOT_A_NULL"}
#: what a FAILED contrast with no power record is reported as (POST-HOC addendum 4 item 4)
NO_POWER_CHECK_LABEL = "UNDECIDED (no registered power check)"

READINGS: dict[str, str] = {
    "needs_power": "a contrast needs the power check when its family is one of "
                   f"{list(POWER_FAMILIES)} (section 19's H1 = primary, H1b, S1(b) and H3) and its full R19 verdict is "
                   "not PASS, read from the scorer's contrasts_registered.csv (column r19_verdict_full on the "
                   "primary-cluster row) or from an H3 contrast table; under addendum 1 item 3 a learned-arm contrast is "
                   "at best UNDECIDED in discovery, so the check is reported for the discovery-scope verdict as well "
                   "and both are printed",
    "r19_scope": "the injected R19 verdict is read on the same scope as the un-injected contrast (discovery: the "
                 "freezing-screen scope of items 1-3, 5 and the scoring-filter sensitivities; the full verdict printed "
                 "beside), because R19 item 4 is NOT_EVALUATED in discovery and the refit sensitivities are not re-run "
                 "on injected targets (INFERRED; needs a POST-HOC addendum)",
    "injection_seed": "the signal is drawn once per contrast with the contrast's decision seed (104729) and reused for "
                      "every kappa, so kappa scales one fixed signal (the section 8 wording 'y + kappa s' with one s)",
    "h3_u_share": "for H3 an An(III) state takes the u of the Ln(III) state whose Shannon CN8 radius is nearest "
                  "(chemistry.support_graph.metal_properties -> r_cn8; ties by the state label); an An(III) state "
                  "without a CN8 radius, and every non-An(III) state, keeps its own u and is listed",
    "refit": "every arm of the contrast is refitted at the hyperparameters its discovery record SELECTED on the same "
             "fold (h3.selected_hyperparameters), without re-tuning, on the injected training rows; the closed-form "
             "comparators need no hyperparameters",
    "scored_rows": "the injected run scores exactly the un-injected run's scored rows "
                   "(transfer.assert_same_scored_rows); X(?) rows are dropped from the refits of every arm and are "
                   "scored in no design, so the paired comparison is on identical units",
    "no_persisted_values": "no injected target or injected prediction is written: the runner writes metrics, R19 rows "
                           "and kappa_min only (brief section 33; assert_no_injected_values checks every frame)",
    "per_unit_metrics": "per (contrast, kappa) the runner ALSO writes the bootstrap input as metrics -- per averaging "
                        "unit the MAE of both arms, their paired difference and the registered cluster labels "
                        "(power.per_unit_rows -> evaluation/power/per_unit_mae.csv, tables/power_per_unit_mae.csv) -- "
                        "so kappa_min and every R19 item behind it are re-derivable without repeating the refits "
                        "(task X finding V-L2). A per-cell MAE and a cluster label are metrics, not injected targets "
                        "and not per-row predictions: brief section 33 is unchanged and the same "
                        "assert_no_injected_values check runs on the frame",
    "split_half_correlation": "the split-half correlation is Pearson across the units present in both halves, stepped "
                              "up by Spearman-Brown; the Spearman version is printed beside (the registration says "
                              "'split-half ... Spearman-Brown corrected' without naming the correlation)",
    "jackknife_se": "jackknife-by-publication within-unit SE^2 of a unit is (G-1)/G * sum_g (x_(-g) - mean_g x_(-g))^2 "
                    "over the G publications (groups) the unit has, the standard delete-one jackknife variance; the "
                    "between-unit variance is the sample variance (ddof 1) of the full-sample per-unit values",
    "per_unit_branches": "section 8's rule is per UNIT: every unit measured in >= 4 publication groups enters the "
                         "split-half branch (each of the 20 seeded splits halves THAT unit's own groups; the two "
                         "half-values are correlated across those units, Spearman-Brown), every unit with 2-3 groups "
                         "enters the delete-one-publication jackknife branch (SE^2 over its own groups), and a unit with "
                         "one group has no reliability (counted, never scored). The quantity's reported reliability is "
                         "the MINIMUM over the computed branches -- conservative: a correlation over all units is only "
                         "as reliable as its least reliable stratum -- with both branch values printed beside and the "
                         "per-unit method recorded (task X finding V-02; the combination rule is INFERRED and needs a "
                         "POST-HOC addendum)",
    "embedding_stability": "Procrustes-aligned bootstrap stability of an embedding: every bootstrap replicate's matrix "
                           "is aligned to the full-sample one by the orthogonal Procrustes solution (centred, not "
                           "scaled), and the stability of a unit is the mean cosine similarity of its aligned replicate "
                           "vectors with its full-sample vector; the table's stability is the mean over units and is "
                           "gated by the 0.3 floor like a reliability. The alignment maximises agreement, so the "
                           "statistic is upward-biased and its no-correspondence value is NOT 0 (measured on random "
                           "matrices: 0.42 at 12 units x 4 dimensions, 0.23 at 30 x 4, 0.27 at 60 x 8, 0.23 at 200 x "
                           "16) -- at gen19's embedding sizes that sits at the registered 0.3 floor, so a "
                           "unit-label-permuted null (null_stability) is computed and printed beside every value and a "
                           "raw value inside the null band is flagged within_null_band. The gate stays on the raw "
                           "value: using the null in a decision needs a POST-HOC addendum",
    "reliability_not_fitted": "the runner computes the reliability of the quantities that need no learned fit (B7 "
                              "slopes -- closed form; support-score components -- target-free; per-system logSF "
                              "amplitudes -- observed pairs) by default; the factor loadings u_m / v_s and the learned "
                              "embeddings need a B6 / M-model fit per half, which the runner performs only with "
                              "--include-learned (no outer fold is scored: the fits are on publication-group halves of "
                              "the training corpus and no prediction is written)",
}


# --------------------------------------------------------------------------------------------- #
# which contrasts need the check
# --------------------------------------------------------------------------------------------- #

def contrasts_needing_power(contrasts: pd.DataFrame, *, families: Sequence[str] = POWER_FAMILIES,
                            scope_col: str = f"verdict_{H3.VERDICT_SCOPE}") -> pd.DataFrame:
    """The failed H1 / H1b / H3 contrasts of a scorer contrast table (:data:`READINGS` ``needs_power``): the
    primary-cluster rows whose family is registered for the check and whose full R19 verdict is not PASS."""
    if contrasts is None or contrasts.empty:
        return pd.DataFrame(columns=["family", "contrast", "design", "candidate", "comparator", "r19_verdict_full",
                                     scope_col, "needs_power_check"])
    fr = contrasts
    if "primary_cluster_unit" in fr.columns:
        fr = fr[fr["primary_cluster_unit"].astype(bool)]
    fr = fr[fr["family"].astype(str).isin(list(families))].copy()
    full = fr["r19_verdict_full"].astype(str) if "r19_verdict_full" in fr.columns else pd.Series("", index=fr.index)
    fr["needs_power_check"] = (full != "PASS").to_numpy()
    keep = [c for c in ("family", "contrast", "design", "candidate", "comparator", "point", "margin",
                        "r19_verdict_full", scope_col, "key", "needs_power_check") if c in fr.columns]
    return fr[keep].reset_index(drop=True)


def read_contrast_files(paths_: Iterable[Path]) -> pd.DataFrame:
    """Concatenate the scorer's contrast CSVs that exist (registered contrasts, and an H3 contrast table)."""
    frames = []
    for p in paths_:
        p = Path(p)
        if p.exists():
            fr = pd.read_csv(p)
            fr["source_file"] = p.name
            frames.append(fr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------------------------- #
# the H3 u-sharing map
# --------------------------------------------------------------------------------------------- #

def h3_u_share(states: Iterable[Any]) -> dict[str, str]:
    """``{An(III) state: the Ln(III) state whose Shannon CN8 radius is nearest}`` (:data:`READINGS` ``h3_u_share``);
    states without a CN8 radius are left out (they keep their own ``u``)."""
    labels = sorted({str(s) for s in states if isinstance(s, str) and not str(s).endswith("(?)")})
    ln = {}
    an = {}
    for s in labels:
        p = SG.metal_properties(s)
        r = float(p.get("r_cn8")) if p.get("r_cn8") is not None else float("nan")
        if p.get("ox") != 3 or not np.isfinite(r):
            continue
        if p.get("series") == "Ln":
            ln[s] = r
        elif p.get("series") == "An":
            an[s] = r
    if not ln:
        return {}
    out = {}
    for s, r in sorted(an.items()):
        out[s] = min(sorted(ln), key=lambda t: (abs(ln[t] - r), t))
    return out


def u_share_for(family: str, states: Iterable[Any]) -> dict[str, str]:
    """The ``u_share`` map of a contrast's family: H3 shares the An(III) u (section 8); H1 / H1b / S1(b) share none."""
    return h3_u_share(states) if str(family) == "H3" else {}


# --------------------------------------------------------------------------------------------- #
# the injected run of one contrast
# --------------------------------------------------------------------------------------------- #

def injected_run(frame: pd.DataFrame, *, kappa: float, seed: int, scored_index: Iterable[Any],
                 u_share: Mapping[str, str] | None = None, y_col: str = I.TARGET_COL) -> ET.InjectedRun:
    """``transfer.prepare_injected_run`` on the corpus frame (reused, never re-implemented): the signal is drawn on all
    rows, X(?) rows are dropped from the refits of every arm, ``y' = y + kappa s`` is applied to training and test rows
    alike, and the scored rows stay the un-injected run's."""
    return ET.prepare_injected_run(frame, kappa=float(kappa), seed=int(seed), scored_index=scored_index, y_col=y_col,
                                   state_col=SG.METAL_COL, system_col=SG.SYSTEM_COL, u_share=dict(u_share or {}))


def fold_inputs(run: ET.InjectedRun, fold: FI.Fold, *, labels_of: Callable[[Sequence[str]], pd.Index],
                scored_ids: Sequence[str]) -> tuple[pd.DataFrame, pd.Index]:
    """``(injected training frame, scored labels)`` of one outer fold through ``InjectedRun.fold_inputs``: training is
    the input rows minus the fold's hidden rows minus the X(?) rows, and the scored rows must be hidden rows of the
    fold and scored rows of the run."""
    hidden = labels_of(list(fold.hidden_row_ids))
    scored = labels_of(list(scored_ids))
    return run.fold_inputs(hidden, scored)


def assert_no_injected_values(frame: pd.DataFrame, what: str = "power-check output",
                             forbidden: Sequence[str] = (I.TARGET_COL, "log_D_injected", "injected_signal",
                                                         "mean_logD", "y", "pred")) -> None:
    """Refuse an output frame that carries an injected target, signal or per-row prediction (:data:`READINGS`
    ``no_persisted_values``): the power check writes metrics only (brief section 33)."""
    hit = sorted(set(map(str, frame.columns)) & set(forbidden))
    if hit:
        raise AssertionError(f"{what}: injected values are never persisted as data; drop the column(s) {hit} "
                             "(section 8: injected values exist only inside this check)")


def per_unit_rows(pu: Any, *, contrast: str, kappa: float, candidate: str, comparator: str, design: str
                  ) -> pd.DataFrame:
    """The BOOTSTRAP INPUT of one injected kappa, per averaging unit: both arms' MAE, their paired difference and the
    registered cluster labels the paired cluster bootstrap resamples (``discovery.PairedUnits``).

    Without this, ``kappa_min`` and every R19 item behind it can be re-derived only by repeating the refits, because
    no injected prediction is persisted (task X finding V-L2).  These are METRICS -- a per-cell mean absolute error and
    a cluster label -- not injected targets and not per-row predictions, so section 8 / brief section 33 still holds
    (:func:`assert_no_injected_values` is applied to the frame)."""
    out = pd.DataFrame({"unit": pd.Index(pu.cand_mae.index).astype(str),
                        "mae_candidate": pu.cand_mae.to_numpy(dtype=float),
                        "mae_comparator": pu.comp_mae.to_numpy(dtype=float)})
    out["delta_unit"] = out["mae_comparator"] - out["mae_candidate"]
    for name, lab in (getattr(pu, "clusters", None) or {}).items():
        out[f"cluster_{name}"] = pd.Series(lab).astype(str).to_numpy()
    out.insert(0, "kappa", float(kappa))
    out.insert(0, "design", str(design))
    out.insert(0, "comparator", str(comparator))
    out.insert(0, "candidate", str(candidate))
    out.insert(0, "contrast", str(contrast))
    assert_no_injected_values(out, f"power per-unit metrics ({contrast} kappa={kappa:g})")
    return out


@dataclass(frozen=True)
class KappaResult:
    """One kappa of one contrast: the R19 result of the injected contrast and whether it passes."""

    kappa: float
    passed: bool
    scope_verdict: str
    full_verdict: str
    point: float
    margin: float
    n_units: int
    n_rows: int
    contrast: dict[str, Any] = field(repr=False, default_factory=dict)

    def record(self, regime: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {**dict(regime or {}), "kappa": self.kappa, "passed": self.passed, "scope_verdict": self.scope_verdict,
                "r19_verdict_full": self.full_verdict, "point": self.point, "margin": self.margin,
                "n_units": self.n_units, "n_rows": self.n_rows, **self.item_statuses()}

    def item_statuses(self) -> dict[str, Any]:
        """The per-item R19 statuses of this kappa, so an UNDECIDED_UNDERPOWERED verdict can be AUDITED: without them a
        kappa that fails with a point estimate far above the margin is indistinguishable from one that fails on the
        point estimate, and the reader cannot tell a genuinely underpowered design from an item that no kappa can move
        (e.g. a sensitivity that is UNTESTABLE because its refit is not run)."""
        r19 = (self.contrast or {}).get("r19")
        items = list(getattr(r19, "items", ()) or ())
        out: dict[str, Any] = {f"r19_item{int(i['item'])}": str(i["status"]) for i in items}
        failed = [int(i["item"]) for i in items if str(i["status"]) == "FAIL"]
        blocked = [int(i["item"]) for i in items if str(i["status"]) not in ("PASS", "FAIL")]
        out["r19_items_failed"] = ";".join(str(i) for i in failed)
        out["r19_items_not_decided"] = ";".join(str(i) for i in blocked)
        out["r19_first_failing_item_detail"] = next((str(i.get("detail") or "") for i in items
                                                     if str(i["status"]) == "FAIL"), "")
        return out


def _favours(point: Any) -> str:
    """Which side a contrast's point estimate favours.  ``discovery.evaluate_contrast`` / ``h3.h3_contrast`` define
    Delta = MAE(comparator) - MAE(candidate), so a POSITIVE point favours the CANDIDATE and a negative one the
    comparator; either way the contrast is not a null."""
    try:
        p = float(point)
    except (TypeError, ValueError):
        return "its point estimate is not computed"
    if not np.isfinite(p):
        return "its point estimate is not finite"
    if p > 0:
        return (f"its point estimate ({p:+.4g}) favours the CANDIDATE arm (Delta = MAE(comparator) - MAE(candidate), so "
                "positive favours the candidate)")
    if p < 0:
        return (f"its point estimate ({p:+.4g}) favours the COMPARATOR (Delta = MAE(comparator) - MAE(candidate), so "
                "negative favours the comparator)")
    return "its point estimate is exactly 0"


def kappa_min(results: Iterable[KappaResult], *, uninjected_verdict: str | None = None,
              uninjected_point: Any = None) -> dict[str, Any]:
    """``transfer.power_verdict`` over the registered kappa grid: kappa_min and INFORMATIVE_NULL /
    UNDECIDED_UNDERPOWERED (a null with kappa_min > 0.25 is underpowered) -- or, when the UN-INJECTED contrast is not a
    null at all (its reported-scope verdict PASSES), POWERED_NOT_A_NULL / NOT_A_NULL_UNDERPOWERED: section 8 scopes the
    check to a contrast "reported as a null", so a contrast that passes its scope is never reported as one (task X
    findings V-L3 / V-P04).

    POST-HOC addendum 4 item 4 fixes the WORDING of the three outcomes (:data:`ADDENDUM4_WORDING`) and is quoted in the
    record: **POWERED_NOT_A_NULL** for a contrast that is not a null (its point estimate has a direction and the
    registered wording "null" never applies), **UNDECIDED (underpowered)** for a genuine failure with no passing kappa,
    and **UNDECIDED (no registered power check)** for a failed contrast with no record at all -- never "no effect".
    ``uninjected_point`` is the un-injected point estimate, named in the sentence so the direction is on the record
    (:func:`_favours`)."""
    by_kappa = {float(r.kappa): bool(r.passed) for r in results}
    verdict = ET.power_verdict(by_kappa, uninjected_verdict=uninjected_verdict)
    sensitivity = ("a signal of kappa_min SD would have been detected" if verdict["informative"]
                   else "the design could not detect a signal at or below 0.25 log D"
                        + ("" if by_kappa and any(by_kappa.values()) else "; no registered kappa makes R19 pass"))
    direction = _favours(uninjected_point)
    not_a_null = ("not a null: the un-injected contrast PASSES its reported scope and "
                  f"{direction}, so the registered wording 'null' never applies and nothing is reported as a null here; "
                  f"sensitivity for the record -- {sensitivity}")
    reported = {"INFORMATIVE_NULL": f"the null is informative: {sensitivity}",
                "UNDECIDED_UNDERPOWERED": f"UNDECIDED (underpowered): {sensitivity}",
                "POWERED_NOT_A_NULL": f"POWERED_NOT_A_NULL -- {not_a_null}",
                "NOT_A_NULL_UNDERPOWERED": f"POWERED_NOT_A_NULL -- {not_a_null}"}[verdict["verdict"]]
    return {**verdict, "passes_by_kappa": by_kappa, "kappas": list(KAPPAS),
            "kappa_min_informative": ET.KAPPA_MIN_INFORMATIVE, "reported": reported,
            "reported_label": REPORTED_LABEL[verdict["verdict"]], "uninjected_point": uninjected_point,
            "uninjected_point_favours": direction, "wording_rule": ADDENDUM4_WORDING}


def power_record(contrast_key: str, *, family: str, design: str, arms: Sequence[str], results: Sequence[KappaResult],
                 seed: int, u_share: Mapping[str, str], n_dropped_unknown_state: int,
                 uninjected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The power-check record of one contrast (metrics only)."""
    un = dict(uninjected or {})
    km = kappa_min(results, uninjected_verdict=un.get("reported_verdict") or un.get(f"verdict_{H3.VERDICT_SCOPE}"),
                   uninjected_point=un.get("point"))
    return {"schema": SCHEMA, "contrast": contrast_key, "family": family, "design": design, "arms": list(arms),
            "injection_seed": int(seed), "u_share": {str(k): str(v) for k, v in sorted(dict(u_share).items())},
            "n_u_shared_states": len(u_share), "n_rows_dropped_unknown_state": int(n_dropped_unknown_state),
            "per_kappa": [r.record() for r in results], **km,
            "uninjected": dict(uninjected or {}), "verdict_scope": H3.VERDICT_SCOPE,
            "per_unit_metrics": PER_UNIT_REL, "per_unit_metrics_persisted": True,
            "readings": {k: READINGS[k] for k in ("injection_seed", "h3_u_share", "refit", "scored_rows", "r19_scope",
                                                  "no_persisted_values", "per_unit_metrics")},
            "wording_rule": ADDENDUM4_WORDING,
            "label": "signal-injection power check (section 8); injected values are not data (brief section 33)"}


# --------------------------------------------------------------------------------------------- #
# reliability before correlation (section 8)
# --------------------------------------------------------------------------------------------- #

def _corr(a: np.ndarray, b: np.ndarray, kind: str = "pearson") -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x, y = x[ok], y[ok]
    if kind == "spearman":
        return EM.spearman_rho(x, y)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def split_halves(groups: Sequence[str], *, n_halves: int = N_SPLIT_HALVES, seed: int = SPLIT_HALF_SEED
                 ) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """``n_halves`` seeded random splits of the publication groups into two halves (sorted labels, ``default_rng(seed)``;
    an odd count puts the extra group in the first half)."""
    g = sorted({str(x) for x in groups})
    if len(g) < 2:
        return []
    rng = np.random.default_rng(int(seed))
    out = []
    for _ in range(int(n_halves)):
        perm = rng.permutation(len(g))
        cut = (len(g) + 1) // 2
        a = tuple(sorted(g[i] for i in perm[:cut]))
        b = tuple(sorted(g[i] for i in perm[cut:]))
        out.append((a, b))
    return out


def unit_groups(unit_of_row: pd.Series, group_of_row: pd.Series) -> dict[str, list[str]]:
    """Per unit the sorted publication groups its rows come from (rows with a missing unit or group are skipped)."""
    u = unit_of_row.astype(object)
    g = group_of_row.reindex(u.index).astype(object)
    ok = u.notna() & g.notna()
    out: dict[str, set[str]] = {}
    for uu, gg in zip(u[ok].astype(str), g[ok].astype(str)):
        out.setdefault(uu, set()).add(gg)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _row_key(unit_of_row: pd.Series, group_of_row: pd.Series) -> pd.Series:
    u = unit_of_row.astype(object)
    g = group_of_row.reindex(u.index).astype(object)
    key = pd.Series([None if (pd.isna(a) or pd.isna(b)) else f"{a}\x1f{b}" for a, b in zip(u, g)], index=u.index,
                    dtype=object)
    return key


def split_half_reliability(estimate_rows: Callable[[pd.Index], pd.Series], unit_of_row: pd.Series,
                           group_of_row: pd.Series, *, units: Sequence[str] | None = None, n_halves: int = N_SPLIT_HALVES,
                           seed: int = SPLIT_HALF_SEED, min_groups: int = MIN_GROUPS_FOR_SPLIT_HALF) -> dict[str, Any]:
    """Per-unit split-half reliability by publication group (section 8; :data:`READINGS` ``per_unit_branches``).

    ``unit_of_row`` / ``group_of_row`` label every row (same index); ``estimate_rows(index)`` returns the per-unit
    quantity computed on those rows (a Series indexed by unit).  For each of the ``n_halves`` seeded splits every unit
    with >= ``min_groups`` groups has ITS OWN groups halved (``default_rng(seed)``; an odd count puts the extra group in
    the first half); the quantity is estimated on the union of the first halves and on the union of the second halves,
    the two values are correlated across the eligible units (Pearson; Spearman printed beside) and stepped up by
    ``transfer.spearman_brown``.  The reported reliability is the mean over the halves."""
    ug = unit_groups(unit_of_row, group_of_row)
    universe = sorted(ug) if units is None else [u for u in sorted(set(str(x) for x in units)) if u in ug]
    elig = [u for u in universe if len(ug[u]) >= int(min_groups)]
    base = {"method": "split_half_by_publication_group", "min_groups": int(min_groups), "n_units_universe": len(universe),
            "n_units": len(elig), "units": elig, "floor": RELIABILITY_FLOOR, "reading": READINGS["split_half_correlation"]}
    if not elig:
        return {**base, "status": "NOT_APPLICABLE", "reliability": float("nan"), "gate": ET.reliability_gate(float("nan")),
                "detail": f"no unit has >= {min_groups} publication groups"}
    rng = np.random.default_rng(int(seed))
    key = _row_key(unit_of_row, group_of_row)
    per = []
    for k in range(int(n_halves)):
        side: dict[str, str] = {}
        for u in elig:
            gs = ug[u]
            perm = rng.permutation(len(gs))
            cut = (len(gs) + 1) // 2
            for i, gi in enumerate(perm):
                side[f"{u}\x1f{gs[gi]}"] = "a" if i < cut else "b"
        assign = key.map(side)
        rows_a, rows_b = key.index[(assign == "a").to_numpy()], key.index[(assign == "b").to_numpy()]
        xa = estimate_rows(rows_a)
        xb = estimate_rows(rows_b)
        xa = pd.Series(dtype=float) if xa is None else pd.to_numeric(xa, errors="coerce").astype(float)
        xb = pd.Series(dtype=float) if xb is None else pd.to_numeric(xb, errors="coerce").astype(float)
        a = xa.reindex(elig).to_numpy(dtype=float)
        b = xb.reindex(elig).to_numpy(dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        r = _corr(a, b)
        rs = _corr(a, b, "spearman")
        per.append({"half": k, "n_units": int(both.sum()), "r": r, "spearman": rs, "spearman_brown": ET.spearman_brown(r),
                    "spearman_brown_of_spearman": ET.spearman_brown(rs)})
    def _nanmean(xs: Sequence[float]) -> float:
        arr = np.array(list(xs), dtype=float)
        return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")

    vals = np.array([p["spearman_brown"] for p in per], dtype=float)
    rel = _nanmean(vals)
    return {**base, "status": "computed" if np.isfinite(rel) else "NOT_RUN", "n_halves": int(n_halves), "seed": int(seed),
            "reliability": rel,
            "reliability_spearman_based": _nanmean([p["spearman_brown_of_spearman"] for p in per]),
            "mean_half_correlation": _nanmean([p["r"] for p in per]), "n_halves_finite": int(np.isfinite(vals).sum()),
            "mean_units_per_half": float(np.mean([p["n_units"] for p in per])), "per_half": per,
            "gate": ET.reliability_gate(rel)}


def jackknife_reliability(full: pd.Series, leave_one_out: Mapping[str, pd.Series],
                          unit_groups_: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """Jackknife-by-publication reliability (section 8): between-unit variance / (between-unit variance + mean
    within-unit SE^2), with the within-unit SE^2 from the delete-one-publication estimates
    (:data:`READINGS` ``jackknife_se``).  With ``unit_groups_`` a unit's SE^2 uses only the estimates that left out one
    of ITS OWN publication groups (deleting a foreign group changes nothing and would deflate the SE)."""
    if full is None or not len(full):
        return {"method": "jackknife_by_publication", "status": "NOT_RUN", "reliability": float("nan"),
                "gate": ET.reliability_gate(float("nan")), "detail": "no full-sample per-unit value"}
    x = pd.to_numeric(full, errors="coerce").astype(float)
    between = float(np.nanvar(x.to_numpy(dtype=float), ddof=1)) if x.notna().sum() > 1 else float("nan")
    se2: dict[Any, float] = {}
    n_used: dict[Any, int] = {}
    for unit in x.index:
        gs = None if unit_groups_ is None else set(str(g) for g in unit_groups_.get(str(unit), ()))
        vals = np.array([float(s.get(unit, np.nan)) for g, s in leave_one_out.items() if gs is None or str(g) in gs],
                        dtype=float)
        vals = vals[np.isfinite(vals)]
        g = vals.size
        if g < 2:
            continue
        se2[unit] = float((g - 1) / g * float(((vals - vals.mean()) ** 2).sum()))
        n_used[unit] = g
    mean_se2 = float(np.mean(list(se2.values()))) if se2 else float("nan")
    rel = ET.jackknife_reliability(between, mean_se2)
    return {"method": "jackknife_by_publication", "status": "computed" if se2 else "NOT_RUN",
            "n_units": int(len(x)), "n_units_with_se": len(se2), "n_publications_left_out": len(leave_one_out),
            "between_unit_variance": between, "mean_within_unit_se2": mean_se2, "reliability": rel,
            "floor": RELIABILITY_FLOOR, "gate": ET.reliability_gate(rel),
            "per_unit_se2": {str(k): v for k, v in sorted(se2.items(), key=lambda kv: str(kv[0]))},
            "per_unit_n_jackknife": {str(k): v for k, v in sorted(n_used.items(), key=lambda kv: str(kv[0]))},
            "reading": READINGS["jackknife_se"]}


def jackknife_by_unit(estimate_rows: Callable[[pd.Index], pd.Series], full: pd.Series, unit_of_row: pd.Series,
                      group_of_row: pd.Series, *, units: Sequence[str] | None = None,
                      min_groups: int = MIN_GROUPS_FOR_JACKKNIFE, max_groups: int = MIN_GROUPS_FOR_SPLIT_HALF - 1
                      ) -> dict[str, Any]:
    """The jackknife branch of section 8 for the units with ``min_groups .. max_groups`` publication groups (2-3 by
    default): every group any of them has is deleted in turn from THEIR rows only, ``estimate_rows`` re-estimates them,
    and :func:`jackknife_reliability` uses each unit's own delete-one values.  Units with a single group are counted
    (``n_units_single_group``) and have no reliability."""
    ug = unit_groups(unit_of_row, group_of_row)
    universe = sorted(ug) if units is None else [u for u in sorted(set(str(x) for x in units)) if u in ug]
    elig = [u for u in universe if int(min_groups) <= len(ug[u]) <= int(max_groups)]
    single = [u for u in universe if len(ug[u]) < int(min_groups)]
    base = {"method": "jackknife_by_publication", "min_groups": int(min_groups), "max_groups": int(max_groups),
            "n_units_universe": len(universe), "n_units": len(elig), "units": elig, "n_units_single_group": len(single),
            "units_single_group": single}
    if not elig:
        return {**base, "status": "NOT_APPLICABLE", "reliability": float("nan"), "gate": ET.reliability_gate(float("nan")),
                "floor": RELIABILITY_FLOOR, "n_publications_left_out": 0,
                "detail": f"no unit has {min_groups}-{max_groups} publication groups"}
    u_str = unit_of_row.astype(object).astype(str)
    g_str = group_of_row.reindex(unit_of_row.index).astype(object).astype(str)
    in_elig = u_str.isin(elig).to_numpy() & unit_of_row.notna().to_numpy() & group_of_row.reindex(unit_of_row.index).notna().to_numpy()
    rows = unit_of_row.index[in_elig]
    groups = sorted({g for u in elig for g in ug[u]})
    loo: dict[str, pd.Series] = {}
    for g in groups:
        idx = rows[(g_str.loc[rows] != g).to_numpy()]
        est = estimate_rows(idx)
        loo[g] = pd.Series(dtype=float) if est is None else pd.to_numeric(est, errors="coerce").astype(float).reindex(elig)
    full_e = pd.to_numeric(full, errors="coerce").astype(float).reindex(elig) if full is not None else pd.Series(dtype=float)
    rec = jackknife_reliability(full_e, loo, {u: ug[u] for u in elig})
    return {**rec, **base, "n_units": len(elig)}


def reliability_of(quantity: str, *, estimate_rows: Callable[[pd.Index], pd.Series], full: pd.Series,
                   unit_of_row: pd.Series, group_of_row: pd.Series, n_halves: int = N_SPLIT_HALVES,
                   seed: int = SPLIT_HALF_SEED) -> dict[str, Any]:
    """Section 8 reliability of one derived per-unit quantity, per unit (:data:`READINGS` ``per_unit_branches``):
    :func:`split_half_reliability` over the units with >= 4 publication groups, :func:`jackknife_by_unit` over the units
    with 2-3, single-group units counted; ``reliability`` = the minimum over the computed branches (both printed).
    The unit universe is ``full.index`` (units with a full-sample estimate); rows of other units are ignored."""
    if quantity not in RELIABILITY_QUANTITIES:
        raise ValueError(f"{quantity!r} is not in the section 8 reliability scope {list(RELIABILITY_QUANTITIES)}")
    units = [str(u) for u in (full.index if full is not None else [])]
    ug = unit_groups(unit_of_row, group_of_row)
    sh = split_half_reliability(estimate_rows, unit_of_row, group_of_row, units=units, n_halves=n_halves, seed=seed)
    jk = jackknife_by_unit(estimate_rows, full, unit_of_row, group_of_row, units=units)
    computed = {name: r for name, r in (("split_half", sh), ("jackknife", jk)) if r["status"] == "computed"}
    vals = [r["reliability"] for r in computed.values() if np.isfinite(r["reliability"])]
    rel = float(min(vals)) if vals else float("nan")
    method_of = {u: ("split_half" if u in set(sh["units"]) else "jackknife" if u in set(jk["units"]) else
                     "single_group" if u in set(jk["units_single_group"]) else "no_rows") for u in units}
    n_no_rows = sum(1 for m in method_of.values() if m == "no_rows")
    return {"quantity": quantity, "scope": "section 8 reliability before correlation",
            "method": "per unit: split_half_by_publication_group (>= 4 groups) | jackknife_by_publication (2-3 groups)",
            "status": "computed" if vals else "NOT_RUN", "reliability": rel,
            "reliability_rule": "minimum over the computed branches (conservative)",
            "reliability_split_half": sh["reliability"], "reliability_jackknife": jk["reliability"],
            "n_units": len(units), "n_units_split_half": sh["n_units"], "n_units_jackknife": jk["n_units"],
            "n_units_single_group": jk["n_units_single_group"], "n_units_without_rows": n_no_rows,
            "n_groups": len({g for u in units for g in ug.get(u, ())}), "n_halves": int(n_halves), "seed": int(seed),
            "n_publications_left_out": jk.get("n_publications_left_out"),
            "between_unit_variance": jk.get("between_unit_variance"), "mean_within_unit_se2": jk.get("mean_within_unit_se2"),
            "split_half": sh, "jackknife": jk, "per_unit_method": method_of, "floor": RELIABILITY_FLOOR,
            "gate": ET.reliability_gate(rel), "reading": READINGS["per_unit_branches"],
            "detail": "; ".join(f"{k}: {r['detail']}" for k, r in (("split_half", sh), ("jackknife", jk)) if r.get("detail"))}


def gate_correlation(reliability: float, correlation: Mapping[str, Any] | float, *, name: str = "") -> dict[str, Any]:
    """A correlation printed with its gate: below the 0.3 floor it may neither support nor close a claim and is
    reported UNDECIDED (unreliable)."""
    gate = ET.reliability_gate(float(reliability))
    body = dict(correlation) if isinstance(correlation, Mapping) else {"value": float(correlation)}
    return {"name": name, "reliability": float(reliability), "floor": RELIABILITY_FLOOR, "gate": gate,
            "reportable": gate == "RELIABLE", "correlation": body,
            "verdict": "UNDECIDED (unreliable)" if gate != "RELIABLE" else "readable",
            "rule": "section 8: a quantity below the 0.3 reliability floor may neither support nor close a "
                    "correlation-based claim"}


# --------------------------------------------------------------------------------------------- #
# Procrustes-aligned bootstrap stability of an embedding (section 8, brief figures 10-11)
# --------------------------------------------------------------------------------------------- #

def procrustes_align(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """``other`` rotated (and reflected) onto ``reference`` by the orthogonal Procrustes solution on centred matrices
    (no scaling); both are ``n_units x d`` on the same units in the same order."""
    a, b = np.asarray(reference, dtype=float), np.asarray(other, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"procrustes_align: shapes differ ({a.shape} vs {b.shape})")
    ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(bc.T @ ac, full_matrices=False)
    return bc @ (u @ vt)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((na > 0) & (nb > 0), (a * b).sum(axis=1) / (na * nb), np.nan)


def _aligned_stability(ref: pd.DataFrame, replicates: Sequence[pd.DataFrame], cols: Sequence[str], *,
                       permute_seed: int | None = None) -> tuple[dict[Any, float], int]:
    """Per-unit mean cosine similarity of the Procrustes-aligned replicates with ``ref``.  ``permute_seed`` shuffles each
    replicate's unit labels first, which gives the null: the stability expected when no unit corresponds."""
    per_unit: dict[Any, list[float]] = {u: [] for u in ref.index}
    rng = None if permute_seed is None else np.random.default_rng(int(permute_seed))
    n_used = 0
    for rep in replicates:
        common = ref.index.intersection(rep.index)
        if len(common) < 3:
            continue
        use = [c for c in cols if c in rep.columns]
        if len(use) != len(cols):
            continue
        other = rep.loc[common, use].to_numpy(dtype=float)
        if rng is not None:
            other = other[rng.permutation(len(common))]
        aligned = procrustes_align(ref.loc[common].to_numpy(dtype=float), other)
        centred = ref.loc[common].to_numpy(dtype=float) - ref.loc[common].to_numpy(dtype=float).mean(axis=0)
        for u, c in zip(common, _cosine(centred, aligned)):
            if np.isfinite(c):
                per_unit[u].append(float(c))
        n_used += 1
    return {u: float(np.mean(v)) for u, v in per_unit.items() if v}, n_used


def embedding_stability(reference: pd.DataFrame, replicates: Sequence[pd.DataFrame], *,
                        null_seed: int = SPLIT_HALF_SEED) -> dict[str, Any]:
    """Procrustes-aligned bootstrap stability of an embedding table (:data:`READINGS` ``embedding_stability``):
    every replicate is aligned to ``reference`` on the units they share, and a unit's stability is the mean cosine
    similarity of its aligned replicate vectors with its reference vector.  Gated by the 0.3 floor.

    The alignment MAXIMISES agreement, so the statistic is biased upward and its no-correspondence value is not 0: with
    ``n_units`` of order the embedding dimension it can reach 0.4 (measured: 0.42 at 12 units x 4 dimensions, 0.23 at 30
    x 4, 0.27 at 60 x 8, 0.23 at 200 x 16).  ``null_stability`` is therefore computed by permuting each replicate's unit
    labels before aligning -- the same statistic when no unit corresponds -- and ``stability_above_null`` is printed
    beside it.  The registered gate stays on the raw value (:data:`READINGS` ``embedding_stability``); reading the null
    into a decision needs a POST-HOC addendum, so a raw value inside the null band is flagged
    ``within_null_band`` rather than silently called reliable."""
    if reference is None or not len(reference) or not len(replicates):
        return {"method": "procrustes_aligned_bootstrap", "status": "NOT_RUN", "stability": float("nan"),
                "gate": ET.reliability_gate(float("nan")), "detail": "no reference embedding or no replicate"}
    cols = [c for c in reference.columns if pd.api.types.is_numeric_dtype(reference[c])]
    ref = reference[cols]
    means, n_used = _aligned_stability(ref, replicates, cols)
    null_means, _ = _aligned_stability(ref, replicates, cols, permute_seed=null_seed)
    stab = float(np.mean(list(means.values()))) if means else float("nan")
    null = float(np.mean(list(null_means.values()))) if null_means else float("nan")
    within = bool(np.isfinite(stab) and np.isfinite(null) and stab <= null)
    return {"method": "procrustes_aligned_bootstrap", "status": "computed" if means else "NOT_RUN",
            "stability": stab, "null_stability": null,
            "stability_above_null": (stab - null) if (np.isfinite(stab) and np.isfinite(null)) else float("nan"),
            "within_null_band": within, "null_seed": int(null_seed),
            "n_replicates": int(n_used), "n_units": int(len(means)), "n_dimensions": int(len(cols)),
            "per_unit_stability": {str(k): v for k, v in sorted(means.items(), key=lambda kv: str(kv[0]))},
            "floor": RELIABILITY_FLOOR, "gate": ET.reliability_gate(stab), "reading": READINGS["embedding_stability"],
            "note": ("the Procrustes statistic is upward-biased: compare it with null_stability (unit labels permuted) "
                     "before reading it as agreement" + (" -- this value is INSIDE the null band" if within else ""))}


# --------------------------------------------------------------------------------------------- #
# the quantities the runner can compute without a learned fit
# --------------------------------------------------------------------------------------------- #

def logsf_amplitudes(pairs: pd.DataFrame, *, system_col: str = EM.SYSTEM_COL) -> pd.Series:
    """Per-system logSF amplitude: the median |observed logSF| of the system's comparable pairs (the section 8
    reliability unit is the system)."""
    if pairs is None or pairs.empty:
        return pd.Series(dtype=float)
    g = pairs.groupby(pairs[system_col].astype(str), sort=True)["logsf_obs"]
    return g.agg(lambda s: float(np.median(np.abs(pd.to_numeric(s, errors="coerce").dropna())))).rename("logsf_amplitude")


def b7_slope_series(slopes: pd.DataFrame, column: str, *, unit_cols: Sequence[str] = ("system", "anion")) -> pd.Series:
    """One B7 slope (``n`` or ``p_eff``) per fitted (system, anion) unit, only where the axis was fitted (an ``assumed``
    slope is the prior, not an estimate)."""
    if slopes is None or slopes.empty or column not in slopes.columns:
        return pd.Series(dtype=float)
    fr = slopes
    status = f"{column}_status" if column == "n" else "p_eff_status"
    if status in fr.columns:
        fr = fr[fr[status].astype(str) == "fitted"]
    idx = pd.Index(fr[list(unit_cols)].astype(str).agg(" | ".join, axis=1))
    return pd.Series(pd.to_numeric(fr[column], errors="coerce").to_numpy(dtype=float), index=idx, name=column)


def support_component_series(support: pd.DataFrame, component: str, *, unit_cols: Sequence[str] = EM.CELL_COLS
                             ) -> pd.Series:
    """One support-score component per cell (the mean over the cell's rows; the components are target-free)."""
    col = component if component in (support.columns if support is not None else ()) else f"support_{component}"
    if support is None or support.empty or col not in support.columns:
        return pd.Series(dtype=float)
    idx = support[list(unit_cols)].astype(str).agg(EM.UNIT_KEY_SEP.join, axis=1)
    return pd.to_numeric(support[col], errors="coerce").groupby(idx.to_numpy(), sort=True).mean().rename(component)


def reliability_table(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """One row per reliability record (the report table of section 8)."""
    rows = []
    for r in records:
        rows.append({"quantity": r.get("quantity"), "unit": r.get("unit"), "method": r.get("method"),
                     "status": r.get("status"), "reliability": r.get("reliability", r.get("stability")),
                     "reliability_split_half": r.get("reliability_split_half"),
                     "reliability_jackknife": r.get("reliability_jackknife"),
                     "floor": r.get("floor", RELIABILITY_FLOOR), "gate": r.get("gate"), "n_units": r.get("n_units"),
                     "n_units_split_half": r.get("n_units_split_half"), "n_units_jackknife": r.get("n_units_jackknife"),
                     "n_units_single_group": r.get("n_units_single_group"),
                     "n_groups": r.get("n_groups"), "n_halves": r.get("n_halves"),
                     "n_publications_left_out": r.get("n_publications_left_out"),
                     "between_unit_variance": r.get("between_unit_variance"),
                     "mean_within_unit_se2": r.get("mean_within_unit_se2"), "rows_read": r.get("rows_read", ""),
                     "detail": r.get("detail", "")})
    return pd.DataFrame(rows, columns=["quantity", "unit", "method", "status", "reliability", "reliability_split_half",
                                       "reliability_jackknife", "floor", "gate", "n_units", "n_units_split_half",
                                       "n_units_jackknife", "n_units_single_group", "n_groups", "n_halves",
                                       "n_publications_left_out", "between_unit_variance", "mean_within_unit_se2",
                                       "rows_read", "detail"])


def power_root(out_root: Path) -> Path:
    return Path(out_root) / "evaluation" / "power"


def json_safe(obj: Any) -> Any:
    return H3.json_safe(obj)

"""``scripts/g19_score_discovery.py`` -- aggregate the discovery predictions on the SELECTION half (pre-registration
sections 4, 6, 7 item 4, 8, 9 S1, 12, 19; sealed 2026-09-15).

Reads the per-fold records of ``scripts/g19_run_discovery.py`` (``evaluation/discovery/<arm>/...``) and the pre-seal
predictions of the deterministic comparators (``evaluation/preseal/predictions``, read with a parquet row filter
``half == "S"``: confirmation-half rows are never materialised), and writes

* ``tables/discovery_summary.csv`` / ``.md`` -- section 4 metrics per arm, design, variant and seed under the registered
  averaging unit (V1: outer fold; V5: hidden cell; V2: metal state), interval metrics where intervals exist, scoring
  filters and V5 strata;
* ``tables/discovery_comparator_intervals.csv`` -- B0 / B3x / B3i coverage and width per discovery seed and their mean
  (section 15 resolution: seed 104729 from the pre-seal run, the other seeds from the comparator-interval jobs);
* ``evaluation/discovery/contrasts_registered.csv`` / ``contrasts_exploratory.csv`` (column ``family``, one row per
  registered cluster unit, BH-adjusted p beside the raw p) and ``r19_items.csv``;
* ``tables/discovery_pair_summary.csv`` -- V5-PAIR logSF / direction of M2 where run, and the S1(c) selection-half
  counterweight (B3x / B3i re-fitted on the same batched folds, ``models.s1c_yardsticks``);
* ``evaluation/discovery/decisions/decisions.json`` -- the stop rule (section 7 item 4), the ladder (section 6), S1(a) /
  S1(b) components labelled "discovery, optimistically biased", the freezing screen; ``decisions/stop_rule.json``; and
  the freezing candidates merged into ``decisions/plan_state.json`` (the runner's conditional jobs).

**POST-HOC addendum 1** (below the sealed footer): discovery runs seed 104729 only, so R19 item 4 is NOT_EVALUATED for
every learned-arm contrast in every table, and R19 item 6 of such a contrast is the *reduced sensitivity set (addendum
1)* -- the V5 strict and HNO3-only refits that were run plus every registered scoring-filter sensitivity -- with the
sensitivities that were not run named in the contrast tables and in ``decisions.json`` -> ``addendum_1`` /
``not_run.addendum_1``.  Closed-form arms keep the full registered set.  The stop rule, the ladder and the freezing
screen are unchanged: their scopes contain neither item 4 nor the refit sensitivities.

Every scoring frame passes the selection-half assertion and ``registered.assert_not_scored``; nothing reads V6.  The
scorer refuses to run unless the pre-registration seal check passes with the registered digest
(``g19_run_discovery.refuse_unless_sealed``), and reads a discovery record only when its digest, fold hash and fold set
are exactly what the current code, fold files and plan state produce (``g19_run_discovery.verified_predictions``): a
stale or foreign record raises, a job with folds still missing is not scored.  When the re-coloured batched-vs-exact
check failed, every S1 component is reported UNDECIDED and heavy-arm V5 contrasts carry ``batched (check failed)``
(section 7 item 6).  Registered contrasts not evaluated (section 19), the BH family size and the uncertainty metrics
that need a predictive SD are listed as not run.

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_score_discovery.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.chemistry import support_graph as SG  # noqa: E402
from gen19ct.evaluation import calibration as EC  # noqa: E402
from gen19ct.evaluation import discovery as D  # noqa: E402
from gen19ct.evaluation import metrics as EM  # noqa: E402
from gen19ct.evaluation import pairs as EP  # noqa: E402
from gen19ct.evaluation import transfer as ET  # noqa: E402
from gen19ct.folds import io as FI  # noqa: E402
from gen19ct.folds import source_holdout as SH  # noqa: E402
from gen19ct.manifest import Run, write_csv, write_json, write_text  # noqa: E402

NAME = "g19_score_discovery"
PRESEAL_PRED = paths.G19_ROOT / "evaluation" / "preseal" / "predictions"
CROSSINGS_CSV = paths.FOLDS_DIR / "wildcard_copy_crossings.csv"
WC_COL = "wildcard_copy_partner_in_training"
#: pre-seal prediction job of every comparator setting
PRESEAL_JOB: dict[tuple[str, str], tuple[str, str]] = {
    ("V5", "primary"): ("V5__primary", "V5__primary__exact"), ("V5", "loose"): ("V5__loose", "V5__loose__exact"),
    ("V5", "strict"): ("V5__strict", "V5__strict__exact"), ("V5", "hno3_only"): ("V5__hno3_only", "V5__hno3_only__exact"),
    ("V5", "cell_only"): ("V5__cell_only", "V5__cell_only__exact"),
    ("V5", "parent_structure"): ("V5__parent_structure", "V5__parent_structure__exact"),
    ("V5", "sr_iii_dropped"): ("V5__sr_iii_dropped_training", "V5__primary__exact"),
    ("V5", "V5P"): ("V5P__base", "V5P__base__cell_x_group"),
    ("V1", "primary"): ("V1__copy", "V1__copy__exact"), ("V2", "primary"): ("V2__element", "V2__element__exact"),
    ("V1", "sr_iii_dropped"): ("V1__sr_iii_dropped_training", "V1__copy__exact"),
    ("V1", "near_duplicate_key"): ("V1__near_duplicate_key", "V1__near_duplicate_key__exact"),
    ("V1", "compilation_doi"): ("V1__compilation_doi", "V1__compilation_doi__exact"),
    ("V2", "state"): ("V2__state", "V2__state__exact"),
    ("V2", "sr_iii_dropped"): ("V2__sr_iii_dropped_training", "V2__element__exact")}
DETERMINISTIC_ARMS = frozenset({"B0", "B1", "B2", "B3", "B3x", "B3i", "B3l", "B4", "B4x", "B4l", "B7"})
GROUPING_UNTESTABLE = ("UNTESTABLE: a learned arm's V1 grouping sensitivity needs outer folds and an inner design under "
                       "that grouping, which are not built (discovery.V1_GROUPING_SETTINGS)")


def _runner_module():
    name = "g19_run_discovery"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================================= #
# row attributes
# ============================================================================================= #

def _preseal_module():
    name = "g19_run_preseal"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_attrs() -> pd.DataFrame:
    """Per MODEL row (index ``canonical_measurement_id``): the pre-seal run's attributes (target, cell, publication
    group, condition key, strata, section 2 flags) plus the registered half of every design."""
    feas = json.loads((paths.DATA_AUDIT_DIR / "feasibility.json").read_text(encoding="utf-8"))
    attrs = _preseal_module().build_attrs(feas)
    fr = attrs.assign(**{FI.GROUP_COL: attrs[EM.PUB_GROUP_COL].astype(str)})
    for design in ("V5", "V5P", "V5PAIR", "V1", "V2"):
        attrs[f"registered_half_{design}"] = D.registered_row_halves(fr, design).to_numpy()
    attrs[f"registered_half_V5-P"] = attrs["registered_half_V5P"]
    return attrs


def wildcard_flags(pred: pd.DataFrame, stem: str, crossings: pd.DataFrame) -> pd.DataFrame:
    """Per prediction row: a ``leakage.wildcard_copy_pairs`` partner in its own fold's training rows (any / strict)."""
    sub = crossings[crossings["stem"] == stem]
    any_keys = set(zip(sub["fold_id"].astype(str), sub["row_id"].astype(str)))
    strict = sub[sub["strict_copy"].astype(str).str.lower() == "true"]
    strict_keys = set(zip(strict["fold_id"].astype(str), strict["row_id"].astype(str)))
    keys = list(zip(pred["fold_id"].astype(str), pred["row_id"].astype(str)))
    return pred.assign(**{WC_COL: [k in any_keys for k in keys], WC_COL + "_strict": [k in strict_keys for k in keys]})


# ============================================================================================= #
# prediction store
# ============================================================================================= #

class Store:
    """Scoring frames of discovery arms and pre-seal comparators, selection half only."""

    def __init__(self, out_root: Path, attrs: pd.DataFrame, v6: pd.Series, crossings: pd.DataFrame, state: D.PlanState,
                 preseal_dir: Path = PRESEAL_PRED, *, code: str | None = None, folds_dir: Path | None = None,
                 runners: Mapping[str, Any] | None = None):
        self.out_root, self.attrs, self.v6, self.crossings, self.state = Path(out_root), attrs, v6, crossings, state
        self.preseal_dir = preseal_dir
        self.rd = _runner_module()
        self.code = self.rd.current_code_digest() if code is None else code
        self.folds_dir = paths.FOLDS_DIR if folds_dir is None else Path(folds_dir)
        self.runners = runners
        flag = attrs["acidic_coextractant_modifier"].astype(bool) if "acidic_coextractant_modifier" in attrs.columns \
            else pd.Series(False, index=attrs.index)
        self.excluded_ids = sorted(attrs.index[flag.to_numpy()].astype(str))
        self._pre: dict = {}
        self._cache: dict = {}
        self._folds: dict = {}
        self.record_sets: dict[str, dict] = {}
        self.remainder_groups = sorted(
            g for g, n in attrs[EM.PUB_GROUP_COL].astype(str).value_counts().items() if n < SH.REGISTERED_MIN_ROWS)

    def verified(self, arm: str, design_dir: str, seed: int, steps: Sequence[str] = ("point",)) -> pd.DataFrame | None:
        """``g19_run_discovery.verified_predictions`` (stale / foreign records raise; incomplete -> ``None``)."""
        pred, st = self.rd.verified_predictions(self.out_root, arm, design_dir, int(seed), code=self.code,
                                                state=self.state, excluded_ids=self.excluded_ids,
                                                folds_dir=self.folds_dir, runners=self.runners, steps=steps,
                                                fold_cache=self._folds)
        self.record_sets[f"{arm}/{design_dir}/s{seed}"] = {k: v for k, v in st.items() if k != "arm"}
        return pred

    def design_dir(self, arm: str, design: str, setting: str) -> str | None:
        v5s = self.state.heavy_v5_scheme or "batched"
        v1s = self.state.heavy_v1_scheme or "grouped10"
        if design == "V1":
            scheme = "exact" if arm in D.B6_ARMS else v1s
            if setting == "primary":
                return f"V1__copy_{scheme}"
            if setting in D.V1_REFIT_SETTINGS:
                variant, drop, _ = D.V1_REFIT_SETTINGS[setting]
                return f"V1__{variant}_{scheme}" + (D.SR_DROPPED_SUFFIX if drop else "")
            return None
        if design == "V2":
            if setting == "primary":
                return "V2__element_exact"
            if setting in D.V2_REFIT_SETTINGS:
                variant, drop, _ = D.V2_REFIT_SETTINGS[setting]
                return f"V2__{variant}_exact" + (D.SR_DROPPED_SUFFIX if drop else "")
            return None
        if arm in D.B6_ARMS:
            if design == "V5":
                if setting == "V5P":
                    return "V5P__base_cell_x_group"
                if setting == "sr_iii_dropped":
                    return "V5__primary_exact_sr_iii_dropped"
                return f"V5__{setting}_exact"
            return None
        if design == "V5":
            if setting == "V5P":
                return "V5P__base_batched"
            if setting == "sr_iii_dropped":
                return f"V5__primary_{v5s}_sr_iii_dropped"
            return f"V5__{setting}_{v5s}"
        return {"V5PAIR": "V5PAIR__primary_batched"}.get(design)

    def v1_scheme(self, arm: str, design: str) -> str:
        if design != "V1" or arm in DETERMINISTIC_ARMS or arm in D.B6_ARMS:
            return "exact"
        return "grouped" if (self.state.heavy_v1_scheme or "grouped10") == "grouped10" else "exact"

    def frame(self, arm: str, design: str, setting: str, seed: int | None) -> pd.DataFrame | None:
        """The scoring frame (``discovery.scoring_frame``) or ``None`` when the arm has no predictions there."""
        arm = D.ARM_ALIASES.get(arm, arm)
        key = (arm, design, setting, seed if arm not in DETERMINISTIC_ARMS else None)
        if key in self._cache:
            return self._cache[key]
        label = "V5-P" if setting == "V5P" else design
        if arm in DETERMINISTIC_ARMS:
            src = PRESEAL_JOB.get((design, setting))
            if src is None:
                return None
            job, stem = src
            if job not in self._pre:
                p = self.preseal_dir / f"{job}.parquet"
                self._pre[job] = D.read_preseal_selection(p) if p.exists() else None
            pre = self._pre[job]
            if pre is None:
                return None
            pred = pre[pre["arm"] == arm]
            if pred.empty:
                return None
        else:
            dd = self.design_dir(arm, design, setting)
            if dd is None or seed is None:
                return None
            pred = self.verified(arm, dd, int(seed))
            if pred is None:
                return None
            d_, v_, s_ = pred["design"].iloc[0], pred["variant"].iloc[0], pred["scheme"].iloc[0]
            stem = FI.design_stem(d_, v_, s_)
        pred = wildcard_flags(pred, stem, self.crossings)
        hc = f"registered_half_{'V5' if label in ('V5', 'V5-P') else design}"
        fr = D.scoring_frame(pred, self.attrs, design=label, v6_mask=self.v6, what=f"{arm}/{design}/{setting}/s{seed}",
                             half_col=hc)
        self._cache[key] = fr
        return fr


# ============================================================================================= #
# contrasts
# ============================================================================================= #

def _pair(store: Store, cand: str, comp: str, design: str, setting: str, seed: int, *, filt: str = "none",
          v2_summary: str = "focus7"):
    c = store.frame(cand, design, setting, seed)
    k = store.frame(comp, design, setting, seed)
    if c is None or k is None:
        return None
    if design == "V2" and v2_summary == "focus7":                  # section 3.3 primary summary (READINGS v2_summary)
        c = c[c[EM.METAL_STATE_COL].isin(D.V2_FOCUS7)]
        k = k[k[EM.METAL_STATE_COL].isin(D.V2_FOCUS7)]
        if c.empty:
            return None
    if filt != "none":
        c, k = D.filtered_pair(c, k, filt)
        if c.empty:
            return None
    label = "V5-P" if setting == "V5P" else design
    return D.paired_units(c, k, label, candidate=cand, comparator=comp, v6_mask=store.v6,
                          cand_v1_scheme=store.v1_scheme(D.ARM_ALIASES.get(cand, cand), design),
                          comp_v1_scheme=store.v1_scheme(D.ARM_ALIASES.get(comp, comp), design),
                          remainder_groups=store.remainder_groups)


def is_learned(*arms: str) -> bool:
    """Whether a contrast involves a learned arm (addendum 1 items 3-4: seed 104729 only, reduced sensitivity set)."""
    return any(D.ARM_ALIASES.get(a, a) not in DETERMINISTIC_ARMS for a in arms)


def evaluate_spec(store: Store, spec: D.ContrastSpec, delta5: float, *, v2_summary: str = "focus7"
                  ) -> dict[str, Any] | None:
    """R19 of one contrast on the selection half (``discovery.evaluate_contrast``); ``None`` when an arm is not run.

    Addendum 1: a contrast with a learned arm is evaluated on discovery seed 104729 only (item 3, R19 item 4
    NOT_EVALUATED), and its R19 item 6 on the reduced sensitivity set -- the V5 strict and HNO3-only refits that were
    run plus every registered scoring-filter sensitivity (item 4) -- with the sensitivities that were not run named.
    A contrast between closed-form arms keeps the full registered set and all five discovery seeds."""
    if "M3" in (spec.candidate, spec.comparator):
        return None
    design = spec.design
    kw = {"v2_summary": v2_summary}
    primary = _pair(store, spec.candidate, spec.comparator, design, "primary", D.PRIMARY_SEED, **kw)
    if primary is None:
        return None
    learned = is_learned(spec.candidate, spec.comparator)
    seeds = {}
    for s in (D.PLAN_SEEDS if learned else D.DISCOVERY_SEEDS):
        pu = primary if s == D.PRIMARY_SEED else _pair(store, spec.candidate, spec.comparator, design, "primary", s, **kw)
        seeds[s] = None if pu is None else pu.delta
    sens: dict[str, Any] = {}
    reg = ET.REGISTERED_SENSITIVITIES[design]
    reduced: list[str] = []
    for name in D.SCORING_FILTER_SENSITIVITIES:
        if name in reg:
            pu = _pair(store, spec.candidate, spec.comparator, design, "primary", D.PRIMARY_SEED, filt=name, **kw)
            sens[name] = ET.UNTESTABLE if pu is None else pu.delta
            reduced.append(name)
    reasons: dict[str, str] = {}
    not_run = D.LEARNED_REFITS_NOT_RUN.get(design, {}) if learned else {}
    if design == "V5":
        for setting, name in D.V5_SETTING_SENSITIVITY.items():
            if name in not_run:                       # addendum 1 item 4: not run at all for a learned arm
                sens[name] = ET.UNTESTABLE
                reasons[name] = not_run[name]
                continue
            pu = _pair(store, spec.candidate, spec.comparator, design, setting, D.PRIMARY_SEED)
            sens[name] = ET.UNTESTABLE if pu is None else pu.delta
            reduced.append(name)
    elif design in ("V1", "V2"):
        table = D.V1_REFIT_SETTINGS if design == "V1" else D.V2_REFIT_SETTINGS
        for setting, (_, _, name) in table.items():
            if name in not_run:
                sens[name] = ET.UNTESTABLE
                reasons[name] = not_run[name]
                continue
            if design == "V1" and setting in D.V1_GROUPING_SETTINGS and learned:
                sens[name] = ET.UNTESTABLE
                reasons[name] = GROUPING_UNTESTABLE
                continue
            pu = _pair(store, spec.candidate, spec.comparator, design, setting, D.PRIMARY_SEED, **kw)
            sens[name] = ET.UNTESTABLE if pu is None else pu.delta
            reduced.append(name)
    res = D.evaluate_contrast(name=spec.name, family=spec.family, design=design, primary=primary,
                              margin=D.margin_value(spec, delta5), seed_deltas=seeds, sensitivities=sens,
                              learned=learned, reduced_sensitivities=sorted(set(reduced)) if learned else None)
    res["note"] = spec.note
    res["untestable_reason"] = {k: reasons.get(k, D.REFIT_NOT_RUN) for k, v in res["sensitivities"].items()
                                if v == ET.UNTESTABLE}
    res["batching_label"] = D.heavy_v5_batching_label((spec.candidate, spec.comparator), design, store.state)
    forced = store.state.s1_forced_undecided and spec.family in D.S1_FAMILIES
    res["reported_verdict"] = "UNDECIDED" if forced else res["r19"].verdict
    return res


def ladder(store: Store, delta5: float, results: dict[str, dict]) -> dict[str, Any]:
    """Section 6 ladder M1, M2 (M3+ not implemented) against the retained predecessor."""
    out: dict[str, Any] = {"M0": {"step": "M0", "kept": True, "note": "base (B5)"}}
    for step in ("M1", "M2"):
        pred = D.retained_predecessor(out, step)
        per = {}
        for design in ("V5", "V1", "V2"):
            spec = D.ContrastSpec("ladder", step, pred, design, "delta5" if design == "V5" else "0.05",
                                  f"ladder step {step} vs retained predecessor {pred}")
            r = evaluate_spec(store, spec, delta5)
            per[design] = r
            if r is not None:
                results[f"{spec.name}@{design}#ladder"] = r
        out[step] = D.ladder_step(step, pred, per["V5"], None if per["V1"] is None else per["V1"]["tost"],
                                  None if per["V2"] is None else per["V2"]["tost"])
        pending = [p for p in D.LADDER[1:D.LADDER.index(step)] if out.get(p, {}).get("kept") is None]
        if pending:
            out[step]["predecessor_provisional"] = f"the keep decision of {pending} is pending"
    out["M3+"] = {"step": "M3-M7", "kept": None, "note": D.NOT_IMPLEMENTED}
    return out


# ============================================================================================= #
# section 4 metric tables
# ============================================================================================= #

def summary_tables(store: Store) -> pd.DataFrame:
    parts = []
    arms = D.FITTED_ARMS
    for arm in arms:
        # addendum 1 items 3-4: a learned arm runs seed 104729 and the V5 strict / HNO3-only refits only
        for design, settings in (("V5", ("primary",) + D.LEARNED_REFIT_VARIANTS),
                                 ("V1", ("primary",)), ("V2", ("primary",))):
            for setting in settings:
                for s in D.PLAN_SEEDS:
                    fr = store.frame(arm, design, setting, s)
                    if fr is None:
                        continue
                    label = "V5-P" if setting == "V5P" else design
                    reg = EM.Regime(design=label, arm=arm, variant=setting, half="selection", seed=int(s),
                                    seed_set="discovery")
                    kw = dict(v6_mask=store.v6)
                    if design == "V1":
                        kw.update(v1_scheme=store.v1_scheme(arm, design), v1_group_col=EM.PUB_GROUP_COL,
                                  remainder_groups=store.remainder_groups)
                    for fname in ("none",) + tuple(n for n in D.SCORING_FILTER_SENSITIVITIES
                                                  if n in ET.REGISTERED_SENSITIVITIES[label if label != "V5-P" else "V5-P"]):
                        frx = D.apply_scoring_filter(fr, fname, wildcard_col=WC_COL)
                        if frx.empty:
                            continue
                        status = "registered" if fname == "none" else "registered_sensitivity"
                        r2 = reg if status == "registered" else replace(reg, status="exploratory")
                        s1 = EM.design_logd_summary(frx, r2, exploratory_unit_readings=design == "V1", **kw)
                        s1["scoring_filter"] = fname
                        parts.append(s1)
                        if fname == "none" and design == "V5" and setting == "primary":
                            for sc in ("metal_class", "dga_stratum", "acid_stratum"):
                                st = EM.design_logd_summary(frx, reg, exploratory_unit_readings=False, strata_col=sc, **kw)
                                st = st[st["stratum"] != "all"]
                                st["scoring_filter"] = fname
                                parts.append(st)
                        if fname == "none" and np.isfinite(frx["lower_80"].to_numpy(dtype=float)).all():
                            iv = EM.design_interval_summary(frx, reg, exploratory_unit_readings=design == "V1",
                                                            levels=EC.LEVELS, **kw)
                            iv["scoring_filter"] = fname
                            parts.append(iv)
    if not parts:
        return pd.DataFrame(columns=list(EM.SUMMARY_COLUMNS) + ["unit_reading", "scoring_filter"])
    out = pd.concat(parts, ignore_index=True)
    out["label"] = "discovery, optimistically biased (selection half)"
    return out


def comparator_intervals(store: Store) -> pd.DataFrame:
    """Section 15 as amended by addendum 1 item 3: every learned comparison is on seed 104729, so the deterministic
    comparators' coverage and width are the seed-104729 (pre-seal) ones on every design and no comparator-interval job
    is run in discovery (selection half).  At confirmation they are drawn with each withheld seed and averaged."""
    rows = []
    for design, arms in D.COMPARATOR_INTERVAL_JOBS.items():
        for arm in arms:
            base = store.frame(arm, design, "primary", None)
            if base is None:
                continue
            cols = [c for lv in EC.LEVELS for c in EM.interval_columns(lv)]
            if any(c not in base.columns for c in cols) or \
                    not np.isfinite(base[cols].to_numpy(dtype=float)).any():
                continue                                   # no intervals stored for this arm and design
            frames = {D.PRIMARY_SEED: base}
            idx = base.index
            if any(set(f.index) != set(idx) for f in frames.values()):
                raise AssertionError(f"{arm}/{design}: comparator interval seeds score different rows")
            units = (base[list(EM.CELL_COLS)].astype(str).agg(" x ".join, axis=1) if design == "V5"
                     else base[EM.METAL_STATE_COL].astype(str) if design == "V2"
                     else EM.v1_scoring_units(base["fold_id"], scheme="exact"))
            y = base[EM.Y_COL].astype(float)
            met = EC.seed_mean_interval_metrics(y, {s: f.loc[idx] for s, f in frames.items()}, units=units.loc[idx])
            met.insert(0, "arm", arm)
            met.insert(1, "design", design)
            met["half"] = "selection"
            met["seed_set"] = "discovery seed 104729 only (addendum 1 item 3)"
            rows.append(met)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def coverage_by_domain_status(store: Store) -> pd.DataFrame:
    """Sections 12 / 13: coverage and width per domain-status category (with its cell count) of every fitted arm with
    intervals on V5-primary, from the per-fold support files the runner wrote (``_support``); rows whose label is
    ambiguous between the two readings of the anion clause are counted and excluded."""
    parts = []
    for arm in D.FITTED_ARMS:
        dd = store.design_dir(arm, "V5", "primary")
        for s in D.PLAN_SEEDS:
            fr = store.frame(arm, "V5", "primary", s)
            sup_dir = D.discovery_root(store.out_root) / "_support" / str(dd) / f"s{s}"
            if fr is None or not sup_dir.exists() or not np.isfinite(fr["lower_80"].to_numpy(dtype=float)).all():
                continue
            if not _support_files_current(store, arm, str(dd), s, sup_dir):
                continue
            sup = pd.concat([pd.read_parquet(p, columns=["row_id", "fold_id", "domain_status", "domain_status_ambiguous",
                                                         "support_score"]) for p in sorted(sup_dir.glob("*.parquet"))],
                            ignore_index=True)
            key = sup.set_index(["fold_id", "row_id"])
            idx = pd.MultiIndex.from_arrays([fr["fold_id"].astype(str), fr["row_id"].astype(str)])
            if not idx.isin(key.index).all():
                continue
            lab = key.reindex(idx)
            fr = fr.assign(domain_status=lab["domain_status"].to_numpy(), support_score=lab["support_score"].to_numpy(),
                           domain_status_ambiguous=lab["domain_status_ambiguous"].astype(bool).to_numpy())
            n_amb = int(fr["domain_status_ambiguous"].sum())
            reg = EM.Regime(design="V5", arm=arm, variant="primary", half="selection", seed=int(s), seed_set="discovery")
            tab = EC.coverage_by_category(fr[~fr["domain_status_ambiguous"]], reg, category_col="domain_status",
                                          unit_cols=EM.CELL_COLS, v6_mask=store.v6)
            tab["n_rows_domain_status_ambiguous_excluded"] = n_amb
            parts.append(tab)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _support_files_current(store: Store, arm: str, design_dir: str, seed: int, sup_dir: Path) -> bool:
    """Every support file of the arm's verified job carries the digest the current code gives it (raises on a stale
    one); ``False`` when a file is missing."""
    d = D.discovery_root(store.out_root) / arm / design_dir / f"s{seed}"
    rec = next((D.read_record(j) for j in sorted(d.glob("*.json"))), None)
    if rec is None:
        return False
    job = D.job_from_record(rec)
    if job.stem not in store._folds:
        store._folds[job.stem] = FI.read_design(job.stem, store.folds_dir)
    for f, _ in D.fittable_folds(job, store._folds[job.stem], store.excluded_ids):
        js = sup_dir / f"{D.safe_fold_name(f.fold_id)}.json"
        body = D.read_record(js)
        if body is None or not js.with_suffix(".parquet").exists():
            return False
        if body.get("digest") != store.rd.support_digest(job, f, store.code) or body.get("fold_hash") != f.fold_hash:
            raise D.StaleRecordError(f"{js}: support file of another code or fold file")
    return True


# ============================================================================================= #
# V5-PAIR (section 9 S1(c) selection-half counterweight)
# ============================================================================================= #

def pair_outputs(store: Store) -> tuple[pd.DataFrame | None, dict | None, list[dict]]:
    pred = store.verified("M2", "V5PAIR__primary_batched", D.PRIMARY_SEED)
    if pred is None:
        return None, None, []
    from gen19ct.data import load
    from gen19ct.folds import registered as FR
    from gen19ct.models import interface as I
    from gen19ct.models import s1c_yardsticks as SY

    stem = "V5PAIR__primary__batched"
    model = load.load_model_rows()
    fr = I.prepare_frame(model)
    fr[FI.GROUP_COL] = fr[I.PUB_GROUP_COL]
    systems, comps = I.load_descriptor_tables()
    table = I.RowTable(fr, systems=systems, components=comps)
    v6 = FR.v6_target_mask(model, FR.v6_system_set(model)).reindex(fr.index).fillna(False).astype(bool)
    coext = fr[FI.ROW_ID].astype(str).isin(set(store.attrs.index[store.attrs["acidic_coextractant_modifier"]]))
    refit = SY.refit_lookup_yardsticks(stem, fr, v6_mask=v6, guard=SY.registered_guard(fr), table=table, systems=systems,
                                       components=comps, exclude_from_scoring=coext, halves=("S",))
    bp = pd.read_parquet(paths.FOLDS_DIR / f"{stem}__pairs.parquet")
    inp = SY.yardstick_pair_inputs(refit, store.attrs, v6_mask=store.v6, builder_pairs=bp[bp["half"].astype(str) == "S"]
                                   if "half" in bp.columns else None)
    sel = SY.select_half(inp, "selection")
    labels = [ET.fold_qualified_label(f, r) for f, r in zip(pred["fold_id"], pred["row_id"])]
    cand = pd.Series(pred["mean_logD"].to_numpy(dtype=float), index=pd.Index(labels))
    rec_hash = {D.read_record(p.with_suffix(".json")).get("design_hash")
                for p in (D.discovery_root(store.out_root) / "M2" / "V5PAIR__primary_batched" / f"s{D.PRIMARY_SEED}")
                .glob("*.parquet")}
    if len(rec_hash) != 1:
        raise AssertionError(f"M2 V5-PAIR records name {len(rec_hash)} design hashes")
    cand_design = ET.s1c_fold_design_id(stem, next(iter(rec_hash)))
    cand_ls = EP.derived_logsf(sel["pairs"], cand)
    half = ET.s1c_half(sel["pairs"], cand_ls, half="selection", seed=D.PRIMARY_SEED, lookup_logsf=sel["lookup_logsf"],
                       folds=sel["folds"], v6_mask=sel["v6_mask"],
                       fold_designs={**sel["fold_designs"], "candidate": cand_design})
    verdict = ET.s1c_paired_verdict(confirmation=None, selection=half)
    regime = EM.Regime(design="V5-PAIR", arm="M2", variant="primary", half="selection", seed=D.PRIMARY_SEED,
                       seed_set="discovery")
    summ = EP.pair_summary(sel["pairs"], cand_ls, regime, v6_mask=sel["v6_mask"], folds=sel["folds"],
                           strata_col="category_class")
    summ["label"] = "discovery, optimistically biased (selection half)"

    def slim(d):
        return {k: v for k, v in d.items() if k not in ("bootstrap", "per_cell_pair")}
    bh_rows = []
    for kind, block in (("direction M2 - {}", half["direction"]), ("logSF MAE {} - M2", half["logsf_mae"])):
        for y, comp in block.items():
            br = comp.get("bootstrap")
            if br is None:
                continue
            rec = br.record({"family": "S1(c)", "contrast": kind.format(y), "design": "V5-PAIR", "candidate": "M2",
                             "comparator": y, "half": "selection", "seed_set": "discovery",
                             "decision_seed": D.PRIMARY_SEED, "label": "discovery, optimistically biased (selection half)",
                             "batching_label": "", "reported_verdict": "UNDECIDED" if store.state.s1_forced_undecided
                             else verdict["verdict"]})
            rec["primary_cluster_unit"] = True
            rec["key"] = f"S1(c): {kind.format(y)}"
            bh_rows.append(rec)
    s1c = {"label": "S1(c) selection-half counterweight (discovery seed 104729); the confirmation half decides S1(c)",
           "verdict_with_confirmation_missing": verdict["verdict"], "checks": verdict["checks"],
           "reported_verdict": "UNDECIDED" if store.state.s1_forced_undecided else verdict["verdict"],
           "min_delta": half["min_delta"], "min_delta_yardstick": half["min_delta_yardstick"],
           "direction": {k: slim(v) for k, v in half["direction"].items()},
           "logsf_mae": {k: slim(v) for k, v in half["logsf_mae"].items()}, "fold_design": half["fold_design"],
           "n_pairs": half["n_pairs"]}
    return summ, json.loads(json.dumps(s1c, default=float)), bh_rows


# ============================================================================================= #
# main
# ============================================================================================= #

def score(out_root: Path, attrs: pd.DataFrame, *, crossings: pd.DataFrame | None = None, delta5: float | None = None,
          preseal_dir: Path = PRESEAL_PRED, with_pairs: bool = True, code: str | None = None,
          folds_dir: Path | None = None, runners: Mapping[str, Any] | None = None,
          prereg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Every scorer output as in-memory objects (written by :func:`main`).  ``code`` / ``folds_dir`` / ``runners``
    default to the runner's current code digest, the registered folds and the runner's arms (tests pass synthetic
    ones); records are verified against them."""
    v6 = attrs["v6_target_row"].astype(bool)
    state = D.PlanState.read(D.discovery_root(out_root) / "decisions" / "plan_state.json")
    crossings = crossings if crossings is not None else pd.read_csv(CROSSINGS_CSV, dtype={"row_id": str, "partner_id": str})
    store = Store(out_root, attrs, v6, crossings, state, preseal_dir, code=code, folds_dir=folds_dir, runners=runners)
    d5 = D.registered_delta5() if delta5 is None else float(delta5)
    results: dict[str, dict] = {}
    for spec in D.REGISTERED_CONTRASTS + D.EXPLORATORY_CONTRASTS:
        r = evaluate_spec(store, spec, d5)
        if r is not None:
            results[f"{spec.name}@{spec.design}"] = r
        if spec.design == "V2":                                    # every selection-half state, exploratory
            r = evaluate_spec(store, replace(spec, family="exploratory", note="V2 all selection-half states"), d5,
                              v2_summary="all_states")
            if r is not None:
                results[f"{spec.name}@V2#all_states"] = r
    lad = ladder(store, d5, results)
    stop = D.stop_rule(results.get("M2 vs B3i@V5"), results.get("B6 vs B3i@V5"))
    rows, items = [], []
    for key, r in results.items():
        for rec in D.contrast_rows(r):
            rec["key"] = key
            rows.append(rec)
        items.append(D.r19_item_rows(r).assign(key=key))
    pair_summ, s1c, s1c_rows = pair_outputs(store) if with_pairs else (None, None, [])
    rows += s1c_rows
    evaluated = [k for k, r in results.items() if r["family"] in D.REGISTERED_FAMILIES]
    accounting = D.registered_family_accounting(evaluated + (["S1(c)"] if s1c_rows else []))
    contrasts = D.apply_bh(pd.DataFrame(rows), m_registered_full=accounting["m_full"]) if rows else pd.DataFrame()
    cands = D.freezing_candidates(results, state)
    learned_keys = sorted(k for k, r in results.items() if r.get("learned_arm"))
    addendum = {
        "addendum": D.N_ADDENDA_EXPECTED,
        "label": "POST-HOC addendum 1 (below the sealed footer, 2026-09-15): the compute-driven reduction",
        "seeds_run_in_discovery": list(D.PLAN_SEEDS),
        "r19_item_4": {"status_in_discovery": D.ITEM4_NOT_EVALUATED, "detail": D.ITEM4_NOT_EVALUATED_DETAIL,
                       "contrasts": learned_keys},
        "sensitivity_set": {"label": D.ADDENDUM_LABEL,
                            "refits_run_for_a_learned_arm": list(D.LEARNED_REFIT_VARIANTS),
                            "not_run_by_design": {d: sorted(v) for d, v in D.LEARNED_REFITS_NOT_RUN.items()},
                            "reasons": D.LEARNED_REFITS_NOT_RUN,
                            "closed_form_arms": "keep the full registered sensitivity set"},
        "v5pair": D.READINGS["addendum1_v5pair"], "inner_design": D.READINGS["addendum1_inner_design"],
        "comparator_intervals": D.READINGS["comparator_intervals"],
        "unchanged": ("designs, folds, halves, hiding and guards; the V6 carve-out; metrics and averaging units; "
                      "comparators; margins; S1, S2 and F1-F6; R19 items 1-3 and 5; the stop rule; the grids of "
                      "sections 5 and 6; confirmation on the withheld seeds")}
    decisions = {
        "schema": D.SCHEMA, "label": "discovery, selection half, optimistically biased; decisions on seed 104729",
        "addendum_1": addendum, "delta5": d5, "stop_rule": stop, "ladder": lad,
        "S1_components": {**D.s1ab_components(results, state), "S1c_selection": s1c},
        "heavy_v5_batching_label": state.heavy_v5_label, "freezing_candidates": cands,
        "power_check": D.power_check_plan(results), "plan_state": state.record(), "readings": D.READINGS,
        "not_run": {"M3+": D.NOT_IMPLEMENTED, "V0": D.READINGS["selection_half_only"],
                    "confirmation": "the confirmation half and V6 are never read by discovery",
                    "registered_family": [c for c in accounting["contrasts"] if c["status"] != "evaluated"],
                    "uncertainty_metrics": D.UNCERTAINTY_NOT_RUN,
                    "addendum_1": {"discovery_seeds": [s for s in D.DISCOVERY_SEEDS if s not in D.PLAN_SEEDS],
                                   "r19_item_4": D.ITEM4_NOT_EVALUATED_DETAIL,
                                   "sensitivities": {d: sorted(v) for d, v in D.LEARNED_REFITS_NOT_RUN.items()},
                                   "v5p_heavy_arm_runs": "addendum 1 item 4: not run",
                                   "comparator_interval_jobs": "addendum 1 item 3: not run (seed 104729 pre-seal "
                                                               "intervals are used)"}},
        "registered_family_accounting": accounting,
        "record_sets": store.record_sets, "prereg_gate": None if prereg is None else dict(prereg),
        "available_contrasts": sorted(results), "n_registered_contrasts": sum(r["family"] in D.REGISTERED_FAMILIES
                                                                              for r in results.values())}
    return {"summary": summary_tables(store), "comparator_intervals": comparator_intervals(store), "contrasts": contrasts,
            "coverage_by_domain_status": coverage_by_domain_status(store),
            "r19_items": pd.concat(items, ignore_index=True) if items else pd.DataFrame(), "decisions": decisions,
            "pairs": pair_summ, "state": state, "freezing_candidates": cands}


def contrasts_markdown(con: pd.DataFrame, decisions: Mapping[str, Any]) -> str:
    lines = ["# Discovery contrasts (selection half; optimistically biased)", "",
             "Generated by `scripts/g19_score_discovery.py`. Delta = metric(comparator) - metric(candidate) of the unit "
             "macro MAE (positive favours the candidate), seed 104729, primary cluster unit; verdict columns are the R19 "
             "scopes of `gen19ct.evaluation.discovery.SCOPES`. The confirmation half and V6 were not read.", "",
             f"Stop rule (section 7 item 4): stop = {decisions['stop_rule']['stop']}.", "",
             "POST-HOC addendum 1: discovery runs seed 104729 only, so **R19 item 4 is NOT_EVALUATED** for every "
             "learned-arm contrast (column `r19 item 4`) and the full R19 verdict of such a contrast is at best "
             "UNDECIDED; the stop-rule, ladder and freezing scopes contain neither item 4 nor the refit sensitivities, "
             "so no decision changes. Item 6 of a learned-arm contrast is the *reduced sensitivity set (addendum 1)* "
             "(column `sensitivity set`): the V5 strict and HNO3-only refits plus every registered scoring-filter "
             "sensitivity; the sensitivities that were not run are named in `sensitivities not run` and in "
             "`decisions.json` -> `addendum_1`. Closed-form arms keep the full registered set.", "",
             "| family | contrast | design | Delta | margin | pct 95% | p | p_BH | p_BH (full family) | stop-rule scope | "
             "ladder scope | full R19 | r19 item 4 | sensitivity set | sensitivities not run | reported | batching | TOST |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def cell(r, k, fmt="{}"):
        v = r.get(k)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return ""
        return fmt.format(v)
    if not con.empty:
        prim = con[con["primary_cluster_unit"].astype(bool)]
        for _, r in prim.sort_values(["bh_family", "family", "contrast"]).iterrows():
            lines.append(f"| {r['family']} | {r['contrast']} | {r['design']} | {r['point']:.3f} | {cell(r, 'margin', '{:.3f}')} | "
                         f"[{r['percentile_low']:.3f}, {r['percentile_high']:.3f}] | {r['p_two_sided']:.4f} | "
                         f"{cell(r, 'p_bh', '{:.4f}')} | {cell(r, 'p_bh_full_family', '{:.4f}')} | "
                         f"{cell(r, 'verdict_stop_rule')} | {cell(r, 'verdict_ladder')} | {cell(r, 'r19_verdict_full')} | "
                         f"{cell(r, 'r19_item4')} | {cell(r, 'sensitivity_set')} | {cell(r, 'sensitivities_not_run')} | "
                         f"{cell(r, 'reported_verdict')} | {cell(r, 'batching_label')} | {cell(r, 'tost_verdict_eps0.05')} |")
    acc = decisions.get("registered_family_accounting") or {}
    if acc:
        lines += ["", f"Registered family (section 19, addendum 1 reading 6(f)): m = {acc['m_full']} "
                      f"({acc.get('m_discovery')} discovery contrasts + {len(acc.get('confirmation_only') or [])} "
                      f"confirmation-only S2 contrasts entered as p = 1), evaluated {acc['m_evaluated']}; "
                      "p_BH is over the contrasts evaluated, p_BH (full family) counts every registered contrast, a "
                      "contrast not run entering as p = 1. Adjusted p is shown, never used."]
    if decisions.get("S1_components", {}).get("S1_forced_undecided"):
        lines += ["", "S1 is reported UNDECIDED: the re-coloured batched-vs-exact check failed (section 7 item 6); every "
                      "heavy-arm V5 result is labelled 'batched (check failed)'."]
    return "\n".join(lines) + "\n"


def summary_markdown(summ: pd.DataFrame) -> str:
    lines = ["# Discovery metrics (selection half; optimistically biased)", "",
             "Macro MAE (log D) under the registered averaging unit, scoring filter none, seed per column. Generated by "
             "`scripts/g19_score_discovery.py` from `tables/discovery_summary.csv`.", ""]
    if summ.empty:
        return "\n".join(lines + ["No discovery predictions yet."]) + "\n"
    s = summ[(summ["metric"] == "mae") & (summ["aggregation"] == "unit_macro") & (summ["stratum"] == "all")
             & (summ["scoring_filter"] == "none") & (summ["status"] == "registered")]
    lines += ["| arm | design | variant | " + " | ".join(f"s{x}" for x in D.PLAN_SEEDS) + " |",
              "|---|---|---|" + "---|" * len(D.PLAN_SEEDS)]
    for (arm, design, variant), g in s.groupby(["arm", "design", "variant"], sort=True):
        vals = [g.loc[g["seed"] == sd, "value"] for sd in D.PLAN_SEEDS]
        lines.append(f"| {arm} | {design} | {variant} | " + " | ".join(f"{v.iloc[0]:.3f}" if len(v) else "" for v in vals)
                     + " |")
    return "\n".join(lines) + "\n"


def main(argv=None, *, check=None, digests=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--no-pairs", action="store_true", help="skip the V5-PAIR / S1(c) block")
    ap.add_argument("--expect-addenda", type=int, default=None,
                    help=f"POST-HOC addenda expected below the sealed footer (default {D.N_ADDENDA_EXPECTED}, the "
                         "addendum this code implements); scoring is refused on any other count")
    ap.add_argument("--no-manifest", action="store_true")
    ns = ap.parse_args(argv)
    out_root = Path(ns.out_root)
    prereg = _runner_module().refuse_unless_sealed(check, digests, expect_addenda=ns.expect_addenda)
    log("row attributes")
    attrs = build_attrs()
    with (Run(NAME, args=vars(ns), extra={"readings": D.READINGS, "prereg_gate": prereg})
          if not ns.no_manifest else _Null()) as run:
        res = score(out_root, attrs, with_pairs=not ns.no_pairs, prereg=prereg)
        ev = D.discovery_root(out_root)
        tables = out_root / "tables"
        outs = []
        con = res["contrasts"]
        if not con.empty:
            outs.append(write_csv(con[con["bh_family"] == "registered"], ev / "contrasts_registered.csv"))
            outs.append(write_csv(con[con["bh_family"] == "exploratory"], ev / "contrasts_exploratory.csv"))
            outs.append(write_csv(res["r19_items"], ev / "r19_items.csv"))
        outs.append(write_csv(res["summary"], tables / "discovery_summary.csv"))
        outs.append(write_text(tables / "discovery_summary.md", summary_markdown(res["summary"])))
        outs.append(write_text(tables / "discovery_contrasts.md", contrasts_markdown(con, res["decisions"])))
        if not res["coverage_by_domain_status"].empty:
            outs.append(write_csv(res["coverage_by_domain_status"], tables / "discovery_coverage_by_domain_status.csv"))
        if not res["comparator_intervals"].empty:
            outs.append(write_csv(res["comparator_intervals"], tables / "discovery_comparator_intervals.csv"))
        if res["pairs"] is not None:
            outs.append(write_csv(res["pairs"], tables / "discovery_pair_summary.csv"))
        outs.append(write_json(ev / "decisions" / "decisions.json", _json(res["decisions"])))
        outs.append(write_json(ev / "decisions" / "stop_rule.json", _json(res["decisions"]["stop_rule"])))
        state = res["state"]
        state.freezing_candidates = res["freezing_candidates"]
        outs.append(write_json(ev / "decisions" / "plan_state.json", state.record()))
        if run is not None:
            run.outputs(*outs)
            run.extra.update({"confirmation_half_read": False, "v6_target_rows_scored": 0,
                              "stop": res["decisions"]["stop_rule"]["stop"], "code_sha256": _runner_module()
                              .current_code_digest()})
    log(f"stop rule: {res['decisions']['stop_rule']['stop']}; contrasts: {len(res['decisions']['available_contrasts'])}")
    return 0


def _json(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())

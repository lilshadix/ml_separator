"""gen20.1 decision rules (PRE_REGISTRATION.md §1, §7, §8, §11; IMPLEMENTATION_PLAN.md §6): RULE-T, RULE-C with
the intersection-union verdict order, RULE-H4, NC1, Benjamini-Hochberg over the m = 83 exploratory family, the
pre-written sentences, and the confirmation logic.

Every function here is pure: it reads evidence dicts produced by ``gen201.inference`` (or hand-built ones in the
tests) and classifies unrounded floats with no override.  A value equal to τ satisfies "≥"; a value equal to δ_Y
satisfies (a); NA (NaN) fails every condition that uses it; a design not run leaves (h) pending, which gives
INSUFFICIENT_COMPUTE when (a)-(g) hold for every comparator on design BP and otherwise counts as failed.
"""
from __future__ import annotations

import math

import numpy as np

from . import bootstrap  # noqa: F401

TAUS: tuple[float, ...] = (0.4, 0.5)
ALPHA_K = 0.0125
DELTA_FLOOR = 0.02
DELTA_REL = 0.10
R2_BAL_GAIN = 0.05
EQUIV_MARGIN = 0.02
BH_Q = 0.05
M_EXPLORATORY = 83
OTHER_DESIGNS: tuple[str, ...] = ("B", "BR", "BQ", "BPG", "A")
H1_COMPARATORS: tuple[str, ...] = ("BD_MEAN", "BD_METAL", "BD_COND", "BD_NN")
SF_COMPARATORS: tuple[str, ...] = ("FLAT", "PAIRMEAN", "G14_ZERO")
CLAIMS: dict[str, tuple[str, str, tuple[str, ...]]] = {     # id -> (candidate, target, comparators)
    "H1": ("A", "logd", H1_COMPARATORS), "H2": ("B", "logsf", SF_COMPARATORS), "H3": ("C", "logsf", SF_COMPARATORS)}
CLASS_ORDER: dict[str, int] = {"REACHED": 3, "POINT_ONLY": 2, "NOT_REACHED": 1, "RULED_OUT": 0}
VERDICT_ORDER: tuple[str, ...] = ("NA", "BASELINE_BETTER", "INSUFFICIENT_COMPUTE", "PASS", "ESTIMAND_INCONSISTENT",
                                  "INCONCLUSIVE")
H4_ORDER: tuple[str, ...] = ("NA", "INSUFFICIENT_COMPUTE", "C_BETTER", "B_BETTER", "EQUIVALENT", "NOT_ESTABLISHED")
CONDITIONS_BP: tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g")
CONDITIONS_ALL: tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k")

FEATURE_LABEL = "feature set fixed from gen13 outer results on these folds"
CONFIRMATION_LABEL = "partition-robustness check on the same corpus, not independent validation"
NOT_CONFIRMED_LABEL = "discovery only, not confirmed"
CONFIRMATION_PENDING = "confirmation registered (S11), not yet run"
NC1_FLAG = "inference flagged by negative control"
LOW_POWER = "passed at low power; effect likely overestimated"
NODMD_FLIP = "not robust to known corrupt level data"
REACHED_ALSO = "reached also by {baseline}: the threshold alone carries no evidence of model skill"
NOT_EVALUABLE = "not evaluable (NA)"
LEARNER_SEALED = ("gen13 tree_pipeline: SimpleImputer(median, keep_empty_features=True) then ExtraTreesRegressor(400 "
                  "trees, random_state = fold.model_seed) on LEAN209 + METAL_BASIC2, configurations K1-K4 "
                  "(min_samples_leaf 8, 8, 2, 2; max_features 0.3, 0.5, 0.3, 0.5) selected per outer fold on inner data")


def _fin(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _ge(x, t) -> bool:
    return _fin(x) and float(x) >= t


def _gt(x, t) -> bool:
    return _fin(x) and float(x) > t


def _lt(x, t) -> bool:
    return _fin(x) and float(x) < t


def _maxmde(ev: dict) -> float:
    vals = [float(ev.get(k, float("nan"))) for k in ("mde80_dec", "mde80_cr")]
    vals = [v for v in vals if _fin(v)]
    return max(vals) if vals else float("nan")


def delta_margin(met01_Y) -> float:
    """δ_Y = max(0.02, 0.10·MET-01(Y)), Y's seed-mean outer value (O6)."""
    return max(DELTA_FLOOR, DELTA_REL * float(met01_Y)) if _fin(met01_Y) else float("nan")


def seeds_required(n_seeds: int) -> int:
    """'≥ 4 of 5 seeds' (RULE-C (f), RULE-T (v)); for other seed counts all but one."""
    return 4 if n_seeds == 5 else max(1, n_seeds - 1)


# --------------------------------------------------------------------------------------
# RULE-T
# --------------------------------------------------------------------------------------
def rule_t(ev: dict, tau: float) -> dict:
    """REACHED / POINT_ONLY / NOT_REACHED / RULED_OUT / NA for one arm and τ (PRE_REGISTRATION §7, O7 default).

    ``ev``: point, per_seed, chem_pct95_low/high (2.5th/97.5th percentiles of the seed-mean draws),
    chem_bca95_low, pub_pct95_low/high, r2_bal (seed-mean R2_balanced; the within-cell variant for T1w)."""
    pt = ev.get("point")
    out = {"tau": tau, "point": pt, "chem_pct97_5": ev.get("chem_pct95_high"), "pub_pct97_5": ev.get("pub_pct95_high")}
    if not _fin(pt):
        out.update({"class": "NA", "conditions": {}, "ruled_out": False, "also_ruled_out": False})
        return out
    per = [v for v in ev.get("per_seed", [])]
    n_ok = sum(1 for v in per if _ge(v, tau))
    c = {"i": float(pt) >= tau,
         "ii": _ge(ev.get("chem_pct95_low"), tau),
         "iii": _ge(ev.get("chem_bca95_low"), tau),
         "iv": _ge(ev.get("pub_pct95_low"), tau),
         "v": n_ok >= seeds_required(len(per)),
         "vi": _ge(ev.get("r2_bal"), tau)}
    ruled_out = _lt(ev.get("chem_pct95_high"), tau) and _lt(ev.get("pub_pct95_high"), tau)
    if all(c.values()):
        cls = "REACHED"
    elif c["i"]:
        cls = "POINT_ONLY"
    elif ruled_out:
        cls = "RULED_OUT"
    else:
        cls = "NOT_REACHED"
    out.update({"class": cls, "conditions": c, "seeds_at_or_above": n_ok, "ruled_out": ruled_out,
                "also_ruled_out": bool(cls == "POINT_ONLY" and ruled_out)})
    return out


def class_rank(cls: str) -> int | None:
    return CLASS_ORDER.get(cls)


def level_attributable(cls_A: str, cls_level: str, r2w_A, r2w_level_curve) -> dict:
    """T1 'level-attributable' annotation: BD_LEVEL's class ≥ A's (class order REACHED > POINT_ONLY > NOT_REACHED >
    RULED_OUT; 'not evaluable (NA)' if either class is NA), or R2_within_cell(A) ≤ 0, or R2_within_cell(A) ≤
    R2_within_cell(BD_LEVEL_CURVE)."""
    if cls_A == "NA" or cls_level == "NA" or cls_A not in CLASS_ORDER or cls_level not in CLASS_ORDER:
        clause = NOT_EVALUABLE
        by_class = False
    else:
        by_class = CLASS_ORDER[cls_level] >= CLASS_ORDER[cls_A]
        clause = bool(by_class)
    nonpos = _fin(r2w_A) and float(r2w_A) <= 0
    le_curve = _fin(r2w_A) and _fin(r2w_level_curve) and float(r2w_A) <= float(r2w_level_curve)
    attributable = bool(by_class or nonpos or le_curve)
    return {"level_attributable": attributable, "class_clause": clause, "within_nonpositive": bool(nonpos),
            "within_le_level_curve": bool(le_curve),
            "text": "level-attributable" if attributable else "not level-attributable"}


# --------------------------------------------------------------------------------------
# RULE-C, verdict order
# --------------------------------------------------------------------------------------
def rule_c(ev: dict) -> dict:
    """Conditions (a)-(k) of one component X vs Y (Δ = MET-01(Y) − MET-01(X), positive favours X).

    (h) reads ``ev['design_points']`` {design: point Δ or None when the design was not run}; a missing design
    leaves (h) pending (``h_missing``).  (j)/(k) apply to log SF only (None = not applicable)."""
    target = ev.get("target", "logsf")
    dY = delta_margin(ev.get("met01_Y"))
    per = ev.get("delta_seeds", [])
    n_pos = sum(1 for v in per if _gt(v, 0))
    c: dict = {}
    c["a"] = _ge(ev.get("delta"), dY)
    c["b"] = _gt(ev.get("chem_pct9875_low"), 0)
    c["c"] = _gt(ev.get("chem_bca9875_low"), 0)
    c["d"] = _lt(ev.get("p_wcr"), ALPHA_K) and _gt(ev.get("wcr_mean"), 0)
    c["e"] = _gt(ev.get("pub_pct9875_low"), 0)
    c["f"] = n_pos >= seeds_required(len(per)) if per else False
    c["g"] = bool(ev.get("loco_all_positive", False))
    pts = ev.get("design_points") or {}
    missing = [d for d in OTHER_DESIGNS if pts.get(d, None) is None]
    c["h"] = None if missing else all(_gt(pts[d], 0) for d in OTHER_DESIGNS)
    c["i"] = _gt(ev.get("delta_mae_bal"), 0)
    if target == "logsf":
        c["j"] = (_ge(ev.get("delta_r2_bal"), R2_BAL_GAIN) and _gt(ev.get("delta_r2_bal_chem_low"), 0)
                  and _gt(ev.get("delta_r2_bal_pub_low"), 0) and _gt(ev.get("r2_bal_X"), 0))
        c["k"] = _gt(ev.get("skill_flat_X"), 0)
    else:
        c["j"] = c["k"] = None
    baseline_better = (_lt(ev.get("chem_pct9875_high"), 0) and _lt(ev.get("p_wcr"), ALPHA_K)
                       and _lt(ev.get("wcr_mean"), 0))
    return {"X": ev.get("X"), "Y": ev.get("Y"), "target": target, "delta": ev.get("delta"), "delta_Y": dY,
            "conditions": c, "bp_ag": all(c[k] for k in CONDITIONS_BP), "h_missing": missing,
            "baseline_better": bool(baseline_better), "mde_max": _maxmde(ev), "seeds_positive": n_pos}


def _resolved(comp: dict, h_value: bool) -> dict:
    c = dict(comp["conditions"])
    if c["h"] is None:
        c["h"] = h_value
    return c


def verdict_label(base: str, sub=()) -> str:
    return f"{base} ({'; '.join(sub)})" if sub else base


def verdict_h(components: list[dict], *, candidate_run: bool = True) -> dict:
    """H1-H3 verdict in the sealed order NA → BASELINE_BETTER → INSUFFICIENT_COMPUTE → PASS →
    ESTIMAND_INCONSISTENT → INCONCLUSIVE (ENCOURAGING; UNDERPOWERED).  ``components`` are ``rule_c`` results."""
    if not candidate_run or not components:
        return {"verdict": "NA", "sublabels": [], "label": "NA", "components": components,
                "reason": "candidate not run" if not candidate_run else "no comparator evaluated"}
    if any(c["baseline_better"] for c in components):
        bb = [c["Y"] for c in components if c["baseline_better"]]
        return {"verdict": "BASELINE_BETTER", "sublabels": [], "label": "BASELINE_BETTER", "components": components,
                "baseline_better_against": bb}
    ag = all(c["bp_ag"] for c in components)
    missing = sorted({d for c in components for d in c["h_missing"]})
    if missing and ag:
        return {"verdict": "INSUFFICIENT_COMPUTE", "sublabels": [], "label": "INSUFFICIENT_COMPUTE",
                "components": components, "designs_not_run": missing}
    resolved = [_resolved(c, False) for c in components]          # a missing design counts as failed (h)
    ah = all(all(r[k] for k in CONDITIONS_BP + ("h",)) for r in resolved)
    ik = all(all(r[k] for k in ("i", "j", "k") if r[k] is not None) for r in resolved)
    out = {"components": components, "designs_not_run": missing, "resolved_conditions": resolved}
    if ah and ik:
        low = [c["Y"] for c in components if _fin(c["mde_max"]) and c["mde_max"] > c["delta_Y"]]
        out.update({"verdict": "PASS", "sublabels": [], "label": "PASS", "low_power_components": low})
        return out
    if ah:
        out.update({"verdict": "ESTIMAND_INCONSISTENT", "sublabels": [], "label": "ESTIMAND_INCONSISTENT"})
        return out
    sub = []
    if all(c["conditions"]["a"] for c in components):
        sub.append("ENCOURAGING")
    failing = [c for c, r in zip(components, resolved) if not all(v for v in r.values() if v is not None)]
    if any(_fin(c["mde_max"]) and c["mde_max"] > c["delta_Y"] for c in failing):
        sub.append("UNDERPOWERED")
    out.update({"verdict": "INCONCLUSIVE", "sublabels": sub, "label": verdict_label("INCONCLUSIVE", sub)})
    return out


def rule_h4(ev_cb: dict | None, ev_bc: dict | None, *, c_run: bool = True) -> dict:
    """RULE-H4: NA → INSUFFICIENT_COMPUTE → C_BETTER → B_BETTER → EQUIVALENT → NOT_ESTABLISHED (UNDERPOWERED).
    ``ev_cb``: X = C, Y = B (Δ = MET-01(B) − MET-01(C)); ``ev_bc``: X = B, Y = C (−Δ), each with its own (h)."""
    if not c_run or ev_cb is None or ev_bc is None:
        return {"verdict": "NA", "sublabels": [], "label": "NA", "reason": "C not run"}
    r1, r2 = rule_c(ev_cb), rule_c(ev_bc)
    missing = sorted(set(r1["h_missing"]) | set(r2["h_missing"]))
    out = {"components": [r1, r2], "designs_not_run": missing, "delta": ev_cb.get("delta")}
    if missing and (r1["bp_ag"] or r2["bp_ag"]):
        out.update({"verdict": "INSUFFICIENT_COMPUTE", "sublabels": [], "label": "INSUFFICIENT_COMPUTE"})
        return out
    a1, a2 = _resolved(r1, False), _resolved(r2, False)
    if all(v for v in a1.values() if v is not None) and all(a1[k] is not None for k in CONDITIONS_ALL):
        out.update({"verdict": "C_BETTER", "sublabels": [], "label": "C_BETTER"})
        return out
    if all(v for v in a2.values() if v is not None) and all(a2[k] is not None for k in CONDITIONS_ALL):
        out.update({"verdict": "B_BETTER", "sublabels": [], "label": "B_BETTER"})
        return out
    equiv = (_gt(ev_cb.get("chem_pct9875_low"), -EQUIV_MARGIN) and _lt(ev_cb.get("chem_pct9875_high"), EQUIV_MARGIN)
             and _gt(ev_cb.get("pub_pct9875_low"), -EQUIV_MARGIN) and _lt(ev_cb.get("pub_pct9875_high"), EQUIV_MARGIN))
    out["equivalence_intervals_inside"] = bool(equiv)
    if equiv:
        out.update({"verdict": "EQUIVALENT", "sublabels": [], "label": "EQUIVALENT"})
        return out
    sub = ["UNDERPOWERED"] if _gt(_maxmde(ev_cb), EQUIV_MARGIN) else []
    out.update({"verdict": "NOT_ESTABLISHED", "sublabels": sub, "label": verdict_label("NOT_ESTABLISHED", sub)})
    return out


def nc1(ev_fwd: dict, ev_bwd: dict) -> dict:
    """NC1: A vs A_PRIME on COH-D-C3 through RULE-C (b)-(g), without the margin (a), in both directions; the flag
    is raised when either direction meets all of (b)-(g)."""
    res = {}
    for name, ev in (("A_over_A_PRIME", ev_fwd), ("A_PRIME_over_A", ev_bwd)):
        r = rule_c(ev)
        cond = {k: r["conditions"][k] for k in ("b", "c", "d", "e", "f", "g")}
        res[name] = {"conditions": cond, "meets_b_to_g": all(cond.values()), "delta": ev.get("delta"),
                     "chem_pct9875_low": ev.get("chem_pct9875_low"), "chem_pct9875_high": ev.get("chem_pct9875_high"),
                     "p_wcr": ev.get("p_wcr")}
    res["flag"] = any(v["meets_b_to_g"] for v in res.values() if isinstance(v, dict))
    return res


# --------------------------------------------------------------------------------------
# annotations on H verdicts
# --------------------------------------------------------------------------------------
def prnd_status(upper_bounds: dict, points: dict) -> str:
    """Recovery-plan item 4 on the Pr/Nd rows: 'verified' when the 98.75 % upper bound of MAE_PrNd(X) − MAE_PrNd(Y)
    is ≤ 0.02 for every Y, 'failed' when some point exceeds 0.02, otherwise 'not verified'."""
    if upper_bounds and all(_fin(v) and float(v) <= DELTA_FLOOR for v in upper_bounds.values()):
        return "verified"
    if any(_gt(v, DELTA_FLOOR) for v in points.values()):
        return "failed"
    return "not verified"


def annotate(verdict: dict, *, hid: str, nc1_flag: bool = False, nodmd_label: str | None = None,
             prnd: str | None = None, confirmation: str | None = None) -> list[str]:
    """Fixed annotations of an H verdict (they never change it)."""
    ann = []
    base = verdict.get("verdict")
    if base == "PASS" and verdict.get("low_power_components"):
        ann.append(LOW_POWER)
    if base in ("PASS", "C_BETTER", "B_BETTER") and nc1_flag:
        ann.append(NC1_FLAG)
    if nodmd_label is not None and nodmd_label.split(" ")[0] != verdict.get("verdict"):
        ann.append(f"{NODMD_FLIP} (COH-NODMD verdict: {nodmd_label})")
    if base == "PASS" and hid in ("H2", "H3") and prnd is not None:
        ann.append(f"Pr/Nd item 4: {prnd}" + ("" if prnd == "verified" else " (no 'full success')"))
    ann.append(confirmation if confirmation else CONFIRMATION_PENDING)
    ann.append(FEATURE_LABEL)
    return ann


# --------------------------------------------------------------------------------------
# exploratory family and BH
# --------------------------------------------------------------------------------------
def _h_contrasts(pop_d: str | None, pop_sf: str | None, *, h1=True, h23=True, h4=True) -> list[tuple]:
    out = []
    if h1 and pop_d:
        out += [(pop_d, "A", y, "MET01") for y in H1_COMPARATORS]
    if h23 and pop_sf:
        out += [(pop_sf, "B", y, "MET01") for y in SF_COMPARATORS]
        out += [(pop_sf, "C", y, "MET01") for y in SF_COMPARATORS]
    if h4 and pop_sf:
        out += [(pop_sf, "C", "B", "MET01")]
    return out


def exploratory_family() -> list[dict]:
    """The sealed exploratory family (PRE_REGISTRATION §8.1): one test per (EX, population, named comparator),
    m = 83; EX-15 excluded (O4 OFF).  Each entry: ex, id, population, variant, candidate, comparator, metric."""
    fam: list[dict] = []

    def add(ex, pop, cand, comp, metric="MET01", variant="primary", kind="contrast"):
        fam.append({"ex": ex, "id": f"{ex}|{pop}|{cand}>{comp}|{metric}", "population": pop, "variant": variant,
                    "candidate": cand, "comparator": comp, "metric": metric, "kind": kind})
    add("EX-01", "COH-SF-C3", "C_selSF", "B")
    add("EX-02", "COH-SF-C3", "B_selSF", "B")
    add("EX-03", "COH-SF-C3", "C_selSF", "B_selSF")
    add("EX-04", "COH-SF-C3", "C_ACFG", "B")
    add("EX-05", "COH-SF-C3", "C_PAIR", "B")
    for y in SF_COMPARATORS:
        add("EX-06", "COH-SF-C3", "C_PAIR", y)
    add("EX-07", "COH-D-C3", "A", "BD_LEVEL_CURVE", "MET20")
    add("EX-08", "COH-D-C3", "A", "BD_LEVEL", "MET20")
    for pop in ("COH-D-C3M", "COH-D-ALL"):
        for (p, c, y, m) in _h_contrasts(pop, None):
            add("EX-09", p, c, y, m)
    for pop in ("COH-SF-ANY", "COH-SF-C1", "COH-SF-NDSM"):
        for (p, c, y, m) in _h_contrasts(None, pop):
            add("EX-09", p, c, y, m)
    for (p, c, y, m) in _h_contrasts("COH-NOSC009-D", "COH-NOSC009-SF"):
        add("EX-09", p, c, y, m)
    for (p, c, y, m) in _h_contrasts("COH-D-C3@NODMD", "COH-SF-C3@NODMD"):
        add("EX-10", p, c, y, m, variant="nodmd")
    for (p, c, y, m) in _h_contrasts("COH-REPCAP-D", "COH-REPCAP-SF"):
        add("EX-10", p, c, y, m)
    add("EX-11", "COH-D-C3", "A", "BL_D_INNER")
    add("EX-11", "COH-SF-C3", "B", "BL_SF_INNER")
    add("EX-11", "COH-SF-C3", "C", "BL_SF_INNER")
    add("EX-12", "COH-D-C3+COH-SF-C3", "A>BD_COND", "B>BD_COND_SF", "spearman", kind="spearman")
    add("EX-12", "COH-D-C3+COH-SF-C3", "A>BD_NN", "B>BD_NN_SF", "spearman", kind="spearman")
    add("EX-13", "COH-D-C3", "A", "A_PRIME")
    add("EX-13", "COH-D-C3", "A_PRIME", "A")
    for x in ("BD_METAL_SF", "BD_COND_SF", "BD_NN_SF", "BD_LEVEL_CURVE_SF"):
        add("EX-14", "COH-SF-C3", x, "FLAT")
    if len(fam) != M_EXPLORATORY:  # pragma: no cover - sealed size
        raise AssertionError(f"exploratory family has {len(fam)} tests, sealed m = {M_EXPLORATORY}")
    return fam


def bh(p_values, *, m: int = M_EXPLORATORY, q: float = BH_Q) -> dict:
    """Benjamini-Hochberg over the whole family (gen20k.inference.benjamini_hochberg); an NA p (arm not run,
    test not computable) enters as p = 1 with m unchanged.  Returns {'p_used', 'q', 'reject'} arrays."""
    from gen20k.inference import benjamini_hochberg
    p = np.array([float(v) if _fin(v) else 1.0 for v in p_values], dtype=float)
    if len(p) != m:
        raise AssertionError(f"BH family size {len(p)} != sealed m = {m}")
    qv = benjamini_hochberg(p)
    return {"p_used": p, "q": qv, "reject": qv <= q}


# --------------------------------------------------------------------------------------
# pre-written sentences (PRE_REGISTRATION §1 and §7), verbatim
# --------------------------------------------------------------------------------------
EASIER_D = ("Absolute log D was shown to be predictable beyond its pre-named baselines and strict-adjacent log SF "
            "was not")
EASIER_SF = ("Strict-adjacent log SF was shown to be predictable beyond its pre-named baselines and absolute log D "
             "was not")
EASIER_NEITHER = "neither target was shown to be more predictable than the other beyond its baselines"
BASELINE_SETS_DIFFER = ("the baseline sets differ: the log SF set contains the prior model G14_ZERO, the log D set "
                        "no prior model")
MORE_RELIABLE = ('"More reliable" is never a verdict; the MET-23 panels of both targets are printed side by side with '
                 'no ranking.')

PAT = {
    "PAT-1": ("Variant A was shown to beat every pre-named log D baseline for unseen chemotypes and publications (H1 "
              "PASS). B and C were not shown to beat every log SF baseline (H2: <verdict>, H3: <verdict>); this is not "
              "evidence that strict-adjacent selectivity is unpredictable."),
    "PAT-2": "The log-D-then-subtract route predicts strict-adjacent selectivity beyond its baselines.",
    "PAT-3": ("Derived log SF (B) beats every pre-named log SF baseline although A was not shown to beat every log D "
              "baseline (H1: <verdict>); A's cell-level error cancels in the difference (I5)."),
    "PAT-4": ("Direct contrast training (C) predicts strict-adjacent log SF beyond its baselines; the derived route "
              "(B) was not shown to (H2: <verdict>)."),
    "PAT-5": ("None of A, B or C (<learner, features and grid as sealed in §4.1>) was shown to beat its pre-named "
              "training-only baselines under design BP (H1: <verdict>, H2: <verdict>, H3: <verdict>). This does not "
              "show that either target is unpredictable."),
    "PAT-6": ("Absolute log D reaches pooled R² ≥ τ with A (T1 REACHED; T1w: <class>; <level-attributable / not "
              "level-attributable>). Strict-adjacent log SF was not shown to reach τ with B or C (T2: <class>, T3: "
              "<class>), so the log D result is not evidence that selectivity is predicted."),
}
H4_SENTENCES = {
    "C_BETTER": "direct contrast training beats log D followed by subtraction on strict-adjacent log SF",
    "B_BETTER": "log D followed by subtraction beats direct contrast training on strict-adjacent log SF",
    "EQUIVALENT": "the two routes are equivalent within ±0.02 log10",
    "NOT_ESTABLISHED": "no difference could be established",
}
PATTERN_NOT_ANTICIPATED = "pattern not anticipated"


def _fill(template: str, values: list[str], placeholder: str) -> str:
    out = template
    for v in values:
        out = out.replace(placeholder, v, 1)
    return out


def easier_sentence(labels: dict) -> dict:
    """PRE_REGISTRATION §1: the 'easier and more reliable' sentence, with every verdict in full and the
    baseline-set statement."""
    base = {k: labels.get(k, "NA").split(" ")[0] for k in ("H1", "H2", "H3", "H4")}
    h1 = base["H1"] == "PASS"
    sf = base["H2"] == "PASS" or base["H3"] == "PASS"
    if h1 and not sf:
        s, sid = EASIER_D, "EASIER-D"
    elif sf and not h1:
        s, sid = EASIER_SF, "EASIER-SF"
    else:
        s, sid = EASIER_NEITHER, "EASIER-NEITHER"
    full = (f"{s} (H1: {labels.get('H1', 'NA')}; H2: {labels.get('H2', 'NA')}; H3: {labels.get('H3', 'NA')}; "
            f"H4: {labels.get('H4', 'NA')}); {BASELINE_SETS_DIFFER}.")
    return {"id": sid, "sentence": s, "text": full}


def pat_sentences(labels: dict, *, h4_label: str | None = None) -> list[dict]:
    """The applicable PAT-1..PAT-5 sentence (exactly one of them applies to every H1-H3 PASS pattern), with
    <verdict> placeholders filled; PAT-4 carries H4 beside it."""
    p = {k: labels.get(k, "NA").split(" ")[0] == "PASS" for k in ("H1", "H2", "H3")}
    lab = {k: labels.get(k, "NA") for k in ("H1", "H2", "H3", "H4")}
    out = []
    if p["H1"] and p["H2"]:
        out.append({"id": "PAT-2", "text": PAT["PAT-2"]})
    elif p["H1"] and not p["H2"] and not p["H3"]:
        out.append({"id": "PAT-1", "text": _fill(PAT["PAT-1"], [lab["H2"], lab["H3"]], "<verdict>")})
    elif not p["H1"] and p["H2"]:
        out.append({"id": "PAT-3", "text": _fill(PAT["PAT-3"], [lab["H1"]], "<verdict>")})
    if p["H3"] and not p["H2"]:
        txt = _fill(PAT["PAT-4"], [lab["H2"]], "<verdict>") + f" (H4: {h4_label or lab['H4']})"
        out.append({"id": "PAT-4", "text": txt})
    if not any(p.values()):
        txt = PAT["PAT-5"].replace("<learner, features and grid as sealed in §4.1>", LEARNER_SEALED, 1)
        out.append({"id": "PAT-5", "text": _fill(txt, [lab["H1"], lab["H2"], lab["H3"]], "<verdict>")})
    if not out:  # pragma: no cover - the five patterns cover every H1-H3 PASS combination
        out.append({"id": "PATTERN-NOT-ANTICIPATED",
                    "text": f"{PATTERN_NOT_ANTICIPATED}: H1: {lab['H1']}, H2: {lab['H2']}, H3: {lab['H3']}"})
    return out


def threshold_sentence(tau: float, classes: dict, level_text: str) -> dict | None:
    """PAT-6 at one τ when T1 is REACHED and T2, T3 are not; another pattern with a REACHED class is printed
    verdict by verdict as 'pattern not anticipated'; no REACHED class -> None (classes are printed in TAB-06)."""
    c = {k: classes.get(k, "NA") for k in ("T1", "T1w", "T2", "T3")}
    if c["T1"] == "REACHED" and c["T2"] != "REACHED" and c["T3"] != "REACHED":
        txt = PAT["PAT-6"].replace("<class>", c["T1w"], 1)
        txt = txt.replace("<level-attributable / not level-attributable>", level_text, 1)
        txt = _fill(txt, [c["T2"], c["T3"]], "<class>")
        return {"id": "PAT-6", "tau": tau, "text": f"At τ = {tau:g}: {txt}"}
    if any(v == "REACHED" for v in c.values()):
        return {"id": "PATTERN-NOT-ANTICIPATED", "tau": tau,
                "text": (f"At τ = {tau:g}: {PATTERN_NOT_ANTICIPATED} (T1: {c['T1']}; T1w: {c['T1w']}; T2: {c['T2']}; "
                         f"T3: {c['T3']}).")}
    return None


def h4_sentence(h4: dict) -> dict:
    base = h4.get("verdict", "NA")
    s = H4_SENTENCES.get(base)
    text = f"H4 ({h4.get('label', base)}): {s}." if s else f"H4: {h4.get('label', base)}."
    return {"id": "H4", "verdict": base, "text": text}


# --------------------------------------------------------------------------------------
# confirmation (PRE_REGISTRATION §11)
# --------------------------------------------------------------------------------------
def confirm_h(hid: str, discovery: dict, confirmation: dict | None) -> dict:
    """An H is 'confirmed' only if it PASSes on both seed sets (H4: the same verdict on both)."""
    if confirmation is None:
        return {"status": NOT_CONFIRMED_LABEL, "confirmed": False}
    d, c = discovery.get("verdict"), confirmation.get("verdict")
    if hid == "H4":
        ok = d == c and d not in (None, "NA")
    else:
        ok = d == "PASS" and c == "PASS"
    status = (f"confirmed ({CONFIRMATION_LABEL})" if ok else
              f"not confirmed (discovery: {discovery.get('label', d)}; confirmation: {confirmation.get('label', c)}; "
              f"{CONFIRMATION_LABEL})")
    return {"status": status, "confirmed": bool(ok), "discovery": discovery.get("label", d),
            "confirmation": confirmation.get("label", c)}


def confirm_t(discovery_class: str, confirmation_class: str | None) -> dict:
    """A threshold class is final REACHED (RULED_OUT) only if so on both seed sets; otherwise both are shown."""
    if confirmation_class is None:
        return {"final": None, "text": f"{discovery_class} ({NOT_CONFIRMED_LABEL})"}
    if discovery_class == confirmation_class and discovery_class in ("REACHED", "RULED_OUT"):
        return {"final": discovery_class, "text": f"{discovery_class} (final; {CONFIRMATION_LABEL})"}
    return {"final": None, "text": (f"discovery: {discovery_class}; confirmation: {confirmation_class} "
                                    f"({CONFIRMATION_LABEL})")}

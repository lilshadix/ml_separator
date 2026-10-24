"""Stage-1 item 1.4 (CLAUDE_CODE_PROMPT.md 6.4; 6.0, 6.10, 8, 9): M09, the B1 level window as a tail-risk reporting
axis.  Descriptive only (the window axis: never E-A, never placed next to 0.452; excluded from the confirmation package,
power <= 0.15 [I2 confirm-1]).  Every in-corpus number is EXPLORATORY (seen); the grid was fixed after seeing V1.

Rows: the round-4 REF E-A rows (r5.common.load_rows: C3, k = 3 = four DOPT_Q reads, 1,802 queries per seed, 5 dev
seeds); BPG primary, BP a labelled sensitivity.  Every rule, control and alternative is scored on the same rows with B1's
residual res = y - B1 (B0 and HIST_D7 inside R^2 beside it on the curve table).

Levels.  lev = r4.refs.predict_levels with the fold's stored B1 amplitude (round4 results/REF/fitted_<design>.json) on
the row's DOPT_Q read set (the measured value at a read metal); s = max(|lev_A|, |lev_B|).  lev_A - lev_B reproduces
the parquet's B1 (identity check, brief 6.0).  HIST_D7-form levels (-0.072) are kept for the evidence-base
reproduction (EB Appendix 2 built the window on them).

Rules (inside = kept; every rule is label-free):
* frozen grid, "fixed after seeing V1": half-widths H1, H1.5, H2, H2.5, H3, Hinf (s <= h); lower-only LO-2, LO-2.5,
  LO-3, LO-4 (min(lev_A, lev_B) >= L); upper-only UP2, UP2.5, UP3 (max(lev_A, lev_B) <= U);
* operating points OP75 / OP90: s <= the label-free quantile (inverted CDF) of s at coverage 0.75 / 0.90 over the
  outer fold's TRAINING rows: every C3 training cell of the fold record (r4.refs.training_cells: the inner blocks'
  records plus the orphans), masked with its nested DOPT_Q[:3] read set (the masks of B1's own training situations),
  levels with the fold's beta, every strict-adjacent query.  No test row enters;
* cheapest alternatives: READMIN-2 (min read >= -2, the M10 reads-only flag), READSIN2 (all four reads in [-2, 2]),
  WHOLECELL2 (every row of a cell instance kept iff H2 keeps >= half of its rows), LO-4 (= "drop predicted
  min < -4"), NOC5 (drop C5-BTBP; a metadata rule, seen 0.486 at 91.8 %);
* nested choices on the same training rows (the M39 optimism ledger): NEST_HW over the six half-widths and NEST_FAM over
  the family (grid + OP75 + OP90 + READMIN-2 + READSIN2 + WHOLECELL2): maximal inside R^2 subject to training coverage
  >= 0.70, ties toward larger training coverage, then registry order.

Estimands.  (a) Tail flag: outside / inside ratio of P(|res| > c), c = 0.5 and 0.3 (per-seed shares, seed means, ratio
of the means), with a 95 % two-sided percentile publication bootstrap (B = 2,000), all rows and without C5-BTBP.
(b) Bulk precision: median |res| inside / outside / all, and the q80 Mondrian conformal half-width on the inside rows,
window-aware (pair type x inside) against pair-type-only, both calibrated leave-one-cluster-out (cluster_cpg) within seed.
Controls at matched counts: random abstention keeping the rule's number of rows per (seed, cell) [CELL, primary], per
(seed, publication, pair type) [PUBPT] and per seed [UNSTRAT]; 2,000 draws, common random numbers across the rules of one
(design, variant, control); where the draws reproduce the rule (a rule constant within the control's strata, e.g. the
cell-level rules under CELL, or coverage 1) the share of draws at or above the rule is "n/a" and the row is flagged;
the CELL expectation also in closed form (E[SSE] = sum_r pi_r e_r^2, pi = the cell's kept
share; plug-in R^2 = 1 - E[SSE] / sum pi (y - ybar_pi)^2).  "Inside relative SSE" = 1 - MSE_inside / MSE_all per seed,
then the seed mean (at matched counts it ranks exactly as the inside SSE).
Spread: units = cluster_cpg (per-unit dSSE = E[SSE_CELL] - SSE_window, contributions c_u summing to the excess relative
SSE over the CELL expectation; SF_POOL / SIGN / SF_LOG / IM_REL via r5.common, DIAGNOSTIC); leave-one-publication-out
(r5.common.lopo); top-3 unit share; leave-top-k-cells-out (k = 5, 10; cells ranked by B1 SSE, outcome-selected);
winsorised curves (|res| capped at 0.3 and 0.5); C5-BTBP in / out (+ the single cell c950dc2982e96b83);
duplicate sensitivities (V2-1/2/3 cells; second cells of the Stage-0 bit-identity links).
Report rules (in-corpus CONSISTENCY CHECKs, never tests): "row flag" only if NEST_FAM's inside relative SSE beats the
97.5 % quantile of its CELL control, otherwise "cell or condition flag"; the tail flag passes if the |res| > 0.5 ratio at
OP75 is >= 3 with a bootstrap lower bound > 1, with and without C5-BTBP; bulk precision needs >= 15 % narrowing of the
mean q80 half-width at OP75 (expected to fail).

Leakage (brief 8): isolation on every fold (g21.evaluate.assert_test_isolation); training rows assert their cells are
in the outer training index and not in the test index; read sets equal the cached DOPT_Q[:3] and the parquet; no scored
pair with both ends read; strict adjacency; prediction intervention on every test cell instance (unread metals
scrambled, levels bit-identical) with a block-Y peek positive control that must be caught; fit poisoning of the
operating points and nested choices (r4.leakage.poisoned_bench of the fold's non-training labels; thresholds, choices
and training rows bit-identical) with a positive control that adds the test cells to the training rows and must be
caught.
"""
# Adapted (logic only; nothing run in place):
#   logic re-implemented from a session-scratchpad check (the evidence-base reconciliation check cited by EB
#     Appendix 2; not in the repository, so not a copy record and not checked by the seal): the HIST_D7-form level
#     window [-2, 2], its per-seed coverage and inside R^2 (the reproduction target of brief 6.0);
#   ideas/critiques.json M09 and M39 test designs (design-folder text, not code): the frozen grid, the nested choice
#     (inside R^2 at training coverage >= 0.70), the stratified controls and the window-aware Mondrian interval.
from __future__ import annotations

import json
import math
import time
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import paths
from . import common as CM
from r4 import refs as RF  # noqa: E402
from g21 import cache as GC  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import inference as INF  # noqa: E402
from gen20k.data import LANTHANIDES, N_LN  # noqa: E402

ITEM_ID, ESTIMAND = "1.4", "window axis (M09 level window; descriptive)"
OUT = paths.RES / "WINDOW"
LEAK = paths.RES / "leakage"
SEEN = "EXPLORATORY (seen)"
FIXED = "grid fixed after seeing V1"
IDX = {n: i for i, n in enumerate(LANTHANIDES)}
K_EA = 3                                                  # E-A's code budget: four reads (brief 3.3)
KEY = list(CM.KEY)

HALF_WIDTHS = (1.0, 1.5, 2.0, 2.5, 3.0, math.inf)
LOWER_ONLY = (-2.0, -2.5, -3.0, -4.0)
UPPER_ONLY = (2.0, 2.5, 3.0)
OP_COVERAGES = {"OP75": 0.75, "OP90": 0.90}               # label-free quantiles of s on training folds
NEST_MIN_COVERAGE = 0.70                                  # M39 / critiques M09: inside R^2 at coverage >= 0.70
TAIL_CUTS = (0.5, 0.3)
TAIL_RATIO_MIN = 3.0
BULK_NARROWING_MIN = 0.15
MONDRIAN_LEVEL = 0.80
WINSOR_CAPS = (0.3, 0.5)
TOPK = (5, 10)
REPS_CONTROL = 2000
REPS_BOOT = 2000
CI_LEVEL = 0.95
CI_LO, CI_HI = 0.025, 0.975                               # its percentile bounds, literal (ITEM_SPEC: inverted_cdf at
                                                          # 0.025; (1 - 0.95) / 2 = 0.025000000000000022 rounds up one rank)
CTRL_SAME_SD = 1e-12                                      # control draws this flat reproduce the rule (tie noise only)
NA_REPRO = "n/a (control reproduces the rule)"
BATCH = 250


def _hname(h: float) -> str:
    return "Hinf" if math.isinf(h) else f"H{h:g}"


HW_RULES = tuple(_hname(h) for h in HALF_WIDTHS)
LO_RULES = tuple(f"LO{v:g}" for v in LOWER_ONLY)
UP_RULES = tuple(f"UP{v:g}" for v in UPPER_ONLY)
GRID = HW_RULES + LO_RULES + UP_RULES
OP_RULES = tuple(OP_COVERAGES)
READ_RULES = ("READMIN-2", "READSIN2", "WHOLECELL2")
FAMILY = GRID + OP_RULES + READ_RULES                     # NEST_FAM's family (label-free score rules)
NEST_RULES = ("NEST_HW", "NEST_FAM")
ALT_RULES = ("READMIN-2", "READSIN2", "WHOLECELL2", "LO-4", "NOC5")
ALIAS = {"LO-4": 'drop predicted min < -4', "NOC5": "drop C5-BTBP", "READMIN-2": "min read >= -2 (M10 reads-only)",
         "READSIN2": "all four reads in [-2, 2] (M10 reads-only)", "WHOLECELL2": "whole-cell rule (H2 majority)"}
RULES = GRID + OP_RULES + READ_RULES + ("NOC5",) + NEST_RULES
LEVEL_WINDOWS = set(GRID) | set(OP_RULES)
CELL_LEVEL_RULES = ("Hinf",) + READ_RULES + ("NOC5",)    # constant within (seed, cell): the CELL control reproduces them
MAIN_RULES = ("OP75", "OP90", "H2", "NEST_FAM", "NEST_HW")
PRIMARY = "OP75"
CONTROLS = ("CELL", "PUBPT", "UNSTRAT")                   # CELL is the primary control
CONTROL_KEYS = {"CELL": ["split_seed", "cell_id"], "PUBPT": ["split_seed", "publication", "pt"],
                "UNSTRAT": ["split_seed"]}
VARIANTS = ("all", "noC5", "noC5cell", "noV2", "noBITDUP")
DRAW_VARIANTS = ("all", "noC5")                           # Monte Carlo controls and the publication bootstrap
ARMS = ("B1", "B0", "HIST_D7")
TARGETS = {"B1_vs_REF_max_abs": 8.9e-16,                  # brief 6.0 (P1 section 2)
           "rows_per_design": 9010,
           "BPG": {"R2_inside": 0.540, "R2_per_seed": (0.532, 0.521, 0.546, 0.547, 0.553),
                   "coverage_per_seed": (0.743, 0.754)},   # EB Appendix 2 (HIST_D7-form levels in [-2, 2])
           "BP": {"R2_inside": 0.538}}
TOL3 = 0.0005                                             # a target quoted to 3 decimals
SEEN_VALUES = {"H2": "0.540 at 74.7 % (post hoc; one of 4 windows) [EB 2.11; IR M09]",
               "NEST_HW": "0.527 at 75.3 % (M39; other folds' OOF rows) [IR M39]",
               "NEST_FAM": "0.516 at 78.7 % (family nesting) [IR M09, M39]",
               "CELL": "cell-matched 0.510-0.511 [0.500, 0.522] at the H2 coverage [IR M09]",
               "NOC5": "0.486 at 91.8 % [IR M09]", "tail": "P(|res| > 0.5) 0.009 inside vs 0.043 outside [IR M09]",
               "bulk": "window-aware Mondrian 11-12 % narrower than pair-type-only [IR M09]"}


class LeakageError(AssertionError):
    pass


# ------------------------------------------------------------------------------------------------------------
# labels and seeds
# ------------------------------------------------------------------------------------------------------------
def label_for(rule: str = "", variant: str = "all", design: str = "BPG", extra: str = "") -> str:
    """The section-9.1 label of one number (rule / data variant / design / kind)."""
    parts = [SEEN]
    if rule in GRID:
        parts.append(FIXED)
    if rule == "H2":
        parts.append("post hoc window [-2, 2]")
    if rule in OP_RULES:
        parts.append("operating point fitted on training folds")
    if rule in NEST_RULES:
        parts.append("nested on training folds (M39 optimism ledger)")
    if variant == "noC5":
        parts.append("SENSITIVITY (C5-BTBP out)")
    elif variant == "noC5cell":
        parts.append("SENSITIVITY (cell c950dc2982e96b83 out)")
    elif variant in ("noV2", "noBITDUP"):
        parts.append(f"SENSITIVITY (duplicates: {variant})")
    elif variant.startswith("top"):
        parts.append(f"SENSITIVITY (outcome-selected: {variant} cells by B1 SSE removed)")
    if design == "BP":
        parts.append("SENSITIVITY (design BP)")
    if extra:
        parts.append(extra)
    return "; ".join(parts)


def control_seed(design: str, variant: str, control: str) -> int:
    return paths.seed_for(f"WINDOW|control|{design}|{variant}|{control}")


def boot_seed(design: str, variant: str) -> int:
    return paths.seed_for(f"WINDOW|pubboot|{design}|{variant}")


def intervention_seed(design: str) -> int:
    return paths.seed_for(f"WINDOW|intervention|{design}")


def poison_seed(split_seed: int, fold: int) -> int:
    return int(split_seed) * 10 + int(fold)               # round 4's fit-poisoning seed


# ------------------------------------------------------------------------------------------------------------
# levels (exact r4.refs path)
# ------------------------------------------------------------------------------------------------------------
def b1_fit(beta: float) -> dict:
    return {"spec": RF.SPECS["B1"], "kind": "R", "coef": np.array([float(beta)])}


def levels(yrow, metals, read, beta) -> np.ndarray:
    """B1-form levels from the read metals only: r4.refs.predict_levels with u = beta at Gd (= y at a read metal)."""
    ym = np.full(N_LN, np.nan)
    rl = sorted(int(m) for m in read)
    ym[rl] = np.asarray(yrow, float)[rl]
    return RF.predict_levels(b1_fit(beta), SimpleNamespace(y_masked=ym), metals)


def read_set_of(seq, k: int = K_EA) -> frozenset:
    """Read set of code budget k (brief 3.3): the union of the DOPT_Q sequence steps seq[:k]."""
    return frozenset(int(m) for p in seq[:k] for m in p)


def pair_frame_cols(lev, qa, qb) -> dict:
    la, lb = lev[qa], lev[qb]
    return {"levA": la, "levB": lb, "s": np.maximum(np.abs(la), np.abs(lb)), "lmin": np.minimum(la, lb),
            "lmax": np.maximum(la, lb)}


# ------------------------------------------------------------------------------------------------------------
# rules
# ------------------------------------------------------------------------------------------------------------
def rule_inside(F: pd.DataFrame, name: str) -> np.ndarray:
    """Inside (kept) mask of one rule on a frame with s, lmin, lmax, readmin, readmax, inst, publication and, for the
    operating points, thr_OP75 / thr_OP90 (per-row fold thresholds); NEST_* read choice_<rule> per row."""
    if name in NEST_RULES:
        ch = F[f"choice_{name}"].astype(str).to_numpy()
        out = np.zeros(len(F), bool)
        for r in np.unique(ch):
            sel = ch == r
            out[sel] = rule_inside(F, r)[sel]
        return out
    s = F["s"].to_numpy(float)
    if name in HW_RULES:
        h = math.inf if name == "Hinf" else float(name[1:])
        return s <= h
    if name in LO_RULES:
        return F["lmin"].to_numpy(float) >= float(name[2:])
    if name in UP_RULES:
        return F["lmax"].to_numpy(float) <= float(name[2:])
    if name in OP_RULES:
        return s <= F[f"thr_{name}"].to_numpy(float)
    if name == "READMIN-2":
        return F["readmin"].to_numpy(float) >= -2.0
    if name == "READSIN2":
        return (F["readmin"].to_numpy(float) >= -2.0) & (F["readmax"].to_numpy(float) <= 2.0)
    if name == "WHOLECELL2":
        share = pd.Series((s <= 2.0).astype(float)).groupby(F["inst"].astype(str).to_numpy()).transform("mean")
        return share.to_numpy() >= 0.5
    if name == "NOC5":
        return F["publication"].astype(str).to_numpy() != paths.C5
    raise KeyError(name)


def quantile_threshold(s, coverage: float) -> float:
    """Label-free operating point: the smallest s with empirical CDF >= coverage (inverted CDF)."""
    return float(np.quantile(np.asarray(s, float), coverage, method="inverted_cdf"))


def _r2(y, p) -> float:
    return float(CM.M4.MT._r2(y, p))


def select_rule(F: pd.DataFrame, family, *, min_cov: float = NEST_MIN_COVERAGE) -> tuple[str, pd.DataFrame]:
    """Nested choice on training rows: maximal inside R^2 among the rules with training coverage >= min_cov; ties toward
    larger coverage, then registry order.  Falls back to Hinf (keep everything) when no rule qualifies."""
    y, p = F["y"].to_numpy(float), F["pred"].to_numpy(float)
    rows = []
    for i, name in enumerate(family):
        m = rule_inside(F, name)
        cov = float(m.mean()) if len(m) else 0.0
        r2 = _r2(y[m], p[m]) if m.sum() >= 2 else math.nan
        rows.append({"rule": name, "order": i, "train_coverage": cov, "train_R2_in": r2})
    tab = pd.DataFrame(rows)
    ok = tab[(tab.train_coverage >= min_cov) & np.isfinite(tab.train_R2_in)]
    if not len(ok):
        return "Hinf", tab
    best = ok.sort_values(["train_R2_in", "train_coverage", "order"], ascending=[False, False, True]).iloc[0]
    return str(best.rule), tab


# ------------------------------------------------------------------------------------------------------------
# training rows and the fold's operating points (label-free quantiles; nested choices)
# ------------------------------------------------------------------------------------------------------------
TRAIN_COLS = ["ci", "cell_id", "publication", "ia", "ib", "y", "pred", "levA", "levB", "readmin", "readmax"]


def training_frame(bench, rec: dict, beta: float, *, include_test: bool = False) -> pd.DataFrame:
    """Every strict-adjacent query of the fold's C3 training cells, masked with the cell's nested DOPT_Q[:3] read set of
    the fold record, levels with the fold's beta.  ``include_test`` (positive control of the poisoning check only) adds
    the fold's held-out cells, which a correct operating point never sees."""
    tr = set(map(int, rec["outer"]["train_index"]))
    te = set(map(int, rec["outer"]["test_index"]))
    cells = list(RF.training_cells(bench, rec))
    if include_test:
        cells += list(rec["outer"]["cells"])
    seen, rows = set(), []
    for cell in cells:
        if int(cell["k_max"]) < K_EA or not cell["query"]:
            continue
        ci = int(cell["ci"])
        if ci in seen:
            raise AssertionError(f"training cell {cell['cell_id']} appears twice")
        seen.add(ci)
        if not include_test and (ci not in tr or ci in te):
            raise LeakageError(f"operating-point row from a non-training cell {cell['cell_id']}")
        read = read_set_of(cell["seqs"]["DOPT_Q"])
        yrow = np.asarray(bench.Y[ci], float)
        lev = levels(yrow, cell["metals"], read, beta)
        rv = yrow[sorted(read)]
        for a, b in cell["query"]:
            if a in read and b in read:
                raise AssertionError("training query with both ends read")
            rows.append((ci, str(cell["cell_id"]), str(cell["publication"]), int(a), int(b), yrow[a] - yrow[b],
                         lev[a] - lev[b], lev[a], lev[b], float(rv.min()), float(rv.max())))
    F = pd.DataFrame(rows, columns=TRAIN_COLS)
    F["s"] = np.maximum(F.levA.abs(), F.levB.abs())
    F["lmin"] = np.minimum(F.levA, F.levB)
    F["lmax"] = np.maximum(F.levA, F.levB)
    F["inst"] = F.cell_id
    return F


def fold_ops(bench, rec: dict, beta: float, *, include_test: bool = False) -> tuple[dict, pd.DataFrame]:
    """The fold's label-free operating points (OP75, OP90) and the nested choices (NEST_HW, NEST_FAM) from its training
    rows only.  Returns (ops record, training frame)."""
    F = training_frame(bench, rec, beta, include_test=include_test)
    thr = {name: quantile_threshold(F.s, c) for name, c in OP_COVERAGES.items()}
    for name, v in thr.items():
        F[f"thr_{name}"] = v
    hw, _ = select_rule(F, HW_RULES)
    fam, tab = select_rule(F, FAMILY)
    o = {"design": rec["design"], "split_seed": int(rec["split_seed"]), "fold": int(rec["fold"]), "beta": float(beta),
         "train_cells": int(F.ci.nunique()), "train_rows": int(len(F)),
         **{f"thr_{k}": v for k, v in thr.items()}, "choice_NEST_HW": hw, "choice_NEST_FAM": fam}
    for r in tab.itertuples(index=False):
        o[f"train_cov_{r.rule}"] = float(r.train_coverage)
        o[f"train_R2_{r.rule}"] = float(r.train_R2_in)
    return o, F


def ops_equal(a: tuple, b: tuple) -> bool:
    """Bit-identity of two fold_ops results (records and training frames)."""
    (oa, Fa), (ob, Fb) = a, b
    if set(oa) != set(ob):
        return False
    for k, v in oa.items():
        w = ob[k]
        if isinstance(v, float):
            if not (np.float64(v).tobytes() == np.float64(w).tobytes()):
                return False
        elif v != w:
            return False
    if list(Fa.columns) != list(Fb.columns) or len(Fa) != len(Fb):
        return False
    for c in Fa.columns:
        x, y = Fa[c].to_numpy(), Fb[c].to_numpy()
        if x.dtype.kind == "f":
            if x.tobytes() != y.tobytes():
                return False
        elif not np.array_equal(x, y):
            return False
    return True


# ------------------------------------------------------------------------------------------------------------
# row assembly (test rows of every fold)
# ------------------------------------------------------------------------------------------------------------
CHECK_KEYS = ("folds", "isolation_ok", "rows", "cell_instances", "read_set_mismatch_parquet",
              "read_level_mismatch", "read_flag_mismatch", "both_ends_read", "not_strict_adjacent", "query_not_in_cell",
              "intervention_checked", "intervention_failed", "peek_control_runs", "peek_control_caught",
              "poison_checked", "poison_failed", "poison_control_runs", "poison_control_caught")


def peek_levels(yrow, metals, read, beta) -> np.ndarray:
    """Positive control of the intervention check: a predictor that reads every measured metal (a block-Y peek)."""
    yrow = np.asarray(yrow, float)
    return levels(yrow, metals, [m for m in range(N_LN) if np.isfinite(yrow[m])], beta)


def assemble(bench, design: str, *, folds: int | None = None, ops: bool = True, intervention: bool = True,
             poison: bool = True, log=print) -> dict:
    """Row-level frame of one design: the E-A rows with levels, s, reads, rule memberships and the fold's operating
    points; the per-fold ops table; the leakage / support counters."""
    from r4 import leakage as RL
    t0 = time.time()
    t = CM.load_rows(design)
    fitted = json.loads((paths.R4RES / "REF" / f"fitted_{design}.json").read_bytes())
    keys = sorted(set(zip(t.split_seed.astype(int), t.fold.astype(int))))
    if folds is not None:
        keys = keys[:folds]
    ck = Counter({k: 0 for k in CHECK_KEYS})
    rng = np.random.default_rng(intervention_seed(design))
    parts, oprows = [], []
    for seed, fold in keys:
        tf = time.time()
        rec = GC.load(design, seed, fold)
        paths.redirect_inner_cache()
        E.assert_test_isolation(bench, rec)
        ck["isolation_ok"] += 1
        ck["folds"] += 1
        beta = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
        outer = {str(c["cell_id"]): c for c in rec["outer"]["cells"]}
        g = t[(t.split_seed == seed) & (t.fold == fold)]
        for cid, gc in g.groupby("cell_id", sort=True):
            cell = outer[str(cid)]
            ci = int(cell["ci"])
            read = read_set_of(cell["seqs"]["DOPT_Q"])
            rs = set(gc.read_set.astype(str))
            if len(rs) != 1 or frozenset(IDX[m] for m in next(iter(rs)).split(",")) != read:
                ck["read_set_mismatch_parquet"] += 1
            yrow = np.asarray(bench.Y[ci], float)
            lev = levels(yrow, cell["metals"], read, beta)
            lev7 = levels(yrow, cell["metals"], read, RF.HIST_D7_BETA)
            if any(lev[m] != yrow[m] for m in read):
                ck["read_level_mismatch"] += 1
            qa, qb = gc.ia.to_numpy(int), gc.ib.to_numpy(int)
            qset = set(map(tuple, cell["query"]))
            ck["query_not_in_cell"] += int(sum((int(a), int(b)) not in qset for a, b in zip(qa, qb)))
            ra, rb = np.isin(qa, list(read)), np.isin(qb, list(read))
            ck["both_ends_read"] += int((ra & rb).sum())
            ck["read_flag_mismatch"] += int((ra != gc.a_read.to_numpy(bool)).sum() + (rb != gc.b_read.to_numpy(bool)).sum())
            ck["not_strict_adjacent"] += int((gc.dZ.to_numpy(int) != 1).sum())
            if intervention:
                ys = E.scramble_hidden(yrow, tuple(sorted(read)), rng)
                ck["intervention_checked"] += 1
                if not np.array_equal(levels(ys, cell["metals"], read, beta), lev, equal_nan=True):
                    ck["intervention_failed"] += 1
                qm = sorted(set(qa) | set(qb))
                ck["peek_control_runs"] += 1
                if not np.array_equal(peek_levels(ys, cell["metals"], read, beta)[qm],
                                      peek_levels(yrow, cell["metals"], read, beta)[qm]):
                    ck["peek_control_caught"] += 1
            rv = yrow[sorted(read)]
            d = gc.copy()
            for k, v in pair_frame_cols(lev, qa, qb).items():
                d[k] = v
            d7 = pair_frame_cols(lev7, qa, qb)
            d["levA_D7"], d["levB_D7"], d["s_D7"] = d7["levA"], d7["levB"], d7["s"]
            d["B1_lev"] = lev[qa] - lev[qb]
            d["HIST_D7_lev"] = lev7[qa] - lev7[qb]
            d["readmin"], d["readmax"] = float(rv.min()), float(rv.max())
            d["beta_fold"] = beta
            parts.append(d)
            ck["cell_instances"] += 1
        if ops:
            o, F = fold_ops(bench, rec, beta)
            if poison:
                pb, _ = RL.poisoned_bench(bench, rec["outer"]["train_index"], seed=poison_seed(seed, fold))
                same = ops_equal((o, F), fold_ops(pb, rec, beta))
                ck["poison_checked"] += 1
                ck["poison_failed"] += int(not same)
                ck["poison_control_runs"] += 1
                ck["poison_control_caught"] += int(not ops_equal(fold_ops(bench, rec, beta, include_test=True),
                                                                 fold_ops(pb, rec, beta, include_test=True)))
                for k in [k for k in RF._ORPHANS if k[0] == id(pb)]:
                    RF._ORPHANS.pop(k, None)
                o["poison_identical"] = bool(same)
            oprows.append(o)
        log(f"    {design} {seed} fold {fold}: {len(g)} rows, {g.cell_id.nunique()} cells"
            + (f", training rows {oprows[-1]['train_rows']} ({oprows[-1]['train_cells']} cells)" if ops else "")
            + f" ({time.time() - tf:.1f}s)")
    R = pd.concat(parts, ignore_index=True).sort_values(KEY).reset_index(drop=True)
    R["pt"] = R.A.astype(str) + "/" + R.B.astype(str)
    R["inst"] = R.split_seed.astype(str) + "|" + R.cell_id.astype(str)
    O = pd.DataFrame(oprows)
    if ops:
        R = R.merge(O[["split_seed", "fold", "thr_OP75", "thr_OP90", "choice_NEST_HW", "choice_NEST_FAM"]],
                    on=["split_seed", "fold"], how="left", validate="many_to_one")
        rules = RULES
    else:
        rules = GRID + READ_RULES + ("NOC5",)
    for r in rules:
        R[f"in_{r}"] = rule_inside(R, r)
    R["in_D7_H2"] = R.s_D7.to_numpy(float) <= 2.0
    ck["rows"] = int(len(R))
    checks = dict(ck)
    checks["B1_identity_max_abs"] = float(np.max(np.abs(R.B1_lev - R.B1)))
    checks["HIST_D7_identity_max_abs"] = float(np.max(np.abs(R.HIST_D7_lev - R.HIST_D7)))
    checks["nan_levels"] = int(R[["levA", "levB", "s"]].isna().any(axis=1).sum())
    checks["seconds"] = round(time.time() - t0, 1)
    checks["label"] = "leakage / support counters (construction checks of brief 8; no score)"
    return {"rows": R, "ops": O, "checks": checks}


def checks_failed(checks: dict) -> list:
    """Names of failed leakage / support counters (empty = all passed)."""
    bad = [k for k in ("read_set_mismatch_parquet", "read_level_mismatch", "read_flag_mismatch", "both_ends_read",
                       "not_strict_adjacent", "query_not_in_cell", "intervention_failed", "poison_failed", "nan_levels")
           if checks.get(k)]
    if checks.get("isolation_ok") != checks.get("folds"):
        bad.append("isolation")
    if checks.get("peek_control_caught") != checks.get("peek_control_runs"):
        bad.append("peek_control_not_caught")
    if checks.get("poison_control_caught") != checks.get("poison_control_runs"):
        bad.append("poison_control_not_caught")
    if checks.get("B1_identity_max_abs", 1.0) > TARGETS["B1_vs_REF_max_abs"]:
        bad.append("B1_identity")
    return bad


# ------------------------------------------------------------------------------------------------------------
# reproduction (brief 6.0; TECHNICAL SMOKE)
# ------------------------------------------------------------------------------------------------------------
def repro_checks(R: pd.DataFrame, design: str, *, scores: bool = True) -> dict:
    """Identity: B1 from the fold betas equals REF within 8.9e-16 on every row.  Scores (post-seal only): the [-2, 2]
    window on HIST_D7-form levels (EB Appendix 2's method) -- coverage per seed and B1 inside R^2 -- against the targets;
    the same window on B1-form levels (the brief's rule) is reported beside it, not gating."""
    out = {"label": "TECHNICAL SMOKE (reproduction target, brief 6.0)", "design": design, "rows": int(len(R)),
           "rows_target": TARGETS["rows_per_design"], "B1_identity_max_abs": float(np.max(np.abs(R.B1_lev - R.B1))),
           "HIST_D7_identity_max_abs": float(np.max(np.abs(R.HIST_D7_lev - R.HIST_D7)))}
    out["identity_ok"] = bool(out["B1_identity_max_abs"] <= TARGETS["B1_vs_REF_max_abs"])
    if not scores:
        return out
    v = Rows(R)
    res = {}
    for name, col in (("HIST_D7_levels", "in_D7_H2"), ("B1_levels", "in_H2")):
        st = mask_stats(v, R[col].to_numpy(bool))
        res[name] = {"coverage_per_seed": st["coverage_per_seed"], "coverage": st["coverage"], "R2_in": st["R2_in"],
                     "R2_in_per_seed": st["R2_in_per_seed"]}
    out["window_H2"] = res
    tg = TARGETS.get(design, {})
    eb = res["HIST_D7_levels"]
    ok = {"rows": out["rows"] == TARGETS["rows_per_design"], "identity": out["identity_ok"]}
    if "R2_inside" in tg:
        ok["R2_inside"] = abs(eb["R2_in"] - tg["R2_inside"]) <= TOL3
    if "R2_per_seed" in tg:
        ok["R2_per_seed"] = bool(np.all(np.abs(np.array(eb["R2_in_per_seed"]) - np.array(tg["R2_per_seed"])) <= TOL3))
    if "coverage_per_seed" in tg:
        lo, hi = tg["coverage_per_seed"]
        cps = np.array(eb["coverage_per_seed"])
        ok["coverage_per_seed"] = bool(cps.min() >= lo - TOL3 and cps.max() <= hi + TOL3)
    out["targets"] = {k: (v2 if not isinstance(v2, tuple) else list(v2)) for k, v2 in tg.items()}
    out["target_ok"] = ok
    out["B1_levels_close_to_target"] = (bool(abs(res["B1_levels"]["R2_in"] - tg["R2_inside"]) <= 3 * TOL3)
                                        if "R2_inside" in tg else None)
    out["ok"] = bool(all(ok.values()))
    return out


# ------------------------------------------------------------------------------------------------------------
# row sets and statistics
# ------------------------------------------------------------------------------------------------------------
class Rows:
    """One scored row set (a data variant): per-seed index arrays, residuals and cached stratum codes."""

    def __init__(self, F: pd.DataFrame):
        self.F = F.reset_index(drop=True)
        self.seeds = np.sort(self.F.split_seed.unique())
        self.S = len(self.seeds)
        self.sc = np.searchsorted(self.seeds, self.F.split_seed.to_numpy())
        self.idx = [np.flatnonzero(self.sc == i) for i in range(self.S)]
        self.y = self.F.y.to_numpy(float)
        self.e = self.y - self.F.B1.to_numpy(float)
        self.e2 = self.e ** 2
        self.mse_all = np.array([self.e2[ix].mean() for ix in self.idx])
        self._codes = {}

    def m(self, rule: str) -> np.ndarray:
        return self.F[f"in_{rule}"].to_numpy(bool)

    def codes(self, control: str) -> np.ndarray:
        """Stratum codes in sorted key order (the seed outermost, so strata of one seed are contiguous)."""
        if control not in self._codes:
            self._codes[control] = self.F.groupby(CONTROL_KEYS[control], sort=True).ngroup().to_numpy()
        return self._codes[control]


def _sst(y) -> float:
    return float(((y - y.mean()) ** 2).sum()) if len(y) else 0.0


def _r2e(y, e2) -> float:
    sst = _sst(y)
    return float(1.0 - e2.sum() / sst) if len(y) >= 2 and sst > 0 else math.nan


def mask_stats(v: Rows, m: np.ndarray, *, arm: str = "B1", cap: float | None = None) -> dict:
    """Per-seed coverage, inside / outside / all R^2 and MSE, inside relative SSE (1 - MSE_in / MSE_all) and var(y);
    seed means plus per-seed lists.  ``cap`` winsorises |res| at cap (SST unchanged)."""
    e = v.y - v.F[arm].to_numpy(float)
    if cap is not None:
        e = np.clip(e, -cap, cap)
    e2 = e ** 2
    rec = {k: [] for k in ("coverage", "n_in", "R2_in", "R2_out", "R2_all", "MSE_in", "MSE_out", "MSE_all", "relMSE_in",
                           "var_y_in", "var_y_all")}
    for ix in v.idx:
        ms = m[ix]
        y, ee = v.y[ix], e2[ix]
        n_in = int(ms.sum())
        rec["coverage"].append(n_in / len(ix))
        rec["n_in"].append(n_in)
        rec["R2_in"].append(_r2e(y[ms], ee[ms]))
        rec["R2_out"].append(_r2e(y[~ms], ee[~ms]))
        rec["R2_all"].append(_r2e(y, ee))
        mi = float(ee[ms].mean()) if n_in else math.nan
        mo = float(ee[~ms].mean()) if n_in < len(ix) else math.nan
        ma = float(ee.mean())
        rec["MSE_in"].append(mi)
        rec["MSE_out"].append(mo)
        rec["MSE_all"].append(ma)
        rec["relMSE_in"].append(1.0 - mi / ma if n_in else math.nan)
        rec["var_y_in"].append(float(y[ms].var()) if n_in else math.nan)
        rec["var_y_all"].append(float(y.var()))
    out = {k: float(np.mean(x)) for k, x in rec.items()}
    out["coverage_per_seed"] = [float(x) for x in rec["coverage"]]
    out["R2_in_per_seed"] = [float(x) for x in rec["R2_in"]]
    out["relMSE_in_per_seed"] = [float(x) for x in rec["relMSE_in"]]
    return out


def expected_control(v: Rows, m: np.ndarray, control: str = "CELL") -> dict:
    """Closed-form expectation of random abstention at the rule's per-stratum counts: E[SSE] = sum pi e^2 (exact),
    relative SSE 1 - E[MSE] / MSE_all (exact), plug-in R^2 = 1 - E[SSE] / sum pi (y - ybar_pi)^2 (approximation)."""
    codes = v.codes(control)
    N = np.bincount(codes)
    n = np.bincount(codes, weights=m.astype(float), minlength=len(N))
    pi = (n / N)[codes]
    r2, rel, esse = [], [], []
    for s, ix in enumerate(v.idx):
        w, y, ee = pi[ix], v.y[ix], v.e2[ix]
        W = w.sum()
        es = float((w * ee).sum())
        esse.append(es)
        if W <= 0:
            r2.append(math.nan)
            rel.append(math.nan)
            continue
        yb = float((w * y).sum() / W)
        est = float((w * (y - yb) ** 2).sum())
        r2.append(1.0 - es / est if est > 0 else math.nan)
        rel.append(1.0 - (es / W) / v.mse_all[s])
    return {"R2_plugin": float(np.mean(r2)), "relMSE": float(np.mean(rel)), "E_SSE_per_seed": esse, "pi": pi}


def control_draws(v: Rows, control: str, masks: dict, reps: int, seed: int, *, batch: int = BATCH) -> dict:
    """Random abstention keeping each rule's number of rows per stratum; common random numbers across the rules (one
    random order per draw, sorted by stratum then U(0, 1)).  Returns {rule: {'R2': (reps,), 'relMSE': (reps,)}}."""
    codes = v.codes(control)
    N = len(codes)
    C = int(codes.max()) + 1
    Ncnt = np.bincount(codes, minlength=C)
    starts = np.concatenate([[0], np.cumsum(Ncnt)[:-1]])
    seed_of_code = np.zeros(C, int)
    seed_of_code[codes] = v.sc
    rng = np.random.default_rng(seed)
    order = np.empty((reps, N), np.int32)
    for b0 in range(0, reps, batch):
        b1 = min(reps, b0 + batch)
        order[b0:b1] = np.argsort(codes[None, :] + rng.random((b1 - b0, N)), axis=1)
    out = {}
    for name, m in masks.items():
        nk = np.bincount(codes, weights=m.astype(float), minlength=C).astype(int)
        tot = int(nk.sum())
        offs = np.cumsum(nk) - nk
        pos = np.repeat(starts, nk) + (np.arange(tot) - np.repeat(offs, nk))
        n_seed = np.bincount(seed_of_code, weights=nk, minlength=v.S).astype(int)
        if (n_seed < 2).any():
            out[name] = {"R2": np.full(reps, np.nan), "relMSE": np.full(reps, np.nan)}
            continue
        bnd = np.concatenate([[0], np.cumsum(n_seed)[:-1]])
        R2, REL = np.empty(reps), np.empty(reps)
        for b0 in range(0, reps, batch):
            b1 = min(reps, b0 + batch)
            ix = order[b0:b1][:, pos]
            yk, ek = v.y[ix], v.e2[ix]
            sy = np.add.reduceat(yk, bnd, axis=1)
            syy = np.add.reduceat(yk * yk, bnd, axis=1)
            sse = np.add.reduceat(ek, bnd, axis=1)
            sst = syy - sy * sy / n_seed[None, :]
            R2[b0:b1] = np.mean(1.0 - sse / sst, axis=1)
            REL[b0:b1] = np.mean(1.0 - (sse / n_seed[None, :]) / v.mse_all[None, :], axis=1)
        out[name] = {"R2": R2, "relMSE": REL}
    del order
    return out


def _q(x, q):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.quantile(x, q, method="inverted_cdf")) if len(x) else math.nan


def tail_sums(v: Rows, m: np.ndarray, cut: float, pub_codes: np.ndarray, P: int) -> dict:
    """Per (publication, seed) counts of inside / outside rows and of rows with |res| > cut."""
    big = np.abs(v.e) > cut
    key = pub_codes * v.S + v.sc

    def bc(w):
        return np.bincount(key, weights=w.astype(float), minlength=P * v.S).reshape(P, v.S)
    return {"k_in": bc(m & big), "n_in": bc(m), "k_out": bc(~m & big), "n_out": bc(~m)}


def tail_ratio_from(K: dict) -> tuple:
    """(P_in, P_out, ratio): per-seed shares, seed means, ratio of the means (inf when P_in = 0 < P_out)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pin = np.nanmean(K["k_in"] / K["n_in"], axis=-1)
        pout = np.nanmean(K["k_out"] / K["n_out"], axis=-1)
        ratio = pout / pin
    return pin, pout, ratio


def tail_stats(v: Rows, m: np.ndarray, cut: float, pub_codes: np.ndarray, P: int, W: np.ndarray | None) -> dict:
    T = tail_sums(v, m, cut, pub_codes, P)
    pin, pout, ratio = tail_ratio_from({k: x.sum(axis=0) for k, x in T.items()})
    out = {"P_in": float(pin), "P_out": float(pout), "ratio": float(ratio),
           "k_in_per_seed": T["k_in"].sum(axis=0).astype(int).tolist(), "n_in_per_seed": T["n_in"].sum(axis=0).astype(int).tolist(),
           "k_out_per_seed": T["k_out"].sum(axis=0).astype(int).tolist(),
           "n_out_per_seed": T["n_out"].sum(axis=0).astype(int).tolist()}
    if W is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            Kb = {k: W @ x for k, x in T.items()}
        _, _, rb = tail_ratio_from(Kb)
        out.update({"boot_lo": _q(rb, CI_LO), "boot_hi": _q(rb, CI_HI), "boot_P_le_1": float(np.nanmean(rb <= 1.0)),
                    "boot_n_inf": int(np.isinf(rb).sum()), "boot_n_nan": int(np.isnan(rb).sum()), "boot_B": int(len(rb))})
    return out


def conformal_q(x, level: float = MONDRIAN_LEVEL) -> float:
    """Split-conformal half-width: the ceil((n + 1) level)-th smallest |res| of the calibration set (inf if > n)."""
    x = np.asarray(x, float)
    n = len(x)
    k = math.ceil((n + 1) * level)
    if n == 0 or k > n:
        return math.inf
    return float(np.partition(x, k - 1)[k - 1])


def mondrian_halfwidths(v: Rows, m: np.ndarray, *, level: float = MONDRIAN_LEVEL) -> tuple[np.ndarray, np.ndarray]:
    """q80 Mondrian half-widths per row, calibrated leave-one-cluster-out within seed: pair-type-only (PT) and
    window-aware (pair type x inside / outside the window; PTW).  The row's own cluster never calibrates it."""
    ae = np.abs(v.e)
    pt = v.F.pt.astype(str).to_numpy()
    cl = v.F.cluster_cpg.astype(str).to_numpy()
    hw_pt, hw_ptw = np.full(len(ae), np.nan), np.full(len(ae), np.nan)
    for ix in v.idx:
        for p in np.unique(pt[ix]):
            sp = ix[pt[ix] == p]
            for g in np.unique(cl[sp]):
                tgt = sp[cl[sp] == g]
                cal = sp[cl[sp] != g]
                hw_pt[tgt] = conformal_q(ae[cal], level)
                for st in (True, False):
                    tw = tgt[m[tgt] == st]
                    if len(tw):
                        hw_ptw[tw] = conformal_q(ae[cal[m[cal] == st]], level)
    return hw_pt, hw_ptw


def bulk_stats(v: Rows, m: np.ndarray) -> dict:
    """Median |res| inside / outside / all and the inside-row q80 Mondrian half-widths, window-aware vs pair-type-only."""
    ae = np.abs(v.e)
    hw_pt, hw_ptw = mondrian_halfwidths(v, m)
    cl = v.F.cluster_cpg.astype(str).to_numpy()
    rec = {k: [] for k in ("med_in", "med_out", "med_all", "hw_pt_mean", "hw_ptw_mean", "hw_pt_median", "hw_ptw_median",
                           "cov_pt_in", "cov_ptw_in", "n_inf_pt", "n_inf_ptw", "min_cluster_cov_ptw")}
    for ix in v.idx:
        ms = m[ix]
        a = ae[ix]
        rec["med_in"].append(float(np.median(a[ms])) if ms.any() else math.nan)
        rec["med_out"].append(float(np.median(a[~ms])) if (~ms).any() else math.nan)
        rec["med_all"].append(float(np.median(a)))
        ii = ix[ms]
        hp, hw = hw_pt[ii], hw_ptw[ii]
        fp, fw = np.isfinite(hp), np.isfinite(hw)
        rec["hw_pt_mean"].append(float(hp[fp].mean()) if fp.any() else math.nan)
        rec["hw_ptw_mean"].append(float(hw[fw].mean()) if fw.any() else math.nan)
        rec["hw_pt_median"].append(float(np.median(hp)) if len(hp) else math.nan)
        rec["hw_ptw_median"].append(float(np.median(hw)) if len(hw) else math.nan)
        rec["cov_pt_in"].append(float((ae[ii] <= hp).mean()) if len(ii) else math.nan)
        rec["cov_ptw_in"].append(float((ae[ii] <= hw).mean()) if len(ii) else math.nan)
        rec["n_inf_pt"].append(int((~fp).sum()))
        rec["n_inf_ptw"].append(int((~fw).sum()))
        cc = [float((ae[ii][cl[ii] == g] <= hw[cl[ii] == g]).mean()) for g in np.unique(cl[ii])]
        rec["min_cluster_cov_ptw"].append(min(cc) if cc else math.nan)
    out = {k: float(np.mean(x)) for k, x in rec.items()}
    out["narrowing_mean"] = 1.0 - out["hw_ptw_mean"] / out["hw_pt_mean"]
    out["narrowing_median"] = 1.0 - out["hw_ptw_median"] / out["hw_pt_median"]
    out["median_abs_res_narrowing_vs_all"] = 1.0 - out["med_in"] / out["med_all"]
    return out


def unit_frame(v: Rows, m: np.ndarray, unit: str = "cluster_cpg") -> pd.DataFrame:
    """Per unit (seed means of per-seed sums): rows, inside rows, coverage, MSE inside / outside, tail counts at 0.5,
    SSE of the window and of the CELL expectation; dSSE = E[SSE_CELL] - SSE_window, L = log(E / SSE_window),
    rel = 1 - SSE_window / E, and the contribution c_u = mean_s dSSE_us / (n_in_s MSE_all_s) (sums to the excess)."""
    pi = expected_control(v, m, "CELL")["pi"]
    n_in_seed = np.array([m[ix].sum() for ix in v.idx], float)
    g = pd.DataFrame({"u": v.F[unit].astype(str).to_numpy(), "s": v.sc, "one": 1.0, "in": m.astype(float),
                      "sse_in": m * v.e2, "sse_out": (~m) * v.e2, "esse": pi * v.e2,
                      "tail_in": (m & (np.abs(v.e) > 0.5)).astype(float), "tail_out": (~m & (np.abs(v.e) > 0.5)).astype(float)})
    ps = g.groupby(["u", "s"])[["one", "in", "sse_in", "sse_out", "esse", "tail_in", "tail_out"]].sum()
    s_idx = ps.index.get_level_values("s").to_numpy()
    ps["c"] = (ps.esse - ps.sse_in) / (n_in_seed[s_idx] * v.mse_all[s_idx])
    mu = ps.groupby(level="u").mean()
    out = pd.DataFrame({"rows_per_seed": mu.one, "inside_per_seed": mu["in"], "coverage": mu["in"] / mu.one,
                        "MSE_in": mu.sse_in / mu["in"].where(mu["in"] > 0), "MSE_out": mu.sse_out / (mu.one - mu["in"]).where(mu.one > mu["in"]),
                        "tail05_in_per_seed": mu.tail_in, "tail05_out_per_seed": mu.tail_out,
                        "SSE_window": mu.sse_in, "SSE_cell_expected": mu.esse,
                        "n_seeds": ps.groupby(level="u").size(), "c_u": ps.c.groupby(level="u").sum() / v.S})
    out.index.name = unit
    out["MSE_ratio_in_out"] = out.MSE_in / out.MSE_out
    out["dSSE"] = out.SSE_cell_expected - out.SSE_window
    with np.errstate(divide="ignore", invalid="ignore"):
        out["L"] = np.log(out.SSE_cell_expected / out.SSE_window)
        out["rel"] = 1.0 - out.SSE_window / out.SSE_cell_expected
    out["win"], out["loss"], out["tie"] = out.dSSE > 0, out.dSSE < 0, out.dSSE == 0
    return out


def unit_tests_row(us: pd.DataFrame, key: str) -> dict:
    """SF_POOL (exact over 2^G when G <= 13), SIGN, SF_LOG and IM_REL on the unit table (DIAGNOSTIC)."""
    fin = np.isfinite(us.L).all()
    sfp = CM.sf_pool(us.dSSE.to_numpy(float), key + "|SF_POOL")
    sg = CM.sign_test(us.dSSE.to_numpy(float))
    sfl = CM.sf_pool(us.L.to_numpy(float), key + "|SF_LOG") if fin else None
    im = CM.im_t(us.rel.to_numpy(float))
    return {"G": int(len(us)), "W/L/T": f"{sg['W']}/{sg['L']}/{sg['T']}", "SF_POOL_p": sfp["p"], "SF_POOL_exact": sfp["exact"],
            "SIGN_p": sg["p"], "SF_LOG_p": sfl["p"] if sfl else math.nan, "IM_REL_t": im["t"], "IM_REL_p": im["p"],
            "sum_c_u": float(us.c_u.sum()), "top3_share": CM.top3_share(us.c_u.to_numpy(float))}


# ------------------------------------------------------------------------------------------------------------
# frame-level statistics (for leave-one-publication-out and leave-top-k-cells-out)
# ------------------------------------------------------------------------------------------------------------
def frame_stat(F: pd.DataFrame, rule: str, stat: str) -> float:
    v = Rows(F)
    m = v.m(rule)
    if stat == "R2_in":
        return mask_stats(v, m)["R2_in"]
    if stat == "coverage":
        return mask_stats(v, m)["coverage"]
    if stat == "excess_relMSE_cell":
        return mask_stats(v, m)["relMSE_in"] - expected_control(v, m, "CELL")["relMSE"]
    if stat == "excess_R2_cell_plugin":
        return mask_stats(v, m)["R2_in"] - expected_control(v, m, "CELL")["R2_plugin"]
    if stat.startswith("tail_ratio_"):
        cut = float(stat.split("_")[-1])
        pubs = np.sort(v.F.publication.astype(str).unique())
        pc = np.searchsorted(pubs, v.F.publication.astype(str).to_numpy())
        return tail_stats(v, m, cut, pc, len(pubs), None)["ratio"]
    raise KeyError(stat)


LOPO_STATS = ("R2_in", "excess_relMSE_cell", "excess_R2_cell_plugin", "tail_ratio_0.5")


def top_cells(F: pd.DataFrame, k: int) -> list:
    """The k cells with the largest B1 SSE over every row of every seed (label-based; ties -> cell_id order)."""
    s = ((F.y - F.B1) ** 2).groupby(F.cell_id.astype(str).to_numpy()).sum()
    d = pd.DataFrame({"cell_id": s.index.astype(str), "sse": s.to_numpy(float)})
    return d.sort_values(["sse", "cell_id"], ascending=[False, True]).cell_id.head(k).tolist()


def duplicate_sets() -> dict:
    """Duplicate sensitivities (as item 1.6): the round-3 V2 drop set (V2-1/2/3) and the second cell of every Stage-0
    flagged bit-identity link (results/stage0/dupscan_links.csv, flag_bit)."""
    out = {}
    try:
        from . import population as PO
        out["noV2"] = set(PO.v2drop_set())
    except Exception as exc:  # population.py is written by another agent; report the variant as unavailable
        out["noV2"] = None
        print(f"  duplicate sensitivity noV2 unavailable: {exc!r}")
    p = paths.RES / "stage0" / "dupscan_links.csv"
    if p.exists():
        d = pd.read_csv(p)
        flag = d.flag_bit.astype(str).str.lower() == "true"
        out["noBITDUP"] = set(d[flag].cell_b.astype(str))
    else:
        out["noBITDUP"] = None
    return out


def variant_rows(R: pd.DataFrame, variant: str, dups: dict | None = None) -> pd.DataFrame | None:
    cid, pub = R.cell_id.astype(str), R.publication.astype(str)
    if variant == "all":
        return R
    if variant == "noC5":
        return R[pub != paths.C5]
    if variant == "noC5cell":
        return R[cid != paths.C5_CELL]
    if variant in ("noV2", "noBITDUP"):
        cells = (dups or {}).get(variant)
        return None if cells is None else R[~cid.isin(cells)]
    raise KeyError(variant)


# ------------------------------------------------------------------------------------------------------------
# the item's tables for one design
# ------------------------------------------------------------------------------------------------------------
def _rec(design, table, rule, variant, label, **kw) -> dict:
    return {"design": design, "table": table, "rule": rule, "variant": variant, **kw, "label": label}


def _ls(x) -> str:
    return ";".join(f"{float(v):.4f}" for v in x)


def cell_constant_fold_fits(F: pd.DataFrame, rule: str) -> int:
    """Fold-fits whose ``rule`` membership is constant within every (seed, cell) of the scored rows: there the CELL
    control reproduces the rule whatever rule a nested choice picked (a level window can keep or drop whole cells)."""
    g = F.groupby(["split_seed", "fold", "cell_id"])[f"in_{rule}"].nunique()
    return int((g.groupby(level=[0, 1]).max() <= 1).sum())


def analyse(R: pd.DataFrame, O: pd.DataFrame, design: str, *, reps_control: int = REPS_CONTROL,
            reps_boot: int = REPS_BOOT, lopo: bool = True, log=print) -> dict:
    """Every table of the item for one design (row-level memberships already in R)."""
    t0 = time.time()
    dups = duplicate_sets()
    T = {}
    V = {}
    for var in VARIANTS:
        F = variant_rows(R, var, dups)
        if F is not None and len(F):
            V[var] = Rows(F)
    # ---- curve: every rule x variant (coverage, R2 in / out, MSE, relative SSE, CELL expectation, winsorised) ----
    rows = []
    for var, v in V.items():
        for r in RULES:
            m = v.m(r)
            st = mask_stats(v, m)
            ex = expected_control(v, m, "CELL")
            rec = _rec(design, "curve", r, var, label_for(r, var, design), alias=ALIAS.get(r, ""),
                       coverage=st["coverage"], coverage_per_seed=_ls(st["coverage_per_seed"]), R2_in=st["R2_in"],
                       R2_in_per_seed=_ls(st["R2_in_per_seed"]), R2_out=st["R2_out"], R2_all=st["R2_all"],
                       MSE_in=st["MSE_in"], MSE_out=st["MSE_out"], MSE_all=st["MSE_all"], relMSE_in=st["relMSE_in"],
                       relMSE_in_per_seed=_ls(st["relMSE_in_per_seed"]), var_y_in=st["var_y_in"],
                       var_y_all=st["var_y_all"], CELL_expected_R2_plugin=ex["R2_plugin"],
                       CELL_expected_relMSE=ex["relMSE"], excess_R2_vs_CELL_plugin=st["R2_in"] - ex["R2_plugin"],
                       excess_relMSE_vs_CELL=st["relMSE_in"] - ex["relMSE"])
            for arm in ARMS[1:]:
                rec[f"R2_in_{arm}"] = mask_stats(v, m, arm=arm)["R2_in"]
            rows.append(rec)
    T["curve"] = pd.DataFrame(rows)
    T["curve"]["label"] = [lb + ("; R2_in_HIST_D7 DIAGNOSTIC (HIST_D7 not fold-independent)") for lb in T["curve"].label]
    log(f"    {design} curve: {len(T['curve'])} rows ({time.time() - t0:.0f}s)")
    # ---- winsorised curves ----
    rows = []
    for var in DRAW_VARIANTS:
        v = V[var]
        for r in GRID + OP_RULES + NEST_RULES:
            m = v.m(r)
            for cap in WINSOR_CAPS:
                st = mask_stats(v, m, cap=cap)
                rows.append(_rec(design, "winsor", r, var, label_for(r, var, design, f"SENSITIVITY (|res| winsorised at {cap:g})"),
                                 cap=cap, coverage=st["coverage"], R2w_in=st["R2_in"], R2w_all=st["R2_all"],
                                 R2w_out=st["R2_out"], MSEw_in=st["MSE_in"], MSEw_all=st["MSE_all"],
                                 R2w_in_minus_all=st["R2_in"] - st["R2_all"]))
    T["winsor"] = pd.DataFrame(rows)
    # ---- Monte Carlo controls at matched counts (CELL primary, PUBPT, UNSTRAT) ----
    ctrl_rules = RULES
    rows, CD = [], {}
    for var in DRAW_VARIANTS:
        v = V[var]
        masks = {r: v.m(r) for r in ctrl_rules}
        point = {r: mask_stats(v, m) for r, m in masks.items()}
        for c in CONTROLS:
            tc = time.time()
            D = control_draws(v, c, masks, reps_control, control_seed(design, var, c))
            CD[(var, c)] = D
            for r in ctrl_rules:
                d, p = D[r], point[r]
                # draws that keep the rule's own rows (a stratum-constant rule, coverage 1): the shares are tie noise
                same = bool(np.nanstd(d["R2"]) < CTRL_SAME_SD and np.nanstd(d["relMSE"]) < CTRL_SAME_SD)
                rows.append(_rec(design, "controls", r, var, label_for(r, var, design, f"random-abstention control {c}"
                                                                       + (" (primary)" if c == "CELL" else "")),
                                 control=c, reps=reps_control, seed=control_seed(design, var, c), coverage=p["coverage"],
                                 ctrl_reproduces_rule=same,
                                 R2_in=p["R2_in"], ctrl_R2_mean=float(np.nanmean(d["R2"])), ctrl_R2_sd=float(np.nanstd(d["R2"])),
                                 ctrl_R2_q025=_q(d["R2"], CI_LO), ctrl_R2_q975=_q(d["R2"], CI_HI),
                                 ctrl_R2_max=float(np.nanmax(d["R2"])),
                                 P_ctrl_R2_ge_rule=NA_REPRO if same else float(np.nanmean(d["R2"] >= p["R2_in"])),
                                 excess_R2=p["R2_in"] - float(np.nanmean(d["R2"])), relMSE_in=p["relMSE_in"],
                                 ctrl_relMSE_mean=float(np.nanmean(d["relMSE"])), ctrl_relMSE_q025=_q(d["relMSE"], CI_LO),
                                 ctrl_relMSE_q975=_q(d["relMSE"], CI_HI),
                                 P_ctrl_relMSE_ge_rule=NA_REPRO if same else float(np.nanmean(d["relMSE"] >= p["relMSE_in"])),
                                 excess_relMSE=p["relMSE_in"] - float(np.nanmean(d["relMSE"]))))
            log(f"    {design} {var} control {c}: {len(ctrl_rules)} rules x {reps_control} draws ({time.time() - tc:.0f}s)")
    T["controls"] = pd.DataFrame(rows)
    # ---- tail flag (publication bootstrap) ----
    rows, TS = [], {}
    for var in DRAW_VARIANTS:
        v = V[var]
        pubs = np.sort(v.F.publication.astype(str).unique())
        pc = np.searchsorted(pubs, v.F.publication.astype(str).to_numpy())
        W = INF._draw_counts(len(pubs), reps_boot, boot_seed(design, var))
        for r in RULES:
            for cut in TAIL_CUTS:
                ts = tail_stats(v, v.m(r), cut, pc, len(pubs), W)
                TS[(var, r, cut)] = ts
                rows.append(_rec(design, "tail", r, var, label_for(r, var, design, "tail flag; 95 % publication bootstrap"),
                                 cut=cut, P_in=ts["P_in"], P_out=ts["P_out"], ratio=ts["ratio"], boot_lo=ts["boot_lo"],
                                 boot_hi=ts["boot_hi"], boot_P_le_1=ts["boot_P_le_1"], boot_n_inf=ts["boot_n_inf"],
                                 boot_n_nan=ts["boot_n_nan"], boot_B=ts["boot_B"], publications=len(pubs),
                                 boot_seed=boot_seed(design, var), k_in_per_seed=str(ts["k_in_per_seed"]),
                                 n_in_per_seed=str(ts["n_in_per_seed"]), k_out_per_seed=str(ts["k_out_per_seed"]),
                                 n_out_per_seed=str(ts["n_out_per_seed"])))
    T["tail"] = pd.DataFrame(rows)
    log(f"    {design} tail: {len(T['tail'])} rows ({time.time() - t0:.0f}s)")
    # ---- bulk precision (Mondrian q80, LOCO calibrated) ----
    rows, BS = [], {}
    for var in DRAW_VARIANTS:
        v = V[var]
        for r in MAIN_RULES:
            bs = bulk_stats(v, v.m(r))
            BS[(var, r)] = bs
            rows.append(_rec(design, "bulk", r, var, label_for(r, var, design, "bulk precision; q80 Mondrian, LOCO-calibrated"),
                             **bs))
    T["bulk"] = pd.DataFrame(rows)
    # ---- cheapest alternatives at matched coverage ----
    rows = []
    cdf = T["controls"]
    for var in DRAW_VARIANTS:
        v = V[var]
        s = v.F.s.to_numpy(float)
        for alt in ALT_RULES:
            ma = v.m(alt)
            mw = np.zeros(len(ma), bool)
            for ix in v.idx:                                # the window at the alternative's per-seed coverage
                n = int(ma[ix].sum())
                o = ix[np.lexsort((ix, s[ix]))]
                mw[o[:n]] = True
            sa, sw = mask_stats(v, ma), mask_stats(v, mw)
            ea, ew = expected_control(v, ma, "CELL"), expected_control(v, mw, "CELL")
            cr = cdf[(cdf.rule == alt) & (cdf.variant == var) & (cdf.control == "CELL")].iloc[0]
            rows.append(_rec(design, "alternatives", alt, var, label_for(alt, var, design, "cheapest alternative at matched coverage"),
                             alias=ALIAS.get(alt, ""), coverage=sa["coverage"], R2_in_alt=sa["R2_in"],
                             relMSE_in_alt=sa["relMSE_in"], R2_in_window_matched=sw["R2_in"],
                             relMSE_in_window_matched=sw["relMSE_in"], window_minus_alt_R2=sw["R2_in"] - sa["R2_in"],
                             window_minus_alt_relMSE=sw["relMSE_in"] - sa["relMSE_in"],
                             alt_CELL_ctrl_R2_mean=cr.ctrl_R2_mean, alt_CELL_ctrl_R2_q975=cr.ctrl_R2_q975,
                             alt_CELL_ctrl_reproduces_rule=bool(cr.ctrl_reproduces_rule),
                             alt_excess_relMSE_vs_CELL=sa["relMSE_in"] - ea["relMSE"],
                             window_matched_excess_relMSE_vs_CELL=sw["relMSE_in"] - ew["relMSE"]))
    T["alternatives"] = pd.DataFrame(rows)
    # ---- M39 optimism ledger with family nesting ----
    rows = []
    cur = T["curve"]
    nf = len(O)
    for var in DRAW_VARIANTS:
        base = cur[(cur.rule == "H2") & (cur.variant == var)].iloc[0]
        for r in ("H2", "NEST_HW", "NEST_FAM", "OP75", "OP90"):
            c = cur[(cur.rule == r) & (cur.variant == var)].iloc[0]
            rec = _rec(design, "optimism", r, var, label_for(r, var, design, "M39 optimism ledger"),
                       R2_in=c.R2_in, coverage=c.coverage, relMSE_in=c.relMSE_in, optimism_H2_minus_rule=base.R2_in - c.R2_in,
                       seen_before_round5=SEEN_VALUES.get(r, ""))
            if r in NEST_RULES:
                ch = O[f"choice_{r}"].astype(str)
                rec["fold_fits"] = nf
                rec["choice_counts"] = json.dumps(ch.value_counts().sort_index().to_dict())
                rec["share_choosing_H2"] = float((ch == "H2").mean())
                rec["share_choosing_level_window"] = float(ch.isin(LEVEL_WINDOWS).mean())
                rec["fold_fits_choosing_cell_level_rule"] = int(ch.isin(CELL_LEVEL_RULES).sum())
                rec["share_choosing_cell_level_rule"] = float(ch.isin(CELL_LEVEL_RULES).mean())
                rec["fold_fits_cell_constant_membership"] = (cell_constant_fold_fits(V[var].F, r) if var in V
                                                             else np.nan)
            rows.append(rec)
    T["optimism"] = pd.DataFrame(rows)
    # ---- unit table (cluster_cpg), tests, top-3 share ----
    urows, trows = [], []
    for var in DRAW_VARIANTS:
        v = V[var]
        for r in ("OP75", "H2", "NEST_FAM"):
            us = unit_frame(v, v.m(r))
            lab = label_for(r, var, design, "unit table (cluster_cpg); per-unit window vs CELL expectation")
            urows.append(us.reset_index().assign(design=design, rule=r, variant=var, label=lab))
            tr = unit_tests_row(us, f"WINDOW|units|{design}|{var}|{r}")
            trows.append(_rec(design, "unit_tests", r, var, label_for(r, var, design, "DIAGNOSTIC (unit tests; no in-corpus "
                                                                      "p-value is called significant)"), **tr))
    T["units"] = pd.concat(urows, ignore_index=True)
    T["unit_tests"] = pd.DataFrame(trows)
    # ---- pair-type table ----
    rows = []
    v = V["all"]
    for r in ("OP75", "H2"):
        m = v.m(r)
        big = np.abs(v.e) > 0.5
        for p, ix in v.F.groupby("pt", sort=False).indices.items():
            ms = m[ix]
            rows.append(_rec(design, "pairtypes", r, "all", label_for(r, "all", design, "per pair type"), pt=p,
                             rows_per_seed=len(ix) / v.S, coverage=float(ms.mean()),
                             MSE_in=float(v.e2[ix][ms].mean()) if ms.any() else math.nan,
                             MSE_out=float(v.e2[ix][~ms].mean()) if (~ms).any() else math.nan,
                             P05_in=float(big[ix][ms].mean()) if ms.any() else math.nan,
                             P05_out=float(big[ix][~ms].mean()) if (~ms).any() else math.nan))
    T["pairtypes"] = pd.DataFrame(rows)
    # ---- leave-one-publication-out ----
    rows = []
    if lopo:
        for var in DRAW_VARIANTS:
            F = V[var].F
            for r in ("OP75", "H2", "NEST_FAM"):
                for stat in LOPO_STATS:
                    lp = CM.lopo(F, lambda X, r=r, stat=stat: frame_stat(X, r, stat), by="publication")
                    rows.append(_rec(design, "lopo", r, var, label_for(r, var, design, "leave-one-publication-out "
                                                                       "jackknife, t(P - 1) 95 %"), stat=stat,
                                     **{k: lp[k] for k in ("point", "se", "lo", "hi", "P", "min", "min_by", "max", "max_by",
                                                           "n_undefined")}))
        log(f"    {design} lopo: {len(rows)} rows ({time.time() - t0:.0f}s)")
    T["lopo"] = pd.DataFrame(rows)
    # ---- leave-top-k-cells-out ----
    rows = []
    for k in TOPK:
        top = top_cells(V["all"].F, k)
        F = V["all"].F[~V["all"].F.cell_id.astype(str).isin(top)]
        for r in ("OP75", "H2", "NEST_FAM"):
            rows.append(_rec(design, "topk", r, f"top{k}", label_for(r, f"top{k}", design), k=k, cells_removed=";".join(top),
                             **{stat: frame_stat(F, r, stat) for stat in ("coverage",) + LOPO_STATS}))
    T["topk"] = pd.DataFrame(rows)
    # ---- C5-BTBP in / out and duplicate sensitivities (closed-form CELL expectation; point tail ratios) ----
    rows = []
    for var, v in V.items():
        for r in MAIN_RULES:
            F = v.F
            rows.append(_rec(design, "c5" if var in ("all", "noC5", "noC5cell") else "duplicates", r, var,
                             label_for(r, var, design), rows_per_seed=len(F) / v.S, cells=int(F.cell_id.nunique()),
                             **{stat: frame_stat(F, r, stat) for stat in ("coverage",) + LOPO_STATS}))
    allc = pd.DataFrame(rows)
    T["c5"] = allc[allc.table == "c5"].reset_index(drop=True)
    T["duplicates"] = allc[allc.table == "duplicates"].reset_index(drop=True)
    # ---- report rules (in-corpus CONSISTENCY CHECKs) ----
    T["report_rules"] = report_rules(T, design)
    log(f"    {design} analysis done ({time.time() - t0:.0f}s)")
    return T


def report_rules(T: dict, design: str) -> pd.DataFrame:
    """Row flag (NEST_FAM vs its CELL control q97.5), tail flag (OP75 primary; ratio >= 3 and lower bound > 1, with and
    without C5-BTBP) and bulk precision (OP75; >= 15 % narrowing).  Secondary evaluations at other rules are reported,
    never decisive."""
    lab = "CONSISTENCY CHECK (in-corpus report rule; thresholds near seen values; not a test)"
    if design == "BP":
        lab += "; SENSITIVITY (design BP)"
    ctrl, tail, bulk, opt = T["controls"], T["tail"], T["bulk"], T.get("optimism")
    rows = []
    for var in DRAW_VARIANTS:
        c = ctrl[(ctrl.rule == "NEST_FAM") & (ctrl.variant == var) & (ctrl.control == "CELL")].iloc[0]
        row_flag = bool(c.relMSE_in > c.ctrl_relMSE_q975)
        detail = (f"evaluated at NEST_FAM on inside relative SSE {c.relMSE_in:.4f} vs CELL q97.5 "
                  f"{c.ctrl_relMSE_q975:.4f}; inside R2 {c.R2_in:.4f} is "
                  f"{'below' if c.R2_in < c.ctrl_R2_mean else 'at or above'} the CELL R2 mean {c.ctrl_R2_mean:.4f} "
                  f"(q97.5 {c.ctrl_R2_q975:.4f})")
        if opt is not None and len(opt):
            o = opt[(opt.rule == "NEST_FAM") & (opt.variant == var)].iloc[0]
            detail += (f"; NEST_FAM chose a cell-level rule in {int(o.fold_fits_choosing_cell_level_rule)} of "
                       f"{int(o.fold_fits)} fold-fits, and its membership is constant within every (seed, cell) in "
                       f"{int(o.fold_fits_cell_constant_membership)} of {int(o.fold_fits)} (there its CELL control "
                       "reproduces the rule)")
        rows.append({"design": design, "rule_name": "row_flag", "evaluated_at": "NEST_FAM", "variant": var,
                     "primary": var == "all", "value": c.relMSE_in, "threshold": c.ctrl_relMSE_q975,
                     "detail": detail, "outcome": "ROW FLAG" if row_flag else "CELL OR CONDITION FLAG", "label": lab})
    for r in ("OP75", "OP90", "H2", "NEST_FAM"):
        res = {}
        for var in DRAW_VARIANTS:
            t5 = tail[(tail.rule == r) & (tail.variant == var) & (tail.cut == 0.5)].iloc[0]
            res[var] = (t5.ratio, t5.boot_lo, t5.boot_hi)
        ok = all(x[0] >= TAIL_RATIO_MIN and x[1] > 1.0 for x in res.values())
        rows.append({"design": design, "rule_name": "tail_flag", "evaluated_at": r, "variant": "all & noC5",
                     "primary": r == PRIMARY, "value": res["all"][0], "threshold": TAIL_RATIO_MIN,
                     "detail": "; ".join(f"{k}: ratio {x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]" for k, x in res.items()),
                     "outcome": "PASS" if ok else "FAIL", "label": lab})
    for r in MAIN_RULES:
        for var in DRAW_VARIANTS:
            b = bulk[(bulk.rule == r) & (bulk.variant == var)].iloc[0]
            ok = bool(b.narrowing_mean >= BULK_NARROWING_MIN)
            rows.append({"design": design, "rule_name": "bulk_precision", "evaluated_at": r, "variant": var,
                         "primary": r == PRIMARY and var == "all", "value": b.narrowing_mean, "threshold": BULK_NARROWING_MIN,
                         "detail": f"mean q80 half-width PTW {b.hw_ptw_mean:.4f} vs PT {b.hw_pt_mean:.4f}; median |res| in "
                                   f"{b.med_in:.4f} / all {b.med_all:.4f}",
                         "outcome": "PASS" if ok else "FAIL (expected)", "label": lab})
    return pd.DataFrame(rows)


def row_labels(R: pd.DataFrame, design: str) -> list:
    return [label_for("", "all", design, "row-level output (window axis; never E-A)")] * len(R)


def summary(T: dict, design: str, checks: dict, repro: dict, seconds: float, seal: str) -> dict:
    """Key numbers of one design (post-seal only); every number sits next to its label."""
    cur, ctrl, tail, bulk = T["curve"], T["controls"], T["tail"], T["bulk"]
    head = []
    for var in DRAW_VARIANTS:
        for r in MAIN_RULES + ("NOC5", "READMIN-2", "WHOLECELL2"):
            c = cur[(cur.rule == r) & (cur.variant == var)].iloc[0]
            cc = ctrl[(ctrl.rule == r) & (ctrl.variant == var) & (ctrl.control == "CELL")].iloc[0]
            t5 = tail[(tail.rule == r) & (tail.variant == var) & (tail.cut == 0.5)].iloc[0]
            head.append({"rule": r, "variant": var, "coverage": c.coverage, "R2_in": c.R2_in, "R2_out": c.R2_out,
                         "relMSE_in": c.relMSE_in, "CELL_R2_mean": cc.ctrl_R2_mean, "CELL_R2_q975": cc.ctrl_R2_q975,
                         "CELL_relMSE_q975": cc.ctrl_relMSE_q975, "CELL_ctrl_reproduces_rule": bool(cc.ctrl_reproduces_rule),
                         "excess_relMSE_vs_CELL_MC": cc.excess_relMSE,    # Monte Carlo; the closed form is on the curve
                         "tail05_ratio": t5.ratio, "tail05_boot": [t5.boot_lo, t5.boot_hi], "label": c.label})
    return {"label": SEEN + " (window axis, M09; descriptive; never E-A, never next to 0.452)", "item": ITEM_ID,
            "design": design, "seal": seal, "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "seconds": seconds, "repro": repro, "checks": checks, "headline": head,
            "report_rules": T["report_rules"].to_dict("records"), "optimism": T["optimism"].to_dict("records"),
            "unit_tests": T["unit_tests"].to_dict("records"), "seen_before_round5": SEEN_VALUES}


# ------------------------------------------------------------------------------------------------------------
# the frozen item specification
# ------------------------------------------------------------------------------------------------------------
def item_spec_text() -> str:
    seeds = ", ".join(f"{d}/{v}/{c}: {control_seed(d, v, c)}" for d in paths.DESIGNS for v in DRAW_VARIANTS for c in CONTROLS)
    bseeds = ", ".join(f"{d}/{v}: {boot_seed(d, v)}" for d in paths.DESIGNS for v in DRAW_VARIANTS)
    iseeds = ", ".join(f"{d}: {intervention_seed(d)}" for d in paths.DESIGNS)
    return f"""# Item 1.4 specification: M09 level window as a tail-risk reporting axis (window axis)

Frozen at the Stage-0 seal, before the item's first post-seal run (brief 1.2: Stage-0 seal -> item code seal -> reproduction run labelled TECHNICAL SMOKE -> arms; the pre-seal --smoke runs are PRE-SEAL ENGINEERING CHECKS, PROTOCOL 4 and 10).  Descriptive only: every in-corpus number is EXPLORATORY (seen), the grid was fixed after seeing V1, no in-corpus p-value is called significant, and no window value is placed next to E-A's 0.452 (brief 3.3).  Excluded from the confirmation package (the window as a row flag has power <= 0.15; brief 4.3 item 4, I2 confirm-1).  Code: `r5/window.py`, `scripts/s1_window.py`, `tests/test_window.py`.  Sources: brief 6.4, 6.0, 6.10, 8, 9; IR M09, M10, M39 (and `ideas/critiques.json` M09 / M39 test designs); EB 2.11 and Appendix 2; E2 1.6.

## Rows and levels

- Rows: round-4 `predictions/REF_<design>.parquet` through `r5.common.load_rows` (C3, k = 3 = four DOPT_Q reads; 1,802 queries per seed; dev seeds {", ".join(map(str, paths.DEV_SEEDS))}).  BPG decides; BP is a labelled SENSITIVITY on the same test folds.  Pooled within seed, then the seed mean.
- Residual res = y - B1 (B1 from the parquet).  B0 and HIST_D7 inside R2 are shown beside B1 on the curve table (HIST_D7 DIAGNOSTIC).
- Levels: `r4.refs.predict_levels` with u = beta at Gd, beta = the fold's stored round-4 amplitude (`round4_20261022/results/REF/fitted_<design>.json`, `ref:B1`), on the row's read set = union of the cached `seqs['DOPT_Q'][:3]` (asserted equal to the parquet's `read_set`); a read metal's level is its measured value.  s = max(|lev_A|, |lev_B|).  lev_A - lev_B must equal the parquet's B1 within 8.9e-16 on every row.

## Frozen rule list ("fixed after seeing V1")

| rule | inside (kept) iff |
|---|---|
| H1, H1.5, H2, H2.5, H3, Hinf | s <= h, h in {{1, 1.5, 2, 2.5, 3, inf}} (both levels in [-h, h]) |
| LO-2, LO-2.5, LO-3, LO-4 | min(lev_A, lev_B) >= L (LO-4 = "drop predicted min < -4") |
| UP2, UP2.5, UP3 | max(lev_A, lev_B) <= U |
| OP75, OP90 | s <= the label-free quantile of s (numpy `method='inverted_cdf'`) at coverage 0.75 / 0.90 over the outer fold's TRAINING rows |
| READMIN-2 | min of the four read values >= -2 (M10 reads-only flag) |
| READSIN2 | all four read values in [-2, 2] (M10 reads-only flag) |
| WHOLECELL2 | whole-cell rule: every row of a cell instance kept iff H2 keeps >= 50 % of that instance's rows |
| NOC5 | publication != pub_6a0ef8528c ("drop C5-BTBP"; metadata rule) |
| NEST_HW | per fold, the half-width rule (H1 .. Hinf) chosen on the training rows |
| NEST_FAM | per fold, the rule chosen on the training rows from the family = grid + OP75 + OP90 + READMIN-2 + READSIN2 + WHOLECELL2 |

- Training rows of a fold (operating points and nested choices; never a test row): every strict-adjacent query of the C3 training cells of the cached fold record (`r4.refs.training_cells`: inner-block records plus orphans), masked with the cell's nested DOPT_Q[:3] read set from that record, levels with the fold's beta, labels = training labels.
- Nested selection criterion: maximal inside R2 on the training rows among rules with training coverage >= 0.70; ties toward larger training coverage, then the registry order above; Hinf when none qualifies.
- Primary operating point: OP75.  OP90, H2 (the post hoc [-2, 2]) and NEST_FAM are reported beside it.

## Estimands and statistics

- Curve (every rule x variant): coverage, inside / outside / all R2, MSE inside / outside / all, inside relative SSE = 1 - MSE_in / MSE_all (per seed, then the seed mean), var(y) inside / all, per-seed values; the CELL closed-form expectation (exact E[SSE] = sum pi e^2 with pi the cell's kept share; plug-in R2) and the excess over it.
- (a) Tail flag: P_in(c), P_out(c) = seed means of the per-seed shares of rows with |res| > c, c in {{0.5, 0.3}}; ratio = P_out / P_in.  95 % two-sided percentile publication bootstrap, B = {REPS_BOOT} (`gen20k.inference._draw_counts` over the variant's publications; percentiles by `inverted_cdf`, inf kept), all rows and without C5-BTBP.
- (b) Bulk precision: median |res| inside / outside / all; q80 split-conformal half-width (the ceil((n + 1) 0.8)-th smallest |res|, inf when it exceeds n) per row, calibrated leave-one-cluster-out (cluster_cpg) within seed: pair-type-only (PT) and window-aware (pair type x inside / outside; PTW); reported on the inside rows: mean and median half-widths, narrowing = 1 - mean PTW / mean PT, empirical coverage, minimum per-cluster coverage, counts of infinite half-widths.
- Controls at matched counts (random abstention keeping the rule's number of rows per stratum): CELL = (seed, cell) [primary], PUBPT = (seed, publication, pair type), UNSTRAT = (seed); {REPS_CONTROL} draws each with common random numbers across the rules of one (design, variant, control); mean, sd, 2.5 / 97.5 % quantiles, max and the share of draws at or above the rule, for inside R2 and inside relative SSE.
- Cheapest alternatives at matched coverage: READMIN-2, READSIN2, WHOLECELL2, LO-4, NOC5, each against the s-ranked window at its own per-seed coverage (the n rows with the smallest s; ties by row order) and its own CELL control.
- Winsorised curves: |res| capped at 0.3 and 0.5 (SST unchanged) for the grid, OP and NEST rules.
- M39 optimism ledger with family nesting: H2 (post hoc), NEST_HW, NEST_FAM, OP75, OP90; optimism = R2_in(H2) - R2_in(rule); per-rule choice counts over the 25 fold-fits, the share choosing H2 and the share choosing a level window; seen values printed beside.
- Units (cluster_cpg, 13): rows, coverage, MSE inside / outside and their ratio, tail counts, dSSE = E[SSE_CELL] - SSE_window, L = log(E / SSE_window), rel = 1 - SSE_window / E, contribution c_u (sum = excess relative SSE over the CELL expectation); SF_POOL (exact over 2^13), SIGN, SF_LOG, IM_REL via `r5.common` (DIAGNOSTIC); top-3 share of c_u.  Rules OP75, H2, NEST_FAM.
- Leave-one-publication-out (`r5.common.lopo`, t(P - 1) 95 %) of inside R2, the excess relative SSE and plug-in excess R2 over the CELL expectation, and the |res| > 0.5 ratio (OP75, H2, NEST_FAM; all and without C5-BTBP).  Leave-top-k-cells-out, k in {{5, 10}} (cells ranked by B1 SSE over all rows; SENSITIVITY, outcome-selected).
- C5-BTBP in / out and cell c950dc2982e96b83 out; duplicate sensitivities noV2 (V2-1/2/3 cells, `r5.population.v2drop_set`) and noBITDUP (cell_b of every flagged link in `results/stage0/dupscan_links.csv`); per pair type coverage and MSE inside / outside.  The rules' memberships and thresholds are fixed on all training cells; a variant only restricts the scored rows.

## Report rules (in-corpus CONSISTENCY CHECKs; never tests; BPG decides, BP reported)

- Row flag: "row flag" only if NEST_FAM's inside relative SSE (all rows) exceeds the 97.5 % quantile of its CELL control; otherwise "cell or condition flag".  The noC5 evaluation is reported beside it.
- Tail flag: passes if, at OP75, the |res| > 0.5 ratio is >= 3 with a bootstrap lower bound > 1, both on all rows and without C5-BTBP.  OP90, H2 and NEST_FAM are reported as secondary evaluations.
- Bulk precision: passes if, at OP75 on all rows, the narrowing of the mean q80 half-width (PTW vs PT on the inside rows) is >= 15 %; expected to fail.
- Seen before round 5: {SEEN_VALUES['H2']}; {SEEN_VALUES['CELL']}; {SEEN_VALUES['NEST_FAM']}; {SEEN_VALUES['tail']}; {SEEN_VALUES['bulk']}; {SEEN_VALUES['NOC5']}.

## Reproduction targets (brief 6.0; TECHNICAL SMOKE; a mismatch stops the item)

- B1 from the fold betas equals `REF_<design>.parquet` within 8.9e-16 on every row (9,010 per design, 18,020 over both).
- EB Appendix 2 (window on HIST_D7-form levels in [-2, 2]): BPG coverage per seed within [0.743, 0.754], B1 inside R2 0.540 (per seed 0.532, 0.521, 0.546, 0.547, 0.553), BP inside R2 0.538; tolerance 0.0005 (quoted to 3 decimals).  The same window on B1-form levels (the brief's rule) is reported beside it, not gating.

## Leakage (brief 8)

- `g21.evaluate.assert_test_isolation` on every fold; training rows assert their cells lie in the outer training index and not in the test index.
- Support: read sets equal the cache and the parquet; read-metal levels equal the reads; no scored pair with both ends read; read flags match; strict adjacency; every query belongs to the cell's query set.
- Prediction intervention on every test cell instance: unread metals scrambled (`g21.evaluate.scramble_hidden`), levels bit-identical; positive control: a block-Y peek (every measured metal read) must be caught.  The pre-seal `--smoke` replaces y and every prediction column by seeded noise before any table runs (no real-label metric is computed; s0-mechanics-22); `--no-lopo` is smoke-only; the seal tag is written next to every output.
- Fit poisoning of the operating points and nested choices on every fold (`r4.leakage.poisoned_bench(bench, train_index, seed = split_seed * 10 + fold)`): thresholds, choices and training rows bit-identical; positive control: the same computation with the fold's test cells added must change.  Written to `results/leakage/poison_WINDOW_ops_<design>.json`.  The stored fold betas were poisoning-checked in round 4.

## RNG seeds (`r5.paths.seed_for(key)` = 20261023 + crc32(key))

- Controls: key `WINDOW|control|<design>|<variant>|<control>` -> {seeds}.
- Publication bootstrap: key `WINDOW|pubboot|<design>|<variant>` -> {bseeds}.
- Intervention scrambles: key `WINDOW|intervention|<design>` -> {iseeds}.
- Fit poisoning: split_seed * 10 + fold (round 4's).  Unit sign-flips: exact at G <= 13 (key `WINDOW|units|<design>|<variant>|<rule>|SF_POOL` otherwise).

## Outputs (`results/WINDOW/`)

`repro_<design>.json`; `rows_<design>.parquet` (row-level levels, s, reads, thresholds, choices and every rule's membership); `ops_<design>.csv` (per-fold thresholds, choices, training coverages and training R2 -- training-side); `checks_<design>.json`; one CSV per table `<table>_<design>.csv` (curve, winsor, controls, tail, bulk, alternatives, optimism, units, unit_tests, pairtypes, lopo, topk, c5, duplicates, report_rules), every row labelled; `summary_<design>.json`; `ITEM_TABLE.md` (the section-9.3 per-item table) when both designs ran.

## Command lines (run from D:\\ml_separator_gh)

```
set PYTHONIOENCODING=utf-8
rem pre-seal engineering check on real data (no metric printed or written; allowed before the seal)
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_window.py --smoke --design BPG
rem after the seal: reproduction run (TECHNICAL SMOKE), then the arms
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_window.py --repro --design both
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_window.py --design both
.venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_window.py -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
```
"""

"""E-DEC, selective direction calls (CLAUDE_CODE_PROMPT.md section 6.5; conventions 6.0, 6.10, 8, 9), with the E-REC
own-prior row (I2 recipe-3 salvage) and the E-RANK table (IR M15 salvage).  Stage-1 item 1.5.  Descriptive throughout:
no kill fires, nothing is called significant, a direction-call number is never an R2 gain and never sits next to E-A's
0.452.  Frozen arm list, report rules and seeds: results/EDEC/ITEM_SPEC.md.

Written from the definitions in IR M14 / M15 / M16 / M23 (ideas/critiques.json) and I2 recipe-3; the recipe-3 comparator
conventions (chemotype x pair-type majority by leave-own-publication-out with a pair-type fallback below 3 donor rows,
C5-BTBP never a donor; coverage-matched comparators; the three-way rule at 0.1) follow
    round5_design_20261023/ideas2/scripts/recipe3_skepB_signcheck.py
    LF SHA-256 c9ae42cc2b54ea3c6cb2ddb676b0eb4b76b3a5e5b5bfa5be631e0c4a3e2aebb6
(conventions re-implemented, no code copied; that script is read, never run).  The E-REC arms COPY / SH_r2 / SH_r4
come from r5.ect unchanged.

**Rows.**  E-A rows (r5.common.load_rows: round-4 REF OOF, C3, k = 3 = four DOPT_Q reads, 1,802 queries per seed, 13
cluster_cpg clusters).  A prediction p of log SF(A/B) (A the lighter metal) makes a three-way decision at a threshold
tau: lighter (p >= tau), heavier (p <= -tau), inseparable / measure (|p| < tau; p = 0 is never a call).  A **false
sign** is a call whose sign differs from y's with |y| >= 0.10; a call on a pair with |y| < 0.10 is never a false sign.
Selective (per-call) risk = false signs / calls; marginal (per-pair) risk = false signs / rows.
**Hull rule (primary).**  A pair may be called only inside the read hull: a read metal at or below A and one at or
above B (the bracketing rule, which also covers Tm/Yb and Yb/Lu).  Out-of-hull rows are never called and stay in the
coverage denominator; calling every row (no hull) is a sensitivity.
**Calibration (never on the rows it calls).**  Leave-one-cluster-out inside each seed: the threshold for cluster c uses
only the same seed's OOF rows of the other clusters (their hull-eligible rows under the hull rule).  Grid
tau in {3.000, 2.995, ..., 0.005}.  Global thresholds are primary; Mondrian thresholds (per pair type) secondary.
  LTT (primary)  Learn-then-Test: H0(tau): selective risk > alpha; p(tau) = BinomCDF(k(tau); n(tau), alpha) on the
                 calibration calls; fixed-sequence testing at delta = 0.10 from the largest tau with
                 n(tau) >= n_min = ceil(log delta / log(1 - alpha)) (a label-free start), rejecting while p <= delta;
                 tau_hat = the last rejected tau; no rejection -> no calls.
  CRC            conformal risk control on the monotone per-pair loss l_i(tau) = 1[called_i(tau)] 1[false sign_i]
                 (bounded by 1): tau_hat = min{tau: (K(tau) + 1) / (N + 1) <= alpha}; it controls the marginal
                 (per-pair) false-sign rate, not the per-call rate (the per-call ratio is not monotone: IR M14).
  PLUGIN         tau_hat = min{tau: n(tau) >= 1 and k(tau) / n(tau) <= alpha} (no guarantee).
  alpha 0.10 primary, 0.05 sensitivity.  Exchangeability across clusters is not given (Kish 2.24): the guarantees of
  LTT and CRC are stated for exchangeable rows only.
**Risk-coverage.**  Rows ordered by |p|; the expected selective false-sign risk after k calls (ties at equal |p| shared
as the expectation over random tie-breaking); AURC = its mean over k = 1..n; E-AURC = AURC - AURC of the oracle order
[O] (errors last).  Arms B1, B0, HBK@ML (primary set), RIDGE2, LINE, G14@ZERO (comparators), HIST_D7 (DIAGNOSTIC).
**Net benefit.**  Decisions "separable" (event |y| >= log10 2, SF >= 2) and "inseparable" (|y| < 0.10); P(event) by
Platt scaling of |p| (logistic in |p|, 2 parameters, Newton) fitted leave-one-cluster-out; NB(p_t) = TP/n - FP/n *
p_t / (1 - p_t) at p_t in {0.1, 0.2, 0.3, 0.5} against treat-all / treat-none; B1 against B0, HBK@ML, RIDGE2 and the
Platt-scaled LINE; calibration-in-the-large (observed - expected) per cluster.
**Interval line (M16).**  B1 split-conformal |residual| intervals, row-weighted, Mondrian by pair type; the nominal
level is chosen by nested leave-one-cluster-out (inner LOCO on the other clusters; the smallest nominal in
{0.80, 0.85, 0.90, ..., 0.99} whose pooled inner coverage reaches 0.90), then calibrated on every other cluster.
**One-sided bounds (M23).**  Rows with a read end at or below the user-stated limit L = -4.0: the read A end at the
limit gives y <= p + q_up, the read B end y >= p - q_dn (q = 0.95 one-sided split-conformal quantiles of B1's signed
residual on the other clusters).  Labelled "C5-BTBP floor sensitivity".
**Measurements saved.**  ICP-MS: 0 by construction (the 4-read experiment measured every metal); subset-feed /
radiotracer follow-up: cells with every scored pair called (follow-up experiment avoided) and unread pair-end metals
not needed; literature mining: pairs signed without a new measurement (= calls).
**E-RANK (M15).**  Candidate groups = (seed, fold, pair type) with >= 5 test cells (within-publication groups:
(seed, fold, pair type, publication) with >= 3); candidates ranked by |p|; endpoints top-1, top-3 hit, regret (log
units), Spearman(|p|, |y|); RANDOM = the exact expectation; AMP_ORACLE [O] ranks by the cell's mean |y| over its other
scored pairs.  LINE share = (LINE - RANDOM) / (B1 - RANDOM) per endpoint (regret: reversed), with a leave-one-cluster-out
range.  Never a claim.
**E-REC own-prior row (recipe-3).**  On the r5.ect rows (identical E-A queries): targets with >= 1 CLEAN, bit-rule
sibling (n_sib >= 1), C5-BTBP out, the quantised pub_a49146c8fd out.  Arms COPY_own (0 reads), SH_own@2, SH_own@4
(r5.ect SH_r2 / SH_r4) against CHPT_MAJ (median y of other publications' rows of the same chemotype and pair type,
>= 3 rows, else of the pair type; C5-BTBP never a donor), G14 (zero-shot), D7@2 / D7@4 (B1's form with the fixed
-0.072 at 2 / 4 reads) and B1@4, on identical rows: risk-coverage / AURC, risk at matched coverage (each arm its own
top-|p| rows; c_ref = the share with |COPY| >= 0.10, plus 0.25 / 0.50 / 0.75; boundary ties shared), the three-way rule
at 0.10 (three-class accuracy, opposite-direction call rate).  Strata: n_sib 1 / >= 2, condition-change type, DGA
(sc009) / non-DGA, giant cluster / rest, without pub_b0b3296ec5; sensitivities with C5-BTBP, with the quantised
publication, quantised targets out, P1's CLEAN rule without the bit rule (NB_ arms).  Unit: publication x chemotype.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
from scipy import stats as SS

from . import common as CM
from . import paths
from r4 import metrics as M4  # noqa: E402
from r4.refs import HIST_D7_BETA  # noqa: E402
from gen20k.data import LANTHANIDES  # noqa: E402

# Copy rule (brief 1.2): every copied or re-implemented source with its LF SHA-256 at the copy; checked by
# scripts/r5_seal.py (copied_sources reads this declaration).
COPIED_FROM = {
    "generations/gen21/round5_design_20261023/ideas2/scripts/recipe3_skepB_signcheck.py":
        "c9ae42cc2b54ea3c6cb2ddb676b0eb4b76b3a5e5b5bfa5be631e0c4a3e2aebb6",
}

SEEN, SENS, ORC, DIAG, CC, SMOKE = ("EXPLORATORY (seen)", "SENSITIVITY", "ORACLE [O]", "DIAGNOSTIC", "CONSISTENCY CHECK",
                                    "TECHNICAL SMOKE")
NOT_R2 = "E-DEC: a decision estimand; never an R2 gain, never next to E-A's 0.452"
OUT = paths.RES / "EDEC"
LN = tuple(LANTHANIDES)
PAIR_ORDER = ("La/Ce", "Ce/Pr", "Pr/Nd", "Sm/Eu", "Eu/Gd", "Gd/Tb", "Tb/Dy", "Dy/Ho", "Ho/Er", "Er/Tm", "Tm/Yb", "Yb/Lu")
Y_MIN = 0.10                                   # a false sign counts only when |y| >= 0.10
SEP = math.log10(2.0)                          # "separable": SF >= 2
ALPHAS = (0.10, 0.05)                          # 0.10 primary
ALPHA = 0.10
DELTA = 0.10                                   # LTT error level
GRID = np.round(np.arange(600, 0, -1) * 0.005, 3)    # 3.000 ... 0.005 (descending)
PROCS = ("LTT", "CRC", "PLUGIN")
ARMS = ("B1", "B0", "HBK@ML", "RIDGE2", "LINE", "G14@ZERO", "HIST_D7")
ROLE = {"B1": "primary set", "B0": "primary set (cheapest same-information comparator)", "HBK@ML": "primary set",
        "RIDGE2": "comparator", "LINE": "comparator (slope only)", "G14@ZERO": "comparator (zero-shot)",
        "HIST_D7": "DIAGNOSTIC (fixed -0.072; not fold-independent)"}
POISON_ARMS = ("B1", "B0", "HBK@ML")
N_CLUSTERS = 13                     # cluster_cpg clusters on E-A rows (round 4; METRICS section 3)
POISON_SCOPE = ("calibration labels only: the held-out cluster's calibration-frame labels are replaced; the prediction "
                "models (B1 fold beta, HBK@ML kernel) are not refitted, and their outer-fold training may include the "
                "held-out cluster's cells; B0 is training-free (no such channel) (s0-items-7)")
NB_ARMS = ("B1", "B0", "HBK@ML", "RIDGE2", "LINE")
NB_PT = (0.1, 0.2, 0.3, 0.5)
DECISIONS = ("separable_SF>=2", "inseparable_|y|<0.10")
RANK_ARMS = ("B1", "B0", "LINE", "RIDGE2", "HBK@ML", "G14@ZERO")
RANK_MIN, RANK_MIN_PUB = 5, 3
COV_GRID = tuple(float(x) for x in np.round(np.arange(1, 21) * 0.05, 2))
NOMINAL = (0.80, 0.85) + tuple(float(x) for x in np.round(np.arange(90, 100) / 100, 2))
TARGET_COV = 0.90
LIMIT = -4.0
BOUND_LEVEL = 0.95
PRIMARY = ("LTT", 0.10, True, False)           # procedure, alpha, hull, Mondrian
QUANT_PUB = "pub_a49146c8fd"
ACIDTYPE_PUB = "pub_b0b3296ec5"
DGA = M4.DGA_CHEMOTYPE
EREC_ARMS = {"COPY_own": "COPY", "SH_own@2": "SH_r2", "SH_own@4": "SH_r4", "CHPT_MAJ": "CHPT_MAJ", "G14": "G14Z",
             "D7@2": "D7_r2", "D7@4": "D7_r4", "B1@4": "B1_r4"}
EREC_OWN = ("COPY_own", "SH_own@2", "SH_own@4")
EREC_COV = (("c_ref", None), ("0.25", 0.25), ("0.50", 0.50), ("0.75", 0.75))
EREC_CONTRASTS = (("SH_own@4", "B1@4"), ("SH_own@2", "B1@4"), ("COPY_own", "B1@4"), ("SH_own@2", "D7@2"),
                  ("COPY_own", "CHPT_MAJ"), ("COPY_own", "D7@2"))
SEEN_VALUES = {
    "B1 coverage at alpha 0.10 (as stated / CRC / plug-in / valid fixed-sequence) [IR M14]": [0.606, 0.519, 0.507, 0.149],
    "B1 - B0 coverage [IR M14]": "+0.047 to +0.058",
    "E-RANK LINE shares: Spearman / top-3 / within publication [IR M15]": [0.61, 0.67, 0.66],
    "recipe-3 at matched coverage: COPY / D7@4 / B1@4 / SH_own@2 [I2 recipe-3]": [0.922, 0.919, 0.933, 0.949],
    "recipe-3 opposite-direction call rates: COPY / SH_own@2 / D7@2 [I2 recipe-3]": [0.011, 0.009, 0.023],
    "note": "seen during design with other implementations (HULL reads, other calibration); recorded, never a target",
}


def nanmean(x) -> float:
    a = np.asarray(list(x), float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def per_seed_str(v) -> str:
    return ";".join("NaN" if not np.isfinite(x) else f"{x:.4f}" for x in np.asarray(list(v), float))


def label(design: str, *extra) -> str:
    parts = [SEEN]
    if design == "BP":
        parts.append("SENSITIVITY (BP; BPG decides)")
    parts += [e for e in extra if e]
    return "; ".join(parts)


def cfg_name(proc: str, alpha: float, hull: bool, mond: bool) -> str:
    return f"{proc}_a{int(round(alpha * 100)):02d}_{'hull' if hull else 'all'}_{'mond' if mond else 'glob'}"


def cfg_label(cfg: str) -> str:
    proc, a, h, m = cfg.split("_")
    parts = []
    if cfg == cfg_name(*PRIMARY):
        parts.append("PRIMARY (LTT, alpha 0.10, hull rule, global thresholds)")
    elif proc != "LTT":
        parts.append(f"{proc} beside the primary" + (" (controls the per-pair rate)" if proc == "CRC" else " (no guarantee)"))
    if a != "a10":
        parts.append("SENSITIVITY (alpha 0.05)")
    if h == "all":
        parts.append("SENSITIVITY (no hull rule)")
    if m == "mond":
        parts.append("SECONDARY (Mondrian by pair type)")
    return "; ".join(parts)


def cfg_alpha(cfg: str) -> float:
    return int(cfg.split("_")[1][1:]) / 100.0


def configs(arms=ARMS):
    """Every (arm, procedure, alpha, hull, Mondrian); Mondrian runs under the hull rule only."""
    for arm in arms:
        for proc in PROCS:
            for a in ALPHAS:
                for hull, mond in ((True, False), (True, True), (False, False)):
                    yield arm, proc, a, hull, mond


# ------------------------------------------------------------------------------------------------------------
# rows
# ------------------------------------------------------------------------------------------------------------
def hull_flags(read_set, ia, ib) -> np.ndarray:
    """True where a read metal lies at or below A and one at or above B (label-free; from the read set only)."""
    rs = pd.Series(read_set).astype(str)
    lohi = {s: (min(LN.index(m) for m in s.split(",")), max(LN.index(m) for m in s.split(","))) for s in rs.unique()}
    lo = rs.map(lambda s: lohi[s][0]).to_numpy(int)
    hi = rs.map(lambda s: lohi[s][1]).to_numpy(int)
    return (lo <= np.asarray(ia, int)) & (hi >= np.asarray(ib, int))


def prepare(t: pd.DataFrame) -> pd.DataFrame:
    """E-A rows plus pair, hull flag and cluster as str; asserts the support contract (no pair with both ends read,
    strict adjacency, Nd/Sm absent)."""
    d = t.reset_index(drop=True).copy()
    d["pair"] = d.A.astype(str) + "/" + d.B.astype(str)
    d["cluster_cpg"] = d.cluster_cpg.astype(str)
    d["hull"] = hull_flags(d.read_set, d.ia, d.ib)
    if (d.a_read & d.b_read).any():
        raise AssertionError("a scored pair has both ends read")
    if not ((d.ib - d.ia) == 1).all() or (d.pair == "Nd/Sm").any() or not d.pair.isin(PAIR_ORDER).all():
        raise AssertionError("row set is not strict-adjacent")
    return d


def support_record(d: pd.DataFrame, va=None, vb=None, limit: float = LIMIT) -> dict:
    """Section 8 item 5, computed on the rows (not asserted): pairs with both ends read; rows that are not strict-adjacent
    (Nd/Sm, or not one of the 12 pair types); rows whose hull flag differs from a recount on the read sets alone (a
    cumulative count, independent of hull_flags); and, given the read-end values va / vb, the bound-line rows (a read
    end at or below ``limit``) whose bounded end is unread and the V1 values present at an unread end."""
    ia, ib = d.ia.to_numpy(int), d.ib.to_numpy(int)
    ar, br = d.a_read.to_numpy(bool), d.b_read.to_numpy(bool)
    pair = (d.A.astype(str) + "/" + d.B.astype(str)).to_numpy(str)
    rs = d.read_set.astype(str).to_numpy(str)
    sets = {s: np.isin(np.arange(len(LN)), [LN.index(m) for m in s.split(",")]) for s in set(rs)}
    R = np.stack([sets[s] for s in rs]) if len(d) else np.zeros((0, len(LN)), bool)
    i = np.arange(len(d))
    below = np.cumsum(R, 1)[i, ia] > 0                              # a read metal at or below A
    above = np.cumsum(R[:, ::-1], 1)[i, len(LN) - 1 - ib] > 0      # a read metal at or above B
    rec = {"rows": int(len(d)), "both_ends_read_rows": int((ar & br).sum()),
           "non_strict_adjacent_rows": int(((ib - ia != 1) | (pair == "Nd/Sm") | ~np.isin(pair, PAIR_ORDER)).sum()),
           "hull_recount_mismatch_rows": int((d.hull.to_numpy(bool) != (below & above)).sum())}
    rec["strict_adjacent"] = rec["non_strict_adjacent_rows"] == 0
    rec["hull_from_read_sets_only"] = rec["hull_recount_mismatch_rows"] == 0
    if va is not None:
        va, vb = np.asarray(va, float), np.asarray(vb, float)
        ca, cb = np.isfinite(va) & (va <= limit), np.isfinite(vb) & (vb <= limit)
        rec["bound_rows"] = int((ca | cb).sum())
        rec["bound_rows_unread_end"] = int(((ca & ~ar) | (~ca & cb & ~br)).sum())
        rec["v1_values_at_unread_ends"] = int((np.isfinite(va) & ~ar).sum() + (np.isfinite(vb) & ~br).sum())
    rec["ok"] = bool(rec["both_ends_read_rows"] == 0 and rec["strict_adjacent"] and rec["hull_from_read_sets_only"]
                     and rec.get("bound_rows_unread_end", 0) == 0 and rec.get("v1_values_at_unread_ends", 0) == 0)
    return rec


def read_values(d: pd.DataFrame, bench) -> tuple[np.ndarray, np.ndarray]:
    """V1 log D at the read end of each row (NaN at an unread end): the only V1 values beyond y the bound line uses."""
    f = bench.frame.reset_index(drop=True)
    pos = {c: i for i, c in enumerate(f["cell_id"].astype(str))}
    Y = np.asarray(bench.Y, float)
    ii = d.cell_id.astype(str).map(pos).to_numpy(int)
    va, vb = np.full(len(d), np.nan), np.full(len(d), np.nan)
    ma, mb = d.a_read.to_numpy(bool), d.b_read.to_numpy(bool)
    va[ma] = Y[ii[ma], d.ia.to_numpy(int)[ma]]
    vb[mb] = Y[ii[mb], d.ib.to_numpy(int)[mb]]
    if not (np.isfinite(va[ma]).all() and np.isfinite(vb[mb]).all()):
        raise AssertionError("a read end has no finite V1 value")
    return va, vb


def surrogate(d: pd.DataFrame, design: str, cols, key: str = "EDEC|SMOKE_SURROGATE") -> pd.DataFrame:
    """TECHNICAL SMOKE frame: y and every prediction column in ``cols`` replaced by seeded noise (NaN pattern kept)."""
    S = d.copy()
    rng = np.random.default_rng(paths.seed_for(f"{key}|{design}"))
    S["y"] = rng.normal(0.0, 0.3, len(S))
    for c in cols:
        if c in S.columns:
            v = S[c].to_numpy(float)
            S[c] = np.where(np.isfinite(v), S.y.to_numpy() + rng.normal(0.0, 0.2, len(S)), np.nan)
    return S


# ------------------------------------------------------------------------------------------------------------
# calls, calibration
# ------------------------------------------------------------------------------------------------------------
def score(p) -> np.ndarray:
    """|p| for a callable prediction, -1 for p = 0 or NaN (never called)."""
    p = np.asarray(p, float)
    return np.where(np.isfinite(p) & (p != 0), np.abs(p), -1.0)


def false_sign(p, y) -> np.ndarray:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return np.isfinite(p) & (p != 0) & (np.abs(y) >= Y_MIN) & (np.sign(p) != np.sign(y))


def wrong_sign(p, y) -> np.ndarray:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return np.isfinite(p) & (p != 0) & (np.sign(p) != np.sign(y))


def grid_counts(s, e, grid=GRID) -> tuple[np.ndarray, np.ndarray]:
    """n(tau) = #{s >= tau} and k(tau) = #{s >= tau and e} for every tau of the (descending) grid."""
    s = np.asarray(s, float)
    e = np.asarray(e, np.int64)
    o = np.argsort(s, kind="mergesort")
    ss = s[o]
    ce = np.concatenate([[0], np.cumsum(e[o])])
    idx = np.searchsorted(ss, grid, side="left")
    return len(s) - idx, ce[-1] - ce[idx]


def n_min(alpha: float, delta: float = DELTA) -> int:
    """The smallest call count at which zero false signs can reject H0: risk > alpha at level delta."""
    return int(math.ceil(math.log(delta) / math.log(1.0 - alpha) - 1e-12))


def calibrate(n, k, N: int, alpha: float, proc: str, grid=GRID, delta: float = DELTA) -> tuple[float, int]:
    """(tau_hat, grid index) of one procedure on calibration counts; (inf, -1) means no calls."""
    n, k = np.asarray(n), np.asarray(k)
    if proc == "LTT":
        start = np.flatnonzero(n >= n_min(alpha, delta))
        if not len(start):
            return math.inf, -1
        j0 = int(start[0])
        pv = SS.binom.cdf(k[j0:], n[j0:], alpha)
        fail = np.flatnonzero(pv > delta)
        jl = (int(fail[0]) if len(fail) else len(pv)) - 1
        return (float(grid[j0 + jl]), j0 + jl) if jl >= 0 else (math.inf, -1)
    if proc == "CRC":
        ok = (k + 1.0) / (N + 1.0) <= alpha
        bad = np.flatnonzero(~ok)
        j = (int(bad[0]) if len(bad) else len(ok)) - 1
        return (float(grid[j]), j) if j >= 0 else (math.inf, -1)
    if proc == "PLUGIN":
        ok = np.flatnonzero((n >= 1) & (k <= alpha * n))
        return (float(grid[ok[-1]]), int(ok[-1])) if len(ok) else (math.inf, -1)
    raise ValueError(proc)


def loco_taus(d: pd.DataFrame, arm: str, alpha: float, proc: str, *, hull: bool = True, mondrian: bool = False,
              leak: bool = False) -> pd.DataFrame:
    """Leave-one-cluster-out thresholds per (seed, cluster[, pair type]).  ``leak`` (positive control only) calibrates
    on every row of the seed, the held-out cluster included."""
    out = []
    for s, g in d.groupby("split_seed", sort=True):
        p, y = g[arm].to_numpy(float), g.y.to_numpy(float)
        sc, e = score(p), false_sign(p, y)
        el = g.hull.to_numpy(bool) if hull else np.ones(len(g), bool)
        cl, pt = g.cluster_cpg.to_numpy(str), g.pair.to_numpy(str)
        strata = [(q, pt == q) for q in PAIR_ORDER] if mondrian else [("ALL", np.ones(len(g), bool))]
        for q, qm in strata:
            if not qm.any():
                continue
            base = el & qm
            tn, tk = grid_counts(sc[base], e[base])
            N = int(base.sum())
            for c in np.unique(cl[qm]):
                m = base & (cl == c)
                if leak:
                    n, k, Nc = tn, tk, N
                else:
                    cn, ck = grid_counts(sc[m], e[m])
                    n, k, Nc = tn - cn, tk - ck, N - int(m.sum())
                tau, j = calibrate(n, k, Nc, alpha, proc)
                out.append((int(s), str(c), q, tau, Nc, int(n[j]) if j >= 0 else 0, int(k[j]) if j >= 0 else 0))
    return pd.DataFrame(out, columns=["split_seed", "cluster_cpg", "stratum", "tau", "n_cal_rows", "n_cal_calls", "k_cal_fs"])


def apply_calls(d: pd.DataFrame, arm: str, taus: pd.DataFrame, *, hull: bool, mondrian: bool) -> np.ndarray:
    tm = {(int(s), c, q): t for s, c, q, t in taus[["split_seed", "cluster_cpg", "stratum", "tau"]].itertuples(index=False)}
    q = d.pair.to_numpy(str) if mondrian else np.full(len(d), "ALL")
    tau = np.array([tm.get((int(s), c, qq), math.inf) for s, c, qq in zip(d.split_seed, d.cluster_cpg, q)], float)
    el = d.hull.to_numpy(bool) if hull else np.ones(len(d), bool)
    return el & (score(d[arm]) >= tau)


def run_selective(d: pd.DataFrame, cfgs=None) -> tuple[dict, pd.DataFrame]:
    """Calls {(arm, config): bool row array} and every fitted threshold."""
    calls, taus = {}, []
    for arm, proc, a, hull, mond in (cfgs if cfgs is not None else configs()):
        tt = loco_taus(d, arm, a, proc, hull=hull, mondrian=mond)
        name = cfg_name(proc, a, hull, mond)
        calls[(arm, name)] = apply_calls(d, arm, tt, hull=hull, mondrian=mond)
        taus.append(tt.assign(arm=arm, config=name, procedure=proc, alpha=a, hull=hull, mondrian=mond))
    return calls, pd.concat(taus, ignore_index=True)


def sel_seed(y, p, call, elig) -> dict:
    y, p, call, elig = (np.asarray(v) for v in (y, p, call, elig))
    fs = false_sign(p, y) & call
    ws = wrong_sign(p, y) & call
    sep = np.abs(y) >= Y_MIN
    n, nc, ne, nsc, nws = len(y), int(call.sum()), int(elig.sum()), int((call & sep).sum()), int(ws.sum())
    return {"rows": n, "eligible": ne, "calls": nc, "coverage": nc / n if n else np.nan,
            "coverage_eligible": nc / ne if ne else np.nan, "fs_rate": fs.sum() / nc if nc else np.nan,
            "marginal_fs": fs.sum() / n if n else np.nan, "fs_rate_sep_calls": fs.sum() / nsc if nsc else np.nan,
            "wrong_small_share": (ws & ~sep).sum() / nws if nws else np.nan,
            "sign_acc": 1.0 - nws / nc if nc else np.nan}


def sel_per_seed(d: pd.DataFrame, call: np.ndarray, arm: str, hull: bool = True) -> list:
    out = []
    for _, g in d.groupby("split_seed", sort=True):
        ix = g.index.to_numpy()
        el = g.hull.to_numpy(bool) if hull else np.ones(len(g), bool)
        out.append(sel_seed(g.y.to_numpy(float), g[arm].to_numpy(float), call[ix], el))
    return out


def sel_summary(per: list) -> dict:
    keys = per[0].keys() if per else []
    out = {k: nanmean(r[k] for r in per) for k in keys}
    out["coverage_per_seed"] = per_seed_str(r["coverage"] for r in per)
    out["fs_rate_per_seed"] = per_seed_str(r["fs_rate"] for r in per)
    return out


POOLED_KEYS = ("fs_pooled", "false_signs_pooled", "calls_pooled", "calls_distinct", "false_signs_distinct",
               "seeds_with_calls")


def pooled_counts(d: pd.DataFrame, call, arm: str, alpha: float = ALPHA) -> dict:
    """Counts beside a seed-mean per-call rate (descriptive; the seed mean of per-seed ratios stays the registered
    statistic, and a seed without calls drops out of it): false signs / calls pooled over the seeds, the distinct
    (cell, pair) queries behind them (the seeds share cells, so pooled counts repeat queries), the seeds with calls, and
    a flag when a seed has fewer calls than n_min(alpha)."""
    call = np.asarray(call, bool)
    fs = false_sign(d[arm], d.y) & call
    per = pd.Series(call).groupby(d.split_seed.to_numpy(), sort=True).sum().astype(int)
    q = (d.cell_id.astype(str) + "|" + d.A.astype(str) + "/" + d.B.astype(str)).to_numpy(str)
    nm, n, k = n_min(alpha), int(call.sum()), int(fs.sum())
    low, S = int((per < nm).sum()), int(len(per))
    cps = ";".join(str(int(v)) for v in per)
    flag = (f"seed mean rests on few calls (per-seed calls {cps}, below n_min {nm} in {low} of {S} seeds"
            + (", seeds without calls dropped)" if (per == 0).any() else ")")) if low else ""
    return {"fs_pooled": k / n if n else np.nan, "false_signs_pooled": k, "calls_pooled": n,
            "calls_distinct": int(len(np.unique(q[call]))), "false_signs_distinct": int(len(np.unique(q[fs]))),
            "seeds_with_calls": int((per > 0).sum()), "seeds": S, "calls_per_seed": cps, "n_min": nm, "few_calls": flag}


def counts_str(pc: dict) -> str:
    """One-cell form of pooled_counts."""
    if not pc["calls_pooled"]:
        return f"no calls in {pc['seeds']} seeds"
    s = (f"{pc['false_signs_pooled']}/{pc['calls_pooled']} = {pc['fs_pooled']:.4f} pooled, {pc['calls_distinct']} "
         f"distinct queries ({pc['false_signs_distinct']} with a false sign), calls in {pc['seeds_with_calls']} of "
         f"{pc['seeds']} seeds")
    return s + (f"; {pc['few_calls']}" if pc["few_calls"] else "")


# ------------------------------------------------------------------------------------------------------------
# risk-coverage
# ------------------------------------------------------------------------------------------------------------
def rc_expected(sc, e) -> np.ndarray:
    """Expected selective risk after k = 1..n calls (rows with sc < 0 never called; ties shared)."""
    sc, e = np.asarray(sc, float), np.asarray(e, float)
    ok = sc >= 0
    s, ee = sc[ok], e[ok]
    n = len(s)
    if n == 0:
        return np.zeros(0)
    o = np.argsort(-s, kind="mergesort")
    s, ee = s[o], ee[o]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    ends = np.r_[starts[1:], n]
    size = ends - starts
    grp = np.repeat(np.arange(len(starts)), size)
    gsum = np.add.reduceat(ee, starts)
    before = np.r_[0.0, np.cumsum(gsum)][:-1]
    k = np.arange(1, n + 1)
    return (before[grp] + (k - starts[grp]) * (gsum[grp] / size[grp])) / k


def aurc(sc, e) -> float:
    r = rc_expected(sc, e)
    return float(r.mean()) if len(r) else float("nan")


def aurc_oracle(sc, e) -> float:
    """[O] the same rows with every false sign called last."""
    ok = np.asarray(sc, float) >= 0
    ee = np.asarray(e, float)[ok]
    n, E = len(ee), float(ee.sum())
    if n == 0:
        return float("nan")
    k = np.arange(1, n + 1)
    return float((np.maximum(0.0, k - (n - E)) / k).mean())


def topm_weights(sc, m: int) -> np.ndarray:
    """Inclusion weights of the top-m rows by score; ties at the boundary share the remaining mass (the expectation
    over random tie-breaking).  Uncallable rows (sc < 0) are never included."""
    sc = np.asarray(sc, float)
    m = int(min(max(m, 0), int((sc >= 0).sum())))
    w = np.zeros(len(sc))
    if m == 0:
        return w
    t = np.sort(sc)[::-1][m - 1]
    above, at = sc > t, sc == t
    w[above] = 1.0
    w[at] = (m - above.sum()) / at.sum()
    return w


def cls3(v, thr: float = Y_MIN) -> np.ndarray:
    v = np.asarray(v, float)
    return np.where(v >= thr, 1, np.where(v <= -thr, -1, 0))


# ------------------------------------------------------------------------------------------------------------
# Platt scaling, net benefit
# ------------------------------------------------------------------------------------------------------------
def platt_fit(z, ev, ridge: float = 1e-6, iters: int = 100) -> np.ndarray:
    """(a, b) of P(event) = 1 / (1 + exp(-(a + b z))) by Newton's method (tiny ridge for stability)."""
    z, ev = np.asarray(z, float), np.asarray(ev, float)
    X = np.column_stack([np.ones_like(z), z])
    w = np.zeros(2)
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -35, 35)))
        H = X.T @ (X * (mu * (1 - mu))[:, None]) + ridge * np.eye(2)
        step = np.linalg.solve(H, X.T @ (ev - mu) - ridge * w)
        w = w + step
        if np.max(np.abs(step)) < 1e-12:
            break
    return w


def platt_predict(w, z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(w[0] + w[1] * np.asarray(z, float), -35, 35)))


def event(y, decision: str) -> np.ndarray:
    y = np.abs(np.asarray(y, float))
    return y >= SEP if decision == DECISIONS[0] else y < Y_MIN


def platt_loco(d: pd.DataFrame, arm: str, decision: str, *, leak: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """P(event) per row from Platt scaling of |p| fitted on the same seed's other clusters."""
    pi = np.full(len(d), np.nan)
    par = []
    for s, g in d.groupby("split_seed", sort=True):
        z, ev, cl = np.abs(g[arm].to_numpy(float)), event(g.y, decision), g.cluster_cpg.to_numpy(str)
        ix = g.index.to_numpy()
        for c in np.unique(cl):
            tr = np.ones(len(g), bool) if leak else cl != c
            w = platt_fit(z[tr], ev[tr])
            pi[ix[cl == c]] = platt_predict(w, z[cl == c])
            par.append((int(s), str(c), arm, decision, float(w[0]), float(w[1]), int(tr.sum())))
    return pi, pd.DataFrame(par, columns=["split_seed", "cluster_cpg", "arm", "decision", "a", "b", "n_fit"])


def net_benefit(pi, ev, pt: float) -> float:
    pi, ev = np.asarray(pi, float), np.asarray(ev, bool)
    treat = pi >= pt
    return float(((treat & ev).sum() - (treat & ~ev).sum() * pt / (1 - pt)) / len(ev))


# ------------------------------------------------------------------------------------------------------------
# interval line (M16) and one-sided bounds (M23)
# ------------------------------------------------------------------------------------------------------------
def conf_rank(n: int, level: float) -> int:
    """1-based rank of the split-conformal quantile: ceil((n + 1) level); > n means an infinite bound."""
    return int(math.ceil(round((n + 1) * level, 9)))


def conf_quantiles(scores, levels) -> np.ndarray:
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    out = np.full(len(levels), math.inf)
    for i, lv in enumerate(levels):
        k = conf_rank(n, lv)
        if 1 <= k <= n:
            out[i] = s[k - 1]
    return out


def interval_loco(d: pd.DataFrame, arm: str = "B1") -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Half-width and coverage per row of B1's Mondrian (pair type) row-weighted split-conformal interval, the nominal
    level chosen by nested leave-one-cluster-out (smallest nominal with pooled inner coverage >= 0.90)."""
    hw = np.full(len(d), np.nan)
    choice = []
    lv = np.asarray(NOMINAL, float)
    for s, g in d.groupby("split_seed", sort=True):
        r = np.abs(g.y.to_numpy(float) - g[arm].to_numpy(float))
        cl, pt, ix = g.cluster_cpg.to_numpy(str), g.pair.to_numpy(str), g.index.to_numpy()
        clusters = np.unique(cl)
        for c in clusters:
            hit, tot = np.zeros(len(lv)), 0
            for c2 in clusters:
                if c2 == c:
                    continue
                cal = (cl != c) & (cl != c2)
                for q in np.unique(pt[cl == c2]):
                    rows = (cl == c2) & (pt == q)
                    qq = conf_quantiles(r[cal & (pt == q)], lv)
                    hit += (r[rows][:, None] <= qq[None, :]).sum(0)
                    tot += int(rows.sum())
            cov_in = hit / tot if tot else np.full(len(lv), np.nan)
            okk = np.flatnonzero(cov_in >= TARGET_COV)
            j = int(okk[0]) if len(okk) else len(lv) - 1
            for q in np.unique(pt[cl == c]):
                rows = (cl == c) & (pt == q)
                hw[ix[rows]] = conf_quantiles(r[(cl != c) & (pt == q)], [lv[j]])[0]
            choice.append((int(s), str(c), float(lv[j]), float(cov_in[j]) if tot else np.nan, tot))
    cov = np.abs(d.y.to_numpy(float) - d[arm].to_numpy(float)) <= hw
    return hw, cov, pd.DataFrame(choice, columns=["split_seed", "cluster_cpg", "nominal_chosen", "inner_coverage", "inner_rows"])


def bounds_loco(d: pd.DataFrame, va, vb, arm: str = "B1", limit: float = LIMIT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-sided bounds for rows whose read end is at or below ``limit``; quantiles of the signed residual from the
    same seed's other clusters (every row there)."""
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    cens_a = np.isfinite(va) & (va <= limit)
    cens_b = np.isfinite(vb) & (vb <= limit)
    rows, qs = [], []
    for s, g in d.groupby("split_seed", sort=True):
        ix = g.index.to_numpy()
        res = g.y.to_numpy(float) - g[arm].to_numpy(float)
        cl = g.cluster_cpg.to_numpy(str)
        for c in np.unique(cl):
            cal = cl != c
            qu = conf_quantiles(res[cal], [BOUND_LEVEL])[0]
            qd = conf_quantiles(-res[cal], [BOUND_LEVEL])[0]
            qs.append((int(s), str(c), float(qu), float(qd), int(cal.sum())))
            for i in ix[cl == c]:
                if cens_a[i] or cens_b[i]:
                    p, y = float(d.at[i, arm]), float(d.at[i, "y"])
                    side = "upper (A read at or below L)" if cens_a[i] else "lower (B read at or below L)"
                    bound = p + qu if cens_a[i] else p - qd
                    rows.append((int(s), str(c), d.at[i, "cell_id"], d.at[i, "pair"], str(d.at[i, "publication"]), side,
                                 bound, bool(y <= bound) if cens_a[i] else bool(y >= bound), qu if cens_a[i] else qd))
    R = pd.DataFrame(rows, columns=["split_seed", "cluster_cpg", "cell_id", "pair", "publication", "side", "bound",
                                    "covered", "offset"])
    return R, pd.DataFrame(qs, columns=["split_seed", "cluster_cpg", "q_up", "q_dn", "n_cal"])


# ------------------------------------------------------------------------------------------------------------
# measurements saved
# ------------------------------------------------------------------------------------------------------------
def savings_seed(g: pd.DataFrame, call: np.ndarray) -> dict:
    """One seed: cells whose every scored pair is called; unread pair-end metals still needed for the uncalled pairs."""
    cells_all, n_need, n_base, n_calls = 0, 0, 0, int(call.sum())
    pos = {i: k for k, i in enumerate(g.index.to_numpy())}
    for _, h in g.groupby("cell_id", sort=False):
        reads = set(str(h.read_set.iloc[0]).split(","))
        ch = np.array([call[pos[i]] for i in h.index], bool)
        ends = set(h.A.astype(str)) | set(h.B.astype(str))
        unc = h[~ch]
        need = (set(unc.A.astype(str)) | set(unc.B.astype(str))) - reads
        n_base += len(ends - reads)
        n_need += len(need)
        cells_all += int(ch.all())
    ncell = g.cell_id.nunique()
    return {"cells": ncell, "cells_fully_called": cells_all, "share_cells_fully_called": cells_all / ncell,
            "unread_pair_end_metals": n_base, "metals_still_needed": n_need,
            "share_metal_measurements_saved": 1.0 - n_need / n_base if n_base else np.nan, "pairs_signed": n_calls}


# ------------------------------------------------------------------------------------------------------------
# E-RANK
# ------------------------------------------------------------------------------------------------------------
def _spearman(a, v) -> float:
    ra, rv = SS.rankdata(a), SS.rankdata(v)
    if ra.std() == 0 or rv.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rv)[0, 1])


def rank_endpoints(a, v) -> dict:
    """top-1 hit, top-3 hit, regret and Spearman for one candidate group (a = arm score, v = |y|); ties at the arm's
    top shared (expectation over random tie-breaking)."""
    a, v = np.asarray(a, float), np.asarray(v, float)
    best = v == v.max()
    top = a == a.max()
    fin = np.isfinite(a)
    w3 = topm_weights(np.where(fin, a - (a[fin].min() if fin.any() else 0.0), -1.0), 3)
    return {"top1": float((top & best).sum() / top.sum()), "top3": float(w3[best].sum() / best.sum()),
            "regret": float(v.max() - v[top].mean()), "spearman": _spearman(a, v)}


def rank_random(v) -> dict:
    v = np.asarray(v, float)
    n, best = len(v), v == v.max()
    return {"top1": float(best.sum() / n), "top3": float(min(3, n) / n), "regret": float(v.max() - v.mean()),
            "spearman": 0.0}


def amp_oracle_score(d: pd.DataFrame) -> np.ndarray:
    """[O] the cell's mean |y| over its other scored pairs in the same seed (-inf when it has no other pair)."""
    ay = d.y.abs()
    key = [d.split_seed, d.cell_id]
    S, C = ay.groupby(key).transform("sum"), ay.groupby(key).transform("size")
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = ((S - ay) / (C - 1)).to_numpy(float)
    return np.where(C.to_numpy() > 1, sc, -np.inf)


def rank_table_seed(d: pd.DataFrame, by, nmin: int) -> dict:
    """Per arm: seed means of the group-mean endpoints; also group counts."""
    d = d.assign(_amp=amp_oracle_score(d))
    arms = list(RANK_ARMS) + ["RANDOM", "AMP_ORACLE"]
    per_seed = {a: [] for a in arms}
    ng, ncand = [], []
    for s, gs in d.groupby("split_seed", sort=True):
        acc = {a: [] for a in arms}
        for _, g in gs.groupby(by, sort=True):
            if len(g) < nmin:
                continue
            v = g.y.abs().to_numpy(float)
            for a in RANK_ARMS:
                acc[a].append(rank_endpoints(np.abs(g[a].to_numpy(float)), v))
            acc["RANDOM"].append(rank_random(v))
            acc["AMP_ORACLE"].append(rank_endpoints(g._amp.to_numpy(float), v))
            ncand.append(len(g))
        ng.append(len(acc["RANDOM"]))
        for a in arms:
            per_seed[a].append({k: nanmean(x[k] for x in acc[a]) for k in ("top1", "top3", "regret", "spearman")})
    return {"per_seed": per_seed, "groups_per_seed": float(np.mean(ng)) if ng else 0.0,
            "median_candidates": float(np.median(ncand)) if ncand else float("nan")}


def line_shares(per_seed: dict) -> dict:
    out = {}
    for k in ("top1", "top3", "regret", "spearman"):
        b1, ln, rd = (nanmean(x[k] for x in per_seed[a]) for a in ("B1", "LINE", "RANDOM"))
        num, den = (rd - ln, rd - b1) if k == "regret" else (ln - rd, b1 - rd)
        out[k] = float(num / den) if den != 0 else float("nan")
    return out


# ------------------------------------------------------------------------------------------------------------
# E-REC own-prior row
# ------------------------------------------------------------------------------------------------------------
def fold_betas(design: str) -> dict:
    """{(seed, fold): beta} of round 4's B1 (results/REF/fitted_<design>.json), as r5.ect reads them."""
    fitted = json.loads((paths.R4RES / "REF" / f"fitted_{design}.json").read_text(encoding="utf-8"))
    out = {tuple(int(x) for x in k.split("_")): float(v["ref:B1"]["coef"][0]) for k, v in fitted.items()
           if re.fullmatch(r"\d+_\d+", k)}
    if len(out) != 25:
        raise AssertionError(f"{design}: expected 25 fold betas, found {len(out)}")
    return out


def d7_from_b1(b0, b1, beta, beta_d7: float = HIST_D7_BETA) -> np.ndarray:
    """B1's form with the fixed D7 amplitude on the same reads: B1 = B0 + beta h (h the Gd template on the read set),
    so D7 = B0 + (beta_d7 / beta)(B1 - B0) exactly; refuses a zero beta."""
    beta = np.asarray(beta, float)
    if np.any(np.abs(beta) < 1e-9):
        raise ValueError("beta = 0: the Gd template is not identified from B1 - B0")
    b0, b1 = np.asarray(b0, float), np.asarray(b1, float)
    return b0 + (beta_d7 / beta) * (b1 - b0)


def chpt_majority(d: pd.DataFrame, min_rows: int = 3) -> dict:
    """CHPT_MAJ for every (publication, chemotype, pair) of ``d``: {key: (median y, source, donor rows)} over the rows of
    OTHER publications with the same chemotype and pair type (>= min_rows rows), else of the pair type; C5-BTBP is never
    a donor.  Donor rows: one per (cell, pair) (the first seed)."""
    u = d[d.split_seed == d.split_seed.min()]
    don = u[u.publication.astype(str) != paths.C5]
    out = {}
    for (pub, chem, q), _ in d.groupby(["publication", "chemotype", "pair"], sort=True):
        o = don[(don.pair == q) & (don.publication.astype(str) != str(pub))]
        oc = o[o.chemotype.astype(str) == str(chem)]
        if len(oc) >= min_rows:
            out[(str(pub), str(chem), q)] = (float(oc.y.median()), "chemotype x pair type", len(oc))
        else:
            out[(str(pub), str(chem), q)] = (float(o.y.median()) if len(o) else np.nan, "pair type", len(o))
    return out


def erec_frame(E: pd.DataFrame, d: pd.DataFrame, design: str) -> tuple[pd.DataFrame, dict]:
    """The r5.ect rows of one design plus D7_r2 / D7_r4 (fixed -0.072), CHPT_MAJ, pair, DGA and the identity checks
    (B1_r4 = REF B1, D7_r4 = REF HIST_D7, y and row keys = the E-A rows)."""
    key = ["split_seed", "fold", "cell_id", "A", "B"]
    F = E.sort_values(key).reset_index(drop=True).copy()
    R = d.sort_values(key).reset_index(drop=True)
    if len(F) != len(R) or not (F[key].astype(str).to_numpy() == R[key].astype(str).to_numpy()).all():
        raise AssertionError("E-CT rows and E-A rows differ")
    betas = fold_betas(design)
    beta = np.array([betas[(int(s), int(f))] for s, f in zip(F.split_seed, F.fold)], float)
    F["beta_fold"] = beta
    F["D7_r2"] = d7_from_b1(F.B0_r2, F.B1_r2, beta)
    F["D7_r4"] = d7_from_b1(F.B0_r4, F.B1_r4, beta)
    F["pair"] = F.A.astype(str) + "/" + F.B.astype(str)
    F["cluster_cpg"] = R.cluster_cpg.astype(str).to_numpy()
    F["pubchem"] = R.pubchem.astype(str).to_numpy()
    F["dga"] = F.chemotype.astype(str) == DGA
    maj = chpt_majority(R)
    k3 = list(zip(F.publication.astype(str), F.chemotype.astype(str), F.pair))
    F["CHPT_MAJ"] = [maj[k][0] for k in k3]
    F["chpt_source"] = [maj[k][1] for k in k3]
    F["chpt_donor_rows"] = [maj[k][2] for k in k3]
    checks = {"rows": int(len(F)), "b1_r4_vs_ref_maxabs": float(np.max(np.abs(F.B1_r4 - R.B1))),
              "d7_r4_vs_ref_hist_d7_maxabs": float(np.max(np.abs(F.D7_r4 - R.HIST_D7))),
              "y_vs_ref_maxabs": float(np.max(np.abs(F.y - R.y))), "beta_min_abs": float(np.min(np.abs(beta)))}
    return F, checks


def erec_strata(F: pd.DataFrame) -> dict:
    """{name: (row mask, arm prefix for the own-series arms, label)}."""
    nq = F.publication.astype(str) != QUANT_PUB
    ge1 = F.in_GE1.to_numpy(bool) & nq.to_numpy()
    P = F[ge1]
    first = P[P.split_seed == P.split_seed.min()] if len(P) else P
    giant = str(first.groupby("cluster_cpg").size().idxmax()) if len(first) else ""
    S = {"PRIMARY_OWN": (ge1, "", "PRIMARY (n_sib >= 1, bit rule, C5-BTBP out, pub_a49146c8fd out)"),
         "WITH_C5": (F.in_GE1_WITH_C5.to_numpy(bool) & nq.to_numpy(), "", SENS + " (C5-BTBP in)"),
         "WITH_QUANT_PUB": (F.in_GE1.to_numpy(bool), "", SENS + " (pub_a49146c8fd in)"),
         "QUANT_TARGETS_OUT": (ge1 & ~F.t_quant.to_numpy(bool), "", SENS + " (every quantised target out)"),
         "NOBIT": (F.in_GE1_NOBIT.to_numpy(bool) & nq.to_numpy(), "NB_", SENS + " (duplicates: P1's CLEAN rule without the bit rule)"),
         "NSIB_1": (ge1 & (F.n_sib.to_numpy() == 1), "", "stratum n_sib = 1"),
         "NSIB_GE2": (ge1 & (F.n_sib.to_numpy() >= 2), "", "stratum n_sib >= 2"),
         "DGA": (ge1 & F.dga.to_numpy(bool), "", "stratum DGA (sc009)"),
         "NON_DGA": (ge1 & ~F.dga.to_numpy(bool), "", "stratum non-DGA"),
         f"GIANT_{giant}": (ge1 & (F.cluster_cpg.to_numpy(str) == giant), "", "stratum giant cluster"),
         "REST_CLUSTERS": (ge1 & (F.cluster_cpg.to_numpy(str) != giant), "", "stratum outside the giant cluster"),
         f"WITHOUT_{ACIDTYPE_PUB}": (ge1 & (F.publication.astype(str) != ACIDTYPE_PUB).to_numpy(), "",
                                     "influence (recipe-1 gate publication out)")}
    for c in sorted(set(F.loc[ge1, "cls"].astype(str))):
        S[f"CLS_{c or 'NONE'}"] = (ge1 & (F.cls.astype(str).to_numpy() == c), "", "stratum condition-change type")
    return S


def erec_cols(prefix: str) -> dict:
    return {a: (prefix + c if (a in EREC_OWN and prefix) else c) for a, c in EREC_ARMS.items()}


def erec_seed(y, P: dict, copy) -> dict:
    """One seed of one stratum (identical rows): per arm AURC, E-AURC, risk at matched coverage, the 0.10 rule."""
    y = np.asarray(y, float)
    n = len(y)
    c_ref = float((np.abs(np.asarray(copy, float)) >= Y_MIN).mean()) if n else np.nan
    out = {"rows": n, "c_ref": c_ref}
    for a, p in P.items():
        sc, fs, ws = score(p), false_sign(p, y), wrong_sign(p, y)
        r = {"AURC": aurc(sc, fs), "E_AURC": aurc(sc, fs) - aurc_oracle(sc, fs)}
        for cname, c in EREC_COV:
            cc = c_ref if c is None else c
            w = topm_weights(sc, int(round(cc * n)))
            sw = w.sum()
            r[f"fs@{cname}"] = float((w * fs).sum() / sw) if sw else np.nan
            r[f"sign_acc@{cname}"] = float(1 - (w * ws).sum() / sw) if sw else np.nan
        c3p, c3y = cls3(p), cls3(y)
        call01 = c3p != 0
        r["rule01_coverage"] = float(call01.mean())
        r["rule01_fs"] = float((fs & call01).sum() / call01.sum()) if call01.any() else np.nan
        r["three_way_acc"] = float((c3p == c3y).mean())
        r["opposite_call_rate"] = float((c3p * c3y == -1).mean())
        out[a] = r
    return out


def erec_unit_frame(S: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Per publication x chemotype unit: seed means of matched-coverage (c_ref) calls and false signs per arm."""
    rec = []
    for s, g in S.groupby("split_seed", sort=True):
        y = g.y.to_numpy(float)
        n = len(g)
        c_ref = float((np.abs(g[cols["COPY_own"]].to_numpy(float)) >= Y_MIN).mean())
        W = {}
        for a, c in cols.items():
            p = g[c].to_numpy(float)
            w = topm_weights(score(p), int(round(c_ref * n)))
            W[a] = (w, w * false_sign(p, y))
        for u, h in g.groupby("pubchem", sort=True):
            ii = np.flatnonzero(g.pubchem.to_numpy(str) == u)
            r = {"split_seed": int(s), "unit": u, "rows": len(h)}
            for a, (w, wf) in W.items():
                r[f"calls_{a}"], r[f"fs_{a}"] = float(w[ii].sum()), float(wf[ii].sum())
            rec.append(r)
    U = pd.DataFrame(rec)
    return U.groupby("unit").mean(numeric_only=True).drop(columns="split_seed")


# ------------------------------------------------------------------------------------------------------------
# leakage: fit poisoning of every calibrated quantity (section 8 item 2)
# ------------------------------------------------------------------------------------------------------------
def fitted_quantities(d: pd.DataFrame, cfgs, *, leak: bool = False) -> dict:
    taus = []
    for arm, proc, a, hull, mond in cfgs:
        taus.append(loco_taus(d, arm, a, proc, hull=hull, mondrian=mond, leak=leak).assign(
            config=f"{arm}|{cfg_name(proc, a, hull, mond)}"))
    plat = pd.concat([platt_loco(d, arm, dec, leak=leak)[1] for arm in NB_ARMS for dec in DECISIONS], ignore_index=True)
    out = {"taus": pd.concat(taus, ignore_index=True), "platt": plat}
    if not leak:
        hw, _, ch = interval_loco(d)
        out["interval_choice"] = ch
        out["interval_hw"] = d[["split_seed", "cluster_cpg"]].assign(hw=hw)
        res = d.y.to_numpy(float) - d.B1.to_numpy(float)
        q = []
        for s, g in d.groupby("split_seed", sort=True):
            cl = g.cluster_cpg.to_numpy(str)
            rr = res[g.index.to_numpy()]
            for c in np.unique(cl):
                q.append((int(s), str(c), conf_quantiles(rr[cl != c], [BOUND_LEVEL])[0],
                          conf_quantiles(-rr[cl != c], [BOUND_LEVEL])[0]))
        out["bounds_q"] = pd.DataFrame(q, columns=["split_seed", "cluster_cpg", "q_up", "q_dn"])
    return out


def _cluster_part(T: pd.DataFrame, c: str, seeds) -> pd.DataFrame:
    return T[(T.cluster_cpg == c) & T.split_seed.isin(seeds)].reset_index(drop=True)


def poison_check(d: pd.DataFrame, design: str, *, seeds=None, log=print) -> dict:
    """For every cluster c: replace the labels of c's rows by noise and refit every calibrated quantity (thresholds of
    every procedure / alpha / hull / Mondrian for B1, B0, HBK@ML; Platt parameters; the interval's nested level and
    half-widths; the bound quantiles).  Every quantity fitted for c must be bit-identical.  Positive control: the
    leaky calibration (c's rows included) must change for every cluster."""
    seeds = tuple(sorted(d.split_seed.unique())) if seeds is None else tuple(seeds)
    d = d[d.split_seed.isin(seeds)].reset_index(drop=True)
    cfgs = list(configs(POISON_ARMS))
    base = fitted_quantities(d, cfgs)
    base_leak = fitted_quantities(d, cfgs, leak=True)
    rng = np.random.default_rng(paths.seed_for(f"EDEC|POISON|{design}"))
    rep = {"design": design, "seeds": [int(s) for s in seeds], "clusters": 0, "configs": len(cfgs),
           "quantities_compared": 0, "clusters_with_diff": [], "pc_caught": 0, "pc_clusters": 0}
    for c in sorted(d.cluster_cpg.unique()):
        dp = d.copy()
        m = (dp.cluster_cpg == c).to_numpy()
        dp.loc[m, "y"] = rng.normal(0.0, 3.0, int(m.sum()))
        f2 = fitted_quantities(dp, cfgs)
        same = True
        for k, T in base.items():
            a, b = _cluster_part(T, c, seeds), _cluster_part(f2[k], c, seeds)
            rep["quantities_compared"] += int(a.size)
            if not a.equals(b):
                same = False
        if not same:
            rep["clusters_with_diff"].append(c)
        l2 = fitted_quantities(dp, cfgs, leak=True)
        caught = any(not _cluster_part(base_leak[k], c, seeds).equals(_cluster_part(l2[k], c, seeds)) for k in base_leak)
        rep["pc_caught"] += int(caught)
        rep["pc_clusters"] += 1
        rep["clusters"] += 1
    rep["checks_ok"] = bool(not rep["clusters_with_diff"] and rep["pc_caught"] == rep["pc_clusters"])
    # s0-mechanics-5: a pass needs every dev seed and every cluster_cpg cluster (13 on E-A rows)
    rep["coverage_full"] = bool(rep["seeds"] == list(paths.DEV_SEEDS) and rep["clusters"] == N_CLUSTERS
                                and rep["pc_clusters"] == rep["clusters"])
    rep["passed"] = bool(rep["checks_ok"] and rep["coverage_full"])
    rep["scope"] = POISON_SCOPE
    log(f"  fit poisoning {design}: {rep['clusters']} clusters x {len(seeds)} seeds, {rep['quantities_compared']} values "
        f"compared; clusters with a change {len(rep['clusters_with_diff'])}; leaky positive control caught "
        f"{rep['pc_caught']}/{rep['pc_clusters']} -> passed {rep['passed']}")
    return rep


# ------------------------------------------------------------------------------------------------------------
# the primary statistics (LOPO, C5-BTBP in / out)
# ------------------------------------------------------------------------------------------------------------
def primary_stats(d: pd.DataFrame) -> dict:
    """B1 / B0: AURC (all rows), LTT alpha 0.10 hull-rule coverage and per-call false-sign rate, NB at p_t 0.3
    (separable); and the B1 - B0 contrasts.  Recalibrates on ``d`` (LOPO and C5 frames recalibrate too).  Beside the
    per-call rate: its pooled counts (pooled_counts; descriptive, not registered statistics)."""
    d = d.reset_index(drop=True)
    out = {}
    for arm in ("B1", "B0"):
        out[f"AURC_{arm}"] = nanmean(aurc(score(g[arm]), false_sign(g[arm], g.y)) for _, g in d.groupby("split_seed"))
        tt = loco_taus(d, arm, ALPHA, "LTT", hull=True, mondrian=False)
        call = apply_calls(d, arm, tt, hull=True, mondrian=False)
        per = sel_per_seed(d, call, arm)
        out[f"cov_{arm}"], out[f"fs_{arm}"] = nanmean(r["coverage"] for r in per), nanmean(r["fs_rate"] for r in per)
        pc = pooled_counts(d, call, arm)
        out.update({f"{k}_{arm}": pc[k] for k in POOLED_KEYS + ("calls_per_seed", "few_calls")})
        pi, _ = platt_loco(d, arm, DECISIONS[0])
        ev = event(d.y, DECISIONS[0])
        out[f"NB30_{arm}"] = nanmean(net_benefit(pi[g.index], ev[g.index], 0.3) for _, g in d.groupby("split_seed"))
    out["dAURC_B0_minus_B1"] = out["AURC_B0"] - out["AURC_B1"]
    out["dcov_B1_minus_B0"] = out["cov_B1"] - out["cov_B0"]
    out["dNB30_B1_minus_B0"] = out["NB30_B1"] - out["NB30_B0"]
    return out


PRIMARY_KEYS = ("AURC_B1", "AURC_B0", "dAURC_B0_minus_B1", "cov_B1", "cov_B0", "dcov_B1_minus_B0", "fs_B1", "fs_B0",
                "NB30_B1", "NB30_B0", "dNB30_B1_minus_B0")
COUNT_KEYS = tuple(f"{k}_{arm}" for arm in ("B1", "B0") for k in POOLED_KEYS)    # beside fs_B1 / fs_B0; no LOPO
JK_LABEL = "jackknife t(P-1) interval (descriptive, anti-conservative)"


class _Memo:
    def __init__(self, fn):
        self.fn, self.cache = fn, {}

    def __call__(self, x: pd.DataFrame) -> dict:
        k = (len(x), tuple(sorted(x.publication.astype(str).unique())), tuple(sorted(x.cell_id.astype(str).unique())))
        if k not in self.cache:
            self.cache[k] = self.fn(x)
        return self.cache[k]


DUP_COLUMNS = ("noV2", "noBITDUP", "V2-3_stratum", "without_V2-3")      # registry 1.5-S11..S13 (s0-fidelity-8)


def duplicate_frames(d: pd.DataFrame) -> dict:
    """Duplicate variants of the main E-DEC frame (each recalibrated by primary_stats): V2-1/2/3 cells dropped; the
    second cell of every Stage-0 flagged bit link dropped; the V2-3 condition-aware stratum (11 C3 cells, S1 open)
    and the rows without it."""
    from . import population as PO
    s = PO.duplicate_cell_sets()
    c = d.cell_id.astype(str)
    return {"noV2": d[~c.isin(s["noV2"])], "noBITDUP": d[~c.isin(s["noBITDUP"])], "V2-3_stratum": d[c.isin(s["V2-3"])],
            "without_V2-3": d[~c.isin(s["V2-3"])]}


def _safe(memo, x) -> dict:
    try:
        return memo(x.reset_index(drop=True)) if len(x) else {}
    except Exception:                       # noqa: BLE001  (a tiny stratum cannot always be calibrated: reported NaN)
        return {}


def t_primary(d: pd.DataFrame, design: str) -> tuple[pd.DataFrame, dict]:
    """Point, LOPO (r5.common.lopo, t(P - 1)), C5-BTBP out, c950dc2982e96b83 out and the duplicate variants (noV2,
    noBITDUP, the V2-3 stratum and its complement; each recalibrated) for every primary statistic; then the pooled
    counts beside fs_B1 / fs_B0 in every frame (COUNT_KEYS; descriptive rows without LOPO)."""
    memo = _Memo(primary_stats)
    sp = CM.c5_split(d)
    c5 = {k: memo(v.reset_index(drop=True)) for k, v in sp.items()}
    dup = {k: _safe(memo, v) for k, v in duplicate_frames(d).items()}
    frames = {"point": c5["C5_in"], "C5_out": c5["C5_out"], "C5_cell_out": c5["C5_cell_out"], **dup}
    rows, lopo = [], {}
    for k in PRIMARY_KEYS:
        L = CM.lopo(d, lambda x, k=k: memo(x)[k], by="publication")
        lopo[k] = L
        few = ""
        if k.startswith("fs_"):
            arm = k.split("_")[1]
            hit = [f"{f} [{r[f'calls_per_seed_{arm}']}]" for f, r in frames.items() if r.get(f"few_calls_{arm}")]
            few = f"seed mean rests on few calls (per-seed calls below n_min {n_min(ALPHA)}) in: " + ", ".join(hit) if hit else ""
        rows.append({"statistic": k, "point": c5["C5_in"][k], "C5_out": c5["C5_out"][k], "C5_cell_out": c5["C5_cell_out"][k],
                     **{c: dup[c].get(k, np.nan) for c in DUP_COLUMNS},
                     "LOPO_lo": L["lo"], "LOPO_hi": L["hi"], "LOPO_min": L["min"], "LOPO_min_by": L["min_by"],
                     "LOPO_max": L["max"], "LOPO_max_by": L["max_by"], "P": L["P"], "LOPO_se": L["se"],
                     "LOPO_n_undefined": L["n_undefined"], "few_calls": few,
                     "label": label(design, f"LOPO_lo / LOPO_hi: {JK_LABEL}; LOPO_min / LOPO_max: the honest spread",
                                    "C5 and duplicate columns SENSITIVITY",
                                    "per-call rate (registered: the seed mean of per-seed ratios; pooled counts in the "
                                    f"{k.split('_')[1]} count rows)" if k.startswith("fs_") else "")})
    for k in COUNT_KEYS:
        rows.append({"statistic": k, "point": c5["C5_in"][k], "C5_out": c5["C5_out"][k], "C5_cell_out": c5["C5_cell_out"][k],
                     **{c: dup[c].get(k, np.nan) for c in DUP_COLUMNS}, "few_calls": "",
                     "label": label(design, f"count beside the registered seed mean fs_{k.rsplit('_', 1)[1]} (pooled over "
                                    "the seeds, which share cells; distinct = (cell, pair) queries); descriptive, not a "
                                    "registered statistic; no LOPO", "C5 and duplicate columns SENSITIVITY")})
    T = pd.DataFrame(rows)
    for c in ("P", "LOPO_n_undefined"):
        T[c] = T[c].astype("Int64")
    return T, {"c5": c5, "duplicates": dup,
               "lopo": {k: {kk: v for kk, v in L.items() if kk != "values"} for k, L in lopo.items()}}


# ------------------------------------------------------------------------------------------------------------
# tables of one design
# ------------------------------------------------------------------------------------------------------------
def no_hull_note(d, arm: str, cfg: str) -> str:
    """The note on a no-hull (S03) row: whether the arm can call any out-of-hull row at all."""
    oh = ~d.hull.to_numpy(bool)
    n_oh, k = int(oh.sum()), int((score(d[arm]) >= 0)[oh].sum())
    if k:
        return f"{k} of {n_oh} out-of-hull rows callable for {arm} (p != 0): the no-hull sensitivity can add calls"
    s = (f"{arm} never calls outside the read hull: p = 0 (uncallable) on all {n_oh} out-of-hull rows "
         f"({n_oh / max(d.split_seed.nunique(), 1):.1f} per seed), so the no-hull sensitivity adds no calls; ")
    if cfg.startswith("CRC"):
        return s + ("any difference in calls, coverage or false-sign rates from the hull-rule configuration comes only "
                    "from CRC's larger calibration denominator (never-callable rows count as zero-loss rows); eligible / "
                    "coverage_eligible differ because every row is eligible; equal values do not show that the hull "
                    "rule is irrelevant")
    return s + ("the calls, coverage and false-sign rates equal the hull-rule configuration by construction (eligible / "
                "coverage_eligible differ only because every row is eligible), which does not show that the hull rule "
                "is irrelevant")


def t_selective(d, sel, design, c5name, pool=None) -> pd.DataFrame:
    """``pool`` {(arm, config): pooled_counts} adds the pooled counts beside the seed-mean per-call rate."""
    rows = []
    for (arm, cfg), per in sel.items():
        rec = {"c5": c5name, "arm": arm, "role": ROLE[arm], "config": cfg, **sel_summary(per)}
        if pool is not None:
            pc = pool[(arm, cfg)]
            rec.update({k: pc[k] for k in POOLED_KEYS + ("calls_per_seed", "n_min", "few_calls")})
        rec["no_hull_note"] = no_hull_note(d, arm, cfg) if "_all_" in cfg else ""
        rec["label"] = label(design, cfg_label(cfg), "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)",
                             DIAG if arm == "HIST_D7" else "")
        rows.append(rec)
    return pd.DataFrame(rows)


def t_pairtype(d, calls, design, c5name) -> pd.DataFrame:
    """Per-pair-type validity (the primary report), hull-rule configurations."""
    rows = []
    pt = d.pair.to_numpy(str)
    for (arm, cfg), call in calls.items():
        if "_hull_" not in cfg:
            continue
        for q in PAIR_ORDER:
            sub = d[pt == q]
            if not len(sub):
                continue
            per = [sel_seed(g.y, g[arm], call[g.index.to_numpy()], g.hull) for _, g in sub.groupby("split_seed")]
            s = sel_summary(per)
            pc = pooled_counts(sub, call[sub.index.to_numpy()], arm, cfg_alpha(cfg))
            nseed = sub.split_seed.nunique()
            rows.append({"c5": c5name, "arm": arm, "config": cfg, "pair": q, "rows_per_seed": len(sub) / nseed,
                         "out_of_hull_rows_per_seed": float((~sub.hull).sum() / nseed),
                         "separable_share": float((sub.y.abs() >= Y_MIN).mean()), **s,
                         **{k: pc[k] for k in POOLED_KEYS + ("calls_per_seed", "n_min", "few_calls")},
                         "label": label(design, cfg_label(cfg), "per-pair-type validity (primary report)",
                                        "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)",
                                        DIAG if arm == "HIST_D7" else "")})
    return pd.DataFrame(rows)


def t_contrasts(sel, design, c5name, pool=None) -> pd.DataFrame:
    """``pool`` {(arm, config): pooled_counts} adds the seeds behind dfs_rate and both arms' pooled counts."""
    rows = []
    cfgs = sorted({c for _, c in sel})
    for cfg in cfgs:
        b = sel[("B1", cfg)]
        for x in ARMS[1:]:
            o = sel[(x, cfg)]
            dc = [r1["coverage"] - r0["coverage"] for r1, r0 in zip(b, o)]
            dfs = [r1["fs_rate"] - r0["fs_rate"] for r1, r0 in zip(b, o)]
            rec = {"c5": c5name, "config": cfg, "contrast": f"B1 - {x}", "dcoverage": nanmean(dc),
                   "dcoverage_per_seed": per_seed_str(dc), "seeds_B1_higher": int(sum(v > 0 for v in dc)),
                   "dfs_rate": nanmean(dfs), "dfs_rate_per_seed": per_seed_str(dfs)}
            if pool is not None:
                pb, po = pool[("B1", cfg)], pool[(x, cfg)]
                rec.update({"dfs_rate_seeds": int(np.isfinite(np.asarray(dfs, float)).sum()),
                            "dfs_rate_pooled": pb["fs_pooled"] - po["fs_pooled"],
                            "fs_pooled_counts": f"B1 {counts_str(pb)} | {x} {counts_str(po)}"})
            rec["label"] = label(design, cfg_label(cfg), "descriptive contrast; never an R2 gain",
                                 "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)", DIAG if x == "HIST_D7" else "")
            rows.append(rec)
    return pd.DataFrame(rows)


def t_risk_coverage(d, design, c5name) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, curve = [], []
    for arm in ARMS:
        per = []
        cur = {c: [] for c in COV_GRID}
        for _, g in d.groupby("split_seed", sort=True):
            p, y = g[arm].to_numpy(float), g.y.to_numpy(float)
            sc, fs = score(p), false_sign(p, y)
            h = g.hull.to_numpy(bool)
            r = rc_expected(sc, fs)
            per.append({"AURC": float(r.mean()), "AURC_oracle": aurc_oracle(sc, fs),
                        "AURC_hull_rows": aurc(sc[h], fs[h]), "AURC_oracle_hull_rows": aurc_oracle(sc[h], fs[h]),
                        "full_coverage_fs": float(fs.sum() / (sc >= 0).sum())})
            for c in COV_GRID:
                k = int(round(c * len(g)))
                cur[c].append(float(r[k - 1]) if 1 <= k <= len(r) else np.nan)
        rec = {"c5": c5name, "arm": arm, "role": ROLE[arm]}
        for k in per[0]:
            rec[k] = nanmean(x[k] for x in per)
        rec["E_AURC"] = rec["AURC"] - rec["AURC_oracle"]
        rec["E_AURC_hull_rows"] = rec["AURC_hull_rows"] - rec["AURC_oracle_hull_rows"]
        rec["AURC_per_seed"] = per_seed_str(x["AURC"] for x in per)
        rec["label"] = label(design, "AURC_oracle* and E_AURC* use the oracle order: ORACLE [O]",
                             "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)", DIAG if arm == "HIST_D7" else "")
        rows.append(rec)
        for c in COV_GRID:
            curve.append({"c5": c5name, "arm": arm, "coverage": c, "risk": nanmean(cur[c]), "risk_per_seed": per_seed_str(cur[c]),
                          "label": label(design, "risk-coverage curve (per-call false-sign rate, ties shared)",
                                         "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)")})
    return pd.DataFrame(rows), pd.DataFrame(curve)


def t_netbenefit(d, design, c5name) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    nb_rows, citl, params, probs = [], [], [], {}
    for dec in DECISIONS:
        ev = event(d.y, dec)
        vals = {}
        for arm in NB_ARMS:
            pi, par = platt_loco(d, arm, dec)
            probs[(arm, dec)] = pi
            params.append(par)
            vals[arm] = {pt: [net_benefit(pi[g.index], ev[g.index], pt) for _, g in d.groupby("split_seed")] for pt in NB_PT}
            for c, g in d.groupby("cluster_cpg", sort=True):
                O = [ev[h.index].mean() for _, h in g.groupby("split_seed")]
                Ex = [pi[h.index].mean() for _, h in g.groupby("split_seed")]
                citl.append({"c5": c5name, "decision": dec, "arm": arm, "cluster_cpg": c, "rows_per_seed": len(g) / g.split_seed.nunique(),
                             "observed": nanmean(O), "expected": nanmean(Ex), "O_minus_E": nanmean(o - e for o, e in zip(O, Ex)),
                             "O_over_E": nanmean(o / e for o, e in zip(O, Ex) if e > 0),
                             "label": label(design, "calibration-in-the-large per cluster (Platt, leave-one-cluster-out)",
                                            "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)")})
        vals["treat-all"] = {pt: [net_benefit(np.ones(len(g)), ev[g.index], pt) for _, g in d.groupby("split_seed")] for pt in NB_PT}
        vals["treat-none"] = {pt: [0.0] * d.split_seed.nunique() for pt in NB_PT}
        for arm, v in vals.items():
            for pt in NB_PT:
                rec = {"c5": c5name, "decision": dec, "arm": arm, "p_t": pt, "NB": nanmean(v[pt]), "NB_per_seed": per_seed_str(v[pt])}
                if arm != "B1":
                    dd = [a - b for a, b in zip(vals["B1"][pt], v[pt])]
                    rec["dNB_B1_minus_arm"], rec["dNB_per_seed"] = nanmean(dd), per_seed_str(dd)
                rec["label"] = label(design, "net benefit (Platt-scaled |p|, leave-one-cluster-out)",
                                     "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)")
                nb_rows.append(rec)
    plab = label(design, "fitted Platt parameters (a, b) of |p| per (seed, held-out cluster), fitted on the other clusters",
                 "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)")
    return (pd.DataFrame(nb_rows), pd.DataFrame(citl), pd.concat(params, ignore_index=True).assign(c5=c5name, label=plab),
            probs)


def t_interval(d, design, c5name) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    hw, cov, ch = interval_loco(d)
    per = [(cov[g.index].mean(), np.mean(hw[g.index][np.isfinite(hw[g.index])]), np.isinf(hw[g.index]).mean())
           for _, g in d.groupby("split_seed")]
    fin = np.isfinite(hw)
    line = pd.DataFrame([{"c5": c5name, "arm": "B1", "target_coverage": TARGET_COV, "coverage": nanmean(x[0] for x in per),
                          "coverage_per_seed": per_seed_str(x[0] for x in per), "mean_half_width": nanmean(x[1] for x in per),
                          "median_half_width": float(np.median(hw[fin])) if fin.any() else np.nan,
                          "times_divide_factor": float(10 ** nanmean(x[1] for x in per)),
                          "share_infinite": nanmean(x[2] for x in per), "nominal_chosen_mean": float(ch.nominal_chosen.mean()),
                          "nominal_chosen_range": f"{ch.nominal_chosen.min():.2f}-{ch.nominal_chosen.max():.2f}",
                          "label": label(design, "M16 interval line (row-weighted Mondrian, nested LOCO level); descriptive",
                                         "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)")}])
    bycl = []
    for c, g in d.groupby("cluster_cpg", sort=True):
        bycl.append({"c5": c5name, "cluster_cpg": c, "rows_per_seed": len(g) / g.split_seed.nunique(),
                     "coverage": float(cov[g.index].mean()), "mean_half_width": float(np.mean(hw[g.index][np.isfinite(hw[g.index])]))
                     if np.isfinite(hw[g.index]).any() else np.nan,
                     "nominal_chosen": ";".join(f"{v:.2f}" for v in ch[ch.cluster_cpg == c].nominal_chosen),
                     "label": label(design, "M16 interval line per cluster", "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)")})
    return line, pd.DataFrame(bycl).merge(ch.groupby("cluster_cpg").inner_coverage.mean().rename("inner_coverage_mean"),
                                          on="cluster_cpg", how="left"), hw, cov


def t_bounds(d, va, vb, design, c5name) -> tuple[pd.DataFrame, pd.DataFrame]:
    R, _ = bounds_loco(d, va, vb)
    ns = d.split_seed.nunique()
    if len(R):
        per = [g.covered.mean() for _, g in R.groupby("split_seed")]
        rec = {"rows_per_seed": len(R) / ns, "upper_rows_per_seed": float((R.side.str.startswith("upper")).sum() / ns),
               "lower_rows_per_seed": float((R.side.str.startswith("lower")).sum() / ns),
               "C5_share": float((R.publication == paths.C5).mean()), "coverage": nanmean(per),
               "coverage_per_seed": per_seed_str(per), "mean_offset": float(R.offset.mean())}
    else:
        rec = {"rows_per_seed": 0.0, "upper_rows_per_seed": 0.0, "lower_rows_per_seed": 0.0, "C5_share": np.nan,
               "coverage": np.nan, "coverage_per_seed": "", "mean_offset": np.nan}
    line = pd.DataFrame([{"c5": c5name, "limit_logD": LIMIT, "level": BOUND_LEVEL, "arm": "B1", **rec,
                          "label": label(design, "M23 deployable one-sided bounds; C5-BTBP floor sensitivity",
                                         "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)")}])
    return line, R.assign(c5=c5name, label=label(design, "M23 bound rows (one per bounded query; quantiles from the same "
                                                 "seed's other clusters); C5-BTBP floor sensitivity",
                                                 "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)"))


def t_savings(d, calls, design, c5name) -> pd.DataFrame:
    cfg = cfg_name(*PRIMARY)
    rows = []
    for arm in ("B1", "B0", "HBK@ML"):
        call = calls[(arm, cfg)]
        per = [savings_seed(g, call[g.index.to_numpy()]) for _, g in d.groupby("split_seed", sort=True)]
        s = {k: nanmean(x[k] for x in per) for k in per[0]}
        base = {"c5": c5name, "arm": arm, "config": cfg}
        lab = label(design, cfg_label(cfg), "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)")
        rows.append({**base, "scenario": "ICP-MS (all 14 Ln from one experiment)", "unit": "experiments saved",
                     "value": 0.0, "note": "0 by construction: the 4-read experiment already measured every metal", "label": lab})
        rows.append({**base, "scenario": "subset feed / radiotracer, one full follow-up experiment per cell",
                     "unit": "share of cells whose follow-up is avoided (every scored pair called)",
                     "value": s["share_cells_fully_called"], "per_seed": per_seed_str(x["share_cells_fully_called"] for x in per),
                     "note": f"{s['cells_fully_called']:.1f} of {s['cells']:.0f} cells per seed", "label": lab})
        rows.append({**base, "scenario": "radiotracer, one metal per measurement", "unit": "share of unread pair-end metal measurements saved",
                     "value": s["share_metal_measurements_saved"], "per_seed": per_seed_str(x["share_metal_measurements_saved"] for x in per),
                     "note": f"{s['metals_still_needed']:.1f} of {s['unread_pair_end_metals']:.0f} still needed per seed", "label": lab})
        rows.append({**base, "scenario": "literature mining (partial series as reported)", "unit": "pairs signed without a new measurement per seed",
                     "value": s["pairs_signed"], "per_seed": per_seed_str(x["pairs_signed"] for x in per), "note": "", "label": lab})
    return pd.DataFrame(rows)


def t_clusters(d, calls, probs, design, c5name) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The unit table (cluster_cpg) at the primary configuration and its descriptive tests (DIAGNOSTIC)."""
    cfg = cfg_name(*PRIMARY)
    ns = d.split_seed.nunique()
    nseed = d.groupby("split_seed").size()
    ev = event(d.y, DECISIONS[0])
    rows = []
    for c, g in d.groupby("cluster_cpg", sort=True):
        rec = {"c5": c5name, "cluster_cpg": c, "rows_per_seed": len(g) / ns, "cells": g.cell_id.nunique(),
               "publications": g.publication.nunique(), "holds_C5": bool((g.publication.astype(str) == paths.C5).any()),
               "hull_rows_per_seed": float(g.hull.sum() / ns)}
        for arm in ("B1", "B0", "HBK@ML"):
            call = calls[(arm, cfg)]
            per = [sel_seed(h.y, h[arm], call[h.index.to_numpy()], h.hull) for _, h in g.groupby("split_seed")]
            rec[f"coverage_{arm}"] = nanmean(r["coverage"] for r in per)
            rec[f"fs_rate_{arm}"] = nanmean(r["fs_rate"] for r in per)
            rec[f"calls_{arm}"] = nanmean(r["calls"] for r in per)
            rec[f"fs_counts_{arm}"] = counts_str(pooled_counts(g, call[g.index.to_numpy()], arm))
        for x in ("B0", "HBK@ML"):
            rec[f"contrib_cov_B1_minus_{x}"] = nanmean(
                (calls[("B1", cfg)][h.index.to_numpy()].sum() - calls[(x, cfg)][h.index.to_numpy()].sum()) / nseed[s]
                for s, h in g.groupby("split_seed"))
        for arm in ("B1", "B0"):
            pi = probs[(arm, DECISIONS[0])]
            rec[f"CITL_sep_{arm}"] = nanmean(ev[h.index].mean() - pi[h.index].mean() for _, h in g.groupby("split_seed"))
        rec["label"] = label(design, cfg_label(cfg), "unit table", "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)")
        rows.append(rec)
    U = pd.DataFrame(rows)
    tests = []
    for x in ("B0", "HBK@ML"):
        v = U[f"contrib_cov_B1_minus_{x}"].to_numpy(float)
        sf = CM.sf_pool(v, f"EDEC|UNIT|{design}|{c5name}|B1-{x}")
        sg = CM.sign_test(v)
        tests.append({"c5": c5name, "contrast": f"B1 - {x} (coverage contribution, primary configuration)", "G": sf["G"],
                      "W/L/T": f"{sg['W']}/{sg['L']}/{sg['T']}", "pooled_dcoverage": float(v.sum()), "SF_POOL_p": sf["p"],
                      "SF_exact": sf["exact"], "SIGN_p": sg["p"], "top3_share": CM.top3_share(v),
                      "label": label(design, DIAG + " (descriptive unit tests; no in-corpus p-value is called significant)",
                                     "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)")})
    return U, pd.DataFrame(tests)


def t_rank(d, design, c5name) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, shares = [], []
    for gname, by, nmin in (("pair-type groups (seed, fold, pair; >= 5)", ["fold", "pair"], RANK_MIN),
                            ("within publication (seed, fold, pair, publication; >= 3)", ["fold", "pair", "publication"], RANK_MIN_PUB)):
        T = rank_table_seed(d, by, nmin)
        for a, per in T["per_seed"].items():
            rec = {"c5": c5name, "grouping": gname, "arm": a, "groups_per_seed": T["groups_per_seed"],
                   "median_candidates": T["median_candidates"]}
            for k in ("top1", "top3", "regret", "spearman"):
                rec[k] = nanmean(x[k] for x in per)
                rec[f"{k}_per_seed"] = per_seed_str(x[k] for x in per)
            rec["label"] = label(design, "E-RANK (M15 salvage; reporting appendix; never a claim)",
                                 ORC if a == "AMP_ORACLE" else ("exact expectation" if a == "RANDOM" else ""),
                                 "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)")
            rows.append(rec)
        sh = line_shares(T["per_seed"])
        jk = {c: line_shares(rank_table_seed(d[d.cluster_cpg != c], by, nmin)["per_seed"]) for c in sorted(d.cluster_cpg.unique())}
        for k in ("top1", "top3", "regret", "spearman"):
            v = np.array([jk[c][k] for c in jk], float)
            ok = np.isfinite(v)
            shares.append({"c5": c5name, "grouping": gname, "endpoint": k, "LINE_share": sh[k],
                           "LOCO_min": float(v[ok].min()) if ok.any() else np.nan,
                           "LOCO_min_by": list(jk)[int(np.nanargmin(v))] if ok.any() else "",
                           "LOCO_max": float(v[ok].max()) if ok.any() else np.nan,
                           "LOCO_max_by": list(jk)[int(np.nanargmax(v))] if ok.any() else "",
                           "label": label(design, "E-RANK LINE share (LINE - RANDOM) / (B1 - RANDOM); leave-one-cluster-out range",
                                          "" if c5name == "C5_in" else "SENSITIVITY (C5-BTBP out)")})
    return pd.DataFrame(rows), pd.DataFrame(shares)


def t_erec(F: pd.DataFrame, design: str) -> dict:
    """The E-REC own-prior tables: population, arms per stratum, contrasts, units and tests, LOPO."""
    pop, arms, con, lopo_rows = [], [], [], []
    strata = erec_strata(F)
    units, tests = pd.DataFrame(), []
    for sname, (mask, prefix, slab) in strata.items():
        cols = erec_cols(prefix)
        S = F[mask]
        S = S[np.isfinite(S[list(cols.values())].to_numpy(float)).all(1)].reset_index(drop=True)
        ns = S.split_seed.nunique()
        pop.append({"stratum": sname, "cells": S.cell_id.nunique(), "publications": S.publication.nunique(),
                    "units_pubchem": S.pubchem.nunique(), "clusters": S.cluster_cpg.nunique(),
                    "rows_per_seed": len(S) / ns if ns else 0.0, "rows_dropped_incomplete": int(mask.sum() - len(S)),
                    "chpt_pairtype_fallback_share": float((S.chpt_source == "pair type").mean()) if len(S) else np.nan,
                    "label": "METADATA (population counts; no score)"})
        if not len(S):
            continue
        per = [erec_seed(g.y, {a: g[c].to_numpy(float) for a, c in cols.items()}, g[cols["COPY_own"]])
               for _, g in S.groupby("split_seed", sort=True)]
        lab = label(design, "E-REC own-prior row of E-DEC (recipe-3 salvage); identical rows; " + slab,
                    "never next to 0.452")
        lab_arms = label(design, "E-REC own-prior row of E-DEC (recipe-3 salvage); identical rows; " + slab,
                         "E_AURC uses the oracle order: " + ORC, "never next to 0.452")
        for a in cols:
            rec = {"stratum": sname, "arm": a, "column": cols[a], "rows_per_seed": len(S) / ns, "c_ref": nanmean(r["c_ref"] for r in per)}
            for k in per[0][a]:
                rec[k] = nanmean(r[a][k] for r in per)
            rec["fs@c_ref_per_seed"] = per_seed_str(r[a]["fs@c_ref"] for r in per)
            rec["AURC_per_seed"] = per_seed_str(r[a]["AURC"] for r in per)
            rec["label"] = lab_arms
            arms.append(rec)
        for x, b in EREC_CONTRASTS:
            dfs = [r[b]["fs@c_ref"] - r[x]["fs@c_ref"] for r in per]
            dau = [r[b]["AURC"] - r[x]["AURC"] for r in per]
            con.append({"stratum": sname, "contrast": f"{b} - {x} (positive: {x} makes fewer false signs)",
                        "dfs@c_ref": nanmean(dfs), "dfs@c_ref_per_seed": per_seed_str(dfs), "dAURC": nanmean(dau),
                        "dAURC_per_seed": per_seed_str(dau), "label": lab})
        if sname == "PRIMARY_OWN":
            U = erec_unit_frame(S, cols)
            units = U.reset_index()
            units["label"] = label(design, "E-REC unit table (publication x chemotype; matched coverage c_ref)")
            for x, b in EREC_CONTRASTS[:4]:
                v = (U[f"fs_{b}"] - U[f"fs_{x}"]).to_numpy(float)
                sf = CM.sf_pool(v, f"EDEC|EREC|{design}|{x}-vs-{b}")
                sg = CM.sign_test(v)
                tests.append({"contrast": f"{b} - {x} (false-sign count at c_ref per unit)", "G": sf["G"],
                              "W/L/T": f"{sg['W']}/{sg['L']}/{sg['T']}", "pooled": float(v.sum()), "SF_POOL_p": sf["p"],
                              "SF_exact": sf["exact"], "SF_seed": sf["seed"], "SIGN_p": sg["p"], "top3_share": CM.top3_share(v),
                              "label": label(design, DIAG + " (descriptive unit tests)")})

                def fn(z, x=x, b=b, cols=cols):
                    vals = []
                    for _, g in z.groupby("split_seed", sort=True):
                        e = erec_seed(g.y, {k: g[cols[k]].to_numpy(float) for k in (x, b, "COPY_own")}, g[cols["COPY_own"]])
                        vals.append(e[b]["fs@c_ref"] - e[x]["fs@c_ref"])
                    return nanmean(vals)
                L = CM.lopo(S, fn, by="publication")
                lopo_rows.append({"contrast": f"{b} - {x} (fs@c_ref)", "point": L["point"], "LOPO_lo": L["lo"], "LOPO_hi": L["hi"],
                                  "LOPO_min": L["min"], "LOPO_min_by": L["min_by"], "LOPO_max": L["max"], "LOPO_max_by": L["max_by"],
                                  "P": L["P"], "label": label(design, "E-REC LOPO (t(P-1); descriptive)")})
    return {"erec_population": pd.DataFrame(pop), "erec_arms": pd.DataFrame(arms), "erec_contrasts": pd.DataFrame(con),
            "erec_units": units, "erec_unit_tests": pd.DataFrame(tests), "erec_lopo": pd.DataFrame(lopo_rows)}


def design_tables(d: pd.DataFrame, design: str, *, va=None, vb=None, F: pd.DataFrame | None = None,
                  erec_status: str = "", log=print) -> tuple[dict, dict, dict]:
    """Every table of one design.  Returns (tables, row frames, extras)."""
    T, rows, extra = {}, {}, {}
    sp = CM.c5_split(d)
    for c5name in ("C5_in", "C5_out"):
        dd = sp[c5name].reset_index(drop=True)
        calls, taus = run_selective(dd)
        sel = {k: sel_per_seed(dd, v, k[0], hull="_hull_" in k[1]) for k, v in calls.items()}
        pool = {k: pooled_counts(dd, v, k[0], cfg_alpha(k[1])) for k, v in calls.items()}
        parts = {"selective": t_selective(dd, sel, design, c5name, pool), "pairtype": t_pairtype(dd, calls, design, c5name),
                 "contrasts": t_contrasts(sel, design, c5name, pool)}
        parts["risk_coverage"], parts["rc_curve"] = t_risk_coverage(dd, design, c5name)
        parts["netbenefit"], parts["citl"], parts["platt"], probs = t_netbenefit(dd, design, c5name)
        parts["clusters"], parts["unit_tests"] = t_clusters(dd, calls, probs, design, c5name)
        parts["interval"], parts["interval_clusters"], hw, cov = t_interval(dd, design, c5name)
        if va is not None:
            keep = sp[c5name].index.to_numpy()
            parts["bounds"], parts["bounds_rows"] = t_bounds(dd, np.asarray(va)[keep], np.asarray(vb)[keep], design, c5name)
        parts["savings"] = t_savings(dd, calls, design, c5name)
        parts["rank"], parts["rank_line_share"] = t_rank(dd, design, c5name)
        for k, v in parts.items():
            T[k] = pd.concat([T[k], v], ignore_index=True) if k in T else v
        T["thresholds"] = pd.concat([T.get("thresholds", pd.DataFrame()), taus.assign(c5=c5name)], ignore_index=True)
        R = dd[["split_seed", "fold", "cell_id", "A", "B", "pair", "publication", "chemotype", "cluster_cpg", "read_set",
                "a_read", "b_read", "hull", "y"] + list(ARMS)].copy()
        new = {f"call|{a}|{c}": v for (a, c), v in calls.items()}
        new.update({f"P|{a}|{dec}": v for (a, dec), v in probs.items()})
        new.update({"interval_hw_B1": hw, "interval_covered_B1": cov})
        rows[c5name] = pd.concat([R, pd.DataFrame(new, index=R.index)], axis=1)
        log(f"  {design} {c5name}: {len(dd)} rows, {len(calls)} call configurations, {len(taus)} thresholds")
    T["primary"], extra["primary"] = t_primary(d, design)
    if F is not None:
        T.update(t_erec(F, design))
        extra["erec_status"] = "DONE"
    else:
        extra["erec_status"] = erec_status or "PENDING-ECT"
        T["erec_population"] = pd.DataFrame([{"stratum": "E-REC own-prior row", "status": extra["erec_status"],
                                              "label": "NOT RUN (needs the r5.ect arms)"}])
    return T, rows, extra


def _counts_as_int(df: pd.DataFrame) -> pd.DataFrame:
    """The COUNT_KEYS rows other than the pooled rate shown as integers in the markdown (NaN kept)."""
    out = df.astype(object)
    for k in COUNT_KEYS:
        if k in out.index and not k.startswith("fs_"):
            out.loc[k] = [int(round(v)) if isinstance(v, (int, float, np.number)) and np.isfinite(v) else v
                          for v in out.loc[k]]
    return out


def _n(v) -> str:
    return str(int(round(float(v)))) if np.isfinite(float(v)) else "NaN"


def changed_sentence(P: pd.DataFrame, sel: pd.DataFrame, primary: dict, seeds: int) -> str:
    """The section-9.3 'what changed' sentence, written from the computed primary statistics of this design."""
    def sel_v(arm, cfg, col):
        r = sel[(sel.arm == arm) & (sel.config == cfg) & (sel.c5 == "C5_in")]
        return float(r[col].iloc[0]) if len(r) else float("nan")
    crc, plg = cfg_name("CRC", ALPHA, True, False), cfg_name("PLUGIN", ALPHA, True, False)
    few = (primary.get("c5", {}).get("C5_in", {}) or {}).get("few_calls_B1", "")
    return (f"{SEEN}: under the valid primary calibration (LTT, alpha 0.10, hull rule, global thresholds) B1 signs "
            f"{P.loc['cov_B1', 'point']:.4f} of the adjacent pairs from 4 reads with C5-BTBP in and "
            f"{P.loc['cov_B1', 'C5_out']:.4f} with it out (B0 {P.loc['cov_B0', 'point']:.4f} / "
            f"{P.loc['cov_B0', 'C5_out']:.4f}), while CRC and the plug-in at alpha 0.10 (hull rule, global) call "
            f"{sel_v('B1', crc, 'coverage'):.4f} / {sel_v('B1', plg, 'coverage'):.4f} with C5-BTBP in (CRC controls the "
            f"per-pair rate, realised {sel_v('B1', crc, 'marginal_fs'):.4f}; the plug-in has no guarantee); the realised "
            f"LTT per-call false-sign rate of B1 is {P.loc['fs_B1', 'point']:.4f} as the registered seed mean "
            f"({_n(P.loc['false_signs_pooled_B1', 'point'])}/{_n(P.loc['calls_pooled_B1', 'point'])} = "
            f"{P.loc['fs_pooled_B1', 'point']:.4f} pooled, {_n(P.loc['calls_distinct_B1', 'point'])} distinct queries, "
            f"{_n(P.loc['false_signs_distinct_B1', 'point'])} with a false sign; calls in "
            f"{_n(P.loc['seeds_with_calls_B1', 'point'])} of {seeds} seeds) against alpha 0.10"
            + (f"; the {few}" if few else "") + ".  Nothing here is a test or a claim; the seen valid fixed-sequence "
            "coverage 0.149 [IR M14] came from another implementation and is not a target.")


def per_item(T: dict, extra: dict, design: str, d: pd.DataFrame, *, seal_tag: str, leakage: str, seconds: float,
             ect_checks: dict | None) -> dict:
    """The section-9.3 per-item table of one design (r5.common.per_item_table)."""
    P = T["primary"].set_index("statistic")
    pop = CM.population(d, "cluster_cpg")
    sel = T["selective"]
    prim = sel[(sel.config == cfg_name(*PRIMARY)) & (sel.c5 == "C5_in") & sel.arm.isin(["B1", "B0", "HBK@ML"])]
    seed_cols = ["arm", "coverage_per_seed", "calls_per_seed", "fs_rate_per_seed", "fs_rate", "fs_pooled",
                 "false_signs_pooled", "calls_pooled", "calls_distinct", "false_signs_distinct", "seeds_with_calls",
                 "few_calls"]
    PK = P.loc[list(PRIMARY_KEYS)]
    lopo = PK[["point", "LOPO_min", "LOPO_min_by", "LOPO_max", "LOPO_max_by", "LOPO_se", "LOPO_n_undefined", "P",
               "LOPO_lo", "LOPO_hi"]].rename(columns={
        "LOPO_min": "LOPO_min (spread)", "LOPO_max": "LOPO_max (spread)", "LOPO_se": "jackknife SE",
        "LOPO_n_undefined": "undefined leave-outs", "LOPO_lo": f"{JK_LABEL} lo", "LOPO_hi": f"{JK_LABEL} hi"})
    rows_c5 = list(PRIMARY_KEYS) + [k for k in COUNT_KEYS if k in P.index]
    ut = T["unit_tests"]
    ut0 = ut[ut.c5 == "C5_in"]
    er = T.get("erec_population", pd.DataFrame())
    erp = er[er.stratum == "PRIMARY_OWN"] if "stratum" in er and len(er) else er
    return {
        "item": f"1.5 E-DEC selective direction calls (M14) with the E-REC own-prior row and the E-RANK table ({design})",
        "population": {"E-A rows": pop, "E-REC PRIMARY_OWN": erp.drop(columns=["label"], errors="ignore").to_dict("records")[0]
                       if len(erp) and "cells" in erp else extra.get("erec_status", "")},
        "design": design + (" (decides)" if design == "BPG" else " (SENSITIVITY)"),
        "read_budget": "4 reads (E-A rows, DOPT_Q k = 3); E-REC: COPY 0, SH_own@2 / D7@2 2, SH_own@4 / D7@4 / B1@4 4 reads",
        "arms": "selective: B1, B0, HBK@ML (primary set), RIDGE2, LINE, G14@ZERO, HIST_D7 (DIAGNOSTIC); net benefit: B1, "
                "B0, HBK@ML, RIDGE2, Platt-LINE; E-RANK: + RANDOM, AMP_ORACLE [O]; E-REC: COPY_own, SH_own@2, SH_own@4, "
                "CHPT_MAJ, G14, D7@2, D7@4, B1@4",
        "metrics": "not applicable: E-DEC is a decision estimand (coverage, per-call false-sign rate, AURC, net benefit); "
                   "never an R2 gain (primary statistics below)",
        "per_seed": prim[[c for c in seed_cols if c in prim]].rename(columns={
            "fs_rate": "fs_rate (seed mean; registered)", "fs_pooled": "fs_rate pooled"}).reset_index(drop=True),
        "units": ut0.drop(columns=["c5", "label"]).reset_index(drop=True),
        "lopo_top3": lopo,
        "c5": _counts_as_int(P.loc[rows_c5, ["point", "C5_out", "C5_cell_out", "few_calls"]]),
        "duplicates": _counts_as_int(P.loc[rows_c5, ["point"] + list(DUP_COLUMNS)]),
        "leakage": leakage + (f"; E-REC identities {ect_checks}" if ect_checks else ""),
        "compute": f"{seconds:.0f} s",
        "label": label(design, NOT_R2, f"seal {seal_tag}"),
        "changed": changed_sentence(P, sel, extra.get("primary", {}), int(d.split_seed.nunique())),
        "next_step": "New-source direction-call evaluation is data-gated (>= 5 sources and >= 100 calls, IR M14); the "
                     "registry continues with item 1.6 (E-D).",
    }

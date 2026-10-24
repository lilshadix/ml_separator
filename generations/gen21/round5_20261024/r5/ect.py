"""E-CT, the condition-transfer estimand (CLAUDE_CODE_PROMPT.md section 6.1; conventions 6.0, 6.10, 8, 9): the frozen
predictor SH, every comparator, control and annex arm of the 6.1 table, and the in-corpus row builder.

Copied and adapted (never run in place) from round5_design_20261023/probes/scripts/
    P1_ct_build.py    LF SHA-256 afd7ae542f24df305c6f174fd3e816948b28298720d543772b685ded9d7138d0  (arms, row builder)
    P1_ct_diag.py     LF SHA-256 4234aa3d51a49690d75288f4c2b4907692806d6a5955b17111fa9cc2c8f66b2d  (OFF_PUB / OFF_XEXT,
                      the SH = OFF read-alignment identity)
    P1SK_recompute.py LF SHA-256 1bbca9d66a5d03a88a07b2662a2d645364a9b1bd86f9a115fed84887ff7663de  (shape-sd near copies)
and round5_design_20261023/ideas2/scripts/
    shift_build.py         LF SHA-256 3ed0db9e9a14792dd6e4de3d1397cf8423940cdad57206e4bcfe40751c6c1ccb  (ACOPY, AINT)
    recipe_fixedladder.py  LF SHA-256 dba2e664fc659f62cbb5a40d7232e5f9984254777e8689b5f0405c44c47676ea  (frozen ladder)
    recipe_build.py        LF SHA-256 0b572f7b5ad6133f4846ae9d9f2ab0745db5060b2a3161ccdaf25c87a183070e  (HULL order)
The sibling pools come from r5.population (metadata only plus the frozen bit-identity rule of r5.dupes).

**Explicit inputs (section 6.1 "Outputs", section 4.3 item 12).**  The predictor core never reaches into a fold cache,
a parquet or the bench: a ``Target`` carries the cell's measured metal set, its scored strict-adjacent pairs, its legal
reads per read budget r (values at the read metals only; ``make_target`` masks everything else away), the B1 Gd
amplitude beta and an optional 0-read shrinkage target; a pool is a matrix of full sibling series.
  * in-corpus: read sets = union of the cached fold sequence ``cell['seqs']['DOPT_Q'][:r - 1]``, beta = the fold's
    fitted round-4 amplitude (round4_20261022/results/REF/fitted_<design>.json), pools from r5.population;
  * external (C-ECT): read sets from the frozen all-C3 DOPT_Q object (``external_read_sets``: r5.frozen.external_sequence
    and read_set), beta = beta_frozen (results/stage0/beta_frozen.json), pools = the laboratory's other full-series
    conditions of the extractant.  ``b1_residual_sf`` gives OFF's donor residuals on new data.

**Arms** (read budgets in reads r; code budget k = r - 1; section 3.3).  n = |pool|, w = n / (n + K), K = 1.78.
  B1_r{2,3,4}   PWL_R[y_S - u_S] + u, u = beta 1[Gd], S the nested DOPT_Q set of r reads.   B0_r{2,3,4}  PWL_R[y_S].
  SIB_r         mean_j (Y_j + PWL_R[y_S - Y_j,S])  (= T + PWL_R[y_S - T_S], T the sibling mean).
  SH_r          w SIB_r + (1 - w) B1_r (frozen predictor; at 2 and 3 reads it shrinks toward B1 on the same reads).
  SHK2_r4       SH with K = 2 (E6 reproduction).   COPY  T (0 reads).   SH0  w COPY + (1 - w) G14@ZERO.
  OFF / OFF_ALL / OFF_PUB / OFF_XEXT   B1_r4 + sum_j r_j,p / (n_p + K), r = y - B1 (OOF, same seed and design) of the
                CLEAN C3 siblings / every other CLEAN C3 cell of the publication x extractant / every other non-dropped C3
                cell of the publication / the publication's other extractants only.  No donor at the pair: OFF and
                OFF_ALL fall back to B1 (P1), OFF_PUB and OFF_XEXT are NaN (scored where they exist).
  NEAR_r4, NEARSH_r4   the nearest-condition sibling(s) (ties averaged; r5.population distance), shrunk with
                w = n_tied / (n_tied + K).
  E_RAND1_*     exact expectation over one uniformly drawn sibling, raw and shrunk (w = 1 / (1 + K)); r = 0, 2, 3, 4.
  E_RAND2_*     the same over every sibling pair (w = 2 / (2 + K)); M04's "average at least 2"; r = 0, 2, 4.
  E_SUB{m}_*    cells with n >= 5: all m-subsets (a seeded random 5,000 when more), m = 1, 2, 3, 5; raw and SH_m.
  PB_{CHEM,XEXT,PUB}   placebo template (same chemotype other publications / same extractant other publications / other
                extractants same publication) blended at the target's own w; _raw unblended; scored where the pool exists.
  E_PB_*_CM     count-matched placebos: n-subsets of the placebo pool (n = the target's n; all subsets or a seeded random
                5,000), blended at the same w = n / (n + K) as SH (I2 rules 13-14); NaN when the pool holds < n series.
  PB_LABX       same extractant, same laboratory (frozen PROVISIONAL key, conservative), other publication (IR M01 step 8).
  PB_PUBHOM / PB_PUBHET   PB_PUB split into same-chemotype (homologue) and other-chemotype extractants (M06 annex).
  ACOPY_r0, AINT_r0   shift-1 annex: mean / per-metal interpolation in log10[acid] (flat beyond) of the acid-only
                siblings; dx_near, dx_mean, extrap, below describe the target's acid position.
  NP_LAD_r{2..5}, NP_DOPTQ_r5   E-REC no-prior policies: the frozen recipe-2 ladder (legal-HULL fallback) and the 5-read
                DOPT_Q set, both in B1's form with the same beta (5 reads = a different read budget, labelled so).
  Variant pools (sensitivities): NB_ (P1's CLEAN rule without the bit-identity rule), V1_ (every covering sibling),
  I1_ (CLEAN minus the item-1 link only), SD_ (shape-sd >= 0.02), OFS_ (offset / near-identity links dropped), QNT_
  (quantised cells dropped), V25_ (V2-5 floor readings dropped), INWIN_ / OUTWIN_ (M04: references inside / outside the
  [-2, 2] level window at every target metal).

**Information contract (section 8).**  Allowed: the target's legal reads, the siblings' full series, the fold's beta,
the siblings' OOF residuals (leave-target-out).  ``guard_pool`` refuses a pool that holds the target itself, or (except
for the NB_, V1_ and I1_ sensitivity pools, which exist to show the duplicate's effect) a series bit-identical to it
under the frozen section-5.1 rule.  ``Corpus.intervention`` replaces the target's unread labels and requires
bit-identical predictions; its positive controls must be caught.
"""
from __future__ import annotations

import itertools
import json
import math
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import paths
from . import dupes as DU
from . import population as POP
from r3 import interp  # noqa: E402
from gen20k import design as DS  # noqa: E402

# Copy rule (brief 1.2): every copied or re-implemented source with its LF SHA-256 at the copy; checked by
# scripts/r5_seal.py (copied_sources reads this declaration).
COPIED_FROM = {
    "generations/gen21/round5_design_20261023/probes/scripts/P1_ct_build.py":
        "afd7ae542f24df305c6f174fd3e816948b28298720d543772b685ded9d7138d0",
    "generations/gen21/round5_design_20261023/probes/scripts/P1_ct_diag.py":
        "4234aa3d51a49690d75288f4c2b4907692806d6a5955b17111fa9cc2c8f66b2d",
    "generations/gen21/round5_design_20261023/probes/scripts/P1SK_recompute.py":
        "1bbca9d66a5d03a88a07b2662a2d645364a9b1bd86f9a115fed84887ff7663de",
    "generations/gen21/round5_design_20261023/ideas2/scripts/shift_build.py":
        "3ed0db9e9a14792dd6e4de3d1397cf8423940cdad57206e4bcfe40751c6c1ccb",
    "generations/gen21/round5_design_20261023/ideas2/scripts/recipe_fixedladder.py":
        "dba2e664fc659f62cbb5a40d7232e5f9984254777e8689b5f0405c44c47676ea",
    "generations/gen21/round5_design_20261023/ideas2/scripts/recipe_build.py":
        "0b572f7b5ad6133f4846ae9d9f2ab0745db5060b2a3161ccdaf25c87a183070e",
}

LN = POP.LN
N_LN = len(LN)
GD = LN.index("Gd")
X_R = np.asarray(interp._coords("R"), float)
K_SH = 1.78                      # (1 - r) / r with r = 0.36, the same-publication roughness correlation [IR M01; EB 1.3]
K_E6 = 2.0
READS = (2, 3, 4)
SUB_M = (1, 2, 3, 5)
MAX_DRAWS = 5000
WINDOW = (-2.0, 2.0)             # M04 / M09 level window, fixed after seeing V1
SHAPE_SD_MAX = 0.02              # E6 near-copy rule (label-based; sensitivity only, IR rule 7)
LADDER = {2: ("La", "Ho"), 3: ("La", "Nd", "Ho"), 4: ("La", "Nd", "Er", "Lu"), 5: ("La", "Nd", "Gd", "Er", "Lu")}
ITEM1 = frozenset(("44012938119033a5", "5169c377777d282b"))
PUB_ACIDTYPE = "pub_b0b3296ec5"
PUB_ORNL = "pub_19783c039b"
LAB_KEY = paths.RES / "stage0" / "lab_key.csv"
CM_POOLS = ("CHEM", "XEXT", "PUB")
PB_POOLS = ("CHEM", "XEXT", "PUB", "LABX", "PUBHOM", "PUBHET")
VARIANTS_FULL = ("NB", "V1")                       # every read budget
VARIANTS_R4 = ("I1", "SD", "OFS", "QNT", "V25")    # COPY, SIB_r4, SH_r4
# duplicate sensitivity of the control pools (s0-items-1, s0-items-2): every placebo, OFF, near and acid pool with the
# offset / near-identity links of r5.dupes.offset_scan removed (``OFS_<pool>``); the primary pools keep the frozen
# bit-identity rule only (DATA_CONTRACT 2.1, 2.5).  OFS_near is the nearest-condition rule re-applied to the OFS pool.
OFS_CONTROL_POOLS = ("near", "acid", "off_all", "off_pub", "off_xext") + tuple("PB_" + p for p in PB_POOLS)
GUARD_POOLS = (("bit", False), ("NB", True), ("V1", True), ("I1", True), ("SD", False), ("OFS", False), ("QNT", False),
               ("V25", False), ("INWIN", False), ("OUTWIN", False), ("near", False), ("acid", False), ("off", False),
               ("off_all", False), ("off_pub", False), ("off_xext", False), ("NB_off", True), ("NB_off_all", True),
               ("V1_off", True)) + tuple(("PB_" + p, False) for p in PB_POOLS) \
    + tuple(("OFS_" + k, False) for k in OFS_CONTROL_POOLS)        # (pool, bit-identical copy allowed)
OFF_POOLS = ("off", "off_all", "off_pub", "off_xext", "NB_off", "NB_off_all", "V1_off", "OFS_off_all", "OFS_off_pub",
             "OFS_off_xext")                                     # residual donors (C3 OOF residuals), never templates


class LeakageError(AssertionError):
    """A pool or input that breaks the E-CT information contract."""


# ------------------------------------------------------------------------------------------------------------
# frozen predictor core (pure; no cache, bench or parquet)
# ------------------------------------------------------------------------------------------------------------
def mask(y, read_set) -> np.ndarray:
    """(14,) array with y at the read metals only, NaN elsewhere; every read metal must be finite."""
    y = np.asarray(y, float)
    rs = sorted(int(m) for m in read_set)
    out = np.full(N_LN, np.nan)
    out[rs] = y[rs]
    if not np.isfinite(out[rs]).all():
        raise ValueError(f"read metals {[LN[m] for m in rs]} not all finite")
    return out


def read_metals(ym) -> list:
    return [m for m in range(N_LN) if np.isfinite(ym[m])]


def b0_levels(ym, metals) -> np.ndarray:
    return interp.pwl_levels(np.asarray(ym, float), metals, "R")


def b1_levels(ym, metals, beta: float) -> np.ndarray:
    """B1 = PWL_R[y_S - u_S] + u with u = beta at Gd (r4.refs, exact at the reads)."""
    u = np.zeros(N_LN)
    u[GD] = float(beta)
    return interp.pwl_levels(np.asarray(ym, float) - u, metals, "R") + u


def corrected_templates(ym, metals, T) -> np.ndarray:
    """Row-wise T_j + PWL_R[y_S - T_j,S] on the target's metals (exact at the reads; NaN off the target's metals).
    Without reads the raw templates are returned on the target's metals."""
    T = np.atleast_2d(np.asarray(T, float))
    mets = np.asarray(sorted(int(m) for m in metals), int)
    out = np.full(T.shape, np.nan)
    rd = read_metals(ym)
    if not rd:
        out[:, mets] = T[:, mets]
        return out
    rd = np.asarray(rd, int)
    order = np.argsort(X_R[rd])
    xs, rdo = X_R[rd][order], rd[order]
    xm = X_R[mets]
    for i in range(len(T)):
        res = ym[rdo] - T[i, rdo]
        out[i, mets] = T[i, mets] + np.interp(xm, xs, res)        # np.interp holds the end values (as pwl_levels)
        out[i, rd] = T[i, rd] + (ym[rd] - T[i, rd])
    return out


def pair_sf(levels, pairs) -> np.ndarray:
    """log SF(A/B) = level_A - level_B for each scored pair (last axis)."""
    L = np.asarray(levels, float)
    ia = np.array([p[0] for p in pairs], int)
    ib = np.array([p[1] for p in pairs], int)
    return L[..., ia] - L[..., ib]


def weight(n: int, K: float = K_SH) -> float:
    return float(n) / (float(n) + K) if n > 0 else 0.0


def blend(a, b, w: float):
    return w * np.asarray(a, float) + (1.0 - w) * np.asarray(b, float)


@dataclass(frozen=True)
class Target:
    """One E-CT target: measured metals, scored pairs, legal reads per read budget r (masked), beta, 0-read target."""
    cell_id: str
    metals: tuple
    pairs: tuple
    reads: dict
    beta: float
    zero: np.ndarray | None = None


def make_target(cell_id: str, metals, pairs, y, read_sets: dict, beta: float, zero=None) -> Target:
    """Mask y to each read set (values elsewhere never enter).  Checks: read sets nested and inside the metals, no
    scored pair with both ends read at any budget, pairs inside the metals."""
    metals = tuple(sorted(int(m) for m in metals))
    pairs = tuple((int(a), int(b)) for a, b in pairs)
    ms = set(metals)
    if any(a not in ms or b not in ms for a, b in pairs):
        raise ValueError(f"{cell_id}: a scored pair lies outside the measured metals")
    rr = sorted(read_sets)
    for lo, hi in zip(rr[:-1], rr[1:]):
        if not set(read_sets[lo]) <= set(read_sets[hi]):
            raise ValueError(f"{cell_id}: read sets not nested ({lo} vs {hi} reads)")
    reads = {}
    for r in rr:
        rs = frozenset(int(m) for m in read_sets[r])
        if len(rs) != r or not rs <= ms:
            raise ValueError(f"{cell_id}: {r}-read set {sorted(rs)} invalid for metals {metals}")
        if any(a in rs and b in rs for a, b in pairs):
            raise ValueError(f"{cell_id}: a scored pair has both ends read at {r} reads")
        reads[r] = mask(y, rs)
    z = None if zero is None else np.asarray(zero, float).copy()
    return Target(str(cell_id), metals, pairs, reads, float(beta), z)


def b1_sf(t: Target, r: int) -> np.ndarray:
    return pair_sf(b1_levels(t.reads[r], t.metals, t.beta), t.pairs)


def b0_sf(t: Target, r: int) -> np.ndarray:
    return pair_sf(b0_levels(t.reads[r], t.metals), t.pairs)


def check_cover(t: Target, Ys) -> np.ndarray:
    Ys = np.atleast_2d(np.asarray(Ys, float))
    if len(Ys) and not np.isfinite(Ys[:, list(t.metals)]).all():
        raise ValueError(f"{t.cell_id}: a pool series does not cover the target's metals")
    return Ys


def template_sf(t: Target, r: int, Ys) -> np.ndarray:
    """(n, P) per-series pair SF: raw templates at r = 0, read-corrected at r reads.  Returned C-contiguous (one row per
    series), the layout of Corpus.target_block's row-indexed pool matrices (``pair_sf``'s column indexing alone gives
    an F-ordered array)."""
    Ys = check_cover(t, Ys)
    if len(Ys) == 0:
        return np.zeros((0, len(t.pairs)))
    ym = np.full(N_LN, np.nan) if r == 0 else t.reads[r]
    return np.ascontiguousarray(pair_sf(corrected_templates(ym, t.metals, Ys), t.pairs))


def pool_mean(S) -> np.ndarray:
    """(P,) mean over the series axis of a pool matrix S (n, P), always summed in C-contiguous row order.  numpy reduces
    axis 0 of an F-ordered matrix pairwise and of a C-ordered one row by row; the two differ by up to 1 ulp at n >= 8.
    Every sibling average of the external path (``predict_sh``) and of the in-corpus path (``Corpus.target_block``)
    goes through here, so the two agree bit for bit (r5.m03 uses the same C-ordered mean)."""
    return np.ascontiguousarray(S, dtype=float).mean(0)


def predict_sh(t: Target, Ys, *, K: float = K_SH, reads=READS) -> dict:
    """The frozen predictor and its same-read references: B1_r, B0_r, SIB_r, SH_r (r in ``reads``), COPY, SH0 (when the
    target has a 0-read target), SHK2_r4.  With an empty pool SIB / SH / COPY are NaN."""
    Ys = check_cover(t, Ys)
    n = len(Ys)
    w, w2 = weight(n, K), weight(n, K_E6)
    out = {}
    for r in reads:
        out[f"B1_r{r}"] = b1_sf(t, r)
        out[f"B0_r{r}"] = b0_sf(t, r)
    nanv = np.full(len(t.pairs), np.nan)
    if n == 0:
        out["COPY"] = out["SH0"] = out["SHK2_r4"] = nanv
        for r in reads:
            out[f"SIB_r{r}"] = out[f"SH_r{r}"] = nanv
        return out
    out["COPY"] = pool_mean(template_sf(t, 0, Ys))
    out["SH0"] = blend(out["COPY"], t.zero, w) if t.zero is not None else nanv
    for r in reads:
        s = pool_mean(template_sf(t, r, Ys))
        out[f"SIB_r{r}"] = s
        out[f"SH_r{r}"] = blend(s, out[f"B1_r{r}"], w)
    if 4 in reads:
        out["SHK2_r4"] = blend(out["SIB_r4"], out["B1_r4"], w2)
    return out


def off_sf(base, resid, *, K: float = K_SH, empty: str = "zero"):
    """B1 + sum_j r_j,p / (n_p + K) over the finite donor residuals per pair; (values, n_p).  No donor at a pair:
    ``empty='zero'`` returns the base (OFF, OFF_ALL), ``'nan'`` returns NaN (OFF_PUB, OFF_XEXT)."""
    base = np.asarray(base, float)
    R = np.atleast_2d(np.asarray(resid, float)) if len(resid) else np.zeros((0, len(base)))
    fin = np.isfinite(R)
    n_p = fin.sum(0)
    s = np.where(fin, R, 0.0).sum(0)
    val = base + s / (n_p + K)
    if empty == "nan":
        val = np.where(n_p > 0, val, np.nan)
    elif empty != "zero":
        raise ValueError(empty)
    return val, n_p


def b1_residual_sf(y_donor, metals_donor, read_set_donor, beta: float, pairs) -> np.ndarray:
    """External OFF donor residual y - B1 at ``pairs`` from the donor's own legal reads (NaN where the donor lacks a
    metal or has both ends of the pair read)."""
    y = np.asarray(y_donor, float)
    ms = set(int(m) for m in metals_donor)
    rs = frozenset(int(m) for m in read_set_donor)
    lev = b1_levels(mask(y, rs), sorted(ms), beta)
    out = np.full(len(pairs), np.nan)
    for q, (a, b) in enumerate(pairs):
        if a in ms and b in ms and not (a in rs and b in rs):
            out[q] = (y[a] - y[b]) - (lev[a] - lev[b])
    return out


def subset_index(n: int, m: int, key: str, max_draws: int = MAX_DRAWS) -> np.ndarray:
    """(D, m) index rows: every m-subset of range(n) when C(n, m) <= max_draws, else max_draws seeded random subsets
    (rng = default_rng(paths.seed_for(key)); P1's scheme of independent draws without replacement inside a draw)."""
    if m > n or m < 1:
        return np.zeros((0, max(m, 0)), int)
    if math.comb(n, m) <= max_draws:
        return np.array(list(itertools.combinations(range(n), m)), int)
    rng = np.random.default_rng(paths.seed_for(key))
    return np.array([np.sort(rng.choice(n, m, replace=False)) for _ in range(max_draws)], int)


def draw_means(S, idx) -> np.ndarray:
    """(D, P) means of the rows of S (n, P) selected by each row of idx (D, m)."""
    S = np.asarray(S, float)
    return S[idx].mean(axis=1) if len(idx) else np.zeros((0, S.shape[1]))


def expected_se(draws, y) -> np.ndarray:
    """Evaluator side: exact expectation over the draws of the squared error per pair."""
    D = np.atleast_2d(np.asarray(draws, float))
    if len(D) == 0:
        return np.full(np.shape(y), np.nan)
    return ((D - np.asarray(y, float)[None, :]) ** 2).mean(0)


def bit_identical(a, b, *, min_ident: int = DU.BIT_MIN_IDENT, min_frac: float = DU.BIT_MIN_FRAC) -> bool:
    """The frozen section-5.1 rule on one pair of series: >= 3 shared metals float64-identical and >= 50 % of shared."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sh = np.isfinite(a) & np.isfinite(b)
    ne = int((sh & (a == b)).sum())
    return ne >= min_ident and ne >= min_frac * int(sh.sum())


def guard_pool(target_id: str, target_series, pool_ids, pool_series, *, allow_bit: bool = False, name: str = "") -> None:
    """Refuse the target itself in its own pool (always), and a bit-identical copy of it (unless ``allow_bit``, for the
    NB_ / V1_ / I1_ duplicate sensitivities).  ``target_series`` is evaluator-side (the full V1 series)."""
    ids = [str(s) for s in pool_ids]
    if str(target_id) in ids:
        raise LeakageError(f"{target_id}: the target itself is in its own {name or 'pool'}")
    if not allow_bit:
        P = np.atleast_2d(np.asarray(pool_series, float)) if len(ids) else np.zeros((0, N_LN))
        for s, row in zip(ids, P):
            if bit_identical(target_series, row):
                raise LeakageError(f"{target_id}: {name or 'pool'} member {s} is bit-identical to the target")


# ------------------------------------------------------------------------------------------------------------
# external inputs (C-ECT, section 4.3 item 12)
# ------------------------------------------------------------------------------------------------------------
def external_read_sets(ext: dict, metals, reads=READS) -> dict:
    """{r: read set} for a new cell from the frozen all-C3 DOPT_Q object (r5.frozen.load_external)."""
    from . import frozen as FR
    dz = FR.design_for_metals(metals)
    seq = FR.external_sequence(ext, dz)
    return {r: FR.read_set(seq, r) for r in reads if r - 1 <= len(seq)}


def load_beta_frozen(path=paths.RES / "stage0" / "beta_frozen.json") -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))["beta_frozen"])


def external_target(cell_id: str, metals, pairs, values_at_reads: dict, ext: dict, beta: float, reads=READS) -> Target:
    """A new cell: the evaluator passes only the measured values at the frozen policy's read metals."""
    rsets = external_read_sets(ext, metals, reads)
    y = np.full(N_LN, np.nan)
    need = set().union(*rsets.values())
    for m in need:
        if m not in values_at_reads:
            raise ValueError(f"{cell_id}: read metal {LN[m]} has no value")
        y[m] = float(values_at_reads[m])
    return make_target(cell_id, metals, pairs, y, rsets, beta)


# ------------------------------------------------------------------------------------------------------------
# label-free read policies for the E-REC annex (recipe-2 frozen ladder, legal-HULL fallback)
# ------------------------------------------------------------------------------------------------------------
def legal_sequence(dz, order) -> bool:
    """True when adding the metals of ``order`` one by one stays inside gen20k.design.legal_metals."""
    cur = frozenset()
    for m in order:
        if m not in DS.legal_metals(dz, cur):
            return False
        cur = cur | {m}
    return True


def hull_legal(dz, r: int):
    """Lightest and heaviest measured metal, then the farthest metal in -CN8 radius (max-min distance; ties to the lower
    Z), every addition restricted to gen20k.design.legal_metals; None when r legal reads cannot be reached."""
    ms = sorted(dz.cell.metals)
    if len(ms) < r:
        return None
    first = [ms[0], ms[-1]]
    if not legal_sequence(dz, first):
        return None
    cur = list(first)
    while len(cur) < r:
        leg = [m for m in DS.legal_metals(dz, frozenset(cur)) if m not in cur]
        if not leg:
            return None
        d = [min(abs(X_R[m] - X_R[o]) for o in cur) for m in leg]
        best = max(d)
        cur.append(min(m for m, v in zip(leg, d) if v == best))
    return frozenset(cur[:r])


def ladder_reads(dz, r: int):
    """(read set, source) of the frozen ladder at r reads ('ladder' when its metals are measured and legal in ladder
    order, else 'hull'); (None, 'none') when neither reaches r legal reads."""
    lad = [LN.index(m) for m in LADDER[r]]
    if set(lad) <= set(dz.cell.metals) and legal_sequence(dz, lad):
        return frozenset(lad), "ladder"
    h = hull_legal(dz, r)
    return (h, "hull") if h is not None else (None, "none")


# ------------------------------------------------------------------------------------------------------------
# in-corpus population, variants and the row builder
# ------------------------------------------------------------------------------------------------------------
def shape_sd(yt, ys, mets) -> float:
    """E6 near-copy statistic: sd of (Y_t - Y_s) over the target's metals after an offset and a radius slope."""
    d = np.asarray(yt, float)[mets] - np.asarray(ys, float)[mets]
    A = np.c_[np.ones(len(mets)), X_R[mets]]
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    return float(np.std(d - A @ coef, ddof=0))


def interp_template(Ys, x, xt) -> np.ndarray:
    """shift_build.interp_template (flat beyond the range): per-metal interpolation in x between levels, cells at
    one x (rounded to 4 decimals) averaged."""
    xr = np.round(np.asarray(x, float), 4)
    xu = np.unique(xr)
    Yu = np.array([np.asarray(Ys, float)[xr == v].mean(0) for v in xu])
    if len(xu) == 1:
        return Yu[0]
    return np.array([np.interp(xt, xu, Yu[:, m]) for m in range(Yu.shape[1])])


def load_lab_map(path=LAB_KEY) -> dict:
    """{publication_id: (lab_conservative, lab_liberal)} from the frozen PROVISIONAL laboratory key (empty if absent)."""
    if not path.exists():
        return {}
    k = pd.read_csv(path)
    k = k[k.unit_type == "corpus"]
    out = {}
    for r in k.itertuples(index=False):
        if isinstance(r.publication_ids, str):
            for p in r.publication_ids.replace("|", ";").split(";"):
                if p.strip():
                    out[p.strip()] = (str(r.lab_conservative), str(r.lab_liberal))
    return out


class Corpus:
    """Bench, metadata pools (r5.population), variant pools and label-free per-target constants."""

    def __init__(self, bench=None, *, lab_key=LAB_KEY, log=print):
        from gen20k import data as D
        from gen20k.data import cell_view
        t0 = time.time()
        self.bench = bench if bench is not None else D.load_bench()
        f = self.bench.frame.reset_index(drop=True)
        self.Y = np.asarray(self.bench.Y, float)
        self.ids = f["cell_id"].astype(str).tolist()
        self.pos = {c: i for i, c in enumerate(self.ids)}
        self.meta = POP.metadata_frame(f)
        self.finite = POP.finite_from_metals(self.meta)
        if not np.array_equal(self.finite, np.isfinite(self.Y)):
            raise AssertionError("metals metadata disagrees with the finite pattern of V1")
        self.pub = self.meta["publication_id"].astype(str).to_numpy()
        self.ext = self.meta["extractant"].astype(str).to_numpy()
        self.chem = self.meta["chemotype"].astype(str).to_numpy()
        self.c3 = POP.c3_cells_from_ref()
        self.c3set = set(self.c3)
        self.v2 = POP.v2_lists()
        self.v2drop = POP.v2drop_set()
        self.bit_table = DU.bit_identity(self.Y, self.ids)
        self.bit_links = DU.flagged_links(self.bit_table)
        self.bits = {frozenset(x) for x in self.bit_links}
        self.pools = POP.build_pools(self.meta, self.finite, self.c3, self.v2drop, self.bit_links)
        self.strata = POP.strata(self.pools)
        self.cm = POP.CondMeta(self.meta)
        self.lab = load_lab_map(lab_key)
        self.lab_key_found = bool(self.lab)
        self.acid = np.log10(self.meta["cond__acid_concentration_M"].astype(float).to_numpy())
        self.dz = {t: DS.make_design(cell_view(self.bench, self.pos[t]), "adjacent", "metal") for t in self.c3}
        self._ladder = {}
        self._guarded = set()
        self._variants()
        self.seconds = round(time.time() - t0, 1)
        log(f"  corpus: {len(self.ids)} bench cells, {len(self.c3)} C3 targets, {len(self.bit_links)} bit-flagged links, "
            f"{len(self.ofs_links)} offset / near-id links, {len(self.quant_cells)} quantised cells, "
            f"lab key {'found' if self.lab_key_found else 'MISSING'} ({self.seconds}s)")

    # ---- variants -------------------------------------------------------------------------------------------
    def _variants(self) -> None:
        Y, pos = self.Y, self.pos
        ofs = DU.offset_scan(Y, self.ids, self.bit_links)
        self.ofs_links = {frozenset((a, b)) for a, b in zip(ofs.cell_a, ofs.cell_b)}
        q = DU.quant_scan(Y, self.ids)
        self.quant_cells = set(q.cell_id[q.Q_SMALLINT | q.Q_COARSE].astype(str))
        Y25 = Y.copy()
        self.floor = {}
        for c, mets in self.v2["V2-5"].items():
            idx = [LN.index(m) for m in mets]
            Y25[pos[c], idx] = np.nan
            self.floor[c] = frozenset(idx)
        pools25 = POP.build_pools(self.meta, np.isfinite(Y25), self.c3, self.v2drop, self.bit_links)
        lo, hi = WINDOW
        self.var = {}
        for t, P in self.pools.items():
            i = pos[t]
            mets = np.where(self.finite[i])[0]
            bit = P["bit"]
            v = {"bit": bit, "NB": P["clean"], "V1": P["v1"],
                 "I1": [s for s in P["clean"] if frozenset((t, s)) != ITEM1],
                 "SD": [s for s in bit if shape_sd(Y[i], Y[pos[s]], mets) >= SHAPE_SD_MAX],
                 "OFS": [s for s in bit if frozenset((t, s)) not in self.ofs_links],
                 "QNT": [] if t in self.quant_cells else [s for s in bit if s not in self.quant_cells],
                 "V25": pools25[t]["bit"],
                 "INWIN": [s for s in bit if np.all((Y[pos[s], mets] >= lo) & (Y[pos[s], mets] <= hi))]}
            v["OUTWIN"] = [s for s in bit if s not in set(v["INWIN"])]
            v["near"] = P["near"]
            v["acid"] = [s for s in bit if set(P["sib_fams"][s].split(",")) - POP.FAMS_TRIVIAL == {"acid_conc"}
                         and np.isfinite(self.acid[i]) and np.isfinite(self.acid[pos[s]])]
            v["off"], v["off_all"], v["off_pub"], v["off_xext"] = P["off_sibs"], P["off_all"], P["off_pub"], P["off_xext"]
            v["NB_off"], v["NB_off_all"] = P["off_sibs_clean"], P["off_all_clean"]
            v["V1_off"] = [s for s in P["v1"] if s in self.c3set]
            v["PB_CHEM"], v["PB_XEXT"], v["PB_PUB"] = P["plc_chem"], P["plc_xext"], P["plc_pub"]
            v["PB_PUBHOM"] = [s for s in P["plc_pub"] if self.chem[pos[s]] == self.chem[i]]
            v["PB_PUBHET"] = [s for s in P["plc_pub"] if self.chem[pos[s]] != self.chem[i]]
            v["PB_LABX"] = self._labx(t)
            for k in OFS_CONTROL_POOLS:
                if k == "near":
                    v["OFS_near"] = self._near_of(t, v["OFS"])
                else:
                    v["OFS_" + k] = [s for s in v[k] if frozenset((t, s)) not in self.ofs_links]
            self.var[t] = v

    def _near_of(self, t: str, pool) -> list:
        """population.build_pools' nearest-condition rule (cond__ distance, ties averaged) on ``pool``."""
        if not pool:
            return []
        i = self.pos[t]
        dd = np.array([self.cm.cond_dist(i, self.pos[s]) for s in pool])
        return [pool[q] for q in np.where(np.isclose(dd, dd.min(), atol=1e-9))[0]]

    def ladder(self, t: str, r: int):
        """Memoised ``ladder_reads`` of target t (label-free, fixed per cell)."""
        if (t, r) not in self._ladder:
            self._ladder[(t, r)] = ladder_reads(self.dz[t], r)
        return self._ladder[(t, r)]

    def _labx(self, t: str) -> list:
        """Same extractant, same laboratory (conservative key), other publication; covering, non-V2, no bit link."""
        i = self.pos[t]
        lab = self.lab.get(self.pub[i], (None, None))[0]
        if lab is None:
            return []
        cover = np.all(self.finite[:, self.finite[i]], axis=1)
        out = []
        for j in np.where((self.ext == self.ext[i]) & (self.pub != self.pub[i]) & cover)[0]:
            s = self.ids[j]
            if s != t and s not in self.v2drop and frozenset((t, s)) not in self.bits \
                    and self.lab.get(self.pub[j], (None, None))[0] == lab:
                out.append(s)
        return out

    # ---- label-free constants per target ---------------------------------------------------------------------
    def target_table(self) -> pd.DataFrame:
        """One metadata row per C3 target: pool sizes per variant, strata, change class, laboratory, acid position
        flags (metadata), quantisation / floor flags.  No label of the target, no score."""
        S = {k: set(v) for k, v in self.strata.items()}
        rows = []
        for t in self.c3:
            i, P, v = self.pos[t], self.pools[t], self.var[t]
            r = {"cell_id": t, "publication": P["publication"], "extractant": P["extractant"], "chemotype": P["chemotype"],
                 "c5": bool(P["c5"]), "t_v2drop": bool(P["t_v2drop"]), "cls": P["cls"], "cls_clean": P["cls_clean"],
                 "lab_cons": self.lab.get(self.pub[i], ("NA", "NA"))[0], "lab_lib": self.lab.get(self.pub[i], ("NA", "NA"))[1],
                 "t_quant": t in self.quant_cells, "t_floor": ",".join(LN[m] for m in sorted(self.floor.get(t, ())))}
            r["n_sib"] = len(v["bit"])
            for k in ("NB", "V1", "I1", "SD", "OFS", "QNT", "V25", "INWIN", "OUTWIN", "near", "acid", "off", "off_all",
                      "off_pub", "off_xext", "NB_off", "V1_off") + tuple("PB_" + p for p in PB_POOLS) \
                    + tuple("OFS_" + k for k in OFS_CONTROL_POOLS):
                r["n_" + k] = len(v[k])
            r["n_bit_removed"] = len(P["bit_removed"])
            if v["acid"]:                                # shift-1 acid position (metadata: acid concentrations only)
                xa, xt = self.acid[[self.pos[s] for s in v["acid"]]], self.acid[i]
                r["dx_mean_abs"], r["dx_near"] = abs(float(xt - xa.mean())), float(np.min(np.abs(xa - xt)))
                r["acid_position"] = "below" if xt < xa.min() - 1e-9 else ("above" if xt > xa.max() + 1e-9 else "interior")
            else:
                r["dx_mean_abs"], r["dx_near"], r["acid_position"] = np.nan, np.nan, ""
            for k in S:
                r["in_" + k] = t in S[k]
            nc = not r["c5"]
            for k in VARIANTS_R4:
                extra = (not r["t_quant"]) if k == "QNT" else True
                r[f"in_PRIMARY_{k}"] = bool(r["n_" + k] >= 2 and nc and extra and not r["t_v2drop"])
            rows.append(r)
        return pd.DataFrame(rows)

    def links_table(self) -> pd.DataFrame:
        """Every V1 sibling link with CLEAN / BIT status and the sensitivity flags (shape sd, offset / near-id)."""
        L = POP.links_table(self.pools)
        sd, ofs = [], []
        for r in L.itertuples(index=False):
            i, j = self.pos[r.target], self.pos[r.sibling]
            sd.append(shape_sd(self.Y[i], self.Y[j], np.where(self.finite[i])[0]))
            ofs.append(frozenset((r.target, r.sibling)) in self.ofs_links)
        L["shape_sd"] = sd
        L["shape_sd_nearcopy"] = L.shape_sd < SHAPE_SD_MAX
        L["offset_or_nearid"] = ofs
        L["item1"] = [frozenset((a, b)) == ITEM1 for a, b in zip(L.target, L.sibling)]
        return L

    # ---- one design ------------------------------------------------------------------------------------------
    def design_ctx(self, design: str) -> dict:
        """REF rows (r5.common.load_rows), OOF residuals, sibling read sets / folds, fold betas."""
        from . import common as CM
        R = CM.load_rows(design)
        if set(R.cell_id) != self.c3set:
            raise AssertionError(f"{design}: REF cells differ from the C3 targets")
        fitted = json.loads((paths.R4RES / "REF" / f"fitted_{design}.json").read_text(encoding="utf-8"))
        betas = {tuple(int(x) for x in k.split("_")): float(v["ref:B1"]["coef"][0]) for k, v in fitted.items()
                 if re.fullmatch(r"\d+_\d+", k)}
        if len(betas) != 25:
            raise AssertionError(f"{design}: expected 25 fold betas, found {len(betas)}")
        res = dict(zip(zip(R.split_seed, R.cell_id, R.A, R.B), (R.y - R.B1).to_numpy(float)))
        meta = {(s, c): (rs, fo) for s, c, rs, fo in
                R.drop_duplicates(["split_seed", "cell_id"])[["split_seed", "cell_id", "read_set", "fold"]].itertuples(index=False)}
        groups = {k: g.index.to_numpy() for k, g in R.groupby(["split_seed", "cell_id"], sort=False)}
        return {"design": design, "rows": R, "betas": betas, "res": res, "meta": meta, "groups": groups}

    def target_block(self, ctx: dict, cell: dict, seed: int, fold: int, *, Yv=None, y_score=None, guard: bool = True,
                     inject=None) -> dict:
        """Every arm for one (seed, fold, target): {column: (P,) array}.  ``Yv`` is the value matrix the reads and the
        sibling series come from (V1 by default; the intervention check passes a poisoned copy); ``y_score`` the
        evaluator's labels for the expectation arms (E_*); ``inject`` ({pool: [ids]}) exists for the leakage positive
        controls only."""
        t = cell["cell_id"]
        i, P, V = self.pos[t], self.pools[t], dict(self.var[t])
        if inject:
            for k, extra in inject.items():
                V[k] = list(V[k]) + list(extra)
        Yv = self.Y if Yv is None else Yv
        R = ctx["rows"]
        gi = ctx["groups"][(seed, t)]
        g = R.loc[gi]
        if not (g.fold == fold).all():
            raise AssertionError(f"{t}: parquet fold differs from the cache fold")
        pairs = tuple(zip(g.ia.astype(int), g.ib.astype(int)))
        seq = cell["seqs"]["DOPT_Q"]
        rsets = {r: frozenset(m for p in seq[:r - 1] for m in p) for r in READS}
        if len(seq) >= 4:
            rsets[5] = frozenset(m for p in seq[:4] for m in p)
        beta = ctx["betas"][(seed, fold)]
        T = make_target(t, cell["metals"], pairs, Yv[i], rsets, beta, zero=g["G14@ZERO"].to_numpy(float))
        if guard and (inject or t not in self._guarded):
            for k, allow in GUARD_POOLS:
                ids = V[k]
                guard_pool(t, self.Y[i], ids, self.Y[[self.pos[s] for s in ids]] if ids else np.zeros((0, N_LN)),
                           allow_bit=allow, name=k)
            if not inject:
                self._guarded.add(t)          # pools and V1 are fixed per target: one guard pass per target
        nP = len(pairs)
        nanv = np.full(nP, np.nan)
        out = {}
        b1 = {r: b1_sf(T, r) for r in READS}
        for r in READS:
            out[f"B1_r{r}"] = b1[r]
            out[f"B0_r{r}"] = b0_sf(T, r)
        out["NP_DOPTQ_r5"] = b1_sf(T, 5) if 5 in rsets else nanv
        # per-series SF for the union of every pool (computed once per read budget).  A V2-5 target's V25 pool covers
        # only its non-floor metals, so it is computed apart on those metals (pairs touching a floor metal stay NaN).
        fl = self.floor.get(t, frozenset())
        skip = set(OFF_POOLS) | ({"V25"} if fl else set())
        uni = sorted({s for k in V if k not in skip for s in V[k]})
        ui = {s: q for q, s in enumerate(uni)}
        Yu = Yv[[self.pos[s] for s in uni]] if uni else np.zeros((0, N_LN))
        S = {r: template_sf(T, r, Yu) for r in (0,) + READS}
        S25 = {}
        if fl and V["V25"] and not (rsets[4] & fl):
            T25 = Target(t, tuple(m for m in T.metals if m not in fl), T.pairs, T.reads, T.beta, T.zero)
            Y25 = Yv[[self.pos[s] for s in V["V25"]]]
            S25 = {r: template_sf(T25, r, Y25) for r in (0, 4)}

        def M(ids, r, key=None):
            if key == "V25" and fl:
                return S25[r] if S25 else np.zeros((0, nP))
            return S[r][[ui[s] for s in ids]] if ids else np.zeros((0, nP))
        # point arms per pool variant
        for pre, key, rr in (("", "bit", READS), ("NB_", "NB", READS), ("V1_", "V1", READS)) + \
                tuple((k + "_", k, (4,)) for k in VARIANTS_R4 + ("INWIN", "OUTWIN")):
            ids = V[key]
            n = len(ids)
            w = weight(n)
            if n == 0 or len(M(ids, 0, key)) == 0:
                for col in ["COPY", "SH0", "SHK2_r4"] + [f"SIB_r{r}" for r in rr] + [f"SH_r{r}" for r in rr]:
                    out[pre + col] = nanv
                continue
            out[pre + "COPY"] = pool_mean(M(ids, 0, key))
            out[pre + "SH0"] = blend(out[pre + "COPY"], T.zero, w)
            for r in rr:
                s = pool_mean(M(ids, r, key))
                out[f"{pre}SIB_r{r}"] = s
                out[f"{pre}SH_r{r}"] = blend(s, b1[r], w)
            out[pre + "SHK2_r4"] = blend(out[pre + "SIB_r4"], b1[4], weight(n, K_E6))
        n = len(V["bit"])
        w = weight(n)
        # NEAR
        nn = len(V["near"])
        if nn:
            out["NEAR_r4"] = pool_mean(M(V["near"], 4))
            out["NEARSH_r4"] = blend(out["NEAR_r4"], b1[4], weight(nn))
        else:
            out["NEAR_r4"] = out["NEARSH_r4"] = nanv
        nn_o = len(V["OFS_near"])                     # duplicate sensitivity (s0-items-2): NEAR on the OFS pool
        if nn_o:
            out["OFS_NEAR_r4"] = pool_mean(M(V["OFS_near"], 4))
            out["OFS_NEARSH_r4"] = blend(out["OFS_NEAR_r4"], b1[4], weight(nn_o))
        else:
            out["OFS_NEAR_r4"] = out["OFS_NEARSH_r4"] = nanv
        # OFF family (OOF residuals of C3 donors, same seed and design; leave-target-out by construction).  OFF and
        # OFF_ALL (and their NB_ / V1_ versions) exist only when the target has a sibling in that pool (P1).
        for col, key, empty, need in (("OFF", "off", "zero", "bit"), ("OFF_ALL", "off_all", "zero", "bit"),
                                      ("OFF_PUB", "off_pub", "nan", None), ("OFF_XEXT", "off_xext", "nan", None),
                                      ("NB_OFF", "NB_off", "zero", "NB"), ("NB_OFF_ALL", "NB_off_all", "zero", "NB"),
                                      ("V1_OFF", "V1_off", "zero", "V1")):
            donors = V[key]
            if t in donors:
                raise LeakageError(f"{t}: OFF donor pool {key} holds the target")
            if need is not None and not V[need]:
                out[col], out[col + "_np"] = nanv, np.zeros(nP)
                continue
            Rm = np.array([[ctx["res"].get((seed, d, LN[a], LN[b]), np.nan) for (a, b) in pairs] for d in donors]) \
                if donors else np.zeros((0, nP))
            out[col], out[col + "_np"] = off_sf(b1[4], Rm, empty=empty)
        # duplicate sensitivity (s0-items-1): OFF pools with offset / near-identity links removed; OFS_OFF_ALL exists
        # only when the target has an OFS sibling (as OFF_ALL with a BIT sibling)
        n_o = len(V["OFS"])
        w_o = weight(n_o)
        for col, key, empty, need in (("OFS_OFF_ALL", "OFS_off_all", "zero", "OFS"),
                                      ("OFS_OFF_PUB", "OFS_off_pub", "nan", None),
                                      ("OFS_OFF_XEXT", "OFS_off_xext", "nan", None)):
            donors = V[key]
            if t in donors:
                raise LeakageError(f"{t}: OFF donor pool {key} holds the target")
            if need is not None and not V[need]:
                out[col], out[col + "_np"] = nanv, np.zeros(nP)
                continue
            Rm = np.array([[ctx["res"].get((seed, d, LN[a], LN[b]), np.nan) for (a, b) in pairs] for d in donors]) \
                if donors else np.zeros((0, nP))
            out[col], out[col + "_np"] = off_sf(b1[4], Rm, empty=empty)
        # placebos at the target's own w
        for p in PB_POOLS:
            ids = V["PB_" + p]
            if ids and n:
                raw = pool_mean(M(ids, 4))
                out[f"PB_{p}_raw"], out[f"PB_{p}"] = raw, blend(raw, b1[4], w)
            else:
                out[f"PB_{p}_raw"] = out[f"PB_{p}"] = nanv
            ids = V["OFS_PB_" + p]                    # duplicate sensitivity at the OFS pool's own w
            if ids and n_o:
                raw = pool_mean(M(ids, 4))
                out[f"OFS_PB_{p}_raw"], out[f"OFS_PB_{p}"] = raw, blend(raw, b1[4], w_o)
            else:
                out[f"OFS_PB_{p}_raw"] = out[f"OFS_PB_{p}"] = nanv
        # expectation arms (evaluator side: they need labels to score; predictions are the draws)
        ys = y_score
        if ys is None:
            ys = Yv[i, [a for a, _ in pairs]] - Yv[i, [b for _, b in pairs]]
        ys = np.asarray(ys, float)
        bitS = {r: M(V["bit"], r) for r in (0,) + READS}
        for r in (0,) + READS:
            out[f"E_RAND1_raw_r{r}"] = expected_se(bitS[r], ys) if n else nanv
            if r:
                out[f"E_RAND1_sh_r{r}"] = expected_se(blend(bitS[r], b1[r], weight(1)), ys) if n else nanv
        pidx = np.array(list(itertools.combinations(range(n), 2)), int) if n >= 2 else np.zeros((0, 2), int)
        for r in (0, 2, 4):
            dr = draw_means(bitS[r], pidx) if n >= 2 else None
            out[f"E_RAND2_raw_r{r}"] = expected_se(dr, ys) if n >= 2 else nanv
            if r:
                out[f"E_RAND2_sh_r{r}"] = expected_se(blend(dr, b1[r], weight(2)), ys) if n >= 2 else nanv
        for m in SUB_M:
            if n >= 5:
                idx = subset_index(n, m, f"ECT|SUB|{t}|m{m}")
                dr = draw_means(bitS[4], idx)
                out[f"E_SUB{m}_raw_r4"] = expected_se(dr, ys)
                out[f"E_SUB{m}_sh_r4"] = expected_se(blend(dr, b1[4], weight(m)), ys)
            else:
                out[f"E_SUB{m}_raw_r4"] = out[f"E_SUB{m}_sh_r4"] = nanv
        for p in CM_POOLS:
            ids = V["PB_" + p]
            if n and len(ids) >= n:
                idx = subset_index(len(ids), n, f"ECT|CM_{p}|{t}")
                dr = draw_means(M(ids, 4), idx)
                out[f"E_PB_{p}_CM_raw"] = expected_se(dr, ys)
                out[f"E_PB_{p}_CM"] = expected_se(blend(dr, b1[4], w), ys)
            else:
                out[f"E_PB_{p}_CM_raw"] = out[f"E_PB_{p}_CM"] = nanv
            ids = V["OFS_PB_" + p]                    # duplicate sensitivity: count-matched to the OFS pool
            if n_o and len(ids) >= n_o:
                idx = subset_index(len(ids), n_o, f"ECT|CM_OFS_{p}|{t}")
                dr = draw_means(M(ids, 4), idx)
                out[f"E_OFS_PB_{p}_CM_raw"] = expected_se(dr, ys)
                out[f"E_OFS_PB_{p}_CM"] = expected_se(blend(dr, b1[4], w_o), ys)
            else:
                out[f"E_OFS_PB_{p}_CM_raw"] = out[f"E_OFS_PB_{p}_CM"] = nanv
        # shift-1 annex (acid-only siblings; 0 reads)
        acid = V["acid"]
        if acid:
            xa = self.acid[[self.pos[s] for s in acid]]
            out["ACOPY_r0"] = pool_mean(M(acid, 0))
            out["AINT_r0"] = pair_sf(interp_template(Yv[[self.pos[s] for s in acid]], xa, self.acid[i]), pairs)
        else:
            out["ACOPY_r0"] = out["AINT_r0"] = nanv
        acid_o = V["OFS_acid"]                         # duplicate sensitivity (s0-items-2)
        if acid_o:
            xa = self.acid[[self.pos[s] for s in acid_o]]
            out["OFS_ACOPY_r0"] = pool_mean(M(acid_o, 0))
            out["OFS_AINT_r0"] = pair_sf(interp_template(Yv[[self.pos[s] for s in acid_o]], xa, self.acid[i]), pairs)
        else:
            out["OFS_ACOPY_r0"] = out["OFS_AINT_r0"] = nanv
        # E-REC no-prior ladder (label-free; fixed per cell; B1 form with the fold beta)
        for r in (2, 3, 4, 5):
            rs, src = self.ladder(t, r)
            if rs is None:
                out[f"NP_LAD_r{r}"] = nanv
                continue
            if any(a in rs and b in rs for a, b in pairs):
                raise AssertionError(f"{t}: ladder read set reads both ends of a scored pair")
            out[f"NP_LAD_r{r}"] = pair_sf(b1_levels(mask(Yv[i], rs), T.metals, beta), pairs)
        return out

    def target_consts(self, ctx: dict, cell: dict, seed: int, fold: int) -> dict:
        """Per-row label-free flags of one target group: read-alignment rows (every BIT sibling a C3 OFF donor read at
        the target's read set in the same fold, with a row for the pair), V2-5 row validity, read sets and sources."""
        t = cell["cell_id"]
        g = ctx["rows"].loc[ctx["groups"][(seed, t)]]
        V = self.var[t]
        seq = cell["seqs"]["DOPT_Q"]
        r4 = frozenset(m for p in seq[:3] for m in p)
        rs_t, fo_t = ctx["meta"][(seed, t)]
        geo = []
        for a, b in zip(g.A, g.B):
            ok = bool(V["bit"])
            for s in V["bit"]:
                if s not in self.c3set or (seed, s) not in ctx["meta"]:
                    ok = False
                    break
                rs, fo = ctx["meta"][(seed, s)]
                if rs != rs_t or fo != fo_t or (seed, s, a, b) not in ctx["res"]:
                    ok = False
                    break
            geo.append(ok)
        fl = self.floor.get(t, frozenset())
        v25 = [not (r4 & fl) and LN.index(a) not in fl and LN.index(b) not in fl for a, b in zip(g.A, g.B)]
        srcs = {f"np_lad_src_r{r}": np.array([self.ladder(t, r)[1]] * len(g), dtype=object) for r in (2, 3, 4, 5)}
        return {"geom_same": np.array(geo, bool), "v25_row_ok": np.array(v25, bool), **srcs}

    def build_rows(self, design: str, *, y_mode: str = "labels", log=print) -> tuple[pd.DataFrame, dict]:
        """Every arm on every round-4 REF row of ``design`` (k = 3, 4 reads).  ``y_mode='labels'`` scores the
        expectation arms against the true labels; ``'surrogate'`` (technical smoke) against seeded noise, so no real
        error is computed.  Returns (rows, checks)."""
        from g21 import cache as GC
        t0 = time.time()
        ctx = self.design_ctx(design)
        R = ctx["rows"]
        tt = self.target_table().set_index("cell_id")
        cols, checks = {}, {"design": design, "rows": int(len(R)), "groups": 0, "b1_maxabs": 0.0, "y_maxabs": 0.0,
                            "readset_mismatch": 0, "nested_reads_ok": True, "legal_steps_ok": True,
                            "both_ends_read_rows": int((R.a_read & R.b_read).sum())}
        idx_all = []
        rng = np.random.default_rng(paths.seed_for(f"ECT|SMOKE_SURROGATE|{design}"))
        for seed in paths.DEV_SEEDS:
            for fold in range(5):
                rec = GC.load(design, seed, fold)
                for cell in rec["outer"]["cells"]:
                    t = cell["cell_id"]
                    if (seed, t) not in ctx["groups"]:
                        continue
                    gi = ctx["groups"][(seed, t)]
                    g = R.loc[gi]
                    i = self.pos[t]
                    y_lab = self.Y[i, g.ia.astype(int)] - self.Y[i, g.ib.astype(int)]
                    checks["y_maxabs"] = max(checks["y_maxabs"], float(np.max(np.abs(y_lab - g.y.to_numpy(float)))))
                    ys = y_lab if y_mode == "labels" else rng.normal(0.0, 0.2, len(gi))
                    blk = self.target_block(ctx, cell, seed, fold, y_score=ys)
                    blk.update(self.target_consts(ctx, cell, seed, fold))
                    checks["b1_maxabs"] = max(checks["b1_maxabs"], float(np.max(np.abs(blk["B1_r4"] - g.B1.to_numpy(float)))))
                    seq = cell["seqs"]["DOPT_Q"]
                    rs = ",".join(LN[m] for m in sorted({m for p in seq[:3] for m in p}))
                    checks["readset_mismatch"] += int(rs != str(g.read_set.iloc[0]))
                    sets = [frozenset(m for p in seq[:k] for m in p) for k in range(1, len(seq) + 1)]
                    checks["nested_reads_ok"] &= all(a < b for a, b in zip(sets[:-1], sets[1:]))
                    dz = self.dz[t]
                    for k in range(1, min(len(seq), 4)):
                        new = set(seq[k]) - set(sets[k - 1])
                        if not new <= set(DS.legal_metals(dz, sets[k - 1])):
                            checks["legal_steps_ok"] = False
                    for k, v in blk.items():
                        cols.setdefault(k, []).append((len(idx_all), np.asarray(v)))
                    idx_all.append(gi)
                    checks["groups"] += 1
        order = np.concatenate(idx_all)
        n = len(order)
        lens = [len(x) for x in idx_all]
        offs = np.concatenate([[0], np.cumsum(lens)])
        out = R.loc[order].reset_index(drop=True).copy()
        for k, parts in cols.items():
            proto = parts[0][1]
            arr = np.full(n, np.nan) if proto.dtype.kind in "fiu" else np.empty(n, dtype=object)
            if proto.dtype.kind == "b":
                arr = np.zeros(n, bool)
            filled = np.zeros(n, bool)
            for gidx, v in parts:
                a, b = offs[gidx], offs[gidx + 1]
                arr[a:b] = v
                filled[a:b] = True
            if not filled.all():
                raise AssertionError(f"column {k} missing for some targets")
            out[k] = arr
        out = out.rename(columns={"G14@ZERO": "G14Z"})
        keep_t = [c for c in tt.columns if c not in ("publication", "extractant", "chemotype")]
        out = out.join(tt[keep_t], on="cell_id")
        out["design"] = design
        out = out.sort_values(["split_seed", "fold", "cell_id", "A", "B"]).reset_index(drop=True)
        if out.duplicated(["split_seed", "fold", "cell_id", "A", "B"]).any() or len(out) != len(R):
            raise AssertionError("row set changed while building")
        checks["seconds"] = round(time.time() - t0, 1)
        checks["y_mode"] = y_mode
        log(f"  {design}: {checks['groups']} target groups, {len(out)} rows, {out.shape[1]} columns ({checks['seconds']}s)")
        return out, checks

    # ---- leakage (section 8 item 4) ----------------------------------------------------------------------------
    def intervention(self, design: str, *, seeds=None, y_mode: str = "labels", log=print) -> dict:
        """Prediction intervention on every target of ``seeds`` (default all dev seeds): (1) every metal outside the
        union of all read sets the arms use replaced by noise in the target's own row: every prediction column must be
        bit-identical; (2) every metal outside the 4-read DOPT_Q set replaced: every column except the larger-budget
        NP_* columns bit-identical.  Positive controls (must be caught): the target in its own BIT pool (guard raises),
        a bit-identical copy of the target in its pool (guard raises), the target in its own pool with the guard off
        (the intervention detects it)."""
        from g21 import cache as GC
        ctx = self.design_ctx(design)
        seeds = tuple(seeds) if seeds is not None else tuple(paths.DEV_SEEDS)
        rep = {"design": design, "seeds": list(seeds), "targets": 0, "pass1_diff_targets": 0, "pass2_diff_targets": 0,
               "pass1_cols_diff": [], "pass2_cols_diff": [], "pc_self_guard_caught": 0, "pc_copy_guard_caught": 0,
               "pc_self_intervention_caught": 0, "pc_targets": 0}
        rng = np.random.default_rng(paths.seed_for(f"ECT|INTERVENTION|{design}"))
        skip = {"E_"}
        for seed in seeds:
            for fold in range(5):
                rec = GC.load(design, seed, fold)
                for cell in rec["outer"]["cells"]:
                    t = cell["cell_id"]
                    if (seed, t) not in ctx["groups"]:
                        continue
                    i = self.pos[t]
                    g = ctx["rows"].loc[ctx["groups"][(seed, t)]]
                    ys = (self.Y[i, g.ia.astype(int)] - self.Y[i, g.ib.astype(int)]) if y_mode == "labels" \
                        else rng.normal(0.0, 0.2, len(g))       # E_* columns are not compared; smoke never scores
                    base = self.target_block(ctx, cell, seed, fold, y_score=ys)
                    seq = cell["seqs"]["DOPT_Q"]
                    used = {m for p in seq[:4] for m in p}
                    for r in (2, 3, 4, 5):
                        rs, _ = ladder_reads(self.dz[t], r)
                        used |= set(rs or ())
                    r4 = {m for p in seq[:3] for m in p}
                    for pas, keep in ((1, used), (2, r4)):
                        Yp = self.Y.copy()
                        un = [m for m in cell["metals"] if m not in keep]
                        Yp[i, un] = rng.normal(0.0, 3.0, len(un))
                        b2 = self.target_block(ctx, cell, seed, fold, Yv=Yp, y_score=ys)
                        bad = [k for k in base if not any(k.startswith(s) for s in skip)
                               and not (pas == 2 and k.startswith("NP_"))
                               and not np.array_equal(np.asarray(base[k]), np.asarray(b2[k]), equal_nan=True)]
                        if bad:
                            rep[f"pass{pas}_diff_targets"] += 1
                            rep[f"pass{pas}_cols_diff"] = sorted(set(rep[f"pass{pas}_cols_diff"]) | set(bad))
                    rep["targets"] += 1
                    if self.var[t]["bit"]:
                        rep["pc_targets"] += 1
                        try:
                            self.target_block(ctx, cell, seed, fold, y_score=ys, inject={"bit": [t]})
                        except LeakageError:
                            rep["pc_self_guard_caught"] += 1
                        try:
                            guard_pool(t, self.Y[i], ["copy_of_" + t], self.Y[i][None, :], name="bit")
                        except LeakageError:
                            rep["pc_copy_guard_caught"] += 1
                        Yp = self.Y.copy()
                        un = [m for m in cell["metals"] if m not in r4]
                        Yp[i, un] = rng.normal(0.0, 3.0, len(un))
                        a1 = self.target_block(ctx, cell, seed, fold, y_score=ys, inject={"bit": [t]}, guard=False)
                        a2 = self.target_block(ctx, cell, seed, fold, Yv=Yp, y_score=ys, inject={"bit": [t]}, guard=False)
                        if not np.array_equal(a1["SH_r4"], a2["SH_r4"], equal_nan=True):
                            rep["pc_self_intervention_caught"] += 1
        rep["checks_ok"] = bool(rep["pass1_diff_targets"] == 0 and rep["pass2_diff_targets"] == 0
                                and rep["pc_self_guard_caught"] == rep["pc_targets"]
                                and rep["pc_copy_guard_caught"] == rep["pc_targets"]
                                and rep["pc_self_intervention_caught"] == rep["pc_targets"])
        # s0-mechanics-5: a pass needs full coverage (every dev seed, at least one target and one positive control)
        rep["coverage_full"] = bool(rep["seeds"] == list(paths.DEV_SEEDS) and rep["targets"] > 0 and rep["pc_targets"] > 0)
        rep["passed"] = bool(rep["checks_ok"] and rep["coverage_full"])
        log(f"  intervention {design}: {rep['targets']} targets; pass1 diffs {rep['pass1_diff_targets']}, pass2 diffs "
            f"{rep['pass2_diff_targets']}; positive controls caught {rep['pc_self_guard_caught']}/{rep['pc_copy_guard_caught']}/"
            f"{rep['pc_self_intervention_caught']} of {rep['pc_targets']} -> passed {rep['passed']}")
        return rep

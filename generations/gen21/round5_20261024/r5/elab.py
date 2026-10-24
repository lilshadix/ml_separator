"""E-LAB, campaign calibration of a new extractant (lab-1): CLAUDE_CODE_PROMPT.md section 6.3 (conventions 6.0, 6.10,
8, 9; I2 lab-1; ideas2/lab.md).  Stage-1 item 1.3, the E-CT annex; outputs under results/ELAB/.

Copied and adapted (never run in place) from round5_design_20261023/ideas2/scripts/
    lab1_recompute_skA.py  LF SHA-256 8a82cd2a5d38a7b60f6daa0c4ffc4b20f3c6f47bbdd384ba6d04b7700faf253e  (qdev,
                           interp_levels, lm_cw -> lm_shape, E_RM, E_PBC, E_PBA, NC05 / NC03, leave-two-out)
    lab1skepB_placebo.py   LF SHA-256 df14531850d0521e679260823990c19404993fe7c4b7ebf1cb056d9ce8f60f3f  (E_PCM,
                           E_PCMC: per-metal count-matched level-matched placebos)
    lab1skepB_nearcopy.py  LF SHA-256 1f5434f6c23e7933416c9a91f0a28fd9aef39ac4c7941f4bdb1bedb2c84cb902  (the label-free
                           near-copy rule; here through r5.dupes.near_copy_mask, copied from the same file)
    lab1_controls_skA.py   LF SHA-256 b0d7f2a8cc09ea114d2d49fa85074a3c7cbde54bf610e989063bbca3b815dd7b  (global-K grid)
    lab_calib.py           LF SHA-256 9b80f5ca68d29c38d0ec5f577d8595bb09d03c5c7842ae5facedc5477ee71de8  (POPX0, MX)
and the theme note ideas2/lab.md (LF SHA-256 aa66feee53e133000985272f901df4a740fb4a73723febb60390588fb7156bf8).

**Model** (every arm; exact at the 4 reads).  levels = PWL_R[y_S - v_S] + v along the -Shannon CN8 radius axis
(r3.interp 'R'), S = the cell's DOPT_Q read set of 4 reads (union of the cached fold sequence ``seqs['DOPT_Q'][:3]``,
code budget k = 3), u = beta 1[Gd] with beta the fold's fitted round-4 amplitude (B1 is v = u).  MX_LM10 replaces u by
    v_m = (1 - w_m) u_m + w_m c_m,   w_m = n_m / (n_m + 6.1),
c_m = the mean over donor series of each donor's deviation from its own least-squares quadratic in the radius axis
(donors with >= 6 finite metals), using only donor metals whose V1 log D lies within 1.0 log of the target's B1
level L_t,m (label-free: L_t is the true value at the 4 reads and B1's interpolation elsewhere); n_m = the donors that
pass at m.  K = 6.1 and the 1.0 tolerance were pinned before any lab-1 score but on these data (disclosed as seen).

**Pools** (per C3 target t; donors are bench cells, all 521, outside the round-3 V2-1/2/3 lists, >= 6 finite metals,
never t):  SRC = same publication, other extractant (lab-1 as published; the reproduction pool).  CLEAN = SRC minus
the links t-j flagged by the frozen section-5.1 bit-identity rule or by r5.dupes' near-identity scan (max |d| <= 0.05
over >= 6 shared metals: the V2-missed duplicates of I2 items 4-7, quarantined), with metal-concentration duplicate
groups (r5.dupes.metal_conc_duplicates) reduced to their first member present.  PRIM (the primary pool, per design and
seed) = CLEAN minus the label-free lab-1 near-copies: same medium (acid type, diluent, |dlog10 acid| <= 0.15) and log D
within 0.15 of t's at every one of t's 4 read metals.  XPUB (placebo pool) = every other publication's donor cell minus
the quarantined links with t; XPUB_FAM = XPUB with t's chem_family.  C5 variants drop pub_6a0ef8528c (C5_out) or cell
c950dc2982e96b83 (C5_cell_out) from every pool and from the rows.

**Populations / strata** (rows = the round-4 REF rows of each target, 4 reads, BPG primary, BP sensitivity):
SRC (POPX as published: CLEAN target with >= 1 SRC donor); PRIMARY = POPX with near-copies excluded (>= 1 CLEAN
donor; PRIM may still be empty at a read set, then every MX arm equals B1 there); POPX0 = PRIMARY with no
same-publication same-extractant donor cell (bit links removed), POPX1 the rest; SCOPE_IN = amide O-donor solvating
extractants (frozen mechanism map r5.supply: NEUTRAL_AMIDE) in nitrate media (acid one-hot HNO3), NULL_PRIOR the other
classes (declared null-prior strata); LOWD / HIGHD = rows with min(L_A, L_B) < -2 / >= -2 (label-free B1 levels; the
low-D band split); SS_OS = targets with both same-series (sharing an experiment_series_id) and other-series CLEAN
donors; SM_DM = the same split by medium (Skeptic B's reading, DIAGNOSTIC); NO_LEAD2 = PRIMARY without pub_126bbe7976
and pub_1bd76c48d9; OUT_GIANT = PRIMARY outside source cluster g:pub_74428aeab2.

**Arms** (point arms are pair-SF predictions; E_* are expected squared errors over NDRAW = 30 seeded draws):
  B1; MX_LM10 (PRIM); MX_LM05 / MX_LM20 (tolerance 0.5 / 2.0, SENSITIVITY); MX_K{6p1,12,20,35,60,100} (unmatched shape
  over PRIM, w = n / (n + K): the global-K shrinkage control); MX_GK_O (the best K on the scored rows: ORACLE [O]);
  MX_GK_LOPO (K chosen on the other publications' PRIMARY rows, ties averaged); MX_LM10_SS / _OS (same-series /
  other-series donors), MX_LM10_SM / _DM (same / other medium); MX_LM10_LO / _HI (the blend applied only at metals with
  L < -2 / >= -2, DIAGNOSTIC); sensitivities MX_LM10_NB (SRC pool, lab-1 as published), MX_LM10_NC (SRC minus the
  near-copy rule only, Skeptic B's +7.6 %), MX_LM10_OFS (PRIM minus constant-offset links), MX_LM10_NC05 / _NC03
  (Skeptic A's label-based offset-sd rules on SRC: they read the target's unread labels, [O]).
  E_RM (random own-campaign donors: per metal the number MX_LM10 keeps, drawn from PRIM donors measured there; same
  w_m); E_PBC (each PRIM donor replaced by an other-campaign cell of the donor's chemotype, then level-matched); E_PBA
  (as E_PBC from any other-campaign cell); E_PCM (per metal n_m level-matched donors from XPUB, w = k / (k + 6.1));
  E_PCMC (E_PCM from XPUB_FAM).
**Units**: publication; laboratory (frozen PROVISIONAL key results/stage0/lab_key.csv, conservative and liberal);
source cluster cluster_cpg.  **Spread**: LOPO jackknife t(P - 1), leave-two-publications-out, leave-top-3-out, top-3
share.  Every number carries "EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression".

**Information contract (section 8).**  Allowed: the target's 4 legal reads; the campaign's other series (full); the
fold's beta.  Forbidden: the target's unread metals, the target (or a bit-identical copy) in a primary pool.  The
duplicate links are a frozen Stage-0 data rule applied once to V1 (label-based) and held fixed.  ``guard`` refuses a
pool that holds the target, or (except SRC, the reproduction pool) a bit-identical copy; ``intervention`` scrambles
the target's unread labels and requires bit-identical predictions and draws for every arm except the two declared
label-based sensitivities (NC05 / NC03).
"""
from __future__ import annotations

import itertools
import json
import re
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats as SS

from . import paths
from . import common as CM
from . import dupes as DU
from . import population as POP
from r3 import interp  # noqa: E402
from r4 import refs as RF  # noqa: E402
from g21 import cache as GC  # noqa: E402

# Copy rule (brief 1.2): every copied or re-implemented source with its LF SHA-256 at the copy; checked by
# scripts/r5_seal.py (copied_sources reads this declaration).
COPIED_FROM = {
    "generations/gen21/round5_design_20261023/ideas2/scripts/lab1_recompute_skA.py":
        "8a82cd2a5d38a7b60f6daa0c4ffc4b20f3c6f47bbdd384ba6d04b7700faf253e",
    "generations/gen21/round5_design_20261023/ideas2/scripts/lab1skepB_placebo.py":
        "df14531850d0521e679260823990c19404993fe7c4b7ebf1cb056d9ce8f60f3f",
    "generations/gen21/round5_design_20261023/ideas2/scripts/lab1skepB_nearcopy.py":
        "1f5434f6c23e7933416c9a91f0a28fd9aef39ac4c7941f4bdb1bedb2c84cb902",
    "generations/gen21/round5_design_20261023/ideas2/scripts/lab1_controls_skA.py":
        "b0d7f2a8cc09ea114d2d49fa85074a3c7cbde54bf610e989063bbca3b815dd7b",
    "generations/gen21/round5_design_20261023/ideas2/scripts/lab_calib.py":
        "9b80f5ca68d29c38d0ec5f577d8595bb09d03c5c7842ae5facedc5477ee71de8",
    "generations/gen21/round5_design_20261023/ideas2/lab.md":
        "aa66feee53e133000985272f901df4a740fb4a73723febb60390588fb7156bf8",
}

LN = POP.LN
N_LN = len(LN)
GD = LN.index("Gd")
X_R = np.asarray(interp._coords("R"), float)
K_LM = 6.1                       # w = n / (n + 6.1): r = 0.14 roughness correlation (E6 section 5) [I2 lab-1]
TOL = 1.0                        # level-match tolerance (log units), fixed before any lab-1 score
TOLS = {"MX_LM10": 1.0, "MX_LM05": 0.5, "MX_LM20": 2.0}
NDRAW = 30
MIN_METALS = 6
DEG = 2
K_GRID = (6.1, 12.0, 20.0, 35.0, 60.0, 100.0)
LOWD = -2.0
NC_SD = {"MX_LM10_NC05": 0.05, "MX_LM10_NC03": 0.03}
LEAD2 = ("pub_126bbe7976", "pub_1bd76c48d9")
GIANT = "g:pub_74428aeab2"
SCOPE_CLASS, SCOPE_ACID = "NEUTRAL_AMIDE", "hno3"
B1_TOL = 8.9e-16
REPRO_TOL = 1e-12
LAB_KEY = paths.RES / "stage0" / "lab_key.csv"
LAB_ROWS2 = paths.DESIGN / "ideas2" / "lab_out" / "lab_rows2.parquet"     # read-only reproduction reference
LAB_ROWS = paths.DESIGN / "ideas2" / "lab_out" / "lab_rows.parquet"
C5_VARIANTS = ("C5_in", "C5_out", "C5_cell_out")
PRIMARY_C5, PRIMARY_DESIGN = "C5_in", "BPG"
UNITS = ("publication", "lab_cons", "lab_lib", "cluster_cpg")
PRIMARY_UNIT = "lab_cons"           # brief 6.3 "Units. Laboratory (frozen key)": conservative, PROVISIONAL (U6 open)
UNIT_TEXT = {"lab_cons": "laboratory, conservative key (PROVISIONAL; U6 open); primary unit (brief 6.3)",
             "lab_lib": "laboratory, liberal key (PROVISIONAL)",
             "publication": "publication = bridge to the seen +7.6 % [-1.1, +16.4], p 0.093"}
UNIT_WORD = {"publication": "publications", "lab_cons": "laboratories (conservative key)",       # the W/L/T unit (A2)
             "lab_lib": "laboratories (liberal key)", "cluster_cpg": "source clusters"}


def kname(K: float) -> str:
    return "MX_K" + f"{K:g}".replace(".", "p")


K_ARMS = tuple(kname(k) for k in K_GRID)
POINT_ARMS = (("B1", "MX_LM10", "MX_LM05", "MX_LM20") + K_ARMS
              + ("MX_LM10_SS", "MX_LM10_OS", "MX_LM10_SM", "MX_LM10_DM", "MX_LM10_LO", "MX_LM10_HI",
                 "MX_LM10_NB", "MX_LM10_NC", "MX_LM10_OFS", "MX_LM10_NC05", "MX_LM10_NC03"))
DRAW_ARMS = ("E_RM", "E_PBC", "E_PBA", "E_PCM", "E_PCMC")
DERIVED = ("MX_GK_LOPO",)
ORACLE_ARMS = frozenset({"MX_GK_O", "MX_LM10_NC05", "MX_LM10_NC03"})
DIAG_ARMS = frozenset({"MX_LM10_LO", "MX_LM10_HI", "MX_LM10_SM", "MX_LM10_DM"})
SENS_ARMS = frozenset({"MX_LM05", "MX_LM20", "MX_LM10_NB", "MX_LM10_NC", "MX_LM10_OFS", "MX_LM10_NC05", "MX_LM10_NC03"})
LABEL_BASED = frozenset({"MX_LM10_NC05", "MX_LM10_NC03"})
REPRO_ARMS = ("B1", "MX_LM10_NB", "MX_LM05_NB", "MX_LM20_NB", "MX_K6p1_NB")
GUARD = (("src", True), ("clean", False), ("ofs", False), ("xpub", False), ("xpub_fam", False))
STRATA = ("PRIMARY", "POPX0", "POPX1", "SCOPE_IN", "NULL_PRIOR", "LOWD", "HIGHD", "SS_OS", "SM_DM", "NO_LEAD2",
          "OUT_GIANT", "SRC")
FULL_STRATA = ("PRIMARY", "POPX0", "SCOPE_IN")
C5_STRATA = ("PRIMARY", "POPX0")
SENS_STRATA = frozenset({"SRC", "NO_LEAD2", "OUT_GIANT"})
DIAG_STRATA = frozenset({"SM_DM"})
MAIN = (("MX_LM10", "B1"), ("MX_LM10", "E_PBC"), ("MX_LM10", "E_PCM"), ("MX_LM10", "E_RM"), ("E_PBC", "B1"),
        ("E_PCM", "B1"))
FULL = MAIN + (("MX_LM10", "E_PCMC"), ("MX_LM10", "E_PBA"), ("E_PCMC", "B1"), ("E_PBA", "B1"), ("E_RM", "B1"),
               ("MX_LM10", "MX_GK_O"), ("MX_LM10", "MX_GK_LOPO"), ("MX_GK_LOPO", "B1"), ("MX_LM10", "MX_K6p1"),
               ("MX_K6p1", "B1"), ("MX_LM10_LO", "B1"), ("MX_LM10_HI", "B1"), ("MX_LM10_NB", "B1"),
               ("MX_LM10_NC", "B1"), ("MX_LM10_OFS", "B1"), ("MX_LM10_NC05", "B1"), ("MX_LM10_NC03", "B1"),
               ("MX_LM05", "B1"), ("MX_LM20", "B1"))
SS_CON = (("MX_LM10_SS", "B1"), ("MX_LM10_OS", "B1"), ("MX_LM10_SS", "MX_LM10_OS"))
SM_CON = (("MX_LM10_SM", "B1"), ("MX_LM10_DM", "B1"), ("MX_LM10_SM", "MX_LM10_DM"))
SEEN = "EXPLORATORY (seen)"
BEST25 = "best of about 25 arms on seen rows; expect regression"
SENS, ORACLE, DIAG, CC, SMOKE = "SENSITIVITY", "ORACLE [O]", "DIAGNOSTIC", "CONSISTENCY CHECK", "TECHNICAL SMOKE"
NOT_NEXT_TO = ("E-LAB (campaign calibration of a new extractant; donors = the campaign's other extractants' full "
               "series): never placed next to E-A's 0.452; B1 on the same rows is the comparator")
REPRO = {  # brief 6.0 (I2 lab-1; Skeptic A); 4-decimal values from ideas2/lab_out/lab_calib2.log and lab_popx0.log
    "POPX": {"cells": 92, "publications": 16, "rows_per_seed": 787},
    "POPX0": {"cells": 60, "publications": 13, "rows_per_seed": 534},
    "rows_both_designs": 7870, "mx_lm10_maxabs": 0.0,
    "R2_3dp": {"BPG": {"B1": 0.444, "MX_LM10_NB": 0.494}},
    "R2_4dp_log": {"BPG": {"B1": 0.4437, "MX_LM10_NB": 0.4936}, "BP": {"B1": 0.4423, "MX_LM10_NB": 0.4936}},
    "POPX0_R2_4dp_log": {"BPG": {"B1": 0.4008, "MX_LM10_NB": 0.4670}, "BP": {"B1": 0.3987, "MX_LM10_NB": 0.4669}}}
SEEN_VALUES = {  # brief 6.3 / I2 lab-1, quoted as written; none is a reproduction target except REPRO
    "POPX_B1_to_MX_LM10": "0.444 -> 0.494 (+9.0 % [+0.3, +17.6]; 10 up / 4 down, p 0.037)", "POPX0_rel": "+11.1 %",
    "cleaned_rel": "+7.6 % [-1.1, +16.4], p 0.093 (label-free 0.15 near-copy rule; Skeptic B)",
    "vs_E_PBC": "+10.2 % [+1.6, +18.9]", "vs_E_PCM": "+11.4 % [+2.3, +20.4], 11 / 3, p 0.008",
    "leave_two_publications_out": "+2.6 % (without pub_126bbe7976, pub_1bd76c48d9)", "top3_share": 0.93,
    "vs_E_RM": "+1.8 % [-0.5, +4.2] (8 up, 1 down)", "vs_best_global_K": "+4.7 % [+2.2, +7.2] (K = 12: +4.5 % vs B1)",
    "same_series_vs_other_series": "+3.8 % vs -3.6 % (36 cells, 4 publications)",
    "by_class": "DGA +11.2 %, malonamide +16.5 %, pyridine/phenanthroline amides 0.0 %, other -2.2 %",
    "source_clusters": "3 up / 1 down, p 0.125"}
SEEN_POOL = {  # A2: the donor pool of every seen value (lab-1 scored the SRC pool; Skeptic B its near-copy-cleaned form)
    "src": "lab-1 SRC pool, before near-copy cleaning",
    "nc": "Skeptic B: lab-1 SRC pool minus the label-free near-copies (the MX_LM10_NC pool)"}
SEEN_BP = {  # A2: BP seen values where the source gives them (I2 lab-1 reproduction, Skeptic A: BP +9.20 % / +11.34 %; R^2
             # from lab_calib2.log as in REPRO); every other seen value is BPG's and says so on BP
    "POPX_B1_to_MX_LM10": "0.4423 -> 0.4936 (+9.20 %)", "POPX0_rel": "+11.34 %"}
NOT_ON_SRC = "not computed on the SRC pool (registered on the cleaned pools only)"
EXTERNAL_RULE = ("external leg descriptive unless >= 8 independent laboratories with >= 2 extractants x >= 6 Ln in one "
                 "campaign exist; kill if the laboratory-median relative SSE (MX_LM10 - B1) <= 0 or its one-sided 95 % "
                 "lower bound <= 0; narrow to 'level-conditional weighting, not campaign' if MX_LM10 - E_PCM <= 0; harm "
                 "flag if any class with >= 3 units loses > 5 %.  Operating characteristics: false kill of a true effect "
                 "0.41-0.52 at 8 units, 0.25 at 11, 0.06-0.10 at 16 [I2 lab-1]")


class LeakageError(AssertionError):
    """A donor pool that breaks the E-LAB information contract."""


# ------------------------------------------------------------------------------------------------------------
# pure core (lab1_recompute_skA.py)
# ------------------------------------------------------------------------------------------------------------
def qdev(y) -> np.ndarray:
    """Deviation of a series from its own least-squares quadratic in the radius axis (NaN below 6 finite metals)."""
    y = np.asarray(y, float)
    out = np.full(N_LN, np.nan)
    m = np.isfinite(y)
    if m.sum() >= MIN_METALS:
        co = np.polyfit(X_R[m], y[m], DEG)
        out[m] = y[m] - np.polyval(co, X_R[m])
    return out


def detrend(Y) -> np.ndarray:
    Y = np.asarray(Y, float)
    return np.array([qdev(Y[i]) for i in range(len(Y))])


def interp_levels(yi, reads, metals, v) -> np.ndarray:
    """Shape-vector interpolation PWL_R[y_S - v_S] + v (exact at the reads)."""
    xr = X_R[reads]
    o = np.argsort(xr)
    out = np.full(N_LN, np.nan)
    mm = np.asarray(metals, int)
    out[mm] = np.interp(X_R[mm], xr[o], (yi[reads] - v[reads])[o]) + v[mm]
    out[reads] = yi[reads]
    return out


def lm_shape(Dv, Y, donors, lv, tol: float = TOL, K: float = K_LM, mask=None):
    """(c, w): per-metal mean donor deviation over the donors whose log D lies within ``tol`` of lv (or over ``mask``)
    and w = n / (n + K).  No donor: (0, 0), i.e. B1."""
    donors = np.asarray(donors, int)
    if len(donors) == 0:
        return np.zeros(N_LN), np.zeros(N_LN)
    A = Dv[donors]
    with np.errstate(invalid="ignore"):
        msk = (np.isfinite(A) & (np.abs(Y[donors] - lv[None, :]) <= tol)) if mask is None else mask
    n = msk.sum(0).astype(float)
    c = np.where(n > 0, np.where(msk, A, 0.0).sum(0) / np.maximum(n, 1), 0.0)
    return c, n / (n + K)


def shape_all(Dv, donors, K: float):
    """Unmatched shape over every finite donor metal (lab_calib.shape; the global-K control)."""
    donors = np.asarray(donors, int)
    if len(donors) == 0:
        return np.zeros(N_LN), np.zeros(N_LN)
    return lm_shape(Dv, Dv, donors, np.zeros(N_LN), K=K, mask=np.isfinite(Dv[donors]))


def matched_counts(Dv, Y, donors, lv, tol: float = TOL) -> np.ndarray:
    donors = np.asarray(donors, int)
    if len(donors) == 0:
        return np.zeros(N_LN, int)
    with np.errstate(invalid="ignore"):
        return (np.isfinite(Dv[donors]) & (np.abs(Y[donors] - lv[None, :]) <= tol)).sum(0)


def blend(u, c, w, sub=None) -> np.ndarray:
    v = np.asarray(u, float).copy()
    idx = np.arange(N_LN) if sub is None else np.asarray(sub, int)
    if len(idx):
        v[idx] = (1.0 - w[idx]) * u[idx] + w[idx] * c[idx]
    return v


def offsd(yi, yj) -> float:
    """Skeptic A's label-based near-copy statistic: sd of the offset-removed difference over >= 6 shared metals."""
    sh = np.isfinite(yi) & np.isfinite(yj)
    if sh.sum() < 6:
        return np.inf
    dy = yi[sh] - yj[sh]
    return float(np.std(dy - dy.mean()))


def draws_rm(rng, Dv, Y, donors, lv, ndraw: int = NDRAW) -> list:
    """E_RM: per metal the number of donors level matching keeps, drawn at random from the donors measured there."""
    donors = np.asarray(donors, int)
    if len(donors) == 0:
        return [(np.zeros(N_LN), np.zeros(N_LN))] * ndraw
    fin = np.isfinite(Dv[donors])
    nsel = matched_counts(Dv, Y, donors, lv)
    out = []
    for _ in range(ndraw):
        msk = np.zeros_like(fin)
        for m in range(N_LN):
            if nsel[m] > 0:
                msk[rng.choice(np.where(fin[:, m])[0], int(nsel[m]), replace=False), m] = True
        out.append(lm_shape(Dv, Y, donors, lv, mask=msk))
    return out


def draws_pbc(rng, Dv, Y, donors, lv, chem, pool, ndraw: int = NDRAW) -> list:
    """E_PBC: every own donor replaced by a random other-campaign cell of the donor's chemotype (any pool cell if none
    is left), then level-matched; E_PBA when ``chem`` is None (replacement from any pool cell)."""
    donors, pool = np.asarray(donors, int), np.asarray(pool, int)
    out = []
    for _ in range(ndraw):
        if chem is None:
            rep = rng.choice(pool, min(len(donors), len(pool)), replace=False) if len(pool) else np.zeros(0, int)
        else:
            rep = []
            for j in donors:
                pa = np.setdiff1d(pool, rep)
                pc = pa[chem[pa] == chem[j]]
                src = pc if len(pc) else pa
                if not len(src):
                    break
                rep.append(int(rng.choice(src)))
        out.append(lm_shape(Dv, Y, np.asarray(rep, int), lv))
    return out


def draws_pcm(rng, Dv, Y, nm, lv, pool, ndraw: int = NDRAW, K: float = K_LM) -> list:
    """E_PCM: per metal, min(n_m, available) level-matched donors drawn from the pool; w = k / (k + K)."""
    pool = np.asarray(pool, int)
    with np.errstate(invalid="ignore"):
        cand = [pool[np.isfinite(Dv[pool, m]) & (np.abs(Y[pool, m] - lv[m]) <= TOL)] if len(pool) else pool
                for m in range(N_LN)]
    out = []
    for _ in range(ndraw):
        c, w = np.zeros(N_LN), np.zeros(N_LN)
        for m in range(N_LN):
            if nm[m] == 0 or not len(cand[m]):
                continue
            k = min(int(nm[m]), len(cand[m]))
            dd = rng.choice(cand[m], k, replace=False)
            c[m] = Dv[dd, m].mean()
            w[m] = k / (k + K)
        out.append((c, w))
    return out


def lab_map(path=LAB_KEY) -> dict:
    """{publication_id: (lab_conservative, lab_liberal)} from the frozen PROVISIONAL laboratory key (corpus rows)."""
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


def fold_betas(design: str) -> dict:
    fitted = json.loads((paths.R4RES / "REF" / f"fitted_{design}.json").read_text(encoding="utf-8"))
    out = {tuple(int(x) for x in k.split("_")): float(v["ref:B1"]["coef"][0]) for k, v in fitted.items()
           if re.fullmatch(r"\d+_\d+", k)}
    if len(out) != 25:
        raise AssertionError(f"{design}: expected 25 fold betas, found {len(out)}")
    return out


def rng_for(arm: str, c5: str, design: str, seed: int, t: str):
    """Registry scheme (REGISTRY.csv): default_rng(20261023 + crc32('ELAB|arm|c5|design|seed|target'))."""
    return np.random.default_rng(paths.seed_for(f"ELAB|{arm}|{c5}|{design}|{seed}|{t}"))


# ------------------------------------------------------------------------------------------------------------
# the corpus: pools and the row builder
# ------------------------------------------------------------------------------------------------------------
class Campaign:
    """Bench, frozen duplicate links, static donor pools per C3 target, laboratory key and label-free metadata."""

    def __init__(self, bench=None, *, lab_key=LAB_KEY, log=print):
        from gen20k import data as D
        from . import supply as SU
        t0 = time.time()
        self.bench = bench if bench is not None else D.load_bench()
        f = self.bench.frame.reset_index(drop=True)
        self.Y = np.asarray(self.bench.Y, float)
        self.ids = f["cell_id"].astype(str).tolist()
        self.pos = {c: i for i, c in enumerate(self.ids)}
        self.N = len(self.ids)
        self.meta = POP.metadata_frame(f)
        self.finite = POP.finite_from_metals(self.meta)
        if not np.array_equal(self.finite, np.isfinite(self.Y)):
            raise AssertionError("metals metadata disagrees with the finite pattern of V1")
        self.pub = self.meta["publication_id"].astype(str).to_numpy()
        self.ext = self.meta["extractant"].astype(str).to_numpy()
        self.chem = self.meta["chemotype"].astype(str).to_numpy()
        self.family = f["chem_family"].astype(str).to_numpy()
        self.series = [frozenset(s for s in str(x).split(";") if s and s != "nan") for x in f["experiment_series_ids"]]
        self.mech = np.array([SU.classify_extractant(fam.replace("_", " "), str(nm))["mechanism_class"]
                              for fam, nm in zip(self.family, f["extractant_name"].astype(str))])
        acid_cols = [c for c in self.meta.columns if c.startswith("cond__acid__")]
        A = self.meta[acid_cols].fillna(0).to_numpy(float)
        self.acid_name = np.where(A.max(1) > 0, np.array([c[len("cond__acid__"):] for c in acid_cols])[A.argmax(1)], "none")
        self.scope = (self.mech == SCOPE_CLASS) & (self.acid_name == SCOPE_ACID)
        self.acid_id, self.dil_id, self.lacid = DU.medium_keys(self.meta)
        self.v2drop = POP.v2drop_set()
        self.has = (self.finite.sum(1) >= MIN_METALS) & np.array([c not in self.v2drop for c in self.ids])
        self.Dv = detrend(self.Y)
        # frozen duplicate links (Stage-0 data rules on V1; label-based, applied once)
        self.bit = {frozenset(x) for x in DU.flagged_links(DU.bit_identity(self.Y, self.ids))}
        ofs = DU.offset_scan(self.Y, self.ids, [tuple(x) for x in self.bit])
        self.nearid = {frozenset((a, b)) for a, b, fl in zip(ofs.cell_a, ofs.cell_b, ofs.flag_nearid) if fl}
        self.offset = {frozenset((a, b)) for a, b, fl in zip(ofs.cell_a, ofs.cell_b, ofs.flag_offset) if fl}
        self.quar = self.bit | self.nearid
        self.mgroups = DU.metal_conc_duplicates(self.meta, self.finite)
        self._mrank = {self.pos[c]: (g, r) for g, mem in enumerate(self.mgroups) for r, c in enumerate(mem)}
        self.lab = lab_map(lab_key)
        self.c3 = POP.c3_cells_from_ref()
        self.keep = {"C5_in": np.ones(self.N, bool), "C5_out": self.pub != paths.C5,
                     "C5_cell_out": np.array([c != paths.C5_CELL for c in self.ids])}
        self._static()
        self.seconds = round(time.time() - t0, 1)
        log(f"  campaign: {self.N} bench cells, {len(self.c3)} C3 targets, SRC {len(self.popx_src)}, PRIMARY "
            f"{len(self.primary)}; links: bit {len(self.bit)}, near-id {len(self.nearid)}, offset {len(self.offset)}, "
            f"metal-conc groups {len(self.mgroups)}; lab key {'found' if self.lab else 'MISSING'} ({self.seconds}s)")

    # ---- static pools --------------------------------------------------------------------------------------
    def _free(self, t: str, arr, links) -> np.ndarray:
        arr = np.asarray(arr, int)
        return arr[np.array([frozenset((t, self.ids[j])) not in links for j in arr], bool)] if len(arr) else arr

    def dedup(self, arr) -> np.ndarray:
        """Metal-concentration duplicate groups reduced to their first member present in the pool."""
        arr = np.asarray(arr, int)
        present = set(arr.tolist())
        out = []
        for j in arr:
            g = self._mrank.get(int(j))
            if g is not None and any(self.pos[c] in present for c in self.mgroups[g[0]][:g[1]]):
                continue
            out.append(int(j))
        return np.asarray(out, int)

    def same_med(self, i: int) -> np.ndarray:
        return (self.acid_id == self.acid_id[i]) & (self.dil_id == self.dil_id[i]) & \
            (np.abs(self.lacid - self.lacid[i]) <= DU.NC_DACID)

    def _static(self) -> None:
        ar = np.arange(self.N)
        self.P, rows = {}, []
        for t in self.c3:
            i = self.pos[t]
            base = self.has & (ar != i)
            src = np.where(base & (self.pub == self.pub[i]) & (self.ext != self.ext[i]))[0]
            clean = self.dedup(self._free(t, src, self.quar))
            ms_src = np.where(base & (self.pub == self.pub[i]) & (self.ext == self.ext[i]))[0]
            xpub = self._free(t, np.where(base & (self.pub != self.pub[i]))[0], self.quar)
            P = {"src": src, "clean": clean, "ofs": self._free(t, clean, self.offset), "xpub": xpub,
                 "xpub_fam": xpub[self.family[xpub] == self.family[i]]}
            ss = np.array([bool(self.series[i] & self.series[j]) for j in clean], bool)
            sm = self.same_med(i)[clean] if len(clean) else np.zeros(0, bool)
            self.P[t] = P
            rows.append({"cell_id": t, "publication": self.pub[i], "t_v2drop": t in self.v2drop, "n_src": len(src),
                         "n_clean": len(clean), "n_ofs": len(P["ofs"]), "n_xpub": len(xpub), "n_xpub_fam": len(P["xpub_fam"]),
                         "n_ms_src": len(ms_src), "n_ms": len(self._free(t, ms_src, self.bit)),
                         "n_ss": int(ss.sum()), "n_os": int((~ss).sum()), "n_sm": int(sm.sum()), "n_dm": int((~sm).sum()),
                         "chem_family": self.family[i], "mech_class": self.mech[i], "acid": self.acid_name[i],
                         "scope_in": bool(self.scope[i]), "lab_cons": self.lab.get(self.pub[i], ("pub:" + self.pub[i],) * 2)[0],
                         "lab_lib": self.lab.get(self.pub[i], ("pub:" + self.pub[i],) * 2)[1],
                         "lab_key_missing": self.pub[i] not in self.lab})
        self.static = pd.DataFrame(rows)
        st = self.static
        self.popx_src = sorted(st.cell_id[(~st.t_v2drop) & (st.n_src > 0)])
        self.primary = sorted(st.cell_id[(~st.t_v2drop) & (st.n_clean > 0)])

    def pools(self, t: str, c5: str = PRIMARY_C5) -> dict:
        keep = self.keep[c5]
        return {k: v[keep[v]] for k, v in self.P[t].items()}

    def guard(self, i: int, donors, *, allow_bit: bool = False, name: str = "") -> None:
        t = self.ids[i]
        donors = np.asarray(donors, int)
        if (donors == i).any():
            raise LeakageError(f"{t}: pool {name} holds the target itself")
        if not allow_bit:
            bad = [self.ids[j] for j in donors if frozenset((t, self.ids[j])) in self.bit]
            if bad:
                raise LeakageError(f"{t}: pool {name} holds a bit-identical copy {bad[:2]}")

    # ---- one target ------------------------------------------------------------------------------------------
    def target_arms(self, t: str, reads, metals, beta: float, ia, ib, *, design: str, seed: int, c5: str = PRIMARY_C5,
                    Yv=None, Dv=None, inject=None, guard: bool = True, mode: str = "all"):
        """Every arm for one (design, seed, target): ({point arm: (P,) SF}, {draw arm: (NDRAW, P) SF}, info).
        ``Yv`` / ``Dv`` default to V1 and its detrended deviations (the intervention passes poisoned copies);
        ``inject`` ({pool: [cell index]}, 'prim' applied after the near-copy rule) exists for the positive controls;
        ``mode='repro'`` returns B1 and the SRC-pool reproduction arms only."""
        i = self.pos[t]
        Yv = self.Y if Yv is None else Yv
        Dv = self.Dv if Dv is None else Dv
        yi = Yv[i]
        reads = sorted(int(m) for m in reads)
        metals = [int(m) for m in metals]
        ia, ib = np.asarray(ia, int), np.asarray(ib, int)
        u = np.zeros(N_LN)
        u[GD] = float(beta)
        lv = interp_levels(yi, reads, metals, u)
        P = self.pools(t, c5)
        for k, extra in (inject or {}).items():
            if k != "prim":
                P[k] = np.concatenate([P[k], np.asarray(extra, int)])
        if guard:
            for k, allow in GUARD:
                self.guard(i, P[k], allow_bit=allow, name=k)

        def lev(cw, sub=None):
            c, w = cw
            L = interp_levels(yi, reads, metals, blend(u, c, w, sub))
            return L[ia] - L[ib]

        def nearcopy(d):
            if not len(d):
                return np.zeros(0, bool)
            return DU.near_copy_mask(yi[reads], Yv[d][:, reads], self.same_med(i)[d])
        pt = {"B1": lv[ia] - lv[ib]}
        src = P["src"]
        if mode == "repro":
            pt["MX_LM10_NB"] = lev(lm_shape(Dv, Yv, src, lv, 1.0))
            pt["MX_LM05_NB"] = lev(lm_shape(Dv, Yv, src, lv, 0.5))
            pt["MX_LM20_NB"] = lev(lm_shape(Dv, Yv, src, lv, 2.0))
            pt["MX_K6p1_NB"] = lev(shape_all(Dv, src, 6.1))
            return pt, {}, {"lvA": lv[ia], "lvB": lv[ib]}
        clean = P["clean"]
        nc = nearcopy(clean)
        prim = clean[~nc]
        if "prim" in (inject or {}):
            prim = np.concatenate([prim, np.asarray(inject["prim"], int)])
        cw = {a: lm_shape(Dv, Yv, prim, lv, tol) for a, tol in TOLS.items()}
        for a in TOLS:
            pt[a] = lev(cw[a])
        for K in K_GRID:
            pt[kname(K)] = lev(shape_all(Dv, prim, K))
        ss = np.array([bool(self.series[i] & self.series[j]) for j in prim], bool)
        sm = self.same_med(i)[prim] if len(prim) else np.zeros(0, bool)
        pt["MX_LM10_SS"], pt["MX_LM10_OS"] = lev(lm_shape(Dv, Yv, prim[ss], lv)), lev(lm_shape(Dv, Yv, prim[~ss], lv))
        pt["MX_LM10_SM"], pt["MX_LM10_DM"] = lev(lm_shape(Dv, Yv, prim[sm], lv)), lev(lm_shape(Dv, Yv, prim[~sm], lv))
        fl = np.isfinite(lv)
        pt["MX_LM10_LO"] = lev(cw["MX_LM10"], np.where(fl & (lv < LOWD))[0])
        pt["MX_LM10_HI"] = lev(cw["MX_LM10"], np.where(fl & (lv >= LOWD))[0])
        pt["MX_LM10_NB"] = lev(lm_shape(Dv, Yv, src, lv))
        nc_src = nearcopy(src)
        pt["MX_LM10_NC"] = lev(lm_shape(Dv, Yv, src[~nc_src], lv))
        ofs = P["ofs"]
        pt["MX_LM10_OFS"] = lev(lm_shape(Dv, Yv, ofs[~nearcopy(ofs)], lv))
        osd = np.array([offsd(Yv[i], Yv[j]) for j in src]) if len(src) else np.zeros(0)
        for a, thr in NC_SD.items():
            pt[a] = lev(lm_shape(Dv, Yv, src[osd >= thr], lv))
        nm = matched_counts(Dv, Yv, prim, lv)
        dr = {"E_RM": draws_rm(rng_for("E_RM", c5, design, seed, t), Dv, Yv, prim, lv),
              "E_PBC": draws_pbc(rng_for("E_PBC", c5, design, seed, t), Dv, Yv, prim, lv, self.chem, P["xpub"]),
              "E_PBA": draws_pbc(rng_for("E_PBA", c5, design, seed, t), Dv, Yv, prim, lv, None, P["xpub"]),
              "E_PCM": draws_pcm(rng_for("E_PCM", c5, design, seed, t), Dv, Yv, nm, lv, P["xpub"]),
              "E_PCMC": draws_pcm(rng_for("E_PCMC", c5, design, seed, t), Dv, Yv, nm, lv, P["xpub_fam"])}
        dr = {k: np.array([lev(x) for x in v]) for k, v in dr.items()}
        un = [m for m in metals if m not in reads]
        info = {"lvA": lv[ia], "lvB": lv[ib], "n_src": len(src), "n_clean": len(clean), "n_nc": int(nc.sum()),
                "n_nc_src": int(nc_src.sum()),
                "n_prim": len(prim), "n_ofs_prim": int((~nearcopy(ofs)).sum()), "n_ss": int(ss.sum()), "n_os": int((~ss).sum()),
                "n_sm": int(sm.sum()), "n_dm": int((~sm).sum()), "n_xpub": len(P["xpub"]), "n_xpub_fam": len(P["xpub_fam"]),
                "unread": len(un), "unread_no_match": int(sum(nm[m] == 0 for m in un)),
                "mean_w_unread": float(np.mean([nm[m] / (nm[m] + K_LM) for m in un])) if un else np.nan}
        return pt, dr, info

    # ---- one design ------------------------------------------------------------------------------------------
    def iter_targets(self, design: str, R: pd.DataFrame, betas: dict, cells=None, seeds=None):
        """(seed, fold, target, read set, metals, beta, row positions) over the cached fold records."""
        groups = R.groupby(["split_seed", "cell_id"], sort=False).indices
        cells = set(cells) if cells is not None else None
        for seed in (seeds or paths.DEV_SEEDS):
            for fold in range(5):
                rec = GC.load(design, seed, fold)
                for cell in rec["outer"]["cells"]:
                    t = cell["cell_id"]
                    gi = groups.get((seed, t))
                    if gi is None or (cells is not None and t not in cells):
                        continue
                    reads = sorted({int(m) for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                    yield seed, fold, t, reads, [int(m) for m in cell["metals"]], betas[(seed, fold)], gi

    def build(self, design: str, *, Yv=None, score: bool = True, mode: str = "all", log=print):
        """Row frame of the POPX (SRC) targets with every arm per C5 variant (suffix '@<variant>' off the primary),
        the strata and unit columns, plus the identity checks on every REF row.  ``Yv`` (surrogate) replaces V1 for
        reads, donors and labels; ``score=False`` leaves the E_* columns empty (TECHNICAL SMOKE on real data)."""
        t0 = time.time()
        R = CM.load_rows(design)
        if set(R.cell_id) != set(self.c3):
            raise AssertionError(f"{design}: REF cells differ from the C3 targets")
        real = Yv is None
        Yw = self.Y if real else np.asarray(Yv, float)
        Dw = self.Dv if real else detrend(Yw)
        ci = R.cell_id.map(self.pos).to_numpy()
        yref = Yw[ci, R.ia.to_numpy(int)] - Yw[ci, R.ib.to_numpy(int)]
        y_maxabs = float(np.nanmax(np.abs(yref - R.y.to_numpy(float)))) if real else None
        if not real:
            R["y"] = yref
        betas = fold_betas(design)
        n = len(R)
        variants = (PRIMARY_C5,) if mode == "repro" else C5_VARIANTS
        cols, info_rows = {}, []
        b1_par = R.B1.to_numpy(float)
        b1_new = np.full(n, np.nan)
        pl_maxabs, rs_bad, fold_bad, met_bad = 0.0, 0, 0, 0
        popx = set(self.popx_src)
        for seed, fold, t, reads, metals, beta, gi in self.iter_targets(design, R, betas):
            g = R.iloc[gi]
            i = self.pos[t]
            fold_bad += int((g.fold != fold).any())
            rs_bad += int({LN.index(m) for m in str(g.read_set.iloc[0]).split(",")} != set(reads))
            met_bad += int(set(metals) != set(np.where(self.finite[i])[0].tolist()))
            ia, ib = g.ia.to_numpy(int), g.ib.to_numpy(int)
            ym = np.full(N_LN, np.nan)
            ym[reads] = Yw[i, reads]
            fitted = {"spec": RF.SPECS["B1"], "kind": "R", "coef": np.array([beta])}
            Lp = RF.predict_levels(fitted, SimpleNamespace(y_masked=ym), metals)
            u = np.zeros(N_LN)
            u[GD] = beta
            Lb = interp_levels(Yw[i], reads, metals, u)
            b1_new[gi] = Lb[ia] - Lb[ib]
            pl_maxabs = max(pl_maxabs, float(np.max(np.abs((Lp[ia] - Lp[ib]) - b1_new[gi]))))
            if t not in popx:
                continue
            ys = R.y.to_numpy(float)[gi]
            for c5 in variants:
                pt, dr, info = self.target_arms(t, reads, metals, beta, ia, ib, design=design, seed=seed, c5=c5,
                                                Yv=Yw, Dv=Dw, mode=mode)
                sfx = "" if c5 == PRIMARY_C5 else "@" + c5
                for a, v in pt.items():
                    cols.setdefault(a + sfx, np.full(n, np.nan))[gi] = v
                for a, v in dr.items():
                    if score:
                        cols.setdefault(a + sfx, np.full(n, np.nan))[gi] = ((v - ys[None, :]) ** 2).mean(0)
                    cols.setdefault("ndraw_nan_" + a + sfx, np.full(n, np.nan))[gi] = np.isnan(v).sum(0)
                if c5 == PRIMARY_C5:
                    for k in ("lvA", "lvB"):
                        cols.setdefault(k, np.full(n, np.nan))[gi] = info[k]
                    if mode != "repro":
                        for k in ("n_prim", "n_nc"):
                            cols.setdefault(k, np.full(n, np.nan))[gi] = info[k]
                        info_rows.append({"design": design, "split_seed": seed, "fold": fold, "cell_id": t,
                                          **{k: v for k, v in info.items() if k not in ("lvA", "lvB")}})
        checks = {"design": design, "rows": int(n), "rows_per_seed": R.groupby("split_seed").size().astype(int).to_dict(),
                  "cells": int(R.cell_id.nunique()), "surrogate": not real,
                  "b1_parquet_maxabs": float(np.nanmax(np.abs(b1_new - b1_par))) if real else None,
                  "b1_predict_levels_maxabs": pl_maxabs, "y_maxabs": y_maxabs, "b1_rebuilt_nan": int(np.isnan(b1_new).sum()),
                  "readset_mismatch_groups": rs_bad, "fold_mismatch_groups": fold_bad, "metals_mismatch_groups": met_bad,
                  "both_ends_read_rows": int((R.a_read & R.b_read).sum()),
                  "adjacent_lighter_first": bool(((R.ib - R.ia) >= 1).all() and (R.dZ.abs() == 1).all())}
        checks["b1_identity_passed"] = bool(real and checks["b1_parquet_maxabs"] <= B1_TOL)
        R["B1_par"] = b1_par                      # the round-4 OOF B1; the arms' B1 is rebuilt the same way they are
        R = R.assign(**{k: v for k, v in cols.items()})
        R["B1_rebuilt"] = b1_new
        R = R[R.cell_id.isin(popx)].reset_index(drop=True)
        if not real:
            R["B1"] = R["B1_rebuilt"]
        checks["popx_rows"] = int(len(R))
        self.annotate(R)
        if mode != "repro":
            for c5 in C5_VARIANTS:
                if score:
                    gk_lopo(R, c5)
            checks["finite_on_primary"] = finite_report(R)
        checks["seconds"] = round(time.time() - t0, 1)
        log(f"  {design}: {len(R)} POPX rows ({R.cell_id.nunique()} cells), {len(cols)} arm columns, "
            f"{checks['seconds']}s")
        return R, checks, pd.DataFrame(info_rows)

    def annotate(self, R: pd.DataFrame) -> None:
        """Strata flags and unit columns on a row frame (in place)."""
        st = self.static.set_index("cell_id")
        for c in ("chem_family", "mech_class", "acid", "scope_in", "lab_cons", "lab_lib", "n_ms", "n_ms_src", "n_ss",
                  "n_os", "n_sm", "n_dm", "lab_key_missing"):
            R[c] = R.cell_id.map(st[c]).to_numpy()
        prim = R.cell_id.isin(set(self.primary)).to_numpy()
        R["in_SRC"] = True
        R["in_PRIMARY"] = prim
        R["in_POPX0"] = prim & (R.n_ms == 0).to_numpy()
        R["in_POPX1"] = prim & (R.n_ms > 0).to_numpy()
        R["in_SCOPE_IN"] = prim & R.scope_in.astype(bool).to_numpy()
        R["in_NULL_PRIOR"] = prim & ~R.scope_in.astype(bool).to_numpy()
        lo = np.fmin(R.lvA.to_numpy(float), R.lvB.to_numpy(float)) < LOWD
        R["in_LOWD"] = prim & lo
        R["in_HIGHD"] = prim & ~lo
        R["in_SS_OS"] = prim & ((R.n_ss > 0) & (R.n_os > 0)).to_numpy()
        R["in_SM_DM"] = prim & ((R.n_sm > 0) & (R.n_dm > 0)).to_numpy()
        R["in_NO_LEAD2"] = prim & ~R.publication.isin(LEAD2).to_numpy()
        R["in_OUT_GIANT"] = prim & (R.cluster_cpg != GIANT).to_numpy()
        R["pair"] = R.A.astype(str) + "/" + R.B.astype(str)

    # ---- section-8 prediction intervention -----------------------------------------------------------------
    def intervention(self, design: str, *, Yv=None, seeds=None, log=print) -> dict:
        """Replace every unread label of each POPX target (seeded noise) and rebuild its arms: every point arm and
        every draw must be bit-identical, except the declared label-based sensitivities (NC05 / NC03), whose change
        is recorded.  Positive controls: the target in its own CLEAN pool (guard), a bit-identical copy in a strict
        pool (guard, the frozen item-4 link), and the target leaked into PRIM with the guard off (must change)."""
        t0 = time.time()
        Yb = self.Y if Yv is None else np.asarray(Yv, float)
        Db = self.Dv if Yv is None else detrend(Yb)
        R = CM.load_rows(design)
        betas = fold_betas(design)
        rng = np.random.default_rng(paths.seed_for(f"ELAB|INTERVENTION|{design}"))
        out = {"targets": 0, "target_seeds": 0, "mismatch": [], "label_based_changed": 0, "pc_self_caught": 0,
               "pc_self_applicable": 0, "pc_leak_caught": 0, "pc_leak_applicable": 0, "pc_bit_caught": 0,
               "pc_bit_applicable": 0}
        seen = set()
        for seed, fold, t, reads, metals, beta, gi in self.iter_targets(design, R, betas, cells=self.popx_src, seeds=seeds):
            g = R.iloc[gi]
            i = self.pos[t]
            ia, ib = g.ia.to_numpy(int), g.ib.to_numpy(int)
            un = [m for m in metals if m not in reads]
            Yp = Yb.copy()
            Yp[i, un] = rng.normal(0.0, 2.0, len(un))
            Dp = Db.copy()
            Dp[i] = qdev(Yp[i])
            kw = dict(design=design, seed=seed)
            a0 = self.target_arms(t, reads, metals, beta, ia, ib, Yv=Yb, Dv=Db, **kw)
            a1 = self.target_arms(t, reads, metals, beta, ia, ib, Yv=Yp, Dv=Dp, **kw)
            out["target_seeds"] += 1
            seen.add(t)
            for part in (0, 1):
                for a, v in a0[part].items():
                    same = np.array_equal(v, a1[part][a], equal_nan=True)
                    if a in LABEL_BASED:
                        out["label_based_changed"] += int(not same)
                    elif not same:
                        out["mismatch"].append(f"{seed}|{t}|{a}")
            if seed == (seeds or paths.DEV_SEEDS)[0]:
                out["pc_self_applicable"] += 1
                try:
                    self.target_arms(t, reads, metals, beta, ia, ib, Yv=Yb, Dv=Db, inject={"clean": [i]}, **kw)
                except LeakageError:
                    out["pc_self_caught"] += 1
                P = self.pools(t)
                if any(frozenset((t, self.ids[j])) in self.bit for j in P["src"]):
                    out["pc_bit_applicable"] += 1
                    try:
                        self.guard(i, P["src"], allow_bit=False, name="src as a strict pool")
                    except LeakageError:
                        out["pc_bit_caught"] += 1
                out["pc_leak_applicable"] += 1
                b0 = self.target_arms(t, reads, metals, beta, ia, ib, Yv=Yb, Dv=Db, inject={"prim": [i]}, guard=False, **kw)
                b1 = self.target_arms(t, reads, metals, beta, ia, ib, Yv=Yp, Dv=Dp, inject={"prim": [i]}, guard=False, **kw)
                out["pc_leak_caught"] += int(not np.array_equal(b0[0]["MX_LM10"], b1[0]["MX_LM10"], equal_nan=True))
        out["targets"] = len(seen)
        out["n_mismatch"] = len(out["mismatch"])
        out["mismatch"] = out["mismatch"][:20]
        out["checks_ok"] = bool(out["n_mismatch"] == 0 and out["pc_self_caught"] == out["pc_self_applicable"]
                                and out["pc_leak_caught"] == out["pc_leak_applicable"]
                                and out["pc_bit_caught"] == out["pc_bit_applicable"] and out["pc_bit_applicable"] > 0)
        out["seeds"] = list(paths.DEV_SEEDS if seeds is None else seeds)
        # s0-mechanics-5: a pass needs every dev seed and at least one target and one positive control of each kind
        out["coverage_full"] = bool(out["seeds"] == list(paths.DEV_SEEDS) and out["target_seeds"] > 0
                                    and out["pc_self_applicable"] > 0 and out["pc_leak_applicable"] > 0)
        out["passed"] = bool(out["checks_ok"] and out["coverage_full"])
        out["surrogate"] = Yv is not None
        out["seconds"] = round(time.time() - t0, 1)
        log(f"  intervention {design}: {out['target_seeds']} target-seeds, mismatches {out['n_mismatch']}, label-based "
            f"changed {out['label_based_changed']}, positive controls self {out['pc_self_caught']}/"
            f"{out['pc_self_applicable']}, bit {out['pc_bit_caught']}/{out['pc_bit_applicable']}, leak "
            f"{out['pc_leak_caught']}/{out['pc_leak_applicable']} ({out['seconds']}s)")
        return out


# ------------------------------------------------------------------------------------------------------------
# statistics on identical rows (E_* columns are expected squared errors)
# ------------------------------------------------------------------------------------------------------------
def err2(d: pd.DataFrame, arm: str) -> np.ndarray:
    v = d[arm].to_numpy(float)
    return v if arm.startswith("E_") else (v - d.y.to_numpy(float)) ** 2


def _seed_sums(d: pd.DataFrame, arms) -> pd.DataFrame:
    g = pd.DataFrame({"s": d.split_seed.to_numpy(), **{a: err2(d, a) for a in arms}})
    return g.groupby("s")[list(arms)].sum()


def r2_seeds(d: pd.DataFrame, arm: str) -> pd.Series:
    sst = d.groupby("split_seed").y.apply(lambda y: float(((y - y.mean()) ** 2).sum()))
    return 1.0 - _seed_sums(d, [arm])[arm] / sst.to_numpy()


def r2(d, arm) -> float:
    return float(r2_seeds(d, arm).mean())


def mse(d, arm) -> float:
    return float((_seed_sums(d, [arm])[arm] / d.groupby("split_seed").size().to_numpy()).mean())


def relsse_seeds(d: pd.DataFrame, cand: str, ref: str = "B1") -> pd.Series:
    s = _seed_sums(d, [cand, ref]) if cand != ref else _seed_sums(d, [cand])
    return 1.0 - s[cand] / s[ref]


def relsse(d, cand, ref="B1") -> float:
    return float(relsse_seeds(d, cand, ref).mean())


def relsse_pooled(d, cand, ref="B1") -> float:
    """Source convention (lab1_recompute_skA.rel): 1 - sum over every seed of SSE_cand / SSE_ref."""
    return float(1.0 - err2(d, cand).sum() / err2(d, ref).sum())


def unit_stats(d: pd.DataFrame, cand: str, ref: str, unit: str) -> pd.DataFrame:
    """r5.common.unit_stats with expected-error columns allowed (seed means of n, SSE_ref, SSE_cand; dSSE, L, rel)."""
    g = pd.DataFrame({"u": d[unit].astype(str).to_numpy(), "s": d.split_seed.to_numpy(), "one": 1.0,
                      "er": err2(d, ref), "ec": err2(d, cand)})
    ps = g.groupby(["u", "s"])[["one", "er", "ec"]].sum()
    m = ps.groupby(level="u").mean()
    out = pd.DataFrame({"n": m.one, "SSE_ref": m.er, "SSE_cand": m.ec, "n_seeds": ps.groupby(level="u").size()})
    out.index.name = unit
    out["dSSE"] = out.SSE_ref - out.SSE_cand
    with np.errstate(divide="ignore", invalid="ignore"):
        out["L"] = np.log(out.SSE_ref / out.SSE_cand)
        out["rel"] = 1.0 - out.SSE_cand / out.SSE_ref
    out["win"], out["loss"], out["tie"] = out.dSSE > 0, out.dSSE < 0, out.dSSE == 0
    return out


def leave_out(d: pd.DataFrame, cand: str, ref: str, k: int = 2, by: str = "publication") -> pd.DataFrame:
    """Seed-mean relative SSE with every k-subset of ``by`` left out (from per-unit per-seed sums)."""
    g = pd.DataFrame({"u": d[by].astype(str).to_numpy(), "s": d.split_seed.to_numpy(), "c": err2(d, cand), "r": err2(d, ref)})
    S = g.groupby(["u", "s"])[["c", "r"]].sum().unstack("s", fill_value=0.0)
    C, Rr, U = S["c"].to_numpy(), S["r"].to_numpy(), list(S.index)
    tc, tr = C.sum(0), Rr.sum(0)
    rows = []
    for q in itertools.combinations(range(len(U)), k):
        q = list(q)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = 1.0 - (tc - C[q].sum(0)) / (tr - Rr[q].sum(0))
        rows.append({"left_out": ";".join(U[j] for j in q), "rel": float(np.nanmean(v))})
    return pd.DataFrame(rows, columns=["left_out", "rel"])


def rel_without(d, cand, ref, units, by="publication") -> float:
    return relsse(d[~d[by].astype(str).isin(set(units))], cand, ref)


def infer(d: pd.DataFrame, cand: str, ref: str, unit: str, key: str, *, enum: bool = False) -> dict:
    """Unit statistics and the section-4.2 descriptive tests (r5.common) for one contrast and unit.  ``enum`` adds
    SF_POOL_p_enum, the SF_POOL p over all 2^G sign vectors, beside the registered p (kept as registered)."""
    us = unit_stats(d, cand, ref, unit)
    ut = CM.unit_tests(us, key)
    im = ut["IM_REL"]
    lb = im["mean"] - float(SS.t.ppf(0.95, im["df"])) * im["se"] if im["G"] >= 2 and np.isfinite(im["se"]) else np.nan
    return {"G": ut["G"], "W/L/T": ut["W/L/T"], "SF_POOL_p": ut["SF_POOL"]["p"], "SF_POOL_exact": ut["SF_POOL"]["exact"],
            "SF_POOL_p_enum": CM.sf_pool(us.dSSE, exact_max=len(us))["p"] if enum else np.nan,
            "SIGN_p": ut["SIGN"]["p"], "SF_LOG_p": ut["SF_LOG"]["p"] if ut["SF_LOG"] else np.nan,
            "IM_REL_mean": im["mean"], "IM_REL_t": im["t"], "IM_REL_p": im["p"], "IM_REL_lb95_1s": lb,
            "unit_median_rel": float(us.rel.median()), "top3_share": CM.top3_share(us.dSSE.to_numpy())}


def finite_report(R: pd.DataFrame) -> dict:
    """Fixed denominators: NaN count of every primary arm column on PRIMARY rows (must be 0)."""
    d = R[R.in_PRIMARY]
    return {c: int(d[c].isna().sum()) for c in R.columns if c.split("@")[0] in POINT_ARMS + DRAW_ARMS + DERIVED}


# ------------------------------------------------------------------------------------------------------------
# variant frames, the global-K controls, labels
# ------------------------------------------------------------------------------------------------------------
BASE_ARMS = POINT_ARMS + DRAW_ARMS + DERIVED


def vframe(R: pd.DataFrame, c5: str = PRIMARY_C5) -> pd.DataFrame:
    """Rows and arm columns of one C5 variant under the base arm names."""
    if c5 == PRIMARY_C5:
        return R
    keep = (R.publication != paths.C5) if c5 == "C5_out" else (R.cell_id != paths.C5_CELL)
    d = R[keep.to_numpy()]
    ren = {f"{a}@{c5}": a for a in BASE_ARMS if f"{a}@{c5}" in d.columns}
    return d.drop(columns=[a for a in ren.values() if a in d.columns]).rename(columns=ren)


def select_k(d: pd.DataFrame) -> list:
    """Global-K arms with the largest seed-mean relative SSE vs B1 on the rows of d (exact ties all returned)."""
    sc = {a: relsse(d, a) for a in K_ARMS} if len(d) else {}
    sc = {a: v for a, v in sc.items() if np.isfinite(v)}
    if not sc:
        return [K_ARMS[0]]
    best = max(sc.values())
    return [a for a in K_ARMS if a in sc and sc[a] == best]


def tied_mean(d: pd.DataFrame, arms) -> np.ndarray:
    return d[list(arms)].to_numpy(float).mean(1)


def gk_lopo(R: pd.DataFrame, c5: str) -> pd.DataFrame:
    """MX_GK_LOPO: for each publication p, the K (ties averaged) chosen on the PRIMARY rows of the other
    publications; publications outside PRIMARY use the K chosen on every PRIMARY row.  Adds the column (suffixed
    for a C5 variant) to R in place and returns the choices."""
    d = vframe(R, c5)
    P = d[d.in_PRIMARY]
    allk = select_k(P)
    choice = {p: select_k(P[P.publication != p]) for p in sorted(P.publication.unique())}
    col = np.full(len(R), np.nan)
    pos = pd.Series(np.arange(len(R)), index=R.index)
    for p, g in d.groupby("publication"):
        ks = choice.get(p, allk)
        col[pos[g.index].to_numpy()] = tied_mean(g, ks)
    R["MX_GK_LOPO" + ("" if c5 == PRIMARY_C5 else "@" + c5)] = col
    rows = [{"c5": c5, "publication": p, "K_chosen": ";".join(choice.get(p, allk)), "in_primary": p in choice}
            for p in sorted(d.publication.unique())]
    return pd.DataFrame(rows)


def _poison_rows(R: pd.DataFrame, rows: np.ndarray, rng) -> pd.DataFrame:
    """A copy of R with y and every arm / expected-error / draw column of ``rows`` replaced by seeded noise (NaN
    pattern kept): every label-derived quantity of those rows."""
    Rp = R.copy()
    cols = ["y"] + [c for c in Rp.columns if c != "y" and Rp[c].dtype.kind == "f"
                    and (c.split("@")[0] in BASE_ARMS + K_ARMS + ("MX_GK_O",) or c.startswith(("E_", "ndraw_")))]
    for c in cols:
        v = Rp[c].to_numpy(float).copy()
        sub = v[rows]
        noise = (np.abs(rng.normal(0.0, 0.3, len(sub))) if c.startswith("E_") else rng.normal(0.0, 1.0, len(sub)))
        v[rows] = np.where(np.isfinite(sub), noise, np.nan)
        Rp[c] = v
    return Rp


def gk_poison_check(R: pd.DataFrame, c5: str = PRIMARY_C5, key: str = "ELAB|GK_POISON") -> dict:
    """Section-8 fit poisoning of the LOPO K choice through the production code: for each PRIMARY publication p, y and
    every arm / expected-error / draw column of p's rows are replaced by seeded noise (s0-mechanics-15) and
    ``gk_lopo`` is rerun; p's chosen K must be unchanged.  Positive control: ``select_k`` on the PRIMARY rows with p
    included (a leaky choice) must change for at least one p."""
    d = vframe(R, c5)
    P = d[d.in_PRIMARY]
    pubs = sorted(P.publication.unique())
    rng = np.random.default_rng(paths.seed_for(key))
    c0 = gk_lopo(R.copy(), c5).set_index("publication").K_chosen
    bad, pc_caught = [], 0
    allk = select_k(P)
    for p in pubs:
        rows = np.flatnonzero(((R.publication == p) & R.index.isin(P.index)).to_numpy())
        Rp = _poison_rows(R, rows, rng)
        cp = gk_lopo(Rp.copy(), c5).set_index("publication").K_chosen
        if cp.get(p) != c0.get(p):
            bad.append(p)
        dp = vframe(Rp, c5)
        Pp = dp[dp.in_PRIMARY]
        pc_caught += int(select_k(Pp) != allk)
    return {"publications": int(len(pubs)), "changed": bad, "pc_runs": int(len(pubs)), "pc_caught": int(pc_caught),
            "passed": bool(not bad and pc_caught > 0),
            "scope": "y and every arm / E_* / ndraw column of the held-out publication's rows poisoned; gk_lopo rerun"}


def equals_headline(d: pd.DataFrame, h: pd.DataFrame, cand: str, ref: str) -> bool:
    """True when (y, cand, ref) on d's rows equal the headline's (y, MX_LM10, B1) on h (PRIMARY, C5_in) bitwise, on the
    same rows: then the contrast is the headline itself and only its registry flip key differs."""
    def same(a, b):
        return np.array_equal(d[a].to_numpy(float), h[b].to_numpy(float), equal_nan=True)
    return bool(d.index.equals(h.index) and same("y", "y") and same(cand, "MX_LM10") and same(ref, "B1"))


def changed_vs_c5_in(R: pd.DataFrame, d: pd.DataFrame, c5: str) -> tuple:
    """(point arms, draw arms) of a C5 variant frame d whose values on d's rows differ from their C5_in values; the
    point arms include MX_GK_LOPO.  (0, 0) for C5_in."""
    if c5 == PRIMARY_C5:
        return 0, 0
    base = R.loc[d.index]

    def n(arms):
        return int(sum(not np.array_equal(d[a].to_numpy(float), base[a].to_numpy(float), equal_nan=True)
                       for a in arms if a in d and a in base))
    return n(POINT_ARMS + DERIVED), n(DRAW_ARMS)


def with_oracle(d: pd.DataFrame):
    """d with MX_GK_O = the best global-K arm on d's own rows (ORACLE [O]; ties averaged)."""
    ks = select_k(d)
    return d.assign(MX_GK_O=tied_mean(d, ks)), ks


def label(*, design: str, c5: str, stratum: str, arms=(), lead=(), extra=(), sens: bool = False,
          seen_note: str = "") -> str:
    """The section-9.1 label of one number.  ``lead`` (e.g. CONSISTENCY CHECK, DIAGNOSTIC) goes first, ``extra`` after
    EXPLORATORY (seen), ``seen_note`` in brackets after it; SENSITIVITY once for BP, a C5 variant, a sensitivity
    stratum or arm, or ``sens``."""
    tags = [SEEN + (f" ({seen_note})" if seen_note else "")]
    if any(a in ORACLE_ARMS for a in arms):
        tags.insert(0, ORACLE)
    elif any(a in DIAG_ARMS for a in arms) or stratum in DIAG_STRATA:
        tags.insert(0, DIAG)
    tags = [t for t in lead if t not in tags] + tags + list(extra)
    if sens or design != PRIMARY_DESIGN or c5 != PRIMARY_C5 or stratum in SENS_STRATA or any(a in SENS_ARMS for a in arms):
        tags.append(SENS)
    return "; ".join(tags + [BEST25])


def rows_label(design: str) -> str:
    """Row label of rows_<design>.parquet (one per row): the design's label; the '@C5_out' / '@C5_cell_out' columns
    are C5 sensitivities (each arm's own label is in arms_ / contrasts_<design>.csv)."""
    return (f"{label(design=design, c5=PRIMARY_C5, stratum='PRIMARY')}; {NOT_NEXT_TO}; columns '@C5_out' / "
            f"'@C5_cell_out': {SENS}")


def seen_text(design: str, key: str | None = None, *, text: str | None = None, pool: str = "src", note: str = "") -> str:
    """A seen value (SEEN_VALUES[key] or ``text``) with its donor pool named; on BP the BP seen value where the source
    gives one (SEEN_BP), else marked 'BPG seen value'."""
    bp = design != PRIMARY_DESIGN
    if bp and key in SEEN_BP:
        v, tags = SEEN_BP[key], [SEEN_POOL[pool], "BP seen value"]
    else:
        v, tags = (str(SEEN_VALUES[key]) if text is None else text), [SEEN_POOL[pool]] + (["BPG seen value"] if bp else [])
    return f"{v} [{'; '.join(tags + ([note] if note else []))}]"


def stratum_frame(R: pd.DataFrame, c5: str, stratum: str) -> pd.DataFrame:
    d = vframe(R, c5)
    return d[d["in_" + stratum].to_numpy(bool)]


def arms_for(stratum: str) -> tuple:
    extra = ("MX_LM10_SS", "MX_LM10_OS") if stratum == "SS_OS" else (("MX_LM10_SM", "MX_LM10_DM") if stratum == "SM_DM" else ())
    base = tuple(a for a in BASE_ARMS if a not in ("MX_LM10_SS", "MX_LM10_OS", "MX_LM10_SM", "MX_LM10_DM"))
    return base + ("MX_GK_O",) + extra


def contrasts_for(stratum: str, c5: str) -> tuple:
    con = FULL if (c5 == PRIMARY_C5 and stratum in FULL_STRATA) else MAIN
    return con + (SS_CON if stratum == "SS_OS" else ()) + (SM_CON if stratum == "SM_DM" else ()) + \
        ((("MX_LM10_NB", "B1"), ("MX_LM10_NC", "B1")) if stratum == "SRC" else ())


def plan(c5: str) -> tuple:
    return STRATA if c5 == PRIMARY_C5 else C5_STRATA


# ------------------------------------------------------------------------------------------------------------
# tables
# ------------------------------------------------------------------------------------------------------------
def t_populations(R: pd.DataFrame, design: str, pools: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for c5 in C5_VARIANTS:
        for S in plan(c5):
            d = stratum_frame(R, c5, S)
            ps = d.groupby("split_seed").size()
            r = {"design": design, "c5": c5, "stratum": S, "cells": int(d.cell_id.nunique()),
                 "publications": int(d.publication.nunique()), "labs_cons": int(d.lab_cons.nunique()),
                 "labs_lib": int(d.lab_lib.nunique()), "clusters": int(d.cluster_cpg.nunique()),
                 "rows_per_seed": float(ps.mean()) if len(ps) else 0.0,
                 "rows_per_seed_distinct": ";".join(str(int(x)) for x in sorted(set(ps))),
                 "cells_with_nearcopy_donor": int(d[d.n_nc > 0].cell_id.nunique()) if "n_nc" in d else np.nan,
                 "rows_prim_empty": int((d.n_prim == 0).sum()) if "n_prim" in d else np.nan,
                 "label": "counts (no score)"}
            rows.append(r)
    return pd.DataFrame(rows)


def t_arms(R: pd.DataFrame, design: str) -> pd.DataFrame:
    rows = []
    for c5 in C5_VARIANTS:
        for S in plan(c5):
            d = stratum_frame(R, c5, S)
            if d.empty:
                continue
            d, ks = with_oracle(d)
            for a in arms_for(S):
                if a not in d or d[a].isna().any():
                    continue
                rs, rl = r2_seeds(d, a), relsse_seeds(d, a)
                rows.append({"design": design, "c5": c5, "stratum": S, "arm": a, "R2": float(rs.mean()), "MSE": mse(d, a),
                             "relSSE_vs_B1": float(rl.mean()), "relSSE_vs_B1_pooled": relsse_pooled(d, a),
                             "MAE_balanced": CM.bal_mae(d, a) if not a.startswith("E_") else np.nan,
                             "R2_per_seed": ";".join(f"{v:.4f}" for v in rs), "rel_per_seed": ";".join(f"{v:.4f}" for v in rl),
                             "oracle_K": ";".join(ks) if a == "MX_GK_O" else "",
                             "label": label(design=design, c5=c5, stratum=S, arms=(a,))})
    return pd.DataFrame(rows)


def t_contrasts(R: pd.DataFrame, design: str) -> pd.DataFrame:
    """Every registered contrast per C5 variant, stratum and unit.  Descriptive columns (A2): ``equals_headline``
    (the contrast is MX_LM10 - B1 on PRIMARY, C5_in, bitwise), the point / draw arms a C5 variant changes on the
    stratum's rows, and SF_POOL_p_enum (all 2^G sign vectors) on the publication rows that equal the headline, whose
    registered random-flip p differs only by its registry key."""
    rows = []
    h = stratum_frame(R, PRIMARY_C5, "PRIMARY")
    for c5 in C5_VARIANTS:
        for S in plan(c5):
            d = stratum_frame(R, c5, S)
            if d.publication.nunique() < 2:
                continue
            chg = changed_vs_c5_in(R, d, c5)
            d, ks = with_oracle(d)
            for cand, ref in contrasts_for(S, c5):
                if d[cand].isna().any() or d[ref].isna().any():
                    continue
                same = equals_headline(d, h, cand, ref)
                rl = relsse_seeds(d, cand, ref)
                lp = CM.lopo(d, lambda x: relsse(x, cand, ref), by="publication")
                l2 = leave_out(d, cand, ref, 2).dropna(subset=["rel"]).reset_index(drop=True)
                usp = unit_stats(d, cand, ref, "publication")
                top = list(usp.dSSE.sort_values(ascending=False).index[:3])
                base = {"design": design, "c5": c5, "stratum": S, "contrast": f"{cand} - {ref}", "cand": cand, "ref": ref,
                        "rel": float(rl.mean()), "rel_pooled": relsse_pooled(d, cand, ref), "dR2": r2(d, cand) - r2(d, ref),
                        "rel_per_seed": ";".join(f"{v:.4f}" for v in rl), "oracle_K": ";".join(ks) if "MX_GK_O" in (cand, ref) else "",
                        "LOPO_lo": lp["lo"], "LOPO_hi": lp["hi"], "LOPO_P": lp["P"], "LOPO_min": lp["min"], "LOPO_min_without": lp["min_by"],
                        "LTPO_min": float(l2.rel.min()) if len(l2) else np.nan,
                        "LTPO_min_without": l2.left_out[l2.rel.idxmin()] if len(l2) else "",
                        "without_LEAD2": rel_without(d, cand, ref, LEAD2), "without_top3": rel_without(d, cand, ref, top),
                        "top3_publications": ";".join(top), "equals_headline": same,
                        "point_arms_changed_vs_C5_in": chg[0], "draw_arms_changed_vs_C5_in": chg[1]}
                for unit in UNITS:
                    key = f"ELAB|{design}|{c5}|{S}|{cand}-{ref}|{unit}"
                    rows.append({**base, "unit": unit, **infer(d, cand, ref, unit, key, enum=same and unit == "publication"),
                                 "label": label(design=design, c5=c5, stratum=S, arms=(cand, ref))})
    return pd.DataFrame(rows)


def t_units(R: pd.DataFrame, design: str) -> pd.DataFrame:
    rows = []
    for S in ("PRIMARY", "POPX0"):
        d = stratum_frame(R, PRIMARY_C5, S)
        for cand, ref in MAIN + (("MX_LM10_NB", "B1"),):
            for unit in UNITS:
                us = unit_stats(d, cand, ref, unit)
                meta = d.groupby(unit).agg(cells=("cell_id", "nunique"), publications=("publication", "nunique"))
                us = us.join(meta)
                for u, r in us.iterrows():
                    rows.append({"design": design, "stratum": S, "contrast": f"{cand} - {ref}", "unit": unit, "unit_id": u,
                                 **{k: (bool(v) if k in ("win", "loss", "tie") else v) for k, v in r.items()},
                                 "label": label(design=design, c5=PRIMARY_C5, stratum=S, arms=(cand, ref))})
    return pd.DataFrame(rows)


def t_lopo(R: pd.DataFrame, design: str) -> pd.DataFrame:
    rows = []
    for S in ("PRIMARY", "POPX0"):
        d = stratum_frame(R, PRIMARY_C5, S)
        for cand, ref in MAIN:
            lp = CM.lopo(d, lambda x: relsse(x, cand, ref), by="publication")
            for p, v in lp["values"].items():
                rows.append({"design": design, "stratum": S, "contrast": f"{cand} - {ref}", "left_out": p, "rel": v,
                             "point": lp["point"], "label": label(design=design, c5=PRIMARY_C5, stratum=S, arms=(cand, ref))})
    return pd.DataFrame(rows)


def t_ltpo(R: pd.DataFrame, design: str) -> pd.DataFrame:
    d = stratum_frame(R, PRIMARY_C5, "PRIMARY")
    out = []
    for cand, ref in (("MX_LM10", "B1"), ("MX_LM10", "E_PCM"), ("MX_LM10", "E_PBC")):
        l2 = leave_out(d, cand, ref, 2).assign(design=design, stratum="PRIMARY", contrast=f"{cand} - {ref}",
                                               label=label(design=design, c5=PRIMARY_C5, stratum="PRIMARY", arms=(cand, ref)))
        out.append(l2)
    return pd.concat(out, ignore_index=True)


def t_globalk(R: pd.DataFrame, design: str) -> pd.DataFrame:
    rows = []
    for S in FULL_STRATA:
        d = stratum_frame(R, PRIMARY_C5, S)
        ks = select_k(d)
        for a in K_ARMS:              # a fixed K is EXPLORATORY (seen); only the best K on these rows (MX_GK_O) is [O]
            lead = (f"{ORACLE} (the best K of the grid on its own rows = MX_GK_O; optimistic for the control)",) \
                if a in ks else ()
            rows.append({"design": design, "stratum": S, "arm": a, "relSSE_vs_B1": relsse(d, a),
                         "MX_LM10_rel_vs_arm": relsse(d, "MX_LM10", a), "oracle_best": a in ks,
                         "label": label(design=design, c5=PRIMARY_C5, stratum=S, arms=(a,), lead=lead)})
    ch = pd.concat([gk_lopo(R.copy(), c5) for c5 in C5_VARIANTS], ignore_index=True)
    ch = ch.assign(design=design, stratum="LOPO_CHOICE",
                   label=[label(design=design, c5=c, stratum="PRIMARY", arms=("MX_GK_LOPO",),
                                seen_note="K chosen without the publication's rows") for c in ch.c5])
    return pd.concat([pd.DataFrame(rows), ch], ignore_index=True)


def t_classes(R: pd.DataFrame, design: str) -> pd.DataFrame:
    d = stratum_frame(R, PRIMARY_C5, "PRIMARY")
    rows = []
    for col in ("chem_family", "mech_class"):
        for cls, g in d.groupby(col):
            r = {"design": design, "by": col, "class": cls, "cells": int(g.cell_id.nunique()),
                 "publications": int(g.publication.nunique()), "labs_cons": int(g.lab_cons.nunique()),
                 "rows_per_seed": float(len(g) / g.split_seed.nunique()), "scope_in_cells": int(g[g.scope_in.astype(bool)].cell_id.nunique())}
            for cand, ref in (("MX_LM10", "B1"), ("E_PCM", "B1"), ("MX_LM10", "E_PCM")):
                r[f"rel {cand} - {ref}"] = relsse(g, cand, ref)
            r["harm_flag_reading"] = bool(r["labs_cons"] >= 3 and r["rel MX_LM10 - B1"] < -0.05)
            r["label"] = label(design=design, c5=PRIMARY_C5, stratum="PRIMARY", arms=("MX_LM10", "B1", "E_PCM"),
                               lead=(f"{CC} (harm-flag reading in-corpus; the flag belongs to the external leg)",))
            rows.append(r)
    return pd.DataFrame(rows)


def _pick(C: pd.DataFrame, stratum: str, contrast: str, unit: str = "publication", c5: str = PRIMARY_C5) -> dict:
    q = C[(C.stratum == stratum) & (C.contrast == contrast) & (C.unit == unit) & (C.c5 == c5)]
    return q.iloc[0].to_dict() if len(q) else {}


def t_rules(T: dict, design: str) -> pd.DataFrame:
    """Section-6.3 report rules: in-corpus readings (no kill fires in-corpus; the external leg is data-gated).  A2:
    every W/L/T names its unit; every seen value names its donor pool (lab-1's SRC pool before near-copy cleaning, or
    Skeptic B's cleaned SRC pool) and ``in_corpus_on_seen_pool`` gives the like-for-like in-corpus value where a
    registered arm uses that pool (MX_LM10_NB, MX_LM10_NC); every label goes through ``label`` (BP: SENSITIVITY)."""
    C, K, P = T["contrasts"], T["classes"], T["populations"]

    def fmt(r):
        if not r:
            return "NOT AVAILABLE"
        p = f"SF_POOL p {r['SF_POOL_p']:.4f}"
        if np.isfinite(r.get("SF_POOL_p_enum", np.nan)):
            p += f" ({CM.B_FLIP:,} random flips; exact enumeration over 2^{r['G']} sign vectors {r['SF_POOL_p_enum']:.4f})"
        return (f"{r['rel']:+.4f} [LOPO {r['LOPO_lo']:+.4f}, {r['LOPO_hi']:+.4f}]; {UNIT_WORD[r['unit']]} W/L/T "
                f"{r['W/L/T']}; {p}; top-3 share {r['top3_share']:.3f}")

    def short(r):
        if not r:
            return "NOT AVAILABLE"
        return f"{r['rel']:+.4f} [LOPO {r['LOPO_lo']:+.4f}, {r['LOPO_hi']:+.4f}]; {UNIT_WORD[r['unit']]} W/L/T {r['W/L/T']}"

    def lab(stratum="PRIMARY", arms=("MX_LM10", "B1"), **k):
        return label(design=design, c5=PRIMARY_C5, stratum=stratum, arms=arms, **k)

    def seen(key=None, **k):
        return seen_text(design, key, **k)
    rule_lab = "rule (no number)" + (f"; {SENS}" if design != PRIMARY_DESIGN else "")
    same_pool = "= in_corpus_value (the same pool)"
    rows = []

    def add(rule, value, seen_value, on_seen_pool, outcome, label_):
        rows.append({"design": design, "rule": rule, "in_corpus_value": value, "seen": seen_value,
                     "in_corpus_on_seen_pool": on_seen_pool, "outcome": outcome, "label": label_})
    head = _pick(C, "PRIMARY", "MX_LM10 - B1", unit=PRIMARY_UNIT)
    head_pub = _pick(C, "PRIMARY", "MX_LM10 - B1")
    nb, nc = _pick(C, "SRC", "MX_LM10_NB - B1"), _pick(C, "PRIMARY", "MX_LM10_NC - B1")
    add(f"headline: MX_LM10 - B1 on POPX with near-copies excluded (PRIMARY; unit {UNIT_TEXT[PRIMARY_UNIT]})", fmt(head),
        "n/a (lab-1 used publication units)", "", "reported, not tested", lab())
    for u in ("lab_lib", "publication"):
        add(f"headline beside it: MX_LM10 - B1 (PRIMARY; unit {UNIT_TEXT[u]})", fmt(_pick(C, "PRIMARY", "MX_LM10 - B1", unit=u)),
            seen("cleaned_rel", pool="nc") if u == "publication" else "n/a",
            f"MX_LM10_NC - B1 on PRIMARY: {fmt(nc)}" if u == "publication" else "", "reported, not tested",
            lab(sens=True) if u == "lab_lib" else lab(extra=("bridge",)))
    add("bridge: MX_LM10_NB - B1 (lab-1 pool as published) on SRC", fmt(nb), seen("POPX_B1_to_MX_LM10"), same_pool,
        "reported", lab("SRC", ("MX_LM10_NB", "B1")))
    add("bridge: MX_LM10_NC - B1 (near-copy rule only; Skeptic B)", fmt(nc), seen("cleaned_rel", pool="nc"), same_pool,
        "reported", lab(arms=("MX_LM10_NC", "B1")))
    idn = C[C.equals_headline.astype(bool) & (C.unit == "publication")] if "equals_headline" in C else C.iloc[:0]
    c5r = C[(C.stratum == "PRIMARY") & (C.c5 != PRIMARY_C5) & (C.contrast == "MX_LM10 - B1") & (C.unit == "publication")]
    enum = sorted({f"over all 2^{int(r['G'])} sign vectors {r['SF_POOL_p_enum']:.4f}" for _, r in idn.iterrows()
                   if np.isfinite(r.get("SF_POOL_p_enum", np.nan))})
    add("note: contrasts identical to the headline MX_LM10 - B1 (MX_LM10_NC; the C5 variants) and the exact SF_POOL p",
        f"MX_LM10_NC equals MX_LM10 bitwise on every PRIMARY row: {'yes' if nc and bool(nc.get('equals_headline')) else 'NO'}; "
        "point arms (incl. MX_GK_LOPO) changed on PRIMARY by "
        + (" / ".join(f"{r['c5']} {int(r['point_arms_changed_vs_C5_in'])}" for _, r in c5r.iterrows()) or "NOT AVAILABLE")
        + f" of {len(POINT_ARMS + DERIVED)} (E_* draw arms changed: "
        + (" / ".join(f"{r['c5']} {int(r['draw_arms_changed_vs_C5_in'])}" for _, r in c5r.iterrows()) or "NOT AVAILABLE")
        + f" of {len(DRAW_ARMS)}; their draws are keyed per variant); registered SF_POOL p of the identical "
        f"publication-unit contrasts ({CM.B_FLIP:,} random flips, keyed per row): "
        + ("; ".join(f"{r['stratum']} {r['c5']} {r['contrast']} {r['SF_POOL_p']:.4f}" for _, r in idn.iterrows()) or "none")
        + f"; exact enumeration {' / '.join(enum) or 'NOT AVAILABLE'}",
        "", "", "note (descriptive; the registered p is kept)", lab(sens=True))
    add("POPX0 (new extractants): MX_LM10 - B1", fmt(_pick(C, "POPX0", "MX_LM10 - B1")), seen("POPX0_rel"),
        f"MX_LM10_NB - B1 on POPX0: {fmt(_pick(C, 'POPX0', 'MX_LM10_NB - B1'))}", "reported", lab("POPX0"))
    for con, sk in (("MX_LM10 - E_PBC", "vs_E_PBC"), ("MX_LM10 - E_PCM", "vs_E_PCM"), ("MX_LM10 - E_RM", "vs_E_RM")):
        add(f"control: {con} (unit {UNIT_TEXT[PRIMARY_UNIT]})", fmt(_pick(C, "PRIMARY", con, unit=PRIMARY_UNIT)), "n/a", "",
            "reported", lab(arms=tuple(con.split(" - "))))
        add(f"control: {con} (unit {UNIT_TEXT['publication']})", fmt(_pick(C, "PRIMARY", con)), seen(sk), NOT_ON_SRC,
            "reported", lab(arms=tuple(con.split(" - ")), extra=("bridge",)))
    gk = _pick(C, "PRIMARY", "MX_LM10 - MX_GK_O")
    add("control: MX_LM10 - best global-K shrinkage (oracle K on the scored rows)", fmt(gk) + (f"; K {gk.get('oracle_K')}" if gk else ""),
        seen("vs_best_global_K"), NOT_ON_SRC, "reported", lab(arms=("MX_LM10", "MX_GK_O")))
    add("control: MX_LM10 - MX_GK_LOPO (K chosen without the publication)", fmt(_pick(C, "PRIMARY", "MX_LM10 - MX_GK_LOPO")),
        "not run by lab-1", "", "reported", lab(arms=("MX_LM10", "MX_GK_LOPO")))
    add("leave-two-publications-out (without pub_126bbe7976, pub_1bd76c48d9) and minimum over all pairs",
        (f"{head_pub['without_LEAD2']:+.4f}; min {head_pub['LTPO_min']:+.4f} without {head_pub['LTPO_min_without']}"
         if head_pub else "NOT AVAILABLE"), seen("leave_two_publications_out"),
        (f"MX_LM10_NB - B1 on SRC: {nb['without_LEAD2']:+.4f}; min {nb['LTPO_min']:+.4f} without {nb['LTPO_min_without']}"
         if nb else "NOT AVAILABLE"), "reported", lab())
    add("top-3 publication share of the MX_LM10 - B1 gain", f"{head_pub['top3_share']:.3f}" if head_pub else "NOT AVAILABLE",
        seen("top3_share"), f"MX_LM10_NB - B1 on SRC: {nb['top3_share']:.3f}" if nb else "NOT AVAILABLE", "reported", lab())
    add("top-3 laboratory share (conservative key) of the MX_LM10 - B1 gain",
        f"{head['top3_share']:.3f}" if head else "NOT AVAILABLE", "n/a", "", "reported", lab())
    ssb, osb, ss = (_pick(C, "SS_OS", c) for c in ("MX_LM10_SS - B1", "MX_LM10_OS - B1", "MX_LM10_SS - MX_LM10_OS"))
    pso = P[(P.stratum == "SS_OS") & (P.c5 == PRIMARY_C5)]
    nso = (f"{int(pso.cells.iloc[0])} cells, {int(pso.publications.iloc[0])} publications, {int(pso.clusters.iloc[0])} "
           f"source cluster{'' if int(pso.clusters.iloc[0]) == 1 else 's'}") if len(pso) else "counts NOT AVAILABLE"
    add(f"same-series vs other-series donors (experiment_series_id shared; SS_OS, {nso}): MX_LM10_SS - B1, "
        "MX_LM10_OS - B1 and MX_LM10_SS - MX_LM10_OS",
        f"MX_LM10_SS - B1 {short(ssb)} | MX_LM10_OS - B1 {short(osb)} | MX_LM10_SS - MX_LM10_OS {fmt(ss)}",
        seen("same_series_vs_other_series", note="MX_LM10_SS - B1 vs MX_LM10_OS - B1"), NOT_ON_SRC, "reported",
        lab("SS_OS", ("MX_LM10_SS", "MX_LM10_OS", "B1")))
    lo, hi = _pick(C, "PRIMARY", "MX_LM10_LO - B1"), _pick(C, "PRIMARY", "MX_LM10_HI - B1")
    lw, hw = _pick(C, "LOWD", "MX_LM10 - B1"), _pick(C, "HIGHD", "MX_LM10 - B1")
    add("low-D band split (rows min level < -2 vs >= -2; blend at metals < -2 vs >= -2)",
        f"rows LOWD {lw.get('rel', np.nan):+.4f} / HIGHD {hw.get('rel', np.nan):+.4f}; metals LO {lo.get('rel', np.nan):+.4f} "
        f"/ HI {hi.get('rel', np.nan):+.4f}", seen(text="sharing r 0.53 at log D < -2 (descriptive, lab.md 6)"), NOT_ON_SRC,
        "reported", lab(arms=("MX_LM10_LO", "MX_LM10_HI", "MX_LM10", "B1")))
    for lab_unit in ("lab_cons", "lab_lib"):
        r = _pick(C, "PRIMARY", "MX_LM10 - B1", unit=lab_unit)
        if r:
            would = bool(r["unit_median_rel"] <= 0 or not (r["IM_REL_lb95_1s"] > 0))
            add(f"external-kill statistic read in-corpus ({lab_unit}): laboratory-median rel SSE and one-sided 95 % lower bound",
                f"median {r['unit_median_rel']:+.4f}; IM lower bound {r['IM_REL_lb95_1s']:+.4f}; G {r['G']}",
                "n/a (external statistic)", "", f"would {'KILL' if would else 'pass'} if applied (it is not: external leg only)",
                lab(lead=(DIAG,)))
    nar = _pick(C, "PRIMARY", "MX_LM10 - E_PCM")
    add("narrowing reading: MX_LM10 - E_PCM <= 0 -> 'level-conditional weighting, not campaign'",
        f"{nar.get('rel', np.nan):+.4f}", seen("vs_E_PCM"), NOT_ON_SRC,
        ("would narrow" if nar and nar["rel"] <= 0 else "would not narrow") + " (in-corpus reading; the rule belongs to the external leg)",
        lab(arms=("MX_LM10", "E_PCM"), lead=(CC,)))
    harm = K[(K.by == "chem_family") & K.harm_flag_reading] if len(K) else K
    add("harm-flag reading: a class (chem_family) with >= 3 laboratory units losing > 5 %",
        ";".join(harm["class"].astype(str)) or "none", seen("by_class"), NOT_ON_SRC, "reading only",
        lab(arms=("MX_LM10", "B1", "E_PCM"), lead=(CC,)))
    si, nu = _pick(C, "SCOPE_IN", "MX_LM10 - B1"), _pick(C, "NULL_PRIOR", "MX_LM10 - B1")
    add("scope: amide O-donor solvating extractants in nitrate media vs the declared null-prior strata",
        f"SCOPE_IN {si.get('rel', np.nan):+.4f}; NULL_PRIOR {nu.get('rel', np.nan):+.4f}", "n/a", "", "reported",
        lab("SCOPE_IN"))
    add("external leg", "NOT RUN (Stage 1 is in-corpus; data-gated, section 7)", "supply: about 3-6 laboratory units "
        "(I2 lab-1)", "", EXTERNAL_RULE, rule_lab)
    add("label rule", "every number carries 'best of about 25 arms on seen rows; expect regression'", "", "", "applied",
        rule_lab)
    return pd.DataFrame(rows)


def all_tables(R: pd.DataFrame, design: str) -> dict:
    T = {"populations": t_populations(R, design), "arms": t_arms(R, design), "contrasts": t_contrasts(R, design),
         "units": t_units(R, design), "lopo": t_lopo(R, design), "ltpo": t_ltpo(R, design),
         "globalK": t_globalk(R, design), "classes": t_classes(R, design)}
    T["rules"] = t_rules(T, design)
    return T


def restated_targets(T: dict, design: str) -> dict:
    """Brief 6.3: the reproduction targets restated on the cleaned population (PRIMARY / POPX0)."""
    P, A = T["populations"], T["arms"]
    out = {"label": label(design=design, c5=PRIMARY_C5, stratum="PRIMARY",
                          seen_note="restated on the near-copy-cleaned population"), "design": design}
    for S in ("PRIMARY", "POPX0"):
        p = P[(P.stratum == S) & (P.c5 == PRIMARY_C5)].iloc[0]
        a = A[(A.stratum == S) & (A.c5 == PRIMARY_C5)].set_index("arm")
        out[S] = {"cells": int(p.cells), "publications": int(p.publications), "rows_per_seed": float(p.rows_per_seed),
                  "R2_B1": float(a.R2.get("B1", np.nan)), "R2_MX_LM10": float(a.R2.get("MX_LM10", np.nan))}
    return out


def per_item(T: dict, design: str, *, seal_tag: str, leakage: str, seconds: float) -> dict:
    """The section-9.3 per-item table (r5.common.per_item_table)."""
    P, A, C = T["populations"], T["arms"], T["contrasts"]
    p = P[(P.stratum == "PRIMARY") & (P.c5 == PRIMARY_C5)].iloc[0]
    a = A[(A.stratum == "PRIMARY") & (A.c5 == PRIMARY_C5)]
    cm = C[(C.stratum == "PRIMARY") & (C.c5 == PRIMARY_C5)]
    units = cm[cm.contrast.isin([f"{x} - {y}" for x, y in MAIN])][["contrast", "unit", "G", "W/L/T", "SF_POOL_p", "SIGN_p",
                                                                    "IM_REL_p", "unit_median_rel", "label"]]
    lt = cm[(cm.unit == "publication") & cm.contrast.isin([f"{x} - {y}" for x, y in MAIN])][
        ["contrast", "rel", "LOPO_lo", "LOPO_hi", "LOPO_min", "LOPO_min_without", "LTPO_min", "without_LEAD2", "top3_share"]]
    # A2: beside the registered p, the exact-enumeration p where the contrast equals the headline bitwise; and the
    # point / draw arms each C5 variant changes on PRIMARY
    c5 = C[(C.stratum == "PRIMARY") & C.unit.isin([PRIMARY_UNIT, "publication"]) & (C.contrast == "MX_LM10 - B1")][
        ["c5", "unit", "rel", "W/L/T", "SF_POOL_p", "SF_POOL_p_enum", "LOPO_lo", "LOPO_hi", "equals_headline",
         "point_arms_changed_vs_C5_in", "draw_arms_changed_vs_C5_in", "label"]]
    dup = C[(C.stratum == "PRIMARY") & (C.c5 == PRIMARY_C5) & C.unit.isin([PRIMARY_UNIT, "publication"])
            & C.cand.isin(["MX_LM10_NB", "MX_LM10_NC", "MX_LM10_OFS", "MX_LM10_NC05", "MX_LM10_NC03"])][
        ["contrast", "unit", "rel", "W/L/T", "SF_POOL_p", "SF_POOL_p_enum", "LOPO_lo", "LOPO_hi", "equals_headline", "label"]]
    head = _pick(C, "PRIMARY", "MX_LM10 - B1", unit=PRIMARY_UNIT)
    return {"item": "1.3 E-LAB (lab-1): campaign calibration of a new extractant (E-CT annex)",
            "population": f"PRIMARY (POPX, near-copies excluded): {int(p.cells)} cells, {int(p.publications)} publications, "
                          f"{int(p.labs_cons)} / {int(p.labs_lib)} laboratories (conservative / liberal, PROVISIONAL key), "
                          f"{int(p.clusters)} source clusters, {p.rows_per_seed:.0f} rows per seed",
            "design": f"{design} ({'primary' if design == PRIMARY_DESIGN else 'sensitivity'})",
            "read_budget": "4 reads (DOPT_Q, code budget k = 3; E-A's read sets); the campaign's other series in full",
            "arms": ", ".join(BASE_ARMS + ("MX_GK_O",)),
            "metrics": a[["arm", "R2", "MSE", "relSSE_vs_B1", "MAE_balanced", "label"]],
            "per_seed": a[["arm", "R2_per_seed", "rel_per_seed"]],
            "units": units, "lopo_top3": lt, "c5": c5, "duplicates": dup, "leakage": leakage,
            "compute": f"{seconds:.0f} s (one process)",
            "label": f"{label(design=design, c5=PRIMARY_C5, stratum='PRIMARY')}; {NOT_NEXT_TO}; seal {seal_tag}",
            "changed": (f"On the near-copy-cleaned POPX the level-matched campaign shape gives {head.get('rel', np.nan):+.4f} "
                        f"relative SSE vs B1 on identical rows, with laboratory (conservative, PROVISIONAL key) W/L/T "
                        f"{head.get('W/L/T')} and SF_POOL p {head.get('SF_POOL_p', np.nan):.4f} (publication units, the "
                        "bridge to lab-1: seen before cleaning +0.090, after +0.076); it remains a seen, best-of-about-25 "
                        "exploratory gain whose mechanism (campaign vs level-driven weighting) the controls only bound.")
            if head else "NOT AVAILABLE",
            "next_step": "External leg data-gated (section 7): descriptive unless >= 8 independent laboratories with >= 2 "
                         "extractants x >= 6 Ln in one campaign; no in-corpus kill fires."}


# ------------------------------------------------------------------------------------------------------------
# reproduction (section 6.0) and the smoke surrogate
# ------------------------------------------------------------------------------------------------------------
def reproduction(C: Campaign, R: pd.DataFrame, checks: dict, design: str) -> dict:
    """TECHNICAL SMOKE: B1 identity on every REF row; POPX / POPX0 counts; the SRC-pool arms against the idea's
    lab_rows2.parquet (MX_LM10, MX_LM05, MX_LM20, MX) and n_MS against lab_rows.parquet; on BPG, B1 0.444 and MX_LM10
    0.494 on POPX (3 decimals).  Every target must match or the item stops."""
    out = {"label": SMOKE, "design": design, "checks": {}}
    ok = {}
    ok["b1_identity"] = checks["b1_parquet_maxabs"] is not None and checks["b1_parquet_maxabs"] <= B1_TOL
    out["checks"]["b1_parquet_maxabs"] = checks["b1_parquet_maxabs"]
    out["checks"]["b1_rows"] = checks["rows"]
    st = C.static.set_index("cell_id")
    R = R.assign(n_ms_src=R.cell_id.map(st.n_ms_src).to_numpy())
    for nm, d in (("POPX", R), ("POPX0", R[R.n_ms_src == 0])):
        got = {"cells": int(d.cell_id.nunique()), "publications": int(d.publication.nunique()),
               "rows_per_seed": int(round(len(d) / max(d.split_seed.nunique(), 1)))}
        out["checks"][nm + "_counts"] = got
        ok[nm + "_counts"] = got == REPRO[nm]
    L2 = pd.read_parquet(LAB_ROWS2, columns=["design", "split_seed", "cell_id", "pair", "MX_LM10", "MX_LM05", "MX_LM20", "MX"])
    L1 = pd.read_parquet(LAB_ROWS, columns=["design", "split_seed", "cell_id", "pair", "n_MS"])
    key = ["split_seed", "cell_id", "pair"]
    m = R.merge(L2[L2.design == design].drop(columns="design"), on=key, how="left", validate="one_to_one") \
        .merge(L1[L1.design == design].drop(columns="design"), on=key, how="left", validate="one_to_one")
    for mine, ref in (("MX_LM10_NB", "MX_LM10"), ("MX_LM05_NB", "MX_LM05"), ("MX_LM20_NB", "MX_LM20"), ("MX_K6p1_NB", "MX")):
        dv = (m[mine] - m[ref]).abs()
        out["checks"][f"{mine}_vs_lab_rows2_{ref}_maxabs"] = float(dv.max())
        ok[f"{mine}_vs_{ref}"] = bool(m[ref].notna().all() and dv.max() <= REPRO_TOL)
    out["checks"]["rows_compared"] = int(len(m))
    out["checks"]["n_MS_mismatch_rows"] = int((m.n_MS != m.n_ms_src).sum())
    ok["n_MS"] = out["checks"]["n_MS_mismatch_rows"] == 0
    r2v = {a: r2(R, a) for a in ("B1", "MX_LM10_NB")}
    r2v0 = {a: r2(R[R.n_ms_src == 0], a) for a in ("B1", "MX_LM10_NB")}
    out["checks"]["R2_POPX"] = r2v
    out["checks"]["R2_POPX0"] = r2v0
    if design in REPRO["R2_3dp"]:
        for a, v in REPRO["R2_3dp"][design].items():
            ok[f"R2_{a}_3dp"] = round(r2v[a], 3) == v
    out["consistency"] = {f"R2_{a}_4dp_log": {"got": round(r2v[a], 4), "log": v} for a, v in REPRO["R2_4dp_log"].get(design, {}).items()}
    out["consistency"].update({f"POPX0_R2_{a}_4dp_log": {"got": round(r2v0[a], 4), "log": v}
                               for a, v in REPRO["POPX0_R2_4dp_log"].get(design, {}).items()})
    out["targets"] = ok
    out["passed"] = bool(all(ok.values()))
    return out


def surrogate_Y(Y, key: str = "ELAB|SMOKE_SURROGATE") -> np.ndarray:
    """Seeded noise in place of every V1 log D (NaN pattern kept): TECHNICAL SMOKE only."""
    rng = np.random.default_rng(paths.seed_for(key))
    Y = np.asarray(Y, float)
    return np.where(np.isfinite(Y), rng.normal(0.0, 1.5, Y.shape), np.nan)

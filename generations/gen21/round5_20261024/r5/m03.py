"""M03, the E-CT sibling-weighting factorial (CLAUDE_CODE_PROMPT.md section 6.2; conventions 6.0, 6.10, 8, 9): every
fixed kernel arm of the frozen grid, the nested leave-one-publication-out selection with a one-SE rule toward uniform,
the 2-nearest arms, the 0-read acid-trend arms, the selection-aware nulls, the strata, the report rules and the
section-8 leakage checks of this item.  Item 1.2 of the section-6.10 registry; outputs under results/ECT_M03/.

Built on r5.ect (the frozen E-CT predictor core: Target, read-corrected templates, B1, SH, SIB, NEAR, RAND2, ACOPY,
AINT, the pool guard) and r5.population (BIT sibling pools, the cond__ distance, change classes, strata).  Adapted
(copied, never run in place) from round5_design_20261023/ideas2/scripts/
    recipe_acidfit.py       LF SHA-256 417280ddb17ca25e93855e2ce2cb719c8ab082f1e22a9db2f65f294f1b7e832f  (MEAN, LIN,
                            LINCLIP, LINHALF on the acid-only siblings)
    recipe4_acid_skA.py     LF SHA-256 59c4873710b79bd9e6aa76b350ad079fae82fb74bf41c9722849049f57279778  (PAIRMEAN on
                            the bracketing levels, LINHALF_CLIP)
    recipe4_skepB_check.py  LF SHA-256 45771a8034f3b497415c61b33e73127d1a09321bfb2459840102f855349266ab  (POOLPAIR, the
                            leave-one-publication-out pooled pair-type slope; the acid-permutation control)
and the acid-kernel grid of IR M03's test design (ideas/critiques.json, M03: ell in {0.1, 0.25, 0.5, 1, inf}).

**Frozen arms** (4 reads, code k = 3, on the identical E-CT rows; n = |pool|, K = 1.78; r5.ect for T_j, B1, SH, SIB).
  Kernel factorial  {uniform, acid kernel, read-match kernel (offset removed), both} x count shrinkage, as one product
      grid (ACID = rows ell_read = inf, READ = column ell_acid = inf, uniform = both inf):
        w_j = exp(-|dlog10 acid_j| / ell_acid) * exp(-d_read_j / ell_read)       (ell = inf: factor 1)
        |dlog10 acid|: concentrations floored at 1e-4 M; one side missing 1, both missing 0 (r5.population's rule);
        d_read_j = RMS over the target's 4 DOPT_Q read metals of (y_S - Y_j,S) minus its mean (offset removed);
        raw  KW_..._raw = sum_j w_j T_j / sum_j w_j  (T_j the read-corrected template, r5.ect);
        sh   KW_..._sh  = w B1 + ... : w = n_eff / (n_eff + K), n_eff = (sum w_j)^2 / sum w_j^2 (Kish), blended with
             B1 on the same 4 reads.  The uniform limit is SIB_r4 (raw) and SH_r4 (sh) exactly (r5.ect values).
        ell_acid in LA_GRID = (inf, 1, 0.5, 0.25, 0.1) decades; ell_read in LR_GRID = (inf, 0.4, 0.2, 0.1, 0.05) log10 D.
  NEST_{ACID,READ,BOTH}_{sh,raw}  nested leave-one-publication-out choice inside the family (the uniform limit always
      in the grid): for each outer publication p, the loss of every grid point on the other publications (seed-mean
      SSE per publication, summed); one-SE rule toward uniform: keep every config whose paired loss excess over the best
      is <= sqrt(Q) sd_q of its per-publication excess (``paired``, primary; ``classic`` = the SE of the best config's
      summed loss, and ``argmin`` = no one-SE rule, are SENSITIVITY rules), then the least complex (number of finite
      ells, then the summed grid steps away from inf), then the lowest loss; exact ties are averaged, never broken by
      table order.  The chosen configs are reported per outer publication.
  NEAR2 / LEVEL2 / READ2 (_raw, _sh)  mean of the 2 siblings nearest in condition distance (r5.population.cond_dist) /
      in level at the 4 reads (RMS of y_S - Y_j,S) / in read pattern (d_read); ties at the 2nd distance (np.isclose,
      atol 1e-9) by the exact expectation over random tie-breaking; _sh blends with B1 at w = 2 / (2 + K).
  Context arms from r5.ect (identical values): B1_r4, SH_r4, SIB_r4, COPY, SH0, NEAR_r4, NEARSH_r4 and the matched-count
      random-2 control E_RAND2_{raw,sh}_r4 (exact expectation over every sibling pair; an expected squared error).
  0-read acid-trend arms on the acid-only BIT siblings (cond__ differs in acid concentration only; >= 2 at >= 2 levels):
      ACOPY_r0 = MEAN (r5.ect); AINT_r0 (r5.ect, piecewise-linear in log10 acid, flat beyond); PAIRMEAN_r0 = mean of
      the two bracketing acid levels (levels rounded to 4 decimals of log10 acid, level means; NaN when not bracketed);
      LIN_r0 = OLS line of log SF on log10 acid; LINCLIP_r0 = LIN at the target acid clamped to the siblings' range;
      LINHALF_r0 = MEAN + 0.5 (LIN - MEAN); LINHALFCLIP_r0 = MEAN + 0.5 (LINCLIP - MEAN); POOLPAIR_r0 = MEAN +
      b_pair,-pub (clamped acid - mean sibling acid), b pooled over the other publications' (target, pair) sibling sets
      of the population (sum sxy / sum sxx; 0 when sum sxx = 0).  The claim statistic is LINHALF - max(MEAN, PAIRMEAN)
      on the bracketed rows.
**Selection-aware nulls**: acid permuted within publication (finite acid values permuted among the publication's bench
cells, B >= 200, the full nested selection re-run each time); a permuted-read control (each target's read distances
shuffled among its siblings, fixed across seeds, re-selected each time); the matched-count random-2 control.

**Information contract (section 8).**  Allowed: the target's legal 4 reads, the siblings' full series, acid metadata,
the fold's beta; the hyperparameters of a publication's rows are chosen on the other publications only.  The pool guard
(r5.ect.guard_pool) refuses the target in its own pool and (except in the NB duplicate sensitivity) a bit-identical copy;
``intervention`` replaces every unread label of the target and requires bit-identical arms, and catches its positive
controls; ``fit_poisoning`` randomises the held-out publication's labels and requires identical selections and slopes.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import common as CM
from . import ect as E
from . import ect_report as REP
from . import paths
from r4 import metrics as M4  # noqa: E402

# Copy rule (brief 1.2): every copied or re-implemented source with its LF SHA-256 at the copy; checked by
# scripts/r5_seal.py (copied_sources reads this declaration).
COPIED_FROM = {
    "generations/gen21/round5_design_20261023/ideas2/scripts/recipe_acidfit.py":
        "417280ddb17ca25e93855e2ce2cb719c8ab082f1e22a9db2f65f294f1b7e832f",
    "generations/gen21/round5_design_20261023/ideas2/scripts/recipe4_acid_skA.py":
        "59c4873710b79bd9e6aa76b350ad079fae82fb74bf41c9722849049f57279778",
    "generations/gen21/round5_design_20261023/ideas2/scripts/recipe4_skepB_check.py":
        "45771a8034f3b497415c61b33e73127d1a09321bfb2459840102f855349266ab",
}

LN = E.LN
N_LN = E.N_LN
INF = math.inf
K_SH = E.K_SH
LA_GRID = (INF, 1.0, 0.5, 0.25, 0.1)          # acid kernel length, decades of [acid]; index 0 = uniform limit
LR_GRID = (INF, 0.4, 0.2, 0.1, 0.05)          # read-match kernel length, log10 D (offset removed)
CONFIGS = tuple((a, r) for a in range(len(LA_GRID)) for r in range(len(LR_GRID)))
UNIFORM = (0, 0)
FAMILIES = {"ACID": tuple(c for c in CONFIGS if c[1] == 0), "READ": tuple(c for c in CONFIGS if c[0] == 0),
            "BOTH": CONFIGS}
SHRINKS = ("sh", "raw")
SE_RULES = ("paired", "classic", "argmin")
PRIMARY_RULE = "paired"
ACID_FLOOR = 1e-4
TIE_ATOL = 1e-9
LOSS_TOL = 1e-12
B_PERM = 500
DENSE_MIN = 5
DEPLOY_GAIN, DEPLOY_PUBS = 0.01, 7
LEVEL_SHARE = 0.60
ORNL = E.PUB_ORNL
DGA = M4.DGA_CHEMOTYPE
SEEN, SENS, CC, DIAG, SMOKE = REP.SEEN, REP.SENS, REP.CC, REP.DIAG, REP.SMOKE
ORACLE = "ORACLE [O]"
NOT_NEXT_TO = REP.NOT_NEXT_TO
OUTDIR = "ECT_M03"
# population -> (build pool, label, description).  Nested selection is re-run inside every population.
POPULATIONS = {
    "PRIMARY": ("bit", SEEN, "BIT pool, n_sib >= 2, without C5-BTBP (training and scoring)"),
    "PRIMARY_NO_ORNL": ("bit", SENS, "PRIMARY without pub_19783c039b (the deploy-rule population)"),
    "WITH_C5": ("bit", SENS, "BIT pool, n_sib >= 2, with C5-BTBP in training and scoring"),
    "WITH_C5_NO_C5CELL": ("bit", SENS, "WITH_C5 without the single cell c950dc2982e96b83"),
    "PRIMARY_NOBIT": ("NB", SENS, "P1's CLEAN pool without the bit-identity rule, n_sib >= 2, without C5 (duplicates)"),
    "PRIMARY_OFS": ("OFS", SENS, "BIT pool minus offset / near-identity links (r5.ect OFS), n_sib >= 2 on it, without C5 "
                                 "(duplicate sensitivity, s0-items-2)"),
    "PRIMARY_NO_ORNL_OFS": ("OFS", SENS, "PRIMARY_OFS without pub_19783c039b (the deploy rule on the OFS pool)"),
}
POOLS = ("bit", "NB", "OFS")                  # build pools (s1_m03 builds every one)
DECISION_POP = "PRIMARY_NO_ORNL"
NULL_POPS = ("PRIMARY", "PRIMARY_NO_ORNL")
STRATA_POPS = ("PRIMARY", "WITH_C5")
ACID_POPS = ("PRIMARY", "PRIMARY_NO_ORNL", "WITH_C5", "PRIMARY_OFS")
LEVEL_POPS = ("PRIMARY", "PRIMARY_NO_ORNL", "WITH_C5", "PRIMARY_OFS")
STRATA = {"ACID_INTERP": "target acid inside the BIT siblings' acid range", "ACID_EXTRAP": "target acid outside it",
          "CONST_ACID": "every BIT sibling at the target's acid", "ACID_UNKNOWN": "target or sibling acid missing",
          "DENSE": "n_sib >= 5 (dense grids)", "SPARSE": "n_sib 2-4", "DILUENT": "change class diluent only",
          "ACIDCHG": "change class acid concentration only or acid type (+ concentration)",
          "OTHERCHG": "every other change class", "DGA": "chemotype sc009", "NONDGA": "other chemotypes"}
ECT_IDENT = ("B1_r4", "SH_r4", "SIB_r4", "COPY", "SH0", "NEAR_r4", "NEARSH_r4", "E_RAND2_raw_r4", "E_RAND2_sh_r4",
             "ACOPY_r0", "AINT_r0")
ECT_IDENT_NB = {"B1_r4": "B1_r4", "SH_r4": "NB_SH_r4", "SIB_r4": "NB_SIB_r4", "COPY": "NB_COPY", "SH0": "NB_SH0"}
ECT_IDENT_OFS = {"B1_r4": "B1_r4", "SH_r4": "OFS_SH_r4", "SIB_r4": "OFS_SIB_r4", "COPY": "OFS_COPY", "SH0": "OFS_SH0",
                 "NEAR_r4": "OFS_NEAR_r4", "NEARSH_r4": "OFS_NEARSH_r4", "ACOPY_r0": "OFS_ACOPY_r0",
                 "AINT_r0": "OFS_AINT_r0"}
ROW_ARMS = ("B1_r4", "SH_r4", "SIB_r4", "COPY", "SH0", "NEAR_r4", "NEARSH_r4", "NEAR2_raw", "NEAR2_sh", "LEVEL2_raw",
            "LEVEL2_sh", "READ2_raw", "READ2_sh", "E_RAND2_raw_r4", "E_RAND2_sh_r4", "ACOPY_r0", "AINT_r0",
            "PAIRMEAN_r0", "LIN_r0", "LINCLIP_r0", "LINHALF_r0", "LINHALFCLIP_r0", "sxy")
ACID_ARMS = ("ACOPY_r0", "AINT_r0", "LIN_r0", "LINCLIP_r0", "LINHALF_r0", "LINHALFCLIP_r0", "POOLPAIR_r0")


# ------------------------------------------------------------------------------------------------------------
# frozen grid
# ------------------------------------------------------------------------------------------------------------
def _lab(v: float) -> str:
    return "inf" if v == INF else f"{v:g}"


def cfg_label(c) -> str:
    return f"a{_lab(LA_GRID[c[0]])}_r{_lab(LR_GRID[c[1]])}"


def cfg_name(c, shrink: str) -> str:
    return f"KW_{cfg_label(c)}_{shrink}"


def nest_name(fam: str, shrink: str, rule: str = PRIMARY_RULE) -> str:
    return f"NEST_{fam}_{shrink}" + ("" if rule == PRIMARY_RULE else f"_{rule}")


def parse_nest(name: str):
    p = name.split("_")
    return p[1], p[2], (p[3] if len(p) > 3 else PRIMARY_RULE)


def rule_label(lab: str, *arms) -> str:
    """Section-9.1 label of an arm or a contrast: a nested arm under the classic or argmin one-SE rule is SENSITIVITY
    (ITEM_SPEC 5 and 10), so on PRIMARY it carries both labels; the other populations are SENSITIVITY already."""
    sens = any(str(a).startswith("NEST_") and parse_nest(str(a))[2] != PRIMARY_RULE for a in arms)
    return f"{lab}; {SENS} (one-SE rule)" if sens and lab != SENS else lab


def complexity(c) -> tuple:
    """One-SE order toward uniform: number of finite ells, then the summed grid steps away from inf."""
    return (int(c[0] > 0) + int(c[1] > 0), int(c[0]) + int(c[1]))


def config_table() -> pd.DataFrame:
    rows = []
    for c in CONFIGS:
        rows.append({"config": cfg_label(c), "ell_acid": _lab(LA_GRID[c[0]]), "ell_read": _lab(LR_GRID[c[1]]),
                     "uniform": c == UNIFORM, "in_ACID": c in FAMILIES["ACID"], "in_READ": c in FAMILIES["READ"],
                     "in_BOTH": True, "complexity": "%d,%d" % complexity(c)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------
# pure arm core (no cache, bench or parquet)
# ------------------------------------------------------------------------------------------------------------
def acid_log(v):
    v = np.asarray(v, float)
    nan = np.isnan(v)
    return np.where(nan, np.nan, np.log10(np.maximum(np.where(nan, 1.0, v), ACID_FLOOR)))


def acid_dist(a_t, a_s) -> np.ndarray:
    """|dlog10 acid| with concentrations floored at 1e-4 M; one side missing 1, both missing 0 (cond_dist's rule)."""
    lt, ls = acid_log(a_t), acid_log(a_s)
    nt, ns = np.isnan(lt), np.isnan(ls)
    d = np.abs(np.where(nt, 0.0, lt) - np.where(ns, 0.0, ls))
    return np.where(nt & ns, 0.0, np.where(nt | ns, 1.0, d))


def acid_position(a_t, a_s) -> str:
    """Target acid against its pool's finite acid values: interp / extrap / constant / unknown."""
    lt = float(acid_log(a_t))
    ls = acid_log(np.asarray(a_s, float))
    ls = ls[np.isfinite(ls)]
    if not np.isfinite(lt) or len(ls) == 0:
        return "unknown"
    lo, hi = float(ls.min()), float(ls.max())
    if hi - lo <= 1e-9 and abs(lt - lo) <= 1e-9:
        return "constant"
    if hi - lo > 1e-9 and lo - 1e-9 <= lt <= hi + 1e-9:
        return "interp"
    return "extrap"


def read_distances(y_at, Ys_at) -> tuple[np.ndarray, np.ndarray]:
    """(level distance, read-pattern distance): RMS of y_S - Y_j,S over the read metals, and the same with its mean
    (the offset) removed."""
    D = np.asarray(y_at, float)[None, :] - np.atleast_2d(np.asarray(Ys_at, float))
    dl = np.sqrt((D ** 2).mean(1))
    Dc = D - D.mean(1, keepdims=True)
    return dl, np.sqrt((Dc ** 2).mean(1))


def two_nearest(S, d) -> np.ndarray:
    """Mean of the 2 rows of S nearest in d; ties at the 2nd distance by the exact expectation over random
    tie-breaking (no table order).  NaN with fewer than 2 rows."""
    S = np.atleast_2d(np.asarray(S, float))
    d = np.asarray(d, float)
    if len(d) < 2:
        return np.full(S.shape[1], np.nan)
    d2 = np.sort(d)[1]
    tied = np.isclose(d, d2, atol=TIE_ATOL)
    inside = (d < d2) & ~tied
    k = 2 - int(inside.sum())
    return (S[inside].sum(0) + k * S[tied].mean(0)) / 2.0


def kernel_predict(S, mask, b1, da, dr, sib=None, sh=None, *, K: float = K_SH, configs=CONFIGS):
    """Every grid point for G groups at once.  S (G, N, P) templates (0-padded), mask (G, N), b1 (G, P), da / dr
    (G, N).  Returns raw, sh (G, C, P), n_eff (G, C) and the max deviation of the computed uniform limit from the exact
    (sib, sh) arrays, which then replace it (the uniform limit is r5.ect's SIB_r4 / SH_r4 exactly)."""
    S, b1 = np.asarray(S, float), np.asarray(b1, float)
    ia = np.array([0.0 if LA_GRID[a] == INF else 1.0 / LA_GRID[a] for a, _ in configs])
    ir = np.array([0.0 if LR_GRID[r] == INF else 1.0 / LR_GRID[r] for _, r in configs])
    lw = -(ia[None, :, None] * np.asarray(da, float)[:, None, :] + ir[None, :, None] * np.asarray(dr, float)[:, None, :])
    lw = np.where(np.asarray(mask, bool)[:, None, :], lw, -np.inf)
    lw = lw - lw.max(axis=2, keepdims=True)
    W = np.exp(lw)
    sw = W.sum(2)
    raw = np.einsum("gcn,gnp->gcp", W, S) / sw[:, :, None]
    neff = sw ** 2 / (W ** 2).sum(2)
    w = (neff / (neff + K))[:, :, None]
    shk = w * raw + (1.0 - w) * b1[:, None, :]
    dev = 0.0
    if sib is not None and UNIFORM in configs:
        u = configs.index(UNIFORM)
        dev = float(max(np.max(np.abs(raw[:, u] - sib)), np.max(np.abs(shk[:, u] - sh))))
        raw[:, u], shk[:, u] = sib, sh
    return raw, shk, neff, dev


def acid_trend(v, la, lt) -> dict:
    """0-read acid-trend arms from the acid-only siblings' pair SF v (n, P), their log10 acid la and the target's lt."""
    v = np.atleast_2d(np.asarray(v, float))
    la = np.asarray(la, float)
    P = v.shape[1]
    nanv = np.full(P, np.nan)
    out = {"n_acid": int(len(la)), "acid_ptp": float(np.ptp(la)) if len(la) else np.nan, "acid_inside": False,
           "PAIRMEAN_r0": nanv, "LIN_r0": nanv, "LINCLIP_r0": nanv, "LINHALF_r0": nanv, "LINHALFCLIP_r0": nanv,
           "sxy": np.zeros(P), "sxx": 0.0, "xpp": np.nan}
    if len(la) == 0 or not np.isfinite(lt):
        out["sxy"] = nanv
        return out
    mean = v.mean(0)
    out["acid_inside"] = bool(la.min() - 1e-9 <= lt <= la.max() + 1e-9)
    xr, ltr = np.round(la, 4), round(float(lt), 4)
    xu = np.unique(xr)
    lo, hi = xu[xu < ltr], xu[xu > ltr]
    if len(lo) and len(hi):
        out["PAIRMEAN_r0"] = 0.5 * (v[xr == lo.max()].mean(0) + v[xr == hi.min()].mean(0))
    if len(la) >= 2 and np.ptp(la) > 1e-9:
        m = float(la.mean())
        sxx = float(((la - m) ** 2).sum())
        sxy = ((la - m)[:, None] * (v - mean[None, :])).sum(0)
        b = sxy / sxx
        ltc = min(max(float(lt), float(la.min())), float(la.max()))
        out.update({"LIN_r0": mean + b * (lt - m), "LINCLIP_r0": mean + b * (ltc - m),
                    "LINHALF_r0": mean + 0.5 * b * (lt - m), "LINHALFCLIP_r0": mean + 0.5 * b * (ltc - m),
                    "sxy": sxy, "sxx": sxx, "xpp": ltc - m})
    return out


def poolpair(d: pd.DataFrame, sxy_col: str = "sxy") -> np.ndarray:
    """POOLPAIR_r0 = ACOPY_r0 + b_pair,-pub x_pp per row of ``d``; b pooled over the unique (target, pair) records of
    every other publication in ``d`` (sum sxy / sum sxx, 0 when sum sxx <= 0)."""
    u = d.drop_duplicates(["cell_id", "A", "B"])
    ok = np.isfinite(u[sxy_col].to_numpy(float)) & np.isfinite(u.sxx.to_numpy(float))
    u = u[ok]
    tab = pd.DataFrame({"pair": (u.A + "/" + u.B).to_numpy(), "pub": u.publication.astype(str).to_numpy(),
                        "sxy": u[sxy_col].to_numpy(float), "sxx": u.sxx.to_numpy(float)})
    pair = (d.A + "/" + d.B).to_numpy()
    pub = d.publication.astype(str).to_numpy()
    b = np.zeros(len(d))
    for pb in sorted(set(pub)):                       # sums over the other publications only (never total - own)
        o = tab[tab.pub != pb].groupby("pair")[["sxy", "sxx"]].sum()
        r = np.where(pub == pb)[0]
        for q in r:
            if pair[q] in o.index:
                num, den = float(o.loc[pair[q], "sxy"]), float(o.loc[pair[q], "sxx"])
                b[q] = num / den if den > 1e-12 else 0.0
    return d.ACOPY_r0.to_numpy(float) + b * d.xpp.to_numpy(float)


def rand2_se(S, b1, ys) -> tuple[np.ndarray, np.ndarray]:
    """r5.ect's matched-count random-2 control: exact expected squared error over every sibling pair, raw and shrunk."""
    n = len(S)
    if n < 2:
        nanv = np.full(len(b1), np.nan)
        return nanv, nanv
    pidx = np.array(list(itertools.combinations(range(n), 2)), int)
    dr = E.draw_means(S, pidx)
    return E.expected_se(dr, ys), E.expected_se(E.blend(dr, b1, E.weight(2)), ys)


# ------------------------------------------------------------------------------------------------------------
# nested leave-one-publication-out selection
# ------------------------------------------------------------------------------------------------------------
def sse_by_pub(err, sc, pc, n_seed: int, n_pub: int) -> np.ndarray:
    """(n_pub, C) seed-mean SSE per publication (mean over the seeds holding the publication's rows)."""
    code = np.asarray(sc) * n_pub + np.asarray(pc)
    M = np.zeros((n_seed * n_pub, err.shape[1]))
    np.add.at(M, code, err)
    cnt = np.bincount(code, minlength=n_seed * n_pub).reshape(n_seed, n_pub)
    ns = np.maximum((cnt > 0).sum(0), 1)
    return M.reshape(n_seed, n_pub, -1).sum(0) / ns[:, None]


def choose(sse, train, cfgs, rule: str) -> tuple[np.ndarray, dict]:
    """Chosen family indices for one outer fold from the training publications' per-publication losses."""
    X = np.asarray(sse, float)[np.asarray(train, bool)]
    Q = len(X)
    L = X.sum(0)
    Lmin = float(L.min())
    tol = LOSS_TOL * max(1.0, abs(Lmin))
    best = np.where(L - Lmin <= tol)[0]
    ref = X[:, best].mean(1)
    if rule == "argmin":
        se = np.zeros(len(L))
        cand = best
    else:
        if rule == "paired":
            se = np.sqrt(Q) * (X - ref[:, None]).std(axis=0, ddof=1) if Q >= 2 else np.zeros(len(L))
        elif rule == "classic":
            se = np.full(len(L), np.sqrt(Q) * float(ref.std(ddof=1)) if Q >= 2 else 0.0)
        else:
            raise ValueError(rule)
        cand = np.where(L - Lmin <= se + tol)[0]
        kap = [complexity(cfgs[c]) for c in cand]
        kmin = min(kap)
        cand = cand[[k == kmin for k in kap]]
        Lc = L[cand]
        cand = cand[Lc - Lc.min() <= tol]
    u = cfgs.index(UNIFORM)
    info = {"chosen": ";".join(cfg_label(cfgs[c]) for c in cand), "n_chosen": int(len(cand)),
            "uniform_chosen": bool(u in set(cand.tolist())), "ell_acid_chosen": ";".join(_lab(LA_GRID[cfgs[c][0]]) for c in cand),
            "ell_read_chosen": ";".join(_lab(LR_GRID[cfgs[c][1]]) for c in cand), "Q_train_pubs": int(Q),
            "L_min": Lmin, "L_uniform": float(L[u]), "SE_uniform": float(se[u]),
            "one_se_set": int((L - Lmin <= se + tol).sum()) if rule != "argmin" else int(len(best))}
    return cand, info


def nested_select(PM, y, sc, pc, cfgs, rule: str = PRIMARY_RULE, pubs=None) -> tuple[np.ndarray, list]:
    """Nested leave-one-publication-out predictions of one family (columns of PM aligned with cfgs); pc are codes
    0..P-1, sc seed codes 0..S-1.  Returns (predictions, one choice record per outer publication)."""
    PM, y = np.asarray(PM, float), np.asarray(y, float)
    sc, pc = np.asarray(sc, int), np.asarray(pc, int)
    nP = int(pc.max()) + 1
    sse = sse_by_pub((PM - y[:, None]) ** 2, sc, pc, int(sc.max()) + 1, nP)
    pred = np.full(len(y), np.nan)
    recs = []
    for q in range(nP):
        ch, info = choose(sse, np.arange(nP) != q, list(cfgs), rule)
        r = pc == q
        pred[r] = PM[r][:, ch].mean(1)
        recs.append({"outer_publication": pubs[q] if pubs is not None else q, **info})
    return pred, recs


def frame_arrays(d: pd.DataFrame) -> tuple:
    sc = pd.factorize(d.split_seed, sort=True)[0]
    pc, pubs = pd.factorize(d.publication.astype(str), sort=True)
    return d.y.to_numpy(float), sc, pc, list(pubs)


def nested_predict(d: pd.DataFrame, fam: str, shrink: str, rule: str = PRIMARY_RULE) -> tuple[np.ndarray, list]:
    y, sc, pc, pubs = frame_arrays(d)
    cfgs = FAMILIES[fam]
    PM = d[[cfg_name(c, shrink) for c in cfgs]].to_numpy(float)
    return nested_select(PM, y, sc, pc, cfgs, rule, pubs)


def dr2_fast(y, cand, ref, sc) -> float:
    """Seed mean of R2(cand) - R2(ref), R2 pooled within seed (REP.r2 on arrays)."""
    y, sc = np.asarray(y, float), np.asarray(sc, int)
    n = np.bincount(sc)
    mu = np.bincount(sc, y) / n
    sst = np.bincount(sc, (y - mu[sc]) ** 2)
    return float(((np.bincount(sc, (y - ref) ** 2) - np.bincount(sc, (y - cand) ** 2)) / sst).mean())


# ------------------------------------------------------------------------------------------------------------
# kernel data (padded per group) and the in-corpus builder
# ------------------------------------------------------------------------------------------------------------
@dataclass
class KernelData:
    S: np.ndarray        # (G, N, P) read-corrected templates at 4 reads (0-padded)
    mask: np.ndarray     # (G, N)
    b1: np.ndarray       # (G, P)
    y: np.ndarray        # (G, P) evaluator labels (or the smoke surrogate)
    sib: np.ndarray      # (G, P) SIB_r4 (exact)
    sh: np.ndarray       # (G, P) SH_r4 (exact)
    da: np.ndarray       # (G, N)
    dr: np.ndarray       # (G, N)
    ti: np.ndarray       # (G,) bench index of the target
    sj: np.ndarray       # (G, N) bench index of each sibling (0 padding; see mask)
    rowpos: np.ndarray   # (G, P) row in the frame (-1 padding)
    cell: np.ndarray     # (G,)
    seed: np.ndarray     # (G,)


def pack(groups: list) -> KernelData:
    G = len(groups)
    N = max(len(g["da"]) for g in groups)
    P = max(len(g["b1"]) for g in groups)
    S, mask, sj = np.zeros((G, N, P)), np.zeros((G, N), bool), np.zeros((G, N), int)
    b1, y, sib, sh = (np.zeros((G, P)) for _ in range(4))
    rowpos = np.full((G, P), -1, int)
    da, dr = np.zeros((G, N)), np.zeros((G, N))
    for k, g in enumerate(groups):
        n, p = len(g["da"]), len(g["b1"])
        S[k, :n, :p], mask[k, :n], sj[k, :n] = g["S"], True, g["js"]
        b1[k, :p], y[k, :p], sib[k, :p], sh[k, :p] = g["b1"], g["y"], g["SIB_r4"], g["SH_r4"]
        da[k, :n], dr[k, :n], rowpos[k, :p] = g["da"], g["dr"], g["rowpos"]
    return KernelData(S, mask, b1, y, sib, sh, da, dr, np.array([g["ti"] for g in groups], int), sj, rowpos,
                      np.array([g["cell"] for g in groups], dtype=object), np.array([g["seed"] for g in groups], int))


def to_rows(KD: KernelData, arr: np.ndarray, nrows: int) -> np.ndarray:
    """(G, C, P) -> (nrows, C) through rowpos."""
    out = np.full((nrows, arr.shape[1]), np.nan)
    ok = KD.rowpos >= 0
    out[KD.rowpos[ok]] = np.transpose(arr, (0, 2, 1))[ok]
    return out


def add_config_columns(R: pd.DataFrame, KD: KernelData) -> float:
    """Every grid point (raw and sh) as a row column; returns the uniform-limit deviation (construction check)."""
    raw, shk, _, dev = kernel_predict(KD.S, KD.mask, KD.b1, KD.da, KD.dr, KD.sib, KD.sh)
    for shrink, arr in (("raw", raw), ("sh", shk)):
        M = to_rows(KD, arr, len(R))
        for k, c in enumerate(CONFIGS):
            R[cfg_name(c, shrink)] = M[:, k]
    return dev


def pools_of(corpus, t: str, pool: str) -> tuple[list, list, list]:
    """(kernel pool, acid-only pool, near siblings) of target t: BIT pools, P1's CLEAN pool without the bit rule, or
    the OFS duplicate sensitivity (BIT pool minus offset / near-identity links; acid pool minus those links; near
    siblings by the nearest-condition rule re-applied to the OFS pool, r5.ect.Corpus._near_of)."""
    V, P = corpus.var[t], corpus.pools[t]
    if pool == "bit":
        return list(V["bit"]), list(V["acid"]), list(V["near"])
    if pool == "NB":
        return list(V["NB"]), [], list(P["near_clean"])
    if pool == "OFS":
        return list(V["OFS"]), list(V["OFS_acid"]), list(V["OFS_near"])
    raise ValueError(pool)


def default_targets(corpus, pool: str) -> list:
    """Targets a pool's build covers: WITH_C5 (BIT), PRIMARY_NOBIT (NB), PRIMARY_OFS (OFS: n_sib >= 2 on the OFS
    pool, without C5, not V2-dropped; r5.ect target_table in_PRIMARY_OFS)."""
    if pool == "bit":
        return list(corpus.strata["WITH_C5"])
    if pool == "NB":
        return list(corpus.strata["PRIMARY_NOBIT"])
    if pool == "OFS":
        return [t for t in corpus.c3 if len(corpus.var[t]["OFS"]) >= 2 and not corpus.pools[t]["c5"]
                and not corpus.pools[t]["t_v2drop"]]
    raise ValueError(pool)


def guard_target(corpus, t: str, ids, acid_ids, pool: str) -> None:
    i = corpus.pos[t]
    for nm, pl, allow in (("M03_" + pool, ids, pool == "NB"), ("M03_acid", acid_ids, False)):
        Yp = corpus.Y[[corpus.pos[s] for s in pl]] if pl else np.zeros((0, N_LN))
        E.guard_pool(t, corpus.Y[i], pl, Yp, allow_bit=allow, name=nm)


def make_target(ctx: dict, cell: dict, seed: int, fold: int, y_t, g: pd.DataFrame | None = None) -> E.Target:
    """r5.ect's in-corpus target (the same construction as Corpus.target_block, reads 2-4)."""
    t = str(cell["cell_id"])
    g = ctx["rows"].loc[ctx["groups"][(seed, t)]] if g is None else g
    if not (g.fold == fold).all():
        raise AssertionError(f"{t}: parquet fold differs from the cache fold")
    pairs = tuple(zip(g.ia.astype(int), g.ib.astype(int)))
    seq = cell["seqs"]["DOPT_Q"]
    rsets = {r: frozenset(m for p in seq[:r - 1] for m in p) for r in E.READS}
    return E.make_target(t, cell["metals"], pairs, y_t, rsets, ctx["betas"][(seed, fold)],
                         zero=g["G14@ZERO"].to_numpy(float))


def group_inputs(corpus, T: E.Target, i: int, ids, acid_ids, near_ids, Yv, acid_raw, *, peek: bool = False) -> dict:
    """Every M03 prediction input and point arm of one (seed, target) group from the value matrix ``Yv`` (the target
    enters only through T, i.e. its legal reads).  ``peek`` (positive control only) computes the read distances from
    the target's full series in Yv, which the intervention must catch."""
    pos = corpus.pos
    js = np.array([pos[s] for s in ids], int)
    Ys = Yv[js]
    # C-ordered like the row-indexed pool matrices of Corpus.target_block: numpy's mean over an F-ordered axis 0
    # sums in a different order (1 ulp at n >= 8), so this keeps SIB / SH / COPY bit-identical to item 1.1's columns
    S = np.ascontiguousarray(E.template_sf(T, 4, Ys))
    b1 = E.b1_sf(T, 4)
    n = len(ids)
    sib = S.mean(0)
    sh = E.blend(sib, b1, E.weight(n))
    rd = E.read_metals(T.reads[4])
    if peek:
        mets = list(T.metals)
        yt = np.asarray(Yv[i], float)[mets]
    else:
        mets, yt = rd, T.reads[4][rd]
    dl, dr = read_distances(yt, Ys[:, mets])
    da = acid_dist(acid_raw[i], acid_raw[js])
    dc = np.array([corpus.cm.cond_dist(i, int(j)) for j in js])
    copy = np.ascontiguousarray(E.template_sf(T, 0, Ys)).mean(0)
    out = {"S": S, "b1": b1, "B1_r4": b1, "SIB_r4": sib, "SH_r4": sh, "da": da, "dr": dr, "dl": dl, "dc": dc, "js": js,
           "COPY": copy, "SH0": E.blend(copy, T.zero, E.weight(n))}
    w2 = E.weight(2)
    for nm, dd in (("NEAR2", dc), ("LEVEL2", dl), ("READ2", dr)):
        raw = two_nearest(S, dd)
        out[nm + "_raw"], out[nm + "_sh"] = raw, E.blend(raw, b1, w2)
    qi = {s: q for q, s in enumerate(ids)}
    nn = [qi[s] for s in near_ids]
    out["NEAR_r4"] = S[nn].mean(0) if nn else np.full(len(b1), np.nan)
    out["NEARSH_r4"] = E.blend(out["NEAR_r4"], b1, E.weight(len(nn)))
    raw, shk, _, _ = kernel_predict(S[None], np.ones((1, n), bool), b1[None], da[None], dr[None], sib[None], sh[None])
    out["K_raw"], out["K_sh"] = raw[0], shk[0]
    P = len(T.pairs)
    if acid_ids:
        ja = [pos[s] for s in acid_ids]
        la, lt = corpus.acid[ja], corpus.acid[i]
        va = np.ascontiguousarray(E.template_sf(T, 0, Yv[ja]))
        out["ACOPY_r0"] = va.mean(0)
        out["AINT_r0"] = E.pair_sf(E.interp_template(Yv[ja], la, lt), T.pairs)
        out.update(acid_trend(va, la, lt))
    else:
        out["ACOPY_r0"] = out["AINT_r0"] = np.full(P, np.nan)
        out.update(acid_trend(np.zeros((0, P)), np.zeros(0), np.nan))
    out["acid_pos_bit"] = acid_position(acid_raw[i], acid_raw[js])
    return out


def _same(a, b) -> bool:
    """Bit-identity of two group outputs (strings compared as strings; NaN equal to NaN)."""
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return bool(np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True))


def _maxdiff(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.isnan(a), np.isnan(b)
    if (na != nb).any():
        return float("inf")
    return float(np.max(np.abs(a[~na] - b[~nb]))) if (~na).any() else 0.0


def build(corpus, design: str, *, pool: str = "bit", targets=None, y_mode: str = "labels", check_seeds=None,
          log=print) -> tuple[pd.DataFrame, KernelData, dict]:
    """Every M03 arm on the round-4 REF rows of the population's targets (default: WITH_C5 for the BIT pool,
    PRIMARY_NOBIT for the NB pool).  ``y_mode='surrogate'`` (technical smoke) scores the expectation arms against
    seeded noise and puts that noise in y.  ``check_seeds``: dev seeds on which every shared arm is compared with
    r5.ect.Corpus.target_block (bit-identity expected).  Returns (rows, kernel data, checks)."""
    from g21 import cache as GC
    t0 = time.time()
    ctx = corpus.design_ctx(design)
    Rall = ctx["rows"]
    if targets is None:
        targets = default_targets(corpus, pool)
    tset = set(str(t) for t in targets)
    check_seeds = set(paths.DEV_SEEDS if check_seeds is None else check_seeds)
    acid_raw = corpus.cm.F[:, corpus.cm.ia].astype(float)
    rng = np.random.default_rng(paths.seed_for(f"M03|SMOKE_YS|{design}|{pool}"))
    checks = {"design": design, "pool": pool, "groups": 0, "rows": 0, "b1_maxabs": 0.0, "y_maxabs": 0.0,
              "readset_mismatch": 0, "both_ends_read_rows": 0, "ect_identity_groups": 0,
              "ect_identity_maxabs": {}, "y_mode": y_mode}
    groups, parts, idx = [], {}, []
    guarded = set()
    nrow = 0
    for seed in paths.DEV_SEEDS:
        for fold in range(5):
            rec = GC.load(design, seed, fold)
            for cell in sorted(rec["outer"]["cells"], key=lambda c: str(c["cell_id"])):
                t = str(cell["cell_id"])
                if t not in tset or (seed, t) not in ctx["groups"]:
                    continue
                gi = ctx["groups"][(seed, t)]
                g = Rall.loc[gi]
                i = corpus.pos[t]
                T = make_target(ctx, cell, seed, fold, corpus.Y[i], g)
                ids, acid_ids, near_ids = pools_of(corpus, t, pool)
                if len(ids) < 2:
                    raise AssertionError(f"{t}: M03 needs n_sib >= 2 (found {len(ids)})")
                if t not in guarded:
                    guard_target(corpus, t, ids, acid_ids, pool)
                    guarded.add(t)
                y_lab = corpus.Y[i, g.ia.astype(int)] - corpus.Y[i, g.ib.astype(int)]
                checks["y_maxabs"] = max(checks["y_maxabs"], float(np.max(np.abs(y_lab - g.y.to_numpy(float)))))
                ys = y_lab if y_mode == "labels" else rng.normal(0.0, 0.2, len(gi))
                gin = group_inputs(corpus, T, i, ids, acid_ids, near_ids, corpus.Y, acid_raw)
                gin["E_RAND2_raw_r4"], gin["E_RAND2_sh_r4"] = rand2_se(gin["S"], gin["b1"], ys)
                checks["b1_maxabs"] = max(checks["b1_maxabs"], float(np.max(np.abs(gin["b1"] - g.B1.to_numpy(float)))))
                seq = cell["seqs"]["DOPT_Q"]
                rs = ",".join(LN[m] for m in sorted({m for p in seq[:3] for m in p}))
                checks["readset_mismatch"] += int(rs != str(g.read_set.iloc[0]))
                checks["both_ends_read_rows"] += int((g.a_read & g.b_read).sum())
                if seed in check_seeds:
                    blk = corpus.target_block(ctx, cell, seed, fold, y_score=ys)
                    keys = {k: k for k in ECT_IDENT} if pool == "bit" else (ECT_IDENT_NB if pool == "NB"
                                                                             else ECT_IDENT_OFS)
                    for mine, theirs in keys.items():
                        dv = _maxdiff(gin[mine], blk[theirs])
                        checks["ect_identity_maxabs"][mine] = max(checks["ect_identity_maxabs"].get(mine, 0.0), dv)
                    checks["ect_identity_groups"] += 1
                Pn = len(gi)
                for k in ROW_ARMS:
                    parts.setdefault(k, []).append(np.asarray(gin[k], float))
                for k in ("n_acid", "acid_ptp", "sxx", "xpp"):
                    parts.setdefault(k, []).append(np.full(Pn, gin[k], float))
                parts.setdefault("acid_inside", []).append(np.full(Pn, bool(gin["acid_inside"])))
                parts.setdefault("acid_pos_bit", []).append(np.array([gin["acid_pos_bit"]] * Pn, dtype=object))
                parts.setdefault("n_sib", []).append(np.full(Pn, len(ids), int))
                groups.append({"S": gin["S"], "b1": gin["b1"], "y": ys, "SIB_r4": gin["SIB_r4"], "SH_r4": gin["SH_r4"],
                               "da": gin["da"], "dr": gin["dr"], "js": gin["js"], "ti": i,
                               "rowpos": np.arange(nrow, nrow + Pn), "cell": t, "seed": seed})
                idx.append(gi)
                nrow += Pn
    R = Rall.loc[np.concatenate(idx)].reset_index(drop=True).copy()
    for k, v in parts.items():
        R[k] = np.concatenate(v)
    if y_mode != "labels":
        R["y"] = np.concatenate([g["y"] for g in groups])
    R = R.rename(columns={"G14@ZERO": "G14Z"})
    S = corpus.strata
    for k in ("PRIMARY", "WITH_C5", "PRIMARY_NOBIT"):
        R["in_" + k] = R.cell_id.isin(set(S[k]))
    R["in_PRIMARY_OFS"] = R.cell_id.isin(set(default_targets(corpus, "OFS")))
    cls_key = "cls_clean" if pool == "NB" else "cls"
    R["cls"] = [corpus.pools[c][cls_key] for c in R.cell_id]
    R["c5"] = R.publication.astype(str) == paths.C5
    R["in_ACIDTREND"] = (R.n_acid >= 2) & (R.acid_ptp > 1e-9)
    add_strata(R)
    R["pool"] = pool
    R["design"] = design
    KD = pack(groups)
    checks.update({"groups": len(groups), "rows": int(len(R)), "targets": int(R.cell_id.nunique()),
                   "max_pool": int(KD.mask.sum(1).max()), "max_pairs": int((KD.rowpos >= 0).sum(1).max()),
                   "seconds": round(time.time() - t0, 1)})
    log(f"  {design} {pool}: {checks['groups']} groups, {len(R)} rows, {checks['targets']} targets, ect identity on "
        f"{checks['ect_identity_groups']} groups ({checks['seconds']}s)")
    return R, KD, checks


def add_strata(R: pd.DataFrame) -> None:
    """Pre-declared section-6.2 strata (metadata only)."""
    R["st_ACID_INTERP"] = R.acid_pos_bit == "interp"
    R["st_ACID_EXTRAP"] = R.acid_pos_bit == "extrap"
    R["st_CONST_ACID"] = R.acid_pos_bit == "constant"
    R["st_ACID_UNKNOWN"] = R.acid_pos_bit == "unknown"
    R["st_DENSE"] = R.n_sib >= DENSE_MIN
    R["st_SPARSE"] = R.n_sib < DENSE_MIN
    R["st_DILUENT"] = R.cls == "diluent_only"
    R["st_ACIDCHG"] = R.cls.isin(["acid_conc_only", "acid_type(+conc)"])
    R["st_OTHERCHG"] = ~(R.st_DILUENT | R.st_ACIDCHG)
    R["st_DGA"] = R.chemotype.astype(str) == DGA
    R["st_NONDGA"] = ~R.st_DGA


def pop_mask(R: pd.DataFrame, name: str) -> np.ndarray:
    pub, cell = R.publication.astype(str), R.cell_id.astype(str)
    m = {"PRIMARY": R.in_PRIMARY, "PRIMARY_NO_ORNL": R.in_PRIMARY & (pub != ORNL), "WITH_C5": R.in_WITH_C5,
         "WITH_C5_NO_C5CELL": R.in_WITH_C5 & (cell != paths.C5_CELL), "PRIMARY_NOBIT": R.in_PRIMARY_NOBIT,
         "PRIMARY_OFS": R.in_PRIMARY_OFS, "PRIMARY_NO_ORNL_OFS": R.in_PRIMARY_OFS & (pub != ORNL)}[name]
    return m.to_numpy(bool)


# ------------------------------------------------------------------------------------------------------------
# technical smoke surrogate
# ------------------------------------------------------------------------------------------------------------
META_FLOATS = {"dZ", "k", "k_max", "n_metals", "draw", "ia", "ib", "fold", "split_seed", "n_acid", "acid_ptp", "sxx",
               "xpp", "n_sib"}


def surrogate(R: pd.DataFrame, KD: KernelData, design: str, pool: str) -> tuple[pd.DataFrame, KernelData]:
    """TECHNICAL SMOKE: y, the templates, B1 and every prediction / label-derived column replaced by seeded noise (NaN
    pattern kept); SIB / SH recomputed with r5.ect's formulas, so the code runs on the real structure and no real error
    is ever computed."""
    rng = np.random.default_rng(paths.seed_for(f"M03|SMOKE_SURROGATE|{design}|{pool}"))
    pm = KD.rowpos >= 0
    y = np.where(pm, rng.normal(0.0, 0.2, KD.y.shape), 0.0)
    S = np.where(KD.mask[:, :, None] & pm[:, None, :], y[:, None, :] + rng.normal(0.0, 0.15, KD.S.shape), 0.0)
    b1 = np.where(pm, y + rng.normal(0.0, 0.15, y.shape), 0.0)
    sib, sh = np.zeros_like(b1), np.zeros_like(b1)
    for g in range(len(S)):
        n, p = int(KD.mask[g].sum()), int(pm[g].sum())
        sib[g, :p] = S[g, :n, :p].mean(0)
        sh[g, :p] = E.blend(sib[g, :p], b1[g, :p], E.weight(n))
    KD2 = replace(KD, S=S, b1=b1, y=y, sib=sib, sh=sh)
    R2 = R.copy()
    for c in R2.columns:
        if c in META_FLOATS or c.startswith(("st_", "in_")) or R2[c].dtype.kind != "f":
            continue
        v = R2[c].to_numpy(float)
        noise = np.abs(rng.normal(0.0, 0.05, len(R2))) if c.startswith("E_") else rng.normal(0.0, 0.2, len(R2))
        R2[c] = np.where(np.isfinite(v), noise, np.nan)
    n = len(R2)
    for col, arr in (("y", y), ("B1_r4", b1), ("SIB_r4", sib), ("SH_r4", sh)):
        R2[col] = to_rows(KD2, arr[:, None, :], n)[:, 0]
    return R2, KD2


# ------------------------------------------------------------------------------------------------------------
# analysis
# ------------------------------------------------------------------------------------------------------------
NEST_COLS = tuple(nest_name(f, s, r) for s in SHRINKS for f in FAMILIES for r in SE_RULES)
MAIN_ARMS = ["B1_r4", "SH_r4", "SIB_r4", "COPY", "SH0", "NEAR_r4", "NEARSH_r4", "NEAR2_raw", "NEAR2_sh", "LEVEL2_raw",
             "LEVEL2_sh", "READ2_raw", "READ2_sh", "E_RAND2_raw_r4", "E_RAND2_sh_r4"]
CONTRASTS = [("NEST_BOTH_sh", "SH_r4"), ("NEST_ACID_sh", "SH_r4"), ("NEST_READ_sh", "SH_r4"),
             ("NEST_BOTH_raw", "SIB_r4"), ("NEST_ACID_raw", "SIB_r4"), ("NEST_READ_raw", "SIB_r4"),
             ("NEST_BOTH_sh_classic", "SH_r4"), ("NEST_BOTH_sh_argmin", "SH_r4"),
             ("NEST_ACID_sh_classic", "SH_r4"), ("NEST_READ_sh_classic", "SH_r4"),
             ("SH_r4", "B1_r4"), ("NEST_BOTH_sh", "B1_r4"),
             ("NEAR2_raw", "SIB_r4"), ("LEVEL2_raw", "SIB_r4"), ("READ2_raw", "SIB_r4"), ("LEVEL2_raw", "NEAR2_raw"),
             ("NEAR2_raw", "E_RAND2_raw_r4"), ("LEVEL2_raw", "E_RAND2_raw_r4"), ("READ2_raw", "E_RAND2_raw_r4"),
             ("NEAR2_sh", "SH_r4"), ("LEVEL2_sh", "SH_r4"), ("READ2_sh", "SH_r4"), ("NEAR2_sh", "E_RAND2_sh_r4"),
             ("LEVEL2_sh", "E_RAND2_sh_r4")]
STRATA_CONTRASTS = [("NEST_BOTH_sh", "SH_r4"), ("NEST_ACID_sh", "SH_r4"), ("NEST_READ_sh", "SH_r4"),
                    ("LEVEL2_raw", "SIB_r4"), ("NEAR2_raw", "SIB_r4"), ("SH_r4", "B1_r4")]
STRATA_ARMS = ["B1_r4", "SH_r4", "SIB_r4", "NEST_BOTH_sh", "NEST_ACID_sh", "NEST_READ_sh", "NEAR2_raw", "LEVEL2_raw",
               "E_RAND2_raw_r4"]
ACID_CONTRASTS = {"bracketed": [("LINHALF_r0", "ACOPY_r0"), ("LINHALF_r0", "PAIRMEAN_r0"), ("PAIRMEAN_r0", "ACOPY_r0"),
                                ("AINT_r0", "ACOPY_r0"), ("AINT_r0", "PAIRMEAN_r0"), ("LINHALFCLIP_r0", "ACOPY_r0"),
                                ("POOLPAIR_r0", "ACOPY_r0"), ("SH_r4", "ACOPY_r0")],
                  "all": [("LINHALF_r0", "ACOPY_r0"), ("LINCLIP_r0", "ACOPY_r0"), ("LINHALFCLIP_r0", "ACOPY_r0"),
                          ("AINT_r0", "ACOPY_r0"), ("POOLPAIR_r0", "ACOPY_r0"), ("LIN_r0", "ACOPY_r0")],
                  "exterior": [("LIN_r0", "ACOPY_r0"), ("LINCLIP_r0", "ACOPY_r0"), ("LINHALF_r0", "ACOPY_r0"),
                               ("POOLPAIR_r0", "ACOPY_r0"), ("AINT_r0", "ACOPY_r0")]}


def nest_all(d: pd.DataFrame, pop: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every nested arm (family x shrink x one-SE rule) inside population ``pop``; the choice records."""
    y, sc, pc, pubs = frame_arrays(d)
    recs = []
    for shrink in SHRINKS:
        for fam, cfgs in FAMILIES.items():
            PM = d[[cfg_name(c, shrink) for c in cfgs]].to_numpy(float)
            for rule in SE_RULES:
                pred, ch = nested_select(PM, y, sc, pc, cfgs, rule, pubs)
                d[nest_name(fam, shrink, rule)] = pred
                recs += [{"population": pop, "family": fam, "shrink": shrink, "rule": rule, **r,
                          "label": DIAG if rule == PRIMARY_RULE else f"{DIAG}; {SENS} (one-SE rule)"} for r in ch]
    return d, pd.DataFrame(recs)


def renested_lopo(d: pd.DataFrame, cand: str, ref: str) -> dict:
    """Leave-one-publication-out of a nested contrast with the whole nested selection re-run without the left-out
    publication (training and scoring)."""
    fam, shrink, rule = parse_nest(cand)

    def fn(x):
        pred, _ = nested_predict(x, fam, shrink, rule)
        xx = REP.complete(x.assign(_NEST=pred), ["_NEST", ref])
        return REP.r2(xx, "_NEST") - REP.r2(xx, ref)
    return CM.lopo(d, fn, by="publication")


def contrasts(d: pd.DataFrame, pop: str, lab: str, cons) -> tuple[list, list]:
    rows, units = [], []
    for cand, ref in cons:
        if cand not in d.columns or ref not in d.columns:
            continue
        r, u = REP.infer(d, cand, ref, stratum_name=pop, label=rule_label(lab, cand, ref), key=f"M03|{pop}|{cand}-{ref}")
        if cand.startswith("NEST_") and "note" not in r:
            lo = renested_lopo(REP.complete(d, [cand, ref]), cand, ref)
            r.update({f"lopo_renested_{k}": lo[k] for k in ("point", "se", "lo", "hi", "min", "min_by", "max", "max_by")})
        rows.append(r)
        if len(u):
            units.append(u)
    return rows, units


def grid_table(d: pd.DataFrame, pop: str, lab: str) -> pd.DataFrame:
    rows = []
    for shrink in SHRINKS:
        ref = "SH_r4" if shrink == "sh" else "SIB_r4"
        r_ref = REP.r2_seeds(d, ref)
        for c in CONFIGS:
            col = cfg_name(c, shrink)
            rs = REP.r2_seeds(d, col)
            u = REP.unit_contrib(d, col, ref, unit=("publication",))
            rows.append({"population": pop, "shrink": shrink, "config": cfg_label(c), "arm": col,
                         "ell_acid": _lab(LA_GRID[c[0]]), "ell_read": _lab(LR_GRID[c[1]]), "uniform_limit": c == UNIFORM,
                         "R2": float(rs.mean()), "R2_per_seed": ";".join(f"{v:.4f}" for v in rs),
                         "uniform_ref": ref, "dR2_vs_uniform": float((rs - r_ref).mean()), "MSE": REP.mse(d, col),
                         "pubs_improve": int((u.dSSE > 0).sum()), "pubs_worse": int((u.dSSE < 0).sum()), "P": int(len(u)),
                         "label": lab})
        g = pd.DataFrame([r for r in rows if r["shrink"] == shrink and r["population"] == pop])
        b = g.loc[g.R2.idxmax()]
        rows.append({**b.to_dict(), "config": f"best fixed grid point in hindsight ({b.config})",
                     "label": f"{ORACLE}: headroom only, never a score, baseline or selection rule"})
    return pd.DataFrame(rows)


def acid_tables(d: pd.DataFrame, pop: str, lab: str) -> tuple[pd.DataFrame, list, list, dict]:
    """The 0-read acid-trend arms on the acid-trend rows (>= 2 acid-only siblings at >= 2 levels): arm tables per
    subset, contrasts, and the claim statistic LINHALF - max(MEAN, PAIRMEAN) on the bracketed rows."""
    a = d[d.in_ACIDTREND].copy()
    if a.empty:
        return pd.DataFrame(), [], [], {"population": pop, "note": "no acid-trend rows"}
    a["POOLPAIR_r0"] = poolpair(a)
    br = np.isfinite(a.PAIRMEAN_r0.to_numpy(float))
    subs = {"all": a, "bracketed": a[br], "exterior": a[~a.acid_inside.to_numpy(bool)],
            "edge (inside, not bracketed)": a[a.acid_inside.to_numpy(bool) & ~br]}
    tabs, inf, units = [], [], []
    base = list(ACID_ARMS) + ["COPY", "SH0", "SH_r4", "B1_r4"]
    for sn, s in subs.items():
        if s.empty:
            continue
        arms = base + (["PAIRMEAN_r0"] if sn == "bracketed" else [])
        tabs.append(REP.arm_table(s, arms, ref="B1_r4", stratum_name=f"{pop}|acid-trend {sn}", label=lab))
        if sn in ACID_CONTRASTS and s.publication.nunique() >= 2:
            r, u = contrasts(s, f"{pop}|acid-trend {sn}", lab, ACID_CONTRASTS[sn])
            inf += r
            units += u
    s = REP.complete(subs["bracketed"], ["LINHALF_r0", "ACOPY_r0", "PAIRMEAN_r0"])
    claim = {"population": pop, "rows": "acid-trend bracketed", **REP.describe(s)}
    if len(s) and s.publication.nunique() >= 2:
        def fn(x):
            return REP.r2(x, "LINHALF_r0") - max(REP.r2(x, "ACOPY_r0"), REP.r2(x, "PAIRMEAN_r0"))
        lo = CM.lopo(s, fn, by="publication")
        claim.update({"R2_LINHALF": REP.r2(s, "LINHALF_r0"), "R2_MEAN": REP.r2(s, "ACOPY_r0"),
                      "R2_PAIRMEAN": REP.r2(s, "PAIRMEAN_r0"), "claim_stat": lo["point"], "lopo_lo": lo["lo"],
                      "lopo_hi": lo["hi"], "lopo_min": lo["min"], "lopo_min_without": lo["min_by"], "lopo_max": lo["max"],
                      "lopo_max_without": lo["max_by"], "P": lo["P"]})
    return pd.concat(tabs, ignore_index=True), inf, units, claim


def level_rule(d: pd.DataFrame, pop: str, lab: str) -> pd.DataFrame:
    """Per publication with a target at n_sib >= 5 (rows of those targets): does LEVEL2 beat both the plain mean and
    NEAR2?  Raw forms (primary) and shrunk forms (sensitivity)."""
    dd = d[d.n_sib >= DENSE_MIN]
    rows = []
    for variant, (lv, mn, nr) in {"raw": ("LEVEL2_raw", "SIB_r4", "NEAR2_raw"),
                                  "sh": ("LEVEL2_sh", "SH_r4", "NEAR2_sh")}.items():
        x = REP.complete(dd, [lv, mn, nr])
        if x.empty:
            continue
        um = REP.unit_contrib(x, lv, mn, unit=("publication",)).set_index("publication")
        un = REP.unit_contrib(x, lv, nr, unit=("publication",)).set_index("publication")
        for p in um.index:
            rows.append({"population": pop, "variant": variant, "publication": p, "rows_per_seed": um.at[p, "rows"],
                         f"SSE_{lv}": um.at[p, "sse_c"], f"SSE_{mn}": um.at[p, "sse_r"], f"SSE_{nr}": un.at[p, "sse_r"],
                         "beats_mean": bool(um.at[p, "dSSE"] > 0), "beats_near2": bool(un.at[p, "dSSE"] > 0),
                         "beats_both": bool(um.at[p, "dSSE"] > 0 and un.at[p, "dSSE"] > 0),
                         "label": lab if variant == "raw" else SENS})
    return pd.DataFrame(rows)


def strata_tables(d: pd.DataFrame, pop: str, lab: str) -> tuple[pd.DataFrame, list, list]:
    """Descriptive slices of the population's nested predictions (selection is not re-run inside a stratum)."""
    tabs, inf, units = [], [], []
    for st in STRATA:
        s = d[d["st_" + st].to_numpy(bool)]
        if s.empty:
            tabs.append(pd.DataFrame([{"stratum": f"{pop}|{st}", "arm": "", "note": "no rows", "label": lab}]))
            continue
        tabs.append(REP.arm_table(s, STRATA_ARMS, ref="B1_r4", stratum_name=f"{pop}|{st}", label=lab,
                                  extra={"stratum_definition": STRATA[st]}))
        for cand, ref in STRATA_CONTRASTS:
            r, u = REP.infer(s, cand, ref, stratum_name=f"{pop}|{st}", label=lab, key=f"M03|{pop}|{st}|{cand}-{ref}")
            inf.append(r)
            if len(u):
                units.append(u)
    return pd.concat(tabs, ignore_index=True), inf, units


def null_distribution(R: pd.DataFrame, KD: KernelData, corpus, pop: str, kind: str, B: int,
                      fams=None, log=print) -> tuple[pd.DataFrame, dict]:
    """Selection-aware null: ``kind='ACID'`` permutes acid within publication, ``'READ'`` shuffles each target's read
    distances among its siblings; the nested selection (primary one-SE rule, sh families) is re-run for every
    replicate.  Returns (replicate draws, observed values)."""
    fams = fams or (("ACID", "BOTH") if kind == "ACID" else ("READ", "BOTH"))
    m = pop_mask(R, pop)
    rows = np.where(m)[0]
    local = np.full(len(R), -1, int)
    local[rows] = np.arange(len(rows))
    gsel = np.where(m[KD.rowpos[:, 0]])[0]
    sub = replace(KD, S=KD.S[gsel], mask=KD.mask[gsel], b1=KD.b1[gsel], y=KD.y[gsel], sib=KD.sib[gsel],
                  sh=KD.sh[gsel], da=KD.da[gsel], dr=KD.dr[gsel], ti=KD.ti[gsel], sj=KD.sj[gsel],
                  rowpos=np.where(KD.rowpos[gsel] >= 0, local[np.maximum(KD.rowpos[gsel], 0)], -1),
                  cell=KD.cell[gsel], seed=KD.seed[gsel])
    d = R.iloc[rows]
    y, sc, pc, pubs = frame_arrays(d)
    ref = d.SH_r4.to_numpy(float)
    cols = {f: [CONFIGS.index(c) for c in FAMILIES[f]] for f in fams}

    def stat(da, dr):
        _, shk, _, _ = kernel_predict(sub.S, sub.mask, sub.b1, da, dr, sub.sib, sub.sh)
        PM = to_rows(sub, shk, len(rows))
        out = {}
        for f in fams:
            pred, ch = nested_select(PM[:, cols[f]], y, sc, pc, FAMILIES[f], PRIMARY_RULE)
            out[f] = (dr2_fast(y, pred, ref, sc), int(sum(c["uniform_chosen"] for c in ch)))
        return out
    obs = stat(sub.da, sub.dr)
    rng = np.random.default_rng(paths.seed_for(f"M03|{kind}PERM|{pop}"))
    acid_raw = corpus.cm.F[:, corpus.cm.ia].astype(float)
    pub_idx = {}
    for p in sorted(set(d.publication.astype(str))):
        j = np.where((corpus.pub == p) & np.isfinite(acid_raw))[0]
        pub_idx[p] = j
    cells = sorted(set(sub.cell.tolist()))
    gcell = {c: np.where(sub.cell == c)[0] for c in cells}
    nsib = sub.mask.sum(1)
    draws = []
    t0 = time.time()
    for b in range(B):
        if kind == "ACID":
            a = acid_raw.copy()
            for p, j in pub_idx.items():
                a[j] = acid_raw[j[rng.permutation(len(j))]]
            da = np.where(sub.mask, acid_dist(a[sub.ti][:, None], a[sub.sj]), 0.0)
            st = stat(da, sub.dr)
        elif kind == "READ":
            dr = sub.dr.copy()
            for c in cells:
                gs = gcell[c]
                n = int(nsib[gs[0]])
                perm = rng.permutation(n)
                dr[gs, :n] = sub.dr[gs][:, perm]
            st = stat(sub.da, dr)
        else:
            raise ValueError(kind)
        for f, (v, nu) in st.items():
            draws.append({"population": pop, "null": f"{kind}_PERM", "replicate": b, "family": f, "dR2_vs_SH": v,
                          "uniform_folds": nu})
    secs = time.time() - t0
    log(f"    null {kind} {pop}: {B} replicates x {len(fams)} families, {secs:.1f}s")
    return pd.DataFrame(draws), {"obs": obs, "seconds": secs, "folds": int(pc.max()) + 1}


def null_summary(draws: pd.DataFrame, obs: dict, pop: str, kind: str, B: int, folds: int) -> list:
    out = []
    for f, (v, nu) in obs.items():
        x = draws[draws.family == f].dR2_vs_SH.to_numpy(float) if len(draws) else np.zeros(0)
        nuf = draws[draws.family == f].uniform_folds.to_numpy(float) if len(draws) else np.zeros(0)
        ge = int((x >= v - 1e-12).sum())
        out.append({"population": pop, "null": f"{kind}_PERM", "arm": nest_name(f, "sh"), "obs_dR2_vs_SH": v,
                    "obs_uniform_folds": nu, "folds": folds, "B": int(len(x)),
                    "null_mean": float(x.mean()) if len(x) else np.nan, "null_sd": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                    "null_q95": float(np.quantile(x, 0.95)) if len(x) else np.nan, "null_max": float(x.max()) if len(x) else np.nan,
                    "n_null_ge_obs": ge, "p_perm": (1.0 + ge) / (1.0 + len(x)) if len(x) else np.nan,
                    "null_uniform_folds_mean": float(nuf.mean()) if len(nuf) else np.nan,
                    "label": f"{DIAG} (selection-aware null; no in-corpus p-value is called significant)"})
    return out


def deploy_rule_name(fam: str) -> str:
    return f"M03 deploy rule ({fam} kernel family)"


def deploy_rule(d: pd.DataFrame, ch: pd.DataFrame, pop: str, fam: str, lab: str) -> dict:
    cand = nest_name(fam, "sh")
    x = REP.complete(d, [cand, "SH_r4"])
    gain = REP.r2(x, cand) - REP.r2(x, "SH_r4")
    u = REP.unit_contrib(x, cand, "SH_r4", unit=("publication",))
    c = ch[(ch.population == pop) & (ch.family == fam) & (ch.shrink == "sh") & (ch.rule == PRIMARY_RULE)]
    n_imp, P, nu, folds = int((u.dSSE > 0).sum()), int(len(u)), int(c.uniform_chosen.sum()), int(len(c))
    c1, c2, c3 = gain < DEPLOY_GAIN, n_imp < DEPLOY_PUBS, nu > folds / 2.0
    dep = c1 or c2 or c3
    return {"rule": deploy_rule_name(fam), "population": pop, "statistic":
            f"dR2 {cand} - SH_r4; publications improving; folds choosing ell = inf (uniform)", "dR2": gain,
            "threshold_dR2": DEPLOY_GAIN, "pubs_improve": n_imp, "P": P, "threshold_pubs": DEPLOY_PUBS,
            "uniform_folds": nu, "folds": folds, "fires_gain_lt_0.01": bool(c1), "fires_fewer_than_7_pubs": bool(c2),
            "fires_uniform_most_folds": bool(c3),
            "outcome": ("deploy the uniform shrunk mean (SH)" if dep else
                        "nested weighting passes the in-corpus consistency check (EXPLORATORY; no claim)"),
            "label": lab}


def analysis(builds: dict, corpus, design: str, *, B: int = B_PERM, log=print) -> dict:
    """Every table of the item.  ``builds``: {'bit': (R, KD), 'NB': (R, KD)} with the config columns added."""
    T = {k: [] for k in ("populations", "comparators", "grid", "choices", "inference", "units", "strata", "acid",
                         "acid_claim", "level_rule", "nulls", "null_draws")}
    frames, rules, nullobs = {}, [], {}
    for pop, (pool, lab, desc) in POPULATIONS.items():
        if pool not in builds:
            continue
        R, _ = builds[pool]
        d = R[pop_mask(R, pop)].copy()
        if d.empty or d.publication.nunique() < 2:
            T["populations"].append(pd.DataFrame([{"population": pop, "status": "NOT RUN (fewer than 2 publications)"}]))
            continue
        d, ch = nest_all(d, pop)
        frames[pop] = d
        T["choices"].append(ch)
        T["populations"].append(pd.DataFrame([{"population": pop, "pool": pool, "definition": desc, **REP.describe(d),
                                               "c5_cells": int(d[d.c5].cell_id.nunique()),
                                               "acid_trend_cells": int(d[d.in_ACIDTREND].cell_id.nunique()),
                                               "dense_cells": int(d[d.st_DENSE].cell_id.nunique()), "label": lab}]))
        ct = REP.arm_table(d, MAIN_ARMS + list(NEST_COLS), ref="B1_r4", stratum_name=pop, label=lab)
        ct["label"] = [rule_label(lab, a) for a in ct.arm]
        T["comparators"].append(ct)
        T["grid"].append(grid_table(d, pop, lab))
        inf, un = contrasts(d, pop, lab, CONTRASTS)
        T["inference"] += inf
        T["units"] += un
        if pop in STRATA_POPS:
            st, inf, un = strata_tables(d, pop, lab)
            T["strata"].append(st)
            T["inference"] += inf
            T["units"] += un
        if pool in ("bit", "OFS") and pop in ACID_POPS:
            at, inf, un, claim = acid_tables(d, pop, lab)
            if len(at):
                T["acid"].append(at)
            T["inference"] += inf
            T["units"] += un
            # no threshold, so not a CONSISTENCY CHECK (brief 9.1): EXPLORATORY (seen), as DECISION_LOG 1.2-R04 seals it
            T["acid_claim"].append(pd.DataFrame([{**claim, "pool": pool, "label": SEEN if pop == "PRIMARY" else SENS}]))
        if pool in ("bit", "OFS") and pop in LEVEL_POPS:
            T["level_rule"].append(level_rule(d, pop, lab))
        log(f"  {design} {pop}: {d.cell_id.nunique()} cells, {d.publication.nunique()} publications, {len(d)} rows")
    if "bit" in builds and B > 0:
        R, KD = builds["bit"]
        for pop in NULL_POPS:
            if pop not in frames:
                continue
            for kind in ("ACID", "READ"):
                dr_, info = null_distribution(R, KD, corpus, pop, kind, B, log=log)
                T["null_draws"].append(dr_)
                T["nulls"].append(pd.DataFrame(null_summary(dr_, info["obs"], pop, kind, B, info["folds"])))
                nullobs[(pop, kind)] = info
    ch = pd.concat(T["choices"], ignore_index=True) if T["choices"] else pd.DataFrame()
    for pop, fams, lab in ((DECISION_POP, ("BOTH",), CC), (DECISION_POP, ("ACID", "READ"), SENS),
                           ("PRIMARY", ("BOTH", "ACID", "READ"), SENS), ("WITH_C5", ("BOTH",), SENS),
                           ("PRIMARY_NOBIT", ("BOTH",), SENS), ("PRIMARY_NO_ORNL_OFS", ("BOTH", "ACID", "READ"), SENS),
                           ("PRIMARY_OFS", ("BOTH",), SENS)):
        if pop in frames:
            for f in fams:
                rules.append(deploy_rule(frames[pop], ch, pop, f, lab))
    lr = pd.concat(T["level_rule"], ignore_index=True) if T["level_rule"] else pd.DataFrame()
    for pop in LEVEL_POPS:
        for variant in ("raw", "sh"):
            x = lr[(lr.population == pop) & (lr.variant == variant)] if len(lr) else lr
            if not len(x):
                continue
            share = float(x.beats_both.mean())
            rules.append({"rule": f"M03 level rule (LEVEL2_{variant} vs plain mean and NEAR2_{variant})", "population": pop,
                          "statistic": "share of publications with a target at n_sib >= 5 where LEVEL2 beats both",
                          "share": share, "pubs_beats_both": int(x.beats_both.sum()), "P": int(len(x)),
                          "threshold_share": LEVEL_SHARE,
                          "outcome": "level rule kept (in-corpus consistency check)" if share >= LEVEL_SHARE else "level rule retired",
                          "label": CC if (pop == "PRIMARY" and variant == "raw") else SENS})
    cl = pd.concat(T["acid_claim"], ignore_index=True) if T["acid_claim"] else pd.DataFrame()
    for _, c in cl.iterrows():
        if "claim_stat" in c and pd.notna(c.get("claim_stat")):
            rules.append({"rule": "acid-trend claim statistic LINHALF - max(MEAN, PAIRMEAN) (bracketed rows)",
                          "population": c.population, "statistic": "pooled R2 difference with a LOPO interval",
                          "claim_stat": c.claim_stat, "lopo_lo": c.lopo_lo, "lopo_hi": c.lopo_hi,
                          "outcome": ("positive, LOPO t(P-1) lower bound above 0 (descriptive; anti-conservative)"
                                      if c.lopo_lo > 0 else
                                      ("positive, LOPO t(P-1) interval includes 0 (descriptive)" if c.claim_stat > 0
                                       else "not positive")) +
                          " (reported; in-corpus no claim)", "label": c.label})
    T["rules"] = pd.DataFrame(rules)
    out = {k: (pd.concat(v, ignore_index=True) if isinstance(v, list) and v and isinstance(v[0], pd.DataFrame)
               else (pd.DataFrame(v) if isinstance(v, list) else v)) for k, v in T.items()}
    for k, v in out.items():
        if isinstance(v, pd.DataFrame) and len(v):
            v.insert(0, "design", design)
    out["_frames"] = frames
    out["_nullobs"] = nullobs
    return out


def nested_long(frames: dict) -> pd.DataFrame:
    """Row-level nested predictions of every population (long format)."""
    out = []
    for pop, d in frames.items():
        key = d[["design", "split_seed", "fold", "cell_id", "A", "B", "y"]].copy()
        for c in NEST_COLS:
            x = key.copy()
            x["population"], x["arm"], x["pred"] = pop, c, d[c].to_numpy(float)
            out.append(x)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------------------------------------------------------------
# leakage (section 8 items 2 and 4)
# ------------------------------------------------------------------------------------------------------------
def intervention(corpus, design: str, *, pool: str = "bit", targets=None, seeds=None, log=print) -> dict:
    """Every unread metal of the target (outside its 4-read DOPT_Q set) replaced by noise: every M03 input and arm
    must be bit-identical.  Positive controls (must be caught on every group): the target in its own pool (the guard
    raises), a bit-identical copy of it (BIT pool: the guard raises), the target in its own pool with the guard off
    (the kernel arms change), and the read distances computed from the full target series (they change)."""
    from g21 import cache as GC
    ctx = corpus.design_ctx(design)
    targets = set(default_targets(corpus, pool) if targets is None else targets)
    seeds = tuple(paths.DEV_SEEDS if seeds is None else seeds)
    acid_raw = corpus.cm.F[:, corpus.cm.ia].astype(float)
    rng = np.random.default_rng(paths.seed_for(f"M03|INTERVENTION|{design}|{pool}"))
    rep = {"design": design, "pool": pool, "seeds": list(seeds), "groups": 0, "diff_groups": 0, "cols_diff": [],
           "pc_self_guard": 0, "pc_copy_guard": 0, "pc_self_kernel": 0, "pc_peek_read": 0, "pc_groups": 0}
    skip = {"js"}
    for seed in seeds:
        for fold in range(5):
            rec = GC.load(design, seed, fold)
            for cell in sorted(rec["outer"]["cells"], key=lambda c: str(c["cell_id"])):
                t = str(cell["cell_id"])
                if t not in targets or (seed, t) not in ctx["groups"]:
                    continue
                i = corpus.pos[t]
                ids, acid_ids, near_ids = pools_of(corpus, t, pool)
                T = make_target(ctx, cell, seed, fold, corpus.Y[i])
                base = group_inputs(corpus, T, i, ids, acid_ids, near_ids, corpus.Y, acid_raw)
                r4 = set(E.read_metals(T.reads[4]))
                Yp = corpus.Y.copy()
                un = [m for m in cell["metals"] if m not in r4]
                Yp[i, un] = rng.normal(0.0, 3.0, len(un))
                Tp = make_target(ctx, cell, seed, fold, Yp[i])
                pois = group_inputs(corpus, Tp, i, ids, acid_ids, near_ids, Yp, acid_raw)
                bad = [k for k in base if k not in skip and not _same(base[k], pois[k])]
                rep["groups"] += 1
                if bad:
                    rep["diff_groups"] += 1
                    rep["cols_diff"] = sorted(set(rep["cols_diff"]) | set(bad))
                rep["pc_groups"] += 1
                try:
                    E.guard_pool(t, corpus.Y[i], ids + [t], corpus.Y[[corpus.pos[s] for s in ids] + [i]],
                                 allow_bit=pool == "NB", name="M03 pool + target")
                except E.LeakageError:
                    rep["pc_self_guard"] += 1
                if pool in ("bit", "OFS"):
                    try:
                        E.guard_pool(t, corpus.Y[i], ["copy_of_" + t], corpus.Y[i][None, :], name="M03 bit")
                    except E.LeakageError:
                        rep["pc_copy_guard"] += 1
                pc_ids = ids + [t]
                a1 = group_inputs(corpus, T, i, pc_ids, [], near_ids, corpus.Y, acid_raw)
                a2 = group_inputs(corpus, Tp, i, pc_ids, [], near_ids, Yp, acid_raw)
                rep["pc_self_kernel"] += int(not np.array_equal(a1["K_sh"], a2["K_sh"]))
                p1 = group_inputs(corpus, T, i, ids, [], near_ids, corpus.Y, acid_raw, peek=True)
                p2 = group_inputs(corpus, Tp, i, ids, [], near_ids, Yp, acid_raw, peek=True)
                rep["pc_peek_read"] += int(not np.array_equal(p1["dr"], p2["dr"]))
    n = rep["pc_groups"]
    rep["checks_ok"] = bool(rep["diff_groups"] == 0 and rep["pc_self_guard"] == n and rep["pc_self_kernel"] == n
                            and rep["pc_peek_read"] == n and (pool == "NB" or rep["pc_copy_guard"] == n))
    rep["coverage_full"] = bool(list(seeds) == list(paths.DEV_SEEDS) and n > 0)       # s0-mechanics-5
    rep["passed"] = bool(rep["checks_ok"] and rep["coverage_full"])
    log(f"  intervention {design} {pool}: {rep['groups']} groups, {rep['diff_groups']} with differences; positive "
        f"controls {rep['pc_self_guard']}/{rep['pc_copy_guard']}/{rep['pc_self_kernel']}/{rep['pc_peek_read']} of {n} "
        f"-> passed {rep['passed']}")
    return rep


def fit_poisoning(frames: dict, design: str, *, log=print) -> dict:
    """Section 8 item 2 for M03's fitted quantities: for every population, family, shrink, one-SE rule and outer
    publication p, p's labels (y) and every prediction of p's rows (p's series) are randomised; the selection for
    outer p must be bit-identical (chosen configs, L_min, SE).  The same for the POOLPAIR slope of p's rows (p's sxy
    randomised)."""
    rng = np.random.default_rng(paths.seed_for(f"M03|FITPOISON|{design}"))
    rep = {"design": design, "fold_fits": 0, "identical": 0, "poolpair_pubs": 0, "poolpair_identical": 0,
           "pc_runs": 0, "pc_caught": 0, "pc_by_family": {}, "poolpair_pc_runs": 0, "poolpair_pc_caught": 0}

    def leaky_poolpair(dd):
        """Positive control: the pooled slope over every publication, the outer one included (a leak)."""
        u = dd.drop_duplicates(["cell_id", "A", "B"])
        ok = np.isfinite(u.sxy.to_numpy(float)) & np.isfinite(u.sxx.to_numpy(float))
        u = u[ok]
        o = pd.DataFrame({"pair": (u.A + "/" + u.B).to_numpy(), "sxy": u.sxy.to_numpy(float),
                          "sxx": u.sxx.to_numpy(float)}).groupby("pair")[["sxy", "sxx"]].sum()
        pair = (dd.A + "/" + dd.B).to_numpy()
        b = np.array([float(o.loc[q, "sxy"]) / float(o.loc[q, "sxx"]) if q in o.index and float(o.loc[q, "sxx"]) > 1e-12
                      else 0.0 for q in pair])
        return dd.ACOPY_r0.to_numpy(float) + b * dd.xpp.to_numpy(float)
    for pop, d in frames.items():
        y, sc, pc, pubs = frame_arrays(d)
        nP = int(pc.max()) + 1
        allm = np.ones(nP, bool)
        for shrink in SHRINKS:
            for fam, cfgs in FAMILIES.items():
                PM = d[[cfg_name(c, shrink) for c in cfgs]].to_numpy(float)
                sse = sse_by_pub((PM - y[:, None]) ** 2, sc, pc, int(sc.max()) + 1, nP)
                fk = rep["pc_by_family"].setdefault(fam, {"runs": 0, "caught": 0})
                for rule in SE_RULES:
                    for q in range(nP):
                        tr = np.arange(nP) != q
                        c0, i0 = choose(sse, tr, list(cfgs), rule)
                        y2, PM2 = y.copy(), PM.copy()
                        r = pc == q
                        y2[r] = rng.normal(0.0, 1.0, int(r.sum()))
                        PM2[r] = rng.normal(0.0, 1.0, (int(r.sum()), PM.shape[1]))
                        sse2 = sse_by_pub((PM2 - y2[:, None]) ** 2, sc, pc, int(sc.max()) + 1, nP)
                        c1, i1 = choose(sse2, tr, list(cfgs), rule)
                        rep["fold_fits"] += 1
                        rep["identical"] += int(np.array_equal(c0, c1) and i0["L_min"] == i1["L_min"]
                                                and i0["SE_uniform"] == i1["SE_uniform"])
                        # leaky positive control (s0-mechanics-16): the selection with q's own losses included
                        cL0, iL0 = choose(sse, allm, list(cfgs), rule)
                        cL1, iL1 = choose(sse2, allm, list(cfgs), rule)
                        caught = int(not (np.array_equal(cL0, cL1) and iL0["L_min"] == iL1["L_min"]))
                        rep["pc_runs"] += 1
                        rep["pc_caught"] += caught
                        fk["runs"] += 1
                        fk["caught"] += caught
        a = d[d.in_ACIDTREND] if "in_ACIDTREND" in d else d.iloc[0:0]
        if len(a) and a.publication.nunique() >= 2:
            base = poolpair(a)
            leak0 = leaky_poolpair(a)
            for p in sorted(a.publication.astype(str).unique()):
                a2 = a.copy()
                r = (a2.publication.astype(str) == p).to_numpy()
                a2.loc[r, "sxy"] = rng.normal(0.0, 1.0, int(r.sum()))
                rep["poolpair_pubs"] += 1
                rep["poolpair_identical"] += int(np.array_equal(base[r], poolpair(a2)[r], equal_nan=True))
                rep["poolpair_pc_runs"] += 1
                rep["poolpair_pc_caught"] += int(not np.array_equal(leak0[r], leaky_poolpair(a2)[r], equal_nan=True))
    fam_ok = bool(rep["pc_by_family"]) and all(v["caught"] > 0 for v in rep["pc_by_family"].values())
    rep["passed"] = bool(rep["identical"] == rep["fold_fits"] and rep["poolpair_identical"] == rep["poolpair_pubs"]
                         and fam_ok and (rep["poolpair_pc_runs"] == 0 or rep["poolpair_pc_caught"] > 0))
    log(f"  fit poisoning {design}: {rep['identical']}/{rep['fold_fits']} selections and {rep['poolpair_identical']}/"
        f"{rep['poolpair_pubs']} POOLPAIR slopes identical; leaky positive control caught {rep['pc_caught']}/"
        f"{rep['pc_runs']} ({ {k: v['caught'] for k, v in rep['pc_by_family'].items()} }), POOLPAIR "
        f"{rep['poolpair_pc_caught']}/{rep['poolpair_pc_runs']} -> passed {rep['passed']}")
    return rep


# ------------------------------------------------------------------------------------------------------------
# reproduction targets (section 6.0; TECHNICAL SMOKE; a mismatch stops the item)
# ------------------------------------------------------------------------------------------------------------
REPRO = {"P1_PRIMARY_NOBIT": {"counts": (99, 12, 14, 690.0),
                              "values": {"B1_r4": 0.5309, "KW_ainf_rinf_sh (kernel code, uniform limit = SH)": 0.7082,
                                         "KW_ainf_rinf_raw (kernel code, uniform limit = SIB)": 0.6987}}}
REPRO_TOL = 5e-5 + 1e-12
B1_TOL = 8.9e-16                 # brief 6.0: B1 from the fold betas within 8.9e-16 (seen 8.88e-16)


def reproduction(builds: dict, checks: dict, design: str) -> dict:
    """B1 from the fold betas against the parquet (<= 8.9e-16), y exact, read sets identical, every shared arm
    bit-identical to r5.ect, the kernel's uniform limit within 1e-12 of SH / SIB, and P1's PRIMARY without the bit rule
    (B1 0.5309, SH 0.7082, SIB 0.6987, BPG) recomputed through this module's kernel code on the NB pool."""
    out = {"label": SMOKE, "design": design, "checks": checks, "targets": {}}
    ok = True
    for pool, c in checks.items():
        ok &= c["b1_maxabs"] <= B1_TOL and c["y_maxabs"] == 0.0 and c["readset_mismatch"] == 0
        ok &= all(v == 0.0 for v in c["ect_identity_maxabs"].values()) and c["ect_identity_groups"] > 0
        ok &= c.get("uniform_dev", 0.0) <= 1e-12
    R, KD = builds["NB"]
    raw, shk, _, dev = kernel_predict(KD.S, KD.mask, KD.b1, KD.da, KD.dr)       # no override: the kernel code itself
    u = CONFIGS.index(UNIFORM)
    x = R.copy()
    x["_KSH"] = to_rows(KD, shk, len(R))[:, u]
    x["_KRAW"] = to_rows(KD, raw, len(R))[:, u]
    d = x[x.in_PRIMARY_NOBIT.to_numpy(bool)]
    info = REP.describe(d)
    got_c = (info["cells"], info["publications"], info["units"], info["rows_per_seed"])
    spec = REPRO["P1_PRIMARY_NOBIT"]
    c_ok = got_c == spec["counts"]
    vals = dict(zip(spec["values"], (REP.r2(d, "B1_r4"), REP.r2(d, "_KSH"), REP.r2(d, "_KRAW"))))
    rec = {"counts_expected": list(spec["counts"]), "counts_found": list(got_c), "counts_ok": bool(c_ok),
           "values_found": vals, "kernel_uniform_vs_ect_maxabs": float(max(np.max(np.abs(d._KSH - d.SH_r4)),
                                                                         np.max(np.abs(d._KRAW - d.SIB_r4))))}
    ok &= c_ok and rec["kernel_uniform_vs_ect_maxabs"] <= 1e-12
    if design == "BPG":
        v_ok = {a: bool(abs(vals[a] - v) <= REPRO_TOL) for a, v in spec["values"].items()}
        rec.update({"values_expected": spec["values"], "values_ok": v_ok})
        ok &= all(v_ok.values())
    else:
        rec["note"] = "no BP value target registered (BPG decides); counts and identities must match"
    out["targets"]["P1_PRIMARY_NOBIT"] = rec
    out["passed"] = bool(ok)
    return out


# ------------------------------------------------------------------------------------------------------------
# the section-9.3 per-item table
# ------------------------------------------------------------------------------------------------------------
UNIT_TESTS = ("G_units", "SIGN_W/L/T", "SIGN_p", "p_signflip_unit", "signflip_unit_exact",
              "p_signflip_chemotype_blocks", "G_chemotypes", "IMt_rho_p", "IMt_log_p", "eqw_pub_relSSE", "label")
DUP_POPS = ("PRIMARY", "PRIMARY_NOBIT", "PRIMARY_OFS", "PRIMARY_NO_ORNL_OFS")      # ITEM_SPEC 13.10
DUP_COLS = ("dR2", "lopo_renested_lo", "lopo_renested_hi", "lopo_renested_min", "lopo_renested_min_by",
            "lopo_renested_max", "lopo_renested_max_by", "rows_per_seed", "label")
KERNEL_WORDS = {"ACID": "acid-distance", "READ": "read-match"}


def _row(df, **eq) -> dict:
    """The first row of ``df`` with every column equal to the given value ({} when there is none)."""
    if not isinstance(df, pd.DataFrame) or not len(df) or any(k not in df.columns for k in eq):
        return {}
    m = np.ones(len(df), bool)
    for k, v in eq.items():
        m &= (df[k] == v).to_numpy(bool)
    return df[m].iloc[0].to_dict() if m.any() else {}


def _num(v, fmt: str = "+.4f") -> str:
    return format(float(v), fmt) if v is not None and pd.notna(v) else "n.a."


def decision_sentences(T: dict, design: str) -> tuple[str, str]:
    """Section 9.3's two sentences from the computed deploy-rule row (BOTH family, DECISION_POP) and what bounds it:
    the ACID / READ families alone, the selection-aware nulls of NEST_BOTH_sh, the classic one-SE sensitivity and the
    re-nested leave-one-publication-out minimum."""
    rules, inf = T["rules"], T["inference"]
    seal = "C-ECT stays sealed on SH@4 vs B1@4 (CONFIRMATION_PACKAGE; new laboratories, section 7.5, data-gated)"
    dec = _row(rules, rule=deploy_rule_name("BOTH"), population=DECISION_POP)
    if not dec:
        return f"The deploy rule was not computed on {DECISION_POP} (rules_{design}.csv).", seal + "."
    passes = not str(dec["outcome"]).startswith("deploy")
    fam = {f: _row(rules, rule=deploy_rule_name(f), population=DECISION_POP) for f in KERNEL_WORDS}
    alone = [f for f, r in fam.items() if r and not str(r["outcome"]).startswith("deploy")]
    if len(alone) == 1:
        carrier = f"the {KERNEL_WORDS[alone[0]]} kernel " + ("carries it" if passes else "alone passes")
    else:
        carrier = "both kernel families pass alone" if alone else "neither kernel family passes alone"
    split = "; ".join(f"{f} family alone: {'passes' if f in alone else 'deploy SH'}, dR2 {_num(r.get('dR2'))}"
                      for f, r in fam.items() if r)
    nl = T.get("nulls")
    p_acid = _row(nl, population=DECISION_POP, null="ACID_PERM", arm=nest_name("BOTH", "sh")).get("p_perm")
    p_read = _row(nl, population=DECISION_POP, null="READ_PERM", arm=nest_name("BOTH", "sh")).get("p_perm")
    ch = T.get("choices")
    cl = (ch[(ch.population == DECISION_POP) & (ch.family == "BOTH") & (ch.shrink == "sh") & (ch.rule == "classic")]
          if isinstance(ch, pd.DataFrame) and len(ch) else pd.DataFrame())
    cl_dr2 = _row(inf, stratum=DECISION_POP, cand=nest_name("BOTH", "sh", "classic"), ref="SH_r4").get("dR2")
    cl_nu = int(cl.uniform_chosen.sum()) if len(cl) else "n.a."
    rn = _row(inf, stratum=DECISION_POP, cand=nest_name("BOTH", "sh"), ref="SH_r4")
    expected = "against the expected 'uniform shrunk mean'" if passes else "as expected"
    changed = (f"On {DECISION_POP} (the deploy-rule population, {design}) the registered deploy rule gives "
               f"'{dec['outcome']}' (dR2 NEST_BOTH_sh - SH_r4 {_num(dec['dR2'])}, {int(dec['pubs_improve'])} of "
               f"{int(dec['P'])} publications improve, uniform chosen in {int(dec['uniform_folds'])} of "
               f"{int(dec['folds'])} folds), {expected}; {carrier} ({split}; acid-permutation null p "
               f"{_num(p_acid, '.3f')}, read-permutation null p {_num(p_read, '.3f')}), under the classic one-SE rule "
               f"({SENS}) uniform is chosen in {cl_nu} of {len(cl)} folds (dR2 {_num(cl_dr2)}), and the re-nested "
               f"leave-one-publication-out minimum is {_num(rn.get('lopo_renested_min'))} without "
               f"{rn.get('lopo_renested_min_by', 'n.a.')} (every rule outcome: rules_{design}.csv).")
    if passes:
        word = KERNEL_WORDS[alone[0]] if len(alone) == 1 else "kernel"
        nxt = (f"{seal}; nested {word} weighting passed in-corpus only (EXPLORATORY; no claim) and would need its own "
               "registration on new data.")
    else:
        nxt = f"{seal}; the deploy rule keeps the uniform shrunk mean SH, so no sibling weighting is carried forward."
    return changed, nxt


def per_item(T: dict, design: str, *, seal_tag: str, leakage: str, seconds: float) -> dict:
    """The section-9.3 fields; ``unit_table`` (the c_u per unit) goes under the unit tests (per_item_markdown)."""
    inf = T["inference"]
    sel = inf[(inf.cand == "NEST_BOTH_sh") & (inf.ref == "SH_r4")].set_index("stratum")
    p = sel.loc["PRIMARY"]
    comp = T["comparators"]
    met = comp[comp.stratum == "PRIMARY"][["arm", "R2", "MSE", "relSSE_vs_B1_r4", "R2_per_seed", "rows_per_seed", "label"]]
    un = T["units"]
    un = un[(un.stratum == "PRIMARY") & (un.contrast == "NEST_BOTH_sh - SH_r4")][["publication", "chemotype", "c", "rows",
                                                                                  "rho", "logratio"]] if len(un) else pd.DataFrame()
    tests = sel.loc[[s for s in ("PRIMARY", DECISION_POP) if s in sel.index]][list(UNIT_TESTS)].reset_index()
    tests.insert(1, "contrast", "NEST_BOTH_sh - SH_r4")
    for c in ("G_units", "G_chemotypes"):
        tests[c] = [int(v) if pd.notna(v) else v for v in tests[c]]
    c5 = {f"{s} NEST_BOTH_sh - SH_r4": float(sel.loc[s, "dR2"]) for s in ("PRIMARY", "WITH_C5", "WITH_C5_NO_C5CELL",
                                                                          "PRIMARY_NO_ORNL") if s in sel.index}
    dup = sel.loc[[s for s in DUP_POPS if s in sel.index]][list(DUP_COLS)].reset_index()
    changed, nxt = decision_sentences(T, design)
    return {
        "item": "1.2 M03 sibling-weighting factorial (E-CT annex)",
        "population": {k: p[k] for k in ("cells", "publications", "units", "rows_per_seed", "cpg_clusters", "kish_cpg")},
        "design": design,
        "read_budget": "4 reads (code k = 3) for every kernel, 2-nearest and context arm; acid-trend arms 0 reads (labelled)",
        "arms": ("KW grid (5 acid x 5 read lengths, raw and count-shrunk; uniform limit = SIB / SH), NEST_ACID / READ / "
                 "BOTH (sh, raw; one-SE rules paired, classic, argmin), NEAR2, LEVEL2, READ2 (raw, sh), E_RAND2, "
                 "B1, SH, SIB, COPY, SH0, NEAR, NEARSH; ACOPY (MEAN), AINT, PAIRMEAN, LIN, LINCLIP, LINHALF, "
                 "LINHALFCLIP, POOLPAIR"),
        "metrics": met,
        "per_seed": f"NEST_BOTH_sh - SH_r4 per seed (PRIMARY): {p['dR2_per_seed']}",
        "units": tests, "unit_table": un,
        "lopo_top3": {"dR2": p["dR2"], "lopo_renested_lo": p.get("lopo_renested_lo"),
                      "lopo_renested_hi": p.get("lopo_renested_hi"), "lopo_rowdrop_lo": p["lopo_lo"],
                      "lopo_rowdrop_hi": p["lopo_hi"], "top3_share": p["top3_share"],
                      "p_signflip_unit": p["p_signflip_unit"]},
        "c5": c5, "duplicates": dup, "leakage": leakage,
        "compute": f"{seconds:.0f} s (build + nulls + tables); seal {seal_tag}",
        "label": f"{SEEN}; {NOT_NEXT_TO}",
        "changed": changed, "next_step": nxt,
    }


def per_item_markdown(item: dict, *, title: str) -> str:
    """r5.common.per_item_table, with the c_u unit table (``unit_table``) printed under the unit tests of its field."""
    txt = CM.per_item_table(item, None, title=title)
    ut, tests = item.get("unit_table"), item["units"]
    if not (isinstance(ut, pd.DataFrame) and len(ut) and isinstance(tests, pd.DataFrame)):
        return txt
    head = f"## {dict(CM.ITEM_FIELDS)['units']}\n\n{CM.md_table(tests)}\n"
    if txt.count(head) != 1:
        raise AssertionError("per_item_markdown: the unit-test sub-table is not in the per-item table exactly once")
    return txt.replace(head, f"{head}\nc_u per publication x chemotype unit, NEST_BOTH_sh - SH_r4 on PRIMARY "
                             f"(units_{item['design']}.csv):\n\n{CM.md_table(ut)}\n", 1)

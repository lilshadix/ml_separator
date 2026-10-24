"""Confirmation-package statistics for round 5 (CLAUDE_CODE_PROMPT.md section 4.2, 4.3 items 1-3 and 9-10, 7.5).

* C-GD and Gd-dip transport: the 95 % laboratory-mean Gd-chord t-interval (df G - 1), raw or winsorised to
  [-0.40, +0.20], read against beta_frozen, beta_frozen / 2 and 0 with C-GD's three-way rules
  (confirm: upper bound < beta; kill: lower bound > beta; A2 "helps": upper bound < beta / 2; B1-harm flag: lower
  bound > beta / 2; A1 against 0 is descriptive).  Each bound-versus-reference rule is a one-sided 0.025 test.
* C-ECT: exact one-sided sign-flip on the unit values -- under D3 fallback (ii) the chemotype units (connected
  components of the laboratory-chemotype graph; unit value = unweighted mean of its laboratories' values, each the
  unweighted mean of its families' log(sum SSE_B1@4 / sum SSE_SH) ratios; CONFIRMATION_PACKAGE 4.2) -- the exact
  binomial SIGN test on unit wins, the kill rule (mean <= 0 or win share < 0.5) and the fixed sequence SH@4 -> SH@3.
  A unit without a defined value is excluded before the call; a NaN or +-inf value raises.
* Clopper-Pearson bounds, and the null / world generators used by scripts/s0_sizegate.py and s0_power_worlds.py.

Pure numpy / scipy.  The loaders at the bottom read seen design-folder planning tables (label-based; disclosed by the
scripts that use them); nothing here reads the bench.

Re-implemented from these design-folder scripts (none is run or imported; LF SHA-256 of each source file):
  ideas2/scripts/confirm1skepB_strata.py    24ffdd32e979f6a68ddb428412761a7d254b1f9c1307ad1004d9553b1c997e4e
      two-stage chemotype -> laboratory null and the laboratory-level one-sample t
  ideas2/scripts/confirm4_skepB_power.py    7a61a4109d55f8c36fbe8770512ff9ab3edf8fd231494bc65187a41e16c3fac3
      C-ECT worlds (corpus, halved effect, sign-symmetrised null), the tau noise model, the fixed sequence
  ideas2/scripts/confirm4_skepB_power2.py   086bfc886df83d6c6f7c75fa66e99ebf4753532fff4135da482243d08db87997
      non-DGA C-ECT worlds W1 (corpus shifted to the non-DGA mean) and W2 (the 5 non-DGA units)
  ideas2/scripts/confirm4_power_skA.py      5c10eed3474a558fe90a215f1f900368df710f00dbe12649c4864a9f6e7e76b8
      null-like mixtures (each family sign-flipped at random with probability p)
  ideas2/scripts/confirm_breadth_window.py  7457b847d338659f1e68d5ad683b702a131112371cd022e840912ba1fe7a0106
      the frozen winsorisation bounds [-0.40, +0.20]
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as SS

from . import paths

# Copy rule (brief 1.2): every re-implemented source with its LF SHA-256, checked by scripts/r5_seal.py.
COPIED_FROM = {
    "generations/gen21/round5_design_20261023/ideas2/scripts/confirm1skepB_strata.py":
        "24ffdd32e979f6a68ddb428412761a7d254b1f9c1307ad1004d9553b1c997e4e",
    "generations/gen21/round5_design_20261023/ideas2/scripts/confirm4_skepB_power.py":
        "7a61a4109d55f8c36fbe8770512ff9ab3edf8fd231494bc65187a41e16c3fac3",
    "generations/gen21/round5_design_20261023/ideas2/scripts/confirm4_skepB_power2.py":
        "086bfc886df83d6c6f7c75fa66e99ebf4753532fff4135da482243d08db87997",
    "generations/gen21/round5_design_20261023/ideas2/scripts/confirm4_power_skA.py":
        "5c10eed3474a558fe90a215f1f900368df710f00dbe12649c4864a9f6e7e76b8",
    "generations/gen21/round5_design_20261023/ideas2/scripts/confirm_breadth_window.py":
        "7457b847d338659f1e68d5ad683b702a131112371cd022e840912ba1fe7a0106",
}

ALPHA = 0.025                     # one-sided, per claim (section 4.2)
SIZE_FLAG = 0.035                 # size gate: flagged if the lower limit of the two-sided 95 % Clopper-Pearson interval
                                  # (one-sided 97.5 %; P2's convention) exceeds this (section 4.3 item 2)
CLIP = (-0.40, 0.20)              # frozen winsorisation of laboratory chords (I2 confirm-1 / confirm-3)
ICC_GRID = (0.23, 0.3, 0.5, 0.69)
EXACT_MAX_G = 13                  # exact enumeration up to 2^13 = 8,192 sign vectors (section 4.1 item 2)
N_RANDOM_FLIPS = 4999
BETA_R4_FOLD_MEAN = -0.094        # IR M33 / EB section 1.5; reference only (section 7.5)
BETA_SENS = (-0.118, -0.072)      # I2 confirm-1 sensitivities
HDEHP_LIKE = -0.26                # UNVERIFIED planning value (I2 confirm-1); PROVISIONAL / PENDING-U5
P507_LIKE = -0.105                # UNVERIFIED search-snippet reading (I2 confirm-2); PROVISIONAL / PENDING-U5
G_FLOOR = {"C-GD": 6, "Gd-dip transport": 10, "C-ECT": 7}

NULL_SIZE_CSV = paths.DESIGN / "ideas2" / "confirm1skepB_out" / "clustered_null_size.csv"
STRATA_CSV = paths.DESIGN / "ideas2" / "confirm1skepB_out" / "strata_units.csv"
ECT_UNITS_CSV = paths.DESIGN / "ideas2" / "confirm4_skepB_out" / "units_primary.csv"
ECT_NONDGA_CSV = paths.DESIGN / "ideas2" / "confirm4_skepB_out" / "nondga_units.csv"
BETA_FROZEN_JSON = paths.RES / "stage0" / "beta_frozen.json"

# chord statistics: id -> (claim, winsorisation, role)
CHORD_STATS = {"CGD_W": ("C-GD", CLIP, "primary"),
               "CGD_R": ("C-GD", None, "sensitivity"),
               "GDT_R": ("Gd-dip transport", None, "primary (M33 laboratory-mean t-interval)")}
# chord rules: (reference, tail, rule, gating).  tail 'lower' = upper bound below the reference; 'upper' = lower
# bound above it.  A1 against 0 is descriptive (I2 confirm-3), so its rows never gate.
CHORD_RULES = (("beta", "lower", "confirm", True),
               ("beta", "upper", "kill", True),
               ("beta_half", "lower", "helps", True),
               ("beta_half", "upper", "harm", True),
               ("zero", "lower", "A1", False),
               ("zero", "upper", "above0", False))


# ---------------------------------------------------------------------------------------------------- bounds
def clopper_pearson(k, n, conf: float = 0.95):
    """Two-sided Clopper-Pearson interval for k successes in n trials (exact; 0 / 1 at the edges)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    a = (1 - conf) / 2
    lo = np.where(k > 0, SS.beta.ppf(a, np.maximum(k, 1), n - k + 1), 0.0)
    hi = np.where(k < n, SS.beta.ppf(1 - a, k + 1, np.maximum(n - k, 1)), 1.0)
    if lo.ndim == 0:
        return float(lo), float(hi)
    return lo, hi


# ---------------------------------------------------------------------------------------------------- tests
def sign_matrix(G: int) -> np.ndarray:
    """All 2^G sign vectors (rows); row 0 is all +1."""
    return 1.0 - 2.0 * ((np.arange(2 ** G)[:, None] >> np.arange(G)) & 1)


def signflip_p(X, rng=None, B: int = N_RANDOM_FLIPS, chunk: int = 512) -> np.ndarray:
    """One-sided sign-flip p of T = sum(x) per row (H1: mean > 0).  G <= 13: exact enumeration,
    p = #{s : T(s) >= T} / 2^G; otherwise B random flips, p = (1 + #{T* >= T}) / (1 + B) (section 4.1 item 2).
    Ties count as >= (conservative); the tolerance absorbs summation-order rounding."""
    X = np.atleast_2d(np.asarray(X, float))
    if not np.isfinite(X).all():
        raise ValueError("non-finite unit value")
    R, G = X.shape
    T = X.sum(1)
    tol = 1e-9 * np.abs(X).sum(1) + 1e-15
    if G <= EXACT_MAX_G:
        E, add, den = sign_matrix(G), 0, 2.0 ** G
    else:
        rng = rng if rng is not None else np.random.default_rng(paths.seed_for(f"signflip|{G}"))
        E, add, den = rng.choice((-1.0, 1.0), size=(B, G)), 1, B + 1.0
    cnt = np.empty(R)
    for i in range(0, R, chunk):
        s = slice(i, i + chunk)
        cnt[s] = ((X[s] @ E.T) >= (T[s] - tol[s])[:, None]).sum(1)
    return (cnt + add) / den


def sign_test_p(X) -> np.ndarray:
    """Exact one-sided binomial sign test on unit wins (x > 0), ties (x == 0) dropped: P(Bin(n, 1/2) >= wins)."""
    X = np.atleast_2d(np.asarray(X, float))
    if not np.isfinite(X).all():
        raise ValueError("non-finite unit value")
    w, n = (X > 0).sum(1), (X != 0).sum(1)
    return np.where(n > 0, SS.binom.sf(w - 1, n, 0.5), 1.0)


def t_interval(X, conf: float = 0.95, clip=None):
    """Laboratory-mean t-interval (df G - 1) per row; clip = (lo, hi) winsorises every laboratory value first."""
    X = np.atleast_2d(np.asarray(X, float))
    if not np.isfinite(X).all():
        raise ValueError("non-finite unit value")
    if clip is not None:
        X = np.clip(X, *clip)
    G = X.shape[1]
    m = X.mean(1)
    h = SS.t.ppf(1 - (1 - conf) / 2, G - 1) * X.std(1, ddof=1) / np.sqrt(G)
    return m - h, m + h


def chord_refs(beta: float) -> dict:
    return {"beta": beta, "beta_half": beta / 2, "zero": 0.0}


def chord_rule_hits(lo, hi, refs: dict) -> dict:
    """Boolean arrays of every rule in CHORD_RULES for interval bounds (lo, hi)."""
    return {rule: (hi < refs[ref]) if tail == "lower" else (lo > refs[ref]) for ref, tail, rule, _ in CHORD_RULES}


def ect_decisions(X, X3=None, rng=None) -> dict:
    """C-ECT on unit values X (R x G): confirm = exact sign-flip p <= 0.025; kill = mean <= 0 or win share < 0.5
    (only where not confirmed; 'overlap' counts rows meeting both); SIGN; fixed-sequence SH@3 on X3 if given.  A unit
    with no defined value is excluded before the call: a NaN or +-inf value raises (CONFIRMATION_PACKAGE section 4.2)."""
    X = np.atleast_2d(np.asarray(X, float))
    p = signflip_p(X, rng)
    conf = p <= ALPHA + 1e-12
    kill_raw = (X.mean(1) <= 0) | ((X > 0).mean(1) < 0.5)
    out = {"p_sf": p, "confirm": conf, "kill": kill_raw & ~conf, "overlap": kill_raw & conf,
           "inconclusive": ~conf & ~kill_raw, "sign_reject": sign_test_p(X) <= ALPHA + 1e-12}
    if X3 is not None:
        out["seq_SH3"] = conf & (signflip_p(X3, rng) <= ALPHA + 1e-12)
    return out


# ---------------------------------------------------------------------------------------------------- draws
def two_stage(rng, R: int, comp, sd_within: float, icc: float) -> np.ndarray:
    """R x G laboratory values (mean 0) under the two-stage chemotype -> laboratory null: a laboratory of chemotype j
    is b_j + e_g, b_j ~ N(0, sd_between^2), e_g ~ N(0, sd_within^2), sd_between = sd_within sqrt(icc / (1 - icc))
    (confirm1skepB_strata.py); comp lists the laboratories per chemotype."""
    comp = tuple(int(c) for c in comp)
    sdb = sd_within * np.sqrt(icc / (1 - icc))
    cb = rng.normal(0, sdb, size=(R, len(comp)))
    return np.repeat(cb, comp, axis=1) + rng.normal(0, sd_within, size=(R, sum(comp)))


def iid(rng, R: int, G: int, kind: str, sd: float = 1.0, pool=None) -> np.ndarray:
    """R x G i.i.d. draws: 'normal' N(0, sd^2); 't3' Student t(3) scaled to sd; 'pool' resampled from pool (as given);
    'swap' pool values with random signs (exactly symmetric)."""
    if kind == "normal":
        return rng.normal(0, sd, size=(R, G))
    if kind == "t3":
        return rng.standard_t(3, size=(R, G)) * sd / np.sqrt(3.0)
    pool = np.asarray(pool, float)
    X = pool[rng.integers(len(pool), size=(R, G))]
    if kind == "pool":
        return X
    if kind == "swap":
        return X * rng.choice((-1.0, 1.0), size=(R, G))
    raise ValueError(kind)


def tau_noise(X, tau: float, rng) -> np.ndarray:
    """Per-source noise on a log SSE ratio (P2's tau; confirm4_skepB_power.py): X + 2 log((1 - e1) / (1 - e2)),
    e ~ N(0, (tau / sqrt 2)^2)."""
    if tau <= 0:
        return X
    s = tau / np.sqrt(2)
    return X + 2 * np.log((1 - rng.normal(0, s, size=X.shape)) / (1 - rng.normal(0, s, size=X.shape)))


# ---------------------------------------------------------------------------------------------------- inputs
def load_compositions(path=NULL_SIZE_CSV):
    """(compositions in file order, sd_within grid, the table) of the confirm-1 clustered-null file."""
    t = pd.read_csv(path)
    comps = list(dict.fromkeys(tuple(ast.literal_eval(c)) for c in t["comp"]))
    return comps, tuple(sorted(t.sd_within.unique())), t


def load_unit_chords(path=STRATA_CSV) -> pd.DataFrame:
    """Seen corpus pub x chemotype unit Gd chords (raw unit means; 44 C3 units; cls DGA / nonDGA_HS / nonDGA_LS)."""
    t = pd.read_csv(path)
    return t[["unit", "cls", "c", "n"]].rename(columns={"c": "chord"})


def load_ect_units():
    """Seen P1 PRIMARY per-family log ratios (14 pub x chemotype units, BPG): L_SH4, L_SH3; and the 5 non-DGA
    no-C5 units' L_SH4 (any n_sib)."""
    u = pd.read_csv(ECT_UNITS_CSV)
    n = pd.read_csv(ECT_NONDGA_CSV)
    nd = n[(~n.dga.astype(bool)) & (~n.c5.astype(bool))].L_SH4.to_numpy(float)
    return u.L_SH4.to_numpy(float), u.L_SH3.to_numpy(float), nd


def parse_beta_frozen(d: dict) -> float:
    """The numeric beta_frozen of a beta_frozen.json record: field beta_frozen, beta or value (a number, or a dict
    holding beta / value)."""
    v = d.get("beta_frozen", d.get("beta", d.get("value")))
    if isinstance(v, dict):
        v = v.get("beta", v.get("value"))
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise KeyError("no numeric beta_frozen / beta / value field")
    return float(v)


def load_beta_frozen(path=BETA_FROZEN_JSON, allow_todo: bool = False) -> dict:
    """beta_frozen from results/stage0/beta_frozen.json (section 4.3 item 9).  When the file is missing this raises
    FileNotFoundError, unless ``allow_todo`` is True (construction tests only): then the round-4 fold mean -0.094 is
    returned marked TODO (every output computed with it must be rerun once beta_frozen is sealed)."""
    p = Path(path)
    if not p.exists() and not allow_todo:
        raise FileNotFoundError(f"{p}: beta_frozen is a required Stage-0 input (brief 4.3 item 9)")
    if p.exists():
        v = parse_beta_frozen(json.loads(p.read_text(encoding="utf-8")))
        src = next(iter(digests([p])))
        return {"value": v, "source": src, "sha256_lf": paths.sha256_text_lf(p), "todo": None}
    return {"value": BETA_R4_FOLD_MEAN, "source": "round-4 fold mean (IR M33; EB section 1.5)", "sha256_lf": None,
            "todo": "TODO: results/stage0/beta_frozen.json missing; rerun after beta_frozen is computed (4.3 item 9)"}


def digests(files) -> dict:
    """LF SHA-256 of each input file, keyed by its path relative to the repository root."""
    return {str(Path(f).resolve().relative_to(paths.ROOT)).replace("\\", "/"): paths.sha256_text_lf(f) for f in files}

"""Stage-1 item 1.7 (CLAUDE_CODE_PROMPT.md 6.7; 6.0, 6.10, 8, 9): E-W, the M13 wider-pair table.  Every in-corpus
number is EXPLORATORY (seen); nothing here is a test, and nothing is ever pooled with the strict-adjacent pairs.

Rows.  Every C3 cell instance (k_max >= 3) of every round-4 dev fold at E-A's read set (k = 3: four reads, the union of
the cached ``cell['seqs']['DOPT_Q'][:3]``; brief 3.3).  Level curves per arm from the read metals only:
  B1       r4.refs.predict_levels with the fold's stored round-4 beta (results/REF/fitted_<design>.json);
  B0       the same path with beta = 0 (= PWL_R);
  LINE     the slope-only line through the reads (gen20k.adapters.naive_line on the gen20 support; round 4's LINE);
  HIST_D7  the fixed -0.072 (DIAGNOSTIC: not fold-independent).
A pair (A lighter, B heavier) of measured metals is scored as SF = level(A) - level(B) on identical rows for every arm.
Families: STRICT = the cell's strict-adjacent queries (E-A's rows: the reproduction target and the multiplier
denominators only); WIDE = the frozen cut-pair set PAIRS (dZ = Z_B - Z_A in 2..4, Nd/Sm included, plus the NdFeB feed
pairs Nd/Tb, Nd/Dy, Pr/Dy) with both ends measured and at least one end unread (a pair with both ends read is known,
never scored).  dZ is the atomic-number difference (Pm, Z 61, is absent: Nd/Sm has dZ 2).

Statistics (per frame, identical rows): pooled R^2 inside each seed then the seed mean (never clipped); skill against
LINE and against B0 = seed mean of 1 - SSE_B1 / SSE_ref; dR2 = (SSE_ref - SSE_B1) / SST; the MSE and var(y)
multipliers = seed means of the per-seed ratio to the strict-adjacent rows of the same variant; MAE relative to the
median |y|; sign agreement (sign(p) == sign(y), a zero prediction counted as a disagreement unless y = 0 exactly) in
the frozen |y| bins; cluster coverage over the 13 cluster_cpg clusters; leave-one-publication-out (jackknife t(P - 1)
interval and the min / max leave-out values); the cluster_cpg unit table (r5.common.unit_stats / unit_tests,
DIAGNOSTIC) and its top-3 unit share; C5-BTBP in / out; duplicate sensitivities.

The one pre-declared open contrast: Nd/Sm B1 - LINE in R^2 units, computed inside each cluster_cpg cluster that holds
>= 10 Nd/Sm rows per seed (seed mean) other than the giant cluster, equal-weighted over those clusters, with a
leave-one-cluster-out jackknife-t 95 % interval.  "Interpolation beats slope at the light|middle cut" is retired if the
contrast is <= +0.05 or its lower bound is <= 0 (or fewer than 2 clusters qualify).  Heavy cuts (both ends in Ho..Lu)
are explicit non-claims, reported by whether Tm or Lu is read.
"""
# No design-folder script is copied: the design is re-implemented from the round-5 design folder's
#   ideas/IDEAS_RANKED.md section 12 (LF SHA-256 41f7bc735c3a6d36f4c0a5c1c6bd6ae341f91f4aaed754fc7d99bffe115595e2) and
#   ideas/critiques.json M13 (LF SHA-256 585468e5ea8267f7e8df650a2115843640b37d6d1e29f6050f50e2c306d129d3).
# Level curves: round4_20261022/r4/refs.py predict_levels, imported unchanged (LF SHA-256
#   2ab162ca47b854985380751bee0bbff1560d4fb36f8c9637ce3e1c08d6196eee); LINE: gen20k.adapters.naive_line, the call of
#   g21/evaluate.py gen20_baselines['LINE'] (LF SHA-256 d75752734d0e68c511dea1b56d6b1f20ef55e9c98c7fa8feba3bd052c7ed2209).
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as SS

from . import paths
from . import common as CM
from r4 import refs as RF  # noqa: E402
from g21 import cache as GC  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import adapters as AD  # noqa: E402
from gen20k.data import LANTHANIDES, N_LN, Z_VEC  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402

# Copy rule (brief 1.2): every copied or re-implemented source with its LF SHA-256 at the copy; checked by
# scripts/r5_seal.py (copied_sources reads this declaration).
COPIED_FROM = {
    "generations/gen21/round5_design_20261023/ideas/IDEAS_RANKED.md":
        "41f7bc735c3a6d36f4c0a5c1c6bd6ae341f91f4aaed754fc7d99bffe115595e2",
    "generations/gen21/round5_design_20261023/ideas/critiques.json":
        "585468e5ea8267f7e8df650a2115843640b37d6d1e29f6050f50e2c306d129d3",
    "generations/gen21/round4_20261022/r4/refs.py":
        "2ab162ca47b854985380751bee0bbff1560d4fb36f8c9637ce3e1c08d6196eee",
    "generations/gen21/g21/evaluate.py":
        "d75752734d0e68c511dea1b56d6b1f20ef55e9c98c7fa8feba3bd052c7ed2209",
}

ITEM_ID, ESTIMAND = "1.7", "E-W"
OUT = paths.RES / "EW"
SEEN = "EXPLORATORY (seen)"
KEY = list(CM.KEY)                                       # split_seed, fold, cell_id, A, B
SEEDS = tuple(int(s) for s in paths.DEV_SEEDS)
K_EA = 3                                                 # E-A's code budget: four reads
ARMS = ("B1", "B0", "LINE", "HIST_D7")                   # frozen arm list (HIST_D7 DIAGNOSTIC)
COMPARATORS = ("LINE", "B0")                             # skill of B1 against these
IDX = {n: i for i, n in enumerate(LANTHANIDES)}
ZV = np.asarray(Z_VEC, int)
DZ_SET = (2, 3, 4)
NDFEB = (("Nd", "Tb"), ("Nd", "Dy"), ("Pr", "Dy"))       # NdFeB feed pairs (dZ 5, 6, 7)
FEED = (("Nd", "Sm"),) + NDFEB + (("Eu", "Tb"),)         # the feed-defined pair set (brief 6.7)
REGION = {**{m: "light" for m in ("La", "Ce", "Pr", "Nd")}, **{m: "middle" for m in ("Sm", "Eu", "Gd", "Tb", "Dy")},
          **{m: "heavy" for m in ("Ho", "Er", "Tm", "Yb", "Lu")}}
TMLU = (IDX["Tm"], IDX["Lu"])
SIGN_EDGES = (0.05, 0.10, 0.20, 0.50)                    # |y| bins [0, .05) [.05, .10) [.10, .20) [.20, .50) [.50, inf)
SIGN_BINS = ("[0,0.05)", "[0.05,0.10)", "[0.10,0.20)", "[0.20,0.50)", "[0.50,inf)")
GIANT = "g:pub_74428aeab2"                               # round 4's giant cluster_cpg (64.9 % of E-A rows; asserted)
MIN_ROWS = 10                                            # Nd/Sm rows per seed for a cluster to enter the contrast
RETIRE_AT = 0.05                                         # retired if contrast <= +0.05 or its lower bound <= 0
LEVEL = 0.95                                             # jackknife-t interval (stated once, REGISTRY / brief 6.10)
TARGETS = {"B1_vs_REF_max_abs": 8.9e-16, "rows_per_design": 9010}   # brief 6.0 (P1 section 2): 18,020 over both
VARIANTS = ("all", "noC5", "noC5cell")
DUP_VARIANTS = ("noV2", "noBITDUP")
MAIN_FRAMES = ("dZ2", "dZ3-4", "Nd/Sm", "feed NdFeB", "feed set")
SEEN_IR = (                                              # IR M13 / brief 6.7 seen values (BPG; IR used the fixed -0.072)
    ("dZ2", "R2_B1", 0.676), ("dZ2", "R2_B0", 0.659), ("dZ2", "R2_LINE", 0.447), ("Nd/Sm", "R2_B1", 0.765),
    ("Nd/Sm", "R2_LINE", 0.681), ("dZ3-4", "R2_B1", 0.811), ("dZ2", "macro_R2_B1", "0.521-0.529"),
    ("dZ2", "var_mult", 2.2), ("dZ2", "MSE_mult_B1", 1.30), ("STRICT", "skill_vs_LINE", 0.28),
    ("Nd/Sm", "skill_vs_LINE", 0.26), ("Nd/Sm", "dR2_B1_minus_LINE", 0.084),
    ("NDSM_OUTSIDE_GIANT", "dR2_B1_minus_LINE", "0.011-0.013"), ("Ho/Tm", "R2_B1", 0.16), ("Er/Yb", "R2_B1", 0.07),
    ("Tm/Lu", "R2_B1", 0.10), ("TRAP", "R2_B1", 0.551))
pd.set_option("future.no_silent_downcasting", True)


# ------------------------------------------------------------------------------------------------------------
# the frozen pair set and row descriptors
# ------------------------------------------------------------------------------------------------------------
def pair_name(a: int, b: int) -> str:
    return f"{LANTHANIDES[a]}/{LANTHANIDES[b]}"


def pair_universe() -> tuple:
    """Every (a, b) index pair of the frozen cut set, A lighter: dZ in DZ_SET plus the NdFeB pairs, sorted by (a, b)."""
    out = {(a, b) for a in range(N_LN) for b in range(N_LN) if ZV[b] - ZV[a] in DZ_SET}
    out |= {(IDX[x], IDX[y]) for x, y in NDFEB}
    return tuple(sorted(out))


PAIRS = pair_universe()
PAIR_NAMES = tuple(pair_name(a, b) for a, b in PAIRS)
FEED_NAMES = tuple(f"{x}/{y}" for x, y in FEED)
NDFEB_NAMES = tuple(f"{x}/{y}" for x, y in NDFEB)


def cut_class(dz: int) -> str:
    return f"dZ{dz}" if dz in DZ_SET else "feed dZ5+"


def region(a: int, b: int) -> str:
    ra, rb = REGION[LANTHANIDES[a]], REGION[LANTHANIDES[b]]
    return ra if ra == rb else f"{ra}|{rb}"


def wider_pairs(metals, read) -> list:
    """Frozen cut pairs with both ends measured in the cell and at least one end unread."""
    ms, rd = set(metals), set(read)
    return [(a, b) for a, b in PAIRS if a in ms and b in ms and not (a in rd and b in rd)]


def rs_str(read) -> str:
    """Round 4's read_set string ('La,Pr,Gd,Yb': names in series order)."""
    return ",".join(LANTHANIDES[m] for m in sorted(read))


# ------------------------------------------------------------------------------------------------------------
# level curves (exact round-4 paths)
# ------------------------------------------------------------------------------------------------------------
def fit_beta(beta: float) -> dict:
    return {"name": "B1", "spec": RF.SPECS["B1"], "kind": "R", "coef": np.array([float(beta)])}


def stored_betas(design: str) -> dict:
    """(seed, fold) -> (beta, lambda) of round 4's stored B1 fits."""
    d = json.loads((paths.R4RES / "REF" / f"fitted_{design}.json").read_bytes())
    return {tuple(map(int, k.split("_"))): (float(v["ref:B1"]["coef"][0]), str(v["ref:B1"]["lambda"]))
            for k, v in d.items() if "_" in k and k.replace("_", "").isdigit()}


def arm_levels(basis, cell: dict, yrow, beta: float, k: int = K_EA):
    """(obs, {arm: level vector (14,)}) from the cell's first k DOPT_Q steps; only the read metals enter."""
    obs = E.observe(np.asarray(yrow, float), cell, cell["seqs"]["DOPT_Q"], k)
    m = cell["metals"]
    L = {"B1": RF.predict_levels(fit_beta(beta), obs, m), "B0": RF.predict_levels(fit_beta(0.0), obs, m),
         "LINE": np.asarray(AD.naive_line(basis, obs.support), float),
         "HIST_D7": RF.predict_levels(RF.HIST_D7_FIT, obs, m)}
    return obs, L


class Checks:
    """Leakage / support counters (brief 8 items 1, 4, 5); every *_bad / *_failed counter must stay 0."""

    def __init__(self):
        self.c = {"folds": 0, "isolation_ok": 0, "cell_instances": 0, "strict_rows": 0, "wide_rows": 0,
                  "wide_both_read_excluded": 0, "both_ends_read_bad": 0, "orientation_bad": 0, "support_not_exact_bad": 0,
                  "nonfinite_pred_bad": 0, "unmeasured_metal_bad": 0, "intervention_checked": 0,
                  "intervention_failed": 0, "peek_control_runs": 0, "peek_control_caught": 0}

    def add(self, k, v=1):
        self.c[k] += int(v)


def bad_checks(c: dict, *, intervention: bool = True) -> list:
    """The failed leakage / support counters of a Checks dict (a positive control not caught counts as failed).  With
    ``intervention`` (every sealed run) the check must also have full coverage (s0-mechanics-5): every cell instance
    intervened on, and one caught block-Y peek per fold."""
    out = [k for k, v in c.items() if (k.endswith("_bad") or k == "intervention_failed") and v]
    if c.get("peek_control_caught", 0) != c.get("peek_control_runs", 0):
        out.append("peek_control_not_caught")
    if intervention:
        if not c.get("intervention_checked", 0) or c.get("intervention_checked") != c.get("cell_instances"):
            out.append("intervention_coverage")
        if c.get("peek_control_runs") != c.get("folds"):
            out.append("peek_control_coverage")
    return out


def cell_rows(basis, cell: dict, yrow, beta: float, base: dict, chk: Checks, rng=None) -> list:
    """STRICT and WIDE rows of one cell instance (rng given: the prediction-intervention check runs)."""
    yrow = np.asarray(yrow, float)
    metals = tuple(cell["metals"])
    if not np.isfinite(yrow[list(metals)]).all():
        chk.add("unmeasured_metal_bad")
    obs, L = arm_levels(basis, cell, yrow, beta)
    read = set(obs.read)
    for a in ("B1", "B0", "HIST_D7"):                     # exact supports (LINE is a slope, exempt as in round 4)
        chk.add("support_not_exact_bad", sum(L[a][m] != yrow[m] for m in read))
    if rng is not None:
        ys = E.scramble_hidden(yrow, tuple(read), rng)
        obs_s, Ls = arm_levels(basis, cell, ys, beta)
        ok = obs_s.read == obs.read and all(np.array_equal(L[a][list(metals)], Ls[a][list(metals)], equal_nan=True)
                                            for a in ARMS)
        chk.add("intervention_checked")
        chk.add("intervention_failed", not ok)
    strict = [(int(a), int(b)) for a, b in cell["query"]]
    wide = wider_pairs(metals, read)
    chk.add("wide_both_read_excluded", sum(1 for a, b in PAIRS if a in metals and b in metals and a in read and b in read))
    rows, rss = [], rs_str(read)
    tm_r, lu_r = TMLU[0] in read, TMLU[1] in read
    for fam, pairs in (("STRICT", strict), ("WIDE", wide)):
        for a, b in pairs:
            dz = int(ZV[b] - ZV[a])
            if dz <= 0 or (fam == "STRICT" and dz != 1) or (fam == "WIDE" and dz < 2):
                chk.add("orientation_bad")
            if a in read and b in read:
                chk.add("both_ends_read_bad")
            p = {f"p_{n}": float(L[n][a] - L[n][b]) for n in ARMS}
            if not all(np.isfinite(v) for v in p.values()):
                chk.add("nonfinite_pred_bad")
            nm = pair_name(a, b)
            rows.append({**base, "read_set": rss, "family": fam, "A": LANTHANIDES[a], "B": LANTHANIDES[b], "ia": a,
                         "ib": b, "pair": nm, "dZ": dz, "cut_class": "strict" if fam == "STRICT" else cut_class(dz),
                         "region": region(a, b), "feed": nm in FEED_NAMES, "ndfeb": nm in NDFEB_NAMES,
                         "heavy_nonclaim": fam == "WIDE" and region(a, b) == "heavy", "a_read": a in read,
                         "b_read": b in read, "tm_read": tm_r, "lu_read": lu_r, "y": float(yrow[a] - yrow[b]), **p})
            chk.add("strict_rows" if fam == "STRICT" else "wide_rows")
    return rows


def compute_design(bench, design: str, *, folds=None, cluster_of=None, intervention: bool = True, log=print) -> dict:
    """Row-level STRICT + WIDE outputs of one design over the round-4 dev folds (``folds``: first n only, smoke)."""
    t0 = time.time()
    Y = np.asarray(bench.Y, float)
    betas = stored_betas(design)
    chk = Checks()
    rng = np.random.default_rng(paths.seed_for(f"EW|intervention|{design}")) if intervention else None
    fl = folds_for(bench.frame, design, seeds=list(paths.DEV_SEEDS))
    if folds is not None:
        fl = fl[:folds]
    rows = []
    for f in fl:
        rec = GC.load(f.design, f.seed, f.fold)
        E.assert_test_isolation(bench, rec)
        chk.add("folds")
        chk.add("isolation_ok")
        beta, lam = betas[(int(f.seed), int(f.fold))]
        peeked = False
        for cell in rec["outer"]["cells"]:
            if cell["k_max"] < 3:
                continue
            chk.add("cell_instances")
            ci, cid = int(cell["ci"]), str(cell["cell_id"])
            base = {"design": design, "split_seed": int(f.seed), "fold": int(f.fold), "cell_id": cid,
                    "publication": str(cell["publication"]), "chemotype": str(cell["chemotype"]),
                    "extractant": str(cell["extractant"]), "cluster_cpg": (cluster_of or {}).get(cid, ""),
                    "k_max": int(cell["k_max"]), "n_metals": len(cell["metals"]), "beta": beta, "lambda": lam}
            rows += cell_rows(bench.basis, cell, Y[ci], beta, base, chk, rng)
            if intervention and not peeked:                 # positive control: a block-Y peek must be caught
                # s0-mechanics-6: through the real level function and comparator -- a predictor that reads the cell's
                # whole DOPT_Q sequence (a metal outside the K_EA read set) must change when the unread metals change
                seq = cell["seqs"]["DOPT_Q"]
                read = set(m for p in seq[:K_EA] for m in p)
                leaky = set(m for p in seq for m in p)
                if leaky - read:
                    ys = E.scramble_hidden(Y[ci], tuple(read), rng)
                    _, L1 = arm_levels(bench.basis, cell, Y[ci], beta, k=len(seq))
                    _, L2 = arm_levels(bench.basis, cell, ys, beta, k=len(seq))
                    mets = list(cell["metals"])
                    chk.add("peek_control_runs")
                    chk.add("peek_control_caught", not all(np.array_equal(L1[a][mets], L2[a][mets], equal_nan=True)
                                                           for a in ARMS))
                    peeked = True
        log(f"  {design} {f.seed} fold {f.fold}: rows so far {len(rows)}; {time.time() - t0:.0f}s")
    R = pd.DataFrame(rows)
    if len(R):
        R = R.sort_values(KEY).reset_index(drop=True)
        if R.duplicated(KEY).any():
            raise AssertionError("duplicated row key")
    return {"rows": R, "checks": chk.c, "seconds": round(time.time() - t0, 1)}


def cluster_map() -> dict:
    """cell_id -> cluster_cpg on the full C3 E-A row set (round 4's 13 clusters); identical under BPG and BP."""
    m = {}
    for d in paths.DESIGNS:
        mm = CM.load_rows(d).groupby("cell_id").cluster_cpg.first().astype(str).to_dict()
        if m and mm != m:
            raise AssertionError("cluster_cpg differs between designs")
        m = mm
    return m


def row_labels(R: pd.DataFrame) -> pd.Series:
    lab = {(d, f): label_for("all", d, "row-level output; HIST_D7 column DIAGNOSTIC"
                             + ("; STRICT = E-A rows (reproduction / denominators only)" if f == "STRICT" else ""))
           for d in R.design.astype(str).unique() for f in R.family.astype(str).unique()}
    return pd.Series([lab[(d, f)] for d, f in zip(R.design.astype(str), R.family.astype(str))], index=R.index,
                     dtype="category")


# ------------------------------------------------------------------------------------------------------------
# reproduction target (brief 6.0; TECHNICAL SMOKE)
# ------------------------------------------------------------------------------------------------------------
def repro_checks(R: pd.DataFrame, design: str) -> dict:
    """STRICT rows against round 4's REF parquet: y exact, B1 within 8.9e-16, B0 / HIST_D7 / LINE, read sets; every
    REF row of the folds present must be matched (all 9,010 per design in a full run)."""
    S = R[R.family.astype(str) == "STRICT"]
    ref = CM.load_rows(design)
    folds = set(zip(S.split_seed.astype(int), S.fold.astype(int)))
    full = len(folds) == len(set(zip(ref.split_seed.astype(int), ref.fold.astype(int))))
    ref = ref[[(int(s), int(f)) in folds for s, f in zip(ref.split_seed, ref.fold)]]
    cols = ["y", "B0", "B1", "HIST_D7", "LINE", "read_set"]

    def keyed(t):
        return t.assign(**{c: t[c].astype(str) for c in ("cell_id", "A", "B")})
    m = keyed(S).merge(keyed(ref)[KEY + cols].rename(columns={c: c + "_ref" for c in cols}), on=KEY)
    diff = {c: float((m[("p_" + c) if c != "y" else "y"] - m[c + "_ref"]).abs().max()) if len(m) else float("nan")
            for c in ("y", "B0", "B1", "HIST_D7", "LINE")}
    out = {"label": "TECHNICAL SMOKE (reproduction run; brief 6.0)", "design": design, "full_run": bool(full),
           "folds": len(folds), "rows_STRICT": int(len(S)), "rows_REF": int(len(ref)), "rows_matched": int(len(m)),
           "max_abs_diff": diff, "target_B1_max_abs": TARGETS["B1_vs_REF_max_abs"],
           "read_sets_equal_share": float((m.read_set.astype(str) == m.read_set_ref.astype(str)).mean()) if len(m)
           else float("nan"),
           "rows_WIDE": int((R.family.astype(str) == "WIDE").sum())}
    ok = (len(m) == len(ref) == len(S) and diff["y"] == 0.0 and diff["B1"] <= TARGETS["B1_vs_REF_max_abs"]
          and out["read_sets_equal_share"] == 1.0 and (not full or len(S) == TARGETS["rows_per_design"]))
    out["identity_ok"] = bool(ok)
    out["ok"] = bool(ok and full)
    return out


# ------------------------------------------------------------------------------------------------------------
# statistics: per-seed sufficient sums (pooled R^2 / SSE ratios inside each seed, then the seed mean)
# ------------------------------------------------------------------------------------------------------------
SUMC = ("one", "y", "yy") + tuple(f"se_{a}" for a in ARMS)
J = {c: i for i, c in enumerate(SUMC)}


def prepare(R: pd.DataFrame) -> pd.DataFrame:
    """Analysis frame: squared / absolute errors per arm, string keys."""
    R = R.copy()
    for c in ("design", "cell_id", "publication", "chemotype", "extractant", "cluster_cpg", "family", "A", "B", "pair",
              "cut_class", "region", "read_set"):
        if c in R:
            R[c] = R[c].astype(str)
    y = R.y.to_numpy(float)
    for a in ARMS:
        e = R[f"p_{a}"].to_numpy(float) - y
        R[f"se_{a}"], R[f"ae_{a}"] = e * e, np.abs(e)
    R["one"], R["yy"] = 1.0, y * y
    return R


def sums(F: pd.DataFrame, by: str | None = None):
    """(A, groups): A[g, s, j] = sum over the rows of group g in seed s of SUMC[j] (seeds = the 5 dev seeds, zero
    rows where a group has none)."""
    if by is None:
        g = F.groupby("split_seed")[list(SUMC)].sum().reindex(list(SEEDS), fill_value=0.0)
        return g.to_numpy(float)[None], ["all"]
    g = F.groupby([F[by].astype(str), "split_seed"])[list(SUMC)].sum()
    groups = sorted(g.index.get_level_values(0).unique())
    g = g.reindex(pd.MultiIndex.from_product([groups, list(SEEDS)]), fill_value=0.0)
    return g.to_numpy(float).reshape(len(groups), len(SEEDS), len(SUMC)), groups


def _sst(a):
    return a[:, J["yy"]] - a[:, J["y"]] ** 2 / np.maximum(a[:, J["one"]], 1e-300)


def r2_ps(a, arm: str) -> np.ndarray:
    n, sst = a[:, J["one"]], _sst(a)
    ok = (n >= 2) & (sst > 0)
    return np.where(ok, 1.0 - a[:, J[f"se_{arm}"]] / np.where(ok, sst, 1.0), np.nan)


def skill_ps(a, arm: str, ref: str) -> np.ndarray:
    """1 - SSE_arm / SSE_ref per seed (relative pair-SSE; positive favours arm)."""
    r = a[:, J[f"se_{ref}"]]
    ok = (a[:, J["one"]] > 0) & (r > 0)
    return np.where(ok, 1.0 - a[:, J[f"se_{arm}"]] / np.where(ok, r, 1.0), np.nan)


def delta_ps(a, arm: str, ref: str) -> np.ndarray:
    """(SSE_ref - SSE_arm) / SST per seed = R2_arm - R2_ref (pooled-R^2 units)."""
    n, sst = a[:, J["one"]], _sst(a)
    ok = (n >= 2) & (sst > 0)
    return np.where(ok, (a[:, J[f"se_{ref}"]] - a[:, J[f"se_{arm}"]]) / np.where(ok, sst, 1.0), np.nan)


def mse_ps(a, arm: str) -> np.ndarray:
    n = a[:, J["one"]]
    return np.where(n > 0, a[:, J[f"se_{arm}"]] / np.maximum(n, 1e-300), np.nan)


def var_ps(a) -> np.ndarray:
    n = a[:, J["one"]]
    return np.where(n > 0, _sst(a) / np.maximum(n, 1e-300), np.nan)


def smean(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def _span(v) -> str:
    return ";".join(f"{x:.4f}" for x in np.asarray(v, float))


def lopo(F: pd.DataFrame, stat, by: str = "publication", level: float = LEVEL) -> dict:
    """r5.common.lopo on per-seed sums: stat(a (seeds x SUMC)) -> float; leave-one-<by>-out by subtraction."""
    A, groups = sums(F, by)
    tot = A.sum(0)
    point = float(stat(tot))
    jk = np.array([float(stat(tot - A[i])) for i in range(len(groups))])
    ok = np.isfinite(jk)
    v, gs = jk[ok], [g for g, o in zip(groups, ok) if o]
    P = len(v)
    if P < 2:
        return {"point": point, "se": np.nan, "lo": np.nan, "hi": np.nan, "P": P, "min": np.nan, "min_by": None,
                "max": np.nan, "max_by": None}
    se = float(np.sqrt((P - 1) / P * ((v - v.mean()) ** 2).sum()))
    q = float(SS.t.ppf(0.5 + level / 2, P - 1))
    return {"point": point, "se": se, "lo": point - q * se, "hi": point + q * se, "P": P, "min": float(v.min()),
            "min_by": gs[int(np.argmin(v))], "max": float(v.max()), "max_by": gs[int(np.argmax(v))]}


def jackknife_mean(values, level: float = LEVEL) -> dict:
    """Equal-weight mean over G units with its leave-one-unit-out jackknife-t interval (t(G - 1))."""
    v = np.asarray(values, float)
    G = len(v)
    est = float(v.mean()) if G else float("nan")
    if G < 2:
        return {"estimate": est, "G": G, "se": float("nan"), "lo": float("nan"), "hi": float("nan"), "df": max(G - 1, 0)}
    loo = (v.sum() - v) / (G - 1)
    se = float(np.sqrt((G - 1) / G * ((loo - loo.mean()) ** 2).sum()))
    q = float(SS.t.ppf(0.5 + level / 2, G - 1))
    return {"estimate": est, "G": G, "se": se, "lo": est - q * se, "hi": est + q * se, "df": G - 1}


def seed_rows(F: pd.DataFrame) -> pd.DataFrame:
    """Per seed: median |y|, and per arm MAE, sign agreement (zero prediction = disagreement unless y = 0) and the
    share of exactly-zero predictions."""
    out = {}
    for s, g in F.groupby("split_seed"):
        y = g.y.to_numpy(float)
        d = {"medabs": float(np.median(np.abs(y)))}
        for a in ARMS:
            p = g[f"p_{a}"].to_numpy(float)
            d[f"mae_{a}"] = float(np.abs(p - y).mean())
            d[f"agree_{a}"] = float((np.sign(p) == np.sign(y)).mean())
            d[f"zero_{a}"] = float((p == 0).mean())
        out[int(s)] = d
    return pd.DataFrame(out).T


def coverage(F: pd.DataFrame) -> dict:
    """Cluster coverage: clusters present, clusters with >= MIN_ROWS rows per seed, largest share, Kish over clusters."""
    c = F.groupby("cluster_cpg").size().astype(float) / len(SEEDS)
    if c.empty:
        return {"clusters": 0, "clusters_ge10": 0, "largest_cluster": "", "largest_share": np.nan, "kish_clusters": np.nan}
    n = c.to_numpy(float)
    return {"clusters": int((n > 0).sum()), "clusters_ge10": int((n >= MIN_ROWS).sum()), "largest_cluster": str(c.idxmax()),
            "largest_share": float(n.max() / n.sum()), "kish_clusters": float(n.sum() ** 2 / (n ** 2).sum())}


def level_block(F: pd.DataFrame, Fs: pd.DataFrame, *, lopo_on: bool = True) -> dict:
    """Every per-frame statistic of the E-W table on identical rows (Fs: the strict rows of the same variant, for
    the multipliers)."""
    if F.empty:
        return {"rows_per_seed": 0.0}
    a = sums(F)[0][0]
    a_s = sums(Fs)[0][0] if len(Fs) else np.full_like(a, np.nan)
    out = {"rows_per_seed": float(a[:, J["one"]].mean()), "cells": int(F.cell_id.nunique()),
           "publications": int(F.publication.nunique()), **coverage(F)}
    for arm in ARMS:
        out[f"R2_{arm}"] = smean(r2_ps(a, arm))
    out["R2_B1_per_seed"], out["R2_LINE_per_seed"] = _span(r2_ps(a, "B1")), _span(r2_ps(a, "LINE"))
    for ref in COMPARATORS:
        sk = skill_ps(a, "B1", ref)
        out[f"skill_vs_{ref}"], out[f"skill_vs_{ref}_per_seed"] = smean(sk), _span(sk)
        out[f"skill_vs_{ref}_seeds_pos"] = int((sk > 0).sum())
        out[f"dR2_B1_minus_{ref}"] = smean(delta_ps(a, "B1", ref))
    for arm in ("B1", "LINE", "B0"):
        out[f"MSE_{arm}"] = smean(mse_ps(a, arm))
    out["var_y"] = smean(var_ps(a))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["MSE_mult_B1"] = smean(mse_ps(a, "B1") / mse_ps(a_s, "B1"))
        out["var_mult"] = smean(var_ps(a) / var_ps(a_s))
    sr = seed_rows(F)
    out["median_abs_y"] = float(sr.medabs.mean())
    for arm in ("B1", "LINE", "B0"):
        out[f"MAE_{arm}"] = float(sr[f"mae_{arm}"].mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"MAE_rel_median_{arm}"] = smean(sr[f"mae_{arm}"] / sr.medabs)
        out[f"sign_agree_{arm}"] = float(sr[f"agree_{arm}"].mean())
        out[f"zero_pred_share_{arm}"] = float(sr[f"zero_{arm}"].mean())
    for ref in COMPARATORS:
        us = CM.unit_stats(F, "p_B1", f"p_{ref}", "cluster_cpg")
        out[f"units_WLT_vs_{ref}"] = f"{int(us.win.sum())}/{int(us.loss.sum())}/{int(us.tie.sum())}"
        out[f"top3_unit_share_vs_{ref}"] = CM.top3_share(us.dSSE.to_numpy(float))
    if lopo_on:
        for nm, fn in (("R2_B1", lambda x: smean(r2_ps(x, "B1"))), ("skill_vs_LINE", lambda x: smean(skill_ps(x, "B1", "LINE"))),
                       ("skill_vs_B0", lambda x: smean(skill_ps(x, "B1", "B0")))):
            lp = lopo(F, fn)
            out.update({f"lopo_{nm}_lo": lp["lo"], f"lopo_{nm}_hi": lp["hi"], f"lopo_{nm}_min": lp["min"],
                        f"lopo_{nm}_min_by": lp["min_by"], f"lopo_{nm}_max": lp["max"], f"lopo_{nm}_max_by": lp["max_by"],
                        "lopo_P": lp["P"]})
    return out


# ------------------------------------------------------------------------------------------------------------
# frames and labels
# ------------------------------------------------------------------------------------------------------------
def label_for(variant: str = "all", design: str = "BPG", extra: str = "", *, ew: bool = True) -> str:
    """The section-9.1 label of one number (data variant / design), with the E-W estimand note unless ew=False."""
    parts = [SEEN]
    if variant == "noC5":
        parts.append("SENSITIVITY (C5-BTBP out)")
    elif variant == "noC5cell":
        parts.append(f"SENSITIVITY (cell {paths.C5_CELL} out)")
    elif variant in DUP_VARIANTS:
        parts.append(f"SENSITIVITY (duplicates: {variant})")
    if design == "BP":
        parts.append("SENSITIVITY (design BP)")
    if ew:
        parts.append("E-W (wider pairs; never pooled with strict pairs, never next to E-A's 0.452)")
    if extra:
        parts.append(extra)
    return "; ".join(parts)


def duplicate_sets() -> dict:
    """noV2: round-3 V2-1/2/3 cells (r5.population.v2drop_set); noBITDUP: the second cell of every Stage-0 flagged
    bit-identity link (results/stage0/dupscan_links.csv, flag_bit)."""
    from . import population as PO
    out = {"noV2": set(map(str, PO.v2drop_set()))}
    p = paths.RES / "stage0" / "dupscan_links.csv"
    if p.exists():
        d = pd.read_csv(p)
        out["noBITDUP"] = set(d[d.flag_bit.astype(str).str.lower() == "true"].cell_b.astype(str))
    return out


def variant(F: pd.DataFrame, v: str, dup: dict | None = None) -> pd.DataFrame:
    if v == "all":
        return F
    if v == "noC5":
        return F[F.publication != paths.C5]
    if v == "noC5cell":
        return F[F.cell_id != paths.C5_CELL]
    if v in DUP_VARIANTS:
        return F[~F.cell_id.isin((dup or {}).get(v, set()))]
    raise KeyError(v)


def class_masks(W: pd.DataFrame) -> dict:
    """Pooled frames (each its own row; never pooled with STRICT)."""
    return {"dZ2": W.dZ == 2, "dZ2 without Nd/Sm": (W.dZ == 2) & (W.pair != "Nd/Sm"), "dZ3": W.dZ == 3, "dZ4": W.dZ == 4,
            "dZ3-4": W.dZ.isin([3, 4]), "Nd/Sm": W.pair == "Nd/Sm", "feed NdFeB": W.ndfeb.astype(bool),
            "feed set": W.feed.astype(bool), "light (dZ2-4)": (W.region == "light") & W.dZ.isin(DZ_SET),
            "light|middle (dZ2-4)": (W.region == "light|middle") & W.dZ.isin(DZ_SET),
            "middle (dZ2-4)": (W.region == "middle") & W.dZ.isin(DZ_SET),
            "middle|heavy (dZ2-4)": (W.region == "middle|heavy") & W.dZ.isin(DZ_SET),
            "heavy (dZ2-4; NON-CLAIM)": (W.region == "heavy") & W.dZ.isin(DZ_SET)}


def frame_of(W: pd.DataFrame, name: str) -> pd.DataFrame:
    if name in class_masks(W):
        return W[class_masks(W)[name]]
    return W[W.pair == name]


def _meta(name: str) -> dict:
    a, b = (IDX[x] for x in name.split("/"))
    dz = int(ZV[b] - ZV[a])
    return {"pair": name, "A": LANTHANIDES[a], "B": LANTHANIDES[b], "dZ": dz, "cut_class": cut_class(dz),
            "region": region(a, b), "feed": name in FEED_NAMES, "ndfeb": name in NDFEB_NAMES,
            "heavy_nonclaim": region(a, b) == "heavy"}


def _pair_label(name: str, v: str, design: str) -> str:
    m = _meta(name)
    extra = ["HIST_D7 columns DIAGNOSTIC"]
    if m["heavy_nonclaim"]:
        extra.append("NON-CLAIM (heavy cut; linked to Tm/Lu reads)")
    if name == "Nd/Sm":
        extra.append("light|middle cut")
    return label_for(v, design, "; ".join(extra))


# ------------------------------------------------------------------------------------------------------------
# tables
# ------------------------------------------------------------------------------------------------------------
def cut_table(W, S, design, variants=VARIANTS, lopo_on=True) -> pd.DataFrame:
    """One row per cut pair (33) and variant."""
    rows = []
    for v in variants:
        Wv, Sv = variant(W, v), variant(S, v)
        for name in PAIR_NAMES:
            F = Wv[Wv.pair == name]
            rows.append({"design": design, "estimand": ESTIMAND, "variant": v, **_meta(name),
                         **level_block(F, Sv, lopo_on=lopo_on and v != "noC5cell"),
                         "label": _pair_label(name, v, design)})
    return pd.DataFrame(rows)


def class_table(W, S, cuts: pd.DataFrame, design, variants=VARIANTS, lopo_on=True) -> pd.DataFrame:
    """Pooled frames plus the macro average over the frame's pair types (per-pair R^2 from the cut table)."""
    rows = []
    for v in variants:
        Wv, Sv = variant(W, v), variant(S, v)
        cv = cuts[cuts.variant == v]
        for name, mask in class_masks(Wv).items():
            F = Wv[mask]
            pts = sorted(F.pair.unique())
            cp = cv[cv.pair.isin(pts)]
            extra = "HIST_D7 columns DIAGNOSTIC; macro = equal-weight mean of per-pair R2 over the frame's pair types"
            if "NON-CLAIM" in name:
                extra += "; NON-CLAIM (heavy cuts)"
            rows.append({"design": design, "estimand": ESTIMAND, "variant": v, "frame": name, "pair_types": len(pts),
                         "pairs": ",".join(pts), **level_block(F, Sv, lopo_on=lopo_on and v != "noC5cell"),
                         **{f"macro_R2_{a}": smean(cp[f"R2_{a}"]) if len(cp) else np.nan for a in ("B1", "B0", "LINE")},
                         "label": label_for(v, design, extra)})
    return pd.DataFrame(rows)


def strict_reference(S, design, variants=VARIANTS) -> pd.DataFrame:
    """E-A rows as multiplier denominators and the strict skill vs LINE / B0 only (no E-A level is reported here)."""
    rows = []
    for v in variants:
        a = sums(variant(S, v))[0][0]
        rows.append({"design": design, "variant": v, "frame": "STRICT (E-A rows)", "rows_per_seed": float(a[:, 0].mean()),
                     "MSE_B1": smean(mse_ps(a, "B1")), "var_y": smean(var_ps(a)),
                     "skill_vs_LINE": smean(skill_ps(a, "B1", "LINE")), "skill_vs_B0": smean(skill_ps(a, "B1", "B0")),
                     "skill_vs_LINE_per_seed": _span(skill_ps(a, "B1", "LINE")),
                     "label": label_for(v, design, "E-A rows: multiplier denominators and strict skill only; "
                                                   "no E-A level reported here", ew=False)})
    return pd.DataFrame(rows)


def substitution_trap(W, S, design) -> pd.DataFrame:
    """'strict + Nd/Sm' pooled: shown only as the labelled substitution-trap row (brief 3.3; IR M13)."""
    F = pd.concat([S, W[W.pair == "Nd/Sm"]], ignore_index=True)
    a = sums(F)[0][0]
    return pd.DataFrame([{"design": design, "frame": "SUBSTITUTION TRAP: strict pairs pooled with Nd/Sm",
                          "rows_per_seed": float(a[:, 0].mean()), **{f"R2_{x}": smean(r2_ps(a, x)) for x in ARMS},
                          "R2_B1_per_seed": _span(r2_ps(a, "B1")),
                          "label": SEEN + "; SUBSTITUTION TRAP (strict pairs pooled with Nd/Sm): never an E-A or an E-W "
                                          "value, never next to 0.452" + ("; SENSITIVITY (design BP)" if design == "BP" else "")}])


def sign_table(W, design, variants=("all", "noC5")) -> pd.DataFrame:
    """Sign agreement per frame (pairs and pooled frames) x |y| bin x arm; zero predictions counted."""
    rows = []
    for v in variants:
        Wv = variant(W, v)
        Wv = Wv.assign(_bin=np.asarray(SIGN_BINS, dtype=object)[np.digitize(np.abs(Wv.y.to_numpy(float)), SIGN_EDGES)])
        frames = {**{n: Wv[Wv.pair == n] for n in PAIR_NAMES}, **{n: Wv[m] for n, m in class_masks(Wv).items()}}
        for name, F in frames.items():
            for b in SIGN_BINS:
                G = F[F._bin == b]
                if G.empty:
                    continue
                sr = seed_rows(G)
                rows.append({"design": design, "variant": v, "frame": name, "abs_y_bin": b,
                             "rows_per_seed": len(G) / len(SEEDS), "y_zero_share": float((G.y == 0).mean()),
                             **{f"sign_agree_{a}": float(sr[f"agree_{a}"].mean()) for a in ARMS},
                             **{f"zero_pred_share_{a}": float(sr[f"zero_{a}"].mean()) for a in ARMS},
                             "label": label_for(v, design, "sign agreement, zero predictions counted as disagreement; "
                                                           "HIST_D7 columns DIAGNOSTIC")})
    return pd.DataFrame(rows)


def heavy_table(W, S, design, variants=("all", "noC5")) -> pd.DataFrame:
    """Heavy cuts (both ends in Ho..Lu) as explicit non-claims, by whether Tm or Lu is read."""
    rows = []
    for v in variants:
        Wv, Sv = variant(W, v), variant(S, v)
        H = Wv[Wv.heavy_nonclaim.astype(bool)]
        tl = H.tm_read.astype(bool) | H.lu_read.astype(bool)
        for name in ["heavy (pooled)"] + sorted(H.pair.unique(), key=lambda p: PAIR_NAMES.index(p)):
            Hn = H if name == "heavy (pooled)" else H[H.pair == name]
            tln = tl.loc[Hn.index]
            for st, F in (("all", Hn), ("Tm or Lu read", Hn[tln]), ("neither Tm nor Lu read", Hn[~tln]),
                          ("Tm read", Hn[Hn.tm_read.astype(bool)]), ("Lu read", Hn[Hn.lu_read.astype(bool)])):
                b = level_block(F, Sv, lopo_on=False)
                rows.append({"design": design, "variant": v, "frame": name, "stratum": st,
                             "tm_or_lu_read_share": float(tln.mean()) if len(Hn) else np.nan,
                             **{k: b.get(k, np.nan) for k in ("rows_per_seed", "cells", "R2_B1", "R2_LINE", "R2_B0",
                                                              "skill_vs_LINE", "skill_vs_B0", "MAE_B1",
                                                              "MAE_rel_median_B1", "sign_agree_B1", "zero_pred_share_B1",
                                                              "clusters")},
                             "label": label_for(v, design, "NON-CLAIM (heavy cut; linked to Tm/Lu reads)")})
    return pd.DataFrame(rows)


def giant_cluster(S: pd.DataFrame) -> str:
    c = S.groupby("cluster_cpg").size()
    return str(c.idxmax()) if len(c) else ""


def ndsm_clusters(W: pd.DataFrame, *, giant: str = GIANT, min_rows: float = MIN_ROWS, drop_giant: bool = True):
    """Per cluster_cpg: Nd/Sm rows per seed, within-cluster R^2 of B1 and LINE (seed means) and B1 - LINE =
    seed mean of (SSE_LINE - SSE_B1) / SST_c; ``qualifies`` = >= min_rows rows per seed and not the giant cluster."""
    F = W[W.pair == "Nd/Sm"]
    if F.empty:
        return pd.DataFrame(columns=["cluster_cpg", "rows_per_seed", "R2_B1", "R2_LINE", "B1_minus_LINE", "qualifies"])
    A, groups = sums(F, "cluster_cpg")
    rows = []
    for i, c in enumerate(groups):
        a = A[i]
        d = smean(delta_ps(a, "B1", "LINE"))
        n = float(a[:, J["one"]].mean())
        rows.append({"cluster_cpg": c, "rows_per_seed": n, "publications": int(F[F.cluster_cpg == c].publication.nunique()),
                     "R2_B1": smean(r2_ps(a, "B1")), "R2_LINE": smean(r2_ps(a, "LINE")), "B1_minus_LINE": d,
                     "B1_minus_LINE_per_seed": _span(delta_ps(a, "B1", "LINE")), "giant": c == giant,
                     "qualifies": bool(n >= min_rows and not (drop_giant and c == giant) and np.isfinite(d))})
    return pd.DataFrame(rows)


def ndsm_rule(cl: pd.DataFrame) -> dict:
    """The pre-declared report rule on the qualifying clusters."""
    q = cl[cl.qualifies] if len(cl) else cl
    jk = jackknife_mean(q.B1_minus_LINE.to_numpy(float) if len(q) else [])
    retired = bool(jk["G"] < 2 or not (jk["estimate"] > RETIRE_AT) or not (jk["lo"] > 0))
    why = ("fewer than 2 qualifying clusters" if jk["G"] < 2 else
           "; ".join(x for x in (f"contrast {jk['estimate']:+.4f} <= +{RETIRE_AT}" if not jk["estimate"] > RETIRE_AT else "",
                                 f"lower bound {jk['lo']:+.4f} <= 0" if not jk["lo"] > 0 else "") if x) or "rule not met")
    return {**jk, "clusters": ",".join(q.cluster_cpg) if len(q) else "", "retired": retired,
            "outcome": ("RETIRED: 'interpolation beats slope at the light|middle cut' (" + why + "); narrowed to 'a line "
                        "through 4 reads suffices at Nd/Sm outside the giant cluster'") if retired else
            "NOT RETIRED: contrast > +0.05 and its jackknife-t lower bound > 0 (EXPLORATORY seen; open contrast)"}


def ndsm_contrast(W, S, design) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Primary pre-declared contrast plus labelled sensitivities and the pooled Nd/Sm row (seen +0.084)."""
    rows, cls = [], []
    specs = (("PRIMARY (pre-declared)", "all", True, MIN_ROWS), ("SENSITIVITY: giant cluster kept", "all", False, MIN_ROWS),
             ("SENSITIVITY: every cluster with >= 2 rows per seed", "all", True, 2.0),
             ("SENSITIVITY: C5-BTBP out", "noC5", True, MIN_ROWS))
    for name, v, drop, mr in specs:
        cl = ndsm_clusters(variant(W, v), min_rows=mr, drop_giant=drop)
        r = ndsm_rule(cl)
        # label_for adds the E-W marker and, under BP, 'SENSITIVITY (design BP)' once; the PRIMARY row also carries
        # the label DECISION_LOG registers for rule 1.7-R01
        lab = label_for(v, design, "pre-declared open contrast (descriptive report rule; leave-one-cluster-out "
                                   "jackknife-t 95 %); CONSISTENCY CHECK (registered label of rule 1.7-R01)"
                        if name.startswith("PRIMARY") else name)
        rows.append({"design": design, "spec": name, "variant": v, "drop_giant": drop, "min_rows_per_seed": mr,
                     "giant": GIANT, "estimate": r["estimate"], "G": r["G"], "P": np.nan, "jk_se": r["se"],
                     "lo95": r["lo"], "hi95": r["hi"], "df": r["df"],
                     "interval": "leave-one-cluster-out jackknife-t over the G qualifying clusters (t(G - 1))",
                     "clusters": r["clusters"], "retire_threshold": RETIRE_AT, "retired": r["retired"],
                     "outcome": r["outcome"], "label": lab})
        cls.append(cl.assign(design=design, spec=name, label=lab))
    F = W[W.pair == "Nd/Sm"]
    for name, G in (("pooled Nd/Sm (all clusters; seen +0.084)", F), ("pooled Nd/Sm outside the giant cluster",
                                                                      F[F.cluster_cpg != GIANT])):
        if G.empty:
            continue
        a = sums(G)[0][0]
        lp = lopo(G, lambda x: smean(delta_ps(x, "B1", "LINE")))
        rows.append({"design": design, "spec": name, "variant": "all", "drop_giant": "outside" in name,
                     "min_rows_per_seed": np.nan, "giant": GIANT, "estimate": smean(delta_ps(a, "B1", "LINE")),
                     "G": int(G.cluster_cpg.nunique()), "P": lp["P"], "jk_se": lp["se"], "lo95": lp["lo"],
                     "hi95": lp["hi"], "df": lp["P"] - 1,
                     "interval": "leave-one-publication-out jackknife-t over the P publications (t(P - 1)); G = "
                                 "clusters present (not the interval's units)",
                     "clusters": "", "retire_threshold": np.nan, "retired": np.nan,
                     "outcome": f"pooled; interval = leave-one-publication-out jackknife-t (P = {lp['P']}); "
                                f"LOPO range [{lp['min']:+.4f} ({lp['min_by']}), {lp['max']:+.4f} ({lp['max_by']})]",
                     "label": label_for("all", design, "pooled row beside the pre-declared contrast (not the rule)")})
    return pd.DataFrame(rows), pd.concat(cls, ignore_index=True)


def unit_tables(W, design, variants=("all", "noC5")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """cluster_cpg unit tables of B1 vs LINE and B1 vs B0 on the main frames, with the section-4.2 descriptive tests
    (DIAGNOSTIC; exact sign-flip at G <= 13)."""
    U, T = [], []
    for v in variants:
        Wv = variant(W, v)
        for fr in MAIN_FRAMES:
            F = frame_of(Wv, fr)
            if F.empty:
                continue
            for ref in COMPARATORS:
                us = CM.unit_stats(F, "p_B1", f"p_{ref}", "cluster_cpg")
                lab = label_for(v, design, "DIAGNOSTIC (unit table / descriptive tests; no p-value is significant)")
                U.append(us.reset_index().assign(design=design, variant=v, frame=fr, contrast=f"B1 vs {ref}", label=lab))
                key = f"EW|{design}|{v}|{fr}|B1-{ref}"
                tt = CM.unit_tests(us, key)
                T.append({"design": design, "variant": v, "frame": fr, "contrast": f"B1 vs {ref}", "G": tt["G"],
                          "W/L/T": tt["W/L/T"], "SF_POOL_p": tt["SF_POOL"]["p"], "SF_POOL_exact": tt["SF_POOL"]["exact"],
                          "SIGN_p": tt["SIGN"]["p"], "SF_LOG_p": tt["SF_LOG"]["p"] if tt["SF_LOG"] else np.nan,
                          "IM_REL_p": tt["IM_REL"]["p"], "IM_REL_mean": tt["IM_REL"]["mean"],
                          "top3_unit_share": CM.top3_share(us.dSSE.to_numpy(float)),
                          "top3_unit_share_abs": CM.top3_share(np.abs(us.dSSE.to_numpy(float))), "label": lab})
    return pd.concat(U, ignore_index=True) if U else pd.DataFrame(), pd.DataFrame(T)


def duplicate_table(W, S, design, dup: dict) -> pd.DataFrame:
    rows = []
    for v in DUP_VARIANTS:
        if v not in dup:
            continue
        Wv, Sv = variant(W, v, dup), variant(S, v, dup)
        for fr in MAIN_FRAMES:
            F0, F = frame_of(W, fr), frame_of(Wv, fr)
            b = level_block(F, Sv, lopo_on=False)
            rows.append({"design": design, "variant": v, "frame": fr, "dropped_cells": int(F0.cell_id.nunique() - F.cell_id.nunique()),
                         **{k: b.get(k, np.nan) for k in ("rows_per_seed", "R2_B1", "R2_LINE", "R2_B0", "skill_vs_LINE",
                                                          "skill_vs_B0", "R2_B1_per_seed")},
                         "label": label_for(v, design)})
    return pd.DataFrame(rows)


def population_table(R: pd.DataFrame, design: str) -> pd.DataFrame:
    """Section-9.3 population of one design (counts only, no score): the scored WIDE rows (cells, publications,
    cluster_cpg units, rows per seed) and, on its own labelled line, the STRICT E-A rows (reproduction target and
    multiplier denominators only)."""
    fam = R.family.astype(str)
    W, S = R[fam == "WIDE"], R[fam == "STRICT"]
    gone = set(S.cell_id.astype(str)) - set(W.cell_id.astype(str))
    gpub = sorted(set(S.publication.astype(str)) - set(W.publication.astype(str)))
    rows = []
    for f, F, role, lab in (
            ("WIDE", W, f"scored E-W rows (frozen cut pairs, both ends measured, at least one end unread); C3 cells "
                        f"without a WIDE row: {len(gone)} (publications without one: {', '.join(gpub) or 'none'})",
             label_for("all", design, "population counts (no score)")),
            ("STRICT", S, "E-A rows (E-A's C3 cells and read sets): reproduction target and MSE / var(y) multiplier "
                          "denominators only; not scored by E-W",
             label_for("all", design, "STRICT = E-A rows (reproduction / denominators only); population counts "
                                      "(no score)", ew=False))):
        rows.append({"design": design, "family": f, **CM.population(F.assign(cluster_cpg=F.cluster_cpg.astype(str))),
                     "rows_all_seeds": int(len(F)), "role": role, "label": lab})
    return pd.DataFrame(rows)


def seen_table(T: dict, design: str) -> pd.DataFrame:
    """IR M13 seen values beside the recomputed ones (B1 = fold beta; HIST_D7 = the fixed -0.072 IR used)."""
    cl = T["classes"][T["classes"].variant == "all"].set_index("frame")
    cu = T["cuts"][T["cuts"].variant == "all"].set_index("pair")
    sr = T["strict_reference"][T["strict_reference"].variant == "all"].iloc[0]
    tr = T["substitution_trap"].iloc[0]
    rows = []
    for fr, q, want in SEEN_IR:
        if fr == "STRICT":
            got, got_d7 = float(sr[q]), np.nan
        elif fr == "TRAP":
            got, got_d7 = float(tr["R2_B1"]), float(tr["R2_HIST_D7"])
        elif fr == "NDSM_OUTSIDE_GIANT":
            nd = T["ndsm_contrast"]
            r = nd[nd.spec == "pooled Nd/Sm outside the giant cluster"]
            got, got_d7 = (float(r.estimate.iloc[0]) if len(r) else np.nan), np.nan
        else:
            src = cl.loc[fr] if fr in cl.index else cu.loc[fr]
            got = float(src[q])
            d7 = q.replace("B1", "HIST_D7") if q in ("R2_B1",) else None
            got_d7 = float(src[d7]) if d7 else np.nan
        rows.append({"design": design, "frame": fr, "quantity": q, "seen_IR_M13": str(want), "recomputed_B1": got,
                     "recomputed_HIST_D7": got_d7, "label": "CONSISTENCY CHECK (seen value, IR M13 / brief 6.7; "
                     "recomputed; not a test); " + SEEN + ("; SENSITIVITY (design BP)" if design == "BP" else "")})
    return pd.DataFrame(rows)


def analyse(R: pd.DataFrame, design: str, *, lopo_on: bool = True, log=print) -> dict:
    """Every table of the item for one design."""
    t0 = time.time()
    P = prepare(R)
    S, W = P[P.family == "STRICT"], P[P.family == "WIDE"]
    g = giant_cluster(S)
    if len(S) and g != GIANT:
        raise AssertionError(f"giant cluster {g} != frozen {GIANT}")
    dup = duplicate_sets()
    T = {"cuts": cut_table(W, S, design, lopo_on=lopo_on)}
    T["classes"] = class_table(W, S, T["cuts"], design, lopo_on=lopo_on)
    T["strict_reference"] = strict_reference(S, design)
    T["substitution_trap"] = substitution_trap(W, S, design)
    T["signs"] = sign_table(W, design)
    T["heavy"] = heavy_table(W, S, design)
    T["ndsm_contrast"], T["ndsm_clusters"] = ndsm_contrast(W, S, design)
    T["units"], T["unit_tests"] = unit_tables(W, design)
    T["duplicates"] = duplicate_table(W, S, design, dup)
    T["seen"] = seen_table(T, design)
    log(f"    analysis {design}: {len(T)} tables ({time.time() - t0:.0f}s)")
    return T


def summary(T: dict, design: str, checks: dict, repro: dict, seconds: float) -> dict:
    cl = T["classes"]

    def pick(fr, v="all"):
        r = cl[(cl.frame == fr) & (cl.variant == v)]
        return r.iloc[0][["rows_per_seed", "R2_B1", "R2_B0", "R2_LINE", "skill_vs_LINE", "skill_vs_B0", "MSE_mult_B1",
                          "var_mult", "macro_R2_B1", "R2_B1_per_seed", "label"]].to_dict() if len(r) else None
    nd = T["ndsm_contrast"]
    prim = nd[nd.spec.str.startswith("PRIMARY")].iloc[0].to_dict()
    return {"label": SEEN + " (E-W; never pooled with strict pairs, never next to E-A's 0.452)", "design": design,
            "seconds": seconds, "repro": repro, "checks": checks,
            "frames": {fr: pick(fr) for fr in MAIN_FRAMES}, "frames_noC5": {fr: pick(fr, "noC5") for fr in MAIN_FRAMES},
            "ndsm_primary": prim, "ndsm_sensitivities": nd[~nd.spec.str.startswith("PRIMARY")].to_dict("records"),
            "heavy_nonclaim": T["heavy"][(T["heavy"].variant == "all") & (T["heavy"].frame == "heavy (pooled)")].to_dict("records")}


# ------------------------------------------------------------------------------------------------------------
# the frozen item specification (results/EW/ITEM_SPEC.md)
# ------------------------------------------------------------------------------------------------------------
def item_spec_text() -> str:
    def lf(p):
        try:
            return paths.sha256_text_lf(p)
        except OSError:
            return "missing"
    by_dz = {d: [n for (a, b), n in zip(PAIRS, PAIR_NAMES) if ZV[b] - ZV[a] == d] for d in sorted({int(ZV[b] - ZV[a])
                                                                                                  for a, b in PAIRS})}
    lines = [
        "# Item 1.7 specification: E-W wider-pair table (M13)",
        "",
        "Frozen at the Stage-0 seal, before the item's first post-seal run (brief 1.2: Stage-0 seal -> item code seal -> "
        "reproduction run labelled TECHNICAL SMOKE -> arms; the pre-seal --smoke runs are PRE-SEAL ENGINEERING CHECKS, "
        "PROTOCOL 4 and 10).  Every in-corpus number is EXPLORATORY (seen); nothing here is a test.  E-W is its own "
        "estimand: never pooled with the strict-adjacent pairs and never placed next to E-A's 0.452 (brief 3.3).  Code: "
        "`r5/ew.py`, `scripts/s1_ew.py`, `tests/test_ew.py`.  Sources: brief 6.7, 6.0, 6.10, 8, 9; IR M13 "
        f"(`ideas/IDEAS_RANKED.md` LF SHA-256 {lf(paths.DESIGN / 'ideas' / 'IDEAS_RANKED.md')}; `ideas/critiques.json` "
        f"LF SHA-256 {lf(paths.DESIGN / 'ideas' / 'critiques.json')}).  No design-folder script is copied or run.",
        "",
        "## Rows",
        "",
        "- Every C3 cell instance (k_max >= 3) of the 50 round-4 dev folds (5 seeds x 5 folds x BPG / BP) at E-A's read "
        "set: four reads, the union of the cached `cell['seqs']['DOPT_Q'][:3]` (g21.cache.load; brief 3.3).",
        f"- WIDE rows: the frozen cut-pair set ({len(PAIRS)} pair types), A lighter, dZ = Z_B - Z_A (Pm absent: Nd/Sm "
        "has dZ 2): " + "; ".join(f"dZ {d}: {', '.join(v)}" for d, v in by_dz.items()) + ".  Both ends measured, at "
        "least one end unread (a pair with both ends read is known and never scored; counted in `checks`).",
        f"- Feed-defined pair set: {', '.join(FEED_NAMES)} (NdFeB: {', '.join(NDFEB_NAMES)}).  Note: brief 6.7 writes "
        "'dZ 5-6' for the NdFeB pairs; by atomic number Nd/Tb = 5, Nd/Dy = 6 and Pr/Dy = 7.",
        "- STRICT rows: the cell's strict-adjacent queries (E-A's 1,802 rows per seed), used only for the reproduction "
        "target and as the denominators of the MSE / var(y) multipliers; no E-A level is reported by this item.",
        "- Regions: light La-Nd, middle Sm-Dy, heavy Ho-Lu; a pair spanning two is 'light|middle' etc.  Heavy cuts = both "
        "ends in Ho-Lu: explicit NON-CLAIMS, reported by whether Tm or Lu is read.",
        "",
        "## Arms (identical rows; level curves from the read metals only)",
        "",
        "- B1: `r4.refs.predict_levels` with the fold's stored round-4 beta (`round4_20261022/results/REF/fitted_<design>.json`; "
        "no refit).  B0: the same path with beta = 0 (= PWL_R).  LINE: the slope-only line through the reads "
        "(`gen20k.adapters.naive_line` on the gen20 support; round 4's LINE column).  HIST_D7: the fixed -0.072 "
        "(DIAGNOSTIC; not fold-independent; IR M13's seen numbers used it).",
        "- Comparators of B1: LINE and B0.  No arm is fitted or selected in this item, so fit and cache poisoning do not "
        "apply; no new cache is written.",
        "",
        "## Per-cut table (one row per cut pair and variant) and pooled frames",
        "",
        "- R2 per arm (pooled inside each seed, then the seed mean; never clipped), per-seed R2 of B1 and LINE; skill "
        "against LINE and against B0 = seed mean of 1 - SSE_B1 / SSE_ref (and dR2 = (SSE_ref - SSE_B1) / SST); MSE and "
        "var(y) multipliers = seed means of the per-seed ratio to the STRICT rows of the same variant; MAE and MAE / "
        "median |y| (B1, LINE, B0); sign agreement (zero prediction = disagreement unless y = 0 exactly) overall and in "
        f"the |y| bins {', '.join(SIGN_BINS)}; zero-prediction shares; cluster coverage (clusters present, clusters with "
        f">= {MIN_ROWS} rows per seed, largest cluster and share, Kish over clusters); cluster_cpg unit W/L/T and top-3 "
        "unit share of dSSE vs LINE and vs B0; leave-one-publication-out (jackknife t(P - 1) 95 % interval, min / max "
        "with the publication) of R2_B1, skill vs LINE and skill vs B0.",
        "- Pooled frames (each its own row): " + ", ".join(class_masks(pd.DataFrame({"dZ": [], "pair": [], "ndfeb": [],
                                                                                      "feed": [], "region": []}))) +
        "; with the macro average = equal-weight mean of the per-pair R2 over the frame's pair types.",
        "- 'Strict + Nd/Sm' pooled appears only as the labelled SUBSTITUTION TRAP row (its own file).",
        "",
        "## The pre-declared open contrast (report rule)",
        "",
        f"- Nd/Sm B1 - LINE in R2 units: inside each cluster_cpg cluster, seed mean of (SSE_LINE - SSE_B1) / SST_c; "
        f"clusters with >= {MIN_ROWS} Nd/Sm rows per seed (seed mean), the giant cluster {GIANT} (64.9 % of E-A rows; "
        "asserted) removed; equal-weighted mean over those clusters; leave-one-cluster-out jackknife-t 95 % interval "
        "(t(G - 1)), stated once.",
        f"- Rule: 'Interpolation beats slope at the light|middle cut' is RETIRED if the contrast is <= +{RETIRE_AT} or its "
        "lower bound is <= 0, or fewer than 2 clusters qualify (expected: retired; narrowed to 'a line through 4 reads "
        "suffices at Nd/Sm outside the giant cluster').  BPG decides; BP is a sensitivity.  Seen before the code: B1 - "
        "LINE +0.084 [-0.075, +0.243] pooled, +0.011 to +0.013 outside the giant cluster [IR M13].",
        "- Labelled sensitivities beside it: giant cluster kept; every cluster with >= 2 rows per seed; C5-BTBP out; the "
        "pooled Nd/Sm B1 - LINE with its leave-one-publication-out interval, all clusters and outside the giant.",
        "",
        "## Every item reports",
        "",
        f"- Variants {', '.join(VARIANTS)} (C5-BTBP = {paths.C5}; single-cell influence of {paths.C5_CELL}); duplicate "
        "sensitivities noV2 (round-3 V2-1/2/3 cells) and noBITDUP (second cell of every Stage-0 flagged bit-identity "
        "link); BPG (primary) and BP (sensitivity); per-seed values; the cluster_cpg unit table with SF_POOL / SIGN / "
        "SF_LOG / IM_REL (DIAGNOSTIC; r5.common); leave-one-publication-out; the top-3 unit share; the section-9.3 "
        "per-item table (`ITEM_TABLE.md`).",
        "- Seen-value table: IR M13's values beside the recomputed ones (CONSISTENCY CHECK, not a stop condition).",
        "- Labels: EXPLORATORY (seen) on every in-corpus number; SENSITIVITY (BP, C5-BTBP out, single cell out, "
        "duplicates); DIAGNOSTIC (HIST_D7 columns, unit tests); NON-CLAIM (heavy cuts); SUBSTITUTION TRAP; CONSISTENCY "
        "CHECK (seen table); TECHNICAL SMOKE (reproduction run).  No in-corpus p-value is called significant.",
        "",
        "## Reproduction target (brief 6.0; a mismatch stops the item)",
        "",
        f"- STRICT rows: B1 from the fold betas equals `round4_20261022/predictions/REF_<design>.parquet` within "
        f"{TARGETS['B1_vs_REF_max_abs']} on every row ({TARGETS['rows_per_design']:,} per design, 18,020 in all); y exact; "
        "read sets identical; B0, HIST_D7 and LINE differences reported.",
        "",
        "## Leakage (brief 8)",
        "",
        "- Isolation (`g21.evaluate.assert_test_isolation`) on every fold.  Support: exact B0 / B1 / HIST_D7 levels at "
        "every read metal; no scored pair with both ends read; A lighter, dZ = 1 for STRICT and >= 2 for WIDE; finite "
        "predictions; unique row keys.  Prediction intervention on every cell instance (every unread label replaced by "
        "N(0, 5) noise; all four arms' level curves and the read set bit-identical) with a block-Y peek positive control "
        "per fold that must be caught; the control goes through the real level function and comparator (a predictor "
        "reading the cell's whole DOPT_Q sequence must change when the unread metals change; s0-mechanics-6).  A sealed "
        "run needs full coverage: every cell instance intervened on and one caught peek per fold (s0-mechanics-5); "
        "`--no-intervention` / `--no-lopo` are smoke-only.  The seal tag and code digests are written next to every "
        "output; `--from-rows` refuses rows or checks of another seal or code (s0-mechanics-18).",
        "- The pre-seal `--smoke` replaces y and every prediction column by seeded noise before any table runs, so no "
        "real-label metric is computed (s0-mechanics-22).",
        "",
        "## RNG seeds",
        "",
        "- Intervention scrambles: `default_rng(r5.paths.seed_for('EW|intervention|<design>'))` (20261023 + crc32).  "
        "Unit sign-flip tests: exact over 2^G at G <= 13 (seed_for('<key>|SF_POOL') / '|SF_LOG' only if G > 13, key = "
        "'EW|<design>|<variant>|<frame>|B1-<ref>').  No other random procedure.",
        "",
        "## Command lines (run from D:\\ml_separator_gh)",
        "",
        "```",
        "set PYTHONIOENCODING=utf-8",
        "rem pre-seal engineering check on real data (no metric printed or written; allowed before the seal)",
        ".venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ew.py --smoke --design both",
        "rem after the seal: reproduction run (TECHNICAL SMOKE), then the item",
        ".venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ew.py --repro --design both",
        ".venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ew.py --design both",
        ".venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_ew.py -q "
        "-p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest",
        "```",
        "",
    ]
    return "\n".join(lines)

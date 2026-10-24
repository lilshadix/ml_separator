"""Round-5 shared statistics (CLAUDE_CODE_PROMPT.md sections 4.1, 4.2, 6.0, 9.2, 9.3).

* Rows: round-4 OOF parquets ``predictions/<cid>_<design>.parquet`` restricted as round 4's report/r4_decide.py
  ``load()`` does (k == 3, k_max >= 3, sorted by its KEY), with ``cluster_cpg`` from r4.metrics.add_clusters and
  ``pubchem`` (publication x chemotype, P1's E-CT unit).  A run counts only if its round-4 intervention (and, when
  present, identity) check passed, as ``r4_decide.run_valid``.
* Levels: pooled R^2 / MSE inside each seed, then the seed mean; balanced MAE = round 4's chemotype-macro MAE
  (r4.metrics -> r3.metrics.per_seed ``MAE_macro_chem``); relative pair-SSE = seed mean of 1 - SSE_cand / SSE_ref.
  Positive contrasts always favour the candidate.
* Per-unit statistics (section 9.2): seed means of n, SSE_ref, SSE_cand; dSSE = SSE_ref - SSE_cand;
  L = log(SSE_ref / SSE_cand); win = dSSE > 0, a tie an exact 0.
* Tests (section 4.2; one-sided, H1: the candidate is better): SF_POOL exact sign-flip by full enumeration when
  G <= 13, else B = 4,999 random flips, p = (1 + #{T* >= T}) / (1 + B), RNG seed paths.seed_for(key); SIGN exact
  binomial, ties dropped; IM t (Ibragimov-Mueller: one-sample t over unit estimates, df G - 1).
* Spread: leave-one-publication-out jackknife with a t(P - 1) interval (descriptive; anti-conservative under the
  clustering of publications); top-3 unit share; C5-BTBP in / out.
* ``per_item_table``: the section-9.3 table as markdown.
* ``run_record``: START / END lines of RUN_RECORD.md (brief 10) for every non-smoke Stage-1 run.

Nothing here reads a withheld seed or writes outside the round-5 folder.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as SS

from . import paths
from r4 import metrics as M4  # noqa: E402

KEY = ["split_seed", "fold", "cell_id", "A", "B"]          # report/r4_decide.py KEY
EXACT_MAX = 13                                             # 2^13 = 8,192 sign vectors (section 4.1 item 2)
B_FLIP = 4999
RUN_RECORD = paths.R5 / "RUN_RECORD.md"                    # append-only run log (UTC); tests point it elsewhere
LABELS = ("TECHNICAL SMOKE", "EXPLORATORY (seen)", "PARTITION STABILITY", "INDEPENDENT CONFIRMATION", "SENSITIVITY",
          "ORACLE", "DIAGNOSTIC", "CONSISTENCY CHECK")      # section 9.1


# ------------------------------------------------------------------------------------------------------------
# rows
# ------------------------------------------------------------------------------------------------------------
def run_valid(cid: str, design: str) -> bool:
    """Round 4's rule (r4_decide.run_valid): the intervention passed and, when present, the identity check passed."""
    iv = paths.R4RES / cid / f"intervention_{design}.json"
    idn = paths.R4RES / cid / f"identity_{design}.json"
    if not iv.exists() or json.loads(iv.read_text(encoding="utf-8")).get("passed") is not True:
        return False
    return not idn.exists() or json.loads(idn.read_text(encoding="utf-8")).get("passed") is True


def load_rows(design: str, cid: str = "REF", *, check_valid: bool = True) -> pd.DataFrame:
    """E-A rows of one round-4 run: C3 (k_max >= 3) at k = 3, sorted by KEY, with copy_group, cluster_cpg, pubchem."""
    p = paths.R4PRED / f"{cid}_{design}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    if check_valid and not run_valid(cid, design):
        raise RuntimeError(f"{cid}_{design}: round-4 intervention / identity check did not pass")
    t = pd.read_parquet(p)
    t = t[(t.k == 3) & (t.k_max >= 3)].sort_values(KEY).reset_index(drop=True)
    if t.duplicated(KEY).any():
        raise AssertionError(f"{cid}_{design}: duplicated row key")
    t = M4.add_clusters(t, None)                           # frame is unused by add_clusters
    t["pubchem"] = t.publication.astype(str) + "|" + t.chemotype.astype(str)
    return t


def population(t: pd.DataFrame, unit: str = "cluster_cpg") -> dict:
    """Cells, publications, units and rows per seed (section 9.3 'population')."""
    ns = t.split_seed.nunique()
    return {"cells": int(t.cell_id.nunique()), "publications": int(t.publication.nunique()),
            "unit": unit, "units": int(t[unit].nunique()), "rows_per_seed": float(len(t) / ns) if ns else 0.0,
            "seeds": int(ns)}


def c5_split(t: pd.DataFrame) -> dict:
    """C5-BTBP in / out (section 5.2) plus the single-cell influence row of c950dc2982e96b83."""
    pub, cell = t.publication.astype(str), t.cell_id.astype(str)
    return {"C5_in": t, "C5_out": t[pub != paths.C5], "C5_cell_out": t[cell != paths.C5_CELL]}


# ------------------------------------------------------------------------------------------------------------
# levels and pooled contrasts (seed means)
# ------------------------------------------------------------------------------------------------------------
def _sse(y, p) -> float:
    return float(((np.asarray(y, float) - np.asarray(p, float)) ** 2).sum())


def r2_seeds(t: pd.DataFrame, col: str) -> pd.Series:
    """Pooled R^2 inside each seed (r3.metrics._r2: NaN when undefined, never clipped)."""
    return pd.Series({s: float(M4.MT._r2(g.y, g[col])) for s, g in t.groupby("split_seed")}, dtype=float)


def r2_seedmean(t: pd.DataFrame, col: str) -> float:
    return float(r2_seeds(t, col).mean())


def mse_seedmean(t: pd.DataFrame, col: str) -> float:
    return float(np.mean([_sse(g.y, g[col]) / len(g) for _, g in t.groupby("split_seed")]))


def rel_sse_seeds(t: pd.DataFrame, cand: str, ref: str = "B1") -> pd.Series:
    return pd.Series({s: 1.0 - _sse(g.y, g[cand]) / _sse(g.y, g[ref]) for s, g in t.groupby("split_seed")}, dtype=float)


def rel_sse(t: pd.DataFrame, cand: str, ref: str = "B1") -> float:
    """Seed mean of the pooled relative pair-SSE reduction 1 - SSE_cand / SSE_ref (M35's one currency)."""
    return float(rel_sse_seeds(t, cand, ref).mean())


def bal_mae_seeds(t: pd.DataFrame, col: str) -> pd.Series:
    """Round 4's balanced MAE per seed: |error| -> extractant -> chemotype -> equal chemotype weights."""
    ps = M4.MT.per_seed(t, col)
    return pd.Series(ps.MAE_macro_chem.to_numpy(float), index=ps.split_seed.to_numpy())


def bal_mae(t: pd.DataFrame, col: str) -> float:
    return float(bal_mae_seeds(t, col).mean())


def dbal_mae(t: pd.DataFrame, cand: str, ref: str = "B1") -> float:
    """Seed mean of balanced MAE(ref) - balanced MAE(cand); M35's guard needs >= -0.001 against B1."""
    return float((bal_mae_seeds(t, ref) - bal_mae_seeds(t, cand)).mean())


def arm_table(t: pd.DataFrame, arms, ref: str = "B1") -> pd.DataFrame:
    """R^2, MSE, relative SSE vs ref, balanced MAE and per-seed R^2 for arms on identical rows (section 9.3)."""
    rows = []
    for a in arms:
        rs = r2_seeds(t, a)
        rows.append({"arm": a, "R2": float(rs.mean()), "MSE": mse_seedmean(t, a), f"relSSE_vs_{ref}": rel_sse(t, a, ref),
                     "MAE_balanced": bal_mae(t, a), "R2_per_seed": ";".join(f"{v:.4f}" for v in rs)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------
# per-unit statistics (section 9.2)
# ------------------------------------------------------------------------------------------------------------
def unit_stats(t: pd.DataFrame, cand: str, ref: str, unit: str) -> pd.DataFrame:
    """Per unit: seed means of n, SSE_ref, SSE_cand (per-seed sums, averaged over the seeds holding the unit);
    dSSE = SSE_ref - SSE_cand; L = log(SSE_ref / SSE_cand); win / loss / tie with a tie an exact 0."""
    g = pd.DataFrame({"u": t[unit].astype(str).to_numpy(), "s": t.split_seed.to_numpy(), "one": 1.0,
                      "er": (t[ref] - t.y).to_numpy(float) ** 2, "ec": (t[cand] - t.y).to_numpy(float) ** 2})
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


def flip_matrix(G: int, key: str | None = None, *, exact_max: int = EXACT_MAX, B: int = B_FLIP):
    """All 2^G sign vectors (first row = identity) when G <= exact_max, else B random +-1 rows from
    default_rng(paths.seed_for(key)).  Returns (F, exact, seed)."""
    if G == 0:
        return np.ones((1, 0)), True, None
    if G <= exact_max:
        return np.array(list(itertools.product((1.0, -1.0), repeat=G))), True, None
    if key is None:
        raise ValueError(f"G = {G} > {exact_max}: a random sign-flip needs a registry key for its seed")
    seed = paths.seed_for(key)
    return np.random.default_rng(seed).choice((-1.0, 1.0), size=(B, G)), False, seed


def sf_pool(values, key: str | None = None, *, exact_max: int = EXACT_MAX, B: int = B_FLIP) -> dict:
    """One-sided sign-flip test of H0: mean <= 0 on per-unit values (SF_POOL on dSSE; SF_LOG on L), T = sum.
    Exact: p = #{T* >= T} / 2^G over every sign vector (identity included).  Random (G > exact_max):
    p = (1 + #{T* >= T}) / (1 + B).  T* >= T uses a 1e-10 relative tolerance so exact ties count."""
    x = np.asarray(values, float)
    if not np.isfinite(x).all():
        raise ValueError("sf_pool: non-finite unit value")
    G = len(x)
    T = float(x.sum())
    F, exact, seed = flip_matrix(G, key, exact_max=exact_max, B=B)
    tol = 1e-10 * float(np.abs(x).sum())
    ge = int(sum(int(((F[i:i + 4096] @ x) >= T - tol).sum()) for i in range(0, len(F), 4096)))
    p = ge / len(F) if exact else (1.0 + ge) / (1.0 + len(F))
    return {"T": T, "G": G, "p": float(p), "exact": exact, "n_flips": int(len(F)), "n_ge": ge, "seed": seed}


def sign_test(wins) -> dict:
    """Exact one-sided binomial sign test on unit wins, ties dropped.  ``wins``: per-unit dSSE (or +1 / -1 / 0
    codes; > 0 win, < 0 loss, exact 0 tie) or a unit_stats frame.  A boolean vector cannot carry ties: refused."""
    if isinstance(wins, pd.DataFrame):
        W, L, T = int(wins["win"].sum()), int(wins["loss"].sum()), int(wins["tie"].sum())
    else:
        x = np.asarray(wins)
        if x.dtype == bool:
            raise TypeError("sign_test: pass dSSE or +1/-1/0 codes (a boolean win vector drops the ties silently)")
        x = x.astype(float)
        if not np.isfinite(x).all():
            raise ValueError("sign_test: non-finite unit value")
        W, L, T = int((x > 0).sum()), int((x < 0).sum()), int((x == 0).sum())
    n = W + L
    p = float(SS.binom.sf(W - 1, n, 0.5)) if n > 0 else 1.0
    return {"W": W, "L": L, "T": T, "n": n, "p": p}


def im_t(values) -> dict:
    """Ibragimov-Mueller t: one-sample one-sided t over the G unit estimates, df = G - 1 (0/0 -> p = 1)."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    G = len(x)
    if G < 2:
        return {"t": float("nan"), "df": max(G - 1, 0), "p": float("nan"), "mean": float(x.mean()) if G else float("nan"),
                "se": float("nan"), "G": G}
    se = float(x.std(ddof=1) / np.sqrt(G))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.float64(x.mean()) / np.float64(se)
    p = float(SS.t.sf(np.nan_to_num(t, nan=-np.inf), G - 1))
    return {"t": float(t), "df": G - 1, "p": p, "mean": float(x.mean()), "se": se, "G": G}


def unit_tests(us: pd.DataFrame, key: str) -> dict:
    """Section 4.2 tests on one unit_stats frame: SF_POOL (pooled, primary), SIGN (win share, primary),
    SF_LOG and IM_REL (equal-weight breadth, diagnostics).  ``key`` seeds the random flips when G > 13."""
    return {"G": int(len(us)), "W/L/T": f"{int(us.win.sum())}/{int(us.loss.sum())}/{int(us.tie.sum())}",
            "SF_POOL": sf_pool(us.dSSE, key + "|SF_POOL"), "SIGN": sign_test(us),
            "SF_LOG": sf_pool(us.L, key + "|SF_LOG") if np.isfinite(us.L).all() else None, "IM_REL": im_t(us.rel)}


# ------------------------------------------------------------------------------------------------------------
# spread
# ------------------------------------------------------------------------------------------------------------
def lopo(t: pd.DataFrame, fn, by: str = "publication", level: float = 0.95) -> dict:
    """Leave-one-<by>-out jackknife of a statistic fn(frame) -> float: jackknife SE around the full-data point,
    t(P - 1) interval, and the min / max leave-out values with the group left out.  The t(P - 1) interval treats the
    groups as independent; publications are nested in cpg clusters (39 in 13 on E-A, Kish 2.24; 12 in 3 on E-CT
    PRIMARY, Kish 1.25), so like JK_G1 it is too narrow (anti-conservative) and is descriptive only; the min / max
    leave-out range is the honest spread (METRICS section 6)."""
    point = float(fn(t))
    groups = sorted(t[by].astype(str).unique())
    col = t[by].astype(str)
    jk = np.array([float(fn(t[col != g])) for g in groups])
    ok = np.isfinite(jk)
    v, gs = jk[ok], [g for g, o in zip(groups, ok) if o]
    P = len(v)
    if P < 2:
        return {"point": point, "se": float("nan"), "lo": float("nan"), "hi": float("nan"), "P": P, "df": max(P - 1, 0),
                "n_undefined": int((~ok).sum()), "min": float("nan"), "min_by": None, "max": float("nan"),
                "max_by": None, "values": dict(zip(groups, jk.tolist()))}
    se = float(np.sqrt((P - 1) / P * ((v - v.mean()) ** 2).sum()))
    q = float(SS.t.ppf(0.5 + level / 2, P - 1))
    return {"point": point, "se": se, "lo": point - q * se, "hi": point + q * se, "P": P, "df": P - 1,
            "n_undefined": int((~ok).sum()), "min": float(v.min()), "min_by": gs[int(np.argmin(v))],
            "max": float(v.max()), "max_by": gs[int(np.argmax(v))], "values": dict(zip(groups, jk.tolist()))}


def top3_share(contrib) -> float:
    """Share of the total contribution held by the three largest unit contributions (NaN when the total <= 0)."""
    c = np.asarray(contrib, float)
    c = c[np.isfinite(c)]
    tot = float(c.sum())
    if tot <= 0:
        return float("nan")
    return float(np.sort(c)[::-1][:3].sum() / tot)


# ------------------------------------------------------------------------------------------------------------
# the section-9.3 per-item table
# ------------------------------------------------------------------------------------------------------------
ITEM_FIELDS = (
    ("item", "Item id and estimand"),
    ("population", "Population (cells, publications, units, rows per seed)"),
    ("design", "Design"),
    ("read_budget", "Read budget"),
    ("arms", "Arms on identical rows"),
    ("metrics", "R², MSE, relative SSE vs B1"),
    ("per_seed", "Per-seed values"),
    ("units", "Unit W/L/T and the descriptive tests"),
    ("lopo_top3", "LOPO and top-3 share"),
    ("c5", "C5-BTBP in / out"),
    ("duplicates", "Duplicate sensitivities"),
    ("leakage", "Leakage status"),
    ("compute", "Compute"),
    ("label", "Label"),
    ("changed", "What changed scientifically"),
    ("next_step", "Next registered step"),
)


def _fmt(v, floatfmt: str = ".4f") -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "NaN" if not np.isfinite(v) else format(float(v), floatfmt)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_fmt(x, floatfmt)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x, floatfmt) for x in v)
    return str(v).replace("|", "\\|").replace("\n", " ")


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Plain markdown table (no tabulate dependency); the index is written only when it is named."""
    d = df.reset_index() if df.index.name is not None else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in d.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, floatfmt) for v in row) + " |")
    return "\n".join(lines)


def per_item_table(item: dict, path=None, *, title: str | None = None, floatfmt: str = ".4f") -> str:
    """The section-9.3 table.  ``item`` must hold every key of ITEM_FIELDS (state 'n/a' explicitly); a value is a
    string / number / dict / list (one cell) or a DataFrame (its own sub-table).  ``label`` must name a section-9.1
    label.  Writes LF markdown to ``path`` when given and returns the text."""
    missing = [k for k, _ in ITEM_FIELDS if k not in item]
    if missing:
        raise KeyError(f"per_item_table: missing section-9.3 fields {missing}")
    if not any(lb in str(item["label"]) for lb in LABELS):
        raise ValueError(f"per_item_table: label {item['label']!r} names no section-9.1 label")
    head = title or f"{_fmt(item['item'])}: per-item table"
    lines, subs = [f"# {head}", "", f"**Label: {_fmt(item['label'])}**", "", "| field | value |", "|---|---|"], []
    for k, name in ITEM_FIELDS:
        v = item[k]
        if isinstance(v, pd.DataFrame):
            lines.append(f"| {name} | see '{name}' below |")
            subs += ["", f"## {name}", "", md_table(v, floatfmt)]
        else:
            lines.append(f"| {name} | {_fmt(v, floatfmt)} |")
    text = "\n".join(lines + subs) + "\n"
    if path is not None:
        paths.write_text_lf(Path(path), text)
    return text


# ------------------------------------------------------------------------------------------------------------
# the run record (brief 10)
# ------------------------------------------------------------------------------------------------------------
def run_record(item: str, mode: str, tag: str, status=None, note: str = "", *, event: str | None = None) -> str:
    """Append '- <UTC> START|END <item> <mode> seal=<tag> status=<s> <note>' to RUN_RECORD.md (binary LF, append-only).
    ``event`` defaults to START when ``status`` is None, else END.  Returns the line written."""
    ev = event or ("START" if status is None else "END")
    utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    line = f"- {utc} {ev} {item} {mode} seal={tag}"
    if status is not None:
        line += f" status={status}"
    if note:
        line += f" {note}"
    line = " ".join(line.split("\n")).rstrip()
    with open(RUN_RECORD, "ab") as fh:
        fh.write((line + "\n").encode("utf-8"))
    return line

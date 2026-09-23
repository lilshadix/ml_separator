"""``evaluation/figures.py`` -- brief section 25 figures 7-13, drawn from frames the caller assembled from files.

Nothing here reads a discovery record or fits anything: ``scripts/g19_make_figures.py`` assembles the frames (through the
verified readers of ``gen19ct.evaluation.discovery`` / the scorer's ``Store``, after discovery is complete) and passes
them in; the tests pass synthetic frames.  Every figure function returns a :class:`FigureResult` naming the input files
it used (``inputs``), the PNG it wrote and the companion CSV of the plotted points (``figures/data/F<nn>_*.csv``), so
every number on a figure is traceable; when the frame is absent the function returns ``status = "skipped"`` with the
reason and writes nothing.  matplotlib Agg, PNG at dpi 130, one scientific question per figure stated in its title
(README conventions).

| figure | question | frame |
|---|---|---|
| F07 | does the deployed predictor reconstruct hidden chemistry better than the lookup, under V5 / V1 / V2? | rows: ``log_D``, ``mean_logD``, ``metal_class``, ``unit`` per (design, arm) |
| F08 | is the error larger on cells farther from their training support, per domain status? | cells: ``mae``, ``support_score``, ``domain_status``, ``metal_class``, ``system`` per arm |
| F09 | are the intervals calibrated (reliability diagram by design; coverage by domain status)? | rows with ``lower_<pct>`` / ``upper_<pct>``, ``design``, ``unit``, ``domain_status`` |
| F10 | does the learned metal embedding recover the lanthanide series and separate the actinides? | ``e_m_*`` per metal state (+ bootstrap replicates for Procrustes ellipses) |
| F11 | does the learned extractant embedding group systems by family? | ``e_l_*`` per system with ``family`` |
| F12 | are the hidden Pr / Nd cells of V6 reconstructed (log D per row, logSF per system)? | the confirmation run's V6 rows and pairs |
| F13 | is the selectivity direction right (confusion matrix on V5-PAIR and V6 pairs)? | pairs: ``observed_logsf``, ``predicted_logsf`` per design |

Figures 14-15 (the process Pareto front and the probability-of-specification map) belong to the process step.
"""
from __future__ import annotations

import math
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.evaluation import calibration as EC  # noqa: E402
from gen19ct.evaluation import metrics as EM  # noqa: E402
from gen19ct.evaluation import transfer as ET  # noqa: E402

SCHEMA = "gen19.figures.v1"
DPI = 130
CLASS_COLORS: dict[str, str] = {"lanthanide": "#2a78d6", "actinide": "#eb6834", "other": "#1baf7a"}
STATUS_COLORS: dict[str, str] = {
    "IN_DOMAIN": "#1b7f3b", "INTERPOLATION": "#5cb85c", "CONDITION_EXTRAPOLATION": "#a8c400",
    "CROSS_METAL_TRANSFER": "#2a78d6", "CROSS_LIGAND_TRANSFER": "#7a4ecf", "CROSS_METAL_LIGAND_TRANSFER": "#eb6834",
    "FAMILY_EXTRAPOLATION": "#c2185b", "UNSUPPORTED": "#555555"}
GRID = {"color": "#e6e6e6", "lw": 0.6}
FIGURE_FILES: dict[str, str] = {
    "F07": "F07_pred_vs_measured.png", "F08": "F08_error_vs_support.png", "F09": "F09_uncertainty_calibration.png",
    "F10": "F10_metal_embedding.png", "F11": "F11_extractant_embedding.png", "F12": "F12_prnd_reconstruction.png",
    "F13": "F13_direction_confusion.png"}
#: atomic numbers of the f-block (a fact table for the series line of F10; not a chemistry descriptor)
Z_OF: dict[str, int] = {**{s: 57 + i for i, s in enumerate(("La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
                                                              "Ho", "Er", "Tm", "Yb", "Lu"))},
                        **{s: 89 + i for i, s in enumerate(("Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
                                                              "Es", "Fm", "Md", "No", "Lr"))}}
LN = frozenset(s for s, z in Z_OF.items() if z < 72)
AN = frozenset(s for s, z in Z_OF.items() if z >= 89)
#: section 4: direction accuracy on |observed logSF| >= 0.3 for EVERY design (the registered primary reading) plus, for
#: V6 only, the additional >= 0.1 reading -- one F13 panel per (design, threshold) (task X finding V-08)
PRIMARY_DIRECTION_THRESHOLD = 0.3
DIRECTION_THRESHOLDS: dict[str, tuple[float, ...]] = {"V5-PAIR": (PRIMARY_DIRECTION_THRESHOLD,),
                                                      "V6": (PRIMARY_DIRECTION_THRESHOLD, 0.1)}


@dataclass
class FigureResult:
    figure: str
    status: str                      # written | skipped
    path: str | None = None
    data_path: str | None = None
    inputs: list[str] = field(default_factory=list)
    reason: str = ""
    stats: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return asdict(self)


def skipped(figure: str, reason: str, inputs: Iterable[str] = ()) -> FigureResult:
    return FigureResult(figure=figure, status="skipped", inputs=sorted(set(str(i) for i in inputs)), reason=reason)


def _rel(p: Path) -> str:
    try:
        return paths.rel(p)
    except ValueError:
        return str(p)


def _suptitle(fig: plt.Figure, text: str, *, fontsize: float = 10, width: int | None = None) -> None:
    """``fig.suptitle`` with every line wrapped to the figure width (about 14 characters per inch at fontsize 10), so a
    one-panel figure never clips its question."""
    w = width or max(60, int(fig.get_figwidth() * 14))
    fig.suptitle("\n".join(textwrap.fill(line, w) for line in str(text).split("\n")), fontsize=fontsize)


def _save(fig: plt.Figure, out: Path) -> str:
    paths.ensure_dir(Path(out).parent)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return _rel(Path(out))


def _data_csv(df: pd.DataFrame, figures_dir: Path, name: str) -> str:
    from gen19ct.manifest import write_csv

    p = write_csv(df, Path(figures_dir) / "data" / f"{name}.csv")
    return _rel(p)


# --------------------------------------------------------------------------------------------- #
# F07 predicted vs measured under each strict hold-out
# --------------------------------------------------------------------------------------------- #

def unit_macro_mae(frame: pd.DataFrame, unit_col: str = "unit") -> tuple[float, int]:
    err = (frame["mean_logD"].astype(float) - frame["log_D"].astype(float)).abs()
    per = err.groupby(frame[unit_col].astype(str)).mean()
    return float(per.mean()), int(len(per))


def fig07_pred_vs_measured(frames: Mapping[str, Mapping[str, pd.DataFrame]], figures_dir: Path, *, inputs: Iterable[str],
                           deployed: str, comparator: str = "B3i", designs: Sequence[str] = ("V5", "V1", "V2")
                           ) -> FigureResult:
    """``frames[design][arm]`` rows with ``log_D``, ``mean_logD``, ``metal_class``, ``unit``.  Panels: arms x designs."""
    arms = [deployed, comparator] if deployed != comparator else [deployed]
    have = [(d, a) for d in designs for a in arms if d in frames and a in frames[d] and len(frames[d][a])]
    if not have:
        return skipped("F07", "no prediction frame for the deployed predictor or the comparator on V5 / V1 / V2", inputs)
    fig, axes = plt.subplots(len(arms), len(designs), figsize=(4.4 * len(designs), 4.2 * len(arms)), squeeze=False)
    allv = np.concatenate([np.r_[frames[d][a]["log_D"].to_numpy(dtype=float), frames[d][a]["mean_logD"].to_numpy(dtype=float)]
                           for d, a in have])
    lo, hi = float(np.nanmin(allv)) - 0.3, float(np.nanmax(allv)) + 0.3
    data = []
    for i, arm in enumerate(arms):
        for j, design in enumerate(designs):
            ax = axes[i][j]
            fr = frames.get(design, {}).get(arm)
            ax.plot([lo, hi], [lo, hi], color="#8a8a8a", lw=1.0, zorder=1)
            if fr is None or not len(fr):
                ax.set_title(f"{arm} on {design}: not computed (no predictions)", fontsize=9)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                continue
            for cls in ("lanthanide", "actinide", "other"):
                s = fr[fr["metal_class"].astype(str) == cls]
                if len(s):
                    ax.scatter(s["log_D"], s["mean_logD"], s=8, alpha=0.45, color=CLASS_COLORS[cls], edgecolors="none",
                               label=f"{cls} ({s['unit'].astype(str).nunique()} units, {len(s)} rows)", zorder=2)
            mae, n_units = unit_macro_mae(fr)
            data.append({"arm": arm, "design": design, "macro_mae": mae, "n_units": n_units, "n_rows": int(len(fr)),
                         "averaging_unit": EM.registered_unit_cols(design) if design in ("V5", "V1", "V2") else design})
            ax.set_title(f"{arm} on {design}: macro MAE {mae:.3f} log D over {n_units} {'cells' if design == 'V5' else 'units'}",
                         fontsize=9)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.grid(**GRID)
            ax.set_xlabel("measured log D")
            ax.legend(fontsize=7, loc="upper left", frameon=False)
        axes[i][0].set_ylabel(f"predicted log D ({arm})")
    _suptitle(fig, f"Does the deployed predictor ({deployed}) reconstruct hidden chemistry better than the within-system lookup "
                 f"({comparator}) under the strict hold-outs V5 (hidden cell), V1 (hidden publication) and V2 (hidden metal)?\n"
                 "selection half, seed 104729 (learned arm); discovery, optimistically biased", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(figures_dir) / FIGURE_FILES["F07"]
    path = _save(fig, out)
    data_path = _data_csv(pd.DataFrame(data), Path(figures_dir), "F07_pred_vs_measured")
    return FigureResult(figure="F07", status="written", path=path, data_path=data_path,
                        inputs=sorted(set(str(i) for i in inputs)), stats={"panels": data})


# --------------------------------------------------------------------------------------------- #
# F08 error vs support score
# --------------------------------------------------------------------------------------------- #

def error_vs_support_stats(cells: pd.DataFrame, *, x: str = "support_score", y: str = "mae", cluster_col: str = "system",
                           n_resamples: int = ET.N_RESAMPLES, seed: int = ET.BOOTSTRAP_SEED) -> dict[str, Any]:
    """Spearman rho(cell MAE, x) with the system-cluster bootstrap percentile interval (section 9 S1(e); 10,000 resamples,
    seed 19 by default).  NaN rows are dropped and counted."""
    sub = cells[[x, y, cluster_col]].copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    n_dropped = int(sub[[x, y]].isna().any(axis=1).sum())
    sub = sub.dropna(subset=[x, y])
    if len(sub) < 3 or sub[cluster_col].nunique() < 2:
        return {"x": x, "rho": float("nan"), "percentile_low": float("nan"), "percentile_high": float("nan"),
                "n_cells": int(len(sub)), "n_clusters": int(sub[cluster_col].nunique()), "n_dropped_nan": n_dropped,
                "status": "not computed: fewer than 3 cells or 2 clusters"}

    def stat(fr: pd.DataFrame) -> float:
        return EM.spearman_rho(fr[x].to_numpy(dtype=float), fr[y].to_numpy(dtype=float))
    br = ET.cluster_bootstrap_statistic(sub, cluster_col, stat, n_resamples=n_resamples, seed=seed,
                                        contrast=f"spearman({y},{x})", cluster_unit=cluster_col)
    lo, hi = br.percentile_interval()
    return {"x": x, "rho": br.point, "percentile_low": lo, "percentile_high": hi, "n_cells": int(len(sub)),
            "n_clusters": br.n_clusters, "n_resamples": br.n_resamples, "bootstrap_seed": br.seed, "n_dropped_nan": n_dropped,
            "status": "computed"}


def fig08_error_vs_support(cells_by_arm: Mapping[str, pd.DataFrame], figures_dir: Path, *, inputs: Iterable[str],
                           stats: Mapping[str, Mapping[str, Any]] | None = None) -> FigureResult:
    """``cells_by_arm[arm]`` rows per hidden cell: ``mae``, ``support_score``, ``domain_status``, ``metal_class``, ``system``."""
    arms = [a for a, c in cells_by_arm.items() if c is not None and len(c)]
    if not arms:
        return skipped("F08", "no per-cell error / support table", inputs)
    fig, axes = plt.subplots(1, len(arms), figsize=(5.6 * len(arms), 4.8), sharey=True, squeeze=False)
    rows = []
    for ax, arm in zip(axes[0], arms):
        c = cells_by_arm[arm]
        for status in list(STATUS_COLORS) + sorted(set(c["domain_status"].astype(str)) - set(STATUS_COLORS)):
            s = c[c["domain_status"].astype(str) == status]
            if len(s):
                ax.scatter(s["support_score"], s["mae"], s=24, alpha=0.75, color=STATUS_COLORS.get(status, "#999999"),
                           edgecolors="white", linewidths=0.5, label=f"{status} ({len(s)} cells)")
        st = (stats or {}).get(arm) or {}
        rho = st.get("rho", float("nan"))
        lo, hi = st.get("percentile_low", float("nan")), st.get("percentile_high", float("nan"))
        ax.set_title(f"{arm}: Spearman rho {rho:.2f} [{lo:.2f}, {hi:.2f}] over {st.get('n_cells', len(c))} cells "
                     f"(system-cluster bootstrap)", fontsize=9)
        ax.set_xlabel("support_score v1 (section 13; 1 = well supported)")
        ax.grid(**GRID)
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")
        for _, r in c.iterrows():
            rows.append({"arm": arm, **{k: r[k] for k in ("system", "domain_status", "metal_class", "mae", "support_score")
                                        if k in c.columns}})
    axes[0][0].set_ylabel("cell MAE (log D)")
    _suptitle(fig, "Is the deployed predictor wrong by more on hidden cells that sit farther from their training support, and "
                 "does this hold inside every domain-status category?\nV5-primary, selection half; S1(e) needs rho <= -0.10 with "
                 "the system-cluster interval excluding 0 and reliable support components", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = Path(figures_dir) / FIGURE_FILES["F08"]
    path = _save(fig, out)
    data_path = _data_csv(pd.DataFrame(rows), Path(figures_dir), "F08_error_vs_support")
    return FigureResult(figure="F08", status="written", path=path, data_path=data_path,
                        inputs=sorted(set(str(i) for i in inputs)),
                        stats={"per_arm": {a: dict((stats or {}).get(a) or {}) for a in arms}})


# --------------------------------------------------------------------------------------------- #
# F09 uncertainty calibration
# --------------------------------------------------------------------------------------------- #

def coverage_table(rows: pd.DataFrame, *, unit_col: str = "unit", levels: Sequence[float] = EC.LEVELS) -> dict[float, float]:
    """Unit-macro empirical coverage per level (the section 4 averaging: per unit, then equal-weight mean)."""
    y = rows["log_D"].to_numpy(dtype=float)
    out = {}
    for lvl in levels:
        lo_c, hi_c = EM.interval_columns(lvl)
        lo = pd.to_numeric(rows[lo_c], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(rows[hi_c], errors="coerce").to_numpy(dtype=float)
        covered = ((y >= lo) & (y <= hi)).astype(float)
        per = pd.Series(covered).groupby(rows[unit_col].astype(str).to_numpy()).mean()
        out[lvl] = float(per.mean()) if len(per) else float("nan")
    return out


def fig09_calibration(rows_by_design: Mapping[str, pd.DataFrame], figures_dir: Path, *, inputs: Iterable[str], arm: str,
                      status_col: str = "domain_status") -> FigureResult:
    """``rows_by_design[design]``: scored rows with ``log_D``, ``lower_<pct>`` / ``upper_<pct>``, ``unit`` and (V5) the
    domain status.  Left: reliability diagram (nominal vs empirical unit-macro coverage) per design; right: 50 / 80 / 95 %
    coverage per domain-status category on V5 with the S1(d) bands."""
    designs = [d for d, fr in rows_by_design.items() if fr is not None and len(fr)]
    if not designs:
        return skipped("F09", f"no interval rows for {arm}", inputs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9))
    nominal = list(EC.LEVELS)
    data = []
    ax1.plot([0, 1], [0, 1], color="#8a8a8a", lw=1.0)
    for d in designs:
        cov = coverage_table(rows_by_design[d])
        ax1.plot(nominal, [cov[l] for l in nominal], marker="o", label=f"{d} ({rows_by_design[d]['unit'].astype(str).nunique()} units)")
        for l in nominal:
            data.append({"design": d, "arm": arm, "category": "all", "level": l, "coverage_unit_macro": cov[l],
                         "n_units": int(rows_by_design[d]["unit"].astype(str).nunique()), "n_rows": int(len(rows_by_design[d]))})
    for l, (lo, hi) in EC.S1D_BANDS.items():
        ax1.vlines(l, lo, hi, color="#c2185b", lw=6, alpha=0.25)
    ax1.set_xlabel("nominal coverage")
    ax1.set_ylabel(f"empirical unit-macro coverage of {arm}")
    ax1.set_title("reliability diagram by design (bands: S1(d) at 50 / 80 / 95 %)", fontsize=9)
    ax1.set_xlim(0.4, 1.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(**GRID)
    ax1.legend(fontsize=7.5, frameon=False, loc="lower right")
    v5 = rows_by_design.get("V5")
    if v5 is not None and status_col in v5.columns and len(v5):
        cats = [c for c in STATUS_COLORS if (v5[status_col].astype(str) == c).any()]
        cats += sorted(set(v5[status_col].astype(str)) - set(STATUS_COLORS))
        width = 0.26
        for k, l in enumerate(nominal):
            vals, ns = [], []
            for c in cats:
                s = v5[v5[status_col].astype(str) == c]
                vals.append(coverage_table(s)[l])
                ns.append(int(s["unit"].astype(str).nunique()))
                data.append({"design": "V5", "arm": arm, "category": c, "level": l, "coverage_unit_macro": vals[-1],
                             "n_units": ns[-1], "n_rows": int(len(s))})
            xs = np.arange(len(cats)) + (k - 1) * width
            ax2.bar(xs, vals, width=width, label=f"{int(l * 100)} %", alpha=0.85)
            if k == 1:
                for xx, n in zip(xs, ns):
                    ax2.text(xx, 1.01, f"n={n}", ha="center", fontsize=6.5)
        lo80, hi80 = EC.S1D_CATEGORY_BAND_80
        ax2.axhspan(lo80, hi80, color="#c2185b", alpha=0.08, label="S1(d) 80 % band per category")
        ax2.set_xticks(np.arange(len(cats)))
        ax2.set_xticklabels(cats, rotation=35, ha="right", fontsize=7)
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel("unit-macro coverage (V5 cells)")
        ax2.set_title("coverage by domain-status category (V5-primary; n = cells)", fontsize=9)
        ax2.legend(fontsize=7, frameon=False, loc="lower left")
        ax2.grid(axis="y", **GRID)
    else:
        ax2.set_title("coverage by domain status: not computed (no V5 rows with a domain status)", fontsize=9)
    _suptitle(fig, f"Are the split-conformal intervals of {arm} calibrated -- by design, and inside every domain-status category?\n"
                 "selection half, seed 104729; discovery, optimistically biased", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = Path(figures_dir) / FIGURE_FILES["F09"]
    path = _save(fig, out)
    data_path = _data_csv(pd.DataFrame(data), Path(figures_dir), "F09_uncertainty_calibration")
    return FigureResult(figure="F09", status="written", path=path, data_path=data_path,
                        inputs=sorted(set(str(i) for i in inputs)), stats={"coverage": data})


# --------------------------------------------------------------------------------------------- #
# F10 / F11 embeddings
# --------------------------------------------------------------------------------------------- #

def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centred PCA to two components: ``(coords, components (2 x d), mean, explained variance ratio)``."""
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0)
    xc = x - mean
    if xc.shape[0] < 2:
        return np.zeros((xc.shape[0], 2)), np.zeros((2, xc.shape[1])), mean, np.zeros(2)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(2, vt.shape[0])
    comps = np.zeros((2, xc.shape[1]))
    comps[:k] = vt[:k]
    coords = xc @ comps.T
    var = s ** 2
    ratio = np.zeros(2)
    if var.sum() > 0:
        ratio[:k] = var[:k] / var.sum()
    return coords, comps, mean, ratio


def element_of(state: str) -> str:
    return str(state).split("(")[0].strip()


def _ellipse(ax, pts: np.ndarray, color: str) -> None:
    if pts.shape[0] < 3:
        return
    cov = np.cov(pts.T)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0, None)
    angle = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * 2 * math.sqrt(vals[1]), 2 * 2 * math.sqrt(vals[0])
    ax.add_patch(Ellipse(pts.mean(axis=0), w, h, angle=angle, facecolor=color, alpha=0.12, edgecolor=color, lw=0.8))


def fig10_metal_embedding(emb: pd.DataFrame | None, figures_dir: Path, *, inputs: Iterable[str], arm: str = "M2",
                          replicates: Sequence[pd.DataFrame] | None = None,
                          stability: Mapping[str, Any] | None = None, replicate_label: str = "bootstrap") -> FigureResult:
    """``emb``: index = metal state (``Nd(III)``), numeric columns ``e_m_*``.  2-D PCA; the Ln(III) series joined in Z
    order, An states, others; Procrustes-aligned replicate ellipses (2 SD) when ``replicates`` are given.
    ``replicate_label`` names what the replicates ARE in the title -- ``"bootstrap"`` resamples, or ``"fold-replicate"``
    for the other outer-fold refits of the same design (leave-cells-out perturbations, which are not a bootstrap)."""
    if emb is None or not len(emb):
        return skipped("F10", "no metal-embedding table (evaluation/power/embeddings/metal_embeddings.csv; written by "
                              "g19_run_power.py --include-learned once implemented)", inputs)
    cols = [c for c in emb.columns if str(c).startswith("e_m_")]
    if not cols:
        return skipped("F10", "metal-embedding table without e_m_* columns", inputs)
    x = emb[cols].to_numpy(dtype=float)
    coords, comps, mean, ratio = pca_2d(x)
    states = [str(s) for s in emb.index]
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    rows = []
    for st, (px, py) in zip(states, coords):
        el = element_of(st)
        cls = "lanthanide" if el in LN else ("actinide" if el in AN else "other")
        ax.scatter(px, py, s=36, color=CLASS_COLORS[cls], edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(st, (px, py), fontsize=7, xytext=(3, 3), textcoords="offset points")
        rows.append({"metal_state": st, "class": cls, "Z": Z_OF.get(el), "pc1": px, "pc2": py})
    ln = sorted([(Z_OF[element_of(s)], i) for i, s in enumerate(states) if element_of(s) in LN and s.endswith("(III)")])
    if len(ln) >= 2:
        idx = [i for _, i in ln]
        ax.plot(coords[idx, 0], coords[idx, 1], color=CLASS_COLORS["lanthanide"], lw=1.0, alpha=0.7, zorder=2,
                label="Ln(III) series in Z order")
    n_rep = 0
    if replicates:
        from gen19ct.evaluation import power as PW
        for rep in replicates:
            common = [s for s in states if s in rep.index]
            if len(common) < 3:
                continue
            ref = emb.loc[common, cols].to_numpy(dtype=float)
            other = rep.loc[common, [c for c in cols if c in rep.columns]].to_numpy(dtype=float)
            if other.shape != ref.shape:
                continue
            aligned = PW.procrustes_align(ref, other)
            proj = (aligned - mean) @ comps.T
            n_rep += 1
            for st, p in zip(common, proj):
                rows.append({"metal_state": st, "class": "replicate", "Z": Z_OF.get(element_of(st)), "pc1": p[0], "pc2": p[1]})
        if n_rep:
            df = pd.DataFrame(rows)
            for st in states:
                pts = df[(df["metal_state"] == st) & (df["class"] == "replicate")][["pc1", "pc2"]].to_numpy(dtype=float)
                el = element_of(st)
                cls = "lanthanide" if el in LN else ("actinide" if el in AN else "other")
                _ellipse(ax, pts, CLASS_COLORS[cls])
    stab = (stability or {}).get("stability")
    stab_txt = (f"; Procrustes {replicate_label} stability {stab:.2f} over {n_rep} aligned replicates (null "
                f"{(stability or {}).get('null_stability', float('nan')):.2f}, gate {(stability or {}).get('gate')})"
                if stab is not None and np.isfinite(stab) else
                "; stability: not computed" if not replicates else f"; {n_rep} aligned {replicate_label} replicates (2-SD ellipses)")
    if (stability or {}).get("registered_reliability"):
        # the record is descriptive: the registered section 8 reliability of the embeddings has not run
        stab_txt += " -- descriptive: the registered section 8 embedding reliability is NOT_RUN"
    ax.set_xlabel(f"PC1 ({ratio[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({ratio[1]:.0%} of variance)")
    ax.grid(**GRID)
    ax.legend(fontsize=7.5, frameon=False, loc="best")
    _suptitle(fig, f"Does the learned metal embedding e_m of {arm} recover the lanthanide series and separate the actinides?\n"
                 f"2-D PCA of {len(states)} metal states{stab_txt}; a reliability below 0.3 makes any reading UNDECIDED", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(figures_dir) / FIGURE_FILES["F10"]
    path = _save(fig, out)
    data_path = _data_csv(pd.DataFrame(rows), Path(figures_dir), "F10_metal_embedding")
    return FigureResult(figure="F10", status="written", path=path, data_path=data_path,
                        inputs=sorted(set(str(i) for i in inputs)),
                        stats={"explained_variance_ratio": [float(r) for r in ratio], "n_states": len(states), "n_replicates": n_rep,
                               "replicate_label": replicate_label, "stability": dict(stability or {})})


def fig11_extractant_embedding(emb: pd.DataFrame | None, figures_dir: Path, *, inputs: Iterable[str], arm: str = "M2",
                               family_col: str = "family") -> FigureResult:
    """``emb``: index = system key, numeric ``e_l_*`` columns and a ``family`` column."""
    if emb is None or not len(emb):
        return skipped("F11", "no extractant-embedding table (evaluation/power/embeddings/system_embeddings.csv)", inputs)
    cols = [c for c in emb.columns if str(c).startswith("e_l_")]
    if not cols:
        return skipped("F11", "extractant-embedding table without e_l_* columns", inputs)
    coords, _, _, ratio = pca_2d(emb[cols].to_numpy(dtype=float))
    fam = emb[family_col].astype(str) if family_col in emb.columns else pd.Series("unknown", index=emb.index)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    cmap = plt.get_cmap("tab20")
    rows = []
    for k, f in enumerate(sorted(fam.unique())):
        m = (fam == f).to_numpy()
        ax.scatter(coords[m, 0], coords[m, 1], s=30, color=cmap(k % 20), edgecolors="white", linewidths=0.5,
                   label=f"{f} ({int(m.sum())})")
    for (sysk, f), (px, py) in zip(zip(emb.index, fam), coords):
        rows.append({"system": str(sysk), "family": f, "pc1": px, "pc2": py})
    ax.set_xlabel(f"PC1 ({ratio[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({ratio[1]:.0%} of variance)")
    ax.grid(**GRID)
    ax.legend(fontsize=6.5, frameon=False, loc="best", ncol=2)
    _suptitle(fig, f"Does the learned extractant embedding e_l of {arm} group systems by structural family?\n"
                 f"2-D PCA of {len(emb)} training systems, coloured by the Gen19 family classification", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(figures_dir) / FIGURE_FILES["F11"]
    path = _save(fig, out)
    data_path = _data_csv(pd.DataFrame(rows), Path(figures_dir), "F11_extractant_embedding")
    return FigureResult(figure="F11", status="written", path=path, data_path=data_path,
                        inputs=sorted(set(str(i) for i in inputs)),
                        stats={"explained_variance_ratio": [float(r) for r in ratio], "n_systems": int(len(emb)),
                               "n_families": int(fam.nunique())})


# --------------------------------------------------------------------------------------------- #
# F12 Pr/Nd reconstruction (V6, confirmation files only)
# --------------------------------------------------------------------------------------------- #

def fig12_prnd_reconstruction(rows: pd.DataFrame | None, pairs: pd.DataFrame | None, figures_dir: Path, *,
                              inputs: Iterable[str], arm: str = "deployed") -> FigureResult:
    """``rows``: the confirmation run's hidden Pr / Nd rows (``system``, ``metal_state``, ``log_D``, ``mean_logD``,
    ``lower_80``, ``upper_80``); ``pairs``: per system observed and predicted logSF (``system``, ``observed_logsf``,
    ``predicted_logsf``).  Only the confirmation files feed this figure (V6 is touched once, section 3.4)."""
    if (rows is None or not len(rows)) and (pairs is None or not len(pairs)):
        return skipped("F12", "no confirmation V6 files (evaluation/confirmation/v6_rows.csv, v6_pairs.csv): V6 is run once, "
                              "at confirmation", inputs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2))
    data = []
    if rows is not None and len(rows):
        allv = np.r_[rows["log_D"].to_numpy(dtype=float), rows["mean_logD"].to_numpy(dtype=float)]
        lo, hi = float(np.nanmin(allv)) - 0.3, float(np.nanmax(allv)) + 0.3
        ax1.plot([lo, hi], [lo, hi], color="#8a8a8a", lw=1.0)
        for st, color in (("Pr(III)", "#2a78d6"), ("Nd(III)", "#eb6834")):
            s = rows[rows["metal_state"].astype(str) == st]
            if len(s):
                yerr = None
                if "lower_80" in s.columns and "upper_80" in s.columns:
                    yerr = np.vstack([(s["mean_logD"] - s["lower_80"]).to_numpy(dtype=float),
                                      (s["upper_80"] - s["mean_logD"]).to_numpy(dtype=float)])
                ax1.errorbar(s["log_D"], s["mean_logD"], yerr=yerr, fmt="o", ms=4, color=color, alpha=0.6, elinewidth=0.6,
                             label=f"{st} ({s['system'].astype(str).nunique()} systems, {len(s)} rows)")
        mae, n = unit_macro_mae(rows.assign(unit=rows["system"].astype(str)))
        ax1.set_title(f"hidden Pr / Nd rows: system-macro MAE {mae:.3f} log D over {n} systems (80 % intervals)", fontsize=9)
        ax1.set_xlabel("measured log D")
        ax1.set_ylabel(f"predicted log D ({arm})")
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(lo, hi)
        ax1.grid(**GRID)
        ax1.legend(fontsize=7.5, frameon=False, loc="upper left")
        data.append({"panel": "rows", "system_macro_mae": mae, "n_systems": n, "n_rows": int(len(rows))})
    else:
        ax1.set_title("hidden Pr / Nd rows: not computed (input missing)", fontsize=9)
    if pairs is not None and len(pairs):
        med = pairs.groupby(pairs["system"].astype(str)).agg(observed=("observed_logsf", "median"),
                                                             predicted=("predicted_logsf", "median"), n=("observed_logsf", "size"))
        ax2.axhline(0, color="#8a8a8a", lw=0.8)
        ax2.axvline(0, color="#8a8a8a", lw=0.8)
        agree = np.sign(med["observed"]) == np.sign(med["predicted"])
        ax2.scatter(med["observed"], med["predicted"], s=40, c=np.where(agree, "#1b7f3b", "#c2185b"), edgecolors="white")
        for sysk, r in med.iterrows():
            ax2.annotate(f"{str(sysk)[:12]} (n={int(r['n'])})", (r["observed"], r["predicted"]), fontsize=6, xytext=(3, 3),
                         textcoords="offset points")
            data.append({"panel": "pairs", "system": sysk, "observed_median_logsf": r["observed"],
                         "predicted_median_logsf": r["predicted"], "n_pairs": int(r["n"]), "sign_agrees": bool(agree.loc[sysk])})
        ax2.set_title(f"per-system median logSF(Nd/Pr): sign agrees in {int(agree.sum())} of {len(med)} systems "
                      f"(S2(a) needs >= 11 of 13)", fontsize=9)
        ax2.set_xlabel("observed median logSF")
        ax2.set_ylabel("predicted median logSF")
        ax2.grid(**GRID)
    else:
        ax2.set_title("per-system logSF: not computed (input missing)", fontsize=9)
    _suptitle(fig, "Are the hidden Pr and Nd cells of the 13 V6 systems reconstructed -- log D per row and the Pr/Nd selectivity per system?\n"
                 "V6 double-cell hold-out, the single confirmation run (withheld seeds)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = Path(figures_dir) / FIGURE_FILES["F12"]
    path = _save(fig, out)
    data_path = _data_csv(pd.DataFrame(data), Path(figures_dir), "F12_prnd_reconstruction")
    return FigureResult(figure="F12", status="written", path=path, data_path=data_path,
                        inputs=sorted(set(str(i) for i in inputs)), stats={"panels": data})


# --------------------------------------------------------------------------------------------- #
# F13 selectivity-direction confusion matrix
# --------------------------------------------------------------------------------------------- #

def direction_confusion(pairs: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Counts on pairs with |observed logSF| >= threshold: observed sign (rows) x predicted sign (columns; a predicted
    zero is its own column and counts 1/2 in the accuracy, section 4).  The gate is the registered one of
    ``evaluation.pairs`` -- ``>= threshold - metrics.FLOAT_TOL`` -- so a difference of two stored log D values that is
    exactly 0.3 but represented as 0.29999999999999993 qualifies here as it does in the scorer's S1(c) block
    (``decisions.json -> S1_components.S1c_selection.direction_gate``; task X finding V-TR-09)."""
    obs = pd.to_numeric(pairs["observed_logsf"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(pairs["predicted_logsf"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(obs) & np.isfinite(pred) & (np.abs(obs) >= threshold - EM.FLOAT_TOL)
    obs, pred = obs[ok], pred[ok]
    so, sp = np.sign(obs), np.sign(pred)
    counts = {f"obs_{a}_pred_{b}": int(((so == sa) & (sp == sb)).sum())
              for a, sa in (("pos", 1), ("neg", -1)) for b, sb in (("pos", 1), ("zero", 0), ("neg", -1))}
    n = int(ok.sum())
    correct = float(((so == sp) & (sp != 0)).sum() + 0.5 * (sp == 0).sum())
    return {"threshold": threshold, "n_pairs": n, "n_excluded_below_threshold_or_nan": int((~ok).sum()),
            "direction_accuracy": correct / n if n else float("nan"), "counts": counts}


def fig13_direction_confusion(pairs_by_design: Mapping[str, pd.DataFrame], figures_dir: Path, *, inputs: Iterable[str],
                              arm: str = "M2", thresholds: Mapping[str, Sequence[float]] = DIRECTION_THRESHOLDS,
                              skipped_note: str = "") -> FigureResult:
    """One confusion panel per (design, registered threshold): |logSF| >= 0.3 on every design (primary) and the
    additional >= 0.1 reading on V6 (section 4); the data CSV flags the primary reading per row.  ``skipped_note`` is
    printed in the title and recorded in ``reason`` / ``stats`` for a panel the caller could not supply (e.g. the V6
    pairs before the confirmation run)."""
    designs = [d for d, p in pairs_by_design.items() if p is not None and len(p)]
    if not designs:
        return skipped("F13", "no pair-level predictions (V5-PAIR from the M2 records; V6 from the confirmation files)", inputs)
    panels = [(d, float(t)) for d in designs for t in (thresholds.get(d) or (PRIMARY_DIRECTION_THRESHOLD,))]
    fig, axes = plt.subplots(1, len(panels), figsize=(max(9.6, 5.4 * len(panels)), 4.6), squeeze=False)
    data = []
    for ax, (d, thr) in zip(axes[0], panels):
        cm = direction_confusion(pairs_by_design[d], thr)
        c = cm["counts"]
        mat = np.array([[c["obs_pos_pred_pos"], c["obs_pos_pred_zero"], c["obs_pos_pred_neg"]],
                        [c["obs_neg_pred_pos"], c["obs_neg_pred_zero"], c["obs_neg_pred_neg"]]], dtype=float)
        ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(3):
                ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=11,
                        color="white" if mat[i, j] > mat.max() / 2 else "black")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["pred +", "pred 0 (1/2)", "pred -"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["obs +", "obs -"])
        primary = thr == PRIMARY_DIRECTION_THRESHOLD
        tag = "registered primary reading" if primary else "additional V6 reading (section 4)"
        ax.set_title(f"{d}, |logSF| >= {thr:g} ({tag}): direction accuracy {cm['direction_accuracy']:.3f} on "
                     f"{cm['n_pairs']} pairs (FLAT = 0.5)", fontsize=9)
        data.append({"design": d, "arm": arm, "registered_primary": primary,
                     "reading": "section 4 primary (every design)" if primary else "section 4 additional (V6 only)",
                     **{k: v for k, v in cm.items() if k != "counts"}, **c})
    _suptitle(fig, f"Does {arm} get the selectivity direction right on hidden cell pairs (V5-PAIR) and on the V6 Pr/Nd pairs?\n"
                 "rows: observed sign; columns: predicted sign; a predicted zero counts 1/2 (section 4)"
                 + (f"\n{skipped_note}" if skipped_note else ""), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = Path(figures_dir) / FIGURE_FILES["F13"]
    path = _save(fig, out)
    data_path = _data_csv(pd.DataFrame(data), Path(figures_dir), "F13_direction_confusion")
    return FigureResult(figure="F13", status="written", path=path, data_path=data_path,
                        inputs=sorted(set(str(i) for i in inputs)), reason=skipped_note,
                        stats={"per_design": data, "designs_drawn": designs, "skipped_note": skipped_note})


# --------------------------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------------------------- #

def figures_index(results: Sequence[FigureResult], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"schema": SCHEMA, "figures": [r.record() for r in results],
            "n_written": sum(r.status == "written" for r in results),
            "n_skipped": sum(r.status == "skipped" for r in results), **dict(extra or {})}

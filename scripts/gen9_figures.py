#!/usr/bin/env python
"""The six gen9 figures, on the same axes as gen8's wherever a comparison is meant.

Figures are here to make a claim checkable, not to decorate the report, so each one
is built around a single question and drawn so the answer is legible without the
caption:

1. **slope recovery** — measured against predicted slope, old model and new, on
   identical axes with the ``y = x`` line drawn. The gen8 version of this plot is a
   horizontal smear at zero; if gen9 worked, the new panel climbs toward the diagonal.
2. **span recovery** — the same for dynamic range, which is the quantity gen8
   measured at 5 % of truth.
3. **response surfaces** — representative held-out ligands from each failure class,
   truth against every model and adapter, because an aggregate can hide overshoot,
   oscillation and sign reversal and a curve cannot.
4. **acquisition regret** — every policy's distance from the one-shot oracle.
5. **adaptation curve** — k = 0/1/2/3/5 for gen8's best and gen9's finalists.
6. **error decomposition** — how much of the final number each component bought.

Every panel that can be empty prints why and is skipped rather than written blank.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RUN = REPO_ROOT / "runs" / "gen9_shape"
FIGURES = RUN / "figures"
MEMBERSHIP = RUN / "curves" / "curve_membership.parquet"
FAILURE_CLASSES = (REPO_ROOT / "runs" / "gen8_architecture" / "case_studies"
                   / "ligand_failure_classes.csv")

PRIMARY_AXES = ("extractant", "acid", "metal_series")
AXIS_TITLE = {"extractant": "extractant titration", "acid": "acid titration",
              "metal_series": "lanthanide series"}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False, "figure.autolayout": False,
    })
    return plt


# --------------------------------------------------------------------------- #
# 1 & 2 — slope and span recovery
# --------------------------------------------------------------------------- #

def figure_slope_recovery(curves: pd.DataFrame, models: list[str], out: Path,
                          *, column="slope", label="slope", limit=None) -> Path | None:
    plt = _style()
    axes_present = [a for a in PRIMARY_AXES if a in set(curves["axis_label"])]
    if not axes_present or not models:
        print(f"  (skipping {label} recovery: nothing to plot)")
        return None
    fig, grid = plt.subplots(len(axes_present), len(models),
                             figsize=(3.1 * len(models), 3.0 * len(axes_present)),
                             squeeze=False)
    for r, axis in enumerate(axes_present):
        block_axis = curves[curves["axis_label"] == axis]
        true_all = block_axis[f"{column}_true"].to_numpy(dtype=float)
        if limit is None:
            hi = float(np.nanquantile(np.abs(true_all), 0.98)) or 1.0
            lo = -hi if column == "slope" else 0.0
        else:
            lo, hi = limit
        for c, model in enumerate(models):
            ax = grid[r][c]
            block = block_axis[block_axis["model"] == model]
            x = block[f"{column}_true"].to_numpy(dtype=float)
            y = block[f"{column}_pred"].to_numpy(dtype=float)
            ax.plot([lo, hi], [lo, hi], color="0.4", lw=1.0, ls="--", zorder=1)
            ax.scatter(x, y, s=7, alpha=0.28, linewidths=0, color="#2b6cb0", zorder=2)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() > 2:
                ratio = float(np.nanmedian(y[finite & (np.abs(x) > 1e-6)]
                                           / x[finite & (np.abs(x) > 1e-6)])) \
                    if column == "slope" else float(np.nanmedian(
                        y[finite & (x > 0.5)] / x[finite & (x > 0.5)]))
                ax.set_title(f"{model}\nmedian ratio {ratio:+.2f}  (n={int(finite.sum())})",
                             fontsize=8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            if c == 0:
                ax.set_ylabel(f"predicted {label}\n{AXIS_TITLE.get(axis, axis)}")
            if r == len(axes_present) - 1:
                ax.set_xlabel(f"measured {label}")
    fig.suptitle(f"Figure {'1' if column == 'slope' else '2'} — "
                 f"{label} recovery on held-out chemotypes", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# --------------------------------------------------------------------------- #
# 3 — response surfaces
# --------------------------------------------------------------------------- #

def pick_examples(curves: pd.DataFrame, classes: pd.DataFrame, *, reference: str,
                  candidate: str, n_per_class: int = 1) -> list[dict]:
    """One curve per failure class, plus the biggest gain and the biggest loss.

    Choosing the extremes on purpose: an aggregate that improves while its worst case
    explodes is the failure mode a table cannot show.
    """
    wide = curves.pivot_table(index=["curve_id", "axis_label", "extractant"],
                              columns="model", values="shape_mae", aggfunc="mean")
    if reference not in wide.columns or candidate not in wide.columns:
        return []
    wide = wide.dropna(subset=[reference, candidate])
    wide["gain"] = wide[reference] - wide[candidate]
    wide = wide.reset_index()
    picks: list[dict] = []
    extractant_curves = wide[wide["axis_label"] == "extractant"]
    source = extractant_curves if len(extractant_curves) >= 4 else wide
    for label, frame, ascending in (("largest improvement", source, False),
                                    ("largest degradation", source, True)):
        row = frame.sort_values("gain", ascending=ascending).head(1)
        if len(row):
            picks.append({"label": label, "curve_id": row["curve_id"].iloc[0],
                          "axis_label": row["axis_label"].iloc[0],
                          "extractant": row["extractant"].iloc[0]})
    if len(classes):
        joined = wide.merge(classes[["extractant", "failure_class"]], on="extractant", how="left")
        for failure_class in ("RESIDUAL_SHAPE", "PURE_LEVEL", "ALREADY_GOOD"):
            block = joined[joined["failure_class"] == failure_class]
            block = block[block["axis_label"] == "extractant"] if len(
                block[block["axis_label"] == "extractant"]) else block
            for _, row in block.sort_values("gain", ascending=False).head(n_per_class).iterrows():
                picks.append({"label": failure_class, "curve_id": row["curve_id"],
                              "axis_label": row["axis_label"], "extractant": row["extractant"]})
    seen, unique = set(), []
    for pick in picks:
        if pick["curve_id"] in seen:
            continue
        seen.add(pick["curve_id"])
        unique.append(pick)
    return unique


def figure_response_surfaces(oof: pd.DataFrame, membership: pd.DataFrame,
                             picks: list[dict], models: list[str], out: Path) -> Path | None:
    plt = _style()
    if not picks:
        print("  (skipping response surfaces: no example curves)")
        return None
    columns = min(3, len(picks))
    rows = int(np.ceil(len(picks) / columns))
    fig, grid = plt.subplots(rows, columns, figsize=(4.1 * columns, 3.2 * rows), squeeze=False)
    palette = ["#d64545", "#2b6cb0", "#2f855a", "#805ad5", "#dd6b20"]
    for i, pick in enumerate(picks):
        ax = grid[i // columns][i % columns]
        block = membership[membership["curve_id"] == pick["curve_id"]]
        row_ids = block["row_id"].to_numpy()
        x = block["axis_value"].to_numpy(dtype=float)
        order = np.argsort(x)
        row_ids, x = row_ids[order], x[order]
        drawn = False
        for j, model in enumerate(models):
            sub = oof[(oof["model"] == model) & (oof["row_id"].isin(set(row_ids)))]
            if sub.empty:
                continue
            sub = sub.groupby("row_id")[["log_D", "prediction"]].mean()
            sub = sub.reindex(row_ids)
            if not drawn:
                ax.plot(x, sub["log_D"], "o-", color="0.15", lw=1.8, ms=5,
                        label="measured", zorder=5)
                drawn = True
            ax.plot(x, sub["prediction"], "s--", color=palette[j % len(palette)],
                    lw=1.3, ms=4, alpha=0.9, label=model.replace("GEN9_", ""))
        ax.set_title(f"{pick['label']} — {AXIS_TITLE.get(pick['axis_label'], pick['axis_label'])}",
                     fontsize=8)
        ax.set_xlabel("axis coordinate (log10 for concentrations)")
        ax.set_ylabel("log D")
        if i == 0:
            ax.legend(fontsize=7, frameon=False)
    for k in range(len(picks), rows * columns):
        grid[k // columns][k % columns].axis("off")
    fig.suptitle("Figure 3 — held-out response surfaces: truth against each global model",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# --------------------------------------------------------------------------- #
# 4 — acquisition regret
# --------------------------------------------------------------------------- #

def figure_acquisition_regret(summary: pd.DataFrame, out: Path) -> Path | None:
    """Every policy's distance from the one-shot oracle.

    The axis starts just below the oracle rather than at zero: the spread that matters
    here is 0.48 to 0.82, and a zero-based axis compresses it into invisibility.
    """
    plt = _style()
    if summary is None or summary.empty:
        print("  (skipping acquisition regret: no summary)")
        return None
    frame = summary.sort_values("mae")
    floor = float(frame["mae"].min())
    left = floor - 0.06 * (float(frame["mae"].max()) - floor)
    fig, ax = plt.subplots(figsize=(7.4, 0.30 * len(frame) + 1.9))
    colours = ["#d64545" if p in ("ORACLE", "SURROGATE_ORACLE")
               else "#2b6cb0" if p.startswith("LEARNED") else "0.6"
               for p in frame["policy"]]
    ax.barh(range(len(frame)), frame["mae"] - left, left=left, color=colours, height=0.72)
    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels(frame["policy"], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(left, float(frame["mae"].max()) * 1.02)
    for name, style, colour in (("ORACLE", "--", "#d64545"), ("MEDOID", ":", "0.2"),
                                ("RANDOM", "-.", "0.2")):
        value = frame.loc[frame["policy"] == name, "mae"]
        if len(value):
            ax.axvline(float(value.iloc[0]), color=colour, ls=style, lw=1.0, zorder=3)
            ax.annotate(name, (float(value.iloc[0]), -0.9), rotation=90, fontsize=6.5,
                        color=colour, ha="right", va="bottom", annotation_clip=False)
    for i, value in enumerate(frame["mae"]):
        ax.annotate(f"{value:.3f}", (value, i), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6.5, color="0.25")
    ax.set_xlabel("one-shot macro MAE (log units) — lower is better")
    ax.set_title("Figure 4 — first-experiment policies against the one-shot oracle",
                 fontsize=10, pad=18)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# --------------------------------------------------------------------------- #
# 5 — adaptation curve
# --------------------------------------------------------------------------- #

GEN8_REFERENCE = {0: 1.0605, 1: 0.6674, 2: 0.5879, 3: 0.5230, 5: 0.4743}


def figure_adaptation(curve: pd.DataFrame, out: Path) -> Path | None:
    plt = _style()
    if curve is None or curve.empty:
        print("  (skipping adaptation curve: no data)")
        return None
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ks = sorted(GEN8_REFERENCE)
    ax.plot(ks, [GEN8_REFERENCE[k] for k in ks], "o--", color="0.35", lw=1.6,
            label="gen8 best deployable")
    palette = ["#2b6cb0", "#2f855a", "#805ad5", "#dd6b20"]
    for i, (arm, block) in enumerate(curve.groupby("arm", sort=True)):
        block = block.sort_values("k")
        ax.plot(block["k"], block["mae"], "s-", lw=1.6, ms=5,
                color=palette[i % len(palette)], label=arm)
    ax.set_xlabel("measurements of the new ligand (k)")
    ax.set_ylabel("macro MAE, common cohort (log units)")
    ax.set_xticks(ks)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Figure 5 — adaptation curve", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# --------------------------------------------------------------------------- #
# 6 — error decomposition
# --------------------------------------------------------------------------- #

def figure_decomposition(chain: pd.DataFrame, out: Path) -> Path | None:
    """One panel per k — the chains are separate walks and must not be joined.

    The first version drew all of them as one line, which produced a meaningless
    jump where the k = 1 chain ended and the k = 2 chain began.
    """
    plt = _style()
    if chain is None or chain.empty:
        print("  (skipping decomposition: no attribution chain)")
        return None
    ks = sorted(chain["k"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 4.4), squeeze=False,
                             sharey=False)
    short = {"OLD GLOBAL + CENTRAL + OLD ADAPTER": "old model\ncentral\noffset",
             "NEW GLOBAL + CENTRAL + OLD ADAPTER": "NEW model\ncentral\noffset",
             "OLD GLOBAL + LEARNED ACQ + OLD ADAPTER": "old model\nLEARNED\noffset",
             "NEW GLOBAL + LEARNED ACQ + OLD ADAPTER": "NEW model\nLEARNED\noffset",
             "NEW GLOBAL + LEARNED ACQ + NEW ADAPTER": "NEW model\nLEARNED\nSERIES"}
    for column, k in enumerate(ks):
        ax = axes[0][column]
        frame = chain[chain["k"] == k].reset_index(drop=True)
        ax.plot(range(len(frame)), frame["mae"], "o-", color="#2b6cb0", lw=1.8, ms=7)
        for i, row in enumerate(frame.itertuples()):
            delta = "" if i == 0 else f"\n{frame['mae'].iloc[i - 1] - row.mae:+.3f}"
            colour = ("0.2" if i == 0 else
                      "#2f855a" if frame["mae"].iloc[i - 1] > row.mae else "#c53030")
            ax.annotate(f"{row.mae:.3f}{delta}", (i, row.mae), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=8, color=colour)
        ax.set_xticks(range(len(frame)))
        ax.set_xticklabels([short.get(s, s) for s in frame["step"]], fontsize=7.5)
        ax.set_xlim(-0.5, len(frame) - 0.5)
        ax.margins(y=0.22)
        ax.set_title(f"k = {k}", fontsize=10)
        if column == 0:
            ax.set_ylabel("macro MAE, common cohort (log units)")
    fig.suptitle("Figure 6 — what each component bought, one change at a time",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=Path, default=RUN / "shape")
    parser.add_argument("--oof", type=Path, nargs="*", default=None)
    parser.add_argument("--acquisition", type=Path, default=RUN / "acquisition")
    parser.add_argument("--frontier", type=Path, default=RUN / "frontier")
    parser.add_argument("--reference", default="REC_ecfp_plus_recovered")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    written = []

    curve_path = args.shape / "curve_shape.parquet"
    if curve_path.exists():
        curves = pd.read_parquet(curve_path)
        models = sorted(curves["model"].unique())
        candidate = args.candidate or next(
            (m for m in models if m.startswith("GEN9_") and "ROW_ONLY" not in m), None)
        ordered = [m for m in (args.reference, "GEN9_A0_ROW_ONLY", candidate)
                   if m and m in models]
        ordered += [m for m in models if m not in ordered]
        print("figure 1/2 — slope and span recovery")
        written.append(figure_slope_recovery(curves, ordered[:4], args.out / "fig1_slope_recovery.png"))
        written.append(figure_slope_recovery(curves, ordered[:4], args.out / "fig2_span_recovery.png",
                                             column="span", label="span", limit=(0.0, 6.0)))
        if args.oof:
            oof = pd.concat([pd.read_parquet(p) for p in args.oof], ignore_index=True)
            membership = pd.read_parquet(MEMBERSHIP)
            classes = pd.read_csv(FAILURE_CLASSES) if FAILURE_CLASSES.exists() else pd.DataFrame()
            picks = pick_examples(curves, classes, reference=args.reference,
                                  candidate=candidate or ordered[-1])
            print("figure 3 — response surfaces")
            written.append(figure_response_surfaces(oof, membership, picks,
                                                    ordered[:4], args.out / "fig3_response_surfaces.png"))
    else:
        print(f"(no {curve_path}; skipping figures 1-3)")

    summary_path = args.acquisition / "primary_summary.csv"
    print("figure 4 — acquisition regret")
    written.append(figure_acquisition_regret(
        pd.read_csv(summary_path) if summary_path.exists() else None,
        args.out / "fig4_acquisition_regret.png"))

    curve_path = args.frontier / "adaptation_curve.csv"
    print("figure 5 — adaptation curve")
    written.append(figure_adaptation(
        pd.read_csv(curve_path) if curve_path.exists() else None,
        args.out / "fig5_adaptation.png"))

    chain_path = args.frontier / "attribution_chain.csv"
    print("figure 6 — error decomposition")
    written.append(figure_decomposition(
        pd.read_csv(chain_path) if chain_path.exists() else None,
        args.out / "fig6_decomposition.png"))

    (args.out / "figures.json").write_text(json.dumps(
        {"written": [str(p) for p in written if p]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

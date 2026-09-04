#!/usr/bin/env python
"""Figure 5 — training coverage, not model capacity, is the binding constraint.

Refined from ``figures/scripts/plot_fig5_generalization.py``.  The numbers, cohorts,
seeds, metrics and uncertainty methods are unchanged; this pass restructures the figure
so that the *controlled* coverage experiment leads and the *observational* distance
split is read last, as recommended in ``figures/FINAL_FIGURE_REPORT.md`` §4 and
``figures/FIGURE_AUDIT.md`` §5.

Panel order in the refined figure (the original order is given in brackets):

A [was B]  gen6 Experiment A levels.  Byte-identical held-out rows, identical learner and
           folds; only the *training* mask differs.  ``BASE`` trains on extractants with
           >= 10 condition cells ("restricted coverage"), ``EXPANDED`` on those with >= 3
           ("full coverage"), ``EXPANDED_SHUFFLED`` keeps the added rows and permutes
           their targets among themselves.
           Source ``runs/gen6_expA_5seed/hard_chemistry_metrics.csv``.
B [was C]  the same experiment's paired improvements with chemotype-block BCa 95 % CIs.
           Source ``runs/gen6_expA_5seed/contrast_summary.csv``.
C [was A]  final pipeline, 99 held-out extractants split into terciles of nearest-
           training-neighbour Tanimoto, macro MAE at k = 0/1/2/3/5.
           Source ``runs/gen10_final/final_locked/kshot_detail.parquet`` via
           ``figures/derived/kshot_per_ligand.csv``.  The tercile definition is gen10's
           own (``scripts/gen10_budget_simulation.py``); the script reproduces all 15
           numbers of ``runs/gen10_final/budget_simulation/marginal_gains.csv`` as a
           check and refuses to draw if they do not match.

Run from anywhere:
    python figure_refinement/ml_separator/figure_05_training_coverage/figure_script.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import mlsep_data as D                               # noqa: E402
import pubstyle as PS                                # noqa: E402

# --------------------------------------------------------------------------- #
# Frozen recipe identifiers.  These are repository arm names and never reach the
# artwork; every drawn label is the reader-facing name defined below.
# --------------------------------------------------------------------------- #
MODEL = "GEN9_SHAPE_RECOMPOSED"
FEATURES = "MC_lig2d_ext_massaction"                 # gen6's champion feature set
PIPELINE = {0: ("ZERO_SHOT_REF", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
            2: ("SERIES_ML", "CENTRAL_THEN_SPREAD"), 3: ("SERIES_ML", "CENTRAL_THEN_SPREAD"),
            5: ("SERIES_ML", "CENTRAL_THEN_SPREAD")}
GEN9_RULE = {0: ("ZERO_SHOT_REF", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
             2: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD"),
             3: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD"),
             5: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD")}
TERCILES = ["near", "mid", "far"]
TERCILE_STYLE = {"near": (PS.SKY, "o"), "mid": (PS.BLUE, "s"), "far": (PS.VERMILLION, "^")}

#: repository arm -> (pubstyle method key, legend text short enough for the reserved band)
ARMS = [("BASE", "restricted", "Restricted coverage"),
        ("EXPANDED", "full", "Full coverage"),
        ("EXPANDED_SHUFFLED", "shuffled", "Shuffled-target control")]
ENDPOINTS = [("all", "all"), ("nn<0.6", "$T<0.6$"), ("nn<0.4", "$T<0.4$")]

#: forest rows: (comparison, endpoint, tick label, pubstyle key of the arm being judged)
CONTRASTS = [("EXPANDED_vs_BASE", "all", "all\nextractants", "full"),
             ("EXPANDED_vs_BASE", "nn<0.6", "$T<0.6$", "full"),
             ("EXPANDED_vs_BASE", "nn<0.4", "$T<0.4$", "full"),
             ("EXPANDED_ROWMATCHED_vs_BASE", "all", "row-count\ncontrol", "rowmatched"),
             ("EXPANDED_vs_EXPANDED_SHUFFLED", "all", "shuffled\ncontrol", "shuffled")]


# --------------------------------------------------------------------------- #
# Data — identical to the original script
# --------------------------------------------------------------------------- #
def adaptation_curves(per_ligand: pd.DataFrame, rule: dict) -> pd.DataFrame:
    parts = []
    for k, (adapter, policy) in rule.items():
        block = per_ligand[(per_ligand.global_model == MODEL) & (per_ligand.adapter == adapter)
                           & (per_ligand.policy == policy) & (per_ligand.k == k)]
        parts.append(block.set_index("extractant")["mae"].rename(f"m{k}"))
    curves = pd.concat(parts, axis=1).dropna()
    meta = per_ligand.drop_duplicates("extractant").set_index("extractant")[
        ["tanimoto_cluster", "nn_train_tanimoto"]]
    curves = curves.join(meta, how="left")
    curves["tercile"] = pd.qcut(1.0 - curves.nn_train_tanimoto, 3, labels=TERCILES)
    return curves


def verify_marginal_gains(curves: pd.DataFrame) -> pd.DataFrame:
    """Reproduce runs/gen10_final/budget_simulation/marginal_gains.csv (gen9 adapter rule)."""
    published = pd.read_csv(D.run("gen10_final/budget_simulation/marginal_gains.csv"),
                            index_col=0)
    gains = pd.DataFrame({"first_point": curves.m0 - curves.m1,
                          "second_point": curves.m1 - curves.m2,
                          "third_point": curves.m2 - curves.m3,
                          "fourth_and_fifth_each": (curves.m3 - curves.m5) / 2.0,
                          "m0": curves.m0, "tercile": curves.tercile})
    mine = gains.groupby("tercile", observed=True).mean(numeric_only=True)
    rows = []
    for tercile in TERCILES:
        for column in ["first_point", "second_point", "third_point",
                       "fourth_and_fifth_each", "m0"]:
            ref = float(published.loc[tercile, column])
            got = float(mine.loc[tercile, column])
            rows.append({"tercile": tercile, "quantity": column, "published": ref,
                         "reproduced": got, "abs_delta": abs(ref - got),
                         "pass": bool(abs(ref - got) < 1e-9)})
    return pd.DataFrame(rows)


def tercile_contrast(curves: pd.DataFrame, column: str, replicates: int = 5000,
                     seed: int = 8675309) -> dict:
    """far - near, resampling chemotypes independently within each tercile.

    The two terciles hold different extractants, so this cannot be paired; it is an
    unpaired block bootstrap and is reported as such.  Same seed and same replicate
    count as the published figure.
    """
    rng = np.random.default_rng(seed)
    blocks = {t: curves[curves.tercile == t] for t in ("near", "far")}
    names = {t: sorted(b.tanimoto_cluster.unique()) for t, b in blocks.items()}
    draws = np.empty(replicates)
    for i in range(replicates):
        means = {}
        for t, b in blocks.items():
            pick = rng.choice(names[t], size=len(names[t]), replace=True)
            idx = np.concatenate([b.index[b.tanimoto_cluster == n].to_numpy() for n in pick])
            means[t] = b.loc[idx, column].mean()
        draws[i] = means["far"] - means["near"]
    point = float(blocks["far"][column].mean() - blocks["near"][column].mean())
    return {"metric": column, "point": point,
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "n_chemotypes_near": len(names["near"]), "n_chemotypes_far": len(names["far"])}


def block_band(fig, axes, letters, colour: str = "#F0F0F0") -> Rectangle:
    """A pale band behind a group of panels, sized from what is actually drawn.

    Panels A and B are one controlled experiment and panel C is a different, weaker kind
    of evidence; the band is the only cue that survives being read at 180 mm, so it is
    measured from the panels' tight bounding boxes (which already include tick labels,
    axis labels and straplines) plus their panel letters, rather than hand-placed.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [ax.get_tightbbox(renderer).transformed(inv) for ax in axes]
    boxes += [t.get_window_extent(renderer).transformed(inv) for t in letters]
    x0 = max(min(b.x0 for b in boxes) - 0.011, 0.001)
    x1 = min(max(b.x1 for b in boxes) + 0.011, 0.999)
    y0 = max(min(b.y0 for b in boxes) - 0.030, 0.001)
    y1 = min(max(b.y1 for b in boxes) + 0.012, 0.999)
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
                     facecolor=colour, edgecolor="none", zorder=-5)
    fig.add_artist(rect)
    for ax in axes:
        ax.patch.set_alpha(0.0)
    return rect


def main() -> int:
    PS.apply()
    per_ligand = D.kshot_per_ligand()
    curves = adaptation_curves(per_ligand, PIPELINE)
    checks = verify_marginal_gains(adaptation_curves(per_ligand, GEN9_RULE))
    if not checks["pass"].all():
        print(checks.to_string())
        raise SystemExit("distance-tercile reproduction failed — figure not drawn")

    fig = plt.figure(figsize=(PS.W2, 3.45))
    # The top 5 % is reserved so the panel letters, and the band that groups A with B,
    # sit inside the declared canvas instead of forcing a tight-bbox expansion.
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.945), w_pad=0.045)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.00, 1.17, 1.13])
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ------------------------------------------------------------------ A ------
    # Controlled experiment, levels.  Same held-out rows in every bar.
    hard = pd.read_csv(D.run("gen6_expA_5seed/hard_chemistry_metrics.csv"))
    hard = hard[hard.feature_set == FEATURES]
    x = np.arange(len(ENDPOINTS))
    width = 0.26
    panelA = []
    for j, (arm, key, legend_text) in enumerate(ARMS):
        values, lo_err, hi_err = [], [], []
        for endpoint, _ in ENDPOINTS:
            block = hard[(hard.arm == arm) & (hard.endpoint == endpoint)]
            per_seed = block.groupby("split_seed")["macro_mae"].mean()
            values.append(float(per_seed.mean()))
            lo_err.append(float(per_seed.mean() - per_seed.min()))
            hi_err.append(float(per_seed.max() - per_seed.mean()))
            panelA.append({"arm": arm, "reader_name": legend_text, "endpoint": endpoint,
                           "macro_mae": values[-1], "seed_min": float(per_seed.min()),
                           "seed_max": float(per_seed.max()),
                           "n_ligands": float(block.n_ligands.mean()),
                           "n_seeds": int(len(per_seed))})
        pos = x + (j - 1) * width
        axA.bar(pos, values, width=width, color=PS.colour(key), linewidth=0,
                label=legend_text, zorder=2)
        # capthick is set explicitly: the shared style sets markeredgewidth to 0, which
        # otherwise makes the caps invisible and leaves the whisker ends undefined.
        axA.errorbar(pos, values, yerr=np.array([lo_err, hi_err]), fmt="none",
                     ecolor=PS.INK, elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=3)
    # n comes from the data, and the arms must agree on it: the whole design of the
    # experiment is that the held-out rows are byte-identical across arms, so an endpoint
    # whose n depends on the arm would mean the comparison is void.
    n_by_endpoint = {}
    for record in panelA:
        n_by_endpoint.setdefault(record["endpoint"], set()).add(round(record["n_ligands"], 6))
    if any(len(v) != 1 for v in n_by_endpoint.values()):
        raise SystemExit(f"held-out extractant counts differ between arms: {n_by_endpoint}")
    axA.set_xticks(x)
    axA.set_xticklabels([f"{lab}\n$n$={next(iter(n_by_endpoint[key])):.0f}"
                         for key, lab in ENDPOINTS])
    axA.set_xlim(-0.55, 2.55)
    # Headroom reserved for the key: the tallest bar plus its seed range reaches 1.67,
    # so everything above 1.75 is guaranteed empty and the key cannot cover a bar.
    axA.set_ylim(0, 2.45)
    axA.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    # The spine stops where the ticks stop; the strip above is the reserved key area and
    # a spine running through it would read as an axis that carries data.
    axA.spines["left"].set_bounds(0.0, 2.0)
    axA.set_xlabel("held-out subset by Tanimoto $T$")
    axA.set_ylabel("macro MAE (log$_{10}$ $D$)")
    PS.strapline(axA, "controlled: training set varied")
    handles, labels = axA.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=PS.INK, linewidth=0.9))
    labels.append("range over 5 split seeds")
    axA.legend(handles, labels, loc="upper left", fontsize=PS.BASE - 0.7,
               handlelength=1.3, handletextpad=0.5, labelspacing=0.35,
               borderaxespad=0.25)

    # ------------------------------------------------------------------ B ------
    # Same experiment, paired.  One row per pre-registered contrast.
    contrasts = pd.read_csv(D.run("gen6_expA_5seed/contrast_summary.csv"))
    contrasts = contrasts[(contrasts.feature_set == FEATURES) & (contrasts.statistic == "mae")]
    panelB, ys, points, los, his, colours = [], [], [], [], [], []
    for i, (comparison, endpoint, tick, key) in enumerate(CONTRASTS):
        row = contrasts[(contrasts.comparison == comparison)
                        & (contrasts.endpoint == endpoint)]
        if row.empty:
            raise SystemExit(f"missing contrast {comparison}@{endpoint}")
        row = row.iloc[0]
        ys.append(len(CONTRASTS) - 1 - i)
        points.append(float(row.pooled_point_delta))
        los.append(float(row.bca_low))
        his.append(float(row.bca_high))
        colours.append(PS.colour(key))
        panelB.append({"comparison": comparison, "endpoint": endpoint,
                       "reader_row": " ".join(tick.split("\n")),
                       "point": float(row.pooled_point_delta),
                       "bca_low": float(row.bca_low), "bca_high": float(row.bca_high),
                       "block_macro_delta": float(row.block_macro_delta),
                       "units_improved": int(row.pooled_units_improved),
                       "units_total": int(row.pooled_units_total),
                       "seeds_positive": int(row.seeds_positive),
                       "n_seeds": int(row.n_seeds)})
    # The plot region ends at 0.70 (the widest interval reaches 0.685); the strip to the
    # right of it carries no interval and holds the count column.
    x_lo, x_plot_hi, x_hi, x_count = -0.075, 0.70, 1.02, 1.00
    axB.axvline(0, color=PS.INK, linewidth=0.7, zorder=1)
    axB.errorbar(points, ys, xerr=[np.array(points) - np.array(los),
                                   np.array(his) - np.array(points)],
                 fmt="none", ecolor=PS.INK, elinewidth=0.9, capsize=2.0, capthick=0.9,
                 zorder=2)
    axB.scatter(points, ys, s=26, c=colours, edgecolor="white", linewidth=0.6, zorder=3)
    # The two negative controls answer a different question from the three subset rows,
    # so they are separated by a rule rather than by position alone.  The rule stops
    # where the plotting region stops, not where the axes stops.
    axB.plot([x_lo, x_plot_hi], [1.5, 1.5], color=PS.GREY, linewidth=0.6,
             linestyle=(0, (2, 2)), zorder=0)
    axB.set_yticks(ys)
    axB.set_yticklabels([tick for _, _, tick, _ in CONTRASTS])
    # The rows are categories, not a scale, and a left spine 0.075 units from the zero
    # reference line reads as a second, meaningless vertical rule.
    axB.spines["left"].set_visible(False)
    axB.tick_params(axis="y", length=0, pad=4)
    axB.set_ylim(-0.60, 4.70)
    axB.set_xlim(x_lo, x_hi)
    axB.set_xticks([0.0, 0.2, 0.4, 0.6])
    axB.spines["bottom"].set_bounds(x_lo, x_plot_hi)
    axB.set_xlabel("macro MAE removed (log$_{10}$ $D$)",
                   x=((x_lo + x_plot_hi) / 2 - x_lo) / (x_hi - x_lo))
    PS.strapline(axB, "controlled: paired gain")
    # How many independent scoring units moved, not just the mean: the reviewer question
    # "is one cluster carrying this?" is answered in the panel rather than the caption.
    axB.text(x_count, 4.42, "clusters\nimproved", ha="right", va="bottom",
             fontsize=PS.BASE - 1.2, color=PS.GREY, linespacing=1.2)
    for y, row in zip(ys, panelB):
        axB.text(x_count, y, f"{row['units_improved']}/{row['units_total']}",
                 ha="right", va="center", fontsize=PS.BASE - 1.0, color=PS.INK)
    # The strip just below the separating rule holds no marker and no interval, so it is
    # the one place a group tag can head the two control rows without a leader line.
    axB.text(x_plot_hi, 1.40, "controls, all extractants", ha="right", va="top",
             fontsize=PS.BASE - 1.0, color=PS.GREY)

    # ------------------------------------------------------------------ C ------
    # Observational split of a different (deployed) model — demoted to last.
    ks = sorted(PIPELINE)
    panelC = []
    # All three terciles are evaluated at the same k, so their intervals would be drawn
    # on top of one another.  The markers stay on the integer k; only the interval bars
    # are offset, by 1.5 % of the axis, so each one can be attributed to its tercile.
    dodge = {"far": -0.09, "mid": 0.0, "near": 0.09}
    for tercile in TERCILES[::-1]:                    # far first: legend order = curve order
        block = curves[curves.tercile == tercile]
        point, lo, hi = [], [], []
        for k in ks:
            p, l, h = D.block_bootstrap_mean(block[f"m{k}"], block.tanimoto_cluster)
            point.append(p)
            lo.append(l)
            hi.append(h)
            panelC.append({"tercile": tercile, "k": k, "macro_mae": p, "ci95_low": l,
                           "ci95_high": h, "n_extractants": int(len(block))})
        colour, marker = TERCILE_STYLE[tercile]
        nn = block.nn_train_tanimoto
        axC.plot(ks, point, color=colour, marker=marker, markersize=3.4, linewidth=1.3,
                 label=f"{tercile}  $T$ {nn.min():.2f}–{nn.max():.2f}, $n$={len(block)}")
        axC.vlines(np.asarray(ks, dtype=float) + dodge[tercile], lo, hi, color=colour,
                   linewidth=0.9, alpha=0.75)
    axC.set_xticks(ks)
    axC.set_xlim(-0.35, 5.35)
    # The lowest interval bound in the panel is 0.193 (mid tercile, k = 5) and the
    # highest 1.464 (far, k = 0); the published render cut the axis at 0.28 and clipped
    # three of the mid tercile's intervals, which makes an interval look open-ended.
    # The whole of every interval is inside these limits, and the strip above 1.55 —
    # empty because every curve has fallen away by k = 1 — carries the key and the caveat.
    axC.set_ylim(0.16, 2.02)
    axC.set_yticks(np.arange(0.2, 1.81, 0.2))
    # As in panel A, the spine stops at the last tick so that the reserved key strip is
    # not read as part of the scale; the bottom corner stays joined.
    axC.spines["left"].set_bounds(0.16, 1.8)
    axC.set_xlabel("measured support points, $k$")
    axC.set_ylabel("macro MAE (log$_{10}$ $D$)")
    PS.strapline(axC, "observational, not an intervention")
    axC.legend(loc="upper right", fontsize=PS.BASE - 0.7, handlelength=1.5,
               handletextpad=0.5, labelspacing=0.3, borderaxespad=0.25)
    contrast = tercile_contrast(curves, "m0")
    caveat = (f"far $-$ near at $k=0$: {contrast['point']:+.2f}\n"
              f"95 % CI [{contrast['ci95_low']:+.2f}, {contrast['ci95_high']:+.2f}]"
              " includes 0").replace("-", "−")
    axC.text(0.975, 0.635, caveat,
             transform=axC.transAxes, ha="right", va="top", fontsize=PS.BASE - 1.0,
             color=PS.INK, linespacing=1.35,
             bbox={"facecolor": "white", "edgecolor": PS.PALE, "linewidth": 0.6,
                   "boxstyle": "round,pad=0.28"})

    letters = PS.add_panel_letters(fig, [axA, axB, axC])
    block_band(fig, (axA, axB), letters[:2])
    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)

    values = {
        "figure": "figure_05_training_coverage",
        "panel_order_note": "A and B were panels B and C of figures/main/Fig5_generalization.png; "
                            "C was panel A.  Values are unchanged.",
        "panelA": panelA,
        "panelB": panelB,
        "panelAB_cohort": "gen6 Experiment A: 5,248 rows / 152 extractants / 131 ECFP "
                          "clusters / 79 chemotypes; macro = one vote per ECFP cluster; "
                          f"feature set {FEATURES}; 5 split seeds",
        "panelC": panelC,
        "panelC_tercile_contrasts": [tercile_contrast(curves, f"m{k}") for k in ks],
        "panelC_cohort": "99 extractants of the k-shot common cohort, final pipeline",
        "panelC_verification": checks.to_dict("records"),
        "lint": {"ok": report.ok, "violations": [str(v) for v in report.violations]},
    }
    (HERE / "values.json").write_text(json.dumps(values, indent=1))
    print("marginal-gain reproduction:", int(checks["pass"].sum()), "/", len(checks), "pass")
    print(pd.DataFrame(panelA).pivot_table(index="arm", columns="endpoint",
                                           values="macro_mae").round(4).to_string())
    print(pd.DataFrame(panelB)[["comparison", "endpoint", "point", "bca_low",
                                "bca_high"]].round(4).to_string())
    print(pd.DataFrame(panelC).pivot_table(index="tercile", columns="k",
                                           values="macro_mae").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

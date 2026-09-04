#!/usr/bin/env python
"""Figure 6 — which single measurement to make.

Refined from ``figures/scripts/plot_fig6_acquisition.py``.  The numbers, the cohort, the
policy set, the aggregation and the interval method are unchanged; the script still
asserts that it reproduces ``runs/gen10_final/acquisition/realised_summary.csv`` for every
policy before it draws anything.  This pass changes layout, typography, labelling and
information design only — see ``notes.md``.

The one substantive presentation change is that the paired difference against random
choice, which the predecessor drew as an interval re-centred on each policy's own bar in
the level panel, now has a panel of its own (B).  Same statistic, same numbers, an
encoding a reader cannot mistake for a marginal interval on the level.

Run from anywhere:
    python figure_refinement/ml_separator/figure_06_acquisition/figure_script.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl                # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import mlsep_data as D                  # noqa: E402
import pubstyle as PS                   # noqa: E402

#: Only the geometry feature set is on the panel, exactly as in the published figure:
#: the three variants that additionally read the model's own prediction belong to a second
#: feature set and are listed in ``values.json`` instead.
FEATURE_SET = "geometry"

#: Registry identifier -> what the rule actually does, in the reader's language.
#: The registry names are repository identifiers and must not appear in the artwork; the
#: mapping is reproduced verbatim in ``notes.md`` so any bar can be traced back.
#: Every phrasing below is the k = 1 behaviour of the registered policy, because this
#: figure is entirely at k = 1 (``FARTHEST_FROM_EXISTING`` with nothing yet selected is
#: the point farthest from the pool centre, not from a previous measurement).
POLICY_LABEL = {
    "ORACLE":                   "best in hindsight (not deployable)",
    "SURROGATE_ORACLE":         "best on ranker's target (not deployable)",
    "MEDOID":                   "medoid candidate",
    "CENTRAL":                  "nearest the mean condition",
    # The anchor is chosen per fold between the medoid and the centre distance
    # (gen9.acquisition.BlendedAcquisition.anchors), so the label says "a centrality
    # anchor" rather than naming one of the two.
    "LEARNED_BLEND[geometry]":  "learned ranker (blended with centrality)",
    "LEARNED_SCALAR[geometry]": "learned ranker (regression)",
    "MEDIAN_PREDICTION":        "median predicted extraction",
    "LEARNED_RANK[geometry]":   "learned ranker (pairwise)",
    "MAX_CONDITION_COVERAGE":   "smallest covering radius",
    "MIN_GP_DESIGN_VAR":        "lowest design variance",
    "MID_ACID":                 "median acidity",
    "MAX_PREDICTION":           "strongest predicted extraction",
    "MAX_ENSEMBLE_SD":          "most uncertain candidate",
    "RANDOM":                   "random choice",
    "MIN_ENSEMBLE_SD":          "least uncertain candidate",
    "FARTHEST_FROM_EXISTING":   "farthest from the pool centre",
    "MIN_PREDICTION":           "weakest predicted extraction",
}

#: Families, colours and family labels are carried over from the published figure
#: unchanged; only the wording of the legend entries is reader-facing.
FAMILY = {
    "ORACLE": "oracle", "SURROGATE_ORACLE": "oracle",
    "MEDOID": "centrality", "CENTRAL": "centrality", "MIN_GP_DESIGN_VAR": "centrality",
    "MEDIAN_PREDICTION": "centrality", "MAX_CONDITION_COVERAGE": "centrality",
    "MID_ACID": "centrality",
    "MAX_ENSEMBLE_SD": "uncertainty", "MIN_ENSEMBLE_SD": "uncertainty",
    "MAX_PREDICTION": "extreme", "MIN_PREDICTION": "extreme",
    "FARTHEST_FROM_EXISTING": "extreme",
    "RANDOM": "random",
}
FAMILY_COLOUR = {"centrality": PS.VERMILLION, "learned": PS.BLUE,
                 "uncertainty": PS.SKY, "extreme": PS.PURPLE,
                 "random": PS.GREY, "oracle": PS.GREEN}
FAMILY_LABEL = {"centrality": "geometric centrality", "learned": "learned ranker",
                "uncertainty": "model uncertainty", "extreme": "extreme / spread",
                "random": "random choice (reference)",
                "oracle": "oracle — reads the answer, not deployable"}

STRIPE = "#F4F4F4"        # zebra row bands, drawn identically in all three panels
ORACLE_BAND = "#E4F2EC"   # a stronger tint behind the two non-deployable rows


def row_bands(ax, y: np.ndarray, families: list[str]) -> None:
    """Alternating row stripes plus a tinted band on the non-deployable rows.

    The stripes exist so a reader can carry one policy across three panels that share a
    y order but only label it once; the tint is one of three redundant markers on the
    oracle rows (tint, hatch, and the words "not deployable" in panel A).
    """
    for i, (yi, fam) in enumerate(zip(y, families)):
        if fam == "oracle":
            ax.axhspan(yi - 0.5, yi + 0.5, color=ORACLE_BAND, zorder=0, linewidth=0)
        elif i % 2 == 1:
            ax.axhspan(yi - 0.5, yi + 0.5, color=STRIPE, zorder=0, linewidth=0)


def main() -> int:
    PS.apply()
    mpl.rcParams["hatch.linewidth"] = 0.45

    # ---------------------------------------------------------------- data ---
    detail = pd.read_parquet(D.run("gen10_final/acquisition/realised_detail.parquet"))
    detail = detail[detail.feature_set == FEATURE_SET]

    per_ligand = detail.groupby(["policy", "extractant", "tanimoto_cluster"],
                                observed=True).agg(
        mae=("mae", "mean"), harms=("harms", "mean"), zero_shot=("zero_shot", "mean"),
        oracle_mae=("oracle_mae", "mean"),
        gap_recovered=("gap_recovered", "mean")).reset_index()
    summary = per_ligand.groupby("policy").agg(
        mae=("mae", "mean"), frac_harmed=("harms", "mean"),
        n_ligands=("extractant", "nunique")).reset_index()

    published = pd.read_csv(D.run("gen10_final/acquisition/realised_summary.csv"))
    merged = summary.merge(published[["policy", "mae", "frac_harmed", "n_ligands"]],
                           on="policy", suffixes=("_mine", "_published"))
    merged["delta_mae"] = (merged.mae_mine - merged.mae_published).abs()
    merged["delta_harm"] = (merged.frac_harmed_mine - merged.frac_harmed_published).abs()
    ok = bool((merged.delta_mae < 1e-9).all() and (merged.delta_harm < 1e-9).all()
              and (merged.n_ligands_mine == merged.n_ligands_published).all())
    if not ok:
        print(merged.sort_values("delta_mae", ascending=False).head(10).to_string())
        raise SystemExit("acquisition summary did NOT reproduce — figure not drawn")

    long = per_ligand.rename(columns={"policy": "arm"})
    comparisons = {p: ("RANDOM", p) for p in summary.policy if p != "RANDOM"}
    paired = D.paired_chemotype_bootstrap(long, comparisons, arm_column="arm",
                                          seed_column="__none__").set_index("comparison")

    summary["family"] = summary.policy.map(FAMILY).fillna("learned")
    summary = summary.sort_values("mae").reset_index(drop=True)
    zero_shot = float(per_ligand[per_ligand.policy == "RANDOM"].zero_shot.mean())
    n_extractants = int(summary.n_ligands.max())
    n_seeds = int(detail.split_seed.nunique())
    n_repeats = int(detail.repeat.nunique())

    y = np.arange(len(summary))[::-1]
    families = list(summary.family)
    colours = [FAMILY_COLOUR[f] for f in families]
    is_oracle = np.array([f == "oracle" for f in families])
    random_harm = float(summary.loc[summary.policy == "RANDOM", "frac_harmed"].iloc[0])

    # Shared vertical frame.  Every panel uses it, so the three share one y order and one
    # set of row bands; the headroom at top and bottom is reserved for the reference-line
    # labels and the cohort note, which is why it is declared once here rather than being
    # discovered per panel.
    ylim = (float(y.min()) - 2.8, float(y.max()) + 1.6)
    y_rule = (float(y.min()) - 0.6, float(y.max()) + 0.6)   # reference lines span the rows
    y_top_note = float(y.max()) + 1.05          # inside the reserved top headroom
    y_cohort_head = float(y.min()) - 1.15       # inside the reserved bottom headroom
    y_cohort_note = float(y.min()) - 1.25

    # ---------------------------------------------------------------- fig ----
    fig = plt.figure(figsize=(PS.W2, 5.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.24, 1.22, 0.80])
    axA, axB, axC = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                     fig.add_subplot(gs[0, 2]))

    for ax in (axA, axB, axC):
        row_bands(ax, y, families)
        ax.set_ylim(*ylim)
        ax.set_yticks(y)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    # ------------------------------------------------------------------ A ---
    barsA = axA.barh(y, summary.mae, height=0.68, color=colours, linewidth=0, zorder=2)
    for bar, oracle in zip(barsA, is_oracle):
        if oracle:
            bar.set_hatch("////")
            bar.set_edgecolor("white")
            bar.set_linewidth(0.0)
    # Reference lines span the bar rows only, so the reserved headroom above and below
    # stays genuinely empty for the labels that live there.
    axA.vlines(zero_shot, *y_rule, color=PS.INK, linewidth=0.8, linestyle=(0, (4, 2)),
               zorder=3)
    # One value label per bar.  The region to the right of every bar is empty by
    # construction (the longest bar is 0.78, the axis runs to 1.15 and the zero-shot rule
    # is at 0.98), so no label can land on data or on the rule.
    for yi, value in zip(y, summary.mae):
        axA.text(value + 0.022, yi, f"{value:.3f}", va="center", ha="left",
                 fontsize=PS.BASE - 1.0, color=PS.INK, zorder=4)
    axA.set_yticklabels([POLICY_LABEL[p] for p in summary.policy])
    axA.set_xlim(0.0, 1.15)
    axA.set_xticks([0.0, 0.4, 0.8])
    axA.set_xlabel("macro MAE after one\nmeasurement (log$_{10}$ $D$)")
    PS.strapline(axA, "identical candidate pools")
    axA.text(zero_shot - 0.025, y_top_note, f"zero-shot error {zero_shot:.2f}",
             ha="right", va="center", fontsize=PS.BASE - 1.0, color=PS.INK)
    # The cohort headline is set in ink and the qualifiers in grey, because this figure
    # runs on a different cohort from the multi-k figures and that has to be unmissable.
    axA.text(0.01, y_cohort_head, f"cohort: {n_extractants} held-out extractants",
             ha="left", va="bottom", fontsize=PS.BASE - 1.0, color=PS.INK)
    axA.text(0.01, y_cohort_note,
             f"{n_seeds} split seeds $\\times$ {n_repeats} pool draws\n"
             "the multi-$k$ figures use 99 of them",
             ha="left", va="top", fontsize=PS.BASE - 1.0, color=PS.GREY,
             linespacing=1.25)

    # ------------------------------------------------------------------ B ---
    point = np.array([0.0 if p == "RANDOM" else float(paired.loc[p, "point"])
                      for p in summary.policy])
    low = np.array([0.0 if p == "RANDOM" else float(paired.loc[p, "bca_low"])
                    for p in summary.policy])
    high = np.array([0.0 if p == "RANDOM" else float(paired.loc[p, "bca_high"])
                     for p in summary.policy])
    is_random = np.array([p == "RANDOM" for p in summary.policy])
    barsB = axB.barh(y[~is_random], point[~is_random], height=0.68,
                     color=[c for c, r in zip(colours, is_random) if not r],
                     linewidth=0, zorder=2)
    for bar, oracle in zip(barsB, is_oracle[~is_random]):
        if oracle:
            bar.set_hatch("////")
            bar.set_edgecolor("white")
            bar.set_linewidth(0.0)
    axB.errorbar(point[~is_random], y[~is_random],
                 xerr=[point[~is_random] - low[~is_random],
                       high[~is_random] - point[~is_random]],
                 fmt="none", ecolor=PS.INK, elinewidth=0.8, capsize=1.6, zorder=4)
    # A dot on the point estimate: four of the deployable rules sit so close to zero that
    # their bar is a hairline, and without it the row reads as missing rather than null.
    axB.plot(point[~is_random], y[~is_random], linestyle="none", marker="o",
             markersize=1.9, color=PS.INK, zorder=5)
    axB.vlines(0.0, *y_rule, color=PS.INK, linewidth=0.8, zorder=3)
    axB.text(0.008, float(y[is_random][0]), "reference", va="center", ha="left",
             fontsize=PS.BASE - 1.0, color=PS.GREY, zorder=4)
    axB.set_yticklabels([])
    axB.set_xlim(-0.175, 0.325)
    axB.set_xticks([-0.1, 0.0, 0.1, 0.2, 0.3])
    axB.set_xlabel("macro MAE removed vs\nrandom choice (log$_{10}$ $D$)")
    PS.strapline(axB, "paired difference, BCa 95 %")
    axB.text(-0.012, y_top_note, "worse", ha="right", va="center",
             fontsize=PS.BASE - 1.0, color=PS.GREY)
    axB.text(0.012, y_top_note, "better than random", ha="left", va="center",
             fontsize=PS.BASE - 1.0, color=PS.GREY)

    # ------------------------------------------------------------------ C ---
    barsC = axC.barh(y, summary.frac_harmed, height=0.68, color=colours, linewidth=0,
                     zorder=2)
    for bar, oracle in zip(barsC, is_oracle):
        if oracle:
            bar.set_hatch("////")
            bar.set_edgecolor("white")
            bar.set_linewidth(0.0)
    axC.vlines(random_harm, *y_rule, color=PS.GREY, linewidth=0.8, linestyle=(0, (4, 2)),
               zorder=3)
    axC.set_yticklabels([])
    axC.set_xlim(0.0, 0.50)
    axC.set_xticks([0.0, 0.2, 0.4])
    axC.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axC.set_xlabel(f"extractants made\nworse (% of {n_extractants})")
    PS.strapline(axC, "deployment risk")
    axC.text(random_harm - 0.015, y_top_note, "random choice", ha="right", va="center",
             fontsize=PS.BASE - 1.0, color=PS.GREY)

    # The three panels are one table read three ways: same rows, same order, same bands.
    # Asserted rather than assumed, because a reader uses panel A's labels to read B and C.
    for ax in (axB, axC):
        assert np.array_equal(np.sort(ax.get_yticks()), np.sort(axA.get_yticks()))
        assert ax.get_ylim() == axA.get_ylim()

    # ------------------------------------------------------------- legend ---
    order = ["centrality", "learned", "uncertainty", "extreme", "random", "oracle"]
    handles = []
    for fam in order:
        kw = dict(color=FAMILY_COLOUR[fam], linewidth=5.0, solid_capstyle="butt")
        handles.append(plt.Line2D([], [], **kw))
    fig.legend(handles, [FAMILY_LABEL[f] for f in order], loc="outside lower center",
               ncol=3, frameon=False, handlelength=1.5, columnspacing=1.8)

    # Guard, not decoration: this figure's subject is a policy registry, so the one way it
    # can regress is a raw identifier leaking into a tick label or a legend entry.
    drawn = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
    leaked = sorted(k for k in POLICY_LABEL if k.split("[")[0] in drawn)
    assert not leaked, f"registry identifiers reached the artwork: {leaked}"

    PS.add_panel_letters(fig, [axA, axB, axC])
    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)

    # -------------------------------------------------------------- values --
    values = {
        "figure": "figure_06_acquisition",
        "source": "runs/gen10_final/acquisition/realised_detail.parquet",
        "feature_set": FEATURE_SET,
        "cohort": {"name": "C-KSHOT", "n_extractants": n_extractants,
                   "n_seeds": n_seeds, "n_repeats": n_repeats,
                   "zero_shot_macro_mae": zero_shot, "k": 1},
        "reproduces_realised_summary": ok,
        "policy_label_map": POLICY_LABEL,
        "panelA_panelC": [
            {"policy": r.policy, "label": POLICY_LABEL[r.policy], "family": r.family,
             "macro_mae": round(float(r.mae), 6),
             "frac_harmed": round(float(r.frac_harmed), 6),
             "n_extractants": int(r.n_ligands)}
            for r in summary.itertuples()],
        "panelB_paired_vs_random": paired.reset_index().round(6).to_dict("records"),
        "lint": {"ok": report.ok, "violations": [str(v) for v in report.violations]},
    }
    (HERE / "values.json").write_text(json.dumps(values, indent=1))

    print("reproduces realised_summary.csv for all", len(merged), "policies:", ok)
    print(summary.round(4).to_string())
    for name in ["MEDOID", "CENTRAL", "MAX_ENSEMBLE_SD", "FARTHEST_FROM_EXISTING",
                 "ORACLE"]:
        r = paired.loc[name]
        print(f"{name:24s} vs RANDOM  {r['point']:+.4f} "
              f"[{r['bca_low']:+.4f},{r['bca_high']:+.4f}] "
              f"{int(r['units_improved'])}/{int(r['n_units'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

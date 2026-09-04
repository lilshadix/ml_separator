#!/usr/bin/env python
"""Figure 2 — what a handful of measurements buys on a genuinely unseen extractant.

Refined from ``figures/scripts/plot_fig2_fewshot_frontier.py``.  The numbers are
unchanged and remain the ones verified in ``figures/METRIC_AUDIT.md`` (rows 1-11); this
pass changes layout, typography, label wording and legend placement only.

Run from anywhere:
    python figure_refinement/ml_separator/figure_02_few_shot_frontier/figure_script.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import mlsep_data as D                 # noqa: E402
import pubstyle as PS                  # noqa: E402

KS = [0, 1, 2, 3, 5]

#: Each rule is one deterministic deployment recipe, fixed before the run and applied at
#: every k.  Nothing is selected per seed, per extractant or on the test metric.
RULES: dict[str, tuple[str, dict[int, tuple[str, str]]]] = {
    "no_model": ("GEN9_SHAPE_RECOMPOSED", {
        k: ("NO_MODEL", "CENTRAL_THEN_SPREAD") for k in (1, 2, 3, 5)}),
    "baseline": ("REC_ecfp_plus_recovered", {
        0: ("ZERO_SHOT_REF", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
        **{k: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD") for k in (2, 3, 5)}}),
    "recomposed": ("GEN9_SHAPE_RECOMPOSED", {
        0: ("ZERO_SHOT_REF", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
        **{k: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD") for k in (2, 3, 5)}}),
    "pipeline": ("GEN9_SHAPE_RECOMPOSED", {
        0: ("ZERO_SHOT_REF", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
        **{k: ("SERIES_ML", "CENTRAL_THEN_SPREAD") for k in (2, 3, 5)}}),
    "oracle": ("GEN9_SHAPE_RECOMPOSED", {
        k: ("SERIES_ML", "ORACLE[OFFSET_K1]") for k in (1, 2, 3, 5)}),
}
#: Short enough that a five-entry legend fits in the empty corner of panel A.
LEGEND = {
    "no_model": "Measurements only",
    "baseline": "Baseline model",
    "recomposed": "+ shape recomposition",
    "pipeline": "+ series adaptation (final)",
    "oracle": "Oracle points (not deployable)",
}


def rule_frame(per_ligand: pd.DataFrame, key: str) -> pd.DataFrame:
    model, spec = RULES[key]
    parts = []
    for k, (adapter, policy) in spec.items():
        block = per_ligand[(per_ligand.global_model == model)
                           & (per_ligand.adapter == adapter)
                           & (per_ligand.policy == policy) & (per_ligand.k == k)]
        if block.empty:
            raise SystemExit(f"{key}: no rows for k={k} ({model}/{adapter}/{policy})")
        parts.append(block.assign(rule=key)[
            ["rule", "k", "extractant", "tanimoto_cluster", "mae"]])
    return pd.concat(parts, ignore_index=True)


def macro(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, block in frame.groupby("k"):
        point, lo, hi = D.block_bootstrap_mean(block["mae"], block["tanimoto_cluster"])
        rows.append({"k": int(k), "macro_mae": point, "lo": lo, "hi": hi,
                     "n": block["extractant"].nunique()})
    return pd.DataFrame(rows).sort_values("k")


def paired(frame: pd.DataFrame, k_ref: int, k_cand: int) -> dict:
    long = frame[frame.k.isin([k_ref, k_cand])].copy()
    long["arm"] = np.where(long.k == k_ref, "ref", "cand")
    return D.paired_chemotype_bootstrap(
        long, {"d": ("ref", "cand")}, seed_column="__none__").iloc[0].to_dict()


def main() -> int:
    PS.apply()
    per_ligand = D.kshot_per_ligand()
    frames = {key: rule_frame(per_ligand, key) for key in RULES}
    curves = {key: macro(f) for key, f in frames.items()}

    per_seed = D.kshot_per_seed()
    spread = {}
    for key, (model, spec) in RULES.items():
        rows = []
        for k, (adapter, policy) in spec.items():
            b = per_seed[(per_seed.global_model == model) & (per_seed.adapter == adapter)
                         & (per_seed.policy == policy) & (per_seed.k == k)]
            rows.append({"k": int(k), "lo": float(b.mae.min()), "hi": float(b.mae.max())})
        spread[key] = pd.DataFrame(rows).sort_values("k").set_index("k")

    fig = plt.figure(figsize=(PS.W2, 4.65))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.10, 1.00], width_ratios=[1.16, 1.00])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # ---------------------------------------------------------------- A ------
    for key in ("no_model", "baseline", "recomposed", "pipeline", "oracle"):
        c = curves[key]
        s = spread[key]
        kw = PS.m(key, label=LEGEND[key], markersize=3.4)
        axA.plot(c.k, c.macro_mae, **kw)
        axA.vlines(c.k, s.loc[c.k, "lo"], s.loc[c.k, "hi"], color=kw["color"],
                   linewidth=0.9, alpha=0.7)
    pipe = curves["pipeline"].set_index("k")["macro_mae"]
    # Value labels go into empty space, never on top of a line: above the k = 0 point
    # (the lines fall away steeply to its right) and below the k = 1 and k = 5 points
    # (nothing is drawn under the frontier).
    for k, dx, dy, ha, va in [(0, 4, 7, "left", "bottom"), (1, 0, -11, "center", "top"),
                              (5, 7, 0, "left", "center")]:
        axA.annotate(f"{pipe[k]:.3f}", (k, pipe[k]), textcoords="offset points",
                     xytext=(dx, dy), fontsize=PS.BASE - 1.2, ha=ha, va=va,
                     color=PS.colour("pipeline"))
    axA.set_xticks(KS)
    axA.set_xlim(-0.3, 5.95)   # right margin holds the end-point value label
    axA.set_ylim(0.36, 1.30)
    axA.set_xlabel("measured support points from the unseen extractant, $k$")
    axA.set_ylabel("macro MAE (log$_{10}$ $D$)")
    axA.legend(loc="upper right", bbox_to_anchor=(1.02, 1.03), ncol=1)

    # ---------------------------------------------------------------- B ------
    steps = [(0, 1, "1st"), (1, 2, "2nd"), (2, 3, "3rd"), (3, 5, "4th–5th")]
    bars, lo, hi, improved = [], [], [], []
    for k_ref, k_cand, _ in steps:
        d = paired(frames["pipeline"], k_ref, k_cand)
        scale = 1.0 / (k_cand - k_ref)
        bars.append(d["point"] * scale)
        lo.append(d["bca_low"] * scale)
        hi.append(d["bca_high"] * scale)
        improved.append(d["units_improved"] / d["n_units"])
    x = np.arange(len(bars))
    axB.bar(x, bars, width=0.6, color=PS.colour("pipeline"), linewidth=0)
    axB.errorbar(x, bars, yerr=[np.array(bars) - np.array(lo), np.array(hi) - np.array(bars)],
                 fmt="none", ecolor=PS.INK, elinewidth=0.8, capsize=2.0)
    for xi, b in enumerate(bars):
        axB.text(xi, b + 0.028, f"{b:.3f}", ha="center", fontsize=PS.BASE - 1.0)
    axB.set_xticks(x)
    axB.set_xticklabels([f"{s[2]}\n{f:.0%}" for s, f in zip(steps, improved)])
    axB.set_xlabel("measurement added / extractants it improves")
    axB.set_ylabel("macro MAE removed\nper measurement")
    axB.set_ylim(0, 0.64)

    # ---------------------------------------------------------------- C ------
    contrasts = [("recomposed", "baseline", "recomposed", "shape recomposition"),
                 ("pipeline", "recomposed", "pipeline", "series adaptation")]
    comp = []
    for cand_key, ref_key, style_key, nice in contrasts:
        ref = frames[ref_key].rename(columns={"mae": "mae_ref"})
        cand = frames[cand_key].rename(columns={"mae": "mae_cand"})
        merged = ref.merge(cand, on=["k", "extractant", "tanimoto_cluster"])
        for k, block in merged.groupby("k"):
            long = pd.concat([block.assign(arm="ref", mae=block.mae_ref),
                              block.assign(arm="cand", mae=block.mae_cand)],
                             ignore_index=True)
            r = D.paired_chemotype_bootstrap(long, {"d": ("ref", "cand")},
                                             seed_column="__none__").iloc[0]
            comp.append({"component": nice, "style": style_key, "k": int(k),
                         "point": r["point"], "lo": r["bca_low"], "hi": r["bca_high"],
                         "improved": int(r["units_improved"]), "n": int(r["n_units"])})
    comp = pd.DataFrame(comp)
    width = 0.36
    for offset, (nice, style_key) in zip((-width / 2, width / 2),
                                         [("shape recomposition", "recomposed"),
                                          ("series adaptation", "pipeline")]):
        block = comp[comp.component == nice].sort_values("k").reset_index(drop=True)
        pos = np.arange(len(block)) + offset
        axC.bar(pos, block.point, width=width, color=PS.colour(style_key),
                linewidth=0, label=nice)
        axC.errorbar(pos, block.point,
                     yerr=[block.point - block.lo, block.hi - block.point],
                     fmt="none", ecolor=PS.INK, elinewidth=0.7, capsize=1.7)
    axC.axhline(0, color=PS.INK, linewidth=0.7)
    axC.set_xticks(np.arange(len(KS)))
    axC.set_xticklabels(KS)
    axC.set_xlabel("$k$")
    axC.set_ylabel("macro MAE removed\nvs the previous stage")
    axC.set_ylim(-0.008, 0.062)
    axC.legend(loc="upper left", ncol=1)
    axC.text(0.985, 0.985, "series adaptation is the identity at $k\\leq1$",
             transform=axC.transAxes, ha="right", va="top",
             fontsize=PS.BASE - 1.5, color=PS.GREY)

    # ---------------------------------------------------------------- D ------
    wide = frames["pipeline"].pivot_table(index="extractant", columns="k", values="mae")
    zero, one = wide[0].to_numpy(), wide[1].to_numpy()
    better = int((one < zero).sum())
    axD.scatter(zero, one, s=11, facecolor="none", edgecolor=PS.colour("pipeline"),
                linewidth=0.7, alpha=0.85)
    lim = (0.0, 3.35)
    axD.plot(lim, lim, color=PS.INK, linewidth=0.8, linestyle="--", zorder=0)
    axD.set_xlim(*lim)
    axD.set_ylim(*lim)
    axD.set_aspect("equal")
    axD.set_xlabel("zero-shot MAE per extractant")
    axD.set_ylabel("MAE after one measurement")
    axD.text(0.04, 0.97, f"{better} of {len(zero)} improved\n"
                         f"median $\\Delta$ = {np.median(zero - one):+.2f}\n"
                         f"{len(zero) - better} made worse",
             transform=axD.transAxes, va="top", fontsize=PS.BASE - 1.0)

    PS.add_panel_letters(fig, [axA, axB, axC, axD])
    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)

    values = {
        "figure": "figure_02_few_shot_frontier",
        "cohort": json.loads((D.DERIVED / "kshot_cohort.json").read_text()),
        "panelA": {k: c.round(6).to_dict("records") for k, c in curves.items()},
        "panelB": [{"step": s[2], "gain_per_measurement": b, "bca_low": l,
                    "bca_high": h, "frac_improved": f}
                   for s, b, l, h, f in zip(steps, bars, lo, hi, improved)],
        "panelC": comp.round(6).to_dict("records"),
        "panelD": {"n": int(len(zero)), "improved": better,
                   "median_delta": float(np.median(zero - one))},
        "lint": {"ok": report.ok, "violations": [str(v) for v in report.violations]},
    }
    (HERE / "values.json").write_text(json.dumps(values, indent=1))
    for key in ("no_model", "baseline", "recomposed", "pipeline", "oracle"):
        print(f"{key:12s}", [round(v, 4) for v in curves[key].macro_mae.tolist()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

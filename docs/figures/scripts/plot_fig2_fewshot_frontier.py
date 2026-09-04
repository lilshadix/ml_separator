"""Figure 2 — what a handful of measurements buys.

Run from the repository root:  ``.venv/bin/python figures/scripts/plot_fig2_fewshot_frontier.py``
Requires ``figures/derived/kshot_per_ligand.csv`` (produced by ``prepare_kshot_tables.py``;
this script calls it automatically if the file is missing).

Every number is computed from ``runs/gen10_final/final_locked/kshot_detail.parquet``.
The four deployment rules are pre-specified, fixed for all k, and evaluated on identical
held-out ligands and identical support/query draws, so all comparisons are paired.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402
import _stats  # noqa: E402
import _style  # noqa: E402

KS = [0, 1, 2, 3, 5]

#: key -> (global model, {k: (adapter, policy)}).  Each rule is one deterministic
#: deployment recipe; nothing is chosen per seed, per ligand or per k on the test metric.
RULES: dict[str, tuple[str, dict[int, tuple[str, str]]]] = {
    "no_model": ("GEN9_SHAPE_RECOMPOSED", {
        1: ("NO_MODEL", "CENTRAL_THEN_SPREAD"), 2: ("NO_MODEL", "CENTRAL_THEN_SPREAD"),
        3: ("NO_MODEL", "CENTRAL_THEN_SPREAD"), 5: ("NO_MODEL", "CENTRAL_THEN_SPREAD")}),
    "frozen": ("REC_ecfp_plus_recovered", {
        0: ("ZERO_SHOT_REF", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
        2: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD"), 3: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD"),
        5: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD")}),
    "recomposed": ("GEN9_SHAPE_RECOMPOSED", {
        0: ("ZERO_SHOT_REF", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
        2: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD"), 3: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD"),
        5: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD")}),
    "pipeline": ("GEN9_SHAPE_RECOMPOSED", {
        0: ("ZERO_SHOT_REF", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
        2: ("SERIES_ML", "CENTRAL_THEN_SPREAD"), 3: ("SERIES_ML", "CENTRAL_THEN_SPREAD"),
        5: ("SERIES_ML", "CENTRAL_THEN_SPREAD")}),
    "oracle": ("GEN9_SHAPE_RECOMPOSED", {
        1: ("SERIES_ML", "ORACLE[OFFSET_K1]"),
        2: ("SERIES_ML", "ORACLE[OFFSET_K1]"), 3: ("SERIES_ML", "ORACLE[OFFSET_K1]"),
        5: ("SERIES_ML", "ORACLE[OFFSET_K1]")}),
}
LABELS = {
    "no_model": "Measurements only (no model)",
    "frozen": "Baseline model",
    "recomposed": "+ shape recomposition",
    "pipeline": "+ series-local adaptation (final)",
    "oracle": "Oracle support points for a level-only fit (not deployable)",
}


def load() -> pd.DataFrame:
    path = _paths.DERIVED / "kshot_per_ligand.csv"
    if not path.exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_kshot_tables.py"))],
                       check=True)
    return pd.read_csv(path)


def rule_frame(per_ligand: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per-ligand MAE at every k for one deployment rule."""
    model, spec = RULES[key]
    parts = []
    for k, (adapter, policy) in spec.items():
        block = per_ligand[(per_ligand.global_model == model) & (per_ligand.adapter == adapter)
                           & (per_ligand.policy == policy) & (per_ligand.k == k)]
        if block.empty:
            raise SystemExit(f"{key}: no rows for k={k} ({model}/{adapter}/{policy})")
        parts.append(block.assign(rule=key)[
            ["rule", "k", "extractant", "tanimoto_cluster", "nn_train_tanimoto", "mae"]])
    return pd.concat(parts, ignore_index=True)


def macro_with_ci(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, block in frame.groupby("k"):
        point, low, high = _stats.block_bootstrap_mean(block["mae"], block["tanimoto_cluster"])
        rows.append({"k": int(k), "macro_mae": point, "lo": low, "hi": high,
                     "n_ligands": block["extractant"].nunique()})
    return pd.DataFrame(rows).sort_values("k")


def paired_delta(frame: pd.DataFrame, k_ref: int, k_cand: int) -> dict:
    """reference(k_ref) - candidate(k_cand), one unit per extractant, chemotype blocks."""
    long = frame[frame.k.isin([k_ref, k_cand])].copy()
    long["arm"] = np.where(long.k == k_ref, "ref", "cand")
    out = _stats.paired_chemotype_bootstrap(long, {"delta": ("ref", "cand")},
                                            seed_column="__none__")
    return out.iloc[0].to_dict()


def seed_range(key: str) -> pd.DataFrame:
    """Min/max macro MAE over the five split seeds for one rule (reproducibility, not a CI)."""
    per_seed = pd.read_csv(_paths.DERIVED / "kshot_per_seed.csv")
    model, spec = RULES[key]
    rows = []
    for k, (adapter, policy) in spec.items():
        block = per_seed[(per_seed.global_model == model) & (per_seed.adapter == adapter)
                         & (per_seed.policy == policy) & (per_seed.k == k)]
        rows.append({"k": int(k), "lo": float(block["mae"].min()),
                     "hi": float(block["mae"].max()), "n_seeds": int(len(block))})
    return pd.DataFrame(rows).sort_values("k")


def main() -> int:
    _style.apply()
    per_ligand = load()
    frames = {key: rule_frame(per_ligand, key) for key in RULES}
    curves = {key: macro_with_ci(f) for key, f in frames.items()}
    spreads = {key: seed_range(key) for key in RULES}

    fig = plt.figure(figsize=(_style.W2, 4.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], width_ratios=[1.14, 1.0],
                          hspace=0.42, wspace=0.30,
                          left=0.075, right=0.985, top=0.93, bottom=0.095)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # ---------------- A: the frontier -------------------------------------------
    order = ["no_model", "frozen", "recomposed", "pipeline", "oracle"]
    for key in order:
        c = curves[key]
        s = spreads[key].set_index("k")
        kw = _style.m(key)
        kw["label"] = LABELS[key]
        axA.plot(c.k, c.macro_mae, **kw)
        axA.vlines(c.k, s.loc[c.k, "lo"], s.loc[c.k, "hi"], color=kw["color"],
                   linewidth=0.8, alpha=0.75)
    axA.set_xlabel("measured support points from the held-out extractant, $k$")
    axA.set_ylabel("macro MAE  (log$_{10}$ $D$ units)")
    axA.set_xticks(KS)
    axA.set_xlim(-0.3, 5.3)
    axA.set_ylim(0.36, 1.12)
    pipe = curves["pipeline"].set_index("k")["macro_mae"]
    for k, dx, dy in [(0, 7, -11), (1, 5, -11), (5, -17, -11)]:
        axA.annotate(f"{pipe[k]:.3f}", (k, pipe[k]), textcoords="offset points",
                     xytext=(dx, dy), fontsize=6.3, color=_style.colour("pipeline"),
                     annotation_clip=False)
    axA.legend(loc="upper right", bbox_to_anchor=(1.04, 1.05))
    _style.panel(axA, "A", dx=-0.135)

    # ---------------- B: marginal value of each measurement ---------------------
    steps = [(0, 1, "1st"), (1, 2, "2nd"), (2, 3, "3rd"), (3, 5, "4th\u20135th\n(each)")]
    bars, los, his, names, improved = [], [], [], [], []
    pipeline = frames["pipeline"]
    for k_ref, k_cand, name in steps:
        d = paired_delta(pipeline, k_ref, k_cand)
        scale = 1.0 / (k_cand - k_ref)
        bars.append(d["point"] * scale)
        los.append(d["bca_low"] * scale)
        his.append(d["bca_high"] * scale)
        names.append(name)
        improved.append(d["units_improved"] / d["n_units"])
    x = np.arange(len(bars))
    axB.bar(x, bars, width=0.6, color=_style.colour("pipeline"), alpha=0.85, linewidth=0)
    axB.errorbar(x, bars, yerr=[np.array(bars) - np.array(los), np.array(his) - np.array(bars)],
                 fmt="none", ecolor=_style.BLACK, elinewidth=0.7, capsize=1.8)
    for xi, b in enumerate(bars):
        axB.text(xi, b + 0.022, f"{b:.3f}", ha="center", fontsize=6.2)
    axB.set_xticks(x)
    axB.set_xticklabels([f"{n.splitlines()[0]}\n{f:.0%}" for n, f in zip(names, improved)])
    axB.set_xlabel("measurement added / % of extractants it improves")
    axB.set_ylabel("macro MAE removed\nper measurement")
    axB.set_ylim(0.0, 0.64)
    axB.axhline(0, color=_style.BLACK, linewidth=0.6)
    _style.panel(axB, "B", dx=-0.20)

    # ---------------- C: the two components of the gain, paired -----------------
    contrasts = [("recomposed", "frozen", "recomposed", "shape recomposition"),
                 ("pipeline", "recomposed", "pipeline", "series-local adaptation")]
    comp_rows = []
    for cand_key, ref_key, style_key, nice in contrasts:
        ref = frames[ref_key].rename(columns={"mae": "mae_ref"})
        cand = frames[cand_key].rename(columns={"mae": "mae_cand"})
        merged = ref.merge(cand, on=["k", "extractant", "tanimoto_cluster"])
        for k, block in merged.groupby("k"):
            long = pd.concat([block.assign(arm="ref", mae=block.mae_ref),
                              block.assign(arm="cand", mae=block.mae_cand)], ignore_index=True)
            r = _stats.paired_chemotype_bootstrap(long, {"d": ("ref", "cand")},
                                                  seed_column="__none__").iloc[0]
            comp_rows.append({"component": nice, "style": style_key, "k": int(k),
                              "point": r["point"], "lo": r["bca_low"], "hi": r["bca_high"],
                              "improved": int(r["units_improved"]), "n": int(r["n_units"])})
    comp = pd.DataFrame(comp_rows)
    width = 0.36
    for offset, (nice, style_key) in zip((-width / 2, width / 2),
                                         [("shape recomposition", "recomposed"),
                                          ("series-local adaptation", "pipeline")]):
        block = comp[comp.component == nice].sort_values("k").reset_index(drop=True)
        pos = np.arange(len(block)) + offset
        axC.bar(pos, block.point, width=width, color=_style.colour(style_key), alpha=0.9,
                linewidth=0, label=nice)
        axC.errorbar(pos, block.point,
                     yerr=[block.point - block.lo, block.hi - block.point],
                     fmt="none", ecolor=_style.BLACK, elinewidth=0.6, capsize=1.5)
    axC.axhline(0, color=_style.BLACK, linewidth=0.6)
    axC.set_xticks(np.arange(len(KS)))
    axC.set_xticklabels(KS)
    axC.set_xlabel("$k$")
    axC.set_ylabel("macro MAE removed\n(paired, vs the previous stage)")
    axC.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.06))
    axC.set_ylim(-0.006, 0.052)
    axC.text(0.98, 0.98, "series adaptation is the identity at $k\\leq1$",
             transform=axC.transAxes, fontsize=5.9, color=_style.GREY,
             ha="right", va="top")
    _style.panel(axC, "C", dx=-0.135)

    # ---------------- D: per-extractant paired scatter, k = 0 vs k = 1 ----------
    wide = pipeline.pivot_table(index=["extractant", "tanimoto_cluster"], columns="k",
                                values="mae")
    z, one = wide[0].to_numpy(), wide[1].to_numpy()
    better = int((one < z).sum())
    axD.scatter(z, one, s=8, facecolor="none", edgecolor=_style.colour("pipeline"),
                linewidth=0.55, alpha=0.85)
    lim = (0.0, 3.35)
    axD.plot(lim, lim, color=_style.BLACK, linewidth=0.7, linestyle="--", zorder=0)
    axD.set_xlim(*lim)
    axD.set_ylim(*lim)
    axD.set_aspect("equal")
    axD.set_xlabel("zero-shot MAE per extractant")
    axD.set_ylabel("MAE after one measurement")
    axD.text(0.04, 0.96, f"{better} of {len(z)} improved\nmedian $\\Delta$ = "
                         f"{np.median(z - one):+.2f}\n{len(z) - better} worse",
             transform=axD.transAxes, fontsize=6.3, va="top")
    _style.panel(axD, "D", dx=-0.20)

    written = _style.save(fig, "Fig2_fewshot_frontier", _paths.MAIN)

    record = {
        "figure": "Fig2_fewshot_frontier",
        "source": "runs/gen10_final/final_locked/kshot_detail.parquet",
        "cohort": json.loads((_paths.DERIVED / "kshot_cohort.json").read_text()),
        "rules": {k: {"global_model": v[0], "by_k": {str(kk): list(vv) for kk, vv in v[1].items()}}
                  for k, v in RULES.items()},
        "panelA_mean": {k: c.round(6).to_dict("records") for k, c in curves.items()},
        "panelA_seed_range": {k: s.round(6).to_dict("records") for k, s in spreads.items()},
        "panelB": [{"step": n, "gain_per_measurement": b, "bca_low": lo, "bca_high": hi,
                    "frac_units_improved": f}
                   for n, b, lo, hi, f in zip(names, bars, los, his, improved)],
        "panelC": comp.round(6).to_dict("records"),
        "panelD": {"n_units": int(len(z)), "n_improved": better,
                   "median_delta": float(np.median(z - one)),
                   "n_worsened": int((one > z).sum())},
    }
    (_paths.DERIVED / "fig2_values.json").write_text(json.dumps(record, indent=1))
    print("\n".join(str(p) for p in written))
    for key in order:
        print(f"{key:12s}", [round(v, 4) for v in curves[key].macro_mae.tolist()])
    print("panelB", [round(b, 4) for b in bars])
    print("panelC");print(comp.round(4).to_string())
    print("panelD", record["panelD"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

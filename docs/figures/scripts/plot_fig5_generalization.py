"""Figure 5 — how error depends on distance from the training chemistry, and whether
adding training chemistry causally repairs it.

Run from the repository root:
    ``.venv/bin/python figures/scripts/plot_fig5_generalization.py``

Panel A  final pipeline, 99 held-out extractants split into terciles of nearest-training-
         neighbour Tanimoto, macro MAE at k = 0/1/2/3/5.
         Source ``runs/gen10_final/final_locked/kshot_detail.parquet``.
         The tercile definition is gen10's own (``scripts/gen10_budget_simulation.py``);
         this script reproduces all 15 numbers of
         ``runs/gen10_final/budget_simulation/marginal_gains.csv`` as a check.
Panels B, C  gen6 Experiment A: byte-identical held-out rows, identical learner and
         folds, only the *training* mask differs.  ``BASE`` trains on extractants with
         >= 10 condition cells, ``EXPANDED`` on those with >= 3.
         Source ``runs/gen6_expA_5seed/{hard_chemistry_metrics,contrast_summary}.csv``.
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

MODEL = "GEN9_SHAPE_RECOMPOSED"
FEATURES = "MC_lig2d_ext_massaction"          # gen6's champion feature set
PIPELINE = {0: ("ZERO_SHOT_REF", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
            2: ("SERIES_ML", "CENTRAL_THEN_SPREAD"), 3: ("SERIES_ML", "CENTRAL_THEN_SPREAD"),
            5: ("SERIES_ML", "CENTRAL_THEN_SPREAD")}
GEN9_RULE = {0: ("ZERO_SHOT_REF", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
             2: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD"),
             3: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD"),
             5: ("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD")}
TERCILES = ["near", "mid", "far"]
TERCILE_COLOUR = {"near": _style.SKY, "mid": _style.BLUE, "far": _style.VERMILLION}


def per_ligand_table() -> pd.DataFrame:
    path = _paths.DERIVED / "kshot_per_ligand.csv"
    if not path.exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_kshot_tables.py"))],
                       check=True)
    return pd.read_csv(path)


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
    published = pd.read_csv(_paths.run("gen10_final/budget_simulation/marginal_gains.csv"),
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
    unpaired block bootstrap and is reported as such.
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


def main() -> int:
    _style.apply()
    per_ligand = per_ligand_table()
    curves = adaptation_curves(per_ligand, PIPELINE)
    checks = verify_marginal_gains(adaptation_curves(per_ligand, GEN9_RULE))
    if not checks["pass"].all():
        print(checks.to_string())
        raise SystemExit("distance-tercile reproduction failed — figure not drawn")

    fig = plt.figure(figsize=(_style.W2, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.46,
                          left=0.075, right=0.99, top=0.895, bottom=0.215)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ------------------------------------------------------------------ A ------
    ks = sorted(PIPELINE)
    panelA = []
    for tercile in TERCILES:
        block = curves[curves.tercile == tercile]
        point, lo, hi = [], [], []
        for k in ks:
            p, l, h = _stats.block_bootstrap_mean(block[f"m{k}"], block.tanimoto_cluster)
            point.append(p)
            lo.append(l)
            hi.append(h)
            panelA.append({"tercile": tercile, "k": k, "macro_mae": p,
                           "ci95_low": l, "ci95_high": h, "n_extractants": len(block)})
        colour = TERCILE_COLOUR[tercile]
        nn = block.nn_train_tanimoto
        axA.plot(ks, point, color=colour, marker="o", markersize=3.2, linewidth=1.2,
                 label=f"{tercile}  ($T$ {nn.min():.2f}–{nn.max():.2f}, $n$={len(block)})")
        axA.vlines(ks, lo, hi, color=colour, linewidth=0.8, alpha=0.55)
    axA.set_xticks(ks)
    axA.set_xlabel("measured support points, $k$")
    axA.set_ylabel("macro MAE  (log$_{10}$ $D$ units)")
    axA.set_title("distance to training chemistry", fontsize=6.8, color=_style.GREY)
    contrast = tercile_contrast(curves, "m0")
    axA.text(0.98, 0.53, f"far $-$ near at $k=0$:\n{contrast['point']:+.2f} "
                         f"[{contrast['ci95_low']:+.2f}, {contrast['ci95_high']:+.2f}]",
             transform=axA.transAxes, ha="right", va="top", fontsize=5.9,
             color=_style.GREY)
    axA.legend(loc="upper right", fontsize=6.0)
    axA.set_ylim(0.28, 1.72)
    _style.panel(axA, "A", dx=-0.145)

    # ------------------------------------------------------------------ B ------
    hard = pd.read_csv(_paths.run("gen6_expA_5seed/hard_chemistry_metrics.csv"))
    hard = hard[hard.feature_set == FEATURES]
    endpoints = [("all", "all\nextractants"), ("nn<0.6", "$T<0.6$"), ("nn<0.4", "$T<0.4$")]
    arms = [("BASE", "base91"), ("EXPANDED", "expanded152"), ("EXPANDED_SHUFFLED", "shuffled")]
    x = np.arange(len(endpoints))
    width = 0.26
    panelB = []
    for j, (arm, style_key) in enumerate(arms):
        values, errs, ns = [], [], []
        for endpoint, _ in endpoints:
            block = hard[(hard.arm == arm) & (hard.endpoint == endpoint)]
            per_seed = block.groupby("split_seed")["macro_mae"].mean()
            values.append(float(per_seed.mean()))
            errs.append((float(per_seed.max() - per_seed.mean()),
                         float(per_seed.mean() - per_seed.min())))
            ns.append(float(block.n_ligands.mean()))
            panelB.append({"arm": arm, "endpoint": endpoint, "macro_mae": values[-1],
                           "seed_min": float(per_seed.min()), "seed_max": float(per_seed.max()),
                           "n_ligands_mean": ns[-1], "n_seeds": int(len(per_seed))})
        pos = x + (j - 1) * width
        axB.bar(pos, values, width=width, color=_style.colour(style_key), alpha=0.9,
                linewidth=0, label=_style.label(style_key))
        axB.errorbar(pos, values,
                     yerr=np.array([[e[1] for e in errs], [e[0] for e in errs]]),
                     fmt="none", ecolor=_style.BLACK, elinewidth=0.6, capsize=1.4)
    axB.set_xticks(x)
    axB.set_xticklabels([f"{lab}\n$n$={n:.0f}" for (_, lab), n in zip(endpoints, ns)])
    axB.set_ylabel("macro MAE  (log$_{10}$ $D$ units)")
    axB.set_xlabel("held-out subset by distance $T$")
    axB.set_title("identical held-out rows", fontsize=6.8, color=_style.GREY)
    axB.legend(loc="upper left", fontsize=5.9, bbox_to_anchor=(-0.04, 1.04))
    axB.set_ylim(0, 2.25)
    _style.panel(axB, "B", dx=-0.19)

    # ------------------------------------------------------------------ C ------
    contrasts = pd.read_csv(_paths.run("gen6_expA_5seed/contrast_summary.csv"))
    contrasts = contrasts[(contrasts.feature_set == FEATURES) & (contrasts.statistic == "mae")]
    wanted = [("EXPANDED_vs_BASE", "all", "full coverage\nvs base, all"),
              ("EXPANDED_vs_BASE", "nn<0.6", "full vs base\n$T<0.6$"),
              ("EXPANDED_vs_BASE", "nn<0.4", "full vs base\n$T<0.4$"),
              ("EXPANDED_ROWMATCHED_vs_BASE", "all", "row-matched\ncontrol, all"),
              ("EXPANDED_vs_EXPANDED_SHUFFLED", "all", "vs shuffled\ntarget, all")]
    panelC, ys, points, los, his, colours = [], [], [], [], [], []
    for i, (comparison, endpoint, label) in enumerate(wanted):
        row = contrasts[(contrasts.comparison == comparison)
                        & (contrasts.endpoint == endpoint)]
        if row.empty:
            raise SystemExit(f"missing contrast {comparison}@{endpoint}")
        row = row.iloc[0]
        ys.append(len(wanted) - 1 - i)
        points.append(row.pooled_point_delta)
        los.append(row.bca_low)
        his.append(row.bca_high)
        colours.append(_style.colour("expanded152") if comparison.startswith("EXPANDED_vs_BASE")
                       else _style.GREY)
        panelC.append({"comparison": comparison, "endpoint": endpoint,
                       "point": float(row.pooled_point_delta), "bca_low": float(row.bca_low),
                       "bca_high": float(row.bca_high),
                       "block_macro_delta": float(row.block_macro_delta),
                       "units_improved": int(row.pooled_units_improved),
                       "units_total": int(row.pooled_units_total),
                       "seeds_positive": int(row.seeds_positive), "n_seeds": int(row.n_seeds)})
    axC.errorbar(points, ys, xerr=[np.array(points) - np.array(los),
                                   np.array(his) - np.array(points)],
                 fmt="o", markersize=3.6, elinewidth=0.9, capsize=1.8,
                 ecolor=_style.BLACK, mfc="none", mew=0.9)
    for xi, yi, c in zip(points, ys, colours):
        axC.plot([xi], [yi], marker="o", markersize=3.6, color=c)
    axC.axvline(0, color=_style.BLACK, linewidth=0.6)
    axC.set_yticks(ys)
    axC.set_yticklabels([label for _, _, label in wanted], fontsize=6.2)
    axC.set_xlabel("macro MAE removed (paired, BCa 95 % CI)")
    axC.set_title("chemistry, not row count", fontsize=6.8, color=_style.GREY)
    axC.set_xlim(-0.05, 0.95)
    _style.panel(axC, "C", dx=-0.46)

    written = _style.save(fig, "Fig5_generalization", _paths.MAIN)
    record = {"figure": "Fig5_generalization",
              "panelA": panelA,
              "panelA_tercile_contrasts": [tercile_contrast(curves, f"m{k}") for k in ks],
              "panelA_cohort": "99 extractants of the k-shot common cohort, final pipeline",
              "panelA_verification": checks.to_dict("records"),
              "panelB": panelB, "panelC": panelC,
              "panelBC_cohort": "gen6 Experiment A: 5,248 rows / 152 extractants / 131 ECFP "
                                "clusters / 79 chemotypes; macro = one vote per ECFP cluster; "
                                f"feature set {FEATURES}; 5 split seeds"}
    (_paths.DERIVED / "fig5_values.json").write_text(json.dumps(record, indent=1))
    print("\n".join(str(p) for p in written))
    print("marginal-gain reproduction:", int(checks["pass"].sum()), "/", len(checks), "pass")
    print(pd.DataFrame(panelA).pivot_table(index="tercile", columns="k",
                                           values="macro_mae").round(4).to_string())
    print(pd.DataFrame(panelC).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

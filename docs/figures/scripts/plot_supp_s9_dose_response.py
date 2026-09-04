"""Figure S9 — the coverage effect is chemical transfer, not a class prior.

If the gain in Figure 5B were only "the arm that saw this class of chemistry predicts this
class better", it would be flat in how much closer the expansion actually brought the
nearest training neighbour.  It is not: the gain tracks the closeness gained.

Source: ``runs/gen6_expA_5seed/oof_predictions.parquet`` (columns ``nn_reference_tanimoto``
and ``nn_expanded_tanimoto`` per row, predictions for all four arms) — the per-extractant
MAE is recomputed here rather than read from a summary.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths, _style  # noqa: E402

FEATURES = "MC_lig2d_ext_massaction"
BINS = [(0.0, 1e-9, "0\n(no closer)"), (1e-9, 0.05, "0–0.05"),
        (0.05, 0.15, "0.05–0.15"), (0.15, 1.01, "> 0.15")]


def main() -> int:
    _style.apply()
    oof = pd.read_parquet(_paths.run("gen6_expA_5seed/oof_predictions.parquet"))
    oof = oof[oof.feature_set == FEATURES]
    per = oof.assign(err_base=(oof.prediction_BASE - oof.log_D).abs(),
                     err_exp=(oof.prediction_EXPANDED - oof.log_D).abs())
    per = per.groupby(["split_seed", "extractant"]).agg(
        base=("err_base", "mean"), expanded=("err_exp", "mean"),
        nn_ref=("nn_reference_tanimoto", "first"),
        nn_exp=("nn_expanded_tanimoto", "first")).reset_index()
    per["gain"] = per.base - per.expanded
    per["closer"] = (per.nn_exp - per.nn_ref).clip(lower=0)
    ligand = per.groupby("extractant").agg(gain=("gain", "mean"), closer=("closer", "mean"),
                                           nn_ref=("nn_ref", "mean")).reset_index()

    rho, pval = spearmanr(ligand.closer, ligand.gain)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(_style.W2, 2.5),
                                   gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.30})
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.21)

    record = []
    heights, labels, ns = [], [], []
    for lo, hi, label in BINS:
        block = ligand[(ligand.closer >= lo) & (ligand.closer < hi)] if lo > 0 else \
            ligand[ligand.closer < hi]
        heights.append(float(block.gain.mean()))
        labels.append(label)
        ns.append(len(block))
        record.append({"bin": label.replace("\n", " "), "n_extractants": len(block),
                       "mean_gain": heights[-1], "median_gain": float(block.gain.median()),
                       "mean_nn_to_base": float(block.nn_ref.mean()),
                       "frac_improved": float((block.gain > 0).mean())})
    x = np.arange(len(BINS))
    axA.bar(x, heights, width=0.62, color=_style.colour("expanded152"), alpha=0.9, linewidth=0)
    for xi, (h, n) in enumerate(zip(heights, ns)):
        axA.text(xi, h + 0.012, f"{h:+.3f}", ha="center", fontsize=6.0)
    axA.set_xticks(x)
    axA.set_xticklabels([f"{l}\n$n$={n}" for l, n in zip(labels, ns)], fontsize=6.0)
    axA.axhline(0, color=_style.BLACK, linewidth=0.6)
    axA.set_xlabel("Tanimoto gained: nn(expanded) $-$ nn(base)")
    axA.set_ylabel("macro MAE removed by\nfull training coverage")
    _style.panel(axA, "A", dx=-0.20)

    axB.scatter(ligand.closer, ligand.gain, s=9, facecolor="none",
                edgecolor=_style.colour("expanded152"), linewidth=0.6, alpha=0.85)
    axB.axhline(0, color=_style.BLACK, linewidth=0.6)
    axB.set_xlabel("Tanimoto gained")
    axB.set_ylabel("macro MAE removed")
    axB.text(0.97, 0.05, f"Spearman $\\rho$ = {rho:+.3f}, $p$ = {pval:.1e}\n"
                         f"{len(ligand)} extractants, 5 split seeds",
             transform=axB.transAxes, ha="right", fontsize=6.0)
    _style.panel(axB, "B", dx=-0.22)

    written = _style.save(fig, "FigS9_dose_response", _paths.SUPP)
    (_paths.DERIVED / "figS9_values.json").write_text(json.dumps(
        {"feature_set": FEATURES, "spearman_rho": float(rho), "spearman_p": float(pval),
         "n_extractants": int(len(ligand)), "bins": record}, indent=1))
    print("\n".join(str(p) for p in written))
    print(pd.DataFrame(record).round(4).to_string())
    print(f"Spearman rho={rho:.4f} p={pval:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

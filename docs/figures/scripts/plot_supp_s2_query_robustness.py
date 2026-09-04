"""Figure S2 — the relative-position mechanism is query-set dependent.

The same fitted model is asked for the same absolute conditions while the *candidate
design* around them changes.  A model that reads only absolute conditions must answer
identically; one that reads where a row sits inside the planned titration need not.

Source: ``runs/gen10_final/query_consistency/summary_by_arm.csv`` (5 split seeds; 66
extractants with synthesisable concentration axes for DECOY/EXTEND/DENSITY, 130 for the
subset variants), rendered here without transformation.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths, _style  # noqa: E402

FAMILIES = [("DECOY", "two candidate points\n2–3 decades outside"),
            ("EXTEND", "one boundary extended\n0.5–2 decades"),
            ("NESTED", "one or two end\npoints dropped"),
            ("SPARSE", "thinned to 2–8 points,\nendpoints kept"),
            ("DENSITY", "interior points added,\nwindow unchanged"),
            ("PERMUTE", "candidate order\npermuted"),
            ("CONTEXT", "whole-ligand vs\nper-series call")]
ARMS = [("FROZEN", "frozen"), ("REL_MONOLITH", "relpos"), ("SHAPE_RECOMPOSED", "recomposed")]
NICE = {"FROZEN": "baseline model", "REL_MONOLITH": "+ relative position",
        "SHAPE_RECOMPOSED": "+ recomposition"}


def main() -> int:
    _style.apply()
    table = pd.read_csv(_paths.run("gen10_final/query_consistency/summary_by_arm.csv"))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(_style.W2, 2.7))
    fig.subplots_adjust(left=0.19, right=0.985, top=0.90, bottom=0.19, wspace=0.10)

    y = np.arange(len(FAMILIES))[::-1]
    height = 0.24
    record = []
    for j, (arm, style_key) in enumerate(ARMS):
        med, p95, ns = [], [], []
        for family, _ in FAMILIES:
            row = table[(table.arm == arm) & (table.family == family)]
            med.append(float(row.shift_median__median.iloc[0]))
            p95.append(float(row.shift_p95__p95.iloc[0]))
            ns.append(int(row.n_ligands.iloc[0]))
            record.append({"arm": arm, "family": family, "shift_median": med[-1],
                           "shift_p95_of_p95": p95[-1], "n_ligands": ns[-1],
                           "n_comparisons": int(row.n_comparisons.iloc[0])})
        offset = (1 - j) * height
        axA.barh(y + offset, med, height=height, color=_style.colour(style_key),
                 alpha=0.9, linewidth=0, label=NICE[arm])
        axB.barh(y + offset, p95, height=height, color=_style.colour(style_key),
                 alpha=0.9, linewidth=0)
    for ax, title, xlab in ((axA, "typical shift", "median |Δ prediction| (log$_{10}$ $D$)"),
                            (axB, "tail", "p95 of the per-ligand p95 shift")):
        ax.set_yticks(y)
        ax.set_xlabel(xlab)
        ax.set_title(title, fontsize=6.9, color=_style.GREY)
    axA.set_yticklabels([label for _, label in FAMILIES], fontsize=6.0)
    axB.set_yticklabels([])
    axA.axvline(0.0247, color=_style.BLACK, linewidth=0.8, linestyle="--")
    axA.text(0.0247, len(FAMILIES) - 0.5, "  macro gain the\n  mechanism buys",
             fontsize=5.9, va="top")
    axA.legend(loc="lower right", fontsize=6.0)
    _style.panel(axA, "A", dx=-0.52)
    _style.panel(axB, "B", dx=-0.06)

    written = _style.save(fig, "FigS2_query_robustness", _paths.SUPP)
    (_paths.DERIVED / "figS2_values.json").write_text(json.dumps(record, indent=1))
    print("\n".join(str(p) for p in written))
    print(pd.DataFrame(record).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import sys
sys.path.insert(0, "generations/gen13_separation")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

A = pd.read_csv("generations/gen13_separation/analysis/stage3/s3_direction_accuracy_fixed.csv")
G = pd.read_csv("generations/gen13_separation/analysis/stage3/s3_direction_gain.csv")
order = ["always_heavy", "donor_pair_min", "lean_all", "donor_geometry"]
label = {"always_heavy": "always predict\nheavy-selective",
         "donor_pair_min": "one number:\nshortest donor path",
         "lean_all": "full compact set\n(209 columns)",
         "donor_geometry": "donor topology only\n(39 columns)"}
colour = {"always_heavy": "#9e9e9e", "donor_pair_min": "#c7a76c",
          "lean_all": "#2c7fb8", "donor_geometry": "#b0413e"}
designs = ["B", "BP", "A"]
dlabel = {"B": "B\nchemotype hold-out", "BP": "BP\n+ publications masked", "A": "A\nextractant hold-out"}

fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.6), sharey=True)
for ax, d in zip(axes, designs):
    sub = A[A.design == d].set_index("model")
    xs = np.arange(len(order))
    for i, m in enumerate(order):
        r = sub.loc[m]
        ax.bar(i, r.macro_accuracy, 0.62, color=colour[m], alpha=.9)
        ax.errorbar(i, r.macro_accuracy, yerr=[[r.macro_accuracy - r.ci_low], [r.ci_high - r.macro_accuracy]],
                    color="0.15", capsize=4, lw=1.2, fmt="none")
        ax.text(i, r.macro_accuracy + 0.012, f"{r.macro_accuracy:.2f}", ha="center", fontsize=8.5)
    g = G[(G.design == d) & (G.model == "donor_geometry")]
    ax.set_xticks(xs); ax.set_xticklabels([label[m] for m in order], fontsize=8.5)
    ax.set_title(dlabel[d] + f"\ndonor topology gains {g.gain.iat[0]:+.2f} "
                 f"[{g.ci_low.iat[0]:+.2f}, {g.ci_high.iat[0]:+.2f}], p = {g.p_two_sided.iat[0]:.3f}",
                 fontsize=9)
    ax.set_ylim(0, 1.0); ax.grid(alpha=.25, axis="y")
axes[0].set_ylabel("accuracy on the direction of selectivity\n(macro over 82 extractants, "
                   "chemotype-blocked interval)", fontsize=9)
fig.suptitle("Which end of the lanthanide series an extractant prefers is predictable from donor topology alone\n"
             "held-out chemotypes; 39 topological columns lose nothing against the full 209-column compact set",
             fontsize=11)
fig.tight_layout()
fig.savefig("generations/gen13_separation/figures/stage2/s3_direction_accuracy.png", dpi=160, bbox_inches="tight")
print("written")

import sys; sys.path.insert(0,"gen13_separation")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
b = pd.read_csv("gen13_separation/analysis/stage3/s3_direction_value_BP.csv").set_index("arm")
order = ["DIR_TRAIN_MAJORITY","MEAN_CURVE","DIR_PREDICTED","FULL_MODEL","DIR_ORACLE"]
lab = {"DIR_TRAIN_MAJORITY":"majority direction\n(2 d.o.f.)","MEAN_CURVE":"corpus mean curve\n(0 d.o.f.)",
       "DIR_PREDICTED":"direction from donor\ntopology (2 d.o.f.)","FULL_MODEL":"full regression\n(209 columns)",
       "DIR_ORACLE":"true direction\n(oracle, 2 d.o.f.)"}
col = {"DIR_TRAIN_MAJORITY":"#9e9e9e","MEAN_CURVE":"#bdbdbd","DIR_PREDICTED":"#b0413e",
       "FULL_MODEL":"#2c7fb8","DIR_ORACLE":"#4d9221"}
fig, ax = plt.subplots(figsize=(8.6,4.8))
for i,m in enumerate(order):
    r=b.loc[m]
    ax.bar(i, r.macro_mae_extractant, .64, color=col[m], alpha=.9)
    ax.errorbar(i, r.macro_mae_extractant, yerr=r.macro_mae_extractant_seed_sd, color="0.15",
                capsize=4, lw=1.1, fmt="none")
    ax.text(i, r.macro_mae_extractant+0.012, f"{r.macro_mae_extractant:.3f}", ha="center", fontsize=9)
ax.set_xticks(range(len(order))); ax.set_xticklabels([lab[m] for m in order], fontsize=8.5)
ax.set_ylabel("extractant-macro MAE of log SF\n(publication-masked design, 90 extractants, 5 seeds)", fontsize=9)
ax.set_ylim(0,0.80); ax.grid(alpha=.25, axis="y")
ax.annotate("", xy=(2,0.60), xytext=(3,0.60), arrowprops=dict(arrowstyle="<->", color="0.3"))
ax.text(2.5, 0.615, "+0.009, p = 0.76\nindistinguishable", ha="center", fontsize=8, color="0.25")
ax.annotate("", xy=(3,0.68), xytext=(4,0.68), arrowprops=dict(arrowstyle="<->", color="0.3"))
ax.text(3.5, 0.695, "+0.070, p = 0.006\nheadroom in the direction", ha="center", fontsize=8, color="0.25")
ax.set_title("Almost all the transferable chemistry in this corpus is one bit per extractant:\n"
             "which end of the lanthanide series it prefers", fontsize=11)
fig.tight_layout(); fig.savefig("gen13_separation/figures/stage2/s3_direction_value.png", dpi=160, bbox_inches="tight")
print("written")

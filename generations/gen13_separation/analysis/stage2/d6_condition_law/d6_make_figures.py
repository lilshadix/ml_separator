"""D6 figures: the observed condition dependence and the arms' blindness to it."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from d6_common import (load_cohort, cell_amplitudes, predicted_curves, OUT, FIG,
                       PRED_DIR, MIN_RZ_SPAN)

df = load_cohort()
obs = df.merge(cell_amplitudes(df), on="cell_id")
obs = obs[obs.rz_span >= MIN_RZ_SPAN].copy()
obs["log_acid"] = np.log10(obs["cond__acid_concentration_M"].astype(float))
g = obs.dropna(subset=["log_acid"]).groupby("extractant")
QUAL = [k for k, gg in g if len(gg) >= 3 and gg.log_acid.nunique() >= 2]
q = obs[obs.extractant.isin(QUAL)].dropna(subset=["log_acid"]).copy()

arm = "C_DIRECT_ROW"
pred = pd.read_parquet(os.path.join(PRED_DIR, arm + ".parquet"))
pred = pred[pred.cell_id.isin(set(q.cell_id))]
pc = predicted_curves(pred)
pc = pc[pc.rz_span >= MIN_RZ_SPAN].groupby("cell_id").amp_pred.mean().reset_index()
q = q.merge(pc, on="cell_id")

top = (q.groupby("extractant_name").size().sort_values(ascending=False).head(6).index)
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=False)
for ax, name in zip(axes.ravel(), top):
    s = q[q.extractant_name == name]
    ax.scatter(s.log_acid, s.amp, s=34, c="#1f4e79", label="observed", zorder=3)
    ax.scatter(s.log_acid, s.amp_pred, s=34, c="#c0392b", marker="^",
               label=f"{arm} prediction", zorder=3)
    if s.log_acid.nunique() >= 2:
        xs = np.linspace(s.log_acid.min(), s.log_acid.max(), 10)
        b = np.polyfit(s.log_acid, s.amp, 1)
        ax.plot(xs, np.polyval(b, xs), c="#1f4e79", lw=1.4, alpha=.75)
        bp = np.polyfit(s.log_acid, s.amp_pred, 1)
        ax.plot(xs, np.polyval(bp, xs), c="#c0392b", lw=1.4, alpha=.75, ls="--")
        ax.set_title(f"{name}  (n={len(s)})\nobs slope {b[0]:+.2f} / decade, "
                     f"pred {bp[0]:+.2f}", fontsize=9.5)
    ax.axhline(0, c="0.75", lw=.8, zorder=1)
    ax.set_xlabel("log$_{10}$ [acid] (M)", fontsize=9)
    ax.set_ylabel("rank-1 amplitude", fontsize=9)
    ax.tick_params(labelsize=8)
axes[0, 0].legend(fontsize=8, framealpha=.9)
fig.suptitle("D6  the separation curve moves with acid concentration inside one "
             "extractant; the model's curve does not", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.955])
p1 = os.path.join(FIG, "d6_condition_law_amplitude_vs_acid.png")
fig.savefig(p1, dpi=145)
plt.close(fig)

# panel 2: observed vs predicted within-extractant spread, all arms
S = pd.read_csv(OUT + "/d6_step4_amplitude_spread_by_arm.csv")
O = pd.read_csv(OUT + "/d6_step4_oracle_decomposition.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.6))
yp = np.arange(len(S))
a1.barh(yp + .2, S.sd_obs, height=.38, color="#1f4e79", label="observed")
a1.barh(yp - .2, S.sd_pred, height=.38, color="#c0392b", label="predicted")
a1.set_yticks(yp)
a1.set_yticklabels(S.arm, fontsize=8)
a1.invert_yaxis()
a1.set_xlabel("within-extractant sd of the rank-1 amplitude", fontsize=9)
a1.set_title("arms reproduce 4-21% of the movement", fontsize=10)
for i, r in S.iterrows():
    a1.text(r.sd_obs + .006, i + .2, f"{100*r.sd_ratio_pred_over_obs:.0f}%",
            va="center", fontsize=7.5)
a1.legend(fontsize=8)
a1.tick_params(labelsize=8)

w = .34
xa = np.arange(len(O))
a2.bar(xa - w, O.gain_within_rank2, w, color="#2e8b57",
       label="condition (within-extractant) oracle")
a2.bar(xa, O.gain_level_rank2, w, color="#8e44ad",
       label="extractant-level curve oracle")
a2.axhline(0.02, c="0.35", ls=":", lw=1.2)
a2.text(len(O) - .5, 0.026, "0.02 interest threshold", fontsize=7.5, ha="right")
a2.set_xticks(xa)
a2.set_xticklabels(O.arm, rotation=32, ha="right", fontsize=7.5)
a2.set_ylabel("extractant-macro MAE saved (log units)", fontsize=9)
a2.set_title("what each oracle is worth on the 12 qualifying extractants", fontsize=10)
a2.legend(fontsize=8)
a2.tick_params(labelsize=8)
fig.tight_layout()
p2 = os.path.join(FIG, "d6_condition_law_blindness_and_headroom.png")
fig.savefig(p2, dpi=145)
plt.close(fig)
print("wrote", p1)
print("wrote", p2)

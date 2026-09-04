#!/usr/bin/env python
"""Figure 5 — what "calibrated" means for this model, and which way it is wrong.

Refined from two upstream renders that both carried the word *calibration* and were
never placed side by side:

* ``automl/figures.py::fig_uncertainty``            -> ``fig6_uncertainty_calibration.png``
* ``automl/figures_reanalysis.py::fig_calibration`` -> ``re_fig4_calibration.png``

They measure two different senses of calibration on two different cohorts, and both
point the same way — the model under-states.  Panel A is the first render's table with
its second column restored; panels B and C supersede the second render.

No number is recomputed.  Everything drawn comes from the vendored copies of
``uncertainty_calibration.csv``, ``calibration_test_binned.csv`` and
``calibration_test_strict.csv``; the only arithmetic is a ratio of two plotted
quantities (panel A) and a difference of two tabulated R² values (panel C), and the
latter is asserted against the bootstrap point estimate the table already carries.

Run from anywhere:
    python figure_refinement/lanthanidestrain/figure_05_uncertainty_calibration/figure_script.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import lstrain_data as LD              # noqa: E402
import pubstyle as PS                  # noqa: E402

# --------------------------------------------------------------------------- #
# Reader-facing names.  No repository identifier reaches the artwork; the mapping is
# in notes.md and is asserted at the end of main().
# --------------------------------------------------------------------------- #
#: ``model`` column of calibration_test_*.csv -> what the reader is looking at.
#: "full (CatBoost+repaired+S0)" = the nested stack of gradient-boosted trees, the
#: repaired fully-connected network and the S0 simplicial 3D encoder
#: (automl/topo/best_stack.py, combos["full (CatBoost+repaired+S0)"]).
MODEL = "full (CatBoost+repaired+S0)"

#: ``transform`` column -> the map fitted to the predicted adjacent-pair difference,
#: nested by extractant (automl/topo/calibration_test.py::_fit_apply).
TRANSFORMS = [
    ("raw",      "as measured"),
    ("scale",    "one-factor rescale"),
    ("affine",   "linear rescale\n(factor + offset)"),
    ("isotonic", "any monotone map"),
]

#: ``key`` column -> how two lanthanides are declared to have been measured under the
#: same conditions (automl/topo/dualkey_test.py: BINNED / STRICT).
KEYS = [
    ("binned", "conditions matched in bins",  PS.BLUE),
    ("strict", "conditions matched exactly",  PS.ORANGE),
]

#: Cohort size of panel A.  Not carried by uncertainty_calibration.csv itself; it is
#: the row count of the run that wrote it — automl/reports/ensemble.txt line 1,
#: "stacking 37 base models over 5946 rows", whose printed calibration table is
#: identical to the CSV to every digit it prints.  Corroborated by
#: automl/reports/tables.md: "has3d = the full series (5946 rows)".
N_OOF_ROWS = 5946

MINUS = "−"          # typographic minus, so −0.045 does not read as a hyphen


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def uncertainty_rows() -> dict:
    """Realised error and quoted disagreement, by quintile of disagreement."""
    t = LD.read("uncertainty_calibration").sort_values("quintile")
    mae = t["mae"].to_numpy(float)
    spread = t["spread"].to_numpy(float)
    return {"quintile": (t["quintile"].to_numpy(int) + 1).tolist(),
            "realised_mean_abs_error_logD": mae.tolist(),
            "quoted_ensemble_disagreement_sd_logD": spread.tolist(),
            "ratio_realised_over_quoted": (mae / spread).tolist(),
            "n_out_of_fold_rows": N_OOF_ROWS,
            "n_rows_per_quintile": int(round(N_OOF_ROWS / len(mae)))}


def calibration_rows() -> dict:
    """Span ratio and R² for the deployed stack, both pair-matching definitions."""
    out = {}
    for key, _label, _c in KEYS:
        t = LD.read(f"calibration_test_{key}")
        sub = t[t["model"] == MODEL]
        raw_r2 = float(sub.loc[sub["transform"] == "raw", "r2"].iloc[0])
        rows = []
        for ident, _name in TRANSFORMS:
            r = sub[sub["transform"] == ident].iloc[0]
            rows.append({"transform": ident,
                         "span_ratio": float(r["span_ratio"]),
                         "adjacent_pair_logSF_r2": float(r["r2"]),
                         "delta_r2_vs_as_measured": float(r["r2"]) - raw_r2})
        gain = sub[sub["transform"] == "scale_gain"]
        boot = None
        if not gain.empty:
            g = gain.iloc[0]
            boot = {"transform": "scale", "delta_r2": float(g["r2"]),
                    "lo_90": float(g["lo"]), "hi_90": float(g["hi"]),
                    "p_positive": float(g["p_positive"]),
                    "n_draws": 400, "level": "90 % cluster bootstrap over extractants"}
            # The bootstrap point estimate must equal the difference of the two
            # tabulated R² values; if it does not, the table has drifted and this
            # figure must not be drawn.
            recomputed = next(r["delta_r2_vs_as_measured"] for r in rows
                              if r["transform"] == "scale")
            assert abs(recomputed - boot["delta_r2"]) < 1e-9, (
                f"{key}: scale_gain {boot['delta_r2']} != r2(scale) - r2(raw) "
                f"{recomputed}")
        out[key] = {"n_pairs": int(sub["n_pairs"].iloc[0]),
                    "raw_adjacent_pair_logSF_r2": raw_r2,
                    "rows": rows, "bootstrap_on_one_factor_rescale": boot}
    return out


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def panel_a(ax, u: dict) -> None:
    x = np.arange(len(u["quintile"]))
    mae = np.asarray(u["realised_mean_abs_error_logD"])
    sd = np.asarray(u["quoted_ensemble_disagreement_sd_logD"])
    ratio = np.asarray(u["ratio_realised_over_quoted"])

    # The gap is the finding, so it is drawn as an object rather than left to the eye.
    ax.vlines(x, sd, mae, color=PS.PALE, linewidth=5.0, zorder=1)
    ax.plot(x, mae, marker="o", markersize=4.4, color=PS.BLUE, linewidth=1.4, zorder=3)
    ax.plot(x, sd, marker="D", markersize=4.0, color=PS.VERMILLION, linewidth=1.4,
            linestyle="--", zorder=3)

    for xi, lo, hi, r in zip(x, sd, mae, ratio):
        ax.text(xi + 0.14, (lo + hi) / 2, f"{r:.1f}×", ha="left", va="center",
                fontsize=PS.BASE - 1.0, color=PS.INK)

    # Direct labels rather than a legend: no box, and nothing can land on the data.
    ax.text(-0.34, 1.435, "calibrated would mean the two series coincide",
            fontsize=PS.BASE - 1.0, color=PS.GREY, ha="left", va="top")
    ax.text(-0.34, 1.355, "errors actually made — mean absolute error",
            fontsize=PS.BASE - 1.0, color=PS.BLUE, ha="left", va="top")
    ax.text(-0.34, 1.280, "rises with disagreement, but Q4 sits below Q3",
            fontsize=PS.BASE - 1.0, color=PS.GREY, ha="left", va="top")
    ax.text(-0.34, 0.020, "error bar the model quotes —\nspread across the base models",
            fontsize=PS.BASE - 1.0, color=PS.VERMILLION, ha="left", va="bottom",
            linespacing=1.4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{q}" for q in u["quintile"]])
    ax.set_xlim(-0.42, 4.80)
    ax.set_ylim(0, 1.46)
    ax.set_xlabel("quintile of ensemble disagreement\n"
                  "Q1 = most confident, Q5 = least confident")
    ax.set_ylabel("log D units  (absolute prediction, not a difference)")
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.grid(axis="y", visible=True)
    ax.set_axisbelow(True)
    PS.strapline(ax, f"{u['n_out_of_fold_rows']:,} held-out predictions, "
                     f"five equal quintiles of about "
                     f"{u['n_rows_per_quintile']:,}")


def _rows_geometry(n_transforms: int, top_band: float):
    """Row centres, and the blank band above them that holds a legend or a note."""
    y = np.arange(n_transforms, dtype=float)
    bottom = n_transforms - 0.44
    return y, top_band, bottom


def panel_b(ax, cal: dict) -> None:
    y, top, bottom = _rows_geometry(len(TRANSFORMS), -1.30)
    h = 0.32
    for i, (key, label, colour) in enumerate(KEYS):
        offs = (i - 0.5) * h * 1.06
        vals = [r["span_ratio"] for r in cal[key]["rows"]]
        ax.barh(y + offs, vals, height=h, color=colour, zorder=3,
                label=f"{label}  ({cal[key]['n_pairs']:,} pairs)")
        for yi, v in zip(y + offs, vals):
            ax.text(v + 0.018, yi, f"{v:.2f}", va="center", ha="left",
                    fontsize=PS.BASE - 1.0, color=PS.INK)

    # Reference drawn only across the rows, so the band above it stays empty and the
    # legend placed there cannot cover data.
    ax.plot([1.0, 1.0], [-0.52, bottom], color=PS.INK, linewidth=1.0, zorder=4)
    ax.text(1.035, 1.55, "correct spread", fontsize=PS.BASE - 1.0, color=PS.INK,
            ha="center", va="center", rotation=90)

    ax.set_yticks(y)
    ax.set_yticklabels([name for _ident, name in TRANSFORMS])
    ax.set_ylim(bottom, top)
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("SD of predicted ÷ SD of measured separation\n"
                  "ratio: 1 = correct spread, below 1 = too flat")
    ax.grid(axis="x", visible=True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.012, 1.02), ncol=1,
              fontsize=PS.BASE - 1.0, handlelength=1.2, handleheight=0.8,
              labelspacing=0.28, borderaxespad=0.0)
    PS.strapline(ax, "adjacent lanthanide pairs, deployed three-model stack")


def panel_c(ax, cal: dict) -> None:
    y, top, bottom = _rows_geometry(len(TRANSFORMS), -1.30)
    h = 0.32
    for i, (key, _label, colour) in enumerate(KEYS):
        offs = (i - 0.5) * h * 1.06
        rows = cal[key]["rows"]
        vals = [r["delta_r2_vs_as_measured"] for r in rows]
        ax.barh(y[1:] + offs, vals[1:], height=h, color=colour, zorder=3)
        ends = list(vals)
        boot = cal[key]["bootstrap_on_one_factor_rescale"]
        if boot is not None:
            j = [r["transform"] for r in rows].index(boot["transform"])
            ax.errorbar(boot["delta_r2"], y[j] + offs,
                        xerr=[[boot["delta_r2"] - boot["lo_90"]],
                              [boot["hi_90"] - boot["delta_r2"]]],
                        fmt="none", ecolor=PS.INK, elinewidth=0.9, capsize=1.8,
                        zorder=5)
            # value label clears the whole whisker, on the side of the sign
            ends[j] = boot["hi_90"] if boot["delta_r2"] >= 0 else boot["lo_90"]
        for yi, v, end in zip(y[1:] + offs, vals[1:], ends[1:]):
            right = v >= 0
            anchor = max(end, v) if right else min(end, v)
            ax.text(anchor + (0.0026 if right else -0.0030), yi,
                    f"{v:+.4f}".replace("-", MINUS), va="center",
                    ha="left" if right else "right",
                    fontsize=PS.BASE - 1.0, color=PS.INK)
        # The reference row is zero by construction, so it carries the level the
        # changes are measured from instead of a bar.  A level, stated as a level:
        # nothing on this panel may be subtracted from anything on another.
        ax.text(0.0030, y[0] + offs,
                f"R² = {cal[key]['raw_adjacent_pair_logSF_r2']:+.3f}",
                va="center", ha="left", fontsize=PS.BASE - 1.0, color=colour)

    ax.plot([0.0, 0.0], [-0.52, bottom], color=PS.INK, linewidth=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(["as measured\n(reference)"]
                       + [name for _ident, name in TRANSFORMS[1:]])
    ax.set_ylim(bottom, top)
    ax.set_xlim(-0.066, 0.040)
    ax.set_xticks([-0.04, -0.02, 0.0, 0.02])
    ax.set_xlabel("change in adjacent-pair log SF R²\n"
                  "+ = recalibration buys accuracy")
    ax.grid(axis="x", visible=True)
    ax.set_axisbelow(True)
    ax.text(-0.064, top + 0.06,
            "whisker = 90 % cluster bootstrap over extractants;\n"
            "best of three transforms, uncorrected for the choice",
            fontsize=PS.BASE - 1.0, color=PS.GREY, ha="left", va="top",
            linespacing=1.4)
    PS.strapline(ax, "same pairs, same model and same colours as B")


# --------------------------------------------------------------------------- #
def main() -> int:
    PS.apply()
    u = uncertainty_rows()
    cal = calibration_rows()

    fig = plt.figure(figsize=(PS.W2, 4.40))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.00, 1.14],
                          height_ratios=[1.00, 1.00])
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    panel_a(axA, u)
    panel_b(axB, cal)
    panel_c(axC, cal)

    PS.add_panel_letters(fig, [axA, axB, axC])

    # No repository identifier may reach a drawn text artist.
    banned = ("CatBoost", "repaired", "S0", "composition_key", "span_ratio",
              "isotonic", "affine", "adj_r2", "raw", "binned", "strict",
              "uncertainty_calibration", "nnls", "T0w")
    drawn = " ".join(str(t.get_text()) for t in fig.findobj(matplotlib.text.Text)
                     if str(t.get_text()).strip())
    for token in banned:
        assert token not in drawn, f"repository identifier {token!r} reached the artwork"

    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)

    values = {
        "figure": "Figure 5 — uncertainty and magnitude calibration",
        "source_repository": LD.provenance()["source_repository"],
        "source_commit": LD.provenance()["commit"],
        "supersedes": ["automl/reports/figures/fig6_uncertainty_calibration.png",
                       "automl/reports/figures/re_fig4_calibration.png"],
        "panel_A": {
            "question": "does the model's quoted uncertainty match the errors it makes?",
            "table": "uncertainty_calibration.csv",
            "model": "out-of-fold ensemble stack (non-negative least squares combiner)",
            "units": "log D (absolute prediction, not a difference)",
            **u,
        },
        "panel_B": {
            "question": "how wide are the predicted adjacent-lanthanide separations?",
            "tables": ["calibration_test_binned.csv", "calibration_test_strict.csv"],
            "model": MODEL,
            "quantity": "span_ratio = SD(predicted difference) / SD(measured difference)",
            **{k: {"n_pairs": v["n_pairs"],
                   "span_ratio": {r["transform"]: r["span_ratio"] for r in v["rows"]}}
               for k, v in cal.items()},
        },
        "panel_C": {
            "question": "does recalibration buy adjacent-pair accuracy?",
            "quantity": "adjacent-pair log SF R2 of the transform minus that of the "
                        "model as measured; positive favours recalibration",
            **{k: {"n_pairs": v["n_pairs"],
                   "adjacent_pair_logSF_r2": {r["transform"]:
                                              r["adjacent_pair_logSF_r2"]
                                              for r in v["rows"]},
                   "delta_r2_vs_as_measured": {r["transform"]:
                                               r["delta_r2_vs_as_measured"]
                                               for r in v["rows"]},
                   "bootstrap_on_one_factor_rescale":
                       v["bootstrap_on_one_factor_rescale"]}
               for k, v in cal.items()},
        },
        "label_map": {
            MODEL: "deployed three-model stack "
                   "(gradient-boosted trees + repaired neural network + 3D encoder)",
            "composition_key (binned)": "conditions matched in bins",
            "strict_composition_key (strict)": "conditions matched exactly",
            "raw": "as measured",
            "scale": "one-factor rescale",
            "affine": "linear rescale (factor + offset)",
            "isotonic": "any monotone map",
            "span_ratio": "SD of predicted separation / SD of measured separation",
            "mae": "errors actually made (mean |error| of the stack)",
            "spread": "error bar the model quotes (SD across the base models)",
        },
    }
    (HERE / "values.json").write_text(json.dumps(values, indent=1) + "\n")
    print(f"wrote {HERE / 'values.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

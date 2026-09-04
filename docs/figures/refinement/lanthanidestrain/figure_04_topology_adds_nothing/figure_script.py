#!/usr/bin/env python
"""Figure 4 — beyond the distance encoder, topology adds nothing.

Composed for the negative topology result of `lanthanidestrain`
(`automl/reports/TOPOLOGY_TESTS.md`, finding I17).  The result had no figure of its
own: the four rendered `topo_*` figures in the repository all argue the *earlier*,
positive claim.  This script draws the null itself, and the evidence for it.

Every number comes from the vendored copy of the upstream result tables
(`figure_refinement/lanthanidestrain/data/`), except the five pair-level
correlations of panel B, which upstream computes from out-of-fold parquets under
`automl/artifacts/` — not committed upstream, therefore not vendored here.  Those
five are transcribed from the table in `TOPOLOGY_TESTS.md` §1 at the vendored
commit and are flagged as transcribed in `values.json`.  Nothing is recomputed.

Run from anywhere:
    python figure_refinement/lanthanidestrain/figure_04_topology_adds_nothing/figure_script.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from matplotlib.patches import Patch     # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common"))
import lstrain_data as LD                # noqa: E402
import pubstyle as PS                    # noqa: E402

# --------------------------------------------------------------------------- #
# Semantic colours, fixed across all five panels.
#   EDGE  — a 3-D representation with edges only (pairwise distances)
#   TOPO  — a representation that adds higher-order topology: 2-simplices in the
#           message passing, or persistent-homology descriptors
#   NONE  — no 3-D topology at all: the tabular reference, and the two controls
# The repository's own palette is deliberately not reused: its `test_palette.py`
# found that no four-colour subset of it passes CVD and contrast checks at once.
# --------------------------------------------------------------------------- #
EDGE, TOPO, NONE = PS.BLUE, PS.VERMILLION, PS.GREY
#: Identity never rests on colour alone: the edge-only arm is a circle and the
#: higher-order-topology arm a triangle wherever the two appear side by side.
MK_EDGE, MK_TOPO = "o", "^"

#: Pair-level Pearson correlations between the three out-of-fold shape predictions,
#: and between each encoder and the tabular model's error.  Transcribed from the
#: two tables in `automl/reports/TOPOLOGY_TESTS.md` §1 (commit 1cdeac3): the
#: out-of-fold parquets they are computed from live under `automl/artifacts/`,
#: which upstream does not commit.  See notes.md, "Scientific integrity".
CORRELATIONS = {
    "distance_vs_simplicial": 0.963,
    "tabular_vs_distance": 0.743,
    "tabular_vs_simplicial": 0.749,
    "error_vs_distance": 0.196,
    "error_vs_simplicial": 0.176,
}

#: Repository identifier -> reader-facing name.  Reproduced in notes.md.
NAME_MAP = {
    "c15_plw4": "distance encoder (edges only)",
    "c17_plw4": "simplicial encoder (edges + triangles)",
    "c17_plw2": "simplicial encoder, published training objective",
    "tabular_only": "tabular shape only",
    "g9": "22 persistence summary statistics",
    "g11": "279 persistence-image pixels",
    "block_mean": "block-mean control",
    "anch_q60_q60": "the anchored model's shape channel, unchanged",
    "S0": "topological network",
    "T0w": "matched control network (same recipe, 3-D encoder removed)",
    "repaired": "2-D fingerprint + descriptor network (scaling repaired)",
    "sel_adj_logSF_r2": "adjacent-pair log10 separation-factor R2",
}

#: Any of these surviving into drawn text would leak a repository identifier.
FORBIDDEN = ("anch_", "q60", "c15", "c17", "plw", "snn", "picnn", "T0w",
             "g9__", "adj_r2", "sel_", "_ens", "oof", "resid_blocks")

#: Which R2.  Not overall log D R2: the score of the adjacent-lanthanide pair
#: separation factors, on the log10 scale the labels are measured on.
R2LAB = "adjacent-pair separation-factor $R^2$"


def num(v: float, digits: int = 4, sign: bool = True) -> str:
    """A signed number with a real minus sign, matching the tick labels."""
    return f"{v:{'+' if sign else ''}.{digits}f}".replace("-", "\u2212")
R2SUB = "log$_{10}$ scale  ·  higher is better"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def blend_weights() -> list[dict]:
    """Panel A: what the nested blend fit pays for each 3-D encoder."""
    p = LD.read_json("topo_shape")["part1_encoder_comparison"]
    rows = [
        dict(label="tabular shape only\n(reference)", w_edge=None, w_topo=None,
             r2=p["tabular_only"]["r2"], kind="none"),
        dict(label="distance encoder\n(edges only)",
             w_edge=p["blend_dist"]["w_mean"], w_topo=None,
             r2=p["blend_dist"]["r2"], kind="edge"),
        dict(label="simplicial encoder\n(edges + triangles)",
             w_edge=None, w_topo=p["blend_snn4"]["w_mean"],
             r2=p["blend_snn4"]["r2"], kind="topo"),
        dict(label="the same encoder, as\noriginally published",
             w_edge=None, w_topo=p["blend_snn2"]["w_mean"],
             r2=p["blend_snn2"]["r2"], kind="topo"),
        dict(label="both encoders offered\nat once",
             w_edge=p["blend_dist_plus_snn4"]["w_dist_mean"],
             w_topo=p["blend_dist_plus_snn4"]["w_snn_mean"],
             r2=p["blend_dist_plus_snn4"]["r2"], kind="both"),
    ]
    return rows


def triangle_ablation() -> list[dict]:
    """Panel C: 2-simplices against edge-only message passing, matched seeds."""
    p = LD.read_json("topo_shape")["part2_triangle_ablation"]
    out = []
    for cut in ("f4.0", "f3.5"):
        blk = p[cut]
        radius = cut[1:]
        n = int(blk["triangles"]["n_seeds"])
        assert n == int(blk["no_triangles"]["n_seeds"])
        out.append(dict(label=f"{radius} Å cut-off,\nencoder on its own",
                        edge=blk["no_triangles"]["encoder_alone"]["r2"],
                        topo=blk["triangles"]["encoder_alone"]["r2"], n_seeds=n))
        out.append(dict(label=f"{radius} Å cut-off,\nblended into the model",
                        edge=blk["no_triangles"]["r2"],
                        topo=blk["triangles"]["r2"], n_seeds=n))
    for r in out:
        r["delta"] = r["topo"] - r["edge"]
    # The report quotes the blend differences; check the two we can cross-check.
    assert abs(out[1]["delta"] - p["f4.0"]["blend_r2_tri_minus_notri"]) < 1e-12
    assert abs(out[3]["delta"] - p["f3.5"]["blend_r2_tri_minus_notri"]) < 1e-12
    return out


def persistence_cells() -> list[dict]:
    """Panel D: persistence descriptors given to the shape model only.

    The four-seed ensemble score is the R2 of the *averaged* predictions, so it is
    read from the ``_ens4`` row rather than averaged from the per-seed rows.  The
    reference cell was run twice on two disjoint seed blocks; the block that the
    persistence cells share (42/51/67/83) is the first one in the table, and its
    value is asserted against the +0.3188 quoted in the report.
    """
    df = LD.read("anchored_champion")
    seeds = (42, 51, 67, 83)

    def cell(stem: str) -> tuple[float, list[float]]:
        ens = df.loc[df["cell"] == f"{stem}_ens4", "adj_r2"]
        if ens.empty:
            raise SystemExit(f"{stem}: no ensemble row in anchored_champion.csv")
        per = []
        for s in seeds:
            hit = df.loc[df["cell"] == f"{stem}_s{s}", "adj_r2"]
            if hit.empty:
                raise SystemExit(f"{stem}: seed {s} missing")
            per.append(float(hit.iloc[0]))
        return float(ens.iloc[0]), per

    spec = [
        ("anch_q60_q60", "no persistence\nfeatures (reference)", "none"),
        ("anch_g9", "+ 22 persistence\nsummary statistics", "topo"),
        ("anch_g11", "+ 279 persistence-\nimage pixels", "topo"),
        ("anch_g9_g11", "+ both", "topo"),
        ("anch_g9_bm", "+ the 22 statistics,\nblock-mean control", "control"),
    ]
    rows = []
    for stem, label, kind in spec:
        ens, per = cell(stem)
        rows.append(dict(stem=stem, label=label, kind=kind, r2=ens, seeds=per,
                         n_seeds=len(per)))
    assert abs(rows[0]["r2"] - 0.3188) < 5e-5, rows[0]["r2"]
    return rows


def stack_contrasts() -> list[dict]:
    """Panel E: the three pre-registered stack contrasts, with both intervals."""
    df = LD.read("stack_test").set_index("contrast")
    spec = [
        ("1_primary",
         "adding the topological network\nto the 2-D baseline", "topo"),
        ("2_control",
         "adding a matched control network\n(same recipe, no 3-D input)", "none"),
        ("3_decisive",
         "the topological network against\nthat control, in the same slot", "topo"),
    ]
    out = []
    for key, label, kind in spec:
        r = df.loc[key]
        out.append(dict(key=key, label=label, kind=kind,
                        delta=float(r["delta"]), lo=float(r["lo"]),
                        hi=float(r["hi"]), lo_c=float(r["lo_3test"]),
                        hi_c=float(r["hi_3test"]),
                        clears=bool(float(r["lo_3test"]) > 0),
                        n_boot=int(r["n_boot"])))
    return out


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def panel_a(ax, rows) -> None:
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        if r["kind"] == "none":
            # No encoder is offered at all, so the weight is zero by construction.
            # Marked, because an empty row reads as a missing measurement.
            ax.plot([0.0], [yi], marker="|", ms=8, color=NONE, mew=1.5, zorder=4)
            ax.annotate("no encoder offered", (0.0, yi), xytext=(5, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=PS.BASE - 1.0, color=NONE)
        if r["w_edge"] is not None:
            ax.barh(yi, r["w_edge"], height=0.46, color=EDGE, linewidth=0,
                    zorder=3)
            if r["kind"] != "both":
                # On the joint-fit row the same 0.35 is annotated one row above;
                # repeating it would crowd out the zero that row exists to show.
                ax.annotate(num(r["w_edge"], 2, sign=False), (r["w_edge"], yi),
                            xytext=(4, 0), textcoords="offset points",
                            va="center", ha="left",
                            fontsize=PS.BASE - 1.0, color=EDGE)
        if r["w_topo"] is not None and r["kind"] == "topo":
            ax.barh(yi, r["w_topo"], height=0.46, color=TOPO, linewidth=0,
                    zorder=3)
            ax.annotate(num(r["w_topo"], 2, sign=False), (r["w_topo"], yi),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", ha="left",
                        fontsize=PS.BASE - 1.0, color=TOPO)
        elif r["w_topo"] is not None:
            # Offered alongside the distance encoder the simplicial one is given
            # exactly zero, so there is no bar to draw: mark the end of the
            # distance bar instead, or the row reads as a missing measurement.
            ax.plot([r["w_edge"]], [yi], marker="|", ms=8, color=TOPO, mew=1.5,
                    zorder=4)
            ax.annotate(num(r["w_topo"], 2), (r["w_edge"], yi), xytext=(6, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=PS.BASE - 1.0, color=TOPO)
        ax.annotate(num(r["r2"]), (0.615, yi), ha="right", va="center",
                    fontsize=PS.BASE - 1.0, color=PS.INK)
    ax.annotate("adjacent-pair\n$R^2$ of the blend", (0.615, len(rows) - 0.62),
                ha="right", va="bottom", fontsize=PS.BASE - 1.0, color=NONE,
                linespacing=1.25)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=PS.BASE - 0.5)
    ax.set_xlim(0, 0.62)
    ax.set_ylim(-0.80, len(rows) + 0.22)
    # Reference rules span the rows they refer to, never the reserved bands:
    # an axvline here cut straight through the footnote below the last bar.
    ax.vlines(0, -0.32, len(rows) - 0.68, color=PS.INK, lw=0.7, zorder=1)
    ax.annotate("each encoder is a 32-seed ensemble", (0.012, -0.52),
                ha="left", va="center", fontsize=PS.BASE - 1.0, color=NONE)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.spines["bottom"].set_bounds(0.0, 0.4)
    ax.set_xlabel("mean blend weight on the encoder's\n"
                  "shape prediction  ·  0 = ignored\n"
                  "chosen nested, per held-out extractant")
    PS.strapline(ax, "what the fit pays each encoder")


def panel_b(ax) -> None:
    c = CORRELATIONS
    rows = [
        ("the two encoders\nwith each other", [(c["distance_vs_simplicial"], "both")]),
        ("each encoder with\nthe tabular shape",
         [(c["tabular_vs_distance"], "edge"), (c["tabular_vs_simplicial"], "topo")]),
        ("each encoder with the\ntabular model's error\n(all a blend can use)",
         [(c["error_vs_distance"], "edge"), (c["error_vs_simplicial"], "topo")]),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (_, pts) in zip(y, rows):
        offsets = [0.0] if len(pts) == 1 else [0.17, -0.17]
        for (v, kind), dy in zip(pts, offsets):
            if kind == "both":
                ax.plot([v], [yi + dy], marker="D", ms=5.4, mfc=EDGE, mec=TOPO,
                        mew=1.5, zorder=4)
            else:
                ax.plot([v], [yi + dy],
                        marker=MK_EDGE if kind == "edge" else MK_TOPO, ms=5.4,
                        color=EDGE if kind == "edge" else TOPO,
                        mec="white", mew=0.6, zorder=4)
            # Left of the marker, so the axis can stop at r = 1 rather than run
            # past the largest value a correlation can take.
            ax.annotate(f"{v:.3f}", (v, yi + dy), xytext=(-7, 0), va="center",
                        ha="right", textcoords="offset points",
                        fontsize=PS.BASE - 1.0, color=PS.INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=PS.BASE - 0.5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.62, len(rows) + 0.22)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.spines["bottom"].set_bounds(0.0, 1.0)
    ax.set_xlabel("Pearson $r$ between out-of-fold\n"
                  "predictions  ·  905 adjacent pairs")
    PS.strapline(ax, "how much the encoders agree")


def panel_c(ax, rows) -> None:
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        # The two ends differ by as little as 0.0006, so they are offset
        # vertically as well as coloured and shaped: overlapping markers would
        # read as one measurement rather than two that agree.
        ax.plot([r["edge"], r["topo"]], [yi + 0.15, yi - 0.15], color=PS.PALE,
                lw=2.0, solid_capstyle="round", zorder=2)
        ax.plot([r["edge"]], [yi + 0.15], MK_EDGE, ms=5.4, color=EDGE,
                mec="white", mew=0.6, zorder=4)
        ax.plot([r["topo"]], [yi - 0.15], MK_TOPO, ms=5.4, color=TOPO,
                mec="white", mew=0.6, zorder=4)
        ax.annotate(num(r["delta"]), (0.386, yi), ha="right", va="center",
                    fontsize=PS.BASE - 1.0, color=PS.INK)
    ax.annotate("triangles\n− edges", (0.386, len(rows) - 0.62), ha="right",
                va="bottom", fontsize=PS.BASE - 1.0, color=NONE, linespacing=1.25)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=PS.BASE - 0.5)
    ax.set_xlim(0.212, 0.39)
    ax.set_ylim(-0.62, len(rows) + 0.22)
    ax.set_xticks([0.22, 0.26, 0.30, 0.34])
    ax.spines["bottom"].set_bounds(0.22, 0.34)
    ax.set_xlabel(f"{R2LAB}\n{R2SUB}\n"
                  f"same {rows[0]['n_seeds']} training seeds on both sides")
    PS.strapline(ax, "triangles against edges alone")


def panel_d(ax, rows) -> None:
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        face = {"none": NONE, "topo": TOPO, "control": "white"}[r["kind"]]
        ax.barh(yi, r["r2"], height=0.46, color=face, zorder=3,
                edgecolor=TOPO if r["kind"] == "control" else "none",
                linewidth=1.0 if r["kind"] == "control" else 0.0,
                hatch="////" if r["kind"] == "control" else None)
        # A cap at the bar end, so a score of +0.0022 is still visible as a
        # measured value rather than as an absent bar.
        ax.plot([r["r2"]], [yi], marker="|", ms=9,
                color=NONE if r["kind"] == "none" else TOPO, mew=1.5, zorder=4)
        ax.plot([min(r["seeds"]), max(r["seeds"])], [yi, yi], color=PS.INK,
                lw=0.6, alpha=0.55, zorder=4.5)
        ax.plot(r["seeds"], [yi] * len(r["seeds"]), "o", ms=3.0, mfc="white",
                mec=PS.INK, mew=0.7, ls="none", zorder=5)
        ax.annotate(num(r["r2"]), (0.505, yi), ha="right", va="center",
                    fontsize=PS.BASE - 1.0, color=PS.INK)
    ax.vlines(rows[0]["r2"], -0.32, len(rows) - 0.68, color=NONE, lw=0.8,
              ls=(0, (4, 3)), zorder=1)
    ax.vlines(0, -0.32, len(rows) - 0.68, color=PS.INK, lw=0.9, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=PS.BASE - 0.5)
    ax.set_xlim(-0.20, 0.51)
    ax.set_ylim(-0.62, len(rows) + 0.22)
    ax.set_xticks([-0.1, 0.0, 0.1, 0.2, 0.3])
    ax.spines["bottom"].set_bounds(-0.16, 0.33)
    ax.set_xlabel(f"{R2LAB}\n{R2SUB}\n"
                  "bars: 4-seed ensemble  ·  circles: single seeds")
    PS.strapline(ax, "persistence in the shape model")


def panel_e(ax, rows, cohort: str) -> None:
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        col = TOPO if r["kind"] == "topo" else NONE
        ax.plot([r["lo_c"], r["hi_c"]], [yi, yi], color=col, lw=5.0, alpha=0.28,
                solid_capstyle="butt", zorder=2)
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=col, lw=1.9,
                solid_capstyle="round", zorder=3)
        ax.plot([r["delta"]], [yi], "o", ms=5.4, color=col, mec="white",
                mew=1.0, zorder=4)
        ax.annotate(num(r["delta"], 3), (0.083, yi), ha="right", va="center",
                    fontsize=PS.BASE - 1.0, color=PS.INK)
        ax.annotate("clears zero" if r["clears"] else "spans zero",
                    (0.088, yi), ha="left", va="center",
                    fontsize=PS.BASE - 1.0, color=PS.INK)
    ax.vlines(0, -0.45, len(rows) - 0.55, color=PS.INK, lw=1.1, zorder=1)
    ax.annotate("corrected interval", (0.088, len(rows) - 0.55), ha="left",
                va="bottom", fontsize=PS.BASE - 1.0, color=NONE)
    ax.annotate(cohort, (-0.043, -0.92), ha="left", va="center",
                fontsize=PS.BASE - 1.0, color=NONE)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=PS.BASE - 0.5)
    ax.set_xlim(-0.045, 0.152)
    ax.set_ylim(-1.25, len(rows) + 0.16)
    ax.set_xticks([-0.02, 0.0, 0.02, 0.04, 0.06])
    ax.spines["bottom"].set_bounds(-0.02, 0.07)
    ax.set_xlabel(f"Δ {R2LAB} (log$_{{10}}$ scale)  ·  "
                  "positive = the added arm is better\n"
                  "thick line: 90 % paired cluster bootstrap over extractants\n"
                  "pale band: the same interval widened for the three "
                  "pre-registered tests")
    PS.strapline(ax, "pre-registered stack tests against a weaker, 2-D baseline")


# --------------------------------------------------------------------------- #
def main() -> int:
    PS.apply()

    rows_a = blend_weights()
    rows_c = triangle_ablation()
    rows_d = persistence_cells()
    rows_e = stack_contrasts()
    cohort = ("4,746 measurements · 162 extractants · 905 adjacent-lanthanide "
              "pairs · leave-extractants-out CV, 5 folds × 3 repeats")

    fig = plt.figure(figsize=(PS.W2, 6.75))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.00, 1.00, 0.80],
                          width_ratios=[1.00, 1.00])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, :])

    panel_a(axA, rows_a)
    panel_b(axB)
    panel_c(axC, rows_c)
    panel_d(axD, rows_d)
    panel_e(axE, rows_e, cohort)

    for ax in (axA, axB, axC, axD, axE):
        ax.grid(False)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)

    def key(colour, marker, label, **kw):
        return Line2D([], [], color=colour, lw=5.0, marker=marker, ms=5.0,
                      mec="white", mew=0.7, label=label, **kw)

    handles = [
        key(EDGE, MK_EDGE, "edges only"),
        key(NONE, "s", "no 3-D topology: reference or control"),
        key(TOPO, MK_TOPO, "2-simplices, or persistent homology"),
        Patch(facecolor="white", edgecolor=TOPO, hatch="////",
              label="same columns, within-block variation removed"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2,
               fontsize=PS.BASE - 1.0, handlelength=1.6, handleheight=0.9,
               columnspacing=2.0)

    PS.add_panel_letters(fig, [axA, axB, axC, axD, axE])

    # No repository identifier may reach the artwork.
    leaked = []
    for t in fig.findobj(matplotlib.text.Text):
        s = str(t.get_text())
        if not s.strip() or not t.get_visible():
            continue
        for token in FORBIDDEN:
            if token.lower() in s.lower():
                leaked.append((token, s))
    if leaked:
        raise SystemExit(f"repository identifiers in the artwork: {leaked}")

    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    print(f"declared {w:.2f} x {h:.2f} in; drawn {tight.width:.2f} x "
          f"{tight.height:.2f} in "
          f"({max(tight.width / w, tight.height / h):.3f}x)")

    report = PS.save(fig, HERE, "figure", strict=False)
    print(report)

    values = {
        "figure": "figure_04_topology_adds_nothing",
        "source_repository": LD.provenance(),
        "cohort": cohort,
        "metric": "adjacent-pair log10 separation-factor R2, out-of-fold, "
                  "leave-extractants-out; higher is better",
        "panelA_blend_weight": [
            {k: v for k, v in r.items() if k != "label"} | {"row": r["label"]}
            for r in rows_a],
        "panelB_correlations": {
            "values": CORRELATIONS,
            "provenance": "transcribed from automl/reports/TOPOLOGY_TESTS.md "
                          "section 1 at commit 1cdeac3; computed upstream from "
                          "out-of-fold parquets under automl/artifacts/, which "
                          "is not committed upstream and is therefore not "
                          "vendored here"},
        "panelC_triangle_ablation": rows_c,
        "panelD_persistence": rows_d,
        "panelE_stack_contrasts": rows_e,
        "name_map": NAME_MAP,
        "lint": {"ok": report.ok, "violations": [str(v) for v in report.violations]},
    }
    (HERE / "values.json").write_text(json.dumps(values, indent=1))

    print("A weights:", [(r["w_edge"], r["w_topo"]) for r in rows_a])
    print("C deltas :", [round(r["delta"], 4) for r in rows_c])
    print("D r2     :", [round(r["r2"], 4) for r in rows_d])
    print("E deltas :", [(r["key"], round(r["delta"], 4), r["clears"])
                         for r in rows_e])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

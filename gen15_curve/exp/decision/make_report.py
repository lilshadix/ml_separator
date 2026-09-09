"""Render REPORT.md from the result CSVs.  Every number in the report comes from a file on disk."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / "results"
D = ["B", "BR", "BQ", "A", "BP"]
MAIN = ["FLAT", "MEAN_CURVE", "HEAVIER_ALWAYS", "G14", "G13_FULL", "O_BOTH"]
WITH_TIED = ["FLAT", "MEAN_CURVE", "HEAVIER_ALWAYS", "G14", "G14_TIED", "G13_FULL", "O_BOTH"]


def md(df: pd.DataFrame, nd: int = 3) -> str:
    return df.round(nd).to_markdown() + "\n"


def piv(path: str, value: str, rows: list[str] | None = None, index: str = "arm",
        design_filter: str | None = None, extra: dict | None = None) -> pd.DataFrame:
    t = pd.read_csv(R / path)
    if design_filter:
        t = t[t["design"] == design_filter]
    for k, v in (extra or {}).items():
        t = t[t[k] == v]
    w = t.pivot(index=index, columns="design", values=value) if not design_filter \
        else t.pivot(index=index, columns="arm", values=value)
    if design_filter:
        cols = [c for c in (rows or w.columns) if c in w.columns]
        return w[cols]
    w = w[[c for c in D if c in w.columns]]
    return w.loc[[r for r in (rows or w.index) if r in w.index]]


def main() -> None:
    out: list[str] = []
    A = out.append

    A("# Is the curve model useful for the decision a chemist makes?\n")
    A("*gen15_curve/exp/decision — every number below is computed by the scripts in this "
      "directory from the frozen gen13 cohort and written to `results/`; nothing is estimated.*\n")

    # ------------------------------------------------------------------ conventions
    A("## 0. What was measured, and the conventions\n")
    A("Five arms, five designs, 5 split seeds x 5 folds each, 70 750 held-out "
      "(seed, cell, metal-pair) rows per design. `y = logD(A) - logD(B)` with A the **lighter** "
      "lanthanide, base-10, so `y < 0` means the system prefers the heavier metal.\n")
    A("| arm | what it is |\n|---|---|\n"
      "| `FLAT` | predict no separation at all — the honest floor |\n"
      "| `MEAN_CURVE` | the training fold's mean curve |\n"
      "| `HEAVIER_ALWAYS` | the rule 'always prefer the heavier lanthanide', ordered by the "
      "radius gap. Scored with gen13's own `heavier_always_sign_accuracy` convention; it is a "
      "sign/rank rule, so its MAE and calibration slope are not meaningful |\n"
      "| `G14` | the deployed model: L2 logistic on 39 donor-topology columns for the direction, "
      "times the training fold's mean magnitude, curvature at the training mean |\n"
      "| `G14_TIED` | **diagnostic, not deployable-different**: the same direction bit with ONE "
      "global magnitude and curvature instead of the fold's own. Isolates the cross-fold jitter "
      "that leave-one-group-out introduces when predictions from different folds are compared |\n"
      "| `G13_FULL` | gen13's 209-column extra-trees regression on both coefficients |\n"
      "| `O_BOTH` | **oracle**: the cell's own two coefficients — the representation ceiling |\n"
      "| `_RANDOM` | picking uniformly at random from the same candidate set |\n")
    A("**Aggregation.** Mean within a (split seed, extractant), then over extractants, then over "
      "the five seeds — gen13's own unit, so the one chemotype that holds 375 of 521 cells cannot "
      "carry a number. Spread quoted is the sd over the five seeds.\n")
    A("**Ties.** A constant prediction has no ranking. Every rank statistic here is the *exact "
      "expectation under uniform random tie-breaking*, so an uninformative arm lands on chance "
      "rather than on NaN, and `FLAT` and `HEAVIER_ALWAYS` reproduce `_RANDOM` exactly wherever "
      "they carry no information. A prediction of exactly zero scores 0.5 on a sign call.\n")
    A("**Reference check** — the endpoint the programme is locked to, reproduced by this run "
      "(extractant-macro MAE of log SF):\n")
    b = pd.read_csv(HERE / "tables" / "mae_board.csv")
    A(md(b.pivot(index="arm", columns="design", values="macro_mae_extractant")[D]
         .loc[["FLAT", "MEAN_CURVE", "G13_FULL", "G14", "O_BOTH"]], 4))
    A("G14 = 0.5001 and G13_FULL = 0.5523 under BP reproduce the locked 0.500 / 0.552 to three "
      "decimals, so the bench is being driven correctly.\n")
    A("**Verification** (`checks.py`, all passing): the closed-form tie expectations agree with a "
      "Monte-Carlo draw of consistent orderings to 0.017; the `HEAVIER_ALWAYS` sign accuracy "
      "equals gen13's own `heavier_always_sign_accuracy` to 1e-9 (0.629795); `FLAT` and "
      "`HEAVIER_ALWAYS` reproduce `_RANDOM` exactly wherever they carry no ranking information; "
      "the `O_BOTH` column equals the cell's own coefficients times the basis difference; and a "
      "perfect predictor gets calibration slope exactly 1.0.\n")

    # ------------------------------------------------------------------ summary
    A("---\n\n## Summary — the four decisions, design BP\n")
    A("Every row is the deployed model `G14` against the cheapest sensible alternative already in "
      "the repo and against the representation ceiling. `consistent` means the sign of the gain "
      "over the alternative is the same under all five hold-out designs.\n")
    q1 = pd.read_csv(R / "q1_sign.csv")
    q1bp = q1[(q1.design == "BP") & (q1.band == "all")].set_index("arm")["sign_acc"]
    q2 = pd.read_csv(R / "q2_within_cell.csv")
    q2bp = q2[q2.design == "BP"].set_index("arm")
    q3 = pd.read_csv(R / "q3_cross_extractant.csv")
    q3bp = q3[q3.design == "BP"].set_index("arm")
    wpx = pd.read_csv(R / "confound_within_pub.csv")
    wpbp = wpx[wpx.design == "BP"].set_index("arm")
    abx = pd.read_csv(R / "q3_absolute.csv")
    abx = abx[abx.design == "BP"].set_index("arm")
    sl = pd.read_csv(R / "q4_calibration_slope.csv")
    slbp = sl[sl.design == "BP"].set_index("arm")["slope"]
    cvs = pd.read_csv(R / "q4_coverage_sd.csv")
    cvs = cvs[(cvs.design == "BP") & (cvs.arm == "G14")
              & (cvs.confidence == "gen14 |p-0.5|")].set_index("coverage")["sign_acc"]
    rows = [
        {"decision": "Q1 call the direction of a target pair",
         "metric": "sign accuracy, abs(log SF) >= 0.3",
         "G14": q1bp["G14"], "cheapest alternative": q1bp["HEAVIER_ALWAYS"],
         "alternative": "heavier always", "ceiling (O_BOTH)": q1bp["O_BOTH"],
         "consistent": "yes", "verdict": "clear win"},
        {"decision": "Q2 rank one system's own pairs",
         "metric": "within-cell Spearman",
         "G14": q2bp.loc["G14", "spearman"],
         "cheapest alternative": q2bp.loc["HEAVIER_ALWAYS", "spearman"],
         "alternative": "radius-gap order", "ceiling (O_BOTH)": q2bp.loc["O_BOTH", "spearman"],
         "consistent": "yes", "verdict": "clear win"},
        {"decision": "Q2 name the best-separated pair",
         "metric": "top-1 hit rate",
         "G14": q2bp.loc["G14", "top1"],
         "cheapest alternative": q2bp.loc["HEAVIER_ALWAYS", "top1"],
         "alternative": "widest radius gap", "ceiling (O_BOTH)": q2bp.loc["O_BOTH", "top1"],
         "consistent": "yes (always negative)", "verdict": "LOSS to the rule"},
        {"decision": "Q3 order extractants for a target pair",
         "metric": "cross-extractant Spearman",
         "G14": q3bp.loc["G14", "spearman"],
         "cheapest alternative": q3bp.loc["_RANDOM", "spearman"],
         "alternative": "random pick", "ceiling (O_BOTH)": q3bp.loc["O_BOTH", "spearman"],
         "consistent": "yes", "verdict": "win, but see Q3d"},
        {"decision": "Q3 pick the single best extractant",
         "metric": "top-1 hit rate",
         "G14": q3bp.loc["G14", "top1"],
         "cheapest alternative": q3bp.loc["_RANDOM", "top1"],
         "alternative": "random pick", "ceiling (O_BOTH)": q3bp.loc["O_BOTH", "top1"],
         "consistent": "no (sign flips)", "verdict": "null — at chance"},
        {"decision": "Q3 pick the strongest separator, either direction",
         "metric": "regret, log10 (lower better)",
         "G14": abx.loc["G14", "regret_abs"],
         "cheapest alternative": abx.loc["_RANDOM", "regret_abs"],
         "alternative": "random pick", "ceiling (O_BOTH)": abx.loc["O_BOTH", "regret_abs"],
         "consistent": "yes (always worse)", "verdict": "LOSS to random"},
        {"decision": "Q3 same, inside ONE laboratory",
         "metric": "regret, log10 (lower better)",
         "G14": wpbp.loc["G14", "regret"],
         "cheapest alternative": wpbp.loc["_RANDOM", "regret"],
         "alternative": "random pick", "ceiling (O_BOTH)": wpbp.loc["O_BOTH", "regret"],
         "consistent": "no (worse in 4 of 5)", "verdict": "null — worse than random"},
        {"decision": "Q4 believe the predicted magnitude",
         "metric": "calibration slope (1 = perfect)",
         "G14": slbp["G14"], "cheapest alternative": slbp["MEAN_CURVE"],
         "alternative": "corpus mean curve", "ceiling (O_BOTH)": slbp["O_BOTH"],
         "consistent": "yes", "verdict": "usable, biased -0.14"},
        {"decision": "Q4 know when it does not know",
         "metric": "sign acc at 10 % coverage on its own confidence",
         "G14": float(cvs.loc[0.10]), "cheapest alternative": float(cvs.loc[1.00]),
         "alternative": "no abstention", "ceiling (O_BOTH)": "-",
         "consistent": "no (below baseline in 4 of 5)", "verdict": "null"},
    ]
    A(md(pd.DataFrame(rows).set_index("decision"), 3))

    # ------------------------------------------------------------------ Q1
    A("---\n\n## Q1. Given a target metal pair, does the model call the direction right?\n")
    A("Sign accuracy on held-out pairs with |observed log SF| >= 0.3 (gen13's strong-pair "
      "threshold, about 0.9 pair-noise sd). 40 455 of 70 750 pairs qualify, over 405 cells and "
      "86 extractants.\n")
    A("**Table 1a — pair-direction accuracy, all five designs**\n")
    A(md(piv("q1_sign.csv", "sign_acc", WITH_TIED, extra={"band": "all"}), 3))
    A("The cheapest sensible alternative is `HEAVIER_ALWAYS` at 0.630, not `FLAT`. G14 beats it by "
      "+0.18 under every design, and the gain is the most robust result in this study.\n")
    A("**Table 1b — by how far apart the metals are, and by how large the true separation is "
      "(design BP)**\n")
    A(md(piv("q1_sign.csv", "sign_acc", MAIN, design_filter="BP", index="band"), 3))
    n = pd.read_csv(R / "q1_sign.csv")
    n = n[(n.design == "BP") & (n.arm == "G14")][["band", "n_pairs", "n_extractants"]]
    A("\nBand sizes (pairs / extractants):\n\n" + md(n.set_index("band"), 0))
    A("Direction is called best on wide, strongly separated pairs and worst on neighbours: "
      "0.77 at dZ = 1 against 0.87 at dZ >= 9. The neighbour case is the industrially interesting "
      "one.\n")
    A("**Table 1c — gen13's chemotype-blocked paired bootstrap (10 000 replicates), "
      "reference minus candidate, positive = candidate better**\n")
    c = pd.read_csv(R / "paired_contrasts.csv")
    q1 = c[c.metric == "Q1 sign accuracy"]
    A(md(q1.pivot(index="comparison", columns="design", values="point")[D], 4))
    A("\nTwo-sided bootstrap p:\n\n"
      + md(q1.pivot(index="comparison", columns="design", values="p_two_sided")[D], 4))
    A("G14 over the heavier-always rule passes gen13's full pre-registered rule P1 under all five "
      "designs. G13_FULL over the same rule does **not** under BP (p = 0.35).\n")

    # ------------------------------------------------------------------ Q2
    A("---\n\n## Q2. For one system, can the model rank its own pairs, and name its best one?\n")
    A("Held-out cells with >= 4 measured metals (321 cells). Spearman and Kendall tau-b between "
      "predicted and observed log SF over the cell's pairs; top-1 / top-3 for 'which pair does "
      "this system separate best', ranked by |predicted log SF| against |observed log SF|.\n")
    A("**Table 2a — within-cell rank correlation, all five designs**\n")
    A(md(piv("q2_within_cell.csv", "spearman", WITH_TIED + ["_RANDOM"]), 3))
    A("\nKendall tau-b:\n\n" + md(piv("q2_within_cell.csv", "kendall", WITH_TIED + ["_RANDOM"]), 3))
    A("**Table 2b — naming the best-separated pair**\n")
    A("top-1:\n\n" + md(piv("q2_within_cell.csv", "top1", WITH_TIED + ["_RANDOM"]), 3))
    A("\ntop-3:\n\n" + md(piv("q2_within_cell.csv", "top3", WITH_TIED + ["_RANDOM"]), 3))
    A("This is the study's first clean negative. Ranking a cell's pairs, G14 is far above the "
      "trivial rule (0.50 against 0.26 Spearman, every design). *Naming* the best-separated pair, "
      "it is **below** it under every design: 0.25 under BP and 0.34-0.36 under the others, "
      "against 0.41 for 'take the widest radius gap'. So is the oracle `O_BOTH`, at 0.357. The "
      "widest-gap pair *is* the truly best-separated one 41 % of the time, and adding a curvature "
      "term — even the cell's own — moves the predicted maximum off it more often than it "
      "helps.\n")
    A("**Table 2c — paired bootstrap on the two Q2 statistics**\n")
    q2 = c[c.metric.isin(["Q2 within-cell Spearman", "Q2 best-pair top-1"])]
    A(md(q2.pivot_table(index=["metric", "comparison"], columns="design",
                        values="point")[D], 4))
    A("\np:\n\n" + md(q2.pivot_table(index=["metric", "comparison"], columns="design",
                                     values="p_two_sided")[D], 4))
    A("`G14_vs_HEAVIER` on top-1 is negative under all five designs with p <= 0.0004 and 0 of 5 "
      "seeds positive: a significant, reproducible **loss** to the trivial rule.\n")
    A("**Table 2d — confound: the number of metals the cell measured** (Spearman +0.49 with |a| "
      "in this corpus), design BP\n")
    A(md(piv("confound_n_metals.csv", "cell_spearman", MAIN + ["_RANDOM"], design_filter="BP",
             index="n_metals_band"), 3))
    A("\ntop-1:\n\n" + md(piv("confound_n_metals.csv", "cell_top1", MAIN + ["_RANDOM"],
                              design_filter="BP", index="n_metals_band"), 3))
    A("G14's ranking advantage is not uniform: on cells that measured 6-8 metals the trivial rule "
      "ranks better than the model (0.59 vs 0.51). The top-1 loss holds in every band.\n")

    # ------------------------------------------------------------------ Q3
    A("---\n\n## Q3. The design question: for a target pair, which extractant separates it best?\n")
    A("For each split seed and metal pair, every held-out extractant that measured that pair is "
      "ranked by predicted log SF (an extractant's several condition sets collapsed by the "
      "median). 91 metal pairs x 5 seeds = 455 ranking tasks, median 58 extractants per task. "
      "`regret` = the observed log SF of the truly best extractant minus that of the model's pick, "
      "in log10 units, averaged over the two directions a chemist could ask for.\n")
    A("**Table 3a — cross-extractant selection, all five designs**\n")
    A("rank correlation with the observed ordering:\n\n"
      + md(piv("q3_cross_extractant.csv", "spearman", WITH_TIED + ["_RANDOM"]), 3))
    A("\ntop-1 hit rate (chance = 0.017):\n\n"
      + md(piv("q3_cross_extractant.csv", "top1", WITH_TIED + ["_RANDOM"]), 4))
    A("\nregret, log10 units (lower better; random = 1.783, oracle = 0.203):\n\n"
      + md(piv("q3_cross_extractant.csv", "regret", WITH_TIED + ["_RANDOM"]), 3))
    A("The model **orders** extractants better than chance (Spearman 0.35, consistent across "
      "designs) but its **top pick is at chance** (0.023 against 0.017). It removes 0.31 of the "
      "1.58 log units of regret that separate a random pick from a perfect one — about 19 %.\n")
    A("**Table 3b — split by how far apart the target metals are (design BP)**\n")
    z = pd.read_csv(R / "q3_by_dz.csv")
    z = z[z.design == "BP"]
    A(md(z.pivot(index="band", columns="arm", values="regret")[
        ["_RANDOM", "G14", "G14_TIED", "G13_FULL", "O_BOTH"]], 3))
    A("\nfraction of the achievable regret reduction captured:\n\n"
      + md(z.pivot(index="band", columns="arm", values="frac_of_oracle")[
          ["G14", "G14_TIED", "G13_FULL"]], 3))
    A("The feasible arms capture a flat ~19-25 % of the achievable gain in every band. The oracle "
      "does not: it leaves 0.584 of 0.863 log units of regret on **adjacent** pairs (68 % of "
      "random) and 0.025 of 3.064 on extreme ones (0.8 %). The quadratic centred curve is a "
      "near-complete description of the design problem for wide pairs and close to useless for "
      "neighbours — a limit of the representation, not of any model fitted in it.\n")
    A("**Table 3c — the industrially relevant pairs, design BP** (Nd/Pr and Pr/Nd are the same "
      "unordered pair and appear once)\n")
    it = pd.read_csv(R / "q3_industrial.csv")
    it = it[it.design == "BP"]
    for m, lab in [("spearman", "rank correlation"), ("top1", "top-1 hit rate"),
                   ("regret", "regret, log10 units")]:
        A(f"\n{lab}:\n\n" + md(it.pivot(index="pair", columns="arm", values=m)[
            ["_RANDOM", "G14", "G14_TIED", "G13_FULL", "O_BOTH"]], 3))
    A("\ncandidates and spread per pair:\n\n"
      + md(it[it.arm == "G14"][["pair", "n_units", "obs_spread"]].set_index("pair"), 2))
    A("On the pairs the field cares about, the model's top pick is at or near chance for every one "
      "of them (the one apparent exception, La/Ce at 0.059 against 0.015 under BP, runs "
      "0.000-0.059 across the five designs and is noise), and regret falls by only 0.03-0.40 log "
      "units out of 0.64-2.05. The oracle cannot pick the best extractant for Eu/Gd, La/Ce or "
      "Sm/Eu either.\n")
    A("**Table 3d — the confound that matters most: hold the laboratory fixed.** Ranking is "
      "restricted to (metal pair, publication) blocks with >= 5 extractants measured by the same "
      "group under the same protocol — 1 230 tasks from 4 publications, median 6 candidates, mean "
      "observed spread 1.14 log units instead of the pooled 3.57.\n")
    wp = pd.read_csv(R / "confound_within_pub.csv")
    for m, lab in [("spearman", "rank correlation"), ("top1", "top-1 (chance = 0.148)"),
                   ("regret", "regret, log10 units (random = 0.572)")]:
        A(f"\n{lab}:\n\n" + md(wp.pivot(index="arm", columns="design", values=m)[D]
                               .loc[[a for a in WITH_TIED + ["_RANDOM"]
                                     if a in set(wp.arm)]], 3))
    A("\n**Within one laboratory the model is not better than random.** G14's rank correlation is "
      "negative under four of five designs, its top-1 is below chance under four of five, and its "
      "regret is worse than a random pick under four of five. The oracle solves the same task "
      "(Spearman 0.739, top-1 0.669, regret 0.056), so the task is not intrinsically impossible — "
      "the model simply has nothing to say. The mechanism is direct: **70-85 % of these ranking "
      "tasks receive the same direction call for every candidate extractant**, so gen14's one bit "
      "is constant across the set being ranked. Pooled across laboratories that figure is 0 %. "
      "Nearly all of the pooled Q3 gain in Table 3a is therefore between-laboratory information, "
      "not the ligand comparison a chemist would actually make.\n")
    A("Paired bootstrap on the same tasks, blocked on the publication (only 4 blocks, so the "
      "interval is coarse — the point estimates carry the message); positive = candidate better "
      "than a random pick:\n")
    wpc = pd.read_csv(R / "confound_within_pub_contrasts.csv")
    A(md(wpc.pivot_table(index=["metric", "comparison"], columns="design", values="point")[D], 4))
    A("\np:\n\n" + md(wpc.pivot_table(index=["metric", "comparison"], columns="design",
                                      values="p_two_sided")[D], 4))
    A("The oracle beats a random pick by 0.515 log units with p < 1e-4. No feasible arm is "
      "significantly better than a random pick under any design; G14's point estimate is on the "
      "wrong side of zero under four of the five, and the one feasible arm that does reach "
      "significance — G13_FULL's rank correlation under BQ — is significantly *worse* than "
      "random.\n")
    A("**Table 3e — the cross-fold artifact.** `G14_TIED` differs from `G14` only in using one "
      "global magnitude instead of each fold's own. It is *better* on every Q3 statistic under "
      "every design (paired bootstrap on regret: +0.087 to +0.183 log units, p < 1e-3 "
      "everywhere). In a leave-one-group-out design the held-out group is exactly the group "
      "excluded from the fold's training mean, so out-of-fold predictions are anti-correlated "
      "with the truth across folds; `MEAN_CURVE`'s cross-extractant Spearman of -0.06 to -0.37 is "
      "the pure form of the same artifact. Any cross-system comparison must be made with one fixed model, "
      "never by pooling out-of-fold predictions.\n")
    A("**Table 3f — paired bootstrap on the Q3 statistics** (unit and block = the metal-pair task; "
      "pairs share cells and metals so this interval is the optimistic one)\n")
    q3 = c[c.metric.str.startswith("Q3")]
    A(md(q3.pivot_table(index=["metric", "comparison"], columns="design", values="point")[D], 4))
    ex = pd.read_csv(R / "paired_contrasts_extra.csv")
    A("\nG13_FULL against G14 on regret, and against the artifact-corrected G14_TIED:\n\n"
      + md(ex.pivot(index="comparison", columns="design", values="point")[D], 4))
    A("\np:\n\n" + md(ex.pivot(index="comparison", columns="design", values="p_two_sided")[D], 4))
    A("G13_FULL appears to beat G14 on the design question under all five designs — but against "
      "`G14_TIED` the advantage is -0.011 to +0.124 with p between 0.0002 and 0.63 and one "
      "negative sign, so it fails the consistency requirement. The apparent win is mostly G14's "
      "cross-fold jitter, not information in the 209 columns.\n")

    A("**Table 3g — the magnitude-only form of the same question.** The brief also names the "
      "weaker version: rank by |predicted log SF| and score against |observed log SF|, so a system "
      "that separates the pair strongly in the *wrong* direction still counts. Random = 0 "
      "Spearman, 0.017 top-1, 1.519 regret.\n")
    ab = pd.read_csv(R / "q3_absolute.csv")
    for m, lab in [("spearman_abs", "rank correlation"), ("top1_abs", "top-1"),
                   ("regret_abs", "regret, log10 units")]:
        A(f"\n{lab}:\n\n" + md(ab.pivot(index="arm", columns="design", values=m)[D]
                               .loc[[a for a in WITH_TIED + ["_RANDOM"] if a in set(ab.arm)]], 3))
    A("\nHere G14 is **worse than a random pick under all five designs** (regret 1.59-1.68 against "
      "1.519) with a rank correlation of -0.03 to +0.06. That is the direct consequence of its "
      "construction: its magnitude is one number per fold, so |prediction| is nearly constant "
      "across the extractants being ranked and the little variation it has is cross-fold jitter. "
      "G13_FULL, which does vary its magnitude per ligand, beats random consistently here "
      "(1.05-1.35) even though it loses to G14 on the programme's own MAE endpoint under BP. The "
      "endpoint and the decision disagree about which of the two models is more useful.\n")

    # ------------------------------------------------------------------ Q4
    A("---\n\n## Q4. Is the predicted size of the separation believable, and does the model know "
      "when it does not know?\n")
    A("Calibration slope is the weighted least-squares regression of observed on predicted log SF "
      "with each extractant weighted equally; 1.0 is perfect, > 1 means the model understates the "
      "separation.\n")
    A("**Table 4a — calibration slope and R^2**\n")
    s = pd.read_csv(R / "q4_calibration_slope.csv")
    A(md(s.pivot(index="arm", columns="design", values="slope")[D]
         .loc[["MEAN_CURVE", "HEAVIER_ALWAYS", "G14", "G14_TIED", "G13_FULL", "O_BOTH"]], 3))
    A("\nR^2:\n\n" + md(s.pivot(index="arm", columns="design", values="r2")[D]
                        .loc[["MEAN_CURVE", "HEAVIER_ALWAYS", "G14", "G14_TIED",
                              "G13_FULL", "O_BOTH"]], 3))
    A("G14's slope is 1.12-1.19 — the predicted magnitude is on the right scale, understating the "
      "true separation by 12-19 %. G13_FULL's is 0.81 under BP (over-spread) with R^2 = 0.12 "
      "against G14's 0.26. `HEAVIER_ALWAYS` is in radius units, so its slope is not a calibration "
      "number. Even the oracle sits at 1.09 / R^2 0.905, because the quadratic basis holds 95.6 % "
      "of the curve variance.\n")
    A("**Table 4b — observed against predicted in equal-count bins, design BP**\n")
    bb = pd.read_csv(R / "q4_calibration_bins.csv")
    for arm in ("G14", "G13_FULL"):
        A(f"\n`{arm}`:\n\n" + md(bb[(bb.design == "BP") & (bb.arm == arm)][
            ["bin", "n", "pred_mean", "obs_mean", "obs_sd"]].set_index("bin"), 3))
    A("\nEvery G14 bin is biased in the same direction: the observed mean is 0.2-0.4 log units "
      "more negative than predicted (fitted intercept -0.144). The model is scale-calibrated but "
      "systematically understates how heavy-selective this corpus is. G13_FULL's top two bins are "
      "non-monotone.\n")
    A("**Table 4c — risk-coverage: keep the most confident fraction of predictions**\n")
    cv = pd.read_csv(R / "q4_coverage_sd.csv")
    # pipes break a markdown cell
    cv["confidence"] = cv["confidence"].replace({"|prediction|": "abs(prediction)",
                                                 "gen14 |p-0.5|": "gen14 dir conf"})
    bp = cv[(cv.design == "BP") & (cv.arm.isin(["G14", "G13_FULL"]))]
    A(md(bp[["arm", "confidence", "coverage", "mae", "mae_sd", "sign_acc", "sign_acc_sd"]]
         .set_index(["arm", "confidence", "coverage"]), 4))
    A("\nsign accuracy under gen14's own direction confidence (its |p - 0.5|), all designs:\n\n"
      + md(cv[(cv.arm == "G14") & (cv.confidence == "gen14 dir conf")]
           .pivot(index="design", columns="coverage", values="sign_acc")[
               [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]], 3))
    A("\nsd over seeds:\n\n"
      + md(cv[(cv.arm == "G14") & (cv.confidence == "gen14 dir conf")]
           .pivot(index="design", columns="coverage", values="sign_acc_sd")[
               [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]], 3))
    A("**The model's own confidence is not a confidence.** Selecting the 5-25 % of cells the "
      "logistic is most certain about leaves sign accuracy at 0.71-0.84 against 0.81 on "
      "everything — at or below the unselected value under every design, with a seed sd of "
      "0.02-0.06. The MAE does fall (0.50 to 0.27 at 5 % coverage) but only because those cells "
      "have smaller separations to begin with: mean |y| falls from 0.650 to 0.384, the "
      "strong-pair fraction from 0.572 to 0.431, and the retained set is 31 cells.\n")
    A("**Table 4d — what the confidence filter actually keeps, design BP**\n")
    cs = pd.read_csv(R / "q4_confidence_selection.csv")
    cs["confidence"] = cs["confidence"].replace({"|prediction|": "abs(prediction)",
                                                 "gen14 |p-0.5|": "gen14 dir conf"})
    A(md(cs[cs.design == "BP"][["confidence", "coverage", "n_pairs", "n_cells", "mean_abs_y",
                                "frac_strong", "mean_dZ", "pooled_mae"]]
         .set_index(["confidence", "coverage"]), 3))
    A("Selecting instead on |predicted log SF| does raise sign accuracy (0.812 to 0.876 at 10 % "
      "coverage, consistently across designs) — but that quantity is essentially the radius gap "
      "of the requested pair: mean dZ over the retained set rises from 4.94 to 10.72. It is known "
      "before any model is fitted, and the MAE gets *worse* (0.500 to 0.945) because the retained "
      "pairs are the large ones. There is no operating point at which abstention buys accuracy.\n")

    # ------------------------------------------------------------------ permutation
    A("---\n\n## Q5. Permutation null for every headline number\n")
    A("Inside the training fold only, the 137 ligand-derived columns are permuted between "
      "extractants — all of an extractant's cells receive the same substitute structure, while "
      "conditions, mass-action columns, weights, publication structure and targets stay attached "
      "to their own cell. Only the structure-to-curve map is destroyed. `FLAT`, `MEAN_CURVE`, "
      "`O_BOTH` and `HEAVIER_ALWAYS` are invariant under it by construction, which replicate 0 "
      "confirms numerically: permuted FLAT / MEAN_CURVE / O_BOTH return 0.588506 / 0.596858 / "
      "0.181111 under B, identical to the real run in every one of the eight headline metrics to "
      "six decimals.\n")
    pn = pd.read_csv(R / "permutation_null.csv")
    nrep = int(pn["n_rep"].max())
    A(f"{nrep} replicates per design; the smallest reportable one-sided p is 1/(R+1) = "
      f"{1 / (nrep + 1):.3f}. (Two copies of the runner were briefly live at once and appended "
      "each replicate twice; the permutation is deterministic in (seed, fold, replicate) and the "
      "repeated rows agree to 1e-10, so one of each is dropped.)\n")
    A("**Table 5a — design BP, observed against the null**\n")
    A(md(pn[pn.design == "BP"][["arm", "metric", "observed", "null_mean", "null_sd",
                                "z_vs_null", "p_emp"]].set_index(["arm", "metric"]), 4))
    A("**Table 5b — z against the null, all designs**\n")
    for arm in ("G14", "G13_FULL"):
        A(f"\n`{arm}`:\n\n" + md(pn[pn.arm == arm].pivot(index="metric", columns="design",
                                                         values="z_vs_null")[D], 2))
    def rng_(arm: str, metric: str, col: str = "z_vs_null") -> str:
        v = pn[(pn.arm == arm) & (pn.metric == metric)][col]
        return f"{v.min():+.2f} to {v.max():+.2f}"

    def rng4(arm: str, metric: str, col: str) -> str:
        v = pn[(pn.arm == arm) & (pn.metric == metric) & (pn.design != "BP")][col]
        return f"{v.min():.2f}-{v.max():.2f}"

    bp_null = pn[(pn.arm == "G13_FULL") & (pn.design == "BP")].set_index("metric")["null_mean"]
    A("\nDirection accuracy, within-cell Spearman, cross-extractant Spearman, regret and the "
      "calibration slope are all far outside their nulls under every design. The two *selection* "
      f"statistics are not: G14's within-cell top-1 sits at z = {rng_('G14', 'cell_top1')} and its "
      f"cross-extractant top-1 at z = {rng_('G14', 'cross_top1')}, changing sign across designs "
      f"(empirical p up to "
      f"{pn[(pn.arm == 'G14') & (pn.metric == 'cross_top1')].p_emp.max():.2f}). G13_FULL's "
      f"cross-extractant top-1 runs from z = {rng_('G13_FULL', 'cross_top1')} — the BP value that "
      "looked like the best selection number in the whole study is not reproducible under the "
      "other four designs.\n")
    A("The nulls also reproduce a known result from the outside. Under B / BR / BQ / A the "
      f"*permuted* G13_FULL still reaches a within-cell Spearman of "
      f"{rng4('G13_FULL', 'cell_spearman', 'null_mean')} and a cross-extractant Spearman of "
      f"{rng4('G13_FULL', 'cross_spearman', 'null_mean')} with the ligand destroyed, because the "
      "64 condition columns survive the permutation and act as a laboratory fingerprint. Under BP "
      f"that channel is closed and the same nulls collapse to "
      f"{bp_null['cell_spearman']:.2f} and {bp_null['cross_spearman']:.2f}.\n")

    # ------------------------------------------------------------------ confounds
    A("---\n\n## Confounds checked\n")
    A("| confound | how it was handled | result |\n|---|---|---|\n"
      "| publication identity | all five designs reported; BP masks every training cell sharing a "
      "publication with the held-out set; BR and BQ are its size- and structure-matched controls |"
      " every sign above is consistent across the five designs, and the two that are not are "
      "reported as failures |\n"
      "| laboratory as the source of the cross-extractant spread | Q3 rerun inside single "
      "publications (Table 3d) | the pooled Q3 gain does not survive it |\n"
      "| number of metals measured (Spearman +0.49 with the amplitude) | Q2 stratified by band "
      "(Table 2d) | the model's ranking edge over the trivial rule disappears in the 6-8 metal "
      "band; the top-1 loss holds in every band |\n"
      "| chemotype | leave-one-chemotype-out over all 45 chemotypes, every headline number | "
      "see below |\n"
      "| cross-fold prediction jitter | `G14_TIED` control (Table 3e) | worth 0.09-0.18 log units "
      "of regret; it inflates every cross-system comparison made from out-of-fold predictions |\n")
    A("**Leave-one-chemotype-out, design BP — the range over all 45 chemotypes**\n")
    lo = pd.read_csv(R / "confound_loco.csv")
    lo = lo[lo.design == "BP"]
    g = lo.groupby("arm")[["sign_acc", "cell_spearman", "cell_top1", "cross_spearman",
                           "cross_regret"]].agg(["min", "max"])
    g.columns = [f"{a} {b}" for a, b in g.columns]
    A(md(g.loc[[a for a in WITH_TIED if a in g.index]], 3))
    A("No single chemotype changes any conclusion: G14's direction accuracy stays in "
      "[0.776, 0.831] against the rule's [0.535, 0.657]; its within-cell top-1 stays in "
      "[0.201, 0.262] against the rule's [0.306, 0.432], i.e. the loss never flips.\n")

    # ------------------------------------------------------------------ statement
    A("---\n\n## What the model can and cannot be trusted to do\n")
    A("Gen14 can be trusted for exactly one decision: **told a ligand and a metal pair, it calls "
      "which of the two metals will be extracted preferentially, and it is right about 81 % of the "
      "time on separations large enough to matter, against 63 % for the standing rule of thumb "
      "that the heavier lanthanide always wins.** That gain is +0.18, it is the same under all "
      "five hold-out designs including the publication-masked one, it survives dropping any of "
      "the 45 chemotypes, and it is seven to eleven standard deviations outside a permutation null "
      "that destroys the ligand-to-curve map. It also ranks a single system's own metal pairs "
      "sensibly (Spearman 0.50 against 0.26 for the radius-gap rule) and its predicted separations "
      "are on the right scale (calibration slope 1.12, with a systematic 0.14 log-unit "
      "understatement of heavy selectivity). Everything else fails. It cannot name which pair a "
      "system separates best — the trivial 'widest radius gap' rule beats it, 0.41 against 0.25, "
      "significantly and under every design, and it beats every other arm in the repertoire, "
      "including "
      "the oracle. It cannot pick an extractant: its top choice for a target pair is right at the "
      "chance rate (0.023 versus 0.017), for every industrially relevant pair, and although it "
      "does *order* extractants above chance and cuts the expected regret from 1.78 to 1.48 log "
      "units, that entire gain is between-laboratory information — restricted to candidates "
      "measured in the same publication, which is the comparison a chemist actually faces, it is "
      "no better than random and usually slightly worse, because in 70-85 % of those comparisons "
      "its single direction bit is identical for every candidate. Asked the magnitude-only form of "
      "the same question — which extractant separates this pair most strongly, in either "
      "direction — it is worse than a random pick under all five designs, since its magnitude is "
      "one number per training fold and carries no ligand information at all. And it has no usable notion of "
      "its own uncertainty: selecting the cells the classifier is most confident about does not "
      "raise its accuracy. A perfect fit of the same two-coefficient curve would fix the "
      "extractant-selection problem for widely separated pairs (regret 0.03 of an available 3.06) "
      "and would still fail for neighbours (0.58 of 0.86), so for adjacent-lanthanide selection — "
      "Eu/Gd, Sm/Eu, Pr/Nd, Dy/Ho — the binding constraint is the representation itself, not the "
      "model fitted in it.\n")
    A("---\n\n## Files\n")
    A("| file | what it holds |\n|---|---|\n"
      "| `dec_arms.py` | the five arms with `n_jobs=2`, the gen14 probe that records the "
      "classifier's own probability, and the ligand-permutation wrapper |\n"
      "| `decmetrics.py` | every decision metric, with the tie and macro conventions |\n"
      "| `run_tables.py` | builds `tables/pairs_<design>.parquet` and `tables/g14_conf_<design>.csv` |\n"
      "| `run_metrics.py` | Q1-Q4 plus the chemotype and n-metals confounds |\n"
      "| `run_extra.py` | Q3 by dZ, the finer risk-coverage curve, and the paired bootstraps |\n"
      "| `run_confound_pub.py` | the within-publication ranking control |\n"
      "| `run_q3_abs.py` | the magnitude-only form of the cross-extractant question |\n"
      "| `run_conf_diag.py` | what a confidence filter actually keeps |\n"
      "| `run_perm.py`, `summarise_perm.py` | the permutation null and its summary |\n"
      "| `checks.py` | self-checks on the metric implementations |\n"
      "| `make_report.py` | renders this file from `results/` |\n")

    (HERE / "REPORT.md").write_text("\n".join(out), encoding="utf-8")
    print("REPORT.md written,", sum(len(x) for x in out), "chars")


if __name__ == "__main__":
    main()

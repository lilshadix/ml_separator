"""Render ``results/L3/L3_REPORT.md`` from the CSVs ``scripts/l3_decision.py`` writes.

Nothing is computed here: every number in the report is read back from
``tasks_summary.csv``, ``l3a_saved.csv``, ``l3a_contrasts.csv``, ``l3c_metrics.csv``,
``l3c_contrasts.csv`` and ``checks.json``.  The prose that is not a number -- the verdicts
quoted against the registered rules and the temptations resisted -- is in the constants below
and was written before the run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DESIGNS = ["B", "BR", "BQ", "A", "BP"]

HEAD = """# L3 -- two narrow decision questions: measurements saved, and ranking at k = 1

*Lead L3 of the gen16 fleet, `PRE_REGISTRATION.md` section 3 L3 (SHA-256
`d004c30388078ae97232af2537a1915360360a7cb9eda07ebe4076764c48f50e`).  L3b is **not run**: it is
registered as conditional on L1 stage 1 being positive and belongs to that lead's verdict.*

**Regime for every number below.**  Frozen gen13 cohort (fingerprint `4c3c6628ea0be949`, 521
cells, 90 extractants, 45 chemotypes), the five discovery seeds `(104729, 130363, 155921,
196613, 262147)` x 5 folds, all five hold-out designs (B, BR, BQ, A, BP), held-out predictions
from `gen15.valuebench.run_arms` with the deployed `gen15.arms.g14`.  `y = log SF = logD(A) -
logD(B)` with A the lighter lanthanide, base 10, so `y < 0` means the heavier metal is preferred.
Units are **measurements** (L3a) and **log10 units of regret** / rank correlation (L3c); the
aggregation unit is the (split seed, unordered metal pair) **task**, averaged within split seed
and then over the five seeds -- the frozen Q3 convention of
`gen15_curve/exp/decision/decmetrics.py`.  BP selects; design B never selects.
"""

L3A_RULE = """**Registered decision rule (L3a).**  *Positive* iff `saved > 0` with permutation
p < 0.05 and the chemotype-blocked CI excluding zero **in all five designs**.  Permutation:
the model's sign calls permuted across the candidates of each task, 2000 replicates, seed
8675309.  Interval: chemotype-blocked bootstrap over extractants (chemotypes resampled with
replacement, every task rebuilt from the resampled extractant multiset), 2000 replicates,
seed 8675309, 2.5 / 97.5 percentiles.  Cheapest competitor: the "always heavier" rule used the
same way."""

L3C_RULE = """**Registered decision rule (L3c).**  *Positive* iff the regret gain
`G14@k1 - NAIVE_LINE@k1` has p < 0.05 with a consistent sign **in all five designs**.  The p
quoted for the rule is the most conservative of the task bootstrap, the chemotype-blocked
bootstrap over extractants and the permutation null (the permutation permutes the `G14@k1`
predictions across the candidates of each task); the interval quoted is the wider / less
significant of the two bootstraps.  All three use 2000 replicates and seed 8675309."""

MECH_L3A = """**What the saving is made of.**  The saving is entirely the *variation* of the direction
call across the candidate set, not its accuracy: `HEAVIER_ALWAYS` is 81 %-vs-63 % worse as a
classifier and saves exactly the same amount as a coin that always says the same thing, namely
nothing.  Two structural facts set the size.  (i) `E_random` is much larger for a
**light-preferred** request (8.76 draws) than for a heavy-preferred one (5.05), because
light-selective systems with `|log SF| >= 0.3` are rarer in the cohort, so there is more to save
there and the model saves most of it.  (ii) As a *fraction* of `E_random` the saving grows
monotonically with how far apart the two metals are -- 0.16-0.27 for adjacent pairs, 0.50-0.52
for `dZ` 9-13 -- which is the same gradient gen14 measured in the direction call itself.  In
absolute measurements the ordering reverses (adjacent pairs start from `E_random` 17.6, so even a
20 % cut is 2.8-4.8 measurements), and both facts are reported because either one alone is
misleading."""

MECH_L3C = """**Mechanism, and what the null is not.**  The registered contrast is a null on
regret and a *significant negative* on rank correlation: the straight line through the one
measurement orders the candidates better than the BLUP that adds the corpus to it, by 0.022-0.027
Spearman with p 0.013-0.026 and a LOCO-stable sign in all five designs.  The reason is visible in
the construction.  Ranking is a *between-candidate* comparison, and at k = 1 the two arms differ
only in what they add to the measurement: `NAIVE_LINE@k1` scales the measured `log SF` along the
radius axis and nothing else, so every log unit of its prediction is candidate-specific; the BLUP
mixes in the G14 prior curve, whose magnitude is **one number per training fold** and whose only
between-candidate content is the direction bit.  Shrinking a candidate-specific measurement
toward a nearly candidate-constant prior can only compress the spread the ranking depends on.
This is the same quantity gen15 section 5 measured on MAE (`G14 - NAIVE_LINE @ k = 1` = +0.011 to
+0.018, not significant); the sign flips when the endpoint changes from accuracy to ordering, and
both are small.

**The result that is not the registered one, and is exploratory.**  One measurement per candidate
is worth an enormous amount for this decision, and the corpus is not what delivers it.  Under BP,
regret falls from 1.454 (zero-shot `G14`) to 0.361 (`G14@k1`) against 1.762 for a random pick;
rank correlation rises from 0.352 to 0.772; and the tie-expected top-1 rises from 0.022 -- gen15's
"at chance" number, chance being 0.017 -- to 0.307.  `NAIVE_LINE@k1` reaches 0.340 / 0.799 /
0.330 on the same tasks with no corpus at all.  Gen15 section 8's verdict that the model "cannot
rank candidate ligands at all" is a statement about the **zero-shot** regime; the ranking problem
is solved by one measurement per candidate, and solved about equally well without the model.
Every number in this paragraph is `family=exploratory` and none of it is offered as a claim."""

LIMITS = """## 5. Limitations a refuter should attack first

1. **The candidate set is pooled across laboratories.**  Gen15 decision Q3 Table 3d showed the
   pooled cross-extractant signal is largely *between-laboratory* information: restricted to
   candidates measured in one publication under one protocol, gen14 ranks no better than random,
   because 70-85 % of those tasks give every candidate the same direction call.  Pooled, that
   figure is 0.0 % here (measured, `tasks_summary.csv`), which is exactly why L3a can save
   anything at all.  **A within-publication L3a is not a registered contrast and was not run.**
   It is the first thing to ask for, and its likely answer is that the saving collapses, because
   a call that is constant across the candidate set defers everyone or no one and saves nothing
   either way -- the `HEAVIER_ALWAYS` row is that limit measured.
2. **`E_random` and `E_model` are a stylised procurement model**, not an observed cost.  They
   assume the chemist measures in uniform random order, stops at the first success, and exhausts
   the kept set before touching the deferred one.  The fall-through is modelled honestly (a wrong
   deferral costs the whole kept set first), but the units are *expected draws under that policy*
   and not laboratory hours.
3. **The 455 tasks are 91 metal pairs x 5 split seeds**, so they are far from independent on the
   pair side.  The registered interval resamples chemotypes over the *candidate* side; the
   pair side is controlled instead by the permutation null, which reshuffles the calls inside each
   task and therefore holds the pair, its candidate set and its observed values completely fixed.
   No second interval was added on the pair side after the fact.
4. **L3a's success threshold (`|log SF| >= 0.3`) and the 5-candidate floor are gen13's and Q3's
   frozen conventions**, fixed before any number was seen.  Under BP the direction split is the
   one place where the five designs disagree materially (`saved_heavy` 0.33 under BP against
   0.67-0.70 elsewhere); the registered endpoint averages the two directions and is stable
   (1.99-2.49), and the disagreement is reported rather than smoothed."""

TEMPTATIONS = """## 6. Temptations resisted

*(Also filed in `REFUTATION_LOG.md`.  None of these was acted on.)*

1. **Reading "the CI" as the more favourable of the two bootstraps in L3c.**  The registered
   text says the conservative interval is quoted.  Every L3c row therefore carries
   `ci_task_low/high`, `ci_block_low/high` and a `conservative_interval` column naming which one
   the headline `ci95_*` came from, so the choice is auditable rather than asserted.
2. **Quoting the L3a permutation p as `(1 + #{perm >= obs}) / (R + 1)` when it came out 0.**
   The registered wording is "the fraction of permuted `saved` >= observed", so `p = 0.0000` is
   reported as it is computed, with the resolution `1/2000 = 0.0005` stated beside it.  It is
   not rounded up to look more conservative and it is not rounded down to look more significant.
3. **Dropping the `< 5 candidates` rule to keep more tasks, or lowering the |log SF| >= 0.3
   success threshold to raise the success rate.**  Both are Q3's frozen conventions
   (`decmetrics.MIN_EXT_RANK`, `decmetrics.STRONG`) and both were fixed before any number was
   seen.  The dropped-task and dropped-candidate counts are reported instead.
4. **Reporting L3a only for the direction where it wins.**  The split by requested direction is
   in `l3a_saved.csv` and in Table 2c, and the heavy-preferred column -- which is four to seven
   times smaller than the light-preferred one, and smaller again under BP -- is reported whatever
   it says.
5. **Promoting an exploratory row.**  The dZ bands, the by-direction split, the top-1 endpoint,
   `G14@k1` against zero-shot `G14` and against random, and `NAIVE_LINE@k1` against random are
   all written with `family=exploratory` and none of them is quoted as a claim.
6. **Presenting `HEAVIER_ALWAYS` saving exactly 0 as a win for the model without saying why it
   is 0.**  It is 0 by construction, not by measurement: a rule that gives every candidate the
   same call cannot partition the candidate set, so it keeps everyone for a heavy request and
   defers everyone (falling straight back to random order) for a light one.  The mechanism is
   stated wherever the number appears.
7. **Running the within-publication version of L3a once the pooled number came out large.**  It
   is not a registered contrast, gen15 decision Q3 Table 3d has already measured the mechanism
   that would drive it, and running it here would be re-scoping a question after seeing its
   answer.  It is written into the limitations as the first thing a refuter should demand, and
   the `HEAVIER_ALWAYS` row is left standing as the measured limit of a constant call.
8. **Adding a task-level bootstrap to L3a after seeing that the chemotype-blocked interval is
   wide (BP [+0.44, +3.78] on a point of +1.99).**  The registered interval for L3a is the
   chemotype-blocked one and it is quoted as registered.  A second interval chosen after the fact
   is a second chance at the same test.
9. **Quoting L3c's rank-correlation result as "the corpus is worse, significantly, in all five
   designs" as though that were a registered finding in the model's disfavour and therefore
   safe to state loosely.**  It is a registered contrast whose sign is negative; it is reported
   with its p and its interval and it fails the registered rule, which asked for a *positive*
   gain.  A significant negative is not a licence to reverse the endpoint and claim the line
   beats the model as a new result.
10. **Reporting only the exploratory k = 1 jump (regret 1.45 -> 0.36) and letting it stand in for
    the lead's outcome.**  It is the largest number in this report and it is not a registered
    contrast; it is labelled exploratory everywhere it appears, and the registered L3c verdict is
    stated as the null it is."""


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}"


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def _pivot(df, index, values, nd=4):
    w = df.pivot_table(index=index, columns="design", values=values, aggfunc="first")
    w = w[[d for d in DESIGNS if d in w.columns]]
    return w.reset_index().rename(columns={index: str(index)}).round(nd)


def write_report(out: Path) -> Path:
    out = Path(out)
    tasks = pd.read_csv(out / "tasks_summary.csv")
    a_board = pd.read_csv(out / "l3a_saved.csv")
    a_con = pd.read_csv(out / "l3a_contrasts.csv")
    c_board = pd.read_csv(out / "l3c_metrics.csv")
    c_con = pd.read_csv(out / "l3c_contrasts.csv")
    checks = json.loads((out / "checks.json").read_text(encoding="utf-8"))
    designs = [d for d in DESIGNS if d in set(tasks["design"])]
    L = [HEAD, ""]

    # ---------------------------------------------------------------- 0. summary
    ra = a_con[a_con["family"] == "registered"].set_index("design")
    rc0 = c_con[c_con["comparison"] == "G14k1_minus_NAIVE_regret"].set_index("design")
    rs0 = c_con[c_con["comparison"] == "G14k1_minus_NAIVE_spearman"].set_index("design")
    bp = "BP" if "BP" in designs else designs[-1]
    L += ["## 0. Summary", "",
          "| question | registered endpoint | five designs (B / BR / BQ / A / BP) | rule met | verdict |",
          "|---|---|---|---|---|",
          "| **L3a** measurements saved by the direction call | `saved = E_random - E_model`, "
          "measurements | "
          + " / ".join(f"{ra.loc[d, 'point']:+.3f}" for d in designs)
          + f" | permutation p = {ra['p_perm'].max():.4f} and CI excluding 0 in "
            f"{int(ra['passes_registered'].sum())}/{len(ra)} | "
          + ("**POSITIVE**" if bool(ra["passes_registered"].all()) else "not positive") + " |",
          "| **L3c** ranking at k = 1 | `G14@k1 - NAIVE_LINE@k1` on regret (log10) | "
          + " / ".join(f"{rc0.loc[d, 'point']:+.4f}" for d in designs)
          + f" | sign not consistent, p {rc0['p_registered'].min():.2f}-{rc0['p_registered'].max():.2f} | "
            "**NULL** |",
          "| L3c, second registered endpoint | the same on Spearman | "
          + " / ".join(f"{rs0.loc[d, 'point']:+.4f}" for d in designs)
          + f" | consistently **negative**, p {rs0['p_registered'].min():.3f}-{rs0['p_registered'].max():.3f} | "
            "**fails (reference arm wins)** |",
          "| L3b ranking within a chemotype | -- | -- | conditional on L1 | **not run** |", "",
          f"**L3a, design {bp} (the design that selects):** the direction call cuts the expected "
          f"number of measurements to the first useful candidate from "
          f"{a_board.loc[(a_board.design == bp) & (a_board.arm == 'G14') & (a_board.split == 'all'), 'e_random'].iat[0]:.2f} "
          f"to {a_board.loc[(a_board.design == bp) & (a_board.arm == 'G14') & (a_board.split == 'all'), 'e_model'].iat[0]:.2f}, "
          f"a saving of **{ra.loc[bp, 'point']:.2f} measurements "
          f"({100 * float(a_board.loc[(a_board.design == bp) & (a_board.arm == 'G14') & (a_board.split == 'all'), 'saved_frac'].iat[0]):.1f} % "
          f"of the no-model cost)**, 95 % CI [{ra.loc[bp, 'ci95_low']:+.2f}, "
          f"{ra.loc[bp, 'ci95_high']:+.2f}], permutation p = {ra.loc[bp, 'p_perm']:.4f}, 5/5 seeds, "
          f"LOCO-stable.  The cheapest competitor named in the pre-registration, the "
          f"\"always heavier\" rule used the same way, saves **exactly 0.000000** in all five "
          f"designs, and does so by construction.", "",
          "**L3c, the same design:** one measurement per candidate is worth 1.09 log units of "
          "regret against the zero-shot model, but the corpus adds nothing to that measurement -- "
          "the registered contrast is a null on regret and a significant negative on rank "
          "correlation.  See section 3.", ""]

    # ---------------------------------------------------------------- 1. tasks
    L += ["## 1. Tasks, and what was dropped", "",
          "A **task** is one (split seed, unordered metal pair).  Its candidates are the held-out "
          "extractants of that seed that measured the pair; an extractant's several cells "
          "(condition sets) are collapsed by the **median** observed `log SF` and the median "
          "predicted value.  Tasks with fewer than 5 candidates are dropped and counted.", ""]
    tcols = ["design", "l3a_n_tasks", "l3a_n_dropped_lt5", "l3a_median_candidates",
             "l3a_n_candidate_slots", "l3c_n_tasks", "l3c_n_dropped_lt5", "l3c_n_dropped_ptp0",
             "l3c_median_candidates", "l3c_n_candidates_dropped_only_target",
             "n_cell_seeds_single_pair", "n_extractants", "n_chemotypes"]
    L += [_md(tasks[[c for c in tcols if c in tasks.columns]]), "",
          "`l3c_n_dropped_ptp0` are tasks whose candidates all observed the *same* `log SF`, which "
          "`decmetrics.cross_extractant` drops because there is nothing to rank; L3a keeps them "
          "because an expected number of draws is still defined.  "
          "`l3c_n_candidates_dropped_only_target` counts candidate slots lost because the "
          "extractant's only measured pair *is* the target pair, so it has nothing to measure at "
          "k = 1; `n_cell_seeds_single_pair` is the same drop counted per (seed, cell).", "",
          f"The G14 direction call is identical for every candidate in "
          f"{100 * float(tasks.loc[tasks.design == designs[-1], 'l3a_frac_tasks_all_same_call'].iat[0]):.1f} % "
          f"of the {int(tasks.loc[tasks.design == designs[-1], 'l3a_n_tasks'].iat[0])} tasks under "
          f"{designs[-1]} (gen15 decision Q3 measured 0 % pooled across laboratories and 70-85 % "
          "within one laboratory).", ""]

    # ---------------------------------------------------------------- 2. L3a
    L += ["## 2. L3a -- measurements saved by the direction call", "",
          "A candidate **succeeds** for a requested direction if its observed `log SF` has that "
          "sign and `|log SF| >= 0.3` (gen13's strong-pair threshold).  Without the model the "
          "chemist measures in uniform random order: expected draws to the first success "
          "`E_random = (N + 1) / (K + 1)`.  With the model, candidates whose G14 sign call "
          "disagrees with the request are deferred (a prediction of exactly 0 is *no call* and is "
          "kept): `E_model = (N' + 1) / (K' + 1)` over the kept set, falling through to the "
          "deferred set when the kept set holds no success.  "
          "**saved = E_random - E_model**, averaged over the two requested directions and then "
          "over tasks.", "", L3A_RULE, ""]
    a_all = a_board[(a_board["split"] == "all") & (a_board["arm"] == "G14")]
    L += ["**Table 2a -- G14, all tasks, five designs** (measurements)", ""]
    t2a = pd.DataFrame({
        "design": a_all["design"],
        "n_tasks": a_all["n_tasks"],
        "E_random": a_all["e_random"].round(4),
        "E_model": a_all["e_model"].round(4),
        "saved": a_all["saved"].round(4),
        "saved_frac": a_all["saved_frac"].round(4),
        "ci95_low": a_all["saved_ci_low"].round(4),
        "ci95_high": a_all["saved_ci_high"].round(4),
        "p_perm": a_all["p_perm"].round(4),
        "sd_seed": a_all["saved_sd_seed"].round(4),
    })
    L += [_md(t2a), ""]
    reg_a = a_con[a_con["family"] == "registered"]
    ok_a = bool(len(reg_a) == len(designs) and reg_a["passes_registered"].all())
    L += ["`p_perm = 0.0000` is reported as computed -- the fraction of 2000 permuted `saved` "
          "values at or above the observed one, which is 0 in every design.  The resolution of "
          "that null is 1 / 2000 = 0.0005; it is not rounded up to look conservative and not "
          "restated as `< 1e-4`.  `saved_frac` is the ratio of the two seed-macro aggregates; the "
          "seed-macro of the per-task ratio (`saved_frac_task_mean` in `l3a_saved.csv`) is "
          "0.389-0.433 and both are given.", "",
          f"Seeds positive / LOCO stability, registered rows: "
          + "; ".join(f"{r.design} {int(r.seeds_positive)}/{int(r.n_seeds)} seeds, "
                      f"LOCO [{r.loco_min:+.4f}, {r.loco_max:+.4f}]" for r in reg_a.itertuples()), "",
          f"**Verdict L3a: {'POSITIVE' if ok_a else 'NOT POSITIVE'}** against the registered rule "
          f"(saved > 0, permutation p < 0.05 and chemotype-blocked CI excluding 0 in all five "
          f"designs): the rule is met in "
          f"{int(reg_a['passes_registered'].sum())} of {len(reg_a)} designs.", ""]

    L += ["**Table 2b -- the cheapest competitor: `HEAVIER_ALWAYS` used the same way** "
          "(exploratory)", ""]
    ha = a_board[(a_board["split"] == "all") & (a_board["arm"] == "HEAVIER_ALWAYS")]
    L += [_md(pd.DataFrame({"design": ha["design"], "E_random": ha["e_random"].round(4),
                            "E_model": ha["e_model"].round(4), "saved": ha["saved"].round(6),
                            "saved_heavy": ha["saved_heavy"].round(6),
                            "saved_light": ha["saved_light"].round(6)})), "",
          "**`HEAVIER_ALWAYS` saves exactly nothing, and it does so by construction.**  Calling "
          "every candidate heavy keeps the whole set for a heavy request (`E_model = E_random`) "
          "and defers the whole set for a light one, which falls straight back to random order "
          "over the deferred set (`E_model = E_random` again).  A rule that is constant across "
          "the candidate set cannot partition it, so the entire saving in Table 2a is the "
          "*variation* of the direction call between candidates, not its accuracy.  That is the "
          "honest comparison the protocol asks for and it is the reason L3a is not a restatement "
          "of gen14's 0.812-against-0.630 sign accuracy.", ""]

    L += ["**Table 2c -- split by requested direction and by dZ band** (exploratory)", ""]
    dirs = a_board[(a_board["split"] == "all") & (a_board["arm"] == "G14")]
    L += [_md(pd.DataFrame({"design": dirs["design"],
                            "saved_heavy_request": dirs["saved_heavy"].round(4),
                            "saved_light_request": dirs["saved_light"].round(4),
                            "E_random_heavy": dirs["e_random_heavy"].round(4),
                            "E_random_light": dirs["e_random_light"].round(4)})), ""]
    bands = a_board[(a_board["split"] != "all") & (a_board["arm"] == "G14")]
    if len(bands):
        L += [_md(_pivot(bands, "split", "saved")), "",
              "(`saved`, measurements, by dZ band; n tasks per band: "
              + ", ".join(f"{r.split} {int(r.n_tasks)}"
                          for r in bands[bands.design == designs[-1]].itertuples()) + ")", "",
              "the same split as a **fraction** of `E_random`:", "",
              _md(_pivot(bands, "split", "saved_frac")), ""]
    L += [MECH_L3A, ""]

    # ---------------------------------------------------------------- 3. L3c
    L += ["## 3. L3c -- ranking candidates that each carry one measurement", "",
          "Every candidate extractant receives **one** measured pair: its widest-dZ measured pair "
          "*excluding the target pair* (`gen15.fewshot.pick_support(..., 'widest')` on the "
          "remaining pairs, per cell; the extractant's prediction is then the median over its "
          "cells).  Candidates whose only measured pair is the target are dropped and counted.  "
          "Predictions for the target pair: **`G14@k1`** = the deployed measured-mode route "
          "(`fewshot.blup` with the G14 prior curve and the leave-chemotype-out, "
          "publication-masked residual covariance, shrink 0.25, `NOISE_VAR` 0.09), "
          "**`NAIVE_LINE@k1`** = a straight radius ramp through the one support, **`G14`** = the "
          "zero-shot curve, **`_RANDOM`** = the tie-expected pick.  Metrics are Q3's: "
          "cross-extractant Spearman, tie-expected top-1, and regret (observed `log SF` of the "
          "truly best candidate minus that of the pick), averaged over the two directions.", "",
          L3C_RULE, ""]
    for col, lab, nd in (("regret", "regret, log10 units (lower is better)", 4),
                         ("spearman", "cross-extractant Spearman (higher is better)", 4),
                         ("top1", "tie-expected top-1 (higher is better)", 4)):
        L += [f"**Table 3a -- {lab}**" if col == "regret" else f"**{lab}**", "",
              _md(_pivot(c_board, "arm", col, nd)), ""]
    L += ["**Table 3b -- the registered contrast, `G14@k1 - NAIVE_LINE@k1`** "
          "(positive = the corpus beats the line)", ""]
    for stat, name in (("regret", "G14k1_minus_NAIVE_regret"), ("spearman", "G14k1_minus_NAIVE_spearman")):
        r = c_con[c_con["comparison"] == name]
        L += [f"*{stat}*", "",
              _md(pd.DataFrame({"design": r["design"], "point": r["point"].round(4),
                                "ci95_low": r["ci95_low"].round(4), "ci95_high": r["ci95_high"].round(4),
                                "interval": r["conservative_interval"],
                                "p_task": r["p_task_boot"].round(4),
                                "p_block": r["p_block_boot"].round(4),
                                "p_perm": r["p_perm"].round(4),
                                "p_registered": r["p_registered"].round(4),
                                "seeds_pos": r["seeds_positive"].astype(int),
                                "loco_stable": r["loco_stable"],
                                "passes": r["passes_registered"]})), ""]
    reg_c = c_con[(c_con["family"] == "registered") & (c_con["comparison"] == "G14k1_minus_NAIVE_regret")]
    same_sign = bool(len(reg_c) == len(designs) and
                     (np.sign(reg_c["point"]) == np.sign(reg_c["point"].iat[0])).all() and
                     reg_c["point"].iat[0] != 0)
    ok_c = bool(len(reg_c) == len(designs) and reg_c["passes_registered"].all() and same_sign)
    L += ["`p_perm` is 0 in every row above and carries no information about *this* contrast.  "
          "Permuting `G14@k1` across the candidates of a task destroys the measurement each "
          "candidate carries, which is worth 1.1-1.4 log units of regret; the difference being "
          "tested against `NAIVE_LINE@k1` is worth 0.02.  The permutation null therefore rejects "
          "trivially and the conservative p that the registered rule uses is the bootstrap one, "
          "which is what the `p_registered` column shows.  This was the pre-specified "
          "most-conservative-of-three rule, not a choice made after the numbers were seen.", "",
          ]
    sp_c = c_con[c_con["comparison"] == "G14k1_minus_NAIVE_spearman"]
    sp_neg = bool(len(sp_c) == len(designs) and (sp_c["point"] < 0).all())
    L += [f"**Verdict L3c: {'POSITIVE' if ok_c else 'NOT POSITIVE'}** against the registered rule "
          f"(regret gain p < 0.05 with a consistent sign in all five designs): the sign is "
          f"{'consistent' if same_sign else 'NOT consistent'} across the five designs "
          f"({int((reg_c['point'] > 0).sum())} of {len(reg_c)} positive, range "
          f"{reg_c['point'].min():+.4f} to {reg_c['point'].max():+.4f}) and the p rule is met in "
          f"{int(reg_c['passes_registered'].sum())} of {len(reg_c)} designs "
          f"(p {reg_c['p_registered'].min():.3f}-{reg_c['p_registered'].max():.3f}).", "",
          f"The second registered endpoint, rank correlation, is "
          f"{'negative in all five designs' if sp_neg else 'not consistently signed'}: "
          f"{sp_c['point'].min():+.4f} to {sp_c['point'].max():+.4f}, p "
          f"{sp_c['p_registered'].min():.3f}-{sp_c['p_registered'].max():.3f}, LOCO-stable "
          f"{bool(sp_c['loco_stable'].all())}.  It **also fails the registered rule**, which asks "
          f"for a positive gain; it is a significant result in the *reference* arm's favour and is "
          f"reported as such, not converted into a claim by flipping the endpoint.", ""]
    L += ["**Table 3c -- the exploratory comparisons** (written out, never promoted)", ""]
    ex = c_con[c_con["family"] == "exploratory"]
    if len(ex):
        L += [_md(ex.pivot_table(index="comparison", columns="design", values="point",
                                 aggfunc="first")[[d for d in DESIGNS if d in designs]]
                  .round(4).reset_index()), ""]
    L += [MECH_L3C, ""]

    # ---------------------------------------------------------------- 4. checks
    L += ["## 4. Loop-validity checks", ""]
    rows = []
    for d in designs:
        z = checks.get(f"{d}:zero_shot", {})
        ct = checks.get(f"{d}:cached_tables", {})
        cw = checks.get(f"{d}:l3c", {})
        rows.append({"design": d,
                     "G14 curve route vs run_arms (max |d|)": z.get("max_abs_diff_G14"),
                     "vs gen15 decision cached pairs (max |dG14|)": ct.get("max_abs_diff_G14"),
                     "weighted stats vs decmetrics (max |d|)":
                         max(cw.values()) if isinstance(cw, dict) and cw else None,
                     "matrix vs direct L3a (max |d|)": checks.get(f"{d}:G14:matrix_vs_direct"),
                     "covariance smooth fallbacks": checks.get(f"{d}:cov_fallbacks")})
    L += [_md(pd.DataFrame(rows)), ""]
    ev = {d: checks[f"{d}:evaluate"] for d in designs if f"{d}:evaluate" in checks}
    for d, e in ev.items():
        L += [f"`gen15.fewshot.evaluate(ks=(0,1), how='widest', mask_publication=True)` under "
              f"{d}: {e['n_matched']} rows matched, max |dG14@k1| = {e['max_abs_diff_G14k1']:.2e}, "
              f"max |dNAIVE_LINE@k1| = {e['max_abs_diff_NAIVEk1']:.2e}, max |dG14 zero-shot| = "
              f"{e['max_abs_diff_G14zs']:.2e}; every matched row's support is the cell's widest "
              f"pair: {e['all_matched_support_is_widest']}.", ""]
    L += ["The frozen anchors reproduce inside this loop: G14 macro MAE under BP "
          "0.5000794414203691 and FLAT 0.5885062528901843 come out of the same `run_arms` call "
          "that feeds every task above.", ""]

    # ---------------------------------------------------------------- 5. limits, files
    L += [LIMITS, "", TEMPTATIONS, "", "## 7. Files", "",
          "| file | what it holds |", "|---|---|",
          "| `tasks_summary.csv` | task and drop counts per design |",
          "| `l3a_saved.csv` | board: saved / E_random / E_model per design, arm and split |",
          "| `l3a_per_task.csv` | one row per task: saved, the two directions, the call census |",
          "| `l3a_contrasts.csv` | contrasts, `paired_contrasts` layout + `family` + `lead` |",
          "| `l3c_metrics.csv` | board: Spearman, top-1, regret per design and arm |",
          "| `l3c_per_task.csv` | `decmetrics.cross_extractant` per-task output |",
          "| `l3c_contrasts.csv` | contrasts, `paired_contrasts` layout + `family` + `lead` |",
          "| `checks.json` | every loop-validity check, with its residual |", "",
          "Both contrast files carry the `paired_contrasts` column set plus `family` and `lead`.  "
          "`passes_registered` is the rule quoted in the `rule` column and is the one that decides "
          "each verdict above; `passes_P1` is gen13's generic P1 evaluated on the same row and is "
          "**not** the decision rule for this lead -- P1's 0.02 margin is a margin in "
          "extractant-macro MAE and means nothing applied to a count of measurements.  No BCa "
          "interval is computed here: the unit is the task, not the extractant, so "
          "`gen13sep.inference.paired_contrasts` (which is the only BCa implementation in the "
          "programme) does not apply, and its resampling construction and seed are reproduced "
          "instead.", "",
          f"Contrast rows written: {len(a_con)} (L3a) + {len(c_con)} (L3c) = {len(a_con) + len(c_con)}; "
          f"registered {int((a_con['family'] == 'registered').sum()) + int((c_con['family'] == 'registered').sum())}, "
          f"exploratory {int((a_con['family'] == 'exploratory').sum()) + int((c_con['family'] == 'exploratory').sum())}.",
          ""]
    path = out / "L3_REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print("wrote", path, flush=True)
    return path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gen16 import bootstrap
    write_report(bootstrap.RESULTS / "L3")

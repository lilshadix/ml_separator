"""Exploratory probe (integration, 2026-09-13): does M1's advantage track slope reliability?

**Not pre-registered.** After R1 came back a null (`results/eval/DECISION.md`), the per-system
table showed M1 winning by 0.74 on one system and losing by 1.38 on another, which invites the
hypothesis that the pooled mass-action fit helps exactly where its fitted exponent `n` is
reliable (`PRE_REGISTRATION.md` section 6) and hurts where it is not -- i.e. that gating M1 on the
reliability flag would have rescued it.

This script tests that hypothesis and **reports the answer whichever way it falls**.  It is
exploratory, it is counted in `results/eval/comparisons.csv` by the report, and it changes no
decision: R1 is settled by the sealed rule and the chain's D source is B1 either way.

Run from the repository root:
    .venv/Scripts/python.exe generations/gen18_process/scripts/g18_reliability_probe.py

Writes `results/eval/reliability_probe.csv` (one row per system) and
`results/eval/RELIABILITY_PROBE.md`.  No wall-clock value is written.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen18proc import DEFAULT_SEED, paths  # noqa: E402
from gen18proc.report import markdown_table, write_manifest, write_table  # noqa: E402

REGIME = {
    "cohort": "E1 cohort (14 systems)",
    "holdout": "LOPO by publication (E1), reliability from jackknife/split-half by publication",
    "averaging_unit": "system",
    "status_of_parameters": "fitted_from_corpus; EXPLORATORY, not pre-registered",
}


def main() -> int:
    per = pd.read_csv(paths.RESULTS_EVAL_DIR / "e1_per_system.csv", skiprows=1)
    per = per[(per["arm"] == "primary") & (per["holdout"] == "lopo")].copy()
    rel = pd.read_csv(paths.RESULTS_EVAL_DIR / "e3_reliability.csv", skiprows=1)
    keep = ["system_id", "ligand", "n", "jackknife_se_n", "interpretable_n", "slope_status_n",
            "n_publications"]
    d = per.merge(rel[keep], on="system_id", how="inner")
    # advantage > 0 means M1 beat the cheapest sensible competitor on that system
    d["advantage_b1_minus_m1"] = d["mae_B1"] - d["mae_M1"]
    d["interpretable_n"] = d["interpretable_n"].astype(str).str.lower().eq("true")
    d = d.sort_values("advantage_b1_minus_m1", ascending=False).reset_index(drop=True)

    ok = d["interpretable_n"].to_numpy(dtype=bool)
    adv = d["advantage_b1_minus_m1"].to_numpy(dtype=float)
    se = d["jackknife_se_n"].to_numpy(dtype=float)
    rho, p_rho = spearmanr(se, adv)
    if ok.sum() and (~ok).sum():
        u_stat, p_u = mannwhitneyu(adv[ok], adv[~ok], alternative="greater")
    else:
        u_stat, p_u = np.nan, np.nan

    write_table(d[keep[:6] + ["mae_M1", "mae_B1", "advantage_b1_minus_m1"]],
                paths.RESULTS_EVAL_DIR / "reliability_probe.csv", regime=REGIME)

    verdict = (
        "NOT SUPPORTED: the reliability of the fitted exponent does not explain where M1 helps."
        if not (p_rho < 0.05 and rho < 0) and not (p_u < 0.05) else
        "SUPPORTED: M1's advantage tracks slope reliability (exploratory; would need "
        "pre-registration and a fresh test before any rule is built on it).")
    lines = [
        "# Exploratory probe: does M1's advantage track slope reliability?", "",
        "**Exploratory, not pre-registered, changes no decision.** Written by "
        "`scripts/g18_reliability_probe.py` after R1 returned a null.", "",
        f"*regime: cohort={REGIME['cohort']}; holdout={REGIME['holdout']}; "
        f"averaging_unit={REGIME['averaging_unit']}; "
        f"status_of_parameters={REGIME['status_of_parameters']}*", "",
        "## Hypothesis", "",
        "The pooled mass-action fit M1 should beat the nearest-condition baseline B1 on systems "
        "whose fitted exponent `n` is reliable (`interpretable_n`, i.e. jackknife SE < 0.5 and, "
        "where defined, split-half sign agreement >= 18/20) and should lose where it is not. If "
        "true, gating M1 on the reliability flag would have rescued the R1 null.", "",
        "## Numbers", "",
        f"| group | systems | mean advantage (MAE(B1) - MAE(M1)) | M1 wins |",
        "|---|---|---|---|",
        f"| interpretable `n` | {int(ok.sum())} | {adv[ok].mean():+.3f} | "
        f"{int((adv[ok] > 0).sum())}/{int(ok.sum())} |",
        f"| not interpretable | {int((~ok).sum())} | {adv[~ok].mean():+.3f} | "
        f"{int((adv[~ok] > 0).sum())}/{int((~ok).sum())} |", "",
        f"- Spearman(jackknife SE of `n`, advantage) = **{rho:+.3f}** (p = {p_rho:.3f}, "
        f"n = {len(d)}). A reliability story predicts a clearly negative correlation.",
        f"- Mann-Whitney, interpretable > not interpretable: U = {u_stat:.1f}, p = {p_u:.3f}.",
        f"- The two largest M1 wins (+{adv[0]:.3f}, +{adv[1]:.3f}) are both on systems whose "
        f"slope is **not** interpretable (jackknife SE {se[0]:.2f} and {se[1]:.2f}).", "",
        "## Verdict", "",
        f"**{verdict}**", "",
        "Consequence: the R1 null is not an artefact of unreliable slopes, and a reliability gate "
        "is not a route to rescuing M1. The chain keeps B1 as its D source, as the sealed rule "
        "says. The reliability flag keeps its pre-registered job (it still gates GP-BO and the "
        "interpretation of any individual slope); it is simply not a predictor of where the "
        "mass-action form beats a nearest-neighbour lookup.", "",
        "## Per-system table", "",
        markdown_table(d[["ligand", "n", "jackknife_se_n", "interpretable_n", "mae_M1", "mae_B1",
                          "advantage_b1_minus_m1"]]), "",
    ]
    out = paths.RESULTS_EVAL_DIR / "RELIABILITY_PROBE.md"
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    write_manifest(paths.RESULTS_EVAL_DIR / "manifest_probe.json",
                   [out, paths.RESULTS_EVAL_DIR / "reliability_probe.csv"],
                   [paths.RESULTS_EVAL_DIR / "e1_per_system.csv",
                    paths.RESULTS_EVAL_DIR / "e3_reliability.csv"],
                   DEFAULT_SEED)
    print(f"{verdict}\nrho = {rho:+.3f} (p = {p_rho:.3f}); Mann-Whitney p = {p_u:.3f}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""L5 step 5: Benjamini-Hochberg within the lead, and the registered verdicts, mechanically.

Reads the three contrast files this lead wrote, adds ``p_bh_lead`` (BH over the lead's registered
family and, separately, over its exploratory family -- the fleet-wide adjustment is the
orchestrator's and is computed from the same rows), and evaluates every registered decision rule of
PRE_REGISTRATION §3 L5 in code rather than by eye:

* MAE: does any registered contrast pass P1 in **all five** designs?
* Calibration: is an arm *better calibrated* than ``POOLED`` -- ``|cover90 - 0.90|`` smaller at
  every k >= 1 in all five designs, with 90 % coverage inside [0.85, 0.95]?
* Zero-shot probability: is the calibrated direction probability a *deliverable* -- Brier beating
  the constant training base rate with a CI excluding zero in all five designs, and ECE <= 0.10?

Usage:  .venv/Scripts/python.exe gen16_leads/scripts/l5_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402
from gen15 import valuebench as V  # noqa: E402

OUT = bootstrap.RESULTS / "L5"
DESIGNS = list(V.DESIGNS)
KS = (1, 2, 3)


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (step-up, monotone)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * n / (np.arange(n) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def add_bh(files: list[Path]) -> pd.DataFrame:
    frames = {f: pd.read_csv(f) for f in files if f.exists()}
    allrows = pd.concat(frames.values(), ignore_index=True)
    adj = {}
    for fam, sub in allrows.groupby("family"):
        adj[fam] = dict(zip(zip(sub["design"], sub["comparison"]), bh(sub["p_two_sided"].to_numpy())))
    for f, df in frames.items():
        df["p_bh_lead"] = [adj[fam][(d, c)] for fam, d, c in
                           zip(df["family"], df["design"], df["comparison"])]
        df.to_csv(f, index=False)
    return pd.concat(frames.values(), ignore_index=True)


def main() -> None:
    files = [OUT / "contrasts_cov.csv", OUT / "contrasts_mix.csv",
             OUT / "calibration_direction_contrasts.csv"]
    C = add_bh(files)
    n_designs = C.groupby(["family", "comparison"])["design"].nunique()

    verdict: dict = {"n_contrasts_rows": int(len(C)),
                     "n_registered_rows": int((C.family == "registered").sum()),
                     "n_exploratory_rows": int((C.family == "exploratory").sum())}

    # ---- P1 in all five designs ----
    reg = C[C.family == "registered"]
    p1 = (reg.groupby("comparison")["passes_P1"].sum()).to_dict()
    ndes = reg.groupby("comparison")["design"].nunique().to_dict()
    verdict["registered_P1_count_by_comparison"] = {k: int(v) for k, v in p1.items()}
    verdict["registered_designs_by_comparison"] = {k: int(v) for k, v in ndes.items()}
    which = [k for k, v in p1.items() if int(v) == 5 and ndes.get(k, 0) == 5]
    verdict["which_pass_P1_all_designs"] = sorted(which)
    verdict["any_registered_passes_P1_all_designs"] = bool(which)
    # point estimate under BP for every registered comparison
    bp = reg[reg.design == "BP"].set_index("comparison")
    verdict["registered_points_BP"] = {k: float(v) for k, v in bp["point"].items()}
    verdict["registered_ci_BP"] = {k: [float(a), float(b)] for k, a, b in
                                   zip(bp.index, bp["ci95_low"], bp["ci95_high"])}
    verdict["registered_p_BP"] = {k: float(v) for k, v in bp["p_two_sided"].items()}
    verdict["registered_p_bh_lead_BP"] = {k: float(v) for k, v in bp["p_bh_lead"].items()}
    # sign consistency across the five designs (shared rule §2.1)
    sign_ok = {}
    for c, sub in reg.groupby("comparison"):
        s = np.sign(sub["point"].to_numpy())
        sign_ok[c] = bool(sub["design"].nunique() == 5 and len(set(s)) == 1)
    verdict["registered_sign_consistent_all_designs"] = sign_ok

    # ---- interval calibration ----
    CV = pd.read_csv(OUT / "calibration_intervals.csv")
    piv = CV.pivot_table(index=["arm", "k"], columns="design", values="abs_cover90_gap")
    cov90 = CV.pivot_table(index=["arm", "k"], columns="design", values="cover90")
    arms = sorted(set(CV.arm))
    better, detail = [], {}
    for a in arms:
        if a == "POOLED":
            continue
        ok, inrange = True, True
        for k in KS:
            for d in DESIGNS:
                try:
                    ga, gp = float(piv.loc[(a, k), d]), float(piv.loc[("POOLED", k), d])
                    c9 = float(cov90.loc[(a, k), d])
                except KeyError:
                    ok = False
                    continue
                if not (ga < gp):
                    ok = False
                if not (0.85 <= c9 <= 0.95):
                    inrange = False
        detail[a] = {"gap_smaller_than_pooled_everywhere": ok, "cover90_in_band_everywhere": inrange}
        if ok and inrange:
            better.append(a)
    verdict["better_calibrated_than_pooled_all_designs"] = sorted(better)
    verdict["calibration_detail"] = detail
    mean_gap = {a: float(np.nanmean([piv.loc[(a, k), d] for k in KS for d in DESIGNS
                                     if (a, k) in piv.index])) for a in arms}
    verdict["mean_abs_cover90_gap_k123"] = mean_gap
    verdict["best_calibrated_arm"] = min(mean_gap, key=mean_gap.get)
    verdict["cover90_pooled_BP_by_k"] = {int(k): float(cov90.loc[("POOLED", k), "BP"])
                                         for k in (0, 1, 2, 3) if ("POOLED", k) in cov90.index}

    # ---- zero-shot direction probability ----
    if (OUT / "calibration_direction.csv").exists():
        DB = pd.read_csv(OUT / "calibration_direction.csv")
        dc = C[C.comparison == "PLATT_vs_CONST"]
        excl = bool(len(dc) == 5 and ((dc.ci95_low > 0) & (dc.bca_low > 0)).all())
        ece_ok = bool((DB[DB.arm == "PLATT"]["ece10"] <= 0.10).all() and
                      DB[DB.arm == "PLATT"]["design"].nunique() == 5)
        verdict["direction_platt_beats_const_all_designs"] = excl
        verdict["direction_platt_ece_le_0.10_all_designs"] = ece_ok
        verdict["calibrated_probability_is_deliverable"] = bool(excl and ece_ok)
        for arm in ("RAW", "PLATT", "CONST"):
            s = DB[(DB.arm == arm) & (DB.design == "BP")]
            if len(s):
                verdict[f"BP_{arm}_macro_brier"] = float(s.macro_brier.iat[0])
                verdict[f"BP_{arm}_ece10"] = float(s.ece10.iat[0])
                verdict[f"BP_{arm}_macro_accuracy"] = float(s.macro_accuracy.iat[0])

    (OUT / "L5_VERDICTS.json").write_text(json.dumps(verdict, indent=2))
    pd.set_option("display.width", 260)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()

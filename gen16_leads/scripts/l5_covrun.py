"""L5 step 2 + 3 + 4a: five covariance estimators, MIX6meanPC, and interval calibration.

ONE ``l5_fewshot.evaluate`` pass per design carries every estimator, so the union of every
(strategy, estimator) support set is excluded from scoring for all of them and every arm is scored
on byte-identical pairs.  That is the "run all estimators in one evaluate pass" option of the L5
brief; no pair-table intersection is needed and nothing is dropped after the fact.

What is emitted per design:

* ``board_cov.csv``   extractant-macro MAE (and the full frozen metric panel) for
  ``G14_<EST>@doptk<k>``, ``FLAT_<EST>@doptk<k>``, ``NAIVE_LINE_<EST>@doptk<k>`` and
  ``MIX6meanPC_POOLED@doptk<k>``.
* ``contrasts_cov.csv`` / ``contrasts_mix.csv``  one ``paired_contrasts`` call per design over the
  registered and the exploratory families together (so the 10 000 chemotype block draws are shared
  by every comparison), split into the two files afterwards.
* ``calibration_intervals.csv``  coverage of nominal 50/80/90/95 % intervals from the BLUP
  posterior sd (incl. measurement noise), mean sd (sharpness) and |cover90 - 0.90| per arm and k.
* ``covariance_conditioning.csv``  the min/max eigenvalue and condition number of every estimator
  on the REAL leave-chemotype-out publication-masked residual rows.

The three support strategies of the deployed protocol (widest, dopt, random) are all *selected*,
because that is what fixes the scoring set gen15 §5 used, but only the ``dopt`` modes are scored:
dopt is the registered support and scoring 153 modes instead of 55 buys nothing.

Usage:  .venv/Scripts/python.exe gen16_leads/scripts/l5_covrun.py [designs]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402
from gen16 import l5_cov as LC  # noqa: E402
from gen16 import l5_fewshot as L5  # noqa: E402
from gen15 import arms as A  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402

OUT = bootstrap.RESULTS / "L5"
OUT.mkdir(parents=True, exist_ok=True)
DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else list(V.DESIGNS)
ESTS = list(LC.ESTIMATORS)                      # POOLED, LW, LOWRANK2, LOWRANK3, HIER
KS = (0, 1, 2, 3)
HOWS = ("widest", "dopt", "random")
MIX = "MIX6meanPC"
LEVELS = (0.50, 0.80, 0.90, 0.95)


def scored_modes() -> list[str]:
    m = [f"G14_{e}@doptk{k}" for e in ESTS for k in KS]
    m += [f"FLAT_{e}@doptk{k}" for e in ESTS for k in KS]
    m += [f"NAIVE_LINE_{e}@doptk{k}" for e in ESTS for k in KS if k]
    m += [f"{MIX}_POOLED@doptk{k}" for k in KS]
    return m


def comparisons() -> tuple[dict, dict[str, str]]:
    """(label -> (reference, candidate), label -> family)."""
    comps, fam = {}, {}
    for k in (1, 2, 3):
        for e in ESTS:
            if e == "POOLED":
                continue
            comps[f"{e}_vs_POOLED@k{k}"] = (f"G14_POOLED@doptk{k}", f"G14_{e}@doptk{k}")
            fam[f"{e}_vs_POOLED@k{k}"] = "registered"
        comps[f"{MIX}_vs_POOLED@k{k}"] = (f"G14_POOLED@doptk{k}", f"{MIX}_POOLED@doptk{k}")
        fam[f"{MIX}_vs_POOLED@k{k}"] = "registered"
    # ---- exploratory: the cheapest competitor, and the floor ----
    for k in (1, 2, 3):
        for e in ESTS:
            comps[f"{e}_vs_NAIVE@k{k}"] = (f"NAIVE_LINE_{e}@doptk{k}", f"G14_{e}@doptk{k}")
            fam[f"{e}_vs_NAIVE@k{k}"] = "exploratory"
        comps[f"{MIX}_vs_NAIVE@k{k}"] = (f"NAIVE_LINE_POOLED@doptk{k}", f"{MIX}_POOLED@doptk{k}")
        fam[f"{MIX}_vs_NAIVE@k{k}"] = "exploratory"
    for k in KS:
        comps[f"POOLED_vs_FLAT@k{k}"] = (f"FLAT_POOLED@doptk{k}", f"G14_POOLED@doptk{k}")
        fam[f"POOLED_vs_FLAT@k{k}"] = "exploratory"
    for e in ESTS:
        if e == "POOLED":
            continue
        comps[f"{e}_vs_POOLED_FLATprior@k3"] = (f"FLAT_POOLED@doptk3", f"FLAT_{e}@doptk3")
        fam[f"{e}_vs_POOLED_FLATprior@k3"] = "exploratory"
    return comps, fam


def coverage_rows(t: pd.DataFrame, design: str) -> list[dict]:
    """g15_uncertainty.coverage_table logic, one row per (arm, k)."""
    rows = []
    arms = [(f"G14_{e}", f"sd_{e}", e) for e in ESTS] + [(f"{MIX}_POOLED", f"sd_{MIX}_POOLED", MIX)]
    for pred_stem, sd_stem, label in arms:
        for k in KS:
            pc, sc = f"{pred_stem}@doptk{k}", f"{sd_stem}@doptk{k}"
            if pc not in t.columns or sc not in t.columns:
                continue
            pred, sd = t[pc], t[sc]
            err = (t["y"] - pred).abs()
            rec = {"design": design, "arm": label, "k": k, "n_pairs": int(len(t)),
                   "mae": float(err.mean()), "rmse": float(np.sqrt((err ** 2).mean())),
                   "mean_sd": float(sd.mean()), "median_sd": float(sd.median()),
                   "z_sd": float((err / sd).std()), "median_abs_z": float((err / sd).median())}
            for lv in LEVELS:
                z = stats.norm.ppf(0.5 + lv / 2)
                rec[f"cover{int(lv * 100)}"] = float((err <= z * sd).mean())
            rec["abs_cover90_gap"] = abs(rec["cover90"] - 0.90)
            rows.append(rec)
    return rows


def main() -> None:
    t0 = time.time()
    bench = V.load()
    boards, cons, cover, checks = [], [], [], {}
    modes = scored_modes()
    comps, fam = comparisons()
    for d in DESIGNS:
        td = time.time()
        r = L5.evaluate(bench, {"G14": A.g14, "FLAT": A.flat}, d, ks=KS, hows=HOWS,
                        cov_kind="empirical", mask_publication=True,
                        cov_estimators=LC.ESTIMATORS, mixtures={MIX: 6}, verbose=True)
        t = r.pairs
        if t.empty:
            print(f"  {d}: nothing scorable")
            continue
        have = [m for m in modes if m in t.columns]
        # ---- invariants: at k = 0 no measurement is used, so the covariance cannot matter,
        # and the shifted mixture prior mean is the G14 curve itself ----
        k0 = {e: float((t[f"G14_{e}@doptk0"] - t["G14_POOLED@doptk0"]).abs().max()) for e in ESTS}
        k0[MIX] = float((t[f"{MIX}_POOLED@doptk0"] - t["G14_POOLED@doptk0"]).abs().max())
        checks[d] = {"k0_identity_maxabs": k0, "n_pairs": int(len(t)),
                     "n_cells": int(t.cell_id.nunique()), "n_extractants": int(t.extractant.nunique()),
                     "n_chemotypes": int(t.chemotype.nunique()), "seconds": time.time() - td}
        assert max(k0.values()) < 1e-9, f"{d}: k=0 predictions are not estimator-invariant: {k0}"
        pe = per_extractant(t, have)
        pe.insert(0, "design", d)
        pe.to_parquet(OUT / f"perext_cov_{d}.parquet")
        bd = summarise(pe, t, have)
        bd.insert(0, "design", d)
        boards.append(bd)
        c = paired_contrasts(pe, comps, value="mae_all", replicates=10_000)
        if len(c):
            c.insert(0, "design", d)
            c["family"] = c["comparison"].map(fam)
            c["lead"] = "L5"
            cons.append(c)
        cover.extend(coverage_rows(t, d))
        print(f"  [{d}] scored {len(have)} modes on {len(t)} pairs, "
              f"{time.time() - td:.0f}s", flush=True)
        del t, r

    B = pd.concat(boards, ignore_index=True)
    B.to_csv(OUT / "board_cov.csv", index=False)
    C = pd.concat(cons, ignore_index=True)
    is_mix = C.comparison.str.startswith(MIX)
    C[~is_mix].to_csv(OUT / "contrasts_cov.csv", index=False)
    C[is_mix].to_csv(OUT / "contrasts_mix.csv", index=False)
    mix_arms = [f"{MIX}_POOLED@doptk{k}" for k in KS] + \
               [f"G14_POOLED@doptk{k}" for k in KS] + \
               [f"NAIVE_LINE_POOLED@doptk{k}" for k in KS if k]
    B[B.arm.isin(mix_arms)].to_csv(OUT / "board_mix.csv", index=False)
    CV = pd.DataFrame(cover)
    CV.to_csv(OUT / "calibration_intervals.csv", index=False)
    (OUT / "covrun_checks.json").write_text(json.dumps(checks, indent=2))

    pd.set_option("display.width", 260)
    print("\n=== extractant-macro MAE, common scoring set, dopt support ===")
    w = V.wide(B)
    order = [f"G14_{e}@doptk{k}" for k in KS for e in ESTS] + \
            [f"{MIX}_POOLED@doptk{k}" for k in KS] + \
            [f"NAIVE_LINE_{e}@doptk{k}" for k in (1, 2, 3) for e in ESTS] + \
            [f"FLAT_{e}@doptk{k}" for k in KS for e in ESTS]
    print(w.reindex([m for m in order if m in w.index]).round(4).to_string())
    print("\n=== registered contrasts (positive favours the candidate) ===")
    show = ["design", "comparison", "point", "ci95_low", "ci95_high", "bca_low", "bca_high",
            "p_two_sided", "n_units", "seeds_positive", "loco_sign_stable", "passes_P1"]
    print(C[C.family == "registered"][show].round(4).to_string(index=False))
    print("\n=== interval calibration (cover90 and sharpness) ===")
    piv = CV.pivot_table(index=["arm", "k"], columns="design", values="cover90")
    print(piv.round(3).to_string())
    print(f"\n[l5-covrun] total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

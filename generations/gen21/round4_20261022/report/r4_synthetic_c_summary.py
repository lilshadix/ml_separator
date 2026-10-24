#!/usr/bin/env python
"""Summary of the block-C synthetic discrimination study (reporting only; reads results/synthetic_c/rep_*.json).

Counts k/n with exact Clopper-Pearson 95 % intervals for the rules AS THEY ARE APPLIED (PROTOCOL section 12), an upper
bound of the sealed rule (vs B1 only, no harm rule, no B2, one fold, no seed mean):
  transfer claim = crit1(S13) and S13 - S11 >= 0.005 and S13 - S15 >= 0.005 and (S15 - S11) < 0.5 (S13 - S11)
                   and S13 > max(S16 draws) and mean(S16 - S11) < 0.5 (S13 - S11)     (recomputed here from the gains)
  selectivity sub-label = S13 - S14 > 0, overall and given the claim
  gate-equivalent = S13's best positive-w_X inner score below S11's selected and B2's best inner score
Also: crit1 of S11 and S13, the S13 - S11 >= 0.005 leg, mean dR2 vs B1 per arm, and the PAIRED S13 - S11 difference of
each world against NULL (same Ln targets per replicate).  Writes results/synthetic_c/SUMMARY.md and summary.json (the
firing counts report/r4_decide.py C reads for the sealed synthetic-power consequence).

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/report/r4_synthetic_c_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as SS

R4 = Path(__file__).resolve().parents[1]
WORLDS = ("NULL", "SMOOTH_NULL", "AN_SIGNAL")


def cp(v) -> str:
    v = np.asarray(v, bool)
    k, n = int(v.sum()), len(v)
    if n == 0:
        return "0/0"
    lo = 0.0 if k == 0 else SS.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else SS.beta.ppf(0.975, k + 1, n - k)
    return f"{k}/{n} [{lo:.2f}, {hi:.2f}]"


def claim(r) -> bool:
    g = r["gain"]
    s16 = [v for k, v in g.items() if k.startswith("S16")]
    up = g["S13:0"] - g["S11:0"]
    rule = (up >= 0.005 and g["S13:0"] - g["S15:0"] >= 0.005 and (g["S15:0"] - g["S11:0"]) < 0.5 * up
            and g["S13:0"] > max(s16) and np.mean([v - g["S11:0"] for v in s16]) < 0.5 * up)
    return bool(r["crit1_S13"] and rule)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    reps = [json.loads(p.read_text()) for p in sorted((R4 / "results" / "synthetic_c").glob("rep_*.json"))]
    lines = ["# Block C synthetic discrimination study", "",
             f"{len(reps)} paired replicates on two BPG folds (seed 130363, folds 2 and 3): synthetic Ln targets (chemotype "
             "nuisance, noise sd 0.11, signal 0.15 · u · tetrad_e3) shared by the three worlds of a replicate; synthetic "
             "An targets on the real An(III) record set. Counts k/n with exact 95 % intervals. Rules are an upper bound of "
             "the sealed ones (vs B1 only, no harm rule, no B2, one fold, no seed mean).", "",
             "| world | crit1 S11 | crit1 S13 | S13 − S11 ≥ 0.005 | transfer claim | selectivity overall / given claim | "
             "gate-equivalent | S13 chose w_X > 0 | mean ΔR² S11 / S13 / S14 / S15 / S16 | paired S13 − S11 minus NULL's |",
             "|" + "---|" * 10]
    summ = {}
    for w in WORLDS:
        rs = [r["worlds"][w] for r in reps if w in r["worlds"]]
        if not rs:
            continue
        g = lambda k: np.mean([r["gain"][k] for r in rs])  # noqa: E731
        s16 = np.mean([np.mean([v for k, v in r["gain"].items() if k.startswith("S16")]) for r in rs])
        leg = [r["gain"]["S13:0"] - r["gain"]["S11:0"] >= 0.005 for r in rs]
        cl = [claim(r) for r in rs]
        pos = [isinstance(r["choices"]["S13:0"], dict) and r["choices"]["S13:0"]["w_X"] > 0 for r in rs]
        d = [(r["worlds"][w]["gain"]["S13:0"] - r["worlds"][w]["gain"]["S11:0"])
             - (r["worlds"]["NULL"]["gain"]["S13:0"] - r["worlds"]["NULL"]["gain"]["S11:0"]) for r in reps]
        summ[w] = {"n": len(rs), "transfer_claim": int(sum(cl)), "crit1_S13": int(sum(r["crit1_S13"] for r in rs)),
                   "crit1_S11": int(sum(r["crit1_S11"] for r in rs)), "S13_minus_S11_leg": int(sum(leg)),
                   "gate_equivalent": int(sum(r["gate_win"] for r in rs)), "S13_positive_w_X": int(sum(pos)),
                   "mean_gain": {k: float(g(k)) for k in ("S11:0", "S13:0", "S14:0", "S15:0")} | {"S16": float(s16)},
                   "paired_S13_minus_S11_vs_NULL": [float(x) for x in d]}
        lines.append(f"| {w} | {cp([r['crit1_S11'] for r in rs])} | {cp([r['crit1_S13'] for r in rs])} | {cp(leg)} | "
                     f"{cp(cl)} | {cp([r['selectivity'] for r in rs])} / "
                     f"{cp([r['selectivity'] for r, c in zip(rs, cl) if c])} | "
                     f"{cp([r['gate_win'] for r in rs])} | {int(np.sum(pos))}/{len(rs)} | "
                     f"{g('S11:0'):+.4f} / {g('S13:0'):+.4f} / {g('S14:0'):+.4f} / {g('S15:0'):+.4f} / {s16:+.4f} | "
                     f"{np.mean(d):+.4f} (min {np.min(d):+.4f}, max {np.max(d):+.4f}) |")
    lines += ["", "S11's chosen networks: " + ", ".join(str(r["worlds"]["NULL"]["choices"]["S11:0"]) for r in reps)]
    out = R4 / "results" / "synthetic_c"
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

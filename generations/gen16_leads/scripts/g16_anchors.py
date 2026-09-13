"""Gen16 phase 0: reproduce the three anchors of START_HERE.md section 3 and record every digit.

  1. gen13 locked stage-3 headline, G13_ET_TOPO39 under BP  = 0.7683085207475452  (exact)
  2. gen14 deployed model G14 under BP                       = 0.5001              (4 decimals)
  3. FLAT under BP                                           = 0.589

plus G14 / FLAT under all five designs with per-design timing, and a subprocess check that the
fold plan is byte-identical across processes.  Writes results/anchors/{anchors.json,
g14_flat_five_designs.csv, g14_vs_flat_contrasts_five_designs.csv, ANCHORS.md}.

Usage:  .venv/Scripts/python.exe generations/gen16_leads/scripts/g16_anchors.py [--skip-g13] [--hash-runs N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402
from gen16 import anchors as AN  # noqa: E402
from gen16.foldhash import fold_plan_digest  # noqa: E402

OUT = bootstrap.RESULTS / "anchors"
HASH_SCRIPT = bootstrap.G16 / "scripts" / "g16_fold_hash.py"
FIVE = ("B", "BR", "BQ", "A", "BP")


def _py(v):
    return v.item() if hasattr(v, "item") else v


def subprocess_hashes(n: int) -> list[dict]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    out = []
    for _ in range(n):
        r = subprocess.run([sys.executable, str(HASH_SCRIPT)], capture_output=True, text=True,
                           check=True, env=env)
        out.append(json.loads(r.stdout.strip().splitlines()[-1]))
    return out


def environment() -> dict:
    import joblib
    import numpy
    import pandas
    import scipy
    import sklearn
    try:
        import rdkit
        rd = rdkit.__version__
    except Exception:  # pragma: no cover
        rd = None
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "joblib": joblib.__version__, "rdkit": rd,
            "LOKY_MAX_CPU_COUNT": os.environ.get("LOKY_MAX_CPU_COUNT"),
            "joblib_effective_n_jobs_minus1": joblib.effective_n_jobs(-1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-g13", action="store_true", help="skip the extra-trees anchor")
    ap.add_argument("--hash-runs", type=int, default=2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from gen14.dirbench import load
    from gen15 import valuebench as V

    rec: dict = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
                 "environment": environment(), "timing_seconds": {}}
    T = rec["timing_seconds"]

    # -- fold plan: separate processes first, then in-session -------------------------------
    t0 = time.time()
    subs = subprocess_hashes(args.hash_runs) if args.hash_runs > 0 else []
    T["fold_hash_subprocesses"] = time.time() - t0
    t0 = time.time()
    bench = load()
    T["bench_load"] = time.time() - t0
    t0 = time.time()
    digest = fold_plan_digest(bench.frame)
    T["fold_hash_in_session"] = time.time() - t0
    stable = bool(all(s["sha256"] == digest["sha256"]
                      and s["sha256_train_test"] == digest["sha256_train_test"] for s in subs))
    rec["cohort"] = {"cells": int(len(bench.frame)),
                     "extractants": int(bench.frame.extractant.nunique()),
                     "chemotypes": int(bench.frame.chemotype.nunique())}
    rec["fold_plan"] = {**digest, "subprocess_sha256": [s["sha256"] for s in subs],
                        "subprocess_sha256_train_test": [s["sha256_train_test"] for s in subs],
                        "subprocess_runs": len(subs), "subprocess_stable": stable}
    print(f"[fold plan] sha256 {digest['sha256']}  n_folds {digest['n_folds']}  "
          f"subprocess runs {len(subs)} identical={stable}", flush=True)

    # -- anchors 2 and 3, and the five-design board -----------------------------------------
    B, C, timing = AN.value_board(bench, FIVE)
    T.update({f"value_{d}": s for d, s in timing.items()})
    W = V.wide(B)
    g14 = {d: float(W.loc["G14", d]) for d in FIVE}
    flat = {d: float(W.loc["FLAT", d]) for d in FIVE}
    con = {}
    for d in FIVE:
        row = C[(C.design == d) & (C.comparison == "G14_vs_FLAT")].iloc[0]
        con[d] = {k: _py(row[k]) for k in AN.CONTRAST_COLUMNS}
    rec["five_designs"] = {
        "metric": "extractant-macro MAE of log SF, 5 discovery seeds",
        "G14_macro_mae_extractant": g14, "FLAT_macro_mae_extractant": flat,
        "G14_vs_FLAT": con, "G14_reported_in_GEN14_REPORT": AN.REPORTED_G14_FIVE,
        "G14_matches_report_3dp": {d: round(g14[d], 3) == AN.REPORTED_G14_FIVE[d] for d in FIVE}}
    B.assign(seconds_design=B.design.map(timing)).to_csv(OUT / "g14_flat_five_designs.csv",
                                                          index=False)
    C.to_csv(OUT / "g14_vs_flat_contrasts_five_designs.csv", index=False)
    print("\n=== extractant-macro MAE of log SF, 5 discovery seeds ===")
    print(W.to_string(float_format=lambda x: f"{x:.16f}"), flush=True)
    print("timing per design (s):", {d: round(s, 1) for d, s in timing.items()}, flush=True)

    anchors = {
        "G14_BP_macro_mae_extractant": {
            "expected": AN.EXPECTED_G14_BP, "expected_4dp": 0.5001, "obtained": g14["BP"],
            "obtained_repr": repr(g14["BP"]), "match_4dp": round(g14["BP"], 4) == 0.5001,
            "exact_match": g14["BP"] == AN.EXPECTED_G14_BP},
        "FLAT_BP_macro_mae_extractant": {
            "expected": AN.EXPECTED_FLAT_BP, "expected_3dp": 0.589, "obtained": flat["BP"],
            "obtained_repr": repr(flat["BP"]), "match_3dp": round(flat["BP"], 3) == 0.589,
            "match_4dp": round(flat["BP"], 4) == 0.5885,
            "exact_match": flat["BP"] == AN.EXPECTED_FLAT_BP},
        "G14_vs_FLAT_BP": {
            "expected": AN.EXPECTED_G14_VS_FLAT_BP,
            "obtained": {k: con["BP"][k] for k in ("point", "ci95_low", "ci95_high",
                                                   "p_two_sided")},
            "match_4dp": all(round(con["BP"][k], 4) == round(v, 4)
                             for k, v in AN.EXPECTED_G14_VS_FLAT_BP.items())},
    }

    # -- anchor 1: gen13 stage-3 headline -----------------------------------------------------
    if not args.skip_g13:
        acc, board, secs = AN.g13_direction_anchor(bench)
        T["G13_ET_TOPO39_BP"] = secs
        anchors["G13_ET_TOPO39_BP_macro_direction_accuracy"] = {
            "expected": AN.LOCKED_G13_BP, "expected_repr": repr(AN.LOCKED_G13_BP),
            "obtained": acc, "obtained_repr": repr(acc), "exact_match": acc == AN.LOCKED_G13_BP,
            "abs_diff": abs(acc - AN.LOCKED_G13_BP),
            "board": {k: _py(v) for k, v in board.iloc[0].to_dict().items()}}
        print(f"\n[anchor 1] G13_ET_TOPO39 @ BP = {acc!r}  locked {AN.LOCKED_G13_BP!r}  "
              f"{'MATCH' if acc == AN.LOCKED_G13_BP else 'MISMATCH'}  ({secs:.0f}s)", flush=True)
    rec["anchors"] = anchors
    rec["g13_anchor_run"] = not args.skip_g13
    rec["all_reproduced"] = bool(
        stable and anchors["G14_BP_macro_mae_extractant"]["match_4dp"]
        and anchors["FLAT_BP_macro_mae_extractant"]["match_3dp"]
        and anchors["G14_vs_FLAT_BP"]["match_4dp"]
        and (args.skip_g13 or anchors["G13_ET_TOPO39_BP_macro_direction_accuracy"]["exact_match"]))
    (OUT / "anchors.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    write_markdown(rec)
    print(f"\nall_reproduced={rec['all_reproduced']} -> {OUT / 'anchors.json'}", flush=True)
    return 0 if rec["all_reproduced"] else 1


def write_markdown(rec: dict) -> None:
    a, fd, T = rec["anchors"], rec["five_designs"], rec["timing_seconds"]
    L = ["# Gen16 anchors (phase 0)", "",
         f"Generated {rec['generated']}.  Regime: 5 discovery split seeds "
         "(`gen13sep.splits.SPLIT_SEEDS`), extractant-macro metrics.  Produced by "
         "`gen16_leads/scripts/g16_anchors.py`; guarded by `gen16_leads/tests/test_anchors.py`.",
         "", "| anchor | regime | expected | obtained | match |", "|---|---|---|---|---|"]
    g = a.get("G13_ET_TOPO39_BP_macro_direction_accuracy")
    if g:
        status = "exact" if g["exact_match"] else f"MISMATCH (abs diff {g['abs_diff']:.3g})"
        L.append("| gen13 stage-3 headline `G13_ET_TOPO39` | macro direction accuracy, BP | "
                 f"`{g['expected_repr']}` | `{g['obtained_repr']}` | {status} |")
    else:
        L.append("| gen13 stage-3 headline `G13_ET_TOPO39` | macro direction accuracy, BP | "
                 "`0.7683085207475452` | not run (`--skip-g13`) | - |")
    x = a["G14_BP_macro_mae_extractant"]
    status = ("4 dp" + (" and exact" if x["exact_match"] else "")) if x["match_4dp"] else "MISMATCH"
    L.append(f"| gen14 deployed model `G14` | macro MAE of log SF, BP | 0.5001 "
             f"(`{x['expected']!r}`) | `{x['obtained_repr']}` | {status} |")
    x = a["FLAT_BP_macro_mae_extractant"]
    status = ("3 dp" + (" and exact" if x["exact_match"] else "")) if x["match_3dp"] else "MISMATCH"
    L.append(f"| `FLAT` (no separation) | macro MAE of log SF, BP | 0.589 "
             f"(`{x['expected']!r}`) | `{x['obtained_repr']}` | {status} |")
    x = a["G14_vs_FLAT_BP"]
    e, o = x["expected"], x["obtained"]
    L.append("| `G14_vs_FLAT` contrast | chemotype-blocked paired bootstrap, BP | "
             f"{e['point']:+.4f} [{e['ci95_low']:+.4f}, {e['ci95_high']:+.4f}] p {e['p_two_sided']:.4f} | "
             f"{o['point']:+.4f} [{o['ci95_low']:+.4f}, {o['ci95_high']:+.4f}] p {o['p_two_sided']:.4f} | "
             f"{'4 dp' if x['match_4dp'] else 'MISMATCH'} |")
    L += ["", "## G14 and FLAT under all five designs", "",
          "Extractant-macro MAE of log SF, 5 discovery seeds.  GEN14_REPORT quotes G14 as "
          "0.493 / 0.491 / 0.491 / 0.492 / 0.500.", "",
          "| arm | B | BR | BQ | A | BP |", "|---|---|---|---|---|---|"]
    for arm, key in (("G14", "G14_macro_mae_extractant"), ("FLAT", "FLAT_macro_mae_extractant")):
        L.append(f"| `{arm}` | " + " | ".join(f"{fd[key][d]:.6f}" for d in FIVE) + " |")
    L.append("| `G14_vs_FLAT` point | "
             + " | ".join(f"{fd['G14_vs_FLAT'][d]['point']:+.4f}" for d in FIVE) + " |")
    L.append("| `G14_vs_FLAT` 95 % CI | "
             + " | ".join(f"[{fd['G14_vs_FLAT'][d]['ci95_low']:+.4f}, "
                          f"{fd['G14_vs_FLAT'][d]['ci95_high']:+.4f}]" for d in FIVE) + " |")
    L.append("| `G14_vs_FLAT` p | "
             + " | ".join(f"{fd['G14_vs_FLAT'][d]['p_two_sided']:.4f}" for d in FIVE) + " |")
    L.append("| seconds | " + " | ".join(f"{T['value_' + d]:.0f}" for d in FIVE) + " |")
    L.append("| G14 matches report at 3 dp | "
             + " | ".join(str(fd["G14_matches_report_3dp"][d]) for d in FIVE) + " |")
    fp = rec["fold_plan"]
    L += ["", "## Fold-plan determinism", "",
          "SHA-256 over the concatenated sorted test-index arrays of every fold, designs in order "
          f"B, BR, BQ, A, BP: `{fp['sha256']}`  ",
          f"stricter train+test digest: `{fp['sha256_train_test']}`  ",
          f"folds per design: {fp['n_folds']}  ",
          f"{fp['subprocess_runs']} separate subprocess runs identical to the in-session digest: "
          f"**{fp['subprocess_stable']}**", "",
          "## Environment", "", "```", json.dumps(rec["environment"], indent=2), "```", "",
          f"Timings (s): {json.dumps({k: round(v, 1) for k, v in T.items()})}", "",
          f"**all_reproduced = {rec['all_reproduced']}**"]
    (OUT / "ANCHORS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

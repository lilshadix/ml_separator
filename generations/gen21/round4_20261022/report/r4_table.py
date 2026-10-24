#!/usr/bin/env python
"""Compact result table (brief section 9) for one round-4 configuration, from its stored results only (no refit,
no new computation on predictions).  Reporting code; not part of the stage-0 sealed set.

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/report/r4_table.py REF
writes results/<ID>/TABLE.md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

R4 = Path(__file__).resolve().parents[1]
DESIGNS = ("BPG", "BP")


def f4(v):
    return "NaN" if pd.isna(v) else f"{v:+.4f}" if abs(v) < 1 else f"{v:.3f}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cid = sys.argv[1]
    res = R4 / "results" / cid
    lines = [f"# {cid}: compact result table (exploratory development; V1 / C3 / k = 3 / DOPT_Q)", ""]
    run = (R4 / "RUN_RECORD.md").read_text(encoding="utf-8")
    for d in DESIGNS:
        s = pd.read_csv(res / f"summary_{d}.csv")
        c = pd.read_csv(res / f"contrasts_{d}.csv")
        cl = json.loads((res / f"clusters_{d}.json").read_text())
        iv = json.loads((res / f"intervention_{d}.json").read_text())
        ident = json.loads((res / f"identity_{d}.json").read_text()) if (res / f"identity_{d}.json").exists() else None
        if ident is not None and "n_inf" not in ident:                      # slot identity check
            ident = {"passed": ident["passed"], "n_inf": ident["lambda_inf_by_column"], "n_folds": 25, "n_orphan_folds": "-"}
        lv = pd.read_csv(res / f"levels_{d}.csv")
        m = re.findall(rf"END {cid} {d} key=\S+: .*?(\d+)s", run)
        lines += [f"## {d}" + (" (deciding design)" if d == "BPG" else " (robustness)"), "",
                  f"- **Counts**: {cl['cells']} cells, {cl['rows_per_seed']} pairs per seed, {cl['extractant']['nominal']} "
                  f"extractants, {cl['publication']['nominal']} publications; chemotype clusters {cl['chemotype']['nominal']} "
                  f"(Kish {cl['chemotype']['kish_rows']:.2f}); source clusters cluster_cpg {cl['cluster_cpg']['nominal']} "
                  f"(Kish {cl['cluster_cpg']['kish_rows']:.2f}, largest {cl['cluster_cpg']['largest_share']:.0%}).",
                  f"- **Intervention**: {'passed' if iv['passed'] else 'FAILED'} ({iv.get('n_predictions')} predictions, max change "
                  f"{iv.get('max_abs_change')}); identity: " +
                  (f"{'passed' if ident['passed'] else 'FAILED'} (lambda = inf in {ident['n_inf']} of {ident['n_folds']} folds; "
                   f"orphan folds {ident['n_orphan_folds']})" if ident else "n/a") +
                  (f"; compute {m[-1]} s." if m else "."), ""]
        a = s[s.subset == "all"].set_index("arm")
        lv = lv[(lv.statistic == "R2") & (lv.cluster == "cluster_cpg")].set_index("arm")
        lines += ["| arm | R² (seed mean) | R² 95% (cpg boot) | min seed | balanced MAE | MAE | RMSE |", "|---|---|---|---|---|---|---|"]
        own = [x for x in a.index if x not in ("B0", "B1", "B2", "HIST_D7", "R14_REPRO", "PWL_R", "PWL_Z", "HBK@ML", "RIDGE2",
                                               "LINE", "G14@ZERO", "FLAT")]
        order = ["B0", "B1", "B2"] + own + ["HIST_D7", "R14_REPRO", "PWL_Z", "HBK@ML", "RIDGE2", "FLAT"]
        for arm in [x for x in order if x in a.index]:
            r = a.loc[arm]
            ci = f"[{lv.loc[arm, 'ci_low']:.3f}, {lv.loc[arm, 'ci_high']:.3f}]" if arm in lv.index else ""
            lines.append(f"| {arm} | {r.R2:.4f} | {ci} | {r.R2_min_seed:.4f} | {r.MAE_balanced:.4f} | {r.MAE:.4f} | {r.RMSE:.4f} |")
        lines += ["", "| contrast | statistic | clusters | point | percentile 95% | BCa 95% | jackknife-t 95% | LOCO min | seeds + |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in c[c.statistic.isin(["R2", "MAE_balanced"])].itertuples():
            lines.append(f"| {r.name} | {r.statistic} | {r.cluster} | {f4(r.point)} | [{f4(r.ci_low)}, {f4(r.ci_high)}] | "
                         f"[{f4(r.bca_low)}, {f4(r.bca_high)}] | [{f4(r.jk_t_low)}, {f4(r.jk_t_high)}] | {f4(r.loco_min)} | "
                         f"{int(r.seeds_positive)}/{int(r.n_seeds)} |")
        sub = s[s.subset != "all"]
        shown = ["B0", "B1", "B2"] + own[:3]
        lines += ["", "| subset | " + " | ".join(f"{x} R²" for x in shown) + " | " + " | ".join(f"{x} MAE" for x in shown[1:]) +
                  " | rows / seed |", "|" + "---|" * (2 * len(shown) + 1)]
        for name, g in sub.groupby("subset", sort=False):
            g = g.set_index("arm")
            lines.append(f"| {name} | " + " | ".join(f"{g.loc[x, 'R2']:.4f}" for x in shown) + " | " +
                         " | ".join(f"{g.loc[x, 'MAE']:.4f}" for x in shown[1:]) + f" | {g.loc['B0', 'n_rows_per_seed']:.0f} |")
        lines.append("")
    pois = R4 / "results" / "leakage" / f"poison_{cid}.json"
    if pois.exists():
        p = json.loads(pois.read_text())
        lines.append(f"**Fit poisoning** ({p['n_folds']} folds, both designs): {'all identical' if p['all_identical'] else 'FAILED'}.")
    (res / "TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

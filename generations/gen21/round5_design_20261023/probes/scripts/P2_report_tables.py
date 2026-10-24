#!/usr/bin/env python
"""P2 (M34): compact markdown tables for P2_power_replay.md from probes/P2_out/*.csv (read-only on the inputs).
    .venv/Scripts/python.exe -B generations/gen21/round5_design_20261023/probes/scripts/P2_report_tables.py
writes probes/P2_out/P2_report_tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "P2_out"
UN = {"cluster_cpg": "cpg", "publication": "pub", "chemotype": "chem", "pubchem": "pub×chem"}
TESTS = ["SIGN", "SF_LOG", "WCR_LOG", "T_LOG", "IM_REL", "SF_POOL", "WCR_POOL", "JK_KISH", "JK_G1", "SEALED"]
lines: list[str] = []


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> None:
    cols = [str(c) for c in df.columns]
    lines.append("| " + " | ".join([df.index.name or ""] + cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for idx, r in df.iterrows():
        cells = []
        for v in r.to_numpy():
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    lines.append("")


def h(s):
    lines.append(s)
    lines.append("")


def g80(sub, col, target=0.8):
    s = sub.sort_values("G")
    Gs, pw = s.G.to_numpy(float), s[col].to_numpy(float)
    ok = np.where(pw >= target)[0]
    if not len(ok):
        return ">80"
    i = ok[0]
    if i == 0:
        return f"≤{int(Gs[0])}"
    g0, g1, p0, p1 = Gs[i - 1], Gs[i], pw[i - 1], pw[i]
    return f"{np.exp(np.log(g0) + (target - p0) / (p1 - p0) * (np.log(g1) - np.log(g0))):.0f}"


def main():
    designs = [d for d in ("BPG", "BP") if (OUT / f"P2_grid_{d}.csv").exists()]
    g = pd.concat([pd.read_csv(OUT / f"P2_grid_{d}.csv") for d in designs], ignore_index=True)
    dev = pd.concat([pd.read_csv(OUT / f"P2_dev_tests_{d}.csv").assign(design=d) for d in designs], ignore_index=True)

    # T1 dev
    h("### T1. Dev contrasts on the 229 C3 cells (EXPLORATORY; seed-mean per-unit SSE; one-sided p, H1: candidate better)")
    for d in designs:
        x = dev[dev.design == d].copy()
        x["unit"] = x.unit_type.map(UN)
        x["W/L/T"] = x.wins.astype(str) + "/" + x.losses.astype(str) + "/" + x.ties.astype(str)
        x = x.set_index(x.contrast + " · " + x.unit)
        x.index.name = f"{d}: contrast · unit"
        md(x[["M", "W/L/T", "meanL_nontied", "pooled_rel_SSE", "p_SIGN", "p_SF_LOG", "p_WCR_LOG", "p_T_LOG", "p_SF_POOL",
              "p_WCR_POOL", "p_JK_KISH", "p_JK_G1"]], "{:.4f}")

    # T2 size
    nul = g[g.world.isin(["SWAP", "SYM"]) & ~g.pool.str.contains(r"\|")].copy()
    nul["flag"] = nul.cp_lo_025 > 0.035
    nul["flag05"] = nul.cp_lo_05 > 0.07
    h("### T2. Size under the gating nulls (SWAP on 10 real pools + SYM B1 vs B1; tau 0 / 0.065 (SYM 0.065 / 0.13); "
      "G 5-80; both designs). Cell entry: max rejection at nominal 0.025 [cells with CP-95 % lower bound > 0.035 / cells]")
    rows = {}
    for t in TESTS:
        r = {}
        for u in UN:
            s = nul[(nul.test == t) & (nul.unit_type == u)]
            if s.empty:
                r[UN[u]] = ""
                continue
            r[UN[u]] = f"{s.rej_025.max():.3f} [{int(s.flag.sum())}/{len(s)}]"
        s = nul[nul.test == t]
        r["mean over cells"] = f"{s.rej_025.mean():.4f}"
        r["max at 0.05 [CP>0.07]"] = f"{s.rej_05.max():.3f} [{int(s.flag05.sum())}/{len(s)}]"
        rows[t] = r
    x = pd.DataFrame(rows).T
    x.index.name = "test"
    md(x)
    # size by G (median across pools) for SYM / B1-B0 SWAP
    h("### T2b. Size at 0.025 by G, mean over all gating null cells of that G (both designs, all units, pools, tau)")
    x = nul.pivot_table(index="test", columns="G", values="rej_025", aggfunc="mean").reindex(TESTS)
    md(x, "{:.4f}")
    h("### T2c. Size at 0.025, SWAP null of the B1-B0 pool and SYM (B1 vs B1, tau 0.065), BPG, G = 8 / 20 / 40")
    for pool, world, tau in (("B1-B0", "SWAP", 0.0), ("B1-B0", "SWAP", 0.065), ("B1-B1", "SYM", 0.065)):
        s = nul[(nul.pool == pool) & (nul.world == world) & (nul.tau == tau) & (nul.design == "BPG") & nul.G.isin([8, 20, 40])]
        x = s.assign(col=s.unit_type.map(UN) + " G" + s.G.astype(str)).pivot_table(index="test", columns="col", values="rej_025")
        x = x[[f"{UN[u]} G{G}" for u in UN for G in (8, 20, 40)]].reindex(TESTS)
        x.index.name = f"{pool} {world} tau {tau}"
        md(x)

    # T3 power B1-B0
    for d in designs:
        h(f"### T3 ({d}). Power of B1-B0 (EMP world), one-sided 0.025; columns unit · G; tau 0 | tau 0.065")
        for tau in (0.0, 0.065):
            s = g[(g.pool == "B1-B0") & (g.world == "EMP") & (g.design == d) & (g.tau == tau) & g.G.isin([5, 8, 12, 20, 40, 80])]
            x = s.assign(col=s.unit_type.map(UN) + " " + s.G.astype(str)).pivot_table(index="test", columns="col",
                                                                                     values="rej_025")
            x = x[[f"{UN[u]} {G}" for u in UN for G in (8, 12, 20, 40, 80)]].reindex(TESTS)
            x.index.name = f"tau {tau}"
            md(x, "{:.2f}")

    # T4 G80
    for pool in ("B1-B0", "B1s50-B0", "B1s2-B0", "B1-B0|perseed", "B1-B0|DGA", "B1-B0|nonDGA"):
        for a in ("025", "05"):
            if pool != "B1-B0" and a == "05":
                continue
            s = g[(g.pool == pool) & (g.world == "EMP")]
            if s.empty:
                continue
            rows = {}
            for t in TESTS:
                r = {}
                for (d, u, tau), ss in s[s.test == t].groupby(["design", "unit_type", "tau"]):
                    r[f"{d} {UN[u]} τ{tau:g}"] = g80(ss, f"rej_{a}")
                rows[t] = r
            x = pd.DataFrame(rows).T
            if x.empty:
                continue
            order = [f"{d} {UN[u]} τ{tau:g}" for d in designs for u in UN for tau in (0.0, 0.065)]
            x = x[[c for c in order if c in x.columns]].dropna(how="all")
            x.index.name = "test"
            h(f"### T4. New units G for 80 % power (log-G interpolation), {pool}, one-sided 0.{a}")
            md(x)

    # T5 estimand mismatch / slots
    h("### T5. EMP rejection at 0.025 for contrasts that are null or negative on the pooled estimand (BPG), G 20 / 40")
    pools = ["HBK@ML-B0", "HIST_D7-B1", "S01-B1", "S02-B1", "S05-B1", "S06-B1", "S11-B1"]
    for tau in (0.0, 0.065):
        s = g[g.pool.isin(pools) & (g.world == "EMP") & (g.design == "BPG") & (g.tau == tau) & g.G.isin([20, 40]) &
              g.unit_type.isin(["cluster_cpg", "publication", "chemotype"]) &
              g.test.isin(["SIGN", "SF_LOG", "WCR_LOG", "SF_POOL", "WCR_POOL", "JK_KISH"])]
        x = s.assign(col=s.unit_type.map(UN) + " G" + s.G.astype(str), row=s.pool + " · " + s.test).pivot_table(
            index="row", columns="col", values="rej_025")
        x = x[[f"{UN[u]} G{G}" for u in ("cluster_cpg", "publication", "chemotype") for G in (20, 40)]]
        x.index.name = f"tau {tau}: pool · test"
        md(x)

    # T6 C-worlds
    h("### T6. Skewed heterogeneous nulls from the B1-B0 pool (NOT gating): max rejection at 0.025 over G 8-80, both tau, BPG")
    c = g[g.world.str.startswith("C-") & (g.design == "BPG") & (g.G >= 8)]
    x = c.assign(col=c.world + " " + c.unit_type.map(UN)).pivot_table(index="test", columns="col", values="rej_025",
                                                                      aggfunc="max")
    x = x[[f"{w} {UN[u]}" for w in ("C-LOG", "C-MED", "C-POOL") for u in UN]].reindex([t for t in TESTS if t != "SEALED"])
    md(x)

    chk = {d: json.loads((OUT / f"P2_checks_{d}.json").read_text()) for d in designs}
    h("### checks")
    lines.append("```")
    lines.append(json.dumps(chk, indent=1))
    lines.append("```")
    (OUT / "P2_report_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

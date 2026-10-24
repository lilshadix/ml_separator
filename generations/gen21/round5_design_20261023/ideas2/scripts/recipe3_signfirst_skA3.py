#!/usr/bin/env python
"""Skeptic A for recipe-3, part 3 (EXPLORATORY).  Reads only recipe3_skA_out/skA_rows.parquet.
(a) outside the dominant cpg cluster: COPY_own (mean of siblings) vs always-heavier, D7@2/@4, SH_own@2/@4,
    covered and uncovered rows; (b) cells whose every pair is covered (an ICP-MS run could be skipped);
(c) power of the stated kill rule: with 11 families drawn from the observed family accuracies, how often is the
    family-level one-sided 95% lower bound >= 0.90 (i.e. how often would the idea survive if the truth were
    the corpus)?"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\recipe3_skA_out")
log = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


def acc(p, y):
    return float((np.sign(p) == np.sign(y)).mean()) if len(y) else float("nan")


def main() -> int:
    R = pd.read_parquet(OUT / "skA_rows.parquet")
    o = R[(R.n_own >= 1) & ~R.c5].copy()
    o["big"], o["sel"] = o.y.abs() >= 0.1, o.copy_own.abs() >= 0.1
    dom = "g:pub_74428aeab2"
    for lab, z in (("dominant cluster", o[o.cluster == dom]), ("outside dominant", o[o.cluster != dom])):
        for rl, m in (("covered |y|>=.1", z.big & z.sel), ("all covered", z.sel), ("uncovered |y|>=.1", z.big & ~z.sel)):
            q = z[m]
            P(f"[{lab}] {rl}: n {len(q)} pubs {q.pub.nunique()} cells {q.cell_id.nunique()} | COPY {acc(q.copy_own, q.y):.3f} "
              f"AH {(q.y < 0).mean():.3f} D7@2 {acc(q.d7_2, q.y):.3f} D7@4 {acc(q.d7_4, q.y):.3f} SH@2 {acc(q.sh_2, q.y):.3f} "
              f"SH@4 {acc(q.sh_4, q.y):.3f}")
        P(f"[{lab}] coverage among |y|>=.1: {(z.sel & z.big).sum() / z.big.sum():.3f}; all rows {z.sel.mean():.3f}")
    # (b) cells fully covered
    g = o.groupby("cell_id").agg(allcov=("sel", "all"), pub=("pub", "first"), n=("sel", "size"))
    P(f"\ncells with every strict-adjacent pair covered (|COPY|>=0.1): {int(g.allcov.sum())} of {len(g)} "
      f"({g.allcov.mean():.3f}); pubs {g[g.allcov].pub.nunique()}")
    gg = o[o.cell_id.isin(g.index[g.allcov])]
    P(f"  on those cells, all-row sign acc {acc(gg.copy_own, gg.y):.3f}, n {len(gg)}; any wrong call in cell: "
      f"{gg.assign(w=np.sign(gg.copy_own) != np.sign(gg.y)).groupby('cell_id').w.any().mean():.3f}")
    # (c) power of the kill rule
    z = o[o.sel].assign(fam=lambda d: d.pub + "|" + d.ext, ok=lambda d: (np.sign(d.copy_own) == np.sign(d.y)).astype(float))
    fam = z.groupby("fam").ok.agg(["sum", "count"])
    rng = np.random.default_rng(3)
    surv_pool, surv_t, surv_point = 0, 0, 0
    B = 2000
    for _ in range(B):
        idx = rng.integers(0, len(fam), 11)
        s, c = fam["sum"].to_numpy()[idx], fam["count"].to_numpy()[idx]
        # family bootstrap lower bound of pooled acc
        W = rng.multinomial(11, np.ones(11) / 11, 500)
        lb = np.percentile((W @ s) / np.maximum(W @ c, 1), 5)
        surv_pool += lb >= 0.90
        a = s / c
        tl = a.mean() - 1.812 * a.std(ddof=1) / np.sqrt(11)
        surv_t += tl >= 0.90
        surv_point += s.sum() / c.sum() >= 0.90
    P(f"\nkill-rule power (11 families resampled from the corpus's {len(fam)} families, all covered rows):")
    P(f"  P(pooled point >= 0.90) {surv_point / B:.3f}; P(family-bootstrap 1-sided 95% LB >= 0.90) {surv_pool / B:.3f}; "
      f"P(equal-weight t LB >= 0.90) {surv_t / B:.3f}")
    # same on covered |y|>=.1 rows vs 0.95 (hypothesis level)
    z2 = z[z.big]
    fam2 = z2.groupby("fam").ok.agg(["sum", "count"])
    sp = 0
    for _ in range(B):
        idx = rng.integers(0, len(fam2), 11)
        s, c = fam2["sum"].to_numpy()[idx], fam2["count"].to_numpy()[idx]
        W = rng.multinomial(11, np.ones(11) / 11, 500)
        sp += np.percentile((W @ s) / np.maximum(W @ c, 1), 5) >= 0.90
    P(f"  covered |y|>=.1 rows: P(family-bootstrap LB >= 0.90) {sp / B:.3f}")
    (OUT / "skA3_log.txt").write_text("\n".join(log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

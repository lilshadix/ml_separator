#!/usr/bin/env python
"""Skeptic A for idea shift-1, follow-up (EXPLORATORY, 229 already-seen C3 cells, 5 dev seeds).
Is the acid-distance gradient a distance effect or an extrapolation-direction effect?  Position (inside / above /
below the acid-only siblings' range) x distance, stratified permutation, far-side contrast without below-range
cells, and the pub_19783c039b near-copy pair.  Rebuilds rows with shift1_skA's own sibling code (imported).

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift1_skA2.py
Writes only ideas2/scripts/shift1_skA_out/shift1_skA2.log.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sk", HERE / "shift1_skA.py")
sk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sk)
from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES  # noqa: E402

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def main():
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f["cell_id"].astype(str).to_numpy()
    pos = {c: i for i, c in enumerate(ids)}
    x = np.log10(f["cond__acid_concentration_M"].astype(float).to_numpy())
    S = pd.read_parquet(sk.SHIFT_ROWS)
    S = S[(S.n_AS >= 1)].copy()
    # positions recomputed from the bench (own code verified dx_mean / ACOPY_0 identical in shift1_skA)
    S["adx"] = S.dx_mean.abs()
    ex = S.extrap.astype(bool).to_numpy()
    S["pos"] = np.where(~ex, "inside", np.where(S.dx_mean < 0, "below", "above"))
    d = S[(S.design == "BPG") & ~S.c5].copy()
    d["near"] = d.adx <= 0.30 + 1e-9
    say("[1] position x factor-of-2 split, BPG no C5 (cells; mean row MSE; R2 pooled within seed)")
    rows = []
    for (p_, nr), g in d.groupby(["pos", "near"]):
        rows.append({"pos": p_, "near": nr, "cells": g.cell_id.nunique(), "pubs": g.publication.nunique(),
                     "MSE_copy0": ((g.y - g.ACOPY_0) ** 2).mean(), "MSE_B1@4": ((g.y - g.B1_3) ** 2).mean(),
                     "MSE_SH@4": ((g.y - g.SH_3) ** 2).mean(), "R2_copy0": sk.r2_seedmean(g, "ACOPY_0"),
                     "R2_B1@4": sk.r2_seedmean(g, "B1_3")})
    say(pd.DataFrame(rows).round(4).to_string(index=False))

    # cell table
    e = d.assign(se0=(d.y - d.ACOPY_0) ** 2, seB=(d.y - d.B1_3) ** 2).groupby("cell_id").agg(
        pub=("publication", "first"), adx=("adx", "first"), pos=("pos", "first"), se0=("se0", "mean"),
        seB=("seB", "mean"), x_t=("x_t", "first"))
    e["below"] = (e.pos == "below").astype(float)
    e["ratio"] = np.log(e.se0 / e.seB)
    # distance within position strata: unit = pub x position
    e["unit"] = e.pub + "|" + e.pos
    ee = e.groupby("unit").filter(lambda g: len(g) >= 2 and g.adx.nunique() >= 2)
    for col in ("se0", "seB", "ratio"):
        o, p = sk.perm_p(ee, "adx", col, unit="unit", n=5000)
        say(f"[2] |dx| vs cell {col}, permuted within publication x position (inside/above/below): stat {o:+.3f}, "
            f"p {p:.3f} ({ee.unit.nunique()} units, {len(ee)} cells)")
    eb = e.groupby("pub").filter(lambda g: len(g) >= 3 and g.below.nunique() >= 2)
    for col in ("se0", "seB", "ratio"):
        o, p = sk.perm_p(eb, "below", col, unit="pub", n=5000)
        say(f"[3] below-range flag vs cell {col}, within publication: stat {o:+.3f}, p {p:.3f} "
            f"({eb.pub.nunique()} pubs, {len(eb)} cells)")
    # without below-range cells
    nb = e[e.pos != "below"].groupby("pub").filter(lambda g: len(g) >= 3 and g.adx.nunique() >= 2)
    for col in ("se0", "seB", "ratio"):
        o, p = sk.perm_p(nb, "adx", col, unit="pub", n=5000)
        say(f"[4] |dx| vs cell {col} without below-range targets: stat {o:+.3f}, p {p:.3f} "
            f"({nb.pub.nunique()} pubs, {len(nb)} cells)")

    # far side contrast without below-range cells
    for nm, sub in (("far, all", d[~d.near]), ("far, not below", d[~d.near & (d.pos != "below")]),
                    ("far, below only", d[~d.near & (d.pos == "below")]), ("near, all", d[d.near]),
                    ("near, not below", d[d.near & (d.pos != "below")])):
        u = sub.assign(dse=(sub.y - sub.B1_3) ** 2 - (sub.y - sub.ACOPY_0) ** 2).groupby("publication").dse.sum()
        say(f"[5] {nm:16s}: cells {sub.cell_id.nunique()}, pubs {sub.publication.nunique()}; copy0 "
            f"{sk.r2_seedmean(sub, 'ACOPY_0'):.3f} vs B1@4 {sk.r2_seedmean(sub, 'B1_3'):.3f} (d {sk.r2_seedmean(sub, 'ACOPY_0') - sk.r2_seedmean(sub, 'B1_3'):+.3f}); "
            f"pubs where copy0 better {int((u > 0).sum())}/{len(u)}")

    # 19783c039b near-copy pair
    say("\n[6] pub_19783c039b acid-only family: cell, [acid] M, # metals bit-identical to 44012938119033a5, cell MSE copy0 / B1@4")
    fam = sorted({c for c in d[d.publication == "pub_19783c039b"].cell_id})
    ref = "44012938119033a5"
    for c in sorted(fam, key=lambda c: x[pos[c]]):
        i, j = pos[c], pos[ref]
        m = np.isfinite(Y[i]) & np.isfinite(Y[j])
        g = d[d.cell_id == c]
        say(f"   {c} acid {10 ** x[i]:.3f} M  n_ident_with_{ref[:6]} {int((np.abs(Y[i, m] - Y[j, m]) < 1e-9).sum())}/{int(m.sum())} "
            f"ext {f.loc[i, 'extractant'][:30]}  pos {g.pos.iloc[0]} |dx| {g.adx.iloc[0]:.3f}  MSE copy0 "
            f"{((g.y - g.ACOPY_0) ** 2).mean():.4f} B1@4 {((g.y - g.B1_3) ** 2).mean():.4f}")
    Sb = S[(S.design == "BPG") & (S.publication == "pub_19783c039b")]
    for c in sorted(set(Sb.cell_id), key=lambda c: x[pos[c]]):
        i = pos[c]
        say(f"   {c} acid {10 ** x[i]:.3f}: L {f.loc[i, 'cond__extractant_concentration_M']} T {f.loc[i, 'cond__temperature_C']} "
            f"t {f.loc[i, 'cond__contact_time_min']} metal {f.loc[i, 'cond__metal_concentration_mM']} n_AS {Sb[Sb.cell_id == c].n_AS.iloc[0]}")
    i, j = pos["5169c377777d282b"], pos[ref]
    say("   5169c377777d282b log D: " + " ".join(f"{v:.3f}" for v in Y[i]))
    say("   44012938119033a5 log D: " + " ".join(f"{v:.3f}" for v in Y[j]))
    say("   cond diff: " + ", ".join(f"{c}={f.loc[i, c]} vs {f.loc[j, c]}" for c in f.columns if c.startswith("cond__")
                                       and not (pd.isna(f.loc[i, c]) and pd.isna(f.loc[j, c])) and f.loc[i, c] != f.loc[j, c]))
    say(f"   in V2 lists: {any(c in set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS) for c in ('5169c377777d282b', ref))}")
    # c8195b4ca7 3-of-6 identical pair
    for a, b in (("d4c1e23a091d0960", "e4ee526bda3920c2"),):
        i, j = pos[a], pos[b]
        say(f"   {a} acid {10 ** x[i]:.3f} M: " + " ".join(f"{v:.3f}" for v in Y[i]))
        say(f"   {b} acid {10 ** x[j]:.3f} M: " + " ".join(f"{v:.3f}" for v in Y[j]))
    # drop the pub_19783c039b near-copy pair as targets AND siblings is not possible without a rebuild; report
    # the statistic without the two targets
    e2 = e[~e.index.isin(["5169c377777d282b", "44012938119033a5"])].groupby("pub").filter(
        lambda g: len(g) >= 3 and g.adx.nunique() >= 2)
    o, p = sk.perm_p(e2, "adx", "se0", n=5000)
    say(f"[7] within-pub stat without the two near-copy targets: {o:+.3f} (p {p:.3f})")
    e3 = e[e.pub != "pub_19783c039b"].groupby("pub").filter(lambda g: len(g) >= 3 and g.adx.nunique() >= 2)
    for col in ("se0", "seB", "ratio"):
        o, p = sk.perm_p(e3, "adx", col, n=5000)
        say(f"[8] without pub_19783c039b: |dx| vs {col}: {o:+.3f} (p {p:.3f}; {e3.pub.nunique()} pubs, {len(e3)} cells)")
    (sk.OUT / "shift1_skA2.log").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()

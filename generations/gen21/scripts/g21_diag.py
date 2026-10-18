#!/usr/bin/env python
"""Error analyses that motivated attempts 2 and 3, and the attempt-3 smoke look, from saved
predictions only (no refit except the smoke fold).  Output: results/diag/diag.txt

    .venv/Scripts/python.exe -W ignore generations/gen21/scripts/g21_diag.py
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from g21 import paths  # noqa: E402


def sse_shares(t, arm):
    e = (t[arm] - t.y) ** 2
    q = pd.qcut(t.y.abs(), 5)
    top_q = float(e[q == q.cat.categories[-1]].sum() / e.sum())
    ext = (t.assign(e=e).groupby("extractant").e.sum() / e.sum()).sort_values(ascending=False)
    grp = (t.assign(e=e).groupby(["extractant", "publication"]).agg(e=("e", "sum"), cells=("cell_id", "nunique")))
    grp["share"] = grp.e / e.sum()
    return top_q, q.cat.categories[-1], ext, grp.sort_values("share", ascending=False)


def shared_within_series(t, arm):
    t = t.assign(err=t[arm] - t.y, grp=t.extractant + "|" + t.publication)
    key = ["split_seed", "grp", "A", "B"]
    gm = t.groupby(key).err.transform("mean")
    cnt = t.groupby(key).err.transform("size")
    s = t[cnt >= 2]
    return float(len(s) / t.split_seed.nunique()), float(1 - ((s.err - gm[cnt >= 2]) ** 2).sum() / (s.err ** 2).sum())


def main() -> int:
    out = io.StringIO()
    with redirect_stdout(out):
        t1 = pd.read_parquet(paths.PREDICTIONS / "attempt1_BP.parquet")
        t1 = t1[(t1.k_max >= 3) & (t1.k == 3)]
        top, cut, ext, grp = sse_shares(t1, "RIDGE2+OBS@RESEL")
        print("== after attempt 1 (RIDGE2+OBS@RESEL, BP, C3, k=3)")
        print(f"share of squared error in the top |y| quintile {cut}: {top:.3f}")
        print("top two extractants' share:", round(float(ext.iloc[:2].sum()), 3), ext.iloc[:2].round(3).to_dict())
        t2 = pd.read_parquet(paths.PREDICTIONS / "attempt2_BP.parquet")
        t2 = t2[(t2.k_max >= 3) & (t2.k == 3)]
        top, cut, ext, grp = sse_shares(t2, "HB2+GP")
        n, sh = shared_within_series(t2, "HB2+GP")
        print("\n== after attempt 2 (HB2+GP, BP, C3, k=3)")
        print(f"rows per seed in series x pair cells with >= 2 held-out cells: {n:.0f}; share of their squared error "
              f"explained by the series x pair mean: {sh:.3f}")
        g0 = grp.iloc[0]
        print(f"largest series: {grp.index[0][0][:40]}... ({int(g0.cells)} cells), share of all squared error {g0.share:.3f}")
    # attempt-3 smoke look (one dev fold, DOPT_Q pooling) -- what was seen before the staggered policy
    from g21 import attempts as AT, cache as C, evaluate as E
    from gen20k import data as D
    bench = D.load_bench()
    spec = AT.get("3d", bench)
    rec = C.load("BP", 130363, 2)
    blk = dict(rec["outer"])
    blk["fitted"] = spec.fit(bench, rec)
    r = pd.DataFrame(E.block_rows(bench, rec, blk, spec.methods, policy="DOPT_Q", ks=(3,)))
    r = r[r.k_max >= 3]
    with redirect_stdout(out):
        print("\n== attempt-3 smoke look: BP seed 130363 fold 2, DOPT_Q reads, k=3, C3 rows of that fold"
              f" ({r.cell_id.nunique()} cells, {len(r)} pairs)")
        for arm in ("RIDGE2", "HB2+GP", "HBG@OWN", "HBG@POOL", "RIDGE2@POOL"):
            y, p = r.y.to_numpy(), r[arm].to_numpy()
            print(f"  {arm:12s} R2 {1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}  MAE {np.abs(y - p).mean():.4f}")
    d = paths.RESULTS / "diag"
    d.mkdir(parents=True, exist_ok=True)
    (d / "diag.txt").write_text(out.getvalue(), encoding="utf-8")
    print(out.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

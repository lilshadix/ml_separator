#!/usr/bin/env python
"""Round-2 diagnosis of the round-1 candidate HB2+GP (attempt 2, DOPT_Q reads, k = 3, C3, BP).
Saved predictions and fitted parameters only; output results/diag/diag_r2.txt.

    .venv/Scripts/python.exe -W ignore generations/gen21/scripts/g21_diag_r2.py

A  error by read status and by Z-distance of the query to the nearest read metal
B  error by chemotype: share of squared error that is a chemotype x pair-type mean, against a
   label-permutation null (how much a chemistry-specific prior could remove at most)
C  read consistency: leave-one-read-out standardized residuals of the reads under the fitted HB
   model; error of cells that contain an inconsistent read (what a robust model could address)
D  does the structure model know something HB2+GP does not: regression of the HB2+GP residual
   on the zero-shot G14 disagreement (G14@ZERO - HB2+GP), per pair
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from g21 import methods as M  # noqa: E402
from g21 import paths  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import IDX, Z_VEC  # noqa: E402


def hb_from_row(bench, row, name="HB2+GP"):
    m = M.HBModel(M.hb_basis(bench, "B2"), use_gp=True)
    m.m = np.array(json.loads(row[f"{name}.m"]))
    m.S = np.array(json.loads(row[f"{name}.S_theta"]))
    m.tau2, m.rho, m.sig2 = float(row[f"{name}.tau2"]), float(row[f"{name}.rho"]), float(row[f"{name}.sig2"])
    return m


def main() -> int:
    bench = D.load_bench()
    t = pd.read_parquet(paths.PREDICTIONS / "attempt2_BP.parquet")
    t = t[(t.k_max >= 3) & (t.k == 3)].copy()
    t["e"] = t["HB2+GP"] - t.y
    fits = pd.read_csv(paths.RESULTS / "attempt2" / "fitted_BP.csv")
    out = io.StringIO()
    rng = np.random.default_rng(7)
    with redirect_stdout(out):
        sst = ((t.y - t.groupby("split_seed").y.transform("mean")) ** 2).sum()
        print(f"HB2+GP k=3 C3 BP: rows {len(t)}, SSE {float((t.e ** 2).sum()):.2f}, SST {float(sst):.2f}")
        # A
        t["cls"] = np.where(t.a_read | t.b_read, "one_end_read", "none_read")
        print("\nA. by read status\n", t.groupby("cls").agg(n=("e", "size"), mse=("e", lambda v: float((v ** 2).mean())),
                                                         ysd=("y", "std")).round(4).to_string())
    # distance to nearest read metal needs the read sets: rebuild from the cache
    dist, cons = [], []
    for (seed, fold), g in t.groupby(["split_seed", "fold"]):
        rec = C.load("BP", int(seed), int(fold))
        row = fits[(fits.split_seed == seed) & (fits.fold == fold)].iloc[0]
        hb = hb_from_row(bench, row)
        Cfull = hb.cov_full(hb.S, hb.tau2, hb.rho, hb.sig2)
        mu = hb.B.T @ hb.m
        for cell in rec["outer"]["cells"]:
            if cell["k_max"] < 3:
                continue
            o = E.observe(bench.Y[cell["ci"]], cell, cell["seqs"]["DOPT_Q"], 3)
            read = list(o.read)
            # leave-one-read-out: predict each read metal's level difference to the others' mean
            z = []
            for r in read:
                rest = [x for x in read if x != r]
                Dm = np.zeros((len(rest) - 1, 14))
                for i, x in enumerate(rest[1:]):
                    Dm[i, rest[0]], Dm[i, x] = 1, -1
                obs = np.array([o.y_masked[rest[0]] - o.y_masked[x] for x in rest[1:]])
                d = np.zeros(14); d[rest[0]], d[r] = 1, -1          # target: y(rest0) - y(r)
                S = Dm @ Cfull @ Dm.T
                w = np.linalg.solve(S, obs - Dm @ mu)
                pm = d @ mu + d @ Cfull @ Dm.T @ w
                pv = d @ Cfull @ d - d @ Cfull @ Dm.T @ np.linalg.solve(S, Dm @ Cfull @ d)
                z.append(abs((o.y_masked[rest[0]] - o.y_masked[r]) - pm) / np.sqrt(max(pv, 1e-12)))
            cons.append({"split_seed": seed, "cell_id": cell["cell_id"], "max_z": float(max(z))})
            for a, b in cell["query"]:
                dist.append({"split_seed": seed, "cell_id": cell["cell_id"], "A": D.LANTHANIDES[a], "B": D.LANTHANIDES[b],
                             "dmin": int(min(min(abs(Z_VEC[a] - Z_VEC[r]), abs(Z_VEC[b] - Z_VEC[r])) for r in read))})
    t = t.merge(pd.DataFrame(dist), on=["split_seed", "cell_id", "A", "B"]).merge(pd.DataFrame(cons), on=["split_seed", "cell_id"])
    with redirect_stdout(out):
        print("\nA2. by Z-distance from the query pair to the nearest read metal (0 = an endpoint is read)\n",
              t.groupby("dmin").agg(n=("e", "size"), mse=("e", lambda v: float((v ** 2).mean())), ysd=("y", "std")).round(4).to_string())
        # B: chemotype x pair mean share, against permuted chemotype labels within seed
        def share(df, key):
            gm = df.groupby(["split_seed", key, "A", "B"]).e.transform("mean")
            return float(1 - ((df.e - gm) ** 2).sum() / (df.e ** 2).sum())
        obs_share = share(t, "chemotype")
        null = []
        for _ in range(200):
            tp = t.copy()
            cells = tp[["split_seed", "cell_id", "chemotype"]].drop_duplicates()
            cells["perm"] = cells.groupby("split_seed").chemotype.transform(lambda s: rng.permutation(s.to_numpy()))
            tp = tp.merge(cells[["split_seed", "cell_id", "perm"]], on=["split_seed", "cell_id"])
            null.append(share(tp, "perm"))
        print(f"\nB. share of squared error that is a chemotype x pair mean: {obs_share:.3f} "
              f"(permutation null mean {np.mean(null):.3f}, 95th pct {np.percentile(null, 95):.3f})")
        mean_e = t.groupby("chemotype").agg(n=("e", "size"), mean_err=("e", "mean"), mse=("e", lambda v: float((v ** 2).mean())))
        mean_e["sse_share"] = mean_e.n * mean_e.mse / float((t.e ** 2).sum())
        print(mean_e.sort_values("sse_share", ascending=False).head(6).round(4).to_string())
        # C: inconsistent reads
        t["incons"] = t.max_z > 3
        print("\nC. cells with a read whose leave-one-out |z| > 3:",
              f"{t.drop_duplicates(['split_seed','cell_id']).incons.mean():.3f} of cells;")
        print(t.groupby("incons").agg(n=("e", "size"), mse=("e", lambda v: float((v ** 2).mean())),
                                      sse_share=("e", lambda v: float((v ** 2).sum() / (t.e ** 2).sum()))).round(4).to_string())
        print(t.groupby(pd.cut(t.max_z, [0, 1, 2, 3, 5, 1e9])).agg(n=("e", "size"), mse=("e", lambda v: float((v ** 2).mean()))).round(4).to_string())
        # D: residual vs structure-model disagreement
        x = (t["G14@ZERO"] - t["HB2+GP"]).to_numpy()
        r = (t.y - t["HB2+GP"]).to_numpy()
        b = float(x @ r / (x @ x))
        print(f"\nD. regression of (y - HB2+GP) on (G14@ZERO - HB2+GP): slope {b:.3f}, corr {np.corrcoef(x, r)[0, 1]:.3f}, "
              f"in-sample SSE reduction {1 - ((r - b * x) ** 2).sum() / (r ** 2).sum():.3f}")
        for (lo, hi) in ((0, 0.1), (0.1, 0.3), (0.3, 9)):
            s = (np.abs(x) >= lo) & (np.abs(x) < hi)
            print(f"   |disagreement| in [{lo}, {hi}): n {s.sum()}, corr {np.corrcoef(x[s], r[s])[0, 1]:.3f}")
    d = paths.RESULTS / "diag"
    d.mkdir(parents=True, exist_ok=True)
    (d / "diag_r2.txt").write_text(out.getvalue(), encoding="utf-8")
    print(out.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

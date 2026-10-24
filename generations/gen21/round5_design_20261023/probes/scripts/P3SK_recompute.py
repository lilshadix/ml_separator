#!/usr/bin/env python
"""SKEPTIC CHECK of probe P3 (k = 4 rung).  EXPLORATORY: already-seen 229 C3 cells, 5 dev seeds.

Independent recomputation of P3's decision-relevant numbers WITHOUT reading P3's rows.parquet / summariser:
  * k = 3 rows and read sets from round 4's REF_{BPG,BP}.parquet; k = 4 DOPT_Q-nested read sets from round 3's
    stored R3-B0_{d}.parquet (k = 4 rows), not from the cache sequence;
  * own PWL_R + Gd-potential level code (np.interp in -CN8 radius, constant beyond the extreme reads);
  * own MAXMIN_R and BISECT_R picks written from the P3 report text; own RANDOM expectation;
  * beta: stored k = 3 REF beta (round-4 results/REF/fitted_*.json) and a fresh k = 4 refit via r4.refs.fit_reference;
  * pooled per-seed R2 for C3 population row, C4 fixed cohort, with / without pub_6a0ef8528c, cluster bootstrap,
    leave-one-publication-out and leave-one-cluster-out.
Writes only P3SK_out/ next to this script.  Run from D:\\ml_separator_gh with .venv\\Scripts\\python.exe -W ignore.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

R4 = Path(r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, str(R4))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import refs as RF  # noqa: E402
from r4 import metrics as M4  # noqa: E402
from r3 import interp  # noqa: E402
from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k import design as DS  # noqa: E402
from gen20k import inference as INF  # noqa: E402
from gen20k.data import LANTHANIDES, Z_VEC  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402

OUT = Path(__file__).resolve().parent / "P3SK_out"
OUT.mkdir(exist_ok=True)
XR = np.asarray(interp._coords("R"), float)
ZV = np.asarray(Z_VEC, int)
GD = LANTHANIDES.index("Gd")
IDX = {n: i for i, n in enumerate(LANTHANIDES)}
C5 = "pub_6a0ef8528c"
REFIT = "--no-refit" not in sys.argv


def levels(yrow, metals, read, beta):
    """level(m) = I_S[y_S - u_S](m) + u(m), u = beta at Gd only (own implementation)."""
    rd = sorted(read, key=lambda m: XR[m])
    xs = XR[rd]
    vs = np.array([yrow[m] - (beta if m == GD else 0.0) for m in rd])
    out = np.full(len(LANTHANIDES), np.nan)
    for m in metals:
        if m in read:
            out[m] = yrow[m]
        else:
            out[m] = float(np.interp(XR[m], xs, vs)) + (beta if m == GD else 0.0)
    return out


def my_maxmin(read, legal):
    best, bkey = None, None
    for m in legal:
        d = min(abs(XR[m] - XR[s]) for s in read)
        key = (d, -ZV[m])
        if bkey is None or key > bkey:
            best, bkey = m, key
    return best


def my_bisect(metals, read, legal):
    """From the report text: segment (gaps between consecutive reads + two end segments) with the most unread
    measured metals that contains a legal metal, ties -> lighter segment; inside it the legal metal nearest the
    target (inner: radius midpoint of the bracketing reads; end: the cell's extreme metal), ties -> lower Z."""
    ms = sorted(metals, key=lambda m: ZV[m])
    rd = sorted(read, key=lambda m: ZV[m])
    segs = [([m for m in ms if ZV[m] < ZV[rd[0]]], XR[ms[0]])]
    for a, b in zip(rd, rd[1:]):
        segs.append(([m for m in ms if ZV[a] < ZV[m] < ZV[b]], 0.5 * (XR[a] + XR[b])))
    segs.append(([m for m in ms if ZV[m] > ZV[rd[-1]]], XR[ms[-1]]))
    order = sorted(range(len(segs)), key=lambda i: (-len(segs[i][0]), i))
    for i in order:
        cand = [m for m in segs[i][0] if m in legal]
        if cand:
            t = segs[i][1]
            return min(cand, key=lambda m: (abs(XR[m] - t), ZV[m]))
    raise AssertionError


def main():
    t0 = time.time()
    bench = D.load_bench()
    Y = bench.Y
    cid2ci = {c: i for i, c in enumerate(bench.frame.cell_id)}
    out_rows = []
    checks = {"k3_B1_maxdiff": 0.0, "k3_B0_maxdiff": 0.0, "k3_y_maxdiff": 0.0, "k4_B0_vs_R3B0_maxdiff": 0.0,
              "read3_vs_cache_mismatch": 0, "dopt5_vs_cache_mismatch": 0, "dopt5_not_legal": 0, "both_read": 0,
              "maxmin_vs_P3": None, "bisect_vs_P3": None, "beta4_vs_P3_maxdiff": None}
    betas = []
    for design in ("BPG", "BP"):
        ref = pd.read_parquet(R4 / "predictions" / f"REF_{design}.parquet")
        r3 = pd.read_parquet(R4 / "predictions" / f"R3-B0_{design}.parquet")
        r3 = r3[(r3.k == 4) & (r3.k_max >= 4)]
        r3rs = r3.groupby(["split_seed", "cell_id"]).read_set.first().to_dict()
        r3pw = r3.set_index(["split_seed", "cell_id", "A", "B"]).PWL_R.to_dict()
        fitted = json.loads((paths.RES / "REF" / f"fitted_{design}.json").read_text())
        refg = {k: g for k, g in ref.groupby(["split_seed", "cell_id"])}
        for f in folds_for(bench.frame, design, seeds=list(paths.DEV_SEEDS)):
            rec = C.load(f.design, f.seed, f.fold)
            b3 = float(fitted[f"{f.seed}_{f.fold}"]["ref:B1"]["coef"][0])
            if REFIT:
                fit4 = RF.fit_reference(bench, rec, Y, "B1", k=4)
                b4 = float(fit4["coef"][0])
                lam4 = str(fit4["lambda"])
            else:
                b4, lam4 = np.nan, ""
            betas.append({"design": design, "split_seed": int(f.seed), "fold": int(f.fold), "beta3": b3, "beta4": b4, "lam4": lam4})
            for cell in rec["outer"]["cells"]:
                if cell["k_max"] < 3:
                    continue
                cid = cell["cell_id"]
                g = refg[(int(f.seed), cid)]
                assert (g.fold == f.fold).all()
                ci = cid2ci[cid]
                assert ci == cell["ci"]
                yrow = Y[ci]
                metals = list(cell["metals"])
                assert set(metals) == set(np.where(np.isfinite(yrow))[0].tolist()), cid
                read3 = frozenset(IDX[n] for n in g.read_set.iloc[0].split(","))
                cache3 = frozenset(m for p in cell["seqs"]["DOPT_Q"][:3] for m in p)
                checks["read3_vs_cache_mismatch"] += int(read3 != cache3)
                L3b0 = levels(yrow, metals, read3, 0.0)
                L3b1 = levels(yrow, metals, read3, b3)
                legal, dopt5, mm, bs = [], None, None, None
                if cell["k_max"] >= 4:
                    dz = DS.make_design(E._view(cell), "adjacent", "metal")
                    legal = list(DS.legal_metals(dz, read3))
                    read4r3 = frozenset(IDX[n] for n in r3rs[(int(f.seed), cid)].split(","))
                    d5 = read4r3 - read3
                    assert len(d5) == 1 and read3 <= read4r3
                    dopt5 = next(iter(d5))
                    cache5 = set(m for p in cell["seqs"]["DOPT_Q"][:4] for m in p) - read3
                    checks["dopt5_vs_cache_mismatch"] += int(cache5 != {dopt5})
                    checks["dopt5_not_legal"] += int(dopt5 not in legal)
                    mm = my_maxmin(read3, legal)
                    bs = my_bisect(metals, read3, legal)
                L4 = {m: levels(yrow, metals, read3 | {m}, b4) for m in legal}
                L4b3 = levels(yrow, metals, read3 | {dopt5}, b3) if dopt5 is not None else None
                L4b0 = levels(yrow, metals, read3 | {dopt5}, 0.0) if dopt5 is not None else None
                for r in g.itertuples(index=False):
                    a, b = IDX[r.A], IDX[r.B]
                    y = float(yrow[a] - yrow[b])
                    checks["k3_y_maxdiff"] = max(checks["k3_y_maxdiff"], abs(y - r.y))
                    p3b0 = L3b0[a] - L3b0[b]
                    p3b1 = L3b1[a] - L3b1[b]
                    checks["k3_B0_maxdiff"] = max(checks["k3_B0_maxdiff"], abs(p3b0 - r.B0))
                    checks["k3_B1_maxdiff"] = max(checks["k3_B1_maxdiff"], abs(p3b1 - r.B1))
                    rowd = {"design": design, "split_seed": int(f.seed), "fold": int(f.fold), "cell_id": cid,
                            "publication": r.publication, "chemotype": r.chemotype, "k_max": int(cell["k_max"]),
                            "n_metals": len(metals), "A": r.A, "B": r.B, "y": y, "se_k3": (p3b1 - y) ** 2,
                            "se_k3_B0": (p3b0 - y) ** 2}
                    if legal:
                        for m in legal:
                            checks["both_read"] += int(a in read3 | {m} and b in read3 | {m})
                        se = {m: (L4[m][a] - L4[m][b] - y) ** 2 for m in legal}
                        rowd.update({"se_DOPTQ": se[dopt5], "se_MAXMIN_R": se[mm], "se_BISECT_R": se[bs],
                                     "se_RANDOM": float(np.mean(list(se.values()))),
                                     "se_DOPTQ_b3": (L4b3[a] - L4b3[b] - y) ** 2,
                                     "se_DOPTQ_B0": (L4b0[a] - L4b0[b] - y) ** 2,
                                     "pick_DOPTQ": LANTHANIDES[dopt5], "pick_MAXMIN_R": LANTHANIDES[mm],
                                     "pick_BISECT_R": LANTHANIDES[bs], "n_legal": len(legal)})
                        pw = r3pw[(int(f.seed), cid, r.A, r.B)]
                        checks["k4_B0_vs_R3B0_maxdiff"] = max(checks["k4_B0_vs_R3B0_maxdiff"], abs((L4b0[a] - L4b0[b]) - pw))
                    else:
                        for c in ("se_DOPTQ", "se_MAXMIN_R", "se_BISECT_R", "se_RANDOM", "se_DOPTQ_b3", "se_DOPTQ_B0"):
                            rowd[c] = rowd["se_k3"] if c != "se_DOPTQ_B0" else rowd["se_k3_B0"]
                    out_rows.append(rowd)
            print(f"{design} {f.seed} f{f.fold}: b3 {b3:+.4f} b4 {b4:+.4f} ({lam4}) {time.time() - t0:.0f}s", flush=True)
    T = pd.DataFrame(out_rows)
    T.to_parquet(OUT / "rows.parquet", index=False)
    pd.DataFrame(betas).to_csv(OUT / "betas.csv", index=False)
    # compare with P3's own picks / betas (read only)
    p3c = pd.read_parquet(Path(__file__).resolve().parent / "P3_out" / "cells.parquet")
    mine = T[T.n_legal.notna()].groupby(["design", "split_seed", "cell_id"])[["pick_DOPTQ", "pick_MAXMIN_R", "pick_BISECT_R"]].first().reset_index()
    mg = mine.merge(p3c, on=["design", "split_seed", "cell_id"], suffixes=("", "_p3"))
    checks["n_cell_instances_compared"] = len(mg)
    checks["dopt_vs_P3"] = float((mg.pick_DOPTQ == mg.pick_DOPTQ_p3).mean())
    checks["maxmin_vs_P3"] = float((mg.pick_MAXMIN_R == mg.pick_MAXMIN_R_p3).mean())
    checks["bisect_vs_P3"] = float((mg.pick_BISECT_R == mg.pick_BISECT_R_p3).mean())
    if REFIT:
        p3f = pd.DataFrame(json.loads((Path(__file__).resolve().parent / "P3_out" / "fits.json").read_text()))
        bm = pd.DataFrame(betas).merge(p3f, on=["design", "split_seed", "fold"])
        checks["beta4_vs_P3_maxdiff"] = float((bm.beta4 - bm.beta_k4).abs().max())
    print(json.dumps(checks, indent=1, default=float))
    (OUT / "checks.json").write_text(json.dumps(checks, indent=1, default=float), encoding="utf-8")
    print(f"EXIT 0 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""PROBE P3 (round-5 design; ideas H-BUDGET / M18 / M17) -- compute step.

EXPLORATORY: every number this produces is on the 229 already-seen C3 cells and the 5 dev seeds.

For every outer fold of BPG and BP (5 dev seeds x 5 folds, the round-4 fold cache) this script
  * refits B1 exactly as round 4's r4/refs.py ``fit_reference`` does, at k = 3 (identity check against the stored
    REF fit) and at k = 4 (same code, ``k=4``: training masks nested DOPT_Q / UNIFORM / 4 seeded RANDOM legal sets
    at budget 4, training cells with k_max >= 4, inner-fold lambda from the same grid incl. infinity);
  * for every held-out C3 cell (k_max >= 3) rebuilds the k = 3 DOPT_Q read set from the cache (4 metals) and
    predicts every strict-adjacent query with B0 (PWL_R), B1 (k = 3 fit) and HIST_D7 (fixed -0.072);
  * enumerates EVERY legal 5th metal (gen20k.design.legal_metals: measured, unread, no strict-adjacent pair with
    both ends read, maximum independent set still reachable) and predicts the same queries with B0, B1 (k = 4 fit),
    B1 with the k = 3 beta (sensitivity) and HIST_D7;
  * records the label-free 5th-read choice of each policy (DOPT_Q-nested = the cache's own 4th support step,
    MAXMIN_R / MAXMIN_Z, BISECT_R / BISECT_Z / BISECT_RMID, HULL_BISECT) per cell.

Outputs (probe-owned, under probes/scripts/P3_out/): rows.parquet (k = 3 rows and k = 4 rows per candidate),
cells.parquet (per-cell read sets, legal sets and policy picks), fits.json (beta / lambda per fold and k).
Run from D:\\ml_separator_gh with .venv\\Scripts\\python.exe -W ignore.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

R4 = Path(r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, str(R4))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import refs as RF  # noqa: E402
from r3 import interp  # noqa: E402
from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k import design as DS  # noqa: E402
from gen20k.data import LANTHANIDES, N_LN, Z_VEC  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402

OUT = Path(__file__).resolve().parent / "P3_out"
XR = np.asarray(interp._coords("R"), float)       # -CN8 radius coordinate of PWL_R
XZ = np.asarray(interp._coords("Z"), float)
ZV = np.asarray(Z_VEC, int)


def lv(Yrow, metals, read, coef):
    ym = np.full(N_LN, np.nan)
    rl = sorted(read)
    ym[rl] = Yrow[rl]
    f = {"spec": RF.SPECS["B1"], "kind": "R", "coef": np.array([float(coef)])}
    return RF.predict_levels(f, SimpleNamespace(y_masked=ym), metals)


# ----------------------------------------------------------------------------------------------
# label-free 5th-read policies (inputs: metal identities, the 4 read metals, the legal set; never a label)
# ----------------------------------------------------------------------------------------------
def maxmin(read, legal, x):
    """Farthest point: the legal metal whose distance to the nearest read metal (coordinate x) is largest;
    ties -> lower Z (gen20 UNIFORM convention)."""
    return max(legal, key=lambda m: (min(abs(x[m] - x[s]) for s in read), -ZV[m]))


def segments(metals, read):
    ms = sorted(metals, key=lambda m: ZV[m])
    rd = sorted(read, key=lambda m: ZV[m])
    segs = [("end_lo", [m for m in ms if ZV[m] < ZV[rd[0]]], None)]
    for a, b in zip(rd, rd[1:]):
        segs.append(("inner", [m for m in ms if ZV[a] < ZV[m] < ZV[b]], (a, b)))
    segs.append(("end_hi", [m for m in ms if ZV[m] > ZV[rd[-1]]], None))
    return ms, segs


def bisect(metals, read, legal, x, *, end_target="extreme", hull_first=False):
    """BISECT: the segment (gap between consecutive reads, or an end segment beyond the extreme read) with the
    most unread measured metals that contains a legal metal (ties -> the lighter segment); inside it, the legal
    metal nearest the target: interior gap -> coordinate midpoint of the two bracketing reads; end segment ->
    the cell's extreme metal (end_target='extreme', hull closure as BISECT's own first two reads) or the midpoint
    between the extreme read and the cell's extreme metal (end_target='mid').  hull_first: if an end segment
    holds a legal metal, read the legal metal nearest the cell's extreme first (the heavier end on ties)."""
    ms, segs = segments(metals, read)
    lg = set(legal)
    rd = sorted(read, key=lambda m: ZV[m])
    lo_ext, hi_ext = ms[0], ms[-1]

    def target(kind, seg, br):
        if kind == "inner":
            return 0.5 * (x[br[0]] + x[br[1]])
        ext = lo_ext if kind == "end_lo" else hi_ext
        edge = rd[0] if kind == "end_lo" else rd[-1]
        return x[ext] if end_target == "extreme" else 0.5 * (x[ext] + x[edge])

    if hull_first:
        for kind in ("end_hi", "end_lo"):
            seg = next(s for s in segs if s[0] == kind)
            cand = [m for m in seg[1] if m in lg]
            if cand:
                ext = hi_ext if kind == "end_hi" else lo_ext
                return min(cand, key=lambda m: (abs(x[m] - x[ext]), ZV[m]))
    order = sorted(range(len(segs)), key=lambda i: (-len(segs[i][1]), i))
    for i in order:
        kind, seg, br = segs[i]
        cand = [m for m in seg if m in lg]
        if cand:
            t = target(kind, seg, br)
            return min(cand, key=lambda m: (abs(x[m] - t), ZV[m]))
    raise AssertionError("no legal metal in any segment")


def outside_hull(a, b, read):
    """Strict-adjacent pair lies outside the read hull (constant continuation beyond the extreme reads)."""
    zr = [ZV[m] for m in read]
    lo, hi = min(zr), max(zr)
    return any((m not in read) and (ZV[m] < lo or ZV[m] > hi) for m in (a, b))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    bench = D.load_bench()
    Y = bench.Y
    stored = {d: json.loads((paths.RES / "REF" / f"fitted_{d}.json").read_text()) for d in ("BPG", "BP")}
    rows, cells, fits = [], [], []
    for design in ("BPG", "BP"):
        for f in folds_for(bench.frame, design, seeds=list(paths.DEV_SEEDS)):
            rec = C.load(f.design, f.seed, f.fold)
            E.assert_test_isolation(bench, rec)
            fit3 = RF.fit_reference(bench, rec, Y, "B1", k=3)
            fit4 = RF.fit_reference(bench, rec, Y, "B1", k=4)
            st = stored[design][f"{f.seed}_{f.fold}"]["ref:B1"]
            b3, b4 = float(fit3["coef"][0]), float(fit4["coef"][0])
            fits.append({"design": design, "split_seed": int(f.seed), "fold": int(f.fold),
                         "beta_k3": b3, "lambda_k3": str(fit3["lambda"]), "beta_k3_stored": float(st["coef"][0]),
                         "lambda_k3_stored": str(st["lambda"]), "beta_k4": b4, "lambda_k4": str(fit4["lambda"]),
                         "n_rows_k3": fit3["n_rows"], "n_rows_k4": fit4["n_rows"],
                         "cov_k3": fit3["coverage"], "cov_k4": fit4["coverage"]})
            for cell in rec["outer"]["cells"]:
                if cell["k_max"] < 3:
                    continue
                ci, metals = cell["ci"], cell["metals"]
                seq = cell["seqs"]["DOPT_Q"]
                read3 = frozenset(m for p in seq[:3] for m in p)
                assert len(read3) == 4
                dz = DS.make_design(E._view(cell), "adjacent", "metal")
                legal = sorted(DS.legal_metals(dz, read3), key=lambda m: ZV[m]) if cell["k_max"] >= 4 else []
                if cell["k_max"] >= 4:
                    assert legal, cell["cell_id"]
                    d5 = sorted(set(m for p in seq[:4] for m in p) - read3)
                    assert len(d5) == 1 and d5[0] in legal
                    dopt5 = d5[0]
                else:
                    assert not DS.legal_metals(dz, read3)
                    dopt5 = -1
                # the looser E3 rule (measured, unread, no strict neighbour read; no reachability look-ahead)
                qs = set(cell["query"])
                loose = sorted((m for m in metals if m not in read3 and not any(((m, s) in qs or (s, m) in qs) for s in read3)),
                               key=lambda m: ZV[m])
                assert set(legal) <= set(loose) and (cell["k_max"] >= 4 or not loose)
                Yr = Y[ci]
                base = {"design": design, "split_seed": int(f.seed), "fold": int(f.fold), "cell_id": cell["cell_id"],
                        "extractant": cell["extractant"], "chemotype": cell["chemotype"], "publication": cell["publication"],
                        "n_metals": len(metals), "k_max": int(cell["k_max"])}
                pk = {}
                if legal:
                    pk = {"DOPTQ": dopt5, "MAXMIN_R": maxmin(read3, legal, XR), "MAXMIN_Z": maxmin(read3, legal, XZ),
                          "BISECT_R": bisect(metals, read3, legal, XR), "BISECT_Z": bisect(metals, read3, legal, XZ),
                          "BISECT_RMID": bisect(metals, read3, legal, XR, end_target="mid"),
                          "HULL_BISECT": bisect(metals, read3, legal, XR, hull_first=True),
                          "MAXMIN_R_LOOSE": maxmin(read3, loose, XR)}
                cells.append({**base, "read3": ",".join(LANTHANIDES[m] for m in sorted(read3)),
                              "legal": ",".join(LANTHANIDES[m] for m in legal), "n_legal": len(legal),
                              "n_legal_loose": len(loose), "legal_loose": ",".join(LANTHANIDES[m] for m in sorted(loose)),
                              **{f"pick_{p}": (LANTHANIDES[m] if m >= 0 else "") for p, m in pk.items()}})
                # k = 3
                L3 = {"B0": lv(Yr, metals, read3, 0.0), "B1": lv(Yr, metals, read3, b3), "HIST_D7": lv(Yr, metals, read3, RF.HIST_D7_BETA)}
                for a, b in cell["query"]:
                    assert not (a in read3 and b in read3)
                    r = {**base, "k": 3, "cand": "", "A": LANTHANIDES[a], "B": LANTHANIDES[b],
                         "a_read": a in read3, "b_read": b in read3, "hull_out": outside_hull(a, b, read3),
                         "n_read": 4, "y": float(Yr[a] - Yr[b])}
                    for n, v in L3.items():
                        r[n] = float(v[a] - v[b])
                    r["B1_k3beta"] = r["B1"]
                    rows.append(r)
                # k = 4: every 5th metal admissible under the loose rule (superset of the protocol's legal set)
                for m in loose:
                    read4 = read3 | {m}
                    L4 = {"B0": lv(Yr, metals, read4, 0.0), "B1": lv(Yr, metals, read4, b4),
                          "B1_k3beta": lv(Yr, metals, read4, b3), "HIST_D7": lv(Yr, metals, read4, RF.HIST_D7_BETA)}
                    for a, b in cell["query"]:
                        assert not (a in read4 and b in read4), "a 5th read closed a query"
                        r = {**base, "k": 4, "cand": LANTHANIDES[m], "strict_legal": m in legal, "A": LANTHANIDES[a], "B": LANTHANIDES[b],
                             "a_read": a in read4, "b_read": b in read4, "hull_out": outside_hull(a, b, read4),
                             "n_read": 5, "y": float(Yr[a] - Yr[b])}
                        for n, v in L4.items():
                            r[n] = float(v[a] - v[b])
                        rows.append(r)
            print(f"{design} {f.seed} fold {f.fold}: beta3 {b3:+.4f} (stored {float(st['coef'][0]):+.4f}, lam {fit3['lambda']}) "
                  f"beta4 {b4:+.4f} (lam {fit4['lambda']}); rows {len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_parquet(OUT / "rows.parquet", index=False)
    pd.DataFrame(cells).to_parquet(OUT / "cells.parquet", index=False)
    (OUT / "fits.json").write_text(json.dumps(fits, indent=1, default=str), encoding="utf-8")
    print(f"EXIT 0 ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

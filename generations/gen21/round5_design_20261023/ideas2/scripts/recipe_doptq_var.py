#!/usr/bin/env python
"""Theme "minimal-measurement recipes", two more quick checks (EXPLORATORY; already-seen cells).

  V1  The programme's reference read policy as a comparator for the HULL rule: nested DOPT_Q read sets from the gen21
      fold cache (BPG, 5 dev seeds; 2 / 3 / 4 / 5 reads = seq[:1..4]), scored in this theme's frame (training-free D7
      and SH_own; all strict-adjacent pairs), seed-mean R2, against HULL on the same cells.
  V2  Within-campaign acid dependence of adjacent log SF: groups = publication x extractant x every recorded condition
      except acid concentration / contact time / metal concentration; groups with >= 3 distinct acid concentrations.
      Per (group, pair type): share of the within-group SS explained by a group-specific linear slope in log10[acid],
      against the share expected from noise alone (1/(n-1)); residual SD.  Uses every bench cell (not only C3),
      C5-BTBP and round-3 V2-1/2/3 cells removed.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe_doptq_var.py
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES, Z_VEC  # noqa: E402
import recipe_analyze as RA  # noqa: E402
from recipe_extra import hull_order  # noqa: E402

OUT = RA.OUT


def main() -> int:
    cells = pd.read_parquet(OUT / "recipe_cells.parquet")
    pairs = pd.read_parquet(OUT / "recipe_pairs.parquet")
    npz = {n: dict(np.load(OUT / f"recipe_k{n}.npz")) for n in range(2, 6)}
    gidx = {n: {cmb: q for q, cmb in enumerate(itertools.combinations(range(N_LN), n))} for n in range(2, 6)}
    ci_of = dict(zip(cells.cell_id, cells.ci))
    pops = {"ALL229": cells, "noC5_214": cells[~cells.c5], "OWN1_noC5": cells[(cells.n_own >= 1) & ~cells.c5],
            "OWN2_noC5": cells[(cells.n_own >= 2) & ~cells.c5]}
    # ---------------- V1 ----------------
    dsets = {}                                            # (seed, cell) -> {n: global subset id}
    for seed in paths.DEV_SEEDS:
        for fold in range(5):
            rec = C.load("BPG", seed, fold)
            for cell in rec["outer"]["cells"]:
                seq = cell["seqs"]["DOPT_Q"]
                d = {}
                for k in range(1, 5):
                    if k <= len(seq):
                        rd = tuple(sorted({m for p in seq[:k] for m in p}))
                        if len(rd) in gidx:
                            d[len(rd)] = gidx[len(rd)][rd]
                dsets[(seed, cell["cell_id"])] = d
    rows = []
    for pk, sub in pops.items():
        idx = sub.ci.to_numpy()
        N, sy, syy = (sub[k].to_numpy(float) for k in ("N", "sy", "syy"))
        sst = syy.sum() - sy.sum() ** 2 / N.sum()
        for arm in ("D7", "SH_own") if pk.startswith("OWN") else ("D7",):
            for n in (2, 3, 4, 5):
                hull = pairs[pairs.n_read == n].groupby("ci")[f"hull_{arm}"].sum().reindex(idx).fillna(0).to_numpy()
                r2s, cov = [], []
                for seed in paths.DEV_SEEDS:
                    s = 0.0
                    got = 0
                    for cid, c in zip(sub.cell_id, idx):
                        q = dsets.get((seed, cid), {}).get(n)
                        if q is None:                 # policy has no n-read set (k_max too small): use HULL
                            s += hull[list(idx).index(c)]
                        else:
                            s += float(npz[n][f"sse_{arm}"][c, q])
                            got += 1
                    r2s.append(1 - s / sst)
                    cov.append(got / len(idx))
                rows.append({"pop": pk, "arm": arm, "n_read": n, "DOPT_Q": float(np.mean(r2s)),
                             "DOPT_Q_seed_min": float(np.min(r2s)), "DOPT_Q_seed_max": float(np.max(r2s)),
                             "share_cells_with_DOPT_set": float(np.mean(cov)), "HULL": float(1 - hull.sum() / sst)})
    V1 = pd.DataFrame(rows)
    V1.to_csv(OUT / "recipe_V1_doptq.csv", index=False)
    print("=== V1 DOPT_Q-nested vs HULL (training-free frame, all strict-adjacent pairs; BPG read sets, seed mean)")
    print(V1.round(3).to_string(index=False))

    # ---------------- V2 ----------------
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    keep = [(c not in drop) and (p != "pub_6a0ef8528c") for c, p in zip(ids, f.publication_id.astype(str))]
    cond = [c for c in f.columns if c.startswith("cond__") and c not in
            ("cond__acid_concentration_M", "cond__contact_time_min", "cond__metal_concentration_mM")]
    key = f[cond].astype(float).round(6).fillna(-999.0).map(repr).agg("|".join, axis=1)
    f = f.assign(gkey=f.publication_id.astype(str) + "#" + f.extractant.astype(str) + "#" + key)
    acid = f.cond__acid_concentration_M.to_numpy(float)
    recs = []
    for g, rows_ in f[keep].groupby("gkey"):
        ii = rows_.index.to_numpy()
        ii = ii[np.isfinite(acid[ii]) & (acid[ii] > 0)]
        if len(np.unique(np.round(np.log10(acid[ii]), 6))) < 3:
            continue
        for a in range(N_LN - 1):
            if Z_VEC[a + 1] - Z_VEC[a] != 1:
                continue
            sf = Y[ii, a] - Y[ii, a + 1]
            ok = np.isfinite(sf)
            if ok.sum() < 3:
                continue
            la = np.log10(acid[ii][ok])
            if len(np.unique(np.round(la, 6))) < 3:
                continue
            v = sf[ok]
            ss = ((v - v.mean()) ** 2).sum()
            b = np.polyfit(la, v, 1)
            res = v - np.polyval(b, la)
            recs.append({"group": g, "pub": g.split("#")[0], "pair": f"{LANTHANIDES[a]}/{LANTHANIDES[a + 1]}",
                         "n": int(ok.sum()), "ss": ss, "ss_res": float((res ** 2).sum()), "slope": float(b[0]),
                         "mean": float(v.mean()), "acid_range_log": float(la.max() - la.min())})
    V2t = pd.DataFrame(recs)
    V2t.to_csv(OUT / "recipe_V2_acid.csv", index=False)
    tot = V2t.ss.sum()
    expl = (V2t.ss - V2t.ss_res).sum()
    noise_exp = (V2t.ss / (V2t.n - 1)).sum()            # expected explained SS if SF were pure noise (1 df per fit)
    print("\n=== V2 within-campaign acid dependence of adjacent log SF (acid-only groups, >= 3 acid levels)")
    print(f"groups {V2t.group.nunique()}, pubs {V2t.pub.nunique()}, (group,pair) {len(V2t)}, points {V2t.n.sum()}")
    print(f"within-group SD of SF {math.sqrt(tot / (V2t.n - 1).sum()):.3f}; residual SD after linear log-acid "
          f"{math.sqrt(V2t.ss_res.sum() / (V2t.n - 2).sum()):.3f}")
    print(f"share of within-group SS explained by a linear log-acid slope {expl / tot:.3f}; "
          f"expected under pure noise {noise_exp / tot:.3f}")
    # between-group (same pair type) variance of the group means, for scale
    bg = V2t.groupby("pair")["mean"].agg(["std", "count"])
    print("between-group SD of group-mean SF by pair:", bg.round(3).to_dict())
    per_pub = V2t.groupby("pub").apply(lambda d: pd.Series({"pairs": len(d), "points": d.n.sum(),
                                                             "expl_share": (d.ss - d.ss_res).sum() / d.ss.sum(),
                                                             "noise_share": (d.ss / (d.n - 1)).sum() / d.ss.sum(),
                                                             "wg_sd": math.sqrt(d.ss.sum() / (d.n - 1).sum()),
                                                             "acid_range": d.acid_range_log.median()}))
    print(per_pub.round(3).to_string())
    per_pair = V2t.groupby("pair").apply(lambda d: pd.Series({"groups": len(d),
                                                               "expl_share": (d.ss - d.ss_res).sum() / d.ss.sum(),
                                                               "noise_share": (d.ss / (d.n - 1)).sum() / d.ss.sum(),
                                                               "wg_sd": math.sqrt(d.ss.sum() / (d.n - 1).sum()),
                                                               "median_slope": d.slope.median()}))
    print(per_pair.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

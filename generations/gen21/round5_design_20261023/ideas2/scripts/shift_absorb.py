#!/usr/bin/env python
"""shift theme, step 7 (EXPLORATORY): how much of the acid-induced change in adjacent log SF is a smooth deformation
that 4 reads + PWL_R already absorb?  For each acid-series group with >= 10 per-metal slopes (shift_slopes.py
grouping), centred slope profile g_m = dlogD_m/dlog10[acid] - mean.  For the five most common C3 DOPT_Q read sets,
interpolate g through the read metals (PWL_R) and compare the strict-adjacent pair slopes s_p = g_A - g_B with the
residual pair slopes after interpolation: share absorbed = 1 - sum(res^2) / sum(s^2).

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift_absorb.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES as LN, N_LN  # noqa: E402


def ols(x, y):
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.ptp(x) < 0.1:
        return np.nan
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def main():
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    v2 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet", columns=["split_seed", "cell_id", "read_set"])
    rs = ref.drop_duplicates(["split_seed", "cell_id"]).read_set.value_counts()
    top = rs.index[:5].tolist()
    print("most common DOPT_Q read sets (cell x seed instances):", rs.head(5).to_dict())

    def lab(r, pre):
        return "+".join(c for c in f.columns if c.startswith(pre) and r[c] == 1)
    f["k"] = [(r.publication_id, r.extractant, lab(r, "cond__acid__"), lab(r, "cond__diluent__"), lab(r, "cond__additive__"),
               round(float(r.cond__extractant_concentration_M), 3), str(r.cond__temperature_C)) for _, r in f.iterrows()]
    f["x"] = np.log10(f.cond__acid_concentration_M.astype(float))
    keep = ~f.cell_id.isin(v2) & f.x.notna()
    rows = []
    for k, g in f[keep].groupby("k"):
        idx = g.index.to_numpy()
        if len(g) < 2 or np.ptp(g.x) < 0.1:
            continue
        bm = np.array([ols(g.x.to_numpy(), Y[idx, m]) for m in range(N_LN)])
        if np.isfinite(bm).sum() < 10:
            continue
        gm = bm - np.nanmean(bm)
        metals = [m for m in range(N_LN) if np.isfinite(gm[m])]
        pairs = [(a, a + 1) for a in range(N_LN - 1) if a != 3 and a in metals and a + 1 in metals]
        s = np.array([gm[a] - gm[c] for a, c in pairs])
        for rset in top:
            rd = [LN.index(z) for z in rset.split(",")]
            if not all(m in metals for m in rd):
                continue
            ym = np.full(N_LN, np.nan)
            ym[rd] = gm[rd]
            p = interp.pwl_levels(ym, metals, "R")
            sp = np.array([p[a] - p[c] for a, c in pairs])
            rows.append({"pub": k[0][:14], "acid": k[2], "read_set": rset, "pairs": len(pairs),
                         "rms_pair_slope": float(np.sqrt((s ** 2).mean())),
                         "rms_resid_after_reads": float(np.sqrt(((s - sp) ** 2).mean())),
                         "share_absorbed": float(1 - ((s - sp) ** 2).sum() / (s ** 2).sum())})
    R = pd.DataFrame(rows)
    print(R.round(3).to_string(index=False))
    G = R.groupby(["pub", "acid"]).agg(rms_pair_slope=("rms_pair_slope", "first"), rms_resid=("rms_resid_after_reads", "mean"),
                                       share_absorbed=("share_absorbed", "mean"))
    print("\nper group (mean over read sets):")
    print(G.round(3).to_string())
    print(f"\nmedian share absorbed {G.share_absorbed.median():.3f}; pooled share absorbed "
          f"{1 - (R.rms_resid_after_reads ** 2 * R.pairs).sum() / (R.rms_pair_slope ** 2 * R.pairs).sum():.3f}")
    print(f"median rms pair slope {G.rms_pair_slope.median():.3f} log/decade; median rms residual {G.rms_resid.median():.3f}")


if __name__ == "__main__":
    main()

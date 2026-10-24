"""addendum_gd_reconcile.py -- EXPLORATORY reconciliation check for IDEAS2_RANKED.md (confirm-2 / C-GD).

Question: skeptic A (confirm-2) reports that a heavy-selective (HS) Gd amplitude transfers across publications on
Gd-chord cells (LOPO -59 % Gd-cell SSE vs beta). Skeptic B reports that the same stratum amplitude, applied in the
deployed k = 3 pipeline (round-4 REF rows, OOF), gains only through C5-BTBP. Both can be true if the deployed gain
depends on (i) whether Gd is read and (ii) which units the stratum admits. This script splits skeptic B's own OOF rows
(confirm2skB_out/O_rows_{BPG,BP}.parquet, arms B1 / POOL / STRAT) by:
  - Gd read vs Gd hidden (from the REF read_set, label-free);
  - eligibility as confirm-2 freezes it: the cell reports Eu, Gd and Tb (pairs Eu/Gd and Gd/Tb exist), and is not the
    component-identity / aqueous-complexant publication pub_84113c17ad (P5 finding F6; M29 rule);
  - with and without C5-BTBP (pub_6a0ef8528c).
Every number is on the 229 already-seen C3 cells and the 5 dev seeds: EXPLORATORY, not evidence.
Reads only existing parquets; writes nothing except stdout.
"""
import numpy as np
import pandas as pd

ROOT = "D:/ml_separator_gh/generations/gen21"
O_PATH = ROOT + "/round5_design_20261023/ideas2/confirm2skB_out/O_rows_{d}.parquet"
REF_PATH = ROOT + "/round4_20261022/predictions/REF_{d}.parquet"
C5 = "pub_6a0ef8528c"
COMPLEXANT = "pub_84113c17ad"


def r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)


def summarise(df, label, arms=("B1", "POOL", "STRAT")):
    if len(df) == 0:
        print(f"  {label:<46s} (empty)")
        return
    out = {}
    for a in arms:
        sse = df.groupby("split_seed").apply(lambda g: np.sum((g.y - g[a]) ** 2), include_groups=False).mean()
        r2s = df.groupby("split_seed").apply(lambda g: r2(g.y.values, g[a].values), include_groups=False).mean()
        out[a] = (sse, r2s)
    rows = len(df) / df.split_seed.nunique()
    rel = 100.0 * (out["B1"][0] - out["STRAT"][0]) / out["B1"][0]
    relp = 100.0 * (out["POOL"][0] - out["STRAT"][0]) / out["POOL"][0]
    # per-publication sign of the SSE gain STRAT vs B1 (seed mean)
    pg = df.assign(g=(df.y - df.B1) ** 2 - (df.y - df.STRAT) ** 2).groupby(["publication", "split_seed"]).g.sum()
    pg = pg.groupby("publication").mean()
    up, dn = int((pg > 1e-12).sum()), int((pg < -1e-12).sum())
    print(f"  {label:<46s} rows/seed {rows:6.1f} pubs {df.publication.nunique():2d} | "
          f"R2 B1 {out['B1'][1]:.3f} POOL {out['POOL'][1]:.3f} STRAT {out['STRAT'][1]:.3f} | "
          f"SSE/seed B1 {out['B1'][0]:.3f} STRAT {out['STRAT'][0]:.3f} | STRAT vs B1 {rel:+.1f}% "
          f"(vs POOL {relp:+.1f}%) | pubs STRAT better {up} / worse {dn}")


def main():
    for d in ("BPG", "BP"):
        o = pd.read_parquet(O_PATH.format(d=d))
        r = pd.read_parquet(REF_PATH.format(d=d))
        assert len(o) == len(r)
        assert (o.split_seed.values == r.split_seed.values).all()
        assert (o.cell_id.values == r.cell_id.values).all()
        assert np.allclose(o.y.values, r.y.values, atol=0, rtol=0)
        assert np.allclose(o.B1.values, r.B1.values, atol=1e-12)
        o = o.copy()
        o["A"], o["B"], o["read_set"] = r.A.values, r.B.values, r.read_set.values
        o["gd_read"] = o.read_set.str.split(",").apply(lambda s: "Gd" in s)
        # label-free eligibility: the cell reports Eu, Gd and Tb  <=>  both strict pairs Eu/Gd and Gd/Tb exist
        pairs = o.A + "/" + o.B
        has = pairs.groupby(o.cell_id).agg(lambda s: {"Eu/Gd", "Gd/Tb"} <= set(s))
        o["triad"] = o.cell_id.map(has)
        print(f"== {d}: confirm-2 reconciliation on skeptic B's OOF rows (EXPLORATORY) ==")
        print(f"   fold beta means: pool {o.b_pool.mean():.3f}  HS {o.b_hs.mean():.3f}; "
              f"Gd read in {100 * o.drop_duplicates(['split_seed', 'cell_id']).gd_read.mean():.1f}% of cell x seed read sets")
        for strat_col in ("cls", "cls_full"):
            hs = o[o[strat_col] == "nonDGA_HS"]
            print(f" -- HS defined by {strat_col} ({'reads-only slope' if strat_col == 'cls' else 'full-series slope'})")
            summarise(hs, "all HS rows")
            summarise(hs[hs.publication != C5], "HS without C5")
            e = hs[(hs.publication != C5) & (hs.publication != COMPLEXANT)]
            summarise(e, "HS without C5, without complexant pub")
            e2 = e[e.triad]
            summarise(e2, "  ... and Eu-Gd-Tb cells only (eligible)")
            summarise(e2[~e2.gd_read], "  ... eligible, Gd hidden")
            summarise(e2[e2.gd_read], "  ... eligible, Gd read")
            summarise(e2[e2.gdpair], "  ... eligible, Eu/Gd + Gd/Tb pairs only")
            w5 = hs[hs.triad]
            summarise(w5, "with C5: Eu-Gd-Tb cells, complexant pub kept")
        # per-publication table for the eligible, C5-free, reads-defined stratum
        hs = o[(o.cls == "nonDGA_HS") & (o.publication != C5) & (o.publication != COMPLEXANT) & o.triad]
        g = hs.assign(gain=(hs.y - hs.B1) ** 2 - (hs.y - hs.STRAT) ** 2).groupby(["publication", "split_seed"]).gain.sum()
        print("   per-publication SSE gain STRAT vs B1 (seed mean), eligible reads-defined HS, no C5:")
        print("   " + ", ".join(f"{k}: {v:+.3f}" for k, v in g.groupby('publication').mean().sort_values().items()))
        print()


if __name__ == "__main__":
    main()

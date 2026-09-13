"""Step 3c: guard the precision story against the confounds this corpus is known to carry.

The protocol requires every correlation here to be checked against publication identity, the number
of metals a cell measured (Spearman +0.49 with |a|) and chemotype, and to report leave-one-
chemotype-out stability.  Two questions specifically:

* is the replicate-derived precision weight a *different object* from the |a|-confidence weight
  gen14 already tested and found worth +-0.003?
* does the reliability of a and b hold up chemotype by chemotype and publication by publication, or
  is it carried by one block?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from noise import coef_cov, reliability, sigma_table  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen13sep.amplitude_bench import cell_weights  # noqa: E402
from gen14.models import TAU, confidence_weights  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
MIN_METALS = 5


def main() -> None:
    bench = V.load()
    f = bench.frame
    a, b = bench.coef[:, 0], bench.coef[:, 1]
    m = f.n_metals.to_numpy(dtype=float)
    se = sigma_table(bench, "resid_const")
    cov = coef_cov(bench, se)
    va, vb = cov[:, 0, 0], cov[:, 1, 1]
    sa = np.sqrt(va)

    r = bench.basis[0]
    obs = ~np.isnan(bench.Y)
    span = np.array([float(r[o].max() - r[o].min()) if o.sum() > 1 else 0.0 for o in obs])

    rich = m >= MIN_METALS
    w = cell_weights(bench.groups[rich], bench.n_obs[rich])
    conf = confidence_weights(w, a[rich], bench.groups[rich], TAU)
    prec = w / (0.02 + va[rich])
    for g in np.unique(bench.groups[rich]):
        k = bench.groups[rich] == g
        prec[k] *= w[k].sum() / prec[k].sum()

    rows = [
        {"pair": "precision weight vs gen14 |a|-confidence weight (rich cells)",
         "spearman": float(spearmanr(prec, conf).statistic),
         "pearson": float(np.corrcoef(prec, conf)[0, 1])},
        {"pair": "se(a) vs n_metals (all cells)", "spearman": float(spearmanr(sa, m).statistic),
         "pearson": float(np.corrcoef(sa, m)[0, 1])},
        {"pair": "se(a) vs radius span of the measured metals",
         "spearman": float(spearmanr(sa, span).statistic),
         "pearson": float(np.corrcoef(sa, span)[0, 1])},
        {"pair": "se(a) vs |a|", "spearman": float(spearmanr(sa, np.abs(a)).statistic),
         "pearson": float(np.corrcoef(sa, np.abs(a))[0, 1])},
        {"pair": "|a| vs n_metals (the known confound)",
         "spearman": float(spearmanr(np.abs(a), m).statistic),
         "pearson": float(np.corrcoef(np.abs(a), m)[0, 1])},
    ]
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "s3c_confounds.csv", index=False)
    print("=== what the precision weight actually is ===")
    print(d.round(4).to_string(index=False))
    print(f"\nspread of se(a) among the cells gen14 trains on (n_metals >= 5): "
          f"min {sa[rich].min():.4f} median {np.median(sa[rich]):.4f} max {sa[rich].max():.4f}; "
          f"the precision weight therefore varies by a factor "
          f"{(0.02 + va[rich].max()) / (0.02 + va[rich].min()):.2f}")

    # variance explained in se(a) by publication / chemotype / n_metals
    def eta2(y, g):
        g = np.asarray(g)
        tot = ((y - y.mean()) ** 2).sum()
        within = sum(((y[g == k] - y[g == k].mean()) ** 2).sum() for k in np.unique(g))
        return 1 - within / tot if tot > 0 else np.nan

    ee = pd.DataFrame([{"grouping": k, "eta2_of_se_a": eta2(sa, v), "n_levels": len(np.unique(v))}
                       for k, v in (("publication_id", f.publication_id.astype(str).to_numpy()),
                                    ("chemotype", bench.groups),
                                    ("n_metals", m))])
    ee.to_csv(OUT / "s3c_eta2.csv", index=False)
    print("\nhow much of the per-cell precision is explained by each confound")
    print(ee.round(4).to_string(index=False))

    # leave-one-chemotype-out reliability
    loco = []
    for g in np.unique(bench.groups):
        k = bench.groups != g
        loco.append({"left_out": g, "n_left_out": int((~k).sum()),
                     "rel_a": reliability(a[k], va[k])["reliability"],
                     "rel_b": reliability(b[k], vb[k])["reliability"]})
    lo = pd.DataFrame(loco)
    lo.to_csv(OUT / "s3c_loco_reliability.csv", index=False)
    full_a = reliability(a, va)["reliability"]
    full_b = reliability(b, vb)["reliability"]
    print(f"\n=== leave-one-chemotype-out reliability (full corpus: a {full_a:.3f}, b {full_b:.3f}) ===")
    print(f"a: min {lo.rel_a.min():.3f} max {lo.rel_a.max():.3f}   "
          f"b: min {lo.rel_b.min():.3f} max {lo.rel_b.max():.3f}   over {len(lo)} chemotypes")
    print(lo.sort_values('rel_b').head(4).round(4).to_string(index=False))

    # by publication (>= 8 cells)
    pub = f.publication_id.astype(str).to_numpy()
    rows = []
    for p in np.unique(pub):
        k = pub == p
        if k.sum() < 8:
            continue
        rows.append({"publication": p, "n_cells": int(k.sum()),
                     "rel_a": reliability(a[k], va[k])["reliability"],
                     "rel_b": reliability(b[k], vb[k])["reliability"],
                     "median_n_metals": float(np.median(m[k]))})
    pr = pd.DataFrame(rows).sort_values("rel_b")
    pr.to_csv(OUT / "s3c_reliability_by_publication.csv", index=False)
    print(f"\n=== reliability within a publication ({len(pr)} publications with >= 8 cells) ===")
    print(pr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

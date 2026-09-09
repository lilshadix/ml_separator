"""Why the similarity kernel gains under B/BR/BQ and not under BP.

The chemotype hold-out removes the test ligand's own Tanimoto-0.7 cluster from training, but it
does NOT remove the test cell's *publication*.  A paper that reports one ligand usually reports
several, and the ones it reports together are more similar to each other than to the corpus at
large -- so a Tanimoto kernel can put a large share of its weight on training cells from the test
cell's own laboratory, and the magnitude of the radius coefficient is strongly a property of the
laboratory (gen15's O_PUBMAG oracle scores 0.448 against G14's 0.500).

This script measures that share directly, per design: the fraction of the kernel's normalised
training weight that sits in the test cell's own publication, and the fraction of the five most
similar training cells that come from it.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "gen15_curve"))
sys.path.insert(0, str(HERE))

from gen15 import valuebench as V                     # noqa: E402
from gen15.valuebench import Ctx, MIN_METALS          # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights  # noqa: E402
from gen13sep.splits import all_folds                 # noqa: E402
from gen14.dirbench import feature_sets               # noqa: E402
import kernels as KK                                   # noqa: E402

OUT = HERE / "results"
DESIGNS = ["B", "BR", "BQ", "A", "BP"]


def main() -> None:
    bench = V.load()
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    pub = bench.frame.publication_id.astype(str).to_numpy()
    chem = bench.frame.chemotype.astype(str).to_numpy()
    rows = []
    for d in DESIGNS:
        for f in all_folds(bench.frame, design=d):
            ctx = Ctx(bench=bench, design=d, seed=f.seed, fold=f.fold, train=f.train_index,
                      test=f.test_index,
                      w=cell_weights(bench.groups[f.train_index], bench.n_obs[f.train_index]),
                      model_seed=f.model_seed, fs=fs, X=X, rich=rich)
            tr, w = ctx.rich_train(), ctx.rich_weights()
            if len(tr) == 0:
                continue
            K = KK.tanimoto_gram(ctx)
            S = K[np.ix_(ctx.test, tr)]                       # similarity to each training cell
            ww = KK._norm_w(w)
            Wt = S * ww[None, :]
            Wt = Wt / np.maximum(Wt.sum(axis=1, keepdims=True), 1e-12)
            same_pub = pub[tr][None, :] == pub[ctx.test][:, None]
            kk = min(5, S.shape[1])
            idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            top5 = np.array([same_pub[i, idx[i]].mean() for i in range(len(ctx.test))])
            rows.append(pd.DataFrame({
                "design": d, "seed": f.seed, "cell": ctx.test, "rich": rich[ctx.test],
                "kernel_weight_same_pub": (Wt * same_pub).sum(axis=1),
                "top5_cells_same_pub": top5,
                "pub_in_train": np.isin(pub[ctx.test], np.unique(pub[tr])),
                "chem_in_train": np.isin(chem[ctx.test], np.unique(chem[tr])),
            }))
        print(f"  {d} done", flush=True)
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(OUT / "publeak_cells.csv", index=False)
    r = t[t["rich"]]
    g = r.groupby("design").agg(
        n=("cell", "size"),
        kernel_weight_same_pub=("kernel_weight_same_pub", "mean"),
        top5_cells_same_pub=("top5_cells_same_pub", "mean"),
        frac_pub_in_train=("pub_in_train", "mean"),
        frac_chem_in_train=("chem_in_train", "mean"))
    g = g.reindex([d for d in DESIGNS if d in g.index])
    g.to_csv(OUT / "publeak_summary.csv")
    print("\n=== share of the Tanimoto kernel's training weight sitting in the test cell's own"
          " publication (well-determined cells) ===")
    print(g.round(4).to_string())

    # does the kernel's advantage over G14 live in the cells whose publication IS in training?
    ad = pd.read_csv(OUT / "ad_cells.csv")
    m = ad.merge(t[["design", "seed", "cell", "kernel_weight_same_pub", "pub_in_train"]],
                 on=["design", "seed", "cell"], how="left")
    m = m[m["n_metals"] >= MIN_METALS]
    out = []
    for d, sub in m.groupby("design"):
        for lab, s in (("pub in train", sub[sub["pub_in_train"]]),
                       ("pub NOT in train", sub[~sub["pub_in_train"].astype(bool)])):
            if len(s) < 10:
                continue
            out.append({"design": d, "stratum": lab, "n": len(s),
                        "mae_amp_gp": s["err_amp"].mean(), "mae_amp_g14": s["err_g14"].mean(),
                        "gp_minus_g14": s["err_amp"].mean() - s["err_g14"].mean()})
    o = pd.DataFrame(out)
    o.to_csv(OUT / "publeak_strata.csv", index=False)
    print("\n=== the GP magnitude arm's amplitude MAE, split by whether the test cell's"
          " publication is in training ===")
    print(o.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

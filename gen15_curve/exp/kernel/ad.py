"""The applicability-domain probe: how far is a held-out ligand from the training set, and does
the error know about it?

Three questions, all answered on the real folds of all five designs:

1. What is the Tanimoto from every held-out cell to its nearest training ligand?  The chemotype
   hold-out is *defined* as single-linkage Tanimoto 0.7 on the same fingerprint block, so this
   distribution is not a property of the data but of the split, and it should be reported as such.
2. Does the pairwise error grow as that similarity falls?  If it does not, the kernel has no
   applicability domain to speak of on this corpus.
3. Is the Tanimoto GP's posterior sd a useful confidence?  A confidence is useful if the error on
   the low-sd half is materially below the error on the high-sd half.

Also reports the confound guards the protocol requires: publication identity, the number of metals
a cell measured, and chemotype.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "gen15_curve"))
sys.path.insert(0, str(HERE))

from gen15 import valuebench as V                      # noqa: E402
from gen15.valuebench import Ctx, MIN_METALS           # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights   # noqa: E402
from gen13sep.splits import all_folds                  # noqa: E402
from gen14.dirbench import feature_sets                # noqa: E402
import kernels as KK                                    # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
DESIGNS = ["B", "BR", "BQ", "A", "BP"]


def probe(bench) -> pd.DataFrame:
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    rows = []
    for d in DESIGNS:
        for f in all_folds(bench.frame, design=d):
            ctx = Ctx(bench=bench, design=d, seed=f.seed, fold=f.fold, train=f.train_index,
                      test=f.test_index,
                      w=cell_weights(bench.groups[f.train_index], bench.n_obs[f.train_index]),
                      model_seed=f.model_seed, fs=fs, X=X, rich=rich)
            p = KK.ad_probe(ctx)
            t = pd.DataFrame(p)
            t["design"] = d
            t["seed"] = f.seed
            t["fold"] = f.fold
            rows.append(t)
        print(f"  {d} probed", flush=True)
    t = pd.concat(rows, ignore_index=True)
    fr = bench.frame
    for c in ("extractant", "chemotype", "publication_id", "n_metals"):
        t[c] = fr[c].to_numpy()[t["cell"].to_numpy()]
    return t


def bands(t: pd.DataFrame, col: str = "s_max") -> pd.DataFrame:
    """Error by similarity band, per design, on well-determined cells only."""
    r = t[t["rich"]].copy()
    edges = [0.0, 0.45, 0.55, 0.62, 0.68, 0.70, 1.001]
    r["band"] = pd.cut(r[col], edges, right=False)
    g = r.groupby(["design", "band"], observed=True).agg(
        n=("err_amp", "size"),
        mean_s=(col, "mean"),
        mae_amp_gp=("err_amp", "mean"),
        mae_amp_g14=("err_g14", "mean"),
        mae_mag_gp=("err_mag", "mean"),
        true_mag=("true_mag", "mean"))
    g["gp_minus_g14"] = g["mae_amp_gp"] - g["mae_amp_g14"]
    return g.reset_index()


def confidence(t: pd.DataFrame) -> pd.DataFrame:
    r = t[t["rich"]].copy()
    out = []
    for d, sub in r.groupby("design"):
        med = sub["gp_sd"].median()
        lo, hi = sub[sub["gp_sd"] <= med], sub[sub["gp_sd"] > med]
        out.append({"design": d, "n": len(sub), "median_gp_sd": med,
                    "mae_amp_low_sd": lo["err_amp"].mean(), "mae_amp_high_sd": hi["err_amp"].mean(),
                    "spearman_sd_vs_err": spearmanr(sub["gp_sd"], sub["err_amp"]).statistic,
                    "spearman_smax_vs_err": spearmanr(sub["s_max"], sub["err_amp"]).statistic})
    return pd.DataFrame(out)


def confounds(bench) -> pd.DataFrame:
    """Is ligand similarity to a peer informative about |a| once the confounds are held fixed?

    For every pair of *distinct extractants* that both have a well-determined curve, relate the
    Tanimoto between them to the absolute difference of their mean |a|, then repeat inside strata
    of the confounds: same publication vs different, same chemotype vs different, and matched on
    the number of metals measured.
    """
    class _F:
        pass
    f = _F()
    f.bench = bench
    T = KK.tanimoto_gram(f)
    fr = bench.frame
    rich = fr.n_metals.to_numpy() >= MIN_METALS
    amp = bench.coef[:, 0]
    ext = fr.extractant.to_numpy()
    rows = []
    for e in np.unique(ext[rich]):
        m = (ext == e) & rich
        i = int(np.flatnonzero(m)[0])
        rows.append({"i": i, "ext": e, "mag": float(np.abs(amp[m]).mean()),
                     "amp": float(amp[m].mean()),
                     "chem": fr.chemotype.to_numpy()[i],
                     "pubs": set(fr.publication_id.to_numpy()[m]),
                     "nm": float(fr.n_metals.to_numpy()[m].mean())})
    L = pd.DataFrame(rows)
    ii = L["i"].to_numpy()
    n = len(L)
    a, b = np.triu_indices(n, 1)
    tan = T[np.ix_(ii, ii)][a, b]
    dmag = np.abs(L["mag"].to_numpy()[a] - L["mag"].to_numpy()[b])
    same_sign = (np.sign(L["amp"].to_numpy()[a]) == np.sign(L["amp"].to_numpy()[b])).astype(float)
    same_chem = (L["chem"].to_numpy()[a] == L["chem"].to_numpy()[b])
    same_pub = np.array([bool(L["pubs"].iat[x] & L["pubs"].iat[y]) for x, y in zip(a, b)])
    dnm = np.abs(L["nm"].to_numpy()[a] - L["nm"].to_numpy()[b])
    out = []

    def add(label, mask):
        if mask.sum() < 30:
            out.append({"stratum": label, "n_pairs": int(mask.sum()), "spearman_tan_vs_dmag": np.nan,
                        "mean_same_sign_hi_tan": np.nan, "mean_same_sign_lo_tan": np.nan})
            return
        s = spearmanr(tan[mask], dmag[mask]).statistic
        q = np.quantile(tan[mask], 0.75)
        hi = mask & (tan >= q)
        lo = mask & (tan < np.quantile(tan[mask], 0.25))
        out.append({"stratum": label, "n_pairs": int(mask.sum()), "spearman_tan_vs_dmag": s,
                    "mean_same_sign_hi_tan": float(same_sign[hi].mean()),
                    "mean_same_sign_lo_tan": float(same_sign[lo].mean())})

    all_m = np.ones(len(tan), dtype=bool)
    add("all extractant pairs", all_m)
    add("same chemotype", same_chem)
    add("different chemotype (the deployed regime)", ~same_chem)
    add("share a publication", same_pub)
    add("share no publication", ~same_pub)
    add("diff chemotype AND no shared publication", (~same_chem) & (~same_pub))
    add("matched n_metals (|dn| <= 1), diff chemotype", (dnm <= 1) & (~same_chem))
    return pd.DataFrame(out)


def loco(t: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-chemotype-out stability of the GP magnitude arm against G14, under BP."""
    r = t[(t["design"] == "BP") & t["rich"]]
    rows = []
    for c, sub in r.groupby("chemotype"):
        if len(sub) < 5:
            continue
        rows.append({"chemotype": c, "n": len(sub),
                     "mae_gp": sub["err_amp"].mean(), "mae_g14": sub["err_g14"].mean(),
                     "delta": sub["err_amp"].mean() - sub["err_g14"].mean()})
    d = pd.DataFrame(rows).sort_values("delta")
    return d


def main() -> None:
    bench = V.load()
    t = probe(bench)
    t.drop(columns=["rich"]).to_csv(OUT / "ad_cells.csv", index=False)
    print("\n=== error by Tanimoto-to-nearest-training-ligand band (well-determined cells) ===")
    print(bands(t).round(4).to_string(index=False))
    print("\n=== GP posterior sd as a confidence ===")
    c = confidence(t)
    c.to_csv(OUT / "ad_confidence.csv", index=False)
    print(c.round(4).to_string(index=False))
    print("\n=== confound guards on the similarity-magnitude relation (extractant pairs) ===")
    cf = confounds(bench)
    cf.to_csv(OUT / "ad_confounds.csv", index=False)
    print(cf.round(4).to_string(index=False))
    print("\n=== leave-one-chemotype-out, BP, GP magnitude minus G14 (amplitude MAE) ===")
    lo = loco(t)
    lo.to_csv(OUT / "ad_loco.csv", index=False)
    print(lo.round(4).to_string(index=False))
    print("\nchemotypes where the GP helps: %d of %d" % (int((lo["delta"] < 0).sum()), len(lo)))
    bands(t).to_csv(OUT / "ad_bands.csv", index=False)


if __name__ == "__main__":
    main()

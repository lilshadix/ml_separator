"""Stress the one positive in this experiment before anybody believes it.

``probe_baselines`` left exactly one live number: a ridge on MoLFormer's mean-pooled vector predicts
``log|a|`` out of fold under BP at Spearman +0.23, where gen14's TOPO39, gen13's LEAN209, Morgan
counts, both ChemBERTa checkpoints and a four-number size block all sit at or below zero.  Two things
must be true before that is worth anything, and the second is already in doubt:

1. **It must not be chance.**  Null here is a *row permutation of the embedding table*: every ligand
   keeps a real MoLFormer vector, but somebody else's.  That destroys the ligand-to-vector map while
   preserving the fold plan, the class balance, the n_metals structure and the vectors' own
   covariance -- so the null Spearman is what this pipeline produces from a representation that
   cannot know anything.

2. **It must not depend on the estimator.**  The cosine-kernel version of the same probe came back
   at -0.14 under BP while ridge came back at +0.23.  A signal whose sign flips with the penalty is
   not a signal.  Swept here over ridge alpha and both kernels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT / "gen15_curve"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import features as FEAT                                            # noqa: E402
from signal_probe import probe, summarise                          # noqa: E402
from gen15 import valuebench as V                                  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
TAG = "molformer__mean"


def main() -> None:
    design = sys.argv[1] if len(sys.argv) > 1 else "BP"
    n_perm = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    bench = V.load()

    # ---- 1. estimator sweep ---------------------------------------------------------
    rows = []
    for how, alpha in (("ridge", 10.0), ("ridge", 100.0), ("ridge", 1000.0), ("ridge", 10000.0),
                       ("cosine", 0.1), ("cosine", 1.0), ("cosine", 10.0),
                       ("rbf", 0.1), ("rbf", 1.0), ("rbf", 10.0)):
        d = probe(bench, TAG, "log_abs_a", design, how=how, alpha=alpha)
        s = summarise(d, TAG, "log_abs_a", design, f"{how}@{alpha:g}")
        if s:
            rows.append(s)
            print(f"{how:7s} alpha={alpha:<8g} rho={s['spearman']:+.3f} "
                  f"partial={s['partial_spearman_given_n_metals']:+.3f} "
                  f"loco[{s['loco_spearman_min']:+.3f},{s['loco_spearman_max']:+.3f}]", flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / f"probe_estimator_sweep_{design}.csv", index=False)

    # ---- 2. permutation null for the configuration that looked best -----------------
    orig = FEAT.block(TAG).copy()
    obs = probe(bench, TAG, "log_abs_a", design, how="ridge", alpha=100.0)
    obs_rho = summarise(obs, TAG, "log_abs_a", design, "ridge")["spearman"]
    rng = np.random.default_rng(20240909)
    orig_block = FEAT.block
    null = []
    for i in range(n_perm):
        perm = orig.copy()
        perm.index = pd.Index(rng.permutation(orig.index.to_numpy()), name="smiles")
        perm = perm.loc[orig.index]
        FEAT.block = lambda name, _p=perm: _p if name == TAG else orig_block(name)
        FEAT.cell_block.cache_clear()
        d = probe(bench, TAG, "log_abs_a", design, how="ridge", alpha=100.0)
        r = summarise(d, TAG, "log_abs_a", design, "ridge")["spearman"]
        null.append(r)
        print(f"  perm {i + 1}/{n_perm}: rho={r:+.3f}", flush=True)
    FEAT.block = orig_block
    FEAT.cell_block.cache_clear()

    null = np.array(null)
    p = float((np.sum(null >= obs_rho) + 1) / (len(null) + 1))
    res = {"design": design, "block": TAG, "observed_spearman": obs_rho,
           "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
           "null_q95": float(np.quantile(null, 0.95)), "null_max": float(null.max()),
           "n_permutations": int(len(null)), "p_one_sided": p}
    pd.DataFrame([res]).to_csv(OUT / f"probe_permutation_{design}.csv", index=False)
    pd.DataFrame({"null_spearman": null}).to_csv(OUT / f"probe_null_draws_{design}.csv", index=False)
    print("\n=== permutation null (ligand -> vector map shuffled) ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("\n=== estimator sweep ===")
    print(S[["estimator", "spearman", "partial_spearman_given_n_metals",
             "loco_spearman_min", "loco_spearman_max", "sd_pred"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""Is MoLFormer's out-of-fold magnitude correlation NEW, or does something cheap already have it?

``signal_probe`` found that a ridge on MoLFormer's mean-pooled vector predicts ``log|a|`` out of fold
at Spearman +0.23 under BP (partial +0.25 given n_metals, leave-one-chemotype-out stable).  BP holds
out the extractant, its whole chemotype and every publication of the test cells, so that is not
memorisation.  It is still worthless unless it beats the cheapest thing that could produce it:

* ``size4``   four rdkit numbers -- heavy atoms, molecular weight, rotatable bonds, ring count.
  A mean-pooled transformer vector encodes length; if "how big is the ligand" already gives +0.23,
  the 768 dimensions have added nothing.
* ``TOPO39``  gen14's own deployed 39 donor-topology columns, run through the identical ridge.
  If these match it, the finding is a re-discovery, not a new signal.
* ``LEAN209`` gen13's whole 209-column matrix, the largest hand-built block in the repo.

Same folds, same rich-cell training set, same weights, same estimator -- only the columns change.
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
from gen13sep.amplitude_bench import LEAN_BLOCKS                   # noqa: E402
from gen14.dirbench import feature_sets                            # noqa: E402
from gen15 import valuebench as V                                  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def size_block(smiles: list[str]) -> pd.DataFrame:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            rows.append([np.nan] * 4)
            continue
        rows.append([m.GetNumHeavyAtoms(), Descriptors.MolWt(m),
                     rdMolDescriptors.CalcNumRotatableBonds(m),
                     rdMolDescriptors.CalcNumRings(m)])
    return pd.DataFrame(rows, index=pd.Index(smiles, name="smiles"),
                        columns=["n_heavy", "mw", "n_rot", "n_ring"])


def main() -> None:
    designs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["B", "BP"]
    bench = V.load()
    smiles = sorted(set(bench.frame.extractant.astype(str)))

    # cache the size block into the same parquet-backed namespace features.py serves
    path = HERE / "size4.parquet"
    if not path.exists():
        size_block(smiles).to_parquet(path)
    orig_block = FEAT.block

    fs = feature_sets(bench)
    X_lean = bench.matrix(LEAN_BLOCKS)
    extra = {"size4": pd.read_parquet(path),
             "TOPO39": pd.DataFrame(X_lean[:, fs["TOPO39"]]).groupby(
                 bench.frame.extractant.astype(str).to_numpy()).first(),
             "LEAN209": pd.DataFrame(X_lean).groupby(
                 bench.frame.extractant.astype(str).to_numpy()).first()}
    for k, v in extra.items():
        v.index.name = "smiles"
        v.columns = [f"{k}__{i:04d}" for i in range(v.shape[1])]

    def patched(name: str) -> pd.DataFrame:
        if name in extra:
            return extra[name]
        return orig_block(name)

    FEAT.block = patched
    FEAT.cell_block.cache_clear()

    tags = ["size4", "TOPO39", "LEAN209", "molformer__mean", "chemberta_mtr__mean", "morgan2"]
    rows = []
    for tag in tags:
        for design in designs:
            d = probe(bench, tag, "log_abs_a", design, how="ridge", alpha=100.0)
            s = summarise(d, tag, "log_abs_a", design, "ridge")
            if s:
                rows.append(s)
                print(f"{tag:22s} {design:3s} rho={s['spearman']:+.3f} "
                      f"partial={s['partial_spearman_given_n_metals']:+.3f} "
                      f"loco[{s['loco_spearman_min']:+.3f},{s['loco_spearman_max']:+.3f}] "
                      f"sd_pred={s['sd_pred']:.3f}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "probe_baselines.csv", index=False)
    print("\n=== log|a| out-of-fold Spearman: embedding vs the cheap alternatives ===")
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

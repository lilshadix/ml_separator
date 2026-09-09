"""Confound checks for the embedding experiment.

Three things could make an embedding gain look real when it is not:

1. **near-duplicate ligands.**  A chemical LM maps two ligands that differ by one methylene to almost
   the same vector, so a held-out ligand with a twin in training is answered by memory.  Counted here
   at cosine >= 0.99, and -- the part that actually matters -- split by whether the twin is in the
   *same chemotype*, because the chemotype hold-out only protects against the same-chemotype case.
2. **publication identity.**  The 64 condition columns identify a cell's publication at 94 %; an
   embedding is a function of the ligand only, so it cannot leak conditions, but a ligand studied by
   one laboratory only is still a publication label in disguise under designs that are not BP.
3. **number of measured metals**, Spearman +0.49 with |a| in this corpus.  Reported as the partial
   association of the representation's magnitude prediction with |a| once n_metals is regressed out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (str(ROOT / "gen15_curve"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import features as FEAT                                  # noqa: E402
from gen15 import valuebench as V                         # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def cosine_gram(X: np.ndarray) -> np.ndarray:
    Z = X - X.mean(0, keepdims=True)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (Z / n) @ (Z / n).T


def raw_cosine_gram(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (X / n) @ (X / n).T


def duplicate_report(bench, tags: list[str], thresholds=(0.999, 0.99, 0.95)) -> pd.DataFrame:
    fr = bench.frame
    sm = sorted(set(fr.extractant.astype(str)))
    chemo = fr.drop_duplicates("extractant").set_index("extractant")["chemotype"].astype(str)
    ch = chemo.reindex(sm).to_numpy()
    amp_by_ext = (fr.assign(a=bench.coef[:, 0])
                  .groupby("extractant")["a"].mean().reindex(sm).to_numpy())
    rows = []
    for tag in tags:
        F = FEAT.block(tag).reindex(sm)
        X = F.to_numpy(dtype=float)
        for gram_name, G in (("raw", raw_cosine_gram(X)), ("centred", cosine_gram(X))):
            np.fill_diagonal(G, -np.inf)
            for th in thresholds:
                pair = G >= th
                has = pair.any(1)
                cross = np.array([bool((pair[i] & (ch != ch[i])).any()) for i in range(len(sm))])
                # of the twin pairs, how often do the two ligands share a direction of selectivity?
                ii, jj = np.where(np.triu(pair, 1))
                agree = float(np.mean(np.sign(amp_by_ext[ii]) == np.sign(amp_by_ext[jj]))) \
                    if len(ii) else float("nan")
                rows.append({"block": tag, "cosine": gram_name, "threshold": th,
                             "n_ligands": len(sm), "n_with_twin": int(has.sum()),
                             "n_with_cross_chemotype_twin": int(cross.sum()),
                             "n_pairs": int(len(ii)), "twin_direction_agreement": agree,
                             "nn_cosine_median": float(np.median(G.max(1)))})
    return pd.DataFrame(rows)


def chemotype_purity(bench, tags: list[str]) -> pd.DataFrame:
    """Does a ligand's nearest neighbour in the representation share its chemotype?

    If it always does, the chemotype hold-out has already removed every neighbour a test ligand
    could have leaned on, and the gain (if any) cannot be twin memorisation.
    """
    fr = bench.frame
    sm = sorted(set(fr.extractant.astype(str)))
    chemo = fr.drop_duplicates("extractant").set_index("extractant")["chemotype"].astype(str)
    ch = chemo.reindex(sm).to_numpy()
    rows = []
    for tag in tags:
        X = FEAT.block(tag).reindex(sm).to_numpy(dtype=float)
        G = raw_cosine_gram(X)
        np.fill_diagonal(G, -np.inf)
        nn = G.argmax(1)
        rows.append({"block": tag, "nn_same_chemotype": float((ch[nn] == ch).mean()),
                     "n_chemotypes": int(len(set(ch)))})
    return pd.DataFrame(rows)


def n_metals_confound(bench, tags: list[str]) -> pd.DataFrame:
    """Spearman of |a| with n_metals, and with a ligand-level embedding norm, for reference."""
    from scipy.stats import spearmanr
    fr = bench.frame
    a = np.abs(bench.coef[:, 0])
    nm = fr.n_metals.to_numpy(dtype=float)
    rows = [{"block": "-", "quantity": "n_metals", "spearman_with_abs_a": float(spearmanr(nm, a).statistic)}]
    ext = tuple(fr.extractant.astype(str).tolist())
    for tag in tags:
        X = FEAT.cell_block(tag, ext)
        rows.append({"block": tag, "quantity": "embedding_norm",
                     "spearman_with_abs_a": float(spearmanr(np.linalg.norm(X, axis=1), a).statistic)})
    return pd.DataFrame(rows)


def main() -> None:
    tags = sys.argv[1].split(",")
    bench = V.load()
    d = duplicate_report(bench, tags)
    p = chemotype_purity(bench, tags)
    n = n_metals_confound(bench, tags)
    d.to_csv(OUT / "confound_duplicates.csv", index=False)
    p.to_csv(OUT / "confound_nn_chemotype.csv", index=False)
    n.to_csv(OUT / "confound_n_metals.csv", index=False)
    print("=== near-duplicate ligands ===")
    print(d.round(4).to_string(index=False))
    print("\n=== nearest neighbour shares chemotype ===")
    print(p.round(4).to_string(index=False))
    print("\n=== n_metals / norm confound ===")
    print(n.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""Per-extractant feature blocks for the embedding experiment.

Two families, both a deterministic function of the SMILES alone (no label ever touches them, so
building them once for all 521 cells is not leakage -- the fold discipline lives in the estimators):

* ``emb:<tag>:<pool>``  a pretrained chemical language model's vector (built by ``encode.py``);
* ``morgan<r>``         a Morgan *count* fingerprint folded to 2048 bits, the fallback the task
  names and, more usefully, the control: a pretrained representation has to beat the cheap
  substructure count, not only the 39 hand-built topology columns.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
if str(ROOT / "generations" / "gen15_curve") not in sys.path:
    sys.path.insert(0, str(ROOT / "generations" / "gen15_curve"))

EMB_PATH = HERE / "ligand_embeddings.parquet"
MORGAN_PATH = HERE / "morgan_fp.parquet"


@lru_cache(maxsize=1)
def _emb_table() -> pd.DataFrame:
    if not EMB_PATH.exists():
        raise FileNotFoundError(EMB_PATH)
    return pd.read_parquet(EMB_PATH)


def available_embeddings() -> list[str]:
    """``<tag>__<pool>`` prefixes present in the parquet."""
    try:
        cols = _emb_table().columns
    except FileNotFoundError:
        return []
    return sorted({c.rsplit("__", 1)[0] for c in cols})


def build_morgan(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> pd.DataFrame:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            rows.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = gen.GetCountFingerprintAsNumPy(m)
        rows.append(np.asarray(fp, dtype=np.float32))
    return pd.DataFrame(np.vstack(rows), index=pd.Index(smiles, name="smiles"),
                        columns=[f"morgan{radius}__{k:04d}" for k in range(n_bits)])


@lru_cache(maxsize=4)
def _morgan_table(radius: int) -> pd.DataFrame:
    path = HERE / f"morgan{radius}_fp.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from gen15 import valuebench as V
    bench = V.load()
    sm = sorted(set(bench.frame.extractant.astype(str).tolist()))
    df = build_morgan(sm, radius=radius)
    df = df.loc[:, df.to_numpy().any(axis=0)]           # drop bits nothing sets
    df.to_parquet(path)
    return df


def block(name: str) -> pd.DataFrame:
    """A SMILES-indexed feature frame by name.

    The ``l2:`` prefix row-normalises the block first.  A mean-pooled transformer vector's norm
    tracks the token count, i.e. the size of the ligand, so an un-normalised block hands the linear
    model a molecular-weight axis dressed up as 384 dimensions; ``l2:`` removes it.
    """
    if name.startswith("l2:"):
        F = block(name[3:])
        X = F.to_numpy(dtype=float)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return pd.DataFrame(X / n, index=F.index, columns=F.columns)
    if name.startswith("morgan"):
        return _morgan_table(int(name[len("morgan"):]))
    t = _emb_table()
    cols = [c for c in t.columns if c.startswith(name + "__")]
    if not cols:
        raise KeyError(f"no columns for {name!r}; have {available_embeddings()}")
    return t[cols]


@lru_cache(maxsize=32)
def cell_block(name: str, extractant_key: tuple) -> np.ndarray:
    """The feature frame broadcast to one row per cell, aligned to ``extractant_key``."""
    F = block(name)
    return F.reindex(list(extractant_key)).to_numpy(dtype=np.float64)


def cells(ctx, name: str) -> np.ndarray:
    ext = tuple(ctx.bench.frame.extractant.astype(str).tolist())
    return cell_block(name, ext)

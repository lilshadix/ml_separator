"""Leakage helpers for round 4 (PROTOCOL.md section 8).

``poisoned_bench``: a copy of the bench in which every label-bearing field of every cell OUTSIDE the outer
training fold is replaced by random numbers on its own NaN pattern -- the log D matrix ``Y``, the cached gen13
amplitude coefficients ``coef``, and the frame's label columns (``logD__*``, ``repsd__*``,
``replicate_sd_median``).  Pre-measurement metadata (ids, chemotype, descriptor blocks, metal lists, replicate
counts) is kept.  A fit that uses only training labels is bit-identical on the poisoned bench.
"""
from __future__ import annotations

import copy

import numpy as np

LABEL_PREFIXES = ("logD__", "repsd__")
LABEL_COLUMNS = ("replicate_sd_median",)


def poisoned_bench(bench, train_index, seed: int = 20261022):
    rng = np.random.default_rng(seed)
    tr = np.asarray(train_index)
    other = np.setdiff1d(np.arange(bench.Y.shape[0]), tr)
    pb = copy.copy(bench)
    Y = np.array(bench.Y, copy=True)
    Y[other] = np.where(np.isfinite(Y[other]), rng.normal(0, 3, Y[other].shape), np.nan)
    pb.Y = Y
    coef = np.array(bench.coef, copy=True)
    coef[other] = np.where(np.isfinite(coef[other]), rng.normal(size=coef[other].shape), np.nan)
    pb.coef = coef
    fr = bench.frame.copy()
    cols = [c for c in fr.columns if c.startswith(LABEL_PREFIXES) or c in LABEL_COLUMNS]
    for c in cols:
        v = fr[c].to_numpy(float, copy=True)
        v[other] = np.where(np.isfinite(v[other]), rng.normal(size=len(other)), np.nan)
        fr[c] = v
    pb.frame = fr
    return pb, cols


def isolation_rows(bench, rec) -> list[dict]:
    """Overlap counts (cells, extractants, chemotypes, publications, gen19 copy groups) between outer training and
    test, and between every inner block's training and validation cells, plus containment of the inner blocks in
    outer training.  Required zero: cells, chemotypes and publications everywhere (both designs), copy groups
    under BPG; BP does not isolate copy groups by design (reported, not required)."""
    from gen20k.splits import copy_groups_of
    fr = bench.frame
    cg = copy_groups_of(fr)
    keys = {"cell": np.arange(len(fr)), "extractant": fr["extractant"].astype(str).to_numpy(),
            "chemotype": fr["chemotype"].astype(str).to_numpy(), "publication": fr["publication_id"].astype(str).to_numpy(),
            "copy_group": cg}
    design = rec["design"]
    otr = np.asarray(rec["outer"]["train_index"])

    def row(level, block, a, b):
        r = {"design": design, "seed": int(rec["split_seed"]), "fold": int(rec["fold"]), "level": level, "block": block,
             "n_train": int(len(a)), "n_other": int(len(b))}
        for k, v in keys.items():
            r[f"overlap_{k}"] = int(len(set(v[a].tolist()) & set(v[b].tolist())))
        need = ["cell", "chemotype", "publication"] + (["copy_group"] if design == "BPG" else [])
        r["required_zero_ok"] = all(r[f"overlap_{k}"] == 0 for k in need)
        return r
    rows = [row("outer", -1, otr, np.asarray(rec["outer"]["test_index"]))]
    for blk in rec["inner"]:
        itr, iva = np.asarray(blk["train_index"]), np.asarray(blk["val_index"])
        r = row("inner", int(blk["inner_fold"]), itr, iva)
        inside = set(itr.tolist()) <= set(otr.tolist()) and set(iva.tolist()) <= set(otr.tolist())
        r["inside_outer_train"] = bool(inside)
        r["required_zero_ok"] = bool(r["required_zero_ok"] and inside)
        rows.append(r)
    return rows


def deep_equal(a, b) -> bool:
    """Structural bit-equality of fitted objects (round 3's rule)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    if hasattr(a, "detach") and hasattr(a, "dtype") and hasattr(a, "shape"):   # torch tensor / parameter (A4): by value;
        return (type(a) is type(b) and a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape)  # vars() of a tensor is {}
                and np.array_equal(a.detach().cpu().numpy(), b.detach().cpu().numpy(), equal_nan=True))
    if hasattr(a, "__dict__") and not isinstance(a, type):
        return type(a) is type(b) and deep_equal(vars(a), vars(b))
    return bool(a == b)

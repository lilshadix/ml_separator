"""Two additions to the fold plan: the conditional design BP1, and chemotype-budget thinning.

Why
---
The programme reports five designs (A, B, BR, BQ, BP) that all answer one question -- *a new
ligand, in a new laboratory*.  BP is the deployment-relevant one and it is deliberately the
harshest: every training cell from a held-out cell's publication is dropped, so the 64 condition
columns cannot act as a laboratory fingerprint.  Nothing in the plan answers the other deployment
question a separation chemist actually has -- *this laboratory has already run this system; predict
a new condition set* -- and that is the design under which anchor/one-measurement arms are well
posed.  Roberts et al. (Ecography 40:913-929, 2017) make the general point: the blocking must match
the prediction the model is for, and reporting only the strictest block understates what the model
can do in the matched case exactly as reporting only random CV overstates it.

``bp1_folds`` builds that design.  The held-out unit is a whole **condition series** (publication x
condition_key), so no replicate of the held-out condition set can be in training, and a held-out
cell is *kept* only if at least one other cell of its own publication survives in training.  Two
strictnesses:

``BP1``   the publication is in training (same laboratory, new conditions; the ligand may or may
          not be in training).
``BP1X``  the *(publication, extractant)* pair is in training (same laboratory, same ligand, new
          conditions) -- the literal anchor case.

Both are conditional designs: the extractant appears on both sides of the split on purpose.  They
must be reported **next to** BP, never instead of it; BP1 alone is a leaky number in exactly the
sense Kapoor & Narayanan (Patterns 4:100804, 2023) catalogue.

``thin_folds`` removes whole chemotypes from a fold's training set so a learning curve can be drawn
over the number of chemically independent training units.  The test set is untouched, so the only
thing that varies along the curve is how many independent units the model saw -- the requirement
Viering & Loog (IEEE TPAMI 45:7799-7819, 2023) note is usually violated by naive learning curves.

``run_folds`` is ``gen15.valuebench.run_arms`` with the fold list supplied by the caller instead of
built from a design name, so both of the above score through the frozen gen13 metric path.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "gen13_separation", ROOT / "gen14_direction", ROOT / "gen15_curve"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep.amplitude_bench import LEAN_BLOCKS, BenchData, _pair_frame, cell_weights  # noqa: E402
from gen13sep.splits import (FOLD_SEED_OFFSET, FOLD_SEED_STRIDE, MODEL_SEED_BASE,  # noqa: E402
                             N_INNER_SPLITS, N_SPLITS, SPLIT_SEEDS, Fold)
from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402
from gen14.dirbench import feature_sets  # noqa: E402
from gen15.valuebench import Ctx, MIN_METALS  # noqa: E402

BP1_DESIGNS = ("BP1", "BP1X")


# --------------------------------------------------------------------------------------
# BP1 -- the conditional (within-publication) design
# --------------------------------------------------------------------------------------
def _series_key(frame: pd.DataFrame) -> np.ndarray:
    """The held-out unit: one condition series of one publication."""
    return (frame["publication_id"].astype(str) + "‖"
            + frame["condition_key"].astype(str)).to_numpy()


def bp1_coverage(frame: pd.DataFrame, *, min_metals: int = MIN_METALS,
                 require_extractant: bool = False) -> dict:
    """How much of the corpus can this design even score, before any fold is built."""
    rich = frame["n_metals"].to_numpy() >= min_metals
    sub = frame[rich]
    g = sub.groupby("publication_id").agg(n_series=("condition_key", "nunique"))
    ok_pub = set(g.index[g.n_series >= 2])
    m = sub["publication_id"].isin(ok_pub)
    if require_extractant:
        pe = sub.groupby(["publication_id", "extractant"])["condition_key"].nunique()
        ok_pe = set(pe.index[pe >= 2])
        m = m & pd.Series([(p, e) in ok_pe for p, e in zip(sub.publication_id, sub.extractant)],
                          index=sub.index)
    kept = sub[m]
    return {"design": "BP1X" if require_extractant else "BP1",
            "n_rich_cells": int(rich.sum()), "n_scorable_cells": int(len(kept)),
            "n_publications": int(kept.publication_id.nunique()),
            "n_extractants": int(kept.extractant.nunique()),
            "n_chemotypes": int(kept.chemotype.nunique()),
            "n_series": int(kept.groupby(["publication_id", "condition_key"]).ngroups)}


def bp1_folds(frame: pd.DataFrame, *, seeds: Sequence[int] = SPLIT_SEEDS,
              n_splits: int = N_SPLITS, n_inner: int = N_INNER_SPLITS,
              min_metals: int = MIN_METALS, require_extractant: bool = False) -> list[Fold]:
    """Leave-condition-series-out folds, test restricted to cells whose laboratory is in training.

    The grouping variable is (publication, condition_key), so a held-out condition set never has a
    replicate in training.  ``require_extractant`` additionally demands that the held-out cell's
    own (publication, extractant) pair appears in training -- design BP1X.
    """
    design = "BP1X" if require_extractant else "BP1"
    for col in ("publication_id", "condition_key", "chemotype", "extractant", "n_metals"):
        if col not in frame.columns:
            raise KeyError(f"design {design} needs a {col} column")
    series = _series_key(frame)
    pubs = frame["publication_id"].astype(str).to_numpy()
    extr = frame["extractant"].astype(str).to_numpy()
    chem = frame["chemotype"].astype(str).to_numpy()
    rich = frame["n_metals"].to_numpy() >= min_metals

    folds: list[Fold] = []
    for seed in seeds:
        for k, (train_index, test_index) in enumerate(seeded_group_kfold(series, n_splits, seed)):
            # no condition series may straddle the split
            assert not (set(series[train_index]) & set(series[test_index])), \
                f"{design} leaks a condition series"
            train_pubs = set(pubs[train_index])
            train_pairs = set(zip(pubs[train_index], extr[train_index]))
            keep = np.array([
                rich[i] and pubs[i] in train_pubs
                and (not require_extractant or (pubs[i], extr[i]) in train_pairs)
                for i in test_index], dtype=bool)
            te = test_index[keep]
            if len(te) == 0:
                continue
            inner_seed = int(seed) * 31 + k * 7919 + 17
            itr, iva = next(iter(seeded_group_kfold(chem[train_index], n_inner, inner_seed)))
            folds.append(Fold(design=design, seed=int(seed), fold=k,
                              train_index=train_index, test_index=te,
                              inner_train_index=train_index[itr],
                              inner_validation_index=train_index[iva],
                              held_out_groups=tuple(sorted(set(chem[te])))))
    if not folds:
        raise RuntimeError(f"design {design} produced no scorable folds")
    return folds


# --------------------------------------------------------------------------------------
# chemotype-budget thinning, for the learning curve
# --------------------------------------------------------------------------------------
def thin_folds(fold: Fold, groups: np.ndarray, k: int, draw: int) -> Fold | None:
    """A copy of ``fold`` whose training set keeps only ``k`` randomly chosen training chemotypes.

    The test set and every scored pair are byte-identical to the untinned fold, so two budgets are
    compared on exactly the same held-out extractants -- which is what makes the per-extractant
    difference a legitimate paired, cluster-bootstrappable quantity.
    """
    tr = fold.train_index
    chemos = np.array(sorted(set(groups[tr])))
    if k >= len(chemos):
        return fold
    rng = np.random.default_rng(int(fold.seed) * 131 + fold.fold * 7919 + k * 17 + draw)
    keep = set(rng.choice(chemos, size=int(k), replace=False).tolist())
    mask = np.array([g in keep for g in groups[tr]], dtype=bool)
    sub = tr[mask]
    if len(sub) < 20:
        return None
    n_inner = max(2, min(N_INNER_SPLITS, len(keep)))
    itr, iva = next(iter(seeded_group_kfold(groups[sub], n_inner,
                                            int(fold.seed) * 31 + fold.fold * 7919 + k)))
    return Fold(design=fold.design, seed=fold.seed, fold=fold.fold,
                train_index=sub, test_index=fold.test_index,
                inner_train_index=sub[itr], inner_validation_index=sub[iva],
                held_out_groups=fold.held_out_groups)


# --------------------------------------------------------------------------------------
# scoring with an explicit fold list
# --------------------------------------------------------------------------------------
Arm = Callable[[Ctx], np.ndarray]


def run_folds(bench: BenchData, arms: dict[str, Arm], folds: Sequence[Fold], *,
              design: str = "BP1", extra: dict | None = None,
              X: np.ndarray | None = None, fs: dict | None = None,
              verbose: bool = False) -> pd.DataFrame:
    """``gen15.valuebench.run_arms`` with the fold list supplied by the caller."""
    fs = feature_sets(bench) if fs is None else fs
    X = bench.matrix(LEAN_BLOCKS) if X is None else X
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    parts, timing = [], {k: 0.0 for k in arms}
    for f in folds:
        pairs = _pair_frame(bench.frame, bench.Y, f.test_index, f.seed, f.fold)
        if pairs.empty:
            continue
        ctx = Ctx(bench=bench, design=design, seed=f.seed, fold=f.fold, train=f.train_index,
                  test=f.test_index,
                  w=cell_weights(bench.groups[f.train_index], bench.n_obs[f.train_index]),
                  model_seed=f.model_seed, fs=fs, X=X, rich=rich)
        ia, ib, loc = pairs["ia"].to_numpy(), pairs["ib"].to_numpy(), pairs["cell_local"].to_numpy()
        t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
        for name, arm in arms.items():
            t0 = time.time()
            coef = np.asarray(arm(ctx), dtype=float).reshape(len(f.test_index), bench.basis.shape[0])
            curve = coef @ bench.basis
            t[name] = curve[loc, ia] - curve[loc, ib]
            timing[name] += time.time() - t0
        t["n_train_cells"] = len(f.train_index)
        t["n_train_chemotypes"] = len(set(bench.groups[f.train_index]))
        for kk, vv in (extra or {}).items():
            t[kk] = vv
        parts.append(t)
    if verbose:
        print("   " + "  ".join(f"{k} {v:.1f}s" for k, v in timing.items()), flush=True)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)

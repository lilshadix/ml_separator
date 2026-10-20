"""Row builder for round 3: one row per (design, seed, fold, held-out cell, budget k, strict-adjacent
query pair) with the read set, policy, data version and one prediction column per method.

Methods use gen21's interface ``fn(block, cell, obs, k) -> level vector (14,)``; ``obs`` (g21.evaluate
.Observed) exposes only the read metals' log D (``y_masked``), the gen20 star support, the cell's
metal identities and its query set.  The scored target ``y`` is read from the data version's matrix,
never passed to a method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths  # noqa: F401
from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import design as DS  # noqa: E402
from gen20k.data import LANTHANIDES, Z_VEC  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402


def seq_for(block: dict, cell: dict, policy, k=None) -> list:
    if isinstance(policy, str):
        return cell["seqs"][policy]
    if getattr(policy, "per_k", False):
        return policy(block, cell, k)
    return policy(block, cell)


def cell_rows(rec, block, cell, methods, *, policy, ks, Y, version: str, logd_sink=None, logd_arms=()):
    pol = policy if isinstance(policy, str) else getattr(policy, "name", "CUSTOM")
    rows = []
    for k in ks:
        seq = seq_for(block, cell, policy, k)
        if k > len(seq):
            continue
        obs = E.observe(Y[cell["ci"]], cell, seq, k)
        DS.assert_no_closure(DS.make_design(E._view(cell), "adjacent", "metal"), list(obs.support.pairs))
        preds = {n: np.asarray(fn(block, cell, obs, k), float) for n, fn in methods.items()}
        read = set(obs.read)
        rs = ",".join(LANTHANIDES[m] for m in sorted(read))
        if logd_sink is not None and k >= 1:
            rr = sorted(read)
            for n in logd_arms:
                v = preds[n]
                lvl = v + float(np.nanmean(obs.y_masked[rr] - v[rr]))
                for m in cell["metals"]:
                    if m not in read:
                        logd_sink.append({"design": rec["design"], "split_seed": rec["split_seed"], "fold": rec["fold"],
                                          "cell_id": cell["cell_id"], "chemotype": cell["chemotype"], "k_max": cell["k_max"],
                                          "k": k, "policy": pol, "metal": LANTHANIDES[m], "arm": n,
                                          "y": float(Y[cell["ci"]][m]), "pred": float(lvl[m])})
        for a, b in cell["query"]:
            if a in read and b in read:
                raise AssertionError("a scored pair has both ends read")
            r = {"data_version": version, "design": rec["design"], "split_seed": rec["split_seed"], "fold": rec["fold"],
                 "cell_id": cell["cell_id"], "extractant": cell["extractant"], "chemotype": cell["chemotype"],
                 "publication": cell["publication"], "n_metals": len(cell["metals"]), "model": "metal", "k_max": cell["k_max"],
                 "policy": pol, "draw": 0, "k": k, "read_set": rs, "A": LANTHANIDES[a], "B": LANTHANIDES[b], "ia": a, "ib": b,
                 "dZ": int(Z_VEC[b] - Z_VEC[a]), "regime": "Q", "a_read": a in read, "b_read": b in read, "lab_known": False,
                 "y": float(Y[cell["ci"]][a] - Y[cell["ci"]][b])}
            for n, v in preds.items():
                r[n] = float(v[a] - v[b])
            rows.append(r)
    return rows


def run_design(bench, design, methods, *, fit=None, policy="DOPT_Q", ks=(0, 1, 2, 3, 4, 5, 6), Y=None, version="V1",
               seeds=None, log=print, logd_arms=(), cells_filter=None):
    """All cached folds of one design.  ``fit(bench, rec) -> dict`` fits on the outer training fold
    only and is stored in ``block['fitted']``; returns (rows, logd rows, fitted per fold)."""
    Y = bench.Y if Y is None else Y
    seeds = list(seeds or paths.DEV_SEEDS)
    rows, logd, fitted = [], [], {}
    for f in folds_for(bench.frame, design, seeds=seeds):
        rec = C.load(f.design, f.seed, f.fold)
        E.assert_test_isolation(bench, rec)
        blk = dict(rec["outer"], Y=Y, policy=policy, _rec=rec)
        if fit is not None:
            blk["fitted"] = fit(bench, rec)
        fitted[(f.seed, f.fold)] = blk.get("fitted")
        for cell in blk["cells"]:
            if cells_filter is not None and not cells_filter(cell):
                continue
            rows += cell_rows(rec, blk, cell, methods, policy=policy, ks=ks, Y=Y, version=version, logd_sink=logd, logd_arms=logd_arms)
        if log:
            log(f"  {design} {f.seed} fold {f.fold}: {len(rows)} rows")
    return pd.DataFrame(rows), pd.DataFrame(logd), fitted


def intervention(bench, design, methods, fitted, *, policy="DOPT_Q", ks=(0, 1, 2, 3, 4, 5, 6), Y=None, seeds=None, rng_seed=20261019,
                 rec_transform=None, fold_filter=None, methods_factory=None):
    """Every metal not read at budget k, in every held-out cell, replaced by a random number with the
    reads and masks unchanged: every prediction (and the read sequence) must be bit-identical.
    fold_filter: optional set of (seed, fold) to check (smoke tests); default every fold of the seeds.
    methods_factory: optional callable(bench_like) -> methods; the scrambled pass then predicts with methods built
    from a copy of the bench whose Y is the scrambled matrix, so a predict-time closure over bench.Y is caught."""
    Y = bench.Y if Y is None else Y
    rng = np.random.default_rng(rng_seed)
    worst, n = 0.0, 0
    for f in folds_for(bench.frame, design, seeds=list(seeds or paths.DEV_SEEDS)):
        if fold_filter is not None and (f.seed, f.fold) not in fold_filter:
            continue
        rec = C.load(f.design, f.seed, f.fold)
        if rec_transform is not None:
            rec = rec_transform(rec)
        blk = dict(rec["outer"], Y=Y, policy=policy, _rec=rec)
        if fitted.get((f.seed, f.fold)) is not None:
            blk["fitted"] = fitted[(f.seed, f.fold)]
        for k in ks:
            Ys = Y.copy()
            for cell in blk["cells"]:
                if getattr(policy, "per_k", False) and cell["k_max"] < k:
                    continue
                seq = seq_for(blk, cell, policy, k)
                read = DS.reveal(Y[cell["ci"]], list(seq[:min(k, len(seq))])).read
                Ys[cell["ci"]] = E.scramble_hidden(Y[cell["ci"]], read, rng)
            b2 = dict(blk, Y=Ys)
            pol2 = policy.fresh() if hasattr(policy, "fresh") else policy   # recompute, never a memo lookup
            methods2 = methods
            if methods_factory is not None:
                import copy
                bs = copy.copy(bench)
                bs.Y = Ys
                methods2 = methods_factory(bs)
            for cell in blk["cells"]:
                if getattr(policy, "per_k", False) and cell["k_max"] < k:
                    continue
                s1, s2 = seq_for(blk, cell, policy, k), seq_for(b2, cell, pol2, k)
                if list(s1) != list(s2):
                    return {"design": design, "passed": False, "reason": f"read sequence moved: {cell['cell_id']} k {k}"}
                r1 = cell_rows(rec, blk, cell, methods, policy=policy, ks=(k,), Y=Y, version="x")
                r2 = cell_rows(rec, b2, cell, methods2, policy=pol2, ks=(k,), Y=Ys, version="x")
                for a, b in zip(r1, r2):
                    for m in methods:
                        if np.isnan(a[m]) != np.isnan(b[m]):
                            return {"design": design, "passed": False, "reason": f"NaN pattern {m} {cell['cell_id']} k {k}"}
                        if np.isfinite(a[m]):
                            worst = max(worst, abs(a[m] - b[m]))
                            n += 1
    return {"design": design, "n_predictions": n, "max_abs_change": worst, "passed": worst == 0.0}

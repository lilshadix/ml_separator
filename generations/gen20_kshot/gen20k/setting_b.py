"""Setting B (exploratory only): experimental-condition adaptation inside one extractant.

The Setting B audit (AUDIT.md B.7) found that within one (extractant, publication) group the
adjacent separation factor barely moves between conditions (median change 0.043 log units, 91 %
below the 0.3 replicate floor), so copying the value measured at another condition already scores
0.105 against FLAT 0.160, and that the eligible population is 23 non-duplicate units / 17
extractants / 11 chemotypes with 64 % of the query pairs in one chemotype.  Setting B is therefore
run as an exploratory arm with the COPY null, never as a registered claim.

Unit: an (extractant, publication) group with >= 2 cells sharing >= 1 strict-adjacent pair, after
removing the five identical-log-D cell pairs listed in ``DUPLICATE_CELL_PAIRS`` (double
digitisations split only by a NaN condition).  For every query cell the support cell is the other
cell of the group with the most shared adjacent pairs (ties: nearest log acid concentration, then
cell_id).  The support's measurements are all pairs of the support cell; the query is the query
cell's adjacent pairs that the support cell also measures (so COPY is defined for every query).
Arms: COPY (support cell's value), FLAT, each zero-shot arm on the query cell, each arm + BLUP20
with the support cell's pairs treated as measurements of the query cell, LINE through the
support pairs.  Folds: the design's folds; the query cell must be held out (its support cell, being
the same extractant, is held out with it, which is what makes the support a 'measurement').
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import adapters as AD
from . import arms as AR
from . import cov as CV
from . import design as DS
from .data import LANTHANIDES, Z_VEC, cell_view, is_strict_adjacent, pair_value
from .splits import folds_for

DUPLICATE_CELL_PAIRS: tuple[tuple[str, str], ...] = (
    ("5a6a2f3bf13a7d55", "c61ecb085d8df371"), ("2e7197a7efd242c9", "1052c7804a84fd4c"),
    ("5df28757679cae26", "f8ed87d3cf7f4281"), ("247f57741141188a", "62e7cdd6fa003391"),
    ("170f4449e685df34", "80831f6aeb20c04c"))


def units(bench) -> list[dict]:
    """Eligible (extractant, publication) groups and, per query cell, its support cell."""
    f = bench.frame
    dup = {c for pr in DUPLICATE_CELL_PAIRS for c in pr}
    acid = pd.to_numeric(f["cond__acid_concentration_M"], errors="coerce").to_numpy(float)
    out = []
    for (ext, pub), g in f.groupby(["extractant", "publication_id"]):
        idx = [int(i) for i in g.index if str(f.cell_id.iat[i]) not in dup or True]
        # drop the second member of each identical pair (keep the first) so a unit never scores a copy of itself
        keep = []
        for i in idx:
            cid = str(f.cell_id.iat[i])
            if any(cid == b_ and str(f.cell_id.iat[j]) == a_ for a_, b_ in DUPLICATE_CELL_PAIRS for j in keep):
                continue
            keep.append(i)
        if len(keep) < 2:
            continue
        adj = {i: {(a, b) for a, b in cell_view(bench, i).pairs if is_strict_adjacent(a, b)} for i in keep}
        members = []
        for q in keep:
            cands = [(len(adj[q] & adj[s]), -abs(np.log10(max(acid[s], 1e-6)) - np.log10(max(acid[q], 1e-6))) if np.isfinite(acid[s]) and np.isfinite(acid[q]) else -9.0,
                      -int(hash_order(str(f.cell_id.iat[s]))), s) for s in keep if s != q and adj[q] & adj[s]]
            if not cands:
                continue
            cands.sort(reverse=True)
            s = cands[0][3]
            members.append({"query": q, "support": s, "query_pairs": sorted(adj[q] & adj[s])})
        if members:
            out.append({"extractant": ext, "publication": str(pub), "chemotype": str(f.chemotype.iat[keep[0]]),
                        "cells": keep, "members": members})
    return out


def hash_order(s: str) -> int:
    import hashlib
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def run(bench, design: str, seeds, log=print) -> pd.DataFrame:
    us = units(bench)
    q2unit = {m["query"]: (u, m) for u in us for m in u["members"]}
    rows = []
    for fold in folds_for(bench.frame, design, seeds=list(seeds)):
        ctx = CV.make_ctx(bench, fold, fold.train_index, fold.test_index)
        facts = AR.standard_arms()
        fitted = {n: facts[n]().fit(ctx) for n in ("FLAT", "G14", "MEAN_CURVE")}
        cov = CV.fold_covariance(bench, fold, AR.as_fit_curves(facts["G14"]))
        s2 = cov.noise_s2
        for ci in fold.test_index:
            ci = int(ci)
            if ci not in q2unit:
                continue
            u, m = q2unit[ci]
            s = m["support"]
            if s not in set(int(i) for i in fold.test_index):
                raise AssertionError("support cell is not held out with its query cell")
            sv = cell_view(bench, s)
            sup = DS.reveal(bench.Y[s], sv.pairs)
            R = CV.noise_matrix(sup.pairs, s2)
            preds = {"COPY": None}
            curves = {n: a.curves([ci])[0] for n, a in fitted.items()}
            for n, c in curves.items():
                preds[f"{n}@ZERO"] = c
                preds[f"{n}@BLUP20"] = AD.blup(c, cov.cov, sup, R)
            preds["LINE"] = AD.naive_line(bench.basis, sup)
            for a, b in m["query_pairs"]:
                rec = {"design": design, "split_seed": fold.seed, "fold": fold.fold, "unit": f"{u['extractant']}@@{u['publication']}",
                       "extractant": u["extractant"], "chemotype": u["chemotype"], "publication": u["publication"],
                       "cell_id": str(bench.frame.cell_id.iat[ci]), "support_cell_id": str(bench.frame.cell_id.iat[s]),
                       "n_support_pairs": sup.k, "A": LANTHANIDES[a], "B": LANTHANIDES[b], "dZ": int(Z_VEC[b] - Z_VEC[a]),
                       "y": pair_value(bench.Y[ci], a, b), "COPY": pair_value(bench.Y[s], a, b)}
                for n, c in preds.items():
                    if c is not None:
                        rec[n] = float(c[a] - c[b])
                rows.append(rec)
        if log:
            log(f"  setting B {design} seed {fold.seed} fold {fold.fold}: {len(rows)} rows so far")
    return pd.DataFrame(rows)

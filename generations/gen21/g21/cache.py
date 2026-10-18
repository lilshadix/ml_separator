"""Per-fold context cache: everything gen20's metal-model DOPT_Q evaluation fixes before a method
runs, rebuilt with gen20's own frozen functions so the RIDGE2 baseline is reproduced, not
re-implemented.

For one outer fold (design, split seed, fold) the cache holds

* ``outer``: the gen20 selection record (noise factor, ridge lambda; chosen on the inner split,
  NNC arms off, which returns before any neural fit), the G14 reference covariance and noise
  ``s2``, the RIDGE2 prior, and for every held-out cell its metals, fixed strict-adjacent query,
  ``k_max``, the DOPT_Q and UNIFORM support sequences (label-free) and the G14 / FLAT zero-shot
  curves;
* ``inner``: a list, one entry per inner chemotype fold of the training cells (gen13's inner seed;
  inner-training cells with the design mask re-applied, inner-validation cells, nested covariance),
  holding the same objects, so a gen21 method can fit its own free parameters on out-of-fold
  training cells only.  gen20's noise factor and ridge lambda are reused unchanged there.

No held-out label is stored here: labels are read later through ``g21.reveal`` only.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import replace

import numpy as np

from . import paths
from gen20k import arms as AR  # noqa: E402
from gen20k import cov as CV  # noqa: E402
from gen20k import design as DS  # noqa: E402
from gen20k import runner as RN  # noqa: E402
from gen20k.data import cell_view  # noqa: E402
from gen20k.splits import N_INNER, folds_for, inner_seed  # noqa: E402
from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402

KMAX = 6                         # the metal model's maximum on a 14-metal cell
CACHE_VERSION = 2


def cache_path(design: str, seed: int, fold: int):
    return paths.CACHE / f"{design}_{seed}_{fold}.pkl"


def _cells_block(bench, cells, cov, nv, *, fold_seed, zero_g14, zero_flat, min_k=1):
    out = []
    for ci in cells:
        cv = cell_view(bench, int(ci))
        dz = DS.make_design(cv, "adjacent", "metal")
        if not dz.query or dz.k_max < min_k:
            continue
        kmax = min(dz.k_max, KMAX)
        seqs = {"DOPT_Q": DS.support_sequence(dz, "DOPT_Q", kmax, cov=cov, noise_var=nv),
                "UNIFORM": DS.support_sequence(dz, "UNIFORM", kmax)}
        out.append({"ci": int(ci), "cell_id": cv.cell_id, "extractant": cv.extractant, "chemotype": cv.chemotype,
                    "publication": cv.publication, "metals": tuple(cv.metals), "query": tuple(dz.query),
                    "k_max": int(dz.k_max), "seqs": seqs,
                    "zero_g14": np.asarray(zero_g14[int(ci)], float), "zero_flat": np.asarray(zero_flat[int(ci)], float)})
    return out


def build_fold(bench, fold) -> dict:
    t0 = time.time()
    cfg = RN.RunConfig(label="g21_cache", designs=(fold.design,), arms=("FLAT", "G14"), models=("metal",))
    facts = AR.standard_arms()
    # ---- outer (gen20 run_fold, metal model, NNC off) ----
    ctx = CV.make_ctx(bench, fold, fold.train_index, fold.test_index)
    g14 = facts["G14"]().fit(ctx)
    flat = facts["FLAT"]().fit(ctx)
    ref = CV.fold_covariance(bench, fold, AR.as_fit_curves(facts["G14"]))
    CV.assert_cov_provenance(ref, fold.train_index)
    sel = RN.select_on_inner(bench, fold, cfg)
    s2 = ref.noise_s2 * sel["noise_factor"]
    test = [int(c) for c in fold.test_index]
    zg = dict(zip(test, g14.curves(test)))
    zf = dict(zip(test, flat.curves(test)))
    outer = {"s2": float(s2), "noise_factor": float(sel["noise_factor"]), "ridge_lambda": float(sel["ridge_lambda"]),
             "prior": np.asarray(g14.prior_coef(), float), "cov": ref.cov, "noise_s2_robust": float(ref.noise_s2),
             "train_index": np.asarray(fold.train_index), "test_index": np.asarray(fold.test_index),
             "cells": _cells_block(bench, test, ref.cov, 2 * s2, fold_seed=fold.seed, zero_g14=zg, zero_flat=zf),
             "selection_scores": sel["scores"]}
    # ---- inner cross-fit: every inner chemotype fold of the training cells (design mask re-applied,
    #      nested covariance from the inner-training part's own inner cross-fits) ----
    inner = []
    tr = np.asarray(fold.train_index)
    groups = bench.groups[tr].astype(str)
    for j, (tr_rel, va_rel) in enumerate(seeded_group_kfold(groups, N_INNER, inner_seed(fold.seed, fold.fold))):
        iva = tr[va_rel]
        itr = CV.inner_mask(bench, fold.design, tr[tr_rel], iva)
        if set(map(int, itr)) & set(map(int, iva)):
            raise AssertionError("inner-training and inner-validation cells overlap")
        if len(itr) < 20 or len(iva) == 0:
            continue
        ictx = CV.make_ctx(bench, fold, itr, iva)
        ifold = replace(fold, train_index=itr)
        ig14 = facts["G14"]().fit(ictx)
        iflat = facts["FLAT"]().fit(ictx)
        icov = CV.fold_covariance(bench, ifold, AR.as_fit_curves(facts["G14"]))
        CV.assert_cov_provenance(icov, itr)
        is2 = icov.noise_s2 * sel["noise_factor"]
        ivl = [int(c) for c in iva]
        izg = dict(zip(ivl, ig14.curves(ivl)))
        izf = dict(zip(ivl, iflat.curves(ivl)))
        inner.append({"inner_fold": j, "s2": float(is2), "prior": np.asarray(ig14.prior_coef(), float), "cov": icov.cov,
                      "ridge_lambda": float(sel["ridge_lambda"]), "train_index": np.asarray(itr), "val_index": iva,
                      "cells": _cells_block(bench, ivl, icov.cov, 2 * is2, fold_seed=fold.seed, zero_g14=izg, zero_flat=izf)})
    return {"version": CACHE_VERSION, "design": fold.design, "split_seed": int(fold.seed), "fold": int(fold.fold),
            "outer": outer, "inner": inner, "seconds": time.time() - t0}


def build_all(bench, designs, seeds, *, log=print, workers: int = 1) -> list:
    paths.CACHE.mkdir(parents=True, exist_ok=True)
    jobs = []
    for d in designs:
        for f in folds_for(bench.frame, d, seeds=list(seeds)):
            p = cache_path(f.design, f.seed, f.fold)
            if p.exists():
                continue
            jobs.append(f)
    log(f"{len(jobs)} folds to build")
    for j, f in enumerate(jobs):
        rec = build_fold(bench, f)
        with open(cache_path(f.design, f.seed, f.fold), "wb") as fh:
            pickle.dump(rec, fh)
        log(f"  [{j + 1}/{len(jobs)}] {f.design} {f.seed} {f.fold}: {len(rec['outer']['cells'])} test / "
            f"{len(rec['inner']['cells'])} inner cells, {rec['seconds']:.0f}s")
    return jobs


def load(design: str, seed: int, fold: int) -> dict:
    with open(cache_path(design, seed, fold), "rb") as fh:
        return pickle.load(fh)

"""Stage-0 frozen objects for new data (CLAUDE_CODE_PROMPT.md section 4.3 items 9 and 12).

* ``all_cells_fold``: one gen13 ``Fold`` whose outer ``train_index`` is every bench cell and whose test set is empty.
  Its single inner split (gen20's noise-factor selection) is dealt by ``gen20k.splits._redeal_inner`` (BPG's own
  dealing, identical to gen13's for BP: chemotype groups, ``N_INNER`` blocks, first block) with the declared inner seed
  ``gen20k.splits.inner_seed(104729, 0)``, i.e. the fold carries seed 104729 and fold number 0.  With every cell
  training, BP and BPG share the outer training set; they differ only in the publication / copy-group mask that
  ``gen20k.cov.inner_mask`` re-applies inside, so the inner blocks, covariances and lambda selection depend on design.
* ``build_fold_record``: ``g21.cache.build_fold`` copied line for line (outer part = ``outer_block``, inner cross-fit
  blocks = ``inner_blocks``) with one change: the zero-shot G14 / FLAT curves of the outer test cells are computed only
  when there are test cells (gen15's arms cannot predict an empty index).  The record has no ``seconds`` field, so a
  rebuild pickles to the same bytes.  On a dev fold it reproduces the stored gen21 cache (tests/test_frozen.py).
* ``fit_beta``: ``r4.refs.fit_reference(bench, rec, Y, 'B1', k=3)`` on the all-cell record; the inner-CV value on the
  BPG record is beta_frozen (item 9; sensitivities: design BP, lambda fixed at 0.03, the round-4 fold mean).
* ``external_dopt`` / ``external_sequence`` / ``read_set``: the external read policy (item 12).  C_all and s2_all are
  build_fold's outer covariance and noise variance with the whole corpus training (all 229 C3 cells included; the
  outer block of the BPG all-cell record), so a new cell is read along
  ``support_sequence(dz, 'DOPT_Q', min(k_max, 6), cov=C_all, noise_var=2 * s2_all)``; r reads = union of seq[:r - 1].

No held-out label exists here (no fold has test cells); ``fit_beta`` reads training labels only, as fit_reference does.
Every computation runs with one BLAS thread (``single_thread``), so a rerun is bit-identical on any core count.
"""
# build_fold_record / outer_block / inner_blocks are copied from generations/gen21/g21/cache.py
# (LF SHA-256 96becffc03b1d3e6b282476dae20361dac2f7f781e8d53e5f81e226684526d6d); fit_beta calls
# round4_20261022/r4/refs.py fit_reference unchanged (LF SHA-256 2ab162ca47b854985380751bee0bbff1560d4fb36f8c9637ce3e1c08d6196eee).
from __future__ import annotations

import functools
import hashlib
import json
import math
import pickle
import re
import time
from dataclasses import replace

import numpy as np
from threadpoolctl import threadpool_limits

from . import paths
from r4 import refs as RF  # noqa: E402
from g21 import cache as GC  # noqa: E402
from gen13sep.splits import Fold  # noqa: E402
from gen20k import arms as AR  # noqa: E402
from gen20k import cov as CV  # noqa: E402
from gen20k import design as DS  # noqa: E402
from gen20k import runner as RN  # noqa: E402
from gen20k.data import CellView, cell_view  # noqa: E402
from gen20k.splits import N_INNER, _redeal_inner, copy_groups_of, inner_seed  # noqa: E402
from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402

# Copy rule (brief 1.2): every copied or re-implemented source with its LF SHA-256 at the copy; checked by
# scripts/r5_seal.py (copied_sources reads this declaration).
COPIED_FROM = {
    "generations/gen21/g21/cache.py":
        "96becffc03b1d3e6b282476dae20361dac2f7f781e8d53e5f81e226684526d6d",
    "generations/gen21/round4_20261022/r4/refs.py":
        "2ab162ca47b854985380751bee0bbff1560d4fb36f8c9637ce3e1c08d6196eee",
}

paths.redirect_inner_cache()

G21_CACHE_LF_SHA256 = "96becffc03b1d3e6b282476dae20361dac2f7f781e8d53e5f81e226684526d6d"
R4_REFS_LF_SHA256 = "2ab162ca47b854985380751bee0bbff1560d4fb36f8c9637ce3e1c08d6196eee"
DECLARED_SEED, DECLARED_FOLD = 104729, 0          # inner seed = gen20k.splits.inner_seed(104729, 0) = 3246616
PRIMARY_DESIGN = "BPG"
K_B1 = 3                                          # code budget k = 3, i.e. four reads (section 3.3)
LAMBDA_FIXED = 0.03                               # the value round 4 chose in 23 of 25 folds (PC P5)
KMAX = GC.KMAX                                    # 6
EXT_VERSION = 1
PICKLE_PROTOCOL = 4
BLAS_THREADS = 1      # the last bits of the B1 ridge fit depend on the OpenBLAS thread count (measured: beta moves by
#                       up to 7e-17 between 1, 2, 3, 4 and 8 threads), so every frozen object is computed single-threaded
FROZEN_DIR = paths.CACHE / "frozen"
EXT_DIR = paths.CACHE / "ext"
EXT_PATH = EXT_DIR / "ext_dopt.pkl"
STAGE0 = paths.RES / "stage0"


def single_thread(fn):
    """Run ``fn`` with BLAS_THREADS OpenBLAS / OpenMP threads (see BLAS_THREADS)."""
    @functools.wraps(fn)
    def wrapped(*a, **kw):
        with threadpool_limits(limits=BLAS_THREADS):
            return fn(*a, **kw)
    return wrapped


# ------------------------------------------------------------------------------------------
# the all-cell fold and its record (g21.cache.build_fold)
# ------------------------------------------------------------------------------------------
def all_cells_fold(bench, design: str = PRIMARY_DESIGN, *, train_index=None, seed: int = DECLARED_SEED,
                   fold: int = DECLARED_FOLD) -> Fold:
    """Outer train = every bench cell (or ``train_index``), no test cell; inner split as gen20k.splits._redeal_inner."""
    if design not in ("BP", "BPG"):
        raise KeyError(design)
    tr = np.arange(len(bench.frame)) if train_index is None else np.sort(np.asarray(train_index, dtype=int))
    e = np.zeros(0, dtype=tr.dtype)
    f0 = Fold(design=design, seed=int(seed), fold=int(fold), train_index=tr, test_index=e, inner_train_index=e,
              inner_validation_index=e, held_out_groups=())
    return _redeal_inner(bench.frame, f0, tr)


@single_thread
def outer_block(bench, fold) -> tuple[dict, dict]:
    """build_fold's outer part (gen20 run_fold, metal model, NNC off): G14 covariance from the training fold's inner
    cross-fits, noise factor and ridge lambda from gen20's inner selection, s2 = robust noise x factor."""
    cfg = RN.RunConfig(label="g21_cache", designs=(fold.design,), arms=("FLAT", "G14"), models=("metal",))
    facts = AR.standard_arms()
    ctx = CV.make_ctx(bench, fold, fold.train_index, fold.test_index)
    g14 = facts["G14"]().fit(ctx)
    flat = facts["FLAT"]().fit(ctx)
    ref = CV.fold_covariance(bench, fold, AR.as_fit_curves(facts["G14"]))
    CV.assert_cov_provenance(ref, fold.train_index)
    sel = RN.select_on_inner(bench, fold, cfg)
    s2 = ref.noise_s2 * sel["noise_factor"]
    test = [int(c) for c in fold.test_index]
    zg = dict(zip(test, g14.curves(test))) if test else {}      # the one change: no zero-shot call on an empty index
    zf = dict(zip(test, flat.curves(test))) if test else {}
    outer = {"s2": float(s2), "noise_factor": float(sel["noise_factor"]), "ridge_lambda": float(sel["ridge_lambda"]),
             "prior": np.asarray(g14.prior_coef(), float), "cov": ref.cov, "noise_s2_robust": float(ref.noise_s2),
             "train_index": np.asarray(fold.train_index), "test_index": np.asarray(fold.test_index),
             "cells": GC._cells_block(bench, test, ref.cov, 2 * s2, fold_seed=fold.seed, zero_g14=zg, zero_flat=zf),
             "selection_scores": sel["scores"]}
    return outer, sel


@single_thread
def inner_blocks(bench, fold, sel: dict) -> list:
    """build_fold's inner cross-fit: every inner chemotype fold of the training cells (design mask re-applied, nested
    covariance from the inner-training part's own inner cross-fits); gen20's noise factor and lambda reused."""
    facts = AR.standard_arms()
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
                      "cells": GC._cells_block(bench, ivl, icov.cov, 2 * is2, fold_seed=fold.seed, zero_g14=izg,
                                               zero_flat=izf)})
    return inner


def build_fold_record(bench, fold) -> tuple[dict, float]:
    """(record in g21.cache's format without ``seconds``, seconds)."""
    t0 = time.time()
    outer, sel = outer_block(bench, fold)
    inner = inner_blocks(bench, fold, sel)
    rec = {"version": GC.CACHE_VERSION, "design": fold.design, "split_seed": int(fold.seed), "fold": int(fold.fold),
           "outer": outer, "inner": inner}
    return rec, time.time() - t0


def dump_bytes(obj) -> bytes:
    return pickle.dumps(obj, protocol=PICKLE_PROTOCOL)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def record_path(design: str, tag: str = "all"):
    return FROZEN_DIR / f"fold_{tag}_{design}_{DECLARED_SEED}_{DECLARED_FOLD}.pkl"


def load_or_build(bench, design: str = PRIMARY_DESIGN, *, tag: str = "all", train_index=None, rebuild: bool = False,
                  log=print) -> tuple[dict, dict]:
    """The all-cell record of ``design`` from cache/frozen/, built (and pickled) when missing.  ``rebuild`` rebuilds
    and requires the bytes to equal the stored pickle (never overwrites a differing one)."""
    p = record_path(design, tag)
    if p.exists() and not rebuild:
        b = p.read_bytes()
        return pickle.loads(b), {"path": p.relative_to(paths.R5).as_posix(), "sha256": sha256_bytes(b), "status": "loaded",
                                 "seconds": None}
    rec, secs = build_fold_record(bench, all_cells_fold(bench, design, train_index=train_index))
    b = dump_bytes(rec)
    if p.exists():
        if p.read_bytes() != b:
            raise AssertionError(f"rebuilt record differs from the stored {p.name}; not overwritten")
        status = "rebuilt, bit-identical to the stored pickle"
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
        status = "built"
    log(f"  record {design}/{tag}: {status} ({secs:.0f}s) -> {p}")
    return pickle.loads(b), {"path": p.relative_to(paths.R5).as_posix(), "sha256": sha256_bytes(b), "status": status,
                             "seconds": round(secs, 1)}


def record_checks(bench, rec: dict, train_index=None) -> dict:
    """Construction checks (no label): the outer training set is every intended cell and nothing is held out; every
    training cell is inner-validated exactly once; inner blocks inside the training set with the design mask applied;
    every C3 cell has a nested inner record; orphans."""
    n = len(bench.frame)
    want = np.arange(n) if train_index is None else np.sort(np.asarray(train_index, dtype=int))
    out = rec["outer"]
    tr = np.asarray(out["train_index"])
    key = (bench.frame["publication_id"].astype(str).to_numpy() if rec["design"] == "BP"
           else copy_groups_of(bench.frame))
    val = np.concatenate([np.asarray(b["val_index"]) for b in rec["inner"]]) if rec["inner"] else np.zeros(0, int)
    mask_leaks = sum(len(set(key[np.asarray(b["train_index"])]) & set(key[np.asarray(b["val_index"])]))
                     for b in rec["inner"])
    inner_cells = [c["ci"] for b in rec["inner"] for c in b["cells"]]
    c3 = c3_cells(bench)
    rep = {"n_bench_cells": int(n), "n_train": int(len(tr)), "train_is_intended_set": bool(np.array_equal(tr, want)),
           "n_test": int(len(out["test_index"])), "n_outer_cell_records": len(out["cells"]),
           "n_inner_blocks": len(rec["inner"]), "inner_block_sizes": [[int(len(b["train_index"])), int(len(b["val_index"]))]
                                                                      for b in rec["inner"]],
           "every_train_cell_validated_once": bool(len(val) == len(tr) and np.array_equal(np.sort(val), np.sort(tr))),
           "inner_inside_train": bool(all(set(map(int, b["train_index"])) <= set(map(int, tr)) for b in rec["inner"])),
           "inner_mask_leaks": int(mask_leaks), "n_inner_cell_records": len(inner_cells),
           "inner_cell_records_unique": len(inner_cells) == len(set(inner_cells)),
           "n_c3_cells": len(c3), "c3_cells_with_inner_record": len(set(c3) & set(inner_cells)),
           "orphan_cells": len(RF.orphan_cells(bench, rec)),
           "inner_seed": inner_seed(rec["split_seed"], rec["fold"])}
    ok = (rep["train_is_intended_set"] and rep["n_test"] == 0 and rep["n_outer_cell_records"] == 0
          and rep["inner_inside_train"] and rep["inner_mask_leaks"] == 0 and rep["inner_cell_records_unique"])
    if train_index is None:
        ok = ok and rep["every_train_cell_validated_once"] and rep["c3_cells_with_inner_record"] == len(c3)
    rep["ok"] = bool(ok)
    return rep


def compare(a, b, path: str = "") -> tuple[list, float]:
    """(structural mismatches, max |a - b| over numeric leaves) of two nested records."""
    if isinstance(a, dict) and isinstance(b, dict):
        bad, mx = ([f"{path}: keys {sorted(set(a) ^ set(b), key=str)}"] if set(a) != set(b) else []), 0.0
        for k in a:
            if k in b:
                bb, m = compare(a[k], b[k], f"{path}/{k}")
                bad += bb
                mx = max(mx, m)
        return bad, mx
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"], 0.0
        bad, mx = [], 0.0
        for i, (x, y) in enumerate(zip(a, b)):
            bb, m = compare(x, y, f"{path}[{i}]")
            bad += bb
            mx = max(mx, m)
        return bad, mx
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        if x.shape != y.shape:
            return [f"{path}: shape {x.shape} != {y.shape}"], 0.0
        if x.dtype.kind in "fc" or y.dtype.kind in "fc":
            xf, yf = x.astype(float), y.astype(float)
            if not np.array_equal(np.isnan(xf), np.isnan(yf)):
                return [f"{path}: NaN pattern"], 0.0
            d = np.abs(np.nan_to_num(xf) - np.nan_to_num(yf))
            return [], float(d.max()) if d.size else 0.0
        return ([] if np.array_equal(x, y) else [f"{path}: values"]), 0.0
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if math.isnan(a) and math.isnan(b):
                return [], 0.0
            if math.isinf(a) or math.isinf(b):
                return ([] if a == b else [f"{path}: {a} != {b}"]), 0.0
            return [], abs(float(a) - float(b))
    return ([] if a == b else [f"{path}: {a!r} != {b!r}"]), 0.0


# ------------------------------------------------------------------------------------------
# beta_frozen (item 9)
# ------------------------------------------------------------------------------------------
def _lam(x):
    return "inf" if isinstance(x, float) and math.isinf(x) else float(x)


@single_thread
def fit_beta(bench, rec: dict, *, lam_grid=RF.LAMBDA_GRID, k: int = K_B1) -> dict:
    """B1's Gd amplitude by r4.refs.fit_reference (inner-CV lambda over ``lam_grid``, then refit on every training
    situation of the record)."""
    t0 = time.time()
    fit = RF.fit_reference(bench, rec, bench.Y, "B1", k=k, lam_grid=lam_grid)
    return {"beta": float(fit["coef"][0]), "lambda": _lam(fit["lambda"]), "lambda_grid": [_lam(x) for x in lam_grid],
            "inner_scores": {str(kk): float(v) for kk, v in fit["inner_scores"].items()}, "k": int(k),
            "n_situations": int(fit["n_situations"]), "n_rows": int(fit["n_rows"]), "coverage": fit["coverage"],
            "seconds": round(time.time() - t0, 1)}


def round4_fold_betas(design: str) -> dict:
    """Round 4's stored per-fold B1 fits (results/REF/fitted_<design>.json): mean, sd, range, lambda counts."""
    d = json.loads((paths.R4RES / "REF" / f"fitted_{design}.json").read_bytes())
    keys = sorted(k for k in d if re.fullmatch(r"\d+_\d+", k))
    b = np.array([float(d[k]["ref:B1"]["coef"][0]) for k in keys])
    lam = [str(d[k]["ref:B1"]["lambda"]) for k in keys]
    return {"source": f"round4_20261022/results/REF/fitted_{design}.json",
            "source_lf_sha256": paths.sha256_text_lf(paths.R4RES / "REF" / f"fitted_{design}.json"),
            "n_folds": len(keys), "mean": float(b.mean()), "sd": float(b.std(ddof=1)), "min": float(b.min()),
            "max": float(b.max()), "lambda_counts": {x: lam.count(x) for x in sorted(set(lam))}}


# ------------------------------------------------------------------------------------------
# the external DOPT_Q read policy (item 12)
# ------------------------------------------------------------------------------------------
def c3_cells(bench) -> list[int]:
    """Label-free (metal sets only): bench cells with a strict-adjacent query and k_max >= 3."""
    out = []
    for ci in range(len(bench.frame)):
        dz = DS.make_design(cell_view(bench, ci), "adjacent", "metal")
        if dz.query and dz.k_max >= 3:
            out.append(ci)
    return out


def external_dopt(rec: dict, *, record_sha256: str | None = None) -> dict:
    """C_all, s2_all (and 2 * s2_all, the DOPT_Q noise variance) from the outer block of an all-cell record."""
    o = rec["outer"]
    return {"version": EXT_VERSION, "policy": "DOPT_Q", "model": "metal", "query_name": "adjacent", "kmax_cap": KMAX,
            "design": rec["design"], "split_seed": int(rec["split_seed"]), "fold": int(rec["fold"]),
            "inner_seed": inner_seed(rec["split_seed"], rec["fold"]), "C_all": np.asarray(o["cov"], float).copy(),
            "s2_all": float(o["s2"]), "noise_var": 2.0 * float(o["s2"]), "noise_factor": float(o["noise_factor"]),
            "noise_s2_robust": float(o["noise_s2_robust"]), "n_train": int(len(o["train_index"])),
            "train_index": np.asarray(o["train_index"]).copy(), "record_sha256": record_sha256}


def design_for_metals(metals, *, cell_id: str = "external") -> "DS.CellDesign":
    """The strict-adjacent metal-model design of a new cell from its measured metal indices (no value needed)."""
    cv = CellView(index=-1, cell_id=str(cell_id), extractant="", chemotype="", publication="",
                  metals=tuple(sorted(int(m) for m in metals)))
    return DS.make_design(cv, "adjacent", "metal")


@single_thread
def external_sequence(ext: dict, dz) -> list[tuple[int, int]]:
    """support_sequence(dz, 'DOPT_Q', min(k_max, 6), cov=C_all, noise_var=2 * s2_all)."""
    if not dz.query:
        raise ValueError("cell has no strict-adjacent query")
    return DS.support_sequence(dz, "DOPT_Q", min(dz.k_max, ext["kmax_cap"]), cov=ext["C_all"],
                               noise_var=ext["noise_var"])


def read_set(seq, reads: int) -> frozenset:
    """r reads = the union of the first r - 1 sequence steps (section 3.3: 4 reads = seq[:3], 3 reads = seq[:2])."""
    if reads < 2 or reads - 1 > len(seq):
        raise ValueError(f"{reads} reads need {reads - 1} sequence steps; the sequence has {len(seq)}")
    return frozenset(m for p in seq[:reads - 1] for m in p)


@single_thread
def corpus_sequences(bench, cells, cov, noise_var) -> dict:
    """{cell_id: DOPT_Q sequence} of bench cells under (cov, noise_var), as external_sequence computes it."""
    out = {}
    for ci in cells:
        dz = DS.make_design(cell_view(bench, ci), "adjacent", "metal")
        out[str(bench.frame.cell_id.iat[ci])] = DS.support_sequence(dz, "DOPT_Q", min(dz.k_max, KMAX), cov=cov,
                                                                     noise_var=noise_var)
    return out


def build_external(bench, rec: dict, *, record_sha256: str | None) -> dict:
    """The pickled external object: external_dopt(rec) plus the label-free sequences of the corpus C3 cells."""
    ext = external_dopt(rec, record_sha256=record_sha256)
    ext["corpus_c3_sequences"] = corpus_sequences(bench, c3_cells(bench), ext["C_all"], ext["noise_var"])
    return ext


def load_external(path=EXT_PATH) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)

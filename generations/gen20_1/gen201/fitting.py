"""gen20.1 fitting: one (design, split seed, fold, variant) unit -> the long log D and log SF prediction tables.

IMPLEMENTATION_PLAN.md §3-§4 and PRE_REGISTRATION.md §4; build contract "Wave-2 interfaces".

``run_unit(bench, design, fold, *, variant, folds_ctx)``:

1. training cells = the Fold's (masked) ``train_index``; variant ``nodmd`` removes the 5 DMDPhPDA cells from it,
   from every inner set (``folds.inner_folds(..., drop=...)``, never re-dealt) and from the scored rows;
2. the 4-fold masked chemotype inner cross-fit (``folds.inner_folds``; skipped folds flagged): every
   configuration K1..K4 of the raw RowForest (A), the centred RowForest (C) and the COND RowForest (BD_COND) is
   fitted on each non-skipped inner-training set and predicts its inner-validation cells; pooled inner OOF scores:
   A and BD_COND MET-01 (extractant-macro MAE of log D over every observed metal of every inner-validation cell),
   C MET-20 (the same on within-cell-centred log D, centring over the cell's observed metals after prediction),
   B_selSF / C_selSF MET-01 of inner strict-adjacent log SF; argmin with ties |delta| < 1e-12 to the lowest K,
   all inner folds skipped -> K1;
3. refits on the whole outer training set: A_K1..A_K4 and C_K1..C_K4 (the "not selectable" appendix), BD_COND at
   its selected K, BD_LEVEL (A's K without metal columns), A_PRIME (A's K, seed + 1; BP primary only), C_PAIR
   (designs BP and B, primary only), the mean / NN / pair baselines and the gen20 references;
4. predictions for every test cell: log D rows = every observed metal; log SF rows = every gen13-adjacent pair
   (strict |dZ| = 1, plus Nd-Sm, flagged); every arm column must be finite on every row (else ``CoverageError``).

B = A_hat[c,a] - A_hat[c,b] from A's selected model (no re-selection); C pair = c_hat[c,a] - c_hat[c,b];
C_ACFG = centred with A's configuration; B_selSF / C_selSF use the configuration selected on inner strict-adjacent
log SF MET-01 (B_selSF_raw = that raw model's log D).  BL_D_INNER / BL_SF_INNER choices are recorded (EX-11).

Nothing here reads an outer-test label: test cells contribute their feature rows, and which metals they measured is
taken from ``UnitContext.obs`` (built once from the unmasked bench), so a bench whose test-cell labels are masked
gives identical predictions (LT-11).  The ``y`` column is copied from ``bench.Y`` for the scorer only.

``run_design`` writes predictions/<label>/<design>/<variant>/s<seed>_f<fold>_{logd,logsf}.parquet,
``_selection.json``, ``_inner_{logd,logsf}.parquet`` and finally ``_DONE.json`` (digests + code digest + the arm set,
tree counts and library versions the unit was produced with); every file goes to ``<name>.tmp`` first and is moved
into place with ``os.replace`` (DONE last), so a crash never leaves a torn DONE marker.  A unit whose DONE marker
matches the current code, arm set, tree counts and library versions is skipped (resumable).  One process at a time
writes a (label, design): ``run_design`` holds ``<root>/<label>/<design>/.lock`` (pid + process start time; a lock
whose process is gone is stale and taken over).  ``CODE_AT_IMPORT`` is the code digest of the files this process
imported; ``run_design`` and ``write_unit`` refuse (``CodeChanged``) once the files on disk differ from it.
``load_tables`` concatenates the units of one (label, design, variant) after verifying every DONE marker and that
all units share one code digest, arm set, tree counts and library versions.

Smoke (S8) failures of the centred learner C or of C_PAIR are recordable: a non-finite C-family / C_PAIR column
raises ``ArmFailure`` (a ``CoverageError`` naming the arm), and any exception inside a C or C_PAIR fit or prediction
is re-raised as ``ArmFailure`` of that learner, so ``scripts/g201_smoke.py`` can mark the learner "not run" (PRE_REG
§4.2, §4.5, O11) and repeat the smoke run without it.
"""
from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import bootstrap
from . import baselines as BL
from . import data as D
from . import folds as FO
from . import learners as L

N_LN = D.N_LN
VARIANTS: tuple[str, ...] = ("primary", "nodmd")
CPAIR_DESIGNS: tuple[str, ...] = ("BP", "B")
APRIME_DESIGNS: tuple[str, ...] = ("BP",)
SELECTED_ARMS: tuple[str, ...] = ("A", "C", "BD_COND", "B_selSF", "C_selSF")
LOGD_KEYS: tuple[str, ...] = ("design", "variant", "split_seed", "fold", "ci", "cell_id", "extractant", "chemotype",
                              "publication_id", "n_metals", "metal", "im", "y")
LOGSF_KEYS: tuple[str, ...] = ("design", "variant", "split_seed", "fold", "ci", "cell_id", "extractant", "chemotype",
                               "publication_id", "n_metals", "a", "b", "ia", "ib", "dZ", "pair", "strict", "gen13adj", "y")
H1_MEMBERS: tuple[str, ...] = ("BD_MEAN", "BD_METAL", "BD_COND", "BD_NN")
SF_INNER_MEMBERS: tuple[str, ...] = ("FLAT", "PAIRMEAN")
KS = L.CONFIG_ORDER

#: files whose content defines the predictions (code digest of a unit; scoring-side modules excluded)
CODE_FILES: tuple[str, ...] = ("gen201/__init__.py", "gen201/bootstrap.py", "gen201/data.py", "gen201/env.py",
                               "gen201/folds.py", "gen201/guard.py", "gen201/run.py", "gen201/seal.py",
                               "gen201/learners.py", "gen201/baselines.py", "gen201/fitting.py",
                               "scripts/g201_discovery.py")


#: the arm columns of the centred learner C (logd and logsf use the same names); C_PAIR is its own learner
C_FAMILY_ARMS: tuple[str, ...] = ("C", "C_K1", "C_K2", "C_K3", "C_K4", "C_ACFG", "C_selSF")
#: the learners that run only "after a passing smoke test" (PRE_REGISTRATION.md §4.2, §4.5, O11)
SMOKE_GATED: tuple[str, ...] = ("C", "C_PAIR")


class CoverageError(AssertionError):
    """A scored row lacks a finite prediction, or the scored rows differ from the row manifests (LT-16)."""


def arm_family(arm: str) -> str:
    """The learner an arm column belongs to: "C" for every C-family column, otherwise the arm itself."""
    return "C" if arm in C_FAMILY_ARMS else str(arm)


class ArmFailure(CoverageError):
    """One arm failed: a non-finite prediction column, or an exception inside its fit / prediction (``__cause__``).
    ``family`` names the learner ("C" for every C-family column), so S8 can record a C / C_PAIR failure."""

    def __init__(self, arm: str, message: str):
        super().__init__(message)
        self.arm = str(arm)
        self.family = arm_family(self.arm)


class CodeChanged(RuntimeError):
    """The code files on disk differ from the ones this process imported (STOP: restart the run)."""


class DesignLocked(RuntimeError):
    """Another live process is writing this (label, design)."""


#: exceptions inside a C / C_PAIR fit that are never turned into a recordable ArmFailure (environment / protocol)
_NOT_ARM_FAILURES = (MemoryError, L.LeakageError, ArmFailure)


def _arm_call(arm: str, fn, *args, **kwargs):
    """``fn(*args, **kwargs)``; any exception except memory exhaustion and leakage becomes ``ArmFailure(arm)``."""
    try:
        return fn(*args, **kwargs)
    except _NOT_ARM_FAILURES:
        raise
    except Exception as e:
        raise ArmFailure(arm, f"{arm}: fit/prediction failed: {e!r}") from e


def logd_arm_columns(design: str, variant: str, *, include_c: bool = True) -> list[str]:
    cols = ["A", *[f"A_K{k[1]}" for k in KS]]
    if design in APRIME_DESIGNS and variant == "primary":
        cols.append("A_PRIME")
    cols += ["BD_MEAN", "BD_METAL", "BD_COND", "BD_NN", "BD_LEVEL", "BD_LEVEL_CURVE"]
    if include_c:
        cols += ["C", *[f"C_K{k[1]}" for k in KS], "C_ACFG", "C_selSF"]
    cols.append("B_selSF_raw")
    return cols


def logsf_arm_columns(design: str, variant: str, *, include_c: bool = True, include_c_pair: bool = True) -> list[str]:
    cols = ["B", *[f"B_K{k[1]}" for k in KS], "B_selSF"]
    if include_c:
        cols += ["C", *[f"C_K{k[1]}" for k in KS], "C_ACFG", "C_selSF"]
    if include_c_pair and design in CPAIR_DESIGNS and variant == "primary":
        cols.append("C_PAIR")
    cols += ["FLAT", "PAIRMEAN", "G14_ZERO", "MEAN_CURVE_REF", "NN_TANIMOTO_REF", "HEAVIER_ALWAYS",
             "BD_METAL_SF", "BD_COND_SF", "BD_NN_SF", "BD_LEVEL_CURVE_SF"]
    return cols


# --------------------------------------------------------------------------------------
# context (label-free, built once per bench)
# --------------------------------------------------------------------------------------
@dataclass
class UnitContext:
    """Label-free arrays shared by every unit: features, fingerprints, measured-metal mask, the full row tables
    (every observed entry and every gen13-adjacent pair of every cell), DMDPhPDA cells, row manifests (lazy)."""
    X_lean: np.ndarray
    X_cond: np.ndarray
    ecfp: np.ndarray
    obs: np.ndarray
    groups: np.ndarray
    extractants: np.ndarray
    publications: np.ndarray
    chemotypes: np.ndarray
    full_logd: pd.DataFrame
    full_logsf: pd.DataFrame
    dmd: np.ndarray
    c3: np.ndarray
    manifests: dict = field(default_factory=dict)

    def manifest(self, name: str) -> pd.DataFrame | None:
        if name not in self.manifests:
            p = bootstrap.MANIFESTS / "row_manifests" / f"{name}.parquet"
            self.manifests[name] = pd.read_parquet(p) if p.exists() else None
        return self.manifests[name]


def make_context(bench) -> UnitContext:
    """Built from the unmasked bench (label-free: features, NaN masks, identity columns)."""
    f = bench.frame
    D.assert_feature_columns(D.lean_columns(bench))
    D.assert_feature_columns(bench.columns(D.COND_BLOCKS))
    allc = np.arange(len(f))
    full_logd = D.logd_rows(bench, allc)
    full_logsf = D.logsf_rows(bench, allc, query="adjacent_gen13")
    return UnitContext(X_lean=D.X_lean(bench), X_cond=D.X_cond(bench), ecfp=D.ecfp(bench), obs=D.observed(bench).copy(),
                       groups=np.asarray(bench.groups).astype(str), extractants=f["extractant"].astype(str).to_numpy(),
                       publications=f["publication_id"].astype(str).to_numpy(),
                       chemotypes=f["chemotype"].astype(str).to_numpy(),
                       full_logd=full_logd, full_logsf=full_logsf, dmd=D.dmd_cells(bench), c3=D.c3_cells(bench))


# --------------------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------------------
def assert_split(bench, design: str, train, test, ctx: UnitContext) -> None:
    """LT-10: no test cell in training; no chemotype (B family) / extractant (A) crossing; no publication crossing
    under BP/BPG; no copy-group crossing under BPG."""
    tr, te = set(np.asarray(train, dtype=int).tolist()), set(np.asarray(test, dtype=int).tolist())
    if tr & te:
        raise L.LeakageError(f"{len(tr & te)} test cells in the training set")
    trl, tel = sorted(tr), sorted(te)
    if design == "A":
        if set(ctx.extractants[trl]) & set(ctx.extractants[tel]):
            raise L.LeakageError("extractant crossing under design A")
    elif set(ctx.chemotypes[trl]) & set(ctx.chemotypes[tel]):
        raise L.LeakageError(f"chemotype crossing under design {design}")
    if design in ("BP", "BPG") and set(ctx.publications[trl]) & set(ctx.publications[tel]):
        raise L.LeakageError(f"publication crossing under design {design}")
    if design == "BPG":
        from gen20k.splits import copy_groups_of
        g = copy_groups_of(bench.frame)
        if set(g[trl]) & set(g[tel]):
            raise L.LeakageError("copy-group crossing under design BPG")


def assert_inner(design: str, inner: FO.InnerFold, train, ctx: UnitContext) -> None:
    """LT-10/LT-18: inner-training and inner-validation cells disjoint, inside the outer training set, chemotype
    grouped, publication-masked under BP/BPG."""
    itr, iva = set(inner.train_pos.tolist()), set(inner.val_pos.tolist())
    trs = set(np.asarray(train, dtype=int).tolist())
    if itr & iva:
        raise L.LeakageError("an inner-validation cell is in the inner-training set")
    if not (itr | iva) <= trs:
        raise L.LeakageError("an inner set leaves the outer training set")
    if itr and iva and set(ctx.chemotypes[sorted(itr)]) & set(ctx.chemotypes[sorted(iva)]):
        raise L.LeakageError("chemotype crossing between inner training and inner validation")
    if design in ("BP", "BPG") and itr and iva and set(ctx.publications[sorted(itr)]) & set(ctx.publications[sorted(iva)]):
        raise L.LeakageError("publication crossing between inner training and inner validation")


# --------------------------------------------------------------------------------------
# small numerics
# --------------------------------------------------------------------------------------
def macro_mae(units, abs_err) -> float:
    """Extractant-macro MAE: mean over units of each unit's mean |e| (MET-01 form); NaN when empty."""
    abs_err = np.asarray(abs_err, dtype=np.float64)
    if abs_err.size == 0:
        return float("nan")
    names, code = np.unique(np.asarray(units).astype(str), return_inverse=True)
    s = np.bincount(code, weights=abs_err, minlength=len(names))
    n = np.bincount(code, minlength=len(names))
    return float(np.mean(s / n))


def _rmse(e) -> float:
    e = np.asarray(e, dtype=np.float64)
    return float(np.sqrt(np.mean(e * e))) if e.size else float("nan")


def _centre_over(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """values minus their mean within each group (cell)."""
    _, code = np.unique(groups, return_inverse=True)
    s = np.bincount(code, weights=values)
    n = np.bincount(code)
    return values - (s / n)[code]


def peak_rss() -> dict:
    """Process working set and its peak (bytes); POSIX: ru_maxrss."""
    try:
        if os.name == "nt":
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            fn = k32.K32GetProcessMemoryInfo
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
            if fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return {"rss": int(c.WorkingSetSize), "peak_rss": int(c.PeakWorkingSetSize)}
            return {"rss": None, "peak_rss": None}
        import resource
        return {"rss": None, "peak_rss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024}
    except Exception:  # pragma: no cover
        return {"rss": None, "peak_rss": None}


class _Timer:
    def __init__(self):
        self.family: dict[str, float] = {}
        self.fits: list[dict] = []

    def add(self, family: str, seconds: float) -> None:
        self.family[family] = self.family.get(family, 0.0) + float(seconds)

    def fit(self, family: str, config: str, where: str, n_cells: int, n_rows: int, seconds: float) -> None:
        self.add(family, seconds)
        self.fits.append({"family": family, "config": config, "where": where, "n_cells": int(n_cells),
                          "n_rows": int(n_rows), "seconds": round(float(seconds), 4), **peak_rss()})


# --------------------------------------------------------------------------------------
# the unit
# --------------------------------------------------------------------------------------
@dataclass
class UnitResult:
    logd: pd.DataFrame
    logsf: pd.DataFrame
    selection: dict
    inner_oof: dict[str, pd.DataFrame]


def _select_test_cells(test, ctx: UnitContext, max_test_cells: int | None) -> np.ndarray:
    te = np.sort(np.asarray(test, dtype=int))
    if max_test_cells is None or len(te) <= max_test_cells:
        return te
    c3 = set(ctx.c3.tolist())
    first = [c for c in te if c in c3] + [c for c in te if c not in c3]
    return np.sort(np.asarray(first[: int(max_test_cells)], dtype=int))


def _fit_predict(learner, X, Y, groups, train, cells, timer: _Timer, family: str, where: str):
    t0 = time.perf_counter()
    learner.fit(X, Y, groups, train)
    P = learner.predict(X, cells)
    timer.fit(family, learner.config, where, len(train), learner.n_rows_, time.perf_counter() - t0)
    learner.discard()
    return P


def _inner_rows(Y: np.ndarray, val: np.ndarray, ctx: UnitContext):
    """(row cell positions in val, metals, y) of every observed entry of the inner-validation cells, and the strict
    pairs (positions, ia, ib, y)."""
    Yv = Y[val]
    pos, mi = np.nonzero(np.isfinite(Yv))
    ys = Yv[pos, mi]
    sf = ctx.full_logsf
    sf = sf[sf["strict"].to_numpy() & np.isin(sf["ci"].to_numpy(), val)]
    loc = {int(c): i for i, c in enumerate(val)}
    ppos = np.array([loc[int(c)] for c in sf["ci"]], dtype=int)
    ia, ib = sf["ia"].to_numpy(dtype=int), sf["ib"].to_numpy(dtype=int)
    ysf = Yv[ppos, ia] - Yv[ppos, ib] if len(ppos) else np.zeros(0)
    return pos, mi, ys, ppos, ia, ib, ysf


def run_unit(bench, design: str, fold, *, variant: str = "primary", folds_ctx: UnitContext | None = None,
             include_c: bool = True, include_c_pair: bool = True, max_test_cells: int | None = None,
             label: str | None = None) -> UnitResult:
    """Fit every arm of one (design, seed, fold, variant) unit and predict every scored row of its test cells.

    ``include_c`` / ``include_c_pair`` are switched off only when the S8 smoke test failed for that learner (the arm
    is then "not run" and its columns are absent); ``max_test_cells`` restricts the scored test cells (smoke only)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if fold.design != design:
        raise ValueError(f"fold design {fold.design} != {design}")
    t_unit = time.perf_counter()
    restores0 = L.FILTER_RESTORES
    ctx = folds_ctx if folds_ctx is not None else make_context(bench)
    Y = np.asarray(bench.Y, dtype=np.float64)
    G = ctx.groups
    seed = int(fold.model_seed)
    timer = _Timer()
    drop = ctx.dmd if variant == "nodmd" else np.zeros(0, dtype=int)
    train = np.asarray(fold.train_index, dtype=int)
    train = train[~np.isin(train, drop)]
    test_all = np.asarray(fold.test_index, dtype=int)
    test_all = test_all[~np.isin(test_all, drop)]
    assert_split(bench, design, train, test_all, ctx)
    test = _select_test_cells(test_all, ctx, max_test_cells)
    if np.isin(test, ctx.dmd).any() and variant == "nodmd":  # pragma: no cover - defensive
        raise L.LeakageError("DMDPhPDA cell scored under COH-NODMD")
    run_cpair = bool(include_c_pair and design in CPAIR_DESIGNS and variant == "primary")
    run_aprime = bool(design in APRIME_DESIGNS and variant == "primary")

    # ---------------- inner cross-fit -------------------------------------------------------
    inner = FO.inner_folds(bench, design, fold, drop=drop if len(drop) else None)
    for inn in inner:
        assert_inner(design, inn, train, ctx)
    active = [inn for inn in inner if not inn.skipped]
    pooled ={"ext_d": [], "ext_sf": [], "ci_d": [], "im": [], "y_d": [], "ci_sf": [], "ia": [], "ib": [], "y_sf": [],
              "inner_d": [], "inner_sf": []}
    oof_d = {f"{fam}_{k}": [] for fam in ("raw", "cen", "cond") for k in KS}
    oof_sf = {f"{fam}_{k}": [] for fam in ("raw", "cen") for k in KS}
    base_d = {m: [] for m in ("BD_MEAN", "BD_METAL", "BD_NN")}
    base_sf = {"PAIRMEAN": []}
    for inn in active:
        itr, iva = inn.train_pos, inn.val_pos
        pos, mi, ys, ppos, ia, ib, ysf = _inner_rows(Y, iva, ctx)
        cells_d = iva[pos]
        pooled["ext_d"].append(ctx.extractants[cells_d]); pooled["ci_d"].append(cells_d); pooled["im"].append(mi)
        pooled["y_d"].append(ys); pooled["inner_d"].append(np.full(len(ys), inn.j))
        cells_sf = iva[ppos]
        pooled["ext_sf"].append(ctx.extractants[cells_sf]); pooled["ci_sf"].append(cells_sf)
        pooled["ia"].append(ia); pooled["ib"].append(ib); pooled["y_sf"].append(ysf)
        pooled["inner_sf"].append(np.full(len(ysf), inn.j))
        for k in KS:
            Pr = _fit_predict(L.RowForest("raw", k, seed), ctx.X_lean, Y, G, itr, iva, timer, "A", f"inner{inn.j}")
            oof_d[f"raw_{k}"].append(Pr[pos, mi])
            oof_sf[f"raw_{k}"].append(Pr[ppos, ia] - Pr[ppos, ib])
            if include_c:
                Pc = _arm_call("C", _fit_predict, L.RowForest("centred", k, seed), ctx.X_lean, Y, G, itr, iva, timer,
                               "C", f"inner{inn.j}")
                oof_d[f"cen_{k}"].append(Pc[pos, mi])
                oof_sf[f"cen_{k}"].append(Pc[ppos, ia] - Pc[ppos, ib])
            Pd = _fit_predict(L.RowForest("raw", k, seed), ctx.X_cond, Y, G, itr, iva, timer, "BD_COND", f"inner{inn.j}")
            oof_d[f"cond_{k}"].append(Pd[pos, mi])
        t0 = time.perf_counter()
        mb = BL.MeanBaselines().fit(Y, G, itr)
        nn = BL.NearestNeighbour().fit(Y, G, itr, fp=ctx.ecfp, extractants=ctx.extractants, means=mb)
        Pnn, _ = nn.predict(iva)
        base_d["BD_MEAN"].append(mb.predict_mean(len(iva))[pos, mi])
        base_d["BD_METAL"].append(mb.predict_metal(len(iva))[pos, mi])
        base_d["BD_NN"].append(Pnn[pos, mi])
        pm = BL.PairMean().fit(Y, G, itr)
        base_sf["PAIRMEAN"].append(pm.predict(ia, ib)[0])
        timer.add("baselines_inner", time.perf_counter() - t0)

    cat = lambda v, dt=np.float64: (np.concatenate(v).astype(dt) if v else np.zeros(0, dtype=dt))  # noqa: E731
    ext_d, ext_sf = cat(pooled["ext_d"], object), cat(pooled["ext_sf"], object)
    ci_d, im_d, y_d = cat(pooled["ci_d"], int), cat(pooled["im"], int), cat(pooled["y_d"])
    ci_sf, ia_sf, ib_sf, y_sf = cat(pooled["ci_sf"], int), cat(pooled["ia"], int), cat(pooled["ib"], int), cat(pooled["y_sf"])
    ytil = _centre_over(y_d, ci_d) if len(y_d) else y_d
    scores: dict[str, dict] = {a: {} for a in SELECTED_ARMS}
    rmse: dict[str, dict] = {a: {} for a in SELECTED_ARMS}
    for k in KS:
        pr = cat(oof_d[f"raw_{k}"])
        scores["A"][k] = macro_mae(ext_d, np.abs(y_d - pr)); rmse["A"][k] = _rmse(y_d - pr)
        pdd = cat(oof_d[f"cond_{k}"])
        scores["BD_COND"][k] = macro_mae(ext_d, np.abs(y_d - pdd)); rmse["BD_COND"][k] = _rmse(y_d - pdd)
        psf = cat(oof_sf[f"raw_{k}"])
        scores["B_selSF"][k] = macro_mae(ext_sf, np.abs(y_sf - psf)); rmse["B_selSF"][k] = _rmse(y_sf - psf)
        if include_c:
            pc = cat(oof_d[f"cen_{k}"])
            pct = _centre_over(pc, ci_d) if len(pc) else pc
            scores["C"][k] = macro_mae(ext_d, np.abs(ytil - pct)); rmse["C"][k] = _rmse(ytil - pct)
            pcs = cat(oof_sf[f"cen_{k}"])
            scores["C_selSF"][k] = macro_mae(ext_sf, np.abs(y_sf - pcs)); rmse["C_selSF"][k] = _rmse(y_sf - pcs)
    criteria = {"A": "MET-01 log D (extractant-macro MAE, every observed metal of every inner-validation cell)",
                "C": "MET-20 (extractant-macro MAE of within-cell-centred log D, centred over the cell's observed metals)",
                "BD_COND": "MET-01 log D (as A)",
                "B_selSF": "MET-01 of inner strict-adjacent log SF (raw RowForest differences)",
                "C_selSF": "MET-01 of inner strict-adjacent log SF (centred RowForest differences)"}
    chosen, sel_rec = {}, {}
    skipped = [int(inn.j) for inn in inner if inn.skipped]
    for arm in SELECTED_ARMS:
        if arm in ("C", "C_selSF") and not include_c:
            continue
        k, why = L.select_config(scores[arm]) if active else ("K1", "all four inner folds skipped: K1")
        chosen[arm] = k
        n_rows = int(len(y_sf) if arm.endswith("selSF") else len(y_d))
        sel_rec[arm] = {"criterion": criteria[arm], "scores": scores[arm], "inner_rmse_unused": rmse[arm],
                        "chosen": k, "reason": why, "n_inner_val_cells": int(sum(len(i.val_pos) for i in active)),
                        "n_inner_rows": n_rows, "skipped_inner_folds": skipped,
                        "all_inner_skipped": not active}

    # BL_D_INNER / BL_SF_INNER (EX-11)
    bl_scores = {m: macro_mae(ext_d, np.abs(y_d - cat(base_d[m]))) for m in ("BD_MEAN", "BD_METAL", "BD_NN")}
    bl_scores["BD_COND"] = scores["BD_COND"][chosen["BD_COND"]] if active else float("nan")
    bl_scores = {m: bl_scores[m] for m in H1_MEMBERS}
    bld, bld_why = L.select_config(bl_scores, order=H1_MEMBERS)
    sf_scores = {"FLAT": macro_mae(ext_sf, np.abs(y_sf)), "PAIRMEAN": macro_mae(ext_sf, np.abs(y_sf - cat(base_sf["PAIRMEAN"])))}
    bls, bls_why = L.select_config(sf_scores, order=SF_INNER_MEMBERS)

    # ---------------- outer refits ----------------------------------------------------------
    A_K, C_K = {}, {}
    for k in KS:
        A_K[k] = _fit_predict(L.RowForest("raw", k, seed), ctx.X_lean, Y, G, train, test, timer, "A", "outer")
        if include_c:
            C_K[k] = _arm_call("C", _fit_predict, L.RowForest("centred", k, seed), ctx.X_lean, Y, G, train, test, timer,
                               "C", "outer")
    kA = chosen["A"]
    P_cond = _fit_predict(L.RowForest("raw", chosen["BD_COND"], seed), ctx.X_cond, Y, G, train, test, timer, "BD_COND", "outer")
    P_level = _fit_predict(L.bd_level(kA, seed), ctx.X_lean, Y, G, train, test, timer, "BD_LEVEL", "outer")
    P_prime = (_fit_predict(L.a_prime(kA, seed), ctx.X_lean, Y, G, train, test, timer, "A_PRIME", "outer")
               if run_aprime else None)
    t0 = time.perf_counter()
    mb = BL.MeanBaselines().fit(Y, G, train)
    P_mean, P_metal = mb.predict_mean(len(test)), mb.predict_metal(len(test))
    nn = BL.NearestNeighbour().fit(Y, G, train, fp=ctx.ecfp, extractants=ctx.extractants, means=mb)
    P_nn, FB_nn = nn.predict(test)
    mc = BL.mean_curve(Y, G, train)
    P_lcurve = P_level + mc[None, :]
    pm = BL.PairMean().fit(Y, G, train)
    ha = BL.HeavierAlways().fit(Y, G, train)
    timer.add("baselines_outer", time.perf_counter() - t0)
    t0 = time.perf_counter()
    refs = BL.gen20_ref_curves(bench, fold, train, test)
    timer.add("gen20_refs", time.perf_counter() - t0)

    # ---------------- long tables -----------------------------------------------------------
    loc = np.full(len(ctx.obs), -1, dtype=int)
    loc[test] = np.arange(len(test))
    ld = ctx.full_logd[np.isin(ctx.full_logd["ci"].to_numpy(), test)].reset_index(drop=True)
    ls = ctx.full_logsf[np.isin(ctx.full_logsf["ci"].to_numpy(), test)].reset_index(drop=True)
    r, m = loc[ld["ci"].to_numpy()], ld["im"].to_numpy(dtype=int)
    rs, ia, ib = loc[ls["ci"].to_numpy()], ls["ia"].to_numpy(dtype=int), ls["ib"].to_numpy(dtype=int)

    def head(t, extra):
        out = pd.DataFrame({"design": design, "variant": variant, "split_seed": int(fold.seed), "fold": int(fold.fold)},
                           index=t.index)
        for c in t.columns:
            out[c] = t[c].to_numpy()
        for k_, v in extra.items():
            out[k_] = v
        return out

    logd = head(ld, {"y": Y[ld["ci"].to_numpy(), m]})
    logsf = head(ls, {"y": Y[ls["ci"].to_numpy(), ia] - Y[ls["ci"].to_numpy(), ib]})
    logd = logd[list(LOGD_KEYS)].copy()
    logsf = logsf[list(LOGSF_KEYS)].copy()
    dcols, scols = {}, {}
    dcols["A"] = A_K[kA][r, m]
    for k in KS:
        dcols[f"A_K{k[1]}"] = A_K[k][r, m]
    if run_aprime:
        dcols["A_PRIME"] = P_prime[r, m]
    dcols["BD_MEAN"], dcols["BD_METAL"] = P_mean[r, m], P_metal[r, m]
    dcols["BD_COND"], dcols["BD_NN"] = P_cond[r, m], P_nn[r, m]
    dcols["BD_LEVEL"], dcols["BD_LEVEL_CURVE"] = P_level[r, m], P_lcurve[r, m]
    if include_c:
        dcols["C"] = C_K[chosen["C"]][r, m]
        for k in KS:
            dcols[f"C_K{k[1]}"] = C_K[k][r, m]
        dcols["C_ACFG"] = C_K[kA][r, m]
        dcols["C_selSF"] = C_K[chosen["C_selSF"]][r, m]
    dcols["B_selSF_raw"] = A_K[chosen["B_selSF"]][r, m]

    pdiff = BL.pair_diff
    scols["B"] = pdiff(A_K[kA], rs, ia, ib)
    for k in KS:
        scols[f"B_K{k[1]}"] = pdiff(A_K[k], rs, ia, ib)
    scols["B_selSF"] = pdiff(A_K[chosen["B_selSF"]], rs, ia, ib)
    if include_c:
        scols["C"] = pdiff(C_K[chosen["C"]], rs, ia, ib)
        for k in KS:
            scols[f"C_K{k[1]}"] = pdiff(C_K[k], rs, ia, ib)
        scols["C_ACFG"] = pdiff(C_K[kA], rs, ia, ib)
        scols["C_selSF"] = pdiff(C_K[chosen["C_selSF"]], rs, ia, ib)
    cpair_rec = None
    if run_cpair:
        t0 = time.perf_counter()
        ci_glob = ls["ci"].to_numpy(dtype=int)

        def _cpair():
            m_ = L.CPair(seed).fit(ctx.X_lean, Y, G, train)
            return m_, m_.predict_pairs(ctx.X_lean, ci_glob, ia, ib), m_.predict_pairs(ctx.X_lean, ci_glob, ib, ia)

        cp, f_ab, f_ba = _arm_call("C_PAIR", _cpair)
        dt = time.perf_counter() - t0
        timer.fit("C_PAIR", "C_PAIR", "outer", len(train), cp.n_pairs_ * 2, dt)
        cpair_rec = {"n_train_pair_rows": int(cp.n_pairs_ * 2), "n_estimators": cp.n_estimators,
                     "antisymmetric": bool(np.array_equal(f_ab, -f_ba)),
                     "max_abs_antisymmetry_residual": float(np.max(np.abs(f_ab + f_ba))) if len(f_ab) else 0.0}
        scols["C_PAIR"] = f_ab
        cp.discard()
    scols["FLAT"] = np.zeros(len(ls))
    P_pm, FB_pm = pm.predict(ia, ib)
    scols["PAIRMEAN"] = P_pm
    scols["G14_ZERO"] = pdiff(refs["G14"], rs, ia, ib)
    scols["MEAN_CURVE_REF"] = pdiff(refs["MEAN_CURVE"], rs, ia, ib)
    scols["NN_TANIMOTO_REF"] = pdiff(refs["NN_TANIMOTO"], rs, ia, ib)
    scols["HEAVIER_ALWAYS"] = ha.predict(ia, ib)
    scols["BD_METAL_SF"] = pdiff(P_metal, rs, ia, ib)
    scols["BD_COND_SF"] = pdiff(P_cond, rs, ia, ib)
    scols["BD_NN_SF"] = pdiff(P_nn, rs, ia, ib)
    scols["BD_LEVEL_CURVE_SF"] = pdiff(P_lcurve, rs, ia, ib)

    want_d = logd_arm_columns(design, variant, include_c=include_c)
    want_s = logsf_arm_columns(design, variant, include_c=include_c, include_c_pair=include_c_pair)
    if set(want_d) != set(dcols) or set(want_s) != set(scols):  # pragma: no cover - defensive
        raise AssertionError(f"arm columns differ from the contract: {set(want_d) ^ set(dcols)} {set(want_s) ^ set(scols)}")
    logd = pd.concat([logd, pd.DataFrame({c: np.asarray(dcols[c], dtype=np.float64) for c in want_d})], axis=1)
    logsf = pd.concat([logsf, pd.DataFrame({c: np.asarray(scols[c], dtype=np.float64) for c in want_s})], axis=1)
    for c, dt in (("split_seed", "int64"), ("fold", "int64"), ("ci", "int64"), ("n_metals", "int64"), ("im", "int64")):
        logd[c] = logd[c].astype(dt)
    for c, dt in (("split_seed", "int64"), ("fold", "int64"), ("ci", "int64"), ("n_metals", "int64"), ("ia", "int64"),
                  ("ib", "int64"), ("dZ", "int64")):
        logsf[c] = logsf[c].astype(dt)
    logd["y"] = logd["y"].astype(np.float64)
    logsf["y"] = logsf["y"].astype(np.float64)
    assert_finite(logd, want_d, "logd")
    assert_finite(logsf, want_s, "logsf")
    cov = check_coverage(logd, logsf, design, fold, variant, test, ctx, restricted=len(test) < len(test_all))

    # ---------------- inner OOF tables and the selection record -----------------------------
    inner_logd = pd.DataFrame({"inner": cat(pooled["inner_d"], int), "ci": ci_d, "im": im_d, "y": y_d})
    for key, v in oof_d.items():
        if v:
            inner_logd[key] = cat(v)
    for key, v in base_d.items():
        inner_logd[key] = cat(v)
    inner_logsf = pd.DataFrame({"inner": cat(pooled["inner_sf"], int), "ci": ci_sf, "ia": ia_sf, "ib": ib_sf, "y": y_sf})
    for key, v in oof_sf.items():
        if v:
            inner_logsf[key] = cat(v)
    inner_logsf["FLAT"] = np.zeros(len(y_sf))
    inner_logsf["PAIRMEAN"] = cat(base_sf["PAIRMEAN"])

    fb_metal = [[int(c), int(x)] for c, x in zip(ld["ci"], m) if mb.metal_fallback[x]]
    fb_nn_any = FB_nn[r, m] > 0
    n_best = np.array([len(b) for b in nn.neighbours(test)], dtype=int)
    fb_nn_all = FB_nn[r, m] == n_best[r]
    selection = {
        "label": label, "design": design, "variant": variant, "split_seed": int(fold.seed), "fold": int(fold.fold),
        "model_seed": seed, "n_estimators": L.N_ESTIMATORS, "c_pair_n_estimators": L.C_PAIR_N_ESTIMATORS,
        "registered_tree_counts": L.registered_tree_counts(),
        "n_train_cells": int(len(train)), "n_train_rows": int(np.isfinite(Y[train]).sum()),
        "n_test_cells": int(len(test_all)), "n_test_cells_scored": int(len(test)),
        "max_test_cells": max_test_cells, "n_logd_rows": int(len(logd)), "n_logsf_rows": int(len(logsf)),
        "dmd_removed": {"train": int(np.isin(fold.train_index, drop).sum()), "test": int(np.isin(fold.test_index, drop).sum())},
        "inner": [{"j": int(inn.j), "n_train": int(len(inn.train_pos)), "n_train_unmasked": int(len(inn.train_pos_unmasked)),
                   "n_val": int(len(inn.val_pos)), "skipped": bool(inn.skipped)} for inn in inner],
        "skipped_inner_folds": skipped, "all_inner_skipped": not active,
        **sel_rec, "chosen": chosen,
        "BL_D_INNER": {"criterion": "MET-01 log D on the pooled inner OOF rows (BD_COND: its selected configuration, "
                                    "optimistic best-of-four)", "scores": bl_scores, "chosen": bld, "reason": bld_why},
        "BL_SF_INNER": {"criterion": "MET-01 of inner strict-adjacent log SF", "scores": sf_scores, "chosen": bls,
                        "reason": bls_why},
        "fallbacks": {"BD_METAL_rows": len(fb_metal), "BD_METAL": fb_metal,
                      "BD_NN_rows_any": int(fb_nn_any.sum()), "BD_NN_rows_all": int(np.sum(fb_nn_all)),
                      "BD_NN": [[int(c), int(x)] for c, x, f in zip(ld["ci"], m, fb_nn_any) if f],
                      "PAIRMEAN_rows": int(FB_pm.sum()),
                      "PAIRMEAN": [[int(c), int(a), int(b)] for c, a, b, f in zip(ls["ci"], ia, ib, FB_pm) if f],
                      "HEAVIER_ALWAYS_absent_types": int((~pm.present_[ia, ib]).sum())},
        "arms_logd": want_d, "arms_logsf": want_s, "include_c": bool(include_c), "c_pair": cpair_rec,
        "a_prime_seed": seed + 1 if run_aprime else None, "coverage": cov,
        "timing": {"seconds_by_family": {k: round(v, 3) for k, v in timer.family.items()},
                   "seconds_unit": round(time.perf_counter() - t_unit, 3), "fits": timer.fits,
                   "warning_filter_restores": int(L.FILTER_RESTORES - restores0), **peak_rss()},
    }
    return UnitResult(logd=logd, logsf=logsf, selection=selection, inner_oof={"logd": inner_logd, "logsf": inner_logsf})


def assert_finite(table: pd.DataFrame, arms, name: str) -> None:
    """n_pred_finite == n_rows for every arm (a non-finite prediction is a run failure, not an NA); raises
    ``ArmFailure`` naming the first failing arm."""
    for a in arms:
        v = table[a].to_numpy(dtype=np.float64)
        if v.dtype != np.float64 or not np.isfinite(v).all():
            raise ArmFailure(a, f"{name}: arm {a} has {int((~np.isfinite(v)).sum())} non-finite predictions of {len(v)}")


def check_coverage(logd: pd.DataFrame, logsf: pd.DataFrame, design: str, fold, variant: str, test, ctx: UnitContext,
                   *, restricted: bool = False) -> dict:
    """LT-16: the log D rows are exactly every observed entry of the scored test cells; the strict log SF rows are
    exactly their strict-adjacent pairs and the Nd-Sm rows their observed Nd-Sm pairs; when the S3 row manifests
    hold this (design, seed, fold), the log D rows equal COH-D-ALL(@NODMD) and the strict rows COH-SF-ANY(@NODMD)
    on the scored cells, and COH-SF-NDSM(@NODMD) is covered."""
    test = np.asarray(test, dtype=int)
    want_d = {(int(c), int(m)) for c, m in zip(*np.nonzero(ctx.obs[test]))}
    want_d = {(int(test[c]), m) for c, m in want_d}
    got_d = set(zip(logd["ci"].astype(int), logd["im"].astype(int)))
    if got_d != want_d or len(got_d) != len(logd):
        raise CoverageError(f"logd rows differ from the observed entries of the test cells ({len(got_d ^ want_d)} keys)")
    got_s = set(zip(logsf["ci"].astype(int), logsf["ia"].astype(int), logsf["ib"].astype(int)))
    if len(got_s) != len(logsf):
        raise CoverageError("duplicated logsf keys")
    want_s = set()
    for c in test:
        ms = np.flatnonzero(ctx.obs[c])
        for i in ms:
            for j in ms:
                if i < j and (abs(int(D.Z[i]) - int(D.Z[j])) == 1 or (D.LANTHANIDES[i], D.LANTHANIDES[j]) == ("Nd", "Sm")):
                    want_s.add((int(c), int(i), int(j)))
    if got_s != want_s:
        raise CoverageError(f"logsf rows differ from the gen13-adjacent pairs of the test cells ({len(got_s ^ want_s)} keys)")
    rec = {"logd_rows": int(len(logd)), "logsf_rows": int(len(logsf)), "manifests_checked": []}
    suffix = "@NODMD" if variant == "nodmd" else ""
    tset = set(test.tolist())
    for name, keys, target in (("COH-D-ALL", ("ci", "im"), "logd"), ("COH-SF-ANY", ("ci", "ia", "ib"), "strict"),
                               ("COH-SF-NDSM", ("ci", "ia", "ib"), "ndsm")):
        man = ctx.manifest(name + suffix)
        if man is None:
            continue
        sub = man[(man["design"] == design) & (man["split_seed"] == int(fold.seed)) & (man["fold"] == int(fold.fold))]
        if sub.empty:
            continue
        if restricted:
            sub = sub[sub["ci"].isin(tset)]
        mk = set(zip(*(sub[k].astype(int) for k in keys)))
        if target == "logd":
            ok = mk == got_d
        elif target == "strict":
            st = logsf[logsf["strict"].to_numpy()]
            ok = mk == set(zip(st["ci"].astype(int), st["ia"].astype(int), st["ib"].astype(int)))
        else:
            ok = mk <= got_s
        if not ok:
            raise CoverageError(f"{name + suffix}: scored rows differ from the S3 row manifest for {design}/{fold.seed}/{fold.fold}")
        rec["manifests_checked"].append(name + suffix)
    return rec


# --------------------------------------------------------------------------------------
# fold gates for outcome-bearing scripts (IMPLEMENTATION_PLAN.md §4 "Fold gates")
# --------------------------------------------------------------------------------------
def fold_gates(bench, folds_by_design, *, subprocess_check: bool = True) -> dict:
    """Gates (a)-(e) on the six designs of the discovery seeds: (a)/(b) gen16 digests (in process and, unless
    ``subprocess_check`` is False, under PYTHONHASHSEED 1 and 2), (c) gen20 folds_BP.json sizes, (d) B/BR/BQ/BPG
    test sets equal BP's, (e) the outer+inner plan digest equals the S3 record.  Raises ``FoldGateError`` (STOP
    INCOMPARABLE_FOLDS) on any failure; returns the gate records."""
    missing = [d for d in FO.DESIGNS if d not in folds_by_design]
    if missing:
        raise FO.FoldGateError(f"fold gates need all six designs; missing {missing}")
    dig = FO.gen16_digest(folds_by_design)
    plan = FO.plan_digest(bench, folds_by_design)
    gates = {"a": FO.gate_a(dig), "b": FO.gate_b(dig), "c": FO.gate_c(folds_by_design["BP"]),
             "d": FO.gate_d(folds_by_design)}
    sub = None
    if subprocess_check:
        sub = FO.gate_b_subprocess(("1", "2"), plan_sha256=plan["sha256"])
        gates["b_subprocess"] = sub
    gates["e"] = FO.gate_e(plan, recorded=FO.recorded_plan(), subprocess_runs=None if sub is None else sub["runs"])
    bad = [k for k, v in gates.items() if not v.get("pass")]
    if bad:
        raise FO.FoldGateError(f"STOP INCOMPARABLE_FOLDS: fold gates failed: {bad}")
    return {k: {kk: vv for kk, vv in v.items() if kk not in ("runs",)} for k, v in gates.items()}


# --------------------------------------------------------------------------------------
# designs: writing, resuming, loading
# --------------------------------------------------------------------------------------
def code_digest(files=CODE_FILES) -> dict:
    """sha256 over the LF-normalised text of the code files that define the predictions (gen201 package minus the
    scoring-side modules, plus the discovery script); per-file digests included."""
    h = hashlib.sha256()
    per = {}
    for rel in files:
        p = bootstrap.G201 / rel
        if not p.exists():
            per[rel] = None
            h.update(f"{rel}:missing\n".encode())
            continue
        txt = p.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
        d = hashlib.sha256(txt).hexdigest()
        per[rel] = d
        h.update(f"{rel}:{d}\n".encode())
    return {"sha256": h.hexdigest(), "files": per}


#: the code digest of the files this process imported (read once, when the code was loaded): predictions are made by
#: the imported modules, so a unit is labelled with this digest and never with one re-read from disk later
CODE_AT_IMPORT: dict = code_digest()


def assert_code_unchanged() -> dict:
    """The current on-disk code digest; raises ``CodeChanged`` (STOP) when it differs from ``CODE_AT_IMPORT``."""
    code = code_digest()
    if code["sha256"] != CODE_AT_IMPORT["sha256"]:
        changed = sorted(k for k in set(code["files"]) | set(CODE_AT_IMPORT["files"])
                         if code["files"].get(k) != CODE_AT_IMPORT["files"].get(k))
        raise CodeChanged(f"STOP: gen201 code files changed since this process imported them ({changed}); "
                          "restart the run")
    return code


def runtime_versions() -> dict:
    """Interpreter and library versions that can change the fitted trees (recorded in every DONE marker; a resume
    under other versions refits the unit)."""
    import numpy
    import pandas
    import scipy
    import sklearn
    try:
        import pyarrow
        pa = pyarrow.__version__
    except Exception:  # pragma: no cover
        pa = None
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "pyarrow": pa}


def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def unit_paths(root: Path, label: str, design: str, variant: str, seed: int, fold: int) -> dict[str, Path]:
    d = Path(root) / label / design / variant
    stem = f"s{int(seed)}_f{int(fold)}"
    return {"dir": d, "logd": d / f"{stem}_logd.parquet", "logsf": d / f"{stem}_logsf.parquet",
            "selection": d / f"{stem}_selection.json", "inner_logd": d / f"{stem}_inner_logd.parquet",
            "inner_logsf": d / f"{stem}_inner_logsf.parquet", "done": d / f"{stem}_DONE.json"}


#: DONE-marker fields that must be equal across the units of one (label, design, variant) and, on resume, equal to
#: what the current run would produce (besides the code digest)
UNIT_SIGNATURE: tuple[str, ...] = ("arms_logd", "arms_logsf", "n_estimators", "c_pair_n_estimators", "runtime")


def unit_expectation(design: str, variant: str, *, include_c: bool = True, include_c_pair: bool = True) -> dict:
    """The UNIT_SIGNATURE a unit fitted now would carry: its arm set (which depends on the S8 flags), the tree counts
    and the interpreter / library versions."""
    return {"arms_logd": logd_arm_columns(design, variant, include_c=include_c),
            "arms_logsf": logsf_arm_columns(design, variant, include_c=include_c, include_c_pair=include_c_pair),
            "n_estimators": int(L.N_ESTIMATORS), "c_pair_n_estimators": int(L.C_PAIR_N_ESTIMATORS),
            "runtime": runtime_versions()}


def done_matches(paths: dict, code_sha: str, expect: dict | None = None) -> bool:
    """True when the unit's DONE marker parses, carries ``code_sha``, equals ``expect`` on every expected key (arm set,
    tree counts, library versions: ``unit_expectation``) and the parquets / selection record still hash as recorded."""
    if not paths["done"].exists():
        return False
    try:
        rec = json.loads(paths["done"].read_text(encoding="utf-8"))
    except Exception:
        return False
    if rec.get("code_digest") != code_sha:
        return False
    for k, v in (expect or {}).items():
        if rec.get(k) != _jsonable(v):
            return False
    for k in ("logd", "logsf", "selection"):
        if not paths[k].exists() or _sha(paths[k]) != rec.get(f"{k}_sha256"):
            return False
    return True


WRITE_TRIES = 5


def _retry(fn, *args, tries: int = WRITE_TRIES, **kwargs):
    """``fn(*args, **kwargs)`` retried on PermissionError (Windows: an indexer or antivirus holding the file open),
    waiting 0.5, 1, 2, 4 s, then giving up."""
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.5 * 2 ** i)


def _tmp(p: Path) -> Path:
    return p.with_name(p.name + ".tmp")


def _write_parquet_tmp(df: pd.DataFrame, p: Path) -> Path:
    t = _tmp(p)
    _retry(df.to_parquet, t, index=False)
    return t


def _write_json_tmp(obj, p: Path) -> Path:
    t = _tmp(p)

    def _w():
        with open(t, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(_jsonable(obj), fh, indent=1, sort_keys=True)
    _retry(_w)
    return t


def write_unit(res: UnitResult, paths: dict, code: dict, *, label: str) -> dict:
    """Write one unit.  The previous DONE marker is removed first (it never vouches for half-replaced files); every
    file is written to ``<name>.tmp`` and moved into place with ``os.replace``; the DONE marker goes last, also via
    tmp + ``os.replace``, so a crash leaves either no marker or a complete one.  PermissionError is retried.
    Refuses (``CodeChanged``) when ``code`` is not ``CODE_AT_IMPORT`` or the files on disk changed."""
    if code["sha256"] != CODE_AT_IMPORT["sha256"]:
        raise CodeChanged("STOP: the unit's code digest differs from the code this process imported; restart the run")
    assert_code_unchanged()
    paths["dir"].mkdir(parents=True, exist_ok=True)
    _retry(paths["done"].unlink, missing_ok=True)
    tmps = {"logd": _write_parquet_tmp(res.logd, paths["logd"]),
            "logsf": _write_parquet_tmp(res.logsf, paths["logsf"]),
            "inner_logd": _write_parquet_tmp(res.inner_oof["logd"], paths["inner_logd"]),
            "inner_logsf": _write_parquet_tmp(res.inner_oof["logsf"], paths["inner_logsf"]),
            "selection": _write_json_tmp(res.selection, paths["selection"])}
    for k, t in tmps.items():
        _retry(os.replace, t, paths[k])
    s = res.selection
    rec = {"label": label, "design": s["design"], "variant": s["variant"], "split_seed": s["split_seed"],
           "fold": s["fold"], "code_digest": code["sha256"], "code_files": code["files"],
           "logd_sha256": _sha(paths["logd"]), "logsf_sha256": _sha(paths["logsf"]),
           "selection_sha256": _sha(paths["selection"]), "inner_logd_sha256": _sha(paths["inner_logd"]),
           "inner_logsf_sha256": _sha(paths["inner_logsf"]), "n_logd": int(len(res.logd)),
           "n_logsf": int(len(res.logsf)), "arms_logd": s["arms_logd"], "arms_logsf": s["arms_logsf"],
           "seconds": s["timing"]["seconds_unit"], "n_estimators": s["n_estimators"],
           "c_pair_n_estimators": s["c_pair_n_estimators"], "runtime": runtime_versions(),
           "registered_tree_counts": s["registered_tree_counts"],
           "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t = _write_json_tmp(rec, paths["done"])
    assert_code_unchanged()
    _retry(os.replace, t, paths["done"])
    return rec


# --------------------------------------------------------------------------------------
# one writer per (label, design)
# --------------------------------------------------------------------------------------
LOCK_NAME = ".lock"
LOCK_FRESH_SECONDS = 30.0          # an unreadable lock younger than this is being written: treated as held
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5


def _kernel32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = ctypes.c_void_p
    k.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    k.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    k.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
    return k


def process_state(pid: int) -> tuple[bool, int | None]:
    """(alive, start time) of a process, without signalling it (``os.kill`` on Windows would terminate it); the
    start time is the Windows creation FILETIME (None elsewhere or when unknown), so a reused pid is told apart."""
    pid = int(pid)
    if pid <= 0:
        return False, None
    if os.name == "nt":
        try:
            k = _kernel32()
            h = k.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
            if not h:
                return ctypes.get_last_error() == _ERROR_ACCESS_DENIED, None
            try:
                code = ctypes.c_ulong()
                alive = (not k.GetExitCodeProcess(h, ctypes.byref(code))) or code.value == _STILL_ACTIVE
                ft = [ctypes.c_ulonglong() for _ in range(4)]
                start = int(ft[0].value) if k.GetProcessTimes(h, *(ctypes.byref(x) for x in ft)) else None
                return bool(alive), start
            finally:
                k.CloseHandle(h)
        except Exception:  # pragma: no cover - unknown: never steal a lock
            return True, None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, None
    except PermissionError:
        return True, None
    except OSError:
        return False, None
    return True, None


def _read_lock(p: Path):
    """The lock record; None when unreadable (being written, or torn by a crash); FileNotFoundError passes."""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception:
        return None


def _lock_held(holder, p: Path) -> bool:
    if not isinstance(holder, dict):
        try:
            return time.time() - p.stat().st_mtime < LOCK_FRESH_SECONDS
        except FileNotFoundError:
            return False
    if holder.get("host") != platform.node():
        return True                    # another machine: cannot be checked, never stolen
    alive, start = process_state(holder.get("pid", -1))
    if not alive:
        return False
    rec = holder.get("process_start")
    return rec is None or start is None or int(rec) == int(start)


@contextlib.contextmanager
def design_lock(directory, *, log=print):
    """Exclusive writer lock ``<directory>/.lock`` (``O_CREAT | O_EXCL``; pid, process start time, host).  A lock
    held by a live process raises ``DesignLocked`` (STOP); a lock whose process is gone, or whose pid now belongs to
    another process, is stale and taken over.  Released on exit."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / LOCK_NAME
    me = {"pid": os.getpid(), "process_start": process_state(os.getpid())[1], "host": platform.node(),
          "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for _ in range(3):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                holder = _read_lock(p)
            except FileNotFoundError:
                continue
            if _lock_held(holder, p):
                raise DesignLocked(f"STOP: {p} is held by {holder}: another process is writing this design (remove "
                                   "the lock by hand only if no such process runs)")
            log(f"taking over the stale lock {p} (holder {holder} is gone)")
            try:
                p.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(me, fh)
        break
    else:
        raise DesignLocked(f"STOP: could not take the lock {p}")
    try:
        yield me
    finally:
        try:
            if (_read_lock(p) or {}).get("pid") == me["pid"]:
                p.unlink()
        except FileNotFoundError:
            pass


def run_design(bench, design: str, *, seeds, label: str, variants=VARIANTS, resume: bool = True, smoke: bool = False,
               fold_ids=None, max_test_cells: int | None = None, include_c: bool = True, include_c_pair: bool = True,
               out_root=None, folds_ctx: UnitContext | None = None, folds_list=None, check_plan: bool = True,
               log=print) -> dict:
    """Fit and write every (seed, fold, variant) unit of ``design``; resumable on DONE markers.

    Refuses a label starting with "smoke" unless ``smoke=True`` (and a smoke run must carry such a label);
    ``fold_ids`` (iterable of (seed, fold)) and ``max_test_cells`` are smoke-only restrictions; writing under
    generations/gen20_1/predictions requires the registered tree counts.  With the discovery seeds, the design's
    outer+inner fold digest must equal the S3 record (gate (e), per design).  Refuses (``CodeChanged``) when the
    code on disk differs from the code this process imported; holds the (label, design) writer lock while it runs;
    a unit is skipped only when its DONE marker matches the imported code digest and ``unit_expectation`` (arm set
    for these ``include_c`` / ``include_c_pair`` flags, tree counts, library versions)."""
    if label.startswith("smoke") and not smoke:
        raise ValueError(f"label {label!r} starts with 'smoke': pass smoke=True (smoke output never enters a report)")
    if smoke and not label.startswith("smoke"):
        raise ValueError("a smoke run must carry a label starting with 'smoke'")
    if (fold_ids is not None or max_test_cells is not None) and not smoke:
        raise ValueError("fold_ids / max_test_cells restrict a run and are allowed for smoke runs only")
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"unknown variant {v!r}")
    root = bootstrap.assert_under_g201(out_root if out_root is not None else bootstrap.PREDICTIONS)
    if os.path.normcase(str(root)).startswith(os.path.normcase(str(bootstrap.PREDICTIONS.resolve()))) \
            and not L.registered_tree_counts():
        raise RuntimeError(f"refusing to write under {bootstrap.PREDICTIONS}: tree counts {L.N_ESTIMATORS}/"
                           f"{L.C_PAIR_N_ESTIMATORS} differ from the registered 400/200")
    t0 = time.time()
    seeds = tuple(int(s) for s in seeds)
    F = list(folds_list) if folds_list is not None else FO.build(bench, seeds=seeds, designs=(design,))[design]
    if any(f.design != design for f in F):
        raise ValueError("a fold of another design")
    plan_check = None
    if check_plan and seeds == FO.SEEDS:
        rec = FO.recorded_plan()
        got = FO.plan_digest(bench, {design: F}, designs=(design,))["per_design"][design]
        if rec is None or rec.get("per_design", {}).get(design) != got:
            raise FO.FoldGateError(f"design {design}: outer+inner fold digest {got} differs from the S3 record "
                                   f"{None if rec is None else rec.get('per_design', {}).get(design)}")
        plan_check = {"per_design_sha256": got, "equals_recorded": True}
    ctx = folds_ctx if folds_ctx is not None else make_context(bench)
    assert_code_unchanged()
    code = CODE_AT_IMPORT
    want = None if fold_ids is None else {(int(s), int(f)) for s, f in fold_ids}
    units = []
    with design_lock(bootstrap.assert_under_g201(root / label / design), log=log):
        for variant in variants:
            expect = unit_expectation(design, variant, include_c=include_c, include_c_pair=include_c_pair)
            for f in F:
                if want is not None and (int(f.seed), int(f.fold)) not in want:
                    continue
                paths = unit_paths(root, label, design, variant, f.seed, f.fold)
                if resume and done_matches(paths, code["sha256"], expect=expect):
                    units.append({"split_seed": int(f.seed), "fold": int(f.fold), "variant": variant,
                                  "status": "skipped (DONE)"})
                    log(f"[{label}] {design}/{variant} s{f.seed} f{f.fold}: DONE marker matches, skipped")
                    continue
                res = run_unit(bench, design, f, variant=variant, folds_ctx=ctx, include_c=include_c,
                               include_c_pair=include_c_pair, max_test_cells=max_test_cells, label=label)
                rec = write_unit(res, paths, code, label=label)
                units.append({"split_seed": int(f.seed), "fold": int(f.fold), "variant": variant, "status": "done",
                              "seconds": rec["seconds"],
                              "seconds_by_family": res.selection["timing"]["seconds_by_family"],
                              "warning_filter_restores": res.selection["timing"]["warning_filter_restores"],
                              "n_logd": rec["n_logd"], "n_logsf": rec["n_logsf"], "chosen": res.selection["chosen"]})
                log(f"[{label}] {design}/{variant} s{f.seed} f{f.fold}: {len(f.train_index)} train / "
                    f"{res.selection['n_test_cells_scored']} test cells, {rec['n_logd']} logd + {rec['n_logsf']} "
                    f"logsf rows, chosen {res.selection['chosen']}, {rec['seconds']:.0f}s")
    return {"design": design, "label": label, "seeds": list(seeds), "variants": list(variants), "smoke": smoke,
            "root": str(root), "code_digest": code["sha256"], "include_c": bool(include_c),
            "include_c_pair": bool(include_c_pair), "n_estimators": int(L.N_ESTIMATORS),
            "c_pair_n_estimators": int(L.C_PAIR_N_ESTIMATORS), "plan_check": plan_check, "units": units,
            "seconds": time.time() - t0}


def load_tables(label: str, design: str, variant: str, *, root=None, expect_units: int | None = None):
    """(logd, logsf) concatenated over every unit of (label, design, variant), each verified against its DONE
    marker (parquet sha256s), all sharing one code digest and one UNIT_SIGNATURE (arm set, tree counts, library
    versions); an unreadable DONE marker is an incomplete unit (``CoverageError``)."""
    root = Path(root) if root is not None else bootstrap.PREDICTIONS
    d = root / label / design / variant
    dones = sorted(d.glob("s*_f*_DONE.json"))
    if not dones:
        raise FileNotFoundError(f"no completed unit under {d}")
    codes, sigs, parts_d, parts_s = set(), {}, [], []
    for p in dones:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            raise CoverageError(f"{p}: unreadable DONE marker ({e}); the unit is incomplete") from e
        paths = unit_paths(root, label, design, variant, rec["split_seed"], rec["fold"])
        for k in ("logd", "logsf"):
            if _sha(paths[k]) != rec[f"{k}_sha256"]:
                raise CoverageError(f"{paths[k]} differs from its DONE marker")
        codes.add(rec["code_digest"])
        sig = {k: rec.get(k) for k in UNIT_SIGNATURE}
        sigs.setdefault(json.dumps(sig, sort_keys=True), (p.name, sig))
        parts_d.append(pd.read_parquet(paths["logd"]))
        parts_s.append(pd.read_parquet(paths["logsf"]))
    if len(codes) != 1:
        raise CoverageError(f"units of {label}/{design}/{variant} were produced by different code versions: {sorted(codes)}")
    if len(sigs) != 1:
        found = list(sigs.values())
        differ = sorted(k for k in UNIT_SIGNATURE if len({json.dumps(s[k], sort_keys=True) for _, s in found}) > 1)
        raise CoverageError(f"units of {label}/{design}/{variant} differ in {differ} (e.g. {found[0][0]} vs "
                            f"{found[1][0]}): arm sets, tree counts or library versions are mixed")
    if expect_units is not None and len(dones) != int(expect_units):
        raise CoverageError(f"{label}/{design}/{variant}: {len(dones)} completed units, expected {expect_units}")
    logd = pd.concat(parts_d, ignore_index=True).sort_values(["split_seed", "fold", "ci", "im"], kind="mergesort")
    logsf = pd.concat(parts_s, ignore_index=True).sort_values(["split_seed", "fold", "ci", "ia", "ib"], kind="mergesort")
    return logd.reset_index(drop=True), logsf.reset_index(drop=True)


def load_selections(label: str, design: str, variant: str, *, root=None) -> list[dict]:
    root = Path(root) if root is not None else bootstrap.PREDICTIONS
    d = root / label / design / variant
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("s*_f*_selection.json"))]

"""Stage-1 item 1.0, EA_CHECK (CLAUDE_CODE_PROMPT.md sections 4.1, 6.0, 6.10 row 1.0, 8, 9): the M35 positive control
and the six-design check.  Estimand E-A; every number EXPLORATORY (seen); every E-A contrast DIAGNOSTIC (M34's kill
fired, section 4.2); no E-A slot.

* Fold records for designs A, B, BR, BQ (section 4.1 item 5): ``g21.cache.build_fold`` called directly on every fold of
  ``gen20k.splits.folds_for`` (5 dev seeds x 5 folds per design), with one BLAS thread so a rebuild is bit-identical,
  pickled to <round5>/cache/<design>_<seed>_<fold>.pkl (never ``g21.cache.build_all``, never gen21/cache/).  Every
  record is rebuilt from ``r4.leakage.poisoned_bench`` (every label field of every cell outside the outer training fold
  randomised: held-out cells and the cells a BR / BQ mask drops) and compared field by field (section 8 item 3).  The
  cache manifest (manifests/ea_check_caches.json) keys each pickle by data content (bench log D, coefficient and
  cell-order digests), code (LF digests of the modules build_fold runs), split (fold index digests), support contract
  and the gen20 hyperparameter grids; ``load_record`` refuses a pickle whose key no longer matches.
* Rows: four reads (code budget k = 3) on each design's DOPT_Q nested read sets, built with ``r3.harness.cell_rows``
  exactly as round 4 built REF.  B0 (PWL_R) and HIST_D7 (B1's form with the fixed full-corpus -0.072; not
  fold-independent) are training-free, so they are scored on all six designs; B1 exists only on BP / BPG (round-4 REF
  parquets).  Clusters (``cluster_cpg``) come from the BPG REF rows, so every design uses the same 13 unit labels.
* M35 gate (section 4.1 item 2; BPG decides, BP a labelled sensitivity): the seed-mean relative pair-SSE
  1 - SSE_cand / SSE_ref >= 0.02, SF_POOL (exact enumeration of the 2^G sign vectors at G <= 13) one-sided p <= 0.025 on
  the seed-mean per-unit dSSE of the cluster_cpg units, and the balanced-MAE guard (ref - cand) >= -0.001.  Contrasts:
  B1 - B0 (the positive control, outside any family) and HIST_D7 - B1; HIST_D7 - B0 on the six designs.
* Mandatory tables (item 5): per-cluster with provenance (ORNL-only and the data_location class), big fold against the
  other folds, C5-BTBP in / out (+ the single-cell influence of c950dc2982e96b83), the six-design table; breadth
  diagnostic (item 6); M08's relation-class moment table (item 8) when REGISTRY.csv marks it PLANNED; per-seed values,
  leave-one-publication-out and the top-3 unit share (section 6.0); the section-9.3 per-item table of each design and
  of both (``per_item``; two sentences built from the computed outcomes).
* Reproduction target (section 6.0): B1 recomputed from round 4's stored fold betas on the gen21 caches matches the REF
  parquet within 8.9e-16 on all 18,020 rows (B0 and HIST_D7 recomputed through the same row builder likewise).

Nothing is copied from the design folder (reuse by import only).  No function here reads a withheld seed or writes
outside the round-5 folder.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import paths
from . import common as CM
from r4 import refs as RF  # noqa: E402
from r4 import leakage as LK  # noqa: E402
from r3 import harness as H  # noqa: E402
from r3 import interp  # noqa: E402
from g21 import cache as GC  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from gen20k import design as DS  # noqa: E402
from gen20k.splits import copy_groups_of, folds_for  # noqa: E402

paths.redirect_inner_cache()

ITEM = "1.0"
ITEM_NAME = "EA_CHECK"
OUT = paths.RES / ITEM_NAME
LEAK = paths.RES / "leakage"
MANIFEST = paths.MAN / "ea_check_caches.json"
POISON_REPORT = LEAK / "cache_poison_EA_CHECK.json"
INTERVENTION_REPORT = LEAK / "intervention_EA_CHECK.json"
REGISTRY = paths.R5 / "REGISTRY.csv"
PROV_CELLS = paths.R4RES / "provenance" / "cells.csv"
FOLD_TRAINING = paths.R4RES / "provenance" / "fold_training_{design}.csv"

NEW_DESIGNS = ("A", "B", "BR", "BQ")
SIX = ("A", "B", "BR", "BQ", "BP", "BPG")
K = 3                                  # code budget k = 3: four reads (section 3.3)
N_READS = 4
POLICY = "DOPT_Q"
UNIT = "cluster_cpg"
PICKLE_PROTOCOL = 4
BLAS_THREADS = 1

GATE_REL, GATE_P, GUARD = 0.02, 0.025, -0.001          # section 4.1 items 2-3
CONTRASTS = (("B1", "B0", "positive control, outside any family (section 4.1 item 7)"),
             ("HIST_D7", "B1", "diagnostic; HIST_D7 is not fold-independent"))
SIX_CONTRAST = ("HIST_D7", "B0")                        # the training-free contrast available on every design
STRATA = ("C5_in", "C5_out", "C5_cell_out")
SMALL_CELLS, SMALL_ROWS = 3, 20                          # breadth: < 3 cells or < 20 rows per seed -> "small sources"
REPRO_TOL = 8.9e-16
REPRO_ROWS = 18020                                       # 1,802 queries x 5 seeds x 2 designs
SEEN = {"B1 - B0": {"BPG": {"relSSE_pct": 5.85, "SF_POOL_p": 0.00317}, "BP": {"SF_POOL_p": 0.00269}}}  # IR M35; PC P2
PAIRS = ("La/Ce", "Ce/Pr", "Pr/Nd", "Sm/Eu", "Eu/Gd", "Gd/Tb", "Tb/Dy", "Dy/Ho", "Ho/Er", "Er/Tm", "Tm/Yb", "Yb/Lu")

L_SEEN, L_DIAG, L_SENS = "EXPLORATORY (seen)", "DIAGNOSTIC", "SENSITIVITY"
L_SMOKE, L_CC = "TECHNICAL SMOKE", "CONSISTENCY CHECK"
REQUIRED_ZERO = {"A": ("cell", "extractant"), "B": ("cell", "extractant", "chemotype"),
                 "BR": ("cell", "extractant", "chemotype"), "BQ": ("cell", "extractant", "chemotype"),
                 "BP": ("cell", "extractant", "chemotype", "publication"),
                 "BPG": ("cell", "extractant", "chemotype", "publication", "copy_group")}
LOC_RULES = (("FIG", re.compile(r"fig|graph", re.I)), ("TAB", re.compile(r"tab", re.I)))
# P5 F1 (brief 5.3, DATA_CONTRACT 1; reviewer exception to D8, s0-fidelity-11): the 10 C3 cells curated by Aurora Clark
# with an empty data_location whose location is in comments_raw -- pub_b9dfc468cd "table1" (4 cells), pub_19783c039b
# "fig2 raffinate" (1), pub_f8a621ede4 "fig3" / "fig3b" (5) -- are "located" in the round-5 label; the raw cells.csv
# class is kept beside it (loc_class_raw).
P5_F1 = {"pub_b9dfc468cd": "TAB", "pub_19783c039b": "FIG", "pub_f8a621ede4": "FIG"}
P5_F1_CURATOR = "Aurora Clark"
P5_F1_N = 10
M08_CLASSES = (("SE_SP", "same extractant, same publication (other conditions)"),
               ("SE_OP", "same extractant, other publication"),
               ("OE_SP_SC", "other extractant, same publication, same chemotype"),
               ("OE_SP_OC", "other extractant, same publication, other chemotype"),
               ("OE_OP_SC", "other extractant, other publication, same chemotype"),
               ("OE_OP_OC", "other extractant, other publication, other chemotype (baseline)"))
M08_MIN_UNITS, M08_MAX_TOP = 3, 0.80                    # identified: >= 3 product units and the top unit <= 80 %


def label(design: str, stratum: str = "C5_in", *extra: str) -> str:
    """Section 9.1 label: EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY unless BPG with C5-BTBP in."""
    parts = [L_SEEN, L_DIAG] + ([L_SENS] if design != "BPG" or stratum != "C5_in" else []) + list(extra)
    return "; ".join(parts)


def rng_seed(*key) -> int:
    return paths.seed_for("|".join(["EA_CHECK"] + [str(k) for k in key]))


# ------------------------------------------------------------------------------------------------------------
# fold records (section 4.1 item 5) and the cache-poisoning check (section 8 item 3)
# ------------------------------------------------------------------------------------------------------------
def cache_path(design: str, seed: int, fold: int):
    return paths.CACHE / f"{design}_{seed}_{fold}.pkl"


def code_digests() -> dict:
    """LF digests of the code that builds or poisons a record (cache key: code)."""
    import g21.cache, gen20k.arms, gen20k.cov, gen20k.data, gen20k.design, gen20k.runner, gen20k.splits  # noqa: E401
    import gen13sep.splits
    mods = (g21.cache, gen20k.arms, gen20k.cov, gen20k.data, gen20k.design, gen20k.runner, gen20k.splits,
            gen13sep.splits, LK)
    out = {}
    for m in mods:
        p = Path(m.__file__).resolve()
        try:
            rel = p.relative_to(paths.ROOT).as_posix()
        except ValueError:
            rel = p.as_posix()
        out[rel] = paths.sha256_text_lf(p)
    return out


def _sha_arrays(*arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def data_digest(bench) -> dict:
    """Cache key: data version and content (V1 log D matrix, gen13 coefficients, cell order)."""
    cells = "\n".join(bench.frame["cell_id"].astype(str)).encode()
    return {"data_version": "V1", "Y": _sha_arrays(np.asarray(bench.Y, float)),
            "coef": _sha_arrays(np.asarray(bench.coef, float)), "cell_order": hashlib.sha256(cells).hexdigest(),
            "n_cells": int(len(bench.frame))}


def fold_digest(f) -> str:
    return _sha_arrays(np.asarray(f.train_index, np.int64), np.asarray(f.test_index, np.int64),
                       np.asarray(f.inner_train_index, np.int64), np.asarray(f.inner_validation_index, np.int64))


SUPPORT_CONTRACT = ("adjacent queries, metal model; DOPT_Q (fold covariance, noise_var = 2 s2) and UNIFORM sequences "
                    f"up to k_max = min(k_max, {GC.KMAX}); G14 / FLAT zero-shot curves; cache version {GC.CACHE_VERSION}")
HYPERPARAMETERS = "gen20 RunConfig defaults (noise-factor and ridge-lambda grids, selection_ks), NNC arms off"


def build_record(bench, fold) -> dict:
    """g21.cache.build_fold with one BLAS thread (the stored gen21 format, including its ``seconds`` field)."""
    with threadpool_limits(limits=BLAS_THREADS):
        return GC.build_fold(bench, fold)


def strip_seconds(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "seconds"}


def diff_paths(a, b, path: str = "", out=None, limit: int = 20) -> list:
    """Field-by-field bit comparison of two nested records (NaN equal to NaN); the mismatching paths."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            out.append(f"{path}: keys differ")
            return out
        for k in a:
            diff_paths(a[k], b[k], f"{path}/{k}", out, limit)
    elif isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            out.append(f"{path}: length differs")
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            diff_paths(x, y, f"{path}[{i}]", out, limit)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        nan_ok = x.dtype.kind in "fc" and y.dtype.kind in "fc"
        if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y, equal_nan=nan_ok):
            out.append(f"{path}: array differs")
    elif not LK.deep_equal(a, b):
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def poison_check(bench, fold, rec: dict) -> dict:
    """Rebuild the record from a bench whose every label field outside the outer training fold is random; every field
    (outer covariance, s2, prior, lambda, noise factor, selection scores, held-out cells' supports, sequences and
    zero-shot curves, every inner block) must be bit-identical."""
    t0 = time.time()
    seed = rng_seed("poison", fold.design, fold.seed, fold.fold)
    pb, cols = LK.poisoned_bench(bench, fold.train_index, seed=seed)
    rebuilt = build_record(pb, fold)
    bad = diff_paths(strip_seconds(rec), strip_seconds(rebuilt))
    n_other = int(len(bench.frame) - len(fold.train_index))
    return {"identical": not bad, "differences": bad[:10], "rng_seed": seed,
            "poisoned_cells": n_other, "poisoned_test_cells": int(len(fold.test_index)),
            "poisoned_design_excluded_cells": int(n_other - len(fold.test_index)), "poisoned_label_columns": len(cols),
            "seconds": round(time.time() - t0, 2)}


def isolation(bench, rec: dict) -> dict:
    """Design-aware isolation (section 8 item 1).  g21.evaluate.assert_test_isolation requires zero publication and
    chemotype crossing, which designs A, B, BR and BQ do not promise by construction; here every overlap is reported and
    the design's own guarantees are required: A cells and extractants; B / BR / BQ also chemotypes; BP also
    publications; BPG also copy groups.  Inner blocks must lie inside the outer training fold and be disjoint."""
    fr = bench.frame
    keys = {"cell": np.arange(len(fr)), "extractant": fr["extractant"].astype(str).to_numpy(),
            "chemotype": fr["chemotype"].astype(str).to_numpy(), "publication": fr["publication_id"].astype(str).to_numpy(),
            "copy_group": copy_groups_of(fr)}
    tr, te = np.asarray(rec["outer"]["train_index"]), np.asarray(rec["outer"]["test_index"])
    over = {k: int(len(set(v[tr].tolist()) & set(v[te].tolist()))) for k, v in keys.items()}
    trs = set(map(int, tr))
    inside = all(set(map(int, b["train_index"])) <= trs and set(map(int, b["val_index"])) <= trs for b in rec["inner"])
    disjoint = all(not (set(map(int, b["train_index"])) & set(map(int, b["val_index"]))) for b in rec["inner"])
    cells_held = all(int(c["ci"]) in set(map(int, te)) for c in rec["outer"]["cells"])
    need = REQUIRED_ZERO[rec["design"]]
    ok = all(over[k] == 0 for k in need) and inside and disjoint and cells_held
    return {"design": rec["design"], "seed": int(rec["split_seed"]), "fold": int(rec["fold"]),
            **{f"overlap_{k}": v for k, v in over.items()}, "required_zero": list(need), "inner_inside_outer_train": inside,
            "inner_disjoint": disjoint, "outer_cells_held_out": cells_held, "passed": bool(ok)}


def load_manifest(path=MANIFEST) -> dict:
    if not path.exists():
        return {"records": {}}
    return json.loads(path.read_bytes().decode("utf-8"))


def _write_json(path, obj) -> None:
    paths.write_text_lf(path, json.dumps(obj, indent=1, default=str) + "\n")


def build_caches(bench, designs=NEW_DESIGNS, seeds=paths.DEV_SEEDS, *, force: bool = False, log=print) -> dict:
    """Build (or verify) every fold record of ``designs`` x ``seeds`` x 5 folds, run the poisoning and isolation checks
    on each, and write the manifest and the poisoning report after every fold (an interrupted build resumes)."""
    man = load_manifest()
    man.update({"item": ITEM, "builder": "g21.cache.build_fold (direct; never build_all)", "blas_threads": BLAS_THREADS,
                "support_contract": SUPPORT_CONTRACT, "hyperparameters": HYPERPARAMETERS,
                "data": data_digest(bench), "code": code_digests(), "label": "construction (no score)"})
    man.setdefault("records", {})
    t_all = time.time()
    jobs = [f for d in designs for f in folds_for(bench.frame, d, seeds=list(seeds))]
    log(f"{len(jobs)} fold records ({', '.join(designs)} x {len(seeds)} seeds x 5 folds)")
    for j, f in enumerate(jobs):
        name = cache_path(f.design, f.seed, f.fold).name
        p = cache_path(f.design, f.seed, f.fold)
        ent = man["records"].get(name)
        if (not force and ent is not None and p.exists() and ent.get("sha256") == paths.sha256_file(p)
                and ent.get("code") == man["code"] and ent.get("data") == man["data"]
                and ent.get("fold_digest") == fold_digest(f) and ent.get("poison", {}).get("identical") is True):
            log(f"  [{j + 1}/{len(jobs)}] {name}: present, key and poisoning check unchanged")
            continue
        rec = build_record(bench, f)
        b = pickle.dumps(rec, protocol=PICKLE_PROTOCOL)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
        pois = poison_check(bench, f, rec)
        iso = isolation(bench, rec)
        man["records"][name] = {"design": f.design, "seed": int(f.seed), "fold": int(f.fold), "path": f"cache/{name}",
                                "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b),
                                "build_seconds": round(float(rec["seconds"]), 2), "fold_digest": fold_digest(f),
                                "n_train": int(len(f.train_index)), "n_test": int(len(f.test_index)),
                                "n_outer_cells": len(rec["outer"]["cells"]), "n_inner_blocks": len(rec["inner"]),
                                "data": man["data"], "code": man["code"], "poison": pois, "isolation": iso}
        _write_json(MANIFEST, man)
        log(f"  [{j + 1}/{len(jobs)}] {name}: built {rec['seconds']:.1f}s, poisoned rebuild {pois['seconds']:.1f}s "
            f"{'identical' if pois['identical'] else 'DIFFERS ' + str(pois['differences'][:2])}; isolation "
            f"{'ok' if iso['passed'] else 'FAILED'} ({len(rec['outer']['cells'])} test cells, {len(rec['inner'])} inner blocks)")
    sel = {k: v for k, v in man["records"].items() if v["design"] in designs and v["seed"] in set(seeds)}
    man["summary"] = {"n_records": len(man["records"]),
                      "poison_identical": int(sum(v["poison"]["identical"] for v in man["records"].values())),
                      "isolation_passed": int(sum(v["isolation"]["passed"] for v in man["records"].values())),
                      "build_seconds_total": round(sum(v["build_seconds"] for v in man["records"].values()), 1),
                      "poison_seconds_total": round(sum(v["poison"]["seconds"] for v in man["records"].values()), 1),
                      "this_call_seconds": round(time.time() - t_all, 1)}
    _write_json(MANIFEST, man)
    rep = {"item": ITEM, "check": "cache poisoning over every non-training cell (section 8 item 3)",
           "label": "leakage check (no score)", "n_folds": len(sel),
           "passed": bool(sel and all(v["poison"]["identical"] for v in sel.values())),
           "isolation_passed": bool(sel and all(v["isolation"]["passed"] for v in sel.values())),
           "folds": [{"fold": k, **{x: v["poison"][x] for x in ("identical", "poisoned_test_cells",
                                                                 "poisoned_design_excluded_cells", "rng_seed", "seconds")}}
                     for k, v in sorted(sel.items())],
           "failures": [{"fold": k, "differences": v["poison"]["differences"]} for k, v in sorted(sel.items())
                        if not v["poison"]["identical"]]}
    _write_json(POISON_REPORT, rep)
    return man


def cache_problems(bench, designs=NEW_DESIGNS, seeds=paths.DEV_SEEDS, man=None) -> list:
    """Why the round-5 caches cannot be used (empty when every record exists, matches its key and passed its checks)."""
    man = load_manifest() if man is None else man
    out, code, data = [], code_digests(), data_digest(bench)
    for d in designs:
        for f in folds_for(bench.frame, d, seeds=list(seeds)):
            name = cache_path(d, f.seed, f.fold).name
            ent = man["records"].get(name)
            if ent is None or not cache_path(d, f.seed, f.fold).exists():
                out.append(f"{name}: missing")
            elif ent["code"] != code or ent["data"] != data or ent["fold_digest"] != fold_digest(f):
                out.append(f"{name}: cache key changed (code / data / split)")
            elif not ent["poison"]["identical"] or not ent["isolation"]["passed"]:
                out.append(f"{name}: poisoning or isolation check failed")
    return out


def load_record(design: str, seed: int, fold: int, man=None) -> dict:
    """A round-5 record (A / B / BR / BQ) whose bytes match the manifest; BP / BPG come from gen21/cache (read-only)."""
    if design in ("BP", "BPG"):
        return GC.load(design, seed, fold)
    man = load_manifest() if man is None else man
    p = cache_path(design, seed, fold)
    ent = man["records"].get(p.name)
    b = p.read_bytes()
    if ent is None or hashlib.sha256(b).hexdigest() != ent["sha256"]:
        raise RuntimeError(f"{p.name}: not in the cache manifest or its bytes changed; rebuild with --build-cache")
    return pickle.loads(b)


# ------------------------------------------------------------------------------------------------------------
# rows, intervention, support checks
# ------------------------------------------------------------------------------------------------------------
def b0_levels(block, cell, obs, k):
    """B0 = PWL_R (r3.configs.baseline_methods' PWL_R, the REF column B0)."""
    return interp.pwl_levels(obs.y_masked, cell["metals"], "R")


def hist_d7_levels(block, cell, obs, k):
    """HIST_D7 (r4.registry._hist_d7): B1's form with the fixed -0.072."""
    return RF.predict_levels(RF.HIST_D7_FIT, obs, cell["metals"])


def b1_fit(beta: float) -> dict:
    return {"name": "B1", "spec": RF.SPECS["B1"], "kind": "R", "coef": np.array([float(beta)])}


def b1_method(beta: float):
    fit = b1_fit(beta)

    def fn(block, cell, obs, k):
        return RF.predict_levels(fit, obs, cell["metals"])
    return fn


def peek_levels(block, cell, obs, k):
    """Leakage positive control (section 8 item 4): reads the cell's hidden labels through block['Y']."""
    return np.asarray(block["Y"][cell["ci"]], float)


METHODS = {"B0": b0_levels, "HIST_D7": hist_d7_levels}


def fold_rows(bench, rec: dict, methods=None, *, Y=None) -> list:
    """Query rows of every held-out cell at four reads (k = 3; cells with k_max < 3 have no k = 3 row)."""
    Y = bench.Y if Y is None else Y
    methods = METHODS if methods is None else methods
    blk = dict(rec["outer"], Y=Y, policy=POLICY, _rec=rec)
    rows = []
    for cell in blk["cells"]:
        rows += H.cell_rows(rec, blk, cell, methods, policy=POLICY, ks=(K,), Y=Y, version="V1")
    return rows


def intervention(bench, rec: dict, methods=None, *, k: int = K) -> dict:
    """Every unread metal of every held-out C3 cell replaced by a random number: read sequences and every prediction
    must be bit-identical; the block['Y'] peek (positive control) must move.  Also the support checks of section 8
    item 5: exactly four reads, readings kept exactly at read metals, no scored pair with both ends read, |dZ| = 1."""
    methods = METHODS if methods is None else methods
    rng = np.random.default_rng(rng_seed("intervention", rec["design"], rec["split_seed"], rec["fold"]))
    Y = np.asarray(bench.Y, float)
    blk = dict(rec["outer"], Y=Y, policy=POLICY, _rec=rec)
    cells = [c for c in blk["cells"] if c["k_max"] >= k]
    Ys = Y.copy()
    for cell in cells:
        read = DS.reveal(Y[cell["ci"]], list(cell["seqs"][POLICY][:k])).read
        Ys[cell["ci"]] = E.scramble_hidden(Y[cell["ci"]], read, rng)
    b2 = dict(blk, Y=Ys)
    meth = {**methods, "PEEK": peek_levels}
    worst, n, peek, nan_bad, seq_bad = 0.0, 0, 0.0, 0, 0
    sup = {"cells": 0, "reads_not_4": 0, "support_not_exact": 0, "both_read": 0, "not_adjacent": 0}
    for cell in cells:
        seq = cell["seqs"][POLICY]
        obs = E.observe(Y[cell["ci"]], cell, seq, k)
        sup["cells"] += 1
        sup["reads_not_4"] += int(len(obs.read) != N_READS)
        for name, fn in methods.items():
            v = np.asarray(fn(blk, cell, obs, k), float)
            sup["support_not_exact"] += int(any(v[m] != obs.y_masked[m] for m in obs.read))
        r1 = H.cell_rows(rec, blk, cell, meth, policy=POLICY, ks=(k,), Y=Y, version="x")
        r2 = H.cell_rows(rec, b2, cell, meth, policy=POLICY, ks=(k,), Y=Ys, version="x")
        for a, b in zip(r1, r2):
            seq_bad += int(a["read_set"] != b["read_set"])
            sup["both_read"] += int(a["a_read"] and a["b_read"])
            sup["not_adjacent"] += int(a["dZ"] != 1)
            for m in methods:
                if np.isnan(a[m]) != np.isnan(b[m]):
                    nan_bad += 1
                elif np.isfinite(a[m]):
                    worst = max(worst, abs(a[m] - b[m]))
                    n += 1
            if np.isfinite(a["PEEK"]) and np.isfinite(b["PEEK"]):
                peek = max(peek, abs(a["PEEK"] - b["PEEK"]))
    sup_ok = sup["reads_not_4"] == 0 and sup["support_not_exact"] == 0 and sup["both_read"] == 0 and sup["not_adjacent"] == 0
    return {"design": rec["design"], "seed": int(rec["split_seed"]), "fold": int(rec["fold"]), "n_predictions": n,
            "max_abs_change": worst, "nan_pattern_changes": nan_bad, "read_set_changes": seq_bad,
            "peek_caught": bool(peek > 0), "support": sup, "support_ok": bool(sup_ok),
            "passed": bool(worst == 0.0 and nan_bad == 0 and seq_bad == 0 and peek > 0 and sup_ok)}


def prep(t: pd.DataFrame, cmap: dict) -> pd.DataFrame:
    """E-A rows (k = 3, k_max >= 3) sorted by the round-4 key, with the BPG cluster labels and pubchem."""
    t = t[(t.k == K) & (t.k_max >= 3)].sort_values(CM.KEY).reset_index(drop=True)
    t["cluster_cpg"] = t.cell_id.astype(str).map(cmap)
    if t.cluster_cpg.isna().any():
        raise KeyError("a cell has no BPG cluster label")
    t["pubchem"] = t.publication.astype(str) + "|" + t.chemotype.astype(str)
    return t


def cluster_map(t_bpg: pd.DataFrame) -> dict:
    return dict(zip(t_bpg.cell_id.astype(str), t_bpg.cluster_cpg.astype(str)))


def same_partition(a, b) -> bool:
    """Two labelings of the same items define the same partition (names may differ)."""
    a, b = pd.Series(np.asarray(a)).astype(str), pd.Series(np.asarray(b)).astype(str)
    return bool(len(a) == len(b) and pd.crosstab(a, b).astype(bool).sum(1).eq(1).all()
                and pd.crosstab(b, a).astype(bool).sum(1).eq(1).all())


def ea_rows(design: str, cmap: dict | None = None) -> pd.DataFrame:
    """The round-4 REF E-A rows of BP / BPG (r5.common.load_rows), with the BPG cluster labels when ``cmap`` is given
    (the union-find names of r4.metrics.add_clusters depend on row order; the partition must be identical)."""
    t = CM.load_rows(design)
    if cmap is not None:
        lab = t.cell_id.astype(str).map(cmap)
        if lab.isna().any() or not same_partition(lab, t.cluster_cpg):
            raise AssertionError(f"{design}: cluster_cpg partition differs from BPG's")
        t["cluster_cpg"] = lab
    return t


def design_rows(bench, design: str, cmap: dict, man=None, *, seeds=paths.DEV_SEEDS, fold_filter=None, log=None):
    """Rows of one round-5 design (A / B / BR / BQ) with B0 and HIST_D7, plus per-fold isolation and intervention."""
    man = load_manifest() if man is None else man
    rows, iso, iv = [], [], []
    for f in folds_for(bench.frame, design, seeds=list(seeds)):
        if fold_filter is not None and (f.seed, f.fold) not in fold_filter:
            continue
        rec = load_record(design, f.seed, f.fold, man)
        iso.append(isolation(bench, rec))
        iv.append(intervention(bench, rec))
        rows += fold_rows(bench, rec)
        if log:
            log(f"  {design} {f.seed} fold {f.fold}: {len(rows)} rows")
    return prep(pd.DataFrame(rows), cmap), iso, iv


def assert_same_queries(rows: dict, ref: str = "BPG") -> dict:
    """Fixed denominators: every design scores the same (seed, cell, A, B) queries with the same y."""
    key = ["split_seed", "cell_id", "A", "B"]
    base = rows[ref].set_index(key).sort_index()
    out = {}
    for d, t in rows.items():
        x = t.set_index(key).sort_index()
        same = x.index.equals(base.index)
        out[d] = {"rows": int(len(t)), "rows_per_seed": float(len(t) / t.split_seed.nunique()),
                  "cells_per_seed": float(t.groupby("split_seed").cell_id.nunique().mean()),
                  "same_queries": bool(same), "same_y": bool(same and np.array_equal(x.y.to_numpy(), base.y.to_numpy())),
                  "same_clusters": bool(same and (x.cluster_cpg.to_numpy() == base.cluster_cpg.to_numpy()).all())}
    bad = [d for d, v in out.items() if not (v["same_queries"] and v["same_y"] and v["same_clusters"])]
    if bad:
        raise AssertionError(f"designs {bad} do not score the BPG queries (fixed denominators): {out}")
    return out


def readset_table(rows: dict, ref: str = "BPG") -> pd.DataFrame:
    """Label-free: share of C3 cells (seed mean) whose four-read DOPT_Q set differs from BPG's."""
    key = ["split_seed", "cell_id"]
    base = rows[ref].drop_duplicates(key).set_index(key).read_set
    out = []
    for d, t in rows.items():
        x = t.drop_duplicates(key).set_index(key).read_set.reindex(base.index)
        diff = (x != base).groupby(level=0).mean()
        out.append({"design": d, "share_cells_readset_differs_from_BPG": float(diff.mean()),
                    "per_seed": ";".join(f"{v:.3f}" for v in diff), "label": "DIAGNOSTIC (label-free count)"})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------------------
# provenance (section 4.1 item 5; IR M21 salvage)
# ------------------------------------------------------------------------------------------------------------
def loc_class(s) -> str:
    """data_location class: FIG (figure / graph), TAB (table), MIXED (both), NONE (no location recorded), OTHER."""
    s = "" if s is None or (isinstance(s, float) and math.isnan(s)) else str(s).strip()
    if not s or s.lower() == "nan":
        return "NONE"
    hit = {name for name, rx in LOC_RULES if rx.search(s)}
    return "MIXED" if len(hit) == 2 else (hit.pop() if hit else "OTHER")


def provenance_cells(path=PROV_CELLS) -> pd.DataFrame:
    """Per cell (metadata only): data_location class after the P5 F1 label (``loc_class``: the 10 comment-located C3
    cells are located, DATA_CONTRACT 1), the raw cells.csv class (``loc_class_raw``), ORNL import flag, curator."""
    c = pd.read_csv(path, keep_default_na=False, dtype=str)
    raw = c.data_location.map(loc_class)
    f1 = (raw == "NONE") & c.publication_id.isin(list(P5_F1)) & (c.curator.astype(str) == P5_F1_CURATOR) \
        & c.c3.astype(str).str.lower().eq("true")
    if int(f1.sum()) != P5_F1_N:
        raise AssertionError(f"P5 F1: expected {P5_F1_N} comment-located C3 cells, found {int(f1.sum())}")
    lab = raw.where(~f1, c.publication_id.map(P5_F1))
    out = pd.DataFrame({"cell_id": c.cell_id.astype(str), "loc_class": lab, "loc_class_raw": raw,
                        "p5_f1_relabelled": f1.to_numpy(bool),
                        "ornl_import": c.ornl_import_all.astype(str).str.lower().eq("true"),
                        "curator": c.curator.astype(str)})
    return out.set_index("cell_id")


def unit_provenance(t: pd.DataFrame, prov: pd.DataFrame, unit: str = UNIT) -> pd.DataFrame:
    cells = t.drop_duplicates("cell_id")[[unit, "cell_id", "publication", "chemotype"]].join(prov, on="cell_id")
    if cells.loc_class.isna().any():
        raise KeyError("a scored cell has no provenance row")
    g = cells.groupby(unit)

    def classes(s):
        vc = s.value_counts()
        return "; ".join(f"{k} {int(vc[k])}" for k in sorted(vc.index))
    out = pd.DataFrame({"n_cells": g.cell_id.nunique(), "n_publications": g.publication.nunique(),
                        "n_chemotypes": g.chemotype.nunique(), "n_ornl_import_cells": g.ornl_import.sum().astype(int),
                        "ornl_only": g.ornl_import.all(), "data_location_classes": g.loc_class.agg(classes)})
    if "loc_class_raw" in cells.columns:                 # the raw cells.csv class beside the round-5 label
        out["data_location_classes_raw_cells_csv"] = g.loc_class_raw.agg(classes)
    out.index.name = unit
    return out


# ------------------------------------------------------------------------------------------------------------
# the M35 gate and the mandatory tables
# ------------------------------------------------------------------------------------------------------------
def _n_seeds(t) -> int:
    return max(int(t.split_seed.nunique()), 1)


def contrast_stats(t: pd.DataFrame, cand: str, ref: str, key: str, unit: str = UNIT) -> tuple[dict, pd.DataFrame]:
    """Every M35 quantity of one contrast on one row set: levels, relative SSE (seed mean and per seed), dR2, balanced
    MAE and its guard, the unit tests (SF_POOL decides; SIGN; SF_LOG and IM_REL diagnostics), the top-3 unit share and
    the three gate legs.  Positive always favours ``cand``."""
    us = CM.unit_stats(t, cand, ref, unit)
    tests = CM.unit_tests(us, f"{key}|{unit}")
    rel_s = CM.rel_sse_seeds(t, cand, ref)
    rc, rr = CM.r2_seeds(t, cand), CM.r2_seeds(t, ref)
    dbal = CM.dbal_mae(t, cand, ref)
    rel, p = float(rel_s.mean()), float(tests["SF_POOL"]["p"])
    row = {"contrast": f"{cand} - {ref}", "cand": cand, "ref": ref, "cells": int(t.cell_id.nunique()),
           "publications": int(t.publication.nunique()), "rows_per_seed": float(len(t) / _n_seeds(t)),
           "unit": unit, "G": int(tests["G"]), "R2_cand": float(rc.mean()), "R2_ref": float(rr.mean()),
           "dR2": float((rc - rr).mean()), "MSE_cand": CM.mse_seedmean(t, cand), "MSE_ref": CM.mse_seedmean(t, ref),
           "MAE_bal_cand": CM.bal_mae(t, cand), "MAE_bal_ref": CM.bal_mae(t, ref), "dMAE_bal": dbal,
           "relSSE": rel, "relSSE_per_seed": ";".join(f"{v:.5f}" for v in rel_s),
           "dR2_per_seed": ";".join(f"{v:.5f}" for v in (rc - rr)), "seeds_relSSE_positive": int((rel_s > 0).sum()),
           "W/L/T": tests["W/L/T"], "SF_POOL_T": float(tests["SF_POOL"]["T"]), "SF_POOL_p": p,
           "SF_POOL_exact": bool(tests["SF_POOL"]["exact"]), "SF_POOL_n_flips": int(tests["SF_POOL"]["n_flips"]),
           "SIGN_p": float(tests["SIGN"]["p"]),
           "SF_LOG_p": float(tests["SF_LOG"]["p"]) if tests["SF_LOG"] is not None else float("nan"),
           "IM_REL_t": float(tests["IM_REL"]["t"]), "IM_REL_p": float(tests["IM_REL"]["p"]),
           "top3_unit_share": CM.top3_share(us.dSSE),
           "gate_relSSE_ge_0.02": bool(rel >= GATE_REL), "gate_SF_POOL_p_le_0.025": bool(p <= GATE_P),
           "guard_dMAE_bal_ge_-0.001": bool(dbal >= GUARD)}
    row["gate_pass"] = bool(row["gate_relSSE_ge_0.02"] and row["gate_SF_POOL_p_le_0.025"] and row["guard_dMAE_bal_ge_-0.001"])
    return row, us


def unit_table(us: pd.DataFrame, prov_units: pd.DataFrame, lab: str) -> pd.DataFrame:
    """The per-cluster table: per-unit seed means, dSSE share, W/L/T and the provenance columns."""
    out = us.join(prov_units, how="left")
    tot = float(out.dSSE.sum())
    out["share_of_dSSE"] = out.dSSE / tot if tot != 0 else np.nan
    out = out.rename(columns={"n": "rows_per_seed"}).sort_values("rows_per_seed", ascending=False)
    out["label"] = lab
    return out


def add_big_fold(t: pd.DataFrame) -> pd.DataFrame:
    """big_fold = the fold holding the most scored rows of its seed (EB section 2.14: 58.2 % of rows)."""
    cnt = t.groupby(["split_seed", "fold"]).size()
    big = {(s, int(cnt.loc[s].idxmax())) for s in cnt.index.get_level_values(0).unique()}
    t = t.copy()
    t["big_fold"] = [(s, int(f)) in big for s, f in zip(t.split_seed, t.fold)]
    return t


def bigfold_table(t: pd.DataFrame, cand: str, ref: str, design: str, key: str) -> pd.DataFrame:
    if "big_fold" not in t:
        t = add_big_fold(t)
    ftr = FOLD_TRAINING.with_name(FOLD_TRAINING.name.format(design=design))
    if not ftr.exists():                                  # s0-mechanics-9: a sealed input; never skipped silently
        raise FileNotFoundError(ftr)
    tr = pd.read_csv(ftr).set_index(["seed", "fold"]).train_c3_cells
    big = t[t.big_fold][["split_seed", "fold"]].drop_duplicates()
    tot_d = float(sum(((g[ref] - g.y) ** 2).sum() - ((g[cand] - g.y) ** 2).sum() for _, g in t.groupby("split_seed")))
    rows = []
    for part, sub in (("big fold", t[t.big_fold]), ("other folds", t[~t.big_fold])):
        row, _ = contrast_stats(sub, cand, ref, f"{key}|{part}")
        d = float(sum(((g[ref] - g.y) ** 2).sum() - ((g[cand] - g.y) ** 2).sum() for _, g in sub.groupby("split_seed")))
        row = {"part": part, "share_of_rows": float(len(sub) / len(t)), "share_of_dSSE": d / tot_d if tot_d else np.nan, **row}
        if part == "big fold":
            row["folds"] = ";".join(f"{int(s)}:{int(f)}" for s, f in big.itertuples(index=False))
            if tr is not None:
                got = [tr.get((int(s), int(f))) for s, f in big.itertuples(index=False)]
                row["train_c3_cells"] = ";".join("NA" if v is None else str(int(v)) for v in got)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["label"] = label(design)
    return out


def lopo_table(t: pd.DataFrame, cand: str, ref: str) -> tuple[dict, pd.DataFrame]:
    """Leave-one-publication-out jackknife of the seed-mean relative SSE (common.lopo)."""
    jk = CM.lopo(t, lambda d: CM.rel_sse(d, cand, ref), by="publication")
    vals = pd.DataFrame({"publication_left_out": list(jk["values"]), "relSSE": list(jk["values"].values())})
    return {k: v for k, v in jk.items() if k != "values"}, vals


def breadth_units(t: pd.DataFrame, prov_units: pd.DataFrame, variant: str) -> pd.Series:
    """Unit labels of the breadth diagnostic: RAW (the cpg clusters), SMALL_POOLED (clusters with < 3 cells or < 20
    rows per seed pooled into 'small sources'; the sealed primary), ORNL_COLLAPSED (ORNL-only clusters as one unit)."""
    cl = t[UNIT].astype(str)
    if variant == "RAW":
        return cl
    if variant == "SMALL_POOLED":
        cells = t.groupby(UNIT).cell_id.nunique()
        rps = t.groupby(UNIT).size() / _n_seeds(t)
        small = set(cells.index[(cells < SMALL_CELLS) | (rps < SMALL_ROWS)].astype(str))
        return cl.where(~cl.isin(small), "small sources")
    if variant == "ORNL_COLLAPSED":
        ornl = set(prov_units.index[prov_units.ornl_only.astype(bool)].astype(str))
        return cl.where(~cl.isin(ornl), "ORNL-only")
    raise KeyError(variant)


BREADTH_VARIANTS = (("SMALL_POOLED", "sealed breadth diagnostic (never a gate)"),
                    ("ORNL_COLLAPSED", "sensitivity: ORNL-only clusters collapsed into one unit"),
                    ("RAW", "reference: the 13 cpg clusters unpooled (the replay's unit set)"))


def breadth(t: pd.DataFrame, cand: str, ref: str, prov_units: pd.DataFrame, design: str, stratum: str, key: str):
    """Section 4.1 item 6: equal-weight per-unit log SSE ratio mean(L) with SF_LOG (exact sign-flip on L) and IM t,
    per variant; plus publication-level relative SSE inside the giant cluster.  DIAGNOSTIC, never a gate."""
    lab = label(design, stratum, "breadth diagnostic, never a gate")
    summ, units = [], []
    for v, what in BREADTH_VARIANTS:
        tb = t.assign(_bu=breadth_units(t, prov_units, v).to_numpy())
        us = CM.unit_stats(tb, cand, ref, "_bu")
        ok = np.isfinite(us.L).all()
        sf = CM.sf_pool(us.L, f"{key}|breadth|{v}|SF_LOG") if ok else None
        im = CM.im_t(us.L)
        summ.append({"contrast": f"{cand} - {ref}", "variant": v, "role": what, "G": int(len(us)),
                     "units": ";".join(map(str, us.index)), "mean_L": float(us.L.mean()),
                     "equal_weight_relSSE": float(us.rel.mean()), "W/L/T": f"{int(us.win.sum())}/{int(us.loss.sum())}/{int(us.tie.sum())}",
                     "SF_LOG_p": sf["p"] if sf else float("nan"), "SF_LOG_exact": sf["exact"] if sf else None,
                     "IM_t": im["t"], "IM_p": im["p"], "label": lab})
        units.append(us.reset_index().rename(columns={"_bu": "unit"}).assign(contrast=f"{cand} - {ref}", variant=v, label=lab))
    sizes = t.groupby(UNIT).size()
    giant = str(sizes.idxmax())
    tg = t[t[UNIT].astype(str) == giant]
    up = CM.unit_stats(tg, cand, ref, "publication").rename(columns={"n": "rows_per_seed"})
    tot = float(up.dSSE.sum())
    up["share_of_giant_dSSE"] = up.dSSE / tot if tot else np.nan
    up = up.reset_index().assign(contrast=f"{cand} - {ref}", giant_cluster=giant,
                                 giant_share_of_rows=float(len(tg) / len(t)), label=lab)
    return pd.DataFrame(summ), pd.concat(units, ignore_index=True), up


# ------------------------------------------------------------------------------------------------------------
# M08 appendix: relation-class moment table with an identifiability column (section 4.1 item 8)
# ------------------------------------------------------------------------------------------------------------
def registry_status(tag: str = "M08", path=REGISTRY) -> dict:
    """Status of the REGISTRY.csv row that names ``tag`` (PLANNED / EXCLUDED / ...); MISSING when absent."""
    if not path.exists():
        return {"status": "MISSING", "reason": f"{path.name} does not exist"}
    r = pd.read_csv(path, keep_default_na=False, dtype=str)
    hit = r[r.apply(lambda row: any(tag in str(v) for v in row.values), axis=1)]
    if hit.empty:
        return {"status": "MISSING", "reason": f"no {path.name} row names {tag}"}
    scol = next((c for c in r.columns if "status" in c.lower()), None)
    if scol is None:
        return {"status": "MISSING", "reason": f"{path.name} has no status column"}
    st = sorted({str(s).strip().upper() for s in hit[scol]})
    return {"status": st[0] if len(st) == 1 else "AMBIGUOUS:" + ",".join(st), "rows": hit.to_dict("records")}


def _m08_seed(g: pd.DataFrame, resid_of: str):
    cells = g.drop_duplicates("cell_id")[["cell_id", "extractant", "publication", "chemotype", UNIT]].reset_index(drop=True)
    n = len(cells)
    ci = dict(zip(cells.cell_id, range(n)))
    pj = {p: j for j, p in enumerate(PAIRS)}
    Ex = np.full((n, len(PAIRS)), np.nan)
    Ex[[ci[c] for c in g.cell_id], [pj[f"{a}/{b}"] for a, b in zip(g.A, g.B)]] = (g.y - g[resid_of]).to_numpy(float)
    F = np.isfinite(Ex).astype(float)
    E0 = np.nan_to_num(Ex)
    P, Q, N, Aabs = E0 @ E0.T, (E0 ** 2) @ F.T, F @ F.T, np.abs(E0) @ np.abs(E0).T
    ext, pub, chem = (cells[c].astype(str).to_numpy() for c in ("extractant", "publication", "chemotype"))
    cl = cells[UNIT].astype(str).to_numpy()
    se, sp, sc = ext[:, None] == ext[None, :], pub[:, None] == pub[None, :], chem[:, None] == chem[None, :]
    up = np.triu(np.ones((n, n), bool), 1) & (N > 0)
    masks = {"SE_SP": se & sp, "SE_OP": se & ~sp, "OE_SP_SC": ~se & sp & sc, "OE_SP_OC": ~se & sp & ~sc,
             "OE_OP_SC": ~se & ~sp & sc, "OE_OP_OC": ~se & ~sp & ~sc}

    def rho(M):
        di, dj = float(Q[M].sum()), float(Q.T[M].sum())
        return float(P[M].sum() / math.sqrt(di * dj)) if di > 0 and dj > 0 else float("nan")
    out, prods, loco = {}, [], []
    for c, m in masks.items():
        M = m & up
        ii, jj = np.nonzero(M)
        keys = [a if a == b else "cross:" + "|".join(sorted((a, b))) for a, b in zip(cl[ii], cl[jj])]
        out[c] = {"rho": rho(M), "cell_pairs": int(M.sum()), "products": float(N[M].sum()),
                  "publications": int(len(set(pub[ii]) | set(pub[jj]))), "clusters": int(len(set(cl[ii]) | set(cl[jj])))}
        prods.append(pd.DataFrame({"class": c, "unit": keys, "abs": Aabs[ii, jj]}))
        for gname in sorted(set(cl)):
            keep = cl != gname
            loco.append({"class": c, "cluster_left_out": gname, "rho": rho(M & keep[:, None] & keep[None, :])})
    return out, pd.concat(prods, ignore_index=True), pd.DataFrame(loco)


def m08_moments(t: pd.DataFrame, design: str, stratum: str, resid_of: str = "B1") -> pd.DataFrame:
    """Relation-class moments of B1's residual e = y - B1 on E-A rows: for cell pairs of a relation class sharing a
    strict-adjacent pair type, rho = sum e_ip e_jp / sqrt(sum e_ip^2 sum e_jp^2) (the shared share of the pair-residual
    second moment), per seed then the seed mean.  Identifiability: product units (a cluster when both cells share it,
    else the cross-cluster pair), the share of sum |e_ip e_jp| held by the top unit, the leave-one-cluster-out range;
    'identified' needs >= 3 units and a top-unit share <= 0.80."""
    per, prods, locos = [], [], []
    for s, g in t.groupby("split_seed"):
        o, pr, lo = _m08_seed(g, resid_of)
        per.append(o)
        prods.append(pr)
        locos.append(lo.assign(seed=s))
    pr = pd.concat(prods, ignore_index=True).groupby(["class", "unit"])["abs"].sum()
    lo = pd.concat(locos, ignore_index=True).groupby(["class", "cluster_left_out"]).rho.mean()
    lab = label(design, stratum, "M08 appendix")
    rows = []
    for c, what in M08_CLASSES:
        r = np.array([o[c]["rho"] for o in per], float)
        u = pr.loc[c] if c in pr.index.get_level_values(0) else pd.Series(dtype=float)
        tot = float(u.sum())
        top = str(u.idxmax()) if len(u) and tot > 0 else ""
        share = float(u.max() / tot) if len(u) and tot > 0 else float("nan")
        lc = lo.loc[c].dropna() if c in lo.index.get_level_values(0) else pd.Series(dtype=float)
        ident = bool(len(u) >= M08_MIN_UNITS and np.isfinite(share) and share <= M08_MAX_TOP)
        rows.append({"class": c, "relation": what, "rho": float(np.nanmean(r)) if np.isfinite(r).any() else float("nan"),
                     "rho_per_seed": ";".join(f"{v:.4f}" for v in r),
                     "cell_pairs_per_seed": float(np.mean([o[c]["cell_pairs"] for o in per])),
                     "products_per_seed": float(np.mean([o[c]["products"] for o in per])),
                     "publications": int(max(o[c]["publications"] for o in per)),
                     "clusters": int(max(o[c]["clusters"] for o in per)), "product_units": int(len(u)),
                     "top_unit": top, "top_unit_share": share,
                     "loco_min": float(lc.min()) if len(lc) else float("nan"),
                     "loco_min_by": str(lc.idxmin()) if len(lc) else "",
                     "loco_max": float(lc.max()) if len(lc) else float("nan"),
                     "loco_max_by": str(lc.idxmax()) if len(lc) else "",
                     "identifiability": "identified" if ident else
                     "not identified (< 3 units or one unit holds > 80 % of the identifying products)",
                     "label": lab})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------
# one design's E-A tables and the six-design table
# ------------------------------------------------------------------------------------------------------------
DUP_VARIANTS = (("noV2", "V2-1/2/3 cells dropped", "drop"),
                ("noBITDUP", "second cell of every Stage-0 flagged bit-identity link dropped", "drop"),
                ("V2-3", "V2-3 condition-aware stratum: the 11 C3 hidden-merge cells reported apart (S1 open)", "keep"),
                ("without_V2-3", "the rows without the V2-3 stratum", "drop:V2-3"))


DUP_COLUMNS = ("cells", "publications", "rows_per_seed", "G", "R2_cand", "R2_ref", "dR2", "relSSE", "SF_POOL_p",
               "dMAE_bal", "W/L/T", "gate_pass")


def duplicate_table(t: pd.DataFrame, design: str) -> pd.DataFrame:
    """Duplicate sensitivities of every E-A contrast (registry 1.0-S07..S09; brief 5.3, 9.3): the M35 quantities on
    each duplicate variant of the C5-in rows.  SENSITIVITY; the V2-3 stratum is a handful of cells (descriptive).
    A variant inside one unit (G = 1: the V2-3 stratum, 11 cells in g:pub_74428aeab2) keeps its descriptive
    quantities (relative SSE, R2 of both arms, dR2, dMAE_bal); SF_POOL, W/L/T and the gate are marked not computable."""
    from . import population as PO
    sets = PO.duplicate_cell_sets()
    rows = []
    for cand, ref, _ in CONTRASTS:
        for v, desc, how in DUP_VARIANTS:
            cells = sets[v.split("_", 1)[1] if how == "drop:V2-3" else v]
            sub = t[t.cell_id.astype(str).isin(cells)] if how == "keep" else t[~t.cell_id.astype(str).isin(cells)]
            base = {"design": design, "variant": v, "definition": desc, "contrast": f"{cand} - {ref}",
                    "cells_dropped": int(t.cell_id.nunique() - sub.cell_id.nunique()) if how != "keep" else None,
                    "cells_kept": int(sub.cell_id.nunique()),
                    "label": label(design, "C5_in", L_SENS + " (duplicate sensitivity)")}
            if sub.empty:
                rows.append({**base, "note": "no rows: not computed"})
                continue
            row, _ = contrast_stats(sub, cand, ref, f"{cand} - {ref}|{design}|DUP|{v}")
            vals = {k: row[k] for k in DUP_COLUMNS}
            if row["G"] < 2:                                 # one unit: no sign-flip, no unit W/L/T, no gate
                na = f"not computable (G = {row['G']})"
                vals.update({"SF_POOL_p": float("nan"), "W/L/T": na, "gate_pass": na})
                base["label"] = label(design, "C5_in", L_SENS + " (duplicate sensitivity, descriptive)")
                base["note"] = (f"G = {row['G']} {UNIT} unit: SF_POOL, W/L/T and the gate not computable; relative SSE, "
                                "R2 of both arms, dR2 and dMAE_bal descriptive")
            rows.append({**base, **vals})
    return pd.DataFrame(rows)


def score_design(t: pd.DataFrame, prov: pd.DataFrame, design: str, *, m08: bool = False) -> dict:
    """Every E-A table of one design (BP / BPG).  Returns frames keyed by output name (without design suffix)."""
    t = add_big_fold(t)
    strata = CM.c5_split(t)
    gate, units, perseed, bigf, lopo_s, lopo_v, br_s, br_u, br_g, m8 = [], [], [], [], [], [], [], [], [], []
    for cand, ref, role in CONTRASTS:
        cname = f"{cand} - {ref}"
        for st in STRATA:
            ts = strata[st]
            key = f"{cname}|{design}|{st}"
            row, us = contrast_stats(ts, cand, ref, key)
            pu = unit_provenance(ts, prov)
            gate.append({"design": design, "stratum": st, "role": role, **row, "label": label(design, st)})
            units.append(unit_table(us, pu, label(design, st)).reset_index().assign(design=design, stratum=st, contrast=cname))
            rs, rc, rr = CM.rel_sse_seeds(ts, cand, ref), CM.r2_seeds(ts, cand), CM.r2_seeds(ts, ref)
            dm = CM.bal_mae_seeds(ts, ref) - CM.bal_mae_seeds(ts, cand)
            for s in rs.index:
                perseed.append({"design": design, "stratum": st, "contrast": cname, "split_seed": int(s),
                                "relSSE": float(rs[s]), "R2_cand": float(rc[s]), "R2_ref": float(rr[s]),
                                "dR2": float(rc[s] - rr[s]), "dMAE_bal": float(dm[s]), "label": label(design, st)})
            if st != "C5_cell_out":
                b_s, b_u, b_g = breadth(ts, cand, ref, pu, design, st, key)
                br_s.append(b_s.assign(design=design, stratum=st))
                br_u.append(b_u.assign(design=design, stratum=st))
                br_g.append(b_g.assign(design=design, stratum=st))
        bigf.append(bigfold_table(strata["C5_in"], cand, ref, design, f"{cname}|{design}|C5_in").assign(contrast=cname))
        for st in ("C5_in", "C5_out"):
            j, v = lopo_table(strata[st], cand, ref)
            lopo_s.append({"design": design, "stratum": st, "contrast": cname, **j, "label": label(design, st)})
            lopo_v.append(v.assign(design=design, stratum=st, contrast=cname, label=label(design, st)))
    if m08:
        for st in ("C5_in", "C5_out"):
            m8.append(m08_moments(strata[st], design, st).assign(design=design, stratum=st))
    gt = pd.DataFrame(gate)
    c5 = gt[["design", "stratum", "contrast", "cells", "publications", "rows_per_seed", "G", "R2_cand", "R2_ref", "dR2",
             "relSSE", "SF_POOL_p", "dMAE_bal", "gate_pass", "label"]]
    out = {"gate": gt, "c5": c5, "units": pd.concat(units, ignore_index=True), "perseed": pd.DataFrame(perseed),
           "bigfold": pd.concat(bigf, ignore_index=True), "lopo": pd.DataFrame(lopo_s),
           "lopo_values": pd.concat(lopo_v, ignore_index=True), "breadth": pd.concat(br_s, ignore_index=True),
           "breadth_units": pd.concat(br_u, ignore_index=True), "giant_publications": pd.concat(br_g, ignore_index=True),
           "arms": pd.concat([CM.arm_table(strata[st], ["B0", "B1", "HIST_D7"]).assign(design=design, stratum=st,
                                                                                   label=label(design, st))
                              for st in ("C5_in", "C5_out")], ignore_index=True)}
    out["duplicates"] = duplicate_table(strata["C5_in"], design)
    if m08:
        out["m08_moments"] = pd.concat(m8, ignore_index=True)
    return out


def six_design(rows: dict, prov: pd.DataFrame) -> dict:
    """The six-design table (section 4.1 item 5), training-free arms only: B0 and HIST_D7 levels and the HIST_D7 - B0
    contrast (M35 quantities, LOPO range) on A, B, BR, BQ, BP, BPG, C5-BTBP in and out."""
    cand, ref = SIX_CONTRAST
    lv, ct, un = [], [], []
    for d in SIX:
        t = rows[d]
        for st in ("C5_in", "C5_out"):
            ts = CM.c5_split(t)[st]
            lab = label(d, st, "training-free arms only")
            for a in (ref, cand):
                rs = CM.r2_seeds(ts, a)
                lv.append({"design": d, "stratum": st, "arm": a, "R2": float(rs.mean()),
                           "R2_per_seed": ";".join(f"{v:.4f}" for v in rs), "MSE": CM.mse_seedmean(ts, a),
                           "MAE_bal": CM.bal_mae(ts, a), "rows_per_seed": float(len(ts) / _n_seeds(ts)), "label": lab})
            row, us = contrast_stats(ts, cand, ref, f"{cand} - {ref}|{d}|{st}")
            jk = CM.lopo(ts, lambda x: CM.rel_sse(x, cand, ref), by="publication")
            ct.append({"design": d, "stratum": st, **row, "lopo_min": jk["min"], "lopo_min_by": jk["min_by"],
                       "lopo_max": jk["max"], "lopo_max_by": jk["max_by"], "lopo_lo": jk["lo"], "lopo_hi": jk["hi"],
                       "label": lab})
            un.append(unit_table(us, unit_provenance(ts, prov), lab).reset_index().assign(design=d, stratum=st,
                                                                                         contrast=f"{cand} - {ref}"))
    return {"sixdesign_levels": pd.DataFrame(lv), "sixdesign": pd.DataFrame(ct),
            "sixdesign_units": pd.concat(un, ignore_index=True), "sixdesign_readsets": readset_table(rows)}


def pooled_rel_sse(t: pd.DataFrame, cand: str, ref: str) -> float:
    """1 - sum_s SSE_cand / sum_s SSE_ref, the SSE pooled over the seeds (the replay's 'pooled_rel'); M35's sealed
    currency is the seed mean of the per-seed ratios instead (common.rel_sse).  Descriptive, for the consistency note."""
    return float(1.0 - ((t[cand] - t.y) ** 2).sum() / ((t[ref] - t.y) ** 2).sum())


def consistency(gate: pd.DataFrame, rows: dict | None = None) -> dict:
    """CONSISTENCY CHECK against the seen replay of B1 - B0 (IR M35; PC P2), C5-BTBP in.  With ``rows`` (design ->
    E-A rows) the relative-SSE record also carries the pooled-over-seeds value and a note on the two definitions; the
    match is judged on the sealed seed mean, as registered (1.0-R03)."""
    out = {}
    for cname, by in SEEN.items():
        for d, want in by.items():
            r = gate[(gate.design == d) & (gate.stratum == "C5_in") & (gate.contrast == cname)]
            if r.empty:
                continue
            r = r.iloc[0]
            chk = {}
            if "relSSE_pct" in want:
                chk["relSSE_pct"] = {"seen": want["relSSE_pct"], "now": round(100 * float(r.relSSE), 2),
                                     "match": round(100 * float(r.relSSE), 2) == want["relSSE_pct"]}
                if rows is not None and d in rows:
                    cand, ref = cname.split(" - ")
                    pooled = pooled_rel_sse(CM.c5_split(rows[d])["C5_in"], cand, ref)
                    chk["relSSE_pct"].update({
                        "now_seed_mean": float(r.relSSE), "now_pooled_over_seeds": pooled,
                        "now_pooled_over_seeds_pct": round(100 * pooled, 2),
                        "note": (f"the seen {want['relSSE_pct']:.2f} % is the replay's pooled-over-seeds ratio 1 - "
                                 f"sum_s SSE_{cand} / sum_s SSE_{ref} (now {pooled:.5f}, {100 * pooled:.2f} %); the sealed "
                                 f"M35 currency compared here is the seed mean of the per-seed ratios ({float(r.relSSE):.5f}, "
                                 f"{100 * float(r.relSSE):.2f} %); the difference ({float(r.relSSE) - pooled:+.5f}) is "
                                 "definitional; match stays as registered")})
            if "SF_POOL_p" in want:
                chk["SF_POOL_p"] = {"seen": want["SF_POOL_p"], "now": round(float(r.SF_POOL_p), 5),
                                    "match": round(float(r.SF_POOL_p), 5) == want["SF_POOL_p"]}
            out[f"{cname} {d}"] = {"label": L_CC, **chk}
    return out


# ------------------------------------------------------------------------------------------------------------
# the section-9.3 per-item table (r5.common.per_item_table)
# ------------------------------------------------------------------------------------------------------------
NEXT_ITEM = "item 1.1 (E-CT core, results/ECT/); no E-A slot is registered (section 4.5)"


def _c5_in(gates: pd.DataFrame, design: str, contrast: str):
    g = gates[(gates.design == design) & (gates.stratum == "C5_in") & (gates.contrast == contrast)]
    return None if g.empty else g.iloc[0]


def _verdict(r) -> str:
    return (f"{'passes' if bool(r['gate_pass']) else 'fails'} the gate (relative SSE {float(r['relSSE']):+.4f}, SF_POOL "
            f"p {float(r['SF_POOL_p']):.5f}, guard {float(r['dMAE_bal']):+.5f})")


def outcome_sentences(gates: pd.DataFrame, decision: dict) -> tuple[str, str]:
    """The two section-9.3 sentences (what changed scientifically; the next registered step), built from the computed
    outcomes: the positive control on BPG (``decision``; not evaluated when BPG was not scored), M35's kill status,
    HIST_D7 - B1 and the BP sensitivity on the designs in ``gates``."""
    pc, kill = decision.get("positive_control"), decision.get("M35_kill_fires")
    if pc is None:
        s = "the M35 positive control is decided on BPG, which was not evaluated in this run (BPG not scored)"
    else:
        s = (f"the sealed M35 gate {'passes' if pc['gate_pass'] else 'fails'} its own positive control B1 - B0 on BPG "
             f"(relative SSE {float(pc['relSSE']):+.4f}, SF_POOL p {float(pc['SF_POOL_p']):.5f}, guard "
             f"{float(pc['dMAE_bal']):+.5f}), so M35's kill "
             + ("fires and the convention is abandoned (RI 4.2)" if kill else
                "does not fire and the convention is kept (DIAGNOSTIC)"))
    hd = _c5_in(gates, "BPG", "HIST_D7 - B1")
    if hd is not None:
        s += f"; HIST_D7 - B1 {_verdict(hd)} on BPG"
    bp = {c: _c5_in(gates, "BP", c) for c in ("B1 - B0", "HIST_D7 - B1")}
    if bp["B1 - B0"] is not None:
        s += (f"; BP sensitivity (same test folds): B1 - B0 {_verdict(bp['B1 - B0'])}, HIST_D7 - B1 "
              f"{_verdict(bp['HIST_D7 - B1'])}")
    lv = next((r for r in (_c5_in(gates, d, "B1 - B0") for d in paths.DESIGNS) if r is not None), None)
    changed = (f"Nothing on E-A (no E-A slot): {s}. "
               + (f"B1's own level {float(lv['R2_cand']):.3f} stands as E-A's reference; " if lv is not None else "")
               + "every number is a seen diagnostic.")
    if kill is None:
        nxt = f"The M35 decision (1.0-R01) needs the BPG run (BPG decides); the next registered step is {NEXT_ITEM}."
    elif kill:
        nxt = f"M35 is abandoned as the E-A convention (1.0-R01: kill fired); the next registered step is {NEXT_ITEM}."
    else:
        nxt = f"M35 stays the E-A decision convention (DIAGNOSTIC); the next registered step is {NEXT_ITEM}."
    return changed, nxt


def per_item(T: dict, *, pop: dict, leakage: str, compute: str, decision: dict, seal_tag: str) -> dict:
    """The section-9.3 per-item table (r5.common.per_item_table).  ``T`` maps design -> that design's frames (the
    output of score_design, or the stored <name>_<design>.csv tables; gate, perseed, lopo, arms and duplicates are
    used): one design gives PER_ITEM_TABLE_<design>.md, BPG and BP together ITEM_TABLE.md.  ``decision`` is the run's
    decision record (positive_control None when BPG was not scored)."""
    ds = [d for d in paths.DESIGNS if d in T]
    gates = pd.concat([T[d]["gate"] for d in ds], ignore_index=True)
    arms = pd.concat([T[d]["arms"] for d in ds], ignore_index=True)
    perseed = pd.concat([T[d]["perseed"][T[d]["perseed"].stratum == "C5_in"] for d in ds], ignore_index=True)
    units = gates[gates.stratum == "C5_in"][["design", "contrast", "G", "W/L/T", "SF_POOL_p", "SIGN_p", "SF_LOG_p",
                                              "IM_REL_p", "gate_pass", "label"]]
    lp = pd.concat([T[d]["lopo"] for d in ds], ignore_index=True)
    lt = lp.merge(gates[["design", "stratum", "contrast", "top3_unit_share"]], on=["design", "stratum", "contrast"])
    lt = lt[["design", "stratum", "contrast", "point", "lo", "hi", "min", "min_by", "max", "max_by", "P",
             "top3_unit_share", "label"]]
    c5 = gates[["design", "stratum", "contrast", "rows_per_seed", "relSSE", "dR2", "SF_POOL_p", "dMAE_bal", "gate_pass",
                "label"]]
    dup = pd.concat([T[d]["duplicates"] for d in ds], ignore_index=True)
    dup = dup[[c for c in ("design", "variant", "contrast", "cells_kept", "rows_per_seed", "G", "relSSE", "SF_POOL_p",
                           "dR2", "dMAE_bal", "gate_pass", "label", "note") if c in dup]]
    if "note" in dup:
        dup = dup.assign(note=dup.note.fillna(""))
    changed, nxt = outcome_sentences(gates, decision)
    both = len(ds) > 1
    design = ("BPG decides; BP sensitivity (same test folds)" if both else
              "BPG (decides)" if ds == ["BPG"] else f"{ds[0]} (SENSITIVITY; same test folds as BPG, which decides)")
    lab = (f"{L_SEEN}; {L_DIAG} (E-A contrasts after M34's kill); "
           + ("BP and C5-out rows SENSITIVITY" if both or ds == ["BP"] else "C5-out rows SENSITIVITY")
           + f"; HIST_D7 not fold-independent; seal {seal_tag}")
    return {
        "item": f"1.0 EA_CHECK{'' if both else f' ({ds[0]})'}: M35 positive-control and six-design check; estimand E-A "
                "(DIAGNOSTIC, seen)",
        "population": pop,
        "design": design + "; six-design table A, B, BR, BQ, BP, BPG for the training-free arms B0 and HIST_D7 only "
                           "(sixdesign.csv)",
        "read_budget": "four reads (code budget k = 3), DOPT_Q nested read sets of each design's fold covariance",
        "arms": "B0, B1, HIST_D7; contrasts B1 - B0 (positive control, outside any family) and HIST_D7 - B1 under the "
                "sealed M35 gate; HIST_D7 - B0 on the six designs",
        "metrics": arms, "per_seed": perseed, "units": units, "lopo_top3": lt, "c5": c5, "duplicates": dup,
        "leakage": leakage, "compute": compute, "label": lab, "changed": changed, "next_step": nxt,
    }


# ------------------------------------------------------------------------------------------------------------
# reproduction target (section 6.0)
# ------------------------------------------------------------------------------------------------------------
def repro(bench, designs=paths.DESIGNS, *, fold_filter=None, log=None) -> dict:
    """B1 from round 4's stored fold betas (results/REF/fitted_<design>.json), B0 and HIST_D7, recomputed through
    fold_rows on the gen21 caches, against the REF parquet on identical keys.  Pass: every |difference| <= 8.9e-16,
    identical keys, y and read sets, and (full run) 18,020 rows.  TECHNICAL SMOKE (prediction identity, no score)."""
    out, total = {"label": L_SMOKE, "tolerance": REPRO_TOL, "designs": {}}, 0
    for d in designs:
        fit = json.loads((paths.R4RES / "REF" / f"fitted_{d}.json").read_bytes().decode("utf-8"))
        rows, iso = [], 0
        for f in folds_for(bench.frame, d, seeds=list(paths.DEV_SEEDS)):
            if fold_filter is not None and (f.seed, f.fold) not in fold_filter:
                continue
            rec = GC.load(d, f.seed, f.fold)
            E.assert_test_isolation(bench, rec)
            iso += 1
            beta = float(fit[f"{f.seed}_{f.fold}"]["ref:B1"]["coef"][0])
            rows += fold_rows(bench, rec, {"B0": b0_levels, "HIST_D7": hist_d7_levels, "B1": b1_method(beta)})
        new = pd.DataFrame(rows)
        new = new[(new.k == K) & (new.k_max >= 3)].sort_values(CM.KEY).reset_index(drop=True)
        ref = CM.load_rows(d)
        if fold_filter is not None:
            ref = ref[[(s, f) in fold_filter for s, f in zip(ref.split_seed, ref.fold)]].reset_index(drop=True)
        same = len(new) == len(ref) and (new[CM.KEY].to_numpy() == ref[CM.KEY].to_numpy()).all()
        rec_d = {"rows": int(len(new)), "rows_ref": int(len(ref)), "folds_checked_isolation": iso, "same_keys": bool(same)}
        if same:
            rec_d["same_y"] = bool(np.array_equal(new.y.to_numpy(float), ref.y.to_numpy(float)))
            rec_d["same_read_set"] = bool((new.read_set.to_numpy() == ref.read_set.to_numpy()).all())
            for c in ("B1", "B0", "HIST_D7"):
                rec_d[f"max_abs_diff_{c}"] = float(np.max(np.abs(new[c].to_numpy(float) - ref[c].to_numpy(float))))
        rec_d["passed"] = bool(same and rec_d.get("same_y") and rec_d.get("same_read_set")
                               and all(rec_d.get(f"max_abs_diff_{c}", np.inf) <= REPRO_TOL for c in ("B1", "B0", "HIST_D7")))
        out["designs"][d] = rec_d
        total += len(new)
        if log:
            log(f"  repro {d}: {len(new)} rows, passed {rec_d['passed']}")
    out["rows"] = int(total)
    out["full"] = fold_filter is None
    out["passed"] = bool(all(v["passed"] for v in out["designs"].values())
                         and (fold_filter is not None or total == REPRO_ROWS))
    return out


def env_note() -> str:
    return f"python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}"

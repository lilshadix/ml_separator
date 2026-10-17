"""gen20.1 fold plans: one ``gen20k.splits.folds_for`` call per design, the 4-fold chemotype inner
cross-fit, the COH-NODMD cell drop, and the fold-identity gates (a)-(e) of IMPLEMENTATION_PLAN.md §4.

* ``build`` calls ``gen20k.splits.folds_for(frame, design, seeds)`` exactly once per design (it runs
  ``assert_design_integrity``); the returned Fold objects are the only folds any arm may use.
* ``inner_folds`` deals the outer training cells into ``N_INNER`` = 4 chemotype folds with
  ``seeded_group_kfold(chemotype[train], 4, gen20k.splits.inner_seed(seed, fold))`` (the split of
  ``gen20k.cov.inner_oof_curves``, cov.py:75-90), re-applies the design mask inside each inner fold
  with ``gen20k.cov.inner_mask`` and flags an inner fold with < 20 masked inner-training cells as
  skipped.  Design A is regrouped by chemotype too (its stored extractant-grouped ``Fold.inner_*``
  is never used; plan §10 row 25).
* ``gen16_digest`` re-implements gen16's ``fold_plan_digest`` (generations/gen16_leads/gen16/foldhash.py,
  never imported: two packages share the name ``gen16``).
* ``plan_digest`` is the new gate-(e) digest over outer and inner folds of the six designs.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import bootstrap
from gen13sep.splits import SPLIT_SEEDS, Fold  # noqa: E402
from gen20k import splits as GS  # noqa: E402
from gen20k.cov import inner_mask  # noqa: E402
from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402

DESIGNS: tuple[str, ...] = ("BP", "B", "BR", "BQ", "BPG", "A")
GEN16_DESIGN_ORDER: tuple[str, ...] = ("B", "BR", "BQ", "A", "BP")     # gen15.valuebench.DESIGNS
N_INNER = GS.N_INNER
MIN_INNER_TRAIN = 20
SEEDS: tuple[int, ...] = tuple(int(s) for s in SPLIT_SEEDS)

# gate (a)/(b) references: generations/gen16_leads/results/anchors/anchors.json:31-40
EXPECTED_TEST_PLAN_SHA256 = "7046c640676f4bf8b065393a1291464231340e7bccd33b31028010bc6d440d16"
EXPECTED_TRAIN_TEST_SHA256 = "9a24b6e985c7515601e247ae04de5068f5c76974ce4992ce1bd2dcaad13c8054"
_B_FAMILY = "b50c9fd6d419a840700cc076e59769af1c00499bf273f1d12a6df2773f0b28ba"
EXPECTED_PER_DESIGN_SHA256 = {"B": _B_FAMILY, "BR": _B_FAMILY, "BQ": _B_FAMILY, "BP": _B_FAMILY,
                              "A": "afda7475714a942972b88c9416bbd22c5e069d18ced8842382a3387ba8e139e7"}
ANCHORS_JSON = bootstrap.GENERATIONS / "gen16_leads" / "results" / "anchors" / "anchors.json"
FOLDS_BP_JSON = bootstrap.G20 / "predictions" / "discovery" / "folds_BP.json"


class FoldGateError(AssertionError):
    """STOP ``INCOMPARABLE_FOLDS``."""


@dataclass(frozen=True)
class InnerFold:
    train_pos: np.ndarray          # masked inner-training cell indices (bench rows), sorted
    val_pos: np.ndarray            # inner-validation cell indices, sorted
    skipped: bool                  # < MIN_INNER_TRAIN masked inner-training cells (or no validation cell)
    j: int = 0                     # inner fold number (dealing order)
    train_pos_unmasked: np.ndarray | None = None


# --------------------------------------------------------------------------------------
# outer folds
# --------------------------------------------------------------------------------------
def _frame(bench_or_frame) -> pd.DataFrame:
    return bench_or_frame if isinstance(bench_or_frame, pd.DataFrame) else bench_or_frame.frame


def build(bench_or_frame, seeds=SEEDS, designs=DESIGNS) -> dict[str, list[Fold]]:
    """``{design: [Fold]}``, exactly one ``gen20k.splits.folds_for`` call per design."""
    frame = _frame(bench_or_frame)
    out = {}
    for d in designs:
        out[d] = list(GS.folds_for(frame, d, seeds=tuple(seeds)))
    return out


def fold_record(bench_or_frame, folds_by_design) -> pd.DataFrame:
    """``gen20k.splits.fold_record`` for every design, concatenated (cell-level plan)."""
    frame = _frame(bench_or_frame)
    return pd.concat([GS.fold_record(frame, folds_by_design[d]) for d in folds_by_design], ignore_index=True)


def drop_cells(fold: Fold, cells) -> Fold:
    """COH-NODMD: the primary Fold with ``cells`` removed from train_index and the stored inner indices;
    test_index is unchanged (scored rows are removed by the row manifests).  Inner cross-fit folds for
    the NODMD variant must come from ``inner_folds(bench, design, primary_fold, drop=cells)``, never
    from a re-dealt reduced training set."""
    drop = np.asarray(sorted(set(int(c) for c in cells)), dtype=int)
    keep = lambda idx: np.asarray(idx)[~np.isin(np.asarray(idx), drop)]  # noqa: E731
    return replace(fold, train_index=keep(fold.train_index), inner_train_index=keep(fold.inner_train_index),
                   inner_validation_index=keep(fold.inner_validation_index))


# --------------------------------------------------------------------------------------
# inner chemotype cross-fit
# --------------------------------------------------------------------------------------
def inner_folds(bench, design: str, fold: Fold, *, drop=None, min_train: int | None = None) -> list[InnerFold]:
    """The 4 masked chemotype inner folds of ``fold``'s training cells (all four, skipped ones flagged).

    ``drop`` (COH-NODMD) removes cells from every masked inner-training and inner-validation set after
    the dealing of the *primary* fold, so the inner partition is never re-dealt."""
    if fold.design != design:
        raise ValueError(f"fold design {fold.design} != {design}")
    min_train = MIN_INNER_TRAIN if min_train is None else int(min_train)
    tr = np.asarray(fold.train_index, dtype=int)
    chem = bench.frame["chemotype"].astype(str).to_numpy()
    dropset = np.asarray(sorted(set(int(c) for c in (drop if drop is not None else []))), dtype=int)
    out = []
    for j, (tr_rel, va_rel) in enumerate(seeded_group_kfold(chem[tr], N_INNER, GS.inner_seed(fold.seed, fold.fold))):
        itr_raw, iva = tr[tr_rel], tr[va_rel]
        itr = np.asarray(inner_mask(bench, design, itr_raw, iva), dtype=int)
        if dropset.size:
            itr = itr[~np.isin(itr, dropset)]
            iva = iva[~np.isin(iva, dropset)]
            itr_raw = itr_raw[~np.isin(itr_raw, dropset)]
        itr, iva = np.sort(itr), np.sort(iva)
        if set(itr.tolist()) & set(iva.tolist()):
            raise FoldGateError("inner train/validation overlap")
        out.append(InnerFold(train_pos=itr, val_pos=iva, skipped=bool(len(itr) < min_train or len(iva) == 0),
                             j=j, train_pos_unmasked=np.sort(itr_raw)))
    return out


def first_inner_matches(bench, fold: Fold) -> bool:
    """Does the first chemotype inner fold (before inner_mask) equal the stored Fold.inner_* indices?"""
    tr = np.asarray(fold.train_index, dtype=int)
    chem = bench.frame["chemotype"].astype(str).to_numpy()
    a, b = next(iter(seeded_group_kfold(chem[tr], N_INNER, GS.inner_seed(fold.seed, fold.fold))))
    return bool(set(tr[a].tolist()) == set(np.asarray(fold.inner_train_index).tolist())
                and set(tr[b].tolist()) == set(np.asarray(fold.inner_validation_index).tolist()))


# --------------------------------------------------------------------------------------
# gates (a)-(e)
# --------------------------------------------------------------------------------------
def gen16_digest(folds_by_design) -> dict:
    """gen16's fold_plan_digest, re-implemented: in design order (B, BR, BQ, A, BP), ``sha256`` over the
    concatenated sorted int64 test-index arrays of every fold, ``sha256_train_test`` with each fold's
    sorted train index appended after its test index, and one test-index hash per design."""
    h_test, h_full = hashlib.sha256(), hashlib.sha256()
    per_design, n_folds = {}, {}
    for d in GEN16_DESIGN_ORDER:
        h_d = hashlib.sha256()
        for f in folds_by_design[d]:
            te = np.sort(np.asarray(f.test_index, dtype=np.int64)).tobytes()
            tr = np.sort(np.asarray(f.train_index, dtype=np.int64)).tobytes()
            h_d.update(te)
            h_test.update(te)
            h_full.update(te)
            h_full.update(tr)
        per_design[d] = h_d.hexdigest()
        n_folds[d] = len(folds_by_design[d])
    return {"sha256": h_test.hexdigest(), "sha256_train_test": h_full.hexdigest(),
            "per_design": per_design, "n_folds": n_folds, "designs": list(GEN16_DESIGN_ORDER)}


def anchors_reference() -> dict:
    """The recorded gen16 digests (read-only), cross-checked against the constants above."""
    rec = json.loads(ANCHORS_JSON.read_text(encoding="utf-8"))["fold_plan"]
    return {"sha256": rec["sha256"], "sha256_train_test": rec["sha256_train_test"], "per_design": rec["per_design"]}


def gate_a(dig: dict) -> dict:
    ok = dig["sha256"] == EXPECTED_TEST_PLAN_SHA256 and dig["sha256_train_test"] == EXPECTED_TRAIN_TEST_SHA256
    res = {"gate": "a", "pass": bool(ok), "sha256": dig["sha256"], "expected_sha256": EXPECTED_TEST_PLAN_SHA256,
           "sha256_train_test": dig["sha256_train_test"], "expected_sha256_train_test": EXPECTED_TRAIN_TEST_SHA256}
    return res


def gate_b(dig: dict) -> dict:
    bad = {d: dig["per_design"].get(d) for d, v in EXPECTED_PER_DESIGN_SHA256.items() if dig["per_design"].get(d) != v}
    return {"gate": "b", "pass": not bad, "per_design": dig["per_design"], "expected": EXPECTED_PER_DESIGN_SHA256,
            "mismatched": bad}


_SUBPROCESS_CODE = r"""
import json, sys
sys.dont_write_bytecode = True
sys.path.insert(0, %r)
from gen201 import data, folds
b = data.load(check=False)
F = folds.build(b)
d = folds.gen16_digest(F)
e = folds.plan_digest(b, F)
print(json.dumps({"gen16": d, "plan": {"sha256": e["sha256"], "per_design": e["per_design"]}}))
"""


def digests_subprocess(hash_seed: str) -> dict:
    """gen16 digest (gates a/b) and the gate-(e) plan digest computed in a fresh interpreter under
    ``PYTHONHASHSEED=hash_seed`` (no bytecode written)."""
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed), "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-B", "-c", _SUBPROCESS_CODE % str(bootstrap.G201)], capture_output=True,
                       text=True, env=env, cwd=str(bootstrap.ROOT))
    if r.returncode != 0:
        raise FoldGateError(f"fold-digest subprocess failed (PYTHONHASHSEED={hash_seed}): {r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def gen16_digest_subprocess(hash_seed: str) -> dict:
    return digests_subprocess(hash_seed)["gen16"]


def gate_b_subprocess(hash_seeds=("1", "2"), *, plan_sha256: str | None = None) -> dict:
    """Gates (a) and (b) recomputed in fresh interpreters under PYTHONHASHSEED 1 and 2; the gate-(e) plan
    digest is recomputed there too and must equal ``plan_sha256`` (the in-process value) when given."""
    runs = {}
    for hs in hash_seeds:
        out = digests_subprocess(hs)
        dig = out["gen16"]
        runs[str(hs)] = {"sha256": dig["sha256"], "sha256_train_test": dig["sha256_train_test"],
                         "per_design": dig["per_design"], "gate_a": gate_a(dig)["pass"], "gate_b": gate_b(dig)["pass"],
                         "plan_sha256": out["plan"]["sha256"], "plan_per_design": out["plan"]["per_design"]}
    ok = all(v["gate_a"] and v["gate_b"] for v in runs.values())
    same = len({json.dumps(v, sort_keys=True) for v in runs.values()}) == 1
    plan_ok = plan_sha256 is None or all(v["plan_sha256"] == plan_sha256 for v in runs.values())
    return {"gate": "b_subprocess", "pass": bool(ok and same and plan_ok), "identical_across_hash_seeds": same,
            "plan_digest_equals_in_process": bool(plan_ok) if plan_sha256 is not None else None, "runs": runs}


def gate_c(folds_bp, path=FOLDS_BP_JSON) -> dict:
    """25 BP (seed, fold, n_train, n_test) equal gen20's folds_BP.json (only those four fields are read)."""
    ref = {}
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    for x in rows:
        ref[(int(x["split_seed"]), int(x["fold"]))] = (int(x["n_train"]), int(x["n_test"]))
    got = {(int(f.seed), int(f.fold)): (int(len(f.train_index)), int(len(f.test_index))) for f in folds_bp}
    matches = sum(1 for k in got if ref.get(k) == got[k])
    ok = len(ref) == 25 and len(got) == 25 and set(ref) == set(got) and matches == 25
    return {"gate": "c", "pass": bool(ok), "n_reference": len(ref), "n_folds": len(got), "n_match": matches,
            "mismatched": {f"{k[0]}/{k[1]}": {"g201": got[k], "gen20": ref.get(k)} for k in got if ref.get(k) != got[k]}}


def gate_d(folds_by_design) -> dict:
    """Design B/BR/BQ/BPG test sets equal design BP's in all 25 (seed, fold); design A's differ."""
    bp = {(f.seed, f.fold): frozenset(np.asarray(f.test_index).tolist()) for f in folds_by_design["BP"]}
    eq = {}
    for d in ("B", "BR", "BQ", "BPG", "A"):
        eq[d] = int(sum(frozenset(np.asarray(f.test_index).tolist()) == bp[(f.seed, f.fold)] for f in folds_by_design[d]))
    ok = all(eq[d] == 25 for d in ("B", "BR", "BQ", "BPG")) and eq["A"] == 0
    return {"gate": "d", "pass": bool(ok), "n_equal_to_BP": eq}


def _upd(h, label: str, arr) -> None:
    a = np.sort(np.asarray(arr, dtype=np.int64))
    h.update(f"{label}:{a.size}:".encode())
    h.update(a.tobytes())


def plan_digest(bench, folds_by_design, designs=DESIGNS) -> dict:
    """Gate (e): SHA-256 over (design, seed, fold, sorted test, sorted train, and the masked
    (inner-train, inner-validation) index pairs of all four chemotype inner folds, skipped folds
    included and flagged), per design and over the six designs in ``designs`` order."""
    h_all = hashlib.sha256()
    per_design, inner_summary = {}, {}
    for d in designs:
        h_d = hashlib.sha256()
        rows = []
        for f in folds_by_design[d]:
            for h in (h_d, h_all):
                h.update(f"|{d}|{int(f.seed)}|{int(f.fold)}|".encode())
                _upd(h, "test", f.test_index)
                _upd(h, "train", f.train_index)
            for inn in inner_folds(bench, d, f):
                for h in (h_d, h_all):
                    h.update(f"|inner{inn.j}|skipped={int(inn.skipped)}|".encode())
                    _upd(h, "itrain", inn.train_pos)
                    _upd(h, "ival", inn.val_pos)
                rows.append({"seed": int(f.seed), "fold": int(f.fold), "inner": inn.j,
                             "n_inner_train": int(len(inn.train_pos)), "n_inner_val": int(len(inn.val_pos)),
                             "skipped": bool(inn.skipped)})
        per_design[d] = h_d.hexdigest()
        inner_summary[d] = rows
    return {"sha256": h_all.hexdigest(), "per_design": per_design, "designs": list(designs),
            "n_inner": N_INNER, "min_inner_train": MIN_INNER_TRAIN, "inner": inner_summary,
            "definition": "sha256 over |design|seed|fold| + test + train + per inner fold |inner j|skipped=0/1| + "
                          "itrain + ival, each array as 'label:size:' + sorted int64 bytes"}


def gate_e(plan: dict, *, recorded: dict | None = None, subprocess_runs: dict | None = None) -> dict:
    """Gate (e): the outer+inner plan digest of the six designs.  At S3 it is new (written to
    manifests/fold_plan_g201.json); it must be identical under PYTHONHASHSEED 1 and 2 and, in every later
    script, equal to the recorded value (``recorded`` = the "gate_e" block of fold_plan_g201.json)."""
    res = {"gate": "e", "sha256": plan["sha256"], "per_design": plan["per_design"]}
    ok = True
    if subprocess_runs is not None:
        same = all(v.get("plan_sha256") == plan["sha256"] for v in subprocess_runs.values())
        res["subprocess_identical"] = bool(same)
        ok &= same
    if recorded is not None:
        eq = recorded.get("sha256") == plan["sha256"] and all(
            recorded.get("per_design", {}).get(d) == h for d, h in plan["per_design"].items())
        res["equals_recorded"] = bool(eq)
        res["recorded_sha256"] = recorded.get("sha256")
        ok &= eq
    res["pass"] = bool(ok)
    return res


def recorded_plan(path=None) -> dict | None:
    """The gate-(e) block of manifests/fold_plan_g201.json, or None before S3 wrote it."""
    p = bootstrap.MANIFESTS / "fold_plan_g201.json" if path is None else path
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)["gate_e"]
    except FileNotFoundError:
        return None


def tested_once_per_seed(folds_by_design, n_cells: int) -> dict:
    """LT-01 / §1.2: test_index is never masked, so every cell is tested exactly once per seed."""
    out = {}
    for d, folds in folds_by_design.items():
        ok = True
        for s in sorted({f.seed for f in folds}):
            held = np.concatenate([np.asarray(f.test_index) for f in folds if f.seed == s])
            ok &= sorted(held.tolist()) == list(range(n_cells))
        out[d] = bool(ok)
    return out

"""Tests of the section 7 compute-plan item 6 re-coloured V5 fold files written by ``scripts/g19_build_folds_max4.py``
(``folds/V5__<variant>__batched_max4``; pre-registration section 7 item 6, section 3.1).

On the built files (no MODEL rows are loaded; the tests skip when the files are not built):

1. every fold hash and design hash re-derives (``io.read_design`` verifies), the scheme is ``batched_max4`` and every
   batch carries ``meta.max_cells_per_batch == 4``;
2. the largest batch of every seed and half holds <= 4 cells (files, per-file statistics and INDEX placeholder agree);
3. per seed the batches cover exactly the cells of the registered ``V5__<variant>__batched`` file of that seed (every
   scored cell exactly once; carved-out cells never batched);
4. no two cells of a batch share a metal state or a system, every cell of a batch lies in the batch's half;
5. INDEX.json carries the three designs with the files' design hashes, the placeholder
   ``V5_batches_per_discovery_seed_max4`` and the reading ``v5_batched_max4``; the manifest agrees;
6. every registered design hash of the full build (``manifests/g19_build_folds.json``) and of the incremental build
   (``manifests/g19_build_folds_incremental.json``) is unchanged in INDEX.json, in the files' summaries and re-derived
   from the files; the recorded digests of every registered fold file still match the bytes on disk;
7. the set of stems the plan can request under the heavy scheme ``batched_max4`` equals the built set.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gen19ct import paths
from gen19ct.folds import io as FI

FOLDS = paths.FOLDS_DIR
INDEX = FOLDS / "INDEX.json"
VARIANTS = ("primary", "strict", "hno3_only")
STEMS = tuple(FI.design_stem("V5", v, "batched_max4") for v in VARIANTS)
FULL_MANIFEST = paths.MANIFESTS_DIR / "g19_build_folds.json"
INCREMENTAL_MANIFEST = paths.MANIFESTS_DIR / "g19_build_folds_incremental.json"
MAX4_MANIFEST = paths.MANIFESTS_DIR / "g19_build_folds_max4.json"
CAP = 4


def _built() -> bool:
    return INDEX.exists() and all((FOLDS / f"{s}.json").exists() for s in STEMS)


pytestmark = pytest.mark.skipif(not _built(), reason="re-coloured fold files not built (run scripts/g19_build_folds_max4.py)")


@pytest.fixture(scope="module")
def index() -> dict:
    return json.loads(INDEX.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def max4() -> dict[str, list[FI.Fold]]:
    return {s: FI.read_design(s, FOLDS) for s in STEMS}                 # verify=True re-derives every hash


@pytest.fixture(scope="module")
def registered() -> dict[str, list[FI.Fold]]:
    return {v: FI.read_design(FI.design_stem("V5", v, "batched"), FOLDS) for v in VARIANTS}


def _cells(f: FI.Fold) -> list[tuple[str, str]]:
    return [tuple(c) for c in f.meta["cells"]]


def test_1_scheme_meta_and_hashes(max4) -> None:
    for stem, folds in max4.items():
        assert folds, stem
        assert {f.scheme for f in folds} == {"batched_max4"} and {f.design for f in folds} == {"V5"}, stem
        assert all(f.meta["max_cells_per_batch"] == CAP for f in folds), stem
        assert {f.seed for f in folds} == set(FI.DISCOVERY_SEEDS), stem
        assert all(f.batch_id == f.fold_id and f.unit_type == "cell" for f in folds), stem
        assert not any(f.meta["carved_out"] or f.meta["carved_out_cells"] for f in folds), stem


def test_2_largest_batch_at_most_4(max4, index) -> None:
    ph = index["placeholders"]["V5_batches_per_discovery_seed_max4"]
    for v, stem in zip(VARIANTS, STEMS):
        folds = max4[stem]
        body = json.loads((FOLDS / f"{stem}.json").read_text(encoding="utf-8"))
        assert body["max_cells_per_batch"] == CAP and body["scheme"] == "batched_max4"
        for seed in FI.DISCOVERY_SEEDS:
            for h in ("S", "C"):
                sizes = [len(_cells(f)) for f in folds if f.seed == seed and f.half == h]
                assert sizes and max(sizes) <= CAP, (stem, seed, h, max(sizes) if sizes else None)
                assert body["batches_per_seed"][str(seed)]["largest_batch"][h] == max(sizes)
                assert body["batches_per_seed"][str(seed)][h] == len(sizes)
                assert ph[v][str(seed)][h] == len(sizes) and ph[v][str(seed)]["largest_batch"][h] == max(sizes)
            assert ph[v][str(seed)]["max_cells_per_batch"] == CAP


def test_3_coverage_equals_the_registered_batched_cells_per_seed(max4, registered) -> None:
    for v, stem in zip(VARIANTS, STEMS):
        for seed in FI.DISCOVERY_SEEDS:
            got = sorted(c for f in max4[stem] if f.seed == seed for c in _cells(f))
            want = sorted(c for f in registered[v] if f.seed == seed for c in _cells(f))
            assert got == want, (stem, seed)
            assert len(got) == len(set(got)), (stem, seed)                 # every scored cell exactly once
            # the registered batches and the re-coloured ones score the same rows of that seed
            assert (sorted(r for f in max4[stem] if f.seed == seed for r in f.scored_row_ids)
                    == sorted(r for f in registered[v] if f.seed == seed for r in f.scored_row_ids)), (stem, seed)


def test_4_no_shared_state_or_system_within_a_batch(max4) -> None:
    halves = FI.registered_halves("V5_system")
    for stem, folds in max4.items():
        for f in folds:
            cs = _cells(f)
            assert len({c[0] for c in cs}) == len(cs), (stem, f.fold_id)
            assert len({c[1] for c in cs}) == len(cs), (stem, f.fold_id)
            assert {halves[c[1]] for c in cs} == {f.half}, (stem, f.fold_id)
            assert f.scored_row_ids and set(f.scored_row_ids) <= set(f.hidden_row_ids), (stem, f.fold_id)


def test_5_index_and_manifest_carry_the_designs(max4, index) -> None:
    for stem, folds in max4.items():
        entry = index["designs"][stem]
        body = json.loads((FOLDS / f"{stem}.json").read_text(encoding="utf-8"))
        assert entry["design_hash"] == FI.design_hash(folds) == body["summary"]["design_hash"]
        assert entry["scheme"] == "batched_max4" and entry["n_folds"] == len(folds) == len(body["folds"])
        assert entry["guard"]["all_ok"] and entry["guard"]["n_checks"] == len(folds)
        assert entry["max_cells_per_batch"] == CAP and "section 7 item 6" in entry["rule"]
        assert stem in index["leakage_sensitivity_wildcard_copies"]["crossings_by_design"]
    assert "v5_batched_max4" in index["readings"]
    assert set(index["placeholders"]["V5_batches_per_discovery_seed_max4"]) == set(VARIANTS)
    if MAX4_MANIFEST.exists():
        man = json.loads(MAX4_MANIFEST.read_text(encoding="utf-8"))
        assert man["design_hashes"] == {s: FI.design_hash(max4[s]) for s in STEMS}
        outs = {o["path"]: o["sha256"] for o in man["outputs"]}
        for s in STEMS:
            for ext in ("json", "parquet"):
                p = FOLDS / f"{s}.{ext}"
                assert outs[paths.rel(p)] == hashlib.sha256(p.read_bytes()).hexdigest(), p.name


def test_6_registered_design_hashes_and_files_unchanged(index) -> None:
    full = json.loads(FULL_MANIFEST.read_text(encoding="utf-8"))
    inc = json.loads(INCREMENTAL_MANIFEST.read_text(encoding="utf-8"))
    want = {**full["design_hashes"], **inc["design_hashes"]}
    assert len(want) == 23 and not any(s.endswith("max4") for s in want)
    for stem, dh in want.items():
        assert index["designs"][stem]["design_hash"] == dh, stem
        body = json.loads((FOLDS / f"{stem}.json").read_text(encoding="utf-8"))
        assert body["summary"]["design_hash"] == dh, stem
        assert FI.design_hash(FI.read_design(stem, FOLDS)) == dh, stem          # re-derived from the parquet rows
    rewritten = {paths.rel(INDEX), paths.rel(FOLDS / "wildcard_copy_crossings.csv")}
    for o in full["outputs"] + inc["outputs"]:
        if o["path"] in rewritten:
            continue
        p = paths.REPO_ROOT / o["path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == o["sha256"], o["path"]
    assert set(index["nested_certificate_safeguard_sample"]["designs"]).isdisjoint(STEMS)


def test_7_plan_requests_exactly_the_built_stems() -> None:
    from gen19ct.evaluation import discovery as D

    stems: set[str] = set()
    for v5 in ("recolour", "passed_after_recolour", "failed"):
        for v1 in ("pending", "passed", "failed"):
            st = D.PlanState(v5_batched_check=v5, v1_tenfold_check=v1,
                             freezing_candidates=[{"contrast": "x@V5#c", "design": "V5", "arms": list(D.HEAVY_ARMS)}])
            stems |= {j.stem for j in D.enumerate_plan(st) if j.kind == "fit" and j.stem.endswith("max4")}
    assert stems == set(STEMS)
    assert D.PlanState(v5_batched_check="failed").heavy_v5_scheme == "batched_max4"
    assert D.PlanState(v5_batched_check="passed_after_recolour").heavy_v5_scheme == "batched_max4"
    assert D.PlanState(v5_batched_check="recolour").heavy_v5_scheme is None      # only the B6 re-check job runs
    for s in STEMS:
        assert D.guard_mode_source(s) == "V5__primary__batched"                    # takes the registered batched sample

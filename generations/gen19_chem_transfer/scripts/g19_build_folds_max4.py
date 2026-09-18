"""``scripts/g19_build_folds_max4.py`` -- the section 7 compute-plan item 6 re-colouring of the heavy-arm V5 batches
(pre-registration section 7 item 6, sealed; section 3.1): at most 4 cells per batch, same seeds, same vertex-order rule.

Trigger: the registered B6 batched-vs-exact check failed on seed 104729 (``evaluation/discovery/decisions/b6_checks.json``:
|delta| 0.0200 >= 0.01, state ``recolour``), so the discovery runner asks for the re-coloured fold files and stops with
``FileNotFoundError`` until they exist (``scripts/g19_run_discovery.py``, ``fold_tasks``).

Nothing is fitted, nothing is scored, nothing runs on V6.  ``log_D`` is read only inside the leakage guard.

What is built.  For every V5 variant a plan under the heavy scheme ``batched_max4`` can request -- ``primary`` (the
repeated B6 check and the heavy arms' main design) and the addendum 1 item 4 learned-arm refit variants ``strict`` and
``hno3_only`` (``gen19ct.evaluation.discovery.LEARNED_REFIT_VARIANTS``; the freezing-candidate refits use the same
set) -- the file ``V5__<variant>__batched_max4`` with the batches of EVERY discovery seed (mirroring the registered
``V5__<variant>__batched`` files; the runner uses seed 104729).  The set of stems is not hard-coded: it is derived from
``enumerate_plan`` under every post-recolour plan state (``recolour``, ``passed_after_recolour``, ``failed``, with and
without a V5 freezing candidate) and asserted equal to the default (:func:`plan_stems`).  Loose, cell-only and
parent-structure are not run for a learned arm under addendum 1 item 4 and no plan requests their ``batched_max4``
file, so none is built.

How.  The registered builder ``scripts/g19_build_folds.py`` is imported as a module (:func:`load_builder`) and its
``Book``, ``_variant_cells``, ``verify_base_files``, ``supersession_link``, ``merge_index`` / ``merge_crossings`` and
manifest pattern are reused unchanged; the batches come from ``cell_holdout.v5_batched_folds(...,
max_cells_per_batch=4)`` (scheme ``batched_max4``, ``meta.max_cells_per_batch``), i.e. ``greedy_colouring`` with the cap
and the same ``SeedSequence([seed, 5, half])`` vertex order and the same eligibility re-check as the registered rule.
Per seed the same assertions as ``build_v5``: every scored cell exactly once, no two cells of a batch share a metal
state or a system, and in addition the largest batch holds <= 4 cells; ``Book.validate`` and ``guard_v5`` run on every
batch (a guard violation aborts).  The scored cells equal those of the registered batched file.

Merge.  The new designs are merged into ``folds/INDEX.json`` (``merge_index``; placeholder
``V5_batches_per_discovery_seed_max4``; reading ``v5_batched_max4``) and ``wildcard_copy_crossings.csv``
(``merge_crossings``), exactly as the ``--merge-index`` mode of the registered builder does.  Before building, every
full-build design file is verified against ``manifests/g19_build_folds.json`` (digests, fold and design hashes) and the
two already-merged batched designs against ``manifests/g19_build_folds_incremental.json``; after writing, every
pre-existing fold file, fold hash, design hash, INDEX design entry, placeholder, the safeguard sample and every
pre-existing crossing row are asserted unchanged.  The safeguard sample is recomputed as the registered builder does
on a merge (its populations are a fixed list without ``batched_max4`` files, so it must reproduce; a difference aborts
-- this script never changes an existing sample; ``discovery.guard_mode_source`` gives a re-coloured file its
registered batched file's verdict).

Manifest ``manifests/g19_build_folds_max4.json`` (``Run``): inputs, outputs, design hashes, per-seed / per-half fold
counts and the supersession chain of ``INDEX.json`` / ``wildcard_copy_crossings.csv`` (``merge.full_build_link`` and
``merge.previous_incremental`` from ``supersession_link``; a re-run links to this script's own previous manifest).

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_build_folds_max4.py
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.chemistry import support_graph as SG  # noqa: E402
from gen19ct.data import load  # noqa: E402
from gen19ct.folds import cell_holdout as CH  # noqa: E402
from gen19ct.folds import io as FI  # noqa: E402
from gen19ct.folds import registered as FR  # noqa: E402
from gen19ct.folds import safeguard as SF  # noqa: E402
from gen19ct.manifest import Run, write_csv, write_json  # noqa: E402

NAME = "g19_build_folds_max4"
MAX_CELLS_PER_BATCH = 4
SCHEME = CH.batched_scheme(MAX_CELLS_PER_BATCH)                  # "batched_max4"
DESIGN = CH.DESIGN                                              # "V5"
PLACEHOLDER = "V5_batches_per_discovery_seed_max4"
READING_KEY = "v5_batched_max4"
#: the V5 variants whose ``batched_max4`` file a plan under the heavy scheme ``batched_max4`` can request (asserted
#: against ``enumerate_plan`` in :func:`plan_stems`): primary (B6 re-check, heavy main designs) and the addendum 1
#: item 4 learned-arm refit variants (also the freezing-candidate refits)
DEFAULT_VARIANTS: tuple[str, ...] = ("primary", "strict", "hno3_only")
RULE = ("section 7 item 6 re-colouring: greedy colouring capped at 4 cells per batch, same seeds and vertex order rule "
        "as the registered batched files; each cell re-checked eligible with the batch's other cells hidden, failures "
        "to singleton batches")
READING = (RULE + "; built by scripts/g19_build_folds_max4.py after the registered B6 batched-vs-exact check failed on "
           "seed 104729 (decisions/b6_checks.json, state 'recolour'); carved-out cells are not batched; the scored "
           "cells equal those of the registered V5__<variant>__batched file; every discovery seed is built, the runner "
           "uses 104729; a re-coloured file takes its registered batched file's safeguard verdict "
           "(discovery.guard_mode_source)")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_builder():
    """``scripts/g19_build_folds.py`` as a module (``Book``, ``_variant_cells``, the merge machinery, ``READINGS``)."""
    spec = importlib.util.spec_from_file_location("g19_build_folds", HERE.with_name("g19_build_folds.py"))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


B = load_builder()
OUT, INDEX_JSON, CROSSINGS_CSV = B.OUT, B.INDEX_JSON, B.CROSSINGS_CSV
INCREMENTAL_MANIFEST = paths.MANIFESTS_DIR / f"{B.INCREMENTAL_NAME}.json"
OWN_MANIFEST = paths.MANIFESTS_DIR / f"{NAME}.json"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    return ap.parse_args(argv)


# --------------------------------------------------------------------------------------------- #
# which stems the plan can request
# --------------------------------------------------------------------------------------------- #

def plan_stems() -> list[str]:
    """Every ``batched_max4`` stem ``gen19ct.evaluation.discovery.enumerate_plan`` emits under any post-recolour plan
    state (``recolour``, ``passed_after_recolour``, ``failed``; V1 check passed or failed; with and without a V5 and a
    V5-PAIR freezing candidate naming every heavy arm).  Asserted equal to ``V5__<v>__batched_max4`` for
    :data:`DEFAULT_VARIANTS`."""
    from gen19ct.evaluation import discovery as D

    if D.PlanState(v5_batched_check="failed").heavy_v5_scheme != SCHEME:
        raise AssertionError("discovery.PlanState.heavy_v5_scheme after a failed re-coloured check is not batched_max4")
    if set(D.LEARNED_REFIT_VARIANTS) != set(DEFAULT_VARIANTS) - {"primary"}:
        raise AssertionError(f"discovery.LEARNED_REFIT_VARIANTS {D.LEARNED_REFIT_VARIANTS} differ from this script's "
                             f"refit variants {sorted(set(DEFAULT_VARIANTS) - {'primary'})}")
    cands = [[], [{"contrast": "M2 vs B3i@V5#synthetic", "design": "V5", "arms": list(D.HEAVY_ARMS)},
                  {"contrast": "M2 vs B3i@V5-PAIR#synthetic", "design": "V5-PAIR", "arms": list(D.HEAVY_ARMS)}]]
    stems: set[str] = set()
    for v5 in ("recolour", "passed_after_recolour", "failed"):
        for v1 in ("pending", "passed", "failed"):
            for fc in cands:
                st = D.PlanState(v5_batched_check=v5, v1_tenfold_check=v1, freezing_candidates=fc)
                stems |= {j.stem for j in D.enumerate_plan(st, include_h3_specs=True)
                          if j.kind in ("fit", "comparator_intervals", "safeguard") and j.stem.endswith("max4")}
    want = {FI.design_stem(DESIGN, v, SCHEME) for v in DEFAULT_VARIANTS}
    if stems != want:
        raise AssertionError(f"the plan requests {sorted(stems)} but this script builds {sorted(want)}")
    return sorted(stems)


# --------------------------------------------------------------------------------------------- #
# pre-existing state
# --------------------------------------------------------------------------------------------- #

def load_incremental_build() -> dict:
    """``manifests/g19_build_folds_incremental.json``: the merged V5-P / V5-PAIR batched designs and their digests."""
    man = json.loads(INCREMENTAL_MANIFEST.read_text(encoding="utf-8"))
    return {"path": INCREMENTAL_MANIFEST, "sha256": paths.digests(INCREMENTAL_MANIFEST)["sha256"],
            "design_hashes": dict(man["design_hashes"]), "digests": {o["path"]: o["sha256"] for o in man["outputs"]}}


def verify_incremental_files(inc: dict, read_folds: bool = True) -> dict[str, list[str]]:
    """Assert that the incremental build's design files (and the batched pair table) still have their recorded digests
    and, with ``read_folds``, re-derive their fold and design hashes.  Returns the fold hashes per stem."""
    rewritten = {paths.rel(INDEX_JSON), paths.rel(CROSSINGS_CSV)}
    for rel, want in sorted(inc["digests"].items()):
        if rel in rewritten:
            continue
        got = paths.digests(paths.REPO_ROOT / rel)["sha256"]
        if got != want:
            raise AssertionError(f"{rel}: digest {got} differs from the incremental manifest {want}")
    out: dict[str, list[str]] = {}
    for stem, dh in sorted(inc["design_hashes"].items()):
        if read_folds:
            folds = FI.read_design(stem, OUT)
            if FI.design_hash(folds) != dh:
                raise AssertionError(f"{stem}: design hash differs from the incremental manifest")
            out[stem] = [f.fold_hash for f in folds]
        else:
            body = json.loads((OUT / f"{stem}.json").read_text(encoding="utf-8"))
            if body["summary"]["design_hash"] != dh:
                raise AssertionError(f"{stem}: design hash differs from the incremental manifest")
            out[stem] = [r["fold_hash"] for r in body["folds"]]
    return out


def link_chain(base: dict) -> dict:
    """``supersession_link`` of the registered builder, extended by one link: when ``INDEX.json`` /
    ``wildcard_copy_crossings.csv`` on disk are the outputs of a previous run of THIS script, its recorded chain is
    carried forward under ``previous_max4_run``."""
    try:
        return {**B.supersession_link(base), "previous_max4_run": None}
    except AssertionError:
        if not OWN_MANIFEST.exists():
            raise
    prev = json.loads(OWN_MANIFEST.read_text(encoding="utf-8"))
    prev_out = {o["path"]: o["sha256"] for o in prev["outputs"]}
    idx_rel, cr_rel = paths.rel(INDEX_JSON), paths.rel(CROSSINGS_CSV)
    if (prev_out.get(idx_rel) != paths.digests(INDEX_JSON)["sha256"]
            or prev_out.get(cr_rel) != paths.digests(CROSSINGS_CSV)["sha256"]):
        raise AssertionError("INDEX.json / wildcard_copy_crossings.csv are neither the full build's, the incremental "
                             "build's nor this script's previous outputs")
    pm = prev["merge"]
    return {"full_build_link": pm["full_build_link"], "previous_incremental": pm["previous_incremental"],
            "previous_max4_run": {"manifest": paths.rel(OWN_MANIFEST), "manifest_sha256": paths.digests(OWN_MANIFEST)["sha256"],
                                  "design_hashes": prev.get("design_hashes"),
                                  "outputs_now_superseded": {idx_rel: prev_out[idx_rel], cr_rel: prev_out[cr_rel]}}}


# --------------------------------------------------------------------------------------------- #
# the re-coloured batches
# --------------------------------------------------------------------------------------------- #

def registered_scored_cells(stem: str) -> dict[int, list[tuple[str, str]]]:
    """The cells batched per seed in a registered batched file (the scored cells; carved-out cells are not batched)."""
    out: dict[int, list] = {}
    for f in FI.read_design(stem, OUT):
        out.setdefault(int(f.seed), []).extend(tuple(c) for c in f.meta["cells"])
    return {s: sorted(v) for s, v in out.items()}


def build_variant(book, cells_tab: pd.DataFrame, name: str, halves: dict, pmap: dict | None) -> dict:
    """The ``batched_max4`` folds of one variant for every discovery seed; returns the per-seed batch statistics."""
    fr = book.frame
    variant = CH.VARIANTS[name]
    cmap = pmap if variant.parent_structure else None
    cells, scored_cells, _ = B._variant_cells(book, cells_tab, variant)
    missing = sorted({c[1] for c in cells} - set(halves))
    if missing:
        raise AssertionError(f"{name}: systems without a registered half: {len(missing)}")
    reg_stem = FI.design_stem(DESIGN, name, "batched")
    reg = registered_scored_cells(reg_stem)
    if set(reg) != set(FI.DISCOVERY_SEEDS) or any(reg[s] != sorted(scored_cells) for s in reg):
        raise AssertionError(f"{name}: the scored cells differ from the registered {reg_stem} file")
    masks = CH.cell_masks(fr, cells, variant, cmap)
    space = CH.EligibilitySpace(fr, variant.medium)
    bfolds, per_seed = [], {}
    for seed in FI.DISCOVERY_SEEDS:
        sf, st = CH.v5_batched_folds(fr, space, variant, scored_cells, masks, book.v6, halves, seed, component_map=cmap,
                                     max_cells_per_batch=MAX_CELLS_PER_BATCH)
        covered = sorted(tuple(c) for f in sf for c in f.meta["cells"])
        if covered != sorted(scored_cells):
            raise AssertionError(f"{name} seed {seed}: batches do not cover every scored cell exactly once")
        for f in sf:
            cs = [tuple(c) for c in f.meta["cells"]]
            if len({c[0] for c in cs}) != len(cs) or len({c[1] for c in cs}) != len(cs):
                raise AssertionError(f"{f.fold_id}: two cells of one batch share a metal state or a system")
            if len(cs) > MAX_CELLS_PER_BATCH:
                raise AssertionError(f"{f.fold_id}: {len(cs)} cells exceed the cap of {MAX_CELLS_PER_BATCH}")
            if f.scheme != SCHEME or f.meta.get("max_cells_per_batch") != MAX_CELLS_PER_BATCH or f.meta["carved_out"]:
                raise AssertionError(f"{f.fold_id}: scheme / meta.max_cells_per_batch / carved_out not as expected")
        if any(st[h]["largest_batch"] > MAX_CELLS_PER_BATCH for h in ("S", "C") if h in st):
            raise AssertionError(f"{name} seed {seed}: largest batch above the cap")
        per_seed[str(seed)] = {"S": st["S"]["n_batches"], "C": st["C"]["n_batches"], "total": st["n_batches"],
                               "cells_moved_to_singleton": {h: st[h]["n_cells_moved_to_singleton"] for h in ("S", "C")},
                               "colour_classes": {h: st[h]["n_colour_classes"] for h in ("S", "C")},
                               "largest_batch": {h: st[h]["largest_batch"] for h in ("S", "C")},
                               "max_cells_per_batch": MAX_CELLS_PER_BATCH}
        bfolds += sf
        log(f"{name} seed {seed}: S {st['S']['n_batches']} + C {st['C']['n_batches']} batches, largest "
            f"{max(st[h]['largest_batch'] for h in ('S', 'C'))}, moved to singleton "
            f"{sum(st[h]['n_cells_moved_to_singleton'] for h in ('S', 'C'))}")
    book.validate(bfolds)
    breps = [CH.guard_v5(f, fr, component_map=cmap) for f in bfolds]
    if not all(r["ok"] for r in breps):
        raise AssertionError(f"{name}: a re-coloured batch fails the V5 guard")
    book.write(bfolds, breps, extra={"thresholds": variant.thresholds.tag, "medium": variant.medium,
                                     "component_aware": variant.component_aware,
                                     "parent_structure": variant.parent_structure, "batches_per_seed": per_seed,
                                     "max_cells_per_batch": MAX_CELLS_PER_BATCH, "rule": RULE,
                                     "registered_batched_design": reg_stem,
                                     "n_scored_cells": len(scored_cells),
                                     "scored_cells_by_half": {h: sum(1 for c in scored_cells if halves[c[1]] == h)
                                                              for h in ("S", "C")}})
    return per_seed


def fold_counts(stem: str) -> dict:
    """Folds per seed and half of a written design (read back, hash-verified)."""
    out: dict = {}
    for f in FI.read_design(stem, OUT):
        d = out.setdefault(str(f.seed), {"S": 0, "C": 0})
        d[f.half] += 1
    return out


# --------------------------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ns = parse_args(argv)
    t0 = time.perf_counter()
    stems = plan_stems()
    log(f"the plan can request {stems} under the heavy scheme {SCHEME}")
    with Run(NAME, args=vars(ns), seed=None,
             extra={"discovery_seeds": list(FI.DISCOVERY_SEEDS), "scheme": SCHEME, "max_cells_per_batch": MAX_CELLS_PER_BATCH,
                    "variants": list(DEFAULT_VARIANTS), "stems": stems, "rule": RULE,
                    "trigger": "decisions/b6_checks.json: V5 batched-vs-exact |delta| >= 0.01 on seed 104729 (state "
                               "'recolour'); pre-registration section 7 item 6"}) as run:
        run.inputs(paths.ARCHIVE_MASTER, FI.PUB_COMPONENTS_CSV, FI.HALVES_CSV, B.CELLS_CSV, B.STATES_CSV, B.FEAS_JSON,
                   B.SYSTEMS_CSV, B.COMPONENTS_CSV, B.COPY_PAIRS_CSV)
        base = B.load_base_build()
        inc = load_incremental_build()
        old_index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
        old_crossings = B.read_crossings(CROSSINGS_CSV)
        for stem, dh in {**base["design_hashes"], **inc["design_hashes"]}.items():
            if old_index["designs"][stem]["design_hash"] != dh:
                raise AssertionError(f"{stem}: INDEX.json design hash differs from the build manifests")
        prior_fold_hashes = B.verify_base_files(base)
        prior_fold_hashes.update(verify_incremental_files(inc))
        log(f"{len(prior_fold_hashes)} pre-existing designs verified against {paths.rel(base['path'])} and "
            f"{paths.rel(inc['path'])}")
        run.inputs(base["path"], inc["path"], *[OUT / f"{FI.design_stem(DESIGN, v, 'batched')}.{e}"
                                                 for v in DEFAULT_VARIANTS for e in ("json", "parquet")])
        run.extra["merge"] = {"base_manifest": paths.rel(base["path"]), "base_manifest_sha256": base["sha256"],
                              "incremental_manifest": paths.rel(inc["path"]), "incremental_manifest_sha256": inc["sha256"],
                              "index_sha256_before": paths.digests(INDEX_JSON)["sha256"],
                              "crossings_sha256_before": paths.digests(CROSSINGS_CSV)["sha256"],
                              "index_was_full_build_output": paths.digests(INDEX_JSON)["sha256"]
                              == base["digests"][paths.rel(INDEX_JSON)],
                              "index_was_incremental_build_output": paths.digests(INDEX_JSON)["sha256"]
                              == inc["digests"][paths.rel(INDEX_JSON)],
                              **link_chain(base)}
        already = [s for s in stems if (OUT / f"{s}.json").exists()]
        if already:
            log(f"re-run: {already} exist and will be rebuilt (merge_index asserts the design hashes reproduce)")
        # the corpus, exactly as the registered builder's --merge-index mode loads it
        feas = json.loads(B.FEAS_JSON.read_text(encoding="utf-8"))
        model = load.load_model_rows()
        frame = FI.slim_frame(model.assign(**{FI.GROUP_COL: FI.publication_groups(model)}))
        systems = SG.load_system_table()
        support_frame = SG.prepare_support_frame(model, systems=systems)[list(SG.REQUIRED_COLUMNS)]
        copy_pairs = B.read_copy_pairs(B.COPY_PAIRS_CSV)
        copy_audit = copy.deepcopy(old_index["leakage_sensitivity_wildcard_copies"])
        v6 = FR.v6_target_mask(model, FR.v6_system_set(model)).reindex(frame.index)
        want_v6 = feas["V6_prnd_double_cell"]["v6_target_rows__rows5_otherln2"]
        want_v6 = want_v6["n_rows"] if isinstance(want_v6, dict) else want_v6
        if int(v6.sum()) != int(want_v6):
            raise AssertionError(f"V6_TARGET_ROWS {int(v6.sum())} != feasibility.json {want_v6}")
        if set(frame.loc[v6.to_numpy(), FI.ROW_ID]) != FI.registered_v6_ids():
            raise AssertionError("V6_TARGET_ROWS ids differ between the builder and io.registered_v6_ids")
        if int(len(frame)) != int(old_index["population"]["model_rows"]):
            raise AssertionError("MODEL rows differ from the population recorded in INDEX.json")
        book = B.Book(frame, v6, support_frame=support_frame, systems=systems, copy_pairs=copy_pairs)
        log(f"MODEL rows {len(frame)}, V6_TARGET_ROWS {int(v6.sum())}, scorable {int(FI.scorable_mask(frame, v6).sum())}")
        cells_tab = pd.read_csv(B.CELLS_CSV)
        halves = FI.registered_halves("V5_system")
        pmap = FR.parent_component_map() if any(CH.VARIANTS[v].parent_structure for v in DEFAULT_VARIANTS) else None
        batches = {name: build_variant(book, cells_tab, name, halves, pmap) for name in DEFAULT_VARIANTS}
        if set(book.designs) != set(stems):
            raise AssertionError(f"built {sorted(book.designs)} != plan stems {stems}")
        placeholders = {PLACEHOLDER: batches}
        # merge into INDEX.json and the crossings CSV, as the registered builder's --merge-index mode does
        crossings = B.merge_crossings(old_crossings, book.crossings, set(book.designs))
        copy_audit["files"] = {"pairs": paths.rel(B.COPY_PAIRS_CSV), "crossings": paths.rel(CROSSINGS_CSV)}
        copy_audit["crossings_by_design"] = {k: v["wildcard_copy_crossings"] for k, v in sorted(book.designs.items())
                                             if "wildcard_copy_crossings" in v}
        big = {paths.rel(p): p.stat().st_size for p in book.outputs if p.stat().st_size > B.GIT_IGNORE_BYTES}
        index = {
            "schema": FI.SCHEMA, "script": NAME, "discovery_seeds": list(FI.DISCOVERY_SEEDS), "designs_built": [DESIGN],
            "population": {"model_rows": int(len(frame)), "v6_target_rows": int(v6.sum()),
                           "scorable_rows": int(FI.scorable_mask(frame, v6).sum()),
                           "unscored_states": list(FI.UNSCORED_STATES)},
            "roles": list(FI.ROLES), "hash_rules": {
                "fold_hash": "sha256 of sorted 'role,row_id' lines (LF)",
                "design_hash": "sha256 of sorted 'fold_id,fold_hash' lines (LF)",
                "assignment_sha256": "sha256 of the LF CSV 'row_id,fold_id,role,unit,half' sorted by row_id, fold_id"},
            "readings": {**B.READINGS, READING_KEY: READING}, "designs": book.designs, "placeholders": placeholders,
            "files_over_5MB": big, "leakage_sensitivity_wildcard_copies": copy_audit}
        index[SF.KEY] = SF.registered_safeguard_samples(OUT)
        if B._canon(index[SF.KEY]) != B._canon(old_index[SF.KEY]):
            raise AssertionError("the nested-certificate safeguard sample would change; this script never changes an "
                                 "existing sample")
        index = B.merge_index(old_index, index, set(book.designs))
        write_csv(crossings, CROSSINGS_CSV)
        write_json(INDEX_JSON, index)
        book.outputs += [CROSSINGS_CSV, INDEX_JSON]
        # nothing pre-existing changed: files, fold hashes, design hashes, INDEX entries, placeholders, crossing rows
        after = B.verify_base_files(base, skip_stems=set(book.designs), read_folds=False)
        after.update(verify_incremental_files(inc, read_folds=False))
        if set(after) != set(prior_fold_hashes) or any(after[s] != prior_fold_hashes[s] for s in after):
            raise AssertionError("a pre-existing fold hash changed")
        written = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
        for stem, entry in old_index["designs"].items():
            if stem in book.designs:
                raise AssertionError(f"{stem}: a pre-existing design was rebuilt by this script")
            if B._canon(written["designs"][stem]) != B._canon(entry):
                raise AssertionError(f"INDEX.json entry {stem} changed")
            if written["designs"][stem]["design_hash"] != entry["design_hash"]:
                raise AssertionError(f"INDEX.json design hash of {stem} changed")
        for k, v in old_index["placeholders"].items():
            if B._canon(written["placeholders"][k]) != B._canon(v):
                raise AssertionError(f"INDEX.json placeholder {k} changed")
        if B._canon(written[SF.KEY]) != B._canon(old_index[SF.KEY]):
            raise AssertionError("INDEX.json safeguard sample changed")
        for stem in stems:
            body = json.loads((OUT / f"{stem}.json").read_text(encoding="utf-8"))
            if written["designs"][stem]["design_hash"] != body["summary"]["design_hash"]:
                raise AssertionError(f"{stem}: INDEX.json design hash differs from the file")
        kept = old_crossings[~old_crossings["stem"].isin(set(book.designs))].reset_index(drop=True)
        back = B.read_crossings(CROSSINGS_CSV)
        if not back[back["stem"].isin(set(kept["stem"]))].reset_index(drop=True).equals(kept):
            raise AssertionError("wildcard_copy_crossings.csv rows of pre-existing designs changed")
        counts = {s: fold_counts(s) for s in stems}
        run.extra["merge"].update({"n_preexisting_designs_verified_unchanged": len(after),
                                   "preexisting_design_hashes": {s: old_index["designs"][s]["design_hash"]
                                                                 for s in sorted(old_index["designs"])},
                                   "index_sha256_after": paths.digests(INDEX_JSON)["sha256"],
                                   "crossings_sha256_after": paths.digests(CROSSINGS_CSV)["sha256"]})
        run.outputs(*book.outputs)
        run.extra["design_hashes"] = {k: v["design_hash"] for k, v in sorted(book.designs.items())}
        run.extra["fold_counts_per_seed_and_half"] = counts
        run.extra["batches_per_seed"] = batches
        run.extra["files_over_5MB"] = big
        log(f"merge: {len(after)} pre-existing designs unchanged (files, fold hashes, design hashes, INDEX entries, "
            f"placeholders, safeguard sample, crossings)")
        for s in stems:
            log(f"{s}: folds per seed/half {counts[s]}")
    log(f"done in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

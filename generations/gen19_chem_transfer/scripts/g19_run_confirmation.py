"""``scripts/g19_run_confirmation.py`` -- THE single registered confirmation run (pre-registration section 15).

One run, once.  It scores the frozen claims of ``decisions/CONFIRMATION_PLAN.md`` on the **confirmation half** with the
**5 withheld seeds**, runs **V6 once** with S2(a)-(c), and evaluates S1(c) and S1(d).  POST-HOC addendum 5 item 1 makes
it the **core** run: no section 11 V6 actinide deltas and no section 8 power check, both reported ``NOT_RUN`` with their
cost and their consequence.

Gates (all five, in order; :func:`gen19ct.evaluation.confirmation.gate`)
-----------------------------------------------------------------------
(a) ``scripts/g19_seal_prereg.py --check`` and the below-footer digest registered for stage ``confirmation``;
(b) the registry holds that stage and this code is the code it was registered with;
(c) ``decisions/CONFIRMATION_PLAN.md`` present, at most 5 eligible claims;
(d) ``--seed-store PATH`` that verifies against ``manifests/confirmation_seeds_sha256.txt``;
(e) the idempotence lock: ``evaluation/confirmation/decisions/confirmation.json`` absent, or ``--resume``.

The withheld seeds
------------------
They enter **only** through ``--seed-store PATH``, at run time, and only as a ``confirmation.SeedStore`` whose ``repr``
is redacted.  Everything written passes through ``confirmation.scrub`` first, which replaces a seed -- as an int or
inside any string, including a fold id -- by an opaque index ``i1``..``i5``; at the end
``confirmation.scan_for_seed_leak`` re-reads every file the run wrote and **fails the run** if a seed's decimal form
appears anywhere.  ``decisions/CONFIRMATION.md`` reveals the seeds only as the ``--verify-seeds`` verdict.

Modes
-----
``--dry-run``      enumerate the jobs, the folds and the cost from the measured unit costs; fit nothing, write nothing.
``--check-only``   run the five gates and print the verdict; enumerate nothing, fit nothing, write nothing.
``--resume``       complete unfitted folds of the recorded run.  It may do nothing else: the recorded code digest must be
                   the live one and the claim list may not grow, so no claim is ever re-scored under different code.

Fitting
-------
The confirmation half needs its own ``prepare_fold``: discovery's refuses a confirmation-half fold by construction
(*"a confirmation-half fold is never fitted in discovery"*).  :func:`prepare_fold` here is discovery's, with the half
switched and both directions of the V6 carve-out asserted (``confirmation.assert_v6_scope``); everything else -- the
section 2 guards, the tuning of addendum 1, the runners, the record schema -- is discovery's, imported and unchanged.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:                                     # noqa: E402
    sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.chemistry import support_graph as SG  # noqa: E402
from gen19ct.evaluation import confirmation as CF  # noqa: E402
from gen19ct.evaluation import discovery as D  # noqa: E402
from gen19ct.evaluation import metrics as EM  # noqa: E402
from gen19ct.evaluation import pairs as EP  # noqa: E402
from gen19ct.evaluation import registry as REG  # noqa: E402
from gen19ct.evaluation import transfer as ET  # noqa: E402
from gen19ct.folds import io as FI  # noqa: E402
from gen19ct.manifest import Run, write_json  # noqa: E402
from gen19ct.models import interface as I  # noqa: E402

NAME = "g19_run_confirmation"
STAGE = CF.STAGE
#: this script's own prediction-affecting code, beside the discovery runner's (``code_digest``)
CODE_FILES: tuple[Path, ...] = (paths.G19_ROOT / "gen19ct" / "evaluation" / "confirmation.py", Path(__file__).resolve())
CODE_OBJECTS: tuple[Any, ...] = (CF.SeedStore, CF.confirmation_scored_ids, CF.assert_confirmation_rows,
                                 CF.assert_v6_scope, CF.item4, CF.score_claim, CF.seed_mean_system_bootstrap,
                                 CF.s1c_confirmation, CF.s2a_sign_count, CF.s2b_magnitude, CF.coverage_bands,
                                 CF.lock_verdict, CF.enumerate_jobs, CF.scrub,
                                 CF.fit_scored_ids, CF.confirmation_job_folds, CF.confirmation_fittable_folds,
                                 CF.s2a_direction, CF.s2c_coverage)
MAX_WORKERS = 2


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def runner_module():
    return _script("g19_run_discovery")


def seal_module():
    return _script("g19_seal_prereg")


def code_digest() -> dict[str, Any]:
    """The confirmation code digest: the discovery code (the arms are fitted with it) plus this script and its module."""
    rd = runner_module()
    return D.code_digest(rd.CODE_FILES + (Path(rd.__file__).resolve(),) + CODE_FILES, rd.RUNNER_OBJECTS + CODE_OBJECTS)


# ============================================================================================= #
# the withheld-seed fold files
# ============================================================================================= #

#: the fold files this run needs that do not exist for a withheld seed (the batched colourings and V6)
BATCHED_STEMS: tuple[str, ...] = ("V5__primary__batched_max4", "V5__strict__batched_max4",
                                  "V5__hno3_only__batched_max4", "V5PAIR__primary__batched")
SEEDLESS_STEMS: tuple[str, ...] = ("V5__primary__exact", "V2__element__exact")
V6_STEM = "V6__prnd__exact"


#: the V5 variants whose ``batched_max4`` colouring this run needs, in the order of :data:`BATCHED_STEMS`
V5_BATCHED_VARIANTS: dict[str, str] = {"V5__primary__batched_max4": "primary", "V5__strict__batched_max4": "strict",
                                       "V5__hno3_only__batched_max4": "hno3_only"}
MAX_CELLS_PER_BATCH = 4
#: the registered UNBATCHED V5-PAIR cell-pair design (seed-independent) the batched colouring is built from
V5PAIR_UNITS_STEM = "V5PAIR__primary__cell_pair"
V5PAIR_UNIT_PAIRS = "V5PAIR__primary__pairs.parquet"
#: the rule of each stem, quoted where the pre-registration states it
FOLD_RULES: dict[str, str] = {
    "V5": "section 3.1 with the section 7 item 6 re-colouring: cell_holdout.v5_batched_folds(max_cells_per_batch=4), "
          "the same SeedSequence([seed, 5, half]) vertex order, batches formed inside each half, carved-out cells not "
          "batched, every cell re-checked eligible with the batch's other cells hidden and failures moved to singleton "
          "batches -- identical to scripts/g19_build_folds_max4.py except the seed",
    "V5PAIR": "section 3.1 resolution: cell_holdout.v5pair_batched_folds from the registered unbatched cell-pair design "
              f"{V5PAIR_UNITS_STEM} and its pair table, stream SeedSequence([seed, 7, half]), the same re-check and "
              "singleton rule; the unbatched units are seed-independent and are NOT rebuilt",
    "V6": "section 3.4 (POST-HOC addendum 6 item 2): the 13 feasibility systems; per system S the V5 state-level rule "
          "applied to Pr(III) x S and Nd(III) x S TOGETHER via support_graph.hide_cells(component_aware=True), which "
          "removes the Pr(III)/Nd(III) rows AND the Pr(?)/Nd(?) rows in S and in every system sharing a component with "
          "S; the scored rows are the hidden Pr(III) and Nd(III) rows of S itself (each metal weight 1/2, section 4); "
          "the design is seed-independent (one fold per system) and is written once per seed index because the arms' "
          "conformal inner folds are drawn with that seed (section 15 resolution)",
}


@dataclass(frozen=True)
class FoldPlan:
    """What must be built before any fit, per seed index: the stem, the rule and the DERIVED counts and hashes.

    ``n_folds``, ``n_scored_units`` and ``n_scored_rows`` are read off the built design, never assumed: the withheld
    seeds' colourings are new objects and the plan's ``expect 27 +- 1 per seed`` is a diagnostic, not an input
    (:data:`CF.FOLD_COUNTS_CONFIRMATION` is compared and the difference recorded).
    """

    stem: str
    seed_index: int
    n_folds: int
    design_hash: str
    rule: str
    path: str = ""
    assignment_sha256: str = ""
    fold_hashes: tuple[str, ...] = ()
    n_scored_units: int = 0
    n_scored_rows: int = 0
    by_half: Mapping[str, Any] = field(default_factory=dict)
    expected_n_folds: int | None = None
    n_folds_confirmation_half: int = 0
    stats: Mapping[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {"stem": self.stem, "seed_index": self.seed_index, "n_folds": self.n_folds,
                "design_hash": self.design_hash, "rule": self.rule, "path": self.path,
                "assignment_sha256": self.assignment_sha256, "n_fold_hashes": len(self.fold_hashes),
                "fold_hashes": list(self.fold_hashes), "n_scored_units": self.n_scored_units,
                "n_scored_rows": self.n_scored_rows, "by_half": dict(self.by_half),
                "n_folds_confirmation_half": self.n_folds_confirmation_half,
                "expected_n_folds_from_the_plan": self.expected_n_folds,
                "n_folds_matches_the_plan": None if self.expected_n_folds is None else
                (int(self.n_folds_confirmation_half or self.n_folds) == int(self.expected_n_folds)),
                "count_rule": "DERIVED from the built design; the plan's figure is a diagnostic. The plan counts the "
                              "CONFIRMATION-half folds of a design file (folds/INDEX.json by_half.C divided by the 5 "
                              "discovery seeds); a file also holds the selection half's batches, which this run never "
                              "fits", "stats": dict(self.stats)}


@dataclass
class FoldCorpus:
    """What the fold builders read: the registered builder's frame, ``V6_TARGET_ROWS``, the halves and the cell tables.

    It is the corpus ``scripts/g19_build_folds.py`` / ``g19_build_folds_max4.py`` build the discovery seeds' files from,
    loaded the same way, so a withheld seed's colouring differs from a discovery seed's ONLY in the seed.
    """

    frame: pd.DataFrame
    v6: pd.Series
    halves: dict[str, str]
    cells_tab: pd.DataFrame
    condition_key: pd.Series
    book: Any
    builder: Any
    pmap: dict[str, str] | None
    v6_systems: tuple[str, ...]
    universe: set[str]


def load_fold_corpus() -> FoldCorpus:
    """The registered builder's corpus, loaded exactly as ``g19_build_folds_max4.main`` loads it."""
    from gen19ct.data import load
    from gen19ct.data import normalize as N
    from gen19ct.folds import registered as FR

    B = _script("g19_build_folds")
    model = load.load_model_rows()
    frame = FI.slim_frame(model.assign(**{FI.GROUP_COL: FI.publication_groups(model)}))
    systems = SG.load_system_table()
    support_frame = SG.prepare_support_frame(model, systems=systems)[list(SG.REQUIRED_COLUMNS)]
    v6_systems = tuple(sorted(FR.v6_system_set(model)))
    v6 = FR.v6_target_mask(model, set(v6_systems)).reindex(frame.index).fillna(False).astype(bool)
    if set(frame.loc[v6.to_numpy(), FI.ROW_ID].astype(str)) != FI.registered_v6_ids():
        raise AssertionError("V6_TARGET_ROWS differ from the registered set")
    copy_pairs = B.read_copy_pairs(B.COPY_PAIRS_CSV)
    book = B.Book(frame, v6, support_frame=support_frame, systems=systems, copy_pairs=copy_pairs)
    ck = N.condition_key(model.loc[frame.index]).reindex(frame.index)
    return FoldCorpus(frame=frame, v6=v6, halves=FI.registered_halves("V5_system"),
                      cells_tab=pd.read_csv(B.CELLS_CSV), condition_key=ck, book=book, builder=B,
                      pmap=FR.parent_component_map(), v6_systems=v6_systems,
                      universe=set(frame[FI.ROW_ID].astype(str)))


def build_v5_batched(fc: FoldCorpus, variant_name: str, seed: int) -> tuple[list[FI.Fold], dict[str, Any]]:
    """One V5 variant's ``batched_max4`` folds for ONE seed, by the rule of :data:`FOLD_RULES` ``V5``."""
    from gen19ct.folds import cell_holdout as CH

    variant = CH.VARIANTS[variant_name]
    cmap = fc.pmap if variant.parent_structure else None
    cells, scored_cells, _ = fc.builder._variant_cells(fc.book, fc.cells_tab, variant)
    masks = CH.cell_masks(fc.frame, cells, variant, cmap)
    space = CH.EligibilitySpace(fc.frame, variant.medium)
    folds, st = CH.v5_batched_folds(fc.frame, space, variant, scored_cells, masks, fc.v6, fc.halves, seed,
                                    component_map=cmap, max_cells_per_batch=MAX_CELLS_PER_BATCH)
    covered = sorted(tuple(c) for f in folds for c in f.meta["cells"])
    if covered != sorted(scored_cells):
        raise AssertionError(f"{variant_name}: batches do not cover every scored cell exactly once")
    for f in folds:
        cs = [tuple(c) for c in f.meta["cells"]]
        if len({c[0] for c in cs}) != len(cs) or len({c[1] for c in cs}) != len(cs):
            raise AssertionError(f"{f.fold_id}: two cells of one batch share a metal state or a system")
        if len(cs) > MAX_CELLS_PER_BATCH or f.meta.get("max_cells_per_batch") != MAX_CELLS_PER_BATCH:
            raise AssertionError(f"{f.fold_id}: the section 7 item 6 cap of {MAX_CELLS_PER_BATCH} is not honoured")
        if f.meta["carved_out"]:
            raise AssertionError(f"{f.fold_id}: a carved-out cell was batched")
    fc.book.validate(folds)
    reps = [CH.guard_v5(f, fc.frame, component_map=cmap) for f in folds]
    if not all(r["ok"] for r in reps):
        raise AssertionError(f"{variant_name}: a withheld-seed batch fails the V5 guard "
                             f"({CF.INCOMPLETE_GUARD_FAILURE})")
    return folds, {"batches": {h: st[h]["n_batches"] for h in ("S", "C") if h in st},
                   "largest_batch": {h: st[h]["largest_batch"] for h in ("S", "C") if h in st},
                   "cells_moved_to_singleton": {h: st[h]["n_cells_moved_to_singleton"] for h in ("S", "C") if h in st},
                   "n_scored_cells": len(scored_cells), "guard_checks": len(reps), "guard_all_ok": True}


def build_v5pair_batched(fc: FoldCorpus, seed: int) -> tuple[list[FI.Fold], pd.DataFrame, dict[str, Any]]:
    """The batched V5-PAIR folds for ONE seed, from the registered unbatched cell-pair design (:data:`FOLD_RULES`)."""
    from gen19ct.folds import cell_holdout as CH

    units = FI.read_design(V5PAIR_UNITS_STEM, paths.FOLDS_DIR)
    unit_pairs = pd.read_parquet(paths.FOLDS_DIR / V5PAIR_UNIT_PAIRS)
    cells = sorted({tuple(c) for u in units for c in u.meta["cells"]})
    masks = CH.cell_masks(fc.frame, cells, CH.VARIANTS["primary"])
    space = CH.EligibilitySpace(fc.frame)
    folds, ptab, st = CH.v5pair_batched_folds(fc.frame, fc.frame[FI.GROUP_COL], fc.condition_key, space, units,
                                              unit_pairs, masks, fc.v6, fc.halves, seed)
    if sorted(u for f in folds for u in f.meta["unit_fold_ids"]) != sorted(u.fold_id for u in units):
        raise AssertionError("V5PAIR batched: batches do not cover every unit fold exactly once")
    if len(ptab) != len(unit_pairs) or set(ptab["unit_fold_id"]) != set(unit_pairs["fold_id"]):
        raise AssertionError("V5PAIR batched: the batch pair list is not the union of the unit pair lists")
    fc.book.validate(folds)
    by_fold = {k: g for k, g in ptab.groupby("fold_id", sort=False)}
    for f in folds:
        CH.check_pair_isolation(f, by_fold[f.fold_id], fc.universe)
    reps = [CH.guard_v5(f, fc.frame) for f in folds]
    if not all(r["ok"] for r in reps):
        raise AssertionError(f"V5PAIR batched: a withheld-seed batch fails the V5 guard ({CF.INCOMPLETE_GUARD_FAILURE})")
    return folds, ptab, {"batches": {h: st[h]["n_batches"] for h in ("S", "C") if h in st},
                         "n_test_test_row_pairs": int(st["n_test_test_row_pairs"]),
                         "unit_design": V5PAIR_UNITS_STEM, "unit_design_hash": FI.design_hash(units),
                         "n_units": len(units), "guard_checks": len(reps), "guard_all_ok": True}


def build_v6_folds(fc: FoldCorpus) -> tuple[list[FI.Fold], dict[str, Any]]:
    """The V6 design of section 3.4, built ONCE inside this run (POST-HOC addendum 6 item 2).

    One fold per V6 system S: the double cell ``Pr(III) x S`` and ``Nd(III) x S`` hidden TOGETHER under the registered
    component-aware state-level rule, so the fold's hidden rows are the Pr and Nd rows -- known state and ``X(?)`` -- of
    S and of every system sharing a component with S.  The SCORED rows are the known-state Pr(III) and Nd(III) rows of S
    itself: an ``X(?)`` row is hidden and never scored (section 2), and a component-sharing system's rows are hidden
    unscored exactly as under V5.  Guard level V5 (``cell_holdout.guard_v5``, the level section 3.4 inherits by
    applying "the V5 state-level rule of section 3.1").  The fold's half is ``NA``: V6 is the single run over all 13
    systems, not a confirmation-half design (:data:`CF.READINGS` ``v6_half``).
    """
    from gen19ct.folds import cell_holdout as CH

    variant = CH.Variant(CF.V6_VARIANT_NAME, CH.PRIMARY, component_aware=True)
    fr = fc.frame
    ids = fr[FI.ROW_ID].astype(str).to_numpy(dtype=object)
    v6 = fc.v6.to_numpy(dtype=bool)
    doubles = CF.v6_cells(fc.v6_systems)
    if len(doubles) != CF.S2A_N_SYSTEMS:
        raise AssertionError(f"section 3.4 registers {CF.S2A_N_SYSTEMS} V6 systems; the corpus gives {len(doubles)}")
    folds, per_system = [], []
    for cells in doubles:
        system = cells[0][1]
        hidden = CH.hidden_mask(fr, list(cells), variant, None)
        own = CH.cell_rows_mask(fr, cells[0]) | CH.cell_rows_mask(fr, cells[1])
        scored = own & hidden & v6
        if not scored.any():
            raise AssertionError(f"V6 {system}: the double cell hides no scorable Pr/Nd row")
        if (scored & ~v6).any():
            raise AssertionError(f"V6 {system}: a scored row is not a V6_TARGET_ROW")
        row_unit = {r: CH.cell_label(cells[0] if s0 else cells[1]) if (s0 or s1) else f"hidden_unscored x {system}"
                    for r, h, s0, s1 in zip(ids, hidden, CH.cell_rows_mask(fr, cells[0]),
                                            CH.cell_rows_mask(fr, cells[1])) if h}
        f = FI.make_fold(design=CF.V6_DESIGN, variant=CF.V6_VARIANT_NAME, scheme="exact",
                         fold_id=f"v6_prnd__{hashlib.sha1(system.encode()).hexdigest()[:10]}",
                         half="NA", seed=None, hidden=ids[hidden].tolist(),
                         scored=ids[scored].tolist(), unit_type="system", units=[system], batch_id=None,
                         row_unit=row_unit, row_half={r: "NA" for r in row_unit},
                         meta={"cells": [list(c) for c in cells], "system": system, "carved_out_cells": [],
                               "carved_out": False, "medium_scored": "all", "component_aware": True,
                               "parent_structure": False, "metals": list(CF.V6_METALS),
                               "n_scored_pr": int((CH.cell_rows_mask(fr, cells[0]) & scored).sum()),
                               "n_scored_nd": int((CH.cell_rows_mask(fr, cells[1]) & scored).sum()),
                               "rule": FOLD_RULES["V6"]})
        folds.append(f)
        per_system.append({"system": system, "n_hidden": int(hidden.sum()), "n_scored": int(scored.sum()),
                           "n_scored_pr": f.meta["n_scored_pr"], "n_scored_nd": f.meta["n_scored_nd"]})
    for f in folds:
        FI.training_ids(f, fc.universe)
    reps = [CH.guard_v5(f, fr) for f in folds]
    if not all(r["ok"] for r in reps):
        raise AssertionError(f"V6: a fold fails the V5 guard ({CF.INCOMPLETE_GUARD_FAILURE})")
    scored_all = sorted({r for f in folds for r in f.scored_row_ids})
    if not set(scored_all) <= set(fr.loc[v6, FI.ROW_ID].astype(str)):
        raise AssertionError("V6: a scored row is outside V6_TARGET_ROWS")
    return folds, {"n_systems": len(folds), "per_system": per_system, "n_scored_rows": len(scored_all),
                   "guard_checks": len(reps), "guard_all_ok": True, "rule": FOLD_RULES["V6"],
                   "sensitivity_7_systems": CF.V6_SENSITIVITY}


def scrub_folds(folds: Sequence[FI.Fold], store: CF.SeedStore | None) -> list[FI.Fold]:
    """The same folds with every seed-bearing label replaced by the opaque index (``s104729_C_b003`` -> ``si3_C_b003``).

    The fold HASH is over row ids only, so scrubbing cannot change it; the DESIGN hash is over ``fold_id,fold_hash``
    lines, so the recorded design hash is the hash of the scrubbed design -- which is what lands on disk, so it
    reproduces.  ``store=None`` leaves the folds alone (the synthetic path of the tests).
    """
    if store is None:
        return list(folds)
    out = []
    for f in folds:
        out.append(FI.make_fold(design=f.design, variant=f.variant, scheme=f.scheme,
                                fold_id=CF.scrub(f.fold_id, store), half=f.half, seed=None,
                                hidden=f.hidden_row_ids, scored=f.scored_row_ids, unit_type=f.unit_type,
                                units=f.units, batch_id=None if f.batch_id is None else CF.scrub(f.batch_id, store),
                                row_unit=f.row_unit, row_half=f.row_half, meta=CF.scrub(f.meta, store)))
    return out


def seed_folds_dir(out_root: Path, seed_index: int) -> Path:
    """``evaluation/confirmation/folds/i<k>`` -- one directory per OPAQUE seed index (the stem carries no seed)."""
    return CF.folds_dir(out_root) / CF.seed_token(seed_index)


def write_seed_design(folds: Sequence[FI.Fold], out_root: Path, seed_index: int, *, rule: str,
                      stats: Mapping[str, Any], store: CF.SeedStore | None) -> FoldPlan:
    """Write one scrubbed design file under ``folds/i<k>`` and return its DERIVED counts and hashes."""
    sc = scrub_folds(folds, store)
    d = seed_folds_dir(out_root, seed_index)
    pq, js, summary = FI.write_design(sc, d, extra={"rule": rule, "seed_index": int(seed_index),
                                                    "seed": CF.seed_token(seed_index),
                                                    "stats": CF.scrub(dict(stats), store),
                                                    "reading": CF.READINGS["v6_folds_in_run"] if sc[0].design == CF.V6_DESIGN
                                                    else CF.READINGS["seed_entry"]})
    stem = FI.design_stem(sc[0].design, sc[0].variant, sc[0].scheme)
    return FoldPlan(stem=stem, seed_index=int(seed_index), n_folds=len(sc), design_hash=summary["design_hash"],
                    rule=rule, path=str(js), assignment_sha256=summary["assignment_sha256"],
                    fold_hashes=tuple(f.fold_hash for f in sc), n_scored_units=int(summary["n_scored_units"]),
                    n_scored_rows=int(summary["n_distinct_scored_rows"]), by_half=summary["by_half"],
                    n_folds_confirmation_half=int((summary["by_half"].get("C") or {}).get("n_folds", 0)),
                    expected_n_folds=CF.FOLD_COUNTS_CONFIRMATION.get(stem),
                    stats=CF.scrub(dict(stats), store))


def assert_fold_v6_scope(plans: Sequence[FoldPlan], folds_by_key: Mapping[tuple[str, int], Sequence[FI.Fold]],
                         v6_ids: set[str]) -> dict[str, Any]:
    """Section 3.1 / 3.4 in both directions ON THE FOLD FILES, before any fit: no non-V6 design has a V6 scored row,
    and every scored row of the V6 design is one."""
    bad_non_v6, missing = [], []
    for (stem, i), folds in sorted(folds_by_key.items()):
        is_v6 = CF.is_v6(folds[0].design)
        for f in folds:
            inside = set(f.scored_row_ids) & v6_ids
            if is_v6 and set(f.scored_row_ids) - v6_ids:
                missing.append(f"{stem}/i{i}/{f.fold_id}")
            if not is_v6 and inside:
                bad_non_v6.append(f"{stem}/i{i}/{f.fold_id}")
    if bad_non_v6:
        raise AssertionError(f"the V6 carve-out is broken in {len(bad_non_v6)} fold(s) of a non-V6 design "
                             f"(first {bad_non_v6[0]}); section 3.1 carves V6_TARGET_ROWS out of EVERY design")
    if missing:
        raise AssertionError(f"{len(missing)} V6 fold(s) score a row outside V6_TARGET_ROWS (first {missing[0]})")
    return {"n_designs_checked": len(folds_by_key), "carve_out_holds_for_every_non_v6_design": True,
             "v6_rows_scored_only_in_the_v6_job": True,
             "rule": CF.READINGS["v6_scope"]}


def build_confirmation_folds(store: CF.SeedStore | None, out_root: Path, *, stems: Sequence[str] = BATCHED_STEMS,
                             dry_run: bool = False, corpus: FoldCorpus | None = None,
                             with_v6: bool = True) -> list[FoldPlan]:
    """Build the withheld-seed fold files this run needs, per seed INDEX, under the registered rules.

    Per seed: the V5 ``batched_max4`` colourings of ``primary`` (the claims) and of ``strict`` / ``hno3_only`` (R19 item
    6's refits), the batched V5-PAIR colouring (S1(c)), and the V6 design of section 3.4 (POST-HOC addendum 6 item 2).
    The seed-INDEPENDENT exact files (:data:`SEEDLESS_STEMS`, and ``V1__copy__exact``) are the registered ones in
    ``folds/`` and are NOT rebuilt -- an arm is still refitted on them per withheld seed because its conformal inner
    folds are drawn with that seed (section 15 resolution).

    Every count is DERIVED from the built design and compared with the plan's figure; nothing is assumed.  Everything
    written carries ``i1``..``i5``: the directory, the fold ids and every recorded field (``confirmation.scrub``).
    ``dry_run`` returns the plan without touching the corpus or the disk.
    """
    indices = tuple(range(1, CF.N_SEEDS + 1)) if store is None else store.indices()
    if dry_run:
        plans = []
        for i in indices:
            for stem in list(stems) + ([V6_STEM] if with_v6 else []):
                design = stem.split("__")[0]
                plans.append(FoldPlan(stem=stem, seed_index=i, n_folds=CF.FOLD_COUNTS_CONFIRMATION.get(stem, 0),
                                      design_hash="", rule=FOLD_RULES[design if design in FOLD_RULES else "V5"],
                                      expected_n_folds=CF.FOLD_COUNTS_CONFIRMATION.get(stem)))
        return plans
    fc = corpus if corpus is not None else load_fold_corpus()
    v6_ids = set(fc.frame.loc[fc.v6.to_numpy(dtype=bool), FI.ROW_ID].astype(str))
    v6_folds, v6_stats = (build_v6_folds(fc) if with_v6 else ([], {}))
    plans, built = [], {}
    for i in indices:
        seed = None if store is None else store.seed(i)
        for stem in stems:
            if stem in V5_BATCHED_VARIANTS:
                folds, stats = build_v5_batched(fc, V5_BATCHED_VARIANTS[stem], seed if seed is not None else i)
                rule = FOLD_RULES["V5"]
            elif stem == "V5PAIR__primary__batched":
                folds, ptab, stats = build_v5pair_batched(fc, seed if seed is not None else i)
                rule = FOLD_RULES["V5PAIR"]
                pp = seed_folds_dir(out_root, i) / f"{stem}__pairs.parquet"
                pp.parent.mkdir(parents=True, exist_ok=True)
                sp = ptab.copy()
                for col in sp.columns:
                    if sp[col].dtype == object:
                        sp[col] = [CF.scrub(v, store) for v in sp[col]] if store is not None else sp[col]
                sp.to_parquet(pp, index=False)
                stats = {**stats, "pairs_file": str(pp), "n_pairs": int(len(sp))}
            else:
                raise ValueError(f"no registered builder for stem {stem!r}")
            p = write_seed_design(folds, out_root, i, rule=rule, stats=stats, store=store)
            plans.append(p)
            built[(p.stem, i)] = scrub_folds(folds, store)
            log(f"folds {p.stem} i{i}: {p.n_folds} folds ({p.n_folds_confirmation_half} in the confirmation "
                f"half, plan expected {p.expected_n_folds}), {p.n_scored_units} scored units, design "
                f"{p.design_hash[:12]}...")
        if with_v6:
            p = write_seed_design(v6_folds, out_root, i, rule=FOLD_RULES["V6"], stats=v6_stats, store=store)
            plans.append(p)
            built[(p.stem, i)] = scrub_folds(v6_folds, store)
            log(f"folds {p.stem} i{i}: {p.n_folds} systems, {p.n_scored_rows} hidden Pr/Nd rows scored")
    scope = assert_fold_v6_scope(plans, built, v6_ids)
    log(f"the V6 carve-out holds for every non-V6 design and V6 rows are scored only in the V6 job: {scope['n_designs_checked']} "
        "design files checked")
    return plans


#: The ONE stage this tree does not implement, named exactly.  Section 3.4's V6 design is *"the V5 state-level rule of
#: section 3.1"*, so the two hooks ``prepare_fold`` owns are supplied here (:func:`v6_guard`, :func:`v6_inner_check`);
#: the rest of the fitting path lives INSIDE the discovery arm runners, which branch on ``g19_run_discovery.v5_family``
#: -- ``("V5", "V5P", "V5PAIR")``, not V6 -- and cannot be reached from this script.  MEASURED, not assumed:
#: ``v5_family(V6 job)`` is False and ``inner_design_signature(V6 job)`` is None.
V6_ARM_FITTING_GAP = (
    "the arms' own V6 branches inside scripts/g19_run_discovery.py. Section 3.4 defines V6 as the V5 state-level rule "
    "of section 3.1, so every V6 job needs the V5 inner design and the V5 guards; this script supplies the two hooks "
    "prepare_fold owns (v6_guard, v6_inner_check, both this file's own code), but FOUR sites branch on "
    "g19_run_discovery.v5_family(job) -- ('V5', 'V5P', 'V5PAIR'), which V6 is not -- and they are inside the runner "
    "classes, unreachable from here: (1) validation_splits and (2) _calibration_splits send a V6 job down the "
    "registered V1 / V2 tuning path instead of the addendum-1 simultaneous V5 design (boosted.inner_design_for DOES "
    "return the V5 design for V6, so this may silently produce numbers rather than raise -- which is worse); "
    "(3) inner_design_record then labels that draw 'the registered V1 / V2 inner design, unchanged by addendum 1', "
    "which is false for V6, and inner_design_signature(job) is None for V6, so the addendum-1 drift check of "
    "simultaneous_inner_design never runs and the resume digest carries no inner-design signature; "
    "(4) ComparatorIntervalsRunner.point picks interface.InnerMetalCalibration -- the V2 METAL-holdout inner design -- "
    "for the B3x / B3i yardsticks on V6, because its branch is 'V5 / V1 / else'. The fix is four one-line V6 branches "
    "in g19_run_discovery.py (v5_family, the two guard dispatchers if they are to own V6 rather than this script, the "
    "comparator splitter) plus a V6 entry in inner_design_signature -- an ORCHESTRATOR decision, because it changes the "
    "discovery stage's code digest and that stage must then be re-registered. V6 runs ONCE (section 3.4), so no fold "
    "of it can be fitted as a rehearsal: this path cannot be validated before the single run, and shipping it unnamed "
    "would risk spending the run on an INCOMPLETE_GUARD_FAILURE at the first V6 fold. Everything else -- the claims "
    "C1-C4, S1(c), S1(d) / S1(e) and the whole S2 assembly that reads the V6 records -- is implemented and tested.")

#: the stages of the single run, and which of them this tree implements.  The runner refuses at the first stage whose
#: code is not written, NAMING it -- it never produces a number from a stage that does not exist.  Everything above the
#: first ``False`` is implemented, tested (``tests/test_confirmation.py``) and exercised by ``--dry-run`` /
#: ``--check-only``; everything below it is the remaining work of the single run and is listed in the refusal.
RUN_STAGES: tuple[tuple[str, bool, str], ...] = (
    ("gates", True, "the five gates of confirmation.gate: seal + registered below-footer digest, the registry's "
                    "'confirmation' stage and this code, the plan with <= 5 claims, a verified --seed-store, the "
                    "idempotence lock"),
    ("plan", True, "decisions/CONFIRMATION_PLAN.md parsed into frozen claims (arm, comparator, design, margin)"),
    ("inventory_and_cost", True, "the job and fold inventory of the core run and its cost from the measured unit costs "
                                 "(--dry-run)"),
    ("seed_handling", True, "the store loaded and verified, seeds held in a redacted SeedStore, every written field "
                            "scrubbed to an opaque index, and the leak scan"),
    ("fold_isolation_and_record_writing", True, "the confirmation-half prepare_fold (discovery's assertions, the half "
                                                "switched, the V6 carve-out asserted in both directions) and run_fold"),
    ("claim_scoring", True, "R19 at confirmation over the 5 seeds: the seed-mean cluster bootstrap for items 1, 2, 3 "
                            "and 5, item 4 at 5 of 5, TOST, BH"),
    ("withheld_seed_fold_files", True, "building the withheld-seed colourings on the corpus: "
                                        "cell_holdout.v5_batched_folds (max_cells_per_batch = 4) and "
                                        "v5pair_batched_folds per seed, and the V6 design of section 3.4 "
                                        "(support_graph.hide_cells(component_aware=True) on Pr(III) x S and Nd(III) x S "
                                        "together, 13 systems)"),
    ("fit_loop", True, "the corpus load and the job -> fold dispatch at <= 2 workers with the wall-clock ledger"),
    ("s1c_s1d_s1e_assembly", True, "assembling S1(c)'s per-seed per-system direction deltas from the V5-PAIR records "
                                    "(the statistic itself is implemented and tested: "
                                    "confirmation.seed_mean_system_bootstrap / s1c_confirmation), and S1(d) / S1(e) "
                                    "from the prediction frames"),
    ("s2_assembly", True, "assembling S2(a)-(c) from the V6 records (the statistics are implemented and tested: "
                           "confirmation.s2a_sign_count / s2b_magnitude / coverage_bands)"),
    ("writers", True, "wiring decisions/confirmation.json, tables/confirmation_*.csv|md and decisions/CONFIRMATION.md "
                       "to a completed run (the writers themselves are implemented and tested: "
                       "confirmation.write_decisions / confirmation_tables / confirmation_report)"),
    ("v6_arm_fitting", False, V6_ARM_FITTING_GAP),
)


def refusal(stage: str) -> str:
    """The refusal text of an unimplemented stage: what is missing, what is not, and what must not happen meanwhile."""
    todo = [(n, why) for n, ok, why in RUN_STAGES if not ok]
    done = [n for n, ok, _ in RUN_STAGES if ok]
    return (f"refused: the confirmation run stage {stage!r} is NOT IMPLEMENTED in this tree, so the run stops here "
            "rather than producing a number from code that does not exist. Implemented and tested: "
            + ", ".join(done) + ". Still to write: "
            + "; ".join(f"{n} -- {why}" for n, why in todo)
            + ". Nothing was fitted, no confirmation-half row was scored, no withheld seed was written anywhere, and "
              "the idempotence lock is untouched, so the single registered run (section 15) has NOT been spent.")


# ============================================================================================= #
# one confirmation fold
# ============================================================================================= #

def v6_guard(job: D.JobSpec, fold: FI.Fold, corpus: Any, universe: pd.Index) -> list[dict]:
    """The outer guard of a V6 fold: ``cell_holdout.guard_v5``, the level section 3.4 inherits.

    Discovery's ``outer_guard`` raises ``ValueError('no outer guard for V6')`` -- it was written for the designs
    discovery runs, and V6 runs only here.  Section 3.4 defines V6 as *"the V5 state-level rule of section 3.1 applied
    to the cells Pr(III) x S and Nd(III) x S together (support_graph.hide_cells(component_aware=True))"*, so the
    registered guard is the component-aware V5 one, with no parent-structure map -- exactly what the in-run fold builder
    (:func:`build_v6_folds`) already checks every fold with before it is written.
    """
    from gen19ct.folds import cell_holdout as CH

    reps = [CH.guard_v5(fold, corpus.slim, component_map=None, universe_index=universe)]
    if not all(r["ok"] for r in reps):
        raise AssertionError(f"{job.key}/{fold.fold_id}: outer guard failed")
    return [{"level": r.get("level"), "ok": bool(r["ok"]), "publication_basis": r.get("publication_basis"),
             "n_train": r.get("n_train"), "n_test": r.get("n_test")} for r in reps]


def v6_inner_check(job: D.JobSpec, corpus: Any) -> Callable[[pd.Index, pd.Index], dict]:
    """The inner isolation level of a V6 job: ``V5``, component-aware (the rule section 3.4 inherits).

    Discovery's ``inner_isolation_check`` raises for V6 for the same reason as :func:`v6_guard`.
    """
    from gen19ct.data import leakage as L

    def check(tr: pd.Index, te: pd.Index) -> dict:
        return L.fold_isolation_check(tr, te, corpus.slim, level="V5", component_aware=True, component_map=None,
                                      near_dup_sig=FI.NEAR_DUP_SIG, near_dup_value_tol=FI.NEAR_DUP_VALUE_TOL,
                                      raise_on_violation=False)
    return check


def prepare_fold(job: D.JobSpec, fold: FI.Fold, ordinal: int, corpus: Any, out_root: Path, state: D.PlanState, *,
                 code: str = "", guard_fn: Callable | None = None, inner_check: Callable | None = None,
                 what: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Discovery's ``prepare_fold`` with the CONFIRMATION half (:data:`confirmation.READINGS` ``half``).

    Discovery's own refuses a confirmation-half fold, so this is the mirror: the scored rows are the rows this run
    scores in the fold (``confirmation.fit_scored_ids`` -- the confirmation-half rows of a design with halves, every
    scored row of a V6 fold, which carries half ``NA`` and is selected by the V6 scope instead: ``READINGS['v6_half']``),
    each one is asserted to be in that half where the design HAS halves, and the V6 carve-out is asserted in BOTH
    directions -- a V6 job scores only ``V6_TARGET_ROWS`` and no other job scores any
    (``confirmation.assert_v6_scope``).  Everything else is discovery's code, called unchanged, except the two hooks
    discovery has no V6 branch for (:func:`v6_guard`, :func:`v6_inner_check`).

    ``what`` is the label every assertion here is raised with; the caller passes a SCRUBBED one, because ``job.key``
    carries the run seed and a traceback must never print a withheld seed (POST-HOC addendum 6 item 4).
    """
    rd = runner_module()
    what = what or f"{job.key}/{fold.fold_id}"
    is_v6 = CF.is_v6(job.design)
    if fold.half == D.SELECTION:
        raise AssertionError(f"{what}: a selection-half fold is never fitted in the confirmation run")
    if is_v6 != CF.is_v6(fold.design):
        raise AssertionError(f"{what}: a {job.design} job was given a {fold.design} fold")
    sel = list(CF.fit_scored_ids(fold))
    sc_ids = [r for r in sel if not bool(corpus.coext_by_id.get(r, False))]
    if not sc_ids:
        raise AssertionError(f"{what}: no scored row this run may score (the fold's "
                             f"{'V6' if is_v6 else 'confirmation-half'} scored rows are all acidic co-extractant rows, "
                             "or it has none)")
    if not is_v6:
        CF.assert_confirmation_rows(sc_ids, corpus.half_by_id[job.design], what)
    sc_labels = corpus.labels_of(sc_ids)
    CF.assert_v6_scope(sc_labels, pd.Series(corpus.v6, index=corpus.frame.index) if not isinstance(corpus.v6, pd.Series)
                       else corpus.v6, job.design, what)
    # from here the fold context is discovery's, built on the same objects: the guards, the training mask, the inner
    # design and the support file are its code, and the only difference is the half the scored rows come from
    fc, info = _prepare_with_half(rd, job, fold, ordinal, corpus, out_root, state, code, sc_ids, sc_labels,
                                  guard_fn or (v6_guard if is_v6 else None),
                                  inner_check or (v6_inner_check if is_v6 else None), what)
    info = dict(info)
    info.update(half="NA" if is_v6 else D.CONFIRMATION, n_scored_confirmation=len(sc_ids),
                reading=CF.READINGS["v6_half"] if is_v6 else CF.READINGS["half"])
    return fc, info


def _prepare_with_half(rd: Any, job: D.JobSpec, fold: FI.Fold, ordinal: int, corpus: Any, out_root: Path,
                       state: D.PlanState, code: str, sc_ids: Sequence[str], sc_labels: pd.Index,
                       guard_fn: Callable | None, inner_check: Callable | None,
                       what: str | None = None) -> tuple[Any, dict[str, Any]]:
    """The body of discovery's ``prepare_fold`` on confirmation-half scored rows (its assertions kept, in its order)."""
    what = what or f"{job.key}/{fold.fold_id}"
    st = corpus.frame.loc[sc_labels, SG.METAL_COL]
    if st.isna().any() or (st == I.SR_III).any():
        raise AssertionError(f"{what}: an X(?) or Sr(III) row would be scored")
    hidden = corpus.labels_of(fold.hidden_row_ids)
    if not sc_labels.isin(hidden).all():
        raise AssertionError(f"{what}: a scored row is not hidden")
    sr_extra = corpus.sr_index.difference(hidden) if job.drop_sr else pd.Index([], dtype=hidden.dtype)
    universe = corpus.frame.index.difference(sr_extra)
    guards = (guard_fn or rd.outer_guard)(job, fold, corpus, universe)
    t = corpus.table
    mask = np.ones(t.n, dtype=bool)
    mask[t.positions(hidden)] = False
    if len(sr_extra):
        mask[t.positions(sr_extra)] = False
    if mask[t.positions(sc_labels)].any():
        raise AssertionError(f"{what}: a scored row is a training row")
    gmode = rd.guard_mode_for(job, state)
    cache = corpus.guard_cache.setdefault((job.design, job.variant, job.drop_sr), {})
    ctx = I.FitContext(systems=corpus.systems, components=corpus.comps, table=t, hidden_index=hidden.union(sr_extra),
                       v6_mask=corpus.v6, exclude_from_scoring=corpus.coext,
                       isolation_check=(inner_check or rd.inner_isolation_check)(job, corpus), guard_cache=cache,
                       seed=int(job.seed))
    fc = rd.FoldContext(job=job, fold=fold, ordinal=ordinal, corpus=corpus, mask=mask, hidden=hidden,
                        sc_ids=list(sc_ids), sc_labels=sc_labels, positions=t.positions(sc_labels), ctx=ctx,
                        out_root=out_root, guard_mode=gmode, batching_label=rd.batching_label(job, state), code=code,
                        state=state, design_hash=corpus.design_hash(job.stem))
    info = {"n_hidden": len(hidden), "n_train": int(mask.sum()), "n_scored_confirmation": len(sc_ids),
            "n_sr_iii_dropped": int(len(sr_extra)), "outer_guard": guards, "inner_guard_mode": gmode}
    return fc, info


def run_fold(job: D.JobSpec, fold: FI.Fold, ordinal: int, corpus: Any, out_root: Path, *, store: CF.SeedStore,
             seed_index: int, runners: Mapping[str, Any], code: str, state: D.PlanState,
             steps: Sequence[str] = ("point", "intervals"), prereg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit (or skip) one confirmation fold and write its prediction parquet and record, SCRUBBED of the seed.

    The record is discovery's schema with ``registry_stage`` = ``confirmation``, the opaque ``seed_index``, and the
    withheld seed nowhere: :func:`confirmation.scrub` runs over the whole record and over the prediction frame's labels
    before either is written.
    """
    rd = runner_module()
    runner = rd.runner_for(job, runners)
    what = f"{CF.scrub(job.key, store)}/{CF.scrub(fold.fold_id, store)}"
    is_v6 = CF.is_v6(job.design)
    pq, js = CF.fold_paths(out_root, job.arm, job.design_dir, seed_index, CF.scrub(fold.fold_id, store))
    if js.exists():
        return {"job": job.key, "fold_id": fold.fold_id, "status": "skipped_done"}
    REG.refuse_unless_writable(STAGE, code)
    fc, info = prepare_fold(job, fold, ordinal, corpus, out_root, state, code=code, what=what)
    outputs = runner.point(fc)
    out = {}
    for arm, o in outputs.items():
        frame = D.prediction_frame(o.pred, job=job, arm=arm, fold=fold, ordinal=ordinal, row_ids=fc.sc_ids,
                                   selected_config=o.selected_config, model_seed=o.model_seed,
                                   intervals_status="split_conformal_inner" if o.intervals else "pending",
                                   batching_label=fc.batching_label, fit_seconds=o.seconds)
        if is_v6:
            # V6 is not a confirmation-HALF design (READINGS['v6_half']): the written rows are checked against the V6
            # scope instead -- every one a V6_TARGET_ROW -- which is the guard section 3.4 gives this design
            CF.assert_v6_scope(corpus.labels_of(list(frame["row_id"])),
                               corpus.v6 if isinstance(corpus.v6, pd.Series)
                               else pd.Series(corpus.v6, index=corpus.frame.index), job.design, f"{what} written")
        else:
            CF.assert_confirmation_rows(frame["row_id"], corpus.half_by_id[job.design], f"{what} written")
        rec = {"schema": CF.SCHEMA, "registry_stage": STAGE, "job": job.record(), "arm": arm, "fold_id": fold.fold_id,
               "fold_hash": fold.fold_hash, "stem": job.stem, "half": "NA" if is_v6 else D.CONFIRMATION,
               "seed_index": int(seed_index),
               "seed_commitment_sha256": store.digest, "code_digest": code,
               "prereg_sha256": D.REGISTERED_PREREG_SHA256,
               "prereg_addenda_sha256": (prereg or {}).get("addenda_sha256"),
               "prereg_n_addenda": (prereg or {}).get("n_addenda"), "addendum_implemented": REG.addenda_count(STAGE),
               "selected_config": o.selected_config, "model_seed": o.model_seed, "arm_record": o.record,
               "steps": {"prediction": {"seconds": round(o.seconds, 3),
                                        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}},
               **info}
        pq2, js2 = CF.fold_paths(out_root, arm, job.design_dir, seed_index, CF.scrub(fold.fold_id, store))
        pq2.parent.mkdir(parents=True, exist_ok=True)
        # the frame's numeric columns hold no seed; its fold_id and any seed-labelled column are scrubbed
        scrubbed = frame.copy()
        for col in scrubbed.columns:
            if scrubbed[col].dtype == object:
                scrubbed[col] = [CF.scrub(v, store) for v in scrubbed[col]]
        scrubbed.to_parquet(pq2, index=False)
        write_json(js2, CF.scrub(rec, store))
        out[arm] = str(js2)
    return {"job": job.key, "fold_id": fold.fold_id, "status": "fitted", "records": out}


# ============================================================================================= #
# scoring one claim over the 5 withheld seeds
# ============================================================================================= #

#: the seed combination of a claim's items 1, 2, 3 and 5 -- the one reading this run takes, and the fact that it is a
#: reading.  Item 4 is per seed and registered (5 of 5); the POOLED point estimate is not registered anywhere, so the
#: closest registered analogue governs: section 9 S1(c)'s confirmation rule, "the MEAN over the 5 withheld seeds of the
#: per-seed macro statistic, its interval a cluster bootstrap of that seed mean with the same resampled clusters applied
#: to every seed".  Because the design's macro statistic is a mean over units, that is IDENTICAL to building the paired
#: units from each unit's seed-mean MAE and bootstrapping them once -- which is what this code does, so no separate
#: resampling loop can disagree with it.
SEED_POOLING = (
    "REGISTERED by POST-HOC addendum 6 item 1 (the addendum the earlier wording REQUESTED): 'for every frozen claim, "
    "the confirmation statistic is the MEAN over the 5 withheld seeds of the per-seed macro value; delta is the seed "
    "mean of the per-seed paired delta; item 1 compares that seed mean with the design's margin; items 2 and 3 use the "
    "percentile and BCa intervals and the two-sided p of a cluster bootstrap (10,000 resamples, seed 19) of the seed "
    "mean, with the same resampled clusters applied to every seed, under every registered cluster unit; item 5 removes "
    "one cluster from all five seeds at once; item 4 is unchanged (delta > 0 in 5 of 5 seeds); item 6 is the reduced set "
    "of addendum 1 item 4 with its refits on seed 104729, as in discovery. Per-seed values are printed beside the seed "
    "mean so the spread is visible.' The code builds the paired units from each unit's seed-MEAN MAE and bootstraps "
    "them once: because the macro statistic is a mean over units, mean-over-seeds-of-macro equals "
    "macro-of-mean-over-seeds exactly, and one resampling of the seed-mean units IS 'the same resampled clusters "
    "applied to every seed', so no separate loop can disagree with it. Item 5's leave-one-cluster-out on those units "
    "removes the cluster from all five seeds at once, as the addendum requires. Every per-seed delta is reported beside "
    "the seed mean (seed_deltas_by_index)")


def seed_mean_paired_units(per_seed: Mapping[int, D.PairedUnits], *, design: str, candidate: str, comparator: str
                           ) -> D.PairedUnits:
    """The paired units of the seed MEAN: each unit's MAE averaged over the withheld seeds (:data:`SEED_POOLING`)."""
    idx = sorted(per_seed)
    if not idx:
        raise ValueError("no seed scored")
    first = per_seed[idx[0]]
    units = first.cand_mae.index
    for i in idx:
        if not per_seed[i].cand_mae.index.equals(units):
            raise AssertionError(f"seed index {i}: different averaging units from seed index {idx[0]}; a claim's units "
                                 "are the confirmation half's scored cells and do not depend on the colouring")
    cand = sum(per_seed[i].cand_mae for i in idx) / len(idx)
    comp = sum(per_seed[i].comp_mae for i in idx) / len(idx)
    return D.PairedUnits(design=design, candidate=candidate, comparator=comparator, cand_mae=cand, comp_mae=comp,
                         clusters=first.clusters, n_rows=first.n_rows)


def score_claim_over_seeds(claim: CF.Claim, per_seed: Mapping[int, D.PairedUnits], *,
                           sensitivity_deltas: Mapping[str, float | str], deterministic: bool = False,
                           reduced_sensitivities: Sequence[str] | None = None,
                           sensitivities_not_run_extra: Mapping[str, str] | None = None) -> dict[str, Any]:
    """R19 at confirmation for one frozen claim: the seed-mean bootstrap for items 1, 2, 3 and 5, the per-seed Deltas
    for item 4 (5 of 5), TOST under the primary cluster, item 6 on addendum 1 item 4's reduced set."""
    pooled = seed_mean_paired_units(per_seed, design=claim.design, candidate=claim.candidate,
                                    comparator=claim.comparator)
    boots = D.bootstraps(pooled, name=claim.key)
    out = CF.score_claim(claim, point=pooled.delta, bootstraps=boots,
                         seed_deltas={int(i): float(per_seed[i].delta) for i in sorted(per_seed)},
                         sensitivity_deltas=sensitivity_deltas, deterministic=deterministic,
                         reduced_sensitivities=reduced_sensitivities,
                         sensitivities_not_run_extra=sensitivities_not_run_extra)
    out["seed_pooling_reading"] = SEED_POOLING
    out["n_units"] = int(len(pooled.cand_mae))
    return out


# ============================================================================================= #
# the fit loop
# ============================================================================================= #

def job_spec_of(job: CF.ConfJob, seed: int, *, transform: str = "", fold_seed: int | None = None) -> D.JobSpec:
    """A ``CF.ConfJob`` as a ``discovery.JobSpec``: the arm, the design file's variant / scheme, the withheld seed.

    ``B6r0`` is read from the ``B6`` fit (rank 0) and ``B3i`` / ``B0`` / ``B3x`` are the closed-form comparators of
    section 5; every one of them is fitted by discovery's own runner (``g19_run_discovery.runner_for``), unchanged.

    The ``kind`` is discovery's: a closed-form comparator of :data:`discovery.COMPARATOR_INTERVAL_ARMS` is a
    ``comparator_intervals`` job, because that is the key ``runner_for`` looks its runner up under
    (``comparator:B3i``); a ``fit`` job for B3i would raise ``KeyError('no runner for B3i')``.  ``fold_seed`` filters a
    MULTI-SEED fold file to one seed's folds -- required for the item 6 refit pass, which reads the registered
    ``V5__strict__batched_max4`` / ``V5__hno3_only__batched_max4`` files holding all five discovery seeds' colourings.
    """
    design, variant, scheme = job.stem.split("__")
    kind = "comparator_intervals" if job.arm in D.COMPARATOR_INTERVAL_ARMS else "fit"
    return D.JobSpec(kind=kind, arm=job.arm, design=design, variant=variant, scheme=scheme, seed=int(seed),
                     fold_seed=None if fold_seed is None else int(fold_seed),
                     stage=CF.STAGE, group=job.purpose, purpose=job.purpose,
                     condition=(transform or job.transform) or None)


def fold_seeded_spec(job: CF.ConfJob, seed: int, corpus: Any) -> D.JobSpec:
    """The job's ``JobSpec`` with ``fold_seed`` set iff its design FILE is multi-seed -- derived, never assumed.

    A withheld-seed colouring written by this run carries ``seed=None`` on every fold (the seed is the opaque index, so
    the file cannot name one), and the registered exact files are seed-free too: neither needs a ``fold_seed``.  The
    registered ``batched_max4`` files hold all five discovery seeds' colourings, and the R19 item 6 refit pass (opaque
    index 0, the PUBLIC seed 104729) reads them: without ``fold_seed`` ``confirmation_job_folds`` refuses the file, which
    is what it must do -- so the seed is read off the file here rather than guessed.
    """
    seeds = {f.seed for f in corpus.folds(job.stem)}
    return job_spec_of(job, seed, fold_seed=int(seed) if seeds - {None} else None)


def confirmation_corpus() -> Any:
    """Discovery's corpus, unchanged: the same frame, ``RowTable``, ``V6_TARGET_ROWS``, acidic co-extractant rows,
    registered halves and condition vectors the discovery run fitted on (``g19_run_discovery.load_corpus``)."""
    rd = runner_module()
    return rd.load_corpus(rd.coextractant_ids())


def v6_by_id(corpus: Any) -> pd.Series:
    """``V6_TARGET_ROWS`` keyed by ``row_id`` (the scoring frames are indexed by row id, the corpus mask by label)."""
    return pd.Series(corpus.v6.to_numpy(dtype=bool), index=corpus.frame[FI.ROW_ID].astype(str).to_numpy())


def attach_seed_folds(corpus: Any, out_root: Path, seed_index: int,
                      stems: Sequence[str] = ()) -> dict[str, list[FI.Fold]]:
    """Point the corpus's fold cache at ONE seed index's withheld-seed designs, and clear the guard cache.

    A confirmation stem (:data:`BATCHED_STEMS`, :data:`V6_STEM`) is read from ``evaluation/confirmation/folds/i<k>``;
    every other stem stays the registered, seed-independent file in ``folds/``, untouched.  The guard cache is cleared
    because it is keyed by design and variant rather than by fold, and the colouring has just changed.
    """
    out: dict[str, list[FI.Fold]] = {}
    d = seed_folds_dir(out_root, seed_index)
    for stem in (stems or tuple(dict.fromkeys(list(BATCHED_STEMS) + [V6_STEM]))):
        js = d / f"{stem}.json"
        if js.exists():
            folds = FI.read_design(js)
            corpus._folds[stem] = folds
            corpus._folds.pop(("design_hash", stem), None)
            out[stem] = folds
    corpus.guard_cache.clear()
    return out


@dataclass
class FitLedger:
    """Wall clock, fold counts and per-job completeness of the run.  A guard failure is recorded as
    :data:`CF.INCOMPLETE_GUARD_FAILURE`, never in the vocabulary of a compute cap (POST-HOC addendum 6 item 3(b))."""

    started: float = 0.0
    seconds: float = 0.0
    fitted: int = 0
    skipped: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    by_job: dict[str, dict[str, Any]] = field(default_factory=dict)

    def note(self, key: str, status: str, seconds: float = 0.0) -> None:
        b = self.by_job.setdefault(key, {"fitted": 0, "skipped": 0, "errors": 0, "seconds": 0.0})
        b[{"fitted": "fitted", "skipped_done": "skipped"}.get(status, "errors")] += 1
        b["seconds"] += float(seconds)
        self.seconds += float(seconds)
        if status == "fitted":
            self.fitted += 1
        elif status == "skipped_done":
            self.skipped += 1

    def record(self) -> dict[str, Any]:
        return {"n_folds_fitted": self.fitted, "n_folds_skipped_already_done": self.skipped,
                "wall_clock_hours": self.seconds / 3600.0, "n_errors": len(self.errors), "errors": self.errors[:20],
                "by_job": dict(sorted(self.by_job.items())), "workers_max": MAX_WORKERS,
                "guard_failure_vocabulary": CF.READINGS["completeness_unit"]}


def fit_loop(jobs: Sequence[CF.ConfJob], store: CF.SeedStore, out_root: Path, *, corpus: Any, state: D.PlanState,
             runners: Mapping[str, Any], code: str, prereg: Mapping[str, Any], ledger: FitLedger | None = None,
             limit: int | None = None) -> FitLedger:
    """Fit every job of the core run, seed index by seed index, in the inventory's registered order.

    Grouping by seed index attaches each seed's fold files once.  Resumability is by digest: :func:`run_fold` returns
    ``skipped_done`` when the record exists, so ``--resume`` completes unfitted folds and re-fits nothing.  A seed index
    of 0 is the R19 item 6 refit pass on the PUBLIC discovery seed 104729 (addendum 1 item 4), which reads the registered
    fold files.  ``limit`` stops after that many fitted folds (the test path).
    """
    led = ledger or FitLedger(started=time.perf_counter())
    by_index: dict[int, list[CF.ConfJob]] = {}
    for j in jobs:
        by_index.setdefault(int(j.seed_index), []).append(j)
    for i in sorted(by_index):
        seed = store.seed(i) if i in store.indices() else CF.ITEM6_REFIT_SEED
        attached = attach_seed_folds(corpus, out_root, i) if i in store.indices() else {}
        log(f"seed index i{i}: {len(by_index[i])} job(s), {len(attached)} withheld-seed design(s) attached")
        for job in by_index[i]:
            spec = fold_seeded_spec(job, seed, corpus)
            # discovery's ``fittable`` is its own: it drops every confirmation-half fold and keeps the folds whose
            # SELECTION-half rows are scorable, so on this half it returns either nothing or exactly what this run may
            # never fit.  The confirmation selector is confirmation.confirmation_fittable_folds
            folds = CF.confirmation_fittable_folds(spec, corpus.folds(spec.stem), corpus.coext_ids)
            if not folds:
                led.errors.append({"job": job.key, "status": CF.INCOMPLETE_GUARD_FAILURE,
                                   "detail": "no fittable fold: the design file is empty, or every fold's scored rows "
                                             "are outside the confirmation half / are acidic co-extractant rows"})
                led.note(job.key, "error")
                continue
            for fold, ordinal in folds:
                if limit is not None and led.fitted >= int(limit):
                    return led
                t0 = time.perf_counter()
                try:
                    r = run_fold(spec, fold, ordinal, corpus, out_root, store=store, seed_index=i, runners=runners,
                                 code=code, state=state, prereg=prereg)
                except AssertionError as exc:          # a guard failure is named as such, never as a compute cap
                    led.errors.append({"job": job.key, "fold_id": CF.scrub(fold.fold_id, store),
                                       "status": CF.INCOMPLETE_GUARD_FAILURE, "detail": CF.scrub(str(exc), store)})
                    led.note(job.key, "error")
                    continue
                except Exception as exc:
                    # NOT a guard failure: a defect, and the run stops.  It is re-raised SCRUBBED and ``from None`` so
                    # that neither the message nor the chained context can print a withheld seed (addendum 6 item 4);
                    # every deeper message that could carry one is built from job.key, which does
                    raise CF.RedactedRunError(f"{CF.scrub(job.key, store)}/{CF.scrub(fold.fold_id, store)}: "
                                              f"{type(exc).__name__}: {CF.scrub(str(exc), store)}") from None
                led.note(job.key, str(r["status"]), time.perf_counter() - t0)
    return led


# ============================================================================================= #
# assembly -- the claims and their reduced sensitivity set
# ============================================================================================= #

#: R19 item 6's REFIT sensitivities and the design file each is refitted on: V5 only, and only the two refits addendum 1
#: item 4 keeps (strict, HNO3-only), on seed 104729; every other refit is NOT_RUN and named
#: (``discovery.LEARNED_REFITS_NOT_RUN``).  Section 11 names no refit sensitivity for the H3 ablation, so C4's item 6 is
#: the two scoring filters alone -- a materially weaker item 6, which the report states.
#: sensitivity name -> (the CANDIDATE's design file, the COMPARATOR's design file) of that setting.  Section 5 fits a
#: closed-form comparator on the EXACT leave-one-cell-out folds of the setting, never on the heavy arm's batched folds
#: (addendum 2 "comparator_folds"), which is how the discovery scorer reads them (``PRESEAL_JOB``).
REFIT_STEMS: dict[str, dict[str, tuple[str, str]]] = {
    "V5": {"strict_setting": ("V5__strict__batched_max4", "V5__strict__exact"),
           "HNO3_only_cells": ("V5__hno3_only__batched_max4", "V5__hno3_only__exact")},
    "V2": {}, "V1": {}, "V5PAIR": {}, "V6": {},
}


def read_seed_predictions(out_root: Path, arm: str, design_dir: str, seed_index: int, *, code: str | None = None,
                          require_complete: Sequence[str] | None = None) -> pd.DataFrame | None:
    """One arm's confirmation predictions of one design and one OPAQUE seed index, every record verified.

    A record is verified against the registry entry of stage ``confirmation`` (addendum 2 item 5); an incomplete record
    set returns ``None`` and the caller reports the CONTRAST incomplete, never a verdict from part of a design
    (POST-HOC addendum 6 item 3(a)).  ``require_complete`` names the fold ids that must all be present.
    """
    d = CF.conf_root(out_root) / "records" / arm / design_dir / CF.seed_token(seed_index)
    if not d.exists():
        return None
    frames, seen = [], set()
    for js in sorted(d.glob("*.json")):
        rec = D.read_record(js)
        if rec is None:
            return None
        v = REG.verify_record(rec, CF.STAGE)
        if v.get("ok") is False:
            raise D.StaleRecordError(f"{js}: {v.get('reason')}")
        if code is not None and str(rec.get("code_digest")) != str(code):
            raise D.StaleRecordError(f"{js}: written under code {str(rec.get('code_digest'))[:12]}..., not "
                                     f"{str(code)[:12]}...")
        pq = js.with_suffix(".parquet")
        if not pq.exists():
            return None
        seen.add(str(rec.get("fold_id")))
        frames.append(pd.read_parquet(pq))
    if not frames:
        return None
    if require_complete is not None and not set(require_complete) <= seen:
        return None
    return pd.concat(frames, ignore_index=True)


def claim_paired_units(out_root: Path, claim: CF.Claim, attrs: pd.DataFrame, corpus: Any, *, seed_index: int,
                       cand_dir: str, comp_dir: str, filt: str = "none",
                       v6: pd.Series | None = None) -> D.PairedUnits | None:
    """The claim's paired per-unit MAE on ONE withheld-seed index, on identical confirmation-half scored rows."""
    cand = read_seed_predictions(out_root, claim.candidate, cand_dir, seed_index)
    comp = read_seed_predictions(out_root, claim.comparator, comp_dir, seed_index)
    if cand is None or comp is None:
        return None
    mask = v6_by_id(corpus) if v6 is None else v6
    what = f"{claim.key}/i{seed_index}"
    a = CF.confirmation_scoring_frame(cand, attrs, design=claim.design, v6_mask=mask, what=f"{what}/candidate")
    b = CF.confirmation_scoring_frame(comp, attrs, design=claim.design, v6_mask=mask, what=f"{what}/comparator")
    keep = a.index.intersection(b.index)
    a, b = a.loc[keep], b.loc[keep]
    if filt != "none":
        a, b = D.filtered_pair(a, b, filt)
    if not len(a):
        return None
    return D.paired_units(a, b, claim.design, candidate=claim.candidate, comparator=claim.comparator,
                          v6_mask=mask, remainder_groups=corpus.remainder_groups)


def refit_delta(out_root: Path, claim: CF.Claim, attrs: pd.DataFrame, corpus: Any, *, stem: str,
                comp_stem: str, v6: pd.Series | None = None) -> D.PairedUnits | None:
    """One R19 item 6 refit's Delta, from the seed-104729 records (opaque index 0 = the PUBLIC discovery seed).

    BOTH arms come from the setting's OWN design files, as the discovery scorer takes them
    (``g19_score_discovery._pair(store, cand, comp, design, setting, seed)`` reads each arm's frame of that setting):
    the candidate from the setting's batched colouring, the comparator from the setting's EXACT folds (section 5's
    ``comparator_folds``).  Reading the comparator from the PRIMARY design instead would intersect two different scored
    populations and average over primary cells, so the number would not be the setting's Delta at all.
    """
    design, variant, scheme = stem.split("__")
    cdesign, cvariant, cscheme = comp_stem.split("__")
    cand = read_seed_predictions(out_root, claim.candidate, f"{design}__{variant}_{scheme}", 0)
    comp = read_seed_predictions(out_root, claim.comparator, f"{cdesign}__{cvariant}_{cscheme}", 0)
    if cand is None or comp is None:
        return None
    mask = v6_by_id(corpus) if v6 is None else v6
    a = CF.confirmation_scoring_frame(cand, attrs, design=claim.design, v6_mask=mask, what=f"{claim.key}/{stem}/cand")
    b = CF.confirmation_scoring_frame(comp, attrs, design=claim.design, v6_mask=mask, what=f"{claim.key}/{stem}/comp")
    keep = a.index.intersection(b.index)
    if not len(keep):
        return None
    return D.paired_units(a.loc[keep], b.loc[keep], claim.design, candidate=claim.candidate,
                          comparator=claim.comparator, v6_mask=mask, remainder_groups=corpus.remainder_groups)


def claim_sensitivities(out_root: Path, claim: CF.Claim, attrs: pd.DataFrame, corpus: Any, *, store: CF.SeedStore,
                        dirs: Mapping[str, str], v6: pd.Series | None = None) -> dict[str, Any]:
    """R19 item 6 over the REDUCED set of addendum 1 item 4: the registered scoring filters on all 5 withheld seeds
    (they are re-scorings of the primary fits), the strict and HNO3-only REFITS on seed 104729 only (addendum 1 item 4,
    kept by addendum 5 item 1), and every other registered sensitivity DECLARED not run by name and reason.

    The names are ``discovery.SCORING_FILTER_SENSITIVITIES`` -- the SENSITIVITY names, which are what
    ``discovery.apply_scoring_filter`` and ``transfer.REGISTERED_SENSITIVITIES`` both speak.
    ``transfer.scoring_filters`` returns the filter LABELS instead (``acid_grid_rows_excluded`` ->
    ``acid_grid_rows_excluded_scoring``) and omits ``non_DGA_stratum`` altogether, so iterating it would pass an unknown
    filter to ``apply_scoring_filter`` and leave a registered sensitivity out of both the decided and the declared list.
    A value that could not be computed is ``transfer.UNTESTABLE``, the one string R19 item 6 reads as inconclusive
    rather than as a failure, and it is declared in ``not_run`` so ``discovery.reduced_item6`` can report it.
    """
    reg = ET.REGISTERED_SENSITIVITIES[claim.design]
    deltas: dict[str, float | str] = {}
    detail: dict[str, Any] = {}
    reduced: list[str] = []
    not_run: dict[str, str] = {}
    for name in D.SCORING_FILTER_SENSITIVITIES:
        if name not in reg:
            continue
        per_seed = {}
        for i in store.indices():
            pu = claim_paired_units(out_root, claim, attrs, corpus, seed_index=i, v6=v6,
                                    cand_dir=str(dirs.get("candidate", "")), comp_dir=str(dirs.get("comparator", "")),
                                    filt=name)
            if pu is not None:
                per_seed[i] = pu
        if len(per_seed) == CF.N_SEEDS:
            pooled = seed_mean_paired_units(per_seed, design=claim.design, candidate=claim.candidate,
                                            comparator=claim.comparator)
            deltas[name] = float(pooled.delta)
            reduced.append(name)
            detail[name] = {"n_seeds": len(per_seed), "n_units": int(len(pooled.cand_mae)), "kind": "scoring filter",
                            "per_seed": {int(i): float(p.delta) for i, p in sorted(per_seed.items())}}
        else:
            deltas[name] = ET.UNTESTABLE
            not_run[name] = (f"UNTESTABLE: the re-scored contrast record set is complete on {len(per_seed)} of "
                             f"{CF.N_SEEDS} withheld seeds, and a scoring-filter sensitivity is evaluated on all five "
                             "(addendum 6 items 1 and 3(a))")
            detail[name] = {"n_seeds": len(per_seed), "status": "INCOMPLETE", "kind": "scoring filter"}
    for name, (stem, comp_stem) in REFIT_STEMS.get(claim.design, {}).items():
        pu = refit_delta(out_root, claim, attrs, corpus, stem=stem, comp_stem=comp_stem, v6=v6)
        if pu is None:
            deltas[name] = ET.UNTESTABLE
            not_run[name] = (f"UNTESTABLE: no complete seed-104729 record set for {stem} (candidate) and {comp_stem} "
                             "(comparator); this is a member of addendum 1 item 4's OWN reduced set, so item 6 is "
                             "NOT_EVALUATED rather than PASS")
        else:
            deltas[name] = float(pu.delta)
            reduced.append(name)
        detail[name] = {"stem": stem, "comparator_stem": comp_stem, "kind": "refit",
                        "seed": "the public discovery seed 104729", "reading": CF.READINGS["item6_seed"],
                        "n_units": None if pu is None else int(len(pu.cand_mae))}
    for name, why in D.LEARNED_REFITS_NOT_RUN.get(claim.design, {}).items():
        if name not in deltas:
            deltas[name] = ET.UNTESTABLE
            detail[name] = {"status": CF.NOT_RUN, "why": why, "kind": "refit"}
    for name in reg:                       # every registered name is DECIDED or DECLARED; none is silently missing
        if name not in deltas:
            deltas[name] = ET.UNTESTABLE
            not_run[name] = D.REFIT_NOT_RUN
            detail[name] = {"status": CF.NOT_RUN, "why": D.REFIT_NOT_RUN}
    return {"deltas": deltas, "detail": detail, "label": D.ADDENDUM_LABEL, "reduced": sorted(set(reduced)),
            "not_run_extra": not_run,
            "scoring_filters_on_all_five_seeds": [n for n in D.SCORING_FILTER_SENSITIVITIES if n in reg],
            "refits_on_seed_104729_only": sorted(REFIT_STEMS.get(claim.design, {})),
            "reading": CF.READINGS["item6_seed"]}


def score_claims(out_root: Path, plan: Mapping[str, Any], attrs: pd.DataFrame, corpus: Any, *, store: CF.SeedStore,
                 design_dirs: Mapping[str, Mapping[str, str]]) -> list[dict[str, Any]]:
    """Every frozen claim over the 5 withheld seeds, with R19 at confirmation and the reduced sensitivity set.

    A claim whose CONTRAST RECORD SET is incomplete on any withheld seed carries no verdict: it is reported with its own
    status and no per-unit row, because item 4 is 5 of 5 and a partial seed set is never rounded up (POST-HOC addendum 6
    items 1 and 3(a)).
    """
    v6 = v6_by_id(corpus)
    out = []
    for claim in plan["claims"]:
        dirs = dict(design_dirs.get(claim.claim_id) or design_dirs.get("default") or {})
        per_seed: dict[int, D.PairedUnits] = {}
        missing: list[int] = []
        for i in store.indices():
            pu = claim_paired_units(out_root, claim, attrs, corpus, seed_index=i, v6=v6,
                                    cand_dir=str(dirs.get("candidate", "")), comp_dir=str(dirs.get("comparator", "")))
            (per_seed.__setitem__(i, pu) if pu is not None else missing.append(i))
        if missing or len(per_seed) != CF.N_SEEDS:
            out.append({"claim_id": claim.claim_id, "claim": claim.key, "family": claim.family, "design": claim.design,
                        "candidate": claim.candidate, "comparator": claim.comparator, "margin": claim.margin,
                        "half": D.CONFIRMATION, "status": "INCOMPLETE", "completeness_unit": CF.COMPLETENESS_UNIT,
                        "seed_indices_missing": missing, "n_seeds_scored": len(per_seed),
                        "r19_verdict": CF.NOT_EVALUATED, "confirmed": False, "point": float("nan"),
                        "detail": f"the contrast record set is incomplete on seed index(es) {missing}: item 4 is 5 of 5, "
                                  "so this contrast carries no verdict, no per-unit row and no failure-condition input",
                        "reading": CF.READINGS["completeness_unit"]})
            continue
        sens = claim_sensitivities(out_root, claim, attrs, corpus, store=store, dirs=dirs, v6=v6)
        row = score_claim_over_seeds(claim, per_seed, sensitivity_deltas=sens["deltas"],
                                     reduced_sensitivities=sens["reduced"],
                                     sensitivities_not_run_extra=sens["not_run_extra"])
        first = per_seed[sorted(per_seed)[0]]
        row.update(status="COMPLETE", completeness_unit=CF.COMPLETENESS_UNIT, sensitivities=sens,
                   n_units_by_seed={int(i): int(len(p.cand_mae)) for i, p in sorted(per_seed.items())},
                   n_clusters={u: int(s.nunique()) for u, s in first.clusters.items()})
        out.append(row)
    return out


# ============================================================================================= #
# assembly -- S1(c), S1(d), S1(e)
# ============================================================================================= #

S1C_STEM = "V5PAIR__primary__batched"
#: where the selection-half counterweight of S1(c) already lives: discovery computed it on seed 104729 and section 9
#: requires BOTH halves, so this run reads it rather than recomputing a number the discovery record already fixes
S1C_SELECTION_REL = "evaluation/discovery/decisions/decisions.json"


def selection_counterweight(out_root: Path) -> dict[str, Any]:
    """The S1(c) selection-half counterweight, READ from the completed discovery run (never recomputed here).

    Section 9 S1(c) as redefined: ``min_Y Delta_Y >= -0.02`` on the selection half, discovery seed 104729.  Discovery
    scored it (``decisions.json -> S1_components.S1c_selection``) and the plan quotes the value; recomputing it in this
    run could only disagree with a number the discovery record has already fixed, so the value is read and its source
    recorded.
    """
    p = Path(out_root) / S1C_SELECTION_REL
    if not p.exists():
        return {"status": "MISSING", "path": str(p), "min_delta": None,
                "detail": "the discovery decisions file is absent, so the selection-half counterweight cannot be read; "
                          "S1(c) is UNDECIDED, never passed (section 9)"}
    b = json.loads(p.read_text(encoding="utf-8"))
    sel = ((b.get("S1_components") or {}).get("S1c_selection") or {})
    md = sel.get("min_delta")
    return {"status": "READ" if md is not None else "MISSING", "path": str(p), "min_delta": md,
            "threshold": sel.get("min_delta_threshold", -0.02),
            "min_delta_yardstick": sel.get("min_delta_yardstick"),
            "margin_inside_threshold": sel.get("min_delta_margin_inside_threshold"),
            "interval_separated_from_threshold": sel.get("min_delta_interval_separated_from_threshold"),
            "seed": D.PRIMARY_SEED, "half": D.SELECTION,
            "detail": "read from the completed discovery run (selection half, discovery seed 104729); section 9 S1(c) "
                      "passes only if the confirmation half AND this counterweight hold"}


def s1c_half_of_seed(out_root: Path, seed_index: int, seed: int, attrs: pd.DataFrame, corpus: Any) -> dict[str, Any]:
    """One withheld seed's S1(c) half: the yardsticks re-fitted on EXACTLY the batched V5-PAIR folds M2 was fitted on.

    "Same fitted folds" as section 9 registers it and addendum 2 resolves it: ``models.s1c_yardsticks`` re-fits B3x and
    B3i (closed-form) on this seed's batched V5-PAIR design file -- the same file M2's records name -- and HEAVIER needs
    no fit.  The half is the CONFIRMATION half.
    """
    from gen19ct.models import s1c_yardsticks as SY

    js = seed_folds_dir(out_root, seed_index) / f"{S1C_STEM}.json"
    pred = read_seed_predictions(out_root, "M2", "V5PAIR__primary_batched", seed_index)
    if pred is None or not js.exists():
        return {"status": "INCOMPLETE", "seed_index": seed_index,
                "detail": f"no complete M2 V5-PAIR record set for seed index i{seed_index}"}
    fr = corpus.frame
    refit = SY.refit_lookup_yardsticks(js, fr, v6_mask=corpus.v6, guard=SY.registered_guard(fr), table=corpus.table,
                                       systems=corpus.systems, components=corpus.comps,
                                       exclude_from_scoring=corpus.coext, halves=("C",))
    bp = seed_folds_dir(out_root, seed_index) / f"{S1C_STEM}__pairs.parquet"
    inp = SY.yardstick_pair_inputs(refit, attrs, v6_mask=v6_by_id(corpus),
                                   builder_pairs=pd.read_parquet(bp) if bp.exists() else None)
    sel = SY.select_half(inp, "confirmation")
    labels = [ET.fold_qualified_label(f, r) for f, r in zip(pred["fold_id"], pred["row_id"])]
    cand = pd.Series(pred["mean_logD"].to_numpy(dtype=float), index=pd.Index(labels))
    hashes = {str(D.read_record(p.with_suffix(".json")).get("design_hash"))
              for p in (CF.conf_root(out_root) / "records" / "M2" / "V5PAIR__primary_batched"
                        / CF.seed_token(seed_index)).glob("*.parquet")}
    if len(hashes) != 1:
        raise AssertionError(f"M2 V5-PAIR records of seed index i{seed_index} name {len(hashes)} design hashes")
    half = ET.s1c_half(sel["pairs"], EP.derived_logsf(sel["pairs"], cand), half="confirmation", seed=int(seed),
                       lookup_logsf=sel["lookup_logsf"], folds=sel["folds"], v6_mask=sel["v6_mask"],
                       fold_designs={**sel["fold_designs"],
                                     "candidate": ET.s1c_fold_design_id(S1C_STEM, next(iter(hashes)))})
    half["seed_index"] = int(seed_index)
    half["status"] = "COMPLETE"
    return half


def s1c_assembly(out_root: Path, attrs: pd.DataFrame, corpus: Any, *, store: CF.SeedStore) -> dict[str, Any]:
    """S1(c) at confirmation: the 5 per-seed halves combined by the registered rule, with the counterweight beside it.

    The seed combination is ``transfer.s1c_seed_combination`` via ``transfer.s1c_paired_verdict``: each Delta_Y is the
    mean over the 5 withheld seeds of the per-seed cell-pair-macro Delta_Y, its interval a system-cluster percentile
    bootstrap of that seed mean with the same resampled systems applied to every seed, and Delta_Y > 0 in 5 of 5 seeds
    (section 9 S1(c) as redefined; addendum 2; addendum 6 item 1 for the claims uses the same construction).
    """
    halves, missing = [], []
    for i in store.indices():
        h = s1c_half_of_seed(out_root, i, store.seed(i), attrs, corpus)
        if h.get("status") != "COMPLETE":
            missing.append(i)
        else:
            halves.append(h)
    cw = selection_counterweight(out_root)
    if missing or len(halves) != CF.N_SEEDS:
        return {"verdict": CF.UNDECIDED, "status": "INCOMPLETE", "seed_indices_missing": missing,
                "n_seeds_scored": len(halves), "counterweight_selection_half": cw,
                "completeness_unit": CF.COMPLETENESS_UNIT,
                "detail": "S1(c) needs all 5 withheld seeds; with fewer it is UNDECIDED, never passed (section 9)",
                "reading": CF.READINGS["s1c_seed_combination"]}
    verdict = ET.s1c_paired_verdict(confirmation=halves, selection=None)
    direction = {y: {int(h["seed_index"]): float(h["direction"][y]["delta"]) for h in halves}
                 for y in halves[0]["direction"]}
    cw_ok = None if cw.get("min_delta") is None else bool(float(cw["min_delta"]) >= float(cw.get("threshold", -0.02)))
    own = CF.s1c_confirmation(
        {y: {i: {"__macro__": v} for i, v in per.items()} for y, per in direction.items()},
        logsf_gain={y: float(halves[0]["logsf_mae"][y]["gain"]) for y in halves[0]["logsf_mae"]
                    if "gain" in halves[0]["logsf_mae"][y]},
        selection_min_delta=cw.get("min_delta"))
    final = verdict["verdict"] if cw_ok else (CF.UNDECIDED if cw_ok is None else "FAIL")
    return {"verdict": final, "status": "COMPLETE", "confirmation_half": verdict,
            "per_seed_direction_delta_by_yardstick": direction,
            "counterweight_selection_half": cw, "counterweight_pass": cw_ok,
            "cross_check_seed_mean_statistic": {k: own.get(k) for k in ("min_delta", "binding_yardstick",
                                                                        "direction_pass", "logsf_pass")},
            "gamma5": CF.GAMMA5, "eta5": CF.ETA5, "completeness_unit": CF.COMPLETENESS_UNIT,
            "reading": CF.READINGS["s1c_seed_combination"],
            "rule": "section 9 S1(c) as redefined 2026-09-15: PASS only if the confirmation half AND the selection-half "
                    "counterweight hold; if V5-PAIR were dropped S1 would be UNDECIDED, never passed"}


def s1d_s1e_assembly(out_root: Path, attrs: pd.DataFrame, corpus: Any, *, store: CF.SeedStore,
                     arm: str = "M2", design_dir: str = "V5__primary_batched_max4") -> dict[str, Any]:
    """S1(d) calibration and S1(e) support distance on the confirmation half's V5-primary cells.

    The interval metrics are the MEAN over the withheld seeds of the per-seed cell-macro coverage
    (``calibration.seed_mean_interval_metrics`` where the frames allow it, else the per-seed macro averaged here), which
    is the same seed combination addendum 6 item 1 fixes for every other statistic of this run.
    """
    from gen19ct.evaluation import calibration as CAL

    v6 = v6_by_id(corpus)
    per_seed, cats, spear = {}, {}, {}
    for i in store.indices():
        pred = read_seed_predictions(out_root, arm, design_dir, i)
        if pred is None:
            continue
        fr = CF.confirmation_scoring_frame(pred, attrs, design="V5", v6_mask=v6, what=f"S1(d)/i{i}")
        cov = {}
        for lvl, tag in ((0.50, "50"), (0.80, "80"), (0.95, "95")):
            lo, hi = f"lo{int(lvl * 100)}", f"hi{int(lvl * 100)}"
            if lo not in fr.columns or hi not in fr.columns:
                continue
            inside = (fr[EM.Y_COL] >= fr[lo]) & (fr[EM.Y_COL] <= fr[hi])
            unit = fr["unit"] if "unit" in fr.columns else fr["fold_id"]
            cov[tag] = float(inside.groupby(unit).mean().mean())      # per cell, then the section 4 equal-weight macro
        per_seed[i] = cov
        if "domain_status" in fr.columns:
            byc = CAL.coverage_by_category(fr, EM.Regime(design="V5", arm=arm, variant="primary",
                                                         half=D.CONFIRMATION, seed=None, seed_set="confirmation"),
                                           category_col="domain_status", unit_cols=["unit"] if "unit" in fr.columns
                                           else ["fold_id"], v6_mask=v6)
            for r in byc.to_dict("records"):
                if str(r.get("aggregation")) == "unit_macro":
                    c = cats.setdefault(str(r.get("domain_status")), {"n_cells": int(r.get("n_units") or 0), "80": []})
                    c["80"].append(float(r.get("coverage_80", float("nan"))))
        if "support_score" in fr.columns and EM.MAE_COL not in fr.columns:
            u = EM.design_per_unit_table(fr, "V5", v6_mask=v6)
            s = fr.groupby(fr["unit"] if "unit" in fr.columns else fr["fold_id"])["support_score"].mean()
            j = pd.concat([u["mae"].rename("mae"), s.rename("support_score")], axis=1).dropna()
            spear[i] = {"spearman": float(j["mae"].corr(j["support_score"], method="spearman")),
                        "n_units": int(len(j))}
    if len(per_seed) != CF.N_SEEDS:
        return {"s1d": {"status": "INCOMPLETE", "n_seeds_scored": len(per_seed),
                        "detail": "S1(d) needs the confirmation-half V5-primary cells on all 5 withheld seeds",
                        "reading": CF.READINGS["completeness_unit"]},
                "s1e": {"status": "INCOMPLETE"}}
    macro = {tag: float(np.nanmean([per_seed[i].get(tag, float("nan")) for i in sorted(per_seed)]))
             for tag in ("50", "80", "95")}
    by_cat = {k: {"n_cells": v["n_cells"], "80": float(np.nanmean(v["80"]))} for k, v in cats.items()}
    return {"s1d": {**CF.s1d_calibration(macro, by_cat), "per_seed": per_seed, "seed_mean": macro,
                    "seed_combination": SEED_POOLING},
            "s1e": {"status": "NOT_COMPUTED" if not spear else "COMPUTED", "per_seed": spear,
                    "detail": "S1(e) is the Spearman of cell MAE with support_score, <= -0.10 with the system-cluster "
                              "interval excluding 0, and the support-score components must clear the section 8 "
                              "reliability floor; support_s7 is UNDECIDED_UNRELIABLE (-0.1364) and is reported beside it"}}


# ============================================================================================= #
# assembly -- S2, the single V6 run
# ============================================================================================= #

#: the V6 record directory and the arms of section 9 S2.  HEAVIER needs no fit; B3x, B3i and B8 are fitted on the SAME
#: V6 folds and withheld seeds as M2 (section 9 S2(a), resolved 2026-09-15), and B3i is also S2(b)'s lookup-derived
#: comparator and the V5 lookup comparator the hidden Pr/Nd log D MAE is measured against (section 9 table).
V6_DESIGN_DIR = f"{CF.V6_DESIGN}__{CF.V6_VARIANT_NAME}_{CF.V6_SCHEME}"
V6_CANDIDATE_ARM = "M2"
#: yardstick label -> the arm whose predictions derive it (HEAVIER is derived from the metals, not from an arm)
V6_YARDSTICK_ARMS: dict[str, str | None] = {"HEAVIER": None, "B3x_derived": "B3x", "B3i_derived": "B3i", "B8": "B8"}
V6_LOOKUP_YARDSTICK = "B3i_derived"
ACID_STRATUM_COL = "acid_stratum"
HNO3 = "HNO3"


def v6_guard_scored_pairs(pairs: pd.DataFrame, *, folds: pd.Series, v6_mask: pd.Series,
                          test_label: Any = "test", what: str = "V6 pair scoring") -> pd.Index:
    """The section 2 pair checks, with the V6 carve-out read in BOTH directions instead of one.

    ``transfer.guard_scored_pairs`` cannot serve the single V6 run: its last step is
    ``metrics.guard_scoring_index(members, v6_mask, design)``, which calls ``registered.assert_not_scored`` and refuses
    ANY ``V6_TARGET_ROWS`` member -- correct for every other design, and by construction fatal to the one run section
    3.4 registers.  ``metrics.guard_scoring_index`` has the registered escape for exactly this case (*"v6_mask=None is
    accepted only for design V6 (the confirmation run scores those rows)"*), but ``guard_scored_pairs`` requires a
    Series and cannot pass it.  So every one of its checks is made here -- pair isolation, both members in
    ``test_label``, fold-qualified member labels whose fold equals the pair's ``fold`` -- and the carve-out check is
    replaced by the STRONGER two-sided one: every member must BE a ``V6_TARGET_ROW``
    (``confirmation.assert_v6_scope``), so a non-V6 row cannot enter a V6 pair either.
    """
    from gen19ct.data import leakage as LK

    for c in ("idx_a", "idx_b", "fold"):
        if c not in pairs.columns:
            raise KeyError(f"{what}: pairs lack column {c!r}")
    LK.pair_isolation_check(pairs, folds, member_cols=("idx_a", "idx_b"))
    in_test = (pairs["idx_a"].map(folds).astype(object).eq(test_label)
               & pairs["idx_b"].map(folds).astype(object).eq(test_label))
    if not bool(in_test.all()):
        raise AssertionError(f"{what}: only {test_label!r}-{test_label!r} pairs are scored; "
                             f"{int((~in_test).sum())} pair(s) have a member in another fold role")
    pf = pairs["fold"].astype(str)
    for col in ("idx_a", "idx_b"):
        lab = pairs[col].astype(str)
        if not lab.str.contains(ET.FOLD_LABEL_SEP, regex=False).all():
            raise ValueError(f"{what}: member labels must be fold-qualified "
                             f"'fold_id{ET.FOLD_LABEL_SEP}row_id' ({col})")
        bad = lab.str.split(ET.FOLD_LABEL_SEP, n=1).str[0] != pf
        if bad.any():
            raise AssertionError(f"{what}: {int(bad.sum())} pair(s) whose {col} was predicted in another fold than the "
                                 f"pair's (first {lab[bad].iloc[0]!r})")
    members = pd.Index(pd.unique(np.concatenate([pairs["idx_a"].to_numpy(dtype=object),
                                                 pairs["idx_b"].to_numpy(dtype=object)])))
    missing = members.difference(v6_mask.index)
    if len(missing):
        raise ValueError(f"{what}: v6_mask does not cover {len(missing)} member label(s) (first {missing[0]!r}); a mask "
                         "keyed otherwise would pass the V6 guard vacuously")
    CF.assert_v6_scope(members, v6_mask, CF.V6_DESIGN, what)
    EM.guard_scoring_index(members, None, CF.V6_DESIGN, what=what)     # the registered V6 escape, named
    return members


def v6_pair_inputs(pred: pd.DataFrame, attrs: pd.DataFrame, corpus: Any, *, v6: pd.Series, what: str) -> dict[str, Any]:
    """The pair inputs of ONE arm's V6 predictions, built exactly as S1(c)'s are
    (``models.s1c_yardsticks.yardstick_pair_inputs``): rows indexed by the fold-qualified label ``fold_id|row_id`` so a
    prediction of one fold can never be paired with a prediction of another, ``fold`` = the fold id, ``folds`` mapping
    every label to ``test`` and ``v6_mask`` keyed by the same labels, so ``transfer.guard_scored_pairs`` is not vacuous.

    The comparable test-test pairs are section 2's: within one fold and one (publication group, system, condition key).
    For V6 that is inside one system, and each V6 fold IS one system, so no pair crosses a fold.
    """
    fr = CF.confirmation_scoring_frame(pred, attrs, design=CF.V6_DESIGN, v6_mask=v6, what=what)
    need = list(EP.PAIR_KEY_COLS) + [EM.METAL_STATE_COL, EM.Y_COL, ACID_STRATUM_COL]
    missing = [c for c in need if c not in fr.columns]
    if missing:
        raise KeyError(f"{what}: the V6 scoring frame lacks {missing}")
    labels = pd.Index([ET.fold_qualified_label(f, r) for f, r in zip(fr["fold_id"], fr["row_id"])], name="label")
    if labels.has_duplicates:
        raise AssertionError(f"{what}: a (fold, row) pair is scored twice")
    df = fr.copy()
    df.index = labels
    df["fold"] = df["fold_id"]
    pairs = EP.comparable_pairs(df, fold_col="fold", carry_cols=("row_id", ACID_STRATUM_COL))
    folds = pd.Series("test", index=labels, dtype=object)
    mask = pd.Series(df["row_id"].map(v6).to_numpy(dtype=bool), index=labels)
    v6_guard_scored_pairs(pairs, folds=folds, v6_mask=mask, what=what)
    pred_logd = pd.Series(df[EM.PRED_COL].to_numpy(dtype=float), index=labels)
    return {"frame": df, "pairs": pairs, "folds": folds, "v6_mask": mask, "logd": pred_logd,
            "logsf": EP.derived_logsf(pairs, pred_logd)}


def v6_pair_strata(pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    """The registered and the reported pair strata of section 3.4: ``pooled``, ``HNO3`` (section 9 S2(a)'s second
    population and section 14's process case), every other acid medium, and TODGA on its own."""
    a = pairs[f"{ACID_STRATUM_COL}_a"].astype(str).to_numpy()
    b = pairs[f"{ACID_STRATUM_COL}_b"].astype(str).to_numpy()
    same = a == b
    out: dict[str, np.ndarray] = {"pooled": np.ones(len(pairs), dtype=bool), HNO3: same & (a == HNO3)}
    for acid in sorted(set(a[same]) - {HNO3}):
        out[f"acid={acid}"] = same & (a == acid)
    out["acid=mixed"] = ~same
    out[f"system={CF.V6_FOCUS_SYSTEM}"] = (pairs[EM.SYSTEM_COL].astype(str) == CF.V6_FOCUS_SYSTEM).to_numpy()
    return out


def v6_direction_by_system(pairs: pd.DataFrame, cand_logsf: pd.Series, yard_pred: pd.Series, *,
                           direction_only: bool, keep: np.ndarray | None = None) -> dict[str, Any]:
    """Per system: the candidate's and the yardstick's direction accuracy on the SAME pairs -- those where the yardstick
    is defined and ``|observed logSF| >= 0.1`` (section 9 S2(a); ``pairs.V6_DIRECTION_THRESHOLD``, the threshold
    ``pairs.direction_thresholds('V6')`` registers for V6 and for no other design).

    ``transfer.paired_direction_contrast`` cannot serve here: its ``_direction_scores`` is hard-wired to
    ``design='V5-PAIR'`` and therefore to the 0.3 threshold, which is S1(c)'s, not S2(a)'s.  The cell-pair unit of
    S1(c)'s paired rule is ``(system, state_a, state_b)``; for V6 there is exactly one per system
    (``(S, Nd(III), Pr(III))``), so the unit and the cluster are both the system (:data:`CF.READINGS`
    ``s2_averaging_unit``).
    """
    thr = EP.V6_DIRECTION_THRESHOLD
    col = f"dir_{thr:g}"
    c = EP.score_pairs(pairs, cand_logsf, design=CF.V6_DESIGN)[col].to_numpy(dtype=float)
    y = EP.score_pairs(pairs, yard_pred, design=CF.V6_DESIGN, direction_only=direction_only)[col].to_numpy(dtype=float)
    ok = np.isfinite(c) & np.isfinite(y)
    if keep is not None:
        ok = ok & np.asarray(keep, dtype=bool)
    sub = pairs[ok]
    if not len(sub):
        return {"delta_by_system": {}, "candidate_accuracy": {}, "yardstick_accuracy": {}, "n_pairs": 0,
                "n_systems": 0, "threshold": thr}
    sysv = sub[EM.SYSTEM_COL].astype(str).to_numpy()
    cs = pd.Series(c[ok], index=sub.index).groupby(sysv).mean()
    ys = pd.Series(y[ok], index=sub.index).groupby(sysv).mean()
    return {"delta_by_system": {str(k): float(cs[k] - ys[k]) for k in cs.index},
            "candidate_accuracy": {str(k): float(v) for k, v in cs.items()},
            "yardstick_accuracy": {str(k): float(v) for k, v in ys.items()},
            "n_pairs": int(len(sub)), "n_systems": int(len(cs)), "threshold": thr}


def v6_logsf_mae_by_system(pairs: pd.DataFrame, logsf: pd.Series, *, keep: np.ndarray | None = None) -> pd.Series:
    """Per-system logSF MAE (the registered V6 averaging unit: :data:`CF.READINGS` ``s2_averaging_unit``)."""
    ok = np.ones(len(pairs), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    sub, v = pairs[ok], logsf.to_numpy(dtype=float)[ok]
    if not len(sub):
        return pd.Series(dtype=float)
    err = np.abs(v - sub["logsf_obs"].to_numpy(dtype=float))
    return pd.Series(err, index=sub.index).groupby(sub[EM.SYSTEM_COL].astype(str).to_numpy()).mean()


def v6_logsf_intervals(frame: pd.DataFrame, pairs: pd.DataFrame, level: str) -> dict[str, pd.DataFrame]:
    """The predicted-logSF interval of every pair at one level, under both constructions of
    :data:`CF.S2C_INTERVAL_READINGS` (:data:`CF.READINGS` ``s2c_interval_reading``).

    ``interval_arithmetic`` (PRIMARY) is ``[lo_a - hi_b, hi_a - lo_b]``: the interval of the DIFFERENCE with no
    assumption about the dependence of the two rows' errors.  ``quadrature`` combines the half-widths in root-sum-square,
    which assumes independence.  Section 9 S2(c) fixes the bands and the population but names neither, so both are
    reported and the choice is declared.
    """
    lo, hi = f"lo{level}", f"hi{level}"
    if lo not in frame.columns or hi not in frame.columns:
        return {}
    la, ha = pairs["idx_a"].map(frame[lo]), pairs["idx_a"].map(frame[hi])
    lb, hb = pairs["idx_b"].map(frame[lo]), pairs["idx_b"].map(frame[hi])
    pa, pb = pairs["idx_a"].map(frame[EM.PRED_COL]), pairs["idx_b"].map(frame[EM.PRED_COL])
    if pd.concat([la, ha, lb, hb, pa, pb], axis=1).isna().any().any():
        return {}
    centre = pa.to_numpy(dtype=float) - pb.to_numpy(dtype=float)
    arith = pd.DataFrame({"lo": la.to_numpy(dtype=float) - hb.to_numpy(dtype=float),
                          "hi": ha.to_numpy(dtype=float) - lb.to_numpy(dtype=float)}, index=pairs.index)
    wa = (ha.to_numpy(dtype=float) - la.to_numpy(dtype=float)) / 2.0
    wb = (hb.to_numpy(dtype=float) - lb.to_numpy(dtype=float)) / 2.0
    w = np.sqrt(wa ** 2 + wb ** 2)
    quad = pd.DataFrame({"lo": centre - w, "hi": centre + w}, index=pairs.index)
    return {"interval_arithmetic": arith, "quadrature": quad}


def s2_seed_block(out_root: Path, attrs: pd.DataFrame, corpus: Any, *, seed_index: int, v6: pd.Series,
                  arm: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Everything S2 reads from ONE withheld seed's V6 records, or ``None`` when the candidate's set is incomplete."""
    cand_pred = read_seed_predictions(out_root, arm, V6_DESIGN_DIR, seed_index)
    if cand_pred is None:
        return None, []
    cin = v6_pair_inputs(cand_pred, attrs, corpus, v6=v6, what=f"S2/{arm}/i{seed_index}")
    pairs = cin["pairs"]
    strata = v6_pair_strata(pairs)
    yards: dict[str, pd.Series] = {"HEAVIER": EP.heavier_direction(pairs)}
    lookup_logd: pd.Series | None = None
    absent: list[str] = []
    for label, yarm in V6_YARDSTICK_ARMS.items():
        if yarm is None:
            continue
        yp = read_seed_predictions(out_root, yarm, V6_DESIGN_DIR, seed_index)
        if yp is None:
            absent.append(label)
            continue
        yin = v6_pair_inputs(yp, attrs, corpus, v6=v6, what=f"S2/{yarm}/i{seed_index}")
        if not yin["pairs"].index.equals(pairs.index):
            raise AssertionError(f"S2/i{seed_index}: {yarm} was scored on a different pair set than {arm}; S2(a) is a "
                                 "PAIRED contrast on identical pairs")
        yards[label] = yin["logsf"]
        if label == V6_LOOKUP_YARDSTICK:
            lookup_logd = yin["logd"]
    sysv = pairs[EM.SYSTEM_COL].astype(str).to_numpy()
    obs_med = pairs.groupby(sysv)["logsf_obs"].median()
    prd_med = cin["logsf"].groupby(sysv).median()
    direction: dict[str, dict[str, dict[str, float]]] = {}
    dir_detail: dict[str, Any] = {}
    for st in CF.S2A_STRATA:
        key = "pooled" if st == "pooled" else HNO3
        per_y = {label: v6_direction_by_system(pairs, cin["logsf"], yards[label],
                                               direction_only=(label == "HEAVIER"), keep=strata[key])
                 for label in V6_YARDSTICK_ARMS if label in yards}
        direction[st] = {lab: b["delta_by_system"] for lab, b in per_y.items()}
        dir_detail[st] = {lab: {k: v for k, v in b.items() if k != "delta_by_system"} for lab, b in per_y.items()}
    flat = EP.flat_logsf(pairs)
    # ``v6_mask=None`` is metrics.guard_scoring_index's registered escape, accepted for design V6 ALONE because "the
    # confirmation run scores those rows"; a mask here would run registered.assert_not_scored, which refuses ANY V6 row
    # and would make the single registered V6 run impossible.  The scope is checked twice over instead, two-sided:
    # confirmation_scoring_frame's assert_v6_scope on the frame and v6_guard_scored_pairs' on the pair members
    logd_unit = EM.design_per_unit_table(cin["frame"].assign(**{EM.PRED_COL: cin["logd"].to_numpy()}),
                                         CF.V6_DESIGN, v6_mask=None)
    lookup_unit = None if lookup_logd is None else EM.design_per_unit_table(
        cin["frame"].assign(**{EM.PRED_COL: lookup_logd.to_numpy()}), CF.V6_DESIGN, v6_mask=None)
    cov: dict[str, dict[str, float]] = {}
    obs_logsf = pairs["logsf_obs"].to_numpy(dtype=float)
    for lvl in ("80", "95"):
        for reading, iv in v6_logsf_intervals(cin["frame"], pairs, lvl).items():
            inside = (obs_logsf >= iv["lo"].to_numpy(dtype=float)) & (obs_logsf <= iv["hi"].to_numpy(dtype=float))
            cov.setdefault(reading, {})[lvl] = float(inside.mean())     # POOLED over the pairs (section 9 S2(c))
    block = {
        "n_rows": int(len(cin["frame"])), "n_pairs": int(len(pairs)),
        "n_pairs_by_stratum": {k: int(np.asarray(m, dtype=bool).sum()) for k, m in strata.items()},
        "observed_median_by_system": {str(k): float(v) for k, v in obs_med.items()},
        "predicted_median_by_system": {str(k): float(v) for k, v in prd_med.items()},
        "direction_delta_by_system": direction, "direction_detail": dir_detail,
        "logsf_mae_by_system": {st: {str(k): float(v) for k, v in
                                     v6_logsf_mae_by_system(pairs, cin["logsf"], keep=m).items()}
                                for st, m in strata.items()},
        "flat_logsf_mae_by_system": {st: {str(k): float(v) for k, v in
                                          v6_logsf_mae_by_system(pairs, flat, keep=m).items()}
                                     for st, m in strata.items()},
        "lookup_logsf_mae_by_system": None if V6_LOOKUP_YARDSTICK not in yards else
        {st: {str(k): float(v) for k, v in
              v6_logsf_mae_by_system(pairs, yards[V6_LOOKUP_YARDSTICK], keep=m).items()} for st, m in strata.items()},
        "logd_mae_by_system": {str(k): float(v) for k, v in logd_unit["mae"].items()},
        "lookup_logd_mae_by_system": None if lookup_unit is None else
        {str(k): float(v) for k, v in lookup_unit["mae"].items()},
        "logsf_interval_coverage": cov,
        "yardsticks_scored": sorted(yards), "yardsticks_not_scored": absent,
    }
    return block, absent


def s2_assembly(out_root: Path, attrs: pd.DataFrame, corpus: Any, *, store: CF.SeedStore,
                arm: str = V6_CANDIDATE_ARM) -> dict[str, Any]:
    """S2(a), S2(b) and S2(c) of section 9 from the single V6 run's records, with the seed combination of addendum 6.

    Per withheld seed the V6 predictions of the candidate and of every yardstick arm become comparable Nd/Pr test-test
    pairs (:func:`s2_seed_block`); the per-system median predicted and observed logSF give S2(a)'s sign count and the
    per-system paired direction accuracies at ``|observed| >= 0.1`` give S2(a)'s ``Delta_Y``, pooled AND on the HNO3
    pairs; the per-system logSF MAE against FLAT and against the lookup-derived value, and the per-system log D MAE of
    the hidden Pr / Nd rows against the V5 lookup comparator, give S2(b); the pair-level interval coverage gives S2(c).
    Every number is REPORTED pooled, per acid medium and for TODGA alone (:data:`CF.READINGS` ``s2_strata``); only the
    pooled and HNO3 tests decide.

    A missing arm leaves S2 UNDECIDED, never passed, and the contrast record set is the completeness unit
    (POST-HOC addendum 6 item 3(a)).
    """
    v6 = v6_by_id(corpus)
    per_seed: dict[int, dict[str, Any]] = {}
    missing_seeds: list[int] = []
    missing_arms: dict[int, list[str]] = {}
    for i in store.indices():
        block, absent = s2_seed_block(out_root, attrs, corpus, seed_index=i, v6=v6, arm=arm)
        if block is None:
            missing_seeds.append(i)
            continue
        per_seed[i] = block
        if absent:
            missing_arms[i] = absent
    if missing_seeds or len(per_seed) != CF.N_SEEDS:
        return {"status": "INCOMPLETE", "seed_indices_missing": missing_seeds, "n_seeds_scored": len(per_seed),
                "s2a": {"status": CF.NOT_EVALUATED}, "s2b": {"status": CF.NOT_EVALUATED},
                "s2c": {"status": CF.NOT_EVALUATED}, "verdict": CF.UNDECIDED,
                "completeness_unit": CF.COMPLETENESS_UNIT,
                "detail": "the single V6 run's record set is incomplete; S2 carries no verdict (addendum 6 item 3(a))",
                "reading": CF.READINGS["v6_folds_in_run"]}

    def seed_mean_macro(key: str, stratum: str | None = "pooled") -> float:
        """The seed mean of the per-seed SYSTEM macro (addendum 6 item 1; CF.READINGS['s2_averaging_unit'])."""
        vals = []
        for i in sorted(per_seed):
            b = per_seed[i][key]
            b = b if stratum is None else (b or {}).get(stratum)
            vals.append(float("nan") if not b else float(np.nanmean(list(b.values()))))
        # a stratum with no pair in any seed (an acid medium that no V6 pair shares) is NaN, not a warning
        return float(np.nanmean(vals)) if any(np.isfinite(v) for v in vals) else float("nan")

    systems = sorted({s for b in per_seed.values() for s in b["observed_median_by_system"]})
    obs = {s: float(np.nanmean([per_seed[i]["observed_median_by_system"].get(s, np.nan) for i in per_seed]))
           for s in systems}
    prd = {s: float(np.nanmean([per_seed[i]["predicted_median_by_system"].get(s, np.nan) for i in per_seed]))
           for s in systems}
    sign = CF.s2a_sign_count(obs, prd)
    direction = {}
    for st in CF.S2A_STRATA:
        by_y: dict[str, dict[int, dict[str, float]]] = {}
        for label in V6_YARDSTICK_ARMS:
            blocks = {i: (per_seed[i]["direction_delta_by_system"].get(st) or {}).get(label) for i in sorted(per_seed)}
            if any(not b for b in blocks.values()):
                continue
            common = sorted(set.intersection(*(set(b) for b in blocks.values())))
            if not common:
                continue
            by_y[label] = {i: {s: b[s] for s in common} for i, b in blocks.items()}
        direction[st] = CF.s2a_direction(by_y, stratum=st)
    dir_status = [d["status"] for d in direction.values()]
    s2a = {"sign": sign, "direction": direction,
           "status": "PASS" if (sign["status"] == "PASS" and all(s == "PASS" for s in dir_status))
                     else ("FAIL" if (sign["status"] == "FAIL" or "FAIL" in dir_status) else CF.UNDECIDED),
           "yardsticks_not_scored": {int(i): v for i, v in sorted(missing_arms.items())},
           "rule": "section 9 S2(a): the sign count over the 13 systems AND min_Y Delta_Y >= 0.05 under the paired rule "
                   "of S1(c), pooled and in the HNO3 pairs; every yardstick, never a max over yardsticks scored on "
                   "different pair sets",
           "reading": CF.READINGS["s2a"]}
    first = per_seed[sorted(per_seed)[0]]
    mae = seed_mean_macro("logsf_mae_by_system")
    flat_mae = seed_mean_macro("flat_logsf_mae_by_system")
    look = float("nan") if first["lookup_logsf_mae_by_system"] is None else seed_mean_macro("lookup_logsf_mae_by_system")
    gain = CF.seed_mean_system_bootstrap(
        {i: {s: float(per_seed[i]["flat_logsf_mae_by_system"]["pooled"][s]
                      - per_seed[i]["logsf_mae_by_system"]["pooled"][s])
             for s in sorted(set(per_seed[i]["logsf_mae_by_system"]["pooled"])
                             & set(per_seed[i]["flat_logsf_mae_by_system"]["pooled"]))}
         for i in sorted(per_seed)})
    logd = seed_mean_macro("logd_mae_by_system", stratum=None)
    logd_look = (float("nan") if first["lookup_logd_mae_by_system"] is None
                 else seed_mean_macro("lookup_logd_mae_by_system", stratum=None))
    s2b = {**CF.s2b_magnitude(logsf_mae=mae, flat_logsf_mae=flat_mae, lookup_logsf_mae=look,
                              gain_interval_excludes_zero=(None if not np.isfinite(gain["point"])
                                                           else bool(gain["excludes_zero"])),
                              logd_macro_mae=logd, v5_lookup_comparator_mae=logd_look),
           "gain_over_flat_bootstrap": gain,
           "averaging_unit": list(EM.REGISTERED_UNIT_COLS[CF.V6_DESIGN]),
           "lookup_yardstick": V6_LOOKUP_YARDSTICK,
           "by_stratum": {st: {"logsf_mae": seed_mean_macro("logsf_mae_by_system", st),
                               "flat_logsf_mae": seed_mean_macro("flat_logsf_mae_by_system", st)}
                          for st in sorted(first["logsf_mae_by_system"])},
           "reading": CF.READINGS["s2_averaging_unit"]}
    s2c = CF.s2c_coverage({r: {lvl: float(np.nanmean([per_seed[i]["logsf_interval_coverage"][r].get(lvl, np.nan)
                                                      for i in sorted(per_seed)])) for lvl in ("80", "95")}
                           for r in CF.S2C_INTERVAL_READINGS
                           if all(per_seed[i]["logsf_interval_coverage"].get(r) for i in per_seed)})
    parts = (s2a["status"], s2b["status"], s2c["status"])
    return {"status": "COMPLETE", "per_seed": per_seed, "s2a": s2a, "s2b": s2b, "s2c": s2c,
            "s2d": {"status": CF.NOT_RUN, "why": "conditional on Phase H; section 14's process layer runs only in the "
                                                 "exploratory mode of addendum 2 because S1 is UNDECIDED "
                                                 "(addendum 3 item 3)"},
            "verdict": "PASS" if all(p == "PASS" for p in parts) else ("FAIL" if "FAIL" in parts else CF.UNDECIDED),
            "n_systems_scored": len(systems), "n_pairs_registered": {"pooled": CF.V6_N_PAIRS, HNO3: CF.V6_N_PAIRS_HNO3},
            "n_pairs_by_stratum_seed_mean": {st: float(np.nanmean([per_seed[i]["n_pairs_by_stratum"].get(st, np.nan)
                                                                   for i in sorted(per_seed)]))
                                             for st in sorted(first["n_pairs_by_stratum"])},
            "sensitivity_7_systems": {"status": CF.NOT_EVALUATED, "n_systems": CF.V6_SENSITIVITY_N_SYSTEMS,
                                      "n_pairs": CF.V6_SENSITIVITY_N_PAIRS,
                                      "why": "section 3.4's 7-system setting is the >= 10 Pr and Nd rows / >= 5 other "
                                             "Ln(III) subset of the same 13 systems and is a RE-SCORING of these "
                                             "records, not a refit; the subset is defined in feasibility.json -> "
                                             "V6_prnd_double_cell.grid and is reported, never deciding"},
            "completeness_unit": CF.COMPLETENESS_UNIT, "seed_combination": SEED_POOLING,
            "readings": {k: CF.READINGS[k] for k in ("s2a", "s2_averaging_unit", "s2_strata",
                                                     "s2c_interval_reading", "v6_half")},
            "without_s1": "S2 without S1 is reported as 'Pr/Nd reconstructed in the V6 systems; not established as "
                          "general transfer' (section 9)"}


# ============================================================================================= #
# writers
# ============================================================================================= #

#: the record directory of each claim's candidate and comparator (the design file each arm is fitted on; section 5 fits
#: a closed-form comparator on the EXACT design, never on the heavy arm's batched folds -- addendum 2 "comparator_folds")
CLAIM_DESIGN_DIRS: dict[str, dict[str, str]] = {
    "C1": {"candidate": "V5__primary_batched_max4", "comparator": "V5__primary_exact"},
    "C2": {"candidate": "V5__primary_batched_max4", "comparator": "V5__primary_exact"},
    "C3": {"candidate": "V5__primary_batched_max4", "comparator": "V5__primary_exact"},
    "C4": {"candidate": "V2__element_exact", "comparator": "V2__element_exact"},
    "default": {"candidate": "V5__primary_batched_max4", "comparator": "V5__primary_exact"},
}


def s1_verdict(claims: Sequence[Mapping[str, Any]], s1c: Mapping[str, Any], s1de: Mapping[str, Any]) -> dict[str, Any]:
    """S1 = (a) and (b) from the frozen claims, (c) from its own rule, (d) the coverage bands, (e) support distance.

    S1 is PASS only when every component is; a missing or incomplete component makes it UNDECIDED, never passed
    (section 9; addendum 6 item 3(a) for the completeness unit).
    """
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for c in claims:
        by_family.setdefault(str(c.get("family")), []).append(c)
    a = [c for c in claims if str(c.get("family")).startswith("primary")]
    b = [c for c in claims if "S1(b)" in str(c.get("family"))]
    def verdict(rows):
        if not rows:
            return CF.NOT_EVALUATED
        if any(r.get("status") != "COMPLETE" for r in rows):
            return CF.UNDECIDED
        return "PASS" if all(r.get("r19_verdict") == "PASS" for r in rows) else "FAIL"
    parts = {"S1a": verdict(a), "S1b": verdict(b), "S1c": str(s1c.get("verdict", CF.UNDECIDED)),
             "S1d": str((s1de.get("s1d") or {}).get("status", CF.NOT_EVALUATED)),
             "S1e": str((s1de.get("s1e") or {}).get("status", CF.NOT_EVALUATED))}
    ok = all(v == "PASS" for v in parts.values())
    bad = any(v == "FAIL" for v in parts.values())
    return {"verdict": "PASS" if ok else ("FAIL" if bad else CF.UNDECIDED), "components": parts,
            "families": {k: [c.get("claim_id") for c in v] for k, v in sorted(by_family.items())},
            "rule": "section 9 S1: (a) the margin, (b) the constant baselines, (c) selectivity direction and logSF "
                    "magnitude, (d) calibration, (e) error grows with support distance; S1 is UNDECIDED, never passed, "
                    "while any component is missing or incomplete"}


def decisions_body(*, plan: Mapping[str, Any], gate: Mapping[str, Any], store: CF.SeedStore,
                   claims: Sequence[Mapping[str, Any]], s1c: Mapping[str, Any], s1de: Mapping[str, Any],
                   s2: Mapping[str, Any], fold_plans: Sequence[FoldPlan], ledger: FitLedger, code: str) -> dict[str, Any]:
    """``decisions/confirmation.json`` (schema ``gen19.confirmation.v1``), SCRUBBED: no seed value anywhere."""
    p = {k: float(v) for k, v in ((c.get("claim"), (c.get("items") or [{}])[1].get("p_two_sided", float("nan")))
                                  for c in claims) if k}
    body = {"confirmation_schema": "gen19.confirmation.v1", "stage": CF.STAGE, "code_digest": code,
            "plan": {k: v for k, v in plan.items() if k != "claims"}, "claim_ids": list(plan["claim_ids"]),
            "claims": list(claims), "s1": s1_verdict(claims, s1c, s1de), "s1c": s1c, **s1de, "s2": s2,
            "bh_per_family": {"registered": CF.bh_adjust(p), "rule": "section 8: BH per family, printed beside raw p, "
                                                                     "deciding nothing"},
            "seed_store": store.public(), "seed_commitment_sha256": store.digest,
            "seed_indices": list(store.indices()), "fold_files": [f.record() for f in fold_plans],
            "fit": ledger.record(), "gate": {k: v for k, v in gate.items() if k != "readings"},
            "not_run": {"v6_actinide_deltas": CF.V6_ACTINIDE_DELTAS_NOT_RUN, "power_check": CF.POWER_CHECK_NOT_RUN,
                        "s2d_process_stability": (s2.get("s2d") or {})},
            "readings": dict(CF.READINGS), "seed_pooling": SEED_POOLING,
            "s1c_yardstick_artefact": s1c_artefact_diagnosis(),
            "single_run": "section 15: one run, on the withheld seeds and the confirmation half, V6 once; whatever it "
                          "returns is the result, no sixth claim is added and nothing is re-run"}
    return CF.scrub(body, store)


def s1c_artefact_diagnosis(path: Path | None = None) -> dict[str, Any]:
    """Why ``evaluation/discovery/_s1c_yardsticks/*/yardsticks.json`` verifies against no entry under one resolution.

    Diagnosed, not refitted.  The artefact is the scorer's S1(c) yardstick side-file, NOT a discovery fold record: it has
    no ``job`` block, and it names ``registry_stage = "scorer"``.  ``registry.job_stage`` is written for the discovery
    RUNNER's records and believes a ``registry_stage`` only when it is one of ``registry.RUNNER_STAGES``
    (``discovery`` / ``discovery_candidates``), so for this file it falls through to ``discovery`` -- against whose entry
    the file mismatches on ``prereg_addenda_sha256`` (it carries the two-addenda digest, the discovery entry the
    one-addendum digest) and ``prereg_n_addenda``.  Verified against its OWN stage -- ``verify_record(rec, None)``, which
    takes the record's ``registry_stage``, or ``verify_record(rec, "scorer")`` -- it verifies against the SUPERSEDED
    two-addenda ``scorer`` entry, which is exactly what addendum 2 item 5 prescribes for a record written earlier.  So
    nothing is stale and nothing is refitted: the correct resolution for a non-runner artefact is its own
    ``registry_stage``, and a reader that routes it through ``job_stage`` is asking the wrong entry.
    """
    rel = "evaluation/discovery/_s1c_yardsticks"
    out: dict[str, Any] = {"what": f"{rel}/<stem>/s104729/yardsticks.json", "action": "diagnosed, NOT refitted",
                           "rule": "addendum 2 item 5: a record is verified against the registry entry of ITS stage, "
                                   "and a record matching a superseded entry of that stage verifies against it",
                           "diagnosis": (s1c_artefact_diagnosis.__doc__ or "").strip()}
    base = (paths.G19_ROOT if path is None else Path(path)) / rel
    rows = []
    for js in sorted(base.rglob("yardsticks.json")) if base.exists() else []:
        rec = json.loads(js.read_text(encoding="utf-8"))
        own = REG.verify_record(rec, None)
        routed = REG.verify_record(rec, REG.job_stage(rec))
        rows.append({"path": paths.rel(js), "registry_stage_named_by_the_record": rec.get("registry_stage"),
                     "stage_registry_job_stage_would_use": REG.job_stage(rec),
                     "verifies_against_its_own_stage": own.get("ok"), "matched_entry": own.get("matched_entry"),
                     "verifies_when_routed_through_job_stage": routed.get("ok"),
                     "mismatches_when_routed": sorted(routed.get("mismatches") or {}),
                     "prereg_n_addenda": rec.get("prereg_n_addenda"), "kind": rec.get("kind")})
    out["artefacts"] = rows
    out["status"] = "EXPLAINED" if rows and all(r["verifies_against_its_own_stage"] for r in rows) else "OPEN"
    return out


def write_outputs(out_root: Path, body: Mapping[str, Any], *, store: CF.SeedStore, seal: Any) -> list[Path]:
    """``decisions/confirmation.json``, ``tables/confirmation_*.csv|md``, then ``decisions/CONFIRMATION.md`` LAST.

    The leak scan runs twice (POST-HOC addendum 6 item 4): once over everything written so far, which must contain NO
    seed, and once after ``CONFIRMATION.md``, which must contain all five and which is the only file allowed to.
    """
    from gen19ct.manifest import write_csv

    written = [CF.write_decisions(out_root, body)]
    tables = CF.confirmation_tables(body)
    CF.tables_dir(out_root).mkdir(parents=True, exist_ok=True)
    for name, frame in sorted(tables.items()):
        csv = CF.tables_dir(out_root) / f"{name}.csv"
        write_csv(frame, csv)
        written.append(csv)
    md = CF.tables_dir(out_root) / "confirmation_claims.md"
    md.write_text(tables["confirmation_claims"].to_markdown(index=False) if len(tables["confirmation_claims"])
                  else "no claim scored\n", encoding="utf-8")
    written.append(md)
    pre = CF.scan_for_seed_leak(store, [CF.conf_root(out_root), *written])
    if not pre["ok"]:
        raise SystemExit(f"refused: a withheld seed leaked into {pre['files_containing_a_withheld_seed']} -- nothing "
                         "written before decisions/CONFIRMATION.md may contain one (addendum 6 item 4)")
    body = {**dict(body), "seed_leak_scan": pre}
    written[0] = CF.write_decisions(out_root, body)
    verdict = None
    try:
        pp = seal.PreregPaths(root=Path(out_root), repo_root=paths.REPO_ROOT)
        ok, msg = seal.verify_seed_store(pp, Path(store.path))
        verdict = {"ok": bool(ok), "message": str(msg)}
    except Exception as exc:                                     # pragma: no cover - the gate already verified it
        verdict = {"ok": bool(store.verified), "message": f"re-verification unavailable: {exc}"}
    rp = CF.report_path(out_root)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(CF.confirmation_report(body, store=store, verify_seeds=verdict), encoding="utf-8")
    written.append(rp)
    post = CF.scan_for_seed_leak(store, [CF.conf_root(out_root), *written], allow=[rp], require=[rp])
    if not post["ok"]:
        raise SystemExit(f"refused: the seed revelation of section 15 / addendum 6 item 4 did not happen as registered "
                         f"(leaks {post['files_containing_a_withheld_seed']}, incomplete {post['revelation_incomplete']})")
    log(f"seed hygiene: {pre['files_scanned']} file(s) scanned before CONFIRMATION.md (clean) and "
        f"{post['files_scanned']} after (only CONFIRMATION.md names the seeds, all {CF.N_SEEDS})")
    return written


# ============================================================================================= #
# the run
# ============================================================================================= #

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--seed-store", default=None,
                    help="path OUTSIDE the repository holding the 5 withheld seeds; the ONLY way they enter this code")
    ap.add_argument("--dry-run", action="store_true", help="enumerate jobs, folds and the cost; fit nothing")
    ap.add_argument("--check-only", action="store_true", help="run the five gates and print the verdict; do nothing else")
    ap.add_argument("--resume", action="store_true",
                    help="complete unfitted folds of the recorded run (the only thing a second invocation may do)")
    ap.add_argument("--workers", type=int, default=1, help=f"at most {MAX_WORKERS}")
    ap.add_argument("--expect-addenda", type=int, default=None)
    ap.add_argument("--max-folds", type=int, default=None,
                    help="stop after this many FITTED folds (an operator pause, never a demotion)")
    ap.add_argument("--no-manifest", action="store_true")
    return ap.parse_args(argv)


def gate_verdict(out_root: Path, *, seed_store_path: str | None, resume: bool, check: Callable[[], int] | None = None,
                 digests: Callable[[], Mapping[str, Any]] | None = None, expect_addenda: int | None = None,
                 seal: Any = None, registry_path: Path | None = None, plan: Mapping[str, Any] | None = None
                 ) -> tuple[dict[str, Any], CF.SeedStore | None]:
    """The five gates.  (d) needs the store, so it is loaded here -- verified, and never printed."""
    code = code_digest()["combined"]
    store = None
    if seed_store_path is not None:
        store = CF.load_seed_store(seed_store_path, root=out_root, seal=seal if seal is not None else seal_module(),
                                   prereg_paths=(seal or seal_module()).PreregPaths(root=Path(out_root),
                                                                                    repo_root=paths.REPO_ROOT))
    g = CF.gate(out_root, seed_store=store, code_digest=code, resume=resume, check=check, digests=digests,
                expect_addenda=expect_addenda, plan=plan, registry_path=registry_path)
    return g, store


def main(argv: Sequence[str] | None = None, *, check: Callable[[], int] | None = None,
         digests: Callable[[], Mapping[str, Any]] | None = None, seal: Any = None,
         registry_path: Path | None = None) -> int:
    ns = parse_args(argv)
    out_root = Path(ns.out_root)
    if int(ns.workers) > MAX_WORKERS:
        raise SystemExit(f"refused: --workers {ns.workers} exceeds {MAX_WORKERS} (the measured limit of this machine)")
    plan = CF.read_plan(CF.plan_path(out_root))
    log(f"plan: {plan['n_claims']} frozen claim(s) {plan['claim_ids']} (sha256 {plan['sha256'][:12]}...)")

    if ns.dry_run:
        jobs = CF.enumerate_jobs(plan)
        cost = CF.cost_estimate(jobs)
        folds = [p.record() for p in build_confirmation_folds(None, out_root, dry_run=True)]
        body = {"mode": "dry_run", "claims": plan["claim_ids"], "n_jobs": cost["n_jobs"], "n_folds": cost["n_folds"],
                "serial_hours": cost["serial_hours"], "wall_hours_2_workers": cost["wall_hours_2_workers"],
                "by_purpose_serial_hours": cost["by_purpose_serial_hours"], "fold_files_to_build": folds,
                "not_run": {"v6_actinide_deltas": CF.V6_ACTINIDE_DELTAS_NOT_RUN["status"],
                            "power_check": CF.POWER_CHECK_NOT_RUN["status"]},
                "jobs": cost["jobs"], "basis": cost["basis"],
                "run_stages": [{"stage": n, "implemented": ok, "what": why} for n, ok, why in RUN_STAGES],
                "note": "no seed was read: --dry-run enumerates from the plan and the measured unit costs alone"}
        print(json.dumps(body, indent=2, default=str))
        return 0

    g, store = gate_verdict(out_root, seed_store_path=ns.seed_store, resume=bool(ns.resume), check=check,
                            digests=digests, expect_addenda=ns.expect_addenda, seal=seal, registry_path=registry_path,
                            plan=plan)
    if ns.check_only:
        print(json.dumps({"mode": "check_only", **{k: v for k, v in g.items() if k != "readings"}}, indent=2,
                         default=str))
        return 0
    assert store is not None      # the gate refuses without a verified store
    log(f"gates passed; seed store verified against {store.digest[:12]}... ({CF.N_SEEDS} seeds, values withheld)")
    todo = next((n for n, ok, _ in RUN_STAGES if not ok), None)
    if todo is not None:
        # refuse BEFORE a manifest, a fold file or the lock is written: an incomplete runner must leave the single
        # registered run unspent (section 15), not half-spent
        raise SystemExit(refusal(todo))
    with Run(NAME, args={k: v for k, v in vars(ns).items() if k != "seed_store"}, seed=None,
             extra={"stage": STAGE, "seed_commitment_sha256": store.digest, "plan_sha256": plan["sha256"],
                    "claims": plan["claim_ids"], "readings": CF.READINGS,
                    "not_run": {"v6_actinide_deltas": CF.V6_ACTINIDE_DELTAS_NOT_RUN,
                                "power_check": CF.POWER_CHECK_NOT_RUN}}) as run:
        run.extra["gate"] = {k: v for k, v in g.items() if k != "readings"}
        rd = runner_module()
        code = code_digest()["combined"]
        # 1. the withheld-seed fold files, built inside the run (addendum 6 item 2), hashes recorded
        plans = build_confirmation_folds(store, out_root)
        run.extra["fold_files"] = [p.record() for p in plans]
        # 2. the fit loop
        state = D.PlanState.read(D.discovery_root(out_root) / "decisions" / "plan_state.json")
        corpus = confirmation_corpus()
        jobs = CF.enumerate_jobs(plan)
        led = fit_loop(jobs, store, out_root, corpus=corpus, state=state, runners=rd.default_runners(), code=code,
                       prereg=g["prereg"], limit=ns.max_folds)
        run.extra["fit"] = CF.scrub(led.record(), store)
        # 3. assembly, on the records the loop wrote
        attrs = _script("g19_score_discovery").build_attrs()
        claims = score_claims(out_root, plan, attrs, corpus, store=store, design_dirs=CLAIM_DESIGN_DIRS)
        s1c = s1c_assembly(out_root, attrs, corpus, store=store)
        s1de = s1d_s1e_assembly(out_root, attrs, corpus, store=store)
        s2 = s2_assembly(out_root, attrs, corpus, store=store)
        body = decisions_body(plan=plan, gate=g, store=store, claims=claims, s1c=s1c, s1de=s1de, s2=s2,
                             fold_plans=plans, ledger=led, code=code)
        # 4. writers -- CONFIRMATION.md LAST, and it alone names the seeds (addendum 6 item 4)
        written = write_outputs(out_root, body, store=store, seal=seal if seal is not None else seal_module())
        run.extra["outputs"] = [str(p) for p in written]
        run.outputs(*written)
    # the run manifest is written when the Run context exits, i.e. AFTER write_outputs' own scans, so it would be the
    # one artefact of this run no scan ever read.  It is scrubbed field by field, but the scan is the proof, not the
    # intent (confirmation.scrub / scan_for_seed_leak), so it is scanned here
    mfs = [paths.MANIFESTS_DIR / f"{NAME}.json", paths.MANIFESTS_DIR / "run_info" / f"{NAME}.json"]
    final = CF.scan_for_seed_leak(store, [p for p in [*mfs, *written] if Path(p).exists()],
                                  allow=[CF.report_path(out_root)], require=[CF.report_path(out_root)])
    if not final["ok"]:
        raise SystemExit(f"refused: a withheld seed reached {final['files_containing_a_withheld_seed']} after the run "
                         "(the run manifest is written when the Run context exits, after write_outputs' own scans)")
    log(f"post-run scan: {final['files_scanned']} file(s) including the run manifest; only decisions/CONFIRMATION.md "
        "names the seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

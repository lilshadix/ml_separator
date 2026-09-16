"""``scripts/g19_build_folds.py`` -- the registered outer folds of V0, V1, V2, V5, V5-P and V5-PAIR
(brief sections 11, 12, 26 folds/, 27; pre-registration sections 2, 3.1-3.3, 3.6, 15, 16).

Nothing is fitted, nothing is scored, nothing runs on V6.  ``log_D`` is read only inside the leakage guard
(``fold_isolation_check`` compares values of already-matched near-duplicate keys, the section 2 setting).

For every design / variant / scheme the script builds the folds (:mod:`gen19ct.folds`), then for every fold

1. asserts training + hidden is a partition of the MODEL rows (training is never stored);
2. runs ``registered.assert_not_scored`` on the scored rows (plus the X(?) / Sr(III) exclusions);
3. runs ``fold_isolation_check`` at the registered level -- V1 ``level="V1"``, ``near_dup_value_tol=0.005``
   (value-blind for the ``group_near_duplicate_key`` sensitivity); V2 ``element_level=True`` (``False`` for the
   state-level sensitivity); V5 ``component_aware=True`` (``False`` for ``cell_only``; the parent-structure map
   for ``parent_structure``); V5-P additionally ``level="V1"`` on its scored rows.  A violation aborts.

Outputs (``generations/gen19_chem_transfer/folds/``): ``<design>__<variant>__<scheme>.parquet`` (long format
``fold_id, row_id, role, unit, half``) + ``.json`` (per-fold counts, units, meta, fold hashes, design hash),
``V5P__base__eligibility.csv``, ``V5PAIR__primary__candidates.csv``, ``V5PAIR__primary__pairs.parquet``,
``wildcard_copy_pairs.csv`` / ``wildcard_copy_crossings.csv`` (the VL-04 leakage sensitivity: value-matched records
that differ only in the state token or the structure key, and the scored rows of every fold with such a partner in
training) and ``INDEX.json`` (every design, the guard summaries, the wildcard-copy audit and the pre-registration
placeholder numbers: V5 batches per discovery seed, V5-P eligible cells / systems after masking, V5-PAIR eligible
cell pairs by class and systems).

Every fold record carries ``support_tau`` beside its ``fold_hash`` (pre-registration section 13): ``tau_in`` /
``tau_ext`` / ``tau_max`` of ``gen19ct.evaluation.support.fold_tau`` on the fold's training rows (MODEL rows minus the
hidden rows; target-free).

Heavy-arm batched V5-P and V5-PAIR folds (section 3.1 resolution of 2026-09-15; designs ``V5P_BATCHED`` and
``V5PAIR_BATCHED``): ``V5P__base__batched`` and ``V5PAIR__primary__batched`` (+ ``V5PAIR__primary__batched__pairs.parquet``),
seed 104729 only, built from the unbatched unit folds (``cell_holdout.v5p_batched_folds`` /
``v5pair_batched_folds``); every batch passes ``fold_isolation_check`` at its design level and V5-PAIR batches pass
``pair_isolation_check``.  ``INDEX.json`` carries their fold hashes, batch counts by half and a per-batch listing, and
``nested_certificate_safeguard_sample`` (section 2 resolution; ``gen19ct.folds.safeguard``).

Incremental build: ``--designs V5P_BATCHED,V5PAIR_BATCHED --merge-index`` builds only those designs, reads their unit
folds from the hash-verified files, and merges them into the existing ``INDEX.json`` and
``wildcard_copy_crossings.csv``.  It first asserts that every design file of the full build still has the digest
recorded in ``manifests/g19_build_folds.json``, and after writing that those files, their fold and design hashes,
their INDEX entries and their crossing rows are unchanged.  A partial run writes the manifest
``manifests/g19_build_folds_incremental.json`` (never the full-build manifest).  Because the full-build manifest keeps
the earlier digests of ``INDEX.json`` and ``wildcard_copy_crossings.csv``, the incremental manifest records how they
supersede them (``merge.full_build_link``: the full-build digests; ``merge.previous_incremental``: the incremental run
it replaces, whose link is carried forward).  ``nested_certificate_safeguard_sample`` and its reading are recomputed
from every fold file on each such run (the one reading a partial run may change, ``MERGE_MUTABLE_READINGS``).

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_build_folds.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.chemistry import support_graph as SG  # noqa: E402
from gen19ct.data import leakage as L  # noqa: E402
from gen19ct.data import load  # noqa: E402
from gen19ct.data import normalize as N  # noqa: E402
from gen19ct.evaluation import support as ES  # noqa: E402
from gen19ct.folds import cell_holdout as CH  # noqa: E402
from gen19ct.folds import io as FI  # noqa: E402
from gen19ct.folds import metal_holdout as MH  # noqa: E402
from gen19ct.folds import random_split as RS  # noqa: E402
from gen19ct.folds import registered as FR  # noqa: E402
from gen19ct.folds import safeguard as SF  # noqa: E402
from gen19ct.folds import source_holdout as SH  # noqa: E402
from gen19ct.manifest import Run, write_csv, write_json  # noqa: E402

NAME = "g19_build_folds"
INCREMENTAL_NAME = "g19_build_folds_incremental"
OUT = paths.FOLDS_DIR
INDEX_JSON = OUT / "INDEX.json"
CROSSINGS_CSV = OUT / "wildcard_copy_crossings.csv"
COPY_PAIRS_CSV = OUT / "wildcard_copy_pairs.csv"
CROSSING_COLS = ["stem", "fold_id", "half", "row_id", "partner_id", "kind", "strict_copy", "same_publication"]
CROSSING_SORT = ["stem", "fold_id", "row_id", "partner_id", "kind"]
V5P_UNITS, V5PAIR_UNITS = "V5P__base__cell_x_group", "V5PAIR__primary__cell_pair"
V5PAIR_UNIT_PAIRS = "V5PAIR__primary__pairs.parquet"
V5PAIR_BATCHED_PAIRS = "V5PAIR__primary__batched__pairs.parquet"
CELLS_CSV = paths.DATA_AUDIT_DIR / "feasibility_v5_cells.csv"
STATES_CSV = paths.DATA_AUDIT_DIR / "feasibility_metal_states.csv"
FEAS_JSON = paths.DATA_AUDIT_DIR / "feasibility.json"
SYSTEMS_CSV = paths.DESCRIPTORS_DIR / "extractant_systems.csv"
COMPONENTS_CSV = paths.DESCRIPTORS_DIR / "extractant_components.csv"
ALL_DESIGNS = ("V0", "V1", "V2", "V5", "V5P", "V5PAIR", "V5P_BATCHED", "V5PAIR_BATCHED")
GIT_IGNORE_BYTES = 5_000_000
#: readings of the pre-registration taken where its text leaves a choice (the most conservative one), recorded in
#: INDEX.json so a fit script can cite them
READINGS: dict[str, str] = {
    "scored_rows": "hidden rows with a known metal state, not Sr(III), not V6_TARGET_ROWS, in every design (X(?) rows are "
                   "never scored, V0 and V1 included)",
    "v6_rows_in_a_hidden_unit": "V6_TARGET_ROWS inside a held-out unit (V0 fold, V1 group, V2 element, hiding scope of a "
                                "cell) are hidden with it and not scored; they stay in training everywhere else. The same "
                                "holds inside inner folds, so every inner split passes its design guard",
    "guard_test_sets": "V1: every hidden row; V2: the target state's rows; V5 / V5-PAIR / V5-P: every row of the fold's "
                       "cells (all media); V5-P also level V1 on its scored (cell x group) rows",
    "v0_guard": "no fold_isolation_check level (section 2 lists V1-V6); the per-seed partition is asserted",
    "v1_grouped10": "all group_cross_publication_copy groups whole; seeded random order (SeedSequence([seed, 1])), each "
                    "group to the fold with the fewest MODEL rows so far",
    "v1_near_duplicate_key": "checked value-blind (near_dup_value_tol=None), section 3.2; compilation_doi at tol 0.005",
    "v5_hno3_only": "eligibility on HNO3 rows alone (feasibility grid); the cell is hidden whole whatever the medium "
                    "(section 3.1 'cells are hidden whole'); its HNO3 rows are scored",
    "v5_batched": "batches are formed inside each half; carved-out cells (any row in V6_TARGET_ROWS) are not batched; "
                  "the re-check applies the single-cell eligibility rule to MODEL rows minus the registered hiding of "
                  "the batch's OTHER cells; failing cells move to singleton batches",
    "v5p": "base = scored cells at k10 p2 m3 (p on g19_publication_id); publication group = "
           "group_cross_publication_copy; a cell stays eligible only if EVERY one of its group-masked training sets "
           "keeps >= 3 other metal states for the system and >= 3 other systems for the metal (the count with any "
           "passing group is reported as the lenient reading)",
    "v5pair": "comparable pairs on the group_cross_publication_copy basis (section 2 'same publication group'; the "
              "g19_publication_id count is reported beside it), counted among the scored rows of the two cells; a "
              "cell pair 'touches' V6_TARGET_ROWS when any row of either cell is in it; each cell is re-checked with "
              "the other cell's registered hiding applied; test-test pairs are generated from the fold's hidden-scored "
              "rows only",
    "inner_v5": "eligibility recomputed on the outer training rows; cells with a V6_TARGET_ROWS row or no scorable row are "
                "excluded; majority group ties to the smallest group id; groups assigned to the 3 inner folds by the "
                "seeded greedy balance of cell counts (SeedSequence([seed, 15])); subsample of 30 per inner fold "
                "(SeedSequence([seed, 16, k])) -- cell_holdout.inner_cell_majority + inner_cell_assignment, the one "
                "implementation also used by models.interface.InnerCellCalibration",
    "inner_v1": "greedy balance of the publication groups' outer-training row counts over 3 inner folds "
                "(SeedSequence([seed, 11])) -- source_holdout.inner_group_assignment, also used by "
                "models.interface.GroupKFoldCalibration (the V1 and V0 calibration design)",
    "inner_v2": "3 of the V2-eligible states of the outer training rows drawn by SeedSequence([seed, 12]) -- "
                "metal_holdout.inner_state_pick, also used by models.interface.InnerMetalCalibration",
    "support_tau": "section 13 thresholds per fold on MODEL rows minus the fold's hidden rows (evaluation.support.fold_tau: "
                   "50th / 95th / 99.5th percentile of the leave-one-row-out condition_distance_pair over training rows "
                   "whose (state, system) pair has >= 2 training rows); written beside fold_hash, not part of it",
    "wildcard_copies": "leakage sensitivity (verification finding VL-04), REPORTED, not a registered guard: "
                       "leakage.wildcard_copy_pairs on MODEL rows (near-duplicate key sig 6 with the oxidation state "
                       "wildcarded and a different state label, or without the structure key and a different system; "
                       "|delta log D| <= 0.005); strict_copy = every state-wildcard pair plus structure-wildcard pairs "
                       "with a bit-identical, non-decade D_raw; a crossing = a scored row whose partner stays in the "
                       "fold's training rows",
    "v5p_batched": "heavy-arm V5-P folds (section 3.1 resolution 2026-09-15), seed 104729 only, batches inside each half: "
                   "units = the unbatched cell x publication-group folds of V5P__base__cell_x_group; two units conflict "
                   "when they share a metal state, a system or a publication group; greedy colouring with vertex order "
                   "SeedSequence([104729, 6, half]); a unit is re-checked with every other unit of its batch hidden: (a) "
                   "the V5-P base eligibility of its cell (k10 p2 m3, non-bridge) on MODEL rows minus the other units' "
                   "hidden rows (the literal 'every eligibility condition': k and p count the cell's rows left after the "
                   "other units' publication groups are removed) and (b) the masking rule (>= 3 other metal states for the system, >= 3 other systems for "
                   "the metal) on MODEL rows minus every unit's hidden rows; a unit failing either moves to a singleton "
                   "batch (its unbatched fold); a batch hides the union and scores the union of its units' rows; guard "
                   "V5 component-aware on every row of the batch's cells plus V1 on its scored rows",
    "v5pair_batched": "heavy-arm V5-PAIR folds (section 3.1 resolution 2026-09-15), seed 104729 only, batches inside each "
                      "half: units = the unbatched cell-pair folds of V5PAIR__primary__cell_pair; two units conflict when "
                      "they share a metal state or a system; greedy colouring with vertex order SeedSequence([104729, 7, "
                      "half]); every cell of a unit is re-checked eligible (k10 p1 m3, non-bridge) on MODEL rows minus the "
                      "registered hiding of every other cell of the batch (its pair partner included); a failing unit moves "
                      "to a singleton batch; test-test comparable pairs are regenerated from the batch's hidden-scored rows "
                      "and equal the union of its units' pairs (V5PAIR__primary__batched__pairs.parquet, column "
                      "unit_fold_id); guard V5 component-aware plus pair_isolation_check",
    "nested_certificate_safeguard": "section 2 resolution 2026-09-15: gen19ct.folds.safeguard.registered_safeguard_samples; "
                                    "a sample of 20 (stratified by metal category) for every outer fold file discovery "
                                    "fits with an arm whose inner splits carry a certificate -- only the V5 inner design "
                                    "(InnerCellCalibration) builds one: V5-primary exact folds with a scored row and "
                                    "seed-104729 batches, V5 cell-only and parent-structure seed-104729 batches (their "
                                    "certificate construction differs from the primary's), V5-P and V5-PAIR unbatched "
                                    "folds and seed-104729 batches; selection half. The ten V1 grouped folds of seed "
                                    "104729 and the V2 selection-half states keep their draw, labelled vacuous under "
                                    "current code (GroupKFoldCalibration and InnerMetalCalibration set no certificate). A "
                                    "population of <= 20 folds is taken whole. Revised by task X (findings VR-03 / VR-04): "
                                    "the first draw covered V5-primary batches, V1 and V2 only",
}
#: readings a partial ``--merge-index`` run may rewrite: they describe an INDEX entry recomputed from every fold file on
#: each run (the safeguard sample), not a rule of a design this run did not rebuild
MERGE_MUTABLE_READINGS: frozenset[str] = frozenset({"nested_certificate_safeguard"})


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--designs", default=",".join(ALL_DESIGNS),
                    help=f"comma list of {ALL_DESIGNS}; INDEX.json is written for the full set, or with --merge-index")
    ap.add_argument("--merge-index", action="store_true",
                    help="partial run: merge the built designs into the existing INDEX.json and crossings CSV after "
                         "asserting that every full-build design file is unchanged")
    return ap.parse_args(argv)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Book:
    """Collects written files, design summaries, guard statistics, the per-fold section 13 thresholds and the
    wildcard-copy crossings."""

    def __init__(self, frame: pd.DataFrame, v6: pd.Series, support_frame: pd.DataFrame | None = None,
                 systems: pd.DataFrame | None = None, copy_pairs: pd.DataFrame | None = None):
        self.frame, self.v6 = frame, v6
        self.outputs: list[Path] = []
        self.designs: dict[str, dict] = {}
        self.universe = set(frame[FI.ROW_ID].astype(str))
        self.support_frame, self.systems, self.copy_pairs = support_frame, systems, copy_pairs
        self.crossings: list[pd.DataFrame] = []
        self.ids = frame[FI.ROW_ID].astype(str)

    def fold_tau(self, fold: FI.Fold) -> dict:
        hidden = self.ids.isin(set(fold.hidden_row_ids)).to_numpy()
        return ES.fold_tau(self.support_frame[~hidden], self.systems)

    def validate(self, folds: list[FI.Fold]) -> None:
        for f in folds:
            FI.training_ids(f, self.universe)
            FI.assert_scoring_clean(f, self.frame, self.v6)

    def write(self, folds: list[FI.Fold], guard_reports: list[dict], extra: dict | None = None) -> dict:
        warn: Counter = Counter()
        for rep in guard_reports:
            for k, v in rep.get("warnings", {}).items():
                warn[k] += int(v)
        g = {"n_checks": len(guard_reports), "all_ok": all(r["ok"] for r in guard_reports),
             "levels": sorted({"+".join(r["levels"]) for r in guard_reports}),
             "near_dup_value_tol": sorted({str(r["near_dup_value_tol"]) for r in guard_reports}),
             "warnings_summed_over_checks": dict(sorted(warn.items()))}
        f0 = folds[0]
        stem = FI.design_stem(f0.design, f0.variant, f0.scheme)
        extra = dict(extra or {})
        fields = None
        if self.support_frame is not None:
            fields = {f.fold_id: {"support_tau": self.fold_tau(f)} for f in folds}
        if self.copy_pairs is not None:
            cr = FI.copy_crossings(folds, self.copy_pairs)
            if len(cr):
                self.crossings.append(cr.assign(stem=stem))
            strict = cr[cr["strict_copy"].astype(bool)] if len(cr) else cr
            extra["wildcard_copy_crossings"] = {
                "n_folds_with_a_crossing": int(cr["fold_id"].nunique()) if len(cr) else 0,
                "n_scored_row_entries_with_a_training_copy": int(cr[["fold_id", "row_id"]].drop_duplicates().shape[0])
                if len(cr) else 0,
                "n_scored_row_entries_with_a_strict_training_copy": int(strict[["fold_id", "row_id"]].drop_duplicates()
                                                                        .shape[0]) if len(strict) else 0,
                "by_kind": {k: int(cr.loc[cr["kind"] == k, ["fold_id", "row_id"]].drop_duplicates().shape[0])
                            for k in L.WILDCARD_COPY_KINDS} if len(cr) else {k: 0 for k in L.WILDCARD_COPY_KINDS},
                "n_folds_with_a_strict_crossing": int(strict["fold_id"].nunique()) if len(strict) else 0}
        pq, js, summary = FI.write_design(folds, OUT, extra={"guard": g, **extra}, fold_fields=fields)
        self.outputs += [pq, js]
        self.designs[stem] = {"design": f0.design, "variant": f0.variant, "scheme": f0.scheme,
                              "parquet": paths.rel(pq), "index": paths.rel(js), **summary, "guard": g,
                              **{k: v for k, v in dict(extra or {}).items() if k != "folds"}}
        log(f"{stem}: {summary['n_folds']} folds, {summary['n_scored_units']} scored units, "
            f"{summary['n_distinct_scored_rows']} scored rows, guard {g['n_checks']} checks ok={g['all_ok']}")
        return summary


# --------------------------------------------------------------------------------------------- #
# designs
# --------------------------------------------------------------------------------------------- #

def build_v0(book: Book) -> None:
    folds = []
    for seed in FI.DISCOVERY_SEEDS:
        folds += RS.v0_folds(book.frame, seed, book.v6)
    RS.check_partition(folds, book.frame)
    book.validate(folds)
    book.write(folds, [], extra={"note": "diagnostic only; no fold_isolation_check level (pre-registration section 2 "
                                         "lists V1-V6); the partition of MODEL rows per seed is asserted"})


def build_v1(book: Book, feas: dict) -> None:
    fr = book.frame
    halves = FI.registered_halves("V1")
    # exact copy-group design (registered) + the value-blind report
    folds = SH.v1_exact_folds(fr, book.v6, "copy", halves=halves)
    if len(folds) != len(halves) or {f.fold_id for f in folds} != set(halves):
        raise AssertionError(f"V1 exact folds {len(folds)} do not match the registered halves units {len(halves)}")
    book.validate(folds)
    reps = [SH.guard_v1(f, fr) for f in folds]
    blind = [L.fold_isolation_check(fr.index.difference(FI.index_of(fr, f.hidden_row_ids)), FI.index_of(fr, f.hidden_row_ids),
                                    fr, "V1", near_dup_sig=FI.NEAR_DUP_SIG, near_dup_value_tol=None,
                                    raise_on_violation=False) for f in folds]
    blind_name = f"near_duplicate_key_sig{FI.NEAR_DUP_SIG}_shared"
    value_blind = {"n_folds": len(blind), "n_folds_with_value_blind_shared_keys": int(sum(1 for r in blind if not r["ok"])),
                   "total_value_blind_shared_keys": int(sum(r["violations"].get(blind_name, 0) for r in blind)),
                   "note": "REPORTED ONLY (section 2 sensitivity): near-duplicate condition keys shared across the "
                           "boundary at any log D value; the registered guard (tol 0.005) passed on every fold"}
    book.write(folds, reps, extra={"min_group_rows": SH.REGISTERED_MIN_ROWS, "value_blind_check_report": value_blind})
    # sensitivities: exact LOGO at the same >= 20 rule on the coarser groupings
    for variant in ("near_duplicate_key", "compilation_doi"):
        folds = SH.v1_exact_folds(fr, book.v6, variant, halves=halves)
        book.validate(folds)
        book.write(folds, [SH.guard_v1(f, fr) for f in folds],
                   extra={"min_group_rows": SH.REGISTERED_MIN_ROWS, "grouping_column": SH.GROUPINGS[variant]})
    # the ten grouped outer folds per discovery seed (heavy arms)
    folds = []
    for seed in FI.DISCOVERY_SEEDS:
        folds += SH.v1_grouped_folds(fr, book.v6, seed, halves=halves)
    book.validate(folds)
    rows = {seed: [len(f.hidden_row_ids) for f in folds if f.seed == seed] for seed in FI.DISCOVERY_SEEDS}
    for seed, sz in rows.items():
        if sum(sz) != len(fr):
            raise AssertionError(f"V1 grouped seed {seed}: folds do not partition MODEL rows")
    book.write(folds, [SH.guard_v1(f, fr) for f in folds],
               extra={"fold_rows_per_seed": {str(s): {"min": min(v), "max": max(v)} for s, v in rows.items()},
                      "assignment": "seeded greedy balance: groups in a SeedSequence([seed, 1]) random order, each to "
                                    "the fold with the fewest MODEL rows so far"})


def build_v2(book: Book) -> None:
    fr = book.frame
    states = pd.read_csv(STATES_CSV)
    registered = sorted(states.loc[states["v2_eligible__rows100_systems5"].astype(bool), "metal_state"])
    if MH.v2_eligible_states(fr) != registered or len(registered) != 23:
        raise AssertionError("V2 eligible states differ from feasibility_metal_states.csv")
    for element_level in (True, False):
        folds = MH.v2_folds(fr, book.v6, element_level=element_level)
        book.validate(folds)
        book.write(folds, [MH.guard_v2(f, fr) for f in folds],
                   extra={"hiding": "element (all known states + X(?))" if element_level else
                          "state (the state + the element's X(?)); other known states stay in training"})


def _variant_cells(book: Book, cells_tab: pd.DataFrame, variant: CH.Variant) -> tuple[list, list, pd.DataFrame]:
    fr = book.frame
    el = CH.eligible_cells(fr, variant.thresholds, variant.medium)
    mine = set(zip(el.loc[el["eligible"], "metal_state"], el.loc[el["eligible"], "extractant_system_key"]))
    col = variant.feasibility_column()
    reg = set(zip(cells_tab.loc[cells_tab[col].astype(bool), "metal_state"],
                  cells_tab.loc[cells_tab[col].astype(bool), "extractant_system_key"]))
    if mine != reg:
        raise AssertionError(f"{variant.name}: eligible cells {len(mine)} != feasibility column {col} ({len(reg)})")
    space = CH.EligibilitySpace(fr, variant.medium)
    for rec in el[el["n_rows"] >= variant.thresholds.k].itertuples(index=False):
        chk = space.check((rec.metal_state, rec.extractant_system_key), variant.thresholds)
        if chk["eligible"] != bool(rec.eligible):
            raise AssertionError(f"{variant.name}: fast re-check disagrees with eligible_cells on "
                                 f"{rec.metal_state} x {rec.extractant_system_key[:40]}")
    carved_reg = set(zip(cells_tab.loc[cells_tab[col].astype(bool) & cells_tab["in_v6_target_rows"].astype(bool), "metal_state"],
                         cells_tab.loc[cells_tab[col].astype(bool) & cells_tab["in_v6_target_rows"].astype(bool), "extractant_system_key"]))
    v6 = book.v6.to_numpy(dtype=bool)
    carved = {c for c in mine if (CH.cell_rows_mask(fr, c) & v6).any()}
    if carved != carved_reg:
        raise AssertionError(f"{variant.name}: carved-out cells {len(carved)} != feasibility in_v6_target_rows {len(carved_reg)}")
    return sorted(mine), sorted(mine - carved), el


def build_v5(book: Book, feas: dict, placeholders: dict) -> dict:
    fr = book.frame
    cells_tab = pd.read_csv(CELLS_CSV)
    halves = FI.registered_halves("V5_system")
    pmap = FR.parent_component_map()
    batches: dict = {}
    keep: dict = {}
    for name, variant in CH.VARIANTS.items():
        cmap = pmap if variant.parent_structure else None
        cells, scored_cells, _ = _variant_cells(book, cells_tab, variant)
        missing = sorted({c[1] for c in cells} - set(halves))
        if missing:
            raise AssertionError(f"{name}: systems without a registered half: {len(missing)}")
        masks = CH.cell_masks(fr, cells, variant, cmap)
        folds = CH.v5_exact_folds(fr, variant, cells, masks, book.v6, halves, component_map=cmap)
        book.validate(folds)
        n_carved = sum(1 for f in folds if f.meta["carved_out"])
        if any(f.scored_row_ids for f in folds if f.meta["carved_out"]):
            raise AssertionError(f"{name}: a carved-out cell is scored")
        reps = [CH.guard_v5(f, fr, component_map=cmap) for f in folds]
        extra = {"thresholds": variant.thresholds.tag, "medium": variant.medium, "component_aware": variant.component_aware,
                 "parent_structure": variant.parent_structure, "n_eligible_cells": len(cells),
                 "n_carved_out_cells": n_carved, "n_scored_cells": len(scored_cells),
                 "n_scored_systems": len({c[1] for c in scored_cells}),
                 "scored_cells_by_half": {h: sum(1 for c in scored_cells if halves[c[1]] == h) for h in ("S", "C")}}
        book.write(folds, reps, extra=extra)
        if name == "primary":
            want = feas["Q1_V5_cells"]["selection_halves_primary_scored"]
            got = extra["scored_cells_by_half"]
            if got != {h: want[h]["n_cells"] for h in ("S", "C")} or len(scored_cells) != feas["Q1_V5_cells"]["v6_carve_out"]["n_primary_cells_scored_in_discovery"]:
                raise AssertionError(f"V5 primary scored cells per half {got} differ from feasibility.json")
            keep["primary_cells"], keep["primary_masks"] = cells, masks
        # batched folds of the heavy arms
        space = CH.EligibilitySpace(fr, variant.medium)
        bfolds, per_seed = [], {}
        for seed in FI.DISCOVERY_SEEDS:
            sf, st = CH.v5_batched_folds(fr, space, variant, scored_cells, masks, book.v6, halves, seed, component_map=cmap)
            covered = sorted(tuple(c) for f in sf for c in f.meta["cells"])
            if covered != sorted(scored_cells):
                raise AssertionError(f"{name} seed {seed}: batches do not cover every scored cell exactly once")
            for f in sf:
                cs = [tuple(c) for c in f.meta["cells"]]
                if len({c[0] for c in cs}) != len(cs) or len({c[1] for c in cs}) != len(cs):
                    raise AssertionError(f"{f.fold_id}: two cells of one batch share a metal state or a system")
            per_seed[str(seed)] = {"S": st["S"]["n_batches"], "C": st["C"]["n_batches"], "total": st["n_batches"],
                                   "cells_moved_to_singleton": {h: st[h]["n_cells_moved_to_singleton"] for h in ("S", "C")},
                                   "colour_classes": {h: st[h]["n_colour_classes"] for h in ("S", "C")},
                                   "largest_batch": {h: st[h]["largest_batch"] for h in ("S", "C")}}
            bfolds += sf
        book.validate(bfolds)
        breps = [CH.guard_v5(f, fr, component_map=cmap) for f in bfolds]
        book.write(bfolds, breps, extra={"thresholds": variant.thresholds.tag, "batches_per_seed": per_seed,
                                         "rule": "greedy colouring of the conflict graph (shared metal state or system) "
                                                 "within each half, vertex order SeedSequence([seed, 5, half]); each cell "
                                                 "re-checked eligible with the batch's other cells hidden, failures to "
                                                 "singleton batches; carved-out cells are not batched"})
        batches[name] = per_seed
    placeholders["V5_batches_per_discovery_seed"] = batches
    return keep


def build_v5p(book: Book, placeholders: dict, keep: dict | None = None) -> None:
    fr = book.frame
    cells_tab = pd.read_csv(CELLS_CSV)
    base = CH.Variant("v5p_base", CH.V5P_BASE)
    _, scored_cells, _ = _variant_cells(book, cells_tab, base)
    halves = FI.registered_halves("V5_system")
    masks = CH.cell_masks(fr, scored_cells, CH.VARIANTS["primary"])
    space = CH.EligibilitySpace(fr)
    folds, tab, counts = CH.v5p_folds(fr, fr[FI.GROUP_COL], space, scored_cells, masks, book.v6, halves)
    p = OUT / "V5P__base__eligibility.csv"
    write_csv(tab, p)
    book.outputs.append(p)
    placeholders["V5P_eligible_after_masking"] = counts
    if not folds:
        log("V5P: no eligible cell; no fold file written")
        return
    book.validate(folds)
    reps = [r for f in folds for r in CH.guard_v5p(f, fr)]
    book.write(folds, reps, extra={"counts": counts})
    if keep is not None:
        keep["v5p_units"] = folds


def _batch_counts(stats: dict, folds: list[FI.Fold]) -> dict:
    out = {"seed": CH.HEAVY_BATCH_SEED, "total": int(stats["n_batches"]), "n_units": int(stats["n_units"])}
    for h in ("S", "C"):
        if h in stats:
            st = stats[h]
            out[h] = {"n_batches": int(st["n_batches"]), "n_units": int(st["n_units"]),
                      "n_colour_classes": int(st["n_colour_classes"]),
                      "n_units_moved_to_singleton": int(st["n_units_moved_to_singleton"]),
                      "n_singleton_batches": int(sum(1 for f in folds if f.half == h and f.meta["n_units"] == 1)),
                      "largest_batch": int(st["largest_batch"])}
            for k in ("recheck_failures_by_rule", "base_rule_conditions_failed"):
                if k in st:
                    out[h][k] = dict(st[k])
    return out


def build_v5p_batched(book: Book, keep: dict, placeholders: dict) -> None:
    """Heavy-arm V5-P batches (section 3.1 resolution): seed 104729, from the unbatched unit folds."""
    fr = book.frame
    units = keep.get("v5p_units") or FI.read_design(V5P_UNITS, OUT)
    space = CH.EligibilitySpace(fr)
    folds, st = CH.v5p_batched_folds(fr, fr[FI.GROUP_COL], space, units, book.v6)
    unit_by_id = {u.fold_id: u for u in units}
    covered = sorted(u for f in folds for u in f.meta["unit_fold_ids"])
    if covered != sorted(unit_by_id):
        raise AssertionError("V5P batched: batches do not cover every unit fold exactly once")
    for f in folds:
        us = [unit_by_id[u] for u in f.meta["unit_fold_ids"]]
        cs = [tuple(u.meta["cells"][0]) for u in us]
        grps = [u.meta["publication_group"] for u in us]
        if (len({c[0] for c in cs}) != len(us) or len({c[1] for c in cs}) != len(us) or len(set(grps)) != len(us)
                or any(u.half != f.half for u in us)):
            raise AssertionError(f"V5P batched {f.fold_id}: two units share a metal state, a system or a group")
    book.validate(folds)
    reps = [r for f in folds for r in CH.guard_v5p(f, fr)]
    counts = _batch_counts(st, folds)
    book.write(folds, reps, extra={"seed": CH.HEAVY_BATCH_SEED, "unit_design": V5P_UNITS,
                                   "unit_design_hash": FI.design_hash(units), "batches_by_half": counts,
                                   "rule": READINGS["v5p_batched"]})
    stem = FI.design_stem("V5P", "base", "batched")
    book.designs[stem]["batch_listing"] = CH.batch_listing(folds)
    placeholders["V5P_heavy_arm_batches"] = counts


def build_v5pair_batched(book: Book, keep: dict, placeholders: dict, model: pd.DataFrame) -> None:
    """Heavy-arm V5-PAIR batches (section 3.1 resolution): seed 104729, from the unbatched cell-pair folds."""
    fr = book.frame
    units = keep.get("v5pair_units") or FI.read_design(V5PAIR_UNITS, OUT)
    unit_pairs = keep.get("v5pair_pairs")
    if unit_pairs is None:
        unit_pairs = pd.read_parquet(OUT / V5PAIR_UNIT_PAIRS)
    cells = sorted({tuple(c) for u in units for c in u.meta["cells"]})
    masks = keep.get("primary_masks") or CH.cell_masks(fr, cells, CH.VARIANTS["primary"])
    ck = N.condition_key(model.loc[fr.index]).reindex(fr.index)
    halves = FI.registered_halves("V5_system")
    space = CH.EligibilitySpace(fr)
    folds, ptab, st = CH.v5pair_batched_folds(fr, fr[FI.GROUP_COL], ck, space, units, unit_pairs, masks, book.v6, halves)
    unit_by_id = {u.fold_id: u for u in units}
    covered = sorted(u for f in folds for u in f.meta["unit_fold_ids"])
    if covered != sorted(unit_by_id):
        raise AssertionError("V5PAIR batched: batches do not cover every unit fold exactly once")
    if len(ptab) != len(unit_pairs) or set(ptab["unit_fold_id"]) != set(unit_pairs["fold_id"]):
        raise AssertionError("V5PAIR batched: the batch pair list is not the union of the unit pair lists")
    book.validate(folds)
    by_fold = {k: g for k, g in ptab.groupby("fold_id", sort=False)}
    for f in folds:
        CH.check_pair_isolation(f, by_fold[f.fold_id], book.universe)
    pp = OUT / V5PAIR_BATCHED_PAIRS
    ptab.to_parquet(pp, index=False, compression="zstd")
    book.outputs.append(pp)
    reps = [CH.guard_v5(f, fr) for f in folds]
    counts = _batch_counts(st, folds)
    counts["n_test_test_row_pairs"] = int(st["n_test_test_row_pairs"])
    book.write(folds, reps, extra={"seed": CH.HEAVY_BATCH_SEED, "unit_design": V5PAIR_UNITS,
                                   "unit_design_hash": FI.design_hash(units), "batches_by_half": counts,
                                   "pairs_file": paths.rel(pp), "rule": READINGS["v5pair_batched"]})
    stem = FI.design_stem("V5PAIR", "primary", "batched")
    book.designs[stem]["batch_listing"] = CH.batch_listing(folds)
    placeholders["V5PAIR_heavy_arm_batches"] = counts


def build_v5pair(book: Book, keep: dict, placeholders: dict) -> None:
    fr = book.frame
    m = load.load_model_rows(copy=False)
    ck = N.condition_key(m.loc[fr.index]).reindex(fr.index)
    halves = FI.registered_halves("V5_system")
    cells, masks = keep["primary_cells"], keep["primary_masks"]
    space = CH.EligibilitySpace(fr)
    folds, ctab, ptab, counts = CH.v5pair_folds(fr, fr[FI.GROUP_COL], ck, space, cells, masks, book.v6, halves)
    pc = OUT / "V5PAIR__primary__candidates.csv"
    write_csv(ctab, pc)
    pp = OUT / "V5PAIR__primary__pairs.parquet"
    ptab.to_parquet(pp, index=False, compression="zstd")
    book.outputs += [pc, pp]
    placeholders["V5PAIR_eligible_cell_pairs"] = counts
    if not folds:
        return
    book.validate(folds)
    by_fold = {k: g for k, g in ptab.groupby("fold_id", sort=False)}
    for f in folds:
        CH.check_pair_isolation(f, by_fold[f.fold_id], book.universe)
    reps = [CH.guard_v5(f, fr) for f in folds]
    book.write(folds, reps, extra={"counts": counts, "pairs_file": paths.rel(pp), "candidates_file": paths.rel(pc)})
    keep["v5pair_units"], keep["v5pair_pairs"] = folds, ptab


def wildcard_copy_audit(model: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """``leakage.wildcard_copy_pairs`` on the MODEL rows with the copy group of each side, and the corpus counts
    (verification finding VL-04): pairs by kind, strictness, same publication / same copy group, and how many copy
    groups (with MODEL rows) would remain if the cross-group pairs linked publications."""
    pairs = L.wildcard_copy_pairs(model, sig=FI.NEAR_DUP_SIG, tol=FI.NEAR_DUP_VALUE_TOL)
    grp = frame[FI.GROUP_COL].astype(str)
    pairs["group_a"] = grp.reindex(pairs["idx_a"]).to_numpy()
    pairs["group_b"] = grp.reindex(pairs["idx_b"]).to_numpy()
    pairs["same_group"] = pairs["group_a"] == pairs["group_b"]
    nodes = sorted(grp.unique())
    counts: dict = {"n_pairs": int(len(pairs)), "n_copy_groups_with_model_rows": len(nodes)}
    for k in L.WILDCARD_COPY_KINDS:
        sub = pairs[pairs["kind"] == k]
        cross = sub[~sub["same_group"]]
        merged = L.union_components(cross["group_a"], cross["group_b"], nodes)
        counts[k] = {"n_pairs": int(len(sub)), "n_strict": int(sub["strict_copy"].sum()),
                     "n_same_publication": int(sub["same_publication"].sum()),
                     "n_cross_copy_group": int(len(cross)), "n_cross_copy_group_strict": int(cross["strict_copy"].sum()),
                     "n_rows_involved": int(pd.unique(pd.concat([sub["id_a"], sub["id_b"]])).size),
                     "n_D_raw_identical_non_decade": int((sub["D_raw_identical"] & ~sub["decade_value"]).sum()),
                     "n_D_raw_identical_non_decade_same_location": int((sub["D_raw_identical"] & ~sub["decade_value"]
                                                                         & sub["same_location"]).sum()),
                     "n_copy_groups_if_cross_group_pairs_linked": int(len(set(merged.values())))}
    strict_cross = pairs[pairs["strict_copy"] & ~pairs["same_group"]]
    merged = L.union_components(strict_cross["group_a"], strict_cross["group_b"], nodes)
    counts["n_copy_groups_if_strict_cross_group_pairs_linked"] = int(len(set(merged.values())))
    counts["rule"] = READINGS["wildcard_copies"]
    return pairs, counts


# --------------------------------------------------------------------------------------------- #
# incremental merge
# --------------------------------------------------------------------------------------------- #

def _canon(obj) -> str:
    return json.dumps(json.loads(json.dumps(obj, default=str)), sort_keys=True)


def load_base_build() -> dict:
    """The full-build manifest: recorded design hashes and output digests."""
    p = paths.MANIFESTS_DIR / f"{NAME}.json"
    man = json.loads(p.read_text(encoding="utf-8"))
    return {"path": p, "sha256": paths.digests(p)["sha256"], "design_hashes": dict(man["design_hashes"]),
            "digests": {o["path"]: o["sha256"] for o in man["outputs"]}}


def verify_base_files(base: dict, skip_stems=(), read_folds: bool = True) -> dict[str, list[str]]:
    """Assert that every full-build output this run does not rewrite still has its recorded digest (every design
    file, the V5-P / V5-PAIR tables, the wildcard-copy pairs) and, with ``read_folds``, that every design file
    re-derives its fold and design hashes.  Returns the fold hashes per stem."""
    rewritten = {paths.rel(INDEX_JSON), paths.rel(CROSSINGS_CSV)}
    skip = {paths.rel(OUT / f"{s}.{e}") for s in skip_stems for e in ("json", "parquet")}
    for rel, want in sorted(base["digests"].items()):
        if rel in rewritten or rel in skip:
            continue
        got = paths.digests(paths.REPO_ROOT / rel)["sha256"]
        if got != want:
            raise AssertionError(f"{rel}: digest {got} differs from the full-build manifest {want}")
    out: dict[str, list[str]] = {}
    for stem, dh in sorted(base["design_hashes"].items()):
        if stem in skip_stems:
            continue
        if read_folds:
            folds = FI.read_design(stem, OUT)
            if FI.design_hash(folds) != dh:
                raise AssertionError(f"{stem}: design hash differs from the full-build manifest")
            out[stem] = [f.fold_hash for f in folds]
        else:
            body = json.loads((OUT / f"{stem}.json").read_text(encoding="utf-8"))
            out[stem] = [r["fold_hash"] for r in body["folds"]]
    return out


def supersession_link(base: dict) -> dict:
    """How the ``INDEX.json`` and ``wildcard_copy_crossings.csv`` this partial run is about to rewrite descend from the
    full build (task X, finding VR-01: the full-build manifest records their earlier digests, so a manifest-vs-disk check
    needs the chain).  ``full_build_link`` holds the full-build manifest's digests of the two files; when the files on
    disk are the previous incremental run's outputs, that run's link is carried forward and the previous incremental
    manifest is recorded under ``previous_incremental``.  Raises when the files on disk descend from neither."""
    idx_rel, cr_rel = paths.rel(INDEX_JSON), paths.rel(CROSSINGS_CSV)
    idx_now, cr_now = paths.digests(INDEX_JSON)["sha256"], paths.digests(CROSSINGS_CSV)["sha256"]
    full = {"base_manifest": paths.rel(base["path"]), "base_manifest_sha256": base["sha256"],
            "superseded_outputs": {idx_rel: base["digests"][idx_rel], cr_rel: base["digests"][cr_rel]}}
    if idx_now == base["digests"][idx_rel] and cr_now == base["digests"][cr_rel]:
        return {"full_build_link": full, "previous_incremental": None}
    prev_p = paths.MANIFESTS_DIR / f"{INCREMENTAL_NAME}.json"
    if not prev_p.exists():
        raise AssertionError("INDEX.json / wildcard_copy_crossings.csv differ from the full build and no incremental "
                             "manifest records them")
    prev = json.loads(prev_p.read_text(encoding="utf-8"))
    prev_out = {o["path"]: o["sha256"] for o in prev["outputs"]}
    if prev_out.get(idx_rel) != idx_now or prev_out.get(cr_rel) != cr_now:
        raise AssertionError("INDEX.json / wildcard_copy_crossings.csv are neither the full build's nor the previous "
                             "incremental build's outputs")
    pm = prev.get("merge", {})
    link = pm.get("full_build_link")
    if link is None:                          # the first incremental run (before the link was recorded)
        if not (pm.get("index_was_full_build_output") and pm.get("index_sha256_before") == base["digests"][idx_rel]
                and pm.get("crossings_sha256_before") == base["digests"][cr_rel]):
            raise AssertionError("the previous incremental build does not descend from the full build")
        link = full
    return {"full_build_link": link,
            "previous_incremental": {"manifest": paths.rel(prev_p), "manifest_sha256": paths.digests(prev_p)["sha256"],
                                     "arguments": prev.get("arguments"), "design_hashes": prev.get("design_hashes"),
                                     "outputs_now_superseded": {idx_rel: idx_now, cr_rel: cr_now}}}


def read_copy_pairs(p: Path) -> pd.DataFrame:
    """The columns of ``wildcard_copy_pairs.csv`` that ``io.copy_crossings`` reads (booleans parsed strictly)."""
    t = pd.read_csv(p, usecols=["kind", "id_a", "id_b", "same_publication", "strict_copy"], dtype=str,
                    keep_default_na=False)
    for c in ("same_publication", "strict_copy"):
        if not t[c].isin(["True", "False"]).all():
            raise ValueError(f"{p.name}: column {c} is not boolean")
        t[c] = t[c] == "True"
    return t


def read_crossings(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype=str, keep_default_na=False)[CROSSING_COLS]


def merge_crossings(old: pd.DataFrame, new: list[pd.DataFrame], rebuilt: set[str]) -> pd.DataFrame:
    """Old crossing rows of the stems this run did not rebuild, plus this run's rows, in the full-build order;
    asserts the kept rows are unchanged and in their original order."""
    kept = old[~old["stem"].isin(rebuilt)].reset_index(drop=True)
    add = (pd.concat(new, ignore_index=True)[CROSSING_COLS].astype(str) if new
           else pd.DataFrame(columns=CROSSING_COLS, dtype=str))
    out = pd.concat([kept, add], ignore_index=True).sort_values(CROSSING_SORT, kind="mergesort").reset_index(drop=True)
    again = out[out["stem"].isin(set(kept["stem"]))].reset_index(drop=True)
    if not again.equals(kept):
        raise AssertionError("wildcard_copy_crossings.csv: rows of pre-existing designs would change")
    return out


def merge_index(old: dict, new: dict, built_stems: set[str]) -> dict:
    """``new`` (this partial run) merged into ``old`` (the existing INDEX.json).  Asserts that the corpus-level
    entries and the readings of the full build are unchanged, that a rebuilt design reproduces its design hash
    and that a rebuilt placeholder reproduces its value."""
    for k in ("schema", "discovery_seeds", "population", "roles", "hash_rules"):
        if _canon(old.get(k)) != _canon(new[k]):
            raise AssertionError(f"INDEX.json {k!r} differs from the full build")
    for k, v in old["readings"].items():
        if k not in MERGE_MUTABLE_READINGS and new["readings"].get(k) != v:
            raise AssertionError(f"INDEX.json reading {k!r} would change")
    out = copy.deepcopy(old)
    for stem in built_stems:
        if stem in old["designs"] and old["designs"][stem]["design_hash"] != new["designs"][stem]["design_hash"]:
            raise AssertionError(f"{stem}: rebuilt design hash differs from INDEX.json")
        out["designs"][stem] = new["designs"][stem]
    for k, v in new["placeholders"].items():
        if k in old["placeholders"] and _canon(old["placeholders"][k]) != _canon(v):
            raise AssertionError(f"INDEX.json placeholder {k!r} would change")
        out["placeholders"][k] = v
    out["readings"] = new["readings"]
    out["designs_built"] = [d for d in ALL_DESIGNS if d in set(old["designs_built"]) | set(new["designs_built"])]
    out["files_over_5MB"] = {**old.get("files_over_5MB", {}), **new["files_over_5MB"]}
    cw = out["leakage_sensitivity_wildcard_copies"]
    cw["crossings_by_design"] = {**cw.get("crossings_by_design", {}),
                                 **new["leakage_sensitivity_wildcard_copies"]["crossings_by_design"]}
    out[SF.KEY] = new[SF.KEY]
    return out


# --------------------------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ns = parse_args(argv)
    designs = [d for d in ns.designs.split(",") if d]
    bad = [d for d in designs if d not in ALL_DESIGNS]
    if bad:
        raise SystemExit(f"unknown designs {bad}")
    if "V5PAIR" in designs and "V5" not in designs:
        raise SystemExit("V5PAIR needs V5 in the same run")
    full = set(designs) == set(ALL_DESIGNS)
    if ns.merge_index and full:
        raise SystemExit("--merge-index is for a partial --designs list")
    t0 = time.perf_counter()
    with Run(NAME if full else INCREMENTAL_NAME, args=vars(ns), seed=None,
             extra={"discovery_seeds": list(FI.DISCOVERY_SEEDS)}) as run:
        run.inputs(paths.ARCHIVE_MASTER, FI.PUB_COMPONENTS_CSV, FI.HALVES_CSV, CELLS_CSV, STATES_CSV, FEAS_JSON,
                   SYSTEMS_CSV, COMPONENTS_CSV)
        base = old_index = old_crossings = None
        if ns.merge_index:
            base = load_base_build()
            old_index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
            old_crossings = read_crossings(CROSSINGS_CSV)
            for stem, dh in base["design_hashes"].items():
                if old_index["designs"][stem]["design_hash"] != dh:
                    raise AssertionError(f"{stem}: INDEX.json design hash differs from the full-build manifest")
            prior_fold_hashes = verify_base_files(base)
            log(f"merge: {len(prior_fold_hashes)} full-build designs verified against {paths.rel(base['path'])}")
            run.inputs(base["path"], COPY_PAIRS_CSV, OUT / f"{V5P_UNITS}.json", OUT / f"{V5P_UNITS}.parquet",
                       OUT / f"{V5PAIR_UNITS}.json", OUT / f"{V5PAIR_UNITS}.parquet", OUT / V5PAIR_UNIT_PAIRS)
            run.extra["merge"] = {"base_manifest": paths.rel(base["path"]), "base_manifest_sha256": base["sha256"],
                                  "index_sha256_before": paths.digests(INDEX_JSON)["sha256"],
                                  "crossings_sha256_before": paths.digests(CROSSINGS_CSV)["sha256"],
                                  "index_was_full_build_output": paths.digests(INDEX_JSON)["sha256"]
                                  == base["digests"][paths.rel(INDEX_JSON)],
                                  **supersession_link(base)}
        feas = json.loads(FEAS_JSON.read_text(encoding="utf-8"))
        model = load.load_model_rows()
        frame = FI.slim_frame(model.assign(**{FI.GROUP_COL: FI.publication_groups(model)}))
        systems = SG.load_system_table()
        support_frame = SG.prepare_support_frame(model, systems=systems)[list(SG.REQUIRED_COLUMNS)]
        if ns.merge_index:
            copy_pairs = read_copy_pairs(COPY_PAIRS_CSV)
            copy_audit = copy.deepcopy(old_index["leakage_sensitivity_wildcard_copies"])
        else:
            copy_pairs, copy_audit = wildcard_copy_audit(model, frame)
        v6 = FR.v6_target_mask(model, FR.v6_system_set(model)).reindex(frame.index)
        want_v6 = feas["V6_prnd_double_cell"]["v6_target_rows__rows5_otherln2"]
        want_v6 = want_v6["n_rows"] if isinstance(want_v6, dict) else want_v6
        if int(v6.sum()) != int(want_v6):
            raise AssertionError(f"V6_TARGET_ROWS {int(v6.sum())} != feasibility.json {want_v6}")
        if set(frame.loc[v6.to_numpy(), FI.ROW_ID]) != FI.registered_v6_ids():
            raise AssertionError("V6_TARGET_ROWS ids differ between the builder and io.registered_v6_ids")
        book = Book(frame, v6, support_frame=support_frame, systems=systems, copy_pairs=copy_pairs)
        placeholders: dict = {}
        keep: dict = {}
        log(f"MODEL rows {len(frame)}, V6_TARGET_ROWS {int(v6.sum())}, scorable {int(FI.scorable_mask(frame, v6).sum())}")
        if "V0" in designs:
            build_v0(book)
        if "V1" in designs:
            build_v1(book, feas)
        if "V2" in designs:
            build_v2(book)
        if "V5" in designs:
            keep = build_v5(book, feas, placeholders)
        if "V5P" in designs:
            build_v5p(book, placeholders, keep)
        if "V5PAIR" in designs:
            build_v5pair(book, keep, placeholders)
        if "V5P_BATCHED" in designs:
            build_v5p_batched(book, keep, placeholders)
        if "V5PAIR_BATCHED" in designs:
            build_v5pair_batched(book, keep, placeholders, model)
        pc, pcr = COPY_PAIRS_CSV, CROSSINGS_CSV
        if ns.merge_index:
            crossings = merge_crossings(old_crossings, book.crossings, set(book.designs))
        else:
            write_csv(copy_pairs.drop(columns=["idx_a", "idx_b"]), pc)
            book.outputs.append(pc)
            crossings = (pd.concat(book.crossings, ignore_index=True) if book.crossings
                         else pd.DataFrame(columns=CROSSING_COLS))
            crossings = crossings[CROSSING_COLS].sort_values(CROSSING_SORT, kind="mergesort")
        copy_audit["files"] = {"pairs": paths.rel(pc), "crossings": paths.rel(pcr)}
        copy_audit["crossings_by_design"] = {k: v["wildcard_copy_crossings"] for k, v in sorted(book.designs.items())
                                             if "wildcard_copy_crossings" in v}
        big = {paths.rel(p): p.stat().st_size for p in book.outputs if p.stat().st_size > GIT_IGNORE_BYTES}
        index = {
            "schema": FI.SCHEMA, "script": NAME, "discovery_seeds": list(FI.DISCOVERY_SEEDS), "designs_built": designs,
            "population": {"model_rows": int(len(frame)), "v6_target_rows": int(v6.sum()),
                           "scorable_rows": int(FI.scorable_mask(frame, v6).sum()),
                           "unscored_states": list(FI.UNSCORED_STATES)},
            "roles": list(FI.ROLES), "hash_rules": {
                "fold_hash": "sha256 of sorted 'role,row_id' lines (LF)",
                "design_hash": "sha256 of sorted 'fold_id,fold_hash' lines (LF)",
                "assignment_sha256": "sha256 of the LF CSV 'row_id,fold_id,role,unit,half' sorted by row_id, fold_id"},
            "readings": READINGS, "designs": book.designs, "placeholders": placeholders, "files_over_5MB": big,
            "leakage_sensitivity_wildcard_copies": copy_audit}
        if full or ns.merge_index:
            index[SF.KEY] = SF.registered_safeguard_samples(OUT)
            log("nested-certificate safeguard sample: " + ", ".join(
                f"{k} {v['n_drawn']}/{v['population_size']}" for k, v in index[SF.KEY]["designs"].items()))
        if ns.merge_index:
            index = merge_index(old_index, index, set(book.designs))
        if full or ns.merge_index:
            write_csv(crossings, pcr)
            write_json(INDEX_JSON, index)
            book.outputs += [pcr, INDEX_JSON]
        if ns.merge_index:
            # nothing of the full build changed: files, fold hashes, INDEX entries, crossing rows
            after = verify_base_files(base, skip_stems=set(book.designs), read_folds=False)
            if any(after[s] != prior_fold_hashes[s] for s in after):
                raise AssertionError("a pre-existing fold hash changed")
            written = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
            for stem, entry in old_index["designs"].items():
                if stem not in book.designs and _canon(written["designs"][stem]) != _canon(entry):
                    raise AssertionError(f"INDEX.json entry {stem} changed")
            kept = old_crossings[~old_crossings["stem"].isin(set(book.designs))].reset_index(drop=True)
            back = read_crossings(CROSSINGS_CSV)
            if not back[back["stem"].isin(set(kept["stem"]))].reset_index(drop=True).equals(kept):
                raise AssertionError("wildcard_copy_crossings.csv rows of pre-existing designs changed")
            run.extra["merge"]["n_preexisting_designs_verified_unchanged"] = len(after)
            run.extra["merge"]["preexisting_design_hashes"] = {s: base["design_hashes"][s] for s in sorted(after)}
            log(f"merge: {len(after)} pre-existing designs unchanged (files, fold hashes, INDEX entries, crossings)")
        run.outputs(*book.outputs)
        run.extra["design_hashes"] = {k: v["design_hash"] for k, v in sorted(book.designs.items())}
        run.extra["files_over_5MB"] = big
    log(f"done in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Test support for ``tests/test_confirmation_rehearsal.py``: a SYNTHETIC mini-corpus on which the single confirmation
run (``scripts/g19_run_confirmation.py``) can be rehearsed WHOLE, with a fake seed store, without touching the real
corpus, the real fold files, the confirmation half of any real design, V6, or the withheld seeds.

Why a separate module and not the test file: at ``--workers 2`` the fit loop spawns worker PROCESSES, which import the
initializer and the fold function by module name.  A spawned worker inherits the parent's ``sys.path``, so this module
(the tests directory is on it) is importable there, and :func:`init` builds the same synthetic corpus in the worker
that the parent built -- the exact equivalent of monkeypatching ``confirmation_corpus`` in process, done in the child.

The corpus: 6 systems x 6 metal states (five Ln(III) and Am(III), an actinide -- claim C4's ACT_PERMUTED control needs
actinide training rows to permute), 18 rows per cell over 6 condition keys, six publication groups, so section 2 comparable pairs exist inside a group at
every condition key (V6's Pr/Nd pairs, S1(c)'s cell pairs, the S2(c) pair calibration): the groups cycle over
(system, state) and the members of each pair unit are forced into one group (``PAIR_GROUP``).  The V6 system is ``S1``: its
Pr(III) and Nd(III) rows are the ``V6_TARGET_ROWS`` of this corpus.  18 rows per cell gives every state 108 rows, which
is what the registered V2 inner design needs (``InnerMetalCalibration(3, 100, 5)``) for B6's tuning on the C4 folds.

Nothing here reads a real withheld seed: the store is five PLACEHOLDER seeds (``FAKE_SEEDS``).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.evaluation import confirmation as CF
from gen19ct.evaluation import discovery as D
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import pairs as EP
from gen19ct.folds import io as FI
from gen19ct.models import interface as I

SCRIPTS = paths.G19_ROOT / "scripts"

#: five PLACEHOLDER seeds -- in the registered range, not discovery seeds, and not the withheld ones (nobody knows those)
FAKE_SEEDS: list[int] = [123457, 234561, 345612, 456123, 561234]

SYSTEMS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6")
STATES: tuple[str, ...] = ("Pr(III)", "Nd(III)", "La(III)", "Ce(III)", "Sm(III)", "Am(III)")
#: SIX publication groups cycling over (system, state): every group holds exactly one state of each system and one
#: system of each state, so the V5 simultaneous inner design (which assigns inner folds by majority group, two groups
#: to a fold here) hides at most two states of a system and two systems of a state at once and every inner cell
#: passes the k10 / p1 / m3 re-check; four groups put 18 cells in one inner fold and failed it
GROUPS: tuple[str, ...] = ("g1", "g2", "g3", "g4", "g5", "g6")
V6_SYSTEM = "S1"
ROWS_PER_CELL = 18
N_KEYS = 6
#: the condition keys of the Am(III) cells (see ``frame``)
AM_KEYS = 4
#: the V5 primary cells of the confirmation half (distinct states AND systems, so any two make a legal batch).  EVERY
#: withheld seed's batched colouring scores all four -- as the real colourings score the same confirmation-half cells
#: and differ only in how they are batched (seed_mean_paired_units asserts identical averaging units across seeds); the
#: seed decides the pairing of the four into two batches.  The exact comparator design covers the same four cells
V5_CELL_POOL: tuple[tuple[str, str], ...] = (("La(III)", "S2"), ("Ce(III)", "S3"), ("Sm(III)", "S4"), ("Pr(III)", "S5"))
#: the cells of the R19 item 6 refit settings (strict / HNO3-only), batched into one fold and covered by their exact files
V5_REFIT_CELLS: tuple[tuple[str, str], ...] = (("Ce(III)", "S3"), ("Sm(III)", "S4"))
#: the S1(c) cell-pair units: two Ln(III) states of one system sharing a publication group, so HEAVIER is defined
V5PAIR_UNITS: tuple[tuple[tuple[str, str], tuple[str, str]], ...] = ((("Pr(III)", "S2"), ("La(III)", "S2")),
                                                                     (("Ce(III)", "S3"), ("Sm(III)", "S3")))
#: the V2 design's folds: three Ln(III) states (claim C4's Ln test set) and one actinide state (which C4 must skip)
V2_STATES: tuple[str, ...] = ("La(III)", "Ce(III)", "Sm(III)", "Am(III)")
V5_STEM, V5PAIR_STEM, V6_STEM = "V5__primary__batched_max4", "V5PAIR__primary__batched", CF.V6_STEM
#: cells whose publication group is forced, so the two members of a pair unit (V6's Pr/Nd of S1, S1(c)'s two cell-pair
#: units) share a group and form comparable pairs at every condition key
PAIR_GROUP: dict[tuple[str, str], str] = {("Pr(III)", V6_SYSTEM): "g1", ("Nd(III)", V6_SYSTEM): "g1",
                                          ("Pr(III)", "S2"): "g2", ("La(III)", "S2"): "g2",
                                          ("Ce(III)", "S3"): "g6", ("Sm(III)", "S3"): "g6"}

PLAN = """# CONFIRMATION_PLAN (synthetic, for the rehearsal)

## 2. The frozen claims

| # | claim | family | arm (candidate) | comparator | margin | design(s) | discovery | verdict |
|---|---|---|---|---|---|---|---|---|
| **C1** | M2 vs B3i @ V5 | primary (H1, S1(a)) | **M2** | **B3i** | **d5 = 0.10568948790529951** | V5-primary | **+0.262976** | UNDECIDED |
| **C2** | M2 vs B0 @ V5 | S1(b) | **M2** | **B0** | **0.05** | V5-primary | **+1.022150** | UNDECIDED |
| **C3** | M2 vs B6r0 @ V5 | S1(b) | **M2** | **B6r0** | **0.05** | V5-primary | **+0.455132** | UNDECIDED |
| **C4** | B6:WITH vs B6:ACT_PERMUTED @ V2 | H3 | **B6:WITH** | **B6:ACT_PERMUTED** | **0.05** | V2 Ln(III) folds | **+0.081519** | UNDECIDED |
"""


def _script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def rc():
    return _script("g19_run_confirmation")


def seal():
    return _script("g19_seal_prereg")


# --------------------------------------------------------------------------------------------- #
# the synthetic corpus
# --------------------------------------------------------------------------------------------- #

def cell_label(state: str, system: str) -> str:
    return f"{state} x {system}"


def frame() -> pd.DataFrame:
    """The mini-corpus frame (``interface.prepare_frame`` layout plus the archive columns the arms' condition block
    and the section 2 guards read), indexed by row label ``R<k>``."""
    rng = np.random.default_rng(1907)
    recs = []
    for si, sy in enumerate(SYSTEMS):
        for mi, st in enumerate(STATES):
            # the groups cycle over (system, state) so an inner fold -- the cells of one majority group -- hides at most two
            # states of a system and two systems of a state at once (the V5 simultaneous inner design stays satisfiable);
            # the two members of every pair unit are forced into ONE group so section 2 comparable pairs exist between them
            grp = GROUPS[(si + mi) % len(GROUPS)]
            if (st, sy) in PAIR_GROUP:
                grp = PAIR_GROUP[(st, sy)]
            for k in range(ROWS_PER_CELL):
                # the actinide cells sit on FOUR condition keys (unbalanced against the Ln cells' six): on a perfectly
                # balanced grid a permutation of Am log D within (system, group) changes B6's condition slope but not
                # the MEAN prediction over a hidden Ln state's rows, and with one-signed residuals the MAE -- hence
                # claim C4's delta -- was invariant to 1e-16; the imbalance makes the control's effect material
                ck = k % (AM_KEYS if st == "Am(III)" else N_KEYS)
                recs.append({FI.ROW_ID: f"R:{si}{mi}{k:02d}", SG.METAL_COL: st,
                             SG.ELEMENT_COL: SG.metal_properties(st)["symbol"], SG.SYSTEM_COL: sy,
                             SG.PUB_COL: f"pub_{grp}", FI.GROUP_COL: grp, I.PUB_GROUP_COL: grp,
                             "acid_primary": "HNO3", SG.ACID_ANION_COL: "nitrate",
                             SG.LOG_ACID_COL: float(ck) / 6.0, SG.LOG_EXT_COL: -1.0 + 0.1 * si,
                             SG.TEMP_COL: 25.0, "condition_key": f"ck{ck}",
                             "duplicate_group_id": f"dg_{si}{mi}{k:02d}", "g19_ox": 3,
                             "solvent_key": "kerosene", "acid_concentration_M": 1.0 + ck,
                             "extractant_primary_concentration_M": 0.1 + 0.01 * si,
                             "components": None, "acid_signature": "HNO3", "acid_anion": "nitrate",
                             "acid_concentration_organic_M": np.nan, "nitrate_concentration_M": 1.0 + ck,
                             "n_organic_extractants": 1, "metal_concentration_M": 1e-3,
                             "phase_ratio_org_aq": 1.0, "solvent_primary": "kerosene", "solvent_components": None,
                             "modifier_name": None, "modifier_concentration_M": np.nan,
                             "contact_time_min": 30.0, "shaking_time_min": 30.0,
                             I.TARGET_COL: float(0.4 * mi + 0.3 * si + 0.2 * ck + 0.05 * rng.normal())})
    df = pd.DataFrame(recs)
    df.index = pd.Index([f"R{i}" for i in range(len(df))])
    return df


def v6_mask(df: pd.DataFrame) -> pd.Series:
    m = (df[SG.SYSTEM_COL] == V6_SYSTEM) & df[SG.METAL_COL].isin(list(CF.V6_METALS))
    return pd.Series(m.to_numpy(dtype=bool), index=df.index)


def attrs_of(df: pd.DataFrame) -> pd.DataFrame:
    """The scorer's row attributes (``g19_score_discovery.build_attrs`` layout, the columns this run reads), indexed by
    ``row_id``: the section 2 pair key, the metal state, log D, the acid stratum and the registered half of every design
    (all rows are in the CONFIRMATION half of every design here)."""
    ids = df[FI.ROW_ID].astype(str).to_numpy()
    out = pd.DataFrame({"row_id": ids, EM.PUB_GROUP_COL: df[FI.GROUP_COL].astype(str).to_numpy(),
                        EM.SYSTEM_COL: df[SG.SYSTEM_COL].astype(str).to_numpy(),
                        EM.CONDITION_KEY_COL: df["condition_key"].astype(str).to_numpy(),
                        EM.METAL_STATE_COL: df[SG.METAL_COL].astype(object).to_numpy(),
                        EM.Y_COL: df[I.TARGET_COL].to_numpy(dtype=float), "acid_stratum": "HNO3",
                        "v6_target_row": v6_mask(df).to_numpy(),
                        # the columns of the pre-seal attrs that R19 item 6's registered scoring filters read
                        # (g19_run_preseal.build_attrs): S1-S3 diglycolamide, the rest non-DGA; no censoring candidate; the
                        # condition key ck5 flagged as an acid-grid row so the filter drops something
                        "dga_stratum": np.where(df[SG.SYSTEM_COL].isin(["S1", "S2", "S3"]), "diglycolamide", "non_DGA"),
                        "censoring_candidate": False,
                        "acid_grid_flag": (df["condition_key"].astype(str) == "ck5").to_numpy()})
    for design in ("V5", "V5P", "V5PAIR", "V1", "V2"):
        out[f"registered_half_{design}"] = "C"
    out["registered_half_V5-P"] = "C"
    return out.set_index("row_id", drop=False).rename_axis(None)


def _rows_of(df: pd.DataFrame, cells) -> list[str]:
    m = np.zeros(len(df), dtype=bool)
    for st, sy in cells:
        m |= ((df[SG.METAL_COL] == st) & (df[SG.SYSTEM_COL] == sy)).to_numpy()
    return df.loc[m, FI.ROW_ID].astype(str).tolist()


def _v5_fold(df: pd.DataFrame, cells, *, variant: str, scheme: str, fold_id: str, seed: int | None,
             batch_id: str | None = None, max_cells: int | None = None) -> FI.Fold:
    cells = [tuple(c) for c in cells]
    hid = _rows_of(df, cells)
    unit = {}
    for st, sy in cells:
        for r in _rows_of(df, [(st, sy)]):
            unit[r] = cell_label(st, sy)
    meta = {"cells": [list(c) for c in cells], "carved_out": False, "carved_out_cells": [], "component_aware": True,
            "parent_structure": False, "medium_scored": "all"}
    if max_cells is not None:
        meta.update(max_cells_per_batch=int(max_cells), n_cells=len(cells))
    return FI.make_fold(design="V5", variant=variant, scheme=scheme, fold_id=fold_id, half="C", seed=seed, hidden=hid,
                        scored=hid, unit_type="cell", units=[cell_label(*c) for c in cells], batch_id=batch_id,
                        row_unit=unit, row_half={r: "C" for r in hid}, meta=meta)


def write_registered_designs(df: pd.DataFrame, folds_dir: Path) -> dict[str, int]:
    """The seed-independent (or public-seed) design files the run reads from ``folds/``: the exact comparator files
    of V5 (primary, strict, HNO3-only), the seed-104729 batched refit files, the V2 element design (three Ln(III)
    states and Am(III)) and a V1 design that no confirmation job reads.  Returns fold counts per stem."""
    folds_dir.mkdir(parents=True, exist_ok=True)
    v6 = set(df.loc[v6_mask(df).to_numpy(), FI.ROW_ID].astype(str))
    counts: dict[str, int] = {}

    def put(folds):
        FI.write_design(folds, folds_dir)
        counts[FI.design_stem(folds[0].design, folds[0].variant, folds[0].scheme)] = len(folds)

    put([_v5_fold(df, [c], variant="primary", scheme="exact", fold_id=f"{c[0]}__{c[1]}", seed=None)
         for c in V5_CELL_POOL])
    for variant in ("strict", "hno3_only"):
        put([_v5_fold(df, [c], variant=variant, scheme="exact", fold_id=f"{c[0]}__{c[1]}", seed=None)
             for c in V5_REFIT_CELLS])
        put([_v5_fold(df, V5_REFIT_CELLS, variant=variant, scheme="batched_max4", fold_id="s104729_C_b000",
                      seed=D.PRIMARY_SEED, batch_id="s104729_C_b000", max_cells=4)])
    v2 = []
    for st in V2_STATES:
        el = SG.metal_properties(st)["symbol"]
        hid = df.loc[(df[SG.ELEMENT_COL] == el).to_numpy(), FI.ROW_ID].astype(str).tolist()
        sc = [r for r in hid if r not in v6]
        v2.append(FI.make_fold(design="V2", variant="element", scheme="exact", fold_id=st, half="C", seed=None,
                               hidden=hid, scored=sc, unit_type="metal_state", units=[st],
                               row_unit={r: st for r in hid}, row_half={r: "C" for r in hid},
                               meta={"element": el, "n_target_state_rows": len(hid)}))
    put(v2)
    v1 = []
    for grp in GROUPS:
        hid = df.loc[(df[FI.GROUP_COL] == grp).to_numpy(), FI.ROW_ID].astype(str).tolist()
        sc = [r for r in hid if r not in v6]
        v1.append(FI.make_fold(design="V1", variant="copy", scheme="exact", fold_id=f"pub_{grp}", half="C", seed=None,
                               hidden=hid, scored=sc, unit_type="publication_group", units=[grp],
                               row_unit={r: grp for r in hid}, row_half={r: "C" for r in hid},
                               meta={"groups": [grp], "n_groups": 1}))
    put(v1)
    return counts


def make_corpus_dir(cdir: Path) -> Path:
    """Write the synthetic corpus to ``cdir`` (frame, attrs, the registered fold designs) so a spawned worker rebuilds
    the very same objects (:func:`build_corpus`)."""
    cdir.mkdir(parents=True, exist_ok=True)
    df = frame()
    df.assign(__label__=df.index.to_numpy()).to_parquet(cdir / "frame.parquet", index=False)
    attrs_of(df).to_parquet(cdir / "attrs.parquet", index=False)
    counts = write_registered_designs(df, cdir / "folds")
    (cdir / "fold_counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    return cdir


def build_corpus(cdir: Path) -> tuple[Any, pd.DataFrame]:
    """``(Corpus, attrs)`` from :func:`make_corpus_dir`'s files -- the discovery runner's ``Corpus`` over the synthetic
    frame, every row in the confirmation half of every design, ``V6_TARGET_ROWS`` = S1's Pr(III) / Nd(III) rows."""
    rd = rc().runner_module()
    df = pd.read_parquet(cdir / "frame.parquet")
    df.index = pd.Index(df.pop("__label__").astype(str).to_numpy())
    for col in ("components", "solvent_components", "modifier_name"):
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    attrs = pd.read_parquet(cdir / "attrs.parquet").set_index("row_id", drop=False).rename_axis(None)
    v6 = v6_mask(df)
    halves = {d: pd.Series("C", index=df.index) for d in ("V5", "V5P", "V5PAIR", "V1", "V2")}
    corpus = rd.Corpus(frame=df, slim=df, table=I.RowTable(df), v6=v6, coext=pd.Series(False, index=df.index),
                       systems=None, comps=None, cv=None, pmap=None, row_half=halves, folds_dir=cdir / "folds")
    return corpus, attrs


# --------------------------------------------------------------------------------------------- #
# the withheld-seed colourings: stand-ins for the three registered builders, on the synthetic corpus
# --------------------------------------------------------------------------------------------- #

def fold_corpus_stub(df: pd.DataFrame) -> Any:
    """What ``build_confirmation_folds`` reads off ``load_fold_corpus()`` (frame, V6 mask, systems, universe)."""
    return SimpleNamespace(frame=df, v6=v6_mask(df), halves={sy: "C" for sy in SYSTEMS}, cells_tab=None,
                           condition_key=df["condition_key"], book=None, builder=None, pmap=None,
                           v6_systems=(V6_SYSTEM,), universe=set(df[FI.ROW_ID].astype(str)))


def build_v5_batched(fc: Any, variant_name: str, seed: int) -> tuple[list[FI.Fold], dict[str, Any]]:
    """A seed-dependent ``batched_max4`` colouring: primary pairs the four confirmation-half cells into 2 batches of 2
    (the PAIRING depends on the seed, the scored cells do not -- as in the real colourings); strict / HNO3-only batch the
    two refit cells.  Fold ids are seed-shaped (``s<seed>_C_b000``), exactly what the real builder writes before the run
    scrubs them."""
    df = fc.frame
    if variant_name == "primary":
        rng = np.random.default_rng([int(seed), 5])
        pick = [V5_CELL_POOL[i] for i in rng.permutation(len(V5_CELL_POOL))]
        batches = [sorted(pick[:2]), sorted(pick[2:])]
    else:
        batches = [list(V5_REFIT_CELLS)]
    folds = [_v5_fold(df, cells, variant=variant_name, scheme="batched_max4", fold_id=f"s{seed}_C_b{k:03d}", seed=seed,
                      batch_id=f"s{seed}_C_b{k:03d}", max_cells=4) for k, cells in enumerate(batches)]
    stats = {h: {"n_batches": len(folds) if h == "C" else 0, "largest_batch": max(len(b) for b in batches),
                 "n_cells_moved_to_singleton": 0} for h in ("S", "C")}
    return folds, stats


def _pairs_of(df: pd.DataFrame, fold: FI.Fold) -> pd.DataFrame:
    """The section 2 comparable test-test pairs of one fold's scored rows (``pairs.comparable_pairs``), in the fold
    builder's pair-table layout."""
    sc = list(fold.scored_row_ids)
    idmap = pd.Series(df.index, index=df[FI.ROW_ID].astype(str))
    rows = df.loc[idmap.loc[sc].to_numpy()]
    d = pd.DataFrame({"fold": fold.fold_id, EM.PUB_GROUP_COL: rows[FI.GROUP_COL].astype(str).to_numpy(),
                      EM.SYSTEM_COL: rows[SG.SYSTEM_COL].astype(str).to_numpy(),
                      EM.CONDITION_KEY_COL: rows["condition_key"].astype(str).to_numpy(),
                      EM.METAL_STATE_COL: rows[SG.METAL_COL].astype(object).to_numpy(),
                      EM.Y_COL: rows[I.TARGET_COL].to_numpy(dtype=float)},
                     index=pd.Index(rows[FI.ROW_ID].astype(str).to_numpy()))
    p = EP.comparable_pairs(d, fold_col="fold")
    return pd.DataFrame({"fold_id": fold.fold_id, "row_id_a": p["idx_a"].astype(str).to_numpy(),
                         "row_id_b": p["idx_b"].astype(str).to_numpy(), "metal_state_a": p["state_a"].to_numpy(),
                         "metal_state_b": p["state_b"].to_numpy(),
                         "publication_group": p[EM.PUB_GROUP_COL].to_numpy(),
                         "condition_key_id": p[EM.CONDITION_KEY_COL].to_numpy(),
                         "category_class": p["category_class"].to_numpy()})


def build_v5pair_batched(fc: Any, seed: int) -> tuple[list[FI.Fold], pd.DataFrame, dict[str, Any]]:
    """A seed-dependent batched V5-PAIR colouring of the two cell-pair units: one batch of both or two batches of one.
    The SCORED pairs are the same in every seed (section 9's confirmation pair set), only the batching differs."""
    df = fc.frame
    rng = np.random.default_rng([int(seed), 7])
    one_batch = bool(rng.integers(0, 2))
    batches = [list(V5PAIR_UNITS)] if one_batch else [[u] for u in V5PAIR_UNITS]
    folds, tabs = [], []
    for k, units in enumerate(batches):
        cells = [c for u in units for c in u]
        hid = _rows_of(df, cells)
        unit = {}
        for st, sy in cells:
            for r in _rows_of(df, [(st, sy)]):
                unit[r] = cell_label(st, sy)
        f = FI.make_fold(design="V5PAIR", variant="primary", scheme="batched", fold_id=f"s{seed}_C_b{k:03d}", half="C",
                         seed=seed, hidden=hid, scored=hid, unit_type="cell", units=[cell_label(*c) for c in cells],
                         batch_id=f"s{seed}_C_b{k:03d}", row_unit=unit, row_half={r: "C" for r in hid},
                         meta={"cells": [list(c) for c in cells],
                               "cell_pairs": [f"{cell_label(*a)} || {cell_label(*b)}" for a, b in units],
                               "unit_fold_ids": [f"{a[0]}__{a[1]}__{b[0]}__{b[1]}" for a, b in units],
                               "n_units": len(units), "category_classes": ["Ln-Ln"] * len(units),
                               "component_aware": True, "parent_structure": False, "carved_out": False,
                               "carved_out_cells": [], "medium_scored": "all"})
        tab = _pairs_of(df, f)
        f = FI.make_fold(design=f.design, variant=f.variant, scheme=f.scheme, fold_id=f.fold_id, half=f.half,
                         seed=f.seed, hidden=f.hidden_row_ids, scored=f.scored_row_ids, unit_type=f.unit_type,
                         units=f.units, batch_id=f.batch_id, row_unit=f.row_unit, row_half=f.row_half,
                         meta={**f.meta, "n_comparable_row_pairs": int(len(tab))})
        folds.append(f)
        tabs.append(tab)
    ptab = pd.concat(tabs, ignore_index=True)
    return folds, ptab, {"batches": {"C": len(folds), "S": 0}, "n_test_test_row_pairs": int(len(ptab)),
                         "unit_design": "synthetic cell-pair units", "n_units": len(V5PAIR_UNITS),
                         "one_batch": one_batch}


def build_v6_folds(fc: Any) -> tuple[list[FI.Fold], dict[str, Any]]:
    """The section 3.4 design on the mini-corpus: ONE system (S1), Pr(III) x S1 and Nd(III) x S1 hidden together."""
    df = fc.frame
    cells = [("Pr(III)", V6_SYSTEM), ("Nd(III)", V6_SYSTEM)]
    hid = _rows_of(df, cells)
    unit = {}
    for st, sy in cells:
        for r in _rows_of(df, [(st, sy)]):
            unit[r] = cell_label(st, sy)
    f = FI.make_fold(design=CF.V6_DESIGN, variant=CF.V6_VARIANT_NAME, scheme=CF.V6_SCHEME,
                     fold_id=f"v6_prnd__{hashlib.sha1(V6_SYSTEM.encode()).hexdigest()[:10]}", half="NA", seed=None,
                     hidden=hid, scored=hid, unit_type="system", units=[V6_SYSTEM], row_unit=unit,
                     row_half={r: "NA" for r in hid},
                     meta={"cells": [list(c) for c in cells], "system": V6_SYSTEM, "carved_out_cells": [],
                           "carved_out": False, "medium_scored": "all", "component_aware": True,
                           "parent_structure": False, "metals": list(CF.V6_METALS),
                           "n_scored_pr": len(_rows_of(df, [cells[0]])), "n_scored_nd": len(_rows_of(df, [cells[1]]))})
    return [f], {"n_systems": 1, "per_system": [{"system": V6_SYSTEM, "n_hidden": len(hid), "n_scored": len(hid)}],
                 "n_scored_rows": len(hid), "guard_checks": 1, "guard_all_ok": True}


# --------------------------------------------------------------------------------------------- #
# the run skeleton (out_root) and the worker initializer
# --------------------------------------------------------------------------------------------- #

def write_store(store_path: Path, root: Path, seeds=None) -> Path:
    """The fake seed store and its matching commitment under ``root/manifests`` (the runner verifies one against the
    other, exactly as it does with the real ones)."""
    sealm = seal()
    payload = {"n": CF.N_SEEDS, "salt": "ab" * 32, "schema": sealm.SEED_SCHEMA,
               "seeds": sorted(FAKE_SEEDS if seeds is None else seeds)}
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "manifests" / "confirmation_seeds_sha256.txt").write_text(sealm.seed_digest(payload) + "\n",
                                                                     encoding="utf-8", newline="\n")
    return store_path


def write_run_skeleton(root: Path, store_path: Path) -> None:
    """Everything the runner reads under ``--out-root`` before it fits: the commitment, the frozen plan, discovery's
    plan state (the B6 check verdicts the real run left) and the discovery decisions file (the S1(c) counterweight)."""
    write_store(store_path, root)
    (root / "decisions").mkdir(parents=True, exist_ok=True)
    (root / "decisions" / "CONFIRMATION_PLAN.md").write_text(PLAN, encoding="utf-8", newline="\n")
    dd = root / "evaluation" / "discovery" / "decisions"
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "plan_state.json").write_text(json.dumps({"v5_batched_check": "passed_after_recolour",
                                                    "v1_tenfold_check": "failed", "guard_mode": {},
                                                    "freezing_candidates": [], "notes": {}}, indent=1),
                                        encoding="utf-8")
    (dd / "decisions.json").write_text(json.dumps({"S1_components": {"S1c_selection": {
        "min_delta": -0.016846876701018858, "min_delta_threshold": -0.02, "min_delta_yardstick": "HEAVIER",
        "min_delta_margin_inside_threshold": 0.00315, "min_delta_interval_separated_from_threshold": False}}},
        indent=1), encoding="utf-8")


def relax_strict_variant() -> None:
    """The rehearsal's ONE departure from the registered inner designs, applied in the parent (the ``Rehearsal``
    fixture, undone by monkeypatch) and in every worker (:func:`init`): the STRICT setting's thresholds -- 20 rows, 2
    publications, 5 partners -- cannot be met on 6 systems x 18 rows in one publication each, so its inner design is
    empty and every strict fold would raise.  The strict variant takes the PRIMARY thresholds here.  This exercises the
    strict jobs' PATHS (the seed-104729 refit pass on the registered multi-seed file, the exact comparator legs, the R19
    item 6 assembly), not the strict eligibility rule, which discovery ran on the real corpus (8 strict confirmation
    folds per seed).  The variant's medium, component rule and name are unchanged."""
    import dataclasses

    from gen19ct.folds import cell_holdout as CH

    CH.VARIANTS["strict"] = dataclasses.replace(CH.VARIANTS["strict"], thresholds=CH.PRIMARY)


def gate_digests() -> dict[str, Any]:
    return {"footer": D.REGISTERED_PREREG_SHA256, "recomputed": D.REGISTERED_PREREG_SHA256,
            "digest_file": D.REGISTERED_PREREG_SHA256, "n_addenda": 8, "addenda_sha256": "a" * 64}


def init(store_path: str, out_root: str) -> None:
    """The worker initializer of the rehearsal (``fit_loop`` hands it to the ``ProcessPoolExecutor`` in place of
    ``g19_run_confirmation._init_conf_worker``): the same contract -- torch threads, the seed store loaded and VERIFIED
    in the worker, the runners -- with the SYNTHETIC corpus of ``G19_REHEARSAL_CORPUS`` instead of the real one and
    the rehearsal's registry file (``G19_REHEARSAL_REGISTRY``) in place of the repository's, which is what the parent
    test monkeypatches in process and cannot reach across a process boundary."""
    import torch

    from gen19ct.evaluation import registry as REG

    torch.set_num_threads(2)
    relax_strict_variant()                                      # the same departure the parent applies (documented there)
    reg = Path(os.environ["G19_REHEARSAL_REGISTRY"])
    REG.registry_path = lambda root=None, _p=reg: _p           # this process only
    r = rc()
    sealm = r.seal_module()
    store = CF.load_seed_store(store_path, root=Path(out_root), seal=sealm,
                               prereg_paths=sealm.PreregPaths(root=Path(out_root), repo_root=paths.REPO_ROOT))
    if not store.verified:
        raise SystemExit("refused: a worker's seed store does not verify against the commitment")
    corpus, attrs = build_corpus(Path(os.environ["G19_REHEARSAL_CORPUS"]))
    r.CW.store = store
    r.CW.corpus = corpus
    r.CW.runners = r.runner_module().default_runners()
    r.CW.attached = None
    r.CW.pair_attrs = attrs

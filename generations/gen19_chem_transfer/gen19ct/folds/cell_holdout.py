"""``folds/cell_holdout.py`` -- V5 missing metal x extractant cell and its registered variants
(pre-registration sections 2, 3.1, 7).

Cell = ``(g19_metal_state, extractant_system_key)`` over MODEL rows.  Eligibility (:func:`eligible_cells`, the
``feasibility_v5_cells.csv`` rule, publication basis ``g19_publication_id``): >= k rows, >= p publications, the
metal keeps >= m other systems and the system >= m other metal states, and the cell edge is not a bridge of the
all-edge known-state bipartite graph.

Hiding (``support_graph.hide_cell``): the registered state-level rule -- the cell's rows and the element's X(?)
rows in the cell's own system and, with ``component_aware``, the same in every system sharing a component
(identical canonical SMILES, or the stereo-free parent structure for the ``parent_structure`` variant).  Other
known states of the element stay in training.

Variants (:data:`VARIANTS`): ``primary`` (k10 p1 m3), ``loose`` (5 1 2), ``strict`` (20 2 5), ``hno3_only``
(primary thresholds on HNO3 rows alone; the cell is hidden whole whatever the medium and its HNO3 rows are
scored), ``cell_only`` (no component-sharing hiding) and ``parent_structure``.

Schemes
* ``exact`` -- one fold per eligible cell.  A cell with any row in ``V6_TARGET_ROWS`` is carved out: hidden,
  never scored (``meta.carved_out``).
* ``batched`` -- the heavy arms: per discovery seed and per half (section 3.1 "batching applies within a half"),
  the scored cells are coloured greedily on the conflict graph (two cells conflict when they share a metal state
  or a system), vertex order drawn from the seed; every eligibility condition is re-checked for each cell with
  the batch's other cells hidden (:meth:`EligibilitySpace.check`), and a failing cell moves to a singleton batch.
* :func:`v5p_folds` -- V5-P: for each scored cell of the V5-P base setting (k10 p2 m3) and each of its
  publication groups (``group_cross_publication_copy``), a fold hiding the cell (registered rule) and every row
  of that group, scoring the cell's rows of that group.  A cell stays eligible only if EVERY one of its
  group-masked training sets keeps >= 3 other metal states for its system and >= 3 other systems for its metal.
* :func:`v5pair_folds` -- V5-PAIR: unordered pairs of primary-eligible cells under one system with >= 3
  comparable row pairs (same publication group, same system, identical ``normalize.condition_key``, different
  metal state), both cells hidden, eligibility re-checked for each with the other hidden, pairs touching
  ``V6_TARGET_ROWS`` excluded; the comparable test-test row pairs are generated from the fold's scored rows
  only, after the fold is formed, and pass ``pair_isolation_check``.
* :func:`v5p_batched_folds` / :func:`v5pair_batched_folds` -- the heavy-arm batches of V5-P and V5-PAIR (section 3.1
  resolution of 2026-09-15; seed 104729): the unbatched folds above are the units, coloured on their conflict graph
  (V5-P: shared metal state, system or publication group; V5-PAIR: shared metal state or system) inside each half,
  re-checked with every other unit of the batch hidden, failures to singleton batches.
* ``max_cells_per_batch`` (:func:`batch_cells`, :func:`v5_batched_folds`, :func:`inner_cells_V5`) -- the section 7
  compute-plan item 6 re-colouring (at most k cells per batch, same seeds); ``None`` is the registered rule.
* :func:`inner_cells_V5` -- section 7 inner cells of an outer training set.
"""
from __future__ import annotations

import hashlib
import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from gen19ct.chemistry import support_graph as SG
from gen19ct.data import leakage as L
from gen19ct.folds import io as FI
from gen19ct.folds import registered as FR

DESIGN = "V5"
PUB_BASIS = "g19_publication_id"
HNO3 = "HNO3"
HALF_CODE = {"S": 1, "C": 2}
V5P_MIN_OTHER = 3
V5P_KEEP_RULE = {"min_cells": 30, "min_systems": 5}
V5PAIR_MIN_COMPARABLE_PAIRS = 3
V5PAIR_KEEP_RULE = {"min_cell_pairs": 30, "min_systems": 10}
INNER_N_FOLDS = 3
INNER_MAX_CELLS = 30

Cell = tuple[str, str]


@dataclass(frozen=True)
class Thresholds:
    k: int
    p: int
    m: int

    @property
    def tag(self) -> str:
        return f"k{self.k}_p{self.p}_m{self.m}"


PRIMARY, LOOSE, STRICT, V5P_BASE = Thresholds(10, 1, 3), Thresholds(5, 1, 2), Thresholds(20, 2, 5), Thresholds(10, 2, 3)


@dataclass(frozen=True)
class Variant:
    name: str
    thresholds: Thresholds
    medium: str = "all"
    component_aware: bool = True
    parent_structure: bool = False

    def feasibility_column(self) -> str:
        pre = "eligible_hno3" if self.medium == HNO3 else "eligible"
        return f"{pre}__{PUB_BASIS}__{self.thresholds.tag}"


VARIANTS: dict[str, Variant] = {
    "primary": Variant("primary", PRIMARY),
    "loose": Variant("loose", LOOSE),
    "strict": Variant("strict", STRICT),
    "hno3_only": Variant("hno3_only", PRIMARY, medium=HNO3),
    "cell_only": Variant("cell_only", PRIMARY, component_aware=False),
    "parent_structure": Variant("parent_structure", PRIMARY, parent_structure=True),
}


# --------------------------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------------------------- #

def cell_label(cell: Cell) -> str:
    """``"Nd(III) x <system key>"`` (SMILES never contain a space)."""
    return f"{cell[0]} x {cell[1]}"


def parse_cell_label(label: str) -> Cell:
    m, s = str(label).split(" x ", 1)
    return m, s


def cell_fold_id(cell: Cell) -> str:
    return f"{cell[0]}__{hashlib.sha1(cell[1].encode()).hexdigest()[:10]}"


def category_class(state_a: str, state_b: str) -> str:
    """V5-PAIR category class of two metal states: ``Ln-Ln``, ``An-Ln``, ``An-An`` or ``other``."""
    cats = sorted(SG.metal_properties(s)["category"] or "none" for s in (state_a, state_b))
    if cats == ["lanthanide", "lanthanide"]:
        return "Ln-Ln"
    if cats == ["actinide", "lanthanide"]:
        return "An-Ln"
    if cats == ["actinide", "actinide"]:
        return "An-An"
    return "other"


def component_map_for(variant: Variant) -> dict[str, str] | None:
    return FR.parent_component_map() if variant.parent_structure else None


def medium_mask(frame: pd.DataFrame, medium: str) -> np.ndarray:
    if medium == "all":
        return np.ones(len(frame), dtype=bool)
    if medium == HNO3:
        return (frame["acid_primary"] == HNO3).to_numpy(dtype=bool)
    raise ValueError(f"unknown medium {medium!r}")


def cell_rows_mask(frame: pd.DataFrame, cell: Cell) -> np.ndarray:
    return ((frame[SG.METAL_COL] == cell[0]) & (frame[SG.SYSTEM_COL] == cell[1])).to_numpy(dtype=bool)


# --------------------------------------------------------------------------------------------- #
# eligibility
# --------------------------------------------------------------------------------------------- #

def eligible_cells(frame: pd.DataFrame, thr: Thresholds, medium: str = "all", pub_col: str = PUB_BASIS) -> pd.DataFrame:
    """Every known-state cell of ``frame`` (restricted to ``medium`` rows) with the eligibility inputs and the
    ``eligible`` flag -- the ``g19_feasibility.v5_block`` rule, recomputable on any (outer training) frame."""
    fr = frame[medium_mask(frame, medium)]
    known = fr[fr[SG.METAL_COL].notna()]
    cols = ["metal_state", "extractant_system_key", "n_rows", "n_publications", "other_systems_for_metal",
            "other_metal_states_for_system", "connected_without_cell", "eligible"]
    if known.empty:
        return pd.DataFrame(columns=cols)
    cells = known.groupby([SG.METAL_COL, SG.SYSTEM_COL], sort=True).agg(
        n_rows=(pub_col, "size"), n_publications=(pub_col, "nunique")).reset_index()
    cells = cells.rename(columns={SG.METAL_COL: "metal_state", SG.SYSTEM_COL: "extractant_system_key"})
    state_nsys = known.groupby(SG.METAL_COL)[SG.SYSTEM_COL].nunique()
    sys_nstate = known.groupby(SG.SYSTEM_COL)[SG.METAL_COL].nunique()
    bridges = SG.bridge_set(SG.bipartite_graph(fr))
    cells["other_systems_for_metal"] = cells["metal_state"].map(state_nsys).astype(int) - 1
    cells["other_metal_states_for_system"] = cells["extractant_system_key"].map(sys_nstate).astype(int) - 1
    cells["connected_without_cell"] = [
        frozenset({SG.node_id("metal_state", m), SG.node_id("extractant_system", s)}) not in bridges
        for m, s in zip(cells["metal_state"], cells["extractant_system_key"])]
    cells["eligible"] = ((cells["n_rows"] >= thr.k) & (cells["n_publications"] >= thr.p)
                         & (cells["other_systems_for_metal"] >= thr.m) & (cells["other_metal_states_for_system"] >= thr.m)
                         & cells["connected_without_cell"])
    return cells[cols]


class EligibilitySpace:
    """Integer-coded known-state rows of one frame (and medium) for fast eligibility re-checks under a ``keep``
    mask (``True`` = the row stays in training).  :meth:`check` applies the :func:`eligible_cells` rule to one cell
    on ``frame[keep]``: its own rows are counted as they stand, and 'after the cell is hidden' is the -1 on the
    partner counts and the removal of the cell edge before the connectivity test, exactly as in the full rule."""

    def __init__(self, frame: pd.DataFrame, medium: str = "all", pub_col: str = PUB_BASIS):
        self.n = len(frame)
        sel = frame[SG.METAL_COL].notna().to_numpy(dtype=bool) & medium_mask(frame, medium)
        self.sel = sel
        self.sc, self.states = self._codes(frame[SG.METAL_COL], sel)
        self.yc, self.systems = self._codes(frame[SG.SYSTEM_COL], sel)
        self.pc, _ = self._codes(frame[pub_col], sel)
        self.state_code = {v: i for i, v in enumerate(self.states)}
        self.sys_code = {v: i for i, v in enumerate(self.systems)}

    @staticmethod
    def _codes(col: pd.Series, sel: np.ndarray) -> tuple[np.ndarray, list]:
        vals = col.to_numpy(dtype=object)
        uniq = sorted({v for v, s in zip(vals, sel) if s})
        lut = {v: i for i, v in enumerate(uniq)}
        return np.array([lut[v] if s else -1 for v, s in zip(vals, sel)], dtype=np.int64), uniq

    def _use(self, keep: np.ndarray | None) -> np.ndarray:
        return self.sel if keep is None else (self.sel & np.asarray(keep, dtype=bool))

    def partners(self, cell: Cell, keep: np.ndarray | None = None) -> tuple[int, int]:
        """(other systems of the metal, other metal states of the system) among the kept known rows."""
        m, s = self.state_code.get(cell[0], -2), self.sys_code.get(cell[1], -2)
        use = self._use(keep)
        sc, yc = self.sc[use], self.yc[use]
        o_sys = np.unique(yc[sc == m])
        o_st = np.unique(sc[yc == s])
        return int((o_sys != s).sum()), int((o_st != m).sum())

    def check(self, cell: Cell, thr: Thresholds, keep: np.ndarray | None = None) -> dict:
        m, s = self.state_code.get(cell[0], -2), self.sys_code.get(cell[1], -2)
        use = self._use(keep)
        sc, yc, pc = self.sc[use], self.yc[use], self.pc[use]
        own = (sc == m) & (yc == s)
        n_rows, n_pubs = int(own.sum()), int(np.unique(pc[own]).size)
        o_sys, o_st = self.partners(cell, keep)
        connected = False
        if n_rows and m >= 0 and s >= 0:
            ns, ny = len(self.states), len(self.systems)
            e = np.unique(sc * ny + yc)
            e = e[e != m * ny + s]
            a, b = e // ny, ns + e % ny
            g = coo_matrix((np.ones(len(e)), (a, b)), shape=(ns + ny, ns + ny))
            _, lab = connected_components(g, directed=False)
            connected = bool(lab[m] == lab[ns + s])
        ok = n_rows >= thr.k and n_pubs >= thr.p and o_sys >= thr.m and o_st >= thr.m and connected
        return {"n_rows": n_rows, "n_publications": n_pubs, "other_systems_for_metal": o_sys,
                "other_metal_states_for_system": o_st, "connected_without_cell": connected, "eligible": bool(ok)}


# --------------------------------------------------------------------------------------------- #
# hiding
# --------------------------------------------------------------------------------------------- #

def hidden_mask(frame: pd.DataFrame, cells: Sequence[Cell], variant: Variant,
                component_map: Mapping[str, str] | None = None) -> np.ndarray:
    """Rows the registered hiding of ``variant`` removes for ``cells`` (``support_graph.hide_cells``)."""
    kept = SG.hide_cells(frame, list(cells), component_aware=variant.component_aware, component_map=component_map)
    return ~frame.index.isin(kept.index)


def cell_masks(frame: pd.DataFrame, cells: Iterable[Cell], variant: Variant,
               component_map: Mapping[str, str] | None = None) -> dict[Cell, np.ndarray]:
    return {c: hidden_mask(frame, [c], variant, component_map) for c in cells}


# --------------------------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------------------------- #

def conflict_neighbours(cells: Sequence[Cell]) -> dict[Cell, set[Cell]]:
    """Two cells conflict when they share a metal state or a system."""
    by_state, by_sys = defaultdict(set), defaultdict(set)
    for c in cells:
        by_state[c[0]].add(c)
        by_sys[c[1]].add(c)
    return {c: (by_state[c[0]] | by_sys[c[1]]) - {c} for c in cells}


def cell_conflict_keys(cell: Cell) -> frozenset:
    """The V5 conflict keys of a cell: its metal state and its system."""
    return frozenset({("metal_state", cell[0]), ("system", cell[1])})


def greedy_colouring_keyed(units: Sequence, keys: Mapping, rng: np.random.Generator,
                           max_size: int | None = None) -> list[list]:
    """Greedy colouring of a conflict graph given by conflict keys: two units conflict when their key sets
    (``keys[unit]``) intersect.  Vertex order = a seeded permutation (``rng``) of the sorted units; each vertex takes
    the smallest colour that no already-coloured neighbour holds and, when ``max_size`` is given, that holds fewer
    than ``max_size`` units.  Classes are returned in colour order, each sorted.  ``max_size=None`` is the
    registered section 3.1 rule."""
    if max_size is not None and int(max_size) < 1:
        raise ValueError(f"max_size must be >= 1, got {max_size}")
    base = sorted(set(units))
    by_key: dict = defaultdict(set)
    for u in base:
        for k in keys[u]:
            by_key[k].add(u)
    colour: dict = {}
    size: Counter = Counter()
    for i in rng.permutation(len(base)):
        u = base[i]
        used = {colour[n] for k in keys[u] for n in by_key[k] if n != u and n in colour}
        c = 0
        while c in used or (max_size is not None and size[c] >= int(max_size)):
            c += 1
        colour[u] = c
        size[c] += 1
    n_col = max(colour.values()) + 1 if colour else 0
    return [sorted(u for u in base if colour[u] == c) for c in range(n_col)]


def greedy_colouring(cells: Sequence[Cell], rng: np.random.Generator,
                     max_cells_per_batch: int | None = None) -> list[list[Cell]]:
    """Colour classes of a greedy colouring of the V5 conflict graph (shared metal state or system); vertex order =
    a seeded permutation of the sorted cells.  Each vertex takes the smallest colour unused by its already-coloured
    neighbours (and, with ``max_cells_per_batch``, holding fewer cells than the cap: the section 7 compute-plan item
    6 re-colouring).  Classes are returned in colour order, each sorted."""
    base = sorted(set(cells))
    return greedy_colouring_keyed(base, {c: cell_conflict_keys(c) for c in base}, rng, max_cells_per_batch)


def batch_units(units: Sequence, keys: Mapping, failing, rng: np.random.Generator, *,
                max_size: int | None = None, singleton_ok=None, label=str) -> tuple[list[list], dict]:
    """The section 3.1 batching loop for any unit type.

    Greedy colour classes (:func:`greedy_colouring_keyed`); inside each class ``failing(batch)`` returns the units
    that fail their eligibility re-check with every other unit of ``batch`` hidden; the failures move to singleton
    batches and the rest is re-checked until no unit fails (hiding fewer units only returns rows to training, so
    the loop is monotone).  Singleton batches follow the colour classes, in sorted order; each must pass
    ``singleton_ok(unit)`` (default: ``failing([unit])`` is empty)."""
    classes = greedy_colouring_keyed(units, keys, rng, max_size)
    out, singles, rechecks = [], [], 0
    for cls in classes:
        cur = list(cls)
        while len(cur) > 1:
            rechecks += len(cur)
            fails = list(failing(cur))
            if not fails:
                break
            singles.extend(fails)
            cur = [u for u in cur if u not in fails]
        if cur:                                   # every unit of the class may have failed
            out.append(cur)
    ok = singleton_ok if singleton_ok is not None else (lambda u: not list(failing([u])))
    for u in sorted(singles):
        if not ok(u):
            raise AssertionError(f"singleton batch {label(u)} is not eligible on its own")
        out.append([u])
    stats = {"n_units": len(set(units)), "n_colour_classes": len(classes), "n_batches": len(out),
             "n_units_moved_to_singleton": len(singles), "n_eligibility_rechecks": rechecks,
             "largest_batch": max((len(b) for b in out), default=0), "max_units_per_batch": max_size}
    return out, stats


def batch_cells(space: EligibilitySpace, cells: Sequence[Cell], thr: Thresholds, masks: Mapping[Cell, np.ndarray],
                rng: np.random.Generator, max_cells_per_batch: int | None = None) -> tuple[list[list[Cell]], dict]:
    """Greedy colour classes with the eligibility re-check; failing cells become singleton batches (appended after
    the colour classes, in sorted order).  Removing a cell only returns rows to training, so the loop is
    monotone; it repeats until no cell of a batch fails.  ``max_cells_per_batch`` caps the colour classes (section 7
    compute-plan item 6); ``None`` is the registered rule."""
    base = sorted(set(cells))

    def failing(cur: list[Cell]) -> list[Cell]:
        total = np.sum([masks[c].astype(np.int32) for c in cur], axis=0)
        return [c for c in cur if not space.check(c, thr, keep=(total - masks[c]) == 0)["eligible"]]

    out, st = batch_units(base, {c: cell_conflict_keys(c) for c in base}, failing, rng, max_size=max_cells_per_batch,
                          singleton_ok=lambda c: space.check(c, thr)["eligible"], label=cell_label)
    stats = {"n_cells": st["n_units"], "n_colour_classes": st["n_colour_classes"], "n_batches": st["n_batches"],
             "n_cells_moved_to_singleton": st["n_units_moved_to_singleton"],
             "n_eligibility_rechecks": st["n_eligibility_rechecks"], "largest_batch": st["largest_batch"],
             "max_cells_per_batch": max_cells_per_batch}
    return out, stats


# --------------------------------------------------------------------------------------------- #
# fold construction
# --------------------------------------------------------------------------------------------- #

def _row_units(ids: np.ndarray, frame: pd.DataFrame, cells: Sequence[Cell], masks: Mapping[Cell, np.ndarray],
               hidden: np.ndarray) -> dict[str, str]:
    """Each hidden row's unit: its own cell when it is a cell row, else the first (sorted) cell whose hiding
    removed it."""
    unit = np.full(len(ids), "", dtype=object)
    for c in sorted(cells):
        own = cell_rows_mask(frame, c) & hidden
        unit[own] = cell_label(c)
    for c in sorted(cells):
        rest = masks[c] & (unit == "")
        unit[rest] = cell_label(c)
    return {r: u for r, u, h in zip(ids, unit, hidden) if h}


def _cell_fold(frame: pd.DataFrame, ids: np.ndarray, variant: Variant, cells: Sequence[Cell],
               masks: Mapping[Cell, np.ndarray], scorable: np.ndarray, v6: np.ndarray, halves: Mapping[str, str], *,
               scheme: str, fold_id: str, seed: int | None, batch_id: str | None, design: str = DESIGN,
               extra_meta: Mapping | None = None, check_hide: bool = True, component_map=None) -> FI.Fold:
    cells = sorted(cells)
    hidden = np.logical_or.reduce([masks[c] for c in cells])
    if check_hide:
        again = hidden_mask(frame, cells, variant, component_map)
        if not np.array_equal(again, hidden):
            raise AssertionError(f"{fold_id}: union of per-cell hiding differs from hide_cells")
    med = medium_mask(frame, variant.medium)
    scored = np.zeros(len(ids), dtype=bool)
    carved = []
    for c in cells:
        own = cell_rows_mask(frame, c)
        if (own & v6).any():
            carved.append(cell_label(c))
            continue
        scored |= own & med & scorable
    rh = {c: halves.get(c[1], "NA") for c in cells}
    fold_half = next(iter(set(rh.values()))) if len(set(rh.values())) == 1 else "NA"
    row_unit = _row_units(ids, frame, cells, masks, hidden)
    by_label = {cell_label(c): rh[c] for c in cells}
    meta = {"cells": [list(c) for c in cells], "carved_out_cells": carved, "carved_out": bool(carved) and len(carved) == len(cells),
            "medium_scored": variant.medium, "component_aware": variant.component_aware,
            "parent_structure": variant.parent_structure, **dict(extra_meta or {})}
    return FI.make_fold(design=design, variant=variant.name, scheme=scheme, fold_id=fold_id, half=fold_half, seed=seed,
                        hidden=ids[hidden].tolist(), scored=ids[scored].tolist(), unit_type="cell",
                        units=[cell_label(c) for c in cells], batch_id=batch_id, row_unit=row_unit,
                        row_half={r: by_label.get(u, "NA") for r, u in row_unit.items()}, meta=meta)


def v5_exact_folds(frame: pd.DataFrame, variant: Variant, cells: Sequence[Cell], masks: Mapping[Cell, np.ndarray],
                   v6_mask: pd.Series, halves: Mapping[str, str], component_map=None) -> list[FI.Fold]:
    ids = frame[FI.ROW_ID].astype(str).to_numpy(dtype=object)
    scorable = FI.scorable_mask(frame, v6_mask).to_numpy(dtype=bool)
    v6 = v6_mask.reindex(frame.index).fillna(False).to_numpy(dtype=bool)
    return [_cell_fold(frame, ids, variant, [c], masks, scorable, v6, halves, scheme="exact", fold_id=cell_fold_id(c),
                       seed=None, batch_id=None, component_map=component_map) for c in sorted(cells)]


def batched_scheme(max_cells_per_batch: int | None = None) -> str:
    """``"batched"`` for the registered rule, ``"batched_max<k>"`` for the capped re-colouring."""
    return "batched" if max_cells_per_batch is None else f"batched_max{int(max_cells_per_batch)}"


def v5_batched_folds(frame: pd.DataFrame, space: EligibilitySpace, variant: Variant, cells: Sequence[Cell],
                     masks: Mapping[Cell, np.ndarray], v6_mask: pd.Series, halves: Mapping[str, str], seed: int,
                     component_map=None, max_cells_per_batch: int | None = None) -> tuple[list[FI.Fold], dict]:
    """Batched folds of one discovery seed; ``cells`` are the variant's SCORED cells (carved-out cells are not
    batched).  Batches are formed separately inside each half.

    ``max_cells_per_batch`` (default ``None`` = the registered section 3.1 rule, unchanged) caps every colour class
    at that many cells -- the section 7 compute-plan item 6 re-colouring (at most 4 cells per batch, same seeds and
    streams) used only if the B6 batched-vs-exact check fails.  The capped folds carry scheme
    ``batched_max<k>`` and ``meta.max_cells_per_batch``; the eligibility re-check is the same."""
    ids = frame[FI.ROW_ID].astype(str).to_numpy(dtype=object)
    scorable = FI.scorable_mask(frame, v6_mask).to_numpy(dtype=bool)
    v6 = v6_mask.reindex(frame.index).fillna(False).to_numpy(dtype=bool)
    scheme = batched_scheme(max_cells_per_batch)
    folds, stats = [], {}
    for half in ("S", "C"):
        hc = sorted(c for c in cells if halves.get(c[1]) == half)
        if not hc:
            continue
        batches, st = batch_cells(space, hc, variant.thresholds, masks, FI.seed_rng(seed, "V5_batch", HALF_CODE[half]),
                                  max_cells_per_batch=max_cells_per_batch)
        stats[half] = st
        for j, b in enumerate(batches):
            bid = f"s{seed}_{half}_b{j:03d}"
            extra = {"n_cells": len(b)}
            if max_cells_per_batch is not None:
                extra["max_cells_per_batch"] = int(max_cells_per_batch)
            folds.append(_cell_fold(frame, ids, variant, b, masks, scorable, v6, halves, scheme=scheme, fold_id=bid,
                                    seed=seed, batch_id=bid, component_map=component_map, extra_meta=extra))
    stats["n_batches"] = int(sum(s["n_batches"] for s in stats.values() if isinstance(s, dict)))
    return folds, stats


def guard_v5(fold: FI.Fold, frame: pd.DataFrame, component_map: Mapping[str, str] | None = None,
             universe_index: pd.Index | None = None) -> dict:
    """``fold_isolation_check(level="V5")`` with the variant's component rule; test = every row of the fold's
    cells (all media)."""
    cells = [tuple(c) for c in fold.meta["cells"]]
    te = np.logical_or.reduce([cell_rows_mask(frame, c) for c in cells])
    return FI.guard(fold, frame, frame.index if universe_index is None else universe_index,
                    frame.loc[te, FI.ROW_ID].astype(str).tolist(), "V5", component_aware=bool(fold.meta["component_aware"]),
                    component_map=component_map, near_dup_value_tol=FI.NEAR_DUP_VALUE_TOL)


# --------------------------------------------------------------------------------------------- #
# V5-P
# --------------------------------------------------------------------------------------------- #

def v5p_folds(frame: pd.DataFrame, groups: pd.Series, space: EligibilitySpace, cells: Sequence[Cell],
              masks: Mapping[Cell, np.ndarray], v6_mask: pd.Series, halves: Mapping[str, str],
              min_other: int = V5P_MIN_OTHER) -> tuple[list[FI.Fold], pd.DataFrame, dict]:
    """V5-P folds for the base setting's scored ``cells`` (``masks`` = registered component-aware hiding).
    Returns the folds of the eligible cells, the per-(cell, group) eligibility table and the counts."""
    ids = frame[FI.ROW_ID].astype(str).to_numpy(dtype=object)
    scorable = FI.scorable_mask(frame, v6_mask).to_numpy(dtype=bool)
    g = groups.reindex(frame.index).astype(str).to_numpy(dtype=object)
    variant = VARIANTS["primary"]
    recs, folds = [], []
    for c in sorted(cells):
        own = cell_rows_mask(frame, c)
        cg = sorted(set(g[own]))
        per = []
        for grp in cg:
            hid = masks[c] | (g == grp)
            o_sys, o_st = space.partners(c, keep=~hid)
            ok = o_sys >= min_other and o_st >= min_other
            per.append((grp, hid, ok))
            recs.append({"metal_state": c[0], "extractant_system_key": c[1], "publication_group": grp,
                         "half": halves.get(c[1], "NA"), "n_cell_rows_in_group": int((own & (g == grp)).sum()),
                         "n_hidden_rows": int(hid.sum()), "other_systems_for_metal_after_masking": o_sys,
                         "other_metal_states_for_system_after_masking": o_st, "fold_passes": ok})
        cell_ok = all(ok for _, _, ok in per)
        for r in recs[-len(per):]:
            r["cell_eligible_all_groups"] = cell_ok
            r["n_groups_of_cell"] = len(per)
        if not cell_ok:
            continue
        for grp, hid, _ in per:
            scored = own & (g == grp) & scorable
            label = cell_label(c)
            folds.append(FI.make_fold(
                design="V5P", variant="base", scheme="cell_x_group", fold_id=f"{cell_fold_id(c)}__{grp}",
                half=halves.get(c[1], "NA"), seed=None, hidden=ids[hid].tolist(), scored=ids[scored].tolist(),
                unit_type="cell", units=[label], row_unit={r: label for r in ids[hid]},
                meta={"cells": [list(c)], "publication_group": grp, "n_groups_of_cell": len(per),
                      "component_aware": True, "parent_structure": False, "medium_scored": "all",
                      "carved_out": False, "carved_out_cells": []}))
    tab = pd.DataFrame(recs)
    elig = tab[tab["cell_eligible_all_groups"]] if len(tab) else tab
    lenient = tab[tab["fold_passes"]] if len(tab) else tab
    n_cells = int(elig[["metal_state", "extractant_system_key"]].drop_duplicates().shape[0]) if len(elig) else 0
    n_sys = int(elig["extractant_system_key"].nunique()) if len(elig) else 0
    counts = {
        "setting": V5P_BASE.tag, "rule": f"every group-masked training set keeps >= {min_other} other metal states for "
                                         f"the system and >= {min_other} other systems for the metal",
        "n_base_scored_cells": len(set(cells)), "n_cell_group_folds_considered": int(len(tab)),
        "n_eligible_cells": n_cells, "n_eligible_systems": n_sys, "n_folds": len(folds),
        "n_eligible_cells_by_half": {h: int(elig.loc[elig["half"] == h, ["metal_state", "extractant_system_key"]]
                                            .drop_duplicates().shape[0]) for h in ("S", "C")} if len(elig) else {},
        "lenient_reading_n_cells_with_any_passing_group": int(lenient[["metal_state", "extractant_system_key"]]
                                                              .drop_duplicates().shape[0]) if len(lenient) else 0,
        "keep_rule": dict(V5P_KEEP_RULE),
        "retained": bool(n_cells >= V5P_KEEP_RULE["min_cells"] and n_sys >= V5P_KEEP_RULE["min_systems"]),
    }
    return folds, tab, counts


def guard_v5p(fold: FI.Fold, frame: pd.DataFrame) -> list[dict]:
    """The V5 component-aware check on the cell's rows and the V1 check on the scored (cell x group) rows.

    The V1 check reads the registered publication group (``group_cross_publication_copy``, the unit V5-P masks), not
    ``g19_publication_id`` (``io.publication_group_frame``): with the raw id a training row of a scored row's copy group
    that carries a different id went undetected (task X, leakage finding VR-02)."""
    reps = [guard_v5(fold, frame)]
    if fold.scored_row_ids:
        reps.append(FI.guard(fold, frame, frame.index, fold.scored_row_ids, "V1", publication_col=FI.GROUP_COL,
                             near_dup_value_tol=FI.NEAR_DUP_VALUE_TOL))
    return reps


# --------------------------------------------------------------------------------------------- #
# V5-PAIR
# --------------------------------------------------------------------------------------------- #

def _comparable_pairs(ids: np.ndarray, rows: np.ndarray, states: np.ndarray, keys: np.ndarray) -> list[tuple[int, int]]:
    """Positional (i, j) comparable pairs among ``rows``: same key (publication group + system + condition key)
    and different metal state; i precedes j in position order."""
    by = defaultdict(list)
    for i in np.flatnonzero(rows):
        by[keys[i]].append(i)
    out = []
    for members in by.values():
        for i, j in itertools.combinations(sorted(members), 2):
            if states[i] != states[j]:
                out.append((i, j))
    return out


def v5pair_folds(frame: pd.DataFrame, groups: pd.Series, condition_keys: pd.Series, space: EligibilitySpace,
                 cells: Sequence[Cell], masks: Mapping[Cell, np.ndarray], v6_mask: pd.Series,
                 halves: Mapping[str, str], thr: Thresholds = PRIMARY,
                 min_pairs: int = V5PAIR_MIN_COMPARABLE_PAIRS) -> tuple[list[FI.Fold], pd.DataFrame, pd.DataFrame, dict]:
    """V5-PAIR folds over the primary-eligible ``cells``.  Returns (folds, candidate table, pair list, counts)."""
    ids = frame[FI.ROW_ID].astype(str).to_numpy(dtype=object)
    scorable = FI.scorable_mask(frame, v6_mask).to_numpy(dtype=bool)
    v6 = v6_mask.reindex(frame.index).fillna(False).to_numpy(dtype=bool)
    states = frame[SG.METAL_COL].to_numpy(dtype=object)
    g = groups.reindex(frame.index).astype(str).to_numpy(dtype=object)
    sysk = frame[SG.SYSTEM_COL].astype(str).to_numpy(dtype=object)
    ck = condition_keys.reindex(frame.index).astype(str).to_numpy(dtype=object)
    key_group = np.array([f"{a}\t{b}\t{c}" for a, b, c in zip(g, sysk, ck)], dtype=object)
    key_pub = np.array([f"{a}\t{b}\t{c}" for a, b, c in zip(frame[PUB_BASIS].astype(str), sysk, ck)], dtype=object)
    variant = VARIANTS["primary"]
    by_sys = defaultdict(list)
    for c in sorted(set(cells)):
        by_sys[c[1]].append(c)
    cand, pair_recs, folds = [], [], []
    for s in sorted(by_sys):
        for a, b in itertools.combinations(by_sys[s], 2):
            ra, rb = cell_rows_mask(frame, a), cell_rows_mask(frame, b)
            rows = (ra | rb) & scorable
            touches = bool(((ra | rb) & v6).any())
            n_pairs = len(_comparable_pairs(ids, rows, states, key_group))
            n_pairs_pub = len(_comparable_pairs(ids, rows, states, key_pub))
            rec = {"metal_state_a": a[0], "metal_state_b": b[0], "extractant_system_key": s,
                   "category_class": category_class(a[0], b[0]), "half": halves.get(s, "NA"),
                   "touches_v6_target_rows": touches, "n_comparable_row_pairs": n_pairs,
                   "n_comparable_row_pairs_g19_publication_id": n_pairs_pub,
                   "a_eligible_with_b_hidden": None, "b_eligible_with_a_hidden": None}
            if touches:
                rec["status"] = "excluded_v6_target_rows"
            elif n_pairs < min_pairs:
                rec["status"] = "fewer_than_min_comparable_pairs"
            else:
                ea = space.check(a, thr, keep=~masks[b])["eligible"]
                eb = space.check(b, thr, keep=~masks[a])["eligible"]
                rec.update(a_eligible_with_b_hidden=ea, b_eligible_with_a_hidden=eb)
                rec["status"] = "eligible" if (ea and eb) else "ineligible_after_recheck"
            cand.append(rec)
            if rec["status"] != "eligible":
                continue
            fid = f"{cell_fold_id(a)}__{cell_fold_id(b)}"
            f = _cell_fold(frame, ids, variant, [a, b], masks, scorable, v6, halves, scheme="cell_pair", fold_id=fid,
                           seed=None, batch_id=None, design="V5PAIR",
                           extra_meta={"category_class": rec["category_class"], "n_comparable_row_pairs": n_pairs,
                                       "cell_pair": f"{cell_label(a)} || {cell_label(b)}"})
            # pairs generated AFTER the fold is formed, from its scored rows only
            fs = set(f.scored_row_ids)
            in_fold = np.fromiter((r in fs for r in ids), dtype=bool, count=len(ids))
            got = _comparable_pairs(ids, in_fold, states, key_group)
            if len(got) != n_pairs:
                raise AssertionError(f"{fid}: {len(got)} test-test pairs after fold formation, expected {n_pairs}")
            for i, j in got:
                pair_recs.append({"fold_id": fid, "row_id_a": ids[i], "row_id_b": ids[j], "metal_state_a": states[i],
                                  "metal_state_b": states[j], "publication_group": g[i],
                                  "condition_key_id": "ck_" + hashlib.sha1(ck[i].encode()).hexdigest()[:16],
                                  "category_class": rec["category_class"]})
            folds.append(f)
    ctab = pd.DataFrame(cand)
    ptab = pd.DataFrame(pair_recs, columns=["fold_id", "row_id_a", "row_id_b", "metal_state_a", "metal_state_b",
                                            "publication_group", "condition_key_id", "category_class"])
    el = ctab[ctab["status"] == "eligible"] if len(ctab) else ctab
    n_sys = int(el["extractant_system_key"].nunique()) if len(el) else 0
    counts = {
        "definition": ("unordered pairs of primary-eligible cells under one system with >= "
                       f"{min_pairs} comparable row pairs (same group_cross_publication_copy, same system, identical "
                       "condition_key, different metal state) among scored rows; both cells hidden (registered "
                       "component-aware rule); each cell re-checked eligible with the other hidden; pairs touching "
                       "V6_TARGET_ROWS excluded"),
        "n_candidate_cell_pairs": int(len(ctab)),
        "status_counts": {k: int(v) for k, v in ctab["status"].value_counts().sort_index().items()} if len(ctab) else {},
        "n_eligible_cell_pairs": int(len(el)),
        "n_eligible_cell_pairs_by_class": {k: int((el["category_class"] == k).sum()) for k in ("Ln-Ln", "An-Ln", "An-An", "other")},
        "n_eligible_systems": n_sys,
        "n_eligible_systems_by_class": {k: int(el.loc[el["category_class"] == k, "extractant_system_key"].nunique())
                                        for k in ("Ln-Ln", "An-Ln", "An-An", "other")},
        "by_half": {h: {"n_cell_pairs": int((el["half"] == h).sum()),
                        "n_systems": int(el.loc[el["half"] == h, "extractant_system_key"].nunique())} for h in ("S", "C")},
        "n_test_test_row_pairs": int(len(ptab)),
        "keep_rule": dict(V5PAIR_KEEP_RULE),
        "retained": bool(len(el) >= V5PAIR_KEEP_RULE["min_cell_pairs"] and n_sys >= V5PAIR_KEEP_RULE["min_systems"]),
    }
    return folds, ctab, ptab, counts


def check_pair_isolation(fold: FI.Fold, pairs: pd.DataFrame, universe_ids: Iterable[str]) -> None:
    """``pair_isolation_check`` of one fold's pair list against the fold's role map (train / hidden_scored /
    hidden_unscored), and both members hidden-scored."""
    role = {r: "train" for r in universe_ids}
    role.update(fold.role_of())
    L.pair_isolation_check(pairs, pd.Series(role, dtype=object), member_cols=("row_id_a", "row_id_b"))
    bad = pairs[(pairs["row_id_a"].map(role) != "hidden_scored") | (pairs["row_id_b"].map(role) != "hidden_scored")]
    if len(bad):
        raise AssertionError(f"{fold.fold_id}: {len(bad)} pair(s) with a member that is not hidden-scored")


# --------------------------------------------------------------------------------------------- #
# batched V5-P and V5-PAIR (heavy arms; section 3.1 resolution of 2026-09-15)
# --------------------------------------------------------------------------------------------- #
#
# Units are the registered UNBATCHED folds (``V5P__base__cell_x_group``, ``V5PAIR__primary__cell_pair``), so a batch
# hides exactly the union of its units' hidden rows and scores exactly the union of their scored rows.  Batches are
# colour classes of a greedy colouring (vertex order from ``SeedSequence([seed, tag, half])``, seed 104729 only),
# formed inside each half; each unit is re-checked with every other unit of its batch hidden and a failing unit
# moves to a singleton batch (which is its unbatched fold).

HEAVY_BATCH_SEED = FI.DISCOVERY_SEEDS[0]


def _id_positions(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    ids = frame[FI.ROW_ID].astype(str).to_numpy(dtype=object)
    return ids, {r: i for i, r in enumerate(ids)}


def _id_mask(pos: Mapping[str, int], n: int, row_ids: Iterable[str]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    idx = [pos[r] for r in row_ids]
    m[idx] = True
    return m


def v5p_unit_keys(cell: Cell, group: str) -> frozenset:
    """V5-P conflict keys: a cell x publication-group unit conflicts with another that shares its metal state, its
    system or its publication group."""
    return frozenset({("metal_state", cell[0]), ("system", cell[1]), ("publication_group", str(group))})


def v5pair_unit_keys(cell_a: Cell, cell_b: Cell) -> frozenset:
    """V5-PAIR conflict keys: a cell pair conflicts with another that shares a metal state or the system."""
    return frozenset({("metal_state", cell_a[0]), ("metal_state", cell_b[0]), ("system", cell_a[1]),
                      ("system", cell_b[1])})


def v5p_batched_folds(frame: pd.DataFrame, groups: pd.Series, space: EligibilitySpace, units: Sequence[FI.Fold],
                      v6_mask: pd.Series, seed: int = HEAVY_BATCH_SEED, *, thr: Thresholds = V5P_BASE,
                      min_other: int = V5P_MIN_OTHER,
                      max_units_per_batch: int | None = None) -> tuple[list[FI.Fold], dict]:
    """Batched V5-P folds (heavy arms) from the unbatched cell x publication-group folds ``units``.

    Conflict: shared metal state, system or publication group (:func:`v5p_unit_keys`).  Re-check of unit ``u`` = (cell
    ``c``, group ``g``) in batch ``B`` -- the unbatched V5-P unit rule with every other unit of ``B`` hidden:

    (a) the V5-P base eligibility of ``c`` (``thr``: k, p, m and the non-bridge condition,
        :meth:`EligibilitySpace.check`) on MODEL rows minus the hidden rows of the OTHER units of ``B`` (the V5
        batched re-check);
    (b) the V5-P masking rule: on MODEL rows minus the hidden rows of EVERY unit of ``B`` (``c``'s registered hiding
        and group ``g`` included), the system of ``c`` keeps >= ``min_other`` other metal states and its metal
        >= ``min_other`` other systems.

    A unit failing (a) or (b) moves to a singleton batch, which is its unbatched fold.  Fold ids
    ``s<seed>_<half>_b<j>``; stream ``SeedSequence([seed, 6, half])``."""
    ids, pos = _id_positions(frame)
    n = len(ids)
    scorable = FI.scorable_mask(frame, v6_mask).to_numpy(dtype=bool)
    g = groups.reindex(frame.index).astype(str).to_numpy(dtype=object)
    if len({f.fold_id for f in units}) != len(units):
        raise ValueError("v5p_batched_folds: duplicate unit fold ids")
    cell_of: dict[str, Cell] = {}
    group_of: dict[str, str] = {}
    half_of: dict[str, str] = {}
    hid: dict[str, np.ndarray] = {}
    scd: dict[str, np.ndarray] = {}
    for f in units:
        if (f.design, f.scheme) != ("V5P", "cell_x_group") or len(f.meta["cells"]) != 1 or f.half not in ("S", "C"):
            raise ValueError(f"{f.fold_id}: not an unbatched V5-P unit fold")
        cell_of[f.fold_id] = tuple(f.meta["cells"][0])
        group_of[f.fold_id] = str(f.meta["publication_group"])
        half_of[f.fold_id] = f.half
        hid[f.fold_id] = _id_mask(pos, n, f.hidden_row_ids)
        scd[f.fold_id] = _id_mask(pos, n, f.scored_row_ids)
    variant = VARIANTS["primary"]
    folds, stats = [], {}
    for half in ("S", "C"):
        hu = sorted(u for u in cell_of if half_of[u] == half)
        if not hu:
            continue
        reasons: Counter = Counter()
        conditions: Counter = Counter()

        def failing(cur: list[str], record: bool = True) -> list[str]:
            total = np.sum([hid[u].astype(np.int32) for u in cur], axis=0)
            out = []
            for u in cur:
                c = cell_of[u]
                base = space.check(c, thr, keep=(total - hid[u]) == 0)
                o_sys, o_st = space.partners(c, keep=total == 0)
                mask_ok = o_sys >= min_other and o_st >= min_other
                if not (base["eligible"] and mask_ok):
                    out.append(u)
                    if record:
                        reasons["base_rule_failed_with_other_units_hidden"] += int(not base["eligible"])
                        reasons["masking_rule_failed_with_batch_hidden"] += int(not mask_ok)
                        conditions["k_rows"] += int(base["n_rows"] < thr.k)
                        conditions["p_publications"] += int(base["n_publications"] < thr.p)
                        conditions["m_other_systems_for_metal"] += int(base["other_systems_for_metal"] < thr.m)
                        conditions["m_other_metal_states_for_system"] += int(base["other_metal_states_for_system"] < thr.m)
                        conditions["non_bridge"] += int(not base["connected_without_cell"])
            return out

        batches, st = batch_units(hu, {u: v5p_unit_keys(cell_of[u], group_of[u]) for u in hu}, failing,
                                  FI.seed_rng(seed, "V5P_batch", HALF_CODE[half]), max_size=max_units_per_batch,
                                  singleton_ok=lambda u: not failing([u], record=False))
        st["recheck_failures_by_rule"] = {k: int(reasons.get(k, 0)) for k in (
            "base_rule_failed_with_other_units_hidden", "masking_rule_failed_with_batch_hidden")}
        st["base_rule_conditions_failed"] = {k: int(conditions.get(k, 0)) for k in (
            "k_rows", "p_publications", "m_other_systems_for_metal", "m_other_metal_states_for_system", "non_bridge")}
        stats[half] = st
        for j, b in enumerate(batches):
            bid = f"s{seed}_{half}_b{j:03d}"
            b = sorted(b, key=lambda u: (cell_of[u], u))
            cells = [cell_of[u] for u in b]
            if len(set(cells)) != len(cells):
                raise AssertionError(f"V5P {bid}: a cell twice in one batch")
            hidden = np.logical_or.reduce([hid[u] for u in b])
            scored = np.logical_or.reduce([scd[u] for u in b])
            again = hidden_mask(frame, cells, variant) | np.isin(g, sorted({group_of[u] for u in b}))
            if not np.array_equal(again, hidden):
                raise AssertionError(f"V5P {bid}: union of unit hiding differs from hide_cells + the batch's groups")
            want = np.logical_or.reduce([cell_rows_mask(frame, cell_of[u]) & (g == group_of[u]) & scorable for u in b])
            if not np.array_equal(want, scored):
                raise AssertionError(f"V5P {bid}: scored rows differ from the units' cell x group rows")
            row_unit = _row_units(ids, frame, cells, {cell_of[u]: hid[u] for u in b}, hidden)
            meta = {"cells": [list(c) for c in cells], "unit_fold_ids": list(b),
                    "publication_groups": [group_of[u] for u in b], "n_units": len(b), "component_aware": True,
                    "parent_structure": False, "medium_scored": "all", "carved_out": False, "carved_out_cells": []}
            if max_units_per_batch is not None:
                meta["max_units_per_batch"] = int(max_units_per_batch)
            folds.append(FI.make_fold(
                design="V5P", variant="base", scheme=batched_scheme(max_units_per_batch), fold_id=bid, half=half,
                seed=seed, hidden=ids[hidden].tolist(), scored=ids[scored].tolist(), unit_type="cell",
                units=[cell_label(c) for c in cells], batch_id=bid, row_unit=row_unit,
                row_half={r: half for r in row_unit}, meta=meta))
    stats["n_batches"] = int(sum(s["n_batches"] for s in stats.values() if isinstance(s, dict)))
    stats["n_units"] = int(sum(s["n_units"] for s in stats.values() if isinstance(s, dict)))
    return folds, stats


def v5pair_batched_folds(frame: pd.DataFrame, groups: pd.Series, condition_keys: pd.Series, space: EligibilitySpace,
                         units: Sequence[FI.Fold], unit_pairs: pd.DataFrame, masks: Mapping[Cell, np.ndarray],
                         v6_mask: pd.Series, halves: Mapping[str, str], seed: int = HEAVY_BATCH_SEED, *,
                         thr: Thresholds = PRIMARY,
                         max_units_per_batch: int | None = None) -> tuple[list[FI.Fold], pd.DataFrame, dict]:
    """Batched V5-PAIR folds (heavy arms) from the unbatched cell-pair folds ``units`` and their pair list.

    Conflict: shared metal state or system (:func:`v5pair_unit_keys`; two cell pairs of one system always conflict).
    Re-check: every cell of every unit is eligible under ``thr`` (:meth:`EligibilitySpace.check`) on MODEL rows minus
    the registered hiding (``masks``) of every OTHER cell of the batch -- its pair partner included, which is the
    unbatched rule when the batch is the unit alone.  A failing unit moves to a singleton batch.

    The comparable test-test pairs of a batch are regenerated from the batch fold's hidden-scored rows after the fold
    is formed and must equal the union of its units' pairs (units of one batch never share a system, so no pair
    crosses units).  Fold ids ``s<seed>_<half>_b<j>``; stream ``SeedSequence([seed, 7, half])``."""
    ids, pos = _id_positions(frame)
    n = len(ids)
    scorable = FI.scorable_mask(frame, v6_mask).to_numpy(dtype=bool)
    v6 = v6_mask.reindex(frame.index).fillna(False).to_numpy(dtype=bool)
    states = frame[SG.METAL_COL].to_numpy(dtype=object)
    g = groups.reindex(frame.index).astype(str).to_numpy(dtype=object)
    sysk = frame[SG.SYSTEM_COL].astype(str).to_numpy(dtype=object)
    ck = condition_keys.reindex(frame.index).astype(str).to_numpy(dtype=object)
    key_group = np.array([f"{a}\t{b}\t{c}" for a, b, c in zip(g, sysk, ck)], dtype=object)
    if len({f.fold_id for f in units}) != len(units):
        raise ValueError("v5pair_batched_folds: duplicate unit fold ids")
    cells_of: dict[str, tuple[Cell, Cell]] = {}
    unit_fold = {f.fold_id: f for f in units}
    for f in units:
        cs = [tuple(c) for c in f.meta["cells"]]
        if (f.design, f.scheme) != ("V5PAIR", "cell_pair") or len(cs) != 2 or cs[0][1] != cs[1][1] or f.half not in ("S", "C"):
            raise ValueError(f"{f.fold_id}: not an unbatched V5-PAIR unit fold")
        a, b = sorted(cs)
        if not np.array_equal(_id_mask(pos, n, f.hidden_row_ids), masks[a] | masks[b]):
            raise AssertionError(f"{f.fold_id}: unit hidden rows differ from the registered hiding of its two cells")
        cells_of[f.fold_id] = (a, b)
    pair_set = {fid: set(zip(p["row_id_a"].astype(str), p["row_id_b"].astype(str)))
                for fid, p in unit_pairs.groupby("fold_id", sort=False)}
    variant = VARIANTS["primary"]
    folds, recs, stats = [], [], {}
    for half in ("S", "C"):
        hu = sorted(u for u in cells_of if unit_fold[u].half == half)
        if not hu:
            continue

        def failing(cur: list[str]) -> list[str]:
            cs = [c for u in cur for c in cells_of[u]]
            total = np.sum([masks[c].astype(np.int32) for c in cs], axis=0)
            return [u for u in cur if not all(space.check(x, thr, keep=(total - masks[x]) == 0)["eligible"]
                                              for x in cells_of[u])]

        def alone_ok(u: str) -> bool:
            a, b = cells_of[u]
            return space.check(a, thr, keep=~masks[b])["eligible"] and space.check(b, thr, keep=~masks[a])["eligible"]

        batches, st = batch_units(hu, {u: v5pair_unit_keys(*cells_of[u]) for u in hu}, failing,
                                  FI.seed_rng(seed, "V5PAIR_batch", HALF_CODE[half]), max_size=max_units_per_batch,
                                  singleton_ok=alone_ok)
        stats[half] = st
        for j, b in enumerate(batches):
            bid = f"s{seed}_{half}_b{j:03d}"
            b = sorted(b)
            cells = [c for u in b for c in cells_of[u]]
            if len(set(cells)) != len(cells) or len({cells_of[u][0][1] for u in b}) != len(b):
                raise AssertionError(f"V5PAIR {bid}: two units share a cell or a system")
            extra = {"unit_fold_ids": list(b), "n_units": len(b),
                     "cell_pairs": [unit_fold[u].meta["cell_pair"] for u in b],
                     "category_classes": [unit_fold[u].meta["category_class"] for u in b],
                     "n_comparable_row_pairs_by_unit": [int(unit_fold[u].meta["n_comparable_row_pairs"]) for u in b],
                     "n_comparable_row_pairs": int(sum(unit_fold[u].meta["n_comparable_row_pairs"] for u in b))}
            if max_units_per_batch is not None:
                extra["max_units_per_batch"] = int(max_units_per_batch)
            f = _cell_fold(frame, ids, variant, cells, masks, scorable, v6, halves, scheme=batched_scheme(max_units_per_batch),
                           fold_id=bid, seed=seed, batch_id=bid, design="V5PAIR", extra_meta=extra)
            if f.meta["carved_out_cells"] or f.half != half:
                raise AssertionError(f"V5PAIR {bid}: carved-out cell or half mismatch")
            if set(f.hidden_row_ids) != set().union(*(unit_fold[u].hidden_row_ids for u in b)):
                raise AssertionError(f"V5PAIR {bid}: hidden rows differ from the union of its units")
            if set(f.scored_row_ids) != set().union(*(unit_fold[u].scored_row_ids for u in b)):
                raise AssertionError(f"V5PAIR {bid}: scored rows differ from the union of its units")
            # pairs generated AFTER the batch fold is formed, from its hidden-scored rows only
            in_fold = _id_mask(pos, n, f.scored_row_ids)
            got = _comparable_pairs(ids, in_fold, states, key_group)
            unit_of_system = {cells_of[u][0][1]: u for u in b}
            by_unit: dict[str, set] = defaultdict(set)
            for i, jj in got:
                u = unit_of_system[sysk[i]]
                by_unit[u].add((ids[i], ids[jj]))
                recs.append({"fold_id": bid, "unit_fold_id": u, "row_id_a": ids[i], "row_id_b": ids[jj],
                             "metal_state_a": states[i], "metal_state_b": states[jj], "publication_group": g[i],
                             "condition_key_id": "ck_" + hashlib.sha1(ck[i].encode()).hexdigest()[:16],
                             "category_class": unit_fold[u].meta["category_class"]})
            for u in b:
                if by_unit.get(u, set()) != pair_set.get(u, set()):
                    raise AssertionError(f"V5PAIR {bid}: regenerated pairs of unit {u} differ from its unbatched pairs")
            folds.append(f)
    ptab = pd.DataFrame(recs, columns=["fold_id", "unit_fold_id", "row_id_a", "row_id_b", "metal_state_a",
                                       "metal_state_b", "publication_group", "condition_key_id", "category_class"])
    stats["n_batches"] = int(sum(s["n_batches"] for s in stats.values() if isinstance(s, dict)))
    stats["n_units"] = int(sum(s["n_units"] for s in stats.values() if isinstance(s, dict)))
    stats["n_test_test_row_pairs"] = int(len(ptab))
    return folds, ptab, stats


def batch_listing(folds: Sequence[FI.Fold]) -> list[dict]:
    """Per-batch listing of batched V5-P / V5-PAIR folds for ``folds/INDEX.json``."""
    return [{"fold_id": f.fold_id, "half": f.half, "seed": f.seed, "n_units": int(f.meta["n_units"]),
             "unit_fold_ids": list(f.meta["unit_fold_ids"]), "metal_states": [c[0] for c in f.meta["cells"]],
             "n_hidden": len(f.hidden_row_ids), "n_scored": len(f.scored_row_ids), "fold_hash": f.fold_hash}
            for f in folds]


# --------------------------------------------------------------------------------------------- #
# inner cells (section 7)
# --------------------------------------------------------------------------------------------- #
#
# ONE implementation of the section 7 inner V5 design.  The fold builder (:func:`inner_cells_V5`) and the
# conformal calibration splitter (``models.interface.InnerCellCalibration``) both take their inner units from
# :func:`inner_cell_majority` (which cells) and :func:`inner_cell_assignment` (which inner fold, which subsample);
# ``tests/test_inner_designs.py`` asserts that the two return identical inner units for one outer fold and seed.

def majority_group(groups: Iterable[str]) -> str:
    """The publication group holding the most of a cell's rows (ties: the smallest group label)."""
    counts = Counter(str(g) for g in groups)
    if not counts:
        raise ValueError("majority_group: a cell without rows")
    top = max(counts.values())
    return min(g for g, n in counts.items() if n == top)


def inner_cell_majority(train_df: pd.DataFrame, thresholds: Thresholds, v6_mask: pd.Series,
                        medium: str = "all") -> dict[Cell, str]:
    """Section 7 inner V5 candidate cells of an outer training set, each with the publication group
    (``group_cross_publication_copy``) holding its majority of rows.

    Candidates: the known-state cells of ``train_df`` eligible under ``thresholds`` recomputed on those rows
    (:func:`eligible_cells`), minus cells with a ``V6_TARGET_ROWS`` row and cells without a scorable
    (:func:`io.scorable_mask`, ``medium``) row."""
    if train_df.empty:
        return {}
    v6a = v6_mask.reindex(train_df.index).fillna(False).to_numpy(dtype=bool)
    scorable = FI.scorable_mask(train_df, v6_mask).to_numpy(dtype=bool) & medium_mask(train_df, medium)
    el = eligible_cells(train_df, thresholds, medium)
    el = el[el["eligible"]]
    groups = FI.publication_groups(train_df, FI.GROUP_COL).to_numpy(dtype=object)
    st = train_df[SG.METAL_COL].to_numpy(dtype=object)
    sy = train_df[SG.SYSTEM_COL].to_numpy(dtype=object)
    out: dict[Cell, str] = {}
    for m, s in zip(el["metal_state"], el["extractant_system_key"]):
        own = (st == m) & (sy == s)
        if (own & v6a).any() or not (own & scorable).any():
            continue
        out[(str(m), str(s))] = majority_group(groups[own])
    return out


def inner_cell_assignment(majority: Mapping[Cell, str], seed: int, n_inner: int = INNER_N_FOLDS,
                          max_cells: int = INNER_MAX_CELLS) -> list[list[Cell]]:
    """The cells of each of the ``n_inner`` inner folds (sorted; an inner fold may be empty).

    Rule (recorded in pre-registration section 7): the majority publication groups, weighted by their number of
    candidate cells, are dealt by :func:`io.greedy_balance` -- groups in the random order of stream
    ``SeedSequence([seed, 15])``, each to the inner fold with the fewest cells so far (ties to the lower index);
    an inner fold with more than ``max_cells`` cells keeps a subsample of ``max_cells`` drawn by stream
    ``SeedSequence([seed, 16, k])``."""
    if not majority:
        return [[] for _ in range(int(n_inner))]
    weights = Counter(majority.values())
    bin_of_group = FI.greedy_balance(dict(weights), n_inner, FI.seed_rng(seed, "inner_V5_groups"))
    out: list[list[Cell]] = []
    for k in range(int(n_inner)):
        ck = sorted(c for c in majority if bin_of_group[majority[c]] == k)
        if len(ck) > max_cells:
            rng = FI.seed_rng(seed, "inner_V5_subsample", k)
            ck = sorted(ck[i] for i in rng.choice(len(ck), size=max_cells, replace=False))
        out.append(ck)
    return out


def inner_cells_V5(train_df: pd.DataFrame, thresholds: Thresholds = PRIMARY, seed: int = FI.DISCOVERY_SEEDS[0], *,
                   medium: str = "all", component_aware: bool = True, component_map: Mapping[str, str] | None = None,
                   batched: bool = True, n_inner: int = INNER_N_FOLDS, max_cells: int = INNER_MAX_CELLS,
                   v6_ids: Iterable[str] | None = None, check: bool = True,
                   max_cells_per_batch: int | None = None) -> list[FI.Fold]:
    """Section 7 inner V5 folds of an outer training set.

    Inner hidden cells: :func:`inner_cell_majority` (eligible under ``thresholds`` recomputed on ``train_df``, no
    ``V6_TARGET_ROWS`` row, at least one scorable row), dealt into ``n_inner`` inner folds with at most
    ``max_cells`` cells each by :func:`inner_cell_assignment`.  ``batched`` follows the outer batching rule inside
    each inner fold (colouring + eligibility re-check on ``train_df``); otherwise one fold per cell.
    ``max_cells_per_batch`` caps the inner colour classes as for the outer rule (``None`` = registered rule).
    ``batch_id`` = ``i<k>_b<j>``."""
    variant = Variant("inner", thresholds, medium=medium, component_aware=component_aware,
                      parent_structure=component_map is not None)
    v6 = FI.v6_mask_for(train_df, v6_ids)
    scorable = FI.scorable_mask(train_df, v6).to_numpy(dtype=bool)
    v6a = v6.to_numpy(dtype=bool)
    majority = inner_cell_majority(train_df, thresholds, v6, medium)
    if not majority:
        return []
    per_fold = inner_cell_assignment(majority, seed, n_inner, max_cells)
    space = EligibilitySpace(train_df, medium) if batched else None
    ids = train_df[FI.ROW_ID].astype(str).to_numpy(dtype=object)
    folds = []
    for k, ck in enumerate(per_fold):
        if not ck:
            continue
        masks = cell_masks(train_df, ck, variant, component_map)
        if batched:
            batches, _ = batch_cells(space, ck, thresholds, masks, FI.seed_rng(seed, "inner_V5_batch", k),
                                     max_cells_per_batch=max_cells_per_batch)
        else:
            batches = [[c] for c in ck]
        for j, b in enumerate(batches):
            bid = f"i{k}_b{j:03d}"
            f = _cell_fold(train_df, ids, variant, b, masks, scorable, v6a, {}, scheme="inner_batched" if batched else "inner_exact",
                           fold_id=f"inner_s{seed}_{bid}", seed=seed, batch_id=bid, design="V5_inner",
                           component_map=component_map, extra_meta={"inner_fold": k, "n_cells": len(b)})
            if check:
                FI.assert_scoring_clean(f, train_df, v6)
                guard_v5(f, train_df, component_map=component_map)
            folds.append(f)
    return folds

"""``models/inner_design.py`` -- the V5 inner design of the learned arms under POST-HOC addendum 1
(``preregistration.md``, addendum 1 of 2026-09-15, items 1-2; sealed sections 3.1, 7 and 12).

Addendum 1 item 1 replaces, for every learned arm and B6 and every V5 design and variant, the batched / per-cell
V5 inner tuning of section 7 by ONE inner fit per inner fold: all inner hidden cells of an inner fold (at most 30) are
hidden simultaneously, each with its registered hiding; eligibility is re-checked with all of them hidden, and a cell
that fails stays hidden but is dropped from the inner score.  The inner cells themselves are unchanged: which cells,
in which inner fold, from ``folds.cell_holdout.inner_cell_majority`` / ``inner_cell_assignment`` through
``interface.InnerCellCalibration.unit_assignment`` (one implementation per design, ``tests/test_inner_designs.py``).
Item 2 fixes what the consumers do with the three splits: a configuration's score on a split is the unit-macro MAE
over the split's cells (``InnerSplit.row_units``), the selection score is the mean over the three splits
(:func:`fold_macro_table`, :func:`fold_mean_scores`; the 0.005 tie rule of section 7 is applied by each tuner), the
conformal residuals of split ``j`` come from the configuration selected on the other two splits (cross-fitted, reusing
the fits already made) and the outer refit iteration / epoch count is the median over the three splits.

:class:`SimultaneousInnerCells`
    ``splits(table, mask, context)`` returns one ``interface.InnerSplit`` per inner fold ``f`` (``fold=f``,
    ``unit=("inner_fold", f)``): ``train_mask`` is the outer-training mask with every cell of the fold hidden under the
    registered rule (``interface.hide_cell_mask``: the cell's state and the element's X(?) rows in the cell's system
    and, component-aware, in every system sharing a component structure); ``cal_positions`` are the scorable rows
    (``interface.scorable_mask(known_state_only=True)``, the fold's medium) of the cells that pass the re-check;
    ``row_units`` is the cell label of every calibration row; ``hidden_positions`` every position the hiding removed;
    ``certificate`` = (the whole table with every cell of the fold hidden, every position of those cells), a superset
    split on both sides for ``ConformalWrapper(guard="nested_certificate")``; ``meta`` names the cells, the scored
    cells and the dropped cells.  Dropped cells are also kept in :attr:`SimultaneousInnerCells.last_dropped`.
:func:`isolation_reports` / :func:`assert_isolated`
    ``data.leakage.fold_isolation_check(level="V5", component_aware=True)`` on every split (training rows against
    calibration rows, optionally the certificate too) -- the section 3.1 guard, for tests and audits.
:func:`learned_arm_inner_design`
    the inner design a learned arm uses under an outer design: V5 (and its variants; V5-P, V5-PAIR and V6 heavy-arm
    fits use the V5 primary inner design, section 3.1 resolution) -> :class:`SimultaneousInnerCells`; V1 / V0 ->
    ``GroupKFoldCalibration``; V2 -> ``InnerMetalCalibration`` (unchanged by the addendum: already one fit per fold).
:func:`assert_learned_arm_design`
    refuses the per-cell ``InnerCellCalibration`` (and every subclass that is not simultaneous) for a learned arm, so
    nothing learned can fall back to the batched / per-cell tuning path; the deterministic comparators keep it.

Nothing here reads ``log_D``.  Where the addendum is silent the reading is listed in :data:`REGISTRATION_CHOICES`.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from gen19ct.chemistry import support_graph as SG
from gen19ct.data import leakage as LK
from gen19ct.folds import cell_holdout as CH
from gen19ct.folds import io as FI
from gen19ct.models import interface as I

NAME = "V5_inner_simultaneous"
ADDENDUM = "POST-HOC addendum 1 (2026-09-15) items 1-2"
#: the registered number of inner folds (section 7) -- the addendum keeps three on every seed
N_INNER_FOLDS = CH.INNER_N_FOLDS
MEDIA: tuple[str, ...] = ("all", CH.HNO3)

#: where the addendum is silent, the reading implemented here (reported with every run that uses this design)
REGISTRATION_CHOICES: dict[str, str] = {
    "inner_cells": "the inner cells and their inner folds are InnerCellCalibration.unit_assignment (cell_holdout."
                   "inner_cell_majority + inner_cell_assignment, streams [seed, 15] and [seed, 16, k]); only the hiding "
                   "changed: all cells of an inner fold are hidden in one training set",
    "hiding": "each cell's registered hiding (hide_cell_mask, component-aware unless the variant says otherwise, X(?) "
              "alias rows included) applied cumulatively; the split's hidden rows are the union",
    "recheck": "cell_holdout.EligibilitySpace.check on the outer-training rows with the OTHER cells of the fold hidden "
               "and nothing restored: every condition -- the cell's own row and publication counts (k, p), the metal's "
               "other systems and the system's other states (>= m), and the cell edge not being a bridge -- is counted "
               "among the rows the other cells' registered hiding leaves in training, exactly as the fold builder's "
               "batch re-check does (keep = (total - own) == 0; section 3.1: 'including the cell's own row and "
               "publication counts (k, p) ... the literal and stricter reading the fold builder applied').  So a cell "
               "whose rows another cell of the same fold removes (a same-state cell in a component-sharing system) "
               "fails k and stays hidden but unscored (task X finding V-F03; on real training rows 1 of 90 cells on "
               "V5 exact Eu(III)__a9ed8f70c7 and 0 of 90 on batched s104729_S_b000, seed 104729).  A failing cell "
               "stays hidden, its rows leave cal_positions and it is recorded in last_dropped / "
               "InnerSplit.meta['dropped_cells']",
    "calibration_rows": "the scorable rows (known state, not Sr(III), not V6_TARGET_ROWS, not context."
                        "exclude_from_scoring; the variant's medium) of the surviving cells, as InnerCellCalibration",
    "certificate": "(the whole table with every cell of the fold hidden, every position of the fold's cells): a superset "
                   "of the split on both sides, so ConformalWrapper(guard='nested_certificate') stays valid; it depends on "
                   "the fold's cells, so it is not shared across outer folds as the per-cell certificates were",
    "empty_fold": "an inner fold without a scorable surviving cell yields no split; require_all_folds=True (default) "
                  "raises instead, because the addendum tunes on three inner folds on every seed",
    "selection_score": "per inner fold the mean over its cells of the cell MAE (unit-macro), then the mean over the "
                       "inner folds (fold_mean_scores); a unit present in several splits of one fold (never under this "
                       "design) is pooled within the fold only",
}


def _thresholds(k: int, p: int, m: int) -> CH.Thresholds:
    return CH.Thresholds(int(k), int(p), int(m))


class SimultaneousInnerCells(I.InnerCellCalibration):
    """The addendum-1 V5 inner design (module docstring): the inner cells of ``InnerCellCalibration``, hidden
    simultaneously per inner fold.

    ``k, p, m``                  the variant's eligibility thresholds (recomputed on the outer training rows)
    ``n_folds``                  inner folds (3); ``max_cells_per_fold`` the seeded cap (30)
    ``component_aware``          the registered hiding (``False``: the V5-cell-only sensitivity)
    ``component_map``            the parent-structure sensitivity's component identities
    ``medium``                   ``"all"`` or ``"HNO3"`` (the HNO3-only sensitivity: cells eligible on the HNO3 rows of
                                 the outer training rows, HNO3 calibration rows only); needs ``frame``
    ``frame``                    the arm frame (``acid_primary``) over at least the outer training rows; only read for
                                 a medium other than ``"all"``
    ``require_all_folds``        raise when an inner fold holds no scorable surviving cell (default), else skip it
    """

    name = NAME

    def __init__(self, k: int = 10, p: int = 1, m: int = 3, n_folds: int = N_INNER_FOLDS,
                 max_cells_per_fold: int = CH.INNER_MAX_CELLS, component_aware: bool = True,
                 component_map: Mapping[str, str] | None = None, medium: str = "all",
                 frame: pd.DataFrame | None = None, require_all_folds: bool = True):
        super().__init__(k, p, m, n_folds, max_cells_per_fold, component_aware, component_map)
        if medium not in MEDIA:
            raise ValueError(f"medium must be one of {MEDIA}, got {medium!r}")
        if medium != "all" and frame is None:
            raise ValueError("a medium-restricted inner design needs the arm frame (acid_primary)")
        if medium != "all" and "acid_primary" not in frame.columns:
            raise KeyError("frame lacks acid_primary (the medium column)")
        self.medium, self.frame = medium, frame
        self.require_all_folds = bool(require_all_folds)
        self.last_dropped: dict[int, list[tuple[str, str]]] = {}
        self.last_checks: dict[int, list[dict[str, Any]]] = {}
        self.last_cells: dict[int, list[tuple[str, str]]] = {}

    @classmethod
    def for_variant(cls, variant: str = "primary", *, frame: pd.DataFrame | None = None,
                    n_folds: int = N_INNER_FOLDS, max_cells_per_fold: int = CH.INNER_MAX_CELLS,
                    require_all_folds: bool = True) -> "SimultaneousInnerCells":
        """The design of a registered V5 variant (``folds.cell_holdout.VARIANTS``)."""
        v = CH.VARIANTS[variant]
        return cls(v.thresholds.k, v.thresholds.p, v.thresholds.m, n_folds, max_cells_per_fold,
                   component_aware=v.component_aware, component_map=CH.component_map_for(v), medium=v.medium,
                   frame=frame, require_all_folds=require_all_folds)

    @property
    def thresholds(self) -> CH.Thresholds:
        return _thresholds(self.k, self.p, self.m)

    def describe(self) -> dict[str, Any]:
        return {"design": "V5", "inner_design": self.name, "addendum": ADDENDUM, "thresholds": self.thresholds.tag,
                "medium": self.medium, "component_aware": self.component_aware,
                "parent_structure": self.component_map is not None, "n_inner": self.n_folds,
                "max_cells": self.max_cells, "one_fit_per_inner_fold": True, "require_all_folds": self.require_all_folds,
                "unit_functions": "cell_holdout.inner_cell_majority + inner_cell_assignment"}

    # ----------------------------------------------------------------------------------------- #
    # inner cells: identical to InnerCellCalibration (the medium-restricted variant as V5ExactInnerCells)
    # ----------------------------------------------------------------------------------------- #
    def _unit_frame(self, table: I.RowTable, mask: np.ndarray) -> pd.DataFrame:
        uf = I.unit_frame(table, mask)
        if self.medium != "all":
            acid = self.frame["acid_primary"].reindex(uf.index)
            if acid.isna().any():
                raise KeyError("frame does not cover every outer training row (acid_primary)")
            uf["acid_primary"] = acid.to_numpy(dtype=object)
        return uf

    def _majority(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> dict[tuple[str, str], str]:
        if self.medium == "all":
            return super()._majority(table, mask, context)
        if context.v6_mask is None:
            raise ValueError("context.v6_mask is required: V6 cells are never inner validation cells")
        uf = self._unit_frame(table, mask)
        I._require_groups(table, np.flatnonzero(mask))
        v6 = pd.Series(table.aligned_bool(context.v6_mask, "v6_mask")[np.flatnonzero(mask)], index=uf.index)
        return CH.inner_cell_majority(uf, self.thresholds, v6, self.medium)

    def unit_assignment(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> list[list[tuple[str, str]]]:
        """The cells of each inner fold -- ``InnerCellCalibration.unit_assignment`` (same functions, same streams, same
        cache) for ``medium="all"``; the medium-restricted majority is never put in the all-media cache."""
        if self.medium == "all":
            return super().unit_assignment(table, mask, context)
        seed = I._require_seed(context)
        return CH.inner_cell_assignment(self._majority(table, mask, context), seed, self.n_folds, self.max_cells)

    def _medium_rows(self, table: I.RowTable, mask: np.ndarray) -> np.ndarray:
        if self.medium == "all":
            return np.ones(table.n, dtype=bool)
        acid = self.frame["acid_primary"].reindex(table.index)
        if acid.isna()[mask].any():
            raise KeyError("frame does not cover every outer training row (acid_primary)")
        return (acid.to_numpy(dtype=object) == self.medium) & mask

    # ----------------------------------------------------------------------------------------- #
    def cell_positions(self, table: I.RowTable, state: str, system: str) -> np.ndarray:
        """Every table position of a cell (in and outside the outer training rows)."""
        return table.by_sm.get((table.sys_code[system], table.state_code[state]), np.zeros(0, dtype=np.int64))

    def _hide(self, table: I.RowTable, base: np.ndarray, state: str, system: str) -> tuple[np.ndarray, np.ndarray]:
        return I.hide_cell_mask(table, base, state, system, component_aware=self.component_aware,
                                component_map=self.component_map)

    def splits(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> list[I.InnerSplit]:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (table.n,):
            raise ValueError("training mask does not match the RowTable")
        per_fold = self.unit_assignment(table, mask, context)
        ok = I.scorable_mask(table, context, known_state_only=True) & self._medium_rows(table, mask)
        thr = self.thresholds
        space = CH.EligibilitySpace(self._unit_frame(table, mask), self.medium)
        mpos = np.flatnonzero(mask)
        universe = np.ones(table.n, dtype=bool)
        self.last_dropped, self.last_checks, self.last_cells = {}, {}, {}
        out: list[I.InnerSplit] = []
        for f in range(self.n_folds):
            cells = [(str(a), str(b)) for a, b in (per_fold[f] if f < len(per_fold) else [])]
            self.last_cells[f] = list(cells)
            # every cell's own registered hiding on the outer training rows; the split hides their union
            drops = {c: self._hide(table, mask, c[0], c[1])[1] for c in cells}
            hidden = np.zeros(table.n, dtype=bool)
            cert_train = universe.copy()
            for c in cells:
                hidden |= drops[c]
                cert_train, _ = self._hide(table, cert_train, c[0], c[1])
            train = mask & ~hidden
            total = (np.sum([drops[c][mpos].astype(np.int32) for c in cells], axis=0) if cells
                     else np.zeros(len(mpos), dtype=np.int32))
            cal_parts, unit_parts, cert_parts, checks, dropped, scored = [], [], [], [], [], []
            for c in cells:
                st, sy = c
                p = self.cell_positions(table, st, sy)
                cert_parts.append(p)
                q = p[mask[p]]
                # every condition counted with the OTHER cells of the fold hidden and nothing restored, the cell's own
                # row and publication counts included (REGISTRATION_CHOICES "recheck": the section 3.1 batch re-check
                # reading, cell_holdout's keep = (total - own) == 0); a cell emptied by another cell's hiding fails k
                keep = (total - drops[c][mpos].astype(np.int32)) == 0
                rep = space.check(c, thr, keep=keep)
                cal = q[ok[q]]
                rec = {"metal_state": st, "system": sy, "n_training_rows": int(len(q)), "n_scorable_rows": int(len(cal)),
                       "n_own_rows_hidden_by_other_cells": int(len(q) - int(np.count_nonzero(keep[np.isin(mpos, q)]))),
                       **{k: (bool(v) if isinstance(v, (bool, np.bool_)) else int(v)) for k, v in rep.items()}}
                if not rep["eligible"] or not len(cal):
                    rec["scored"] = False
                    rec["dropped_reason"] = "failed_recheck" if not rep["eligible"] else "no_scorable_row"
                    dropped.append(c)
                else:
                    rec["scored"] = True
                    scored.append(c)
                    cal_parts.append(cal)
                    unit_parts.append(np.full(len(cal), I.cell_unit_label(st, sy), dtype=object))
                checks.append(rec)
            self.last_dropped[f], self.last_checks[f] = dropped, checks
            if not cal_parts:
                if self.require_all_folds:
                    raise ValueError(f"{self.name}: inner fold {f} holds no scorable surviving cell "
                                     f"({len(cells)} cells, {len(dropped)} dropped); the addendum tunes on "
                                     f"{self.n_folds} inner folds")
                continue
            cal = np.concatenate(cal_parts)
            order = np.argsort(cal, kind="stable")
            cal, units = cal[order], np.concatenate(unit_parts)[order]
            cert_test = np.unique(np.concatenate(cert_parts))
            out.append(I.InnerSplit(unit=("inner_fold", f), fold=f, train_mask=train, cal_positions=cal,
                                    hidden_positions=np.flatnonzero(hidden), certificate=(cert_train, cert_test),
                                    row_units=units,
                                    meta={"design": self.name, "inner_fold": f, "cells": list(cells),
                                          "scored_cells": scored, "dropped_cells": dropped, "n_cells": len(cells),
                                          "n_scored_cells": len(scored), "n_hidden_rows": int(hidden.sum())}))
        return out

    def training_rows_of_cells(self, table: I.RowTable, mask: np.ndarray, cells: Iterable[tuple[str, str]]) -> np.ndarray:
        """Positions inside ``mask`` that the registered hiding of ``cells`` removes (their own rows, X(?) alias rows
        and component-sharing rows) -- what a split's training rows must not contain."""
        gone = np.zeros(table.n, dtype=bool)
        for st, sy in cells:
            gone |= self._hide(table, np.asarray(mask, dtype=bool), st, sy)[1]
        return np.flatnonzero(gone)


# --------------------------------------------------------------------------------------------- #
# the isolation guard on every split (section 3.1; a test / audit helper)
# --------------------------------------------------------------------------------------------- #

def isolation_reports(splits: Sequence[I.InnerSplit], table: I.RowTable, slim: pd.DataFrame, *,
                      component_map: Mapping[str, str] | None = None, component_aware: bool = True,
                      certificate: bool = False, near_dup_sig: int | None = FI.NEAR_DUP_SIG,
                      near_dup_value_tol: float | None = FI.NEAR_DUP_VALUE_TOL) -> list[dict[str, Any]]:
    """``fold_isolation_check(level="V5", component_aware=...)`` on every split: training rows against calibration
    rows (``certificate=True``: the certificate's two sides instead).  ``slim`` is ``folds.io.slim_frame`` of the arm
    frame over the table's rows.  Returns the reports (never raises on a violation: read ``report["ok"]``)."""
    out = []
    for sp in splits:
        if certificate:
            if sp.certificate is None:
                raise ValueError(f"split {sp.unit} carries no certificate")
            tr, te = table.index[sp.certificate[0]], table.index[np.asarray(sp.certificate[1], dtype=np.int64)]
        else:
            tr, te = table.index[sp.train_mask], table.index[np.asarray(sp.cal_positions, dtype=np.int64)]
        rep = LK.fold_isolation_check(tr, te, slim, level="V5", component_aware=component_aware,
                                      component_map=component_map, near_dup_sig=near_dup_sig,
                                      near_dup_value_tol=near_dup_value_tol, raise_on_violation=False)
        rep["unit"] = sp.unit
        rep["fold"] = int(sp.fold)
        out.append(rep)
    return out


def assert_isolated(splits: Sequence[I.InnerSplit], table: I.RowTable, slim: pd.DataFrame, **kw: Any) -> list[dict[str, Any]]:
    """:func:`isolation_reports` that raises ``AssertionError`` on the first split with a violation."""
    reps = isolation_reports(splits, table, slim, **kw)
    for rep in reps:
        if not rep["ok"]:
            bad = {k: v for k, v in rep["violations"].items() if v}
            raise AssertionError(f"inner split {rep['unit']} violates the V5 isolation rule: {bad}")
    return reps


# --------------------------------------------------------------------------------------------- #
# which inner design a learned arm uses, and the refusal of the per-cell path
# --------------------------------------------------------------------------------------------- #

def is_per_cell_design(splitter: Any) -> bool:
    """``True`` for the per-cell ``InnerCellCalibration`` and any subclass that is not simultaneous (the design of the
    deterministic comparators' intervals, and the pre-addendum tuning path of the learned arms)."""
    return isinstance(splitter, I.InnerCellCalibration) and not isinstance(splitter, SimultaneousInnerCells)


def assert_learned_arm_design(splitter: Any, what: str = "a learned arm") -> None:
    """Refuse the per-cell V5 inner design for a learned arm (addendum 1 item 1)."""
    if is_per_cell_design(splitter):
        raise ValueError(f"{what}: the per-cell V5 inner design {type(splitter).__name__} is not a learned-arm inner "
                         f"design under {ADDENDUM}; use inner_design.SimultaneousInnerCells")


def learned_arm_inner_design(design: str, variant: str = "primary", *, frame: pd.DataFrame | None = None,
                             require_all_folds: bool = True, n_folds: int = N_INNER_FOLDS,
                             max_cells_per_fold: int = CH.INNER_MAX_CELLS) -> Any:
    """The inner design of a learned arm (B5 / FLAT_CAT / B6 / B6r0 / B8 / M*) under an outer design: V5 and its
    variants (V5-P, V5-PAIR and V6 heavy-arm fits use the V5 primary inner design) -> :class:`SimultaneousInnerCells`;
    V1 -> ``GroupKFoldCalibration(unit="v1_unit")``; V0 -> ``GroupKFoldCalibration(unit="publication_group")``;
    V2 -> ``InnerMetalCalibration`` (``variant="state"`` has no separate inner design: the inner metals are hidden at
    the element level in every V2 variant, as the fold builder's ``inner_metals_V2`` default)."""
    d = str(design).upper().replace("-", "").replace("_", "")
    if d == "V5":
        return SimultaneousInnerCells.for_variant(variant, frame=frame, n_folds=n_folds,
                                                  max_cells_per_fold=max_cells_per_fold,
                                                  require_all_folds=require_all_folds)
    if d in ("V5P", "V5PAIR", "V6"):
        return SimultaneousInnerCells.for_variant("primary", frame=frame, n_folds=n_folds,
                                                  max_cells_per_fold=max_cells_per_fold,
                                                  require_all_folds=require_all_folds)
    if d == "V1":
        return I.GroupKFoldCalibration(n_folds, unit="v1_unit")
    if d == "V0":
        return I.GroupKFoldCalibration(n_folds, unit="publication_group")
    if d == "V2":
        return I.InnerMetalCalibration(n_folds)
    raise NotImplementedError(f"no learned-arm inner design for {design!r}")


# --------------------------------------------------------------------------------------------- #
# the selection score (addendum 1 item 2): unit-macro per inner fold, mean over the folds
# --------------------------------------------------------------------------------------------- #

SPLIT_UNIT_ERROR_COLUMNS: tuple[str, ...] = ("config", "inner_fold", "unit", "n_rows", "sum_abs_error")


def fold_macro_table(split_unit_errors: pd.DataFrame, *, exclude_folds: Iterable[int] = ()) -> pd.DataFrame:
    """Per ``(config, inner_fold)``: the unit-macro MAE of that inner fold -- the mean over the fold's averaging
    units of the unit's mean absolute error (a unit's rows pooled over the fold's splits) -- with ``n_units`` and
    ``n_rows``.  ``split_unit_errors`` has one row per (configuration, split, unit) with ``inner_fold``, ``n_rows`` and
    ``sum_abs_error`` (:data:`SPLIT_UNIT_ERROR_COLUMNS`; a ``split_id`` / ``split`` column is ignored)."""
    sue = pd.DataFrame(split_unit_errors)
    missing = [c for c in SPLIT_UNIT_ERROR_COLUMNS if c not in sue.columns]
    if missing:
        raise KeyError(f"split_unit_errors: columns missing {missing}")
    drop = {int(f) for f in exclude_folds}
    if drop:
        sue = sue[~sue["inner_fold"].astype(int).isin(drop)]
    if sue.empty:
        return pd.DataFrame(columns=["config", "inner_fold", "macro_mae", "n_units", "n_rows"])
    sue = sue.assign(inner_fold=sue["inner_fold"].astype(int), n_rows=sue["n_rows"].astype(float),
                     sum_abs_error=sue["sum_abs_error"].astype(float))
    agg = sue.groupby(["config", "inner_fold", "unit"], sort=True)[["n_rows", "sum_abs_error"]].sum()
    unit_mae = (agg["sum_abs_error"] / agg["n_rows"]).rename("mae")
    per = unit_mae.groupby(level=["config", "inner_fold"]).agg(macro_mae="mean", n_units="size")
    per["n_rows"] = agg["n_rows"].groupby(level=["config", "inner_fold"]).sum().astype(int)
    return per.reset_index()


def fold_mean_scores(split_unit_errors: pd.DataFrame, *, exclude_folds: Iterable[int] = ()) -> dict[str, float]:
    """``{config: mean over the inner folds of the fold's unit-macro MAE}`` -- the addendum-1 selection score (before
    the tie rule).  Every configuration must be scored on the same folds."""
    per = fold_macro_table(split_unit_errors, exclude_folds=exclude_folds)
    if per.empty:
        return {}
    folds = per.groupby("config")["inner_fold"].agg(lambda s: tuple(sorted(s)))
    if folds.nunique() != 1:
        raise ValueError(f"fold_mean_scores: the configurations are not scored on the same inner folds: {folds.to_dict()}")
    return per.groupby("config")["macro_mae"].mean().to_dict()


def per_fold_unit_macro(abs_errors: Sequence[np.ndarray], units: Sequence[np.ndarray], folds: Sequence[int]) -> pd.Series:
    """The unit-macro MAE per inner fold from per-split arrays (``abs_errors[i]`` and ``units[i]`` aligned, split ``i``
    in inner fold ``folds[i]``): a unit's rows pooled over the fold's splits; index = inner fold."""
    if not (len(abs_errors) == len(units) == len(folds)):
        raise ValueError("per_fold_unit_macro: abs_errors, units and folds must be aligned")
    out: dict[int, float] = {}
    for f in sorted({int(x) for x in folds}):
        idx = [i for i, g in enumerate(folds) if int(g) == f]
        e = np.concatenate([np.asarray(abs_errors[i], dtype=float) for i in idx])
        u = np.concatenate([np.asarray(units[i], dtype=object).astype(str) for i in idx])
        if not len(e):
            continue
        out[f] = float(pd.Series(e).groupby(u, sort=True).mean().mean())
    return pd.Series(out, dtype=float, name="macro_mae")


def median_count(counts: Iterable[int]) -> int:
    """The outer refit's iteration / epoch count: the median over the inner folds of the selected configuration's
    best count, rounded half up, at least 1 (section 6 M-model rule; addendum 1 item 2 for every tuned arm)."""
    arr = np.asarray(list(counts), dtype=float)
    if not len(arr) or not np.isfinite(arr).all():
        raise ValueError("median_count: no finite count")
    return max(1, int(math.floor(float(np.median(arr)) + 0.5)))


def cross_fit_plan(folds: Iterable[int]) -> dict[int, tuple[int, ...]]:
    """``{calibration fold j: the folds its configuration is selected on}`` -- every other fold (addendum 1 item 2).
    Fewer than two folds give an empty plan (no cross-fitted calibration is possible)."""
    fs = sorted({int(f) for f in folds})
    if len(fs) < 2:
        return {}
    return {j: tuple(f for f in fs if f != j) for j in fs}


def wildcard_copy_partners_in_inner_training(splits: Sequence[I.InnerSplit], table: I.RowTable, row_ids: pd.Series,
                                             pairs: pd.DataFrame) -> list[dict[str, Any]]:
    """Per split, the calibration rows whose section 2 wildcard copy (``folds/wildcard_copy_pairs.csv``: the same
    conditions and log_D under another structure key -- ``STRUCTURE_WILDCARD`` -- or another state token --
    ``STATE_WILDCARD``) sits in that split's TRAINING rows.  The registered hiding does not remove such a copy (no shared
    canonical component) and the guard's near-duplicate key includes the system, so ``fold_isolation_check`` passes it.
    A diagnostic only: the registration is silent on inner validation, so the selection score is not changed here
    (task X finding VL-A1-02; ``evaluation.discovery.READINGS['inner_wildcard_copies']``).  ``row_ids`` maps table
    labels to canonical measurement ids."""
    ids = pd.Series(row_ids).reindex(table.index).astype(str).to_numpy(dtype=object)
    if pd.isna(pd.Series(row_ids).reindex(table.index)).any():
        raise KeyError("row_ids does not cover every table row")
    a, b = pairs["id_a"].astype(str), pairs["id_b"].astype(str)
    kind = pairs["kind"].astype(str).to_numpy(dtype=object)
    cross = (pairs["system_a"].astype(str) != pairs["system_b"].astype(str)).to_numpy(dtype=bool)
    out = []
    for sp in splits:
        cal = set(ids[np.asarray(sp.cal_positions, dtype=np.int64)])
        tr = set(ids[np.asarray(sp.train_mask, dtype=bool)])
        hit = ((a.isin(cal) & b.isin(tr)) | (b.isin(cal) & a.isin(tr))).to_numpy(dtype=bool)
        flagged = sorted((set(a[hit]) | set(b[hit])) & cal)
        kinds, counts = np.unique(kind[hit], return_counts=True) if hit.any() else (np.array([]), np.array([]))
        out.append({"inner_fold": int(sp.fold), "n_calibration_rows": int(len(sp.cal_positions)),
                    "n_calibration_rows_with_partner_in_training": int(len(flagged)), "n_pairs": int(hit.sum()),
                    "by_kind": {str(k): int(v) for k, v in zip(kinds, counts)},
                    "n_pairs_cross_system": int((hit & cross).sum()), "n_pairs_same_system": int((hit & ~cross).sum()),
                    "calibration_row_ids": [str(x) for x in flagged]})
    return out


def splits_summary(splits: Sequence[I.InnerSplit]) -> pd.DataFrame:
    """One row per split: fold, cells, scored / dropped cells, training / hidden / calibration row counts."""
    recs = []
    for sp in splits:
        meta = dict(sp.meta or {})
        recs.append({"fold": int(sp.fold), "unit": repr(sp.unit), "n_cells": meta.get("n_cells"),
                     "n_scored_cells": meta.get("n_scored_cells"),
                     "n_dropped_cells": len(meta.get("dropped_cells", ())) if "dropped_cells" in meta else None,
                     "n_train": int(sp.train_mask.sum()), "n_hidden": int(len(sp.hidden_positions)),
                     "n_calibration": int(len(sp.cal_positions)),
                     "n_units": int(len(set(sp.row_units))) if sp.row_units is not None else None})
    return pd.DataFrame(recs)


__all__ = ["NAME", "ADDENDUM", "N_INNER_FOLDS", "REGISTRATION_CHOICES", "SimultaneousInnerCells", "isolation_reports",
           "wildcard_copy_partners_in_inner_training",
           "assert_isolated", "is_per_cell_design", "assert_learned_arm_design", "learned_arm_inner_design",
           "fold_macro_table", "fold_mean_scores", "per_fold_unit_macro", "median_count", "cross_fit_plan",
           "splits_summary", "SPLIT_UNIT_ERROR_COLUMNS"]

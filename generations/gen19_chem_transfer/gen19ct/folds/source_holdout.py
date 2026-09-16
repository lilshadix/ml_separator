"""``folds/source_holdout.py`` -- V1, leave-publication-out (pre-registration sections 2, 3.2, 7).

Publication unit: ``group_cross_publication_copy`` of ``data_audit/leakage_publication_components.csv`` (the
merge of ``g19_publication_id`` groups linked by corrected / primary-source DOIs, archive duplicate groups and
value-matched copies).  Designs built here:

* :func:`v1_exact_folds` -- exact leave-one-group-out over the groups with >= 20 MODEL rows plus ONE remainder
  fold pooling the smaller groups (104 folds on the corpus).  The same rule on ``group_near_duplicate_key`` and
  ``group_compilation_doi`` gives the two registered sensitivities.
* :func:`v1_grouped_folds` -- the ten grouped outer folds of the heavy arms (B5, B8, M0-M7), one set per
  discovery seed: every publication group whole, assigned by :func:`io.greedy_balance` (seeded random order,
  each group to the fold with the fewest MODEL rows so far).
* :func:`inner_folds_V1` -- grouped 3-fold on the publication groups of an outer training set (section 7).

Hidden rows: every row of the held-out group(s).  Scored rows: the hidden rows passing
:func:`io.scorable_mask` (known state, not Sr(III), not ``V6_TARGET_ROWS``).  ``row_unit`` is the row's
publication group under the fold's grouping; ``row_half`` is the registered V1 half of the row's
``group_cross_publication_copy`` unit (the group, or ``REMAINDER`` for groups below 20 rows), so a fold that
mixes halves still says which scored rows belong to which half.

Guard (section 2): ``fold_isolation_check(level="V1", near_dup_value_tol=0.005)``; the
``group_near_duplicate_key`` folds are checked value-blind (``near_dup_value_tol=None``).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from gen19ct.folds import io as FI

DESIGN = "V1"
REGISTERED_MIN_ROWS = 20
REMAINDER = "REMAINDER"
N_GROUPED = 10
N_INNER = 3
#: variant name -> grouping column of ``leakage_publication_components.csv``
GROUPINGS: dict[str, str] = {"copy": FI.GROUP_COL, "near_duplicate_key": "group_near_duplicate_key",
                             "compilation_doi": "group_compilation_doi"}


def unit_of_rows(groups: pd.Series, min_rows: int = REGISTERED_MIN_ROWS) -> pd.Series:
    """The V1 unit of each row: its group when the group has >= ``min_rows`` rows, else ``REMAINDER``."""
    sizes = groups.value_counts()
    big = set(sizes[sizes >= min_rows].index)
    return groups.where(groups.isin(big), REMAINDER)


def row_halves(frame: pd.DataFrame, halves: Mapping[str, str] | None = None) -> pd.Series:
    """Registered V1 half of every row, through its ``group_cross_publication_copy`` unit."""
    halves = FI.registered_halves("V1") if halves is None else halves
    unit = unit_of_rows(FI.publication_groups(frame, FI.GROUP_COL))
    h = unit.map(dict(halves))
    if h.isna().any():
        raise RuntimeError(f"V1 halves do not cover {int(h.isna().sum())} row(s)")
    return h


def _fold_half(halves: Iterable[str]) -> str:
    hs = set(halves)
    return next(iter(hs)) if len(hs) == 1 else "NA"


def _make(ids, hidden_mask, ok, groups, rhalf, *, variant, scheme, fold_id, seed, units, meta, design=DESIGN):
    hid = ids[hidden_mask].tolist()
    scored = [r for r in hid if r in ok]
    rh = dict(zip(hid, rhalf[hidden_mask].tolist()))
    return FI.make_fold(design=design, variant=variant, scheme=scheme, fold_id=fold_id,
                        half=_fold_half(rh[r] for r in scored) if scored else _fold_half(rh.values()), seed=seed,
                        hidden=hid, scored=scored, unit_type="publication_group", units=units,
                        row_unit=dict(zip(hid, groups[hidden_mask].tolist())), row_half=rh, meta=meta)


def v1_exact_folds(frame: pd.DataFrame, v6_mask: pd.Series, variant: str = "copy",
                   min_rows: int = REGISTERED_MIN_ROWS, halves: Mapping[str, str] | None = None) -> list[FI.Fold]:
    """Exact leave-one-group-out (+ one remainder fold) under grouping ``variant``."""
    groups = FI.publication_groups(frame, GROUPINGS[variant])
    unit = unit_of_rows(groups, min_rows)
    ids = frame[FI.ROW_ID].astype(str)
    ok = set(ids[FI.scorable_mask(frame, v6_mask).to_numpy()])
    rhalf = row_halves(frame, halves)
    order = sorted(u for u in unit.unique() if u != REMAINDER) + ([REMAINDER] if (unit == REMAINDER).any() else [])
    folds = []
    for u in order:
        mask = (unit == u).to_numpy()
        members = sorted(groups[mask].unique()) if u == REMAINDER else [u]
        folds.append(_make(ids, mask, ok, groups, rhalf, variant=variant, scheme="exact", fold_id=u, seed=None,
                           units=[u], meta={"n_groups": len(members), "groups": members}))
    return folds


def v1_grouped_folds(frame: pd.DataFrame, v6_mask: pd.Series, seed: int, n_splits: int = N_GROUPED,
                     variant: str = "copy", halves: Mapping[str, str] | None = None) -> list[FI.Fold]:
    """The ``n_splits`` grouped outer folds of one discovery seed (each group whole, seeded greedy balance)."""
    groups = FI.publication_groups(frame, GROUPINGS[variant])
    assign = FI.greedy_balance(groups.value_counts().to_dict(), n_splits, FI.seed_rng(seed, "V1_grouped"))
    ids = frame[FI.ROW_ID].astype(str)
    ok = set(ids[FI.scorable_mask(frame, v6_mask).to_numpy()])
    rhalf = row_halves(frame, halves)
    fold_of = groups.map(assign)
    folds = []
    for k in range(n_splits):
        mask = (fold_of == k).to_numpy()
        members = sorted(groups[mask].unique())
        folds.append(_make(ids, mask, ok, groups, rhalf, variant=variant, scheme=f"grouped{n_splits}",
                           fold_id=f"s{seed}_f{k}", seed=seed, units=[f"grouped_fold_{k}"],
                           meta={"n_groups": len(members), "groups": members}))
    return folds


def guard_kwargs(variant: str) -> dict:
    """The registered guard settings of a V1 variant (section 2 / 3.2)."""
    tol = None if variant == "near_duplicate_key" else FI.NEAR_DUP_VALUE_TOL
    return {"near_dup_value_tol": tol}


def guard_v1(fold: FI.Fold, frame: pd.DataFrame, universe_index: pd.Index | None = None, **override) -> dict:
    """``fold_isolation_check(level="V1")`` with the variant's tolerance.  The publication basis is the registered
    copy group (``io.publication_group_frame``; it is nested in the near-duplicate-key and compilation-DOI groupings, so
    every V1 variant's hidden set is a union of copy groups), not the raw ``g19_publication_id`` -- the same hardening
    as ``cell_holdout.guard_v5p`` (task X, leakage finding VR-02)."""
    kw = {"publication_col": FI.GROUP_COL, **guard_kwargs(fold.variant), **override}
    return FI.guard(fold, frame, frame.index if universe_index is None else universe_index, fold.hidden_row_ids,
                    "V1", **kw)


def inner_group_assignment(group_rows: Mapping[str, int], seed: int, n_inner: int = N_INNER) -> dict[str, int]:
    """Inner fold of every publication group of an outer training set -- the ONE implementation of the section 7
    grouped inner design (V1; also the V0 calibration design), shared by :func:`inner_folds_V1` and
    ``models.interface.GroupKFoldCalibration``.

    Rule (recorded in pre-registration section 7): :func:`io.greedy_balance` of the groups' outer-training row counts
    (``group_rows``) -- groups in the random order of stream ``SeedSequence([seed, 11])``, each to the inner fold
    with the fewest rows so far (ties to the lower index)."""
    return FI.greedy_balance({str(g): int(n) for g, n in group_rows.items()}, n_inner, FI.seed_rng(seed, "inner_V1"))


def inner_folds_V1(train_df: pd.DataFrame, seed: int, n_inner: int = N_INNER, *, v6_ids: Iterable[str] | None = None,
                   check: bool = True) -> list[FI.Fold]:
    """Section 7 inner folds of an outer training set: grouped ``n_inner``-fold on its publication groups
    (:func:`inner_group_assignment`).  Scored rows: :func:`io.scorable_mask` (known state, not Sr(III), not
    ``V6_TARGET_ROWS``, which follow their group).  Every returned row id belongs to ``train_df``; with ``check``
    each split passes the V1 guard inside ``train_df``."""
    groups = FI.publication_groups(train_df, FI.GROUP_COL)
    assign = inner_group_assignment(groups.value_counts().to_dict(), seed, n_inner)
    ids = train_df[FI.ROW_ID].astype(str)
    v6 = FI.v6_mask_for(train_df, v6_ids)
    ok = set(ids[FI.scorable_mask(train_df, v6).to_numpy()])
    fold_of = groups.map(assign)
    folds = []
    for k in range(n_inner):
        mask = (fold_of == k).to_numpy()
        hid = ids[mask].tolist()
        f = FI.make_fold(design="V1_inner", variant="copy", scheme=f"grouped{n_inner}", fold_id=f"inner_s{seed}_f{k}",
                         half="NA", seed=seed, hidden=hid, scored=[r for r in hid if r in ok],
                         unit_type="publication_group", units=[f"inner_fold_{k}"],
                         row_unit=dict(zip(hid, groups[mask].tolist())),
                         meta={"groups": sorted(groups[mask].unique())})
        if check:
            FI.assert_scoring_clean(f, train_df, v6)
            FI.guard(f, train_df.assign(**{FI.GROUP_COL: groups}), train_df.index, f.hidden_row_ids, "V1",
                     publication_col=FI.GROUP_COL, near_dup_value_tol=FI.NEAR_DUP_VALUE_TOL)
        folds.append(f)
    return folds

"""``folds/metal_holdout.py`` -- V2, leave-metal-out (pre-registration sections 3.3 and 7).

Units: the metal states with >= 100 MODEL rows and >= 5 extractant systems (23 on the corpus,
``feasibility_metal_states.csv`` ``v2_eligible__rows100_systems5``).  One fold per state.

* ``element`` (registered): every row of the state's ELEMENT is hidden -- all known states and the X(?) rows
  (``fold_isolation_check(level="V2", element_level=True)``).
* ``state`` (sensitivity): the state's rows and the element's X(?) rows are hidden; other known states of the
  element stay in training (``level="V2"``, the unknown-state alias rule still applies).

Scored rows: the target state's rows passing :func:`io.scorable_mask` (``V6_TARGET_ROWS`` excluded: the Pr(III)
and Nd(III) folds score only their rows outside the V6 target set).  Halves: ``feasibility_halves.csv`` design
``V2_state``.  The guard's test set is the target state's rows (the scoring population), as in
``tests/test_registered_folds.py``.

:func:`inner_metals_V2` -- section 7: leave-one-metal-out over 3 seeded training states that are themselves
V2-eligible on the outer training rows.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from gen19ct.chemistry import support_graph as SG
from gen19ct.folds import io as FI

DESIGN = "V2"
MIN_ROWS, MIN_SYSTEMS = 100, 5
N_INNER_STATES = 3


def element_of(state: str) -> str:
    return str(state).split("(")[0]


def v2_eligible_states(frame: pd.DataFrame, min_rows: int = MIN_ROWS, min_systems: int = MIN_SYSTEMS) -> list[str]:
    """Known metal states with >= ``min_rows`` rows and >= ``min_systems`` systems in ``frame`` (sorted)."""
    known = frame[frame[SG.METAL_COL].notna()]
    g = known.groupby(SG.METAL_COL).agg(n=(SG.SYSTEM_COL, "size"), s=(SG.SYSTEM_COL, "nunique"))
    return sorted(g.index[(g["n"] >= min_rows) & (g["s"] >= min_systems)].astype(str))


def hidden_mask(frame: pd.DataFrame, state: str, element_level: bool) -> pd.Series:
    el = element_of(state)
    if element_level:
        return frame[SG.ELEMENT_COL] == el
    return (frame[SG.METAL_COL] == state) | (frame[SG.METAL_COL].isna() & (frame[SG.ELEMENT_COL] == el))


def _fold(frame, state, element_level, ok, *, design, variant, fold_id, half, seed, meta=None) -> FI.Fold:
    ids = frame[FI.ROW_ID].astype(str)
    mask = hidden_mask(frame, state, element_level).to_numpy()
    target = (frame[SG.METAL_COL] == state).to_numpy()
    hid = ids[mask].tolist()
    scored = [r for r in ids[target].tolist() if r in ok]
    return FI.make_fold(design=design, variant=variant, scheme="exact" if design == DESIGN else "leave_one_state",
                        fold_id=fold_id, half=half, seed=seed, hidden=hid, scored=scored, unit_type="metal_state",
                        units=[state], meta={"element": element_of(state), "n_target_state_rows": int(target.sum()),
                                             **dict(meta or {})})


def v2_folds(frame: pd.DataFrame, v6_mask: pd.Series, element_level: bool = True,
             halves: Mapping[str, str] | None = None, states: Iterable[str] | None = None) -> list[FI.Fold]:
    halves = FI.registered_halves("V2_state") if halves is None else halves
    states = v2_eligible_states(frame) if states is None else sorted(states)
    missing = [s for s in states if s not in halves]
    if missing:
        raise RuntimeError(f"V2 halves missing for {missing}")
    ids = frame[FI.ROW_ID].astype(str)
    ok = set(ids[FI.scorable_mask(frame, v6_mask).to_numpy()])
    variant = "element" if element_level else "state"
    return [_fold(frame, st, element_level, ok, design=DESIGN, variant=variant, fold_id=st, half=halves[st], seed=None)
            for st in states]


def guard_v2(fold: FI.Fold, frame: pd.DataFrame, universe_index: pd.Index | None = None) -> dict:
    state = fold.units[0]
    test = frame.loc[frame[SG.METAL_COL] == state, FI.ROW_ID].astype(str)
    if universe_index is not None:
        test = test[test.index.isin(universe_index)]
    return FI.guard(fold, frame, frame.index if universe_index is None else universe_index, test.tolist(), "V2",
                    element_level=fold.variant == "element", near_dup_value_tol=FI.NEAR_DUP_VALUE_TOL)


def inner_state_pick(train_df: pd.DataFrame, seed: int, n_states: int = N_INNER_STATES, min_rows: int = MIN_ROWS,
                     min_systems: int = MIN_SYSTEMS) -> list[str]:
    """The inner V2 metal states of an outer training set -- the ONE implementation of the section 7 inner V2 design,
    shared by :func:`inner_metals_V2` and ``models.interface.InnerMetalCalibration``.

    Rule (recorded in pre-registration section 7): the states V2-eligible on ``train_df`` (>= ``min_rows`` known-state
    rows, >= ``min_systems`` systems; :func:`v2_eligible_states`, sorted); ``min(n_states, n_eligible)`` of them drawn
    without replacement by stream ``SeedSequence([seed, 12])``; returned sorted."""
    elig = v2_eligible_states(train_df, min_rows, min_systems)
    if not elig:
        return []
    rng = FI.seed_rng(seed, "inner_V2")
    return sorted(elig[i] for i in rng.choice(len(elig), size=min(int(n_states), len(elig)), replace=False))


def inner_metals_V2(train_df: pd.DataFrame, seed: int, n_states: int = N_INNER_STATES, *, element_level: bool = True,
                    v6_ids: Iterable[str] | None = None, check: bool = True) -> list[FI.Fold]:
    """Section 7 inner folds of an outer training set: leave-one-metal-out over the states of
    :func:`inner_state_pick`.  An inner Pr(III) or Nd(III) fold scores only its rows outside ``V6_TARGET_ROWS``."""
    elig = v2_eligible_states(train_df)
    pick = inner_state_pick(train_df, seed, n_states)
    v6 = FI.v6_mask_for(train_df, v6_ids)
    ids = train_df[FI.ROW_ID].astype(str)
    ok = set(ids[FI.scorable_mask(train_df, v6).to_numpy()])
    folds = []
    for st in pick:
        f = _fold(train_df, st, element_level, ok, design="V2_inner", variant="element" if element_level else "state",
                  fold_id=f"inner_s{seed}_{st}", half="NA", seed=seed, meta={"n_eligible_states": len(elig)})
        if check:
            FI.assert_scoring_clean(f, train_df, v6)
            guard_v2(f, train_df)
        folds.append(f)
    return folds

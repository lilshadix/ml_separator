"""``models/s1c_yardsticks.py`` -- the B3x and B3i yardsticks of section 9 S1(c), re-fitted on batched V5-PAIR folds.

Registered reading ("Same fitted folds", section 9 S1(c), resolved by the orchestrator 2026-09-15; not score-driven):
the B3x and B3i yardsticks are re-fitted on exactly the batched V5-PAIR folds the candidate (M2) is fitted on --
closed-form and cheap -- not taken from the unbatched pre-seal folds; HEAVIER needs no fit.  In discovery these are the
seed-104729 batched folds of section 3.1 (``folds/V5PAIR__primary__batched``); at confirmation the batched V5-PAIR folds
are built by the same colouring rule with each withheld seed, and every arm of the contrast is fitted on them.

:func:`refit_lookup_yardsticks`
    reads one batched V5-PAIR fold file (``folds.io.read_design``, every fold hash re-derived), refuses anything else
    (an unbatched ``cell_pair`` file, another design, a file of several seeds), and for every fold with scored rows:
    runs the caller's outer ``guard`` (the registered V5 component-aware check, :func:`registered_guard`), removes the
    fold's hidden rows from training, keeps the fold's scored rows minus ``exclude_from_scoring`` (the acidic
    co-extractant rows, as the pre-seal run), passes them through ``registered.assert_not_scored`` and the X(?) /
    Sr(III) exclusion, fits B3x and B3i (``models.baselines``, the fast path on one :class:`~gen19ct.models.interface.
    RowTable`) and predicts the scored rows.  Every prediction is labelled ``fold_id|row_id``
    (``transfer.fold_qualified_label``): a row is scored in several batched V5-PAIR folds with different training sets.
    The result names its fold design ``<stem>@<design_hash>`` (``transfer.s1c_fold_design_id``), the id the candidate's
    run must name for ``transfer.check_s1c_fold_designs`` to accept the pair.
:func:`yardstick_pair_inputs`
    regenerates the comparable test-test pairs inside each fold after fold assignment (``pairs.comparable_pairs``, which
    runs ``pair_isolation_check``), checks them against the fold builder's pair list when given, and returns the
    ``transfer.s1c_half`` inputs (pairs, fold roles, the ``V6_TARGET_ROWS`` mask by label, the B3x- / B3i-derived logSF
    and the fold designs of the pair set and both yardsticks); :func:`select_half` restricts them to one half.

No learned model is fitted here and nothing is scored: the S1(c) run happens in discovery with M2.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gen19ct.chemistry import support_graph as SG
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import pairs as EP
from gen19ct.evaluation import transfer as T
from gen19ct.folds import io as FI
from gen19ct.folds import registered as FR
from gen19ct.models import baselines as B
from gen19ct.models import interface as I

#: the arms this module re-fits (``transfer.S1C_REFIT_YARDSTICKS``)
REFIT_ARMS: tuple[str, ...] = T.S1C_REFIT_YARDSTICKS
HALF_LABEL: dict[str, str] = {"S": "selection", "C": "confirmation"}
PREDICTION_COLUMNS: tuple[str, ...] = ("label", "fold_id", "row_id", "half", "seed", "arm", "mean_logD",
                                       "fallback_level", "fallback_reason")


@dataclass(frozen=True)
class YardstickRefit:
    """B3x / B3i predictions of one batched V5-PAIR fold file.

    ``fold_design``   ``<stem>@<design_hash>`` of the fold file (``transfer.s1c_fold_design_id``)
    ``predictions``   one row per (fold, scored row, arm): :data:`PREDICTION_COLUMNS`; ``label`` = ``fold_id|row_id``
    ``fold_log``      one row per fold: status, hidden / scored counts, rows removed by ``exclude_from_scoring``"""

    stem: str
    design_hash: str
    fold_design: str
    seed: int
    arms: tuple[str, ...]
    halves: tuple[str, ...]
    predictions: pd.DataFrame = field(repr=False)
    fold_log: pd.DataFrame = field(repr=False)

    def logd(self, arm: str) -> pd.Series:
        """``fold_id|row_id`` -> predicted log D of ``arm``."""
        if arm not in self.arms:
            raise KeyError(f"{arm!r} was not re-fitted (arms {self.arms})")
        p = self.predictions[self.predictions["arm"] == arm]
        return pd.Series(p["mean_logD"].to_numpy(dtype=float), index=pd.Index(p["label"], name="label"), name=arm)


def read_batched_v5pair_design(stem_or_json: str | Path, folds_dir: Path | None = None) -> tuple[list[FI.Fold], str, str]:
    """``(folds, stem, design_hash)`` of a batched V5-PAIR fold file; every fold hash and the design hash are re-derived
    (``folds.io.read_design``).  Refuses an unbatched file, another design and a file drawn with more than one seed."""
    folds = FI.read_design(stem_or_json, folds_dir=folds_dir, verify=True)
    if not folds:
        raise ValueError(f"{stem_or_json}: no folds")
    keys = {(f.design, f.variant, f.scheme) for f in folds}
    if len(keys) != 1:
        raise ValueError(f"{stem_or_json}: one design / variant / scheme per file, got {sorted(keys)}")
    design, variant, scheme = next(iter(keys))
    stem = FI.design_stem(design, variant, scheme)
    if not T.is_batched_v5pair_design(stem):
        raise ValueError(f"S1(c) yardsticks are re-fitted on batched V5-PAIR folds only (section 9 'same fitted folds'); "
                         f"{stem} is not one")
    seeds = {f.seed for f in folds}
    if len(seeds) != 1 or None in seeds:
        raise ValueError(f"{stem}: a batched V5-PAIR file is drawn with one seed, got {sorted(map(str, seeds))}")
    return folds, stem, FI.design_hash(folds)


def registered_guard(frame: pd.DataFrame) -> Callable[[FI.Fold, pd.Index], dict]:
    """The pre-seal run's outer guard of a V5-PAIR fold: ``cell_holdout.guard_v5`` (``fold_isolation_check`` level V5,
    component-aware, value tolerance 0.005) on the slim fold frame of ``frame`` (arm frame with the archive key columns;
    ``group_cross_publication_copy`` taken from ``pub_group``).  Raises on a violation."""
    from gen19ct.folds import cell_holdout as CH

    fr = frame.copy()
    if FI.GROUP_COL not in fr.columns:
        fr[FI.GROUP_COL] = fr[I.PUB_GROUP_COL]
    slim = FI.slim_frame(fr)
    pmap = FR.parent_component_map()

    def guard(fold: FI.Fold, universe: pd.Index) -> dict:
        cmap = pmap if fold.meta.get("parent_structure") else None
        return CH.guard_v5(fold, slim, component_map=cmap, universe_index=universe)
    return guard


def _labels_of(idmap: pd.Series, ids: Iterable[str], what: str) -> pd.Index:
    ids = list(ids)
    missing = [r for r in ids if r not in idmap.index]
    if missing:
        raise AssertionError(f"{what}: {len(missing)} row id(s) absent from the frame (first {missing[0]!r})")
    return pd.Index(idmap.loc[ids].to_numpy())


def refit_lookup_yardsticks(stem_or_json: str | Path, frame: pd.DataFrame, *, v6_mask: pd.Series,
                            guard: Callable[[FI.Fold, pd.Index], Any], folds_dir: Path | None = None,
                            table: I.RowTable | None = None, systems: pd.DataFrame | None = None,
                            components: pd.DataFrame | None = None, exclude_from_scoring: pd.Series | None = None,
                            halves: Iterable[str] = ("S", "C"), arms: Iterable[str] = REFIT_ARMS) -> YardstickRefit:
    """Re-fit B3x and B3i on every fold of a batched V5-PAIR fold file (see the module docstring).

    ``frame``                 the arm frame (``interface.prepare_frame`` layout) of the MODEL rows the folds partition,
                              indexed by row label, with ``canonical_measurement_id``
    ``v6_mask``               ``V6_TARGET_ROWS`` over ``frame.index`` (every scored row passes ``assert_not_scored``)
    ``guard``                 ``(fold, universe_index) -> report``: the outer isolation check run before any fit (raises
                              on a violation; a report with ``ok`` False is refused too) -- :func:`registered_guard` for
                              the registered folds
    ``table``                 a :class:`~gen19ct.models.interface.RowTable` over ``frame`` (built when omitted)
    ``exclude_from_scoring``  rows never scored (e.g. acidic co-extractant rows), boolean over ``frame.index``
    ``halves``                the fold halves to fit (``S`` / ``C``)"""
    arms = tuple(arms)
    if not arms or not set(arms) <= set(REFIT_ARMS) or len(set(arms)) != len(arms):
        raise ValueError(f"S1(c) re-fits the lookup yardsticks {REFIT_ARMS} only, got {arms}")
    halves = tuple(halves)
    if not halves or not set(halves) <= set(HALF_LABEL):
        raise ValueError(f"halves must be among {tuple(HALF_LABEL)}")
    folds, stem, dhash = read_batched_v5pair_design(stem_or_json, folds_dir)
    if frame.index.has_duplicates or I.ID_COL not in frame.columns:
        raise ValueError(f"frame needs a unique index and a {I.ID_COL} column")
    t = I.RowTable(frame, systems=systems, components=components) if table is None else table
    if not t.index.equals(frame.index):
        raise ValueError("table must be built on frame (same index, same order)")
    idmap = pd.Series(frame.index, index=frame[I.ID_COL].astype(str))
    if idmap.index.has_duplicates:
        raise ValueError(f"{I.ID_COL} must be unique")
    v6 = v6_mask.reindex(frame.index)
    if v6.isna().any():
        raise KeyError("v6_mask must cover every frame row")
    v6 = v6.astype(bool)
    excl = (pd.Series(False, index=frame.index) if exclude_from_scoring is None
            else exclude_from_scoring.reindex(frame.index))
    if excl.isna().any():
        raise KeyError("exclude_from_scoring must cover every frame row")
    excl = excl.astype(bool)
    universe = frame.index
    preds, flog = [], []
    for f in folds:
        if f.half not in halves:
            continue
        rec = {"fold_id": f.fold_id, "half": f.half, "seed": f.seed, "n_hidden": len(f.hidden_row_ids),
               "n_scored_fold": len(f.scored_row_ids)}
        if not f.scored_row_ids:
            flog.append({**rec, "status": "no_scored_rows_not_fitted", "n_scored_used": 0, "n_excluded_from_scoring": 0})
            continue
        what = f"S1(c) yardsticks {stem}/{f.fold_id}"
        hid = _labels_of(idmap, f.hidden_row_ids, what)
        rep = guard(f, universe)
        if isinstance(rep, Mapping) and rep.get("ok") is False:
            raise AssertionError(f"{what}: outer guard failed")
        sc_all = _labels_of(idmap, f.scored_row_ids, what)
        keep = ~excl.reindex(sc_all).to_numpy(dtype=bool)
        sc = sc_all[keep]
        rec.update(n_scored_used=int(len(sc)), n_excluded_from_scoring=int((~keep).sum()))
        if not len(sc):
            flog.append({**rec, "status": "only_excluded_rows"})
            continue
        FR.assert_not_scored(sc, v6, what=what)
        st = frame.loc[sc, SG.METAL_COL]
        if st.isna().any() or (st == I.SR_III).any():
            raise AssertionError(f"{what}: an X(?) or Sr(III) row is scored")
        mask = np.ones(t.n, dtype=bool)
        mask[t.positions(hid)] = False
        ctx = I.FitContext(systems=systems, components=components, table=t, hidden_index=hid, v6_mask=v6,
                           exclude_from_scoring=excl, seed=f.seed)
        pos = t.positions(sc)
        ids = t.ids[pos]
        for arm in arms:
            p = B.BaselineArm(arm).fit_table(t, mask, ctx).predict_positions(pos)
            if not np.isfinite(p["mean_logD"].to_numpy(dtype=float)).all():
                raise AssertionError(f"{what}/{arm}: non-finite prediction")
            preds.append(pd.DataFrame({
                "label": [T.fold_qualified_label(f.fold_id, r) for r in ids], "fold_id": f.fold_id, "row_id": ids,
                "half": HALF_LABEL[f.half], "seed": int(f.seed), "arm": arm,
                "mean_logD": p["mean_logD"].to_numpy(dtype=float), "fallback_level": p["fallback_level"].to_numpy(),
                "fallback_reason": p["fallback_reason"].to_numpy()}))
        flog.append({**rec, "status": "fitted"})
    if not preds:
        raise ValueError(f"{stem}: no fold with scored rows in halves {halves}")
    pred = (pd.concat(preds, ignore_index=True).sort_values(["fold_id", "row_id", "arm"], kind="mergesort")
            .reset_index(drop=True))
    if pred.duplicated(["label", "arm"]).any():
        raise AssertionError(f"{stem}: a fold scores a row twice")
    return YardstickRefit(stem=stem, design_hash=dhash, fold_design=T.s1c_fold_design_id(stem, dhash),
                          seed=int(folds[0].seed), arms=arms, halves=halves, predictions=pred[list(PREDICTION_COLUMNS)],
                          fold_log=pd.DataFrame(flog))


def yardstick_pair_inputs(refit: YardstickRefit, attrs: pd.DataFrame, *, v6_mask: pd.Series,
                          builder_pairs: pd.DataFrame | None = None) -> dict[str, Any]:
    """The ``transfer.s1c_half`` inputs of a :class:`YardstickRefit` (both halves; :func:`select_half` picks one).

    ``attrs``          per MODEL row, indexed by ``canonical_measurement_id``: ``publication_group``,
                       ``extractant_system_key``, ``condition_key`` (the section 2 pair key), ``g19_metal_state`` and
                       ``log_D``
    ``v6_mask``        ``V6_TARGET_ROWS`` by ``canonical_measurement_id``
    ``builder_pairs``  the fold builder's pair list (``fold_id``, ``row_id_a``, ``row_id_b``; e.g.
                       ``folds/V5PAIR__primary__batched__pairs.parquet``): the regenerated pairs of the fitted folds must
                       equal it

    Returns ``pairs`` (fold-qualified ``idx_a`` / ``idx_b``, ``fold``, key and cell-pair columns, ``logsf_obs``,
    ``row_id_a`` / ``row_id_b``, ``half``), ``folds`` (label -> ``test``), ``v6_mask`` (label -> bool), ``lookup_logsf``
    (arm -> derived logSF on ``pairs``), ``fold_designs`` (``pairs`` and each arm -> ``refit.fold_design``; the caller
    adds ``candidate``) and ``rows`` (the labelled scored rows)."""
    need = list(EP.PAIR_KEY_COLS) + [EM.METAL_STATE_COL, EM.Y_COL]
    missing = [c for c in need if c not in attrs.columns]
    if missing:
        raise KeyError(f"attrs lacks {missing}")
    if attrs.index.has_duplicates:
        raise ValueError("attrs must be indexed by unique canonical_measurement_id")
    p = refit.predictions
    rows = p[p["arm"] == refit.arms[0]][["label", "fold_id", "row_id", "half"]]
    if not rows["row_id"].isin(attrs.index).all():
        raise KeyError("attrs does not cover every scored row")
    df = rows.join(attrs[need], on="row_id")
    df.index = pd.Index(df["label"].to_numpy(), name="label")
    df["fold"] = df["fold_id"]
    pairs = EP.comparable_pairs(df, fold_col="fold", carry_cols=("row_id", "half"))
    if (pairs["half_a"] != pairs["half_b"]).any():
        raise AssertionError("a pair joins rows of two halves")
    pairs = pairs.rename(columns={"half_a": "half"}).drop(columns=["half_b"])
    if builder_pairs is not None:
        bp = builder_pairs[builder_pairs["fold_id"].isin(set(rows["fold_id"]))]
        got = sorted(zip(pairs["fold"], (tuple(sorted(x)) for x in zip(pairs["row_id_a"], pairs["row_id_b"]))))
        want = sorted(zip(bp["fold_id"], (tuple(sorted(x)) for x in zip(bp["row_id_a"].astype(str),
                                                                          bp["row_id_b"].astype(str)))))
        if got != want:
            raise AssertionError(f"S1(c) yardstick pairs: {len(got)} regenerated != {len(want)} from the fold builder")
    v6 = df["row_id"].map(v6_mask)
    if v6.isna().any():
        raise KeyError("v6_mask does not cover every scored row")
    return {"pairs": pairs, "folds": pd.Series("test", index=df.index, dtype=object),
            "v6_mask": pd.Series(v6.to_numpy(dtype=bool), index=df.index),
            "lookup_logsf": {arm: EP.derived_logsf(pairs, refit.logd(arm)) for arm in refit.arms},
            "fold_designs": {"pairs": refit.fold_design, **{arm: refit.fold_design for arm in refit.arms}},
            "rows": df}


def select_half(inputs: Mapping[str, Any], half: str) -> dict[str, Any]:
    """:func:`yardstick_pair_inputs` restricted to the pairs of one half (``selection`` / ``confirmation``)."""
    if half not in T.S1C_HALVES:
        raise ValueError(f"half must be one of {T.S1C_HALVES}")
    sel = (inputs["pairs"]["half"] == half).to_numpy(dtype=bool)
    return {**inputs, "pairs": inputs["pairs"][sel],
            "lookup_logsf": {k: v[sel] for k, v in inputs["lookup_logsf"].items()}}

"""Fold-local preprocessing.  Nothing here ever sees a test row.

Every transformer is constructed fresh per fold, fitted on that fold's training
rows and then applied to the test rows.  There is no module-level state, no
cached scaler and no whole-cohort statistic, so the class of failure the
repository's leakage audit calls "preprocessing before group splitting" cannot
occur by construction rather than by convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Prediction clip, pre-registered: the training target range plus one decade each side.
CLIP_MARGIN = 1.0


def as_float(frame: pd.DataFrame, columns) -> np.ndarray:
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def extractant_balanced_weights(extractants: pd.Series) -> np.ndarray:
    """Equal total mass per training extractant — the metric the study reports."""
    counts = extractants.astype(str).value_counts()
    weights = extractants.astype(str).map(lambda e: 1.0 / float(counts[e])).to_numpy(dtype=float)
    return weights * (len(weights) / weights.sum())


@dataclass
class FoldPreprocessor:
    """Median imputation with a missingness indicator, optionally standardised.

    ``add_indicator`` matters here rather than being a default: a missing contact
    time and a missing metal concentration are 38 % and 56 % of the cohort, and a
    median fill without a flag tells the model those rows were measured.
    """

    columns: tuple[str, ...]
    standardise: bool = False
    add_indicator: bool = True
    drop_constant: bool = True
    #: Clamp a value outside the training fold's observed range to that range before
    #: standardising.  Necessary, not cosmetic, and only for models that *multiply*
    #: their inputs.  ``lig2d__rd__Ipc`` spans 1.1e8 to 3.8e29 across these ligands;
    #: when the extreme molecule is held out, the training sd is 5.3e12 and the
    #: held-out value standardises to 7.3e16.  A tree only compares and is immune; a
    #: network multiplied it and diverged in 8 of 25 folds of one arm, which the
    #: prediction clip then partly hid.  Clamping touches only out-of-range values,
    #: so binary and in-range columns are unchanged.
    clip_to_train_range: bool = False
    medians_: np.ndarray | None = field(default=None, init=False)
    keep_: np.ndarray | None = field(default=None, init=False)
    indicator_: np.ndarray | None = field(default=None, init=False)
    centre_: np.ndarray | None = field(default=None, init=False)
    scale_: np.ndarray | None = field(default=None, init=False)
    lower_: np.ndarray | None = field(default=None, init=False)
    upper_: np.ndarray | None = field(default=None, init=False)
    names_: tuple[str, ...] = field(default=(), init=False)

    def fit(self, train: pd.DataFrame) -> "FoldPreprocessor":
        raw = as_float(train, self.columns)
        observed = np.isfinite(raw)
        # A column can be entirely missing *inside a training fold* even though the
        # cohort observes it — this is the failure that crashed a gen5 run outright.
        # The median is computed only where there is something to take a median of;
        # the rest are set to 0.0 and then dropped as constant, which is exactly what
        # the previous `nanmedian` path did, minus the All-NaN RuntimeWarning.
        medians = np.zeros(raw.shape[1], dtype=float)
        for column in np.flatnonzero(observed.any(axis=0)):
            medians[column] = float(np.median(raw[observed[:, column], column]))
        filled = np.where(observed, raw, medians)
        keep = np.ones(filled.shape[1], dtype=bool)
        if self.drop_constant:
            keep = filled.std(axis=0) > 0
            if not keep.any():
                keep = np.ones(filled.shape[1], dtype=bool)
        indicator = observed.mean(axis=0) < 1.0 if self.add_indicator else np.zeros(
            filled.shape[1], dtype=bool)
        self.medians_, self.keep_, self.indicator_ = medians, keep, indicator
        block = filled[:, keep]
        if self.clip_to_train_range:
            self.lower_, self.upper_ = block.min(axis=0), block.max(axis=0)
        if self.standardise:
            self.centre_ = block.mean(axis=0)
            scale = block.std(axis=0)
            self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        names = [c for c, k in zip(self.columns, keep) if k]
        names += [f"missing__{c}" for c, i in zip(self.columns, indicator) if i]
        self.names_ = tuple(names)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None:
            raise RuntimeError("FoldPreprocessor.transform before fit")
        raw = as_float(frame, self.columns)
        observed = np.isfinite(raw)
        filled = np.where(observed, raw, self.medians_)
        block = filled[:, self.keep_]
        if self.clip_to_train_range:
            block = np.clip(block, self.lower_, self.upper_)
        if self.standardise:
            block = (block - self.centre_) / self.scale_
        if self.indicator_.any():
            block = np.hstack([block, (~observed[:, self.indicator_]).astype(float)])
        return block

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)


def clip_to_training_range(prediction: np.ndarray, y_train: np.ndarray,
                           margin: float = CLIP_MARGIN) -> np.ndarray:
    low = float(np.min(y_train)) - margin
    high = float(np.max(y_train)) + margin
    return np.clip(np.asarray(prediction, dtype=float), low, high)

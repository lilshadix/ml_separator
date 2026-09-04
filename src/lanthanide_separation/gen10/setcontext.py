"""Phase 3 — a learned, permutation-equivariant representation of the query set.

gen9 reduced "where does this point sit in the design" to five handcrafted columns,
and Phase 1 measures how much those columns depend on points the user never
intended to measure.  The alternative the brief asks for is to hand the model the
*whole* requested design and let it learn what to summarise.  This is the smallest
version of that idea that is still a set function:

.. code-block:: text

    e_i = MLP(absolute condition_i)                point embedding
    g   = pool_j e_j                                set summary (mean, or attention)
    z_i = [e_i, g, absolute condition_i]
    s_i = MLP(z_i) - mean_j MLP(z_j)                zero-mean shape over the curve

The shape head replaces the *shape forest* of :class:`RecomposedModel`: the static
chemistry still comes from the frozen monolith (the level), and the set encoder
predicts only the curve-centred residual from the curve's own coordinates.  That
keeps the comparison clean — handcrafted columns + forest against learned set
representation + MLP, with identical level, identical folds, identical rows —
and it keeps the encoder tiny, because it only ever sees one axis coordinate per
point plus the held axes of its curve.

Invariants, each tested:

* permutation equivariance: reordering the points permutes ``s``;
* variable pool size: any number of points;
* no target access: ``predict`` takes conditions only;
* determinism: seeded, single-threaded, deterministic algorithms.

A learned encoder earns inclusion only on the brief's bar — macro MAE >= 0.01
better on the locked evaluation, or shape MAE substantially better with macro
neutral, *and* better query-set consistency.  It is built to be measured, not to
win.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..gen7.harness import FoldContext
from ..gen9.curves import DEFAULT_AXES
from ..gen9.train import FROZEN_BLOCKS
from ..levels import group_balanced_weights
from .architectures import (
    CLIP_MULTIPLIER, DesignBuilder, _curve_means, _curve_of, _forest, _recompose, stabilise,
)
from .querycurves import primary_curve, query_membership, window_statistics

#: Per-point inputs the encoder reads: the point's coordinate on its curve's axis
#: (log10 molarity or lanthanide index) and which axis that is.  Nothing relative.
POINT_FEATURES = 4   # axis_value, is_extractant, is_acid, is_metal


def _torch():
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def point_matrix(frame: pd.DataFrame, membership: pd.DataFrame,
                 axes: Sequence[str] = DEFAULT_AXES) -> tuple[np.ndarray, np.ndarray]:
    """``(points, curve_index)``: absolute per-point inputs and each row's curve id.

    Rows on no curve get a curve index of -1 and a zero feature row; the encoder
    never sees them and they keep the monolith's prediction.
    """
    primary = primary_curve(membership, axes=axes)
    n = len(frame)
    points = np.zeros((n, POINT_FEATURES), dtype=np.float32)
    curve = np.full(n, -1, dtype=int)
    if primary.empty:
        return points, curve
    lookup = primary.set_index("row_id")
    row_ids = frame["row_id"].astype(str).to_numpy()
    present = lookup.reindex(row_ids)
    ids = {c: i for i, c in enumerate(sorted(present["curve_id"].dropna().unique()))}
    for i, (cid, value, label) in enumerate(zip(present["curve_id"], present["axis_value"],
                                                present["axis_label"])):
        if cid is None or (isinstance(cid, float) and np.isnan(cid)):
            continue
        curve[i] = ids[cid]
        points[i, 0] = float(value)
        points[i, 1] = 1.0 if label == "extractant" else 0.0
        points[i, 2] = 1.0 if label == "acid" else 0.0
        points[i, 3] = 1.0 if label == "metal_series" else 0.0
    return points, curve


class _DeepSetsShape:
    """Tiny DeepSets head; ``forward`` returns a zero-mean shape per curve."""

    def __init__(self, *, static_dim: int, width: int, pooling: str, seed: int):
        torch = _torch()
        torch.manual_seed(seed)
        self.pooling = pooling
        self.phi = torch.nn.Sequential(torch.nn.Linear(POINT_FEATURES, width), torch.nn.Tanh(),
                                       torch.nn.Linear(width, width), torch.nn.Tanh())
        if pooling == "attention":
            self.score = torch.nn.Linear(width, 1)
        self.static = torch.nn.Sequential(torch.nn.Linear(static_dim, width), torch.nn.Tanh())
        self.rho = torch.nn.Sequential(torch.nn.Linear(3 * width + POINT_FEATURES, width),
                                       torch.nn.Tanh(), torch.nn.Linear(width, 1))

    def parameters(self):
        modules = [self.phi, self.static, self.rho] + ([self.score] if self.pooling == "attention" else [])
        for module in modules:
            yield from module.parameters()

    def forward(self, points, static, curve):
        """``points (n, 4)``, ``static (n, d)``, ``curve (n,)`` int >= 0 for curve rows."""
        torch = _torch()
        e = self.phi(points)
        n_curves = int(curve.max().item()) + 1 if len(curve) else 0
        onehot = torch.nn.functional.one_hot(curve.clamp(min=0), num_classes=max(n_curves, 1)).float()
        onehot = onehot * (curve >= 0).float()[:, None]
        if self.pooling == "attention":
            logits = self.score(e).squeeze(-1)
            logits = logits - logits.max()
            weights = torch.exp(logits)[:, None] * onehot
            weights = weights / weights.sum(0, keepdim=True).clamp(min=1e-9)
            g = weights.T @ e
        else:
            counts = onehot.sum(0, keepdim=True).clamp(min=1.0)
            g = (onehot.T @ e) / counts.T
        g_i = onehot @ g
        z = torch.cat([e, g_i, self.static(static), points], dim=1)
        raw = self.rho(z).squeeze(-1)
        # exact zero mean over each curve
        sums = onehot.T @ raw[:, None]
        counts = onehot.sum(0)[:, None].clamp(min=1.0)
        mean_i = (onehot @ (sums / counts)).squeeze(-1)
        return (raw - mean_i) * (curve >= 0).float()


@dataclass
class _FittedSetRecomposed:
    design: DesignBuilder
    axes: tuple[str, ...]
    lo: float
    hi: float
    base: object = None
    head: object = None
    static_scale: tuple = ()
    static_columns: np.ndarray | None = None

    def _static(self, x: np.ndarray):
        torch = _torch()
        sub = x[:, self.static_columns]
        mean, sd = self.static_scale
        return torch.tensor((sub - mean) / sd, dtype=torch.float32)

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        torch = _torch()
        membership = query_membership(query, axes=self.axes)
        static = self.design.transform(query)
        base = np.clip(np.asarray(self.base.predict(static), dtype=float), self.lo, self.hi)
        points, curve = point_matrix(query, membership, self.axes)
        if (curve >= 0).sum() == 0:
            return base
        with torch.no_grad():
            shape = self.head.forward(torch.tensor(points), self._static(static),
                                      torch.tensor(curve)).numpy().astype(float)
        curve_ids = _curve_of(query, membership, self.axes)
        return np.clip(_recompose(base, shape, curve_ids), self.lo, self.hi)

    def set_predict_jobs(self, n_jobs: int) -> "_FittedSetRecomposed":
        if hasattr(self.base, "n_jobs"):
            self.base.n_jobs = int(n_jobs)
        return self


@dataclass
class SetRecomposedModel:
    """``RecomposedModel`` with the shape forest replaced by a DeepSets head.

    ``pooling`` is ``"mean"`` (DeepSets) or ``"attention"`` (one small attention
    block) — the two designs the brief allows, and no more.  ``static_top`` static
    columns — the ones the monolith splits on most — are handed to the head so the
    shape can depend on the chemistry without the head having to see 2,160 columns.
    """

    name: str = "GEN10_SET_RECOMPOSED"
    pooling: str = "mean"
    width: int = 32
    epochs: int = 300
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    static_top: int = 32
    #: The one fitted knob: the epoch count.  A full-batch MLP on 4,900 rows drives
    #: its training loss to a tenth of the target variance and generalises worse
    #: than the frozen model; choosing *when to stop* on held-out training
    #: chemotypes (one inner split, seeded) is the smallest honest repair and is
    #: not a search over architecture.
    inner_validation: bool = True
    inner_fraction: float = 0.2
    blocks: tuple[str, ...] = FROZEN_BLOCKS
    add_indicator: bool = True
    n_estimators: int = 400
    max_features: object = 0.30
    min_samples_leaf: int = 2
    axes: tuple[str, ...] = DEFAULT_AXES
    diagnostics: list = field(default_factory=list)

    def fit(self, train: pd.DataFrame, y_train: np.ndarray,
            context: FoldContext) -> _FittedSetRecomposed:
        torch = _torch()
        if self.pooling not in ("mean", "attention"):
            raise ValueError(f"unknown pooling {self.pooling!r}")
        design = DesignBuilder(blocks=tuple(self.blocks),
                               add_indicator=self.add_indicator).fit(context.cohort, train)
        spread = float(y_train.max() - y_train.min())
        fitted = _FittedSetRecomposed(
            design=design, axes=tuple(self.axes),
            lo=float(y_train.min() - CLIP_MULTIPLIER * spread),
            hi=float(y_train.max() + CLIP_MULTIPLIER * spread))
        weights = group_balanced_weights(train["ecfp_cluster"])
        static = design.transform(train)
        base = _forest(context.model_seed, n_estimators=self.n_estimators,
                       max_features=self.max_features, min_samples_leaf=self.min_samples_leaf)
        base.fit(static, y_train, sample_weight=weights)
        importance = np.asarray(base.feature_importances_)
        top = np.argsort(-importance)[: self.static_top]
        fitted.static_columns = np.sort(top)
        sub = static[:, fitted.static_columns]
        mean, sd = sub.mean(0), sub.std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        fitted.static_scale = (mean, sd)

        membership = query_membership(train, axes=tuple(self.axes))
        curve_ids = _curve_of(train, membership, tuple(self.axes))
        centred = y_train - _curve_means(curve_ids, y_train)
        points, curve = point_matrix(train, membership, tuple(self.axes))
        on = curve >= 0
        target = torch.tensor(np.where(on, centred, 0.0), dtype=torch.float32)
        w = torch.tensor(np.where(on, weights, 0.0), dtype=torch.float32)
        w = w / w.sum()

        p_t, s_t, c_t = torch.tensor(points), fitted._static(static), torch.tensor(curve)

        def train_head(mask_train: np.ndarray, n_epochs: int, mask_valid: np.ndarray | None):
            head = _DeepSetsShape(static_dim=len(fitted.static_columns), width=self.width,
                                  pooling=self.pooling, seed=context.model_seed)
            optimiser = torch.optim.Adam(head.parameters(), lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
            w_train = w * torch.tensor(mask_train, dtype=torch.float32)
            w_train = w_train / w_train.sum().clamp(min=1e-12)
            w_valid = None
            if mask_valid is not None:
                w_valid = w * torch.tensor(mask_valid, dtype=torch.float32)
                w_valid = w_valid / w_valid.sum().clamp(min=1e-12)
            trace = []
            for epoch in range(n_epochs):
                optimiser.zero_grad()
                out = head.forward(p_t, s_t, c_t)
                loss = (w_train * (out - target) ** 2).sum()
                loss.backward()
                optimiser.step()
                if w_valid is not None and (epoch % 10 == 9):
                    with torch.no_grad():
                        valid = float((w_valid * (head.forward(p_t, s_t, c_t) - target) ** 2).sum())
                    trace.append({"epoch": epoch + 1, "train": float(loss.item()), "valid": valid})
            return head, trace

        chosen_epochs, trace = self.epochs, []
        if self.inner_validation:
            # hold out a fifth of the training chemotypes, deterministically
            clusters = train["tanimoto_cluster"].astype(str).to_numpy()
            unique = np.array(sorted(set(clusters)))
            rng = np.random.default_rng(context.model_seed)
            held = set(rng.choice(unique, size=max(1, int(len(unique) * self.inner_fraction)),
                                  replace=False))
            mask_valid = np.isin(clusters, list(held)) & on
            mask_train = (~np.isin(clusters, list(held))) & on
            _, trace = train_head(mask_train, self.epochs, mask_valid)
            if trace:
                best = min(trace, key=lambda t: (t["valid"], t["epoch"]))
                chosen_epochs = int(best["epoch"])
        head, _ = train_head(on, chosen_epochs, None)
        fitted.base, fitted.head = base, head
        self.diagnostics.append({"arm": self.name, "split_seed": context.fold.seed,
                                 "fold": context.fold.fold, "pooling": self.pooling,
                                 "n_curve_rows": int(on.sum()), "chosen_epochs": chosen_epochs,
                                 "validation_trace": trace[::5]})
        return fitted

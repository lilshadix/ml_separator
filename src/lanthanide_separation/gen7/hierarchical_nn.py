"""The offset/shape network: predict a ligand's level and its response separately.

gen5 and gen6 both measured the same split failure — on a held-out chemotype the
model gets the *shape* of the response across metals and conditions roughly right
(shape MAE ~0.50) and the *level* badly wrong (offset MAE ~0.87, and 70 % of the
squared error).  A single regression head has no way to be told that those are two
problems; this module gives them two heads and three extra losses.

.. code-block:: text

    z_L = LigandEncoder(x_L)        x_L is constant within a ligand
    z_M = MetalEncoder(x_M)         Z, lanthanide index, ionic radius (+ RBF)
    z_C = ConditionEncoder(x_C)     cond__*, mass-action logs, solvent physics

    mu  = OffsetHead(z_L)                       the ligand's absolute level
    r   = ResponseHead(z_L, z_M, z_C)           its response to the experiment
    y   = mu + r

Because ``x_L`` is constant within a ligand, ``mu`` is *exactly* constant within a
ligand — the decomposition cannot drift row by row.  The remaining degeneracy is
that ``r`` could absorb a per-ligand constant and hollow out ``mu``; ``L_center``
forbids it by driving each training ligand's mean response to zero.

Four losses, weighted by a searched ``lambda``:

``L_abs``   Huber on the absolute target.  The thing we are actually scored on.
``L_shape`` Huber between ``r`` and ``y - mean_train(y | ligand)``.  The centring
            mean is computed **only from the training rows of the fold** — using
            all rows would leak a held-out ligand's level, which is precisely the
            quantity under test.
``L_pair``  Huber on differences.  Two pair families, and the second is the one
            that matters here:

            * *within-ligand* pairs supervise the shape relationally;
            * *same-condition, different-ligand* pairs supervise the **offset
              difference** — a log-SF-like contrast, the quantity gen2–gen4 showed
              is far better determined than either absolute level.  One row
              therefore teaches the model about level *contrasts* as well as about
              its own value, which is extra supervision from measurements we
              already own.
``L_center`` ``mean_{i in l} r_i`` squared, per training ligand.  Keeps the two
            heads from trading places.

The conditioning between ligand and experiment is a choice, not a given, so
``interaction`` selects it: ``concat`` (the null), ``film`` (the experiment scales
and shifts the ligand code), ``bilinear`` (explicit ``z_L^T W z_M`` products) or
``moe`` (a small mixture of experts gated by the ligand code — different extraction
mechanisms get different condition responses).

Early stopping uses an inner split that holds out **whole chemotypes** of the
training fold.  A random inner split would let a near-duplicate ligand sit on both
sides and choose the epoch that overfits chemistry, which is the failure this
whole generation exists to measure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..levels import _as_float_frame, group_balanced_weights
from .harness import FoldContext


# --------------------------------------------------------------------------- #
# Metal representation (brief §4)
# --------------------------------------------------------------------------- #

#: Shannon ionic radii (Å) for Ln(III) at coordination number 8 — the coordination
#: number the bundle's Architector complexes actually use.  The lanthanide
#: contraction across the series is the single most physical fact about this axis
#: and a 14-way one-hot throws it away.
IONIC_RADII_CN8: dict[str, float] = {
    "La": 1.160, "Ce": 1.143, "Pr": 1.126, "Nd": 1.109, "Pm": 1.093, "Sm": 1.079,
    "Eu": 1.066, "Gd": 1.053, "Tb": 1.040, "Dy": 1.027, "Ho": 1.015, "Er": 1.004,
    "Tm": 0.994, "Yb": 0.985, "Lu": 0.977, "Y": 1.019,
}
#: Number of 4f electrons in Ln(III) — drives the tetrad effect and the
#: half-filled-shell discontinuity at Gd.
F_ELECTRONS: dict[str, int] = {
    "La": 0, "Ce": 1, "Pr": 2, "Nd": 3, "Pm": 4, "Sm": 5, "Eu": 6, "Gd": 7,
    "Tb": 8, "Dy": 9, "Ho": 10, "Er": 11, "Tm": 12, "Yb": 13, "Lu": 14, "Y": 0,
}


def metal_physical_features(symbols: Sequence[str], *, n_rbf: int = 8) -> np.ndarray:
    """Continuous, physically meaningful coordinates for a lanthanide.

    Columns: ionic radius, radius², normalised series position, position²,
    4f count, 4f count normalised, distance from the half-filled shell (|n_f − 7|,
    the tetrad-effect coordinate), plus ``n_rbf`` Gaussian bumps over the radius.
    The RBF expansion is what lets the network share strength between neighbouring
    metals instead of learning 14 unrelated categories, while still being able to
    bend sharply at Gd.
    """
    radii = np.array([IONIC_RADII_CN8.get(str(s), np.nan) for s in symbols], dtype=float)
    nf = np.array([F_ELECTRONS.get(str(s), np.nan) for s in symbols], dtype=float)
    lo, hi = 0.97, 1.17
    position = (radii - lo) / (hi - lo)
    centres = np.linspace(lo, hi, n_rbf)
    width = (hi - lo) / max(1, n_rbf - 1)
    rbf = np.exp(-((radii[:, None] - centres[None, :]) ** 2) / (2 * width ** 2))
    base = np.column_stack([
        radii, radii ** 2, position, position ** 2,
        nf, nf / 14.0, np.abs(nf - 7.0), (nf - 7.0) ** 2 / 49.0,
    ])
    return np.nan_to_num(np.column_stack([base, rbf]), nan=0.0)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class HierarchicalConfig:
    """Everything the network and its objective are frozen with."""

    ligand_blocks: tuple[str, ...] = ("DONORS", "PHYSCHEM")
    condition_blocks: tuple[str, ...] = ("COND", "MASSACTION")
    #: extra ligand feature columns supplied outside the block system (pretrained
    #: embeddings, 3D descriptors); resolved by name against the frame.
    extra_ligand_columns: tuple[str, ...] = ()
    extra_condition_columns: tuple[str, ...] = ()

    ligand_dim: int = 64
    metal_dim: int = 32
    condition_dim: int = 64
    hidden: int = 128
    depth: int = 2
    dropout: float = 0.15
    interaction: str = "film"          # concat | film | bilinear | moe
    n_experts: int = 4
    #: Physics-guided response (brief §8).  Instead of learning the whole response
    #: from scratch, the network predicts the *coefficients of the mass-action law*
    #: from the ligand and metal, and only the deviation from that law is learned::
    #:
    #:     r = n̂ · (log[L] − log[L]₀) + m̂ · (log[H] − log[H]₀) + g(z_L, z_M, z_C)
    #:
    #: ``n̂`` is the solvation number and ``m̂`` the acid-dependence slope, both
    #: bounded to chemically admissible ranges by a scaled sigmoid.  The measured
    #: slopes on this cohort are n ≈ 2.64 (IQR 2.36–2.88) and m ≈ 1.93, and the
    #: bounds bracket those comfortably rather than pinning them.  ``g`` is free, so
    #: the law is a *prior on the functional form*, never a constraint on the answer.
    physics_head: bool = False
    physics_n_range: tuple[float, float] = (0.0, 6.0)
    physics_m_range: tuple[float, float] = (-6.0, 6.0)

    lambda_abs: float = 1.0
    lambda_shape: float = 1.0
    lambda_pair: float = 0.5
    lambda_center: float = 1.0
    pair_mode: str = "both"            # none | within | across | both
    pairs_per_step: int = 512
    huber_delta: float = 1.0

    epochs: int = 400
    patience: int = 40
    batch_size: int = 256
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    inner_holdout_chemotypes: float = 0.2
    n_ensemble: int = 3                # average this many random inits per fold
    predict_offset_shape: bool = True


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #

#: Raw (unstandardised) log-concentration columns the physics head multiplies.
#: They must stay in log10 molarity for ``n̂`` and ``m̂`` to be readable as a
#: solvation number and an acid order rather than as arbitrary weights.
PHYSICS_COLUMNS: tuple[str, ...] = (
    "massact__log10_cond__extractant_concentration_M",
    "massact__log10_cond__acid_concentration_M",
)


def _make_network(config: HierarchicalConfig, d_ligand: int, d_metal: int, d_condition: int):
    import torch
    from torch import nn

    def mlp(d_in: int, d_out: int, hidden: int, depth: int, dropout: float) -> nn.Module:
        layers: list[nn.Module] = []
        d = d_in
        for _ in range(max(1, depth)):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, d_out))
        return nn.Sequential(*layers)

    class Hierarchical(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = config
            self.ligand = mlp(d_ligand, c.ligand_dim, c.hidden, c.depth, c.dropout)
            self.metal = mlp(d_metal, c.metal_dim, c.metal_dim * 2, 1, c.dropout)
            self.condition = mlp(d_condition, c.condition_dim, c.hidden, c.depth, c.dropout)
            # The offset head sees ONLY the ligand: a level that could look at the
            # conditions would stop being a level.
            self.offset = nn.Sequential(
                nn.Linear(c.ligand_dim, c.hidden), nn.SiLU(), nn.Dropout(c.dropout),
                nn.Linear(c.hidden, 1))
            context = c.metal_dim + c.condition_dim
            self.interaction = c.interaction
            if c.interaction == "film":
                self.film = nn.Linear(context, 2 * c.ligand_dim)
                response_in = c.ligand_dim + context
            elif c.interaction == "bilinear":
                rank = 16
                self.left = nn.Linear(c.ligand_dim, rank, bias=False)
                self.right = nn.Linear(context, rank, bias=False)
                response_in = c.ligand_dim + context + rank
            elif c.interaction == "moe":
                self.gate = nn.Linear(c.ligand_dim, c.n_experts)
                self.experts = nn.ModuleList([
                    mlp(c.ligand_dim + context, 1, c.hidden, c.depth, c.dropout)
                    for _ in range(c.n_experts)])
                response_in = None
            elif c.interaction == "concat":
                response_in = c.ligand_dim + context
            else:
                raise ValueError(f"unknown interaction {c.interaction!r}")
            if response_in is not None:
                self.response = mlp(response_in, 1, c.hidden, c.depth, c.dropout)
            self.physics_head = c.physics_head
            if c.physics_head:
                # coefficients depend on the LIGAND and the METAL, never on the
                # concentrations they multiply — otherwise "the slope" would just be
                # another free function of the same inputs and the prior would be vacuous
                self.coefficients = nn.Sequential(
                    nn.Linear(c.ligand_dim + c.metal_dim, c.hidden), nn.SiLU(),
                    nn.Linear(c.hidden, 2))
                self.register_buffer("n_lo", torch.tensor(float(c.physics_n_range[0])))
                self.register_buffer("n_hi", torch.tensor(float(c.physics_n_range[1])))
                self.register_buffer("m_lo", torch.tensor(float(c.physics_m_range[0])))
                self.register_buffer("m_hi", torch.tensor(float(c.physics_m_range[1])))

        def forward(self, x_ligand, x_metal, x_condition, physics=None):
            z_l = self.ligand(x_ligand)
            z_m = self.metal(x_metal)
            z_c = self.condition(x_condition)
            context = torch.cat([z_m, z_c], dim=-1)
            mu = self.offset(z_l).squeeze(-1)
            if self.interaction == "film":
                gamma, beta = self.film(context).chunk(2, dim=-1)
                modulated = (1.0 + gamma) * z_l + beta
                r = self.response(torch.cat([modulated, context], dim=-1)).squeeze(-1)
            elif self.interaction == "bilinear":
                product = self.left(z_l) * self.right(context)
                r = self.response(torch.cat([z_l, context, product], dim=-1)).squeeze(-1)
            elif self.interaction == "moe":
                weights = torch.softmax(self.gate(z_l), dim=-1)
                joint = torch.cat([z_l, context], dim=-1)
                stacked = torch.stack([e(joint).squeeze(-1) for e in self.experts], dim=-1)
                r = (weights * stacked).sum(-1)
            else:
                r = self.response(torch.cat([z_l, context], dim=-1)).squeeze(-1)
            if self.physics_head and physics is not None:
                raw = self.coefficients(torch.cat([z_l, z_m], dim=-1))
                n_hat = self.n_lo + (self.n_hi - self.n_lo) * torch.sigmoid(raw[:, 0])
                m_hat = self.m_lo + (self.m_hi - self.m_lo) * torch.sigmoid(raw[:, 1])
                r = r + n_hat * physics[:, 0] + m_hat * physics[:, 1]
            return mu, r

    return Hierarchical()


# --------------------------------------------------------------------------- #
# Contender
# --------------------------------------------------------------------------- #

@dataclass
class HierarchicalNet:
    """The offset/shape network as a gen7 contender."""

    config: HierarchicalConfig = field(default_factory=HierarchicalConfig)
    name: str = "HNN"

    # -- feature assembly ---------------------------------------------------- #
    def _columns(self, context: FoldContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
        cohort = context.cohort
        ligand = list(cohort.block_columns(
            tuple(b for b in self.config.ligand_blocks if b in cohort.blocks)))
        ligand += [c for c in self.config.extra_ligand_columns if c not in ligand]
        condition = list(cohort.block_columns(
            tuple(b for b in self.config.condition_blocks if b in cohort.blocks)))
        condition += [c for c in self.config.extra_condition_columns if c not in condition]
        return tuple(ligand), tuple(condition)

    def _matrices(self, frame: pd.DataFrame, ligand_columns, condition_columns):
        x_l = _as_float_frame(frame, ligand_columns).to_numpy()
        x_c = _as_float_frame(frame, condition_columns).to_numpy()
        x_m = metal_physical_features(frame["metal_symbol"].astype(str).tolist())
        return x_l, x_m, x_c

    def fit_predict(self, train, y_train, test, context: FoldContext) -> np.ndarray:
        import torch
        from torch import nn

        c = self.config
        torch.manual_seed(context.model_seed)
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
        ligand_columns, condition_columns = self._columns(context)

        xl_tr, xm_tr, xc_tr = self._matrices(train, ligand_columns, condition_columns)
        xl_te, xm_te, xc_te = self._matrices(test, ligand_columns, condition_columns)

        # Physics channel: raw log10 concentrations, centred on the TRAINING median
        # so the offset head keeps its meaning as the level at a typical experiment.
        if c.physics_head:
            available = [col for col in PHYSICS_COLUMNS if col in train.columns]
            if len(available) != len(PHYSICS_COLUMNS):
                raise KeyError(f"physics head needs {PHYSICS_COLUMNS}; frame has {available}")
            phys_tr = _as_float_frame(train, PHYSICS_COLUMNS).to_numpy()
            phys_te = _as_float_frame(test, PHYSICS_COLUMNS).to_numpy()
            centre = np.nan_to_num(np.nanmedian(phys_tr, axis=0))
            phys_tr = np.where(np.isfinite(phys_tr), phys_tr, centre) - centre
            phys_te = np.where(np.isfinite(phys_te), phys_te, centre) - centre
        else:
            phys_tr = np.zeros((len(train), 2))
            phys_te = np.zeros((len(test), 2))

        # fold-local standardisation; medians and moments never see a test row
        def standardise(a_tr, a_te):
            keep = ~np.all(np.isnan(a_tr), axis=0)
            a_tr, a_te = a_tr[:, keep], a_te[:, keep]
            median = np.nanmedian(a_tr, axis=0)
            median = np.where(np.isfinite(median), median, 0.0)
            a_tr = np.where(np.isfinite(a_tr), a_tr, median)
            a_te = np.where(np.isfinite(a_te), a_te, median)
            mean, sd = a_tr.mean(0), a_tr.std(0)
            sd = np.where(sd > 1e-8, sd, 1.0)
            return (a_tr - mean) / sd, (a_te - mean) / sd

        xl_tr, xl_te = standardise(xl_tr, xl_te)
        xm_tr, xm_te = standardise(xm_tr, xm_te)
        xc_tr, xc_te = standardise(xc_tr, xc_te)

        y = np.asarray(y_train, dtype=float)
        y_mean, y_sd = float(y.mean()), float(y.std() or 1.0)
        y_scaled = (y - y_mean) / y_sd

        ligand_labels = train["extractant"].astype(str).to_numpy()
        # TRAINING-FOLD ligand means only — the centring target of L_shape
        level = pd.Series(y_scaled).groupby(ligand_labels).transform("mean").to_numpy()
        centred = y_scaled - level
        weights = group_balanced_weights(train["ecfp_cluster"])
        weights = weights / weights.mean()

        # inner split holding out whole chemotypes of the training fold
        chemotypes = train["tanimoto_cluster"].astype(str).to_numpy()
        rng = np.random.default_rng(context.model_seed)
        unique = np.unique(chemotypes)
        n_hold = max(1, int(round(len(unique) * c.inner_holdout_chemotypes)))
        held = set(rng.permutation(unique)[:n_hold].tolist())
        valid_mask = np.array([g in held for g in chemotypes])
        if valid_mask.all() or not valid_mask.any():
            valid_mask = rng.random(len(y)) < 0.2
        fit_mask = ~valid_mask

        # codes for the pair sampler
        ligand_code = pd.factorize(ligand_labels)[0]
        cell_code = pd.factorize(
            train["condition_id"].astype(str) + "|" + train["metal_symbol"].astype(str))[0]

        device = torch.device("cpu")
        T = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
        # the physics channel is scaled by y_sd because the network works on a
        # standardised target: n̂ recovered from the fitted model is
        # ``coefficient * y_sd`` in log-units per decade of [L]
        tensors = dict(
            xl=T(xl_tr), xm=T(xm_tr), xc=T(xc_tr), y=T(y_scaled), centred=T(centred),
            w=T(weights), level=T(level), phys=T(phys_tr / y_sd))
        huber = nn.HuberLoss(delta=c.huber_delta, reduction="none")

        fit_idx = np.flatnonzero(fit_mask)
        val_idx = np.flatnonzero(valid_mask)
        pair_pool_within = _pair_pool(ligand_code[fit_idx], min_size=2)
        pair_pool_across = _pair_pool(cell_code[fit_idx], min_size=2)

        predictions: list[np.ndarray] = []
        offsets: list[np.ndarray] = []
        for member in range(max(1, c.n_ensemble)):
            torch.manual_seed(context.model_seed + 7919 * member)
            model = _make_network(c, xl_tr.shape[1], xm_tr.shape[1], xc_tr.shape[1]).to(device)
            optimiser = torch.optim.AdamW(model.parameters(), lr=c.learning_rate,
                                          weight_decay=c.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=c.epochs)
            best_state, best_score, bad = None, math.inf, 0
            generator = np.random.default_rng(context.model_seed + 104729 * member)

            for epoch in range(c.epochs):
                model.train()
                order = generator.permutation(fit_idx)
                for start in range(0, len(order), c.batch_size):
                    batch = order[start:start + c.batch_size]
                    if len(batch) < 8:
                        continue
                    idx = torch.as_tensor(batch, dtype=torch.long)
                    mu, r = model(tensors["xl"][idx], tensors["xm"][idx], tensors["xc"][idx],
                                  tensors["phys"][idx])
                    prediction = mu + r
                    w = tensors["w"][idx]
                    loss = c.lambda_abs * (huber(prediction, tensors["y"][idx]) * w).mean()
                    if c.lambda_shape:
                        loss = loss + c.lambda_shape * (
                            huber(r, tensors["centred"][idx]) * w).mean()
                    if c.lambda_center:
                        codes = torch.as_tensor(ligand_code[batch], dtype=torch.long)
                        loss = loss + c.lambda_center * _group_mean_square(r, codes)
                    if c.lambda_pair and c.pair_mode != "none":
                        loss = loss + c.lambda_pair * _pair_loss(
                            model, tensors, huber, generator, c,
                            pair_pool_within if c.pair_mode in ("within", "both") else None,
                            pair_pool_across if c.pair_mode in ("across", "both") else None)
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimiser.step()
                scheduler.step()

                if epoch % 2 == 0 or epoch == c.epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        idx = torch.as_tensor(val_idx, dtype=torch.long)
                        mu, r = model(tensors["xl"][idx], tensors["xm"][idx], tensors["xc"][idx],
                                      tensors["phys"][idx])
                        score = float(torch.abs(mu + r - tensors["y"][idx]).mean())
                    if score < best_score - 1e-4:
                        best_score, bad = score, 0
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    else:
                        bad += 1
                        if bad * 2 >= c.patience:
                            break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                mu, r = model(T(xl_te), T(xm_te), T(xc_te), T(phys_te / y_sd))
                predictions.append((mu + r).numpy())
                offsets.append(mu.numpy())

        prediction = np.mean(predictions, axis=0) * y_sd + y_mean
        if c.predict_offset_shape:
            context.extras["offset_hat"] = np.mean(offsets, axis=0) * y_sd + y_mean
            context.extras["prediction_sd"] = np.std(predictions, axis=0) * y_sd
        span = float(y.max() - y.min())
        return np.clip(prediction, y.min() - 0.5 * span, y.max() + 0.5 * span)


def _group_mean_square(values, codes):
    """``mean_g (mean_{i in g} r_i)^2`` — the offset-consistency penalty."""
    import torch
    unique, inverse = torch.unique(codes, return_inverse=True)
    sums = torch.zeros(len(unique), dtype=values.dtype).index_add_(0, inverse, values)
    counts = torch.zeros(len(unique), dtype=values.dtype).index_add_(
        0, inverse, torch.ones_like(values))
    return torch.square(sums / counts.clamp(min=1.0)).mean()


def _pair_pool(codes: np.ndarray, *, min_size: int) -> list[np.ndarray]:
    """Positional index groups with at least ``min_size`` members."""
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    groups = np.split(order, boundaries)
    return [g for g in groups if len(g) >= min_size]


def _sample_pairs(pool: list[np.ndarray], n: int, rng: np.random.Generator):
    if not pool or n <= 0:
        return None
    which = rng.integers(0, len(pool), size=n)
    left = np.empty(n, dtype=int)
    right = np.empty(n, dtype=int)
    for k, w in enumerate(which):
        group = pool[w]
        a, b = rng.integers(0, len(group), size=2)
        if a == b:
            b = (b + 1) % len(group)
        left[k], right[k] = group[a], group[b]
    return left, right


def _pair_loss(model, tensors, huber, rng, config, within_pool, across_pool):
    """Huber on ``ŷ_i − ŷ_j`` against ``y_i − y_j`` for sampled compatible pairs.

    Two families with different jobs: *within-ligand* pairs teach the shape,
    *same-condition cross-ligand* pairs teach the level **contrast** — the
    quantity a separations chemist actually reads off a table and the one earlier
    generations showed is far better determined than an absolute level.
    """
    import torch
    total = 0.0
    n_terms = 0
    half = max(1, config.pairs_per_step // 2)
    for pool in (within_pool, across_pool):
        if pool is None:
            continue
        sample = _sample_pairs(pool, half, rng)
        if sample is None:
            continue
        left, right = sample
        li = torch.as_tensor(left, dtype=torch.long)
        ri = torch.as_tensor(right, dtype=torch.long)
        mu_l, r_l = model(tensors["xl"][li], tensors["xm"][li], tensors["xc"][li],
                          tensors["phys"][li])
        mu_r, r_r = model(tensors["xl"][ri], tensors["xm"][ri], tensors["xc"][ri],
                          tensors["phys"][ri])
        delta_hat = (mu_l + r_l) - (mu_r + r_r)
        delta = tensors["y"][li] - tensors["y"][ri]
        total = total + huber(delta_hat, delta).mean()
        n_terms += 1
    if n_terms == 0:
        return torch.zeros((), dtype=torch.float32)
    return total / n_terms

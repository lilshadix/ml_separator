"""Conditional Neural Processes over ligands — gen8 brief §5.

The question this module exists to answer is narrow and falsifiable.

Offset correction (``OFFSET_K1``) treats the k measurements of a new ligand as an
estimate of *one number*: the ligand's vertical level.  It throws away everything
about *where* those measurements sat on the response surface, it shrinks nothing,
and it cannot say anything at all when k = 0.  A Conditional Neural Process is the
smallest architecture that removes all three limitations at once:

* the context encoder sees ``(conditions, metal, measured value)``, not just the
  value, so *where* a point was measured can change how it is used;
* the aggregation is a **mean over context points**, which is what makes it a CNP
  — it is permutation invariant and defined for any k, including k = 0, where it
  falls back to a learned null vector and the model is a plain zero-shot regressor;
* the map from context to correction is *learned across ~120 training ligands*, so
  the amount of shrinkage applied to a single noisy measurement is fitted rather
  than assumed to be zero.

Training is episodic: one step draws a batch of training ligands, draws
``k ~ Uniform{0..max_context}`` independently for each, splits that ligand's rows
into context and target, and minimises the target loss.  Sampling k during
training is the whole reason one set of weights handles every k at test time; a
model trained at fixed k would be a different model per k and could not be run
under the gen8 contract, which requires the same trained state for k = 0…5.

Two families are built, each carrying the same three-way ligand-representation
contrast, so "do molecular features add value *after* calibration?" is asked twice.

``CNP_conditions_only`` / ``CNP_ligand`` / ``CNP_ligand_recovered``
    predict log D outright.  The first has no ligand representation at all, so
    whatever it achieves is achievable **without knowing the molecule**; the second
    adds a compact ligand vector (DONORS + PHYSCHEM, 23 numbers) — raw 2048-bit
    ECFP is deliberately *not* fed to a network this small, since this repo's own
    finding on ``lig2d_ext`` is that a wide 2D descriptor block is actively harmful
    here; the third adds the recovered solvent-physics columns.
``CNPRES_*``
    the residual family: the frozen global model's prediction enters as a feature
    and the network predicts the *correction*, exactly the quantity offset
    correction estimates.  This is the honest head-to-head, because a residual CNP
    that simply learned "output the mean context residual" would *be* ``OFFSET_K1``.

The residual family is **anchored** and **translation invariant**.  Anchoring adds
the mean context residual to the output and zero-initialises the head, so at
initialisation the network is *exactly* ``OFFSET_K1`` above k = 0 and *exactly*
zero-shot at k = 0 — a property asserted in ``tests/test_gen8_cnp.py``.  Every
number training moves is therefore a measured claim about learnable structure
beyond the ligand's level.  Two members of the family exist to bound that claim
from either side: ``CNPRES_absolute_x`` drops translation invariance and shows what
memorising absolute condition space costs, and ``CNPRES_shrinkage`` strips the
decoder down to the three context summary statistics, so the *only* thing it can
learn is how much to trust k measurements.

Cross-fitting caveat, stated once and loudly: the residual family needs a frozen
model prediction for its *training* rows too, and the only one available is the
out-of-fold prediction, whose producing model was fitted on folds that include the
currently held-out one.  Test-time inputs are clean (the frozen model never saw the
held-out ligand), so no held-out target reaches a test prediction directly; the
contamination is the ordinary cross-fitting one, an aggregate over ~130 ligands
flowing into the *training distribution* of the correction.  It is the same
convention gen6 Phase 2 used for its cross-fitted residual stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:  # torch is required for this module but not for importing the package
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch is present in this environment
    torch = None
    nn = object

# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #

_CACHE = Path(__file__).resolve().parents[3] / "runs" / "gen7_architecture" / "cache"

#: Metal identity — three numbers, the lanthanide contraction among them.
METAL_COLUMNS: tuple[str, ...] = ("Atomic Number_metal", "lanthanide_index", "Ionic Radius_metal")
#: The mass-action block: log10 of every concentration in the extraction equilibrium.
MASSACTION_COLUMNS: tuple[str, ...] = (
    "massact__log10_cond__acid_concentration_M",
    "massact__log10_cond__contact_time_min",
    "massact__log10_cond__extractant_concentration_M",
    "massact__log10_cond__metal_concentration_mM",
    "massact__log10_cond__temperature_C",
    "massact__logL_x_DENTATE",
    "massact__logL_x_coreCN",
    "massact__logL_x_logH",
)
#: Compact ligand chemistry.  Donor-atom counts and coarse physchem, 23 numbers.
LIGAND_COLUMNS: tuple[str, ...] = (
    "donor__O(amide_carbonyl)", "donor__O(ether)", "donor__N(aromatic)", "donor__N(amine)",
    "donor__S(donor)", "donor__O(hydroxyl)", "donor__O(ester_carbonyl)", "donor__O(carbonyl)",
    "donor__n_total", "DENTATE", "coreCN", "n_ligs", "n_fill",
    "MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "NumAromaticRings", "NumAliphaticRings", "RingCount", "FractionCSP3", "MolLogP",
)
#: Recovered solvent physics and provenance flags, all fully populated.
RECOVERED_COLUMNS: tuple[str, ...] = (
    "rec__solvent_eps", "rec__solvent_mu", "rec__solvent_logp", "rec__solvent_vm",
    "rec__solvent_dD", "rec__solvent_dP", "rec__solvent_dH", "rec__solvent_arom",
    "rec__solvent_hal", "rec__solvent_oh", "rec__solvent_technical",
    "rec__solvent_n_components", "rec__solvent_log_eps", "rec__solvent_polar_fraction",
    "rec__solvent_parsed", "rec__has_phase_modifier", "rec__has_shaking_time",
    "rec__name_mismatch", "rec__n_names_for_structure", "rec__aqueous_complexant",
    "rec__phase_modifier_concentration_M", "rec__shaking_time_min",
)


def condition_columns(frame: pd.DataFrame, *, recovered: bool) -> list[str]:
    """Conditions + metal, in a fixed order, restricted to what the frame carries."""
    names = list(METAL_COLUMNS) + list(MASSACTION_COLUMNS)
    names += [c for c in frame.columns if c.startswith("cond__")]
    if recovered:
        names += [c for c in RECOVERED_COLUMNS if c in frame.columns]
    return [c for c in names if c in frame.columns]


# --------------------------------------------------------------------------- #
# Standardisation, fitted on the training fold only
# --------------------------------------------------------------------------- #

@dataclass
class Standardiser:
    """Median-impute then z-score, with a missingness indicator per leaky column.

    Every statistic is computed on the fold's training rows.  A column that is
    constant on the training fold has scale 1 and therefore contributes exactly
    zero to the network input, which is the honest encoding of "this fold never
    saw this diluent".
    """

    columns: list[str]
    centre: np.ndarray
    scale: np.ndarray
    indicator: list[int]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Sequence[str]) -> "Standardiser":
        values = _raw(frame, columns)
        missing = ~np.isfinite(values)
        centre = np.zeros(values.shape[1])
        scale = np.ones(values.shape[1])
        for j in range(values.shape[1]):
            finite = values[np.isfinite(values[:, j]), j]
            if finite.size:
                centre[j] = float(np.median(finite))
                sd = float(np.std(finite))
                scale[j] = sd if sd > 1e-8 else 1.0
        indicator = [j for j in range(values.shape[1]) if missing[:, j].any()]
        return cls(columns=list(columns), centre=centre, scale=scale, indicator=indicator)

    @property
    def width(self) -> int:
        return len(self.columns) + len(self.indicator)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = _raw(frame, self.columns)
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.centre[None, :], values)
        out = (filled - self.centre[None, :]) / self.scale[None, :]
        if self.indicator:
            out = np.hstack([out, missing[:, self.indicator].astype(float)])
        return np.clip(out, -8.0, 8.0)


def _raw(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    n = len(frame)
    parts = []
    for name in columns:
        if name in frame.columns:
            parts.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float))
        else:
            parts.append(np.full(n, np.nan))
    return np.vstack(parts).T if parts else np.zeros((n, 0))


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #

def _summary_stats(values: np.ndarray) -> np.ndarray:
    """``[log1p(k)/2, mean, sd]`` of the context targets; all zero for an empty context.

    Handed to the decoder alongside the pooled representation.  ``log1p(k)`` is the
    only thing that lets shrinkage be *learned*: one measurement and five carry the
    same mean but not the same reliability, and a model that cannot see k cannot
    trust them differently.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros(3, dtype=np.float32)
    return np.array([np.log1p(values.size) / 2.0, values.mean(),
                     values.std() if values.size > 1 else 0.0], dtype=np.float32)


def _mlp(sizes: Sequence[int]) -> "nn.Module":
    layers: list = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class CNPNet(nn.Module):
    """Encoder -> mean aggregation -> decoder, with an optional attention branch.

    The mean over context points is the load-bearing part: it is what makes the
    architecture a *Conditional Neural Process* rather than a network that happens
    to take k inputs.  ``null`` is the learned representation of an empty context,
    so k = 0 is a first-class case rather than a special branch.
    """

    #: Permutation-invariant summary of the context handed to the decoder verbatim:
    #: ``[log1p(k)/2, mean(context y), sd(context y)]``.  The first is what lets a
    #: learned *shrinkage* exist at all — the amount to trust one measurement is not
    #: the amount to trust five — and the third is the within-ligand spread that
    #: says how noisy that trust should be.  All three are zero for an empty context.
    n_summary = 3

    def __init__(self, d_cond: int, d_ligand: int, *, hidden: int = 64, r_dim: int = 64,
                 z_dim: int = 32, decoder_hidden: int = 128, use_ligand: bool = True,
                 use_prediction: bool = False, use_anchor: bool = False,
                 attentive: bool = False, relative: bool = False,
                 shrinkage_only: bool = False):
        super().__init__()
        self.shrinkage_only = shrinkage_only
        self.use_ligand = use_ligand and not shrinkage_only
        self.use_prediction = use_prediction and not shrinkage_only
        self.use_anchor = use_anchor
        self.attentive = attentive and not shrinkage_only
        self.relative = relative
        self.r_dim = r_dim

        self.encoder = _mlp([d_cond + 1, hidden, r_dim])
        self.null = nn.Parameter(torch.zeros(r_dim))
        if self.use_ligand:
            self.ligand_encoder = _mlp([d_ligand, z_dim, z_dim])
        if self.attentive:
            self.query = nn.Linear(d_cond, r_dim)
            self.key = nn.Linear(d_cond, r_dim)

        width = self.n_summary if shrinkage_only else \
            d_cond + r_dim * (2 if self.attentive else 1) + self.n_summary
        width += z_dim if self.use_ligand else 0
        width += 1 if self.use_prediction else 0
        self.decoder = _mlp([width, decoder_hidden, decoder_hidden, 2])
        last = self.decoder[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, ctx_x, ctx_y, ctx_mask, tgt_x, ligand, tgt_p, summary):
        """``ctx_*`` are (B, K, ...), ``tgt_x`` is (B, T, d_cond); masks are float."""
        batch, n_target, _ = tgt_x.shape
        weight = ctx_mask.unsqueeze(-1)
        count = ctx_mask.sum(dim=1, keepdim=True)
        empty = (count <= 0).float()

        if self.relative:
            # Translation invariance in condition space.  Encoder and decoder are
            # shown *how far each point sits from where the measurements were taken*,
            # never where any of them is in absolute terms.  That removes the channel
            # through which the network can memorise "at this acidity, in this
            # diluent, the frozen model is wrong by this much" — a map fitted on ~120
            # training ligands that is worthless on a new chemotype.  It also makes
            # k = 0 exactly zero-shot: with no context there is no displacement, the
            # input is the zero vector, and the zero-initialised head corrects nothing.
            centre = (ctx_x * weight).sum(dim=1) / count.clamp(min=1.0)
            encoder_x = (ctx_x - centre.unsqueeze(1)) * weight
            decoder_x = (tgt_x - centre.unsqueeze(1)) * (1.0 - empty).unsqueeze(1)
        else:
            encoder_x, decoder_x = ctx_x, tgt_x

        encoded = self.encoder(torch.cat([encoder_x, ctx_y.unsqueeze(-1)], dim=-1)) * weight
        pooled = encoded.sum(dim=1) / count.clamp(min=1.0)
        r = pooled * (1.0 - empty) + self.null.unsqueeze(0) * empty
        summary_wide = summary.unsqueeze(1).expand(batch, n_target, self.n_summary)
        if self.shrinkage_only:
            # The minimal model that can beat offset correction: the only thing the
            # head may see is (k, mean, sd) of the measured residuals, so the only
            # thing it can learn is *how much to trust them* — a shrinkage schedule
            # in k.  Nothing about conditions, molecule or prediction reaches it, so
            # it cannot memorise the training ligands even in principle.
            out = self.decoder(summary_wide)
            mu, raw_scale = out[..., 0], out[..., 1]
            if self.use_anchor:
                mu = mu + summary[:, 1].unsqueeze(1)
            return mu, torch.nn.functional.softplus(raw_scale) + 1e-3
        parts = [decoder_x, r.unsqueeze(1).expand(batch, n_target, self.r_dim), summary_wide]

        if self.attentive:
            q = self.query(decoder_x)
            k = self.key(encoder_x)
            logits = torch.einsum("btd,bkd->btk", q, k) / np.sqrt(self.r_dim)
            logits = logits.masked_fill(ctx_mask.unsqueeze(1) <= 0, -1e9)
            attention = torch.softmax(logits, dim=-1)
            attended = torch.einsum("btk,bkd->btd", attention, encoded)
            attended = attended * (1.0 - empty).unsqueeze(1)
            parts.append(attended)
        if self.use_ligand:
            z = self.ligand_encoder(ligand)
            parts.append(z.unsqueeze(1).expand(batch, n_target, z.shape[-1]))
        if self.use_prediction:
            parts.append(tgt_p.unsqueeze(-1))

        out = self.decoder(torch.cat(parts, dim=-1))
        mu, raw_scale = out[..., 0], out[..., 1]
        if self.use_anchor:
            mu = mu + summary[:, 1].unsqueeze(1)
        return mu, torch.nn.functional.softplus(raw_scale) + 1e-3


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #

@dataclass
class CNPAdapter:
    """One trained-per-fold Conditional Neural Process, under the gen8 contract.

    ``residual=True`` switches the whole thing to the correction parameterisation:
    the encoder sees the frozen model's residual at a context point instead of the
    raw value, and the decoder's output is added to the frozen prediction.
    ``anchor=True`` additionally hard-wires the mean context residual into the
    output, so the untrained network *is* offset correction and training can only
    move it away from that.
    """

    name: str = "CNP"
    use_ligand: bool = True
    recovered: bool = False
    residual: bool = False
    anchor: bool = False
    attentive: bool = False
    relative: bool = False
    shrinkage_only: bool = False
    feed_prediction: bool = True
    hidden: int = 64
    r_dim: int = 64
    z_dim: int = 32
    decoder_hidden: int = 128
    steps: int = 2500
    batch_ligands: int = 24
    max_context: int = 8
    max_target: int = 16
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    scale_weight: float = 0.1
    threads: int = 1
    #: Fraction of the fold's *training ligands* excluded from every episode and used
    #: only to early-stop.  A meta-validation ligand is the honest proxy for a new
    #: one: it is in the training fold, so nothing leaks, but the network has never
    #: adapted to it.  Without this the network memorises the ~120 training ligands'
    #: levels, ignores the context, and is worthless on a ligand it has not seen.
    val_fraction: float = 0.25
    val_episodes: int = 24
    eval_every: int = 100
    val_tolerance: float = 0.002

    _fitted: bool = field(default=False, init=False, repr=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)
    #: Diagnostics: (best step, best meta-validation MAE, meta-validation MAE at init).
    history: tuple = field(default=(0, float("nan"), float("nan")), init=False, repr=False)
    #: How many ``predict`` calls fell back to the frozen prediction.  Must be 0; a
    #: non-zero count means the adapter was silently scored as zero-shot somewhere.
    #: ``unfitted`` counts the *other* silent path — a fold whose ``fit_fold``
    #: returned before setting ``_fitted`` (no torch, or no ligand groups).  That
    #: path also scores the arm as pure zero-shot for the whole fold and is not an
    #: exception, so it has to be counted separately or it is invisible.
    failures: int = field(default=0, init=False, repr=False)
    unfitted: int = field(default=0, init=False, repr=False)
    last_error: str = field(default="", init=False, repr=False)

    # ---- fitting ---------------------------------------------------------- #

    def fit_fold(self, train: pd.DataFrame, y_train: np.ndarray, *, split_seed: int,
                 fold: int, model_seed: int) -> None:
        self._cache = {}
        self._fitted = False
        if torch is None:
            return
        torch.manual_seed(int(model_seed))
        torch.set_num_threads(int(self.threads))
        rng = np.random.default_rng(int(model_seed))

        cond_names = condition_columns(train, recovered=self.recovered)
        self.cond_std = Standardiser.fit(train, cond_names)
        self.lig_std = Standardiser.fit(train, [c for c in LIGAND_COLUMNS if c in train.columns])

        x = self.cond_std.transform(train).astype(np.float32)
        lig = self.lig_std.transform(train).astype(np.float32)
        y = np.asarray(y_train, dtype=np.float64)

        if self.residual:
            if "prediction" not in train.columns:
                raise ValueError(f"{self.name} needs a 'prediction' column on the training frame")
            p = pd.to_numeric(train["prediction"], errors="coerce").to_numpy(dtype=float)
            p = np.where(np.isfinite(p), p, float(np.median(y)))
            self.p_centre, self.p_scale = float(np.mean(p)), max(float(np.std(p)), 1e-6)
            signal = y - p
        else:
            self.p_centre, self.p_scale = 0.0, 1.0
            p = np.zeros_like(y)
            signal = y
        # The residual family is centred at exactly zero rather than at the training
        # mean residual.  The training mean is +0.54 here, because one 1,488-row
        # ligand carries 30% of the rows and the frozen model under-predicts it; a
        # row-weighted recentring would shift every zero-shot prediction by half a
        # log unit.  Centring at zero is also what makes the anchored variant's
        # initialisation *identically* ZERO_SHOT at k = 0 and OFFSET_K1 above it.
        self.y_centre = 0.0 if self.residual else float(np.mean(signal))
        self.y_scale = max(float(np.std(signal)), 1e-6)
        target = ((signal - self.y_centre) / self.y_scale).astype(np.float32)
        p_scaled = ((p - self.p_centre) / self.p_scale).astype(np.float32)

        ligands = train["extractant"].to_numpy()
        order = np.argsort(ligands, kind="stable")
        groups = [g for g in np.split(order, np.unique(ligands[order], return_index=True)[1][1:])
                  if len(g) >= 1]
        if not groups:
            return

        # The meta-validation split is blocked on the **chemotype**, not the ligand.
        # Splitting on the ligand would leave a held-out ligand's near-twin in the
        # episodes, and the early-stopping signal would then measure interpolation
        # inside a chemotype — which is not the regime this study evaluates, and is
        # so much easier that it early-stops far too late.
        chemotype = (train["tanimoto_cluster"].astype(str).to_numpy()
                     if "tanimoto_cluster" in train.columns else ligands)
        group_chemotype = np.array([chemotype[g[0]] for g in groups])
        names = np.unique(group_chemotype)
        held = set(names[rng.permutation(len(names))[:int(round(self.val_fraction * len(names)))]])
        val_groups = [g for g, c in zip(groups, group_chemotype) if c in held]
        fit_groups = [g for g, c in zip(groups, group_chemotype) if c not in held]
        if len(fit_groups) < 8 or not val_groups:
            val_groups, fit_groups = [], list(groups)

        net = CNPNet(d_cond=x.shape[1], d_ligand=lig.shape[1], hidden=self.hidden,
                     r_dim=self.r_dim, z_dim=self.z_dim, decoder_hidden=self.decoder_hidden,
                     use_ligand=self.use_ligand,
                     use_prediction=self.residual and self.feed_prediction,
                     use_anchor=self.anchor, attentive=self.attentive, relative=self.relative,
                     shrinkage_only=self.shrinkage_only)
        optimiser = torch.optim.Adam(net.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.steps)

        # A fixed meta-validation set, drawn once with its own generator, so the
        # early-stopping signal is the same batch at every checkpoint and a drop in
        # it is a real improvement rather than an easier draw.  k is sampled from the
        # k values the study reports, not from the training distribution.
        val_rng = np.random.default_rng(int(model_seed) + 7717)
        validation = [self._episode(val_groups, x, lig, target, p_scaled, val_rng,
                                    k_choices=(0, 1, 2, 3, 5), return_picked=True)
                      for _ in range(self.val_episodes)] if val_groups else []
        val_ligand = np.concatenate([picked for _, picked in validation]) if validation \
            else np.zeros(0, dtype=int)

        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        initial = self._val_errors(net, validation, val_ligand)
        best_step, best_score = 0, float(np.mean(initial)) if initial.size else float("nan")
        initial_score = best_score
        net.train()
        for step in range(1, self.steps + 1):
            batch = self._episode(fit_groups, x, lig, target, p_scaled, rng)
            mu, scale = net(*batch[:-1])
            tgt_y, tgt_mask = batch[-1]
            # One ligand, one vote — the same macro convention the study is scored
            # under, so the training objective and the reported metric agree.  A
            # row-weighted loss would be dominated by the 1,488-row DGA ligand.
            per_row = tgt_mask.sum(dim=1).clamp(min=1.0)
            error = (mu - tgt_y).abs()
            loss = (((error * tgt_mask).sum(dim=1)) / per_row).mean()
            nll = (error.detach() / scale + torch.log(scale))
            loss = loss + self.scale_weight * (((nll * tgt_mask).sum(dim=1)) / per_row).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimiser.step()
            schedule.step()
            if validation and (step % self.eval_every == 0 or step == self.steps):
                net.eval()
                current = self._val_errors(net, validation, val_ligand)
                net.train()
                # A *paired* test against the initialisation, per meta-validation
                # ligand, with a one-standard-error margin.  A checkpoint is chosen
                # ~25 times against ~25 held-out chemotypes, and the standard error
                # of that macro MAE is around 0.06 in these units: an unguarded
                # argmin therefore banks a 0.05 "improvement" that is pure selection
                # noise, which is exactly what an earlier version of this module did
                # — meta-validation fell 9% while the true held-out fold got 40%
                # worse.  The departure from offset correction has to be larger than
                # the noise in the evidence for it, or it is not taken.
                difference = current - initial
                margin = max(self.val_tolerance,
                             float(np.std(difference, ddof=1)) / np.sqrt(len(difference))) \
                    if len(difference) > 1 else self.val_tolerance
                score = float(np.mean(current))
                if float(np.mean(difference)) + margin < 0.0 and score < best_score:
                    best_step, best_score = step, score
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if validation:
            net.load_state_dict(best_state)
        net.eval()
        self.net = net
        self.history = (best_step, float(best_score), float(initial_score))
        self._fitted = True

    @staticmethod
    def _val_errors(net, validation, val_ligand: np.ndarray) -> np.ndarray:
        """Per-meta-validation-ligand MAE, in scaled units.  One ligand, one vote.

        Returned as a *vector* rather than a mean so that two checkpoints can be
        compared ligand-by-ligand on identical draws.  A paired comparison is the
        only one with enough power here: the between-ligand spread of MAE is several
        times the difference any checkpoint makes.
        """
        if not validation:
            return np.zeros(0)
        errors = []
        with torch.no_grad():
            for batch, _ in validation:
                mu, _ = net(*batch[:-1])
                tgt_y, tgt_mask = batch[-1]
                per_row = tgt_mask.sum(dim=1).clamp(min=1.0)
                errors.append((((mu - tgt_y).abs() * tgt_mask).sum(dim=1) / per_row).numpy())
        flat = np.concatenate(errors)
        names = np.unique(val_ligand)
        return np.array([flat[val_ligand == name].mean() for name in names])

    def _episode(self, groups, x, lig, target, p_scaled, rng, k_choices=None,
                 return_picked: bool = False):
        """One episodic batch: per-ligand random k, padded context and targets.

        Drawing k *inside* the episode is the mechanism that makes one set of
        weights valid for every k at test time.  ``k = 0`` is included, so the
        zero-shot case is trained rather than extrapolated to.
        """
        picked = rng.integers(0, len(groups), size=int(self.batch_ligands))
        k_max, t_max = int(self.max_context), int(self.max_target)
        batch = len(picked)
        ctx_x = np.zeros((batch, k_max, x.shape[1]), dtype=np.float32)
        ctx_y = np.zeros((batch, k_max), dtype=np.float32)
        ctx_mask = np.zeros((batch, k_max), dtype=np.float32)
        tgt_x = np.zeros((batch, t_max, x.shape[1]), dtype=np.float32)
        tgt_y = np.zeros((batch, t_max), dtype=np.float32)
        tgt_p = np.zeros((batch, t_max), dtype=np.float32)
        tgt_mask = np.zeros((batch, t_max), dtype=np.float32)
        ligand = np.zeros((batch, lig.shape[1]), dtype=np.float32)
        summary = np.zeros((batch, CNPNet.n_summary), dtype=np.float32)

        for b, index in enumerate(picked):
            rows = groups[index]
            permuted = rows[rng.permutation(len(rows))]
            k = int(rng.choice(k_choices)) if k_choices is not None \
                else int(rng.integers(0, k_max + 1))
            k = min(k, k_max, max(len(permuted) - 1, 0))
            context, targets = permuted[:k], permuted[k:k + t_max]
            if len(targets) == 0:
                context, targets = permuted[:0], permuted[:1]
                k = 0
            if k:
                ctx_x[b, :k] = x[context]
                ctx_y[b, :k] = target[context]
                ctx_mask[b, :k] = 1.0
                summary[b] = _summary_stats(target[context])
            n_t = len(targets)
            tgt_x[b, :n_t] = x[targets]
            tgt_y[b, :n_t] = target[targets]
            tgt_p[b, :n_t] = p_scaled[targets]
            tgt_mask[b, :n_t] = 1.0
            ligand[b] = lig[rows[0]]

        tensor = torch.from_numpy
        batch_tuple = (tensor(ctx_x), tensor(ctx_y), tensor(ctx_mask), tensor(tgt_x),
                       tensor(ligand), tensor(tgt_p), tensor(summary),
                       (tensor(tgt_y), tensor(tgt_mask)))
        return (batch_tuple, np.asarray(picked)) if return_picked else batch_tuple

    # ---- prediction ------------------------------------------------------- #

    def _block_features(self, block: pd.DataFrame, context) -> tuple:
        key = (context.split_seed, context.fold, context.extractant, len(block))
        cached = self._cache.get(key)
        if cached is None:
            x = self.cond_std.transform(block).astype(np.float32)
            lig = self.lig_std.transform(block.iloc[:1]).astype(np.float32)
            cached = (torch.from_numpy(x), torch.from_numpy(lig))
            self._cache[key] = cached
        return cached

    def predict(self, block, prediction, selected, observed, context) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=float)
        if not self._fitted:
            self.unfitted += 1
            return prediction
        try:
            x, lig = self._block_features(block, context)
            # ``ascontiguousarray`` is not cosmetic: the driver may hand over a
            # reversed or strided view, and torch refuses to index with a
            # negative-stride numpy array.  Without the copy that raise would be
            # swallowed by the fallback below and the adapter would silently return
            # the zero-shot prediction for those calls.
            selected = np.ascontiguousarray(np.asarray(selected, dtype=int))
            observed = np.ascontiguousarray(np.asarray(observed, dtype=float))
            if self.residual:
                base = prediction[selected] if selected.size else np.zeros(0)
                signal = observed - base
                p_scaled = ((prediction - self.p_centre) / self.p_scale).astype(np.float32)
            else:
                signal = observed
                p_scaled = np.zeros(len(block), dtype=np.float32)
            ctx_y = ((signal - self.y_centre) / self.y_scale).astype(np.float32)

            k = int(selected.size)
            ctx_x = x[selected].unsqueeze(0) if k else x[:0].unsqueeze(0)
            ctx_y_t = torch.from_numpy(ctx_y).unsqueeze(0) if k \
                else torch.zeros((1, 0), dtype=torch.float32)
            ctx_mask = torch.ones((1, k), dtype=torch.float32)
            summary = torch.from_numpy(
                _summary_stats(ctx_y if k else np.zeros(0, dtype=np.float32))[None, :])
            with torch.no_grad():
                mu, _ = self.net(ctx_x, ctx_y_t, ctx_mask, x.unsqueeze(0), lig,
                                 torch.from_numpy(p_scaled).unsqueeze(0), summary)
            value = mu.squeeze(0).numpy().astype(float) * self.y_scale + self.y_centre
            out = prediction + value if self.residual else value
            if not np.isfinite(out).all():
                self.failures += 1
                return prediction
            return out
        except Exception as error:  # the contract demands finite output for every row
            self.failures += 1
            self.last_error = repr(error)
            return prediction


@dataclass
class ZeroContextView:
    """A k = 0 view of a trained adapter: same weights, empty context, every k.

    The gen8 driver only scores adapters at k >= 1 (k = 0 is logged once as
    ``ZERO_SHOT_REF``), so the zero-shot number of a *trained* adapter would
    otherwise never be recorded.  This wrapper is not trainable — the driver fits
    the parent — and returns a constant per ligand, which is exactly what "k = 0"
    means.
    """

    parent: CNPAdapter
    name: str = ""
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.parent.name}@k0"

    def predict(self, block, prediction, selected, observed, context):
        key = (context.split_seed, context.fold, context.extractant, len(block))
        if key not in self._cache:
            if len(self._cache) > 4:
                self._cache.clear()
            self._cache[key] = self.parent.predict(
                block, prediction, np.zeros(0, dtype=int), np.zeros(0, dtype=float), context)
        return self._cache[key]


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

def build_cnp_adapters(*, steps: int = 2500, batch_ligands: int = 24, max_context: int = 8,
                       include_absolute: bool = True, include_residual: bool = True,
                       include_attentive: bool = True, with_zero_context: bool = True,
                       threads: int = 1) -> list:
    """The gen8 §5 arm list.

    Two families, each carrying the same three-way ligand-representation contrast,
    so "do molecular features add value after calibration?" is asked twice: once of
    a network predicting log D outright, and once of a network predicting the
    frozen model's correction.  The residual family is anchored and translation
    invariant, which is the strongest honest configuration — at initialisation it
    is *exactly* ``OFFSET_K1``, so every number it moves is a claim about learnable
    structure beyond the ligand's level.
    """
    common = dict(steps=steps, batch_ligands=batch_ligands, max_context=max_context,
                  threads=threads)
    residual_common = dict(common, residual=True, anchor=True, relative=True)
    adapters: list = []
    if include_absolute:
        adapters += [
            CNPAdapter(name="CNP_conditions_only", use_ligand=False, **common),
            CNPAdapter(name="CNP_ligand", use_ligand=True, **common),
            CNPAdapter(name="CNP_ligand_recovered", use_ligand=True, recovered=True, **common),
        ]
    if include_residual:
        adapters += [
            CNPAdapter(name="CNPRES_conditions_only", use_ligand=False, **residual_common),
            CNPAdapter(name="CNPRES_ligand", use_ligand=True, **residual_common),
            CNPAdapter(name="CNPRES_ligand_recovered", use_ligand=True, recovered=True,
                       **residual_common),
            CNPAdapter(name="CNPRES_absolute_x", use_ligand=True, residual=True, anchor=True,
                       relative=False, **common),
            CNPAdapter(name="CNPRES_shrinkage", use_ligand=False, shrinkage_only=True,
                       **residual_common),
        ]
        if include_attentive:
            adapters.append(CNPAdapter(name="CNPRES_attentive", use_ligand=True, attentive=True,
                                       **residual_common))
    if with_zero_context:
        adapters += [ZeroContextView(parent=a) for a in list(adapters)]
    return adapters


#: Alias matching the gen8 builder convention ``build_<area>_adapters``.
build_X_adapters = build_cnp_adapters


def parameter_count(adapter: CNPAdapter) -> int:
    """Trainable parameters of a fitted adapter — the brief caps this at 100k."""
    if not getattr(adapter, "_fitted", False):
        return 0
    return int(sum(p.numel() for p in adapter.net.parameters() if p.requires_grad))

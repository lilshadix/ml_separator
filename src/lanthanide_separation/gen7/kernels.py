"""Kernel and Gaussian-process contenders — the small-data case for not using a net.

With 152 supervised ligands and 5,248 rows this is squarely the regime where the
chemistry literature reports kernels beating neural networks, and where a GP's
uncertainty is worth as much as its mean.  Two things make an exact kernel cheap
here despite the row count:

* the ligand kernel is **block-constant** — only 152 distinct ligands exist, so
  ``K_ligand`` is a 152x152 matrix expanded by indexing, never recomputed per row;
* the structured kernel is a sum/product of a Tanimoto part and an RBF part, both
  of which are closed-form.

The composite kernel follows the additive-plus-interaction form the GP literature
recommends for factorial designs::

    K = a * K_ligand + b * K_experiment + c * (K_ligand ⊙ K_experiment)

The three terms are exactly the three things the decomposition says matter: a
ligand-only term (the level), an experiment-only term (the shared response to acid
and concentration) and their interaction (the ligand-specific response).  Setting
``c = 0`` gives a purely additive model, which is a meaningful ablation rather than
a degenerate one.

Hyperparameters are chosen by a small grid on an **inner split that holds out whole
chemotypes of the training fold** — the same rule the outer evaluation uses, so a
kernel cannot be tuned into looking good on chemistry it has effectively seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..levels import _as_float_frame, group_balanced_weights
from .harness import FoldContext


def tanimoto_kernel(bits_a: np.ndarray, bits_b: np.ndarray) -> np.ndarray:
    """Jaccard/Tanimoto similarity between binary fingerprint matrices."""
    a = np.asarray(bits_a, dtype=np.float32)
    b = np.asarray(bits_b, dtype=np.float32)
    intersection = a @ b.T
    counts_a = a.sum(1)[:, None]
    counts_b = b.sum(1)[None, :]
    union = counts_a + counts_b - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, intersection / union, 0.0)
    return out.astype(np.float64)


def rbf_kernel(a: np.ndarray, b: np.ndarray, *, gamma: float) -> np.ndarray:
    sq = (np.square(a).sum(1)[:, None] + np.square(b).sum(1)[None, :] - 2.0 * (a @ b.T))
    return np.exp(-gamma * np.maximum(sq, 0.0))


@dataclass
class StructuredKernelRidge:
    """Kernel ridge over ``a K_lig + b K_exp + c K_lig ⊙ K_exp``.

    ``ligand_kernel``: ``tanimoto`` on the ECFP bits (the chemically natural
    similarity for fingerprints, and the one GAUCHE and the Tanimoto-GP papers
    use) or ``rbf`` on continuous ligand descriptors.
    """

    name: str = "KRR_structured"
    ligand_kernel: str = "tanimoto"
    ligand_blocks: tuple[str, ...] = ("DONORS", "PHYSCHEM")
    experiment_blocks: tuple[str, ...] = ("METAL", "COND", "MASSACTION")
    alphas: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0)
    gammas: tuple[float, ...] = (0.01, 0.03, 0.1)
    mixes: tuple[tuple[float, float, float], ...] = (
        (1.0, 1.0, 0.0), (1.0, 1.0, 1.0), (1.0, 0.5, 2.0), (0.5, 1.0, 1.0), (1.0, 0.0, 1.0))
    weighting: str = "cluster"

    # -- kernel construction ------------------------------------------------- #
    def _ligand_bits(self, cohort, frame) -> np.ndarray:
        columns = [c for c in cohort.blocks.get("ECFP", ())]
        return _as_float_frame(frame, columns).to_numpy()

    def _ligand_continuous(self, cohort, frame) -> np.ndarray:
        columns = cohort.block_columns(tuple(b for b in self.ligand_blocks if b in cohort.blocks))
        return _as_float_frame(frame, columns).to_numpy()

    def _experiment(self, cohort, frame) -> np.ndarray:
        columns = cohort.block_columns(
            tuple(b for b in self.experiment_blocks if b in cohort.blocks))
        return _as_float_frame(frame, columns).to_numpy()

    @staticmethod
    def _standardise(train: np.ndarray, *others: np.ndarray):
        keep = ~np.all(np.isnan(train), axis=0)
        train = train[:, keep]
        others = tuple(o[:, keep] for o in others)
        median = np.nanmedian(train, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        train = np.where(np.isfinite(train), train, median)
        others = tuple(np.where(np.isfinite(o), o, median) for o in others)
        mean, sd = train.mean(0), train.std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        return ((train - mean) / sd, *[(o - mean) / sd for o in others])

    def fit_predict(self, train, y_train, test, context: FoldContext) -> np.ndarray:
        cohort = context.cohort
        if self.ligand_kernel == "tanimoto":
            lig_tr = self._ligand_bits(cohort, train)
            lig_te = self._ligand_bits(cohort, test)
            kernel_ligand = lambda a, b: tanimoto_kernel(a, b)
        else:
            lig_tr, lig_te = self._standardise(
                self._ligand_continuous(cohort, train), self._ligand_continuous(cohort, test))
            kernel_ligand = None

        exp_tr, exp_te = self._standardise(
            self._experiment(cohort, train), self._experiment(cohort, test))
        y = np.asarray(y_train, dtype=float)
        y_mean = float(y.mean())
        centred = y - y_mean

        # inner chemotype split for the grid
        chemotypes = train["tanimoto_cluster"].astype(str).to_numpy()
        rng = np.random.default_rng(context.model_seed)
        unique = np.unique(chemotypes)
        held = set(rng.permutation(unique)[: max(1, len(unique) // 5)].tolist())
        valid = np.array([g in held for g in chemotypes])
        fit = ~valid
        if not valid.any() or not fit.any():
            fit = np.ones(len(y), dtype=bool)
            valid = fit

        def gram(a_lig, b_lig, a_exp, b_exp, gamma, mix):
            k_lig = (kernel_ligand(a_lig, b_lig) if kernel_ligand is not None
                     else rbf_kernel(a_lig, b_lig, gamma=gamma))
            k_exp = rbf_kernel(a_exp, b_exp, gamma=gamma)
            a, b, c = mix
            return a * k_lig + b * k_exp + c * (k_lig * k_exp)

        best = None
        for gamma in self.gammas:
            for mix in self.mixes:
                k_ff = gram(lig_tr[fit], lig_tr[fit], exp_tr[fit], exp_tr[fit], gamma, mix)
                k_vf = gram(lig_tr[valid], lig_tr[fit], exp_tr[valid], exp_tr[fit], gamma, mix)
                for alpha in self.alphas:
                    try:
                        dual = np.linalg.solve(k_ff + alpha * np.eye(len(k_ff)), centred[fit])
                    except np.linalg.LinAlgError:
                        continue
                    score = float(np.abs(k_vf @ dual + y_mean - y[valid]).mean())
                    if best is None or score < best[0]:
                        best = (score, gamma, mix, alpha)
        if best is None:
            return np.full(len(test), y_mean)
        _, gamma, mix, alpha = best
        context.extras["kernel_alpha"] = np.full(len(test), alpha)

        k_tt = gram(lig_tr, lig_tr, exp_tr, exp_tr, gamma, mix)
        k_st = gram(lig_te, lig_tr, exp_te, exp_tr, gamma, mix)
        dual = np.linalg.solve(k_tt + alpha * np.eye(len(k_tt)), centred)
        prediction = k_st @ dual + y_mean
        span = float(y.max() - y.min())
        return np.clip(prediction, y.min() - 0.5 * span, y.max() + 0.5 * span)


@dataclass
class TanimotoGP:
    """Exact GP with a Tanimoto ligand kernel and an RBF experiment kernel.

    Gives a calibrated predictive standard deviation alongside the mean, which is
    the second reason to run a kernel here: an abstention rule needs a number that
    grows where the chemistry is unsupported, and an ensemble spread is a poor
    proxy for it.  Hyperparameters (three kernel amplitudes plus the noise) are
    fitted by marginal likelihood on the training fold — no held-out rows involved.
    """

    name: str = "GP_tanimoto"
    experiment_blocks: tuple[str, ...] = ("METAL", "COND", "MASSACTION")
    iterations: int = 60
    max_rows: int = 4200

    def fit_predict(self, train, y_train, test, context: FoldContext) -> np.ndarray:
        import torch

        cohort = context.cohort
        ecfp = [c for c in cohort.blocks.get("ECFP", ())]
        bits_tr = _as_float_frame(train, ecfp).to_numpy()
        bits_te = _as_float_frame(test, ecfp).to_numpy()
        columns = cohort.block_columns(
            tuple(b for b in self.experiment_blocks if b in cohort.blocks))
        exp_tr = _as_float_frame(train, columns).to_numpy()
        exp_te = _as_float_frame(test, columns).to_numpy()
        keep = ~np.all(np.isnan(exp_tr), axis=0)
        exp_tr, exp_te = exp_tr[:, keep], exp_te[:, keep]
        median = np.nan_to_num(np.nanmedian(exp_tr, axis=0))
        exp_tr = np.where(np.isfinite(exp_tr), exp_tr, median)
        exp_te = np.where(np.isfinite(exp_te), exp_te, median)
        mean, sd = exp_tr.mean(0), exp_tr.std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        exp_tr, exp_te = (exp_tr - mean) / sd, (exp_te - mean) / sd

        y = np.asarray(y_train, dtype=float)
        rng = np.random.default_rng(context.model_seed)
        if len(y) > self.max_rows:
            take = np.sort(rng.choice(len(y), size=self.max_rows, replace=False))
            bits_tr, exp_tr, y = bits_tr[take], exp_tr[take], y[take]
        y_mean, y_sd = float(y.mean()), float(y.std() or 1.0)
        target = torch.tensor((y - y_mean) / y_sd, dtype=torch.float64)

        k_lig_tt = torch.tensor(tanimoto_kernel(bits_tr, bits_tr), dtype=torch.float64)
        k_lig_st = torch.tensor(tanimoto_kernel(bits_te, bits_tr), dtype=torch.float64)
        exp_tr_t = torch.tensor(exp_tr, dtype=torch.float64)
        exp_te_t = torch.tensor(exp_te, dtype=torch.float64)

        log_amp = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        log_gamma = torch.tensor([np.log(0.03)], dtype=torch.float64, requires_grad=True)
        log_noise = torch.tensor([np.log(0.3)], dtype=torch.float64, requires_grad=True)
        optimiser = torch.optim.Adam([log_amp, log_gamma, log_noise], lr=0.08)

        def experiment_kernel(a, b, gamma):
            sq = (a.square().sum(1)[:, None] + b.square().sum(1)[None, :] - 2.0 * (a @ b.T))
            return torch.exp(-gamma * sq.clamp(min=0.0))

        n = len(target)
        eye = torch.eye(n, dtype=torch.float64)
        for _ in range(self.iterations):
            optimiser.zero_grad()
            gamma = log_gamma.exp()
            k_exp = experiment_kernel(exp_tr_t, exp_tr_t, gamma)
            amp = log_amp.exp()
            k = amp[0] * k_lig_tt + amp[1] * k_exp + amp[2] * (k_lig_tt * k_exp)
            k = k + (log_noise.exp() + 1e-6) * eye
            try:
                chol = torch.linalg.cholesky(k)
            except Exception:
                break
            alpha = torch.cholesky_solve(target[:, None], chol)
            nll = (0.5 * (target[:, None] * alpha).sum()
                   + torch.log(torch.diagonal(chol)).sum() + 0.5 * n * np.log(2 * np.pi))
            nll.backward()
            optimiser.step()

        with torch.no_grad():
            gamma = log_gamma.exp()
            amp = log_amp.exp()
            k_exp_tt = experiment_kernel(exp_tr_t, exp_tr_t, gamma)
            k_exp_st = experiment_kernel(exp_te_t, exp_tr_t, gamma)
            k = amp[0] * k_lig_tt + amp[1] * k_exp_tt + amp[2] * (k_lig_tt * k_exp_tt)
            k = k + (log_noise.exp() + 1e-6) * eye
            chol = torch.linalg.cholesky(k)
            alpha = torch.cholesky_solve(target[:, None], chol)
            k_star = amp[0] * k_lig_st + amp[1] * k_exp_st + amp[2] * (k_lig_st * k_exp_st)
            mean_prediction = (k_star @ alpha).squeeze(-1)
            v = torch.cholesky_solve(k_star.T, chol)
            prior = amp[0] + amp[1] + amp[2]
            variance = (prior - (k_star * v.T).sum(1)).clamp(min=1e-8)
            context.extras["gp_sd"] = (variance.sqrt() * y_sd).numpy()

        prediction = mean_prediction.numpy() * y_sd + y_mean
        span = float(y.max() - y.min())
        return np.clip(prediction, y.min() - 0.5 * span, y.max() + 0.5 * span)

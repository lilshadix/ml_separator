"""``models/ladder.py`` -- the ablation ladder steps M3-M7 (PyTorch, CPU), composed on top of ``models.neural``.

Registered text (``preregistration.md``, sealed 2026-09-15): section 6 ladder rows M3-M7 and the "M-model training
settings" resolution (every training constant is imported from ``models.neural`` and never redefined here), section 7
(inner folds only; the 0.005 tie rule; compute plan items 3-6), section 12 (M7: 5 members, heteroscedastic head,
split-conformal calibration fitted in inner folds, the normalised-by-SD variant), section 13 (domain-status
categories, consumed by the runner), section 15 (model seed rule), POST-HOC addendum 1 items 1-4 and readings 6(a)-(g);
brief sections 5, 6, 8, 9, 10, 14, 31, 32.  Nothing in this module edits ``models.neural``: the network is a subclass
of ``FactorisedNet``, the encoder wraps ``NeuralEncoder``, the training loop replays ``neural.train_network``'s loop
with the composite loss, and the tuner replays ``neural.tune`` with ``LadderConfig`` grids.  Nothing here reads a
registered fold file, scores an outer fold or touches V6.

Components (section 6 rows; :class:`LadderConfig` switches)
---------------------------------------------------------
* **M3 mechanism experts** (``experts``): separate bilinear ``W_e = P_e Q_e^T`` and separate condition encoders per
  expert for NEUTRAL_SOLVATING, SOFT_N_DONOR and MIXED_NEUTRAL (each only when it has >= :data:`MIN_EXPERT_ROWS`
  training rows), one POOLED expert for everything else (ACIDIC, CHELATING, SYNERGISTIC, UNKNOWN, and a label unseen
  in training).  Routing by the known system mechanism label (row column ``mechanism``, else
  ``descriptors/extractant_systems.csv``); no learned gate.  The metal / ligand encoders and the prediction head stay
  shared (:data:`REGISTRATION_CHOICES` ``experts_scope``).
* **M4 source hierarchy** (``tau``): a per-publication-group offset ``b_g`` (``group_cross_publication_copy``, else
  ``pub_group``) on the standardised target with the L2 penalty ``lambda_src * sum_g b_g^2 / (2 tau_std^2) / n_train``,
  ``tau_std = tau / y_sd`` (tau is registered in log D), ``lambda_src`` fixed at 1; a group unseen in training gets the
  population prior 0 (``padding_idx`` 0).
* **M5 pairwise loss** (``lambda_pair``): Huber (delta 1, standardised units) on ``(yhat_a - yhat_b) - (y_a - y_b)`` over
  the comparable pairs (``evaluation.pairs.comparable_pairs``: same publication group, same ``extractant_system_key``,
  identical ``normalize.condition_key``, different known metal states) generated INSIDE the fit's training rows after
  the fold assignment; ``leakage.pair_isolation_check`` runs on every set (:func:`training_pairs`,
  :func:`assert_pairs_inside`).
* **M6 physics** (``lambda_phys``, ``phys_terms``): (a) the hinge ``relu(-d yhat / d log10[extractant])`` (autograd on
  the condition input, rescaled by the column's standardisation SD) on training rows of NEUTRAL_SOLVATING /
  SOFT_N_DONOR / MIXED_NEUTRAL systems whose (system, acid) unit spans >= 2 distinct training extractant
  concentrations, never on acid-grid-flagged rows (:func:`hinge_mask`); (b) the smoothness ``sum ||e_Z - e_(Z+1)||^2``
  over adjacent-Z trivalent lanthanide and actinide metal embeddings present in training (:func:`series_adjacent_pairs`).
  No acid monotonicity, no loading term.
* **M7 uncertainty ensemble** (``heteroscedastic``; :class:`EnsembleArm`): 5 members at the retained configuration,
  member ``k`` seeded ``registered_model_seed(fold) + k`` and fitted on a publication-group bootstrap of the training
  rows (group multiplicities as row weights); a heteroscedastic head ``sd = softplus(.) + SD_FLOOR_STD`` trained by the
  Gaussian NLL on the standardised target with the mean detached (the point predictor stays the retained configuration's);
  ensemble ``mean_logD`` = member mean, ``std_logD^2`` = mean member variance + variance of member means;
  split-conformal calibration in inner folds on ``|y - mean| / sd`` (:class:`NormalisedCrossFitConformal`), intervals
  ``mean +- q sd``.

Search (section 6 "Searched, per ladder step"; addendum 1 items 1-2): each step searches only the hyperparameter it
introduces, every earlier value fixed at the outer fold's retained value (:func:`step_grid`, :func:`assert_search_scope`):
M3 none (1 configuration), M4 tau in {0.1, 0.3, 1.0}, M5 lambda_pair in {0.3, 1}, M6 lambda_phys in {0.1, 1}, M7 none.
Selection = the mean over the inner folds of the fold's unit-macro MAE, ties within 0.005 toward the stronger penalty
(smaller tau, larger lambda); the outer refit runs for the median best epoch.

Where the registration is silent the reading implemented here is listed in :data:`REGISTRATION_CHOICES`.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from gen19ct.chemistry import metals as MET
from gen19ct.chemistry import support_graph as SG
from gen19ct.data import leakage as LK
from gen19ct.data import normalize as N
from gen19ct.evaluation import calibration as EC
from gen19ct.evaluation import pairs as EP
from gen19ct.folds import io as FI
from gen19ct.folds import registered as FR
from gen19ct.models import inner_design as ID
from gen19ct.models import interface as I
from gen19ct.models import neural as NN

# --------------------------------------------------------------------------------------------- #
# registered constants (section 6 rows M3-M7, section 12, section 15)
# --------------------------------------------------------------------------------------------- #

STEPS: tuple[str, ...] = ("M3", "M4", "M5", "M6", "M7")
#: the H5 component-ablation arms (section 6 chemistry-priors question): the retained configuration with one
#: physics term toggled (section 6: "each the component on vs off with everything else at the retained configuration")
H5_ABLATION_ARMS: tuple[str, ...] = ("M6a_toggle", "M6b_toggle")
ALL_ARMS: tuple[str, ...] = STEPS + H5_ABLATION_ARMS
EXPERT_MECHANISMS: tuple[str, ...] = ("NEUTRAL_SOLVATING", "SOFT_N_DONOR", "MIXED_NEUTRAL")
POOLED_EXPERT = "POOLED"
#: section 6 M3: an expert of its own only with >= 200 training rows
MIN_EXPERT_ROWS = 200
TAU_GRID: tuple[float, ...] = (0.1, 0.3, 1.0)
LAMBDA_SRC = 1.0
LAMBDA_PAIR_GRID: tuple[float, ...] = (0.3, 1.0)
LAMBDA_PHYS_GRID: tuple[float, ...] = (0.1, 1.0)
HINGE_MECHANISMS: tuple[str, ...] = EXPERT_MECHANISMS
SMOOTH_SERIES: tuple[str, ...] = ("Ln", "An")
SMOOTH_OX = 3
N_MEMBERS = 5
#: heteroscedastic SD floor in standardised target units
SD_FLOOR_STD = 0.05
LAMBDA_HET = 1.0
CONFORMAL_METHOD = "normalized"
EXT_COLUMN = "condition__log10_extractant_M"
CV_EXT_COL = "log10_extractant_primary_M"
CV_GRID_COL = "acid_M_log10_grid"
ACID_UNIT_COL = "acid_primary"
UNKNOWN_MECHANISM = "UNKNOWN"
PHYS_TERMS: tuple[str, ...] = ("ab", "a", "b")
SEARCHED_FIELDS: dict[str, tuple[str, ...]] = {"M3": ("experts",), "M4": ("tau",), "M5": ("lambda_pair",),
                                                "M6": ("lambda_phys",), "M7": ("heteroscedastic",)}

LADDER_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "ladder_step", "config_label", "expert", "group_offset_seen", "tau", "lambda_pair", "lambda_phys", "phys_terms",
    "n_members", "gaussian_sd_logD")
PREDICTION_COLUMNS: tuple[str, ...] = I.PREDICTION_COLUMNS + NN.NEURAL_DIAGNOSTIC_COLUMNS + LADDER_DIAGNOSTIC_COLUMNS

REGISTRATION_CHOICES: dict[str, str] = {
    "training_settings": "every training constant (AdamW 3e-3, 10-epoch warm-up, batch 512, Huber delta 1, clip 5, "
                         "patience 30, <= 300 epochs, 2 threads, dropout 0.1, hidden sizes) is imported from "
                         "models.neural; the composite loss adds the step's penalty to neural's row-mean Huber L_D",
    "experts_scope": "section 6 M3 'separate W and condition heads': per expert a bilinear P_e Q_e^T and a condition "
                     "encoder h_e(x_cond); the metal / ligand encoders (e_m, e_l) and the prediction MLP are shared. "
                     "An expert of the three named mechanisms exists only with >= 200 training rows of the fit "
                     "(else its rows take the pooled expert); the pooled expert carries ACIDIC, CHELATING, "
                     "SYNERGISTIC, UNKNOWN, any other label and a label unseen in training",
    "expert_source": "the routing label is the row's `mechanism` column (prepare_support_frame, from "
                     "descriptors/extractant_systems.csv), else the systems table passed to the arm, else UNKNOWN",
    "source_penalty": "b_g on the standardised target, L2 weight 1 / tau_std^2 with tau_std = tau / y_sd (tau is "
                      "registered in log D), the MAP form sum_g b_g^2 / (2 tau_std^2) divided by n_train so that it "
                      "adds to the per-row mean Huber; lambda_src fixed 1; AdamW weight decay also applies to b_g "
                      "(the retained M1/M2 reading weight_decay_scope); padding index 0 = unseen group = prior 0",
    "group_column": "group_cross_publication_copy when the rows carry it (the registered publication unit), else "
                    "pub_group (the same values under prepare_frame); never a feature",
    "pairs": "comparable pairs are generated on the fit's own training rows (fold column constant 'train'), so a pair "
             "never holds a hidden or validation row; condition_key from normalize.condition_key on the condition "
             "vectors; the pair mini-batch is the next chunk of a seeded permutation of the pairs, one chunk per row "
             "mini-batch (size <= 512), cycling; a fit without a pair has L_pair = 0 (recorded)",
    "pair_loss_units": "Huber (delta 1) on the standardised difference (yhat_a - yhat_b) - (y_a - y_b) / y_sd",
    "hinge": "relu(-d yhat_std / d log10[L]) with the autograd derivative w.r.t. the standardised condition input "
             "divided by the column's standardisation SD; evaluated at training rows only, whose own (system, "
             "acid_primary) unit spans >= 2 distinct finite training log10[L] values (a degenerate range has no "
             "identifiable slope), never at an acid-grid-flagged row (normalize acid_M_log10_grid), only for "
             "NEUTRAL_SOLVATING / SOFT_N_DONOR / MIXED_NEUTRAL systems; mean over the eligible rows of the mini-batch",
    "smoothness": "sum over adjacent-Z pairs of trivalent Ln(III) and An(III) states present in the training rows of "
                  "||e_m(Z) - e_m(Z+1)||^2 on the full metal embedding e_m = A z_m + e_series + e_ox + e_element "
                  "(series and oxidation offsets cancel within a pair; a missing element breaks the chain); one "
                  "representative training row per state; the sum (not mean) as registered, added at every step",
    "phys_weight": "one lambda_phys multiplies both (a) and (b) (section 6: lambda_phys in {0.1, 1}); the H5 arms "
                   "M6a_toggle / M6b_toggle toggle one term with everything else at the retained configuration",
    "member_seed": "member k of M7 uses registered_model_seed(fold) + k (k = 0..4): distinct within a fold and across "
                   "folds (the fold step is 1009); needs a POST-HOC addendum (section 15 names one seed per fold)",
    "bootstrap": "each member resamples the publication groups of its training rows with replacement "
                 "(numpy default_rng(member seed), as many draws as groups) and trains on the drawn groups' rows with "
                 "the multiplicity as row weight (weighted Huber / NLL means); an undrawn group is absent from that "
                 "member's training rows, pairs and hinge set; the target scaling uses the drawn rows unweighted",
    "heteroscedastic_head": "sd_std = softplus(s(x)) + 0.05 with s an MLP (64, 64, SiLU, no dropout) on the DETACHED "
                            "[e_m, e_l, h]; Gaussian NLL 0.5 log sd^2 + 0.5 (y - mean.detach())^2 / sd^2 with weight 1; "
                            "the SD head's gradients are clipped separately (norm 5) and it draws no dropout mask, so "
                            "the mean network's training is bit-identical to the retained configuration's (asserted "
                            "by tests/test_ladder.py)",
    "ensemble": "mean_logD = mean of member means (log D units); std_logD^2 = mean of member sd^2 + population variance "
                "of member means (law of total variance); gaussian_sd_logD is printed beside",
    "m7_epochs": "M7 has no search: every member (inner calibration fit and outer refit) runs for the retained "
                 "predecessor's epoch count of the same outer fold",
    "conformal": "section 12 normalised-by-SD split conformal, fitted in inner folds: scores |y - mean| / std over the "
                 "cross-fitted inner calibration rows (fold j's ensemble fitted on fold j's inner training rows), "
                 "quantile ceil((n + 1) level)-th smallest, interval mean +- q std (calibration.fit_split_conformal, "
                 "method 'normalized'); the interval centre and SD are the outer refit's",
    "tie_order": "within 0.005 of the best: the stronger penalty (smaller tau, larger lambda_pair / lambda_phys), then "
                 "fewer parameters, then the grid position",
    "predecessor_chain": "a step's configuration inherits every field of the outer fold's retained predecessor "
                         "(M1 / M2 values from the discovery records, M3-M6 values from the ladder records) and varies "
                         "only its own field (assert_search_scope)",
}


def member_seed(fold_index: int, k: int) -> int:
    """M7 member seed: ``registered_model_seed(fold_index) + k`` (:data:`REGISTRATION_CHOICES` ``member_seed``)."""
    if not 0 <= int(k) < N_MEMBERS:
        raise ValueError(f"member index {k} outside 0..{N_MEMBERS - 1}")
    return NN.registered_model_seed(fold_index) + int(k)


# --------------------------------------------------------------------------------------------- #
# configuration and grids
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LadderConfig:
    """One ladder configuration: the retained M1 / M2 values plus the component switches of M3-M7."""

    step: str
    emb_dim: int
    weight_decay: float
    rank: int = 0
    experts: bool = False
    tau: float | None = None
    lambda_pair: float = 0.0
    lambda_phys: float = 0.0
    phys_terms: str = "ab"
    heteroscedastic: bool = False

    def __post_init__(self) -> None:
        NN.NeuralConfig(self.emb_dim, self.weight_decay, self.rank)      # validates the M1 / M2 part
        if self.step not in ("M1", "M2") + ALL_ARMS:
            raise ValueError(f"unknown ladder step {self.step!r}")
        if self.tau is not None and not float(self.tau) > 0:
            raise ValueError("tau must be > 0 (log D) or None")
        if float(self.lambda_pair) < 0 or float(self.lambda_phys) < 0:
            raise ValueError("lambda_pair / lambda_phys must be >= 0")
        if self.phys_terms not in PHYS_TERMS:
            raise ValueError(f"phys_terms must be one of {PHYS_TERMS}")

    @property
    def base(self) -> NN.NeuralConfig:
        return NN.NeuralConfig(int(self.emb_dim), float(self.weight_decay), int(self.rank))

    def components(self) -> dict[str, bool]:
        return {"bilinear": self.rank > 0, "experts": bool(self.experts), "source_hierarchy": self.tau is not None,
                "pairwise": self.lambda_pair > 0, "physics_hinge": self.lambda_phys > 0 and "a" in self.phys_terms,
                "physics_smoothness": self.lambda_phys > 0 and "b" in self.phys_terms,
                "heteroscedastic": bool(self.heteroscedastic)}

    def label(self) -> str:
        s = f"{self.step}_d{self.emb_dim}_wd{self.weight_decay:g}_r{self.rank}"
        if self.experts:
            s += "_exp"
        if self.tau is not None:
            s += f"_tau{self.tau:g}"
        if self.lambda_pair > 0:
            s += f"_lp{self.lambda_pair:g}"
        if self.lambda_phys > 0:
            s += f"_lph{self.lambda_phys:g}{self.phys_terms}"
        if self.heteroscedastic:
            s += "_het"
        return s

    def record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, rec: Mapping[str, Any]) -> "LadderConfig":
        r = dict(rec)
        return cls(step=str(r["step"]), emb_dim=int(r["emb_dim"]), weight_decay=float(r["weight_decay"]),
                   rank=int(r.get("rank", 0)), experts=bool(r.get("experts", False)),
                   tau=None if r.get("tau") is None else float(r["tau"]), lambda_pair=float(r.get("lambda_pair", 0.0)),
                   lambda_phys=float(r.get("lambda_phys", 0.0)), phys_terms=str(r.get("phys_terms", "ab")),
                   heteroscedastic=bool(r.get("heteroscedastic", False)))

    @classmethod
    def from_neural(cls, cfg: NN.NeuralConfig) -> "LadderConfig":
        """The retained M1 / M2 configuration of an outer fold as the ladder's starting point."""
        return cls(step=cfg.step, emb_dim=int(cfg.emb_dim), weight_decay=float(cfg.weight_decay), rank=int(cfg.rank))


def step_grid(step: str, prev: LadderConfig) -> list[LadderConfig]:
    """Section 6 'Searched, per ladder step': the configurations of ``step`` with every earlier value of ``prev``."""
    if step == "M3":
        return [replace(prev, step="M3", experts=True)]
    if step == "M4":
        return [replace(prev, step="M4", tau=float(t)) for t in TAU_GRID]
    if step == "M5":
        return [replace(prev, step="M5", lambda_pair=float(lp)) for lp in LAMBDA_PAIR_GRID]
    if step == "M6":
        return [replace(prev, step="M6", lambda_phys=float(lph), phys_terms="ab") for lph in LAMBDA_PHYS_GRID]
    if step == "M7":
        return [replace(prev, step="M7", heteroscedastic=True)]
    raise ValueError(f"no grid for step {step!r}")


def h5_toggle_config(arm: str, retained: LadderConfig) -> LadderConfig:
    """The H5 component-ablation arm: the retained configuration with the hinge (``M6a_toggle``) or the smoothness
    (``M6b_toggle``) toggled, everything else unchanged.  With ``lambda_phys`` 0 the toggled term is switched ON at the
    smallest registered weight; with both terms on, the toggled term is switched OFF."""
    if arm not in H5_ABLATION_ARMS:
        raise ValueError(f"unknown H5 ablation arm {arm!r}")
    term = "a" if arm == "M6a_toggle" else "b"
    other = "b" if term == "a" else "a"
    on = retained.lambda_phys > 0 and term in retained.phys_terms
    if on:
        # switch the term off: keep the other term (if it was on) at the retained weight, else no physics at all
        if other in retained.phys_terms:
            return replace(retained, step=arm, phys_terms=other)
        return replace(retained, step=arm, lambda_phys=0.0, phys_terms="ab")
    if retained.lambda_phys > 0:                       # only the other term was on: add this one at the same weight
        return replace(retained, step=arm, phys_terms="ab")
    return replace(retained, step=arm, lambda_phys=float(min(LAMBDA_PHYS_GRID)), phys_terms=term)


def assert_search_scope(step: str, configs: Sequence[LadderConfig], prev: LadderConfig) -> None:
    """Raise unless every configuration equals ``prev`` outside ``step`` and its own searched field(s)."""
    own = set(SEARCHED_FIELDS[step]) | {"step"}
    base = prev.record()
    for c in configs:
        rec = c.record()
        diff = sorted(k for k in rec if k not in own and rec[k] != base[k])
        if diff:
            raise AssertionError(f"{step}: the configuration {c.label()} changes {diff}, not its own hyperparameters")
        if c.step != step:
            raise AssertionError(f"{step}: configuration labelled {c.step}")


def tie_key(cfg: LadderConfig, n_parameters: int, position: int) -> tuple:
    """Section 7 tie rule toward the stronger penalty (:data:`REGISTRATION_CHOICES` ``tie_order``)."""
    return (-float(cfg.lambda_pair), -float(cfg.lambda_phys), float("inf") if cfg.tau is None else float(cfg.tau),
            int(n_parameters), int(position))


# --------------------------------------------------------------------------------------------- #
# routing, groups, pairs, hinge set, smoothness pairs (all fitted on the fit's training rows)
# --------------------------------------------------------------------------------------------- #

def system_mechanisms(rows: pd.DataFrame, systems: pd.DataFrame | None = None) -> np.ndarray:
    """Per-row mechanism label (object array; ``UNKNOWN`` when neither the row nor the systems table knows it)."""
    out = np.array([UNKNOWN_MECHANISM] * len(rows), dtype=object)
    if SG.MECH_COL in rows.columns:
        v = rows[SG.MECH_COL].to_numpy(dtype=object)
        for i, x in enumerate(v):
            if not SG._missing(x):
                out[i] = str(x)
    if systems is not None and SG.MECH_COL in systems.columns and SG.SYSTEM_COL in rows.columns:
        keys = rows[SG.SYSTEM_COL].to_numpy(dtype=object)
        tab = systems[SG.MECH_COL]
        if SG.SYSTEM_COL in systems.columns and not systems.index.name == SG.SYSTEM_COL:
            tab = systems.set_index(SG.SYSTEM_COL)[SG.MECH_COL]
        for i, k in enumerate(keys):
            if out[i] == UNKNOWN_MECHANISM and k in tab.index and not SG._missing(tab.loc[k]):
                out[i] = str(tab.loc[k])
    return out


@dataclass(frozen=True)
class RoutingTable:
    """M3 routing: expert names (pooled last) and the mechanism -> expert position map fitted on training rows."""

    experts: tuple[str, ...]
    index: Mapping[str, int]
    n_rows: Mapping[str, int]

    @property
    def n_experts(self) -> int:
        return len(self.experts)

    @property
    def pooled(self) -> int:
        return len(self.experts) - 1

    def expert_of(self, mechanism: Any) -> int:
        """Known label of an own expert -> its position; anything else (pooled labels, unseen) -> the pooled expert."""
        return int(self.index.get(str(mechanism), self.pooled))

    def route(self, mechanisms: Iterable[Any]) -> np.ndarray:
        return np.array([self.expert_of(m) for m in mechanisms], dtype=np.int64)

    def record(self) -> dict[str, Any]:
        return {"experts": list(self.experts), "index": dict(self.index), "n_training_rows": dict(self.n_rows)}


def fit_routing(mechanisms: Iterable[Any], min_rows: int = MIN_EXPERT_ROWS) -> RoutingTable:
    """Own experts for the registered mechanisms with >= ``min_rows`` training rows, then POOLED."""
    labels = [str(m) for m in mechanisms]
    counts = {m: labels.count(m) for m in EXPERT_MECHANISMS}
    own = [m for m in EXPERT_MECHANISMS if counts[m] >= int(min_rows)]
    experts = tuple(own) + (POOLED_EXPERT,)
    pooled_n = sum(1 for m in labels if m not in own)
    n_rows = {**{m: int(counts[m]) for m in own}, POOLED_EXPERT: int(pooled_n)}
    return RoutingTable(experts=experts, index={m: i for i, m in enumerate(own)}, n_rows=n_rows)


def group_column(rows: pd.DataFrame) -> str:
    if FI.GROUP_COL in rows.columns:
        return FI.GROUP_COL
    if I.PUB_GROUP_COL in rows.columns:
        return I.PUB_GROUP_COL
    raise KeyError(f"rows carry neither {FI.GROUP_COL} nor {I.PUB_GROUP_COL} (the M4 publication group)")


def condition_vectors_of(rows: pd.DataFrame, cv: pd.DataFrame | None) -> pd.DataFrame:
    """The condition vectors of ``rows`` (the given store restricted to them, else computed on the rows)."""
    if cv is not None:
        if not rows.index.isin(cv.index).all():
            raise ValueError("condition vectors do not cover the rows")
        return cv.loc[rows.index]
    return N.condition_vector(rows)


def training_pairs(rows: pd.DataFrame, cv: pd.DataFrame | None = None, *, group_col: str | None = None) -> pd.DataFrame:
    """M5: the comparable pairs INSIDE ``rows`` (a fit's training rows), generated after the fold assignment with a
    constant fold label so that ``leakage.pair_isolation_check`` (run by ``pairs.comparable_pairs``) sees one fold."""
    gcol = group_col or group_column(rows)
    cvr = condition_vectors_of(rows, cv)
    ck = N.condition_key(cvr)
    df = pd.DataFrame({gcol: rows[gcol].astype(object).to_numpy(), SG.SYSTEM_COL: rows[SG.SYSTEM_COL].astype(object).to_numpy(),
                       "condition_key": ck.reindex(rows.index).astype(object).to_numpy(),
                       SG.METAL_COL: rows[SG.METAL_COL].astype(object).to_numpy(),
                       I.TARGET_COL: pd.to_numeric(rows[I.TARGET_COL], errors="coerce").to_numpy(dtype=float),
                       EP.FOLD_COL: "train"}, index=rows.index)
    return EP.comparable_pairs(df, key_cols=(gcol, SG.SYSTEM_COL, "condition_key"), fold_col=EP.FOLD_COL,
                               state_col=SG.METAL_COL, y_col=I.TARGET_COL)


def assert_pairs_inside(pairs: pd.DataFrame, train_index: pd.Index, hidden_index: pd.Index | None = None) -> None:
    """Every pair member is a training row and none is hidden (``leakage.pair_isolation_check`` on a train / hidden
    fold map raises on a crossing or an unassigned member)."""
    folds = pd.Series("train", index=pd.Index(train_index, dtype=object), dtype=object)
    if hidden_index is not None and len(hidden_index):
        hid = pd.Index(hidden_index, dtype=object).difference(folds.index)
        folds = pd.concat([folds, pd.Series("hidden", index=hid, dtype=object)])
    LK.pair_isolation_check(pairs, folds, member_cols=("idx_a", "idx_b"))
    if len(pairs) and not (pairs["fold"].astype(str) == "train").all():
        raise AssertionError("a training pair lies outside the training rows")


def hinge_mask(rows: pd.DataFrame, mechanisms: np.ndarray, cv: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """M6 (a): the training rows at which the hinge is evaluated (module docstring), with a count record."""
    n = len(rows)
    mech_ok = np.array([str(m) in HINGE_MECHANISMS for m in mechanisms], dtype=bool)
    grid = cv[CV_GRID_COL].to_numpy(dtype=bool) if CV_GRID_COL in cv.columns else np.zeros(n, dtype=bool)
    ext = pd.to_numeric(cv[CV_EXT_COL], errors="coerce").to_numpy(dtype=float) if CV_EXT_COL in cv.columns \
        else pd.to_numeric(rows[SG.LOG_EXT_COL], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(ext)
    cand = mech_ok & ~grid & finite
    acid = rows[ACID_UNIT_COL].astype(object).to_numpy() if ACID_UNIT_COL in rows.columns \
        else rows[SG.ACID_ANION_COL].astype(object).to_numpy()
    sysk = rows[SG.SYSTEM_COL].astype(object).to_numpy()
    out = np.zeros(n, dtype=bool)
    ranges: dict[str, tuple[float, float, int]] = {}
    units: dict[tuple, list[int]] = {}
    for i in np.flatnonzero(cand):
        units.setdefault((str(sysk[i]), str(acid[i])), []).append(int(i))
    for key, pos in units.items():
        vals = np.unique(np.round(ext[pos], 9))
        if len(vals) >= 2:
            out[pos] = True
            ranges[f"{key[0]} | {key[1]}"] = (float(vals.min()), float(vals.max()), len(pos))
    info = {"n_rows": int(n), "n_mechanism_ok": int(mech_ok.sum()), "n_acid_grid_flagged": int(grid.sum()),
            "n_candidates": int(cand.sum()), "n_hinge_rows": int(out.sum()), "n_units_with_range": len(ranges),
            "n_units_degenerate": int(len(units) - len(ranges))}
    return out, {"counts": info, "ranges": ranges}


def series_adjacent_pairs(states: Iterable[str]) -> list[tuple[str, str]]:
    """M6 (b): adjacent-Z pairs of trivalent Ln / An states among ``states`` (``(Z, Z+1)`` both present)."""
    by: dict[str, dict[int, str]] = {s: {} for s in SMOOTH_SERIES}
    for st in set(str(s) for s in states if not SG._missing(s)):
        p = SG.metal_properties(st)
        if p["series"] in by and p["ox"] == SMOOTH_OX and p["Z"] is not None:
            by[p["series"]][int(p["Z"])] = st
    out = []
    for s in SMOOTH_SERIES:
        zs = sorted(by[s])
        for z in zs:
            if z + 1 in by[s]:
                out.append((by[s][z], by[s][z + 1]))
    return out


# --------------------------------------------------------------------------------------------- #
# the encoder (NeuralEncoder + expert and group ids)
# --------------------------------------------------------------------------------------------- #

class LadderEncoder:
    """``neural.NeuralEncoder`` plus the M3 routing table, the M4 group vocabulary and the M6 hinge column, all fitted on
    the fit's training rows only."""

    def __init__(self, config: LadderConfig, *, min_expert_rows: int = MIN_EXPERT_ROWS,
                 systems: pd.DataFrame | None = None, neural: NN.NeuralEncoder | None = None):
        self.config = config
        self.neural = NN.NeuralEncoder() if neural is None else neural
        self._prefitted = neural is not None and neural.dims is not None
        self.min_expert_rows = int(min_expert_rows)
        self.systems = systems
        self.routing: RoutingTable | None = None
        self.groups: list[str] = []
        self.group_index: dict[str, int] = {}
        self.group_col: str | None = None
        self.ext_index: int | None = None
        self.ext_scale: float = 1.0

    @property
    def dims(self) -> NN.InputDims:
        if self.neural.dims is None:
            raise RuntimeError("LadderEncoder: fit first")
        return self.neural.dims

    @property
    def state_digest(self) -> str:
        h = hashlib.sha256(self.neural.state_digest.encode())
        h.update(repr(None if self.routing is None else self.routing.record()).encode())
        h.update(repr(self.groups).encode())
        return h.hexdigest()

    @property
    def n_experts(self) -> int:
        return 1 if self.routing is None else self.routing.n_experts

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    def fit(self, rows: pd.DataFrame, cv: pd.DataFrame | None = None) -> "LadderEncoder":
        if not self._prefitted:                      # a prefitted NeuralEncoder (same training rows) is shared
            self.neural.fit(rows, cv)
        mech = system_mechanisms(rows, self.systems)
        self.routing = fit_routing(mech, self.min_expert_rows) if self.config.experts else None
        if self.config.tau is not None:
            self.group_col = group_column(rows)
            self.groups = sorted({str(g) for g in rows[self.group_col].to_numpy(dtype=object) if not SG._missing(g)})
            self.group_index = {g: i + 1 for i, g in enumerate(self.groups)}
        cols = self.neural.block_columns[NN.CONDITION_BLOCK]
        self.ext_index = cols.index(EXT_COLUMN) if EXT_COLUMN in cols else None
        try:
            self.ext_scale = float(self.neural.features._preps[NN.CONDITION_BLOCK].sd.get(EXT_COLUMN, 1.0))
        except (AttributeError, KeyError):
            self.ext_scale = 1.0
        return self

    def transform(self, rows: pd.DataFrame, cv: pd.DataFrame | None = None) -> NN.EncodedRows:
        enc = self.neural.transform(rows, cv)
        mech = system_mechanisms(rows, self.systems)
        ids = dict(enc.ids)
        ids["expert_id"] = self.routing.route(mech) if self.routing is not None else np.zeros(len(rows), dtype=np.int64)
        if self.config.tau is not None:
            g = rows[self.group_col].to_numpy(dtype=object) if self.group_col in rows.columns else np.array([None] * len(rows))
            ids["group_id"] = np.array([self.group_index.get(str(x), 0) if not SG._missing(x) else 0 for x in g],
                                       dtype=np.int64)
        else:
            ids["group_id"] = np.zeros(len(rows), dtype=np.int64)
        return NN.EncodedRows(enc.index, enc.z_m, enc.z_l, enc.x_c, ids)

    def record(self) -> dict[str, Any]:
        return {"feature_state_digest": self.neural.state_digest, "routing": None if self.routing is None else
                self.routing.record(), "n_groups": self.n_groups, "group_col": self.group_col,
                "ext_index": self.ext_index, "ext_scale": self.ext_scale}


# --------------------------------------------------------------------------------------------- #
# the network
# --------------------------------------------------------------------------------------------- #

class LadderNet(NN.FactorisedNet):
    """``FactorisedNet`` (the retained M1 / M2 part) plus mechanism experts, the group offset and the SD head.
    Construct inside ``neural.deterministic_torch``."""

    def __init__(self, dims: NN.InputDims, config: LadderConfig, *, n_experts: int = 1, n_groups: int = 0):
        super().__init__(dims, config.base)
        self.ladder_config = config
        d, r = int(config.emb_dim), int(config.rank)
        self.n_experts = int(n_experts) if config.experts else 1
        if self.n_experts < 1:
            raise ValueError("at least one expert (the pooled one)")
        if config.experts:
            # expert 0 reuses the shared modules built by FactorisedNet; experts 1.. get their own
            self.cond_experts = nn.ModuleList([self.cond] + [NN._mlp((dims.p_condition,) + NN.COND_HIDDEN, None)
                                                              for _ in range(self.n_experts - 1)])
            if r:
                self.P_experts = nn.Parameter(torch.randn(self.n_experts - 1, d, r) * 0.1)
                self.Q_experts = nn.Parameter(torch.randn(self.n_experts - 1, d, r) * 0.1)
            else:
                self.register_parameter("P_experts", None)
                self.register_parameter("Q_experts", None)
        else:
            self.cond_experts = None
            self.register_parameter("P_experts", None)
            self.register_parameter("Q_experts", None)
        if config.tau is not None:
            self.b_group = nn.Embedding(int(n_groups) + 1, 1, padding_idx=NN.F.RESERVED_INDEX)
            nn.init.zeros_(self.b_group.weight)
        else:
            self.b_group = None
        if config.heteroscedastic:
            # the SD head is built on a forked RNG (derived seed) and has no dropout, so it neither shifts the mean
            # network's initialisation stream nor the dropout masks it draws during training
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed((int(torch.initial_seed()) * 31 + 7) % (2 ** 63 - 1))
                self.sd_head = nn.Sequential(NN._mlp((2 * d + NN.COND_HIDDEN[-1],) + NN.HEAD_HIDDEN, None),
                                             nn.Linear(NN.HEAD_HIDDEN[-1], 1))
        else:
            self.sd_head = None
        n = NN.count_parameters(self)
        if n > NN.MAX_PARAMETERS:
            raise AssertionError(f"section 6: {n} parameters > {NN.MAX_PARAMETERS}")
        self.n_parameters = n

    # ----------------------------------------------------------------------------------------- #
    def condition_h(self, t: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.cond_experts is None:
            return self.cond(t["x_c"])
        outs = torch.stack([m(t["x_c"]) for m in self.cond_experts], dim=0)            # (E, B, 32)
        idx = t["expert_id"].view(1, -1, 1).expand(1, outs.shape[1], outs.shape[2])
        return outs.gather(0, idx).squeeze(0)

    def expert_interaction(self, e_m: torch.Tensor, e_l: torch.Tensor, expert_id: torch.Tensor) -> torch.Tensor:
        if self.P is None:
            return torch.zeros(e_m.shape[0], dtype=e_m.dtype)
        if self.P_experts is None:
            return ((e_m @ self.P) * (e_l @ self.Q)).sum(dim=1)
        P = torch.cat([self.P.unsqueeze(0), self.P_experts], dim=0)                      # (E, d, r)
        Q = torch.cat([self.Q.unsqueeze(0), self.Q_experts], dim=0)
        Pe, Qe = P[expert_id], Q[expert_id]                                              # (B, d, r)
        return (torch.bmm(e_m.unsqueeze(1), Pe).squeeze(1) * torch.bmm(e_l.unsqueeze(1), Qe).squeeze(1)).sum(dim=1)

    def trunk(self, t: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.metal_embedding(t), self.ligand_embedding(t), self.condition_h(t)

    def mean_from(self, t: Mapping[str, torch.Tensor], e_m: torch.Tensor, e_l: torch.Tensor, h: torch.Tensor
                  ) -> torch.Tensor:
        out = self.head(torch.cat([e_m, e_l, h], dim=1)).squeeze(1)
        if self.P is not None:
            out = out + self.expert_interaction(e_m, e_l, t["expert_id"])
        if self.b_group is not None:
            out = out + self.b_group(t["group_id"]).squeeze(1)
        return out

    def forward(self, t: Mapping[str, torch.Tensor]) -> torch.Tensor:
        e_m, e_l, h = self.trunk(t)
        return self.mean_from(t, e_m, e_l, h)

    def sd_from(self, e_m: torch.Tensor, e_l: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Standardised predictive SD from the DETACHED trunk features (``heteroscedastic_head`` reading)."""
        if self.sd_head is None:
            raise RuntimeError("no heteroscedastic head")
        raw = self.sd_head(torch.cat([e_m.detach(), e_l.detach(), h.detach()], dim=1)).squeeze(1)
        return nn.functional.softplus(raw) + SD_FLOOR_STD

    def forward_all(self, t: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        e_m, e_l, h = self.trunk(t)
        mean = self.mean_from(t, e_m, e_l, h)
        return mean, (self.sd_from(e_m, e_l, h) if self.sd_head is not None else None)

    def group_offsets(self) -> np.ndarray | None:
        if self.b_group is None:
            return None
        with torch.no_grad():
            return self.b_group.weight.squeeze(1).numpy().astype(np.float64)

    def expert_bilinear_matrices(self) -> list[np.ndarray] | None:
        if self.P is None:
            return None
        with torch.no_grad():
            mats = [(self.P @ self.Q.T).numpy().astype(np.float64)]
            if self.P_experts is not None:
                mats += [(p @ q.T).numpy().astype(np.float64) for p, q in zip(self.P_experts, self.Q_experts)]
        return mats


def parameter_count(dims: NN.InputDims, config: LadderConfig, *, n_experts: int = 1, n_groups: int = 0) -> int:
    with torch.random.fork_rng(devices=[]):
        return LadderNet(dims, config, n_experts=n_experts, n_groups=n_groups).n_parameters


# --------------------------------------------------------------------------------------------- #
# the fit-time auxiliaries of the composite loss
# --------------------------------------------------------------------------------------------- #

@dataclass
class FitAux:
    """Everything the composite loss needs beyond the encoded rows, built on the fit's training rows only."""

    n_train: int
    pair_positions: np.ndarray                     # (n_pairs, 2) positions into the training rows
    hinge_mask: np.ndarray                         # bool per training row
    ext_index: int | None
    ext_scale: float
    smooth_positions: np.ndarray                   # representative training-row positions of the trivalent states
    smooth_pairs: np.ndarray                       # (n, 2) indices into smooth_positions
    tau_std: float | None
    row_weights: np.ndarray | None = None          # bootstrap multiplicities (M7), else None
    info: dict[str, Any] = field(default_factory=dict)


def build_fit_aux(rows: pd.DataFrame, enc: NN.EncodedRows, encoder: LadderEncoder, config: LadderConfig, *,
                  y_sd: float, cv: pd.DataFrame | None = None, row_weights: np.ndarray | None = None) -> FitAux:
    """The M4 prior scale, the M5 pairs, the M6 hinge set and smoothness pairs of one fit (its training rows)."""
    n = len(rows)
    if not rows.index.equals(enc.index):
        raise AssertionError("build_fit_aux: rows and encoded rows differ in order")
    info: dict[str, Any] = {}
    pos_of = pd.Series(np.arange(n), index=rows.index)
    pairs = np.zeros((0, 2), dtype=np.int64)
    if config.lambda_pair > 0:
        pf = training_pairs(rows, cv)
        assert_pairs_inside(pf, rows.index)
        if len(pf):
            pairs = np.column_stack([pos_of.loc[pf["idx_a"].to_numpy(dtype=object)].to_numpy(dtype=np.int64),
                                     pos_of.loc[pf["idx_b"].to_numpy(dtype=object)].to_numpy(dtype=np.int64)])
        info["pairs"] = {"n_pairs": int(len(pf)), "n_rows_in_pairs": int(len(np.unique(pairs))) if len(pairs) else 0,
                         "category_classes": pf["category_class"].value_counts().to_dict() if len(pf) else {}}
    hm = np.zeros(n, dtype=bool)
    sp, spairs = np.zeros(0, dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
    if config.lambda_phys > 0:
        mech = system_mechanisms(rows, encoder.systems)
        if "a" in config.phys_terms:
            hm, hinfo = hinge_mask(rows, mech, condition_vectors_of(rows, cv))
            if encoder.ext_index is None:
                hm[:] = False
                hinfo["counts"]["note"] = f"{EXT_COLUMN} absent from the condition block: hinge inactive"
            info["hinge"] = hinfo["counts"]
            # a loud flag: an empty hinge set makes the M6a term a no-op (the decision-side reads it; task X VL2-07)
            info["hinge"]["active"] = bool(hm.any())
            if not hm.any():
                info["hinge"].setdefault("note", "no eligible training row: hinge inactive (the M6a term is a no-op)")
            info["hinge_ranges"] = hinfo["ranges"]
        if "b" in config.phys_terms:
            states = rows[SG.METAL_COL].to_numpy(dtype=object)
            first = {}
            for i, s in enumerate(states):
                if not SG._missing(s) and str(s) not in first:
                    first[str(s)] = i
            adj = series_adjacent_pairs(first)
            labels = sorted({a for p in adj for a in p}, key=lambda s: SG.metal_properties(s)["Z"])
            sp = np.array([first[s] for s in labels], dtype=np.int64)
            li = {s: k for k, s in enumerate(labels)}
            spairs = np.array([(li[a], li[b]) for a, b in adj], dtype=np.int64).reshape(-1, 2)
            info["smoothness"] = {"n_pairs": int(len(adj)), "pairs": [list(p) for p in adj], "active": bool(len(adj) > 0)}
    tau_std = None if config.tau is None else float(config.tau) / float(y_sd)
    if config.tau is not None:
        info["source"] = {"tau_logD": float(config.tau), "tau_std": tau_std, "y_sd": float(y_sd),
                          "n_groups": encoder.n_groups, "lambda_src": LAMBDA_SRC}
    if row_weights is not None:
        w = np.asarray(row_weights, dtype=np.float64)
        if w.shape != (n,) or (w < 0).any() or not (w > 0).any():
            raise ValueError("row_weights must be one non-negative weight per training row with a positive sum")
        info["row_weights"] = {"n_positive": int((w > 0).sum()), "sum": float(w.sum())}
    return FitAux(n_train=n, pair_positions=pairs, hinge_mask=hm, ext_index=encoder.ext_index,
                  ext_scale=float(encoder.ext_scale), smooth_positions=sp, smooth_pairs=spairs, tau_std=tau_std,
                  row_weights=None if row_weights is None else np.asarray(row_weights, dtype=np.float64), info=info)


# --------------------------------------------------------------------------------------------- #
# training (neural.train_network's loop with the composite loss)
# --------------------------------------------------------------------------------------------- #

@dataclass
class LadderTrainResult(NN.TrainResult):
    loss_terms: dict[str, float] = field(default_factory=dict)
    valid_sd: np.ndarray | None = None


def _weighted_mean(values: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    if w is None:
        return values.mean()
    return (values * w).sum() / w.sum().clamp_min(1e-12)


def composite_loss(net: LadderNet, tt: Mapping[str, torch.Tensor], y_std: torch.Tensor, b: torch.Tensor, *,
                   config: LadderConfig, aux: FitAux, pair_batch: torch.Tensor | None, w: torch.Tensor | None
                   ) -> tuple[torch.Tensor, dict[str, float]]:
    """The mini-batch loss: L_D + source prior + lambda_pair L_pair + lambda_phys (hinge + smoothness) + NLL."""
    terms: dict[str, float] = {}
    batch = {k: v[b] for k, v in tt.items()}
    need_grad = config.lambda_phys > 0 and "a" in config.phys_terms and aux.ext_index is not None and aux.hinge_mask.any()
    if need_grad:
        batch["x_c"] = batch["x_c"].detach().clone().requires_grad_(True)
    e_m, e_l, h = net.trunk(batch)
    mean = net.mean_from(batch, e_m, e_l, h)
    huber = nn.functional.huber_loss(mean, y_std[b], delta=NN.HUBER_DELTA, reduction="none")
    loss = _weighted_mean(huber, w)
    terms["L_D"] = float(loss.detach())
    if config.tau is not None and net.b_group is not None and aux.tau_std is not None:
        prior = LAMBDA_SRC * (net.b_group.weight ** 2).sum() / (2.0 * aux.tau_std ** 2) / float(aux.n_train)
        loss = loss + prior
        terms["L_source"] = float(prior.detach())
    if config.lambda_pair > 0 and pair_batch is not None and len(pair_batch):
        pa = {k: v[pair_batch[:, 0]] for k, v in tt.items()}
        pb = {k: v[pair_batch[:, 1]] for k, v in tt.items()}
        dpred = net(pa) - net(pb)
        dy = y_std[pair_batch[:, 0]] - y_std[pair_batch[:, 1]]
        lp = nn.functional.huber_loss(dpred, dy, delta=NN.HUBER_DELTA)
        loss = loss + float(config.lambda_pair) * lp
        terms["L_pair"] = float(lp.detach())
    if need_grad:
        hmask = torch.from_numpy(aux.hinge_mask)[b]
        if bool(hmask.any()):
            g = torch.autograd.grad(mean.sum(), batch["x_c"], create_graph=True)[0][:, aux.ext_index] / aux.ext_scale
            hinge = torch.relu(-g[hmask]).mean()
            loss = loss + float(config.lambda_phys) * hinge
            terms["L_hinge"] = float(hinge.detach())
            terms["n_hinge_rows_batch"] = float(int(hmask.sum()))
    elif config.lambda_phys > 0 and "a" in config.phys_terms:
        terms["hinge_inactive"] = 1.0                    # the term is configured but has no eligible row (VL2-07)
    if config.lambda_phys > 0 and "b" in config.phys_terms and len(aux.smooth_pairs):
        rep = {k: v[torch.from_numpy(aux.smooth_positions)] for k, v in tt.items()}
        em = net.metal_embedding(rep)
        sm = ((em[aux.smooth_pairs[:, 0]] - em[aux.smooth_pairs[:, 1]]) ** 2).sum()
        loss = loss + float(config.lambda_phys) * sm
        terms["L_smooth"] = float(sm.detach())
    if config.heteroscedastic and net.sd_head is not None:
        sd = net.sd_from(e_m, e_l, h)
        resid = (y_std[b] - mean.detach())
        nll = 0.5 * torch.log(sd ** 2) + 0.5 * resid ** 2 / sd ** 2
        nll_m = _weighted_mean(nll, w)
        loss = loss + LAMBDA_HET * nll_m
        terms["L_nll"] = float(nll_m.detach())
    return loss, terms


def _predict_all(net: LadderNet, t: Mapping[str, torch.Tensor], chunk: int = 8192) -> tuple[np.ndarray, np.ndarray | None]:
    net.eval()
    n = t["z_m"].shape[0]
    means, sds = [], []
    with NN.torch_threads(), torch.no_grad():
        for s in range(0, n, chunk):
            m, sd = net.forward_all({k: v[s:s + chunk] for k, v in t.items()})
            means.append(m.numpy())
            if sd is not None:
                sds.append(sd.numpy())
    mean = np.concatenate(means).astype(np.float64) if means else np.zeros(0)
    sd = np.concatenate(sds).astype(np.float64) if sds else None
    return mean, sd


def train_ladder_network(train: NN.EncodedRows, y_train: np.ndarray, dims: NN.InputDims, config: LadderConfig,
                         aux: FitAux, *, model_seed: int, n_experts: int = 1, n_groups: int = 0,
                         n_epochs: int | None = None, valid: NN.EncodedRows | None = None,
                         y_valid: np.ndarray | None = None, valid_units: np.ndarray | None = None,
                         patience: int = NN.PATIENCE, max_epochs: int = NN.MAX_EPOCHS,
                         y_scale: tuple[float, float] | None = None) -> LadderTrainResult:
    """``neural.train_network``'s loop (same optimiser, schedule, batching, early stopping) with :func:`composite_loss`."""
    t0 = time.perf_counter()
    y_train = np.asarray(y_train, dtype=np.float64)
    if len(y_train) != len(train) or not len(train):
        raise ValueError("train_ladder_network: target length differs from the training rows (or no rows)")
    if not np.isfinite(y_train).all():
        raise ValueError("train_ladder_network: non-finite training target")
    early = valid is not None
    if early:
        if n_epochs is not None:
            raise ValueError("pass n_epochs or validation rows, not both")
        if y_valid is None or valid_units is None or len(y_valid) != len(valid) or len(valid_units) != len(valid):
            raise ValueError("validation rows need y_valid and valid_units")
        if not len(valid):
            raise ValueError("empty validation set")
        if len(train.index.intersection(valid.index)):
            raise AssertionError("a validation row is a training row")
        _, unit_codes = np.unique(np.asarray(valid_units).astype(str), return_inverse=True)
        n_units = int(unit_codes.max()) + 1
        limit = int(max_epochs)
    else:
        if n_epochs is None or int(n_epochs) < 1:
            raise ValueError("n_epochs >= 1 required without validation rows")
        limit = int(n_epochs)
    mu, sd = NN._sorted_stat(y_train) if y_scale is None else (float(y_scale[0]), float(y_scale[1]))
    y_std = torch.from_numpy(((y_train - mu) / sd).astype(np.float32))
    tt = train.tensors()
    tv = valid.tensors() if early else None
    w_all = None if aux.row_weights is None else torch.from_numpy(aux.row_weights.astype(np.float32))
    n = len(train)
    steps_per_epoch = int(math.ceil(n / NN.BATCH_SIZE))
    n_pairs = int(len(aux.pair_positions))
    pairs_t = torch.from_numpy(np.asarray(aux.pair_positions, dtype=np.int64)) if n_pairs else None
    hist: list[tuple[int, float, float]] = []
    last_terms: dict[str, float] = {}
    with NN.deterministic_torch(model_seed):
        net = LadderNet(dims, config, n_experts=n_experts, n_groups=n_groups)
        mean_params = [q for nme, q in net.named_parameters() if not nme.startswith("sd_head")]
        sd_params = [q for nme, q in net.named_parameters() if nme.startswith("sd_head")]
        opt = torch.optim.AdamW(net.parameters(), lr=NN.LEARNING_RATE, weight_decay=float(config.weight_decay))
        gen = torch.Generator().manual_seed(int(model_seed))
        best, best_epoch, best_state, best_pred, best_sd = float("inf"), 0, None, None, None
        step = epoch = 0
        pair_perm, pair_cursor = None, 0
        for epoch in range(1, limit + 1):
            net.train()
            perm = torch.randperm(n, generator=gen)
            if n_pairs and config.lambda_pair > 0:
                pair_perm = torch.randperm(n_pairs, generator=gen)
                pair_cursor = 0
            loss_sum = 0.0
            for s in range(0, n, NN.BATCH_SIZE):
                b = perm[s:s + NN.BATCH_SIZE]
                pb = None
                if pair_perm is not None:
                    take = min(NN.BATCH_SIZE, n_pairs)
                    if pair_cursor + take > n_pairs:
                        pair_cursor = 0
                    pb = pairs_t[pair_perm[pair_cursor:pair_cursor + take]]
                    pair_cursor += take
                for g in opt.param_groups:
                    g["lr"] = NN.warmup_learning_rate(step, steps_per_epoch)
                opt.zero_grad(set_to_none=True)
                loss, terms = composite_loss(net, tt, y_std, b, config=config, aux=aux, pair_batch=pb,
                                             w=None if w_all is None else w_all[b])
                loss.backward()
                nn.utils.clip_grad_norm_(mean_params, NN.GRAD_CLIP_NORM)      # the registered clip on the mean network
                if sd_params:
                    nn.utils.clip_grad_norm_(sd_params, NN.GRAD_CLIP_NORM)    # the SD head clipped on its own
                opt.step()
                loss_sum += float(loss.detach()) * len(b)
                last_terms = terms
                step += 1
            vm = float("nan")
            if early:
                vp, vsd = _predict_all(net, tv)
                vp = vp * sd + mu
                vm = NN.macro_mae(y_valid, vp, unit_codes, n_units)
                if vm < best:
                    best, best_epoch, best_pred = vm, epoch, vp
                    best_sd = None if vsd is None else vsd * sd
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            hist.append((epoch, loss_sum / n, vm))
            if early and epoch - best_epoch >= int(patience):
                break
        if early:
            if best_state is None:
                raise RuntimeError("no finite validation score")
            net.load_state_dict(best_state)
        else:
            best_epoch = epoch
    net.eval()
    return LadderTrainResult(net=net, best_epoch=int(best_epoch), epochs_run=int(epoch),
                             best_valid_macro_mae=float(best) if early else float("nan"),
                             history=pd.DataFrame(hist, columns=["epoch", "train_loss", "valid_macro_mae"]),
                             seconds=time.perf_counter() - t0, y_mean=mu, y_sd=sd, model_seed=int(model_seed),
                             n_parameters=net.n_parameters, valid_pred=best_pred, loss_terms=dict(last_terms),
                             valid_sd=best_sd)


# --------------------------------------------------------------------------------------------- #
# the arm (M3-M6 and one M7 member)
# --------------------------------------------------------------------------------------------- #

def _empty_prediction_frame(n: int) -> pd.DataFrame:
    out = pd.DataFrame({c: pd.Series([np.nan] * n if c in I._FLOAT_COLUMNS else [None] * n,
                                     dtype=float if c in I._FLOAT_COLUMNS else object) for c in I.PREDICTION_COLUMNS})
    return out


class LadderArm:
    """A ladder configuration at a fixed epoch count (the outer refit, a conformal inner refit, or an M7 member).
    Protocol of ``neural.FactorisedArm`` (``clone`` / ``fit`` / ``predict``, ``fit_table`` / ``predict_positions``)."""

    def __init__(self, config: LadderConfig, *, n_epochs: int, model_seed: int, rows: pd.DataFrame | None = None,
                 condition_vectors: pd.DataFrame | None = None, systems: pd.DataFrame | None = None,
                 min_expert_rows: int = MIN_EXPERT_ROWS, row_weights: pd.Series | None = None):
        if int(n_epochs) < 1 or int(n_epochs) > NN.MAX_EPOCHS:
            raise ValueError(f"n_epochs must be in 1..{NN.MAX_EPOCHS}")
        self.config, self.n_epochs, self.model_seed = config, int(n_epochs), int(model_seed)
        self.name = config.step
        self.rows, self.condition_vectors, self.systems = rows, condition_vectors, systems
        self.min_expert_rows = int(min_expert_rows)
        self.row_weights = row_weights
        self.encoder: LadderEncoder | None = None
        self.result: LadderTrainResult | None = None
        self.aux: FitAux | None = None
        self.train_index: pd.Index | None = None
        self._table: I.RowTable | None = None
        self._train_mask: np.ndarray | None = None

    def clone(self) -> "LadderArm":
        return LadderArm(self.config, n_epochs=self.n_epochs, model_seed=self.model_seed, rows=self.rows,
                         condition_vectors=self.condition_vectors, systems=self.systems,
                         min_expert_rows=self.min_expert_rows, row_weights=self.row_weights)

    def fit(self, train_rows: pd.DataFrame, context: I.FitContext | None = None) -> "LadderArm":
        if context is not None and context.hidden_index is not None and len(context.hidden_index):
            n_bad = len(pd.Index(train_rows.index).intersection(pd.Index(context.hidden_index)))
            if n_bad:
                raise AssertionError(f"{n_bad} hidden row(s) among the training rows")
        if context is not None and self.systems is None and context.systems is not None:
            self.systems = context.systems
        self._table, self._train_mask = None, None
        rows = NN._order_rows(train_rows)
        w = None
        if self.row_weights is not None:
            w = self.row_weights.reindex(rows.index).fillna(0.0).to_numpy(dtype=np.float64)
            keep = w > 0
            rows, w = rows.iloc[np.flatnonzero(keep)], w[keep]
            if not len(rows):
                raise ValueError("the bootstrap left no training row")
        y = NN._target(rows)
        cv = NN._cv_for(self.condition_vectors, rows)
        if cv is None:
            cv = N.condition_vector(rows)            # once per fit; the FeatureSet and the loss terms share it
        self.encoder = LadderEncoder(self.config, min_expert_rows=self.min_expert_rows, systems=self.systems).fit(rows, cv)
        enc = self.encoder.transform(rows, cv)
        _, y_sd = NN._sorted_stat(y)
        self.aux = build_fit_aux(rows, enc, self.encoder, self.config, y_sd=y_sd, cv=cv, row_weights=w)
        self.result = train_ladder_network(enc, y, self.encoder.dims, self.config, self.aux, model_seed=self.model_seed,
                                           n_experts=self.encoder.n_experts, n_groups=self.encoder.n_groups,
                                           n_epochs=self.n_epochs)
        self.train_index = pd.Index(rows.index)
        return self

    def fit_table(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> "LadderArm":
        if self.rows is None:
            raise ValueError("LadderArm.fit_table needs the rows store (LadderArm(rows=...))")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (table.n,) or not mask.any():
            raise ValueError("training mask does not match the RowTable (or is empty)")
        forbidden = I.forbidden_mask(table, context)
        if (mask & forbidden).any():
            raise AssertionError(f"{int((mask & forbidden).sum())} hidden row(s) among the training rows")
        if not table.index.isin(self.rows.index).all():
            raise ValueError("the rows store does not cover the RowTable")
        labels = table.index[mask]
        rows = self.rows.loc[labels].drop(columns=[I.TARGET_COL], errors="ignore").assign(**{I.TARGET_COL: table.y[mask]})
        self.fit(rows, context)
        self._table, self._train_mask = table, mask.copy()
        return self

    def _require_fit(self) -> tuple[LadderEncoder, LadderTrainResult]:
        if self.encoder is None or self.result is None:
            raise RuntimeError(f"{self.name}: fit first")
        return self.encoder, self.result

    def predict_arrays(self, query_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None, NN.EncodedRows]:
        """``(mean_logD, sd_logD or None, encoded rows)`` in log D units."""
        enc, res = self._require_fit()
        if not query_rows.index.is_unique:
            raise ValueError("query rows: the index must be unique")
        if len(query_rows.index.intersection(self.train_index)):
            raise AssertionError("a query row is one of the training rows")
        e = enc.transform(query_rows, NN._cv_for(self.condition_vectors, query_rows))
        mean, sd = _predict_all(res.net, e.tensors())
        return mean * res.y_sd + res.y_mean, (None if sd is None else sd * res.y_sd), e

    def predict(self, query_rows: pd.DataFrame) -> pd.DataFrame:
        mean, sd, e = self.predict_arrays(query_rows)
        return self._frame(query_rows, mean, sd, e, n_members=1)

    def _frame(self, query_rows: pd.DataFrame, mean: np.ndarray, sd: np.ndarray | None, e: NN.EncodedRows, *,
               n_members: int, gaussian_sd: np.ndarray | None = None) -> pd.DataFrame:
        enc = self.encoder
        n = len(query_rows)
        out = _empty_prediction_frame(n)
        out["row_id"] = pd.Series(query_rows.index.to_numpy(dtype=object), dtype=object)
        out["mean_logD"] = np.asarray(mean, dtype=float)
        if sd is not None:
            out["std_logD"] = np.asarray(sd, dtype=float)
        out["fallback_level"] = self.name
        seen = e.seen()
        base_seen = [c for c in seen.columns if c in ("series_offset_seen", "ox_offset_seen", "element_offset_seen",
                                                       "system_offset_seen")]
        reason = np.full(n, "", dtype=object)
        for c in base_seen:
            tag = c.removesuffix("_seen") + "_zero"
            miss = ~seen[c].to_numpy(dtype=bool)
            reason[miss] = [f"{r};{tag}" if r else tag for r in reason[miss]]
        out["fallback_reason"] = pd.Series(reason, dtype=object)
        out["model_step"] = self.name
        out["emb_dim"] = int(self.config.emb_dim)
        out["weight_decay"] = float(self.config.weight_decay)
        out["rank"] = int(self.config.rank)
        out["n_epochs"] = int(self.n_epochs)
        out["model_seed"] = int(self.model_seed)
        for c in base_seen:
            out[c] = seen[c].to_numpy(dtype=bool)
        out["ladder_step"] = self.config.step
        out["config_label"] = self.config.label()
        exp = e.ids["expert_id"]
        names = enc.routing.experts if enc.routing is not None else ("shared",)
        out["expert"] = pd.Series([names[int(k)] for k in exp], dtype=object)
        out["group_offset_seen"] = (e.ids["group_id"] != 0) if self.config.tau is not None else False
        out["tau"] = np.nan if self.config.tau is None else float(self.config.tau)
        out["lambda_pair"] = float(self.config.lambda_pair)
        out["lambda_phys"] = float(self.config.lambda_phys)
        out["phys_terms"] = self.config.phys_terms if self.config.lambda_phys > 0 else ""
        out["n_members"] = int(n_members)
        out["gaussian_sd_logD"] = np.nan if gaussian_sd is None else np.asarray(gaussian_sd, dtype=float)
        return out[list(PREDICTION_COLUMNS)]

    def predict_positions(self, positions: np.ndarray) -> pd.DataFrame:
        if self._table is None or self.rows is None:
            raise RuntimeError("predict_positions needs a fit_table fit")
        p = np.asarray(positions, dtype=np.int64)
        if len(p) and self._train_mask[p].any():
            raise AssertionError("a query row is one of the training rows")
        return self.predict(self.rows.loc[self._table.index[p]])

    @property
    def n_parameters(self) -> int:
        return self._require_fit()[1].n_parameters

    def component_activity(self) -> dict[str, bool | None]:
        """Whether the configured M6 terms have anything to act on in this fit: ``None`` when a term is off, else
        whether its hinge set / smoothness pair set is non-empty (task X finding VL2-07)."""
        comp = self.config.components()
        info = self.aux.info if self.aux else {}
        return {"hinge_active": (info.get("hinge") or {}).get("active", False) if comp["physics_hinge"] else None,
                "smoothness_active": (info.get("smoothness") or {}).get("active", False) if comp["physics_smoothness"]
                else None}

    def fit_record(self) -> dict[str, Any]:
        enc, res = self._require_fit()
        return {"step": self.name, "config": self.config.record(), "config_label": self.config.label(),
                "n_epochs": self.n_epochs, "model_seed": self.model_seed, "n_parameters": res.n_parameters,
                "n_train_rows": len(self.train_index), "y_mean": res.y_mean, "y_sd": res.y_sd,
                "encoder": enc.record(), "model_state_digest": NN.state_digest(res.net), "dims": asdict(enc.dims),
                "seconds": res.seconds, "loss_terms_last_batch": res.loss_terms, "aux": self.aux.info if self.aux else {},
                **self.component_activity(),
                "group_offsets_logD": None if res.net.group_offsets() is None else
                (res.net.group_offsets() * res.y_sd).tolist()}


# --------------------------------------------------------------------------------------------- #
# M7: the 5-member ensemble
# --------------------------------------------------------------------------------------------- #

def group_bootstrap_weights(groups: pd.Series, seed: int) -> pd.Series:
    """Publication-group bootstrap: as many group draws with replacement as there are groups; the multiplicity of a
    row's group is its weight (``REGISTRATION_CHOICES['bootstrap']``)."""
    g = groups.astype(object)
    labels = sorted({str(x) for x in g.to_numpy(dtype=object) if not SG._missing(x)})
    if not labels:
        raise ValueError("no publication group to bootstrap")
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(len(labels), size=len(labels), replace=True)
    mult = np.bincount(draws, minlength=len(labels)).astype(np.float64)
    m = dict(zip(labels, mult))
    return pd.Series([m.get(str(x), 0.0) if not SG._missing(x) else 0.0 for x in g.to_numpy(dtype=object)],
                     index=groups.index, dtype=float)


class EnsembleArm:
    """M7: :data:`N_MEMBERS` heteroscedastic :class:`LadderArm` members at the retained configuration, each on a
    publication-group bootstrap of the training rows with its own seed (module docstring)."""

    def __init__(self, config: LadderConfig, *, n_epochs: int, fold_index: int, rows: pd.DataFrame | None = None,
                 condition_vectors: pd.DataFrame | None = None, systems: pd.DataFrame | None = None,
                 min_expert_rows: int = MIN_EXPERT_ROWS, n_members: int = N_MEMBERS, bootstrap: bool = True):
        if not config.heteroscedastic:
            raise ValueError("the M7 ensemble needs a heteroscedastic configuration")
        self.config, self.n_epochs, self.fold_index = config, int(n_epochs), int(fold_index)
        self.rows, self.condition_vectors, self.systems = rows, condition_vectors, systems
        self.min_expert_rows, self.n_members, self.bootstrap = int(min_expert_rows), int(n_members), bool(bootstrap)
        self.name = config.step
        self.members: list[LadderArm] = []
        self.train_index: pd.Index | None = None
        self._table: I.RowTable | None = None
        self._train_mask: np.ndarray | None = None

    @property
    def seeds(self) -> list[int]:
        return [member_seed(self.fold_index, k) for k in range(self.n_members)]

    def clone(self) -> "EnsembleArm":
        return EnsembleArm(self.config, n_epochs=self.n_epochs, fold_index=self.fold_index, rows=self.rows,
                           condition_vectors=self.condition_vectors, systems=self.systems,
                           min_expert_rows=self.min_expert_rows, n_members=self.n_members, bootstrap=self.bootstrap)

    def fit(self, train_rows: pd.DataFrame, context: I.FitContext | None = None) -> "EnsembleArm":
        self._table, self._train_mask = None, None
        gcol = group_column(train_rows)
        self.members = []
        for k, seed in enumerate(self.seeds):
            w = group_bootstrap_weights(train_rows[gcol], seed) if self.bootstrap else None
            m = LadderArm(self.config, n_epochs=self.n_epochs, model_seed=seed, rows=self.rows,
                          condition_vectors=self.condition_vectors, systems=self.systems,
                          min_expert_rows=self.min_expert_rows, row_weights=w)
            self.members.append(m.fit(train_rows, context))
        self.train_index = pd.Index(train_rows.index)
        return self

    def fit_table(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> "EnsembleArm":
        if self.rows is None:
            raise ValueError("EnsembleArm.fit_table needs the rows store")
        mask = np.asarray(mask, dtype=bool)
        forbidden = I.forbidden_mask(table, context)
        if (mask & forbidden).any():
            raise AssertionError("hidden row(s) among the training rows")
        labels = table.index[mask]
        rows = self.rows.loc[labels].drop(columns=[I.TARGET_COL], errors="ignore").assign(**{I.TARGET_COL: table.y[mask]})
        self.fit(rows, context)
        self._table, self._train_mask = table, mask.copy()
        return self

    def predict_arrays(self, query_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, NN.EncodedRows]:
        """``(mean, std, gaussian_sd, encoded)``: ensemble mean, total SD and the mean member SD (log D units)."""
        if not self.members:
            raise RuntimeError("fit first")
        if len(query_rows.index.intersection(self.train_index)):
            raise AssertionError("a query row is one of the training rows")
        means, sds, e = [], [], None
        for m in self.members:
            mu, sd, e_m = m.predict_arrays(query_rows)
            means.append(mu)
            sds.append(sd)
            if e is None:
                e = e_m                              # member 0's encoding names the frame's expert column
        M, S = np.vstack(means), np.vstack(sds)
        mean = M.mean(axis=0)
        var = (S ** 2).mean(axis=0) + M.var(axis=0)
        return mean, np.sqrt(var), np.sqrt((S ** 2).mean(axis=0)), e

    def predict(self, query_rows: pd.DataFrame) -> pd.DataFrame:
        mean, std, gsd, e = self.predict_arrays(query_rows)
        out = self.members[0]._frame(query_rows, mean, std, e, n_members=len(self.members), gaussian_sd=gsd)
        out["model_seed"] = int(self.seeds[0])
        out["fallback_level"] = self.name
        return out

    def predict_positions(self, positions: np.ndarray) -> pd.DataFrame:
        if self._table is None or self.rows is None:
            raise RuntimeError("predict_positions needs a fit_table fit")
        p = np.asarray(positions, dtype=np.int64)
        if len(p) and self._train_mask[p].any():
            raise AssertionError("a query row is one of the training rows")
        return self.predict(self.rows.loc[self._table.index[p]])

    def fit_record(self) -> dict[str, Any]:
        return {"step": self.name, "config": self.config.record(), "n_members": len(self.members),
                "member_seeds": self.seeds, "bootstrap": self.bootstrap, "n_epochs": self.n_epochs,
                "members": [m.fit_record() for m in self.members], **self.component_activity()}

    def component_activity(self) -> dict[str, bool | None]:
        """A term is active for the ensemble when it is active in at least one member (task X finding VL2-07)."""
        acts = [m.component_activity() for m in self.members]
        out: dict[str, bool | None] = {}
        for key in ("hinge_active", "smoothness_active"):
            vals = [a[key] for a in acts]
            out[key] = None if all(v is None for v in vals) else bool(any(bool(v) for v in vals))
        return out


# --------------------------------------------------------------------------------------------- #
# M7 normalised split conformal (cross-fitted inner folds), the section 12 variant
# --------------------------------------------------------------------------------------------- #

class NormalisedCrossFitConformal(I.ConformalWrapper):
    """``ConformalWrapper`` with scores ``|y - mean| / std`` (section 12 normalised-by-SD split conformal) and intervals
    ``mean +- q std``; ``arms_by_fold[j]`` is refitted on every inner split of fold ``j`` (cross-fitted), and the
    interval centre / SD are the outer refit's (attached by the runner to the stored point predictions)."""

    def __init__(self, arms_by_fold: Mapping[int, Any], splitter: Any, guard: str = "nested_certificate"):
        if not arms_by_fold:
            raise ValueError("at least one calibration fold")
        self.arms_by_fold = {int(k): v for k, v in arms_by_fold.items()}
        super().__init__(next(iter(self.arms_by_fold.values())), splitter=splitter, guard=guard)
        self.calibration: EC.ConformalCalibration | None = None
        self.calibration_frame: pd.DataFrame | None = None

    def _calibrate(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> tuple[np.ndarray, list[Any]]:
        splits = self.splitter.splits(table, mask, context)
        if not splits:
            raise ValueError("the inner design produced no calibration split")
        frames, units = [], []
        for sp in splits:
            if int(sp.fold) not in self.arms_by_fold:
                raise AssertionError(f"inner split {sp.unit} belongs to fold {sp.fold}, not a calibration fold")
            if (sp.train_mask & ~mask).any() or not mask[sp.cal_positions].all() or sp.train_mask[sp.cal_positions].any():
                raise AssertionError(f"inner split {sp.unit} is not inside the outer training rows")
            self._verify(table, sp, context)
            FR.assert_not_scored(table.index[sp.cal_positions], context.v6_mask, "conformal calibration set")
            ctx = context.for_training(None, hidden_index=table.index[sp.hidden_positions])
            arm = self.arms_by_fold[int(sp.fold)].clone().fit_table(table, sp.train_mask, ctx)
            pred = arm.predict_positions(sp.cal_positions)
            mean = pred["mean_logD"].to_numpy(dtype=float)
            sd = pred["std_logD"].to_numpy(dtype=float)
            if not np.isfinite(mean).all() or not np.isfinite(sd).all() or (sd <= 0).any():
                raise AssertionError(f"{self.name}: non-finite mean or non-positive SD in {sp.unit}")
            frames.append(pd.DataFrame({"y": table.y[sp.cal_positions], "mean": mean, "sd": sd, "fold": int(sp.fold)},
                                       index=table.index[sp.cal_positions]))
            units.append(sp.unit)
        cal = pd.concat(frames)
        if cal.index.has_duplicates:
            raise AssertionError("a calibration row is scored in two inner splits")
        self.calibration_frame = cal
        return (np.abs(cal["y"].to_numpy() - cal["mean"].to_numpy()) / cal["sd"].to_numpy()), units

    def fit_table(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> "NormalisedCrossFitConformal":
        if self.seeds is not None:
            raise ValueError("single-seed (the job's run seed)")
        self.fit_seed = context.seed
        self.residuals, self.calibration_units = self._calibrate(table, mask, context)
        cal = self.calibration_frame
        outer_test = table.index[~mask]
        self.calibration = EC.fit_split_conformal(cal["y"], cal["mean"], calibration_index=cal.index,
                                                  outer_test_index=outer_test, v6_mask=context.v6_mask,
                                                  design="V5", sd=cal["sd"], method=CONFORMAL_METHOD)
        self.quantiles = {lv: float(self.calibration.quantiles["__all__"][float(lv)]) for lv in I.LEVELS}
        return self

    def _attach(self, pred: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
        return attach_normalised_intervals(pred, self.quantiles, len(self.residuals))

    def record(self) -> dict[str, Any]:
        if self.residuals is None:
            raise RuntimeError("fit first")
        return {"n_calibration": int(len(self.residuals)), "quantiles": {str(k): v for k, v in self.quantiles.items()},
                "n_inner_splits": len(self.calibration_units), "guard": self.guard, "seed": self.fit_seed,
                "calibration_folds": sorted(self.arms_by_fold), "conformal_method": CONFORMAL_METHOD,
                "splitter": getattr(self.splitter, "name", type(self.splitter).__name__),
                "score": "|y - mean_logD| / std_logD (section 12 normalised split conformal)"}


def attach_normalised_intervals(frame: pd.DataFrame, quantiles: Mapping[float, float], n_calibration: int) -> pd.DataFrame:
    """``mean +- q_level * std_logD`` on a stored prediction frame (the M7 interval of section 12)."""
    out = frame.copy()
    mean = out["mean_logD"].to_numpy(dtype=float)
    sd = out["std_logD"].to_numpy(dtype=float)
    if not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError("normalised intervals need a finite positive std_logD on every row")
    for lv in I.LEVELS:
        pct = int(round(lv * 100))
        q = float(quantiles[lv])
        out[f"lower_{pct}"] = mean - q * sd
        out[f"upper_{pct}"] = mean + q * sd
        out[f"conformal_q{pct}"] = q
    out["conformal_n_calibration"] = float(n_calibration)
    out["intervals_status"] = "split_conformal_inner_normalised"
    return out


# --------------------------------------------------------------------------------------------- #
# tuning (neural.tune's structure with LadderConfig grids)
# --------------------------------------------------------------------------------------------- #

@dataclass
class LadderTuningResult:
    step: str
    selected: LadderConfig
    n_epochs: int
    model_seed: int
    scores: pd.DataFrame
    fits: pd.DataFrame
    inner_predictions: pd.DataFrame
    split_unit_errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    readings: dict[str, str] = field(default_factory=lambda: dict(REGISTRATION_CHOICES))
    min_expert_rows: int = MIN_EXPERT_ROWS

    def arm(self, rows: pd.DataFrame | None = None, condition_vectors: pd.DataFrame | None = None,
            systems: pd.DataFrame | None = None) -> LadderArm:
        return LadderArm(self.selected, n_epochs=self.n_epochs, model_seed=self.model_seed, rows=rows,
                         condition_vectors=condition_vectors, systems=systems, min_expert_rows=self.min_expert_rows)


def select_ladder_config(scores: pd.DataFrame, tolerance: float = NN.SELECTION_TOLERANCE) -> int:
    """Row position of the selected configuration (section 7 tie rule, :func:`tie_key`)."""
    s = scores["inner_macro_mae"].to_numpy(dtype=float)
    if not np.isfinite(s).any():
        raise ValueError("no finite configuration score")
    best = float(np.nanmin(s))
    cand = np.flatnonzero(np.isfinite(s) & (s <= best + float(tolerance) + 1e-12))
    keys = [(tie_key(LadderConfig.from_record(_json_load(scores["config_record"].iloc[i])),
                     int(scores["n_parameters"].iloc[i]), int(i)), int(i)) for i in cand]
    return min(keys)[1]


def _json_load(s: Any) -> dict:
    import json

    return json.loads(s) if isinstance(s, str) else dict(s)


def _json_dump(d: Mapping[str, Any]) -> str:
    import json

    return json.dumps(dict(d), sort_keys=True)


def tune_ladder(outer_train_rows: pd.DataFrame, splits: Sequence[NN.ValidationSplit], configs: Sequence[LadderConfig],
                *, model_seed: int, condition_vectors: pd.DataFrame | None = None, systems: pd.DataFrame | None = None,
                min_expert_rows: int = MIN_EXPERT_ROWS, patience: int = NN.PATIENCE, max_epochs: int = NN.MAX_EPOCHS,
                tolerance: float = NN.SELECTION_TOLERANCE, allow_unguarded: bool = False,
                progress: Callable[[str], None] | None = None) -> LadderTuningResult:
    """Section 6 / 7 search of ``configs`` (one ladder step) over the inner ``splits`` of one outer training set."""
    if not splits:
        raise ValueError("tune_ladder: no inner validation split")
    if not configs:
        raise ValueError("tune_ladder: no configuration")
    steps = {c.step for c in configs}
    if len(steps) != 1:
        raise ValueError("tune_ladder: configurations of one ladder step only")
    outer = outer_train_rows
    for sp in splits:
        if sp.guard is None and not allow_unguarded:
            raise AssertionError(f"split {sp.name}: no fold_isolation_check record")
        if not sp.valid_index.isin(outer.index).all() or not sp.train_index.isin(outer.index).all():
            raise AssertionError(f"split {sp.name}: rows outside the outer training rows")
        if len(sp.train_index.intersection(sp.valid_index)) or len(sp.train_index.intersection(sp.hidden_index)):
            raise AssertionError(f"split {sp.name}: a hidden / validation row is an inner training row")
        if not sp.valid_index.isin(sp.hidden_index).all():
            raise AssertionError(f"split {sp.name}: a validation row is not hidden")
    outer_enc = LadderEncoder(configs[0], min_expert_rows=min_expert_rows, systems=systems).fit(
        NN._order_rows(outer), NN._cv_for(condition_vectors, outer))
    fit_recs: list[dict[str, Any]] = []
    per_cfg: dict[int, dict[str, list]] = {ci: {"pred": [], "y": [], "unit": [], "split": [], "label": []}
                                           for ci in range(len(configs))}
    for sp in splits:            # one split in memory at a time; encoder and target scale: its training rows only
        tr = NN._order_rows(outer.loc[sp.train_index])
        cv_tr = NN._cv_for(condition_vectors, tr)
        if cv_tr is None:
            cv_tr = N.condition_vector(tr)
        ytr = NN._target(tr)
        _, y_sd = NN._sorted_stat(ytr)
        va = outer.loc[sp.valid_index]
        cv_va = NN._cv_for(condition_vectors, va)
        if cv_va is None:
            cv_va = N.condition_vector(va)
        yva = NN._target(va)
        shared = NN.NeuralEncoder().fit(tr, cv_tr)      # the M1 / M2 inputs do not depend on the searched field
        for ci, cfg in enumerate(configs):
            enc = LadderEncoder(cfg, min_expert_rows=min_expert_rows, systems=systems, neural=shared).fit(tr, cv_tr)
            etr = enc.transform(tr, cv_tr)
            eva = enc.transform(va, cv_va)
            aux = build_fit_aux(tr, etr, enc, cfg, y_sd=y_sd, cv=cv_tr)
            if cfg.lambda_pair > 0 and len(aux.pair_positions):
                pf = pd.DataFrame({"idx_a": tr.index[aux.pair_positions[:, 0]], "idx_b": tr.index[aux.pair_positions[:, 1]],
                                   "fold": "train"})
                assert_pairs_inside(pf, tr.index, sp.hidden_index)
            r = train_ladder_network(etr, ytr, enc.dims, cfg, aux, model_seed=model_seed, n_experts=enc.n_experts,
                                     n_groups=enc.n_groups, valid=eva, y_valid=yva, valid_units=sp.valid_units,
                                     patience=patience, max_epochs=max_epochs)
            fit_recs.append({"config": cfg.label(), "config_pos": ci, "split": sp.name, "inner_fold": sp.inner_fold,
                             "n_train_rows": len(etr), "n_valid_rows": len(eva),
                             "n_valid_units": len(set(np.asarray(sp.valid_units).astype(str))),
                             "best_epoch": r.best_epoch, "epochs_run": r.epochs_run,
                             "valid_macro_mae": r.best_valid_macro_mae, "n_parameters": r.n_parameters,
                             "seconds": r.seconds, "n_pairs": int(len(aux.pair_positions)),
                             "n_hinge_rows": int(aux.hinge_mask.sum()), "n_smooth_pairs": int(len(aux.smooth_pairs)),
                             "n_experts": enc.n_experts, "n_groups": enc.n_groups})
            acc = per_cfg[ci]
            acc["pred"].append(r.valid_pred)
            acc["y"].append(yva)
            acc["unit"].append(np.asarray(sp.valid_units).astype(str))
            acc["split"].append(np.array([sp.name] * len(va), dtype=object))
            acc["label"].append(np.asarray(va.index, dtype=object))
            if progress is not None:
                progress(f"{cfg.label()} {sp.name}: best epoch {r.best_epoch}, {r.best_valid_macro_mae:.4f}, "
                         f"{r.seconds:.1f}s")
            del etr, eva
    sue = []
    fold_of_split = {sp.name: int(sp.inner_fold) for sp in splits}
    for ci, cfg in enumerate(configs):
        acc = per_cfg[ci]
        for pr_s, y_s, u_s, sn_s in zip(acc["pred"], acc["y"], acc["unit"], acc["split"]):
            t = pd.DataFrame({"unit": u_s, "e": np.abs(np.asarray(y_s, dtype=np.float64) - pr_s)}).groupby(
                "unit", sort=True)["e"]
            name = str(sn_s[0]) if len(sn_s) else ""
            for unit, n_, tot in zip(t.size().index, t.size().to_numpy(), t.sum().to_numpy()):
                sue.append({"config": cfg.label(), "split": name, "inner_fold": fold_of_split.get(name, -1),
                            "unit": str(unit), "n_rows": int(n_), "sum_abs_error": float(tot)})
    sue_frame = pd.DataFrame(sue, columns=["config", "split", "inner_fold", "unit", "n_rows", "sum_abs_error"])
    fold_scores = ID.fold_macro_table(sue_frame)
    fold_mean = ID.fold_mean_scores(sue_frame)
    score_recs, pooled = [], {}
    for ci, cfg in enumerate(configs):
        acc = per_cfg[ci]
        u = np.concatenate(acc["unit"])
        _, codes = np.unique(u, return_inverse=True)
        pr, yv = np.concatenate(acc["pred"]), np.concatenate(acc["y"])
        pooled[ci] = (pr, yv, u, np.concatenate(acc["split"]), np.concatenate(acc["label"]))
        best_epochs = [rec["best_epoch"] for rec in fit_recs if rec["config_pos"] == ci]
        score_recs.append({"config": cfg.label(), "config_record": _json_dump(cfg.record()), "step": cfg.step,
                           "emb_dim": cfg.emb_dim, "weight_decay": cfg.weight_decay, "rank": cfg.rank,
                           "experts": cfg.experts, "tau": np.nan if cfg.tau is None else cfg.tau,
                           "lambda_pair": cfg.lambda_pair, "lambda_phys": cfg.lambda_phys, "phys_terms": cfg.phys_terms,
                           "heteroscedastic": cfg.heteroscedastic,
                           "n_parameters": parameter_count(outer_enc.dims, cfg, n_experts=outer_enc.n_experts,
                                                           n_groups=outer_enc.n_groups),
                           "inner_macro_mae": float(fold_mean[cfg.label()]),
                           "pooled_unit_macro_mae_diagnostic": NN.macro_mae(yv, pr, codes),
                           "n_units": int(codes.max()) + 1, "n_splits": len(splits),
                           "n_inner_folds": int((fold_scores["config"] == cfg.label()).sum()),
                           "median_best_epoch": NN.median_epochs(best_epochs)})
    scores = pd.DataFrame(score_recs)
    pos = select_ladder_config(scores, tolerance)
    scores["selected"] = np.arange(len(scores)) == pos
    pr, yv, uv, sn, lab = pooled[pos]
    inner = pd.DataFrame({"split": sn, "row_label": lab, "unit": uv, "log_D": yv, "pred": pr})
    return LadderTuningResult(step=next(iter(steps)), selected=configs[pos], n_epochs=int(scores["median_best_epoch"].iloc[pos]),
                              model_seed=int(model_seed), scores=scores, fits=pd.DataFrame(fit_recs),
                              inner_predictions=inner, split_unit_errors=sue_frame, fold_scores=fold_scores,
                              min_expert_rows=int(min_expert_rows))


def select_ladder_excluding_folds(scores: Sequence[Mapping[str, Any]], fits: Sequence[Mapping[str, Any]],
                                  split_unit_errors: Sequence[Mapping[str, Any]], exclude_folds: Iterable[int], *,
                                  tolerance: float = NN.SELECTION_TOLERANCE) -> tuple[LadderConfig, int, dict[str, Any]]:
    """``neural.select_excluding_folds`` for ladder records (the cross-fitted calibration of addendum 1 item 2)."""
    drop = {int(f) for f in exclude_folds}
    sc, fi, ue = pd.DataFrame(list(scores)), pd.DataFrame(list(fits)), pd.DataFrame(list(split_unit_errors))
    if sc.empty or fi.empty or ue.empty:
        raise ValueError("the tuning record lacks scores / fits / split_unit_errors")
    kept = ue[~ue["inner_fold"].astype(int).isin(drop)]
    fi = fi[~fi["inner_fold"].astype(int).isin(drop)]
    if kept.empty:
        raise ValueError(f"no inner split outside folds {sorted(drop)}")
    macro = pd.Series(ID.fold_mean_scores(ue, exclude_folds=drop), dtype=float)
    sub = sc.drop(columns=[c for c in ("inner_macro_mae", "selected") if c in sc.columns]).copy()
    sub["inner_macro_mae"] = sub["config"].map(macro).astype(float)
    sub = sub[np.isfinite(sub["inner_macro_mae"].to_numpy(dtype=float))].reset_index(drop=True)
    pos = select_ladder_config(sub, tolerance)
    row = sub.iloc[pos]
    cfg = LadderConfig.from_record(_json_load(row["config_record"]))
    epochs = NN.median_epochs(fi.loc[fi["config"] == row["config"], "best_epoch"].astype(int).tolist())
    return cfg, epochs, {"excluded_inner_folds": sorted(drop), "config": str(row["config"]),
                         "inner_macro_mae": float(row["inner_macro_mae"]), "n_epochs": int(epochs),
                         "n_splits": int(kept["split"].nunique()),
                         "inner_folds_used": sorted(kept["inner_fold"].astype(int).unique().tolist()),
                         "score": "mean over inner folds of the fold's unit-macro MAE (addendum 1 item 2)"}


def tune_step(step: str, outer_train_rows: pd.DataFrame, splits: Sequence[NN.ValidationSplit], prev: LadderConfig, *,
              fold_index: int, **kwargs: Any) -> LadderTuningResult:
    """One ladder step's search on one outer fold: :func:`step_grid` of ``prev`` (the fold's retained predecessor),
    scope-checked, model seed from ``fold_index`` (section 15)."""
    if step not in ("M3", "M4", "M5", "M6"):
        raise ValueError(f"tune_step handles M3-M6 (M7 has no search); got {step!r}")
    configs = step_grid(step, prev)
    assert_search_scope(step, configs, prev)
    return tune_ladder(outer_train_rows, splits, configs, model_seed=NN.registered_model_seed(fold_index), **kwargs)

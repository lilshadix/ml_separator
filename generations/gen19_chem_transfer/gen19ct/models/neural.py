"""``models/neural.py`` -- the factorised neural ladder steps M1 and M2 (PyTorch, CPU).

Registered text (``preregistration.md``, sealed 2026-09-15): section 6 ladder rows M1 and M2, the "M-model training
settings (resolved by the orchestrator, 2026-09-15, before any M fit)", the allowed-family limits (<= 200 k parameters,
embeddings <= 16 dimensions, <= 2 hidden layers of <= 128 units); section 7 (inner folds only, selection criterion,
tolerance 0.005 toward the smaller configuration, compute plan); section 15 (model seed rule); section 2 "Features";
brief sections 7, 12 and 15.  Nothing in this module reads a registered fold file, scores an outer fold or touches V6.

Model (module constants below)
------------------------------
Inputs come from ``features.FeatureSet.for_arm("M1")`` (one-hot categoricals, training-median imputation with
``__missing`` indicators, training mean / SD standardisation), fitted on the rows passed to ``fit`` only:
``z_m`` = every ``metal__*`` column, ``z_l`` = every ``extractant__*`` column, ``x_cond`` = every ``condition__*``
column; embedding ids ``series_id`` (Ln / An / other), ``ox_id``, ``element_id`` (metal) and ``system_id`` (the
``extractant_system_key`` vocabulary), index 0 = a value absent from the training rows.

* **M1** ``e_m = A z_m + e_series[s] + e_ox[o] + e_element[z]``, ``e_l = W_l z_l + delta_l[system]``,
  ``h = SiLU(Lin(SiLU(Lin(x_cond, 32)), 32))``, ``y = head([e_m, e_l, h])`` with
  ``head = Lin(64) SiLU Dropout(0.1) Lin(64) SiLU Dropout(0.1) Lin(1)``.
* **M2** M1 plus ``I(m, l) = e_m^T P Q^T e_l`` (``P, Q`` of shape ``emb_dim x rank``, so ``rank(W) <= min(rank,
  emb_dim)``) added to the head output.

An offset row of index 0 is a fixed zero vector (``padding_idx``): an unseen metal series / oxidation state / element
or an unseen system contributes no offset, only its descriptor map.  The ``<NA>`` oxidation-state token of an X(?) row
is mapped to index 0 as well (:data:`REGISTRATION_CHOICES` ``unknown_ox_offset``).

Training (fixed for every M model by the section 6 resolution)
--------------------------------------------------------------
AdamW, learning rate 3e-3 with a linear warm-up over the first 10 epochs (per optimizer step), then constant; mini-batch
512 rows, the permutation of every epoch drawn by a ``torch.Generator`` seeded with the model seed; float32;
``torch.set_num_threads(2)`` and ``torch.use_deterministic_algorithms(True)`` inside :func:`deterministic_torch`
(restored afterwards); target standardised with the mean / SD of the fit's own training rows; Huber loss (delta = 1,
standardised units); gradient-norm clip 5.  With validation rows: the macro MAE (log D units, equal weight per
validation unit) after every epoch, patience 30, at most 300 epochs, the best epoch's parameters restored.  Without
validation rows: exactly ``n_epochs`` epochs.

Tuning (section 6 "Searched", section 7)
----------------------------------------
:func:`tune_m1` searches ``emb_dim in {4, 8, 16} x weight_decay in {1e-4, 1e-3, 1e-2}``; :func:`tune_m2` searches
``rank in {2, 4, 8}`` with the M1 values retained for the same outer fold.  Every configuration is fitted with early
stopping on each :class:`ValidationSplit` (an inner split of the outer training rows: its encoder, target scaling and
network are fitted on the split's training rows only).  The configuration score is the macro MAE over the validation
units of the pooled best-epoch predictions of all splits; configurations within 0.005 of the best are resolved toward
the smaller configuration (lower rank, fewer parameters, stronger penalty).  The outer refit
(:meth:`TuningResult.arm`) runs for the median best-epoch count of the selected configuration's inner fits.
:func:`splits_from_folds` turns the fold builders' inner folds (``cell_holdout.inner_cells_V5``,
``source_holdout.inner_folds_V1``, ``metal_holdout.inner_metals_V2``) into validation splits.

Arm protocol (``models.interface``)
-----------------------------------
:class:`FactorisedArm` (``clone`` / ``fit`` / ``predict``, plus the ``fit_table`` / ``predict_positions`` fast path of
``ConformalWrapper`` when constructed with a ``rows`` store covering the RowTable) returns
``interface.PREDICTION_COLUMNS`` (``std_logD`` and the intervals NaN: intervals come from ``ConformalWrapper``;
``fallback_level`` = the step name, ``fallback_reason`` = the offsets that were unseen) followed by
:data:`NEURAL_DIAGNOSTIC_COLUMNS`.  :meth:`FactorisedArm.embeddings`, :meth:`~FactorisedArm.metal_embeddings` and
:meth:`~FactorisedArm.system_embeddings` expose ``e_m`` / ``e_l`` for the reliability and Procrustes analyses.

Where the registration is silent, the reading implemented here is listed in :data:`REGISTRATION_CHOICES`.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from gen19ct.chemistry import support_graph as SG
from gen19ct.data import load as LOAD
from gen19ct.folds import io as FI
from gen19ct.folds import registered as FR
from gen19ct.models import features as F
from gen19ct.models import interface as I

# --------------------------------------------------------------------------------------------- #
# registered constants
# --------------------------------------------------------------------------------------------- #

STEPS: tuple[str, ...] = ("M1", "M2")
LEARNING_RATE = 3e-3
WARMUP_EPOCHS = 10
BATCH_SIZE = 512
HUBER_DELTA = 1.0
GRAD_CLIP_NORM = 5.0
PATIENCE = 30
MAX_EPOCHS = 300
N_THREADS = 2
COND_HIDDEN: tuple[int, ...] = (32, 32)
HEAD_HIDDEN: tuple[int, ...] = (64, 64)
DROPOUT = 0.1
EMB_DIM_GRID: tuple[int, ...] = (4, 8, 16)
WEIGHT_DECAY_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
RANK_GRID: tuple[int, ...] = (2, 4, 8)
SELECTION_TOLERANCE = 0.005
#: section 6 allowed-family limits
MAX_PARAMETERS = 200_000
MAX_EMB_DIM = 16
MAX_HIDDEN_LAYERS = 2
MAX_HIDDEN_UNITS = 128
#: section 15 model seed rule ``42 + fold * 1009 + 9,999,991``
MODEL_SEED_BASE = 42
MODEL_SEED_FOLD_STEP = 1009
MODEL_SEED_OFFSET = 9_999_991

FEATURE_PRESET = "M1"
METAL_BLOCK, EXTRACTANT_BLOCK, CONDITION_BLOCK = "metal", "extractant", "condition"
METAL_OFFSET_IDS: tuple[str, ...] = ("series_id", "ox_id", "element_id")
SYSTEM_OFFSET_ID = "system_id"
OFFSET_IDS: tuple[str, ...] = METAL_OFFSET_IDS + (SYSTEM_OFFSET_ID,)
#: embedding ids whose ``<NA>`` token is mapped to the zero offset
NA_TO_ZERO_IDS: tuple[str, ...] = ("series_id", "ox_id", "element_id")

NEURAL_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "model_step", "emb_dim", "weight_decay", "rank", "n_epochs", "model_seed", "series_offset_seen", "ox_offset_seen",
    "element_offset_seen", "system_offset_seen",
)
PREDICTION_COLUMNS: tuple[str, ...] = I.PREDICTION_COLUMNS + NEURAL_DIAGNOSTIC_COLUMNS

REGISTRATION_CHOICES: dict[str, str] = {
    "inputs": "z_m / z_l / x_cond = every column of the metal / extractant / condition block of the FeatureSet 'M1' "
              "preset (numeric, __missing indicators, one-hot categoricals; median-imputed and standardised on the "
              "fit's training rows)",
    "linear_maps": "e_shared(z_m) = A z_m and W_l z_l are linear maps without bias (the head carries the biases)",
    "e_series": "series_id = Ln / An / other (features.REGISTRATION_CHOICES['e_series']); brief section 15's five "
                "categories are not a separate offset",
    "unknown_ox_offset": "the <NA> token of series_id / ox_id / element_id (an X(?) row has no oxidation state) maps "
                         "to the zero offset, like an unseen value: no learned 'unknown state' offset, no state imputed",
    "offset_init": "free offsets (e_series, e_ox, e_element, delta_l) start at 0; A, W_l and the MLPs use the PyTorch "
                   "default initialisation; P and Q start N(0, 0.1^2)",
    "weight_decay_scope": "AdamW weight decay applies to every parameter (offsets and biases included)",
    "warmup": "linear in the optimizer step over the first 10 epochs: lr_t = 3e-3 * min(1, (t + 1) / (10 * "
              "steps_per_epoch)); constant afterwards",
    "batches": "last partial mini-batch kept; training rows ordered by canonical_measurement_id (else the index "
               "label) before the seeded permutation, so a fit does not depend on the frame's row order",
    "loss_weights": "every training row weight 1 in the Huber loss (row-weighted mean over the mini-batch)",
    "target_scaling": "each fit standardises the target with the mean / SD (ddof 0) of its OWN training rows: the "
                      "outer-training rows for the outer refit, the inner-training rows for an inner tuning or "
                      "calibration fit.  The section 6 M-model resolution says 'the outer-training mean and SD'; read "
                      "literally for an inner fit that would put inner validation targets into the inner fit's scaling, "
                      "against section 2 / brief section 12 ('fitted on the training rows of the fold it serves').  The "
                      "no-leak reading is implemented and needs a POST-HOC addendum (task X finding V-08)",
    "early_stopping": "strict improvement of the validation macro MAE (first best epoch kept on ties); stop when 30 "
                      "epochs pass without one or after 300 epochs",
    "validation_unit": "the unit of each validation row is the fold builder's row_unit (V5 inner: the hidden cell; V1 "
                       "inner: the publication group; V2 inner: the metal state)",
    "config_score": "macro MAE over validation units of the pooled best-epoch predictions of every inner split",
    "tie_order": "within 0.005 of the best: smallest (rank, number of parameters on the outer-training dimensions, "
                 "-weight_decay)",
    "refit_epochs": "median of the selected configuration's best-epoch counts over all its inner fits (every V5 inner "
                    "batch is one fit), rounded half up, at least 1",
    "model_seed": "42 + fold_index * 1009 + 9,999,991 with fold_index the caller's fold number (discovery: the position "
                  "of the (discovery seed, outer fold) pair in the design enumerated once per discovery seed, "
                  "evaluation.discovery.fold_ordinals, so seed 104729 keeps the fold's file position and the other "
                  "seeds get distinct model seeds); every inner fit, calibration fit and the outer refit of one outer "
                  "fold use this seed (common random numbers across configurations)",
    "system_embedding_table": "e_l of a system = mean of its training rows' e_l (share-weighted descriptors vary "
                              "between rows of a multi-component system); e_m of a metal state is row-invariant "
                              "(asserted)",
    "m2_rank": "rank(W) <= min(rank, emb_dim): with emb_dim 4 the ranks 8 and 4 are the same model class",
}


def registered_model_seed(fold_index: int) -> int:
    """Section 15: ``42 + fold * 1009 + 9,999,991``."""
    if int(fold_index) < 0:
        raise ValueError("fold_index must be >= 0")
    return MODEL_SEED_BASE + int(fold_index) * MODEL_SEED_FOLD_STEP + MODEL_SEED_OFFSET


@contextmanager
def torch_threads() -> Iterator[None]:
    """``torch.set_num_threads(2)`` and deterministic algorithms for inference (no RNG is drawn); restored on exit."""
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(N_THREADS)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_det)
        torch.set_num_threads(prev_threads)


@contextmanager
def deterministic_torch(seed: int) -> Iterator[None]:
    """``torch.set_num_threads(2)``, deterministic algorithms and ``torch.manual_seed(seed)`` on a forked CPU RNG;
    the previous thread count, determinism flag and global RNG state are restored on exit."""
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(N_THREADS)
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            yield
    finally:
        torch.use_deterministic_algorithms(prev_det)
        torch.set_num_threads(prev_threads)


# --------------------------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class NeuralConfig:
    """One searched configuration.  ``rank = 0`` is M1 (no bilinear term)."""

    emb_dim: int
    weight_decay: float
    rank: int = 0

    def __post_init__(self) -> None:
        if not (1 <= int(self.emb_dim) <= MAX_EMB_DIM):
            raise ValueError(f"emb_dim {self.emb_dim} outside 1..{MAX_EMB_DIM}")
        if float(self.weight_decay) < 0:
            raise ValueError("weight_decay must be >= 0")
        if int(self.rank) < 0:
            raise ValueError("rank must be >= 0")

    @property
    def step(self) -> str:
        return "M2" if self.rank > 0 else "M1"

    def label(self) -> str:
        return f"{self.step}_d{self.emb_dim}_wd{self.weight_decay:g}" + (f"_r{self.rank}" if self.rank else "")


def m1_grid() -> list[NeuralConfig]:
    """The 9 M1 configurations (section 6)."""
    return [NeuralConfig(d, wd, 0) for d in EMB_DIM_GRID for wd in WEIGHT_DECAY_GRID]


def m2_grid(m1_config: NeuralConfig) -> list[NeuralConfig]:
    """The 3 M2 configurations with the M1 values retained for this outer fold (section 6)."""
    if m1_config.rank != 0:
        raise ValueError("m2_grid needs the retained M1 configuration (rank 0)")
    return [NeuralConfig(m1_config.emb_dim, m1_config.weight_decay, r) for r in RANK_GRID]


# --------------------------------------------------------------------------------------------- #
# encoding (fitted on training rows only)
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class InputDims:
    p_metal: int
    p_ligand: int
    p_condition: int
    n_series: int
    n_ox: int
    n_element: int
    n_system: int


@dataclass
class EncodedRows:
    """Float32 input blocks and int64 offset ids of a set of rows (row order = ``index``)."""

    index: pd.Index
    z_m: np.ndarray
    z_l: np.ndarray
    x_c: np.ndarray
    ids: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.index)

    def take(self, positions: np.ndarray) -> "EncodedRows":
        p = np.asarray(positions, dtype=np.int64)
        return EncodedRows(self.index[p], self.z_m[p], self.z_l[p], self.x_c[p], {k: v[p] for k, v in self.ids.items()})

    def tensors(self) -> dict[str, torch.Tensor]:
        out = {"z_m": torch.from_numpy(np.ascontiguousarray(self.z_m)),
               "z_l": torch.from_numpy(np.ascontiguousarray(self.z_l)),
               "x_c": torch.from_numpy(np.ascontiguousarray(self.x_c))}
        out.update({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in self.ids.items()})
        return out

    def seen(self) -> pd.DataFrame:
        return pd.DataFrame({f"{k.removesuffix('_id')}_offset_seen": v != F.RESERVED_INDEX for k, v in self.ids.items()},
                            index=self.index)


def assert_inputs_allowed(columns: Iterable[str], source_columns: Iterable[str]) -> None:
    """Section 2 'Features': no provenance column, publication / study id, DOI, row id or target is an input column
    or a column the encoder reads (raises ``AssertionError``)."""
    cols, src = list(columns), list(source_columns)
    prov = set(LOAD.PROVENANCE_COLUMNS) | set(F.ID_COLUMNS) | set(F.TARGET_COLUMNS)
    bad_src = sorted(set(src) & prov)
    base = {c.split("__", 1)[1].split("=", 1)[0].removesuffix("__missing") if "__" in c else c for c in cols}
    bad_cols = sorted(base & prov)
    if bad_src or bad_cols:
        raise AssertionError(f"provenance / id / target columns would enter M1/M2: sources {bad_src}, inputs {bad_cols}")
    try:
        F.assert_feature_columns_allowed(cols, "M1/M2 input columns")
        F.assert_feature_columns_allowed(src, "M1/M2 source columns")
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc


class NeuralEncoder:
    """``FeatureSet.for_arm("M1")`` split into the three blocks plus the offset ids (module docstring)."""

    def __init__(self) -> None:
        self.features = F.FeatureSet.for_arm(FEATURE_PRESET)
        self.block_columns: dict[str, tuple[str, ...]] = {}
        self.id_sizes: dict[str, int] = {}
        self.na_index: dict[str, int] = {}
        self.dims: InputDims | None = None

    @property
    def state_digest(self) -> str:
        return self.features.state_digest

    def fit(self, rows: pd.DataFrame, cv: pd.DataFrame | None = None) -> "NeuralEncoder":
        self.features.fit(rows, cv)
        sizes = self.features.id_sizes()
        self.id_sizes = {k: int(sizes[k]) for k in OFFSET_IDS}
        metal = next(b for b in self.features.blocks if isinstance(b, F.MetalBlock))
        self.na_index = {k: int(metal.vocab[k].index([F.NA_TOKEN])[0]) for k in NA_TO_ZERO_IDS}
        fm = self.features.transform(rows.iloc[:1], None if cv is None else cv.loc[rows.index[:1]])
        self.block_columns = {b: tuple(fm.block_columns[b]) for b in (METAL_BLOCK, EXTRACTANT_BLOCK, CONDITION_BLOCK)}
        assert_inputs_allowed([c for cols in self.block_columns.values() for c in cols], self.features.source_columns)
        self.dims = InputDims(len(self.block_columns[METAL_BLOCK]), len(self.block_columns[EXTRACTANT_BLOCK]),
                              len(self.block_columns[CONDITION_BLOCK]), self.id_sizes["series_id"],
                              self.id_sizes["ox_id"], self.id_sizes["element_id"], self.id_sizes[SYSTEM_OFFSET_ID])
        return self

    def transform(self, rows: pd.DataFrame, cv: pd.DataFrame | None = None) -> EncodedRows:
        if self.dims is None:
            raise RuntimeError("NeuralEncoder: fit first")
        fm = self.features.transform(rows, cv)
        for b, cols in self.block_columns.items():
            if tuple(fm.block_columns[b]) != cols:
                raise AssertionError(f"NeuralEncoder: {b} columns differ from the fitted columns")
        used = [c for cols in self.block_columns.values() for c in cols]
        assert_inputs_allowed(used, self.features.source_columns)
        if fm.categorical_columns:
            raise AssertionError("NeuralEncoder: categorical token columns in the M1 preset")
        blocks = {b: fm.frame.loc[:, list(cols)].to_numpy(dtype=np.float32) for b, cols in self.block_columns.items()}
        for b, arr in blocks.items():
            if not np.isfinite(arr).all():
                raise AssertionError(f"NeuralEncoder: non-finite {b} input after imputation")
        ids = {}
        for k in OFFSET_IDS:
            v = np.asarray(fm.ids[k], dtype=np.int64).copy()
            if k in self.na_index and self.na_index[k] != F.RESERVED_INDEX:
                v[v == self.na_index[k]] = F.RESERVED_INDEX
            if v.min(initial=0) < 0 or v.max(initial=0) >= self.id_sizes[k]:
                raise AssertionError(f"NeuralEncoder: {k} outside its vocabulary")
            ids[k] = v
        return EncodedRows(rows.index, blocks[METAL_BLOCK], blocks[EXTRACTANT_BLOCK], blocks[CONDITION_BLOCK], ids)


# --------------------------------------------------------------------------------------------- #
# the network
# --------------------------------------------------------------------------------------------- #

def _mlp(sizes: Sequence[int], dropout: float | None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.SiLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


class FactorisedNet(nn.Module):
    """M1 (``rank = 0``) or M2 (module docstring).  Construct inside :func:`deterministic_torch`."""

    def __init__(self, dims: InputDims, config: NeuralConfig):
        super().__init__()
        d, r = int(config.emb_dim), int(config.rank)
        if len(COND_HIDDEN) > MAX_HIDDEN_LAYERS or len(HEAD_HIDDEN) > MAX_HIDDEN_LAYERS or \
                max(COND_HIDDEN + HEAD_HIDDEN) > MAX_HIDDEN_UNITS:
            raise AssertionError("section 6: at most 2 hidden layers of at most 128 units")
        self.dims, self.config = dims, config
        self.e_shared = nn.Linear(dims.p_metal, d, bias=False)
        self.e_series = nn.Embedding(dims.n_series, d, padding_idx=F.RESERVED_INDEX)
        self.e_ox = nn.Embedding(dims.n_ox, d, padding_idx=F.RESERVED_INDEX)
        self.e_element = nn.Embedding(dims.n_element, d, padding_idx=F.RESERVED_INDEX)
        self.w_l = nn.Linear(dims.p_ligand, d, bias=False)
        self.delta_l = nn.Embedding(dims.n_system, d, padding_idx=F.RESERVED_INDEX)
        for emb in (self.e_series, self.e_ox, self.e_element, self.delta_l):
            nn.init.zeros_(emb.weight)
        self.cond = _mlp((dims.p_condition,) + COND_HIDDEN, None)
        self.head = nn.Sequential(_mlp((2 * d + COND_HIDDEN[-1],) + HEAD_HIDDEN, DROPOUT), nn.Linear(HEAD_HIDDEN[-1], 1))
        if r:
            self.P = nn.Parameter(torch.randn(d, r) * 0.1)
            self.Q = nn.Parameter(torch.randn(d, r) * 0.1)
        else:
            self.register_parameter("P", None)
            self.register_parameter("Q", None)
        n = count_parameters(self)
        if n > MAX_PARAMETERS:
            raise AssertionError(f"section 6: {n} parameters > {MAX_PARAMETERS}")
        self.n_parameters = n

    def metal_embedding(self, t: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.e_shared(t["z_m"]) + self.e_series(t["series_id"]) + self.e_ox(t["ox_id"]) + \
            self.e_element(t["element_id"])

    def ligand_embedding(self, t: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.w_l(t["z_l"]) + self.delta_l(t[SYSTEM_OFFSET_ID])

    def interaction(self, e_m: torch.Tensor, e_l: torch.Tensor) -> torch.Tensor:
        if self.P is None:
            return torch.zeros(e_m.shape[0], dtype=e_m.dtype)
        return ((e_m @ self.P) * (e_l @ self.Q)).sum(dim=1)

    def forward(self, t: Mapping[str, torch.Tensor]) -> torch.Tensor:
        e_m, e_l = self.metal_embedding(t), self.ligand_embedding(t)
        out = self.head(torch.cat([e_m, e_l, self.cond(t["x_c"])], dim=1)).squeeze(1)
        return out + self.interaction(e_m, e_l) if self.P is not None else out

    def bilinear_matrix(self) -> np.ndarray | None:
        """``W = P Q^T`` (``emb_dim x emb_dim``) or None for M1."""
        if self.P is None:
            return None
        with torch.no_grad():
            return (self.P @ self.Q.T).numpy().astype(np.float64)


def parameter_count(dims: InputDims, config: NeuralConfig) -> int:
    """Parameters of the network for ``dims`` (no RNG consumed outside a fork)."""
    with torch.random.fork_rng(devices=[]):
        return FactorisedNet(dims, config).n_parameters


def state_digest(net: nn.Module) -> str:
    """SHA-256 over the parameter names, shapes and float32 bytes."""
    h = hashlib.sha256()
    for k, v in net.state_dict().items():
        a = v.detach().cpu().numpy()
        h.update(k.encode() + str(a.shape).encode() + np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------------------------- #

def _sorted_stat(y: np.ndarray) -> tuple[float, float]:
    ys = np.sort(np.asarray(y, dtype=np.float64))
    mu = float(np.mean(ys))
    sd = float(np.sqrt(np.mean(np.sort((ys - mu) ** 2))))
    return mu, (sd if np.isfinite(sd) and sd > 0 else 1.0)


def macro_mae(y: np.ndarray, pred: np.ndarray, unit_codes: np.ndarray, n_units: int | None = None) -> float:
    """Equal-weight mean over units of the per-unit MAE."""
    codes = np.asarray(unit_codes, dtype=np.int64)
    n = int(codes.max()) + 1 if n_units is None else int(n_units)
    err = np.abs(np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64))
    s = np.bincount(codes, weights=err, minlength=n)
    c = np.bincount(codes, minlength=n)
    ok = c > 0
    return float(np.mean(s[ok] / c[ok]))


@dataclass
class TrainResult:
    net: FactorisedNet
    best_epoch: int
    epochs_run: int
    best_valid_macro_mae: float
    history: pd.DataFrame
    seconds: float
    y_mean: float
    y_sd: float
    model_seed: int
    n_parameters: int
    valid_pred: np.ndarray | None = None


def _predict_std(net: FactorisedNet, t: Mapping[str, torch.Tensor], chunk: int = 8192) -> np.ndarray:
    net.eval()
    n = t["z_m"].shape[0]
    out = []
    with torch_threads(), torch.no_grad():
        for s in range(0, n, chunk):
            out.append(net({k: v[s:s + chunk] for k, v in t.items()}).numpy())
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def warmup_learning_rate(step: int, steps_per_epoch: int) -> float:
    """Learning rate of optimizer step ``step`` (0-based): linear over the first 10 epochs, then 3e-3."""
    return LEARNING_RATE * min(1.0, (int(step) + 1) / (WARMUP_EPOCHS * int(steps_per_epoch)))


def train_network(train: EncodedRows, y_train: np.ndarray, dims: InputDims, config: NeuralConfig, *, model_seed: int,
                  n_epochs: int | None = None, valid: EncodedRows | None = None, y_valid: np.ndarray | None = None,
                  valid_units: np.ndarray | None = None, patience: int = PATIENCE, max_epochs: int = MAX_EPOCHS,
                  y_scale: tuple[float, float] | None = None) -> TrainResult:
    """Fit one network (module docstring, "Training").  ``y_train`` / ``y_valid`` are log D; the scaling is the
    training rows' mean / SD unless ``y_scale`` is given.  Exactly one of ``n_epochs`` and ``valid`` is used."""
    t0 = time.perf_counter()
    y_train = np.asarray(y_train, dtype=np.float64)
    if len(y_train) != len(train) or not len(train):
        raise ValueError("train_network: target length differs from the training rows (or no rows)")
    if not np.isfinite(y_train).all():
        raise ValueError("train_network: non-finite training target")
    early = valid is not None
    if early:
        if n_epochs is not None:
            raise ValueError("train_network: pass n_epochs or validation rows, not both")
        if y_valid is None or valid_units is None or len(y_valid) != len(valid) or len(valid_units) != len(valid):
            raise ValueError("train_network: validation rows need y_valid and valid_units")
        if not len(valid):
            raise ValueError("train_network: empty validation set")
        if len(train.index.intersection(valid.index)):
            raise AssertionError("train_network: a validation row is a training row")
        _, unit_codes = np.unique(np.asarray(valid_units).astype(str), return_inverse=True)
        n_units = int(unit_codes.max()) + 1
        limit = int(max_epochs)
    else:
        if n_epochs is None or int(n_epochs) < 1:
            raise ValueError("train_network: n_epochs >= 1 required without validation rows")
        limit = int(n_epochs)
    mu, sd = _sorted_stat(y_train) if y_scale is None else (float(y_scale[0]), float(y_scale[1]))
    y_std = torch.from_numpy(((y_train - mu) / sd).astype(np.float32))
    tt = train.tensors()
    tv = valid.tensors() if early else None
    n = len(train)
    steps_per_epoch = int(math.ceil(n / BATCH_SIZE))
    hist: list[tuple[int, float, float]] = []
    with deterministic_torch(model_seed):
        net = FactorisedNet(dims, config)
        opt = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=float(config.weight_decay))
        gen = torch.Generator().manual_seed(int(model_seed))
        best, best_epoch, best_state, best_pred = float("inf"), 0, None, None
        step = 0
        epoch = 0
        for epoch in range(1, limit + 1):
            net.train()
            perm = torch.randperm(n, generator=gen)
            loss_sum = 0.0
            for s in range(0, n, BATCH_SIZE):
                b = perm[s:s + BATCH_SIZE]
                for g in opt.param_groups:
                    g["lr"] = warmup_learning_rate(step, steps_per_epoch)
                opt.zero_grad(set_to_none=True)
                pred = net({k: v[b] for k, v in tt.items()})
                loss = nn.functional.huber_loss(pred, y_std[b], delta=HUBER_DELTA)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
                opt.step()
                loss_sum += float(loss.detach()) * len(b)
                step += 1
            vm = float("nan")
            if early:
                vp = _predict_std(net, tv) * sd + mu
                vm = macro_mae(y_valid, vp, unit_codes, n_units)
                if vm < best:
                    best, best_epoch, best_pred = vm, epoch, vp
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            hist.append((epoch, loss_sum / n, vm))
            if early and epoch - best_epoch >= int(patience):
                break
        if early:
            if best_state is None:
                raise RuntimeError("train_network: no finite validation score")
            net.load_state_dict(best_state)
        else:
            best_epoch = epoch
    net.eval()
    return TrainResult(net=net, best_epoch=int(best_epoch), epochs_run=int(epoch),
                       best_valid_macro_mae=float(best) if early else float("nan"),
                       history=pd.DataFrame(hist, columns=["epoch", "train_huber", "valid_macro_mae"]),
                       seconds=time.perf_counter() - t0, y_mean=mu, y_sd=sd, model_seed=int(model_seed),
                       n_parameters=net.n_parameters, valid_pred=best_pred)


# --------------------------------------------------------------------------------------------- #
# the arm
# --------------------------------------------------------------------------------------------- #

def _order_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows in ``canonical_measurement_id`` order (else index label order): the fit does not depend on frame order."""
    if not rows.index.is_unique:
        raise ValueError("rows: the index must be unique")
    key = rows[I.ID_COL].astype(str).to_numpy() if I.ID_COL in rows.columns else rows.index.astype(str).to_numpy()
    return rows.iloc[np.argsort(key, kind="stable")]


def _target(rows: pd.DataFrame) -> np.ndarray:
    if I.TARGET_COL not in rows.columns:
        raise KeyError(f"training rows need {I.TARGET_COL}")
    y = pd.to_numeric(rows[I.TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("training rows with a non-finite log_D")
    return y


def _cv_for(cv: pd.DataFrame | None, rows: pd.DataFrame) -> pd.DataFrame | None:
    if cv is None:
        return None
    if not rows.index.isin(cv.index).all():
        raise ValueError("condition vectors do not cover the rows")
    return cv.loc[rows.index]


class FactorisedArm:
    """M1 / M2 at a fixed configuration and epoch count (the outer refit of :class:`TuningResult`, or a conformal
    inner refit).  ``rows`` / ``condition_vectors``: an optional store covering every RowTable row (fast path)."""

    def __init__(self, config: NeuralConfig, *, n_epochs: int, model_seed: int, rows: pd.DataFrame | None = None,
                 condition_vectors: pd.DataFrame | None = None):
        if int(n_epochs) < 1 or int(n_epochs) > MAX_EPOCHS:
            raise ValueError(f"n_epochs must be in 1..{MAX_EPOCHS}")
        self.config, self.n_epochs, self.model_seed = config, int(n_epochs), int(model_seed)
        self.name = config.step
        self.rows, self.condition_vectors = rows, condition_vectors
        self.encoder: NeuralEncoder | None = None
        self.result: TrainResult | None = None
        self.train_index: pd.Index | None = None
        self._table: I.RowTable | None = None
        self._train_mask: np.ndarray | None = None
        self._metal_table: pd.DataFrame | None = None
        self._system_table: pd.DataFrame | None = None

    def clone(self) -> "FactorisedArm":
        return FactorisedArm(self.config, n_epochs=self.n_epochs, model_seed=self.model_seed, rows=self.rows,
                             condition_vectors=self.condition_vectors)

    # ----------------------------------------------------------------------------------------- #
    def fit(self, train_rows: pd.DataFrame, context: I.FitContext | None = None) -> "FactorisedArm":
        if context is not None and context.hidden_index is not None and len(context.hidden_index):
            n_bad = len(pd.Index(train_rows.index).intersection(pd.Index(context.hidden_index)))
            if n_bad:
                raise AssertionError(f"{n_bad} hidden row(s) among the training rows")
        self._table, self._train_mask = None, None
        rows = _order_rows(train_rows)
        y = _target(rows)
        cv = _cv_for(self.condition_vectors, rows)
        self.encoder = NeuralEncoder().fit(rows, cv)
        enc = self.encoder.transform(rows, cv)
        self.result = train_network(enc, y, self.encoder.dims, self.config, model_seed=self.model_seed,
                                    n_epochs=self.n_epochs)
        self.train_index = pd.Index(rows.index)
        self._embedding_tables(rows, enc)
        return self

    def fit_table(self, table: I.RowTable, mask: np.ndarray, context: I.FitContext) -> "FactorisedArm":
        """``ConformalWrapper`` fast path: the rows come from the ``rows`` store, the target from ``table.y``."""
        if self.rows is None:
            raise ValueError("FactorisedArm.fit_table needs the rows store (FactorisedArm(rows=...))")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (table.n,) or not mask.any():
            raise ValueError("training mask does not match the RowTable (or is empty)")
        forbidden = I.forbidden_mask(table, context)
        if (mask & forbidden).any():
            raise AssertionError(f"{int((mask & forbidden).sum())} hidden row(s) among the training rows")
        if not table.index.isin(self.rows.index).all():
            raise ValueError("the rows store does not cover the RowTable")
        labels = table.index[mask]
        rows = self.rows.loc[labels].drop(columns=[I.TARGET_COL], errors="ignore")
        rows = rows.assign(**{I.TARGET_COL: table.y[mask]})
        self.fit(rows, None)
        self._table, self._train_mask = table, mask.copy()
        return self

    # ----------------------------------------------------------------------------------------- #
    def _require_fit(self) -> tuple[NeuralEncoder, TrainResult]:
        if self.encoder is None or self.result is None:
            raise RuntimeError(f"{self.name}: fit first")
        return self.encoder, self.result

    def _encode(self, rows: pd.DataFrame) -> EncodedRows:
        enc, _ = self._require_fit()
        return enc.transform(rows, _cv_for(self.condition_vectors, rows))

    def predict(self, query_rows: pd.DataFrame) -> pd.DataFrame:
        enc, res = self._require_fit()
        if not query_rows.index.is_unique:
            raise ValueError("query rows: the index must be unique")
        if len(query_rows.index.intersection(self.train_index)):
            raise AssertionError("a query row is one of the training rows")
        e = self._encode(query_rows)
        mean = _predict_std(res.net, e.tensors()) * res.y_sd + res.y_mean
        seen = e.seen()
        n = len(query_rows)
        out = pd.DataFrame({c: pd.Series([np.nan] * n if c in I._FLOAT_COLUMNS else [None] * n,
                                         dtype=float if c in I._FLOAT_COLUMNS else object)
                            for c in I.PREDICTION_COLUMNS})
        out["row_id"] = pd.Series(query_rows.index.to_numpy(dtype=object), dtype=object)
        out["mean_logD"] = mean.astype(float)
        out["fallback_level"] = self.name
        reason = np.full(n, "", dtype=object)
        for c in seen.columns:
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
        for c in seen.columns:
            out[c] = seen[c].to_numpy(dtype=bool)
        return out[list(PREDICTION_COLUMNS)]

    def predict_positions(self, positions: np.ndarray) -> pd.DataFrame:
        if self._table is None or self.rows is None:
            raise RuntimeError("predict_positions needs a fit_table fit")
        p = np.asarray(positions, dtype=np.int64)
        if len(p) and self._train_mask[p].any():
            raise AssertionError("a query row is one of the training rows")
        return self.predict(self.rows.loc[self._table.index[p]])

    # ----------------------------------------------------------------------------------------- #
    def embeddings(self, rows: pd.DataFrame) -> pd.DataFrame:
        """``e_m_00..`` and ``e_l_00..`` (and the bilinear term for M2) of any rows under the fitted encoder."""
        _, res = self._require_fit()
        e = self._encode(rows)
        t = e.tensors()
        res.net.eval()
        with torch_threads(), torch.no_grad():
            em = res.net.metal_embedding(t).numpy().astype(np.float64)
            el = res.net.ligand_embedding(t).numpy().astype(np.float64)
            inter = res.net.interaction(res.net.metal_embedding(t), res.net.ligand_embedding(t)).numpy()
        d = self.config.emb_dim
        out = pd.DataFrame(np.hstack([em, el]), index=rows.index,
                           columns=[f"e_m_{j:02d}" for j in range(d)] + [f"e_l_{j:02d}" for j in range(d)])
        out["interaction_std"] = inter.astype(np.float64)
        return out

    def _embedding_tables(self, rows: pd.DataFrame, enc: EncodedRows) -> None:
        res = self.result
        t = enc.tensors()
        res.net.eval()
        with torch_threads(), torch.no_grad():
            em = res.net.metal_embedding(t).numpy().astype(np.float64)
            el = res.net.ligand_embedding(t).numpy().astype(np.float64)
        d = self.config.emb_dim
        mcols = [f"e_m_{j:02d}" for j in range(d)]
        lcols = [f"e_l_{j:02d}" for j in range(d)]
        st = rows[SG.METAL_COL].to_numpy(dtype=object)
        known = np.array([not SG._missing(v) for v in st])
        mt = pd.DataFrame(em[known], columns=mcols)
        mt[SG.METAL_COL] = st[known].astype(str)
        g = mt.groupby(SG.METAL_COL, sort=True)
        spread = (g[mcols].max() - g[mcols].min()).to_numpy()
        if spread.size and float(np.max(spread)) > 1e-5:
            raise AssertionError("e_m differs between training rows of one metal state")
        self._metal_table = g[mcols].mean().join(g.size().rename("n_training_rows"))
        lt = pd.DataFrame(el, columns=lcols)
        lt[SG.SYSTEM_COL] = rows[SG.SYSTEM_COL].astype(str).to_numpy()
        g = lt.groupby(SG.SYSTEM_COL, sort=True)
        self._system_table = g[lcols].mean().join(g.size().rename("n_training_rows"))

    def metal_embeddings(self) -> pd.DataFrame:
        """``e_m`` of every known metal state of the training rows (index ``g19_metal_state``)."""
        self._require_fit()
        return self._metal_table.copy()

    def system_embeddings(self) -> pd.DataFrame:
        """Mean ``e_l`` of every training system (index ``extractant_system_key``)."""
        self._require_fit()
        return self._system_table.copy()

    def bilinear_matrix(self) -> np.ndarray | None:
        return self._require_fit()[1].net.bilinear_matrix()

    @property
    def n_parameters(self) -> int:
        return self._require_fit()[1].n_parameters

    @property
    def model_state_digest(self) -> str:
        return state_digest(self._require_fit()[1].net)

    def fit_record(self) -> dict[str, Any]:
        enc, res = self._require_fit()
        return {"step": self.name, "config": asdict(self.config), "n_epochs": self.n_epochs,
                "model_seed": self.model_seed, "n_parameters": res.n_parameters, "n_train_rows": len(self.train_index),
                "y_mean": res.y_mean, "y_sd": res.y_sd, "feature_state_digest": enc.state_digest,
                "model_state_digest": state_digest(res.net), "dims": asdict(enc.dims), "seconds": res.seconds}


# --------------------------------------------------------------------------------------------- #
# inner validation splits and tuning
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ValidationSplit:
    """One inner split of the outer training rows (labels of the outer-training frame)."""

    name: str
    inner_fold: int
    train_index: pd.Index
    valid_index: pd.Index
    valid_units: np.ndarray
    hidden_index: pd.Index
    guard: str | None = None


def splits_from_folds(outer_train_rows: pd.DataFrame, folds: Sequence[FI.Fold], *, v6_mask: pd.Series,
                      exclude_from_scoring: pd.Series | None = None, inner_folds: Iterable[int] | None = None,
                      guard: str = "fold_builder_check") -> list[ValidationSplit]:
    """Validation splits from the fold builders' inner folds of ``outer_train_rows``.

    Training rows: the outer-training rows minus the fold's hidden rows.  Validation rows: the fold's scored rows minus
    ``exclude_from_scoring`` (e.g. acidic co-extractant rows); every one passes ``registered.assert_not_scored`` and is
    a known state other than Sr(III).  Unit: ``fold.row_unit`` (default ``fold.units[0]``).  Inner fold: ``meta
    ["inner_fold"]`` (V5 batches) or the fold's position (V1, V2).  ``inner_folds`` keeps only those inner folds (the
    section 7 compute plan: ``[0]`` on the four other discovery seeds).  ``guard`` records how the folds were checked
    (the builders run ``fold_isolation_check`` with ``check=True``)."""
    if I.ID_COL not in outer_train_rows.columns:
        raise KeyError(f"outer training rows need {I.ID_COL}")
    ids = outer_train_rows[I.ID_COL].astype(str)
    if not ids.is_unique:
        raise ValueError(f"{I.ID_COL} must be unique")
    label_of = pd.Series(outer_train_rows.index, index=ids.to_numpy())
    keep = None if inner_folds is None else {int(k) for k in inner_folds}
    excl = set() if exclude_from_scoring is None else \
        set(exclude_from_scoring.index[exclude_from_scoring.fillna(False).to_numpy(dtype=bool)])
    out = []
    for pos, f in enumerate(folds):
        k = int(f.meta["inner_fold"]) if "inner_fold" in f.meta else pos
        if keep is not None and k not in keep:
            continue
        missing = [r for r in f.hidden_row_ids if r not in label_of.index]
        if missing:
            raise AssertionError(f"{f.fold_id}: {len(missing)} hidden id(s) outside the outer training rows")
        hidden = pd.Index(label_of.loc[list(f.hidden_row_ids)].to_numpy())
        scored_ids = [r for r in f.scored_row_ids if label_of[r] not in excl]
        if not scored_ids:
            continue
        valid = pd.Index(label_of.loc[scored_ids].to_numpy())
        FR.assert_not_scored(valid, v6_mask, f"inner validation {f.fold_id}")
        st = outer_train_rows.loc[valid, SG.METAL_COL]
        if st.isna().any() or st.isin(FI.UNSCORED_STATES).any():
            raise AssertionError(f"{f.fold_id}: validation rows include an X(?) or Sr(III) row")
        default = f.units[0] if f.units else f.fold_id
        units = np.array([str(f.row_unit.get(r, default)) for r in scored_ids], dtype=object)
        train = outer_train_rows.index.difference(hidden, sort=False)
        out.append(ValidationSplit(name=f.fold_id, inner_fold=k, train_index=train, valid_index=valid,
                                   valid_units=units, hidden_index=hidden, guard=guard))
    return out


@dataclass
class TuningResult:
    step: str
    selected: NeuralConfig
    n_epochs: int
    model_seed: int
    scores: pd.DataFrame          # one row per configuration
    fits: pd.DataFrame            # one row per (configuration, split)
    inner_predictions: pd.DataFrame  # selected configuration: best-epoch predictions on every validation row
    readings: dict[str, str] = field(default_factory=lambda: dict(REGISTRATION_CHOICES))
    #: one row per (configuration, split, validation unit): ``n_rows``, ``sum_abs_error`` of the best-epoch predictions
    split_unit_errors: pd.DataFrame = field(default_factory=pd.DataFrame)

    def arm(self, rows: pd.DataFrame | None = None, condition_vectors: pd.DataFrame | None = None) -> FactorisedArm:
        """The outer-refit arm: the selected configuration for the median best-epoch count."""
        return FactorisedArm(self.selected, n_epochs=self.n_epochs, model_seed=self.model_seed, rows=rows,
                             condition_vectors=condition_vectors)


def median_epochs(best_epochs: Sequence[int]) -> int:
    """Median best-epoch count, rounded half up, at least 1."""
    if not len(best_epochs):
        raise ValueError("no inner fit")
    return max(1, int(math.floor(float(np.median(np.asarray(best_epochs, dtype=float))) + 0.5)))


def select_config(scores: pd.DataFrame, tolerance: float = SELECTION_TOLERANCE) -> int:
    """Row position of the selected configuration: within ``tolerance`` of the best score, the smallest
    ``(rank, n_parameters, -weight_decay)``."""
    s = scores["inner_macro_mae"].to_numpy(dtype=float)
    if not np.isfinite(s).any():
        raise ValueError("no finite configuration score")
    best = float(np.nanmin(s))
    cand = np.flatnonzero(np.isfinite(s) & (s <= best + float(tolerance) + 1e-12))
    key = [(int(scores["rank"].iloc[i]), int(scores["n_parameters"].iloc[i]), -float(scores["weight_decay"].iloc[i]), i)
           for i in cand]
    return min(key)[3]


def tune(outer_train_rows: pd.DataFrame, splits: Sequence[ValidationSplit], configs: Sequence[NeuralConfig], *,
         model_seed: int, condition_vectors: pd.DataFrame | None = None, patience: int = PATIENCE,
         max_epochs: int = MAX_EPOCHS, tolerance: float = SELECTION_TOLERANCE, allow_unguarded: bool = False,
         progress: Callable[[str], None] | None = None) -> TuningResult:
    """Section 6 / 7 search of ``configs`` over the inner ``splits`` of one outer training set (module docstring)."""
    if not splits:
        raise ValueError("tune: no inner validation split")
    if not configs:
        raise ValueError("tune: no configuration")
    steps = {c.step for c in configs}
    if len(steps) != 1:
        raise ValueError("tune: configurations of one ladder step only")
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
    outer_enc = NeuralEncoder().fit(_order_rows(outer), _cv_for(condition_vectors, outer))
    fit_recs: list[dict[str, Any]] = []
    per_cfg: dict[int, dict[str, list]] = {ci: {"pred": [], "y": [], "unit": [], "split": [], "label": []}
                                           for ci in range(len(configs))}
    for sp in splits:            # one split in memory at a time; its encoder and target scale: its training rows only
        tr = _order_rows(outer.loc[sp.train_index])
        cv_tr = _cv_for(condition_vectors, tr)
        enc = NeuralEncoder().fit(tr, cv_tr)
        etr, ytr = enc.transform(tr, cv_tr), _target(tr)
        va = outer.loc[sp.valid_index]
        eva, yva = enc.transform(va, _cv_for(condition_vectors, va)), _target(va)
        for ci, cfg in enumerate(configs):
            r = train_network(etr, ytr, enc.dims, cfg, model_seed=model_seed, valid=eva, y_valid=yva,
                              valid_units=sp.valid_units, patience=patience, max_epochs=max_epochs)
            fit_recs.append({"config": cfg.label(), "config_pos": ci, "split": sp.name, "inner_fold": sp.inner_fold,
                             "n_train_rows": len(etr), "n_valid_rows": len(eva),
                             "n_valid_units": len(set(np.asarray(sp.valid_units).astype(str))),
                             "best_epoch": r.best_epoch, "epochs_run": r.epochs_run,
                             "valid_macro_mae": r.best_valid_macro_mae, "n_parameters": r.n_parameters,
                             "seconds": r.seconds})
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
    score_recs, pooled, sue = [], {}, []
    fold_of_split = {sp.name: int(sp.inner_fold) for sp in splits}
    for ci, cfg in enumerate(configs):
        acc = per_cfg[ci]
        for pr_s, y_s, u_s, sn_s in zip(acc["pred"], acc["y"], acc["unit"], acc["split"]):
            t = pd.DataFrame({"unit": u_s, "e": np.abs(np.asarray(y_s, dtype=np.float64) - pr_s)}).groupby(
                "unit", sort=True)["e"]
            name = str(sn_s[0]) if len(sn_s) else ""
            for unit, n, tot in zip(t.size().index, t.size().to_numpy(), t.sum().to_numpy()):
                sue.append({"config": cfg.label(), "split": name, "inner_fold": fold_of_split.get(name, -1),
                            "unit": str(unit), "n_rows": int(n), "sum_abs_error": float(tot)})
    for ci, cfg in enumerate(configs):
        acc = per_cfg[ci]
        u = np.concatenate(acc["unit"])
        _, codes = np.unique(u, return_inverse=True)
        pr, yv = np.concatenate(acc["pred"]), np.concatenate(acc["y"])
        score = macro_mae(yv, pr, codes)
        pooled[ci] = (pr, yv, u, np.concatenate(acc["split"]), np.concatenate(acc["label"]))
        best_epochs = [rec["best_epoch"] for rec in fit_recs if rec["config_pos"] == ci]
        score_recs.append({"config": cfg.label(), "emb_dim": cfg.emb_dim, "weight_decay": cfg.weight_decay,
                           "rank": cfg.rank, "n_parameters": parameter_count(outer_enc.dims, cfg),
                           "inner_macro_mae": score, "n_units": int(codes.max()) + 1, "n_splits": len(splits),
                           "median_best_epoch": median_epochs(best_epochs)})
    scores = pd.DataFrame(score_recs)
    pos = select_config(scores, tolerance)
    scores["selected"] = np.arange(len(scores)) == pos
    pr, yv, uv, sn, lab = pooled[pos]
    inner = pd.DataFrame({"split": sn, "row_label": lab, "unit": uv, "log_D": yv, "pred": pr})
    return TuningResult(step=next(iter(steps)), selected=configs[pos], n_epochs=int(scores["median_best_epoch"].iloc[pos]),
                        model_seed=int(model_seed), scores=scores, fits=pd.DataFrame(fit_recs), inner_predictions=inner,
                        split_unit_errors=pd.DataFrame(sue, columns=["config", "split", "inner_fold", "unit", "n_rows",
                                                                     "sum_abs_error"]))


def select_excluding_folds(scores: Sequence[Mapping[str, Any]], fits: Sequence[Mapping[str, Any]],
                           split_unit_errors: Sequence[Mapping[str, Any]], exclude_folds: Iterable[int], *,
                           tolerance: float = SELECTION_TOLERANCE) -> tuple[NeuralConfig, int, dict[str, Any]]:
    """The section 6 / 7 selection re-run on the inner splits OUTSIDE ``exclude_folds`` of a recorded tuning (the
    ``scores`` / ``fits`` / ``split_unit_errors`` records of :func:`tune`): macro MAE over the validation units of those
    splits (a unit pooled over the splits holding it), :func:`select_config`'s tie rule and the median best epoch of the
    chosen configuration's fits on those splits.  The cross-fitted conformal calibration of the discovery runner uses
    it: inner fold ``j``'s residuals come from a configuration and epoch count chosen without fold ``j``'s rows."""
    drop = {int(f) for f in exclude_folds}
    sc = pd.DataFrame(list(scores))
    fi = pd.DataFrame(list(fits))
    ue = pd.DataFrame(list(split_unit_errors))
    if sc.empty or fi.empty or ue.empty:
        raise ValueError("select_excluding_folds: the tuning record lacks scores / fits / split_unit_errors")
    ue = ue[~ue["inner_fold"].astype(int).isin(drop)]
    fi = fi[~fi["inner_fold"].astype(int).isin(drop)]
    if ue.empty:
        raise ValueError(f"select_excluding_folds: no inner split outside folds {sorted(drop)}")
    agg = ue.groupby(["config", "unit"], sort=True)[["n_rows", "sum_abs_error"]].sum()
    macro = (agg["sum_abs_error"] / agg["n_rows"]).groupby(level="config").mean()
    sub = sc.drop(columns=[c for c in ("inner_macro_mae", "selected") if c in sc.columns]).copy()
    sub["inner_macro_mae"] = sub["config"].map(macro).astype(float)
    sub = sub[np.isfinite(sub["inner_macro_mae"].to_numpy(dtype=float))].reset_index(drop=True)
    pos = select_config(sub, tolerance)
    row = sub.iloc[pos]
    cfg = NeuralConfig(int(row["emb_dim"]), float(row["weight_decay"]), int(row["rank"]))
    epochs = median_epochs(fi.loc[fi["config"] == row["config"], "best_epoch"].astype(int).tolist())
    return cfg, epochs, {"excluded_inner_folds": sorted(drop), "config": str(row["config"]),
                         "inner_macro_mae": float(row["inner_macro_mae"]), "n_epochs": int(epochs),
                         "n_splits": int(ue["split"].nunique())}


def tune_m1(outer_train_rows: pd.DataFrame, splits: Sequence[ValidationSplit], *, fold_index: int,
            **kwargs: Any) -> TuningResult:
    """M1: the 9-configuration grid on one outer fold (model seed from ``fold_index``)."""
    return tune(outer_train_rows, splits, m1_grid(), model_seed=registered_model_seed(fold_index), **kwargs)


def tune_m2(outer_train_rows: pd.DataFrame, splits: Sequence[ValidationSplit], m1_config: NeuralConfig, *,
            fold_index: int, **kwargs: Any) -> TuningResult:
    """M2: ranks {2, 4, 8} with the M1 values retained for THIS outer fold (``m1_config``)."""
    return tune(outer_train_rows, splits, m2_grid(m1_config), model_seed=registered_model_seed(fold_index), **kwargs)

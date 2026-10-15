"""Model 2: the neural contrast model adapted from ``lanthanidestrain/automl/topo``.

What is taken from the reference (read-only, not imported): the ``TabularNet`` head (LayerNorm ->
Linear -> SiLU -> Dropout -> Linear -> SiLU -> Dropout -> Linear 1), which the reference built as
the *no-topology control* of its 3D arms, the QuantileTransformer -> StandardScaler front end, and
the within-composition-block **pairwise-contrast loss** with a 3x weight on adjacent metals.  It is
therefore the reference's control architecture with the reference's objective, not its best
system (an anchored CatBoost + 3D-encoder blend that needs geometry assets this repository does
not hold).  Training here is full-batch AdamW (the training set is ~5k rows), the stopping
criterion is the contrast loss on a masked chemotype-grouped validation split of the training
cells, and the final model is refitted on every training cell for the number of epochs the
validation run selected (``fit`` below), so the deployed arm never trains on fewer cells than G14.

What is changed, and why (IMPLEMENTATION_PLAN.md §4.3): rows are (cell, metal) of the gen13
cohort; the target is the cell's centred log D so the level term has weight zero; the block is the
gen13 cell; training cells carry chemotype-balanced weights; the inner criterion is adjacent MAE;
ensembles have 8 seeds; the hold-out is chemotype + publication mask.  No 3D encoder.

k-shot mechanisms on top of the zero-shot ensemble:
* ``FROZEN+BLUP``: the generic residual BLUP (adapters.blup) with this arm's own inner-OOF
  covariance -- handled by the runner, not here.
* ``HEAD_FT``: refit the last linear layer on the support contrasts with an L2 penalty toward the
  zero-shot weights (closed form on the frozen penultimate features).
* ``FULL_FT``: a few Adam steps on all parameters with the same penalty (documented over-fitting
  check at small k; deployed only if it wins on inner folds).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from . import bootstrap  # noqa: F401
from .data import N_LN, Z_VEC
from .design import Support
from gen13sep.amplitude_bench import LEAN_BLOCKS  # noqa: E402
from gen13sep.metals import physics_basis  # noqa: E402
from gen15.valuebench import Ctx  # noqa: E402

bootstrap.torch_threads()
METAL_DESCRIPTORS = ("radius", "radius_sq", "hydration", "tetrad_e1", "tetrad_e3")


@dataclass
class NNCConfig:
    hidden: int = 256
    dropout: float = 0.15
    weight_decay: float = 1e-4
    lr: float = 1e-3
    max_epochs: int = 300
    patience: int = 30
    min_epochs: int = 20
    adjacent_weight: float = 3.0
    huber_delta: float = 1.0
    n_seeds: int = 8
    name: str = "c256_d15_wd1e-4"


CONFIG_GRID: tuple[NNCConfig, ...] = (
    NNCConfig(hidden=256, dropout=0.15, weight_decay=1e-4, name="c256_d15_wd1e-4"),
    NNCConfig(hidden=128, dropout=0.30, weight_decay=1e-3, name="c128_d30_wd1e-3"),
    NNCConfig(hidden=256, dropout=0.15, weight_decay=1e-3, name="c256_d15_wd1e-3"),
)


def metal_matrix() -> np.ndarray:
    pb = physics_basis()
    return np.column_stack([pb[k] for k in METAL_DESCRIPTORS])


class Head(nn.Module):
    def __init__(self, d_in: int, hidden: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden // 2, 1)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.out(self.body(x)).squeeze(-1)


def _rows(X: np.ndarray, Y: np.ndarray, idx, M: np.ndarray):
    """(cell, metal) rows for the cells in ``idx`` whose metal is measured."""
    ci, mi = [], []
    for c in np.asarray(idx):
        for m in np.flatnonzero(~np.isnan(Y[c])):
            ci.append(int(c)); mi.append(int(m))
    ci, mi = np.array(ci, dtype=int), np.array(mi, dtype=int)
    Xr = np.hstack([X[ci], M[mi]])
    yc = Y[ci, mi] - np.array([np.nanmean(Y[c]) for c in ci])
    return ci, mi, Xr, yc


def _pairs(ci: np.ndarray, mi: np.ndarray, adjacent_weight: float):
    """All within-cell row pairs, weight ``adjacent_weight`` when |ΔZ| == 1."""
    order = {}
    for r, c in enumerate(ci):
        order.setdefault(int(c), []).append(r)
    pi, pj, w = [], [], []
    for rows in order.values():
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                i, j = rows[a], rows[b]
                pi.append(i); pj.append(j)
                w.append(adjacent_weight if abs(int(Z_VEC[mi[i]]) - int(Z_VEC[mi[j]])) == 1 else 1.0)
    return np.array(pi), np.array(pj), np.array(w, dtype=np.float32)


def _train_members(Xt, yt, pi_t, pj_t, pw_t, d_in, config, seed_base, n_seeds, epochs_per_seed=None,
                   val=None):
    """Train ``n_seeds`` heads.  With ``val`` (vXt, vpi, vpj, vw, vdy, vadj) each head early-stops on
    the validation contrast loss and reports its best epoch; with ``epochs_per_seed`` each head
    trains exactly that many epochs (the refit).  Returns (models, logs)."""
    models, logs = [], []
    for s in range(n_seeds):
        torch.manual_seed(int(seed_base) + 7919 * s)
        net = Head(d_in, config.hidden, config.dropout)
        opt = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        n_ep = int(epochs_per_seed[s]) if epochs_per_seed is not None else config.max_epochs
        best, best_state, wait, best_adj, best_ep = np.inf, None, 0, float("nan"), n_ep
        for ep in range(n_ep):
            net.train()
            pred = net(Xt)
            dp, dt = pred[pi_t] - pred[pj_t], yt[pi_t] - yt[pj_t]
            loss = (pw_t * nn.functional.huber_loss(dp, dt, delta=config.huber_delta, reduction="none")).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if val is None:
                continue
            vXt, vpi, vpj, vw, vdy, vadj = val
            net.eval()
            with torch.no_grad():
                vp = net(vXt).numpy()
            e = (vp[vpi] - vp[vpj]) - vdy
            hub = np.where(np.abs(e) <= config.huber_delta, 0.5 * e * e, config.huber_delta * (np.abs(e) - 0.5 * config.huber_delta))
            crit = float((vw * hub).mean())
            if crit < best - 1e-6:
                best, wait, best_ep = crit, 0, ep + 1
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                best_adj = float(np.abs(e[vadj]).mean()) if vadj.any() else float("nan")
            elif ep >= config.min_epochs:
                wait += 1
                if wait >= config.patience:
                    break
        if val is not None and best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        models.append(net)
        logs.append({"seed": s, "best_val_crit": best, "val_adjacent_mae_at_best": best_adj, "epochs": best_ep})
    return models, logs


def _val_split(bench, train, design, seed, k=4):
    """Chemotype-grouped validation cells carved from ``train`` (first of ``k`` folds), with the
    design's publication/copy-group mask applied to the remaining training cells."""
    from .cov import inner_mask
    from lanthanide_separation.gen6.cohorts import seeded_group_kfold
    groups = bench.groups[train].astype(str)
    tr_rel, va_rel = next(iter(seeded_group_kfold(groups, k, int(seed))))
    va = np.asarray(train)[va_rel]
    tr = inner_mask(bench, design, np.asarray(train)[tr_rel], va)
    return tr, va


@dataclass
class NNCArm:
    """Zero-shot neural contrast ensemble; ``fit(ctx)`` then ``curves(idx)``.

    ``fit``: (1) carve a chemotype-grouped validation split out of ``ctx.train`` (masked like the
    design), train each member with early stopping on the validation contrast loss to find its best
    epoch; (2) refit each member on **all** of ``ctx.train`` for exactly that many epochs.  The
    preprocessing is fitted on the rows it is used on (training cells only) in both stages."""
    config: NNCConfig = field(default_factory=NNCConfig)
    name: str = "NNC"
    val_seed_tag: int = 17

    def _prepare(self, X, Y, cells, M, w_of):
        ci, mi, Xr, yc = _rows(X, Y, cells, M)
        pre = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                            QuantileTransformer(n_quantiles=min(200, len(Xr)), output_distribution="normal", random_state=0),
                            StandardScaler()).fit(Xr)
        Xt = torch.tensor(pre.transform(Xr), dtype=torch.float32)
        yt = torch.tensor(yc, dtype=torch.float32)
        pi, pj, pw = _pairs(ci, mi, self.config.adjacent_weight)
        pw = pw * np.array([w_of[int(ci[i])] for i in pi], dtype=np.float32)
        pw = pw / pw.mean()
        return pre, Xt, yt, torch.tensor(pi), torch.tensor(pj), torch.tensor(pw), ci, mi

    def fit(self, ctx: Ctx):
        self.ctx = ctx
        bench = ctx.bench
        X = bench.matrix(LEAN_BLOCKS)
        M = metal_matrix()
        tr = np.asarray(ctx.train)
        w_of = {int(c): float(w) for c, w in zip(ctx.train, ctx.w)}
        # stage 1: validation run on a masked split of the training cells
        tr1, va = _val_split(bench, tr, ctx.design, int(ctx.seed) * 31 + int(ctx.fold) * 7919 + self.val_seed_tag)
        pre1, Xt1, yt1, pi1, pj1, pw1, _, _ = self._prepare(X, bench.Y, tr1, M, w_of)
        vci, vmi, vXr, vyc = _rows(X, bench.Y, va, M)
        vXt = torch.tensor(pre1.transform(vXr), dtype=torch.float32)
        vpi, vpj, vw = _pairs(vci, vmi, self.config.adjacent_weight)
        vadj = np.array([abs(int(Z_VEC[vmi[i]]) - int(Z_VEC[vmi[j]])) == 1 for i, j in zip(vpi, vpj)])
        vdy = vyc[vpi] - vyc[vpj]
        vw = vw / vw.mean()
        _, logs = _train_members(Xt1, yt1, pi1, pj1, pw1, Xt1.shape[1], self.config, ctx.model_seed, self.config.n_seeds,
                                 val=(vXt, vpi, vpj, vw, vdy, vadj))
        epochs = [max(int(l["epochs"]), 1) for l in logs]
        # stage 2: refit on every training cell for the selected number of epochs
        self.pre, Xt, yt, pi_t, pj_t, pw_t, _, _ = self._prepare(X, bench.Y, tr, M, w_of)
        self.models, logs2 = _train_members(Xt, yt, pi_t, pj_t, pw_t, Xt.shape[1], self.config, ctx.model_seed,
                                            self.config.n_seeds, epochs_per_seed=epochs)
        self.fit_log = [{**l1, "refit_epochs": l2["epochs"], "n_train_cells": int(len(tr)), "n_val_cells": int(len(va)),
                         "n_stage1_cells": int(len(tr1))} for l1, l2 in zip(logs, logs2)]
        self.fit_cells = tuple(int(c) for c in tr)
        self.X, self.M = X, M
        return self

    def _inputs(self, idx) -> torch.Tensor:
        idx = np.asarray(idx)
        Xr = np.hstack([np.repeat(self.X[idx], N_LN, axis=0), np.tile(self.M, (len(idx), 1))])
        return torch.tensor(self.pre.transform(Xr), dtype=torch.float32)

    def member_curves(self, idx) -> np.ndarray:
        """(n_members, len(idx), 14) centred curves."""
        idx = np.asarray(idx)
        Xt = self._inputs(idx)
        out = []
        with torch.no_grad():
            for net in self.models:
                p = net(Xt).numpy().reshape(len(idx), N_LN)
                out.append(p - p.mean(axis=1, keepdims=True))
        return np.stack(out)

    def curves(self, idx) -> np.ndarray:
        return self.member_curves(idx).mean(axis=0)

    # ---- k-shot mechanisms -------------------------------------------------------------
    def _penultimate(self, ci: int) -> np.ndarray:
        """(n_members, 14, h) penultimate features of one cell's 14 metal rows."""
        Xt = self._inputs([ci])
        with torch.no_grad():
            return np.stack([net.features(Xt).numpy() for net in self.models])

    def head_ft_curves(self, ci: int, support: Support, lam: float) -> np.ndarray:
        """Per member: ridge refit of the output layer on the support contrasts, penalised toward the
        zero-shot weights; returns (n_members, 14) centred curves."""
        H = self._penultimate(ci)
        out = []
        for m, net in enumerate(self.models):
            w0 = net.out.weight.detach().numpy().ravel()
            h = H[m]
            if support.k == 0:
                c = h @ w0
            else:
                A = np.array([h[a] - h[b] for a, b in support.pairs])
                y = np.asarray(support.values, dtype=float)
                G = A.T @ A + lam * np.eye(len(w0))
                w = np.linalg.solve(G, A.T @ y + lam * w0)
                c = h @ w
            out.append(c - c.mean())
        return np.stack(out)

    def full_ft_curves(self, ci: int, support: Support, lam: float, steps: int = 50, lr: float = 1e-3,
                       members: int | None = None) -> np.ndarray:
        """Per member (the first ``members``): ``steps`` Adam steps on all parameters, loss = support
        squared error + lam * ||theta - theta0||^2; returns (n_members, 14) centred curves."""
        import copy
        Xt = self._inputs([ci])
        out = []
        for m, net0 in enumerate(self.models if members is None else self.models[:members]):
            net = copy.deepcopy(net0)
            theta0 = [p.detach().clone() for p in net.parameters()]
            if support.k > 0:
                torch.manual_seed(int(self.ctx.model_seed) + 104729 * (m + 1) + int(ci))
                opt = torch.optim.Adam(net.parameters(), lr=lr)
                ia = torch.tensor([a for a, _ in support.pairs]); ib = torch.tensor([b for _, b in support.pairs])
                y = torch.tensor(support.values, dtype=torch.float32)
                net.eval()                      # dropout off: the fine-tuned curve is a function of its inputs
                for _ in range(steps):
                    p = net(Xt)
                    loss = ((p[ia] - p[ib] - y) ** 2).mean() + lam * sum(((q - q0) ** 2).sum() for q, q0 in zip(net.parameters(), theta0))
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                c = net(Xt).numpy()
            out.append(c - c.mean())
        return np.stack(out)

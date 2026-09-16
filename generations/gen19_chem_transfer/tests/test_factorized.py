"""B6 / B6r0, the metal x extractant factorisation with side information (``gen19ct/models/factorized.py``).

Synthetic data unless stated.  Real MODEL rows are used as INPUT rows only (feature equality with
``FeatureSet.for_arm("B6")``) or with their ``log_D`` OVERWRITTEN by a synthetic target before any table is built; no
registered fold is used, nothing is scored, nothing touches V6, and no test writes a file.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from gen19ct.chemistry import support_graph as SG
from gen19ct.data import load as LOAD
from gen19ct.data import normalize as N
from gen19ct.folds import io as FI
from gen19ct.models import factorized as FZ
from gen19ct.models import features as F
from gen19ct.models import interface as I

SEED = 104729


# --------------------------------------------------------------------------------------------- #
# synthetic builders
# --------------------------------------------------------------------------------------------- #

def planted(seed: int = 0, n_m: int = 16, n_s: int = 30, p_m: int = 3, p_s: int = 4, rank: int = 2,
            rows_per_cell: int = 4, density: float = 0.6, noise: float = 0.1, delta: float = 0.3,
            mixture_systems: int = 0):
    """``y = 1 + a_m + b_s + u_m'v_s + beta'x + anion + noise`` with ``u = A z_m + delta_m``, ``v = B z_s + delta_s``,
    ``a`` / ``b`` partly explained by ``z``; ``mixture_systems`` systems get a second z_s pattern on half their rows."""
    rng = np.random.default_rng(seed)
    Zm, Zs = rng.standard_normal((n_m, p_m)), rng.standard_normal((n_s, p_s))
    A, B = rng.standard_normal((rank, p_m)), rng.standard_normal((rank, p_s))
    U = Zm @ A.T + delta * rng.standard_normal((n_m, rank))
    V = Zs @ B.T + delta * rng.standard_normal((n_s, rank))
    am = Zm @ rng.standard_normal(p_m) * 0.5 + 0.3 * rng.standard_normal(n_m)
    bs = Zs @ rng.standard_normal(p_s) * 0.5 + 0.3 * rng.standard_normal(n_s)
    obs = rng.random((n_m, n_s)) < density
    obs[np.arange(n_m), np.arange(n_m) % n_s] = True
    cells = [(m, s) for m in range(n_m) for s in range(n_s) if obs[m, s]]
    m_idx = np.repeat([c[0] for c in cells], rows_per_cell)
    s_idx = np.repeat([c[1] for c in cells], rows_per_cell)
    n = len(m_idx)
    x = rng.standard_normal((n, 2))
    anion = rng.choice(["nitrate", "chloride", "perchlorate"], n)
    signal = (1.0 + am[m_idx] + bs[s_idx] + np.einsum("ra,ra->r", U[m_idx], V[s_idx]) + x @ [0.4, -0.2]
              + 0.3 * (anion == "chloride"))
    y = signal + noise * rng.standard_normal(n)
    zs_rows = Zs[s_idx].copy()
    for s in range(min(mixture_systems, n_s)):
        rows = np.flatnonzero(s_idx == s)[::2]
        zs_rows[rows, 0] += 0.5
    idx = [f"r{i:05d}" for i in range(n)]
    inputs = FZ.B6Inputs.from_arrays(
        idx, [f"M{m:02d}" for m in m_idx], [f"S{s:03d}" for s in s_idx],
        metal_numeric=pd.DataFrame(Zm[m_idx], columns=[f"zm{j}" for j in range(p_m)]),
        extractant_numeric=pd.DataFrame(zs_rows, columns=[f"zs{j}" for j in range(p_s)]),
        condition_numeric=pd.DataFrame(x, columns=["x0", "x1"]),
        condition_categorical=pd.DataFrame({"acid_anion": anion}))
    truth = {"U": U, "V": V, "m": m_idx, "s": s_idx, "cells": cells, "signal": signal, "Zm": Zm, "Zs": Zs}
    return inputs, y, truth


def table_for(inputs: FZ.B6Inputs, y: np.ndarray, n_groups: int = 6) -> I.RowTable:
    """A registered RowTable over the synthetic rows (arbitrary unit labels; one publication group per system mod n)."""
    sys_codes = np.array([int(s[1:]) for s in inputs.system])
    frame = pd.DataFrame({
        SG.METAL_COL: inputs.metal_unit, SG.ELEMENT_COL: inputs.metal_unit, SG.SYSTEM_COL: inputs.system,
        SG.ACID_ANION_COL: inputs.cond_cat[:, 0], SG.LOG_ACID_COL: inputs.cond_num[:, 0],
        SG.LOG_EXT_COL: inputs.cond_num[:, 1], I.ID_COL: [f"ID:{i:06d}" for i in range(inputs.n)], I.TARGET_COL: y,
        I.PUB_GROUP_COL: [f"g{c % n_groups}" for c in sys_codes]}, index=inputs.index)
    table = I.RowTable(frame)
    FZ.register_table(table, inputs)
    return table


def ctx(table: I.RowTable, mask: np.ndarray | None = None, seed: int = SEED) -> I.FitContext:
    hidden = table.index[~mask] if mask is not None else None
    return I.FitContext(seed=seed, table=table, hidden_index=hidden, v6_mask=pd.Series(False, index=table.index),
                        isolation_check=lambda tr, te: {"ok": True})


def cell_mask(table: I.RowTable, cells) -> np.ndarray:
    hid = np.zeros(table.n, dtype=bool)
    for m, s in cells:
        hid |= (table.state == table.state_code[m]) & (table.sys == table.sys_code[s])
    return hid


class CellSplitter:
    """Leave-one-cell-out over fixed cells: a stand-in inner design for synthetic tables."""

    name = "test_cells"

    def __init__(self, cells):
        self.cells = list(cells)

    def splits(self, table, mask, context):
        out = []
        for f, (m, s) in enumerate(self.cells):
            hid = mask & cell_mask(table, [(m, s)])
            if hid.any():
                out.append(I.InnerSplit(unit=(m, s), fold=f % 3, train_mask=mask & ~hid,
                                        cal_positions=np.flatnonzero(hid), hidden_positions=np.flatnonzero(hid)))
        return out


def labels(truth, cells):
    return [(f"M{m:02d}", f"S{s:03d}") for m, s in cells]


def explicit_design(enc: FZ.Encoded, d: FZ.Design, inputs: FZ.B6Inputs, pos: np.ndarray) -> np.ndarray:
    """The additive model's row-level design [x, z_m, onehot(metal), z_s, onehot(system)] (no intercept column)."""
    mi, si = FZ._lookup(d.metal_labels, inputs.metal_unit[pos]), FZ._lookup(d.system_labels, inputs.system[pos])
    om, os_ = np.zeros((len(pos), d.Jm)), np.zeros((len(pos), d.Js))
    om[np.flatnonzero(mi >= 0), mi[mi >= 0]] = 1.0
    os_[np.flatnonzero(si >= 0), si[si >= 0]] = 1.0
    return np.hstack([enc.X, enc.Zm, om, enc.Zs, os_])


# --------------------------------------------------------------------------------------------- #
# exact algebra
# --------------------------------------------------------------------------------------------- #

def test_rank0_equals_an_independent_additive_ridge() -> None:
    from sklearn.linear_model import Ridge

    inputs, y, truth = planted(seed=3, mixture_systems=3)
    table = table_for(inputs, y)
    hide = cell_mask(table, labels(truth, truth["cells"][:6])) | (table.state == table.state_code["M05"])
    mask = ~hide
    pos, q = np.flatnonzero(mask), np.flatnonzero(hide)
    for lam in FZ.LAMBDAS:
        arm = FZ.B6Factorized(0, lam).fit_table(table, mask, ctx(table, mask))
        d, enc = arm.design, arm.encoder
        Xtr = explicit_design(enc.transform(inputs, pos), d, inputs, pos)
        ref = Ridge(alpha=lam, fit_intercept=True, solver="cholesky").fit(Xtr, y[pos])
        Xq = explicit_design(enc.transform(inputs, q), d, inputs, q)          # M05 unseen: its one-hot column is absent
        got = arm.predict_positions(q)
        assert np.allclose(got["mean_logD"].to_numpy(), ref.predict(Xq), atol=1e-8)
        mp = arm.member_params()
        coef = np.concatenate([mp["gamma"][1:], mp["P_m"][:, 0], mp["F_m"][:, 0], mp["P_s"][:, 0], mp["F_s"][:, 0]])
        assert np.allclose(coef, ref.coef_, atol=1e-8)
        assert np.isclose(mp["gamma"][0], ref.intercept_, atol=1e-8)
        m05 = got["b6_metal_unit"] == "M05"
        assert m05.any() and (got.loc[m05, "fallback_reason"] == "metal_unseen").all()
        assert set(got["fallback_level"]) == {"B6r0"} and got["std_logD"].isna().all()


def test_each_als_half_step_equals_the_explicit_row_level_ridge_solve() -> None:
    inputs, y, truth = planted(seed=5, mixture_systems=4)
    pos = np.arange(inputs.n)
    enc = FZ.B6Encoder().fit(inputs, pos)
    E = enc.transform(inputs, pos)
    d = FZ.Design(inputs, [(pos, enc)], y)
    assert d.Pi > d.Js                                         # multi-pattern (mixture) systems are exercised
    rank, lam = 2, 0.7
    b = rank + 1
    rng = np.random.default_rng(0)
    P = FZ.B6Params(rank, np.array([lam]), rng.standard_normal((1, 1, d.Pc)), rng.standard_normal((1, 1, d.p_m, b)),
                    rng.standard_normal((1, 1, d.Jm, b)), rng.standard_normal((1, 1, d.p_s, b)),
                    rng.standard_normal((1, 1, d.Js, b)))
    mi, si = FZ._lookup(d.metal_labels, inputs.metal_unit), FZ._lookup(d.system_labels, inputs.system)
    C = np.hstack([np.ones((inputs.n, 1)), E.X])
    n = inputs.n

    def ridge(Xd, t):
        pen = np.r_[0.0, np.full(Xd.shape[1] - 1, lam)]
        return np.linalg.solve(Xd.T @ Xd + np.diag(pen), Xd.T @ t)

    um = E.Zm @ P.P_m[0, 0] + P.F_m[0, 0][mi]                   # system step given the metal side
    e = np.hstack([np.ones((n, 1)), um[:, 1:]])
    free = np.zeros((n, d.Js * b))
    for a in range(b):
        free[np.arange(n), si * b + a] = e[:, a]
    theta = ridge(np.hstack([C, (E.Zs[:, :, None] * e[:, None, :]).reshape(n, -1), free]), y - um[:, 0])
    g, Ps, Fs = FZ.system_step(d, P)
    assert np.abs(np.r_[g[0, 0], Ps[0, 0].ravel(), Fs[0, 0].ravel()] - theta).max() < 1e-9
    vs = E.Zs @ P.P_s[0, 0] + P.F_s[0, 0][si]                   # metal step given the system side
    e = np.hstack([np.ones((n, 1)), vs[:, 1:]])
    free = np.zeros((n, d.Jm * b))
    for a in range(b):
        free[np.arange(n), mi * b + a] = e[:, a]
    theta = ridge(np.hstack([C, (E.Zm[:, :, None] * e[:, None, :]).reshape(n, -1), free]), y - vs[:, 0])
    g, Pm, Fm = FZ.metal_step(d, P)
    assert np.abs(np.r_[g[0, 0], Pm[0, 0].ravel(), Fm[0, 0].ravel()] - theta).max() < 1e-9
    # the cell-statistic objective equals the row-level objective
    pred = FZ.predict_member(P, d, 0, 0, E, inputs.metal_unit, inputs.system)["mean"]
    obj, sse = FZ.objective(d, P)
    assert np.isclose(sse[0, 0], np.sum((y - pred) ** 2), rtol=1e-10)
    assert np.isclose(obj[0, 0] - sse[0, 0], lam * (np.sum(P.gamma[0, 0, 1:] ** 2) + sum(
        np.sum(getattr(P, f)[0, 0] ** 2) for f in ("P_m", "F_m", "P_s", "F_s"))), rtol=1e-10)


def test_batched_members_and_lambdas_equal_single_fits() -> None:
    inputs, y, truth = planted(seed=8, mixture_systems=2)
    table = table_for(inputs, y)
    rng = np.random.default_rng(4)
    members = []
    for i in range(3):
        cells = [truth["cells"][j] for j in rng.choice(len(truth["cells"]), 4, replace=False)]
        pos = np.flatnonzero(~cell_mask(table, labels(truth, cells)))
        members.append((pos, FZ.B6Encoder().fit(inputs, pos)))
    universe = np.arange(inputs.n)
    d = FZ.Design(inputs, members, y, universe=universe)
    lams = [0.1, 10.0]
    base = FZ.fit_additive(d, lams)
    P, info = FZ.fit_als(d, 2, base, SEED)
    for s, (pos, enc) in enumerate(members):
        q = np.setdiff1d(universe, pos)
        eq = enc.transform(inputs, q)
        for li, lam in enumerate(lams):
            d1 = FZ.Design(inputs, [(pos, enc)], y)
            b1 = FZ.fit_additive(d1, [lam])
            P1, i1 = FZ.fit_als(d1, 2, b1, SEED)
            for batched, single in ((P, P1), (base, b1)):
                m_b = FZ.predict_member(batched, d, s, li, eq, inputs.metal_unit[q], inputs.system[q])["mean"]
                m_1 = FZ.predict_member(single, d1, 0, 0, eq, inputs.metal_unit[q], inputs.system[q])["mean"]
                assert np.allclose(m_b, m_1, atol=1e-9)
            assert int(info.n_sweeps[s, li]) == int(i1.n_sweeps[0, 0])


# --------------------------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------------------------- #

def test_recovers_a_planted_rank2_structure_with_side_information() -> None:
    inputs, y, truth = planted(seed=0)
    table = table_for(inputs, y)
    rng = np.random.default_rng(1)
    held = [truth["cells"][j] for j in rng.choice(len(truth["cells"]), 25, replace=False)]
    hide = cell_mask(table, labels(truth, held))
    mask, q = ~hide, np.flatnonzero(hide)
    sig = truth["signal"][q]
    true_inter = np.einsum("ra,ra->r", truth["U"][truth["m"][q]], truth["V"][truth["s"][q]])
    mae = {}
    for k in (0, 1, 2):
        p = FZ.B6Factorized(k, 1.0).fit_table(table, mask, ctx(table, mask)).predict_positions(q)
        mae[k] = float(np.mean(np.abs(p["mean_logD"].to_numpy() - sig)))
        if k == 2:
            assert np.corrcoef(p["b6_interaction"].to_numpy(), true_inter)[0, 1] > 0.98
            assert p["b6_metal_seen"].all() and p["b6_system_seen"].all()
    assert mae[2] < 0.1 * mae[0] and mae[2] < 0.5 * mae[1], mae
    # side information: a metal with no training row is predicted through A z_m (its u_m has a small free part)
    unseen = table.state == table.state_code["M03"]
    q2 = np.flatnonzero(unseen)
    m0 = FZ.B6Factorized(0, 0.1).fit_table(table, ~unseen, ctx(table, ~unseen)).predict_positions(q2)
    m2 = FZ.B6Factorized(2, 0.1).fit_table(table, ~unseen, ctx(table, ~unseen)).predict_positions(q2)
    e0 = np.mean(np.abs(m0["mean_logD"].to_numpy() - truth["signal"][q2]))
    e2 = np.mean(np.abs(m2["mean_logD"].to_numpy() - truth["signal"][q2]))
    assert e2 < 0.25 * e0 and not m2["b6_metal_seen"].any()
    # section 7 tuning on leave-one-cell-out inner splits selects a rank >= 2 for B6 and rank 0 for B6r0
    inner = [c for c in labels(truth, truth["cells"]) if c not in set(labels(truth, held))][::9][:10]
    fits = FZ.fit_b6_and_b6r0(table, mask, ctx(table, mask), CellSplitter(inner))
    assert fits["B6"].selected[0] >= 2 and fits["B6r0"].selected[0] == 0
    assert fits["B6"].tuning is fits["B6r0"].tuning


def test_unseen_metal_and_system_use_side_information_only() -> None:
    inputs, y, truth = planted(seed=11)
    table = table_for(inputs, y)
    hide = (table.state == table.state_code["M02"]) | (table.sys == table.sys_code["S004"])
    mask = ~hide
    arm = FZ.B6Factorized(2, 1.0).fit_table(table, mask, ctx(table, mask))
    q = np.flatnonzero(hide)
    pred = arm.predict_positions(q)
    mp, enc = arm.member_params(), arm.encoder.transform(inputs, q)
    um = enc.Zm @ mp["P_m"]
    vs = enc.Zs @ mp["P_s"]
    seen_m = inputs.metal_unit[q] != "M02"
    seen_s = inputs.system[q] != "S004"
    um[seen_m] += mp["F_m"][FZ._lookup(mp["metal_labels"], inputs.metal_unit[q][seen_m])]
    vs[seen_s] += mp["F_s"][FZ._lookup(mp["system_labels"], inputs.system[q][seen_s])]
    manual = mp["gamma"][0] + enc.X @ mp["gamma"][1:] + um[:, 0] + vs[:, 0] + np.sum(um[:, 1:] * vs[:, 1:], axis=1)
    assert np.allclose(pred["mean_logD"].to_numpy(), manual, atol=1e-10)
    assert (pred["b6_metal_seen"].to_numpy() == seen_m).all() and (pred["b6_system_seen"].to_numpy() == seen_s).all()
    both = ~seen_m & ~seen_s
    exp_reason = np.where(both, "metal_unseen;system_unseen", np.where(~seen_m, "metal_unseen",
                                                                       np.where(~seen_s, "system_unseen", "")))
    assert (pred["fallback_reason"].to_numpy() == exp_reason).all()
    # perturbing every learned free part leaves the fully unseen rows unchanged
    only_side = np.flatnonzero(both)
    assert len(only_side)
    before = pred["mean_logD"].to_numpy()[only_side]
    arm.params.F_m[:] += 5.0
    arm.params.F_s[:] -= 3.0
    after = arm.predict_positions(q[only_side])["mean_logD"].to_numpy()
    assert np.allclose(before, after, atol=1e-12)
    # an unseen label with the side features of a seen unit differs from it by exactly the free parts
    arm.params.F_m[:] -= 5.0
    arm.params.F_s[:] += 3.0
    fake = FZ.B6Inputs.from_arrays(
        ["qa", "qb"], ["M07", "M_new"], ["S010", "S010"],
        metal_numeric=pd.DataFrame(np.repeat(truth["Zm"][[7]], 2, axis=0), columns=list(inputs.metal_num_cols)),
        extractant_numeric=pd.DataFrame(np.repeat(truth["Zs"][[10]], 2, axis=0), columns=list(inputs.ext_num_cols)),
        condition_numeric=pd.DataFrame(np.zeros((2, 2)), columns=["x0", "x1"]),
        condition_categorical=pd.DataFrame({"acid_anion": ["nitrate", "nitrate"]}))
    pf = arm.predict_inputs(fake)
    j = FZ._lookup(mp["metal_labels"], np.array(["M07"]))[0]
    vfull = (arm.encoder.transform(fake).Zs @ mp["P_s"] + mp["F_s"][FZ._lookup(mp["system_labels"],
                                                                                np.array(["S010"]))[0]])[0]
    diff = mp["F_m"][j, 0] + mp["F_m"][j, 1:] @ vfull[1:]
    assert np.isclose(pf["mean_logD"].iloc[0] - pf["mean_logD"].iloc[1], diff, atol=1e-10)
    assert pf["fallback_reason"].tolist() == ["", "metal_unseen"]


def test_convergence_trace_and_tolerance() -> None:
    inputs, y, truth = planted(seed=2)
    pos = np.arange(inputs.n)
    enc = FZ.B6Encoder().fit(inputs, pos)
    d = FZ.Design(inputs, [(pos, enc)], y)
    base = FZ.fit_additive(d, [1.0])
    _, capped = FZ.fit_als(d, 2, base, SEED, max_sweeps=7, tol=0.0)
    assert int(capped.n_sweeps[0, 0]) == 7 and not capped.converged[0, 0] and capped.objective_trace.shape[0] == 8
    _, loose = FZ.fit_als(d, 2, base, SEED, max_sweeps=FZ.MAX_SWEEPS, tol=1e-2, init_scale=0.5)
    n = int(loose.n_sweeps[0, 0])
    tr = loose.objective_trace[:n + 1, 0, 0]
    assert loose.converged[0, 0] and n < FZ.MAX_SWEEPS
    assert abs(tr[-2] - tr[-1]) <= 1e-2 * max(abs(tr[-2]), abs(tr[-1]))
    assert all(abs(tr[i - 1] - tr[i]) > 1e-2 * max(abs(tr[i - 1]), abs(tr[i])) for i in range(1, n))
    assert np.all(loose.objective_trace[n:, 0, 0] == tr[-1])                  # a finished member stays frozen
    P, tight = FZ.fit_als(d, 2, base, SEED, max_sweeps=4000, tol=1e-9, init_scale=0.5)
    tr = tight.objective_trace[:int(tight.n_sweeps[0, 0]) + 1, 0, 0]
    assert tight.converged[0, 0]
    assert np.all(np.diff(tr) <= 1e-9 * np.abs(tr[1:]))           # each exact half-step can only lower the objective
    assert tr[-1] <= FZ.objective(d, base)[0][0, 0]                 # ALS starts at and improves on the k=0 solution
    Q = P.copy()                                                    # a stationary point: one more sweep moves ~nothing
    Q.gamma, Q.P_s, Q.F_s = FZ.system_step(d, Q)
    Q.gamma, Q.P_m, Q.F_m = FZ.metal_step(d, Q)
    pos_all = FZ.predict_member(P, d, 0, 0, enc.transform(inputs, pos), inputs.metal_unit, inputs.system)["mean"]
    pos_new = FZ.predict_member(Q, d, 0, 0, enc.transform(inputs, pos), inputs.metal_unit, inputs.system)["mean"]
    assert np.abs(pos_all - pos_new).max() < 1e-3
    table = table_for(inputs, y)
    arm = FZ.B6Factorized(1, 1.0, max_sweeps=4, tol=0.0).fit_table(table, np.ones(inputs.n, bool), ctx(table))
    assert arm.info["n_sweeps"] == 4 and arm.info["converged"] is False


def test_determinism_seed_and_row_order() -> None:
    inputs, y, truth = planted(seed=6, mixture_systems=2)
    table = table_for(inputs, y)
    hide = cell_mask(table, labels(truth, truth["cells"][:5]))
    mask, q = ~hide, np.flatnonzero(hide)
    a = FZ.B6Factorized(3, 0.1).fit_table(table, mask, ctx(table, mask))
    b = FZ.B6Factorized(3, 0.1).fit_table(table, mask, ctx(table, mask))
    for f in FZ.B6Params.FIELDS:
        assert np.array_equal(getattr(a.params, f), getattr(b.params, f))
    pa = a.predict_positions(q)
    pd.testing.assert_frame_equal(pa, b.predict_positions(q))
    # a different seed changes only the factor initialisation (rank 0 does not depend on it)
    c = FZ.B6Factorized(3, 0.1, seed=7).fit_table(table, mask, ctx(table, mask))
    assert c.info["objective_trace"][0] != a.info["objective_trace"][0]
    r0a = FZ.B6Factorized(0, 0.1, seed=1).fit_table(table, mask, ctx(table, mask)).predict_positions(q)
    r0b = FZ.B6Factorized(0, 0.1, seed=2).fit_table(table, mask, ctx(table, mask)).predict_positions(q)
    assert np.array_equal(r0a["mean_logD"].to_numpy(), r0b["mean_logD"].to_numpy())
    # row order of the table does not matter
    perm = np.random.default_rng(0).permutation(inputs.n)
    inputs_p = FZ.B6Inputs.from_arrays(
        inputs.index[perm], inputs.metal_unit[perm], inputs.system[perm],
        metal_numeric=pd.DataFrame(inputs.metal_num[perm], columns=list(inputs.metal_num_cols)),
        extractant_numeric=pd.DataFrame(inputs.ext_num[perm], columns=list(inputs.ext_num_cols)),
        condition_numeric=pd.DataFrame(inputs.cond_num[perm], columns=list(inputs.cond_num_cols)),
        condition_categorical=pd.DataFrame(inputs.cond_cat[perm], columns=list(inputs.cond_cat_cols)))
    table_p = table_for(inputs_p, y[perm])
    mask_p = ~table_p.mask_of(table.index[q])
    ap = FZ.B6Factorized(3, 0.1).fit_table(table_p, mask_p, ctx(table_p, mask_p))
    pp = ap.predict_positions(table_p.positions(table.index[q]))
    assert np.allclose(pp["mean_logD"].to_numpy(), pa["mean_logD"].to_numpy(), atol=1e-8)
    # no state carries over between fits (cold start): a tuning run in between changes nothing
    FZ.run_inner_tuning(table, np.ones(table.n, bool), ctx(table), CellSplitter(labels(truth, truth["cells"][5:8])))
    d = FZ.B6Factorized(3, 0.1).fit_table(table, mask, ctx(table, mask))
    assert np.array_equal(d.params.F_m, a.params.F_m) and np.array_equal(d.params.gamma, a.params.gamma)


def test_hidden_rows_query_rows_and_seed_guards() -> None:
    inputs, y, truth = planted(seed=9)
    table = table_for(inputs, y)
    hide = cell_mask(table, labels(truth, truth["cells"][:2]))
    c = ctx(table, ~hide)
    with pytest.raises(AssertionError):
        FZ.B6Factorized(1, 1.0).fit_table(table, np.ones(table.n, bool), c)          # hidden rows in training
    arm = FZ.B6Factorized(1, 1.0).fit_table(table, ~hide, c)
    with pytest.raises(AssertionError):
        arm.predict_positions(np.flatnonzero(~hide)[:3])                            # a training row as query
    with pytest.raises(AssertionError):
        arm.predict_inputs(inputs, np.flatnonzero(~hide)[:3])
    with pytest.raises(ValueError):
        FZ.B6Factorized(2, 1.0).fit_table(table, ~hide, I.FitContext(seed=None, table=table,
                                                                     hidden_index=table.index[hide]))
    FZ.B6Factorized(0, 1.0).fit_table(table, ~hide, I.FitContext(seed=None, table=table,
                                                                 hidden_index=table.index[hide]))
    y_bad = y.copy()
    y_bad[np.flatnonzero(~hide)[0]] = np.nan
    t_bad = table_for(inputs, y_bad)
    with pytest.raises(ValueError):
        FZ.B6Factorized(0, 1.0).fit_table(t_bad, ~hide, ctx(t_bad, ~hide))
    with pytest.raises(KeyError):
        FZ.B6Factorized(0, 1.0).fit_table(I.RowTable(t_bad_frame(inputs, y)), np.ones(inputs.n, bool),
                                          I.FitContext(seed=1))
    with pytest.raises(ValueError):
        FZ.B6Factorized(-1, 1.0)
    with pytest.raises(ValueError):
        FZ.B6Factorized(1, 0.0)


def t_bad_frame(inputs, y) -> pd.DataFrame:
    """A RowTable frame that was never registered (B6 must refuse to guess its inputs)."""
    return pd.DataFrame({SG.METAL_COL: inputs.metal_unit, SG.ELEMENT_COL: inputs.metal_unit, SG.SYSTEM_COL: inputs.system,
                         SG.ACID_ANION_COL: "nitrate", SG.LOG_ACID_COL: 0.0, SG.LOG_EXT_COL: 0.0,
                         I.ID_COL: [f"X{i}" for i in range(inputs.n)], I.TARGET_COL: y}, index=inputs.index)


# --------------------------------------------------------------------------------------------- #
# tuning, intervals, checks
# --------------------------------------------------------------------------------------------- #

def test_select_config_tie_rule() -> None:
    maes = {(k, l): 1.0 for k in FZ.RANKS for l in FZ.LAMBDAS}
    assert FZ.select_config(maes) == (0, 10.0)                            # all tied -> lowest rank, strongest penalty
    maes[(3, 0.1)] = 0.80
    maes[(2, 1.0)] = 0.804
    maes[(2, 10.0)] = 0.806
    assert FZ.select_config(maes) == (2, 1.0)                              # within 0.005 of the best, lower rank
    maes[(1, 10.0)] = 0.8049
    assert FZ.select_config(maes) == (1, 10.0)
    assert FZ.select_config(maes, ranks=(0,)) == (0, 10.0)
    maes[(0, 1.0)] = 0.2
    assert FZ.select_config(maes, ranks=(0,)) == (0, 1.0)
    with pytest.raises(ValueError):
        FZ.select_config({**maes, (1, 1.0): float("nan")})


def test_tuned_conformal_equals_the_plain_wrapper_at_the_selected_configuration() -> None:
    inputs, y, truth = planted(seed=4, mixture_systems=2)
    table = table_for(inputs, y)
    outer = labels(truth, truth["cells"][:3])
    mask = ~cell_mask(table, outer)
    inner = [c for c in labels(truth, truth["cells"]) if c not in outer][::7][:9]
    spl = CellSplitter(inner)
    c = ctx(table, mask)
    fits = FZ.fit_b6_and_b6r0(table, mask, c, spl, chunk_size=4, calibration="tuning_residuals")
    q = np.flatnonzero(~mask)
    for variant, w in fits.items():
        k, lam = w.selected
        plain = I.ConformalWrapper(FZ.B6Factorized(k, lam, name=variant), splitter=spl).fit_table(table, mask, c)
        assert np.allclose(w.residuals, plain.residuals, atol=1e-8)
        assert w.calibration_units == plain.calibration_units
        pw, pp = w.predict_positions(q), plain.predict_positions(q)
        assert np.allclose(pw["mean_logD"], pp["mean_logD"], atol=1e-10)
        for lv in I.LEVELS:
            assert np.isclose(w.quantiles[lv], plain.quantiles[lv], atol=1e-8)
        assert (pw["upper_80"] >= pw["mean_logD"]).all() and pw["conformal_n_calibration"].iloc[0] == len(w.residuals)
        rec = w.selection_record()
        assert rec["variant"] == variant and rec["n_inner_splits"] == len(inner)
    assert fits["B6r0"].selected[0] == 0
    maes = fits["B6"].tuning.maes()
    assert fits["B6"].selected == FZ.select_config(maes)
    assert fits["B6r0"].selected == FZ.select_config(maes, ranks=(0,))
    # the one-call wrapper gives the same selection and intervals
    solo = FZ.B6TunedConformal("B6", spl, chunk_size=1, calibration="tuning_residuals").fit_table(table, mask, c)
    assert solo.selected == fits["B6"].selected and np.allclose(solo.residuals, fits["B6"].residuals, atol=1e-10)
    # a tuning run of another training set or seed is refused; several seeds per wrapper are refused
    other = FZ.run_inner_tuning(table, mask, ctx(table, mask, seed=1), spl)
    with pytest.raises(AssertionError):
        FZ.B6TunedConformal("B6", spl).fit_from_tuning(other, table, mask, c)
    with pytest.raises(ValueError):
        FZ.B6TunedConformal("B6", spl, seeds=[1, 2])
    # section 7 compute plan item 1: tuning restricted to the first inner fold
    first = FZ.run_inner_tuning(table, mask, c, spl, fold_subset=(0,))
    assert set(first.folds) == {0} and len(first.units) == sum(1 for i in range(len(inner)) if i % 3 == 0)


class _Only:
    """The splits of a splitter restricted to some inner folds."""

    def __init__(self, spl, folds):
        self.spl, self.folds = spl, set(folds)

    def splits(self, table, mask, context):
        return [sp for sp in self.spl.splits(table, mask, context) if sp.fold in self.folds]


def test_cross_fitted_calibration_never_reuses_the_selecting_rows() -> None:
    """Task X finding V-LP-01: with every inner fold tuned, inner fold j's residuals are those of the configuration
    selected on the other folds; with the first inner fold tuned, the selected configuration on the next fold; with no
    second fold the arm is not calibrated."""
    inputs, y, truth = planted(seed=4, mixture_systems=2)
    table = table_for(inputs, y)
    outer = labels(truth, truth["cells"][:3])
    mask = ~cell_mask(table, outer)
    inner = [c for c in labels(truth, truth["cells"]) if c not in outer][::7][:9]
    spl = CellSplitter(inner)
    c = ctx(table, mask)
    fits = FZ.fit_b6_and_b6r0(table, mask, c, spl, chunk_size=4)
    q = np.flatnonzero(~mask)
    for variant, w in fits.items():
        tun = w.tuning
        assert w.calibration == "cross_fit" and w.calibration_record["status"] == "calibrated"
        want = []
        for j in sorted(set(tun.folds)):
            cj = tun.select(w.ranks, w.lambdas, w.tie_margin, exclude_folds=(j,))
            assert w.calibration_record["per_inner_fold"][str(j)]["config"] == f"k{cj[0]}_lam{cj[1]:g}"
            plain = I.ConformalWrapper(FZ.B6Factorized(cj[0], cj[1], name=variant),
                                       splitter=_Only(spl, [j])).fit_table(table, mask, c)
            want.append(plain.residuals)
        assert np.allclose(w.residuals, np.concatenate(want), atol=1e-8)
        assert np.isfinite(w.predict_positions(q)["upper_95"]).all()
    # first inner fold tuned: calibration on the next fold with the selected configuration only
    firsts = FZ.fit_b6_and_b6r0(table, mask, c, spl, inner_mode="first")
    for variant, w in firsts.items():
        assert set(w.tuning.folds) == {0} and w.calibration_record["calibration_fold"] == 1
        k, lam = w.selected
        plain = I.ConformalWrapper(FZ.B6Factorized(k, lam, name=variant),
                                   splitter=_Only(spl, [1])).fit_table(table, mask, c)
        assert np.allclose(w.residuals, plain.residuals, atol=1e-8)
        assert w.selected == FZ.run_inner_tuning(table, mask, c, spl, fold_subset=(0,)).select(w.ranks, w.lambdas)
    # a single inner fold: not calibrated (NaN intervals), never residuals of the selecting rows
    one = FZ.fit_b6_and_b6r0(table, mask, c, _Only(spl, [0]))
    for w in one.values():
        assert w.calibration_record["status"].startswith("not_calibrated") and len(w.residuals) == 0
        pw = w.predict_positions(q)
        assert pw["lower_80"].isna().all() and np.isfinite(pw["mean_logD"]).all()
    lone = FZ.fit_b6_and_b6r0(table, mask, c, _Only(spl, [0]), inner_mode="first")
    assert all(w.calibration_record["status"].startswith("not_calibrated") for w in lone.values())


def test_inner_selection_averages_over_the_design_units_and_the_init_seed() -> None:
    """Task X findings V-03 and V-02: the inner macro MAE averages per InnerSplit.row_units (a unit pooled over the
    splits holding it), not per split; the factor initialisation takes init_seed (the section 15 model seed)."""
    inputs, y, truth = planted(seed=8)
    table = table_for(inputs, y)
    mask = np.ones(table.n, bool)
    sizes = pd.Series(1, index=range(table.n)).groupby([table.state, table.sys]).size()
    cells = labels(truth, truth["cells"][::13][:6])

    class Batched:
        name = "batched_test"

        def splits(self, tab, m, context):
            out = []
            for f in range(3):
                pair = cells[2 * f:2 * f + 2]
                hid = m & cell_mask(tab, pair)
                cal = np.flatnonzero(hid)
                keep = cal[: max(1, len(cal) - 3 * f)]              # unequal cell sizes inside a split
                units = np.array([f"{tab.state_labels[tab.state[p]]} x {tab.sys_labels[tab.sys[p]]}" for p in keep],
                                 dtype=object)
                out.append(I.InnerSplit(unit=f"b{f}", fold=f, train_mask=m & ~hid, cal_positions=keep,
                                        hidden_positions=cal, row_units=units))
            return out
    assert len(sizes)
    tun = FZ.run_inner_tuning(table, mask, ctx(table), Batched(), require_row_units=True)
    cfg = (1, 1.0)
    err = np.concatenate([np.abs(yy - pp) for yy, pp in zip(tun.y, tun.predictions[cfg])])
    units = np.concatenate(tun.row_units)
    per_cell = pd.Series(err).groupby(units).mean()
    assert np.isclose(tun.macro_mae(cfg), per_cell.mean())
    per_split = np.mean([np.mean(np.abs(yy - pp)) for yy, pp in zip(tun.y, tun.predictions[cfg])])
    assert len(per_cell) > len(tun.y) and not np.isclose(tun.macro_mae(cfg), per_split)
    assert np.isclose(tun.macro_mae(cfg, exclude_folds=(0,)), per_cell.drop(sorted(set(tun.row_units[0]))).mean())
    with pytest.raises(ValueError, match="row_units"):
        FZ.run_inner_tuning(table, mask, ctx(table), CellSplitter(cells), require_row_units=True)
    # a splitter without row units counts one unit per split
    plain = FZ.run_inner_tuning(table, mask, ctx(table), CellSplitter(cells))
    assert not plain.units_from_splits and np.isclose(plain.macro_mae(cfg), np.mean(
        [np.mean(np.abs(yy - pp)) for yy, pp in zip(plain.y, plain.predictions[cfg])]))
    # the registered V1 splitter carries row units (publication group, REMAINDER below 20 outer-training rows)
    g = I.GroupKFoldCalibration(3).splits(table, mask, ctx(table))
    assert g and all(sp.row_units is not None and len(sp.row_units) == len(sp.cal_positions) for sp in g)
    # init_seed moves the factor start of k >= 1 only; the split draw keeps context.seed
    a = FZ.run_inner_tuning(table, mask, ctx(table), Batched(), init_seed=10_000_033)
    b = FZ.run_inner_tuning(table, mask, ctx(table), Batched(), init_seed=10_001_042)
    assert a.seed == b.seed == SEED and a.init_seed == 10_000_033
    assert np.array_equal(a.predictions[(0, 1.0)][0], b.predictions[(0, 1.0)][0])
    assert not np.array_equal(a.predictions[(2, 0.1)][0], b.predictions[(2, 0.1)][0])
    w = FZ.fit_b6_and_b6r0(table, mask, ctx(table), Batched(), init_seed=10_000_033)["B6"]
    assert w.selection_record()["init_seed"] == 10_000_033 and w.arm.seed == 10_000_033


def test_inner_tuning_guards_isolation_and_v6_rows() -> None:
    inputs, y, truth = planted(seed=12)
    table = table_for(inputs, y)
    inner = labels(truth, truth["cells"][::11][:4])
    spl = CellSplitter(inner)
    mask = np.ones(table.n, bool)
    bad = I.FitContext(seed=SEED, table=table, v6_mask=pd.Series(False, index=table.index),
                       isolation_check=lambda tr, te: {"ok": False})
    with pytest.raises(AssertionError):
        FZ.run_inner_tuning(table, mask, bad, spl)
    v6 = pd.Series(False, index=table.index)
    v6.iloc[np.flatnonzero(cell_mask(table, inner[:1]))[:1]] = True
    with pytest.raises(AssertionError):
        FZ.run_inner_tuning(table, mask, I.FitContext(seed=SEED, table=table, v6_mask=v6,
                                                      isolation_check=lambda tr, te: {"ok": True}), spl)
    with pytest.raises(ValueError):
        FZ.run_inner_tuning(table, mask, I.FitContext(seed=None, table=table, v6_mask=v6,
                                                      isolation_check=lambda tr, te: {"ok": True}), spl)


def test_group_kfold_calibration_design_runs() -> None:
    inputs, y, truth = planted(seed=13)
    table = table_for(inputs, y, n_groups=9)
    w = FZ.B6TunedConformal("B6r0", I.GroupKFoldCalibration(3)).fit_table(table, np.ones(table.n, bool), ctx(table))
    assert len(w.calibration_units) == 3 and w.selected[0] == 0 and np.isfinite(w.quantiles[0.8])


def test_batched_exact_check_and_the_recolouring_decision() -> None:
    cells = [f"c{i}" for i in range(10)]
    exact = pd.Series(np.linspace(0.5, 1.0, 10), index=cells)
    ok = FZ.batched_exact_check(exact, exact.iloc[::-1] + 0.009)
    assert ok["passed"] and np.isclose(ok["delta"], 0.009) and ok["n_units"] == 10
    bad = FZ.batched_exact_check(exact, exact + 0.01)
    assert not bad["passed"]
    assert FZ.batched_check_decision(ok) == {"status": "passed", "scheme": "batched", "label": "batched",
                                             "s1_forced_undecided": False, "next_action": None}
    step = FZ.batched_check_decision(bad)
    assert step["status"] == "recolour" and "max_cells_per_batch=4" in step["next_action"]
    assert FZ.batched_check_decision(bad, ok)["scheme"] == "batched_max4"
    fail = FZ.batched_check_decision(bad, bad)
    assert fail["label"] == "batched (check failed)" and fail["s1_forced_undecided"]
    with pytest.raises(ValueError):
        FZ.batched_exact_check(exact, exact.iloc[:9])
    assert FZ.INIT_STREAM_TAG not in FI.STREAM_TAGS.values()


# --------------------------------------------------------------------------------------------- #
# features and provenance
# --------------------------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    return LOAD.load_model_rows()


def test_features_equal_the_featureset_b6_preset_and_exclude_provenance(corpus) -> None:
    df = corpus
    assert not set(FZ.SOURCE_COLUMNS) & F.FORBIDDEN_SOURCE_COLUMNS
    F.assert_feature_columns_allowed(FZ.SOURCE_COLUMNS)
    systems = np.array(sorted(df["extractant_system_key"].unique()))
    perm = np.random.default_rng(7).permutation(len(systems))
    tr = df[df["extractant_system_key"].isin(set(systems[perm[:120]]))]
    qu = df[df["extractant_system_key"].isin(set(systems[perm[120:170]]))]
    rows = pd.concat([tr, qu])
    rows = rows.assign(pub_group="g", doi_primary="10.1/x")                 # id / provenance columns a frame may carry
    inputs = FZ.B6Inputs.from_frame(rows)
    ntr = len(tr)
    cv = N.condition_vector(rows)
    ref = F.FeatureSet.for_arm("B6").fit(tr, cv).transform(qu, cv).frame
    full = FZ.B6Encoder(side_categoricals=True).fit(inputs, np.arange(ntr))
    e = full.transform(inputs, np.arange(ntr, len(rows)))
    got = np.hstack([e.Zm, e.Zs, e.X])
    assert list(full.feature_names) == list(ref.columns)
    assert np.array_equal(got, ref.to_numpy(dtype=float))
    lean = FZ.B6Encoder().fit(inputs, np.arange(ntr))                         # the registered reading: numeric z
    assert lean.metal.names == tuple(f"metal__{c}" for c in F.METAL_NUMERIC)
    assert len(lean.ext.names) == 42 and not any("=" in c for c in lean.metal.names + lean.ext.names)
    F.assert_feature_columns_allowed(lean.feature_names)
    assert not any(tok in c.lower() for c in lean.feature_names for tok in ("pub", "doi", "log_d", "measurement"))
    xq = inputs.metal_unit[~rows["g19_metal_state"].notna().to_numpy()]
    assert all(u.endswith("(?)") for u in xq)
    with pytest.raises(ValueError):
        FZ.B6Inputs.from_arrays(["a"], ["M"], ["S"], metal_numeric=pd.DataFrame({"g19_publication_id": [1.0]}),
                                extractant_numeric=pd.DataFrame({"z": [1.0]}),
                                condition_numeric=pd.DataFrame({"x": [1.0]}))
    with pytest.raises(ValueError):
        FZ.B6Inputs.from_arrays(["a"], ["M"], ["S"], metal_numeric=pd.DataFrame({"z": [1.0]}),
                                extractant_numeric=pd.DataFrame({"z": [1.0]}),
                                condition_numeric=pd.DataFrame({"log_D": [1.0]}))


def test_frame_path_on_real_inputs_with_a_synthetic_target(corpus) -> None:
    df = corpus
    keep = df["extractant_system_key"].value_counts().index[:6]
    rows = df[df["extractant_system_key"].isin(keep)].copy()
    rng = np.random.default_rng(3)
    rows["log_D"] = rng.standard_normal(len(rows))                            # the archive target is never used
    fr = I.prepare_frame(rows)
    test = fr["g19_metal_state"].isin(["Eu(III)"]).to_numpy() & fr["g19_metal_state"].notna().to_numpy()
    train, query = fr[~test], fr[test]
    assert len(query)
    arm = FZ.B6Factorized(1, 1.0).fit(train, I.FitContext(seed=SEED))
    p = arm.predict(query)
    assert len(p) == len(query) and np.isfinite(p["mean_logD"]).all() and set(p["b6_rank"]) == {1}
    assert (p["row_id"].to_numpy() == query.index.to_numpy()).all()
    with pytest.raises(AssertionError):
        arm.predict(train.iloc[:2])
    w = FZ.B6TunedConformal("B6r0", I.GroupKFoldCalibration(3))
    w.fit(train, I.FitContext(seed=SEED, v6_mask=pd.Series(False, index=train.index),
                              isolation_check=lambda tr, te: {"ok": True}))
    assert np.isfinite(w.predict(query)[["lower_95", "upper_95"]].to_numpy()).all()


# --------------------------------------------------------------------------------------------- #
# runtime (reported)
# --------------------------------------------------------------------------------------------- #

def test_runtime_per_fit_on_a_12k_row_synthetic() -> None:
    """Real-corpus shape: 74 metal units (p_m = 8), 259 systems (p_s = 42), ~1,600 cells, 12,000 rows, 11 numeric
    condition columns plus a 3-level anion (Pc = 15)."""
    rng = np.random.default_rng(19)
    n_m, n_s, n_rows = 74, 259, 12000
    Zm, Zs = rng.standard_normal((n_m, 8)), rng.standard_normal((n_s, 42))
    cells = set()
    for s in range(n_s):
        for m in rng.choice(n_m, 3, replace=False):
            cells.add((int(m), s))
    while len(cells) < 1610:
        cells.add((int(rng.integers(n_m)), int(rng.integers(n_s))))
    cells = sorted(cells)
    per = rng.multinomial(n_rows - len(cells), np.ones(len(cells)) / len(cells)) + 1
    m_idx = np.repeat([c[0] for c in cells], per)
    s_idx = np.repeat([c[1] for c in cells], per)
    U, V = rng.standard_normal((n_m, 2)), rng.standard_normal((n_s, 2)) * 0.5
    y = U[m_idx, 0] + V[s_idx, 0] + np.einsum("ra,ra->r", U[m_idx], V[s_idx]) + 0.3 * rng.standard_normal(n_rows)
    x = rng.standard_normal((n_rows, 11))
    inputs = FZ.B6Inputs.from_arrays(
        [f"r{i}" for i in range(n_rows)], [f"M{m:02d}" for m in m_idx], [f"S{s:03d}" for s in s_idx],
        metal_numeric=pd.DataFrame(Zm[m_idx], columns=[f"zm{j}" for j in range(8)]),
        extractant_numeric=pd.DataFrame(Zs[s_idx], columns=[f"zs{j}" for j in range(42)]),
        condition_numeric=pd.DataFrame(x, columns=[f"x{j}" for j in range(11)]),
        condition_categorical=pd.DataFrame({"acid_anion": rng.choice(["a", "b", "c"], n_rows)}))
    table = table_for(inputs, y)
    mask = np.ones(n_rows, bool)
    times = {}
    for k in FZ.RANKS:
        t0 = time.perf_counter()
        arm = FZ.B6Factorized(k, 1.0, tol=0.0).fit_table(table, mask, ctx(table))
        times[k] = time.perf_counter() - t0
        assert arm.info["n_sweeps"] == (0 if k == 0 else FZ.MAX_SWEEPS)
    d = arm.design
    print(f"B6 runtime per fit on {n_rows} synthetic rows (cells {d.n_cells}, Jm {d.Jm}, Js {d.Js}, p_m {d.p_m}, "
          f"p_s {d.p_s}, Pc {d.Pc}; 50 sweeps for k >= 1, 1 BLAS thread): "
          + ", ".join(f"k={k} {t:.2f}s" for k, t in times.items()))
    assert all(t < 60.0 for t in times.values())

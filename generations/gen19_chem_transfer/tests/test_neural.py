"""M1 / M2 factorised neural arms (``gen19ct/models/neural.py``).

Every test runs on SYNTHETIC rows: real metal labels and real system keys give the static descriptor tables their
input, but every condition value and every target is generated here.  No MODEL row is read, no registered fold is
used, nothing is scored against a measured log D, nothing touches V6 and no file is written.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import pytest
import torch

from gen19ct.chemistry import support_graph as SG
from gen19ct.data import load as LOAD
from gen19ct.folds import io as FI
from gen19ct.models import features as F
from gen19ct.models import inner_design as ID
from gen19ct.models import interface as I
from gen19ct.models import neural as NN

SYSTEMS = {
    "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC": 1.2,                      # TODGA
    "CCCCOP(=O)(OCCCC)OCCCC": -0.8,                                                    # TBP
    "CCCCC(CC)CN(CC(CC)CCCC)C(=O)COCC(=O)N(CC(CC)CCCC)CC(CC)CCCC": 0.9,               # TEHDGA
    "CCCCC(CC)CN(CC(CC)CCCC)C(=O)C(C)C": -0.3,                                         # DEHiBA
    "CCCCCCCCN(C)C(=O)COCC(=O)N(C)CCCCCCCC": 0.5,                                      # DMDODGA
    "CCCCCCCC(=O)N(CCCCCC)CCCCCC": -0.5,                                               # DHOA
    "CCCCN(CCCC)C(=O)COCC(=O)N(CCCC)CCCC": 0.2,                                        # TBDGA
    "CCCCCCN(CCCCCC)C(=O)CCCCC": -1.0,                                                 # DHHA
}
METALS = ("La(III)", "Ce(III)", "Pr(III)", "Nd(III)", "Sm(III)", "Eu(III)", "Gd(III)", "Dy(III)", "Er(III)",
          "Yb(III)", "Am(III)", "Cm(III)", "U(VI)", "Th(IV)")
NOVEL = "CCCCCCN(CCCCCC)C(=O)CCC(=O)N(CCCCCC)CCCCCC"          # in no descriptor table
REAL_WIDTHS = dict(p_metal=28, p_ligand=75, p_condition=34, n_series=4, n_ox=8, n_element=41, n_system=260)


def _row(i: int, state: str | None, element: str, system: str, rng: np.random.Generator, *, family="diglycolamide",
         mechanism="NEUTRAL_SOLVATING", pub="g0") -> dict:
    acid = float(10 ** rng.uniform(-1.5, 0.7))
    ext = float(10 ** rng.uniform(-2.0, -0.7))
    return {
        I.ID_COL: f"syn_{i:05d}", SG.METAL_COL: state, SG.ELEMENT_COL: element, "g19_ox": np.nan,
        SG.SYSTEM_COL: system, SG.FAMILY_COL: family, SG.MECH_COL: mechanism, SG.SMILES_COL: system,
        "components": np.array([{"role": "organic_extractant", "name": "E", "smiles_raw": None,
                                 "smiles_canonical": system, "structure_source": "synthetic", "concentration_M": ext,
                                 "concentration_raw": f"{ext} M"}], dtype=object),
        "acid_primary": "HNO3", "acid_signature": "HNO3", "acid_anion": "nitrate", "acid_concentration_M": acid,
        "acid_concentration_organic_M": np.nan, "nitrate_concentration_M": np.nan, "n_organic_extractants": 1,
        "extractant_primary_concentration_M": ext, "metal_concentration_M": float(10 ** rng.uniform(-4, -2)),
        "phase_ratio_org_aq": 1.0, "solvent_key": "dodecane:1", "solvent_primary": "dodecane",
        "solvent_components": np.array(["dodecane"], dtype=object), "modifier_name": np.nan,
        "modifier_concentration_M": np.nan, "temperature_C": float(rng.uniform(20, 30)), "contact_time_min": 30.0,
        "shaking_time_min": np.nan, SG.LOG_ACID_COL: math.log10(acid), SG.LOG_EXT_COL: math.log10(ext),
        I.PUB_GROUP_COL: pub, "g19_publication_id": pub, "doi_primary": f"10.0/{pub}",
    }


def synthetic_frame(seed: int = 0, rows_per_cell: int = 6, metals=METALS, systems=tuple(SYSTEMS)) -> pd.DataFrame:
    """Every metal x system cell with ``rows_per_cell`` rows; log D = system offset + radius-shaped metal term +
    their product + 2 log10[L] + noise (all synthetic)."""
    rng = np.random.default_rng(seed)
    recs = []
    for s_i, sy in enumerate(systems):
        for m in metals:
            for _ in range(rows_per_cell):
                recs.append(_row(len(recs), m, SG.metal_properties(m)["symbol"], sy, rng, pub=f"g{s_i % 4}"))
    fr = pd.DataFrame.from_records(recs)
    fr.index = pd.Index([f"r{i}" for i in range(len(fr))])
    r8 = np.array([SG.metal_properties(m)["r_cn8"] for m in fr[SG.METAL_COL]], dtype=float)
    r8[~np.isfinite(r8)] = 1.0
    so = fr[SG.SYSTEM_COL].map(SYSTEMS).fillna(0.0).to_numpy(dtype=float)
    fr[I.TARGET_COL] = so + 3.0 * (1.15 - r8) + 2.0 * so * (1.15 - r8) + 2.0 * fr[SG.LOG_EXT_COL] + \
        0.05 * rng.normal(size=len(fr))
    return fr


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_frame()


def _dims(**kw) -> NN.InputDims:
    return NN.InputDims(**{**REAL_WIDTHS, **kw})


def planted_arrays(seed: int, n_metals=30, n_systems=60, metals_per_system=10, rows_per_cell=6, p_m=28, p_l=75,
                   p_c=34, k=3, amp=1.5):
    """Encoded rows with a planted rank-``k`` bilinear metal x ligand structure in the descriptor blocks."""
    rng = np.random.default_rng(seed)
    U, V = rng.normal(size=(n_metals, k)), rng.normal(size=(n_systems, k))
    zm = U @ rng.normal(size=(p_m, k)).T + 0.1 * rng.normal(size=(n_metals, p_m))
    zl = V @ rng.normal(size=(p_l, k)).T + 0.1 * rng.normal(size=(n_systems, p_l))
    zm, zl = (zm - zm.mean(0)) / zm.std(0), (zl - zl.mean(0)) / zl.std(0)
    M = rng.normal(size=(k, k))
    bm, bl, beta = 0.5 * rng.normal(size=n_metals), 0.5 * rng.normal(size=n_systems), 0.3 * rng.normal(size=p_c)
    cells = np.array([(m, s) for s in range(n_systems) for m in rng.choice(n_metals, metals_per_system, replace=False)])
    rows = np.repeat(cells, rows_per_cell, axis=0)
    xc = rng.normal(size=(len(rows), p_c))
    y = bm[rows[:, 0]] + bl[rows[:, 1]] + xc @ beta + amp * np.einsum("ik,kl,il->i", U[rows[:, 0]], M, V[rows[:, 1]]) \
        + 0.1 * rng.normal(size=len(rows))
    ids = {"series_id": rows[:, 0] % 3 + 1, "ox_id": rows[:, 0] % 5 + 1, "element_id": rows[:, 0] + 1,
           "system_id": rows[:, 1] + 1}
    enc = NN.EncodedRows(pd.RangeIndex(len(rows)), zm[rows[:, 0]].astype(np.float32), zl[rows[:, 1]].astype(np.float32),
                         xc.astype(np.float32), {k_: v.astype(np.int64) for k_, v in ids.items()})
    dims = NN.InputDims(p_m, p_l, p_c, 4, 6, n_metals + 1, n_systems + 1)
    return enc, y, rows, dims, rng


# ------------------------------------------------------------------------------------------------ registered constants

def test_registered_settings_and_seed_rule():
    assert (NN.LEARNING_RATE, NN.WARMUP_EPOCHS, NN.BATCH_SIZE, NN.HUBER_DELTA, NN.GRAD_CLIP_NORM) == (3e-3, 10, 512, 1.0, 5.0)
    assert (NN.PATIENCE, NN.MAX_EPOCHS, NN.N_THREADS, NN.DROPOUT) == (30, 300, 2, 0.1)
    assert NN.COND_HIDDEN == (32, 32) and NN.HEAD_HIDDEN == (64, 64)
    assert [(c.emb_dim, c.weight_decay, c.rank) for c in NN.m1_grid()] == \
        [(d, w, 0) for d in (4, 8, 16) for w in (1e-4, 1e-3, 1e-2)]
    m1 = NN.NeuralConfig(8, 1e-2)
    assert [(c.emb_dim, c.weight_decay, c.rank) for c in NN.m2_grid(m1)] == [(8, 1e-2, 2), (8, 1e-2, 4), (8, 1e-2, 8)]
    with pytest.raises(ValueError):
        NN.m2_grid(NN.NeuralConfig(8, 1e-2, 4))
    assert NN.registered_model_seed(0) == 42 + 9_999_991
    assert NN.registered_model_seed(3) == 42 + 3 * 1009 + 9_999_991
    assert NN.SELECTION_TOLERANCE == 0.005


def test_warmup_schedule_and_optimizer_settings_are_applied(monkeypatch):
    assert NN.warmup_learning_rate(0, 3) == pytest.approx(3e-3 / 30)
    assert NN.warmup_learning_rate(29, 3) == pytest.approx(3e-3) and NN.warmup_learning_rate(500, 3) == 3e-3
    seen: list[tuple[float, float]] = []
    clipped: list[float] = []

    class SpyAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            seen.append((self.param_groups[0]["lr"], self.param_groups[0]["weight_decay"]))
            return super().step(closure)

    real_clip = torch.nn.utils.clip_grad_norm_

    def spy_clip(params, max_norm, *a, **k):
        clipped.append(max_norm)
        return real_clip(params, max_norm, *a, **k)

    monkeypatch.setattr(NN.torch.optim, "AdamW", SpyAdamW)
    monkeypatch.setattr(NN.nn.utils, "clip_grad_norm_", spy_clip)
    enc, y, rows, dims, _ = planted_arrays(seed=4, n_systems=26, rows_per_cell=4)      # 1,040 rows: 3 batches
    NN.train_network(enc, y, dims, NN.NeuralConfig(4, 1e-2), model_seed=7, n_epochs=12)
    assert len(seen) == 36 and {w for _, w in seen} == {1e-2} and set(clipped) == {5.0}
    assert [lr for lr, _ in seen] == pytest.approx([3e-3 * min(1.0, (s + 1) / 30) for s in range(36)])


# ------------------------------------------------------------------------------------------------ parameter count

def test_parameter_count_limits_on_real_widths():
    dims = _dims()
    d, r = 16, 8
    expected = (28 * d + 4 * d + 8 * d + 41 * d + 75 * d + 260 * d + (34 * 32 + 32) + (32 * 32 + 32) +
                ((2 * d + 32) * 64 + 64) + (64 * 64 + 64) + (64 + 1) + 2 * d * r)
    n = NN.parameter_count(dims, NN.NeuralConfig(d, 1e-3, r))
    assert n == expected and n <= NN.MAX_PARAMETERS
    for cfg in NN.m1_grid() + NN.m2_grid(NN.NeuralConfig(16, 1e-3)):
        assert NN.parameter_count(dims, cfg) <= NN.MAX_PARAMETERS
    with pytest.raises(AssertionError):                        # a vocabulary that would break the 200 k limit
        NN.parameter_count(_dims(n_system=20_000), NN.NeuralConfig(16, 1e-3, 8))
    with pytest.raises(ValueError):
        NN.NeuralConfig(32, 1e-3)
    print(f"\n[param count] real widths, d=16 r=8: {n}; M1 d=4: {NN.parameter_count(dims, NN.NeuralConfig(4, 1e-3))}")


# ------------------------------------------------------------------------------------------------ planted structure

def test_m2_recovers_planted_bilinear_structure_better_than_m1_on_held_out_cells():
    enc, y, rows, dims, rng = planted_arrays(seed=0)
    cells = np.unique(rows, axis=0)
    pick = rng.choice(len(cells), size=60, replace=False)
    key = rows[:, 0] * 10_000 + rows[:, 1]
    held = np.isin(key, cells[pick[:30], 0] * 10_000 + cells[pick[:30], 1])
    val = np.isin(key, cells[pick[30:], 0] * 10_000 + cells[pick[30:], 1])
    tr = ~(held | val)
    units = key.astype(str)
    _, held_codes = np.unique(units[held], return_inverse=True)
    mae = {}
    for cfg in (NN.NeuralConfig(8, 1e-3, 0), NN.NeuralConfig(8, 1e-3, 4)):
        r = NN.train_network(enc.take(np.flatnonzero(tr)), y[tr], dims, cfg, model_seed=NN.registered_model_seed(0),
                             valid=enc.take(np.flatnonzero(val)), y_valid=y[val], valid_units=units[val],
                             patience=20, max_epochs=120)
        pred = NN._predict_std(r.net, enc.take(np.flatnonzero(held)).tensors()) * r.y_sd + r.y_mean
        mae[cfg.step] = NN.macro_mae(y[held], pred, held_codes)
    print(f"\n[planted bilinear] held-out cell macro MAE: M1 {mae['M1']:.3f}, M2 {mae['M2']:.3f} (sd y {y.std():.2f})")
    assert mae["M2"] < 0.75 * mae["M1"]


# ------------------------------------------------------------------------------------------------ early stopping

def test_early_stopping_restores_the_best_epoch_and_the_refit_replays_it():
    enc, y, rows, dims, _ = planted_arrays(seed=1, n_systems=20, rows_per_cell=4)
    val = rows[:, 1] >= 17
    tr = ~val
    y_anti = -y                                                  # validation target the model moves away from
    seed = NN.registered_model_seed(2)
    cfg = NN.NeuralConfig(4, 1e-3, 2)
    etr, eva = enc.take(np.flatnonzero(tr)), enc.take(np.flatnonzero(val))
    r = NN.train_network(etr, y[tr], dims, cfg, model_seed=seed, valid=eva, y_valid=y_anti[val],
                         valid_units=rows[val, 1].astype(str), patience=5, max_epochs=60)
    h = r.history
    assert r.best_epoch == int(h["epoch"].iloc[int(np.argmin(h["valid_macro_mae"].to_numpy()))])
    assert r.best_valid_macro_mae == float(h["valid_macro_mae"].min())
    assert r.epochs_run == r.best_epoch + 5 < 60                  # stopped by patience, after the best epoch
    _, codes = np.unique(rows[val, 1].astype(str), return_inverse=True)
    pv = NN._predict_std(r.net, eva.tensors()) * r.y_sd + r.y_mean
    assert NN.macro_mae(y_anti[val], pv, codes) == pytest.approx(r.best_valid_macro_mae, abs=1e-6)
    assert h["valid_macro_mae"].iloc[-1] > r.best_valid_macro_mae    # the last epoch was worse, yet restored
    refit = NN.train_network(etr, y[tr], dims, cfg, model_seed=seed, n_epochs=r.best_epoch)
    assert NN.state_digest(refit.net) == NN.state_digest(r.net)
    assert np.array_equal(NN._predict_std(refit.net, eva.tensors()), NN._predict_std(r.net, eva.tensors()))


def test_median_epochs_and_selection_tolerance():
    assert NN.median_epochs([10, 20, 31]) == 20
    assert NN.median_epochs([10, 21]) == 16                        # 15.5 rounds half up
    assert NN.median_epochs([1]) == 1
    scores = pd.DataFrame({"inner_macro_mae": [0.300, 0.304, 0.3049, 0.290, 0.2949],
                           "rank": [0, 0, 0, 0, 0], "n_parameters": [100, 200, 200, 400, 200],
                           "weight_decay": [1e-4, 1e-4, 1e-2, 1e-3, 1e-3]})
    assert NN.select_config(scores) == 4          # within 0.005 of 0.290: 0.2949 (200 params) beats 0.290 (400)
    scores.loc[4, "inner_macro_mae"] = 0.2960
    assert NN.select_config(scores) == 3
    ranks = pd.DataFrame({"inner_macro_mae": [0.201, 0.200, 0.203], "rank": [2, 4, 8], "n_parameters": [10, 20, 40],
                          "weight_decay": [1e-3] * 3})
    assert NN.select_config(ranks) == 0
    wd = pd.DataFrame({"inner_macro_mae": [0.2, 0.2, 0.2], "rank": [0] * 3, "n_parameters": [10] * 3,
                       "weight_decay": [1e-4, 1e-2, 1e-3]})
    assert NN.select_config(wd) == 1                               # equal size: stronger penalty


# ------------------------------------------------------------------------------------------------ determinism

def test_two_identical_runs_give_identical_predictions_on_cpu():
    enc, y, rows, dims, _ = planted_arrays(seed=3, n_systems=15, rows_per_cell=4)
    cfg = NN.NeuralConfig(8, 1e-3, 4)
    threads = torch.get_num_threads()
    a = NN.train_network(enc, y, dims, cfg, model_seed=NN.registered_model_seed(1), n_epochs=12)
    b = NN.train_network(enc, y, dims, cfg, model_seed=NN.registered_model_seed(1), n_epochs=12)
    c = NN.train_network(enc, y, dims, cfg, model_seed=NN.registered_model_seed(2), n_epochs=12)
    t = enc.tensors()
    pa, pb, pc = (NN._predict_std(r.net, t) for r in (a, b, c))
    identical = bool(np.array_equal(pa, pb)) and NN.state_digest(a.net) == NN.state_digest(b.net)
    print(f"\n[determinism] two identical CPU runs identical: {identical}; other seed max |diff| "
          f"{float(np.max(np.abs(pa - pc))):.3g}")
    assert identical
    assert not np.array_equal(pa, pc)
    assert torch.get_num_threads() == threads                        # thread count and RNG state restored


def test_arm_fit_is_deterministic_and_row_order_free(frame):
    tr = frame[frame[SG.METAL_COL] != "Gd(III)"]
    q = frame[frame[SG.METAL_COL] == "Gd(III)"].drop(columns=[I.TARGET_COL])
    arm = NN.FactorisedArm(NN.NeuralConfig(4, 1e-3, 2), n_epochs=6, model_seed=NN.registered_model_seed(0))
    p1 = arm.clone().fit(tr).predict(q)
    p2 = arm.clone().fit(tr.sample(frac=1.0, random_state=5)).predict(q)
    pd.testing.assert_frame_equal(p1, p2)


# ------------------------------------------------------------------------------------------------ unseen units

def test_unseen_metal_system_and_unknown_state_get_zero_offsets(frame):
    rng = np.random.default_rng(11)
    tr = frame[frame[SG.METAL_COL] != "Dy(III)"]
    queries = pd.DataFrame.from_records([
        _row(90_000, "Dy(III)", "Dy", next(iter(SYSTEMS)), rng),                                 # element unseen
        _row(90_001, "Nd(III)", "Nd", NOVEL, rng, family="succinamide"),                          # system unseen
        _row(90_002, None, "Nd", next(iter(SYSTEMS)), rng),                                       # X(?) row
        _row(90_003, "Nd(III)", "Nd", next(iter(SYSTEMS)), rng),                                  # all seen
    ])
    queries.index = pd.Index(["q_dy", "q_novel", "q_x", "q_seen"])
    arm = NN.FactorisedArm(NN.NeuralConfig(8, 1e-3, 4), n_epochs=15, model_seed=NN.registered_model_seed(4)).fit(tr)
    pred = arm.predict(queries).set_index("row_id")
    assert list(pred.columns[:len(I.PREDICTION_COLUMNS) - 1]) == list(I.PREDICTION_COLUMNS[1:])
    assert np.isfinite(pred["mean_logD"]).all() and pred["std_logD"].isna().all() and pred["lower_95"].isna().all()
    assert (pred["fallback_level"] == "M2").all()
    assert pred.loc["q_dy", "fallback_reason"] == "element_offset_zero" and not pred.loc["q_dy", "element_offset_seen"]
    assert pred.loc["q_novel", "fallback_reason"] == "system_offset_zero"
    assert pred.loc["q_x", "fallback_reason"] == "ox_offset_zero"
    assert pred.loc["q_seen", "fallback_reason"] == ""
    enc = arm._encode(queries)
    net = arm.result.net
    t = enc.tensors()
    assert int(t["element_id"][0]) == 0 and int(t["system_id"][1]) == 0 and int(t["ox_id"][2]) == 0
    with torch.no_grad():
        for w in (net.e_series.weight, net.e_ox.weight, net.e_element.weight, net.delta_l.weight):
            assert torch.count_nonzero(w[0]) == 0                    # the reserved row stayed zero through training
        em, el = net.metal_embedding(t), net.ligand_embedding(t)
        assert torch.allclose(em[0], net.e_shared(t["z_m"][0]) + net.e_series(t["series_id"][0]) + net.e_ox(t["ox_id"][0]),
                              atol=1e-6)
        assert torch.allclose(el[1], net.w_l(t["z_l"][1]), atol=1e-6)
        assert torch.count_nonzero(net.e_element(t["element_id"][:1])) == 0
    emb = arm.embeddings(queries)
    assert emb.shape == (4, 2 * 8 + 1)
    mt, st = arm.metal_embeddings(), arm.system_embeddings()
    assert "Dy(III)" not in mt.index and set(mt.index) == set(tr[SG.METAL_COL]) and set(st.index) == set(SYSTEMS)
    assert np.allclose(mt.loc["Nd(III)"].to_numpy()[:8], emb.loc["q_seen"].to_numpy()[:8], atol=1e-6)
    W = arm.bilinear_matrix()
    assert W.shape == (8, 8) and np.linalg.matrix_rank(W, tol=1e-6) <= 4


# ------------------------------------------------------------------------------------------------ train-only, guards

def test_inputs_are_fitted_on_training_rows_and_never_read_provenance(frame):
    tr = frame[frame[SG.SYSTEM_COL] != "CCCCOP(=O)(OCCCC)OCCCC"]
    q = frame[frame[SG.SYSTEM_COL] == "CCCCOP(=O)(OCCCC)OCCCC"]
    arm = NN.FactorisedArm(NN.NeuralConfig(4, 1e-2), n_epochs=4, model_seed=NN.registered_model_seed(0)).fit(tr)
    rec = arm.fit_record()
    assert rec["feature_state_digest"] == F.FeatureSet.for_arm("M1").fit(NN._order_rows(tr)).state_digest
    assert rec["y_mean"] == pytest.approx(float(tr[I.TARGET_COL].mean()), abs=1e-9)
    assert rec["y_sd"] == pytest.approx(float(tr[I.TARGET_COL].std(ddof=0)), abs=1e-9)
    before = arm.encoder.state_digest
    base = arm.predict(q)
    noisy = q.assign(**{c: "LEAK" for c in ("doi_primary", "g19_publication_id", I.PUB_GROUP_COL)},
                     **{I.TARGET_COL: 99.0, I.ID_COL: [f"zz{i}" for i in range(len(q))]})
    pd.testing.assert_frame_equal(arm.predict(noisy), base)          # provenance, ids and the target are never read
    assert arm.encoder.state_digest == before
    cols = [c for cols in arm.encoder.block_columns.values() for c in cols]
    NN.assert_inputs_allowed(cols, arm.encoder.features.source_columns)
    assert not set(arm.encoder.features.source_columns) & set(LOAD.PROVENANCE_COLUMNS)
    for bad in ("canonical_measurement_id", "doi_primary", "metal__g19_publication_id", "condition__pub_group=g1",
                "log_D", "extractant__source_record_id__missing"):
        with pytest.raises(AssertionError):
            NN.assert_inputs_allowed(cols + [bad], arm.encoder.features.source_columns)
    with pytest.raises(AssertionError):
        NN.assert_inputs_allowed(cols, list(arm.encoder.features.source_columns) + ["g19_study_id"])
    with pytest.raises(AssertionError):
        arm.predict(tr.iloc[:3])                                     # a query row is a training row
    ctx = I.FitContext(hidden_index=pd.Index(tr.index[:2]))
    with pytest.raises(AssertionError):
        arm.clone().fit(tr, ctx)


def test_splits_from_folds_and_tuning_end_to_end(frame):
    outer = frame[~((frame[SG.METAL_COL] == "Eu(III)") & (frame[SG.SYSTEM_COL] == next(iter(SYSTEMS))))]
    ids = outer[I.ID_COL]
    cells = [("Sm(III)", list(SYSTEMS)[1]), ("Er(III)", list(SYSTEMS)[2]), ("Am(III)", list(SYSTEMS)[3])]
    folds = []
    for k, (m, s) in enumerate(cells):
        own = (outer[SG.METAL_COL] == m) & (outer[SG.SYSTEM_COL] == s)
        hid = ids[own].tolist()
        label = f"{m} x {s}"
        folds.append(FI.make_fold(design="V5_inner", variant="inner", scheme="inner_batched", fold_id=f"i{k}_b000",
                                  half="NA", seed=104729, hidden=hid, scored=hid, unit_type="cell", units=[label],
                                  row_unit={r: label for r in hid}, meta={"inner_fold": k}))
    v6 = pd.Series(False, index=outer.index)
    excl = pd.Series(False, index=outer.index)
    excl[outer.index[(outer[SG.METAL_COL] == "Sm(III)") & (outer[SG.SYSTEM_COL] == list(SYSTEMS)[1])][:2]] = True
    splits = NN.splits_from_folds(outer, folds, v6_mask=v6, exclude_from_scoring=excl)
    assert [sp.inner_fold for sp in splits] == [0, 1, 2]
    assert len(splits[0].valid_index) == len(folds[0].scored_row_ids) - 2
    assert all(not len(sp.train_index.intersection(sp.hidden_index)) for sp in splits)
    assert [sp.inner_fold for sp in NN.splits_from_folds(outer, folds, v6_mask=v6, inner_folds=[0])] == [0]
    v6_bad = v6.copy()
    v6_bad[outer.index[outer[I.ID_COL] == folds[1].scored_row_ids[0]]] = True
    with pytest.raises(AssertionError):
        NN.splits_from_folds(outer, folds, v6_mask=v6_bad)
    unguarded = [NN.ValidationSplit(sp.name, sp.inner_fold, sp.train_index, sp.valid_index, sp.valid_units,
                                    sp.hidden_index, None) for sp in splits]
    configs = [NN.NeuralConfig(4, 1e-2), NN.NeuralConfig(8, 1e-4)]
    with pytest.raises(AssertionError):
        NN.tune(outer, unguarded, configs, model_seed=1)
    res = NN.tune(outer, splits[:2], configs, model_seed=NN.registered_model_seed(0), patience=3, max_epochs=12)
    assert len(res.scores) == 2 and int(res.scores["selected"].sum()) == 1 and len(res.fits) == 4
    sel = res.scores[res.scores["selected"]].iloc[0]
    assert res.selected.label() == sel["config"]
    assert res.n_epochs == NN.median_epochs(res.fits.loc[res.fits["config"] == sel["config"], "best_epoch"].tolist())
    assert len(res.inner_predictions) == sum(len(sp.valid_index) for sp in splits[:2])
    u = res.inner_predictions["unit"].to_numpy()
    _, codes = np.unique(u, return_inverse=True)
    assert NN.macro_mae(res.inner_predictions["log_D"], res.inner_predictions["pred"], codes) == \
        pytest.approx(float(sel["inner_macro_mae"]))
    arm = res.arm().fit(outer)
    q = frame.loc[frame.index.difference(outer.index)]
    assert np.isfinite(arm.predict(q)["mean_logD"]).all() and arm.n_epochs == res.n_epochs
    m2 = NN.tune_m2(outer, splits[:1], res.selected, fold_index=0, patience=2, max_epochs=6)
    assert {c for c in m2.scores["rank"]} == {2, 4, 8} and m2.selected.emb_dim == res.selected.emb_dim
    assert m2.step == "M2" and m2.model_seed == NN.registered_model_seed(0)
    # cross-fitted calibration (task X finding V-LP-01): the selection re-run without an inner fold, from the records
    import json
    rec = json.loads(json.dumps({"scores": res.scores.to_dict(orient="records"),
                                 "fits": res.fits.to_dict(orient="records"),
                                 "sue": res.split_unit_errors.to_dict(orient="records")}, default=float))
    ue = res.split_unit_errors
    assert set(ue["inner_fold"]) == {0, 1} and int(ue["n_rows"].sum()) == len(configs) * len(res.inner_predictions)
    cfg_all, ep_all, _ = NN.select_excluding_folds(rec["scores"], rec["fits"], rec["sue"], ())
    assert cfg_all == res.selected and ep_all == res.n_epochs              # no fold excluded: the tuner's selection
    cfg0, ep0, info = NN.select_excluding_folds(rec["scores"], rec["fits"], rec["sue"], (0,))
    only1 = res.fits[res.fits["inner_fold"] == 1]
    best1 = min(configs, key=lambda c: (float(only1.loc[only1["config"] == c.label(), "valid_macro_mae"].iloc[0]),))
    scores1 = {c.label(): float(only1.loc[only1["config"] == c.label(), "valid_macro_mae"].iloc[0]) for c in configs}
    assert info["excluded_inner_folds"] == [0] and info["n_splits"] == 1
    assert scores1[cfg0.label()] <= min(scores1.values()) + NN.SELECTION_TOLERANCE + 1e-12 and best1 is not None
    assert ep0 == int(only1.loc[only1["config"] == cfg0.label(), "best_epoch"].iloc[0])
    with pytest.raises(ValueError):
        NN.select_excluding_folds(rec["scores"], rec["fits"], rec["sue"], (0, 1))


def test_simultaneous_v5_splits_fold_mean_selection_and_m2_keeps_m1_values():
    """Addendum 1 items 1-2 through the neural tuner: the V5 validation splits are the simultaneous inner cells (one per
    inner fold, every cell of the fold absent from the split's training rows, isolation check on each), the selection
    score is the mean over the three folds of the fold's unit-macro MAE, the cross-fitted re-selection of M2 keeps the
    outer fold's M1 values (reading 6(d)), the refit epoch count is the median over the three fits."""
    fr = synthetic_frame(seed=2, rows_per_cell=12)
    first_sys = next(iter(SYSTEMS))
    outer = fr[~((fr[SG.METAL_COL] == "Eu(III)") & (fr[SG.SYSTEM_COL] == first_sys))]
    calls = []

    def guard(tr, te):
        calls.append((pd.Index(tr), pd.Index(te)))
        return {"ok": True}
    c = I.FitContext(seed=104729, v6_mask=pd.Series(False, index=fr.index), isolation_check=guard)
    design = ID.SimultaneousInnerCells(8, 1, 3, 3, 6)
    vs = NN.simultaneous_v5_splits(outer, c, design=design)
    assert [sp.inner_fold for sp in vs] == [0, 1, 2] and len(calls) == 3
    table = I.RowTable(outer)
    inner = design.splits(table, np.ones(table.n, bool), c)
    for sp, isp in zip(vs, inner):
        cells = {tuple(x) for x in isp.meta["cells"]}
        tr = outer.loc[sp.train_index]
        assert not any(((tr[SG.METAL_COL] == a) & (tr[SG.SYSTEM_COL] == b)).any() for a, b in cells)
        va = outer.loc[sp.valid_index]
        assert set(zip(va[SG.METAL_COL], va[SG.SYSTEM_COL])) == {tuple(x) for x in isp.meta["scored_cells"]}
        assert len(set(sp.valid_units)) == len(isp.meta["scored_cells"]) and sp.guard.startswith("interface_inner_design")
        assert not len(sp.train_index.intersection(sp.hidden_index)) and sp.valid_index.isin(sp.hidden_index).all()
    # every split passed the isolation check before it was returned (training rows vs calibration rows)
    for (tr, te), sp in zip(calls, vs):
        assert set(tr) == set(sp.train_index) and set(te) == set(sp.valid_index)
    with pytest.raises(ValueError, match="isolation_check"):
        NN.simultaneous_v5_splits(outer, I.FitContext(seed=104729, v6_mask=pd.Series(False, index=fr.index)), design=design)
    configs = [NN.NeuralConfig(4, 1e-2), NN.NeuralConfig(8, 1e-4)]
    res = NN.tune(outer, vs, configs, model_seed=NN.registered_model_seed(0), patience=2, max_epochs=6)
    assert len(res.fits) == 3 * len(configs) and (res.scores["n_inner_folds"] == 3).all()
    ue = res.split_unit_errors
    agg = ue.groupby(["config", "inner_fold", "unit"])[["n_rows", "sum_abs_error"]].sum()
    per_fold = (agg["sum_abs_error"] / agg["n_rows"]).groupby(level=["config", "inner_fold"]).mean()
    for cfg in configs:
        want = per_fold.loc[cfg.label()].mean()
        got = float(res.scores.loc[res.scores["config"] == cfg.label(), "inner_macro_mae"].iloc[0])
        assert got == pytest.approx(want)
        rows = res.fold_scores[res.fold_scores["config"] == cfg.label()]
        assert sorted(rows["inner_fold"]) == [0, 1, 2] and rows["macro_mae"].mean() == pytest.approx(want)
        fits = res.fits[res.fits["config"] == cfg.label()]
        assert sorted(fits["inner_fold"]) == [0, 1, 2]
    assert res.n_epochs == NN.median_epochs(res.fits.loc[res.fits["config"] == res.selected.label(), "best_epoch"].tolist())
    # M2 on the same splits with the retained M1 values; its cross-fitted re-selection keeps them for every excluded fold
    m2 = NN.tune_m2(outer, vs, res.selected, fold_index=0, patience=2, max_epochs=4)
    assert set(m2.scores["rank"]) == {2, 4, 8} and (m2.scores["emb_dim"] == res.selected.emb_dim).all()
    rec = {"scores": m2.scores.to_dict(orient="records"), "fits": m2.fits.to_dict(orient="records"),
           "sue": m2.split_unit_errors.to_dict(orient="records")}
    for j in (0, 1, 2):
        cfg, epochs, info = NN.select_excluding_folds(rec["scores"], rec["fits"], rec["sue"], (j,))
        assert cfg.emb_dim == res.selected.emb_dim and cfg.weight_decay == res.selected.weight_decay
        assert cfg.rank in (2, 4, 8) and info["inner_folds_used"] == sorted({0, 1, 2} - {j})
        others = m2.fits[(m2.fits["config"] == cfg.label()) & (m2.fits["inner_fold"] != j)]["best_epoch"]
        assert len(others) == 2 and epochs == NN.median_epochs(others.tolist())
    # splits_from_inner_splits refuses a split without row units
    bare = I.InnerSplit(unit="x", fold=0, train_mask=inner[0].train_mask, cal_positions=inner[0].cal_positions,
                        hidden_positions=inner[0].hidden_positions)
    with pytest.raises(ValueError, match="row_units"):
        NN.splits_from_inner_splits(outer, table, [bare])


def test_conformal_wrapper_fast_path_fills_intervals(frame):
    tr = frame[frame[SG.METAL_COL] != "Yb(III)"]
    q = frame[frame[SG.METAL_COL] == "Yb(III)"]
    table = I.RowTable(frame)
    ctx = I.FitContext(seed=104729, table=table, v6_mask=pd.Series(False, index=frame.index),
                       isolation_check=lambda a, b: {"ok": True})
    arm = NN.FactorisedArm(NN.NeuralConfig(4, 1e-3), n_epochs=3, model_seed=NN.registered_model_seed(0), rows=frame)
    w = I.ConformalWrapper(arm, splitter=I.GroupKFoldCalibration(3)).fit(tr, ctx)
    assert len(w.calibration_units) == 3 and np.isfinite(w.residuals).all()
    out = w.predict_positions(table.positions(q.index))
    assert (out["lower_95"] < out["mean_logD"]).all() and (out["mean_logD"] < out["upper_95"]).all()
    assert (out["lower_95"] <= out["lower_50"]).all() and out["conformal_n_calibration"].iloc[0] == len(w.residuals)
    assert (out["model_step"] == "M1").all()
    direct = arm.clone().fit(tr).predict(q.drop(columns=[I.TARGET_COL]))
    assert np.array_equal(direct["mean_logD"].to_numpy(), out["mean_logD"].to_numpy())


# ------------------------------------------------------------------------------------------------ timing

def test_per_epoch_time_on_12k_synthetic_rows_with_real_feature_widths():
    rng = np.random.default_rng(0)
    n = 12_000
    dims = _dims()
    ids = {"series_id": rng.integers(1, dims.n_series, n), "ox_id": rng.integers(1, dims.n_ox, n),
           "element_id": rng.integers(1, dims.n_element, n), "system_id": rng.integers(1, dims.n_system, n)}
    enc = NN.EncodedRows(pd.RangeIndex(n), rng.normal(size=(n, dims.p_metal)).astype(np.float32),
                         rng.normal(size=(n, dims.p_ligand)).astype(np.float32),
                         rng.normal(size=(n, dims.p_condition)).astype(np.float32),
                         {k: v.astype(np.int64) for k, v in ids.items()})
    y = rng.normal(size=n)
    val = enc.take(np.arange(600))
    report = {}
    for cfg in (NN.NeuralConfig(4, 1e-3, 0), NN.NeuralConfig(16, 1e-3, 8)):
        NN.train_network(enc, y, dims, cfg, model_seed=1, n_epochs=1)                     # warm-up of the kernels
        t0 = time.perf_counter()
        r = NN.train_network(enc.take(np.arange(600, n)), y[600:], dims, cfg, model_seed=1, valid=val, y_valid=y[:600],
                             valid_units=(np.arange(600) % 40).astype(str), patience=1000, max_epochs=4)
        report[cfg.label()] = ((time.perf_counter() - t0) / r.epochs_run, r.n_parameters)
    print("\n[timing] 12k synthetic rows, widths metal 28 / extractant 75 / condition 34, 2 threads, per epoch "
          "(train + validation): " + ", ".join(f"{k}: {v[0]:.3f} s ({v[1]} params)" for k, v in report.items()))
    assert all(v[0] < 5.0 for v in report.values())

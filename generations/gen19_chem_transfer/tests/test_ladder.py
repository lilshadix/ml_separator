"""The ablation ladder M3-M7 (``gen19ct/models/ladder.py``) and its runner (``scripts/g19_run_ladder.py``).

Every test runs on SYNTHETIC rows (real metal labels and real system keys feed the static descriptor tables; every
condition, mechanism label, publication group and target is generated here; at most a few hundred rows).  No MODEL row is
read, no registered fold is used, nothing is scored against a measured log D, nothing touches V6, and the runner tests
work in ``tmp_path`` with monkeypatched paths and a stub discovery record.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.evaluation import calibration as EC
from gen19ct.evaluation import discovery as D
from gen19ct.evaluation import transfer as ET
from gen19ct.folds import io as FI
from gen19ct.models import interface as I
from gen19ct.models import ladder as LAD
from gen19ct.models import neural as NN

SCRIPTS = paths.G19_ROOT / "scripts"


def _load_script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RL = _load_script("g19_run_ladder")
RD = RL.RD

# --------------------------------------------------------------------------------------------- synthetic corpus
SYSTEMS = {                                                   # system key -> (offset, mechanism label, pub group)
    "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC": (1.2, "NEUTRAL_SOLVATING", "g0"),          # TODGA
    "CCCCOP(=O)(OCCCC)OCCCC": (-0.8, "NEUTRAL_SOLVATING", "g1"),                                        # TBP
    "CCCCC(CC)CN(CC(CC)CCCC)C(=O)COCC(=O)N(CC(CC)CCCC)CC(CC)CCCC": (0.9, "NEUTRAL_SOLVATING", "g2"),   # TEHDGA
    "CCCCC(CC)CN(CC(CC)CCCC)C(=O)C(C)C": (-0.3, "SOFT_N_DONOR", "g3"),                                  # DEHiBA
    "CCCCCCCCN(C)C(=O)COCC(=O)N(C)CCCCCCCC": (0.5, "SOFT_N_DONOR", "g0"),                               # DMDODGA
    "CCCCCCCC(=O)N(CCCCCC)CCCCCC": (-0.5, "MIXED_NEUTRAL", "g1"),                                       # DHOA
    "CCCCN(CCCC)C(=O)COCC(=O)N(CCCC)CCCC": (0.2, "ACIDIC_CATION_EXCHANGE", "g2"),                       # TBDGA (label only)
    "CCCCCCN(CCCCCC)C(=O)CCCCC": (-1.0, "SYNERGISTIC", "g3"),                                           # DHHA (label only)
}
METALS = ("La(III)", "Ce(III)", "Pr(III)", "Nd(III)", "Sm(III)", "Eu(III)", "Gd(III)", "Am(III)", "Cm(III)", "U(VI)")
ACIDS = (0.37, 1.7, 2.9)
EXTS = (0.05, 0.1, 0.2)


def _row(i: int, state: str, system: str, acid: float, ext: float, pub: str, mech: str, temp: float = 25.0) -> dict:
    return {
        I.ID_COL: f"syn_{i:05d}", SG.METAL_COL: state, SG.ELEMENT_COL: SG.metal_properties(state)["symbol"], "g19_ox": np.nan,
        SG.SYSTEM_COL: system, SG.FAMILY_COL: "diglycolamide", SG.MECH_COL: mech, SG.SMILES_COL: system,
        "components": np.array([{"role": "organic_extractant", "name": "E", "smiles_raw": None, "smiles_canonical": system,
                                 "structure_source": "synthetic", "concentration_M": ext, "concentration_raw": f"{ext} M"}],
                               dtype=object),
        "acid_primary": "HNO3", "acid_signature": "HNO3", "acid_anion": "nitrate", "acid_concentration_M": acid,
        "acid_concentration_organic_M": np.nan, "nitrate_concentration_M": np.nan, "n_organic_extractants": 1,
        "extractant_primary_concentration_M": ext, "metal_concentration_M": 1e-3, "phase_ratio_org_aq": 1.0,
        "solvent_key": "dodecane:1", "solvent_primary": "dodecane", "solvent_components": np.array(["dodecane"], dtype=object),
        "modifier_name": np.nan, "modifier_concentration_M": np.nan, "temperature_C": temp, "contact_time_min": 30.0,
        "shaking_time_min": np.nan, SG.LOG_ACID_COL: math.log10(acid), SG.LOG_EXT_COL: math.log10(ext),
        I.PUB_GROUP_COL: pub, FI.GROUP_COL: pub, "g19_publication_id": pub, "doi_primary": f"10.0/{pub}",
    }


def synthetic_frame(seed: int = 0, systems=None, metals=METALS, acids=ACIDS, exts=EXTS, reps: int = 1,
                    pub_rule=None) -> pd.DataFrame:
    """Every metal x system cell on the same (acid, extractant) grid (so comparable pairs exist inside a publication
    group); log D = offset + radius term + 2 log10[L] + noise.  ``pub_rule(system_index, metal_index)`` overrides the
    per-system publication group (the inner V5 design hides whole groups, so a group must not cover a whole system)."""
    systems = SYSTEMS if systems is None else systems
    rng = np.random.default_rng(seed)
    recs = []
    for s_i, (sy, (off, mech, pub)) in enumerate(systems.items()):
        for m_i, m in enumerate(metals):
            g = pub if pub_rule is None else pub_rule(s_i, m_i)
            for a in acids:
                for e in exts:
                    for _ in range(reps):
                        recs.append(_row(len(recs), m, sy, a, e, g, mech))
    fr = pd.DataFrame.from_records(recs)
    fr.index = pd.Index([f"r{i}" for i in range(len(fr))], dtype=object)
    r8 = np.array([SG.metal_properties(m)["r_cn8"] for m in fr[SG.METAL_COL]], dtype=float)
    r8[~np.isfinite(r8)] = 1.0
    so = fr[SG.SYSTEM_COL].map({k: v[0] for k, v in systems.items()}).to_numpy(dtype=float)
    fr[I.TARGET_COL] = so + 3.0 * (1.15 - r8) + 2.0 * fr[SG.LOG_EXT_COL] + 0.05 * rng.normal(size=len(fr))
    return fr


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    fr = synthetic_frame()
    assert len(fr) == 8 * 10 * 9 == 720
    return fr


def _cfg(step="M2", **kw) -> LAD.LadderConfig:
    base = dict(step=step, emb_dim=4, weight_decay=1e-3, rank=2)
    base.update(kw)
    return LAD.LadderConfig(**base)


# --------------------------------------------------------------------------------------------- registered constants

def test_registered_constants_and_grids():
    assert LAD.STEPS == ("M3", "M4", "M5", "M6", "M7")
    assert LAD.TAU_GRID == (0.1, 0.3, 1.0) and LAD.LAMBDA_PAIR_GRID == (0.3, 1.0) and LAD.LAMBDA_PHYS_GRID == (0.1, 1.0)
    assert LAD.LAMBDA_SRC == 1.0 and LAD.MIN_EXPERT_ROWS == 200 and LAD.N_MEMBERS == 5
    assert LAD.EXPERT_MECHANISMS == ("NEUTRAL_SOLVATING", "SOFT_N_DONOR", "MIXED_NEUTRAL")
    # training settings are neural's, never redefined
    assert (NN.LEARNING_RATE, NN.WARMUP_EPOCHS, NN.BATCH_SIZE, NN.HUBER_DELTA, NN.GRAD_CLIP_NORM, NN.PATIENCE,
            NN.MAX_EPOCHS) == (3e-3, 10, 512, 1.0, 5.0, 30, 300)
    # member seeds: section 15 rule + k, distinct within and across folds
    seeds = [LAD.member_seed(f, k) for f in (0, 1, 7) for k in range(5)]
    assert len(set(seeds)) == 15 and LAD.member_seed(3, 0) == NN.registered_model_seed(3)
    with pytest.raises(ValueError):
        LAD.member_seed(0, 5)


def test_each_step_searches_only_its_own_hyperparameters():
    prev = _cfg("M2")
    m3 = LAD.step_grid("M3", prev)
    assert [c.experts for c in m3] == [True] and m3[0].tau is None and m3[0].lambda_pair == 0
    LAD.assert_search_scope("M3", m3, prev)
    prev3 = m3[0]
    m4 = LAD.step_grid("M4", prev3)
    assert [c.tau for c in m4] == [0.1, 0.3, 1.0] and all(c.experts for c in m4)
    LAD.assert_search_scope("M4", m4, prev3)
    m5 = LAD.step_grid("M5", m4[1])
    assert [c.lambda_pair for c in m5] == [0.3, 1.0] and all(c.tau == 0.3 for c in m5)
    LAD.assert_search_scope("M5", m5, m4[1])
    m6 = LAD.step_grid("M6", m5[0])
    assert [c.lambda_phys for c in m6] == [0.1, 1.0] and all(c.lambda_pair == 0.3 and c.phys_terms == "ab" for c in m6)
    LAD.assert_search_scope("M6", m6, m5[0])
    m7 = LAD.step_grid("M7", m6[1])
    assert len(m7) == 1 and m7[0].heteroscedastic and m7[0].lambda_phys == 1.0
    # a configuration that changes an earlier hyperparameter is refused
    with pytest.raises(AssertionError, match="not its own"):
        LAD.assert_search_scope("M4", [replace(m4[0], rank=8)], prev3)
    with pytest.raises(AssertionError, match="not its own"):
        LAD.assert_search_scope("M5", [replace(m5[0], tau=1.0)], m4[1])
    # a removed step is skipped by the chain: M5 on M3 keeps tau None
    m5b = LAD.step_grid("M5", prev3)
    assert all(c.tau is None and c.experts for c in m5b)
    # labels and records round-trip
    for c in m3 + m4 + m5 + m6 + m7:
        assert LAD.LadderConfig.from_record(json.loads(json.dumps(c.record()))) == c
        assert c.label().startswith(c.step)
    # tie order: stronger penalty first
    assert LAD.tie_key(m4[0], 10, 0) < LAD.tie_key(m4[2], 10, 2)            # smaller tau
    assert LAD.tie_key(m5[1], 10, 1) < LAD.tie_key(m5[0], 10, 0)            # larger lambda_pair


def test_h5_toggle_configs():
    retained = _cfg("M6", lambda_phys=1.0, phys_terms="ab")
    a = LAD.h5_toggle_config("M6a_toggle", retained)
    b = LAD.h5_toggle_config("M6b_toggle", retained)
    assert a.phys_terms == "b" and b.phys_terms == "a" and a.lambda_phys == b.lambda_phys == 1.0
    off = _cfg("M5", lambda_pair=0.3)
    a2 = LAD.h5_toggle_config("M6a_toggle", off)
    assert a2.lambda_phys == 0.1 and a2.phys_terms == "a" and a2.lambda_pair == 0.3
    only_a = _cfg("M6", lambda_phys=0.1, phys_terms="a")
    assert LAD.h5_toggle_config("M6a_toggle", only_a).lambda_phys == 0.0
    assert LAD.h5_toggle_config("M6b_toggle", only_a).phys_terms == "ab"


# --------------------------------------------------------------------------------------------- M3 routing

def test_routing_by_known_label_and_unseen_label_goes_to_pooled(frame):
    mech = LAD.system_mechanisms(frame)
    rt = LAD.fit_routing(mech, min_rows=50)
    assert rt.experts == ("NEUTRAL_SOLVATING", "SOFT_N_DONOR", "MIXED_NEUTRAL", "POOLED")
    assert rt.n_rows["NEUTRAL_SOLVATING"] == 3 * 90 and rt.n_rows["MIXED_NEUTRAL"] == 90 and rt.n_rows["POOLED"] == 180
    assert rt.expert_of("SOFT_N_DONOR") == 1 and rt.expert_of("ACIDIC_CATION_EXCHANGE") == rt.pooled
    assert rt.expert_of("SYNERGISTIC") == rt.pooled and rt.expert_of("UNKNOWN") == rt.pooled
    assert rt.expert_of("ION_PAIR_BASIC") == rt.pooled and rt.expert_of(None) == rt.pooled     # unseen labels
    # below the row threshold a named mechanism falls to the pooled expert
    rt2 = LAD.fit_routing(mech, min_rows=200)                 # NEUTRAL 270 rows stays; SOFT 180 and MIXED 90 fall to POOLED
    assert rt2.experts == ("NEUTRAL_SOLVATING", "POOLED") and rt2.expert_of("MIXED_NEUTRAL") == rt2.pooled
    assert rt2.expert_of("SOFT_N_DONOR") == rt2.pooled and rt2.n_rows["POOLED"] == 720 - 270
    # the encoder routes rows and an unseen label at transform time -> pooled
    enc = LAD.LadderEncoder(_cfg("M3", experts=True), min_expert_rows=50).fit(frame)
    e = enc.transform(frame)
    assert (e.ids["expert_id"] == rt.route(mech)).all()
    q = frame.iloc[:3].copy()
    q[SG.MECH_COL] = "ION_PAIR_BASIC"
    assert (enc.transform(q).ids["expert_id"] == rt.pooled).all()
    # systems-table fallback when the rows lack the column
    tab = pd.DataFrame({SG.SYSTEM_COL: list(SYSTEMS), SG.MECH_COL: [v[1] for v in SYSTEMS.values()]}).set_index(SG.SYSTEM_COL, drop=False)
    q2 = frame.iloc[:5].drop(columns=[SG.MECH_COL])
    assert list(LAD.system_mechanisms(q2, tab)) == list(mech[:5]) and list(LAD.system_mechanisms(q2)) == ["UNKNOWN"] * 5
    # the network: one condition encoder and one bilinear per expert, shared head; routed rows use their expert
    dims = enc.dims
    with NN.deterministic_torch(1):
        net = LAD.LadderNet(dims, _cfg("M3", experts=True), n_experts=enc.n_experts)
    assert len(net.cond_experts) == 4 and net.P_experts.shape == (3, 4, 2)
    t = e.take(np.arange(8)).tensors()
    h = net.condition_h(t)
    for k in range(8):
        ref = net.cond_experts[int(t["expert_id"][k])](t["x_c"][k:k + 1])
        assert torch.allclose(h[k:k + 1], ref)
    mats = net.expert_bilinear_matrices()
    assert len(mats) == 4 and all(m.shape == (4, 4) for m in mats)


# --------------------------------------------------------------------------------------------- M4 source hierarchy

def test_tau_penalty_and_unseen_group_offset_is_zero(frame):
    cfg = _cfg("M4", tau=0.3)
    enc = LAD.LadderEncoder(cfg).fit(frame)
    assert enc.group_col == FI.GROUP_COL and enc.groups == ["g0", "g1", "g2", "g3"]
    e = enc.transform(frame)
    assert set(e.ids["group_id"]) == {1, 2, 3, 4}
    q = frame.iloc[:4].copy()
    q[FI.GROUP_COL] = "never_seen"
    assert (enc.transform(q).ids["group_id"] == 0).all()
    with NN.deterministic_torch(3):
        net = LAD.LadderNet(enc.dims, cfg, n_groups=enc.n_groups)
        with torch.no_grad():
            net.b_group.weight[1:] = torch.tensor([[0.5], [-0.2], [0.1], [0.3]])
    assert net.b_group.weight[0].item() == 0.0                                  # padding row: the population prior
    y = NN._target(frame)
    _, y_sd = NN._sorted_stat(y)
    aux = LAD.build_fit_aux(frame, e, enc, cfg, y_sd=y_sd)
    assert aux.tau_std == pytest.approx(0.3 / y_sd) and aux.info["source"]["tau_logD"] == 0.3
    tt = e.tensors()
    b = torch.arange(16)
    y_std = torch.from_numpy(((y - y.mean()) / y_sd).astype(np.float32))
    net.eval()
    _, terms = LAD.composite_loss(net, tt, y_std, b, config=cfg, aux=aux, pair_batch=None, w=None)
    want = LAD.LAMBDA_SRC * (0.5 ** 2 + 0.2 ** 2 + 0.1 ** 2 + 0.3 ** 2) / (2 * aux.tau_std ** 2) / len(frame)
    assert terms["L_source"] == pytest.approx(want, rel=1e-5)
    # the offset enters the prediction for a seen group and is 0 for an unseen one
    with torch.no_grad():
        seen = net(tt)
        t0 = {k: v.clone() for k, v in tt.items()}
        t0["group_id"] = torch.zeros_like(t0["group_id"])
        unseen = net(t0)
    off = net.b_group.weight.squeeze(1)[tt["group_id"]]
    assert torch.allclose(seen - unseen, off, atol=1e-6)
    # smaller tau -> stronger penalty (tie order) and the L2 weight is 1 / tau_std^2 in standardised units
    aux2 = LAD.build_fit_aux(frame, e, enc, replace(cfg, tau=0.1), y_sd=y_sd)
    assert aux2.tau_std < aux.tau_std


# --------------------------------------------------------------------------------------------- M5 pairs

def test_pairs_are_generated_inside_training_rows_only_and_pass_isolation(frame):
    hidden = frame.index[(frame[SG.METAL_COL] == "Nd(III)") & (frame[SG.SYSTEM_COL] == list(SYSTEMS)[0])]
    train = frame.drop(index=hidden)
    pairs = LAD.training_pairs(train)
    assert len(pairs) > 0
    members = pd.Index(np.concatenate([pairs["idx_a"].to_numpy(dtype=object), pairs["idx_b"].to_numpy(dtype=object)]))
    assert members.isin(train.index).all() and not members.isin(hidden).any()
    assert (pairs["fold"] == "train").all() and (pairs["state_a"] != pairs["state_b"]).all()
    # same publication group, same system, identical condition key (the section 2 rule)
    g = train[FI.GROUP_COL]
    assert (pairs["idx_a"].map(g).to_numpy() == pairs["idx_b"].map(g).to_numpy()).all()
    s = train[SG.SYSTEM_COL]
    assert (pairs["idx_a"].map(s).to_numpy() == pairs["idx_b"].map(s).to_numpy()).all()
    ck = pairs["condition_key"]
    assert ck.notna().all()
    # Nd(III) x hidden system never appears: 9 other metals -> C(10,2) - 9 = 36 pairs per condition in that system
    sys0 = pairs[pairs[SG.SYSTEM_COL] == list(SYSTEMS)[0]]
    assert len(sys0) == 9 * 36
    LAD.assert_pairs_inside(pairs, train.index, hidden)
    # a pair reaching into the hidden rows is refused by pair_isolation_check
    bad = pairs.iloc[:1].copy()
    bad.loc[bad.index[0], "idx_b"] = hidden[0]
    with pytest.raises(AssertionError, match="pair isolation"):
        LAD.assert_pairs_inside(pd.concat([pairs, bad]), train.index, hidden)
    # the fit aux carries positions into the training rows only
    cfg = _cfg("M5", lambda_pair=0.3)
    enc = LAD.LadderEncoder(cfg).fit(train)
    e = enc.transform(train)
    aux = LAD.build_fit_aux(train, e, enc, cfg, y_sd=1.0)
    assert aux.pair_positions.shape == (len(pairs), 2) and aux.pair_positions.max() < len(train)
    assert aux.info["pairs"]["n_pairs"] == len(pairs)


# --------------------------------------------------------------------------------------------- M6 physics

def test_hinge_only_in_range_and_never_on_grid_rows(frame):
    fr = frame.copy()
    # one system gets a single extractant concentration (degenerate range), one row gets a grid-flagged acid
    single = list(SYSTEMS)[1]
    fr.loc[fr[SG.SYSTEM_COL] == single, ["extractant_primary_concentration_M", SG.LOG_EXT_COL]] = [0.1, math.log10(0.1)]
    fr.loc[fr[SG.SYSTEM_COL] == single, "components"] = fr.loc[fr[SG.SYSTEM_COL] == single, "components"].map(
        lambda c: np.array([{**c[0], "concentration_M": 0.1}], dtype=object))
    grid_row = fr.index[0]
    fr.loc[grid_row, "acid_concentration_M"] = 10 ** -0.37                   # 0.4265795..., on the 0.01 log grid
    fr.loc[grid_row, SG.LOG_ACID_COL] = -0.37
    cv = LAD.condition_vectors_of(fr, None)
    assert bool(cv.loc[grid_row, "acid_M_log10_grid"]) and int(cv["acid_M_log10_grid"].sum()) == 1
    mech = LAD.system_mechanisms(fr)
    mask, info = LAD.hinge_mask(fr, mech, cv)
    labels = fr[SG.MECH_COL].to_numpy(dtype=object)
    sysk = fr[SG.SYSTEM_COL].to_numpy(dtype=object)
    assert not mask[fr.index.get_loc(grid_row)]                                            # never on a grid row
    assert not mask[np.isin(labels, ["ACIDIC_CATION_EXCHANGE", "SYNERGISTIC"])].any()       # mechanism restriction
    assert not mask[sysk == single].any()                                                  # degenerate range
    ok = np.isin(labels, LAD.HINGE_MECHANISMS) & (sysk != single)
    ok[fr.index.get_loc(grid_row)] = False
    assert (mask == ok).all() and info["counts"]["n_acid_grid_flagged"] == 1 and info["counts"]["n_units_degenerate"] == 1
    for key, (lo, hi, n) in info["ranges"].items():
        assert lo == pytest.approx(math.log10(0.05)) and hi == pytest.approx(math.log10(0.2)) and n in (89, 90)
    # the hinge term is computed on eligible rows only and penalises a negative slope
    cfg = _cfg("M6", lambda_phys=1.0, phys_terms="a")
    enc = LAD.LadderEncoder(cfg).fit(fr, cv)
    e = enc.transform(fr, cv)
    aux = LAD.build_fit_aux(fr, e, enc, cfg, y_sd=1.0, cv=cv)
    assert (aux.hinge_mask == mask).all() and aux.ext_index is not None and aux.ext_scale > 0
    with NN.deterministic_torch(5):
        net = LAD.LadderNet(enc.dims, cfg)
    net.eval()
    y = NN._target(fr)
    y_std = torch.from_numpy(((y - y.mean()) / y.std()).astype(np.float32))
    tt = e.tensors()
    pos_grid = fr.index.get_loc(grid_row)
    b = torch.tensor([pos_grid] + [int(i) for i in np.flatnonzero(mask)[:15]])
    _, terms = LAD.composite_loss(net, tt, y_std, b, config=cfg, aux=aux, pair_batch=None, w=None)
    assert terms["n_hinge_rows_batch"] == 15 and terms["L_hinge"] >= 0
    b2 = torch.tensor([pos_grid] + [int(i) for i in np.flatnonzero(sysk == single)[:5]])
    _, terms2 = LAD.composite_loss(net, tt, y_std, b2, config=cfg, aux=aux, pair_batch=None, w=None)
    assert "L_hinge" not in terms2                                            # no eligible row -> no hinge term
    # finite-difference check of the autograd slope on one eligible row
    j = aux.ext_index
    row = {k: v[b[1:2]] for k, v in tt.items()}
    x = row["x_c"].clone().requires_grad_(True)
    row2 = dict(row, x_c=x)
    out = net(row2)
    g = torch.autograd.grad(out.sum(), x)[0][0, j].item() / aux.ext_scale
    eps = 1e-3
    xp = row["x_c"].clone()
    xp[0, j] += eps
    xm = row["x_c"].clone()
    xm[0, j] -= eps
    with torch.no_grad():
        fd = (net(dict(row, x_c=xp)) - net(dict(row, x_c=xm))).item() / (2 * eps) / aux.ext_scale
    assert g == pytest.approx(fd, abs=2e-3)


def test_smoothness_over_adjacent_z_only(frame):
    pairs = LAD.series_adjacent_pairs(frame[SG.METAL_COL].unique())
    assert pairs == [("La(III)", "Ce(III)"), ("Ce(III)", "Pr(III)"), ("Pr(III)", "Nd(III)"), ("Sm(III)", "Eu(III)"),
                     ("Eu(III)", "Gd(III)"), ("Am(III)", "Cm(III)")]
    # Nd -> Sm skipped (Pm absent), U(VI) never enters (not trivalent), Th(IV) neither
    assert ("Nd(III)", "Sm(III)") not in pairs
    assert LAD.series_adjacent_pairs(["U(VI)", "Th(IV)", "Am(III)"]) == []
    assert LAD.series_adjacent_pairs(["Nd(III)", "Pr(III)", None, "Nd(?)"]) == [("Pr(III)", "Nd(III)")]
    cfg = _cfg("M6", lambda_phys=0.5, phys_terms="b")
    enc = LAD.LadderEncoder(cfg).fit(frame)
    e = enc.transform(frame)
    aux = LAD.build_fit_aux(frame, e, enc, cfg, y_sd=1.0)
    assert aux.info["smoothness"]["n_pairs"] == 6 and not aux.hinge_mask.any()
    states = frame[SG.METAL_COL].to_numpy(dtype=object)[aux.smooth_positions]
    assert list(states) == ["La(III)", "Ce(III)", "Pr(III)", "Nd(III)", "Sm(III)", "Eu(III)", "Gd(III)", "Am(III)", "Cm(III)"]
    zs = np.array([SG.metal_properties(s)["Z"] for s in states])
    assert all(zs[b] - zs[a] == 1 for a, b in aux.smooth_pairs)
    with NN.deterministic_torch(7):
        net = LAD.LadderNet(enc.dims, cfg)
    net.eval()
    y = NN._target(frame)
    y_std = torch.from_numpy(((y - y.mean()) / y.std()).astype(np.float32))
    tt = e.tensors()
    _, terms = LAD.composite_loss(net, tt, y_std, torch.arange(8), config=cfg, aux=aux, pair_batch=None, w=None)
    with torch.no_grad():
        em = net.metal_embedding({k: v[torch.from_numpy(aux.smooth_positions)] for k, v in tt.items()})
        want = sum(float(((em[a] - em[b]) ** 2).sum()) for a, b in aux.smooth_pairs)
    assert terms["L_smooth"] == pytest.approx(want, rel=1e-5) and "L_hinge" not in terms


# --------------------------------------------------------------------------------------------- parameter count

def test_parameter_count_limit_on_real_widths():
    dims = NN.InputDims(p_metal=28, p_ligand=75, p_condition=34, n_series=4, n_ox=8, n_element=41, n_system=260)
    full = _cfg("M7", emb_dim=16, rank=8, experts=True, tau=0.3, lambda_pair=1.0, lambda_phys=1.0, heteroscedastic=True)
    n = LAD.parameter_count(dims, full, n_experts=4, n_groups=400)
    base = NN.parameter_count(dims, full.base)
    assert base < n <= NN.MAX_PARAMETERS
    with NN.deterministic_torch(0):
        net = LAD.LadderNet(dims, full, n_experts=4, n_groups=400)
    assert net.n_parameters == n and net.sd_head is not None and net.b_group.weight.shape == (401, 1)
    with pytest.raises(AssertionError, match="section 6"):
        with NN.deterministic_torch(0):
            LAD.LadderNet(dims, _cfg("M4", emb_dim=16, tau=0.3), n_groups=200_000)


# --------------------------------------------------------------------------------------------- M7

def test_m7_members_seeds_bootstrap_and_positive_sd(frame):
    cfg = _cfg("M7", heteroscedastic=True)
    hidden = frame.index[(frame[SG.METAL_COL] == "Eu(III)") & (frame[SG.SYSTEM_COL] == list(SYSTEMS)[0])]
    train, test = frame.drop(index=hidden), frame.loc[hidden]
    ens = LAD.EnsembleArm(cfg, n_epochs=2, fold_index=2, rows=frame, n_members=3).fit(train)
    assert ens.seeds == [NN.registered_model_seed(2) + k for k in range(3)] and len(set(ens.seeds)) == 3
    assert [m.model_seed for m in ens.members] == ens.seeds
    # each member trains on a publication-group bootstrap: multiplicities per group, undrawn groups absent
    for m, seed in zip(ens.members, ens.seeds):
        w = LAD.group_bootstrap_weights(train[FI.GROUP_COL], seed)
        per_group = w.groupby(train[FI.GROUP_COL].astype(str)).agg(["min", "max"])
        assert (per_group["min"] == per_group["max"]).all() and per_group["min"].sum() == 4     # 4 draws of 4 groups
        assert set(np.unique(w)) <= {0.0, 1.0, 2.0, 3.0, 4.0}
        assert set(m.train_index) == set(train.index[w.to_numpy() > 0])
        assert m.aux.row_weights is not None and (m.aux.row_weights > 0).all()
    ws = [LAD.group_bootstrap_weights(train[FI.GROUP_COL], s).to_numpy() for s in ens.seeds]
    assert not all(np.array_equal(ws[0], w) for w in ws[1:])
    out = ens.predict(test.drop(columns=[I.TARGET_COL]))
    assert list(out.columns) == list(LAD.PREDICTION_COLUMNS) and (out["n_members"] == 3).all()
    sd = out["std_logD"].to_numpy(dtype=float)
    assert np.isfinite(sd).all() and (sd > 0).all() and (out["gaussian_sd_logD"] > 0).all()
    assert (sd >= out["gaussian_sd_logD"].to_numpy(dtype=float) - 1e-9).all()      # total variance >= member variance
    mean, std, gsd, _ = ens.predict_arrays(test.drop(columns=[I.TARGET_COL]))
    mus = np.vstack([m.predict_arrays(test.drop(columns=[I.TARGET_COL]))[0] for m in ens.members])
    assert np.allclose(mean, mus.mean(0)) and np.allclose(std ** 2, gsd ** 2 + mus.var(0))
    # the SD head is trained on the detached trunk: the mean network of a member equals a plain fit of the same rows
    m0 = ens.members[0]
    plain = LAD.LadderArm(replace(cfg, heteroscedastic=False), n_epochs=2, model_seed=m0.model_seed, rows=frame,
                          row_weights=LAD.group_bootstrap_weights(train[FI.GROUP_COL], m0.model_seed)).fit(train)
    a, _, _ = m0.predict_arrays(test.drop(columns=[I.TARGET_COL]))
    b, _, _ = plain.predict_arrays(test.drop(columns=[I.TARGET_COL]))
    assert np.allclose(a, b, atol=1e-6)                                      # bit-identical mean network
    # the floor holds in standardised units
    assert (m0.result.net.sd_from(*[torch.zeros(1, k) for k in (4, 4, 32)]) >= LAD.SD_FLOOR_STD).all()


def test_normalised_split_conformal_coverage_on_gaussian_data():
    """The section 12 normalised-by-SD split conformal: on heteroscedastic Gaussian data with a correct SD, coverage on
    3,000 held-out rows is within +-0.03 of every nominal level."""
    rng = np.random.default_rng(19)
    n_cal, n_test = 3000, 3000
    sd = np.exp(rng.normal(-0.5, 0.4, size=n_cal + n_test))
    mu = rng.normal(0, 1, size=n_cal + n_test)
    y = mu + sd * rng.normal(size=n_cal + n_test)
    idx = pd.Index([f"c{i}" for i in range(n_cal)] + [f"t{i}" for i in range(n_test)])
    cal_idx, test_idx = idx[:n_cal], idx[n_cal:]
    v6 = pd.Series(False, index=idx)
    cal = EC.fit_split_conformal(y[:n_cal], mu[:n_cal], calibration_index=cal_idx, outer_test_index=test_idx, v6_mask=v6,
                                 design="V5", sd=sd[:n_cal], method=LAD.CONFORMAL_METHOD)
    q = {lv: float(cal.quantiles["__all__"][lv]) for lv in I.LEVELS}
    frame = pd.DataFrame({"mean_logD": mu[n_cal:], "std_logD": sd[n_cal:]}, index=test_idx)
    out = LAD.attach_normalised_intervals(frame, q, n_cal)
    for lv in I.LEVELS:
        pct = int(round(lv * 100))
        cov = float(np.mean((y[n_cal:] >= out[f"lower_{pct}"]) & (y[n_cal:] <= out[f"upper_{pct}"])))
        assert abs(cov - lv) <= 0.03, (lv, cov)
        assert (out[f"conformal_q{pct}"] == q[lv]).all()
    assert (out["intervals_status"] == "split_conformal_inner_normalised").all()
    # the quantile scales the SD: a wider SD row gets a wider interval
    w = (out["upper_80"] - out["lower_80"]).to_numpy()
    assert np.corrcoef(w, sd[n_cal:])[0, 1] > 0.999


# --------------------------------------------------------------------------------------------- tuning and the arm

def _splits(train: pd.DataFrame, n: int = 2) -> list[NN.ValidationSplit]:
    """Two inner splits hiding whole cells (label indices), guard-labelled for the tuner."""
    out = []
    cells = [("Ce(III)", list(SYSTEMS)[2]), ("Gd(III)", list(SYSTEMS)[3])]
    for k, (m, s) in enumerate(cells[:n]):
        hid = train.index[(train[SG.METAL_COL] == m) & (train[SG.SYSTEM_COL] == s)]
        out.append(NN.ValidationSplit(name=f"inner{k}", inner_fold=k, train_index=train.index.difference(hid, sort=False),
                                      valid_index=hid, valid_units=np.array([f"{m} x {s}"] * len(hid), dtype=object),
                                      hidden_index=hid, guard="test"))
    return out


def test_tune_step_selects_and_cross_fit_selection_keeps_earlier_values(frame):
    hidden = frame.index[(frame[SG.METAL_COL] == "Nd(III)") & (frame[SG.SYSTEM_COL] == list(SYSTEMS)[0])]
    train = frame.drop(index=hidden)
    vs = _splits(train)
    prev = _cfg("M3", experts=True)
    res = LAD.tune_step("M4", train, vs, prev, fold_index=1, patience=1, max_epochs=2, min_expert_rows=50)
    assert res.step == "M4" and len(res.fits) == 3 * 2 and set(res.scores["tau"]) == {0.1, 0.3, 1.0}
    assert (res.scores["experts"]).all() and (res.scores["rank"] == 2).all() and res.model_seed == NN.registered_model_seed(1)
    assert res.selected.tau in LAD.TAU_GRID and res.selected.experts and res.n_epochs >= 1
    assert res.min_expert_rows == 50 and res.arm(rows=frame).min_expert_rows == 50
    assert (res.fits["n_experts"] == 4).all() and (res.fits["n_groups"] == 4).all()
    rec = {"scores": res.scores.to_dict(orient="records"), "fits": res.fits.to_dict(orient="records"),
           "sue": res.split_unit_errors.to_dict(orient="records")}
    for j in (0, 1):
        cfg, epochs, info = LAD.select_ladder_excluding_folds(rec["scores"], rec["fits"], rec["sue"], (j,))
        assert cfg.experts and cfg.rank == 2 and cfg.tau in LAD.TAU_GRID and info["inner_folds_used"] == [1 - j]
    with pytest.raises(ValueError, match="M7 has no search"):
        LAD.tune_step("M7", train, vs, prev, fold_index=0)
    # the outer refit predicts with the diagnostics of the step
    arm = res.arm(rows=frame).fit(train)
    out = arm.predict(frame.loc[hidden].drop(columns=[I.TARGET_COL]))
    assert (out["ladder_step"] == "M4").all() and (out["tau"] == res.selected.tau).all() and out["group_offset_seen"].all()
    assert set(out["expert"]) <= set(arm.encoder.routing.experts) and np.isfinite(out["mean_logD"]).all()
    assert out["std_logD"].isna().all()                                       # intervals come from the wrapper
    rec2 = arm.fit_record()
    assert rec2["config"]["tau"] == res.selected.tau and len(rec2["group_offsets_logD"]) == 5


# --------------------------------------------------------------------------------------------- the runner

def _plan_state() -> D.PlanState:
    return D.PlanState(v5_batched_check="passed", v1_tenfold_check="passed")


def test_ladder_state_plan_and_jobs():
    st = RL.LadderState(stop_rule=False, discovery_ladder={"M1": True, "M2": True})
    assert st.retained_before("M3") == "M2" and st.retained_before("M7") == "M2"
    st.steps["M3"] = {"status": "kept", "kept": True}
    st.steps["M4"] = {"status": "removed", "kept": False}
    st.steps["M5"] = {"status": "kept", "kept": True}
    assert st.retained_before("M5") == "M3" and st.retained_before("M6") == "M5" and st.retained_before("M7") == "M5"
    assert st.decision_digest("M6") != st.decision_digest("M5") != st.decision_digest("M4")
    st2 = RL.LadderState(stop_rule=False, discovery_ladder={"M1": False, "M2": False})
    assert st2.retained_before("M3") == "M0"
    st3 = RL.LadderState(stop_rule=False, discovery_ladder={"M1": True, "M2": False})
    assert st3.retained_before("M3") == "M1"
    assert [s for s, _ in RL.ladder_plan(st)] == ["M3", "M4", "M5", "M6", "M7"]
    stop = RL.LadderState(stop_rule=True, discovery_ladder={"M1": True, "M2": True})
    plan = RL.ladder_plan(stop)
    assert [s for s, _ in plan] == ["M3", "M4", "M5", "M6", "M7"]
    assert all(w.startswith("exploratory_not_run") for s, w in plan[:4]) and "M7" in plan[4][1]
    jobs = RL.ladder_jobs("M3", _plan_state())
    assert [(j.arm, j.design, j.variant, j.scheme, j.seed, j.fold_seed) for j in jobs] == \
        [("M3", "V5", "primary", "batched", 104729, 104729), ("M3", "V1", "copy", "grouped10", 104729, 104729),
         ("M3", "V2", "element", "exact", 104729, None)]
    assert all(j.writes == ("M3",) and j.kind == "fit" for j in jobs)
    m2 = RL.discovery_main_jobs(_plan_state(), "M2")
    assert [j.design_dir for j in jobs] == [j.design_dir for j in m2]
    refits = RL.ladder_jobs("M3", _plan_state(), refits=True)
    assert [j.variant for j in refits] == ["strict", "hno3_only"] and all(j.arm == "M3" for j in refits)
    assert RL.demote_from("M5") == ["M7", "M6", "M5"] and RL.demote_from("M3") == list(D.DEMOTION_ORDER)
    # the runner digest depends on the ladder decisions and on the ladder code
    codes = {"own": "ladder-code", "discovery": "disc-code", "combined": "c"}
    r = RL.LadderRunner("M5", st, codes)
    assert r.chain_steps() == ["M2", "M3"]
    st_b = RL.LadderState(stop_rule=False, discovery_ladder={"M1": True, "M2": True}, steps=dict(st.steps))
    st_b.steps["M4"] = {"status": "kept", "kept": True}
    assert RL.LadderRunner("M5", st_b, codes).chain_steps() == ["M2", "M3", "M4"]
    assert st_b.decision_digest("M5") != st.decision_digest("M5")


# ---- a synthetic discovery tree for the gate and one end-to-end M3 fold ---------------------- #

def _cell_fold(df: pd.DataFrame, state: str, system: str) -> FI.Fold:
    own = (df[SG.METAL_COL] == state) & (df[SG.SYSTEM_COL] == system)
    ids = df.loc[own, I.ID_COL].tolist()
    label = f"{state} x {system}"
    return FI.make_fold(design="V5", variant="primary", scheme="exact", fold_id=f"{state}__cell0", half="S", seed=None,
                        hidden=ids, scored=ids, unit_type="cell", units=[label], row_unit={r: label for r in ids},
                        row_half={r: "S" for r in ids}, meta={"cells": [[state, system]], "component_aware": True})


def _corpus(tmp_path: Path, df: pd.DataFrame) -> tuple[Any, FI.Fold, str]:
    fold = _cell_fold(df, "Nd(III)", list(SYSTEMS)[0])
    folds_dir = tmp_path / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    FI.write_design([fold], folds_dir)
    stem = FI.design_stem("V5", "primary", "exact")
    v6 = pd.Series(False, index=df.index)
    coext = pd.Series(False, index=df.index)
    halves = {d: pd.Series("S", index=df.index) for d in ("V5", "V5P", "V5PAIR", "V1", "V2")}
    corpus = RD.Corpus(frame=df, slim=df, table=I.RowTable(df), v6=v6, coext=coext, systems=None, comps=None, cv=None,
                       pmap=None, row_half=halves, folds_dir=folds_dir)
    return corpus, fold, stem


def _ok_guard(job, fold, corpus, universe):
    return [{"level": "stub", "ok": True, "publication_basis": None, "n_train": len(universe), "n_test": 0}]


def _ok_inner(job, corpus):
    return lambda tr, te: {"ok": True}


from typing import Any  # noqa: E402  (after the helpers that use it lazily)


def _write_progress(out: Path, markers: list[str], jobs: dict | None = None) -> None:
    p = RL.progress_path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"markers": markers, "jobs": jobs or {}}), encoding="utf-8")


def test_runner_refuses_without_completion_marker_or_stop_rule(tmp_path, monkeypatch):
    out = tmp_path / "out"
    df = synthetic_frame(seed=1, systems=dict(list(SYSTEMS.items())[:2]), metals=METALS[:4])   # Nd(III) x TODGA is the fold
    corpus, fold, stem = _corpus(tmp_path, df)
    state = D.PlanState()
    # (a) no progress ledger, no records index -> not complete, with reasons
    comp = RL.discovery_complete(out, corpus, state, "code-T", jobs=[])
    assert not comp["complete"] and any("progress ledger" in r for r in comp["reasons"])
    assert any("records_index" in r for r in comp["reasons"])
    with pytest.raises(SystemExit, match="not complete"):
        RL.refuse_unless_ready(out, corpus, state, "code-T", jobs=[])
    # (b) ledger without the final-stage marker -> refused
    _write_progress(out, ["something else"])
    (D.discovery_root(out)).mkdir(parents=True, exist_ok=True)
    (D.discovery_root(out) / "records_index.csv").write_text("record\n", encoding="utf-8")
    comp = RL.discovery_complete(out, corpus, state, "code-T", jobs=[])
    assert not comp["complete"] and any("final stage" in r for r in comp["reasons"])
    # (c) marker present, jobs complete (none) -> complete; a job without records -> incomplete
    _write_progress(out, [D.NOT_IMPLEMENTED])
    assert RL.discovery_complete(out, corpus, state, "code-T", jobs=[])["complete"]
    job = D.JobSpec(kind="fit", arm="M2", design="V5", variant="primary", scheme="exact", seed=D.PRIMARY_SEED, stage="t")
    comp = RL.discovery_complete(out, corpus, state, "code-T", jobs=[job])
    assert not comp["complete"] and comp["incomplete_jobs"][0]["job"] == job.key and comp["n_folds_checked"] == 1
    # a job whose ledger entry carries errors is refused too
    _write_progress(out, [D.NOT_IMPLEMENTED], jobs={job.key: {"errors": [{"error": "x"}]}})
    assert any("errors" in r for r in RL.discovery_complete(out, corpus, state, "code-T", jobs=[])["reasons"])
    _write_progress(out, [D.NOT_IMPLEMENTED])
    # (d) complete but no stop-rule decision -> refused; pending stop -> refused; no ladder decisions -> refused
    with pytest.raises(SystemExit, match="stop"):
        RL.refuse_unless_ready(out, corpus, state, "code-T", jobs=[])
    RL.stop_rule_path(out).parent.mkdir(parents=True, exist_ok=True)
    RL.stop_rule_path(out).write_text(json.dumps({"stop": None}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not decided"):
        RL.refuse_unless_ready(out, corpus, state, "code-T", jobs=[])
    RL.stop_rule_path(out).write_text(json.dumps({"stop": False}), encoding="utf-8")
    with pytest.raises(SystemExit, match="ladder.kept"):
        RL.refuse_unless_ready(out, corpus, state, "code-T", jobs=[])
    RL.decisions_path(out).write_text(json.dumps({"ladder": {"M1": {"kept": True}, "M2": {"kept": None}}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="ladder.kept"):
        RL.refuse_unless_ready(out, corpus, state, "code-T", jobs=[])
    RL.decisions_path(out).write_text(json.dumps({"ladder": {"M1": {"kept": True}, "M2": {"kept": True}}}), encoding="utf-8")
    gate = RL.refuse_unless_ready(out, corpus, state, "code-T", jobs=[])
    assert gate["stop_rule"]["stop"] is False and gate["discovery_ladder"] == {"M1": True, "M2": True}
    # task X finding VL2-04: the file checks run BEFORE the corpus is loaded -- a corpus factory is not called while a
    # ledger, index or decision file is missing, and is called exactly once when they hold
    calls = []

    def factory():
        calls.append(1)
        return corpus
    RL.decision_files_ready(out)
    assert RL.refuse_unless_ready(out, factory, state, "code-T", jobs=[])["discovery_complete"]["complete"] and calls == [1]
    _write_progress(out, ["something else"])
    with pytest.raises(SystemExit, match="nothing heavy was loaded"):
        RL.refuse_unless_ready(out, factory, state, "code-T", jobs=[])
    assert calls == [1]
    _write_progress(out, [D.NOT_IMPLEMENTED])
    RL.stop_rule_path(out).write_text(json.dumps({"stop": None}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not decided"):
        RL.refuse_unless_ready(out, factory, state, "code-T", jobs=[])
    assert calls == [1]
    RL.stop_rule_path(out).write_text(json.dumps({"stop": False}), encoding="utf-8")
    assert RL.discovery_complete_cheap(out)["complete"]
    # (e) the seal gate is the discovery runner's (monkeypatched --check failing -> refused before anything)
    monkeypatch.setattr(RD, "seal_check", lambda: 3)
    with pytest.raises(SystemExit, match="refused"):
        RL.main(["--out-root", str(out), "--check-only", "--no-manifest"])


def _fake_m2_record(out: Path, job: D.JobSpec, fold: FI.Fold, code: str, state: D.PlanState, dh: str, *, kept_cfg: dict) -> str:
    m2 = replace(job, arm="M2", writes=("M2",))
    digest = RD.fold_digest(m2, fold, code, state, RD.NeuralRunner("M2"), out, ordinal=0, design_hash=dh)
    pq, js = D.fold_paths(out, m2, "M2", fold.fold_id)
    pq.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": [], "fold_id": []}).to_parquet(pq, index=False)
    js.write_text(json.dumps({"schema": D.SCHEMA, "job": m2.record(), "fold_id": fold.fold_id, "fold_hash": fold.fold_hash,
                              "digest": digest, "arm": "M2", "steps": {"point": {}, "intervals": {}},
                              "arm_record": {"selected": kept_cfg, "n_epochs": 2, "model_seed": 1}}), encoding="utf-8")
    return digest


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_m3_fold_end_to_end_on_a_synthetic_corpus(tmp_path):
    """One M3 outer fold through the runner: the M2 discovery record of the fold supplies the retained values, the
    addendum-1 inner design supplies the splits, the record lands under evaluation/ladder in the discovery schema, the
    cross-fitted intervals follow, a rerun skips, and a changed ladder decision stales it."""
    torch.set_num_threads(2)
    # 1,440 rows: 8 systems x 10 metals x 9 conditions x 2 replicates (>= 10 rows per cell for the k = 10 threshold);
    # 8 publication groups, each spanning 2 systems x 5 metals, so an inner fold never hides a whole system
    rule = lambda s_i, m_i: f"g{(s_i % 4) * 2 + (m_i // 5)}"        # noqa: E731
    df = synthetic_frame(seed=2, reps=2, pub_rule=rule)
    assert len(df) == 1440 and df[FI.GROUP_COL].nunique() == 8
    corpus, fold, stem = _corpus(tmp_path, df)
    out = tmp_path / "out"
    state = D.PlanState()
    code = "code-T"
    job = D.JobSpec(kind="fit", arm="M3", design="V5", variant="primary", scheme="exact", seed=D.PRIMARY_SEED, stage="L03_M3")
    dh = corpus.design_hash(stem)
    _fake_m2_record(out, job, fold, code, state, dh, kept_cfg={"emb_dim": 4, "weight_decay": 1e-3, "rank": 2})
    lstate = RL.LadderState(stop_rule=False, discovery_ladder={"M1": True, "M2": True})
    codes = {"own": "ladder-code", "discovery": code, "combined": "c"}
    runner = RL.LadderRunner("M3", lstate, codes, tune_kwargs={"patience": 1, "max_epochs": 2}, min_expert_rows=50)
    r = RL.run_ladder_fold(job, fold, 0, corpus, out, steps=("point", "intervals"), runner=runner, code=code, state=state,
                           guard_fn=_ok_guard, inner_check=_ok_inner)
    assert r["status"] == "fitted"
    pq, js = RL.ladder_fold_paths(out, job, "M3", fold.fold_id)
    assert pq.exists() and js.exists() and str(pq).replace("\\", "/").split("/out/")[1].startswith("evaluation/ladder/M3/V5__primary_exact/s104729/")
    rec = json.loads(js.read_text(encoding="utf-8"))
    assert rec["digest"] == RD.fold_digest(job, fold, code, state, runner, out, ordinal=0, design_hash=dh)
    assert set(rec["steps"]) == {"point", "intervals"} and rec["arm"] == "M3" and rec["ladder_code_digest"] == "ladder-code"
    ar = rec["arm_record"]
    assert ar["selected"]["experts"] is True and ar["selected"]["emb_dim"] == 4 and ar["selected"]["rank"] == 2
    assert ar["predecessor"]["step"] == "M2" and ar["predecessor"]["chain"][0]["step"] == "M2" and ar["search"] == ["experts"]
    assert ar["inner_folds_used"] == [0, 1, 2] and len(ar["fits"]) == 3
    assert rec["steps"]["intervals"]["status"] == "calibrated" and rec["steps"]["intervals"]["calibration_folds"] == [0, 1, 2]
    fr = pd.read_parquet(pq)
    assert list(fr.columns) == list(D.PREDICTION_COLUMNS) and set(fr["row_id"]) == set(fold.scored_row_ids)
    assert (fr["half"] == "S").all() and (fr["arm"] == "M3").all() and np.isfinite(fr["lower_80"]).all()
    assert (fr["intervals_status"] == "split_conformal_inner").all() and fr["std_logD"].isna().all()
    # the verified reader accepts the record set, and the frame carries the expert diagnostics
    expected = RL.expected_ladder_records(job, [fold], code=code, state=state, out_root=out, excluded_ids=[], runner=runner)
    got, st = RL.read_ladder_record_set(out, "M3", job.design_dir, D.PRIMARY_SEED, expected=expected, steps=("point", "intervals"))
    assert st["status"] == "complete" and len(got) == len(fold.scored_row_ids)
    got2, st2 = RL.verified_ladder_predictions(out, "M3", job.design_dir, D.PRIMARY_SEED, code=code, state=state, excluded_ids=[],
                                               runner=runner, folds_dir=corpus.folds_dir)
    assert st2["status"] == "complete" and len(got2) == len(got)
    # a rerun skips; a changed ladder decision (M3 kept) changes M4's digest, and a changed M2 record stales M3
    r2 = RL.run_ladder_fold(job, fold, 0, corpus, out, steps=("point", "intervals"), runner=runner, code=code, state=state,
                            guard_fn=_ok_guard, inner_check=_ok_inner)
    assert r2["status"] == "skipped_done"
    st_kept = RL.LadderState(stop_rule=False, discovery_ladder={"M1": True, "M2": True}, steps={"M3": {"status": "kept", "kept": True}})
    m4 = replace(job, arm="M4", writes=("M4",))
    d_kept = RD.fold_digest(m4, fold, code, state, RL.LadderRunner("M4", st_kept, codes), out, ordinal=0, design_hash=dh)
    d_rm = RD.fold_digest(m4, fold, code, state, RL.LadderRunner("M4", lstate, codes), out, ordinal=0, design_hash=dh)
    assert d_kept != d_rm
    _fake_m2_record(out, job, fold, "code-OTHER", state, dh, kept_cfg={"emb_dim": 4, "weight_decay": 1e-3, "rank": 2})
    with pytest.raises(D.StaleRecordError):
        runner.predecessor_config(RD.prepare_fold(job, fold, 0, corpus, out, state, _ok_guard, _ok_inner, code=code)[0])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_m7_fold_end_to_end_with_normalised_conformal_intervals(tmp_path):
    """M7 through the runner on the retained configuration of the fold (M2 kept, M3-M6 removed): the ensemble members,
    the std_logD column, the section 12 normalised split-conformal intervals from the cross-fitted inner folds."""
    torch.set_num_threads(2)
    rule = lambda s_i, m_i: f"g{(s_i % 4) * 2 + (m_i // 5)}"        # noqa: E731
    df = synthetic_frame(seed=4, reps=2, pub_rule=rule)
    corpus, fold, stem = _corpus(tmp_path, df)
    out = tmp_path / "out"
    state, code = D.PlanState(), "code-T"
    job = D.JobSpec(kind="fit", arm="M7", design="V5", variant="primary", scheme="exact", seed=D.PRIMARY_SEED, stage="L07_M7")
    dh = corpus.design_hash(stem)
    _fake_m2_record(out, job, fold, code, state, dh, kept_cfg={"emb_dim": 4, "weight_decay": 1e-3, "rank": 2})
    lstate = RL.LadderState(stop_rule=True, discovery_ladder={"M1": True, "M2": True},
                            steps={s: {"status": "exploratory_not_run", "kept": None} for s in ("M3", "M4", "M5", "M6")})
    codes = {"own": "ladder-code", "discovery": code, "combined": "c"}
    runner = RL.LadderRunner("M7", lstate, codes, n_members=2)
    assert runner.chain_steps() == ["M2"] and lstate.retained_before("M7") == "M2"
    r = RL.run_ladder_fold(job, fold, 0, corpus, out, steps=("point", "intervals"), runner=runner, code=code, state=state,
                           guard_fn=_ok_guard, inner_check=_ok_inner)
    assert r["status"] == "fitted"
    pq, js = RL.ladder_fold_paths(out, job, "M7", fold.fold_id)
    rec = json.loads(js.read_text(encoding="utf-8"))
    ar = rec["arm_record"]
    assert ar["selected"]["heteroscedastic"] is True and ar["selected"]["rank"] == 2 and ar["n_epochs"] == 2
    assert ar["member_seeds"] == [NN.registered_model_seed(0), NN.registered_model_seed(0) + 1] and ar["search"] == []
    assert rec["steps"]["intervals"]["method"].startswith("models.ladder.NormalisedCrossFitConformal")
    assert rec["steps"]["intervals"]["score"].startswith("|y - mean_logD| / std_logD") and rec["steps"]["intervals"]["n_calibration"] > 0
    fr = pd.read_parquet(pq)
    assert list(fr.columns) == list(D.PREDICTION_COLUMNS) and (fr["intervals_status"] == "split_conformal_inner_normalised").all()
    sd = fr["std_logD"].to_numpy(dtype=float)
    assert np.isfinite(sd).all() and (sd > 0).all()
    for pct in (50, 80, 95):
        half = (fr[f"upper_{pct}"] - fr[f"lower_{pct}"]).to_numpy() / 2
        assert np.allclose(half, fr[f"conformal_q{pct}"].to_numpy() * sd)
    assert (fr["conformal_q50"] <= fr["conformal_q80"]).all() and (fr["conformal_q80"] <= fr["conformal_q95"]).all()
    assert (fr["arm"] == "M7").all() and (fr["half"] == "S").all()


class _FakeStore:
    """The subset of the scorer's Store the ladder decision reads: frames per (arm, design, setting, seed)."""

    def __init__(self, frames: dict):
        self.frames, self.state = frames, D.PlanState()
        idx = pd.Index(sorted({r for f in frames.values() for r in f.index}))
        self.v6 = pd.Series(False, index=idx)
        self.remainder_groups: list = []

    def frame(self, arm, design, setting, seed):
        return self.frames.get((arm, design, setting))

    def v1_scheme(self, arm, design):
        return "exact"


def _score_frame(design: str, rng, noise: float, systems=6, metals=("La(III)", "Ce(III)", "Pr(III)", "Nd(III)"), n=5) -> pd.DataFrame:
    recs = []
    for s in range(systems):
        for m in metals:
            for k in range(n):
                y = float(rng.normal())
                grp = f"g{s % 3}"
                recs.append({"row_id": f"{design}_{s}_{m}_{k}", "fold_id": grp if design == "V1" else f"f{s}",
                             EM_SYSTEM: f"S{s}", EM_STATE: m, EM_PUB: grp, "log_D": y, "pred": y + noise * rng.normal(),
                             "dga_stratum": "non_DGA" if s % 2 else "diglycolamide", "acid_grid_flag": k == 4,
                             "censoring_candidate": False, "wildcard_copy_partner_in_training": False})
    fr = pd.DataFrame(recs).set_index("row_id", drop=False)
    fr.index.name = None
    fr["mean_logD"] = fr["pred"]
    return fr


from gen19ct.evaluation import metrics as EM  # noqa: E402
EM_SYSTEM, EM_STATE, EM_PUB = EM.SYSTEM_COL, EM.METAL_STATE_COL, EM.PUB_GROUP_COL


def test_decide_step_uses_the_scorer_r19_machinery_on_synthetic_frames():
    rng = np.random.default_rng(5)
    frames = {}
    for design in ("V5", "V1", "V2"):
        base = _score_frame(design, rng, 0.0)
        # M2: noisy (MAE ~ 0.4); M3: sharp (MAE ~ 0.08): a clear strict-fold benefit on every design
        m2 = base.copy()
        m2["pred"] = base["log_D"] + 0.5 * rng.normal(size=len(base))
        m2["mean_logD"] = m2["pred"]
        m3 = base.copy()
        m3["pred"] = base["log_D"] + 0.1 * rng.normal(size=len(base))
        m3["mean_logD"] = m3["pred"]
        frames[("M2", design, "primary")] = m2
        frames[("M3", design, "primary")] = m3
    store = _FakeStore(frames)
    lstate = RL.LadderState(stop_rule=False, discovery_ladder={"M1": True, "M2": True})
    dec, rows = RL.decide_step("M3", lstate, store, delta5=0.1)
    assert dec["step"] == "M3" and dec["predecessor"] == "M2" and dec["kept"] is True
    assert dec["v5_ladder_scope"]["verdict"] == "PASS" and dec["v1_tost"]["non_inferior"] and dec["v2_tost"]["non_inferior"]
    assert set(dec["contrast_summary"]) == {"V5", "V1", "V2"} and dec["contrast_summary"]["V5"]["point"] > 0.1
    assert dec["missing_designs"] == []
    fams = {r["key"].split("#")[1] for r in rows}
    assert fams == {"ladder", "H4", "H5"}                      # the H4 / H5 rows the same numbers answer (section 19)
    assert all(r["r19_item4"] == D.ITEM4_NOT_EVALUATED for r in rows)          # addendum 1 item 3: learned arms
    assert all(r["sensitivity_set"] == D.ADDENDUM_LABEL for r in rows)
    # a step without a strict-fold benefit is removed, and the chain then skips it
    worse = dict(frames)
    for design in ("V5", "V1", "V2"):
        w = frames[("M2", design, "primary")].copy()
        w["pred"] = w["log_D"] + 0.8 * rng.normal(size=len(w))
        w["mean_logD"] = w["pred"]
        worse[("M3", design, "primary")] = w
    dec2, _ = RL.decide_step("M3", lstate, _FakeStore(worse), delta5=0.1)
    assert dec2["kept"] is False
    lstate.steps["M3"] = {"status": "removed", "kept": False}
    assert lstate.retained_before("M4") == "M2"
    # a missing design leaves the decision pending (never a pass)
    partial = {k: v for k, v in frames.items() if k[1] != "V2"}
    dec3, _ = RL.decide_step("M3", lstate, _FakeStore(partial), delta5=0.1)
    assert dec3["kept"] is None and dec3["missing_designs"] == ["V2"]


# --------------------------------------------------------------------------------------------- task X findings V-04, V-05, VL2-07

def _m2_m3_frames(rng, designs=("V5", "V1", "V2"), *, settings=("primary",)):
    frames = {}
    for design in designs:
        base = _score_frame(design, rng, 0.0)
        for setting in settings:
            m2 = base.copy()
            m2["pred"] = base["log_D"] + 0.5 * rng.normal(size=len(base))
            m2["mean_logD"] = m2["pred"]
            m3 = base.copy()
            m3["pred"] = base["log_D"] + 0.1 * rng.normal(size=len(base))
            m3["mean_logD"] = m3["pred"]
            frames[("M2", design, setting)] = m2
            frames[("M3", design, setting)] = m3
    return frames


def test_undecided_step_stops_the_ladder_instead_of_being_skipped(tmp_path, monkeypatch):
    """Task X finding V-05: a step whose keep decision is undecidable (a design frame missing) stops the ladder with
    status ``undecided``; the chain never builds past it as if it had been removed."""
    out = tmp_path / "out"
    rng = np.random.default_rng(5)
    frames = _m2_m3_frames(rng, designs=("V5", "V1"))                     # no V2 frame: M3 cannot be decided
    store = _FakeStore(frames)
    monkeypatch.setattr(RL, "run_step_jobs", lambda jobs, *a, **k: {"jobs": {j.key: {"complete": True} for j in jobs},
                                                                    "stopped": None})
    gate = {"stop_rule": {"stop": False}, "discovery_ladder": {"M1": True, "M2": True}}
    corpus = SimpleNamespace(fittable=lambda job: [])
    codes = {"own": "o", "discovery": "d", "combined": "c"}
    ledger = RL.run_ladder(corpus, out, _plan_state(), codes, gate, steps=("point", "intervals"), workers=1,
                           store_factory=lambda ls: store, delta5=0.1, with_h5=False)
    assert ledger["stopped"].startswith("M3: keep decision undecided") and "['V2']" in ledger["stopped"]
    lstate = RL.LadderState.read(RL.ladder_state_path(out))
    assert lstate.steps["M3"]["status"] == "undecided" and lstate.steps["M3"]["kept"] is None
    assert "M4" not in lstate.steps and [x["step"] for x in ledger["steps"]] == ["M3"]
    assert "undecided" not in RL.STATUS_DONE + RL.STATUS_SKIPPED            # a rerun re-evaluates it
    assert lstate.retained_before("M4") == "M2"
    # with every design present the same ladder keeps M3 and moves on
    store2 = _FakeStore(_m2_m3_frames(np.random.default_rng(5)))
    out2 = tmp_path / "out2"
    monkeypatch.setattr(RL, "component_activity", lambda *a, **k: {"hinge": {"n_configured": 0, "n_active": 0},
                                                                   "smoothness": {"n_configured": 0, "n_active": 0}})
    ledger2 = RL.run_ladder(corpus, out2, _plan_state(), codes, gate, steps=("point", "intervals"), workers=1,
                            store_factory=lambda ls: store2, delta5=0.1, with_h5=False, only="M3")
    assert ledger2["stopped"] is None and ledger2["steps"] == [{"step": "M3", "kept": True, "status": "kept"}]


def test_m3_contrasts_are_reevaluated_after_the_refits(tmp_path):
    """Task X finding V-04: the H4 / H5 rows of M3 vs M2 are re-evaluated once the strict / HNO3-only refit frames
    exist, contrasts_M3.csv is rewritten (the earlier rows kept) and the keep decision is unchanged."""
    out = tmp_path / "out"
    rng = np.random.default_rng(6)
    frames = _m2_m3_frames(rng)
    store = _FakeStore(frames)
    lstate = RL.LadderState(stop_rule=False, discovery_ladder={"M1": True, "M2": True})
    dec, rows = RL.decide_step("M3", lstate, store, delta5=0.1)
    assert dec["kept"] is True
    dec_dir = RL.ladder_root(out) / "decisions"
    dec_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dec_dir / "contrasts_M3.csv", index=False)
    lstate.steps["M3"] = {"status": "kept", "kept": True, "decision": dec}
    before = pd.read_csv(dec_dir / "contrasts_M3.csv")
    h4 = before[before["key"].str.endswith("#H4")]
    assert (h4["sensitivity_set"] == D.ADDENDUM_LABEL).all()
    # before the refits the strict / HNO3-only items of R19 item 6 are UNTESTABLE (the frames are absent)
    strict_name, hno3_name = D.V5_SETTING_SENSITIVITY["strict"], D.V5_SETTING_SENSITIVITY["hno3_only"]
    sens0 = [json.loads(x) for x in h4["sensitivities"]]
    assert all(s0[strict_name] == ET.UNTESTABLE and s0[hno3_name] == ET.UNTESTABLE for s0 in sens0)
    # the refit frames appear (the H5 block ran M3's strict / HNO3-only refits)
    with_refits = dict(frames)
    with_refits.update(_m2_m3_frames(np.random.default_rng(6), designs=("V5",), settings=("strict", "hno3_only")))
    res = RL.reevaluate_step_after_refits("M3", lstate, _FakeStore(with_refits), 0.1, out)
    assert res["evaluated"] and res["ladder_scope_unchanged"] and res["n_rows"] == len(rows)
    assert set(res["refit_sensitivities_now_run"]) == {strict_name, hno3_name}
    assert (dec_dir / "contrasts_M3_before_refits.csv").exists()
    after = pd.read_csv(dec_dir / "contrasts_M3.csv")
    h4b = after[after["key"].str.endswith("#H4")]
    assert (h4b["refit_sensitivities_evaluated"] == True).all()                              # noqa: E712
    sens1 = [json.loads(x) for x in h4b["sensitivities"]]
    assert all(isinstance(s1[strict_name], float) and isinstance(s1[hno3_name], float) for s1 in sens1)
    assert (h4b["verdict_ladder"].to_numpy() == h4["verdict_ladder"].to_numpy()).all()
    assert (h4b["sensitivities_not_run"].to_numpy() == h4["sensitivities_not_run"].to_numpy()).all()   # the dropped set is unchanged


def test_component_activity_and_vacuity_labels(tmp_path):
    """Task X finding VL2-07: the M6 terms record whether they acted; a contrast whose term acted on no fold is vacuous."""
    out = tmp_path / "out"
    job = D.JobSpec(kind="fit", arm="M6", design="V5", variant="primary", scheme="batched", seed=D.PRIMARY_SEED,
                    fold_seed=D.PRIMARY_SEED, writes=("M6",), stage="L06_M6")
    folds = [SimpleNamespace(fold_id=f"f{i}") for i in range(3)]
    corpus = SimpleNamespace(fittable=lambda j: [(f, i) for i, f in enumerate(folds)])
    for i, f in enumerate(folds):
        pq, js = RL.ladder_fold_paths(out, job, "M6", f.fold_id)
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps({"arm_record": {"fit_record": {"hinge_active": False, "smoothness_active": i == 1}}}),
                      encoding="utf-8")
    act = RL.component_activity(out, "M6", [job], corpus)
    assert act["n_records"] == 3 and act["hinge"] == {"n_configured": 3, "n_active": 0, "vacuous": True}
    assert act["smoothness"] == {"n_configured": 3, "n_active": 1, "vacuous": False}
    v = RL.vacuity_of(act, terms=("hinge",))
    assert v["vacuous"] and "hinge inactive on every fold" in v["reason"]
    assert not RL.vacuity_of(act, terms=("hinge", "smoothness"))["vacuous"]
    assert not RL.vacuity_of(act, terms=("smoothness",))["vacuous"]
    none = RL.component_activity(out, "M6", [job], SimpleNamespace(fittable=lambda j: [(SimpleNamespace(fold_id="zz"), 0)]))
    assert none["n_records_missing"] == 1 and not none["hinge"]["vacuous"]         # absent is not vacuous
    # a record without the flags (a term switched off) configures nothing
    js.write_text(json.dumps({"arm_record": {"fit_record": {"hinge_active": None, "smoothness_active": None}}}), encoding="utf-8")
    assert RL.component_activity(out, "M6", [job], corpus)["hinge"]["n_configured"] == 2


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fit_record_flags_an_inactive_hinge(frame):
    """Task X finding VL2-07: with one extractant concentration per (system, acid) unit the hinge set is empty and the
    fit says so loudly; on the full grid it is active."""
    torch.set_num_threads(2)
    cfg = _cfg("M6", lambda_phys=0.1, phys_terms="a")
    flat = synthetic_frame(seed=1, exts=(0.1,))                          # one concentration: no identifiable slope
    arm = LAD.LadderArm(cfg, n_epochs=1, model_seed=3, rows=flat).fit(flat)
    rec = arm.fit_record()
    assert rec["hinge_active"] is False and rec["smoothness_active"] is None
    assert rec["aux"]["hinge"]["active"] is False and rec["aux"]["hinge"]["n_hinge_rows"] == 0
    assert rec["loss_terms_last_batch"].get("hinge_inactive") == 1.0 and "L_hinge" not in rec["loss_terms_last_batch"]
    assert arm.component_activity() == {"hinge_active": False, "smoothness_active": None}
    hidden = frame.index[(frame[SG.METAL_COL] == "Eu(III)") & (frame[SG.SYSTEM_COL] == list(SYSTEMS)[0])]
    full = LAD.LadderArm(_cfg("M6", lambda_phys=0.1, phys_terms="ab"), n_epochs=1, model_seed=3, rows=frame).fit(frame.drop(index=hidden))
    rec2 = full.fit_record()
    assert rec2["hinge_active"] is True and rec2["smoothness_active"] is True and "L_hinge" in rec2["loss_terms_last_batch"]
    assert LAD.LadderArm(_cfg("M2"), n_epochs=1, model_seed=3, rows=frame).fit(frame.drop(index=hidden)).component_activity() == \
        {"hinge_active": None, "smoothness_active": None}


# ---- POST-HOC addendum 2, "2. Ladder M3-M7" > "Budget": the ladder's own 40 h ledger -------------------------------- #

def test_ladder_budget_is_its_own_40h_ledger_and_the_discovery_clock_does_not_count(tmp_path):
    """Addendum 2: 'the ladder has its own budget of 40 h of wall clock (evaluation/ladder/decisions/wall_clock.json),
    checked before each step; the demotion order M7 -> M6 -> M5 -> M4 -> M3 and every other rule of section 7 items 3-6
    are unchanged; the discovery ledger stays as registered for the discovery stages.'"""
    out = tmp_path / "out"
    assert RL.LADDER_BUDGET_HOURS == 40.0 and RL.BUDGET_HOURS == 40.0 and RL.DISCOVERY_BUDGET_HOURS == D.BUDGET_HOURS == 60.0
    assert RL.DEMOTION_ORDER == D.DEMOTION_ORDER == ("M7", "M6", "M5", "M4", "M3")
    assert RL.demote_from("M3") == ["M7", "M6", "M5", "M4", "M3"] and RL.demote_from("M7") == ["M7"]
    assert "addendum 2" in RL.READINGS["budget"] and "40 h" in RL.READINGS["budget"] and "never added" in RL.READINGS["budget"]
    # a discovery ledger of 57 h (the two registered fallbacks fired) counts for nothing in the ladder budget
    RD.record_wall_clock(out, "2026-09-16T00:00:00+00:00", 57 * 3600.0, ["01_B6", "05_M2"])
    assert RD.wall_clock_total(out) == pytest.approx(57 * 3600.0)
    assert RL.budget_used_seconds(out) == 0.0 and RL.ladder_wall_clock_total(out) == 0.0
    RL.record_ladder_wall_clock(out, "2026-09-20T00:00:00+00:00", 3600.0, ["M3"])
    body = json.loads(RL.ladder_wall_clock_path(out).read_text(encoding="utf-8"))
    assert body["ladder_hours"] == 1.0 and body["discovery_hours_not_counted"] == 57.0 and body["budget_hours"] == 40.0
    b = body["budget"]
    assert b["budget_hours"] == 40.0 and b["used_hours"] == 1.0 and b["remaining_hours"] == 39.0 and not b["exhausted"]
    assert b["discovery_hours_not_counted"] == 57.0 and b["discovery_budget_hours_unchanged"] == 60.0
    assert b["demotion_order"] == ["M7", "M6", "M5", "M4", "M3"] and "addendum 2" in b["note"] and "addendum 2" in body["reading"]
    assert RL.budget_used_seconds(out) == pytest.approx(3600.0)
    # the same invocation re-recorded replaces its entry; another invocation adds
    RL.record_ladder_wall_clock(out, "2026-09-20T00:00:00+00:00", 7200.0, ["M3", "M4"])
    assert RL.budget_used_seconds(out) == pytest.approx(7200.0)
    RL.record_ladder_wall_clock(out, "2026-09-21T00:00:00+00:00", 38 * 3600.0, ["M5"])
    assert RL.budget_used_seconds(out) == pytest.approx(40 * 3600.0)
    body = json.loads(RL.ladder_wall_clock_path(out).read_text(encoding="utf-8"))
    assert body["budget"]["exhausted"] and body["budget"]["used_hours"] == 40.0 and body["budget"]["demoted_now"] == ["M7", "M6", "M5", "M4", "M3"]
    # the arithmetic in one place: 39.99 h is not exhausted, 40 h is; the discovery hours never enter
    assert not RL.ladder_budget_status(39.99 * 3600.0, 57 * 3600.0)["exhausted"]
    assert RL.ladder_budget_status(40 * 3600.0)["exhausted"] and RL.ladder_budget_status(40 * 3600.0)["discovery_hours_not_counted"] is None
    assert RL.ladder_budget_status(0.0, 59.9 * 3600.0)["remaining_hours"] == 40.0


def test_run_ladder_checks_its_own_ledger_before_each_step_and_ignores_the_discovery_clock(tmp_path, monkeypatch):
    """A 57 h discovery run does not demote M3 (addendum 2); a 40 h ladder ledger demotes M3-M7 before the first step in
    the registered order with the addendum-2 note."""
    monkeypatch.setattr(RL, "run_step_jobs", lambda jobs, *a, **k: {"jobs": {j.key: {"complete": True} for j in jobs},
                                                                    "stopped": None})
    monkeypatch.setattr(RL, "component_activity", lambda *a, **k: {"hinge": {"n_configured": 0, "n_active": 0},
                                                                   "smoothness": {"n_configured": 0, "n_active": 0}})
    gate = {"stop_rule": {"stop": False}, "discovery_ladder": {"M1": True, "M2": True}}
    corpus = SimpleNamespace(fittable=lambda job: [])
    codes = {"own": "o", "discovery": "d", "combined": "c"}
    out = tmp_path / "out"
    RD.record_wall_clock(out, "2026-09-16T00:00:00+00:00", 57 * 3600.0, ["01_B6"])            # 57 h of discovery
    store = _FakeStore(_m2_m3_frames(np.random.default_rng(5)))
    ledger = RL.run_ladder(corpus, out, _plan_state(), codes, gate, steps=("point", "intervals"), workers=1,
                           store_factory=lambda ls: store, delta5=0.1, with_h5=False, only="M3")
    assert ledger["stopped"] is None and ledger["budget"] is None
    assert ledger["steps"] == [{"step": "M3", "kept": True, "status": "kept"}]
    lstate = RL.LadderState.read(RL.ladder_state_path(out))
    assert lstate.demoted == [] and lstate.steps["M3"]["status"] == "kept"
    wc = json.loads(RL.ladder_wall_clock_path(out).read_text(encoding="utf-8"))
    assert wc["discovery_hours_not_counted"] == 57.0 and wc["budget"]["budget_hours"] == 40.0 and not wc["budget"]["exhausted"]
    # the ladder's own ledger at 40 h: everything not yet run is demoted before the first step
    out2 = tmp_path / "out2"
    RL.record_ladder_wall_clock(out2, "2026-09-19T00:00:00+00:00", 40 * 3600.0, [])
    ledger2 = RL.run_ladder(corpus, out2, _plan_state(), codes, gate, steps=("point", "intervals"), workers=1,
                            store_factory=lambda ls: store, delta5=0.1, with_h5=False)
    assert ledger2["budget"]["exhausted_before"] == "M3" and ledger2["budget"]["demoted"] == ["M7", "M6", "M5", "M4", "M3"]
    assert ledger2["budget"]["budget_hours"] == 40.0 and ledger2["budget"]["used_hours"] >= 40.0 and ledger2["steps"] == []
    lstate2 = RL.LadderState.read(RL.ladder_state_path(out2))
    assert lstate2.demoted == ["M7", "M6", "M5", "M4", "M3"]
    for s in ("M3", "M4", "M5", "M6", "M7"):
        assert lstate2.steps[s]["status"] == "demoted" and "addendum 2" in lstate2.steps[s]["note"] and "40 h" in lstate2.steps[s]["note"]
    assert "demoted" in RL.STATUS_SKIPPED

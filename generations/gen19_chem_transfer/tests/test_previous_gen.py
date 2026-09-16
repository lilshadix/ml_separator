"""B8 -- the previous best Gen-series models re-fitted inside Gen19 folds (``gen19ct/models/previous_gen.py``).

Build-and-unit-test only.  The level and direction arms are fitted on SYNTHETIC rows (real bundle structures as
system keys, invented targets).  Real MODEL rows enter two slow tests only: the TOPO39 / Ln(III) coverage count, which
fits no learned model, and a plumbing check that fits a 20-tree forest on the rows of an ad-hoc set of systems and
predicts rows of other systems without reading their target -- never a registered fold, never a score.  Nothing
touches V6 and no test writes a file.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.chemistry import metals as MET
from gen19ct.chemistry import support_graph as SG
from gen19ct.data import load as L
from gen19ct.evaluation import pairs as EP
from gen19ct.models import features as F
from gen19ct.models import interface as I
from gen19ct.models import previous_gen as P

LN = P.BASIS_METALS
TODGA = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"
TBP = "CCCCOP(=O)(OCCCC)OCCCC"
#: a structure absent from every descriptor table and from TOPO39
NOVEL = "CCCCCCN(CCCCCC)C(=O)CCC(=O)N(CCCCCC)CCCCCC"
GEN13 = paths.REPO_ROOT / "generations" / "gen13_separation"
GEN14 = paths.REPO_ROOT / "generations" / "gen14_direction"
FAST = P.MonoEtConfig(n_estimators=25, n_jobs=1)


def _import_path(p: Path) -> None:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def topo_systems(n: int, skip: int = 0) -> list[str]:
    """``n`` bundle structures with pairwise distinct TOPO39 vectors (sorted canonical SMILES)."""
    t = F.topo39_table()
    seen, out = set(), []
    for smi in sorted(t.index):
        key = tuple(t.loc[smi].to_numpy(dtype=float))
        if key in seen:
            continue
        seen.add(key)
        out.append(smi)
    return out[skip:skip + n]


# ------------------------------------------------------------------------------------------------ synthetic rows

def synth_rows(systems: dict[str, float], *, pubs=("g1", "g2"), n_cond: int = 2, states=None, seed: int = 0,
               prefix: str = "r", extra_states=("Am(III)",), noise: float = 0.02) -> pd.DataFrame:
    """Arm-frame-like rows: per (system, publication group, condition) one row per metal state; the Ln(III) curve of
    a system is ``amp * radius_basis[0]`` on a random level (heavy-selective when ``amp < 0``)."""
    rng = np.random.default_rng(seed)
    basis = P.radius_basis()
    states = [f"{m}(III)" for m in LN] if states is None else list(states)
    recs = []
    for s, amp in systems.items():
        for g in pubs:
            for c in range(n_cond):
                level = float(rng.normal(0.0, 1.0))
                acid = float(rng.uniform(-1.0, 0.7))
                ext = float(rng.uniform(-2.0, -0.5))
                for st in list(states) + list(extra_states):
                    el = st.split("(")[0]
                    y = level + 0.8 * acid
                    if st.endswith("(III)") and el in LN:
                        y += amp * basis[0, LN.index(el)]
                    y += noise * float(rng.normal())
                    recs.append({I.ID_COL: f"{prefix}{len(recs):05d}", SG.METAL_COL: st, SG.ELEMENT_COL: el,
                                 SG.SYSTEM_COL: s, I.TARGET_COL: y, I.PUB_GROUP_COL: g, SG.PUB_COL: f"pub_{g}",
                                 SG.ACID_ANION_COL: "nitrate", SG.LOG_ACID_COL: acid, SG.LOG_EXT_COL: ext,
                                 SG.SMILES_COL: s, SG.FAMILY_COL: "synthetic", SG.MECH_COL: "NEUTRAL_SOLVATING",
                                 "condition_key": f"ck_{s[:12]}_{g}_{c}"})
    df = pd.DataFrame(recs)
    df.index = pd.Index([f"{prefix}{i:05d}" for i in range(len(df))], name="row")
    return df


def synth_cv(frame: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """A condition vector with the columns the B8 condition and mass-action blocks read."""
    rng = np.random.default_rng(seed)
    n = len(frame)
    metal = rng.normal(-3.0, 0.5, n)
    metal[rng.random(n) < 0.2] = np.nan                               # partially missing -> indicator
    temp = np.where(rng.random(n) < 0.3, np.nan, 25.0)
    obj = np.empty(n, dtype=object)
    for i in range(n):
        obj[i] = ()
    ext = np.empty(n, dtype=object)
    for i, v in enumerate(frame[SG.LOG_EXT_COL].to_numpy(dtype=float)):
        ext[i] = (10.0 ** v,)
    return pd.DataFrame({
        "log10_acid_M": frame[SG.LOG_ACID_COL].to_numpy(dtype=float),
        "log10_extractant_primary_M": frame[SG.LOG_EXT_COL].to_numpy(dtype=float),
        "log10_metal_M": metal, "phase_ratio_org_aq": 1.0, "temperature_C": temp,
        "contact_time_min": np.nan,                                   # all-NaN in training -> dropped (gen6)
        "modifier_name": pd.Series([None if i % 3 else "1-octanol" for i in range(n)], dtype=object).to_numpy(),
        "modifier_concentration_M": np.where(np.arange(n) % 3, np.nan, 0.5),
        "complexant_structure_key": pd.Series([None] * n, dtype=object).to_numpy(),
        "complexant_concentrations_sorted_M": obj,
        "acid_M_log10_grid": np.zeros(n, dtype=bool), "acid_anion": "nitrate",
        "diluent_family": np.where(np.arange(n) % 2, "alkane", "aromatic"),
        "extractant_concentrations_sorted_M": ext,
    }, index=frame.index)


@pytest.fixture(scope="module")
def level_data():
    heavy = topo_systems(6)
    systems = {s: (-0.4 if i % 2 else 0.3) for i, s in enumerate(heavy)}
    systems[TODGA] = -0.5
    fr = synth_rows(systems, seed=3)
    cv = synth_cv(fr)
    train_sys = sorted(systems)[:5]
    tr = fr[fr[SG.SYSTEM_COL].isin(train_sys)]
    qu = fr[~fr[SG.SYSTEM_COL].isin(train_sys)]
    return fr, cv, tr, qu


# ------------------------------------------------------------------------------------------------ seeds and range

def test_model_seed_rule_and_fold_index():
    assert P.model_seed(0) == 42 + 9_999_991
    assert P.model_seed(7) == 42 + 7 * 1009 + 9_999_991
    _import_path(GEN13)
    from gen13sep import splits as S13                               # the programme's formula, read-only

    assert (S13.MODEL_SEED_BASE, S13.FOLD_SEED_STRIDE, S13.FOLD_SEED_OFFSET) == \
        (P.MODEL_SEED_BASE, P.FOLD_SEED_STRIDE, P.FOLD_SEED_OFFSET)
    assert P.fold_index("s104729_S_b007") == 7 and P.fold_index("s130363_f3") == 3
    with pytest.raises(ValueError):
        P.fold_index("no_index_here")
    with pytest.raises(ValueError):
        P.model_seed(-1)
    with pytest.raises(ValueError):
        P.B8Level()                                                   # neither fold nor random_state
    assert P.B8Level(fold=4).random_state == P.model_seed(4)
    assert P.B8Level(fold=4, random_state=11).random_state == 11


def test_clip_range_is_the_gen6_rule():
    from lanthanide_separation.levels import LevelForestParameters, LevelRegressor

    rng = np.random.default_rng(5)
    for _ in range(5):
        y = rng.normal(0, 2, 40)
        lo, hi = P.training_clip_range(y)
        span = y.max() - y.min()
        assert (lo, hi) == (y.min() - 0.5 * span, y.max() + 0.5 * span)
        frame = pd.DataFrame({"a": rng.normal(size=40)})
        ref = LevelRegressor(["a"], LevelForestParameters(n_estimators=3, n_jobs=1)).fit(frame, y)
        assert ref._y_range == (lo, hi)
    m = P.MonoEt(0, FAST).fit(pd.DataFrame({"a": np.arange(10.0)}), np.arange(10.0))
    assert m.y_range == (-4.5, 13.5)
    assert np.array_equal(m.clip([-100.0, 3.0, 100.0]), [-4.5, 3.0, 13.5])
    with pytest.raises(ValueError):
        P.training_clip_range([1.0, np.nan])


# ------------------------------------------------------------------------------------------------ gen6 front + forest

def _gen6_frame(seed: int, n: int = 160):
    rng = np.random.default_rng(seed)
    tab = pd.DataFrame({"t0": rng.normal(size=n), "t1": rng.normal(size=n), "t2": rng.normal(size=n),
                        "t_allnan": np.nan, "t4": rng.integers(0, 3, n).astype(float)})
    tab.loc[rng.random(n) < 0.25, "t1"] = np.nan
    tab.loc[rng.random(n) < 0.10, "t4"] = np.nan
    bits = (rng.random((n, 12)) < 0.3).astype(np.uint8)
    groups = rng.integers(0, 6, n).astype(str)
    y = tab["t0"].fillna(0) * 2 + bits[:, 0] - bits[:, 3] + rng.normal(0, 0.1, n)
    return tab, bits, groups, y.to_numpy(dtype=float)


def test_missing_prep_equals_the_gen6_front():
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from lanthanide_separation.levels import DropAllNaNColumns

    tab, _, _, _ = _gen6_frame(1)
    tr, te = tab.iloc[:100], tab.iloc[100:].copy()
    te.loc[te.index[:5], "t0"] = np.nan                               # missingness new at transform
    prep = P.MissingPrep().fit(tr)
    ref = Pipeline([("drop_empty", DropAllNaNColumns()),
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True))]).fit(tr.to_numpy(dtype=float))
    for part in (tr, te):
        assert np.array_equal(prep.transform(part, dtype=np.float64), ref.transform(part.to_numpy(dtype=float)))
    assert prep.dropped_all_nan == ("t_allnan",)
    assert prep.indicator_for == ("t1", "t4")
    assert prep.output_columns == ("t0", "t1", "t2", "t4", "t1__missing", "t4__missing")
    perm = tr.iloc[np.random.default_rng(2).permutation(len(tr))]
    assert P.MissingPrep().fit(perm).state() == prep.state()          # row-order free
    with pytest.raises(ValueError):
        P.MissingPrep().fit(pd.DataFrame({"a": [np.nan, np.nan]}))


def test_mono_et_reproduces_the_gen6_level_regressor_bit_for_bit():
    from lanthanide_separation.levels import LevelForestParameters, LevelRegressor, group_balanced_weights

    tab, bits, groups, y = _gen6_frame(4)
    wide_cols = [f"b{j}" for j in range(bits.shape[1])]
    frame = pd.concat([tab, pd.DataFrame(bits.astype(float), columns=wide_cols, index=tab.index)], axis=1)
    tr, te = slice(0, 120), slice(120, None)
    seed = P.model_seed(2)
    cfg = P.MonoEtConfig(n_estimators=60, n_jobs=1)
    w = group_balanced_weights(groups[tr])
    mine = P.MonoEt(seed, cfg).fit(tab.iloc[tr], y[tr], wide=bits[tr], sample_weight=w)
    ref = LevelRegressor(list(frame.columns), LevelForestParameters(n_estimators=60, random_state=seed, n_jobs=1))
    ref.fit(frame.iloc[tr], y[tr], groups=groups[tr])
    assert np.array_equal(mine.predict(tab.iloc[te], bits[te]), ref.predict(frame.iloc[te]))
    assert mine.y_range == ref._y_range
    assert mine.design_columns(wide_cols) == tuple(["t0", "t1", "t2", "t4"] + wide_cols + ["t1__missing", "t4__missing"])
    # reproducible: same seed -> identical forest output, also when fitted on 2 threads (and twice); another seed differs
    for _ in range(2):
        again = P.MonoEt(seed, P.MonoEtConfig(n_estimators=60, n_jobs=2)).fit(tab.iloc[tr], y[tr], wide=bits[tr],
                                                                              sample_weight=w)
        assert np.array_equal(again.predict_raw(tab.iloc[te], bits[te]), mine.predict_raw(tab.iloc[te], bits[te]))
        assert again.forest.n_jobs == 2                                    # restored after the one-thread sum
    other = P.MonoEt(seed + 1, cfg).fit(tab.iloc[tr], y[tr], wide=bits[tr], sample_weight=w)
    assert not np.array_equal(other.predict_raw(tab.iloc[te], bits[te]), mine.predict_raw(tab.iloc[te], bits[te]))
    # the registered configuration is the default one
    assert P.MonoEtConfig().registered and not cfg.registered
    f = P.MonoEt(seed).fit(tab.iloc[tr], y[tr], wide=bits[tr], sample_weight=w).forest
    assert (f.n_estimators, f.max_features, f.min_samples_leaf, f.n_jobs) == (400, 0.30, 2, 2)


# ------------------------------------------------------------------------------------------------ B8 level arm

def test_ecfp_cluster_weights_balance_primary_extractant_clusters(level_data):
    from lanthanide_separation.levels import group_balanced_weights

    fr, cv, tr, _ = level_data
    mix = synth_rows({f"{TBP}|{TODGA}": 0.1}, seed=9, prefix="m")     # primary extractant TODGA (row column)
    mix[SG.SMILES_COL] = TODGA
    rows = pd.concat([tr, mix])
    cl = F.EcfpClusters().fit(rows)
    labels = cl.labels(rows)
    w = cl.sample_weights(rows.index)
    assert np.allclose(pd.Series(w, index=rows.index).groupby(labels).sum().to_numpy(), 1.0)
    assert np.array_equal(w, group_balanced_weights(labels))          # gen6 levels.py:554-558
    assert cl.system_labels[f"{TBP}|{TODGA}"] == F.system_cluster_label(TODGA)       # clustered on the PRIMARY one
    cv2 = pd.concat([cv, synth_cv(mix, seed=2)])
    arm = P.B8Level(fold=1, cv=cv2, config=FAST).fit(rows)
    assert arm.model.total_sample_weight == pytest.approx(len(set(labels)))
    assert arm.clusters.cluster_rows == cl.cluster_rows


def test_b8_level_fit_predict_clip_and_guards(level_data):
    fr, cv, tr, qu = level_data
    arm = P.B8Level(fr, cv=cv, fold=3, config=FAST).fit(tr)
    pred = arm.predict(qu)
    assert list(pred.columns) == list(I.PREDICTION_COLUMNS) + list(P.B8_DIAGNOSTIC_COLUMNS)
    assert pred["row_id"].tolist() == qu.index.tolist()
    assert (pred["fallback_level"] == "B8").all() and pred["std_logD"].isna().all()
    assert pred[["lower_50", "upper_95"]].isna().all().all()          # intervals come from the conformal wrapper
    lo, hi = P.training_clip_range(tr[I.TARGET_COL])
    raw = pred["b8_raw_mean"].to_numpy()
    assert np.array_equal(pred["mean_logD"].to_numpy(), np.clip(raw, lo, hi))
    assert arm.model.random_state == P.model_seed(3) == arm.model.forest.random_state
    assert not pred["b8_system_seen"].any()
    assert "contact_time_min" not in " ".join(arm.model.design_columns())       # all-NaN in training -> dropped
    assert "condition__log10_metal_M__missing" in arm.model.design_columns()
    # clipping binds when the forest output is pushed out of range (the forest itself never leaves it)
    arm.model.y_range = (float(np.median(raw)), float(np.median(raw)) + 1e-9)
    clipped = arm.predict(qu)
    assert clipped["b8_clipped"].any() and clipped["fallback_reason"].str.contains("clipped_to_training_range").any()
    arm.model.y_range = (lo, hi)
    # row order of the training rows does not matter
    perm = tr.iloc[np.random.default_rng(0).permutation(len(tr))]
    again = P.B8Level(fr, cv=cv, fold=3, config=FAST).fit(perm).predict(qu)
    assert np.array_equal(again["mean_logD"].to_numpy(), pred["mean_logD"].to_numpy())
    # guards
    with pytest.raises(AssertionError):
        arm.predict(tr.iloc[:3])                                      # a query row is a training row
    ctx = I.FitContext(hidden_index=pd.Index(tr.index[:2]))
    with pytest.raises(AssertionError):
        P.B8Level(fr, cv=cv, fold=3, config=FAST).fit(tr, ctx)        # a hidden row among the training rows
    bad = tr.copy()
    bad.loc[bad.index[0], I.TARGET_COL] = np.nan
    with pytest.raises(ValueError):
        P.B8Level(fr, cv=cv, fold=3, config=FAST).fit(bad)
    # sparse ECFP storage gives the same forest
    sp = P.B8Level(fr, cv=cv, fold=3, config=FAST, ecfp_format="sparse").fit(tr).predict(qu)
    assert np.array_equal(sp["mean_logD"].to_numpy(), pred["mean_logD"].to_numpy())


def test_provenance_and_id_columns_never_become_b8_features(level_data):
    fr, cv, tr, qu = level_data
    leaky = tr.copy()
    y = leaky[I.TARGET_COL].to_numpy()
    for c in ("g19_publication_id", "doi_primary", "g19_study_id", "pub_group", "comments_raw"):
        leaky[c] = y if c != "pub_group" else [f"g{v:.6f}" for v in y]                   # a perfect target leak
    arm = P.B8Level(fold=0, cv=cv, config=FAST).fit(leaky)
    cols = arm.model.design_columns(arm.wide_columns)
    F.assert_feature_columns_allowed(cols)
    assert not any(c.split("__", 1)[-1].split("__")[0] in set(L.PROVENANCE_COLUMNS) | set(F.ID_COLUMNS)
                   | set(F.TARGET_COLUMNS) for c in cols)
    plain = P.B8Level(fold=0, cv=cv, config=FAST).fit(tr)
    assert np.array_equal(arm.predict(qu)["mean_logD"].to_numpy(), plain.predict(qu)["mean_logD"].to_numpy())
    assert arm.features.state_digest == plain.features.state_digest

    class Leaky(F.Block):                                             # a block reading a provenance column is refused
        name = "leaky"
        source_columns = ("g19_publication_id",)

    with pytest.raises(ValueError):
        F.FeatureSet([Leaky()])


def test_b8_level_table_fast_path_and_conformal_wrapper(level_data):
    fr, cv, tr, qu = level_data
    table = I.RowTable(fr)
    mask = table.mask_of(tr.index)
    ctx = I.FitContext(table=table, seed=104729, v6_mask=pd.Series(False, index=fr.index),
                       isolation_check=lambda a, b: {"ok": True})
    arm = P.B8Level(fr, cv=cv, fold=5, config=FAST)
    fast = arm.clone().fit_table(table, mask, ctx).predict_positions(table.positions(qu.index))
    slow = arm.clone().fit(tr).predict(qu)
    assert np.array_equal(fast["mean_logD"].to_numpy(), slow["mean_logD"].to_numpy())
    with pytest.raises(AssertionError):
        arm.clone().fit_table(table, mask, ctx).predict_positions(table.positions(tr.index[:2]))
    hid = I.FitContext(table=table, hidden_index=pd.Index(tr.index[:1]))
    with pytest.raises(AssertionError):
        arm.clone().fit_table(table, mask, hid)
    wrap = I.ConformalWrapper(arm, splitter=I.GroupKFoldCalibration(3)).fit_table(table, mask, ctx)
    out = wrap.predict_positions(table.positions(qu.index))
    assert np.array_equal(out["mean_logD"].to_numpy(), fast["mean_logD"].to_numpy())
    assert (out["upper_80"] - out["lower_80"] > 0).all() and out["conformal_n_calibration"].iloc[0] > 0
    assert len(wrap.calibration_units) >= 1


# ------------------------------------------------------------------------------------------------ gen14 quantities

def test_radius_basis_equals_gen13_physics_basis():
    _import_path(GEN13)
    from gen13sep import basis as B13
    from gen13sep import metals as M13

    assert tuple(M13.LANTHANIDES) == LN
    assert np.array_equal(P.basis_radii(), np.array([M13.SHANNON_RADIUS_CN8[m] for m in LN]))
    assert np.allclose(P.radius_basis(), B13.physics_basis_matrix(("radius", "radius_sq")), atol=1e-12, rtol=0)
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(6, 14))
    Y[rng.random(Y.shape) < 0.4] = np.nan
    Y[:, :2] = rng.normal(size=(6, 2))
    c = Y - np.nanmean(Y, axis=1, keepdims=True)
    ref = B13.fit_all_coefficients(c, B13.physics_basis_matrix(("radius", "radius_sq")))
    assert np.allclose(P.curve_coefficients(c), ref, atol=1e-10)


def test_cell_table_replicates_states_and_labels():
    s_heavy, s_light, s_few = topo_systems(3)
    fr = synth_rows({s_heavy: -0.6, s_light: 0.5}, n_cond=2, seed=11)
    few = synth_rows({s_few: -0.6}, states=[f"{m}(III)" for m in ("La", "Nd", "Eu", "Dy")], seed=12, prefix="f")
    noise = synth_rows({s_heavy: -0.6}, states=["Pm(III)", "Ce(IV)"], extra_states=(), seed=13, prefix="n")
    noise["condition_key"] = fr["condition_key"].iloc[0]                            # same cell as fr's first rows
    noise[I.PUB_GROUP_COL] = fr[I.PUB_GROUP_COL].iloc[0]
    dup = fr.iloc[:3].copy()                                                         # replicates of 3 cell rows
    dup.index = [f"d{i}" for i in range(3)]
    dup[I.TARGET_COL] += 1.0
    rows = pd.concat([fr, few, noise, dup])
    rows.loc[rows.index[5], SG.METAL_COL] = None                                     # an X(?) row never counts
    cells = P.ln3_cell_table(rows, condition_key=rows["condition_key"])
    assert set(cells[SG.SYSTEM_COL]) == {s_heavy, s_light}                           # the 4-state system has no cell
    assert len(cells) == 8 and cells["n_states"].max() <= 14
    assert (cells.loc[cells[SG.SYSTEM_COL] == s_heavy, "heavy_selective"]).all()
    assert not (cells.loc[cells[SG.SYSTEM_COL] == s_light, "heavy_selective"]).any()
    # replicate averaging: the duplicated rows move the cell mean exactly as a per-(cell, metal) mean does
    key = tuple(fr.iloc[0][[I.PUB_GROUP_COL, SG.SYSTEM_COL, "condition_key"]])
    cell_rows = rows[(rows[I.PUB_GROUP_COL] == key[0]) & (rows[SG.SYSTEM_COL] == key[1])
                     & (rows["condition_key"] == key[2]) & P.ln3_states(rows[SG.METAL_COL])]
    means = cell_rows.groupby(SG.METAL_COL)[I.TARGET_COL].mean()
    Y = np.array([means.get(f"{m}(III)", np.nan) for m in LN])
    exp = P.curve_coefficients((Y - np.nanmean(Y))[None, :])[0]
    got = cells[(cells[I.PUB_GROUP_COL] == key[0]) & (cells[SG.SYSTEM_COL] == key[1])
                & (cells["condition_key"] == key[2])]
    assert np.allclose(got[["amp", "curvature"]].to_numpy()[0], exp)
    sysd = P.system_radius_coefficients(rows, condition_key=rows["condition_key"]).set_index(SG.SYSTEM_COL)
    assert sysd.loc[s_few, "status"] == "below_min_states" and sysd.loc[s_few, "n_ln3_states"] == 4
    assert sysd.loc[s_heavy, "status"] == "fit_unit" and sysd.loc[s_heavy, "n_rich_cells"] == 4
    assert np.isclose(sysd.loc[s_heavy, "amp"], cells.loc[cells[SG.SYSTEM_COL] == s_heavy, "amp"].mean())
    # >= 5 states across cells but no single cell with 5: no well-determined curve -> no_rich_cell
    a = synth_rows({s_few: 0.4}, states=["La(III)", "Nd(III)", "Eu(III)"], pubs=("g1",), n_cond=1, seed=14, prefix="a")
    b = synth_rows({s_few: 0.4}, states=["Gd(III)", "Er(III)", "Lu(III)"], pubs=("g2",), n_cond=1, seed=15, prefix="b")
    split = pd.concat([a, b])
    st = P.system_radius_coefficients(split, condition_key=split["condition_key"]).set_index(SG.SYSTEM_COL)
    assert st.loc[s_few, "n_ln3_states"] == 6 and st.loc[s_few, "status"] == "no_rich_cell"


def test_delta_z_magnitudes_are_training_pair_means():
    s1, s2 = topo_systems(2)
    rows = synth_rows({s1: -0.5, s2: 0.4}, seed=21, extra_states=("Am(III)", "Ce(IV)"))
    rows.loc[rows.index[0], SG.METAL_COL] = None
    tab, pooled, pairs = P.delta_z_magnitudes(rows, condition_key=rows["condition_key"])
    zs = {m: z for m, z in MET.ATOMIC_NUMBER.items() if m in MET.LANTHANIDES}
    ln3 = rows[[isinstance(s, str) and s.endswith("(III)") and s.split("(")[0] in zs for s in rows[SG.METAL_COL]]]
    brute: dict[int, list[float]] = {}
    for _, g in ln3.groupby([I.PUB_GROUP_COL, SG.SYSTEM_COL, "condition_key"]):
        v = g[[SG.METAL_COL, I.TARGET_COL]].to_numpy(dtype=object)
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if v[i][0] != v[j][0]:
                    dz = abs(zs[v[i][0].split("(")[0]] - zs[v[j][0].split("(")[0]])
                    brute.setdefault(dz, []).append(abs(float(v[i][1]) - float(v[j][1])))
    assert sorted(brute) == tab["delta_z"].tolist()
    for d, n, m in tab.itertuples(index=False):
        assert n == len(brute[d]) and np.isclose(m, np.mean(brute[d]))
    assert np.isclose(pooled, np.mean([x for v in brute.values() for x in v]))
    assert (pairs["category_class"] == "Ln-Ln").all() and (pairs["delta_z"] >= 1).all()


# ------------------------------------------------------------------------------------------------ B8 direction arm

@pytest.fixture(scope="module")
def direction_data():
    systems = topo_systems(9)
    train = systems[:8]
    sentinel = systems[8]
    amps = {s: (-0.5 if i % 2 == 0 else 0.45) for i, s in enumerate(train)}
    tr = synth_rows(amps, seed=31, prefix="t")
    sen = synth_rows({sentinel: 0.9}, seed=32, prefix="s")
    return tr, sen, train, sentinel


def _pairs_of(rows: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"fold": "test", EP.PAIR_KEY_COLS[0]: rows[I.PUB_GROUP_COL],
                       EP.PAIR_KEY_COLS[1]: rows[SG.SYSTEM_COL], EP.PAIR_KEY_COLS[2]: rows["condition_key"],
                       "g19_metal_state": rows[SG.METAL_COL], "log_D": rows[I.TARGET_COL]}, index=rows.index)
    return EP.comparable_pairs(df)


def test_direction_logistic_equals_the_gen14_estimator(direction_data):
    tr, sen, train, sentinel = direction_data
    d = P.B8Direction().fit(tr, condition_key=tr["condition_key"])
    assert d.status == "fitted" and list(d.units) == sorted(train)
    _import_path(GEN14)
    from gen14 import models as G14M                                  # the estimator code only; no frozen model

    X = F.topo39_for_systems(list(d.units))[list(F.topo39_columns())].to_numpy(dtype=float)
    amp = d.systems.set_index(SG.SYSTEM_COL).loc[list(d.units), "amp"].to_numpy(dtype=float)
    Xte = F.topo39_for_systems([sentinel] + list(d.units))[list(F.topo39_columns())].to_numpy(dtype=float)
    ref = G14M.dir_logistic()(X, amp, np.ones(len(amp)), np.zeros(len(amp)), Xte, 0)
    got = d.p_heavy([sentinel] + list(d.units))
    assert np.allclose(got, ref, atol=1e-6)
    # training labels: heavy-selective systems were generated with a negative radius coefficient
    labels = d.systems.set_index(SG.SYSTEM_COL).loc[list(d.units), "heavy_selective"]
    assert labels.to_dict() == {s: (i % 2 == 0) for i, s in enumerate(train)}


def test_direction_refit_uses_only_training_systems_sentinel(direction_data):
    tr, sen, train, sentinel = direction_data
    d = P.B8Direction().fit(tr, condition_key=tr["condition_key"])
    digest = d.state_digest
    assert sentinel not in d.units and d.prep.systems == sorted(train)
    q_pairs = _pairs_of(sen)
    rec = d.pair_records(q_pairs)
    lnln = (q_pairs["category_class"] == "Ln-Ln").to_numpy()
    assert lnln.any() and (~lnln).any()                               # the synthetic rows carry Am(III) too
    assert rec["defined"].to_numpy()[lnln].all() and (rec["reason"][lnln] == "ok").all()   # predicted, never trained on
    assert (rec["reason"][~lnln] == "not_ln3_ln3").all()
    X, _ = d.prep.transform([sentinel])
    assert np.allclose(rec["p_heavy"], d.logistic.predict_proba(X)[:, 1][0])
    # predicting never refits and never reads a query target
    blind = q_pairs.drop(columns=["y_a", "y_b", "logsf_obs"])
    assert np.array_equal(d.pair_records(blind)["logsf_pred"].to_numpy(), rec["logsf_pred"].to_numpy(), equal_nan=True)
    scrambled = q_pairs.assign(logsf_obs=-q_pairs["logsf_obs"] * 7.0)
    assert np.array_equal(d.predict_pairs(scrambled).to_numpy(), rec["logsf_pred"].to_numpy(), equal_nan=True)
    d.coverage(q_pairs)
    assert d.state_digest == digest
    # the sentinel's rows change the fit only when they are training rows
    both = pd.concat([tr, sen])
    d2 = P.B8Direction().fit(both, condition_key=both["condition_key"])
    assert sentinel in d2.units and d2.state_digest != digest
    assert not np.allclose(d2.logistic.coef_, d.logistic.coef_)
    d3 = P.B8Direction().fit(tr, condition_key=tr["condition_key"])
    assert d3.state_digest == digest                                  # deterministic refit
    ctx = I.FitContext(hidden_index=pd.Index(tr.index[:1]))
    with pytest.raises(AssertionError):
        P.B8Direction().fit(tr, ctx, condition_key=tr["condition_key"])


def test_pair_direction_sign_magnitude_and_coverage(direction_data):
    tr, sen, train, sentinel = direction_data
    d = P.B8Direction().fit(tr, condition_key=tr["condition_key"])
    heavy_sys = train[0]                                              # generated heavy-selective
    light_sys = train[1]
    mag = d.magnitude_table.set_index("delta_z")["mean_abs_logsf"]
    pairs = pd.DataFrame({
        SG.SYSTEM_COL: [heavy_sys, heavy_sys, light_sys, heavy_sys, NOVEL, heavy_sys, None],
        "state_a": ["Nd(III)", "La(III)", "Nd(III)", "Nd(III)", "Nd(III)", "Lu(III)", "Nd(III)"],
        "state_b": ["La(III)", "Nd(III)", "La(III)", "Am(III)", "La(III)", "La(III)", "La(III)"],
    })
    rec = d.pair_records(pairs)
    p = d.p_heavy([heavy_sys, light_sys])
    assert p[0] >= 0.5 > p[1]
    assert rec["logsf_pred"].iloc[0] == pytest.approx(+mag[3])       # heavier Nd more extracted
    assert rec["logsf_pred"].iloc[1] == pytest.approx(-mag[3])       # same pair, lighter first
    assert rec["logsf_pred"].iloc[2] == pytest.approx(-mag[3])       # light-selective system
    assert rec["reason"].tolist()[3:] == ["not_ln3_ln3", "no_topo39", "ok", "unknown_system"]
    assert 14 in mag.index and rec["logsf_pred"].iloc[5] == pytest.approx(mag[14])   # Lu/La, delta Z = 14
    assert rec["direction"].iloc[0] == 1.0 and np.isnan(rec["direction"].iloc[3])
    assert not rec["magnitude_pooled"].to_numpy(dtype=bool).any()
    # HEAVIER-style undefined pairs are NaN, and pair_summary-style scoring can count them (direction_only)
    scored = EP.score_pairs(pairs.assign(logsf_obs=1.0), d.predict_pairs(pairs), design="V5", direction_only=True)
    assert scored["dir_0.3"].isna().sum() == 3
    cov = d.coverage(pairs)
    assert cov["n_fit_units"] == 8 and cov["n_systems_training"] == 8 and cov["n_systems_topo39"] == 8
    assert cov["n_heavy_units"] == 4 and cov["n_light_units"] == 4
    assert cov["n_pairs"] == 7 and cov["n_pairs_with_direction"] == 4
    assert cov["pair_reasons"] == {"ok": 4, "not_ln3_ln3": 1, "no_topo39": 1, "no_direction_model": 0,
                                   "no_magnitude": 0, "unknown_system": 1}
    assert cov["n_training_pairs"] == len(P.training_ln3_pairs(tr, condition_key=tr["condition_key"]))


def test_direction_one_class_and_no_unit_folds():
    s = topo_systems(3)
    one = synth_rows({s[0]: -0.5, s[1]: -0.4}, seed=41)
    d = P.B8Direction().fit(one, condition_key=one["condition_key"])
    assert d.status == "constant_one_class" and d.constant_p == 1.0   # gen14/models.py:140-141
    assert np.allclose(d.p_heavy([s[2], s[0]]), 1.0)
    few = synth_rows({s[0]: -0.5}, states=["La(III)", "Nd(III)"], seed=42)
    few = pd.concat([few, synth_rows({NOVEL: -0.5}, seed=43, prefix="x")])  # enough states, but no TOPO39
    d0 = P.B8Direction().fit(few, condition_key=few["condition_key"])
    cov = d0.coverage(_pairs_of(few))
    assert d0.status == "no_training_units" and cov["n_fit_units"] == 0
    assert cov["n_systems_with_coefficient_without_topo39"] == 1
    assert cov["pair_reasons"]["no_direction_model"] > 0 and cov["n_pairs_with_direction"] == 0


def test_frozen_gen14_model_is_never_loaded():
    src = inspect.getsource(P)
    tree = ast.parse(src)
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names} | {n.module.split(".")[0] for n in ast.walk(tree)
                                     if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"joblib", "pickle", "gen14", "gen13sep", "lanthanide_separation"}
    body = src.replace(P.__doc__ or "", "")                    # the module docstring names the frozen model; code must not
    for token in ("joblib", "deploy", "gen14_direction", "gen13_separation", "bench.pkl"):
        assert token not in body, token


def test_b8_bundle_fits_both_halves(level_data):
    fr, cv, tr, qu = level_data
    b = P.fit_b8(tr, fold=0, frame=fr, cv=cv, condition_key=fr["condition_key"], config=FAST)
    assert b.level.fitted and b.direction.fitted
    assert b.coverage()["n_train_rows"] == len(tr)
    assert len(b.state_digest) == 64


# ------------------------------------------------------------------------------------------------ real rows (slow)

@pytest.mark.slow
def test_topo39_and_ln3_direction_coverage_on_model_rows():
    """Target-bearing training statistics only (no logistic, no forest, nothing scored)."""
    df = L.load_model_rows()
    fr = I.prepare_frame(df)
    from gen19ct.data import normalize as N

    ck = N.condition_key(df)
    sysd = P.system_radius_coefficients(fr, condition_key=ck)
    avail = P._topo39_available(sysd[SG.SYSTEM_COL])
    st = sysd["status"]
    units = sysd[(st == "fit_unit") & sysd[SG.SYSTEM_COL].map(avail)]
    print("B8 direction coverage (all MODEL rows as training input):",
          {"systems": len(sysd), "ge5_ln3_states": int((st != "below_min_states").sum()),
           "with_coefficient": int((st == "fit_unit").sum()), "no_rich_cell": int((st == "no_rich_cell").sum()),
           "fit_units_with_topo39": len(units)})
    assert len(sysd) == 259
    assert int((st != "below_min_states").sum()) == 83 == F.topo39_coverage(df)["n_systems_ge5_ln3_states"]
    assert int((st == "fit_unit").sum()) == 82 and int((st == "no_rich_cell").sum()) == 1
    assert len(units) == 80


@pytest.mark.slow
def test_b8_level_plumbing_on_real_rows_without_scoring():
    """A 20-tree forest on the rows of ~40 systems, predictions for ~150 rows of other systems; the query target is
    never read and no error is computed (an ad-hoc system split, not a registered fold)."""
    from gen19ct.data import normalize as N

    df = L.load_model_rows()
    systems = np.array(sorted(df["extractant_system_key"].unique()))
    perm = np.random.default_rng(19).permutation(len(systems))
    tr_sys, qu_sys = set(systems[perm[:40]]), set(systems[perm[40:60]])
    sub = df[df["extractant_system_key"].isin(tr_sys | qu_sys)]
    fr = I.prepare_frame(sub)
    cv = N.condition_vector(sub)
    tr = fr[fr["extractant_system_key"].isin(tr_sys)]
    qu = fr[fr["extractant_system_key"].isin(qu_sys)].iloc[:150].drop(columns=[I.TARGET_COL])
    arm = P.B8Level(fr, cv=cv, fold=0, config=P.MonoEtConfig(n_estimators=20, n_jobs=2)).fit(tr)
    pred = arm.predict(qu)
    lo, hi = P.training_clip_range(tr[I.TARGET_COL])
    assert len(pred) == len(qu) and np.isfinite(pred["mean_logD"]).all()
    assert ((pred["mean_logD"] >= lo) & (pred["mean_logD"] <= hi)).all()
    cols = arm.model.design_columns(arm.wide_columns)
    F.assert_feature_columns_allowed(cols)
    assert sum(c.startswith("ecfp_") for c in cols) == F.FP_BITS and len(arm.wide_columns) == F.FP_BITS

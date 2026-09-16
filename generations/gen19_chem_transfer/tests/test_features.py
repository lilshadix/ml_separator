"""Shared, train-only feature builders (``gen19ct/models/features.py``).

Real MODEL rows are used as INPUT ROWS only: no target is read, no model is fitted, nothing is scored, and the splits
below are ad-hoc system splits, never a registered outer fold.  No test writes a file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gen19ct.chemistry import support_graph as SG
from gen19ct.data import load as L
from gen19ct.data import normalize as N
from gen19ct.models import features as F

TODGA = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"
TBP = "CCCCOP(=O)(OCCCC)OCCCC"
C5BTBP = "CCCCCc1nnc(-c2cccc(-c3cccc(-c4nnc(CCCCC)c(CCCCC)n4)n3)n2)nc1CCCCC"
TEDGA = "CCN(CC)C(=O)COCC(=O)N(CC)CC"
#: a structure absent from every descriptor table (a succinamide homologue), for unseen-system queries
NOVEL = "CCCCCCN(CCCCCC)C(=O)CCC(=O)N(CCCCCC)CCCCCC"


@pytest.fixture(scope="module")
def corpus() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = L.load_model_rows()
    return df, N.condition_vector(df)


@pytest.fixture(scope="module")
def split(corpus):
    """~1/3 of the systems as 'training', a disjoint set of systems as 'query' rows (not a registered fold)."""
    df, _ = corpus
    systems = np.array(sorted(df["extractant_system_key"].unique()))
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(systems))
    train_sys = set(systems[perm[:90]]) | {TODGA}
    query_sys = set(systems[perm[90:130]]) - train_sys
    tr = df[df["extractant_system_key"].isin(train_sys)]
    qu = df[df["extractant_system_key"].isin(query_sys)]
    return tr, qu


# ------------------------------------------------------------------------------------------------ guards

def test_every_provenance_id_and_target_column_is_refused():
    for c in tuple(L.PROVENANCE_COLUMNS) + F.ID_COLUMNS + F.TARGET_COLUMNS:
        assert F.forbidden_reason(c) is not None, c
        assert F.forbidden_reason(f"metal__{c}") is not None, c
        with pytest.raises(ValueError):
            F.assert_feature_columns_allowed(["metal__Z", c])
    for c in ("pub_group", "g19_publication_id", "canonical_measurement_id", "doi_primary", "log_D"):
        assert c in F.FORBIDDEN_SOURCE_COLUMNS


def test_feature_set_refuses_a_block_that_reads_a_forbidden_column():
    for bad in ("doi_primary", "canonical_measurement_id", "g19_publication_id", "pub_group", "log_D", "comments_raw"):
        class Leaky(F.Block):
            name = "leaky"
            source_columns = (bad,)

        with pytest.raises(ValueError):
            F.FeatureSet([Leaky()])


def test_blocks_see_only_their_source_columns(corpus):
    df, cv = corpus
    rows = df.iloc[:400].copy()
    rows["pub_group"] = "g"                                   # an id column an arm frame would carry
    seen: list[set[str]] = []

    class Spy(F.ConditionBlock):
        name = "spy"
        source_columns = (SG.METAL_COL,)

        def raw(self, view, cvv):
            seen.append(set(view.columns))
            return super().raw(view, cvv)

    fs = F.FeatureSet(["metal", "extractant", Spy()])
    fm = fs.fit(rows, cv).transform(rows, cv)
    assert seen and all(not (s & F.FORBIDDEN_SOURCE_COLUMNS) for s in seen)
    assert all(s <= set(fs.source_columns) for s in seen)
    assert "log_D" not in set().union(*seen) and "canonical_measurement_id" not in set().union(*seen)
    F.assert_feature_columns_allowed(fm.frame.columns)
    for c in fm.frame.columns:
        assert c.split("__", 1)[1].split("=", 1)[0] not in F.FORBIDDEN_SOURCE_COLUMNS


# ------------------------------------------------------------------------------------------------ shapes

def test_presets_shapes_columns_and_dtypes(split, corpus):
    tr, qu = split
    _, cv = corpus
    for arm in ("B5", "M0", "FLAT_CAT", "B6", "B6r0", "B8", "M1"):
        fs = F.FeatureSet.for_arm(arm).fit(tr, cv)
        a, b = fs.transform(tr, cv), fs.transform(qu, cv)
        assert a.columns == b.columns, arm
        assert a.frame.shape == (len(tr), len(a.columns)) and b.frame.shape[0] == len(qu)
        assert a.frame.index.equals(tr.index) and b.frame.index.equals(qu.index)
        F.assert_feature_columns_allowed(a.frame.columns)
        for c in a.categorical_columns:
            assert a.frame[c].dtype == object and b.frame[c].dtype == object
        if arm in ("B6", "B6r0", "M1"):
            assert not a.categorical_columns
            assert np.isfinite(a.numeric_array(np.float64)).all() and np.isfinite(b.numeric_array(np.float64)).all()
        if arm in ("B5", "M0", "FLAT_CAT"):
            with pytest.raises(TypeError):
                a.numeric_array()
        if arm == "B8":
            bits, cols = b.wide["lig2d_ext__ecfp"]
            assert bits.dtype == np.uint8 and bits.shape == (len(qu), F.FP_BITS) and len(cols) == F.FP_BITS
            assert a.numeric_array(include_wide=True).shape == (len(tr), len(a.columns) + F.FP_BITS)
    b5 = F.FeatureSet.for_arm("B5").fit(tr, cv).transform(qu, cv)
    assert len(b5.block_columns["metal"]) == len(F.METAL_NUMERIC) + len(F.METAL_CATEGORICAL) == 11
    assert len(b5.block_columns["extractant"]) == 1 + 19 + 2 + 4 + F.N_PCA + 2 == 44
    assert len(b5.block_columns["condition"]) == len(F.CONDITION_NUMERIC) + len(F.CONDITION_CATEGORICAL) == 13
    assert "metal__polarizability_atom_au" not in b5.columns
    assert set(b5.categorical_columns) == {"metal__category", "metal__series", "metal__hsab_class", "extractant__family",
                                           "extractant__mechanism", "condition__acid_anion", "condition__diluent_family"}
    flat = F.FeatureSet.for_arm("FLAT_CAT").fit(tr, cv).transform(qu, cv)
    assert {"flat_cat__g19_metal_state", "flat_cat__extractant_system_key"} <= set(flat.categorical_columns)
    assert not any(c.startswith(("metal__", "extractant__")) for c in flat.columns)
    m1 = F.FeatureSet.for_arm("M1").fit(tr, cv)
    ids = m1.transform(qu, cv).ids
    assert set(ids) == {"metal_state_id", "series_id", "ox_id", "element_id", "category_id", "system_id"}
    sizes = m1.id_sizes()
    assert all(int(ids[k].max()) < sizes[k] and int(ids[k].min()) >= 0 for k in ids)


# ------------------------------------------------------------------------------------------------ fit on train only

def test_query_rows_never_change_the_fitted_state(split, corpus):
    tr, qu = split
    _, cv = corpus
    fs = F.FeatureSet.for_arm("M1").fit(tr, cv)
    d0 = fs.state_digest
    before = fs.transform(tr, cv).frame
    extreme = qu.iloc[:5].copy()
    extreme["acid_concentration_M"] = 1e6                   # a unique, extreme query value
    extreme["temperature_C"] = 999.0
    cv_ex = N.condition_vector(extreme)
    q = fs.transform(extreme, cv_ex)
    assert fs.state_digest == d0 and q.state_digest == d0
    pd.testing.assert_frame_equal(fs.transform(tr, cv).frame, before)
    prep = fs._preps["condition"]
    raw_tr = F.ConditionBlock().raw(tr, cv.loc[tr.index]).numeric
    assert prep.medians["log10_acid_M"] == float(np.median(raw_tr["log10_acid_M"].dropna()))
    mean, sd = prep.mean["condition__log10_acid_M"], prep.sd["condition__log10_acid_M"]
    assert np.allclose(q.frame["condition__log10_acid_M"], (6.0 - mean) / sd)
    refit = F.FeatureSet.for_arm("M1").fit(pd.concat([tr, extreme.set_axis([f"x{i}" for i in range(5)])]),
                                           pd.concat([cv, cv_ex.set_axis([f"x{i}" for i in range(5)])]))
    assert refit.state_digest != d0                          # the fit does use the rows it is given


def test_pca_is_fitted_on_training_systems_once_each(split, corpus):
    tr, qu = split
    _, cv = corpus
    fs = F.FeatureSet(["extractant"], categorical="onehot", missing="impute", standardize=True).fit(tr, cv)
    blk = fs.blocks[0]
    assert blk.pca_systems == sorted(tr["extractant_system_key"].unique())
    assert not set(blk.pca_systems) & set(qu["extractant_system_key"])
    comps = blk.pca_components.copy()
    fs.transform(qu, cv)
    assert np.array_equal(blk.pca_components, comps)
    # duplicating every row of one system changes row-weighted statistics but not the system-weighted PCA
    one = tr[tr["extractant_system_key"] == TODGA]
    dup = one.set_axis([f"dup{i}" for i in range(len(one))])
    tr2 = pd.concat([tr, dup])
    cv2 = pd.concat([cv, cv.loc[one.index].set_axis(dup.index)])
    fs2 = F.FeatureSet(["extractant"], categorical="onehot", missing="impute", standardize=True).fit(tr2, cv2)
    assert np.array_equal(fs2.blocks[0].pca_components, comps) and np.array_equal(fs2.blocks[0].pca_mean, blk.pca_mean)
    assert fs2.state_digest != fs.state_digest
    # sklearn PCA(svd_solver="full") agrees up to the sign convention
    from sklearn.decomposition import PCA

    X = np.vstack([F.morgan_counts(F.system_record(k).primary_smiles) for k in blk.pca_systems])
    ref = PCA(F.N_PCA, svd_solver="full").fit(X)
    assert np.allclose(np.abs(ref.components_), np.abs(comps), atol=1e-8)
    assert np.allclose(ref.explained_variance_, blk.pca_explained_variance, rtol=1e-8)
    for i in range(F.N_PCA):
        j = int(np.argmax(np.abs(comps[i])))
        assert comps[i, j] > 0


def test_short_fit_pads_pca_components(corpus):
    df, cv = corpus
    rows = df[df["extractant_system_key"].isin([TODGA, C5BTBP, TEDGA])]
    fs = F.FeatureSet(["extractant"]).fit(rows, cv)
    assert fs.blocks[0].pca_n_fitted == 2
    fm = fs.transform(rows, cv)
    assert (fm.frame[[f"extractant__pca_{i:02d}" for i in range(2, F.N_PCA)]] == 0).all().all()


def test_unseen_values_map_to_the_reserved_index(split, corpus):
    tr, qu = split
    _, cv = corpus
    q = qu.iloc[:3].copy()
    q["g19_metal_state"] = "Bk(III)"                         # in metals.csv, absent from every MODEL row
    q["g19_metal"] = "Bk"
    q["extractant_system_key"] = NOVEL
    new_components = []
    for comps in q["components"]:
        comps = [dict(x) for x in comps]
        ext = [x for x in comps if x.get("role") == "organic_extractant"][:1]
        for x in ext:
            x["smiles_canonical"] = NOVEL
        new_components.append(ext + [x for x in comps if x.get("role") != "organic_extractant"])
    q["components"] = pd.Series(new_components, index=q.index, dtype=object)
    cvq = N.condition_vector(q)
    assert all(len(t) == 1 for t in cvq["extractant_concentrations_sorted_M"])
    native = F.FeatureSet.for_arm("B5").fit(tr, cv)
    fm = native.transform(q, cvq)
    for k in ("metal_state_id", "element_id", "system_id"):
        assert (fm.ids[k] == F.RESERVED_INDEX).all(), k
    assert (fm.ids["series_id"] != F.RESERVED_INDEX).all() and (fm.ids["ox_id"] != F.RESERVED_INDEX).all()
    assert (fm.frame["extractant__family"] == F.UNSEEN_TOKEN).all()           # no family label for a novel system
    assert np.isfinite(fm.frame[[f"extractant__pca_{i:02d}" for i in range(F.N_PCA)]].to_numpy()).all()
    assert (~fm.diagnostics["system_seen"]).all() and (~fm.diagnostics["metal_state_seen"]).all()
    oh = F.FeatureSet(["metal", "extractant", "condition"], categorical="onehot", missing="impute").fit(tr, cv)
    ohq = oh.transform(q, cvq)
    fam = [c for c in ohq.columns if c.startswith("extractant__family=")]
    assert fam and (ohq.frame[fam] == 0).all().all()                           # unseen token: all-zero one-hot
    m1 = F.FeatureSet.for_arm("M1").fit(tr, cv)
    prep = m1._preps["extractant"]
    assert np.allclose(m1.transform(q, cvq).frame[fam].to_numpy(),
                       np.array([[-prep.mean[c] / prep.sd[c] for c in fam]] * len(q)))
    flat = F.FeatureSet.for_arm("FLAT_CAT").fit(tr, cv).transform(q, cvq)
    assert (flat.frame["flat_cat__g19_metal_state"] == F.UNSEEN_TOKEN).all()
    assert (flat.frame["flat_cat__extractant_system_key"] == F.UNSEEN_TOKEN).all()
    voc = F.Vocabulary().fit(["b", "a", "a"])
    assert voc.tokens == ["a", "b"] and voc.index(["a", "b", "zz"]).tolist() == [1, 2, 0] and voc.size == 3


# ------------------------------------------------------------------------------------------------ determinism

def test_digest_is_deterministic_and_row_order_free(split, corpus):
    tr, _ = split
    _, cv = corpus
    for arm in ("B5", "M1", "B8"):
        a = F.FeatureSet.for_arm(arm).fit(tr, cv)
        b = F.FeatureSet.for_arm(arm).fit(tr, cv)
        perm = tr.iloc[np.random.default_rng(3).permutation(len(tr))]
        c = F.FeatureSet.for_arm(arm).fit(perm, cv)
        assert a.state_digest == b.state_digest == c.state_digest, arm
        pd.testing.assert_frame_equal(a.transform(tr, cv).frame, c.transform(perm, cv).frame.loc[tr.index])
    assert F.FeatureSet.for_arm("B5").fit(tr, cv).state_digest != F.FeatureSet.for_arm("M1").fit(tr, cv).state_digest


def test_transform_refuses_a_changed_state(split, corpus):
    tr, qu = split
    _, cv = corpus
    fs = F.FeatureSet.for_arm("M1").fit(tr, cv)
    with pytest.raises(RuntimeError):
        F.FeatureSet.for_arm("M1").transform(qu, cv)
    fs._preps["condition"].medians["log10_acid_M"] += 1.0
    with pytest.raises(AssertionError):
        fs.transform(qu, cv)


# ------------------------------------------------------------------------------------------------ block values

def test_condition_block_follows_the_condition_vector(corpus):
    df, cv = corpus
    raw = F.ConditionBlock().raw(df, cv).numeric
    assert int(raw["acid_M_log10_grid"].sum()) == int(cv["acid_M_log10_grid"].sum()) == 302   # section 2
    assert np.allclose(raw["log10_acid_M"], cv["log10_acid_M"], equal_nan=True)
    assert np.allclose(raw["log10_extractant_M"], cv["log10_extractant_primary_M"], equal_nan=True)
    assert np.allclose(raw["phase_ratio_org_aq"], cv["phase_ratio_org_aq"], equal_nan=True)
    assert (raw["modifier_present"] == cv["modifier_name"].notna().astype(float)).all()
    multi = cv["complexant_concentrations_sorted_M"].map(len) > 1
    i = cv.index[multi][0]
    assert np.isclose(raw.at[i, "log10_complexant_M"], np.log10(sum(cv.at[i, "complexant_concentrations_sorted_M"])))
    assert raw.loc[cv["complexant_structure_key"].isna(), "log10_complexant_M"].isna().all()


def test_share_weighted_descriptors_and_primary_donors(corpus):
    df, cv = corpus
    key = "|".join(sorted([TODGA, TBP]))
    rows = df[df["extractant_system_key"] == key]
    assert len(rows)
    raw = F.ExtractantBlock()
    raw.fit(rows, cv.loc[rows.index])
    num = raw.raw(rows, cv.loc[rows.index]).numeric
    comp = F.static_tables().components_by_smiles
    for i in rows.index[:5]:
        c = dict(zip(sorted([TODGA, TBP]), cv.at[i, "extractant_concentrations_sorted_M"]))
        tot = c[TODGA] + c[TBP]
        for d in F.SHARE_WEIGHTED:
            exp = (c[TODGA] * comp.loc[TODGA, d] + c[TBP] * comp.loc[TBP, d]) / tot
            assert np.isclose(num.at[i, f"w_{d}"], exp)
        for d in F.DONOR_COLUMNS + F.DENTICITY_COLUMNS:
            assert num.at[i, d] == comp.loc[TODGA, d]
        assert num.at[i, "n_components"] == 2


def test_unknown_state_rows_keep_element_level_values_only(corpus):
    df, cv = corpus
    rows = df[df["g19_metal_state"].isna() & (df["g19_metal"] == "Nd")].iloc[:3]
    rec = F.metal_record(None, "Nd")
    assert rec["Z"] == 60 and rec["series"] == "Ln" and rec["electronegativity_pauling"] == 1.14
    for c in ("formal_charge", "radius_cn6_A", "radius_cn8_A", "radius_cn9_A", "f_electron_count", "d_electron_count"):
        assert np.isnan(rec[c]), c
    assert rec["hsab_class"] is None and rec["ox_token"] == F.NA_TOKEN
    b8 = F.B8MetalBlock().raw(rows, None).numeric
    assert (b8["lanthanide_index"] == 3).all() and b8["radius_cn8_A"].isna().all()
    am = F.B8MetalBlock().raw(df[df["g19_metal_state"] == "Am(III)"].iloc[:2], None).numeric
    assert am["lanthanide_index"].isna().all() and (am["radius_cn8_A"] == 1.09).all()
    assert F.metal_record("Nd(III)", "Nd")["radius_cn8_A"] == 1.109


def test_massaction_prior_matches_b7():
    from gen19ct.models import mass_action as MA

    assert F.N0 == MA.N_PRIOR == 3.0


def test_ecfp_bits_and_cluster_weights(split, corpus):
    tr, qu = split
    _, cv = corpus
    bits = F.ecfp_bits(TODGA)
    assert np.flatnonzero(bits).tolist() == sorted(SG.fingerprint(TODGA).GetOnBits())
    fm = F.FeatureSet(["massaction", "lig2d_ext"], categorical="onehot").fit(tr, cv).transform(tr, cv)
    assert np.allclose(fm.frame["massaction__n0_x_log10_extractant_M"], 3.0 * cv.loc[tr.index, "log10_extractant_primary_M"])
    todga_rows = np.flatnonzero((tr["extractant_system_key"] == TODGA).to_numpy())
    assert np.array_equal(fm.wide["lig2d_ext__ecfp"][0][todga_rows[0]], bits)
    cl = F.EcfpClusters().fit(tr)
    labels = cl.labels(tr)
    w = cl.sample_weights()
    per = pd.Series(w, index=tr.index).groupby(labels).sum()
    assert np.allclose(per.to_numpy(), 1.0)
    by_label: dict[str, set[bytes]] = {}
    for s, lab in cl.system_labels.items():
        by_label.setdefault(lab, set()).add(F.ecfp_bits(F.system_record(s).primary_smiles).tobytes())
    assert all(len(v) == 1 for v in by_label.values())                    # one label <=> one bit pattern
    assert len(set(cl.system_labels.values())) == len({b for v in by_label.values() for b in v})
    with pytest.raises(KeyError):
        cl.sample_weights(qu.index[:3])


def test_d_desc_scale_matches_the_lookup_engine(split):
    from gen19ct.models import baselines as B
    from gen19ct.models import interface as I

    tr, _ = split
    fr = I.prepare_frame(tr)
    systems, components = I.load_descriptor_tables()
    table = I.RowTable(fr, systems=systems, components=components)
    mask = np.ones(table.n, dtype=bool)
    eng = B.LookupEngine(table, mask, np.zeros(table.n, dtype=bool))
    mu, sd = eng._d_desc_scale(("base",))
    sc = F.DDescScaler().fit(tr)
    assert np.array_equal(mu, sc.mu) and np.array_equal(sd, sc.sd)
    assert sc.distance(TODGA, TODGA) == 0.0 and sc.distance(TODGA, TEDGA) > 0


# ------------------------------------------------------------------------------------------------ TOPO39

def test_topo39_values_match_the_source_table_for_three_known_smiles(corpus):
    df, _ = corpus
    src = pd.read_parquet(F.COORDINATION_PARQUET)
    src = src.set_index("extractant") if src.index.name != "extractant" else src
    cols = [c for c in src.columns if c.startswith(("coord__dist__", "coord__arm__"))]
    assert len(cols) == 39 and F.topo39_columns() == tuple(cols)
    got = F.topo39_for_smiles([TODGA, C5BTBP, TEDGA, NOVEL, None])
    for i, smi in enumerate((TODGA, C5BTBP, TEDGA)):
        assert smi in src.index
        assert np.array_equal(got.iloc[i][cols].to_numpy(dtype=float), src.loc[smi, cols].to_numpy(dtype=float))
    assert got.iloc[3:].isna().all().all()
    sysdf = F.topo39_for_systems([TODGA, C5BTBP, TEDGA])
    assert sysdf["topo39_available"].all()
    cov = F.topo39_coverage(df)
    table_smiles = set(F.topo39_table().index)
    systems = sorted(df["extractant_system_key"].unique())
    with_topo = {s for s in systems if F.system_record(s).primary_smiles in table_smiles}
    assert cov["n_systems"] == len(systems) and cov["n_systems_topo39"] == len(with_topo)
    assert cov["n_rows_topo39"] == int(df["extractant_system_key"].isin(with_topo).sum())
    assert cov["n_table_structures"] == 183
    print("TOPO39 coverage (MODEL rows):", cov)


def test_topo39_prep_is_fitted_on_its_systems_only():
    prep = F.Topo39Prep().fit([TODGA, C5BTBP, TEDGA, NOVEL])
    assert prep.systems == sorted([TODGA, C5BTBP, TEDGA])
    state = F.state_digest(prep.state())
    X, ok = prep.transform([TODGA, NOVEL])
    assert ok.tolist() == [True, False] and np.isnan(X[1]).all() and np.isfinite(X[0]).all()
    assert F.state_digest(prep.state()) == state


# ------------------------------------------------------------------------------------------------ memory

def test_ecfp_block_memory_on_the_full_model_corpus(corpus):
    df, cv = corpus
    dense = F.FeatureSet(["lig2d_ext"], categorical="onehot").fit(df, cv).transform(df, cv)
    bits = dense.wide["lig2d_ext__ecfp"][0]
    sparse_fm = F.FeatureSet(["lig2d_ext"], categorical="onehot", ecfp_format="sparse").fit(df, cv).transform(df, cv)
    sp = sparse_fm.wide["lig2d_ext__ecfp"][0]
    dense_mb = bits.nbytes / 1e6
    csr_mb = (sp.data.nbytes + sp.indices.nbytes + sp.indptr.nbytes) / 1e6
    print(f"LIG2D_EXT on {len(df)} MODEL rows: dense uint8 {dense_mb:.2f} MB, CSR {csr_mb:.2f} MB (nnz {sp.nnz}), "
          f"float32 dense would be {bits.size * 4 / 1e6:.2f} MB")
    assert bits.shape == (len(df), F.FP_BITS) and bits.dtype == np.uint8
    assert bits.nbytes == len(df) * F.FP_BITS
    assert sp.dtype == np.uint8 and sp.shape == bits.shape and (sp != 0).sum() == int(bits.sum())
    assert np.array_equal(sp[:500].toarray(), bits[:500])
    assert dense_mb < 26.0 and csr_mb < 5.0
    assert not dense.diagnostics["lig2d_fingerprint_missing"].any()

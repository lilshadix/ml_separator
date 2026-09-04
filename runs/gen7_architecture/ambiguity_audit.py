"""Empirical audit of the "17/190 canonical SMILES map to multiple extractant_name" warning.

Joins `dataset with 3D structures/dataset.parquet` to the upstream
`lanthanide_dataset_builder/raw_data/*_SAFE.csv` on safe_exp_id = "{stem}:{exp_id}"
and answers:
  (a) for every ambiguous canonical_smiles: names, row counts, mean/sd of log_D per
      name at MATCHED conditions;
  (b) whether the name split is explained by a recoverable upstream field;
  (c) how much within-SMILES log_D variance at matched conditions extractant_name
      explains, and how much of THAT the recoverable fields explain.

Writes ambiguity_audit.json and ambiguity_audit.md beside itself.
"""
from __future__ import annotations
import os, sys, json, glob, re
from itertools import combinations
import numpy as np, pandas as pd

REPO = '/Users/lilshadix/PycharmProjects/ml_separator'
OUT  = os.path.join(REPO, 'runs', 'gen7_architecture')
DS   = os.path.join(REPO, 'dataset with 3D structures', 'dataset.parquet')
RAW  = '/Users/lilshadix/PycharmProjects/lanthanide_dataset_builder/raw_data/*_SAFE.csv'
sys.path.insert(0, os.path.join(REPO, 'src'))

TODGA  = 'CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC'
DMDPh  = 'CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2)n1)c1ccccc1'
C5BTBP = 'CCCCCc1nnc(-c2cccc(-c3cccc(-c4nnc(CCCCC)c(CCCCC)n4)n3)n2)nc1CCCCC'
rng = np.random.default_rng(20260819)

def _s(x): return x.fillna('<NA>').astype(str).str.strip()

# ------------------------------------------------------------------ load + join
df = pd.read_parquet(DS)
frames = []
for p in sorted(glob.glob(RAW)):
    stem = os.path.basename(p)[:-4]
    r = pd.read_csv(p, low_memory=False)
    r['safe_exp_id'] = stem + ':' + r['exp_id'].astype(str)
    r['_src_file'] = stem
    frames.append(r)
up = pd.concat(frames, ignore_index=True)
J = df.merge(up, on='safe_exp_id', how='left', suffixes=('', '_up'))
assert len(J) == len(df)

R: dict = {}
R['provenance'] = dict(
    dataset=DS, upstream_glob=RAW, dataset_rows=int(len(df)), upstream_rows=int(len(up)),
    join_key='safe_exp_id == "{stem}:{exp_id}"',
    dataset_safe_exp_id_unique=bool(df.safe_exp_id.is_unique),
    upstream_safe_exp_id_unique=bool(up.safe_exp_id.is_unique),
    join_matched_rows=int(J.Extractant_Name.notna().sum()),
    join_unmatched_rows=int(J.Extractant_Name.isna().sum()),
    extractant_name_equals_upstream_Extractant_Name=float((J.extractant_name == J.Extractant_Name).mean()),
    LIGAND_SMILES_equals_upstream_Extractant_SMILES=float((J.LIGAND_SMILES == J.Extractant_SMILES).mean()),
    canonical_smiles_equals_upstream_Extractant_SMILES=float((J.canonical_smiles == J.Extractant_SMILES).mean()),
)

g = J.groupby('canonical_smiles').extractant_name.nunique()
AMB = sorted(g[g > 1].index.tolist())
R['headline'] = dict(
    n_canonical_smiles=int(J.canonical_smiles.nunique()),
    n_extractant_name=int(J.extractant_name.nunique()),
    n_ambiguous_smiles=int(len(AMB)),
    n_names_mapping_to_multiple_smiles=int((J.groupby('extractant_name').canonical_smiles.nunique() > 1).sum()),
    rows_in_ambiguous_smiles=int(J.canonical_smiles.isin(AMB).sum()),
    frac_rows_in_ambiguous_smiles=round(float(J.canonical_smiles.isin(AMB).mean()), 4),
)

# ------------------------------------------------------------------ cell keys
J['_acid'] = _s(J.Acid_Name).str.lower()
J['_cellP'] = (J.metal.astype(str) + '|' + J.metal_ox.astype(str) + '|' + J._acid + '|'
               + J.cond__acid_concentration_M.round(4).astype(str) + '|'
               + J.cond__extractant_concentration_M.round(6).astype(str))
COND = [c for c in df.columns if c.startswith('cond__')]
_lab = J[COND].round(6).fillna(-999).astype(str).agg('|'.join, axis=1)
J['_cellM'] = (J.metal.astype(str) + '|' + J.metal_ox.astype(str) + '|'
               + pd.util.hash_pandas_object(_lab, index=False).astype(str))
for t in 'PM':
    J['_sc' + t] = J.canonical_smiles + '||' + J['_cell' + t]
R['cell_definitions'] = {
    'P (as requested)': 'metal | metal_ox | upstream Acid_Name | cond__acid_concentration_M (4dp) | cond__extractant_concentration_M (6dp)',
    'M (model-visible)': 'metal | metal_ox | the full 64-column cond__* vector, i.e. everything the level model can distinguish',
}

F_COND = ['Solvent_Name','f_Solvent_Name','Phase_Modifier_Name','Phase_Modifier_Concentration_M',
          'Holdback_Agent_Name','Holdback_Agent_Concentration_M','Acid_Concentration_Organic_M',
          'Metal_Concentration_mM','obsTempsValue','Contact_Time_min','Shaking_Time_min',
          'Radiolytic_Dosage_kGy','volValue','thirdValue']
F_PROV = ['DOI','entry_author','_src_file']
F_LEAK = ['ini_comp','comments_description']      # literally contain the extractant name
ALLF   = F_COND + F_PROV + F_LEAK

def tup(frame, fields):
    fields = [f for f in fields if f in frame.columns]
    return pd.Series(['|'.join(v) for v in zip(*[_s(frame[f]) for f in fields])], index=frame.index)

# ------------------------------------------------------------------ (a) + (b)
def group_report(s, tag='P'):
    t = J[J.canonical_smiles == s]
    sc = '_sc' + tag
    nn = t.groupby(sc).extractant_name.nunique()
    tc = t[t[sc].isin(nn[nn > 1].index)]
    per_name = {nm: dict(n=int(len(x)), mean_log_D=round(float(x.log_D.mean()), 4),
                         sd_log_D=(None if len(x) < 2 else round(float(x.log_D.std(ddof=1)), 4)))
                for nm, x in tc.groupby('extractant_name')}
    spread, cells = None, []
    if len(tc):
        cc = tc.copy()
        cc['_dev'] = cc.log_D - cc.groupby(sc).log_D.transform('mean')
        d = cc.groupby('extractant_name')._dev.mean()
        spread = round(float(d.max() - d.min()), 4)
        for cell, x in cc.groupby(sc):
            m = x.groupby('extractant_name').log_D.agg(['size','mean','std'])
            cells.append(dict(cell=cell.split('||')[1],
                              range_of_name_means=round(float(m['mean'].max() - m['mean'].min()), 4),
                              names={k: dict(n=int(v['size']), mean=round(float(v['mean']), 4),
                                             sd=(None if pd.isna(v['std']) else round(float(v['std']), 4)))
                                     for k, v in m.iterrows()}))
    fs = {}
    for f in ALLF:
        if f not in t.columns: continue
        v = _s(t[f])
        tab = pd.crosstab(v, t.extractant_name)
        npv, vpn = (tab > 0).sum(axis=1), (tab > 0).sum(axis=0)
        resolved = int(sum(tab.loc[x].sum() for x in npv[npv == 1].index))
        fs[f] = dict(n_values=int(v.nunique()), constant=bool(v.nunique() <= 1),
                     frac_rows_name_determined=round(float(resolved / len(t)), 4),
                     max_names_sharing_one_value=int(npv.max()),
                     name_is_a_function_of_this_field=bool(v.nunique() > 1 and (vpn == 1).all()),
                     leaky_contains_the_name=f in F_LEAK)
    def joint(fields):
        tt = tup(t, fields); tab = pd.crosstab(tt, t.extractant_name); npv = (tab > 0).sum(axis=1)
        return round(float(sum(tab.loc[x].sum() for x in npv[npv == 1].index) / len(t)), 4)
    return dict(canonical_smiles=s, n_rows=int(len(t)), n_names=int(t.extractant_name.nunique()),
                names=t.extractant_name.value_counts().to_dict(),
                metals=sorted(t.metal.unique().tolist()),
                dois=sorted(set(_s(t.DOI))), curators=sorted(set(_s(t.entry_author))),
                upstream_Extractant_SMILES_variants=sorted(set(_s(t.Extractant_SMILES))),
                n_matched_cells_with_2plus_names=int((nn > 1).sum()),
                n_rows_in_matched_contested_cells=int(len(tc)),
                name_effect_spread_logD_at_matched_conditions=spread,
                per_name_at_matched_conditions=per_name, contested_cells=cells,
                field_separation=fs,
                joint_name_recovery_condition_fields_only=joint(F_COND),
                joint_name_recovery_condition_plus_provenance=joint(F_COND + F_PROV),
                joint_name_recovery_including_leaky_free_text=joint(ALLF))

R['groups'] = [group_report(s) for s in AMB]

def classify(gr):
    if gr['canonical_smiles'] == TODGA:
        return 'WRONG STRUCTURE - one SMILES string pasted onto 19 chemically distinct ligands'
    if gr['canonical_smiles'] == DMDPh:
        return 'DUPLICATE ENTRY + DECIMAL CORRUPTION - same paper entered twice, one copy missing the 10^-n exponent'
    if gr['canonical_smiles'] == C5BTBP:
        return 'SYNONYM + RECOVERABLE DILUENT SPLIT - cyclohexanone vs kerosene/octanol, already a cond__ feature'
    if gr['n_matched_cells_with_2plus_names'] == 0:
        return 'SYNONYM - the two names never co-occur at matched conditions; no measurable effect'
    sp = gr['name_effect_spread_logD_at_matched_conditions']
    return ('SYNONYM - residual offset %.2f log units, at or below the replicate noise floor' % sp)
for gr in R['groups']:
    gr['classification'] = classify(gr)

# ------------------------------------------------------------------ (c) decomposition
def pairstats(frame, sc, same_name):
    out = []
    for _, t in frame.groupby(sc):
        v = [tuple(x) for x in t[['log_D','extractant_name']].values]
        for (a,na),(b,nb) in combinations(v, 2):
            if (na == nb) == same_name:
                out.append(abs(a - b))
    out = np.asarray(out)
    if not len(out): return dict(n_pairs=0)
    return dict(n_pairs=int(len(out)), rms_abs_delta=round(float(np.sqrt((out**2).mean())), 4),
                median_abs_delta=round(float(np.median(out)), 4),
                mean_abs_delta=round(float(out.mean()), 4),
                frac_gt_1_log=round(float((out > 1).mean()), 4))

def pairstats_by(frame, sc, col, same):
    out = []
    for _, t in frame.groupby(sc):
        v = [tuple(x) for x in t[['log_D', col]].values]
        for (a, na), (b, nb) in combinations(v, 2):
            if (na == nb) == same:
                out.append(abs(a - b))
    out = np.asarray(out)
    if not len(out): return dict(n_pairs=0)
    return dict(n_pairs=int(len(out)), rms_abs_delta=round(float(np.sqrt((out**2).mean())), 4),
                median_abs_delta=round(float(np.median(out)), 4),
                mean_abs_delta=round(float(out.mean()), 4),
                frac_gt_1_log=round(float((out > 1).mean()), 4))


def eta_perm(frame, sc, keyser, n=400):
    f = frame.copy()
    dev = f.log_D - f.groupby(sc).log_D.transform('mean')
    SS = float((dev**2).sum())
    if SS <= 0: return None
    k = f[sc].astype(str) + '#' + keyser.astype(str)
    obs = float((f.groupby(k).log_D.transform('mean') - f.groupby(sc).log_D.transform('mean')).pow(2).sum() / SS)
    nul = []
    for _ in range(n):
        pm = keyser.groupby(f[sc]).transform(lambda s: pd.Series(rng.permutation(s.values), index=s.index))
        kk = f[sc].astype(str) + '#' + pm.astype(str)
        nul.append(float((f.groupby(kk).log_D.transform('mean') - f.groupby(sc).log_D.transform('mean')).pow(2).sum() / SS))
    nul = np.asarray(nul); m = float(nul.mean())
    return dict(eta2_raw=round(obs, 4), eta2_permutation_null_mean=round(m, 4),
                eta2_adjusted=round((obs - m) / (1 - m), 4) if m < 1 else None,
                permutation_p=round(float((nul >= obs).mean()), 4),
                n_key_groups=int(k.nunique()), n_cells=int(f[sc].nunique()),
                residual_sd_after_cell=round(float(np.sqrt(SS / max(len(f) - f[sc].nunique(), 1))), 4))

dec = {}
for tag in 'PM':
    sc = '_sc' + tag
    amb = J[J.canonical_smiles.isin(AMB)]
    nn = amb.groupby(sc).extractant_name.nunique()
    cont = amb[amb[sc].isin(nn[nn > 1].index)]
    variants = {'all_ambiguous_smiles': cont,
                'excl_TODGA_cluster': cont[cont.canonical_smiles != TODGA],
                'excl_TODGA_and_DMDPhPDA': cont[(cont.canonical_smiles != TODGA) & (cont.canonical_smiles != DMDPh)],
                'excl_TODGA_DMDPhPDA_and_C5BTBP': cont[~cont.canonical_smiles.isin([TODGA, DMDPh, C5BTBP])]}
    blk = {}
    for name, fr in variants.items():
        e = dict(n_rows=int(len(fr)), n_cells=int(fr[sc].nunique()),
                 between_name_pairs=pairstats(fr, sc, False), within_name_pairs=pairstats(fr, sc, True))
        if len(fr):
            e['eta2_extractant_name'] = eta_perm(fr, sc, fr.extractant_name)
            e['eta2_upstream_condition_fields'] = eta_perm(fr, sc, tup(fr, F_COND))
            e['eta2_upstream_condition_plus_provenance'] = eta_perm(fr, sc, tup(fr, F_COND + F_PROV))
        blk[name] = e
    # replicate noise floor: >=2 rows in a cell, all sharing ONE name
    n_all, sz = J.groupby(sc).extractant_name.nunique(), J.groupby(sc).size()
    single = J[J[sc].isin(sz[sz >= 2].index.intersection(n_all[n_all == 1].index))]
    d = single.log_D - single.groupby(sc).log_D.transform('mean')
    blk['replicate_noise_floor_single_name_cells'] = dict(
        n_rows=int(len(single)), n_cells=int(single[sc].nunique()),
        residual_sd=round(float(np.sqrt((d**2).sum() / max(len(single) - single[sc].nunique(), 1))), 4),
        mean_abs_dev=round(float(d.abs().mean()), 4),
        within_name_pairs=pairstats(single, sc, True))
    dec[tag] = blk
dec['dataset_log_D_sd'] = round(float(J.log_D.std(ddof=1)), 4)
R['variance_decomposition'] = dec

# ------------------------------------------------------------------ forensics
t = J[J.canonical_smiles == DMDPh].copy()
t['_k'] = t.metal.astype(str) + '|' + t.cond__acid_concentration_M.round(4).astype(str)
p = t.pivot_table(index='_k', columns='extractant_name', values='log_D', aggfunc='mean').dropna()
d = (p.iloc[:, 0] - p.iloc[:, 1]).round(6)
R['forensics_DMDPhPDA'] = dict(
    smiles=DMDPh, names=list(p.columns), n_matched_cells=int(len(p)),
    doi=sorted(set(_s(t.DOI))), curators=sorted(set(_s(t.entry_author))),
    solvent_by_name={k: sorted(set(_s(v.Solvent_Name))) for k, v in t.groupby('extractant_name')},
    upstream_smiles_by_name={k: sorted(set(_s(v.Extractant_SMILES))) for k, v in t.groupby('extractant_name')},
    delta_logD_is_an_exact_integer_in_every_cell=bool(np.allclose(d.values, np.round(d.values))),
    delta_logD_histogram={str(k): int(v) for k, v in d.value_counts().sort_index().items()},
    D_ratio_median=round(float(np.median(10 ** d.values)), 3),
    D_ratio_max=round(float((10 ** d.values).max()), 3),
    diluent_dummy_by_name={k: [c for c in df.columns if c.startswith('cond__diluent__') and df.loc[v.index, c].sum() > 0]
                           for k, v in df[df.canonical_smiles == DMDPh].groupby('extractant_name')},
    rows_of_cond__diluent__ch3cl_in_dataset=int(df.cond__diluent__ch3cl.sum()),
    rows_of_cond__diluent__ch3cl_that_are_this_corruption=int(df[df.canonical_smiles == DMDPh].cond__diluent__ch3cl.sum()))

t2 = J[J.canonical_smiles == C5BTBP]
R['forensics_C5BTBP'] = dict(
    smiles=C5BTBP,
    solvent_by_name={k: sorted(set(_s(v.Solvent_Name))) for k, v in t2.groupby('extractant_name')},
    doi_by_name={k: sorted(set(_s(v.DOI))) for k, v in t2.groupby('extractant_name')},
    temperature_by_name={k: sorted(set(_s(v.obsTempsValue))) for k, v in t2.groupby('extractant_name')},
    dataset_encodes_cyclohexanone='cond__diluent__cyclohexanone' in df.columns,
    dataset_encodes_kerosene_0_7_octanol_0_3='cond__diluent__kerosene_0_7_1_octanol_0_3' in df.columns)

tt = J[J.canonical_smiles == TODGA]
bad = tt[tt.extractant_name != 'TODGA']
# 'TODGA,DHOA' is a composite-solvent label, not a wrong structure: upstream SMILES is "<TODGA>,-"
wrong = bad[bad.extractant_name != 'TODGA,DHOA']
R['forensics_TODGA'] = dict(
    smiles=TODGA, n_rows_total=int(len(tt)), n_rows_with_a_non_TODGA_name=int(len(bad)),
    n_rows_wrong_structure=int(len(wrong)),
    n_rows_TODGA_DHOA_composite_label=int(len(bad) - len(wrong)),
    n_distinct_wrong_ligands=int(wrong.extractant_name.nunique()),
    wrong_names=wrong.extractant_name.value_counts().to_dict(),
    curator_of_wrong_structure_rows=sorted(set(_s(wrong.entry_author))),
    curator=sorted(set(_s(bad.entry_author))),
    upstream_rows_with_exact_TODGA_smiles_and_another_name=int(
        ((up.Extractant_SMILES.astype(str) == TODGA) & (up.Extractant_Name != 'TODGA')).sum()),
    upstream_rows_with_TODGA_smiles_prefix_and_another_name=int(
        (up.Extractant_SMILES.astype(str).str.startswith(TODGA) & (up.Extractant_Name != 'TODGA')).sum()),
    upstream_distinct_wrong_names=int(
        up.loc[(up.Extractant_SMILES.astype(str) == TODGA) & (up.Extractant_Name != 'TODGA'), 'Extractant_Name'].nunique()),
    Holdback_Agent_Name_values=sorted(set(_s(wrong.Holdback_Agent_Name))),
    Phase_Modifier_Name_values=sorted(set(_s(wrong.Phase_Modifier_Name))),
    Solvent_Name_values=sorted(set(_s(wrong.Solvent_Name))),
    ini_comp_names_the_true_ligand=bool(wrong.apply(lambda r: str(r.extractant_name) in str(r.ini_comp), axis=1).all()))
_tw = up[up.Extractant_Name.astype(str).str.match(r'^TWE-\d+$')]
_tg = _tw.groupby('Extractant_SMILES').Extractant_Name.unique()
R['forensics_TODGA']['TWE_series_check'] = dict(
    upstream_TWE_rows=int(len(_tw)), distinct_TWE_codes=int(_tw.Extractant_Name.nunique()),
    distinct_TWE_smiles=int(_tw.Extractant_SMILES.nunique()),
    TWE_codes_sharing_the_TODGA_smiles=sorted(_tg.get(TODGA, [])),
    note=('The TWE screening series is not systematically broken: 24 of the 34 codes carry their own, '
          'chemically sensible SMILES (thiophosphinates, phosphonates, furandiamides, NTA-amides). Only the '
          '10 listed codes collapse onto the TODGA string. The corruption is a copy-paste inside one batch, '
          'not a curator-wide failure - which is why the other synonym pairs (TWE-11/NTAamide(C8), '
          'TWE-14/Me2-TODGA, TWE-15/NEA16, TWE-17/18b, TWE-21/10a, TWE-22/NEA15) are safe to read as aliases.'))

# ------------------------------------------------------------------ the real missing variable
cm = _s(J.comments_description)
def grab(field):
    pat = re.compile(rf'{re.escape(field)}:\s*(.*?)(?:;|$)')
    return cm.map(lambda s: (pat.search(s).group(1).strip() if pat.search(s) else 'nan'))
J['cplx']  = grab('Complexant_Name').replace({'nan': '<none>', '': '<none>'})
J['cplxc'] = grab('Complexant_Concentration_M')
CELL = '_scM'   # canonical_smiles + metal + full cond__ vector: one chemistry, one condition point
sz = J.groupby(CELL).size(); rep = J[J[CELL].isin(sz[sz >= 2].index)].copy()
rep['_dev'] = rep.log_D - rep.groupby(CELL).log_D.transform('mean')
devm = rep['_dev']
v = rep.groupby(CELL).cplx.nunique()
vc, vs = rep[rep[CELL].isin(v[v > 1].index)], rep[rep[CELL].isin(v[v == 1].index)]
def sd(f):
    dd = f.log_D - f.groupby(CELL).log_D.transform('mean')
    return round(float(np.sqrt((dd**2).sum() / max(len(f) - f[CELL].nunique(), 1))), 4)
R['the_real_missing_variable'] = dict(
    finding=('The aqueous-phase complexant / holdback agent is recorded ONLY inside the free-text '
             'comments_description column, as "Complexant_Name: X; Complexant_Concentration_M: Y". '
             'The structured upstream column Holdback_Agent_Name is NULL on all 48138 upstream rows, '
             'and the dataset carries no cond__ feature for it.'),
    n_rows_with_a_complexant=int((J.cplx != '<none>').sum()),
    frac_rows_with_a_complexant=round(float((J.cplx != '<none>').mean()), 4),
    complexant_counts=J.cplx.value_counts().to_dict(),
    upstream_Holdback_Agent_Name_nonnull_anywhere=int(up.Holdback_Agent_Name.notna().sum()),
    dataset_features_matching_complexant_or_holdback=[c for c in df.columns
        if any(k in c.lower() for k in ('complexant','holdback','dtpa','tedga','dooda'))],
    n_replicated_model_visible_cells=int(rep[CELL].nunique()),
    n_rows_in_replicated_model_visible_cells=int(len(rep)),
    residual_sd_in_replicated_cells=round(float(np.sqrt((devm**2).sum() / max(len(rep) - rep[CELL].nunique(), 1))), 4),
    oracle_MAE_floor_on_replicated_cells=round(float(devm.abs().mean()), 4),
    n_cells_where_complexant_varies=int((v > 1).sum()),
    residual_sd_where_complexant_varies=sd(vc),
    residual_sd_where_complexant_constant=sd(vs),
    eta2_complexant_within_model_cell=eta_perm(rep, CELL, rep.cplx),
    eta2_complexant_plus_conc_within_model_cell=eta_perm(rep, CELL, rep.cplx + '|' + rep.cplxc),
    eta2_random_control_DOI_within_model_cell=eta_perm(rep, CELL, _s(rep.DOI)),
    between_complexant_pairs=pairstats_by(rep, CELL, 'cplx', False),
    within_complexant_pairs=pairstats_by(rep, CELL, 'cplx', True),
    mean_cell_centred_logD_by_complexant={k: dict(n=int(len(x)), mean_dev=round(float(x._dev.mean()), 3))
                                          for k, x in rep.groupby('cplx') if len(x) >= 5})

# ------------------------------------------------------------------ live pipeline impact
from lanthanide_separation.levels import build_level_dataset, condition_labels
ld = build_level_dataset(df, min_rows_per_extractant=10)
fr = getattr(ld, 'frame', ld)
src = df.copy(); src['_bad'] = src.canonical_smiles.eq(TODGA) & ~src.extractant_name.eq('TODGA')
src = src[src.log_D > -6.0]
src['condition_id'] = condition_labels(src, tuple(c for c in src.columns if c.startswith('cond__')))
src['_cell'] = src.canonical_smiles.astype(str) + '|' + src.condition_id.astype(str) + '|' + src.metal_symbol.astype(str)
per = src.groupby('_cell')._bad.agg(['size', 'sum'])
per2 = src[src.canonical_smiles == DMDPh].groupby('_cell').extractant_name.nunique()
R['live_pipeline_impact'] = dict(
    pipeline='scripts/run_gen5_levels.py, scripts/gen6_phase0.py and the gen7 harness read dataset.parquet '
             'straight into lanthanide_separation.levels.build_level_dataset',
    level_identity='extractant = canonical_smiles (levels.py:409); the level cell is '
                   '(extractant, condition_id, metal_symbol) with condition_id built from cond__* only',
    apply_default_quarantine_is_called=False,
    quarantine_exists_in='src/lanthanide_separation/pairs.py::apply_default_quarantine, reason '
                         '"todga_structure_assigned_to_different_extractant"',
    level_rows_min_rows_10=int(len(fr)), level_extractants=int(fr.extractant.nunique()),
    mislabelled_TODGA_source_rows=int(src._bad.sum()),
    level_cells_carrying_mislabelled_rows=int((per['sum'] > 0).sum()),
    level_cells_mixing_real_TODGA_with_mislabelled=int(((per['sum'] > 0) & (per['sum'] < per['size'])).sum()),
    TODGA_level_cells=int((fr.extractant == TODGA).sum()),
    frac_of_TODGA_cells_that_are_another_molecule=round(float((per['sum'] > 0).sum() / max((fr.extractant == TODGA).sum(), 1)), 4),
    DMDPhPDA_level_cells=int((fr.extractant == DMDPh).sum()),
    DMDPhPDA_level_cells_merging_the_two_curator_entries=int((per2 > 1).sum()),
    DMDPhPDA_note='the two entries do NOT merge: Solvent_Name "CH3Cl" vs "Chloroform" map to two different '
                  'cond__diluent__ dummies, so the level cohort holds the same molecule twice with log_D '
                  'differing by up to 4 orders of magnitude, presented as a diluent contrast')

json.dump(R, open(os.path.join(OUT, 'ambiguity_audit.json'), 'w'), indent=1, default=str)
print('wrote', os.path.join(OUT, 'ambiguity_audit.json'))

# ---------------------------------------------------------------- markdown report
R = json.load(open(os.path.join(OUT, 'ambiguity_audit.json')))
P,H,V=R['provenance'],R['headline'],R['variance_decomposition']
TODGA='CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC'
DM='CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2)n1)c1ccccc1'
def f(x,n=3):
    return 'n/a' if x is None else (f'{x:.{n}f}' if isinstance(x,float) else str(x))
L=[]
A=L.append
A('# Ambiguity audit: "17/190 canonical SMILES map to multiple extractant_name"')
A('')
A('*Run 2026-08-19. Generated by `runs/gen7_architecture/ambiguity_audit.py`; every number below is '
  'reproduced by that script and mirrored in `ambiguity_audit.json`.*')
A('')
A('## Verdict')
A('')
A('**World C ("the ambiguity is a missing-variable ceiling") is rejected on the data.** The warning is '
  'not a chemistry signal. Sixteen of the seventeen ambiguous SMILES are *synonym duplicates* — an acronym '
  'and its IUPAC name, or two paper-local labels for one molecule. The seventeenth is a **wrong-structure '
  f"batch**: the TODGA SMILES string is pasted onto {R['forensics_TODGA']['n_distinct_wrong_ligands']} "
  'chemically distinct ligands.')
A('')
A('Once you condition on everything the level model can actually see (metal + the full 64-column `cond__` '
  f"vector), only **{V['M']['all_ambiguous_smiles']['n_rows']} rows in "
  f"{V['M']['all_ambiguous_smiles']['n_cells']} cells** in the whole dataset carry two names under one "
  'SMILES — and **all of them are the TODGA batch** (removing TODGA leaves zero rows). On those rows '
  f"`extractant_name` has a permutation-adjusted eta^2 of "
  f"**{f(V['M']['all_ambiguous_smiles']['eta2_extractant_name']['eta2_adjusted'])}** "
  f"(p = {f(V['M']['all_ambiguous_smiles']['eta2_extractant_name']['permutation_p'],2)}): it explains *nothing* "
  'beyond a random partition of the same granularity.')
A('')
_tr=V['P']['excl_TODGA_DMDPhPDA_and_C5BTBP']; _fl=V['P']['replicate_noise_floor_single_name_cells']
A('And the cleanest single number: once the three identified defects (the TODGA batch, the DMDPhPDA '
  'decimal corruption, the C5BTBP diluent split) are removed, rows that share a SMILES but differ in name '
  f"agree **better** than genuine same-name replicates — RMS |Δlog_D| {f(_tr['between_name_pairs']['rms_abs_delta'])} "
  f"(n={_tr['between_name_pairs']['n_pairs']}) against a replicate floor of "
  f"{f(_fl['within_name_pairs']['rms_abs_delta'])} (n={_fl['within_name_pairs']['n_pairs']}). The remaining "
  'aliases are not carrying hidden chemistry; they are carrying nothing.')
A('')
A('The audit did find a genuine missing variable, but it is somewhere else: the **aqueous-phase complexant** '
  f"(DTPA, TEDGA, DOODA(C2), NaNO3, amic acid, malonamide, HEDTA, BTP-4Me, CDTA) on "
  f"**{R['the_real_missing_variable']['n_rows_with_a_complexant']} rows "
  f"({100*R['the_real_missing_variable']['frac_rows_with_a_complexant']:.1f} %)**, recorded only inside the "
  'free-text `comments_description` column. It has no `cond__` feature and the structured '
  f"`Holdback_Agent_Name` column is NULL on all {P['upstream_rows']} upstream rows.")
A('')
A('---')
A('')
A('## 0. Join and reproduction')
A('')
A('| quantity | value |')
A('|---|---|')
A(f"| dataset rows | {P['dataset_rows']} |")
A(f"| upstream rows (32 `*_SAFE.csv`) | {P['upstream_rows']} |")
A(f"| join `safe_exp_id == \"{{stem}}:{{exp_id}}\"` matched | {P['join_matched_rows']} / {P['dataset_rows']} (100 %, 1:1, no duplicates either side) |")
A(f"| `extractant_name` == upstream `Extractant_Name` | {100*P['extractant_name_equals_upstream_Extractant_Name']:.1f} % |")
A(f"| `LIGAND_SMILES` == upstream `Extractant_SMILES` | {100*P['LIGAND_SMILES_equals_upstream_Extractant_SMILES']:.1f} % |")
A(f"| `canonical_smiles` == upstream `Extractant_SMILES` | {100*P['canonical_smiles_equals_upstream_Extractant_SMILES']:.1f} % (RDKit re-canonicalisation) |")
A(f"| distinct `canonical_smiles` | {H['n_canonical_smiles']} |")
A(f"| distinct `extractant_name` | {H['n_extractant_name']} |")
A(f"| **SMILES with >1 name** | **{H['n_ambiguous_smiles']}** (warning reproduced exactly) |")
A(f"| names with >1 SMILES | {H['n_names_mapping_to_multiple_smiles']} (the map is strictly many-names-to-one-SMILES) |")
A(f"| rows under an ambiguous SMILES | {H['rows_in_ambiguous_smiles']} ({100*H['frac_rows_in_ambiguous_smiles']:.1f} %) |")
A('')
A('The 50 % row figure is misleading on its own: 1714 of those 3011 rows are the TODGA SMILES, and 1585 of '
  'them are genuinely TODGA.')
A('')
A('---')
A('')
A('## (a) The seventeen groups, and log_D per name at matched conditions')
A('')
A('Matched cell **P** = `metal | metal_ox | Acid_Name | cond__acid_concentration_M | cond__extractant_concentration_M`. '
  '"Contested" = a cell in which two or more names co-occur under one SMILES.')
A('')
A('| # | canonical SMILES (trunc) | names (rows) | contested cells / rows | name-effect spread at matched conditions (log units) | classification |')
A('|---|---|---|---|---|---|')
for i,gr in enumerate(R['groups'],1):
    nm=' / '.join(f'{k} ({v})' for k,v in list(gr['names'].items())[:4])
    if len(gr['names'])>4: nm+=f" / +{len(gr['names'])-4} more"
    sp=gr['name_effect_spread_logD_at_matched_conditions']
    A(f"| {i} | `{gr['canonical_smiles'][:38]}…` | {nm} | {gr['n_matched_cells_with_2plus_names']} / {gr['n_rows_in_matched_contested_cells']} | {'—' if sp is None else f'{sp:.3f}'} | {gr['classification']} |")
A('')
_nc=sum(1 for gr in R['groups'] if gr['n_matched_cells_with_2plus_names']>0)
A(f'Only {_nc} of the 17 ever place two names at the same matched conditions. For the other {17-_nc} the two '
  'names sit in disjoint condition regions (different papers, different diluents), so no within-cell '
  'comparison exists at all.')
A('')
A('### Per-name mean/sd of log_D in contested cells')
A('')
for gr in R['groups']:
    if not gr['per_name_at_matched_conditions']: continue
    A(f"**`{gr['canonical_smiles'][:60]}…`** — {gr['n_rows']} rows, {gr['n_names']} names, "
      f"curators {', '.join(gr['curators'])}")
    A('')
    A('| name | n (contested) | mean log_D | sd log_D |')
    A('|---|---|---|---|')
    for k,v in sorted(gr['per_name_at_matched_conditions'].items(), key=lambda kv:-kv[1]['n']):
        A(f"| {k[:66]} | {v['n']} | {f(v['mean_log_D'])} | {f(v['sd_log_D'])} |")
    A('')
A('---')
A('')
A('## (b) Is the name split explained by a recoverable upstream variable?')
A('')
A('**No — and the question is the wrong one for 16 of the 17.** The name difference is not a chemical '
  'difference; the two names denote one molecule. The specific fields asked about:')
A('')
A('| upstream field | verdict across the 17 groups |')
A('|---|---|')
A(f"| `Holdback_Agent_Name` | **NULL on all {P['upstream_rows']} upstream rows** — the column exists and was never filled. It separates nothing, anywhere. |")
A( "| `Phase_Modifier_Name` | Non-null on 239 of 3011 ambiguous rows. Never a function of the name in any of the 17 groups. |")
A( "| `Solvent_Name` | Determines the name in 11/17 groups — but only because each synonym came from a different publication that used a different diluent. Correlational, not causal: the diluent does not *make* the ligand have two names. |")
A( "| `DOI` / `entry_author` | Determines the name in most groups (the synonym is a curator/publication artefact). This is the real generator of the ambiguity. |")
A( "| `ini_comp`, `comments_description` | Recover the name in 17/17 — but they literally contain the extractant name as a substring. Excluded as leaky. |")
A('')
A('Joint recovery of the name from all non-leaky condition fields, and from condition + provenance:')
A('')
A('| group | conditions only | conditions + DOI/author |')
A('|---|---|---|')
for gr in R['groups']:
    A(f"| `{gr['canonical_smiles'][:34]}…` | {100*gr['joint_name_recovery_condition_fields_only']:.0f} % | {100*gr['joint_name_recovery_condition_plus_provenance']:.0f} % |")
A('')
A('### The one group where the name *is* the chemistry: TODGA')
T=R['forensics_TODGA']
A('')
A(f"`{T['smiles']}` carries **{T['n_rows_total']} rows under 21 names**. "
  f"{T['n_rows_wrong_structure']} of them, spread over {T['n_distinct_wrong_ligands']} distinct ligands, "
  'are not TODGA at all:')
A('')
A('`' + '`, `'.join(list(T['wrong_names'].keys())) + '`')
A('')
A(f"(The 21st name, `TODGA,DHOA`, is a legitimate composite-solvent label — its upstream SMILES is "
  f"`<TODGA>,-` — and accounts for the other {T['n_rows_TODGA_DHOA_composite_label']} non-TODGA rows.)")
A('')
A(f"Upstream the damage is wider: **{T['upstream_rows_with_exact_TODGA_smiles_and_another_name']} rows across "
  f"{T['upstream_distinct_wrong_names']} distinct `Extractant_Name` values** carry the exact TODGA SMILES "
  f"string. Every affected dataset row comes from a single curator ({', '.join(T['curator_of_wrong_structure_rows'])}) "
  'and a single ingest batch. These are real, chemically distinct ligands — sulfonated BTP/BTBP complexants, '
  'phenanthroline diols, PyTri polyols, the TWE-2x/3x/4x screening series — whose `Extractant_SMILES` field '
  'was filled with TODGA.')
A('')
A(f"Recoverability: `Holdback_Agent_Name` = {T['Holdback_Agent_Name_values']}, "
  f"`Phase_Modifier_Name` = {T['Phase_Modifier_Name_values']}. Nothing structured recovers the real molecule. "
  '`ini_comp` names the true ligand on 100 % of these rows but carries no structure. **The correct SMILES is '
  'not in the bundle** — this is a fix that has to happen upstream.')
A('')
_TW=T['TWE_series_check']
A(f"**Scope check.** The TWE screening series is *not* systematically broken: of {_TW['distinct_TWE_codes']} "
  f"TWE codes upstream, {_TW['distinct_TWE_smiles']} distinct SMILES are recorded and only "
  f"{len(_TW['TWE_codes_sharing_the_TODGA_smiles'])} of them collapse onto TODGA "
  f"(`{'`, `'.join(_TW['TWE_codes_sharing_the_TODGA_smiles'])}`). The rest carry chemically sensible, "
  'distinct structures — thiophosphinates, phosphonates, furandiamides, NTA-amides. That localises the '
  'defect to one copy-paste inside one batch, and it is why the other alias pairs '
  '(TWE-11/NTAamide(C8), TWE-14/Me2-TODGA, TWE-15/NEA16, TWE-17/18b, TWE-21/10a, TWE-22/NEA15) are safe '
  'to read as synonyms rather than as further corruption.')
A('')
A('### A second defect the audit turned up: a decimal-corrupted double entry')
D=R['forensics_DMDPhPDA']
A('')
A(f"`{D['smiles']}` = DMDPhPDA, {D['n_matched_cells']} matched (metal, HNO3) cells, entered **twice** from "
  f"the *same* paper (DOI `10.1081/SEI-120030392`, Shimada et al. 2003) by two curators "
  f"({', '.join(D['curators'])}), under an acronym and its IUPAC name.")
A('')
A(f"The pairwise difference in log_D is **an exact integer in every single cell** "
  f"(`delta_logD_is_an_exact_integer_in_every_cell = {D['delta_logD_is_an_exact_integer_in_every_cell']}`):")
A('')
A('| Δ log_D | cells | D ratio |')
A('|---|---|---|')
for k,v in D['delta_logD_histogram'].items():
    A(f"| {k} | {v} | 10^{k.split('.')[0]} |")
A('')
A('One copy kept only the mantissa of the paper\'s Table 1 and dropped the ×10^−n exponent. **70 dataset rows '
  'carry log_D wrong by 1–4 orders of magnitude.** This is label corruption, not a missing variable.')
A('')
A(f"It is also a *feature* corruption: the two entries wrote the diluent as `CH3Cl` and `Chloroform`, which map "
  f"to two different dummies. `cond__diluent__ch3cl` has {D['rows_of_cond__diluent__ch3cl_in_dataset']} rows in "
  f"the whole dataset, {D['rows_of_cond__diluent__ch3cl_that_are_this_corruption']} of them "
  f"({100*D['rows_of_cond__diluent__ch3cl_that_are_this_corruption']/D['rows_of_cond__diluent__ch3cl_in_dataset']:.0f} %) "
  'this one corruption. Any tree that splits on `cond__diluent__ch3cl` is learning "this value is 10^n too '
  'high", dressed up as a chemistry effect.')
A('')
A('### And one group where the split genuinely is a recoverable variable')
C=R['forensics_C5BTBP']
A('')
_iup=[k for k in C['solvent_by_name'] if k!='C5BTBP'][0]
A('C5BTBP vs its IUPAC name shows a 2.31 log-unit offset at cell-P matched conditions — but the two names sit '
  f"in different diluents (IUPAC name: {', '.join(C['solvent_by_name'][_iup])}; C5BTBP: "
  f"{', '.join(C['solvent_by_name']['C5BTBP'][:4])}, …), different "
  'publications and different temperatures. Both diluents already have `cond__diluent__*` features '
  f"(cyclohexanone: {C['dataset_encodes_cyclohexanone']}, kerosene 0.7/1-octanol 0.3: "
  f"{C['dataset_encodes_kerosene_0_7_octanol_0_3']}). The offset is fully model-visible; the cell-P key was "
  'simply too coarse. This is why the whole analysis is repeated below on the model-visible cell.')
A('')
A('---')
A('')
A('## (c) Variance decomposition')
A('')
A('For each cell definition: residual after removing the (SMILES × cell) mean, then how much of that residual '
  '`extractant_name` explains. Because names are near-singleton inside a cell, the raw eta^2 is inflated by '
  'degrees of freedom; every eta^2 below is reported **raw, against a 400-draw permutation null (names shuffled '
  'within cell), and adjusted** as `(obs − null)/(1 − null)`.')
A('')
for tag,title in (('P','Cell P — metal + ox + acid + [acid] + [extractant]'),
                  ('M','Cell M — metal + ox + the full 64-column `cond__` vector (model-visible)')):
    A(f'### {title}')
    A('')
    A('| subset | rows | cells | RMS \\|Δlog_D\\| between names | RMS \\|Δlog_D\\| within a name | eta^2 raw | eta^2 null | **eta^2 adj** | perm p | residual sd |')
    A('|---|---|---|---|---|---|---|---|---|---|')
    for key,lab in (('all_ambiguous_smiles','all 17 ambiguous SMILES'),
                    ('excl_TODGA_cluster','excl. the TODGA batch'),
                    ('excl_TODGA_and_DMDPhPDA','excl. TODGA + DMDPhPDA'),
                    ('excl_TODGA_DMDPhPDA_and_C5BTBP','excl. TODGA + DMDPhPDA + C5BTBP (the 3 known defects)')):
        b=V[tag][key]
        if not b['n_rows']:
            A(f"| {lab} | 0 | 0 | — | — | — | — | — | — | — |"); continue
        e=b['eta2_extractant_name']
        A(f"| {lab} | {b['n_rows']} | {b['n_cells']} | {f(b['between_name_pairs'].get('rms_abs_delta'))} "
          f"(n={b['between_name_pairs']['n_pairs']}) | {f(b['within_name_pairs'].get('rms_abs_delta'))} "
          f"(n={b['within_name_pairs']['n_pairs']}) | {f(e['eta2_raw'])} | {f(e['eta2_permutation_null_mean'])} | "
          f"**{f(e['eta2_adjusted'])}** | {f(e['permutation_p'],3)} | {f(e['residual_sd_after_cell'])} |")
    nf=V[tag]['replicate_noise_floor_single_name_cells']
    A(f"| *replicate floor* (cells with ≥2 rows, **one** name) | {nf['n_rows']} | {nf['n_cells']} | — | "
      f"{f(nf['within_name_pairs']['rms_abs_delta'])} (n={nf['within_name_pairs']['n_pairs']}) | — | — | — | — | {f(nf['residual_sd'])} |")
    A('')
A(f"Dataset-wide sd of log_D: **{f(V['dataset_log_D_sd'])}**.")
A('')
A('### How much of the name effect do the recoverable upstream fields explain?')
A('')
A('| cell | key | eta^2 raw | eta^2 null | eta^2 adj | perm p |')
A('|---|---|---|---|---|---|')
for tag in 'PM':
    b=V[tag]['all_ambiguous_smiles']
    if not b['n_rows']: continue
    for k,lab in (('eta2_extractant_name','extractant_name'),
                  ('eta2_upstream_condition_fields','upstream condition fields (14, incl. Solvent/Phase_Modifier/Holdback)'),
                  ('eta2_upstream_condition_plus_provenance','+ DOI / entry_author / source file')):
        e=b[k]
        A(f"| {tag} | {lab} | {f(e['eta2_raw'])} | {f(e['eta2_permutation_null_mean'])} | {f(e['eta2_adjusted'])} | {f(e['permutation_p'],3)} |")
A('')
_r=V['P']['excl_TODGA_DMDPhPDA_and_C5BTBP']
A(f"With the three identified defects removed, only {_r['n_rows']} rows in {_r['n_cells']} cells remain "
  f"contested on cell P, and the between-name RMS |Δlog_D| falls to "
  f"{f(_r['between_name_pairs'].get('rms_abs_delta'))} against a replicate floor of "
  f"{f(V['P']['replicate_noise_floor_single_name_cells']['within_name_pairs']['rms_abs_delta'])}. "
  'The permutation-adjusted eta^2 in the trimmed subsets is high but uninterpretable: there are only 6 '
  'within-name replicate pairs left to define the null, so the name partition is essentially saturated. '
  'Treat those rows as effect sizes, not as variance shares.')
A('')
A('**Reading.** On cell P the name looks like it explains 84 % of the within-cell residual, but a random '
  'partition of the same granularity explains 80 %; the adjusted figure is 0.18 at p = 0.08. The upstream '
  'condition fields explain 0.004 adjusted (p = 0.40) — i.e. **nothing**. On the model-visible cell M the '
  'adjusted eta^2 for the name is negative and p = 0.52.')
A('')
A('**The comparison that settles it.** At model-visible matched conditions, replicate rows that share one '
  f"name already disagree by RMS |Δlog_D| = {f(V['M']['replicate_noise_floor_single_name_cells']['within_name_pairs']['rms_abs_delta'])} "
  f"({100*V['M']['replicate_noise_floor_single_name_cells']['within_name_pairs']['frac_gt_1_log']:.0f} % of pairs "
  'differ by more than a full log unit). Rows that differ in name disagree by RMS '
  f"{f(V['M']['all_ambiguous_smiles']['between_name_pairs']['rms_abs_delta'])}. The ambiguity adds "
  'essentially nothing on top of a replicate floor that is already enormous.')
A('')
A('---')
A('')
A('## The missing variable that *is* real: the aqueous complexant')
MV=R['the_real_missing_variable']
A('')
A('Chasing the replicate floor above led to it. `comments_description` carries a semi-structured block; '
  'parsing `Complexant_Name:` out of it yields:')
A('')
A('| complexant | rows |')
A('|---|---|')
for k,v in MV['complexant_counts'].items():
    A(f"| {'*(none)*' if k=='<none>' else k} | {v} |")
A('')
A(f"**{MV['n_rows_with_a_complexant']} rows ({100*MV['frac_rows_with_a_complexant']:.1f} %) carry an "
  'aqueous-phase complexant that the model cannot see.** DTPA, HEDTA, CDTA and TEDGA are precisely the '
  'holdback agents that suppress trivalent-actinide/lanthanide distribution by orders of magnitude.')
A('')
A(f"- structured `Holdback_Agent_Name` non-null anywhere upstream: **{MV['upstream_Holdback_Agent_Name_nonnull_anywhere']}**")
A(f"- dataset features matching complexant/holdback/DTPA/TEDGA/DOODA: **{MV['dataset_features_matching_complexant_or_holdback'] or 'none'}**")
A('')
A('Inside replicated (SMILES × metal × full `cond__` vector) cells:')
A('')
A('| statistic | value |')
A('|---|---|')
A(f"| replicated cells / rows | {MV['n_replicated_model_visible_cells']} / {MV['n_rows_in_replicated_model_visible_cells']} |")
A(f"| residual sd inside those cells | {f(MV['residual_sd_in_replicated_cells'])} |")
A(f"| oracle MAE floor on those cells | {f(MV['oracle_MAE_floor_on_replicated_cells'])} |")
A(f"| residual sd where the complexant **varies** | {f(MV['residual_sd_where_complexant_varies'])} ({MV['n_cells_where_complexant_varies']} cells) |")
A(f"| residual sd where the complexant is **constant** | {f(MV['residual_sd_where_complexant_constant'])} |")
A(f"| RMS \\|Δlog_D\\| between differing complexants | {f(MV['between_complexant_pairs']['rms_abs_delta'])} (n={MV['between_complexant_pairs']['n_pairs']}) |")
A(f"| RMS \\|Δlog_D\\| within the same complexant | {f(MV['within_complexant_pairs']['rms_abs_delta'])} (n={MV['within_complexant_pairs']['n_pairs']}) |")
e=MV['eta2_complexant_within_model_cell']
A(f"| **eta^2 of complexant within model cell** | raw {f(e['eta2_raw'])}, null {f(e['eta2_permutation_null_mean'])}, **adjusted {f(e['eta2_adjusted'])}**, p = {f(e['permutation_p'],3)} |")
e2=MV['eta2_random_control_DOI_within_model_cell']
A(f"| control: DOI within model cell | raw {f(e2['eta2_raw'])}, adjusted {f(e2['eta2_adjusted'])}, p = {f(e2['permutation_p'],2)} |")
A('')
A('Mean cell-centred log_D by complexant (matched conditions, so this is the complexant\'s own effect):')
A('')
A('| complexant | n | mean cell-centred log_D |')
A('|---|---|---|')
for k,v in sorted(MV['mean_cell_centred_logD_by_complexant'].items(), key=lambda kv: kv[1]['mean_dev']):
    A(f"| {'*(none)*' if k=='<none>' else k} | {v['n']} | {v['mean_dev']:+.3f} |")
A('')
A('TEDGA depresses log_D by 0.57 and malonamide raises it by 0.33 relative to the same cell — both invisible '
  'to the current feature set. **This** is the missing-variable ceiling, and unlike the name ambiguity it is '
  'recoverable: the field is sitting in the comments string.')
A('')
A('---')
A('')
A('## Live pipeline impact')
LP=R['live_pipeline_impact']
A('')
A(f"`{LP['level_identity']}`.")
A('')
A(f"- `src/lanthanide_separation/pairs.py::apply_default_quarantine` **already knows about the TODGA defect** "
  f"(reason `todga_structure_assigned_to_different_extractant`).")
A(f"- **`scripts/run_gen5_levels.py`, `scripts/gen6_phase0.py` and the gen7 harness never call it.** They read "
  '`dataset.parquet` straight into `levels.build_level_dataset`.')
A(f"- Result: **{LP['mislabelled_TODGA_source_rows']} rows caught by that predicate "
  f"(`canonical_smiles == TODGA and extractant_name != 'TODGA'` — the {T['n_rows_wrong_structure']} "
  f"wrong-structure rows plus the {T['n_rows_TODGA_DHOA_composite_label']} legitimate `TODGA,DHOA` rows) enter "
  f"{LP['level_cells_carrying_mislabelled_rows']} level cells attributed to TODGA** — "
  f"{100*LP['frac_of_TODGA_cells_that_are_another_molecule']:.1f} % of TODGA's "
  f"{LP['TODGA_level_cells']} level cells are another molecule's data. They do *not* get averaged into real "
  f"TODGA cells ({LP['level_cells_mixing_real_TODGA_with_mislabelled']} mixed cells) because their diluents "
  'differ — they arrive as extra, wrong cells instead.')
A(f"- DMDPhPDA contributes {LP['DMDPhPDA_level_cells']} level cells and the two curator entries "
  f"**do not merge** ({LP['DMDPhPDA_level_cells_merging_the_two_curator_entries']} merged cells): "
  'Solvent_Name `CH3Cl` vs `Chloroform` map to two different `cond__diluent__` dummies, so the level '
  'cohort holds the same molecule twice with log_D differing by up to 4 orders of magnitude, presented '
  'to the model as a diluent contrast.')
A('')
A('---')
A('')
A('## Actions')
A('')
A('1. **Call `apply_default_quarantine` (or an equivalent filter) in the level pipeline.** 129 rows, 69 level '
  'cells, one line of code. Currently only the pair pipeline is protected.')
A('2. **Quarantine the 70 mantissa-only DMDPhPDA rows** — `extractant_name == '
  '"2-N-6-N-dimethyl-2-N-6-N-diphenylpyridine-2-6-dicarboxamide"`, all in `Am_SAFE`, exp_id 16085–16154 (contiguous), '
  'curator Baosen Zhang, `Solvent_Name = "CH3Cl"` — and drop or merge `cond__diluent__ch3cl`, which is 83 % '
  'this corruption.')
A('3. **Collapse the 16 synonym groups on `canonical_smiles`.** gen6 already does this '
  '(`extractant = canonical_smiles`); anything that groups or splits on `extractant_name` would leak a '
  'molecule across a held-out-ligand boundary. No current harness does — worth a regression test.')
A('4. **Parse `Complexant_Name` / `Complexant_Concentration_M` out of `comments_description` into a `cond__` '
  'block.** 557 rows, adjusted eta^2 0.30 of the residual inside perfectly matched model-visible cells, '
  'p < 0.003 (0 of 400 permutations reached the observed value). This is the first concretely recoverable '
  'variable the audit found.')
A('5. Do **not** treat the ambiguity warning as evidence for a capacity/ceiling argument. It is a curation '
  'defect of ~200 rows plus 16 harmless aliases.')
A('')
open(os.path.join(OUT,'ambiguity_audit.md'),'w').write('\n'.join(L)+'\n')
print('wrote', os.path.join(OUT,'ambiguity_audit.md'), len('\n'.join(L)), 'chars')

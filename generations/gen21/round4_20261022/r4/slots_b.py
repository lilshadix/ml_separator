"""Stage-1 block B (strain-architecture adaptation; brief section 6 'Strain architecture adaptation', slots S05-S10,
S22): a small CatBoost correction of B1's level residual at unread metals, without and with compact 3D geometry.

Support-consistent by construction: the correction is applied only at unread metals (read metals keep their
exact readings), so every pair prediction is a difference of one curve.  The target of a training row is the level
residual of an unread metal m under a training mask S:  t = y(m) - B1_S(m)  (B1's prediction at m given the reads;
read values are real measurements, so t is well defined).  Pairs are scored on these levels, exactly as B1.

Inputs (all label-free for the held-out cell; DATA_CONTRACT "Block B features"):
  metal / read-set geometry   x(m) = -radius CN8, 4f count, distances to the bracketing reads, interpolation fraction,
                              extrapolation side, B1's coherent Gd feature at m, round 3's 6 read-geometry features
  support summaries           slope of the bracketing segment, least-squares slope of the four reads (from the reads)
  chemistry (S05)             donor class one-hot (block A's rule) and 8 RDKit descriptors of the extractant
  geometry (S06)              for the complex of the cell's extractant with metal m (dataset with 3D structures,
                              features_by_safe_exp_id): mean Ln-donor distance, physical coordination number, metal
                              partial charge; each as g(m) and as g(m) - I_S[g(S)](m) (its anomaly relative to the reads);
                              missing geometry -> NaN (CatBoost's missing branch) with a coverage flag
  S09 control                 S06 with every chemotype's geometry taken from another chemotype (5 seeded draws)
  S22 control                 S06 with g replaced by the training-fitted per-metal mean of g (no ligand information)
Model: CatBoostRegressor(depth 4, learning_rate 0.05, l2_leaf_reg 3, RMSE, seed 0, 4 threads); the number of trees is
chosen from {0 (= B1), 100, 300, 1000} by inner chemotype-block pair loss (staged predictions of one 1000-tree fit per
block); weights: every unread-metal row of a training situation carries the cell's chemotype-balanced weight / masks
(B1's convention).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import paths
from . import refs as RF
from . import slots_a as SA
from r3 import corrector as CO  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k.data import N_LN  # noqa: E402
from gen13sep.metals import LANTHANIDES  # noqa: E402

TREE_GRID = (0, 100, 300, 1000)
CB_PARAMS = dict(depth=4, learning_rate=0.05, l2_leaf_reg=3.0, loss_function="RMSE", random_seed=0, thread_count=4,
                 verbose=False, allow_writing_files=False)
GEOM_COLS = ("ln_donor_distance_mean", "coordination_number__physical", "metal_partial_charge")
RD_DESC = ("MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount", "NumAromaticRings")
N_DRAWS = 5
GEO_SHUFFLE_SEEDS = tuple(20261223 + i for i in range(N_DRAWS))
Q = np.arange(N_LN, dtype=float) + np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])   # 4f count (Pm absent)


@lru_cache(maxsize=None)
def rdkit_desc(smiles: str) -> tuple:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    m = Chem.MolFromSmiles(smiles)
    return tuple(float(getattr(Descriptors, d)(m)) for d in RD_DESC)


ANIONS = ("nitrate", "water")
NAN_G = np.full(len(GEOM_COLS), np.nan)


@lru_cache(maxsize=1)
def _tables():
    """(cell_id, metal) -> geometry vector and -> composition (n_ligs, inner-sphere anion, n_fill) of the computed
    complex, from the cell's safe_exp_ids (metal from the archive record: the id prefix is the export file) and the
    complex's build_id in accepted_geometries.csv.  Label-free."""
    from gen20k import data as D
    root = paths.ROOT / "dataset with 3D structures"
    g = pd.read_parquet(root / "features" / "features_by_safe_exp_id.parquet",
                        columns=["safe_exp_id", "build_id"] + list(GEOM_COLS)).set_index("safe_exp_id")
    acc = pd.read_csv(root / "accepted_geometries.csv", usecols=["build_id", "n_ligs", "inner_sphere_anion", "n_fill"])
    comp_of = {b: (int(n), str(a), int(f)) for b, n, a, f in acc.itertuples(index=False)}
    arch = pd.read_parquet(paths.ROOT / "dataset_all_metals" / "clean" / "master_clean.parquet",
                           columns=["source_record_id", "metal_symbol"])
    metal_of = dict(zip(arch.source_record_id.astype(int), arch.metal_symbol))
    f = D.load_bench().frame
    geo, comp = {}, {}
    for cid, ids in zip(f.cell_id, f.safe_exp_ids):
        for x in str(ids).split(";"):
            if not x:
                continue
            m = LANTHANIDES.index(metal_of[int(x.split(":")[-1])])
            if x in g.index and (cid, m) not in geo:
                row = g.loc[x]
                row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                v = row[list(GEOM_COLS)].to_numpy(float)
                if np.isfinite(v).all():
                    geo[(cid, m)] = v
                    comp[(cid, m)] = comp_of.get(str(row["build_id"]))
    return geo, comp


def geometry_table() -> dict:
    return _tables()[0]


def composition_table() -> dict:
    return _tables()[1]


def geometry_shuffle_map(bench, seed: int) -> dict:
    """S09: a derangement pi over the chemotypes that carry geometry (rejection sampling with the sealed seed).  For a
    target (cell, metal) with its OWN finite geometry, the source is a cell of pi(chemotype) with finite geometry at
    the same metal, chosen at random (seeded) among that chemotype's cells; if pi(c) has none at that metal, a seeded
    donor chemotype other than c that has it.  Targets without their own geometry stay missing, so the missingness
    pattern and coverage flag equal S06's exactly."""
    geo = geometry_table()
    f = bench.frame
    chem_of = dict(zip(f.cell_id, f.chemotype.astype(str)))
    by_chem_m = {}
    for (cid, m) in sorted(geo):
        by_chem_m.setdefault((chem_of[cid], m), []).append(cid)
    chems = sorted({chem_of[cid] for cid, _ in geo})
    rng = np.random.default_rng(seed)
    while True:
        perm = [chems[i] for i in rng.permutation(len(chems))]
        if all(a != b for a, b in zip(chems, perm)):
            break
    pi = dict(zip(chems, perm))
    src, fallback = {}, 0
    for (cid, m) in sorted(geo):
        c = chem_of[cid]
        cands = by_chem_m.get((pi[c], m))
        if not cands:
            donors = sorted({ch for (ch, mm) in by_chem_m if mm == m and ch != c})
            cands = by_chem_m[(donors[int(rng.integers(len(donors)))], m)]
            fallback += 1
        src[(cid, m)] = cands[int(rng.integers(len(cands)))]
    return {"pi": pi, "src": src, "fallback": fallback}


def geometry_of(cell, m, mode: dict):
    """(geometry vector, composition) of the cell's complex at metal m under the slot's mode; NaN / None if missing.
    Every mode keeps S06's missingness: a target without its own geometry is missing in S06, S09 and S22."""
    geo, comp = geometry_table(), composition_table()
    key = (cell["cell_id"], m)
    if key not in geo:
        return NAN_G, None
    kind = mode.get("geometry")
    if kind == "shuffled":
        s = mode["map"]["src"][key]
        return geo[(s, m)], comp[(s, m)]
    if kind == "radius_only":
        return mode["metal_means"][m], mode["metal_comp"][m]
    return geo[key], comp[key]


def metal_means(rec, train_cells):
    """S22: training-fitted per-metal mean geometry and modal composition (no ligand information)."""
    geo, comp = geometry_table(), composition_table()
    means = np.full((N_LN, len(GEOM_COLS)), np.nan)
    modal = [None] * N_LN
    for m in range(N_LN):
        keys = [(c["cell_id"], m) for c in train_cells if (c["cell_id"], m) in geo]
        if keys:
            means[m] = np.mean(np.vstack([geo[k] for k in keys]), axis=0)
            cs = [comp[k] for k in keys if comp[k] is not None]
            modal[m] = max(sorted(set(cs)), key=cs.count) if cs else None
    return means, modal


def _comp_feats(c) -> list:
    if c is None:
        return [np.nan] * (2 + len(ANIONS) + 1)
    n_ligs, anion, n_fill = c
    return [float(n_ligs), float(n_fill)] + [float(anion == a) for a in ANIONS] + [float(anion not in ANIONS)]


def row_features(cell, ym, mode: dict, b1_coef: float) -> dict:
    """Feature vector for every unread metal of the cell under the read set implied by ym (NaN = unread).  Geometry
    columns (S06 / S09 / S22): g(m), the complex's composition (ligand count, fill count, inner-sphere anion), the
    anomaly g(m) - I_S[g(S)](m) ONLY when every read metal used by I_S at m has a complex of the same composition
    (otherwise NaN: no cross-composition comparison), a coverage flag and a composition-match flag."""
    x = interp.X_R
    metals = cell["metals"]
    read = sorted((m for m in range(N_LN) if np.isfinite(ym[m])), key=lambda m: x[m])
    xr = np.array([x[m] for m in read])
    yr = np.array([ym[m] for m in read])
    slope_all = float(np.polyfit(xr, yr, 1)[0]) if len(read) >= 2 else 0.0
    geo6 = CO.geometry(ym, metals)
    gd = RF.potentials(ym, metals, (RF.GD,))
    cls = SA.donor_class(str(cell["extractant"]))
    chem = [float(cls == c) for c in SA.CLASSES] + list(rdkit_desc(str(cell["extractant"])))
    use_geo = mode.get("geometry") is not None
    if use_geo:
        G = {m: geometry_of(cell, m, mode) for m in metals}
        wts = RF.interp_weights(ym, metals)
    out = {}
    for b in metals:
        if b in read:
            continue
        j = int(np.searchsorted(xr, x[b]))
        dl = x[b] - xr[j - 1] if j > 0 else np.nan
        dr = xr[j] - x[b] if j < len(read) else np.nan
        side = 0.0 if 0 < j < len(read) else (-1.0 if j == 0 else 1.0)
        seg = (yr[j] - yr[j - 1]) / (xr[j] - xr[j - 1]) if 0 < j < len(read) else np.nan
        frac = dl / (dl + dr) if np.isfinite(dl) and np.isfinite(dr) else np.nan
        v = [x[b], Q[b], dl, dr, frac, side, seg, slope_all, float(gd[b][0]), float(gd[b][0]) * b1_coef] + list(geo6[b]) + chem
        if use_geo:
            gb, cb = G[b]
            match = cb is not None and all(G[j_][1] == cb and np.isfinite(G[j_][0]).all() for j_ in wts[b])
            gi = sum(w * G[j_][0] for j_, w in wts[b].items()) if match else NAN_G
            v += list(gb) + _comp_feats(cb) + list(gb - gi) + [float(np.isfinite(gb).all()), float(match)]
        out[b] = np.array(v, float)
    return out


def _rows(bench, rec, Y, mode, b1, *, cells_subset=None, dopt=None, k=CO.K_TRAIN):
    """Per training situation: unread-metal rows (features, level residual target vs B1, weight) and the pair
    structure needed to score pair loss."""
    from gen13sep.amplitude_bench import cell_weights
    from g21.evaluate import _view
    from gen20k import design as DS
    tr = np.asarray(rec["outer"]["train_index"])
    wmap = dict(zip(tr.tolist(), cell_weights(bench.groups[tr], bench.n_obs[tr]).tolist()))
    sits = []
    for cell in RF.training_cells(bench, rec):
        if cell["k_max"] < k or cell["ci"] not in wmap or (cells_subset is not None and cell["ci"] not in cells_subset):
            continue
        if dopt is None:
            dseq = cell["seqs"]["DOPT_Q"]
        else:
            dz = DS.make_design(_view(cell), "adjacent", "metal")
            dseq = DS.support_sequence(dz, "DOPT_Q", min(dz.k_max, 6), cov=dopt[0], noise_var=dopt[1])
        masks = [("DOPT_Q", frozenset(m for p in dseq[:k] for m in p)),
                 ("UNIFORM", frozenset(m for p in cell["seqs"]["UNIFORM"][:k] for m in p))]
        masks += [("RANDOM", s) for s in CO._random_legal(cell, k, CO.N_RANDOM_MASKS, "r3train")]
        wcell = wmap[cell["ci"]] / len(masks)
        yc = Y[cell["ci"]]
        for kind_m, read in masks:
            ym = np.full(N_LN, np.nan)
            for m in read:
                ym[m] = yc[m]
            base = RF.predict_levels(b1, type("O", (), {"y_masked": ym})(), cell["metals"])
            feats = row_features(cell, ym, mode, float(b1["coef"][0]))
            ms = sorted(feats)
            sits.append({"ci": cell["ci"], "mask": kind_m, "metals": ms, "X": np.vstack([feats[m] for m in ms]),
                         "t": np.array([yc[m] - base[m] for m in ms]), "w": wcell, "query": list(cell["query"]),
                         "base": base, "y": yc})
    return sits


def _pair_loss(sits, corr_of) -> tuple[float, float]:
    """Weighted squared error of hidden strict-adjacent pairs with level = base + correction at unread metals."""
    sse, n = 0.0, 0.0
    for s, c in zip(sits, corr_of):
        lv = s["base"].copy()
        for m, v in zip(s["metals"], c):
            lv[m] = lv[m] + v
        for a, b in s["query"]:
            e = (s["y"][a] - s["y"][b]) - (lv[a] - lv[b])
            sse += s["w"] * e * e
            n += s["w"]
    return sse, n


def _fit_cb(sits, n_trees: int):
    from catboost import CatBoostRegressor
    X = np.vstack([s["X"] for s in sits])
    t = np.concatenate([s["t"] for s in sits])
    w = np.concatenate([np.full(len(s["t"]), s["w"]) for s in sits])
    model = CatBoostRegressor(iterations=n_trees, **CB_PARAMS)
    model.fit(X, t, sample_weight=w)
    return model


def fit_slot(bench, rec, Y, mode: dict, b1: dict) -> dict:
    """Inner choice of the tree count (0 = B1), then a refit on all training situations.  B1 is refitted inside
    every inner block on that block's training rows at B1's selected lambda (as block A)."""
    if mode.get("geometry") == "radius_only":
        mm, mc = metal_means(rec, RF.training_cells(bench, rec))
        mode = dict(mode, metal_means=mm, metal_comp=mc)
    sits = _rows(bench, rec, Y, mode, b1)
    scores = {n: [0.0, 0.0] for n in TREE_GRID}
    for blk in rec["inner"]:
        tr_cells = set(map(int, blk["train_index"]))
        va_cells = set(map(int, blk["val_index"]))
        b1j = _inner_b1(bench, rec, Y, b1, tr_cells, blk)
        mode_j = mode
        if mode.get("geometry") == "radius_only":                    # nested: means of this block's training cells
            mm, mc = metal_means(rec, [c for c in RF.training_cells(bench, rec) if c["ci"] in tr_cells])
            mode_j = dict(mode, metal_means=mm, metal_comp=mc)
        fit_s = _rows(bench, rec, Y, mode_j, b1j, cells_subset=tr_cells, dopt=(blk["cov"], 2 * blk["s2"]))
        val_s = [s for s in _rows(bench, rec, Y, mode_j, b1j, cells_subset=va_cells) if s["mask"] == "DOPT_Q"]
        if not fit_s or not val_s:
            continue
        model = _fit_cb(fit_s, max(TREE_GRID))
        Xv = np.vstack([s["X"] for s in val_s])
        splits = np.cumsum([len(s["t"]) for s in val_s])[:-1]
        for n in TREE_GRID:
            pv = np.zeros(len(Xv)) if n == 0 else model.predict(Xv, ntree_end=n)
            sse, nn = _pair_loss(val_s, np.split(pv, splits))
            scores[n][0] += sse
            scores[n][1] += nn
    sc = {n: v[0] / max(v[1], 1e-12) for n, v in scores.items()}
    best = min(TREE_GRID, key=lambda n: (sc[n], n))
    model = _fit_cb(sits, best) if best > 0 else None
    return {"mode": mode, "n_trees": best, "inner_scores": {str(n): s for n, s in sc.items()}, "model": model, "b1": b1,
            "n_rows": int(sum(len(s["t"]) for s in sits)), "n_situations": len(sits)}


def _inner_b1(bench, rec, Y, b1, tr_cells, blk) -> dict:
    """B1's beta refitted on one inner block's training situations at B1's selected lambda."""
    import math
    sits = RF.training_situations(bench, rec, Y, RF.SPECS["B1"], cells_subset=tr_cells, dopt=(blk["cov"], 2 * blk["s2"]))
    rows = CO._stack(sits)
    coef = np.zeros(1) if (rows is None or math.isinf(b1["lambda"])) else CO.ridge_fit(*rows, b1["lambda"], RF.penalty_vector(RF.SPECS["B1"]))
    return dict(b1, coef=coef)


def predict_levels(fit: dict, obs, cell) -> np.ndarray:
    base = RF.predict_levels(fit["b1"], obs, cell["metals"])
    if fit["model"] is None:
        return base
    feats = row_features(cell, obs.y_masked, fit["mode"], float(fit["b1"]["coef"][0]))
    out = base.copy()
    if feats:
        ms = sorted(feats)
        pv = fit["model"].predict(np.vstack([feats[m] for m in ms]))
        for m, v in zip(ms, pv):
            out[m] = base[m] + float(v)
    return out

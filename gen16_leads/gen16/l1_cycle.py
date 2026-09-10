"""L1 -- the cycle-corrected xTB descriptor.  Shared code for Stage 1 (bookkeeping on the energies
already in hand) and Stage 2 (the same analysis on cycle-corrected dE once the reference species
have been computed on the cluster).

Everything is built on the frozen gen15 ``exp/phys3d`` scaffolding, imported and never modified:

* ``build_block.load_geometries``   -- the 1155 accepted complexes with per-complex element counts,
                                       the standardised Shannon radius ``r`` (``RSTD``) and the
                                       series key ``canonical_smiles || inner_sphere_anion``;
* ``build_block._two_way_residual``  -- the two-way (series x metal) fixed-effect residual, optional
                                       covariates.  ``two_way_fit`` below is a copy that also returns
                                       the coefficients; ``check_two_way_copy`` verifies the copy
                                       reproduces the original residual to 1e-9 before it is used;
* ``build_block._slope_fit``         -- the within-series linear + quadratic fit in ``r``;
* ``descriptor_stats._rho`` / ``_partial_rho`` -- Spearman and partial Spearman given ``n_metals``;
* ``robust_stats``                   -- the chemotype-blocked bootstrap and chemotype-level
                                       permutation constructions (re-implemented here line for line
                                       because that module only exposes ``main``).

Bookkeeping facts established before any model was fitted (``scripts/l1_stage1.py`` re-verifies):

* ``n_fill`` in ``accepted_geometries.csv`` counts fill *donor sites*, not molecules: nitrate is
  bidentate, so ``n_fill = 2*n_NO3 + n_H2O`` and ``coreCN = DENTATE*n_ligs + 2*n_NO3 + n_H2O``
  holds for all 1155 complexes.  An odd nitrate ``n_fill`` means one water is also present.
* the non-ligand atoms of every complex decompose exactly into ``n_NO3`` nitrates and ``n_H2O``
  waters (element counts minus ``n_ligs`` x the RDKit formula of the neutral ligand); all 177
  ligands are neutral as written and are not deprotonated in the complex;
* the xTB total charge is ``3 - n_NO3``: the ``initial_charges`` column sums to exactly that on all
  1145 files that carry it, and the post-relaxation Mulliken ``charge`` column sums to it (tolerance
  0.02 e) on all 1116 files that carry one.  The ten legacy-header files have no energy anyway;
* ``initial_magmoms`` is zero on 1134 of the 1145, and carries the *formal f-electron count* on the
  metal in eleven (Eu 6.0, Yb 1.0).  It is metadata that GFN2 cannot use -- the lanthanides are
  parameterised with f-in-core, so the valence shell is closed -- and it is set inconsistently
  (other Eu and Yb complexes carry 0.0).  ``--uhf 0`` everywhere is therefore correct, and the
  eleven files are reported rather than silently ignored.

Hence the registered ``SPECIES`` model enters each fill species with its *actual molecule count*
(``n_NO3 * gamma_nitrate + n_H2O * gamma_water``); the literal ``n_fill``-column variant is written
out as an exploratory model (``SPECIES_NFILLCOL``).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from gen16 import bootstrap  # noqa: F401  (sys.path + thread cap)

ROOT = bootstrap.ROOT
DATA = ROOT / "dataset with 3D structures"
PHYS3D = ROOT / "gen15_curve" / "exp" / "phys3d"
RESULTS = bootstrap.RESULTS / "L1"
if str(PHYS3D) not in sys.path:
    sys.path.insert(0, str(PHYS3D))
from build_block import (  # noqa: E402
    RSTD, MIN_BLOCK, MIN_SERIES, load_geometries, main_blocks, _two_way_residual, _slope_fit,
)
from descriptor_stats import _rho, _partial_rho  # noqa: E402
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402,F401

MODELS = ("NAIVE", "ELEM", "SPECIES", "SPECIES_CONST")
SETS = ("S8", "S14", "S3")
TARGETS = ("a", "abs_a", "b")
BOOT_REPS = 2000
PERM_REPS = 2000
BOOT_SEED = 20260909          # robust_stats.py's generator seed
MIN_BOOT_ROWS = 20            # robust_stats.py: a resample needs >= 20 rows and >= 5 distinct x
MIN_PERM_UNITS = 10           # robust_stats.py: a chemotype-level rho needs >= 10 units
HARTREE_EV = 27.211386245988
KNOWN_ELEMENTS = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si", "Se", "As"} | set(LANTHANIDES)

# registered decision rule (PRE_REGISTRATION.md section 3, L1)
RULE_POSITIVE_RHO = 0.40
RULE_CLOSED_RHO = 0.25


# --------------------------------------------------------------------------------------
# geometries, fill composition, charge convention
# --------------------------------------------------------------------------------------
def ligand_formula(smiles: str) -> Counter:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return Counter(at.GetSymbol() for at in mol.GetAtoms())


def ligand_formal_charge(smiles: str) -> int:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return int(sum(at.GetFormalCharge() for at in mol.GetAtoms()))


def fill_composition(g: pd.DataFrame, count_cols: list[str]) -> pd.DataFrame:
    """Add ``n_NO3``, ``n_H2O``, ``total_charge`` and ``fill_ok`` from the atom counts.

    Residual element counts = complex counts (metal excluded) - n_ligs x neutral-ligand formula.
    The residual must be exactly ``n_NO3 x {N:1, O:3} + n_H2O x {H:2, O:1}``; anything else sets
    ``fill_ok = False`` and is reported, never silently absorbed.
    """
    g = g.copy()
    formulas = {s: ligand_formula(s) for s in g["canonical_smiles"].unique()}
    # build_block.composition() skips only line 0 of the xyz, so the extxyz *comment* line is
    # counted as one atom of a phantom element ("Properties=species:..." / "energy:").  The real
    # element counts are unaffected; the phantom column is dropped here and the defect is reported
    # rather than fixed in the frozen file (it is constant within a header style, so it enters
    # gen15's construction B as a near-zero-variance covariate).
    els = [c[2:] for c in count_cols if c[2:] in KNOWN_ELEMENTS]
    n_no3 = np.zeros(len(g), dtype=int)
    n_h2o = np.zeros(len(g), dtype=int)
    ok = np.ones(len(g), dtype=bool)
    for i, (smi, nl) in enumerate(zip(g["canonical_smiles"].to_numpy(), g["n_ligs"].to_numpy())):
        f = formulas[smi]
        res = {e: int(g[f"n_{e}"].iat[i]) - int(nl) * f.get(e, 0) for e in els}
        res = {e: v for e, v in res.items() if v != 0}
        nN = res.pop("N", 0)
        nH = res.pop("H", 0)
        nO = res.pop("O", 0)
        if res or nN < 0 or nH < 0 or nH % 2 or nO != 3 * nN + nH // 2:
            ok[i] = False
            continue
        n_no3[i] = nN
        n_h2o[i] = nH // 2
    g["n_NO3"] = n_no3
    g["n_H2O"] = n_h2o
    g["fill_ok"] = ok
    g["total_charge"] = 3 - g["n_NO3"]
    g["lig_n_atoms"] = g["canonical_smiles"].map(lambda s: sum(formulas[s].values()))
    return g


def xyz_charge_audit(rows: pd.DataFrame) -> pd.DataFrame:
    """Per complex: the total charge the xyz declares, the Mulliken charge it carries, and the
    largest |initial_magmom|.  Nothing is inferred; the file is read."""
    import re
    recs = []
    for gk, p, metal in zip(rows["geometry_key"], rows["xyz_path"], rows["metal_symbol"]):
        lines = (DATA / p).read_text().splitlines()
        nat = int(lines[0].split()[0])
        m = re.search(r"Properties=(\S+)", lines[1])
        rec = {"geometry_key": gk, "has_properties_header": bool(m)}
        if m:
            spec = m.group(1).split(":")
            cols, i = [], 0
            while i < len(spec):
                name, _typ, cnt = spec[i], spec[i + 1], int(spec[i + 2])
                i += 3
                cols += [name] if cnt == 1 else [f"{name}{k}" for k in range(cnt)]
            df = pd.DataFrame([ln.split() for ln in lines[2:2 + nat]], columns=cols)
            for c in cols:
                if c != "species":
                    df[c] = df[c].astype(float)
            rec["q_initial_sum"] = float(df["initial_charges"].sum())
            rec["q_metal_initial"] = float(df.loc[df["species"] == metal, "initial_charges"].iloc[0])
            rec["max_abs_magmom"] = float(np.abs(df["initial_magmoms"]).max())
            rec["magmom_on_metal"] = float(df.loc[df["species"] == metal, "initial_magmoms"].iloc[0])
            rec["q_mulliken_sum"] = float(df["charge"].sum()) if "charge" in df.columns else np.nan
        recs.append(rec)
    return pd.DataFrame(recs)


def load_energy_rows() -> tuple[pd.DataFrame, list[str]]:
    """The 1155 accepted complexes with composition, fill counts and ``complex_total_energy_eV``."""
    g, count_cols = load_geometries()
    g = main_blocks(g)
    g = fill_composition(g, count_cols)
    sc = pd.read_parquet(DATA / "features/complex_physical_scalars.parquet")
    g = g.merge(sc[["geometry_key", "complex_total_energy_eV"]], on="geometry_key", how="left")
    if not g["fill_ok"].all():
        bad = g.loc[~g["fill_ok"], "geometry_key"].tolist()
        raise RuntimeError(f"{len(bad)} complexes do not decompose into ligand + NO3 + H2O: {bad[:5]}")
    if int((g["n_fill"] != 2 * g["n_NO3"] + g["n_H2O"]).sum()):
        raise RuntimeError("n_fill != 2*n_NO3 + n_H2O for some complex; the fill bookkeeping changed")
    return g, count_cols


# --------------------------------------------------------------------------------------
# the two-way fit (copy of build_block._two_way_residual that also returns coefficients)
# --------------------------------------------------------------------------------------
def two_way_fit(y: np.ndarray, grp: np.ndarray, met: np.ndarray, cov: np.ndarray | None = None,
                cov_names: list[str] | None = None):
    """Residual, coefficients and design of ``y ~ group FE + metal FE (+ covariates)``.

    Identical construction to ``build_block._two_way_residual`` (full group dummies, metal dummies
    with the first level dropped, ``y`` demeaned, ``np.linalg.lstsq``); returns the fitted
    coefficients and column names as well, and standard errors from the residual variance.
    """
    ok = np.isfinite(y)
    if cov is not None:
        ok &= np.isfinite(cov).all(axis=1)
    gd = pd.get_dummies(pd.Series(grp[ok]), drop_first=False)
    md = pd.get_dummies(pd.Series(met[ok]), drop_first=True)
    names = [f"series::{c}" for c in gd.columns] + [f"metal::{c}" for c in md.columns]
    parts = [gd.to_numpy(dtype=float), md.to_numpy(dtype=float)]
    if cov is not None:
        parts.append(cov[ok])
        names += list(cov_names or [f"cov{i}" for i in range(cov.shape[1])])
    X = np.hstack(parts)
    yy = y[ok] - np.nanmean(y[ok])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    r = np.full(len(y), np.nan)
    r[ok] = yy - X @ beta
    dof = max(1, int(ok.sum()) - np.linalg.matrix_rank(X))
    s2 = float((r[ok] ** 2).sum() / dof)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(XtX_inv) * s2, 0, None))
    coef = pd.DataFrame({"term": names, "coef": beta, "se": se})
    return r, coef, X, ok


def check_two_way_copy(rows: pd.DataFrame, count_cols: list[str]) -> float:
    """Max |difference| between ``two_way_fit`` and the frozen ``_two_way_residual`` on the real data."""
    y = rows["complex_total_energy_eV"].to_numpy(dtype=float)
    grp = rows["series"].to_numpy()
    met = rows["metal_symbol"].to_numpy()
    C = centred_counts(rows, count_cols)
    d = 0.0
    for cov in (None, C):
        r0 = _two_way_residual(y, grp, met, cov=cov)
        r1, *_ = two_way_fit(y, grp, met, cov=cov)
        d = max(d, float(np.nanmax(np.abs(r0 - r1))))
    return d


def centred_counts(rows: pd.DataFrame, count_cols: list[str]) -> np.ndarray:
    """Element counts centred within series, columns with zero variance dropped (construction B)."""
    C = rows[count_cols].to_numpy(dtype=float)
    C = C - pd.DataFrame(C).groupby(rows["series"].to_numpy()).transform("mean").to_numpy()
    keep = C.std(axis=0) > 1e-9
    return C[:, keep]


# --------------------------------------------------------------------------------------
# the four registered models (+ one exploratory variant)
# --------------------------------------------------------------------------------------
def varying_series(rows: pd.DataFrame, col: str = "n_ligs") -> list[str]:
    v = rows.groupby("series")[col].nunique()
    return sorted(v.index[v > 1].tolist())


def species_design(rows: pd.DataFrame, *, fill_cols=("n_NO3", "n_H2O"), ligand_terms: bool = True):
    """Covariate block of the SPECIES model: fill-species counts (global) and, for every series in
    which ``n_ligs`` varies, ``(n_ligs - series mean) x 1[series]`` (one delta per such series)."""
    cols, names = [], []
    for c in fill_cols:
        cols.append(rows[c].to_numpy(dtype=float))
        names.append(f"gamma::{c}")
    if ligand_terms:
        ser = rows["series"].to_numpy()
        nl = rows["n_ligs"].to_numpy(dtype=float)
        nl_c = nl - pd.Series(nl).groupby(ser).transform("mean").to_numpy()
        for s in varying_series(rows):
            cols.append(np.where(ser == s, nl_c, 0.0))
            names.append(f"delta::{s}")
    return np.column_stack(cols), names


def fit_models(rows: pd.DataFrame, count_cols: list[str], energy: str = "complex_total_energy_eV",
               models: tuple[str, ...] = MODELS + ("SPECIES_NFILLCOL",)) -> tuple[pd.DataFrame, dict]:
    """Fit every model on the rows with a finite energy; return rows with ``resid_<MODEL>`` columns
    and a dict of coefficient tables.  NAIVE / ELEM / SPECIES use identical rows; SPECIES_CONST is
    the SPECIES fit restricted to series with constant ``n_ligs`` (registered as such)."""
    rows = rows[np.isfinite(rows[energy].to_numpy(dtype=float))].reset_index(drop=True).copy()
    y = rows[energy].to_numpy(dtype=float)
    grp = rows["series"].to_numpy()
    met = rows["metal_symbol"].to_numpy()
    coefs = {}
    for m in models:
        if m == "NAIVE":
            r, coef, *_ = two_way_fit(y, grp, met)
        elif m == "ELEM":
            C = centred_counts(rows, count_cols)
            r, coef, *_ = two_way_fit(y, grp, met, cov=C, cov_names=[f"elem{i}" for i in range(C.shape[1])])
        elif m == "SPECIES":
            C, names = species_design(rows)
            r, coef, *_ = two_way_fit(y, grp, met, cov=C, cov_names=names)
        elif m == "SPECIES_NFILLCOL":
            # exploratory: the literal n_fill column times a per-fill-species coefficient
            nf = rows["n_fill"].to_numpy(dtype=float)
            is_no3 = (rows["fill_ligand"].to_numpy() == "nitrate").astype(float)
            rows["_nfill_nitrate"] = nf * is_no3
            rows["_nfill_water"] = nf * (1 - is_no3)
            C, names = species_design(rows, fill_cols=("_nfill_nitrate", "_nfill_water"))
            r, coef, *_ = two_way_fit(y, grp, met, cov=C, cov_names=names)
            rows = rows.drop(columns=["_nfill_nitrate", "_nfill_water"])
        elif m == "SPECIES_CONST":
            const = ~rows["series"].isin(varying_series(rows)).to_numpy()
            sub = rows[const].reset_index(drop=True)
            C, names = species_design(sub, ligand_terms=False)
            rs, coef, *_ = two_way_fit(sub[energy].to_numpy(dtype=float), sub["series"].to_numpy(),
                                       sub["metal_symbol"].to_numpy(), cov=C, cov_names=names)
            r = np.full(len(rows), np.nan)
            r[const] = rs
        else:
            raise ValueError(m)
        rows[f"resid_{m}"] = r
        coefs[m] = coef
    return rows, coefs


# --------------------------------------------------------------------------------------
# identifiability of delta_ligand against the within-series radius trend
# --------------------------------------------------------------------------------------
def identifiability(rows: pd.DataFrame) -> pd.DataFrame:
    """For every series in which ``n_ligs`` varies (on the finite-energy rows): the within-series
    correlation of ``n_ligs`` with ``r``, the R^2 of ``n_ligs`` on ``[r, r^2]`` (VIF = 1/(1-R^2)),
    the condition number of the standardised within-series design ``[r, r^2, n_ligs]``, and the
    attenuation factor ``1 - rho^2`` by which a linear trend in ``r`` is shrunk once the step is
    absorbed by a free delta (the residual slope estimates ``beta (1 - rho^2)``, not ``beta``).

    Also the VIF of each ``delta`` column in the SPECIES design *augmented* with that series' own
    ``r`` and ``r^2`` terms -- the direct question "can the step be told apart from the trend?"."""
    out = []
    ser = rows["series"].to_numpy()
    C, names = species_design(rows)
    y = rows["complex_total_energy_eV"].to_numpy(dtype=float)
    _, _, X, ok = two_way_fit(y, ser, rows["metal_symbol"].to_numpy(), cov=C, cov_names=names)
    r_all = rows["r"].to_numpy(dtype=float)
    for s in varying_series(rows):
        m = ser == s
        r = r_all[m]
        nl = rows.loc[m, "n_ligs"].to_numpy(dtype=float)
        rc, nc = r - r.mean(), nl - nl.mean()
        rho = float(np.corrcoef(rc, nc)[0, 1]) if nc.std() > 0 and rc.std() > 0 else np.nan
        A = np.c_[np.ones(m.sum()), rc, rc ** 2]
        b, *_ = np.linalg.lstsq(A, nc, rcond=None)
        r2q = 1 - ((nc - A @ b) ** 2).sum() / (nc ** 2).sum() if (nc ** 2).sum() > 0 else np.nan
        Z = np.c_[rc, rc ** 2 - (rc ** 2).mean(), nc]
        Z = Z / np.where(Z.std(axis=0) > 0, Z.std(axis=0), 1.0)
        cond = float(np.linalg.cond(Z)) if m.sum() >= 4 else np.nan
        # VIF of delta_s inside the augmented SPECIES design (series x r, series x r^2 added)
        j = names.index(f"delta::{s}")
        jcol = X.shape[1] - C.shape[1] + j
        # the two extra columns (this series' own r and r^2 trend) on the ok-rows
        extra_r = np.zeros(ok.sum())
        extra_r2 = np.zeros(ok.sum())
        mm = m[ok]
        rr = r_all[ok][mm]
        extra_r[mm] = rr - rr.mean()
        extra_r2[mm] = (rr - rr.mean()) ** 2 - ((rr - rr.mean()) ** 2).mean()
        aug = np.c_[X, extra_r, extra_r2]
        target = aug[:, jcol]
        others = np.delete(aug, jcol, axis=1)
        bb, *_ = np.linalg.lstsq(others, target, rcond=None)
        r2 = 1 - ((target - others @ bb) ** 2).sum() / ((target - target.mean()) ** 2).sum()
        out.append({"series": s, "n_rows": int(m.sum()), "n_ligs_values": "/".join(map(str, sorted(set(nl.astype(int))))),
                    "corr_nligs_r": rho, "attenuation_1_minus_rho2": 1 - rho ** 2 if np.isfinite(rho) else np.nan,
                    "vif_linear": 1 / (1 - rho ** 2) if np.isfinite(rho) and abs(rho) < 1 else np.inf,
                    "r2_nligs_on_r_r2": r2q, "vif_quadratic": 1 / (1 - r2q) if np.isfinite(r2q) and r2q < 1 else np.inf,
                    "cond_within_series": cond,
                    "vif_delta_in_augmented_design": 1 / (1 - r2) if r2 < 1 else np.inf})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# per-extractant slopes and the registered sets
# --------------------------------------------------------------------------------------
def choose_series(g: pd.DataFrame) -> pd.Series:
    """gen15's rule: per canonical SMILES the series (anion) with most metals, tie -> nitrate."""
    cand = g.groupby(["canonical_smiles", "series"]).metal_symbol.nunique().rename("n").reset_index()
    cand["nitrate"] = cand["series"].str.endswith("nitrate")
    cand = cand.sort_values(["canonical_smiles", "n", "nitrate"], ascending=[True, False, False])
    best = cand.groupby("canonical_smiles").head(1)
    return best.set_index("canonical_smiles")["series"]


def slope_table(rows: pd.DataFrame, models: tuple[str, ...], g_all: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per extractant x model: within-series slope / quadratic / rms of the model residual in
    the standardised Shannon radius, with the number of computed (finite-residual) metals."""
    series_of = choose_series(g_all if g_all is not None else rows)
    out = []
    for smi, ser in series_of.items():
        s = rows[rows["series"] == ser].sort_values("r")
        rvec = s["r"].to_numpy(dtype=float)
        base = {"extractant": smi, "series": ser, "anion": ser.split("||")[1],
                "n_geometry_metals": int((g_all if g_all is not None else rows).loc[lambda d: d["series"] == ser, "metal_symbol"].nunique()),
                "n_ligs_varies": int(s["n_ligs"].nunique() > 1) if len(s) else 0,
                "n_NO3_varies": int(s["n_NO3"].nunique() > 1) if len(s) else 0,
                "n_H2O_varies": int(s["n_H2O"].nunique() > 1) if len(s) else 0,
                "composition_varies": int(s["composition"].nunique() > 1) if len(s) else 0}
        for m in models:
            col = f"resid_{m}"
            yv = s[col].to_numpy(dtype=float) if col in s.columns else np.full(len(s), np.nan)
            sl, cu, rms, n = _slope_fit(rvec, yv, quad=True)
            rec = dict(base)
            rec.update({"model": m, "slope": sl, "quad": cu, "rms": rms, "n_computed_metals": int(n)})
            out.append(rec)
    return pd.DataFrame(out)


def targets() -> pd.DataFrame:
    """The 82 well-determined cohort extractants (>= 5 measured metals) with a, b, |a|, n_metals and
    chemotype -- gen15's frozen ``extractant_targets.parquet``."""
    t = pd.read_parquet(PHYS3D / "extractant_targets.parquet")
    return t[["extractant", "a", "b", "abs_a", "n_metals", "n_cells", "chemotype"]].copy()


def attach_sets(sl: pd.DataFrame) -> pd.DataFrame:
    t = targets()
    df = sl.merge(t, on="extractant", how="inner")
    n = df["n_computed_metals"].to_numpy()
    df["S8"] = n >= 8
    df["S14"] = n == 14
    df["S3"] = n >= 3
    return df


# --------------------------------------------------------------------------------------
# statistics: Spearman, LOCO, partial, blocked bootstrap, permutation null
# --------------------------------------------------------------------------------------
def wide(df: pd.DataFrame, value: str = "slope") -> pd.DataFrame:
    """82-row frame: one column ``<value>__<model>__<set>`` per (model, set), NaN outside the set."""
    t = targets().set_index("extractant")
    cols = {}
    for m in df["model"].unique():
        d = df[df["model"] == m].set_index("extractant")
        for s in SETS:
            v = d[value].where(d[s])
            cols[f"{value}__{m}__{s}"] = v.reindex(t.index)
    return pd.concat([t, pd.DataFrame(cols, index=t.index)], axis=1).reset_index()


def basic_stats(x: np.ndarray, y: np.ndarray, chem: np.ndarray, nm: np.ndarray) -> dict:
    rho, p, n = _rho(x, y)
    loco = []
    for ch in np.unique(chem):
        m = chem != ch
        rr, _, _ = _rho(x[m], y[m])
        if np.isfinite(rr):
            loco.append((ch, rr))
    lv = np.array([v for _, v in loco]) if loco else np.array([np.nan])
    stable = bool(np.isfinite(rho) and len(loco) and np.all(np.sign(lv) == np.sign(rho)))
    return {"rho": rho, "p": p, "n": n, "loco_min": float(np.nanmin(lv)), "loco_max": float(np.nanmax(lv)),
            "loco_sign_stable": stable, "n_loco": len(loco), "partial_rho_n_metals": _partial_rho(x, y, nm),
            "_loco": loco}


def blocked_bootstrap(W: pd.DataFrame, cols: list[str], reps: int = BOOT_REPS, seed: int = BOOT_SEED):
    """robust_stats.py construction: resample the chemotypes with replacement (every member
    extractant travels with its chemotype), one shared set of draws for every column and target."""
    chem = W["chemotype"].to_numpy()
    uch = np.unique(chem)
    idx_of = {c: np.flatnonzero(chem == c) for c in uch}
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([idx_of[c] for c in rng.choice(uch, size=len(uch), replace=True)])
             for _ in range(reps)]
    rows = []
    for tgt in TARGETS:
        y = W[tgt].to_numpy(dtype=float)
        for c in cols:
            x = W[c].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < MIN_BOOT_ROWS:
                rows.append({"column": c, "target": tgt, "ci_lo": np.nan, "ci_hi": np.nan, "n_boot": 0})
                continue
            vals = []
            for b in boots:
                bb = b[np.isfinite(x[b]) & np.isfinite(y[b])]
                if len(bb) < MIN_BOOT_ROWS or len(np.unique(x[bb])) < 5:
                    continue
                r = stats.spearmanr(x[bb], y[bb]).statistic
                if np.isfinite(r):
                    vals.append(r)
            lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) > 100 else (np.nan, np.nan)
            rows.append({"column": c, "target": tgt, "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": len(vals)})
    return pd.DataFrame(rows), rng


def permutation_null(W: pd.DataFrame, cols: list[str], rng: np.random.Generator, reps: int = PERM_REPS):
    """robust_stats.py construction: chemotype means are the units; the target is permuted across
    chemotypes; the statistic is max |rho| over the family ``cols`` (here 4 models x 3 sets)."""
    cm = W.groupby("chemotype")[cols + list(TARGETS)].mean()
    M = cm[cols].to_numpy(dtype=float)
    out_rows, obs_rows = [], []
    for tgt in TARGETS:
        y = cm[tgt].to_numpy(dtype=float)
        obs = []
        for j in range(M.shape[1]):
            ok = np.isfinite(M[:, j]) & np.isfinite(y)
            obs.append(stats.spearmanr(M[ok, j], y[ok]).statistic if ok.sum() >= MIN_PERM_UNITS else np.nan)
        obs = np.array(obs, dtype=float)
        null = np.empty(reps)
        for k in range(reps):
            yp = rng.permutation(y)
            best = 0.0
            for j in range(M.shape[1]):
                ok = np.isfinite(M[:, j]) & np.isfinite(yp)
                if ok.sum() < MIN_PERM_UNITS:
                    continue
                r = abs(stats.spearmanr(M[ok, j], yp[ok]).statistic)
                if np.isfinite(r) and r > best:
                    best = r
            null[k] = best
        thr = float(np.percentile(null, 95))
        for j, c in enumerate(cols):
            obs_rows.append({"column": c, "target": tgt, "rho_chemotype_means": obs[j],
                             "n_chemotypes": int((np.isfinite(M[:, j]) & np.isfinite(y)).sum()),
                             "clears_familywise_bar": bool(np.nan_to_num(abs(obs[j])) >= thr)})
        out_rows.append({"target": tgt, "family_size": len(cols), "n_chemotype_units": len(cm),
                         "perm_reps": reps, "null_p95_max_abs_rho": thr,
                         "null_p50_max_abs_rho": float(np.percentile(null, 50)),
                         "observed_max_abs_rho": float(np.nanmax(np.abs(obs))),
                         "observed_argmax": cols[int(np.nanargmax(np.abs(obs)))],
                         "familywise_p": float((null >= np.nanmax(np.abs(obs))).mean())})
    return pd.DataFrame(out_rows), pd.DataFrame(obs_rows)


def full_stats(df: pd.DataFrame, models: tuple[str, ...] = MODELS, value: str = "slope",
               family_models: tuple[str, ...] = MODELS):
    """Registered statistics per (model, set, target) plus the LOCO sweep, bootstrap and null.

    ``models`` are the rows written out (registered four plus any exploratory variant);
    ``family_models`` are the columns entering the family-wise permutation null, which the
    pre-registration fixes at the four registered models x three sets."""
    W = wide(df, value)
    cols = [f"{value}__{m}__{s}" for m in models for s in SETS]
    fam_cols = [f"{value}__{m}__{s}" for m in family_models for s in SETS]
    chem = W["chemotype"].to_numpy()
    nm = W["n_metals"].to_numpy(dtype=float)
    boot, rng = blocked_bootstrap(W, cols)
    null, obs = permutation_null(W, fam_cols, rng)
    obs = obs.set_index(["column", "target"])
    rows, loco_rows = [], []
    for m in models:
        for s in SETS:
            c = f"{value}__{m}__{s}"
            x = W[c].to_numpy(dtype=float)
            for tgt in TARGETS:
                y = W[tgt].to_numpy(dtype=float)
                st = basic_stats(x, y, chem, nm)
                for ch, rr in st.pop("_loco"):
                    loco_rows.append({"model": m, "set": s, "target": tgt, "held_out_chemotype": ch, "rho": rr})
                b = boot[(boot["column"] == c) & (boot["target"] == tgt)].iloc[0]
                if (c, tgt) in obs.index:
                    o = obs.loc[(c, tgt)]
                    o_rho, o_nch, o_clear = (o["rho_chemotype_means"], int(o["n_chemotypes"]),
                                             bool(o["clears_familywise_bar"]))
                else:                        # exploratory column: outside the registered null family
                    cm = W.groupby("chemotype")[[c, tgt]].mean()
                    okc = np.isfinite(cm[c].to_numpy(float)) & np.isfinite(cm[tgt].to_numpy(float))
                    o_rho = (stats.spearmanr(cm[c].to_numpy(float)[okc], cm[tgt].to_numpy(float)[okc]).statistic
                             if okc.sum() >= MIN_PERM_UNITS else np.nan)
                    o_nch, o_clear = int(okc.sum()), False
                rec = {"model": m, "set": s, "target": tgt, "value": value, **st,
                       "ci95_low": b["ci_lo"], "ci95_high": b["ci_hi"], "n_boot": int(b["n_boot"]),
                       "ci_excludes_zero": bool(np.isfinite(b["ci_lo"]) and b["ci_lo"] * b["ci_hi"] > 0),
                       "rho_chemotype_means": o_rho, "n_chemotypes": o_nch,
                       "clears_familywise_bar": o_clear}
                rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(loco_rows), null, W


# --------------------------------------------------------------------------------------
# the registered decision rule, and the contrasts file
# --------------------------------------------------------------------------------------
RULE_TEXT = ("L1 is positive if, on S8, SPECIES gives |rho(slope, a)| >= 0.40 with a LOCO-stable sign, "
             "|partial rho | n_metals| >= 0.40, and the chemotype-blocked CI excludes zero.  It is closed "
             "if |rho| < 0.25 on S8 with a CI containing zero.  Between 0.25 and 0.40 it is 'not closed, "
             "not positive' and is said so.")


def decision(st: pd.DataFrame, model: str = "SPECIES", set_: str = "S8") -> dict:
    r = st[(st["model"] == model) & (st["set"] == set_) & (st["target"] == "a")].iloc[0]
    rho, part = float(r["rho"]), float(r["partial_rho_n_metals"])
    ci_ex = bool(r["ci_excludes_zero"])
    stable = bool(r["loco_sign_stable"])
    positive = abs(rho) >= RULE_POSITIVE_RHO and stable and abs(part) >= RULE_POSITIVE_RHO and ci_ex
    closed = abs(rho) < RULE_CLOSED_RHO and not ci_ex
    verdict = "positive" if positive else ("closed" if closed else "between")
    return {"model": model, "set": set_, "n": int(r["n"]), "rho_a": rho, "p_a": float(r["p"]),
            "partial_rho_a_given_n_metals": part, "loco_min": float(r["loco_min"]), "loco_max": float(r["loco_max"]),
            "loco_sign_stable": stable, "ci95_low": float(r["ci95_low"]), "ci95_high": float(r["ci95_high"]),
            "ci_excludes_zero": ci_ex, "abs_rho_ge_0.40": abs(rho) >= RULE_POSITIVE_RHO,
            "abs_partial_ge_0.40": abs(part) >= RULE_POSITIVE_RHO, "abs_rho_lt_0.25": abs(rho) < RULE_CLOSED_RHO,
            "verdict": verdict, "rule": RULE_TEXT}


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def contrasts_frame(st: pd.DataFrame, *, lead: str = "L1", stage: str = "stage1",
                    registered: tuple[tuple[str, str, str], ...] = (("SPECIES", "S8", "a"), ("SPECIES_CONST", "S8", "a")),
                    extra_exploratory: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (model, set, target) correlation in the paired_contrasts column layout.

    ``point`` is the Spearman rho, the interval is the chemotype-blocked bootstrap, ``p_two_sided``
    the Spearman p.  ``seeds_positive`` is NaN (a corpus correlation has no split seeds) and
    ``passes_P1`` is False (P1 is the MAE rule; the L1 correlation rule is ``passes_L1_rule``)."""
    frames = [st.assign(stage=stage)]
    if extra_exploratory is not None and len(extra_exploratory):
        frames.append(extra_exploratory.assign(stage=stage))
    s = pd.concat(frames, ignore_index=True)
    reg = set(registered)
    # the registered correlation is the *slope*'s; the quadratic term is exploratory throughout
    s["family"] = [("registered" if (v == "slope" and (m, se, t) in reg) else "exploratory")
                   for m, se, t, v in zip(s["model"], s["set"], s["target"], s["value"])]
    rows = []
    for _, r in s.iterrows():
        rows.append({
            "design": "corpus", "comparison": f"{r['stage']}:{r['model']}_{r['value']}_vs_{r['target']}@{r['set']}",
            "point": r["rho"], "ci95_low": r["ci95_low"], "ci95_high": r["ci95_high"],
            "p_two_sided": r["p"], "seeds_positive": np.nan, "n_seeds": 0,
            "loco_stable": bool(r["loco_sign_stable"]), "loco_min": r["loco_min"], "loco_max": r["loco_max"],
            "passes_P1": False, "family": r["family"], "lead": lead,
            "model": r["model"], "set": r["set"], "target": r["target"], "value": r["value"], "stage": r["stage"],
            "n_units": int(r["n"]), "partial_rho_n_metals": r["partial_rho_n_metals"],
            "ci_excludes_zero": bool(r["ci_excludes_zero"]),
            "rho_chemotype_means": r["rho_chemotype_means"], "clears_familywise_bar": bool(r["clears_familywise_bar"]),
            "passes_L1_rule": bool(r["target"] == "a" and abs(r["rho"]) >= RULE_POSITIVE_RHO and r["loco_sign_stable"]
                                   and abs(r["partial_rho_n_metals"]) >= RULE_POSITIVE_RHO and r["ci_excludes_zero"]),
        })
    out = pd.DataFrame(rows)
    out["p_bh_within_family"] = np.nan
    for fam in ("registered", "exploratory"):
        m = out["family"] == fam
        out.loc[m, "p_bh_within_family"] = bh(out.loc[m, "p_two_sided"].to_numpy())
    return out

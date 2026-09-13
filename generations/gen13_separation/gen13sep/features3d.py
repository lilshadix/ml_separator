"""RESP3D — the coordination sphere's *response* to swapping the lanthanide.

Gen2–gen4 fed absolute 3D descriptors of one complex and found nothing that beat a
width-matched permutation null.  Gen13 uses 3D differently, as a helper for the small
differences between lanthanides: for one ligand (and one inner-sphere anion recipe) the
bundle holds xTB-optimised complexes for several metals.  Each invariant descriptor of the
first coordination sphere is regressed on the standardised Shannon radius across that
ligand's metals, and the *slope* (how much the sphere follows the cation), the intercept
at the mean radius and the residual spread become ligand-level features.  A ligand whose
Ln–donor distances follow the ionic radius with slope 1 is "compliant"; one that cannot
contract (rigid, preorganised) has a smaller slope and, by the preorganisation argument,
should be light-selective.

Every quantity is computed from geometries only; no target is read.  Cells whose ligand
has fewer than ``MIN_METALS`` complexes get NaN (trees impute), and ``resp3d__n_metals``
records the support so a model can discount thin series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths
from .cohort import Gen13Cohort
from .metals import SHANNON_RADIUS_CN8

MIN_METALS = 3
DESCRIPTORS: tuple[str, ...] = (
    "feat3d__complex_physical__ln_donor_distance_mean",
    "feat3d__complex_physical__ln_donor_distance_min",
    "feat3d__complex_physical__ln_donor_distance_max",
    "feat3d__complex_physical__ln_donor_distance_std",
    "feat3d__complex_physical__coordination_number",
    "feat3d__complex_physical__observed_donors_within_3p10A",
    "feat3d__complex_physical__metal_partial_charge",
    "feat3d__complex_physical__donor_partial_charge_mean",
    "feat3d__complex_physical__donor_partial_charge_std",
    "feat3d__complex_physical__donor_partial_charge_min",
    "feat3d__complex_physical__donor_partial_charge_max",
    "feat3d__complex_physical__dipole_magnitude",
    "feat3d__polyhedron_scalars__coreCN_donor_gap",
    "feat3d__polyhedron_scalars__coreCN_max_donor_dist",
    "feat3d__polyhedron_scalars__next_donor_dist",
)


def _short(col: str) -> str:
    return col.split("__", 2)[2]


def build_series_table() -> pd.DataFrame:
    """One row per (canonical_smiles, anion): slope / intercept / residual sd per descriptor."""
    bundle = pd.read_parquet(paths.BUNDLE_PARQUET)
    g = bundle[bundle["geometry_ok"]].drop_duplicates("geometry_key").copy()
    g["anion"] = g["geometry_key"].str.split("|").str[-1]
    r = g["metal"].map(SHANNON_RADIUS_CN8).astype(float)
    rz = (r - np.mean(list(SHANNON_RADIUS_CN8.values()))) / np.std(list(SHANNON_RADIUS_CN8.values()))
    g["rz"] = rz
    rows = []
    for (smi, anion), sub in g.groupby(["canonical_smiles", "anion"]):
        rec = {"canonical_smiles": smi, "anion": anion, "resp3d__n_metals": float(sub["metal"].nunique())}
        for col in DESCRIPTORS:
            y = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
            x = sub["rz"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            name = _short(col)
            if ok.sum() >= MIN_METALS and np.ptp(x[ok]) > 0:
                b, a = np.polyfit(x[ok], y[ok], 1)
                resid = y[ok] - (a + b * x[ok])
                rec[f"resp3d__{name}__slope"] = float(b)
                rec[f"resp3d__{name}__intercept"] = float(a)
                rec[f"resp3d__{name}__resid_sd"] = float(resid.std(ddof=1)) if ok.sum() > 2 else np.nan
            else:
                rec[f"resp3d__{name}__slope"] = np.nan
                rec[f"resp3d__{name}__intercept"] = np.nan
                rec[f"resp3d__{name}__resid_sd"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def build_resp3d(cohort: Gen13Cohort, series: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-cell RESP3D block: the ligand's series for the cell's dominant anion recipe, else the
    ligand's other series, else NaN."""
    series = build_series_table() if series is None else series
    bundle = pd.read_parquet(paths.BUNDLE_PARQUET, columns=["safe_exp_id", "inner_sphere_anion"])
    anion_of = bundle.set_index("safe_exp_id")["inner_sphere_anion"].astype(str)
    cols = [c for c in series.columns if c.startswith("resp3d__")]
    by_key = {(r.canonical_smiles, r.anion): r for r in series.itertuples(index=False)}
    by_smi: dict[str, list] = {}
    for r in series.itertuples(index=False):
        by_smi.setdefault(r.canonical_smiles, []).append(r)
    out = []
    for _, cell in cohort.frame.iterrows():
        ids = str(cell["safe_exp_ids"]).split(";")
        anions = anion_of.reindex(ids).dropna()
        anion = anions.mode().iat[0] if len(anions) else None
        rec = by_key.get((cell["extractant"], anion))
        if rec is None:
            cands = by_smi.get(cell["extractant"], [])
            rec = max(cands, key=lambda r: r.resp3d__n_metals) if cands else None
        if rec is None:
            out.append({c: np.nan for c in cols})
        else:
            out.append({c: getattr(rec, c) for c in cols})
    block = pd.DataFrame(out, index=cohort.frame.index)
    block["resp3d__n_metals"] = block["resp3d__n_metals"].fillna(0.0)
    return block

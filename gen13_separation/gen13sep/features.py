"""Feature blocks for a cell.  Every block is target-free and frozen before any fit.

Blocks (all per cell; ligand blocks are constant within an extractant):

* ``COND``      — the 64 bundle condition columns as numerics (one-hots + 5 continuous).
* ``MASSACT``   — log10 of the positive continuous conditions and the ligand-stoichiometry
                  products ``log[L] x DENTATE``, ``log[L] x coreCN``, ``log[L] x log[H+]``
                  (gen5's mass-action block, gen12's Eu form; recipe columns are the per-cell mode).
* ``ECFP``      — 2,048-bit Morgan fingerprint from the bundle.
* ``PHYSCHEM``  — the 10 RDKit scalars in the bundle.
* ``DONORS``    — the frozen gen6 donor census (``chem__donor__*``, dentate, core CN, n_ligands).
* ``LIG2D``     — the bundle's 206 extended RDKit descriptors (``lig2d__*``); ``Ipc`` is
                  log-transformed because it spans 1e8-1e29 and poisons any scaler.
* ``COORD``     — Gen12.2's frozen 114-column coordination-topology block (v1.1.0), NaN for
                  the two SF extractants it does not cover.  The post-hoc corrected block is
                  available as ``COORD_POSTHOC`` for sensitivity only.

Forbidden tokens are asserted on every matrix so that no identity, provenance or
target column can enter a model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import paths
from .cohort import FORBIDDEN_FEATURE_EXACT, FORBIDDEN_FEATURE_TOKENS, Gen13Cohort

CONTINUOUS_CONDITIONS: tuple[str, ...] = (
    "cond__acid_concentration_M", "cond__extractant_concentration_M", "cond__temperature_C",
    "cond__metal_concentration_mM", "cond__contact_time_min",
)
PHYSCHEM_COLUMNS: tuple[str, ...] = (
    "MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings",
    "NumAliphaticRings", "RingCount", "FractionCSP3", "MolLogP",
)
DONOR_COLUMNS: tuple[str, ...] = (
    "chem__donor__O(amide_carbonyl)", "chem__donor__O(ether)", "chem__donor__N(aromatic)",
    "chem__donor__N(amine)", "chem__donor__S(donor)", "chem__donor__O(hydroxyl)",
    "chem__donor__O(ester_carbonyl)", "chem__donor__O(carbonyl)", "chem__donor__n_total",
    "chem__dentate", "chem__core_cn", "chem__n_ligands", "chem__n_fill",
)
BLOCK_ORDER: tuple[str, ...] = ("COND", "MASSACT", "PHYSCHEM", "DONORS", "ECFP", "LIG2D", "COORD")
#: 3D response block (gen13sep.features3d); opt-in, never in the default ladder's blocks
RESP3D_BLOCK = "RESP3D"
#: external aqueous-logK series prior (scripts/g13_build_logk_prior.py); opt-in helper block
LOGK_BLOCK = "LOGK"
LOGK_PRIOR_PARQUET = paths.GEN13_ROOT / "features" / "logk_prior.parquet"


@dataclass(frozen=True)
class FeatureTable:
    frame: pd.DataFrame                    # cells x all feature columns (float)
    blocks: dict[str, tuple[str, ...]]

    def matrix(self, block_names: tuple[str, ...] | list[str]) -> pd.DataFrame:
        cols: list[str] = []
        for b in block_names:
            if b not in self.blocks:
                raise KeyError(f"unknown block {b!r}; have {sorted(self.blocks)}")
            cols.extend(self.blocks[b])
        _assert_allowed(cols)
        return self.frame.loc[:, cols]


def _assert_allowed(columns) -> None:
    bad = [c for c in columns if c in FORBIDDEN_FEATURE_EXACT or any(tok in c for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise RuntimeError(f"forbidden feature columns: {bad[:5]}")


def _mass_action(cells: pd.DataFrame) -> pd.DataFrame:
    new: dict[str, np.ndarray] = {}
    logs: dict[str, pd.Series] = {}
    for col in CONTINUOUS_CONDITIONS:
        values = pd.to_numeric(cells[col], errors="coerce")
        positive = values.where(values > 0)
        if positive.notna().sum() == 0:
            continue
        logs[col] = np.log10(positive)
        new[f"massact__log10_{col}"] = logs[col].to_numpy(dtype=float)
    log_l = logs.get("cond__extractant_concentration_M")
    log_h = logs.get("cond__acid_concentration_M")
    if log_l is not None:
        for col in ("recipe__DENTATE", "recipe__coreCN"):
            new[f"massact__logL_x_{col.split('__')[1]}"] = (
                log_l * pd.to_numeric(cells[col], errors="coerce")).to_numpy(dtype=float)
        if log_h is not None:
            new["massact__logL_x_logH"] = (log_l * log_h).to_numpy(dtype=float)
    return pd.DataFrame(new, index=cells.index)


def build_features(cohort: Gen13Cohort, *, coordination: str = "frozen", with_resp3d: bool = False,
                   with_logk: bool = False) -> FeatureTable:
    cells = cohort.frame
    paths.assert_bundle_unchanged()
    bundle = pd.read_parquet(paths.BUNDLE_PARQUET)
    ecfp_cols = [c for c in bundle.columns if c.startswith("ecfp_")]
    ligand = (bundle.drop_duplicates("canonical_smiles")
              .set_index("canonical_smiles")[ecfp_cols + list(PHYSCHEM_COLUMNS)])
    chem = pd.read_parquet(paths.CHEMISTRY_MAP_PARQUET).set_index("extractant")
    lig2d = pd.read_parquet(paths.LIG2D_PARQUET).set_index("canonical_smiles")
    lig2d = lig2d.loc[:, [c for c in lig2d.columns if c.startswith("lig2d__")]].apply(
        pd.to_numeric, errors="coerce")
    if "lig2d__rd__Ipc" in lig2d.columns:
        lig2d["lig2d__rd__Ipc"] = np.log10(lig2d["lig2d__rd__Ipc"].clip(lower=1.0))
    coord_path = (paths.GEN122_COORDINATION_PARQUET if coordination == "frozen"
                  else paths.GEN122_COORDINATION_POSTHOC_PARQUET)
    coord = pd.read_parquet(coord_path)
    if coord.index.name != "extractant":
        coord = coord.set_index("extractant")
    coord = coord.apply(pd.to_numeric, errors="coerce")

    ext = cells["extractant"]
    blocks: dict[str, tuple[str, ...]] = {}
    parts: list[pd.DataFrame] = []

    cond = cells[list(cohort.condition_columns)].apply(pd.to_numeric, errors="coerce").astype(float)
    parts.append(cond); blocks["COND"] = tuple(cond.columns)
    ma = _mass_action(cells)
    parts.append(ma); blocks["MASSACT"] = tuple(ma.columns)
    ph = ligand.loc[ext, list(PHYSCHEM_COLUMNS)].astype(float); ph.index = cells.index
    parts.append(ph); blocks["PHYSCHEM"] = tuple(ph.columns)
    dn = chem.loc[ext, list(DONOR_COLUMNS)].apply(pd.to_numeric, errors="coerce").astype(float); dn.index = cells.index
    parts.append(dn); blocks["DONORS"] = tuple(dn.columns)
    fp = ligand.loc[ext, ecfp_cols].astype(float); fp.index = cells.index
    parts.append(fp); blocks["ECFP"] = tuple(fp.columns)
    l2 = lig2d.reindex(ext).astype(float); l2.index = cells.index
    parts.append(l2); blocks["LIG2D"] = tuple(l2.columns)
    co = coord.reindex(ext).astype(float); co.index = cells.index
    co.columns = [c if c.startswith("coord__") else f"coord__{c}" for c in co.columns]
    parts.append(co); blocks["COORD"] = tuple(co.columns)

    if with_resp3d:
        from .features3d import build_resp3d
        r3 = build_resp3d(cohort).astype(float); r3.index = cells.index
        parts.append(r3); blocks[RESP3D_BLOCK] = tuple(r3.columns)
    if with_logk:
        lk = pd.read_parquet(LOGK_PRIOR_PARQUET)
        lk = lk.reindex(ext).astype(float); lk.index = cells.index
        parts.append(lk); blocks[LOGK_BLOCK] = tuple(lk.columns)

    frame = pd.concat(parts, axis=1)
    if frame.columns.duplicated().any():
        raise RuntimeError("duplicate feature columns")
    _assert_allowed(frame.columns)
    return FeatureTable(frame=frame, blocks=blocks)


def block_summary(table: FeatureTable) -> pd.DataFrame:
    rows = []
    for name, cols in table.blocks.items():
        sub = table.frame.loc[:, list(cols)]
        rows.append({"block": name, "n_columns": len(cols),
                     "frac_missing": float(sub.isna().mean().mean()),
                     "n_constant": int((sub.nunique(dropna=True) <= 1).sum())})
    return pd.DataFrame(rows)

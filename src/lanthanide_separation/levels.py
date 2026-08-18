"""Level (``log D``) modelling: cohort, features and estimator.

The pair pipeline (:mod:`.pairs`) models ``log_SF_A_over_B = log D_A − log D_B``.
That answers *"which of two lanthanides does this extractant prefer?"* but not
*"how much of the metal is extracted at these conditions?"* — for which the level
``log D`` is needed.  A ligand can separate beautifully and still extract almost
nothing; the two quantities are independent design constraints.

Levels are a different statistical problem from pairs, and the differences drive
every design choice here:

* **Cohort.** A pair needs two metals measured under *identical* conditions,
  which collapses ~190 extractants to 34.  A level row needs one measurement, so
  the cohort is much larger (92 extractants with ≥ 10 rows) and — more
  importantly — far more chemically diverse: 51 of those 92 are not
  diglycolamides, versus ~6 of the 34 in the pair cohort.
* **Grouping.** The repo's own audit flags ECFP homolog twins as CRITICAL, and
  on this cohort 61 % of rows sit in an ECFP cluster containing more than one
  distinct SMILES (92 extractants → 75 clusters).  Folds are therefore grouped
  on :func:`ecfp_cluster_labels`, not on the extractant, or a "new ligand" fold
  can contain a bit-identical fingerprint from training.
* **What carries the signal.** Measured on the ≥ 10-row cohort, ligand identity
  explains 44 % of ``log D`` variance and metal identity only 5 % — the mirror
  image of the pair target, where the ligand level cancels out.  So the level
  task is mostly *ligand + conditions*, and the unseen-ligand regime is expected
  to be the hard one.
* **No antisymmetry, no transitivity.** Levels need none of the swap machinery,
  and they carry no exact within-cell additivity, so the k-shot "determined row"
  trap of the pair study does not arise.  The per-ligand *offset* is now a
  meaningful free parameter rather than an orientation artefact.
* **Noise floor.** 313 replicated (extractant, condition, metal) cells give a
  median within-cell sd of 0.237 log units, i.e. an MAE floor near 0.19 against
  an overall sd of 1.67.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LEVEL_TARGET_COLUMN = "log_D"

#: Identity columns carried through every level frame (never features).
LEVEL_IDENTITY_COLUMNS: tuple[str, ...] = (
    "row_id", "extractant", "ecfp_cluster", "tanimoto_cluster", "condition_id", "series_id",
    "metal_symbol", "metal_Z", "n_replicates", LEVEL_TARGET_COLUMN,
)
#: Continuous condition columns.  Everything else in ``cond__*`` is a category
#: (acid identity, diluent, additive…); a *series* is one extractant under one
#: categorical setting with only these varied — i.e. a titration.
CONTINUOUS_CONDITION_COLUMNS: tuple[str, ...] = (
    "cond__acid_concentration_M", "cond__contact_time_min", "cond__extractant_concentration_M",
    "cond__metal_concentration_mM", "cond__temperature_C",
)
#: Single-linkage Tanimoto threshold for the strict "new chemotype" grouping.
TANIMOTO_CLUSTER_THRESHOLD = 0.7

#: Feature blocks — one per *property family* the dataset already carries.
#: Every family is evaluated on its own and on top of METAL + COND, so the study
#: answers "which properties predict log D" rather than "does one big model work".
LEVEL_BLOCK_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "COND": ("cond__",),                                  # 64 experimental conditions
    "ECFP": ("ecfp_",),                                   # 2,048-bit ligand fingerprint
    "LIG2D_EXT": ("lig2d__",),                            # 206 extended 2D descriptors (gen4 parquet)
    "COMPLEX_PHYS": ("feat3d__complex_physical__",),      # xTB binding energy, dipole, donor charges
    "POLYHEDRON": ("feat3d__polyhedron",),                # coordination-polyhedron geometry
    "GEOM_COND": ("geom_cond__",),                        # geometry-environment condition bins
    "MASSACTION": ("massact__",),                         # log-concentrations + n*log[L] (mass-action law)
}
LEVEL_METAL_COLUMNS: tuple[str, ...] = ("Atomic Number_metal", "lanthanide_index", "Ionic Radius_metal")
LEVEL_LIGAND_SCALAR_COLUMNS: tuple[str, ...] = (
    "MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "NumAromaticRings", "NumAliphaticRings", "RingCount", "FractionCSP3", "MolLogP",
)
#: Coordination-chemistry descriptors of the ligand: donor-atom census (from the
#: DONOR_TYPES JSON list), denticity, core coordination number, ligand count.
LEVEL_DONOR_SCALAR_COLUMNS: tuple[str, ...] = ("DENTATE", "coreCN", "n_ligs", "n_fill")
DONOR_TYPE_VOCAB: tuple[str, ...] = (
    "O(amide_carbonyl)", "O(ether)", "N(aromatic)", "N(amine)", "S(donor)",
    "O(hydroxyl)", "O(ester_carbonyl)", "O(carbonyl)",
)

#: Legacy composite kept for compatibility with early scripts/tests: fingerprint + scalars.
LEGACY_LIG2D_BLOCKS: tuple[str, ...] = ("ECFP", "PHYSCHEM")

#: Arms.  Naming: ``M`` metal, ``C`` conditions, then the ligand family.
#: Group A — single family alone (what does this property carry by itself?).
#: Group B — family on top of M+C (what does it add once the experiment is known?).
#: Group C — combinations.
LEVEL_ARMS: Mapping[str, tuple[str, ...]] = {
    # A. single families
    "A_metal": ("METAL",),
    "A_cond": ("COND",),
    "A_physchem": ("PHYSCHEM",),
    "A_ecfp": ("ECFP",),
    "A_lig2d_ext": ("LIG2D_EXT",),
    "A_donors": ("DONORS",),
    "A_complex_phys": ("COMPLEX_PHYS",),
    "A_polyhedron": ("POLYHEDRON",),
    # B. each ligand family on top of metal + conditions
    "MC": ("METAL", "COND"),
    "MC_physchem": ("METAL", "COND", "PHYSCHEM"),
    "MC_ecfp": ("METAL", "COND", "ECFP"),
    "MC_lig2d_ext": ("METAL", "COND", "LIG2D_EXT"),
    "MC_donors": ("METAL", "COND", "DONORS"),
    "MC_complex_phys": ("METAL", "COND", "COMPLEX_PHYS"),
    "MC_polyhedron": ("METAL", "COND", "POLYHEDRON"),
    # C. combinations
    "MC_massaction": ("METAL", "COND", "MASSACTION"),
    "MC_ecfp_massaction": ("METAL", "COND", "ECFP", "MASSACTION"),
    "MC_lig2d_ext_massaction": ("METAL", "COND", "LIG2D_EXT", "MASSACTION"),
    "MC_all2d": ("METAL", "COND", "PHYSCHEM", "ECFP", "LIG2D_EXT", "DONORS"),
    "MC_all2d_massaction": ("METAL", "COND", "PHYSCHEM", "ECFP", "LIG2D_EXT", "DONORS", "MASSACTION"),
    "MC_all3d": ("METAL", "COND", "COMPLEX_PHYS", "POLYHEDRON", "GEOM_COND"),
    "MC_everything": ("METAL", "COND", "PHYSCHEM", "ECFP", "LIG2D_EXT", "DONORS",
                      "COMPLEX_PHYS", "POLYHEDRON", "GEOM_COND"),
    "MC_everything_massaction": ("METAL", "COND", "PHYSCHEM", "ECFP", "LIG2D_EXT", "DONORS",
                                 "COMPLEX_PHYS", "POLYHEDRON", "GEOM_COND", "MASSACTION"),
    # legacy names used by the first gen5 draft (kept so old commands still run)
    "L0_metal": ("METAL",),
    "L1_metal_cond": ("METAL", "COND"),
    "L2_metal_cond_lig2d": ("METAL", "COND", "ECFP", "PHYSCHEM"),
    "L3_plus_geom3d": ("METAL", "COND", "ECFP", "PHYSCHEM", "COMPLEX_PHYS", "POLYHEDRON", "GEOM_COND"),
}
LEVEL_BASELINE_ARM = "MC_ecfp"
#: The default set the CLI runs — every family alone, on top of MC, and the combos.
LEVEL_DEFAULT_ARMS: tuple[str, ...] = tuple(a for a in LEVEL_ARMS if not a.startswith("L"))


# --------------------------------------------------------------------------- #
# Cohort
# --------------------------------------------------------------------------- #

def ecfp_cluster_labels(frame: pd.DataFrame, ecfp_columns: Sequence[str]) -> pd.Series:
    """Stable label per bit-identical ECFP fingerprint.

    Distinct SMILES can share a fingerprint (homologues differing only in chain
    length).  Grouping folds on the extractant would then place bit-identical
    training rows in a "held-out ligand" fold; the repo audit marks this CRITICAL
    and measures 49.8 % of pair rows affected (61 % on the level cohort).
    """
    bits = frame[list(ecfp_columns)].to_numpy()
    bits = np.ascontiguousarray(bits.astype(np.int8))
    return pd.Series(
        [hashlib.sha1(row.tobytes()).hexdigest()[:16] for row in bits],
        index=frame.index, name="ecfp_cluster",
    )


def condition_labels(frame: pd.DataFrame, condition_columns: Sequence[str]) -> pd.Series:
    """Stable id for one experimental setup (the full condition vector)."""
    # NaN must hash to a stable token, and `.astype(str)` leaves object columns
    # holding real floats in pandas 3, so normalise through numpy first.
    view = frame[list(condition_columns)].to_numpy(dtype=object)
    rows = ("|".join("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in row)
            for row in view)
    return pd.Series(
        [hashlib.sha1(r.encode()).hexdigest()[:16] for r in rows],
        index=frame.index, name="condition_id",
    )


@dataclass
class LevelData:
    frame: pd.DataFrame
    blocks: dict[str, tuple[str, ...]]
    audit: dict = field(default_factory=dict)

    def block_columns(self, blocks: Iterable[str]) -> tuple[str, ...]:
        out: list[str] = []
        for name in blocks:
            if name not in self.blocks:
                raise KeyError(f"unknown feature block {name!r}; have {sorted(self.blocks)}")
            out.extend(self.blocks[name])
        seen: dict[str, None] = {}
        for c in out:
            seen.setdefault(c, None)
        return tuple(seen)

    def arm_columns(self, arm: str) -> tuple[str, ...]:
        if arm not in LEVEL_ARMS:
            raise KeyError(f"unknown arm {arm!r}; have {sorted(LEVEL_ARMS)}")
        return self.block_columns(LEVEL_ARMS[arm])





def tanimoto_cluster_labels(
    frame: pd.DataFrame, ecfp_columns: Sequence[str], *, threshold: float = TANIMOTO_CLUSTER_THRESHOLD,
) -> pd.Series:
    """Single-linkage clusters of ligands at Tanimoto ≥ ``threshold``.

    Bit-identical grouping (:func:`ecfp_cluster_labels`) still leaves close
    homologues on both sides of a fold: on the level cohort the median nearest
    neighbour of a cluster is Tanimoto 0.72 and 24 of 75 clusters have one at
    ≥ 0.8.  Grouping on these super-clusters is the strict "new chemotype" test.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    ecfp = frame[list(ecfp_columns)]
    keys = ecfp_cluster_labels(frame, ecfp_columns)
    uniq = frame.loc[~keys.duplicated(), list(ecfp_columns)]
    uniq_keys = keys[~keys.duplicated()].to_numpy()
    x = uniq.to_numpy().astype(int)
    inter = x @ x.T
    cnt = x.sum(axis=1)
    denom = cnt[:, None] + cnt[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        tan = np.where(denom > 0, inter / denom, 0.0)
    dist = 1.0 - tan
    np.fill_diagonal(dist, 0.0)
    if len(uniq) < 2:
        labels = np.ones(len(uniq), dtype=int)
    else:
        labels = fcluster(linkage(squareform(dist, checks=False), method="single"),
                          t=1.0 - threshold, criterion="distance")
    lookup = {k: f"tan{int(l):03d}" for k, l in zip(uniq_keys, labels)}
    return pd.Series([lookup[k] for k in keys], index=frame.index, name="tanimoto_cluster")


def series_labels(frame: pd.DataFrame, condition_columns: Sequence[str]) -> pd.Series:
    """One extractant × one categorical condition setting = one measurement series.

    Conditions that differ only in the continuous columns (acid molarity, time,
    temperature…) belong to the same series.  Holding out a *condition* leaves
    its titration neighbours in training; holding out a *series* does not.
    """
    cat = [c for c in condition_columns if c not in CONTINUOUS_CONDITION_COLUMNS]
    view = frame[["extractant"] + cat].to_numpy(dtype=object)
    rows = ("|".join("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in row)
            for row in view)
    return pd.Series([hashlib.sha1(r.encode()).hexdigest()[:16] for r in rows],
                     index=frame.index, name="series_id")


def _one_hot_string_columns(
    frame: pd.DataFrame, *, prefixes: tuple[str, ...], identity: set[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Replace string-typed feature columns by ``<col>=<value>`` indicator columns."""
    mapping: dict[str, list[str]] = {}
    for col in list(frame.columns):
        if col in identity or not col.startswith(prefixes):
            continue
        dtype = frame[col].dtype
        is_string = dtype == object or str(dtype) in ("string", "str") or str(dtype).startswith("string")
        if not is_string:
            continue
        values = frame[col].astype("string")
        dummies = pd.get_dummies(values, prefix=col, prefix_sep="=", dtype=float)
        dummies.loc[values.isna(), :] = np.nan
        frame = pd.concat([frame.drop(columns=[col]), dummies], axis=1)
        mapping[col] = list(dummies.columns)
    return frame, mapping


def _attach_donor_census(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn the ``DONOR_TYPES`` JSON list into ``donor__<type>`` counts plus a total.

    A ligand's donor-atom census (three amide O and one ether O for a DGA, aromatic
    N for BTBP-type ligands…) is the coordination chemist's first descriptor and is
    orthogonal to a fingerprint bit pattern.
    """
    if "DONOR_TYPES" not in frame.columns:
        return frame
    import json as _json

    def census(raw) -> dict[str, float]:
        out = {f"donor__{t}": 0.0 for t in DONOR_TYPE_VOCAB}
        out["donor__n_total"] = np.nan
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return out
        try:
            items = _json.loads(raw) if isinstance(raw, str) else list(raw)
        except (ValueError, TypeError):
            return out
        for t in items:
            key = f"donor__{t}"
            out[key] = out.get(key, 0.0) + 1.0
        out["donor__n_total"] = float(len(items))
        return out

    census_frame = pd.DataFrame([census(v) for v in frame["DONOR_TYPES"].tolist()], index=frame.index)
    census_frame = census_frame[[c for c in census_frame.columns if c in {f"donor__{t}" for t in DONOR_TYPE_VOCAB} | {"donor__n_total"}]]
    return pd.concat([frame, census_frame], axis=1)


def _attach_mass_action(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode the solvent-extraction mass-action law as features.

    For a neutral extractant ``L`` pulling ``Ln(III)`` out of a nitrate medium the
    extraction equilibrium is

        Ln(3+) + 3 NO3(-) + n L(org)  <->  Ln(NO3)3 . Ln(org)

    whose mass-action expression is **linear in the LOGARITHM** of the two
    concentrations that are actually varied:

        log D = log K_ex + n log[L] + 3 log[NO3-]

    ``n`` is the solvation number (2-4 for most extractants) and ``log K_ex`` is a
    per-ligand offset.  Fitted on this cohort where an extractant titration exists
    (15 metal-series with a clean fit) the measured slope is 2.64, IQR
    [2.36, 2.88], 100 % inside the chemically admissible 1.5-4.5, median linear
    R2 0.985; the acid slope is 1.93 over 146 series.  The law holds here.

    The raw ``cond__*`` columns carry molarity, spanning 4-9 orders of magnitude,
    so an axis-aligned tree has to approximate a logarithm with a staircase of
    splits — which is exactly what produces the shrunken predictions and the
    unreachable tails seen in gen5 (dispersion 0.605 on a new ligand).  Supplying
    the logarithms, and the ``n log[L]`` product the law asks for, lets one split
    do what many were doing badly.

    The block never replaces the raw columns; it is additive, so ``COND``-only
    arms stay exactly as they were and the contribution is measurable as an
    ablation.
    """
    out = frame
    new: dict[str, np.ndarray] = {}
    logs: dict[str, pd.Series] = {}
    for col in CONTINUOUS_CONDITION_COLUMNS:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        positive = values.where(values > 0)
        if positive.notna().sum() == 0:
            continue
        series = np.log10(positive)
        logs[col] = series
        new[f"massact__log10_{col}"] = series.to_numpy(dtype=float)
    log_l = logs.get("cond__extractant_concentration_M")
    log_h = logs.get("cond__acid_concentration_M")
    # n * log[L]: the solvation number multiplies the extractant term, and n tracks
    # the ligand's denticity / the metal's coordination number.  Trees cannot form a
    # product from its factors, so hand it over directly.
    if log_l is not None:
        for col in ("DENTATE", "coreCN"):
            if col in out.columns:
                new[f"massact__logL_x_{col}"] = (
                    log_l * pd.to_numeric(out[col], errors="coerce")).to_numpy(dtype=float)
        if log_h is not None:
            new["massact__logL_x_logH"] = (log_l * log_h).to_numpy(dtype=float)
    if not new:
        return out
    return out.assign(**new)


def build_level_dataset(
    source: pd.DataFrame,
    *,
    min_rows_per_extractant: int = 10,
    require_geometry: bool = False,
    replicate_policy: str = "mean",
    drop_below_log_d: float | None = -6.0,
    ligand_descriptors: pd.DataFrame | None = None,
) -> LevelData:
    """Build the level cohort: one row per (extractant, condition, metal).

    ``replicate_policy``:

    * ``"mean"`` (default) — average replicated cells and record ``n_replicates``.
      Keeps every cell exactly once so macro metrics are not distorted by how
      often a cell happens to have been repeated.
    * ``"unique"`` — keep the first occurrence (matches the pair pipeline).
    * ``"all"`` — keep replicates as separate rows (only for a noise-floor study;
      leaks a cell across folds if folds are not grouped on the cell).

    ``drop_below_log_d`` removes values at or below a detection-limit-like floor
    (three rows sit at −12.5 while the next lowest is −5.0); pass ``None`` to keep
    them.  ``require_geometry`` is off by default: geometry is not used by the
    2D arms and requiring it would shrink the cohort for no benefit.

    ``ligand_descriptors`` is the gen4 extended-2D table (one row per
    ``canonical_smiles``, ``lig2d__*`` columns); when given it is joined on the
    extractant and exposed as the ``LIG2D_EXT`` block.
    """
    if replicate_policy not in ("mean", "unique", "all"):
        raise ValueError(f"unknown replicate_policy {replicate_policy!r}")
    frame = source.copy()
    audit: dict = {"source_rows": int(len(frame)), "source_extractants": int(frame["canonical_smiles"].nunique())}

    if require_geometry and "geometry_ok" in frame.columns:
        frame = frame[frame["geometry_ok"].astype(bool)]
        audit["after_geometry"] = int(len(frame))
    if drop_below_log_d is not None:
        dropped = int((frame[LEVEL_TARGET_COLUMN] <= drop_below_log_d).sum())
        frame = frame[frame[LEVEL_TARGET_COLUMN] > drop_below_log_d]
        audit["dropped_below_floor"] = dropped
        audit["log_d_floor"] = float(drop_below_log_d)

    ecfp_columns = tuple(c for c in frame.columns if c.startswith("ecfp_"))
    condition_columns = tuple(c for c in frame.columns if c.startswith("cond__"))
    if not ecfp_columns or not condition_columns:
        raise ValueError("source frame lacks ecfp_* or cond__* columns")
    frame = frame.assign(
        extractant=frame["canonical_smiles"].astype(str),
        metal_symbol=frame["metal_symbol"].astype(str),
        metal_Z=frame["Atomic Number_metal"].astype(float),
        condition_id=condition_labels(frame, condition_columns),
        ecfp_cluster=ecfp_cluster_labels(frame, ecfp_columns),
    )
    frame = frame.assign(
        tanimoto_cluster=tanimoto_cluster_labels(frame, ecfp_columns),
        series_id=series_labels(frame, condition_columns),
    )
    if ligand_descriptors is not None:
        desc = ligand_descriptors.copy()
        if "canonical_smiles" not in desc.columns:
            raise ValueError("ligand_descriptors needs a canonical_smiles column")
        desc = desc.drop_duplicates("canonical_smiles").assign(extractant=lambda d: d["canonical_smiles"].astype(str))
        desc = desc[["extractant"] + [c for c in desc.columns if c.startswith("lig2d__")]]
        frame = frame.merge(desc, on="extractant", how="left", validate="many_to_one")
        audit["ligand_descriptor_coverage"] = float(
            frame[[c for c in desc.columns if c.startswith("lig2d__")][:1]].notna().mean().iloc[0]) if desc.shape[1] > 1 else 0.0
    frame = _attach_donor_census(frame)
    frame = _attach_mass_action(frame)

    # Eligibility counts unique (condition, metal) cells, not raw replicate rows,
    # so a ligand measured 8× at one point does not enter as an 8-row cluster.
    counts = frame.drop_duplicates(["extractant", "condition_id", "metal_symbol"]).groupby("extractant").size()
    keep = counts[counts >= int(min_rows_per_extractant)].index
    n_ext_before = int(counts.size)
    cells_before = int(counts.sum())
    frame = frame[frame["extractant"].isin(keep)]
    audit["min_rows_per_extractant"] = int(min_rows_per_extractant)
    audit["after_min_rows"] = int(len(frame))
    # Recorded because this filter is where chemical diversity is lost: the rarely
    # measured extractants are disproportionately the unusual chemotypes.
    audit["extractants_dropped_by_min_rows"] = n_ext_before - int(len(keep))
    # In CELLS (unique extractant x condition x metal), the same unit the filter uses
    # and the same unit the final cohort is counted in -- not raw replicate rows.
    audit["cells_dropped_by_min_rows"] = cells_before - int(counts[keep].sum())
    audit["extractants_before_min_rows"] = n_ext_before

    cell = ["extractant", "condition_id", "metal_symbol"]
    if replicate_policy == "all":
        frame = frame.assign(n_replicates=1)
    else:
        sizes = frame.groupby(cell)[LEVEL_TARGET_COLUMN].transform("size")
        if replicate_policy == "mean":
            target = frame.groupby(cell)[LEVEL_TARGET_COLUMN].transform("mean")
            frame = frame.assign(**{LEVEL_TARGET_COLUMN: target})
        frame = frame.assign(n_replicates=sizes.astype(int))
        frame = frame.drop_duplicates(subset=cell, keep="first")
    audit["replicate_policy"] = replicate_policy
    audit["after_replicates"] = int(len(frame))

    frame = frame.reset_index(drop=True)
    frame = frame.assign(row_id=[
        hashlib.sha1(f"{e}|{c}|{m}".encode()).hexdigest()[:16]
        for e, c, m in zip(frame["extractant"], frame["condition_id"], frame["metal_symbol"])
    ])

    blocks: dict[str, tuple[str, ...]] = {
        "METAL": tuple(c for c in LEVEL_METAL_COLUMNS if c in frame.columns),
        "PHYSCHEM": tuple(c for c in LEVEL_LIGAND_SCALAR_COLUMNS if c in frame.columns),
        "DONORS": tuple(c for c in frame.columns if c.startswith("donor__"))
                  + tuple(c for c in LEVEL_DONOR_SCALAR_COLUMNS if c in frame.columns),
    }
    identity = set(LEVEL_IDENTITY_COLUMNS)
    # String-valued feature columns (the geom_cond__ bins are categorical labels)
    # would be coerced to NaN by the numeric cast and silently vanish, so they
    # are one-hot encoded here, once, at cohort-build time.
    frame, onehot_map = _one_hot_string_columns(frame, prefixes=tuple(p for ps in LEVEL_BLOCK_PREFIXES.values() for p in ps),
                                                identity=identity)
    audit["one_hot_encoded"] = {k: len(v) for k, v in onehot_map.items()}
    for name, prefixes in LEVEL_BLOCK_PREFIXES.items():
        cols = [c for c in frame.columns if c.startswith(prefixes) and c not in identity]
        if name == "ECFP":  # ecfp_<int> bits only; never the ecfp_cluster identity label
            cols = [c for c in cols if c[len("ecfp_"):].isdigit()]
        blocks[name] = tuple(cols)
    # Columns with no observed value at all carry nothing and would only produce
    # imputer warnings; drop them and *record* them, because some are exactly the
    # descriptors a reader expects to have been tested (xTB binding energy,
    # HOMO/LUMO, strain energy, every geom_cond__ bin are all-null in this table).
    empty_by_block: dict[str, list[str]] = {}
    for name, cols in list(blocks.items()):
        empty = [c for c in cols if frame[c].isna().all()]
        if empty:
            empty_by_block[name] = empty
            blocks[name] = tuple(c for c in cols if c not in set(empty))
    audit["all_null_columns_dropped"] = empty_by_block
    blocks = {k: v for k, v in blocks.items() if v}

    feature_columns = sorted({c for cols in blocks.values() for c in cols})
    keep_columns = list(LEVEL_IDENTITY_COLUMNS) + feature_columns
    missing = [c for c in LEVEL_IDENTITY_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"level frame is missing identity columns {missing}")
    frame = frame[keep_columns]

    audit.update({
        "rows": int(len(frame)),
        "extractants": int(frame["extractant"].nunique()),
        "ecfp_clusters": int(frame["ecfp_cluster"].nunique()),
        "tanimoto_clusters": int(frame["tanimoto_cluster"].nunique()),
        "conditions": int(frame["condition_id"].nunique()),
        "series": int(frame["series_id"].nunique()),
        "metals": int(frame["metal_symbol"].nunique()),
        "block_sizes": {k: len(v) for k, v in blocks.items()},
        "target_sd": float(frame[LEVEL_TARGET_COLUMN].std()),
        "replicated_cells": int((frame["n_replicates"] > 1).sum()),
        "largest_extractant_share": float(frame["extractant"].value_counts(normalize=True).iloc[0]),
        "largest_ecfp_cluster_share": float(frame["ecfp_cluster"].value_counts(normalize=True).iloc[0]),
        "largest_tanimoto_cluster_share": float(frame["tanimoto_cluster"].value_counts(normalize=True).iloc[0]),
    })
    return LevelData(frame=frame, blocks=blocks, audit=audit)


def replicate_noise_floor(source: pd.DataFrame) -> dict[str, float]:
    """Within-cell reproducibility of ``log D`` — the error no model can beat."""
    condition_columns = [c for c in source.columns if c.startswith("cond__")]
    frame = source.assign(
        condition_id=condition_labels(source, condition_columns),
        extractant=source["canonical_smiles"].astype(str),
    )
    stats = frame.groupby(["extractant", "condition_id", "metal_symbol"])[LEVEL_TARGET_COLUMN].agg(["size", "std", "mean"])
    rep = stats[stats["size"] > 1]
    if rep.empty:
        return {"replicated_cells": 0}
    devs = frame.merge(
        rep[["mean"]], left_on=["extractant", "condition_id", "metal_symbol"], right_index=True, how="inner",
    )
    # Pooled within-cell sd (ddof = 1 per cell) is the unbiased reproducibility;
    # the MAE floor a perfect model would still incur against a single new
    # measurement is sd·sqrt(2/π) ≈ 0.80·sd.  Median is robust to the handful of
    # >2-log "replicates" that are probably not true repeats.
    pooled_var = float(((rep["size"] - 1) * rep["std"] ** 2).sum() / (rep["size"] - 1).sum())
    return {
        "replicated_cells": int(len(rep)),
        "median_within_cell_sd": float(rep["std"].median()),
        "pooled_within_cell_sd": float(np.sqrt(pooled_var)),
        "mae_floor_median_sd": float(rep["std"].median() * np.sqrt(2.0 / np.pi)),
        "mae_floor_pooled_sd": float(np.sqrt(pooled_var) * np.sqrt(2.0 / np.pi)),
        "cells_with_sd_over_1": int((rep["std"] > 1.0).sum()),
    }


# --------------------------------------------------------------------------- #
# Estimator
# --------------------------------------------------------------------------- #

def group_balanced_weights(groups: Sequence) -> np.ndarray:
    """Equal total weight per group, so one huge ligand cannot dominate the fit."""
    arr = np.asarray([str(g) for g in groups])
    _, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
    return (1.0 / counts[inverse]).astype(float)


LEARNERS: tuple[str, ...] = ("extratrees", "hgb", "ridge")


@dataclass
class LevelForestParameters:
    """Learner settings.  ``learner`` selects the model family:

    * ``extratrees`` — same family as the pair champion (default);
    * ``hgb`` — histogram gradient boosting (native NaN handling, MAE-friendly);
    * ``ridge`` — standardised linear model, the sanity floor for "is this
      non-linear at all?".
    """

    n_estimators: int = 400
    max_features: float = 0.30
    min_samples_leaf: int = 2
    random_state: int = 42
    n_jobs: int = -1
    learner: str = "extratrees"
    learning_rate: float = 0.05
    ridge_alpha: float = 10.0


class DropAllNaNColumns(BaseEstimator, TransformerMixin):
    """Drop columns that hold no observed value at all in the training fold.

    Keeps every other NaN intact, so a learner with native missing-value support
    still sees (and learns from) partial missingness.  Fitted inside the fold.
    """

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.keep_ = ~np.all(np.isnan(arr), axis=0)
        self.n_features_in_ = arr.shape[1]
        if not self.keep_.any():
            raise ValueError("every feature column is empty in this training fold")
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} columns, got {arr.shape[1]}")
        return arr[:, self.keep_]


class LevelRegressor:
    """ExtraTrees on a level row, with median imputation and missing indicators.

    Deliberately the same estimator family as the pair champion so that any
    difference between the level and pair results is about the *target*, not the
    learner.  Unlike the pair model there is no swap augmentation: a level row
    has no A/B orientation to be antisymmetric about.
    """

    def __init__(self, feature_columns: Iterable[str], parameters: LevelForestParameters | None = None) -> None:
        self.feature_columns = tuple(feature_columns)
        self.parameters = parameters or LevelForestParameters()
        self.pipeline: Pipeline | None = None
        self._y_range: tuple[float, float] | None = None

    def _new_pipeline(self) -> Pipeline:
        p = self.parameters
        # HGB keeps NaN (it learns a default split direction), so it gets no imputer —
        # but its binner cannot handle a column that is *entirely* NaN in training, and
        # raises "window shape cannot be larger than input array shape".  That happens
        # under unseen_chemotype, where holding out the 67.5%-of-rows super-cluster can
        # leave a descriptor with no observed value at all.  The imputer-bearing learners
        # survive it because SimpleImputer drops such columns (with a UserWarning); this
        # step gives HGB the same protection and nothing more.
        if p.learner == "extratrees":
            model = ExtraTreesRegressor(
                n_estimators=p.n_estimators, max_features=p.max_features,
                min_samples_leaf=p.min_samples_leaf, random_state=p.random_state, n_jobs=p.n_jobs,
            )
            return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", model)])
        if p.learner == "hgb":
            model = HistGradientBoostingRegressor(
                max_iter=p.n_estimators, learning_rate=p.learning_rate, min_samples_leaf=max(5, p.min_samples_leaf),
                loss="absolute_error", random_state=p.random_state,
            )
            return Pipeline([("drop_empty", DropAllNaNColumns()), ("model", model)])
        if p.learner == "ridge":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=p.ridge_alpha)),
            ])
        raise ValueError(f"unknown learner {p.learner!r}; choose from {LEARNERS}")

    def fit(self, frame: pd.DataFrame, target: Iterable[float], groups: Iterable | None = None) -> "LevelRegressor":
        x = _as_float_frame(frame, self.feature_columns)
        y = np.asarray(list(target), dtype=float)
        if len(x) != len(y):
            raise ValueError("frame and target must have equal length")
        weights = group_balanced_weights(list(groups)) if groups is not None else None
        self.pipeline = self._new_pipeline()
        self.pipeline.fit(x, y, model__sample_weight=weights)
        span = float(y.max() - y.min())
        self._y_range = (float(y.min()) - 0.5 * span, float(y.max()) + 0.5 * span)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model has not been fitted")
        pred = self.pipeline.predict(_as_float_frame(frame, self.feature_columns))
        # A linear model extrapolates without bound on out-of-range descriptors
        # (one held-out ligand blew a ridge fit up to MAE ~4e6).  Clamp every
        # learner to the training target range plus a margin; trees never leave it.
        if self._y_range is not None:
            lo, hi = self._y_range
            pred = np.clip(pred, lo, hi)
        return pred

    def feature_importance_frame(self) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("model has not been fitted")
        model = self.pipeline.named_steps["model"]
        if not hasattr(model, "feature_importances_"):
            raise AttributeError(f"learner {self.parameters.learner!r} has no impurity importances")
        names = self.pipeline.named_steps["imputer"].get_feature_names_out(self.feature_columns)
        imp = model.feature_importances_.astype(float)
        return pd.DataFrame({"feature": names, "importance": imp}).sort_values(
            "importance", ascending=False, ignore_index=True)


def _as_float_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing feature columns: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    return frame[list(columns)].apply(pd.to_numeric, errors="coerce").astype(float)


def shuffle_block(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    rng: np.random.Generator,
    within: str | None = None,
) -> pd.DataFrame:
    """Negative control: permute a feature block, destroying its association with y.

    With ``within`` the permutation happens inside each level of that column, so
    a block can be shuffled while preserving coarser structure.  A block that
    still "helps" after shuffling was never carrying signal.
    """
    out = frame.copy()
    cols = list(columns)
    if within is None:
        order = rng.permutation(len(out))
        out[cols] = out[cols].to_numpy()[order]
        return out
    for _, idx in out.groupby(within, sort=False).groups.items():
        idx = np.asarray(list(idx))
        order = rng.permutation(len(idx))
        out.loc[idx, cols] = out.loc[idx[order], cols].to_numpy()
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def equal_group_macro_mae(truth: Iterable[float], prediction: Iterable[float], groups: Iterable) -> float:
    """Mean over groups of the per-group MAE (one ligand cluster = one vote)."""
    table = pd.DataFrame({
        "err": np.abs(np.asarray(list(truth), dtype=float) - np.asarray(list(prediction), dtype=float)),
        "group": [str(g) for g in groups],
    })
    if table.empty:
        raise ValueError("no rows to score")
    return float(table.groupby("group", sort=False)["err"].mean().mean())


def level_metric_table(
    predictions: pd.DataFrame,
    arms: Iterable[str],
    *,
    group_column: str = "ecfp_cluster",
    baseline_arm: str = LEVEL_BASELINE_ARM,
    min_group_rows: int = 8,
    min_group_sd: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overall and per-group metrics for every arm.

    Returns ``(overall, per_group)``.  ``macro`` weights each ligand cluster
    equally; ``pooled`` is row-weighted and therefore dominated by the largest
    ligand — both are reported because they can disagree.  ``pooled_r2`` mostly
    measures *between*-ligand ranking (ligand identity is 44 % of variance).

    Two within-ligand R² are reported, and they answer different questions:

    * ``within_ligand_r2`` — **deployable**: ``1 - SSE / SST_within`` where SSE is
      the raw squared error and SST_within is the variance of the truth around
      each ligand's own mean.  Positive means the model beats "predict this
      ligand's mean" without being told that mean.  This is what a chemist
      choosing conditions for a *new* ligand gets.
    * ``within_ligand_r2_shape`` — **oracle-offset**: both prediction and truth
      are centred on their own per-ligand means first.  It scores only the shape
      of the condition response and grants the model a free per-ligand offset it
      does not have at deployment.  A constant-per-ligand predictor scores 0
      here and negative on the deployable version.  Report both; never call the
      shape one "within-ligand R²" on its own.

    Rows with a missing prediction raise — silent partial scoring is a bug.
    """
    arms = list(arms)
    y = predictions[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    per_rows: list[dict] = []
    overall: list[dict] = []
    ext = predictions["extractant"].astype(str).to_numpy() if "extractant" in predictions.columns else None
    for arm in arms:
        p = predictions[f"prediction_{arm}"].to_numpy(dtype=float)
        if not np.isfinite(p).all():
            raise ValueError(f"arm {arm!r} has {int((~np.isfinite(p)).sum())} missing predictions")
        err = np.abs(p - y)
        sub = pd.DataFrame({"group": predictions[group_column].astype(str), "err": err, "y": y, "p": p})
        for name, g in sub.groupby("group", sort=True):
            sd = float(g["y"].std(ddof=0))
            ok = len(g) >= min_group_rows and sd >= min_group_sd
            sse = float(((g["p"] - g["y"]) ** 2).sum())
            sst = float(((g["y"] - g["y"].mean()) ** 2).sum())
            per_rows.append({
                "group": name, "arm": arm, "n_rows": int(len(g)), "mae": float(g["err"].mean()),
                "bias": float((g["y"] - g["p"]).mean()),
                "r2": (1.0 - sse / sst) if (ok and sst > 0) else np.nan,
            })
        macro = float(sub.groupby("group", sort=False)["err"].mean().mean())
        sse = float(((p - y) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum())
        within = np.nan
        within_shape = np.nan
        if ext is not None:
            d = pd.DataFrame({"e": ext, "y": y, "p": p})
            yc = d["y"] - d.groupby("e")["y"].transform("mean")
            sst_w = float((yc ** 2).sum())
            if sst_w > 0:
                # deployable: raw error against the within-ligand variance — no free offset
                within = 1.0 - sse / sst_w
                # shape-only: both sides centred on their own per-ligand means (oracle offset)
                pc = d["p"] - d.groupby("e")["p"].transform("mean")
                within_shape = 1.0 - float(((yc - pc) ** 2).sum()) / sst_w
        overall.append({
            "arm": arm, "n_rows": int(len(y)), "n_groups": int(sub["group"].nunique()),
            "macro_mae": macro, "pooled_mae": float(err.mean()),
            "pooled_r2": 1.0 - sse / sst if sst > 0 else np.nan,
            "within_ligand_r2": within,
            "within_ligand_r2_shape": within_shape,
            "pooled_rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "prediction_dispersion_ratio": float(np.std(p) / np.std(y)) if np.std(y) > 0 else np.nan,
            "median_abs_error": float(np.median(err)),
            "frac_within_0_5_log": float(np.mean(err <= 0.5)),
            "frac_within_1_log": float(np.mean(err <= 1.0)),
        })
    per_group = pd.DataFrame(per_rows)
    over = pd.DataFrame(overall)
    if baseline_arm in arms:
        base = per_group[per_group["arm"] == baseline_arm].set_index("group")["mae"]
        per_group["baseline_mae"] = per_group["group"].map(base)
        per_group["delta_mae_baseline_minus_arm"] = per_group["baseline_mae"] - per_group["mae"]
        base_macro = float(over.loc[over["arm"] == baseline_arm, "macro_mae"].iloc[0])
        over["delta_macro_vs_baseline"] = base_macro - over["macro_mae"]
    return over, per_group


def paired_group_bootstrap(
    predictions: pd.DataFrame,
    comparisons: Mapping[str, tuple[str, str]],
    *,
    group_column: str = "ecfp_cluster",
    resample_column: str | None = None,
    replicates: int = 5000,
    seed: int = 8675309,
) -> pd.DataFrame:
    """Bootstrap the per-group MAE delta, resampling whole held-out blocks.

    ``group_column`` defines the *scoring* unit: the point estimate is the mean
    over those groups of the per-group MAE difference, so it equals the macro
    delta exactly.  ``resample_column`` defines the *independence* unit that is
    actually resampled, and it must be the unit the folds held out — otherwise
    the interval treats correlated groups as independent and comes out too
    narrow.  Under ``unseen_chemotype`` the folds hold out Tanimoto super-clusters
    while the metric is still per ECFP cluster, and resampling ECFP clusters
    there understates the width by ~1.35x.  Defaults to ``group_column``, which
    is the right choice whenever the two coincide.

    One index matrix is drawn and shared by every comparison, so the intervals
    are mutually comparable and do not depend on the order of ``comparisons``
    (a per-comparison draw off one shared RNG stream made a pair's CI a function
    of its position in the dict).
    """
    resample_column = resample_column or group_column
    y = predictions[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    groups = predictions[group_column].astype(str).to_numpy()
    names = np.unique(groups)
    idx_by_group = [np.flatnonzero(groups == g) for g in names]
    # Which resampling block does each scoring group belong to?  A group must sit
    # in exactly one block for the block bootstrap to be well defined.
    blocks = predictions[resample_column].astype(str).to_numpy()
    block_of = []
    for idx in idx_by_group:
        b = np.unique(blocks[idx])
        if b.size != 1:
            raise ValueError(
                f"scoring group spans {b.size} {resample_column!r} blocks; "
                f"{group_column!r} must nest inside {resample_column!r}")
        block_of.append(b[0])
    block_of = np.asarray(block_of)
    block_names = np.unique(block_of)
    members = [np.flatnonzero(block_of == b) for b in block_names]

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(block_names), size=(replicates, len(block_names)))
    # Precompute, per replicate, the scoring-group indices implied by the drawn blocks.
    take = [np.concatenate([members[j] for j in row]) for row in picks]

    rows = []
    for label, (reference, candidate) in comparisons.items():
        pr = predictions[f"prediction_{reference}"].to_numpy(dtype=float)
        pc = predictions[f"prediction_{candidate}"].to_numpy(dtype=float)
        deltas = np.array([
            np.abs(pr[i] - y[i]).mean() - np.abs(pc[i] - y[i]).mean() for i in idx_by_group
        ])
        draws = np.array([deltas[t].mean() for t in take])
        rows.append({
            "comparison": label, "reference": reference, "candidate": candidate,
            "point_delta_mae": float(deltas.mean()),
            "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)),
            "p_worse_one_sided": float((1 + np.sum(draws <= 0)) / (1 + replicates)),
            "groups_improved": int(np.sum(deltas > 0)), "groups_total": int(deltas.size),
            "bootstrap_unit": group_column, "bootstrap_resample_unit": resample_column,
            "bootstrap_blocks": int(len(block_names)), "bootstrap_replicates": int(replicates),
        })
    return pd.DataFrame(rows)

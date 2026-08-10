"""Build leakage-auditable lanthanide selectivity pairs.

The model target is a directly observed, condition-matched separation factor:

    log_SF_A_over_B = log_D_A - log_D_B

where A is the lighter lanthanide and B is the heavier lanthanide.  The primary
``all`` scope includes every observed A/B combination under exactly matched
extractant and experimental conditions.  A legacy ``adjacent`` scope remains
available for reproducing the original nearest-neighbour benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .electronic import (
    COMPLEX_ELECTRONIC_SOURCES,
    COMPOSITION_MATCH_COLUMNS,
    DERIVED_COMPLEX_ELECTRONIC_NAME,
    ELECTRONIC_EVEN_PREFIX,
    ELECTRONIC_ODD_PREFIX,
    ELECTRONIC_PREFIX,
    ENERGY_SOURCE,
    electronic_source_columns,
    pair_electronic_features,
)
from .geometry_descriptors import BLOCK_PREFIXES, validate_blocks
from .pair_response import (
    PAIR_RESPONSE_EVEN_PREFIX,
    PAIR_RESPONSE_ODD_PREFIX,
    PAIR_RESPONSE_PREFIX,
    PAIR_RESPONSE_QUANTITIES,
    pair_response_features,
    pair_response_source_columns,
)


LANTHANIDE_Z: dict[str, int] = {
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
}

MOLECULAR_DESCRIPTOR_COLUMNS = (
    "MolWt",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "NumAliphaticRings",
    "RingCount",
    "FractionCSP3",
    "MolLogP",
)

# These global complex quantities are either coordinate-frame dependent or are
# not comparable across complexes containing different metal elements.  Their
# inclusion can create a shortcut for metal identity without representing the
# local coordination response that is relevant to selectivity.
EXCLUDED_3D_TOKENS = (
    "__dipole_x",
    "__dipole_y",
    "__dipole_z",
    "__complex_total_energy_eV",
    "__complex_free_energy_eV",
)

# The primary feature set deliberately excludes radial-rank donor identities and
# rank-indexed charges/angles. They are discontinuous when two donors swap rank
# and, in this bundle, a few rare non-zero values can dominate a tree model.
COMPACT_3D_COLUMNS = {
    "feat3d__complex_physical__coordination_number",
    "feat3d__complex_physical__observed_donors_within_3p10A",
    "feat3d__complex_physical__dipole_magnitude",
    "feat3d__complex_physical__metal_partial_charge",
    "feat3d__complex_physical__ln_donor_distance_mean",
    "feat3d__complex_physical__ln_donor_distance_std",
    "feat3d__complex_physical__ln_donor_distance_min",
    "feat3d__complex_physical__ln_donor_distance_max",
    "feat3d__complex_physical__donor_partial_charge_mean",
    "feat3d__complex_physical__donor_partial_charge_std",
    "feat3d__complex_physical__donor_partial_charge_min",
    "feat3d__complex_physical__donor_partial_charge_max",
    "feat3d__polyhedron_scalars__coreCN_donor_gap",
}
DERIVED_3D_PREFIX = "feat3d__derived_invariant__"
DONOR_ELEMENT_ATOMIC_NUMBERS: dict[str, int] = {
    "N": 7,
    "O": 8,
    "P": 15,
    "S": 16,
}

# Dataset-specific, provenance-backed quarantine.  The first rule catches rows
# whose canonical structure, ECFP and 3D complex are TODGA although the named
# extractant is a different ligand or a mixture.  The three IDs are sentinel-like
# values separated by more than seven log units from the next observation.
TODGA_SMILES = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"
KNOWN_CENSORED_SAFE_IDS = {
    "Er_SAFE:13975",
    "Er_SAFE:13981",
    "Er_SAFE:13992",
}

PAIR_TARGET_COLUMN = "log_SF_A_over_B"
EXTRACTANT_COLUMN = "canonical_smiles"
PAIR_SCOPES = ("all", "adjacent")


@dataclass(frozen=True)
class PairDataset:
    """Paired data plus the exact feature and provenance contracts."""

    frame: pd.DataFrame
    baseline_columns: tuple[str, ...]
    delta3d_columns: tuple[str, ...]
    audit: dict[str, Any]
    quarantine: pd.DataFrame
    descriptor_columns: tuple[str, ...] = ()
    symmetric3d_columns: tuple[str, ...] = ()
    pair_response_columns: tuple[str, ...] = ()
    electronic_columns: tuple[str, ...] = ()

    @property
    def full_columns(self) -> tuple[str, ...]:
        """The frozen Delta3D contract; unchanged when descriptors are requested."""

        return self.baseline_columns + self.delta3d_columns

    @property
    def extended_columns(self) -> tuple[str, ...]:
        """The frozen contract plus the opt-in metal-site descriptor block."""

        return self.baseline_columns + self.delta3d_columns + self.descriptor_columns

    @property
    def symmetric_extended_columns(self) -> tuple[str, ...]:
        """``extended_columns`` plus the opt-in swap-symmetric 3D block.

        ``sym3d__X = (X_A + X_B) / 2`` is invariant under the A/B swap, exactly
        like the existing ``pair__Z_mean`` and ``pair__ionic_radius_mean``
        contrasts, so :func:`reverse_pair_features` leaves it untouched and the
        antisymmetry of the prediction is preserved.
        """

        return self.extended_columns + self.symmetric3d_columns

    @property
    def extension_columns(self) -> tuple[str, ...]:
        """Every opt-in extension block declared after the frozen contract.

        These are additions only.  When no extension is requested this is empty
        and ``symmetric_extended_columns`` is the whole model contract, so the
        A0--A6 arms are bit-for-bit what they were.
        """

        return (
            self.symmetric3d_columns
            + self.pair_response_columns
            + self.electronic_columns
        )

    @property
    def all_model_columns(self) -> tuple[str, ...]:
        """The frozen contract plus every requested extension block."""

        return self.extended_columns + self.extension_columns


def _required_columns() -> set[str]:
    return {
        EXTRACTANT_COLUMN,
        "extractant_name",
        "extractant_group",
        "metal",
        "log_D",
        "geometry_ok",
        "geometry_key",
        "build_id",
        "geometry_feature_build_id",
        "vr_graph_index",
        "safe_exp_id",
        "Ionic Radius_metal",
    }


def _validate_source(df: pd.DataFrame) -> None:
    missing = sorted(_required_columns() - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if not any(str(c).startswith("cond__") for c in df.columns):
        raise ValueError("No cond__ columns found; exact condition matching is impossible.")
    if not any(str(c).startswith("feat3d__") for c in df.columns):
        raise ValueError("No feat3d__ columns found.")
    if df[EXTRACTANT_COLUMN].isna().any():
        raise ValueError("canonical_smiles contains missing values.")
    if df["metal"].isna().any():
        raise ValueError("metal contains missing values.")
    if df["geometry_ok"].isna().any() or not pd.api.types.is_bool_dtype(
        df["geometry_ok"]
    ):
        raise ValueError("geometry_ok must be a non-null boolean column.")
    # Rows without an accepted geometry legitimately have no VR graph.  A
    # supplied value is always validated, while accepted rows must supply one.
    raw_vr_indices = df["vr_graph_index"]
    vr_index_provided = (
        raw_vr_indices.notna()
        & raw_vr_indices.astype("string").str.strip().ne("").fillna(False)
    ).to_numpy(dtype=bool)
    vr_indices = pd.to_numeric(
        raw_vr_indices.where(vr_index_provided), errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan)
    valid_vr_index = (
        np.isfinite(vr_indices)
        & np.equal(vr_indices, np.floor(vr_indices))
        & (vr_indices >= 0)
    )
    if np.any(vr_index_provided & ~valid_vr_index):
        raise ValueError(
            "Every supplied vr_graph_index must be a finite nonnegative integer."
        )
    accepted_geometry = df["geometry_ok"].to_numpy(dtype=bool)
    if np.any(accepted_geometry & ~valid_vr_index):
        raise ValueError(
            "Every geometry_ok=True row must have a finite nonnegative "
            "vr_graph_index."
        )


def apply_default_quarantine(
    df: pd.DataFrame,
    *,
    quarantine_known_censored_targets: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove provenance-backed structure-label failures.

    Target-extreme rows are retained by default because their interpretation as
    censored values is not backed by an independent source flag. They can be
    excluded explicitly as a sensitivity analysis.
    """

    reasons = pd.Series(pd.NA, index=df.index, dtype="string")
    wrong_todga = df[EXTRACTANT_COLUMN].eq(TODGA_SMILES) & ~df["extractant_name"].eq("TODGA")
    reasons.loc[wrong_todga] = "todga_structure_assigned_to_different_extractant"

    if quarantine_known_censored_targets:
        censored = df["safe_exp_id"].astype(str).isin(KNOWN_CENSORED_SAFE_IDS)
        already_quarantined = censored & reasons.notna()
        newly_quarantined = censored & reasons.isna()
        reasons.loc[already_quarantined] = (
            reasons.loc[already_quarantined] + ";known_sentinel_like_target"
        )
        reasons.loc[newly_quarantined] = "known_sentinel_like_target"

    quarantined = df.loc[reasons.notna()].copy()
    quarantined.insert(0, "quarantine_reason", reasons.loc[reasons.notna()].astype(str))
    clean = df.loc[reasons.isna()].copy()
    return clean, quarantined


def _stable_scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def _stable_id(values: Iterable[Any], length: int = 20) -> str:
    payload = json.dumps([_stable_scalar(v) for v in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _select_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def _numeric_or_nan(value: Any) -> float:
    """Coerce one aggregated cell value to a float, mapping any gap to NaN."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _electronic_values(row: Any) -> dict[str, float]:
    """Read one complex's declared electronic quantities from a cell row."""

    values = {
        source.name: _numeric_or_nan(row[source.column])
        for source in COMPLEX_ELECTRONIC_SOURCES
    }
    values[DERIVED_COMPLEX_ELECTRONIC_NAME] = (
        values["q_metal"] - values["q_donor_mean"]
    )
    values[ENERGY_SOURCE.name] = _numeric_or_nan(row[ENERGY_SOURCE.column])
    return values


def _add_derived_invariant_3d_features(
    df: pd.DataFrame,
    *,
    include_donor_composition_counts: bool = False,
) -> pd.DataFrame:
    """Add compact permutation-invariant shell summaries from ranked inputs.

    Ranked arrays in the bundle can be zero padded.  Zero is not a physical
    Ln--donor distance or an angle between two distinct donor vectors, so padded
    entries are masked before pooling.  The resulting quantiles and Legendre
    moments are invariant to global rotation and donor permutation.
    """

    result = df.copy()
    angle_columns = [
        c for c in result.columns if str(c).startswith("feat3d__polyhedron__donor_angle_")
    ]
    if angle_columns:
        angles = result[angle_columns].apply(pd.to_numeric, errors="coerce")
        angles = angles.mask((angles <= 0.0) | (angles > 180.0))
        result[f"{DERIVED_3D_PREFIX}donor_angle_mean_deg"] = angles.mean(axis=1)
        result[f"{DERIVED_3D_PREFIX}donor_angle_std_deg"] = angles.std(axis=1, ddof=0)
        result[f"{DERIVED_3D_PREFIX}donor_angle_min_deg"] = angles.min(axis=1)
        result[f"{DERIVED_3D_PREFIX}donor_angle_max_deg"] = angles.max(axis=1)
        result[f"{DERIVED_3D_PREFIX}donor_angle_q25_deg"] = angles.quantile(0.25, axis=1)
        result[f"{DERIVED_3D_PREFIX}donor_angle_median_deg"] = angles.median(axis=1)
        result[f"{DERIVED_3D_PREFIX}donor_angle_q75_deg"] = angles.quantile(0.75, axis=1)

        cosines = angles.apply(lambda column: np.cos(np.deg2rad(column)))
        result[f"{DERIVED_3D_PREFIX}donor_angle_legendre_p1_mean"] = cosines.mean(axis=1)
        result[f"{DERIVED_3D_PREFIX}donor_angle_legendre_p2_mean"] = (
            0.5 * (3.0 * cosines.pow(2) - 1.0)
        ).mean(axis=1)
        result[f"{DERIVED_3D_PREFIX}donor_angle_legendre_p3_mean"] = (
            0.5 * (5.0 * cosines.pow(3) - 3.0 * cosines)
        ).mean(axis=1)

    ranked_distance_columns = [
        c
        for c in result.columns
        if str(c).startswith("feat3d__polyhedron__ln_donor_distance_")
    ]
    if ranked_distance_columns:
        ranked_distances = result[ranked_distance_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        ranked_distances = ranked_distances.mask(ranked_distances <= 0.0)
        result[f"{DERIVED_3D_PREFIX}donor_distance_q25"] = ranked_distances.quantile(
            0.25, axis=1
        )
        result[f"{DERIVED_3D_PREFIX}donor_distance_median"] = ranked_distances.median(
            axis=1
        )
        result[f"{DERIVED_3D_PREFIX}donor_distance_q75"] = ranked_distances.quantile(
            0.75, axis=1
        )

    if include_donor_composition_counts:
        donor_element_columns = [
            column
            for column in result.columns
            if str(column).startswith(
                "feat3d__polyhedron__donor_atomic_number_"
            )
        ]
        if not donor_element_columns:
            raise ValueError(
                "Donor-composition counts were requested but ranked donor atomic "
                "numbers are absent."
            )
        donor_atomic_numbers = result[donor_element_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        donor_atomic_numbers = donor_atomic_numbers.mask(donor_atomic_numbers <= 0.0)
        has_donor = donor_atomic_numbers.notna().any(axis=1)
        known_atomic_numbers = tuple(DONOR_ELEMENT_ATOMIC_NUMBERS.values())
        for symbol, atomic_number in DONOR_ELEMENT_ATOMIC_NUMBERS.items():
            counts = donor_atomic_numbers.eq(float(atomic_number)).sum(axis=1).astype(float)
            result[f"{DERIVED_3D_PREFIX}donor_count_{symbol}"] = counts.where(
                has_donor, np.nan
            )
        other_counts = (
            donor_atomic_numbers.notna()
            & ~donor_atomic_numbers.isin(known_atomic_numbers)
        ).sum(axis=1).astype(float)
        result[f"{DERIVED_3D_PREFIX}donor_count_other"] = other_counts.where(
            has_donor, np.nan
        )

    distance_mean = "feat3d__complex_physical__ln_donor_distance_mean"
    distance_std = "feat3d__complex_physical__ln_donor_distance_std"
    distance_min = "feat3d__complex_physical__ln_donor_distance_min"
    distance_max = "feat3d__complex_physical__ln_donor_distance_max"
    if all(c in result.columns for c in (distance_mean, distance_std, distance_min, distance_max)):
        result[f"{DERIVED_3D_PREFIX}shell_clearance_mean"] = (
            result[distance_mean] - result["Ionic Radius_metal"]
        )
        result[f"{DERIVED_3D_PREFIX}shell_distance_span"] = (
            result[distance_max] - result[distance_min]
        )
        denominator = pd.to_numeric(result[distance_mean], errors="coerce").replace(0.0, np.nan)
        result[f"{DERIVED_3D_PREFIX}shell_distance_cv"] = result[distance_std] / denominator

    return result


def _feature_contract(
    df: pd.DataFrame,
    *,
    delta3d_feature_set: str,
    geometry_descriptor_blocks: tuple[str, ...] = (),
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    condition_cols = _select_numeric_columns(
        df, [c for c in df.columns if str(c).startswith("cond__")]
    )
    molecular_cols = _select_numeric_columns(df, MOLECULAR_DESCRIPTOR_COLUMNS)
    ecfp_cols = _select_numeric_columns(
        df, [c for c in df.columns if str(c).startswith("ecfp_")]
    )
    candidate_3d = _select_numeric_columns(
        df, [c for c in df.columns if str(c).startswith("feat3d__")]
    )
    all_invariant_3d = [
        c for c in candidate_3d if not any(token in c for token in EXCLUDED_3D_TOKENS)
    ]
    # Descriptor blocks are opt-in.  When none are requested the contract is
    # bit-for-bit the frozen one, so earlier runs stay reproducible.
    descriptor_prefixes = tuple(
        BLOCK_PREFIXES[block] for block in geometry_descriptor_blocks
    )
    all_invariant_3d = [
        c
        for c in all_invariant_3d
        if not str(c).startswith(tuple(BLOCK_PREFIXES.values()))
        or str(c).startswith(descriptor_prefixes)
    ]
    if delta3d_feature_set == "compact-invariant":
        selected_3d = [
            c
            for c in all_invariant_3d
            if c in COMPACT_3D_COLUMNS
            or str(c).startswith(DERIVED_3D_PREFIX)
            or (descriptor_prefixes and str(c).startswith(descriptor_prefixes))
        ]
    elif delta3d_feature_set == "all-ranked":
        selected_3d = all_invariant_3d
    else:
        raise ValueError("delta3d_feature_set must be 'compact-invariant' or 'all-ranked'.")
    return condition_cols, molecular_cols, ecfp_cols, selected_3d, all_invariant_3d


def _validated_ecfp_cluster_ids(df: pd.DataFrame, ecfp_cols: list[str]) -> pd.Series:
    """Validate binary fingerprints and return one exact hash per source row."""

    if not ecfp_cols:
        return pd.Series("no_ecfp", index=df.index, dtype="string")

    numeric = df[ecfp_cols].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("ECFP block contains missing or non-finite values.")
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("ECFP block must be binary before exact-cluster construction.")

    packed = np.packbits(values.astype(np.uint8, copy=False), axis=1)
    hashes = pd.Series(
        [hashlib.sha256(row.tobytes()).hexdigest()[:20] for row in packed],
        index=df.index,
        dtype="string",
    )
    hashes_per_extractant = hashes.groupby(df[EXTRACTANT_COLUMN], dropna=False).nunique()
    inconsistent = hashes_per_extractant[hashes_per_extractant != 1]
    if not inconsistent.empty:
        examples = [str(value) for value in inconsistent.index[:5]]
        raise ValueError(
            "One canonical_smiles maps to multiple ECFP fingerprints; examples: "
            f"{examples}"
        )
    return hashes


def _pair_cohort_sha256(pairs: pd.DataFrame) -> str:
    """Hash exact pair membership, provenance identities, groups, and targets."""

    columns = (
        "pair_id",
        "condition_id",
        "extractant",
        "ecfp_exact_cluster",
        "pair_label",
        "source_id_A",
        "source_id_B",
        "geometry_key_A",
        "geometry_key_B",
        "geometry_feature_build_id_A",
        "geometry_feature_build_id_B",
        "vr_graph_index_A",
        "vr_graph_index_B",
        PAIR_TARGET_COLUMN,
    )
    ordered = pairs.loc[:, list(columns)].sort_values("pair_id", kind="stable")
    records = [
        [_stable_scalar(value) for value in row]
        for row in ordered.itertuples(index=False, name=None)
    ]
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_lanthanide_pair_dataset(
    source: pd.DataFrame,
    *,
    pair_scope: str = "all",
    require_geometry: bool = True,
    replicate_policy: str = "unique",
    quarantine_known_bad: bool = True,
    quarantine_known_censored_targets: bool = False,
    require_complete_conditions: bool = True,
    delta3d_feature_set: str = "compact-invariant",
    require_complete_3d: bool | None = None,
    geometry_descriptor_blocks: Iterable[str] = (),
    include_donor_composition_counts: bool = False,
    include_symmetric_3d: bool = False,
    include_pair_response_3d: bool = False,
    include_electronic: bool = False,
) -> PairDataset:
    """Create condition-matched lanthanide pairs for the requested scope.

    Parameters
    ----------
    source:
        Row-level extraction dataset.
    pair_scope:
        ``all`` creates every unordered observed metal pair within an exact
        extractant/condition cell. ``adjacent`` retains only atomic-number
        neighbours and exists to reproduce the historical benchmark.
    require_geometry:
        Require accepted 3D geometry for both metals.  Keeping this true gives a
        common cohort for the 2D-vs-3D ablation.
    replicate_policy:
        ``median`` aggregates repeated same-metal measurements inside their
        extractant/condition cell. ``unique`` keeps a pair only when each metal
        cell contains exactly one measurement.
    quarantine_known_bad:
        Apply the provenance-backed TODGA structure/name quarantine.
    quarantine_known_censored_targets:
        Opt-in sensitivity rule for three target-extreme rows. It is false by
        default because no independent censor/provenance flag exists.
    require_complete_conditions:
        Require every ``cond__`` value to be observed. Missing values cannot be
        treated as evidence that two experiments used identical conditions.
    delta3d_feature_set:
        ``compact-invariant`` uses shell distribution summaries and excludes
        discontinuous donor-rank identities. ``all-ranked`` is a sensitivity
        analysis that exposes every otherwise allowed tabular 3D column.
    require_complete_3d:
        Exclude pairs with a missing selected 3D contrast, preventing feature
        availability from becoming a provenance shortcut. By default this is
        true for ``compact-invariant`` and false for padded ``all-ranked`` data.
    geometry_descriptor_blocks:
        Metal-site descriptor blocks to admit into the Delta3D contract. The
        columns must already be attached to ``source`` by
        :func:`~lanthanide_separation.geometry_descriptors.attach_geometry_descriptors`.
        The empty default reproduces the frozen contract exactly.
    include_donor_composition_counts:
        Add permutation-invariant N/O/P/S/other donor counts derived from the
        ranked donor atomic-number fields. This is enabled by the geometry
        ablation runner and remains off for legacy frozen-contract reproduction.
    include_symmetric_3d:
        Additionally emit ``sym3d__X = (X_A + X_B) / 2`` for every selected 3D
        column. The delta contract cancels the absolute coordination
        environment; the symmetric partner restores it while staying invariant
        under the A/B swap, so exact prediction antisymmetry is unaffected.
        Off by default, which reproduces the frozen contract bit-for-bit.
    include_pair_response_3d:
        Emit the declared ``pair3d__`` block: relative, magnitude, ionic-radius
        normalised and excess forms of the metal-substitution response of the
        coordination shell. See :mod:`lanthanide_separation.pair_response`.
    include_electronic:
        Emit the declared ``elec__`` block of complex-level and pair-response
        xTB electronic quantities plus the composition-gated total-energy
        contrast. See :mod:`lanthanide_separation.electronic`. These columns are
        never added to A0--A6; they form their own arms.
    """

    _validate_source(source)
    if pair_scope not in PAIR_SCOPES:
        raise ValueError(f"pair_scope must be one of {PAIR_SCOPES}; got {pair_scope!r}.")
    if replicate_policy not in {"median", "unique"}:
        raise ValueError("replicate_policy must be 'median' or 'unique'.")
    descriptor_blocks = validate_blocks(geometry_descriptor_blocks)
    missing_blocks = [
        block
        for block in descriptor_blocks
        if not any(str(c).startswith(BLOCK_PREFIXES[block]) for c in source.columns)
    ]
    if missing_blocks:
        raise ValueError(
            "Requested geometry descriptor blocks are absent from the source frame: "
            f"{missing_blocks}"
        )
    if require_complete_3d is None:
        require_complete_3d = delta3d_feature_set == "compact-invariant"

    original_rows = int(len(source))
    if quarantine_known_bad:
        df, quarantine = apply_default_quarantine(
            source,
            quarantine_known_censored_targets=quarantine_known_censored_targets,
        )
    else:
        df = source.copy()
        quarantine = source.iloc[0:0].copy()
        quarantine.insert(0, "quarantine_reason", pd.Series(dtype="string"))

    df = df[df["metal"].isin(LANTHANIDE_Z)].copy()
    finite_target = np.isfinite(pd.to_numeric(df["log_D"], errors="coerce"))
    invalid_target_rows = int((~finite_target).sum())
    df = df.loc[finite_target].copy()
    df = _add_derived_invariant_3d_features(
        df,
        include_donor_composition_counts=include_donor_composition_counts,
    )

    (
        condition_cols,
        molecular_cols,
        ecfp_cols,
        invariant_3d_cols,
        all_invariant_3d_cols,
    ) = _feature_contract(
        df,
        delta3d_feature_set=delta3d_feature_set,
        geometry_descriptor_blocks=descriptor_blocks,
    )
    all_condition_cols = [c for c in df.columns if str(c).startswith("cond__")]
    nonnumeric_conditions = sorted(set(all_condition_cols) - set(condition_cols))
    if nonnumeric_conditions:
        raise ValueError(
            "All cond__ columns must be numeric for exact matching; nonnumeric columns: "
            f"{nonnumeric_conditions}"
        )
    incomplete_condition_rows = int(df[condition_cols].isna().any(axis=1).sum())
    if require_complete_conditions:
        df = df.loc[~df[condition_cols].isna().any(axis=1)].copy()
    df["_ecfp_exact_cluster"] = _validated_ecfp_cluster_ids(df, ecfp_cols)
    all_nan_3d_cols = [c for c in invariant_3d_cols if df[c].isna().all()]
    usable_3d_cols = [c for c in invariant_3d_cols if c not in all_nan_3d_cols]

    # canonical_smiles is the split identity.  The current bundle promises that
    # extractant_group is identical; record drift rather than silently trusting it.
    group_identity_mismatches = int(
        (~df["extractant_group"].astype(str).eq(df[EXTRACTANT_COLUMN].astype(str))).sum()
    )

    # Extension blocks read source columns directly rather than reusing the
    # frozen Delta3D selection, because some of them (the total energy) are
    # deliberately excluded from that selection.  A requested source that is
    # absent or entirely unobserved fails closed instead of emitting a column of
    # NaNs that would look like a computed descriptor.
    extension_source_columns: list[str] = []
    if include_pair_response_3d:
        extension_source_columns.extend(pair_response_source_columns())
    if include_electronic:
        extension_source_columns.extend(electronic_source_columns())
    extension_source_columns = list(dict.fromkeys(extension_source_columns))
    missing_extension_sources = [
        column
        for column in extension_source_columns
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column])
    ]
    if missing_extension_sources:
        raise ValueError(
            "Requested extension blocks need numeric source columns that are "
            f"absent from the dataset: {missing_extension_sources}"
        )
    unobserved_extension_sources = [
        column for column in extension_source_columns if df[column].isna().all()
    ]
    if unobserved_extension_sources:
        raise ValueError(
            "Requested extension blocks depend on source columns that are "
            f"entirely unobserved in this dataset: {unobserved_extension_sources}"
        )

    cell_key = [EXTRACTANT_COLUMN, *condition_cols, "metal"]
    numeric_to_aggregate = [
        "log_D",
        "Ionic Radius_metal",
        *molecular_cols,
        *ecfp_cols,
        *usable_3d_cols,
        *extension_source_columns,
    ]
    # Preserve order while removing accidental duplicates.
    numeric_to_aggregate = list(dict.fromkeys(numeric_to_aggregate))
    metadata_to_aggregate = [
        "geometry_key",
        "build_id",
        "safe_exp_id",
        "extractant_name",
        "geometry_feature_build_id",
        "vr_graph_index",
        "_ecfp_exact_cluster",
    ]
    optional_geometry_metadata = [
        column
        for column in ("geometry_status", "geometry_qc_class")
        if column in df.columns
    ]
    metadata_to_aggregate.extend(optional_geometry_metadata)
    # The total-energy contrast is only interpretable between complexes of the
    # same composition, so the composition fields have to travel with the cell.
    composition_columns: list[str] = []
    if include_electronic:
        missing_composition = [
            column
            for column in COMPOSITION_MATCH_COLUMNS
            if column not in df.columns
        ]
        if missing_composition:
            raise ValueError(
                "The electronic block needs the complex composition fields to "
                f"gate its energy contrast; missing: {missing_composition}"
            )
        composition_columns = list(COMPOSITION_MATCH_COLUMNS)
        metadata_to_aggregate.extend(composition_columns)

    selected = list(
        dict.fromkeys(cell_key + ["geometry_ok"] + numeric_to_aggregate + metadata_to_aggregate)
    )
    work = df[selected].copy()
    aggregate: dict[str, Any] = {c: "median" for c in numeric_to_aggregate}
    aggregate["geometry_ok"] = "all"
    aggregate.update({c: "first" for c in metadata_to_aggregate})

    grouped = work.groupby(cell_key, dropna=False, sort=False)
    cells = grouped.agg(aggregate).reset_index()
    # One bulk concat instead of seven single-column assignments.  Repeated
    # assignment on a >2000-column frame triggers pandas block fragmentation and
    # copies the whole frame each time; the values below are identical.
    cell_summaries = pd.DataFrame(
        {
            "n_replicates": grouped.size().to_numpy(),
            "target_std": grouped["log_D"].std().to_numpy(),
            "target_min": grouped["log_D"].min().to_numpy(),
            "target_max": grouped["log_D"].max().to_numpy(),
            "n_geometry_keys": grouped["geometry_key"]
            .nunique(dropna=False)
            .to_numpy(),
            "Z": cells["metal"].map(LANTHANIDE_Z).astype(int).to_numpy(),
        },
        index=cells.index,
    )
    cells = pd.concat([cells, cell_summaries], axis=1)
    cells_with_mixed_composition = 0
    if composition_columns:
        composition_uniques = grouped[composition_columns].nunique(dropna=False)
        cells_with_mixed_composition = int(
            (composition_uniques > 1).any(axis=1).sum()
        )

    pair_rows: list[dict[str, Any]] = []
    candidate_pairs_before_filters = 0
    true_adjacent_before_filters = 0
    excluded_missing_geometry = 0
    excluded_replicates = 0
    composition_matched_pairs = 0
    condition_group_key = [EXTRACTANT_COLUMN, *condition_cols]

    for condition_values, group in cells.groupby(condition_group_key, dropna=False, sort=False):
        condition_values_tuple = (
            condition_values if isinstance(condition_values, tuple) else (condition_values,)
        )
        by_z = {int(row["Z"]): row for _, row in group.iterrows()}
        z_values = sorted(by_z)
        if pair_scope == "all":
            candidate_z_pairs = combinations(z_values, 2)
        else:
            candidate_z_pairs = (
                (z_a, z_a + 1) for z_a in z_values if z_a + 1 in by_z
            )
        for z_a, z_b in candidate_z_pairs:
            row_a = by_z[z_a]
            row_b = by_z[z_b]
            candidate_pairs_before_filters += 1
            true_adjacent_before_filters += int(z_b - z_a == 1)

            if require_geometry and not (bool(row_a["geometry_ok"]) and bool(row_b["geometry_ok"])):
                excluded_missing_geometry += 1
                continue
            if replicate_policy == "unique" and not (
                int(row_a["n_replicates"]) == 1 and int(row_b["n_replicates"]) == 1
            ):
                excluded_replicates += 1
                continue

            metal_a = str(row_a["metal"])
            metal_b = str(row_b["metal"])
            if str(row_a["_ecfp_exact_cluster"]) != str(row_b["_ecfp_exact_cluster"]):
                raise AssertionError("Paired metals have different ligand ECFP fingerprints.")
            condition_id = _stable_id(condition_values_tuple)
            pair_id = _stable_id((condition_id, metal_a, metal_b))
            radius_a = float(row_a["Ionic Radius_metal"])
            radius_b = float(row_b["Ionic Radius_metal"])

            record: dict[str, Any] = {
                "pair_id": pair_id,
                "condition_id": condition_id,
                "extractant": str(row_a[EXTRACTANT_COLUMN]),
                "extractant_name_A": str(row_a["extractant_name"]),
                "ecfp_exact_cluster": str(row_a["_ecfp_exact_cluster"]),
                "metal_A": metal_a,
                "metal_B": metal_b,
                "pair_label": f"{metal_a}-{metal_b}",
                "geometry_ok_A": bool(row_a["geometry_ok"]),
                "geometry_ok_B": bool(row_b["geometry_ok"]),
                "geometry_key_A": str(row_a["geometry_key"]),
                "geometry_key_B": str(row_b["geometry_key"]),
                "build_id_A": str(row_a["build_id"]),
                "build_id_B": str(row_b["build_id"]),
                "geometry_feature_build_id_A": str(row_a["geometry_feature_build_id"]),
                "geometry_feature_build_id_B": str(row_b["geometry_feature_build_id"]),
                "vr_graph_index_A": int(row_a["vr_graph_index"]),
                "vr_graph_index_B": int(row_b["vr_graph_index"]),
                "source_id_A": str(row_a["safe_exp_id"]),
                "source_id_B": str(row_b["safe_exp_id"]),
                "n_replicates_A": int(row_a["n_replicates"]),
                "n_replicates_B": int(row_b["n_replicates"]),
                "target_std_A": float(row_a["target_std"])
                if pd.notna(row_a["target_std"])
                else np.nan,
                "target_std_B": float(row_b["target_std"])
                if pd.notna(row_b["target_std"])
                else np.nan,
                "log_D_A": float(row_a["log_D"]),
                "log_D_B": float(row_b["log_D"]),
                PAIR_TARGET_COLUMN: float(row_a["log_D"] - row_b["log_D"]),
                "pair__Z_A": float(z_a),
                "pair__Z_B": float(z_b),
                "pair__Z_mean": float((z_a + z_b) / 2.0),
                "pair__delta_Z": float(z_a - z_b),
                "pair__ionic_radius_A": radius_a,
                "pair__ionic_radius_B": radius_b,
                "pair__ionic_radius_mean": (radius_a + radius_b) / 2.0,
                "pair__delta_ionic_radius": radius_a - radius_b,
            }
            for column in optional_geometry_metadata:
                value_a = row_a[column]
                value_b = row_b[column]
                record[f"{column}_A"] = None if pd.isna(value_a) else str(value_a)
                record[f"{column}_B"] = None if pd.isna(value_b) else str(value_b)
            record["SF_A_over_B"] = float(10.0 ** record[PAIR_TARGET_COLUMN])

            for column in condition_cols:
                record[f"base__{column}"] = row_a[column]
            for column in [*molecular_cols, *ecfp_cols]:
                record[f"base__{column}"] = row_a[column]
            for column in usable_3d_cols:
                value_a = row_a[column]
                value_b = row_b[column]
                both_observed = pd.notna(value_a) and pd.notna(value_b)
                record[f"delta3d__{column}"] = (
                    float(value_a - value_b) if both_observed else np.nan
                )
                if include_symmetric_3d:
                    # Swap-symmetric partner of the delta contrast.  It carries
                    # the absolute coordination environment that the difference
                    # cancels, without breaking A/B antisymmetry.
                    record[f"sym3d__{column}"] = (
                        float((value_a + value_b) / 2.0) if both_observed else np.nan
                    )

            radius_difference = radius_a - radius_b
            if include_pair_response_3d:
                record.update(
                    pair_response_features(
                        values_a={
                            quantity.name: _numeric_or_nan(row_a[quantity.column])
                            for quantity in PAIR_RESPONSE_QUANTITIES
                        },
                        values_b={
                            quantity.name: _numeric_or_nan(row_b[quantity.column])
                            for quantity in PAIR_RESPONSE_QUANTITIES
                        },
                        radius_difference=radius_difference,
                    )
                )
            if include_electronic:
                composition_matches = all(
                    str(row_a[column]) == str(row_b[column])
                    for column in composition_columns
                )
                composition_matched_pairs += int(composition_matches)
                record.update(
                    pair_electronic_features(
                        values_a=_electronic_values(row_a),
                        values_b=_electronic_values(row_b),
                        radius_difference=radius_difference,
                        composition_matches=composition_matches,
                    )
                )
            pair_rows.append(record)

    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        if pair_scope == "adjacent":
            raise ValueError("No true-adjacent, condition-matched pairs survived the filters.")
        raise ValueError("No condition-matched lanthanide pairs survived the filters.")

    baseline_columns = tuple(
        [
            "pair__Z_A",
            "pair__Z_B",
            "pair__Z_mean",
            "pair__delta_Z",
            "pair__ionic_radius_A",
            "pair__ionic_radius_B",
            "pair__ionic_radius_mean",
            "pair__delta_ionic_radius",
        ]
        + [f"base__{c}" for c in [*condition_cols, *molecular_cols, *ecfp_cols]]
    )
    descriptor_prefixes = tuple(BLOCK_PREFIXES[block] for block in descriptor_blocks)
    all_delta3d_columns = tuple(f"delta3d__{c}" for c in usable_3d_cols)
    delta3d_columns = tuple(
        c
        for c in all_delta3d_columns
        if not str(c).startswith(tuple(f"delta3d__{p}" for p in BLOCK_PREFIXES.values()))
    )
    descriptor_columns = tuple(
        c
        for c in all_delta3d_columns
        if descriptor_prefixes
        and str(c).startswith(tuple(f"delta3d__{p}" for p in descriptor_prefixes))
    )
    if len(delta3d_columns) + len(descriptor_columns) != len(all_delta3d_columns):
        raise AssertionError("Delta3D and descriptor column partition is not exhaustive.")
    symmetric3d_columns = (
        tuple(f"sym3d__{c}" for c in usable_3d_cols) if include_symmetric_3d else ()
    )
    missing_symmetric = [c for c in symmetric3d_columns if c not in pairs.columns]
    if missing_symmetric:
        raise AssertionError(
            f"Symmetric 3D columns were requested but not emitted: {missing_symmetric[:5]}"
        )
    pair_response_columns = (
        tuple(c for c in pairs.columns if str(c).startswith(PAIR_RESPONSE_PREFIX))
        if include_pair_response_3d
        else ()
    )
    electronic_columns = (
        tuple(c for c in pairs.columns if str(c).startswith(ELECTRONIC_PREFIX))
        if include_electronic
        else ()
    )
    # Parity must be declared, never inferred.  A column under an extension
    # namespace that announces neither parity would be silently left unchanged
    # by the A/B swap and would break antisymmetry, so it fails closed here.
    undeclared_parity = [
        column
        for column in pair_response_columns + electronic_columns
        if not str(column).startswith(
            (
                PAIR_RESPONSE_ODD_PREFIX,
                PAIR_RESPONSE_EVEN_PREFIX,
                ELECTRONIC_ODD_PREFIX,
                ELECTRONIC_EVEN_PREFIX,
            )
        )
    ]
    if undeclared_parity:
        raise AssertionError(
            "Extension columns must declare swap parity via their namespace: "
            f"{undeclared_parity[:5]}"
        )
    missing_selected_3d = pairs.loc[:, list(all_delta3d_columns)].isna().any(axis=1)
    pairs_excluded_missing_3d_features = int(missing_selected_3d.sum())
    if require_complete_3d:
        pairs = pairs.loc[~missing_selected_3d].reset_index(drop=True)
        if pairs.empty:
            raise ValueError("No pairs have a complete selected 3D feature block.")

    # Fail loudly if an identifier or target ever enters the model contract.
    forbidden_fragments = (
        "log_D",
        "log_SF",
        "safe_exp_id",
        "build_id",
        "geometry_key",
        "sample_weight",
        "asset_index",
        "xyz_path",
    )
    leaked = [
        c
        for c in baseline_columns
        + all_delta3d_columns
        + symmetric3d_columns
        + pair_response_columns
        + electronic_columns
        if any(fragment.lower() in c.lower() for fragment in forbidden_fragments)
    ]
    if leaked:
        raise AssertionError(f"Forbidden target/identifier features entered the model: {leaked}")

    cohort_sha256 = _pair_cohort_sha256(pairs)
    atomic_number_spans = (
        pairs["pair__Z_B"].to_numpy(dtype=int)
        - pairs["pair__Z_A"].to_numpy(dtype=int)
    )
    span_counts = pd.Series(atomic_number_spans).value_counts().sort_index()
    source_usage = pd.concat(
        [pairs["source_id_A"], pairs["source_id_B"]], ignore_index=True
    ).value_counts()
    source_geometry_ok = df["geometry_ok"].fillna(False).astype(bool)
    source_geometry_qc_counts = (
        df["geometry_qc_class"].astype("string").fillna("UNAVAILABLE_OR_UNCLASSIFIED")
        .value_counts()
        .sort_index()
        .to_dict()
        if "geometry_qc_class" in df.columns
        else {"UNAVAILABLE_OR_UNCLASSIFIED": int((~source_geometry_ok).sum())}
    )

    audit = {
        "source_rows": original_rows,
        "rows_after_quarantine_and_target_validation": int(len(df)),
        "quarantined_rows": int(len(quarantine)),
        "quarantine_reasons": quarantine["quarantine_reason"].value_counts().to_dict(),
        "invalid_nonfinite_target_rows": invalid_target_rows,
        "known_target_extreme_rows_retained": int(
            df["safe_exp_id"].astype(str).isin(KNOWN_CENSORED_SAFE_IDS).sum()
        ),
        "source_geometry_ok_rows": int(source_geometry_ok.sum()),
        "source_geometry_unavailable_or_rejected_rows": int((~source_geometry_ok).sum()),
        "source_geometry_qc_class_counts": {
            str(label): int(count) for label, count in source_geometry_qc_counts.items()
        },
        "incomplete_condition_rows_seen": incomplete_condition_rows,
        "require_complete_conditions": bool(require_complete_conditions),
        "extractant_group_identity_mismatches": group_identity_mismatches,
        "condition_columns": condition_cols,
        "condition_column_count": len(condition_cols),
        "molecular_descriptor_count": len(molecular_cols),
        "ecfp_feature_count": len(ecfp_cols),
        "candidate_invariant_3d_count": len(invariant_3d_cols),
        "available_invariant_3d_count": len(all_invariant_3d_cols),
        "delta3d_feature_set": delta3d_feature_set,
        "include_donor_composition_counts": bool(
            include_donor_composition_counts
        ),
        "excluded_by_delta3d_feature_set": [
            c for c in all_invariant_3d_cols if c not in invariant_3d_cols
        ],
        "usable_invariant_3d_count": len(usable_3d_cols),
        "geometry_descriptor_blocks": list(descriptor_blocks),
        "geometry_descriptor_column_count": len(descriptor_columns),
        "include_symmetric_3d": bool(include_symmetric_3d),
        "symmetric_3d_column_count": len(symmetric3d_columns),
        "include_pair_response_3d": bool(include_pair_response_3d),
        "pair_response_column_count": len(pair_response_columns),
        "include_electronic": bool(include_electronic),
        "electronic_column_count": len(electronic_columns),
        "cells_with_mixed_complex_composition": int(cells_with_mixed_composition),
        "pairs_with_matched_complex_composition": (
            int(composition_matched_pairs) if include_electronic else None
        ),
        "frozen_delta3d_column_count": len(delta3d_columns),
        "all_nan_3d_columns": all_nan_3d_cols,
        "excluded_noninvariant_or_noncomparable_3d_columns": [
            c
            for c in df.columns
            if str(c).startswith("feat3d__")
            and any(token in str(c) for token in EXCLUDED_3D_TOKENS)
        ],
        "same_metal_condition_cells": int(len(cells)),
        "replicated_cells": int((cells["n_replicates"] > 1).sum()),
        "replicated_cells_with_target_conflict": int(
            ((cells["n_replicates"] > 1) & (cells["target_max"] > cells["target_min"])).sum()
        ),
        "maximum_within_cell_target_range": float(
            (cells["target_max"] - cells["target_min"]).max()
        ),
        "cells_with_multiple_geometry_keys": int((cells["n_geometry_keys"] > 1).sum()),
        "pair_scope": pair_scope,
        "candidate_pairs_before_pair_filters": candidate_pairs_before_filters,
        "true_adjacent_pairs_before_pair_filters": true_adjacent_before_filters,
        "pairs_excluded_missing_geometry": excluded_missing_geometry,
        "pairs_excluded_by_replicate_policy": excluded_replicates,
        "pairs_excluded_missing_selected_3d": (
            pairs_excluded_missing_3d_features if require_complete_3d else 0
        ),
        "pairs_with_missing_selected_3d_retained": (
            0 if require_complete_3d else pairs_excluded_missing_3d_features
        ),
        "require_complete_3d": bool(require_complete_3d),
        "pair_rows": int(len(pairs)),
        "cohort_sha256": cohort_sha256,
        "pair_extractants": int(pairs["extractant"].nunique()),
        "pair_ecfp_exact_clusters": int(pairs["ecfp_exact_cluster"].nunique()),
        "pairs_by_type": pairs["pair_label"].value_counts().sort_index().to_dict(),
        "pairs_by_atomic_number_span": {
            str(int(span)): int(count) for span, count in span_counts.items()
        },
        "minimum_atomic_number_span": int(atomic_number_spans.min()),
        "maximum_atomic_number_span": int(atomic_number_spans.max()),
        "nonadjacent_pair_rows": int((atomic_number_spans > 1).sum()),
        "unique_source_rows_in_pairs": int(len(source_usage)),
        "pair_side_usages": int(source_usage.sum()),
        "maximum_pairs_per_source_row": int(source_usage.max()),
        "replicate_policy": replicate_policy,
        "require_geometry": bool(require_geometry),
        "target_definition": (
            "log_D_A - log_D_B; A has lower atomic number; "
            "delta_Z = Z_A - Z_B < 0"
        ),
    }
    return PairDataset(
        frame=pairs,
        baseline_columns=baseline_columns,
        delta3d_columns=delta3d_columns,
        audit=audit,
        quarantine=quarantine,
        descriptor_columns=descriptor_columns,
        symmetric3d_columns=symmetric3d_columns,
        pair_response_columns=pair_response_columns,
        electronic_columns=electronic_columns,
    )


def build_adjacent_pair_dataset(
    source: pd.DataFrame,
    *,
    require_geometry: bool = True,
    replicate_policy: str = "unique",
    quarantine_known_bad: bool = True,
    quarantine_known_censored_targets: bool = False,
    require_complete_conditions: bool = True,
    delta3d_feature_set: str = "compact-invariant",
    require_complete_3d: bool | None = None,
    geometry_descriptor_blocks: Iterable[str] = (),
) -> PairDataset:
    """Build the legacy true-adjacent cohort for reproducibility."""

    return build_lanthanide_pair_dataset(
        source,
        pair_scope="adjacent",
        require_geometry=require_geometry,
        replicate_policy=replicate_policy,
        quarantine_known_bad=quarantine_known_bad,
        quarantine_known_censored_targets=quarantine_known_censored_targets,
        require_complete_conditions=require_complete_conditions,
        delta3d_feature_set=delta3d_feature_set,
        require_complete_3d=require_complete_3d,
        geometry_descriptor_blocks=geometry_descriptor_blocks,
    )


ODD_FEATURE_PREFIXES: tuple[str, ...] = (
    "delta3d__",
    PAIR_RESPONSE_ODD_PREFIX,
    ELECTRONIC_ODD_PREFIX,
)
EVEN_FEATURE_PREFIXES: tuple[str, ...] = (
    "sym3d__",
    PAIR_RESPONSE_EVEN_PREFIX,
    ELECTRONIC_EVEN_PREFIX,
)
PARITY_DECLARING_NAMESPACES: tuple[str, ...] = (
    PAIR_RESPONSE_PREFIX,
    ELECTRONIC_PREFIX,
)


def reverse_pair_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the B/A representation used for swap augmentation and inference.

    Columns are handled by declared parity, never by guesswork:

    * ``pair__{Z,ionic_radius}_{A,B}`` are exchanged;
    * every prefix in :data:`ODD_FEATURE_PREFIXES` is negated;
    * every prefix in :data:`EVEN_FEATURE_PREFIXES` is left untouched, which is
      what makes a symmetric feature legal in an antisymmetric model.

    Any column inside an extension namespace that declares neither parity raises
    rather than being silently treated as even, because that would break the
    exact ``f(A,B) = -f(B,A)`` guarantee.
    """

    undeclared = [
        column
        for column in frame.columns
        if str(column).startswith(PARITY_DECLARING_NAMESPACES)
        and not str(column).startswith(ODD_FEATURE_PREFIXES + EVEN_FEATURE_PREFIXES)
    ]
    if undeclared:
        raise ValueError(
            "Extension feature(s) declare no swap parity and cannot be reversed "
            f"safely: {sorted(undeclared)[:5]}"
        )

    reversed_frame = frame.copy()
    swap_pairs = (
        ("pair__Z_A", "pair__Z_B"),
        ("pair__ionic_radius_A", "pair__ionic_radius_B"),
    )
    for column_a, column_b in swap_pairs:
        if column_a in reversed_frame.columns and column_b in reversed_frame.columns:
            reversed_frame[[column_a, column_b]] = reversed_frame[
                [column_b, column_a]
            ].to_numpy()
    for column in ("pair__delta_Z", "pair__delta_ionic_radius"):
        if column in reversed_frame.columns:
            reversed_frame[column] = -reversed_frame[column]
    odd_columns = [
        c for c in reversed_frame.columns if str(c).startswith(ODD_FEATURE_PREFIXES)
    ]
    if odd_columns:
        reversed_frame[odd_columns] = -reversed_frame[odd_columns]
    return reversed_frame

"""Rotation-invariant metal-site descriptors from the bundled VR coordinates.

The tabular Delta3D contract summarises the first coordination shell with
unweighted distance and pair-angle moments.  Two physically distinct quantities
are missing from that contract and are recovered here without introducing any
new external asset:

``ligand_field``
    Radially weighted rotational invariants of the donor set.  For a point
    charge model the crystal-field coefficients obey

        sum_q |B_kq|^2  proportional to  sum_ij w_i w_j P_k(cos theta_ij)

    with ``w_i = r_i^-(k+1)``.  The spherical-harmonic addition theorem turns the
    invariant into a double sum over donor--donor angles, so no frame, no
    harmonic basis and no donor ordering ever enters the calculation.  The
    unweighted ``k <= 3`` shape terms are close to the pair-angle Legendre means
    that the existing contract already carries; the radially weighted terms are
    new because they couple shell contraction to shell geometry, which is the
    mechanism that distinguishes two adjacent lanthanides.

``enclosure``
    Solid-angle shielding of the metal by every atom of the complex.  The
    existing contract sees only the 8--9 donor atoms; the ligand body that
    controls how rigidly the cavity resists the lanthanide contraction is
    otherwise unused.  Rays are cast from the metal over a deterministic
    Fibonacci sphere and intersected with van der Waals spheres, which is an
    integral quantity and therefore tolerant of the coordinate noise left by
    heterogeneously relaxed source geometries.

Both blocks are invariant to translation, rotation, reflection and donor
relabelling, so the paired ``value_A - value_B`` contrast keeps the exact
antisymmetry the pair model relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DESCRIPTOR_BLOCKS: tuple[str, ...] = ("ligand_field", "enclosure")

LIGAND_FIELD_PREFIX = "feat3d__ligand_field__"
ENCLOSURE_PREFIX = "feat3d__enclosure__"

BLOCK_PREFIXES: dict[str, str] = {
    "ligand_field": LIGAND_FIELD_PREFIX,
    "enclosure": ENCLOSURE_PREFIX,
}

MAX_LEGENDRE_DEGREE = 6
ENCLOSURE_RADII: tuple[float, ...] = (3.5, 5.0, 7.0)
RAY_COUNT = 1024

# Bondi van der Waals radii for the elements present in this bundle.  The
# fallback is only reachable if a future asset introduces a new element; it is
# deliberately a neutral carbon-like value rather than a silent zero.
VAN_DER_WAALS_RADII: dict[int, float] = {
    1: 1.20,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    15: 1.80,
    16: 1.80,
    17: 1.75,
}
FALLBACK_VAN_DER_WAALS_RADIUS = 1.70

MINIMUM_DONORS = 3
MINIMUM_DONOR_DISTANCE_ANGSTROM = 0.5


DESCRIPTOR_PERMUTATIONS: tuple[str, ...] = ("none", "free", "metal-preserving")

DESCRIPTOR_PROFILES: tuple[str, ...] = ("core", "full")

# The ``core`` profile is declared from theory, not from held-out performance.
# Only even ranks k = 2, 4, 6 contribute to the f-electron crystal field in the
# usual point-charge treatment, and those three ranks are what the standard
# crystal-field strength parameter is built from; the odd ranks are retained in
# ``full`` as a geometric sensitivity rather than as a physical claim.  The
# enclosure entries keep the two shell radii that bracket the first and second
# coordination spheres plus the closest-contact and channel-anisotropy scalars.
# Keeping the block narrow matters: adding continuous columns of any kind
# perturbs how ExtraTrees samples split candidates, so a wide block buys itself
# an advantage that has nothing to do with chemistry.
CORE_DESCRIPTOR_NAMES: dict[str, tuple[str, ...]] = {
    "ligand_field": (
        "field_strength_k2",
        "field_strength_k4",
        "field_strength_k6",
        "field_shape_k2",
        "field_shape_k4",
        "field_shape_k6",
        "effective_field_radius",
    ),
    "enclosure": (
        "buried_fraction_r3p5",
        "buried_fraction_r5p0",
        "contact_distance_min",
        "open_anisotropy",
    ),
}


@dataclass(frozen=True)
class MetalSiteDescriptors:
    """Descriptor table plus the contract needed to audit it downstream."""

    frame: pd.DataFrame
    blocks: tuple[str, ...]
    columns: tuple[str, ...]
    audit: dict[str, object]
    metal_atomic_numbers: np.ndarray


def validate_blocks(blocks: Iterable[str]) -> tuple[str, ...]:
    """Normalise a requested block selection, rejecting unknown names."""

    requested = tuple(str(block).strip() for block in blocks if str(block).strip())
    unknown = sorted(set(requested) - set(DESCRIPTOR_BLOCKS))
    if unknown:
        raise ValueError(
            f"Unknown geometry descriptor blocks: {unknown}; "
            f"available blocks are {list(DESCRIPTOR_BLOCKS)}."
        )
    if len(set(requested)) != len(requested):
        raise ValueError("Geometry descriptor blocks must be unique.")
    # Canonical order keeps the emitted column order independent of CLI order.
    return tuple(block for block in DESCRIPTOR_BLOCKS if block in set(requested))


def fibonacci_directions(count: int) -> np.ndarray:
    """Deterministic, nearly uniform unit vectors used for solid-angle sampling."""

    if int(count) < 8:
        raise ValueError("count must be at least 8.")
    index = np.arange(int(count), dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / float(count)
    radius = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    golden_angle = np.pi * (1.0 + 5.0**0.5)
    angle = golden_angle * index
    directions = np.column_stack(
        (radius * np.cos(angle), radius * np.sin(angle), z)
    )
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def _legendre_stack(cosines: np.ndarray, max_degree: int) -> list[np.ndarray]:
    """Legendre polynomials P_0..P_max evaluated by the standard recurrence."""

    stack = [np.ones_like(cosines), cosines.astype(np.float64, copy=True)]
    for degree in range(1, max_degree):
        stack.append(
            ((2 * degree + 1) * cosines * stack[degree] - degree * stack[degree - 1])
            / float(degree + 1)
        )
    return stack


def ligand_field_descriptors(donor_vectors: np.ndarray) -> dict[str, float]:
    """Radially weighted rotational invariants of one donor shell."""

    if donor_vectors.ndim != 2 or donor_vectors.shape[1] != 3:
        raise ValueError("donor_vectors must have shape [n_donors, 3].")
    if len(donor_vectors) < MINIMUM_DONORS:
        raise ValueError(f"At least {MINIMUM_DONORS} donors are required.")
    radii = np.linalg.norm(donor_vectors, axis=1)
    if not np.isfinite(radii).all() or radii.min() < MINIMUM_DONOR_DISTANCE_ANGSTROM:
        raise ValueError("Donor radii must be finite and physically separated from the metal.")

    units = donor_vectors / radii[:, None]
    cosines = np.clip(units @ units.transpose(), -1.0, 1.0)
    polynomials = _legendre_stack(cosines, MAX_LEGENDRE_DEGREE)

    features: dict[str, float] = {}
    for degree in range(1, MAX_LEGENDRE_DEGREE + 1):
        weights = radii ** (-(degree + 1))
        outer = np.outer(weights, weights)
        # sum_q |B_kq|^2 up to the k-independent 4*pi/(2k+1) factor; the sum is a
        # Gram quadratic form and is nonnegative up to rounding.
        quadratic = float((outer * polynomials[degree]).sum())
        strength = float(np.sqrt(max(quadratic, 0.0)))
        total_weight = float(weights.sum())
        features[f"field_strength_k{degree}"] = strength
        features[f"field_shape_k{degree}"] = strength / max(total_weight, 1e-30)

    for order in (1, 2, 3, 4, 5, 6):
        features[f"radial_moment_inverse_r{order}"] = float(
            np.mean(radii ** (-order))
        )
    features["effective_field_radius"] = float(
        np.mean(radii ** (-3.0)) ** (-1.0 / 3.0)
    )
    return features


def enclosure_descriptors(
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    metal_index: int,
    directions: np.ndarray,
) -> dict[str, float]:
    """Solid-angle shielding of the metal by the whole complex."""

    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape [n_atoms, 3].")
    if len(coordinates) != len(atomic_numbers):
        raise ValueError("coordinates and atomic_numbers must have equal length.")

    keep = np.ones(len(coordinates), dtype=bool)
    keep[int(metal_index)] = False
    relative = coordinates[keep] - coordinates[int(metal_index)]
    if not len(relative):
        raise ValueError("The complex contains no non-metal atoms.")
    radii = np.array(
        [
            VAN_DER_WAALS_RADII.get(int(number), FALLBACK_VAN_DER_WAALS_RADIUS)
            for number in atomic_numbers[keep]
        ],
        dtype=np.float64,
    )
    distance = np.linalg.norm(relative, axis=1)

    # Ray/sphere intersection: for the ray t*d the perpendicular offset of atom a
    # is |a|^2 - (a.d)^2, and the first contact is a.d - sqrt(r^2 - offset).
    projection = directions @ relative.transpose()
    perpendicular_squared = distance[None, :] ** 2 - projection**2
    hit = (projection > 0.0) & (perpendicular_squared <= radii[None, :] ** 2)
    entry = np.where(
        hit,
        projection - np.sqrt(np.clip(radii[None, :] ** 2 - perpendicular_squared, 0.0, None)),
        np.inf,
    )
    first_contact = entry.min(axis=1)
    blocked = np.isfinite(first_contact)

    features: dict[str, float] = {}
    for cutoff in ENCLOSURE_RADII:
        label = f"{cutoff:.1f}".replace(".", "p")
        features[f"buried_fraction_r{label}"] = float((first_contact <= cutoff).mean())
    features["open_fraction"] = float((~blocked).mean())
    if blocked.any():
        contacts = first_contact[blocked]
        features["contact_distance_mean"] = float(contacts.mean())
        features["contact_distance_min"] = float(contacts.min())
        features["contact_distance_p10"] = float(np.quantile(contacts, 0.10))
        features["contact_distance_p90"] = float(np.quantile(contacts, 0.90))
    else:
        for name in (
            "contact_distance_mean",
            "contact_distance_min",
            "contact_distance_p10",
            "contact_distance_p90",
        ):
            features[name] = 0.0

    # Anisotropy of the open solid angle.  The mean open direction is frame
    # dependent, but its length is a rotational invariant: zero for an isotropic
    # opening and one for a single narrow channel.
    open_directions = directions[~blocked]
    if len(open_directions):
        features["open_anisotropy"] = float(
            np.linalg.norm(open_directions.mean(axis=0))
        )
    else:
        features["open_anisotropy"] = 0.0
    return features


class MetalSiteDescriptorBuilder:
    """Compute invariant metal-site descriptors for every geometry in the asset."""

    def __init__(
        self,
        path: Path | str,
        *,
        blocks: Sequence[str] = DESCRIPTOR_BLOCKS,
        profile: str = "core",
        ray_count: int = RAY_COUNT,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.blocks = validate_blocks(blocks)
        if not self.blocks:
            raise ValueError("At least one geometry descriptor block must be requested.")
        if profile not in DESCRIPTOR_PROFILES:
            raise ValueError(
                f"profile must be one of {list(DESCRIPTOR_PROFILES)}; received {profile!r}."
            )
        self.profile = str(profile)
        self.ray_count = int(ray_count)
        self.directions = fibonacci_directions(self.ray_count)

        with np.load(self.path, allow_pickle=False) as archive:
            required = {
                "coordinates",
                "atomic_numbers",
                "is_metal",
                "is_coord_donor",
                "node_ptr",
                "build_ids",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"VR archive is missing arrays: {missing}")
            self.coordinates = np.asarray(archive["coordinates"], dtype=np.float64)
            self.atomic_numbers = np.asarray(archive["atomic_numbers"], dtype=np.int64)
            self.is_metal = np.asarray(archive["is_metal"], dtype=np.int8)
            self.is_coord_donor = np.asarray(archive["is_coord_donor"], dtype=np.int8)
            self.node_ptr = np.asarray(archive["node_ptr"], dtype=np.int64)
            self.build_ids = np.asarray(archive["build_ids"]).astype(str)

        self._validate_archive()

    def _validate_archive(self) -> None:
        n_graphs = len(self.build_ids)
        if len(set(self.build_ids.tolist())) != n_graphs:
            raise ValueError("VR build_ids must be unique.")
        if self.node_ptr.shape != (n_graphs + 1,):
            raise ValueError("node_ptr must have length n_graphs + 1.")
        n_nodes = len(self.atomic_numbers)
        if (
            self.node_ptr[0] != 0
            or self.node_ptr[-1] != n_nodes
            or np.any(np.diff(self.node_ptr) < 0)
        ):
            raise ValueError("node_ptr is not a valid monotone pointer array.")
        if self.coordinates.shape != (n_nodes, 3):
            raise ValueError("coordinates must have shape [n_nodes, 3].")
        if not np.isfinite(self.coordinates).all():
            raise ValueError("coordinates contain non-finite values.")
        if np.any(self.atomic_numbers < 1) or np.any(self.atomic_numbers > 118):
            raise ValueError("atomic_numbers must be in [1, 118].")
        for name, values in (
            ("is_metal", self.is_metal),
            ("is_coord_donor", self.is_coord_donor),
        ):
            if len(values) != n_nodes or not np.isin(values, (0, 1)).all():
                raise ValueError(f"{name} must be a binary array over all nodes.")

    def asset_sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _selected(self, block: str, features: dict[str, float]) -> dict[str, float]:
        if self.profile == "full":
            return features
        wanted = CORE_DESCRIPTOR_NAMES[block]
        missing = sorted(set(wanted) - set(features))
        if missing:
            raise AssertionError(
                f"Core profile for {block} names unknown descriptors: {missing}"
            )
        return {name: features[name] for name in wanted}

    def _graph_slice(self, graph_index: int) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        start = int(self.node_ptr[graph_index])
        stop = int(self.node_ptr[graph_index + 1])
        coordinates = self.coordinates[start:stop]
        numbers = self.atomic_numbers[start:stop]
        metal_positions = np.flatnonzero(self.is_metal[start:stop] == 1)
        if len(metal_positions) != 1:
            raise ValueError(
                f"{self.build_ids[graph_index]}: expected exactly one metal node, "
                f"found {len(metal_positions)}"
            )
        donor_mask = self.is_coord_donor[start:stop] == 1
        return coordinates, numbers, int(metal_positions[0]), donor_mask

    def build(self) -> MetalSiteDescriptors:
        records: list[dict[str, object]] = []
        donor_counts: list[int] = []
        atom_counts: list[int] = []
        metal_numbers: list[int] = []
        for graph_index in range(len(self.build_ids)):
            coordinates, numbers, metal_index, donor_mask = self._graph_slice(graph_index)
            metal_numbers.append(int(numbers[metal_index]))
            record: dict[str, object] = {
                "geometry_feature_build_id": str(self.build_ids[graph_index]),
                "vr_graph_index": int(graph_index),
            }
            donor_counts.append(int(donor_mask.sum()))
            atom_counts.append(int(len(coordinates)))
            if "ligand_field" in self.blocks:
                donor_vectors = coordinates[donor_mask] - coordinates[metal_index]
                for name, value in self._selected(
                    "ligand_field", ligand_field_descriptors(donor_vectors)
                ).items():
                    record[f"{LIGAND_FIELD_PREFIX}{name}"] = value
            if "enclosure" in self.blocks:
                for name, value in self._selected(
                    "enclosure",
                    enclosure_descriptors(
                        coordinates, numbers, metal_index, self.directions
                    ),
                ).items():
                    record[f"{ENCLOSURE_PREFIX}{name}"] = value
            records.append(record)

        frame = pd.DataFrame.from_records(records)
        columns = tuple(
            column
            for column in frame.columns
            if column not in {"geometry_feature_build_id", "vr_graph_index"}
        )
        values = frame.loc[:, list(columns)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Geometry descriptors contain non-finite values.")
        constant = [
            column
            for column in columns
            if float(np.nanstd(frame[column].to_numpy(dtype=float))) == 0.0
        ]

        audit = {
            "vr_asset_path": str(self.path),
            "vr_asset_sha256": self.asset_sha256(),
            "blocks": list(self.blocks),
            "profile": self.profile,
            "descriptor_columns": list(columns),
            "descriptor_column_count": len(columns),
            "geometry_count": int(len(frame)),
            "ray_count": self.ray_count,
            "max_legendre_degree": MAX_LEGENDRE_DEGREE,
            "enclosure_radii_angstrom": list(ENCLOSURE_RADII),
            "donor_count_min": int(min(donor_counts)),
            "donor_count_max": int(max(donor_counts)),
            "atom_count_min": int(min(atom_counts)),
            "atom_count_max": int(max(atom_counts)),
            "constant_descriptor_columns": constant,
            "descriptor_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(
                    frame.sort_values("geometry_feature_build_id", kind="stable"),
                    index=False,
                )
                .to_numpy()
                .tobytes()
            ).hexdigest(),
        }
        return MetalSiteDescriptors(
            frame=frame,
            blocks=self.blocks,
            columns=columns,
            audit=audit,
            metal_atomic_numbers=np.asarray(metal_numbers, dtype=np.int64),
        )


def permute_descriptors(
    descriptors: MetalSiteDescriptors,
    *,
    mode: str,
    seed: int,
) -> MetalSiteDescriptors:
    """Return a descriptor table whose values are detached from their geometry.

    This is the negative control the tabular arm needs.  Adding any block of
    continuous columns to a mostly-binary fingerprint matrix changes how
    ExtraTrees samples split candidates, so a raw "with block versus without
    block" gain does not by itself demonstrate that the block carries chemistry.
    A permuted block has identical width, identical marginal distributions and
    no structural link to the row it lands on.

    ``free``
        Exchange descriptor rows across the whole asset.
    ``metal-preserving``
        Exchange descriptor rows only among geometries of the same lanthanide,
        so the paired A-minus-B contrast keeps the scale that the lanthanide
        contraction imposes and only the ligand identity link is destroyed.
        This is the stricter control of the two.
    """

    if mode not in DESCRIPTOR_PERMUTATIONS:
        raise ValueError(
            f"mode must be one of {list(DESCRIPTOR_PERMUTATIONS)}; received {mode!r}."
        )
    if mode == "none":
        return descriptors

    generator = np.random.default_rng(int(seed))
    count = len(descriptors.frame)
    if mode == "free":
        permutation = generator.permutation(count)
    else:
        permutation = np.arange(count)
        for element in np.unique(descriptors.metal_atomic_numbers):
            positions = np.flatnonzero(descriptors.metal_atomic_numbers == element)
            permutation[positions] = generator.permutation(positions)
    if len(permutation) != count or len(np.unique(permutation)) != count:
        raise AssertionError("Descriptor permutation is not a bijection.")

    frame = descriptors.frame.copy()
    columns = list(descriptors.columns)
    frame[columns] = frame[columns].to_numpy()[permutation]
    fixed_points = int((permutation == np.arange(count)).sum())
    audit = dict(descriptors.audit)
    audit.update(
        {
            "descriptor_permutation_mode": mode,
            "descriptor_permutation_seed": int(seed),
            "descriptor_permutation_fixed_points": fixed_points,
            "descriptor_permutation_note": (
                "Negative control: descriptor values are detached from their geometry. "
                "Any run using this mode must not be reported as a chemistry result."
            ),
        }
    )
    return MetalSiteDescriptors(
        frame=frame,
        blocks=descriptors.blocks,
        columns=descriptors.columns,
        audit=audit,
        metal_atomic_numbers=descriptors.metal_atomic_numbers,
    )


def attach_geometry_descriptors(
    source: pd.DataFrame,
    descriptors: MetalSiteDescriptors,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Left-join descriptors onto row-level data and audit the join.

    Rows without an accepted geometry have no descriptor and keep a missing
    value; the pair builder already drops pairs whose selected 3D contrast is
    incomplete, so descriptor availability never becomes a provenance shortcut.
    """

    for column in ("geometry_ok", "geometry_feature_build_id", "vr_graph_index"):
        if column not in source.columns:
            raise ValueError(f"Source frame is missing required column: {column}")
    collisions = sorted(set(descriptors.columns) & set(source.columns))
    if collisions:
        raise ValueError(f"Descriptor columns already exist in the source: {collisions}")

    keys = source["geometry_feature_build_id"].astype("string")
    accepted = source["geometry_ok"].to_numpy(dtype=bool)
    known = set(descriptors.frame["geometry_feature_build_id"].astype(str))
    unmatched = sorted(
        {
            str(value)
            for value, is_accepted in zip(keys.tolist(), accepted, strict=True)
            if is_accepted and str(value) not in known
        }
    )
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} accepted geometries have no descriptor row; "
            f"examples: {unmatched[:5]}"
        )

    # The bundle promises vr_graph_index and geometry_feature_build_id agree.
    # Verify rather than trust: a silent drift would mis-assign 3D features.
    index_by_key = dict(
        zip(
            descriptors.frame["geometry_feature_build_id"].astype(str),
            descriptors.frame["vr_graph_index"].astype(int),
            strict=True,
        )
    )
    declared = pd.to_numeric(source["vr_graph_index"], errors="coerce").to_numpy(dtype=float)
    mismatches = 0
    for key, is_accepted, value in zip(keys.tolist(), accepted, declared, strict=True):
        if is_accepted and index_by_key[str(key)] != int(value):
            mismatches += 1
    if mismatches:
        raise ValueError(
            f"{mismatches} rows disagree between vr_graph_index and the descriptor asset."
        )

    merged = source.merge(
        descriptors.frame.drop(columns=["vr_graph_index"]),
        how="left",
        on="geometry_feature_build_id",
        validate="many_to_one",
    )
    if len(merged) != len(source):
        raise AssertionError("Descriptor join changed the source row count.")
    attached = merged.loc[:, list(descriptors.columns)]
    audit = dict(descriptors.audit)
    audit.update(
        {
            "source_rows": int(len(source)),
            "rows_with_accepted_geometry": int(accepted.sum()),
            "rows_with_descriptors": int(attached.notna().all(axis=1).sum()),
            "descriptor_join_key": "geometry_feature_build_id",
        }
    )
    if audit["rows_with_descriptors"] != audit["rows_with_accepted_geometry"]:
        raise AssertionError("Descriptor coverage does not match accepted geometry coverage.")
    return merged, audit

"""Metal-centred, rotation-invariant neural message passing on VR simplices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SimplicialGraph:
    """One target-independent metal-centred subcomplex."""

    atomic_numbers: np.ndarray
    continuous_node_features: np.ndarray
    edge_index: np.ndarray
    edge_filtration: np.ndarray
    triangle_index: np.ndarray
    triangle_filtration: np.ndarray
    triangle_shape_features: np.ndarray
    metal_index: int


@dataclass
class PackedSimplicialBatch:
    atomic_numbers: torch.Tensor
    continuous_node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_filtration: torch.Tensor
    triangle_index: torch.Tensor
    triangle_filtration: torch.Tensor
    triangle_shape_features: torch.Tensor
    node_batch: torch.Tensor
    metal_indices: torch.Tensor
    donor_mask: torch.Tensor
    n_graphs: int

    def to(self, device: torch.device) -> "PackedSimplicialBatch":
        return PackedSimplicialBatch(
            atomic_numbers=self.atomic_numbers.to(device),
            continuous_node_features=self.continuous_node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_filtration=self.edge_filtration.to(device),
            triangle_index=self.triangle_index.to(device),
            triangle_filtration=self.triangle_filtration.to(device),
            triangle_shape_features=self.triangle_shape_features.to(device),
            node_batch=self.node_batch.to(device),
            metal_indices=self.metal_indices.to(device),
            donor_mask=self.donor_mask.to(device),
            n_graphs=self.n_graphs,
        )


class VietorisRipsStore:
    """Validated random-access view of the bundled 0/1/2-simplex arrays.

    The full VR complex can contain many ligand-internal simplices.  A fixed
    radius around the single Ln atom selects the chemically relevant local
    subcomplex without consulting the target. Size caps fail closed rather than
    truncating simplices and invalidating closure.
    """

    def __init__(
        self,
        path: Path,
        *,
        radius_angstrom: float = 5.0,
        max_edges: int = 4096,
        max_triangles: int = 8192,
        use_partial_charges: bool = False,
        shell_mode: str = "coordination",
    ) -> None:
        self.path = path.expanduser().resolve()
        self.radius_angstrom = float(radius_angstrom)
        self.max_edges = int(max_edges)
        self.max_triangles = int(max_triangles)
        self.use_partial_charges = bool(use_partial_charges)
        self.shell_mode = str(shell_mode)
        if self.radius_angstrom <= 0.0:
            raise ValueError("radius_angstrom must be positive.")
        if self.max_edges < 1 or self.max_triangles < 1:
            raise ValueError("max_edges and max_triangles must be positive.")
        if self.shell_mode not in {"coordination", "radius"}:
            raise ValueError("shell_mode must be 'coordination' or 'radius'.")

        with np.load(self.path, allow_pickle=False) as archive:
            required = {
                "coordinates",
                "atomic_numbers",
                "partial_charges",
                "is_metal",
                "is_coord_donor",
                "node_ptr",
                "edge_index",
                "edge_filtration",
                "edge_ptr",
                "triangle_index",
                "triangle_filtration",
                "triangle_ptr",
                "build_ids",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"VR archive is missing arrays: {missing}")
            self.coordinates = np.asarray(archive["coordinates"], dtype=np.float32)
            self.atomic_numbers = np.asarray(archive["atomic_numbers"], dtype=np.int64)
            self.partial_charges = np.asarray(archive["partial_charges"], dtype=np.float32)
            self.is_metal = np.asarray(archive["is_metal"], dtype=np.int8)
            self.is_coord_donor = np.asarray(archive["is_coord_donor"], dtype=np.int8)
            self.node_ptr = np.asarray(archive["node_ptr"], dtype=np.int64)
            self.edge_index = np.asarray(archive["edge_index"], dtype=np.int64)
            self.edge_filtration = np.asarray(archive["edge_filtration"], dtype=np.float32)
            self.edge_ptr = np.asarray(archive["edge_ptr"], dtype=np.int64)
            self.triangle_index = np.asarray(archive["triangle_index"], dtype=np.int64)
            self.triangle_filtration = np.asarray(
                archive["triangle_filtration"], dtype=np.float32
            )
            self.triangle_ptr = np.asarray(archive["triangle_ptr"], dtype=np.int64)
            self.build_ids = np.asarray(archive["build_ids"]).astype(str)

        self._validate_archive()
        self._index = {build_id: index for index, build_id in enumerate(self.build_ids)}
        self._cache: dict[str, SimplicialGraph] = {}

    def _validate_archive(self) -> None:
        n_graphs = len(self.build_ids)
        if len(set(self.build_ids.tolist())) != n_graphs:
            raise ValueError("VR build_ids must be unique.")
        for name, pointer, expected_end in (
            ("node_ptr", self.node_ptr, len(self.atomic_numbers)),
            ("edge_ptr", self.edge_ptr, self.edge_index.shape[1]),
            ("triangle_ptr", self.triangle_ptr, self.triangle_index.shape[1]),
        ):
            if pointer.shape != (n_graphs + 1,):
                raise ValueError(f"{name} must have length n_graphs + 1.")
            if pointer[0] != 0 or pointer[-1] != expected_end or np.any(np.diff(pointer) < 0):
                raise ValueError(f"{name} is not a valid monotone pointer array.")
        n_nodes = len(self.atomic_numbers)
        if self.coordinates.shape != (n_nodes, 3):
            raise ValueError("coordinates must have shape [n_nodes, 3].")
        for name, values in (
            ("partial_charges", self.partial_charges),
            ("is_metal", self.is_metal),
            ("is_coord_donor", self.is_coord_donor),
        ):
            if len(values) != n_nodes:
                raise ValueError(f"{name} length must match atomic_numbers.")
        if not np.isfinite(self.coordinates).all():
            raise ValueError("coordinates contain non-finite values.")
        if np.any(self.atomic_numbers < 1) or np.any(self.atomic_numbers > 118):
            raise ValueError("atomic_numbers must be in [1, 118].")
        if not np.isin(self.is_metal, (0, 1)).all() or not np.isin(
            self.is_coord_donor, (0, 1)
        ).all():
            raise ValueError("is_metal and is_coord_donor must be binary.")
        if self.edge_index.shape[0] != 2 or self.triangle_index.shape[0] != 3:
            raise ValueError("edge_index/triangle_index have invalid simplex dimensions.")
        if len(self.edge_filtration) != self.edge_index.shape[1]:
            raise ValueError("edge filtration length mismatch.")
        if len(self.triangle_filtration) != self.triangle_index.shape[1]:
            raise ValueError("triangle filtration length mismatch.")
        if (
            not np.isfinite(self.edge_filtration).all()
            or not np.isfinite(self.triangle_filtration).all()
            or np.any(self.edge_filtration < 0.0)
            or np.any(self.triangle_filtration < 0.0)
        ):
            raise ValueError("Simplex filtrations must be finite and nonnegative.")

    def require_build_ids(self, build_ids: Iterable[str]) -> None:
        missing = sorted({str(value) for value in build_ids} - set(self._index))
        if missing:
            raise ValueError(f"VR archive lacks {len(missing)} required build IDs: {missing[:5]}")

    def _triangle_shapes(
        self,
        coordinates: np.ndarray,
        triangle_index: np.ndarray,
        metal_index: int,
    ) -> np.ndarray:
        """Rotation/permutation-invariant local geometry for retained 2-simplices."""

        count = triangle_index.shape[1]
        if count == 0:
            return np.empty((0, 7), dtype=np.float32)
        points = coordinates[triangle_index.transpose()]
        side_lengths = np.stack(
            (
                np.linalg.norm(points[:, 0] - points[:, 1], axis=1),
                np.linalg.norm(points[:, 0] - points[:, 2], axis=1),
                np.linalg.norm(points[:, 1] - points[:, 2], axis=1),
            ),
            axis=1,
        )
        side_lengths.sort(axis=1)
        areas = 0.5 * np.linalg.norm(
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
            axis=1,
        )
        squared_side_sum = np.square(side_lengths).sum(axis=1)
        quality = np.divide(
            4.0 * np.sqrt(3.0) * areas,
            squared_side_sum,
            out=np.zeros_like(areas),
            where=squared_side_sum > 1e-12,
        )
        metal_angle_cosine = np.zeros(count, dtype=np.float32)
        contains_metal = np.zeros(count, dtype=np.float32)
        for index, vertices in enumerate(triangle_index.transpose()):
            other_vertices = [int(value) for value in vertices if int(value) != metal_index]
            if len(other_vertices) != 2 or metal_index not in vertices:
                continue
            first = coordinates[other_vertices[0]] - coordinates[metal_index]
            second = coordinates[other_vertices[1]] - coordinates[metal_index]
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denominator > 1e-12:
                metal_angle_cosine[index] = float(
                    np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
                )
            contains_metal[index] = 1.0
        return np.column_stack(
            (
                side_lengths,
                areas / (self.radius_angstrom**2),
                quality,
                metal_angle_cosine,
                contains_metal,
            )
        ).astype(np.float32, copy=False)

    def graph(self, build_id: str) -> SimplicialGraph:
        key = str(build_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if key not in self._index:
            raise KeyError(f"Unknown VR build_id: {key}")
        graph_index = self._index[key]
        node_start = int(self.node_ptr[graph_index])
        node_stop = int(self.node_ptr[graph_index + 1])
        coordinates = self.coordinates[node_start:node_stop]
        metal_positions = np.flatnonzero(self.is_metal[node_start:node_stop] == 1)
        if len(metal_positions) != 1:
            raise ValueError(f"{key}: expected exactly one metal node, found {len(metal_positions)}")
        original_metal_index = int(metal_positions[0])
        radial_distance = np.linalg.norm(
            coordinates - coordinates[original_metal_index], axis=1
        ).astype(np.float32)
        local_metal_flags = self.is_metal[node_start:node_stop] == 1
        local_donor_flags = self.is_coord_donor[node_start:node_stop] == 1
        if self.shell_mode == "coordination":
            keep_nodes = (local_metal_flags | local_donor_flags) & (
                radial_distance <= self.radius_angstrom
            )
        else:
            keep_nodes = radial_distance <= self.radius_angstrom
        keep_nodes[original_metal_index] = True
        selected = np.flatnonzero(keep_nodes)
        remap = np.full(len(coordinates), -1, dtype=np.int64)
        remap[selected] = np.arange(len(selected), dtype=np.int64)

        raw_charges = self.partial_charges[node_start:node_stop][selected]
        if self.use_partial_charges:
            charge_missing = ~np.isfinite(raw_charges)
            charges = np.nan_to_num(raw_charges, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Primary geometry-only mode cannot exploit xTB availability as a
            # provenance shortcut. Charges are an explicit sensitivity run.
            charges = np.zeros_like(raw_charges)
            charge_missing = np.zeros_like(raw_charges, dtype=bool)
        metal = self.is_metal[node_start:node_stop][selected].astype(np.float32)
        donor = self.is_coord_donor[node_start:node_stop][selected].astype(np.float32)
        continuous = np.column_stack(
            (
                charges,
                charge_missing.astype(np.float32),
                metal,
                donor,
                radial_distance[selected] / self.radius_angstrom,
            )
        ).astype(np.float32, copy=False)

        edge_start = int(self.edge_ptr[graph_index])
        edge_stop = int(self.edge_ptr[graph_index + 1])
        edges = self.edge_index[:, edge_start:edge_stop] - node_start
        if edges.size and (edges.min() < 0 or edges.max() >= len(coordinates)):
            raise ValueError(f"{key}: edge indices escape their graph node range.")
        edge_keep = keep_nodes[edges].all(axis=0)
        edges = remap[edges[:, edge_keep]]
        edge_filtration = self.edge_filtration[edge_start:edge_stop][edge_keep]
        if len(edge_filtration) > self.max_edges:
            raise ValueError(
                f"{key}: selected subcomplex has {len(edge_filtration)} edges; "
                "increase max_edges rather than breaking simplicial closure."
            )

        triangle_start = int(self.triangle_ptr[graph_index])
        triangle_stop = int(self.triangle_ptr[graph_index + 1])
        triangles = self.triangle_index[:, triangle_start:triangle_stop] - node_start
        if triangles.size and (
            triangles.min() < 0 or triangles.max() >= len(coordinates)
        ):
            raise ValueError(f"{key}: triangle indices escape their graph node range.")
        triangle_keep = keep_nodes[triangles].all(axis=0)
        if self.shell_mode == "coordination" and triangles.shape[1]:
            contains_metal = (triangles == original_metal_index).any(axis=0)
            donor_count = local_donor_flags[triangles].sum(axis=0)
            triangle_keep &= contains_metal & (donor_count == 2)
        retained_original_triangles = triangles[:, triangle_keep]
        triangle_shape_features = self._triangle_shapes(
            coordinates, retained_original_triangles, original_metal_index
        )
        triangles = remap[retained_original_triangles]
        triangle_filtration = self.triangle_filtration[
            triangle_start:triangle_stop
        ][triangle_keep]
        if len(triangle_filtration) > self.max_triangles:
            raise ValueError(
                f"{key}: selected subcomplex has {len(triangle_filtration)} triangles; "
                "increase max_triangles rather than breaking simplicial closure."
            )
        edge_boundaries: dict[tuple[int, int], float] = {}
        for edge, filtration in zip(
            edges.transpose(), edge_filtration, strict=True
        ):
            vertices = (int(edge[0]), int(edge[1]))
            if vertices[0] == vertices[1]:
                raise ValueError(f"{key}: retained VR edge is a self-loop.")
            boundary = tuple(sorted(vertices))
            if boundary in edge_boundaries:
                raise ValueError(f"{key}: retained VR edge is duplicated: {boundary}.")
            edge_boundaries[boundary] = float(filtration)
        for triangle, filtration in zip(
            triangles.transpose(), triangle_filtration, strict=True
        ):
            vertices = [int(value) for value in triangle]
            if len(set(vertices)) != 3:
                raise ValueError(f"{key}: retained triangle has repeated vertices.")
            boundaries = (
                tuple(sorted((vertices[0], vertices[1]))),
                tuple(sorted((vertices[0], vertices[2]))),
                tuple(sorted((vertices[1], vertices[2]))),
            )
            if any(boundary not in edge_boundaries for boundary in boundaries):
                raise ValueError(
                    f"{key}: retained triangle lacks a boundary edge in the VR asset."
                )
            tolerance = max(1e-6, 1e-5 * max(1.0, abs(float(filtration))))
            if max(edge_boundaries[boundary] for boundary in boundaries) > (
                float(filtration) + tolerance
            ):
                raise ValueError(
                    f"{key}: triangle filtration precedes one of its boundary edges."
                )
        if triangle_shape_features.shape != (triangles.shape[1], 7):
            raise AssertionError(f"{key}: internal triangle shape alignment failure.")
        if not np.isfinite(triangle_shape_features).all():
            raise ValueError(f"{key}: triangle shape features are non-finite.")

        graph = SimplicialGraph(
            atomic_numbers=self.atomic_numbers[node_start:node_stop][selected],
            continuous_node_features=continuous,
            edge_index=edges.astype(np.int64, copy=False),
            edge_filtration=edge_filtration.astype(np.float32, copy=False),
            triangle_index=triangles.astype(np.int64, copy=False),
            triangle_filtration=triangle_filtration.astype(np.float32, copy=False),
            triangle_shape_features=triangle_shape_features,
            metal_index=int(remap[original_metal_index]),
        )
        self._cache[key] = graph
        return graph

    def warm_cache(self, build_ids: Iterable[str]) -> None:
        unique = sorted({str(value) for value in build_ids})
        self.require_build_ids(unique)
        for build_id in unique:
            self.graph(build_id)

    def pack(self, build_ids: Sequence[str]) -> PackedSimplicialBatch:
        graphs = [self.graph(str(build_id)) for build_id in build_ids]
        atomic_numbers: list[np.ndarray] = []
        continuous: list[np.ndarray] = []
        edges: list[np.ndarray] = []
        edge_filtration: list[np.ndarray] = []
        triangles: list[np.ndarray] = []
        triangle_filtration: list[np.ndarray] = []
        triangle_shape_features: list[np.ndarray] = []
        node_batch: list[np.ndarray] = []
        metal_indices: list[int] = []
        donor_masks: list[np.ndarray] = []
        offset = 0
        for batch_index, graph in enumerate(graphs):
            n_nodes = len(graph.atomic_numbers)
            atomic_numbers.append(graph.atomic_numbers)
            continuous.append(graph.continuous_node_features)
            if graph.edge_index.shape[1]:
                edges.append(graph.edge_index + offset)
                edge_filtration.append(graph.edge_filtration)
            if graph.triangle_index.shape[1]:
                triangles.append(graph.triangle_index + offset)
                triangle_filtration.append(graph.triangle_filtration)
                triangle_shape_features.append(graph.triangle_shape_features)
            node_batch.append(np.full(n_nodes, batch_index, dtype=np.int64))
            metal_indices.append(offset + graph.metal_index)
            donor_masks.append(graph.continuous_node_features[:, 3] > 0.5)
            offset += n_nodes

        packed_edges = (
            np.concatenate(edges, axis=1) if edges else np.empty((2, 0), dtype=np.int64)
        )
        packed_triangles = (
            np.concatenate(triangles, axis=1)
            if triangles
            else np.empty((3, 0), dtype=np.int64)
        )
        return PackedSimplicialBatch(
            atomic_numbers=torch.from_numpy(np.concatenate(atomic_numbers)),
            continuous_node_features=torch.from_numpy(np.concatenate(continuous, axis=0)),
            edge_index=torch.from_numpy(packed_edges),
            edge_filtration=torch.from_numpy(
                np.concatenate(edge_filtration)
                if edge_filtration
                else np.empty(0, dtype=np.float32)
            ),
            triangle_index=torch.from_numpy(packed_triangles),
            triangle_filtration=torch.from_numpy(
                np.concatenate(triangle_filtration)
                if triangle_filtration
                else np.empty(0, dtype=np.float32)
            ),
            triangle_shape_features=torch.from_numpy(
                np.concatenate(triangle_shape_features, axis=0)
                if triangle_shape_features
                else np.empty((0, 7), dtype=np.float32)
            ),
            node_batch=torch.from_numpy(np.concatenate(node_batch)),
            metal_indices=torch.tensor(metal_indices, dtype=torch.long),
            donor_mask=torch.from_numpy(np.concatenate(donor_masks)),
            n_graphs=len(graphs),
        )


def set_torch_determinism(seed: int, *, strict: bool = True) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=not strict)


class RadialBasis(nn.Module):
    def __init__(self, count: int, maximum: float) -> None:
        super().__init__()
        centers = torch.linspace(0.0, float(maximum), int(count))
        self.register_buffer("centers", centers)
        spacing = float(maximum) / max(int(count) - 1, 1)
        self.gamma = 1.0 / max(spacing * spacing, 1e-6)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (values.unsqueeze(-1) - self.centers) ** 2)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def _incidence_mean(
    messages: torch.Tensor,
    simplex_index: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    if messages.shape[0] == 0:
        return messages.new_zeros((n_nodes, messages.shape[1]))
    width = simplex_index.shape[0]
    vertices = simplex_index.transpose(0, 1).reshape(-1)
    repeated = messages.repeat_interleave(width, dim=0)
    output = messages.new_zeros((n_nodes, messages.shape[1]))
    output.index_add_(0, vertices, repeated)
    counts = messages.new_zeros((n_nodes, 1))
    counts.index_add_(0, vertices, messages.new_ones((len(vertices), 1)))
    return output / counts.clamp_min(1.0)


def _graph_mean(
    values: torch.Tensor,
    batch: torch.Tensor,
    n_graphs: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        values = values[mask]
        batch = batch[mask]
    output = values.new_zeros((n_graphs, values.shape[1]))
    counts = values.new_zeros((n_graphs, 1))
    if len(values):
        output.index_add_(0, batch, values)
        counts.index_add_(0, batch, values.new_ones((len(values), 1)))
    return output / counts.clamp_min(1.0)


# Declared encoder ladder (STEP 3).  Each mode removes one level of
# structure from the same network, so a gain can be attributed to the
# simplicial order that produced it rather than to capacity:
#
#   distances             filtration geometry only -- no atom identity and no
#                         xTB-derived node features; the metal/donor role flags
#                         survive because the pooled readout is defined by them.
#   nodes_distances       atom identity plus per-atom features (which include
#                         the Ln-atom radial distance); no 1- or 2-simplices.
#   nodes_edges           the above plus 1-simplex (pairwise-distance) messages.
#   nodes_edges_triangles the full 0/1/2-simplex network.  Default; numerically
#                         identical to the pre-ladder implementation.
SIMPLEX_ORDERS: tuple[str, ...] = (
    "distances",
    "nodes_distances",
    "nodes_edges",
    "nodes_edges_triangles",
)

# Column indices of continuous_node_features: charge, charge_missing, is_metal,
# is_donor, radial distance / shell radius.  The distance-only encoder keeps
# only the last three: the two role flags and the metal-centred distance.
_GEOMETRY_ONLY_NODE_FEATURES: tuple[int, ...] = (2, 3, 4)


class SimplicialEncoder(nn.Module):
    """0/1/2-simplex incidence network with invariant filtration features."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        layers: int = 3,
        dropout: float = 0.10,
        rbf_count: int = 16,
        max_filtration: float = 4.0,
        simplex_order: str = "nodes_edges_triangles",
    ) -> None:
        super().__init__()
        if simplex_order not in SIMPLEX_ORDERS:
            raise ValueError(
                f"Unknown simplex_order {simplex_order!r}; expected one of "
                f"{SIMPLEX_ORDERS}."
            )
        self.simplex_order = str(simplex_order)
        self.use_atom_identity = simplex_order != "distances"
        self.use_edges = simplex_order in {
            "distances",
            "nodes_edges",
            "nodes_edges_triangles",
        }
        self.use_triangles = simplex_order == "nodes_edges_triangles"
        if int(hidden_dim) < 4:
            raise ValueError("hidden_dim must be at least 4.")
        if int(layers) < 1 or int(rbf_count) < 2:
            raise ValueError("layers must be positive and rbf_count must be at least 2.")
        if float(max_filtration) <= 0.0:
            raise ValueError("max_filtration must be positive.")
        self.hidden_dim = int(hidden_dim)
        # Metal state anchors the complex; donor mean/dispersion retains shell
        # heterogeneity; direct triangle pooling prevents the 2-simplex signal
        # from being represented only after a node-incidence bottleneck.
        self.output_dim = self.hidden_dim * 5
        atomic_dim = min(24, self.hidden_dim // 2)
        self.atomic_embedding = nn.Embedding(119, atomic_dim, padding_idx=0)
        self.node_input = _mlp(atomic_dim + 5, self.hidden_dim, self.hidden_dim, dropout)
        self.rbf = RadialBasis(rbf_count, max_filtration)
        edge_input_dim = self.hidden_dim * 2 + rbf_count
        triangle_input_dim = self.hidden_dim * 2 + 3 * rbf_count + 4
        self.edge_networks = nn.ModuleList(
            _mlp(edge_input_dim, self.hidden_dim, self.hidden_dim, dropout)
            for _ in range(int(layers))
        )
        self.triangle_networks = nn.ModuleList(
            _mlp(triangle_input_dim, self.hidden_dim, self.hidden_dim, dropout)
            for _ in range(int(layers))
        )
        self.node_updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
            )
            for _ in range(int(layers))
        )

    def forward(self, batch: PackedSimplicialBatch) -> torch.Tensor:
        n_nodes = int(batch.atomic_numbers.shape[0])
        n_edges = int(batch.edge_index.shape[1])
        n_triangles = int(batch.triangle_index.shape[1])
        if batch.continuous_node_features.shape != (n_nodes, 5):
            raise ValueError("continuous_node_features must have shape [N, 5].")
        if batch.edge_index.shape[0] != 2 or batch.edge_filtration.shape != (n_edges,):
            raise ValueError("Packed edge tensors have inconsistent shapes.")
        if (
            batch.triangle_index.shape[0] != 3
            or batch.triangle_filtration.shape != (n_triangles,)
            or batch.triangle_shape_features.shape != (n_triangles, 7)
        ):
            raise ValueError("Packed triangle tensors have inconsistent shapes.")
        atomic = self.atomic_embedding(batch.atomic_numbers.clamp(0, 118))
        continuous = batch.continuous_node_features
        if not self.use_atom_identity:
            # Distance-only arm: strip chemical identity and the xTB-derived
            # node channels, keep the metal/donor roles the readout is defined
            # by plus the metal-centred radial distance.  Zeroing rather than
            # resizing keeps every weight shape -- and so the model capacity --
            # identical to the full encoder.
            atomic = torch.zeros_like(atomic)
            geometry_mask = torch.zeros_like(continuous)
            for column in _GEOMETRY_ONLY_NODE_FEATURES:
                geometry_mask[:, column] = 1.0
            continuous = continuous * geometry_mask
        node_state = self.node_input(torch.cat((atomic, continuous), dim=1))
        for edge_network, triangle_network, node_update in zip(
            self.edge_networks, self.triangle_networks, self.node_updates, strict=True
        ):
            if self.use_edges and batch.edge_index.shape[1]:
                source = node_state[batch.edge_index[0]]
                target = node_state[batch.edge_index[1]]
                edge_messages = edge_network(
                    torch.cat(
                        (
                            source + target,
                            torch.abs(source - target),
                            self.rbf(batch.edge_filtration),
                        ),
                        dim=1,
                    )
                )
            else:
                edge_messages = node_state.new_zeros((0, self.hidden_dim))
            edge_aggregate = _incidence_mean(
                edge_messages, batch.edge_index, len(node_state)
            )

            if self.use_triangles and batch.triangle_index.shape[1]:
                triangle_nodes = node_state[batch.triangle_index.transpose(0, 1)]
                triangle_sum = triangle_nodes.sum(dim=1)
                triangle_dispersion = (
                    torch.abs(triangle_nodes[:, 0] - triangle_nodes[:, 1])
                    + torch.abs(triangle_nodes[:, 0] - triangle_nodes[:, 2])
                    + torch.abs(triangle_nodes[:, 1] - triangle_nodes[:, 2])
                ) / 3.0
                triangle_messages = triangle_network(
                    torch.cat(
                        (
                            triangle_sum / 3.0,
                            triangle_dispersion,
                            self.rbf(
                                batch.triangle_shape_features[:, :3]
                            ).flatten(start_dim=1),
                            batch.triangle_shape_features[:, 3:],
                        ),
                        dim=1,
                    )
                )
            else:
                triangle_messages = node_state.new_zeros((0, self.hidden_dim))
            triangle_aggregate = _incidence_mean(
                triangle_messages, batch.triangle_index, len(node_state)
            )
            node_state = node_state + node_update(
                torch.cat((node_state, edge_aggregate, triangle_aggregate), dim=1)
            )

        metal_state = node_state[batch.metal_indices]
        donor_mean = _graph_mean(
            node_state,
            batch.node_batch,
            batch.n_graphs,
            mask=batch.donor_mask,
        )
        donor_dispersion = _graph_mean(
            torch.abs(node_state - donor_mean[batch.node_batch]),
            batch.node_batch,
            batch.n_graphs,
            mask=batch.donor_mask,
        )
        triangle_batch = (
            batch.node_batch[batch.triangle_index[0]]
            if n_triangles and self.use_triangles
            else batch.node_batch.new_zeros((0,))
        )
        triangle_mean = _graph_mean(
            triangle_messages,
            triangle_batch,
            batch.n_graphs,
        )
        all_mean = _graph_mean(node_state, batch.node_batch, batch.n_graphs)
        return torch.cat(
            (metal_state, donor_mean, donor_dispersion, triangle_mean, all_mean),
            dim=1,
        )


class AntisymmetricSimplicialPairRegressor(nn.Module):
    """Shared complex encoder plus an exactly odd pair readout."""

    def __init__(
        self,
        *,
        context_dim: int,
        hidden_dim: int = 64,
        layers: int = 3,
        dropout: float = 0.10,
        rbf_count: int = 16,
        max_filtration: float = 4.0,
        simplex_order: str = "nodes_edges_triangles",
    ) -> None:
        super().__init__()
        if int(context_dim) < 1:
            raise ValueError("context_dim must be positive.")
        self.encoder = SimplicialEncoder(
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
            rbf_count=rbf_count,
            max_filtration=max_filtration,
            simplex_order=simplex_order,
        )
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim + int(context_dim), hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        complexes: PackedSimplicialBatch,
        symmetric_context: torch.Tensor,
    ) -> torch.Tensor:
        if complexes.n_graphs % 2:
            raise ValueError("Packed pair batch must contain A graphs followed by B graphs.")
        batch_size = complexes.n_graphs // 2
        if symmetric_context.ndim != 2 or symmetric_context.shape[0] != batch_size:
            raise ValueError("symmetric_context must have one row per packed pair.")
        if symmetric_context.shape[1] != self.head[0].in_features - self.encoder.output_dim:
            raise ValueError("symmetric_context has the wrong feature dimension.")
        embeddings = self.encoder(complexes)
        delta = embeddings[:batch_size] - embeddings[batch_size:]
        forward = self.head(torch.cat((delta, symmetric_context), dim=1)).squeeze(1)
        reverse = self.head(torch.cat((-delta, symmetric_context), dim=1)).squeeze(1)
        return 0.5 * (forward - reverse)

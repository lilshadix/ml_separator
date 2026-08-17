"""Extended 2D ligand descriptors for amide-type (DGA-like) extractants.

The frozen 2D ligand contract is 2048 ECFP bits plus ten generic RDKit scalars.
Hashed circular fingerprints saturate after a radius of two or three bonds, so
an N-octyl and an N-dodecyl homologue set the same bits and the ten scalars
only see the resulting change in molecular weight and rotatable-bond count.
This module adds two blocks that a tree model can split on directly:

``lig2d__rd__<name>``
    Every descriptor in :data:`rdkit.Chem.Descriptors.descList`, computed on the
    canonical SMILES.  Descriptors that raise are recorded as NaN; columns that
    are constant or entirely NaN across the ligand set carry no information for
    a tree and are dropped by :func:`build_descriptor_table`.

``lig2d__hc__<name>``
    Hand-crafted graph descriptors aimed at diglycolamide, malonamide and
    related amide extractants: donor-atom counts, N-substituent alkyl-chain
    statistics (chain length, alpha/beta branching, methyl and aryl groups,
    symmetry), backbone substitution, stereocentres, the longest sp3 carbon
    chain and a few chelate-motif counts.  Every value is derived by SMARTS
    matches or explicit graph walks so the definitions below are exact.

Definitions used by the hand-crafted block
------------------------------------------
* amide N: an aliphatic nitrogen single-bonded to a carbon that carries a
  double bond to oxygen (SMARTS ``[N]-[#6]=[OX1]``).  Thioamide N
  (``[N]-[#6]=[SX1]``) is counted separately and treated as amide-like for the
  substituent statistics so that thioamide analogues do not fall out of the
  chain statistics.
* N-substituent: for each amide-like N, every heavy neighbour that is not one
  of its (thio)carbonyl carbons starts a substituent.  The substituent atom set
  is the connected component reached from that neighbour without passing
  through any amide-like nitrogen.  Ring amides (pyrrolidine, piperidine,
  morpholine) therefore contribute two substituents that share one atom set.
* chain length: the number of carbons on the longest simple path over sp3
  carbons that starts at the attachment atom and stays inside the substituent.
  A methyl is 1, n-octyl is 8, 2-ethylhexyl is 6, an aryl attachment is 0.
* alpha/beta branching: the attachment carbon (alpha) is branched when it is an
  sp3 carbon with more than two heavy neighbours (secondary or tertiary alkyl on
  N); the substituent is beta-branched when any sp3 carbon bonded to the alpha
  carbon has more than two heavy neighbours (2-ethylhexyl, isobutyl).
* longest_carbon_chain: the longest simple path over sp3 carbons anywhere in
  the molecule; heteroatoms, aromatic and carbonyl carbons break the chain.

RDKit is imported lazily inside the functions so this module can be imported
in an environment without RDKit; only building a table requires it.
"""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rdkit.Chem import Mol


SMILES_COLUMN = "canonical_smiles"
DESCRIPTOR_PREFIX = "lig2d__"
RDKIT_PREFIX = "lig2d__rd__"
HAND_CRAFTED_PREFIX = "lig2d__hc__"

# Fixed output schema of hand_crafted_descriptors: every call returns exactly
# these keys, NaN where a value is undefined or its computation failed.
HAND_CRAFTED_DESCRIPTOR_NAMES: tuple[str, ...] = (
    # composition and donor atoms
    "n_heavy_atoms",
    "n_carbons",
    "n_aromatic_atoms",
    "n_amide_N",
    "n_thioamide_N",
    "n_NH_amide",
    "n_amide_N_in_ring",
    "n_carbonyl_O",
    "n_ether_O",
    "n_amine_N",
    "n_aromatic_N",
    "n_phosphoryl_O",
    "n_thiocarbonyl_S",
    "n_hydroxyl_O",
    "n_donor_atoms",
    # N-substituent statistics
    "n_N_substituents",
    "chain_len_max",
    "chain_len_min",
    "chain_len_mean",
    "chain_len_asymmetry",
    "n_branched_substituents",
    "n_alpha_branched_substituents",
    "n_beta_branched_substituents",
    "n_methyl_on_N",
    "n_aryl_on_N",
    "n_C_in_N_substituents",
    "subst_heavy_max",
    "frac_heavy_in_largest_N_substituent",
    "is_symmetric_amide",
    "is_symmetric_across_N",
    # backbone and stereochemistry
    "n_substituted_alpha_carbons",
    "n_stereocentres",
    "n_assigned_stereocentres",
    "assigned_cip_homochiral",
    # whole-molecule chain and chelate motifs
    "longest_carbon_chain",
    "n_dga_motifs",
    "n_malonamide_motifs",
    "amide_carbonyl_min_path",
)

# Simple-path enumeration is exponential in pathological ring systems; the
# budget bounds the work per call and, if ever hit, returns the best path seen.
_PATH_SEARCH_BUDGET = 200_000

_SMARTS: dict[str, str] = {
    "amide": "[N]-[#6]=[OX1]",
    "thioamide": "[N]-[#6]=[SX1]",
    "carbonyl_O": "[OX1]=[#6]",
    # O bonded to two carbons; ester-type O next to a carbonyl carbon is not a
    # useful donor and is excluded.
    "ether_O": "[#8X2;!$([#8]-[#6]=[OX1])]([#6])[#6]",
    "aromatic_N": "[n]",
    "phosphoryl_O": "[OX1]=[#15]",
    "thiocarbonyl_S": "[SX1]=[#6,#15]",
    "hydroxyl_O": "[OX2H]",
    "alpha_carbon": "[#6](=[OX1])-[#6;!a]",
    # Diglycolamide N-C(=O)-C-O-C-C(=O)-N; the bridging carbons may be sp3 or
    # aromatic (pyranone/furan analogues) and the oxygen aliphatic or aromatic.
    "dga_motif": "[N]C(=O)[#6;X4,a][#8X2][#6;X4,a]C(=O)[N]",
    "malonamide_motif": "[N]C(=O)[CX4]C(=O)[N]",
}


def _import_chem() -> Any:
    """Import :mod:`rdkit.Chem` lazily with an actionable error message."""

    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - exercised only without RDKit
        raise ImportError(
            "RDKit is required to compute ligand descriptors; run with an "
            "interpreter that has rdkit installed (e.g. the 'lanth' conda env)."
        ) from exc
    return Chem


def mol_from_smiles(smiles: str) -> Mol | None:
    """Parse a SMILES string, returning ``None`` instead of raising."""

    Chem = _import_chem()
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:  # noqa: BLE001 - parsing must never abort a table build
        return None


# --------------------------------------------------------------------------- #
# Graph helpers
# --------------------------------------------------------------------------- #


def _heavy_neighbours(atom: Any) -> list[Any]:
    return [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]


def _heavy_degree(atom: Any) -> int:
    return len(_heavy_neighbours(atom))


def _is_sp3_carbon(atom: Any) -> bool:
    if atom.GetAtomicNum() != 6 or atom.GetIsAromatic():
        return False
    return atom.GetHybridization() == _import_chem().HybridizationType.SP3


def _component(start: int, allowed: set[int], adjacency: dict[int, list[int]]) -> set[int]:
    """Connected component of ``start`` restricted to ``allowed`` atoms."""

    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbour in adjacency[current]:
            if neighbour in allowed and neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return seen


def _bfs_depths(start: int, nodes: set[int], adjacency: dict[int, list[int]]) -> dict[int, int]:
    depths = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbour in adjacency[current]:
            if neighbour in nodes and neighbour not in depths:
                depths[neighbour] = depths[current] + 1
                queue.append(neighbour)
    return depths


def _is_tree(nodes: set[int], adjacency: dict[int, list[int]]) -> bool:
    edges = sum(1 for node in nodes for other in adjacency[node] if other in nodes) // 2
    return edges == len(nodes) - 1


def _dfs_longest_from(
    start: int, nodes: set[int], adjacency: dict[int, list[int]], budget: int
) -> int:
    """Longest simple path (in atoms) from ``start`` inside ``nodes`` by DFS."""

    best = 1
    path = [start]
    on_path = {start}
    iterators = [iter(adjacency[start])]
    steps = 0
    while iterators:
        steps += 1
        if steps > budget:
            break
        try:
            candidate = next(iterators[-1])
        except StopIteration:
            iterators.pop()
            on_path.discard(path.pop())
            continue
        if candidate not in nodes or candidate in on_path:
            continue
        path.append(candidate)
        on_path.add(candidate)
        iterators.append(iter(adjacency[candidate]))
        best = max(best, len(path))
    return best


def _longest_path_from(start: int, nodes: set[int], adjacency: dict[int, list[int]]) -> int:
    """Number of atoms on the longest simple path from ``start`` within ``nodes``."""

    if start not in nodes:
        return 0
    component = _component(start, nodes, adjacency)
    if _is_tree(component, adjacency):
        return max(_bfs_depths(start, component, adjacency).values()) + 1
    return _dfs_longest_from(start, component, adjacency, _PATH_SEARCH_BUDGET)


def _longest_path(nodes: set[int], adjacency: dict[int, list[int]]) -> int:
    """Number of atoms on the longest simple path anywhere within ``nodes``."""

    best = 0
    remaining = set(nodes)
    while remaining:
        seed = next(iter(remaining))
        component = _component(seed, remaining, adjacency)
        remaining -= component
        if _is_tree(component, adjacency):
            # Tree diameter by the classic double BFS.
            far = max(_bfs_depths(seed, component, adjacency).items(), key=lambda kv: kv[1])[0]
            best = max(best, max(_bfs_depths(far, component, adjacency).values()) + 1)
        else:
            for node in sorted(component):
                best = max(
                    best, _dfs_longest_from(node, component, adjacency, _PATH_SEARCH_BUDGET)
                )
    return best


def _adjacency(mol: Any) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = {}
    for atom in mol.GetAtoms():
        adjacency[atom.GetIdx()] = [n.GetIdx() for n in _heavy_neighbours(atom)]
    return adjacency


def _match_count(mol: Any, key: str) -> int:
    Chem = _import_chem()
    pattern = Chem.MolFromSmarts(_SMARTS[key])
    return len(mol.GetSubstructMatches(pattern, uniquify=True))


# --------------------------------------------------------------------------- #
# Hand-crafted blocks
# --------------------------------------------------------------------------- #


class _AmideContext:
    """Amide-like nitrogens and their (thio)carbonyl carbons for one molecule."""

    def __init__(self, mol: Any) -> None:
        Chem = _import_chem()
        self.amide_N: set[int] = set()
        self.thioamide_N: set[int] = set()
        self.carbonyl_C_by_N: dict[int, set[int]] = {}
        self.amide_carbonyl_C: set[int] = set()
        for key, target in (("amide", self.amide_N), ("thioamide", self.thioamide_N)):
            pattern = Chem.MolFromSmarts(_SMARTS[key])
            for n_idx, c_idx, _ in mol.GetSubstructMatches(pattern, uniquify=False):
                target.add(n_idx)
                self.carbonyl_C_by_N.setdefault(n_idx, set()).add(c_idx)
                self.amide_carbonyl_C.add(c_idx)

    @property
    def amide_like_N(self) -> set[int]:
        return self.amide_N | self.thioamide_N


class _Substituent:
    """One non-carbonyl heavy substituent on an amide-like nitrogen."""

    __slots__ = (
        "nitrogen",
        "attachment",
        "atoms",
        "chain_len",
        "alpha_branched",
        "beta_branched",
        "is_methyl",
        "is_aryl",
    )

    def __init__(
        self,
        mol: Any,
        nitrogen: int,
        attachment: int,
        ctx: _AmideContext,
        adjacency: dict[int, list[int]],
    ) -> None:
        self.nitrogen = nitrogen
        self.attachment = attachment
        # The substituent is everything reachable from the attachment atom
        # without crossing an amide-like nitrogen (this one or any other).
        allowed = {idx for idx in adjacency if idx not in ctx.amide_like_N} | {attachment}
        self.atoms = _component(attachment, allowed, adjacency)

        atom = mol.GetAtomWithIdx(attachment)
        sp3 = {idx for idx in self.atoms if _is_sp3_carbon(mol.GetAtomWithIdx(idx))}
        self.chain_len = _longest_path_from(attachment, sp3, adjacency)
        self.is_aryl = bool(atom.GetIsAromatic())
        self.is_methyl = len(self.atoms) == 1 and atom.GetAtomicNum() == 6 and not self.is_aryl
        self.alpha_branched = attachment in sp3 and _heavy_degree(atom) > 2
        self.beta_branched = attachment in sp3 and any(
            neighbour in sp3
            and neighbour != nitrogen
            and _heavy_degree(mol.GetAtomWithIdx(neighbour)) > 2
            for neighbour in adjacency[attachment]
        )


def _composition_block(mol: Any, ctx: _AmideContext) -> dict[str, float]:
    heavy = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    n_nitrogen = sum(1 for atom in heavy if atom.GetAtomicNum() == 7)
    n_amide = len(ctx.amide_N)
    n_thioamide = len(ctx.thioamide_N - ctx.amide_N)
    n_carbonyl_O = _match_count(mol, "carbonyl_O")
    n_ether_O = _match_count(mol, "ether_O")
    n_amine_N = n_nitrogen - len(ctx.amide_like_N)
    n_NH = sum(
        1 for idx in ctx.amide_like_N if mol.GetAtomWithIdx(idx).GetTotalNumHs() > 0
    )
    n_ring = sum(1 for idx in ctx.amide_like_N if mol.GetAtomWithIdx(idx).IsInRing())
    return {
        "n_heavy_atoms": float(len(heavy)),
        "n_carbons": float(sum(1 for atom in heavy if atom.GetAtomicNum() == 6)),
        "n_aromatic_atoms": float(sum(1 for atom in heavy if atom.GetIsAromatic())),
        "n_amide_N": float(n_amide),
        "n_thioamide_N": float(n_thioamide),
        "n_NH_amide": float(n_NH),
        "n_amide_N_in_ring": float(n_ring),
        "n_carbonyl_O": float(n_carbonyl_O),
        "n_ether_O": float(n_ether_O),
        "n_amine_N": float(n_amine_N),
        "n_aromatic_N": float(_match_count(mol, "aromatic_N")),
        "n_phosphoryl_O": float(_match_count(mol, "phosphoryl_O")),
        "n_thiocarbonyl_S": float(_match_count(mol, "thiocarbonyl_S")),
        "n_hydroxyl_O": float(_match_count(mol, "hydroxyl_O")),
        "n_donor_atoms": float(n_carbonyl_O + n_ether_O + n_amine_N),
    }


def _substituent_block(
    mol: Any, ctx: _AmideContext, adjacency: dict[int, list[int]]
) -> dict[str, float]:
    substituents: list[_Substituent] = []
    for n_idx in sorted(ctx.amide_like_N):
        carbonyls = ctx.carbonyl_C_by_N.get(n_idx, set())
        for neighbour in adjacency[n_idx]:
            if neighbour in carbonyls:
                continue
            substituents.append(_Substituent(mol, n_idx, neighbour, ctx, adjacency))

    n_heavy = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    values: dict[str, float] = {
        "n_N_substituents": float(len(substituents)),
        "n_branched_substituents": float(
            sum(1 for s in substituents if s.alpha_branched or s.beta_branched)
        ),
        "n_alpha_branched_substituents": float(sum(1 for s in substituents if s.alpha_branched)),
        "n_beta_branched_substituents": float(sum(1 for s in substituents if s.beta_branched)),
        "n_methyl_on_N": float(sum(1 for s in substituents if s.is_methyl)),
        "n_aryl_on_N": float(sum(1 for s in substituents if s.is_aryl)),
    }
    if not substituents:
        # No amide-like nitrogen: sizes are zero, chain statistics and the
        # symmetry flags are undefined rather than zero.
        values.update(
            {
                "n_C_in_N_substituents": 0.0,
                "subst_heavy_max": 0.0,
                "frac_heavy_in_largest_N_substituent": 0.0,
            }
        )
        for name in (
            "chain_len_max",
            "chain_len_min",
            "chain_len_mean",
            "chain_len_asymmetry",
            "is_symmetric_amide",
            "is_symmetric_across_N",
        ):
            values[name] = float("nan")
        return values

    lengths = [s.chain_len for s in substituents]
    union_atoms: set[int] = set().union(*(s.atoms for s in substituents))
    per_nitrogen: dict[int, list[int]] = {}
    for s in substituents:
        per_nitrogen.setdefault(s.nitrogen, []).append(s.chain_len)
    signatures = {tuple(sorted(chains)) for chains in per_nitrogen.values()}
    largest = max(len(s.atoms) for s in substituents)
    values.update(
        {
            "chain_len_max": float(max(lengths)),
            "chain_len_min": float(min(lengths)),
            "chain_len_mean": float(sum(lengths) / len(lengths)),
            "chain_len_asymmetry": float(max(lengths) - min(lengths)),
            "n_C_in_N_substituents": float(
                sum(1 for idx in union_atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6)
            ),
            "subst_heavy_max": float(largest),
            "frac_heavy_in_largest_N_substituent": float(largest / n_heavy) if n_heavy else 0.0,
            "is_symmetric_amide": 1.0 if len(set(lengths)) == 1 else 0.0,
            "is_symmetric_across_N": 1.0 if len(signatures) == 1 else 0.0,
        }
    )
    return values


def _backbone_block(mol: Any) -> dict[str, float]:
    Chem = _import_chem()
    pattern = Chem.MolFromSmarts(_SMARTS["alpha_carbon"])
    alpha = {match[2] for match in mol.GetSubstructMatches(pattern, uniquify=False)}
    substituted = sum(1 for idx in alpha if _heavy_degree(mol.GetAtomWithIdx(idx)) > 2)
    return {"n_substituted_alpha_carbons": float(substituted)}


def _stereo_block(mol: Any) -> dict[str, float]:
    Chem = _import_chem()
    centres = Chem.FindMolChiralCenters(
        mol, force=True, includeUnassigned=True, useLegacyImplementation=False
    )
    labels = [label for _, label in centres if label in ("R", "S")]
    if len(labels) >= 2:
        homochiral = 1.0 if len(set(labels)) == 1 else 0.0
    else:
        homochiral = float("nan")
    return {
        "n_stereocentres": float(len(centres)),
        "n_assigned_stereocentres": float(len(labels)),
        "assigned_cip_homochiral": homochiral,
    }


def _chain_block(mol: Any, adjacency: dict[int, list[int]]) -> dict[str, float]:
    sp3 = {atom.GetIdx() for atom in mol.GetAtoms() if _is_sp3_carbon(atom)}
    return {"longest_carbon_chain": float(_longest_path(sp3, adjacency))}


def _motif_block(mol: Any, ctx: _AmideContext) -> dict[str, float]:
    Chem = _import_chem()
    carbonyls = sorted(ctx.amide_carbonyl_C)
    if len(carbonyls) >= 2:
        distances = Chem.GetDistanceMatrix(mol)
        min_path = min(
            float(distances[i][j])
            for pos, i in enumerate(carbonyls)
            for j in carbonyls[pos + 1:]
        )
    else:
        min_path = float("nan")
    return {
        "n_dga_motifs": float(_match_count(mol, "dga_motif")),
        "n_malonamide_motifs": float(_match_count(mol, "malonamide_motif")),
        "amide_carbonyl_min_path": min_path,
    }


def hand_crafted_descriptors(mol: Mol | None) -> dict[str, float]:
    """Hand-crafted DGA/amide descriptors keyed by :data:`HAND_CRAFTED_DESCRIPTOR_NAMES`.

    Never raises for a valid RDKit molecule: each block is guarded so a failure
    in one leaves the others populated, and the failed block's keys are NaN.
    ``None`` (an unparsable SMILES) yields an all-NaN row.
    """

    values: dict[str, float] = {name: float("nan") for name in HAND_CRAFTED_DESCRIPTOR_NAMES}
    if mol is None:
        return values
    Chem = _import_chem()
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:  # noqa: BLE001 - keep the original graph if RemoveHs fails
        pass

    blocks: list[Any] = [lambda: _backbone_block(mol), lambda: _stereo_block(mol)]
    try:
        adjacency = _adjacency(mol)
        ctx = _AmideContext(mol)
    except Exception:  # noqa: BLE001 - graph-dependent blocks stay NaN
        pass
    else:
        blocks.extend(
            [
                lambda: _composition_block(mol, ctx),
                lambda: _substituent_block(mol, ctx, adjacency),
                lambda: _motif_block(mol, ctx),
                lambda: _chain_block(mol, adjacency),
            ]
        )
    for block in blocks:
        try:
            result = block()
        except Exception:  # noqa: BLE001 - NaN on failure, never crash
            continue
        unknown = set(result) - set(HAND_CRAFTED_DESCRIPTOR_NAMES)
        if unknown:
            raise AssertionError(f"Hand-crafted block emitted unknown names: {sorted(unknown)}")
        for name, value in result.items():
            values[name] = float(value) if value is not None else float("nan")
    return values


# --------------------------------------------------------------------------- #
# RDKit descriptor block and table assembly
# --------------------------------------------------------------------------- #


def rdkit_descriptor_row(mol: Mol | None) -> dict[str, float]:
    """All of ``Descriptors.descList`` for one molecule; NaN where a descriptor raises."""

    from rdkit.Chem import Descriptors

    row: dict[str, float] = {}
    for name, function in Descriptors.descList:
        if mol is None:
            row[name] = float("nan")
            continue
        try:
            value = float(function(mol))
        except Exception:  # noqa: BLE001 - descriptor failure is data, not an error
            value = float("nan")
        row[name] = value if math.isfinite(value) else float("nan")
    return row


def _uninformative_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    dropped: list[str] = []
    for column in columns:
        series = frame[column]
        if series.isna().all() or series.nunique(dropna=True) <= 1:
            dropped.append(column)
    return dropped


def build_descriptor_table(
    smiles: Iterable[str],
    *,
    drop_uninformative: bool = True,
) -> pd.DataFrame:
    """Compute one descriptor row per unique SMILES (first-appearance order).

    Returns a frame with ``canonical_smiles`` followed by ``lig2d__rd__*`` and
    ``lig2d__hc__*`` float columns.  With ``drop_uninformative`` the RDKit
    columns that are all-NaN or constant across the ligands are removed; the
    hand-crafted schema is always kept complete.  Build metadata is stored in
    ``frame.attrs`` under ``dropped_rdkit_columns``,
    ``constant_hand_crafted_columns`` and ``unparsed_smiles``.
    """

    unique = list(dict.fromkeys(str(value) for value in smiles))
    records: list[dict[str, Any]] = []
    unparsed: list[str] = []
    for value in unique:
        mol = mol_from_smiles(value)
        if mol is None:
            unparsed.append(value)
        record: dict[str, Any] = {SMILES_COLUMN: value}
        record.update(
            {f"{RDKIT_PREFIX}{name}": val for name, val in rdkit_descriptor_row(mol).items()}
        )
        record.update(
            {
                f"{HAND_CRAFTED_PREFIX}{name}": val
                for name, val in hand_crafted_descriptors(mol).items()
            }
        )
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    descriptor_columns = [column for column in frame.columns if column != SMILES_COLUMN]
    frame[descriptor_columns] = frame[descriptor_columns].astype(np.float64)

    rdkit_columns = [column for column in descriptor_columns if column.startswith(RDKIT_PREFIX)]
    hand_columns = [
        column for column in descriptor_columns if column.startswith(HAND_CRAFTED_PREFIX)
    ]
    dropped = _uninformative_columns(frame, rdkit_columns) if drop_uninformative else []
    if dropped:
        frame = frame.drop(columns=dropped)
    frame.attrs["dropped_rdkit_columns"] = dropped
    frame.attrs["constant_hand_crafted_columns"] = _uninformative_columns(frame, hand_columns)
    frame.attrs["unparsed_smiles"] = unparsed
    return frame.reset_index(drop=True)


def load_descriptor_table(path: Path | str) -> pd.DataFrame:
    """Load a descriptor parquet and validate its schema.

    Requires a unique ``canonical_smiles`` key column and only ``lig2d__``
    descriptor columns otherwise; descriptor columns are returned as float64.
    """

    frame = pd.read_parquet(Path(path).expanduser())
    if SMILES_COLUMN not in frame.columns:
        raise ValueError(f"Descriptor table is missing the {SMILES_COLUMN!r} column.")
    if frame[SMILES_COLUMN].isna().any() or frame[SMILES_COLUMN].duplicated().any():
        raise ValueError(f"{SMILES_COLUMN!r} must be non-null and unique.")
    descriptor_columns = [column for column in frame.columns if column != SMILES_COLUMN]
    foreign = [column for column in descriptor_columns if not column.startswith(DESCRIPTOR_PREFIX)]
    if foreign:
        raise ValueError(f"Unexpected non-descriptor columns: {foreign[:5]}")
    frame[SMILES_COLUMN] = frame[SMILES_COLUMN].astype(str)
    frame[descriptor_columns] = frame[descriptor_columns].astype(np.float64)
    return frame.reset_index(drop=True)


def descriptor_columns(frame: pd.DataFrame, block: str | None = None) -> list[str]:
    """Descriptor column names in ``frame``; ``block`` may be ``"rd"`` or ``"hc"``."""

    if block is None:
        prefix = DESCRIPTOR_PREFIX
    elif block == "rd":
        prefix = RDKIT_PREFIX
    elif block == "hc":
        prefix = HAND_CRAFTED_PREFIX
    else:
        raise ValueError("block must be None, 'rd' or 'hc'.")
    return [column for column in frame.columns if str(column).startswith(prefix)]

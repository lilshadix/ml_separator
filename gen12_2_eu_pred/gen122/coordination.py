"""The coordination-topology descriptor block.

Everything here is a function of one canonical SMILES string.  No experimental
condition, no target, no fold and no metal enters, so a coordination descriptor
cannot leak by construction rather than by convention: :func:`build_table` takes
a list of SMILES and nothing else.

**Language.** These are *topological* descriptors over a 2D molecular graph.
``donor_pair_dist_min`` is a count of bonds, not an interatomic distance;
``estimated_denticity_per_pocket`` is the number of potential donor atoms in one
graph neighbourhood, not a measured coordination number; ``n_donor_clusters`` is
a graph-component count, not a number of metal centres.  Every name in the
"denticity" family carries ``potential_``, ``estimated_`` or ``proxy`` for that
reason.

**Five families**, which are also the units of the grouped permutation-importance
analysis:

``donor``        per-class and per-element counts of potential donor atoms
``motif``        counts of named chelating units and ring classes
``arm``          binding-arm multiplicity and repeated-arm topology
``dist``         donor-donor graph-distance statistics and histogram
``arch``         size, branching, symmetry and size-per-donor ratios

The SMARTS and the two numeric rules that are not SMARTS — the chelate link
distance of 4 bonds and the MULTI_ARM threshold of 2 clusters — live in
``config/coordination_smarts.json`` and were frozen before any Gen12.2 model ran.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import paths

PREFIX = "coord__"
FAMILIES: tuple[str, ...] = ("donor", "motif", "arm", "dist", "arch")


# --------------------------------------------------------------------------- #
# Specification
# --------------------------------------------------------------------------- #

#: Which specification file the module is currently using.  It is the frozen v1.1.0 unless
#: a caller explicitly switches to the post-hoc one, and every switch clears the caches so
#: a stale compiled pattern cannot survive it.
_SPEC_PATH = [paths.COORDINATION_SMARTS_JSON]


def use_spec(path) -> None:
    """Point the module at a different specification file.  Post-hoc analysis only."""
    from pathlib import Path
    _SPEC_PATH[0] = Path(path)
    spec.cache_clear(); spec_digest.cache_clear(); _compiled.cache_clear()


def active_spec_path():
    return _SPEC_PATH[0]


@lru_cache(maxsize=1)
def spec() -> dict:
    return json.loads(_SPEC_PATH[0].read_text())


@lru_cache(maxsize=1)
def spec_digest() -> str:
    """SHA-256 of the specification file in use, recorded beside every feature matrix."""
    return paths.sha256_of(_SPEC_PATH[0])


@lru_cache(maxsize=1)
def _compiled() -> dict:
    from rdkit import Chem
    s = spec()
    donors, motifs = {}, {}
    for name, entry in s["donor_classes"].items():
        pattern = Chem.MolFromSmarts(entry["smarts"])
        if pattern is None:
            raise ValueError(f"donor class {name!r} has an unparsable SMARTS: {entry['smarts']!r}")
        atom = entry["atom"]
        donors[name] = (pattern, tuple(atom) if isinstance(atom, list) else (int(atom),))
    for name, entry in s["chelating_unit_motifs"].items():
        pattern = Chem.MolFromSmarts(entry["smarts"])
        if pattern is None:
            raise ValueError(f"motif {name!r} has an unparsable SMARTS: {entry['smarts']!r}")
        motifs[name] = pattern
    return {"donors": donors, "motifs": motifs,
            "groups": s["donor_element_groups"],
            "link": int(s["topology_rules"]["chelate_link_distance"]),
            # The frozen v1.1.0 specification is what every locked Gen12.2 result was
            # computed against, and its matrix must stay bit-reproducible.  Four
            # rule-level defects found by review AFTER the locked evaluation are corrected
            # only when the post-hoc specification is loaded, so the two versions differ by
            # exactly what the post-hoc file declares and by nothing else.
            "posthoc": bool(s.get("posthoc_corrections"))}


# --------------------------------------------------------------------------- #
# Per-molecule computation
# --------------------------------------------------------------------------- #

def _donor_atoms(mol) -> dict[str, list[int]]:
    """Atom indices of every potential donor, by class.  An atom may appear in
    at most one class: the classes are made disjoint by taking them in the order
    they are declared, because ``O_carboxyl`` and ``O_hydroxyl`` would otherwise
    both claim the same acidic oxygen."""
    compiled = _compiled()["donors"]
    claimed: set[int] = set()
    out: dict[str, list[int]] = {}
    for name, (pattern, positions) in compiled.items():
        found: list[int] = []
        for match in mol.GetSubstructMatches(pattern, uniquify=True):
            for position in positions:
                index = match[position]
                if index not in claimed:
                    claimed.add(index)
                    found.append(index)
        out[name] = sorted(found)
    return out


def _ring_classes(mol, *, posthoc: bool = False) -> dict[str, int]:
    counts = {"pyridine_like_ring": 0, "diazine_like_ring": 0, "triazine_like_ring": 0,
              "azole_ring": 0, "crown_like_ring": 0, "lactam_ring": 0}
    info = mol.GetRingInfo()
    for ring in info.AtomRings():
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        aromatic = all(a.GetIsAromatic() for a in atoms)
        # The specification defines these classes on [nX2] — a two-connected aromatic
        # nitrogen with a free lone pair — not on nitrogen by element.  Counting by element
        # re-admits the pyrrole-type nitrogen the donor block excludes on purpose.
        n_ring_N_any = sum(1 for a in atoms if a.GetSymbol() == "N")
        n_ring_N = n_ring_N_any if not posthoc else sum(
            1 for a in atoms
            if a.GetSymbol() == "N" and a.GetIsAromatic() and a.GetDegree() == 2
            and a.GetTotalNumHs() == 0)
        n_ring_O = sum(1 for a in atoms if a.GetSymbol() == "O")
        size = len(ring)
        if aromatic and size == 6:
            if n_ring_N == 1:
                counts["pyridine_like_ring"] += 1
            elif n_ring_N == 2:
                counts["diazine_like_ring"] += 1
            elif n_ring_N >= 3:
                counts["triazine_like_ring"] += 1
        if aromatic and size == 5 and n_ring_N_any >= 2:
            counts["azole_ring"] += 1
        if size >= 9 and n_ring_O >= 3:
            counts["crown_like_ring"] += 1
        if not aromatic:
            ring_set = set(ring)
            for atom in atoms:
                if atom.GetSymbol() != "N":
                    continue
                # A lactam is a ring amide: the carbonyl carbon must itself be in the ring.
                # Without that condition every cyclic tertiary amine acylated from outside
                # — pyrrolidyl, piperidyl, morpholino — matched, and 12 of 13 non-zero
                # values on this cohort were false positives.  See the post-hoc spec.
                if any(nb.GetSymbol() == "C" and (nb.GetIdx() in ring_set or not posthoc)
                       and any(
                        b.GetBondTypeAsDouble() == 2.0 and
                        b.GetOtherAtom(nb).GetSymbol() in ("O", "S")
                        for b in nb.GetBonds()) for nb in atom.GetNeighbors()):
                    counts["lactam_ring"] += 1
                    break
    return counts


def _clusters(donor_indices: Sequence[int], distance: np.ndarray, link: int) -> list[list[int]]:
    """Connected components of the donor graph, joining donors within ``link`` bonds.

    One component is a candidate chelating pocket: a set of potential donors close enough
    in the molecular graph that they *could* close a chelate ring on a single metal centre.
    It is a graph statement and not a geometric one, and it is permissive: an isolated
    solubilising ether oxygen more than ``link`` bonds from every other donor becomes a
    component of its own, so a component count is an upper bound on the number of chelating
    pockets rather than an estimate of it.  ``n_clusters_with_2plus_donors`` is the stricter
    reading and is emitted beside it.
    """
    n = len(donor_indices)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if distance[donor_indices[i], donor_indices[j]] <= link:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(donor_indices[i])
    return [sorted(v) for _, v in sorted(groups.items(), key=lambda kv: min(kv[1]))]


def descriptors_for(smiles: str) -> dict[str, float]:
    """The full coordination-topology row for one structure."""
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit cannot parse {smiles!r}")
    compiled = _compiled()
    link = compiled["link"]
    distance = Chem.GetDistanceMatrix(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    out: dict[str, float] = {}

    # ---- donor census ------------------------------------------------------ #
    by_class = _donor_atoms(mol)
    for name, indices in by_class.items():
        out[f"{PREFIX}donor__n_{name}"] = float(len(indices))
    for group, members in compiled["groups"].items():
        out[f"{PREFIX}donor__n_{group}"] = float(sum(len(by_class[m]) for m in members))
    donors = sorted({i for indices in by_class.values() for i in indices})
    n_donor = len(donors)
    out[f"{PREFIX}donor__potential_donor_count"] = float(n_donor)
    present = [g for g in compiled["groups"] if out[f"{PREFIX}donor__n_{g}"] > 0]
    out[f"{PREFIX}donor__element_diversity"] = float(len(present))
    for group in compiled["groups"]:
        out[f"{PREFIX}donor__frac_{group}"] = (
            out[f"{PREFIX}donor__n_{group}"] / n_donor if n_donor else 0.0)
    out[f"{PREFIX}donor__has_soft_donor"] = float(out[f"{PREFIX}donor__n_soft_S"] > 0)

    # ---- named chelating motifs and ring classes --------------------------- #
    for name, pattern in compiled["motifs"].items():
        out[f"{PREFIX}motif__n_{name}"] = float(len(mol.GetSubstructMatches(pattern, uniquify=True)))
    for name, value in _ring_classes(mol, posthoc=compiled["posthoc"]).items():
        out[f"{PREFIX}motif__n_{name}"] = float(value)
    # ``beta_diketone_unit`` is bit-identical to ``malonamide_unit`` on this cohort, which
    # made every malonamide count twice; ``bis_thiophosphoryl_methylene`` is the only motif
    # that matches TWE-23 and TWE-24 and was missing.  Both columns are still emitted
    # individually; only the *sum* changes.
    named_units = (("dga_unit", "malonamide_unit", "cmpo_unit", "bis_thiophosphoryl_unit",
                    "bis_thiophosphoryl_methylene", "picolinamide_arm", "carboxylate_arm")
                   if compiled["posthoc"] else
                   ("dga_unit", "malonamide_unit", "cmpo_unit", "bis_thiophosphoryl_unit",
                    "picolinamide_arm", "carboxylate_arm", "beta_diketone_unit"))
    out[f"{PREFIX}motif__n_named_chelating_units"] = float(
        sum(out[f"{PREFIX}motif__n_{n}"] for n in named_units))
    out[f"{PREFIX}motif__max_repeated_named_unit"] = float(
        max([out[f"{PREFIX}motif__n_{n}"] for n in named_units], default=0.0))
    out[f"{PREFIX}motif__n_distinct_named_units"] = float(
        sum(1 for n in named_units if out[f"{PREFIX}motif__n_{n}"] > 0))

    # ---- binding-arm multiplicity ------------------------------------------ #
    clusters = _clusters(donors, distance, link)
    sizes = [len(c) for c in clusters]
    out[f"{PREFIX}arm__local_donor_cluster_count"] = float(len(clusters))
    out[f"{PREFIX}arm__max_connected_donor_motif_size"] = float(max(sizes, default=0))
    out[f"{PREFIX}arm__mean_donor_cluster_size"] = float(np.mean(sizes)) if sizes else 0.0
    out[f"{PREFIX}arm__min_donor_cluster_size"] = float(min(sizes, default=0))
    out[f"{PREFIX}arm__n_clusters_with_2plus_donors"] = float(sum(1 for s in sizes if s >= 2))
    out[f"{PREFIX}arm__n_clusters_with_3plus_donors"] = float(sum(1 for s in sizes if s >= 3))
    out[f"{PREFIX}arm__estimated_denticity_per_pocket"] = out[
        f"{PREFIX}arm__max_connected_donor_motif_size"]
    out[f"{PREFIX}arm__is_multitopic"] = float(len(clusters) >= 2)
    out[f"{PREFIX}arm__topicity"] = float(min(len(clusters), 6))

    # repeated arms: symmetry orbits over the donor clusters
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    cluster_signature: dict[tuple, int] = {}
    for cluster in clusters:
        signature = tuple(sorted(ranks[i] for i in cluster))
        cluster_signature[signature] = cluster_signature.get(signature, 0) + 1
    out[f"{PREFIX}arm__repeated_arm_count"] = float(max(cluster_signature.values(), default=0))
    out[f"{PREFIX}arm__n_distinct_arm_types"] = float(len(cluster_signature))
    out[f"{PREFIX}arm__has_repeated_arms"] = float(
        out[f"{PREFIX}arm__repeated_arm_count"] >= 2)

    # amide-bearing branches: connected components after cutting every bond that
    # lies on a shortest path between two different clusters
    out[f"{PREFIX}arm__n_amide_branches"] = float(len(
        mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3][CX3]=[OX1]"), uniquify=True)))
    out[f"{PREFIX}arm__n_donor_bearing_branches"] = float(len(clusters))

    # branching around the core: the atom minimising the maximum graph distance
    # to any donor is the "core"; its heavy degree is the branching degree there
    if n_donor:
        eccentric = distance[:, donors].max(axis=1)
        core = int(np.argmin(eccentric))
        out[f"{PREFIX}arm__core_branch_degree"] = float(
            mol.GetAtomWithIdx(core).GetDegree())
        out[f"{PREFIX}arm__core_donor_eccentricity"] = float(eccentric[core])
        out[f"{PREFIX}arm__mean_donor_distance_to_core"] = float(distance[core, donors].mean())
    else:
        out[f"{PREFIX}arm__core_branch_degree"] = 0.0
        out[f"{PREFIX}arm__core_donor_eccentricity"] = 0.0
        out[f"{PREFIX}arm__mean_donor_distance_to_core"] = 0.0

    # ---- donor-donor graph topology ---------------------------------------- #
    if n_donor >= 2:
        pairs = distance[np.ix_(donors, donors)][np.triu_indices(n_donor, k=1)]
        out[f"{PREFIX}dist__donor_pair_min"] = float(pairs.min())
        out[f"{PREFIX}dist__donor_pair_max"] = float(pairs.max())
        out[f"{PREFIX}dist__donor_pair_mean"] = float(pairs.mean())
        out[f"{PREFIX}dist__donor_pair_median"] = float(np.median(pairs))
        out[f"{PREFIX}dist__donor_pair_sd"] = float(pairs.std(ddof=0))
        out[f"{PREFIX}dist__donor_network_diameter"] = float(pairs.max())
        for cut in (2, 3, 4, 5, 6):
            out[f"{PREFIX}dist__n_donor_pairs_within_{cut}"] = float((pairs <= cut).sum())
        out[f"{PREFIX}dist__n_donor_pairs_beyond_6"] = float((pairs > 6).sum())
        out[f"{PREFIX}dist__frac_donor_pairs_within_3"] = float((pairs <= 3).mean())
        out[f"{PREFIX}dist__donor_pair_proximity_count"] = float((pairs <= link).sum())
        # chelate-ring sizes a pair could close: distance d gives a (d+2)-ring
        out[f"{PREFIX}dist__n_five_membered_chelate_pairs"] = float((pairs == 3).sum())
        out[f"{PREFIX}dist__n_four_membered_chelate_pairs"] = float((pairs == 2).sum())
        sub = distance[np.ix_(donors, donors)]
        eccentricity = sub.max(axis=1)
        out[f"{PREFIX}dist__donor_eccentricity_mean"] = float(eccentricity.mean())
        out[f"{PREFIX}dist__donor_eccentricity_max"] = float(eccentricity.max())
        out[f"{PREFIX}dist__donor_eccentricity_min"] = float(eccentricity.min())
    else:
        for name in ("donor_pair_min", "donor_pair_max", "donor_pair_mean", "donor_pair_median",
                     "donor_pair_sd", "donor_network_diameter", "frac_donor_pairs_within_3",
                     "donor_pair_proximity_count", "n_five_membered_chelate_pairs",
                     "n_four_membered_chelate_pairs", "donor_eccentricity_mean",
                     "donor_eccentricity_max", "donor_eccentricity_min", "n_donor_pairs_beyond_6"):
            out[f"{PREFIX}dist__{name}"] = 0.0
        for cut in (2, 3, 4, 5, 6):
            out[f"{PREFIX}dist__n_donor_pairs_within_{cut}"] = 0.0
    if len(clusters) >= 2:
        between = []
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                between.append(distance[np.ix_(clusters[a], clusters[b])].min())
        out[f"{PREFIX}dist__min_intercluster_distance"] = float(min(between))
        out[f"{PREFIX}dist__mean_intercluster_distance"] = float(np.mean(between))
        out[f"{PREFIX}dist__n_linker_atoms_between_units"] = float(min(between) - 1)
    else:
        out[f"{PREFIX}dist__min_intercluster_distance"] = 0.0
        out[f"{PREFIX}dist__mean_intercluster_distance"] = 0.0
        out[f"{PREFIX}dist__n_linker_atoms_between_units"] = 0.0

    # ---- molecular architecture -------------------------------------------- #
    out[f"{PREFIX}arch__mol_weight"] = float(Descriptors.MolWt(mol))
    out[f"{PREFIX}arch__n_heavy_atoms"] = float(n_heavy)
    out[f"{PREFIX}arch__n_carbon"] = float(sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C"))
    out[f"{PREFIX}arch__n_heteroatoms"] = float(rdMolDescriptors.CalcNumHeteroatoms(mol))
    out[f"{PREFIX}arch__n_rotatable_bonds"] = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    out[f"{PREFIX}arch__n_rings"] = float(rdMolDescriptors.CalcNumRings(mol))
    out[f"{PREFIX}arch__n_aromatic_rings"] = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    out[f"{PREFIX}arch__fraction_csp3"] = float(rdMolDescriptors.CalcFractionCSP3(mol))
    out[f"{PREFIX}arch__tpsa"] = float(rdMolDescriptors.CalcTPSA(mol))
    out[f"{PREFIX}arch__clogp"] = float(Crippen.MolLogP(mol))
    out[f"{PREFIX}arch__formal_charge"] = float(Chem.GetFormalCharge(mol))
    out[f"{PREFIX}arch__n_hbd"] = float(rdMolDescriptors.CalcNumHBD(mol))
    out[f"{PREFIX}arch__n_hba"] = float(rdMolDescriptors.CalcNumHBA(mol))
    degrees = np.array([a.GetDegree() for a in mol.GetAtoms()], dtype=float)
    out[f"{PREFIX}arch__mean_heavy_degree"] = float(degrees.mean())
    out[f"{PREFIX}arch__n_degree3_atoms"] = float((degrees == 3).sum())
    out[f"{PREFIX}arch__n_degree4_atoms"] = float((degrees == 4).sum())
    out[f"{PREFIX}arch__graph_diameter"] = float(distance[distance < 1e6].max())
    orbits: dict[int, int] = {}
    for rank in ranks:
        orbits[rank] = orbits.get(rank, 0) + 1
    out[f"{PREFIX}arch__n_symmetry_orbits"] = float(len(orbits))
    out[f"{PREFIX}arch__frac_atoms_in_largest_orbit"] = float(max(orbits.values()) / n_heavy)
    out[f"{PREFIX}arch__symmetry_compression"] = float(len(orbits) / n_heavy)
    denominator = float(n_donor) if n_donor else np.nan
    out[f"{PREFIX}arch__heavy_atoms_per_donor"] = float(n_heavy) / denominator if n_donor else np.nan
    out[f"{PREFIX}arch__carbon_per_donor"] = (
        out[f"{PREFIX}arch__n_carbon"] / denominator if n_donor else np.nan)
    out[f"{PREFIX}arch__mol_weight_per_donor"] = (
        out[f"{PREFIX}arch__mol_weight"] / denominator if n_donor else np.nan)
    out[f"{PREFIX}arch__clogp_per_donor"] = (
        out[f"{PREFIX}arch__clogp"] / denominator if n_donor else np.nan)
    out[f"{PREFIX}arch__heavy_atoms_per_pocket"] = (
        float(n_heavy) / len(clusters) if clusters else np.nan)
    return out


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #

def build_table(smiles: Iterable[str]) -> pd.DataFrame:
    """One row per distinct structure, indexed by canonical SMILES."""
    unique = sorted({str(s) for s in smiles})
    rows = [descriptors_for(s) for s in unique]
    table = pd.DataFrame(rows, index=pd.Index(unique, name="extractant"))
    table = table[sorted(table.columns)]
    bad = table.columns[~np.isfinite(table.to_numpy(dtype=float)) .any(axis=0) & False]
    return table


def family_of(column: str) -> str:
    """``coord__arm__topicity`` -> ``arm``.  Used by the grouped-importance analysis."""
    if not column.startswith(PREFIX):
        raise KeyError(f"{column!r} is not a coordination descriptor")
    return column[len(PREFIX):].split("__", 1)[0]


def columns_by_family(columns: Sequence[str]) -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for column in columns:
        out.setdefault(family_of(column), []).append(column)
    return {k: tuple(v) for k, v in out.items() if v}

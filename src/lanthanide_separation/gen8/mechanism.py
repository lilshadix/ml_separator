"""A compact, mechanistic ligand representation for gen8 (brief section 12).

Why this module exists
----------------------
Everything the project has tried so far describes an extractant either with a
2048-bit ECFP fingerprint or with a 206-column RDKit descriptor table.  Both are
*generic* molecular representations: they are optimised for "are these two
molecules similar" and know nothing about solvent extraction.  The measured
verdict on them is bad -- the 206-column ``lig2d_ext`` block is the only feature
block whose *removal* is a significant improvement, and a no-ligand model is
within ~0.09 log units of the best ligand-aware one.

The hypothesis this module encodes is that the small amount of ligand signal
that does exist is *coordination chemistry*, not molecular similarity: which
atoms can donate a lone pair to a hard Ln(III) centre, how hard or soft they
are, how many of them can wrap the metal at once, how far apart they sit along
the chain, and whether the ligand works by solvation (neutral) or by cation
exchange (acidic/ionic).  That is a few dozen numbers, not a few thousand bits.

Relation to the existing ``DONORS`` block
-----------------------------------------
``levels._attach_donor_census`` already emits a 13-column census
(``donor__O(amide_carbonyl)``, ``donor__O(ether)``, ``donor__N(aromatic)``,
``donor__N(amine)``, ``donor__S(donor)``, ``donor__O(hydroxyl)``,
``donor__O(ester_carbonyl)``, ``donor__O(carbonyl)``, ``donor__n_total``,
plus ``DENTATE``/``coreCN``/``n_ligs``/``n_fill`` from the 3D bundle).  That
census is a bag of counts read out of a precomputed ``DONOR_TYPES`` list.  It
has no notion of

* **phosphorus chemistry** -- P=O vs P=S is the single largest known lever on
  Ln/An selectivity and the census lumps every S into one ``S(donor)`` bucket
  and has no phosphoryl class at all;
* **softness** -- a hard carboxylate O and a soft thiophosphoryl S are two
  independent counts with no ordering between them;
* **spacing / chelate geometry** -- how far apart the donors are along the
  molecular graph, which is what decides whether they can chelate;
* **charge class** -- neutral solvating extractant vs acidic cation exchanger;
* **rigidity** -- a preorganised aromatic backbone vs a floppy chain.

This module recomputes the donor classification itself, from SMILES, with a
finer and disjoint class vocabulary, and adds exactly those five missing axes.
It is deliberately capped at a few dozen columns.

Contract
--------
* deterministic (pure function of the SMILES string, no RNG, no ordering
  dependence),
* never reads or touches ``log_D`` or any experimental quantity,
* one output row per input SMILES, in the same order,
* an unparseable SMILES yields an all-NaN row; the number of such rows is
  recorded on ``frame.attrs["n_unparsed"]`` (see :func:`n_unparsed`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, rdMolDescriptors


# --------------------------------------------------------------------------
# donor classes
# --------------------------------------------------------------------------
#: Disjoint donor-atom classes.  Every coordinating heteroatom is assigned to
#: exactly one of these, so the class counts are additive.
_O_CLASSES: tuple[str, ...] = (
    "amide_O",       # O=C-N          amide / DGA carbonyl -- the workhorse donor
    "ester_O",       # O=C-O-C        ester carbonyl
    "ketone_O",      # O=C            everything else carbonyl
    "ether_O",       # C-O-C, P-O-C   the DGA/crown central O and phosphate esters
    "hydroxyl_O",    # -OH            alcohol / phenol / P-OH / S-OH
    "phosphoryl_O",  # O=P            TBP, CMPO, phosphine oxide
    "carboxyl_O",    # -C(=O)OH/O(-)  hard, anionic when deprotonated
    "oxide_O",       # N-oxide, S=O   dative oxide, hard
)
_N_CLASSES: tuple[str, ...] = (
    "aromatic_N",    # pyridine/triazine/tetrazole ring N -- softer, the BTBP motif
    "amine_N",       # sp3 amine
)
_S_CLASSES: tuple[str, ...] = (
    "thioether_S",       # C-S-C, thiol
    "thiophosphoryl_S",  # S=P   the soft An-selective analogue of P=O
    "thiocarbonyl_S",    # S=C   thioamide
)
_DONOR_CLASSES: tuple[str, ...] = _O_CLASSES + _N_CLASSES + _S_CLASSES

#: HSAB softness on a 0 (hardest, carboxylate O) to 1 (softest, P=S) scale.
#: Values are an ordering of donor classes by their Ln/An discrimination, not
#: measured absolute softness parameters; only the ordering is load-bearing.
_SOFTNESS: dict[str, float] = {
    "carboxyl_O": 0.00,
    "hydroxyl_O": 0.05,
    "ether_O": 0.10,
    "oxide_O": 0.12,
    "phosphoryl_O": 0.15,
    "amide_O": 0.20,
    "ester_O": 0.20,
    "ketone_O": 0.22,
    "amine_N": 0.50,
    "aromatic_N": 0.60,
    "thioether_S": 0.90,
    "thiocarbonyl_S": 0.95,
    "thiophosphoryl_S": 1.00,
}

#: A chelate ring of 4-7 members puts its two donors 2-5 bonds apart on the
#: molecular graph.  Donor pairs in that window can bite the same metal.
_CHELATE_MIN_DIST = 2
_CHELATE_MAX_DIST = 5


MECHANISM_COLUMNS: tuple[str, ...] = tuple(
    f"mech__n_{c}" for c in _DONOR_CLASSES
) + (
    # element-level aggregates
    "mech__n_O_donor",
    "mech__n_N_donor",
    "mech__n_S_donor",
    "mech__n_P",
    "mech__n_donor_total",
    "mech__frac_O_donor",
    "mech__frac_N_donor",
    "mech__frac_S_donor",
    # HSAB
    "mech__softness_mean",
    # charge / extraction mechanism class
    "mech__formal_charge",
    "mech__n_acidic_H",
    "mech__is_neutral_extractant",
    # denticity / donor spacing
    "mech__denticity_proxy",
    "mech__n_chelate_pairs",
    "mech__donor_dist_min",
    "mech__donor_dist_mean",
    # rigidity / preorganisation
    "mech__ring_count",
    "mech__frac_rotatable",
    # size and bulk physicochemistry
    "mech__heavy_atoms",
    "mech__crippen_logp",
    "mech__tpsa",
    "mech__hbd",
    "mech__hba",
)


# --------------------------------------------------------------------------
# per-atom classification
# --------------------------------------------------------------------------
def _neighbour_symbols(atom: Chem.Atom) -> list[str]:
    return [nb.GetSymbol() for nb in atom.GetNeighbors()]


def _double_bonded_to(atom: Chem.Atom, symbol: str) -> bool:
    for bond in atom.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetOtherAtom(atom).GetSymbol() == symbol:
            return True
    return False


def _double_bond_partner(atom: Chem.Atom, symbol: str) -> Chem.Atom | None:
    for bond in atom.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        other = bond.GetOtherAtom(atom)
        if other.GetSymbol() == symbol:
            return other
    return None


def _is_carboxyl_carbon(carbon: Chem.Atom) -> bool:
    """C(=O)[OH] or C(=O)[O-] -- a carboxylic acid or carboxylate carbon."""
    if carbon.GetSymbol() != "C":
        return False
    has_carbonyl = False
    has_acidic = False
    for bond in carbon.GetBonds():
        other = bond.GetOtherAtom(carbon)
        if other.GetSymbol() != "O":
            continue
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            has_carbonyl = True
        elif bond.GetBondType() == Chem.BondType.SINGLE and (
            other.GetTotalNumHs() > 0 or other.GetFormalCharge() < 0
        ):
            has_acidic = True
    return has_carbonyl and has_acidic


def _classify_oxygen(atom: Chem.Atom) -> str | None:
    # carboxyl first: both oxygens of the -COOH / -COO(-) group belong to it
    for nb in atom.GetNeighbors():
        if _is_carboxyl_carbon(nb):
            return "carboxyl_O"
    if _double_bonded_to(atom, "P"):
        return "phosphoryl_O"
    if _double_bonded_to(atom, "S") or (
        atom.GetFormalCharge() < 0 and any(s == "N" for s in _neighbour_symbols(atom))
    ):
        # sulfoxide / sulfone O, and the amine- or pyridine-N-oxide O
        return "oxide_O"
    carbon = _double_bond_partner(atom, "C")
    if carbon is not None:
        symbols = _neighbour_symbols(carbon)
        if "N" in symbols:
            return "amide_O"
        if symbols.count("O") >= 2:
            return "ester_O"
        return "ketone_O"
    if atom.GetTotalNumHs() > 0:
        return "hydroxyl_O"
    if atom.GetDegree() == 2:
        # C-O-C, C-O-P, P-O-P: an ether-type bridging oxygen
        return "ether_O"
    return None


def _classify_nitrogen(atom: Chem.Atom) -> str | None:
    if atom.GetIsAromatic():
        if atom.GetFormalCharge() > 0:
            return None  # N-oxide / quaternised aromatic N is not a donor
        # Only a pyridine-type N coordinates: two ring connections and no
        # substituent on N, so its lone pair sits in the ring plane.  A
        # pyrrole-type N puts that pair into the aromatic sextet and cannot
        # donate -- and it is pyrrole-type whether the third connection is an
        # H (pyrrole, NH-azoles) or a carbon (N-alkyl pyrazole/triazole,
        # carbazole).  Keying only on the H, as an earlier version did,
        # silently counted every N-substituted azole N as a donor.
        if atom.GetDegree() != 2 or atom.GetTotalNumHs() > 0:
            return None
        return "aromatic_N"
    if atom.GetFormalCharge() > 0:
        return None  # ammonium / quaternary ammonium: no lone pair
    for nb in atom.GetNeighbors():
        # amide, thioamide, sulfonamide, phosphoramide: the lone pair is
        # delocalised into the adjacent pi system and does not coordinate
        if nb.GetSymbol() == "C" and (
            _double_bonded_to(nb, "O") or _double_bonded_to(nb, "S")
        ):
            return None
        if nb.GetSymbol() in {"P", "S"}:
            return None
    return "amine_N"


def _classify_sulfur(atom: Chem.Atom) -> str | None:
    if _double_bonded_to(atom, "P"):
        return "thiophosphoryl_S"
    if _double_bonded_to(atom, "C"):
        return "thiocarbonyl_S"
    if _double_bonded_to(atom, "O"):
        return None  # sulfonyl/sulfoxide S is not the donor; its O is
    if atom.GetDegree() <= 2:
        return "thioether_S"  # includes thiophene S and thiols
    return None


def _classify_atom(atom: Chem.Atom) -> str | None:
    symbol = atom.GetSymbol()
    if symbol == "O":
        return _classify_oxygen(atom)
    if symbol == "N":
        return _classify_nitrogen(atom)
    if symbol == "S":
        return _classify_sulfur(atom)
    return None


def _count_acidic_hydrogens(mol: Chem.Mol) -> int:
    """Protons an extractant can exchange for a metal cation.

    Carboxylic acid, phosphonic/phosphinic/phosphoric P-OH, sulfonic S-OH and
    thiol S-H.  This is what separates a cation-exchange extractant (HDEHP,
    a diglycolamic acid) from a neutral solvating one (TODGA, TBP, CMPO).
    """
    total = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "O" and atom.GetTotalNumHs() > 0:
            neighbours = _neighbour_symbols(atom)
            if "P" in neighbours or "S" in neighbours:
                total += atom.GetTotalNumHs()
            elif any(_is_carboxyl_carbon(nb) for nb in atom.GetNeighbors()):
                total += atom.GetTotalNumHs()
        elif atom.GetSymbol() == "S" and atom.GetTotalNumHs() > 0:
            total += atom.GetTotalNumHs()
        elif atom.GetFormalCharge() < 0 and atom.GetSymbol() in {"O", "S"}:
            # already-deprotonated site: same mechanism, counted the same way
            if any(_is_carboxyl_carbon(nb) for nb in atom.GetNeighbors()) or any(
                s in {"P", "S"} for s in _neighbour_symbols(atom)
            ):
                total += 1
    return int(total)


def _has_permanent_ionic_site(mol: Chem.Mol) -> bool:
    """True for salts and permanently charged extractants (e.g. Aliquat 336).

    A quaternary ammonium, a counter-ion fragment, or a non-zero net charge all
    mean the ligand works by anion exchange rather than by neutral solvation.
    A zwitterionic N-oxide (aromatic ``[n+]``-``[O-]``) is deliberately NOT
    counted: those extractants are neutral solvating donors.
    """
    if Chem.GetFormalCharge(mol) != 0:
        return True
    if len(Chem.GetMolFrags(mol)) > 1:
        return True
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "N" and atom.GetFormalCharge() > 0 and atom.GetDegree() == 4:
            return True
    return False


def _features_for_mol(mol: Chem.Mol) -> dict[str, float]:
    counts = dict.fromkeys(_DONOR_CLASSES, 0.0)
    donor_idx: list[int] = []
    donor_cls: list[str] = []
    for atom in mol.GetAtoms():
        cls = _classify_atom(atom)
        if cls is None:
            continue
        counts[cls] += 1.0
        donor_idx.append(atom.GetIdx())
        donor_cls.append(cls)

    n_o = sum(counts[c] for c in _O_CLASSES)
    n_n = sum(counts[c] for c in _N_CLASSES)
    n_s = sum(counts[c] for c in _S_CLASSES)
    n_p = float(sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "P"))
    n_tot = n_o + n_n + n_s

    softness = (
        float(np.mean([_SOFTNESS[c] for c in donor_cls])) if donor_cls else 0.0
    )

    # --- donor spacing / denticity -------------------------------------
    if len(donor_idx) >= 2:
        dist = Chem.GetDistanceMatrix(mol)
        pairs = [
            dist[i, j]
            for a, i in enumerate(donor_idx)
            for j in donor_idx[a + 1:]
        ]
        pairs_arr = np.asarray(pairs, dtype=float)
        finite = pairs_arr[np.isfinite(pairs_arr)]
        dist_min = float(finite.min()) if finite.size else 0.0
        dist_mean = float(finite.mean()) if finite.size else 0.0
        chelating = np.zeros(len(donor_idx), dtype=bool)
        n_chelate = 0
        for a, i in enumerate(donor_idx):
            for b in range(a + 1, len(donor_idx)):
                d = dist[i, donor_idx[b]]
                if _CHELATE_MIN_DIST <= d <= _CHELATE_MAX_DIST:
                    n_chelate += 1
                    chelating[a] = True
                    chelating[b] = True
        denticity = float(chelating.sum())
    else:
        dist_min = 0.0
        dist_mean = 0.0
        n_chelate = 0
        denticity = float(len(donor_idx))

    n_rot = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    n_bonds = float(mol.GetNumBonds())
    charge = float(Chem.GetFormalCharge(mol))
    acidic = float(_count_acidic_hydrogens(mol))

    return {
        **{f"mech__n_{c}": counts[c] for c in _DONOR_CLASSES},
        "mech__n_O_donor": n_o,
        "mech__n_N_donor": n_n,
        "mech__n_S_donor": n_s,
        "mech__n_P": n_p,
        "mech__n_donor_total": n_tot,
        "mech__frac_O_donor": n_o / n_tot if n_tot else 0.0,
        "mech__frac_N_donor": n_n / n_tot if n_tot else 0.0,
        "mech__frac_S_donor": n_s / n_tot if n_tot else 0.0,
        "mech__softness_mean": softness,
        "mech__formal_charge": charge,
        "mech__n_acidic_H": acidic,
        "mech__is_neutral_extractant": float(
            acidic == 0.0 and not _has_permanent_ionic_site(mol)
        ),
        "mech__denticity_proxy": denticity,
        "mech__n_chelate_pairs": float(n_chelate),
        "mech__donor_dist_min": dist_min,
        "mech__donor_dist_mean": dist_mean,
        "mech__ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
        "mech__frac_rotatable": n_rot / n_bonds if n_bonds else 0.0,
        "mech__heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "mech__crippen_logp": float(Crippen.MolLogP(mol)),
        "mech__tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "mech__hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "mech__hba": float(rdMolDescriptors.CalcNumHBA(mol)),
    }


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def mechanism_features(smiles: Sequence[str]) -> pd.DataFrame:
    """Featurise ``smiles`` into :data:`MECHANISM_COLUMNS`.

    Returns an index-free frame with one row per input SMILES, in the SAME
    order as the input.  A SMILES RDKit cannot parse (or a null) produces an
    all-NaN row; the count is stored on ``frame.attrs["n_unparsed"]``.
    """
    rows: list[dict[str, float]] = []
    nan_row = dict.fromkeys(MECHANISM_COLUMNS, float("nan"))
    n_failed = 0
    # RDKit prints parse failures to stderr; we report them as a count instead.
    # The suppression is scoped to this loop -- disabling the log at import
    # time would silence RDKit for every other module in the process too.
    with rdBase.BlockLogs():
        for raw in smiles:
            mol = None
            if isinstance(raw, str) and raw:
                mol = Chem.MolFromSmiles(raw)
            if mol is None:
                n_failed += 1
                rows.append(dict(nan_row))
                continue
            rows.append(_features_for_mol(mol))
    frame = pd.DataFrame(rows, columns=list(MECHANISM_COLUMNS), dtype=float)
    frame = frame.reset_index(drop=True)
    frame.attrs["n_unparsed"] = n_failed
    return frame


def n_unparsed(frame: pd.DataFrame) -> int:
    """Number of SMILES that failed to parse in the call that built ``frame``."""
    return int(frame.attrs.get("n_unparsed", 0))


def mechanism_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance between two standardised feature matrices.

    ``A`` is ``(n, d)``, ``B`` is ``(m, d)``; the result is ``(n, m)``.  The
    inputs are expected to be already standardised (z-scored on a common
    reference), because an unstandardised mechanism matrix is dominated by
    ``mech__heavy_atoms``.

    NaNs are tolerated: a pair's distance is computed on the dimensions both
    rows have and rescaled by ``sqrt(d / d_shared)`` so it stays comparable to
    a fully observed pair.  A pair with no shared dimension is NaN.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"dimension mismatch: A has {A.shape[1]}, B has {B.shape[1]}")
    d = A.shape[1]
    a_ok = np.isfinite(A)
    b_ok = np.isfinite(B)
    a_filled = np.where(a_ok, A, 0.0)
    b_filled = np.where(b_ok, B, 0.0)
    # squared distance restricted to jointly observed dimensions
    sq = (
        (a_filled ** 2 * a_ok) @ b_ok.T.astype(float)
        + a_ok.astype(float) @ (b_filled ** 2 * b_ok).T
        - 2.0 * a_filled @ b_filled.T
    )
    shared = a_ok.astype(float) @ b_ok.T.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sqrt(np.clip(sq, 0.0, None) * (d / shared))
    out[shared == 0] = np.nan
    return out


__all__ = [
    "MECHANISM_COLUMNS",
    "mechanism_features",
    "mechanism_distance",
    "n_unparsed",
]

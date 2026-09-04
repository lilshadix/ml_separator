"""Features aimed at the *level*, from the chemistry of what sets a distribution ratio.

Everything the cohort carries describes the ligand in isolation: a fingerprint, a
descriptor table, a donor census.  None of it says anything about the quantity that
actually determines how far a metal moves into the organic phase — how well the
neutral complex prefers *that organic phase* over water.  That is a property of a
pair (ligand, diluent), and no column in the bundle is a function of both.

The mass-action expression the MASSACTION block already encodes,

.. code-block:: text

    log D = log K_ex + n log[L] + 3 log[NO3-]

hides the whole level in ``log K_ex``, and ``log K_ex`` factorises roughly as

.. code-block:: text

    log K_ex  ~  (complexation strength)  +  (partition of the complex)

The first term tracks donor identity, denticity and the chelate effect; the second
tracks lipophilicity — of the complex, not the free ligand, so it scales with the
number of ligands wrapped around the metal.  This module supplies both, plus the
cross terms:

``ligphys__logp_x_dentate``      lipophilicity weighted by how many donors bind
``ligphys__logp_per_donor``      lipophilicity per donor: chain length against head group
``ligphys__complex_logp``        ``n_ligs * MolLogP`` — the partitioning species
``ligphys__logp_minus_solvent``  **the coupling term**: ligand logP relative to the
                                 diluent's own logP.  A lipophilic extractant gains
                                 far less in 1-octanol than in dodecane, and nothing
                                 in the current feature set can express that.
``ligphys__hansen_distance``     Hansen distance between ligand and diluent, the
                                 standard "like dissolves like" metric
``ligphys__donor_hardness``      HSAB-flavoured donor score: hard O donors (amide,
                                 ether, hydroxyl) score high, soft S and aromatic N
                                 low.  Ln(III) is a hard acid, so this is the
                                 direction of the expected effect and is stated in
                                 advance rather than fitted.
``ligphys__chelate_order``       ``log10(dentate)`` — the chelate effect is entropic
                                 and enters the free energy logarithmically

Everything here is a deterministic function of the SMILES and the diluent, both of
which are known before an experiment is run, so the block is safe at inference and
carries no target information whatsoever.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

#: HSAB-flavoured donor hardness toward Ln(III), a hard trivalent acid.
#: Higher = harder = better matched.  Scale is ordinal, not thermodynamic.
DONOR_HARDNESS: Mapping[str, float] = {
    "donor__O(amide_carbonyl)": 1.00,
    "donor__O(carbonyl)": 0.95,
    "donor__O(ester_carbonyl)": 0.90,
    "donor__O(hydroxyl)": 0.85,
    "donor__O(ether)": 0.60,
    "donor__N(amine)": 0.40,
    "donor__N(aromatic)": 0.25,
    "donor__S(donor)": 0.05,
}

#: Hansen parameters of the *ligand* are not in the bundle; the diluent's are
#: supplied by :mod:`gen7.recovered`.  A ligand proxy is built from MolLogP and TPSA,
#: which is crude and is labelled as such — the point is the *distance*, and both
#: sides move consistently.
LIGAND_HANSEN_FROM = ("MolLogP", "TPSA", "MolWt")


def attach_ligand_physics(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add the ``ligphys__*`` block; returns the frame and the column names."""
    out: dict[str, np.ndarray] = {}

    def column(name: str) -> np.ndarray:
        if name not in frame.columns:
            return np.full(len(frame), np.nan)
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)

    logp = column("MolLogP")
    tpsa = column("TPSA")
    molwt = column("MolWt")
    dentate = column("DENTATE")
    n_ligs = column("n_ligs")
    core_cn = column("coreCN")
    donors_total = column("donor__n_total")

    with np.errstate(divide="ignore", invalid="ignore"):
        out["ligphys__logp_x_dentate"] = logp * dentate
        out["ligphys__logp_per_donor"] = np.where(donors_total > 0, logp / donors_total, np.nan)
        out["ligphys__complex_logp"] = logp * np.where(n_ligs > 0, n_ligs, np.nan)
        out["ligphys__tpsa_per_donor"] = np.where(donors_total > 0, tpsa / donors_total, np.nan)
        out["ligphys__logp_density"] = np.where(molwt > 0, logp / (molwt / 100.0), np.nan)
        out["ligphys__chelate_order"] = np.log10(np.where(dentate > 0, dentate, np.nan))
        out["ligphys__donor_saturation"] = np.where(
            core_cn > 0, donors_total * np.where(n_ligs > 0, n_ligs, 1.0) / core_cn, np.nan)

    # donor hardness: weighted mean and total over the census
    weights = np.zeros(len(frame))
    counts = np.zeros(len(frame))
    for name, hardness in DONOR_HARDNESS.items():
        values = column(name)
        values = np.where(np.isfinite(values), values, 0.0)
        weights += hardness * values
        counts += values
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ligphys__donor_hardness_mean"] = np.where(counts > 0, weights / counts, np.nan)
    out["ligphys__donor_hardness_total"] = weights

    # --- the coupling terms: ligand against the diluent it is dissolved in ---- #
    solvent_logp = column("rec__solvent_logp")
    solvent_eps = column("rec__solvent_eps")
    solvent_dD = column("rec__solvent_dD")
    solvent_dP = column("rec__solvent_dP")
    solvent_dH = column("rec__solvent_dH")
    if np.isfinite(solvent_logp).any():
        out["ligphys__logp_minus_solvent"] = logp - solvent_logp
        out["ligphys__complex_logp_minus_solvent"] = (
            logp * np.where(n_ligs > 0, n_ligs, 1.0) - solvent_logp)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["ligphys__logp_over_solvent_eps"] = np.where(
                solvent_eps > 0, logp / np.log10(solvent_eps + 1.0), np.nan)
        # crude ligand Hansen proxy: dispersion tracks logP, polarity tracks TPSA
        ligand_dD = 16.0 + 0.30 * np.clip(logp, -2, 12)
        ligand_dP = 0.35 * np.sqrt(np.clip(tpsa, 0, 400))
        ligand_dH = 0.25 * np.sqrt(np.clip(tpsa, 0, 400))
        out["ligphys__hansen_distance"] = np.sqrt(
            4.0 * (ligand_dD - solvent_dD) ** 2
            + (ligand_dP - solvent_dP) ** 2 + (ligand_dH - solvent_dH) ** 2)

    columns = tuple(sorted(out))
    return frame.assign(**out), columns

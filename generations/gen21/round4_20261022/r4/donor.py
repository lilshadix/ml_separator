"""Donor-defined DGA / non-DGA subgroup (PROTOCOL section 6, METRICS.md), from the extractant structure only.

DGA = the extractant SMILES (the cohort's extractant key) contains a diglycolamide core, two amide carbonyls
joined by C-O-C: SMARTS ``[NX3]C(=O)[CX4]O[CX4]C(=O)[NX3]``.  No label, residual or curated family enters the
rule; on C3 it gives 40 of 79 extractants (21 in sc009, 19 in other chemotypes; the two sc009 DOODA
diamides with an O-C-C-O spacer are not DGA), identical to the curated gen13 ``chem_family`` = diglycolamide.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

SMARTS = "[NX3]C(=O)[CX4]O[CX4]C(=O)[NX3]"


@lru_cache(maxsize=None)
def is_dga(smiles: str) -> bool:
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"unparsable extractant SMILES: {smiles}")
    return bool(m.HasSubstructMatch(Chem.MolFromSmarts(SMARTS)))


def add_donor(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["dga_donor"] = t.extractant.astype(str).map(is_dga)
    return t

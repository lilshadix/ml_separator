"""Data-quality quarantines, run as a sensitivity analysis and never as a silent edit.

Two defects were found in the corpus during gen7.  Neither is touched in the primary
cohort — the frozen cohort is what makes gen5/gen6/gen7 comparable, and quietly
deleting rows from it would break every earlier number.  They are applied here as an
explicit arm so their effect is measured rather than assumed.

**DMDPhPDA exponent corruption (70 cells, 140 rows).**
``CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2)n1)c1ccccc1`` appears twice under two names, from
one DOI, entered by two curators.  Matching the two copies on (metal, acid, acid
concentration, extractant concentration) gives 70 paired cells whose ``log_D``
differences are *exactly* the integers 0, 1, 2, 3 and 4 (counts 11/17/14/14/14) — a
distribution no experiment produces.  The offset tracks acid concentration
(4, 3, 2, 1, 0 as HNO₃ goes 1→5 M), which is the signature of a table reported in
scientific notation transcribed mantissa-only.  The copy under the long IUPAC name is
the corrupt one: it is flat in acid concentration (0.301, 0.531, 0.380, 0.204, 0.799),
which is not physically possible for a dicarboxamide extracting from nitrate, while
the ``DMDPhPDA`` copy rises monotonically from −3.70 to −0.20 as the mass-action law
requires.

**Unmodelled second species (414 rows).**
414 rows carry an ``extractant_name`` that is not the modal name of their structure;
in the largest case 20 water-soluble complexants (SO3-Ph-BTP, the TWE- series,
PHEN-6OH, PyTri-diol, CITAM…) are recorded against TODGA's SMILES.  These rows
describe two-species experiments the model sees as one.  Note the measured
qualification: **no model-visible cell mixes flagged and unflagged rows**, so this
creates no direct contradiction — it misattributes rows to a ligand rather than
contradicting it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The structure entered twice, and the name whose copy is corrupt.
DMDPHPDA_SMILES = "CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2)n1)c1ccccc1"
DMDPHPDA_CORRUPT_NAME = "2-N-6-N-dimethyl-2-N-6-N-diphenylpyridine-2-6-dicarboxamide"


def quarantine_mask(frame: pd.DataFrame, *, drop_corrupt_duplicate: bool = True,
                    drop_unmodelled_species: bool = True) -> np.ndarray:
    """Boolean mask of rows to KEEP under the quarantine."""
    keep = np.ones(len(frame), dtype=bool)
    if drop_corrupt_duplicate and "nuisance__extractant_name" in frame.columns:
        keep &= ~((frame["extractant"].astype(str) == DMDPHPDA_SMILES)
                  & (frame["nuisance__extractant_name"].astype(str) == DMDPHPDA_CORRUPT_NAME)
                  ).to_numpy()
    if drop_unmodelled_species and "rec__name_mismatch" in frame.columns:
        keep &= ~(frame["rec__name_mismatch"].fillna(0).to_numpy() > 0)
    return keep


def quarantine_audit(frame: pd.DataFrame) -> dict:
    out = {"n_rows": int(len(frame))}
    for label, kwargs in (("corrupt_duplicate", dict(drop_unmodelled_species=False)),
                          ("unmodelled_species", dict(drop_corrupt_duplicate=False)),
                          ("both", {})):
        keep = quarantine_mask(frame, **kwargs)
        out[label] = {"kept": int(keep.sum()), "dropped": int((~keep).sum())}
    return out

"""The Gen12 Eu cohort: one row per (extractant, condition) cell of europium.

Construction mirrors the repository's level-cohort builder
(:func:`lanthanide_separation.levels.build_level_dataset`) so that the
identity keys are *the same objects* the frozen programme used:

* ``condition_id`` — SHA-1 of the full 64-column condition vector;
* ``series_id``    — extractant × categorical conditions (a titration);
* ``ecfp_cluster`` — SHA-1 of the 2048 Morgan bits (bit-identical homologues);
* ``chemotype``    — the gen6 frozen all-190 single-linkage Tanimoto-0.7
  super-cluster (conservative: it merges through non-Eu ligands, never splits);
* ``row_id``       — SHA-1 of ``extractant|condition_id|metal`` (same formula as
  the frozen cohort, so Eu cells can be cross-checked against it).

Differences from the frozen builder, all deliberate and recorded in
DATA_AUDIT.md: no ``min_rows`` filter (a one-cell extractant is a legitimate
zero-shot test unit), no geometry requirement, no ``log_D`` floor (no Eu value
is at or below −6), and the lanthanide-level 3D blocks are not attached.

Replicates are collapsed to their mean per cell with ``n_replicates`` and
``replicate_range`` recorded — the programme's convention since gen5.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import paths
from .chemistry import ECFP_COLUMNS, ecfp_clusters, eu_local_chemotypes, frozen_chemotypes
from lanthanide_separation.levels import (  # noqa: E402
    CONTINUOUS_CONDITION_COLUMNS, LEVEL_LIGAND_SCALAR_COL
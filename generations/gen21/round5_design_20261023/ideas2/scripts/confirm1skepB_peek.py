#!/usr/bin/env python
"""confirm-1 Skeptic B, peek (EXPLORATORY, seen cells): bench columns only."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "round4_20261022"))
from r4 import paths as _P  # noqa
from gen20k import data as D
b = D.load_bench()
f = b.frame
print(f.shape)
print([c for c in f.columns])
print(f[[c for c in f.columns if c in ("extractant","chemotype","publication_id","extractant_name","ligand_name","name")]].head())

#!/usr/bin/env python
"""Write the confirmation-seed commitment (sha256 of the sorted seed list) before discovery runs.
The seeds themselves follow the public rule in PRE_REGISTRATION.md section 0 and are recomputed by
g20_confirm.py; only the commitment is stored here."""
from __future__ import annotations

import sys
from pathlib import Path

G20 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G20)); sys.path.insert(0, str(G20 / "scripts"))

from gen20k import bootstrap  # noqa: E402
from g20_confirm import COMMIT, commitment, confirmation_seeds  # noqa: E402

if __name__ == "__main__":
    bootstrap.MANIFESTS.mkdir(parents=True, exist_ok=True)
    c = commitment(confirmation_seeds())
    COMMIT.write_text(c + "\n")
    print("commitment written:", c)

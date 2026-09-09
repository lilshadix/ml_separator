"""The frozen confirmation claim list (PRE_REGISTRATION.md section 4).

Empty until the end of discovery.  At freeze time the orchestrator adds at most five ``Claim``
entries, each pointing at the lead's own arm code so the confirmation run executes byte-for-byte
the same pipeline on the withheld seeds.  ``run(seeds)`` must return ``(board, contrasts)`` where
``contrasts`` carries the ``gen13sep.inference.paired_contrasts`` columns and one row per design
for ``comparison``.  ``discovery_contrasts`` is the lead's discovery CSV, relative to
``gen16_leads/``, from which the discovery column of the confirmation table is read.

Once ``scripts/g16_confirm.py`` has run, this file is not edited again.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

Runner = Callable[[Sequence[int]], tuple[pd.DataFrame, pd.DataFrame]]


@dataclass(frozen=True)
class Claim:
    id: str
    lead: str
    statement: str
    comparison: str               # the registered comparison name in the contrasts tables
    discovery_contrasts: str      # path relative to gen16_leads/
    run: Runner                   # seeds -> (board, contrasts), all five designs


CLAIMS: dict[str, Claim] = {}

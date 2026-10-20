"""Direction C: a small mechanistic chemistry block (pinned 2026-10-19 from the feature inventory, before any
C configuration was run; see ATTEMPTS.md).

Five target-free descriptors from bench.frames, chosen for a stated mechanism and checked against the
inventory (no label-derived, availability-derived or identifier columns; no chemotype or family IDs):
  n_aromatic_N > 0           soft N-heterocycle donors (BTP/BTBP/phenanthroline class)      COORD
  n_dga_unit                 diglycolamide unit present                                     COORD
  n_substituted_alpha_C      backbone alpha-substitution (reverse-selectivity DGA class)    LIG2D (NaN -> 0)
  is_HCl                     chloride rather than nitrate medium                            COND
  log10 [acid] / M           aqueous acidity                                                COND
Standardised with the mean / sd of the OUTER TRAINING cells of each fold (never the held-out cells).
"""
from __future__ import annotations

import numpy as np

from . import paths  # noqa: F401

COLUMNS = ("aromN", "dga", "alphaC", "HCl", "logH")


class ChemBlock:
    n = len(COLUMNS)

    def __init__(self, bench):
        f = bench.frames
        aromN = (f["COORD"]["coord__donor__n_N_aromatic"].fillna(0).to_numpy(float) > 0).astype(float)
        dga = f["COORD"]["coord__motif__n_dga_unit"].fillna(0).to_numpy(float)
        alpha = f["LIG2D"]["lig2d__hc__n_substituted_alpha_carbons"].fillna(0).to_numpy(float)
        hcl = f["COND"]["cond__acid__hcl"].fillna(0).to_numpy(float)
        logh = np.log10(np.clip(f["COND"]["cond__acid_concentration_M"].to_numpy(float), 1e-4, None))
        self.raw = np.c_[aromN, dga, alpha, hcl, logh]
        self._stats: dict = {}

    def stats(self, rec):
        key = (rec["design"], rec["split_seed"], rec["fold"])
        if key not in self._stats:
            tr = np.asarray(rec["outer"]["train_index"])
            mu = np.nanmean(self.raw[tr], axis=0)
            sd = np.nanstd(self.raw[tr], axis=0)
            sd[sd < 1e-9] = 1.0
            self._stats[key] = (mu, sd)
        return self._stats[key]

    def vector(self, ci: int, rec) -> np.ndarray:
        mu, sd = self.stats(rec)
        return np.nan_to_num((self.raw[int(ci)] - mu) / sd)

"""Comparing a gen11 arm to the control, and deciding whether it cleared the bar.

Two rules from the brief shape everything here.

**Every comparison is paired on identical evaluation units.**  Arms differ only
in which rows entered *training*; the rows they are scored on are byte-identical,
so the right statistic is a paired bootstrap over independent chemistry units,
not two independent means.  gen6's :func:`per_unit_statistics` /
:func:`paired_unit_bootstrap` already implement it with the blocking gen10 used,
so gen11 calls them rather than growing a second convention.

**A headline MAE improvement without a decomposition is insufficient** (§11).  So
a comparison is never reported as one number.  :func:`decompose` splits the
difference into the pieces that can move independently — the per-ligand level,
the within-curve shape, and the strata where auxiliary data could plausibly act
(chemotype distance, whether the auxiliary pool contained that extractant at all,
whether the metal group is actinide) — because "0.02 better" means something
different if it is 0.2 on four far ligands and zero everywhere else.

The stopping rule (§18) is encoded once, in :data:`STOPPING_RULE`, and evaluated
mechanically.  It is written down before the arms run so that "did it pass" is a
lookup rather than an argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..gen6.metrics import paired_unit_bootstrap, per_unit_statistics
from ..levels import LEVEL_TARGET_COLUMN

#: gen11's pre-registered inclusion criteria.  ``A`` .. ``E`` are the brief's §18.
STOPPING_RULE: Mapping[str, dict] = {
    "A_ZERO_SHOT": {
        "metric": "macro_mae", "k": 0, "min_improvement": 0.02,
        "requires": "consistent seed direction",
        "description": "zero-shot macro MAE improves by >= 0.02 on the locked benchmark",
    },
    "B_FEW_SHOT": {
        "metric": "macro_mae", "k": [1, 2, 3, 5], "min_improvement": 0.01,
        "requires": "at least two k, no meaningful degradation at later k",
        "description": "few-shot macro MAE improves by >= 0.01 at two or more k",
    },
    "C_SHAPE": {
        "metric": "shape_mae_by_axis", "min_improvement": 0.10,
        "requires": "macro non-worse",
        "description": "shape MAE improves by >= 0.10 on a scientifically important axis",
    },
    "D_FAR_CHEMOTYPE": {
        "metric": "macro_mae_far", "min_improvement": None,
        "requires": "statistically supported, overall benchmark not degraded",
        "description": "clear improvement on distant held-out chemotypes",
    },
    "E_ROBUSTNESS": {
        "metric": "query_consistency", "min_improvement": None,
        "requires": "matched predictive accuracy",
        "description": "meaningfully lower query-design sensitivity at matched accuracy",
    },
}

#: Chemotype-distance strata.  Cut points are gen10's, reused so the far stratum
#: means the same thing it did in the gen10 report.
CHEMOTYPE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("far", -np.inf, 0.4),
    ("mid", 0.4, 0.6),
    ("near", 0.6, np.inf),
)


def band_of(similarity: pd.Series) -> pd.Series:
    """Label each row near / mid / far by its nearest training-ligand Tanimoto."""
    out = pd.Series(index=similarity.index, dtype=object)
    for name, low, high in CHEMOTYPE_BANDS:
        out[(similarity > low) & (similarity <= high)] = name
    return out.fillna("far")


def wide_predictions(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (seed, cohort row) with a ``prediction_<arm>`` column per arm.

    Built by an inner join on ``(split_seed, row_id)`` so that a comparison can
    only ever be drawn on rows every arm actually predicted; an arm that silently
    lost rows shows up as a shrunken frame rather than as a flattering mean.
    """
    keys = ["split_seed", "row_id"]
    carry = [LEVEL_TARGET_COLUMN, "extractant", "ecfp_cluster", "tanimoto_cluster",
             "series_id", "metal_symbol", "fold", "nn_train_tanimoto"]
    merged: pd.DataFrame | None = None
    for name, frame in frames.items():
        block = frame.copy()
        block["row_id"] = block["row_id"].astype(str)
        columns = keys + [c for c in carry if c in block.columns] + ["prediction"]
        block = block[columns].rename(columns={"prediction": f"prediction_{name}"})
        if merged is None:
            merged = block
        else:
            merged = merged.merge(block[keys + [f"prediction_{name}"]], on=keys, how="inner",
                                  validate="one_to_one")
    if merged is None:
        raise ValueError("no arms supplied")
    return merged


def compare(frames: Mapping[str, pd.DataFrame], *, reference: str,
            candidates: Sequence[str], block: str = "tanimoto_cluster",
            replicates: int = 5000) -> pd.DataFrame:
    """Paired bootstrap of every candidate against ``reference``.

    ``block`` is the resampling unit.  ``tanimoto_cluster`` is the chemotype
    blocking gen10 reported as the *optimistic* interval; pass ``doi`` for the
    publication-blocked variant §13 asks for, which gen10 measured as ~40 % wider.
    """
    wide = wide_predictions(frames)
    arms = [reference, *candidates]
    per_unit = per_unit_statistics(wide, arms)
    block_of_unit = None
    if block != "ecfp_cluster" and block in wide.columns:
        block_of_unit = (wide.drop_duplicates("ecfp_cluster")
                         .set_index("ecfp_cluster")[block].astype(str).to_dict())
    comparisons = {candidate: (reference, candidate) for candidate in candidates}
    return paired_unit_bootstrap(per_unit, comparisons, block_of_unit=block_of_unit,
                                 replicates=replicates)


def per_seed_direction(frames: Mapping[str, pd.DataFrame], *, reference: str,
                       candidate: str) -> pd.DataFrame:
    """Macro MAE of both arms in every seed, and the sign of the difference.

    "No single-seed result may become a headline" (§17) cuts both ways: an effect
    that is positive on average but negative in two of five seeds is reported as
    such rather than as its mean.
    """
    wide = wide_predictions(frames)
    rows = []
    for seed, block in wide.groupby("split_seed"):
        macro = {}
        for arm in (reference, candidate):
            error = (block[f"prediction_{arm}"] - block[LEVEL_TARGET_COLUMN]).abs()
            macro[arm] = error.groupby(block["ecfp_cluster"]).mean().mean()
        rows.append({"split_seed": int(seed), "reference_macro_mae": macro[reference],
                     "candidate_macro_mae": macro[candidate],
                     "improvement": macro[reference] - macro[candidate]})
    out = pd.DataFrame(rows)
    out["improved"] = out["improvement"] > 0
    return out


def decompose(frames: Mapping[str, pd.DataFrame], *, reference: str, candidate: str,
              aux_extractants: Sequence[str] = ()) -> pd.DataFrame:
    """Where the difference between two arms actually sits.

    Splits by chemotype-distance band, by whether the auxiliary pool contained
    that extractant at all, and into the level (per-ligand mean residual) and
    shape (deviation about it) components that gen7–gen10 showed move
    independently.
    """
    wide = wide_predictions(frames)
    wide = wide.assign(band=band_of(wide["nn_train_tanimoto"])
                       if "nn_train_tanimoto" in wide.columns else "unknown")
    aux_set = set(aux_extractants)
    wide["extractant_in_aux"] = wide["extractant"].isin(aux_set)

    records = []
    for arm in (reference, candidate):
        residual = wide[f"prediction_{arm}"] - wide[LEVEL_TARGET_COLUMN]
        level = residual.groupby([wide["split_seed"], wide["extractant"]]).transform("mean")
        wide[f"abs_{arm}"] = residual.abs()
        wide[f"level_{arm}"] = level.abs()
        wide[f"shape_{arm}"] = (residual - level).abs()

    for keys, block in [(("band",), wide.groupby("band")),
                        (("extractant_in_aux",), wide.groupby("extractant_in_aux")),
                        (("ALL",), [("ALL", wide)])]:
        for name, part in block:
            row = {"stratum_kind": keys[0], "stratum": str(name),
                   "n_rows": int(len(part)), "n_ligands": int(part["extractant"].nunique())}
            for component in ("abs", "level", "shape"):
                ref = part.groupby("ecfp_cluster")[f"{component}_{reference}"].mean().mean()
                cand = part.groupby("ecfp_cluster")[f"{component}_{candidate}"].mean().mean()
                row[f"{component}_reference"] = float(ref)
                row[f"{component}_candidate"] = float(cand)
                row[f"{component}_improvement"] = float(ref - cand)
            records.append(row)
    return pd.DataFrame(records)


def ligands_moved(frames: Mapping[str, pd.DataFrame], *, reference: str,
                  candidate: str) -> dict:
    """How many ligands improved and how many worsened, not just the average."""
    wide = wide_predictions(frames)
    per_ligand = {}
    for arm in (reference, candidate):
        error = (wide[f"prediction_{arm}"] - wide[LEVEL_TARGET_COLUMN]).abs()
        per_ligand[arm] = error.groupby(wide["extractant"]).mean()
    delta = per_ligand[reference] - per_ligand[candidate]
    return {"n_ligands": int(len(delta)),
            "n_improved": int((delta > 0).sum()),
            "n_worsened": int((delta < 0).sum()),
            "n_unchanged": int((delta == 0).sum()),
            "median_improvement": float(delta.median()),
            "worst_regression": float(delta.min()),
            "best_improvement": float(delta.max())}

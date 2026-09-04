"""Finding the auxiliary data that makes the model *worse*, and naming it.

The brief (§14) states the rule this module exists to serve: **negative transfer
is a first-class result**.  A pooled "+0.01 macro MAE" is compatible with an
archive in which uranium helps a lot, americium hurts a lot, and the two cancel;
reporting only the pool would then be reporting an accident of composition.  So
the question here is not "did the auxiliary data help" but "which part of it did
what, and to which lanthanide chemistry".

Three design decisions carry the whole file.

**Leave-one-metal-out from the full pool, not one-metal-in.**  Adding a metal to
an empty pool measures that metal against *nothing*; removing it from the full
pool measures it against *everything else*, which is the deployment question — the
archive is used whole or not at all.  The two disagree whenever metals are
redundant, and 90 % of this pool is actinide, so redundancy is the expected case.
The cost is that every LOMO arm is a full refit of a nearly-full pool: dropping
22 cells costs exactly what dropping 1,664 costs.

That cost is turned into evidence rather than merely paid.  The partition
includes groups small enough that their removal *cannot* matter chemically
(``other_actinide`` is ~22 cells of Pa and Cf, ~0.4 % of the pool), and the
spread of those deltas is the **noise floor** of the sweep: a per-metal
contribution smaller than what removing 22 cells produces is not a measurement.
:func:`noise_floor` computes it, and it is reported next to the contributions
rather than left for the reader to infer.

**A metal's contribution is confounded with its size.**  Am is a third of the
pool; "removing Am hurts" and "removing 1,664 training cells hurts" are the same
sentence unless a size-matched random removal is run alongside.  ``LOMO_RANDOM``
draws n cells uniformly from the pool, so the difference between the Am delta and
the random-n delta is the part attributable to Am *being Am*.

There is a subtlety in what "removing" does under the headline weighting.
``HIERARCHICAL`` at ``aux_lambda = 1`` sets the auxiliary block's *total* mass to
the lanthanide core's, whatever the block contains
(``arms._scale_auxiliary``).  A LOMO arm therefore does not train on *less*
auxiliary influence — it redistributes the removed metal's mass over the metals
that remain.  So ``contribution(g)`` answers "is g worth its place in the pool",
not "is more auxiliary data better"; the latter is arm E against the control.
The size-matched random removal is renormalised the same way, which is what makes
it the right comparison rather than merely a smaller pool.  The driver checks the
renormalisation held by reading ``aux_weight_share`` out of each arm's
diagnostics instead of trusting this paragraph.

**Sign convention, stated once.**

.. code-block:: text

    contribution(g) = macro_MAE(pool without g) - macro_MAE(full pool)

    contribution > 0  ->  removing g made prediction worse  ->  g HELPED
    contribution < 0  ->  removing g made prediction better ->  g HURT
                          (this is negative transfer, and it is a result)

This is exactly ``analysis.compare(reference=<lomo arm>, candidate=<full pool>)``
with ``statistic == "mae"``, so the interval, the p-value and the
units-improved count come from gen6's paired blocked bootstrap unchanged; no new
statistics are written here.  ``units_improved`` then reads as "chemotype units
for which the group helped", and ``p_worse_one_sided`` as "probability the group
is not helpful".

**What this module must not do**: drop a harmful group from the headline.  §14
allows exclusion only as a *pre-declared follow-up ablation*.  The driver
therefore always emits the full-pool result and every ablation side by side, and
the ablation table carries a ``source`` column so a reader can never mistake one
for the other.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..levels import LEVEL_TARGET_COLUMN
from .analysis import compare, decompose, wide_predictions
from .arms import ACTINIDE, AuxiliaryPool
from .runner import RunSpec

#: A metal gets its own leave-one-out arm when the pool holds at least this many
#: of its cells.  Below it the metal joins its ``metal_category`` group, because a
#: 5-cell removal is a fit whose answer is known in advance to be zero and whose
#: cost is a full fit.  Declared before the sweep runs so the partition is a rule
#: rather than a reaction to the deltas.
MIN_CELLS_FOR_OWN_GROUP = 100

#: Fixed draws for the size-matched random removals (§5's F/G logic applied to
#: subtraction instead of addition).  Fixed so a draw cannot be re-rolled.
RANDOM_REMOVAL_SEEDS: tuple[int, ...] = (901, 902)

#: Group key for "every actinide at once" — the LOMO complement of arm C.
ALL_ACTINIDE_KEY = "GROUP_all_actinide"


# --------------------------------------------------------------------------- #
# The partition
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MetalGroup:
    """One removable unit of the auxiliary pool."""

    key: str
    metals: tuple[str, ...]
    category: str
    n_cells: int
    kind: str  # "metal" | "category_remainder" | "category" | "random_matched"

    @property
    def label(self) -> str:
        return f"LOMO_{self.key}"


def metal_groups(features: pd.DataFrame, *,
                 min_cells: int = MIN_CELLS_FOR_OWN_GROUP) -> tuple[MetalGroup, ...]:
    """Partition the auxiliary pool into removable groups.

    Every auxiliary cell belongs to exactly one group, which is what makes the
    contributions add up to something interpretable: the groups are not
    overlapping selections chosen to make a point.
    """
    counts = features["metal_symbol"].astype(str).value_counts()
    category_of = (features.drop_duplicates("metal_symbol")
                   .set_index(features.drop_duplicates("metal_symbol")["metal_symbol"].astype(str))
                   ["metal_category"].astype(str).to_dict())

    groups: list[MetalGroup] = []
    remainder: dict[str, list[str]] = {}
    for metal, n in counts.items():
        metal = str(metal)
        category = category_of.get(metal, "unknown")
        if int(n) >= min_cells:
            groups.append(MetalGroup(key=metal, metals=(metal,), category=category,
                                     n_cells=int(n), kind="metal"))
        else:
            remainder.setdefault(category, []).append(metal)

    for category, metals in sorted(remainder.items()):
        members = tuple(sorted(metals))
        n = int(counts[list(members)].sum())
        groups.append(MetalGroup(key=f"other_{category}", metals=members, category=category,
                                 n_cells=n, kind="category_remainder"))

    covered = sum(g.n_cells for g in groups)
    if covered != len(features):
        raise SystemExit(f"metal groups cover {covered} of {len(features)} auxiliary cells; "
                         "the partition is not a partition")
    return tuple(sorted(groups, key=lambda g: (-g.n_cells, g.key)))


def actinide_group(features: pd.DataFrame) -> MetalGroup:
    """Every actinide, as one removable group — the LOMO complement of arm C."""
    mask = features["metal_category"].astype(str) == ACTINIDE
    metals = tuple(sorted(features.loc[mask, "metal_symbol"].astype(str).unique()))
    return MetalGroup(key=ALL_ACTINIDE_KEY, metals=metals, category=ACTINIDE,
                      n_cells=int(mask.sum()), kind="category")


def groups_table(groups: Sequence[MetalGroup], *, n_pool: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "group": g.key, "kind": g.kind, "metal_category": g.category,
        "metals": "|".join(g.metals), "n_metals": len(g.metals),
        "n_cells": g.n_cells, "pool_share": g.n_cells / n_pool if n_pool else np.nan,
    } for g in groups])


# --------------------------------------------------------------------------- #
# Pool surgery
# --------------------------------------------------------------------------- #

def drop_metals(pool: AuxiliaryPool, metals: Sequence[str]) -> AuxiliaryPool:
    """The pool with every cell of ``metals`` removed.

    Removal is by ``metal_symbol`` on the *features*, never by editing
    ``safe_ids``: the leakage table stays exactly the reviewed artefact it was,
    and a LOMO arm differs from the full-pool arm in one respect only.
    """
    wanted = {str(m) for m in metals}
    present = set(pool.features["metal_symbol"].astype(str))
    unknown = sorted(wanted - present)
    if unknown:
        raise SystemExit(f"cannot drop metals absent from the pool: {unknown}")
    keep = ~pool.features["metal_symbol"].astype(str).isin(wanted)
    if bool(keep.all()):
        raise SystemExit(f"dropping {sorted(wanted)} removed nothing")
    return pool.restrict(keep.to_numpy())


def drop_random_matched(pool: AuxiliaryPool, n_drop: int, *, seed: int) -> AuxiliaryPool:
    """Remove ``n_drop`` uniformly random cells — the size control for a LOMO arm.

    Without this, "removing Am hurts" cannot be told apart from "removing 36 % of
    the training rows hurts".  The draw is seeded from a fixed constant, so it is
    a declared experiment rather than a resample taken after seeing the result.
    """
    n = len(pool.features)
    if n_drop >= n:
        raise SystemExit(f"random removal of {n_drop} would empty a pool of {n}")
    rng = np.random.default_rng(seed)
    drop = rng.choice(n, size=int(n_drop), replace=False)
    keep = np.ones(n, dtype=bool)
    keep[drop] = False
    return pool.restrict(keep)


# --------------------------------------------------------------------------- #
# Arm naming
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LabelledRunSpec(RunSpec):
    """A :class:`RunSpec` whose name carries which group was removed.

    The pool surgery happens outside the spec — two LOMO arms are the same
    ``RunSpec`` fed different pools — so without a label every arm in the sweep
    would write to one file name and the sweep would silently become its last
    member.  Subclassing rather than editing :class:`RunSpec` keeps the frozen
    primary run's contract untouched.
    """

    label: str = ""

    @property
    def name(self) -> str:
        base = super().name
        return f"{base}|{self.label}" if self.label else base


# --------------------------------------------------------------------------- #
# Level / shape components
# --------------------------------------------------------------------------- #

def attach_components(wide: pd.DataFrame, arms: Sequence[str]) -> pd.DataFrame:
    """Add ``abs_``/``level_``/``shape_`` columns per arm, on a copy.

    The definitions are :func:`gen11.analysis.decompose`'s, restated here only
    because ``decompose`` returns aggregates and this file needs the components
    at chemotype and ligand granularity.  ``assert_matches_decompose`` proves the
    two agree rather than asserting it in a comment.

    The level is the per-``(seed, ligand)`` mean residual and the shape is the
    deviation about it, which is the split gen7–gen10 showed moves
    independently — a per-ligand constant that structure does not predict, and a
    within-curve response that it does.
    """
    out = wide.copy()
    for arm in arms:
        column = f"prediction_{arm}"
        if column not in out.columns:
            raise KeyError(f"missing prediction column {column!r}")
        residual = out[column] - out[LEVEL_TARGET_COLUMN]
        level = residual.groupby([out["split_seed"], out["extractant"]]).transform("mean")
        out[f"abs_{arm}"] = residual.abs()
        out[f"level_{arm}"] = level.abs()
        out[f"shape_{arm}"] = (residual - level).abs()
    return out


def macro(block: pd.DataFrame, column: str, *, unit: str = "ecfp_cluster") -> float:
    """Macro mean of ``column``: mean within each scoring unit, then across units."""
    if block.empty:
        return float("nan")
    return float(block.groupby(unit)[column].mean().mean())


def assert_matches_decompose(frames: Mapping[str, pd.DataFrame], *, reference: str,
                             candidate: str, tolerance: float = 1e-12) -> dict:
    """Check :func:`attach_components` reproduces ``analysis.decompose``'s ALL row.

    A real check: it fails if the two conventions ever drift apart.
    """
    wide = attach_components(wide_predictions(frames), [reference, candidate])
    mine = {f"{component}_{arm}": macro(wide, f"{component}_{arm}")
            for component in ("abs", "level", "shape") for arm in (reference, candidate)}
    table = decompose(frames, reference=reference, candidate=candidate)
    row = table[table["stratum"] == "ALL"].iloc[0]
    deltas = {}
    for component in ("abs", "level", "shape"):
        deltas[f"{component}_reference"] = abs(mine[f"{component}_{reference}"]
                                               - float(row[f"{component}_reference"]))
        deltas[f"{component}_candidate"] = abs(mine[f"{component}_{candidate}"]
                                               - float(row[f"{component}_candidate"]))
    worst = max(deltas.values())
    if worst > tolerance:
        raise SystemExit(f"attach_components disagrees with analysis.decompose by {worst:.3g}: "
                         f"{deltas}")
    return {"max_abs_delta_vs_decompose": float(worst), "tolerance": tolerance}


# --------------------------------------------------------------------------- #
# Comparison summaries
# --------------------------------------------------------------------------- #

def comparison_summary(frames: Mapping[str, pd.DataFrame], *, reference: str, candidate: str,
                       block: str = "tanimoto_cluster", replicates: int = 5000) -> dict:
    """Point deltas, blocked bootstrap interval and level/shape split for one pair.

    ``reference - candidate`` throughout, so a positive number always means "the
    candidate is better".  For a LOMO arm the reference *is* the arm without the
    group, which makes the sign the group's contribution (module docstring).
    """
    wide = attach_components(wide_predictions(frames), [reference, candidate])
    summary = {
        "reference": reference, "candidate": candidate,
        "n_rows": int(len(wide)),
        "n_seeds": int(wide["split_seed"].nunique()),
        "n_ligands": int(wide["extractant"].nunique()),
        "n_units": int(wide["ecfp_cluster"].nunique()),
        "n_blocks": int(wide[block].nunique()) if block in wide.columns else np.nan,
    }
    for component in ("abs", "level", "shape"):
        ref = macro(wide, f"{component}_{reference}")
        cand = macro(wide, f"{component}_{candidate}")
        name = {"abs": "macro_mae", "level": "level_mae", "shape": "shape_mae"}[component]
        summary[f"{name}_reference"] = ref
        summary[f"{name}_candidate"] = cand
        summary[f"{name}_delta"] = ref - cand

    boot = compare(frames, reference=reference, candidates=[candidate], block=block,
                   replicates=replicates)
    for _, row in boot.iterrows():
        stat = str(row["statistic"])
        summary[f"boot_{stat}_delta"] = float(row["point_delta"])
        summary[f"boot_{stat}_ci95_low"] = float(row["ci95_low"])
        summary[f"boot_{stat}_ci95_high"] = float(row["ci95_high"])
        summary[f"boot_{stat}_p_worse_one_sided"] = float(row["p_worse_one_sided"])
        summary[f"boot_{stat}_units_improved"] = int(row["units_improved"])
        summary[f"boot_{stat}_units_total"] = int(row["units_total"])
    summary["bootstrap_block"] = block
    return summary


def per_seed_signs(frames: Mapping[str, pd.DataFrame], *, reference: str,
                   candidate: str) -> dict:
    """How many split seeds agree with the pooled direction.

    §17: an effect that is positive on average and negative in two of five seeds
    is not the same object as one that is positive in five of five.
    """
    wide = wide_predictions(frames)
    deltas = []
    for seed, block in wide.groupby("split_seed"):
        ref = macro(block.assign(e=(block[f"prediction_{reference}"]
                                    - block[LEVEL_TARGET_COLUMN]).abs()), "e")
        cand = macro(block.assign(e=(block[f"prediction_{candidate}"]
                                     - block[LEVEL_TARGET_COLUMN]).abs()), "e")
        deltas.append(ref - cand)
    arr = np.asarray(deltas, dtype=float)
    return {"n_seeds": int(arr.size),
            "n_seeds_positive": int((arr > 0).sum()),
            "n_seeds_negative": int((arr < 0).sum()),
            "seed_delta_min": float(arr.min()) if arr.size else np.nan,
            "seed_delta_max": float(arr.max()) if arr.size else np.nan}


# --------------------------------------------------------------------------- #
# Where the difference sits
# --------------------------------------------------------------------------- #

def chemotype_deltas(frames: Mapping[str, pd.DataFrame], *, reference: str, candidate: str,
                     group_column: str = "tanimoto_cluster") -> pd.DataFrame:
    """Per-chemotype delta, with the counts §14 asks for rather than only a mean.

    Within a chemotype the statistic is still macro over ECFP scoring units, so a
    chemotype containing one 1,500-row ligand and one 6-row ligand is not
    reported as the big ligand.
    """
    wide = attach_components(wide_predictions(frames), [reference, candidate])
    rows = []
    for name, part in wide.groupby(group_column):
        row = {
            "chemotype": str(name),
            "n_rows": int(len(part)),
            "n_ligands": int(part["extractant"].nunique()),
            "n_units": int(part["ecfp_cluster"].nunique()),
            "mean_nn_train_tanimoto": float(part["nn_train_tanimoto"].mean())
            if "nn_train_tanimoto" in part.columns else np.nan,
        }
        for component, label in (("abs", "macro_mae"), ("level", "level_mae"),
                                 ("shape", "shape_mae")):
            ref = macro(part, f"{component}_{reference}")
            cand = macro(part, f"{component}_{candidate}")
            row[f"{label}_reference"] = ref
            row[f"{label}_candidate"] = cand
            row[f"{label}_delta"] = ref - cand
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("macro_mae_delta")
    return out.reset_index(drop=True)


def ligand_deltas(frames: Mapping[str, pd.DataFrame], *, reference: str,
                  candidate: str) -> pd.DataFrame:
    """Per-ligand table, so a regression is nameable rather than only countable."""
    wide = attach_components(wide_predictions(frames), [reference, candidate])
    rows = []
    for name, part in wide.groupby("extractant"):
        per_seed = []
        for _, seed_block in part.groupby("split_seed"):
            per_seed.append(float(seed_block[f"abs_{reference}"].mean()
                                  - seed_block[f"abs_{candidate}"].mean()))
        arr = np.asarray(per_seed, dtype=float)
        row = {
            "extractant": str(name),
            "chemotype": str(part["tanimoto_cluster"].iloc[0])
            if "tanimoto_cluster" in part.columns else "",
            "ecfp_cluster": str(part["ecfp_cluster"].iloc[0]),
            "n_rows": int(len(part)),
            "n_seeds": int(arr.size),
            "n_seeds_worse": int((arr < 0).sum()),
            "mean_nn_train_tanimoto": float(part["nn_train_tanimoto"].mean())
            if "nn_train_tanimoto" in part.columns else np.nan,
        }
        for component, label in (("abs", "mae"), ("level", "level_mae"),
                                 ("shape", "shape_mae")):
            ref = float(part[f"{component}_{reference}"].mean())
            cand = float(part[f"{component}_{candidate}"].mean())
            row[f"{label}_reference"] = ref
            row[f"{label}_candidate"] = cand
            row[f"{label}_delta"] = ref - cand
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("mae_delta")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The noise floor
# --------------------------------------------------------------------------- #

def noise_floor(contributions: pd.DataFrame, *, share_threshold: float = 0.02) -> dict:
    """What a *chemically meaningless* removal produces, as a scale for the rest.

    Only groups holding less than ``share_threshold`` of the pool qualify.  Their
    deltas are known in advance to carry no per-metal information — removing 22
    of 5,096 cells cannot change the chemistry — so the largest absolute delta
    among them is the magnitude a per-metal contribution has to exceed before it
    is a measurement rather than refit variance.

    The size-matched random removals are deliberately **excluded**.  They remove
    a third of the pool, which is a real intervention with a real effect; folding
    them in here would inflate a noise floor with a size effect and then use the
    inflated floor to dismiss the size effect.  They are reported by
    :func:`size_control` instead.

    This is not a significance test and is not offered as one; it is a floor.
    """
    if contributions.empty:
        return {"n_floor_arms": 0}
    floor = contributions[(contributions["kind"] != "random_matched")
                          & (contributions["pool_share"] < share_threshold)]
    if floor.empty:
        return {"n_floor_arms": 0, "share_threshold": share_threshold}
    values = floor["contribution_macro_mae"].astype(float).to_numpy()
    return {
        "n_floor_arms": int(len(floor)),
        "floor_arms": "|".join(floor["group"].astype(str)),
        "floor_abs_max": float(np.abs(values).max()),
        "floor_abs_median": float(np.median(np.abs(values))),
        "floor_min": float(values.min()),
        "floor_max": float(values.max()),
        "share_threshold": share_threshold,
    }


def weight_share_audit(diagnostics_dir, *, expected: float, tolerance: float = 0.02
                       ) -> pd.DataFrame:
    """Per-arm auxiliary weight share, read from the diagnostics the fits wrote.

    The module docstring claims a LOMO arm redistributes the removed metal's mass
    rather than training on less of it.  That claim is checkable: under
    ``HIERARCHICAL`` at ``aux_lambda = lam`` the auxiliary block's share of total
    sample weight is ``lam / (1 + lam)`` in *every* arm regardless of what was
    removed.  A row flagged ``False`` here means the weighting did not behave as
    the design assumed and the contributions are not comparable across arms.
    """
    from pathlib import Path

    rows = []
    for path in sorted(Path(diagnostics_dir).glob("diagnostics_*.csv")):
        frame = pd.read_csv(path)
        if frame.empty or "aux_weight_share" not in frame.columns:
            continue
        share = frame["aux_weight_share"].astype(float)
        rows.append({
            "arm": str(frame["arm"].iloc[0]),
            "n_folds": int(len(frame)),
            "n_aux_min": int(frame["n_aux"].min()),
            "n_aux_max": int(frame["n_aux"].max()),
            "aux_weight_share_min": float(share.min()),
            "aux_weight_share_max": float(share.max()),
            "expected": expected,
            "as_designed": bool((share - expected).abs().max() <= tolerance),
        })
    return pd.DataFrame(rows)


def size_control(contributions: pd.DataFrame) -> dict:
    """The size-matched random removals, against the metal they are matched to.

    The pool is 90 % actinide and one metal is a third of it, so "removing Am
    hurts" and "removing a third of the training rows hurts" are the same
    sentence until a random removal of the same n is run.  The residual —
    contribution(Am) minus the mean random contribution at matched n — is the
    part attributable to Am being Am.
    """
    if contributions.empty or "kind" not in contributions.columns:
        return {"n_random_arms": 0}
    random = contributions[contributions["kind"] == "random_matched"]
    if random.empty:
        return {"n_random_arms": 0}
    n_cells = int(random["n_cells"].iloc[0])
    matched = contributions[(contributions["kind"] == "metal")
                            & (contributions["n_cells"] == n_cells)]
    values = random["contribution_macro_mae"].astype(float).to_numpy()
    out = {
        "n_random_arms": int(len(random)),
        "random_n_cells": n_cells,
        "random_contribution_mean": float(values.mean()),
        "random_contribution_min": float(values.min()),
        "random_contribution_max": float(values.max()),
    }
    if len(matched):
        row = matched.iloc[0]
        out["matched_metal"] = str(row["group"])
        out["matched_metal_contribution"] = float(row["contribution_macro_mae"])
        out["excess_over_random"] = float(row["contribution_macro_mae"] - values.mean())
    return out

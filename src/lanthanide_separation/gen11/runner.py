"""gen10's evaluation contract, driving a gen11 transfer arm.

Nothing about scoring changes.  The cohort is gen7's cache at fingerprint
``bed178ec1a7a82b0``, the folds are ``seeded_group_kfold`` over the Tanimoto
chemotype at the same five seeds, the model seed is ``42 + fold*1009 + 9_999_991``,
and the OOF frame lands in the schema every gen8/gen9/gen10 tool already reads.
Auxiliary rows enter training only.  They are never scored, never appear in an
OOF frame, and never redefine a test row — :func:`assert_evaluation_unchanged`
proves the evaluated rows are byte-identical to gen10's.

**The metal representation is an arm dimension, not a detail.**  The frozen
``METAL`` block is a 14-entry lanthanide dictionary, so 5,353 of the 5,438
auxiliary rows have all three columns missing: under median imputation every
actinide would arrive as an average lanthanide carrying an actinide's log D,
which is not transfer but noise injection.  gen11 therefore runs a 2x2 —
{control, +auxiliary} x {``FROZEN_3``, ``GENERAL``} — so that "the auxiliary data
helped" can be told apart from "a metal representation that works outside the
lanthanide series helped".  Reporting only the diagonal would confound them.

``FROZEN_3`` with no auxiliary rows is arm A, and it must reproduce gen10
exactly; :func:`lanthanide_separation.gen11.transfer.nesting_delta` asserts it at
the thread-order floor.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..gen7.harness import DEFAULT_SEEDS, Gen7Cohort, build_folds, evaluate_contender, score_oof
from ..gen9.train import FROZEN_BLOCKS
from ..gen10.architectures import ContenderAdapter
from ..gen10.runner import COHORT_FINGERPRINT, prepared_cohort
from ..levels import LEVEL_TARGET_COLUMN
from .arms import ARM_BY_KEY, AuxiliaryPool, MATCHED_SEEDS, WEIGHTINGS
from .transfer import MECHANISMS, TransferRecomposedModel

#: The gen11 metal-representation schemes, and the feature block each contributes.
METAL_SCHEMES: tuple[str, ...] = ("FROZEN_3", "FROZEN_3_INDICATED", "GENERAL", "NO_RADIUS")

#: Block name gen11 registers on the cohort for a non-frozen metal scheme.
GEN11_METAL_BLOCK = "METAL_GEN11"
#: Block name for the annotation-safe MASSACTION variant.
GEN11_MASSACTION_BLOCK = "MASSACTION_GEN11"

#: MASSACTION terms whose *missingness* is a property of the frozen bundle's
#: annotation coverage rather than of the chemistry.
#:
#: ``DENTATE`` (ligand denticity) and ``coreCN`` (complex coordination number)
#: come from the bundle's 3D pipeline, which only ever ran on its own 190
#: structures.  Measured on the built auxiliary frame, ``logL_x_coreCN`` is null
#: for 0.02 % of cohort rows and 98.44 % of auxiliary rows, and ``logL_x_DENTATE``
#: for 0.02 % against 40.66 %.  With ``add_indicator=True`` the imputer turns each
#: into a near-perfect "is this an auxiliary row" flag — a provenance feature,
#: which §2 forbids outright.
#:
#: ``metal_is_lanthanide`` is deliberately *not* on this list even though it
#: separates the two sets just as sharply.  It is a real property of the metal,
#: and a forest that splits on it simply declines to transfer — a null result,
#: not a false positive.  The distinction that matters is whether the column
#: encodes chemistry or bookkeeping.
ANNOTATION_COUPLED_MASSACTION: tuple[str, ...] = (
    "massact__logL_x_DENTATE", "massact__logL_x_coreCN",
)

#: How MASSACTION is built.  ``FROZEN_8`` is gen10's block; ``ANNOTATION_SAFE``
#: drops the two columns above.  Any arm carrying auxiliary rows must use
#: ``ANNOTATION_SAFE``, and its control must use it too or the comparison is
#: between two different designs.
MASSACTION_SCHEMES: tuple[str, ...] = ("FROZEN_8", "ANNOTATION_SAFE")


def blocks_for_scheme(scheme: str, massaction: str = "FROZEN_8") -> tuple[str, ...]:
    """Feature blocks of the design under ``scheme`` and ``massaction``.

    ``FROZEN_3`` + ``FROZEN_8`` is the frozen tuple untouched, so the control
    cannot drift.  Every other scheme *replaces* the block it modifies rather
    than adding to it: keeping the lanthanide-only columns alongside a general
    block would let the forest read "lanthanide_index is missing" as "this is an
    auxiliary row" and use it as a provenance flag.
    """
    if scheme not in METAL_SCHEMES:
        raise KeyError(f"unknown metal scheme {scheme!r}; have {METAL_SCHEMES}")
    if massaction not in MASSACTION_SCHEMES:
        raise KeyError(f"unknown massaction scheme {massaction!r}; have {MASSACTION_SCHEMES}")
    blocks = list(FROZEN_BLOCKS)
    if scheme != "FROZEN_3":
        blocks = [GEN11_METAL_BLOCK if b == "METAL" else b for b in blocks]
    if massaction != "FROZEN_8":
        blocks = [GEN11_MASSACTION_BLOCK if b == "MASSACTION" else b for b in blocks]
    return tuple(blocks)


def _ensure_metal_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Supply the archive-named metal inputs a general scheme needs.

    The cohort frame carries the frozen bundle's metal columns
    (``Atomic Number_metal``, ``Ionic Radius_metal``); :mod:`.metalrep` speaks the
    archive's names.  Both are the *same numbers* — the frozen-dict agreement
    audit found delta 0.0 on atomic number, lanthanide index and ionic radius for
    all 14 lanthanides — so this is a rename, not a second source of truth.

    ``metal_oxidation_state`` is set to 3 for cohort rows because the bundle
    records ``metal_ox = 3`` on all 5,992 of them; that is a read, not the kind of
    inference §7 forbids.
    """
    out = frame
    if "metal_oxidation_state" not in out.columns:
        out = out.assign(metal_oxidation_state=3.0)
    if "ionic_radius_cn8_A" not in out.columns:
        if "Ionic Radius_metal" not in out.columns:
            raise SystemExit("frame has neither ionic_radius_cn8_A nor Ionic Radius_metal")
        out = out.assign(ionic_radius_cn8_A=out["Ionic Radius_metal"].to_numpy(dtype=float))
    return out


def attach_metal_scheme(cohort: Gen7Cohort, aux: pd.DataFrame | None, scheme: str,
                        massaction: str = "FROZEN_8"
                        ) -> tuple[Gen7Cohort, pd.DataFrame | None]:
    """Register the gen11 blocks on a *copy* of the cohort and on ``aux``.

    The cohort fingerprint digests row ids and targets only — deliberately, so a
    feature block may be added without changing the evaluation contract — but the
    copy is made anyway so a scheme can never mutate the object another arm is
    still using.
    """
    from .metalrep import build as build_metal

    if scheme == "FROZEN_3" and massaction == "FROZEN_8":
        return cohort, aux

    frame = cohort.frame.copy()
    blocks = dict(cohort.data.blocks)

    if scheme != "FROZEN_3":
        frame = _ensure_metal_inputs(frame)
        metal_columns = build_metal(frame, scheme=scheme)
        for column in metal_columns.columns:
            frame[column] = metal_columns[column].to_numpy()
        blocks[GEN11_METAL_BLOCK] = tuple(metal_columns.columns)

    if massaction != "FROZEN_8":
        kept = tuple(c for c in cohort.blocks["MASSACTION"]
                     if c not in ANNOTATION_COUPLED_MASSACTION)
        if len(kept) == len(cohort.blocks["MASSACTION"]):
            raise SystemExit("ANNOTATION_SAFE dropped nothing; the MASSACTION block moved")
        blocks[GEN11_MASSACTION_BLOCK] = kept

    data = replace(cohort.data, frame=frame, blocks=blocks)
    patched = Gen7Cohort(data=data, chemistry=cohort.chemistry, fingerprint=cohort.fingerprint)
    if patched.fingerprint != COHORT_FINGERPRINT:
        raise SystemExit(f"gen11 blocks changed the cohort fingerprint: {patched.fingerprint}")

    if aux is None or aux.empty or scheme == "FROZEN_3":
        return patched, aux
    aux_metal = build_metal(_ensure_metal_inputs(aux), scheme=scheme)
    aux = aux.copy()
    for column in aux_metal.columns:
        aux[column] = aux_metal[column].to_numpy()
    return patched, aux


@dataclass(frozen=True)
class RunSpec:
    """One fully specified gen11 arm.  Every field is fixed before the run."""

    arm: str
    mechanism: str = "JOINT"
    weighting: str = "HIERARCHICAL"
    metal_scheme: str = "GENERAL"
    massaction_scheme: str = "ANNOTATION_SAFE"
    aux_lambda: float = 1.0
    control: str = "NONE"
    policy: str = "HEADLINE"
    matched_seed: int = MATCHED_SEEDS[0]
    matched_size: int | None = None

    def __post_init__(self) -> None:
        if self.arm not in ARM_BY_KEY:
            raise KeyError(f"unknown arm {self.arm!r}")
        if self.mechanism not in MECHANISMS:
            raise KeyError(f"unknown mechanism {self.mechanism!r}")
        if self.weighting not in WEIGHTINGS:
            raise KeyError(f"unknown weighting {self.weighting!r}")
        if self.metal_scheme not in METAL_SCHEMES:
            raise KeyError(f"unknown metal scheme {self.metal_scheme!r}")
        if self.massaction_scheme not in MASSACTION_SCHEMES:
            raise KeyError(f"unknown massaction scheme {self.massaction_scheme!r}")
        if self.arm != "A_GEN10_CONTROL" and self.massaction_scheme == "FROZEN_8":
            raise SystemExit(
                f"{self.arm} carries auxiliary rows but uses MASSACTION FROZEN_8; its "
                "annotation-coupled columns are a provenance flag on auxiliary rows")

    @property
    def name(self) -> str:
        parts = [self.arm, self.mechanism, self.metal_scheme, self.massaction_scheme,
                 self.weighting, f"lam{self.aux_lambda:g}"]
        # A size-matched arm is a *draw*, and each draw is its own arm: without
        # the seed in the name the five repeats would overwrite one another's OOF
        # and the "distribution over draws" would silently become one draw.
        if ARM_BY_KEY[self.arm].match_size_to is not None:
            parts.append(f"draw{self.matched_seed}")
        if self.control != "NONE":
            parts.append(self.control)
        if self.policy != "HEADLINE":
            parts.append(self.policy)
        return "|".join(parts)


def build_model(spec: RunSpec, pool: AuxiliaryPool, *, metal_columns: Sequence[str] = ()
                ) -> TransferRecomposedModel:
    return TransferRecomposedModel(
        name=spec.name, arm=ARM_BY_KEY[spec.arm], pool=pool, mechanism=spec.mechanism,
        weighting=spec.weighting, aux_lambda=spec.aux_lambda, control=spec.control,
        matched_size=spec.matched_size, matched_seed=spec.matched_seed,
        metal_columns=tuple(metal_columns),
        blocks=blocks_for_scheme(spec.metal_scheme, spec.massaction_scheme))


def assert_evaluation_unchanged(oof: pd.DataFrame, cohort: Gen7Cohort,
                                seeds: Sequence[int]) -> dict:
    """The rows scored must be gen10's rows, unchanged and complete.

    An auxiliary row that leaked into evaluation would show up here as an extra
    ``row_id`` or a changed target, which is why this is asserted rather than
    assumed.
    """
    expected = cohort.frame[["row_id", LEVEL_TARGET_COLUMN]].astype({"row_id": str})
    per_seed = int(len(expected))
    if len(oof) != per_seed * len(seeds):
        raise SystemExit(f"OOF has {len(oof)} rows, expected {per_seed * len(seeds)}")
    extra = set(oof["row_id"].astype(str)) - set(expected["row_id"])
    if extra:
        raise SystemExit(f"{len(extra)} evaluated rows are not cohort rows (e.g. {sorted(extra)[:3]})")
    merged = oof.merge(expected, on="row_id", how="left", suffixes=("", "_expected"))
    delta = (merged[LEVEL_TARGET_COLUMN] - merged[f"{LEVEL_TARGET_COLUMN}_expected"]).abs().max()
    if float(delta) > 0.0:
        raise SystemExit(f"evaluated target differs from the cohort's by {delta:.3g}")
    if oof["prediction"].isna().any():
        raise SystemExit("unpredicted rows in the OOF frame")
    return {"n_rows": int(len(oof)), "n_seeds": int(len(seeds)),
            "max_abs_target_delta": float(delta), "n_extra_rows": 0}


def run_spec(spec: RunSpec, cohort: Gen7Cohort, pool: AuxiliaryPool, out_dir: Path, *,
             seeds: Sequence[int] = DEFAULT_SEEDS, verbose: bool = True) -> pd.DataFrame:
    """Fit and score one gen11 arm; write its OOF parquet and diagnostics."""
    from .metalrep import REPRESENTATIONS

    patched, aux = attach_metal_scheme(cohort, pool.features, spec.metal_scheme,
                                       spec.massaction_scheme)
    # REPRESENTATIONS maps scheme -> (builder, columns); the controls only ever
    # permute the *columns*, never the builder.
    metal_columns = REPRESENTATIONS[spec.metal_scheme][1]
    model = build_model(spec, replace(pool, features=aux) if aux is not None else pool,
                        metal_columns=metal_columns)
    contender = ContenderAdapter(model, name=spec.name)
    started = time.time()
    oof = evaluate_contender(contender, patched, seeds=seeds, verbose=verbose)
    audit = assert_evaluation_unchanged(oof, patched, seeds)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = spec.name.replace("|", "__")
    oof.to_parquet(out_dir / f"oof_{safe}.parquet", index=False)
    pd.DataFrame(model.diagnostics).to_csv(out_dir / f"diagnostics_{safe}.csv", index=False)
    (out_dir / f"audit_{safe}.json").write_text(
        json.dumps({"spec": asdict(spec), "evaluation": audit,
                    "seconds": round(time.time() - started, 1)}, indent=2))
    if verbose:
        print(f"  [{spec.name}] {time.time() - started:.0f}s", flush=True)
    return oof


def leaderboard(frames: Sequence[pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    combined = pd.concat(list(frames), ignore_index=True)
    scores = score_oof(combined, per_seed=True)
    scores.to_csv(out_dir / "scores_by_seed.csv", index=False)
    board = scores.groupby("model")[["macro_mae", "offset_mae", "shape_mae", "pooled_mae"]].mean()
    board["macro_sd"] = scores.groupby("model")["macro_mae"].std()
    board["n_seeds"] = scores.groupby("model").size()
    board = board.sort_values("macro_mae")
    board.to_csv(out_dir / "leaderboard.csv")
    return board

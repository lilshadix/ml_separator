#!/usr/bin/env python
"""GEN11's reporting layer — every headline table regenerated from raw predictions.

The brief (§22) requires that every reported table can be rebuilt from the stored
OOF frames.  So this script reads *predictions*, not summaries: the leaderboard,
the bootstrap, the shape endpoints and the decomposition are all recomputed here
from ``runs/gen11_transfer/arms/oof_*.parquet`` with the same functions gen6/gen9
used, and the Markdown report is rendered from the CSVs this script has just
written.  No number in the report is typed in, and none is copied from another
run's summary file.  ``runs/gen11_transfer/control/gen10_control.json`` is the one
exception, and it is itself a re-derivation of gen10's headline numbers from
gen10's raw predictions rather than a transcription of its report.

**The framing this file exists to protect.**  Every auxiliary arm runs at a
changed design corner — ``GENERAL`` metal representation and ``ANNOTATION_SAFE``
MASSACTION — because the frozen design is unusable on auxiliary rows (98.4 % of
them have no frozen metal coordinate, and the two annotation-coupled MASSACTION
terms would act as an "is auxiliary" provenance flag).  A raw
``auxiliary arm − frozen gen10`` difference therefore confounds two things, and
the tables never report it without splitting it:

    design effect   = control@FROZEN_3/FROZEN_8  −  control@GENERAL/ANNOTATION_SAFE
    transfer effect = control@GENERAL/ANNOTATION_SAFE  −  auxiliary arm
    total           = design effect + transfer effect   (exactly, by construction)

Both are signed so that **positive means lower MAE, i.e. better**.  The
design-matched baseline for every arm is *derived* from the arm names present, not
declared, so a mis-typed constant cannot silently make the comparison the
flattering one.

**Absent inputs are recorded, never faked.**  A table whose inputs do not exist is
skipped and named in ``headline.json → skipped``; it does not appear as an empty
CSV that could be mistaken for a measurement.  The same rule governs the stopping
rule (§18): a criterion whose evidence has not been produced is ``NOT_EVALUABLE``
with the reason attached, which is a different statement from ``FAIL``.

Usage::

    PYTHONPATH=src python scripts/gen11_report_tables.py
    PYTHONPATH=src python scripts/gen11_report_tables.py --arms runs/gen11_transfer/arms
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen6.metrics import (  # noqa: E402
    decompose_level_shape, paired_unit_bootstrap, per_unit_statistics,
)
from lanthanide_separation.gen9.metrics import (  # noqa: E402
    curve_shape_table, macro_mae, summarise_shape,
)
from lanthanide_separation.gen11 import analysis  # noqa: E402
from lanthanide_separation.gen11.arms import ARM_BY_KEY, CONTROLS  # noqa: E402
from lanthanide_separation.gen11.runner import MASSACTION_SCHEMES, METAL_SCHEMES  # noqa: E402

GEN11 = REPO_ROOT / "runs" / "gen11_transfer"
OUT = GEN11 / "headline_tables"

#: The frozen design corner.  This is a *definition* (it is what "frozen" means),
#: not a result, so it is the one design fact stated rather than derived.
FROZEN_CORNER = ("FROZEN_3", "FROZEN_8")
#: The arm key that carries no auxiliary rows.
CONTROL_ARM_KEY = "A_GEN10_CONTROL"

#: gen10's re-derived control numbers, used only to say where gen11 started.
GEN10_CONTROL_JSON = GEN11 / "control" / "gen10_control.json"

#: §21's thirteen decision questions.
#:
#: The brief document is not present in this worktree, so the wording below is
#: **reconstructed** from the contracts the gen11 modules state in their own
#: docstrings (the section each question maps to is given).  The report renders
#: from this constant, so correcting a question here corrects the report; no
#: answer text is stored anywhere else.
SECTION_21_QUESTIONS: tuple[dict, ...] = (
    {"n": 1, "maps_to": "§1/§18A",
     "q": "Does the audited multi-metal archive improve zero-shot prediction for unseen "
          "lanthanide chemistry on the frozen gen10 benchmark?"},
    {"n": 2, "maps_to": "§3/§7",
     "q": "How much of any observed change is the design change (general metal representation "
          "and annotation-safe MASSACTION) rather than the auxiliary data?"},
    {"n": 3, "maps_to": "§5 B-E",
     "q": "Which auxiliary composition helps, if any — extra lanthanides (B), actinides (C), "
          "other metals (D), or everything (E)?"},
    {"n": 4, "maps_to": "§5 F/G",
     "q": "At equal row count, is any gain chemical relevance or merely row count "
          "(F actinides-matched vs G random-matched)?"},
    {"n": 5, "maps_to": "§6",
     "q": "Which transfer mechanism does the work — joint training, auxiliary pretraining, or a "
          "shared representation with a lanthanide-only head?"},
    {"n": 6, "maps_to": "§12",
     "q": "Does any effect survive the pre-registered weighting schemes and auxiliary mass "
          "settings, or is it a weighting artefact?"},
    {"n": 7, "maps_to": "§11",
     "q": "Does the effect sit in the per-ligand level or in the within-curve shape?"},
    {"n": 8, "maps_to": "§18D",
     "q": "Does the archive help most where the cohort is thinnest — on distant held-out "
          "chemotypes?"},
    {"n": 9, "maps_to": "§8/§18C",
     "q": "Does the archive improve curve shape on a scientifically important axis?"},
    {"n": 10, "maps_to": "§18B",
     "q": "Does the archive change the k = 0/1/2/3/5 few-shot frontier?"},
    {"n": 11, "maps_to": "§19",
     "q": "Do the negative controls (permuted auxiliary target, shuffled metal labels) stay "
          "flat, as they must for any gain to be real?"},
    {"n": 12, "maps_to": "§13",
     "q": "Does any result survive publication-blocked auxiliary admissibility and "
          "publication-blocked intervals?"},
    {"n": 13, "maps_to": "§18",
     "q": "Is there enough evidence to replace frozen gen10 with an auxiliary arm?"},
)


# --------------------------------------------------------------------------- #
# Arm names carry the whole design, so they are parsed rather than pattern-matched
# --------------------------------------------------------------------------- #

def parse_arm_name(name: str) -> dict:
    """Decompose ``RunSpec.name`` back into the design it encodes.

    ``RunSpec.name`` is ``arm|mechanism|metal|massaction|weighting|lam<x>`` plus
    optional ``draw<seed>``, control and policy tags.  Parsing it is what lets the
    design-matched baseline be *found* instead of declared: an arm is compared to
    the control that shares its ``(metal_scheme, massaction_scheme)`` corner.
    """
    parts = name.split("|")
    if len(parts) < 6:
        raise ValueError(f"unparseable arm name {name!r}")
    arm, mechanism, metal, massaction, weighting, lam = parts[:6]
    if metal not in METAL_SCHEMES or massaction not in MASSACTION_SCHEMES:
        raise ValueError(f"arm name {name!r} has an unknown design corner")
    record = {"model": name, "arm": arm, "mechanism": mechanism, "metal_scheme": metal,
              "massaction_scheme": massaction, "weighting": weighting,
              "aux_lambda": float(lam[3:]) if lam.startswith("lam") else np.nan,
              "matched_seed": np.nan, "control": "NONE", "policy": "HEADLINE"}
    for extra in parts[6:]:
        if extra.startswith("draw"):
            record["matched_seed"] = float(extra[4:])
        elif extra in CONTROLS:
            record["control"] = extra
        else:
            record["policy"] = extra
    record["design_corner"] = f"{metal}|{massaction}"
    record["carries_auxiliary"] = arm != CONTROL_ARM_KEY
    if record["control"] != "NONE":
        kind = "NEGATIVE_CONTROL"
    elif arm == CONTROL_ARM_KEY:
        kind = "DESIGN_CONTROL"
    else:
        kind = "AUXILIARY"
    record["arm_kind"] = kind
    return record


def load_arms(arms_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(arms_dir.glob("oof_*.parquet")):
        frame = pd.read_parquet(path)
        for name, block in frame.groupby("model"):
            frames[str(name)] = block.reset_index(drop=True)
    return frames


def design_table(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([parse_arm_name(name) for name in sorted(frames)])


def baseline_for(design: pd.DataFrame, model: str) -> str | None:
    """The control run at ``model``'s own design corner, or ``None`` if unrun."""
    corner = design.loc[design["model"] == model, "design_corner"].iloc[0]
    match = design[(design["arm"] == CONTROL_ARM_KEY) & (design["design_corner"] == corner)
                   & (design["control"] == "NONE")]
    return None if match.empty else str(match["model"].iloc[0])


def frozen_anchor(design: pd.DataFrame) -> str | None:
    corner = "|".join(FROZEN_CORNER)
    match = design[(design["arm"] == CONTROL_ARM_KEY) & (design["design_corner"] == corner)]
    return None if match.empty else str(match["model"].iloc[0])


def key_signature(frame: pd.DataFrame) -> tuple:
    """What must match for two arms to be paired row-for-row."""
    return (int(len(frame)), tuple(sorted(frame["split_seed"].unique().tolist())),
            int(frame["row_id"].nunique()))


# --------------------------------------------------------------------------- #
# T2 — leaderboard
# --------------------------------------------------------------------------- #

def per_seed_scores(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Macro / offset / shape / pooled MAE for every (arm, split seed).

    ``macro_mae`` is gen9's — one ECFP cluster, one vote — and the level/shape
    split is gen6's :func:`decompose_level_shape`, so a gen11 number and a gen10
    number are the same arithmetic.
    """
    rows = []
    for name, frame in frames.items():
        for seed, block in frame.groupby("split_seed"):
            summary = decompose_level_shape(block["log_D"], block["prediction"],
                                            block["extractant"]).summary
            error = (block["prediction"] - block["log_D"]).abs()
            rows.append({
                "model": name, "split_seed": int(seed), "n_rows": int(len(block)),
                "macro_mae": macro_mae(block),
                "offset_mae": float(summary["offset_mae"]),
                "shape_mae": float(summary["shape_mae"]),
                "pooled_mae": float(error.mean()),
                "macro_mae_ligand": float(summary["macro_mae_ligand"]),
            })
    return pd.DataFrame(rows).sort_values(["model", "split_seed"], ignore_index=True)


def pooled_unit_scores(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Seed-pooled per-ECFP-unit statistics — the quantity the bootstrap resamples.

    Reported beside the seed-mean so that T3's point delta reconciles exactly with
    a difference of two T2 columns; the seed-mean and the seed-pooled macro are
    both legitimate and are ~0.001 apart, and quoting one while bootstrapping the
    other is how a table stops adding up.
    """
    rows = []
    for name, frame in frames.items():
        per_unit = per_unit_statistics(
            frame.rename(columns={"prediction": "prediction_arm"}), ["arm"])
        rows.append({"model": name,
                     "macro_mae_pooled_seeds": float(per_unit["mae"].mean()),
                     "offset_mae_pooled_seeds": float(per_unit["offset_mae"].mean()),
                     "shape_mae_pooled_seeds": float(per_unit["shape_mae"].mean()),
                     "n_units": int(len(per_unit))})
    return pd.DataFrame(rows)


def build_t2(frames: Mapping[str, pd.DataFrame], design: pd.DataFrame) -> pd.DataFrame:
    """Per-arm leaderboard with the design/transfer split built in."""
    per_seed = per_seed_scores(frames)
    metrics = ["macro_mae", "offset_mae", "shape_mae", "pooled_mae"]
    mean = per_seed.groupby("model")[metrics].mean()
    sd = per_seed.groupby("model")[metrics].std().add_suffix("_sd")
    board = mean.join(sd)
    board["n_seeds"] = per_seed.groupby("model")["split_seed"].nunique()
    board = board.reset_index().merge(pooled_unit_scores(frames), on="model", how="left")
    board = design.merge(board, on="model", how="left")

    anchor = frozen_anchor(design)
    anchor_macro = float(board.loc[board["model"] == anchor, "macro_mae"].iloc[0]) \
        if anchor else np.nan

    baselines, base_macro, design_effect, transfer_effect, total = [], [], [], [], []
    for model in board["model"]:
        base = baseline_for(design, model)
        baselines.append(base)
        b = float(board.loc[board["model"] == base, "macro_mae"].iloc[0]) if base else np.nan
        m = float(board.loc[board["model"] == model, "macro_mae"].iloc[0])
        base_macro.append(b)
        design_effect.append(anchor_macro - b)
        # A control *is* the baseline at its own corner.  "Transfer effect 0.0"
        # would read as a measured null; it is not a measurement at all.
        transfer_effect.append(np.nan if base == model else b - m)
        total.append(anchor_macro - m)
    board["design_matched_baseline"] = baselines
    board["baseline_macro_mae"] = base_macro
    board["design_effect_macro"] = design_effect
    board["transfer_effect_macro"] = transfer_effect
    board["total_vs_frozen_macro"] = total
    board["frozen_anchor_macro_mae"] = anchor_macro

    # Per-seed direction: an effect positive on average but negative in two of five
    # seeds is not the same claim as one positive everywhere (§17).
    improved = []
    for model, base in zip(board["model"], baselines):
        if base is None or base == model:
            improved.append(np.nan)
            continue
        a = per_seed[per_seed["model"] == base].set_index("split_seed")["macro_mae"]
        b = per_seed[per_seed["model"] == model].set_index("split_seed")["macro_mae"]
        common = a.index.intersection(b.index)
        improved.append(int(((a[common] - b[common]) > 0).sum()))
    board["n_seeds_improved_vs_baseline"] = improved
    return per_seed, board.sort_values(["arm_kind", "macro_mae"], ignore_index=True)


# --------------------------------------------------------------------------- #
# T3 — paired bootstrap
# --------------------------------------------------------------------------- #

def _pairwise_bootstrap(frames: Mapping[str, pd.DataFrame], reference: str,
                        candidates: Sequence[str], *, block: str, replicates: int,
                        kind: str) -> pd.DataFrame:
    subset = {name: frames[name] for name in [reference, *candidates]}
    # ``wide_predictions`` inner-joins, so a candidate that lost rows would shrink
    # the evaluation set for every arm in the call and flatter all of them at once.
    # Checked here, because the shrunken frame is otherwise invisible downstream.
    paired = len(analysis.wide_predictions(subset))
    if paired != len(frames[reference]):
        raise SystemExit(
            f"pairing {reference} with {list(candidates)} left {paired} of "
            f"{len(frames[reference])} rows; the arms are not scored on the same rows")
    table = analysis.compare(subset, reference=reference, candidates=list(candidates),
                             block=block, replicates=replicates)
    if table.empty:
        return table
    table.insert(0, "comparison_kind", kind)
    table.insert(1, "block_unit", block)
    table.insert(2, "n_rows_paired", paired)
    return table


def build_t3(frames: Mapping[str, pd.DataFrame], design: pd.DataFrame, *,
             replicates: int, block: str = "tanimoto_cluster") -> pd.DataFrame:
    """Chemotype-blocked paired bootstrap, split into TRANSFER and DESIGN rows.

    A ``TRANSFER`` row compares an arm to the control at its *own* design corner;
    a ``DESIGN`` row compares a control corner to the frozen anchor.  Reading only
    the first would attribute the design change to the auxiliary data, and reading
    only ``candidate − frozen anchor`` would attribute the auxiliary data's effect
    to the design.  Percentile, BCa and cluster-robust intervals all travel with
    the row because gen6 measured the percentile one at ~12.7 % one-sided Type-I
    error on this exact unit/block mismatch.
    """
    parts: list[pd.DataFrame] = []
    signatures = {name: key_signature(frame) for name, frame in frames.items()}

    # TRANSFER: candidates grouped by the baseline they are matched to.
    for base in sorted({b for b in (baseline_for(design, m) for m in design["model"]) if b}):
        candidates = [m for m in design["model"]
                      if m != base and baseline_for(design, m) == base
                      and signatures[m] == signatures[base]]
        if candidates:
            parts.append(_pairwise_bootstrap(frames, base, candidates, block=block,
                                             replicates=replicates, kind="TRANSFER"))

    # DESIGN: every control corner against the frozen anchor.
    anchor = frozen_anchor(design)
    if anchor is not None:
        corners = [m for m in design.loc[design["arm"] == CONTROL_ARM_KEY, "model"]
                   if m != anchor and signatures[m] == signatures[anchor]]
        if corners:
            parts.append(_pairwise_bootstrap(frames, anchor, corners, block=block,
                                             replicates=replicates, kind="DESIGN"))
        # TOTAL: auxiliary arms against the frozen anchor, reported only so the
        # additive identity can be checked, never as the transfer claim.
        aux = [m for m in design.loc[design["carries_auxiliary"], "model"]
               if signatures[m] == signatures[anchor]]
        if aux:
            parts.append(_pairwise_bootstrap(frames, anchor, aux, block=block,
                                             replicates=replicates,
                                             kind="TOTAL_CONFOUNDED"))

    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    table = pd.concat(parts, ignore_index=True)
    low = table.get("bca_low", pd.Series(np.nan, index=table.index))
    high = table.get("bca_high", pd.Series(np.nan, index=table.index))
    # "Significant" means the interval excludes zero in *either* direction.  A
    # design change that significantly hurts is as much a finding as one that
    # helps, and a one-sided flag would hide it.
    table["significant_bca"] = (low > 0) | (high < 0)
    table["bca_direction"] = np.where(low > 0, "candidate better",
                                      np.where(high < 0, "candidate worse", "not separated"))
    return table


def band_bootstrap(frames: Mapping[str, pd.DataFrame], reference: str, candidate: str, *,
                   band: str, replicates: int, band_scheme: str = "GEN11_ANALYSIS",
                   cuts: Mapping[str, float] | None = None) -> dict:
    """Paired bootstrap restricted to one chemotype-distance band.

    Criterion D asks for a *statistically supported* improvement on distant
    chemotypes, which a stratified point estimate cannot supply on its own.
    """
    wide = analysis.wide_predictions({reference: frames[reference], candidate: frames[candidate]})
    labels = assign_bands(wide["nn_train_tanimoto"], band_scheme, cuts)
    part = wide[labels == band]
    if part.empty or part["ecfp_cluster"].nunique() < 3:
        return {"band": band, "band_scheme": band_scheme, "n_rows": int(len(part)),
                "n_units": int(part["ecfp_cluster"].nunique()) if len(part) else 0,
                "status": "TOO_FEW_UNITS"}
    per_unit = per_unit_statistics(part, [reference, candidate])
    block_of_unit = (part.drop_duplicates("ecfp_cluster")
                     .set_index("ecfp_cluster")["tanimoto_cluster"].astype(str).to_dict())
    boot = paired_unit_bootstrap(per_unit, {"band": (reference, candidate)},
                                 block_of_unit=block_of_unit, replicates=replicates)
    row = boot[boot["statistic"] == "mae"].iloc[0].to_dict()
    row.update({"band": band, "band_scheme": band_scheme, "n_rows": int(len(part)),
                "n_units": int(part["ecfp_cluster"].nunique()), "status": "OK"})
    return row


# --------------------------------------------------------------------------- #
# Chemotype-distance bands
# --------------------------------------------------------------------------- #

def gen10_tercile_cuts() -> dict | None:
    """gen10's own equal-frequency distance terciles, as composition recomputed them.

    ``gen11.analysis.CHEMOTYPE_BANDS`` states cut points of 0.4 / 0.6 and describes
    them as gen10's; ``composition.py`` recomputes gen10's terciles from
    ``runs/gen10_final/budget_simulation/adaptation_curves.csv`` and gets
    0.588 / 0.657.  The two disagree, so both are reported and neither is silently
    preferred.
    """
    path = GEN11 / "composition" / "composition_audit.json"
    if not path.exists():
        return None
    buckets = json.loads(path.read_text()).get("distance_buckets", {})
    if "near_min_tanimoto" not in buckets or "mid_min_tanimoto" not in buckets:
        return None
    return {"near": float(buckets["near_min_tanimoto"]),
            "mid": float(buckets["mid_min_tanimoto"])}


def assign_bands(similarity: pd.Series, scheme: str, cuts: Mapping[str, float] | None) -> pd.Series:
    if scheme == "GEN11_ANALYSIS":
        return analysis.band_of(similarity)
    if cuts is None:
        raise ValueError("GEN10_TERCILE bands need recomputed cut points")
    out = pd.Series("far", index=similarity.index, dtype=object)
    out[similarity >= cuts["mid"]] = "mid"
    out[similarity >= cuts["near"]] = "near"
    return out


# --------------------------------------------------------------------------- #
# T4 — shape by axis
# --------------------------------------------------------------------------- #

def _prediction_digest(frame: pd.DataFrame) -> str:
    """A content hash of one arm's predictions, for cache validity.

    Keyed on the predictions themselves rather than on a file timestamp, so a
    refitted arm can never be served from a stale cache entry.
    """
    keyed = frame[["row_id", "split_seed", "prediction"]].sort_values(["split_seed", "row_id"])
    return str(pd.util.hash_pandas_object(keyed, index=False).sum())


def build_t4(frames: Mapping[str, pd.DataFrame], design: pd.DataFrame,
             cache_dir: Path | None = None) -> tuple:
    """gen9's eight shape endpoints per arm per axis, averaged over seeds.

    Curve reconstruction is the slowest step in this script and the report is
    re-run every time another arm lands, so a per-arm cache keyed on the
    predictions' content is kept; it changes nothing about the numbers.
    """
    from lanthanide_separation.gen9.train import load_membership

    membership = load_membership()
    rows, curves = [], []
    for name, frame in frames.items():
        digest = _prediction_digest(frame)
        cached = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached = cache_dir / f"shape_{digest}.parquet"
            if cached.exists():
                table = pd.read_parquet(cached)
                curves.append(table)
                for seed, block in table.groupby("split_seed"):
                    summary = summarise_shape(block)
                    summary.insert(0, "split_seed", int(seed))
                    summary.insert(0, "model", name)
                    rows.append(summary)
                continue
        parts = []
        for seed, block in frame.groupby("split_seed"):
            table = curve_shape_table(block.drop_duplicates("row_id"), membership)
            if table.empty:
                continue
            table = table.assign(model=name, split_seed=int(seed))
            parts.append(table)
            summary = summarise_shape(table)
            summary.insert(0, "split_seed", int(seed))
            summary.insert(0, "model", name)
            rows.append(summary)
        if parts:
            combined = pd.concat(parts, ignore_index=True)
            curves.append(combined)
            if cached is not None:
                combined.to_parquet(cached, index=False)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    per_seed = pd.concat(rows, ignore_index=True)
    mean = per_seed.groupby(["model", "axis_label"]).mean(numeric_only=True).reset_index()
    mean = mean.drop(columns=[c for c in ("split_seed",) if c in mean.columns])

    keep = ["model", "axis_label", "n_curves", "n_ligands", "shape_mae", "slope_mae",
            "slope_true_median", "slope_pred_median", "span_recovery_median",
            "within_curve_spearman", "within_curve_sign_accuracy"]
    mean = mean[[c for c in keep if c in mean.columns]]
    mean = design[["model", "arm", "arm_kind", "design_corner"]].merge(mean, on="model")

    baseline_of = {m: baseline_for(design, m) for m in design["model"]}
    lookup = mean.set_index(["model", "axis_label"])["shape_mae"].to_dict()
    mean["design_matched_baseline"] = mean["model"].map(baseline_of)
    mean["baseline_shape_mae"] = [
        lookup.get((base, axis), np.nan) if base else np.nan
        for base, axis in zip(mean["design_matched_baseline"], mean["axis_label"])]
    mean["shape_improvement_vs_baseline"] = np.where(
        mean["design_matched_baseline"] == mean["model"], np.nan,
        mean["baseline_shape_mae"] - mean["shape_mae"])
    anchor = frozen_anchor(design)
    mean["frozen_anchor_shape_mae"] = [lookup.get((anchor, axis), np.nan) if anchor else np.nan
                                       for axis in mean["axis_label"]]
    mean["shape_improvement_vs_frozen"] = mean["frozen_anchor_shape_mae"] - mean["shape_mae"]
    return mean, pd.concat(curves, ignore_index=True)


# --------------------------------------------------------------------------- #
# T5 — k-shot frontier
# --------------------------------------------------------------------------- #

def find_kshot(root: Path) -> tuple[Path | None, str]:
    """Locate a gen11 k-shot artefact, preferring the raw per-ligand detail."""
    for candidate in sorted(root.rglob("kshot_detail.parquet")):
        return candidate, "detail"
    for candidate in sorted(root.rglob("frontier_common.csv")):
        return candidate, "frontier"
    return None, "missing"


def build_t5(path: Path, kind: str, design: pd.DataFrame) -> pd.DataFrame:
    """The k = 0/1/2/3/5 frontier, rebuilt from the k-shot detail when it exists.

    Rebuilt with gen10's aggregation — mean over (seed, fold, repeat) per ligand,
    then mean over the common cohort's ligands — so a gen11 frontier number and a
    gen10 one are comparable.
    """
    if kind == "detail":
        detail = pd.read_parquet(path)
        detail["arm"] = np.where(detail["adapter"] == "ZERO_SHOT_REF", "ZERO_SHOT",
                                 detail["adapter"] + "@" + detail["policy"])
        ok = detail.groupby("extractant").apply(
            lambda b: bool((b["n_pool"] >= 5).all() and (b["n_eval"] >= 2).all()),
            include_groups=False)
        common = detail[detail["extractant"].isin(set(ok.index[ok]))]
        # gen10's k-shot detail carries both ``model`` and ``global_model``; the
        # latter is the arm whose predictions were adapted, which is the one that
        # names a gen11 arm.
        key = "global_model" if "global_model" in common.columns else "model"
        common = common.drop(columns=[c for c in ("model",) if c != key and c in common.columns])
        per_ligand = common.groupby([key, "arm", "adapter", "policy", "k", "extractant"])[
            "mae"].mean().reset_index()
        table = per_ligand.groupby([key, "arm", "adapter", "policy", "k"]).agg(
            mae=("mae", "mean"), n_ligands=("extractant", "nunique")).reset_index()
        table = table.rename(columns={key: "model"})
    else:
        table = pd.read_csv(path)
        if "global_model" in table.columns:
            table = table.drop(columns=[c for c in ("model",) if c in table.columns])
            table = table.rename(columns={"global_model": "model"})

    table["source"] = str(path.relative_to(REPO_ROOT))
    table["regenerated_from_raw"] = kind == "detail"
    known = set(design["model"])
    table["model_is_gen11_arm"] = table["model"].isin(known)
    if table["model_is_gen11_arm"].any():
        merged = design[["model", "arm", "arm_kind", "design_corner"]].merge(
            table, on="model", how="right")
        return merged
    return table


# --------------------------------------------------------------------------- #
# T6 — near / mid / far, level vs shape
# --------------------------------------------------------------------------- #

def build_t6(frames: Mapping[str, pd.DataFrame], design: pd.DataFrame,
             aux_extractants: Sequence[str]) -> pd.DataFrame:
    """Where a difference sits: distance band, aux-coverage, and level vs shape.

    Computed under both band definitions (see :func:`gen10_tercile_cuts`) because
    the two in the repository disagree, and the far stratum is what criterion D
    turns on.
    """
    cuts = gen10_tercile_cuts()
    aux_set = set(aux_extractants)
    records = []
    for model in design["model"]:
        base = baseline_for(design, model)
        if base is None or base == model:
            continue
        if key_signature(frames[model]) != key_signature(frames[base]):
            continue
        wide = analysis.wide_predictions({base: frames[base], model: frames[model]})
        wide["extractant_in_aux"] = wide["extractant"].isin(aux_set)
        for arm in (base, model):
            residual = wide[f"prediction_{arm}"] - wide["log_D"]
            level = residual.groupby([wide["split_seed"], wide["extractant"]]).transform("mean")
            wide[f"abs_{arm}"] = residual.abs()
            wide[f"level_{arm}"] = level.abs()
            wide[f"shape_{arm}"] = (residual - level).abs()

        schemes = [("GEN11_ANALYSIS", cuts)] + ([("GEN10_TERCILE", cuts)] if cuts else [])
        for scheme, scheme_cuts in schemes:
            wide["band"] = assign_bands(wide["nn_train_tanimoto"], scheme, scheme_cuts)
            strata = [("band", g) for g in wide.groupby("band")]
            strata += [("extractant_in_aux", g) for g in wide.groupby("extractant_in_aux")]
            strata += [("ALL", ("ALL", wide))]
            for kind, (label, part) in strata:
                if kind != "band" and scheme != schemes[0][0]:
                    continue  # aux-coverage and ALL do not depend on the band scheme
                row = {"candidate": model, "reference": base,
                       "arm": design.loc[design["model"] == model, "arm"].iloc[0],
                       "band_scheme": scheme if kind == "band" else "N/A",
                       "stratum_kind": kind, "stratum": str(label),
                       "n_rows": int(len(part)),
                       "n_ligands": int(part["extractant"].nunique()),
                       "n_units": int(part["ecfp_cluster"].nunique())}
                for component in ("abs", "level", "shape"):
                    ref = part.groupby("ecfp_cluster")[f"{component}_{base}"].mean().mean()
                    cand = part.groupby("ecfp_cluster")[f"{component}_{model}"].mean().mean()
                    row[f"{component}_reference"] = float(ref)
                    row[f"{component}_candidate"] = float(cand)
                    row[f"{component}_improvement"] = float(ref - cand)
                records.append(row)
    if not records:
        return pd.DataFrame()
    table = pd.DataFrame(records)
    if cuts:
        table["gen10_tercile_near_min_tanimoto"] = cuts["near"]
        table["gen10_tercile_mid_min_tanimoto"] = cuts["mid"]
    table["gen11_analysis_bands"] = str(analysis.CHEMOTYPE_BANDS)
    return table


# --------------------------------------------------------------------------- #
# T7 — negative controls
# --------------------------------------------------------------------------- #

def build_t7(design: pd.DataFrame, board: pd.DataFrame, t3: pd.DataFrame,
             *, threshold: float) -> pd.DataFrame:
    """The §19 controls, and whether each one stayed flat.

    A negative control that clears the same bar the real arm must clear is not a
    small anomaly — it means the bar is measuring something other than the
    auxiliary information, so the verdict column says ``ARTEFACT_WARNING`` rather
    than reporting a gain.
    """
    controls = design[design["control"] != "NONE"]
    if controls.empty:
        return pd.DataFrame()
    macro = board.set_index("model")["macro_mae"].to_dict()
    seeds_improved = board.set_index("model")["n_seeds_improved_vs_baseline"].to_dict()
    n_seeds = board.set_index("model")["n_seeds"].to_dict()
    boot = None
    if not t3.empty and "comparison_kind" in t3.columns:
        transfer = t3[(t3["statistic"] == "mae") & (t3["comparison_kind"] == "TRANSFER")]
        boot = transfer.drop_duplicates("candidate").set_index("candidate")

    rows = []
    for _, control in controls.iterrows():
        model = control["model"]
        base = baseline_for(design, model)
        twin_parts = [control["arm"], control["mechanism"], control["metal_scheme"],
                      control["massaction_scheme"], control["weighting"],
                      f"lam{control['aux_lambda']:g}"]
        if control["policy"] != "HEADLINE":
            twin_parts.append(control["policy"])
        twin = "|".join(twin_parts)
        improvement = (macro.get(base, np.nan) - macro.get(model, np.nan)) if base else np.nan
        bca_low = float(boot.loc[model, "bca_low"]) if boot is not None and model in boot.index \
            and "bca_low" in boot.columns else np.nan
        benefit = (np.isfinite(improvement) and improvement >= threshold) or \
                  (np.isfinite(bca_low) and bca_low > 0)
        rows.append({
            "control_arm": model, "control": control["control"], "arm": control["arm"],
            "design_matched_baseline": base,
            "uncorrupted_twin": twin if twin in macro else None,
            "baseline_macro_mae": macro.get(base, np.nan),
            "control_macro_mae": macro.get(model, np.nan),
            "twin_macro_mae": macro.get(twin, np.nan),
            "control_improvement_vs_baseline": improvement,
            "twin_improvement_vs_baseline": (macro.get(base, np.nan) - macro.get(twin, np.nan))
            if base and twin in macro else np.nan,
            "bca_low": bca_low,
            "n_seeds_improved": seeds_improved.get(model, np.nan),
            "n_seeds": n_seeds.get(model, np.nan),
            "flat_threshold": threshold,
            "verdict": "ARTEFACT_WARNING" if benefit else "FLAT",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# §18 — the stopping rule, evaluated mechanically
# --------------------------------------------------------------------------- #

def _row(arm: str, kind: str, criterion: str, *, quantity: str, value, threshold,
         requirement: str, requirement_met, status: str, evidence: str, reason: str = "") -> dict:
    return {"arm": arm, "arm_kind": kind, "criterion": criterion,
            "description": analysis.STOPPING_RULE[criterion]["description"],
            "measured_quantity": quantity,
            "measured_value": float(value) if value is not None and np.isfinite(
                np.asarray(value, dtype=float)) else np.nan,
            "threshold": threshold, "requirement": requirement,
            "requirement_met": requirement_met, "status": status,
            "evidence_table": evidence, "reason": reason}


def build_stopping_rule(design: pd.DataFrame, board: pd.DataFrame, t3: pd.DataFrame,
                        t4: pd.DataFrame, t5: pd.DataFrame, t6: pd.DataFrame,
                        far_boot: pd.DataFrame, *, consistency_available: bool) -> pd.DataFrame:
    """§18 A-E per arm.  ``NOT_EVALUABLE`` is used only where evidence is absent."""
    macro = board.set_index("model")
    rows: list[dict] = []

    for _, arm in design.iterrows():
        model, kind = arm["model"], arm["arm_kind"]
        if kind == "DESIGN_CONTROL":
            # A control carries no auxiliary rows, so §18 — which asks whether the
            # *archive* earned adoption — has nothing to score on it.  Said once per
            # criterion rather than by omission, so the table's shape is the same
            # for every arm and a reader can see the rule was applied.
            for criterion, rule in analysis.STOPPING_RULE.items():
                rows.append(_row(model, kind, criterion, quantity="not applicable",
                                 value=np.nan, threshold=rule["min_improvement"],
                                 requirement=rule["requires"], requirement_met=None,
                                 status="NOT_APPLICABLE", evidence="t2_leaderboard.csv",
                                 reason="this arm carries no auxiliary rows; it is a design "
                                        "control, and §18 governs auxiliary arms"))
            continue
        base = baseline_for(design, model)
        stats = macro.loc[model] if model in macro.index else None

        # --- A ------------------------------------------------------------ #
        if base is None or stats is None or not np.isfinite(stats.get("transfer_effect_macro", np.nan)):
            rows.append(_row(model, kind, "A_ZERO_SHOT", quantity="macro MAE improvement vs "
                             "design-matched control", value=np.nan, threshold=0.02,
                             requirement="consistent seed direction", requirement_met=None,
                             status="NOT_EVALUABLE", evidence="t2_leaderboard.csv",
                             reason="the control at this arm's design corner has not been run"))
        else:
            improvement = float(stats["transfer_effect_macro"])
            n_improved = float(stats["n_seeds_improved_vs_baseline"])
            n_seeds = float(stats["n_seeds"])
            consistent = bool(np.isfinite(n_improved) and n_improved == n_seeds)
            passed = improvement >= 0.02 and consistent
            rows.append(_row(model, kind, "A_ZERO_SHOT",
                             quantity="macro MAE improvement vs design-matched control",
                             value=improvement, threshold=0.02,
                             requirement=f"all {int(n_seeds)} seeds improve",
                             requirement_met=f"{int(n_improved)}/{int(n_seeds)}"
                             if np.isfinite(n_improved) else "unknown",
                             status="PASS" if passed else "FAIL",
                             evidence="t2_leaderboard.csv, t3_paired_bootstrap.csv"))

        # --- B ------------------------------------------------------------ #
        if t5.empty or model not in set(t5.get("model", pd.Series(dtype=str))):
            rows.append(_row(model, kind, "B_FEW_SHOT",
                             quantity="macro MAE improvement at k in {1,2,3,5}", value=np.nan,
                             threshold=0.01, requirement="improvement at >= 2 k, none degraded",
                             requirement_met=None, status="NOT_EVALUABLE",
                             evidence="t5_kshot.csv",
                             reason="no k-shot frontier has been produced for this arm"))
        else:
            base_rows = t5[(t5["model"] == base)] if base else t5.iloc[:0]
            arm_rows = t5[t5["model"] == model]
            improvements = {}
            for k in (1, 2, 3, 5):
                b = base_rows[base_rows["k"] == k]["mae"].min()
                a = arm_rows[arm_rows["k"] == k]["mae"].min()
                if np.isfinite(b) and np.isfinite(a):
                    improvements[k] = float(b - a)
            if not improvements:
                # The arm has a frontier but its design-matched control does not, so
                # there is nothing to subtract.  That is missing evidence, not a fail.
                rows.append(_row(model, kind, "B_FEW_SHOT",
                                 quantity="macro MAE improvement at k in {1,2,3,5}",
                                 value=np.nan, threshold=0.01,
                                 requirement="improvement at >= 2 k, none degraded",
                                 requirement_met=None, status="NOT_EVALUABLE",
                                 evidence="t5_kshot.csv",
                                 reason="no k-shot frontier for this arm's design-matched "
                                        "control, so no paired k-shot difference exists"))
            else:
                n_over = sum(1 for v in improvements.values() if v >= 0.01)
                degraded = [k for k, v in improvements.items() if v < -0.01]
                passed = n_over >= 2 and not degraded
                rows.append(_row(model, kind, "B_FEW_SHOT",
                                 quantity="best-arm macro MAE improvement at k in "
                                          f"{sorted(improvements)}",
                                 value=min(improvements.values()), threshold=0.01,
                                 requirement="improvement >= 0.01 at >= 2 k, no k degraded "
                                             "by > 0.01",
                                 requirement_met=f"{n_over} k over threshold; degraded at "
                                                 f"{degraded}",
                                 status="PASS" if passed else "FAIL", evidence="t5_kshot.csv"))

        # --- C ------------------------------------------------------------ #
        axis_rows = t4[t4["model"] == model] if not t4.empty else pd.DataFrame()
        if axis_rows.empty or axis_rows["shape_improvement_vs_baseline"].isna().all():
            rows.append(_row(model, kind, "C_SHAPE",
                             quantity="best per-axis shape MAE improvement", value=np.nan,
                             threshold=0.10, requirement="macro non-worse", requirement_met=None,
                             status="NOT_EVALUABLE", evidence="t4_shape_by_axis.csv",
                             reason="no curve-shape table for this arm at its design corner"))
        else:
            best = axis_rows.loc[axis_rows["shape_improvement_vs_baseline"].idxmax()]
            macro_ok = bool(stats is not None
                            and float(stats.get("transfer_effect_macro", np.nan)) >= 0)
            passed = float(best["shape_improvement_vs_baseline"]) >= 0.10 and macro_ok
            rows.append(_row(model, kind, "C_SHAPE",
                             quantity=f"shape MAE improvement on axis {best['axis_label']!r} "
                                      f"({int(best['n_curves'])} curves)",
                             value=float(best["shape_improvement_vs_baseline"]), threshold=0.10,
                             requirement="macro MAE non-worse than the design-matched control",
                             requirement_met=macro_ok, status="PASS" if passed else "FAIL",
                             evidence="t4_shape_by_axis.csv"))

        # --- D ------------------------------------------------------------ #
        far = far_boot[(far_boot["candidate"] == model)] if not far_boot.empty else pd.DataFrame()
        far = far[far["band_scheme"] == "GEN11_ANALYSIS"] if not far.empty else far
        if far.empty or far.iloc[0].get("status") != "OK":
            reason = ("no design-matched control for this arm" if base is None else
                      "too few independent chemotype units in the far band to bootstrap")
            rows.append(_row(model, kind, "D_FAR_CHEMOTYPE",
                             quantity="macro MAE improvement on far chemotypes", value=np.nan,
                             threshold="statistically supported (BCa low > 0)",
                             requirement="overall benchmark not degraded", requirement_met=None,
                             status="NOT_EVALUABLE", evidence="t6_decomposition.csv", reason=reason))
        else:
            record = far.iloc[0]
            supported = bool(np.isfinite(record.get("bca_low", np.nan))
                             and record["bca_low"] > 0)
            not_degraded = bool(stats is not None
                                and float(stats.get("transfer_effect_macro", np.nan)) >= 0)
            passed = supported and not_degraded
            rows.append(_row(model, kind, "D_FAR_CHEMOTYPE",
                             quantity=f"far-band macro MAE improvement "
                                      f"({int(record['n_units'])} units, "
                                      f"{int(record['n_rows'])} rows)",
                             value=float(record["point_delta"]),
                             threshold="BCa 95 % lower bound > 0",
                             requirement="overall macro not degraded",
                             requirement_met=not_degraded,
                             status="PASS" if passed else "FAIL",
                             evidence="t6_decomposition.csv, t3_paired_bootstrap.csv"))

        # --- E ------------------------------------------------------------ #
        rows.append(_row(model, kind, "E_ROBUSTNESS",
                         quantity="query-design sensitivity at matched accuracy", value=np.nan,
                         threshold="lower than the control", requirement="matched accuracy",
                         requirement_met=None,
                         status="NOT_EVALUABLE" if not consistency_available else "FAIL",
                         evidence="runs/gen11_transfer/consistency/",
                         reason="" if consistency_available else
                                "the query-consistency run has not been produced for gen11 arms"))
    table = pd.DataFrame(rows)
    if not table.empty:
        table["negative_control_note"] = np.where(
            table["arm_kind"] == "NEGATIVE_CONTROL",
            "a PASS here is an artefact warning, not a success", "")
    return table


# --------------------------------------------------------------------------- #
# The decision report, rendered from the tables
# --------------------------------------------------------------------------- #

def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not np.isfinite(number) else f"{number:.{digits}f}"


def _cell(value, digits: int, *, as_int: bool = False) -> str:
    """One Markdown cell.

    Arm names contain ``|``, which is the Markdown column separator, so escaping
    is not cosmetic: without it every table in the report silently loses columns.
    """
    if isinstance(value, (bool, np.bool_)):
        text = str(bool(value))
    elif isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        text = f"{int(number)}" if as_int and np.isfinite(number) else _fmt(number, digits)
    else:
        text = str(value)
    return text.replace("|", "\\|")


def _integral_columns(frame: pd.DataFrame, columns: Sequence[str]) -> set:
    """Columns whose every value is a whole number — counts, rendered as counts.

    Decided from the data rather than from column names, so a count added later
    does not print as ``4896.0000`` because nobody updated a list.
    """
    out = set()
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) and (finite % 1 == 0).all():
            out.add(column)
    return out


def _md_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 4) -> str:
    columns = [c for c in columns if c in frame.columns]
    if not columns or frame.empty:
        return "_no rows_\n"
    integral = _integral_columns(frame, columns)
    lines = ["| " + " | ".join(c.replace("|", "\\|") for c in columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_cell(row[c], digits, as_int=c in integral)
                                       for c in columns) + " |")
    return "\n".join(lines) + "\n"


def render_report(out_dir: Path, headline: dict) -> str:
    """Build ``GEN11_DECISION_REPORT.md`` by reading back the CSVs just written.

    Reading from disk rather than from the in-memory frames is deliberate: the
    report can then only say what a table says, and a table that was skipped
    produces a "not answerable" answer rather than a silently stale sentence.
    """
    def table(name: str) -> pd.DataFrame:
        path = out_dir / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    t1, t2, t3 = table("t1_arm_composition.csv"), table("t2_leaderboard.csv"), \
        table("t3_paired_bootstrap.csv")
    t4, t5, t6 = table("t4_shape_by_axis.csv"), table("t5_kshot.csv"), \
        table("t6_decomposition.csv")
    t7, rule = table("t7_controls.csv"), table("stopping_rule.csv")
    skipped = headline.get("skipped", {})

    def arms_of(kind: str) -> pd.DataFrame:
        return t2[t2["arm_kind"] == kind] if not t2.empty else pd.DataFrame()

    lines: list[str] = []
    add = lines.append

    add("# GEN11 decision report")
    add("")
    add(f"_Rendered from `runs/gen11_transfer/headline_tables/` at "
        f"{headline['generated_utc']}. Every number below is read back from one of those "
        f"CSVs, which are themselves recomputed from the raw OOF predictions in "
        f"`{headline['arms_dir']}`. Nothing here is transcribed._")
    add("")
    add("## 0. What was available when this report was rendered")
    add("")
    add(f"* arms with stored predictions: **{len(headline['arms_present'])}** of "
        f"**{headline['n_arms_preregistered']}** pre-registered")
    add(f"* design-matched baseline: `{headline['baseline'] or 'NOT RUN'}`")
    add(f"* frozen gen10 anchor: `{headline['frozen_anchor'] or 'NOT RUN'}`")
    add(f"* seeds per arm: {headline['seeds']}")
    add(f"* cohort fingerprint asserted by the runner: `{headline['cohort_fingerprint']}`")
    check = headline.get("frozen_anchor_reproduces_gen10")
    if check:
        add(f"* the frozen anchor recomputed here scores macro MAE "
            f"**{_fmt(check['gen11_recomputed'], 12)}** against gen10's own re-derivation from "
            f"gen10's raw predictions, **{_fmt(check['gen10_rederived'], 12)}** "
            f"(|Δ| = {check['abs_delta']:.3g}) — the anchor is the same model, not a "
            f"look-alike")
    if headline["arms_missing"]:
        add("")
        add("Pre-registered arms **not** on disk when this ran (the primary run was still in "
            "progress, or the stage has not been launched):")
        add("")
        for name in headline["arms_missing"]:
            add(f"* `{name}`")
    if skipped:
        add("")
        add("Tables skipped, with the reason (an absent table is recorded, never emitted "
            "empty):")
        add("")
        for name, reason in skipped.items():
            add(f"* `{name}` — {reason}")
    add("")

    add("## 1. The one framing that must not be dropped")
    add("")
    add("Every auxiliary arm runs at a **changed design corner** "
        "(`GENERAL` metal representation + `ANNOTATION_SAFE` MASSACTION), because the frozen "
        "design cannot represent auxiliary rows. So `auxiliary arm − frozen gen10` is not the "
        "transfer effect; it is the sum of two effects, and every table here splits them:")
    add("")
    add("```")
    add("design effect   = macro(control @ FROZEN_3|FROZEN_8) − macro(control @ GENERAL|ANNOTATION_SAFE)")
    add("transfer effect = macro(control @ same corner as the arm) − macro(arm)")
    add("total           = design effect + transfer effect          (positive = better)")
    add("```")
    add("")
    if not t2.empty and "design_effect_macro" in t2.columns:
        corner = t2[t2["arm_kind"] == "DESIGN_CONTROL"]
        add("Measured design effects (`t2_leaderboard.csv`), each control corner against the "
            "frozen anchor:")
        add("")
        add(_md_table(corner, ["model", "macro_mae", "offset_mae", "shape_mae", "pooled_mae",
                               "n_seeds", "design_effect_macro"]))
    else:
        add("_`t2_leaderboard.csv` is not available, so the design effect cannot be quoted._")
    add("")
    design_rows = (t3[(t3["statistic"] == "mae") & (t3["comparison_kind"] == "DESIGN")]
                   if not t3.empty and "comparison_kind" in t3.columns else pd.DataFrame())
    if not design_rows.empty:
        add("The same comparison as a paired, chemotype-blocked bootstrap "
            "(`t3_paired_bootstrap.csv`, `comparison_kind = DESIGN`). A design corner whose "
            "interval excludes zero is a real cost or benefit that an auxiliary arm inherits "
            "before it transfers anything:")
        add("")
        add(_md_table(design_rows, ["candidate", "point_delta", "bca_low", "bca_high",
                                    "cluster_robust_low", "cluster_robust_high",
                                    "units_improved", "units_total", "bca_direction"]))
        add("")

    add("## 2. Leaderboard (`t2_leaderboard.csv`; per-seed values in "
        "`t2_leaderboard_by_seed.csv`)")
    add("")
    add(_md_table(t2, ["model", "arm_kind", "design_corner", "macro_mae", "macro_mae_sd",
                       "offset_mae", "shape_mae", "pooled_mae", "n_seeds",
                       "transfer_effect_macro", "design_effect_macro",
                       "total_vs_frozen_macro", "n_seeds_improved_vs_baseline"]))
    add("")

    def section(number: int, title: str, name: str, frame: pd.DataFrame,
                columns: Sequence[str], note: str = "") -> None:
        add(f"## {number}. {title} (`{name}`)")
        add("")
        if name in skipped:
            add(f"_Not produced: {skipped[name]}._")
        elif frame.empty:
            add(f"_`{name}` exists but has no rows._")
        else:
            if note:
                add(note)
                add("")
            add(_md_table(frame, list(columns)))
        add("")

    section(3, "What each arm is made of", "t1_arm_composition.csv", t1,
            ["arm", "aux_rows", "aux_structures", "aux_metals", "aux_new_chemotypes",
             "aux_rows_on_new_chemistry", "aux_rows_fingerprint_identical_to_cohort",
             "aux_usable_curves", "aux_effective_chemotypes", "total_rows",
             "n_oof_variants_run"],
            "The archive's contribution is metals and curves, not ligands — read any transfer "
            "result against these counts.")

    section(4, "Where a difference sits", "t6_decomposition.csv",
            t6[t6["stratum_kind"].isin(["ALL", "band"])] if not t6.empty else t6,
            ["candidate", "band_scheme", "stratum_kind", "stratum", "n_rows", "n_units",
             "abs_improvement", "level_improvement", "shape_improvement"],
            "Positive = the candidate is better. `level` is the per-ligand mean residual and "
            "`shape` the deviation about it; they move independently.")

    far = table("t6_far_band_bootstrap.csv")
    if not far.empty:
        add("Criterion D needs the far band to be *statistically* better, not merely better on "
            "a point estimate, so the far stratum carries its own chemotype-blocked paired "
            "bootstrap (`t6_far_band_bootstrap.csv`):")
        add("")
        add(_md_table(far, ["candidate", "band_scheme", "n_rows", "n_units", "point_delta",
                            "bca_low", "bca_high", "units_improved", "units_total", "status"]))
        add("")

    section(5, "Curve shape by axis", "t4_shape_by_axis.csv",
            t4[t4["shape_improvement_vs_baseline"].notna()] if not t4.empty
            and "shape_improvement_vs_baseline" in t4.columns else t4,
            ["model", "axis_label", "n_curves", "shape_mae", "baseline_shape_mae",
             "shape_improvement_vs_baseline", "slope_mae", "span_recovery_median"])

    section(6, "Few-shot frontier", "t5_kshot.csv", t5,
            ["model", "arm", "k", "mae", "n_ligands", "regenerated_from_raw"])

    section(7, "Negative controls", "t7_controls.csv", t7,
            ["control_arm", "control", "baseline_macro_mae", "control_macro_mae",
             "twin_macro_mae", "control_improvement_vs_baseline",
             "twin_improvement_vs_baseline", "bca_low", "verdict"],
            "A negative control must stay flat. `ARTEFACT_WARNING` means the corruption did "
            "not remove the benefit, so the benefit was not the auxiliary information.")

    add("## 8. The thirteen questions (§21)")
    add("")
    add("_Question wording is reconstructed from the gen11 module contracts; the brief "
        "document is not in this worktree. Each answer names the table it came from._")
    add("")

    answers = _answer_questions(t1, t2, t3, t4, t5, t6, t7, rule, headline, arms_of)
    for question, answer in zip(SECTION_21_QUESTIONS, answers):
        add(f"### {question['n']}. {question['q']}")
        add("")
        add(f"_(maps to {question['maps_to']})_")
        add("")
        add(answer)
        add("")

    add("## 9. The stopping rule (§18), evaluated mechanically")
    add("")
    if rule.empty:
        add("_`stopping_rule.csv` was not produced._")
    else:
        counts = rule.groupby("status").size().to_dict()
        add("Status counts across all (arm, criterion) cells: "
            + ", ".join(f"**{k}** {v}" for k, v in sorted(counts.items())) + ".")
        add("")
        add(_md_table(rule, ["arm", "arm_kind", "criterion", "measured_quantity",
                             "measured_value", "threshold", "requirement_met", "status",
                             "reason"]))
    add("")

    add("## 10. Reproducing every number here (§22)")
    add("")
    add("```sh")
    add("PYTHONPATH=src python scripts/gen11_report_tables.py")
    add("```")
    add("")
    add("It reads only `runs/gen11_transfer/arms/oof_*.parquet` (raw predictions), "
        "`runs/gen11_transfer/composition/` (counts) and, where present, the gen11 k-shot "
        "detail; it recomputes macro/offset/shape with `gen6.metrics` and `gen9.metrics`, the "
        "bootstrap with `gen6.metrics.paired_unit_bootstrap`, and rewrites this file from the "
        "CSVs it produced. Deleting `runs/gen11_transfer/headline_tables/` and rerunning "
        "reproduces both the tables and this report.")
    add("")
    add("The only cached step is gen9's curve reconstruction, in "
        "`headline_tables/.shape_cache/`. Its key is a content hash of the arm's predictions, "
        "so a refitted arm cannot be served a stale entry, and deleting the directory rebuilds "
        "it. `--skip-shape` omits it entirely, at the cost of making criterion C "
        "`NOT_EVALUABLE`.")
    add("")
    if headline.get("caveats"):
        add("## 11. Caveats this run detected")
        add("")
        for caveat in headline["caveats"]:
            add(f"* {caveat}")
        add("")
    return "\n".join(lines)


def _answer_questions(t1, t2, t3, t4, t5, t6, t7, rule, headline, arms_of) -> list[str]:
    """One answer per §21 question, each grounded in a named table."""
    aux = arms_of("AUXILIARY")
    boot = t3[t3["statistic"] == "mae"] if not t3.empty and "statistic" in t3.columns \
        else pd.DataFrame()
    transfer_boot = boot[boot["comparison_kind"] == "TRANSFER"] if not boot.empty else boot
    answers: list[str] = []

    def no(reason: str) -> str:
        return f"**Not answerable**, because {reason}."

    # 1 — zero-shot
    if aux.empty:
        answers.append(no("no arm carrying auxiliary rows has produced predictions yet "
                          "(`t2_leaderboard.csv` contains only control arms)"))
    elif aux["transfer_effect_macro"].isna().all():
        answers.append(no("the control at the auxiliary arms' design corner "
                          "(`GENERAL|ANNOTATION_SAFE`) has not been run, so no auxiliary arm has "
                          "an honest baseline (`t2_leaderboard.csv`)"))
    else:
        best = aux.loc[aux["transfer_effect_macro"].idxmax()]
        detail = ""
        if not transfer_boot.empty and (transfer_boot["candidate"] == best["model"]).any():
            row = transfer_boot[transfer_boot["candidate"] == best["model"]].iloc[0]
            detail = (f" Chemotype-blocked paired bootstrap (`t3_paired_bootstrap.csv`): "
                      f"point {_fmt(row['point_delta'])}, BCa 95 % "
                      f"[{_fmt(row.get('bca_low'))}, {_fmt(row.get('bca_high'))}], "
                      f"percentile [{_fmt(row.get('ci95_low'))}, {_fmt(row.get('ci95_high'))}], "
                      f"{int(row['units_improved'])}/{int(row['units_total'])} units improved.")
        improved = best["n_seeds_improved_vs_baseline"]
        improved = int(improved) if np.isfinite(improved) else "?"
        answers.append(
            f"Best auxiliary arm is `{best['model']}` with a **transfer effect of "
            f"{_fmt(best['transfer_effect_macro'])} macro MAE** against its design-matched "
            f"control (`t2_leaderboard.csv`), improving in "
            f"{improved}/{int(best['n_seeds'])} seeds. The §18A bar is "
            f"0.02 with consistent seed direction.{detail}")

    # 2 — design vs transfer
    corners = t2[t2["arm_kind"] == "DESIGN_CONTROL"] if not t2.empty else pd.DataFrame()
    if corners.empty or corners["design_effect_macro"].isna().all():
        answers.append(no("the control grid has not finished: without both the frozen anchor and "
                          "the changed-design control, the design effect cannot be separated "
                          "from the transfer effect (`t2_leaderboard.csv`)"))
    else:
        design_boot = (boot[boot["comparison_kind"] == "DESIGN"].set_index("candidate")
                       if not boot.empty else pd.DataFrame())
        pieces = []
        for _, row in corners.iterrows():
            if not np.isfinite(row["design_effect_macro"]):
                continue
            tag = ""
            if not design_boot.empty and row["model"] in design_boot.index:
                b = design_boot.loc[row["model"]]
                tag = (f" [BCa {_fmt(b.get('bca_low'))}, {_fmt(b.get('bca_high'))}; "
                       f"{b.get('bca_direction')}]")
            pieces.append(f"`{row['design_corner']}` {_fmt(row['design_effect_macro'])}{tag}")
        answers.append(
            "Measured directly by running the control at every design corner "
            "(`t2_leaderboard.csv`, column `design_effect_macro`, positive = the changed design "
            "is better than frozen; intervals from `t3_paired_bootstrap.csv`): "
            + "; ".join(pieces) + ". Any `TOTAL_CONFOUNDED` row in "
            "`t3_paired_bootstrap.csv` is the sum of this and the transfer effect and must not "
            "be quoted as a transfer result.")

    # 3 — which composition
    if aux.empty:
        answers.append(no("no composition arm (B/C/D/E) has produced predictions yet "
                          "(`t2_leaderboard.csv`)"))
    else:
        ranked = aux.sort_values("transfer_effect_macro", ascending=False)
        pieces = [f"`{r['arm']}` {_fmt(r['transfer_effect_macro'])}" for _, r in ranked.iterrows()]
        answers.append("Transfer effect by composition (`t2_leaderboard.csv`): "
                       + "; ".join(pieces) + ". Read these against `t1_arm_composition.csv`: the "
                       "auxiliary pool adds metals and curves, not ligands.")

    # 4 — relevance vs row count
    matched = aux[aux["arm"].isin(["F_ACTINIDES_ONLY_MATCHED", "G_RANDOM_AUX_MATCHED"])] \
        if not aux.empty else pd.DataFrame()
    if matched.empty:
        answers.append(no("the size-matched arms F and G have not been run "
                          "(`t2_leaderboard.csv`); the F/G contrast is the only thing that "
                          "separates chemical relevance from row count"))
    else:
        pieces = [f"`{r['model']}` {_fmt(r['transfer_effect_macro'])}"
                  for _, r in matched.iterrows()]
        answers.append("Size-matched draws (`t2_leaderboard.csv`): " + "; ".join(pieces)
                       + ". Note `t1_arm_composition.csv`: row-level matching leaves F and G "
                         "with far fewer usable curves than D, so they are matched on rows, not "
                         "on curves.")

    # 5 — mechanism
    mechanisms = aux["mechanism"].unique().tolist() if not aux.empty else []
    if len(mechanisms) <= 1:
        answers.append(no(f"only the {mechanisms[0] if mechanisms else 'JOINT'} mechanism has "
                          "been run; §6 requires the mechanisms to be compared, not collapsed "
                          "(`t2_leaderboard.csv`)"))
    else:
        pieces = [f"`{r['model']}` {_fmt(r['transfer_effect_macro'])}"
                  for _, r in aux.sort_values("transfer_effect_macro", ascending=False).iterrows()]
        answers.append("Transfer effect by mechanism (`t2_leaderboard.csv`): " + "; ".join(pieces))

    # 6 — weighting
    weights = aux[["weighting", "aux_lambda"]].drop_duplicates() if not aux.empty \
        else pd.DataFrame()
    if len(weights) <= 1:
        answers.append(no("only one weighting/auxiliary-mass setting has been run, so §12's "
                          "robustness question has no contrast to measure "
                          "(`t2_leaderboard.csv`)"))
    else:
        pieces = [f"`{r['weighting']}`/lam{r['aux_lambda']:g} {_fmt(r['transfer_effect_macro'])}"
                  for _, r in aux.iterrows()]
        answers.append("Transfer effect across weighting schemes (`t2_leaderboard.csv`): "
                       + "; ".join(pieces))

    # 7 — level vs shape
    if t6.empty:
        answers.append(no("`t6_decomposition.csv` was not produced — it needs an auxiliary arm "
                          "and its design-matched control to exist together"))
    else:
        allrows = t6[t6["stratum_kind"] == "ALL"]
        pieces = [f"`{r['candidate']}`: level {_fmt(r['level_improvement'])}, shape "
                  f"{_fmt(r['shape_improvement'])}, total {_fmt(r['abs_improvement'])}"
                  for _, r in allrows.iterrows()]
        answers.append("Level/shape split of the transfer effect (`t6_decomposition.csv`, "
                       "`stratum = ALL`): " + "; ".join(pieces))

    # 8 — far chemotypes
    if t6.empty:
        answers.append(no("`t6_decomposition.csv` was not produced"))
    else:
        bands = t6[(t6["stratum_kind"] == "band")]
        if bands.empty:
            answers.append(no("no band stratification survived in `t6_decomposition.csv`"))
        else:
            pieces = []
            for (cand, scheme), block in bands.groupby(["candidate", "band_scheme"]):
                inner = ", ".join(f"{r['stratum']} {_fmt(r['abs_improvement'])} "
                                  f"({int(r['n_units'])} units)" for _, r in block.iterrows())
                pieces.append(f"`{cand}` [{scheme}]: {inner}")
            answers.append("Improvement by chemotype-distance band "
                           "(`t6_decomposition.csv`): " + "; ".join(pieces)
                           + ". Both band definitions present in the repository are reported "
                             "because they disagree (see the caveats).")

    # 9 — shape by axis
    if t4.empty:
        answers.append(no("`t4_shape_by_axis.csv` was not produced"))
    else:
        candidates = t4[t4["shape_improvement_vs_baseline"].notna()]
        if candidates.empty:
            answers.append(no("no arm in `t4_shape_by_axis.csv` has a design-matched control to "
                              "be compared against"))
        else:
            best = candidates.loc[candidates["shape_improvement_vs_baseline"].idxmax()]
            answers.append(
                f"Best per-axis shape gain (`t4_shape_by_axis.csv`): `{best['model']}` on axis "
                f"`{best['axis_label']}`, shape MAE {_fmt(best['baseline_shape_mae'])} → "
                f"{_fmt(best['shape_mae'])} ({_fmt(best['shape_improvement_vs_baseline'])} over "
                f"{int(best['n_curves'])} curves). The §18C bar is 0.10 with macro non-worse.")

    # 10 — k-shot
    if t5.empty:
        answers.append(no("no k-shot frontier exists for gen11 arms "
                          "(`t5_kshot.csv` skipped); the frontier is produced by pointing "
                          "`scripts/gen10_final_locked.py` at `runs/gen11_transfer/arms`"))
    else:
        pieces = []
        for (model, k), block in t5.groupby(["model", "k"]):
            pieces.append(f"`{model}` k={k}: {_fmt(block['mae'].min())}")
        answers.append("Frontier (`t5_kshot.csv`, best deployable arm per k): "
                       + "; ".join(pieces))

    # 11 — negative controls
    if t7.empty:
        answers.append(no("the §19 negative-control arms (permuted auxiliary target, shuffled "
                          "metal labels) have not been run (`t7_controls.csv` skipped); until "
                          "they are, no positive result can be attributed to auxiliary "
                          "information rather than to the extra rows themselves"))
    else:
        pieces = [f"`{r['control']}` {_fmt(r['control_improvement_vs_baseline'])} → "
                  f"**{r['verdict']}**" for _, r in t7.iterrows()]
        answers.append("Negative controls (`t7_controls.csv`): " + "; ".join(pieces))

    # 12 — publication blocking
    policy_arms = t2[t2["policy"] != "HEADLINE"] if not t2.empty and "policy" in t2.columns \
        else pd.DataFrame()
    if policy_arms.empty:
        answers.append(no("no `PUBLICATION_BLOCKED` arm has been run, so §13's "
                          "publication-level robustness cannot be assessed from arm predictions "
                          "(`t2_leaderboard.csv`) — the publication-blocked *interval* variant is "
                          "available separately by rerunning this script with `--block doi`"))
    else:
        pieces = [f"`{r['model']}` {_fmt(r['transfer_effect_macro'])}"
                  for _, r in policy_arms.iterrows()]
        answers.append("Publication-blocked admissibility (`t2_leaderboard.csv`): "
                       + "; ".join(pieces))

    # 13 — the verdict
    if rule.empty:
        answers.append(no("`stopping_rule.csv` was not produced"))
    else:
        counts = rule.groupby("status").size().to_dict()
        auxiliary = rule[rule["arm_kind"] == "AUXILIARY"]
        scored = auxiliary[auxiliary["status"].isin(["PASS", "FAIL"])]
        real = auxiliary[auxiliary["status"] == "PASS"]
        if auxiliary.empty:
            answers.append(no("no arm carrying auxiliary rows appears in `stopping_rule.csv`, so "
                              f"§18 has nothing to score (status counts {counts}); frozen gen10 "
                              "remains the control by default, which is not the same as gen11 "
                              "having tested it and failed"))
        elif scored.empty:
            answers.append(no("every §18 cell for every auxiliary arm is `NOT_EVALUABLE` — the "
                              f"evidence each criterion needs has not been produced ({counts}); "
                              "frozen gen10 remains the control, but no criterion has actually "
                              "been measured against it"))
        elif real.empty:
            answers.append(
                "**No.** No auxiliary arm passes any §18 criterion it could be scored on "
                f"(`stopping_rule.csv`, status counts {counts}). Frozen gen10 remains the "
                "control. Read the `NOT_EVALUABLE` cells separately from the `FAIL` cells: the "
                "first is missing evidence, the second is a measured negative.")
        else:
            listing = ", ".join(f"`{r['arm']}`/{r['criterion']}" for _, r in real.iterrows())
            answers.append(f"Criteria passed by auxiliary arms (`stopping_rule.csv`): "
                           f"{listing}. Status counts: {counts}. §18 adoption also requires the "
                           "negative controls to be flat (`t7_controls.csv`) — check question 11 "
                           "before treating this as an adoption decision.")
    return answers


# --------------------------------------------------------------------------- #

def preregistered_arms() -> list[str]:
    """Every arm name the run script declares, so "missing" is a real count."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_gen11_run", REPO_ROOT / "scripts" / "gen11_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return sorted({s.name for stage in module.STAGES.values() for s in stage})


def aux_extractants() -> list[str]:
    path = GEN11 / "featurizer" / "aux_features.parquet"
    if not path.exists():
        return []
    column = "extractant_primary_smiles"
    return sorted(set(pd.read_parquet(path, columns=[column])[column].dropna()))


def build_t1(design: pd.DataFrame) -> pd.DataFrame | None:
    """Composition per arm, joined to whether that arm actually produced predictions."""
    path = GEN11 / "composition" / "arm_composition.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    curves = GEN11 / "composition" / "curve_inventory_by_arm.csv"
    if curves.exists():
        curve = pd.read_csv(curves)
        aux_only = curve[curve["scope"] == "auxiliary_only"][
            ["arm", "usable_curves", "curves_on_new_chemistry", "curves_far"]]
        table = table.merge(aux_only.rename(columns={
            "usable_curves": "aux_usable_curves",
            "curves_on_new_chemistry": "aux_curves_on_new_chemistry",
            "curves_far": "aux_curves_far"}), on="arm", how="left")
    conc = GEN11 / "composition" / "concentration_diagnostics.csv"
    if conc.exists():
        diag = pd.read_csv(conc)
        aux_scope = diag[diag["scope"] == "aux"][
            ["arm", "largest_publication_share", "effective_publications",
             "effective_structures", "effective_chemotypes"]]
        table = table.merge(aux_scope.rename(columns={
            "largest_publication_share": "aux_largest_publication_share",
            "effective_publications": "aux_effective_publications",
            "effective_structures": "aux_effective_structures",
            "effective_chemotypes": "aux_effective_chemotypes"}), on="arm", how="left")
    run = design.groupby("arm")["model"].agg(list).to_dict() if not design.empty else {}
    table["n_oof_variants_run"] = table["arm"].map(lambda a: len(run.get(a, [])))
    table["oof_variants_run"] = table["arm"].map(lambda a: ";".join(sorted(run.get(a, []))))
    table["predictions_available"] = table["n_oof_variants_run"] > 0
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path, default=GEN11 / "arms")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=GEN11 / "GEN11_DECISION_REPORT.md",
                        help="where the rendered decision report goes; redirect it when "
                             "running against anything other than the real arms")
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--block", default="tanimoto_cluster",
                        help="bootstrap resampling unit; 'doi' gives §13's publication blocking")
    parser.add_argument("--flat-threshold", type=float, default=0.02,
                        help="a negative control clearing this much is an artefact warning")
    parser.add_argument("--skip-shape", action="store_true",
                        help="do not rebuild the curve-shape table (T4 and criterion C become "
                             "NOT_EVALUABLE); gen9's curve reconstruction is the slowest step "
                             "here at roughly 4,700 curves per arm per seed")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    frames = load_arms(args.arms)
    if not frames:
        raise SystemExit(f"no OOF parquets in {args.arms} — nothing to report on")
    design = design_table(frames)
    skipped: dict[str, str] = {}
    caveats: list[str] = []

    try:
        preregistered = preregistered_arms()
    except Exception as error:  # noqa: BLE001 — a missing run script must not stop the report
        preregistered = sorted(frames)
        caveats.append(f"could not read the pre-registered arm list from scripts/gen11_run.py "
                       f"({error}); 'missing arms' is therefore only what is on disk")
    missing_arms = [name for name in preregistered if name not in frames]

    print(f"arms on disk: {len(frames)} / {len(preregistered)} pre-registered")
    for name in sorted(frames):
        print(f"  present  {name}")
    for name in missing_arms:
        print(f"  MISSING  {name}")

    # --- T1 ------------------------------------------------------------------ #
    t1 = build_t1(design)
    if t1 is None:
        skipped["t1_arm_composition.csv"] = "runs/gen11_transfer/composition/ has no " \
                                            "arm_composition.csv"
        t1 = pd.DataFrame()
    else:
        t1.to_csv(args.out / "t1_arm_composition.csv", index=False)

    # --- T2 ------------------------------------------------------------------ #
    per_seed, t2 = build_t2(frames, design)
    # T2 is two files on purpose: the seed-averaged leaderboard is what gets read,
    # and the per-seed table is what makes "consistent seed direction" checkable
    # rather than assertable.
    per_seed.to_csv(args.out / "t2_leaderboard_by_seed.csv", index=False)
    t2.to_csv(args.out / "t2_leaderboard.csv", index=False)
    print("\n=== t2 leaderboard ===")
    print(t2[["model", "arm_kind", "macro_mae", "offset_mae", "shape_mae", "pooled_mae",
              "n_seeds", "transfer_effect_macro", "design_effect_macro"]]
          .round(4).to_string(index=False))

    baseline = None
    aux_models = design.loc[design["carries_auxiliary"], "model"].tolist()
    if aux_models:
        baseline = baseline_for(design, aux_models[0])
    if baseline is None:
        corner = "|".join(("GENERAL", "ANNOTATION_SAFE"))
        match = design[(design["arm"] == CONTROL_ARM_KEY) & (design["design_corner"] == corner)]
        baseline = None if match.empty else str(match["model"].iloc[0])
    anchor = frozen_anchor(design)

    # --- T3 ------------------------------------------------------------------ #
    t3 = build_t3(frames, design, replicates=args.replicates, block=args.block)
    if t3.empty:
        skipped["t3_paired_bootstrap.csv"] = "no two arms share a design corner or the frozen " \
                                             "anchor, so there is no paired comparison to draw"
    else:
        t3.to_csv(args.out / "t3_paired_bootstrap.csv", index=False)
        print("\n=== t3 paired bootstrap (mae) ===")
        show = t3[t3["statistic"] == "mae"]
        print(show[["comparison_kind", "reference", "candidate", "point_delta", "bca_low",
                    "bca_high", "units_improved", "units_total"]].round(4).to_string(index=False))

    # --- T4 ------------------------------------------------------------------ #
    if args.skip_shape:
        t4, curves = pd.DataFrame(), pd.DataFrame()
        skipped["t4_shape_by_axis.csv"] = "--skip-shape was passed; the curve-shape table was " \
                                          "not rebuilt, so no shape claim is made"
    else:
        try:
            t4, curves = build_t4(frames, design, cache_dir=args.out / ".shape_cache")
        except FileNotFoundError as error:
            t4, curves = pd.DataFrame(), pd.DataFrame()
            skipped["t4_shape_by_axis.csv"] = f"gen9 curve membership is unavailable ({error})"
    if t4.empty:
        skipped.setdefault("t4_shape_by_axis.csv", "no curve reached gen9's minimum point count")
    else:
        t4.to_csv(args.out / "t4_shape_by_axis.csv", index=False)
        curves.to_parquet(args.out / "t4_curve_shape.parquet", index=False)

    # --- T5 ------------------------------------------------------------------ #
    path, kind = find_kshot(GEN11)
    if path is None:
        t5 = pd.DataFrame()
        skipped["t5_kshot.csv"] = ("no k-shot artefact under runs/gen11_transfer/; produce it "
                                   "with scripts/gen10_final_locked.py pointed at the gen11 arms")
    else:
        t5 = build_t5(path, kind, design)
        t5.to_csv(args.out / "t5_kshot.csv", index=False)

    # --- T6 ------------------------------------------------------------------ #
    t6 = build_t6(frames, design, aux_extractants())
    if t6.empty:
        if not aux_models:
            reason = "no arm carrying auxiliary rows has produced predictions"
        elif baseline is None:
            reason = ("the control at the auxiliary arms' design corner "
                      "(GENERAL|ANNOTATION_SAFE) has not been run, so no arm has a "
                      "design-matched control to be decomposed against")
        else:
            reason = ("no arm shares an identical (row_id, split_seed) key with its "
                      "design-matched control, so no paired decomposition is defined")
        skipped["t6_decomposition.csv"] = reason
    else:
        t6.to_csv(args.out / "t6_decomposition.csv", index=False)

    # --- far-band bootstrap, for §18 D --------------------------------------- #
    cuts = gen10_tercile_cuts()
    far_rows = []
    for model in design.loc[design["carries_auxiliary"] | (design["control"] != "NONE"), "model"]:
        base = baseline_for(design, model)
        if base is None or base == model:
            continue
        if key_signature(frames[model]) != key_signature(frames[base]):
            continue
        for scheme in ["GEN11_ANALYSIS"] + (["GEN10_TERCILE"] if cuts else []):
            record = band_bootstrap(frames, base, model, band="far",
                                    replicates=args.replicates, band_scheme=scheme, cuts=cuts)
            record.update({"candidate": model, "reference": base})
            far_rows.append(record)
    far_boot = pd.DataFrame(far_rows)
    if not far_boot.empty:
        far_boot.to_csv(args.out / "t6_far_band_bootstrap.csv", index=False)

    # --- T7 ------------------------------------------------------------------ #
    t7 = build_t7(design, t2, t3, threshold=args.flat_threshold)
    if t7.empty:
        skipped["t7_controls.csv"] = ("no negative-control arm has been run "
                                      "(scripts/gen11_run.py --stage controls)")
    else:
        t7.to_csv(args.out / "t7_controls.csv", index=False)

    # --- stopping rule -------------------------------------------------------- #
    consistency = GEN11 / "consistency"
    consistency_available = consistency.exists() and any(consistency.iterdir())
    rule = build_stopping_rule(design, t2, t3, t4, t5, t6, far_boot,
                               consistency_available=consistency_available)
    if rule.empty:
        skipped["stopping_rule.csv"] = "no arm other than the frozen anchor has been run"
    else:
        rule.to_csv(args.out / "stopping_rule.csv", index=False)
        print("\n=== stopping rule ===")
        print(rule[["arm", "criterion", "measured_value", "threshold", "status"]]
              .to_string(index=False))

    # --- caveats the tables themselves expose --------------------------------- #
    if cuts is not None:
        gen11_far = [b for b in analysis.CHEMOTYPE_BANDS if b[0] == "far"][0]
        if abs(gen11_far[2] - cuts["mid"]) > 1e-9:
            caveats.append(
                f"`gen11.analysis.CHEMOTYPE_BANDS` puts the far/mid cut at Tanimoto "
                f"{gen11_far[2]:.3f} and calls it gen10's, but `composition.py` recomputes "
                f"gen10's own tercile cut at {cuts['mid']:.6f} (near at {cuts['near']:.6f}). "
                f"They disagree, so `t6_decomposition.csv` reports both and criterion D is "
                f"evaluated on the gen11 module's definition.")
    if baseline is None:
        caveats.append("the design-matched baseline (control at `GENERAL|ANNOTATION_SAFE`) is "
                       "not on disk, so no transfer effect can be quoted; every auxiliary "
                       "comparison here is NOT_EVALUABLE rather than negative.")
    if anchor is None:
        caveats.append("the frozen gen10 anchor is not on disk, so the design effect cannot be "
                       "separated from the transfer effect.")
    if not t3.empty and "comparison_kind" in t3.columns:
        hurt = t3[(t3["statistic"] == "mae") & (t3["comparison_kind"] == "DESIGN")
                  & (t3.get("bca_high", pd.Series(np.nan, index=t3.index)) < 0)]
        for _, row in hurt.iterrows():
            corner = design.loc[design["model"] == row["candidate"], "design_corner"].iloc[0]
            used = bool((design.loc[design["carries_auxiliary"], "design_corner"] == corner).any())
            consequence = (
                "this is the corner the auxiliary arms use, so each of them starts from this "
                "deficit and its transfer effect must exceed it before the arm merely matches "
                "frozen gen10" if used else
                "no auxiliary arm on disk uses this corner, so it is reported as an isolated "
                "measurement of that one design change rather than as a deficit any arm carries")
            caveats.append(
                f"the design change alone — `{row['candidate']}` against the frozen anchor, no "
                f"auxiliary rows — is significantly *worse*: {row['point_delta']:.4f} macro MAE, "
                f"BCa [{row['bca_low']:.4f}, {row['bca_high']:.4f}]. {consequence}.")
    if not t3.empty and "largest_block_unit_share" in t3.columns:
        share = float(t3["largest_block_unit_share"].max())
        if share > 0.15:
            caveats.append(f"one chemotype block holds {share:.1%} of the scoring units in the "
                           f"bootstrap; gen6 measured the percentile interval at ~12.7 % "
                           f"one-sided Type-I error under exactly this leverage, so the BCa and "
                           f"cluster-robust columns of `t3_paired_bootstrap.csv` are the ones to "
                           f"read.")

    # --- headline.json -------------------------------------------------------- #
    gen10_control = json.loads(GEN10_CONTROL_JSON.read_text()) \
        if GEN10_CONTROL_JSON.exists() else {}
    # The frozen anchor must *be* gen10.  gen10's own re-derivation of its macro
    # MAE from its raw predictions is the only external number this script uses,
    # and it is used as a check rather than as a result.
    gen10_macro = next((m.get("recomputed") for m
                        in gen10_control.get("zero_shot_frozen_cohort", {}).get("metrics", [])
                        if "macro MAE" in str(m.get("metric", ""))), None)
    anchor_check = None
    if anchor is not None and gen10_macro is not None:
        recomputed = float(t2.loc[t2["model"] == anchor, "macro_mae"].iloc[0])
        anchor_check = {"gen11_recomputed": recomputed,
                        "gen10_rederived": float(gen10_macro),
                        "abs_delta": abs(recomputed - float(gen10_macro)),
                        "source": "runs/gen11_transfer/control/gen10_control.json"}
        if anchor_check["abs_delta"] > 1e-9:
            caveats.append(
                f"the frozen anchor recomputed here ({recomputed:.10f}) differs from gen10's own "
                f"re-derivation ({float(gen10_macro):.10f}) by {anchor_check['abs_delta']:.3g}; "
                f"gen11's control is not bit-identical to gen10 and every design effect below "
                f"inherits that difference.")
    headline = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "arms_dir": str(args.arms.resolve().relative_to(REPO_ROOT))
        if REPO_ROOT in args.arms.resolve().parents else str(args.arms),
        "arms_present": sorted(frames),
        "arms_missing": missing_arms,
        "n_arms_preregistered": len(preregistered),
        "seeds": sorted({int(s) for f in frames.values() for s in f["split_seed"].unique()}),
        "cohort_fingerprint": gen10_control.get("identity", {}).get("cohort", {})
                                           .get("fingerprint_recomputed"),
        "baseline": baseline,
        "frozen_anchor": anchor,
        "bootstrap": {"block_unit": args.block, "replicates": args.replicates},
        "design_effect_macro": {
            str(r["model"]): (None if not np.isfinite(r["design_effect_macro"])
                              else float(r["design_effect_macro"]))
            for _, r in t2[t2["arm_kind"] == "DESIGN_CONTROL"].iterrows()},
        "transfer_effect_macro": {
            str(r["model"]): (None if not np.isfinite(r["transfer_effect_macro"])
                              else float(r["transfer_effect_macro"]))
            for _, r in t2[t2["carries_auxiliary"]].iterrows()},
        "leaderboard": {str(r["model"]): {"macro_mae": float(r["macro_mae"]),
                                          "offset_mae": float(r["offset_mae"]),
                                          "shape_mae": float(r["shape_mae"]),
                                          "pooled_mae": float(r["pooled_mae"]),
                                          "n_seeds": int(r["n_seeds"])}
                        for _, r in t2.iterrows()},
        "gen10_control_macro_mae": gen10_macro,
        "frozen_anchor_reproduces_gen10": anchor_check,
        "stopping_rule": ({} if rule.empty else
                          {f"{r['arm']}|{r['criterion']}": r["status"]
                           for _, r in rule.iterrows()}),
        "stopping_rule_status_counts": ({} if rule.empty
                                        else rule.groupby("status").size().to_dict()),
        "tables_written": sorted(p.name for p in args.out.glob("*.csv")),
        "skipped": skipped,
        "caveats": caveats,
        "section_21_questions_source": "reconstructed in scripts/gen11_report_tables.py "
                                       "(SECTION_21_QUESTIONS); the brief is not in this worktree",
    }
    (args.out / "headline.json").write_text(json.dumps(headline, indent=2, default=float))

    report = render_report(args.out, headline)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(f"\nwrote {args.out}/ and {args.report}")
    if skipped:
        print("skipped tables:")
        for name, reason in skipped.items():
            print(f"  {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

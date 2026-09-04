#!/usr/bin/env python
"""§16 — does auxiliary training make the design-dependence worse?

gen10's headline *known limit* is that its shape model is design-relative: the
prediction it gives for a point depends on which *other* points the user happened
to list alongside it.  ``runs/gen10_final/query_consistency`` measured that at
~0.12 log units of median movement on unchanged points under a two-decade decoy.

gen11 proposes to train on 5,096 auxiliary cells that are ~90 % actinide.  A
macro-MAE gain bought by making the answer *more* a function of the question is
not a gain a chemist can use, so the same benchmark has to be re-run on the gen11
arms before any headline is written.  This script is that re-run.

Nothing about the benchmark is reimplemented.  The perturbations, the comparison
rule and the level/shape split are :mod:`lanthanide_separation.gen10.consistency`
and :mod:`~lanthanide_separation.gen10.perturb`, imported; the only new code here
is the part that fits a *gen11* arm (``gen11.runner.RunSpec`` ->
``gen11.transfer.TransferRecomposedModel``) instead of a gen10 one, and the part
that pairs arms unit-for-unit so "worse than the control" is a paired statement
rather than two independent distributions.

**Three things make this an honest comparison and each is checked, not assumed.**

*The design corner.*  Every auxiliary arm must run at metal scheme ``GENERAL``
and MASSACTION ``ANNOTATION_SAFE`` (the frozen 14-entry lanthanide metal block
leaves 98 % of auxiliary rows with no metal coordinate, and the frozen
MASSACTION block's annotation-coupled terms become an "is auxiliary" flag).  The
baseline is therefore the *control at that same corner*, not gen10's anchor.
Both are carried, so a design-corner effect cannot be read as a transfer effect.

*The zero control.*  ``FROZEN_MONOLITH`` is gen10's frozen arm — one forest on
the static design with representation ``NONE``.  It reads no design context at
all, so every shift it reports must be zero to the thread-order floor.  A
non-zero value there means the harness is measuring its own plumbing, which is
the failure gen9's acquisition study shipped with.  Note that a *recomposed*
model with representation ``NONE`` would not do: recomposition takes each curve's
level from the mean of the base prediction over the curve's rows, so it reads the
design through curve membership even when the feature block is empty.  The
monolith is the only architecture in the stack that is design-blind by
construction.

*Pairing.*  Arms are fitted on the same folds and asked the same questions in the
same order, so every (seed, fold, ligand, curve, variant) unit exists for all of
them.  ``delta_vs_baseline.csv`` is the paired difference over those units with a
sign test, because the unpaired difference of two medians over ~10^4 correlated
comparisons is not a measurement of anything.

Affordability: gen10 ran this at 5 seeds over every test ligand and took 4.6
hours for 3 arms.  gen11 runs it at **2 seeds** — say "2 seeds" wherever these
numbers are quoted.  Measured on this machine the cost is almost entirely
*fitting* (5 arms x 10 folds, and the auxiliary arms carry ~4,200 extra rows);
the query sweep is a fraction of that, so the ligand sample *defaults* to gen10
parity — every test ligand, gen10's own 12-curve cap.  ``--max-ligands`` thins it
for a contended machine; whichever applies, the ligands actually drawn are
recorded in ``sampling.json`` and the cap in ``control.json``.  The nine
perturbation families are never reduced.

Keeping gen10's *sampling rule* buys one more thing, cap or no cap.  The
perturbations are drawn
from ``default_rng([seed, fold])`` created fresh per ligand, so a given (seed,
fold, ligand, curve) yields byte-identical variants in both runs, and the gen11
anchor — the transfer class with an empty auxiliary block, which is gen10's
``RecomposedModel`` — can be checked against gen10's *published* numbers unit for
unit.  :func:`gen10_reproduction` does that and writes the result into
``control.json``.

Outputs, all under ``runs/gen11_transfer/consistency``:

``rows.parquet``            one record per (arm, seed, fold, ligand, curve, variant)
``summary_by_arm.csv``      per arm and perturbation family
``by_perturbation.csv``     per arm, family and individual variant label
``delta_vs_baseline.csv``   paired arm-minus-baseline, with a sign test
``shifted_window.csv``      the SHIFT diagnostic, which is not a consistency failure
``accuracy_context.csv``    macro MAE of the same arms, so §16 is read as a trade
``control.json``            the zero control, the gen10 reproduction, the provenance
``sampling.json``           the ligands, curves and caps actually used
``README.md``               the answer in words
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen7.harness import (  # noqa: E402
    DEFAULT_SEEDS, FoldContext, build_folds,
)
from lanthanide_separation.gen10.architectures import (  # noqa: E402
    MonolithModel, prime_raw_cache,
)
from lanthanide_separation.gen10.consistency import (  # noqa: E402
    MAX_CURVES_PER_LIGAND, VARIANTS, build_variants, compare_variant, eligible_curves,
    shift_for_shifted_window,
)
from lanthanide_separation.gen10.runner import DETERMINISM_TOLERANCE, prepared_cohort  # noqa: E402
from lanthanide_separation.gen11.overlap import build_overlap_map  # noqa: E402
from lanthanide_separation.gen11.pools import build_pool  # noqa: E402
from lanthanide_separation.gen11.runner import (  # noqa: E402
    RunSpec, attach_metal_scheme, blocks_for_scheme, build_model,
)
from lanthanide_separation.gen9.train import FROZEN_BLOCKS  # noqa: E402
from lanthanide_separation.levels import LEVEL_TARGET_COLUMN  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen11_transfer" / "consistency"
AUX_FEATURES = REPO_ROOT / "runs" / "gen11_transfer" / "featurizer" / "aux_features.parquet"

#: The design corner every auxiliary arm is forced to; see the module docstring.
AUX_CORNER = dict(metal_scheme="GENERAL", massaction_scheme="ANNOTATION_SAFE")

#: ``FROZEN_MONOLITH`` is the correctness check, not a competitor: it must be 0.
BASELINE = "A_GEN10_CONTROL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1"
ZERO_CONTROL = "FROZEN_MONOLITH"


@dataclass(frozen=True)
class ArmDef:
    """One fitted-once, queried-many arm.  ``spec`` is None for the zero control."""

    name: str
    spec: RunSpec | None
    blocks: tuple[str, ...]
    role: str


def arm_table(names: list[str] | None) -> list[ArmDef]:
    """The arms this benchmark can run, keyed by the name they report under."""
    specs = {
        # bit-for-bit gen10, so gen11's numbers can be quoted against gen10's on
        # *this* ligand sample rather than against a differently-sampled run.
        "A_GEN10_CONTROL|JOINT|FROZEN_3|FROZEN_8|HIERARCHICAL|lam1": RunSpec(
            "A_GEN10_CONTROL", metal_scheme="FROZEN_3", massaction_scheme="FROZEN_8"),
        BASELINE: RunSpec("A_GEN10_CONTROL", **AUX_CORNER),
        "C_LN_PLUS_ACTINIDES|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1": RunSpec(
            "C_LN_PLUS_ACTINIDES", **AUX_CORNER),
        "E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1": RunSpec(
            "E_LN_PLUS_ALL", **AUX_CORNER),
    }
    for key, spec in specs.items():
        if spec.name != key:
            raise SystemExit(f"arm key {key!r} does not match RunSpec.name {spec.name!r}")
    out = [ArmDef(ZERO_CONTROL, None, FROZEN_BLOCKS, "zero_control")]
    for key, spec in specs.items():
        role = ("baseline" if key == BASELINE else
                "gen10_anchor" if spec.metal_scheme == "FROZEN_3" else "auxiliary")
        out.append(ArmDef(key, spec,
                          blocks_for_scheme(spec.metal_scheme, spec.massaction_scheme), role))
    if names is None:
        return out
    known = {a.name: a for a in out}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; have {sorted(known)}")
    return [known[n] for n in names]


def fit_arm(arm: ArmDef, train: pd.DataFrame, y: np.ndarray, context: FoldContext,
            pool, n_jobs: int):
    """Fit one arm on one fold and hand back an object with ``predict``."""
    from lanthanide_separation.gen11.metalrep import REPRESENTATIONS

    if arm.spec is None:
        model = MonolithModel(name=arm.name, representation_name="NONE",
                              blocks=tuple(arm.blocks))
        return model.fit(train, y, context).set_predict_jobs(n_jobs)
    metal_columns = REPRESENTATIONS[arm.spec.metal_scheme][1]
    model = build_model(arm.spec, pool, metal_columns=metal_columns)
    return model.fit(train, y, context).set_predict_jobs(n_jobs)


def _predictions(model, frame: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(frame),
                     index=frame["row_id"].astype(str).to_numpy())


def run(arms: list[ArmDef], seeds: list[int], max_ligands: int | None,
        max_curves: int, n_jobs: int, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    cohort = prepared_cohort()
    overlap = build_overlap_map(cohort.frame)
    if not AUX_FEATURES.exists():
        raise SystemExit(f"auxiliary features not found: {AUX_FEATURES}")
    aux_features = pd.read_parquet(AUX_FEATURES)
    build = build_pool(cohort, overlap, aux_features, policy="HEADLINE", seeds=seeds)

    # One patched cohort serves every arm.  ``attach_metal_scheme`` only *adds*
    # blocks, so the frozen tuple is still intact on it and the gen10 anchor and
    # the zero control read exactly the columns gen10 gave them; the fingerprint
    # assertion inside it is what makes that claim checkable rather than hopeful.
    patched, aux = attach_metal_scheme(cohort, build.pool.features,
                                       AUX_CORNER["metal_scheme"],
                                       AUX_CORNER["massaction_scheme"])
    pool = replace(build.pool, features=aux)
    for block_tuple in {tuple(a.blocks) for a in arms}:
        prime_raw_cache(patched, block_tuple)

    frame = patched.frame
    target = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    records: list[dict] = []
    sampling: list[dict] = []
    fit_seconds: list[dict] = []
    folds_done: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    def provenance() -> dict:
        return {
            "folds_completed": folds_done, "sampling": sampling,
            "fit_seconds": fit_seconds,
            "pool_rows": int(len(build.pool.features)),
            "pool_admissible_mean_over_folds": float(build.survival["n_admissible"].mean()),
            "cohort_fingerprint": patched.fingerprint,
        }

    def checkpoint() -> pd.DataFrame:
        """Flush the rows and the provenance after every fold.

        Five arms over ten folds is hours of forest fitting on a contended
        machine, and a run that writes only at the end is a run whose result is
        all-or-nothing.  Every summary downstream reads whole (seed, fold)
        blocks, so a checkpoint is a complete smaller experiment rather than a
        truncated one — ``--summarise-only`` will produce every table from it and
        ``control.json`` records which folds it contains.
        """
        table = pd.DataFrame.from_records(records)
        if not table.empty:
            table.to_parquet(out_dir / "rows.parquet", index=False)
            (out_dir / "sampling.json").write_text(json.dumps(provenance(), indent=2))
        return table

    for seed in seeds:
        for fold in build_folds(frame, seed):
            train = frame.iloc[fold.train_index]
            test = frame.iloc[fold.test_index].drop(columns=[LEVEL_TARGET_COLUMN])
            context = FoldContext(fold=fold, cohort=patched, feature_columns=(),
                                  model_seed=fold.model_seed)
            y = target[fold.train_index]
            fitted = {}
            for arm in arms:
                started = time.time()
                fitted[arm.name] = fit_arm(arm, train, y, context, pool, n_jobs)
                elapsed = time.time() - started
                fit_seconds.append({"split_seed": int(seed), "fold": int(fold.fold),
                                    "arm": arm.name, "seconds": round(elapsed, 1)})
                print(f"  seed {seed} fold {fold.fold} fit {arm.name} ({elapsed:.0f}s)",
                      flush=True)

            # Ligands are taken in order of appearance and only counted when they
            # actually own an eligible curve, so a cap of N buys N *measured*
            # ligands rather than N candidates of which most contribute nothing.
            chosen = 0
            for ligand in dict.fromkeys(test["extractant"].astype(str)):
                if max_ligands is not None and chosen >= max_ligands:
                    break
                rows = test[test["extractant"].astype(str) == ligand].reset_index(drop=True)
                # Never Python's salted ``hash``: gen8 lost cross-run pairing to it.
                rng = np.random.default_rng([int(seed), int(fold.fold)])
                n_all = eligible_curves(rows, max_curves=None)["curve_id"].nunique()
                curves = eligible_curves(rows, max_curves=max_curves, rng=rng)
                if curves.empty:
                    continue
                chosen += 1
                sampling.append({
                    "split_seed": int(seed), "fold": int(fold.fold), "extractant": ligand,
                    "n_rows": int(len(rows)), "n_eligible_curves": int(n_all),
                    "n_curves_used": int(curves["curve_id"].nunique())})
                for curve_id, block in curves.groupby("curve_id", sort=True):
                    axis = str(block["axis"].iloc[0])
                    label = str(block["axis_label"].iloc[0])
                    variants = build_variants(rows, str(curve_id), axis, curves, rng=rng)
                    if not variants:
                        continue
                    reference = variants[0]
                    assert reference.family == "REFERENCE"
                    base = {name: _predictions(model, reference.frame)
                            for name, model in fitted.items()}
                    for position, variant in enumerate(variants[1:], start=1):
                        for name, model in fitted.items():
                            prediction = _predictions(model, variant.frame)
                            if variant.family == "SHIFT":
                                stats = shift_for_shifted_window(
                                    base[name], prediction,
                                    variant.detail["paired_with"], variant.compare_ids)
                            else:
                                stats = compare_variant(base[name], prediction,
                                                        variant.compare_ids)
                            records.append({
                                "arm": name, "split_seed": int(seed),
                                "fold": int(fold.fold), "extractant": ligand,
                                "curve_id": str(curve_id), "axis_label": label,
                                "family": variant.family, "variant": variant.label,
                                # ``DENSITY`` labels itself by how many *fresh*
                                # points it inserted, and two different insert
                                # counts can round to the same fresh count when a
                                # grid point coincides with a measured one.  The
                                # position in the variant list cannot collide, and
                                # it is what the paired comparison joins on.
                                "variant_index": int(position),
                                "n_curve_rows": int(len(block)),
                                "n_eligible_curves_ligand": int(n_all),
                                "n_curves_used_ligand": int(curves["curve_id"].nunique()),
                                "window_width": float(np.ptp(
                                    block["axis_value"].to_numpy(dtype=float))),
                                **{f"detail__{k}": v for k, v in variant.detail.items()
                                   if not isinstance(v, list)},
                                **stats})
            folds_done.append({"split_seed": int(seed), "fold": int(fold.fold),
                               "n_ligands": int(chosen)})
            checkpoint()
            print(f"  seed {seed} fold {fold.fold}: {chosen} ligands, "
                  f"{len(records)} comparisons so far (checkpointed)", flush=True)

    return checkpoint(), provenance()


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #

#: Per-comparison statistics :func:`compare_variant` produces.  ``shift_*`` are the
#: absolute movement of an unchanged prediction; ``shape_shift_*`` and
#: ``level_shift`` are that movement split into the part one k-shot measurement
#: could absorb (level) and the part it could not (shape).
STATISTICS = ("shift_median", "shift_p90", "shift_p95", "shift_max",
              "shape_shift_median", "shape_shift_max", "level_shift")


def summarise(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, block in table.groupby(keys, sort=True, observed=True):
        values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, values))
        record["n_comparisons"] = int(len(block))
        record["n_ligands"] = int(block["extractant"].nunique())
        record["n_curves"] = int(block["curve_id"].nunique())
        for column in STATISTICS:
            if column not in block.columns:
                continue
            series = block[column].dropna()
            if series.empty:
                continue
            record[f"{column}__median"] = float(series.median())
            record[f"{column}__p90"] = float(series.quantile(0.90))
            record[f"{column}__p95"] = float(series.quantile(0.95))
            record[f"{column}__max"] = float(series.max())
        # How much of the typical movement a single calibration measurement could
        # remove.  A design dependence that is pure level is a nuisance; one that
        # is shape is a wrong curve.
        median_shift = record.get("shift_median__median")
        if median_shift:
            record["shape_fraction_of_shift"] = float(
                record.get("shape_shift_median__median", np.nan) / median_shift)
        rows.append(record)
    return pd.DataFrame.from_records(rows)


#: The unit a paired comparison is drawn over: the same question, asked of two arms.
UNIT = ["split_seed", "fold", "extractant", "curve_id", "family", "variant_index"]


def paired_delta(table: pd.DataFrame, baseline: str, arms: list[str]) -> pd.DataFrame:
    """Arm-minus-baseline on identical units, with a two-sided sign test.

    The arms answer the *same* questions on the *same* folds, so the unpaired
    difference of two medians throws away the pairing and understates precision
    while overstating independence.  The sign test is used rather than a t-test
    because the per-unit shift distribution is a non-negative, heavy-tailed thing
    whose mean is dominated by a handful of curves.
    """
    from scipy import stats as sstats

    base = table[table["arm"] == baseline].set_index(UNIT)
    out = []
    for arm in arms:
        if arm == baseline:
            continue
        other = table[table["arm"] == arm].set_index(UNIT)
        common = base.index.intersection(other.index)
        if not len(common):
            continue
        for family, family_block in pd.DataFrame(index=common).reset_index().groupby(
                "family", sort=True):
            index = pd.MultiIndex.from_frame(family_block[UNIT])
            for column in ("shift_median", "shift_p95", "shift_max",
                           "shape_shift_median", "level_shift"):
                if column not in base.columns:
                    continue
                a = base.loc[index, column].to_numpy(dtype=float)
                b = other.loc[index, column].to_numpy(dtype=float)
                finite = np.isfinite(a) & np.isfinite(b)
                a, b = a[finite], b[finite]
                if not len(a):
                    continue
                delta = b - a
                # Ties are exact-zero movements — common for families where both
                # arms are stable — and a sign test must drop them, not count them.
                nonzero = delta[delta != 0]
                worse = int((nonzero > 0).sum())
                p = (float(sstats.binomtest(worse, len(nonzero), 0.5).pvalue)
                     if len(nonzero) else float("nan"))
                out.append({
                    "arm": arm, "baseline": baseline, "family": family,
                    "statistic": column, "n_units": int(len(a)),
                    "baseline_median": float(np.median(a)),
                    "arm_median": float(np.median(b)),
                    "median_paired_delta": float(np.median(delta)),
                    "mean_paired_delta": float(delta.mean()),
                    "p95_paired_delta": float(np.quantile(delta, 0.95)),
                    "n_ties": int(len(delta) - len(nonzero)),
                    "frac_units_worse": float(worse / len(nonzero)) if len(nonzero) else float("nan"),
                    "sign_test_p": p,
                })
    return pd.DataFrame.from_records(out)


#: gen10's own run of this benchmark.  ``SHAPE_RECOMPOSED`` there is
#: ``RecomposedModel(representation_name="GEN9", gen9_compat=True)``, which is what
#: the gen11 transfer class collapses to when its auxiliary block is empty.
GEN10_ROWS = REPO_ROOT / "runs" / "gen10_final" / "query_consistency" / "rows.parquet"
GEN10_ANCHOR = "A_GEN10_CONTROL|JOINT|FROZEN_3|FROZEN_8|HIERARCHICAL|lam1"
GEN10_EQUIVALENT = "SHAPE_RECOMPOSED"


def gen10_reproduction(table: pd.DataFrame) -> dict:
    """Does the gen11 anchor reproduce gen10's *published* consistency numbers?

    This is stronger than the usual "it scores similarly" and it costs nothing.
    The perturbations are drawn from ``np.random.default_rng([seed, fold])``,
    created fresh per ligand, so a given (seed, fold, ligand, curve) produces
    byte-identical variants here and in gen10's run; and the gen11 transfer class
    with an empty auxiliary block *is* gen10's ``RecomposedModel``.  The two runs
    must therefore agree on every shared unit to the forest's thread-order floor.

    An agreement is a receipt that this driver reproduces the frozen benchmark
    rather than a new one that happens to be numerically nearby.  A disagreement
    would mean something about the query path differs — most likely the patched
    cohort perturbing curve construction — and every gen11 number below would be
    measured against a moved reference.
    """
    if GEN10_ANCHOR not in set(table["arm"]) or not GEN10_ROWS.exists():
        return {"status": "not attempted",
                "reason": ("gen10 anchor arm absent from this run"
                           if GEN10_ANCHOR not in set(table["arm"])
                           else f"{GEN10_ROWS} not on disk")}
    keys = ["split_seed", "fold", "extractant", "curve_id", "family", "variant"]
    columns = ["shift_median", "shift_p95", "shift_max", "level_shift", "n_compared"]
    mine = table[(table["arm"] == GEN10_ANCHOR) & (table["family"] != "SHIFT")]
    # gen10's rows carry no ``variant_index``; the (family, variant) label pair is
    # unique except for the rare DENSITY collision, which is dropped from both
    # sides rather than matched arbitrarily.
    mine = mine.drop_duplicates(subset=keys, keep=False)[keys + columns]
    theirs = pd.read_parquet(GEN10_ROWS)
    theirs = theirs[theirs["arm"] == GEN10_EQUIVALENT]
    theirs = theirs.drop_duplicates(subset=keys, keep=False)[keys + columns]
    merged = mine.merge(theirs, on=keys, how="inner", suffixes=("", "_gen10"))
    if merged.empty:
        return {"status": "no overlap", "n_gen11_units": int(len(mine)),
                "n_gen10_units": int(len(theirs))}
    out = {"status": "compared", "n_units_compared": int(len(merged)),
           "n_gen11_units": int(len(mine)), "n_gen10_units": int(len(theirs)),
           "tolerance": DETERMINISM_TOLERANCE}
    worst = 0.0
    for column in columns:
        delta = (merged[column] - merged[f"{column}_gen10"]).abs().max()
        out[f"max_abs_delta__{column}"] = float(delta)
        worst = max(worst, float(delta))
    out["max_abs_delta"] = worst
    out["reproduces_gen10"] = bool(worst <= DETERMINISM_TOLERANCE)
    return out


def accuracy_context(arm_names: list[str], arms_dir: Path,
                     seeds: list[int]) -> pd.DataFrame:
    """Macro MAE of the same arms, read off the primary run's OOF parquets.

    A consistency number alone cannot answer §16 — the question is a trade, not a
    level.  "0.02 log units less stable" means one thing against a 0.07 macro-MAE
    gain and another against nothing, so the accuracy side is *computed* here from
    whatever the primary run has already written rather than quoted.  Arms whose
    OOF is not yet on disk are reported as absent, never as zero.

    Restricted to the seeds this benchmark ran, so the two halves of the trade are
    measured on the same folds.
    """
    from lanthanide_separation.gen7.harness import score_oof

    rows = []
    for name in arm_names:
        path = arms_dir / f"oof_{name.replace('|', '__')}.parquet"
        if not path.exists():
            rows.append({"arm": name, "macro_mae": float("nan"),
                         "oof_status": "absent — not yet written by the primary run"})
            continue
        oof = pd.read_parquet(path)
        oof = oof[oof["split_seed"].isin(seeds)]
        if oof.empty:
            rows.append({"arm": name, "macro_mae": float("nan"),
                         "oof_status": f"present but carries no seed in {seeds}"})
            continue
        scores = score_oof(oof, per_seed=True)
        rows.append({"arm": name, "macro_mae": float(scores["macro_mae"].mean()),
                     "shape_mae": float(scores["shape_mae"].mean()),
                     "n_seeds_scored": int(scores["split_seed"].nunique()),
                     "oof_status": "read"})
    return pd.DataFrame.from_records(rows)


def _cell(value) -> str:
    """One markdown table cell.

    Arm names are pipe-delimited by construction (``ARM|MECHANISM|METAL|...``),
    which is also markdown's column separator, so an unescaped arm name silently
    shatters a row into eight columns.  Escaped rather than renamed, because the
    name in the table has to be the name that indexes the CSVs.
    """
    if isinstance(value, float):
        return "n/a" if not np.isfinite(value) else f"{value:.4f}"
    return str(value).replace("|", "\\|")


#: The perturbation whose gen10 number is the published limit: two candidate
#: points two decades outside the measured window, changing nothing else.
HEADLINE_FAMILY = "DECOY"
#: Movement below this is not distinguishable from the fold-to-fold noise of the
#: benchmark itself and is reported as "unchanged".  Declared here, before the
#: numbers are read, and quoted as a fraction of gen10's own median.
MATERIAL_DELTA = 0.02


def write_readme(path: Path, board: pd.DataFrame, deltas: pd.DataFrame,
                 payload: dict, provenance: dict, accuracy: pd.DataFrame) -> None:
    """The answer in words, with every number read back out of the tables.

    Written by the driver rather than by hand so the prose cannot drift from the
    CSVs beside it; the verdict sentence is a function of the measured paired
    delta, not of what the run was hoping to find.
    """
    lines: list[str] = []
    add = lines.append
    add("# gen11 §16 — query-design consistency of the auxiliary-trained arms\n")
    add("**Question.** GEN10's known limit is that its prediction for a point depends on "
        "which *other* points the user listed. Does training on the auxiliary multi-metal "
        "archive make that dependence better, worse, or unchanged?\n")

    baseline = payload["baseline"]
    zero = payload.get("zero_control_max_shift")
    add("## Correctness check\n")
    add(f"- Design-blind control (`{ZERO_CONTROL}`, gen10's frozen monolith, representation "
        f"`NONE`): max shift over every perturbation = **{zero!r}** "
        f"(tolerance {payload['exact_invariance_tolerance']:g}); "
        f"passes = **{payload.get('zero_control_is_zero')}**.")
    add(f"- Exact invariances (`CONTEXT`, `PERMUTE`, every arm): max shift = "
        f"**{payload['exact_invariance_max_shift']:.3g}**; passes = "
        f"**{payload['exact_invariance_holds']}**.")
    add("\nBoth are properties the harness cannot fail unless it is wired wrong, which is "
        "exactly why they are the check: a non-zero design-blind control would mean the "
        "benchmark was measuring its own plumbing rather than the model.\n")
    repro = payload.get("gen10_reproduction", {})
    if repro.get("status") == "compared":
        add(f"- Reproduction of gen10's own run: the gen11 anchor "
            f"(`{GEN10_ANCHOR}`, empty auxiliary block) against gen10's "
            f"`{GEN10_EQUIVALENT}` on {repro['n_units_compared']} shared "
            f"(seed, fold, ligand, curve, variant) units — max |Δ| = "
            f"**{repro['max_abs_delta']:.3g}**, reproduces = "
            f"**{repro['reproduces_gen10']}**. The perturbations are drawn from "
            f"`default_rng([seed, fold])` created fresh per ligand, so the two runs ask "
            f"literally the same questions; this is the receipt that gen11 measures the "
            f"frozen benchmark rather than a nearby one.\n")
    else:
        add(f"- Reproduction of gen10's own run: {repro.get('status', 'not attempted')}"
            f" ({repro.get('reason', '')}).\n")

    add("## What was run\n")
    add(f"- **{payload['n_seeds']} split seeds** ({payload['seeds']}) — a deliberate "
        f"reduction from gen10's five, which took 4.6 h for three arms. Say \"2 seeds\" "
        f"whenever these numbers are quoted.")
    add(f"- All {len(payload['variant_families'])} perturbation families were kept "
        f"({', '.join(payload['variant_families'])}). The seed count is the only thing cut; "
        f"the perturbations never are.")
    cap = payload["max_ligands_per_fold"]
    sample = f"{cap} measured ligands per fold" if cap else "every test ligand of every fold"
    add(f"- {sample}, over "
        f"{payload['n_folds_completed']} completed (seed, fold) blocks, at most "
        f"{payload['max_curves_per_ligand']} curves each (gen10's own cap): "
        f"**{payload['n_ligands']} distinct ligands, {payload['n_curves']} curves, "
        f"{payload['n_comparisons']} comparisons**. `sampling.json` lists every one.")
    add(f"- Arms present: {', '.join(payload['arms_present'])}."
        + (f" Absent: {', '.join(payload['arms_absent'])}." if payload["arms_absent"] else ""))
    add(f"- Auxiliary pool: {provenance.get('pool_rows')} cells, mean "
        f"{provenance.get('pool_admissible_mean_over_folds', float('nan')):.0f} admissible "
        f"per fold after the leakage filter. Cohort fingerprint "
        f"`{provenance.get('cohort_fingerprint')}`.\n")

    add("## Movement of the *unchanged* points, in log units\n")
    add("Each row is one arm under one perturbation family. `median` is the median over "
        "comparisons of the per-comparison median shift; `p95` the median of the "
        "per-comparison p95; `p95-of-p95` the 95th percentile of those; `worst` the single "
        "largest movement seen. `level` and `shape` split the movement into the part one "
        "calibration measurement could absorb and the part it could not.\n")
    columns = ["arm", "family", "n_comparisons", "shift_median__median",
               "shift_p95__median", "shift_p95__p95", "shift_max__max",
               "level_shift__median", "shape_shift_median__median"]
    header = ["arm", "family", "n", "median", "p95", "p95-of-p95", "worst", "level", "shape"]
    present = [c for c in columns if c in board.columns]
    names = [h for c, h in zip(columns, header) if c in board.columns]
    add("| " + " | ".join(names) + " |")
    add("|" + "---|" * len(names))
    for _, row in board.sort_values(["family", "arm"]).iterrows():
        add("| " + " | ".join(_cell(row[c]) for c in present) + " |")
    add("")

    add("## Paired difference from the design-matched control\n")
    add(f"Baseline: `{baseline}` — the control at the same design corner "
        f"(`GENERAL` metal scheme, `ANNOTATION_SAFE` MASSACTION) every auxiliary arm is "
        f"forced to. Comparing against gen10's frozen corner instead would confound the "
        f"design change with the transfer. Units are paired: the same seed, fold, ligand, "
        f"curve and variant asked of both arms. Positive = the auxiliary arm is *less* "
        f"stable.\n")
    verdicts: list[str] = []
    if len(deltas):
        # The design-blind control is in ``delta_vs_baseline.csv`` — its delta is
        # exactly minus the baseline's own movement, which is a restatement of the
        # zero check, not a comparison — so it is kept out of the table read as
        # "which model is less stable".
        head = deltas[(deltas["statistic"] == "shift_median")
                      & (deltas["arm"] != ZERO_CONTROL)]
        add("| arm | family | n units | baseline median | arm median | paired Δ | frac worse | sign-test p |")
        add("|---|---|---|---|---|---|---|---|")
        for _, row in head.sort_values(["arm", "family"]).iterrows():
            add(f"| {_cell(row['arm'])} | {row['family']} | {int(row['n_units'])} | "
                f"{row['baseline_median']:.4f} | {row['arm_median']:.4f} | "
                f"{row['median_paired_delta']:+.4f} | {row['frac_units_worse']:.3f} | "
                f"{row['sign_test_p']:.3g} |")
        add("")
        roles = payload.get("arm_roles", {})
        for arm in sorted(head["arm"].unique()):
            block = head[(head["arm"] == arm) & (head["family"] == HEADLINE_FAMILY)]
            if block.empty:
                block = head[head["arm"] == arm]
            delta = float(block["median_paired_delta"].iloc[0])
            base_value = float(block["baseline_median"].iloc[0])
            family = str(block["family"].iloc[0])
            p_value = float(block["sign_test_p"].iloc[0])
            direction = ("WORSE" if delta > MATERIAL_DELTA else
                         "BETTER" if delta < -MATERIAL_DELTA else "UNCHANGED")
            relative = (delta / base_value * 100.0) if base_value else float("nan")
            # The gen10 anchor is not a transfer arm.  Its delta measures the
            # *design corner* — GENERAL metal + ANNOTATION_SAFE MASSACTION against
            # the frozen blocks — with no auxiliary row anywhere in it, and reading
            # it as "auxiliary training did this" is precisely the confound the
            # four-corner control grid exists to prevent.
            kind = ("design corner only, no auxiliary rows"
                    if roles.get(arm) == "gen10_anchor" else "auxiliary training")
            verdicts.append(
                f"- `{arm}` ({kind}) under `{family}`: **{direction}** — paired median "
                f"shift moves {delta:+.4f} log units ({relative:+.1f} % of the control's "
                f"{base_value:.4f}), sign-test p = {p_value:.3g}.")
    else:
        add("_No paired table: the baseline arm was absent from this run._\n")

    add("## The other half of the trade: accuracy\n")
    add("Macro MAE of the same arms on the same 2 seeds, read from the primary run's OOF "
        "parquets under `runs/gen11_transfer/arms/`. §16 is a trade — instability is only "
        "a verdict once it is set against what the arm bought.\n")
    add("| arm | macro MAE | shape MAE | status |")
    add("|---|---|---|---|")
    for _, row in accuracy.iterrows():
        add(f"| {_cell(row['arm'])} | {_cell(float(row.get('macro_mae', np.nan)))} | "
            f"{_cell(float(row.get('shape_mae', np.nan)))} | {row['oof_status']} |")
    add("")

    add("## The answer\n")
    if verdicts:
        add(f"Judged on `{HEADLINE_FAMILY}` — two candidate points two decades outside the "
            f"measured window, the perturbation gen10 published its ~0.12 limit on — and "
            f"calling anything under {MATERIAL_DELTA} log units unchanged:\n")
        lines.extend(verdicts)
        add("")
    add("Read the per-family table above before generalising: a family where the paired "
        "delta is small but the control's own movement is large is still a model whose "
        "answer depends on the question, and auxiliary training was never expected to "
        "*fix* that — only not to make it worse.\n")
    path.write_text("\n".join(lines))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2,
                        help="how many of DEFAULT_SEEDS to use (2 is the declared budget)")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--max-ligands", type=int, default=0,
                        help="measured ligands per fold; 0 means every test ligand, which "
                             "is gen10 parity. This is the sample, never the perturbations")
    parser.add_argument("--max-curves", type=int, default=MAX_CURVES_PER_LIGAND)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="forest threads at predict time; gen10 used 1 because "
                             "thousands of tiny predicts oversubscribe more than they "
                             "parallelise, and the answer is identical either way")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--summarise-only", action="store_true")
    args = parser.parse_args(argv)

    arms = arm_table(args.arms)
    seeds = list(DEFAULT_SEEDS[:max(1, min(5, args.seeds))])
    max_ligands = args.max_ligands if args.max_ligands and args.max_ligands > 0 else None
    started = time.time()
    if args.summarise_only:
        table = pd.read_parquet(args.out / "rows.parquet")
        provenance = json.loads((args.out / "sampling.json").read_text())
        arm_names = sorted(table["arm"].unique())
    else:
        table, provenance = run(arms, seeds, max_ligands, args.max_curves,
                                args.n_jobs, args.out)
        arm_names = [a.name for a in arms]
    if table.empty:
        raise SystemExit("no eligible curves — nothing was measured")

    args.out.mkdir(parents=True, exist_ok=True)
    non_shift = table[table["family"] != "SHIFT"]
    shifted = table[table["family"] == "SHIFT"]

    summarise(non_shift, ["arm", "family"]).to_csv(
        args.out / "summary_by_arm.csv", index=False)
    summarise(non_shift, ["arm", "family", "variant"]).to_csv(
        args.out / "by_perturbation.csv", index=False)
    if len(shifted):
        summarise(shifted, ["arm", "axis_label"]).to_csv(
            args.out / "shifted_window.csv", index=False)

    present = [a for a in arm_names if a in set(table["arm"])]
    deltas = pd.DataFrame()
    if BASELINE in present:
        deltas = paired_delta(non_shift, BASELINE, present)
        deltas.to_csv(args.out / "delta_vs_baseline.csv", index=False)

    control = non_shift[non_shift["arm"] == ZERO_CONTROL]
    exact = non_shift[non_shift["family"].isin(["CONTEXT", "PERMUTE"])]
    exact_max = float(exact["shift_max"].max()) if len(exact) else float("nan")
    payload = {
        "seeds": seeds, "n_seeds": len(seeds),
        "arms_requested": arm_names,
        "arm_roles": {a.name: a.role for a in arm_table(None)},
        "arms_present": present,
        "arms_absent": sorted(set(arm_names) - set(present)),
        "baseline": BASELINE,
        "max_ligands_per_fold": max_ligands,
        "max_curves_per_ligand": args.max_curves,
        "folds_completed": provenance.get("folds_completed", []),
        "n_folds_completed": len(provenance.get("folds_completed", [])),
        "variant_families": list(VARIANTS),
        "n_comparisons": int(len(table)),
        "n_ligands": int(table["extractant"].nunique()),
        "n_curves": int(table["curve_id"].nunique()),
        "exact_invariance_families": ["CONTEXT", "PERMUTE"],
        "exact_invariance_max_shift": exact_max,
        "exact_invariance_tolerance": DETERMINISM_TOLERANCE,
        "exact_invariance_holds": bool(exact_max <= DETERMINISM_TOLERANCE),
        "elapsed_s": round(time.time() - started, 1),
    }
    if len(control):
        payload["zero_control_max_shift"] = float(control["shift_max"].max())
        payload["zero_control_is_zero"] = bool(
            control["shift_max"].max() <= DETERMINISM_TOLERANCE)
    else:
        payload["zero_control_max_shift"] = None
        payload["zero_control_is_zero"] = None
    payload["gen10_reproduction"] = gen10_reproduction(table)
    (args.out / "control.json").write_text(json.dumps(payload, indent=2))
    (args.out / "sampling.json").write_text(json.dumps(provenance, indent=2))

    board = summarise(non_shift, ["arm", "family"])
    accuracy = accuracy_context(
        [a for a in present if a != ZERO_CONTROL],
        REPO_ROOT / "runs" / "gen11_transfer" / "arms", seeds)
    accuracy.to_csv(args.out / "accuracy_context.csv", index=False)
    write_readme(args.out / "README.md", board, deltas, payload, provenance, accuracy)

    print("\n=== gen11 query-set consistency, shift in log units ===")
    columns = [c for c in ("arm", "family", "n_comparisons", "n_ligands",
                           "shift_median__median", "shift_p95__median",
                           "shift_p95__p95", "shift_max__max",
                           "level_shift__median", "shape_shift_median__median")
               if c in board.columns]
    print(board[columns].round(4).to_string(index=False))
    if len(deltas):
        print("\n=== paired delta vs the design-matched control ===")
        head = deltas[(deltas["statistic"] == "shift_median")
                      & (deltas["arm"] != ZERO_CONTROL)]
        print(head[["arm", "family", "n_units", "baseline_median", "arm_median",
                    "median_paired_delta", "frac_units_worse", "sign_test_p"]
                   ].round(4).to_string(index=False))

    # The two hard checks.  Both are properties of every arm in the stack, so a
    # failure is a harness bug and the numbers above mean nothing.
    if len(control) and control["shift_max"].max() > DETERMINISM_TOLERANCE:
        raise SystemExit(
            f"the design-blind control moved by {control['shift_max'].max():.3g}; the "
            "benchmark is measuring its own plumbing")
    if np.isfinite(exact_max) and exact_max > DETERMINISM_TOLERANCE:
        raise SystemExit(
            f"a CONTEXT/PERMUTE variant moved a prediction by {exact_max:.3g}; these are "
            "exact invariances of every arm and anything above the float floor is a bug")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

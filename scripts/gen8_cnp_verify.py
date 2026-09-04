#!/usr/bin/env python
"""Adversarial verification of the gen8 §5 CNP result.

Same fold plan, same adapters, same driver as ``scripts/gen8_cnp.py``.  Three
things are added, all of them designed to *break* the reported result rather than
confirm it:

``--poison``
    every adapter's ``predict`` is wrapped so the frame it receives has ``log_D``
    and every other target-derived column replaced by NaN.  ``observed`` is still
    ``truth[selected]``, taken from the *unpoisoned* frame by the driver before
    the wrapper runs, so a method that only reads ``observed`` is numerically
    unaffected.  Any change in the output parquet is proof the method reads the
    target of rows it did not measure.

``--trace``
    the wrapper additionally records which columns of the block each adapter
    touched, via a ``__getitem__``-logging proxy, so a leak is caught by name and
    not only by its numerical footprint.

``--count-fallbacks``
    every ``predict`` return is compared to the passed ``prediction`` array; an
    exact tie is counted.  A silent ``except -> return prediction`` fallback and a
    genuine "this method decided not to move" are indistinguishable in the score,
    so both are counted here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402
from lanthanide_separation.gen8 import evaluate as gen8_evaluate  # noqa: E402
from lanthanide_separation.gen8.adapters import default_adapters  # noqa: E402
from lanthanide_separation.gen8.cnp import (  # noqa: E402
    LIGAND_COLUMNS, MASSACTION_COLUMNS, METAL_COLUMNS, RECOVERED_COLUMNS,
    build_cnp_adapters, parameter_count,
)

CACHE = ROOT / "runs" / "gen7_architecture" / "cache"
FINALISTS = ROOT / "runs" / "gen7_architecture" / "finalists"
OUT = ROOT / "runs" / "gen8_architecture" / "cnp_verify"
MODEL = "REC_ecfp_plus_recovered"
MODEL_SEED = lambda fold: 42 + fold * 1009 + 9999991  # noqa: E731

#: Columns of the merged block that are functions of the target.  ``prediction`` is
#: NOT one of them: it is the frozen model's output and a legitimate adapter input.
TARGET_DERIVED = ("log_D", "extra__offset_hat", "extra__deconfound_offset_sd",
                  "extra__anchor_spread", "extra__padre_pairs",
                  "extra__selected_arm_index", "extra__prediction_sd", "disagreement")

TOUCHED: Counter = Counter()
TIES: Counter = Counter()
CALLS: Counter = Counter()
TIES_K0: Counter = Counter()
CALLS_K0: Counter = Counter()


# --------------------------------------------------------------------------- #
# Adversarial references: the CNP's own claim, without the network
# --------------------------------------------------------------------------- #

class ShrinkOffset:
    """``prediction + lambda_k * mean(measured residual)`` with lambda_k fitted on the fold.

    ``CNPRES_shrinkage`` is a network whose head sees only ``(log1p(k)/2, mean, sd)``
    of the measured residuals, so the only function it can express is a shrinkage
    schedule in k.  If that is the whole of its reported gain, a two-parameter grid
    search on the *same* training rows must reach the same place.  This adapter is
    that grid search: episodic, macro-weighted, one lambda per k, nothing learned
    about conditions or chemistry.  It is trained under the identical contract
    (``fit_fold`` on the fold's training rows only) and reads only ``observed``.
    """

    def __init__(self, name: str = "SHRINK_FITTED", draws: int = 40):
        self.name = name
        self.draws = draws
        self.lam: dict[int, float] = {}

    def fit_fold(self, train, y_train, *, split_seed, fold, model_seed):
        rng = np.random.default_rng(int(model_seed) + 31337)
        p = pd.to_numeric(train["prediction"], errors="coerce").to_numpy(dtype=float)
        y = np.asarray(y_train, dtype=float)
        residual = y - p
        ligands = train["extractant"].to_numpy()
        groups = [np.flatnonzero(ligands == name) for name in np.unique(ligands)]
        groups = [g for g in groups if len(g) >= 4 and np.isfinite(residual[g]).all()]
        grid = np.linspace(0.0, 1.3, 131)
        self.lam = {}
        for k in (1, 2, 3, 5):
            per_ligand = []
            for rows in groups:
                if len(rows) <= k:
                    continue
                errors = np.zeros(len(grid))
                for _ in range(self.draws):
                    order = rng.permutation(len(rows))
                    context, targets = rows[order[:k]], rows[order[k:]]
                    m = float(residual[context].mean())
                    errors += np.abs(residual[targets][None, :] - grid[:, None] * m).mean(axis=1)
                per_ligand.append(errors / self.draws)
            self.lam[k] = float(grid[int(np.argmin(np.mean(per_ligand, axis=0)))]) \
                if per_ligand else 1.0

    def predict(self, block, prediction, selected, observed, context):
        prediction = np.asarray(prediction, dtype=float)
        k = int(len(selected))
        if k == 0 or not self.lam:
            return prediction
        lam = self.lam.get(k, self.lam[min(self.lam, key=lambda j: abs(j - k))])
        return prediction + lam * float(np.mean(observed - prediction[selected]))


class ShrinkAnalytic:
    """The James-Stein shrinkage implied by the fold's own variance decomposition.

    ``lambda*_k = s2_level / (s2_level + s2_within / k)``, both variances estimated
    from the frozen model's residuals on the *training* rows only.  Zero fitted
    degrees of freedom beyond two moments.  If this matches the network, the
    network learned two numbers.
    """

    def __init__(self, name: str = "SHRINK_ANALYTIC"):
        self.name = name
        self.s2_level = 1.0
        self.s2_within = 0.0

    def fit_fold(self, train, y_train, *, split_seed, fold, model_seed):
        p = pd.to_numeric(train["prediction"], errors="coerce").to_numpy(dtype=float)
        residual = np.asarray(y_train, dtype=float) - p
        ligands = train["extractant"].to_numpy()
        frame = pd.DataFrame({"lig": ligands, "r": residual}).dropna()
        level = frame.groupby("lig")["r"].mean()
        within = frame["r"] - frame["lig"].map(level)
        self.s2_level = float(np.var(level.to_numpy(), ddof=1)) if len(level) > 1 else 1.0
        self.s2_within = float(np.var(within.to_numpy(), ddof=1)) if len(within) > 1 else 0.0

    def predict(self, block, prediction, selected, observed, context):
        prediction = np.asarray(prediction, dtype=float)
        k = int(len(selected))
        if k == 0:
            return prediction
        lam = self.s2_level / max(self.s2_level + self.s2_within / k, 1e-9)
        return prediction + lam * float(np.mean(observed - prediction[selected]))


class TracingFrame(pd.DataFrame):
    """A frame that records every column name an adapter asks for."""

    _metadata = ["_owner"]

    @property
    def _constructor(self):
        return pd.DataFrame

    def __getitem__(self, key):
        owner = getattr(self, "_owner", "?")
        names = [key] if isinstance(key, str) else (
            list(key) if isinstance(key, (list, tuple, pd.Index)) else [])
        for name in names:
            if isinstance(name, str):
                TOUCHED[(owner, name)] += 1
        return super().__getitem__(key)


def instrument(adapter, *, poison: bool, trace: bool, count_ties: bool):
    original = adapter.predict
    name = adapter.name

    def wrapped(block, prediction, selected, observed, context):
        frame = block
        if poison:
            frame = block.copy()
            for column in TARGET_DERIVED:
                if column in frame.columns:
                    frame[column] = np.nan
        if trace:
            frame = TracingFrame(frame if poison else block.copy())
            object.__setattr__(frame, "_owner", name)
        out = original(frame, prediction, selected, observed, context)
        if count_ties:
            values = np.asarray(out, dtype=float)
            reference = np.asarray(prediction, dtype=float)
            tie = values.shape == reference.shape and np.array_equal(values, reference)
            # k = 0 is the *designed* tie for an anchored residual adapter; k >= 1 is
            # the one that would mean a silent fallback scored as zero-shot.
            if len(selected) == 0:
                CALLS_K0[name] += 1
                TIES_K0[name] += int(tie)
            else:
                CALLS[name] += 1
                TIES[name] += int(tie)
        return out

    adapter.predict = wrapped
    return adapter


def load_cohort() -> pd.DataFrame:
    all_columns = pd.read_parquet(CACHE / "cohort.parquet").columns
    cond = [c for c in all_columns if c.startswith("cond__")]
    wanted = ["row_id"] + list(METAL_COLUMNS) + list(MASSACTION_COLUMNS) + cond \
        + [c for c in LIGAND_COLUMNS if c in all_columns]
    wanted = list(dict.fromkeys(c for c in wanted if c in all_columns))
    cohort = pd.read_parquet(CACHE / "cohort.parquet", columns=wanted)
    recovered = pd.read_parquet(CACHE / "recovered_cells.parquet",
                                columns=["row_id"] + list(RECOVERED_COLUMNS))
    cohort = cohort.merge(recovered, on="row_id", how="left", validate="one_to_one")
    assert cohort["row_id"].is_unique
    return cohort


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="104729")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-ligands", type=int, default=24)
    parser.add_argument("--policies", default="RANDOM")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tag", default="verify")
    parser.add_argument("--poison", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--no-attentive", action="store_true")
    parser.add_argument("--no-shrink-reference", action="store_true")
    args = parser.parse_args()

    oof = pd.read_parquet(FINALISTS / "oof_predictions.parquet")
    oof = oof[oof["model"] == MODEL].reset_index(drop=True)
    seeds = sorted(oof["split_seed"].unique().tolist()) if args.seeds == "all" \
        else [int(s) for s in args.seeds.split(",")]
    oof = oof[oof["split_seed"].isin(seeds)].reset_index(drop=True)

    cohort = load_cohort()
    identity = pd.read_parquet(CACHE / "cohort.parquet",
                               columns=["row_id", "extractant", "tanimoto_cluster", "log_D"])
    assert (cohort["row_id"].to_numpy() == identity["row_id"].to_numpy()).all()
    groups = identity["tanimoto_cluster"].astype(str).to_numpy()

    plan: dict[tuple[int, int], np.ndarray] = {}
    for split_seed in seeds:
        marked = oof[oof["split_seed"] == split_seed].set_index("row_id")["fold"].to_dict()
        for fold, (train_index, test_index) in enumerate(
                seeded_group_kfold(groups, 5, int(split_seed))):
            observed = np.array([marked[r] for r in identity["row_id"].to_numpy()[test_index]])
            if not (observed == fold).all():
                raise SystemExit(f"fold plan disagrees at seed {split_seed} fold {fold}")
            assert not (set(groups[train_index]) & set(groups[test_index])), "chemotype leak"
            plan[(int(split_seed), fold)] = train_index
    print(f"fold plan verified against {MODEL} for seeds {seeds}", flush=True)

    prediction_by_seed = {
        int(s): oof[oof["split_seed"] == s].set_index("row_id")["prediction"] for s in seeds}

    def fold_trainer(split_seed: int, fold: int):
        index = plan[(int(split_seed), int(fold))]
        frame = cohort.iloc[index].copy()
        frame["extractant"] = identity["extractant"].to_numpy()[index]
        frame["tanimoto_cluster"] = identity["tanimoto_cluster"].to_numpy()[index]
        frame["prediction"] = prediction_by_seed[int(split_seed)].reindex(
            frame["row_id"]).to_numpy()
        y = identity["log_D"].to_numpy(dtype=float)[index]
        return frame, y, MODEL_SEED(int(fold))

    adapters = build_cnp_adapters(steps=args.steps, batch_ligands=args.batch_ligands,
                                  include_attentive=not args.no_attentive,
                                  threads=args.threads)
    adapters = list(adapters) + list(default_adapters())
    if not args.no_shrink_reference:
        adapters += [ShrinkOffset(), ShrinkAnalytic()]
    for adapter in adapters:
        instrument(adapter, poison=args.poison, trace=args.trace, count_ties=True)
    print(f"{len(adapters)} adapters, poison={args.poison}, trace={args.trace}", flush=True)

    started = time.time()
    detail = gen8_evaluate.evaluate_fewshot(
        oof, cohort, adapters,
        policies=tuple(p for p in args.policies.split(",") if p),
        repeats=args.repeats, fold_trainer=fold_trainer, verbose=True)
    print(f"elapsed {time.time() - started:.1f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"detail_{args.tag}.parquet"
    detail.to_parquet(path, index=False)

    audit = {a.name: {"parameters": parameter_count(a) if hasattr(a, "net") else 0,
                      "internal_fallbacks": int(getattr(a, "failures", 0)),
                      "last_error": getattr(a, "last_error", ""),
                      "predict_calls_k_ge_1": int(CALLS[a.name]),
                      "returned_prediction_exactly_k_ge_1": int(TIES[a.name]),
                      "predict_calls_k0": int(CALLS_K0[a.name]),
                      "returned_prediction_exactly_k0": int(TIES_K0[a.name]),
                      "lambda": getattr(a, "lam", None) or (
                          {"s2_level": getattr(a, "s2_level", None),
                           "s2_within": getattr(a, "s2_within", None)}
                          if hasattr(a, "s2_level") else None)}
             for a in adapters}
    (OUT / f"audit_{args.tag}.json").write_text(json.dumps(audit, indent=2))
    for name, row in audit.items():
        print(f"  {name:28s} params={row['parameters']:>6,} "
              f"internal_fallbacks={row['internal_fallbacks']} "
              f"ties(k>=1)={row['returned_prediction_exactly_k_ge_1']}/{row['predict_calls_k_ge_1']} "
              f"ties(k=0)={row['returned_prediction_exactly_k0']}/{row['predict_calls_k0']} "
              f"{row['last_error']}", flush=True)

    if args.trace:
        touched = {}
        for (owner, column), count in TOUCHED.items():
            touched.setdefault(owner, {})[column] = count
        (OUT / f"touched_{args.tag}.json").write_text(json.dumps(touched, indent=2, sort_keys=True))
        for owner, columns in sorted(touched.items()):
            bad = [c for c in columns if c in TARGET_DERIVED]
            print(f"  TRACE {owner:28s} {len(columns)} columns; TARGET-DERIVED READS: {bad}",
                  flush=True)

    summary = gen8_evaluate.summarise(detail)
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(summary.to_string(index=False))
    print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

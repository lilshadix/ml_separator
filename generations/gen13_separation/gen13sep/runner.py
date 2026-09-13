"""Run a ladder of arms over the frozen fold plan and write the pair table.

For each (split seed, fold) the runner builds one ``FitContext`` from the training
cells, fits every arm on it, and scores every arm on byte-identical held-out pairs.
The output is one long parquet per arm under ``predictions/<design>/`` with columns

    split_seed, fold, cell_id, extractant, chemotype, n_metals, A, B, dZ, y, prediction

plus ``fold_plan.csv`` and ``similarity.parquet`` (max train Tanimoto per held-out cell).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from . import paths
from .cohort import Gen13Cohort
from .features import FeatureTable
from .metals import ATOMIC_NUMBER, LANTHANIDES
from .metrics import pair_rows_for_cell
from .models import Arm, FitContext
from .splits import Fold, all_folds, assert_fold_integrity, fold_plan, max_train_tanimoto

Z_VEC = np.array([ATOMIC_NUMBER[m] for m in LANTHANIDES])


@dataclass(frozen=True)
class RunSpec:
    design: str
    blocks: tuple[str, ...]
    seeds: tuple[int, ...]
    key_mode: str
    label: str

    def as_dict(self) -> dict:
        return {"design": self.design, "blocks": list(self.blocks), "seeds": list(self.seeds),
                "key_mode": self.key_mode, "label": self.label}


def _pair_frame(cohort: Gen13Cohort, fold: Fold, Y: np.ndarray) -> pd.DataFrame:
    frame = cohort.frame
    rows = []
    for local, ci in enumerate(fold.test_index):
        for a, b, y in pair_rows_for_cell(Y[ci]):
            rows.append({"split_seed": fold.seed, "fold": fold.fold, "cell_local": local,
                         "cell_id": frame["cell_id"].iat[ci], "extractant": frame["extractant"].iat[ci],
                         "chemotype": frame["chemotype"].iat[ci], "n_metals": int(frame["n_metals"].iat[ci]),
                         "A": LANTHANIDES[a], "B": LANTHANIDES[b], "ia": a, "ib": b,
                         "dZ": int(Z_VEC[b] - Z_VEC[a]), "y": y})
    return pd.DataFrame(rows)


def run_ladder(cohort: Gen13Cohort, features: FeatureTable, arms: Sequence[Arm], spec: RunSpec,
               *, folds: Sequence[Fold] | None = None, write: bool = True, verbose: bool = True) -> dict[str, pd.DataFrame]:
    frame = cohort.frame
    Y = cohort.target_matrix
    X = features.matrix(spec.blocks).to_numpy(dtype=float)
    offsets = {}
    start = 0
    for b in spec.blocks:
        n = len(features.blocks[b]); offsets[b] = np.arange(start, start + n); start += n
    fp = features.matrix(["ECFP"]).to_numpy(dtype=float)
    groups = frame["chemotype"].to_numpy()
    folds = list(folds) if folds is not None else all_folds(frame, design=spec.design, seeds=spec.seeds)
    integrity = assert_fold_integrity(frame, folds)
    out_dir = paths.PREDICTION_DIR / spec.label
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        fold_plan(frame, folds).to_csv(out_dir / "fold_plan.csv", index=False)
        max_train_tanimoto(frame, folds, fp).to_parquet(out_dir / "similarity.parquet", index=False)
        with open(out_dir / "run_spec.json", "w", encoding="utf-8") as fh:
            json.dump({**spec.as_dict(), "integrity": integrity, "cohort_fingerprint": cohort.fingerprint(),
                       "n_cells": int(len(frame)), "arms": [a.name for a in arms],
                       "n_features": int(X.shape[1])}, fh, indent=2)

    tables: dict[str, list[pd.DataFrame]] = {a.name: [] for a in arms}
    curve_tables: dict[str, list[pd.DataFrame]] = {a.name: [] for a in arms if not a.pairwise_only}
    bases: list[dict] = []
    timings: list[dict] = []
    selections: list[dict] = []
    for fold in folds:
        pairs = _pair_frame(cohort, fold, Y)
        ctx = FitContext(X_train=X[fold.train_index], Y_train=Y[fold.train_index],
                         groups_train=groups[fold.train_index], fingerprints_train=fp[fold.train_index],
                         seed=fold.model_seed,
                         inner_train=np.searchsorted(fold.train_index, fold.inner_train_index),
                         inner_validation=np.searchsorted(fold.train_index, fold.inner_validation_index),
                         extra={"blocks": offsets})
        X_te = X[fold.test_index]; fp_te = fp[fold.test_index]
        ia = pairs["ia"].to_numpy(); ib = pairs["ib"].to_numpy(); rows = pairs["cell_local"].to_numpy()
        for arm in arms:
            t0 = time.time()
            arm.fit(ctx)
            if arm.pairwise_only:
                pred = arm.predict_pairs(X_te, ia, ib, rows)
            else:
                curve = arm.predict_curves(X_te, fp_te)
                pred = curve[rows, ia] - curve[rows, ib]
                ct = pd.DataFrame(curve, columns=[f"c__{m}" for m in LANTHANIDES])
                ct.insert(0, "cell_id", frame["cell_id"].to_numpy()[fold.test_index])
                ct.insert(0, "fold", fold.fold); ct.insert(0, "split_seed", fold.seed)
                curve_tables[arm.name].append(ct)
                basis = getattr(getattr(arm, "chosen_", arm), "basis_", None)
                if basis is not None:
                    bases.append({"arm": arm.name, "split_seed": fold.seed, "fold": fold.fold,
                                  "basis": np.asarray(basis).tolist()})
            if not np.isfinite(pred).all():
                raise RuntimeError(f"{arm.name}: non-finite prediction at seed {fold.seed} fold {fold.fold}")
            t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
            t["prediction"] = pred.astype(float)
            tables[arm.name].append(t)
            if hasattr(arm, "selection_") and arm.selection_:
                selections.append({"arm": arm.name, "split_seed": fold.seed, "fold": fold.fold, **arm.selection_[-1]})
            timings.append({"arm": arm.name, "split_seed": fold.seed, "fold": fold.fold,
                            "seconds": round(time.time() - t0, 2), "n_train_cells": int(len(fold.train_index)),
                            "n_test_cells": int(len(fold.test_index)), "n_test_pairs": int(len(pairs))})
        if verbose:
            print(f"[{spec.label}] seed {fold.seed} fold {fold.fold}: {len(fold.train_index)} train / "
                  f"{len(fold.test_index)} test cells, {len(pairs)} pairs", flush=True)
    results = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    if write:
        for name, table in results.items():
            table.to_parquet(out_dir / f"{name}.parquet", index=False)
        pd.DataFrame(timings).to_csv(out_dir / "timings.csv", index=False)
        (out_dir / "curves").mkdir(exist_ok=True)
        for name, parts in curve_tables.items():
            if parts:
                pd.concat(parts, ignore_index=True).to_parquet(out_dir / "curves" / f"{name}.parquet", index=False)
        with open(out_dir / "bases.json", "w", encoding="utf-8") as fh:
            json.dump(bases, fh)
        if selections:
            pd.DataFrame(selections).to_csv(out_dir / "selections.csv", index=False)
    return results


def load_pair_table(label: str, arms: Sequence[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Wide pair table: one column per arm, aligned on the pair key."""
    out_dir = paths.PREDICTION_DIR / label
    files = sorted(out_dir.glob("*.parquet"))
    names = [f.stem for f in files if f.stem != "similarity"]
    if arms is not None:
        names = [n for n in names if n in set(arms)]
    key = ["split_seed", "fold", "cell_id", "extractant", "chemotype", "n_metals", "A", "B", "dZ", "y"]
    wide: pd.DataFrame | None = None
    for name in names:
        t = pd.read_parquet(out_dir / f"{name}.parquet").rename(columns={"prediction": name})
        wide = t if wide is None else wide.merge(t, on=key, how="inner", validate="one_to_one")
    if wide is None:
        raise FileNotFoundError(f"no arm tables under {out_dir}")
    return wide, names

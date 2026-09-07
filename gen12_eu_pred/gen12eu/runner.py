"""Run one contender over the whole fold plan and emit its out-of-fold predictions.

The runner owns the contract so no experiment can quietly relax it:

* every arm sees the identical fold plan, built once and reused;
* a contender is handed the training frame, its targets, the test frame and an
  inner validation block carved from the training rows — never the fold indices,
  never the test targets;
* the test targets are compared against a hash taken before the arm ran, so an
  arm that mutated the truth is caught rather than rewarded;
* the emitted frame is one row per (split_seed, fold, row_id) with the
  prediction, the band and the similarity, which is what every table and every
  bootstrap downstream is aggregated from.
"""
from __future__ import annotations

import hashlib
import time
from typing import Sequence

import numpy as np
import pandas as pd

from . import splits
from .chemistry import band_of
from .models import TARGET, FoldContext


def _target_digest(values: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(values, dtype=float).tobytes(),
                           digest_size=8).hexdigest()


def run_contender(contender, frame: pd.DataFrame, folds: Sequence[splits.Fold],
                  blocks: dict, similarity: pd.DataFrame | None = None,
                  *, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns ``(predictions, per_fold_selection)``."""
    records: list[pd.DataFrame] = []
    selections: list[dict] = []
    for fold in folds:
        started = time.time()
        train = frame.iloc[fold.train_index]
        test = frame.iloc[fold.test_index]
        y_train = train[TARGET].to_numpy(dtype=float)
        y_test = test[TARGET].to_numpy(dtype=float)
        before = _target_digest(y_test)
        context = FoldContext(
            blocks=blocks, model_seed=fold.model_seed,
            inner_train=frame.iloc[fold.inner_train_index],
            inner_validation=frame.iloc[fold.inner_validation_index],
            y_inner_train=frame.iloc[fold.inner_train_index][TARGET].to_numpy(dtype=float),
            y_inner_validation=frame.iloc[fold.inner_validation_index][TARGET].to_numpy(dtype=float),
            split_seed=fold.seed, fold=fold.fold)
        prediction = np.asarray(contender.fit_predict(train, y_train, test, context), dtype=float)
        if prediction.shape != (len(test),):
            raise ValueError(f"{contender.name} returned {prediction.shape}, expected {(len(test),)}")
        if not np.isfinite(prediction).all():
            raise ValueError(f"{contender.name} returned non-finite predictions")
        if _target_digest(test[TARGET].to_numpy(dtype=float)) != before:
            raise AssertionError(f"{contender.name} mutated the held-out target")
        block = pd.DataFrame({
            "arm": contender.name, "design": fold.design, "split_seed": fold.seed,
            "fold": fold.fold, "row_id": test["row_id"].to_numpy(),
            "extractant": test["extractant"].to_numpy(),
            "chemotype": test["chemotype"].to_numpy(),
            "ecfp_cluster": test["ecfp_cluster"].to_numpy(),
            "series_id": test["series_id"].to_numpy(),
            "doi": test["doi"].to_numpy(),
            "review_flags": test["review_flags"].to_numpy(),
            TARGET: y_test, "prediction": prediction,
        })
        records.append(block)
        selections.append({"arm": contender.name, "design": fold.design, "split_seed": fold.seed,
                           "fold": fold.fold, "seconds": round(time.time() - started, 2),
                           **getattr(contender, "selected_", {})})
        if verbose:
            print(f"  {contender.name} seed={fold.seed} fold={fold.fold} "
                  f"n_test={len(test)} {time.time() - started:.1f}s", flush=True)
    predictions = pd.concat(records, ignore_index=True)
    if similarity is not None:
        key = ["design", "split_seed", "fold", "extractant"]
        predictions = predictions.merge(
            similarity[key + ["max_train_tanimoto", "nearest_train_extractant"]],
            on=key, how="left", validate="many_to_one")
        if predictions["max_train_tanimoto"].isna().any():
            raise AssertionError("a held-out extractant has no similarity entry")
        predictions["band"] = band_of(predictions["max_train_tanimoto"]).to_numpy()
    return predictions, pd.DataFrame(selections)

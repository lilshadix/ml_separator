"""Regression tests for the performance work in STEP 16.

Three hot paths were rewritten because they copied or walked a >2000-column
frame far more often than necessary.  None of them may change a number:

* ``_as_float_frame`` gained a numeric fast path;
* ``_reversal_input`` narrows what ``reverse_pair_features`` has to copy;
* ``_training_shuffle`` writes one dense block instead of one ``.loc`` per group.

Each test compares the optimised path against the literal original.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lanthanide_separation.ablation import _training_shuffle
from lanthanide_separation.evaluation import (
    AntisymmetricExtraTreesRegressor,
    _as_float_frame,
    _reversal_input,
)
from lanthanide_separation.feature_registry import build_feature_registry
from lanthanide_separation.pairs import (
    PAIR_TARGET_COLUMN,
    build_lanthanide_pair_dataset,
    reverse_pair_features,
)

from .test_extension_features import make_source_frame


class FloatFrameTests(unittest.TestCase):
    def test_numeric_fast_path_matches_the_coercing_path(self) -> None:
        frame = pd.DataFrame(
            {"a": [1, 2, 3], "b": [0.5, np.nan, -2.5], "c": ["x", "y", "z"]}
        )
        numeric = _as_float_frame(frame, ["a", "b"])
        legacy = (
            frame.loc[:, ["a", "b"]].apply(pd.to_numeric, errors="coerce").astype(float)
        )
        pd.testing.assert_frame_equal(numeric, legacy)

    def test_non_numeric_columns_still_coerce(self) -> None:
        frame = pd.DataFrame({"a": ["1", "oops", "3"]})
        result = _as_float_frame(frame, ["a"])
        self.assertTrue(np.isnan(result.iloc[1, 0]))
        self.assertEqual(result.iloc[0, 0], 1.0)


class ReversalNarrowingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pair_data = build_lanthanide_pair_dataset(
            make_source_frame(),
            pair_scope="all",
            include_pair_response_3d=True,
            include_electronic=True,
        )
        self.registry = build_feature_registry(self.pair_data)

    def test_narrowed_reversal_equals_full_frame_reversal(self) -> None:
        frame = self.pair_data.frame
        for arm in ("A0", "A1", "A2", "A5", "G2", "E3"):
            columns = list(self.registry.ablation_columns(arm))
            full = reverse_pair_features(frame).loc[:, columns]
            narrowed = reverse_pair_features(
                _reversal_input(frame, columns)
            ).loc[:, columns]
            pd.testing.assert_frame_equal(full, narrowed, check_dtype=False)

    def test_narrowing_keeps_both_members_of_a_swapped_pair(self) -> None:
        frame = self.pair_data.frame
        narrowed = _reversal_input(frame, ["pair__Z_A"])
        self.assertIn("pair__Z_B", narrowed.columns)

    def test_predictions_are_exactly_antisymmetric(self) -> None:
        frame = self.pair_data.frame
        columns = self.registry.ablation_columns("C3")
        model = AntisymmetricExtraTreesRegressor(
            columns, n_estimators=12, random_state=0, n_jobs=1
        )
        model.fit(
            frame,
            frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float),
            frame["extractant"].to_numpy(),
        )
        forward = model.predict(frame)
        reverse = model.predict(reverse_pair_features(frame))
        np.testing.assert_allclose(reverse, -forward, atol=1e-12)


class TrainingShuffleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pair_data = build_lanthanide_pair_dataset(
            make_source_frame(),
            pair_scope="all",
            include_pair_response_3d=True,
            include_electronic=True,
        )
        self.registry = build_feature_registry(self.pair_data)
        self.frame = self.pair_data.frame.reset_index(drop=True)

    def _legacy_shuffle(self, frame: pd.DataFrame, columns: tuple[str, ...], seed: int):
        """The pre-optimisation implementation, kept only as a test oracle."""

        unit_columns = ("geometry_key_A", "geometry_key_B")
        result = frame.copy()
        rng = np.random.default_rng(int(seed))
        for _, pair_frame in frame.groupby("pair_label", sort=True, dropna=False):
            grouped = pair_frame.groupby(list(unit_columns), sort=True, dropna=False)
            group_positions = [
                np.asarray(value, dtype=int) for value in grouped.indices.values()
            ]
            pair_positions = frame.index.get_indexer(pair_frame.index)
            group_positions = [pair_positions[local] for local in group_positions]
            representatives = [
                frame.iloc[positions].loc[:, list(columns)].iloc[0].to_numpy(copy=True)
                for positions in group_positions
            ]
            group_count = len(group_positions)
            if group_count <= 1:
                donor_indices = np.arange(group_count)
            else:
                randomized_order = rng.permutation(group_count)
                shift = int(rng.integers(1, group_count))
                donor_for_randomized_order = np.roll(randomized_order, shift)
                donor_indices = np.empty(group_count, dtype=int)
                donor_indices[randomized_order] = donor_for_randomized_order
            for recipient_index, positions in enumerate(group_positions):
                result.loc[result.index[positions], list(columns)] = representatives[
                    int(donor_indices[recipient_index])
                ]
        return result

    def test_block_shuffle_matches_the_original_implementation(self) -> None:
        for family in ("3D_LOCAL", "3D_PAIR_RESPONSE", "ELEC_PAIR"):
            columns = self.registry.columns_for_family(family)
            if not columns:
                continue
            optimised, audit = _training_shuffle(self.frame, columns, seed=1009)
            legacy = self._legacy_shuffle(self.frame, columns, seed=1009)
            np.testing.assert_allclose(
                optimised.loc[:, list(columns)].to_numpy(dtype=float),
                legacy.loc[:, list(columns)].to_numpy(dtype=float),
                equal_nan=True,
                err_msg=family,
            )
            self.assertEqual(audit["training_rows"], len(self.frame))
            self.assertTrue(audit["donors_from_training_only"])

    def test_shuffle_touches_only_the_named_block(self) -> None:
        columns = self.registry.columns_for_family("3D_PAIR_RESPONSE")
        shuffled, _ = _training_shuffle(self.frame, columns, seed=2017)
        untouched = [c for c in self.registry.columns if c not in set(columns)]
        np.testing.assert_allclose(
            shuffled.loc[:, untouched].to_numpy(dtype=float),
            self.frame.loc[:, untouched].to_numpy(dtype=float),
            equal_nan=True,
        )

    def test_shuffle_preserves_the_marginal_distribution_of_the_block(self) -> None:
        columns = self.registry.columns_for_family("ELEC_PAIR")
        shuffled, _ = _training_shuffle(self.frame, columns, seed=3019)
        for column in columns:
            before = np.sort(self.frame[column].to_numpy(dtype=float))
            after = np.sort(shuffled[column].to_numpy(dtype=float))
            self.assertEqual(len(before), len(after))
            # Whole complex-pair vectors move between recipients, so the
            # multiset of values is preserved up to repeated-row multiplicity.
            self.assertAlmostEqual(
                float(np.nanmean(before)), float(np.nanmean(after)), places=6
            )


if __name__ == "__main__":
    unittest.main()

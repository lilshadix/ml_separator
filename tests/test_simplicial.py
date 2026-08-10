from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

try:
    import torch

    from lanthanide_separation.deep_evaluation import (
        CONTEXT_MODES,
        SimplicialData,
        permuted_geometry_view,
        symmetric_context_columns,
    )
    from lanthanide_separation.simplicial import (
        SIMPLEX_ORDERS,
        AntisymmetricSimplicialPairRegressor,
        PackedSimplicialBatch,
    )
except ModuleNotFoundError:  # Optional deep dependency.
    torch = None


@unittest.skipIf(torch is None, "torch deep extra is not installed")
class SimplicialModelTests(unittest.TestCase):
    @staticmethod
    def pair_batch(
        first_metal: int,
        second_metal: int,
        *,
        first_distances: tuple[float, float] = (0.8, 0.9),
        second_distances: tuple[float, float] = (0.75, 0.85),
        first_filtration: tuple[float, float, float] = (2.4, 2.7, 3.5),
        second_filtration: tuple[float, float, float] = (2.3, 2.6, 3.4),
    ) -> PackedSimplicialBatch:
        atomic_numbers = torch.tensor(
            [first_metal, 8, 8, second_metal, 8, 8], dtype=torch.long
        )
        continuous = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, first_distances[0]],
                [0.0, 0.0, 0.0, 1.0, first_distances[1]],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, second_distances[0]],
                [0.0, 0.0, 0.0, 1.0, second_distances[1]],
            ],
            dtype=torch.float32,
        )
        return PackedSimplicialBatch(
            atomic_numbers=atomic_numbers,
            continuous_node_features=continuous,
            edge_index=torch.tensor(
                [[0, 0, 1, 3, 3, 4], [1, 2, 2, 4, 5, 5]], dtype=torch.long
            ),
            edge_filtration=torch.tensor(
                [*first_filtration, *second_filtration], dtype=torch.float32
            ),
            triangle_index=torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.long),
            triangle_filtration=torch.tensor(
                [first_filtration[2], second_filtration[2]], dtype=torch.float32
            ),
            triangle_shape_features=torch.tensor(
                [
                    [*sorted(first_filtration), 0.2, 0.8, 0.1, 1.0],
                    [*sorted(second_filtration), 0.2, 0.8, 0.1, 1.0],
                ],
                dtype=torch.float32,
            ),
            node_batch=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
            metal_indices=torch.tensor([0, 3], dtype=torch.long),
            donor_mask=torch.tensor([False, True, True, False, True, True]),
            n_graphs=2,
        )

    def test_pair_prediction_is_exactly_antisymmetric(self) -> None:
        torch.manual_seed(3)
        model = AntisymmetricSimplicialPairRegressor(
            context_dim=2, hidden_dim=16, layers=2, dropout=0.0, rbf_count=6
        )
        model.eval()
        context = torch.tensor([[0.5, -0.2]], dtype=torch.float32)
        forward = model(self.pair_batch(57, 58), context)
        reverse = model(
            self.pair_batch(
                58,
                57,
                first_distances=(0.75, 0.85),
                second_distances=(0.8, 0.9),
                first_filtration=(2.3, 2.6, 3.4),
                second_filtration=(2.4, 2.7, 3.5),
            ),
            context,
        )
        torch.testing.assert_close(forward, -reverse, atol=1e-7, rtol=0.0)

    def test_every_encoder_rung_stays_exactly_antisymmetric(self) -> None:
        context = torch.tensor([[0.5, -0.2]], dtype=torch.float32)
        for order in SIMPLEX_ORDERS:
            with self.subTest(simplex_order=order):
                torch.manual_seed(3)
                model = AntisymmetricSimplicialPairRegressor(
                    context_dim=2,
                    hidden_dim=16,
                    layers=2,
                    dropout=0.0,
                    rbf_count=6,
                    simplex_order=order,
                )
                model.eval()
                forward = model(self.pair_batch(57, 58), context)
                reverse = model(
                    self.pair_batch(
                        58,
                        57,
                        first_distances=(0.75, 0.85),
                        second_distances=(0.8, 0.9),
                        first_filtration=(2.3, 2.6, 3.4),
                        second_filtration=(2.4, 2.7, 3.5),
                    ),
                    context,
                )
                torch.testing.assert_close(forward, -reverse, atol=1e-7, rtol=0.0)

    def test_encoder_rungs_have_identical_capacity(self) -> None:
        # A rung must differ from the next only in the information it reads, not
        # in how many parameters it has, or a difference between them cannot be
        # attributed to simplicial order.
        counts = set()
        for order in SIMPLEX_ORDERS:
            torch.manual_seed(3)
            model = AntisymmetricSimplicialPairRegressor(
                context_dim=2, hidden_dim=16, layers=2, dropout=0.0,
                rbf_count=6, simplex_order=order,
            )
            counts.add(sum(parameter.numel() for parameter in model.parameters()))
        self.assertEqual(len(counts), 1)

    def test_lower_rungs_ignore_the_structure_they_exclude(self) -> None:
        context = torch.tensor([[0.5, -0.2]], dtype=torch.float32)
        # Same graphs, different 2-simplex geometry.
        first = self.pair_batch(57, 58)
        second = self.pair_batch(57, 58)
        second.triangle_shape_features[:, 3:] = 0.0
        for order, should_differ in (
            ("nodes_distances", False),
            ("nodes_edges", False),
            ("nodes_edges_triangles", True),
        ):
            with self.subTest(simplex_order=order):
                torch.manual_seed(11)
                model = AntisymmetricSimplicialPairRegressor(
                    context_dim=2, hidden_dim=16, layers=2, dropout=0.0,
                    rbf_count=6, simplex_order=order,
                )
                model.eval()
                changed = not torch.allclose(
                    model(first, context), model(second, context), atol=1e-9
                )
                self.assertEqual(changed, should_differ)

    def test_distance_only_rung_ignores_atom_identity(self) -> None:
        torch.manual_seed(5)
        model = AntisymmetricSimplicialPairRegressor(
            context_dim=2, hidden_dim=16, layers=2, dropout=0.0,
            rbf_count=6, simplex_order="distances",
        )
        model.eval()
        context = torch.tensor([[0.5, -0.2]], dtype=torch.float32)
        original = self.pair_batch(57, 58)
        relabelled = self.pair_batch(57, 58)
        # Swap the donor element without touching any distance.
        relabelled.atomic_numbers[relabelled.atomic_numbers.eq(8)] = 7
        torch.testing.assert_close(
            model(original, context), model(relabelled, context), atol=1e-7, rtol=0.0
        )


@unittest.skipIf(torch is None, "torch deep extra is not installed")
class GeometryNullControlTests(unittest.TestCase):
    """The learned-geometry permutation control (STEP 10).

    A null arm is only usable as evidence if the permutation is provably
    confined to the training rows; otherwise a weak control would be
    indistinguishable from a leaking one.
    """

    @staticmethod
    def data(n: int = 12) -> SimplicialData:
        labels = ["La/Nd" if index % 2 else "Nd/Yb" for index in range(n)]
        frame = pd.DataFrame({"pair_label": labels})
        return SimplicialData(
            frame=frame,
            store=None,
            context=np.zeros((n, 1), dtype=np.float32),
            target=np.arange(n, dtype=np.float32),
            groups=np.array([f"lig{index // 3}" for index in range(n)]),
            build_ids_a=np.array([f"a{index}" for index in range(n)]),
            build_ids_b=np.array([f"b{index}" for index in range(n)]),
            simplex_cost=np.arange(1, n + 1, dtype=np.int64),
        )

    def test_held_out_rows_are_never_touched(self) -> None:
        data = self.data()
        train = np.arange(0, 8, dtype=np.int64)
        held_out = np.arange(8, 12, dtype=np.int64)
        permuted, audit = permuted_geometry_view(data, train, seed=17)
        np.testing.assert_array_equal(
            permuted.build_ids_a[held_out], data.build_ids_a[held_out]
        )
        np.testing.assert_array_equal(
            permuted.build_ids_b[held_out], data.build_ids_b[held_out]
        )
        self.assertEqual(audit["test_rows_touched"], 0)
        self.assertEqual(audit["train_rows"], len(train))

    def test_permutation_stays_inside_one_pair_label(self) -> None:
        data = self.data()
        train = np.arange(0, 12, dtype=np.int64)
        permuted, _ = permuted_geometry_view(data, train, seed=5)
        labels = data.frame["pair_label"].to_numpy()
        source_index = {
            f"a{index}": index for index in range(len(data.build_ids_a))
        }
        for row, build_id in enumerate(permuted.build_ids_a):
            self.assertEqual(labels[row], labels[source_index[build_id]])

    def test_the_geometry_multiset_is_preserved(self) -> None:
        # The control must change only the structure-to-target correspondence,
        # not the marginal distribution of structures.
        data = self.data()
        train = np.arange(0, 12, dtype=np.int64)
        permuted, _ = permuted_geometry_view(data, train, seed=3)
        self.assertEqual(
            sorted(permuted.build_ids_a.tolist()), sorted(data.build_ids_a.tolist())
        )
        self.assertEqual(
            sorted(permuted.simplex_cost.tolist()), sorted(data.simplex_cost.tolist())
        )

    def test_the_two_sides_of_a_pair_move_together(self) -> None:
        # Permuting A and B independently would break the complex-pair identity
        # and make the control a different model, not a null of the same one.
        data = self.data()
        permuted, _ = permuted_geometry_view(
            data, np.arange(0, 12, dtype=np.int64), seed=9
        )
        for build_a, build_b in zip(
            permuted.build_ids_a, permuted.build_ids_b, strict=True
        ):
            self.assertEqual(build_a[1:], build_b[1:])

    def test_it_is_deterministic_in_the_seed(self) -> None:
        data = self.data()
        train = np.arange(0, 10, dtype=np.int64)
        first, _ = permuted_geometry_view(data, train, seed=21)
        second, _ = permuted_geometry_view(data, train, seed=21)
        np.testing.assert_array_equal(first.build_ids_a, second.build_ids_a)

    def test_an_empty_training_set_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            permuted_geometry_view(
                self.data(), np.array([], dtype=np.int64), seed=1
            )


@unittest.skipIf(torch is None, "torch deep extra is not installed")
class ContextModeTests(unittest.TestCase):
    """The learned-geometry-only arm must really be ligand-2D-free."""

    class _PairData:
        baseline_columns = (
            "pair__Z_mean",
            "pair__ionic_radius_mean",
            "pair__delta_Z",
            "base__cond__acid_molarity",
            "base__MolWt",
            "base__TPSA",
            "base__ecfp_17",
        )

    def test_full_mode_keeps_the_ligand_descriptors(self) -> None:
        columns = symmetric_context_columns(self._PairData(), context_mode="full")
        self.assertIn("base__MolWt", columns)
        self.assertIn("base__cond__acid_molarity", columns)

    def test_conditions_only_drops_every_ligand_descriptor(self) -> None:
        columns = symmetric_context_columns(
            self._PairData(), context_mode="conditions_only"
        )
        self.assertNotIn("base__MolWt", columns)
        self.assertNotIn("base__TPSA", columns)
        self.assertIn("base__cond__acid_molarity", columns)
        self.assertIn("pair__Z_mean", columns)

    def test_no_mode_ever_admits_a_fingerprint_bit_or_an_odd_column(self) -> None:
        for mode in CONTEXT_MODES:
            columns = symmetric_context_columns(self._PairData(), context_mode=mode)
            self.assertFalse(
                [column for column in columns if column.startswith("base__ecfp_")],
                msg=mode,
            )
            # The context enters the head unchanged under the A/B swap, so an
            # odd column there would break the structural antisymmetry.
            self.assertNotIn("pair__delta_Z", columns, msg=mode)

    def test_unknown_mode_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            symmetric_context_columns(self._PairData(), context_mode="geometry_only")


if __name__ == "__main__":
    unittest.main()

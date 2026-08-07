from __future__ import annotations

import unittest

try:
    import torch

    from lanthanide_separation.simplicial import (
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


if __name__ == "__main__":
    unittest.main()

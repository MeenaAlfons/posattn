from ..positional_mask import GaussianPositionalMask
import torch
import unittest


class MeshgridCacheMock:
    def __init__(self, meshgrid):
        self.meshgrid = meshgrid

    def make(self, structure_size, *args, **kwargs):
        assert torch.all(
            torch.tensor(self.meshgrid.shape[:-1]) == 2 * structure_size - 1
        ), "meshgrid.shape: {} structure_size: {}".format(
            self.meshgrid.shape, structure_size
        )
        return self.meshgrid


class TestGaussMask1D(unittest.TestCase):
    def test_1D(self):
        meshgrid = torch.tensor([[-1], [0], [1]])
        dtype = meshgrid.dtype
        device = meshgrid.device
        meshgrid_cache = MeshgridCacheMock(meshgrid)

        tests = [
            {
                'sigma': 0.3,
            }, {
                'sigma': 0.5,
            }, {
                'sigma': 1,
            }, {
                'sigma': 2,
            }
        ]
        for test in tests:
            sigma = test['sigma']
            mask_module = GaussianPositionalMask(1, meshgrid_cache, sigma, True)
            structure_size = torch.tensor([2], dtype=torch.int64)
            mask_size = torch.tensor([3], dtype=torch.int64)
            mask = mask_module(structure_size, mask_size, dtype, device)
            self.assertEqual(mask.shape, (3, 1))
            self.assertEqual(mask.max(), 1)
            self.assertEqual(
                mask[0, 0], torch.exp(-0.5 * meshgrid[0]**2 / sigma**2)
            )
            self.assertEqual(
                mask[1, 0], torch.exp(-0.5 * meshgrid[1]**2 / sigma**2)
            )
            self.assertEqual(
                mask[2, 0], torch.exp(-0.5 * meshgrid[2]**2 / sigma**2)
            )

            self.assertEqual(mask[1, 0], 1)

    def test_2D(self):
        meshgrid = torch.tensor(
            [
                [[-1.0, -1], [-1.0, 0], [-1.0, 1]],
                [[-0.5, -1], [-0.5, 0], [-0.5, 1]],
                [[0.0, -1], [0.0, 0], [0.0, 1]],
                [[0.5, -1], [0.5, 0], [0.5, 1]],
                [[1.0, -1], [1.0, 0], [1.0, 1]],
            ]
        )
        dtype = meshgrid.dtype
        device = meshgrid.device
        meshgrid_cache = MeshgridCacheMock(meshgrid)
        tests = [
            {
                'sigma': 0.3,
            }, {
                'sigma': 0.5,
            }, {
                'sigma': 1,
            }, {
                'sigma': 2,
            }
        ]
        for test in tests:
            sigma = test['sigma']
            mask_module = GaussianPositionalMask(2, meshgrid_cache, sigma, True)
            structure_size = torch.tensor([3, 2], dtype=torch.int64)
            mask_size = torch.tensor([5, 3], dtype=torch.int64)
            mask = mask_module(structure_size, mask_size, dtype, device)
            self.assertEqual(mask.shape, (5, 3, 1))
            self.assertEqual(mask.max(), 1)

            for row in range(5):
                for col in range(3):
                    self.assertEqual(
                        mask[row, col],
                        torch.exp(
                            (
                                meshgrid[row, col, 0]**2 +
                                meshgrid[row, col, 1]**2
                            ) * -0.5 / sigma**2
                        )
                    )

    def test_2D_different_sigma(self):
        meshgrid = torch.tensor(
            [
                [[-1.0, -1], [-1.0, 0], [-1.0, 1]],
                [[-0.5, -1], [-0.5, 0], [-0.5, 1]],
                [[0.0, -1], [0.0, 0], [0.0, 1]],
                [[0.5, -1], [0.5, 0], [0.5, 1]],
                [[1.0, -1], [1.0, 0], [1.0, 1]],
            ]
        )
        dtype = meshgrid.dtype
        device = meshgrid.device
        meshgrid_cache = MeshgridCacheMock(meshgrid)
        tests = [
            {
                'sigma1': 0.3,
                'sigma2': 0.3,
            }, {
                'sigma1': 0.3,
                'sigma2': 0.3,
            }, {
                'sigma1': 0.3,
                'sigma2': 0.3,
            }, {
                'sigma1': 0.3,
                'sigma2': 0.3,
            }
        ]
        for test in tests:
            sigma1 = test['sigma1']
            sigma2 = test['sigma2']
            mask_module = GaussianPositionalMask(2, meshgrid_cache, 0, True)
            mask_module.sigma.data = torch.tensor([sigma1, sigma2])
            structure_size = torch.tensor([3, 2], dtype=torch.int64)
            mask_size = torch.tensor([5, 3], dtype=torch.int64)
            mask = mask_module(structure_size, mask_size, dtype, device)
            self.assertEqual(mask.shape, (5, 3, 1))
            self.assertEqual(mask.max(), 1)

            for row in range(5):
                for col in range(3):
                    self.assertEqual(
                        mask[row, col],
                        torch.exp(
                            (
                                meshgrid[row, col, 0]**2 / sigma1**2 +
                                meshgrid[row, col, 1]**2 / sigma2**2
                            ) * -0.5
                        )
                    )


# TODO test mask_size different than strcuture_size

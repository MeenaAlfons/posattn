from ..unfold_v2 import unfold_k, unfold_p
import torch
import unittest

# Manual testing covered:
# unfold_k:
# - 1D
# - 2D
# - mask_size <=> structure_size
# - unsquare patch_size
# - even patch_size
#
# unfold_p:
# - 1D
# - 2D
# - mask_size <=> structure_size
# - unsquare patch_size
# - even patch_size

padding_constant = 13.0
batch_size = 3
num_heads = 2
head_dim = 4
tests = [
    {
        'structure_size': torch.tensor([5], dtype=torch.int64),
        'mask_size': torch.tensor([1], dtype=torch.int64),
        'expected_patch_size': torch.tensor([1], dtype=torch.int64),
    },
    {
        'structure_size': (5, ),
        'mask_size': (3, ),
        'expected_patch_size': (3, ),
    },
    {
        'structure_size': (5, ),
        'mask_size': (5, ),
        'expected_patch_size': (5, ),
    },
    {
        'structure_size': (5, ),
        'mask_size': (7, ),
        'expected_patch_size': (5, ),
    },
    {
        'structure_size': (5, ),
        'mask_size': (9, ),
        'expected_patch_size': (5, ),
    },
    ##### 2D
    {
        'structure_size': (3, 3),
        'mask_size': (1, 1),
        'expected_patch_size': (1, 1),
    },
    {
        'structure_size': (3, 3),
        'mask_size': (3, 3),
        'expected_patch_size': (3, 3),
    },
    {
        'structure_size': (3, 3),
        'mask_size': (5, 5),
        'expected_patch_size': (3, 3),
    },
    {
        'structure_size': (5, 5),
        'mask_size': (1, 1),
        'expected_patch_size': (1, 1),
    },
    {
        'structure_size': (5, 5),
        'mask_size': (3, 3),
        'expected_patch_size': (3, 3),
    },
    {
        'structure_size': (5, 5),
        'mask_size': (5, 5),
        'expected_patch_size': (5, 5),
    },
    {
        'structure_size': (5, 5),
        'mask_size': (7, 7),
        'expected_patch_size': (5, 5),
    },
    {
        'structure_size': (5, 5),
        'mask_size': (9, 9),
        'expected_patch_size': (5, 5),
    },
    {
        'structure_size': (5, 5),
        'mask_size': (5, 7),
        'expected_patch_size': (5, 5),
    },
    {
        'structure_size': (5, 5),
        'mask_size': (3, 5),
        'expected_patch_size': (3, 5),
    },
    {
        'structure_size': (5, 5),
        'mask_size': (5, 3),
        'expected_patch_size': (5, 3),
    },
    {
        'structure_size': (5, 3),
        'mask_size': (5, 3),
        'expected_patch_size': (5, 3),
    },
    {
        'structure_size': (5, 3),
        'mask_size': (3, 5),
        'expected_patch_size': (3, 3),
    },

    # Even structure_size
    {
        'structure_size': (4, ),
        'mask_size': (1, ),
        'expected_patch_size': (1, ),
    },
    {
        'structure_size': (4, ),
        'mask_size': (3, ),
        'expected_patch_size': (3, ),
    },
    {
        'structure_size': (4, ),
        'mask_size': (5, ),
        'expected_patch_size': (4, ),
    },
    {
        'structure_size': (4, ),
        'mask_size': (7, ),
        'expected_patch_size': (4, ),
    },
    {
        'structure_size': (6, ),
        'mask_size': (1, ),
        'expected_patch_size': (1, ),
    },
    {
        'structure_size': (6, ),
        'mask_size': (3, ),
        'expected_patch_size': (3, ),
    },
    {
        'structure_size': (6, ),
        'mask_size': (5, ),
        'expected_patch_size': (5, ),
    },
    {
        'structure_size': (6, ),
        'mask_size': (7, ),
        'expected_patch_size': (6, ),
    },
    {
        'structure_size': (6, ),
        'mask_size': (9, ),
        'expected_patch_size': (6, ),
    },
]


class TestUnfold(unittest.TestCase):
    def test_unfold_k(self):
        for test in tests:
            # print("test:", test)
            structure_size = torch.tensor(
                test['structure_size'], dtype=torch.int64
            )
            mask_size = torch.tensor(test['mask_size'], dtype=torch.int64)
            expected_patch_size = torch.tensor(
                test['expected_patch_size'], dtype=torch.int64
            )
            with self.subTest(test):
                k = torch.rand(
                    batch_size,
                    num_heads,
                    head_dim,
                    *tuple(structure_size),
                )
                k_patched = unfold_k(
                    k,
                    structure_size=structure_size,
                    mask_size=mask_size,
                    padding_constant=padding_constant
                )
                assert k_patched.shape == (
                    batch_size,
                    num_heads,
                    head_dim,
                    *tuple(structure_size),
                    *tuple(expected_patch_size),
                )

    def test_unfold_p(self):
        for test in tests:
            # print("test:", test)
            structure_size = torch.tensor(
                test['structure_size'], dtype=torch.int64
            )
            mask_size = torch.tensor(test['mask_size'], dtype=torch.int64)
            expected_patch_size = torch.tensor(
                test['expected_patch_size'], dtype=torch.int64
            )
            with self.subTest(test):
                p = torch.rand(
                    num_heads,
                    head_dim,
                    *tuple(mask_size),
                )
                p_patched = unfold_p(
                    p,
                    structure_size=structure_size,
                    mask_size=mask_size,
                    padding_constant=padding_constant
                )
                assert p_patched.shape == (
                    num_heads,
                    head_dim,
                    *tuple(structure_size),
                    *tuple(expected_patch_size),
                )

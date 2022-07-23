import torch
import torch.nn.functional as F
"""
Here we reestablish unfold_k, unfold_p, and make_mask side by side to make sure their conventions match.

Definitions:
    SeqLen -int- the number of keys/queries in the input
    structure_size -tuple()- The sizes of the positional dimensions of the input.
        prod(structure_size) = SeqLen
    mask_size -tuple()- The size of the relative mask which is centered around
        the position of the query or around the center of the relative
        positional encoding. mask_size <= 2*structure_size-1
    C -float- A constant to represent missing or ignored values in the output.

    Applying a relative mask has the effect of limiting the keys for each query along with their respective relative positional encoding.

    Inputs:
    k: The keys input of size [Batch, Head, HeadDims, <structure_size>]
    p: The relative positional encoding of size [Head, HeadDims, <2*structure_size-1>]

    Outputs:
    p_patched: The relative positional encoding of size [Head, HeadDims, <structure_size>]
    mask: Boolean mask to indicate the relevant values from the patched output. [<structure_size>, <structure_size>]

    The patched outputs of p need to have the relative positional encoding respective to the keys in k relative to each query.

    Example: SeqLen = 5, structure_size = (5), mask_size = (5),
    k = [1,2,3,4,5], p = [-4,-3,-2,-1,0,1,2,3,4]
    p_patched = [
        [ 0, 1, 2, 3, 4],
        [-1, 0, 1, 2, 3],
        [-2,-1, 0, 1, 2],
        [-3,-2,-1, 0, 1],
        [-4,-3,-2,-1, 0],
    ]
    mask = [
        [1,1,1,0,0],
        [1,1,1,1,0],
        [1,1,1,1,1],
        [0,1,1,1,1],
        [0,0,1,1,1],
    ]

    Example: SeqLen = 5, structure_size = (5), mask_size = (3),
    k = [1,2,3,4,5], p = [-4,-3,-2,-1,0,1,2,3,4]
    p_patched = [
        [ 0, 1, 2, 3, 4],
        [-1, 0, 1, 2, 3],
        [-2,-1, 0, 1, 2],
        [-3,-2,-1, 0, 1],
        [-4,-3,-2,-1, 0],
    ]
    mask = [
        [1,1,0,0,0],
        [1,1,1,0,0],
        [0,1,1,1,0],
        [0,0,1,1,1],
        [0,0,0,1,1],
    ]

    Example: SeqLen = 5, structure_size = (5), mask_size = (7),
    k = [1,2,3,4,5], p = [-4,-3,-2,-1,0,1,2,3,4]
    p_patched = [
        [ 0, 1, 2, 3, 4],
        [-1, 0, 1, 2, 3],
        [-2,-1, 0, 1, 2],
        [-3,-2,-1, 0, 1],
        [-4,-3,-2,-1, 0],
    ]
    mask = [
        [1,1,1,1,0],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [0,1,1,1,1],
    ]

    Example: SeqLen = 5, structure_size = (5), mask_size = (9),
    k = [1,2,3,4,5], p = [-4,-3,-2,-1,0,1,2,3,4]
    p_patched = [
        [ 0, 1, 2, 3, 4],
        [-1, 0, 1, 2, 3],
        [-2,-1, 0, 1, 2],
        [-3,-2,-1, 0, 1],
        [-4,-3,-2,-1, 0],
    ]
    mask = [
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
    ]

    Example: SeqLen = 7, structure_size = (7), mask_size = (5),
    k = [1,2,3,4,5,6,7], p = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
    p_patched = [
        [ 0, 1, 2, 3, 4, 5, 6],
        [-1, 0, 1, 2, 3, 4, 5],
        [-2,-1, 0, 1, 2, 3, 4],
        [-3,-2,-1, 0, 1, 2, 3],
        [-4,-3,-2,-1, 0, 1, 2],
        [-5,-4,-3,-2,-1, 0, 1],
        [-6,-5,-4,-3,-2,-1, 0],
    ]
    mask = [
        [1,1,1,0,0,0,0],
        [1,1,1,1,0,0,0],
        [1,1,1,1,1,0,0],
        [0,1,1,1,1,1,0],
        [0,0,1,1,1,1,1],
        [0,0,0,1,1,1,1],
        [0,0,0,0,1,1,1],
    ]

Implementation:
    p_patched:
        1. Unfold patches of size structure_size from p
        2. Flip positional dimensions
    mask:
        1. Create tensor of ones of size [<structure_size>, <structure_size>]
        2. Set the upper right and lower left corners to 0.
           The side of the triangle = structure_size - mask_size//2 - 1

"""


def unfold_p(p, structure_size):
    num_positional_dims = len(structure_size)
    assert torch.all(
        torch.tensor(
            p.shape[-num_positional_dims:], device=structure_size.device
        ) == 2 * structure_size - 1
    ), "p.shape={}, structure_size={}".format(p.shape, structure_size)

    # Unfold patches of size structure_size
    for dim_size in structure_size:
        p = p.unfold(-num_positional_dims, dim_size, 1)

    assert torch.all(
        torch.tensor(
            p.shape[-num_positional_dims:], device=structure_size.device
        ) == structure_size
    )
    assert torch.all(
        torch.tensor(
            p.shape[-2 * num_positional_dims:-num_positional_dims],
            device=structure_size.device
        ) == structure_size
    )

    # Flip the positional dimension of the queries.
    flip_dims = tuple(range(-2 * num_positional_dims, -num_positional_dims))
    p = p.flip(flip_dims)

    return p


def make_mask(structure_size, mask_size, device):
    mask = torch.ones(
        tuple(structure_size) + tuple(structure_size),
        dtype=torch.bool,
        device=device
    )

    # Fill upper right and lower left corners with False.
    num_positional_dims = len(structure_size)
    diagonal_number = mask_size // 2 + 1
    for i, s in enumerate(structure_size):
        upper_row_idx, upper_col_idx = torch.triu_indices(
            s, s, diagonal_number[i]
        )
        upper_right_index = [slice(None)] * len(mask.shape)
        upper_right_index[-2 * num_positional_dims + i] = upper_row_idx
        upper_right_index[-num_positional_dims + i] = upper_col_idx
        mask[upper_right_index] = False
        lower_row_idx, lower_col_idx = torch.tril_indices(
            s, s, -diagonal_number[i]
        )
        lower_left_index = [slice(None)] * len(mask.shape)
        lower_left_index[-2 * num_positional_dims + i] = lower_row_idx
        lower_left_index[-num_positional_dims + i] = lower_col_idx
        mask[lower_left_index] = False

    return mask
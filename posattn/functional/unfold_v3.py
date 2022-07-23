import torch
import torch.nn.functional as F
"""
Here we reestablish unfold_k, make_mask side by side to make sure their conventions match.

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
    p: The relative positional encoding of size [Head, HeadDims, <mask_size>]

    Outputs:
    k_patched: The patched keys of size [Batch, Head, HeadDims, <structure_size>, <mask_size>]
    p: The relative positional encoding of size [Head, HeadDims, <mask_size>]
    mask: Boolean mask to indicate the relevant values from the patched output. [<structure_size>, <mask_size>]

    The patched outputs of k need to have respective values for relative positional encoding in p relative to each query.

    Example: SeqLen = 5, structure_size = (5), mask_size = (5),
    k = [1,2,3,4,5], p = [-2,-1,0,1,2]
    k_patched = [
        [C,C,1,2,3],
        [C,1,2,3,4],
        [1,2,3,4,5],
        [2,3,4,5,C],
        [3,4,5,C,C],
    ]
    mask = [
        [0,0,1,1,1],
        [0,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,0],
        [1,1,1,0,0],
    ]

    Example: SeqLen = 5, structure_size = (5), mask_size = (3)
    k = [1,2,3,4,5], p = [-1,0,1]
    k_patched = [
        [C,1,2],
        [1,2,3],
        [2,3,4],
        [3,4,5],
        [4,5,C],
    ]
    mask = [
        [0,1,1],
        [1,1,1],
        [1,1,1],
        [1,1,1],
        [1,1,0],
    ]

    Example: SeqLen = 5, structure_size = (5), mask_size = (7)
    k = [1,2,3,4,5], p = [-3,-2,-1,0,1,2,3]
    k_patched = [
        [C,C,C,1,2,3,4],
        [C,C,1,2,3,4,5],
        [C,1,2,3,4,5,C],
        [1,2,3,4,5,C,C],
        [2,3,4,5,C,C,C],
    ]
    mask = [
        [0,0,0,1,1,1,1],
        [0,0,1,1,1,1,1],
        [0,1,1,1,1,1,0],
        [1,1,1,1,1,0,0],
        [1,1,1,1,0,0,0],
    ]

    Example: SeqLen = 5, structure_size = (5), mask_size = (9)
    k = [1,2,3,4,5], p = [-4,-3,-2,-1,0,1,2,3,4]
    k_patched = [
        [C,C,C,C,1,2,3,4,5],
        [C,C,C,1,2,3,4,5,C],
        [C,C,1,2,3,4,5,C,C],
        [C,1,2,3,4,5,C,C,C],
        [1,2,3,4,5,C,C,C,C],
    ]
    mask = [
        [0,0,0,0,1,1,1,1,1],
        [0,0,0,1,1,1,1,1,0],
        [0,0,1,1,1,1,1,0,0],
        [0,1,1,1,1,1,0,0,0],
        [1,1,1,1,1,0,0,0,0],
    ]

    Example: SeqLen = 7, structure_size = (7), mask_size = (5)
    k = [1,2,3,4,5,6,7], p = [-2,-1,0,1,2]
    k_patched = [
        [C,C,1,2,3],
        [C,1,2,3,4],
        [1,2,3,4,5],
        [2,3,4,5,6],
        [3,4,5,6,7],
        [4,5,6,7,C],
        [5,6,7,C,C],
    ]
    mask = [
        [0,0,1,1,1],
        [0,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,0],
        [1,1,1,0,0],
    ]

Implementation:
    k_patched:
        1. Pad k (mask_size) with C.
           pad_size = mask_size/2 on the left and on the right of
           positional dimensions. The total size will be structure_size + mask_size - 1
        2. Unfold patches of size mask_size. The positional dimensions will be structure_size.
    mask:
        1. Create tensor of ones of size [<structure_size>, <mask_size>]
        2. Set the upper left and lower right corners to 0.
           The side of the triangle = mask_size/2

"""


def get_pad_size(mask_size):
    return mask_size // 2


def unfold_k(k, structure_size, mask_size, padding_constant=0.0):
    """
    Inputs:
    k is the keys.
    k.shape: [Batch, Head, HeadDims, D1, D2, ...]
    structure_size: [D1, D2, ...]
    mask_size: [m1, m2, ...]
    padding_constant: A constant to represent missing or ignored values in the output.

    Outputs:
    k_patched is the patched keys for every query in strcture_size.
    k_patched.shape: [Batch, Head, HeadDims, D1, D2, ..., m1, m2, ...]
    """
    assert len(structure_size) == len(mask_size)
    assert torch.all(mask_size % 2 == 1)
    num_positional_dims = len(structure_size)
    assert torch.all(
        torch.tensor(
            k.shape[-num_positional_dims:], device=structure_size.device
        ) == structure_size
    ), "k.shape={}, structure_size={}".format(k.shape, structure_size)

    # Pad k (mask_size) with C.
    # pad_size = mask_size//2 on the left and on the right of positional dimensions.
    # The total size will be structure_size + mask_size - 1
    pad_size = get_pad_size(mask_size)
    pad = tuple()
    for ps in pad_size:
        pad = (ps, ps) + pad
    k = F.pad(k, pad, value=padding_constant)
    assert torch.all(
        torch.tensor(
            k.shape[-num_positional_dims:], device=structure_size.device
        ) == (structure_size + mask_size - 1)
    )

    # Unfold patches of size mask_size
    for dim_size in mask_size:
        k = k.unfold(-num_positional_dims, dim_size, 1)

    assert torch.all(
        torch.tensor(
            k.shape[-num_positional_dims:], device=structure_size.device
        ) == mask_size
    ), "k.shape={}, mask_size={}".format(k.shape, mask_size)
    assert torch.all(
        torch.tensor(
            k.shape[-2 * num_positional_dims:-num_positional_dims],
            device=structure_size.device
        ) == structure_size
    )

    return k


def make_mask(structure_size, mask_size, device):
    mask = torch.ones(
        tuple(structure_size) + tuple(mask_size),
        dtype=torch.bool,
        device=device
    )
    # Set the upper left and lower right corners to 0.
    # The side of the triangle = mask_size//2
    num_positional_dims = len(structure_size)
    pad_size = get_pad_size(mask_size)
    diagonal_number = pad_size + 1
    for i, s in enumerate(structure_size):
        upper_row_idx, upper_col_idx = torch.triu_indices(
            s, mask_size[i], diagonal_number[i], device=device
        )
        upper_col_idx = mask_size[i] - upper_col_idx - 1
        upper_right_index = [slice(None)] * len(mask.shape)
        upper_right_index[-2 * num_positional_dims + i] = upper_row_idx
        upper_right_index[-num_positional_dims + i] = upper_col_idx
        mask[upper_right_index] = False
        lower_row_idx, lower_col_idx = torch.tril_indices(
            s,
            mask_size[i],
            -diagonal_number[i] - (s - mask_size[i]),
            device=device
        )
        lower_col_idx = mask_size[i] - lower_col_idx - 1
        lower_left_index = [slice(None)] * len(mask.shape)
        lower_left_index[-2 * num_positional_dims + i] = lower_row_idx
        lower_left_index[-num_positional_dims + i] = lower_col_idx
        mask[lower_left_index] = False
    return mask


def extract_mask(x, structure_size, mask_size):
    """
    Inputs:
    x.shape: [D1, D2, ..., D1, D2, ...]
    structure_size: [D1, D2, ...]
    mask_size: [m1, m2, ...]

    Outputs:
    output.shape: [D1, D2, ..., m1, m2, ...]
    """
    assert len(structure_size) == len(mask_size)
    assert torch.all(mask_size % 2 == 1)
    num_positional_dims = len(structure_size)
    assert torch.all(
        torch.tensor(
            x.shape[-num_positional_dims:], device=structure_size.device
        ) == structure_size
    ), "x.shape={}, structure_size={}".format(x.shape, structure_size)

    assert len(structure_size) == 1, "Only support 1D for now"

    # Pad k (mask_size) with C.
    # pad_size = mask_size//2 on the left and on the right of the content dimensions not the positional dimensions.
    # The total size will be structure_size + mask_size - 1
    pad_size = get_pad_size(mask_size)
    pad = tuple()
    for ps in pad_size:
        pad = (ps, ps) + pad
    x = F.pad(x, pad, value=False)
    assert torch.all(
        torch.tensor(
            x.shape[-num_positional_dims:], device=structure_size.device
        ) == (structure_size + mask_size - 1)
    )

    # Extract patches of size mask_size from each position in x
    startIdx = torch.arange(structure_size[0])
    x = torch.stack(
        tuple(x[i, s:(s + mask_size[0])] for i, s in enumerate(startIdx)),
        dim=0
    )
    return x

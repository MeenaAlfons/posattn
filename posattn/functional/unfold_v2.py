import torch
import torch.nn.functional as F
"""
Here we reestablish unfold_k and unfold_p side by side to make sure their conventions match.

Definitions:
    SeqLen -int- the number of keys/queries in the input
    structure_size -tuple()- The sizes of the positional dimensions of the input.
        prod(structure_size) = SeqLen
    mask_size -tuple()- The size of the relative mask which is centered around
        the position of the query or around the center of the relative
        positional encoding. mask_size <= 2*structure_size-1
    patch_size -tuple()- The size of the patches of keys that will be returned
        in the output. This could be <= structure_size.
    C -float- A constant to represent missing or ignored values in the output.

    Applying a relative mask has the effect of limiting the keys for each query along with their respective relative positional encoding.

    Inputs:
    k: The keys input of size [Batch, Head, HeadDims, <structure_size>]
    p: The relative positional encoding of size [Head, HeadDims, <mask_size>]

    Outputs:
    k_patched: The patched keys of size [Batch, Head, HeadDims, <structure_size>, <patch_size>]
    p_patched: The patched relative positional encoding of size [Head, HeadDims, <structure_size>, <patch_size>]

    The patched outputs of k and p need to have respective values for the key and its relative positional encoding relative to the query.

    Example: SeqLen = 5, structure_size = (5), mask_size = (5), patch_size = (5)
    k = [1,2,3,4,5], p = [-2,-1,0,1,2]
    k_patched = [
        [1,2,3,C,C],
        [1,2,3,4,C],
        [1,2,3,4,5],
        [C,2,3,4,5],
        [C,C,3,4,5],
    ]
    p_patched = [
        [ 0, 1, 2, C, C],
        [-1, 0, 1, 2, C],
        [-2,-1, 0, 1, 2],
        [ C,-2,-1, 0, 1],
        [ C, C,-2,-1, 0],
    ]

    Example: SeqLen = 5, structure_size = (5), mask_size = (3), patch_size = (3)
    k = [1,2,3,4,5], p = [-1,0,1]
    k_patched = [
        [1,2,C],
        [1,2,3],
        [2,3,4],
        [3,4,5],
        [C,4,5],
    ]
    p_patched = [
        [ 0, 1, C],
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
        [ C,-1, 0],
    ]

    Example: SeqLen = 5, structure_size = (5), mask_size = (7), patch_size = (5)
    k = [1,2,3,4,5], p = [-3,-2,-1,0,1,2,3]
    k_patched = [
        [1,2,3,4,C],
        [1,2,3,4,5],
        [1,2,3,4,5],
        [1,2,3,4,5],
        [C,2,3,4,5],
    ]
    p_patched = [
        [ 0, 1, 2, 3, C],
        [-1, 0, 1, 2, 3],
        [-2,-1, 0, 1, 2],
        [-3,-2,-1, 0, 1],
        [ C,-3,-2,-1, 0],
    ]

    Example: SeqLen = 7, structure_size = (7), mask_size = (5), patch_size = (5)
    k = [1,2,3,4,5,6,7], p = [-2,-1,0,1,2]
    k_patched = [
        [1,2,3,C,C],
        [1,2,3,4,C],
        [1,2,3,4,5],
        [2,3,4,5,6],
        [3,4,5,6,7],
        [C,4,5,6,7],
        [C,C,5,6,7],
    ]
    p_patched = [
        [ 0,  1,  2,  C, C],
        [-1,  0,  1,  2, C],
        [-2, -1,  0,  1, 2],
        [-2, -1,  0,  1, 2],
        [-2, -1,  0,  1, 2],
        [ C, -2, -1,  0, 1],
        [ C,  C, -2, -1, 0],
    ]


Implementation:
    k_patched:
        1. Unfold patches of size patch_size. The positional dimensions will be structure_size - patch_size + 1.
        2. Repeat first and last patches to reach structure_size. The number of repeats for each = (structure_size - (structure_size - patch_size + 1))/2 = (patch_size - 1)/2
        3. Fill upper right and lower left corners with C. The side of the triangle = p - mask_size//2 - 1. Apparently this equals (R-mask_size)/2 introduced in calculating p_patched.
    p_patched:
        1. Pad p (mask_size) with C to achieve size of R=2*patch_size-1. That
           means pad_size = (R - mask_size)/2 on the left and on the right of
           positional dimensions.
        2. Unfold patches of size patch_size.
        3. Flip the positional dimension of the queries.
        4. Repeat the middle patch number of times = stucture_size - patch_size + 1.
"""


def get_patch_size(structure_size, mask_size):
    patch_size = tuple()
    for i, m in enumerate(mask_size):
        s = structure_size[i]
        if m > s:
            p = s
        else:
            p = m
        patch_size += (p, )
    return torch.tensor(
        patch_size, dtype=torch.int64, device=structure_size.device
    )


def get_pad_size(mask_size, patch_size):
    # R is the size of the padded relative positional encoding
    # that can be unfold with patch_size to get the needed patches
    R = 2 * patch_size - 1

    # Pad relative positional encoding with padding_constant
    # The size of the padding in each positional dimension is (R-mask_size)/2
    # notice that R and mask_size are both lists of odd values
    pad_size = (R - mask_size) // 2

    # Analysing k_patched gives a different formula for pad_size
    # which should give the same result
    pad_size_2 = patch_size - mask_size // 2 - 1
    assert torch.all(pad_size == pad_size_2)

    return pad_size


def unfold_k(k, structure_size, mask_size, padding_constant=0.0):
    """
    Inputs:
    k is the relative positional encoding. It is already limited by the mask_size.
    k.shape: [Batch, Head, HeadDims, D1, D2, ...]
    structure_size: [D1, D2, ...]
    mask_size: [m1, m2, ...]
    padding_constant: A constant to represent missing or ignored values in the output.

    Outputs:
    k_patched is the patched relative positional encoding.
    k_patched.shape: [Batch, Head, HeadDims, D1, D2, ..., p1, p2, ...]
    where patch_size: [p1, p2, ...] (computed from mask_size and structure_size)
    """
    assert len(structure_size) == len(mask_size)
    assert torch.all(mask_size % 2 == 1)
    num_positional_dims = len(structure_size)
    assert torch.all(
        torch.tensor(
            k.shape[-num_positional_dims:], device=structure_size.device
        ) == structure_size
    ), "k.shape={}, structure_size={}".format(k.shape, structure_size)

    # Unfold patches of size patch_size
    patch_size = get_patch_size(structure_size, mask_size)
    for dim_size in patch_size:
        k = k.unfold(-num_positional_dims, dim_size, 1)

    assert torch.all(
        torch.tensor(
            k.shape[-num_positional_dims:], device=structure_size.device
        ) == patch_size
    ), "k.shape={}, patch_size={}".format(k.shape, patch_size)
    assert torch.all(
        torch.tensor(
            k.shape[-2 * num_positional_dims:-num_positional_dims],
            device=structure_size.device
        ) == (structure_size - patch_size + 1)
    )

    # Repeat first and last patches to reach structure_size.
    # The number of additional repeats for each = (structure_size - (structure_size - patch_size + 1))/2 = (patch_size - 1)/2
    current_size = structure_size - patch_size + 1
    num_additional_repeats = (patch_size - 1) // 2
    for i, additional_repeat in enumerate(num_additional_repeats):
        if additional_repeat > 0:
            repeats = torch.ones(
                current_size[i], dtype=torch.int64, device=k.device
            )
            # if current_size[i] == 1, both repeats are added to it
            repeats[0] += additional_repeat
            repeats[-1] += additional_repeat
            if patch_size[i] % 2 == 0:
                assert current_size[i] == 1
                repeats[0] += 1
            k = k.repeat_interleave(repeats, dim=(-2 * num_positional_dims + i))

    # After the repeat, the positional dimensions should be equal to the sturcture_size
    assert torch.all(
        torch.tensor(
            k.shape[-2 * num_positional_dims:-num_positional_dims],
            device=structure_size.device
        ) == structure_size
    ), "k.shape={}, structure_size={}".format(k.shape, structure_size)

    # Fill upper right and lower left corners with C.
    # This step is not needed in production, but it is useful for debugging.
    # A similar mask will be used to fill attention_logits with -inf.
    if __debug__:
        print("Executed __debug__ block")
        pad_size = get_pad_size(mask_size, patch_size)
        diagonal_number = patch_size - pad_size  # - 1
        for i, s in enumerate(structure_size):
            upper_row_idx, upper_col_idx = torch.triu_indices(
                s, patch_size[i], diagonal_number[i]
            )
            upper_right_index = [slice(None)] * len(k.shape)
            upper_right_index[-2 * num_positional_dims + i] = upper_row_idx
            upper_right_index[-num_positional_dims + i] = upper_col_idx
            k[upper_right_index] = padding_constant
            lower_row_idx, lower_col_idx = torch.tril_indices(
                s, patch_size[i], -diagonal_number[i] - (s - patch_size[i])
            )
            lower_left_index = [slice(None)] * len(k.shape)
            lower_left_index[-2 * num_positional_dims + i] = lower_row_idx
            lower_left_index[-num_positional_dims + i] = lower_col_idx
            k[lower_left_index] = padding_constant

    return k


def unfold_p(p, structure_size, mask_size, padding_constant=0.0):
    """
    Inputs:
    p is the relative positional encoding. It is already limited by the mask_size.
    p.shape: [Head, HeadDims, m1, m2, ...]
    structure_size: [D1, D2, ...]
    mask_size: [m1, m2, ...]
    padding_constant: A constant to represent missing or ignored values in the output.

    Outputs:
    p_patched is the patched relative positional encoding.
    p_patched.shape: [Head, HeadDims, D1, D2, ..., p1, p2, ...]
    where patch_size: [p1, p2, ...] (computed from mask_size and structure_size)
    """
    assert len(structure_size) == len(mask_size)
    assert torch.all(mask_size % 2 == 1)
    num_positional_dims = len(structure_size)
    assert torch.all(
        torch.tensor(
            p.shape[-num_positional_dims:], device=structure_size.device
        ) == mask_size
    ), "p.shape={}, mask_size={}".format(p.shape, mask_size)

    # Pad p (mask_size) with C to achieve size of R=2*patch_size-1
    # Notice that `pad` needs the last dimension first.
    patch_size = get_patch_size(structure_size, mask_size)
    pad_size = get_pad_size(mask_size, patch_size)
    pad = tuple()
    for ps in pad_size:
        pad = (ps, ps) + pad
    p = F.pad(p, pad, value=padding_constant)

    # Unfold patches of size patch_size
    for dim_size in patch_size:
        p = p.unfold(-num_positional_dims, dim_size, 1)

    assert torch.all(
        torch.tensor(
            p.shape[-num_positional_dims:], device=structure_size.device
        ) == patch_size
    )
    assert torch.all(
        torch.tensor(
            p.shape[-2 * num_positional_dims:-num_positional_dims],
            device=structure_size.device
        ) == patch_size
    )

    # Flip the positional dimension of the queries.
    flip_dims = tuple(range(-2 * num_positional_dims, -num_positional_dims))
    p = p.flip(flip_dims)

    # p now has a number of patches in each dimension equal to (R-patch_size+1) = patch_size
    # which could be less than the strucutre_size in any of the dimensions.
    # For that case, the middle patch needs to be repeated number of times equal to structure_size - patch_size + 1

    # Repeat the middle patch number of times = structure_size - patch_size + 1.
    repeat_middle_patch = structure_size - patch_size + 1
    for i, middle_repeat in enumerate(repeat_middle_patch):
        if middle_repeat > 1:
            repeats = torch.ones(
                patch_size[i], dtype=torch.int64, device=p.device
            )
            # what if patch_size[i] is even? Works well
            repeats[patch_size[i] // 2] = middle_repeat
            p = p.repeat_interleave(repeats, dim=(-2 * num_positional_dims + i))

    # After the repeat, the positional dimensions should be equal to the sturcture_size
    assert torch.all(
        torch.tensor(
            p.shape[-2 * num_positional_dims:-num_positional_dims],
            device=structure_size.device
        ) == structure_size
    )

    return p


def make_mask(structure_size, mask_size, device):
    num_positional_dims = len(structure_size)
    patch_size = get_patch_size(structure_size, mask_size)

    mask = torch.ones(
        tuple(structure_size) + tuple(patch_size),
        dtype=torch.bool,
        device=device
    )

    # Fill upper right and lower left corners with False.
    pad_size = get_pad_size(mask_size, patch_size)
    diagonal_number = patch_size - pad_size  # - 1
    for i, s in enumerate(structure_size):
        upper_row_idx, upper_col_idx = torch.triu_indices(
            s, patch_size[i], diagonal_number[i]
        )
        upper_right_index = [slice(None)] * len(mask.shape)
        upper_right_index[-2 * num_positional_dims + i] = upper_row_idx
        upper_right_index[-num_positional_dims + i] = upper_col_idx
        mask[upper_right_index] = False
        lower_row_idx, lower_col_idx = torch.tril_indices(
            s, patch_size[i], -diagonal_number[i] - (s - patch_size[i])
        )
        lower_left_index = [slice(None)] * len(mask.shape)
        lower_left_index[-2 * num_positional_dims + i] = lower_row_idx
        lower_left_index[-num_positional_dims + i] = lower_col_idx
        mask[lower_left_index] = False

    return mask


def extract_mask(x, structure_size, mask_size):
    """
    x.shape: [D1, D2, ..., D1, D2, ...]
    structure_size: [D1, D2, ...]
    mask_size: [m1, m2, ...]

    output.shape: [D1, D2, ..., p1, p2, ...]
    where patch_size: [p1, p2, ...] (computed from mask_size and structure_size)
    """
    assert len(structure_size) == len(mask_size)
    assert torch.all(mask_size % 2 == 1)
    num_positional_dims = len(structure_size)
    assert torch.all(
        torch.tensor(
            x.shape[-num_positional_dims:], device=structure_size.device
        ) == structure_size
    ), "x.shape={}, structure_size={}".format(x.shape, structure_size)
    assert torch.all(
        torch.tensor(
            x.shape[-2 * num_positional_dims:-num_positional_dims],
            device=structure_size.device
        ) == structure_size
    ), "x.shape={}, structure_size={}".format(x.shape, structure_size)

    assert len(structure_size) == 1, "Only support 1D for now"

    # Extract patches of size patch_size from each row of x
    patch_size = get_patch_size(structure_size, mask_size)
    startIdx = torch.arange(structure_size[0] - patch_size[0] + 1)

    # Repeat first and last patches to reach structure_size.
    # The number of additional repeats for each = (structure_size - (structure_size - patch_size + 1))/2 = (patch_size - 1)/2
    current_size = structure_size - patch_size + 1
    num_additional_repeats = (patch_size - 1) // 2
    i = 0  # Support only 1D for now
    additional_repeat = num_additional_repeats[i]
    if additional_repeat > 0:
        repeats = torch.ones(
            current_size[i], dtype=torch.int64, device=x.device
        )
        # if current_size[i] == 1, both repeats are added to it
        repeats[0] += additional_repeat
        repeats[-1] += additional_repeat
        if patch_size[i] % 2 == 0:
            assert current_size[i] == 1
            repeats[0] += 1
        startIdx = startIdx.repeat_interleave(
            repeats, dim=(-num_positional_dims + i)
        )

    x = torch.stack(
        tuple(x[i, s:(s + patch_size[0])] for i, s in enumerate(startIdx)),
        dim=0
    )
    return x

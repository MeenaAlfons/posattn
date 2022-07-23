import torch
import torch.nn.functional as F


def unfold_structure(x, structure_size):
    """
    Unfold patches of sturcture_size from x.
    The unfolding is applied on the last set of dimensions.

    Args:
        x: [..., 2*D1-1, 2*D2-1, ...]
        structure_size: [D1, D2, ...]

    Returns:
        [..., D1, D2, ..., D1, D2, ...]
    """
    num_positional_dims = len(structure_size)
    for positional_dim in structure_size:
        x = x.unfold(-num_positional_dims, positional_dim, 1)
    return x


def relative_unfold(x, structure_size):
    """
    Unfold patches of sturcture_size from x
    + Flip the positional dimensions.

    Args:
        x: [..., 2*D1-1, 2*D2-1, ...]
        structure_size: [D1, D2, ...]

    Returns:
        [..., flip(D1), flip(D2), ..., D1, D2, ...]
    """
    x = unfold_structure(x, structure_size)
    num_positional_dims = len(structure_size)
    flip_dims = tuple(range(-2 * num_positional_dims, -num_positional_dims))
    # torch.flip makes a copy of the data which renders it inefficient.
    # I tried integer indexing but it also makes a copy to get a flipped view.
    # Note that negative strides are not supported yet.
    # See:
    #       - https://github.com/pytorch/pytorch/issues/229
    #       - https://github.com/pytorch/pytorch/issues/59786
    x = x.flip(flip_dims)
    return x


def relative_unfold_flatten(x, structure_size):
    """
    Unfold patches of sturcture_size from x
    + Flip the positional dimensions.
    + Flatten positional dimensions

    Args:
        x: [..., 2*D1-1, 2*D2-1, ...]
        structure_size: [D1, D2, ...]
    Returns:
        [..., D1*D2*..., D1*D2*...]
    """
    x = relative_unfold(x, structure_size)
    num_positional_dims = len(structure_size)
    non_positional_dims = x.shape[:-2 * num_positional_dims]
    seq_len = torch.prod(structure_size).item()
    x = x.reshape(*non_positional_dims, seq_len, seq_len)
    return x


def relative_patch_unfold_flatten(x, structure_size, patch_size):
    """
    
    Args:
        x.shape: [..., 2*D1-1, 2*D2-1, ...]
        structure_size: [D1, D2, ...]
        patch_size: [S1, S2, ...]
    """

    pass


def patch_unfold_1d(x, patch_size):
    """
    Unfold patches of sturcture_size from x
    
    Args:
        x: [Batch, Head, SeqLen, HeadDims]
    Returns:
        [Batch, Head, SeqLen, PatchLen, HeadDims]
        PatchLen = prod(structure_size)
    """
    seq_len = x.shape[-2]
    x = x.movedim(-1, -2)
    pad_size = int(patch_size) // 2
    x = F.pad(x, (pad_size, pad_size), mode='constant', value=0.0)
    x = x.unfold(-1, patch_size, 1)
    assert (x.shape[-1] == patch_size)
    assert (x.shape[-2] == seq_len)
    x = x.permute(0, 1, 3, 4, 2)
    return x

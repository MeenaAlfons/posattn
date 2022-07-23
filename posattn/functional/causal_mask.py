import torch


def make_causal_mask(structure_size, device):
    """
    Creates a causal mask for the transformer.
    """
    # 1D should look like this:
    # structure_size = (4,)
    # 1 0 0 0
    # 1 1 0 0
    # 1 1 1 0
    # 1 1 1 1

    # 2D should look like this:
    # structure_size = (3, 3)
    # 1 0 0 0 0 0 0 0 0
    # 1 1 0 0 0 0 0 0 0
    # 1 1 1 0 0 0 0 0 0
    # 1 0 0 1 0 0 0 0 0
    # 1 1 0 1 1 0 0 0 0
    # 1 1 1 1 1 1 0 0 0
    # 1 0 0 1 0 0 1 0 0
    # 1 1 0 1 1 0 1 1 0
    # 1 1 1 1 1 1 1 1 1

    if len(structure_size) == 1:
        seq_len = structure_size[0]
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        for i in range(seq_len):
            mask[i, 0:(i + 1)] = 1
        mask.requires_grad = False
        return mask
    else:
        raise NotImplementedError("Only 1D causal mask is supported.")

    # Make a structured mask of size [D1, D2, D1, D2]
    # where structured_mask[1,1] should be a causal mask of size [D1, D2]
    # with ones for the positions where D1<i and D1<j
    structured_mask = torch.zeros(
        tuple(structure_size) + tuple(structure_size),
        dtype=dtype,
        device=device
    )

    # generate all positions (Meshgrid)
    seq_len = torch.prod(structure_size).item()
    num_dims = len(structure_size)
    vectors = []
    for dim in structure_size:
        vectors += [torch.arange(dim, dtype=dtype, device=device)]
    meshgrid = torch.stack(
        torch.meshgrid(
            vectors,
            indexing='ij',
        ), dim=len(vectors)
    ).view(seq_len, num_dims)

    # position is a one dimensional vector of number of elements equal to num_dims.
    # Example positionfor 2D strucutre_size of (3,3):
    # - [0, 1]
    # - [1, 2]
    # - [2, 0]
    for position in meshgrid:
        # We need to generate the volume of the mask for each position.
        # which is the cross product of the following boolean vectors:
        # - (vectors[0] < position[0])
        # - (vectors[1] < position[1])
        # - etc.

        # The following two lines have the same effect
        structured_mask[position][causality_condition] = 1
        structured_mask[position] = causality_condition

    # For each position in the structured mask,
    # mask_structured[position][causality_condition] = 1

    # Flatten the mask into [seq_len, seq_len]
    # where seq_len = D1 * D2
    mask_flat = mask_structured.view(seq_len, seq_len)

    return mask_flat

import torch


def make_position_meshgrid(
    dims, dtype, device, relative, normalized, all_positive=False
):
    vectors = []
    for dim in dims:
        vectors += [
            make_single_dimension(
                int(dim), dtype, device, relative, normalized, all_positive
            )
        ]

    return torch.stack(
        torch.meshgrid(
            vectors,
            indexing='ij',
        ), dim=len(vectors)
    )


def make_single_dimension(
    size, dtype, device, relative, normalized, all_positive=False
):
    if normalized:
        minimum = 0 if all_positive else -1
        maximum = 1
        if relative:
            return torch.linspace(
                minimum, maximum, 2 * size - 1, dtype=dtype, device=device
            )
        else:
            return torch.linspace(
                minimum, maximum, size, dtype=dtype, device=device
            )
    else:
        if relative:
            start = 0 if all_positive else (-size + 1)
            end = 2 * size - 1 if all_positive else size
            return torch.arange(start, end, dtype=dtype, device=device)
        else:
            return torch.arange(size, dtype=dtype, device=device)

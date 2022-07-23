import torch
import torch.nn as nn


class GaussianPositionalMask(nn.Module):
    def __init__(self, num_positional_dims, meshgrid_cache, sigma, learn_sigma):
        super().__init__()
        self.num_positional_dims = num_positional_dims
        self.meshgrid_cache = meshgrid_cache

        # For a general parameterization of the Gaussian, we need to parameterize the full covariance matrix.
        # However, it's a bit complex to parameterize the covariance matrix to be learnable while staying constrained to be positive semidefinite.
        # For now, we are going to use a diagonal covariance matrix (independent dimensions).

        # Sigma for each dimension
        self.sigma = nn.Parameter(torch.zeros(num_positional_dims))
        self.sigma.requires_grad = learn_sigma
        nn.init.constant_(self.sigma, sigma)

    def forward(self, structure_size, mask_size, dtype, device):
        assert len(structure_size) == self.num_positional_dims

        # Meshgrid of positions
        # position:
        # 1D: [rD1, 1]
        # 2D: [rD1, rD2, 2]
        # 3D: [rD1, rD2, rD3, 3]
        meshgrid = self.meshgrid_cache.make(
            structure_size,
            dtype=dtype,
            device=device,
            relative=True,
            normalized=True
        )

        # Crop the meshgrid to the mask size
        from_idx = (structure_size * 2 - 1 - mask_size) // 2
        to_idx = from_idx + mask_size  # exclusive
        index = tuple(slice(f, t) for f, t in zip(from_idx, to_idx))
        meshgrid = meshgrid[index]

        # gauss_mask:
        # 1D: [rD1, 1]
        # 2D: [rD1, rD2, 1]
        # 3D: [rD1, rD2, rD3, 1]

        # Formula for Gaussian function:
        # f1(x) = 1 / (sqrt(2 * pi) * sigma) * exp(-(x - mu)^2 / (2 * sigma^2))
        # f2(x, y) = f1(x) * f1(y)
        # mu = 0
        # f2(x, y) = constant * exp(exponent)
        # constant = 1 / product(sigma) * (1 / (2 * pi)^(D / 2))
        # exponent = -0.5 * (x^2 * sigma_x^2 + y^2 * sigma_y^2)

        sigma_receprocal_2 = 1.0 / self.sigma**2

        sigma_receprocal_2_view = sigma_receprocal_2[(None, ) *
                                                     self.num_positional_dims +
                                                     (Ellipsis, )]

        assert sigma_receprocal_2_view.dim() == meshgrid.dim()

        # sum( x^2 / sigma_x^2 + y^2 / sigma_y^2)
        exponent = -0.5 * torch.sum(
            meshgrid.pow(2) * sigma_receprocal_2_view, dim=-1, keepdim=True
        )

        gauss_mask = torch.exp(exponent)
        return gauss_mask

    def get_mask_size(self, positional_mask_threshold, structure_size):
        # z = gauss(x1, x2) = exp( -0.5 * (x1^2 / sigma_x1^2 + x2^2 / sigma_x2^2) )
        # We need to find x1 and x2 at which y = positional_mask_threshold.
        # We put x1 = 0 to find x2 and x2 =0 to find x1.
        # x1 = gauss_inverse(z) = sqrt(-2* ln(z) * sigma_x1^2)
        # x2 = gauss_inverse(z) = sqrt(-2* ln(z) * sigma_x2^2)
        # Using x1 and x2 we can find the size of the mask.
        # Assuming that the meshgrid has the range [-1, 1] for the positional range of [-S+1, S-1]
        # where S is the structure size in one of the dimensions.
        # Therefore, half the mask size in the direction of x1 is: (x1 / 1) * (S - 1)
        # The full mask_size = 2 * half_mask_size + 1

        max_mask_size = 2 * structure_size - 1

        if positional_mask_threshold <= 0:
            return max_mask_size

        x = torch.sqrt(
            -2 * torch.log(
                torch.
                tensor(positional_mask_threshold, device=structure_size.device)
            ) * self.sigma**2
        )
        half_mask_size = torch.ceil((x / 1) * (structure_size - 1)).int()
        mask_size = 2 * half_mask_size + 1

        for i in range(len(structure_size)):
            mask_size[i] = torch.clamp(mask_size[i], max=max_mask_size[i])
        return mask_size

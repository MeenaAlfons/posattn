import torch.nn as nn
from . import mlp


class ImplicitPositionalEncoding(nn.Module):
    def __init__(
        self, meshgrid_cache, num_positional_dims, hidden_dim, output_dim,
        num_layers, activation, final_activation, activation_params, normalized
    ):
        super().__init__()
        self.meshgrid_cache = meshgrid_cache
        self.num_positional_dims = num_positional_dims
        self.normalized = normalized

        def activation_func(layer_index, num_layers):
            if layer_index == num_layers - 1:
                return mlp.make_activation(
                    final_activation, activation_params, layer_index, num_layers
                )
            else:
                return mlp.make_activation(
                    activation, activation_params, layer_index, num_layers
                )

        if activation == 'Sine':
            initializer = mlp.make_siren_initializer(w0=activation_params['w0'])
        else:
            initializer = mlp.default_initializer

        self.mlp = mlp.MLP(
            dim_in=num_positional_dims,
            dim_hidden=hidden_dim,
            dim_out=output_dim,
            num_layers=num_layers,
            activation=activation_func,
            bias=True,
            initializer=initializer
        )

    def forward(self, structure_size, mask_size, dtype, device):
        assert len(structure_size) == self.num_positional_dims

        # Meshgrid of positions
        # position:
        # 1D: [2*D1-1, 1]
        # 2D: [2*D1-1, 2*D2-1, 2]
        # 3D: [2*D1-1, 2*D2-1, 2*D3-1, 3]
        meshgrid = self.meshgrid_cache.make(
            structure_size,
            dtype=dtype,
            device=device,
            relative=True,
            normalized=self.normalized
        )

        # Crop the meshgrid to the mask size
        from_idx = (structure_size * 2 - 1 - mask_size) // 2
        to_idx = from_idx + mask_size  # exclusive
        index = tuple(slice(f, t) for f, t in zip(from_idx, to_idx))
        meshgrid = meshgrid[index]

        # p: [2*D1-1, 2*D2-1, OutDim]
        p = self.mlp(meshgrid)
        return p

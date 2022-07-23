import torch
import torch.nn as nn
import einops


class BlockArgs:
    def __init__(
        self,
        encoder_args,
        resolution_reduction_args,
    ):
        self.encoder_args = encoder_args
        self.resolution_reduction_args = resolution_reduction_args

    def __repr__(self):
        return (f"BlockArgs {self.__dict__}")


class EncoderArgs:
    def __init__(
        self,
        model_dim,
        dim_feedforward,
        dropout,
        self_attn,
    ):
        self.model_dim = model_dim
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.self_attn = self_attn

    def __repr__(self):
        return (f"EncoderArgs {self.__dict__}")


class ResolutionReductionArgs:
    def __init__(self, enabled, num_positional_dims, kernel_size):
        self.enabled = enabled
        self.num_positional_dims = num_positional_dims
        self.kernel_size = kernel_size

    def __repr__(self):
        return (f"ResolutionReductionArgs {self.__dict__}")


class Transformer(nn.Module):
    def __init__(self, block_args_list):
        super().__init__()
        layers = []
        for block_args in block_args_list:
            print(
                'creating layer {} with args: {}'.format(
                    len(layers), block_args.__dict__
                )
            )
            layers.append(
                nn.ModuleList(
                    [
                        EncoderBlock(**block_args.encoder_args.__dict__),
                        ResolutionReduction(
                            **block_args.resolution_reduction_args.__dict__
                        )
                    ]
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x, mask=None, structure_size=None):
        for l in self.layers:
            x = l[0](x, mask=mask, structure_size=structure_size)
            x, structure_size = l[1](
                x, mask=mask, structure_size=structure_size
            )
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        model_dim,
        dim_feedforward,
        dropout,
        self_attn,
    ):
        super().__init__()

        # Attention layer
        self.self_attn = self_attn

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(model_dim, dim_feedforward), nn.Dropout(dropout),
            nn.ReLU(inplace=True), nn.Linear(dim_feedforward, model_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, structure_size=None):
        # Attention part
        x = x + self.dropout(
            self.self_attn(x, mask=mask, structure_size=structure_size)
        )
        x = self.norm1(x)

        # MLP part
        x = x + self.dropout(self.linear_net(x))
        x = self.norm2(x)

        return x


class ResolutionReduction(nn.Module):
    def __init__(self, enabled, num_positional_dims, kernel_size):
        super().__init__()
        self.enabled = enabled
        self.num_positional_dims = num_positional_dims

        if enabled:
            if num_positional_dims == 1:
                self.max_pool = nn.MaxPool1d(
                    kernel_size=kernel_size, stride=kernel_size
                )
            elif num_positional_dims == 2:
                self.max_pool = nn.MaxPool2d(
                    kernel_size=kernel_size, stride=kernel_size
                )

    def forward(self, x, mask=None, structure_size=None):
        if not self.enabled:
            return x, structure_size

        if mask is not None:
            raise NotImplementedError(
                "ResolutionReduction with mask not implemented"
            )

        assert len(structure_size) == self.num_positional_dims

        seq_len = x.shape[1]
        assert seq_len == torch.prod(structure_size)

        # Note: A better implementation could be done if we replaces all linear layers with Conv1d
        # Then we can always keep the seq_len last dimension.

        x = einops.rearrange(x, 'b l d -> b d l')
        x = x.view(x.shape[0], x.shape[1], *structure_size)

        x = self.max_pool(x)

        new_structure_size = torch.tensor(
            x.shape[2:], device=structure_size.device
        )
        new_seq_len = torch.prod(new_structure_size)

        x = x.view(x.shape[0], x.shape[1], new_seq_len)
        x = einops.rearrange(x, 'b d l -> b l d')

        return x, new_structure_size

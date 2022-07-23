import torch
from torch import nn
from ..functional import CausalMaskCache


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        model_dim,
        transformer,
        dropout,
        causal,
        num_positional_dims,
        cls_token,
    ):
        super().__init__()

        self.cls_token = cls_token
        self.causal = causal
        self.causal_mask_cache = CausalMaskCache(self)
        self.num_positional_dims = num_positional_dims

        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_dim, model_dim)
        )

        # Transformer layer
        self.transformer = transformer

        # Output classifier from the last token
        # TODO This layer could be simplified (e.g. by using a single linear layer)
        self.output_net = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes)
        )

    def name(self):
        return "TransformerClassifier"

    def forward(self, x):
        structure_size = torch.tensor(
            x.shape[1:-1], dtype=torch.int64, device=x.device
        )
        if self.num_positional_dims == 1:
            structure_size = torch.prod(structure_size, dim=0, keepdim=True)

        mask = None
        if self.causal:
            # Generate proper mask for causal settings
            mask = self.causal_mask_cache.make(structure_size, x.device)

        # Flatten positional dimensions
        x = x.view(x.size(0), -1, x.size(-1))

        # Input layer
        x = self.input_net(x)

        # Transformer layer
        x = self.transformer(x, mask=mask, structure_size=structure_size)

        # Take the last token
        if self.cls_token == 'last':
            x = x[:, -1, :]
        elif self.cls_token == 'mean':
            x = torch.mean(x, dim=1)
        elif self.cls_token == 'max':
            x = torch.max(x, dim=1)[0]
        else:
            raise ValueError(f"Unknown cls_token {self.cls_token}")

        # Output layer
        x = self.output_net(x)

        return x

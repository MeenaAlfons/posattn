import torch
import torch.nn as nn
import einops


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        model_dim,
        num_heads,
        attention,
    ):
        super().__init__()
        assert model_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.attention = attention
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv_proj = nn.Linear(model_dim, 3 * model_dim)
        self.o_proj = nn.Linear(model_dim, model_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, structure_size=None):
        # TODO consider taking SeqLen as the last dimension returning it as last dimension.
        # This will save a lot of copying
        # This is generally challenging as all linear layers work on the last dimension by default :(

        assert x.dim() == 3, "Input must have shape = (Batch, SeqLen, InputDim)"

        # qkv.shape = (Batch, SeqLen, 3*EmbedDim)
        qkv = self.qkv_proj(x)

        # q, k, v: [Batch, Head, HeadDims, SeqLen]
        q, k, v = torch.split(
            einops.rearrange(
                qkv, 'b s (n h d)->n b h d s', n=3, h=self.num_heads
            ), 1
        )
        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)

        # values: [Batch, Head, HeadDims, SeqLen]
        values = self.attention(q, k, v, structure_size, mask)

        # Merge heads
        values = einops.rearrange(values, 'b h d q -> b q (h d)')

        # Output values
        values = self.o_proj(values)

        return values

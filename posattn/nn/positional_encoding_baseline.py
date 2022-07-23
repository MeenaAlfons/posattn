import torch
import torch.nn as nn
import math


class BaselinePositionalEncoding(nn.Module):
    def __init__(self, output_dim, hidden_dim, **kwargs):
        super().__init__()
        print('BaselinePositionalEncoding')
        self.cache = {}
        self.output_dim = output_dim
        self.num_positional_dims = 1
        self.hidden_dim = hidden_dim
        self.w = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def make_pe(self, size, dtype, device):
        print('called make_pe device {}'.format(device))
        max_len = 2 * size - 1

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        position = torch.arange(0, max_len, dtype=dtype,
                                device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_dim, 2, dtype=dtype, device=device) *
            (-math.log(10000.0) / self.hidden_dim)
        )

        pe_sin = torch.sin(position * div_term)
        pe_cos = torch.cos(position * div_term)

        pe = torch.reshape(
            torch.cat((pe_sin[..., None], pe_cos[..., None]), dim=-1),
            [pe_sin.size(0), self.hidden_dim]
        )

        # This way gives the following warning:
        # Warning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.
        # pe1 = torch.zeros(max_len, self.hidden_dim, device=device)
        # pe1[:, 0::2] = pe_sin
        # pe1[:, 1::2] = pe_cos
        # assert torch.allclose(pe, pe1, atol=1e-5)

        assert torch.all(
            torch.tensor(pe.shape, device=device) ==
            torch.tensor([max_len, self.hidden_dim], device=device)
        ), "pe.shape={}, size={}".format(pe.shape, size)

        return pe

    def pe(self, size, dtype, device):
        key = '{}'.format(size)
        if key not in self.cache:
            self.cache[key] = self.make_pe(size, dtype, device)
        return self.cache[key]

    def forward(self, structure_size, mask_size, dtype, device):
        assert len(structure_size) == 1
        pe = self.pe(structure_size[0], dtype, device)

        # Crop the meshgrid to the mask size
        from_idx = (structure_size * 2 - 1 - mask_size) // 2
        to_idx = from_idx + mask_size  # exclusive
        index = tuple(slice(f, t) for f, t in zip(from_idx, to_idx))
        index += (slice(None), )
        pe = pe[index]

        # pe: [2*SeqLen-1, HiddenDim]
        # out: [2*SeqLen-1, OutputDim]
        out = self.w(pe)

        return out

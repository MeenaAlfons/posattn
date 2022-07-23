import torch
import einops
from .positional_attention import PositionalAttentionBase
from ..functional.unfold_v1 import unfold_p, make_mask


class PositionalAttentionV1(PositionalAttentionBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def content_logits(self, q, k, structure_size, mask_size):
        """
        Computes the content logits
        including content_bias_logits using `self.u`.

        Args:
            q: [Batch, Head, HeadDims, SeqLen]
            k: [Batch, Head, HeadDims, SeqLen]
        
        Returns:
            content_logits: [Batch, Head, SeqLen, SeqLen]
        """
        # [Batch, Head, QSeqLen, KSeqLen]
        content_logits = torch.einsum('b h d q, b h d k -> b h q k', q, k)

        # [Batch, Head, 1, KSeqLen]
        content_bias_logits = torch.einsum('h d, b h d k -> b h k', self.u,
                                           k)[:, :, None, :]

        return content_logits + content_bias_logits

    def positional_logits(self, q, structure_size, mask_size):
        """
        Computes the positional logits using self.positional_encoding module
        including positional_bias_logits using `self.v`.

        Args:
            q: [Batch, Head, HeadDims, SeqLen]
        
        Returns:
            positional_logits: [Batch, Head, SeqLen, SeqLen]
        """
        # p: [<2*structure_size-1>, Head*HeadDims]
        p = self.positional_encoding(
            structure_size,
            2 * structure_size - 1,
            dtype=q.dtype,
            device=q.device
        )

        # Move last dimension to the front and split heads
        p = einops.rearrange(p, '... (h d) -> h d ...', h=self.num_heads)

        # p: [Head * HeadDims, <structure_size>, <structure_size>]
        p = unfold_p(p, structure_size)

        # p: [Head, HeadDims, QSeqLen, KSeqLen]
        p = p.contiguous().view(q.size(1), q.size(2), q.size(3), -1)

        # [Batch, Head, QSeqLen, KSeqLen]
        relative_position_logits = torch.einsum(
            'b h d q, h d q k -> b h q k', q, p
        )

        # [1, Head, QSeqLen, KSeqLen]
        relative_position_bias_logits = torch.einsum(
            'h d, h d q k -> h q k', self.v, p
        )[None, :, :, :]

        return relative_position_logits + relative_position_bias_logits

    def output_values(self, attention, v, structure_size, mask_size):
        """
        Computes the weighted sum of values in `v` according to the scores in `attention`.
        
        Args:
            attention: [Batch, Head, SeqLen, SeqLen]
            v: [Batch, Head, HeadDims, SeqLen]
            
        Returns:
            [Batch, Head, HeadDims, SeqLen]
        """
        values = torch.einsum('b h q k, b h d k -> b h d q', attention, v)
        return values

    def positional_weights_and_mask(
        self, structure_size, mask_size, dtype, device
    ):
        """
        Computes the positional weights and mask.
        The positiona_weights need to be extracted from self.positional_mask module.
        The positional_mask needs to be a boolean mask to indicate which attention logits
        are relevant from the attention_logits of size [SeqLen, SeqLen]

        Returns:
            positional_weights: [1, 1, SeqLen, SeqLen]
            positional_mask: [SeqLen, SeqLen]
        """
        # positional_weights: [<2*structure_size-1>, 1]
        positional_weights = self.positional_mask(
            structure_size, structure_size * 2 - 1, dtype=dtype, device=device
        )
        assert positional_weights.size(-1) == 1

        # remove last dimension
        positional_weights = positional_weights.view(
            positional_weights.size()[:-1]
        )

        # positional_weights: [<structure_size>, <structure_size>]
        positional_weights = unfold_p(positional_weights, structure_size)

        # positional_weights: [1, 1, SeqLen, SeqLen]
        positional_weights = positional_weights.contiguous().view(
            1, 1, torch.prod(structure_size), -1
        )

        # positional_mask: [<structure_size>, <structure_size>]
        positional_mask = make_mask(structure_size, mask_size, device)

        # positional_mask: [SeqLen, SeqLen]
        positional_mask = positional_mask.view(torch.prod(structure_size), -1)

        return positional_weights, positional_mask

    def merge_masks(self, mask, positional_mask, structure_size, mask_size):
        """
        Merges the mask with the positional mask.
        Args:
            mask: [SeqLen, SeqLen]
            positional_mask: [SeqLen, SeqLen]
        
        Returns:
            mask: [SeqLen, SeqLen]
        """
        mask_so_far = None
        if mask is not None:
            mask_so_far = mask

        if positional_mask is not None:
            if mask_so_far is None:
                mask_so_far = positional_mask
            else:
                mask_so_far = mask_so_far & positional_mask
        return mask_so_far

import torch
import einops
from .positional_attention import PositionalAttentionBase
from ..functional.unfold_v3 import unfold_k, make_mask, extract_mask


class PositionalAttentionV3(PositionalAttentionBase):
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
            content_logits: [Batch, Head, SeqLen, MaskSize]
        """
        # k_patched: [Batch, Head, HeadDims, <structure_size>, <mask_size>]
        k_patched = unfold_k(
            k.view((q.size(0), q.size(1), q.size(2)) + tuple(structure_size)),
            structure_size, mask_size
        )

        # k_patched: [Batch, Head, HeadDims, SeqLen, MaskSize]
        k_patched = k_patched.contiguous().view(tuple(q.shape) + (-1, ))

        # [Batch, Head, SeqLen, MaskSize]
        content_logits = torch.einsum(
            'b h d q, b h d q m -> b h q m', q, k_patched
        )

        # [Batch, Head, SeqLen, MaskSize]
        content_bias_logits = torch.einsum(
            'h d, b h d q m -> b h q m', self.u, k_patched
        )

        return content_logits + content_bias_logits

    def positional_logits(self, q, structure_size, mask_size):
        """
        Computes the positional logits using self.positional_encoding module
        including positional_bias_logits using `self.v`.

        Args:
            q: [Batch, Head, HeadDims, SeqLen]
        
        Returns:
            positional_logits: [Batch, Head, SeqLen, MaskSize]
        """
        # p: [<mask_size>, Head*HeadDims]
        p = self.positional_encoding(
            structure_size, mask_size, dtype=q.dtype, device=q.device
        )

        # Move last dimension to the front and split heads
        p = einops.rearrange(p, '... (h d) -> h d ...', h=self.num_heads)

        # p: [Head, HeadDims, MaskSize]
        p = p.view(q.size(1), q.size(2), -1)

        # [Batch, Head, SeqLen, MaskSize]
        relative_position_logits = torch.einsum(
            'b h d q, h d m -> b h q m', q, p
        )

        # [1, Head, 1, MaskSize]
        relative_position_bias_logits = torch.einsum(
            'h d, h d m -> h m', self.v, p
        )[None, :, None, :]

        return relative_position_logits + relative_position_bias_logits

    def output_values(self, attention, v, structure_size, mask_size):
        """
        Computes the weighted sum of values in `v` according to the scores in `attention`.
        
        Args:
            attention: [Batch, Head, SeqLen, MaskSize]
            v: [Batch, Head, HeadDims, SeqLen]
            
        Returns:
            [Batch, Head, HeadDims, SeqLen]
        """
        # v_patched: [Batch, Head, HeadDims, <structure_size>, <mask_size>]
        v_patched = unfold_k(
            v.view((v.size(0), v.size(1), v.size(2)) + tuple(structure_size)),
            structure_size, mask_size
        )

        # v_patched: [Batch, Head, HeadDims, SeqLen, MaskSize]
        v_patched = v_patched.contiguous().view(tuple(v.shape) + (-1, ))

        # [Batch, Head, HeadDims, SeqLen]
        values = torch.einsum(
            'b h q m, b h d q m -> b h d q', attention, v_patched
        )
        return values

    def positional_weights_and_mask(
        self, structure_size, mask_size, dtype, device
    ):
        """
        Computes the positional weights and mask.
        The positiona_weights need to be extracted from self.positional_mask module.
        The positional_mask needs to be a boolean mask to indicate which attention logits
        are relevant from the attention_logits of size [SeqLen, MaskSize]

        Returns:
            positional_weights: [1, 1, 1, MaskSize]
            positional_mask: [SeqLen, MaskSize]
        """
        # positional_weights: [<mask_size>, 1]
        positional_weights = self.positional_mask(
            structure_size, mask_size, dtype=dtype, device=device
        )
        assert positional_weights.size(-1) == 1

        # positional_weights: [1, 1, 1, MaskSize]
        positional_weights = positional_weights.view(1, 1, 1, -1)

        # positional_mask: [<structure_size>, <mask_size>]
        positional_mask = make_mask(structure_size, mask_size, device=device)

        # positional_mask: [SeqLen, MaskSize]
        positional_mask = positional_mask.view(torch.prod(structure_size), -1)

        return positional_weights, positional_mask

    def merge_masks(self, mask, positional_mask, structure_size, mask_size):
        """
        Merges the mask with the positional mask.
        The implementation depends on the layout of the logits.

        Args:
            mask: [SeqLen, SeqLen]
            positional_mask: [SeqLen, MaskSize]
        
        Returns:
            mask: [SeqLen, MaskSize]
        """
        mask_so_far = None
        if mask is not None:
            mask_so_far = extract_mask(mask, structure_size, mask_size)

        if positional_mask is not None:
            if mask_so_far is None:
                mask_so_far = positional_mask
            else:
                mask_so_far = mask_so_far & positional_mask
        return mask_so_far

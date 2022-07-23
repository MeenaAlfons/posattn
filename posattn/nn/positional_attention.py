import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalAttentionBase(nn.Module):
    def __init__(
        self,
        model_dim,
        num_heads,
        positional_encoding,
        positional_mask,
        positional_mask_threshold,
        positional_logits_operation,
        multiply_positional_mask_with,
    ):
        super().__init__()
        assert model_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.num_heads = num_heads
        self.positional_encoding = positional_encoding
        self.positional_mask = positional_mask
        self.positional_mask_threshold = positional_mask_threshold
        self.positional_logits_operation = positional_logits_operation
        self.multiply_positional_mask_with = multiply_positional_mask_with

        head_dim = model_dim // num_heads
        self.v = nn.Parameter(torch.Tensor(num_heads, head_dim))
        self.u = nn.Parameter(torch.Tensor(num_heads, head_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        # TODO This initialization is arbitrary and needs to be revisited.
        nn.init.xavier_uniform_(self.v.data)
        nn.init.xavier_uniform_(self.u.data)

    def forward(self, q, k, v, structure_size, mask=None):
        """
        Computes the weighted values according to
          - The content attention between q and k.
          - The positional attention according to the relative position between q and k.
          - Limiting the attention and applying weights to the attention according to the mask.

        Args:
            q: [Batch, Head, HeadDims, QSeqLen]
            k: [Batch, Head, HeadDims, KSeqLen]
            v: [Batch, Head, HeadDims, KSeqLen]
            mask: [SeqLen, SeqLen]
            
        Returns:
            [Batch, Head, HeadDims, QSeqLen]
        """
        assert torch.prod(structure_size) == q.size(3)
        # TODO more asserts

        mask_size = None
        positional_weights = None
        positional_mask = None
        if self.positional_mask:
            # Get Mask Size
            mask_size = self.positional_mask.get_mask_size(
                self.positional_mask_threshold, structure_size
            )

            positional_weights, positional_mask = self.positional_weights_and_mask(
                structure_size, mask_size, q.dtype, q.device
            )

        mask = self.merge_masks(
            mask, positional_mask, structure_size, mask_size
        )

        # Compute content logits
        attention_logits = self.content_logits(q, k, structure_size, mask_size)

        # Compute positional logits
        if self.positional_encoding:
            positional_logits = self.positional_logits(
                q, structure_size, mask_size
            )

            # If needed, apply positional weights from the mask on positional logits
            if positional_weights is not None and self.multiply_positional_mask_with == 'positional_logits':
                positional_logits *= positional_weights

            # Merges positional logits with content logits
            if self.positional_logits_operation == 'add':
                attention_logits += positional_logits
            elif self.positional_logits_operation == 'multiply':
                attention_logits *= positional_logits
            else:
                # TODO Concat is another option
                raise ValueError(
                    f"Unknown positional logits operation: {self.positional_logits_operation}"
                )

        # If needed, apply positional weights from the mask on attention logits
        if positional_weights is not None and self.multiply_positional_mask_with == 'all_logits':
            attention_logits *= positional_weights

        # Scaling
        attention_logits = attention_logits / math.sqrt(q.size(-2))

        # Masking
        if mask is not None:
            attention_logits = attention_logits.masked_fill(~mask, -torch.inf)

        attention = F.softmax(attention_logits, dim=-1)

        values = self.output_values(attention, v, structure_size, mask_size)
        return values

    def content_logits(self, q, k, structure_size, mask_size):
        """
        Computes the content logits
        including content_bias_logits using `self.u`.

        Args:
            q: [Batch, Head, HeadDims, SeqLen]
            k: [Batch, Head, HeadDims, SeqLen]
        
        Returns:
            content_logits: [Batch, Head, SeqLen, PatchSize]
            PatchSize: depends on the implementation
        """
        raise NotImplementedError("Please implement content_logits")

    def positional_logits(self, q, structure_size, mask_size):
        """
        Computes the positional logits using self.positional_encoding module
        including positional_bias_logits using `self.v`.

        Args:
            q: [Batch, Head, HeadDims, SeqLen]
        
        Returns:
            positional_logits: [Batch, Head, SeqLen, PatchSize]
            PatchSize: depends on the implementation
        """
        raise NotImplementedError("Please implement positional_logits")

    def output_values(self, attention, v, structure_size, mask_size):
        """
        Computes the weighted sum of values in `v` according to the scores in `attention`.
        
        Args:
            attention: [Batch, Head, SeqLen, PatchSize]
            v: [Batch, Head, HeadDims, SeqLen]

            PatchSize: depends on the implementation
            
        Returns:
            [Batch, Head, HeadDims, SeqLen]
        """
        raise NotImplementedError("Please implement output_values")

    def positional_weights_and_mask(
        self, structure_size, mask_size, dtype, device
    ):
        """
        Computes the positional weights and mask.
        The positiona_weights need to be extracted from self.positional_mask module.
        The positional_mask needs to be a boolean mask to indicate which attention logits
        are relevant from the attention_logits of size [SeqLen, PatchSize]

        Returns:
            positional_weights: [1, 1, SeqLen or 1, PatchSize]
            positional_mask: [SeqLen, PatchSize]
        """
        raise NotImplementedError(
            "Please implement positional_weights_and_mask"
        )

    def merge_masks(self, mask, positional_mask, structure_size, mask_size):
        """
        Merges the mask with the positional mask.
        The implementation depends on the PatchSize and the layout of the logits.

        Args:
            mask: [SeqLen, SeqLen]
            positional_mask: [SeqLen, PatchSize]
        
        Returns:
            mask: [SeqLen, PatchSize]
        """
        raise NotImplementedError("Please implement merge_masks")

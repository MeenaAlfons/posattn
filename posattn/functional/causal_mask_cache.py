from .causal_mask import make_causal_mask


class CausalMaskCache:
    def __init__(self, module):
        self.module = module
        self.cache = {}

    def make(self, structure_size, device):
        # Assuming that the positional dimensions (D1, D2, ...) are flattened into QSeqLen
        # in a row-major order
        buffer_name = 'causal_mask_{}'.format(tuple(structure_size))
        if buffer_name not in self.cache:
            print("creating causal mask")
            causal_mask = make_causal_mask(structure_size, device)
            causal_mask.requires_grad = False
            self.module.register_buffer(buffer_name, causal_mask)
            self.cache[buffer_name] = causal_mask

        return self.cache[buffer_name]

from .position_meshgrid import make_position_meshgrid


class PositionMeshgridCache:
    def __init__(self, module=None):
        self.module = module
        self.cache = {}

    def set_module(self, module):
        self.module = module

    # This method is used by child blocks to get position meshgrid
    # which is registered as a buffer and reused across all blocks
    def make(
        self, structure_size, dtype, device, relative=True, normalized=True
    ):
        buffer_name = 'position_meshgrid_{}_{}_{}'.format(
            structure_size.tolist(), relative, normalized
        )
        if buffer_name not in self.cache:
            print("creating meshgrid: ", buffer_name)
            position_meshgrid = make_position_meshgrid(
                structure_size,
                dtype,
                device,
                relative=relative,
                normalized=normalized
            )
            position_meshgrid.requires_grad = False
            self.module.register_buffer(buffer_name, position_meshgrid)
            self.cache[buffer_name] = position_meshgrid

        return self.cache[buffer_name]

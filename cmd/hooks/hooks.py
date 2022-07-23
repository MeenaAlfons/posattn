from .visuals import visualize_relative_1d
import wandb
import re


class LogWhenNeededHook:
    @staticmethod
    def is_applicable(name, module):
        return False

    @staticmethod
    def is_forward_hook():
        return False

    @staticmethod
    def is_backward_hook():
        return False

    def __init__(self, hook, context):
        self.hook = hook
        self.context = context

    def __call__(self, module, inputs, output):
        if self.context['log_needed']:
            self.hook(module, inputs, output)


class MaskLogHook:
    @staticmethod
    def is_applicable(name, module):
        return name.endswith('positional_mask')

    @staticmethod
    def is_forward_hook():
        return True

    @staticmethod
    def is_backward_hook():
        return False

    def __init__(self, name, module, config):
        self.name = name
        regex = r"transformer.layers.(\d+)"
        match = re.match(regex, name)
        if match is not None:
            self.attention_layer = int(match.group(1))
        else:
            raise Exception(
                "Could not parse layer number from name {}".format(name)
            )

        self.positional_mask_threshold = config['positional_mask_threshold']

    def __call__(self, module, inputs, output):
        log = {}

        structure_size_same_device = inputs[0]

        structure_size = inputs[0].detach().cpu().flatten()
        for i, v in enumerate(structure_size):
            log[f'layer.{self.attention_layer}.structure_size.{i}'] = v.item()

        # This is the mask_size passed to the module to crop the gauss mask.
        # This may not be equal to the effective mask_size because the positional_attention
        # may need to get a bigger tensor for efficiency reasons.
        # requested_mask_size = inputs[1].detach().cpu()

        # flatten makes sure that mask_size is a 1D tensor (not 0-D)
        mask_size = module.get_mask_size(
            self.positional_mask_threshold, structure_size_same_device
        ).detach().cpu().flatten()
        for i, v in enumerate(mask_size):
            log[f'layer.{self.attention_layer}.mask_size.{i}'] = v.item()

        # flatten makes sure that sigma is a 1D tensor (not 0-D)
        sigma = module.sigma.detach().cpu().flatten()
        for i, v in enumerate(sigma):
            log[f'layer.{self.attention_layer}.sigma.{i}'] = v.item()

        if wandb.run:
            wandb.log(log, commit=False)


class PositionalEncodingHeatmapHook:
    @staticmethod
    def is_applicable(name, module):
        return name.endswith('positional_encoding')

    @staticmethod
    def is_forward_hook():
        return True

    @staticmethod
    def is_backward_hook():
        return False

    def __init__(self, name, module, config):
        self.name = name
        self.show = config["show_visuals"]

    def __call__(self, module, inputs, output):
        # tuple(D1, D2, ...)
        structure_size = inputs[0].detach().cpu().flatten()
        extract_size = inputs[1].detach().cpu().flatten()

        # [<extract_size>, ModelDim]
        relative_positional_encoding = output.detach().cpu()

        fig = None
        if len(structure_size) == 1:
            fig = visualize_relative_1d(
                relative_positional_encoding, extract_size, self.name
            )

        if wandb.run and fig:
            key = 'visual/pe/{}'.format(self.name)
            wandb.log({key: fig}, commit=False)

        if self.show and fig:
            fig.show()

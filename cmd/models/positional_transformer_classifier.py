import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
from experiment import TorchModel, Config
from posattn import Transformer, BlockArgs, EncoderArgs, ResolutionReductionArgs
from posattn import TransformerClassifier, ImplicitPositionalEncoding, BaselinePositionalEncoding, PositionMeshgridCache, GaussianPositionalMask, MultiheadAttention
from posattn import PositionalAttentionV1, PositionalAttentionV2, PositionalAttentionV3
import utils


def create_transformer_classifier(config):
    transformer = create_transformer(config)

    return TransformerClassifier(
        input_dim=config["input_dim"],
        model_dim=config["model_dim"],
        num_classes=config["num_classes"],
        transformer=transformer,
        dropout=config["dropout"],
        causal=config["causal"],
        num_positional_dims=config["pe"]["num_positional_dims"],
        cls_token=config["cls_token"],
    )


def create_transformer(config):
    meshgrid_cache = PositionMeshgridCache()

    # Build proper config for each layer
    block_args_list = []
    single_positional_encoding = None
    for layer_index in range(config["num_layers"]):
        model_dim = config["model_dim"]

        if config["same_positional_encoding"]:
            if single_positional_encoding is None:
                single_positional_encoding = make_positional_encoding(
                    meshgrid_cache, model_dim, config, layer_index
                )
            positional_encoding = single_positional_encoding
        else:
            positional_encoding = make_positional_encoding(
                meshgrid_cache, model_dim, config, layer_index
            )

        positional_mask = make_positional_mask(
            meshgrid_cache, config, layer_index
        )

        positional_attention_classes = {
            'v1': PositionalAttentionV1,
            'v2': PositionalAttentionV2,
            'v3': PositionalAttentionV3,
        }

        positional_attention = positional_attention_classes[
            config['positional_attention_version']
        ](
            model_dim=model_dim,
            num_heads=config["num_heads"],
            positional_encoding=positional_encoding,
            positional_mask=positional_mask,
            positional_mask_threshold=config["positional_mask_threshold"],
            positional_logits_operation=config["positional_logits_operation"],
            multiply_positional_mask_with=config["multiply_positional_mask_with"
                                                ],
        )

        self_attn = MultiheadAttention(
            model_dim=model_dim,
            num_heads=config["num_heads"],
            attention=positional_attention,
        )

        resolution_reduction_args = make_resolution_reduction_args(
            config, layer_index
        )

        block_args = BlockArgs(
            encoder_args=EncoderArgs(
                model_dim=model_dim,
                dim_feedforward=config["dim_feedforward"],
                dropout=config["dropout"],
                self_attn=self_attn,
            ),
            resolution_reduction_args=resolution_reduction_args,
        )
        block_args_list.append(block_args)

    transformer = Transformer(block_args_list)
    meshgrid_cache.set_module(transformer)
    return transformer


def make_resolution_reduction_args(config, layer_index):
    return ResolutionReductionArgs(
        num_positional_dims=config["pe"]["num_positional_dims"],
        enabled=config['resolution_reduction'][layer_index],
        kernel_size=config['resolution_reduction_kernel_size'],
    )


def make_positional_encoding(meshgrid_cache, model_dim, config, layer_index):
    if not config['include_position']:
        return None

    pe_base = config["pe"]
    pe = {k: v for k, v in pe_base.items()}

    if 'activation_params' in pe_base:
        activation_params = {
            k: v
            for k, v in pe_base['activation_params'].items()
            if not k.startswith('layer_')
        }

        layer_key = 'layer_{}'.format(layer_index)
        if layer_key in pe_base['activation_params']:
            if pe_base['activation_params'][layer_key]['w0'] > 0.0:
                activation_params['w0'] = pe_base['activation_params'][layer_key
                                                                      ]['w0']
            if pe_base['activation_params'][layer_key]['w0_initial'] > 0.0:
                activation_params['w0_initial'] = pe_base['activation_params'][
                    layer_key]['w0_initial']

        pe['activation_params'] = activation_params

    if config["positional_encoding"] == 'implicit':
        return ImplicitPositionalEncoding(
            meshgrid_cache=meshgrid_cache, output_dim=model_dim, **pe
        )
    elif config["positional_encoding"] == 'baseline':
        return BaselinePositionalEncoding(output_dim=model_dim, **pe)
    else:
        return None


def make_positional_mask(meshgrid_cache, config, layer_index):
    if not config["apply_positional_mask"]:
        return None

    sigma = config["sigma"]['default']
    layer_key = 'layer_{}'.format(layer_index)
    if layer_key in config["sigma"] and config["sigma"][layer_key] > 0.0:
        sigma = config["sigma"][layer_key]
    if sigma == 0.0:
        raise ValueError("Sigma cannot be 0.0")

    return GaussianPositionalMask(
        num_positional_dims=config["pe"]["num_positional_dims"],
        meshgrid_cache=meshgrid_cache,
        sigma=sigma,
        learn_sigma=config["learn_sigma"],
    )


class PositionalTransformerClassifierModel(TorchModel):
    def __init__(
        self, learning_rate, input_dim, scheduler, scheduler_params, classifier,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self._model = classifier
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)

        if scheduler != '':
            print(
                'Using scheduler {} with params {}'.format(
                    scheduler, scheduler_params
                )
            )
            self._scheduler = getattr(optim.lr_scheduler, scheduler)(
                self._optimizer, **scheduler_params
            )
        else:
            self._scheduler = None

    def name(self):
        return self._model.name()

    def model(self):
        return self._model

    def criterion(self):
        return self._criterion

    def optimizer(self):
        return self._optimizer

    def scheduler(self):
        return self._scheduler

    def dummy_input(self, batch_size):
        return torch.randn(batch_size, 32, 32, self.input_dim)


def modelFactory(config: Config):
    unflatten_config = utils.unflatten_dict(config)

    classifier = create_transformer_classifier(unflatten_config)

    return PositionalTransformerClassifierModel(
        device=config.device,
        learning_rate=config.learning_rate,
        input_dim=config.input_dim,
        classifier=classifier,
        accumulation_steps=config.accumulation_steps,
        scheduler=config.scheduler,
        scheduler_params=unflatten_config["scheduler_params"],
        profile=config.profile,
    )

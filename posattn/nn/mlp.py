import torch
import torch.nn as nn
import math


class MLP(nn.Module):
    """
    activation: callable(index) -> nn.Module
    final_activation: callable() -> nn.Module

    Examples of callable() -> nn.Module:
    - lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True)

    initializer: callable(weight, bias, layer_index, num_layers) -> None
    """
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        activation,
        bias=True,
        initializer=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers

        if activation is None:
            activation = lambda i, n: nn.Identity()

        if initializer is None:
            initializer = default_initializer

        layers = []
        for index in range(num_layers):
            is_first = index == 0
            is_last = index == num_layers - 1
            layer_dim_in = dim_in if is_first else dim_hidden
            layer_dim_out = dim_out if is_last else dim_hidden

            layer = nn.Linear(layer_dim_in, layer_dim_out, bias=bias)
            layers.append(layer)
            layers.append(activation(index, num_layers))

            initializer(layer.weight, layer.bias, index, num_layers)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def default_initializer(weight, bias, layer_index, num_layers):
    nn.init.kaiming_uniform_(weight, nonlinearity="linear")
    if bias is not None:
        bias.data.fill_(0.0)


def make_siren_initializer(w0=1., c=6.):
    def initializer(weight, bias, layer_index, num_layers):
        is_last = layer_index == num_layers - 1
        is_first = layer_index == 0

        # The input dimension is the second dimension of the weight matrix.
        dim_out, dim_in = weight.shape

        if not is_last:
            # I don't understand this initialization.
            w_std = (1. / dim_in) if is_first else (math.sqrt(c / dim_in) / w0)
            nn.init.uniform_(weight, -w_std, w_std)
            # weight.data.uniform_(-w_std, w_std)
        else:
            nn.init.kaiming_uniform_(weight, nonlinearity="linear")

        if bias is not None:
            nn.init.zeros_(bias)
            # bias.data.fill_(0.0)

    return initializer


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def get_activation_class(activation):
    if hasattr(nn, activation):
        return getattr(nn, activation)
    elif activation == 'Sine':
        return Sine
    else:
        raise ValueError(f'Unknown activation {activation}')


def make_activation(activation, activation_params, layer_index, num_layers):
    activation_class = get_activation_class(activation)
    params = {}

    if activation == 'Sine':
        params['w0'] = activation_params['w0']
        if layer_index == 0 and 'w0_initial' in activation_params and activation_params[
            'w0_initial'] is not None and activation_params['w0_initial'] > 0:
            params['w0'] = activation_params['w0_initial']

    return activation_class(**params)

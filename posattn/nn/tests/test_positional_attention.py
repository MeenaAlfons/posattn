from ..positional_attention_v1 import PositionalAttentionV1
from ..positional_attention_v2 import PositionalAttentionV2
from ..positional_attention_v3 import PositionalAttentionV3
from ..positional_encoding_mlp import ImplicitPositionalEncoding
from ..positional_encoding_baseline import BaselinePositionalEncoding
from ...functional import PositionMeshgridCache
from ...functional import make_causal_mask
from ..positional_mask import GaussianPositionalMask
import torch
import unittest
import sys
from pathlib import Path
import copy

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))
from utils import record

batch_size = 2
model_dim = 4
num_heads = 2
head_dim = model_dim // num_heads
learn_sigma = True
positional_mask_threshold = 0.1
positional_logits_operation = 'add'
pe = {
    'num_layers': 3,
    'hidden_dim': model_dim,
    'activation': 'Sine',
    'final_activation': 'Identity',
    'activation_params': {
        'w0': 10.0
    },
    'normalized': True,
}


def create_params(
    sigma, num_positional_dims, positional_encoding_type,
    multiply_positional_mask_with
):
    meshgrid_cache = PositionMeshgridCache()
    if positional_encoding_type == 'baseline':
        positional_encoding = BaselinePositionalEncoding(
            output_dim=model_dim,
            **pe,
        )
    elif positional_encoding_type == 'implicit':
        positional_encoding = ImplicitPositionalEncoding(
            meshgrid_cache=meshgrid_cache,
            output_dim=model_dim,
            **pe,
            num_positional_dims=num_positional_dims,
        )
    else:
        positional_encoding = None

    positional_mask = GaussianPositionalMask(
        num_positional_dims=num_positional_dims,
        meshgrid_cache=meshgrid_cache,
        sigma=sigma,
        learn_sigma=learn_sigma,
    )
    meshgrid_cache.set_module(positional_mask)
    return {
        'model_dim': model_dim,
        'num_heads': num_heads,
        'positional_encoding': positional_encoding,
        'positional_mask': positional_mask,
        'positional_mask_threshold': positional_mask_threshold,
        'positional_logits_operation': positional_logits_operation,
        'multiply_positional_mask_with': multiply_positional_mask_with,
    }


class TestPositionalAttention(unittest.TestCase):
    def test_all(self):
        record.enabled = True
        model_classes = [
            PositionalAttentionV1,
            PositionalAttentionV2,
            PositionalAttentionV3,
        ]
        tests = [
            {
                'sigma': 0.075,
                'structure_size': torch.tensor([5], dtype=torch.int64),
                'positional_encoding': 'baseline',
                'multiply_positional_mask_with': 'all_logits',
            },
            {
                'sigma': 0.075,
                'structure_size': torch.tensor([5], dtype=torch.int64),
                'positional_encoding': 'implicit',
                'multiply_positional_mask_with': 'all_logits',
            },
            {
                'sigma': 0.075,
                'structure_size': torch.tensor([5, 5], dtype=torch.int64),
                'positional_encoding': 'implicit',
                'multiply_positional_mask_with': 'all_logits',
            },
            {
                'sigma': 0.075,
                'structure_size': torch.tensor([5], dtype=torch.int64),
                'positional_encoding': 'baseline',
                'multiply_positional_mask_with': 'positional_logits',
            },
            {
                'sigma': 0.075,
                'structure_size': torch.tensor([5], dtype=torch.int64),
                'positional_encoding': 'implicit',
                'multiply_positional_mask_with': 'positional_logits',
            },
            {
                'sigma': 0.075,
                'structure_size': torch.tensor([5, 5], dtype=torch.int64),
                'positional_encoding': 'implicit',
                'multiply_positional_mask_with': 'positional_logits',
            },
        ]
        tests_with_mask = []
        for test in tests:
            test['mask'] = None
            if len(test['structure_size']) == 1:
                test_with_mask = copy.deepcopy(test)
                test_with_mask['mask'] = make_causal_mask(
                    test['structure_size'], device='cpu'
                )
                tests_with_mask.append(test_with_mask)
        tests += tests_with_mask

        for test in tests:
            print('test:', test)
            num_positional_dims = len(test['structure_size'])
            seq_len = torch.prod(test['structure_size'])
            q = torch.rand(batch_size, num_heads, head_dim, seq_len)
            k = torch.rand(batch_size, num_heads, head_dim, seq_len)
            v = torch.rand(batch_size, num_heads, head_dim, seq_len)
            models = []
            params = create_params(
                test['sigma'], num_positional_dims, test['positional_encoding'],
                test['multiply_positional_mask_with']
            )
            for model_class in model_classes:
                this_params = copy.deepcopy(params)
                model = model_class(**this_params)
                model.u.data.fill_(1.0)
                model.v.data.fill_(1.0)
                models.append(model)

            outputs = []
            for model in models:
                print('Excuting {}'.format(model.__class__.__name__))
                output = model(q, k, v, test['structure_size'], test['mask'])
                outputs.append(output)
                record('output', output)
                output.backward(torch.ones(output.shape))
                print('output: {}'.format(output.shape))
                for name, parameter in model.named_parameters():
                    record(name, parameter)
                    record(name + '_grad', parameter.grad)

                print('Recorded:')
                print(record.items)
                record.clear()

            for i in range(1, len(outputs)):
                with self.subTest(
                    'Output of {} vs {}'.format(
                        model_classes[0].__name__, model_classes[i].__name__
                    )
                ):
                    self.assertTrue(
                        torch.allclose(outputs[0], outputs[i]),
                        msg="\n{}\n!=\n{}\n".format(outputs[0], outputs[i])
                    )

            for i in range(1, len(models)):
                with self.subTest(
                    'Grads of {} vs {}'.format(
                        model_classes[0].__name__, model_classes[i].__name__
                    )
                ):
                    named_parameters_0 = {
                        k: v
                        for k, v in models[0].named_parameters()
                    }
                    named_parameters_i = {
                        k: v
                        for k, v in models[i].named_parameters()
                    }
                    for name in named_parameters_0.keys():
                        if named_parameters_0[
                            name].grad is not None and named_parameters_i[
                                name].grad is not None:
                            self.assertTrue(
                                torch.allclose(
                                    named_parameters_0[name].grad,
                                    named_parameters_i[name].grad,
                                    atol=1e-5
                                ),
                                msg="\n{}\n{}\n!=\n{}\n".format(
                                    name, named_parameters_0[name].grad,
                                    named_parameters_i[name].grad
                                )
                            )

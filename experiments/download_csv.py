import pandas as pd
import wandb
from tqdm import tqdm
from os import environ

config_keys = [
    'model',
    'dataset',
    'branch',
    "batch_size",
    'causal',
    'commit',
    'device',
    'branch',
    'dropout',
    'gpu_name',
    'cls_token',
    'cpu_count',
    'model_dim',
    'input_dim',
    'num_heads',
    'num_layers',
    'params_count',
    'flops_count',
    'flops_count_10x',
    'random_state',
    'learning_rate',
    'num_epochs',
    'pe.activation',
    'pe.hidden_dim',
    'pe.normalized',
    'pe.num_layers',
    'sigma.default',
    'sigma.layer_0',
    'sigma.layer_1',
    'sigma.layer_2',
    'sigma.layer_3',
    'sigma.layer_4',
    'sigma.layer_5',
    'small_dataset',
    'dim_feedforward',
    'include_position',
    'accumulation_steps',
    'pe.final_activation',
    'positional_encoding',
    'apply_positional_mask',
    'pe.num_positional_dims',
    'pe.activation_params.w0',
    'same_positional_encoding',
    'positional_mask_threshold',
    'positional_logits_operation',
    'positional_attention_version',
    'multiply_positional_mask_with',
    'pe.activation_params.w0_initial',
    'gpu_count',
    'learn_sigma',
    'channels_last',
    'num_classes',
]

summary_keys = [
    ["val/acc", "max"],
    'layer.0.mask_size.0',
    'layer.1.mask_size.0',
    'layer.2.mask_size.0',
    'layer.3.mask_size.0',
    'layer.4.mask_size.0',
    'layer.5.mask_size.0',
    'layer.0.mask_size.1',
    'layer.1.mask_size.1',
    'layer.2.mask_size.1',
    'layer.3.mask_size.1',
    'layer.4.mask_size.1',
    'layer.5.mask_size.1',
    'layer.0.sigma.0',
    'layer.1.sigma.0',
    'layer.2.sigma.0',
    'layer.3.sigma.0',
    'layer.4.sigma.0',
    'layer.5.sigma.0',
    'layer.0.sigma.1',
    'layer.1.sigma.1',
    'layer.2.sigma.1',
    'layer.3.sigma.1',
    'layer.4.sigma.1',
    'layer.5.sigma.1',
    '_timestamp',
    '_runtime',
    'epoch',
    ['train/loss', 'min'],
    ['per_sample_time', 'mean'],
    ['val/loss', 'min'],
    ['val/apmp', 'min'],
]


def extract_nested_key(obj, nested_key):
    if type(nested_key) is str:
        nested_key = [nested_key]

    for key in nested_key:
        if key in obj:
            obj = obj[key]
        else:
            return None
    return obj


def key_from_nested_key(nested_key):
    if type(nested_key) is str:
        return nested_key
    return ".".join(nested_key)


def extract(obj, nested_keys):
    results = {}
    for nested_key in nested_keys:
        key = key_from_nested_key(nested_key)
        results[key] = extract_nested_key(obj, nested_key)
    return results


def parse_run(run, sweep_cache):
    parsed = {
        'createdAt': run.createdAt,
        'name': run.name,
        'id': run.id,
        'state': run.state,
        'sweep_id': None,
        'sweep_name': None,
    }
    if run.sweep is not None:
        parsed['sweep_id'] = run.sweep.id
        parsed['sweep_name'] = run.sweep.name

    parsed.update(extract(run.config, config_keys))
    parsed.update(extract(run.summary._json_dict, summary_keys))
    return parsed


def dataframe_from_wandb_runs(runs):
    data = {}
    for run in runs:
        for k, v in run.items():
            data[k] = data.get(k, []) + [v]
    df = pd.DataFrame(data)
    return df


class SweepCache():
    def __init__(self, entity, project):
        self.api = wandb.Api({"entity": entity, "project": project})

    def get_sweep(self, sweep_id):
        if sweep_id not in self._sweeps:
            self._sweeps[sweep_id] = self.api.sweep(sweep_id)
        return self._sweeps[sweep_id]


def read_wandb_runs(entity, project, after=None):
    api = wandb.Api()
    filters = None
    if after:
        filters = {"$and": [{'created_at': {"$gt": after}}]}
    runs = api.runs(entity + "/" + project, filters)
    sweep_cache = SweepCache(entity, project)
    for run in tqdm(runs):
        yield parse_run(run, sweep_cache)


if __name__ == "__main__":
    if environ.get('WANDB_PROJECT') is None:
        environ['WANDB_PROJECT'] = 'posattn'

    if environ.get('WANDB_ENTITY') is None:
        environ['WANDB_ENTITY'] = 'meenaalfons-team'

    runs = read_wandb_runs(
        environ['WANDB_ENTITY'], environ['WANDB_PROJECT'], '2022-05-25T##'
    )
    df = dataframe_from_wandb_runs(runs)
    rename_columns = {
        'val/acc.max': 'val_acc_max',
        'pe.num_positional_dims': 'pe_num_positional_dims'
    }
    df.rename(columns=rename_columns, inplace=True)
    print(df.iloc[0].to_string())
    print(df.columns)
    df.to_csv("data/runs.csv")

import yaml
import utils
import argparse
import time
import random
import flatdict
import torch
import os
import subprocess

from experiment import Config
from os import environ


def get_args_and_config():
    args, config = parse_args()
    set_env_vars(args)

    with open('config.yaml') as f:
        defaults = yaml.load(f, Loader=yaml.FullLoader)
        flatDefaults = flatdict.FlatDict(defaults, delimiter='.')
        utils.ensure_config(config, flatDefaults)
        ensure_config(config)

    return args, config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-mode',
        '--mode',
        type=str,
        default='train',
        help='input mode either train or test (default: train)'
    )
    parser.add_argument(
        '-no_wandb',
        '--no_wandb',
        type=bool,
        default=False,
        help='No wandb (default: False)'
    )
    parser.add_argument(
        '-debug',
        '--debug',
        type=bool,
        default=False,
        help='Enable debug output (default: False)'
    )

    # run_id and resume are mostly not used
    parser.add_argument(
        '-rid',
        '--run_id',
        type=str,
        default='',
        help='input run id to resume (default: )'
    )
    parser.add_argument(
        '-r',
        '--resume',
        type=str,
        default='allow',
        help='Resume (default: allow)'
    )

    # Sweep parameters
    parser.add_argument(
        '-sid',
        '--sweep_id',
        type=str,
        default='',
        help='sweep id to run an agent (default: )'
    )
    # This count is mostly not used because we usually run one agent per machine.
    parser.add_argument(
        '-count',
        '--count',
        type=int,
        default=1,
        help='Number of trials to run (default: 1)'
    )

    # Any further arguments will be passed to the config
    # Nested config are provided in dot notation
    parser.add_argument('-D', action='append', default=[])

    args = parser.parse_args()
    config = {L[0]: L[1] for L in [s.split('=') for s in args.D]}
    # config has dot notation and will not be converted to dict
    # because sweeps don't support nested configs
    # config = unflatten_dict(config)

    config = parse_special_values(config)
    config = Config(**config)

    # Validate arguments
    if args.mode not in ['train', 'test', 'visual', 'sweep', 'sweep_agent']:
        raise ValueError(f'Invalid mode: {args.mode}')

    if args.mode == 'sweep_agent' and args.sweep_id == '':
        raise ValueError(f'Invalid sweep id: {args.sweep_id}')

    # If this is a sweep_agent, disable resume
    if args.mode == 'sweep_agent':
        args.resume = ''

    return args, config


def set_env_vars(args):
    if environ.get('WANDB_PROJECT') is None:
        environ['WANDB_PROJECT'] = 'posattn'

    if environ.get('WANDB_ENTITY') is None:
        environ['WANDB_ENTITY'] = 'meenaalfons-team'

    if args.run_id != '':
        environ['WANDB_RUN_ID'] = args.run_id

    if args.resume != '':
        environ['WANDB_RESUME'] = args.resume


def parse_special_values(config):
    result = {}
    for key, value in config.items():
        if isinstance(value, str):
            if value.lower() in ['t', 'true']:
                result[key] = True
            if value.lower() in ['f', 'false']:
                result[key] = False
            elif value.lower() in ['none']:
                result[key] = None
            else:
                result[key] = value
        else:
            result[key] = value
    return result


def ensure_config(config):
    if config['random_state'] is None:
        random.seed(time.time())
        config.random_state = random.randint(0, 2**32)
        config.update(
            {'random_state': int(config.random_state)}, allow_val_change=True
        )

    if 'device' not in config:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        config.device = device

    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(device)
        config.gpu_name = gpu_name

    config.gpu_count = torch.cuda.device_count()
    config.cpu_count = os.cpu_count()

    if 'commit' not in config:
        config.commit = get_git_revision_short_hash()

    if 'branch' not in config:
        config.branch = get_git_current_branch()


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short',
                                    'HEAD']).decode('ascii').strip()


def get_git_current_branch() -> str:
    # Can simply use git branch --show-current
    # but --show-current was introduced in Git 2.22.0
    return subprocess.check_output(['git', 'symbolic-ref', '--short',
                                    'HEAD']).decode('ascii').strip()

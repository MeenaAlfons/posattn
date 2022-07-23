import torch
import wandb
import sys
from pathlib import Path

from experiment import WandbExperiment, BaseExperiment
from models import models
from datasets import datasets

sys.path.append(str(Path(__file__).parent.absolute()))
from config import get_args_and_config

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils import measure


def main():
    args, config = get_args_and_config()

    if args.debug == True:
        measure.enabled = True
        torch.autograd.set_detect_anomaly(True)

    modelFactory = models[config.model].modelFactory
    dataFactory = datasets[config.dataset].dataFactory

    experiment = BaseExperiment(
        config, modelFactory, dataFactory
    ) if args.no_wandb else WandbExperiment(config, modelFactory, dataFactory)

    mode_func = {
        'train': lambda: experiment.train(),
        'test': lambda: experiment.test(),
        'visual': lambda: experiment.visual(),
    }

    if args.mode in ['train', 'test', 'visual']:
        mode_func[args.mode]()

    elif args.mode == 'sweep_agent':
        wandb.agent(args.sweep_id, mode_func['train'], count=args.count)


if __name__ == '__main__':
    main()

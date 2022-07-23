from atexit import register
from typing import Callable
import time
import pprint
from .visuals import plot_data_with_pe, plot_grid_data_with_pe
import torch
import numpy
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from hooks import register_hooks, Context


class Config():
    def __init__(self, **kwargs):
        object.__setattr__(self, "__dict__", dict())
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __setitem__(self, key, val):
        self.__dict__[key] = val

    __setattr__ = __setitem__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __contains__(self, key):
        return key in self.__dict__

    def __iter__(self):
        for key in self.__dict__:
            yield (key, self.__dict__[key])

    def keys(self):
        return [k for k in self.__dict__.keys() if not k.startswith("_")]

    def items(self):
        return [
            (k, v) for k, v in self.__dict__.items() if not k.startswith("_")
        ]

    def update(self, d, allow_val_change=None):
        self.__dict__.update(d)

    def __str__(self):
        return pprint.pformat(self.__dict__)


class Data:
    def train_val(self):
        pass

    def train_test(self):
        pass


class Model:
    def train_epoch(self, dataloader):
        return {
            'loss': 0.1,
        }

    def evaluate(self, dataloader):
        return {
            'loss': 0.1,
            'acc': 0.2,
        }


DataFactory = Callable[[Config], Data]
ModelFactory = Callable[[Config], Data]


def seed_global(random_state):
    torch.manual_seed(random_state)
    numpy.random.seed(random_state)
    random.seed(random_state)


def modelFactoryWrapper(modelFactory):
    def wrapper(config: Config):
        model = modelFactory(config, )
        config.params_count = model.params_count()
        config.flops_count = model.flops_count(batch_size=1)
        config.flops_count_10x = model.flops_count(batch_size=10)
        config.model_name = model.name()
        return model

    return wrapper


class BaseExperiment:
    def __init__(
        self, config: Config, modelFactory: ModelFactory,
        dataFactory: DataFactory
    ):
        self.config = config
        self._modelFactory = modelFactory
        self._dataFactory = dataFactory

    def dataFactory(self):
        return self._dataFactory

    def modelFactory(self):
        return modelFactoryWrapper(self._modelFactory)

    def resume(self, model: Model) -> int:
        return 0

    def logMetrics(self, log: dict):
        print(log)

    def saveCheckpoint(self, model: Model, epoch: int):
        pass

    def saveModel(self, model: Model):
        pass

    def loadModel(self, model: Model):
        pass

    def train(self):
        seed_global(self.config.random_state)
        print('Training started with config:', self.config)
        data = self.dataFactory()(self.config)
        model = self.modelFactory()(self.config)
        context = Context()
        register_hooks(model.model(), self.config.hooks, self.config, context)

        params_count = model.params_count()
        print('Params count:', params_count)

        initial_epoch = self.resume(model)

        trainloader, valloader = data.train_val()
        num_samples = len(trainloader.dataset)
        num_batches = len(trainloader)
        log_batch_every = max(1, num_batches // self.config.log_times_per_epoch)
        for epoch in range(initial_epoch, self.config.num_epochs):
            start_time = time.time()

            def beforeBatch(metrics):
                if metrics['batch'] % log_batch_every == 0:
                    context['log_needed'] = True

            def afterBatch(metrics):
                if metrics['batch'] % log_batch_every == 0:
                    self.logMetrics(
                        {
                            'epoch': epoch,
                            'batch': metrics['batch'],
                            'running_loss': metrics['loss'],
                        }
                    )
                    context['log_needed'] = False

            train_results = model.train_epoch(
                trainloader, beforeBatch, afterBatch
            )
            val_results = model.evaluate(valloader)
            end_time = time.time()

            log = {
                'epoch': epoch + 1,
                'train/loss': train_results['loss'],
                'val/loss': val_results['loss'],
                'val/acc': val_results['acc'],
                'val/apmp': 1000000 * val_results['acc'] / params_count,
                'lr': model.optimizer().param_groups[0]['lr'],
                'per_sample_time': (end_time - start_time) / num_samples,
            }
            self.logMetrics(log)

            self.saveCheckpoint(model, epoch)

            model.run_scheduler(epoch, val_results['loss'])

        self.saveModel(model)

    def test(self):
        seed_global(self.config.random_state)
        print('Testing started with config:', self.config.__dict__)
        data = self.dataFactory()(self.config)
        model = self.modelFactory()(self.config)
        self.loadModel(model)

        _, testloader = data.train_test()
        num_samples = len(testloader)

        start_time = time.time()
        test_results = model.evaluate(testloader)
        end_time = time.time()

        log = {
            'epoch': 1,
            'test/loss': test_results['loss'],
            'test/acc': test_results['acc'],
            'per_sample_time': (end_time - start_time) / num_samples,
        }
        self.logMetrics(log)

    def visual(self):
        data = self.dataFactory(self.config)
        model = self.modelFactory(self.config)
        self.loadModel(self.config, model)

        trainloader, valloader = data.train_val()
        inputs, labels = next(iter(trainloader))

        model.model().eval()
        _, l1, pe = model.model()(inputs, with_l1=True, with_pe=True)
        l1 = l1.detach().numpy()
        pe = pe.detach().numpy()

        # Choose the middle 10 features ti display
        if pe.shape[2] > 10:
            middle = int(pe.shape[2] / 2)
            start = middle - 5
            end = middle + 5
            plot_data_with_pe(l1[:, :, start:end], pe[:, :, start:end])
        else:
            plot_data_with_pe(l1, pe)

        # Display all feature as a heatmap
        plot_grid_data_with_pe(l1, pe)

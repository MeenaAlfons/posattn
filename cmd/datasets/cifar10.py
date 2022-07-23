import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from experiment import Data, Config


def move_channels_last(x):
    return x.permute(1, 2, 0)


class CIFARData(Data):
    def __init__(self, batch_size, random_state, channels_last, small=False):
        self._batch_size = batch_size
        self._random_state = random_state
        self._small = small

        layers = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        if channels_last:
            layers.append(transforms.Lambda(move_channels_last))

        self._transform = transforms.Compose(layers)

    def train_val(self):
        rng = np.random.default_rng(self._random_state)
        randInt = lambda: rng.integers(1, 1e9).item()

        traindataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self._transform
        )

        if self._small:
            small_size = min(1000, 0.1 * len(traindataset))
            rest_size = len(traindataset) - small_size
            traindataset, restset = torch.utils.data.random_split(
                traindataset, [small_size, rest_size],
                generator=torch.Generator().manual_seed(randInt())
            )

        train_size = int(len(traindataset) * 0.8)
        val_size = len(traindataset) - train_size

        trainset, valset = torch.utils.data.random_split(
            traindataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(randInt())
        )

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=2,
            generator=torch.Generator().manual_seed(randInt())
        )

        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=2,
            generator=torch.Generator().manual_seed(randInt())
        )
        return trainloader, valloader

    def train_test(self):
        rng = np.random.default_rng(self._random_state)
        randInt = lambda: rng.integers(1, 1e9).item()

        traindataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self._transform
        )
        trainloader = torch.utils.data.DataLoader(
            traindataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=2,
            generator=torch.Generator().manual_seed(randInt())
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=self._transform
        )

        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=2,
            generator=torch.Generator().manual_seed(randInt())
        )
        return trainloader, testloader


def dataFactory(config: Config):
    return CIFARData(
        config.batch_size, config.random_state, config.channels_last,
        config.small_dataset
    )


if __name__ == "__main__":
    data = dataFactory(Config(batch_size=32, random_state=42))
    trainloader, valloader = data.train_val()

    for i, (inputs, labels) in enumerate(trainloader):
        print(inputs.shape, labels.shape)
        if i == 2:
            break

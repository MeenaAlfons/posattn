import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from experiment import Data, Config


def reshape(x):
    after = x.view(28, 28, 1)
    return after


class MNISTData(Data):
    def __init__(self, batch_size, random_state, small=False):
        self._batch_size = batch_size
        self._random_state = random_state
        self._small = small

        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, )),
                transforms.Lambda(reshape)
            ]
        )

    def train_val(self):
        rng = np.random.default_rng(self._random_state)
        randInt = lambda: rng.integers(1, 1e9).item()

        traindataset = torchvision.datasets.MNIST(
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
        print('Train size:', train_size)
        print('Val size:', val_size)
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

        traindataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=self._transform
        )
        trainloader = torch.utils.data.DataLoader(
            traindataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=2,
            generator=torch.Generator().manual_seed(randInt())
        )
        testset = torchvision.datasets.MNIST(
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
    return MNISTData(
        config.batch_size, config.random_state, config.small_dataset
    )


if __name__ == "__main__":
    data = dataFactory(Config(batch_size=32, random_state=42))
    trainloader, valloader = data.train_val()

    for i, (inputs, labels) in enumerate(trainloader):
        print(
            'inputs.shape {}, labels.shape {}'.format(
                inputs.shape, labels.shape
            )
        )
        plt.hist(inputs.numpy().reshape(-1), density=True)
        plt.show()
        plt.imshow(
            inputs[0], cmap='gray', vmin=-1, vmax=1, interpolation='none'
        )
        plt.show()
        plt.figure(figsize=(15, 1))
        plt.axes([0.05, 0, 0.9, 1])
        plt.imshow(
            inputs[0].reshape(1, -1),
            cmap='gray',
            vmin=-1,
            vmax=1,
            interpolation='none'
        )
        plt.axis('off')
        plt.show()
        if i == 0:
            break

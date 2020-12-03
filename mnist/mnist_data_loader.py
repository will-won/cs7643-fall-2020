from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST
import unittest


class MnistDataLoader:
    """
    MNIST dataloader
    """

    def __init__(self, train_batch_size: int, validation_batch_size: int, test_batch_size: int):
        """
        DataLoader initializer
        creates dataloader.train, dataloader.validation, dataloader.test

        :param train_batch_size:
        :param validation_batch_size:
        :param test_batch_size:
        """
        transform = Compose([ToTensor(),
                             Normalize(
                                 (0.1307,), (0.3081,))
                             ])
        self.train = DataLoader(
            dataset=MNIST('./downloaded_data/mnist', train=True, download=True, transform=transform),
            batch_size=train_batch_size,
            shuffle=True
        )

        self.validation = iter(
            DataLoader(
                dataset=MNIST('./downloaded_data/mnist', train=False, download=True, transform=transform),
                batch_size=validation_batch_size,
                shuffle=True
            )
        )

        self.test = DataLoader(
            dataset=MNIST('./downloaded_data/mnist', train=False, download=True, transform=transform),
            batch_size=test_batch_size,
            shuffle=True
        )


class TestMnistDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        self.mnist_data_loader = MnistDataLoader(100, 15, 10)

    def test_train(self) -> None:
        data, label = next(iter(self.mnist_data_loader.train))
        self.assertEqual(label.size()[0], 100)

    def test_validattion(self) -> None:
        data, label = next(self.mnist_data_loader.validation)
        self.assertEqual(label.size()[0], 15)

    def test_test(self) -> None:
        data, label = next(iter(self.mnist_data_loader.test))
        self.assertEqual(label.size()[0], 10)

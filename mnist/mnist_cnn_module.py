import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest


class MnistCnnModule(nn.Module):
    """
    MNIST dataset: CNN network
    """
    
    def __init__(self):
        super(MnistCnnModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2)

        self.linear_size = 7 * 7 * 64
        self.fc1 = nn.Linear(in_features=self.linear_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.linear_size)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TestMnistCnnModule(unittest.TestCase):
    def setUp(self) -> None:
        self.cnn_module = MnistCnnModule()

    def test_forward(self) -> None:
        """
        Only tests whether network runs without any glitch
        """
        data = torch.zeros([128, 1, 28, 28], dtype=torch.float)
        out = self.cnn_module(data)
        print(out)

        self.assertEqual(out.size()[1], 10)

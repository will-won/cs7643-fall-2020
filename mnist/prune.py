import torch
import sys
from mnist_cnn_module import MnistCnnModule
from mnist_data_loader import MnistDataLoader
from trainer import Trainer
from threshold_pruning_method import ThresholdPruningMethod
from torch.nn.utils import prune


def main():
    prune_threshold = float(sys.argv[1])
    
    mnist_data_loader = MnistDataLoader(train_batch_size=1000,
                                        validation_batch_size=100,
                                        test_batch_size=1000)
    cnn_module = MnistCnnModule()
    cnn_module.load_state_dict(torch.load('./trained_model/saved.pt'))

    trainer = Trainer(module=cnn_module, data_loader=mnist_data_loader, lr=1e-5)
    trainer.test()

    # do pruning
    parameters_to_prune = [(cnn_module.conv1, 'weight'),
                           (cnn_module.conv2, 'weight'),
                           (cnn_module.conv3, 'weight'),
                           (cnn_module.fc1, 'weight'),
                           (cnn_module.fc2, 'weight')]
    prune.global_unstructured(parameters_to_prune, pruning_method=ThresholdPruningMethod, threshold=prune_threshold)
    print("After pruning:")
    trainer.test()


if __name__ == '__main__':
    main()

import torch
import sys
from mnist_cnn_module import MnistCnnModule
from mnist_data_loader import MnistDataLoader
from trainer import Trainer
from threshold_pruning_method import ThresholdPruningMethod
from torch.nn.utils import prune


def do_prune(cnn_module, threshold):
    # do pruning
    parameters_to_prune = [(cnn_module.conv1, 'weight'),
                           (cnn_module.conv2, 'weight'),
                           (cnn_module.conv3, 'weight'),
                           (cnn_module.fc1, 'weight'),
                           (cnn_module.fc2, 'weight')]
    prune.global_unstructured(parameters_to_prune, pruning_method=ThresholdPruningMethod, threshold=threshold)

    for child in cnn_module.children():
        prune.remove(child, 'weight')


def main():
    prune_threshold = float(sys.argv[1])

    mnist_data_loader = MnistDataLoader(train_batch_size=1000,
                                        validation_batch_size=100,
                                        test_batch_size=1000)
    cnn_module = MnistCnnModule()
    trainer = Trainer(module=cnn_module, data_loader=mnist_data_loader, lr=1e-5)
    cnn_module.load_state_dict(torch.load('./trained_model/saved.pt'))

    # print("Before pruning:")
    # before_correct, _ = trainer.test()

    print(f"Using threshold: {prune_threshold}")
    do_prune(cnn_module, threshold=prune_threshold)

    # print("After pruning:")
    # prune_correct, _ = trainer.test()

    # trainer.lr = 1e-6
    # for finetune_step in range(5):
    #     trainer.train(epoch=4, validation_step=10000)
    #     do_prune(cnn_module, threshold=prune_threshold)
    
    # print("After finetuning:")
    # finetune_correct, _ = trainer.test()

    # print(f"{prune_threshold},{before_correct},{prune_correct},{finetune_correct}")

    prune_count = 0
    total_count = 0
    for name, w in cnn_module.named_parameters():
        if "weight" in name:
            total_count += w.numel()
            prune_count += w.nonzero().size(0)
    print(f"{prune_threshold},{total_count},{prune_count}")


if __name__ == '__main__':
    main()

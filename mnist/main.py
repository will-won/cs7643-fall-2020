import os
from mnist_cnn_module import MnistCnnModule
from trainer import Trainer
from mnist_data_loader import MnistDataLoader


def main():
    if not os.path.exists('./trained_model'):
        os.makedirs('./trained_model')

    mnist_data_loader = MnistDataLoader(train_batch_size=1000,
                                        validation_batch_size=100,
                                        test_batch_size=1000)
    cnn_module = MnistCnnModule()
    trainer = Trainer(module=cnn_module, data_loader=mnist_data_loader, lr=1e-5)

    trainer.train(epoch=5, validation_step=10)


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from mnist_data_loader import MnistDataLoader


class Trainer:
    def __init__(self, module: nn.Module, data_loader: MnistDataLoader, lr: float = 1e-4):
        self.module = module
        self.data_loader = data_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.module.parameters(), lr=lr)

        # run train on gpu, if possible
        if torch.cuda.is_available():
            print("CUDA available. Using GPU.")
            self.device = 'cuda:0'
        else:
            print("CUDA unavailable. Using CPU.")
            self.device = 'cpu'

        self.module = self.module.to(self.device)
        self.module.train()

    def train_step(self, data, label):
        self.optimizer.zero_grad()

        out = self.module(data)
        loss = self.criterion(out, label)
        loss.backward()

        self.optimizer.step()

    def validate(self, data, label, epoch: int, batch_index: int):
        self.module.eval()

        # compute loss
        out = self.module(data)
        loss = self.criterion(out, label)

        # compute accuracy
        prediction = out.data.max(1, keepdim=True)[1]
        corrects_count = prediction.eq(label.data.view_as(prediction)).cpu().sum()
        batch_size = prediction.size()[0]
        accuracy = corrects_count.item() / batch_size

        print(f"Validation (epoch: {epoch}, batch: {batch_index}): Loss: {loss}, Accuracy: {accuracy * 100}% ({corrects_count} / {batch_size})")

        # back to training
        self.module.train()

    def test(self):
        self.module.eval()

        # test
        batch_size = 0
        corrects_count = 0
        for _, (test_data, test_label) in enumerate(self.data_loader.test):
            test_data = test_data.to(self.device)
            test_label = test_label.to(self.device)

            out = self.module(test_data)
            loss = self.criterion(out, test_label)

            # compute accuracy
            prediction = out.data.max(1, keepdim=True)[1]
            corrects_count += prediction.eq(test_label.data.view_as(prediction)).cpu().sum().item()
            batch_size += prediction.size()[0]

        accuracy = corrects_count / batch_size
        print(f"Test Accuracy: {accuracy * 100}% ({corrects_count} / {batch_size})")

        # back to training
        self.module.train()

    def save(self):
        torch.save(self.module.state_dict(), f='./trained_model/saved.pt')

    def train(self, epoch: int = 1, validation_step: int = 100) -> None:
        for e in range(epoch):
            for batch_index, (train_data, train_label) in enumerate(self.data_loader.train):
                train_data = train_data.to(self.device)
                train_label = train_label.to(self.device)

                self.train_step(train_data, train_label)

                if batch_index % validation_step == 0:
                    validation_data, validation_label = next(self.data_loader.validation)

                    validation_data = validation_data.to(self.device)
                    validation_label = validation_label.to(self.device)

                    self.validate(validation_data, validation_label, epoch=e, batch_index=batch_index)

            # after each epoch, save
            self.save()

        # training done: do test
        print("Training done. Testing:")
        self.test()

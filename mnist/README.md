# mnist/
MNIST-CNN implementation in PyTorch

## How to run
1. Run `main.py` to automatically download MNIST dataset and train the model.
```
python3 main.py
```
You can change the number of epochs and the learning rate in `main.py`.

After each epoch is done, it automatically saves the model into `./trained_model/saved.pt` file.


2. Run `prune.py` to load pretrained `saved.pt` model and run the static-threshold-based pruning. You should pass the pruning threshold.
```
python3 prune.py 1e-3
```

3. Run `finetune.py` to load pretrained `saved.pt` model and run the static-threshold-based pruning, followed by post fine-tuning. You should pass the pruning threshold.
```
python3 finetune.py 1e-3
```
You can update the fine-tuning learning rate in the `finetune.py` file.

## Expected Output
1. `main.py`: You can see the training steps and test accuracy.
```
CUDA available. Using GPU.
Validation (epoch: 0, batch: 0): Loss: 2.3034071922302246, Accuracy: 9.0% (9 / 100)
Validation (epoch: 0, batch: 10): Loss: 2.2907118797302246, Accuracy: 21.0% (21 / 100)
```

2. `prune.py`: You can see the accuracy differences after static pruning.
```
CUDA available. Using GPU.
Test Accuracy: 94.66% (9466 / 10000)
After pruning:
Test Accuracy: 9.8% (980 / 10000)
```

3. `finetune.py`: You can see the accuracy differences after static pruning and post fine-tuning.
```
CUDA available. Using GPU.
Before pruning:
Test Accuracy: 94.66% (9466 / 10000)
Using threshold: 0.5
After pruning:
Test Accuracy: 9.8% (980 / 10000)
Validation (epoch: 0, batch: 0): Loss: 2.305284023284912, Accuracy: 6.0% (6 / 100)
Validation (epoch: 1, batch: 0): Loss: 2.304542303085327, Accuracy: 9.0% (9 / 100)
Validation (epoch: 2, batch: 0): Loss: 2.3011891841888428, Accuracy: 11.0% (11 / 100)
Validation (epoch: 3, batch: 0): Loss: 2.3012120723724365, Accuracy: 14.000000000000002% (14 / 100)
Training done. Testing:
Test Accuracy: 11.35% (1135 / 10000)
After finetuning:
Test Accuracy: 9.8% (980 / 10000)
```

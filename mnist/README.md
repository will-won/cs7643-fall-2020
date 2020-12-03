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

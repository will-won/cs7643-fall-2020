# cross-val/
Files for ResNet pruning cross-validation

## How to run
1. Train the model
```
python train.py --model='resnet9' --checkpoint='resnet9'  
```

2. Prune the trained model  
```
python prune.py --model='resnet9' --checkpoint='resnet9'
```

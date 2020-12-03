#!/bin/zsh

prune_threshold=(0.01 0.015 0.02 0.025 0.03)

for t in "${prune_threshold[@]}"; do 
    rm -f ./trained_model/saved.pt
    cp ./mnist_trained_model/pretrained.pt ./trained_model/saved.pt

    python3 finetune.py "${t}"
    echo "==================================DATA HERE"
done

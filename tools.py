import random
import torch
import numpy as np


def learn(model, set_loaders, n_epochs):
    train_loader, test_loader = set_loaders

    for epoch in range(n_epochs):
        for data, labels in train_loader:
            pred = model.forward(data)
            print(pred)
            print(labels)
            exit()
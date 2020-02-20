#!/usr/bin/env python
# coding: utf-8

# This is a simple example on how you can use a jupyter notebook to train your model :) 


import torch
import torch.nn as nn
from dataloaders import load_cifar10
from task3 import Trainer, compute_loss_and_accuracy, create_plots
from time import gmtime, strftime


import torchvision
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax,
        # as this is done in nn.CrossEntropyLoss
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters():        # Unfreeze the last fully-connected
            param.requires_grad = True                  # layer
        for param in self.model.layer4.parameters():    # Unfreeze the last 5 convolutional
            param.requires_grad = True                  # layers

    def forward(self, x):
        x = self.model(x)
        return x

epochs = 10
batch_size = 32
learning_rate = 5e-4 # Should be 5e-5 for LeNet
early_stop_count = 20
dataloaders = load_cifar10(batch_size)
model = Model()
trainer = Trainer(
    batch_size,
    learning_rate,
    early_stop_count,
    epochs,
    model,
    dataloaders
)
trainer.train()
time = strftime("%m-%d%H%M%S", gmtime())
model_name = "task4_v1_" + time
create_plots(trainer, model_name)
with open("progress.txt", "a") as text_file:
    trainer.print_val_test_train_stats()
    trainer.load_best_model()
    print("Best accuracy " + model_name)
    text_file.write("Best accuracy " + model_name + "\n")
    output = trainer.print_val_test_train_stats()
    text_file.write(output)
    text_file.write("\n\n")


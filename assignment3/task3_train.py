#!/usr/bin/env python
# coding: utf-8

# This is a simple example on how you can use a jupyter notebook to train your model :) 


import torch
import torch.nn as nn
from dataloaders import load_cifar10
from task3 import Trainer, compute_loss_and_accuracy, create_plots
from time import gmtime, strftime

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        print("Initializing m: " + str(type(m)))
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu') 


class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = [32, 64, 64, 128, 64]  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters[0],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters[0],
                out_channels=num_filters[1],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d((2,2), 2),

            nn.Conv2d(
                in_channels=num_filters[1],
                out_channels=num_filters[2],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=num_filters[2],
                out_channels=num_filters[3],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            # nn.Conv2d(
            #     in_channels=num_filters[3],
            #     out_channels=num_filters[4],
            #     kernel_size=3,
            #     stride=1,
            #     padding=1
            # ),
            # nn.ReLU(),

            nn.Conv2d(
                in_channels=num_filters[3],
                out_channels=num_filters[4],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d((2,2), 2),

        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 4096
        self.classifier_weights = [self.num_output_features, 256, 64, num_classes]
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_weights[0], self.classifier_weights[1]),
            nn.ReLU(),
            nn.Linear(self.classifier_weights[1], self.classifier_weights[2]),
            nn.ReLU(),
            nn.Linear(self.classifier_weights[2 ], self.classifier_weights[3]),
        )
        self.apply(init_weights)

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        out = self.feature_extractor(x)
        out = out.view(batch_size, -1)
        out = self.classifier(out)        
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
                f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

epochs = 10
batch_size = 64
learning_rate = 5e-4 # Should be 5e-5 for LeNet
early_stop_count = 20
dataloaders = load_cifar10(batch_size)
model = ExampleModel(image_channels=3, num_classes=10)
trainer = Trainer(
    batch_size,
    learning_rate,
    early_stop_count,
    epochs,
    model,
    dataloaders
)
trainer.train()
with open("progress.txt", "a") as text_file:
    trainer.print_val_test_train_stats()
    trainer.load_best_model()
    time = strftime("%m-%d%H%M%S", gmtime())
    model_name = "task3_v8_6_filter_size_3_3_second_layer_no_dropout" + time
    print("Best accuracy " + model_name)
    text_file.write("Best accuracy " + model_name + "\n")
    output = trainer.print_val_test_train_stats()
    text_file.write(output)
    text_file.write("\n\n")


create_plots(trainer, model_name)
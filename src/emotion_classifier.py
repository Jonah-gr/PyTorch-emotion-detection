import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from device import DEVICE
from data_downloading import data_download
from label_and_dir import label_and_dir
from data_generator import data_loader

"""
#downloading data from Kaggle
# data_download()

#importing data
train_dir, valid_dir, test_dir, train_label, valid_label, test_label = label_and_dir()
train_generator, valid_generator, test_generator = data_loader(train_dir, valid_dir, test_dir, train_label, valid_label, test_label)
"""

# Assuming you have set up your data loaders and DEVICE
train_dir, valid_dir, test_dir, train_label, valid_label, test_label = label_and_dir()
# Assuming you have set up your data loaders and DEVICE
train_loader, valid_loader, test_loader = data_loader(train_dir, valid_dir, test_dir, train_label, valid_label, test_label)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.norm(self.pool(self.conv1(x))))
        return x

class Network(nn.Module):
    def __init__(self, num_classes=7):  # Assuming you have 7 emotion classes
        super(Network, self).__init()
        self.seq = nn.Sequential(
            Down(3, 32),  # Input has 3 channels for RGB
            nn.Dropout2d(0.2),
            Down(32, 64),
            nn.Dropout2d(0.2),
            Down(64, 128),
            nn.Dropout2d(0.2),
            Down(128, 256),
            nn.Dropout2d(0.2),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),  # Assuming 48x48 input images
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Output has num_classes units for emotion prediction
        )

    def forward(self, x):
        return self.seq(x)

if __name__ == "__main__":
    net = Network(num_classes=7).to(DEVICE)  # Set the appropriate number of classes
    total_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"Network has {total_parameters} total parameters")

    # Training loop for demonstration
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        for batch, labels in train_loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Testing the network with a batch from the dataloader
    batch, labels = next(iter(test_loader))
    batch, labels = batch.to(DEVICE), labels.to(DEVICE)
    x = net(batch)
    print(x.shape)

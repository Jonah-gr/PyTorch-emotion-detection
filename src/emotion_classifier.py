import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from device import DEVICE
from label_and_dir import label_and_dir
from data_generator import data_loader



class Network_complex(nn.Module):
    def __init__(self, num_classes=7):
        super(Network_complex, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 4096), nn.ReLU(),  # 3x3 due to downsampling with MaxPool2d
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.seq(x)
    

"""
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

class Network(nn.Module):
    def __init__(self, num_classes=7):
        super(Network, self).__init__()
        self.seq = nn.Sequential(
            Down(1, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 512),  # The original VGG16 structure has one more 512 channel layer
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.seq(x)
"""
"""
class Network(nn.Module):
    def __init__(self, include_top=True, weights=None, input_shape=(1, 48, 48), pooling=None, num_classes=7):
        super(Network, self).__init__()

        self.include_top = include_top
        self.weights = weights
        self.input_shape = input_shape
        self.pooling = pooling
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        if self.include_top:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.features(x)
        if self.include_top:
            x = self.classifier(x)
        return x
"""




class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.norm(self.pool(self.conv1(x))))
        return x

class Network(nn.Module):
    def __init__(self, num_classes=7):  # Assuming you have 7 emotion classes
        super(Network, self).__init__()
        self.seq = nn.Sequential(
            Down(1, 32),  
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

    net = Network().to(DEVICE)  # Set the appropriate number of classes
    total_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"Network has {total_parameters} total parameters")

    batch, labels = train_loader.__iter__().__next__()
    x = net(batch)
    print(x.shape)
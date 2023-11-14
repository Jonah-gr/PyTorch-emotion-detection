import torch.nn as nn
from device import DEVICE
from label_and_dir import label_and_dir
from data_generator import data_loader



class Network(nn.Module):
    def __init__(self, num_classes=7):
        super(Network, self).__init__()
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
            # nn.Flatten(),
            # nn.Linear(512 * 3 * 3, 4096), nn.ReLU(),  # 3x3 due to downsampling with MaxPool2d
            # nn.Dropout(0.5),
            # nn.Linear(4096, 4096), nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(4096, num_classes)
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 4096), nn.ReLU(),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.seq(x)
    
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
            nn.Flatten()
        )
        self.fc = nn.Linear(256 * 6 * 6, num_classes)  # Adjust based on the flattened size

    def forward(self, x):
        x = self.seq(x)
        x = self.fc(x)
        return x

# Use this updated Network class when defining your model for emotion recognition to ensure the proper handling of the input shapes.
"""



if __name__ == "__main__":
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
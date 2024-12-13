import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicCNN(nn.Module):
    def __init__(self, n_classes, input_channels=1):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers to be dynamically initialized
        self.fc1 = None
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.pool(F.relu(self.conv1(x)))
        print(f"Shape after conv1 and pool: {x.shape}")
        x = self.pool(F.relu(self.conv2(x)))
        print(f"Shape after conv2 and pool: {x.shape}")

        # Flatten tensor while maintaining batch size
        x = x.view(x.size(0), -1)  # Flatten
        print(f"Shape after flattening: {x.shape}")

        # Dynamically initialize fc1 if not already done
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)

        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        print(f"Logits shape: {logits.shape}")
        return logits

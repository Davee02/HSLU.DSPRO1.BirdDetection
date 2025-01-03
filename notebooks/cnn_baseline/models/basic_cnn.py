import torch.nn as nn
import torch.nn.functional as F
import torch

class BasicCNN(nn.Module):
    def __init__(self, input_shape, n_classes, input_channels=1):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dynamically calculate the size of the flattened features
        with torch.no_grad():
            # Add batch dimension (1) to the dummy input
            dummy_input = torch.zeros(1, input_channels, *input_shape)
            #print(f"Dummy input shape: {dummy_input.shape}")
            dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
            #print(f"Shape after conv1 and pool during init: {dummy_output.shape}")
            dummy_output = self.pool(F.relu(self.conv2(dummy_output)))
            #print(f"Shape after conv2 and pool during init: {dummy_output.shape}")
            self.flattened_size = dummy_output.view(1, -1).shape[1]
            #print(f"Flattened size during init: {self.flattened_size}")

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        x = self.pool(F.relu(self.conv1(x)))
        #print(f"Shape after conv1 and pool: {x.shape}")
        x = self.pool(F.relu(self.conv2(x)))
        #print(f"Shape after conv2 and pool: {x.shape}")

        # Flatten tensor while maintaining batch size
        x = x.view(x.size(0), -1)  # Flatten
        #print(f"Shape after flattening: {x.shape}")

        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        #print(f"Logits shape: {logits.shape}")
        return logits

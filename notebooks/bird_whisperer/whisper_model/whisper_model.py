import os
import whisper

import torch
from torch import nn
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self, n_classes, input_shape):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Dynamically calculate the size of the flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)  # Batch size = 1
            dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
            dummy_output = self.pool(F.relu(self.conv2(dummy_output)))
            flattened_size = dummy_output.numel()

        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class WhisperModel(torch.nn.Module):
    def __init__(self, n_classes, models_root_dir, device, variant="base"):
        super(WhisperModel, self).__init__()
        assert n_classes is not None, "'n_classes' cannot be None. Specify 'n_classes' present in the dataset."
        
        download_dir = os.path.join(models_root_dir, "base")
        os.makedirs(download_dir, exist_ok=True)

        self.audio_encoder = whisper.load_model(variant, download_root=download_dir).encoder

        # Compute the output shape of the Whisper encoder
        with torch.no_grad():
            dummy_audio_input = torch.randn(1, 80, 3000, device=device) # Batch size = 1, Mel-spectrogram dimensions
            encoder_output = self.audio_encoder(dummy_audio_input)
            encoder_output_shape = encoder_output.shape[-2:] # Take spatial dimensions

        self.classifier = CNN(n_classes, input_shape=encoder_output_shape)

    def forward(self, x):
        # Pass input through Whisper encoder
        features = self.audio_encoder(x)

        # Unsqueeze to add a channel dimension
        features = features.unsqueeze(1)

        # Pass the features through the CNN classifier
        logits = self.classifier(features)

        return logits

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
import time
import sys

# Set random seeds for reproducibility
SEED = 42
def set_seed(seed):
    print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Define a simple CNN model
class BasicCNN(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (input_shape[0] // 4) * (input_shape[1] // 4), 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Dataset class
class BirdSpectrogramDataset(Dataset):
    def __init__(self, df, file_index, target_shape=(128, 128)):
        self.df = df
        self.file_index = file_index
        self.target_shape = target_shape
        self.labels = sorted(df['en'].unique())
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec_id = row['id']
        spectrogram = self._load_spectrogram(spec_id)
        label = self.label2idx[row['en']]
        return torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0), label

    def _load_spectrogram(self, spec_id):
        if spec_id not in self.file_index:
            raise FileNotFoundError(f"Spectrogram ID {spec_id} not found in file index.")
        spec_files = self.file_index[spec_id]
        spectrograms = [np.load(file) for file in spec_files]
        combined_spec = np.mean(spectrograms, axis=0)
        return self._normalize_shape(combined_spec)

    def _normalize_shape(self, spectrogram):
        if spectrogram.shape == self.target_shape:
            return spectrogram
        elif spectrogram.shape[1] < self.target_shape[1]:
            pad_width = self.target_shape[1] - spectrogram.shape[1]
            return np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        else:
            return spectrogram[:, :self.target_shape[1]]

# Logging utility
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()

# Define a function to redirect stdout and stderr to a log file
def redirect_output_to_log(log_file_path):
    log = open(log_file_path, "a")
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)
    return log

# Define a function to setup logging
def setup_logging():
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = time.time()

    log_file_path = os.path.join(logs_dir, f"{timestamp}.log")
    json_log_file_path = os.path.join(logs_dir, f"{timestamp}.json")

    redirect_output_to_log(log_file_path)
    print(f"Logging to {log_file_path}")

    return log_file_path, json_log_file_path

# Training procedure with advanced logging and metrics
def train(device, model, train_loader, val_loader, criterion, optimizer, n_epochs, save_model_path, json_log_file_path):
    best_macro_avg_f1 = 0
    best_epoch = 0
    train_losses = []
    val_metrics = []

    os.makedirs(save_model_path, exist_ok=True)

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        model.train()

        train_loss = 0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
        train_losses.append(avg_train_loss)

        # Validation phase
        print(f"\nValidating after Epoch {epoch + 1}")
        val_accuracy, macro_avg_f1, class_report = validate(device, model, val_loader, criterion)
        val_metrics.append({
            "epoch": epoch + 1,
            "accuracy": val_accuracy,
            "macro_avg_f1": macro_avg_f1,
            "classification_report": class_report
        })

        # Save the best model
        if macro_avg_f1 > best_macro_avg_f1:
            best_macro_avg_f1 = macro_avg_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_model_path, "best_model.pth"))
            print(f"Saved the best model at Epoch {epoch + 1}")

        # Save metrics to JSON
        with open(json_log_file_path, "w") as f:
            json.dump({"train_losses": train_losses, "val_metrics": val_metrics}, f, indent=4)

    print(f"\nBest Macro Avg F1 Score: {best_macro_avg_f1:.4f} at Epoch {best_epoch}")

# Validation function
def validate(device, model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    actual_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            actual_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    accuracy = 100. * correct / total
    class_report = classification_report(actual_labels, predicted_labels, output_dict=True, zero_division=0)
    macro_avg_f1 = class_report["macro avg"]["f1-score"]

    print(f"Validation Accuracy: {accuracy:.2f}% | Macro Avg F1 Score: {macro_avg_f1:.4f}")
    return accuracy, macro_avg_f1, class_report

if __name__ == "__main__":
    train_spectrogram_folder = "../../data/processed/bird-whisperer/spectrograms/train/"
    train_parquet_file = "../../data/cleaned/80_20_cleaned_train.parquet"
    test_spectrogram_folder = "../../data/processed/bird-whisperer/spectrograms/test/"
    test_parquet_file = "../../data/cleaned/80_20_cleaned_test.parquet"

    train_file_index = {}
    for file in os.listdir(train_spectrogram_folder):
        file_id = file.split('_')[0]
        if file_id not in train_file_index:
            train_file_index[file_id] = []
        train_file_index[file_id].append(os.path.join(train_spectrogram_folder, file))

    test_file_index = {}
    for file in os.listdir(test_spectrogram_folder):
        file_id = file.split('_')[0]
        if file_id not in test_file_index:
            test_file_index[file_id] = []
        test_file_index[file_id].append(os.path.join(test_spectrogram_folder, file))

    train_df = pd.read_parquet(train_parquet_file)
    test_df = pd.read_parquet(test_parquet_file)

    train_dataset = BirdSpectrogramDataset(train_df, train_file_index)
    test_dataset = BirdSpectrogramDataset(test_df, test_file_index)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicCNN(input_shape=(128, 128), n_classes=len(train_dataset.labels))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    json_log_file_path = os.path.join(logs_dir, "training_metrics.json")
    save_model_path = "models"

    train(device, model, train_loader, test_loader, criterion, optimizer, n_epochs=10, save_model_path=save_model_path, json_log_file_path=json_log_file_path)

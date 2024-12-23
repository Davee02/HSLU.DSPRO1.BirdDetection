import os
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets.dataset_utils import get_dataloaders
from notebooks.bird_whisperer.whisper_model import whisper_model

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def evaluate_model(model, device, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:  # Third element is not needed
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

if __name__ == "__main__":
    dataset_root = "../../data/processed/bird-whisperer/"
    batch_size = 32
    num_workers = 4
    test_parquet_name = "test.parquet"
    checkpoint_path = "/mnt/d/DSPRO1/trained_models/03_base_full_with_augmented.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataloader, unique_labels = get_dataloaders(
        dataset_root,
        batch_size,
        num_workers,
        test_parquet_name=test_parquet_name,
        with_augmented=False
    )
    print(f"Number of test samples: {len(test_dataloader.dataset)}")

    model = whisper_model.WhisperModel(
        n_classes=len(unique_labels),
        models_root_dir=None,  # Not used in evaluation
        variant=None,          # Not used in evaluation
        device=device,
        dropout_p=0.0          # Not used in evaluation
    )
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully.")

    # Evaluate
    print("Evaluating model...")
    predictions, true_labels = evaluate_model(model, device, test_dataloader)

    # Metrics
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(true_labels, predictions, target_names=unique_labels))

    # Confusion Matrix
    plot_confusion_matrix(true_labels, predictions, unique_labels)

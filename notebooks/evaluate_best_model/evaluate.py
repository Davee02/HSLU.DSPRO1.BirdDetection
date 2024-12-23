# evaluation.py
import sys
from pathlib import Path

# Add the root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from pathlib import Path
import sys
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datasets.dataset_utils import get_dataloader
from model_loader import load_model
from utils import load_config

def evaluate_model(config):
    # Load configurations
    data_config = config['data']
    model_config = config['model']

    # Prepare data loaders
    test_dataloader, label2bird_dict = get_dataloader(
        data_config['dataset_root'],
        data_config['batch_size'],
        data_config['num_workers'],
        data_config['test_parquet_name'],
        with_augmented=data_config['with_augmented']
    )

    # Load model
    model, device = load_model(
        len(label2bird_dict),
        model_config['model_dir'],
        model_config['checkpoint_path']
    )
    model.eval()

    true_labels = []
    predictions = []

    # Evaluation loop
    with torch.no_grad():
        for mel_spectrograms, labels, _ in test_dataloader:
            mel_spectrograms = mel_spectrograms.to(device)
            logits = model(mel_spectrograms).cpu()
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.numpy())
            true_labels.extend(labels.numpy())

    # Map predictions to bird names
    predicted_bird_names = [label2bird_dict[pred] for pred in predictions]
    true_bird_names = [label2bird_dict[true] for true in true_labels]

    # Metrics calculation
    print("Confusion Matrix:")
    print(confusion_matrix(true_bird_names, predicted_bird_names))

    print("\nClassification Report:")
    print(classification_report(true_bird_names, predicted_bird_names))

    print("\nAccuracy:", accuracy_score(true_bird_names, predicted_bird_names))

if __name__ == "__main__":
    config = load_config("./config.yaml")
    evaluate_model(config)
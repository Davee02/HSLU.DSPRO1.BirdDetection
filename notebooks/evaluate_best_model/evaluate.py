import sys
from pathlib import Path
import json
import numpy as np

# Add the root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from pathlib import Path
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
from datasets.dataset_utils import get_dataloader
from model_loader import load_model
from utils import load_config

def evaluate_model(config):
    data_config = config['data']
    model_config = config['model']

    test_dataloader, label2bird_dict = get_dataloader(
        data_config['dataset_root'],
        data_config['batch_size'],
        data_config['num_workers'],
        data_config['test_parquet_name'],
        with_augmented=data_config['with_augmented']
    )

    model, device = load_model(
        len(label2bird_dict),
        model_config['model_dir'],
        model_config['checkpoint_path']
    )
    model.eval()

    true_labels = []
    predictions = []
    prediction_probs = []

    # Evaluation loop
    with torch.no_grad():
        for mel_spectrograms, labels, _ in test_dataloader:
            mel_spectrograms = mel_spectrograms.float().to(device)
            logits = model(mel_spectrograms).cpu()
            preds = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1).numpy()
            prediction_probs.extend(probabilities.tolist())
            predictions.extend(preds.numpy().tolist())
            true_labels.extend(labels.numpy().tolist())

    # Map predictions to bird names
    predicted_bird_names = [label2bird_dict[pred] for pred in predictions]
    true_bird_names = [label2bird_dict[true] for true in true_labels]

    # Calculate the frequency of each bird in the test set
    bird_counts = Counter(true_bird_names)

    confusion_mat = confusion_matrix(true_bird_names, predicted_bird_names)
    normalized_confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    class_report = classification_report(true_bird_names, predicted_bird_names, output_dict=True)
    accuracy = accuracy_score(true_bird_names, predicted_bird_names)

    for bird, count in bird_counts.items():
        if bird in class_report:
            class_report[bird]['support'] = count

    # Per-class accuracy
    per_class_accuracy = {
        bird: confusion_mat[i, i] / bird_counts[bird]
        for i, bird in enumerate(label2bird_dict.values()) if bird_counts[bird] > 0
    }

    # Misclassified samples
    misclassified = [
        {"true_label": true_bird_names[i], "predicted_label": predicted_bird_names[i], "probabilities": prediction_probs[i]}
        for i in range(len(true_bird_names)) if true_bird_names[i] != predicted_bird_names[i]
    ]

    log_data = {
        "confusion_matrix": confusion_mat.tolist(),
        "normalized_confusion_matrix": normalized_confusion_mat.tolist(),
        "classification_report": class_report,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "misclassified_samples": misclassified
    }

    log_file_path = "evaluation_log.json"
    with open(log_file_path, "w") as log_file:
        json.dump(log_data, log_file, indent=4)
    print(f"Evaluation log saved to {log_file_path}")

if __name__ == "__main__":
    config = load_config("./config.yaml")
    evaluate_model(config)

# evaluation.py
from pathlib import Path
import sys
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_preparation import load_evaluation_data
from model_loader import load_model
from utils import load_config

def evaluate_model(config):
    # Load configurations
    data_config = config['data']
    model_config = config['model']

    # Prepare data
    evaluation_data, true_labels, bird2label_dict, label2bird_dict = load_evaluation_data(
        data_config['evaluation_data_path']
    )

    # Load model
    model, device = load_model(
        len(bird2label_dict),
        model_config['model_dir'],
        model_config['checkpoint_path']
    )

    # Evaluation loop
    model.eval()
    predictions = []

    with torch.no_grad():
        for input_data in evaluation_data:
            input_tensor = torch.tensor(input_data).unsqueeze(0).float().to(device)
            logits = model(input_tensor).cpu()
            predicted_label = torch.argmax(logits, dim=1).item()
            predictions.append(predicted_label)

    # Map predictions to bird names
    predicted_bird_names = [label2bird_dict[pred] for pred in predictions]
    true_bird_names = [label2bird_dict[true] for true in true_labels]

    # Metrics calculation
    print("Confusion Matrix:")
    print(confusion_matrix(true_bird_names, predicted_bird_names))

    print("Classification Report:")
    print(classification_report(true_bird_names, predicted_bird_names))

    print("Accuracy:", accuracy_score(true_bird_names, predicted_bird_names))

if __name__ == '__main__':
    # Load configuration
    config = load_config("./config.yaml")

    # Run evaluation
    evaluate_model(config)

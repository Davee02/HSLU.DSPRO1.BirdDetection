import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from your_dataset_module import BirdDataset
from notebooks.bird_whisperer.whisper_model import whisper_model

# Load Model

def load_model(n_species, model_dir, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = whisper_model.WhisperModel(
        n_classes=n_species, models_root_dir=model_dir, variant="base", device=device, dropout_p=0.0
    )
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model successfully loaded from checkpoint: {checkpoint_path}")
    return model, device

# Evaluation Function

def evaluate_model(model, device, dataloader, label2bird_dict):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

# Visualize Confusion Matrix

def plot_confusion_matrix(y_true, y_pred, label2bird_dict):
    cm = confusion_matrix(y_true, y_pred)
    class_labels = [label2bird_dict[i] for i in range(len(label2bird_dict))]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Main Evaluation Script

if __name__ == "__main__":
    # Configuration
    config = load_config("./config.yaml")
    model_config = config['model']
    data_config = config['data']

    # Prepare Data
    bird2label_dict, label2bird_dict = prepare_data(
        data_config['train_data_path'],
        data_config['test_data_path']
    )
    test_dataset = BirdDataset(data_config['test_data_path'])  # Replace with your dataset class
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load Model
    model, device = load_model(
        len(bird2label_dict),
        model_config['model_dir'],
        model_config['checkpoint_path']
    )

    # Evaluate Model
    print("Evaluating the model...")
    predictions, true_labels = evaluate_model(model, device, test_dataloader, label2bird_dict)

    # Calculate Metrics
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(true_labels, predictions, target_names=label2bird_dict.values()))

    # Plot Confusion Matrix
    plot_confusion_matrix(true_labels, predictions, label2bird_dict)

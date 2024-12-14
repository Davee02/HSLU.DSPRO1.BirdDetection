# main.py

import os
import random
import numpy as np
import torch
from trainer import train
from logging_utils import setup_logging
from datasets.dataset_utils import get_dataloaders
from models.basic_cnn import BasicCNN

LR = 0.01
SEED = 42
EPOCHS = 20
WITH_AUGMENTED = True

def set_seed(seed):
    print(f"Setting seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def print_dataset_info(train_dataloader, test_dataloader):
    num_batches_train = len(train_dataloader)
    num_batches_test = len(test_dataloader)

    print("\n================================= Dataset Information =================================")
    print("Number of Samples in Train Dataset: ", len(train_dataloader.dataset))
    print("Number of Batches in Train Dataloader: ", num_batches_train)
    print("Train Batch Size: ", train_dataloader.batch_size)
    print()
    print("Number of Samples in Test Dataset: ", len(test_dataloader.dataset))
    print("Number of Batches in Test Dataloader: ", num_batches_test)
    print("Test Batch Size: ", test_dataloader.batch_size)
    print()
    print(f"Use augmented data: {WITH_AUGMENTED}\n\n")

def main(json_log_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        raise Exception("This script is not optimized for CPU. It is recommended to use a GPU.")

    set_seed(SEED)

    dataset_root = os.path.join(os.path.dirname(__file__), "../../data/processed/bird-whisperer")
    train_dataloader, test_dataloader, unique_labels = get_dataloaders(dataset_root, with_augmented=WITH_AUGMENTED)
    print_dataset_info(train_dataloader, test_dataloader)

    save_model_path = os.path.join(os.path.dirname(__file__), "../../data/basic_cnn/models")
    print(f"Saving models to: {save_model_path}")

    model = BasicCNN(input_shape=(80, 3000), n_classes=len(unique_labels), input_channels=1)
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    start_epoch = 0
    models_save_dir = os.path.join(save_model_path, "trained")
    train(device, model, train_dataloader, test_dataloader, criterion, optimizer, unique_labels, start_epoch, EPOCHS, models_save_dir, json_log_file_path)

if __name__ == "__main__":
    _, json_log_file_path = setup_logging()
    main(json_log_file_path)

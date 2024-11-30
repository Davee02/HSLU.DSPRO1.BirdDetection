import os
import random
import numpy as np
import torch
from utils import trainer
from utils.datasets.dataset_utils import get_dataloaders
from whisper_model import whisper_model

LR = 0.01
SEED = 42

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU if available for torch, else use CPU
  print(f"Using device: {device}")
  if device == "cpu":
    raise Exception("This script is not optimized for CPU. It is recommended to use a GPU.")

  set_seed(SEED) # set seed for repro

  dataset_root = os.path.join(os.path.dirname(__file__), "../data/processed/bird-whisperer") # define path to dataset
  train_dataloader, test_dataloader, unique_labels = get_dataloaders(dataset_root)

  save_model_path = os.path.join(os.path.dirname(__file__), "../data/bird-whisperer/models") # define path to where the model will be saved
  print(f"Saving models to: {save_model_path}")

  model = whisper_model.WhisperModel(n_classes=len(unique_labels), models_root_dir=save_model_path, variant="tiny") 
  model = model.to(device) # move model to device (GPU or CPU)

  criterion = torch.nn.CrossEntropyLoss() # loss function
  optimizer = torch.optim.SGD(model.parameters(), lr=LR)

  start_epoch = 0
  n_epochs = 1
  models_save_dir = os.path.join(save_model_path, "trained")
  trainer.train(device, model, train_dataloader, test_dataloader, criterion, optimizer, unique_labels, start_epoch, n_epochs, models_save_dir)

if __name__ == "__main__":
  main()
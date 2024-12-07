import os
import random
import numpy as np
import torch

from trainer import train
from logging_utils import setup_logging
from datasets.dataset_utils import get_dataloaders
from whisper_model import whisper_model
from experiment_utils import load_from_checkpoint, parse_arguments

def set_seed(seed):
  print(f"Setting seed: {seed}")
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)

def print_dataset_info(train_dataloader, test_dataloader, args):
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
    print(f"Use augmented data: {args.with_augmented}\n\n")

def main(json_log_file_path, args):
  torch.set_float32_matmul_precision('high')
    
  print("\n================================= Bird Whisperer Trainer =================================")
  print(f"Arguments: {args}\n")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU if available for torch, else use CPU
  print(f"Using device: {device}")
  if device.type == "cpu":
    raise Exception("This script is not optimized for CPU. It is recommended to use a GPU.")

  set_seed(args.seed) # set seed for repro

  dataset_root = os.path.join(os.path.dirname(__file__), "../../data/processed/bird-whisperer") # define path to dataset
  train_dataloader, test_dataloader, unique_labels = get_dataloaders(dataset_root, with_augmented=args.with_augmented, batch_size=args.batch_size)
  print_dataset_info(train_dataloader, test_dataloader, args)

  save_model_path = os.path.join(os.path.dirname(__file__), "../../data/bird-whisperer/models") # define path to where the model will be saved
  print(f"Saving models to: {save_model_path}")

  models_save_dir = os.path.join(save_model_path, "trained")

  model = whisper_model.WhisperModel(n_classes=len(unique_labels), models_root_dir=save_model_path, variant=args.whisper_base_variant, device=device) 
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=True)

  start_epoch = 0
  best_epoch = 0
  best_f1_score = 0

  checkpoint_file = args.checkpoint_file
  if checkpoint_file:
    checkpoint_path = os.path.join(models_save_dir, checkpoint_file)
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model, optimizer, start_epoch, _, best_f1_score, best_epoch = load_from_checkpoint(checkpoint_path, model, optimizer)

  model = model.to(device) # move model to device (GPU or CPU)
  criterion = torch.nn.CrossEntropyLoss() # loss function

  print("Compiling model...")
  compiled_model = torch.compile(model)
  print("Successfully compiled model")
    
  train(device, compiled_model, train_dataloader, test_dataloader, criterion, optimizer, unique_labels, start_epoch, args.epochs, models_save_dir, json_log_file_path, best_f1_score=best_f1_score, best_epoch=best_epoch, debug=args.debug)

if __name__ == "__main__":
  args = parse_arguments()
  _, json_log_file_path = setup_logging()
  main(json_log_file_path, args)
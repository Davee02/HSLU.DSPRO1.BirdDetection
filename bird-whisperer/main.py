import os
import random
import numpy as np
import torch
from whisper_model import whisper_model

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

  set_seed(42) # set seed for repro

  save_model_path = os.path.join(os.path.dirname(__file__), "/../data/bird-whisperer/models") # define path to where the model will be saved
  os.makedirs(save_model_path, exist_ok=True)
  print(f"Saving model to: {save_model_path}")

  model = whisper_model.WhisperModel(n_classes=100) 
  model = model.to(device) # move model to device (GPU or CPU)

if __name__ == "__main__":
  main()
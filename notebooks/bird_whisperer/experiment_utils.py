import os
import torch

def load_from_checkpoint(checkpoint_path, model, optimizer):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at path {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    epoch_avg_loss = checkpoint["epoch_avg_loss"]
    best_f1_score = checkpoint["epoch_avg_loss"]
    best_epoch = checkpoint["best_epoch"]

    return model, optimizer, epoch, epoch_avg_loss, best_f1_score, best_epoch
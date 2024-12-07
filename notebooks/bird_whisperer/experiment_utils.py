import argparse
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Bird Whisperer Trainer")

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility (default: 42)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--with_augmented", action="store_true", default=True, help="Use augmented data (default: True)")
    parser.add_argument("--whisper_base_variant", type=str, default="tiny", help="Whisper model base variant (default: tiny)")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="Checkpoint file to load (default: None)")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode (default: False)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")

    return parser.parse_args()
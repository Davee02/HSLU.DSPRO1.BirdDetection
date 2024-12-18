import argparse
import os
import torch

def load_from_checkpoint(checkpoint_path, model, optimizer, map_location):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at path {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # when saved to disk the key in the state dict all get the prefix "_orig_mod." which makes them unrecognizable when loaded again
    # thus we need to remove the prefix from the keys of the state dict before loading it
    remove_prefix = '_orig_mod.'
    model_state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in checkpoint["model_state_dict"].items()}

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    epoch_avg_loss = checkpoint["epoch_avg_loss"]
    best_f1_score = checkpoint["best_f1_score"]
    best_epoch = checkpoint["best_epoch"]

    return model, optimizer, epoch, epoch_avg_loss, best_f1_score, best_epoch

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Bird Whisperer Trainer")

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility (default: 42)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    parser.add_argument("--with_augmented", action="store_true", default=False, help="Use augmented data (default: True)")
    parser.add_argument("--whisper_base_variant", type=str, default="tiny", help="Whisper model base variant (default: tiny)")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="Checkpoint file to load (default: None)")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode (default: False)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW optimizer (default: 1e-2)")
    parser.add_argument("--train_parquet_name", type=str, default="train.parquet", help="File name of the train parquet file (default: train.parquet)")
    parser.add_argument("--test_parquet_name", type=str, default="test.parquet", help="File name of the test parquet file (default: test.parquet)")
    parser.add_argument("--dataset_root", type=dir_path, default=os.path.join(os.path.dirname(__file__), "../../data/processed/bird-whisperer"), help="Path to dataset root (default: ../../data/processed/bird-whisperer)")
    parser.add_argument("--dropout_p", type=float, default=0.5, help="Dropout probability for the FCNs (default: 0.5)")

    return parser.parse_args()

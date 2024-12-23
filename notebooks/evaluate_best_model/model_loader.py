import torch
from notebooks.bird_whisperer.whisper_model import whisper_model

def load_model(n_classes, model_dir, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = whisper_model.WhisperModel(
        n_classes=n_classes, models_root_dir=model_dir, variant="base", device=device, dropout_p=0.0
    )
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model successfully loaded from checkpoint: {checkpoint_path}")
    return model, device
import sys
import io
from pathlib import Path
from flask import Flask, jsonify, request
import torch

sys.path.append(str(Path(__file__).resolve().parents[2])) # add the path to the root of the project to sys.path so that we can import modules from the project
from notebooks.audio_processing import AudioProcessor
from notebooks.bird_whisperer.whisper_model import whisper_model

def prepare_audio_stuff():
    SAMPLE_RATE = 16 * 1000
    SEGMENT_DURATION = 30
    ap = AudioProcessor(sample_rate=SAMPLE_RATE, segment_duration=SEGMENT_DURATION, target_db_level=-20)

    log_mel_params = {
        "n_fft": int(0.025 * SAMPLE_RATE), # 25 milliseconds in samples
        "hop_length": int(0.010 * SAMPLE_RATE), # 10 milliseconds in samples
        "n_mels": 80 # Number of Mel bands
    }
    return ap, log_mel_params

def preprocess_audio(audio_data, ap, log_mel_params):
    audio_data = ap.process_audio_file_with_denoising(audio_data)
    spectogram = ap.create_log_mel_spectrogram(audio_data, n_fft=log_mel_params["n_fft"], hop_length=log_mel_params["hop_length"], n_mels=log_mel_params["n_mels"])
    normalized_spectogram = ap.normalize_spectrogram(spectogram)
    return normalized_spectogram

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = whisper_model.WhisperModel(n_classes=186, models_root_dir="/home/david/git/HSLU.DSPRO1.BirdDetection/data/bird-whisperer/models/", variant="base", device=device, dropout_p=0.0) 
    model = model.to(device) # move model to device (GPU or CPU)

    checkpoint_path = "/mnt/d/DSPRO1/trained_models/03_base_full_with_augmented.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

app = Flask(__name__)
ap, log_mel_params = prepare_audio_stuff()
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if 'recording' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['recording']
    try:
        # Read the file as a byte stream
        audio_data = io.BytesIO(file.read())

        normalized_spectogram = preprocess_audio(audio_data, ap, log_mel_params)

        return jsonify({'ok': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()

import io
from pathlib import Path
import sys
from flask import Flask, jsonify, request

def get_audio_stuff():
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from notebooks.audio_processing import AudioProcessor

    SAMPLE_RATE = 16 * 1000
    SEGMENT_DURATION = 30
    ap = AudioProcessor(sample_rate=SAMPLE_RATE, segment_duration=SEGMENT_DURATION, target_db_level=-20)

    log_mel_params = {
        "n_fft": int(0.025 * SAMPLE_RATE), # 25 milliseconds in samples
        "hop_length": int(0.010 * SAMPLE_RATE), # 10 milliseconds in samples
        "n_mels": 80 # Number of Mel bands
    }
    return ap, log_mel_params

app = Flask(__name__)
ap, log_mel_params = get_audio_stuff()


@app.route('/predict', methods=['POST'])
def predict():
    if 'recording' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['recording']
    try:
        # Read the file as a byte stream
        audio_data = io.BytesIO(file.read())

        # Load the audio with librosa
        audio_data = ap.process_audio_file_with_denoising(audio_data)
        spectogram = ap.create_log_mel_spectrogram(audio_data, n_fft=log_mel_params["n_fft"], hop_length=log_mel_params["hop_length"], n_mels=log_mel_params["n_mels"])
        normalized_spectogram = ap.normalize_spectrogram(spectogram)

        return jsonify({'ok': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()

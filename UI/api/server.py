from pathlib import Path
import sys
from flask import Flask, jsonify, request
import torch
import io

sys.path.append(str(Path(__file__).resolve().parents[2])) # add the path to the root of the project to sys.path so that we can import modules from the project

from audio_processing import prepare_audio_stuff, preprocess_audio
from model_loader import load_model
from data_preparation import prepare_data
from utils import load_config

app = Flask(__name__)
app.json.sort_keys = False

def create_app(config):
    # Load configurations
    audio_config = config['audio']
    data_config = config['data']
    model_config = config['model']

    # Prepare audio processor and parameters
    ap, log_mel_params = prepare_audio_stuff(
        audio_config['sample_rate'],
        audio_config['segment_duration'],
        audio_config['target_db_level']
    )

    # Prepare data
    bird2label_dict, label2bird_dict = prepare_data(
        data_config['train_data_path'],
        data_config['test_data_path']
    )

    # Load model
    model, device = load_model(
        len(bird2label_dict),
        model_config['model_dir'],
        model_config['checkpoint_path']
    )

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            if 'recording' not in request.files:
                return jsonify({'error': 'Recording file is missing in the request'}), 400

            file = request.files['recording']
            audio_data = io.BytesIO(file.read())
            normalized_spectogram = preprocess_audio(audio_data, ap, log_mel_params)

            with torch.no_grad():
                input_tensor = torch.tensor(normalized_spectogram).unsqueeze(0).float().to(device)
                logits = model(input_tensor).cpu()
                probabilities = torch.softmax(logits, dim=1).numpy()

                dict_probabilities = {
                    label2bird_dict[i]: prob.astype(float) for i, prob in enumerate(probabilities[0])
                }
                sorted_dict_probabilities = dict(
                    sorted(dict_probabilities.items(), key=lambda item: item[1], reverse=True)
                )

                return jsonify({"predictions": sorted_dict_probabilities}), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return app

if __name__ == '__main__':
    # Load configuration
    config = load_config("./config.yaml")

    app = create_app(config)
    app.run(host=config['app']['host'], port=config['app']['port'])

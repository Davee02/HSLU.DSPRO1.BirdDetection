from notebooks.audio_processing import AudioProcessor

def prepare_audio_stuff(sample_rate, segment_duration, target_db_level):
    ap = AudioProcessor(sample_rate=sample_rate, segment_duration=segment_duration, target_db_level=target_db_level)
    log_mel_params = {
        "n_fft": int(0.025 * sample_rate),  # 25 milliseconds in samples
        "hop_length": int(0.010 * sample_rate),  # 10 milliseconds in samples
        "n_mels": 80  # Number of Mel bands
    }
    return ap, log_mel_params

def preprocess_audio(audio_data, ap, log_mel_params):
    audio_data = ap.process_audio_file_with_denoising(audio_data)
    spectogram = ap.create_log_mel_spectrogram(audio_data, **log_mel_params)
    normalized_spectogram = ap.normalize_spectrogram(spectogram)
    return normalized_spectogram

import numpy as np
import librosa

class AudioProcessor:
    def __init__(self, sample_rate=22050, segment_duration=20, target_db_level=-20, seed=42):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.target_db_level = target_db_level
        np.random.seed(seed)  # Make the seed a parameter

    def normalize_audio(self, y):
        rms = librosa.feature.rms(y=y)[0]
        current_db = librosa.amplitude_to_db(rms, ref=np.max)
        db_adjustment = self.target_db_level - np.mean(current_db)
        return y * (10 ** (db_adjustment / 20))

    def add_gaussian_noise(self, y, noise_level=0.005):
        noise = np.random.normal(0, noise_level, y.shape)
        return y + noise

    def spectral_gating(self, y, threshold_ratio=1.2):
        stft = librosa.stft(y)
        magnitude, phase = np.abs(stft), np.angle(stft)
        noise_magnitude = np.median(magnitude, axis=1).reshape((-1, 1))
        mask = magnitude >= (noise_magnitude * threshold_ratio)
        denoised_stft = mask * magnitude * np.exp(1j * phase)
        return librosa.istft(denoised_stft)

    def process_audio_file_with_denoising(self, audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
        y, _ = librosa.effects.trim(y, top_db=20)
        y = self.spectral_gating(y)
        y = self.normalize_audio(y)
        segment_length = self.segment_duration * self.sample_rate
        return np.pad(y, (0, max(0, segment_length - len(y))))[:segment_length]

    def create_log_mel_spectrogram(self, y, n_fft=2048, hop_length=512, n_mels=128):
        # Added parameters n_fft, hop_length, n_mels
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        return librosa.power_to_db(mel_spec, ref=np.max)

    def apply_spectrogram_augmentation(self, S_db, time_mask_lower=50, time_mask_upper=100, freq_mask_lower=1, freq_mask_upper=7):
        # Made lower and upper bounds for time and frequency masks parameters
        time_mask = np.random.randint(time_mask_lower, time_mask_upper)
        t0 = np.random.randint(0, max(S_db.shape[1] - time_mask, 1))
        freq_mask = np.random.randint(freq_mask_lower, freq_mask_upper)
        f0 = np.random.randint(0, max(S_db.shape[0] - freq_mask, 1))
        S_db[:, t0:t0 + time_mask] = 0
        S_db[f0:f0 + freq_mask, :] = 0
        return S_db

    def create_combined_features(self, y, n_mfcc=13):
        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=n_mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sample_rate)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=self.sample_rate)
        return np.concatenate([
            (mfcc - np.mean(mfcc)) / np.std(mfcc),
            (chroma - np.mean(chroma)) / np.std(chroma),
            (tonnetz - np.mean(tonnetz)) / np.std(tonnetz)
        ], axis=0)

    def save_log_mel_spectrogram(self, S_db, output_path):
        np.save(output_path, S_db)

    def save_combined_features(self, features, output_path):
        np.save(output_path, features)

    def add_augmentations(self, y):
        if np.random.rand() < 0.3:
            y = librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=2)
        if np.random.rand() < 0.3:
            y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.95, 1.05))
        if np.random.rand() < 0.3:
            y = self.add_gaussian_noise(y)
        return y

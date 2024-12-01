import numpy as np
import torch
import librosa

class AudioProcessor:
    def __init__(self, sample_rate=16000, segment_duration=20, target_db_level=-20, seed=42):
        """
        Initializes the AudioProcessor with specified parameters.

        Parameters:
        sample_rate (int): The sample rate for audio processing.
        segment_duration (int): The duration (in seconds) to which audio segments will be padded or truncated.
        target_db_level (int): The target decibel level for normalization.
        seed (int): Random seed for reproducibility.
        """
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.target_db_level = target_db_level
        np.random.seed(seed)
        torch.manual_seed(seed)

    def normalize_audio(self, y):
        """
        Normalizes the audio signal to the target decibel level.

        Parameters:
        y (np.ndarray): The audio signal to normalize.

        Returns:
        np.ndarray: The normalized audio signal.
        """
        rms = librosa.feature.rms(y=y)[0]
        current_db = librosa.amplitude_to_db(rms, ref=np.max)
        db_adjustment = self.target_db_level - np.mean(current_db)
        return y * (10 ** (db_adjustment / 20))

    def add_gaussian_noise(self, y, noise_level=0.005):
        """
        Adds Gaussian noise to the audio signal.

        Parameters:
        y (np.ndarray): The original audio signal.
        noise_level (float): Standard deviation of the Gaussian noise to be added.

        Returns:
        np.ndarray: The audio signal with added noise.
        """
        noise = np.random.normal(0, noise_level, y.shape)
        return y + noise

    def spectral_gating(self, y, threshold_ratio=1.2):
        """
        Applies spectral gating noise reduction to the audio signal.

        Parameters:
        y (np.ndarray): The original audio signal.
        threshold_ratio (float): The ratio used to determine the noise threshold.

        Returns:
        np.ndarray: The denoised audio signal.
        """
        stft = librosa.stft(y)
        magnitude, phase = np.abs(stft), np.angle(stft)
        noise_magnitude = np.median(magnitude, axis=1).reshape((-1, 1))
        mask = magnitude >= (noise_magnitude * threshold_ratio)
        denoised_stft = mask * magnitude * np.exp(1j * phase)
        return librosa.istft(denoised_stft)

    def ensure_target_duration(self, y):
        """
        Trims or pads the audio signal to the target duration.

        Parameters:
        y (np.ndarray): The audio signal to process.

        Returns:
        np.ndarray: The audio signal of fixed duration.
        """
        target_length = self.segment_duration * self.sample_rate - 1
        if len(y) > target_length:
            # Trim the audio
            return y[:target_length]
        elif len(y) < target_length:
            # Repeat the audio to reach the target length
            num_repeats = target_length // len(y)
            remainder = target_length % len(y)
            return np.concatenate([y] * num_repeats + [y[:remainder]])
        else:
            return y

    def process_audio_file_with_denoising(self, audio_path):
        """
        Loads an audio file, applies denoising, normalization, and trims or pads it to a fixed duration.

        Parameters:
        audio_path (str): Path to the audio file to be processed.

        Returns:
        np.ndarray: The processed audio signal of fixed duration.
        """
        y, sr = librosa.load(audio_path, sr=None)
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
        y, _ = librosa.effects.trim(y, top_db=20)
        y = self.spectral_gating(y)
        y = self.normalize_audio(y)

        return self.ensure_target_duration(y)

    def create_log_mel_spectrogram(self, y, n_fft=2048, hop_length=512, n_mels=128):
        """
        Generates a log-scaled Mel spectrogram from an audio signal.

        Parameters:
        y (np.ndarray): The audio signal.
        n_fft (int): Number of FFT components.
        hop_length (int): Number of samples between successive frames.
        n_mels (int): Number of Mel bands to generate.

        Returns:
        np.ndarray: The log-scaled Mel spectrogram in decibels.
        """
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        return librosa.power_to_db(mel_spec, ref=np.max)

    def normalize_spectrogram(self, S_db):
        """
        Normalizes the spectrogram to have zero mean and standard deviation of 1.

        Parameters:
        S_db (np.ndarray): The input spectrogram in decibels.

        Returns:
        np.ndarray: The normalized spectrogram.
        """

        return (S_db - np.mean(S_db)) / np.std(S_db)

    def apply_spectrogram_augmentation(self, S_db, time_mask_lower=50, time_mask_upper=100, freq_mask_lower=1, freq_mask_upper=7):
        """
        Applies random time and frequency masking to a spectrogram for data augmentation.

        Parameters:
        S_db (np.ndarray): The input spectrogram in decibels.
        time_mask_lower (int): Lower bound for time mask width.
        time_mask_upper (int): Upper bound for time mask width.
        freq_mask_lower (int): Lower bound for frequency mask width.
        freq_mask_upper (int): Upper bound for frequency mask width.

        Returns:
        np.ndarray: The augmented spectrogram.
        """
        time_mask = np.random.randint(time_mask_lower, time_mask_upper)
        t0 = np.random.randint(0, max(S_db.shape[1] - time_mask, 1))
        freq_mask = np.random.randint(freq_mask_lower, freq_mask_upper)
        f0 = np.random.randint(0, max(S_db.shape[0] - freq_mask, 1))
        S_db[:, t0:t0 + time_mask] = 0
        S_db[f0:f0 + freq_mask, :] = 0
        return S_db

    def create_combined_features(self, y, n_mfcc=13):
        """
        Extracts and combines MFCC, chroma, and tonnetz features from an audio signal.

        Parameters:
        y (np.ndarray): The audio signal.
        n_mfcc (int): Number of MFCCs to return.

        Returns:
        np.ndarray: Combined and normalized feature array.
        """
        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=n_mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sample_rate)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=self.sample_rate)
        return np.concatenate([
            (mfcc - np.mean(mfcc)) / np.std(mfcc),
            (chroma - np.mean(chroma)) / np.std(chroma),
            (tonnetz - np.mean(tonnetz)) / np.std(tonnetz)
        ], axis=0)

    def save_log_mel_spectrogram(self, S_db, output_path, type="npy"):
        """
        Saves the log-scaled Mel spectrogram to a NumPy file.

        Parameters:
        S_db (np.ndarray): The log-scaled Mel spectrogram in decibels.
        output_path (str): The file path to save the spectrogram.
        type (str): The file format to save the spectrogram in ("npy" or "torch").

        Returns:
        None
        """
        if type == "npy":
            np.save(output_path, S_db)
        elif type == "torch":
            torch.save(torch.tensor(S_db), output_path)
        else:
            raise ValueError("Invalid file type. Choose 'npy' or 'torch'.")
        
    def load_log_mel_spectrogram(self, input_path, type="npy"):
        """
        Loads a log-scaled Mel spectrogram from a file.

        Parameters:
        input_path (str): The file path to load the spectrogram from.
        type (str): The file format to load the spectrogram from ("npy" or "torch").

        Returns:
        np.ndarray: The loaded log-scaled Mel spectrogram in decibels.
        """
        if type == "npy":
            return np.load(input_path)
        elif type == "torch":
            return torch.load(input_path).numpy()
        else:
            raise ValueError("Invalid file type. Choose 'npy' or 'torch'.")

    def save_combined_features(self, features, output_path):
        """
        Saves the combined feature array to a NumPy file.

        Parameters:
        features (np.ndarray): The combined feature array.
        output_path (str): The file path to save the features.

        Returns:
        None
        """
        np.save(output_path, features)

    def add_augmentations(self, y):
        """
        Applies random augmentations to the audio signal, including pitch shift, time stretch, and adding noise.

        Parameters:
        y (np.ndarray): The original audio signal.

        Returns:
        np.ndarray: The augmented audio signal.
        """
        if np.random.rand() < 0.3:
            y = librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=2)
        if np.random.rand() < 0.3:
            y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.95, 1.05))
        if np.random.rand() < 0.3:
            y = self.add_gaussian_noise(y)

        return self.ensure_target_duration(y)
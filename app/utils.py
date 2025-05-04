import torch
import torchaudio
import librosa
import numpy as np

def load_model(path):
    model = torch.load(path, map_location='cpu')
    # model.eval()
    return model

def preprocess_audio(file_path, sr=16000, duration=5):
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db[np.newaxis, np.newaxis, ...]  # (1, 1, freq, time)
    return torch.tensor(mel_db).float()

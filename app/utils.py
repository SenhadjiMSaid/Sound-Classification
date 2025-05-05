import torch
import torchaudio
import librosa
import numpy as np

def load_model(path):
    try:
        # First, import necessary components for the model
        import torch.nn as nn
        
        # Define the exact EfficientResNetAudio class from the notebook
        class EfficientResNetAudio(nn.Module):
            def __init__(self, num_classes=10, input_channels=1):
                super(EfficientResNetAudio, self).__init__()
                # We need to handle the ResNet import differently
                try:
                    import torchvision.models as models
                    self.resnet = models.resnet18(pretrained=False)
                    self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    in_features = self.resnet.fc.in_features
                    self.resnet.fc = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(in_features, num_classes)
                    )
                except ImportError:
                    # If torchvision is not available, raise a clear error
                    raise ImportError("torchvision is required. Please install it with 'pip install torchvision'")
                    
            def forward(self, x):
                return self.resnet(x)
        
        # Try to load the saved model - this handles both full model and state_dict cases
        try:
            # First, try loading as a full model
            model = torch.load(path, map_location='cpu')
            if not isinstance(model, nn.Module):
                # If not a model, it's probably a state dict
                model_instance = EfficientResNetAudio(num_classes=10, input_channels=1)
                model_instance.load_state_dict(model)
                model = model_instance
        except Exception as e:
            print(f"Error while loading model: {e}")
            # Create a new model and load the state dict
            model = EfficientResNetAudio(num_classes=10, input_channels=1)
            model.load_state_dict(torch.load(path, map_location='cpu'))
        
        # Set to evaluation mode
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error in load_model: {e}")
        raise

def preprocess_audio(file_path, sr=16000, duration=5):
    """
    Process audio file to match exactly how it was done during training
    """
    # Match the exact parameters from your training notebook
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    
    # Calculate the expected length based on duration
    target_length = int(sr * duration)
    
    # Ensure consistent length
    if len(y) > target_length:
        # If longer, take a segment (middle segment to avoid silence at beginning/end)
        start = (len(y) - target_length) // 2
        y = y[start:start + target_length]
    else:
        # If shorter, pad with zeros
        y = np.pad(y, (0, max(0, target_length - len(y))))
    
    # Use the same spectrogram parameters as in training
    mel = librosa.feature.melspectrogram(
        y=y, 
        sr=sr,
        n_fft=1024,  # From your CONFIG
        hop_length=512,  # From your CONFIG
        n_mels=128  # From your CONFIG
    )
    
    # Convert to dB scale
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Apply the same normalization as in training
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    
    # Reshape to match the input format expected by the model
    # Adding batch and channel dimensions: [1, 1, freq, time]
    mel_db = mel_db[np.newaxis, np.newaxis, ...]
    
    return torch.tensor(mel_db).float()
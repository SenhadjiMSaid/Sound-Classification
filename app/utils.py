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
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db[np.newaxis, np.newaxis, ...]  # (1, 1, freq, time)
    return torch.tensor(mel_db).float()
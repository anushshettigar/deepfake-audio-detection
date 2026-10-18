"""
Spectral Classifier for DeepShield Ensemble
LFCC + Light CNN classifier for complementary features to RawNet2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np


class LFCCExtractor:
    """LFCC feature extractor matching RawNet2 input expectations."""
    
    def __init__(self, sample_rate=16000, n_lfcc=60, n_fft=512, hop_length=160, win_length=400):
        self.sample_rate = sample_rate
        self.n_lfcc = n_lfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Create linear filterbank manually (librosa doesn't have linear filters)
        # Linear spaced filters from 0 to Nyquist
        freqs = np.linspace(0, sample_rate // 2, n_lfcc + 2)
        fft_freqs = np.linspace(0, sample_rate // 2, n_fft // 2 + 1)
        
        self.filterbank = np.zeros((n_lfcc, n_fft // 2 + 1))
        for i in range(n_lfcc):
            f_low, f_center, f_high = freqs[i], freqs[i+1], freqs[i+2]
            # Triangular filter
            low_idx = np.searchsorted(fft_freqs, f_low)
            center_idx = np.searchsorted(fft_freqs, f_center)
            high_idx = np.searchsorted(fft_freqs, f_high)
            
            if center_idx > low_idx:
                self.filterbank[i, low_idx:center_idx] = np.linspace(0, 1, center_idx - low_idx)
            if high_idx > center_idx:
                self.filterbank[i, center_idx:high_idx] = np.linspace(1, 0, high_idx - center_idx)
    
    def __call__(self, waveform):
        """
        Extract LFCC features.
        Input: waveform [T] or [1, T] numpy or torch
        Output: LFCC [n_lfcc*3, time_frames] (with delta and delta-delta)
        """
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        
        # STFT
        stft = librosa.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window='hann'
        )
        power = np.abs(stft) ** 2
        
        # Apply linear filterbank
        linear_spec = np.dot(self.filterbank, power)
        log_linear = np.log(linear_spec + 1e-10)
        
        # DCT for LFCC (cepstral coefficients)
        lfcc = librosa.feature.mfcc(
            S=log_linear, n_mfcc=self.n_lfcc, dct_type=2
        )
        
        # Delta and delta-delta
        delta = librosa.feature.delta(lfcc)
        delta2 = librosa.feature.delta(lfcc, order=2)
        
        # Concatenate: [n_lfcc*3, time]
        features = np.concatenate([lfcc, delta, delta2], axis=0)
        return features.astype(np.float32)


class SpectralCNN(nn.Module):
    """Light CNN for LFCC features."""
    
    def __init__(self, n_lfcc=60, num_classes=2, dropout=0.3):
        super().__init__()
        
        # Input: [B, 1, n_lfcc*3, time]  (LFCC + delta + delta2 = 180 channels)
        in_channels = n_lfcc * 3
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(dropout)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(dropout)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x: [B, 1, freq, time]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SpectralClassifier:
    """Wrapper for spectral classifier with preprocessing."""
    
    def __init__(self, model_path=None, device='cuda'):
        self.device = torch.device(device)
        self.lfcc_extractor = LFCCExtractor()
        self.model = SpectralCNN().to(self.device)
        
        if model_path:
            self.load(model_path)
        
        self.model.eval()
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    @torch.no_grad()
    def predict(self, waveform):
        """
        Predict from waveform.
        Input: waveform [T] or [1, T] numpy or torch
        Output: probabilities [2], logits [2]
        """
        # Extract LFCC
        lfcc = self.lfcc_extractor(waveform)  # [180, time]
        
        # Convert to tensor
        x = torch.from_numpy(lfcc).float().unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 180, time]
        
        # Forward
        logits = self.model(x)  # [1, 2]
        probs = F.softmax(logits, dim=1)[0]
        
        return probs.cpu().numpy(), logits.cpu().numpy()[0]


def create_spectral_model(device='cuda'):
    """Factory function for training."""
    model = SpectralCNN().to(device)
    return model


if __name__ == "__main__":
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SpectralCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    x = torch.randn(4, 1, 180, 200).to(device)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print("Spectral CNN ready")
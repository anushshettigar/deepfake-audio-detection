"""
Ensemble Model for DeepShield
Combines RawNet2 (waveform) + SpectralCNN (LFCC) for robust detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from backend.models.DeepShield_Deployment.asvspoof5_rawnet2.model import RawNet2
from spectral_classifier import SpectralCNN


class DeepShieldEnsemble(nn.Module):
    """Ensemble of RawNet2 + SpectralCNN with learnable fusion."""
    
    def __init__(self, device, fusion_type='weighted'):
        super().__init__()
        self.device = device
        self.fusion_type = fusion_type
        
        # RawNet2 branch
        self.rawnet2 = RawNet2(device=device).to(device)
        
        # SpectralCNN branch
        self.spectral = SpectralCNN().to(device)
        
        # Fusion layer
        if fusion_type == 'weighted':
            # Learnable weights for each branch
            self.raw_weight = nn.Parameter(torch.tensor(0.6))
            self.spec_weight = nn.Parameter(torch.tensor(0.4))
        elif fusion_type == 'mlp':
            # MLP fusion
            self.fusion_mlp = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            ).to(device)
        else:
            # Simple average
            self.raw_weight = None
            self.spec_weight = None
    
    def forward(self, waveform, lfcc_features=None, is_test=False):
        """
        Forward pass.
        Args:
            waveform: [B, 64000] raw audio
            lfcc_features: [B, 1, 180, time] precomputed LFCC (optional)
            is_test: if True, return softmax probabilities
        Returns:
            emb, logits (or probs if is_test)
        """
        # RawNet2 forward
        emb_raw, logits_raw = self.rawnet2(waveform, is_test=is_test)
        
        # Spectral forward
        if lfcc_features is not None:
            logits_spec = self.spectral(lfcc_features)
        else:
            # In inference, lfcc_features should be provided
            logits_spec = torch.zeros_like(logits_raw)
        
        # Fusion
        if self.fusion_type == 'weighted':
            # Softmax weights to ensure they sum to 1
            w_raw = torch.sigmoid(self.raw_weight)
            w_spec = torch.sigmoid(self.spec_weight)
            total = w_raw + w_spec
            w_raw = w_raw / total
            w_spec = w_spec / total
            
            fused_logits = w_raw * logits_raw + w_spec * logits_spec
        elif self.fusion_type == 'mlp':
            # Concatenate logits from both branches
            combined = torch.cat([logits_raw, logits_spec], dim=1)  # [B, 4]
            fused_logits = self.fusion_mlp(combined)
        else:
            fused_logits = (logits_raw + logits_spec) / 2
        
        if is_test:
            fused_probs = F.softmax(fused_logits, dim=1)
            return emb_raw, fused_probs
        
        return emb_raw, fused_logits
    
    def get_branch_outputs(self, waveform, lfcc_features):
        """Get individual branch outputs for analysis."""
        emb_raw, logits_raw = self.rawnet2(waveform)
        logits_spec = self.spectral(lfcc_features)
        return logits_raw, logits_spec


class EnsembleWrapper:
    """Inference wrapper combining RawNet2 + SpectralCNN checkpoints."""
    
    def __init__(self, rawnet2_checkpoint, spectral_checkpoint, device='cuda', fusion_type='weighted'):
        self.device = torch.device(device)
        self.fusion_type = fusion_type
        
        # Load RawNet2
        self.rawnet2 = RawNet2(device=device).to(device)
        rawnet2_ckpt = torch.load(rawnet2_checkpoint, map_location=device, weights_only=False)
        self.rawnet2.load_state_dict(rawnet2_ckpt['model_state_dict'])
        self.rawnet2.eval()
        
        # Load SpectralCNN
        self.spectral = SpectralCNN().to(device)
        spec_ckpt = torch.load(spectral_checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in spec_ckpt:
            self.spectral.load_state_dict(spec_ckpt['model_state_dict'])
        else:
            self.spectral.load_state_dict(spec_ckpt)
        self.spectral.eval()
        
        # Fusion weights (can be tuned on validation)
        if fusion_type == 'weighted':
            self.raw_weight = 0.6
            self.spec_weight = 0.4
    
    @torch.no_grad()
    def predict(self, waveform, lfcc_features=None):
        """
        Predict from waveform and optional LFCC features.
        Returns: dict with label, confidence, probabilities
        """
        self.rawnet2.eval()
        self.spectral.eval()
        
        # RawNet2
        waveform_t = waveform.to(self.device) if isinstance(waveform, torch.Tensor) else \
                     torch.from_numpy(waveform).float().to(self.device)
        if waveform_t.ndim == 1:
            waveform_t = waveform_t.unsqueeze(0)
        
        _, logits_raw = self.rawnet2(waveform_t, is_test=True)
        probs_raw = F.softmax(logits_raw, dim=1)[0]
        
        # Spectral (if LFCC provided)
        if lfcc_features is not None:
            lfcc_t = lfcc_features.to(self.device) if isinstance(lfcc_features, torch.Tensor) else \
                     torch.from_numpy(lfcc_features).float().unsqueeze(0).to(self.device)
            logits_spec = self.spectral(lfcc_t)
            probs_spec = F.softmax(logits_spec, dim=1)[0]
        else:
            probs_spec = torch.zeros_like(probs_raw)
        
        # Fusion
        if self.fusion_type == 'weighted':
            w_raw = self.raw_weight
            w_spec = self.spec_weight
            total = w_raw + w_spec
            w_raw, w_spec = w_raw/total, w_spec/total
            fused_probs = w_raw * probs_raw + w_spec * probs_spec
        else:
            fused_probs = (probs_raw + probs_spec) / 2
        
        # Decision
        fake_prob = fused_probs[0].item()
        real_prob = fused_probs[1].item()
        
        if fake_prob >= real_prob:
            label = "DEEPFAKE"
            class_id = 0
            confidence = fake_prob
        else:
            label = "REAL"
            class_id = 1
            confidence = real_prob
        
        return {
            "label": label,
            "class_id": class_id,
            "confidence": round(confidence * 100, 2),
            "fake_probability": round(fake_prob * 100, 2),
            "real_probability": round(real_prob * 100, 2),
            "rawnet2_fake": round(probs_raw[0].item() * 100, 2),
            "rawnet2_real": round(probs_raw[1].item() * 100, 2),
            "spectral_fake": round(probs_spec[0].item() * 100, 2),
            "spectral_real": round(probs_spec[1].item() * 100, 2),
        }


def train_spectral_standalone(train_loader, val_loader, device, epochs=20):
    """Train SpectralCNN standalone on LFCC features."""
    from spectral_classifier import LFCCExtractor
    from torch.utils.data import DataLoader
    import pandas as pd
    from deepfake_audio_preprocessing import preprocess_audio_for_training
    from torch.utils.data import Dataset
    
    class SpectralDataset(Dataset):
        def __init__(self, dataframe, augment=True):
            self.df = dataframe.reset_index(drop=True)
            self.augment = augment
            self.lfcc = LFCCExtractor()
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            audio_path = Path(row['audio_path'])
            target = int(row['target'])
            
            waveform = preprocess_audio_for_training(audio_path)
            lfcc = self.lfcc(waveform.numpy())  # [180, time]
            lfcc = torch.from_numpy(lfcc).float()
            
            return lfcc.unsqueeze(0), torch.tensor(target, dtype=torch.long)
    
    # This would be called with actual data loaders
    pass


if __name__ == "__main__":
    # Test ensemble creation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ensemble = DeepShieldEnsemble(device, fusion_type='weighted')
    print(f"Ensemble created on {device}")
    print(f"RawNet2 params: {sum(p.numel() for p in ensemble.rawnet2.parameters()):,}")
    print(f"Spectral params: {sum(p.numel() for p in ensemble.spectral.parameters()):,}")
    print(f"Total params: {sum(p.numel() for p in ensemble.parameters()):,}")
    
    # Test forward
    x = torch.randn(2, 64000).to(device)
    lfcc = torch.randn(2, 1, 180, 200).to(device)
    emb, logits = ensemble(x, lfcc)
    print(f"Output logits: {logits.shape}")
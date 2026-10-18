#!/usr/bin/env python3
"""
DeepShield RawNet2 Training Script

Trains RawNet2 from scratch on expanded dataset with:
- Generator-aware splits
- Balanced sampling
- Realistic augmentation
- Mixed precision training
"""

import os
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "sample_rate": 16000,
    "n_mels": 64,
    "n_fft": 512,
    "hop_length": 160,
    "win_length": 400,
    "duration": 4,  # seconds
    "num_samples": 64000,  # 4 * 16000
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "mixed_precision": True,
    "seed": 42,
    "early_stopping_patience": 10,
    "num_workers": 4,
    "class_weights": [0.5, 2.0],  # Adjust based on data balance
}

# Paths
DATASET_DIR = Path("dataset")
MANIFEST_PATH = DATASET_DIR / "manifest.jsonl"
MODEL_DIR = Path("backend/models/DeepShield_Deployment/models")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Set seeds
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG["seed"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# ============================================================
# RawNet2 Architecture (from ASVspoof 2021 baseline)
# ============================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class RawNet2(nn.Module):
    def __init__(self, device, num_classes=2):
        super().__init__()
        self.device = device
        
        # Frontend: LFCC or Mel spectrogram
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(32, 32, 3, stride=1)
        self.layer2 = self._make_layer(32, 64, 4, stride=2)
        self.layer3 = self._make_layer(64, 128, 6, stride=2)
        self.layer4 = self._make_layer(128, 256, 3, stride=2)
        
        # GRU layer
        # After 3 stride-2 pools: 64 -> 32 -> 16 -> 8 freq bins
        self.gru = nn.GRU(
            input_size=256 * 8,  # 8 = freq bins after pooling
            hidden_size=1024,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Classifier
        self.fc = nn.Linear(2048, num_classes)  # 2048 = 1024 * 2 (bidirectional)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, is_test=False):
        # x: [B, 1, n_mels, time]
        x = self.frontend(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Reshape for GRU: [B, C, F, T] -> [B, T, C*F]
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        
        # GRU
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        
        # Take last timestep
        x = x[:, -1, :]
        
        # Classifier
        logits = self.fc(x)
        
        if is_test:
            return x, F.softmax(logits, dim=1)
        return logits


# ============================================================
# Dataset
# ============================================================

class DeepfakeDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, config: Dict, augment: bool = False):
        self.config = config
        self.split = split
        self.augment = augment and split == "train"
        
        # Load manifest
        self.records = []
        with open(manifest_path) as f:
            for line in f:
                r = json.loads(line)
                if r["split"] == split:
                    self.records.append(r)
        
        logger.info(f"Loaded {len(self.records)} samples for {split}")
        
        # We'll use librosa for audio loading and mel computation
        # to avoid torchaudio/torchcodec issues on Windows
    
    def __len__(self):
        return len(self.records)
    
    def load_audio(self, filepath: Path):
        """Load and preprocess audio using librosa."""
        try:
            import librosa
            waveform, sr = librosa.load(str(filepath), sr=self.config["sample_rate"], mono=True)
            waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        except Exception as e:
            logger.warning(f"Failed to load {filepath} with librosa: {e}")
            try:
                import soundfile as sf
                waveform, sr = sf.read(str(filepath))
                if len(waveform.shape) > 1:
                    waveform = waveform.mean(axis=1)
                if sr != self.config["sample_rate"]:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.config["sample_rate"])
                waveform = torch.from_numpy(waveform).float().unsqueeze(0)
                sr = self.config["sample_rate"]
            except Exception as e2:
                logger.warning(f"Failed to load {filepath} with soundfile: {e2}")
                waveform = torch.zeros(1, self.config["num_samples"])
                sr = self.config["sample_rate"]
        
        # Pad or truncate
        target_len = self.config["num_samples"]
        if waveform.shape[1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_len]
        
        # Peak normalization
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak
        
        return waveform
    
    def compute_mel(self, waveform):
        """Compute mel spectrogram using librosa."""
        import librosa
        import librosa.feature
        
        # Convert to numpy for librosa
        y = waveform.squeeze(0).numpy()
        
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.config["sample_rate"],
            n_fft=self.config["n_fft"],
            hop_length=self.config["hop_length"],
            win_length=self.config["win_length"],
            n_mels=self.config["n_mels"],
            power=2.0,
        )
        
        # Convert to dB
        mel = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        
        # Convert back to torch
        mel = torch.from_numpy(mel).float()
        
        return mel
    
    def __getitem__(self, idx):
        record = self.records[idx]
        filepath = Path(record["filepath"])
        label = record["label"]
        
        waveform = self.load_audio(filepath)
        mel = self.compute_mel(waveform)
        
        # Augmentation
        if self.augment:
            # Random gain
            if random.random() < 0.5:
                gain_db = random.uniform(-6, 6)
                mel = mel + gain_db
            
            # Frequency masking
            if random.random() < 0.3:
                freq_mask_param = 16
                f = mel.shape[0]
                f0 = random.randint(0, max(0, f - freq_mask_param))
                mel[f0:f0+freq_mask_param, :] = 0
            
            # Time masking
            if random.random() < 0.3:
                time_mask_param = 32
                t = mel.shape[1]
                t0 = random.randint(0, max(0, t - time_mask_param))
                mel[:, t0:t0+time_mask_param] = 0
        
        # Add channel dimension: [1, n_mels, time]
        mel = mel.unsqueeze(0)
        
        return mel, torch.tensor(label, dtype=torch.long)


# ============================================================
# Training
# ============================================================

def create_dataloaders(config):
    """Create train/val/test dataloaders with balanced sampling."""
    
    train_dataset = DeepfakeDataset(MANIFEST_PATH, "train", config, augment=True)
    val_dataset = DeepfakeDataset(MANIFEST_PATH, "val", config, augment=False)
    test_dataset = DeepfakeDataset(MANIFEST_PATH, "test", config, augment=False)
    
    # Balanced sampler for training
    train_labels = [r["label"] for r in train_dataset.records]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in train_labels]
    
    sampler = WeightedRandomSampler(
        sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    # Adjust batch size for small datasets
    train_batch_size = min(config["batch_size"], max(1, len(train_dataset)))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=0,  # Use 0 for small datasets to avoid multiprocessing issues
        pin_memory=False,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, scaler, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for mel, target in pbar:
        mel = mel.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        
        if config["mixed_precision"]:
            with torch.amp.autocast(device_type='cuda', enabled=DEVICE.type=='cuda'):
                logits = model(mel)
                loss = criterion(logits, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(mel)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({"loss": loss.item(), "acc": correct/total})
    
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for mel, target in tqdm(loader, desc="Validation"):
            mel = mel.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            
            _, probs = model(mel, is_test=True)
            logits = torch.log(probs + 1e-8)
            loss = criterion(logits, target)
            
            total_loss += loss.item()
            pred = probs.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of class 1 (REAL)
    
    accuracy = correct / total
    avg_loss = total_loss / len(loader)
    
    # Compute metrics
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=[0, 1]
    )
    
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except:
        roc_auc = 0.0
    
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "preds": all_preds,
        "targets": all_targets,
        "probs": all_probs
    }


def main():
    logger.info("=== DeepShield RawNet2 Training ===")
    logger.info(f"Config: {json.dumps(CONFIG, indent=2)}")
    
    # Check manifest exists
    if not MANIFEST_PATH.exists():
        logger.error(f"Manifest not found: {MANIFEST_PATH}")
        logger.info("Run prepare_dataset.py first")
        return
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(CONFIG)
    
    # Model
    model = RawNet2(DEVICE).to(DEVICE)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    class_weights = torch.tensor(CONFIG["class_weights"], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=CONFIG["mixed_precision"] and DEVICE.type=='cuda')
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        logger.info(f"\n=== Epoch {epoch}/{CONFIG['epochs']} ===")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, CONFIG)
        val_metrics = validate(model, val_loader, criterion)
        
        scheduler.step(val_metrics["accuracy"])
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        logger.info(f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        logger.info(f"Val Precision (FAKE/REAL): {val_metrics['precision']}")
        logger.info(f"Val Recall (FAKE/REAL): {val_metrics['recall']}")
        logger.info(f"Val F1 (FAKE/REAL): {val_metrics['f1']}")
        logger.info(f"Val CM: {val_metrics['confusion_matrix']}")
        
        # Save checkpoint
        is_best = val_metrics["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": CONFIG,
            }, CHECKPOINT_DIR / "best_model.pt")
            
            logger.info(f"New best model saved! Val Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "config": CONFIG,
        }, CHECKPOINT_DIR / "last_model.pt")
        
        if patience_counter >= CONFIG["early_stopping_patience"]:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Final evaluation on test set
    logger.info("\n=== Final Test Evaluation ===")
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = validate(model, test_loader, criterion)
    
    logger.info(f"Test Acc: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"Test Precision (FAKE/REAL): {test_metrics['precision']}")
    logger.info(f"Test Recall (FAKE/REAL): {test_metrics['recall']}")
    logger.info(f"Test F1 (FAKE/REAL): {test_metrics['f1']}")
    logger.info(f"Test CM: {test_metrics['confusion_matrix']}")
    
    # Export for deployment
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "DeepShield_RawNet2_new.pt")
    
    # Save metadata
    metadata = {
        "project": "DeepShield",
        "model": "RawNet2",
        "dataset": "Expanded PhonemeDF + ASVspoof + WaveFake + LibriSpeech",
        "sample_rate": CONFIG["sample_rate"],
        "channels": 1,
        "classes": {"0": "FAKE", "1": "REAL"},
        "training_config": CONFIG,
        "test_metrics": test_metrics,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
    }
    
    with open(MODEL_DIR / "model_metadata_new.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\nTraining complete! Best model at epoch {best_epoch}")
    logger.info(f"Model saved to: {MODEL_DIR}/DeepShield_RawNet2_new.pt")


if __name__ == "__main__":
    main()
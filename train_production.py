#!/usr/bin/env python
"""
DeepShield Production Training Script
=====================================
Trains RawNet2 + SpectralCNN ensemble on PhonemeDF + DFADD + evaluation fakes.
Uses DFADD real audios for better real class representation.
"""

import torch
import random
import numpy as np
import time
import json
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd

from backend.models.DeepShield_Deployment.asvspoof5_rawnet2.model import RawNet2
from spectral_classifier import SpectralCNN, LFCCExtractor
from deepfake_audio_preprocessing import preprocess_audio_for_training

# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 16
SAMPLES_PER_EPOCH = 10000
MAX_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
USE_AMP = True

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
EARLY_STOPPING_PATIENCE = 4

VAL_SUBSET_SIZE = 1000
RANDOM_SEED = 42

CHECKPOINT_DIR = Path("ml/models/checkpoints/rawnet2_production")
BEST_RAWNET2 = CHECKPOINT_DIR / "best_rawnet2.pt"
BEST_SPECTRAL = CHECKPOINT_DIR / "best_spectral.pt"
BEST_ENSEMBLE = CHECKPOINT_DIR / "best_ensemble.pt"
HISTORY_FILE = CHECKPOINT_DIR / "training_history.json"

# ============================================================
# UTILITIES
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def print_gpu_memory():
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU: {allocated:.3f}GB allocated, {reserved:.3f}GB reserved")

# ============================================================
# DATASETS
# ============================================================
class ProductionDataset(Dataset):
    """Combined dataset: PhonemeDF + DFADD + evaluation fakes, with DFADD real audios."""
    
    def __init__(self, split="train", augment=True, samples_per_epoch=None):
        self.split = split
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch
        
        # Load PhonemeDF
        phonemedf_root = Path("ml/data/splits/PhonemeDF_2026_clean")
        if split == "train":
            phonemedf_csv = phonemedf_root / "train.csv"
        elif split in {"val", "validation"}:
            phonemedf_csv = phonemedf_root / "val.csv"
        else:
            phonemedf_csv = phonemedf_root / "test.csv"
        
        phonemedf_df = pd.read_csv(phonemedf_csv)
        
        # Load DFADD
        dfadd_root = Path("ml/data/external/DFADD")
        import pyarrow.parquet as pq
        
        dfadd_split = "train" if split == "train" else "valid" if split in {"val", "validation"} else "test"
        
        dfadd_real_paths = []
        dfadd_fake_paths = []
        
        for pf in Path("ml/data/external/DFADD").rglob("*.parquet"):
            pf_obj = pq.ParquetFile(pf)
            for batch in pf_obj.iter_batches(columns=["audio", "audio_name", "split", "label"], batch_size=1000):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    if str(row.get("split")) != dfadd_split:
                        continue
                    label = str(row.get("label", "")).strip().lower()
                    audio_bytes = row.get("audio")
                    if audio_bytes is None:
                        continue
                    if isinstance(audio_bytes, dict):
                        audio_bytes = audio_bytes.get("bytes")
                    if audio_bytes is None:
                        continue
                    
                    # Save to temp and decode
                    import tempfile, subprocess
                    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                        tmp.write(bytes(audio_bytes))
                        tmp_path = tmp.name
                    try:
                        result = subprocess.run(
                            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", tmp_path,
                             "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
                        )
                        if result.returncode == 0 and result.stdout:
                            # Save as wav for faster loading
                            import soundfile as sf
                            waveform = np.frombuffer(result.stdout, dtype=np.float32)
                            save_dir = Path("ml/data/dfadd_cache") / dfadd_split
                            save_dir.mkdir(parents=True, exist_ok=True)
                            wav_path = save_dir / f"{row.get('audio_name', 'audio')}.wav"
                            sf.write(str(wav_path), waveform, 16000)
                            
                            if label in {"real", "bonafide", "genuine", "1"}:
                                dfadd_real_paths.append(str(wav_path))
                            elif label in {"spoofed", "spoof", "fake", "synthetic", "0"}:
                                dfadd_fake_paths.append(str(wav_path))
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
        
        print(f"DFADD {split}: real={len(dfadd_real_paths)}, fake={len(dfadd_fake_paths)}")
        
        # Build combined dataframe
        rows = []
        
        # PhonemeDF
        for _, row in phonemedf_df.iterrows():
            rows.append({
                "audio_path": row["audio_path"],
                "target": int(row["target"]),
                "source": "phonemedf"
            })
        
        # DFADD real
        for p in dfadd_real_paths:
            rows.append({"audio_path": p, "target": 1, "source": "dfadd"})
        
        # DFADD fake
        for p in dfadd_fake_paths:
            rows.append({"audio_path": p, "target": 0, "source": "dfadd"})
        
        # Evaluation fakes (if train)
        if split == "train":
            eval_fake_dir = Path("ml/evaluation/real_world/fake")
            for f in eval_fake_dir.glob("*.mp3"):
                rows.append({"audio_path": str(f.resolve()), "target": 0, "source": "eval"})
        
        self.df = pd.DataFrame(rows)
        print(f"{split} dataset: {len(self.df)} total, fake={(self.df.target==0).sum()}, real={(self.df.target==1).sum()}")
        
        # For balanced sampling
        self.fake_indices = self.df.index[self.df.target == 0].tolist()
        self.real_indices = self.df.index[self.df.target == 1].tolist()
        
        if samples_per_epoch is None:
            self.samples_per_epoch = len(self.df)
        else:
            self.samples_per_epoch = samples_per_epoch
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        if self.split == "train":
            # Balanced sampling
            if random.random() < 0.5:
                idx = random.choice(self.fake_indices)
            else:
                idx = random.choice(self.real_indices)
        else:
            idx = idx % len(self.df)
        
        row = self.df.iloc[idx]
        audio_path = Path(row["audio_path"])
        target = int(row["target"])
        
        waveform = preprocess_audio_for_training(audio_path, augment=self.augment)
        return waveform, torch.tensor(target, dtype=torch.long)


class SpectralDataset(Dataset):
    """Dataset for spectral classifier using LFCC features."""
    
    def __init__(self, split="train", augment=True):
        self.split = split
        self.augment = augment
        self.lfcc = LFCCExtractor()
        
        # Reuse the same data loading logic
        prod_ds = ProductionDataset(split=split, augment=False, samples_per_epoch=None)
        self.df = prod_ds.df
        print(f"Spectral {split}: {len(self.df)} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = Path(row["audio_path"])
        target = int(row["target"])
        
        waveform = preprocess_audio_for_training(audio_path, augment=self.augment)
        lfcc = self.lfcc(waveform.numpy())
        lfcc = torch.from_numpy(lfcc).float()
        
        return lfcc.unsqueeze(0), torch.tensor(target, dtype=torch.long)


# ============================================================
# MODELS
# ============================================================
def build_rawnet2(device):
    model = RawNet2(device=device).to(device)
    return model

def build_spectral(device):
    model = SpectralCNN().to(device)
    return model


# ============================================================
# TRAINING
# ============================================================
def train_rawnet2(model, train_loader, val_loader, device, checkpoint_path):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    
    best_f1 = 0
    patience_counter = 0
    history = []
    
    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, dtype=torch.long, non_blocking=True)
            
            with torch.autocast("cuda", torch.float16, enabled=USE_AMP):
                _, logits = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item()
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)
        
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, dtype=torch.long, non_blocking=True)
                
                with torch.autocast("cuda", torch.float16, enabled=USE_AMP):
                    _, logits = model(x, is_test=True)
                    loss = criterion(logits, y)
                
                val_loss += loss.item()
                pred = logits.argmax(1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
                val_preds.extend(pred.cpu().tolist())
                val_labels.extend(y.cpu().tolist())
        
        val_acc = val_correct / val_total
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        scheduler.step(val_f1)
        
        print(f"Epoch {epoch}: Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} F1={val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1,
            }, checkpoint_path)
            print(f"  >>> Best saved: F1={val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping")
                break
    
    return history


def train_spectral(model, train_loader, val_loader, device, checkpoint_path):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, dtype=torch.long, non_blocking=True)
            
            with torch.autocast("cuda", torch.float16, enabled=USE_AMP):
                logits = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, dtype=torch.long, non_blocking=True)
                
                with torch.autocast("cuda", torch.float16, enabled=USE_AMP):
                    logits = model(x)
                    loss = criterion(logits, y)
                
                val_loss += loss.item()
                pred = logits.argmax(1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
                val_preds.extend(pred.cpu().tolist())
                val_labels.extend(y.cpu().tolist())
        
        val_acc = val_correct / val_total
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        scheduler.step(val_f1)
        
        print(f"Spectral Epoch {epoch}: Val Acc={val_acc:.4f} F1={val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1,
            }, checkpoint_path)
            print(f"  >>> Best spectral saved: F1={val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break
    
    return history


# ============================================================
# ENSEMBLE
# ============================================================
class EnsembleWrapper:
    """Inference wrapper combining RawNet2 + SpectralCNN."""
    
    def __init__(self, rawnet2_checkpoint, spectral_checkpoint, device, raw_weight=0.6, spec_weight=0.4):
        self.device = device
        self.raw_weight = raw_weight
        self.spec_weight = spec_weight
        
        # Load RawNet2
        self.rawnet2 = RawNet2(device=device).to(device)
        rawnet2_ckpt = torch.load(rawnet2_checkpoint, map_location=device, weights_only=False)
        self.rawnet2.load_state_dict(rawnet2_ckpt["model_state_dict"])
        self.rawnet2.eval()
        
        # Load SpectralCNN
        self.spectral = SpectralCNN().to(device)
        spec_ckpt = torch.load(spectral_checkpoint, map_location=device, weights_only=False)
        self.spectral.load_state_dict(spec_ckpt["model_state_dict"])
        self.spectral.eval()
        
        self.lfcc = LFCCExtractor()
        
        total = raw_weight + spec_weight
        self.raw_weight = raw_weight / total
        self.spec_weight = spec_weight / total
    
    @torch.no_grad()
    def predict(self, waveform):
        """Predict from waveform [1, 64000] or [64000]."""
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        
        # RawNet2
        _, logits_raw = self.rawnet2(waveform, is_test=True)
        probs_raw = torch.softmax(logits_raw, dim=1)[0]
        
        # Spectral
        lfcc_feat = self.lfcc(waveform.squeeze().cpu().numpy())
        lfcc_t = torch.from_numpy(lfcc_feat).float().unsqueeze(0).unsqueeze(0).to(self.device)
        logits_spec = self.spectral(lfcc_t)
        probs_spec = torch.softmax(logits_spec, dim=1)[0]
        
        # Fuse
        fused_probs = self.raw_weight * probs_raw + self.spec_weight * probs_spec
        
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


# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(RANDOM_SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Datasets
    print("\n=== BUILDING DATASETS ===")
    train_dataset = ProductionDataset(split="train", augment=True, samples_per_epoch=SAMPLES_PER_EPOCH)
    val_dataset = ProductionDataset(split="validation", augment=False)
    
    # For spectral
    spectral_train = SpectralDataset(split="train", augment=True)
    spectral_val = SpectralDataset(split="validation", augment=False)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    spectral_train_loader = DataLoader(spectral_train, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    spectral_val_loader = DataLoader(spectral_val, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    
    # Train RawNet2
    print("\n=== TRAINING RAWNET2 ===")
    rawnet2 = build_rawnet2(device)
    train_rawnet2(rawnet2, train_loader, val_loader, device, BEST_RAWNET2)
    
    # Train SpectralCNN
    print("\n=== TRAINING SPECTRALCNN ===")
    spectral = build_spectral(device)
    train_spectral(spectral, spectral_train_loader, spectral_val_loader, device, BEST_SPECTRAL)
    
    # Build and test ensemble
    print("\n=== BUILDING ENSEMBLE ===")
    ensemble = EnsembleWrapper(BEST_RAWNET2, BEST_SPECTRAL, device)
    
    # Test on evaluation fakes
    print("\n=== TESTING ON EVALUATION FAKES ===")
    eval_fake_dir = Path("ml/evaluation/real_world/fake")
    
    # ElevenLabs
    elevenlabs_files = list(eval_fake_dir.glob("ElevenLabs*.mp3"))
    print(f"\nElevenLabs ({len(elevenlabs_files)} files):")
    elevenlabs_correct = 0
    for f in sorted(eval_fake_dir.glob("ElevenLabs*.mp3")):
        waveform = preprocess_audio_for_training(f, augment=False)
        result = ensemble.predict(waveform)
        correct = result["label"] == "DEEPFAKE"
        if correct: elevenlabs_correct += 1
        print(f"  {f.name}: {result['label']} ({result['confidence']}%) {'✅' if correct else '❌'}")
    print(f"  ElevenLabs accuracy: {elevenlabs_correct}/{len(elevenlabs_files)}")
    
    # Generic audio files
    audio_files = list(Path("ml/evaluation/real_world/fake").glob("audio *.mp3"))
    audio_correct = 0
    for f in sorted(Path("ml/evaluation/real_world/fake").glob("audio *.mp3")):
        waveform = preprocess_audio_for_training(f, augment=False)
        result = ensemble.predict(waveform)
        correct = result["label"] == "DEEPFAKE"
        if correct: audio_correct += 1
    print(f"\nGeneric audio files: {audio_correct}/{len(audio_files)} DEEPFAKE")
    
    # Save ensemble config
    ensemble_config = {
        "rawnet2_checkpoint": str(BEST_RAWNET2),
        "spectral_checkpoint": str(BEST_SPECTRAL),
        "raw_weight": ensemble.raw_weight,
        "spec_weight": ensemble.spec_weight,
        "fusion_type": "weighted",
    }
    with open(CHECKPOINT_DIR / "ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)
    
    print(f"\n=== PRODUCTION TRAINING COMPLETE ===")
    print(f"RawNet2: {BEST_RAWNET2}")
    print(f"SpectralCNN: {BEST_SPECTRAL}")
    print(f"Ensemble config: {CHECKPOINT_DIR / 'ensemble_config.json'}")
    
    return ensemble


if __name__ == "__main__":
    ensemble = main()
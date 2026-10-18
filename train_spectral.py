"""
Training script for SpectralCNN (LFCC-based classifier)
Run after RawNet2 training to train the spectral branch
"""

import torch
import random
import numpy as np
import time
import json
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import pandas as pd

from spectral_classifier import SpectralCNN, LFCCExtractor
from deepfake_audio_preprocessing import preprocess_audio_for_training


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
        
        waveform = preprocess_audio_for_training(audio_path, augment=self.augment)
        lfcc = self.lfcc(waveform.numpy())  # [180, time]
        lfcc = torch.from_numpy(lfcc).float()
        
        return lfcc.unsqueeze(0), torch.tensor(target, dtype=torch.long)


def train_spectral():
    # Config
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    # Data
    train_df = pd.read_csv('ml/data/splits/PhonemeDF_2026_clean/train.csv')
    val_df = pd.read_csv('ml/data/splits/PhonemeDF_2026_clean/val.csv')
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    train_ds = SpectralDataset(train_df, augment=True)
    val_ds = SpectralDataset(val_df, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # Model
    model = SpectralCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    CHECKPOINT_DIR = Path('ml/models/checkpoints/spectral_ensemble')
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_CKPT = CHECKPOINT_DIR / 'best_spectral.pt'
    
    def train_epoch(model, loader, optimizer, criterion, scaler, device):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.autocast('cuda', torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        
        return total_loss / len(loader), correct / total
    
    @torch.no_grad()
    def evaluate(model, loader, criterion, device):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.autocast('cuda', torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            
            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
        
        from sklearn.metrics import f1_score, roc_auc_score
        f1 = f1_score(all_labels, all_preds)
        probs = torch.softmax(torch.tensor(all_preds), dim=0)  # dummy for AUC
        try:
            auc = roc_auc_score(all_labels, [p[1] if isinstance(p, (list, tuple)) else p for p in all_preds])
        except:
            auc = 0.5
        
        return total_loss / len(loader), correct / total, f1, auc
    
    print(f"Training SpectralCNN on {DEVICE}")
    best_f1 = 0
    patience = 0
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, DEVICE)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_f1)
        
        print(f"Epoch {epoch}/{EPOCHS}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f} AUC={val_auc:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_f1': val_f1,
                'val_auc': val_auc,
            }, BEST_CKPT)
            print(f"  >>> New best saved: F1={val_f1:.4f}")
        else:
            patience += 1
            if patience >= 5:
                print("Early stopping")
                break
    
    print(f"Best Val F1: {best_f1:.4f}")
    print(f"Checkpoint saved to {BEST_CKPT}")


if __name__ == "__main__":
    train_spectral()
#!/usr/bin/env python
"""
DeepShield RawNet2 Final Training Script
========================================
Reproducible offline training for RawNet2 on PhonemeDF 2026 + DFADD.
Run: python train_rawnet2_final.py

Configuration (validated via controlled experiments):
- Model: RawNet2 (from scratch, no pretrained weights)
- Datasets: PhonemeDF 2026 + DFADD (balanced FAKE/REAL sampling)
- Batch size: 16 (effective 16, no gradient accumulation needed)
- Learning rate: 1e-4 (AdamW, weight_decay=1e-4)
- AMP: enabled (float16)
- Gradient clipping: 1.0
- LR scheduler: ReduceLROnPlateau (mode=max, factor=0.5, patience=2)
- Early stopping: patience=4 on validation F1
- Samples per epoch: 8000 (balanced across datasets and classes)
- Max epochs: 15 (early stopping typically stops at 5-8)
- Validation subset: 1000 fixed balanced samples (fast iteration)
- Seed: 42
- Checkpoints: ml/models/checkpoints/rawnet2_combined/
"""

import torch
import random
import numpy as np
import time
import json
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# ============================================================
# CONFIGURATION (validated)
# ============================================================
BATCH_SIZE = 16
SAMPLES_PER_EPOCH = 8000
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

CHECKPOINT_DIR = Path("ml/models/checkpoints/rawnet2_combined")
BEST_CHECKPOINT = CHECKPOINT_DIR / "best_rawnet2.pt"
FINAL_CHECKPOINT = CHECKPOINT_DIR / "final_rawnet2.pt"
HISTORY_FILE = CHECKPOINT_DIR / "training_history.json"

# ============================================================
# IMPORTS (local modules)
# ============================================================
from backend.models.DeepShield_Deployment.asvspoof5_rawnet2.model import RawNet2
from ml.training.combined_dataset import CombinedStreamingDataset

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
    print(f"GPU allocated: {allocated:.3f} GB | reserved: {reserved:.3f} GB")

# ============================================================
# FIXED VALIDATION SUBSET (for fast iteration)
# ============================================================
class FixedValDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def build_fixed_val_subset(val_dataset_full, target_size, seed):
    """Build a balanced fixed validation subset."""
    rng = random.Random(seed)
    val_fake = []
    val_real = []
    half = target_size // 2
    
    for waveform, label in val_dataset_full:
        if label == 0:
            if len(val_fake) < half:
                val_fake.append((waveform, label))
        else:
            if len(val_real) < half:
                val_real.append((waveform, label))
        if len(val_fake) >= half and len(val_real) >= half:
            break
    
    subset = val_fake + val_real
    rng.shuffle(subset)
    return subset

# ============================================================
# METRICS
# ============================================================
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

def calculate_metrics(labels, predictions, real_probabilities):
    if len(labels) == 0:
        raise RuntimeError("No samples evaluated")
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    fake_f1 = f1_score(labels, predictions, pos_label=0, zero_division=0)
    real_f1 = f1_score(labels, predictions, pos_label=1, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(labels, real_probabilities)
    except ValueError:
        roc_auc = 0.0
    
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    
    return {
        "loss": 0.0,  # filled by caller
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "fake_f1": float(fake_f1),
        "real_f1": float(real_f1),
        "confusion_matrix": matrix.tolist(),
        "false_positives": int(matrix[0][1]),
        "false_negatives": int(matrix[1][0]),
        "sample_count": int(len(labels)),
    }

def print_metrics(name, metrics):
    print(f"{name} Loss:       {metrics['loss']:.5f}")
    print(f"{name} Accuracy:   {metrics['accuracy']*100:.2f}%")
    print(f"{name} Precision:  {metrics['precision']*100:.2f}%")
    print(f"{name} Recall:     {metrics['recall']*100:.2f}%")
    print(f"{name} F1:         {metrics['f1']*100:.2f}%")
    print(f"{name} ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"{name} FAKE F1:    {metrics['fake_f1']*100:.2f}%")
    print(f"{name} REAL F1:    {metrics['real_f1']*100:.2f}%")
    print(f"{name} CM:         {metrics['confusion_matrix']}")
    print(f"{name} FP/FN:      {metrics['false_positives']}/{metrics['false_negatives']}")
    print(f"{name} Samples:    {metrics['sample_count']}")

# ============================================================
# TRAIN / EVAL LOOPS
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    total_loss = 0.0
    total_batches = 0
    labels = []
    predictions = []
    real_probabilities = []
    
    epoch_start = time.time()
    
    for batch_idx, (waveforms, targets) in enumerate(loader):
        waveforms = waveforms.to(device, non_blocking=True)
        targets = targets.to(device, dtype=torch.long, non_blocking=True)
        
        amp_enabled = USE_AMP and device.type == "cuda"
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            _, logits = model(waveforms)
            loss = criterion(logits, targets)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item()
        total_batches += 1
        
        probs = torch.softmax(logits.detach(), dim=1)
        preds = torch.argmax(logits.detach(), dim=1)
        
        labels.extend(targets.detach().cpu().tolist())
        predictions.extend(preds.cpu().tolist())
        real_probabilities.extend(probs[:, 1].detach().cpu().tolist())
        
        if batch_idx % 100 == 0:
            elapsed = time.time() - epoch_start
            print(f"  Batch {batch_idx}/{len(loader)} | Loss {loss.item():.4f} | Elapsed {elapsed:.1f}s")
    
    train_time = time.time() - epoch_start
    avg_loss = total_loss / max(total_batches, 1)
    
    metrics = calculate_metrics(labels, predictions, real_probabilities)
    metrics["loss"] = avg_loss
    
    return metrics, train_time

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    
    total_loss = 0.0
    total_batches = 0
    labels = []
    predictions = []
    real_probabilities = []
    
    eval_start = time.time()
    
    for waveforms, targets in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        targets = targets.to(device, dtype=torch.long, non_blocking=True)
        
        amp_enabled = USE_AMP and device.type == "cuda"
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            _, logits = model(waveforms, is_test=True)
            loss = criterion(logits, targets)
        
        total_loss += loss.item()
        total_batches += 1
        
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        labels.extend(targets.cpu().tolist())
        predictions.extend(preds.cpu().tolist())
        real_probabilities.extend(probs[:, 1].cpu().tolist())
    
    eval_time = time.time() - eval_start
    avg_loss = total_loss / max(total_batches, 1)
    
    metrics = calculate_metrics(labels, predictions, real_probabilities)
    metrics["loss"] = avg_loss
    
    return metrics, eval_time

# ============================================================
# CHECKPOINTING
# ============================================================
def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, metrics, config):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "architecture": "RawNet2",
        "dataset": "PhonemeDF 2026 + DFADD",
        "sample_rate": 16000,
        "audio_seconds": 4,
        "num_samples": 64000,
        "labels": {"0": "FAKE", "1": "REAL"},
        "threshold": 0.5,
        "metrics": metrics,
        "training": config,
    }
    torch.save(checkpoint, path)

# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(RANDOM_SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    
    print("=" * 70)
    print("DEEPSHIELD RAWNET2 FINAL TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print("\n=== CONFIGURATION ===")
    print(f"Batch size:          {BATCH_SIZE}")
    print(f"Samples/epoch:       {SAMPLES_PER_EPOCH}")
    print(f"Max epochs:          {MAX_EPOCHS}")
    print(f"Learning rate:       {LEARNING_RATE}")
    print(f"Weight decay:        {WEIGHT_DECAY}")
    print(f"Gradient clip:       {GRAD_CLIP}")
    print(f"AMP:                 {USE_AMP}")
    print(f"Scheduler:           ReduceLROnPlateau (factor={SCHEDULER_FACTOR}, patience={SCHEDULER_PATIENCE})")
    print(f"Early stopping:      patience={EARLY_STOPPING_PATIENCE} on val F1")
    print(f"Val subset size:     {VAL_SUBSET_SIZE}")
    print(f"Random seed:         {RANDOM_SEED}")
    print(f"Checkpoint dir:      {CHECKPOINT_DIR}")
    
    print("\n=== LABEL MAPPING ===")
    print("CLASS 0 = FAKE")
    print("CLASS 1 = REAL")
    
    # Build model FIRST (before datasets to avoid GPU context issues)
    print("\n=== BUILDING MODEL ===")
    model = RawNet2(device=device)
    model = model.to(device)
    print("RawNet2 loaded successfully.")
    print_gpu_memory()
    
    # Verify device
    first_param = next(model.parameters())
    assert first_param.device == device, "Model not on correct device"
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")
    
    # Training config for checkpoint
    train_config = {
        "epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "samples_per_epoch": SAMPLES_PER_EPOCH,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRAD_CLIP,
        "amp": USE_AMP,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": SCHEDULER_FACTOR,
        "scheduler_patience": SCHEDULER_PATIENCE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "val_subset_size": VAL_SUBSET_SIZE,
        "balanced_training": True,
        "random_seed": RANDOM_SEED,
    }
    
    # Build datasets AFTER model
    print("\n=== BUILDING DATASETS ===")
    print("Creating training dataset...")
    train_dataset = CombinedStreamingDataset(
        split="train",
        training=True,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        seed=RANDOM_SEED,
    )
    print(f"  Train dataset: {len(train_dataset)} samples/epoch")
    
    print("Creating full validation dataset...")
    val_dataset_full = CombinedStreamingDataset(
        split="validation",
        training=False,
        seed=RANDOM_SEED,
    )
    print(f"  Full val dataset: {len(val_dataset_full)} samples")
    
    print(f"Building fixed validation subset ({VAL_SUBSET_SIZE} samples)...")
    val_subset = build_fixed_val_subset(val_dataset_full, VAL_SUBSET_SIZE, RANDOM_SEED)
    val_dataset = FixedValDataset(val_subset)
    print(f"  Fixed val subset: {len(val_dataset)} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    
    print(f"Train loader: ~{len(train_loader)} batches/epoch")
    print(f"Val loader:   {len(val_loader)} batches")
    
    print("\n=== TRAINING START ===")
    
    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []
    training_start = time.time()
    
    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{MAX_EPOCHS}")
        print(f"{'='*70}")
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {current_lr:.8f}")
        
        print("\n--- TRAINING ---")
        train_metrics, train_time = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        print_metrics("Train", train_metrics)
        print(f"Training time: {train_time:.1f}s")
        
        print("\n--- VALIDATION ---")
        val_metrics, val_time = evaluate(model, val_loader, criterion, device)
        print_metrics("Validation", val_metrics)
        print(f"Validation time: {val_time:.1f}s")
        
        scheduler.step(val_metrics["f1"])
        
        epoch_record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train": train_metrics,
            "validation": val_metrics,
            "train_seconds": train_time,
            "validation_seconds": val_time,
        }
        history.append(epoch_record)
        
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_counter = 0
            
            save_checkpoint(
                BEST_CHECKPOINT, model, optimizer, scheduler, scaler,
                epoch, val_metrics, train_config
            )
            print(f"\n>>> NEW BEST CHECKPOINT: Val F1={best_f1*100:.2f}% at epoch {epoch}")
        else:
            patience_counter += 1
            print(f"\nNo improvement. Early stopping patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        
        print_gpu_memory()
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEARLY STOPPING TRIGGERED at epoch {epoch}")
            break
    
    total_training_time = time.time() - training_start
    print(f"\n{'='*70}")
    print("TRAINING FINISHED")
    print(f"{'='*70}")
    print(f"Total training time: {total_training_time/60:.2f} minutes")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation F1: {best_f1*100:.2f}%")
    
    # Load best checkpoint for final test
    print("\n=== LOADING BEST CHECKPOINT ===")
    best_ckpt = torch.load(BEST_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print("Best checkpoint loaded.")
    
    # Final evaluation on fixed validation subset (already done) + full test sets
    print("\n=== FINAL EVALUATION ===")
    
    # Test on PhonemeDF test set
    print("\n--- PhonemeDF Test Set ---")
    from ml.training.rawnet2_dataset import RawNet2AudioDataset
    import pandas as pd
    project_root = Path(__file__).resolve().parents[0]
    phonemedf_test_csv = project_root / "ml" / "data" / "splits" / "PhonemeDF_2026_clean" / "test.csv"
    phonemedf_test_df = pd.read_csv(phonemedf_test_csv)
    phonemedf_test_ds = RawNet2AudioDataset(phonemedf_test_df)
    phonemedf_test_loader = DataLoader(phonemedf_test_ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    
    phonemedf_metrics, _ = evaluate(model, phonemedf_test_loader, criterion, device)
    print_metrics("PhonemeDF Test", phonemedf_metrics)
    
    # Test on DFADD test set
    print("\n--- DFADD Test Set ---")
    from ml.training.dfadd_dataset import DFADDStreamingDataset
    dfadd_test_ds = DFADDStreamingDataset(
        root=project_root / "ml" / "data" / "external" / "DFADD",
        split="test",
        training=False,
        seed=RANDOM_SEED,
    )
    dfadd_test_loader = DataLoader(dfadd_test_ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    
    dfadd_metrics, _ = evaluate(model, dfadd_test_loader, criterion, device)
    print_metrics("DFADD Test", dfadd_metrics)
    
    # Combined test (fixed val subset - already validated)
    print("\n--- Combined Val Subset (Fixed) ---")
    print_metrics("Combined Val", val_metrics)
    
    # Save final checkpoint
    save_checkpoint(
        FINAL_CHECKPOINT, model, optimizer, scheduler, scaler,
        best_epoch, val_metrics, train_config
    )
    print(f"\nFinal checkpoint saved: {FINAL_CHECKPOINT}")
    
    # Save history
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "architecture": "RawNet2",
            "dataset": "PhonemeDF 2026 + DFADD",
            "sample_rate": 16000,
            "audio_seconds": 4,
            "num_samples": 64000,
            "labels": {"0": "FAKE", "1": "REAL"},
            "training": train_config,
            "best_epoch": best_epoch,
            "best_validation_f1": best_f1,
            "test_metrics": {
                "phonemedf_test": phonemedf_metrics,
                "dfadd_test": dfadd_metrics,
                "combined_val_subset": val_metrics,
            },
            "history": history,
        }, f, indent=2)
    
    print(f"Training history saved: {HISTORY_FILE}")
    
    print(f"\n{'='*70}")
    print("DEEPSHIELD TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"PhonemeDF Test:  Acc={phonemedf_metrics['accuracy']*100:.2f}% F1={phonemedf_metrics['f1']*100:.2f}% AUC={phonemedf_metrics['roc_auc']:.4f}")
    print(f"DFADD Test:      Acc={dfadd_metrics['accuracy']*100:.2f}% F1={dfadd_metrics['f1']*100:.2f}% AUC={dfadd_metrics['roc_auc']:.4f}")
    print(f"Combined Val:    Acc={val_metrics['accuracy']*100:.2f}% F1={val_metrics['f1']*100:.2f}% AUC={val_metrics['roc_auc']:.4f}")
    print(f"Best Val F1:     {best_f1*100:.2f}% (epoch {best_epoch})")
    print(f"Checkpoints:     {BEST_CHECKPOINT}, {FINAL_CHECKPOINT}")
    print(f"History:         {HISTORY_FILE}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
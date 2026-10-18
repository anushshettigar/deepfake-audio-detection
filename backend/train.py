from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from detector.preprocessing import extract_features


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
DATASET_DIR = BACKEND_DIR.parent / "dataset"
MODEL_DIR = BACKEND_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

REAL_DIR = DATASET_DIR / "real"
FAKE_DIR = DATASET_DIR / "fake"


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".webm"
}


# ============================================================
# DATASET
# ============================================================

class AudioDataset(Dataset):

    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file_path = self.files[index]
        label = self.labels[index]

        features = extract_features(file_path)

        # MFCC features:
        # 120 x time
        features = torch.tensor(
            features,
            dtype=torch.float32
        )

        # Add channel dimension
        features = features.unsqueeze(0)

        return features, torch.tensor(
            label,
            dtype=torch.long
        )


# ============================================================
# COLLECT AUDIO
# ============================================================

def collect_files():

    files = []
    labels = []

    for directory, label in [
        (REAL_DIR, 0),
        (FAKE_DIR, 1)
    ]:

        if not directory.exists():
            continue

        for file_path in directory.rglob("*"):

            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in AUDIO_EXTENSIONS
            ):
                files.append(file_path)
                labels.append(label)

    return files, labels


# ============================================================
# MODEL
# ============================================================

class DeepfakeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                32,
                2
            )
        )

    def forward(self, x):

        x = self.network(x)

        return self.classifier(x)


# ============================================================
# TRAINING
# ============================================================

def train():

    files, labels = collect_files()

    print()
    print("========================================")
    print(" DeepShield Audio Detection Training")
    print("========================================")
    print()

    print(f"Real audio : {labels.count(0)}")
    print(f"Fake audio : {labels.count(1)}")
    print(f"Total      : {len(files)}")
    print()

    if len(files) < 10:

        print(
            "ERROR: Not enough audio files."
        )

        print(
            "Add genuine audio to:"
        )

        print(REAL_DIR)

        print(
            "Add deepfake audio to:"
        )

        print(FAKE_DIR)

        return

    dataset = AudioDataset(
        files,
        labels
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Training device: {device}"
    )

    model = DeepfakeCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0

        correct = 0
        total = 0

        for features, targets in loader:

            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(features)

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

        accuracy = (
            correct / total
        ) * 100

        average_loss = (
            total_loss /
            len(loader)
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{EPOCHS} "
            f"| Loss: {average_loss:.4f} "
            f"| Accuracy: {accuracy:.2f}%"
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        MODEL_DIR /
        "deepfake_cnn.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    metadata = {

        "model": "DeepShieldCNN",

        "classes": {
            "0": "REAL",
            "1": "DEEPFAKE"
        },

        "sample_rate": 16000,

        "duration": 5,

        "features": "MFCC + delta + delta-delta",

        "epochs": EPOCHS,

        "final_accuracy": accuracy
    }

    metadata_path = (
        MODEL_DIR /
        "model_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    print()
    print("========================================")
    print(" TRAINING COMPLETE")
    print("========================================")
    print()
    print(
        f"Model saved to:"
    )
    print(model_path)
    print()
    print(
        f"Metadata saved to:"
    )
    print(metadata_path)
    print()


if __name__ == "__main__":
    train()
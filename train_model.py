from pathlib import Path
import random
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from detector.preprocessing import extract_features


BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    BASE_DIR
    / "LA"
    / "ASVspoof2019_LA_train"
    / "flac"
)

PROTOCOL_FILE = (
    BASE_DIR
    / "LA"
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.train.trn.txt"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_FILE = MODEL_DIR / "deepfake_audio_model.pt"
SCALER_FILE = MODEL_DIR / "feature_scaler.json"

MAX_SAMPLES_PER_CLASS = 3000
RANDOM_SEED = 42

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class AudioClassifier(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)


def read_protocol():

    if not PROTOCOL_FILE.exists():
        raise FileNotFoundError(
            f"Protocol file not found:\n{PROTOCOL_FILE}"
        )

    records = []

    with PROTOCOL_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            filename = parts[1]

            label = parts[-1].lower()

            if label not in {
                "bonafide",
                "spoof"
            }:
                continue

            records.append(
                (
                    filename,
                    label
                )
            )

    return records


def build_dataset(records):

    real_files = []
    fake_files = []

    for filename, label in records:

        audio_file = (
            DATASET_DIR
            / f"{filename}.flac"
        )

        if not audio_file.exists():
            continue

        if label == "bonafide":
            real_files.append(audio_file)

        elif label == "spoof":
            fake_files.append(audio_file)

    random.shuffle(real_files)
    random.shuffle(fake_files)

    real_files = real_files[
        :MAX_SAMPLES_PER_CLASS
    ]

    fake_files = fake_files[
        :MAX_SAMPLES_PER_CLASS
    ]

    print(
        f"REAL files selected: {len(real_files)}"
    )

    print(
        f"DEEPFAKE files selected: {len(fake_files)}"
    )

    features = []
    labels = []

    total = (
        len(real_files)
        + len(fake_files)
    )

    current = 0

    for audio_file in real_files:

        current += 1

        print(
            f"Extracting features "
            f"{current}/{total}: "
            f"{audio_file.name}"
        )

        try:

            feature_matrix = extract_features(
                str(audio_file)
            )

            feature_vector = np.concatenate(
                [
                    feature_matrix.mean(axis=1),
                    feature_matrix.std(axis=1)
                ]
            )

            features.append(
                feature_vector
            )

            labels.append(0)

        except Exception as error:

            print(
                f"Skipped {audio_file.name}: "
                f"{error}"
            )

    for audio_file in fake_files:

        current += 1

        print(
            f"Extracting features "
            f"{current}/{total}: "
            f"{audio_file.name}"
        )

        try:

            feature_matrix = extract_features(
                str(audio_file)
            )

            feature_vector = np.concatenate(
                [
                    feature_matrix.mean(axis=1),
                    feature_matrix.std(axis=1)
                ]
            )

            features.append(
                feature_vector
            )

            labels.append(1)

        except Exception as error:

            print(
                f"Skipped {audio_file.name}: "
                f"{error}"
            )

    if not features:
        raise RuntimeError(
            "No usable audio files were found."
        )

    X = np.asarray(
        features,
        dtype=np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int64
    )

    return X, y


def split_dataset(X, y):

    indices = np.arange(
        len(X)
    )

    np.random.shuffle(indices)

    split_index = int(
        len(indices) * 0.80
    )

    train_indices = indices[
        :split_index
    ]

    test_indices = indices[
        split_index:
    ]

    return (
        X[train_indices],
        y[train_indices],
        X[test_indices],
        y[test_indices]
    )


def normalize_features(
    X_train,
    X_test
):

    mean = X_train.mean(
        axis=0
    )

    std = X_train.std(
        axis=0
    )

    std[std < 1e-8] = 1.0

    X_train = (
        X_train - mean
    ) / std

    X_test = (
        X_test - mean
    ) / std

    scaler = {
        "mean": mean.tolist(),
        "std": std.tolist()
    }

    return (
        X_train,
        X_test,
        scaler
    )


def train_model(
    X_train,
    y_train,
    X_test,
    y_test
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nTraining device: {device}"
    )

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        ),
        torch.tensor(
            y_train,
            dtype=torch.long
        )
    )

    test_dataset = TensorDataset(
        torch.tensor(
            X_test,
            dtype=torch.float32
        ),
        torch.tensor(
            y_test,
            dtype=torch.long
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = AudioClassifier(
        X_train.shape[1]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * inputs.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

        train_loss = (
            total_loss / total
        )

        train_accuracy = (
            correct / total
        ) * 100

        model.eval()

        test_correct = 0
        test_total = 0

        with torch.no_grad():

            for inputs, targets in test_loader:

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                predictions = (
                    outputs.argmax(
                        dim=1
                    )
                )

                test_correct += (
                    predictions == targets
                ).sum().item()

                test_total += (
                    targets.size(0)
                )

        test_accuracy = (
            test_correct / test_total
        ) * 100

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Test Accuracy: {test_accuracy:.2f}%"
        )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_size":
                X_train.shape[1],

            "classes": {
                "0": "REAL",
                "1": "DEEPFAKE"
            }
        },
        MODEL_FILE
    )

    return model


def main():

    print("=" * 60)
    print("DeepShield Audio Detection Model Training")
    print("=" * 60)

    print(
        f"\nDataset:\n{DATASET_DIR}"
    )

    print(
        f"\nProtocol:\n{PROTOCOL_FILE}"
    )

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"\nTraining audio directory not found:\n"
            f"{DATASET_DIR}"
        )

    records = read_protocol()

    print(
        f"\nProtocol records found: "
        f"{len(records)}"
    )

    X, y = build_dataset(
        records
    )

    print(
        f"\nFeature matrix shape: "
        f"{X.shape}"
    )

    print(
        f"Labels shape: "
        f"{y.shape}"
    )

    X_train, y_train, X_test, y_test = (
        split_dataset(
            X,
            y
        )
    )

    (
        X_train,
        X_test,
        scaler
    ) = normalize_features(
        X_train,
        X_test
    )

    with SCALER_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            scaler,
            file
        )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    model = train_model(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"\nModel saved to:\n{MODEL_FILE}"
    )

    print(
        f"\nScaler saved to:\n{SCALER_FILE}"
    )

    print(
        "\nClasses:"
    )

    print(
        "0 = REAL"
    )

    print(
        "1 = DEEPFAKE"
    )


if __name__ == "__main__":
    main()
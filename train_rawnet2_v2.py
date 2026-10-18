import os

os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True"
)

import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.models.DeepShield_Deployment.asvspoof5_rawnet2.model import RawNet2
from ml.training.combined_dataset import CombinedStreamingDataset


SAMPLE_RATE = 16000
AUDIO_SECONDS = 4
NUM_SAMPLES = SAMPLE_RATE * AUDIO_SECONDS

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
EFFECTIVE_BATCH_SIZE = (
    BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
)

SAMPLES_PER_EPOCH = 40000
MAX_EPOCHS = 15

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 4
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5

GRAD_CLIP = 5.0

NUM_WORKERS = 0

RANDOM_SEED = 42

USE_AMP = True

CHECKPOINT_DIR = Path(
    "ml/models/checkpoints/rawnet2_combined"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "best_rawnet2.pt"
)

LATEST_CHECKPOINT = (
    CHECKPOINT_DIR / "latest_rawnet2.pt"
)

FINAL_CHECKPOINT = (
    CHECKPOINT_DIR / "final_rawnet2.pt"
)

HISTORY_FILE = (
    CHECKPOINT_DIR / "training_history.json"
)

CLASS_NAMES = {
    0: "FAKE",
    1: "REAL",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def get_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda:0")

    return torch.device("cpu")


def clear_cuda_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_cuda_memory(stage):
    if not torch.cuda.is_available():
        return

    free_memory, total_memory = (
        torch.cuda.mem_get_info(0)
    )

    allocated = (
        torch.cuda.memory_allocated(0)
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved(0)
        / (1024 ** 3)
    )

    free = (
        free_memory
        / (1024 ** 3)
    )

    total = (
        total_memory
        / (1024 ** 3)
    )

    print(
        f"\n=== CUDA MEMORY: {stage} ==="
    )

    print(
        "Total:",
        f"{total:.2f} GB"
    )

    print(
        "Free:",
        f"{free:.2f} GB"
    )

    print(
        "PyTorch allocated:",
        f"{allocated:.2f} GB"
    )

    print(
        "PyTorch reserved:",
        f"{reserved:.2f} GB"
    )


def print_system_information(device):
    print("\n=== SYSTEM ===")

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():
        print(
            "CUDA version:",
            torch.version.cuda
        )

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        properties = (
            torch.cuda.get_device_properties(0)
        )

        print(
            "GPU VRAM:",
            round(
                properties.total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )

        print(
            "Current CUDA device:",
            torch.cuda.current_device()
        )

    print(
        "Selected device:",
        device
    )


def get_model_devices(model):
    devices = set()

    for parameter in model.parameters():
        devices.add(
            parameter.device
        )

    for buffer in model.buffers():
        devices.add(
            buffer.device
        )

    return devices


def verify_model_device(model, device):
    print(
        "\n=== MODEL DEVICE CHECK ==="
    )

    if device.type == "cuda":
        selected_device = torch.device(
            "cuda",
            torch.cuda.current_device()
        )
    else:
        selected_device = torch.device(
            "cpu"
        )

    model_devices = get_model_devices(
        model
    )

    print(
        "Model parameter/buffer devices:",
        model_devices
    )

    print(
        "Selected device:",
        selected_device
    )

    if not model_devices:
        raise RuntimeError(
            "RawNet2 contains no parameters "
            "or buffers."
        )

    for model_device in model_devices:
        if model_device.type != (
            selected_device.type
        ):
            raise RuntimeError(
                "RawNet2 device type mismatch.\n"
                f"Model device: {model_device}\n"
                f"Selected device: "
                f"{selected_device}"
            )

        if model_device.type == "cuda":
            if (
                model_device.index
                != selected_device.index
            ):
                raise RuntimeError(
                    "RawNet2 CUDA device mismatch.\n"
                    f"Model device: "
                    f"{model_device}\n"
                    f"Selected device: "
                    f"{selected_device}"
                )

    print(
        "MODEL DEVICE CHECK PASSED"
    )


def get_logits(model_output):
    if isinstance(
        model_output,
        (tuple, list)
    ):
        if len(model_output) < 2:
            raise RuntimeError(
                "RawNet2 returned fewer "
                "than two outputs."
            )

        logits = model_output[1]

    else:
        logits = model_output

    if not torch.is_tensor(logits):
        raise RuntimeError(
            "RawNet2 classification output "
            "is not a tensor."
        )

    if logits.ndim != 2:
        raise RuntimeError(
            "Expected 2D logits, got "
            f"{logits.shape}."
        )

    if logits.shape[1] != 2:
        raise RuntimeError(
            "Expected two classes, got "
            f"{logits.shape}."
        )

    return logits


def validate_batch(
    waveforms,
    labels,
    device
):
    if not torch.is_tensor(
        waveforms
    ):
        raise RuntimeError(
            "Dataset returned non-tensor "
            "waveforms."
        )

    if not torch.is_tensor(
        labels
    ):
        raise RuntimeError(
            "Dataset returned non-tensor "
            "labels."
        )

    if waveforms.ndim != 2:
        raise RuntimeError(
            "Expected waveform shape "
            "[B, 64000], got "
            f"{waveforms.shape}."
        )

    if waveforms.shape[1] != (
        NUM_SAMPLES
    ):
        raise RuntimeError(
            f"Expected {NUM_SAMPLES} samples, "
            f"got {waveforms.shape[1]}."
        )

    labels = labels.view(-1).long()

    if waveforms.shape[0] != (
        labels.shape[0]
    ):
        raise RuntimeError(
            "Waveform and label batch "
            "sizes do not match."
        )

    waveforms = waveforms.float()

    if not torch.isfinite(
        waveforms
    ).all():
        raise RuntimeError(
            "Non-finite waveform values "
            "detected."
        )

    if torch.any(labels < 0):
        raise RuntimeError(
            "Negative label detected."
        )

    if torch.any(labels > 1):
        raise RuntimeError(
            "Invalid class label detected."
        )

    waveforms = waveforms.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True
    )

    labels = labels.to(
        device=device,
        dtype=torch.long,
        non_blocking=True
    )

    return waveforms, labels


def create_dataset(
    split,
    training
):
    if training:
        samples_per_epoch = (
            SAMPLES_PER_EPOCH
        )
    else:
        samples_per_epoch = None

    return CombinedStreamingDataset(
        split=split,
        training=training,
        samples_per_epoch=(
            samples_per_epoch
        ),
        seed=RANDOM_SEED,
    )


def create_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(device):
    print(
        "\n=== BUILDING MODEL ==="
    )

    clear_cuda_memory()

    print_cuda_memory(
        "BEFORE RAWNET2"
    )

    print(
        "\nCreating RawNet2 directly "
        f"on {device}..."
    )

    try:
        model = RawNet2(
            device=device
        )
    except RuntimeError as exc:
        if "out of memory" in str(
            exc
        ).lower():
            clear_cuda_memory()

            print_cuda_memory(
                "RAWNET2 CONSTRUCTOR OOM"
            )

        raise

    print(
        "RawNet2 loaded successfully."
    )

    print_cuda_memory(
        "AFTER RAWNET2"
    )

    verify_model_device(
        model,
        device
    )

    return model


@torch.no_grad()
def rawnet2_smoke_test(
    model,
    device
):
    print(
        "\n=== RAWNET2 SMOKE TEST ==="
    )

    model.eval()

    waveforms = torch.zeros(
        1,
        NUM_SAMPLES,
        dtype=torch.float32,
        device=device
    )

    print(
        "Input:",
        waveforms.shape
    )

    print(
        "Input device:",
        waveforms.device
    )

    output = model(
        waveforms,
        is_test=False
    )

    logits = get_logits(
        output
    )

    probabilities = torch.softmax(
        logits.float(),
        dim=1
    )

    print(
        "Logits:",
        logits.shape
    )

    print(
        "Probabilities:",
        probabilities.shape
    )

    if logits.shape != (
        1,
        2
    ):
        raise RuntimeError(
            "Unexpected RawNet2 output "
            f"shape: {logits.shape}"
        )

    if not torch.isfinite(
        logits
    ).all():
        raise RuntimeError(
            "RawNet2 produced non-finite "
            "logits."
        )

    sums = probabilities.sum(
        dim=1
    )

    print(
        "Probability sums:",
        sums
    )

    if not torch.allclose(
        sums,
        torch.ones_like(sums),
        atol=1e-5
    ):
        raise RuntimeError(
            "Probability sums are invalid."
        )

    print(
        "RAWNET2 SMOKE TEST PASSED"
    )

    model.train()


def create_optimizer(
    model
):
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


def create_scheduler(
    optimizer
):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=1e-7,
    )


def create_scaler(
    device
):
    if device.type != "cuda":
        return None

    return torch.amp.GradScaler(
        "cuda",
        enabled=True
    )


def amp_smoke_test(
    model,
    optimizer,
    criterion,
    device
):
    if device.type != "cuda":
        return False

    print(
        "\n=== AMP SMOKE TEST ==="
    )

    scaler = create_scaler(
        device
    )

    waveforms = torch.zeros(
        1,
        NUM_SAMPLES,
        dtype=torch.float32,
        device=device
    )

    labels = torch.zeros(
        1,
        dtype=torch.long,
        device=device
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    try:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True
        ):
            output = model(
                waveforms,
                is_test=False
            )

            logits = get_logits(
                output
            )

            loss = criterion(
                logits.float(),
                labels
            )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "AMP produced non-finite loss."
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        optimizer.zero_grad(
            set_to_none=True
        )

        clear_cuda_memory()

        print(
            "AMP SMOKE TEST PASSED"
        )

        return True

    except Exception as exc:
        optimizer.zero_grad(
            set_to_none=True
        )

        clear_cuda_memory()

        print(
            "\nAMP smoke test failed:"
        )

        print(
            repr(exc)
        )

        print(
            "Continuing with FP32."
        )

        return False


def calculate_roc_auc(
    labels,
    fake_probabilities
):
    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    scores = np.asarray(
        fake_probabilities,
        dtype=np.float64
    )

    positive = (
        labels == 0
    )

    negative = (
        labels == 1
    )

    positive_count = int(
        positive.sum()
    )

    negative_count = int(
        negative.sum()
    )

    if (
        positive_count == 0
        or negative_count == 0
    ):
        return None

    order = np.argsort(
        -scores,
        kind="mergesort"
    )

    sorted_labels = (
        labels[order]
    )

    sorted_scores = (
        scores[order]
    )

    tp = 0
    fp = 0

    previous_score = None

    points = [
        (0.0, 0.0)
    ]

    for label, score in zip(
        sorted_labels,
        sorted_scores
    ):
        if (
            previous_score is not None
            and score != previous_score
        ):
            points.append(
                (
                    fp / negative_count,
                    tp / positive_count
                )
            )

        if label == 0:
            tp += 1
        else:
            fp += 1

        previous_score = score

    points.append(
        (
            fp / negative_count,
            tp / positive_count
        )
    )

    auc = 0.0

    for i in range(
        1,
        len(points)
    ):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]

        auc += (
            (x2 - x1)
            * (y1 + y2)
            / 2.0
        )

    return float(
        max(
            0.0,
            min(
                1.0,
                auc
            )
        )
    )


def calculate_metrics(
    labels,
    predictions,
    fake_probabilities
):
    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    predictions = np.asarray(
        predictions,
        dtype=np.int64
    )

    fake_probabilities = np.asarray(
        fake_probabilities,
        dtype=np.float64
    )

    total = len(labels)

    if total == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": None,
            "fake_precision": 0.0,
            "fake_recall": 0.0,
            "fake_f1": 0.0,
            "real_precision": 0.0,
            "real_recall": 0.0,
            "real_f1": 0.0,
            "false_positives": 0,
            "false_negatives": 0,
            "sample_count": 0,
            "confusion_matrix": [
                [0, 0],
                [0, 0]
            ],
        }

    real_tp = int(
        np.sum(
            (labels == 1)
            & (predictions == 1)
        )
    )

    real_fp = int(
        np.sum(
            (labels == 0)
            & (predictions == 1)
        )
    )

    real_fn = int(
        np.sum(
            (labels == 1)
            & (predictions == 0)
        )
    )

    real_tn = int(
        np.sum(
            (labels == 0)
            & (predictions == 0)
        )
    )

    fake_tp = real_tn
    fake_fp = real_fn
    fake_fn = real_fp

    real_precision = (
        real_tp
        / (real_tp + real_fp)
        if real_tp + real_fp > 0
        else 0.0
    )

    real_recall = (
        real_tp
        / (real_tp + real_fn)
        if real_tp + real_fn > 0
        else 0.0
    )

    real_f1 = (
        2
        * real_precision
        * real_recall
        / (real_precision + real_recall)
        if real_precision + real_recall > 0
        else 0.0
    )

    fake_precision = (
        fake_tp
        / (fake_tp + fake_fp)
        if fake_tp + fake_fp > 0
        else 0.0
    )

    fake_recall = (
        fake_tp
        / (fake_tp + fake_fn)
        if fake_tp + fake_fn > 0
        else 0.0
    )

    fake_f1 = (
        2
        * fake_precision
        * fake_recall
        / (fake_precision + fake_recall)
        if fake_precision + fake_recall > 0
        else 0.0
    )

    accuracy = (
        real_tp + real_tn
    ) / total

    precision = (
        fake_precision
        + real_precision
    ) / 2.0

    recall = (
        fake_recall
        + real_recall
    ) / 2.0

    f1 = (
        fake_f1
        + real_f1
    ) / 2.0

    roc_auc = calculate_roc_auc(
        labels,
        fake_probabilities
    )

    return {
        "loss": 0.0,
        "accuracy": float(
            accuracy
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),
        "fake_precision": float(
            fake_precision
        ),
        "fake_recall": float(
            fake_recall
        ),
        "fake_f1": float(
            fake_f1
        ),
        "real_precision": float(
            real_precision
        ),
        "real_recall": float(
            real_recall
        ),
        "real_f1": float(
            real_f1
        ),
        "false_positives": int(
            real_fp
        ),
        "false_negatives": int(
            real_fn
        ),
        "sample_count": int(
            total
        ),
        "confusion_matrix": [
            [
                real_tn,
                real_fp
            ],
            [
                real_fn,
                real_tp
            ]
        ],
    }


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    amp_enabled
):
    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0
    sample_count = 0

    labels_all = []
    predictions_all = []
    fake_probabilities_all = []

    batch_count = 0
    optimizer_steps = 0

    start = time.time()

    for batch_index, batch in enumerate(
        loader,
        start=1
    ):
        waveforms, labels = batch

        waveforms, labels = (
            validate_batch(
                waveforms,
                labels,
                device
            )
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                amp_enabled
                and device.type == "cuda"
            )
        ):
            output = model(
                waveforms,
                is_test=False
            )

            logits = get_logits(
                output
            )

            loss = criterion(
                logits.float(),
                labels
            )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Non-finite loss at "
                f"epoch {epoch}, "
                f"batch {batch_index}."
            )

        scaled_loss = (
            loss
            / GRADIENT_ACCUMULATION_STEPS
        )

        if scaler is not None:
            scaler.scale(
                scaled_loss
            ).backward()
        else:
            scaled_loss.backward()

        if (
            batch_index
            % GRADIENT_ACCUMULATION_STEPS
            == 0
        ):
            if scaler is not None:
                scaler.unscale_(
                    optimizer
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            if scaler is not None:
                scaler.step(
                    optimizer
                )

                scaler.update()

            else:
                optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            optimizer_steps += 1

        probabilities = (
            torch.softmax(
                logits.float(),
                dim=1
            )
        )

        predictions = (
            torch.argmax(
                probabilities,
                dim=1
            )
        )

        fake_probabilities = (
            probabilities[:, 0]
        )

        batch_size = (
            labels.shape[0]
        )

        running_loss += (
            float(
                loss.detach().cpu()
            )
            * batch_size
        )

        sample_count += (
            batch_size
        )

        labels_all.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        predictions_all.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        fake_probabilities_all.extend(
            fake_probabilities.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        batch_count += 1

        if batch_count == 1:
            print(
                "\nFirst training batch:"
            )

            print(
                "Waveforms:",
                tuple(
                    waveforms.shape
                )
            )

            print(
                "Device:",
                waveforms.device
            )

            print(
                "Logits:",
                tuple(
                    logits.shape
                )
            )

            print(
                "Loss:",
                float(
                    loss.detach().cpu()
                )
            )

        if batch_count % 500 == 0:
            elapsed = (
                time.time()
                - start
            )

            print(
                f"Epoch {epoch} | "
                f"batch {batch_count} | "
                f"samples {sample_count} | "
                f"loss "
                f"{running_loss / sample_count:.5f} | "
                f"time "
                f"{elapsed / 60:.1f} min"
            )

    if (
        batch_count
        % GRADIENT_ACCUMULATION_STEPS
        != 0
    ):
        if scaler is not None:
            scaler.unscale_(
                optimizer
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        if scaler is not None:
            scaler.step(
                optimizer
            )

            scaler.update()

        else:
            optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        optimizer_steps += 1

    metrics = calculate_metrics(
        labels_all,
        predictions_all,
        fake_probabilities_all
    )

    metrics["loss"] = (
        running_loss
        / max(
            sample_count,
            1
        )
    )

    metrics[
        "optimizer_steps"
    ] = optimizer_steps

    metrics[
        "batch_count"
    ] = batch_count

    metrics[
        "epoch_time_seconds"
    ] = (
        time.time()
        - start
    )

    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    name,
    amp_enabled
):
    model.eval()

    print(
        f"\n=== {name.upper()} EVALUATION ==="
    )

    running_loss = 0.0
    sample_count = 0

    labels_all = []
    predictions_all = []
    fake_probabilities_all = []

    start = time.time()

    for batch_index, batch in enumerate(
        loader,
        start=1
    ):
        waveforms, labels = batch

        waveforms, labels = (
            validate_batch(
                waveforms,
                labels,
                device
            )
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                amp_enabled
                and device.type == "cuda"
            )
        ):
            output = model(
                waveforms,
                is_test=False
            )

            logits = get_logits(
                output
            )

            loss = criterion(
                logits.float(),
                labels
            )

        probabilities = (
            torch.softmax(
                logits.float(),
                dim=1
            )
        )

        predictions = (
            torch.argmax(
                probabilities,
                dim=1
            )
        )

        fake_probabilities = (
            probabilities[:, 0]
        )

        batch_size = (
            labels.shape[0]
        )

        running_loss += (
            float(
                loss.detach().cpu()
            )
            * batch_size
        )

        sample_count += (
            batch_size
        )

        labels_all.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )

        predictions_all.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

        fake_probabilities_all.extend(
            fake_probabilities.cpu()
            .numpy()
            .tolist()
        )

        if batch_index % 500 == 0:
            print(
                f"{name}: "
                f"{sample_count} samples processed"
            )

    metrics = calculate_metrics(
        labels_all,
        predictions_all,
        fake_probabilities_all
    )

    metrics["loss"] = (
        running_loss
        / max(
            sample_count,
            1
        )
    )

    metrics[
        "evaluation_time_seconds"
    ] = (
        time.time()
        - start
    )

    return metrics


def print_metrics(
    name,
    metrics
):
    print(
        f"\n=== {name.upper()} METRICS ==="
    )

    print(
        "Loss:",
        f"{metrics['loss']:.6f}"
    )

    print(
        "Accuracy:",
        f"{metrics['accuracy']:.4f}"
    )

    print(
        "Precision:",
        f"{metrics['precision']:.4f}"
    )

    print(
        "Recall:",
        f"{metrics['recall']:.4f}"
    )

    print(
        "F1:",
        f"{metrics['f1']:.4f}"
    )

    print(
        "ROC-AUC:",
        (
            "unavailable"
            if metrics["roc_auc"] is None
            else f"{metrics['roc_auc']:.4f}"
        )
    )

    print(
        "FAKE precision:",
        f"{metrics['fake_precision']:.4f}"
    )

    print(
        "FAKE recall:",
        f"{metrics['fake_recall']:.4f}"
    )

    print(
        "FAKE F1:",
        f"{metrics['fake_f1']:.4f}"
    )

    print(
        "REAL precision:",
        f"{metrics['real_precision']:.4f}"
    )

    print(
        "REAL recall:",
        f"{metrics['real_recall']:.4f}"
    )

    print(
        "REAL F1:",
        f"{metrics['real_f1']:.4f}"
    )

    print(
        "False positives:",
        metrics["false_positives"]
    )

    print(
        "False negatives:",
        metrics["false_negatives"]
    )

    print(
        "Confusion matrix:",
        metrics["confusion_matrix"]
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_f1,
    history,
    amp_enabled
):
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            (
                scaler.state_dict()
                if scaler is not None
                else None
            ),

        "epoch":
            epoch,

        "best_metric":
            best_f1,

        "history":
            history,

        "architecture":
            "RawNet2",

        "dataset":
            "PhonemeDF 2026 + DFADD",

        "sample_rate":
            SAMPLE_RATE,

        "audio_seconds":
            AUDIO_SECONDS,

        "num_samples":
            NUM_SAMPLES,

        "labels":
            {
                "0": "FAKE",
                "1": "REAL",
            },

        "threshold":
            0.5,

        "training":
            {
                "batch_size":
                    BATCH_SIZE,

                "gradient_accumulation_steps":
                    GRADIENT_ACCUMULATION_STEPS,

                "effective_batch_size":
                    EFFECTIVE_BATCH_SIZE,

                "samples_per_epoch":
                    SAMPLES_PER_EPOCH,

                "max_epochs":
                    MAX_EPOCHS,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "optimizer":
                    "AdamW",

                "scheduler":
                    "ReduceLROnPlateau",

                "amp":
                    amp_enabled,

                "seed":
                    RANDOM_SEED,

                "gradient_clip":
                    GRAD_CLIP,
            },

        "preprocessing":
            {
                "sample_rate":
                    SAMPLE_RATE,

                "channels":
                    1,

                "duration_seconds":
                    AUDIO_SECONDS,

                "num_samples":
                    NUM_SAMPLES,
            },

        "torch_version":
            torch.__version__,

        "cuda_version":
            torch.version.cuda,
    }

    temp_path = Path(
        str(path) + ".tmp"
    )

    torch.save(
        checkpoint,
        temp_path
    )

    os.replace(
        temp_path,
        path
    )


def save_history(history):
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            history,
            file,
            indent=2
        )


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    device
):
    if not path.exists():
        return (
            0,
            -float("inf"),
            [],
            scaler
        )

    print(
        "\n=== RESUMING CHECKPOINT ==="
    )

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    expected_labels = {
        "0": "FAKE",
        "1": "REAL",
    }

    if checkpoint.get(
        "labels"
    ) != expected_labels:
        raise RuntimeError(
            "Checkpoint label mapping "
            "is incompatible."
        )

    if checkpoint.get(
        "architecture"
    ) != "RawNet2":
        raise RuntimeError(
            "Checkpoint is not RawNet2."
        )

    if checkpoint.get(
        "sample_rate"
    ) != SAMPLE_RATE:
        raise RuntimeError(
            "Checkpoint sample-rate "
            "mismatch."
        )

    if checkpoint.get(
        "num_samples"
    ) != NUM_SAMPLES:
        raise RuntimeError(
            "Checkpoint input-size "
            "mismatch."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(
        device
    )

    if (
        "optimizer_state_dict"
        in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    if (
        "scheduler_state_dict"
        in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    if (
        scaler is not None
        and checkpoint.get(
            "scaler_state_dict"
        ) is not None
    ):
        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

    epoch = int(
        checkpoint.get(
            "epoch",
            0
        )
    )

    best_f1 = float(
        checkpoint.get(
            "best_metric",
            -float("inf")
        )
    )

    history = checkpoint.get(
        "history",
        []
    )

    del checkpoint

    clear_cuda_memory()

    print(
        "Resumed epoch:",
        epoch
    )

    print(
        "Best validation F1:",
        best_f1
    )

    return (
        epoch,
        best_f1,
        history,
        scaler
    )


def main():
    seed_everything(
        RANDOM_SEED
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = get_device()

    print(
        "\n=== DEEPSHIELD RAWNET2 TRAINING ==="
    )

    print_system_information(
        device
    )

    print(
        "\n=== CONFIGURATION ==="
    )

    print(
        "Sample rate:",
        SAMPLE_RATE
    )

    print(
        "Audio duration:",
        AUDIO_SECONDS
    )

    print(
        "Samples:",
        NUM_SAMPLES
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Gradient accumulation:",
        GRADIENT_ACCUMULATION_STEPS
    )

    print(
        "Effective batch size:",
        EFFECTIVE_BATCH_SIZE
    )

    print(
        "Samples per epoch:",
        SAMPLES_PER_EPOCH
    )

    print(
        "Maximum epochs:",
        MAX_EPOCHS
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    print(
        "Weight decay:",
        WEIGHT_DECAY
    )

    print(
        "AMP requested:",
        USE_AMP
    )

    print(
        "\n=== LABEL MAPPING ==="
    )

    print(
        "CLASS 0 = FAKE"
    )

    print(
        "CLASS 1 = REAL"
    )

    print(
        "\n=== BUILDING DATASETS ==="
    )

    train_dataset = create_dataset(
        "train",
        True
    )

    validation_dataset = create_dataset(
        "validation",
        False
    )

    test_dataset = create_dataset(
        "test",
        False
    )

    print(
        "\n=== DATASET OBJECTS ==="
    )

    print(
        "Training dataset:",
        type(
            train_dataset
        ).__name__
    )

    print(
        "Validation dataset:",
        type(
            validation_dataset
        ).__name__
    )

    print(
        "Test dataset:",
        type(
            test_dataset
        ).__name__
    )

    print(
        "Training distribution:",
        "balanced FAKE / REAL"
    )

    print(
        "Canonical labels:"
    )

    print(
        "  0 = FAKE"
    )

    print(
        "  1 = REAL"
    )

    print(
        "\n=== BUILDING LOADERS ==="
    )

    train_loader = create_loader(
        train_dataset
    )

    validation_loader = create_loader(
        validation_dataset
    )

    test_loader = create_loader(
        test_dataset
    )

    print(
        "Training loader ready."
    )

    print(
        "Validation loader ready."
    )

    print(
        "Test loader ready."
    )

    clear_cuda_memory()

    model = build_model(
        device
    )

    rawnet2_smoke_test(
        model,
        device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(
        model
    )

    scheduler = create_scheduler(
        optimizer
    )

    amp_enabled = False

    if (
        USE_AMP
        and device.type == "cuda"
    ):
        amp_enabled = amp_smoke_test(
            model,
            optimizer,
            criterion,
            device
        )

    scaler = (
        create_scaler(device)
        if amp_enabled
        else None
    )

    print(
        "\nAMP enabled:",
        amp_enabled
    )

    start_epoch = 0
    best_f1 = -float("inf")
    history = []

    if LATEST_CHECKPOINT.exists():
        (
            start_epoch,
            best_f1,
            history,
            scaler
        ) = load_checkpoint(
            LATEST_CHECKPOINT,
            model,
            optimizer,
            scheduler,
            scaler,
            device
        )

        verify_model_device(
            model,
            device
        )

    print_cuda_memory(
        "READY FOR TRAINING"
    )

    print(
        "\n=== TRAINING START ==="
    )

    epochs_without_improvement = 0

    for epoch in range(
        start_epoch + 1,
        MAX_EPOCHS + 1
    ):
        print(
            "\n"
            + "=" * 70
        )

        print(
            f"EPOCH {epoch}/{MAX_EPOCHS}"
        )

        print(
            "=" * 70
        )

        print(
            "Learning rate:",
            optimizer.param_groups[0][
                "lr"
            ]
        )

        train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                epoch,
                amp_enabled
            )
        )

        print_metrics(
            "training",
            train_metrics
        )

        clear_cuda_memory()

        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            "validation",
            amp_enabled
        )

        print_metrics(
            "validation",
            validation_metrics
        )

        validation_f1 = (
            validation_metrics["f1"]
        )

        scheduler.step(
            validation_f1
        )

        record = {
            "epoch":
                epoch,

            "learning_rate":
                float(
                    optimizer.param_groups[0][
                        "lr"
                    ]
                ),

            "training":
                train_metrics,

            "validation":
                validation_metrics,

            "timestamp":
                time.time(),
        }

        history.append(
            record
        )

        save_history(
            history
        )

        save_checkpoint(
            LATEST_CHECKPOINT,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_f1,
            history,
            amp_enabled
        )

        print(
            "\nLatest checkpoint saved."
        )

        if validation_f1 > best_f1:
            best_f1 = validation_f1
            epochs_without_improvement = 0

            save_checkpoint(
                BEST_CHECKPOINT,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_f1,
                history,
                amp_enabled
            )

            print(
                "NEW BEST CHECKPOINT SAVED."
            )

            print(
                "Validation F1:",
                f"{best_f1:.4f}"
            )

        else:
            epochs_without_improvement += 1

            print(
                "No validation improvement."
            )

            print(
                "No-improvement epochs:",
                epochs_without_improvement
            )

        print_cuda_memory(
            f"END EPOCH {epoch}"
        )

        clear_cuda_memory()

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            print(
                "\nEARLY STOPPING."
            )

            break

    if not BEST_CHECKPOINT.exists():
        raise RuntimeError(
            "Best checkpoint was not created."
        )

    print(
        "\n=== LOADING BEST CHECKPOINT ==="
    )

    best_checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location="cpu"
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    del best_checkpoint

    clear_cuda_memory()

    verify_model_device(
        model,
        device
    )

    print(
        "\n=== FINAL TEST ==="
    )

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        "test",
        amp_enabled
    )

    print_metrics(
        "final test",
        test_metrics
    )

    history.append(
        {
            "final_test":
                test_metrics,

            "best_validation_f1":
                (
                    float(best_f1)
                    if math.isfinite(
                        best_f1
                    )
                    else None
                ),

            "amp_enabled":
                amp_enabled,

            "device":
                str(device),

            "gpu":
                (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
        }
    )

    save_history(
        history
    )

    save_checkpoint(
        FINAL_CHECKPOINT,
        model,
        optimizer,
        scheduler,
        scaler,
        MAX_EPOCHS,
        best_f1,
        history,
        amp_enabled
    )

    print(
        "\n=== TRAINING COMPLETE ==="
    )

    print(
        "Best:",
        BEST_CHECKPOINT
    )

    print(
        "Latest:",
        LATEST_CHECKPOINT
    )

    print(
        "Final:",
        FINAL_CHECKPOINT
    )

    print(
        "History:",
        HISTORY_FILE
    )

    print(
        "\n0 = FAKE"
    )

    print(
        "1 = REAL"
    )


if __name__ == "__main__":
    main()
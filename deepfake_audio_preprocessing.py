

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import random
import soundfile as sf
from scipy import signal

TARGET_SAMPLE_RATE = 16000
TARGET_LENGTH = 64000  # 4 seconds at 16kHz


def load_audio_librosa(file_path: str | Path) -> np.ndarray:
    """
    Load audio using librosa (for training).
    Returns float32 mono audio at 16kHz, NOT normalized or padded.
    """
    audio, _ = librosa.load(
        str(file_path),
        sr=TARGET_SAMPLE_RATE,
        mono=True,
        dtype=np.float32
    )
    return audio


def load_audio_ffmpeg(file_path: str | Path) -> np.ndarray:
    """
    Load audio using FFmpeg (for inference).
    Returns float32 mono audio at 16kHz, NOT normalized or padded.
    """
    import subprocess
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(file_path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-"
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    if not result.stdout:
        raise RuntimeError("FFmpeg returned no audio samples")
    waveform = np.frombuffer(result.stdout, dtype=np.float32).copy()
    return waveform


def normalize_audio(waveform: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Normalize audio: DC offset removal + peak normalization.
    Input: numpy array or torch tensor
    Output: torch tensor
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform).float()
    else:
        waveform = waveform.float()
    
    # Remove DC offset
    waveform = waveform - waveform.mean()
    
    # Peak normalization
    peak = torch.max(torch.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    
    return waveform


def pad_or_truncate(waveform: torch.Tensor, target_length: int = TARGET_LENGTH) -> torch.Tensor:
    """
    Pad or truncate waveform to target length.
    Input: torch tensor [samples]
    Output: torch tensor [target_length]
    """
    current_length = waveform.numel()
    
    if current_length < target_length:
        waveform = F.pad(waveform, (0, target_length - current_length))
    elif current_length > target_length:
        waveform = waveform[:target_length]
    
    return waveform


def prepare_waveform(waveform: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Full preprocessing pipeline: normalize + pad/truncate.
    Input: raw waveform (numpy or torch) at 16kHz mono
    Output: torch tensor [1, 64000] with batch dimension
    """
    waveform = normalize_audio(waveform)
    waveform = pad_or_truncate(waveform)
    waveform = waveform.unsqueeze(0)  # Add batch dim: [64000] -> [1, 64000]
    return waveform


def preprocess_audio_file(file_path: str | Path, use_ffmpeg: bool = False) -> torch.Tensor:
    """
    Load and preprocess an audio file for RawNet2 inference.
    
    Args:
        file_path: Path to audio file
        use_ffmpeg: If True, use FFmpeg (for inference). If False, use librosa (for training).
    
    Returns:
        torch.Tensor of shape [1, 64000] ready for model input
    """
    if use_ffmpeg:
        waveform = load_audio_ffmpeg(file_path)
    else:
        waveform = load_audio_librosa(file_path)
    
    if waveform.size == 0:
        raise ValueError(f"Empty audio: {file_path}")
    
    return prepare_waveform(waveform)


def preprocess_audio_for_training(file_path: str | Path, augment: bool = True) -> torch.Tensor:
    """
    Preprocess audio for training (uses librosa).
    Returns tensor [64000] without batch dimension (dataset adds it).
    """
    waveform = load_audio_librosa(file_path)
    if waveform.size == 0:
        raise ValueError(f"Empty audio: {file_path}")
    if augment:
        waveform = apply_random_augmentation(waveform, TARGET_SAMPLE_RATE, prob=0.5)
    waveform = normalize_audio(waveform)
    waveform = pad_or_truncate(waveform)
    return waveform  # [64000], no batch dim


# ============================================================
# CODEC AUGMENTATION (for training robustness)
# ============================================================

import tempfile
import subprocess
import random


def apply_codec_augmentation(waveform: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """
    Apply random codec augmentation using FFmpeg.
    Simulates real-world compression: MP3, AAC, Opus, resampling.
    """
    aug_type = random.random()
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
        tmp_in_path = tmp_in.name
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
        tmp_out_path = tmp_out.name
    
    try:
        # Write input as WAV
        sf.write(tmp_in_path, waveform, sample_rate, subtype='PCM_16')
        
        if aug_type < 0.25:
            # MP3 compression (32-128 kbps)
            bitrate = random.choice([32, 48, 64, 96, 128])
            cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                   '-i', tmp_in_path,
                   '-c:a', 'libmp3lame', '-b:a', f'{bitrate}k',
                   '-ar', str(sample_rate), '-ac', '1',
                   tmp_out_path]
        elif aug_type < 0.5:
            # AAC compression (32-128 kbps)
            bitrate = random.choice([32, 48, 64, 96, 128])
            cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                   '-i', tmp_in_path,
                   '-c:a', 'aac', '-b:a', f'{bitrate}k',
                   '-ar', str(sample_rate), '-ac', '1',
                   tmp_out_path]
        elif aug_type < 0.65:
            # Opus compression (16-64 kbps)
            bitrate = random.choice([16, 24, 32, 48, 64])
            cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                   '-i', tmp_in_path,
                   '-c:a', 'libopus', '-b:a', f'{bitrate}k',
                   '-ar', str(sample_rate), '-ac', '1',
                   tmp_out_path]
        elif aug_type < 0.8:
            # Resample to 8kHz then back to 16kHz
            cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                   '-i', tmp_in_path,
                   '-ar', '8000', '-ac', '1',
                   '-ar', str(sample_rate), '-ac', '1',
                   tmp_out_path]
        else:
            # No codec, just add noise
            return add_background_noise(waveform)
        
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0:
            return add_background_noise(waveform)
        
        # Read augmented audio
        aug_waveform, _ = sf.read(tmp_out_path, dtype='float32')
        return aug_waveform.astype(np.float32)
        
    except Exception:
        return add_background_noise(waveform)
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)
        Path(tmp_out_path).unlink(missing_ok=True)


def add_background_noise(waveform: np.ndarray, snr_db_range: tuple = (15, 30)) -> np.ndarray:
    """Add random background noise (white/pink/brown) at random SNR."""
    noise_type = random.choice(['white', 'pink', 'brown'])
    snr_db = random.uniform(*snr_db_range)
    
    signal_power = np.mean(waveform ** 2)
    if signal_power == 0:
        return waveform
    
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    if noise_type == 'white':
        noise = np.random.normal(0, np.sqrt(noise_power), waveform.shape)
    elif noise_type == 'pink':
        # Pink noise via Voss-McCartney algorithm
        noise = _generate_pink_noise(len(waveform))
        noise = noise * np.sqrt(noise_power / (np.mean(noise ** 2) + 1e-10))
    else:  # brown
        noise = _generate_brown_noise(len(waveform))
        noise = noise * np.sqrt(noise_power / (np.mean(noise ** 2) + 1e-10))
    
    return (waveform + noise).astype(np.float32)


def _generate_pink_noise(n: int) -> np.ndarray:
    """Generate pink noise (1/f) using Voss-McCartney."""
    rows = 16
    array = np.random.randn(rows, n)
    for i in range(rows):
        array[i] = np.cumsum(array[i])
    pink = array.sum(axis=0)
    pink = pink / (np.max(np.abs(pink)) + 1e-10)
    return pink[:n]


def _generate_brown_noise(n: int) -> np.ndarray:
    """Generate brown noise (1/f^2) via integration of white noise."""
    white = np.random.randn(n)
    brown = np.cumsum(white)
    brown = brown / (np.max(np.abs(brown)) + 1e-10)
    return brown[:n]


def apply_random_augmentation(waveform: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE, prob: float = 0.5) -> np.ndarray:
    """
    Apply random augmentation with probability.
    Returns augmented waveform.
    """
    if random.random() > prob:
        return waveform
    
    aug_choice = random.random()
    if aug_choice < 0.4:
        return apply_codec_augmentation(waveform, sample_rate)
    elif aug_choice < 0.7:
        return add_background_noise(waveform)
    elif aug_choice < 0.85:
        # Random gain
        gain_db = random.uniform(-6, 6)
        gain = 10 ** (gain_db / 20)
        return (waveform * gain).astype(np.float32)
    else:
        # Random time stretch (0.9-1.1x) via resample
        factor = random.uniform(0.9, 1.1)
        new_len = int(len(waveform) * factor)
        stretched = signal.resample(waveform, new_len)
        # Pad or truncate back to original length
        if len(stretched) > len(waveform):
            return stretched[:len(waveform)].astype(np.float32)
        else:
            return np.pad(stretched, (0, len(waveform) - len(stretched))).astype(np.float32)


# ============================================================
# COMPATIBILITY FUNCTIONS (for existing code)
# ============================================================

def preprocess_audio(file_path: str | Path) -> dict:
    """
    Legacy compatibility function for ml/preprocessing/audio.py interface.
    Returns dict with audio, sample_rate, duration, samples.
    """
    audio = load_audio_librosa(file_path)
    duration = len(audio) / TARGET_SAMPLE_RATE
    return {
        "audio": audio,
        "sample_rate": TARGET_SAMPLE_RATE,
        "duration": duration,
        "samples": len(audio)
    }


SAMPLE_RATE = TARGET_SAMPLE_RATE
DURATION = 4  # Changed from 5 to 4 seconds to match RawNet2
SAMPLES = TARGET_LENGTH


def load_audio(file_path: str | Path) -> np.ndarray:
    """Legacy compatibility for backend/detector/preprocessing.py"""
    audio = load_audio_librosa(file_path)
    # Apply same normalization as training
    audio = normalize_audio(audio).numpy()
    # Pad/truncate
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]
    return audio.astype(np.float32)


def extract_features(file_path: str | Path) -> np.ndarray:
    """Legacy MFCC feature extraction (for old CNN model)"""
    audio = load_audio(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)
    return features.astype(np.float32)
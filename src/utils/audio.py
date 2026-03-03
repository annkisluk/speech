
import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional
import soundfile as sf


def load_audio(
    file_path: str, 
    target_sr: int = 8000,
    normalize: bool = True
) -> Tuple[torch.Tensor, int]:

    # Load audio
    audio_np, sr = sf.read(file_path, always_2d=True)
    audio = torch.tensor(audio_np.T, dtype=torch.float32)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
        sr = target_sr
    
    # Normalize to [-1, 1]
    if normalize:
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
    
    return audio, sr


def save_audio(
    audio: torch.Tensor,
    file_path: str,
    sr: int = 8000
):
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Convert to numpy
    audio_np = audio.cpu().numpy()
    
    # Save using soundfile
    sf.write(file_path, audio_np.T, sr)


def normalize_audio(audio: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return audio / (torch.max(torch.abs(audio)) + eps)


def calculate_si_snr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    # Ensure shape is [B, T]
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    
    # Zero-mean normalization 
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    # Compute scaling factor α = <estimate, target> / ||target||^2
    dot_product = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    alpha = dot_product / target_energy
    
    # Project estimate onto target direction: s_target = α * target
    s_target = alpha * target
    
    # Compute noise as orthogonal component
    e_noise = estimate - s_target
    
    # Calculate SI-SNR in dB
    target_power = torch.sum(s_target ** 2, dim=-1) + eps
    noise_power = torch.sum(e_noise ** 2, dim=-1) + eps
    si_snr = 10 * torch.log10(target_power / noise_power)
    
    return si_snr


def si_snr_loss(
    estimate: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    si_snr_values = calculate_si_snr(estimate, target)
    # Return negative for minimization (maximize SI-SNR)
    return -torch.mean(si_snr_values)


def apply_time_stretch(
    audio: torch.Tensor,
    rate: float = 1.0
) -> torch.Tensor:
    if rate == 1.0:
        return audio
    return audio


def apply_pitch_shift(
    audio: torch.Tensor,
    n_steps: int = 0,
    sr: int = 8000
) -> torch.Tensor:
    if n_steps == 0:
        return audio
    return audio


def pad_audio_batch(
    audio_list: list,
    pad_value: float = 0.0
) -> torch.Tensor:
    max_len = max(audio.shape[-1] for audio in audio_list)
    batch_size = len(audio_list)
    
    # Create padded tensor
    padded = torch.full((batch_size, max_len), pad_value)
    
    # Fill with actual audio
    for i, audio in enumerate(audio_list):
        length = audio.shape[-1]
        padded[i, :length] = audio.squeeze()
    
    return padded


def trim_audio_batch(
    audio_batch: torch.Tensor,
    lengths: torch.Tensor
) -> list:
    audio_list = []
    for i, length in enumerate(lengths):
        audio_list.append(audio_batch[i, :length])
    
    return audio_list


def extract_features_for_clustering(
    audio: torch.Tensor,
    encoder: torch.nn.Module,
    use_mean_pooling: bool = True
) -> torch.Tensor:
    encoder.eval()
    with torch.no_grad():
        # Extract features: E(x) -> [B, L, D]
        features = encoder(audio)
        
        if use_mean_pooling:
            # MeanP(E(x)): Average over time dimension -> [B, D]
            features = torch.mean(features, dim=1)
    
    return features

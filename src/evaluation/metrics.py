"""
Evaluation Metrics for Speech Enhancement

Implements the three metrics used in the paper:
1. SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
2. SDR (Signal-to-Distortion Ratio)
3. PESQ (Perceptual Evaluation of Speech Quality)

Paper Reference: Section IV.A - Experimental Setup
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

try:
    from pesq import pesq as pesq_metric
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    warnings.warn("PESQ not installed. Install with: pip install pesq")

try:
    from pystoi import stoi as stoi_metric
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    warnings.warn("STOI not installed. Install with: pip install pystoi")


def calculate_si_snr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Calculate Scale-Invariant Signal-to-Noise Ratio (SI-SNR)
    
    Paper Reference: Loss function from [21]
    
    Args:
        estimate: Enhanced speech [B, T] or [B, 1, T]
        target: Clean speech [B, T] or [B, 1, T]
        eps: Small constant for numerical stability
    
    Returns:
        SI-SNR values in dB [B]
    """
    # Ensure shape is [B, T]
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    
    # Zero-mean normalization
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    # Compute scaling factor
    dot_product = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    alpha = dot_product / target_energy
    
    # Project estimate onto target direction
    s_target = alpha * target
    
    # Noise is the orthogonal component
    e_noise = estimate - s_target
    
    # Calculate SI-SNR
    target_power = torch.sum(s_target ** 2, dim=-1) + eps
    noise_power = torch.sum(e_noise ** 2, dim=-1) + eps
    si_snr = 10 * torch.log10(target_power / noise_power)
    
    return si_snr


def calculate_sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Calculate Signal-to-Distortion Ratio (SDR)
    
    Paper Reference: Section IV.A
    
    Args:
        estimate: Enhanced speech [B, T] or [B, 1, T]
        target: Clean speech [B, T] or [B, 1, T]
        eps: Small constant for numerical stability
    
    Returns:
        SDR values in dB [B]
    """
    # Ensure shape is [B, T]
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    
    # Standard SDR (BSS_eval): project estimate onto target direction
    # s_target = <est, target> / ||target||^2 * target
    dot_product = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    s_target = (dot_product / target_energy) * target
    
    # Distortion is the component orthogonal to target
    e_distortion = estimate - s_target
    
    # SDR = 10 * log10(||s_target||^2 / ||e_distortion||^2)
    target_power = torch.sum(s_target ** 2, dim=-1) + eps
    distortion_power = torch.sum(e_distortion ** 2, dim=-1) + eps
    sdr = 10 * torch.log10(target_power / distortion_power)
    
    return sdr


def calculate_pesq(
    estimate: np.ndarray,
    target: np.ndarray,
    sr: int = 8000,
    mode: str = 'nb'
) -> float:
    """
    Calculate PESQ (Perceptual Evaluation of Speech Quality)
    
    Args:
        estimate: Enhanced speech (numpy array, 1D)
        target: Clean speech (numpy array, 1D)
        sr: Sample rate (8000 for narrowband)
        mode: 'nb' (narrowband) for 8kHz
    
    Returns:
        PESQ score (higher is better, range: -0.5 to 4.5)
    """
    if not PESQ_AVAILABLE:
        warnings.warn("PESQ not available, returning 0.0")
        return 0.0
    
    # Ensure 1D arrays
    if estimate.ndim > 1:
        estimate = estimate.flatten()
    if target.ndim > 1:
        target = target.flatten()
    
    # Ensure same length
    min_len = min(len(estimate), len(target))
    estimate = estimate[:min_len]
    target = target[:min_len]
    
    try:
        score = pesq_metric(sr, target, estimate, mode)
        return float(score)
    except Exception as e:
        warnings.warn(f"PESQ calculation failed: {e}")
        return 0.0


class MetricsCalculator:
    """
    Unified metrics calculator
    
    Usage:
        calc = MetricsCalculator(sample_rate=8000)
        metrics = calc.calculate_all(enhanced, clean)
    """
    
    def __init__(
        self,
        sample_rate: int = 8000,
        metrics: List[str] = ['si_snr', 'sdr', 'pesq']
    ):
        """
        Args:
            sample_rate: Audio sample rate (8000 as per paper)
            metrics: List of metrics to compute
        """
        self.sample_rate = sample_rate
        self.metrics = metrics
        self.pesq_mode = 'nb' if sample_rate == 8000 else 'wb'
    
    def calculate_all(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate all metrics for a single sample
        
        Args:
            estimate: Enhanced speech [T] or [1, T]
            target: Clean speech [T] or [1, T]
        
        Returns:
            Dictionary with metric values
        """
        results = {}
        
        # Ensure 1D tensors
        if estimate.dim() > 1:
            estimate = estimate.squeeze()
        if target.dim() > 1:
            target = target.squeeze()
        
        # Calculate torch-based metrics
        if 'si_snr' in self.metrics:
            si_snr = calculate_si_snr(
                estimate.unsqueeze(0), 
                target.unsqueeze(0)
            )
            results['si_snr'] = si_snr.item()
        
        if 'sdr' in self.metrics:
            sdr = calculate_sdr(
                estimate.unsqueeze(0),
                target.unsqueeze(0)
            )
            results['sdr'] = sdr.item()
        
        # Convert to numpy for PESQ
        estimate_np = estimate.cpu().numpy()
        target_np = target.cpu().numpy()
        
        if 'pesq' in self.metrics:
            results['pesq'] = calculate_pesq(
                estimate_np,
                target_np,
                self.sample_rate,
                self.pesq_mode
            )
        
        return results
    
    def calculate_batch(
        self,
        estimate_batch: torch.Tensor,
        target_batch: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, List[float]]:
        """
        Calculate metrics for a batch of samples
        
        Args:
            estimate_batch: Enhanced speech [B, T] or [B, 1, T]
            target_batch: Clean speech [B, T] or [B, 1, T]
            lengths: Actual lengths (for handling padding) [B]
        
        Returns:
            Dictionary with lists of metric values
        """
        batch_size = estimate_batch.shape[0]
        results = {metric: [] for metric in self.metrics}
        
        for i in range(batch_size):
            if lengths is not None:
                length = lengths[i].item()
                # Handle [B, 1, T] (with channel dim) and [B, T] shapes
                sample_est = estimate_batch[i]  # [1, T] or [T]
                sample_tgt = target_batch[i]
                if sample_est.dim() == 2:
                    # [1, T] -> trim time dim -> squeeze channel
                    estimate = sample_est[0, :length]
                    target = sample_tgt[0, :length]
                else:
                    # [T] -> trim directly
                    estimate = sample_est[:length]
                    target = sample_tgt[:length]
            else:
                estimate = estimate_batch[i]
                target = target_batch[i]
            
            sample_metrics = self.calculate_all(estimate, target)
            
            for metric, value in sample_metrics.items():
                results[metric].append(value)
        
        return results
    
    def aggregate_metrics(
        self,
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Aggregate metrics from multiple samples
        
        Args:
            metrics_list: List of metric dictionaries
        
        Returns:
            Dictionary with mean values
        """
        aggregated = {}
        
        for metric in self.metrics:
            values = [m[metric] for m in metrics_list if metric in m]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
        
        return aggregated


if __name__ == "__main__":
    print("Testing metrics...")
    
    # Test signals
    sr = 8000
    duration = 2
    t = torch.linspace(0, duration, sr * duration)
    
    clean = torch.sin(2 * np.pi * 440 * t)
    perfect = clean.clone()
    noisy = clean + 0.1 * torch.randn_like(clean)
    enhanced = clean + 0.05 * torch.randn_like(clean)
    
    print("\nSI-SNR:")
    si_snr_perfect = calculate_si_snr(perfect.unsqueeze(0), clean.unsqueeze(0))
    si_snr_noisy = calculate_si_snr(noisy.unsqueeze(0), clean.unsqueeze(0))
    si_snr_enhanced = calculate_si_snr(enhanced.unsqueeze(0), clean.unsqueeze(0))
    
    print(f"  Perfect: {si_snr_perfect.item():.2f} dB")
    print(f"  Noisy: {si_snr_noisy.item():.2f} dB")
    print(f"  Enhanced: {si_snr_enhanced.item():.2f} dB")
    
    print("\n✓ Metrics working!")
#!/usr/bin/env python3
"""
Predator — Spectral Preprocessing Reference (Phase 33.6a)
Platform-agnostic frequency-domain preprocessing parity reference.

Implements identical preprocessing to CudaFlickerCore::compute_normalized_spectrum:
1. 512-sample temporal time series at Fs = 4000 Hz (128 ms window)
2. Hanning window: w[n] = 0.5 * (1 - cos(2*pi*n / 511)) for n = 0..511
3. 512-point Real-to-Complex FFT (unnormalized, 257 bins, 0 to 2000 Hz, df = 7.8125 Hz)
4. Power spectrum: P[k] = Re(X_k)^2 + Im(X_k)^2
5. Noise floor estimation: median(P[5:128]) clamped to >= 1e-4
6. Scale-invariant log-power normalization: S[k] = log10(1 + P[k] / median_noise)
"""

import numpy as np
import torch

def get_hanning_window_numpy(n: int = 512) -> np.ndarray:
    """Returns 512-sample symmetric Hanning window matching CudaFlickerCore constant memory."""
    idx = np.arange(n, dtype=np.float64)
    return (0.5 * (1.0 - np.cos(2.0 * np.pi * idx / (n - 1)))).astype(np.float32)

def get_hanning_window_torch(n: int = 512, device=None) -> torch.Tensor:
    """Returns 512-sample symmetric Hanning window as PyTorch tensor."""
    idx = torch.arange(n, dtype=torch.float64, device=device)
    win = 0.5 * (1.0 - torch.cos(2.0 * torch.pi * idx / (n - 1)))
    return win.to(torch.float32)


def compute_normalized_spectrum_numpy(time_series: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Computes 257-bin normalized power spectrum from 512-sample time series using NumPy.
    
    Args:
        time_series: 1D array of 512 float values
        
    Returns:
        tuple of (normalized_spectrum_257, median_noise)
    """
    ts = np.asarray(time_series, dtype=np.float32)
    if ts.shape != (512,):
        raise ValueError(f"Expected time_series shape (512,), got {ts.shape}")

    window = get_hanning_window_numpy(512)
    windowed = ts * window

    # 512-pt RFFT (unnormalized, matches cuFFT R2C)
    fft_c = np.fft.rfft(windowed, n=512)
    power = (np.real(fft_c) ** 2 + np.imag(fft_c) ** 2).astype(np.float32)

    # Median noise in bins 5..127 (123 elements)
    noise_slice = power[5:128]
    sorted_slice = np.sort(noise_slice)
    # Exact middle element (index 61 of 123)
    median_noise = float(max(1e-4, sorted_slice[len(sorted_slice) // 2]))

    norm_spectrum = np.log10(1.0 + power / median_noise).astype(np.float32)
    return norm_spectrum, median_noise

def compute_normalized_spectrum_torch(time_series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes 257-bin normalized power spectrum from 512-sample time series using PyTorch.
    Supports batched tensors: (B, 512) -> ((B, 257), (B, 1)).
    
    Args:
        time_series: Tensor of shape (512,) or (B, 512)
        
    Returns:
        tuple of (normalized_spectrum, median_noise)
    """
    single = False
    if time_series.dim() == 1:
        time_series = time_series.unsqueeze(0)
        single = True

    b, n = time_series.shape
    if n != 512:
        raise ValueError(f"Expected sequence length 512, got {n}")

    window = get_hanning_window_torch(512, device=time_series.device)
    windowed = time_series * window

    # 512-pt RFFT unnormalized
    fft_c = torch.fft.rfft(windowed, n=512, norm="backward")
    power = torch.real(fft_c) ** 2 + torch.imag(fft_c) ** 2

    # Noise slice in bins 5..127 (123 elements)
    noise_slice = power[:, 5:128]
    # torch.median returns (values, indices)
    median_noise = torch.median(noise_slice, dim=1, keepdim=True).values
    median_noise = torch.clamp(median_noise, min=1e-4)

    norm_spectrum = torch.log10(1.0 + power / median_noise)

    if single:
        return norm_spectrum.squeeze(0), median_noise.squeeze(0).squeeze(0)
    return norm_spectrum, median_noise

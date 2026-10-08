#!/usr/bin/env python3
"""
Predator — SpectralCombNet: Frequency-Domain Drone Propeller Neural Discriminator
Trained on NVIDIA GB10 GPU (DGX Spark) for deployment via TensorRT FP16 on Jetson Orin Nano.

Input: 257-bin cuFFT power spectrum (0 to 2000 Hz, sample rate 4000 Hz, N=512)
Outputs:
  - drone_prob: Probability of drone propeller harmonic comb (0.0 to 1.0)
  - fund_freq_hz: Predicted fundamental blade passage frequency (70 to 800 Hz)
  - harmonic_purity: Multi-harmonic comb coherence score (0.0 to 1.0)
"""

import os
import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# Model Architecture: 1D Dilated Residual Harmonic Network (SpectralCombNet)
# ==============================================================================

class HarmonicResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class SpectralCombNet(nn.Module):
    def __init__(self, in_bins: int = 257, base_channels: int = 32):
        super().__init__()
        # Input stem: (B, 1, 257) -> (B, base_channels, 257)
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU()
        )

        # Multi-scale harmonic dilation blocks
        # Dilation 1: local peak shape
        self.block1 = HarmonicResBlock(base_channels, dilation=1)
        # Dilation 2: 2nd harmonic octave resonance (2 * f0)
        self.block2 = HarmonicResBlock(base_channels, dilation=2)
        # Dilation 3: 3rd harmonic resonance (3 * f0)
        self.block3 = HarmonicResBlock(base_channels, dilation=3)
        # Dilation 4: 4th harmonic resonance (4 * f0)
        self.block4 = HarmonicResBlock(base_channels, dilation=4)

        # Bottleneck projection
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.GELU()
        )

        # Output feature dimension: (base_channels * 2) * 2 (avg + max pool) = 128
        feat_dim = base_channels * 4

        # Multi-task heads
        self.classifier_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.freq_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        self.purity_head = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, 1, 257)
        out = self.stem(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.bottleneck(out)

        # Hybrid Global Average + Max Pooling
        avg_pool = torch.mean(out, dim=-1)
        max_pool, _ = torch.max(out, dim=-1)
        feat = torch.cat([avg_pool, max_pool], dim=-1)

        drone_prob = self.classifier_head(feat)
        fund_freq = self.freq_head(feat)
        harmonic_purity = self.purity_head(feat)

        return drone_prob, fund_freq, harmonic_purity


# ==============================================================================
# Synthetic Event Propeller Spectrum Generator (EGM Simulator)
# ==============================================================================

class SyntheticSpectrumDataset(Dataset):
    def __init__(self, num_samples: int = 100000, sample_rate_hz: float = 4000.0, num_bins: int = 257):
        self.num_samples = num_samples
        self.sample_rate_hz = sample_rate_hz
        self.num_bins = num_bins
        self.df = sample_rate_hz / 512.0  # 7.8125 Hz

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 50% Positive (Drone harmonic comb), 50% Negative (Clutter / AC / Noise)
        is_drone = (random.random() < 0.5)

        spectrum = np.zeros(self.num_bins, dtype=np.float32)
        target_f0 = 0.0
        target_purity = 0.0

        # Background thermal & Poisson shot noise (exponentially distributed)
        base_noise_level = np.random.uniform(0.5, 3.0)
        noise = np.random.exponential(scale=base_noise_level, size=self.num_bins).astype(np.float32)
        spectrum += noise

        if is_drone:
            # Random fundamental frequency in [70, 800 Hz]
            target_f0 = np.random.uniform(70.0, 780.0)
            target_purity = np.random.uniform(0.70, 1.0)

            # Signal-to-Noise Ratio (dB): range from -2 dB (heavily shaded/standoff) to +28 dB (bright sunlight)
            snr_db = np.random.uniform(-2.0, 28.0)
            snr_linear = 10.0 ** (snr_db / 10.0)
            peak_power = base_noise_level * snr_linear

            # Blade count (2, 3, or 4 blades)
            blade_count = random.choice([2, 3, 4])

            # Harmonics (f0, 2f0, 3f0, 4f0)
            num_harmonics = random.randint(2, 4)
            decay_exp = np.random.uniform(0.8, 1.8)

            for h in range(1, num_harmonics + 1):
                harmonic_freq = h * target_f0
                if harmonic_freq >= self.sample_rate_hz / 2.0 - 20.0:
                    break

                bin_pos = harmonic_freq / self.df
                center_bin = int(round(bin_pos))

                # Harmonic amplitude with decay
                h_amp = (peak_power / (h ** decay_exp))

                # Shaded low-contrast condition: random cycle dropouts
                if random.random() < 0.20:
                    h_amp *= np.random.uniform(0.3, 0.7)

                # Inject Gaussian spectral peak with sub-bin spreading
                for offset in [-2, -1, 0, 1, 2]:
                    k = center_bin + offset
                    if 0 <= k < self.num_bins:
                        dist = abs(k - bin_pos)
                        spread = math.exp(-0.5 * (dist / 0.75) ** 2)
                        spectrum[k] += h_amp * spread
        else:
            # Negative Clutter Generation (Wind foliage, sparse Poisson noise, foliage turbulence, AC, wideband steps)
            clutter_type = random.choice(["wind_foliage", "foliage_turbulence", "sparse_poisson", "ac_powerline", "pure_noise", "wideband_step"])

            if clutter_type == "wind_foliage":
                # Low-frequency 1/f red noise (5 to 50 Hz wind sway)
                decay = np.random.uniform(1.2, 2.5)
                amp = np.random.uniform(5.0, 40.0)
                for k in range(1, 30):
                    spectrum[k] += amp / ((k + 1) ** decay)

            elif clutter_type == "foliage_turbulence":
                # Multi-frequency wind turbulence across leaves (5 to 60 Hz with fluttering phase)
                num_leaves = random.randint(3, 8)
                for _ in range(num_leaves):
                    f_leaf = np.random.uniform(4.0, 65.0)
                    bin_pos = f_leaf / self.df
                    center_bin = int(round(bin_pos))
                    leaf_amp = base_noise_level * np.random.uniform(5.0, 40.0)
                    for offset in [-2, -1, 0, 1, 2]:
                        k = center_bin + offset
                        if 0 <= k < self.num_bins:
                            spectrum[k] += leaf_amp * math.exp(-0.5 * (offset / 1.2) ** 2)

            elif clutter_type == "sparse_poisson":
                # Real event camera sparse Poisson noise: 3 to 45 delta spikes in 512-sample temporal buffer
                num_events = random.randint(3, 45)
                t_indices = np.random.choice(512, size=num_events, replace=False)
                t_signal = np.zeros(512, dtype=np.float32)
                t_signal[t_indices] = np.random.uniform(1.0, 3.0, size=num_events)
                hanning = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(512) / 511.0))
                windowed = t_signal * hanning
                fft_res = np.fft.rfft(windowed)
                power_sparse = (np.abs(fft_res[:self.num_bins]) ** 2).astype(np.float32)
                spectrum += power_sparse * np.random.uniform(2.0, 20.0)

            elif clutter_type == "ac_powerline":
                # 100/120/150/180/200/240/300 Hz lighting flicker without drone harmonics
                ac_freqs = [100.0, 120.0, 150.0, 180.0, 200.0, 240.0, 300.0]
                selected_ac = random.sample(ac_freqs, k=random.randint(1, 3))
                for ac_f in selected_ac:
                    bin_pos = ac_f / self.df
                    center_bin = int(round(bin_pos))
                    ac_power = base_noise_level * np.random.uniform(10.0, 80.0)
                    for offset in [-1, 0, 1]:
                        k = center_bin + offset
                        if 0 <= k < self.num_bins:
                            spectrum[k] += ac_power * math.exp(-abs(offset))

            elif clutter_type == "wideband_step":
                # Sharp camera movement / light transition: flat wideband burst
                burst_amp = np.random.uniform(2.0, 15.0)
                spectrum += burst_amp * np.random.uniform(0.5, 1.5, size=self.num_bins).astype(np.float32)

            target_f0 = 0.0
            target_purity = 0.0

        # Normalization: Log-power scaling for numerical stability in FP16
        # log10(1 + P / median_noise)
        median_noise = max(1e-6, float(np.median(spectrum[5:128])))
        norm_spectrum = np.log10(1.0 + spectrum / median_noise).astype(np.float32)

        x_tensor = torch.from_numpy(norm_spectrum).unsqueeze(0) # (1, 257)
        y_drone = torch.tensor([1.0 if is_drone else 0.0], dtype=torch.float32)
        y_f0 = torch.tensor([target_f0], dtype=torch.float32)
        y_purity = torch.tensor([target_purity], dtype=torch.float32)

        return x_tensor, y_drone, y_f0, y_purity


# ==============================================================================
# Training and Export Pipeline
# ==============================================================================

def train_model(epochs: int = 15, batch_size: int = 128, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing SpectralCombNet training on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})...")

    model = SpectralCombNet(in_bins=257, base_channels=32).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] SpectralCombNet Architecture initialized. Trainable Parameters: {total_params:,}")

    train_dataset = SyntheticSpectrumDataset(num_samples=80000)
    val_dataset = SyntheticSpectrumDataset(num_samples=10000)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for x, y_drone, y_f0, y_purity in train_loader:
            x, y_drone, y_f0, y_purity = x.to(device), y_drone.to(device), y_f0.to(device), y_purity.to(device)

            optimizer.zero_grad()
            p_drone, pred_f0, pred_purity = model(x)

            loss_cls = criterion_bce(p_drone, y_drone)
            
            # Mask regression loss only on true positive drone samples
            mask = (y_drone > 0.5).squeeze(-1)
            if mask.sum() > 0:
                loss_freq = criterion_mse(pred_f0[mask] / 100.0, y_f0[mask] / 100.0)
            else:
                loss_freq = torch.tensor(0.0, device=device)

            # Enforce purity across ALL samples: push to 0.0 for noise/foliage/clutter, push to target for drones
            loss_purity = criterion_bce(pred_purity, y_purity)

            total_loss = loss_cls + 0.25 * loss_freq + 0.25 * loss_purity
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * x.size(0)
            preds = (p_drone >= 0.5).float()
            correct += (preds == y_drone).sum().item()
            total += x.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_f0_error = 0.0
        val_f0_count = 0

        with torch.no_grad():
            for x, y_drone, y_f0, y_purity in val_loader:
                x, y_drone, y_f0 = x.to(device), y_drone.to(device), y_f0.to(device)
                p_drone, pred_f0, _ = model(x)

                preds = (p_drone >= 0.5).float()
                val_correct += (preds == y_drone).sum().item()
                val_total += x.size(0)

                mask = (y_drone > 0.5).squeeze(-1)
                if mask.sum() > 0:
                    err = torch.abs(pred_f0[mask] - y_f0[mask]).sum().item()
                    val_f0_error += err
                    val_f0_count += mask.sum().item()

        val_acc = val_correct / val_total
        mean_f0_err = (val_f0_error / val_f0_count) if val_f0_count > 0 else 0.0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Mean F0 Error: {mean_f0_err:.2f} Hz")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "spectral_combnet_best.pt")

    print(f"\n[INFO] Training complete! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    return model


def export_to_onnx(model, output_path: str = "spectral_combnet.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 1, 257, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")

    input_names = ["spectrum_in"]
    output_names = ["drone_prob", "fund_freq_hz", "harmonic_purity"]
    dynamic_axes = {
        "spectrum_in": {0: "batch_size"},
        "drone_prob": {0: "batch_size"},
        "fund_freq_hz": {0: "batch_size"},
        "harmonic_purity": {0: "batch_size"}
    }

    print(f"[INFO] Exporting model to ONNX: {output_path} (Dynamic Batching for 1152 cells)...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print(f"[SUCCESS] Exported {output_path} successfully ({os.path.getsize(output_path):,} bytes).")


if __name__ == "__main__":
    trained_model = train_model(epochs=15, batch_size=128, lr=1e-3)
    export_to_onnx(trained_model, "spectral_combnet.onnx")

#!/usr/bin/env python3
"""
Predator — SpectralCombNet v3: Frequency-Domain Drone Propeller Neural Discriminator
Trained on NVIDIA GB10 GPU (DGX Spark) for deployment via TensorRT FP16 on Jetson Orin Nano.

Phase 33.6d:
- Trained on CombinedSpectralDataset (Physical multi-rotor synthetic pulses + Real Orin darkroom spectra).
- Exact runtime preprocessing parity (compute_normalized_spectrum_torch).
- Multi-task learning: Drone probability, fundamental BPF (Hz), harmonic purity.
- Temperature scaling calibration (Platt/Guo) for well-calibrated probabilities.
- Self-contained ONNX opset 17 export with dynamic batching [1..128].
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
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict, Any

from real_spectra_dataset import CombinedSpectralDataset

# ==============================================================================
# Model Architecture: 1D Dilated Residual Harmonic Network (SpectralCombNet v3)
# ==============================================================================

class HarmonicResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # Multi-scale harmonic dilation blocks matching blade passage harmonic comb
        self.block1 = HarmonicResBlock(base_channels, dilation=1) # Local peak profile
        self.block2 = HarmonicResBlock(base_channels, dilation=2) # 2nd harmonic (2 * f0)
        self.block3 = HarmonicResBlock(base_channels, dilation=3) # 3rd harmonic (3 * f0)
        self.block4 = HarmonicResBlock(base_channels, dilation=4) # 4th harmonic (4 * f0)

        # Bottleneck projection
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.GELU()
        )

        # Output feature dimension: (base_channels * 2) * 2 (avg + max pool) = 128
        feat_dim = base_channels * 4

        # Multi-task heads
        self.classifier_linear = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
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

        # Learnable / calibrated temperature parameter for probability calibration
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, 1, 257)
        out = self.stem(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.bottleneck(out)

        # Hybrid Global Average + Max Pooling across spectral dimension
        avg_pool = torch.mean(out, dim=-1)
        max_pool, _ = torch.max(out, dim=-1)
        feat = torch.cat([avg_pool, max_pool], dim=-1)

        # Classification logit scaled by temperature
        logit = self.classifier_linear(feat)
        drone_prob = torch.sigmoid(logit / torch.clamp(self.temperature, min=0.1, max=10.0))

        fund_freq = self.freq_head(feat)
        harmonic_purity = self.purity_head(feat)

        return drone_prob, fund_freq, harmonic_purity

    def forward_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns unscaled raw classification logit for temperature calibration."""
        out = self.stem(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.bottleneck(out)

        avg_pool = torch.mean(out, dim=-1)
        max_pool, _ = torch.max(out, dim=-1)
        feat = torch.cat([avg_pool, max_pool], dim=-1)

        logit = self.classifier_linear(feat)
        fund_freq = self.freq_head(feat)
        harmonic_purity = self.purity_head(feat)
        return logit, fund_freq, harmonic_purity


# ==============================================================================
# Temperature Calibration (Platt / Guo et al. Scaling)
# ==============================================================================

class TemperatureCalibrator(nn.Module):
    """
    Optimizes temperature T on validation logits to minimize Negative Log-Likelihood (NLL).
    Prevents overconfident neural predictions on unseen real clutter.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.clamp(self.temperature, min=0.1, max=10.0)

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 100) -> float:
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        bce_loss = nn.BCEWithLogitsLoss()

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = bce_loss(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())


# ==============================================================================
# Training and Evaluation Engine
# ==============================================================================

def train_spectral_combnet_v3(epochs: int = 20,
                              samples_per_epoch: int = 80000,
                              batch_size: int = 128,
                              lr: float = 1e-3,
                              real_spectra_path: str = "real_darkroom_spectra.bin") -> SpectralCombNet:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("  Predator SpectralCombNet v3 Training Pipeline (Phase 33.6d)")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 75)

    model = SpectralCombNet(in_bins=257, base_channels=32).to(device)

    # Multi-task loss functions with elementwise reduction for hard negative weighting
    bce_loss_fn = nn.BCELoss(reduction="none")
    freq_loss_fn = nn.SmoothL1Loss()
    purity_loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"[INFO] Initializing CombinedSpectralDataset (samples={samples_per_epoch})...")
    full_dataset = CombinedSpectralDataset(
        num_samples=samples_per_epoch,
        real_spectra_path=real_spectra_path,
        positive_ratio=0.45,
        real_negative_ratio=0.30
    )

    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[INFO] Train samples: {train_size:,} | Validation samples: {val_size:,}")
    print(f"[INFO] Batch size: {batch_size} | Steps per epoch: {len(train_loader):,}")
    print("-" * 75)

    best_val_loss = float("inf")
    best_f1 = 0.0
    best_model_path = "spectral_combnet_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_freq_loss = 0.0
        train_purity_loss = 0.0
        start_time = time.time()

        for x_batch, y_drone, y_f0, y_purity in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_drone = y_drone.to(device, non_blocking=True)
            y_f0 = y_f0.to(device, non_blocking=True)
            y_purity = y_purity.to(device, non_blocking=True)

            optimizer.zero_grad()
            drone_prob, pred_f0, pred_purity = model(x_batch)

            # 1. Classification loss with hard negative weighting (penalize noise false alarms)
            bce_raw = bce_loss_fn(drone_prob, y_drone)
            hard_neg_weight = torch.where((y_drone < 0.5) & (drone_prob > 0.15), 3.0, 1.0)
            loss_cls = torch.mean(bce_raw * hard_neg_weight)

            # 2. Octave-aware frequency loss (matches integer harmonic ratios f0, 2*f0, f0/2)
            pos_mask = (y_drone > 0.5).squeeze()
            if pos_mask.sum() > 0:
                p_f = pred_f0[pos_mask]
                t_f = y_f0[pos_mask]
                err1 = torch.abs(p_f - t_f)
                err2 = torch.abs(p_f - 2.0 * t_f)
                err_half = torch.abs(p_f - 0.5 * t_f)
                octave_err = torch.minimum(err1, torch.minimum(err2, err_half))
                loss_freq = torch.mean(octave_err / 100.0)
                loss_purity = purity_loss_fn(pred_purity[pos_mask], y_purity[pos_mask])
            else:
                loss_freq = torch.tensor(0.0, device=device)
                loss_purity = torch.tensor(0.0, device=device)

            # Negative purity penalty (clutter must have near-zero purity)
            neg_mask = (y_drone <= 0.5).squeeze()
            if neg_mask.sum() > 0:
                loss_neg_purity = purity_loss_fn(pred_purity[neg_mask], torch.zeros_like(pred_purity[neg_mask]))
            else:
                loss_neg_purity = torch.tensor(0.0, device=device)

            total_loss = loss_cls + 0.35 * loss_freq + 0.30 * loss_purity + 0.20 * loss_neg_purity
            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += total_loss.item()
            train_cls_loss += loss_cls.item()
            train_freq_loss += loss_freq.item()
            train_purity_loss += loss_purity.item()


        scheduler.step()
        elapsed = time.time() - start_time

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total_val = 0
        tp, fp, fn, tn = 0, 0, 0, 0
        freq_errors = []

        with torch.no_grad():
            for x_val, y_d_val, y_f_val, y_p_val in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                y_d_val = y_d_val.to(device, non_blocking=True)
                y_f_val = y_f_val.to(device, non_blocking=True)
                y_p_val = y_p_val.to(device, non_blocking=True)

                drone_prob, pred_f0, pred_purity = model(x_val)
                l_cls = torch.mean(bce_loss_fn(drone_prob, y_d_val))
                pos_m = (y_d_val > 0.5).squeeze()
                if pos_m.sum() > 0:
                    p_f_v = pred_f0[pos_m]
                    t_f_v = y_f_val[pos_m]
                    e1 = torch.abs(p_f_v - t_f_v)
                    e2 = torch.abs(p_f_v - 2.0 * t_f_v)
                    e_half = torch.abs(p_f_v - 0.5 * t_f_v)
                    oct_err = torch.minimum(e1, torch.minimum(e2, e_half))
                    l_freq = torch.mean(oct_err / 100.0)
                    l_pur = purity_loss_fn(pred_purity[pos_m], y_p_val[pos_m])
                    freq_errors.extend(oct_err.cpu().numpy().tolist())
                else:
                    l_freq = torch.tensor(0.0)
                    l_pur = torch.tensor(0.0)

                val_loss += (l_cls + 0.35 * l_freq + 0.30 * l_pur).item()

                pred_bin = (drone_prob >= 0.50).float()
                correct += (pred_bin == y_d_val).sum().item()
                total_val += y_d_val.size(0)

                tp += ((pred_bin == 1.0) & (y_d_val == 1.0)).sum().item()
                fp += ((pred_bin == 1.0) & (y_d_val == 0.0)).sum().item()
                fn += ((pred_bin == 0.0) & (y_d_val == 1.0)).sum().item()
                tn += ((pred_bin == 0.0) & (y_d_val == 0.0)).sum().item()


        val_acc = 100.0 * correct / max(1, total_val)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-6, precision + recall)
        mae_freq = np.mean(freq_errors) if freq_errors else 0.0

        n_train_batches = len(train_loader)
        n_val_batches = len(val_loader)
        avg_train_loss = train_loss / n_train_batches
        avg_val_loss = val_loss / n_val_batches

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Acc: {val_acc:.2f}% | P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f} | "
              f"Freq MAE: {mae_freq:.1f} Hz")

        if f1 > best_f1 or (f1 == best_f1 and avg_val_loss < best_val_loss):
            best_f1 = f1
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> [SAVED] Best checkpoint saved to {best_model_path} (F1: {best_f1:.4f})")

    # Load best weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"\n[INFO] Loaded best model checkpoint with F1: {best_f1:.4f}")

    # ==============================================================================
    # Post-Training Temperature Calibration
    # ==============================================================================
    print("[INFO] Calibrating temperature scaling on validation set...")
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x_val, y_d_val, _, _ in val_loader:
            x_val = x_val.to(device)
            logit, _, _ = model.forward_logits(x_val)
            all_logits.append(logit)
            all_labels.append(y_d_val.to(device))

    val_logits_t = torch.cat(all_logits, dim=0)
    val_labels_t = torch.cat(all_labels, dim=0)

    calibrator = TemperatureCalibrator().to(device)
    opt_temp = calibrator.fit(val_logits_t, val_labels_t)
    model.temperature.data.copy_(torch.tensor([opt_temp], device=device))
    print(f"[INFO] Calibrated Temperature parameter: T = {opt_temp:.4f}")

    # Save calibrated model
    torch.save(model.state_dict(), best_model_path)
    print(f"[INFO] Saved calibrated model to: {best_model_path}")
    print("=" * 75)
    return model


def export_to_onnx(model: SpectralCombNet, output_path: str = "spectral_combnet.onnx"):
    print(f"[INFO] Exporting SpectralCombNet v3 to self-contained ONNX: {output_path}")
    model.eval()
    dummy_input = torch.randn(1, 1, 257, dtype=torch.float32, device=next(model.parameters()).device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["spectrum_in"],
        output_names=["drone_prob", "fund_freq_hz", "harmonic_purity"],
        dynamic_axes={
            "spectrum_in": {0: "batch_size"},
            "drone_prob": {0: "batch_size"},
            "fund_freq_hz": {0: "batch_size"},
            "harmonic_purity": {0: "batch_size"}
        },
        dynamo=False
    )
    print(f"[SUCCESS] Exported ONNX model ({os.path.getsize(output_path):,} bytes)")


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    model = train_spectral_combnet_v3(epochs=epochs, samples_per_epoch=60000, batch_size=128)
    export_to_onnx(model, "spectral_combnet.onnx")

#!/usr/bin/env python3
"""
Predator — Real Orin Spectra Dataset Loader (Phase 33.6c)
Loads real spectra dumped from CudaFlickerCore on Jetson Orin Nano hardware
and provides a unified dataset mixing physical synthetic signals and real sensor noise.
"""

import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional

from synthetic_event_generator import EventLevelSyntheticGenerator

class RealSpectraDataset(Dataset):
    """
    Dataset wrapping real 257-bin spectra extracted from physical IMX636 recordings on Jetson Orin Nano.
    All samples from this dataset are negative clutter / dark-room thermal noise (y_drone = 0.0).
    """
    def __init__(self, bin_path: str = "real_darkroom_spectra.bin"):
        self.bin_path = bin_path
        self.spectra = np.empty((0, 257), dtype=np.float32)
        if os.path.exists(bin_path):
            self._load_binary(bin_path)

    def _load_binary(self, path: str):
        with open(path, "rb") as f:
            count_bytes = f.read(4)
            if len(count_bytes) < 4:
                return
            count = struct.unpack("<I", count_bytes)[0]
            bins = struct.unpack("<I", f.read(4))[0]
            if bins != 257:
                raise ValueError(f"Expected 257 bins, got {bins}")
            raw_bytes = f.read()
            expected_bytes = count * bins * 4
            if len(raw_bytes) != expected_bytes:
                actual_count = len(raw_bytes) // (bins * 4)
                count = actual_count
                raw_bytes = raw_bytes[:count * bins * 4]
            self.spectra = np.frombuffer(raw_bytes, dtype=np.float32).reshape(count, bins).copy()
        print(f"[INFO] Loaded {len(self.spectra)} real Orin spectra from: {path}")

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = self.spectra[idx]
        x_tensor = torch.from_numpy(spec).unsqueeze(0) # (1, 257)
        y_drone = torch.tensor([0.0], dtype=torch.float32)
        y_f0 = torch.tensor([0.0], dtype=torch.float32)
        y_purity = torch.tensor([0.0], dtype=torch.float32)
        return x_tensor, y_drone, y_f0, y_purity


class CombinedSpectralDataset(Dataset):
    """
    Unified training dataset for SpectralCombNet v3 (Phase 33.6c, 33.6d):
    - 50% Positives: Physical multirotor propeller harmonic combs (EventLevelSyntheticGenerator)
    - 30% Negatives (Synthetic Clutter): 1/f foliage sway, wind turbulence, step edges, AC flicker
    - 20% Negatives (Real Hardware Noise): Real IMX636 dark-room spectra recorded on Jetson Orin Nano
    """
    def __init__(self,
                 num_samples: int = 100000,
                 real_spectra_path: Optional[str] = "real_darkroom_spectra.bin",
                 positive_ratio: float = 0.50,
                 real_negative_ratio: float = 0.20):
        self.num_samples = num_samples
        self.positive_ratio = positive_ratio
        self.real_negative_ratio = real_negative_ratio
        self.synthetic_gen = EventLevelSyntheticGenerator()

        self.real_dataset: Optional[RealSpectraDataset] = None
        if real_spectra_path and os.path.exists(real_spectra_path):
            self.real_dataset = RealSpectraDataset(real_spectra_path)
            if len(self.real_dataset) == 0:
                self.real_dataset = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = np.random.rand()

        # 1. Positive: Physical Drone Propeller Harmonic Comb
        if p < self.positive_ratio:
            ts, f0, purity, _ = self.synthetic_gen.generate_positive_time_series()
            from spectral_preprocessing import compute_normalized_spectrum_torch
            ts_tensor = torch.from_numpy(ts)
            norm_spec, _ = compute_normalized_spectrum_torch(ts_tensor)
            x_tensor = norm_spec.unsqueeze(0) # (1, 257)
            y_drone = torch.tensor([1.0], dtype=torch.float32)
            y_f0 = torch.tensor([f0], dtype=torch.float32)
            y_purity = torch.tensor([purity], dtype=torch.float32)
            return x_tensor, y_drone, y_f0, y_purity

        # 2. Negative: Real Hardware Darkroom Spectra (if available)
        elif self.real_dataset is not None and len(self.real_dataset) > 0 and (p < self.positive_ratio + self.real_negative_ratio):
            real_idx = np.random.randint(0, len(self.real_dataset))
            return self.real_dataset[real_idx]

        # 3. Negative: Physical Synthetic Clutter (Foliage, AC, Step Edges, Sensor Poisson Noise)
        else:
            ts, f0, purity, _ = self.synthetic_gen.generate_negative_time_series()
            from spectral_preprocessing import compute_normalized_spectrum_torch
            ts_tensor = torch.from_numpy(ts)
            norm_spec, _ = compute_normalized_spectrum_torch(ts_tensor)
            x_tensor = norm_spec.unsqueeze(0) # (1, 257)
            y_drone = torch.tensor([0.0], dtype=torch.float32)
            y_f0 = torch.tensor([0.0], dtype=torch.float32)
            y_purity = torch.tensor([0.0], dtype=torch.float32)
            return x_tensor, y_drone, y_f0, y_purity

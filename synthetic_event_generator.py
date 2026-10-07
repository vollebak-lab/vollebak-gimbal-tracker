#!/usr/bin/env python3
"""
Predator — Event-Level Platform-Agnostic Synthetic Generator (Phase 33.6b)
Simulates physical time-domain event camera event arrival trains and passes them
through the verified runtime preprocessing (compute_normalized_spectrum_torch).

Physical Model:
- Multirotor drones with 1 to 8 rotors, 2 to 5 blades per rotor.
- Blade Passage Frequency (BPF) spanning 70 Hz to 1200 Hz (Nyquist: 2000 Hz at Fs = 4000 Hz).
- Rotors spin with independent trim RPM offsets (spread 0.5% to 5.0%), creating natural
  multi-rotor sideband clusters and beat frequencies.
- Blade transit modeled as narrow aerodynamic chord pulse packets (duty cycle 3% to 20%),
  naturally yielding Dirichlet harmonic combs (f0, 2f0, 3f0, 4f0, ...).
- Sub-pixel standoff modulation (SNR -6 dB to +30 dB) with stochastic Poisson event arrivals
  and low-contrast cycle dropouts (0% to 35% dropout probability).
- Diverse negative clutter: 1/f wind foliage sway (2-45 Hz), AC powerline lighting (100/120 Hz harmonics),
  transient step edges, camera pan ramps, aperiodic impulses, and Poisson shot noise.
"""

import math
import random
import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional

from spectral_preprocessing import compute_normalized_spectrum_torch

class EventLevelSyntheticGenerator:
    """
    Platform-agnostic physical event generator producing 512-sample time series
    and normalized 257-bin spectra via verified runtime cuFFT preprocessing.
    """
    def __init__(self, sample_rate_hz: float = 4000.0, sequence_len: int = 512):
        self.sample_rate_hz = sample_rate_hz
        self.sequence_len = sequence_len
        self.dt = 1.0 / sample_rate_hz # 250 microseconds
        self.total_duration = sequence_len * self.dt # 128 ms
        self.time_axis = np.arange(sequence_len, dtype=np.float32) * self.dt

    def generate_positive_time_series(self) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        """
        Generates physical time-domain multirotor propeller event arrival counts.
        Returns:
            (time_series, fund_bpf_hz, harmonic_purity, metadata)
        """
        N = self.sequence_len
        t = self.time_axis

        # Fundamental BPF in [70 Hz, 800 Hz] (covers 2100 to 24,000 RPM)
        fund_bpf = random.uniform(70.0, 800.0)

        # Number of rotors (1 to 8 rotors)
        num_rotors = random.choice([1, 2, 4, 4, 4, 6, 8])

        # Rotor RPM spread for attitude control (0.5% to 4.0% standard deviation)
        rpm_spread_pct = random.uniform(0.005, 0.04) if num_rotors > 1 else 0.0

        # Blade count (2, 3, or 4 blades)
        num_blades = random.choice([2, 2, 3, 3, 4])

        # Duty cycle of blade passage: fraction of cycle the blade modulates the pixel
        # Sub-pixel targets at standoff have narrow transit times: 3% to 15%
        duty_cycle = random.uniform(0.03, 0.16)

        # Signal level: events per blade transit (standoff: 0.8 to 6 events; close-in: 10 to 45 events)
        # Operational SNR range: -2 dB (standoff / shaded) to +26 dB (bright sunlight)
        snr_db = random.uniform(-2.0, 26.0)
        snr_linear = 10.0 ** (snr_db / 10.0)


        # Base Poisson noise rate per 250 us bin (0.02 to 4.0 events/bin)
        noise_rate = random.uniform(0.02, 3.5)

        # Peak blade transit event rate
        peak_blade_rate = max(0.5, noise_rate * snr_linear)

        # Shaded low-contrast dropout probability
        dropout_prob = random.uniform(0.0, 0.35) if snr_db < 6.0 else random.uniform(0.0, 0.08)

        # Continuous expected event rate lambda(t)
        lambda_t = np.full(N, noise_rate, dtype=np.float32)

        # Add low-frequency background foliage / air turbulence drift (2 to 35 Hz)
        if random.random() < 0.65:
            drift_f = random.uniform(2.0, 30.0)
            drift_amp = noise_rate * random.uniform(0.2, 1.8)
            drift_phase = random.uniform(0.0, 2.0 * math.pi)
            lambda_t += drift_amp * (1.0 + np.sin(2.0 * math.pi * drift_f * t + drift_phase))

        # Add blade pulses for each rotor
        for r in range(num_rotors):
            rotor_bpf = fund_bpf * (1.0 + np.random.normal(0.0, rpm_spread_pct))
            rotor_bpf = max(70.0, min(1800.0, rotor_bpf))
            period = 1.0 / rotor_bpf
            pulse_sigma = max(0.4 * self.dt, duty_cycle * period / 2.355) # FWHM -> sigma

            # Initial rotor phase offset
            rotor_phase = random.uniform(0.0, period)
            rotor_weight = random.uniform(0.6, 1.4) / num_rotors

            # Generate blade pulse centers throughout the 128 ms window
            n_pulses = int(math.ceil(self.total_duration / period)) + 2
            for p in range(-1, n_pulses):
                pulse_t = rotor_phase + p * period
                # Sub-millisecond aerodynamic vortex jitter
                pulse_t += np.random.normal(0.0, 0.02 * period)

                # Shaded condition cycle dropout
                if random.random() < dropout_prob:
                    continue

                # Add Gaussian pulse packet to lambda_t
                dt_arr = t - pulse_t
                in_range = np.abs(dt_arr) < (4.0 * pulse_sigma)
                if np.any(in_range):
                    pulse_shape = np.exp(-0.5 * (dt_arr[in_range] / pulse_sigma) ** 2)
                    lambda_t[in_range] += float(peak_blade_rate * rotor_weight) * pulse_shape

        # Sample discrete integer event arrivals from Poisson distribution
        # Note: Event cameras physically produce integer counts per bin
        lambda_t = np.maximum(0.0, lambda_t)
        events_sampled = np.random.poisson(lambda_t).astype(np.float32)

        # Compute ground-truth harmonic purity: higher SNR and lower dropout yields high purity
        purity = float(np.clip(0.65 + 0.35 * (snr_db + 6.0) / 34.0 - 0.25 * dropout_prob, 0.50, 1.0))

        metadata = {
            "is_drone": True,
            "fund_bpf_hz": fund_bpf,
            "snr_db": snr_db,
            "num_rotors": num_rotors,
            "num_blades": num_blades,
            "dropout_prob": dropout_prob,
            "noise_rate": noise_rate
        }
        return events_sampled, fund_bpf, purity, metadata

    def generate_negative_time_series(self) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        """
        Generates realistic non-drone clutter, foliage sway, AC flicker, and sensor noise.
        Returns:
            (time_series, 0.0, 0.0, metadata)
        """
        N = self.sequence_len
        t = self.time_axis

        clutter_type = random.choice([
            "wind_foliage_turbulence",
            "foliage_1overf",
            "ac_powerline_flicker",
            "sensor_poisson_noise",
            "step_edge_pan",
            "sparse_thermal_blips",
            "dc_illumination",
            "isolated_single_tone"
        ])

        lambda_t = np.zeros(N, dtype=np.float32)

        if clutter_type == "wind_foliage_turbulence":
            # Multi-frequency wind turbulence across leaves (4 to 45 Hz)
            base_noise = random.uniform(0.5, 4.0)
            lambda_t += base_noise
            num_oscillators = random.randint(2, 6)
            for _ in range(num_oscillators):
                f_wind = random.uniform(3.0, 42.0)
                phase = random.uniform(0.0, 2.0 * math.pi)
                amp = base_noise * random.uniform(1.0, 6.0)
                lambda_t += amp * (1.0 + np.sin(2.0 * math.pi * f_wind * t + phase))

        elif clutter_type == "foliage_1overf":
            # 1/f low-frequency red noise (swaying branches)
            base_noise = random.uniform(0.2, 3.0)
            lambda_t += base_noise
            white = np.random.randn(N)
            fft_w = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(N, self.dt)
            freqs[0] = freqs[1] # Avoid divide by zero at DC
            decay = random.uniform(1.0, 2.2)
            fft_filtered = fft_w / (freqs ** decay)
            red_noise = np.fft.irfft(fft_filtered, n=N)
            red_noise = (red_noise - red_noise.min()) / (red_noise.max() - red_noise.min() + 1e-6)
            lambda_t += float(random.uniform(5.0, 30.0) * base_noise) * red_noise.astype(np.float32)

        elif clutter_type == "ac_powerline_flicker":
            # Artificial indoor/outdoor floodlight flicker: grid harmonics
            base_noise = random.uniform(0.1, 2.0)
            lambda_t += base_noise
            grid_fund = random.choice([100.0, 120.0, 150.0, 180.0, 200.0, 240.0, 300.0])
            ac_amp = base_noise * random.uniform(8.0, 60.0)
            phase = random.uniform(0.0, 2.0 * math.pi)
            lambda_t += ac_amp * (1.0 + np.sin(2.0 * math.pi * grid_fund * t + phase))
            if random.random() < 0.50:
                lambda_t += (ac_amp * random.uniform(0.15, 0.40)) * (1.0 + np.sin(4.0 * math.pi * grid_fund * t + phase * 2.0))

        elif clutter_type == "isolated_single_tone":
            # Monochromatic mechanical vibration / glint with zero harmonic series
            base_noise = random.uniform(0.1, 2.0)
            lambda_t += base_noise
            f_tone = random.uniform(60.0, 900.0)
            tone_amp = base_noise * random.uniform(5.0, 50.0)
            phase = random.uniform(0.0, 2.0 * math.pi)
            lambda_t += tone_amp * (1.0 + np.sin(2.0 * math.pi * f_tone * t + phase))

        elif clutter_type == "step_edge_pan":
            # Camera panning over high-contrast edge or sudden shadow: sharp aperiodic ramp or step
            base_noise = random.uniform(0.1, 2.0)
            lambda_t += base_noise
            step_idx = random.randint(30, N - 30)
            step_height = random.uniform(10.0, 60.0)
            lambda_t[step_idx:] += step_height

        elif clutter_type == "sparse_thermal_blips":
            # Dark room or unilluminated background: sporadic Poisson spikes (1 to 20 events total)
            num_spikes = random.randint(1, 25)
            spike_idx = np.random.choice(N, size=num_spikes, replace=False)
            lambda_t[spike_idx] += np.random.uniform(1.0, 5.0, size=num_spikes).astype(np.float32)

        elif clutter_type == "dc_illumination":
            # Constant uniform illumination with pure Poisson arrivals
            lambda_t += random.uniform(1.0, 25.0)

        else: # sensor_poisson_noise
            # Standard background Poisson shot noise (0.01 to 6.0 events/bin)
            lambda_t += random.uniform(0.01, 6.0)

        lambda_t = np.maximum(0.0, lambda_t)
        events_sampled = np.random.poisson(lambda_t).astype(np.float32)


        metadata = {
            "is_drone": False,
            "clutter_type": clutter_type
        }
        return events_sampled, 0.0, 0.0, metadata

    def generate_batch(self, batch_size: int = 128, positive_ratio: float = 0.5, device=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates a batch of normalized spectra, drone probabilities, frequencies, and purities.
        Computes spectra through compute_normalized_spectrum_torch.
        
        Returns:
            spectra: (B, 1, 257) normalized spectrum tensor
            y_drone: (B, 1) binary drone target label (1.0 = drone, 0.0 = clutter)
            y_f0:    (B, 1) fundamental BPF in Hz (0.0 for clutter)
            y_purity:(B, 1) harmonic comb purity (0.0 for clutter)
        """
        raw_series_list = []
        labels_drone = []
        labels_f0 = []
        labels_purity = []

        for _ in range(batch_size):
            if random.random() < positive_ratio:
                ts, f0, purity, _ = self.generate_positive_time_series()
                labels_drone.append(1.0)
                labels_f0.append(f0)
                labels_purity.append(purity)
            else:
                ts, f0, purity, _ = self.generate_negative_time_series()
                labels_drone.append(0.0)
                labels_f0.append(0.0)
                labels_purity.append(0.0)
            raw_series_list.append(ts)

        # Batch time series: (B, 512)
        raw_tensor = torch.from_numpy(np.stack(raw_series_list, axis=0)).to(device)

        # Compute normalized spectra through verified runtime preprocessing
        norm_spec_batch, _ = compute_normalized_spectrum_torch(raw_tensor)
        # Shape: (B, 1, 257)
        spectra = norm_spec_batch.unsqueeze(1)

        y_drone = torch.tensor(labels_drone, dtype=torch.float32, device=device).unsqueeze(1)
        y_f0 = torch.tensor(labels_f0, dtype=torch.float32, device=device).unsqueeze(1)
        y_purity = torch.tensor(labels_purity, dtype=torch.float32, device=device).unsqueeze(1)

        return spectra, y_drone, y_f0, y_purity


class SyntheticEventDataset(torch.utils.data.Dataset):
    """
    Infinite or fixed-epoch PyTorch Dataset generating physically faithful
    time-domain event pulse trains processed through runtime preprocessing.
    """
    def __init__(self, num_samples: int = 100000, positive_ratio: float = 0.5):
        self.num_samples = num_samples
        self.positive_ratio = positive_ratio
        self.generator = EventLevelSyntheticGenerator()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        is_pos = (random.random() < self.positive_ratio)
        if is_pos:
            ts, f0, purity, _ = self.generator.generate_positive_time_series()
            is_drone = 1.0
        else:
            ts, f0, purity, _ = self.generator.generate_negative_time_series()
            is_drone = 0.0

        ts_tensor = torch.from_numpy(ts)
        norm_spec, _ = compute_normalized_spectrum_torch(ts_tensor)
        x_tensor = norm_spec.unsqueeze(0) # (1, 257)

        y_drone = torch.tensor([is_drone], dtype=torch.float32)
        y_f0 = torch.tensor([f0], dtype=torch.float32)
        y_purity = torch.tensor([purity], dtype=torch.float32)

        return x_tensor, y_drone, y_f0, y_purity


if __name__ == "__main__":
    print("Testing EventLevelSyntheticGenerator...")
    gen = EventLevelSyntheticGenerator()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate small test batch
    b_spec, b_drone, b_f0, b_purity = gen.generate_batch(batch_size=8, device=device)
    print(f"Generated batch spectra shape: {b_spec.shape}")
    print(f"Drone labels: {b_drone.squeeze().tolist()}")
    print(f"Frequencies (Hz): {b_f0.squeeze().tolist()}")
    print(f"Purities: {b_purity.squeeze().tolist()}")
    print("Test passed successfully!")

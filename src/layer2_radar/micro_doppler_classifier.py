# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: STFT-based blade-rate extraction with harmonic analysis classifies rotary UAS vs birds/clutter using only Doppler time-series data
#   FAILURE_MODE: Simple Doppler threshold cannot distinguish drone micro-Doppler from bird wing-beat or wind-blown debris
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: [numpy, scipy]
# ---
"""
Micro-Doppler Drone Propeller Classifier.

Extracts blade-pass frequency and harmonics from range-gated Doppler
time series using STFT. Classifies targets as Rotary UAS, Fixed Wing,
Bird, or Clutter based on spectral features.

This module is pure scipy/numpy — no Uhnder SDK dependency. Can be
tested with synthetic Doppler data from ``SimulatedRadarBackend``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

from src.layer2_radar.radar_interface import DopplerTimeSeries

logger = logging.getLogger(__name__)


class TargetClass(Enum):
    """Classification result for a radar target."""
    ROTARY_UAS = auto()     # Multi-rotor drone
    FIXED_WING = auto()     # Fixed-wing drone or aircraft
    BIRD = auto()           # Avian target (wing-beat)
    CLUTTER = auto()        # Environmental clutter
    UNKNOWN = auto()        # Insufficient data


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Output of the micro-Doppler classifier.

    Attributes:
        target_class: Classified target type.
        confidence: Classification confidence (0.0–1.0).
        fundamental_hz: Detected blade-pass fundamental frequency.
        harmonics_hz: Detected harmonic frequencies.
        harmonic_strengths_db: Relative strength of each harmonic.
        spectral_entropy: Shannon entropy of the Doppler spectrum.
    """
    target_class: TargetClass
    confidence: float
    fundamental_hz: float
    harmonics_hz: list[float]
    harmonic_strengths_db: list[float]
    spectral_entropy: float


class MicroDopplerClassifier:
    """STFT-based micro-Doppler feature extraction and classification.

    Processes Doppler time-series data to extract blade-pass frequency
    signatures and classify targets. Multi-rotor drones produce
    characteristic harmonic ladders at the blade-pass frequency
    (typically 50–500 Hz for consumer quadcopters).

    Args:
        stft_window_ms: STFT window duration in milliseconds.
        stft_overlap: STFT window overlap fraction (0.0–1.0).
        min_blade_hz: Minimum expected blade-pass frequency.
        max_blade_hz: Maximum expected blade-pass frequency.
        bird_max_hz: Maximum expected bird wing-beat frequency.
        min_harmonics: Minimum number of harmonics to classify as rotary UAS.
        harmonic_snr_threshold_db: Minimum harmonic strength above noise floor.
    """

    def __init__(
        self,
        stft_window_ms: float = 50.0,
        stft_overlap: float = 0.75,
        min_blade_hz: float = 50.0,
        max_blade_hz: float = 500.0,
        bird_max_hz: float = 15.0,
        min_harmonics: int = 2,
        harmonic_snr_threshold_db: float = 6.0,
    ) -> None:
        self._stft_window_ms = stft_window_ms
        self._stft_overlap = stft_overlap
        self._min_blade_hz = min_blade_hz
        self._max_blade_hz = max_blade_hz
        self._bird_max_hz = bird_max_hz
        self._min_harmonics = min_harmonics
        self._harmonic_snr_threshold_db = harmonic_snr_threshold_db

        # Stats
        self.classifications_total: int = 0

    def classify(
        self, doppler_ts: DopplerTimeSeries,
    ) -> ClassificationResult:
        """Classify a target from its Doppler time series.

        Processing pipeline:
        1. Remove DC offset (mean Doppler = bulk radial velocity)
        2. Apply STFT to extract spectrogram
        3. Compute mean power spectral density
        4. Find peaks in PSD
        5. Check for harmonic ladder (rotary UAS signature)
        6. Check for low-frequency periodicity (bird wing-beat)

        Args:
            doppler_ts: Doppler time series for the target.

        Returns:
            ClassificationResult with target class and features.
        """
        doppler = doppler_ts.doppler_mps.astype(np.float64)
        timestamps = doppler_ts.timestamps_us.astype(np.float64)

        if doppler.size < 32:
            return self._unknown_result()

        # Estimate sample rate from timestamps
        dt_us = np.median(np.diff(timestamps))
        if dt_us <= 0:
            return self._unknown_result()
        fs = 1e6 / dt_us

        # Remove DC offset (bulk radial velocity)
        doppler_ac = doppler - np.mean(doppler)

        # STFT parameters
        nperseg = int(self._stft_window_ms * fs / 1000.0)
        nperseg = max(nperseg, 16)  # Floor at 16 samples
        noverlap = int(nperseg * self._stft_overlap)

        # Compute STFT
        try:
            freqs, _, Zxx = scipy_signal.stft(
                doppler_ac,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                window="hann",
            )
        except Exception:
            logger.exception("STFT computation failed")
            return self._unknown_result()

        # Mean power spectral density (across time)
        psd = np.mean(np.abs(Zxx) ** 2, axis=1)

        # Only look at positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        psd_pos = psd[pos_mask]

        if freqs_pos.size == 0:
            return self._unknown_result()

        # Convert to dB
        psd_db = 10 * np.log10(psd_pos + 1e-30)
        noise_floor_db = float(np.median(psd_db))

        # Spectral entropy (measure of spectral "peakiness")
        psd_norm = psd_pos / (psd_pos.sum() + 1e-30)
        spectral_entropy = float(
            -np.sum(psd_norm * np.log2(psd_norm + 1e-30))
        )

        # Find spectral peaks
        peak_indices, peak_props = scipy_signal.find_peaks(
            psd_db,
            height=noise_floor_db + self._harmonic_snr_threshold_db,
            distance=max(1, int(5.0 / (freqs_pos[1] - freqs_pos[0]))),
        )

        if len(peak_indices) == 0:
            self.classifications_total += 1
            return ClassificationResult(
                target_class=TargetClass.CLUTTER,
                confidence=0.7,
                fundamental_hz=0.0,
                harmonics_hz=[],
                harmonic_strengths_db=[],
                spectral_entropy=spectral_entropy,
            )

        peak_freqs = freqs_pos[peak_indices]
        peak_heights = psd_db[peak_indices] - noise_floor_db

        # --- Check for harmonic ladder (rotary UAS) ---
        fundamental, harmonics, harmonic_strengths = (
            self._find_harmonic_ladder(peak_freqs, peak_heights)
        )

        if (
            self._min_blade_hz <= fundamental <= self._max_blade_hz
            and len(harmonics) >= self._min_harmonics
        ):
            confidence = min(1.0, len(harmonics) / 5.0 + 0.3)
            self.classifications_total += 1
            return ClassificationResult(
                target_class=TargetClass.ROTARY_UAS,
                confidence=confidence,
                fundamental_hz=fundamental,
                harmonics_hz=harmonics,
                harmonic_strengths_db=harmonic_strengths,
                spectral_entropy=spectral_entropy,
            )

        # --- Check for bird wing-beat ---
        low_freq_peaks = peak_freqs[peak_freqs <= self._bird_max_hz]
        if len(low_freq_peaks) > 0:
            bird_fundamental = float(low_freq_peaks[0])
            self.classifications_total += 1
            return ClassificationResult(
                target_class=TargetClass.BIRD,
                confidence=0.5,
                fundamental_hz=bird_fundamental,
                harmonics_hz=[],
                harmonic_strengths_db=[],
                spectral_entropy=spectral_entropy,
            )

        # --- Default: UNKNOWN ---
        self.classifications_total += 1
        return ClassificationResult(
            target_class=TargetClass.UNKNOWN,
            confidence=0.3,
            fundamental_hz=float(peak_freqs[0]) if len(peak_freqs) > 0 else 0.0,
            harmonics_hz=[],
            harmonic_strengths_db=[],
            spectral_entropy=spectral_entropy,
        )

    def _find_harmonic_ladder(
        self,
        peak_freqs: np.ndarray,
        peak_heights: np.ndarray,
        tolerance_ratio: float = 0.1,
    ) -> tuple[float, list[float], list[float]]:
        """Search for a harmonic ladder in detected spectral peaks.

        Tries each peak as a candidate fundamental and checks how
        many integer multiples appear in the peak list.

        Args:
            peak_freqs: Detected peak frequencies.
            peak_heights: Peak heights above noise floor.
            tolerance_ratio: Allowed frequency deviation from ideal harmonic.

        Returns:
            Tuple of (fundamental_hz, harmonic_frequencies, harmonic_strengths).
        """
        best_fundamental = 0.0
        best_harmonics: list[float] = []
        best_strengths: list[float] = []

        for i, candidate_f0 in enumerate(peak_freqs):
            if candidate_f0 < self._min_blade_hz:
                continue
            if candidate_f0 > self._max_blade_hz:
                continue

            harmonics = [float(candidate_f0)]
            strengths = [float(peak_heights[i])]

            # Check for harmonics 2f, 3f, 4f, ...
            for n in range(2, 8):
                expected_freq = candidate_f0 * n
                tolerance = expected_freq * tolerance_ratio

                matches = np.where(
                    np.abs(peak_freqs - expected_freq) < tolerance
                )[0]

                if len(matches) > 0:
                    best_match = matches[np.argmin(
                        np.abs(peak_freqs[matches] - expected_freq)
                    )]
                    harmonics.append(float(peak_freqs[best_match]))
                    strengths.append(float(peak_heights[best_match]))

            if len(harmonics) > len(best_harmonics):
                best_fundamental = float(candidate_f0)
                best_harmonics = harmonics
                best_strengths = strengths

        return best_fundamental, best_harmonics, best_strengths

    @staticmethod
    def _unknown_result() -> ClassificationResult:
        """Return an UNKNOWN classification with zero confidence."""
        return ClassificationResult(
            target_class=TargetClass.UNKNOWN,
            confidence=0.0,
            fundamental_hz=0.0,
            harmonics_hz=[],
            harmonic_strengths_db=[],
            spectral_entropy=0.0,
        )

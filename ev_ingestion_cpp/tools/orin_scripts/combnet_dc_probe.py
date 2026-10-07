#!/usr/bin/env python3
"""
Hypothesis probe (read-only, does not touch the running service):

H1: SpectralCombNet outputs ~0 on every live cell because real cuFFT spectra of
    Hann-windowed, non-negative event-count series carry a large DC / low-bin
    leakage component that is ABSENT from the synthetic positive training class
    but PRESENT in the 'sparse_poisson' negative class (which was generated from
    a real time series). The net learned "DC present => clutter".

Test: build the SAME 250 Hz rotor signal three ways and run them through the
deployed TensorRT engine:
  A) training-style synthetic spectrum (exponential floor + Gaussian comb, no DC)
  B) realistic event-count time series -> Hann -> rFFT power (exactly what
     cuda_flicker_core.cu feeds the net; DC retained)
  C) same as B but mean-subtracted before windowing (DC removed)
Expected if H1 true: A ~ 1.0, B ~ 0.0, C >> B.
"""
import numpy as np
import sys

ENGINE = "/home/orin/ev_deploy/models/spectral_combnet_fp16.engine"
FS, N, NB = 4000.0, 512, 257
rng = np.random.default_rng(7)
HANN = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N) / (N - 1)))  # matches c_hanning_window / trainer


def normalize(p):
    """Runtime normalization from get_active_cells_with_spectra (median floor 1e-4)."""
    med = max(1e-4, float(np.median(p[5:128])))
    return np.log10(1.0 + p / med).astype(np.float32)


def synth_training_positive(f0=250.0, snr_db=12.0):
    """Mirror of train_spectral_combnet.py positive generator."""
    df = FS / N
    base = 1.5
    spec = rng.exponential(base, NB).astype(np.float32)
    peak = base * 10 ** (snr_db / 10)
    for h in range(1, 4):
        bp = h * f0 / df
        c = int(round(bp))
        for o in (-2, -1, 0, 1, 2):
            k = c + o
            if 0 <= k < NB:
                spec[k] += (peak / h ** 1.2) * np.exp(-0.5 * ((k - bp) / 0.75) ** 2)
    return spec


def event_series(f0=250.0, sig_events_per_pass=1.0, bg_rate_hz=200.0):
    """Realistic 4 kHz count series: periodic blade bursts + Poisson background."""
    t = np.arange(N) / FS
    x = rng.poisson(bg_rate_hz / FS, N).astype(np.float32)
    phase = (t * f0) % 1.0
    burst = (phase < 0.08).astype(np.float32)          # short blade-pass pulse
    x += rng.poisson(burst * sig_events_per_pass * 2.0)
    return x


def power(x, remove_dc):
    if remove_dc:
        x = x - x.mean()
    return (np.abs(np.fft.rfft(x * HANN)) ** 2).astype(np.float32)


def run_trt(batch):
    import tensorrt as trt
    import ctypes
    cudart = ctypes.CDLL("libcudart.so")
    logger = trt.Logger(trt.Logger.ERROR)
    with open(ENGINE, "rb") as f:
        eng = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    B = batch.shape[0]
    ctx.set_input_shape("spectrum_in", (B, 1, NB))

    def dmalloc(nbytes):
        p = ctypes.c_void_p()
        assert cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes)) == 0
        return p

    inp = np.ascontiguousarray(batch.reshape(B, 1, NB), dtype=np.float32)
    outs = {n: np.zeros((B, 1), np.float32) for n in ("drone_prob", "fund_freq_hz", "harmonic_purity")}
    d_in = dmalloc(inp.nbytes)
    d_out = {n: dmalloc(a.nbytes) for n, a in outs.items()}
    H2D, D2H = 1, 2
    cudart.cudaMemcpy(d_in, inp.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(inp.nbytes), H2D)
    ctx.set_tensor_address("spectrum_in", d_in.value)
    for n, p in d_out.items():
        ctx.set_tensor_address(n, p.value)
    assert ctx.execute_async_v3(0)
    cudart.cudaDeviceSynchronize()
    for n, a in outs.items():
        cudart.cudaMemcpy(a.ctypes.data_as(ctypes.c_void_p), d_out[n], ctypes.c_size_t(a.nbytes), D2H)
    return outs["drone_prob"].ravel()


def main():
    cases, labels = [], []
    for snr in (6.0, 12.0, 20.0):
        cases.append(normalize(synth_training_positive(snr_db=snr))); labels.append(f"A synth-train  SNR={snr:4.1f}dB")
    for spp in (0.5, 1.0, 3.0):
        x = event_series(sig_events_per_pass=spp)
        cases.append(normalize(power(x, False))); labels.append(f"B real-FFT DC  sig/pass={spp}")
        cases.append(normalize(power(x, True)));  labels.append(f"C real-FFT noDC sig/pass={spp}")
    xb = event_series(sig_events_per_pass=0.0)
    cases.append(normalize(power(xb, False))); labels.append("N bg-only DC")
    cases.append(normalize(power(xb, True)));  labels.append("N bg-only noDC")

    batch = np.stack(cases)
    # Diagnostic: how big is the DC bin relative to the floor in the normalized input?
    probs = run_trt(batch)
    print(f"{'case':34s} {'drone_prob':>10s} {'norm[0]':>8s} {'norm[1]':>8s} {'norm[32]':>8s}")
    for lab, s, p in zip(labels, batch, probs):
        print(f"{lab:34s} {p:10.4f} {s[0]:8.2f} {s[1]:8.2f} {s[32]:8.2f}")


if __name__ == "__main__":
    sys.exit(main())

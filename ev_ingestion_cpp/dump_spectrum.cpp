/**
 * @file dump_spectrum.cpp
 * @brief Spectrum dump tool for Phase 33.6a runtime preprocessing parity verification.
 *
 * Runs deterministic time series through CudaFlickerCore (cuFFT + Hanning + median noise log-norm)
 * and dumps inputs and normalized 257-bin spectra to JSON for Python reference parity validation.
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>

#include "cuda_flicker_core.cuh"

namespace {

struct TestCase {
    std::string name;
    std::vector<float> time_series; // 512 samples
    std::vector<float> spectrum_cpp; // 257 bins
    float median_noise{0.0f};
};

std::vector<TestCase> generate_test_cases() {
    std::vector<TestCase> cases;
    constexpr int N = 512;
    constexpr double Fs = 4000.0;

    // Case 1: Pure 200 Hz Sine Wave
    {
        TestCase tc;
        tc.name = "sine_200hz";
        tc.time_series.resize(N);
        for (int t = 0; t < N; ++t) {
            double time_s = t / Fs;
            tc.time_series[t] = static_cast<float>(10.0 * (1.0 + std::sin(2.0 * M_PI * 200.0 * time_s)));
        }
        cases.push_back(tc);
    }

    // Case 2: Multi-Harmonic Comb (200 Hz + 400 Hz + 600 Hz with 1/f decay)
    {
        TestCase tc;
        tc.name = "harmonic_comb_200_400_600hz";
        tc.time_series.resize(N);
        for (int t = 0; t < N; ++t) {
            double time_s = t / Fs;
            double sig = 15.0 * std::sin(2.0 * M_PI * 200.0 * time_s)
                       +  7.5 * std::sin(2.0 * M_PI * 400.0 * time_s)
                       +  3.75 * std::sin(2.0 * M_PI * 600.0 * time_s)
                       +  5.0; // DC offset
            // Add a small deterministic pseudo-random noise floor
            double noise = 0.5 * std::sin(t * 13.37 + 1.2);
            tc.time_series[t] = static_cast<float>(sig + noise);
        }
        cases.push_back(tc);
    }

    // Case 3: Delta Impulse (Single sample spike)
    {
        TestCase tc;
        tc.name = "delta_impulse_mid";
        tc.time_series.assign(N, 0.0f);
        tc.time_series[256] = 100.0f;
        cases.push_back(tc);
    }

    // Case 4: Step Edge / Aperiodic Motion
    {
        TestCase tc;
        tc.name = "step_edge";
        tc.time_series.resize(N);
        for (int t = 0; t < N; ++t) {
            tc.time_series[t] = (t < 256) ? 2.0f : 45.0f;
        }
        cases.push_back(tc);
    }

    // Case 5: Sparse Event Pulse Train (5 pulses at 100 Hz, with 3 sub-samples each)
    {
        TestCase tc;
        tc.name = "sparse_pulse_train_100hz";
        tc.time_series.assign(N, 0.0f);
        // At 4000 Hz, 100 Hz has period of 40 samples
        for (int p = 0; p < 12; ++p) {
            int center = 20 + p * 40;
            if (center + 1 < N) {
                tc.time_series[center - 1] += 3.0f;
                tc.time_series[center]     += 8.0f;
                tc.time_series[center + 1] += 3.0f;
            }
        }
        cases.push_back(tc);
    }

    // Case 6: Uniform Pseudorandom Noise
    {
        TestCase tc;
        tc.name = "pseudo_random_noise";
        tc.time_series.resize(N);
        uint32_t lcg = 123456789;
        for (int t = 0; t < N; ++t) {
            lcg = lcg * 1664525u + 1013904223u;
            float val = static_cast<float>((lcg >> 16) & 0x7FFF) / 32767.0f * 10.0f;
            tc.time_series[t] = val;
        }
        cases.push_back(tc);
    }

    // Case 7: Constant DC Signal
    {
        TestCase tc;
        tc.name = "dc_constant";
        tc.time_series.assign(N, 25.0f);
        cases.push_back(tc);
    }

    // Case 8: High Frequency Tone (1800 Hz)
    {
        TestCase tc;
        tc.name = "high_freq_1800hz";
        tc.time_series.resize(N);
        for (int t = 0; t < N; ++t) {
            double time_s = t / Fs;
            tc.time_series[t] = static_cast<float>(12.0 * (1.0 + std::sin(2.0 * M_PI * 1800.0 * time_s)));
        }
        cases.push_back(tc);
    }

    return cases;
}

void write_json(const std::string& out_path, const std::vector<TestCase>& cases) {
    std::ofstream out(out_path);
    if (!out.good()) {
        throw std::runtime_error("Failed to open output file: " + out_path);
    }

    out << "[\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& tc = cases[i];
        out << "  {\n";
        out << "    \"name\": \"" << tc.name << "\",\n";
        out << "    \"median_noise\": " << std::setprecision(8) << tc.median_noise << ",\n";

        // time_series
        out << "    \"time_series\": [";
        for (size_t t = 0; t < tc.time_series.size(); ++t) {
            out << std::setprecision(8) << tc.time_series[t];
            if (t + 1 < tc.time_series.size()) out << ", ";
        }
        out << "],\n";

        // spectrum_cpp
        out << "    \"spectrum_cpp\": [";
        for (size_t k = 0; k < tc.spectrum_cpp.size(); ++k) {
            out << std::setprecision(8) << tc.spectrum_cpp[k];
            if (k + 1 < tc.spectrum_cpp.size()) out << ", ";
        }
        out << "]\n";

        out << "  }" << (i + 1 < cases.size() ? "," : "") << "\n";
    }
    out << "]\n";
}

} // anonymous namespace

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "  Predator CudaFlickerCore Spectrum Dump Tool (33.6a)   \n";
    std::cout << "========================================================\n";

    std::string out_path = "spectral_parity_ref.json";
    if (argc > 1) {
        out_path = argv[1];
    }

    predator::CudaFlickerCore cuda_core(1280, 720, 32, 18, 4000.0, 512);

    auto test_cases = generate_test_cases();
    std::cout << "[INFO] Processing " << test_cases.size() << " test cases through CudaFlickerCore...\n";

    for (auto& tc : test_cases) {
        tc.spectrum_cpp.resize(257, 0.0f);
        cuda_core.compute_normalized_spectrum(tc.time_series.data(), tc.spectrum_cpp.data(), &tc.median_noise);
        std::cout << "  -> Processed '" << tc.name << "': median_noise=" << tc.median_noise 
                  << ", max_log_power=" << *std::max_element(tc.spectrum_cpp.begin(), tc.spectrum_cpp.end()) << "\n";
    }

    write_json(out_path, test_cases);
    std::cout << "[INFO] Successfully dumped reference spectra to: " << out_path << "\n";
    std::cout << "========================================================\n";
    return 0;
}

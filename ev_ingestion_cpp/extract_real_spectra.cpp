/**
 * @file extract_real_spectra.cpp
 * @brief Utility for Phase 33.6c: Extracts real 257-bin dark-room negative spectra from recorded IMX636 fixture.
 *
 * Replays events from darkroom_evt21.cd through CudaFlickerCore (cuFFT + Hanning + median normalization)
 * and dumps real dark-room negative spectra to a binary float file for SpectralCombNet v3 training.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>

#include "cuda_flicker_core.cuh"
#include <metavision/sdk/base/events/event_cd.h>

namespace {

struct CdRecord {
    uint16_t x;
    uint16_t y;
    int16_t p;
    int16_t reserved;
    int64_t t;
};

} // anonymous namespace

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "  Predator Real Orin Spectra Extractor (Phase 33.6c)    \n";
    std::cout << "========================================================\n";

    std::string cd_path = "/home/orin/ev_deploy/fixtures/darkroom_evt21.cd";
    std::string out_path = "real_darkroom_spectra.bin";
    size_t max_events = 2000000; // 2 million events default (~10-20 seconds of dark room)
    size_t max_spectra_to_save = 10000; // Up to 10k real spectra

    if (argc > 1) cd_path = argv[1];
    if (argc > 2) out_path = argv[2];
    if (argc > 3) max_events = std::stoull(argv[3]);
    if (argc > 4) max_spectra_to_save = std::stoull(argv[4]);

    std::ifstream cd_file(cd_path, std::ios::binary);
    if (!cd_file.good()) {
        std::cerr << "[ERROR] Failed to open fixture file: " << cd_path << "\n";
        return 1;
    }

    std::cout << "[INFO] Loading up to " << max_events << " events from: " << cd_path << "\n";
    std::vector<Metavision::EventCD> events;
    events.reserve(std::min(max_events, static_cast<size_t>(5000000)));

    CdRecord rec;
    while (events.size() < max_events && cd_file.read(reinterpret_cast<char*>(&rec), sizeof(rec))) {
        events.emplace_back(rec.x, rec.y, rec.p, rec.t);
    }
    std::cout << "[INFO] Successfully loaded " << events.size() << " events. (t_span: "
              << (events.empty() ? 0 : (events.back().t - events.front().t) / 1000) << " ms)\n";

    predator::CudaFlickerCore cuda_core(1280, 720, 32, 18, 4000.0, 512);

    std::vector<std::vector<float>> saved_spectra;
    saved_spectra.reserve(max_spectra_to_save);

    // Stream through in 40 ms slices (25 Hz analysis cadence)
    constexpr uint64_t slice_us = 40000;
    size_t event_idx = 0;
    uint64_t current_slice_end = (events.empty() ? 0 : events[0].t) + slice_us;
    size_t slice_count = 0;

    predator::Matrix3x3 H_identity = predator::Matrix3x3::identity();
    std::vector<predator::FlickerDetectionResult> raw_dets;

    while (event_idx < events.size() && saved_spectra.size() < max_spectra_to_save) {
        size_t batch_start = event_idx;
        while (event_idx < events.size() && static_cast<uint64_t>(events[event_idx].t) < current_slice_end) {
            ++event_idx;
        }
        size_t batch_len = event_idx - batch_start;

        if (batch_len > 0) {
            uint64_t raw_c = 0, ret_c = 0;
            cuda_core.ingest_event_batch(&events[batch_start], batch_len, H_identity, raw_c, ret_c, nullptr, 0.0f, true);
        }

        // Run cuFFT analysis
        raw_dets.clear();
        cuda_core.execute_batched_spectral_analysis(0.0, raw_dets);


        // Query active cells (min_events = 2.0 to capture real noise/thermal cells)
        std::vector<int> cell_indices;
        std::vector<std::vector<float>> spectra;
        cuda_core.get_active_cells_with_spectra(cell_indices, spectra, 2.0f);

        for (const auto& spec : spectra) {
            if (saved_spectra.size() < max_spectra_to_save) {
                saved_spectra.push_back(spec);
            }
        }

        current_slice_end += slice_us;
        ++slice_count;
    }

    std::cout << "[INFO] Processed " << slice_count << " 40-ms slices. Collected "
              << saved_spectra.size() << " real dark-room 257-bin spectra.\n";

    // Write binary file: [uint32_t count, uint32_t bins=257, float data...]
    std::ofstream out(out_path, std::ios::binary);
    if (!out.good()) {
        std::cerr << "[ERROR] Failed to open output file: " << out_path << "\n";
        return 1;
    }

    uint32_t count = static_cast<uint32_t>(saved_spectra.size());
    uint32_t bins = 257;
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(&bins), sizeof(bins));

    for (const auto& spec : saved_spectra) {
        out.write(reinterpret_cast<const char*>(spec.data()), 257 * sizeof(float));
    }

    std::cout << "[INFO] Saved real dark-room spectra to: " << out_path
              << " (" << (count * 257 * sizeof(float)) / 1024 << " KB)\n";
    std::cout << "========================================================\n";
    return 0;
}

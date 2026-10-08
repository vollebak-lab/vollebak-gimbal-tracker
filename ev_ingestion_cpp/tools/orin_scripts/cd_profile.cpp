// Scratch: spatial/temporal profile of an evt21_capture .cd fixture (hot pixels vs. field-wide noise).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

struct CdRecord { uint16_t x, y; int16_t p, reserved; int64_t t; };

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s file.cd\n", argv[0]); return 2; }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::perror("open"); return 2; }
    std::vector<uint32_t> count(1280 * 720, 0);
    std::vector<CdRecord> buf(1 << 20);
    uint64_t n = 0, on = 0; int64_t t0 = -1, t1 = 0;
    size_t got;
    while ((got = std::fread(buf.data(), sizeof(CdRecord), buf.size(), f)) > 0) {
        for (size_t i = 0; i < got; ++i) {
            const auto& e = buf[i];
            ++count[e.y * 1280u + e.x]; ++n; on += e.p;
            if (t0 < 0) t0 = e.t; t1 = e.t;
        }
    }
    std::fclose(f);
    const double span = 1e-6 * (t1 - t0);
    std::vector<uint32_t> sorted(count);
    std::sort(sorted.begin(), sorted.end(), std::greater<uint32_t>());
    uint64_t active = 0; for (auto c : count) active += c > 0;
    auto top_share = [&](size_t k) { uint64_t s = 0; for (size_t i = 0; i < k; ++i) s += sorted[i]; return 100.0 * s / n; };
    std::printf("events=%llu span=%.2fs rate=%.0f ev/s ON=%.4f%% active_pixels=%llu (%.1f%% of array)\n",
                (unsigned long long)n, span, n / span, 100.0 * on / n, (unsigned long long)active, 100.0 * active / (1280.0 * 720));
    std::printf("top pixel rates (ev/s): %.0f %.0f %.0f %.0f %.0f | median active pixel: %.2f ev/s\n",
                sorted[0] / span, sorted[1] / span, sorted[2] / span, sorted[3] / span, sorted[4] / span, sorted[active / 2] / span);
    std::printf("share of events from top 10 / 100 / 1000 / 10000 pixels: %.2f%% / %.2f%% / %.2f%% / %.2f%%\n",
                top_share(10), top_share(100), top_share(1000), top_share(10000));
    // Row/column structure: fraction of events in the busiest row and column.
    std::vector<uint64_t> row(720, 0), col(1280, 0);
    for (uint32_t y = 0; y < 720; ++y) for (uint32_t x = 0; x < 1280; ++x) { row[y] += count[y * 1280 + x]; col[x] += count[y * 1280 + x]; }
    const auto rmax = std::max_element(row.begin(), row.end()), cmax = std::max_element(col.begin(), col.end());
    std::printf("busiest row %ld: %.2f%% of events; busiest column %ld: %.2f%% (uniform would be %.3f%% / %.3f%%)\n",
                (long)(rmax - row.begin()), 100.0 * *rmax / n, (long)(cmax - col.begin()), 100.0 * *cmax / n, 100.0 / 720, 100.0 / 1280);
    return 0;
}

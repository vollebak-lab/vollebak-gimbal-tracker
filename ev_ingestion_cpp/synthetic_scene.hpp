#pragma once
/**
 * @file synthetic_scene.hpp
 * @brief Deterministic synthetic CD event scenes for ingest benchmarks and decoder parity tests.
 *
 * Composition: 4 rotor disks (r = 4 px, 150-261 Hz, 2 blades, an ON front then an OFF front 200 us later
 * per blade pass, rows fire within 40 us of each other), then the remainder split 50/50 between row
 * segments (8-64 px firing in the same microsecond, as moving edges do) and isolated noise.
 * Output is sorted by (t, y, p, x): the canonical order the reference EVT encoders preserve.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace predator::synth {

/**
 * @tparam Ev event type constructible as Ev(x, y, p, t) with members x, y, p, t
 * @param rate_ev_s target event rate (the rotors alone contribute ~0.1 Mev/s)
 * @param duration_s scene length in seconds
 * @param t0_us timestamp of the scene start
 * @param seed RNG seed (same seed -> identical scene)
 */
template <class Ev>
std::vector<Ev> make_scene(double rate_ev_s, double duration_s, int64_t t0_us, uint64_t seed, int width = 1280,
                           int height = 720) {
    std::mt19937_64 rng(seed);
    const int64_t span_us = static_cast<int64_t>(duration_s * 1e6);
    const size_t target   = static_cast<size_t>(rate_ev_s * duration_s);
    std::vector<Ev> ev;
    ev.reserve(target + target / 8);

    std::uniform_int_distribution<int64_t> ut(0, span_us - 1);
    std::uniform_int_distribution<int> ux(0, width - 1), uy(0, height - 1), up(0, 1);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    auto push = [&](int x, int y, int p, int64_t t) {
        ev.push_back(Ev(static_cast<unsigned short>(x), static_cast<unsigned short>(y), static_cast<short>(p), t));
    };

    // Rotors: each blade pass sweeps the disk; every disk row fires within a few microseconds.
    const int radius = 4;
    for (int r = 0; r < 4; ++r) {
        const int cx = 100 + r * 300, cy = 150 + r * 120;
        const double f_rot          = 150.0 + 37.0 * r;
        const double pass_period_us = 1e6 / (2.0 * f_rot);
        for (double tp = 0.0; tp < static_cast<double>(span_us) - 400.0; tp += pass_period_us) {
            for (int dy = -radius; dy <= radius; ++dy) {
                const int row_jitter = static_cast<int>(u01(rng) * 40.0);
                for (int pol = 1; pol >= 0; --pol) {  // ON front, then OFF front 200 us later
                    const int64_t t = static_cast<int64_t>(tp) + row_jitter + (pol ? 0 : 200);
                    for (int dx = -radius; dx <= radius; ++dx) {
                        if (dx * dx + dy * dy > radius * radius || u01(rng) > 0.8) continue;
                        push(cx + dx, cy + dy, pol, t0_us + t);
                    }
                }
            }
        }
    }

    const size_t remaining = target > ev.size() ? target - ev.size() : 0;
    // Row segments (half of the remainder).
    std::uniform_int_distribution<int> ulen(8, 64);
    size_t seg_events = 0;
    while (seg_events < remaining / 2) {
        const int len = ulen(rng);
        const int x0  = std::uniform_int_distribution<int>(0, width - len)(rng);
        const int y = uy(rng), p = up(rng);
        const int64_t t = ut(rng);
        for (int k = 0; k < len; ++k) {
            if (u01(rng) > 0.9) continue;
            push(x0 + k, y, p, t0_us + t);
            ++seg_events;
        }
    }
    // Isolated noise (the rest).
    while (ev.size() < target) push(ux(rng), uy(rng), up(rng), t0_us + ut(rng));

    std::sort(ev.begin(), ev.end(), [](const Ev& a, const Ev& b) {
        if (a.t != b.t) return a.t < b.t;
        if (a.y != b.y) return a.y < b.y;
        if (a.p != b.p) return a.p < b.p;
        return a.x < b.x;
    });
    return ev;
}

}  // namespace predator::synth

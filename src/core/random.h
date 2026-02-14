#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <random>
#include <thread>

#include "math/vec3.h"

namespace orion {

class RNG {
public:
    explicit RNG(std::uint64_t seed = defaultSeed())
        : engine_(seed), dist01_(0.0, 1.0) {}

    [[nodiscard]] double uniform() {
        return dist01_(engine_);
    }

    [[nodiscard]] double range(double min, double max) {
        return min + (max - min) * uniform();
    }

    [[nodiscard]] int rangeInt(int minInclusive, int maxInclusive) {
        std::uniform_int_distribution<int> dist(minInclusive, maxInclusive);
        return dist(engine_);
    }

    [[nodiscard]] Vec3 randomVec(double min, double max) {
        return Vec3(range(min, max), range(min, max), range(min, max));
    }

    [[nodiscard]] Vec3 inUnitSphere() {
        while (true) {
            const Vec3 p = randomVec(-1.0, 1.0);
            if (p.lengthSquared() < 1.0) {
                return p;
            }
        }
    }

    [[nodiscard]] Vec3 unitVector() {
        return normalize(inUnitSphere());
    }

    [[nodiscard]] Vec3 inUnitDisk() {
        while (true) {
            const Vec3 p(range(-1.0, 1.0), range(-1.0, 1.0), 0.0);
            if (p.lengthSquared() < 1.0) {
                return p;
            }
        }
    }

    [[nodiscard]] static std::uint64_t defaultSeed() {
        const auto now = static_cast<std::uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        const auto tid = static_cast<std::uint64_t>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()));
        return now ^ (tid + 0x9E3779B97F4A7C15ULL + (now << 6) + (now >> 2));
    }

private:
    std::mt19937_64 engine_;
    std::uniform_real_distribution<double> dist01_;
};

}  // namespace orion

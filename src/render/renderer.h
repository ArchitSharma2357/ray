#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "material/material.h"
#include "render/camera.h"
#include "render/image.h"
#include "scene/hittable.h"

namespace orion {

struct RenderSettings {
    int width = 1280;
    int height = 720;
    int samplesPerPixel = 300;
    int maxDepth = 16;
    int threadCount = static_cast<int>(std::thread::hardware_concurrency());
    std::uint64_t seed = 1337;
    double exposure = 1.0;
};

class Renderer {
public:
    explicit Renderer(RenderSettings settings)
        : settings_(std::move(settings)) {
        if (settings_.threadCount <= 0) {
            settings_.threadCount = 1;
        }
    }

    [[nodiscard]] Image render(
        const Hittable& world,
        const Camera& camera,
        const std::function<void(int, int)>& onProgress = {}) const {
        Image image(settings_.width, settings_.height);

        // Dynamic row scheduling keeps worker utilization balanced across difficult/easy rows.
        std::atomic<int> nextRow(settings_.height - 1);
        std::atomic<int> rowsDone(0);
        std::mutex progressMutex;

        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(settings_.threadCount));

        for (int threadIndex = 0; threadIndex < settings_.threadCount; ++threadIndex) {
            (void)threadIndex;
            workers.emplace_back([&]() {
                while (true) {
                    const int y = nextRow.fetch_sub(1);
                    if (y < 0) {
                        break;
                    }

                    for (int x = 0; x < settings_.width; ++x) {
                        // Derive a deterministic RNG stream per pixel for reproducible renders.
                        const std::uint64_t pixelSeed = mixSeed(
                            settings_.seed ^
                            (static_cast<std::uint64_t>(x + 1) * 0x9E3779B97F4A7C15ULL) ^
                            (static_cast<std::uint64_t>(y + 1) * 0xBF58476D1CE4E5B9ULL)
                        );
                        RNG rng(pixelSeed);
                        Color pixel(0.0, 0.0, 0.0);

                        for (int sample = 0; sample < settings_.samplesPerPixel; ++sample) {
                            const double u = (static_cast<double>(x) + rng.uniform()) /
                                             static_cast<double>(settings_.width - 1);
                            const double v = (static_cast<double>(y) + rng.uniform()) /
                                             static_cast<double>(settings_.height - 1);

                            const Ray ray = camera.sampleRay(u, v, rng);
                            pixel += trace(ray, world, rng);
                        }

                        // Flip Y so output file is top-to-bottom.
                        const int outputY = settings_.height - 1 - y;
                        image.setPixel(x, outputY, pixel);
                    }

                    const int completed = rowsDone.fetch_add(1) + 1;
                    if (onProgress) {
                        std::lock_guard<std::mutex> lock(progressMutex);
                        onProgress(completed, settings_.height);
                    }
                }
            });
        }

        for (auto& worker : workers) {
            worker.join();
        }

        return image;
    }

    [[nodiscard]] const RenderSettings& settings() const {
        return settings_;
    }

private:
    RenderSettings settings_;

    [[nodiscard]] static std::uint64_t mixSeed(std::uint64_t value) {
        value ^= value >> 30;
        value *= 0xBF58476D1CE4E5B9ULL;
        value ^= value >> 27;
        value *= 0x94D049BB133111EBULL;
        value ^= value >> 31;
        return value;
    }

    [[nodiscard]] Color trace(const Ray& initialRay, const Hittable& world, RNG& rng) const {
        // Iterative path integrator:
        // radiance = sum(emission + escaped-sky) along a stochastic bounce path.
        Ray ray = initialRay;
        Color throughput(1.0, 1.0, 1.0);
        Color radiance(0.0, 0.0, 0.0);

        for (int bounce = 0; bounce < settings_.maxDepth; ++bounce) {
            HitRecord hit;
            if (!world.hit(ray, 0.001, std::numeric_limits<double>::infinity(), hit)) {
                radiance += throughput * sky(ray.direction);
                break;
            }

            radiance += throughput * hit.material->emitted();

            Ray scattered;
            Color attenuation;
            if (!hit.material->scatter(ray, hit, rng, attenuation, scattered)) {
                break;
            }

            throughput *= attenuation;

            // Russian roulette limits long low-energy paths while preserving an unbiased estimate.
            if (bounce > 3) {
                const double maxChannel = std::max({throughput.x(), throughput.y(), throughput.z()});
                const double survive = std::clamp(maxChannel, 0.05, 0.95);

                if (rng.uniform() > survive) {
                    break;
                }

                throughput /= survive;
            }

            ray = scattered;
        }

        return radiance;
    }

    [[nodiscard]] static Color sky(const Vec3& direction) {
        // Procedural sky dome: blue gradient plus a sharp sun lobe for specular highlights.
        const Vec3 unit = normalize(direction);
        const double t = 0.5 * (unit.y() + 1.0);
        const Color gradient = (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.45, 0.67, 1.0);

        const Vec3 sunDir = normalize(Vec3(0.3, 0.8, 0.2));
        const double sun = std::pow(std::max(dot(unit, sunDir), 0.0), 384.0);
        return gradient + sun * Color(8.0, 6.5, 5.0);
    }
};

}  // namespace orion

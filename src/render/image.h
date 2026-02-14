#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "math/vec3.h"

namespace orion {

class Image {
public:
    Image(int width, int height)
        : width_(width), height_(height), pixels_(static_cast<std::size_t>(width * height), Color(0.0, 0.0, 0.0)) {}

    [[nodiscard]] int width() const { return width_; }
    [[nodiscard]] int height() const { return height_; }

    void setPixel(int x, int y, const Color& value) {
        pixels_[index(x, y)] = value;
    }

    [[nodiscard]] const Color& getPixel(int x, int y) const {
        return pixels_[index(x, y)];
    }

    void writePPM(const std::string& path, int samplesPerPixel, double exposure = 1.0) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open output file: " + path);
        }

        out << "P6\n" << width_ << ' ' << height_ << "\n255\n";

        for (const Color& rawPixel : pixels_) {
            Color color = rawPixel / static_cast<double>(samplesPerPixel);
            color *= exposure;
            color = toneMapACES(color);
            color = gammaCorrect(color, 2.2);
            color = clampVec(color, 0.0, 0.999);

            const auto r = static_cast<std::uint8_t>(256.0 * color.x());
            const auto g = static_cast<std::uint8_t>(256.0 * color.y());
            const auto b = static_cast<std::uint8_t>(256.0 * color.z());
            out.put(static_cast<char>(r));
            out.put(static_cast<char>(g));
            out.put(static_cast<char>(b));
        }
    }

private:
    int width_;
    int height_;
    std::vector<Color> pixels_;

    [[nodiscard]] std::size_t index(int x, int y) const {
        return static_cast<std::size_t>(y * width_ + x);
    }

    [[nodiscard]] static Color toneMapACES(const Color& color) {
        auto aces = [](double x) {
            constexpr double a = 2.51;
            constexpr double b = 0.03;
            constexpr double c = 2.43;
            constexpr double d = 0.59;
            constexpr double e = 0.14;
            return std::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
        };
        return Color(aces(color.x()), aces(color.y()), aces(color.z()));
    }

    [[nodiscard]] static Color gammaCorrect(const Color& color, double gamma) {
        const double invGamma = 1.0 / gamma;
        return Color(
            std::pow(std::max(color.x(), 0.0), invGamma),
            std::pow(std::max(color.y(), 0.0), invGamma),
            std::pow(std::max(color.z(), 0.0), invGamma));
    }
};

}  // namespace orion

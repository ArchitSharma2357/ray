#pragma once

#include <cmath>

#include "core/random.h"
#include "core/ray.h"

namespace orion {

struct CameraSettings {
    Point3 lookFrom{13.0, 2.0, 3.0};
    Point3 lookAt{0.0, 0.0, 0.0};
    Vec3 vUp{0.0, 1.0, 0.0};

    double verticalFovDeg = 30.0;
    double aspectRatio = 16.0 / 9.0;

    double aperture = 0.08;
    double focusDistance = 10.0;
};

class Camera {
public:
    explicit Camera(const CameraSettings& settings) {
        constexpr double pi = 3.14159265358979323846;

        const double theta = settings.verticalFovDeg * pi / 180.0;
        const double h = std::tan(theta / 2.0);

        const double viewportHeight = 2.0 * h;
        const double viewportWidth = settings.aspectRatio * viewportHeight;

        w_ = normalize(settings.lookFrom - settings.lookAt);
        u_ = normalize(cross(settings.vUp, w_));
        v_ = cross(w_, u_);

        origin_ = settings.lookFrom;
        horizontal_ = settings.focusDistance * viewportWidth * u_;
        vertical_ = settings.focusDistance * viewportHeight * v_;
        lowerLeftCorner_ = origin_ - horizontal_ / 2.0 - vertical_ / 2.0 - settings.focusDistance * w_;

        lensRadius_ = settings.aperture * 0.5;
    }

    [[nodiscard]] Ray sampleRay(double s, double t, RNG& rng) const {
        const Vec3 diskSample = lensRadius_ * rng.inUnitDisk();
        const Vec3 offset = u_ * diskSample.x() + v_ * diskSample.y();
        return Ray(
            origin_ + offset,
            lowerLeftCorner_ + s * horizontal_ + t * vertical_ - origin_ - offset);
    }

private:
    Point3 origin_;
    Point3 lowerLeftCorner_;
    Vec3 horizontal_;
    Vec3 vertical_;
    Vec3 u_;
    Vec3 v_;
    Vec3 w_;
    double lensRadius_ = 0.0;
};

}  // namespace orion

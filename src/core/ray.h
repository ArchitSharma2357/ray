#pragma once

#include "math/vec3.h"

namespace orion {

class Ray {
public:
    Point3 origin;
    Vec3 direction;

    Ray() = default;
    Ray(const Point3& originIn, const Vec3& directionIn)
        : origin(originIn), direction(directionIn) {}

    [[nodiscard]] Point3 at(double t) const {
        return origin + t * direction;
    }
};

}  // namespace orion

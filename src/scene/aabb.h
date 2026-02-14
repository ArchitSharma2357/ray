#pragma once

#include <algorithm>

#include "core/ray.h"

namespace orion {

class AABB {
public:
    Point3 minPoint;
    Point3 maxPoint;

    AABB() = default;
    AABB(const Point3& minIn, const Point3& maxIn)
        : minPoint(minIn), maxPoint(maxIn) {}

    [[nodiscard]] bool hit(const Ray& ray, double tMin, double tMax) const {
        for (int axis = 0; axis < 3; ++axis) {
            const double invD = 1.0 / ray.direction[axis];
            double t0 = (minPoint[axis] - ray.origin[axis]) * invD;
            double t1 = (maxPoint[axis] - ray.origin[axis]) * invD;
            if (invD < 0.0) {
                std::swap(t0, t1);
            }
            tMin = std::max(t0, tMin);
            tMax = std::min(t1, tMax);
            if (tMax <= tMin) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] Point3 centroid() const {
        return 0.5 * (minPoint + maxPoint);
    }
};

[[nodiscard]] inline AABB surroundingBox(const AABB& a, const AABB& b) {
    const Point3 small(
        std::min(a.minPoint.x(), b.minPoint.x()),
        std::min(a.minPoint.y(), b.minPoint.y()),
        std::min(a.minPoint.z(), b.minPoint.z()));
    const Point3 big(
        std::max(a.maxPoint.x(), b.maxPoint.x()),
        std::max(a.maxPoint.y(), b.maxPoint.y()),
        std::max(a.maxPoint.z(), b.maxPoint.z()));
    return AABB(small, big);
}

}  // namespace orion

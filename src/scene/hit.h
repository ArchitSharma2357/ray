#pragma once

#include <memory>

#include "core/ray.h"

namespace orion {

class Material;

struct HitRecord {
    Point3 point;
    Vec3 normal;
    std::shared_ptr<Material> material;
    double t = 0.0;
    double u = 0.0;
    double v = 0.0;
    bool frontFace = false;

    void setFaceNormal(const Ray& ray, const Vec3& outwardNormal) {
        frontFace = dot(ray.direction, outwardNormal) < 0.0;
        normal = frontFace ? outwardNormal : -outwardNormal;
    }
};

}  // namespace orion

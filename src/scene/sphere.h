#pragma once

#include <memory>

#include "material/material.h"
#include "scene/hittable.h"

namespace orion {

class Sphere final : public Hittable {
public:
    Point3 center;
    double radius;
    std::shared_ptr<Material> material;

    Sphere(Point3 centerIn, double radiusIn, std::shared_ptr<Material> materialIn)
        : center(centerIn), radius(radiusIn), material(std::move(materialIn)) {}

    [[nodiscard]] bool hit(const Ray& ray, double tMin, double tMax, HitRecord& record) const override {
        constexpr double pi = 3.14159265358979323846;
        const Vec3 oc = ray.origin - center;
        const double a = ray.direction.lengthSquared();
        const double halfB = dot(oc, ray.direction);
        const double c = oc.lengthSquared() - radius * radius;

        const double discriminant = halfB * halfB - a * c;
        if (discriminant < 0.0) {
            return false;
        }

        const double sqrtd = std::sqrt(discriminant);

        double root = (-halfB - sqrtd) / a;
        if (root < tMin || root > tMax) {
            root = (-halfB + sqrtd) / a;
            if (root < tMin || root > tMax) {
                return false;
            }
        }

        record.t = root;
        record.point = ray.at(record.t);
        const Vec3 outwardNormal = (record.point - center) / radius;
        record.setFaceNormal(ray, outwardNormal);
        const double theta = std::acos(std::clamp(-outwardNormal.y(), -1.0, 1.0));
        const double phi = std::atan2(-outwardNormal.z(), outwardNormal.x()) + pi;
        record.u = phi / (2.0 * pi);
        record.v = theta / pi;
        record.material = material;

        return true;
    }

    [[nodiscard]] AABB boundingBox() const override {
        const Vec3 extent(radius, radius, radius);
        return AABB(center - extent, center + extent);
    }
};

}  // namespace orion

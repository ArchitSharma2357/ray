#pragma once

#include <algorithm>
#include <memory>

#include "material/material.h"
#include "scene/hittable.h"

namespace orion {

class Triangle final : public Hittable {
public:
    Point3 v0;
    Point3 v1;
    Point3 v2;
    Vec3 uv0;
    Vec3 uv1;
    Vec3 uv2;
    bool hasUv = false;
    std::shared_ptr<Material> material;

    Triangle(Point3 a, Point3 b, Point3 c, std::shared_ptr<Material> materialIn)
        : v0(a), v1(b), v2(c), material(std::move(materialIn)) {}

    Triangle(Point3 a, Point3 b, Point3 c, Vec3 uvA, Vec3 uvB, Vec3 uvC, std::shared_ptr<Material> materialIn)
        : v0(a),
          v1(b),
          v2(c),
          uv0(uvA),
          uv1(uvB),
          uv2(uvC),
          hasUv(true),
          material(std::move(materialIn)) {}

    [[nodiscard]] bool hit(const Ray& ray, double tMin, double tMax, HitRecord& record) const override {
        constexpr double epsilon = 1e-9;

        const Vec3 edge1 = v1 - v0;
        const Vec3 edge2 = v2 - v0;

        const Vec3 pvec = cross(ray.direction, edge2);
        const double det = dot(edge1, pvec);

        if (std::fabs(det) < epsilon) {
            return false;
        }

        const double invDet = 1.0 / det;
        const Vec3 tvec = ray.origin - v0;
        const double u = dot(tvec, pvec) * invDet;
        if (u < 0.0 || u > 1.0) {
            return false;
        }

        const Vec3 qvec = cross(tvec, edge1);
        const double v = dot(ray.direction, qvec) * invDet;
        if (v < 0.0 || u + v > 1.0) {
            return false;
        }

        const double t = dot(edge2, qvec) * invDet;
        if (t < tMin || t > tMax) {
            return false;
        }

        record.t = t;
        record.point = ray.at(t);
        const Vec3 outwardNormal = normalize(cross(edge1, edge2));
        record.setFaceNormal(ray, outwardNormal);
        if (hasUv) {
            const double w = 1.0 - u - v;
            const Vec3 uv = w * uv0 + u * uv1 + v * uv2;
            record.u = uv.x();
            record.v = uv.y();
        } else {
            record.u = 0.0;
            record.v = 0.0;
        }
        record.material = material;

        return true;
    }

    [[nodiscard]] AABB boundingBox() const override {
        constexpr double pad = 1e-4;
        const Point3 minPoint(
            std::min({v0.x(), v1.x(), v2.x()}) - pad,
            std::min({v0.y(), v1.y(), v2.y()}) - pad,
            std::min({v0.z(), v1.z(), v2.z()}) - pad);
        const Point3 maxPoint(
            std::max({v0.x(), v1.x(), v2.x()}) + pad,
            std::max({v0.y(), v1.y(), v2.y()}) + pad,
            std::max({v0.z(), v1.z(), v2.z()}) + pad);
        return AABB(minPoint, maxPoint);
    }
};

}  // namespace orion

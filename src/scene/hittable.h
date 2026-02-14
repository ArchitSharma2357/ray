#pragma once

#include <limits>
#include <memory>
#include <vector>

#include "scene/aabb.h"
#include "scene/hit.h"

namespace orion {

class Hittable {
public:
    virtual ~Hittable() = default;

    [[nodiscard]] virtual bool hit(
        const Ray& ray,
        double tMin,
        double tMax,
        HitRecord& record) const = 0;

    [[nodiscard]] virtual AABB boundingBox() const = 0;
};

class HittableList final : public Hittable {
public:
    std::vector<std::shared_ptr<Hittable>> objects;

    HittableList() = default;

    explicit HittableList(std::vector<std::shared_ptr<Hittable>> objectsIn)
        : objects(std::move(objectsIn)) {}

    void add(const std::shared_ptr<Hittable>& object) {
        objects.push_back(object);
    }

    [[nodiscard]] bool hit(const Ray& ray, double tMin, double tMax, HitRecord& record) const override {
        HitRecord tempRecord;
        bool hitAnything = false;
        double closest = tMax;

        for (const auto& object : objects) {
            if (object->hit(ray, tMin, closest, tempRecord)) {
                hitAnything = true;
                closest = tempRecord.t;
                record = tempRecord;
            }
        }

        return hitAnything;
    }

    [[nodiscard]] AABB boundingBox() const override {
        if (objects.empty()) {
            return AABB(Point3(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));
        }

        AABB output = objects.front()->boundingBox();
        for (std::size_t i = 1; i < objects.size(); ++i) {
            output = surroundingBox(output, objects[i]->boundingBox());
        }
        return output;
    }
};

}  // namespace orion

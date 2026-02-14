#pragma once

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "core/random.h"
#include "scene/hittable.h"

namespace orion {

class BVHNode final : public Hittable {
public:
    BVHNode() = default;

    explicit BVHNode(std::vector<std::shared_ptr<Hittable>> srcObjects, RNG& rng)
        : BVHNode(srcObjects, 0, srcObjects.size(), rng) {}

    [[nodiscard]] bool hit(const Ray& ray, double tMin, double tMax, HitRecord& record) const override {
        if (!box_.hit(ray, tMin, tMax)) {
            return false;
        }

        HitRecord leftRecord;
        HitRecord rightRecord;

        const bool hitLeft = left_->hit(ray, tMin, tMax, leftRecord);
        const bool hitRight = right_->hit(ray, tMin, hitLeft ? leftRecord.t : tMax, rightRecord);

        if (hitRight) {
            record = rightRecord;
            return true;
        }

        if (hitLeft) {
            record = leftRecord;
            return true;
        }

        return false;
    }

    [[nodiscard]] AABB boundingBox() const override {
        return box_;
    }

private:
    std::shared_ptr<Hittable> left_;
    std::shared_ptr<Hittable> right_;
    AABB box_;

    BVHNode(std::vector<std::shared_ptr<Hittable>>& objects, std::size_t start, std::size_t end, RNG& rng) {
        const int axis = rng.rangeInt(0, 2);
        auto comparator = [axis](const std::shared_ptr<Hittable>& a, const std::shared_ptr<Hittable>& b) {
            return a->boundingBox().centroid()[axis] < b->boundingBox().centroid()[axis];
        };

        const std::size_t span = end - start;

        if (span == 1) {
            left_ = right_ = objects[start];
        } else if (span == 2) {
            if (comparator(objects[start], objects[start + 1])) {
                left_ = objects[start];
                right_ = objects[start + 1];
            } else {
                left_ = objects[start + 1];
                right_ = objects[start];
            }
        } else {
            std::sort(objects.begin() + static_cast<std::ptrdiff_t>(start),
                      objects.begin() + static_cast<std::ptrdiff_t>(end),
                      comparator);

            const std::size_t mid = start + span / 2;
            left_ = std::shared_ptr<BVHNode>(new BVHNode(objects, start, mid, rng));
            right_ = std::shared_ptr<BVHNode>(new BVHNode(objects, mid, end, rng));
        }

        box_ = surroundingBox(left_->boundingBox(), right_->boundingBox());
    }
};

}  // namespace orion

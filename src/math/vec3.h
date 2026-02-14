#pragma once

#include <algorithm>
#include <cmath>
#include <ostream>

namespace orion {

class Vec3 {
public:
    double e[3];

    constexpr Vec3() : e{0.0, 0.0, 0.0} {}
    constexpr Vec3(double x, double y, double z) : e{x, y, z} {}

    [[nodiscard]] constexpr double x() const { return e[0]; }
    [[nodiscard]] constexpr double y() const { return e[1]; }
    [[nodiscard]] constexpr double z() const { return e[2]; }

    [[nodiscard]] constexpr Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
    [[nodiscard]] constexpr double operator[](int i) const { return e[i]; }
    [[nodiscard]] constexpr double& operator[](int i) { return e[i]; }

    constexpr Vec3& operator+=(const Vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    constexpr Vec3& operator-=(const Vec3& v) {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        return *this;
    }

    constexpr Vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    constexpr Vec3& operator*=(const Vec3& v) {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }

    constexpr Vec3& operator/=(double t) {
        return *this *= (1.0 / t);
    }

    [[nodiscard]] double length() const {
        return std::sqrt(lengthSquared());
    }

    [[nodiscard]] constexpr double lengthSquared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    [[nodiscard]] bool nearZero() const {
        constexpr double s = 1e-8;
        return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
    }
};

using Point3 = Vec3;
using Color = Vec3;

inline std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    os << v.x() << ' ' << v.y() << ' ' << v.z();
    return os;
}

[[nodiscard]] constexpr Vec3 operator+(const Vec3& u, const Vec3& v) {
    return Vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

[[nodiscard]] constexpr Vec3 operator-(const Vec3& u, const Vec3& v) {
    return Vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

[[nodiscard]] constexpr Vec3 operator*(const Vec3& u, const Vec3& v) {
    return Vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

[[nodiscard]] constexpr Vec3 operator*(double t, const Vec3& v) {
    return Vec3(t * v.x(), t * v.y(), t * v.z());
}

[[nodiscard]] constexpr Vec3 operator*(const Vec3& v, double t) {
    return t * v;
}

[[nodiscard]] constexpr Vec3 operator/(const Vec3& v, double t) {
    return (1.0 / t) * v;
}

[[nodiscard]] constexpr double dot(const Vec3& u, const Vec3& v) {
    return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

[[nodiscard]] constexpr Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(
        u.y() * v.z() - u.z() * v.y(),
        u.z() * v.x() - u.x() * v.z(),
        u.x() * v.y() - u.y() * v.x());
}

[[nodiscard]] inline Vec3 normalize(const Vec3& v) {
    return v / v.length();
}

[[nodiscard]] inline Vec3 clampVec(const Vec3& v, double minVal, double maxVal) {
    return Vec3(
        std::clamp(v.x(), minVal, maxVal),
        std::clamp(v.y(), minVal, maxVal),
        std::clamp(v.z(), minVal, maxVal));
}

[[nodiscard]] inline Vec3 reflect(const Vec3& in, const Vec3& normal) {
    return in - 2.0 * dot(in, normal) * normal;
}

[[nodiscard]] inline Vec3 refract(const Vec3& uv, const Vec3& normal, double etaiOverEtat) {
    const double cosTheta = std::fmin(dot(-uv, normal), 1.0);
    const Vec3 rOutPerp = etaiOverEtat * (uv + cosTheta * normal);
    const Vec3 rOutParallel = -std::sqrt(std::fabs(1.0 - rOutPerp.lengthSquared())) * normal;
    return rOutPerp + rOutParallel;
}

}  // namespace orion

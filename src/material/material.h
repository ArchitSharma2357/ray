#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "core/random.h"
#include "scene/hit.h"

namespace orion {

struct ImageTexture {
    int width = 0;
    int height = 0;
    std::vector<Color> texels;

    [[nodiscard]] bool valid() const {
        return width > 0 && height > 0
            && texels.size() == static_cast<std::size_t>(width * height);
    }

    [[nodiscard]] Color sample(double u, double v) const {
        if (!valid()) {
            return Color(1.0, 1.0, 1.0);
        }

        double wrappedU = u - std::floor(u);
        double wrappedV = v - std::floor(v);
        if (wrappedU < 0.0) {
            wrappedU += 1.0;
        }
        if (wrappedV < 0.0) {
            wrappedV += 1.0;
        }

        const int x = std::clamp(
            static_cast<int>(wrappedU * static_cast<double>(width - 1) + 0.5),
            0,
            width - 1);
        const int y = std::clamp(
            static_cast<int>((1.0 - wrappedV) * static_cast<double>(height - 1) + 0.5),
            0,
            height - 1);
        return texels[static_cast<std::size_t>(y * width + x)];
    }
};

class Material {
public:
    virtual ~Material() = default;

    // Returns true when a new bounce ray is produced; false terminates the path at this material.
    [[nodiscard]] virtual bool scatter(
        const Ray& incoming,
        const HitRecord& hit,
        RNG& rng,
        Color& attenuation,
        Ray& scattered) const = 0;

    // Non-zero only for light-emitting materials.
    [[nodiscard]] virtual Color emitted() const {
        return Color(0.0, 0.0, 0.0);
    }
};

class Lambertian final : public Material {
public:
    explicit Lambertian(Color albedoIn)
        : albedo(albedoIn) {}

    [[nodiscard]] bool scatter(
        const Ray&,
        const HitRecord& hit,
        RNG& rng,
        Color& attenuation,
        Ray& scattered) const override {
        // Cosine-ish diffuse lobe via normal + random unit vector.
        Vec3 direction = hit.normal + rng.unitVector();
        if (direction.nearZero()) {
            direction = hit.normal;
        }
        scattered = Ray(hit.point, normalize(direction));
        attenuation = albedo;
        return true;
    }

private:
    Color albedo;
};

class Metal final : public Material {
public:
    Metal(Color albedoIn, double fuzzIn)
        : albedo(albedoIn), fuzz(std::clamp(fuzzIn, 0.0, 1.0)) {}

    [[nodiscard]] bool scatter(
        const Ray& incoming,
        const HitRecord& hit,
        RNG& rng,
        Color& attenuation,
        Ray& scattered) const override {
        // Perfect reflection perturbed by fuzz amount.
        const Vec3 reflected = reflect(normalize(incoming.direction), hit.normal);
        scattered = Ray(hit.point, normalize(reflected + fuzz * rng.inUnitSphere()));
        attenuation = albedo;
        return dot(scattered.direction, hit.normal) > 0.0;
    }

private:
    Color albedo;
    double fuzz;
};

class Dielectric final : public Material {
public:
    explicit Dielectric(double iorIn)
        : ior(iorIn) {}

    [[nodiscard]] bool scatter(
        const Ray& incoming,
        const HitRecord& hit,
        RNG& rng,
        Color& attenuation,
        Ray& scattered) const override {
        attenuation = Color(1.0, 1.0, 1.0);

        const double refractionRatio = hit.frontFace ? (1.0 / ior) : ior;
        const Vec3 unitDirection = normalize(incoming.direction);

        const double cosTheta = std::fmin(dot(-unitDirection, hit.normal), 1.0);
        const double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);

        const bool cannotRefract = refractionRatio * sinTheta > 1.0;
        Vec3 direction;

        // Schlick approximation chooses reflection/refraction stochastically.
        if (cannotRefract || reflectance(cosTheta, refractionRatio) > rng.uniform()) {
            direction = reflect(unitDirection, hit.normal);
        } else {
            direction = refract(unitDirection, hit.normal, refractionRatio);
        }

        scattered = Ray(hit.point, normalize(direction));
        return true;
    }

private:
    double ior;

    [[nodiscard]] static double reflectance(double cosine, double refIdx) {
        double r0 = (1.0 - refIdx) / (1.0 + refIdx);
        r0 *= r0;
        return r0 + (1.0 - r0) * std::pow(1.0 - cosine, 5.0);
    }
};

class Emissive final : public Material {
public:
    explicit Emissive(Color radianceIn)
        : radiance(radianceIn) {}

    [[nodiscard]] bool scatter(
        const Ray&,
        const HitRecord&,
        RNG&,
        Color&,
        Ray&) const override {
        // Light sources terminate transport in this simple material model.
        return false;
    }

    [[nodiscard]] Color emitted() const override {
        return radiance;
    }

private:
    Color radiance;
};

class CoatedDiffuse final : public Material {
public:
    CoatedDiffuse(Color baseColorIn, double coatStrengthIn, double roughnessIn)
        : baseColor(baseColorIn),
          coatStrength(std::clamp(coatStrengthIn, 0.0, 1.0)),
          roughness(std::clamp(roughnessIn, 0.0, 1.0)) {}

    [[nodiscard]] bool scatter(
        const Ray& incoming,
        const HitRecord& hit,
        RNG& rng,
        Color& attenuation,
        Ray& scattered) const override {
        // Two-lobe approximation: glossy clear-coat branch + diffuse base branch.
        const Vec3 unitIn = normalize(incoming.direction);
        const Vec3 reflected = reflect(unitIn, hit.normal);
        const Vec3 glossyDir = normalize(reflected + roughness * rng.inUnitSphere());
        const Vec3 diffuseDir = normalize(hit.normal + rng.unitVector());

        if (rng.uniform() < coatStrength) {
            scattered = Ray(hit.point, glossyDir);
            attenuation = Color(0.95, 0.95, 0.95);
        } else {
            scattered = Ray(hit.point, diffuseDir);
            attenuation = baseColor;
        }
        return true;
    }

private:
    Color baseColor;
    double coatStrength;
    double roughness;
};

class SceneMeshMaterial final : public Material {
public:
    enum class Type {
        Lambertian = 0,
        Metal = 1,
        Dielectric = 2,
        Emissive = 3,
        CoatedDiffuse = 4,
    };

    struct Params {
        Type type = Type::Lambertian;
        Color albedo{0.82, 0.82, 0.82};
        Color emission{0.0, 0.0, 0.0};
        double fuzz = 0.04;
        double ior = 1.5;
        double coatStrength = 0.45;
        double roughness = 0.12;
        std::shared_ptr<const ImageTexture> albedoTexture;
    };

    explicit SceneMeshMaterial(Params paramsIn)
        : params(std::move(paramsIn)) {}

    [[nodiscard]] bool scatter(
        const Ray& incoming,
        const HitRecord& hit,
        RNG& rng,
        Color& attenuation,
        Ray& scattered) const override {
        // Imported mesh materials are mapped into one of the built-in scattering behaviors.
        const Color textureColor = sampleAlbedo(hit.u, hit.v);

        if (params.type == Type::Emissive) {
            return false;
        }
        if (params.type == Type::Lambertian) {
            Vec3 direction = hit.normal + rng.unitVector();
            if (direction.nearZero()) {
                direction = hit.normal;
            }
            scattered = Ray(hit.point, normalize(direction));
            attenuation = textureColor;
            return true;
        }
        if (params.type == Type::Metal) {
            const Vec3 reflected = reflect(normalize(incoming.direction), hit.normal);
            scattered = Ray(
                hit.point,
                normalize(reflected + std::clamp(params.fuzz, 0.0, 1.0) * rng.inUnitSphere()));
            attenuation = textureColor;
            return dot(scattered.direction, hit.normal) > 0.0;
        }
        if (params.type == Type::Dielectric) {
            attenuation = Color(1.0, 1.0, 1.0);

            const double ior = std::clamp(params.ior, 1.01, 2.5);
            const double refractionRatio = hit.frontFace ? (1.0 / ior) : ior;
            const Vec3 unitDirection = normalize(incoming.direction);

            const double cosTheta = std::fmin(dot(-unitDirection, hit.normal), 1.0);
            const double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);

            const bool cannotRefract = refractionRatio * sinTheta > 1.0;
            Vec3 direction;

            if (cannotRefract || reflectance(cosTheta, refractionRatio) > rng.uniform()) {
                direction = reflect(unitDirection, hit.normal);
            } else {
                direction = refract(unitDirection, hit.normal, refractionRatio);
            }

            scattered = Ray(hit.point, normalize(direction));
            return true;
        }

        // Coated diffuse.
        const Vec3 unitIn = normalize(incoming.direction);
        const Vec3 reflected = reflect(unitIn, hit.normal);
        const Vec3 glossyDir = normalize(reflected + std::clamp(params.roughness, 0.0, 1.0) * rng.inUnitSphere());
        const Vec3 diffuseDir = normalize(hit.normal + rng.unitVector());

        if (rng.uniform() < std::clamp(params.coatStrength, 0.0, 1.0)) {
            scattered = Ray(hit.point, glossyDir);
            attenuation = Color(0.95, 0.95, 0.95);
        } else {
            scattered = Ray(hit.point, diffuseDir);
            attenuation = textureColor;
        }
        return true;
    }

    [[nodiscard]] Color emitted() const override {
        if (params.type == Type::Emissive) {
            return params.emission;
        }
        return Color(0.0, 0.0, 0.0);
    }

private:
    Params params;

    [[nodiscard]] Color sampleAlbedo(double u, double v) const {
        Color albedo = clampVec(params.albedo, 0.0, 1.0);
        if (params.albedoTexture && params.albedoTexture->valid()) {
            albedo *= params.albedoTexture->sample(u, v);
            albedo = clampVec(albedo, 0.0, 1.0);
        }
        return albedo;
    }

    [[nodiscard]] static double reflectance(double cosine, double refIdx) {
        double r0 = (1.0 - refIdx) / (1.0 + refIdx);
        r0 *= r0;
        return r0 + (1.0 - r0) * std::pow(1.0 - cosine, 5.0);
    }
};

}  // namespace orion

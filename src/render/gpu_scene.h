#pragma once

#include <cstdint>
#include <vector>

#include "math/vec3.h"

namespace orion {

enum class GpuMaterialType : int {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
    Emissive = 3,
    CoatedDiffuse = 4,
};

struct GpuVec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct GpuMaterialData {
    GpuMaterialType type = GpuMaterialType::Lambertian;
    GpuVec3 albedo{1.0f, 1.0f, 1.0f};
    float fuzz = 0.0f;
    float ior = 1.5f;
    float coatStrength = 0.0f;
    float roughness = 0.0f;
    GpuVec3 emission{0.0f, 0.0f, 0.0f};
    int albedoTextureIndex = -1;
};

struct GpuSphereData {
    GpuVec3 center;
    float radius = 1.0f;
    int materialIndex = 0;
};

struct GpuTriangleData {
    GpuVec3 a;
    GpuVec3 b;
    GpuVec3 c;
    GpuVec3 uvA;
    GpuVec3 uvB;
    GpuVec3 uvC;
    int hasUv = 0;
    int materialIndex = 0;
};

struct GpuTextureData {
    int width = 0;
    int height = 0;
    int offset = 0;
};

struct GpuBoxData {
    GpuVec3 minPoint;
    GpuVec3 maxPoint;
    int materialIndex = 0;
};

struct GpuCameraData {
    GpuVec3 origin;
    GpuVec3 lowerLeftCorner;
    GpuVec3 horizontal;
    GpuVec3 vertical;
    GpuVec3 u;
    GpuVec3 v;
    float lensRadius = 0.0f;
};

struct GpuSceneData {
    std::vector<GpuMaterialData> materials;
    std::vector<GpuSphereData> spheres;
    std::vector<GpuTriangleData> triangles;
    std::vector<GpuBoxData> boxes;
    std::vector<GpuTextureData> textures;
    std::vector<GpuVec3> textureTexels;
    GpuCameraData camera;
};

[[nodiscard]] inline GpuVec3 toGpuVec3(const Vec3& v) {
    return GpuVec3{
        static_cast<float>(v.x()),
        static_cast<float>(v.y()),
        static_cast<float>(v.z())};
}

[[nodiscard]] inline Vec3 fromGpuVec3(const GpuVec3& v) {
    return Vec3(v.x, v.y, v.z);
}

}  // namespace orion

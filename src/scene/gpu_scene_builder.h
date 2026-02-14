#pragma once

#include <cmath>
#include <string>
#include <unordered_map>

#include "core/random.h"
#include "render/camera.h"
#include "render/gpu_scene.h"
#include "render/renderer.h"
#include "scene/scene_builder.h"

namespace orion {

[[nodiscard]] inline GpuCameraData buildGpuCamera(const CameraSettings& settings) {
    // Mirror CPU camera construction exactly so CPU/GPU produce matching framing/DoF behavior.
    constexpr double pi = 3.14159265358979323846;

    const double theta = settings.verticalFovDeg * pi / 180.0;
    const double h = std::tan(theta / 2.0);
    const double viewportHeight = 2.0 * h;
    const double viewportWidth = settings.aspectRatio * viewportHeight;

    const Vec3 w = normalize(settings.lookFrom - settings.lookAt);
    const Vec3 u = normalize(cross(settings.vUp, w));
    const Vec3 v = cross(w, u);

    const Point3 origin = settings.lookFrom;
    const Vec3 horizontal = settings.focusDistance * viewportWidth * u;
    const Vec3 vertical = settings.focusDistance * viewportHeight * v;
    const Point3 lowerLeftCorner = origin - horizontal / 2.0 - vertical / 2.0 - settings.focusDistance * w;

    GpuCameraData camera;
    camera.origin = toGpuVec3(origin);
    camera.lowerLeftCorner = toGpuVec3(lowerLeftCorner);
    camera.horizontal = toGpuVec3(horizontal);
    camera.vertical = toGpuVec3(vertical);
    camera.u = toGpuVec3(u);
    camera.v = toGpuVec3(v);
    camera.lensRadius = static_cast<float>(settings.aperture * 0.5);
    return camera;
}

struct GpuWorldData {
    // Flat arrays consumed directly by CUDA kernels.
    std::vector<GpuMaterialData> materials;
    std::vector<GpuSphereData> spheres;
    std::vector<GpuTriangleData> triangles;
    std::vector<GpuTextureData> textures;
    std::vector<GpuVec3> textureTexels;
};

[[nodiscard]] inline GpuWorldData makeShowcaseGpuWorld(std::uint64_t seed) {
    RNG rng(seed);

    GpuWorldData world;
    world.materials.reserve(512);
    world.spheres.reserve(512);

    auto addMaterial = [&](const GpuMaterialData& material) -> int {
        world.materials.push_back(material);
        return static_cast<int>(world.materials.size() - 1);
    };

    auto addSphere = [&](const Point3& center, double radius, int materialIndex) {
        world.spheres.push_back(
            GpuSphereData{toGpuVec3(center), static_cast<float>(radius), materialIndex});
    };

    {
        GpuMaterialData mat;
        mat.type = GpuMaterialType::Lambertian;
        mat.albedo = GpuVec3{0.48f, 0.46f, 0.43f};
        const int material = addMaterial(mat);
        addSphere(Point3(0.0, -1000.0, 0.0), 1000.0, material);
    }

    {
        GpuMaterialData mat;
        mat.type = GpuMaterialType::Emissive;
        mat.emission = GpuVec3{8.0f, 6.5f, 4.5f};
        const int material = addMaterial(mat);
        addSphere(Point3(0.0, 7.5, 0.0), 1.8, material);
    }

    // A cool overhead fill light for stronger highlights on GPU path.
    {
        GpuMaterialData mat;
        mat.type = GpuMaterialType::Emissive;
        mat.emission = GpuVec3{3.0f, 4.2f, 6.0f};
        const int material = addMaterial(mat);
        addSphere(Point3(0.0, 6.5, -5.0), 1.2, material);
    }

    for (int a = -10; a < 10; ++a) {
        for (int b = -10; b < 10; ++b) {
            const double choose = rng.uniform();
            const Point3 center(
                static_cast<double>(a) + 0.9 * rng.uniform(),
                0.2,
                static_cast<double>(b) + 0.9 * rng.uniform());

            if ((center - Point3(4.0, 0.2, 0.0)).length() <= 0.9) {
                continue;
            }

            GpuMaterialData mat;
            if (choose < 0.6) {
                mat.type = GpuMaterialType::Lambertian;
                const Color albedo = rng.randomVec(0.1, 0.95) * rng.randomVec(0.1, 0.95);
                mat.albedo = toGpuVec3(albedo);
            } else if (choose < 0.78) {
                mat.type = GpuMaterialType::Metal;
                mat.albedo = toGpuVec3(rng.randomVec(0.55, 1.0));
                mat.fuzz = static_cast<float>(rng.range(0.0, 0.25));
            } else if (choose < 0.92) {
                mat.type = GpuMaterialType::CoatedDiffuse;
                mat.albedo = toGpuVec3(rng.randomVec(0.2, 0.9));
                mat.coatStrength = static_cast<float>(rng.range(0.2, 0.75));
                mat.roughness = static_cast<float>(rng.range(0.03, 0.35));
            } else {
                mat.type = GpuMaterialType::Dielectric;
                mat.ior = 1.5f;
            }

            const int material = addMaterial(mat);
            addSphere(center, 0.2, material);
        }
    }

    {
        GpuMaterialData mat;
        mat.type = GpuMaterialType::Dielectric;
        mat.ior = 1.5f;
        const int material = addMaterial(mat);
        addSphere(Point3(0.0, 1.0, 0.0), 1.0, material);
    }

    {
        GpuMaterialData mat;
        mat.type = GpuMaterialType::Lambertian;
        mat.albedo = GpuVec3{0.4f, 0.2f, 0.1f};
        const int material = addMaterial(mat);
        addSphere(Point3(-4.0, 1.0, 0.0), 1.0, material);
    }

    {
        GpuMaterialData mat;
        mat.type = GpuMaterialType::Metal;
        mat.albedo = GpuVec3{0.75f, 0.75f, 0.72f};
        mat.fuzz = 0.02f;
        const int material = addMaterial(mat);
        addSphere(Point3(4.0, 1.0, 0.0), 1.0, material);
    }

    return world;
}

[[nodiscard]] inline GpuSceneData makeShowcaseGpuScene(
    const RenderSettings& settings,
    std::uint64_t seed,
    const std::string& rawDemoName = "showcase") {
    GpuSceneData scene;
    GpuWorldData world = makeShowcaseGpuWorld(seed);
    scene.materials = std::move(world.materials);
    scene.spheres = std::move(world.spheres);
    scene.triangles = std::move(world.triangles);
    scene.camera = buildGpuCamera(makeShowcaseCameraSettingsForDemo(
        rawDemoName,
        static_cast<double>(settings.width) / static_cast<double>(settings.height)));
    return scene;
}

[[nodiscard]] inline GpuMaterialData makeSceneEditorGpuMaterial(const SceneEditorPrimitive& primitive) {
    GpuMaterialData mat;
    const Color color = clampVec(primitive.color, 0.0, 1.0);
    const std::string material = normalizeToken(primitive.material);
    const bool isLight = primitive.type == "light";

    if (isLight || material == "emissive" || material == "light") {
        mat.type = GpuMaterialType::Emissive;
        const float intensity = isLight ? 14.0f : 10.0f;
        mat.emission = GpuVec3{
            intensity * static_cast<float>(color.x()),
            intensity * static_cast<float>(color.y()),
            intensity * static_cast<float>(color.z())};
        return mat;
    }

    if (material == "metal") {
        mat.type = GpuMaterialType::Metal;
        mat.albedo = toGpuVec3(color);
        mat.fuzz = 0.04f;
        return mat;
    }

    if (material == "dielectric" || material == "glass") {
        mat.type = GpuMaterialType::Dielectric;
        mat.ior = 1.5f;
        return mat;
    }

    if (material == "coated" || material == "coated_diffuse") {
        mat.type = GpuMaterialType::CoatedDiffuse;
        mat.albedo = toGpuVec3(color);
        mat.coatStrength = 0.55f;
        mat.roughness = 0.08f;
        return mat;
    }

    mat.type = GpuMaterialType::Lambertian;
    mat.albedo = toGpuVec3(color);
    return mat;
}

[[nodiscard]] inline GpuMaterialData makeSceneEditorGpuMaterial(const SceneEditorMeshTriangle& tri) {
    GpuMaterialData mat;
    const std::string type = normalizeToken(tri.materialType);
    const Color color = clampVec(tri.color, 0.0, 1.0);

    if (type == "emissive" || type == "light") {
        mat.type = GpuMaterialType::Emissive;
        const Color emission = clampVec(tri.emission, 0.0, 120.0);
        mat.emission = toGpuVec3(emission);
        return mat;
    }
    if (type == "metal") {
        mat.type = GpuMaterialType::Metal;
        mat.albedo = toGpuVec3(color);
        mat.fuzz = static_cast<float>(std::clamp(tri.fuzz, 0.0, 1.0));
        return mat;
    }
    if (type == "dielectric" || type == "glass") {
        mat.type = GpuMaterialType::Dielectric;
        mat.ior = static_cast<float>(std::clamp(tri.ior, 1.01, 2.5));
        return mat;
    }
    if (type == "coated" || type == "coated_diffuse") {
        mat.type = GpuMaterialType::CoatedDiffuse;
        mat.albedo = toGpuVec3(color);
        mat.coatStrength = static_cast<float>(std::clamp(tri.coatStrength, 0.0, 1.0));
        mat.roughness = static_cast<float>(std::clamp(tri.roughness, 0.0, 1.0));
        return mat;
    }

    mat.type = GpuMaterialType::Lambertian;
    mat.albedo = toGpuVec3(color);
    return mat;
}

[[nodiscard]] inline GpuWorldData makeSceneEditorGpuWorld(
    std::uint64_t seed,
    const std::string& rawSceneSpec,
    const std::string& rawObjPath = "") {
    (void)seed;
    GpuWorldData world;
    world.materials.reserve(256);
    world.spheres.reserve(512);
    world.triangles.reserve(2048);
    world.textures.reserve(32);
    world.textureTexels.reserve(1 << 20);

    auto addMaterial = [&](const GpuMaterialData& material) -> int {
        world.materials.push_back(material);
        return static_cast<int>(world.materials.size() - 1);
    };

    auto addSphere = [&](const Vec3& center, double radius, int materialIndex) {
        world.spheres.push_back(GpuSphereData{
            toGpuVec3(Point3(center.x(), center.y(), center.z())),
            static_cast<float>(std::max(0.02, radius)),
            materialIndex});
    };

    auto addTriangle = [&](
        const Point3& a,
        const Point3& b,
        const Point3& c,
        int materialIndex,
        bool hasUv = false,
        const Vec3& uvA = Vec3(0.0, 0.0, 0.0),
        const Vec3& uvB = Vec3(0.0, 0.0, 0.0),
        const Vec3& uvC = Vec3(0.0, 0.0, 0.0)) {
        const Vec3 normal = cross(b - a, c - a);
        if (normal.lengthSquared() <= 1.0e-16) {
            return;
        }
        world.triangles.push_back(GpuTriangleData{
            toGpuVec3(a),
            toGpuVec3(b),
            toGpuVec3(c),
            toGpuVec3(uvA),
            toGpuVec3(uvB),
            toGpuVec3(uvC),
            hasUv ? 1 : 0,
            materialIndex});
    };

    std::unordered_map<const ImageTexture*, int> textureIndexCache;
    textureIndexCache.reserve(64);
    auto addTexture = [&](const std::shared_ptr<const ImageTexture>& texture) -> int {
        if (!texture || !texture->valid()) {
            return -1;
        }
        auto found = textureIndexCache.find(texture.get());
        if (found != textureIndexCache.end()) {
            // Reuse existing texture slot if multiple triangles reference the same image object.
            return found->second;
        }

        const int width = texture->width;
        const int height = texture->height;
        if (width <= 0 || height <= 0) {
            return -1;
        }
        const std::size_t pixelCount = static_cast<std::size_t>(width * height);
        if (texture->texels.size() != pixelCount) {
            return -1;
        }

        const int offset = static_cast<int>(world.textureTexels.size());
        world.textureTexels.reserve(world.textureTexels.size() + pixelCount);
        for (const Color& texel : texture->texels) {
            world.textureTexels.push_back(toGpuVec3(clampVec(texel, 0.0, 1.0)));
        }
        world.textures.push_back(GpuTextureData{width, height, offset});
        const int textureIndex = static_cast<int>(world.textures.size() - 1);
        textureIndexCache.emplace(texture.get(), textureIndex);
        return textureIndex;
    };

    const std::vector<SceneEditorPrimitive> primitives = parseSceneEditorSpec(rawSceneSpec);
    bool hasLight = false;

    // Convert editor primitives into explicit sphere/triangle lists for GPU traversal.
    for (const SceneEditorPrimitive& primitive : primitives) {
        const std::string type = normalizeToken(primitive.type);
        const int materialIndex = addMaterial(makeSceneEditorGpuMaterial(primitive));

        if (type == "sphere") {
            const double radius =
                std::max(0.05, (primitive.scale.x() + primitive.scale.y() + primitive.scale.z()) / 6.0);
            addSphere(primitive.position, radius, materialIndex);
            continue;
        }

        if (type == "cube") {
            const std::array<Point3, 8> base = {
                Point3(-0.5, -0.5, -0.5),
                Point3(-0.5, -0.5, 0.5),
                Point3(-0.5, 0.5, -0.5),
                Point3(-0.5, 0.5, 0.5),
                Point3(0.5, -0.5, -0.5),
                Point3(0.5, -0.5, 0.5),
                Point3(0.5, 0.5, -0.5),
                Point3(0.5, 0.5, 0.5),
            };
            std::array<Point3, 8> p{};
            for (std::size_t i = 0; i < base.size(); ++i) {
                p[i] = transformScenePoint(primitive, base[i]);
            }

            addTriangle(p[4], p[5], p[7], materialIndex);
            addTriangle(p[4], p[7], p[6], materialIndex);
            addTriangle(p[0], p[2], p[3], materialIndex);
            addTriangle(p[0], p[3], p[1], materialIndex);
            addTriangle(p[2], p[6], p[7], materialIndex);
            addTriangle(p[2], p[7], p[3], materialIndex);
            addTriangle(p[0], p[1], p[5], materialIndex);
            addTriangle(p[0], p[5], p[4], materialIndex);
            addTriangle(p[1], p[3], p[7], materialIndex);
            addTriangle(p[1], p[7], p[5], materialIndex);
            addTriangle(p[0], p[4], p[6], materialIndex);
            addTriangle(p[0], p[6], p[2], materialIndex);
            continue;
        }

        if (type == "plane") {
            const Point3 a = transformScenePoint(primitive, Point3(-0.5, 0.0, -0.5));
            const Point3 b = transformScenePoint(primitive, Point3(0.5, 0.0, -0.5));
            const Point3 c = transformScenePoint(primitive, Point3(0.5, 0.0, 0.5));
            const Point3 d = transformScenePoint(primitive, Point3(-0.5, 0.0, 0.5));
            addTriangle(a, b, c, materialIndex);
            addTriangle(a, c, d, materialIndex);
            continue;
        }

        if (type == "light") {
            hasLight = true;
            const double radius =
                std::max(0.08, (primitive.scale.x() + primitive.scale.y() + primitive.scale.z()) / 9.0);
            addSphere(primitive.position, radius, materialIndex);
        }
    }

    if (!trimAscii(rawObjPath).empty()) {
        const std::vector<SceneEditorMeshTriangle> meshTriangles = loadSceneEditorObjTriangles(rawObjPath);
        std::unordered_map<std::string, int> meshMaterialCache;
        meshMaterialCache.reserve(128);
        for (const SceneEditorMeshTriangle& tri : meshTriangles) {
            const std::string materialKey = sceneEditorMeshMaterialKey(tri);
            auto found = meshMaterialCache.find(materialKey);
            if (found == meshMaterialCache.end()) {
                // Deduplicate materially-identical triangles to keep material table compact.
                GpuMaterialData material = makeSceneEditorGpuMaterial(tri);
                material.albedoTextureIndex = addTexture(tri.albedoTexture);
                found = meshMaterialCache.emplace(materialKey, addMaterial(material)).first;
            }
            addTriangle(
                tri.a,
                tri.b,
                tri.c,
                found->second,
                tri.hasUv,
                tri.uvA,
                tri.uvB,
                tri.uvC);
        }
    }

    if (!hasLight) {
        // Keep default editor scene renderable even when user forgets to add lights.
        GpuMaterialData fallbackLight;
        fallbackLight.type = GpuMaterialType::Emissive;
        fallbackLight.emission = GpuVec3{14.0f, 14.0f, 14.0f};
        const int materialIndex = addMaterial(fallbackLight);
        addSphere(Vec3(2.2, 3.4, 1.3), 0.22, materialIndex);
    }

    return world;
}

[[nodiscard]] inline GpuSceneData makeSceneEditorGpuScene(
    const RenderSettings& settings,
    std::uint64_t seed,
    const std::string& sceneSpec,
    const std::string& objPath = "",
    const CameraOverride* cameraOverride = nullptr) {
    GpuSceneData scene;
    GpuWorldData world = makeSceneEditorGpuWorld(seed, sceneSpec, objPath);
    scene.materials = std::move(world.materials);
    scene.spheres = std::move(world.spheres);
    scene.triangles = std::move(world.triangles);
    scene.textures = std::move(world.textures);
    scene.textureTexels = std::move(world.textureTexels);

    CameraSettings cameraSettings = makeSceneEditorCameraSettings(
        static_cast<double>(settings.width) / static_cast<double>(settings.height));
    applyCameraOverride(cameraSettings, cameraOverride);
    scene.camera = buildGpuCamera(cameraSettings);
    return scene;
}

[[nodiscard]] inline GpuSceneData makeGpuSceneForDemo(
    const RenderSettings& settings,
    std::uint64_t seed,
    const std::string& rawDemoName,
    const std::string& rawSceneSpec = "",
    const std::string& rawObjPath = "",
    const CameraOverride* cameraOverride = nullptr) {
    const std::string demo = normalizeDemoName(rawDemoName);
    // Runtime demo routing currently favors scene_editor for CLI/frontend consistency.
    if (demo == "scene_editor") {
        return makeSceneEditorGpuScene(settings, seed, rawSceneSpec, rawObjPath, cameraOverride);
    }
    return makeShowcaseGpuScene(settings, seed, demo);
}

}  // namespace orion

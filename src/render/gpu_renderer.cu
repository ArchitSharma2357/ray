#include "render/gpu_renderer.h"

#if ORION_HAS_CUDA

#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace orion {
namespace {

// Minimal GPU-side math types kept intentionally POD-friendly for kernel use.
struct DeviceVec3 {
    float x;
    float y;
    float z;

    __host__ __device__ DeviceVec3() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ DeviceVec3(float xIn, float yIn, float zIn) : x(xIn), y(yIn), z(zIn) {}
};

struct DeviceRay {
    DeviceVec3 origin;
    DeviceVec3 direction;

    __host__ __device__ DeviceRay() = default;
    __host__ __device__ DeviceRay(const DeviceVec3& originIn, const DeviceVec3& directionIn)
        : origin(originIn), direction(directionIn) {}
};

struct HitRecord {
    float t;
    DeviceVec3 point;
    DeviceVec3 normal;
    bool frontFace;
    float u;
    float v;
    int materialIndex;
};

__host__ __device__ inline DeviceVec3 makeVec(const GpuVec3& v) {
    return DeviceVec3(v.x, v.y, v.z);
}

__host__ __device__ inline DeviceVec3 operator+(const DeviceVec3& a, const DeviceVec3& b) {
    return DeviceVec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline DeviceVec3 operator-(const DeviceVec3& a, const DeviceVec3& b) {
    return DeviceVec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline DeviceVec3 operator*(const DeviceVec3& a, const DeviceVec3& b) {
    return DeviceVec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline DeviceVec3 operator*(const DeviceVec3& a, float t) {
    return DeviceVec3(a.x * t, a.y * t, a.z * t);
}

__host__ __device__ inline DeviceVec3 operator*(float t, const DeviceVec3& a) {
    return a * t;
}

__host__ __device__ inline DeviceVec3 operator/(const DeviceVec3& a, float t) {
    return a * (1.0f / t);
}

__host__ __device__ inline DeviceVec3& operator+=(DeviceVec3& a, const DeviceVec3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ inline DeviceVec3& operator*=(DeviceVec3& a, const DeviceVec3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

__host__ __device__ inline DeviceVec3& operator/=(DeviceVec3& a, float t) {
    a = a / t;
    return a;
}

__host__ __device__ inline float dot(const DeviceVec3& a, const DeviceVec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline DeviceVec3 cross(const DeviceVec3& a, const DeviceVec3& b) {
    return DeviceVec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__host__ __device__ inline float lengthSquared(const DeviceVec3& v) {
    return dot(v, v);
}

__host__ __device__ inline float length(const DeviceVec3& v) {
    return sqrtf(lengthSquared(v));
}

__host__ __device__ inline DeviceVec3 normalize(const DeviceVec3& v) {
    return v / length(v);
}

__host__ __device__ inline DeviceVec3 reflect(const DeviceVec3& in, const DeviceVec3& normal) {
    return in - 2.0f * dot(in, normal) * normal;
}

__host__ __device__ inline DeviceVec3 refract(const DeviceVec3& uv, const DeviceVec3& normal, float etaiOverEtat) {
    const float cosTheta = fminf(dot(DeviceVec3(-uv.x, -uv.y, -uv.z), normal), 1.0f);
    const DeviceVec3 rOutPerp = etaiOverEtat * (uv + cosTheta * normal);
    const DeviceVec3 rOutParallel = -sqrtf(fabsf(1.0f - lengthSquared(rOutPerp))) * normal;
    return rOutPerp + rOutParallel;
}

__device__ inline std::uint32_t xorshift32(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ inline std::uint32_t mixSeed(std::uint64_t seed, int pixelIndex, int sampleIndex) {
    // Hash global seed + pixel + sample index into a robust per-thread RNG seed.
    std::uint64_t v = seed;
    v ^= static_cast<std::uint64_t>(pixelIndex) * 0x9E3779B185EBCA87ULL;
    v ^= static_cast<std::uint64_t>(sampleIndex + 1) * 0xC2B2AE3D27D4EB4FULL;
    v ^= v >> 33;
    v *= 0xff51afd7ed558ccdULL;
    v ^= v >> 33;
    v *= 0xc4ceb9fe1a85ec53ULL;
    v ^= v >> 33;
    const std::uint32_t mixed = static_cast<std::uint32_t>(v & 0xffffffffu);
    return mixed == 0u ? 1u : mixed;
}

__device__ inline float random01(std::uint32_t& state) {
    const std::uint32_t bits = xorshift32(state);
    return static_cast<float>(bits & 0x00ffffffu) * (1.0f / 16777216.0f);
}

__device__ inline float randomRange(std::uint32_t& state, float minVal, float maxVal) {
    return minVal + (maxVal - minVal) * random01(state);
}

__device__ DeviceVec3 randomInUnitSphere(std::uint32_t& state) {
    while (true) {
        const DeviceVec3 p(
            randomRange(state, -1.0f, 1.0f),
            randomRange(state, -1.0f, 1.0f),
            randomRange(state, -1.0f, 1.0f));
        if (lengthSquared(p) < 1.0f) {
            return p;
        }
    }
}

__device__ DeviceVec3 randomUnitVector(std::uint32_t& state) {
    return normalize(randomInUnitSphere(state));
}

__device__ DeviceVec3 randomInUnitDisk(std::uint32_t& state) {
    while (true) {
        const DeviceVec3 p(
            randomRange(state, -1.0f, 1.0f),
            randomRange(state, -1.0f, 1.0f),
            0.0f);
        if (lengthSquared(p) < 1.0f) {
            return p;
        }
    }
}

__device__ inline DeviceVec3 skyColor(const DeviceVec3& direction) {
    const DeviceVec3 unit = normalize(direction);
    const float t = 0.5f * (unit.y + 1.0f);
    const DeviceVec3 gradient = (1.0f - t) * DeviceVec3(1.0f, 1.0f, 1.0f) + t * DeviceVec3(0.45f, 0.67f, 1.0f);

    const DeviceVec3 sunDir = normalize(DeviceVec3(0.3f, 0.8f, 0.2f));
    const float sun = powf(fmaxf(dot(unit, sunDir), 0.0f), 384.0f);
    return gradient + sun * DeviceVec3(8.0f, 6.5f, 5.0f);
}

__device__ inline DeviceVec3 sampleTextureColor(
    int textureIndex,
    float u,
    float v,
    const GpuTextureData* textures,
    int textureCount,
    const GpuVec3* textureTexels) {
    if (textureIndex < 0 || textureIndex >= textureCount || textures == nullptr || textureTexels == nullptr) {
        return DeviceVec3(1.0f, 1.0f, 1.0f);
    }

    const GpuTextureData tex = textures[textureIndex];
    if (tex.width <= 0 || tex.height <= 0 || tex.offset < 0) {
        return DeviceVec3(1.0f, 1.0f, 1.0f);
    }

    float uu = u - floorf(u);
    if (uu < 0.0f) {
        uu += 1.0f;
    }
    float vv = v - floorf(v);
    if (vv < 0.0f) {
        vv += 1.0f;
    }

    const int x = max(0, min(tex.width - 1, static_cast<int>(uu * static_cast<float>(tex.width - 1) + 0.5f)));
    const int y = max(0, min(tex.height - 1, static_cast<int>((1.0f - vv) * static_cast<float>(tex.height - 1) + 0.5f)));
    const int texelIndex = tex.offset + y * tex.width + x;
    return makeVec(textureTexels[texelIndex]);
}

__device__ inline bool hitSphere(
    const GpuSphereData& sphere,
    const DeviceRay& ray,
    float tMin,
    float tMax,
    HitRecord& record) {
    const DeviceVec3 center = makeVec(sphere.center);
    const DeviceVec3 oc = ray.origin - center;
    const float a = lengthSquared(ray.direction);
    const float halfB = dot(oc, ray.direction);
    const float c = lengthSquared(oc) - sphere.radius * sphere.radius;

    const float discriminant = halfB * halfB - a * c;
    if (discriminant < 0.0f) {
        return false;
    }

    const float sqrtd = sqrtf(discriminant);
    float root = (-halfB - sqrtd) / a;
    if (root < tMin || root > tMax) {
        root = (-halfB + sqrtd) / a;
        if (root < tMin || root > tMax) {
            return false;
        }
    }

    record.t = root;
    record.point = ray.origin + root * ray.direction;
    const DeviceVec3 outward = (record.point - center) / sphere.radius;
    record.frontFace = dot(ray.direction, outward) < 0.0f;
    record.normal = record.frontFace ? outward : DeviceVec3(-outward.x, -outward.y, -outward.z);
    constexpr float pi = 3.14159265358979323846f;
    const float theta = acosf(fmaxf(-1.0f, fminf(1.0f, -outward.y)));
    const float phi = atan2f(-outward.z, outward.x) + pi;
    record.u = phi / (2.0f * pi);
    record.v = theta / pi;
    record.materialIndex = sphere.materialIndex;
    return true;
}

__device__ inline bool hitTriangle(
    const GpuTriangleData& tri,
    const DeviceRay& ray,
    float tMin,
    float tMax,
    HitRecord& record) {
    const DeviceVec3 v0 = makeVec(tri.a);
    const DeviceVec3 v1 = makeVec(tri.b);
    const DeviceVec3 v2 = makeVec(tri.c);

    const DeviceVec3 edge1 = v1 - v0;
    const DeviceVec3 edge2 = v2 - v0;
    const DeviceVec3 pVec = cross(ray.direction, edge2);
    const float det = dot(edge1, pVec);
    if (fabsf(det) <= 1.0e-8f) {
        return false;
    }

    const float invDet = 1.0f / det;
    const DeviceVec3 tVec = ray.origin - v0;
    const float u = dot(tVec, pVec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    const DeviceVec3 qVec = cross(tVec, edge1);
    const float v = dot(ray.direction, qVec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) {
        return false;
    }

    const float t = dot(edge2, qVec) * invDet;
    if (t < tMin || t > tMax) {
        return false;
    }

    const DeviceVec3 outward = normalize(cross(edge1, edge2));
    record.t = t;
    record.point = ray.origin + t * ray.direction;
    record.frontFace = dot(ray.direction, outward) < 0.0f;
    record.normal = record.frontFace ? outward : DeviceVec3(-outward.x, -outward.y, -outward.z);
    if (tri.hasUv != 0) {
        const DeviceVec3 uvA = makeVec(tri.uvA);
        const DeviceVec3 uvB = makeVec(tri.uvB);
        const DeviceVec3 uvC = makeVec(tri.uvC);
        const float w = 1.0f - u - v;
        const DeviceVec3 uv = w * uvA + u * uvB + v * uvC;
        record.u = uv.x;
        record.v = uv.y;
    } else {
        record.u = 0.0f;
        record.v = 0.0f;
    }
    record.materialIndex = tri.materialIndex;
    return true;
}

__device__ inline float schlickReflectance(float cosine, float refIdx) {
    float r0 = (1.0f - refIdx) / (1.0f + refIdx);
    r0 *= r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

__global__ void accumulateSampleKernel(
    GpuVec3* accumulation,
    int width,
    int height,
    int sampleIndex,
    int maxDepth,
    std::uint64_t seed,
    GpuCameraData camera,
    const GpuSphereData* spheres,
    int sphereCount,
    const GpuTriangleData* triangles,
    int triangleCount,
    const GpuMaterialData* materials,
    const GpuTextureData* textures,
    int textureCount,
    const GpuVec3* textureTexels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int pixelIndex = y * width + x;
    std::uint32_t rng = mixSeed(seed, pixelIndex, sampleIndex);

    const float u = (static_cast<float>(x) + random01(rng)) / static_cast<float>(width - 1);
    const float v = (static_cast<float>(height - 1 - y) + random01(rng)) / static_cast<float>(height - 1);

    const DeviceVec3 camOrigin = makeVec(camera.origin);
    const DeviceVec3 camLowerLeft = makeVec(camera.lowerLeftCorner);
    const DeviceVec3 camHorizontal = makeVec(camera.horizontal);
    const DeviceVec3 camVertical = makeVec(camera.vertical);
    const DeviceVec3 camU = makeVec(camera.u);
    const DeviceVec3 camV = makeVec(camera.v);

    const DeviceVec3 disk = camera.lensRadius * randomInUnitDisk(rng);
    const DeviceVec3 lensOffset = camU * disk.x + camV * disk.y;

    DeviceRay ray(
        camOrigin + lensOffset,
        camLowerLeft + u * camHorizontal + v * camVertical - camOrigin - lensOffset);

    DeviceVec3 throughput(1.0f, 1.0f, 1.0f);
    DeviceVec3 radiance(0.0f, 0.0f, 0.0f);

    for (int bounce = 0; bounce < maxDepth; ++bounce) {
        bool hitAnything = false;
        HitRecord closestHit{};
        float closestT = 1.0e30f;

        // NOTE: current GPU path uses linear primitive traversal (no BVH traversal yet).
        for (int i = 0; i < sphereCount; ++i) {
            HitRecord current{};
            if (hitSphere(spheres[i], ray, 0.001f, closestT, current)) {
                hitAnything = true;
                closestT = current.t;
                closestHit = current;
            }
        }

        for (int i = 0; i < triangleCount; ++i) {
            HitRecord current{};
            if (hitTriangle(triangles[i], ray, 0.001f, closestT, current)) {
                hitAnything = true;
                closestT = current.t;
                closestHit = current;
            }
        }

        if (!hitAnything) {
            radiance += throughput * skyColor(ray.direction);
            break;
        }

        const GpuMaterialData material = materials[closestHit.materialIndex];
        DeviceVec3 albedo = makeVec(material.albedo);
        if (material.albedoTextureIndex >= 0) {
            albedo *= sampleTextureColor(
                material.albedoTextureIndex,
                closestHit.u,
                closestHit.v,
                textures,
                textureCount,
                textureTexels);
        }
        const DeviceVec3 emission = makeVec(material.emission);
        radiance += throughput * emission;

        if (material.type == GpuMaterialType::Emissive) {
            break;
        }

        DeviceRay scattered;
        DeviceVec3 attenuation(1.0f, 1.0f, 1.0f);
        bool didScatter = true;

        if (material.type == GpuMaterialType::Lambertian) {
            DeviceVec3 scatterDirection = closestHit.normal + randomUnitVector(rng);
            if (lengthSquared(scatterDirection) < 1.0e-10f) {
                scatterDirection = closestHit.normal;
            }
            scattered = DeviceRay(closestHit.point, normalize(scatterDirection));
            attenuation = albedo;
        } else if (material.type == GpuMaterialType::Metal) {
            const DeviceVec3 reflected = reflect(normalize(ray.direction), closestHit.normal);
            const DeviceVec3 direction = normalize(reflected + material.fuzz * randomInUnitSphere(rng));
            scattered = DeviceRay(closestHit.point, direction);
            attenuation = albedo;
            didScatter = dot(scattered.direction, closestHit.normal) > 0.0f;
        } else if (material.type == GpuMaterialType::Dielectric) {
            attenuation = DeviceVec3(1.0f, 1.0f, 1.0f);
            const float refractionRatio = closestHit.frontFace ? (1.0f / material.ior) : material.ior;
            const DeviceVec3 unitDirection = normalize(ray.direction);
            const float cosTheta = fminf(dot(DeviceVec3(-unitDirection.x, -unitDirection.y, -unitDirection.z), closestHit.normal), 1.0f);
            const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

            DeviceVec3 direction;
            const bool cannotRefract = refractionRatio * sinTheta > 1.0f;
            if (cannotRefract || schlickReflectance(cosTheta, refractionRatio) > random01(rng)) {
                direction = reflect(unitDirection, closestHit.normal);
            } else {
                direction = refract(unitDirection, closestHit.normal, refractionRatio);
            }
            scattered = DeviceRay(closestHit.point, normalize(direction));
        } else if (material.type == GpuMaterialType::CoatedDiffuse) {
            const DeviceVec3 unitIn = normalize(ray.direction);
            const DeviceVec3 reflected = reflect(unitIn, closestHit.normal);
            const DeviceVec3 glossy = normalize(reflected + material.roughness * randomInUnitSphere(rng));
            const DeviceVec3 diffuse = normalize(closestHit.normal + randomUnitVector(rng));

            if (random01(rng) < material.coatStrength) {
                scattered = DeviceRay(closestHit.point, glossy);
                attenuation = DeviceVec3(0.95f, 0.95f, 0.95f);
            } else {
                scattered = DeviceRay(closestHit.point, diffuse);
                attenuation = albedo;
            }
        } else {
            didScatter = false;
        }

        if (!didScatter) {
            break;
        }

        throughput *= attenuation;

        if (bounce > 3) {
            // Russian roulette keeps deep paths affordable on GPU while staying unbiased.
            const float maxChannel = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            const float survive = fminf(0.95f, fmaxf(0.05f, maxChannel));
            if (random01(rng) > survive) {
                break;
            }
            throughput /= survive;
        }

        ray = DeviceRay(closestHit.point + 0.0005f * closestHit.normal, scattered.direction);
    }

    GpuVec3 prev = accumulation[pixelIndex];
    prev.x += radiance.x;
    prev.y += radiance.y;
    prev.z += radiance.z;
    accumulation[pixelIndex] = prev;
}

inline bool checkCuda(cudaError_t err, std::string& errorMessage, const char* stage) {
    if (err == cudaSuccess) {
        return true;
    }
    errorMessage = std::string(stage) + ": " + cudaGetErrorString(err);
    return false;
}

}  // namespace

bool renderWithGpu(
    const GpuSceneData& scene,
    const RenderSettings& settings,
    Image& output,
    const std::function<void(int, int)>& onProgress,
    std::string& errorMessage) {
    if ((scene.spheres.empty() && scene.triangles.empty()) || scene.materials.empty()) {
        errorMessage = "GPU scene is empty.";
        return false;
    }

    if (settings.width < 2 || settings.height < 2 || settings.samplesPerPixel < 1 || settings.maxDepth < 1) {
        errorMessage = "Invalid render settings for GPU renderer.";
        return false;
    }

    GpuVec3* dAccum = nullptr;
    GpuSphereData* dSpheres = nullptr;
    GpuTriangleData* dTriangles = nullptr;
    GpuMaterialData* dMaterials = nullptr;
    GpuTextureData* dTextures = nullptr;
    GpuVec3* dTextureTexels = nullptr;

    const std::size_t pixelCount = static_cast<std::size_t>(settings.width) * static_cast<std::size_t>(settings.height);
    const std::size_t accumBytes = pixelCount * sizeof(GpuVec3);
    const std::size_t sphereBytes = scene.spheres.size() * sizeof(GpuSphereData);
    const std::size_t triangleBytes = scene.triangles.size() * sizeof(GpuTriangleData);
    const std::size_t materialBytes = scene.materials.size() * sizeof(GpuMaterialData);
    const std::size_t textureBytes = scene.textures.size() * sizeof(GpuTextureData);
    const std::size_t textureTexelBytes = scene.textureTexels.size() * sizeof(GpuVec3);

    // Keep cleanup centralized to guarantee device memory release on every early-return path.
    auto cleanup = [&]() {
        if (dAccum != nullptr) {
            cudaFree(dAccum);
            dAccum = nullptr;
        }
        if (dSpheres != nullptr) {
            cudaFree(dSpheres);
            dSpheres = nullptr;
        }
        if (dTriangles != nullptr) {
            cudaFree(dTriangles);
            dTriangles = nullptr;
        }
        if (dMaterials != nullptr) {
            cudaFree(dMaterials);
            dMaterials = nullptr;
        }
        if (dTextures != nullptr) {
            cudaFree(dTextures);
            dTextures = nullptr;
        }
        if (dTextureTexels != nullptr) {
            cudaFree(dTextureTexels);
            dTextureTexels = nullptr;
        }
    };

    if (!checkCuda(cudaMalloc(&dAccum, accumBytes), errorMessage, "cudaMalloc accumulation")) {
        cleanup();
        return false;
    }

    if (sphereBytes > 0) {
        if (!checkCuda(cudaMalloc(&dSpheres, sphereBytes), errorMessage, "cudaMalloc spheres")) {
            cleanup();
            return false;
        }
    }

    if (triangleBytes > 0) {
        if (!checkCuda(cudaMalloc(&dTriangles, triangleBytes), errorMessage, "cudaMalloc triangles")) {
            cleanup();
            return false;
        }
    }

    if (!checkCuda(cudaMalloc(&dMaterials, materialBytes), errorMessage, "cudaMalloc materials")) {
        cleanup();
        return false;
    }

    if (textureBytes > 0) {
        if (!checkCuda(cudaMalloc(&dTextures, textureBytes), errorMessage, "cudaMalloc textures")) {
            cleanup();
            return false;
        }
    }
    if (textureTexelBytes > 0) {
        if (!checkCuda(cudaMalloc(&dTextureTexels, textureTexelBytes), errorMessage, "cudaMalloc texture texels")) {
            cleanup();
            return false;
        }
    }

    if (!checkCuda(cudaMemset(dAccum, 0, accumBytes), errorMessage, "cudaMemset accumulation")) {
        cleanup();
        return false;
    }

    if (sphereBytes > 0) {
        if (!checkCuda(
                cudaMemcpy(dSpheres, scene.spheres.data(), sphereBytes, cudaMemcpyHostToDevice),
                errorMessage,
                "cudaMemcpy spheres H2D")) {
            cleanup();
            return false;
        }
    }

    if (triangleBytes > 0) {
        if (!checkCuda(
                cudaMemcpy(dTriangles, scene.triangles.data(), triangleBytes, cudaMemcpyHostToDevice),
                errorMessage,
                "cudaMemcpy triangles H2D")) {
            cleanup();
            return false;
        }
    }

    if (!checkCuda(
            cudaMemcpy(dMaterials, scene.materials.data(), materialBytes, cudaMemcpyHostToDevice),
            errorMessage,
            "cudaMemcpy materials H2D")) {
        cleanup();
        return false;
    }

    if (textureBytes > 0) {
        if (!checkCuda(
                cudaMemcpy(dTextures, scene.textures.data(), textureBytes, cudaMemcpyHostToDevice),
                errorMessage,
                "cudaMemcpy textures H2D")) {
            cleanup();
            return false;
        }
    }
    if (textureTexelBytes > 0) {
        if (!checkCuda(
                cudaMemcpy(dTextureTexels, scene.textureTexels.data(), textureTexelBytes, cudaMemcpyHostToDevice),
                errorMessage,
                "cudaMemcpy texture texels H2D")) {
            cleanup();
            return false;
        }
    }

    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned int>((settings.width + block.x - 1) / block.x),
        static_cast<unsigned int>((settings.height + block.y - 1) / block.y));

    // Launch one accumulation pass per sample. Accumulation remains in device memory until finished.
    for (int sample = 0; sample < settings.samplesPerPixel; ++sample) {
        accumulateSampleKernel<<<grid, block>>>(
            dAccum,
            settings.width,
            settings.height,
            sample,
            settings.maxDepth,
            settings.seed,
            scene.camera,
            dSpheres,
            static_cast<int>(scene.spheres.size()),
            dTriangles,
            static_cast<int>(scene.triangles.size()),
            dMaterials,
            dTextures,
            static_cast<int>(scene.textures.size()),
            dTextureTexels);

        if (!checkCuda(cudaGetLastError(), errorMessage, "kernel launch")) {
            cleanup();
            return false;
        }

        if (!checkCuda(cudaDeviceSynchronize(), errorMessage, "kernel synchronize")) {
            cleanup();
            return false;
        }

        if (onProgress) {
            onProgress(sample + 1, settings.samplesPerPixel);
        }
    }

    std::vector<GpuVec3> hostAccum(pixelCount);
    if (!checkCuda(
            cudaMemcpy(hostAccum.data(), dAccum, accumBytes, cudaMemcpyDeviceToHost),
            errorMessage,
            "cudaMemcpy accumulation D2H")) {
        cleanup();
        return false;
    }

    cleanup();

    output = Image(settings.width, settings.height);
    for (int y = 0; y < settings.height; ++y) {
        for (int x = 0; x < settings.width; ++x) {
            const std::size_t idx = static_cast<std::size_t>(y) * static_cast<std::size_t>(settings.width) +
                                    static_cast<std::size_t>(x);
            const GpuVec3 pixel = hostAccum[idx];
            output.setPixel(x, y, Color(pixel.x, pixel.y, pixel.z));
        }
    }

    return true;
}

}  // namespace orion

#endif  // ORION_HAS_CUDA

#pragma once

#include <functional>
#include <string>

#include "render/gpu_scene.h"
#include "render/image.h"
#include "render/renderer.h"

namespace orion {

#if !defined(ORION_HAS_CUDA)
#define ORION_HAS_CUDA 0
#endif

[[nodiscard]] inline bool gpuBackendCompiled() {
    return ORION_HAS_CUDA == 1;
}

[[nodiscard]] inline std::string gpuBackendLabel() {
#if ORION_HAS_CUDA
    return "cuda";
#else
    return "cpu-only build";
#endif
}

bool renderWithGpu(
    const GpuSceneData& scene,
    const RenderSettings& settings,
    Image& output,
    const std::function<void(int, int)>& onProgress,
    std::string& errorMessage);

#if !ORION_HAS_CUDA
inline bool renderWithGpu(
    const GpuSceneData&,
    const RenderSettings&,
    Image&,
    const std::function<void(int, int)>&,
    std::string& errorMessage) {
    errorMessage = "GPU backend is unavailable in this build. Install CUDA toolkit and rebuild.";
    return false;
}
#endif

}  // namespace orion

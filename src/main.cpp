#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "render/gpu_renderer.h"
#include "render/renderer.h"
#include "scene/gpu_scene_builder.h"
#include "scene/scene_builder.h"

namespace {

// User-facing backend selection mode.
enum class BackendChoice {
    Auto,
    Cpu,
    Gpu,
};

void printUsage() {
    const auto demos = orion::supportedDemoNames();
    std::ostringstream demoList;
    for (std::size_t i = 0; i < demos.size(); ++i) {
        if (i > 0) {
            demoList << " | ";
        }
        demoList << demos[i];
    }

    std::cout
        << "Raycism Engine\n"
        << "Usage:\n"
        << "  orion_raytracer [options]\n\n"
        << "Options:\n"
        << "  --width <int>       Output width (default: 1280)\n"
        << "  --height <int>      Output height (default: 720)\n"
        << "  --samples <int>     Samples per pixel (default: 300)\n"
        << "  --depth <int>       Max path depth (default: 16)\n"
        << "  --threads <int>     Worker threads (default: hardware concurrency)\n"
        << "  --seed <int>        Random seed (default: 1337)\n"
        << "  --exposure <float>  Exposure multiplier (default: 1.0)\n"
        << "  --backend <mode>    Rendering backend: auto | cpu | gpu (default: auto)\n"
        << "  --demo <name>       Demo scene: " << demoList.str()
        << " (default: " << orion::kDefaultDemoName << ")\n"
        << "  --scene-spec <str>  Scene editor primitive spec (used with --demo scene_editor)\n"
        << "  --obj-path <path>   Optional OBJ mesh file to import into scene_editor\n"
        << "  --camera-pos <x,y,z> Camera position override (scene_editor only)\n"
        << "  --camera-yaw <deg>  Camera yaw in degrees (scene_editor only)\n"
        << "  --camera-pitch <deg> Camera pitch in degrees (scene_editor only)\n"
        << "  --camera-fov <deg>  Camera vertical FOV in degrees (scene_editor only)\n"
        << "  --output <path>     Output PPM path (default: out/scene_editor.ppm)\n"
        << "  --quiet             Minimize console output (useful for live mode)\n"
        << "  --help              Show this message\n";
}

bool readIntArg(const std::string& value, int& out) {
    try {
        std::size_t idx = 0;
        const int parsed = std::stoi(value, &idx);
        if (idx != value.size()) {
            return false;
        }
        out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool readUInt64Arg(const std::string& value, std::uint64_t& out) {
    try {
        std::size_t idx = 0;
        const std::uint64_t parsed = std::stoull(value, &idx);
        if (idx != value.size()) {
            return false;
        }
        out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool readDoubleArg(const std::string& value, double& out) {
    try {
        std::size_t idx = 0;
        const double parsed = std::stod(value, &idx);
        if (idx != value.size()) {
            return false;
        }
        out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool readVec3CsvArg(const std::string& value, orion::Vec3& out) {
    const std::size_t first = value.find(',');
    if (first == std::string::npos) {
        return false;
    }
    const std::size_t second = value.find(',', first + 1);
    if (second == std::string::npos || value.find(',', second + 1) != std::string::npos) {
        return false;
    }

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!readDoubleArg(value.substr(0, first), x)
        || !readDoubleArg(value.substr(first + 1, second - first - 1), y)
        || !readDoubleArg(value.substr(second + 1), z)) {
        return false;
    }

    out = orion::Vec3(x, y, z);
    return true;
}

bool readBackendArg(const std::string& value, BackendChoice& out) {
    if (value == "auto") {
        out = BackendChoice::Auto;
        return true;
    }
    if (value == "cpu") {
        out = BackendChoice::Cpu;
        return true;
    }
    if (value == "gpu") {
        out = BackendChoice::Gpu;
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace orion;

    RenderSettings settings;
    BackendChoice backendChoice = BackendChoice::Auto;
    std::string outputPath = "out/scene_editor.ppm";
    std::string demoName = orion::kDefaultDemoName;
    std::string sceneSpec;
    std::string objPath;
    CameraOverride cameraOverride;
    bool cameraOverrideProvided = false;
    bool quiet = false;

    // Parse all CLI options in a single pass and fail fast on malformed input.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto needsValue = [&](const std::string& flag) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                return false;
            }
            return true;
        };

        if (arg == "--help") {
            printUsage();
            return 0;
        }
        if (arg == "--quiet") {
            quiet = true;
            continue;
        }
        if (arg == "--width") {
            if (!needsValue(arg) || !readIntArg(argv[++i], settings.width) || settings.width < 16) {
                std::cerr << "Invalid --width value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--height") {
            if (!needsValue(arg) || !readIntArg(argv[++i], settings.height) || settings.height < 16) {
                std::cerr << "Invalid --height value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--samples") {
            if (!needsValue(arg) || !readIntArg(argv[++i], settings.samplesPerPixel) || settings.samplesPerPixel < 1) {
                std::cerr << "Invalid --samples value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--depth") {
            if (!needsValue(arg) || !readIntArg(argv[++i], settings.maxDepth) || settings.maxDepth < 1) {
                std::cerr << "Invalid --depth value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--threads") {
            if (!needsValue(arg) || !readIntArg(argv[++i], settings.threadCount) || settings.threadCount < 1) {
                std::cerr << "Invalid --threads value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--seed") {
            if (!needsValue(arg) || !readUInt64Arg(argv[++i], settings.seed)) {
                std::cerr << "Invalid --seed value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--exposure") {
            if (!needsValue(arg) || !readDoubleArg(argv[++i], settings.exposure) || settings.exposure <= 0.0) {
                std::cerr << "Invalid --exposure value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--backend") {
            if (!needsValue(arg) || !readBackendArg(argv[++i], backendChoice)) {
                std::cerr << "Invalid --backend value. Use: auto | cpu | gpu\n";
                return 1;
            }
            continue;
        }
        if (arg == "--output") {
            if (!needsValue(arg)) {
                return 1;
            }
            outputPath = argv[++i];
            continue;
        }
        if (arg == "--demo") {
            if (!needsValue(arg)) {
                return 1;
            }
            demoName = orion::normalizeDemoName(argv[++i]);
            if (!orion::isSupportedDemoName(demoName)) {
                std::cerr << "Invalid --demo value: " << demoName << "\n";
                printUsage();
                return 1;
            }
            continue;
        }
        if (arg == "--scene-spec") {
            if (!needsValue(arg)) {
                return 1;
            }
            sceneSpec = argv[++i];
            continue;
        }
        if (arg == "--obj-path") {
            if (!needsValue(arg)) {
                return 1;
            }
            objPath = argv[++i];
            continue;
        }
        if (arg == "--camera-pos") {
            if (!needsValue(arg) || !readVec3CsvArg(argv[++i], cameraOverride.position)) {
                std::cerr << "Invalid --camera-pos value. Use x,y,z\n";
                return 1;
            }
            cameraOverrideProvided = true;
            continue;
        }
        if (arg == "--camera-yaw") {
            if (!needsValue(arg) || !readDoubleArg(argv[++i], cameraOverride.yawDeg)) {
                std::cerr << "Invalid --camera-yaw value\n";
                return 1;
            }
            cameraOverrideProvided = true;
            continue;
        }
        if (arg == "--camera-pitch") {
            if (!needsValue(arg) || !readDoubleArg(argv[++i], cameraOverride.pitchDeg)) {
                std::cerr << "Invalid --camera-pitch value\n";
                return 1;
            }
            cameraOverrideProvided = true;
            continue;
        }
        if (arg == "--camera-fov") {
            if (!needsValue(arg) || !readDoubleArg(argv[++i], cameraOverride.fovDeg)) {
                std::cerr << "Invalid --camera-fov value\n";
                return 1;
            }
            cameraOverrideProvided = true;
            continue;
        }

        std::cerr << "Unknown option: " << arg << "\n";
        printUsage();
        return 1;
    }

    try {
        cameraOverride.enabled = cameraOverrideProvided;
        const bool canUseGpu = gpuBackendCompiled();
        bool useGpu = false;
        // Resolve runtime backend policy from explicit user request + build capabilities.
        if (backendChoice == BackendChoice::Gpu) {
            useGpu = true;
        } else if (backendChoice == BackendChoice::Auto) {
            useGpu = canUseGpu;
        }

        if (useGpu && !canUseGpu) {
            std::cerr << "GPU backend requested, but this binary was built without CUDA support.\n";
            std::cerr << "Rebuild with CUDA toolkit installed, or run with --backend cpu.\n";
            return 1;
        }

        const std::filesystem::path outPath(outputPath);
        if (outPath.has_parent_path()) {
            std::filesystem::create_directories(outPath.parent_path());
        }

        if (!quiet) {
            std::cout << "Rendering " << settings.width << "x" << settings.height
                      << " | spp=" << settings.samplesPerPixel
                      << " | depth=" << settings.maxDepth
                      << " | threads=" << settings.threadCount
                      << " | backend=" << (useGpu ? "gpu" : "cpu")
                      << " | demo=" << demoName
                      << "\n";
            if (backendChoice == BackendChoice::Auto) {
                std::cout << "Backend auto-detect: " << gpuBackendLabel() << "\n";
            }
        }

        if (useGpu && !isGpuCompatibleDemoName(demoName)) {
            if (backendChoice == BackendChoice::Gpu) {
                std::cerr << "GPU backend requested, but editable environment mode is CPU-only.\n";
                std::cerr << "Run with --backend cpu.\n";
                return 1;
            }
            if (!quiet) {
                std::cout << "Demo '" << demoName << "' is CPU-only, switching to CPU backend.\n";
            }
            useGpu = false;
        }

        const auto start = std::chrono::steady_clock::now();
        int lastPercent = -1;
        const auto onProgress = [&](int done, int total) {
            if (quiet) {
                return;
            }
            const int percent = static_cast<int>((100.0 * done) / total);
            if (percent != lastPercent && percent % 5 == 0) {
                lastPercent = percent;
                std::cout << "Progress: " << percent << "%\n";
            }
        };

        Image image(settings.width, settings.height);
        if (useGpu) {
            std::string gpuError;
            const GpuSceneData gpuScene = makeGpuSceneForDemo(
                settings,
                settings.seed,
                demoName,
                sceneSpec,
                objPath,
                cameraOverride.enabled ? &cameraOverride : nullptr);
            const bool rendered = renderWithGpu(
                gpuScene,
                settings,
                image,
                onProgress,
                gpuError);

            // In auto mode, transparently fall back to CPU if GPU execution fails.
            if (!rendered) {
                if (backendChoice == BackendChoice::Gpu) {
                    std::cerr << "GPU render failed: " << gpuError << "\n";
                    return 1;
                }

                if (!quiet) {
                    std::cerr << "GPU render unavailable (" << gpuError << "), falling back to CPU.\n";
                }
                useGpu = false;
            }
        }

        if (!useGpu) {
            const ScenePackage scene = makeDemoScene(
                settings,
                settings.seed,
                demoName,
                sceneSpec,
                objPath,
                cameraOverride.enabled ? &cameraOverride : nullptr);
            Renderer renderer(settings);
            image = renderer.render(*scene.world, scene.camera, onProgress);
        }

        image.writePPM(outputPath, settings.samplesPerPixel, settings.exposure);

        const auto end = std::chrono::steady_clock::now();
        const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!quiet) {
            std::cout << "Render complete in " << elapsedMs << " ms\n";
            std::cout << "Wrote image to: " << outputPath << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

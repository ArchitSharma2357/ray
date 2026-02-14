#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <GLFW/glfw3.h>

#include "render/gpu_renderer.h"
#include "render/renderer.h"
#include "scene/gpu_scene_builder.h"
#include "scene/scene_builder.h"

namespace {

using namespace orion;

constexpr double kPi = 3.14159265358979323846;

enum class BackendChoice {
    Auto,
    Cpu,
    Gpu,
};

struct StudioSettings {
    int width = 1280;
    int height = 720;
    int threadCount = static_cast<int>(std::thread::hardware_concurrency());

    int previewSpp = 1;
    int refineSpp = 2;

    int maxDepth = 8;
    int previewDepth = 4;

    std::uint64_t seed = 1337;
    double exposure = 1.0;
    double moveSpeed = 6.0;
    double mouseSensitivity = 0.13;
    double mouseSmoothing = 0.55;

    BackendChoice backend = BackendChoice::Auto;
};

struct FreeCameraRig {
    Point3 position{13.0, 2.5, 3.0};
    double yawDeg = -166.0;
    double pitchDeg = -8.0;

    double fovDeg = 26.0;
    double aperture = 0.09;
    double focusDistance = 10.5;

    [[nodiscard]] Vec3 forward() const {
        const double yaw = yawDeg * kPi / 180.0;
        const double pitch = pitchDeg * kPi / 180.0;

        const double c = std::cos(pitch);
        return normalize(Vec3(c * std::cos(yaw), std::sin(pitch), c * std::sin(yaw)));
    }

    [[nodiscard]] Vec3 right() const {
        const Vec3 up(0.0, 1.0, 0.0);
        Vec3 r = cross(forward(), up);
        if (r.nearZero()) {
            r = Vec3(1.0, 0.0, 0.0);
        }
        return normalize(r);
    }

    [[nodiscard]] CameraSettings toCameraSettings(double aspectRatio) const {
        CameraSettings settings;
        settings.lookFrom = position;
        settings.lookAt = position + forward();
        settings.vUp = Vec3(0.0, 1.0, 0.0);
        settings.verticalFovDeg = fovDeg;
        settings.aspectRatio = aspectRatio;
        settings.aperture = aperture;
        settings.focusDistance = focusDistance;
        return settings;
    }

    static FreeCameraRig fromCameraSettings(const CameraSettings& settings) {
        FreeCameraRig rig;
        rig.position = settings.lookFrom;
        rig.fovDeg = settings.verticalFovDeg;
        rig.aperture = settings.aperture;
        rig.focusDistance = settings.focusDistance;

        const Vec3 dir = normalize(settings.lookAt - settings.lookFrom);
        rig.yawDeg = std::atan2(dir.z(), dir.x()) * 180.0 / kPi;
        rig.pitchDeg = std::asin(std::clamp(dir.y(), -1.0, 1.0)) * 180.0 / kPi;
        return rig;
    }
};

struct SharedPreviewState {
    std::mutex mutex;
    bool stop = false;

    // Camera/settings revisions let the worker detect when accumulation must reset.
    CameraSettings camera;
    bool moving = true;
    std::uint64_t cameraRevision = 1;
    std::uint64_t settingsRevision = 1;

    int previewSpp = 1;
    int refineSpp = 2;
    int maxDepth = 8;
    int previewDepth = 4;
    double exposure = 1.0;
    BackendChoice backend = BackendChoice::Auto;

    std::vector<unsigned char> displayRgb;
    std::uint64_t frameSerial = 0;

    int accumulatedSamples = 0;
    double lastBatchMs = 0.0;
    bool usingGpu = false;

    std::string backendStatus;
};

struct UiRow {
    std::string label;
    std::string value;
};

struct PanelLayout {
    float x0;
    float x1;
    float y0;
    float y1;
    int rowCount;
};

struct UiRect {
    float x0;
    float y0;
    float x1;
    float y1;
};

struct UiFrameLayout {
    UiRect sceneRect;
    UiRect leftToggleButton;
    UiRect rightToggleButton;
    UiRect leftFoldTab;
    UiRect rightFoldTab;
};

constexpr PanelLayout kLeftPanel{-0.985f, -0.705f, -0.34f, 0.54f, 7};
constexpr PanelLayout kRightPanel{0.705f, 0.985f, -0.34f, 0.54f, 5};
constexpr UiRect kTopBarRect{-0.985f, 0.942f, 0.985f, 0.985f};
constexpr UiRect kBottomBarRect{-0.985f, -0.985f, 0.985f, -0.952f};

void printUsage() {
    std::cout
        << "Raycism Engine\n"
        << "Usage:\n"
        << "  orion_studio [options]\n\n"
        << "Options:\n"
        << "  --width <int>          Render width (default: 1280)\n"
        << "  --height <int>         Render height (default: 720)\n"
        << "  --threads <int>        CPU worker threads (default: hw concurrency)\n"
        << "  --depth <int>          Refinement max depth (default: 8)\n"
        << "  --preview-depth <int>  Camera-motion depth (default: 4)\n"
        << "  --preview-spp <int>    Camera-motion spp batch (default: 1)\n"
        << "  --refine-spp <int>     Still-camera spp batch (default: 2)\n"
        << "  --seed <int>           Scene random seed (default: 1337)\n"
        << "  --exposure <float>     Display exposure (default: 1.0)\n"
        << "  --move-speed <float>   Camera move speed (default: 6.0)\n"
        << "  --mouse-sensitivity <float>  Mouse look sensitivity (default: 0.13)\n"
        << "  --mouse-smoothing <float>    Mouse smoothing [0..0.95] (default: 0.55)\n"
        << "  --backend <mode>       auto | cpu | gpu (default: auto)\n"
        << "  --help                 Show this message\n\n"
        << "Viewport Controls:\n"
        << "  RMB drag: look around\n"
        << "  WASD + Q/E: move camera\n"
        << "  Shift: faster movement\n"
        << "  Z/X: FOV down/up\n"
        << "  R/F: focus distance up/down\n"
        << "  C/V: aperture up/down\n"
        << "\n"
        << "UI Controls:\n"
        << "  Click +/- buttons on left/right side panels\n"
        << "  Click L/R ON buttons or side tabs to fold panels\n"
        << "  Render stays in center viewport (never under side panels)\n"
        << "\n"
        << "  Esc: quit\n";
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

[[nodiscard]] inline double acesToneMap(double x) {
    constexpr double a = 2.51;
    constexpr double b = 0.03;
    constexpr double c = 2.43;
    constexpr double d = 0.59;
    constexpr double e = 0.14;
    return std::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

std::string formatFixed(double value, int precision = 2) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(precision);
    out << value;
    return out.str();
}

std::string backendToString(BackendChoice backend) {
    if (backend == BackendChoice::Cpu) {
        return "cpu";
    }
    if (backend == BackendChoice::Gpu) {
        return "gpu";
    }
    return "auto";
}

BackendChoice nextBackend(BackendChoice current) {
    if (current == BackendChoice::Auto) {
        return BackendChoice::Cpu;
    }
    if (current == BackendChoice::Cpu) {
        return BackendChoice::Gpu;
    }
    return BackendChoice::Auto;
}

BackendChoice previousBackend(BackendChoice current) {
    if (current == BackendChoice::Auto) {
        return BackendChoice::Gpu;
    }
    if (current == BackendChoice::Gpu) {
        return BackendChoice::Cpu;
    }
    return BackendChoice::Auto;
}

void convertAccumToDisplay(
    const std::vector<double>& accum,
    int totalSamples,
    double exposure,
    std::vector<unsigned char>& rgbOut) {
    const std::size_t pixelCount = accum.size() / 3;
    rgbOut.resize(pixelCount * 3);

    const double invSamples = totalSamples > 0 ? (1.0 / static_cast<double>(totalSamples)) : 0.0;
    const double invGamma = 1.0 / 2.2;

    for (std::size_t i = 0; i < pixelCount; ++i) {
        double r = accum[3 * i + 0] * invSamples * exposure;
        double g = accum[3 * i + 1] * invSamples * exposure;
        double b = accum[3 * i + 2] * invSamples * exposure;

        r = std::pow(std::max(acesToneMap(r), 0.0), invGamma);
        g = std::pow(std::max(acesToneMap(g), 0.0), invGamma);
        b = std::pow(std::max(acesToneMap(b), 0.0), invGamma);

        r = std::clamp(r, 0.0, 0.999);
        g = std::clamp(g, 0.0, 0.999);
        b = std::clamp(b, 0.0, 0.999);

        rgbOut[3 * i + 0] = static_cast<unsigned char>(256.0 * r);
        rgbOut[3 * i + 1] = static_cast<unsigned char>(256.0 * g);
        rgbOut[3 * i + 2] = static_cast<unsigned char>(256.0 * b);
    }
}

bool renderBatchCpu(
    const std::shared_ptr<Hittable>& world,
    const CameraSettings& cameraSettings,
    const RenderSettings& settings,
    Image& outImage) {
    Camera camera(cameraSettings);
    Renderer renderer(settings);
    outImage = renderer.render(*world, camera);
    return true;
}

bool renderBatchGpu(
    GpuSceneData& scene,
    const CameraSettings& cameraSettings,
    const RenderSettings& settings,
    Image& outImage,
    std::string& errorMessage) {
    scene.camera = buildGpuCamera(cameraSettings);
    return renderWithGpu(scene, settings, outImage, {}, errorMessage);
}

bool pointInRect(float x, float y, float x0, float y0, float x1, float y1) {
    return x >= x0 && x <= x1 && y >= y0 && y <= y1;
}

bool pointInRect(float x, float y, const UiRect& rect) {
    return pointInRect(x, y, rect.x0, rect.y0, rect.x1, rect.y1);
}

UiFrameLayout buildUiFrameLayout(bool leftPanelVisible, bool rightPanelVisible) {
    UiFrameLayout layout;

    constexpr float scenePad = 0.014f;
    const float sceneLeftBase = leftPanelVisible ? kLeftPanel.x1 : kTopBarRect.x0;
    const float sceneRightBase = rightPanelVisible ? kRightPanel.x0 : kTopBarRect.x1;

    layout.sceneRect.x0 = sceneLeftBase + scenePad;
    layout.sceneRect.x1 = sceneRightBase - scenePad;
    layout.sceneRect.y0 = kBottomBarRect.y1 + scenePad;
    layout.sceneRect.y1 = kTopBarRect.y0 - scenePad;

    if (layout.sceneRect.x1 <= layout.sceneRect.x0 + 0.05f) {
        const float center = (layout.sceneRect.x0 + layout.sceneRect.x1) * 0.5f;
        layout.sceneRect.x0 = center - 0.025f;
        layout.sceneRect.x1 = center + 0.025f;
    }

    layout.leftToggleButton = UiRect{-0.972f, 0.946f, -0.882f, 0.980f};
    layout.rightToggleButton = UiRect{-0.874f, 0.946f, -0.776f, 0.980f};

    if (leftPanelVisible) {
        layout.leftFoldTab = UiRect{kLeftPanel.x1 - 0.010f, -0.090f, kLeftPanel.x1 + 0.016f, 0.090f};
    } else {
        layout.leftFoldTab = UiRect{kTopBarRect.x0, -0.090f, kTopBarRect.x0 + 0.026f, 0.090f};
    }

    if (rightPanelVisible) {
        layout.rightFoldTab = UiRect{kRightPanel.x0 - 0.016f, -0.090f, kRightPanel.x0 + 0.010f, 0.090f};
    } else {
        layout.rightFoldTab = UiRect{kTopBarRect.x1 - 0.026f, -0.090f, kTopBarRect.x1, 0.090f};
    }

    return layout;
}

void drawSolidRect(float x0, float y0, float x1, float y1) {
    glBegin(GL_QUADS);
    glVertex2f(x0, y0);
    glVertex2f(x1, y0);
    glVertex2f(x1, y1);
    glVertex2f(x0, y1);
    glEnd();
}

void drawLineRect(float x0, float y0, float x1, float y1) {
    glBegin(GL_LINE_LOOP);
    glVertex2f(x0, y0);
    glVertex2f(x1, y0);
    glVertex2f(x1, y1);
    glVertex2f(x0, y1);
    glEnd();
}

const std::array<std::uint8_t, 7>& glyph5x7(char c) {
    static const std::array<std::uint8_t, 7> blank{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

    static const std::array<std::uint8_t, 7> gA{0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    static const std::array<std::uint8_t, 7> gB{0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E};
    static const std::array<std::uint8_t, 7> gC{0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E};
    static const std::array<std::uint8_t, 7> gD{0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E};
    static const std::array<std::uint8_t, 7> gE{0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F};
    static const std::array<std::uint8_t, 7> gF{0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10};
    static const std::array<std::uint8_t, 7> gG{0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E};
    static const std::array<std::uint8_t, 7> gH{0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    static const std::array<std::uint8_t, 7> gI{0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1F};
    static const std::array<std::uint8_t, 7> gJ{0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C};
    static const std::array<std::uint8_t, 7> gK{0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11};
    static const std::array<std::uint8_t, 7> gL{0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F};
    static const std::array<std::uint8_t, 7> gM{0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11};
    static const std::array<std::uint8_t, 7> gN{0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11};
    static const std::array<std::uint8_t, 7> gO{0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    static const std::array<std::uint8_t, 7> gP{0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10};
    static const std::array<std::uint8_t, 7> gQ{0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D};
    static const std::array<std::uint8_t, 7> gR{0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11};
    static const std::array<std::uint8_t, 7> gS{0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E};
    static const std::array<std::uint8_t, 7> gT{0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04};
    static const std::array<std::uint8_t, 7> gU{0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    static const std::array<std::uint8_t, 7> gV{0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04};
    static const std::array<std::uint8_t, 7> gW{0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A};
    static const std::array<std::uint8_t, 7> gX{0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11};
    static const std::array<std::uint8_t, 7> gY{0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04};
    static const std::array<std::uint8_t, 7> gZ{0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F};

    static const std::array<std::uint8_t, 7> g0{0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E};
    static const std::array<std::uint8_t, 7> g1{0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E};
    static const std::array<std::uint8_t, 7> g2{0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F};
    static const std::array<std::uint8_t, 7> g3{0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E};
    static const std::array<std::uint8_t, 7> g4{0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02};
    static const std::array<std::uint8_t, 7> g5{0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E};
    static const std::array<std::uint8_t, 7> g6{0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E};
    static const std::array<std::uint8_t, 7> g7{0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
    static const std::array<std::uint8_t, 7> g8{0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E};
    static const std::array<std::uint8_t, 7> g9{0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E};

    static const std::array<std::uint8_t, 7> gDot{0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C};
    static const std::array<std::uint8_t, 7> gDash{0x00, 0x00, 0x00, 0x1E, 0x00, 0x00, 0x00};
    static const std::array<std::uint8_t, 7> gPlus{0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00};
    static const std::array<std::uint8_t, 7> gColon{0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00};
    static const std::array<std::uint8_t, 7> gSlash{0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10};
    static const std::array<std::uint8_t, 7> gLess{0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02};
    static const std::array<std::uint8_t, 7> gGreater{0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08};
    static const std::array<std::uint8_t, 7> gSpace{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

    const unsigned char up = static_cast<unsigned char>(std::toupper(static_cast<unsigned char>(c)));
    switch (up) {
        case 'A': return gA; case 'B': return gB; case 'C': return gC; case 'D': return gD;
        case 'E': return gE; case 'F': return gF; case 'G': return gG; case 'H': return gH;
        case 'I': return gI; case 'J': return gJ; case 'K': return gK; case 'L': return gL;
        case 'M': return gM; case 'N': return gN; case 'O': return gO; case 'P': return gP;
        case 'Q': return gQ; case 'R': return gR; case 'S': return gS; case 'T': return gT;
        case 'U': return gU; case 'V': return gV; case 'W': return gW; case 'X': return gX;
        case 'Y': return gY; case 'Z': return gZ;
        case '0': return g0; case '1': return g1; case '2': return g2; case '3': return g3;
        case '4': return g4; case '5': return g5; case '6': return g6; case '7': return g7;
        case '8': return g8; case '9': return g9;
        case '.': return gDot; case '-': return gDash; case '+': return gPlus; case ':': return gColon;
        case '/': return gSlash; case '<': return gLess; case '>': return gGreater; case ' ': return gSpace;
        default: return blank;
    }
}

void drawText5x7(
    const std::string& text,
    float xLeft,
    float yTop,
    float scale,
    float r,
    float g,
    float b,
    float a) {
    glColor4f(r, g, b, a);

    float penX = xLeft;
    for (char c : text) {
        const auto& glyph = glyph5x7(c);
        for (int row = 0; row < 7; ++row) {
            const std::uint8_t bits = glyph[static_cast<std::size_t>(row)];
            const float rowTop = yTop - static_cast<float>(row) * scale;
            const float rowBottom = rowTop - scale;
            for (int col = 0; col < 5; ++col) {
                if ((bits & static_cast<std::uint8_t>(1u << (4 - col))) != 0u) {
                    const float left = penX + static_cast<float>(col) * scale;
                    drawSolidRect(left, rowBottom, left + scale, rowTop);
                }
            }
        }
        penX += 6.0f * scale;
    }
}

float measureTextWidth5x7(const std::string& text, float scale) {
    return static_cast<float>(text.size()) * 6.0f * scale;
}

std::string truncateTextToFit5x7(const std::string& text, float scale, float maxWidth) {
    if (text.empty() || scale <= 0.0f || maxWidth <= 0.0f) {
        return {};
    }

    const int maxChars = static_cast<int>(std::floor(maxWidth / (6.0f * scale)));
    if (maxChars <= 0) {
        return {};
    }
    if (static_cast<int>(text.size()) <= maxChars) {
        return text;
    }
    if (maxChars <= 3) {
        return text.substr(0, static_cast<std::size_t>(maxChars));
    }
    return text.substr(0, static_cast<std::size_t>(maxChars - 3)) + "...";
}

struct FittedText5x7 {
    std::string text;
    float scale = 0.0f;
};

FittedText5x7 fitText5x7(
    const std::string& text,
    float preferredScale,
    float minScale,
    float maxWidth) {
    FittedText5x7 fitted;
    if (text.empty() || maxWidth <= 0.0f || preferredScale <= 0.0f) {
        return fitted;
    }

    const float safeMinScale = std::max(0.0001f, minScale);
    float scale = preferredScale;

    const float preferredWidth = measureTextWidth5x7(text, preferredScale);
    if (preferredWidth > maxWidth) {
        const float idealScale = maxWidth / (6.0f * static_cast<float>(text.size()));
        scale = std::max(safeMinScale, std::min(preferredScale, idealScale));
    }

    fitted.scale = scale;
    fitted.text = truncateTextToFit5x7(text, scale, maxWidth);
    return fitted;
}

void drawFittedText5x7(
    const std::string& text,
    float xLeft,
    float yTop,
    float maxWidth,
    float preferredScale,
    float minScale,
    float r,
    float g,
    float b,
    float a,
    bool rightAlign = false) {
    const FittedText5x7 fitted = fitText5x7(text, preferredScale, minScale, maxWidth);
    if (fitted.text.empty() || fitted.scale <= 0.0f) {
        return;
    }

    float x = xLeft;
    if (rightAlign) {
        const float width = measureTextWidth5x7(fitted.text, fitted.scale);
        x += std::max(0.0f, maxWidth - width);
    }

    drawText5x7(fitted.text, x, yTop, fitted.scale, r, g, b, a);
}

void drawButton(float x0, float y0, float x1, float y1, bool hovered, const std::string& label) {
    glColor4f(0.0f, 0.0f, 0.0f, hovered ? 0.90f : 0.78f);
    drawSolidRect(x0, y0, x1, y1);
    glColor4f(1.0f, 1.0f, 1.0f, hovered ? 0.98f : 0.88f);
    drawLineRect(x0, y0, x1, y1);
    glColor4f(1.0f, 1.0f, 1.0f, hovered ? 0.36f : 0.24f);
    drawLineRect(x0 + 0.0018f, y0 + 0.0018f, x1 - 0.0018f, y1 - 0.0018f);

    const float boxWidth = std::max(0.0f, x1 - x0);
    const float boxHeight = std::max(0.0f, y1 - y0);
    const FittedText5x7 fitted = fitText5x7(label, 0.0053f, 0.0034f, boxWidth - 0.005f);
    if (!fitted.text.empty()) {
        const float textW = measureTextWidth5x7(fitted.text, fitted.scale);
        const float textH = 7.0f * fitted.scale;
        const float x = x0 + (boxWidth - textW) * 0.5f;
        const float yTop = y0 + (boxHeight + textH) * 0.5f;
        drawText5x7(fitted.text, x, yTop, fitted.scale, 1.0f, 1.0f, 1.0f, hovered ? 0.99f : 0.95f);
    }
}

void drawFoldTab(
    const UiRect& rect,
    bool hovered,
    bool panelVisible,
    bool leftSide) {
    glColor4f(0.0f, 0.0f, 0.0f, hovered ? 0.90f : 0.79f);
    drawSolidRect(rect.x0, rect.y0, rect.x1, rect.y1);
    glColor4f(1.0f, 1.0f, 1.0f, hovered ? 0.98f : 0.90f);
    drawLineRect(rect.x0, rect.y0, rect.x1, rect.y1);

    const std::string arrow = leftSide
        ? (panelVisible ? "<" : ">")
        : (panelVisible ? ">" : "<");
    const float w = std::max(0.0f, rect.x1 - rect.x0);
    const float h = std::max(0.0f, rect.y1 - rect.y0);
    const FittedText5x7 fitted = fitText5x7(arrow, 0.0092f, 0.0068f, w - 0.006f);
    if (fitted.text.empty()) {
        return;
    }

    const float textW = measureTextWidth5x7(fitted.text, fitted.scale);
    const float textH = 7.0f * fitted.scale;
    const float x = rect.x0 + (w - textW) * 0.5f;
    const float yTop = rect.y0 + (h + textH) * 0.5f;
    drawText5x7(fitted.text, x, yTop, fitted.scale, 1.0f, 1.0f, 1.0f, hovered ? 0.99f : 0.95f);
}

void drawPanelVisibilityButtons(
    const UiFrameLayout& ui,
    bool leftPanelVisible,
    bool rightPanelVisible,
    float mouseX,
    float mouseY) {
    const bool leftHover = pointInRect(mouseX, mouseY, ui.leftToggleButton);
    const bool rightHover = pointInRect(mouseX, mouseY, ui.rightToggleButton);
    drawButton(
        ui.leftToggleButton.x0,
        ui.leftToggleButton.y0,
        ui.leftToggleButton.x1,
        ui.leftToggleButton.y1,
        leftHover,
        leftPanelVisible ? "L ON" : "L OFF");
    drawButton(
        ui.rightToggleButton.x0,
        ui.rightToggleButton.y0,
        ui.rightToggleButton.x1,
        ui.rightToggleButton.y1,
        rightHover,
        rightPanelVisible ? "R ON" : "R OFF");

    drawFoldTab(
        ui.leftFoldTab,
        pointInRect(mouseX, mouseY, ui.leftFoldTab),
        leftPanelVisible,
        true);
    drawFoldTab(
        ui.rightFoldTab,
        pointInRect(mouseX, mouseY, ui.rightFoldTab),
        rightPanelVisible,
        false);
}

void drawPanel(
    const PanelLayout& layout,
    const std::string& title,
    const std::vector<UiRow>& rows,
    float mouseX,
    float mouseY) {
    glColor4f(0.0f, 0.0f, 0.0f, 0.88f);
    drawSolidRect(layout.x0, layout.y0, layout.x1, layout.y1);
    glColor4f(1.0f, 1.0f, 1.0f, 0.92f);
    drawLineRect(layout.x0, layout.y0, layout.x1, layout.y1);
    glColor4f(1.0f, 1.0f, 1.0f, 0.20f);
    drawLineRect(layout.x0 + 0.002f, layout.y0 + 0.002f, layout.x1 - 0.002f, layout.y1 - 0.002f);

    drawFittedText5x7(
        title,
        layout.x0 + 0.012f,
        layout.y1 - 0.012f,
        (layout.x1 - layout.x0) - 0.024f,
        0.0058f,
        0.0042f,
        1.0f,
        1.0f,
        1.0f,
        0.98f);

    const float headerH = 0.047f;
    const float top = layout.y1 - headerH;
    const float rowArea = (top - layout.y0) - 0.008f;
    const float rowH = rowArea / static_cast<float>(layout.rowCount);

    glColor4f(1.0f, 1.0f, 1.0f, 0.30f);
    drawSolidRect(layout.x0 + 0.010f, top - 0.002f, layout.x1 - 0.010f, top + 0.001f);

    for (int i = 0; i < static_cast<int>(rows.size()) && i < layout.rowCount; ++i) {
        const float rowTop = top - static_cast<float>(i) * rowH;
        const float rowBottom = rowTop - rowH + 0.001f;

        const float stripeAlpha = (i % 2 == 0) ? 0.065f : 0.038f;
        glColor4f(1.0f, 1.0f, 1.0f, stripeAlpha);
        drawSolidRect(layout.x0 + 0.006f, rowBottom, layout.x1 - 0.006f, rowTop - 0.001f);

        const float minusX0 = layout.x1 - 0.079f;
        const float minusX1 = layout.x1 - 0.049f;
        const float plusX0 = layout.x1 - 0.044f;
        const float plusX1 = layout.x1 - 0.014f;
        const float by0 = rowBottom + 0.010f;
        const float by1 = rowTop - 0.010f;

        const float textX0 = layout.x0 + 0.012f;
        const float textX1 = minusX0 - 0.008f;
        const float textWidth = std::max(0.0f, textX1 - textX0);
        const float valueWidth = std::min(0.074f, textWidth * 0.40f);
        const float labelWidth = std::max(0.0f, textWidth - valueWidth - 0.006f);
        const float valueX = textX0 + labelWidth + 0.006f;
        const float rowTextTop = rowTop - rowH * 0.34f;

        drawFittedText5x7(
            rows[static_cast<std::size_t>(i)].label,
            textX0,
            rowTextTop,
            labelWidth,
            0.0048f,
            0.0037f,
            1.0f,
            1.0f,
            1.0f,
            0.94f);

        drawFittedText5x7(
            rows[static_cast<std::size_t>(i)].value,
            valueX,
            rowTextTop,
            valueWidth,
            0.0050f,
            0.0039f,
            1.0f,
            1.0f,
            1.0f,
            0.99f,
            true);

        drawButton(
            minusX0,
            by0,
            minusX1,
            by1,
            pointInRect(mouseX, mouseY, minusX0, by0, minusX1, by1),
            "-");
        drawButton(
            plusX0,
            by0,
            plusX1,
            by1,
            pointInRect(mouseX, mouseY, plusX0, by0, plusX1, by1),
            "+");
    }
}

void drawTopBottomBars(
    const std::string& backend,
    const std::string& mode,
    int samples,
    int previewSpp,
    int refineSpp,
    int maxDepth,
    double batchMs,
    const std::string& status,
    bool leftPanelVisible,
    bool rightPanelVisible) {
    glColor4f(0.0f, 0.0f, 0.0f, 0.90f);
    drawSolidRect(kTopBarRect.x0, kTopBarRect.y0, kTopBarRect.x1, kTopBarRect.y1);
    glColor4f(0.0f, 0.0f, 0.0f, 0.86f);
    drawSolidRect(kBottomBarRect.x0, kBottomBarRect.y0, kBottomBarRect.x1, kBottomBarRect.y1);
    glColor4f(1.0f, 1.0f, 1.0f, 0.88f);
    drawLineRect(kTopBarRect.x0, kTopBarRect.y0, kTopBarRect.x1, kTopBarRect.y1);
    glColor4f(1.0f, 1.0f, 1.0f, 0.80f);
    drawLineRect(kBottomBarRect.x0, kBottomBarRect.y0, kBottomBarRect.x1, kBottomBarRect.y1);

    std::string topLeft = "RAYCISM ENGINE";
    std::string topRight = "B " + backend +
                           " | M " + mode +
                           " | S " + std::to_string(samples) +
                           " | SP " + std::to_string(previewSpp) + "/" + std::to_string(refineSpp) +
                           " | D " + std::to_string(maxDepth) +
                           " | " + std::to_string(static_cast<int>(batchMs)) + "MS";

    std::string bottom = "RMB LOOK | WASD MOVE | L/R BTN OR TAB = FOLD";
    bottom += " | L " + std::string(leftPanelVisible ? "ON" : "OFF");
    bottom += " | R " + std::string(rightPanelVisible ? "ON" : "OFF");
    if (!status.empty()) {
        bottom += " | STATUS " + status;
    }

    drawFittedText5x7(topLeft, -0.975f, 0.976f, 0.52f, 0.0058f, 0.0038f, 1.0f, 1.0f, 1.0f, 0.98f);
    drawFittedText5x7(topRight, -0.450f, 0.976f, 1.42f, 0.0055f, 0.0035f, 1.0f, 1.0f, 1.0f, 0.93f);
    drawFittedText5x7(bottom, -0.975f, -0.958f, 1.95f, 0.0050f, 0.0034f, 1.0f, 1.0f, 1.0f, 0.92f);
}

void drawTexturedQuad(
    GLuint texture,
    int imageWidth,
    int imageHeight,
    int framebufferWidth,
    int framebufferHeight,
    const UiRect& targetRect) {
    if (framebufferWidth <= 0 || framebufferHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) {
        return;
    }

    const float imageAspect = static_cast<float>(imageWidth) / static_cast<float>(imageHeight);
    const float targetWidth = std::max(0.0001f, targetRect.x1 - targetRect.x0);
    const float targetHeight = std::max(0.0001f, targetRect.y1 - targetRect.y0);

    // Fit by pixel dimensions so aspect ratio stays correct on any window size.
    const float targetPixelWidth = targetWidth * 0.5f * static_cast<float>(framebufferWidth);
    const float targetPixelHeight = targetHeight * 0.5f * static_cast<float>(framebufferHeight);
    const float targetPixelAspect = targetPixelWidth / std::max(0.0001f, targetPixelHeight);

    float drawPixelWidth = targetPixelWidth;
    float drawPixelHeight = targetPixelHeight;
    if (targetPixelAspect > imageAspect) {
        drawPixelWidth = targetPixelHeight * imageAspect;
    } else {
        drawPixelHeight = targetPixelWidth / imageAspect;
    }

    const float drawWidth = (drawPixelWidth * 2.0f) / static_cast<float>(framebufferWidth);
    const float drawHeight = (drawPixelHeight * 2.0f) / static_cast<float>(framebufferHeight);

    const float centerX = (targetRect.x0 + targetRect.x1) * 0.5f;
    const float centerY = (targetRect.y0 + targetRect.y1) * 0.5f;
    const float halfW = drawWidth * 0.5f;
    const float halfH = drawHeight * 0.5f;
    const float x0 = centerX - halfW;
    const float x1 = centerX + halfW;
    const float y0 = centerY - halfH;
    const float y1 = centerY + halfH;

    glViewport(0, 0, framebufferWidth, framebufferHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(x0, y0);

    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(x1, y0);

    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(x1, y1);

    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(x0, y1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
}

void renderWorker(const StudioSettings& studio, SharedPreviewState& shared) {
    const bool canUseGpu = gpuBackendCompiled();

    auto cpuWorld = makeShowcaseWorldCpu(studio.seed);

    GpuSceneData gpuScene;
    if (canUseGpu) {
        GpuWorldData gpuWorld = makeShowcaseGpuWorld(studio.seed);
        gpuScene.materials = std::move(gpuWorld.materials);
        gpuScene.spheres = std::move(gpuWorld.spheres);
    }

    std::vector<double> accumulation(static_cast<std::size_t>(studio.width * studio.height * 3), 0.0);
    std::vector<unsigned char> rgb;

    std::uint64_t activeCameraRevision = 0;
    std::uint64_t activeSettingsRevision = 0;
    int accumulatedSamples = 0;

    // Background progressive renderer: runs preview/refine batches while UI thread stays responsive.
    while (true) {
        CameraSettings camera;
        bool moving = true;
        bool stop = false;
        std::uint64_t cameraRevision = 0;
        std::uint64_t settingsRevision = 0;

        int previewSpp = 1;
        int refineSpp = 2;
        int maxDepth = 8;
        int previewDepth = 4;
        double exposure = 1.0;
        BackendChoice backend = BackendChoice::Auto;

        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            stop = shared.stop;
            camera = shared.camera;
            moving = shared.moving;
            cameraRevision = shared.cameraRevision;
            settingsRevision = shared.settingsRevision;

            previewSpp = shared.previewSpp;
            refineSpp = shared.refineSpp;
            maxDepth = shared.maxDepth;
            previewDepth = shared.previewDepth;
            exposure = shared.exposure;
            backend = shared.backend;
        }

        if (stop) {
            break;
        }

        if (cameraRevision != activeCameraRevision || settingsRevision != activeSettingsRevision) {
            // Any camera/setting edit invalidates accumulation from previous viewpoint/settings.
            std::fill(accumulation.begin(), accumulation.end(), 0.0);
            accumulatedSamples = 0;
            activeCameraRevision = cameraRevision;
            activeSettingsRevision = settingsRevision;
        }

        bool useGpu = false;
        if (backend == BackendChoice::Gpu) {
            useGpu = canUseGpu;
        } else if (backend == BackendChoice::Auto) {
            useGpu = canUseGpu;
        }

        if (backend == BackendChoice::Gpu && !canUseGpu) {
            std::lock_guard<std::mutex> lock(shared.mutex);
            shared.backendStatus = "GPU requested but this build has no CUDA; using CPU";
        }

        RenderSettings settings;
        settings.width = studio.width;
        settings.height = studio.height;
        settings.threadCount = studio.threadCount;
        // While camera is moving, trade quality for responsiveness.
        settings.samplesPerPixel = moving ? previewSpp : refineSpp;
        settings.maxDepth = moving ? std::min(maxDepth, previewDepth) : maxDepth;
        settings.seed = studio.seed ^ (activeCameraRevision * 0x9E3779B97F4A7C15ULL) ^ static_cast<std::uint64_t>(accumulatedSamples + 1);
        settings.exposure = exposure;

        Image batchImage(studio.width, studio.height);
        std::string batchError;

        const auto start = std::chrono::steady_clock::now();

        bool rendered = false;
        if (useGpu) {
            rendered = renderBatchGpu(gpuScene, camera, settings, batchImage, batchError);
            if (!rendered) {
                useGpu = false;
                if (backend == BackendChoice::Gpu) {
                    std::lock_guard<std::mutex> lock(shared.mutex);
                    shared.backendStatus = "GPU error: " + batchError;
                }
            }
        }

        if (!rendered) {
            rendered = renderBatchCpu(cpuWorld, camera, settings, batchImage);
        }

        const auto end = std::chrono::steady_clock::now();
        const double elapsedMs = static_cast<double>(
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

        std::uint64_t latestCameraRevision = 0;
        std::uint64_t latestSettingsRevision = 0;
        bool shouldStop = false;
        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            latestCameraRevision = shared.cameraRevision;
            latestSettingsRevision = shared.settingsRevision;
            shouldStop = shared.stop;
        }

        if (shouldStop) {
            break;
        }

        if (latestCameraRevision != activeCameraRevision || latestSettingsRevision != activeSettingsRevision) {
            // Drop stale result if state changed while this batch was rendering.
            continue;
        }

        for (int y = 0; y < studio.height; ++y) {
            for (int x = 0; x < studio.width; ++x) {
                const Color c = batchImage.getPixel(x, y);
                const std::size_t idx = static_cast<std::size_t>(y * studio.width + x) * 3;
                accumulation[idx + 0] += c.x();
                accumulation[idx + 1] += c.y();
                accumulation[idx + 2] += c.z();
            }
        }

        accumulatedSamples += settings.samplesPerPixel;
        convertAccumToDisplay(accumulation, accumulatedSamples, exposure, rgb);

        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            shared.displayRgb = rgb;
            shared.accumulatedSamples = accumulatedSamples;
            shared.lastBatchMs = elapsedMs;
            shared.usingGpu = useGpu;
            if (!batchError.empty()) {
                shared.backendStatus = batchError;
            }
            shared.frameSerial += 1;
        }
    }
}

bool handlePanelClick(
    const PanelLayout& layout,
    int rowCount,
    float mouseX,
    float mouseY,
    bool clicked,
    std::function<void(int, bool)> onAdjust,
    std::function<void(int)> onToggle = {}) {
    if (!clicked || !pointInRect(mouseX, mouseY, layout.x0, layout.y0, layout.x1, layout.y1)) {
        return false;
    }

    const float headerH = 0.047f;
    const float top = layout.y1 - headerH;
    const float rowArea = (top - layout.y0) - 0.008f;
    const float rowH = rowArea / static_cast<float>(layout.rowCount);

    for (int i = 0; i < rowCount; ++i) {
        const float rowTop = top - static_cast<float>(i) * rowH;
        const float rowBottom = rowTop - rowH + 0.001f;

        const float minusX0 = layout.x1 - 0.079f;
        const float minusX1 = layout.x1 - 0.049f;
        const float plusX0 = layout.x1 - 0.044f;
        const float plusX1 = layout.x1 - 0.014f;
        const float by0 = rowBottom + 0.010f;
        const float by1 = rowTop - 0.010f;

        if (pointInRect(mouseX, mouseY, minusX0, by0, minusX1, by1)) {
            onAdjust(i, false);
            return true;
        }

        if (pointInRect(mouseX, mouseY, plusX0, by0, plusX1, by1)) {
            onAdjust(i, true);
            return true;
        }

        if (onToggle && pointInRect(mouseX, mouseY, layout.x0 + 0.008f, rowBottom, minusX0 - 0.010f, rowTop)) {
            onToggle(i);
            return true;
        }
    }

    return false;
}

}  // namespace

int main(int argc, char** argv) {
    StudioSettings studio;

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
        if (arg == "--width") {
            if (!needsValue(arg) || !readIntArg(argv[++i], studio.width) || studio.width < 64) {
                std::cerr << "Invalid --width value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--height") {
            if (!needsValue(arg) || !readIntArg(argv[++i], studio.height) || studio.height < 64) {
                std::cerr << "Invalid --height value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--threads") {
            if (!needsValue(arg) || !readIntArg(argv[++i], studio.threadCount) || studio.threadCount < 1) {
                std::cerr << "Invalid --threads value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--depth") {
            if (!needsValue(arg) || !readIntArg(argv[++i], studio.maxDepth) || studio.maxDepth < 1) {
                std::cerr << "Invalid --depth value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--preview-depth") {
            if (!needsValue(arg) || !readIntArg(argv[++i], studio.previewDepth) || studio.previewDepth < 1) {
                std::cerr << "Invalid --preview-depth value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--preview-spp") {
            if (!needsValue(arg) || !readIntArg(argv[++i], studio.previewSpp) || studio.previewSpp < 1) {
                std::cerr << "Invalid --preview-spp value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--refine-spp") {
            if (!needsValue(arg) || !readIntArg(argv[++i], studio.refineSpp) || studio.refineSpp < 1) {
                std::cerr << "Invalid --refine-spp value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--seed") {
            if (!needsValue(arg) || !readUInt64Arg(argv[++i], studio.seed)) {
                std::cerr << "Invalid --seed value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--exposure") {
            if (!needsValue(arg) || !readDoubleArg(argv[++i], studio.exposure) || studio.exposure <= 0.0) {
                std::cerr << "Invalid --exposure value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--move-speed") {
            if (!needsValue(arg) || !readDoubleArg(argv[++i], studio.moveSpeed) || studio.moveSpeed <= 0.0) {
                std::cerr << "Invalid --move-speed value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--mouse-sensitivity") {
            if (!needsValue(arg) || !readDoubleArg(argv[++i], studio.mouseSensitivity) || studio.mouseSensitivity <= 0.0) {
                std::cerr << "Invalid --mouse-sensitivity value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--mouse-smoothing") {
            if (!needsValue(arg) || !readDoubleArg(argv[++i], studio.mouseSmoothing) ||
                studio.mouseSmoothing < 0.0 || studio.mouseSmoothing > 0.95) {
                std::cerr << "Invalid --mouse-smoothing value (use 0..0.95)\n";
                return 1;
            }
            continue;
        }
        if (arg == "--backend") {
            if (!needsValue(arg) || !readBackendArg(argv[++i], studio.backend)) {
                std::cerr << "Invalid --backend value. Use auto|cpu|gpu\n";
                return 1;
            }
            continue;
        }

        std::cerr << "Unknown option: " << arg << "\n";
        printUsage();
        return 1;
    }

    if (!glfwInit()) {
        std::cerr << "GLFW initialization failed\n";
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    GLFWwindow* window = glfwCreateWindow(studio.width, studio.height, "Raycism Engine", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        studio.width,
        studio.height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        nullptr);

    CameraSettings initialCamera = makeShowcaseCameraSettings(
        static_cast<double>(studio.width) / static_cast<double>(studio.height));
    FreeCameraRig cameraRig = FreeCameraRig::fromCameraSettings(initialCamera);

    int previewSpp = studio.previewSpp;
    int refineSpp = studio.refineSpp;
    int maxDepth = studio.maxDepth;
    int previewDepth = std::min(studio.previewDepth, studio.maxDepth);
    double exposure = studio.exposure;
    double moveSpeed = studio.moveSpeed;
    double mouseSensitivity = studio.mouseSensitivity;
    double mouseSmoothing = studio.mouseSmoothing;
    BackendChoice backendChoice = studio.backend;

    SharedPreviewState shared;
    shared.camera = cameraRig.toCameraSettings(static_cast<double>(studio.width) / static_cast<double>(studio.height));
    shared.moving = true;
    shared.previewSpp = previewSpp;
    shared.refineSpp = refineSpp;
    shared.maxDepth = maxDepth;
    shared.previewDepth = previewDepth;
    shared.exposure = exposure;
    shared.backend = backendChoice;
    shared.backendStatus = "Initializing renderer...";

    std::thread worker([&]() { renderWorker(studio, shared); });

    std::cout << "Raycism Engine running.\n";

    bool mouseCaptured = false;
    bool firstMouse = true;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    double smoothedDeltaX = 0.0;
    double smoothedDeltaY = 0.0;
    const bool rawMouseSupported = glfwRawMouseMotionSupported();

    bool leftMouseWasDown = false;
    bool leftPanelVisible = true;
    bool rightPanelVisible = true;

    double previousTime = glfwGetTime();
    auto lastCameraChange = std::chrono::steady_clock::now();

    std::uint64_t lastFrameSerial = 0;
    std::vector<unsigned char> uploadBuffer(static_cast<std::size_t>(studio.width * studio.height * 3), 0);

    auto lastTitleUpdate = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window)) {
        const double now = glfwGetTime();
        const double dt = now - previousTime;
        previousTime = now;

        glfwPollEvents();

        int fbw = 0;
        int fbh = 0;
        glfwGetFramebufferSize(window, &fbw, &fbh);

        double cursorX = 0.0;
        double cursorY = 0.0;
        glfwGetCursorPos(window, &cursorX, &cursorY);

        const float mouseNdcX = fbw > 0 ? static_cast<float>((2.0 * cursorX) / static_cast<double>(fbw) - 1.0) : 0.0f;
        const float mouseNdcY = fbh > 0 ? static_cast<float>(1.0 - (2.0 * cursorY) / static_cast<double>(fbh)) : 0.0f;

        const bool leftMouseDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        const bool leftMouseClicked = leftMouseDown && !leftMouseWasDown;
        leftMouseWasDown = leftMouseDown;

        UiFrameLayout uiLayout = buildUiFrameLayout(leftPanelVisible, rightPanelVisible);

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        bool cameraChanged = false;
        bool settingsChanged = false;

        const bool rightMouseDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        if (rightMouseDown && !mouseCaptured) {
            mouseCaptured = true;
            firstMouse = true;
            smoothedDeltaX = 0.0;
            smoothedDeltaY = 0.0;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            if (rawMouseSupported) {
                glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
            }
        } else if (!rightMouseDown && mouseCaptured) {
            mouseCaptured = false;
            smoothedDeltaX = 0.0;
            smoothedDeltaY = 0.0;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            if (rawMouseSupported) {
                glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
            }
        }

        if (mouseCaptured) {
            double mouseX = 0.0;
            double mouseY = 0.0;
            glfwGetCursorPos(window, &mouseX, &mouseY);

            if (firstMouse) {
                firstMouse = false;
                lastMouseX = mouseX;
                lastMouseY = mouseY;
            }

            const double deltaX = mouseX - lastMouseX;
            const double deltaY = mouseY - lastMouseY;
            lastMouseX = mouseX;
            lastMouseY = mouseY;

            const double clampedDx = std::clamp(deltaX, -240.0, 240.0);
            const double clampedDy = std::clamp(deltaY, -240.0, 240.0);
            const double smooth = std::clamp(mouseSmoothing, 0.0, 0.95);
            smoothedDeltaX = smoothedDeltaX * smooth + clampedDx * (1.0 - smooth);
            smoothedDeltaY = smoothedDeltaY * smooth + clampedDy * (1.0 - smooth);

            cameraRig.yawDeg += smoothedDeltaX * mouseSensitivity;
            cameraRig.pitchDeg -= smoothedDeltaY * mouseSensitivity;
            cameraRig.pitchDeg = std::clamp(cameraRig.pitchDeg, -89.0, 89.0);
            cameraChanged = true;
        }

        const double speedMul = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ? 3.0 : 1.0;
        const double moveDistance = moveSpeed * speedMul * dt;

        const Vec3 fwd = cameraRig.forward();
        const Vec3 right = cameraRig.right();
        const Vec3 up(0.0, 1.0, 0.0);

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            cameraRig.position += fwd * moveDistance;
            cameraChanged = true;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            cameraRig.position -= fwd * moveDistance;
            cameraChanged = true;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            cameraRig.position -= right * moveDistance;
            cameraChanged = true;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            cameraRig.position += right * moveDistance;
            cameraChanged = true;
        }
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            cameraRig.position -= up * moveDistance;
            cameraChanged = true;
        }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            cameraRig.position += up * moveDistance;
            cameraChanged = true;
        }

        const double cameraParamStep = 0.8 * dt;
        if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
            cameraRig.fovDeg = std::max(12.0, cameraRig.fovDeg - 35.0 * dt);
            cameraChanged = true;
        }
        if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
            cameraRig.fovDeg = std::min(95.0, cameraRig.fovDeg + 35.0 * dt);
            cameraChanged = true;
        }
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            cameraRig.focusDistance = std::min(100.0, cameraRig.focusDistance + 10.0 * cameraParamStep);
            cameraChanged = true;
        }
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
            cameraRig.focusDistance = std::max(0.2, cameraRig.focusDistance - 10.0 * cameraParamStep);
            cameraChanged = true;
        }
        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
            cameraRig.aperture = std::min(0.3, cameraRig.aperture + 0.2 * cameraParamStep);
            cameraChanged = true;
        }
        if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS) {
            cameraRig.aperture = std::max(0.0, cameraRig.aperture - 0.2 * cameraParamStep);
            cameraChanged = true;
        }

        bool uiConsumedClick = false;
        if (!mouseCaptured && leftMouseClicked) {
            if (pointInRect(mouseNdcX, mouseNdcY, uiLayout.leftToggleButton) ||
                pointInRect(mouseNdcX, mouseNdcY, uiLayout.leftFoldTab)) {
                leftPanelVisible = !leftPanelVisible;
                uiConsumedClick = true;
            } else if (pointInRect(mouseNdcX, mouseNdcY, uiLayout.rightToggleButton) ||
                       pointInRect(mouseNdcX, mouseNdcY, uiLayout.rightFoldTab)) {
                rightPanelVisible = !rightPanelVisible;
                uiConsumedClick = true;
            }

            if (uiConsumedClick) {
                uiLayout = buildUiFrameLayout(leftPanelVisible, rightPanelVisible);
            }
        }

        if (!mouseCaptured && !uiConsumedClick) {
            if (leftPanelVisible) {
                handlePanelClick(
                    kLeftPanel,
                    7,
                    mouseNdcX,
                    mouseNdcY,
                    leftMouseClicked,
                    [&](int row, bool plus) {
                        switch (row) {
                            case 0:
                                exposure = std::clamp(exposure + (plus ? 0.10 : -0.10), 0.05, 8.0);
                                settingsChanged = true;
                                break;
                            case 1:
                                previewSpp = std::clamp(previewSpp + (plus ? 1 : -1), 1, 64);
                                settingsChanged = true;
                                break;
                            case 2:
                                refineSpp = std::clamp(refineSpp + (plus ? 1 : -1), 1, 1024);
                                settingsChanged = true;
                                break;
                            case 3:
                                maxDepth = std::clamp(maxDepth + (plus ? 1 : -1), 1, 48);
                                previewDepth = std::min(previewDepth, maxDepth);
                                settingsChanged = true;
                                break;
                            case 4:
                                previewDepth = std::clamp(previewDepth + (plus ? 1 : -1), 1, maxDepth);
                                settingsChanged = true;
                                break;
                            case 5:
                                moveSpeed = std::clamp(moveSpeed + (plus ? 0.25 : -0.25), 0.5, 20.0);
                                break;
                            case 6:
                                backendChoice = plus ? nextBackend(backendChoice) : previousBackend(backendChoice);
                                settingsChanged = true;
                                break;
                            default:
                                break;
                        }
                    });
            }

            if (rightPanelVisible) {
                handlePanelClick(
                    kRightPanel,
                    5,
                    mouseNdcX,
                    mouseNdcY,
                    leftMouseClicked,
                    [&](int row, bool plus) {
                        switch (row) {
                            case 0:
                                cameraRig.fovDeg = std::clamp(cameraRig.fovDeg + (plus ? 1.0 : -1.0), 12.0, 95.0);
                                cameraChanged = true;
                                break;
                            case 1:
                                cameraRig.focusDistance = std::clamp(cameraRig.focusDistance + (plus ? 0.35 : -0.35), 0.2, 120.0);
                                cameraChanged = true;
                                break;
                            case 2:
                                cameraRig.aperture = std::clamp(cameraRig.aperture + (plus ? 0.01 : -0.01), 0.0, 0.35);
                                cameraChanged = true;
                                break;
                            case 3:
                                mouseSensitivity = std::clamp(mouseSensitivity + (plus ? 0.01 : -0.01), 0.02, 0.50);
                                break;
                            case 4:
                                mouseSmoothing = std::clamp(mouseSmoothing + (plus ? 0.03 : -0.03), 0.0, 0.95);
                                smoothedDeltaX = 0.0;
                                smoothedDeltaY = 0.0;
                                break;
                            default:
                                break;
                        }
                    });
            }
        }

        if (cameraChanged) {
            lastCameraChange = std::chrono::steady_clock::now();
        }

        // Keep preview mode active briefly after input stops to avoid rapid mode flicker.
        const bool cameraMoving =
            (std::chrono::steady_clock::now() - lastCameraChange) < std::chrono::milliseconds(180);

        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            shared.moving = cameraMoving;
            if (cameraChanged) {
                shared.camera = cameraRig.toCameraSettings(
                    static_cast<double>(studio.width) / static_cast<double>(studio.height));
                shared.cameraRevision += 1;
            }
            if (settingsChanged) {
                shared.previewSpp = previewSpp;
                shared.refineSpp = refineSpp;
                shared.maxDepth = maxDepth;
                shared.previewDepth = previewDepth;
                shared.exposure = exposure;
                shared.backend = backendChoice;
                shared.settingsRevision += 1;
            }
        }

        int samples = 0;
        double batchMs = 0.0;
        bool usingGpu = false;
        std::string backendStatus;

        bool hasNewFrame = false;
        {
            std::lock_guard<std::mutex> lock(shared.mutex);
            samples = shared.accumulatedSamples;
            batchMs = shared.lastBatchMs;
            usingGpu = shared.usingGpu;
            backendStatus = shared.backendStatus;

            if (shared.frameSerial != lastFrameSerial && !shared.displayRgb.empty()) {
                uploadBuffer = shared.displayRgb;
                lastFrameSerial = shared.frameSerial;
                hasNewFrame = true;
            }
        }

        if (hasNewFrame) {
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                studio.width,
                studio.height,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                uploadBuffer.data());
        }

        drawTexturedQuad(texture, studio.width, studio.height, fbw, fbh, uiLayout.sceneRect);

        std::vector<UiRow> leftRows;
        leftRows.push_back(UiRow{"EXPOSURE", formatFixed(exposure, 2)});
        leftRows.push_back(UiRow{"PREV SPP", std::to_string(previewSpp)});
        leftRows.push_back(UiRow{"RFN SPP", std::to_string(refineSpp)});
        leftRows.push_back(UiRow{"MAX DPT", std::to_string(maxDepth)});
        leftRows.push_back(UiRow{"PREV DPT", std::to_string(previewDepth)});
        leftRows.push_back(UiRow{"MOVE SPD", formatFixed(moveSpeed, 2)});
        leftRows.push_back(UiRow{"BACKEND", backendToString(backendChoice)});

        std::vector<UiRow> rightRows;
        rightRows.push_back(UiRow{"FOV", formatFixed(cameraRig.fovDeg, 1)});
        rightRows.push_back(UiRow{"FOCUS", formatFixed(cameraRig.focusDistance, 2)});
        rightRows.push_back(UiRow{"APERT", formatFixed(cameraRig.aperture, 3)});
        rightRows.push_back(UiRow{"M SENS", formatFixed(mouseSensitivity, 2)});
        rightRows.push_back(UiRow{"M SMTH", formatFixed(mouseSmoothing, 2)});

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glDisable(GL_TEXTURE_2D);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        const std::string mode = cameraMoving ? "preview" : "refine";
        const std::string backendUsed = usingGpu ? "gpu" : "cpu";

        glColor4f(1.0f, 1.0f, 1.0f, 0.94f);
        drawLineRect(
            uiLayout.sceneRect.x0 - 0.004f,
            uiLayout.sceneRect.y0 - 0.004f,
            uiLayout.sceneRect.x1 + 0.004f,
            uiLayout.sceneRect.y1 + 0.004f);

        if (leftPanelVisible) {
            drawPanel(kLeftPanel, "RENDER", leftRows, mouseNdcX, mouseNdcY);
        }
        if (rightPanelVisible) {
            drawPanel(kRightPanel, "CAMERA", rightRows, mouseNdcX, mouseNdcY);
        }

        drawTopBottomBars(
            backendUsed,
            mode,
            samples,
            previewSpp,
            refineSpp,
            maxDepth,
            batchMs,
            backendStatus,
            leftPanelVisible,
            rightPanelVisible);

        drawPanelVisibilityButtons(
            uiLayout,
            leftPanelVisible,
            rightPanelVisible,
            mouseNdcX,
            mouseNdcY);

        glDisable(GL_BLEND);

        glfwSwapBuffers(window);

        const auto nowClock = std::chrono::steady_clock::now();
        if (nowClock - lastTitleUpdate > std::chrono::milliseconds(160)) {
            lastTitleUpdate = nowClock;

            std::string title = "Raycism Engine | " + backendUsed +
                                " | mode=" + mode +
                                " | samples=" + std::to_string(samples) +
                                " | p/r spp=" + std::to_string(previewSpp) + "/" + std::to_string(refineSpp) +
                                " | depth=" + std::to_string(maxDepth) +
                                " | batch=" + std::to_string(static_cast<int>(batchMs)) + " ms";

            if (!backendStatus.empty()) {
                title += " | " + backendStatus;
            }

            glfwSetWindowTitle(window, title.c_str());
        }
    }

    {
        std::lock_guard<std::mutex> lock(shared.mutex);
        shared.stop = true;
    }

    if (worker.joinable()) {
        worker.join();
    }

    if (texture != 0) {
        glDeleteTextures(1, &texture);
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

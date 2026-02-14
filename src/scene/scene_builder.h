#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/random.h"
#include "material/material.h"
#include "render/camera.h"
#include "render/renderer.h"
#include "scene/bvh.h"
#include "scene/sphere.h"
#include "scene/triangle.h"

namespace orion {

struct ShowcaseSceneConfig {
    bool enableGround = true;
    bool enablePrimaryLight = true;
    bool enableFillLight = true;
    bool enableRandomField = true;
    bool enableHeroSpheres = true;
    bool enableAccentTriangle = true;

    bool enableCenterCube = false;
    bool enableCubeRing = false;
    int cubeRingCount = 6;
    double cubeScale = 1.0;
};

struct ScenePackage {
    std::shared_ptr<Hittable> world;
    Camera camera;
    std::string outputPath;

    ScenePackage(std::shared_ptr<Hittable> worldIn, Camera cameraIn, std::string outputPathIn)
        : world(std::move(worldIn)), camera(std::move(cameraIn)), outputPath(std::move(outputPathIn)) {}
};

inline constexpr const char* kDefaultDemoName = "scene_editor";

struct CameraOverride {
    bool enabled = false;
    Vec3 position{6.8, 4.2, 9.5};
    double yawDeg = -35.57;
    double pitchDeg = -15.32;
    double fovDeg = 34.0;
};

struct SceneEditorPrimitive {
    // Text-first scene representation used by CLI/web scene editor flows.
    std::string type = "sphere";
    std::string name = "Sphere_01";
    Vec3 position{0.0, 1.0, 0.0};
    Vec3 rotationDeg{0.0, 0.0, 0.0};
    Vec3 scale{1.0, 1.0, 1.0};
    std::string material = "lambertian";
    Color color{0.85, 0.85, 0.85};
};

struct SceneEditorMeshTriangle {
    Point3 a;
    Point3 b;
    Point3 c;
    bool hasUv = false;
    Vec3 uvA{0.0, 0.0, 0.0};
    Vec3 uvB{0.0, 0.0, 0.0};
    Vec3 uvC{0.0, 0.0, 0.0};
    std::shared_ptr<const ImageTexture> albedoTexture;
    std::string materialType = "coated";
    Color color{0.82, 0.82, 0.82};
    Color emission{0.0, 0.0, 0.0};
    double fuzz = 0.04;
    double ior = 1.5;
    double coatStrength = 0.45;
    double roughness = 0.12;
};

[[nodiscard]] inline std::string normalizeToken(std::string token) {
    for (char& ch : token) {
        if (ch == '-' || ch == ' ') {
            ch = '_';
        } else {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
    }
    return token;
}

[[nodiscard]] inline std::vector<std::string> splitString(const std::string& value, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream ss(value);
    std::string part;
    while (std::getline(ss, part, delimiter)) {
        parts.push_back(part);
    }
    return parts;
}

[[nodiscard]] inline bool tryParseDouble(const std::string& value, double& out) {
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

[[nodiscard]] inline Vec3 parseVec3Csv(const std::string& value, const Vec3& fallback) {
    const std::vector<std::string> parts = splitString(value, ',');
    if (parts.size() != 3) {
        return fallback;
    }

    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!tryParseDouble(parts[0], x) || !tryParseDouble(parts[1], y) || !tryParseDouble(parts[2], z)) {
        return fallback;
    }
    return Vec3(x, y, z);
}

[[nodiscard]] inline std::vector<SceneEditorPrimitive> defaultSceneEditorPrimitives() {
    std::vector<SceneEditorPrimitive> objects;

    SceneEditorPrimitive plane;
    plane.type = "plane";
    plane.name = "Plane_01";
    plane.position = Vec3(0.0, 0.0, 0.0);
    plane.scale = Vec3(12.0, 1.0, 12.0);
    plane.material = "lambertian";
    plane.color = Color(0.42, 0.42, 0.42);
    objects.push_back(plane);

    SceneEditorPrimitive sphere;
    sphere.type = "sphere";
    sphere.name = "Sphere_01";
    sphere.position = Vec3(0.0, 1.0, -1.8);
    sphere.scale = Vec3(1.0, 1.0, 1.0);
    sphere.material = "coated";
    sphere.color = Color(0.86, 0.86, 0.86);
    objects.push_back(sphere);

    SceneEditorPrimitive light;
    light.type = "light";
    light.name = "Light_01";
    light.position = Vec3(1.9, 3.2, 1.0);
    light.scale = Vec3(0.5, 0.5, 0.5);
    light.material = "emissive";
    light.color = Color(1.0, 1.0, 1.0);
    objects.push_back(light);

    return objects;
}

[[nodiscard]] inline std::vector<SceneEditorPrimitive> parseSceneEditorSpec(const std::string& rawSpec) {
    if (rawSpec.empty()) {
        return defaultSceneEditorPrimitives();
    }

    // Format: type|name|pos(x,y,z)|rot(x,y,z)|scale(x,y,z)|material|color(r,g,b);...
    std::vector<SceneEditorPrimitive> objects;
    const std::vector<std::string> objectTokens = splitString(rawSpec, ';');
    objects.reserve(objectTokens.size());

    for (std::size_t i = 0; i < objectTokens.size(); ++i) {
        const std::string& token = objectTokens[i];
        if (token.empty()) {
            continue;
        }
        const std::vector<std::string> fields = splitString(token, '|');
        if (fields.size() < 7) {
            continue;
        }

        SceneEditorPrimitive primitive;
        primitive.type = normalizeToken(fields[0]);
        if (primitive.type != "sphere" && primitive.type != "cube" && primitive.type != "plane" && primitive.type != "light") {
            continue;
        }

        primitive.name = fields[1].empty() ? (primitive.type + "_" + std::to_string(i + 1)) : fields[1];
        primitive.position = parseVec3Csv(fields[2], Vec3(0.0, 1.0, 0.0));
        primitive.rotationDeg = parseVec3Csv(fields[3], Vec3(0.0, 0.0, 0.0));
        primitive.scale = parseVec3Csv(fields[4], Vec3(1.0, 1.0, 1.0));
        primitive.scale = Vec3(
            std::clamp(std::fabs(primitive.scale.x()), 0.05, 100.0),
            std::clamp(std::fabs(primitive.scale.y()), 0.05, 100.0),
            std::clamp(std::fabs(primitive.scale.z()), 0.05, 100.0));
        primitive.material = normalizeToken(fields[5]);
        if (primitive.material.empty()) {
            primitive.material = (primitive.type == "light") ? "emissive" : "lambertian";
        }
        primitive.color = clampVec(parseVec3Csv(fields[6], Color(0.85, 0.85, 0.85)), 0.0, 1.0);

        objects.push_back(primitive);
        if (objects.size() >= 256) {
            break;
        }
    }

    if (objects.empty()) {
        return defaultSceneEditorPrimitives();
    }
    return objects;
}

[[nodiscard]] inline std::string trimAscii(const std::string& text) {
    std::size_t start = 0;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start])) != 0) {
        ++start;
    }
    std::size_t end = text.size();
    while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }
    return text.substr(start, end - start);
}

[[nodiscard]] inline bool startsWithToken(const std::string& line, const char token) {
    if (line.size() < 2 || line[0] != token) {
        return false;
    }
    const char sep = line[1];
    return std::isspace(static_cast<unsigned char>(sep)) != 0;
}

[[nodiscard]] inline bool parseObjIndexToken(
    const std::string& token,
    std::size_t valueCount,
    std::size_t& outIndex) {
    if (token.empty() || valueCount == 0) {
        return false;
    }

    long long rawIndex = 0;
    try {
        std::size_t parsedChars = 0;
        rawIndex = std::stoll(token, &parsedChars, 10);
        if (parsedChars != token.size()) {
            return false;
        }
    } catch (...) {
        return false;
    }

    long long resolved = -1;
    if (rawIndex > 0) {
        resolved = rawIndex - 1;
    } else if (rawIndex < 0) {
        resolved = static_cast<long long>(valueCount) + rawIndex;
    } else {
        return false;
    }

    if (resolved < 0 || resolved >= static_cast<long long>(valueCount)) {
        return false;
    }

    outIndex = static_cast<std::size_t>(resolved);
    return true;
}

struct ObjFaceVertexRef {
    std::size_t vertexIndex = 0;
    int texcoordIndex = -1;
};

[[nodiscard]] inline bool parseObjFaceVertexToken(
    const std::string& token,
    std::size_t vertexCount,
    std::size_t texcoordCount,
    ObjFaceVertexRef& out) {
    if (token.empty()) {
        return false;
    }

    const std::size_t slash0 = token.find('/');
    const std::string vertexToken = slash0 == std::string::npos ? token : token.substr(0, slash0);
    if (!parseObjIndexToken(vertexToken, vertexCount, out.vertexIndex)) {
        return false;
    }

    out.texcoordIndex = -1;
    if (slash0 == std::string::npos || texcoordCount == 0) {
        return true;
    }

    const std::size_t slash1 = token.find('/', slash0 + 1);
    const std::string texcoordToken = slash1 == std::string::npos
        ? token.substr(slash0 + 1)
        : token.substr(slash0 + 1, slash1 - slash0 - 1);
    if (texcoordToken.empty()) {
        return true;
    }

    std::size_t texcoordIndex = 0;
    if (!parseObjIndexToken(texcoordToken, texcoordCount, texcoordIndex)) {
        return true;
    }
    out.texcoordIndex = static_cast<int>(texcoordIndex);
    return true;
}

[[nodiscard]] inline std::string parseObjMapPathToken(const std::string& raw) {
    const std::string value = trimAscii(raw);
    if (value.empty()) {
        return {};
    }

    const std::size_t firstQuote = value.find('"');
    if (firstQuote != std::string::npos) {
        const std::size_t lastQuote = value.find_last_of('"');
        if (lastQuote != std::string::npos && lastQuote > firstQuote) {
            return trimAscii(value.substr(firstQuote + 1, lastQuote - firstQuote - 1));
        }
    }

    std::istringstream row(value);
    std::string token;
    std::string lastToken;
    while (row >> token) {
        lastToken = token;
    }
    return trimAscii(lastToken);
}

struct ObjMtlMaterialSpec {
    Color kd{0.82, 0.82, 0.82};
    Color ks{0.0, 0.0, 0.0};
    Color ke{0.0, 0.0, 0.0};
    double ns = 32.0;
    double opacity = 1.0;
    double ior = 1.5;
    int illum = 2;
    std::string mapKd;
    std::string mapKe;
};

using ObjTextureImage = ImageTexture;

[[nodiscard]] inline bool readPpmToken(std::istream& in, std::string& token) {
    token.clear();
    char ch = 0;
    while (in.get(ch)) {
        if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
            if (!token.empty()) {
                return true;
            }
            continue;
        }
        if (ch == '#') {
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            if (!token.empty()) {
                return true;
            }
            continue;
        }
        token.push_back(ch);
    }
    return !token.empty();
}

[[nodiscard]] inline ObjTextureImage loadPpmTexture(const std::filesystem::path& texturePath) {
    std::ifstream in(texturePath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open texture: " + texturePath.string());
    }

    std::string magic;
    if (!readPpmToken(in, magic) || (magic != "P6" && magic != "P3")) {
        throw std::runtime_error("Unsupported texture format (expected PPM P6/P3): " + texturePath.string());
    }

    std::string token;
    if (!readPpmToken(in, token)) {
        throw std::runtime_error("Invalid texture width: " + texturePath.string());
    }
    const int width = std::max(0, std::stoi(token));

    if (!readPpmToken(in, token)) {
        throw std::runtime_error("Invalid texture height: " + texturePath.string());
    }
    const int height = std::max(0, std::stoi(token));

    if (!readPpmToken(in, token)) {
        throw std::runtime_error("Invalid texture max value: " + texturePath.string());
    }
    const int maxValue = std::stoi(token);

    if (width <= 0 || height <= 0 || maxValue <= 0) {
        throw std::runtime_error("Invalid texture dimensions: " + texturePath.string());
    }

    ObjTextureImage image;
    image.width = width;
    image.height = height;
    image.texels.resize(static_cast<std::size_t>(width * height));

    const double invMax = 1.0 / static_cast<double>(maxValue);
    const std::size_t pixelCount = static_cast<std::size_t>(width * height);
    if (magic == "P6") {
        in >> std::ws;
        std::vector<unsigned char> bytes(pixelCount * 3);
        in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (in.gcount() != static_cast<std::streamsize>(bytes.size())) {
            throw std::runtime_error("Texture data truncated: " + texturePath.string());
        }
        for (std::size_t i = 0; i < pixelCount; ++i) {
            image.texels[i] = Color(
                static_cast<double>(bytes[i * 3 + 0]) * invMax,
                static_cast<double>(bytes[i * 3 + 1]) * invMax,
                static_cast<double>(bytes[i * 3 + 2]) * invMax);
        }
        return image;
    }

    for (std::size_t i = 0; i < pixelCount; ++i) {
        std::string rToken;
        std::string gToken;
        std::string bToken;
        if (!readPpmToken(in, rToken) || !readPpmToken(in, gToken) || !readPpmToken(in, bToken)) {
            throw std::runtime_error("Texture data truncated: " + texturePath.string());
        }
        image.texels[i] = Color(
            static_cast<double>(std::stoi(rToken)) * invMax,
            static_cast<double>(std::stoi(gToken)) * invMax,
            static_cast<double>(std::stoi(bToken)) * invMax);
    }
    return image;
}

inline void loadObjMtlLibrary(
    const std::filesystem::path& mtlPath,
    std::unordered_map<std::string, ObjMtlMaterialSpec>& outMaterials) {
    std::ifstream in(mtlPath);
    if (!in) {
        return;
    }

    ObjMtlMaterialSpec* current = nullptr;
    std::string line;
    while (std::getline(in, line)) {
        const std::string trimmed = trimAscii(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        std::istringstream row(trimmed);
        std::string rawKey;
        row >> rawKey;
        if (rawKey.empty()) {
            continue;
        }

        const std::string key = normalizeToken(rawKey);
        if (key == "newmtl") {
            std::string name;
            row >> name;
            if (name.empty()) {
                current = nullptr;
                continue;
            }
            current = &outMaterials[name];
            continue;
        }

        if (current == nullptr) {
            continue;
        }

        if (key == "kd") {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            if (row >> x >> y >> z) {
                current->kd = Color(x, y, z);
            }
            continue;
        }
        if (key == "ks") {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            if (row >> x >> y >> z) {
                current->ks = Color(x, y, z);
            }
            continue;
        }
        if (key == "ke") {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            if (row >> x >> y >> z) {
                current->ke = Color(x, y, z);
            }
            continue;
        }
        if (key == "ns") {
            double ns = 32.0;
            if (row >> ns) {
                current->ns = ns;
            }
            continue;
        }
        if (key == "d") {
            double opacity = 1.0;
            if (row >> opacity) {
                current->opacity = std::clamp(opacity, 0.0, 1.0);
            }
            continue;
        }
        if (key == "tr") {
            double tr = 0.0;
            if (row >> tr) {
                current->opacity = std::clamp(1.0 - tr, 0.0, 1.0);
            }
            continue;
        }
        if (key == "ni") {
            double ior = 1.5;
            if (row >> ior) {
                current->ior = std::clamp(ior, 1.0, 3.0);
            }
            continue;
        }
        if (key == "illum") {
            int illum = 2;
            if (row >> illum) {
                current->illum = illum;
            }
            continue;
        }

        const std::string rest = trimAscii(trimmed.substr(rawKey.size()));
        if (key == "map_kd") {
            current->mapKd = parseObjMapPathToken(rest);
            continue;
        }
        if (key == "map_ke") {
            current->mapKe = parseObjMapPathToken(rest);
            continue;
        }
    }
}

[[nodiscard]] inline std::string sceneEditorMeshMaterialKey(const SceneEditorMeshTriangle& tri) {
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss.precision(4);
    ss << tri.materialType << "|"
       << tri.color.x() << "," << tri.color.y() << "," << tri.color.z() << "|"
       << tri.emission.x() << "," << tri.emission.y() << "," << tri.emission.z() << "|"
       << tri.fuzz << "|" << tri.ior << "|" << tri.coatStrength << "|" << tri.roughness << "|"
       << reinterpret_cast<std::uintptr_t>(tri.albedoTexture.get());
    return ss.str();
}

[[nodiscard]] inline std::vector<SceneEditorMeshTriangle> loadSceneEditorObjTriangles(const std::string& rawObjPath) {
    const std::string objPathText = trimAscii(rawObjPath);
    if (objPathText.empty()) {
        return {};
    }

    const std::filesystem::path objPath(objPathText);
    std::error_code fileEc;
    if (!std::filesystem::exists(objPath, fileEc) || !std::filesystem::is_regular_file(objPath, fileEc)) {
        throw std::runtime_error("OBJ file not found: " + objPath.string());
    }

    constexpr std::uintmax_t kMaxObjBytes = 48u * 1024u * 1024u;
    const std::uintmax_t fileSize = std::filesystem::file_size(objPath, fileEc);
    if (!fileEc && fileSize > kMaxObjBytes) {
        throw std::runtime_error("OBJ file is too large for interactive rendering (max 48 MB).");
    }

    std::ifstream in(objPath);
    if (!in) {
        throw std::runtime_error("Failed to open OBJ file: " + objPath.string());
    }

    struct PendingMeshTriangle {
        std::array<std::size_t, 3> vertex{};
        std::array<int, 3> texcoord{};
        std::string materialName;
    };

    std::vector<Point3> vertices;
    vertices.reserve(8192);
    std::vector<Vec3> texcoords;
    texcoords.reserve(8192);
    std::vector<PendingMeshTriangle> pendingTriangles;
    pendingTriangles.reserve(16384);
    std::unordered_map<std::string, ObjMtlMaterialSpec> mtlMaterials;
    std::unordered_map<std::string, std::shared_ptr<ImageTexture>> textureCache;

    const std::filesystem::path objDir = objPath.parent_path();
    std::string activeMaterialName;

    std::string line;
    std::size_t lineNo = 0;
    constexpr std::size_t kMaxTriangles = 400000;
    // Parse OBJ + optional MTL metadata into an intermediate triangle list.
    while (std::getline(in, line)) {
        ++lineNo;
        const std::string trimmed = trimAscii(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        if (startsWithToken(trimmed, 'v')) {
            std::istringstream row(trimmed.substr(1));
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            row >> x >> y >> z;
            if (row.fail()) {
                throw std::runtime_error("Invalid OBJ vertex at line " + std::to_string(lineNo));
            }
            vertices.emplace_back(x, y, z);
            continue;
        }

        if (trimmed.size() > 2
            && trimmed[0] == 'v'
            && trimmed[1] == 't'
            && std::isspace(static_cast<unsigned char>(trimmed[2])) != 0) {
            std::istringstream row(trimmed.substr(2));
            double u = 0.0;
            double v = 0.0;
            if (!(row >> u >> v)) {
                continue;
            }
            texcoords.emplace_back(u, v, 0.0);
            continue;
        }

        if (trimmed.rfind("usemtl ", 0) == 0) {
            activeMaterialName = trimAscii(trimmed.substr(7));
            continue;
        }

        if (trimmed.rfind("mtllib ", 0) == 0) {
            std::istringstream libs(trimmed.substr(7));
            std::string token;
            while (libs >> token) {
                if (token.empty()) {
                    continue;
                }
                const std::filesystem::path mtlPath = (objDir / token).lexically_normal();
                loadObjMtlLibrary(mtlPath, mtlMaterials);
            }
            continue;
        }

        if (startsWithToken(trimmed, 'f')) {
            std::istringstream row(trimmed.substr(1));
            std::vector<ObjFaceVertexRef> polygon;
            polygon.reserve(8);

            std::string token;
            while (row >> token) {
                ObjFaceVertexRef ref;
                if (parseObjFaceVertexToken(token, vertices.size(), texcoords.size(), ref)) {
                    polygon.push_back(ref);
                }
            }
            if (polygon.size() < 3) {
                continue;
            }

            for (std::size_t i = 1; i + 1 < polygon.size(); ++i) {
                pendingTriangles.push_back(PendingMeshTriangle{
                    {polygon[0].vertexIndex, polygon[i].vertexIndex, polygon[i + 1].vertexIndex},
                    {polygon[0].texcoordIndex, polygon[i].texcoordIndex, polygon[i + 1].texcoordIndex},
                    activeMaterialName});
                if (pendingTriangles.size() >= kMaxTriangles) {
                    break;
                }
            }
            if (pendingTriangles.size() >= kMaxTriangles) {
                break;
            }
        }
    }

    if (vertices.empty() || pendingTriangles.empty()) {
        throw std::runtime_error("OBJ file has no valid triangle geometry: " + objPath.string());
    }

    // Normalize imported geometry into editor-friendly scale/placement.
    Point3 bboxMin(
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity());
    Point3 bboxMax(
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity());

    for (const Point3& v : vertices) {
        bboxMin = Point3(
            std::min(bboxMin.x(), v.x()),
            std::min(bboxMin.y(), v.y()),
            std::min(bboxMin.z(), v.z()));
        bboxMax = Point3(
            std::max(bboxMax.x(), v.x()),
            std::max(bboxMax.y(), v.y()),
            std::max(bboxMax.z(), v.z()));
    }

    const Vec3 extent = bboxMax - bboxMin;
    const double maxExtent = std::max(std::max(extent.x(), extent.y()), extent.z());
    const double scale = maxExtent > 1.0e-6 ? (3.2 / maxExtent) : 1.0;
    const Point3 center = (bboxMin + bboxMax) * 0.5;
    const double minYNormalized = (bboxMin.y() - center.y()) * scale;
    const Vec3 offset(0.0, 0.02 - minYNormalized, -1.8);

    auto resolveTexture = [&](const std::string& rawPath) -> std::shared_ptr<ImageTexture> {
        const std::string pathText = trimAscii(rawPath);
        if (pathText.empty()) {
            return nullptr;
        }

        std::filesystem::path texturePath(pathText);
        if (!texturePath.is_absolute()) {
            texturePath = (objDir / texturePath).lexically_normal();
        }
        const std::string cacheKey = texturePath.string();
        auto found = textureCache.find(cacheKey);
        if (found == textureCache.end()) {
            const std::string ext = normalizeToken(texturePath.extension().string());
            if (ext != ".ppm") {
                return nullptr;
            }
            try {
                found = textureCache.emplace(
                    cacheKey,
                    std::make_shared<ImageTexture>(loadPpmTexture(texturePath))).first;
            } catch (...) {
                return nullptr;
            }
        }
        return found->second;
    };

    std::vector<SceneEditorMeshTriangle> triangles;
    triangles.reserve(pendingTriangles.size());
    for (const PendingMeshTriangle& tri : pendingTriangles) {
        const Point3 v0 = (vertices[tri.vertex[0]] - center) * scale + offset;
        const Point3 v1 = (vertices[tri.vertex[1]] - center) * scale + offset;
        const Point3 v2 = (vertices[tri.vertex[2]] - center) * scale + offset;
        const Vec3 normal = cross(v1 - v0, v2 - v0);
        if (normal.lengthSquared() <= 1.0e-16) {
            continue;
        }

        SceneEditorMeshTriangle meshTriangle{};
        meshTriangle.a = v0;
        meshTriangle.b = v1;
        meshTriangle.c = v2;

        auto mtlIt = mtlMaterials.find(tri.materialName);
        if (mtlIt == mtlMaterials.end()) {
            meshTriangle.materialType = "coated";
            meshTriangle.color = Color(0.82, 0.82, 0.82);
            meshTriangle.coatStrength = 0.45;
            meshTriangle.roughness = 0.12;
            triangles.push_back(meshTriangle);
            continue;
        }

        const ObjMtlMaterialSpec& mtl = mtlIt->second;
        const Color baseColor = clampVec(mtl.kd, 0.0, 1.0);
        bool hasUv = tri.texcoord[0] >= 0 && tri.texcoord[1] >= 0 && tri.texcoord[2] >= 0
            && static_cast<std::size_t>(tri.texcoord[0]) < texcoords.size()
            && static_cast<std::size_t>(tri.texcoord[1]) < texcoords.size()
            && static_cast<std::size_t>(tri.texcoord[2]) < texcoords.size();
        if (hasUv) {
            const Vec3 uv0 = texcoords[static_cast<std::size_t>(tri.texcoord[0])];
            const Vec3 uv1 = texcoords[static_cast<std::size_t>(tri.texcoord[1])];
            const Vec3 uv2 = texcoords[static_cast<std::size_t>(tri.texcoord[2])];
            meshTriangle.hasUv = true;
            meshTriangle.uvA = uv0;
            meshTriangle.uvB = uv1;
            meshTriangle.uvC = uv2;
        }

        Color emission = clampVec(mtl.ke, 0.0, 50.0);

        const double maxEmission = std::max(std::max(emission.x(), emission.y()), emission.z());
        const double specular = std::max(std::max(mtl.ks.x(), mtl.ks.y()), mtl.ks.z());
        const double roughnessFromNs = std::clamp(
            1.0 - std::sqrt(std::clamp(mtl.ns, 0.0, 2000.0) / 2000.0),
            0.02,
            1.0);

        meshTriangle.color = baseColor;
        meshTriangle.materialType = "lambertian";
        meshTriangle.fuzz = roughnessFromNs;
        meshTriangle.ior = std::clamp(mtl.ior, 1.01, 2.5);
        meshTriangle.coatStrength = std::clamp(0.20 + specular * 0.75, 0.10, 0.95);
        meshTriangle.roughness = roughnessFromNs;
        if (meshTriangle.hasUv && !trimAscii(mtl.mapKd).empty()) {
            meshTriangle.albedoTexture = resolveTexture(mtl.mapKd);
        }

        // Heuristic material inference from common MTL fields.
        if (maxEmission > 1.0e-4) {
            meshTriangle.materialType = "emissive";
            meshTriangle.emission = clampVec(12.0 * emission, 0.0, 80.0);
            meshTriangle.color = Color(1.0, 1.0, 1.0);
        } else if (mtl.illum >= 4 || mtl.opacity < 0.7) {
            meshTriangle.materialType = "dielectric";
        } else if (specular > 0.35 && mtl.illum >= 2) {
            meshTriangle.materialType = "metal";
            meshTriangle.color = clampVec(
                0.7 * baseColor + 0.3 * clampVec(mtl.ks, 0.0, 1.0),
                0.0,
                1.0);
        } else if (specular > 0.08 && mtl.illum >= 2) {
            meshTriangle.materialType = "coated";
        }

        triangles.push_back(meshTriangle);
    }

    if (triangles.empty()) {
        throw std::runtime_error("OBJ triangles are degenerate after import: " + objPath.string());
    }

    return triangles;
}

[[nodiscard]] inline CameraSettings makeShowcaseCameraSettings(double aspectRatio) {
    CameraSettings cameraSettings;
    cameraSettings.aspectRatio = aspectRatio;
    cameraSettings.lookFrom = Point3(13.0, 2.5, 3.0);
    cameraSettings.lookAt = Point3(0.0, 0.8, 0.0);
    cameraSettings.verticalFovDeg = 26.0;
    cameraSettings.aperture = 0.09;
    cameraSettings.focusDistance = 10.5;
    return cameraSettings;
}

[[nodiscard]] inline CameraSettings makeShowcaseCloseupCameraSettings(double aspectRatio) {
    CameraSettings cameraSettings;
    cameraSettings.aspectRatio = aspectRatio;
    cameraSettings.lookFrom = Point3(6.4, 1.7, 1.9);
    cameraSettings.lookAt = Point3(0.2, 0.9, 0.0);
    cameraSettings.verticalFovDeg = 32.0;
    cameraSettings.aperture = 0.04;
    cameraSettings.focusDistance = 6.8;
    return cameraSettings;
}

[[nodiscard]] inline CameraSettings makeShowcaseWideCameraSettings(double aspectRatio) {
    CameraSettings cameraSettings;
    cameraSettings.aspectRatio = aspectRatio;
    cameraSettings.lookFrom = Point3(18.0, 6.0, 9.0);
    cameraSettings.lookAt = Point3(0.0, 0.8, 0.0);
    cameraSettings.verticalFovDeg = 22.0;
    cameraSettings.aperture = 0.03;
    cameraSettings.focusDistance = 19.5;
    return cameraSettings;
}

[[nodiscard]] inline CameraSettings makeShowcaseTopdownCameraSettings(double aspectRatio) {
    CameraSettings cameraSettings;
    cameraSettings.aspectRatio = aspectRatio;
    cameraSettings.lookFrom = Point3(0.01, 12.5, 0.1);
    cameraSettings.lookAt = Point3(0.0, 0.7, 0.0);
    cameraSettings.vUp = Vec3(0.0, 0.0, -1.0);
    cameraSettings.verticalFovDeg = 34.0;
    cameraSettings.aperture = 0.0;
    cameraSettings.focusDistance = 12.0;
    return cameraSettings;
}

[[nodiscard]] inline CameraSettings makeCubeRoomCameraSettings(double aspectRatio) {
    CameraSettings cameraSettings;
    cameraSettings.aspectRatio = aspectRatio;
    cameraSettings.lookFrom = Point3(0.0, 1.55, 5.6);
    cameraSettings.lookAt = Point3(0.0, 1.1, -1.1);
    cameraSettings.verticalFovDeg = 36.0;
    cameraSettings.aperture = 0.0;
    cameraSettings.focusDistance = 7.0;
    return cameraSettings;
}

[[nodiscard]] inline CameraSettings makeSceneEditorCameraSettings(double aspectRatio) {
    CameraSettings cameraSettings;
    cameraSettings.aspectRatio = aspectRatio;
    cameraSettings.lookFrom = Point3(6.8, 4.2, 9.5);
    cameraSettings.lookAt = Point3(0.0, 1.0, 0.0);
    cameraSettings.verticalFovDeg = 34.0;
    cameraSettings.aperture = 0.0;
    cameraSettings.focusDistance = 12.2;
    return cameraSettings;
}

inline void applyCameraOverride(CameraSettings& cameraSettings, const CameraOverride* cameraOverride) {
    if (cameraOverride == nullptr || !cameraOverride->enabled) {
        return;
    }

    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    const double yaw = cameraOverride->yawDeg * kDegToRad;
    const double pitch = std::clamp(cameraOverride->pitchDeg, -89.0, 89.0) * kDegToRad;
    const double cp = std::cos(pitch);
    Vec3 forward(
        std::sin(yaw) * cp,
        std::sin(pitch),
        -std::cos(yaw) * cp);

    if (forward.nearZero()) {
        forward = Vec3(0.0, 0.0, -1.0);
    }
    forward = normalize(forward);

    cameraSettings.lookFrom = Point3(
        cameraOverride->position.x(),
        cameraOverride->position.y(),
        cameraOverride->position.z());
    cameraSettings.lookAt = cameraSettings.lookFrom + forward;
    cameraSettings.verticalFovDeg = std::clamp(cameraOverride->fovDeg, 20.0, 100.0);
    cameraSettings.aperture = 0.0;
    cameraSettings.focusDistance = 1.0;
}

[[nodiscard]] inline std::string normalizeDemoName(const std::string& rawDemoName) {
    std::string normalized;
    normalized.reserve(rawDemoName.size());

    for (const char ch : rawDemoName) {
        if (ch == '-' || ch == ' ') {
            normalized.push_back('_');
        } else {
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }

    return normalized;
}

[[nodiscard]] inline bool isSupportedDemoName(const std::string& rawDemoName) {
    const std::string demo = normalizeDemoName(rawDemoName);
    return demo == "scene_editor";
}

[[nodiscard]] inline bool isGpuCompatibleDemoName(const std::string& rawDemoName) {
    const std::string demo = normalizeDemoName(rawDemoName);
    return demo == "scene_editor";
}

[[nodiscard]] inline std::array<const char*, 1> supportedDemoNames() {
    return {
        "scene_editor",
    };
}

[[nodiscard]] inline CameraSettings makeShowcaseCameraSettingsForDemo(
    const std::string& rawDemoName,
    double aspectRatio) {
    const std::string demo = normalizeDemoName(rawDemoName);
    if (demo == "showcase_closeup") {
        return makeShowcaseCloseupCameraSettings(aspectRatio);
    }
    if (demo == "showcase_wide") {
        return makeShowcaseWideCameraSettings(aspectRatio);
    }
    if (demo == "showcase_topdown") {
        return makeShowcaseTopdownCameraSettings(aspectRatio);
    }
    return makeShowcaseCameraSettings(aspectRatio);
}

inline void addCubeAsTriangles(
    HittableList& world,
    const Point3& center,
    const Vec3& halfExtent,
    const std::shared_ptr<Material>& material) {
    const Point3 p000(center.x() - halfExtent.x(), center.y() - halfExtent.y(), center.z() - halfExtent.z());
    const Point3 p001(center.x() - halfExtent.x(), center.y() - halfExtent.y(), center.z() + halfExtent.z());
    const Point3 p010(center.x() - halfExtent.x(), center.y() + halfExtent.y(), center.z() - halfExtent.z());
    const Point3 p011(center.x() - halfExtent.x(), center.y() + halfExtent.y(), center.z() + halfExtent.z());
    const Point3 p100(center.x() + halfExtent.x(), center.y() - halfExtent.y(), center.z() - halfExtent.z());
    const Point3 p101(center.x() + halfExtent.x(), center.y() - halfExtent.y(), center.z() + halfExtent.z());
    const Point3 p110(center.x() + halfExtent.x(), center.y() + halfExtent.y(), center.z() - halfExtent.z());
    const Point3 p111(center.x() + halfExtent.x(), center.y() + halfExtent.y(), center.z() + halfExtent.z());

    // +X
    world.add(std::make_shared<Triangle>(p100, p101, p111, material));
    world.add(std::make_shared<Triangle>(p100, p111, p110, material));

    // -X
    world.add(std::make_shared<Triangle>(p000, p010, p011, material));
    world.add(std::make_shared<Triangle>(p000, p011, p001, material));

    // +Y
    world.add(std::make_shared<Triangle>(p010, p110, p111, material));
    world.add(std::make_shared<Triangle>(p010, p111, p011, material));

    // -Y
    world.add(std::make_shared<Triangle>(p000, p001, p101, material));
    world.add(std::make_shared<Triangle>(p000, p101, p100, material));

    // +Z
    world.add(std::make_shared<Triangle>(p001, p011, p111, material));
    world.add(std::make_shared<Triangle>(p001, p111, p101, material));

    // -Z
    world.add(std::make_shared<Triangle>(p000, p100, p110, material));
    world.add(std::make_shared<Triangle>(p000, p110, p010, material));
}

inline void addQuadAsTriangles(
    HittableList& world,
    const Point3& a,
    const Point3& b,
    const Point3& c,
    const Point3& d,
    const std::shared_ptr<Material>& material) {
    world.add(std::make_shared<Triangle>(a, b, c, material));
    world.add(std::make_shared<Triangle>(a, c, d, material));
}

[[nodiscard]] inline Vec3 rotateEulerDegreesXYZ(const Vec3& value, const Vec3& eulerDeg) {
    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    const double rx = eulerDeg.x() * kDegToRad;
    const double ry = eulerDeg.y() * kDegToRad;
    const double rz = eulerDeg.z() * kDegToRad;

    Vec3 out = value;

    {
        const double c = std::cos(rx);
        const double s = std::sin(rx);
        out = Vec3(out.x(), out.y() * c - out.z() * s, out.y() * s + out.z() * c);
    }
    {
        const double c = std::cos(ry);
        const double s = std::sin(ry);
        out = Vec3(out.x() * c + out.z() * s, out.y(), -out.x() * s + out.z() * c);
    }
    {
        const double c = std::cos(rz);
        const double s = std::sin(rz);
        out = Vec3(out.x() * c - out.y() * s, out.x() * s + out.y() * c, out.z());
    }

    return out;
}

[[nodiscard]] inline Point3 transformScenePoint(const SceneEditorPrimitive& primitive, const Point3& localPoint) {
    const Vec3 scaled(
        localPoint.x() * primitive.scale.x(),
        localPoint.y() * primitive.scale.y(),
        localPoint.z() * primitive.scale.z());
    const Vec3 rotated = rotateEulerDegreesXYZ(scaled, primitive.rotationDeg);
    return Point3(
        primitive.position.x() + rotated.x(),
        primitive.position.y() + rotated.y(),
        primitive.position.z() + rotated.z());
}

[[nodiscard]] inline std::shared_ptr<Material> makeSceneEditorMaterial(const SceneEditorPrimitive& primitive) {
    const Color color = clampVec(primitive.color, 0.0, 1.0);
    const std::string mat = normalizeToken(primitive.material);
    const bool isLight = primitive.type == "light";

    if (isLight || mat == "emissive" || mat == "light") {
        const double intensity = isLight ? 14.0 : 10.0;
        return std::make_shared<Emissive>(intensity * color);
    }
    if (mat == "metal") {
        return std::make_shared<Metal>(color, 0.04);
    }
    if (mat == "dielectric" || mat == "glass") {
        return std::make_shared<Dielectric>(1.5);
    }
    if (mat == "coated" || mat == "coated_diffuse") {
        return std::make_shared<CoatedDiffuse>(color, 0.55, 0.08);
    }
    return std::make_shared<Lambertian>(color);
}

[[nodiscard]] inline std::shared_ptr<Material> makeSceneEditorMeshMaterial(const SceneEditorMeshTriangle& tri) {
    const std::string type = normalizeToken(tri.materialType);
    const Color color = clampVec(tri.color, 0.0, 1.0);
    SceneMeshMaterial::Params params;
    params.albedo = color;
    params.emission = clampVec(tri.emission, 0.0, 120.0);
    params.fuzz = std::clamp(tri.fuzz, 0.0, 1.0);
    params.ior = std::clamp(tri.ior, 1.01, 2.5);
    params.coatStrength = std::clamp(tri.coatStrength, 0.0, 1.0);
    params.roughness = std::clamp(tri.roughness, 0.0, 1.0);
    params.albedoTexture = tri.albedoTexture;

    if (type == "emissive" || type == "light") {
        params.type = SceneMeshMaterial::Type::Emissive;
    } else if (type == "metal") {
        params.type = SceneMeshMaterial::Type::Metal;
    } else if (type == "dielectric" || type == "glass") {
        params.type = SceneMeshMaterial::Type::Dielectric;
    } else if (type == "coated" || type == "coated_diffuse") {
        params.type = SceneMeshMaterial::Type::CoatedDiffuse;
    } else {
        params.type = SceneMeshMaterial::Type::Lambertian;
    }
    return std::make_shared<SceneMeshMaterial>(std::move(params));
}

inline void addSceneEditorCube(HittableList& world, const SceneEditorPrimitive& primitive) {
    const auto material = makeSceneEditorMaterial(primitive);

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

    std::array<Point3, 8> p;
    for (std::size_t i = 0; i < base.size(); ++i) {
        p[i] = transformScenePoint(primitive, base[i]);
    }

    // +X
    world.add(std::make_shared<Triangle>(p[4], p[5], p[7], material));
    world.add(std::make_shared<Triangle>(p[4], p[7], p[6], material));
    // -X
    world.add(std::make_shared<Triangle>(p[0], p[2], p[3], material));
    world.add(std::make_shared<Triangle>(p[0], p[3], p[1], material));
    // +Y
    world.add(std::make_shared<Triangle>(p[2], p[6], p[7], material));
    world.add(std::make_shared<Triangle>(p[2], p[7], p[3], material));
    // -Y
    world.add(std::make_shared<Triangle>(p[0], p[1], p[5], material));
    world.add(std::make_shared<Triangle>(p[0], p[5], p[4], material));
    // +Z
    world.add(std::make_shared<Triangle>(p[1], p[3], p[7], material));
    world.add(std::make_shared<Triangle>(p[1], p[7], p[5], material));
    // -Z
    world.add(std::make_shared<Triangle>(p[0], p[4], p[6], material));
    world.add(std::make_shared<Triangle>(p[0], p[6], p[2], material));
}

inline void addSceneEditorPlane(HittableList& world, const SceneEditorPrimitive& primitive) {
    const auto material = makeSceneEditorMaterial(primitive);

    const Point3 a = transformScenePoint(primitive, Point3(-0.5, 0.0, -0.5));
    const Point3 b = transformScenePoint(primitive, Point3(0.5, 0.0, -0.5));
    const Point3 c = transformScenePoint(primitive, Point3(0.5, 0.0, 0.5));
    const Point3 d = transformScenePoint(primitive, Point3(-0.5, 0.0, 0.5));
    addQuadAsTriangles(world, a, b, c, d, material);
}

[[nodiscard]] inline std::shared_ptr<Hittable> makeSceneEditorWorldCpu(
    std::uint64_t seed,
    const std::string& rawSceneSpec,
    const std::string& rawObjPath = "") {
    auto worldList = std::make_shared<HittableList>();
    const std::vector<SceneEditorPrimitive> primitives = parseSceneEditorSpec(rawSceneSpec);

    // Build editor primitives first, then merge optional imported mesh triangles.
    bool hasLight = false;
    for (const SceneEditorPrimitive& primitive : primitives) {
        if (primitive.type == "sphere") {
            const auto material = makeSceneEditorMaterial(primitive);
            const double radius = std::max(0.05, (primitive.scale.x() + primitive.scale.y() + primitive.scale.z()) / 6.0);
            worldList->add(std::make_shared<Sphere>(Point3(
                primitive.position.x(),
                primitive.position.y(),
                primitive.position.z()), radius, material));
            continue;
        }
        if (primitive.type == "cube") {
            addSceneEditorCube(*worldList, primitive);
            continue;
        }
        if (primitive.type == "plane") {
            addSceneEditorPlane(*worldList, primitive);
            continue;
        }
        if (primitive.type == "light") {
            hasLight = true;
            const auto material = makeSceneEditorMaterial(primitive);
            const double radius = std::max(0.08, (primitive.scale.x() + primitive.scale.y() + primitive.scale.z()) / 9.0);
            worldList->add(std::make_shared<Sphere>(Point3(
                primitive.position.x(),
                primitive.position.y(),
                primitive.position.z()), radius, material));
        }
    }

    if (!trimAscii(rawObjPath).empty()) {
        const std::vector<SceneEditorMeshTriangle> meshTriangles = loadSceneEditorObjTriangles(rawObjPath);
        std::unordered_map<std::string, std::shared_ptr<Material>> meshMaterialCache;
        meshMaterialCache.reserve(128);
        for (const SceneEditorMeshTriangle& tri : meshTriangles) {
            const std::string materialKey = sceneEditorMeshMaterialKey(tri);
            auto found = meshMaterialCache.find(materialKey);
            if (found == meshMaterialCache.end()) {
                found = meshMaterialCache.emplace(materialKey, makeSceneEditorMeshMaterial(tri)).first;
            }
            if (tri.hasUv) {
                worldList->add(std::make_shared<Triangle>(
                    tri.a,
                    tri.b,
                    tri.c,
                    tri.uvA,
                    tri.uvB,
                    tri.uvC,
                    found->second));
            } else {
                worldList->add(std::make_shared<Triangle>(tri.a, tri.b, tri.c, found->second));
            }
        }
    }

    if (!hasLight) {
        // Guarantee at least one light so an empty dark scene is still visible.
        const auto lightMat = std::make_shared<Emissive>(Color(14.0, 14.0, 14.0));
        worldList->add(std::make_shared<Sphere>(Point3(2.2, 3.4, 1.3), 0.22, lightMat));
    }

    RNG bvhRng(seed ^ 0xBAADF00DULL);
    return std::make_shared<BVHNode>(worldList->objects, bvhRng);
}

[[nodiscard]] inline std::shared_ptr<Hittable> makeShowcaseWorldCpu(
    std::uint64_t seed,
    const ShowcaseSceneConfig& config = ShowcaseSceneConfig{}) {
    RNG rng(seed);

    auto worldList = std::make_shared<HittableList>();

    if (config.enableGround) {
        const auto groundMat = std::make_shared<Lambertian>(Color(0.48, 0.46, 0.43));
        worldList->add(std::make_shared<Sphere>(Point3(0.0, -1000.0, 0.0), 1000.0, groundMat));
    }

    if (config.enablePrimaryLight) {
        const auto warmLight = std::make_shared<Emissive>(Color(8.0, 6.5, 4.5));
        worldList->add(std::make_shared<Sphere>(Point3(0.0, 7.5, 0.0), 1.8, warmLight));
    }

    if (config.enableFillLight) {
        const auto panelLight = std::make_shared<Emissive>(Color(3.0, 4.2, 6.0));
        worldList->add(std::make_shared<Triangle>(
            Point3(-3.0, 6.2, -5.0),
            Point3(3.0, 6.2, -5.0),
            Point3(0.0, 8.8, -5.0),
            panelLight));
    }

    if (config.enableRandomField) {
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

                std::shared_ptr<Material> sphereMaterial;
                if (choose < 0.6) {
                    const Color albedo = rng.randomVec(0.1, 0.95) * rng.randomVec(0.1, 0.95);
                    sphereMaterial = std::make_shared<Lambertian>(albedo);
                } else if (choose < 0.78) {
                    const Color albedo = rng.randomVec(0.55, 1.0);
                    const double fuzz = rng.range(0.0, 0.25);
                    sphereMaterial = std::make_shared<Metal>(albedo, fuzz);
                } else if (choose < 0.92) {
                    sphereMaterial = std::make_shared<CoatedDiffuse>(
                        rng.randomVec(0.2, 0.9),
                        rng.range(0.2, 0.75),
                        rng.range(0.03, 0.35));
                } else {
                    sphereMaterial = std::make_shared<Dielectric>(1.5);
                }

                worldList->add(std::make_shared<Sphere>(center, 0.2, sphereMaterial));
            }
        }
    }

    if (config.enableHeroSpheres) {
        worldList->add(std::make_shared<Sphere>(
            Point3(0.0, 1.0, 0.0),
            1.0,
            std::make_shared<Dielectric>(1.5)));

        worldList->add(std::make_shared<Sphere>(
            Point3(-4.0, 1.0, 0.0),
            1.0,
            std::make_shared<Lambertian>(Color(0.4, 0.2, 0.1))));

        worldList->add(std::make_shared<Sphere>(
            Point3(4.0, 1.0, 0.0),
            1.0,
            std::make_shared<Metal>(Color(0.75, 0.75, 0.72), 0.02)));
    }

    if (config.enableAccentTriangle) {
        worldList->add(std::make_shared<Triangle>(
            Point3(-2.0, 0.01, 2.5),
            Point3(2.0, 0.01, 2.5),
            Point3(0.0, 2.2, 4.5),
            std::make_shared<Lambertian>(Color(0.1, 0.45, 0.55))));
    }

    if (config.enableCenterCube) {
        const double scale = std::clamp(config.cubeScale, 0.2, 3.0);
        const auto centerCubeMat = std::make_shared<CoatedDiffuse>(Color(0.9, 0.9, 0.88), 0.5, 0.08);
        addCubeAsTriangles(*worldList, Point3(0.0, 0.95, -3.2), Vec3(0.95 * scale, 0.95 * scale, 0.95 * scale), centerCubeMat);
    }

    if (config.enableCubeRing) {
        const int count = std::clamp(config.cubeRingCount, 3, 16);
        const double scale = std::clamp(config.cubeScale, 0.2, 3.0);

        for (int i = 0; i < count; ++i) {
            const double t = (2.0 * 3.14159265358979323846 * static_cast<double>(i)) / static_cast<double>(count);
            const Point3 center(5.8 * std::cos(t), 0.65, 5.8 * std::sin(t));
            const auto material = (i % 2 == 0)
                ? std::static_pointer_cast<Material>(std::make_shared<Metal>(Color(0.9, 0.9, 0.9), 0.03))
                : std::static_pointer_cast<Material>(std::make_shared<Lambertian>(Color(0.2, 0.2, 0.2)));
            addCubeAsTriangles(*worldList, center, Vec3(0.55 * scale, 0.55 * scale, 0.55 * scale), material);
        }
    }

    RNG bvhRng(seed ^ 0xA2E5B3F4ULL);
    return std::make_shared<BVHNode>(worldList->objects, bvhRng);
}

[[nodiscard]] inline std::shared_ptr<Hittable> makeCubeRoomWorldCpu(std::uint64_t seed) {
    auto worldList = std::make_shared<HittableList>();

    const double xMin = -3.0;
    const double xMax = 3.0;
    const double yMin = 0.0;
    const double yMax = 3.6;
    const double zBack = -4.3;
    const double zFront = 2.3;

    const auto floorMat = std::make_shared<Lambertian>(Color(0.72, 0.72, 0.72));
    const auto ceilingMat = std::make_shared<Lambertian>(Color(0.64, 0.66, 0.68));
    const auto backWallMat = std::make_shared<Lambertian>(Color(0.62, 0.63, 0.66));
    const auto leftWallMat = std::make_shared<Lambertian>(Color(0.73, 0.27, 0.27));
    const auto rightWallMat = std::make_shared<Lambertian>(Color(0.25, 0.34, 0.72));
    const auto cubeMat = std::make_shared<CoatedDiffuse>(Color(0.9, 0.9, 0.87), 0.55, 0.08);
    const auto keyLightMat = std::make_shared<Emissive>(Color(14.0, 13.5, 12.5));
    const auto fillLightMat = std::make_shared<Emissive>(Color(5.5, 6.2, 7.2));

    // Floor.
    addQuadAsTriangles(
        *worldList,
        Point3(xMin, yMin, zBack),
        Point3(xMax, yMin, zBack),
        Point3(xMax, yMin, zFront),
        Point3(xMin, yMin, zFront),
        floorMat);

    // Ceiling.
    addQuadAsTriangles(
        *worldList,
        Point3(xMin, yMax, zFront),
        Point3(xMax, yMax, zFront),
        Point3(xMax, yMax, zBack),
        Point3(xMin, yMax, zBack),
        ceilingMat);

    // Back wall.
    addQuadAsTriangles(
        *worldList,
        Point3(xMin, yMin, zBack),
        Point3(xMin, yMax, zBack),
        Point3(xMax, yMax, zBack),
        Point3(xMax, yMin, zBack),
        backWallMat);

    // Left wall.
    addQuadAsTriangles(
        *worldList,
        Point3(xMin, yMin, zFront),
        Point3(xMin, yMin, zBack),
        Point3(xMin, yMax, zBack),
        Point3(xMin, yMax, zFront),
        leftWallMat);

    // Right wall.
    addQuadAsTriangles(
        *worldList,
        Point3(xMax, yMin, zBack),
        Point3(xMax, yMin, zFront),
        Point3(xMax, yMax, zFront),
        Point3(xMax, yMax, zBack),
        rightWallMat);

    // Area-like key light and cooler fill light.
    worldList->add(std::make_shared<Sphere>(Point3(0.0, 3.2, -1.2), 0.44, keyLightMat));
    worldList->add(std::make_shared<Sphere>(Point3(1.8, 2.9, -3.0), 0.24, fillLightMat));

    // The requested demo subject: a single cube in a room.
    addCubeAsTriangles(*worldList, Point3(0.0, 0.9, -1.6), Vec3(0.9, 0.9, 0.9), cubeMat);

    RNG bvhRng(seed ^ 0xDEADBEEFCAFEBABEULL);
    return std::make_shared<BVHNode>(worldList->objects, bvhRng);
}

inline ScenePackage makeShowcaseScene(
    const RenderSettings& settings,
    std::uint64_t seed,
    const ShowcaseSceneConfig& config = ShowcaseSceneConfig{},
    const std::string& rawDemoName = "showcase") {
    const auto world = makeShowcaseWorldCpu(seed, config);
    const std::string demo = normalizeDemoName(rawDemoName);
    const CameraSettings cameraSettings = makeShowcaseCameraSettingsForDemo(
        demo,
        static_cast<double>(settings.width) / static_cast<double>(settings.height));

    Camera camera(cameraSettings);
    if (demo == "showcase_closeup") {
        return ScenePackage(world, camera, "out/showcase_closeup.ppm");
    }
    if (demo == "showcase_wide") {
        return ScenePackage(world, camera, "out/showcase_wide.ppm");
    }
    if (demo == "showcase_topdown") {
        return ScenePackage(world, camera, "out/showcase_topdown.ppm");
    }
    return ScenePackage(world, camera, "out/showcase.ppm");
}

inline ScenePackage makeCubeRoomScene(const RenderSettings& settings, std::uint64_t seed) {
    const auto world = makeCubeRoomWorldCpu(seed);
    const CameraSettings cameraSettings = makeCubeRoomCameraSettings(
        static_cast<double>(settings.width) / static_cast<double>(settings.height));

    Camera camera(cameraSettings);
    return ScenePackage(world, camera, "out/cube_room.ppm");
}

inline ScenePackage makeSceneEditorScene(
    const RenderSettings& settings,
    std::uint64_t seed,
    const std::string& sceneSpec,
    const std::string& objPath = "",
    const CameraOverride* cameraOverride = nullptr) {
    const auto world = makeSceneEditorWorldCpu(seed, sceneSpec, objPath);
    CameraSettings cameraSettings = makeSceneEditorCameraSettings(
        static_cast<double>(settings.width) / static_cast<double>(settings.height));
    applyCameraOverride(cameraSettings, cameraOverride);

    Camera camera(cameraSettings);
    return ScenePackage(world, camera, "out/scene_editor.ppm");
}

inline ScenePackage makeHeroSpheresScene(const RenderSettings& settings, std::uint64_t seed) {
    ShowcaseSceneConfig config;
    config.enableGround = true;
    config.enablePrimaryLight = true;
    config.enableFillLight = true;
    config.enableRandomField = false;
    config.enableHeroSpheres = true;
    config.enableAccentTriangle = false;
    config.enableCenterCube = false;
    config.enableCubeRing = false;
    return makeShowcaseScene(settings, seed, config);
}

inline ScenePackage makeCubeRingScene(const RenderSettings& settings, std::uint64_t seed) {
    ShowcaseSceneConfig config;
    config.enableGround = true;
    config.enablePrimaryLight = true;
    config.enableFillLight = true;
    config.enableRandomField = false;
    config.enableHeroSpheres = false;
    config.enableAccentTriangle = false;
    config.enableCenterCube = true;
    config.enableCubeRing = true;
    config.cubeRingCount = 8;
    config.cubeScale = 1.0;
    return makeShowcaseScene(settings, seed, config);
}

inline ScenePackage makeDemoScene(
    const RenderSettings& settings,
    std::uint64_t seed,
    const std::string& rawDemoName,
    const std::string& rawSceneSpec = "",
    const std::string& rawObjPath = "",
    const CameraOverride* cameraOverride = nullptr) {
    // Current production flow is pinned to scene_editor for deterministic tooling behavior.
    (void)rawDemoName;
    return makeSceneEditorScene(settings, seed, rawSceneSpec, rawObjPath, cameraOverride);
}

}  // namespace orion

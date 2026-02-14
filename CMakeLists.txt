cmake_minimum_required(VERSION 3.16)
project(RaycismEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckLanguage)
check_language(CUDA)

set(ORION_HAS_CUDA 0)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(ORION_HAS_CUDA 1)
endif()

find_package(OpenGL REQUIRED)
option(ORION_BUILD_STUDIO "Build the GLFW-based Orion Studio executable." ON)

set(ORION_STUDIO_AVAILABLE OFF)
if (ORION_BUILD_STUDIO)
    find_package(glfw3 QUIET)

    if (TARGET glfw OR TARGET glfw3::glfw OR glfw3_FOUND)
        set(ORION_STUDIO_AVAILABLE ON)
    else()
        find_package(PkgConfig QUIET)
        if (PkgConfig_FOUND)
            pkg_check_modules(GLFW3 QUIET glfw3)
            if (GLFW3_FOUND)
                set(ORION_STUDIO_AVAILABLE ON)
            endif()
        endif()
    endif()

    if (NOT ORION_STUDIO_AVAILABLE)
        message(WARNING
            "GLFW was not found. Skipping `orion_studio` build. "
            "Only `orion_raytracer` will be built. "
            "To enable studio on Windows, install GLFW (for example via vcpkg) "
            "and re-run CMake with your toolchain file."
        )
    endif()
endif()

if (ORION_HAS_CUDA)
    if (CMAKE_VERSION VERSION_LESS 3.24)
        set(ORION_CUDA_ARCH "75")
    else()
        set(ORION_CUDA_ARCH "native")
    endif()
endif()

function(orion_configure_target target_name)
    target_include_directories(${target_name} PRIVATE src)
    target_compile_definitions(${target_name} PRIVATE ORION_HAS_CUDA=${ORION_HAS_CUDA})

    if (ORION_HAS_CUDA)
        target_sources(${target_name} PRIVATE src/render/gpu_renderer.cu)
        set_target_properties(${target_name} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_ARCHITECTURES "${ORION_CUDA_ARCH}")
    endif()

    if (MSVC)
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /O2>)
    else()
        target_compile_options(${target_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -O3>)
        if (ORION_HAS_CUDA)
            target_compile_options(${target_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
        endif()
    endif()
endfunction()

add_executable(orion_raytracer
    src/main.cpp
)
orion_configure_target(orion_raytracer)

if (ORION_STUDIO_AVAILABLE)
    add_executable(orion_studio
        src/studio/studio_main.cpp
    )
    orion_configure_target(orion_studio)

    if (TARGET glfw)
        target_link_libraries(orion_studio PRIVATE OpenGL::GL glfw)
    elseif (TARGET glfw3::glfw)
        target_link_libraries(orion_studio PRIVATE OpenGL::GL glfw3::glfw)
    else()
        target_include_directories(orion_studio PRIVATE ${GLFW3_INCLUDE_DIRS})
        target_link_directories(orion_studio PRIVATE ${GLFW3_LIBRARY_DIRS})
        target_link_libraries(orion_studio PRIVATE OpenGL::GL ${GLFW3_LIBRARIES})
    endif()
endif()

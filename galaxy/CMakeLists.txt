cmake_minimum_required(VERSION 3.18)
project(j-gravity CXX C CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# Janus Galaxy — ImGui + OpenGL + CUDA N-body simulator
# ------------------------------------------------------------------
# Build configuration (all paths are relative to this CMakeLists):
#   -DJANUS_VCPKG_ROOT=<vcpkg>/installed/x64-windows   (required on Windows)
#   -DJANUS_CUDA_ARCH=89                               (or e.g. 75;80;86;89;90)
# See ../BUILDING.md for the full procedure.
# ------------------------------------------------------------------

set(JANUS_VCPKG_ROOT "$ENV{JANUS_VCPKG_ROOT}" CACHE PATH
    "vcpkg installed tree for the x64-windows triplet")
set(JANUS_CUDA_ARCH "89" CACHE STRING
    "CUDA compute capability(ies), e.g. 89 or 75;80;86;89;90")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake")
include(JanusDependencies)
janus_imgui_backends(IMGUI_BACKEND_SOURCES)

set(APP_SOURCES
    src/main.cpp
    src/app.cpp
    src/simulation.cpp
    src/galaxy_generator.cpp
    render/src/point_renderer.cpp
    render/src/volume_renderer.cpp
    render/src/camera.cpp
    render/src/matrix.cpp
    render/src/vector.cpp
    render/src/color.cpp
    ${IMGUI_BACKEND_SOURCES}
    gpu-nbody/pair3.cu
    gpu-nbody/treepm_force.cu
    gpu-nbody/gl_interop.cu
    gpu-nbody/volume_grid.cu
)

set(APP_INCLUDES
    src
    render/includes
    gpu-nbody
    render/src/imgui_backends
)

add_executable(j-gravity WIN32 ${APP_SOURCES})
if(WIN32)
    target_sources(j-gravity PRIVATE app_icon.rc)
endif()

# Keep the Windows display name and use a shell-friendly name on Linux.
if(WIN32)
    set_target_properties(j-gravity PROPERTIES OUTPUT_NAME "Janus Galaxy")
else()
    set_target_properties(j-gravity PROPERTIES OUTPUT_NAME "janus-galaxy")
endif()
set_target_properties(j-gravity PROPERTIES
    CUDA_ARCHITECTURES "${JANUS_CUDA_ARCH}")

target_include_directories(j-gravity PRIVATE ${APP_INCLUDES})

find_package(CUDAToolkit REQUIRED)

if(WIN32)
    if(NOT JANUS_VCPKG_ROOT)
        message(FATAL_ERROR
            "JANUS_VCPKG_ROOT is not set. Pass "
            "-DJANUS_VCPKG_ROOT=<path-to-vcpkg>/installed/x64-windows.")
    endif()
    if(NOT EXISTS "${JANUS_VCPKG_ROOT}/include/SDL2/SDL.h")
        message(FATAL_ERROR
            "No vcpkg SDL2 found under ${JANUS_VCPKG_ROOT}. "
            "Check the x64-windows triplet.")
    endif()

    set(SDL2_INCLUDE_DIR "${JANUS_VCPKG_ROOT}/include/SDL2")
    set(SDL2_LIBRARY "${JANUS_VCPKG_ROOT}/lib/SDL2.lib")
    set(SDL2MAIN_LIBRARY "${JANUS_VCPKG_ROOT}/lib/manual-link/SDL2main.lib")
    set(IMGUI_INCLUDE_DIR "${JANUS_VCPKG_ROOT}/include")
    set(IMGUI_LIBRARY "${JANUS_VCPKG_ROOT}/lib/imgui.lib")
    set(GLEW_INCLUDE_DIR "${JANUS_VCPKG_ROOT}/include")
    set(GLEW_LIBRARY "${JANUS_VCPKG_ROOT}/lib/glew32.lib")
    set(GLM_INCLUDE_DIR "${JANUS_VCPKG_ROOT}/include")

    target_include_directories(j-gravity PRIVATE
        ${SDL2_INCLUDE_DIR}
        ${IMGUI_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIR}
        ${GLM_INCLUDE_DIR}
    )

    target_link_libraries(j-gravity PRIVATE
        ${SDL2_LIBRARY}
        ${SDL2MAIN_LIBRARY}
        ${IMGUI_LIBRARY}
        ${GLEW_LIBRARY}
        opengl32
        comdlg32
        shell32
        CUDA::cudart
    )

    target_compile_options(j-gravity PRIVATE
        $<$<COMPILE_LANGUAGE:CXX,C>:/wd4244 /wd4267 /wd4305 /wd4018 /wd4996 /openmp>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4244,/wd4267,/wd4305,/wd4018,/wd4996,/openmp>
    )
    target_compile_definitions(j-gravity PRIVATE _USE_MATH_DEFINES)

    add_custom_command(TARGET j-gravity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${JANUS_VCPKG_ROOT}/bin/SDL2.dll" "${JANUS_VCPKG_ROOT}/bin/glew32.dll"
            "$<TARGET_FILE_DIR:j-gravity>"
        COMMENT "Copying runtime DLLs (SDL2, GLEW)"
    )
else()
    janus_configure_linux(j-gravity)
endif()

message(STATUS "Janus Galaxy: CUDA architectures = ${JANUS_CUDA_ARCH}")

# Both applications load shaders and equation templates from the working directory.
add_custom_command(TARGET j-gravity POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/render/shaders"
        "$<TARGET_FILE_DIR:j-gravity>/render/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/equations"
        "$<TARGET_FILE_DIR:j-gravity>/equations"
    VERBATIM
    COMMENT "Copying shaders and equation templates"
)

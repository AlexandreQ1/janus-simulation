cmake_minimum_required(VERSION 3.18)
project(j-gravity-universe CXX C CUDA)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# Janus Universe — ImGui + OpenGL + CUDA cosmological N-body simulator
# ------------------------------------------------------------------
# Build configuration (all paths are relative to this CMakeLists):
#   -DJANUS_VCPKG_ROOT=<vcpkg>/installed/x64-windows   (required on Windows)
#   -DJANUS_CUDA_ARCH=89                               (or e.g. 75;80;86;89;90)
#   -DJANUS_WITH_3DCONNEXION=ON -DTDX_SDK=<SDK path>  (optional)
# See ../BUILDING.md for the full procedure.
# ------------------------------------------------------------------

set(JANUS_VCPKG_ROOT "$ENV{JANUS_VCPKG_ROOT}" CACHE PATH
    "vcpkg installed tree for the x64-windows triplet")
set(JANUS_CUDA_ARCH "89" CACHE STRING
    "CUDA compute capability(ies), e.g. 89 or 75;80;86;89;90")
option(JANUS_WITH_3DCONNEXION "Build with 3Dconnexion SpaceMouse support" OFF)
set(TDX_SDK "" CACHE PATH "3Dconnexion SDK root (only with JANUS_WITH_3DCONNEXION=ON)")

set(APP_SOURCES
    src/main.cpp
    src/app.cpp
    src/simulation.cpp
    src/universe_generator.cpp
    render/src/point_renderer.cpp
    render/src/camera.cpp
    render/src/spacemouse_nav.cpp
    render/src/matrix.cpp
    render/src/vector.cpp
    render/src/color.cpp
    render/src/imgui_backends/imgui_impl_sdl2.cpp
    render/src/imgui_backends/imgui_impl_opengl3.cpp
    gpu-nbody/pair3.cu
    gpu-nbody/pm_force.cu
    gpu-nbody/treepm_force.cu
    gpu-nbody/gl_interop.cu
)

set(APP_INCLUDES
    src
    render/includes
    gpu-nbody
    render/src/imgui_backends
)

add_executable(j-gravity-universe WIN32 ${APP_SOURCES} app_icon.rc)

# User-facing executable name; the CMake target keeps its historical name.
set_target_properties(j-gravity-universe PROPERTIES OUTPUT_NAME "Janus Universe")

target_include_directories(j-gravity-universe PRIVATE ${APP_INCLUDES})

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

    target_include_directories(j-gravity-universe PRIVATE
        ${SDL2_INCLUDE_DIR}
        ${IMGUI_INCLUDE_DIR}
        ${GLEW_INCLUDE_DIR}
        ${GLM_INCLUDE_DIR}
    )

    set(JANUS_EXTRA_LIBS)
    if(JANUS_WITH_3DCONNEXION)
        if(NOT TDX_SDK)
            message(FATAL_ERROR
                "JANUS_WITH_3DCONNEXION=ON requires -DTDX_SDK=<3Dconnexion SDK>.")
        endif()
        target_include_directories(j-gravity-universe PRIVATE "${TDX_SDK}/inc")
        list(APPEND JANUS_EXTRA_LIBS "${TDX_SDK}/lib/x64/TDxNavLib.lib")
        target_compile_definitions(j-gravity-universe PRIVATE JANUS_WITH_3DCONNEXION=1)
    endif()

    target_link_libraries(j-gravity-universe PRIVATE
        ${SDL2_LIBRARY}
        ${SDL2MAIN_LIBRARY}
        ${IMGUI_LIBRARY}
        ${GLEW_LIBRARY}
        opengl32
        comdlg32
        shell32
        CUDA::cudart
        CUDA::cufft
        ${JANUS_EXTRA_LIBS}
    )

    target_compile_options(j-gravity-universe PRIVATE
        $<$<COMPILE_LANGUAGE:CXX,C>:/wd4244 /wd4267 /wd4305 /wd4018 /wd4996 /openmp>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4244,/wd4267,/wd4305,/wd4018,/wd4996,/openmp>
    )
    target_compile_definitions(j-gravity-universe PRIVATE _USE_MATH_DEFINES)

    set_target_properties(j-gravity-universe PROPERTIES
        CUDA_ARCHITECTURES "${JANUS_CUDA_ARCH}"
    )

    # Copy the shaders and the two runtime DLLs next to the executable.
    add_custom_command(TARGET j-gravity-universe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:j-gravity-universe>/render/shaders"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/render/shaders"
            "$<TARGET_FILE_DIR:j-gravity-universe>/render/shaders"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${JANUS_VCPKG_ROOT}/bin/SDL2.dll" "${JANUS_VCPKG_ROOT}/bin/glew32.dll"
            "$<TARGET_FILE_DIR:j-gravity-universe>"
        COMMENT "Copying OpenGL shaders and runtime DLLs (SDL2, GLEW)"
    )
else()
    find_package(SDL2 REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(glm REQUIRED)
    find_package(imgui CONFIG REQUIRED)
    target_link_libraries(j-gravity-universe PRIVATE
        SDL2::SDL2
        GLEW::GLEW
        glm::glm
        CUDA::cudart
        imgui::imgui
        GL
    )
    set_target_properties(j-gravity-universe PROPERTIES
        CUDA_ARCHITECTURES "${JANUS_CUDA_ARCH}"
    )
endif()

message(STATUS "Janus Universe: CUDA architectures = ${JANUS_CUDA_ARCH}")
if(JANUS_WITH_3DCONNEXION)
    message(STATUS "Janus Universe: 3Dconnexion support = ON (${TDX_SDK})")
else()
    message(STATUS "Janus Universe: 3Dconnexion support = OFF (stub)")
endif()

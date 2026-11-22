include_guard(GLOBAL)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

function(janus_imgui_source_dir output)
    include(FetchContent)
    FetchContent_Declare(janus_imgui
        URL https://codeload.github.com/ocornut/imgui/tar.gz/refs/tags/v1.92.9b-docking
        URL_HASH SHA256=90ded916bd57db2e0e171b6b098940a47c6f5042725dcdc67fb19940ca8bfdcc
        TLS_VERIFY TRUE)
    FetchContent_MakeAvailable(janus_imgui)
    set(source_dir "${janus_imgui_SOURCE_DIR}")

    if(source_dir)
        foreach(source IN ITEMS imgui.cpp imgui_draw.cpp imgui_tables.cpp imgui_widgets.cpp
                backends/imgui_impl_sdl2.cpp backends/imgui_impl_opengl3.cpp)
            if(NOT EXISTS "${source_dir}/${source}")
                message(FATAL_ERROR "Incomplete Dear ImGui source tree: ${source_dir}/${source} not found.")
            endif()
        endforeach()
    endif()
    set(${output} "${source_dir}" PARENT_SCOPE)
endfunction()

function(janus_imgui_backends output)
    set(backend_dir "${CMAKE_CURRENT_SOURCE_DIR}/render/src/imgui_backends")
    if(NOT WIN32)
        janus_imgui_source_dir(imgui_source_dir)
        set(backend_dir "${imgui_source_dir}/backends")
    endif()
    set(${output}
        "${backend_dir}/imgui_impl_sdl2.cpp"
        "${backend_dir}/imgui_impl_opengl3.cpp" PARENT_SCOPE)
endfunction()

function(janus_configure_linux target)
    find_package(SDL2 REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(glm REQUIRED)
    find_package(OpenGL REQUIRED)
    find_package(Threads REQUIRED)
    find_path(STB_IMAGE_WRITE_INCLUDE_DIR NAMES stb_image_write.h PATH_SUFFIXES stb REQUIRED)
    mark_as_advanced(STB_IMAGE_WRITE_INCLUDE_DIR)

    janus_imgui_source_dir(imgui_source_dir)
    if(NOT TARGET janus_imgui)
        add_library(janus_imgui STATIC
            "${imgui_source_dir}/imgui.cpp"
            "${imgui_source_dir}/imgui_draw.cpp"
            "${imgui_source_dir}/imgui_tables.cpp"
            "${imgui_source_dir}/imgui_widgets.cpp")
        target_include_directories(janus_imgui PUBLIC "${imgui_source_dir}")
        target_compile_features(janus_imgui PUBLIC cxx_std_17)
    endif()
    # The headers must match the selected backend sources.
    target_include_directories(${target} BEFORE PRIVATE "${imgui_source_dir}/backends")

    target_include_directories(${target} PRIVATE "${STB_IMAGE_WRITE_INCLUDE_DIR}")
    target_link_libraries(${target} PRIVATE
        SDL2::SDL2 GLEW::GLEW glm::glm OpenGL::GL Threads::Threads
        CUDA::cudart janus_imgui ${CMAKE_DL_LIBS})
endfunction()

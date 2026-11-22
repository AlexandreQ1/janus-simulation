# Building Janus Galaxy and Janus Universe

Both applications require C++17, CUDA, SDL2 2.0.14+, GLEW, GLM, Dear ImGui
**with docking/multi-viewport support**, and `stb_image_write.h`. Universe also
requires cuFFT. There is no CPU fallback. A working NVIDIA driver and a CUDA/OpenGL
capable NVIDIA GPU are required to run the applications.

Build instructions are provided for Windows x64 and Fedora Linux x86_64.
All external paths are CMake cache variables. See the tested configurations
below for the scope of build validation.

## Linux

Requirements: CMake 3.18+, Ninja (or Make), a CUDA-supported C/C++ compiler,
the CUDA development toolkit (including `nvcc` and cuFFT), SDL2 2.0.14+,
GLEW, GLM, OpenGL development libraries, stb and Dear ImGui with docking support.
A driver package alone does not provide the CUDA compiler.

### Fedora 44: install development packages

```sh
sudo dnf install gcc gcc-c++ cmake ninja-build git pkgconf-pkg-config \
  SDL2-devel glew-devel glm-devel libglvnd-devel stb_image_write-devel
```

Fedora provides the screenshot header as
[`stb_image_write-devel`](https://packages.fedoraproject.org/pkgs/stb/stb_image_write-devel/).
CMake downloads and builds the required Dear ImGui version automatically.

### Fedora 44: install the CUDA toolkit

Enable the NVIDIA CUDA repository for Fedora 44 x86_64 and install the toolkit:

```sh
sudo dnf install dnf5-plugins
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/cuda-fedora44.repo
sudo dnf install cuda-toolkit
```

These commands follow the
[NVIDIA Fedora installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#fedora).
Use the repository matching the Fedora release; the URL above is specific to
Fedora 44. The NVIDIA driver must also be installed and working before running
the applications. Driver installation is covered by the NVIDIA guide.

Check the compiler and driver:

```sh
/usr/local/cuda/bin/nvcc --version
nvidia-smi
```

If `nvidia-smi` fails, resolve the driver problem before running a simulation.
If the system compiler is newer than the installed CUDA toolkit supports,
select a supported compiler using `CMAKE_CUDA_HOST_COMPILER` and
`CMAKE_CXX_COMPILER`; do not bypass CUDA's compiler-version checks.

### Dear ImGui dependency management

On Linux, CMake downloads Dear ImGui automatically during the first configure
step using `FetchContent`. The archive is pinned to release
[`v1.92.9b-docking`](https://github.com/ocornut/imgui/releases/tag/v1.92.9b)
and checked against a SHA-256 hash. The `-docking` tag provides the docking and
multi-viewport features required by the applications.
CMake builds the core and matching SDL2/OpenGL3 backends from the same sources.
No manual clone or dependency-specific project option is required.

Internet access is required for the first configuration. Downloaded sources are
kept under `build/linux/_deps/janus_imgui-src` and reused on subsequent builds.
Deleting the build directory requires downloading them again.

For an offline build, use the standard CMake FetchContent override with an
existing source tree from the same `v1.92.9b-docking` release:

```sh
cmake -S . -B build/linux \
  -DFETCHCONTENT_SOURCE_DIR_JANUS_IMGUI=/absolute/path/to/imgui
```

This selects the local sources without downloading them. stb is detected
automatically, including the system `include/stb` directory. For dependencies
installed in a custom prefix, use the standard `CMAKE_PREFIX_PATH` search path.

On Windows, the build links the Dear ImGui library installed by vcpkg under
`JANUS_VCPKG_ROOT`; it does not use the Linux download path.

### Select the CUDA architecture

Set `JANUS_CUDA_ARCH` to the GPU's compute capability without the decimal point.
For example, an RTX 3050 (GA107) requires `86`, and an RTX 4090 requires `89`.
Check the [NVIDIA GPU capability table](https://developer.nvidia.com/cuda/gpus).
The CMake default is `89`; explicitly select the architecture for the target GPU.

The following commands use `86` as an example. For a binary covering several
architectures, pass a quoted list such as `-DJANUS_CUDA_ARCH="86;89"`.

### Configure, build and run

Run from the repository root:

```sh
cmake -S . -B build/linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DJANUS_CUDA_ARCH=86
cmake --build build/linux --parallel 4
```

The executables are `build/linux/galaxy/janus-galaxy` and
`build/linux/universe/janus-universe`. Launch each from its output directory:

```sh
(cd build/linux/galaxy && ./janus-galaxy)
(cd build/linux/universe && ./janus-universe)
```

Shaders and equation templates are copied beside each executable. Resource paths
are relative to the working directory. The two applications can also be built
separately with `-S galaxy` or `-S universe`. With the top-level project, Windows
output paths also include the application subdirectory, for example
`build/all/galaxy/Release`.

For a CUDA installation outside the default location, replace
`CMAKE_CUDA_COMPILER` and, if necessary, set `CUDAToolkit_ROOT`.

3Dconnexion integration is Windows-only; enabling it on Linux is an error.
Some features use Windows-specific dialogs or processes, including video
recording. These features require separate Linux runtime validation.

### Troubleshooting

**Incomplete compiler detection.** If configuration stopped before CUDA was
installed, CMake may retain incomplete compiler information. The error
`target_compile_features no known features for CXX compiler` can result.
With CMake 3.24+, rerun the full configure command above with `--fresh`:

```sh
cmake --fresh -S . -B build/linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DJANUS_CUDA_ARCH=86
```

With older CMake versions, use a new build directory. Explicit compiler paths
also bypass PATH wrappers such as ccache, which requires a writable cache.

**CUDA kernel architecture mismatch.** The message
`no kernel image is available for execution on the device` means the executable
contains no compatible CUDA kernel for the selected GPU. For example, an
`sm_89` binary cannot run on an RTX 3050 requiring `sm_86`. Reconfigure with the
correct architecture and rebuild:

```sh
cmake -S . -B build/linux -DJANUS_CUDA_ARCH=86
cmake --build build/linux --parallel 4
```

Close the running application before launching the rebuilt executable. CUDA
kernels are compiled from `gpu-nbody/*.cu` into the executable. The
`equations/*.cl` files are equation-editor templates; copying them cannot
resolve an architecture mismatch. Inspect the compiled architectures with:

```sh
/usr/local/cuda/bin/cuobjdump --list-elf build/linux/galaxy/janus-galaxy
/usr/local/cuda/bin/cuobjdump --list-elf build/linux/universe/janus-universe
```

### Tested Linux build configuration

| Component | Configuration |
|---|---|
| OS | Fedora Linux 44 x86_64 |
| C/C++ compiler | GCC 16.2.1 |
| CUDA toolkit | 13.5.37 |
| CMake | 4.3.0, Ninja generator |
| Build type | Release |
| CUDA architecture | `86` |
| Dear ImGui | Release `v1.92.9b-docking` |

Both applications compiled and linked, their runtime libraries resolved with
`ldd`, and shaders and equation templates were copied to the output directories.
The embedded CUDA kernels were checked with `cuobjdump`. This verifies the build;
graphical startup, CUDA/OpenGL interoperability and simulation results require
separate runtime validation.

## Windows

The following instructions describe the previously tested configuration:
Windows 11 x64, Visual Studio 2022 Build Tools, CUDA Toolkit 13.1,
CMake 3.18+ and vcpkg.

### 1. Prerequisites

| Tool | Version / notes |
|---|---|
| Windows | 10 or 11, 64-bit |
| Visual Studio | 2022 (Build Tools are enough), workload **Desktop development with C++**, x64 compiler |
| CUDA Toolkit | 13.1 (the projects were built with 13.1; `nvcc` must be on `PATH`) |
| CMake | 3.18 or newer; a generator for Visual Studio 17 2022 |
| vcpkg | the `x64-windows` triplet with `sdl2`, `glew`, `glm`, `imgui` |
| Git | to clone the repository (optional) |

An NVIDIA GPU and a recent driver are required to run the simulators. The
prebuilt binaries target compute capability 8.9; for another GPU set
`JANUS_CUDA_ARCH` (for example `86`, `89`, `90`, or `75;80;86;89;90`).

### 2. Install the vcpkg dependencies

From a PowerShell prompt, with vcpkg installed (for example in
`C:\dev\vcpkg`):

```powershell
C:\dev\vcpkg\vcpkg.exe install sdl2 glew glm imgui --triplet x64-windows
```

The CMake files expect the **installed tree**, not the vcpkg root:

```
C:/dev/vcpkg/installed/x64-windows
```

Set it once in the environment if you prefer:

```powershell
$env:JANUS_VCPKG_ROOT = "C:/dev/vcpkg/installed/x64-windows"
```

Otherwise pass `-DJANUS_VCPKG_ROOT=...` to every `cmake -S` command.

### 3. Build Janus Galaxy

```powershell
cd <repository>
cmake -S galaxy -B build/galaxy -G "Visual Studio 17 2022" -A x64 `
  -DJANUS_VCPKG_ROOT="C:/dev/vcpkg/installed/x64-windows" `
  -DJANUS_CUDA_ARCH=89

cmake --build build/galaxy --config Release --target j-gravity -- /m:8
```

The executable is written to:

```
build/galaxy/Release/Janus Galaxy.exe
```

The post-build step copies `render/shaders/` and the `SDL2.dll` / `glew32.dll`
runtime files next to the executable. Run it from that folder:

```powershell
cd build/galaxy/Release
.\"Janus Galaxy.exe"
```

`Janus Galaxy.exe` links the CUDA runtime statically, so no `cudart` DLL is
needed. The Microsoft Visual C++ runtime (`MSVCP140.dll`, `VCRUNTIME140.dll`,
`VCRUNTIME140_1.dll`) is required; it is present on most systems and can be
installed with the Microsoft Visual C++ 2015-2022 Redistributable (x64).

### 4. Build Janus Universe

```powershell
cmake -S universe -B build/universe -G "Visual Studio 17 2022" -A x64 `
  -DJANUS_VCPKG_ROOT="C:/dev/vcpkg/installed/x64-windows" `
  -DJANUS_CUDA_ARCH=89

cmake --build build/universe --config Release --target j-gravity-universe -- /m:8
```

The executable is written to:

```
build/universe/Release/Janus Universe.exe
```

Janus Universe uses cuFFT for its Particle-Mesh solver and imports
`cufft64_12.dll` at load time. The CUDA Toolkit's `bin\x64` folder is normally
on `PATH`; for a portable folder, copy `cufft64_12.dll` next to the executable
or install the CUDA Toolkit 13.1 on the target machine.

#### Optional 3Dconnexion SpaceMouse support

The source builds **without** the proprietary 3Dconnexion SDK by default; the
SpaceMouse control is then a stub and the checkbox is hidden. To build with it,
install the 3Dconnexion SDK and configure:

```powershell
cmake -S universe -B build/universe-sm -G "Visual Studio 17 2022" -A x64 `
  -DJANUS_VCPKG_ROOT="C:/dev/vcpkg/installed/x64-windows" `
  -DJANUS_CUDA_ARCH=89 `
  -DJANUS_WITH_3DCONNEXION=ON `
  -DTDX_SDK="C:/path/to/3Dconnexion SDK"

cmake --build build/universe-sm --config Release --target j-gravity-universe -- /m:8
```

The 3Dconnexion SDK licence requires attribution in the application's About
window. The source contains that attribution under the
`JANUS_WITH_3DCONNEXION` compile definition; keep it if you distribute a build
that includes the SDK.

### 5. CMake options

| Option | Default | Meaning |
|---|---|---|
| `JANUS_VCPKG_ROOT` | `$ENV{JANUS_VCPKG_ROOT}` | Path to `<vcpkg>/installed/x64-windows`. Required on Windows. |
| `JANUS_CUDA_ARCH` | `89` | `CUDA_ARCHITECTURES` value(s). Semicolon-separated lists are accepted. |
| `JANUS_WITH_3DCONNEXION` | `OFF` | Universe only. Builds the real NavLib SpaceMouse path instead of the stub. |
| `TDX_SDK` | empty | 3Dconnexion SDK root; required when `JANUS_WITH_3DCONNEXION=ON`. |

Example for a broader binary:

```powershell
-DJANUS_CUDA_ARCH="75;80;86;89;90"
```

A multi-architecture build is larger and slower to compile, but runs on more
GPUs.

### 6. Output folder

A runnable folder must contain:

```
Janus Galaxy.exe            (or Janus Universe.exe)
SDL2.dll
glew32.dll
render\shaders\*.vert / *.frag
equations\*.cl              (copied from each application source tree)
janusgalaxy.cfg             (or jgravity.cfg, created on a clean exit)
```

The CMake post-build step copies the shaders, DLLs and equation templates.
Configuration files are created by the application on a clean exit.

### 7. Troubleshooting

| Symptom | Cause / action |
|---|---|
| `JANUS_VCPKG_ROOT is not set` | Pass `-DJANUS_VCPKG_ROOT=<vcpkg>/installed/x64-windows`. |
| `No vcpkg SDL2 found` | The path points to the vcpkg root instead of `installed/x64-windows`, or the triplet is not installed. |
| `nvcc` not found | Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin` to `PATH`, or pass the toolkit to CMake. |
| The app exits before showing a window | Run it from the output folder: the shaders are resolved relative to the working directory. |
| `cufft64_12.dll` missing (Universe) | Install the CUDA Toolkit 13.1 or copy the DLL next to the executable. |
| `MSVCP140.dll` missing | Install the Microsoft Visual C++ 2015-2022 Redistributable (x64). |
| `no kernel image is available` | The binary was built for another compute capability. Rebuild with `-DJANUS_CUDA_ARCH=<your GPU SM>`. |
| The Equation Editor says the template was not found | Rebuild to copy `equations\` next to the executable and launch from that directory. |
| The cfg changes are lost | The cfg is written only on a clean exit (`Esc` / close); a forced kill loses them. |

### 8. Tested configuration

The repository's prebuilt archives were produced with:

- Windows 11 Pro for Workstations, x64
- Visual Studio 2022 Build Tools, MSVC 14.44
- CUDA Toolkit 13.1
- CMake 3.18+ (Visual Studio 17 2022 generator)
- vcpkg `x64-windows`: SDL2 2.32.10, GLEW 2.3.1, GLM, Dear ImGui
- `JANUS_CUDA_ARCH=89`
- Janus Universe built with `JANUS_WITH_3DCONNEXION=ON` for the author's machine;
  the public source default is `OFF`.

# Building Janus Galaxy and Janus Universe on Windows

This document describes the configuration used to build and test the two
applications in this repository: Windows 11 x64, Visual Studio 2022 Build Tools,
CUDA Toolkit 13.1, CMake 3.18+ and vcpkg.

The source has no dependency on a particular machine: all external paths are
passed as CMake cache variables.

## 1. Prerequisites

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

## 2. Install the vcpkg dependencies

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

## 3. Build Janus Galaxy

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

## 4. Build Janus Universe

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

### Optional 3Dconnexion SpaceMouse support

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

## 5. CMake options

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

## 6. Output folder

A runnable folder must contain:

```
Janus Galaxy.exe            (or Janus Universe.exe)
SDL2.dll
glew32.dll
render\shaders\*.vert / *.frag
equations\*.cl              (optional in Universe; copy from galaxy/equations)
janusgalaxy.cfg             (or jgravity.cfg, created on a clean exit)
```

The CMake post-build step creates the first three items. Copy `equations\` from
`galaxy/equations` into the Universe runtime folder if you want the Equation
Editor's template preview and **Open Template** to work.

## 7. Troubleshooting

| Symptom | Cause / action |
|---|---|
| `JANUS_VCPKG_ROOT is not set` | Pass `-DJANUS_VCPKG_ROOT=<vcpkg>/installed/x64-windows`. |
| `No vcpkg SDL2 found` | The path points to the vcpkg root instead of `installed/x64-windows`, or the triplet is not installed. |
| `nvcc` not found | Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin` to `PATH`, or pass the toolkit to CMake. |
| The app exits before showing a window | Run it from the output folder: the shaders are resolved relative to the working directory. |
| `cufft64_12.dll` missing (Universe) | Install the CUDA Toolkit 13.1 or copy the DLL next to the executable. |
| `MSVCP140.dll` missing | Install the Microsoft Visual C++ 2015-2022 Redistributable (x64). |
| `no kernel image is available` | The binary was built for another compute capability. Rebuild with `-DJANUS_CUDA_ARCH=<your GPU SM>`. |
| The Equation Editor says the template was not found | Copy `equations\` next to the executable (Universe does not ship it in the original tree). |
| The cfg changes are lost | The cfg is written only on a clean exit (`Esc` / close); a forced kill loses them. |

## 8. Tested configuration

The repository's prebuilt archives were produced with:

- Windows 11 Pro for Workstations, x64
- Visual Studio 2022 Build Tools, MSVC 14.44
- CUDA Toolkit 13.1
- CMake 3.18+ (Visual Studio 17 2022 generator)
- vcpkg `x64-windows`: SDL2 2.32.10, GLEW 2.3.1, GLM, Dear ImGui
- `JANUS_CUDA_ARCH=89`
- Janus Universe built with `JANUS_WITH_3DCONNEXION=ON` for the author's machine;
  the public source default is `OFF`.

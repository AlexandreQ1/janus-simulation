# Contributing

Thank you for considering a contribution. This project is an independent, experimental simulator built
to support research on **Jean-Pierre Petit's Janus cosmological model**, which uses positive and negative
masses. The model has not been validated by the wider scientific community; see
[SCIENTIFIC_STATUS.md](SCIENTIFIC_STATUS.md) and <https://www.jp-petit.org/> for the reference documents.
The most useful contributions right now are **platform ports**, build reports and reproducible
performance measurements.

## Contributions wanted: Linux and macOS

The codebase is currently **Windows x64 only**. It builds with Visual Studio 2022, CUDA 13.1, CMake and
vcpkg, and the CMake files copy Windows `.dll` files next to the executable. The window layer uses SDL2
and OpenGL; the compute layer uses CUDA and, in Janus Universe, cuFFT.

### Linux

A Linux port is the more direct of the two. The main work is:

- replace MSVC-specific compiler and linker flags with portable CMake;
- find SDL2, GLEW, GLM and Dear ImGui through system packages, vcpkg or a submodule;
- copy or install `libSDL2`, `libGLEW` and the shaders with Linux paths (case-sensitive filesystem);
- make the CUDA architecture configurable (`JANUS_CUDA_ARCH`) and test on the target GPU;
- handle the Windows icon resource (`app_icon.rc`) as an optional Windows-only target;
- check file paths, working-directory assumptions and any Windows API calls;
- provide a tested build command and, if possible, a CI job;
- add a packaging recipe (tarball, AppImage or `.deb`) if you can maintain it.

### macOS

A macOS port is a **larger project**, not a rebuild:

- modern macOS has no NVIDIA CUDA support, so the CUDA kernels cannot be compiled as-is;
- the compute layer needs a Metal, OpenCL or CPU backend, with equivalent kernels for the Janus force,
  tree walk, integration and analysis reductions;
- Janus Universe's cuFFT Particle-Mesh solver needs a replacement (for example FFTW, vDSP or a Metal
  FFT), with the same normalization and memory layout;
- OpenGL is deprecated on macOS; SDL2 still works, but a Metal renderer may be preferable;
- Apple Silicon and Intel Macs have different GPU capabilities and no CUDA at all; performance and
  numerical behaviour must be measured, not assumed.

If you want to work on either port, open an issue first so the approach and scope can be discussed.
State the platform, compiler, GPU and what you can test.

## Other useful contributions

- **Build reports** on Windows with different Visual Studio, CUDA or vcpkg versions.
- **Reproducible performance measurements** (GPU model, VRAM, resolution, particle count, display mode,
  frame rate). The user guides explain the measurement conditions.
- **Bug reports** with the exact steps, the configuration file, the console output and, when relevant, a
  screenshot or a short screen recording.
- **Documentation corrections**, especially where a UI label or a measured number is wrong.
- **Numerical validation**: comparisons with analytic limits, standard benchmarks or independent
  implementations. Please state clearly what was compared and what remains uncertain.

## Scientific responsibility

This repository does not establish the scientific validity of the Janus model. Do not present a
simulation result as evidence for or against the model without independent verification and peer
review. If you add a physical feature, document its equations, assumptions, units and limitations in
the technical references and in the pull request.

## Pull requests

- Keep changes focused; do not mix a port with an unrelated refactor.
- Keep the Windows build working. If a change is platform-specific, guard it in CMake and in the code.
- Use C++17 and the existing code style; avoid introducing new dependencies without a reason.
- Update the relevant document (`README.md`, `BUILDING.md`, user guide or technical reference).
- Include the commands you ran and the results (build, launch, type of test).
- Contributions are accepted under the repository's MIT licence.

## Issues and communication

Open an issue for a bug, a port, a build problem or a question. Include the version, operating system,
GPU, driver, CUDA version and the exact configuration. For a port, use the title **“Port to Linux”** or
**“Port to macOS”** so the work can be tracked.

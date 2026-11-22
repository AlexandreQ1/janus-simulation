# Contributing

Thank you for considering a contribution. This project is an independent, experimental simulator built
to support research on **Jean-Pierre Petit's Janus cosmological model**, which uses positive and negative
masses. The model has not been validated by the wider scientific community; see
[SCIENTIFIC_STATUS.md](SCIENTIFIC_STATUS.md) and <https://www.jp-petit.org/> for the reference documents.
The most useful contributions right now are **platform ports**, build reports and reproducible
performance measurements.

## Contributions wanted: Linux validation and macOS port

The previously tested platform is **Windows x64**, using Visual Studio 2022, CUDA 13.1,
CMake and vcpkg. Both Linux CUDA applications now compile on Fedora 44; graphical
startup and GPU simulation still need validation. The window layer uses SDL2 and OpenGL; the
compute layer uses CUDA and, in Janus Universe, cuFFT.

### Linux

The build instructions are in [BUILDING.md](BUILDING.md#linux). CMake now limits
Windows resources and DLL deployment to Windows, discovers Linux dependencies,
links Universe to cuFFT, and copies shaders and equation templates on both
platforms. CMake fetches a pinned Dear ImGui docking release and builds matching
core/backends. Standard CMake FetchContent overrides support offline builds.

The remaining work is:

- report builds on additional Linux/CUDA/compiler configurations;
- test startup, CUDA/OpenGL interoperability and both simulation solvers;
- check file paths, case-sensitive resources and working-directory assumptions;
- validate exports and identify UI features that still require Windows;
- provide a GPU-backed CI job or reproducible build report;
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

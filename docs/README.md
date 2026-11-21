# Documentation

These documents describe **independent research tools** built to support research on **Jean-Pierre
Petit's Janus cosmological model** (positive and negative masses). The model has not been validated by
the wider scientific community; results require independent verification and peer review. Reference
documents: <https://www.jp-petit.org/> and
[`../SCIENTIFIC_STATUS.md`](../SCIENTIFIC_STATUS.md). Contributions for **Linux** and **macOS** are
wanted; see [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

All documents are in English and contain no images.

| Document | Contents |
|---|---|
| `User Guide — Janus Galaxy.pdf` | Launching, navigation, all controls, display modes, analyses, replay, state files, presets, the `cfg`, performance and troubleshooting. |
| `User Guide — Janus Universe.pdf` | The same for the cosmological simulator, plus expansion, PM/tree solvers, batch runs, galaxy detection, P(k) and the formation history. |
| `Janus Galaxy — Technical & Scientific Reference.pdf` | Janus force law, CUDA kernels, memory layout, numerical stability and analysis formulae. |
| `Janus Universe — Technical & Scientific Reference.pdf` | Cosmology, initial conditions, PM-FFT and tree solvers, stability and analysis formulae. |
| `Janus Universe — HPC Scaling Roadmap.pdf` | Work packages for 40 M–10⁹+ particles, from a single large-memory GPU to multi-GPU and supercomputer systems. A roadmap, not the current build. |
| `HPC Scaling Roadmap.md` | Markdown source of the HPC roadmap. |

The guides quote the UI labels exactly as they appear in the applications. They
describe the 21 November 2026 build and explicitly identify the controls that
are present but not wired in that build (for example the Universe adaptive
time-step group).

The HPC roadmap is an engineering plan. Its work packages (block time-stepping,
2LPT, distributed FFT, MPI, subhalo finders, baryonic physics) are **not**
implemented in the 21 November 2026 build. See
[`../SCIENTIFIC_STATUS.md`](../SCIENTIFIC_STATUS.md) for the scientific status
and the reference list.

**Known correction — PM grid memory.** The Universe user guide still quotes
about 3 GB for a 512³ PM grid. The in-app tooltip has been corrected, and the
actual allocation is about **17 GiB plus cuFFT workspace** (three zero-padded
1024²×513 complex arrays, two density grids and two float4 force fields), so
512³ is not usable on a 16 GB GPU. Use **64³–256³** on the reference laptop;
the breakdown is in the HPC roadmap, Section 7.

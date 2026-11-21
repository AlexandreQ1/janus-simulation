<div class="cover">
<p class="cover-kicker">Technical specification &nbsp;·&nbsp; 2026</p>
<p class="cover-title">Janus Universe</p>
<p class="cover-sub">HPC Scaling Roadmap</p>
<div class="cover-rule"></div>
<p class="cover-lede">From workstation to supercomputer — resolving galaxy formation in the Janus cosmic web.</p>
<p class="cover-foot"><span class="edition">For porting teams &amp; HPC labs</span><br>Engineering roadmap — not a report of achieved results.</p>
</div>

> ⚠️ **These programs are experimental and have not been validated by scientists.** This document
> describes an *engineering roadmap* for scaling them up; it does not assert scientific validity of the
> Janus model. See the project README.

**Status of this document.** This is a roadmap, not a report of achieved results. The capabilities in
Sections 5–7 (block time-stepping, 2LPT and zoom-in initial conditions, distributed FFT, MPI,
subhalo finders, baryonic physics, parallel I/O) are **not implemented** in the 21 November 2026
build. They are work packages for a porting team, with the order and risks stated below.

---

## 1. Purpose & audience

This document specifies what must be **added** to the Janus simulators to go from an interactive,
single-GPU visualization tool to an instrument capable of **resolving the formation of galaxies inside
the cosmic web**, at **40 million particles minimum** and scaling to billions on multi-GPU and
supercomputer systems.

It is written for a **porting team, graduate student, or HPC lab** evaluating the code. It assumes
familiarity with collisionless N-body cosmology (PM/Tree/P³M), MPI/CUDA, and the standard structure-
formation pipeline.

The two applications are:

- **Janus Galaxy** — single-object, non-cosmological (static space).
- **Janus Universe** — cosmological (expanding space, comoving coordinates) — **the target of this
  roadmap**.

**Platform status.** The current build is **Windows x64 only**. Contributors are needed for the
**Linux** and **macOS** ports: Linux is the more direct port (CUDA, SDL2 and OpenGL are available);
macOS additionally requires replacing the CUDA/cuFFT compute backend because modern macOS has no NVIDIA
CUDA support. See [`README.md`](../README.md) and [`CONTRIBUTING.md`](../CONTRIBUTING.md).

---

## 2. Scientific objective

Demonstrate, at scientifically meaningful resolution, that in the **Janus bimetric model** (positive
mass + negative mass replacing particulate dark matter):

1. a **cosmic web** of filaments, walls, voids and nodes forms from Zel'dovich initial conditions;
2. at the **nodes**, positive-mass matter collapses into **bound, rotating proto-galaxies**;
3. this assembly happens **early** (consistent with JWST observations at z ≈ 10–14);
4. the late-time expansion **accelerates** without a cosmological constant.

The milestone of this roadmap is item (2): going from "the web appears" to "**individual galaxies are
resolved and tracked as they form within the web**."

---

## 3. Baseline — current architecture

| Component | Current implementation |
|---|---|
| Force (long range) | Particle-Mesh **PM / FFT**, single-GPU cuFFT, 64³–512³ offered (default 128³); 512³ needs ~17 GiB plus cuFFT workspace, so **64³–256³ on a 16 GB GPU** |
| Force (short range) | GPU Barnes-Hut **tree** (two-tree Janus) + P³M correction |
| Integrator | **DKD leapfrog**, Hubble drag, scale factor `a(t)` from Friedmann; `a_inv3 = 1/a³` in the kick |
| Coordinates | comoving, physical = comoving × `a` |
| Initial conditions | **Zel'dovich** (1st order), CPU FFT, grid capped at 512³ |
| Precision | FP32 particle state and forces (host-side scalars in double where needed) |
| Time stepping | **single global fixed Δt**; the adaptive limiter in `Simulation::computeNextDt` is present but not wired to the UI in this build |
| Structure finding | **FoF** halo finder + cosmological virial ratio (`G_eff = G/a³`) |
| Parallelism | **single GPU** |
| Rendering | CUDA→host→VBO copy (interop path available) |
| Measured interactive scale | ~2 M particles total usable for interactive exploration on an RTX 4090 Laptop (16 GB); 2 M at 8.8–16.8 FPS in Janus Galaxy (reference 8.8–11.2; optimized 11.9–16.8); 5 M at 3.7–4.8 FPS |

The physics core (Janus force, comoving integrator, Friedmann background, Zel'dovich ICs, FoF virial) is
**implemented and described in the technical references**. Its scientific validity is **not established
here**: the validation plan in Section 8 remains to be carried out. The limits below are about **scale
and resolution**, not about a claim of physical correctness.

---

## 4. Bottleneck analysis — what breaks first as N grows

1. **Single global time step.** A single fixed Δt must satisfy the densest, fastest-collapsing region;
   that small step is then **applied to the entire box**. This is the dominant wall: it makes high
   resolution *unusable*, independent of GPU power.
2. **Force resolution capped by the PM grid.** Galaxies live far below the PM mesh cell (the build
   offers 64³–512³, but 512³ needs ~17 GiB, so 64³–256³ on 16 GB); without a finer short-range force
   they are smeared out.
3. **Single-GPU memory & the global FFT.** One GPU caps the particle count; the PM FFT is **global**
   (couples the whole box), so distributing it across GPUs/nodes is the hard part of HPC scaling.
4. **FP32 dynamic range.** A ~Mpc box vs ~kpc galaxy strains single precision on positions.
5. **First-order (Zel'dovich) ICs + CPU FFT.** Insufficient for high-resolution runs; the CPU IC FFT
   does not scale.
6. **Rendering 10⁷–10⁹ raw points** per frame is meaningless without level-of-detail / projection.

---

## 5. Required capabilities (work packages)

Ordered by **impact-to-effort**. WP1–WP2 are prerequisites for *any* high-resolution science on a
single GPU; WP3–WP4 are the supercomputer step; WP5 identifies galaxies; WP6 makes them luminous.

### WP1 — Block / individual time-stepping  *(highest priority)*
Per-particle time steps in power-of-two bins (`Δt_i = Δt₀ / 2^k`), with a synchronization scheme and an
active-particle list each sub-step. Dense regions (forming galaxies) take small steps; voids take large
ones. **This is the single most important addition** — it is what makes GADGET-4 / SWIFT efficient.
*Touches:* the integrator (`simulation.cpp`), the tree walk, and the kick/drift kernels.

### WP2 — Resolution upgrade
- **2LPT initial conditions** (second-order Lagrangian perturbation theory) replacing 1st-order
  Zel'dovich; larger IC grids; optional **"zoom-in"** (a refined Lagrangian region inside a coarse box)
  to concentrate particles where galaxies should form. *Touches:* `universe_generator.cpp`.
- **FP64 positions** (forces/velocities can stay FP32, or double-single).
- **Adaptive gravitational softening** and a **finer short-range force** (deep tree / P³M),
  with **Ewald summation** for periodic boundaries (an Ewald variant already exists in the codebase).

### WP3 — Single-node multi-GPU
- **Domain decomposition** via a space-filling (Peano-Hilbert) curve, load-balanced by work.
- **Distributed FFT** for the PM step (cuFFTMp / heFFTe).
- Intra-node exchange of boundary particles & long-range forces via **NCCL**.

### WP4 — Multi-node (HPC)
- **MPI** domain decomposition across nodes; overlap communication with computation.
- Distributed FFT across nodes; hierarchical (node-local then global) reductions.
- Fault tolerance: restart from checkpoints.

### WP5 — Galaxy identification & history
- A **subhalo finder** (SUBFIND-like): galaxies are *bound substructures inside* FoF halos, not the
  halos themselves. The current FoF gives halos; subhalos give galaxies.
- **Merger trees** to follow hierarchical assembly across snapshots.
- **In-situ analysis** (compute catalogs on the fly — full snapshots at 10⁸–10⁹ are too large to keep).

### WP6 — Baryonic physics  *(phase 2 — the real "galaxies")*
Gravity-only runs show **proto-galaxies** (bound, rotating positive-mass concentrations). *Luminous,
realistic* galaxies require gas:
- **hydrodynamics** (modern SPH, moving-mesh, or AMR);
- **radiative cooling**, **star formation**, **feedback** (supernovae, AGN).
In the Janus framework the negative-mass sector plays the dark-matter role; baryons are added on top to
"light up" the galaxies.

### WP7 — I/O, checkpointing, in-situ
- **Parallel HDF5** snapshots; **checkpoint/restart**; compressed/level-of-detail outputs.

### WP8 — Visualization at scale
- Systematic **CUDA-OpenGL interop** (already present, `m_useInterop`);
- **level-of-detail**: render a subsample or a voxelized density/temperature projection rather than 10⁹
  raw points; optional **off-line / headless** rendering for batch runs on compute nodes.

---

## 6. Hardware scaling tiers

| Tier | Hardware | Adds (WPs) | Reachable scale | Result |
|---|---|---|---|---|
| 0 (now) | RTX 4090 Laptop | — | ~2 M usable (5 M at ~4 FPS) | cosmic web (visual) |
| 1 | RTX PRO 6000 (96 GB), 1 GPU | WP1, WP2, WP8 | 40–300 M (gravity) | web + **resolved proto-galaxies** |
| 2 | 1 node, 4–8 GPU | WP3, WP5 | ~1 B (gravity) | galaxy catalog + merger trees |
| 3 | Supercomputer (many nodes) | WP4, WP6, WP7 | 10⁹–10¹⁰ + baryons | **luminous galaxies in the Janus web** |

**Tier 0 is the only tier that exists today.** Its measured envelope is a 16 GB laptop GPU (RTX 4090
Laptop): 2 M particles total run at **8.8–16.8 FPS in Janus Galaxy** (reference build 8.8–11.2;
optimized build 11.9–16.8) and **11–14 FPS in Janus Universe**. A 5 M-particle Galaxy run was measured
at 3.7–4.8 FPS. Above about 2 M particles, a larger desktop or workstation GPU is required for
practical work; the 40 M–10¹⁰ figures in Tiers 1–3 are **projections that depend on the work packages**,
not current measurements.

---

## 7. Memory & performance budget

Gravity-only state is ~**32 B/particle** (pos + vel as float4×2); with tree, predicted positions,
accelerations and block-step bookkeeping, budget **~64–96 B/particle**:

| N particles | State @ ~64 B | + PM grid 512³ (~17 GiB explicit + cuFFT workspace) | Fits on 96 GB? |
|---|---|---|---|
| 40 M | ~2.6 GB | + ~17 GiB | yes, comfortably |
| 300 M | ~19 GB | + ~17 GiB | yes |
| 1 B | ~64 GB | + ~17 GiB | tight (1 GPU) → multi-GPU |

The 512³ figure is **not** the ~3 GB quoted in the earlier documentation: `pm_init` allocates three
zero-padded 1024²×513 complex arrays (~12 GiB), two 512³ float density grids (~1 GiB) and two 512³
float4 force fields (~4 GiB), plus cuFFT workspace. On the 16 GB reference laptop, the practical PM
range is **64³–256³**; 512³ cannot be allocated there.

So a **single RTX PRO 6000** realistically handles a few hundred million particles in gravity-only mode
once WP1–WP2 land. Billions, or **any baryonic physics**, require multi-GPU (WP3+).

---

## 8. Validation plan

Before trusting any science, reproduce standard benchmarks:

- **Matter power spectrum P(k)** growth vs linear theory at large scales.
- **Halo mass function** vs analytic expectations (within the Janus context).
- **Santa Barbara cluster comparison** (collapse of a single cluster) for the gravity solver.
- **Energy / momentum diagnostics** (already present): symplectic energy behaviour; Janus momentum
  non-conservation is physical and must be characterized, not "fixed".
- **Convergence**: same IC at increasing resolution must converge on bulk statistics.
- **Janus-specific**: early-formation timing vs JWST; late-time `q < 0` acceleration; ξ(r) segregation
  of the two species.

---

## 9. Risks & open questions

- **Block time-stepping correctness** with the two-tree Janus force (opposite-sign coupling) needs care
  in the kick synchronization.
- **Distributed FFT** is the principal HPC engineering risk (global communication, load balance).
- **FP32 → FP64** cost/benefit: where exactly double precision is required.
- **Janus periodic boundaries**: Ewald summation for a *two-species, sign-changing* force is not the
  textbook single-species case and must be validated.
- **Baryon coupling to negative mass**: the gas sector's interaction with the negative-mass fluid is a
  modelling decision, not a settled prescription.

---

## 10. Janus-specific considerations

This is **not** ΛCDM with a different label:

- The "dark matter" is a **dynamical negative-mass fluid**, not a static halo — it must be evolved with
  the same solver as the positive matter (the two-tree Janus force already does this).
- The negative-mass sector **accelerates** early collapse (the JWST-early-galaxies prediction) and
  drives **late-time acceleration** — both are emergent and must survive the resolution increase.
- Momentum is **not conserved** (opposite-sign forces are not equal-and-opposite); barycenter drift is
  a physical signature and a validation target, not a bug.

---

## References

- Springel, V. (2005) — *GADGET-2*, MNRAS 364; Springel et al. (2021) — *GADGET-4*, MNRAS 506.
- Schaller et al. (2024) — *SWIFT* (task-based, block time steps).
- Hockney & Eastwood (1988) — *Computer Simulation Using Particles* (PM/P³M).
- Springel & White — *SUBFIND* substructure finder.
- Scoccimarro (1998); Crocce, Pueblas & Scoccimarro (2006) — **2LPT** initial conditions.
- J.-P. Petit — personal site and publication list: <https://www.jp-petit.org/>
- J.-P. Petit & G. d'Agostini — *The Janus Cosmological Model* (2025 overview):
  <https://www.jp-petit.org/papers/cosmo/2025-04-25-The-Janus-Cosmological-Model.pdf>
- J.-P. Petit, F. Margnat & H. Zejli — *A bimetric cosmological model based on Andreï Sakharov's twin
  universe approach*, *Eur. Phys. J. C* **84**, 1226 (2024),
  DOI <https://doi.org/10.1140/epjc/s10052-024-13569-w>; preprint
  <https://arxiv.org/abs/2412.04644> (the arXiv title spells “Andrei”)
- J.-P. Petit & H. Zejli — *Janus Cosmological Model Mathematically & Physically Consistent*
  (HAL preprint, 2024): <https://hal.science/hal-04583560>

---

## Document status

This document is an **engineering roadmap**, not a report of achieved results. The capabilities in
Sections 5–7 are not implemented in the 21 November 2026 build. The application source is released
under the MIT licence; this document is provided for engineering discussion. Corrections, porting
proposals and benchmark results are welcome through the repository issues.

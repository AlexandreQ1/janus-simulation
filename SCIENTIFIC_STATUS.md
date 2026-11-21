# Scientific status and references

## What this project is

**Janus Galaxy** and **Janus Universe** are GPU N-body simulators built to explore and test the
**Janus cosmological model**, an alternative to the standard ΛCDM model proposed by
**Jean-Pierre Petit** and collaborators. In the Janus model the universe contains both positive and
negative masses: masses of the same sign attract, masses of opposite sign repel. The model attempts to
account for dark matter and dark energy through a negative-mass sector and a bimetric geometry.

These applications are **independent research tools**. They are not an official implementation of the
Janus model, and they are not presented as one. The model itself is the work of Jean-Pierre Petit and
his collaborators; the software is a separate engineering effort that implements a subset of the model
for interactive experiment and visualization.

## What the software does not establish

The programs can integrate the Janus force law, produce a cosmic web, follow the expansion history and
detect bound structures. **They do not establish that the Janus model is correct.** A numerical result
from a simulator is not a scientific validation: it must be compared with observations, checked against
analytic limits, reproduced independently and subjected to peer review.

The model has not been validated by the wider scientific community. This repository does not adjudicate
the scientific status of the model; it provides tools for reproducible tests. Independent validation is
essential.

> **Do not cite results from this repository as evidence for or against the Janus model without
> independent validation.** The appropriate use of this software is to generate hypotheses, explore
> parameter space and prepare reproducible tests that domain scientists can assess.

The HPC scaling roadmap in
[docs/Janus Universe — HPC Scaling Roadmap.pdf](docs/Janus%20Universe%20%E2%80%94%20HPC%20Scaling%20Roadmap.pdf)
is an engineering plan. It describes work that has **not** been carried out in the current build; it
is not a claim of achieved scientific capability.

## Performance envelope

- **Development machine:** NVIDIA RTX 4090 Laptop GPU, **16 GB VRAM**, 2 560 × 1 440 window.
- **Prebuilt binaries:** compiled for compute capability **8.9** (RTX 40-series). They are optimized for
  this class of laptop GPU. Other NVIDIA GPUs require a rebuild with a matching `JANUS_CUDA_ARCH`.
- **PM grid:** 512³ needs **~17 GiB plus cuFFT workspace**; use 64³–256³ on a 16 GB GPU.
- **Reference interactive scale:** about **2 million particles total** (1 M positive + 1 M negative),
  measured at **8.8–16.8 FPS** in Janus Galaxy (reference build 8.8–11.2; optimized 11.9–16.8) and
  **11–14 FPS** in Janus Universe. This is usable for interactive exploration, not a smooth 60 FPS
  experience.
- **5 million particles** in Janus Galaxy: about **3.7–4.8 FPS** — it runs, but it is no longer
  comfortable for interaction.
- **Above about 2 million particles:** a larger desktop or workstation GPU with more VRAM and compute is
  required for practical simulation work. The laptop can still run 5 M particles at ~4 FPS, but
  interactivity is lost. The limit is practical, not a hard cap in the code.
- **40 million particles and beyond:** the target of the HPC roadmap — 40–300 M on a single large-memory
  GPU, then multi-GPU / distributed FFT / MPI for ~10⁹ and baryonic runs — not the current build.

These figures are measurements, not guarantees. A desktop GPU, different cooling, a different particle
count or a different display mode will change them. The user guides give the measured reference points
and the recommended settings.

## References

### Jean-Pierre Petit and collaborators

- **Jean-Pierre Petit — official site and publication list**
  <https://www.jp-petit.org/>
- **J.-P. Petit & G. d'Agostini — *The Janus Cosmological Model* (2025 overview)**
  <https://www.jp-petit.org/papers/cosmo/2025-04-25-The-Janus-Cosmological-Model.pdf>
- **J.-P. Petit, F. Margnat & H. Zejli — *A bimetric cosmological model based on Andreï Sakharov's twin
  universe approach***, *Eur. Phys. J. C* **84**, 1226 (2024)
  DOI <https://doi.org/10.1140/epjc/s10052-024-13569-w> · preprint
  <https://arxiv.org/abs/2412.04644> (arXiv spelling: “Andrei”)
- **J.-P. Petit & H. Zejli — *Janus Cosmological Model Mathematically & Physically Consistent***
  (HAL preprint, 2024) <https://hal.science/hal-04583560>
- **J.-P. Petit, G. d'Agostini & N. Debergh — *The nineteen observational confirmations of the Janus
  Cosmological Model*** (author's overview, PDF dated 2020)
  <https://www.jp-petit.org/new_en/Introduction%20to%20the%20Janus%20Cosmological%20Model.pdf>
- **J.-P. Petit — *Le Modèle Cosmologique Janus* [The Janus Cosmological Model]** (French book,
  22 November 2016) <https://www.jp-petit.org/science/Le_Modele_Cosmologique_Janus.pdf>

### Numerical methods used or proposed

- Springel, V. (2005) — GADGET-2, *MNRAS* 364; Springel et al. (2021) — GADGET-4, *MNRAS* 506.
- Schaller et al. (2024) — SWIFT, task-based block time-stepping.
- Hockney & Eastwood (1988) — *Computer Simulation Using Particles* (PM / P³M).
- Springel & White — SUBFIND substructure finder.
- Scoccimarro (1998); Crocce, Pueblas & Scoccimarro (2006) — 2LPT initial conditions.

## Citation and contact

If you use this software in academic work, cite the repository URL
(<https://github.com/AlexandreQ1/janus-simulation>) and the Janus papers above. State clearly that the
software is an independent implementation and that its output has not been validated
by the authors of the model or by an independent group unless that validation is documented.

For corrections, reproducibility reports or collaboration, open an issue on the repository.
